"""Resumable formal runner for the source-locked non-reflection Default-v1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifact_lease import ArtifactLease, ArtifactLeaseError
from .default_v1_source_lock import (
    DEFAULT_PROTOCOL,
    GENERATED_SUITE,
    IDENTITY_FILE,
    DefaultV1SourceLockError,
    create_or_verify_source_lock,
    protocol_run_id_prefix,
    protocol_scenario_id,
    resolve_protocol_execution_configuration,
    validate_protocol,
    verify_source_lock,
)
from .experiment_identity import hash_file, manifest_path_for_trace
from .game_rules import full_game_rules_meta, game_rules_meta
from .localization_integrity import localization_integrity

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_STS2_CLI_ENV = "STS2_CLI_ROOT"
FORMAL_PROVIDER_AUDIT_ENV = "STS2_BENCH_FORMAL_PROVIDER_AUDIT"
PYTHON_ENV_CONTRACT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUTF8": "1",
}
CREDENTIAL_ENV_NAMES = (
    "STS2_BENCH_API_KEY",
    "QWEN_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
)
TRANSPORT_ERROR_MARKERS = (
    "<urlopen error",
    "timed out",
    "timeout",
    "auth_unavailable",
    "no auth available",
    "llm http 408:",
    "llm http 429:",
    "llm http 500:",
    "llm http 502:",
    "llm http 503:",
    "llm http 504:",
    "connection reset",
    "connection aborted",
    "connection refused",
    "remote end closed",
    "network is unreachable",
    "getaddrinfo failed",
    "ssl: unexpected eof",
    "winerror 10054",
    "winerror 10060",
    "winerror 10061",
    "temporary failure",
)
BEHAVIOR_ERROR_MARKERS = (
    "llm failed after",
    "route plan validation failed:",
    "evidence validation failed:",
    "custom sequence validation failed:",
)


class DefaultV1ExperimentError(RuntimeError):
    """Raised when the formal experiment cannot safely continue."""


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _endpoint_requires_api_key(base_url: str) -> bool:
    hostname = (urllib.parse.urlsplit(base_url).hostname or "").strip().lower()
    return hostname not in {"", "localhost", "127.0.0.1", "::1"}


def _load_and_validate_external_environment(
    env_file: str | Path | None,
    *,
    base_url: str,
    forbidden_roots: tuple[Path, ...],
) -> None:
    if env_file is None:
        raise DefaultV1ExperimentError("A formal frozen run requires an explicit external --env-file")
    candidate = Path(env_file).expanduser()
    lexical_path = Path(os.path.abspath(candidate))
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise DefaultV1ExperimentError("The external environment file does not exist") from exc
    if not path.is_file():
        raise DefaultV1ExperimentError("The external environment file does not exist")
    for root in forbidden_roots:
        lexical_root = Path(os.path.abspath(root))
        resolved_root = root.resolve(strict=False)
        if _path_is_within(lexical_path, lexical_root) or _path_is_within(path, resolved_root):
            raise DefaultV1ExperimentError(
                "The external environment file must be outside the formal result root"
            )
    try:
        link_count = path.stat().st_nlink
    except OSError as exc:
        raise DefaultV1ExperimentError("The external environment file does not exist") from exc
    if link_count != 1:
        raise DefaultV1ExperimentError("The external environment file must be a single-link regular file")

    for name in CREDENTIAL_ENV_NAMES:
        os.environ.pop(name, None)
    loaded: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if normalized_key not in CREDENTIAL_ENV_NAMES:
            continue
        normalized_value = value.strip()
        if (
            len(normalized_value) >= 2
            and normalized_value[0] == normalized_value[-1]
            and normalized_value[0] in {"'", '"'}
        ):
            normalized_value = normalized_value[1:-1]
        else:
            normalized_value = normalized_value.split(" #", 1)[0].strip()
        os.environ[normalized_key] = normalized_value
        loaded.add(normalized_key)

    selected = next(
        (
            os.environ[name].strip()
            for name in CREDENTIAL_ENV_NAMES
            if isinstance(os.environ.get(name), str) and os.environ[name].strip()
        ),
        None,
    )
    if _endpoint_requires_api_key(base_url) and (
        not loaded or selected is None or selected.lower() == "local"
    ):
        raise DefaultV1ExperimentError("The configured non-local model endpoint requires an API key")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DefaultV1ExperimentError(f"Expected a JSON object: {path}")
    return value


def _try_read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return _read_json(path), None
    except (OSError, UnicodeError, json.JSONDecodeError, DefaultV1ExperimentError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


_ACCEPTED_ARTIFACT_EXCLUSIONS = {
    "attempt_result.json",
    "report_required.json",
}


def _accepted_artifact_manifest(attempt_dir: Path) -> dict[str, dict[str, Any]]:
    """Hash every immutable child artifact at the moment a sample is accepted."""

    manifest: dict[str, dict[str, Any]] = {}
    if not attempt_dir.is_dir():
        return manifest
    for path in sorted(
        (candidate for candidate in attempt_dir.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(attempt_dir).as_posix(),
    ):
        relative = path.relative_to(attempt_dir).as_posix()
        if relative in _ACCEPTED_ARTIFACT_EXCLUSIONS:
            continue
        manifest[relative] = {
            "sha256": hash_file(path),
            "size": path.stat().st_size,
        }
    return manifest


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve_output_root(protocol: dict[str, Any], override: str | Path | None) -> Path:
    value = Path(override) if override else Path(str(protocol["output_root"]))
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _source_identity(lock_root: Path) -> dict[str, Any]:
    path = lock_root / IDENTITY_FILE
    if not path.is_file():
        raise DefaultV1ExperimentError(f"Source identity is missing: {path}")
    return _read_json(path)


def _verify_expected_identity(lock_root: Path, expected_identity: str) -> dict[str, Any]:
    try:
        verification = verify_source_lock(lock_root)
    except Exception as exc:  # noqa: BLE001 - identity failure must become an experiment failure
        raise DefaultV1ExperimentError(f"Active source verification failed: {exc}") from exc
    if verification.get("identity_sha256") != expected_identity:
        raise DefaultV1ExperimentError("Active source identity changed")
    return verification


def _initialize_output(
    out_root: Path,
    *,
    protocol_path: Path,
    frozen: dict[str, Any],
) -> None:
    copies = {
        "protocol_record.json": protocol_path,
        "generated_suite_record.json": out_root / GENERATED_SUITE,
    }
    for name, source in copies.items():
        destination = out_root / name
        if destination.is_file():
            if hash_file(destination) != hash_file(source):
                raise DefaultV1ExperimentError(f"Formal output identity artifact drifted: {destination}")
            continue
        shutil.copy2(source, destination)
    copied_identity = _read_json(out_root / IDENTITY_FILE)
    if copied_identity.get("identity_sha256") != frozen.get("identity_sha256"):
        raise DefaultV1ExperimentError("Source identity does not match the active run")


def _require_active_runtime_binding(sts2_cli_root: Path) -> None:
    expected_cli = sts2_cli_root.resolve()
    cli_value = os.environ.get(ACTIVE_STS2_CLI_ENV)
    if not cli_value or Path(cli_value).resolve() != expected_cli:
        raise DefaultV1ExperimentError(
            f"{ACTIVE_STS2_CLI_ENV} must resolve to the active sts2-cli worktree"
        )
    expected = (expected_cli / "lib").resolve()
    for name in ("STS2_LIB", "STS2_GAME_DIR"):
        value = os.environ.get(name)
        if not value or Path(value).resolve() != expected:
            raise DefaultV1ExperimentError(f"{name} must resolve to the active sts2-cli lib directory")
    for name, expected_value in PYTHON_ENV_CONTRACT.items():
        if os.environ.get(name) != expected_value:
            raise DefaultV1ExperimentError(f"Sanitized Python environment requires {name}={expected_value}")
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        if os.environ.get(name):
            raise DefaultV1ExperimentError(f"Sanitized Python environment requires {name} to be unset")


def _record_experiment_invocation(
    out_root: Path,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    lock_root: Path,
    frozen: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    path = out_root / "experiment_manifest.json"
    frozen_identity = str(frozen["identity_sha256"])
    protocol_sha256 = hash_file(protocol_path)
    suite_sha256 = hash_file(lock_root / GENERATED_SUITE)
    invocation = {"started_at": _utc_now(), "resume": resume, "pid": os.getpid()}
    if path.is_file():
        manifest = _read_json(path)
        expected = {
            "experiment_id": protocol["experiment_id"],
            "frozen_identity_sha256": frozen_identity,
            "protocol_sha256": protocol_sha256,
            "generated_suite_sha256": suite_sha256,
            "game_rules": protocol["game_rules"],
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise DefaultV1ExperimentError(f"Existing experiment manifest drifted at {key}")
        invocations = manifest.get("invocations") if isinstance(manifest.get("invocations"), list) else []
        manifest["invocations"] = [*invocations, invocation]
        manifest["latest_invocation_started_at"] = invocation["started_at"]
    else:
        manifest = {
            "schema_version": 2,
            "experiment_id": protocol["experiment_id"],
            "frozen_identity_sha256": frozen_identity,
            "frozen_runtime": frozen.get("runtime"),
            "protocol_sha256": protocol_sha256,
            "generated_suite_sha256": suite_sha256,
            "game_rules": protocol["game_rules"],
            "execution_mode": "active_worktree",
            "benchmark_root": str(frozen["benchmark_root"]),
            "sts2_cli_root": str(frozen["sts2_cli_root"]),
            "output_root": str(out_root),
            "initial_started_at": invocation["started_at"],
            "latest_invocation_started_at": invocation["started_at"],
            "invocations": [invocation],
        }
    _atomic_write_json(path, manifest)
    return manifest


def _next_attempt_index(job_dir: Path) -> int:
    indexes = []
    attempts_root = job_dir / "attempts"
    if attempts_root.is_dir():
        for path in attempts_root.glob("attempt_*"):
            try:
                indexes.append(int(path.name.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max(indexes, default=0) + 1


def _attempt_command(
    protocol: dict[str, Any],
    *,
    lane_name: str,
    seed: str,
    replicate: int,
    attempt_dir: Path,
) -> list[str]:
    seed_path = attempt_dir / "seed.json"
    _atomic_write_json(seed_path, [seed])
    return _expected_attempt_command(
        protocol,
        lane_name=lane_name,
        seed=seed,
        replicate=replicate,
        attempt_dir=attempt_dir,
    )


def _expected_attempt_command(
    protocol: dict[str, Any],
    *,
    lane_name: str,
    seed: str,
    replicate: int,
    attempt_dir: Path,
) -> list[str]:
    seed_path = attempt_dir / "seed.json"
    trace_path = attempt_dir / "trace.jsonl"
    analysis_dir = attempt_dir / "trace_analysis"
    replicate_id = f"replicate-{replicate:02d}"
    return [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        "-m",
        "sts2_bench",
        "run",
        *list(protocol["common_args"]),
        "--seeds",
        str(seed_path),
        "--count",
        "1",
        "--scenario-id",
        protocol_scenario_id(protocol, lane_name),
        "--replicate-id",
        replicate_id,
        "--run-id",
        f"{protocol_run_id_prefix(protocol)}--{lane_name}--{replicate_id}",
        "--out",
        str(trace_path),
        "--trace-analysis-dir",
        str(analysis_dir),
    ]


def _nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _validate_resolved_run_config(
    manifest: dict[str, Any],
    *,
    protocol: dict[str, Any],
    command: list[str],
    seed: str,
    attempt_dir: Path,
) -> dict[str, Any]:
    config = manifest.get("run_config") if isinstance(manifest.get("run_config"), dict) else {}
    runtime = resolve_protocol_execution_configuration(protocol)
    expected = {
        "argv": command[7:],
        "agent.type": runtime["agent"],
        "agent.base_url": runtime["base_url"],
        "agent.model": runtime["model"],
        "agent.llm_timeout": runtime["timeout_sec"],
        "agent.llm_max_retries": runtime["retries"],
        "agent.llm_fallback_mode": runtime["fallback"],
        "agent.llm_temperature": runtime["temperature"],
        "agent.prompt_style": runtime["prompt_style"],
        "agent.include_json_state": runtime["include_json_state"],
        "agent.short_term_memory.enabled": runtime["short_term_memory"],
        "agent.short_term_memory.scope": "episode_local_action_history",
        "agent.short_term_memory.window": runtime["short_term_memory_window"],
        "agent.short_term_memory.mode": runtime["short_term_memory_mode"],
        "agent.run_summary_enabled": runtime["run_summary"],
        "agent.context_management.route_candidate_planner.objective": runtime[
            "route_candidate_objective"
        ],
        "agent.context_management.route_candidate_planner.llm_hint": runtime[
            "map_route_hint"
        ],
        "agent.route_plan.enabled": runtime["route_plan"],
        "agent.route_plan.enforcement": runtime["route_plan_enforcement"],
        "agent.route_plan.route_map_prompt_format": runtime["route_map_prompt_format"],
        "agent.rule_projection_and_recent_changes.enabled": runtime[
            "rule_projection_and_recent_changes"
        ],
        "agent.diagnostics.self_consistency.enabled": runtime["self_consistency"],
        "agent.diagnostics.self_consistency.samples": runtime[
            "self_consistency_samples"
        ],
        "agent.diagnostics.self_consistency.decisions": runtime[
            "self_consistency_decisions"
        ],
        "agent.diagnostics.self_consistency.mode": runtime["self_consistency_mode"],
        "agent.diagnostics.action_prompt_order.mode": runtime["action_prompt_order"],
        "agent.diagnostics.action_prompt_order.seed": runtime[
            "action_prompt_order_seed"
        ],
        "game.character": runtime["character"],
        "game.ascension": runtime["ascension"],
        "game.lang": runtime["language"],
        "benchmark.seeds_file": str(attempt_dir / "seed.json"),
        "benchmark.seeds": [seed],
        "benchmark.count": 1,
        "benchmark.setup_actions_file": None,
        "benchmark.setup_action_count": 0,
        "benchmark.max_steps": runtime["max_steps"],
        "benchmark.max_game_actions": runtime["max_game_actions"],
        "benchmark.include_full_map": runtime["include_full_map"],
        "benchmark.allow_repeat_views": runtime["allow_repeat_views"],
        "benchmark.allow_custom_sequences": runtime["allow_custom_sequences"],
        "benchmark.custom_sequence_exposure": runtime["custom_sequence_exposure"],
        "benchmark.combat_card_sequences": runtime["combat_card_sequences"],
        "benchmark.precomputed_combat_card_sequences": runtime[
            "precomputed_combat_card_sequences"
        ],
        "benchmark.combat_reasoning_pool": runtime["combat_reasoning_pool"],
        "benchmark.combat_turn_board": runtime["combat_turn_board"],
        "benchmark.combat_skill_hints.enabled": runtime["combat_skill_hints"],
        "benchmark.combat_skill_hints.profile": runtime[
            "combat_skill_hints_profile"
        ],
        "benchmark.combat_skill_hints.max_hints": runtime[
            "combat_skill_hints_max_hints"
        ],
        "benchmark.combat_skill_lines": runtime["combat_skill_lines"],
        "logging.out": str(attempt_dir / "trace.jsonl"),
        "logging.overwrite": False,
        "logging.log_level": runtime["log_level"],
        "logging.print_prompts": runtime["print_prompts"],
        "logging.print_model_output": runtime["print_model_output"],
        "logging.trace_analysis": runtime["trace_analysis"],
        "logging.trace_analysis_dir": str(attempt_dir / "trace_analysis"),
    }
    if runtime["agent"] == "coordinator":
        expected.update(
            {
                "agent.context_management.mode": "coordinator",
                "agent.context_management.ca.mode": runtime["context_mode"],
                "agent.context_management.occ.mode": runtime["occ_context_mode"],
                "agent.context_management.occ.run_context": "rm_rp_slice",
                "agent.context_management.occ.reward_posterior_evaluator": runtime[
                    "reward_posterior_evaluator"
                ],
            }
        )
        if runtime["occ_context_mode"] == "reference_turn":
            expected.update(
                {
                    "agent.context_management.occ.reference_context.window": runtime[
                        "turn_chat_window"
                    ],
                    "agent.context_management.occ.reference_context.source": (
                        "rm_rp_slice_plus_short_occ_history"
                    ),
                }
            )
        context_mode = runtime["context_mode"]
        if context_mode == "combat_mdp_state":
            expected.update(
                {
                    "agent.context_management.ca.turn_process": (
                        "structured_current_turn_actions"
                    ),
                    "agent.context_management.ca.combat_reasoning_pool": runtime[
                        "combat_reasoning_pool"
                    ],
                    "agent.context_management.ca.combat_turn_board": runtime[
                        "combat_turn_board"
                    ],
                    "agent.context_management.ca.combat_skill_hints.enabled": runtime[
                        "combat_skill_hints"
                    ],
                    "agent.context_management.ca.combat_skill_hints.profile": runtime[
                        "combat_skill_hints_profile"
                    ],
                    "agent.context_management.ca.combat_skill_hints.max_hints": runtime[
                        "combat_skill_hints_max_hints"
                    ],
                }
            )
        elif context_mode == "combat_dialogue":
            expected.update(
                {
                    "agent.context_management.ca.history": "raw_dialogue",
                    "agent.context_management.ca.dialogue_history_turns": runtime[
                        "combat_dialogue_history_turns"
                    ],
                    "agent.context_management.ca.combat_reasoning_pool": runtime[
                        "combat_reasoning_pool"
                    ],
                    "agent.context_management.ca.combat_skill_lines": runtime[
                        "combat_skill_lines"
                    ],
                }
            )
        elif context_mode in {"combat_state_only", "single_turn"}:
            expected.update(
                {
                    "agent.context_management.ca.history": "none",
                    "agent.context_management.ca.combat_reasoning_pool": runtime[
                        "combat_reasoning_pool"
                    ],
                    "agent.context_management.ca.combat_skill_lines": runtime[
                        "combat_skill_lines"
                    ],
                }
            )
    else:
        expected.update(
            {
                "agent.context_management.mode": runtime["context_mode"],
                "agent.context_management.reward_posterior_evaluator": runtime[
                    "reward_posterior_evaluator"
                ],
                "agent.context_management.combat_reasoning_pool.enabled": runtime[
                    "combat_reasoning_pool"
                ],
                "agent.context_management.combat_turn_board.enabled": runtime[
                    "combat_turn_board"
                ],
                "agent.context_management.combat_skill_hints.enabled": runtime[
                    "combat_skill_hints"
                ],
                "agent.context_management.combat_skill_hints.profile": runtime[
                    "combat_skill_hints_profile"
                ],
                "agent.context_management.combat_skill_hints.max_hints": runtime[
                    "combat_skill_hints_max_hints"
                ],
                "agent.context_management.combat_skill_lines.enabled": runtime[
                    "combat_skill_lines"
                ],
            }
        )
    if protocol.get("_protocol_kind") == "generic_ablation_formal":
        architecture = (
            "role_specialized"
            if runtime["agent"] == "coordinator"
            else "formal_generalist"
            if runtime["agent"] == "formal_generalist"
            else "legacy_single_policy"
        )
        history_representation = {
            "combat_state_only": "none",
            "combat_dialogue": "raw_dialogue",
            "combat_mdp_state": "managed_structured",
            "turn_chat": "legacy_turn_chat",
            "single_turn": "legacy_none",
        }[runtime["context_mode"]]
        expected.update(
            {
                "agent.architecture": architecture,
                "combat_context.variant": runtime["context_mode"],
                "combat_context.dialogue_history_turns": runtime[
                    "combat_dialogue_history_turns"
                ],
                "combat_context.history_representation": history_representation,
                "game_rules.delivery": protocol["default_configuration"][
                    "game_rules_delivery"
                ],
            }
        )
    checks: dict[str, bool] = {}
    for dotted, expected_value in expected.items():
        actual = config if dotted == "argv" else _nested_value(config, *dotted.split("."))
        if dotted == "argv":
            actual = config.get("argv") if isinstance(config, dict) else None
        if (
            dotted == "agent.rule_projection_and_recent_changes.enabled"
            and actual is None
            and expected_value is True
            and protocol.get("_protocol_kind") != "generic_ablation_formal"
        ):
            checks[dotted] = True
        else:
            checks[dotted] = actual == expected_value
    return {"valid": bool(checks) and all(checks.values()), "checks": checks, "expected": expected}


def _expected_request_config(protocol: dict[str, Any]) -> dict[str, Any]:
    runtime = resolve_protocol_execution_configuration(protocol)
    return {
        "base_url": runtime["base_url"],
        "model": runtime["model"],
        "temperature": runtime["temperature"],
        "timeout_sec": runtime["timeout_sec"],
        "retries": runtime["retries"],
        "fallback": runtime["fallback"],
        "http_backend": "urllib",
        "response_format": "text_json_contract",
        "provider_completion_contract": "raw_body_sha256_finish_reason_stop_v1",
    }


def _terminate_process_tree(process: subprocess.Popen[Any]) -> bool:
    if process.poll() is not None:
        return True
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
        else:  # pragma: no cover - Windows is the formal experiment platform
            import signal

            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        return process.poll() is not None
    except OSError:
        return False


def _pid_alive(pid: Any) -> bool:
    try:
        numeric = int(pid)
    except (TypeError, ValueError):
        return False
    if numeric <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            numeric,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):  # type: ignore[attr-defined]
                return False
            return int(exit_code.value) == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(numeric, 0)
    except OSError:
        return False
    return True


def _active_attempt(job_dir: Path) -> dict[str, Any] | None:
    lease_path = job_dir / "active_attempt.json"
    lease, error = _try_read_json(lease_path)
    if error:
        raise DefaultV1ExperimentError(f"Corrupted attempt lease requires inspection: {lease_path}: {error}")
    if not lease:
        return None
    if _pid_alive(lease.get("owner_pid")) or _pid_alive(lease.get("child_pid")):
        return lease
    lease["state"] = "orphaned_processes_gone"
    lease["observed_at"] = _utc_now()
    _atomic_write_json(job_dir / "orphaned_attempt.json", lease)
    lease_path.unlink(missing_ok=True)
    return None


def _experiment_hold(out_root: Path) -> dict[str, Any] | None:
    sticky, error = _try_read_json(out_root / "report_required.json")
    if error:
        return {
            "type": "corrupt_experiment_report_required",
            "error": error,
            "created_at": _utc_now(),
        }
    return sticky


def _publish_experiment_hold(
    out_root: Path,
    report: dict[str, Any],
    *,
    control_lock: threading.Lock | None,
) -> dict[str, Any]:
    lock = control_lock or threading.Lock()
    with lock:
        existing = _experiment_hold(out_root)
        if existing is not None:
            return existing
        _atomic_write_json(out_root / "report_required.json", report)
        return report


def _mark_attempt_cleanup_unconfirmed(lease_path: Path, process: subprocess.Popen[Any]) -> None:
    lease, error = _try_read_json(lease_path)
    if error or not lease:
        return
    lease["child_pid"] = process.pid
    lease["state"] = "cleanup_unconfirmed"
    lease["observed_at"] = _utc_now()
    try:
        _atomic_write_json(lease_path, lease)
    except Exception:
        pass


def _release_attempt_lease(lease_path: Path) -> None:
    lease, error = _try_read_json(lease_path)
    if error:
        return
    if not lease:
        return
    if _pid_alive(lease.get("child_pid")):
        lease["state"] = "cleanup_required"
        lease["observed_at"] = _utc_now()
        try:
            _atomic_write_json(lease_path, lease)
        except Exception:
            pass
        return
    lease_path.unlink(missing_ok=True)


def _subprocess_runner(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path,
    timeout_sec: float,
    lease_path: Path,
) -> int:
    if timeout_sec <= 0:
        raise DefaultV1ExperimentError("Every formal attempt requires a positive hard watchdog")
    creation: dict[str, Any] = {}
    if os.name == "nt":
        creation["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:  # pragma: no cover
        creation["start_new_session"] = True
    child_env = os.environ.copy()
    child_env[FORMAL_PROVIDER_AUDIT_ENV] = "1"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=child_env,
            **creation,
        )
        lease, error = _try_read_json(lease_path)
        if error or not lease:
            _terminate_process_tree(process)
            raise DefaultV1ExperimentError(f"Attempt lease disappeared or became corrupt: {lease_path}")
        lease["child_pid"] = process.pid
        lease["state"] = "running"
        try:
            _atomic_write_json(lease_path, lease)
        except Exception:
            if _terminate_process_tree(process) is False:
                _mark_attempt_cleanup_unconfirmed(lease_path, process)
            raise
        try:
            return int(process.wait(timeout=timeout_sec))
        except subprocess.TimeoutExpired:
            if _terminate_process_tree(process) is False:
                _mark_attempt_cleanup_unconfirmed(lease_path, process)
            stderr.write(f"Default-v1 formal attempt timed out after {timeout_sec:g}s\n")
            stderr.flush()
            return 124
        except BaseException:
            if _terminate_process_tree(process) is False:
                _mark_attempt_cleanup_unconfirmed(lease_path, process)
            raise


def _read_trace_jsonl(trace_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not trace_path.is_file():
        return {
            "well_formed": False,
            "rows": rows,
            "errors": [{"line": None, "error": "trace file is missing"}],
        }
    try:
        raw_lines = trace_path.read_bytes().splitlines()
    except OSError as exc:
        return {
            "well_formed": False,
            "rows": rows,
            "errors": [{"line": None, "error": f"trace file could not be read: {exc}"}],
        }
    for line_number, raw_bytes in enumerate(raw_lines, start=1):
        try:
            raw_line = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append({"line": line_number, "error": f"invalid UTF-8: {exc.reason}"})
            continue
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": f"invalid JSON: {exc.msg}"})
            continue
        if not isinstance(value, dict):
            errors.append({"line": line_number, "error": "trace event must be a JSON object"})
            continue
        rows.append(value)
    return {"well_formed": not errors, "rows": rows, "errors": errors}


def _last_trace_event(trace: dict[str, Any]) -> dict[str, Any]:
    rows = trace.get("rows") if isinstance(trace.get("rows"), list) else []
    return rows[-1] if rows and isinstance(rows[-1], dict) else {}


def _canonical_protocol_sha256(protocol: dict[str, Any]) -> str:
    payload = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_request_audit(
    value: Any,
    *,
    expected_index: int,
) -> tuple[list[dict[str, str]] | None, str | None, str | None]:
    if not isinstance(value, dict) or value.get("request_index") != expected_index:
        return None, None, None
    raw_messages = value.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, None, None
    messages: list[dict[str, str]] = []
    preview_parts: list[str] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            return None, None, None
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None, None, None
        normalized = {"role": role, "content": content}
        messages.append(normalized)
        preview_parts.append(f"===== message[{index}] role={role} =====")
        preview_parts.append(content)
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    messages_sha256 = hashlib.sha256(encoded).hexdigest()
    if value.get("messages_sha256") != messages_sha256:
        return None, None, None
    return messages, messages_sha256, "\n".join(preview_parts)


def _provider_response_text(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _provider_finish_reason(response: dict[str, Any]) -> Any:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("finish_reason")


def _provider_metadata(response: dict[str, Any]) -> dict[str, Any]:
    metadata_keys = ("model", "id", "created", "system_fingerprint", "service_tier")
    metadata = {key: response[key] for key in metadata_keys if response.get(key) is not None}
    finish_reason = _provider_finish_reason(response)
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    return metadata


def _provider_request_audit(value: Any) -> dict[str, Any]:
    request = value if isinstance(value, dict) else {}
    status = request.get("provider_response_status")
    raw_body = request.get("provider_response_body")
    preserved = (
        isinstance(raw_body, str)
        and request.get("provider_response_sha256")
        == hashlib.sha256(raw_body.encode("utf-8")).hexdigest()
    )
    response: dict[str, Any] | None = None
    if preserved:
        try:
            candidate = json.loads(raw_body)
            if isinstance(candidate, dict):
                response = candidate
        except (json.JSONDecodeError, UnicodeError):
            pass
    text = _provider_response_text(response) if isinstance(response, dict) else None
    finish_reason = _provider_finish_reason(response) if isinstance(response, dict) else None
    metadata = _provider_metadata(response) if isinstance(response, dict) else {}
    usage = response.get("usage") if isinstance(response, dict) and isinstance(response.get("usage"), dict) else None
    http_status = request.get("provider_http_status")
    http_status_is_int = isinstance(http_status, int) and not isinstance(http_status, bool)
    http_success = http_status_is_int and 200 <= http_status <= 299
    http_nonretryable_client_error = (
        http_status_is_int
        and 400 <= http_status <= 499
        and http_status not in {408, 429}
    )
    http_retryable_error = (
        http_status_is_int
        and (http_status in {408, 429} or 500 <= http_status <= 599)
    )
    latency = request.get("provider_latency_sec")
    latency_valid = (
        isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(float(latency))
        and latency >= 0
    )
    success_checks = {
        "response_body_preserved": preserved,
        "response_is_json_object": isinstance(response, dict),
        "finish_reason_is_stop": finish_reason == "stop",
        "response_text_is_nonempty_text": isinstance(text, str) and bool(text.strip()),
        "provider_http_status_is_2xx": http_success,
        "finish_reason_matches_record": request.get("provider_finish_reason") == finish_reason,
        "metadata_matches_body": request.get("provider_metadata") == metadata,
        "usage_matches_body": request.get("provider_usage") == usage,
    }
    if status == "succeeded":
        valid = all(success_checks.values()) and latency_valid
    elif status == "model_protocol_error":
        protocol_http_status_valid = http_success or http_nonretryable_client_error
        valid = (
            preserved
            and protocol_http_status_valid
            and isinstance(request.get("provider_error"), str)
            and bool(request.get("provider_error"))
            and request.get("provider_finish_reason") == finish_reason
            and request.get("provider_metadata") == metadata
            and request.get("provider_usage") == usage
            and not (
                isinstance(response, dict)
                and finish_reason == "stop"
                and isinstance(text, str)
                and bool(text.strip())
            )
            and latency_valid
        )
    elif status == "transport_error":
        http_body_valid = (
            raw_body is None
            and request.get("provider_response_sha256") is None
            and request.get("provider_http_status") is None
        ) or (
            preserved
            and http_retryable_error
        )
        valid = (
            http_body_valid
            and request.get("provider_metadata") == {}
            and request.get("provider_finish_reason") is None
            and request.get("provider_usage") is None
            and isinstance(request.get("provider_error"), str)
            and bool(request.get("provider_error"))
            and latency_valid
        )
    else:
        valid = False
    return {
        "valid": valid,
        "status": status,
        "text": text,
        "finish_reason": finish_reason,
        "metadata": metadata,
        "usage": usage,
        "latency_sec": float(latency) if latency_valid else None,
        "checks": success_checks,
    }


_RUNTIME_TAKEOVER_BASE_META_KEYS = {
    "agent",
    "active_agent",
    "policy",
    "model_called",
}
_RUNTIME_TAKEOVER_ROUTE_META_KEYS = {
    *_RUNTIME_TAKEOVER_BASE_META_KEYS,
    "route_plan_effect",
}
_RUNTIME_TAKEOVER_OUTCOMES = {
    "no_active_plan",
    "followed_active_plan",
    "invalidated_active_plan",
}


def _runtime_takeover_candidate(
    event: dict[str, Any],
    decision_row: dict[str, Any] | None,
) -> bool:
    meta = event.get("agent_meta") if isinstance(event.get("agent_meta"), dict) else {}
    return (
        isinstance(decision_row, dict)
        and isinstance(decision_row.get("runtime_takeover"), dict)
    ) or meta.get("policy") == "forced_singleton_map"


def _runtime_takeover_audit(
    event: dict[str, Any],
    decision_row: dict[str, Any] | None,
    *,
    route_plan_enabled: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(decision_row, dict):
        return {"valid": False, "errors": ["missing_decision_row"]}

    marker = (
        decision_row.get("runtime_takeover")
        if isinstance(decision_row.get("runtime_takeover"), dict)
        else {}
    )
    meta = event.get("agent_meta") if isinstance(event.get("agent_meta"), dict) else {}
    state = (
        decision_row.get("authoritative_state")
        if isinstance(decision_row.get("authoritative_state"), dict)
        else {}
    )
    choices = state.get("choices") if isinstance(state.get("choices"), list) else []
    legal_actions = (
        decision_row.get("legal_actions")
        if isinstance(decision_row.get("legal_actions"), list)
        else []
    )
    map_actions = [
        value
        for value in legal_actions
        if isinstance(value, dict) and value.get("kind") == "map"
    ]

    if event.get("type") != "action":
        errors.append("runtime_takeover_must_be_action")
    if (
        event.get("step") != decision_row.get("step")
        or event.get("decision") != decision_row.get("decision")
        or event.get("decision") != state.get("decision")
    ):
        errors.append("decision_binding_mismatch")
    if event.get("decision") != "map_select":
        errors.append("runtime_takeover_must_be_map_select")
    if "prompt" in decision_row:
        errors.append("runtime_takeover_must_not_have_prompt")
    if set(marker) != {"mode", "action", "model_called"}:
        errors.append("runtime_takeover_marker_shape")
    if marker.get("mode") != "forced_singleton_map" or marker.get("model_called") is not False:
        errors.append("runtime_takeover_marker_values")
    expected_meta_keys = (
        _RUNTIME_TAKEOVER_ROUTE_META_KEYS
        if route_plan_enabled
        else _RUNTIME_TAKEOVER_BASE_META_KEYS
    )
    if set(meta) != expected_meta_keys:
        errors.append("runtime_takeover_meta_shape")
    if (
        meta.get("agent") != "runtime"
        or meta.get("active_agent") != "runtime"
        or meta.get("policy") != "forced_singleton_map"
        or meta.get("model_called") is not False
    ):
        errors.append("runtime_takeover_meta_values")
    if len(choices) != 1 or not isinstance(choices[0], dict):
        errors.append("runtime_takeover_requires_one_raw_choice")
    if len(map_actions) != 1:
        errors.append("runtime_takeover_requires_one_map_action")

    choice = choices[0] if len(choices) == 1 and isinstance(choices[0], dict) else {}
    map_action = map_actions[0] if len(map_actions) == 1 else {}
    action = event.get("action") if isinstance(event.get("action"), dict) else {}
    command = event.get("command") if isinstance(event.get("command"), dict) else {}
    action_command = action.get("cmd") if isinstance(action.get("cmd"), dict) else {}
    args = action_command.get("args") if isinstance(action_command.get("args"), dict) else {}
    col = choice.get("col")
    row = choice.get("row")
    if not (
        isinstance(col, int)
        and not isinstance(col, bool)
        and isinstance(row, int)
        and not isinstance(row, bool)
    ):
        errors.append("runtime_takeover_choice_coordinates")
    if marker.get("action") != map_action or action != map_action or command != action_command:
        errors.append("runtime_takeover_action_binding")
    if (
        action_command.get("cmd") != "action"
        or action_command.get("action") != "select_map_node"
        or args.get("col") != col
        or args.get("row") != row
    ):
        errors.append("runtime_takeover_choice_binding")

    if route_plan_enabled:
        route_effect = (
            meta.get("route_plan_effect")
            if isinstance(meta.get("route_plan_effect"), dict)
            else {}
        )
        if (
            route_effect.get("mode") != "forced_singleton_map"
            or route_effect.get("selected_node") != choice
            or route_effect.get("outcome") not in _RUNTIME_TAKEOVER_OUTCOMES
        ):
            errors.append("runtime_takeover_route_effect")

    return {"valid": not errors, "errors": errors}


def _trace_rule_exposure(
    trace: dict[str, Any],
    expected_rules: dict[str, Any],
    expected_request: dict[str, Any] | None = None,
    *,
    projection_enabled: bool = True,
    rules_delivery: str | None = None,
    route_plan_enabled: bool = True,
) -> dict[str, Any]:
    model_events: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    runtime_takeovers: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    decisions_by_step: dict[Any, dict[str, Any]] = {}
    rows = trace.get("rows") if isinstance(trace.get("rows"), list) else []
    for value in rows:
        if not isinstance(value, dict):
            continue
        if value.get("type") == "decision":
            decisions_by_step[value.get("step")] = value
        elif value.get("type") in {"action", "rejected_action", "model_error"}:
            decision_row = decisions_by_step.get(value.get("step"))
            destination = (
                runtime_takeovers
                if _runtime_takeover_candidate(value, decision_row)
                else model_events
            )
            destination.append((value, decision_row))

    expected_id = expected_rules.get("ruleset_id")
    expected_sha = expected_rules.get("ruleset_sha256")
    invalid_event_indexes: list[int] = []
    invalid_request_refs: list[dict[str, int]] = []
    invalid_provider_request_refs: list[dict[str, int]] = []
    request_count = 0
    provider_usage_totals: dict[str, Any] = {}
    provider_latency_sec = 0.0
    model_visible_texts: list[str] = []
    runtime_takeover_audits = [
        _runtime_takeover_audit(
            event,
            decision_row,
            route_plan_enabled=route_plan_enabled,
        )
        for event, decision_row in runtime_takeovers
    ]
    invalid_runtime_takeover_indexes = [
        index
        for index, audit in enumerate(runtime_takeover_audits)
        if audit.get("valid") is not True
    ]

    def merge_usage(left: Any, right: Any) -> Any:
        if isinstance(left, bool) or isinstance(right, bool):
            return right
        if isinstance(left, int) and isinstance(right, int):
            return left + right
        if isinstance(left, dict) and isinstance(right, dict):
            merged = dict(left)
            for key, value in right.items():
                merged[key] = merge_usage(merged[key], value) if key in merged else value
            return merged
        return right if left != right else left
    audited_fields = (
        "game_ruleset_id",
        "game_ruleset_sha256",
        "game_rule_delivery",
        "game_rule_module_ids",
        "game_rule_render_sha256",
        "game_rule_expected_block_sha256",
        "game_rule_prompt_sha256",
        "game_rule_occurrences",
        "game_rule_expected_occurrences",
        "game_rule_marker_occurrences",
        "game_rule_expected_marker_occurrences",
        "game_rule_exposure_valid",
    )
    for event_index, (event, decision_row) in enumerate(model_events):
        meta = event.get("agent_meta") if isinstance(event.get("agent_meta"), dict) else {}
        authoritative_state = (
            decision_row.get("authoritative_state")
            if isinstance(decision_row, dict) and isinstance(decision_row.get("authoritative_state"), dict)
            else None
        )
        prompt = decision_row.get("prompt") if isinstance(decision_row, dict) else None
        raw_request_audits = meta.get("model_request_audits")
        request_audits = raw_request_audits if isinstance(raw_request_audits, list) else []
        static_full_rules = rules_delivery == "static_full_system_prompt"
        first_request_preview = None
        if static_full_rules and request_audits:
            _first_messages, _first_sha256, first_request_preview = _model_request_audit(
                request_audits[0],
                expected_index=0,
            )
        independently_recomputed = (
            full_game_rules_meta(first_request_preview)
            if static_full_rules and isinstance(first_request_preview, str)
            else game_rules_meta(
                authoritative_state,
                prompt=prompt,
                projection_enabled=projection_enabled,
            )
            if isinstance(authoritative_state, dict) and isinstance(prompt, str)
            else None
        )
        provider_audits: list[dict[str, Any]] = []
        expected_delivery = rules_delivery or (
            "state_selected_prompt" if projection_enabled else "omitted"
        )
        valid = (
            independently_recomputed is not None
            and event.get("step") == decision_row.get("step")
            and event.get("decision") == decision_row.get("decision") == authoritative_state.get("decision")
            and independently_recomputed.get("game_ruleset_id") == expected_id
            and independently_recomputed.get("game_ruleset_sha256") == expected_sha
            and independently_recomputed.get("game_rule_delivery")
            == expected_delivery
            and independently_recomputed.get("game_rule_exposure_valid") is True
            and all(meta.get(field) == independently_recomputed.get(field) for field in audited_fields)
            and meta.get(
                "rule_projection_and_recent_changes_enabled",
                projection_enabled,
            )
            is projection_enabled
            and meta.get(
                "recent_state_changes_delivery",
                "prompt" if projection_enabled else "omitted",
            )
            == ("prompt" if projection_enabled else "omitted")
            and (
                expected_request is None
                or (
                    meta.get("model") == expected_request.get("model")
                    and meta.get("json_response_format") is False
                )
            )
            and bool(request_audits)
        )
        for request_index, raw_request_audit in enumerate(request_audits):
            request_count += 1
            messages, _messages_sha256, messages_preview = _model_request_audit(
                raw_request_audit,
                expected_index=request_index,
            )
            if isinstance(messages_preview, str):
                model_visible_texts.append(messages_preview)
            request_rule_exposure = (
                full_game_rules_meta(messages_preview)
                if static_full_rules and isinstance(messages_preview, str)
                else game_rules_meta(
                    authoritative_state,
                    prompt=messages_preview,
                    projection_enabled=projection_enabled,
                )
                if isinstance(authoritative_state, dict) and isinstance(messages_preview, str)
                else None
            )
            provider_audit = _provider_request_audit(raw_request_audit)
            provider_audits.append(provider_audit)
            if isinstance(provider_audit.get("usage"), dict):
                provider_usage_totals = merge_usage(provider_usage_totals, provider_audit["usage"])
            if isinstance(provider_audit.get("latency_sec"), (int, float)):
                provider_latency_sec += float(provider_audit["latency_sec"])
            request_valid = (
                messages is not None
                and any(message.get("content") == prompt for message in messages)
                and request_rule_exposure is not None
                and request_rule_exposure.get("game_ruleset_id") == expected_id
                and request_rule_exposure.get("game_ruleset_sha256") == expected_sha
                and request_rule_exposure.get("game_rule_module_ids")
                == independently_recomputed.get("game_rule_module_ids")
                and request_rule_exposure.get("game_rule_exposure_valid") is True
                and provider_audit.get("valid") is True
            )
            if not request_valid:
                valid = False
                invalid_request_refs.append(
                    {"model_event_index": event_index, "request_index": request_index}
                )
                if provider_audit.get("valid") is not True:
                    invalid_provider_request_refs.append(
                        {"model_event_index": event_index, "request_index": request_index}
                    )
        if event.get("type") in {"action", "rejected_action"}:
            final_provider = provider_audits[-1] if provider_audits else {}
            successful_provider_texts = [
                item.get("text")
                for item in provider_audits
                if item.get("status") == "succeeded"
            ]
            previous_raw_responses = meta.get("previous_raw_responses")
            recorded_previous = previous_raw_responses if isinstance(previous_raw_responses, list) else []
            provider_output_bound = (
                final_provider.get("valid") is True
                and final_provider.get("status") == "succeeded"
                and meta.get("raw_response") == final_provider.get("text")
                and recorded_previous == successful_provider_texts[:-1]
                and meta.get("provider_finish_reason") == final_provider.get("finish_reason") == "stop"
                and meta.get("provider_metadata") == final_provider.get("metadata")
                and meta.get("usage") == final_provider.get("usage")
            )
            if not provider_output_bound:
                valid = False
                invalid_provider_request_refs.append(
                    {"model_event_index": event_index, "request_index": max(len(provider_audits) - 1, 0)}
                )
        if not valid:
            invalid_event_indexes.append(event_index)
    localization_audit = localization_integrity(model_visible_texts)
    return {
        "valid": bool(model_events)
        and request_count > 0
        and not invalid_event_indexes
        and not invalid_runtime_takeover_indexes
        and localization_audit["valid"] is True,
        "model_event_count": len(model_events),
        "request_count": request_count,
        "runtime_takeover_count": len(runtime_takeovers),
        "invalid_runtime_takeover_indexes": invalid_runtime_takeover_indexes,
        "runtime_takeover_audits": runtime_takeover_audits,
        "provider_usage_totals": provider_usage_totals,
        "provider_latency_sec": round(provider_latency_sec, 6),
        "invalid_model_event_indexes": invalid_event_indexes,
        "invalid_request_refs": invalid_request_refs,
        "invalid_provider_request_refs": invalid_provider_request_refs,
        "localization_audit": localization_audit,
    }


def _validate_attempt(
    attempt_dir: Path,
    *,
    returncode: int,
    expected_experiment_id: str,
    expected_lane: str,
    expected_seed: str,
    expected_replicate: int,
    expected_protocol_sha256: str,
    expected_game_rules: dict[str, Any],
    expected_job_relative_path: str,
    expected_job_dir: Path,
    expected_frozen_identity: str,
    expected_protocol: dict[str, Any],
    expected_command: list[str],
) -> dict[str, Any]:
    def read_object(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            return _read_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError, DefaultV1ExperimentError):
            return {}

    trace_path = attempt_dir / "trace.jsonl"
    manifest_path = manifest_path_for_trace(trace_path)
    manifest_value, manifest_error = _try_read_json(manifest_path)
    manifest = manifest_value or {}
    manifest_results = manifest.get("results") if isinstance(manifest.get("results"), list) else []
    attempt_manifest_path = attempt_dir / "attempt_manifest.json"
    attempt_manifest = read_object(attempt_manifest_path)
    analysis_index_path = attempt_dir / "trace_analysis" / "index.json"
    analysis_index_value, analysis_index_error = _try_read_json(analysis_index_path)
    analysis_index = analysis_index_value or {}
    analysis_runs = analysis_index.get("runs") if isinstance(analysis_index.get("runs"), list) else []
    manifest_identity = manifest.get("identity") if isinstance(manifest.get("identity"), dict) else {}
    expected_replicate_id = f"replicate-{expected_replicate:02d}"
    expected_scenario = protocol_scenario_id(expected_protocol, expected_lane)
    expected_run_id = (
        f"{protocol_run_id_prefix(expected_protocol)}--{expected_lane}--"
        f"{expected_replicate_id}"
    )
    trace = _read_trace_jsonl(trace_path)
    last_event = _last_trace_event(trace)
    trace_results = [
        row.get("result")
        for row in trace.get("rows") or []
        if row.get("type") == "result" and isinstance(row.get("result"), dict)
    ]
    result = trace_results[0] if len(trace_results) == 1 else {}
    manifest_result_consistent = not manifest or (
        len(manifest_results) == 1
        and isinstance(manifest_results[0], dict)
        and manifest_results[0] == result
    )
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    terminal = extra.get("authoritative_terminal_state")
    terminal_player = terminal.get("player") if isinstance(terminal, dict) and isinstance(terminal.get("player"), dict) else {}
    terminal_context = terminal.get("context") if isinstance(terminal, dict) and isinstance(terminal.get("context"), dict) else {}
    rule_exposure = _trace_rule_exposure(
        trace,
        expected_game_rules,
        _expected_request_config(expected_protocol),
        projection_enabled=bool(
            expected_protocol.get("default_configuration", {}).get(
                "rule_projection_and_recent_changes",
                True,
            )
        ),
        rules_delivery=expected_protocol.get("default_configuration", {}).get(
            "game_rules_delivery"
        ),
        route_plan_enabled=bool(
            expected_protocol.get("default_configuration", {}).get("route_plan", True)
        ),
    )
    stderr_text = ""
    stderr_path = attempt_dir / "stderr.log"
    if stderr_path.is_file():
        try:
            stderr_text = stderr_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            pass
    runtime_localization_audit = localization_integrity((), stderr=stderr_text)
    resolved_config = _validate_resolved_run_config(
        manifest,
        protocol=expected_protocol,
        command=expected_command,
        seed=expected_seed,
        attempt_dir=attempt_dir,
    )
    expected_attempt_binding = {
        "experiment_id": expected_experiment_id,
        "lane": expected_lane,
        "seed": expected_seed,
        "replicate": expected_replicate,
        "replicate_id": expected_replicate_id,
        "protocol_sha256": expected_protocol_sha256,
        "game_rules": expected_game_rules,
        "job_relative_path": expected_job_relative_path,
        "frozen_identity_sha256": expected_frozen_identity,
    }
    binding_checks = {
        "returncode_zero": returncode == 0,
        "attempt_path_matches_job": attempt_dir.parent.parent.resolve() == expected_job_dir.resolve(),
        "attempt_manifest_matches_job": all(
            attempt_manifest.get(key) == value for key, value in expected_attempt_binding.items()
        ),
        "attempt_manifest_command_matches": attempt_manifest.get("command") == expected_command,
        "attempt_manifest_request_config_matches": attempt_manifest.get("request_config")
        == _expected_request_config(expected_protocol),
        "manifest_succeeded": manifest.get("status") == "succeeded",
        "manifest_readable_or_absent": manifest_error is None,
        "exactly_one_result": len(trace_results) == 1,
        "manifest_has_exactly_one_result": len(manifest_results) == 1
        and isinstance(manifest_results[0], dict),
        "manifest_result_consistent": manifest_result_consistent,
        "seed_matches_job": bool(expected_seed) and result.get("seed") == expected_seed,
        "run_identity_matches_job": bool(manifest_identity)
        and manifest_identity.get("experiment_id") == expected_experiment_id
        and manifest_identity.get("scenario_id") == expected_scenario
        and manifest_identity.get("replicate_id") == expected_replicate_id
        and manifest_identity.get("run_id") == expected_run_id,
        "resolved_run_config_matches": resolved_config["valid"],
        "trace_jsonl_well_formed": trace.get("well_formed") is True,
        "trace_terminated": last_event.get("type") == "result",
        "trace_result_matches_manifest": len(manifest_results) == 1
        and isinstance(manifest_results[0], dict)
        and last_event.get("result") == manifest_results[0],
        "trace_analysis_complete": bool(analysis_index)
        and manifest.get("trace_analysis") == analysis_index
        and analysis_index.get("source") == str(trace_path)
        and analysis_index.get("out_dir") == str(attempt_dir / "trace_analysis")
        and analysis_index.get("run_count") == 1
        and len(analysis_runs) == 1
        and isinstance(analysis_runs[0], dict)
        and analysis_runs[0].get("seed") == expected_seed,
        "trace_analysis_readable_or_absent": analysis_index_error is None,
        "ruleset_matches_formal_protocol": bool(expected_game_rules.get("ruleset_id"))
        and bool(expected_game_rules.get("ruleset_sha256"))
        and (
            expected_game_rules.get("delivery")
            or expected_game_rules.get("default_delivery")
        )
        == (
            expected_protocol.get("default_configuration", {}).get(
                "game_rules_delivery"
            )
            or expected_game_rules.get("default_delivery")
        )
        and (
            expected_protocol.get("default_configuration", {}).get(
                "game_rules_delivery"
            )
            or expected_game_rules.get("default_delivery")
        )
        in {"state_selected_prompt", "static_full_system_prompt"},
        "trace_rule_exposure_complete": rule_exposure["valid"],
        "runtime_localization_available": runtime_localization_audit["valid"] is True,
    }
    terminal_checks = {
        "error_free": result.get("error") is None,
        "not_truncated": result.get("truncated") is False,
        "natural_game_over": result.get("decision") == "game_over",
        "authoritative_terminal_state": isinstance(terminal, dict)
        and terminal.get("decision") == "game_over",
        "authoritative_terminal_engine_error_free": isinstance(terminal, dict)
        and terminal.get("engine_error") is not True
        and terminal.get("error") != "engine_error",
        "terminal_context_complete": all(key in terminal_context for key in ("act", "floor", "room_type")),
        "terminal_player_complete": all(
            key in terminal_player
            for key in ("hp", "max_hp", "gold", "deck_size", "deck", "relics", "potions")
        )
        and isinstance(terminal_player.get("deck"), list)
        and isinstance(terminal_player.get("relics"), list)
        and isinstance(terminal_player.get("potions"), list),
        "terminal_deck_size_matches": isinstance(terminal_player.get("deck"), list)
        and terminal_player.get("deck_size") == len(terminal_player["deck"]),
        "result_matches_terminal": isinstance(terminal, dict)
        and result.get("victory") == terminal.get("victory")
        and result.get("act") == terminal_context.get("act")
        and result.get("floor") == terminal_context.get("floor")
        and result.get("room_type") == terminal_context.get("room_type")
        and result.get("hp") == terminal_player.get("hp")
        and result.get("max_hp") == terminal_player.get("max_hp")
        and result.get("gold") == terminal_player.get("gold")
        and result.get("deck_size") == terminal_player.get("deck_size"),
    }
    error_text = str(result.get("error") or manifest.get("error") or "")
    decision = result.get("decision")
    report_required = (
        decision in {"unrecognized_state", "unknown", "event_blocked"}
        or "unsupported decision:" in error_text
        or "refusing implicit proceed" in error_text
    )
    issue = None
    if report_required:
        issue = {
            "type": "unsupported_headless_decision",
            "decision": decision,
            "error": error_text,
            "last_observed_state": extra.get("last_observed_state"),
        }
        _atomic_write_json(attempt_dir / "report_required.json", issue)
    error_lower = error_text.lower()
    transport_failure = any(marker in error_lower for marker in TRANSPORT_ERROR_MARKERS)
    configuration_failure = any(
        marker in error_lower
        for marker in (
            "llm http 400:",
            "llm http 401:",
            "llm http 403:",
            "llm http 404:",
            "invalid api key",
            "invalid token",
        )
    )
    termination_class = str(
        result.get("termination_class")
        or extra.get("termination_class")
        or ""
    )
    explicit_behavior_failure = termination_class in {
        "model_forfeit",
        "forfeit",
        "decision_budget_exhausted",
        "model_protocol_error",
    } or decision in {"model_forfeit", "forfeit"}
    behavior_failure = explicit_behavior_failure or bool(result.get("truncated") is True) or (
        any(marker in error_lower for marker in BEHAVIOR_ERROR_MARKERS)
        and not transport_failure
        and not configuration_failure
    )
    postrun_analysis_failure = (
        bool(result)
        and all(terminal_checks.values())
        and manifest.get("status") == "failed"
        and binding_checks["trace_analysis_complete"] is not True
    )
    process_failure = (
        returncode == 124
        or postrun_analysis_failure
        or not manifest
        or len(trace_results) != 1
        or trace.get("well_formed") is not True
        or (isinstance(terminal, dict) and terminal.get("engine_error") is True)
        or (isinstance(terminal, dict) and terminal.get("error") == "engine_error")
        or any(
            marker in error_lower
            for marker in (
                "eof from sts2headless",
                "invalid json from sts2headless",
                "broken pipe",
                "engine_error",
                "headless process",
            )
        )
    )
    immutable_binding_checks = {
        key: value
        for key, value in binding_checks.items()
        if key not in {"returncode_zero", "manifest_succeeded"}
    }
    binding_valid = bool(immutable_binding_checks) and all(immutable_binding_checks.values())
    prelaunch_binding_valid = all(
        binding_checks[key]
        for key in (
            "attempt_path_matches_job",
            "attempt_manifest_matches_job",
            "attempt_manifest_command_matches",
            "attempt_manifest_request_config_matches",
        )
    )
    authoritative_result_binding = all(
        binding_checks[key]
        for key in (
            "attempt_path_matches_job",
            "attempt_manifest_matches_job",
            "attempt_manifest_command_matches",
            "attempt_manifest_request_config_matches",
            "exactly_one_result",
            "manifest_result_consistent",
            "seed_matches_job",
            "trace_jsonl_well_formed",
            "trace_terminated",
            "ruleset_matches_formal_protocol",
            "trace_rule_exposure_complete",
        )
    ) and (
        not manifest
        or (
            binding_checks["run_identity_matches_job"]
            and binding_checks["resolved_run_config_matches"]
        )
    ) and (
        not analysis_index
        or binding_checks["trace_analysis_complete"]
    ) and manifest_error is None and analysis_index_error is None
    pristine_prelaunch_failure = (
        not attempt_manifest_path.exists()
        and not manifest_path.exists()
        and (not trace_path.exists() or trace_path.stat().st_size == 0)
    )
    natural_valid = all(binding_checks.values()) and all(terminal_checks.values())
    authoritative_natural_result = authoritative_result_binding and all(terminal_checks.values())
    authoritative_behavior_result = authoritative_result_binding and behavior_failure
    if report_required:
        classification = "report_required"
    elif authoritative_natural_result or natural_valid:
        classification = "completed_game"
    elif authoritative_behavior_result or (binding_valid and behavior_failure):
        classification = "behavior_failure"
    elif transport_failure and binding_valid:
        classification = "transport_infrastructure"
    elif process_failure and (prelaunch_binding_valid or pristine_prelaunch_failure):
        classification = "process_infrastructure"
    else:
        classification = "integrity_failure"
    formal_sample = classification == "completed_game"
    postrun_pending = formal_sample and (
        returncode != 0
        or binding_checks["manifest_succeeded"] is not True
        or binding_checks["trace_analysis_complete"] is not True
    )
    if classification == "integrity_failure" and issue is None:
        issue = {
            "type": "formal_attempt_integrity_failure",
            "decision": decision,
            "error": error_text,
            "failed_checks": sorted(
                key
                for key, passed in {**binding_checks, **terminal_checks}.items()
                if passed is not True
            ),
        }
    return {
        "valid": formal_sample,
        "formal_sample": formal_sample,
        "classification": classification,
        "retryable_infrastructure": classification in {"transport_infrastructure", "process_infrastructure"},
        "authoritative_result_binding": authoritative_result_binding,
        "postrun_pending": postrun_pending,
        "needs_attention": postrun_pending,
        "checks": {**binding_checks, **terminal_checks},
        "result": result,
        "manifest": str(manifest_path),
        "attempt_manifest": str(attempt_manifest_path),
        "trace": str(trace_path),
        "trace_analysis": str(attempt_dir / "trace_analysis"),
        "trace_errors": trace.get("errors") or [],
        "rule_exposure": rule_exposure,
        "localization_audit": {
            "model_visible": rule_exposure["localization_audit"],
            "runtime": runtime_localization_audit,
        },
        "resolved_run_config": resolved_config,
        "report_required": report_required,
        "issue": issue,
    }


def _attempt_directories(job_dir: Path) -> list[Path]:
    attempts_root = job_dir / "attempts"
    if not attempts_root.is_dir():
        return []
    return sorted(
        (path for path in attempts_root.glob("attempt_*" ) if path.is_dir()),
        key=lambda path: path.name,
    )


def _behavior_failure_attempt_count(jobs: list[dict[str, Any]]) -> int:
    count = 0
    for job in jobs:
        history = job.get("attempt_history")
        attempts = history if isinstance(history, list) else [job]
        count += sum(
            isinstance(attempt, dict)
            and attempt.get("classification") == "behavior_failure"
            for attempt in attempts
        )
    return count


def _recover_job(
    job_dir: Path,
    expected_identity: str,
    *,
    out_root: Path,
    control_lock: threading.Lock | None,
    expected_experiment_id: str,
    expected_lane: str,
    expected_seed: str,
    expected_replicate: int,
    expected_protocol_sha256: str,
    expected_game_rules: dict[str, Any],
    expected_job_relative_path: str,
    expected_protocol: dict[str, Any],
) -> dict[str, Any]:
    try:
        active = _active_attempt(job_dir)
    except DefaultV1ExperimentError as exc:
        issue = {
            "type": "corrupt_active_attempt_lease",
            "error": str(exc),
            "active_attempt": str(job_dir / "active_attempt.json"),
            "lane": expected_lane,
            "seed": expected_seed,
            "replicate": expected_replicate,
        }
        job_report_path = job_dir / "report_required.json"
        if not job_report_path.exists():
            _atomic_write_json(job_report_path, issue)
        experiment_hold = _publish_experiment_hold(
            out_root,
            issue,
            control_lock=control_lock,
        )
        status = {
            "status": "needs_attention",
            "formal_sample": False,
            "retryable": False,
            "issue": issue,
            "control_error": str(exc),
            "experiment_hold": experiment_hold,
            "attempt_history": [],
            "resume_action": "blocked_corrupt_active_attempt_lease",
        }
        _atomic_write_json(job_dir / "job_status.json", status)
        return status
    if active is not None:
        return {
            "status": "active_attempt_detected",
            "formal_sample": False,
            "retryable": False,
            "active_attempt": active,
            "attempt_history": [],
        }
    sticky_path = job_dir / "report_required.json"
    sticky, sticky_error = _try_read_json(sticky_path)
    if sticky_error:
        return {
            "status": "needs_attention",
            "formal_sample": False,
            "retryable": False,
            "control_error": sticky_error,
            "attempt_history": [],
        }
    histories: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for attempt_dir in _attempt_directories(job_dir):
        try:
            attempt_index = int(attempt_dir.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            attempt_index = -1
        if attempt_index < 1 or attempt_dir.name != f"attempt_{attempt_index:03d}":
            issue = {
                "type": "malformed_attempt_directory",
                "attempt_dir": str(attempt_dir),
                "lane": expected_lane,
                "seed": expected_seed,
                "replicate": expected_replicate,
            }
            histories.append(
                {
                    "attempt": attempt_index,
                    "attempt_dir": str(attempt_dir),
                    "status": "needs_attention",
                    "formal_sample": False,
                    "classification": "integrity_failure",
                    "validation": {
                        "formal_sample": False,
                        "classification": "integrity_failure",
                        "issue": issue,
                    },
                }
            )
            _atomic_write_json(sticky_path, issue)
            sticky = issue
            continue
        attempt_result_path = attempt_dir / "attempt_result.json"
        attempt_result_missing = not attempt_result_path.is_file()
        stored, stored_error = _try_read_json(attempt_result_path)
        stored_attempt_manifest, _attempt_manifest_error = _try_read_json(
            attempt_dir / "attempt_manifest.json"
        )
        child_manifest, _child_error = _try_read_json(manifest_path_for_trace(attempt_dir / "trace.jsonl"))
        returncode = int(stored.get("returncode", 1)) if stored else (0 if (child_manifest or {}).get("status") == "succeeded" else 1)
        command = _expected_attempt_command(
            expected_protocol,
            lane_name=expected_lane,
            seed=expected_seed,
            replicate=expected_replicate,
            attempt_dir=attempt_dir,
        )
        validation = _validate_attempt(
            attempt_dir,
            returncode=returncode,
            expected_experiment_id=expected_experiment_id,
            expected_lane=expected_lane,
            expected_seed=expected_seed,
            expected_replicate=expected_replicate,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_game_rules=expected_game_rules,
            expected_job_relative_path=expected_job_relative_path,
            expected_job_dir=job_dir,
            expected_frozen_identity=expected_identity,
            expected_protocol=expected_protocol,
            expected_command=command,
        )
        current_artifact_manifest = _accepted_artifact_manifest(attempt_dir)
        recorded_artifact_manifest = (
            stored.get("accepted_artifact_manifest")
            if isinstance(stored, dict)
            else None
        )
        accepted_artifact_anchor_matches = (
            stored is None
            or stored_error is not None
            or (
                isinstance(recorded_artifact_manifest, dict)
                and bool(recorded_artifact_manifest)
                and recorded_artifact_manifest == current_artifact_manifest
            )
        )
        if (
            validation.get("formal_sample") is True
            and stored is not None
            and stored_error is None
            and not accepted_artifact_anchor_matches
        ):
            issue = {
                "type": "accepted_artifact_drift",
                "attempt": attempt_index,
                "attempt_dir": str(attempt_dir),
                "recorded_artifact_manifest": recorded_artifact_manifest,
                "current_artifact_manifest": current_artifact_manifest,
            }
            validation.update(
                {
                    "valid": False,
                    "formal_sample": False,
                    "classification": "integrity_failure",
                    "retryable_infrastructure": False,
                    "report_required": True,
                    "issue": issue,
                }
            )
        validation.setdefault("checks", {})[
            "accepted_artifact_anchor_matches"
        ] = accepted_artifact_anchor_matches
        record = {
            **(stored or {}),
            **(stored_attempt_manifest or {}),
            "attempt": attempt_index,
            "attempt_dir": str(attempt_dir),
            "stored_attempt_result_error": stored_error,
            "returncode": returncode,
            "validation": validation,
            "formal_sample": validation.get("formal_sample") is True,
            "classification": validation.get("classification"),
            "postrun_pending": validation.get("postrun_pending") is True,
            "needs_attention": validation.get("postrun_pending") is True,
            "accepted_artifact_anchor_matches": accepted_artifact_anchor_matches,
        }
        if attempt_result_missing or stored_error:
            if stored_error and attempt_result_path.is_file():
                backup_path = attempt_dir / f"attempt_result.corrupt.{time.time_ns()}.json"
                os.replace(attempt_result_path, backup_path)
                record["corrupt_attempt_result_backup"] = str(backup_path)
            recovered_status = (
                "succeeded"
                if validation.get("classification") == "completed_game"
                else "behavior_failure"
                if validation.get("classification") == "behavior_failure"
                else "retryable_infrastructure"
                if validation.get("retryable_infrastructure") is True
                else "needs_attention"
            )
            record.update(
                {
                    "schema_version": 1,
                    "status": recovered_status,
                    "identity_stable": True,
                    "recovered_at": _utc_now(),
                    "recovery_source": "authoritative_child_artifacts",
                    "accepted_artifact_manifest": current_artifact_manifest
                    if validation.get("formal_sample") is True
                    else None,
                }
            )
            _atomic_write_json(attempt_result_path, record)
        histories.append(record)
        if validation.get("report_required"):
            issue = dict(validation.get("issue") or {})
            issue.update({"lane": expected_lane, "seed": expected_seed, "replicate": expected_replicate, "attempt": attempt_index})
            _atomic_write_json(sticky_path, issue)
            sticky = issue
        if validation.get("formal_sample") is True and (not stored or stored.get("identity_stable") is True):
            accepted.append(record)
    if sticky:
        status = {
            "status": "needs_attention",
            "formal_sample": False,
            "retryable": False,
            "issue": sticky,
            "attempt_history": histories,
            "resume_action": "sticky_report_required",
        }
    elif len(accepted) > 1:
        status = {
            "status": "needs_attention",
            "formal_sample": False,
            "retryable": False,
            "issue": {"type": "multiple_formal_samples_for_one_job", "attempts": [item["attempt"] for item in accepted]},
            "attempt_history": histories,
        }
    elif accepted:
        status = {
            **accepted[0],
            "status": "succeeded",
            "formal_sample": True,
            "identity_stable": True,
            "attempt_history": histories,
            "resume_action": "recovered_from_disk",
        }
    else:
        behavior_attempts = [
            item for item in histories if item.get("classification") == "behavior_failure"
        ]
        replaceable_history = bool(histories) and all(
            item.get("classification") == "behavior_failure"
            or (item.get("validation") or {}).get("retryable_infrastructure") is True
            for item in histories
        )
        replacement_required = bool(behavior_attempts) and replaceable_history
        retryable_infrastructure = (
            bool(histories)
            and histories[-1]["validation"].get("retryable_infrastructure") is True
        )
        last_issue = (histories[-1].get("validation") or {}).get("issue") if histories else None
        status = {
            "status": (
                "behavior_failure"
                if replacement_required
                else "retryable_infrastructure"
                if retryable_infrastructure
                else "pending"
                if not histories
                else "needs_attention"
            ),
            "formal_sample": False,
            "retryable": replacement_required or retryable_infrastructure or not histories,
            "replacement_required": replacement_required,
            "issue": last_issue,
            "attempt_history": histories,
        }
    if status.get("status") != "pending":
        _atomic_write_json(job_dir / "job_status.json", status)
    return status


def _run_job(
    protocol: dict[str, Any],
    *,
    benchmark_root: Path,
    lock_root: Path,
    out_root: Path,
    frozen_identity: str,
    identity_lock: threading.Lock,
    lane_name: str,
    seed: str,
    replicate: int,
    runner: Callable[..., int],
    protocol_sha256: str | None = None,
    control_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    job_dir = out_root / "runs" / seed / f"replicate_{replicate:02d}"
    job_relative_path = job_dir.relative_to(out_root).as_posix()
    current_protocol_sha256 = protocol_sha256 or _canonical_protocol_sha256(protocol)
    recovered = _recover_job(
        job_dir,
        frozen_identity,
        out_root=out_root,
        control_lock=control_lock,
        expected_experiment_id=str(protocol["experiment_id"]),
        expected_lane=lane_name,
        expected_seed=seed,
        expected_replicate=replicate,
        expected_protocol_sha256=current_protocol_sha256,
        expected_game_rules=protocol["game_rules"],
        expected_job_relative_path=job_relative_path,
        expected_protocol=protocol,
    )
    if recovered.get("formal_sample") is True:
        return {**recovered, "resume_action": "skipped_completed"}
    if recovered.get("status") == "needs_attention":
        issue = recovered.get("issue") if isinstance(recovered.get("issue"), dict) else {
            "type": "default_v1_job_needs_attention",
            "seed": seed,
            "replicate": replicate,
        }
        hold = _publish_experiment_hold(out_root, issue, control_lock=control_lock)
        return {**recovered, "experiment_hold": hold}
    if recovered.get("status") == "active_attempt_detected":
        return recovered
    if recovered.get("retryable") is not True:
        return recovered

    launch_lock = control_lock or threading.Lock()
    with launch_lock:
        hold = _experiment_hold(out_root)
    if hold is not None:
        return {
            **recovered,
            "status": "needs_attention",
            "formal_sample": False,
            "retryable": False,
            "seed": seed,
            "replicate": replicate,
            "experiment_hold": hold,
            "issue": hold,
            "resume_action": "blocked_experiment_report_required",
        }

    schedule = protocol["schedule"]
    max_attempts = int(schedule.get("max_attempts_per_invocation", 1))
    timeout_sec = float(schedule.get("run_timeout_sec", 0))
    last_status: dict[str, Any] | None = None
    attempt_history = list(recovered.get("attempt_history") or [])
    for _ in range(max_attempts):
        with launch_lock:
            hold = _experiment_hold(out_root)
            if hold is not None:
                return {
                    "status": "needs_attention",
                    "formal_sample": False,
                    "retryable": False,
                    "seed": seed,
                    "replicate": replicate,
                    "experiment_hold": hold,
                    "issue": hold,
                    "attempt_history": attempt_history,
                    "resume_action": "blocked_experiment_report_required",
                }
            attempt_index = _next_attempt_index(job_dir)
            attempt_dir = job_dir / "attempts" / f"attempt_{attempt_index:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            command = _attempt_command(
                protocol,
                lane_name=lane_name,
                seed=seed,
                replicate=replicate,
                attempt_dir=attempt_dir,
            )
            attempt_manifest = {
                "schema_version": 1,
                "experiment_id": protocol["experiment_id"],
                "lane": lane_name,
                "seed": seed,
                "replicate": replicate,
                "replicate_id": f"replicate-{replicate:02d}",
                "attempt": attempt_index,
                "formal_sample_candidate": True,
                "frozen_identity_sha256": frozen_identity,
                "protocol_sha256": current_protocol_sha256,
                "game_rules": protocol["game_rules"],
                "job_relative_path": job_relative_path,
                "command": command,
                "started_at": _utc_now(),
                "request_config": _expected_request_config(protocol),
            }
            _atomic_write_json(attempt_dir / "attempt_manifest.json", attempt_manifest)
            lease_path = job_dir / "active_attempt.json"
            lease = {
                "schema_version": 1,
                "experiment_id": protocol["experiment_id"],
                "lane": lane_name,
                "seed": seed,
                "replicate": replicate,
                "attempt": attempt_index,
                "owner_pid": os.getpid(),
                "child_pid": None,
                "state": "starting",
                "started_at": _utc_now(),
            }
            if _active_attempt(job_dir) is not None:
                raise DefaultV1ExperimentError(f"Refusing duplicate active attempt for {seed} replicate {replicate}")
            _atomic_write_json(lease_path, lease)
        runner_error = None
        try:
            returncode = runner(
                command,
                stdout_path=attempt_dir / "stdout.log",
                stderr_path=attempt_dir / "stderr.log",
                cwd=benchmark_root,
                timeout_sec=timeout_sec,
                lease_path=lease_path,
            )
        except Exception as exc:  # noqa: BLE001 - every attempt must leave an atomic result
            returncode = 125
            runner_error = f"{type(exc).__name__}: {exc}"
            (attempt_dir / "stderr.log").write_text(runner_error + "\n", encoding="utf-8")
        finally:
            _release_attempt_lease(lease_path)
        validation = _validate_attempt(
            attempt_dir,
            returncode=returncode,
            expected_experiment_id=str(protocol["experiment_id"]),
            expected_lane=lane_name,
            expected_seed=seed,
            expected_replicate=replicate,
            expected_protocol_sha256=current_protocol_sha256,
            expected_game_rules=protocol["game_rules"],
            expected_job_relative_path=job_relative_path,
            expected_job_dir=job_dir,
            expected_frozen_identity=frozen_identity,
            expected_protocol=protocol,
            expected_command=command,
        )
        if runner_error:
            validation["runner_error"] = runner_error
            validation["classification"] = "process_infrastructure"
            validation["retryable_infrastructure"] = True
            validation["valid"] = False
            validation["formal_sample"] = False
        if isinstance(validation.get("issue"), dict):
            validation["issue"].update(
                {
                    "lane": lane_name,
                    "seed": seed,
                    "replicate": replicate,
                    "attempt": attempt_index,
                    "attempt_dir": str(attempt_dir),
                }
            )
            _atomic_write_json(attempt_dir / "report_required.json", validation["issue"])
        try:
            with identity_lock:
                _verify_expected_identity(lock_root, frozen_identity)
            identity_stable = True
            identity_error = None
        except DefaultV1ExperimentError as exc:
            identity_stable = False
            identity_error = str(exc)
        formal_sample = bool(validation["formal_sample"] and identity_stable)
        classification = str(validation.get("classification") or "integrity_failure")
        replacement_required = classification == "behavior_failure" and identity_stable
        status_name = (
            "succeeded"
            if formal_sample and classification == "completed_game"
            else "behavior_failure"
            if replacement_required
            else "retryable_infrastructure"
            if validation.get("retryable_infrastructure") and identity_stable
            else "needs_attention"
        )
        last_status = {
            **attempt_manifest,
            "status": status_name,
            "formal_sample": formal_sample,
            "retryable": replacement_required
            or bool(validation.get("retryable_infrastructure") and identity_stable),
            "replacement_required": replacement_required,
            "classification": classification,
            "returncode": returncode,
            "identity_stable": identity_stable,
            "identity_error": identity_error,
            "validation": validation,
            "postrun_pending": validation.get("postrun_pending") is True,
            "needs_attention": validation.get("postrun_pending") is True,
            "finished_at": _utc_now(),
            "resume_action": "executed",
            "accepted_artifact_manifest": _accepted_artifact_manifest(attempt_dir)
            if formal_sample
            else None,
        }
        _atomic_write_json(attempt_dir / "attempt_result.json", last_status)
        _atomic_write_json(job_dir / "job_status.json", last_status)
        attempt_history.append(last_status)
        if validation.get("report_required"):
            _atomic_write_json(job_dir / "report_required.json", validation["issue"])
            hold = _publish_experiment_hold(
                out_root,
                dict(validation["issue"]),
                control_lock=control_lock,
            )
            return {
                **last_status,
                "attempt_history": attempt_history,
                "experiment_hold": hold,
            }
        if not identity_stable:
            issue = {
                "type": "frozen_identity_verification_failed",
                "seed": seed,
                "replicate": replicate,
                "attempt": attempt_index,
                "error": identity_error,
            }
            hold = _publish_experiment_hold(out_root, issue, control_lock=control_lock)
            return {
                **last_status,
                "attempt_history": attempt_history,
                "issue": issue,
                "experiment_hold": hold,
            }
        if formal_sample:
            return {**last_status, "attempt_history": attempt_history}
        if replacement_required:
            continue
        if validation.get("retryable_infrastructure") is not True:
            return {**last_status, "attempt_history": attempt_history}
    assert last_status is not None
    replacement_required = any(
        item.get("classification") == "behavior_failure" for item in attempt_history
    )
    return {
        **last_status,
        "status": "behavior_failure" if replacement_required else last_status["status"],
        "retryable": replacement_required or bool(last_status.get("retryable")),
        "replacement_required": replacement_required,
        "attempt_history": attempt_history,
    }


def _lane_status_path(out_root: Path, seed: str) -> Path:
    return out_root / "lanes" / seed / "lane_status.json"


def _run_seed_lane(
    protocol: dict[str, Any],
    *,
    benchmark_root: Path,
    lock_root: Path,
    out_root: Path,
    frozen_identity: str,
    identity_lock: threading.Lock,
    lane: dict[str, Any],
    runner: Callable[..., int],
    progress: Callable[[], None],
    protocol_sha256: str | None = None,
    control_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    seed = str(lane["seed"])
    lane_name = str(lane["name"])
    repeats = int(protocol["comparison"]["repeats_per_seed"])
    jobs = []
    experiment_hold: dict[str, Any] | None = None
    hold_lock = control_lock or threading.Lock()
    for replicate in range(1, repeats + 1):
        with hold_lock:
            experiment_hold = _experiment_hold(out_root)
        if experiment_hold is not None:
            break
        result = _run_job(
            protocol,
            benchmark_root=benchmark_root,
            lock_root=lock_root,
            out_root=out_root,
            frozen_identity=frozen_identity,
            identity_lock=identity_lock,
            lane_name=lane_name,
            seed=seed,
            replicate=replicate,
            runner=runner,
            protocol_sha256=protocol_sha256,
            control_lock=control_lock,
        )
        jobs.append(result)
        report_required = bool((result.get("validation") or {}).get("report_required"))
        if report_required or result.get("status") == "needs_attention":
            issue = (result.get("validation") or {}).get("issue") or result.get("issue") or {
                "type": "default_v1_job_needs_attention",
                "seed": seed,
                "replicate": replicate,
            }
            experiment_hold = _publish_experiment_hold(
                out_root,
                issue,
                control_lock=control_lock,
            )
        replacement_pending = bool(result.get("replacement_required"))
        lane_status = {
            "status": (
                "needs_attention"
                if report_required or result.get("status") == "needs_attention"
                else "replacement_pending"
                if replacement_pending
                else "running"
            ),
            "lane": lane_name,
            "seed": seed,
            "completed_replicates": sum(item.get("formal_sample") is True for item in jobs),
            "completed_games": sum(item.get("classification") == "completed_game" for item in jobs),
            "behavior_failures": _behavior_failure_attempt_count(jobs),
            "postrun_pending_replicates": sum(item.get("postrun_pending") is True for item in jobs),
            "processed_replicates": len(jobs),
            "expected_replicates": repeats,
            "jobs": jobs,
            "updated_at": _utc_now(),
        }
        _atomic_write_json(_lane_status_path(out_root, seed), lane_status)
        progress()
        if result.get("formal_sample") is not True or report_required or result.get("identity_stable") is False:
            break
    completed = sum(item.get("formal_sample") is True for item in jobs)
    needs_attention = any(
        (item.get("validation") or {}).get("report_required") or item.get("status") == "needs_attention"
        for item in jobs
    ) or experiment_hold is not None
    replacement_pending = any(item.get("replacement_required") is True for item in jobs)
    lane_status = {
        "status": (
            "succeeded"
            if completed == repeats
            else "needs_attention"
            if needs_attention
            else "replacement_pending"
            if replacement_pending
            else "failed"
        ),
        "lane": lane_name,
        "seed": seed,
        "completed_replicates": completed,
        "completed_games": sum(item.get("classification") == "completed_game" for item in jobs),
        "behavior_failures": _behavior_failure_attempt_count(jobs),
        "postrun_pending_replicates": sum(item.get("postrun_pending") is True for item in jobs),
        "processed_replicates": len(jobs),
        "expected_replicates": repeats,
        "jobs": jobs,
        "experiment_hold": experiment_hold,
        "updated_at": _utc_now(),
    }
    _atomic_write_json(_lane_status_path(out_root, seed), lane_status)
    progress()
    return lane_status


def _collect_progress(out_root: Path, lanes: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    rows = []
    for lane in lanes:
        seed = str(lane["seed"])
        path = _lane_status_path(out_root, seed)
        row, row_error = _try_read_json(path)
        rows.append(
            row
            if row is not None
            else {
                "status": "pending",
                "lane": lane["name"],
                "seed": seed,
                "completed_replicates": 0,
                "processed_replicates": 0,
                "expected_replicates": repeats,
                "jobs": [],
                "control_file_error": row_error,
            }
        )
    issues = [
        item["validation"]["issue"]
        for lane in rows
        for item in lane.get("jobs") or []
        if isinstance(item.get("validation"), dict) and item["validation"].get("issue")
    ]
    replacement_pending = any(lane.get("status") == "replacement_pending" for lane in rows)
    return {
        "status": (
            "needs_attention"
            if issues
            else "replacement_pending"
            if replacement_pending
            else "running"
        ),
        "expected_runs": len(lanes) * repeats,
        "completed_runs": sum(int(lane.get("completed_replicates") or 0) for lane in rows),
        "processed_runs": sum(int(lane.get("processed_replicates") or 0) for lane in rows),
        "issues_requiring_report": issues,
        "lanes": rows,
        "updated_at": _utc_now(),
    }


def _revalidate_all_jobs(
    protocol: dict[str, Any],
    *,
    out_root: Path,
    frozen_identity: str,
    protocol_sha256: str,
    control_lock: threading.Lock | None,
) -> list[dict[str, Any]]:
    repeats = int(protocol["comparison"]["repeats_per_seed"])
    lane_rows: list[dict[str, Any]] = []
    for lane in protocol["seed_lanes"]:
        lane_name = str(lane["name"])
        seed = str(lane["seed"])
        jobs = []
        for replicate in range(1, repeats + 1):
            job_dir = out_root / "runs" / seed / f"replicate_{replicate:02d}"
            jobs.append(
                _recover_job(
                    job_dir,
                    frozen_identity,
                    out_root=out_root,
                    control_lock=control_lock,
                    expected_experiment_id=str(protocol["experiment_id"]),
                    expected_lane=lane_name,
                    expected_seed=seed,
                    expected_replicate=replicate,
                    expected_protocol_sha256=protocol_sha256,
                    expected_game_rules=protocol["game_rules"],
                    expected_job_relative_path=job_dir.relative_to(out_root).as_posix(),
                    expected_protocol=protocol,
                )
            )
        completed = sum(job.get("formal_sample") is True for job in jobs)
        needs_attention = any(job.get("status") in {"needs_attention", "active_attempt_detected"} for job in jobs)
        replacement_pending = any(job.get("replacement_required") is True for job in jobs)
        lane_rows.append(
            {
                "status": (
                    "succeeded"
                    if completed == repeats
                    else "needs_attention"
                    if needs_attention
                    else "replacement_pending"
                    if replacement_pending
                    else "failed"
                ),
                "lane": lane_name,
                "seed": seed,
                "completed_replicates": completed,
                "completed_games": sum(job.get("classification") == "completed_game" for job in jobs),
                "behavior_failures": _behavior_failure_attempt_count(jobs),
                "postrun_pending_replicates": sum(job.get("postrun_pending") is True for job in jobs),
                "processed_replicates": sum(bool(job.get("attempt_history")) for job in jobs),
                "expected_replicates": repeats,
                "jobs": jobs,
                "updated_at": _utc_now(),
            }
        )
    return lane_rows


def run_experiment(
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    *,
    resume: bool,
    expected_identity: str | None,
    env_file: str | Path | None,
    out_root_override: str | Path | None = None,
    source_root: str | Path = ROOT,
    sts2_cli_root: str | Path | None = None,
    runner: Callable[..., int] = _subprocess_runner,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    cli_root = Path(
        sts2_cli_root or os.environ.get(ACTIVE_STS2_CLI_ENV) or source.parent / "sts2-cli"
    ).resolve()
    resolved_protocol = Path(protocol_path).resolve()
    protocol = validate_protocol(resolved_protocol, root=source)
    _require_active_runtime_binding(cli_root)
    out_root = _resolve_output_root(protocol, out_root_override)
    _load_and_validate_external_environment(
        env_file,
        base_url=str(protocol["default_configuration"]["base_url"]),
        forbidden_roots=(out_root,),
    )
    if out_root.exists() and any(out_root.iterdir()) and not resume:
        raise DefaultV1ExperimentError(f"Output root already exists; pass --resume: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    with ArtifactLease(out_root / ".experiment.lease", label=f"Default-v1 experiment {protocol['experiment_id']}"):
        create_or_verify_source_lock(
            out_root,
            protocol_path=resolved_protocol,
            source_root=source,
            sts2_cli_root=cli_root,
        )
        frozen = _source_identity(out_root)
        frozen_identity = str(frozen["identity_sha256"])
        if expected_identity and frozen_identity != expected_identity:
            raise DefaultV1ExperimentError(
                f"Source identity {frozen_identity} does not match explicitly pinned identity {expected_identity}"
            )
        verification_before = _verify_expected_identity(out_root, frozen_identity)
        _initialize_output(
            out_root,
            protocol_path=resolved_protocol,
            frozen=frozen,
        )
        experiment_manifest = _record_experiment_invocation(
            out_root,
            protocol=protocol,
            protocol_path=resolved_protocol,
            lock_root=out_root,
            frozen=frozen,
            resume=resume,
        )

        sticky_report, sticky_error = _try_read_json(out_root / "report_required.json")
        if sticky_error or sticky_report:
            blocked = {
                "schema_version": 2,
                "experiment_id": protocol["experiment_id"],
                "status": "needs_attention",
                "frozen_identity_sha256": frozen_identity,
                "protocol_sha256": str(experiment_manifest["protocol_sha256"]),
                "sticky_report_required": sticky_report,
                "control_error": sticky_error,
                "updated_at": _utc_now(),
            }
            _atomic_write_json(out_root / "experiment_status.json", blocked)
            return blocked

        lanes = [dict(lane) for lane in protocol["seed_lanes"]]
        repeats = int(protocol["comparison"]["repeats_per_seed"])
        progress_lock = threading.Lock()
        identity_lock = threading.Lock()
        control_lock = threading.Lock()

        def write_progress() -> None:
            with progress_lock:
                _atomic_write_json(out_root / "experiment_status.json", _collect_progress(out_root, lanes, repeats))

        write_progress()
        lane_results: dict[str, dict[str, Any]] = {}
        execution_errors: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=int(protocol["schedule"]["workers"])) as executor:
            futures = {
                executor.submit(
                    _run_seed_lane,
                    protocol,
                    benchmark_root=source,
                    lock_root=out_root,
                    out_root=out_root,
                    frozen_identity=frozen_identity,
                    identity_lock=identity_lock,
                    lane=lane,
                    runner=runner,
                    progress=write_progress,
                    protocol_sha256=str(experiment_manifest["protocol_sha256"]),
                    control_lock=control_lock,
                ): str(lane["seed"])
                for lane in lanes
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    lane_results[seed] = future.result()
                except Exception as exc:  # noqa: BLE001 - experiment control state must close atomically
                    error = {"seed": seed, "type": type(exc).__name__, "error": str(exc)}
                    execution_errors.append(error)
                    lane_results[seed] = {
                        "status": "needs_attention",
                        "seed": seed,
                        "jobs": [],
                        "control_error": error,
                    }
                write_progress()

        try:
            verification_after = _verify_expected_identity(out_root, frozen_identity)
        except DefaultV1ExperimentError as exc:
            verification_after = {"status": "failed", "error": str(exc)}
            execution_errors.append({"type": "identity_verification_failed", "error": str(exc)})
        ordered_lanes = _revalidate_all_jobs(
            protocol,
            out_root=out_root,
            frozen_identity=frozen_identity,
            protocol_sha256=str(experiment_manifest["protocol_sha256"]),
            control_lock=control_lock,
        )
        completed = sum(int(lane["completed_replicates"]) for lane in ordered_lanes)
        issues = []
        for lane in ordered_lanes:
            for item in lane.get("jobs") or []:
                issue = (item.get("validation") or {}).get("issue") or item.get("issue")
                if isinstance(issue, dict):
                    issues.append(issue)
        sticky_report, sticky_error = _try_read_json(out_root / "report_required.json")
        if sticky_report:
            issues.append(sticky_report)
        if sticky_error:
            issues.append({"type": "corrupt_experiment_report_required", "error": sticky_error})
        expected = len(lanes) * repeats
        issues.extend(execution_errors)
        replacement_pending = any(
            lane.get("status") == "replacement_pending" for lane in ordered_lanes
        )
        all_attempts = [
            attempt
            for lane in ordered_lanes
            for job in lane.get("jobs") or []
            for attempt in job.get("attempt_history") or []
        ]
        final = {
            "schema_version": 2,
            "experiment_id": protocol["experiment_id"],
            "status": (
                "succeeded"
                if completed == expected and not issues
                else "needs_attention"
                if issues
                else "replacement_pending"
                if replacement_pending
                else "failed"
            ),
            "frozen_identity_sha256": frozen_identity,
            "protocol_sha256": str(experiment_manifest["protocol_sha256"]),
            "game_rules": protocol["game_rules"],
            "identity_verified_before": verification_before,
            "identity_verified_after": verification_after,
            "expected_runs": expected,
            "completed_runs": completed,
            "completed_games": sum(int(lane.get("completed_games") or 0) for lane in ordered_lanes),
            "behavior_failures": sum(int(lane.get("behavior_failures") or 0) for lane in ordered_lanes),
            "postrun_pending_runs": sum(
                int(lane.get("postrun_pending_replicates") or 0) for lane in ordered_lanes
            ),
            "workers": int(protocol["schedule"]["workers"]),
            "lane_semantics": "one_strictly_serial_lane_per_seed",
            "resume_unit": "seed_replicate",
            "issues_requiring_report": issues,
            "all_attempts": all_attempts,
            "lanes": ordered_lanes,
            "updated_at": _utc_now(),
        }
        _atomic_write_json(out_root / "experiment_summary.json", final)
        _atomic_write_json(out_root / "experiment_status.json", final)
        return final


def build_plan(
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    *,
    expected_identity: str | None = None,
    out_root_override: str | Path | None = None,
    source_root: str | Path = ROOT,
    sts2_cli_root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    cli_root = Path(
        sts2_cli_root or os.environ.get(ACTIVE_STS2_CLI_ENV) or source.parent / "sts2-cli"
    ).resolve()
    resolved_protocol = Path(protocol_path).resolve()
    protocol = validate_protocol(resolved_protocol, root=source)
    _require_active_runtime_binding(cli_root)
    repeats = int(protocol["comparison"]["repeats_per_seed"])
    out_root = _resolve_output_root(protocol, out_root_override)
    if expected_identity and (out_root / IDENTITY_FILE).is_file():
        _verify_expected_identity(out_root, expected_identity)
    jobs = [
        {
            "lane": lane["name"],
            "seed": lane["seed"],
            "replicate": replicate,
            "replicate_id": f"replicate-{replicate:02d}",
        }
        for lane in protocol["seed_lanes"]
        for replicate in range(1, repeats + 1)
    ]
    return {
        "experiment_id": protocol["experiment_id"],
        "protocol": str(resolved_protocol),
        "execution_mode": "active_worktree",
        "benchmark_root": str(source),
        "sts2_cli_root": str(cli_root),
        "output_root": str(out_root),
        "workers": int(protocol["schedule"]["workers"]),
        "lane_semantics": "one_strictly_serial_lane_per_seed",
        "resume_unit": "seed_replicate",
        "expected_runs": len(jobs),
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the source-locked non-reflection Default-v1 paper experiment.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-identity")
    parser.add_argument("--env-file")
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            result = build_plan(
                args.protocol,
                out_root_override=args.out_root,
                expected_identity=args.expected_identity,
            )
        else:
            result = run_experiment(
                args.protocol,
                resume=bool(args.resume),
                expected_identity=args.expected_identity,
                env_file=args.env_file,
                out_root_override=args.out_root,
            )
    except (DefaultV1ExperimentError, DefaultV1SourceLockError, ArtifactLeaseError) as exc:
        # Preflight failures happen before this process owns the experiment
        # lease.  They must not create or overwrite the shared formal status;
        # the lease-owning runner records in-scope failures itself.
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if args.dry_run or result.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
