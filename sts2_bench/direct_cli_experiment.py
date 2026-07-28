"""Resumable, source-locked batch runner for the direct-CLI paper baseline."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifact_lease import ArtifactLease
from .direct_cli_baseline import (
    OUTPUT_RETRY_MESSAGE,
    PAPER_BEHAVIOR_TERMINATION_CLASSES,
    SYSTEM_PROMPT,
    _merge_provider_usage,
    _provider_response_metadata,
    count_invalid_cli_inputs,
    extract_chat_text,
    terminal_command_from_response,
    terminal_output_format_valid,
)
from .direct_cli_report import (
    DIRECT_CLI_ANALYSIS_FILES,
    audit_benchmark_trace,
    audit_engine_trace,
    build_analysis_artifact_manifest,
    convert_engine_trace,
)
from .experiment_identity import hash_file, hash_json
from .game_rules import game_rules_identity, render_full_game_rules
from .localization_integrity import localization_integrity
from .trace_analysis import analyze_trace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STS2_CLI_ROOT = ROOT.parent / "sts2-cli"
_BENCHMARK_EXECUTION_FILES = (
    "sts2_bench/__init__.py",
    "sts2_bench/actions.py",
    "sts2_bench/artifact_lease.py",
    "sts2_bench/context.py",
    "sts2_bench/direct_cli_baseline.py",
    "sts2_bench/direct_cli_experiment.py",
    "sts2_bench/direct_cli_report.py",
    "sts2_bench/experiment_identity.py",
    "sts2_bench/game_rules.py",
    "sts2_bench/localization_integrity.py",
    "sts2_bench/route_graph.py",
    "sts2_bench/route_plan.py",
    "sts2_bench/seeds.py",
    "sts2_bench/sequence_safety.py",
    "sts2_bench/trace_analysis.py",
    "tools/replay_direct_cli_context.py",
)
_CLI_PYTHON_EXECUTION_FILES = ("game_log.py", "play.py")
_UNTRACKED_EXECUTABLE_SUFFIXES = {
    ".py",
    ".pyc",
    ".pyo",
    ".pyd",
    ".pth",
    ".dll",
    ".exe",
    ".so",
    ".dylib",
    ".zip",
    ".whl",
}
_UNTRACKED_SENSITIVE_NAMES = {".env", "sitecustomize.py", "usercustomize.py"}
_WINDOWS_RESERVED_SEED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_DEPRECATED_DIRECT_CLI_EXPERIMENT_IDS = {
    "AGENT-01-MC",
    "direct-cli-baseline-rules-v1",
    "direct-cli-baseline-v1",
    "direct-cli-baseline-v1-defect",
    "direct-cli-baseline-v1-necrobinder",
    "direct-cli-baseline-v1-regent",
    "direct-cli-baseline-v1-silent",
    "direct-cli-rules-v2-formal-defect-20260723",
    "direct-cli-rules-v2-formal-ironclad-20260723",
    "direct-cli-rules-v2-formal-necrobinder-20260723",
    "direct-cli-rules-v2-formal-regent-20260723",
    "direct-cli-rules-v2-formal-silent-20260723",
    "direct-cli-rules-v2-formal-defect-v3-20260723",
    "direct-cli-rules-v2-formal-ironclad-v3-20260723",
    "direct-cli-rules-v2-formal-necrobinder-v3-20260723",
    "direct-cli-rules-v2-formal-regent-v3-20260723",
    "direct-cli-rules-v2-formal-silent-v3-20260723",
    "direct-cli-rules-v2-formal-defect-v4-20260723",
    "direct-cli-rules-v2-formal-ironclad-v4-20260723",
    "direct-cli-rules-v2-formal-necrobinder-v4-20260723",
    "direct-cli-rules-v2-formal-regent-v4-20260723",
    "direct-cli-rules-v2-formal-silent-v4-20260723",
    "direct-cli-rules-v2-formal-defect-v5-20260723",
    "direct-cli-rules-v2-formal-ironclad-v5-20260723",
    "direct-cli-rules-v2-formal-necrobinder-v5-20260723",
    "direct-cli-rules-v2-formal-regent-v5-20260723",
    "direct-cli-rules-v2-formal-silent-v5-20260723",
    "direct-cli-rules-v2-formal-defect-v6-20260723",
    "direct-cli-rules-v2-formal-ironclad-v6-20260723",
    "direct-cli-rules-v2-formal-necrobinder-v6-20260723",
    "direct-cli-rules-v2-formal-regent-v6-20260723",
    "direct-cli-rules-v2-formal-silent-v6-20260723",
    "direct-cli-rules-v2-formal-defect-v7-20260723",
    "direct-cli-rules-v2-formal-ironclad-v7-20260723",
    "direct-cli-rules-v2-formal-necrobinder-v7-20260723",
    "direct-cli-rules-v2-formal-regent-v7-20260723",
    "direct-cli-rules-v2-formal-silent-v7-20260723",
    "direct-cli-rules-v2-formal-ironclad-v8-20260723",
    "direct-cli-rules-v2-formal-ironclad-v9-cont-1007-20260723",
    "direct-cli-rules-v2-formal-ironclad-v9-cont-2407-20260723",
    "direct-cli-rules-v2-formal-ironclad-v9-cont-3407-20260723",
    "direct-cli-rules-v2-formal-ironclad-v10-target-input-fix-1007-20260723",
}


class DirectCliExperimentError(RuntimeError):
    """Raised when a formal direct-CLI experiment cannot safely continue."""


def _python_runtime_identity() -> dict[str, Any]:
    script = r"""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

def file_sha256(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

executable = Path(sys.executable).resolve()
distributions = sorted({
    (str(distribution.metadata.get("Name") or "").strip().lower(), str(distribution.version))
    for distribution in importlib.metadata.distributions()
    if str(distribution.metadata.get("Name") or "").strip()
})
customizations = []
for name in ("sitecustomize", "usercustomize"):
    module = sys.modules.get(name)
    path_value = getattr(module, "__file__", None) if module is not None else None
    if path_value:
        path = Path(path_value).resolve()
        customizations.append({"module": name, "path": str(path), "sha256": file_sha256(path)})
print(json.dumps({
    "python_implementation": sys.implementation.name,
    "python_version": ".".join(str(part) for part in sys.version_info[:3]),
    "cache_tag": str(sys.implementation.cache_tag or ""),
    "executable": str(executable),
    "executable_sha256": file_sha256(executable),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "installed_distributions": [{"name": name, "version": version} for name, version in distributions],
    "site_customizations": customizations,
}, sort_keys=True))
"""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        environment.pop(name, None)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", "-s", "-X", "utf8", "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise DirectCliExperimentError(
            "Cannot resolve the Python runtime identity: " + completed.stderr.strip()
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DirectCliExperimentError("Python runtime identity is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DirectCliExperimentError("Python runtime identity is not an object")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DirectCliExperimentError(f"Expected a JSON object: {path}")
    return value


def _try_read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return _read_json(path), None
    except (OSError, UnicodeError, json.JSONDecodeError, DirectCliExperimentError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * (attempt + 1))


def _read_progress_json(path: Path) -> dict[str, Any]:
    """Read a lane snapshot across Windows atomic replacements.

    Formal artifacts use strict reads. Progress files are diagnostic snapshots
    concurrently replaced by seed-lane threads, so only transient Windows file
    sharing failures are retried here.
    """

    for attempt in range(6):
        try:
            return _read_json(path)
        except (FileNotFoundError, PermissionError):
            if attempt == 5:
                raise
            time.sleep(0.01 * (attempt + 1))
    raise AssertionError("unreachable")


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = _read_json(protocol_path)
    experiment_id = protocol.get("experiment_id")
    if experiment_id in _DEPRECATED_DIRECT_CLI_EXPERIMENT_IDS or protocol.get("schema_version") != 2:
        raise DirectCliExperimentError(
            "This Direct CLI configuration is deprecated/exploratory provenance and cannot enter the formal runner"
        )
    required = {
        "experiment_id",
        "output_root",
        "model",
        "base_url",
        "seeds",
        "repeats_per_seed",
        "workers",
        "prompt_contract_required",
    }
    missing = sorted(required - protocol.keys())
    if missing:
        raise DirectCliExperimentError(f"Protocol is missing required fields: {', '.join(missing)}")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, str) and seed for seed in seeds):
        raise DirectCliExperimentError("Protocol seeds must be a non-empty list of strings")
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", seed) for seed in seeds):
        raise DirectCliExperimentError("Protocol seeds must be safe single path components")
    if any(seed.upper() in _WINDOWS_RESERVED_SEED_NAMES for seed in seeds):
        raise DirectCliExperimentError("Protocol seeds must not use Windows reserved device names")
    normalized_seeds = [os.path.normcase(seed).casefold() for seed in seeds]
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise DirectCliExperimentError("Protocol seeds must be unique under Windows path normalization")
    repeats = int(protocol.get("repeats_per_seed") or 0)
    workers = int(protocol.get("workers") or 0)
    if repeats < 1:
        raise DirectCliExperimentError("repeats_per_seed must be at least 1")
    if workers != len(seeds):
        raise DirectCliExperimentError("workers must equal the number of seeds for one serial lane per seed")
    if int(protocol.get("max_attempts_per_invocation", 1)) < 1:
        raise DirectCliExperimentError("max_attempts_per_invocation must be at least 1")
    if protocol.get("prompt_contract_required") is not True:
        raise DirectCliExperimentError(
            "Formal Direct CLI protocol must require a prompt contract for every model-visible terminal decision"
        )
    request_retries = protocol.get("request_retries")
    if isinstance(request_retries, bool) or not isinstance(request_retries, int) or request_retries < 0:
        raise DirectCliExperimentError("request_retries must be a non-negative integer")
    if str(protocol.get("model", "")).strip().lower() == "deepseek-v4-flash":
        if "max_tokens" not in protocol or protocol.get("max_tokens") is not None:
            raise DirectCliExperimentError(
                "deepseek-v4-flash formal Direct CLI runs must omit max_tokens from the provider payload"
            )
        if request_retries != 1:
            raise DirectCliExperimentError(
                "deepseek-v4-flash formal Direct CLI runs require request_retries=1 to match Default Agent"
            )
        if protocol.get("retry_correction_scope") != "interface_specific_same_state":
            raise DirectCliExperimentError(
                "Formal Direct CLI protocol must declare its interface-specific same-state retry correction"
            )
        if protocol.get("cost_accounting") != "aggregate_all_provider_attempts":
            raise DirectCliExperimentError(
                "Formal Direct CLI protocol must aggregate every provider attempt for cost accounting"
            )
        _validate_preflight_evidence(protocol)
    expected_rules = {
        **game_rules_identity(),
        "delivery": "static_full_system_prompt",
        "source": "sts2_bench/game_rules.py",
    }
    if protocol.get("game_rules") != expected_rules:
        raise DirectCliExperimentError("Protocol game_rules identity or delivery contract does not match the implementation")
    return protocol


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _preflight_manifest_path(protocol: dict[str, Any]) -> Path:
    evidence = protocol.get("preflight_evidence")
    relative = evidence.get("manifest_path") if isinstance(evidence, dict) else None
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DirectCliExperimentError("Formal Direct CLI protocol requires a relative preflight manifest_path")
    manifest_path = (ROOT / relative).resolve()
    if not _path_within(manifest_path, ROOT.resolve()):
        raise DirectCliExperimentError("Preflight manifest_path escapes the benchmark root")
    return manifest_path


def _preflight_selection_plan_path(protocol: dict[str, Any]) -> Path:
    evidence = protocol.get("preflight_evidence")
    relative = evidence.get("selection_plan_path") if isinstance(evidence, dict) else None
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DirectCliExperimentError("Formal Direct CLI protocol requires a relative selection_plan_path")
    plan_path = (ROOT / relative).resolve()
    if not _path_within(plan_path, ROOT.resolve()):
        raise DirectCliExperimentError("Preflight selection_plan_path escapes the benchmark root")
    return plan_path


def _validate_preflight_evidence(protocol: dict[str, Any]) -> dict[str, Any]:
    evidence = protocol.get("preflight_evidence")
    if not isinstance(evidence, dict):
        raise DirectCliExperimentError("Formal Direct CLI protocol requires preflight_evidence")
    manifest_path = _preflight_manifest_path(protocol)
    if not manifest_path.is_file():
        raise DirectCliExperimentError(f"Preflight manifest is missing: {manifest_path}")
    expected_manifest_sha = evidence.get("manifest_sha256")
    if not isinstance(expected_manifest_sha, str) or hash_file(manifest_path) != expected_manifest_sha:
        raise DirectCliExperimentError("Preflight manifest hash does not match the formal protocol")
    manifest = _read_json(manifest_path)
    plan_path = _preflight_selection_plan_path(protocol)
    expected_plan_sha = evidence.get("selection_plan_sha256")
    if (
        not plan_path.is_file()
        or not isinstance(expected_plan_sha, str)
        or hash_file(plan_path) != expected_plan_sha
    ):
        raise DirectCliExperimentError("Preflight selection plan hash does not match the formal protocol")
    plan = _read_json(plan_path)
    plan_sources = plan.get("sources")
    plan_contexts = plan.get("contexts")
    plan_call_policy = plan.get("call_policy")
    plan_output_root = plan.get("output_root")
    if (
        plan.get("schema_version") != 1
        or plan.get("status") != "locked_before_provider_calls"
        or plan.get("model") != protocol.get("model")
        or plan.get("base_url") != protocol.get("base_url")
        or plan.get("preflight_output_limit_control") != "provider_default_omitted"
        or plan.get("paper_sample") is not False
        or not isinstance(plan_output_root, str)
        or not plan_output_root
        or Path(plan_output_root).is_absolute()
        or not isinstance(plan_call_policy, dict)
        or plan_call_policy.get("max_provider_calls_per_context") != 1
        or plan_call_policy.get("request_retries") != 0
        or plan_call_policy.get("failure_disposition") != "preserve_and_invalidate_plan"
        or not isinstance(plan_sources, list)
        or not isinstance(plan_contexts, list)
    ):
        raise DirectCliExperimentError("Preflight selection plan contract is invalid")
    resolved_plan_output_root = (ROOT / plan_output_root).resolve()
    if (
        not _path_within(resolved_plan_output_root, ROOT.resolve())
        or resolved_plan_output_root != manifest_path.parent.resolve()
    ):
        raise DirectCliExperimentError("Preflight manifest is not in the plan-locked output root")
    try:
        plan_locked_at = datetime.fromisoformat(str(plan.get("locked_at")).replace("Z", "+00:00"))
        manifest_created_at = datetime.fromisoformat(str(manifest.get("created_at")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise DirectCliExperimentError("Preflight plan or manifest timestamp is invalid") from exc
    if (
        plan_locked_at.tzinfo is None
        or manifest_created_at.tzinfo is None
        or plan_locked_at >= manifest_created_at
    ):
        raise DirectCliExperimentError("Preflight selection plan was not locked before provider calls")
    source_by_id = {
        source.get("source_id"): source
        for source in plan_sources
        if isinstance(source, dict) and isinstance(source.get("source_id"), str)
    }
    context_by_name = {
        context.get("name"): context
        for context in plan_contexts
        if isinstance(context, dict) and isinstance(context.get("name"), str)
    }
    if len(source_by_id) != len(plan_sources) or len(context_by_name) != len(plan_contexts):
        raise DirectCliExperimentError("Preflight selection plan contains duplicate or invalid entries")
    if {
        context.get("source_id") for context in plan_contexts if isinstance(context, dict)
    } != set(source_by_id):
        raise DirectCliExperimentError("Preflight selection plan has unused or missing sources")
    for source in plan_sources:
        relative_source = source.get("artifacts_path") if isinstance(source, dict) else None
        if (
            not isinstance(relative_source, str)
            or not relative_source
            or Path(relative_source).is_absolute()
            or not _path_within((ROOT / relative_source).resolve(), ROOT.resolve())
        ):
            raise DirectCliExperimentError("Preflight selection source path escapes the benchmark root")
    records = manifest.get("artifacts")
    required_count = evidence.get("required_artifact_count")
    required_failure_names = evidence.get("required_historical_failure_contexts")
    required_disposition = evidence.get("required_source_protocol_disposition")
    historical_failure_max_tokens = evidence.get("historical_failure_max_tokens")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "passed"
        or manifest.get("model") != protocol.get("model")
        or manifest.get("base_url") != protocol.get("base_url")
        or manifest.get("output_limit_control") != "provider_default_omitted"
        or manifest.get("selection_plan_sha256") != expected_plan_sha
        or not isinstance(records, list)
        or isinstance(required_count, bool)
        or not isinstance(required_count, int)
        or not isinstance(required_failure_names, list)
        or not all(isinstance(name, str) and name for name in required_failure_names)
        or isinstance(historical_failure_max_tokens, bool)
        or not isinstance(historical_failure_max_tokens, int)
        or historical_failure_max_tokens < 1
    ):
        raise DirectCliExperimentError("Preflight manifest contract is invalid")
    names = [record.get("name") for record in records if isinstance(record, dict)]
    if len(names) != required_count or len(names) != len(records) or len(set(names)) != len(names):
        raise DirectCliExperimentError("Preflight manifest does not contain the required number of unique contexts")
    if set(names) != set(context_by_name) or len(plan_contexts) != required_count:
        raise DirectCliExperimentError("Preflight manifest does not match the locked selection plan")
    expected_json_names = {"manifest.json", *(f"{name}.json" for name in names)}
    actual_json_names = {path.name for path in manifest_path.parent.glob("*.json")}
    if actual_json_names != expected_json_names:
        raise DirectCliExperimentError("Preflight output root contains unmanifested or missing provider calls")
    character_counts: dict[str, int] = {}
    selection_counts: dict[tuple[str, str], int] = {}
    selection_decision_indexes: dict[tuple[str, str], int] = {}
    historical_failure_names: list[str] = []
    manifest_root = manifest_path.parent.resolve()
    for record in records:
        relative = record.get("path")
        character = record.get("character")
        selection = record.get("selection")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise DirectCliExperimentError("Preflight artifact path is invalid")
        artifact_path = (manifest_root / relative).resolve()
        if not _path_within(artifact_path, manifest_root) or not artifact_path.is_file():
            raise DirectCliExperimentError("Preflight artifact is missing or escapes its manifest root")
        if record.get("sha256") != hash_file(artifact_path):
            raise DirectCliExperimentError("Preflight artifact hash does not match its manifest")
        result = _read_json(artifact_path)
        provenance = result.get("source_provenance")
        attempts = result.get("provider_attempts")
        plan_context = context_by_name.get(record.get("name"))
        plan_source = (
            source_by_id.get(plan_context.get("source_id"))
            if isinstance(plan_context, dict)
            else None
        )
        selection_binding = result.get("selection_plan_binding")
        plan_binding_valid = (
            isinstance(plan_context, dict)
            and isinstance(plan_source, dict)
            and isinstance(selection_binding, dict)
            and selection_binding.get("selection_plan_id") == plan.get("selection_plan_id")
            and selection_binding.get("selection_plan_path")
            == plan_path.relative_to(ROOT.resolve()).as_posix()
            and selection_binding.get("selection_plan_sha256") == expected_plan_sha
            and selection_binding.get("selection_name") == record.get("name")
            and selection_binding.get("output_path")
            == f"{plan_output_root.rstrip('/')}/{record.get('name')}.json"
            and selection_binding.get("selection") == selection
            and selection_binding.get("source_id") == plan_source.get("source_id")
            and selection_binding.get("source_artifacts_path") == plan_source.get("artifacts_path")
            and selection_binding.get("source_terminal_decision_count")
            == plan_source.get("terminal_decision_count")
            and selection_binding.get("source_max_tokens") == plan_source.get("source_max_tokens")
            and plan_context.get("character") == character
            and plan_context.get("selection") == selection
            and plan_context.get("decision_index") == result.get("decision_index")
            and result.get("source_config_sha256") == plan_source.get("config_sha256")
            and result.get("source_events_sha256") == plan_source.get("events_sha256")
            and result.get("source_terminal_sha256") == plan_source.get("terminal_sha256")
            and result.get("recorded_max_tokens") == plan_source.get("source_max_tokens")
        )
        preflight_request = result.get("preflight_provider_request")
        request_messages = (
            preflight_request.get("messages")
            if isinstance(preflight_request, dict)
            else None
        )
        request_contract_valid = (
            isinstance(preflight_request, dict)
            and set(preflight_request) == {"model", "messages", "temperature"}
            and preflight_request.get("model") == protocol.get("model")
            and preflight_request.get("temperature") == protocol.get("temperature")
            and isinstance(request_messages, list)
            and all(
                isinstance(message, dict)
                and set(message) == {"role", "content"}
                and isinstance(message.get("role"), str)
                and isinstance(message.get("content"), str)
                for message in request_messages
            )
            and hash_json(request_messages) == result.get("recorded_model_messages_sha256")
        )
        provider_attempt_request_valid = (
            request_contract_valid
            and isinstance(attempts, list)
            and len(attempts) == 1
            and attempts[0].get("model_messages_sha256")
            == result.get("recorded_model_messages_sha256")
            and attempts[0].get("provider_request_sha256")
            == _provider_request_sha256(
                request_messages,
                model=str(protocol.get("model")),
                temperature=float(protocol.get("temperature")),
                max_tokens=None,
            )
        )
        provider_response_contract_valid = False
        if isinstance(attempts, list) and len(attempts) == 1:
            attempt = attempts[0]
            raw_response = attempt.get("provider_response_body")
            if isinstance(raw_response, str) and _provider_response_body_preserved(attempt):
                try:
                    response_body = json.loads(raw_response)
                    response_text = extract_chat_text(response_body)
                    response_metadata = _provider_response_metadata(response_body)
                    response_usage = (
                        response_body.get("usage")
                        if isinstance(response_body, dict) and isinstance(response_body.get("usage"), dict)
                        else {}
                    )
                    provider_response_contract_valid = (
                        isinstance(response_body, dict)
                        and response_metadata.get("finish_reason") == "stop"
                        and attempt.get("provider_metadata") == response_metadata
                        and result.get("provider_metadata") == response_metadata
                        and result.get("usage") == response_usage
                        and result.get("response_text") == response_text
                        and result.get("response_text_sha256")
                        == hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                        and result.get("command") == terminal_command_from_response(response_text)
                        and result.get("output_format_valid") == terminal_output_format_valid(response_text)
                    )
                except (json.JSONDecodeError, UnicodeError, KeyError, TypeError, ValueError, AttributeError):
                    provider_response_contract_valid = False
        if (
            result.get("status") != "passed"
            or result.get("finish_reason") != "stop"
            or result.get("output_format_valid") is not True
            or result.get("preflight_max_tokens") is not None
            or result.get("preflight_output_limit_control") != "provider_default_omitted"
            or result.get("model") != protocol.get("model")
            or result.get("base_url") != protocol.get("base_url")
            or not isinstance(provenance, dict)
            or not plan_binding_valid
            or provenance.get("source_eligibility") != "invalidated_protocol_context"
            or provenance.get("source_protocol_disposition") != required_disposition
            or provenance.get("character") != character
            or not isinstance(provenance.get("source_report_required_sha256"), str)
            or not isinstance(attempts, list)
            or len(attempts) != 1
            or not provider_attempt_request_valid
            or not provider_response_contract_valid
            or attempts[0].get("status") != "succeeded"
            or attempts[0].get("failure_class") is not None
            or not _provider_attempt_bodies_preserved(attempts)
        ):
            raise DirectCliExperimentError("Preflight artifact failed the provider-default evidence contract")
        if not isinstance(character, str) or not character:
            raise DirectCliExperimentError("Preflight artifact character label is invalid")
        character_counts[character] = character_counts.get(character, 0) + 1
        if not isinstance(selection, str) or not selection:
            raise DirectCliExperimentError("Preflight artifact selection label is invalid")
        selection_counts[(character, selection)] = selection_counts.get((character, selection), 0) + 1
        decision_index = result.get("decision_index")
        if isinstance(decision_index, bool) or not isinstance(decision_index, int) or decision_index < 0:
            raise DirectCliExperimentError("Preflight artifact decision index is invalid")
        if record.get("name") != artifact_path.stem:
            raise DirectCliExperimentError("Preflight artifact name does not match its file")
        if selection == "historical_failure":
            recorded_metadata = result.get("recorded_provider_metadata")
            if (
                result.get("recorded_event_type") != "model_protocol_error"
                or not isinstance(recorded_metadata, dict)
                or recorded_metadata.get("finish_reason") != "length"
                or not isinstance(result.get("recorded_event_error"), str)
                or "finish_reason=length" not in result["recorded_event_error"]
                or result.get("recorded_max_tokens") != evidence.get("historical_failure_max_tokens")
            ):
                raise DirectCliExperimentError(
                    "Historical failure preflight did not originate from finish_reason=length"
                )
        else:
            recorded_metadata = result.get("recorded_provider_metadata")
            if (
                result.get("recorded_event_type") != "terminal_decision"
                or result.get("recorded_output_format_valid") is not True
                or not isinstance(recorded_metadata, dict)
                or recorded_metadata.get("finish_reason") != "stop"
            ):
                raise DirectCliExperimentError("Standard preflight did not originate from a terminal decision")
            terminal_decision_count = plan_source.get("terminal_decision_count")
            expected_index = {
                "early": 0,
                "middle": (terminal_decision_count - 1) // 2
                if isinstance(terminal_decision_count, int) and terminal_decision_count > 0
                else None,
                "late": terminal_decision_count - 1
                if isinstance(terminal_decision_count, int) and terminal_decision_count > 0
                else None,
            }.get(selection)
            if decision_index != expected_index:
                raise DirectCliExperimentError("Standard preflight violates the locked endpoint/midpoint rule")
            selection_decision_indexes[(character, selection)] = decision_index
        if selection == "historical_failure":
            historical_failure_names.append(str(record.get("name")))
    expected_characters = {"Ironclad", "Silent", "Defect", "Regent", "Necrobinder"}
    standard_positions = {"early", "middle", "late"}
    if set(character_counts) != expected_characters or any(
        selection_counts.get((character, position)) != 1
        for character in expected_characters
        for position in standard_positions
    ):
        raise DirectCliExperimentError("Preflight manifest lacks exact early/middle/late coverage for every character")
    if sorted(historical_failure_names) != sorted(required_failure_names):
        raise DirectCliExperimentError("Preflight manifest lacks the exact historical failure contexts")
    if any(
        not (
            selection_decision_indexes[(character, "early")]
            < selection_decision_indexes[(character, "middle")]
            < selection_decision_indexes[(character, "late")]
        )
        for character in expected_characters
    ):
        raise DirectCliExperimentError("Preflight early/middle/late decision positions are not ordered")
    return manifest


def _source_inventory(protocol_path: Path, sts2_cli_root: Path) -> list[dict[str, Any]]:
    sources: list[tuple[str, Path, bool]] = []
    for relative in _BENCHMARK_EXECUTION_FILES:
        path = ROOT / relative
        sources.append((f"benchmark/{relative}", path, True))
    try:
        protocol_label = protocol_path.relative_to(ROOT).as_posix()
    except ValueError:
        protocol_label = f"external-protocol/{protocol_path.name}"
    sources.append((f"benchmark/{protocol_label}", protocol_path, True))
    protocol = _read_json(protocol_path)
    if isinstance(protocol.get("preflight_evidence"), dict):
        manifest_path = _preflight_manifest_path(protocol)
        plan_path = _preflight_selection_plan_path(protocol)
        manifest = _read_json(manifest_path)
        preflight_paths = [manifest_path, plan_path]
        for record in manifest.get("artifacts") or []:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise DirectCliExperimentError("Preflight manifest artifact entry is invalid")
            candidate = (manifest_path.parent / record["path"]).resolve()
            if not _path_within(candidate, ROOT.resolve()):
                raise DirectCliExperimentError("Preflight artifact must remain inside the benchmark root")
            preflight_paths.append(candidate)
        for path in preflight_paths:
            relative = path.relative_to(ROOT.resolve()).as_posix()
            sources.append((f"benchmark/{relative}", path, True))
    python_root = sts2_cli_root / "python"
    for relative in _CLI_PYTHON_EXECUTION_FILES:
        path = python_root / relative
        sources.append((f"sts2-cli/python/{relative}", path, True))
    headless_source_root = sts2_cli_root / "src" / "Sts2Headless"
    for pattern in ("*.cs", "*.csproj"):
        for path in sorted(headless_source_root.rglob(pattern)):
            if "bin" in path.parts or "obj" in path.parts:
                continue
            relative = path.relative_to(headless_source_root).as_posix()
            sources.append((f"sts2-cli/src/Sts2Headless/{relative}", path, True))
    stubs_source_root = sts2_cli_root / "src" / "GodotStubs"
    stubs_source_files: list[Path] = []
    for pattern in ("*.cs", "*.csproj"):
        for path in sorted(stubs_source_root.rglob(pattern)):
            if "bin" in path.parts or "obj" in path.parts:
                continue
            stubs_source_files.append(path)
            relative = path.relative_to(stubs_source_root).as_posix()
            sources.append((f"sts2-cli/src/GodotStubs/{relative}", path, True))
    if not stubs_source_files:
        raise DirectCliExperimentError("GodotStubs source closure is missing")
    runtime_root = headless_source_root / "bin" / "Debug" / "net9.0"
    headless_dll = runtime_root / "Sts2Headless.dll"
    stubs_dll = runtime_root / "GodotSharp.dll"
    game_dll = sts2_cli_root / "lib" / "sts2.dll"
    if not headless_dll.is_file() or not stubs_dll.is_file() or not game_dll.is_file():
        raise DirectCliExperimentError(
            "A prebuilt headless runtime, Godot stubs, and lib/sts2.dll are required before source locking"
        )
    headless_mtime = headless_dll.stat().st_mtime
    stubs_mtime = stubs_dll.stat().st_mtime
    stale_inputs = [
        path
        for path in headless_source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".cs", ".csproj"}
        and "bin" not in path.parts
        and "obj" not in path.parts
        and path.stat().st_mtime > headless_mtime
    ]
    stale_stub_inputs = [
        path
        for path in stubs_source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".cs", ".csproj"}
        and "bin" not in path.parts
        and "obj" not in path.parts
        and path.stat().st_mtime > stubs_mtime
    ]
    if game_dll.stat().st_mtime > headless_mtime or stale_inputs or stale_stub_inputs:
        raise DirectCliExperimentError("Headless runtime is stale; build it before creating the source lock")
    for path in sorted(runtime_root.glob("*")):
        if path.is_file() and path.suffix.lower() in {".dll", ".exe", ".json"}:
            sources.append((f"sts2-cli/src/Sts2Headless/bin/Debug/net9.0/{path.name}", path, True))
    library_root = sts2_cli_root / "lib"
    for path in sorted(library_root.glob("*")):
        if path.is_file():
            sources.append((f"sts2-cli/lib/{path.name}", path, True))
    for localization_name in ("localization_eng", "localization_zhs"):
        localization_root = sts2_cli_root / localization_name
        completion_marker = localization_root / "_done.txt"
        localization_files = sorted(path for path in localization_root.rglob("*") if path.is_file())
        localization_json = [path for path in localization_files if path.suffix.lower() == ".json"]
        if not completion_marker.is_file() or not localization_json:
            raise DirectCliExperimentError(
                f"Localization resource tree is incomplete: {localization_root}"
            )
        for path in localization_files:
            relative = path.relative_to(sts2_cli_root).as_posix()
            sources.append((f"sts2-cli/{relative}", path, True))
    dotnet_root = sts2_cli_root / ".tools" / "dotnet"
    dotnet_executable = dotnet_root / ("dotnet.exe" if os.name == "nt" else "dotnet")
    if not dotnet_executable.is_file():
        raise DirectCliExperimentError(f"Required local dotnet host is missing: {dotnet_executable}")
    sources.append((f"sts2-cli/.tools/dotnet/{dotnet_executable.name}", dotnet_executable, True))
    for relative_root in (Path("host") / "fxr", Path("shared") / "Microsoft.NETCore.App"):
        runtime_files = sorted(path for path in (dotnet_root / relative_root).rglob("*") if path.is_file())
        if not runtime_files:
            raise DirectCliExperimentError(
                f"Required local dotnet runtime is missing beneath: {dotnet_root / relative_root}"
            )
        for path in runtime_files:
            relative = path.relative_to(dotnet_root).as_posix()
            sources.append((f"sts2-cli/.tools/dotnet/{relative}", path, True))
    inventory: list[dict[str, Any]] = []
    for label, path, tracked in sources:
        if not path.is_file():
            raise DirectCliExperimentError(f"Required locked source is missing: {path}")
        inventory.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": hash_file(path),
                "size": path.stat().st_size,
                "tracked": tracked,
            }
        )
    return inventory


def build_source_lock(protocol_path: str | Path, *, sts2_cli_root: str | Path) -> dict[str, Any]:
    resolved_protocol = Path(protocol_path).resolve()
    resolved_cli_root = Path(sts2_cli_root).resolve()
    protocol = load_protocol(resolved_protocol)
    files = _source_inventory(resolved_protocol, resolved_cli_root)
    identity_payload = {
        "protocol_sha256": hash_file(resolved_protocol),
        "runtime": _python_runtime_identity(),
        "files": [{"label": item["label"], "sha256": item["sha256"], "size": item["size"]} for item in files],
    }
    return {
        "schema_version": 1,
        "identity_sha256": hash_json(identity_payload),
        **identity_payload,
        "game_rules": protocol["game_rules"],
        "inventory": files,
    }


def _read_and_verify_source_lock(out_root: Path) -> dict[str, Any]:
    freeze_path = out_root / "source_lock.json"
    frozen = _read_json(freeze_path)
    recorded_payload = {
        "protocol_sha256": frozen.get("protocol_sha256"),
        "runtime": frozen.get("runtime"),
        "files": frozen.get("files"),
    }
    inventory = frozen.get("inventory") if isinstance(frozen.get("inventory"), list) else []
    normalized_inventory = [
        {"label": item.get("label"), "sha256": item.get("sha256"), "size": item.get("size")}
        for item in inventory
        if isinstance(item, dict)
    ]
    protocol_label = frozen.get("protocol_label")
    protocol_items = [
        item
        for item in inventory
        if isinstance(item, dict)
        and item.get("label") == protocol_label
        and item.get("sha256") == frozen.get("protocol_sha256")
    ]
    frozen_record_valid = (
        frozen.get("identity_sha256") == hash_json(recorded_payload)
        and frozen.get("runtime") == _python_runtime_identity()
        and bool(inventory)
        and normalized_inventory == frozen.get("files")
        and all(isinstance(item, dict) and item.get("tracked") is True for item in inventory)
        and frozen.get("game_rules")
        == {
            **game_rules_identity(),
            "delivery": "static_full_system_prompt",
            "source": "sts2_bench/game_rules.py",
        }
        and frozen.get("execution_mode") == "active_worktree"
        and isinstance(protocol_label, str)
        and len(protocol_items) == 1
        and Path(str(frozen.get("benchmark_root") or "")).resolve() == ROOT.resolve()
        and isinstance(frozen.get("protocol_path"), str)
        and isinstance(frozen.get("sts2_cli_root"), str)
    )
    if not frozen_record_valid:
        raise DirectCliExperimentError("Source lock record is invalid")
    source_identity = _verify_live_source_inventory(frozen)
    if frozen.get("source_tree_identity_sha256") != source_identity:
        raise DirectCliExperimentError("Locked active source identity drifted")
    return frozen


def _revalidate_source_lock(
    out_root: Path,
    *,
    expected_identity: str,
    expected_source_identity: str,
) -> dict[str, Any]:
    frozen = _read_and_verify_source_lock(out_root)
    if frozen.get("identity_sha256") != expected_identity:
        raise DirectCliExperimentError("Source lock record changed during the experiment")
    if frozen.get("source_tree_identity_sha256") != expected_source_identity:
        raise DirectCliExperimentError("Locked active source identity changed during the experiment")
    return frozen


def initialize_or_verify_source_lock(
    out_root: Path,
    *,
    protocol_path: Path,
    sts2_cli_root: Path,
) -> dict[str, Any]:
    freeze_path = out_root / "source_lock.json"
    if freeze_path.is_file():
        frozen = _read_and_verify_source_lock(out_root)
        resolved_protocol = protocol_path.resolve()
        if (
            resolved_protocol != Path(str(frozen["protocol_path"])).resolve()
            or hash_file(resolved_protocol) != frozen.get("protocol_sha256")
        ):
            raise DirectCliExperimentError(
                "The requested protocol does not match this source-locked experiment"
            )
        resolved_cli_root = sts2_cli_root.resolve()
        if resolved_cli_root != Path(str(frozen["sts2_cli_root"])).resolve():
            raise DirectCliExperimentError(
                "The requested sts2-cli root does not match this source-locked experiment"
            )
        return frozen

    current = build_source_lock(protocol_path, sts2_cli_root=sts2_cli_root)
    source_identity = _live_source_identity(current["inventory"])
    protocol_label = next(
        (
            str(item["label"])
            for item in current["inventory"]
            if Path(str(item["path"])).resolve() == protocol_path.resolve()
        ),
        None,
    )
    if protocol_label is None:
        raise DirectCliExperimentError("Source inventory does not contain the experiment protocol")
    frozen = {
        **current,
        "source_tree_identity_sha256": source_identity,
        "protocol_label": protocol_label,
        "execution_mode": "active_worktree",
        "created_at": _utc_now(),
        "benchmark_root": str(ROOT.resolve()),
        "protocol_path": str(protocol_path.resolve()),
        "sts2_cli_root": str(sts2_cli_root.resolve()),
    }
    _atomic_write_json(freeze_path, frozen)
    return frozen


def _live_source_identity(inventory: list[dict[str, Any]]) -> str:
    records: list[dict[str, Any]] = []
    for item in inventory:
        if not item.get("tracked"):
            continue
        source = Path(str(item["path"])).resolve()
        if not source.is_file():
            raise DirectCliExperimentError(f"Locked active source is missing: {source}")
        record = {
            "label": item["label"],
            "sha256": hash_file(source),
            "size": source.stat().st_size,
        }
        if record["sha256"] != item.get("sha256") or record["size"] != item.get("size"):
            raise DirectCliExperimentError(f"Locked active source file drifted: {item['label']}")
        records.append(record)
    return hash_json(records)


def _verify_live_source_inventory(frozen: dict[str, Any]) -> str:
    inventory = frozen.get("inventory")
    if not isinstance(inventory, list):
        raise DirectCliExperimentError("Source lock is missing its file inventory")
    return _live_source_identity(inventory)


def _option(args: list[str], name: str) -> str | None:
    for index, item in enumerate(args):
        if item == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _attempt_command(
    protocol: dict[str, Any],
    *,
    seed: str,
    attempt_artifacts: Path,
    sts2_cli_root: Path,
    source_lock: str,
    source_tree_identity: str,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        "-s",
        "-X",
        "utf8",
        "-m",
        "sts2_bench.direct_cli_baseline",
        "--sts2-cli-root",
        str(sts2_cli_root),
        "--base-url",
        str(protocol["base_url"]),
        "--model",
        str(protocol["model"]),
        "--temperature",
        str(protocol.get("temperature", 0.0)),
        "--timeout",
        str(protocol.get("request_timeout_sec", 180)),
        "--request-retries",
        str(protocol.get("request_retries", 1)),
        "--history-chars",
        str(protocol.get("history_chars", 24000)),
        "--max-decisions",
        str(protocol.get("max_decisions", 5000)),
        "--seed",
        seed,
        "--character",
        str(protocol.get("character", "Ironclad")),
        "--ascension",
        str(protocol.get("ascension", 0)),
        "--lang",
        str(protocol.get("lang", "en")),
        "--out-dir",
        str(attempt_artifacts),
        "--no-echo-terminal",
        "--source-lock-sha256",
        source_lock,
        "--source-tree-identity-sha256",
        source_tree_identity,
    ]
    if protocol.get("max_tokens") is None:
        command.append("--omit-max-tokens")
    else:
        command.extend(("--max-tokens", str(protocol["max_tokens"])))
    return command


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
        else:  # pragma: no cover - Windows is the active experiment platform
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
        raise DirectCliExperimentError(f"Corrupted attempt lease requires inspection: {lease_path}: {error}")
    if not lease:
        return None
    if _pid_alive(lease.get("owner_pid")) or _pid_alive(lease.get("child_pid")):
        return lease
    lease["state"] = "orphaned_processes_gone"
    lease["observed_at"] = _utc_now()
    _atomic_write_json(job_dir / "orphaned_attempt.json", lease)
    lease_path.unlink(missing_ok=True)
    return None


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
        raise DirectCliExperimentError("Every formal direct-CLI attempt requires a positive hard watchdog")
    creation: dict[str, Any] = {}
    if os.name == "nt":
        creation["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:  # pragma: no cover - Windows is the active experiment platform
        creation["start_new_session"] = True
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(cwd), stdout=stdout, stderr=stderr, text=True, **creation)
        lease, error = _try_read_json(lease_path)
        if error or not lease:
            _terminate_process_tree(process)
            raise DirectCliExperimentError(f"Attempt lease disappeared or became corrupt: {lease_path}")
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
            stderr.write(f"formal direct-CLI attempt timed out after {timeout_sec:g}s\n")
            stderr.flush()
            return 124
        except BaseException:
            if _terminate_process_tree(process) is False:
                _mark_attempt_cleanup_unconfirmed(lease_path, process)
            raise


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _selected_api_key() -> str | None:
    value = (
        os.environ.get("STS2_BENCH_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    return value.strip() if isinstance(value, str) and value.strip() else None


def _endpoint_requires_api_key(base_url: str) -> bool:
    hostname = (urllib.parse.urlsplit(base_url).hostname or "").strip().lower()
    return hostname not in {"", "localhost", "127.0.0.1", "::1"}


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_and_validate_external_environment(
    env_file: str | Path | None,
    *,
    base_url: str,
    forbidden_roots: tuple[Path, ...],
) -> None:
    if env_file is None:
        raise DirectCliExperimentError("A formal source-locked run requires an explicit external --env-file")
    candidate = Path(env_file).expanduser()
    lexical_path = Path(os.path.abspath(candidate))
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise DirectCliExperimentError("The external environment file does not exist") from exc
    if not path.is_file():
        raise DirectCliExperimentError("The external environment file does not exist")
    for root in forbidden_roots:
        lexical_root = Path(os.path.abspath(root))
        resolved_root = root.resolve(strict=False)
        if _path_is_within(lexical_path, lexical_root) or _path_is_within(path, resolved_root):
            raise DirectCliExperimentError(
                "The external environment file must be outside the formal result root"
            )
    try:
        link_count = path.stat().st_nlink
    except OSError as exc:
        raise DirectCliExperimentError("The external environment file does not exist") from exc
    if link_count != 1:
        raise DirectCliExperimentError("The external environment file must be a single-link regular file")
    credential_names = {
        "STS2_BENCH_API_KEY",
        "QWEN_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    }
    for name in (*credential_names, "STS2_LIB", "STS2_GAME_DIR"):
        os.environ.pop(name, None)
    loaded: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if normalized_key not in credential_names:
            continue
        os.environ[normalized_key] = value.strip().strip('"').strip("'")
        loaded.add(normalized_key)
    selected = _selected_api_key()
    credential_loaded = any(name in loaded for name in credential_names)
    if _endpoint_requires_api_key(base_url) and (
        not credential_loaded or selected is None or selected.lower() == "local"
    ):
        raise DirectCliExperimentError("The configured non-local model endpoint requires an API key")


def _audit_successful_provider_response(
    attempt: dict[str, Any],
    *,
    expected_model_raw: Any,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    raw_body = attempt.get("provider_response_body")
    preserved = _provider_response_body_preserved(attempt)
    body: dict[str, Any] | None = None
    text: str | None = None
    finish_reason: Any = None
    usage: dict[str, Any] = {}
    derived_metadata: dict[str, Any] = {}
    if preserved:
        try:
            candidate = json.loads(raw_body)
            if isinstance(candidate, dict):
                body = candidate
                text = extract_chat_text(body)
                usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
                derived_metadata = _provider_response_metadata(body)
                choices = body.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    finish_reason = choices[0].get("finish_reason")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            pass
    metadata = attempt.get("provider_metadata")
    checks = {
        "provider_success_response_preserved": preserved,
        "provider_success_response_is_json_object": isinstance(body, dict),
        "provider_success_response_matches_model_raw": isinstance(expected_model_raw, str)
        and text == expected_model_raw,
        "provider_success_finish_reason_is_stop": finish_reason == "stop",
        "provider_success_finish_reason_matches_metadata": isinstance(metadata, dict)
        and metadata.get("finish_reason") == finish_reason,
        "provider_success_metadata_matches_body": isinstance(metadata, dict)
        and metadata == derived_metadata,
    }
    return checks, usage, derived_metadata


def _provider_attempt_bodies_preserved(attempts: list[dict[str, Any]]) -> bool:
    for attempt in attempts:
        body = attempt.get("provider_response_body")
        digest = attempt.get("provider_response_sha256")
        if body is None and digest is None:
            continue
        if not _provider_response_body_preserved(attempt):
            return False
    return True


def _provider_response_body_preserved(attempt: dict[str, Any]) -> bool:
    body = attempt.get("provider_response_body")
    digest = attempt.get("provider_response_sha256")
    encoding = attempt.get("provider_response_encoding")
    if not isinstance(body, str) or not isinstance(digest, str):
        return False
    if encoding in {None, "utf-8"}:
        return digest == hashlib.sha256(body.encode("utf-8")).hexdigest()
    if encoding != "invalid_utf8_with_base64":
        return False
    encoded = attempt.get("provider_response_body_base64")
    if not isinstance(encoded, str):
        return False
    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return False
    return (
        digest == hashlib.sha256(raw_bytes).hexdigest()
        and body == raw_bytes.decode("utf-8", errors="replace")
    )


def _provider_attempt_usage(attempt: dict[str, Any]) -> dict[str, Any]:
    raw_body = attempt.get("provider_response_body")
    if not isinstance(raw_body, str):
        return {}
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeError):
        return {}
    return dict(body.get("usage")) if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}


def _merged_provider_attempt_usage(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return _merge_provider_usage(*(_provider_attempt_usage(attempt) for attempt in attempts))


def _provider_request_sha256(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _protocol_attempt_response_state(attempt: dict[str, Any]) -> dict[str, Any]:
    raw_body = attempt.get("provider_response_body")
    preserved = _provider_response_body_preserved(attempt)
    body: dict[str, Any] | None = None
    response_text: str | None = None
    finish_reason: Any = None
    if preserved and isinstance(raw_body, str):
        try:
            candidate = json.loads(raw_body)
            if isinstance(candidate, dict):
                body = candidate
        except (json.JSONDecodeError, UnicodeError):
            pass
    if isinstance(body, dict):
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
        try:
            response_text = extract_chat_text(body)
        except (KeyError, TypeError, ValueError, AttributeError):
            response_text = None
    semantic_failure = (
        finish_reason != "stop"
        or not isinstance(response_text, str)
        or not response_text.strip()
    )
    expected_failure_class = (
        "provider_output_truncated" if finish_reason == "length" else "model_protocol_error"
    )
    derived_metadata = _provider_response_metadata(body) if isinstance(body, dict) else {}
    return {
        "valid": preserved
        and semantic_failure
        and attempt.get("failure_class") == expected_failure_class
        and attempt.get("http_status") is None
        and attempt.get("provider_metadata") == derived_metadata
        and isinstance(attempt.get("error"), str)
        and bool(attempt.get("error")),
        "preserved": preserved,
        "body": body,
        "finish_reason": finish_reason,
        "response_text": response_text,
        "expected_failure_class": expected_failure_class,
        "metadata": derived_metadata,
    }


def _transport_attempt_evidence_valid(attempt: dict[str, Any]) -> bool:
    body = attempt.get("provider_response_body")
    digest = attempt.get("provider_response_sha256")
    status = attempt.get("http_status")
    no_response = body is None and digest is None and status is None
    retryable_http = (
        isinstance(status, int)
        and not isinstance(status, bool)
        and (status in {408, 429} or 500 <= status <= 599)
        and _provider_response_body_preserved(attempt)
    )
    auth_unavailable = (
        isinstance(body, str)
        and _provider_response_body_preserved(attempt)
        and ("auth_unavailable" in body.lower() or "no auth available" in body.lower())
    )
    incomplete_response = (
        attempt.get("partial_response") is True
        and isinstance(attempt.get("partial_response_received_bytes"), int)
        and not isinstance(attempt.get("partial_response_received_bytes"), bool)
        and attempt.get("partial_response_received_bytes") >= 0
        and (
            attempt.get("partial_response_expected_remaining_bytes") is None
            or (
                isinstance(attempt.get("partial_response_expected_remaining_bytes"), int)
                and not isinstance(attempt.get("partial_response_expected_remaining_bytes"), bool)
                and attempt.get("partial_response_expected_remaining_bytes") >= 0
            )
        )
        and _provider_response_body_preserved(attempt)
    )
    return (
        attempt.get("failure_class") == "provider_transport_error"
        and isinstance(attempt.get("error"), str)
        and bool(attempt.get("error"))
        and attempt.get("provider_metadata") == {}
        and (no_response or retryable_http or auth_unavailable or incomplete_response)
    )


def _provider_attempt_chain_checks(
    attempts: list[dict[str, Any]],
    *,
    base_messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int | None,
    request_retries: int,
    final_status: str,
) -> dict[str, bool]:
    expected_messages = [dict(message) for message in base_messages]
    hashes_match = True
    request_hashes_match = True
    statuses_valid = bool(attempts)
    failure_classes_valid = bool(attempts)
    failure_evidence_valid = bool(attempts)
    latencies_valid = bool(attempts)
    allowed_retry_failures = {
        "provider_transport_error",
        "provider_output_truncated",
        "model_protocol_error",
    }
    for index, attempt in enumerate(attempts):
        latency = attempt.get("latency_sec")
        latencies_valid = latencies_valid and (
            isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and math.isfinite(float(latency))
            and float(latency) >= 0
        )
        messages_sha256 = hashlib.sha256(
            json.dumps(
                expected_messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        hashes_match = hashes_match and attempt.get("model_messages_sha256") == messages_sha256
        request_hashes_match = request_hashes_match and attempt.get("provider_request_sha256") == (
            _provider_request_sha256(
                expected_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        is_final = index == len(attempts) - 1
        expected_status = final_status if is_final else "failed"
        statuses_valid = statuses_valid and attempt.get("status") == expected_status
        failure_class = attempt.get("failure_class")
        if is_final and final_status == "succeeded":
            failure_classes_valid = failure_classes_valid and failure_class is None
        elif is_final:
            failure_classes_valid = failure_classes_valid and failure_class in {
                "provider_output_truncated",
                "model_protocol_error",
            }
        else:
            failure_classes_valid = failure_classes_valid and failure_class in allowed_retry_failures
        if attempt.get("status") == "failed":
            if failure_class == "provider_transport_error":
                failure_evidence_valid = failure_evidence_valid and _transport_attempt_evidence_valid(attempt)
            else:
                failure_evidence_valid = failure_evidence_valid and _protocol_attempt_response_state(attempt)["valid"]
        if not is_final and failure_class in {"provider_output_truncated", "model_protocol_error"}:
            expected_messages = [
                *expected_messages,
                {"role": "user", "content": OUTPUT_RETRY_MESSAGE},
            ]
    return {
        "provider_attempt_count_within_request_budget": bool(attempts)
        and len(attempts) <= request_retries + 1,
        "provider_attempt_message_hashes_match": hashes_match,
        "provider_attempt_request_hashes_match": request_hashes_match,
        "provider_attempt_statuses_valid": statuses_valid,
        "provider_attempt_failure_classes_valid": failure_classes_valid,
        "provider_attempt_failure_evidence_valid": failure_evidence_valid,
        "provider_attempt_latencies_valid": latencies_valid,
    }


def _audit_protocol_error_response(
    attempt: dict[str, Any],
    *,
    event: dict[str, Any],
    expected_usage: dict[str, Any],
) -> dict[str, bool]:
    raw_body = attempt.get("provider_response_body")
    preserved = _provider_response_body_preserved(attempt)
    body: dict[str, Any] | None = None
    if preserved:
        try:
            candidate = json.loads(raw_body)
            if isinstance(candidate, dict):
                body = candidate
        except (json.JSONDecodeError, UnicodeError):
            pass
    metadata = _provider_response_metadata(body) if isinstance(body, dict) else {}
    response_state = _protocol_attempt_response_state(attempt)
    return {
        "protocol_failure_response_preserved": preserved,
        "protocol_failure_usage_matches_attempt_chain": event.get("usage") == expected_usage,
        "protocol_failure_metadata_matches_body": event.get("provider_metadata") == metadata
        and attempt.get("provider_metadata") == metadata,
        "protocol_failure_semantics_valid": response_state["valid"],
        "protocol_failure_class_matches_body": attempt.get("failure_class")
        == response_state["expected_failure_class"],
        "protocol_failure_error_matches_attempt": isinstance(event.get("error"), str)
        and event.get("error") == attempt.get("error"),
        "protocol_failure_has_no_http_configuration_status": attempt.get("http_status") is None,
        "protocol_failure_latency_is_numeric": isinstance(event.get("latency_sec"), (int, float))
        and not isinstance(event.get("latency_sec"), bool),
    }


def _valid_direct_prompt_contract(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    option_tokens = value.get("canonical_option_tokens")
    accepted_syntax = value.get("accepted_syntax")
    meta_commands = value.get("meta_commands")
    forbidden_meta_commands = value.get("forbidden_meta_commands")
    valid_syntax = {
        "canonical_token",
        "combat_card_or_sequence",
        "comma_separated_unique_tokens",
        "empty_input",
    }
    return bool(
        isinstance(value.get("decision"), str)
        and value["decision"]
        and isinstance(option_tokens, list)
        and all(isinstance(token, str) for token in option_tokens)
        and option_tokens == sorted(set(option_tokens))
        and isinstance(accepted_syntax, list)
        and accepted_syntax
        and all(isinstance(syntax, str) and syntax in valid_syntax for syntax in accepted_syntax)
        and accepted_syntax == sorted(set(accepted_syntax))
        and "canonical_token" in accepted_syntax
        and isinstance(meta_commands, list)
        and all(isinstance(command, str) for command in meta_commands)
        and meta_commands == sorted(set(meta_commands))
        and isinstance(forbidden_meta_commands, list)
        and forbidden_meta_commands == ["save", "saves"]
        and not set(meta_commands).intersection(forbidden_meta_commands)
        and isinstance(value.get("multi_select"), bool)
        and (
            value.get("multi_select") is False
            or "comma_separated_unique_tokens" in accepted_syntax
        )
        and (
            "comma_separated_unique_tokens" not in accepted_syntax
            or value.get("multi_select") is True
        )
        and isinstance(value.get("multi_min"), int)
        and not isinstance(value.get("multi_min"), bool)
        and isinstance(value.get("multi_max"), int)
        and not isinstance(value.get("multi_max"), bool)
        and 0 <= value["multi_min"] <= value["multi_max"]
    )


def _audit_terminal_decision_requests(
    terminal_decisions: list[dict[str, Any]],
    *,
    provider_attempts: list[dict[str, Any]],
    protocol_errors: list[dict[str, Any]],
    terminal: str,
    history_limit: int,
    request_model: str,
    request_temperature: float,
    request_max_tokens: int | None,
    request_retries: int,
    require_prompt_contract: bool = False,
) -> dict[str, Any]:
    rules_identity = game_rules_identity()
    rules_block = render_full_game_rules()
    decisions: list[dict[str, Any]] = []
    previous_end = -1
    for expected_index, item in enumerate(terminal_decisions):
        start = item.get("terminal_window_start")
        end = item.get("terminal_window_end")
        valid_span = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start <= end <= len(terminal)
        )
        visible = terminal[start:end] if valid_span else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                    + visible
                    + "\n\nReturn only the next terminal input line."
                ),
            },
        ]
        messages_sha256 = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rules_occurrences = sum(str(message.get("content") or "").count(rules_block) for message in messages)
        decision_provider_attempts = [
            attempt
            for attempt in provider_attempts
            if attempt.get("decision_index") == expected_index
        ]
        provider_indexes = [attempt.get("provider_attempt_index") for attempt in decision_provider_attempts]
        provider_chain_checks = _provider_attempt_chain_checks(
            decision_provider_attempts,
            base_messages=messages,
            model=request_model,
            temperature=request_temperature,
            max_tokens=request_max_tokens,
            request_retries=request_retries,
            final_status="succeeded",
        )
        raw_output = item.get("model_raw")
        command = item.get("command")
        expected_command = terminal_command_from_response(raw_output) if isinstance(raw_output, str) else None
        echoed_command = command + "\n" if isinstance(command, str) else None
        final_provider_metadata = (
            decision_provider_attempts[-1].get("provider_metadata") if decision_provider_attempts else None
        )
        provider_response_checks, _final_provider_usage, _derived_metadata = _audit_successful_provider_response(
            decision_provider_attempts[-1] if decision_provider_attempts else {},
            expected_model_raw=raw_output,
        )
        provider_usage = _merged_provider_attempt_usage(decision_provider_attempts)
        prompt_contract_valid = _valid_direct_prompt_contract(item.get("prompt_contract"))
        checks = {
            "decision_index_matches_order": item.get("decision_index") == expected_index,
            "terminal_window_span_valid": valid_span,
            "terminal_window_order_valid": valid_span and end > previous_end,
            "terminal_window_fifo_bounds_match": valid_span
            and start == max(0, end - history_limit),
            "terminal_window_length_matches": valid_span and item.get("history_chars") == len(visible),
            "input_prompt_matches_window_suffix": valid_span
            and isinstance(item.get("prompt"), str)
            and visible.endswith(item["prompt"]),
            "terminal_window_hash_matches": valid_span
            and item.get("terminal_window_sha256") == hashlib.sha256(visible.encode("utf-8")).hexdigest(),
            "model_messages_hash_matches": valid_span and item.get("model_messages_sha256") == messages_sha256,
            "provider_attempt_count_matches": item.get("provider_attempt_count")
            == len(decision_provider_attempts),
            "provider_attempt_indexes_contiguous": provider_indexes
            == list(range(len(decision_provider_attempts))),
            **provider_chain_checks,
            "provider_attempt_response_bodies_preserved": _provider_attempt_bodies_preserved(
                decision_provider_attempts
            ),
            "model_raw_is_text": isinstance(raw_output, str),
            "command_matches_model_raw": isinstance(command, str) and command == expected_command,
            "output_format_flag_matches_model_raw": isinstance(raw_output, str)
            and item.get("output_format_valid") == terminal_output_format_valid(raw_output),
            "prompt_contract_recorded": prompt_contract_valid if require_prompt_contract else True,
            "command_echo_matches_terminal": valid_span
            and isinstance(echoed_command, str)
            and terminal[end : end + len(echoed_command)] == echoed_command,
            "usage_is_object": isinstance(item.get("usage"), dict),
            "usage_matches_provider_body": item.get("usage") == provider_usage,
            "latency_is_numeric": isinstance(item.get("latency_sec"), (int, float))
            and not isinstance(item.get("latency_sec"), bool),
            "provider_metadata_matches": isinstance(item.get("provider_metadata"), dict)
            and item.get("provider_metadata") == (final_provider_metadata or {}),
            "rules_block_occurs_exactly_once": rules_occurrences == 1,
            "rules_identity_matches": item.get("game_ruleset_id") == rules_identity["ruleset_id"]
            and item.get("game_ruleset_sha256") == rules_identity["ruleset_sha256"]
            and item.get("game_rule_module_ids") == rules_identity["module_ids"]
            and item.get("game_rule_delivery") == "static_full_system_prompt",
            **provider_response_checks,
        }
        decisions.append(
            {
                "decision_index": item.get("decision_index"),
                "valid": all(checks.values()),
                "checks": checks,
                "reconstructed_model_messages_sha256": messages_sha256,
                "provider_attempt_count": len(decision_provider_attempts),
            }
        )
        if valid_span:
            previous_end = end
    for offset, item in enumerate(protocol_errors):
        expected_index = len(terminal_decisions) + offset
        start = item.get("terminal_window_start")
        end = item.get("terminal_window_end")
        valid_span = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start <= end <= len(terminal)
        )
        visible = terminal[start:end] if valid_span else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                    + visible
                    + "\n\nReturn only the next terminal input line."
                ),
            },
        ]
        messages_sha256 = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rules_occurrences = sum(str(message.get("content") or "").count(rules_block) for message in messages)
        attempts = [attempt for attempt in provider_attempts if attempt.get("decision_index") == expected_index]
        provider_indexes = [attempt.get("provider_attempt_index") for attempt in attempts]
        provider_chain_checks = _provider_attempt_chain_checks(
            attempts,
            base_messages=messages,
            model=request_model,
            temperature=request_temperature,
            max_tokens=request_max_tokens,
            request_retries=request_retries,
            final_status="failed",
        )
        final_attempt = attempts[-1] if attempts else {}
        protocol_response_checks = _audit_protocol_error_response(
            final_attempt,
            event=item,
            expected_usage=_merged_provider_attempt_usage(attempts),
        )
        prompt_contract_valid = _valid_direct_prompt_contract(item.get("prompt_contract"))
        checks = {
            "decision_index_matches_order": item.get("decision_index") == expected_index,
            "terminal_window_span_valid": valid_span,
            "terminal_window_order_valid": valid_span and end > previous_end,
            "terminal_window_fifo_bounds_match": valid_span and start == max(0, end - history_limit),
            "terminal_window_length_matches": valid_span and item.get("history_chars") == len(visible),
            "input_prompt_matches_window_suffix": valid_span
            and isinstance(item.get("prompt"), str)
            and visible.endswith(item["prompt"]),
            "terminal_window_hash_matches": valid_span
            and item.get("terminal_window_sha256") == hashlib.sha256(visible.encode("utf-8")).hexdigest(),
            "model_messages_hash_matches": item.get("model_messages_sha256") == messages_sha256,
            "prompt_contract_recorded": prompt_contract_valid if require_prompt_contract else True,
            "rules_block_occurs_exactly_once": rules_occurrences == 1,
            "rules_identity_matches": item.get("game_ruleset_id") == rules_identity["ruleset_id"]
            and item.get("game_ruleset_sha256") == rules_identity["ruleset_sha256"]
            and item.get("game_rule_module_ids") == rules_identity["module_ids"]
            and item.get("game_rule_delivery") == "static_full_system_prompt",
            "provider_attempt_indexes_contiguous": provider_indexes == list(range(len(attempts))),
            "provider_attempt_count_matches": item.get("provider_attempt_count") == len(attempts),
            **provider_chain_checks,
            "provider_attempt_response_bodies_preserved": _provider_attempt_bodies_preserved(attempts),
            **protocol_response_checks,
        }
        decisions.append(
            {
                "decision_index": item.get("decision_index"),
                "kind": "model_protocol_error",
                "valid": all(checks.values()),
                "checks": checks,
                "reconstructed_model_messages_sha256": messages_sha256,
                "provider_attempt_count": len(attempts),
            }
        )
        if valid_span:
            previous_end = end
    covered_provider_attempts = sum(item["provider_attempt_count"] for item in decisions)
    usage_totals: dict[str, Any] = {}
    total_latency = 0.0
    format_violations = 0
    for item in [*terminal_decisions, *protocol_errors]:
        if isinstance(item.get("usage"), dict):
            usage_totals = _merge_provider_usage(usage_totals, item["usage"])
    for item in provider_attempts:
        latency = item.get("latency_sec")
        if (
            isinstance(latency, (int, float))
            and not isinstance(latency, bool)
            and math.isfinite(float(latency))
            and float(latency) >= 0
        ):
            total_latency += float(latency)
    for item in terminal_decisions:
        if item.get("output_format_valid") is False:
            format_violations += 1
    return {
        "valid": bool(decisions)
        and all(item["valid"] for item in decisions)
        and covered_provider_attempts == len(provider_attempts),
        "decision_count": len(decisions),
        "protocol_error_count": len(protocol_errors),
        "provider_attempt_count": len(provider_attempts),
        "usage_totals": usage_totals,
        "model_latency_sec": round(total_latency, 6),
        "model_output_format_violations": format_violations,
        "decisions": decisions,
    }


def _validate_attempt_artifacts(
    attempt_dir: Path,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    execution_benchmark_root: Path,
    execution_sts2_cli_root: Path,
    source_lock: str,
    source_tree_identity: str,
    expected_seed: str,
    expected_replicate: int,
    returncode: int,
) -> dict[str, Any]:
    artifacts = attempt_dir / "artifacts"
    manifest_path = attempt_dir / "attempt_manifest.json"
    config_path = artifacts / "config.json"
    summary_path = artifacts / "summary.json"
    events_path = artifacts / "events.jsonl"
    terminal_path = artifacts / "terminal.txt"
    engine_trace_path = artifacts / "engine_trace.jsonl"
    benchmark_trace_path = artifacts / "benchmark_trace.jsonl"
    analysis_index_path = artifacts / "trace_analysis" / "summary.json"

    def read_object(path: Path) -> dict[str, Any]:
        try:
            return _read_json(path)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, DirectCliExperimentError):
            return {}

    def read_events(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        values: list[dict[str, Any]] = []
        try:
            for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
                if not raw_line.strip():
                    continue
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    return []
                values.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []
        return values

    attempt_manifest = read_object(manifest_path)
    config = read_object(config_path)
    summary = read_object(summary_path)
    analysis_index = read_object(analysis_index_path)
    events = read_events(events_path)
    terminal = ""
    stderr_text = ""
    terminal_well_formed = False
    if terminal_path.is_file():
        try:
            terminal = terminal_path.read_text(encoding="utf-8-sig")
            terminal_well_formed = True
        except (OSError, UnicodeError):
            pass
    stderr_path = attempt_dir / "stderr.log"
    if stderr_path.is_file():
        try:
            stderr_text = stderr_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            pass
    localization_audit = localization_integrity((terminal,), stderr=stderr_text)
    terminal_decisions = [item for item in events if item.get("type") == "terminal_decision"]
    provider_attempts = [item for item in events if item.get("type") == "provider_request_attempt"]
    protocol_errors = [item for item in events if item.get("type") == "model_protocol_error"]
    expected_rules = {**game_rules_identity(), "delivery": "static_full_system_prompt"}
    expected_protocol_sha = hash_file(protocol_path)
    expected_prompt_sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    expected_controller = execution_benchmark_root / "sts2_bench" / "direct_cli_baseline.py"
    expected_human_cli = execution_sts2_cli_root / "python" / "play.py"
    expected_dotnet = execution_sts2_cli_root / ".tools" / "dotnet" / (
        "dotnet.exe" if os.name == "nt" else "dotnet"
    )
    expected_config = {
        "protocol": "human_cli_raw_fifo_v1",
        "model": protocol["model"],
        "base_url": protocol["base_url"],
        "temperature": protocol.get("temperature", 0.0),
        "timeout": protocol.get("request_timeout_sec", 180),
        "max_tokens": protocol.get("max_tokens"),
        "request_retries": protocol.get("request_retries", 1),
        "thinking_control": "provider_default_omitted",
        "response_format_control": "provider_default_omitted",
        "seed": expected_seed,
        "character": protocol.get("character", "Ironclad"),
        "ascension": protocol.get("ascension", 0),
        "lang": protocol.get("lang", "en"),
        "history_chars": protocol.get("history_chars", 24000),
        "max_decisions": protocol.get("max_decisions", 5000),
        "python_runtime": _python_runtime_identity(),
        "source_locked_runtime_mode": True,
        "engine_telemetry_visible_to_model": False,
        "source_lock_sha256": source_lock,
        "source_tree_identity_sha256": source_tree_identity,
    }
    config_values_match = all(config.get(key) == value for key, value in expected_config.items())
    command = attempt_manifest.get("command") if isinstance(attempt_manifest.get("command"), list) else []
    try:
        command_artifacts = Path(str(_option(command, "--out-dir") or "")).resolve()
    except (OSError, RuntimeError):
        command_artifacts = Path()
    request_audit = _audit_terminal_decision_requests(
        terminal_decisions,
        provider_attempts=provider_attempts,
        protocol_errors=protocol_errors,
        terminal=terminal,
        history_limit=int(config.get("history_chars") or 0),
        request_model=str(config.get("model") or ""),
        request_temperature=float(config.get("temperature") or 0.0),
        request_max_tokens=(
            int(config["max_tokens"])
            if isinstance(config.get("max_tokens"), int) and not isinstance(config.get("max_tokens"), bool)
            else None
        ),
        request_retries=(
            int(config["request_retries"])
            if isinstance(config.get("request_retries"), int)
            and not isinstance(config.get("request_retries"), bool)
            else -1
        ),
        require_prompt_contract=protocol["prompt_contract_required"],
    )
    request_rows = [
        (index, item)
        for index, item in enumerate(events)
        if item.get("type") in {"terminal_decision", "model_protocol_error"}
    ]
    provider_rows = [
        (index, item)
        for index, item in enumerate(events)
        if item.get("type") == "provider_request_attempt"
    ]
    request_event_order_valid = (
        len(request_rows) == len(terminal_decisions) + len(protocol_errors)
        and len(protocol_errors) <= 1
        and all(item.get("decision_index") == decision_index for decision_index, (_, item) in enumerate(request_rows))
        and all(
            previous_position < provider_position < request_position
            for decision_index, (request_position, _) in enumerate(request_rows)
            for provider_position, provider in provider_rows
            if provider.get("decision_index") == decision_index
            for previous_position in [request_rows[decision_index - 1][0] if decision_index else -1]
        )
        and all(
            any(provider.get("decision_index") == decision_index for _, provider in provider_rows)
            for decision_index in range(len(request_rows))
        )
    )
    request_audit["event_order_valid"] = request_event_order_valid
    request_audit["valid"] = request_audit.get("valid") is True and request_event_order_valid
    summary_event_matches = bool(events) and events[-1].get("type") == "summary" and {
        key: value for key, value in events[-1].items() if key != "type"
    } == summary
    termination_class = str(summary.get("termination_class") or "process_infrastructure_error")
    formal_classes = PAPER_BEHAVIOR_TERMINATION_CLASSES
    natural_terminal = termination_class == "natural_game_over"
    outcome_marker_present = (
        summary.get("outcome") == "victory" and ("VICTORY!" in terminal or "胜利!" in terminal)
    ) or (
        summary.get("outcome") == "defeat" and ("DEFEAT" in terminal or "战败" in terminal)
    )
    try:
        engine_audit = audit_engine_trace(engine_trace_path, config=config, cli_summary=summary)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        engine_audit = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    expected_analysis_manifest: dict[str, dict[str, Any]] = {}
    expected_analysis_error: str | None = None
    try:
        expected_benchmark_rows = convert_engine_trace(
            engine_trace_path,
            config=config,
            cli_summary=summary,
        )
        benchmark_audit = audit_benchmark_trace(
            benchmark_trace_path,
            engine_audit=engine_audit,
            expected_rows=expected_benchmark_rows,
        )
        with tempfile.TemporaryDirectory(prefix="sts2-direct-report-audit-") as temporary:
            temporary_root = Path(temporary)
            expected_analysis_dir = temporary_root / "trace_analysis"
            # The trace itself is already checked byte-for-byte against the
            # deterministic engine conversion above.  Reuse its real path so
            # human-facing report source labels are reproduced exactly.
            analyze_trace(benchmark_trace_path, expected_analysis_dir)
            expected_analysis_manifest = build_analysis_artifact_manifest(expected_analysis_dir)
    except Exception as exc:  # noqa: BLE001 - validation must report renderer failures as evidence
        benchmark_audit = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
        expected_analysis_error = f"{type(exc).__name__}: {exc}"
    recorded_analysis = summary.get("analysis") if isinstance(summary.get("analysis"), dict) else {}
    analysis_dir = artifacts / "trace_analysis"
    required_analysis_files = tuple(analysis_dir / name for name in DIRECT_CLI_ANALYSIS_FILES)
    recorded_analysis_manifest = recorded_analysis.get("analysis_artifact_manifest")
    current_analysis_manifest = build_analysis_artifact_manifest(analysis_dir)
    analysis_checks = {
        "benchmark_trace_present": benchmark_trace_path.is_file(),
        "benchmark_trace_authoritative_closure": benchmark_audit.get("valid") is True,
        "recorded_trace_audits_match": recorded_analysis.get("engine_audit") == engine_audit
        and recorded_analysis.get("benchmark_audit") == benchmark_audit,
        "recorded_trace_hashes_match": recorded_analysis.get("engine_trace_sha256")
        == engine_audit.get("engine_trace_sha256")
        and recorded_analysis.get("benchmark_trace_sha256") == benchmark_audit.get("benchmark_trace_sha256"),
        "analysis_artifacts_complete": all(path.is_file() for path in required_analysis_files),
        "analysis_artifact_manifest_complete": isinstance(recorded_analysis_manifest, dict)
        and set(recorded_analysis_manifest) == set(DIRECT_CLI_ANALYSIS_FILES),
        "analysis_artifact_manifest_matches": recorded_analysis_manifest == current_analysis_manifest,
        "analysis_artifacts_match_deterministic_regeneration": expected_analysis_error is None
        and current_analysis_manifest == expected_analysis_manifest,
        "analysis_index_valid": bool(analysis_index),
        "summary_terminal_state_matches_audit": summary.get("authoritative_terminal_state")
        == engine_audit.get("authoritative_terminal_state")
        and summary.get("authoritative_terminal_state_sha256")
        == engine_audit.get("authoritative_terminal_state_sha256"),
        "summary_terminal_deck_matches_audit": summary.get("authoritative_terminal_deck")
        == engine_audit.get("authoritative_terminal_deck")
        and summary.get("authoritative_terminal_deck_sha256")
        == engine_audit.get("authoritative_terminal_deck_sha256"),
    }
    analysis_valid = all(analysis_checks.values())
    analysis_pending = summary.get("analysis_pending") is True
    summary_outcome_valid = {
        "natural_game_over": summary.get("completed") is True
        and summary.get("outcome") in {"victory", "defeat"}
        and summary.get("error") is None
        and summary.get("decision_limit_reached") is False,
        "model_forfeit": summary.get("completed") is False
        and summary.get("outcome") == "forfeit"
        and summary.get("error") is None
        and summary.get("decision_limit_reached") is False,
        "decision_budget_exhausted": summary.get("completed") is False
        and summary.get("outcome") == "decision_budget_exhausted"
        and summary.get("decision_limit_reached") is True,
        "model_protocol_error": summary.get("completed") is False
        and summary.get("outcome") == "model_protocol_error"
        and isinstance(summary.get("error"), str),
    }.get(termination_class, False)
    metrics_valid = (
        isinstance(summary.get("max_visible_floor"), int)
        and not isinstance(summary.get("max_visible_floor"), bool)
        and summary.get("max_visible_floor") == engine_audit.get("cumulative_observed_floor")
        and isinstance(summary.get("invalid_cli_inputs"), int)
        and not isinstance(summary.get("invalid_cli_inputs"), bool)
        and isinstance(summary.get("model_output_format_violations"), int)
        and not isinstance(summary.get("model_output_format_violations"), bool)
        and isinstance(summary.get("usage"), dict)
        and isinstance(summary.get("model_latency_sec"), (int, float))
        and not isinstance(summary.get("model_latency_sec"), bool)
        and math.isfinite(float(summary.get("model_latency_sec")))
        and summary.get("model_latency_sec") >= 0
        and isinstance(summary.get("elapsed_sec"), (int, float))
        and not isinstance(summary.get("elapsed_sec"), bool)
        and summary.get("elapsed_sec") >= 0
        and summary.get("usage") == request_audit.get("usage_totals")
        and summary.get("model_latency_sec") == request_audit.get("model_latency_sec")
        and summary.get("model_output_format_violations")
        == request_audit.get("model_output_format_violations")
        and summary.get("invalid_cli_inputs") == count_invalid_cli_inputs(terminal)
    )
    checks = {
        "returncode_zero": returncode == 0,
        "attempt_manifest_complete": attempt_manifest.get("formal_sample") is True
        and attempt_manifest.get("experiment_id") == protocol.get("experiment_id")
        and attempt_manifest.get("seed") == expected_seed
        and attempt_manifest.get("replicate") == expected_replicate
        and attempt_manifest.get("source_lock_sha256") == source_lock
        and attempt_manifest.get("source_tree_identity_sha256") == source_tree_identity
        and attempt_manifest.get("protocol_sha256") == expected_protocol_sha
        and attempt_manifest.get("game_rules") == protocol.get("game_rules")
        and attempt_manifest.get("execution_benchmark_root") == str(execution_benchmark_root.resolve())
        and attempt_manifest.get("execution_sts2_cli_root") == str(execution_sts2_cli_root.resolve()),
        "command_targets_attempt_artifacts": command_artifacts == artifacts.resolve(),
        "required_core_artifacts_present": all(
            path.is_file()
            for path in (
                config_path,
                summary_path,
                events_path,
                terminal_path,
                engine_trace_path,
            )
        ),
        "terminal_utf8_well_formed": terminal_well_formed,
        "model_visible_localization_complete": localization_audit["valid"] is True,
        "engine_trace_authoritative_closure": engine_audit.get("valid") is True,
        # Missing report artifacts may be rebuilt offline without replaying a
        # valid behavior sample.  A report that already exists but contradicts
        # the engine trace is evidence corruption and must fail closed.
        "present_benchmark_trace_authoritative_closure": not benchmark_trace_path.is_file()
        or benchmark_audit.get("valid") is True,
        "formal_termination_class": termination_class in formal_classes
        and summary.get("formal_behavior_sample") is True,
        "summary_termination_matches": summary_outcome_valid,
        "paper_metrics_complete": metrics_valid,
        "terminal_outcome_matches_summary": bool(outcome_marker_present) if natural_terminal else True,
        "config_matches_protocol": config_values_match,
        "system_prompt_matches_locked_implementation": config.get("system_prompt") == SYSTEM_PROMPT
        and config.get("system_prompt_sha256") == expected_prompt_sha,
        "config_game_rules_match_protocol": config.get("game_rules") == expected_rules,
        "controller_identity_matches": config.get("controller_path") == str(expected_controller.resolve())
        and config.get("controller_sha256") == hash_file(expected_controller),
        "human_cli_entrypoint_identity_matches": expected_human_cli.is_file()
        and config.get("human_cli_path") == str(expected_human_cli.resolve())
        and config.get("human_cli_sha256") == hash_file(expected_human_cli),
        "dotnet_runtime_identity_matches": expected_dotnet.is_file()
        and config.get("dotnet_path") == str(expected_dotnet.resolve())
        and config.get("dotnet_sha256") == hash_file(expected_dotnet),
        "execution_closure_identity_matches": config.get("source_lock_sha256") == source_lock
        and config.get("source_tree_identity_sha256") == source_tree_identity,
        "terminal_decision_count_matches": isinstance(summary.get("decisions"), int)
        and summary.get("decisions") == request_audit.get("decision_count"),
        "terminal_decision_requests_reconstructed": request_audit.get("valid") is True,
        "terminal_decision_rule_exposure_complete": request_audit.get("valid") is True,
        "events_terminate_with_matching_summary": summary_event_matches,
        "analysis_status_truthful": (
            analysis_valid
            and analysis_pending is False
            and summary.get("analysis_error") is None
            and summary.get("needs_attention") is False
        )
        or (
            not analysis_valid
            and analysis_pending is True
            and isinstance(summary.get("analysis_error"), str)
            and summary.get("needs_attention") is True
        ),
    }
    valid = all(checks.values())
    retryable = not valid and termination_class in {
        "provider_configuration_error",
        "provider_transport_error",
        "process_infrastructure_error",
    }
    return {
        "valid": valid,
        "formal_behavior_sample": valid,
        "termination_class": termination_class,
        "localization_audit": localization_audit,
        "retryable": retryable,
        "needs_attention": valid and not analysis_valid,
        "analysis_valid": analysis_valid,
        "analysis_checks": analysis_checks,
        "expected_analysis_artifact_manifest": expected_analysis_manifest,
        "expected_analysis_error": expected_analysis_error,
        "checks": checks,
        "engine_audit": engine_audit,
        "benchmark_audit": benchmark_audit,
        "request_audit": request_audit,
        "attempt_manifest": str(manifest_path),
        "artifacts": str(artifacts),
        "summary": summary,
    }


def _attempt_directories(job_dir: Path) -> list[Path]:
    attempts_root = job_dir / "attempts"
    if not attempts_root.is_dir():
        return []
    return sorted(
        (path for path in attempts_root.glob("attempt_*") if path.is_dir()),
        key=lambda path: path.name,
    )


def _report_issue(
    *,
    protocol: dict[str, Any],
    seed: str,
    replicate: int,
    attempt: int | None,
    validation: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    checked = validation or {}
    engine_audit = checked.get("engine_audit") if isinstance(checked.get("engine_audit"), dict) else {}
    return {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "seed": seed,
        "replicate": replicate,
        "attempt": attempt,
        "reason": reason
        or (
            "engine_error_or_unsupported_decision"
            if engine_audit.get("state_errors") or engine_audit.get("unsupported_decisions")
            else "non_retryable_behavior_artifact_failure"
        ),
        "state_errors": engine_audit.get("state_errors") or [],
        "unsupported_decisions": engine_audit.get("unsupported_decisions") or [],
        "engine_trace_sha256": engine_audit.get("engine_trace_sha256"),
        "created_at": _utc_now(),
    }


def _recover_job(
    job_dir: Path,
    source_lock: str,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    execution_benchmark_root: Path,
    execution_sts2_cli_root: Path,
    source_tree_identity: str,
    expected_seed: str,
    expected_replicate: int,
) -> dict[str, Any]:
    try:
        active = _active_attempt(job_dir)
    except DirectCliExperimentError as exc:
        return {
            "status": "failed",
            "formal_sample": False,
            "retryable": False,
            "control_error": str(exc),
            "report_required": _report_issue(
                protocol=protocol,
                seed=expected_seed,
                replicate=expected_replicate,
                attempt=None,
                reason="corrupt_active_attempt_lease",
            ),
            "attempt_history": [],
        }
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
        sticky = _report_issue(
            protocol=protocol,
            seed=expected_seed,
            replicate=expected_replicate,
            attempt=None,
            reason="corrupt_job_report_required",
        )
        sticky["control_error"] = sticky_error

    histories: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    recovery_report: dict[str, Any] | None = sticky
    for attempt_dir in _attempt_directories(job_dir):
        try:
            attempt_index = int(attempt_dir.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            attempt_index = -1
        if attempt_index < 1 or attempt_dir.name != f"attempt_{attempt_index:03d}":
            issue = _report_issue(
                protocol=protocol,
                seed=expected_seed,
                replicate=expected_replicate,
                attempt=None,
                reason="malformed_attempt_directory",
            )
            issue["attempt_dir"] = str(attempt_dir)
            histories.append(
                {
                    "attempt": attempt_index,
                    "attempt_dir": str(attempt_dir),
                    "status": "failed",
                    "formal_sample": False,
                    "retryable": False,
                    "report_required": issue,
                }
            )
            recovery_report = recovery_report or issue
            continue
        attempt_result_path = attempt_dir / "attempt_result.json"
        attempt_result_missing = not attempt_result_path.is_file()
        stored, stored_error = _try_read_json(attempt_result_path)
        attempt_manifest, _manifest_error = _try_read_json(attempt_dir / "attempt_manifest.json")
        raw_summary, _summary_error = _try_read_json(attempt_dir / "artifacts" / "summary.json")
        stored_returncode = stored.get("returncode") if stored else None
        if isinstance(stored_returncode, int) and not isinstance(stored_returncode, bool):
            returncode = stored_returncode
        else:
            returncode = 0 if (raw_summary or {}).get("formal_behavior_sample") is True else 1
        fresh = _validate_attempt_artifacts(
            attempt_dir,
            protocol=protocol,
            protocol_path=protocol_path,
            execution_benchmark_root=execution_benchmark_root,
            execution_sts2_cli_root=execution_sts2_cli_root,
            source_lock=source_lock,
            source_tree_identity=source_tree_identity,
            expected_seed=expected_seed,
            expected_replicate=expected_replicate,
            returncode=returncode,
        )
        rules_exposure_valid = bool(
            fresh.get("checks", {}).get("config_game_rules_match_protocol")
            and fresh.get("checks", {}).get("terminal_decision_rule_exposure_complete")
        )
        identity_stable = (
            True
            if not stored
            else stored.get("identity_stable") is True
            and stored.get("source_tree_stable") is True
        )
        current_summary_path = attempt_dir / "artifacts" / "summary.json"
        current_summary_sha256 = hash_file(current_summary_path) if current_summary_path.is_file() else None
        current_analysis_manifest = (
            (fresh.get("summary") or {}).get("analysis", {}).get("analysis_artifact_manifest")
            if isinstance((fresh.get("summary") or {}).get("analysis"), dict)
            else None
        )
        accepted_anchor_valid = True
        if stored and (
            stored.get("status") == "succeeded"
            or isinstance(stored.get("prior_accepted_attempt_result"), str)
        ):
            accepted_anchor_valid = (
                stored.get("accepted_summary_sha256") == current_summary_sha256
                and stored.get("accepted_analysis_artifact_manifest") == current_analysis_manifest
            )
        fresh.setdefault("checks", {})["accepted_artifact_anchor_matches"] = accepted_anchor_valid
        fresh["accepted_artifact_anchor_matches"] = accepted_anchor_valid
        fresh["valid"] = fresh.get("valid") is True and accepted_anchor_valid
        if not accepted_anchor_valid:
            fresh["retryable"] = False
        valid = fresh.get("valid") is True and identity_stable
        retryable = fresh.get("retryable") is True or bool(
            stored and stored.get("retryable_superseded_attempt") is True
        )
        completed_behavior_recorded = bool(
            (stored and stored.get("status") == "succeeded")
            or (raw_summary and raw_summary.get("formal_behavior_sample") is True)
        )
        stale_completed_evidence = completed_behavior_recorded and not valid
        if stale_completed_evidence:
            # Never replace a previously accepted behavior sample because a
            # post-run artifact later drifted.  Replaying would silently change
            # the sample; preserve evidence and stop for explicit review.
            retryable = False
        record = {
            **(stored or {}),
            **(attempt_manifest or {}),
            "attempt": attempt_index,
            "attempt_dir": str(attempt_dir),
            "stored_attempt_result_error": stored_error,
            "status": "succeeded" if valid else "failed",
            "formal_sample": valid,
            "returncode": returncode,
            "identity_stable": identity_stable,
            "source_tree_stable": identity_stable,
            "rules_exposure_valid": rules_exposure_valid,
            "source_lock_sha256": source_lock,
            "source_tree_identity_sha256": source_tree_identity,
            "protocol_sha256": hash_file(protocol_path),
            "game_rules": protocol.get("game_rules"),
            "experiment_id": protocol.get("experiment_id"),
            "seed": expected_seed,
            "replicate": expected_replicate,
            "termination_class": fresh.get("termination_class"),
            "retryable": retryable,
            "retryable_superseded_attempt": bool(
                not stale_completed_evidence
                and stored
                and stored.get("retryable_superseded_attempt") is True
            ),
            "analysis_pending": fresh.get("needs_attention") is True,
            "needs_attention": fresh.get("needs_attention") is True,
            "validation": fresh,
            "artifacts": str(attempt_dir / "artifacts"),
            "summary": fresh.get("summary") or {},
            "accepted_artifact_anchor_matches": accepted_anchor_valid,
            "accepted_summary_sha256": (
                stored.get("accepted_summary_sha256")
                if stored and stored.get("accepted_summary_sha256") is not None
                else current_summary_sha256
            ),
            "accepted_analysis_artifact_manifest": (
                stored.get("accepted_analysis_artifact_manifest")
                if stored and stored.get("accepted_analysis_artifact_manifest") is not None
                else current_analysis_manifest
            ),
        }
        if stored and stored.get("status") == "succeeded" and not valid and attempt_result_path.is_file():
            prior_sha256 = hash_file(attempt_result_path)
            prior_path = attempt_dir / f"attempt_result.accepted.{prior_sha256}.json"
            if not prior_path.exists():
                shutil.copy2(attempt_result_path, prior_path)
            record["prior_accepted_attempt_result"] = str(prior_path)
            record["prior_accepted_attempt_result_sha256"] = prior_sha256
        if attempt_result_missing or stored_error:
            if stored_error and attempt_result_path.is_file():
                backup_path = attempt_dir / f"attempt_result.corrupt.{time.time_ns()}.json"
                os.replace(attempt_result_path, backup_path)
                record["corrupt_attempt_result_backup"] = str(backup_path)
            record["recovered_at"] = _utc_now()
            record["recovery_source"] = "authoritative_child_artifacts"
        _atomic_write_json(attempt_result_path, record)
        histories.append(record)
        if valid:
            accepted.append(record)
        elif not retryable and recovery_report is None:
            recovery_report = _report_issue(
                protocol=protocol,
                seed=expected_seed,
                replicate=expected_replicate,
                attempt=attempt_index,
                validation=fresh,
            )

    if len(accepted) > 1 and recovery_report is None:
        recovery_report = _report_issue(
            protocol=protocol,
            seed=expected_seed,
            replicate=expected_replicate,
            attempt=None,
            reason="multiple_formal_samples_for_one_job",
        )
        recovery_report["attempts"] = [item["attempt"] for item in accepted]
    if recovery_report:
        if sticky_error is None:
            _atomic_write_json(sticky_path, recovery_report)
        status = {
            **(histories[-1] if histories else {}),
            "status": "failed",
            "formal_sample": False,
            "retryable": False,
            "seed": expected_seed,
            "replicate": expected_replicate,
            "report_required": recovery_report,
            "attempt_history": histories,
            "resume_action": "blocked_report_required",
        }
    elif accepted:
        status = {
            **accepted[0],
            "status": "succeeded",
            "formal_sample": True,
            "attempt_history": histories,
            "resume_action": "recovered_from_attempt_result",
        }
    else:
        retryable = not histories or histories[-1].get("retryable") is True
        status = {
            "status": "pending" if not histories else "failed",
            "formal_sample": False,
            "retryable": retryable,
            "seed": expected_seed,
            "replicate": expected_replicate,
            "attempt_history": histories,
        }
    if status.get("status") != "pending":
        _atomic_write_json(job_dir / "job_status.json", status)
    return status


def _next_attempt_index(job_dir: Path) -> int:
    attempts_root = job_dir / "attempts"
    indexes = []
    if attempts_root.is_dir():
        for path in attempts_root.glob("attempt_*"):
            try:
                indexes.append(int(path.name.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max(indexes, default=0) + 1


def _experiment_hold(out_root: Path) -> dict[str, Any] | None:
    sticky, error = _try_read_json(out_root / "report_required.json")
    if error:
        return {
            "schema_version": 1,
            "reason": "corrupt_experiment_report_required",
            "control_error": error,
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


def _run_job(
    protocol: dict[str, Any],
    *,
    protocol_path: Path,
    out_root: Path,
    source_lock_record: dict[str, Any],
    source_lock: str,
    source_tree_identity: str,
    seed: str,
    replicate: int,
    runner: Callable[..., int],
    control_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    job_dir = out_root / "runs" / seed / f"replicate_{replicate:02d}"
    job_status_existed = (job_dir / "job_status.json").is_file()
    execution_benchmark_root = Path(str(source_lock_record["benchmark_root"])).resolve()
    execution_sts2_cli_root = Path(str(source_lock_record["sts2_cli_root"])).resolve()
    report_required_path = job_dir / "report_required.json"
    recovered = _recover_job(
        job_dir,
        source_lock,
        protocol=protocol,
        protocol_path=protocol_path,
        execution_benchmark_root=execution_benchmark_root,
        execution_sts2_cli_root=execution_sts2_cli_root,
        source_tree_identity=source_tree_identity,
        expected_seed=seed,
        expected_replicate=replicate,
    )
    if recovered.get("formal_sample") is True:
        recovered_status = {
            **recovered,
            "resume_action": (
                "skipped_completed"
                if job_status_existed
                else str(recovered.get("resume_action") or "recovered_from_attempt_result")
            ),
        }
        _atomic_write_json(job_dir / "job_status.json", recovered_status)
        return recovered_status
    if recovered.get("report_required"):
        report = _publish_experiment_hold(
            out_root,
            dict(recovered["report_required"]),
            control_lock=control_lock,
        )
        return {**recovered, "report_required": report}
    if recovered.get("status") == "active_attempt_detected":
        return recovered
    launch_lock = control_lock or threading.Lock()
    with launch_lock:
        hold = _experiment_hold(out_root)
    if hold is not None:
        return {
            **recovered,
            "status": "failed",
            "formal_sample": False,
            "retryable": False,
            "seed": seed,
            "replicate": replicate,
            "report_required": hold,
            "experiment_hold": hold,
            "resume_action": "blocked_experiment_report_required",
        }
    if recovered.get("retryable") is not True:
        return recovered

    max_attempts = int(protocol.get("max_attempts_per_invocation", 1))
    timeout_sec = float(protocol.get("run_timeout_sec", 0))
    last_status: dict[str, Any] | None = None
    attempt_history = list(recovered.get("attempt_history") or [])
    for _ in range(max_attempts):
        _revalidate_source_lock(
            out_root,
            expected_identity=source_lock,
            expected_source_identity=source_tree_identity,
        )
        with launch_lock:
            hold = _experiment_hold(out_root)
            if hold is not None:
                return {
                    "status": "failed",
                    "formal_sample": False,
                    "retryable": False,
                    "seed": seed,
                    "replicate": replicate,
                    "report_required": hold,
                    "experiment_hold": hold,
                    "attempt_history": attempt_history,
                    "resume_action": "blocked_experiment_report_required",
                }
            if _active_attempt(job_dir) is not None:
                raise DirectCliExperimentError(
                    f"Refusing duplicate active attempt for {seed} replicate {replicate}"
                )
            attempt_index = _next_attempt_index(job_dir)
            attempt_dir = job_dir / "attempts" / f"attempt_{attempt_index:03d}"
            artifacts = attempt_dir / "artifacts"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            command = _attempt_command(
                protocol,
                seed=seed,
                attempt_artifacts=artifacts,
                sts2_cli_root=execution_sts2_cli_root,
                source_lock=source_lock,
                source_tree_identity=source_tree_identity,
            )
            started_at = _utc_now()
            attempt_manifest = {
                "schema_version": 1,
                "experiment_id": protocol["experiment_id"],
                "seed": seed,
                "replicate": replicate,
                "attempt": attempt_index,
                "formal_sample": True,
                "source_lock_sha256": source_lock,
                "source_tree_identity_sha256": source_tree_identity,
                "protocol_sha256": hash_file(protocol_path),
                "game_rules": protocol["game_rules"],
                "execution_benchmark_root": str(execution_benchmark_root.resolve()),
                "execution_sts2_cli_root": str(execution_sts2_cli_root.resolve()),
                "command": command,
                "started_at": started_at,
            }
            _atomic_write_json(attempt_dir / "attempt_manifest.json", attempt_manifest)
            lease_path = job_dir / "active_attempt.json"
            _atomic_write_json(
                lease_path,
                {
                    "schema_version": 1,
                    "experiment_id": protocol["experiment_id"],
                    "seed": seed,
                    "replicate": replicate,
                    "attempt": attempt_index,
                    "owner_pid": os.getpid(),
                    "child_pid": None,
                    "state": "starting",
                    "started_at": started_at,
                },
            )
        runner_error = None
        try:
            returncode = runner(
                command,
                stdout_path=attempt_dir / "stdout.log",
                stderr_path=attempt_dir / "stderr.log",
                cwd=execution_benchmark_root,
                timeout_sec=timeout_sec,
                lease_path=lease_path,
            )
        except Exception as exc:  # noqa: BLE001 - every attempt must close atomically
            returncode = 125
            runner_error = f"{type(exc).__name__}: {exc}"
            (attempt_dir / "stderr.log").write_text(runner_error + "\n", encoding="utf-8")
        finally:
            _release_attempt_lease(lease_path)
        identity_error = None
        try:
            _revalidate_source_lock(
                out_root,
                expected_identity=source_lock,
                expected_source_identity=source_tree_identity,
            )
            identity_stable = True
            source_tree_stable = True
        except DirectCliExperimentError as exc:
            identity_stable = False
            source_tree_stable = False
            identity_error = str(exc)
        validation = _validate_attempt_artifacts(
            attempt_dir,
            protocol=protocol,
            protocol_path=protocol_path,
            execution_benchmark_root=execution_benchmark_root,
            execution_sts2_cli_root=execution_sts2_cli_root,
            source_lock=source_lock,
            source_tree_identity=source_tree_identity,
            expected_seed=seed,
            expected_replicate=replicate,
            returncode=returncode,
        )
        if runner_error:
            validation["runner_error"] = runner_error
            validation["retryable"] = True
        summary = validation.get("summary") if isinstance(validation.get("summary"), dict) else {}
        engine_audit = validation.get("engine_audit") if isinstance(validation.get("engine_audit"), dict) else {}
        validation_valid = validation.get("valid") is True
        retryable = validation.get("retryable") is True
        requires_manual_report = bool(
            not validation_valid
            and (
                engine_audit.get("state_errors")
                or engine_audit.get("unsupported_decisions")
                or not retryable
                or not identity_stable
            )
        )
        rules_exposure_valid = bool(
            validation.get("checks", {}).get("config_game_rules_match_protocol")
            and validation.get("checks", {}).get("terminal_decision_rule_exposure_complete")
        )
        succeeded = bool(identity_stable and source_tree_stable and validation_valid)
        last_status = {
            **attempt_manifest,
            "status": "succeeded" if succeeded else "failed",
            "formal_sample": succeeded,
            "returncode": returncode,
            "identity_stable": identity_stable,
            "source_tree_stable": source_tree_stable,
            "identity_error": identity_error,
            "rules_exposure_valid": rules_exposure_valid,
            "report_required": requires_manual_report,
            "termination_class": validation.get("termination_class"),
            "retryable": retryable,
            "analysis_pending": validation.get("needs_attention") is True,
            "needs_attention": validation.get("needs_attention") is True,
            "protocol_sha256": hash_file(protocol_path),
            "validation": validation,
            "artifacts": str(artifacts),
            "summary": summary,
            "accepted_summary_sha256": hash_file(artifacts / "summary.json")
            if (artifacts / "summary.json").is_file()
            else None,
            "accepted_analysis_artifact_manifest": (
                (summary.get("analysis") or {}).get("analysis_artifact_manifest")
                if isinstance(summary.get("analysis"), dict)
                else None
            ),
            "finished_at": _utc_now(),
            "resume_action": "executed",
        }
        _atomic_write_json(attempt_dir / "attempt_result.json", last_status)
        attempt_history.append(last_status)
        job_status = {**last_status, "attempt_history": attempt_history}
        _atomic_write_json(job_dir / "job_status.json", job_status)
        if requires_manual_report:
            report = _report_issue(
                protocol=protocol,
                seed=seed,
                replicate=replicate,
                attempt=attempt_index,
                validation=validation,
                reason="source_lock_drift" if not identity_stable else None,
            )
            _atomic_write_json(report_required_path, report)
            report = _publish_experiment_hold(out_root, report, control_lock=control_lock)
            return {**job_status, "report_required": report}
        if succeeded:
            return job_status
        if not retryable:
            return job_status
    assert last_status is not None
    return {**last_status, "attempt_history": attempt_history}


def _lane_status_path(out_root: Path, seed: str) -> Path:
    return out_root / "lanes" / seed / "lane_status.json"


def _run_seed_lane(
    protocol: dict[str, Any],
    *,
    protocol_path: Path,
    out_root: Path,
    source_lock_record: dict[str, Any],
    source_lock: str,
    source_tree_identity: str,
    seed: str,
    runner: Callable[..., int],
    progress: Callable[[], None],
    control_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    repeats = int(protocol["repeats_per_seed"])
    experiment_hold: dict[str, Any] | None = None
    for replicate in range(1, repeats + 1):
        lock = control_lock or threading.Lock()
        with lock:
            experiment_hold = _experiment_hold(out_root)
        if experiment_hold is not None:
            break
        result = _run_job(
            protocol,
            protocol_path=protocol_path,
            out_root=out_root,
            source_lock_record=source_lock_record,
            source_lock=source_lock,
            source_tree_identity=source_tree_identity,
            seed=seed,
            replicate=replicate,
            runner=runner,
            control_lock=control_lock,
        )
        jobs.append(result)
        _atomic_write_json(
            _lane_status_path(out_root, seed),
            {
                "status": "running",
                "seed": seed,
                "completed_replicates": sum(item.get("status") == "succeeded" for item in jobs),
                "processed_replicates": len(jobs),
                "expected_replicates": repeats,
                "jobs": jobs,
                "updated_at": _utc_now(),
            },
        )
        progress()
        if result.get("report_required"):
            experiment_hold = dict(result["report_required"])
            break
        if result.get("formal_sample") is not True:
            break
    completed = sum(item.get("status") == "succeeded" for item in jobs)
    lane_status = {
        "status": "succeeded" if completed == repeats else "failed",
        "seed": seed,
        "completed_replicates": completed,
        "processed_replicates": len(jobs),
        "expected_replicates": repeats,
        "jobs": jobs,
        "experiment_hold": experiment_hold,
        "updated_at": _utc_now(),
    }
    _atomic_write_json(_lane_status_path(out_root, seed), lane_status)
    progress()
    return lane_status


def _collect_progress(out_root: Path, seeds: list[str], repeats: int) -> dict[str, Any]:
    lanes = []
    for seed in seeds:
        path = _lane_status_path(out_root, seed)
        if path.is_file():
            lanes.append(_read_progress_json(path))
        else:
            lanes.append(
                {
                    "status": "pending",
                    "seed": seed,
                    "completed_replicates": 0,
                    "processed_replicates": 0,
                    "expected_replicates": repeats,
                    "jobs": [],
                }
            )
    return {
        "status": "running",
        "expected_runs": len(seeds) * repeats,
        "completed_runs": sum(int(lane.get("completed_replicates") or 0) for lane in lanes),
        "processed_runs": sum(int(lane.get("processed_replicates") or 0) for lane in lanes),
        "needs_attention_runs": sum(
            item.get("needs_attention") is True
            for lane in lanes
            for item in (lane.get("jobs") or [])
            if isinstance(item, dict)
        ),
        "lanes": lanes,
        "updated_at": _utc_now(),
    }


def _revalidate_all_jobs(
    protocol: dict[str, Any],
    *,
    protocol_path: Path,
    out_root: Path,
    source_lock: str,
    source_tree_identity: str,
) -> list[dict[str, Any]]:
    frozen = _read_and_verify_source_lock(out_root)
    execution_benchmark_root = Path(str(frozen["benchmark_root"])).resolve()
    execution_sts2_cli_root = Path(str(frozen["sts2_cli_root"])).resolve()
    repeats = int(protocol["repeats_per_seed"])
    recovered_lanes: list[tuple[str, list[dict[str, Any]]]] = []
    for seed in protocol["seeds"]:
        jobs: list[dict[str, Any]] = []
        for replicate in range(1, repeats + 1):
            job = _recover_job(
                out_root / "runs" / seed / f"replicate_{replicate:02d}",
                source_lock,
                protocol=protocol,
                protocol_path=protocol_path,
                execution_benchmark_root=execution_benchmark_root,
                execution_sts2_cli_root=execution_sts2_cli_root,
                source_tree_identity=source_tree_identity,
                expected_seed=seed,
                expected_replicate=replicate,
            )
            jobs.append(job)
            if job.get("report_required"):
                _publish_experiment_hold(
                    out_root,
                    dict(job["report_required"]),
                    control_lock=None,
                )
        recovered_lanes.append((seed, jobs))

    experiment_hold = _experiment_hold(out_root)
    lanes: list[dict[str, Any]] = []
    for seed, jobs in recovered_lanes:
        if experiment_hold is not None:
            jobs = [
                job
                if job.get("formal_sample") is True or job.get("report_required")
                else {
                    **job,
                    "experiment_hold": experiment_hold,
                    "resume_action": "blocked_experiment_report_required",
                }
                for job in jobs
            ]
        completed = sum(job.get("formal_sample") is True for job in jobs)
        lanes.append(
            {
                "status": "succeeded" if completed == repeats and experiment_hold is None else "failed",
                "seed": seed,
                "completed_replicates": completed,
                "processed_replicates": sum(bool(job.get("attempt_history")) for job in jobs),
                "expected_replicates": repeats,
                "jobs": jobs,
                "experiment_hold": experiment_hold,
                "updated_at": _utc_now(),
            }
        )
    return lanes


def _resolve_experiment_paths(
    protocol_path: str | Path,
    *,
    out_root_override: str | Path | None,
) -> tuple[Path, dict[str, Any], Path, Path]:
    resolved_protocol = Path(protocol_path).resolve()
    source_protocol = load_protocol(resolved_protocol)
    out_root = (
        Path(out_root_override).resolve()
        if out_root_override is not None
        else (ROOT / str(source_protocol["output_root"])).resolve()
    )
    source_sts2_cli_root = Path(
        source_protocol.get("sts2_cli_root") or DEFAULT_STS2_CLI_ROOT
    ).resolve()
    return resolved_protocol, source_protocol, out_root, source_sts2_cli_root


def prepare_source_lock(
    protocol_path: str | Path,
    *,
    resume: bool,
    out_root_override: str | Path | None = None,
) -> dict[str, Any]:
    resolved_protocol, protocol, out_root, source_sts2_cli_root = _resolve_experiment_paths(
        protocol_path,
        out_root_override=out_root_override,
    )
    if out_root.exists() and any(out_root.iterdir()) and not resume:
        raise DirectCliExperimentError(f"Output root already exists; pass --resume: {out_root}")
    if out_root.exists() and any(out_root.iterdir()) and not (out_root / "source_lock.json").is_file():
        raise DirectCliExperimentError(
            f"Non-empty output root has no source lock and cannot be resumed safely: {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    with ArtifactLease(
        out_root / ".experiment.lease",
        label=f"direct CLI source-lock preparation {protocol['experiment_id']}",
    ):
        source_lock_record = initialize_or_verify_source_lock(
            out_root,
            protocol_path=resolved_protocol,
            sts2_cli_root=source_sts2_cli_root,
        )
    return {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "status": "prepared",
        "source_lock_sha256": source_lock_record["identity_sha256"],
        "source_tree_identity_sha256": source_lock_record["source_tree_identity_sha256"],
        "execution_mode": "active_worktree",
        "benchmark_root": source_lock_record["benchmark_root"],
        "sts2_cli_root": source_lock_record["sts2_cli_root"],
        "output_root": str(out_root),
        "expected_runs": len(protocol["seeds"]) * int(protocol["repeats_per_seed"]),
    }


def _run_source_locked_experiment_core(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    out_root: Path,
    source_lock_record: dict[str, Any],
    runner: Callable[..., int],
) -> dict[str, Any]:
    seeds = list(protocol["seeds"])
    repeats = int(protocol["repeats_per_seed"])
    source_lock = str(source_lock_record["identity_sha256"])
    source_tree_identity = str(source_lock_record["source_tree_identity_sha256"])
    progress_lock = threading.Lock()
    control_lock = threading.Lock()
    execution_errors: list[dict[str, Any]] = []

    def record_execution_error(*, phase: str, exc: Exception, seed: str | None = None) -> dict[str, Any]:
        error = {
            "phase": phase,
            "seed": seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "created_at": _utc_now(),
        }
        execution_errors.append(error)
        report = {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "seed": seed,
            "replicate": None,
            "attempt": None,
            "reason": "lane_execution_exception" if phase == "seed_lane" else "orchestrator_execution_exception",
            "execution_error": error,
            "created_at": _utc_now(),
        }
        try:
            _publish_experiment_hold(out_root, report, control_lock=control_lock)
        except Exception as report_exc:
            error["report_persist_error"] = {
                "error_type": type(report_exc).__name__,
                "error": str(report_exc),
            }
        return error

    def write_progress() -> None:
        with progress_lock:
            _atomic_write_json(out_root / "experiment_status.json", _collect_progress(out_root, seeds, repeats))

    write_progress()
    lane_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        futures = {
            executor.submit(
                _run_seed_lane,
                protocol,
                protocol_path=protocol_path,
                out_root=out_root,
                source_lock_record=source_lock_record,
                source_lock=source_lock,
                source_tree_identity=source_tree_identity,
                seed=seed,
                runner=runner,
                progress=write_progress,
                control_lock=control_lock,
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                lane_results[seed] = future.result()
            except Exception as exc:
                error = record_execution_error(phase="seed_lane", exc=exc, seed=seed)
                lane_results[seed] = {"status": "failed", "seed": seed, "execution_error": error}
            try:
                write_progress()
            except Exception as exc:
                record_execution_error(phase="progress_write", exc=exc, seed=seed)

    try:
        ordered_lanes = _revalidate_all_jobs(
            protocol,
            protocol_path=protocol_path,
            out_root=out_root,
            source_lock=source_lock,
            source_tree_identity=source_tree_identity,
        )
    except Exception as exc:
        error = record_execution_error(phase="post_run_revalidation", exc=exc)
        ordered_lanes = [
            {
                "status": "failed",
                "seed": seed,
                "completed_replicates": 0,
                "processed_replicates": 0,
                "expected_replicates": repeats,
                "jobs": [],
                "execution_error": error,
                "updated_at": _utc_now(),
            }
            for seed in seeds
        ]
    succeeded = sum(int(lane["completed_replicates"]) for lane in ordered_lanes)
    needs_attention = sum(
        item.get("formal_sample") is True and item.get("needs_attention") is True
        for lane in ordered_lanes
        for item in (lane.get("jobs") or [])
        if isinstance(item, dict)
    )
    analysis_complete = sum(
        item.get("formal_sample") is True
        and isinstance(item.get("validation"), dict)
        and item["validation"].get("analysis_valid") is True
        for lane in ordered_lanes
        for item in (lane.get("jobs") or [])
        if isinstance(item, dict)
    )
    all_attempts = [
        attempt
        for lane in ordered_lanes
        for job in (lane.get("jobs") or [])
        for attempt in (job.get("attempt_history") or [])
        if isinstance(attempt, dict)
    ]
    try:
        _revalidate_source_lock(
            out_root,
            expected_identity=source_lock,
            expected_source_identity=source_tree_identity,
        )
    except Exception as exc:
        record_execution_error(phase="final_source_lock_revalidation", exc=exc)
    try:
        experiment_hold = _experiment_hold(out_root)
    except Exception as exc:
        error = record_execution_error(phase="experiment_hold_read", exc=exc)
        experiment_hold = {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "reason": "unreadable_experiment_report_required",
            "execution_error": error,
            "created_at": _utc_now(),
        }
    issues_requiring_report: list[dict[str, Any]] = []
    seen_issues: set[str] = set()
    for lane in ordered_lanes:
        for job in lane.get("jobs") or []:
            issue = job.get("report_required") if isinstance(job, dict) else None
            if not isinstance(issue, dict):
                continue
            key = json.dumps(issue, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in seen_issues:
                seen_issues.add(key)
                issues_requiring_report.append(issue)
    if isinstance(experiment_hold, dict):
        key = json.dumps(experiment_hold, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in seen_issues:
            issues_requiring_report.append(experiment_hold)
    final = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "status": (
            "succeeded"
            if succeeded == len(seeds) * repeats and experiment_hold is None and not execution_errors
            else "failed"
        ),
        "source_lock_sha256": source_lock,
        "source_tree_identity_sha256": source_tree_identity,
        "execution_mode": "active_worktree",
        "game_rules": protocol["game_rules"],
        "expected_runs": len(seeds) * repeats,
        "completed_runs": succeeded,
        "analysis_complete_runs": analysis_complete,
        "needs_attention_runs": needs_attention,
        "experiment_hold": experiment_hold,
        "issues_requiring_report": issues_requiring_report,
        "execution_errors": execution_errors,
        "all_attempts": all_attempts,
        "workers": len(seeds),
        "lane_semantics": "one_serial_lane_per_seed",
        "lanes": ordered_lanes,
        "updated_at": _utc_now(),
    }
    _atomic_write_json(out_root / "experiment_summary.json", final)
    _atomic_write_json(out_root / "experiment_status.json", final)
    return final


def run_experiment(
    protocol_path: str | Path,
    *,
    resume: bool,
    env_file: str | Path | None,
    runner: Callable[..., int] = _subprocess_runner,
    out_root_override: str | Path | None = None,
) -> dict[str, Any]:
    resolved_protocol, source_protocol, out_root, source_sts2_cli_root = _resolve_experiment_paths(
        protocol_path,
        out_root_override=out_root_override,
    )
    if out_root.exists() and any(out_root.iterdir()) and not resume:
        raise DirectCliExperimentError(f"Output root already exists; pass --resume: {out_root}")
    with ArtifactLease(
        out_root / ".experiment.lease",
        label=f"source-locked direct CLI experiment {source_protocol['experiment_id']}",
    ):
        source_lock_record = initialize_or_verify_source_lock(
            out_root,
            protocol_path=resolved_protocol,
            sts2_cli_root=source_sts2_cli_root,
        )
        if Path(str(source_lock_record["benchmark_root"])).resolve() != ROOT.resolve():
            raise DirectCliExperimentError("Source lock does not bind the active benchmark worktree")
        protocol = load_protocol(resolved_protocol)
        _load_and_validate_external_environment(
            env_file,
            base_url=str(protocol["base_url"]),
            forbidden_roots=(out_root,),
        )
        return _run_source_locked_experiment_core(
            protocol=protocol,
            protocol_path=resolved_protocol,
            out_root=out_root,
            source_lock_record=source_lock_record,
            runner=runner,
        )


def build_plan(
    protocol_path: str | Path,
    *,
    out_root_override: str | Path | None = None,
) -> dict[str, Any]:
    resolved_protocol = Path(protocol_path).resolve()
    protocol = load_protocol(resolved_protocol)
    seeds = list(protocol["seeds"])
    repeats = int(protocol["repeats_per_seed"])
    return {
        "experiment_id": protocol["experiment_id"],
        "protocol": str(resolved_protocol),
        "output_root": str(
            Path(out_root_override).resolve()
            if out_root_override is not None
            else (ROOT / str(protocol["output_root"])).resolve()
        ),
        "workers": len(seeds),
        "lane_semantics": "one_serial_lane_per_seed",
        "expected_runs": len(seeds) * repeats,
        "lanes": [{"seed": seed, "replicates": list(range(1, repeats + 1))} for seed in seeds],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the source-locked direct-CLI paper experiment.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--env-file")
    parser.add_argument("--out-root")
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            print(
                json.dumps(
                    build_plan(args.protocol, out_root_override=args.out_root),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        _, _, out_root, _ = _resolve_experiment_paths(
            args.protocol,
            out_root_override=args.out_root,
        )
        if args.lock_only:
            summary = prepare_source_lock(
                args.protocol,
                resume=bool(args.resume),
                out_root_override=args.out_root,
            )
        else:
            summary = run_experiment(
                args.protocol,
                resume=bool(args.resume),
                env_file=args.env_file,
                out_root_override=args.out_root,
            )
    except DirectCliExperimentError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"prepared", "succeeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
