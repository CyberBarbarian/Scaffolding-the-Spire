from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import sts2_bench.direct_cli_experiment as direct_cli_experiment

from sts2_bench.direct_cli_experiment import (
    DirectCliExperimentError,
    _audit_terminal_decision_requests,
    _load_and_validate_external_environment,
    _run_source_locked_experiment_core,
    build_plan,
    initialize_or_verify_source_lock,
    load_protocol,
    prepare_source_lock,
    run_experiment,
)
from sts2_bench.direct_cli_baseline import (
    OUTPUT_RETRY_MESSAGE,
    SYSTEM_PROMPT,
    ProviderConfigurationError,
    summarize_terminal_run,
)
from sts2_bench.direct_cli_report import (
    analyze_direct_cli_run,
    audit_benchmark_trace,
    build_analysis_artifact_manifest,
)
from sts2_bench.experiment_identity import hash_file
from sts2_bench.game_rules import game_rules_identity


def _write_protocol(
    tmp_path: Path,
    *,
    seeds: list[str] | None = None,
    repeats: int = 2,
    max_attempts: int = 1,
) -> Path:
    selected_seeds = seeds or ["seed_a", "seed_b", "seed_c"]
    cli_root = tmp_path / "sts2-cli"
    (cli_root / "python").mkdir(parents=True, exist_ok=True)
    (cli_root / "python" / "play.py").write_text("VALUE = 1\n", encoding="utf-8")
    (cli_root / "python" / "game_log.py").write_text("VALUE = 1\n", encoding="utf-8")
    for language in ("localization_eng", "localization_zhs"):
        localization = cli_root / language / "cards.json"
        localization.parent.mkdir(parents=True, exist_ok=True)
        localization.write_text('{"ok": true}\n', encoding="utf-8")
        (localization.parent / "_done.txt").write_text("complete\n", encoding="utf-8")
    headless_root = cli_root / "src" / "Sts2Headless"
    headless_root.mkdir(parents=True)
    (headless_root / "Sts2Headless.csproj").write_text("<Project />\n", encoding="utf-8")
    (headless_root / "RunSimulator.cs").write_text("// source\n", encoding="utf-8")
    stubs_root = cli_root / "src" / "GodotStubs"
    stubs_root.mkdir(parents=True)
    (stubs_root / "GodotStubs.csproj").write_text("<Project />\n", encoding="utf-8")
    (stubs_root / "Core.cs").write_text("// stubs source\n", encoding="utf-8")
    lib_dir = cli_root / "lib"
    lib_dir.mkdir()
    (lib_dir / "sts2.dll").write_bytes(b"test-game-runtime")
    runtime_dir = headless_root / "bin" / "Debug" / "net9.0"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "Sts2Headless.dll").write_bytes(b"test-headless-runtime")
    (runtime_dir / "GodotSharp.dll").write_bytes(b"test-godot-runtime")
    dotnet_root = cli_root / ".tools" / "dotnet"
    (dotnet_root / "host" / "fxr" / "test").mkdir(parents=True, exist_ok=True)
    (dotnet_root / "shared" / "Microsoft.NETCore.App" / "test").mkdir(parents=True, exist_ok=True)
    (dotnet_root / "dotnet.exe").write_bytes(b"test-dotnet-host")
    (dotnet_root / "host" / "fxr" / "test" / "hostfxr.dll").write_bytes(b"test-hostfxr")
    (dotnet_root / "shared" / "Microsoft.NETCore.App" / "test" / "coreclr.dll").write_bytes(
        b"test-coreclr"
    )
    protocol = {
        "schema_version": 2,
        "experiment_id": "test-direct-cli",
        "output_root": str(tmp_path / "results"),
        "sts2_cli_root": str(cli_root),
        "model": "fake-model",
        "base_url": "https://example.invalid/v1",
        "seeds": selected_seeds,
        "repeats_per_seed": repeats,
        "workers": len(selected_seeds),
        "prompt_contract_required": True,
        "max_attempts_per_invocation": max_attempts,
        "request_retries": 1,
        "game_rules": {
            **game_rules_identity(),
            "delivery": "static_full_system_prompt",
            "source": "sts2_bench/game_rules.py",
        },
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return path


def _argument(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _request_sha256(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> str:
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_prepared_experiment(protocol_path: Path, *, runner) -> dict:
    protocol = load_protocol(protocol_path)
    out_root = Path(protocol["output_root"]).resolve()
    if not (out_root / "source_lock.json").is_file():
        prepare_source_lock(protocol_path, resume=False)
    source_lock_record = json.loads(
        (out_root / "source_lock.json").read_text(encoding="utf-8")
    )
    return _run_source_locked_experiment_core(
        protocol=protocol,
        protocol_path=protocol_path,
        out_root=out_root,
        source_lock_record=source_lock_record,
        runner=runner,
    )


def _write_success(command: list[str], *, cwd: Path) -> None:
    artifacts = Path(_argument(command, "--out-dir"))
    artifacts.mkdir(parents=True, exist_ok=True)
    summary = {
        "completed": True,
        "outcome": "defeat",
        "decisions": 1,
        "max_visible_floor": 1,
        "invalid_cli_inputs": 1,
        "model_output_format_violations": 0,
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        "model_latency_sec": 0.01,
        "elapsed_sec": 0.02,
        "decision_limit_reached": False,
        "error": None,
        "termination_class": "natural_game_over",
        "formal_behavior_sample": True,
    }
    controller_path = cwd / "sts2_bench" / "direct_cli_baseline.py"
    cli_root = Path(_argument(command, "--sts2-cli-root"))
    human_cli_path = cli_root / "python" / "play.py"
    dotnet_path = cli_root / ".tools" / "dotnet" / "dotnet.exe"
    max_tokens = None if "--omit-max-tokens" in command else int(_argument(command, "--max-tokens"))
    config = {
        "protocol": "human_cli_raw_fifo_v1",
        "model": _argument(command, "--model"),
        "base_url": _argument(command, "--base-url"),
        "temperature": float(_argument(command, "--temperature")),
        "timeout": float(_argument(command, "--timeout")),
        "max_tokens": max_tokens,
        "request_retries": int(_argument(command, "--request-retries")),
        "thinking_control": "provider_default_omitted",
        "response_format_control": "provider_default_omitted",
        "controller_path": str(controller_path),
        "controller_sha256": hash_file(controller_path),
        "human_cli_path": str(human_cli_path.resolve()),
        "human_cli_sha256": hash_file(human_cli_path),
        "dotnet_path": str(dotnet_path.resolve()),
        "dotnet_sha256": hash_file(dotnet_path),
        "seed": _argument(command, "--seed"),
        "character": _argument(command, "--character"),
        "ascension": int(_argument(command, "--ascension")),
        "lang": _argument(command, "--lang"),
        "history_chars": int(_argument(command, "--history-chars")),
        "max_decisions": int(_argument(command, "--max-decisions")),
        "python_runtime": direct_cli_experiment._python_runtime_identity(),
        "source_locked_runtime_mode": True,
        "engine_telemetry_visible_to_model": False,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "game_rules": {
            **game_rules_identity(),
            "delivery": "static_full_system_prompt",
        },
        "source_lock_sha256": _argument(command, "--source-lock-sha256"),
        "source_tree_identity_sha256": _argument(command, "--source-tree-identity-sha256"),
    }
    (artifacts / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    visible_terminal = (
        "Floor: 1\n"
        "Invalid input. Follow the current prompt; type help for input syntax.\n"
        "Choose: "
    )
    terminal_text = visible_terminal + "0\nDEFEAT\n"
    (artifacts / "terminal.txt").write_text(terminal_text, encoding="utf-8")
    terminal_state = {
        "type": "decision",
        "decision": "game_over",
        "victory": False,
        "context": {"act": 1, "floor": 1, "room_type": "Boss"},
        "player": {
            "hp": 0,
            "max_hp": 80,
            "gold": 50,
            "deck_size": 1,
            "deck": [{"id": "CARD.STRIKE_IRONCLAD", "name": "Strike"}],
            "relics": [],
            "potions": [],
        },
    }
    engine_rows = [
        {"step": 1, "type": "state", "data": {"type": "ready"}},
        {
            "step": 1,
            "type": "action",
            "data": {
                "cmd": "start_run",
                "seed": config["seed"],
                "character": config["character"],
                "ascension": config["ascension"],
                "lang": config["lang"],
            },
        },
        {
            "step": 2,
            "type": "state",
            "data": {
                "type": "decision",
                "decision": "map_select",
                "choices": [{"col": 0, "row": 1, "type": "Boss"}],
                "player": terminal_state["player"],
            },
        },
        {
            "step": 2,
            "type": "action",
            "data": {"cmd": "action", "action": "select_map_node", "args": {"col": 0, "row": 1}},
        },
        {"step": 3, "type": "state", "data": terminal_state},
        {"step": 3, "type": "action", "data": {"cmd": "quit"}},
    ]
    engine_trace_path = artifacts / "engine_trace.jsonl"
    engine_trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in engine_rows),
        encoding="utf-8",
    )
    analysis = analyze_direct_cli_run(
        engine_trace_path,
        artifacts / "benchmark_trace.jsonl",
        artifacts / "trace_analysis",
        config=config,
        cli_summary=summary,
    )
    engine_audit = analysis["engine_audit"]
    summary.update(
        {
            "analysis": analysis,
            "analysis_error": None,
            "analysis_pending": False,
            "needs_attention": False,
            "authoritative_terminal_state": engine_audit["authoritative_terminal_state"],
            "authoritative_terminal_state_sha256": engine_audit["authoritative_terminal_state_sha256"],
            "authoritative_terminal_deck": engine_audit["authoritative_terminal_deck"],
            "authoritative_terminal_deck_sha256": engine_audit["authoritative_terminal_deck_sha256"],
        }
    )
    (artifacts / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    model_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + visible_terminal
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    provider_response = json.dumps(
        {
            "id": "request-1",
            "model": "resolved-test-model",
            "system_fingerprint": "test-fingerprint",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            "choices": [{"message": {"content": "0"}, "finish_reason": "stop"}],
        },
        separators=(",", ":"),
    )
    events = [
        {
            "type": "provider_request_attempt",
            "decision_index": 0,
            "provider_attempt_index": 0,
            "model_messages_sha256": hashlib.sha256(
                json.dumps(model_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "provider_request_sha256": _request_sha256(
                model_messages,
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            ),
            "status": "succeeded",
            "failure_class": None,
            "error": None,
            "latency_sec": 0.01,
            "provider_metadata": {
                "model": "resolved-test-model",
                "id": "request-1",
                "system_fingerprint": "test-fingerprint",
                "finish_reason": "stop",
            },
            "provider_response_sha256": hashlib.sha256(provider_response.encode("utf-8")).hexdigest(),
            "provider_response_body": provider_response,
        },
        {
            "type": "terminal_decision",
            "decision_index": 0,
            "prompt": "Choose: ",
            "prompt_contract": {
                "decision": "map_select",
                "canonical_option_tokens": ["0"],
                "accepted_syntax": ["canonical_token"],
                "meta_commands": [],
                "forbidden_meta_commands": ["save", "saves"],
                "multi_select": False,
                "multi_min": 1,
                "multi_max": 1,
            },
            "history_chars": len(visible_terminal),
            "terminal_window_start": 0,
            "terminal_window_end": len(visible_terminal),
            "terminal_window_sha256": hashlib.sha256(visible_terminal.encode("utf-8")).hexdigest(),
            "model_messages_sha256": hashlib.sha256(
                json.dumps(model_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "provider_attempt_count": 1,
            "game_ruleset_id": game_rules_identity()["ruleset_id"],
            "game_ruleset_sha256": game_rules_identity()["ruleset_sha256"],
            "game_rule_module_ids": game_rules_identity()["module_ids"],
            "game_rule_delivery": "static_full_system_prompt",
            "model_raw": "0",
            "command": "0",
            "output_format_valid": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            "latency_sec": 0.01,
            "provider_metadata": {
                "model": "resolved-test-model",
                "id": "request-1",
                "system_fingerprint": "test-fingerprint",
                "finish_reason": "stop",
            },
        },
        {"type": "summary", **summary},
    ]
    (artifacts / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events),
        encoding="utf-8",
    )


def _make_analysis_pending(command: list[str]) -> None:
    artifacts = Path(_argument(command, "--out-dir"))
    shutil.rmtree(artifacts / "trace_analysis")
    (artifacts / "benchmark_trace.jsonl").unlink()
    summary_path = artifacts / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "authoritative_terminal_state",
        "authoritative_terminal_state_sha256",
        "authoritative_terminal_deck",
        "authoritative_terminal_deck_sha256",
    ):
        summary.pop(key, None)
    summary.update(
        {
            "analysis": None,
            "analysis_error": "RuntimeError: injected report failure",
            "analysis_pending": True,
            "needs_attention": True,
        }
    )
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    events_path = artifacts / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[-1] = {"type": "summary", **summary}
    events_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")


def _write_model_protocol_failure(command: list[str], *, cwd: Path) -> None:
    _write_success(command, cwd=cwd)
    artifacts = Path(_argument(command, "--out-dir"))
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    visible_terminal = "Floor: 1\nChoose: "
    (artifacts / "terminal.txt").write_text(visible_terminal, encoding="utf-8")
    player = {
        "hp": 80,
        "max_hp": 80,
        "gold": 50,
        "deck_size": 1,
        "deck": [{"id": "CARD.STRIKE_IRONCLAD", "name": "Strike"}],
        "relics": [],
        "potions": [],
    }
    engine_rows = [
        {
            "step": 1,
            "type": "action",
            "data": {
                "cmd": "start_run",
                "seed": config["seed"],
                "character": config["character"],
                "ascension": config["ascension"],
                "lang": config["lang"],
            },
        },
        {
            "step": 2,
            "type": "state",
            "data": {
                "type": "decision",
                "decision": "map_select",
                "context": {"act": 1, "floor": 1, "room_type": "Map"},
                "choices": [],
                "player": player,
            },
        },
    ]
    engine_trace = artifacts / "engine_trace.jsonl"
    engine_trace.write_text(
        "".join(json.dumps(row) + "\n" for row in engine_rows),
        encoding="utf-8",
    )
    summary = {
        "completed": False,
        "outcome": "model_protocol_error",
        "decisions": 1,
        "max_visible_floor": 1,
        "invalid_cli_inputs": 0,
        "model_output_format_violations": 0,
        "usage": {},
        "model_latency_sec": 0.02,
        "elapsed_sec": 0.02,
        "decision_limit_reached": False,
        "error": "ModelProtocolError: response has no choices",
        "termination_class": "model_protocol_error",
        "formal_behavior_sample": True,
    }
    analysis = analyze_direct_cli_run(
        engine_trace,
        artifacts / "benchmark_trace.jsonl",
        artifacts / "trace_analysis",
        config=config,
        cli_summary=summary,
    )
    engine_audit = analysis["engine_audit"]
    summary.update(
        {
            "analysis": analysis,
            "analysis_error": None,
            "analysis_pending": False,
            "needs_attention": False,
            "authoritative_terminal_state": engine_audit["authoritative_terminal_state"],
            "authoritative_terminal_state_sha256": engine_audit["authoritative_terminal_state_sha256"],
            "authoritative_terminal_deck": engine_audit["authoritative_terminal_deck"],
            "authoritative_terminal_deck_sha256": engine_audit["authoritative_terminal_deck_sha256"],
        }
    )
    (artifacts / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + visible_terminal
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    messages_sha = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    response_body = json.dumps(
        {"model": "resolved-test-model", "id": "request-1", "choices": []},
        separators=(",", ":"),
    )
    retry_messages = [
        *messages,
        {"role": "user", "content": OUTPUT_RETRY_MESSAGE},
    ]
    retry_messages_sha = hashlib.sha256(
        json.dumps(retry_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = game_rules_identity()
    events = [
        {
            "type": "provider_request_attempt",
            "decision_index": 0,
            "provider_attempt_index": 0,
            "model_messages_sha256": messages_sha,
            "provider_request_sha256": _request_sha256(
                messages,
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            ),
            "status": "failed",
            "failure_class": "model_protocol_error",
            "error": "ModelProtocolError: response has no choices",
            "latency_sec": 0.01,
            "provider_metadata": {"model": "resolved-test-model", "id": "request-1"},
            "provider_response_body": response_body,
            "provider_response_sha256": hashlib.sha256(response_body.encode("utf-8")).hexdigest(),
        },
        {
            "type": "provider_request_attempt",
            "decision_index": 0,
            "provider_attempt_index": 1,
            "model_messages_sha256": retry_messages_sha,
            "provider_request_sha256": _request_sha256(
                retry_messages,
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            ),
            "status": "failed",
            "failure_class": "model_protocol_error",
            "error": "ModelProtocolError: response has no choices",
            "latency_sec": 0.01,
            "provider_metadata": {"model": "resolved-test-model", "id": "request-1"},
            "provider_response_body": response_body,
            "provider_response_sha256": hashlib.sha256(response_body.encode("utf-8")).hexdigest(),
        },
        {
            "type": "model_protocol_error",
            "decision_index": 0,
            "provider_attempt_count": 2,
            "prompt": "Choose: ",
            "prompt_contract": {
                "decision": "map_select",
                "canonical_option_tokens": [],
                "accepted_syntax": ["canonical_token"],
                "meta_commands": [],
                "forbidden_meta_commands": ["save", "saves"],
                "multi_select": False,
                "multi_min": 0,
                "multi_max": 0,
            },
            "history_chars": len(visible_terminal),
            "terminal_window_start": 0,
            "terminal_window_end": len(visible_terminal),
            "terminal_window_sha256": hashlib.sha256(visible_terminal.encode("utf-8")).hexdigest(),
            "model_messages_sha256": messages_sha,
            "game_ruleset_id": identity["ruleset_id"],
            "game_ruleset_sha256": identity["ruleset_sha256"],
            "game_rule_module_ids": identity["module_ids"],
            "game_rule_delivery": "static_full_system_prompt",
            "usage": {},
            "latency_sec": 0.01,
            "provider_metadata": {"model": "resolved-test-model", "id": "request-1"},
            "error": "ModelProtocolError: response has no choices",
        },
        {"type": "summary", **summary},
    ]
    (artifacts / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events),
        encoding="utf-8",
    )


def test_protocol_requires_one_worker_per_seed(tmp_path):
    protocol_path = _write_protocol(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["workers"] = 2
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(DirectCliExperimentError, match="one serial lane per seed"):
        load_protocol(protocol_path)


@pytest.mark.parametrize(
    "seeds, message",
    [
        (["../escape"], "safe single path components"),
        (["C:\\absolute"], "safe single path components"),
        (["seed/a"], "safe single path components"),
        (["CON"], "Windows reserved device names"),
        (["prn"], "Windows reserved device names"),
        (["COM1"], "Windows reserved device names"),
        (["lpt9"], "Windows reserved device names"),
        (["Seed", "seed"], "unique under Windows path normalization"),
    ],
)
def test_protocol_rejects_unsafe_or_case_colliding_seed_paths(tmp_path, seeds, message):
    protocol_path = _write_protocol(tmp_path, seeds=seeds)

    with pytest.raises(DirectCliExperimentError, match=message):
        load_protocol(protocol_path)


@pytest.mark.parametrize(
    "name",
    [
        "direct_cli_baseline_v1.json",
        "direct_cli_baseline_rules_v1.json",
        "direct_cli_multichar_v1.json",
        "direct_cli_rules_v2_formal_defect_20260723.json",
        "direct_cli_rules_v2_formal_ironclad_20260723.json",
        "direct_cli_rules_v2_formal_necrobinder_20260723.json",
        "direct_cli_rules_v2_formal_regent_20260723.json",
        "direct_cli_rules_v2_formal_silent_20260723.json",
        "direct_cli_rules_v2_formal_defect_v3_20260723.json",
        "direct_cli_rules_v2_formal_ironclad_v3_20260723.json",
        "direct_cli_rules_v2_formal_necrobinder_v3_20260723.json",
        "direct_cli_rules_v2_formal_regent_v3_20260723.json",
        "direct_cli_rules_v2_formal_silent_v3_20260723.json",
        "direct_cli_rules_v2_formal_defect_v4_20260723.json",
        "direct_cli_rules_v2_formal_ironclad_v4_20260723.json",
        "direct_cli_rules_v2_formal_necrobinder_v4_20260723.json",
        "direct_cli_rules_v2_formal_regent_v4_20260723.json",
        "direct_cli_rules_v2_formal_silent_v4_20260723.json",
        "direct_cli_rules_v2_formal_defect_v5_20260723.json",
        "direct_cli_rules_v2_formal_ironclad_v5_20260723.json",
        "direct_cli_rules_v2_formal_necrobinder_v5_20260723.json",
        "direct_cli_rules_v2_formal_regent_v5_20260723.json",
        "direct_cli_rules_v2_formal_silent_v5_20260723.json",
        "direct_cli_rules_v2_formal_defect_v6_20260723.json",
        "direct_cli_rules_v2_formal_ironclad_v6_20260723.json",
        "direct_cli_rules_v2_formal_necrobinder_v6_20260723.json",
        "direct_cli_rules_v2_formal_regent_v6_20260723.json",
        "direct_cli_rules_v2_formal_silent_v6_20260723.json",
        "direct_cli_rules_v2_formal_defect_v7_20260723.json",
        "direct_cli_rules_v2_formal_ironclad_v7_20260723.json",
        "direct_cli_rules_v2_formal_necrobinder_v7_20260723.json",
        "direct_cli_rules_v2_formal_regent_v7_20260723.json",
        "direct_cli_rules_v2_formal_silent_v7_20260723.json",
    ],
)
def test_legacy_direct_cli_protocols_are_provenance_only_and_rejected_by_formal_loader(name):
    protocol_path = Path(__file__).resolve().parents[1] / "experiments" / name

    with pytest.raises(DirectCliExperimentError, match="deprecated/exploratory provenance"):
        load_protocol(protocol_path)


def test_v8_ironclad_protocol_is_provenance_only_after_isolation_invalidation():
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "direct_cli_rules_v2_formal_ironclad_v8_20260723.json"
    )

    with pytest.raises(DirectCliExperimentError, match="deprecated/exploratory provenance"):
        load_protocol(protocol_path)


def _write_synthetic_preflight_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path]:
    source_root = Path(__file__).resolve().parents[1]
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fake_cli_root = Path(protocol["sts2_cli_root"])
    fake_root = tmp_path / "benchmark"
    replay_tool = fake_root / "tools" / "replay_direct_cli_context.py"
    replay_tool.parent.mkdir(parents=True)
    shutil.copy2(source_root / "tools" / "replay_direct_cli_context.py", replay_tool)
    replay_tool_sha256 = hash_file(replay_tool)
    preflight_root = fake_root / "preflight"
    preflight_root.mkdir(parents=True)
    raw_response = json.dumps(
        {"choices": [{"message": {"content": "0"}, "finish_reason": "stop"}]},
        separators=(",", ":"),
    )
    source_disposition = "invalidated_do_not_resume_or_count_for_paper"
    records: list[dict] = []
    historical_contexts = [
        ("defect_3407_d2", "Defect", 2),
        ("regent_1007_d11", "Regent", 11),
        ("regent_2407_d23", "Regent", 23),
    ]
    plan_path = fake_root / "experiments" / "selection-plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_sources: list[dict] = []
    plan_contexts: list[dict] = []
    source_by_context: dict[str, dict] = {}

    def synthetic_digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    for character in ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]:
        source_id = f"{character.lower()}_standard_source"
        source = {
            "source_id": source_id,
            "character": character,
            "artifacts_path": f"sources/{source_id}/artifacts",
            "config_sha256": synthetic_digest(f"{source_id}:config"),
            "events_sha256": synthetic_digest(f"{source_id}:events"),
            "terminal_sha256": synthetic_digest(f"{source_id}:terminal"),
            "terminal_decision_count": 11,
            "source_max_tokens": 512,
        }
        plan_sources.append(source)
        for selection, decision_index in [("early", 0), ("middle", 5), ("late", 10)]:
            name = f"{character.lower()}_{selection}"
            plan_contexts.append(
                {
                    "name": name,
                    "source_id": source_id,
                    "character": character,
                    "selection": selection,
                    "decision_index": decision_index,
                }
            )
            source_by_context[name] = source
    for name, character, decision_index in historical_contexts:
        source_id = f"{name}_source"
        source = {
            "source_id": source_id,
            "character": character,
            "artifacts_path": f"sources/{source_id}/artifacts",
            "config_sha256": synthetic_digest(f"{source_id}:config"),
            "events_sha256": synthetic_digest(f"{source_id}:events"),
            "terminal_sha256": synthetic_digest(f"{source_id}:terminal"),
            "terminal_decision_count": decision_index,
            "source_max_tokens": 4096,
        }
        plan_sources.append(source)
        plan_contexts.append(
            {
                "name": name,
                "source_id": source_id,
                "character": character,
                "selection": "historical_failure",
                "decision_index": decision_index,
            }
        )
        source_by_context[name] = source
    selection_plan = {
        "schema_version": 1,
        "selection_plan_id": "synthetic-preflight-selection-v1",
        "locked_at": "2026-07-23T00:00:00+00:00",
        "status": "locked_before_provider_calls",
        "model": "deepseek-v4-flash",
        "base_url": "https://zgc.apihy.com/v1",
        "preflight_output_limit_control": "provider_default_omitted",
        "output_root": "preflight",
        "call_policy": {
            "max_provider_calls_per_context": 1,
            "request_retries": 0,
            "failure_disposition": "preserve_and_invalidate_plan",
        },
        "paper_sample": False,
        "sources": plan_sources,
        "contexts": plan_contexts,
    }
    plan_path.write_text(json.dumps(selection_plan, indent=2) + "\n", encoding="utf-8")
    selection_plan_sha256 = hash_file(plan_path)

    def write_artifact(
        *,
        name: str,
        character: str,
        selection: str,
        decision_index: int,
        recorded_event_type: str,
    ) -> None:
        artifact_path = preflight_root / f"{name}.json"
        plan_source = source_by_context[name]
        messages = [
            {"role": "system", "content": "synthetic rules"},
            {"role": "user", "content": f"{character}:{decision_index}"},
        ]
        messages_sha256 = hashlib.sha256(
            json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        provider_metadata = {"finish_reason": "stop"}
        artifact = {
            "status": "passed",
            "finish_reason": "stop",
            "output_format_valid": True,
            "preflight_max_tokens": None,
            "preflight_output_limit_control": "provider_default_omitted",
            "preflight_provider_request": {
                "model": "deepseek-v4-flash",
                "messages": messages,
                "temperature": 0.0,
            },
            "model": "deepseek-v4-flash",
            "base_url": "https://zgc.apihy.com/v1",
            "tool_sha256": replay_tool_sha256,
            "source_config_sha256": plan_source["config_sha256"],
            "source_events_sha256": plan_source["events_sha256"],
            "source_terminal_sha256": plan_source["terminal_sha256"],
            "selection_plan_binding": {
                "selection_plan_id": selection_plan["selection_plan_id"],
                "selection_plan_path": plan_path.relative_to(fake_root).as_posix(),
                "selection_plan_sha256": selection_plan_sha256,
                "selection_name": name,
                "output_path": artifact_path.relative_to(fake_root).as_posix(),
                "selection": selection,
                "source_id": plan_source["source_id"],
                "source_artifacts_path": plan_source["artifacts_path"],
                "source_terminal_decision_count": plan_source["terminal_decision_count"],
                "source_max_tokens": plan_source["source_max_tokens"],
            },
            "source_provenance": {
                "source_eligibility": "invalidated_protocol_context",
                "source_protocol_disposition": source_disposition,
                "character": character,
                "source_report_required_sha256": "a" * 64,
                "accepted_model_messages_sha256": messages_sha256,
            },
            "decision_index": decision_index,
            "recorded_max_tokens": plan_source["source_max_tokens"],
            "recorded_model_messages_sha256": messages_sha256,
            "recorded_event_type": recorded_event_type,
            "recorded_event_error": (
                "ValueError: LLM response did not complete normally (finish_reason=length)"
                if recorded_event_type == "model_protocol_error"
                else None
            ),
            "recorded_output_format_valid": (
                True if recorded_event_type == "terminal_decision" else None
            ),
            "recorded_provider_metadata": {
                "finish_reason": (
                    "length" if recorded_event_type == "model_protocol_error" else "stop"
                )
            },
            "provider_metadata": provider_metadata,
            "usage": {},
            "response_text": "0",
            "response_text_sha256": hashlib.sha256(b"0").hexdigest(),
            "command": "0",
            "provider_attempts": [
                {
                    "provider_attempt_index": 0,
                    "model_messages_sha256": messages_sha256,
                    "provider_request_sha256": _request_sha256(
                        messages,
                        model="deepseek-v4-flash",
                        temperature=0.0,
                        max_tokens=None,
                    ),
                    "status": "succeeded",
                    "failure_class": None,
                    "error": None,
                    "provider_metadata": provider_metadata,
                    "provider_response_encoding": "utf-8",
                    "provider_response_body": raw_response,
                    "provider_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                }
            ],
        }
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        records.append(
            {
                "name": name,
                "path": artifact_path.name,
                "sha256": hash_file(artifact_path),
                "character": character,
                "selection": selection,
            }
        )

    for character in ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]:
        for selection, decision_index in [("early", 0), ("middle", 5), ("late", 10)]:
            write_artifact(
                name=f"{character.lower()}_{selection}",
                character=character,
                selection=selection,
                decision_index=decision_index,
                recorded_event_type="terminal_decision",
            )
    for name, character, decision_index in historical_contexts:
        write_artifact(
            name=name,
            character=character,
            selection="historical_failure",
            decision_index=decision_index,
            recorded_event_type="model_protocol_error",
        )

    manifest_path = preflight_root / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": "2026-07-23T00:01:00+00:00",
        "status": "passed",
        "model": "deepseek-v4-flash",
        "base_url": "https://zgc.apihy.com/v1",
        "output_limit_control": "provider_default_omitted",
        "selection_plan_sha256": selection_plan_sha256,
        "artifacts": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    protocol.update(
        {
            "model": "deepseek-v4-flash",
            "base_url": "https://zgc.apihy.com/v1",
            "temperature": 0.0,
            "max_tokens": None,
            "request_retries": 1,
            "retry_correction_scope": "interface_specific_same_state",
            "cost_accounting": "aggregate_all_provider_attempts",
            "preflight_evidence": {
                "selection_plan_path": plan_path.relative_to(fake_root).as_posix(),
                "selection_plan_sha256": selection_plan_sha256,
                "manifest_path": "preflight/manifest.json",
                "manifest_sha256": hash_file(manifest_path),
                "required_artifact_count": 18,
                "required_historical_failure_contexts": [item[0] for item in historical_contexts],
                "historical_failure_max_tokens": 4096,
                "required_source_protocol_disposition": source_disposition,
            },
        }
    )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(direct_cli_experiment, "ROOT", fake_root)
    return protocol_path, fake_cli_root, fake_root, manifest_path, plan_path


def _repin_synthetic_selection_plan(
    protocol_path: Path,
    manifest_path: Path,
    plan_path: Path,
) -> None:
    plan_sha256 = hash_file(plan_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["artifacts"]:
        artifact_path = manifest_path.parent / record["path"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["selection_plan_binding"]["selection_plan_sha256"] = plan_sha256
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        record["sha256"] = hash_file(artifact_path)
    manifest["selection_plan_sha256"] = plan_sha256
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["preflight_evidence"]["selection_plan_sha256"] = plan_sha256
    protocol["preflight_evidence"]["manifest_sha256"] = hash_file(manifest_path)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def _repin_synthetic_manifest(
    protocol_path: Path,
    manifest_path: Path,
    manifest: dict,
) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["preflight_evidence"]["manifest_sha256"] = hash_file(manifest_path)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize("damage", ["manifest_hash", "artifact_hash", "historical_failure"])
def test_v5_preflight_gate_rejects_tampered_evidence(tmp_path, monkeypatch, damage):
    protocol_path, _, _, manifest_path, _ = _write_synthetic_preflight_protocol(
        tmp_path,
        monkeypatch,
    )
    assert load_protocol(protocol_path)["preflight_evidence"]["required_artifact_count"] == 18
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    if damage == "manifest_hash":
        protocol["preflight_evidence"]["manifest_sha256"] = "0" * 64
    elif damage == "artifact_hash":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_path = manifest_path.parent / manifest["artifacts"][0]["path"]
        artifact_path.write_text(artifact_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = next(
            item for item in manifest["artifacts"] if item["name"] == "defect_3407_d2"
        )
        target["selection"] = "middle"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        protocol["preflight_evidence"]["manifest_sha256"] = hash_file(manifest_path)

    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(DirectCliExperimentError, match="[Pp]reflight"):
        load_protocol(protocol_path)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_request",
        "request_includes_max_tokens",
        "tampered_request_hash",
        "missing_response",
        "rehashed_response_mismatch",
        "historical_failure_not_length",
    ],
)
def test_v5_preflight_gate_rejects_incomplete_or_forged_provider_transaction(
    tmp_path,
    monkeypatch,
    damage,
):
    protocol_path, _, _, manifest_path, _ = _write_synthetic_preflight_protocol(
        tmp_path,
        monkeypatch,
    )
    assert load_protocol(protocol_path)["preflight_evidence"]["required_artifact_count"] == 18
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_name = (
        "defect_3407_d2"
        if damage == "historical_failure_not_length"
        else "ironclad_early"
    )
    target = next(item for item in manifest["artifacts"] if item["name"] == target_name)
    artifact_path = manifest_path.parent / target["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    attempt = artifact["provider_attempts"][0]

    if damage == "missing_request":
        artifact.pop("preflight_provider_request")
    elif damage == "request_includes_max_tokens":
        artifact["preflight_provider_request"]["max_tokens"] = 4096
    elif damage == "tampered_request_hash":
        attempt["provider_request_sha256"] = "0" * 64
    elif damage == "missing_response":
        attempt.pop("provider_response_body")
        attempt.pop("provider_response_sha256")
        attempt.pop("provider_response_encoding")
    elif damage == "rehashed_response_mismatch":
        forged_response = json.dumps(
            {"choices": [{"message": {"content": "1"}, "finish_reason": "stop"}]},
            separators=(",", ":"),
        )
        attempt["provider_response_body"] = forged_response
        attempt["provider_response_sha256"] = hashlib.sha256(
            forged_response.encode("utf-8")
        ).hexdigest()
    else:
        artifact["recorded_provider_metadata"]["finish_reason"] = "stop"

    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    target["sha256"] = hash_file(artifact_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    protocol["preflight_evidence"]["manifest_sha256"] = hash_file(manifest_path)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(DirectCliExperimentError, match="[Pp]reflight|finish_reason=length"):
        load_protocol(protocol_path)


@pytest.mark.parametrize(
    "damage",
    [
        "plan_hash",
        "plan_later_than_manifest",
        "source_hash",
        "context_index",
        "historical_cap",
        "coordinated_non_midpoint",
        "output_root",
        "call_policy",
        "source_path_escape",
    ],
)
def test_v5_preflight_gate_rejects_selection_plan_drift_and_posthoc_selection(
    tmp_path,
    monkeypatch,
    damage,
):
    protocol_path, _, _, manifest_path, plan_path = _write_synthetic_preflight_protocol(
        tmp_path,
        monkeypatch,
    )
    assert load_protocol(protocol_path)["preflight_evidence"]["required_artifact_count"] == 18
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if damage == "plan_hash":
        plan_path.write_text(plan_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif damage == "plan_later_than_manifest":
        plan["locked_at"] = "2026-07-23T00:02:00+00:00"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        _repin_synthetic_selection_plan(protocol_path, manifest_path, plan_path)
    elif damage == "output_root":
        plan["output_root"] = "different-preflight-root"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        _repin_synthetic_selection_plan(protocol_path, manifest_path, plan_path)
    elif damage == "call_policy":
        plan["call_policy"]["max_provider_calls_per_context"] = 2
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        _repin_synthetic_selection_plan(protocol_path, manifest_path, plan_path)
    elif damage == "source_path_escape":
        plan["sources"][0]["artifacts_path"] = "../outside-source"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        _repin_synthetic_selection_plan(protocol_path, manifest_path, plan_path)
    elif damage == "source_hash":
        target = next(item for item in manifest["artifacts"] if item["name"] == "ironclad_early")
        artifact_path = manifest_path.parent / target["path"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["source_config_sha256"] = "0" * 64
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        target["sha256"] = hash_file(artifact_path)
        _repin_synthetic_manifest(protocol_path, manifest_path, manifest)
    elif damage == "context_index":
        context = next(item for item in plan["contexts"] if item["name"] == "ironclad_middle")
        context["decision_index"] = 4
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        _repin_synthetic_selection_plan(protocol_path, manifest_path, plan_path)
    elif damage == "historical_cap":
        context = next(item for item in plan["contexts"] if item["name"] == "defect_3407_d2")
        source = next(item for item in plan["sources"] if item["source_id"] == context["source_id"])
        source["source_max_tokens"] = 2048
        target = next(item for item in manifest["artifacts"] if item["name"] == context["name"])
        artifact_path = manifest_path.parent / target["path"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["recorded_max_tokens"] = 2048
        artifact["selection_plan_binding"]["source_max_tokens"] = 2048
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        target["sha256"] = hash_file(artifact_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        _repin_synthetic_selection_plan(protocol_path, manifest_path, plan_path)
    else:
        context = next(item for item in plan["contexts"] if item["name"] == "ironclad_middle")
        context["decision_index"] = 4
        target = next(item for item in manifest["artifacts"] if item["name"] == context["name"])
        artifact_path = manifest_path.parent / target["path"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["decision_index"] = 4
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        target["sha256"] = hash_file(artifact_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        _repin_synthetic_selection_plan(protocol_path, manifest_path, plan_path)

    with pytest.raises(
        DirectCliExperimentError,
        match="[Pp]reflight|selection plan|endpoint/midpoint|finish_reason=length",
    ):
        load_protocol(protocol_path)


def test_v5_freeze_inventory_includes_preflight_manifest_outputs_and_replay_tool(tmp_path, monkeypatch):
    protocol_path, fake_cli_root, root, manifest_path, plan_path = _write_synthetic_preflight_protocol(
        tmp_path,
        monkeypatch,
    )
    protocol = load_protocol(protocol_path)
    monkeypatch.setattr(
        direct_cli_experiment,
        "_BENCHMARK_EXECUTION_FILES",
        ("tools/replay_direct_cli_context.py",),
    )

    inventory = direct_cli_experiment._source_inventory(protocol_path, fake_cli_root)
    labels = {item["label"] for item in inventory}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_preflight_labels = {
        f"benchmark/{plan_path.relative_to(root).as_posix()}",
        f"benchmark/{manifest_path.relative_to(root).as_posix()}",
        *(
            f"benchmark/{(manifest_path.parent / item['path']).relative_to(root).as_posix()}"
            for item in manifest["artifacts"]
        ),
    }

    assert len(manifest["artifacts"]) == 18
    assert expected_preflight_labels <= labels
    assert "benchmark/tools/replay_direct_cli_context.py" in labels
    frozen_labels = expected_preflight_labels | {"benchmark/tools/replay_direct_cli_context.py"}
    assert all(item["tracked"] is True for item in inventory if item["label"] in frozen_labels)


@pytest.mark.parametrize("max_tokens", [pytest.param("missing", id="missing"), pytest.param(512, id="explicit")])
def test_deepseek_v4_flash_formal_protocol_requires_omitted_provider_output_limit(tmp_path, max_tokens):
    protocol_path = _write_protocol(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["model"] = "deepseek-v4-flash"
    if max_tokens != "missing":
        protocol["max_tokens"] = max_tokens
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(DirectCliExperimentError, match="must omit max_tokens"):
        load_protocol(protocol_path)


def test_deepseek_v4_flash_formal_protocol_requires_one_request_retry(tmp_path):
    protocol_path = _write_protocol(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({"model": "deepseek-v4-flash", "max_tokens": None, "request_retries": 0})
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(DirectCliExperimentError, match="request_retries=1"):
        load_protocol(protocol_path)


def test_plan_exposes_three_serial_seed_lanes(tmp_path):
    protocol_path = _write_protocol(tmp_path, repeats=5)

    plan = build_plan(protocol_path)

    assert plan["workers"] == 3
    assert plan["lane_semantics"] == "one_serial_lane_per_seed"
    assert plan["expected_runs"] == 15
    assert [lane["seed"] for lane in plan["lanes"]] == ["seed_a", "seed_b", "seed_c"]
    assert all(lane["replicates"] == [1, 2, 3, 4, 5] for lane in plan["lanes"])

    overridden = build_plan(protocol_path, out_root_override=tmp_path / "frozen-results")
    assert overridden["output_root"] == str((tmp_path / "frozen-results").resolve())


def test_source_lock_tracks_active_runtime_and_rejects_drift(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = load_protocol(protocol_path)
    cli_root = Path(protocol["sts2_cli_root"])
    out_root = tmp_path / "freeze"
    out_root.mkdir()

    frozen = initialize_or_verify_source_lock(out_root, protocol_path=protocol_path, sts2_cli_root=cli_root)
    assert (out_root / "source_lock.json").is_file()
    assert frozen["execution_mode"] == "active_worktree"
    assert "code_snapshot_created" not in frozen
    assert not (out_root / "run_frozen.ps1").exists()
    assert not (out_root / "code_snapshot").exists()
    labels = {item["label"] for item in frozen["inventory"]}
    assert any(label.startswith("sts2-cli/localization_eng/") for label in labels)
    assert any(label.startswith("sts2-cli/localization_zhs/") for label in labels)

    (cli_root / "python" / "play.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(DirectCliExperimentError, match="Locked active source file drifted"):
        initialize_or_verify_source_lock(out_root, protocol_path=protocol_path, sts2_cli_root=cli_root)


def test_recorded_runtime_identity_matches_the_sanitized_frozen_child():
    script = (
        "import json; "
        "from sts2_bench.direct_cli_baseline import _python_runtime_identity; "
        "print(json.dumps(_python_runtime_identity(), sort_keys=True))"
    )
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    child = subprocess.run(
        [sys.executable, "-B", "-s", "-X", "utf8", "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=True,
    )

    assert direct_cli_experiment._python_runtime_identity() == json.loads(child.stdout)


def test_active_worktree_runs_formal_samples_directly(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    env_file = tmp_path / "external.env"
    env_file.write_text("DEEPSEEK_API_KEY=test-only\n", encoding="utf-8")

    summary = run_experiment(
        protocol_path,
        resume=False,
        env_file=env_file,
        runner=lambda command, **kwargs: (_write_success(command, cwd=kwargs["cwd"]) or 0),
    )
    assert summary["status"] == "succeeded"
    assert summary["completed_runs"] == 1
    assert not (tmp_path / "results" / "code_snapshot").exists()


def test_lock_only_writes_a_source_lock_without_an_execution_copy(tmp_path, capsys):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)

    assert direct_cli_experiment.main(["--protocol", str(protocol_path), "--lock-only"]) == 0
    capsys.readouterr()
    out_root = tmp_path / "results"
    frozen = json.loads((out_root / "source_lock.json").read_text(encoding="utf-8"))
    assert frozen["execution_mode"] == "active_worktree"
    assert not (out_root / "code_snapshot").exists()
    assert not (out_root / "run_frozen.ps1").exists()


def test_dry_run_reports_protocol_validation_errors_without_traceback(tmp_path, capsys):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.pop("prompt_contract_required")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    assert (
        direct_cli_experiment.main(
            ["--protocol", str(protocol_path), "--dry-run"]
        )
        == 1
    )
    output = capsys.readouterr()
    payload = json.loads(output.err)
    assert payload["status"] == "error"
    assert "prompt_contract_required" in payload["error"]
    assert "Traceback" not in output.err


def test_remote_endpoint_requires_key_from_explicit_external_env_file(tmp_path, monkeypatch):
    for name in (
        "STS2_BENCH_API_KEY",
        "QWEN_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("UNRELATED=value\n", encoding="utf-8")
    forbidden_roots = (tmp_path / "formal-results",)

    with pytest.raises(DirectCliExperimentError, match="requires an API key"):
        _load_and_validate_external_environment(
            empty_env,
            base_url="https://example.invalid/v1",
            forbidden_roots=forbidden_roots,
        )

    keyed_env = tmp_path / "keyed.env"
    keyed_env.write_text(
        "QWEN_API_KEY=test-only\nSTS2_LIB=C:/external/lib\nUNRELATED=injected\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STS2_BENCH_API_KEY", "local")
    monkeypatch.setenv("STS2_GAME_DIR", "C:/external/game")
    monkeypatch.delenv("UNRELATED", raising=False)
    _load_and_validate_external_environment(
        keyed_env,
        base_url="https://example.invalid/v1",
        forbidden_roots=forbidden_roots,
    )
    assert "STS2_LIB" not in direct_cli_experiment.os.environ
    assert "STS2_GAME_DIR" not in direct_cli_experiment.os.environ
    assert "UNRELATED" not in direct_cli_experiment.os.environ


def test_external_env_file_must_be_outside_entire_formal_result_root(tmp_path, monkeypatch):
    out_root = tmp_path / "formal-results"
    code_snapshot = out_root / "code_snapshot"
    code_snapshot.mkdir(parents=True)
    inside_result = out_root / "credentials.env"
    inside_result.write_text("DEEPSEEK_API_KEY=must-not-load\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-remains-on-rejection")

    for candidate in (inside_result, code_snapshot / "credentials.env"):
        if not candidate.exists():
            candidate.write_text("DEEPSEEK_API_KEY=must-not-load\n", encoding="utf-8")
        with pytest.raises(DirectCliExperimentError, match="outside"):
            _load_and_validate_external_environment(
                candidate,
                base_url="https://example.invalid/v1",
                forbidden_roots=(out_root, code_snapshot),
            )

    assert direct_cli_experiment.os.environ["DEEPSEEK_API_KEY"] == "ambient-remains-on-rejection"


def test_external_env_hardlinks_are_rejected_before_environment_mutation(tmp_path, monkeypatch):
    out_root = tmp_path / "formal-results"
    out_root.mkdir()
    inside = out_root / "credentials.env"
    inside.write_text("DEEPSEEK_API_KEY=inside\n", encoding="utf-8")
    outside_alias = tmp_path / "outside-hardlink-to-result.env"
    outside_alias.hardlink_to(inside)

    external = tmp_path / "external.env"
    external.write_text("DEEPSEEK_API_KEY=external\n", encoding="utf-8")
    inside_alias = out_root / "result-hardlink-to-external.env"
    inside_alias.hardlink_to(external)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-remains-on-rejection")

    for candidate in (outside_alias, inside_alias):
        with pytest.raises(DirectCliExperimentError, match="outside|single-link"):
            _load_and_validate_external_environment(
                candidate,
                base_url="https://example.invalid/v1",
                forbidden_roots=(out_root, out_root / "code_snapshot"),
            )
        assert direct_cli_experiment.os.environ["DEEPSEEK_API_KEY"] == "ambient-remains-on-rejection"


def test_external_env_accepts_ordinary_single_link_outside_results(tmp_path, monkeypatch):
    out_root = tmp_path / "formal-results"
    out_root.mkdir()
    external = tmp_path / "ordinary-external.env"
    external.write_text("DEEPSEEK_API_KEY=ordinary-external\n", encoding="utf-8")
    assert external.stat().st_nlink == 1
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-must-be-replaced")

    _load_and_validate_external_environment(
        external,
        base_url="https://example.invalid/v1",
        forbidden_roots=(out_root, out_root / "code_snapshot"),
    )

    assert direct_cli_experiment.os.environ["DEEPSEEK_API_KEY"] == "ordinary-external"


def test_external_env_real_path_cannot_enter_results_through_file_link(tmp_path, monkeypatch):
    out_root = tmp_path / "formal-results"
    out_root.mkdir()
    inside = out_root / "credentials.env"
    inside.write_text("DEEPSEEK_API_KEY=must-not-load\n", encoding="utf-8")
    external_target = tmp_path / "external.env"
    external_target.write_text("DEEPSEEK_API_KEY=external\n", encoding="utf-8")
    file_alias = tmp_path / "outside-file-alias.env"
    in_result_alias = out_root / "external-file-alias.env"
    try:
        file_alias.symlink_to(inside)
        in_result_alias.symlink_to(external_target)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-remains-on-rejection")

    for candidate in (file_alias, in_result_alias):
        with pytest.raises(DirectCliExperimentError, match="outside"):
            _load_and_validate_external_environment(
                candidate,
                base_url="https://example.invalid/v1",
                forbidden_roots=(out_root, out_root / "code_snapshot"),
            )

    assert direct_cli_experiment.os.environ["DEEPSEEK_API_KEY"] == "ambient-remains-on-rejection"


def test_external_env_real_path_cannot_enter_results_through_directory_link_or_junction(
    tmp_path,
    monkeypatch,
):
    out_root = tmp_path / "formal-results"
    out_root.mkdir()
    inside = out_root / "credentials.env"
    inside.write_text("DEEPSEEK_API_KEY=must-not-load\n", encoding="utf-8")
    result_alias = tmp_path / "outside-result-alias"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(result_alias), str(out_root)],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if created.returncode != 0:
            pytest.skip("directory junctions are unavailable")
    else:
        result_alias.symlink_to(out_root, target_is_directory=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-remains-on-rejection")

    with pytest.raises(DirectCliExperimentError, match="outside"):
        _load_and_validate_external_environment(
            result_alias / "credentials.env",
            base_url="https://example.invalid/v1",
            forbidden_roots=(out_root, out_root / "code_snapshot"),
        )

    assert direct_cli_experiment.os.environ["DEEPSEEK_API_KEY"] == "ambient-remains-on-rejection"


def test_active_runner_fails_before_attempt_when_external_env_has_no_key(tmp_path, monkeypatch):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("UNRELATED=value\n", encoding="utf-8")
    for name in ("STS2_BENCH_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    runner_called = False

    def forbidden_runner(*args, **kwargs):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner must not start without a credential")

    with pytest.raises(DirectCliExperimentError, match="requires an API key"):
        run_experiment(
            protocol_path,
            resume=False,
            env_file=empty_env,
            runner=forbidden_runner,
        )

    assert runner_called is False
    assert not (tmp_path / "results" / "runs").exists()


def test_request_audit_reconstructs_messages_and_rejects_arbitrary_sha():
    visible_terminal = "Floor: 1\nChoose: "
    terminal = visible_terminal + "0\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + visible_terminal
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    identity = game_rules_identity()
    provider_response = json.dumps(
        {
            "id": "request-1",
            "model": "resolved-test-model",
            "choices": [{"message": {"content": "0"}, "finish_reason": "stop"}],
        },
        separators=(",", ":"),
    )
    provider_metadata = {
        "model": "resolved-test-model",
        "id": "request-1",
        "finish_reason": "stop",
    }
    event = {
        "decision_index": 0,
        "prompt": "Choose: ",
        "history_chars": len(visible_terminal),
        "terminal_window_start": 0,
        "terminal_window_end": len(visible_terminal),
        "terminal_window_sha256": hashlib.sha256(visible_terminal.encode("utf-8")).hexdigest(),
        "model_messages_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "provider_attempt_count": 1,
        "game_ruleset_id": identity["ruleset_id"],
        "game_ruleset_sha256": identity["ruleset_sha256"],
        "game_rule_module_ids": identity["module_ids"],
        "game_rule_delivery": "static_full_system_prompt",
        "usage": {},
        "latency_sec": 0.01,
        "provider_metadata": {},
        "model_raw": "0",
        "command": "0",
        "output_format_valid": True,
        "usage": {},
        "latency_sec": 0.01,
        "provider_metadata": provider_metadata,
    }

    provider_attempt = {
        "decision_index": 0,
        "provider_attempt_index": 0,
        "model_messages_sha256": event["model_messages_sha256"],
        "provider_request_sha256": _request_sha256(
            messages,
            model="test-model",
            temperature=0.0,
            max_tokens=None,
        ),
        "status": "succeeded",
        "failure_class": None,
        "latency_sec": 0.01,
        "provider_metadata": provider_metadata,
        "provider_response_sha256": hashlib.sha256(provider_response.encode("utf-8")).hexdigest(),
        "provider_response_body": provider_response,
    }
    assert _audit_terminal_decision_requests(
        [event],
        provider_attempts=[provider_attempt],
        protocol_errors=[],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=1,
    )["valid"] is True
    assert _audit_terminal_decision_requests(
        [{**event, "model_messages_sha256": "a" * 64}],
        provider_attempts=[provider_attempt],
        protocol_errors=[],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=1,
    )["valid"] is False

    assert _audit_terminal_decision_requests(
        [{**event, "command": "deck"}],
        provider_attempts=[provider_attempt],
        protocol_errors=[],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=1,
    )["valid"] is False


def test_request_audit_accepts_target_prompt_contract_and_rejects_missing_decision():
    visible_terminal = "Combat\nTarget enemy [index]: "
    terminal = visible_terminal + "1\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + visible_terminal
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    identity = game_rules_identity()
    provider_response = json.dumps(
        {
            "id": "request-target-1",
            "model": "resolved-test-model",
            "choices": [{"message": {"content": "1"}, "finish_reason": "stop"}],
        },
        separators=(",", ":"),
    )
    provider_metadata = {
        "model": "resolved-test-model",
        "id": "request-target-1",
        "finish_reason": "stop",
    }
    event = {
        "decision_index": 0,
        "prompt": "Target enemy [index]: ",
        "prompt_contract": {
            "decision": "combat_play",
            "canonical_option_tokens": ["0", "1"],
            "accepted_syntax": ["canonical_token"],
            "meta_commands": ["deck", "discard", "draw", "help", "relics", "state"],
            "forbidden_meta_commands": ["save", "saves"],
            "multi_select": False,
            "multi_min": 1,
            "multi_max": 1,
        },
        "history_chars": len(visible_terminal),
        "terminal_window_start": 0,
        "terminal_window_end": len(visible_terminal),
        "terminal_window_sha256": hashlib.sha256(visible_terminal.encode("utf-8")).hexdigest(),
        "model_messages_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "provider_attempt_count": 1,
        "game_ruleset_id": identity["ruleset_id"],
        "game_ruleset_sha256": identity["ruleset_sha256"],
        "game_rule_module_ids": identity["module_ids"],
        "game_rule_delivery": "static_full_system_prompt",
        "model_raw": "1",
        "command": "1",
        "output_format_valid": True,
        "usage": {},
        "latency_sec": 0.01,
        "provider_metadata": provider_metadata,
    }
    provider_attempt = {
        "decision_index": 0,
        "provider_attempt_index": 0,
        "model_messages_sha256": event["model_messages_sha256"],
        "provider_request_sha256": _request_sha256(
            messages,
            model="test-model",
            temperature=0.0,
            max_tokens=None,
        ),
        "status": "succeeded",
        "failure_class": None,
        "latency_sec": 0.01,
        "provider_metadata": provider_metadata,
        "provider_response_sha256": hashlib.sha256(provider_response.encode("utf-8")).hexdigest(),
        "provider_response_body": provider_response,
    }

    accepted = _audit_terminal_decision_requests(
        [event],
        provider_attempts=[provider_attempt],
        protocol_errors=[],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=1,
        require_prompt_contract=True,
    )
    rejected = _audit_terminal_decision_requests(
        [{**event, "prompt_contract": {**event["prompt_contract"], "decision": ""}}],
        provider_attempts=[provider_attempt],
        protocol_errors=[],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=1,
        require_prompt_contract=True,
    )

    assert accepted["valid"] is True
    assert accepted["decisions"][0]["checks"]["prompt_contract_recorded"] is True
    assert rejected["valid"] is False
    assert rejected["decisions"][0]["checks"]["prompt_contract_recorded"] is False


def test_request_audit_accepts_transport_failures_followed_by_preserved_protocol_failure():
    terminal = "Floor: 1\nChoose: "
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + terminal
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    messages_sha = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = game_rules_identity()
    response_body = '{"choices": []}'
    protocol_error = {
        "decision_index": 0,
        "provider_attempt_count": 2,
        "prompt": "Choose: ",
        "history_chars": len(terminal),
        "terminal_window_start": 0,
        "terminal_window_end": len(terminal),
        "terminal_window_sha256": hashlib.sha256(terminal.encode("utf-8")).hexdigest(),
        "model_messages_sha256": messages_sha,
        "game_ruleset_id": identity["ruleset_id"],
        "game_ruleset_sha256": identity["ruleset_sha256"],
        "game_rule_module_ids": identity["module_ids"],
        "game_rule_delivery": "static_full_system_prompt",
        "usage": {},
        "latency_sec": 0.01,
        "provider_metadata": {},
        "error": "ModelProtocolError: response has no choices",
    }
    attempts = [
        {
            "decision_index": 0,
            "provider_attempt_index": 0,
            "model_messages_sha256": messages_sha,
            "provider_request_sha256": _request_sha256(
                messages,
                model="test-model",
                temperature=0.0,
                max_tokens=None,
            ),
            "status": "failed",
            "failure_class": "provider_transport_error",
            "error": "ConnectionResetError: dropped",
            "latency_sec": 0.01,
            "provider_metadata": {},
        },
        {
            "decision_index": 0,
            "provider_attempt_index": 1,
            "model_messages_sha256": messages_sha,
            "provider_request_sha256": _request_sha256(
                messages,
                model="test-model",
                temperature=0.0,
                max_tokens=None,
            ),
            "status": "failed",
            "failure_class": "model_protocol_error",
            "error": "ModelProtocolError: response has no choices",
            "latency_sec": 0.01,
            "provider_metadata": {},
            "provider_response_body": response_body,
            "provider_response_sha256": hashlib.sha256(response_body.encode("utf-8")).hexdigest(),
        },
    ]

    audit = _audit_terminal_decision_requests(
        [],
        provider_attempts=attempts,
        protocol_errors=[protocol_error],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=1,
    )

    assert audit["valid"] is True
    assert audit["protocol_error_count"] == 1


def _protocol_error_audit_fixture(
    response_body: str,
    *,
    provider_metadata: dict | None = None,
    error: str = "ModelProtocolError: provider response is unusable",
):
    terminal = "Floor: 1\nChoose: "
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + terminal
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    messages_sha = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = game_rules_identity()
    metadata = dict(provider_metadata or {})
    event = {
        "decision_index": 0,
        "provider_attempt_count": 1,
        "prompt": "Choose: ",
        "history_chars": len(terminal),
        "terminal_window_start": 0,
        "terminal_window_end": len(terminal),
        "terminal_window_sha256": hashlib.sha256(terminal.encode("utf-8")).hexdigest(),
        "model_messages_sha256": messages_sha,
        "game_ruleset_id": identity["ruleset_id"],
        "game_ruleset_sha256": identity["ruleset_sha256"],
        "game_rule_module_ids": identity["module_ids"],
        "game_rule_delivery": "static_full_system_prompt",
        "usage": {},
        "latency_sec": 0.01,
        "provider_metadata": metadata,
        "error": error,
    }
    attempt = {
        "decision_index": 0,
        "provider_attempt_index": 0,
        "model_messages_sha256": messages_sha,
        "provider_request_sha256": _request_sha256(
            messages,
            model="test-model",
            temperature=0.0,
            max_tokens=None,
        ),
        "status": "failed",
        "failure_class": "model_protocol_error",
        "error": error,
        "latency_sec": 0.01,
        "provider_metadata": metadata,
        "provider_response_encoding": "utf-8",
        "provider_response_body": response_body,
        "provider_response_sha256": hashlib.sha256(response_body.encode("utf-8")).hexdigest(),
    }
    return terminal, event, attempt


def test_request_audit_rejects_contiguous_attempt_chain_beyond_retry_budget():
    terminal, event, final_attempt = _protocol_error_audit_fixture("[]")
    transport_attempt = {
        "decision_index": 0,
        "provider_attempt_index": 0,
        "model_messages_sha256": final_attempt["model_messages_sha256"],
        "provider_request_sha256": final_attempt["provider_request_sha256"],
        "status": "failed",
        "failure_class": "provider_transport_error",
        "error": "ConnectionResetError: dropped",
        "latency_sec": 0.01,
        "provider_metadata": {},
    }
    final_attempt = {**final_attempt, "provider_attempt_index": 1}

    audit = _audit_terminal_decision_requests(
        [],
        provider_attempts=[transport_attempt, final_attempt],
        protocol_errors=[{**event, "provider_attempt_count": 2}],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=0,
    )

    assert audit["valid"] is False
    assert audit["decisions"][0]["checks"]["provider_attempt_count_within_request_budget"] is False


@pytest.mark.parametrize("latency", [None, True, -0.01, float("nan"), float("inf"), "0.01"])
def test_request_audit_rejects_invalid_provider_attempt_latency(latency):
    terminal, event, attempt = _protocol_error_audit_fixture("[]")
    attempt["latency_sec"] = latency

    audit = _audit_terminal_decision_requests(
        [],
        provider_attempts=[attempt],
        protocol_errors=[event],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=0,
    )

    assert audit["valid"] is False
    assert audit["decisions"][0]["checks"]["provider_attempt_latencies_valid"] is False
    assert audit["model_latency_sec"] == 0.0


def test_request_audit_rejects_valid_stop_response_disguised_as_protocol_error():
    response_body = json.dumps(
        {"choices": [{"message": {"content": "0"}, "finish_reason": "stop"}]},
        separators=(",", ":"),
    )
    terminal, event, attempt = _protocol_error_audit_fixture(
        response_body,
        provider_metadata={"finish_reason": "stop"},
    )

    audit = _audit_terminal_decision_requests(
        [],
        provider_attempts=[attempt],
        protocol_errors=[event],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=0,
    )

    assert audit["valid"] is False
    assert audit["decisions"][0]["checks"]["protocol_failure_semantics_valid"] is False


def test_request_audit_rejects_valid_stop_response_disguised_as_intermediate_retry_failure():
    valid_body = json.dumps(
        {"choices": [{"message": {"content": "discarded-command"}, "finish_reason": "stop"}]},
        separators=(",", ":"),
    )
    terminal, event, first_attempt = _protocol_error_audit_fixture(
        valid_body,
        provider_metadata={"finish_reason": "stop"},
        error="ModelProtocolError: forged retry",
    )
    _, _, final_attempt = _protocol_error_audit_fixture("[]")
    retry_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + terminal
                + "\n\nReturn only the next terminal input line."
            ),
        },
        {"role": "user", "content": OUTPUT_RETRY_MESSAGE},
    ]
    final_attempt = {
        **final_attempt,
        "provider_attempt_index": 1,
        "model_messages_sha256": hashlib.sha256(
            json.dumps(
                retry_messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "provider_request_sha256": _request_sha256(
            retry_messages,
            model="test-model",
            temperature=0.0,
            max_tokens=None,
        ),
    }

    audit = _audit_terminal_decision_requests(
        [],
        provider_attempts=[first_attempt, final_attempt],
        protocol_errors=[
            {
                **event,
                "provider_attempt_count": 2,
                "provider_metadata": {},
                "error": final_attempt["error"],
            }
        ],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=1,
    )

    assert audit["valid"] is False
    assert audit["decisions"][0]["checks"]["provider_attempt_failure_evidence_valid"] is False


@pytest.mark.parametrize("response_body", ["[]", "not-json"])
def test_request_audit_accepts_preserved_non_object_or_malformed_protocol_failure(response_body):
    terminal, event, attempt = _protocol_error_audit_fixture(response_body)

    audit = _audit_terminal_decision_requests(
        [],
        provider_attempts=[attempt],
        protocol_errors=[event],
        terminal=terminal,
        history_limit=24000,
        request_model="test-model",
        request_temperature=0.0,
        request_max_tokens=None,
        request_retries=0,
    )

    assert audit["valid"] is True


@pytest.mark.parametrize("status", [400, 401, 403])
def test_provider_configuration_error_is_never_a_formal_behavior_sample(status):
    error = ProviderConfigurationError(f"LLM HTTP {status}: provider rejected the request")

    summary = summarize_terminal_run(
        "Floor: 1\nChoose: ",
        decisions=0,
        format_violations=0,
        usage={},
        model_latency_sec=0.01,
        elapsed_sec=0.02,
        decision_limit_reached=False,
        error=f"{type(error).__name__}: {error}",
        termination_class="provider_configuration_error",
    )

    assert summary["formal_behavior_sample"] is False
    assert summary["completed"] is False


def test_frozen_record_is_reloaded_after_attempt(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    freeze_path = tmp_path / "results" / "source_lock.json"

    def tampering_runner(command, **kwargs):
        _write_success(command, cwd=kwargs["cwd"])
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        frozen["identity_sha256"] = "0" * 64
        freeze_path.write_text(json.dumps(frozen) + "\n", encoding="utf-8")
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=tampering_runner)
    persisted = json.loads(
        (tmp_path / "results" / "experiment_status.json").read_text(encoding="utf-8")
    )

    assert summary == persisted
    assert summary["status"] == "failed"
    assert {error["phase"] for error in summary["execution_errors"]} == {
        "post_run_revalidation",
        "final_source_lock_revalidation",
    }
    assert all("Source lock record is invalid" in error["error"] for error in summary["execution_errors"])
    sticky = json.loads(
        (tmp_path / "results" / "report_required.json").read_text(encoding="utf-8")
    )
    assert sticky == summary["issues_requiring_report"][0]


def test_source_lock_rejects_active_protocol_drift(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = load_protocol(protocol_path)
    cli_root = Path(protocol["sts2_cli_root"])
    out_root = tmp_path / "freeze"
    out_root.mkdir()
    initialize_or_verify_source_lock(out_root, protocol_path=protocol_path, sts2_cli_root=cli_root)

    protocol["purpose"] = "new protocol identity"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(DirectCliExperimentError, match="Source lock record is invalid|source file drifted"):
        initialize_or_verify_source_lock(out_root, protocol_path=protocol_path, sts2_cli_root=cli_root)


def test_resume_rejects_nonempty_output_without_source_lock(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    out_root = tmp_path / "results"
    out_root.mkdir()
    (out_root / "runs").mkdir()

    with pytest.raises(DirectCliExperimentError, match="has no source lock"):
        prepare_source_lock(protocol_path, resume=True)


def test_freeze_rejects_python_runtime_identity_drift(tmp_path, monkeypatch):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = load_protocol(protocol_path)
    cli_root = Path(protocol["sts2_cli_root"])
    out_root = tmp_path / "freeze"
    out_root.mkdir()
    initialize_or_verify_source_lock(out_root, protocol_path=protocol_path, sts2_cli_root=cli_root)
    monkeypatch.setattr(
        direct_cli_experiment,
        "_python_runtime_identity",
        lambda: {"python_implementation": "other", "python_version": "0.0.0"},
    )

    with pytest.raises(DirectCliExperimentError, match="Source lock record is invalid"):
        initialize_or_verify_source_lock(out_root, protocol_path=protocol_path, sts2_cli_root=cli_root)


def test_freeze_refuses_stale_prebuilt_headless_runtime(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = load_protocol(protocol_path)
    cli_root = Path(protocol["sts2_cli_root"])
    source = cli_root / "src" / "Sts2Headless" / "RunSimulator.cs"
    source.write_text("// changed after build\n", encoding="utf-8")
    headless = cli_root / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0" / "Sts2Headless.dll"
    stale_time = headless.stat().st_mtime + 10
    os.utime(source, (stale_time, stale_time))

    with pytest.raises(DirectCliExperimentError, match="Headless runtime is stale"):
        initialize_or_verify_source_lock(
            tmp_path / "freeze",
            protocol_path=protocol_path,
            sts2_cli_root=cli_root,
        )


def test_source_lock_refuses_stale_godot_stubs_and_tracks_their_sources(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = load_protocol(protocol_path)
    cli_root = Path(protocol["sts2_cli_root"])
    out_root = tmp_path / "freeze"

    frozen = initialize_or_verify_source_lock(
        out_root,
        protocol_path=protocol_path,
        sts2_cli_root=cli_root,
    )

    labels = {item["label"] for item in frozen["inventory"]}
    assert "sts2-cli/src/GodotStubs/GodotStubs.csproj" in labels
    assert "sts2-cli/src/GodotStubs/Core.cs" in labels
    assert not (out_root / "code_snapshot").exists()

    source = cli_root / "src" / "GodotStubs" / "Core.cs"
    stubs = cli_root / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0" / "GodotSharp.dll"
    stale_time = stubs.stat().st_mtime + 10
    os.utime(source, (stale_time, stale_time))

    with pytest.raises(DirectCliExperimentError, match="Headless runtime is stale"):
        initialize_or_verify_source_lock(
            tmp_path / "stale-freeze",
            protocol_path=protocol_path,
            sts2_cli_root=cli_root,
        )


def test_freeze_refuses_missing_godot_stubs_source_closure(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = load_protocol(protocol_path)
    cli_root = Path(protocol["sts2_cli_root"])
    shutil.rmtree(cli_root / "src" / "GodotStubs")

    with pytest.raises(DirectCliExperimentError, match="GodotStubs source closure is missing"):
        initialize_or_verify_source_lock(
            tmp_path / "freeze",
            protocol_path=protocol_path,
            sts2_cli_root=cli_root,
        )


def test_source_lock_rejects_inventory_that_skips_a_tracked_file(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    protocol = load_protocol(protocol_path)
    cli_root = Path(protocol["sts2_cli_root"])
    out_root = tmp_path / "freeze"
    out_root.mkdir()
    initialize_or_verify_source_lock(out_root, protocol_path=protocol_path, sts2_cli_root=cli_root)
    freeze_path = out_root / "source_lock.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen["inventory"][0]["tracked"] = False
    freeze_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(DirectCliExperimentError, match="Source lock record is invalid"):
        initialize_or_verify_source_lock(out_root, protocol_path=protocol_path, sts2_cli_root=cli_root)


def test_runner_parallelizes_seeds_but_serializes_replicates(tmp_path):
    protocol_path = _write_protocol(tmp_path, repeats=2)
    lock = threading.Lock()
    first_wave = threading.Barrier(3)
    active_seeds: set[str] = set()
    max_active = 0
    starts: list[tuple[str, str]] = []

    def fake_runner(command, **kwargs):
        nonlocal max_active
        seed = _argument(command, "--seed")
        artifacts = Path(_argument(command, "--out-dir"))
        replicate = next(part for part in artifacts.parts if part.startswith("replicate_"))
        assert Path(kwargs["cwd"]).resolve() == (
            Path(__file__).resolve().parents[1]
        ).resolve()
        assert Path(_argument(command, "--sts2-cli-root")).resolve() == (
            tmp_path / "sts2-cli"
        ).resolve()
        with lock:
            assert seed not in active_seeds
            active_seeds.add(seed)
            max_active = max(max_active, len(active_seeds))
            starts.append((seed, replicate))
        if replicate == "replicate_01":
            first_wave.wait(timeout=3)
        time.sleep(0.02)
        _write_success(command, cwd=kwargs["cwd"])
        with lock:
            active_seeds.remove(seed)
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=fake_runner)

    assert summary["status"] == "succeeded"
    assert summary["completed_runs"] == 6
    assert summary["needs_attention_runs"] == 0
    assert max_active == 3
    for seed in ("seed_a", "seed_b", "seed_c"):
        assert [replicate for started_seed, replicate in starts if started_seed == seed] == [
            "replicate_01",
            "replicate_02",
        ]


def test_failed_attempt_is_retained_and_retried(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1, max_attempts=2)
    calls = 0

    def flaky_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 17
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=flaky_runner)
    job = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"

    assert summary["status"] == "succeeded"
    assert calls == 2
    assert (job / "attempts" / "attempt_001" / "attempt_result.json").is_file()
    assert (job / "attempts" / "attempt_002" / "attempt_result.json").is_file()
    assert json.loads((job / "job_status.json").read_text(encoding="utf-8"))["attempt"] == 2


def test_malformed_artifact_after_formal_termination_is_retained_for_report_without_replay(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1, max_attempts=2)
    calls = 0

    def malformed_then_valid_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        if calls == 1:
            artifacts = Path(_argument(command, "--out-dir"))
            (artifacts / "terminal.txt").write_bytes(b"partial-utf8-\xe4")
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=malformed_then_valid_runner)
    job = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    first = json.loads(
        (job / "attempts" / "attempt_001" / "attempt_result.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "failed"
    assert calls == 1
    assert first["validation"]["checks"]["terminal_utf8_well_formed"] is False
    report = json.loads((job / "report_required.json").read_text(encoding="utf-8"))
    assert report["reason"] == "non_retryable_behavior_artifact_failure"


def test_engine_error_requires_report_and_is_not_retried(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1, max_attempts=2)
    calls = 0

    def error_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        engine_path = Path(_argument(command, "--out-dir")) / "engine_trace.jsonl"
        rows = [json.loads(line) for line in engine_path.read_text(encoding="utf-8").splitlines()]
        rows.insert(2, {"step": 2, "type": "state", "data": {"type": "error", "message": "bad action"}})
        engine_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=error_runner)
    job = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"

    assert summary["status"] == "failed"
    assert calls == 1
    report = json.loads((job / "report_required.json").read_text(encoding="utf-8"))
    assert report["reason"] == "engine_error_or_unsupported_decision"
    assert report["state_errors"][0]["state"]["message"] == "bad action"


def test_report_required_becomes_experiment_global_sticky_hold(tmp_path):
    protocol_path = _write_protocol(
        tmp_path,
        seeds=["seed_a", "seed_b", "seed_c"],
        repeats=2,
    )
    out_root = tmp_path / "results"
    calls = {"seed_a": 0, "seed_b": 0, "seed_c": 0}
    first_wave = threading.Barrier(3)

    def runner(command, **kwargs):
        seed = _argument(command, "--seed")
        calls[seed] += 1
        first_wave.wait(timeout=5)
        if seed == "seed_a":
            _write_success(command, cwd=kwargs["cwd"])
            engine_path = Path(_argument(command, "--out-dir")) / "engine_trace.jsonl"
            rows = [json.loads(line) for line in engine_path.read_text(encoding="utf-8").splitlines()]
            rows.insert(2, {"step": 2, "type": "state", "data": {"type": "error", "message": "bad action"}})
            engine_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            return 0
        deadline = time.monotonic() + 5
        while not (out_root / "report_required.json").is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (out_root / "report_required.json").is_file()
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=runner)

    assert summary["status"] == "failed"
    assert calls == {"seed_a": 1, "seed_b": 1, "seed_c": 1}
    assert len(summary["issues_requiring_report"]) == 1
    assert summary["issues_requiring_report"][0]["reason"] == "engine_error_or_unsupported_decision"
    assert all(lane["processed_replicates"] == 1 for lane in summary["lanes"])

    resumed = _run_prepared_experiment(
        protocol_path,
        runner=lambda *args, **kwargs: pytest.fail("global sticky hold must not launch a new attempt"),
    )
    assert resumed["status"] == "failed"
    assert resumed["completed_runs"] == summary["completed_runs"]
    assert len(resumed["all_attempts"]) == len(summary["all_attempts"])
    assert all(lane["processed_replicates"] == 1 for lane in resumed["lanes"])


def test_final_summary_aggregates_all_concurrent_job_reports(tmp_path):
    seeds = ["seed_a", "seed_b", "seed_c"]
    protocol_path = _write_protocol(tmp_path, seeds=seeds, repeats=1)
    first_wave = threading.Barrier(len(seeds))

    def error_runner(command, **kwargs):
        seed = _argument(command, "--seed")
        first_wave.wait(timeout=5)
        _write_success(command, cwd=kwargs["cwd"])
        engine_path = Path(_argument(command, "--out-dir")) / "engine_trace.jsonl"
        rows = [json.loads(line) for line in engine_path.read_text(encoding="utf-8").splitlines()]
        rows.insert(2, {"step": 2, "type": "state", "data": {"type": "error", "message": seed}})
        engine_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=error_runner)

    assert summary["status"] == "failed"
    assert isinstance(summary["experiment_hold"], dict)
    assert len(summary["issues_requiring_report"]) == 3
    assert {
        issue["state_errors"][0]["state"]["message"] for issue in summary["issues_requiring_report"]
    } == set(seeds)


def test_global_hold_prevents_another_lane_from_starting_an_in_job_retry(tmp_path):
    protocol_path = _write_protocol(
        tmp_path,
        seeds=["seed_a", "seed_b"],
        repeats=1,
        max_attempts=2,
    )
    out_root = tmp_path / "results"
    calls = {"seed_a": 0, "seed_b": 0}
    first_wave = threading.Barrier(2)

    def runner(command, **kwargs):
        seed = _argument(command, "--seed")
        calls[seed] += 1
        first_wave.wait(timeout=5)
        if seed == "seed_a":
            _write_success(command, cwd=kwargs["cwd"])
            engine_path = Path(_argument(command, "--out-dir")) / "engine_trace.jsonl"
            rows = [json.loads(line) for line in engine_path.read_text(encoding="utf-8").splitlines()]
            rows.insert(2, {"step": 2, "type": "state", "data": {"type": "error", "message": "bad action"}})
            engine_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            return 0
        deadline = time.monotonic() + 5
        while not (out_root / "report_required.json").is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (out_root / "report_required.json").is_file()
        return 1

    summary = _run_prepared_experiment(protocol_path, runner=runner)
    seed_b = next(lane for lane in summary["lanes"] if lane["seed"] == "seed_b")

    assert summary["status"] == "failed"
    assert calls == {"seed_a": 1, "seed_b": 1}
    assert len(seed_b["jobs"][0]["attempt_history"]) == 1
    assert seed_b["jobs"][0]["resume_action"] == "blocked_experiment_report_required"


def test_final_summary_retains_every_attempt_after_retry(tmp_path):
    protocol_path = _write_protocol(
        tmp_path,
        seeds=["seed_a"],
        repeats=1,
        max_attempts=2,
    )
    calls = 0

    def runner(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=runner)
    job = summary["lanes"][0]["jobs"][0]

    assert summary["status"] == "succeeded"
    assert calls == 2
    assert [attempt["attempt"] for attempt in job["attempt_history"]] == [1, 2]
    assert [attempt["attempt"] for attempt in summary["all_attempts"]] == [1, 2]
    assert summary["all_attempts"][0]["status"] == "failed"
    assert summary["all_attempts"][1]["status"] == "succeeded"


def test_resume_skips_completed_jobs(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    assert calls == 1
    status = json.loads(
        (tmp_path / "results" / "runs" / "seed_a" / "replicate_01" / "job_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "succeeded"


def test_resume_fails_closed_instead_of_replaying_completed_sample_with_drifted_report(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1, max_attempts=2)
    calls = 0

    def first_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=first_runner)["status"] == "succeeded"
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    replay_path = job_dir / "attempts" / "attempt_001" / "artifacts" / "trace_analysis" / "replay.md"
    replay_path.write_text(replay_path.read_text(encoding="utf-8") + "\ntampered\n", encoding="utf-8")

    resumed = _run_prepared_experiment(
        protocol_path,
        runner=lambda *args, **kwargs: pytest.fail("a completed behavior sample must never be replayed"),
    )

    assert calls == 1
    assert resumed["status"] == "failed"
    assert len(resumed["all_attempts"]) == 1
    job = resumed["lanes"][0]["jobs"][0]
    assert job["resume_action"] == "blocked_report_required"
    assert job["retryable"] is False
    assert (job_dir / "report_required.json").is_file()
    assert not (job_dir / "attempts" / "attempt_002").exists()


def test_resume_detects_coherent_report_and_manifest_rewrite_and_preserves_prior_acceptance(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1, max_attempts=2)
    calls = 0

    def first_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=first_runner)["status"] == "succeeded"
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    attempt_dir = job_dir / "attempts" / "attempt_001"
    result_path = attempt_dir / "attempt_result.json"
    prior_result_sha = hash_file(result_path)
    artifacts = attempt_dir / "artifacts"
    replay_path = artifacts / "trace_analysis" / "replay.md"
    replay_path.write_text(replay_path.read_text(encoding="utf-8") + "\ncoherent tamper\n", encoding="utf-8")
    summary_path = artifacts / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["analysis"]["analysis_artifact_manifest"] = build_analysis_artifact_manifest(
        artifacts / "trace_analysis"
    )
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    events_path = artifacts / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[-1] = {"type": "summary", **summary}
    events_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")

    resumed = _run_prepared_experiment(
        protocol_path,
        runner=lambda *args, **kwargs: pytest.fail("coherent report drift must not replay behavior"),
    )

    assert calls == 1
    assert resumed["status"] == "failed"
    job = resumed["lanes"][0]["jobs"][0]
    assert job["validation"]["analysis_checks"]["analysis_artifacts_match_deterministic_regeneration"] is False
    assert job["validation"]["checks"]["accepted_artifact_anchor_matches"] is False
    preserved = attempt_dir / f"attempt_result.accepted.{prior_result_sha}.json"
    assert preserved.is_file()
    assert hash_file(preserved) == prior_result_sha
    assert not (job_dir / "attempts" / "attempt_002").exists()


def test_resume_uses_raw_summary_to_avoid_replay_when_completed_attempt_result_is_missing(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1, max_attempts=2)
    calls = 0

    def first_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=first_runner)["status"] == "succeeded"
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    attempt_dir = job_dir / "attempts" / "attempt_001"
    (attempt_dir / "attempt_result.json").unlink()
    replay_path = attempt_dir / "artifacts" / "trace_analysis" / "replay.md"
    replay_path.write_text(replay_path.read_text(encoding="utf-8") + "\ntampered\n", encoding="utf-8")

    resumed = _run_prepared_experiment(
        protocol_path,
        runner=lambda *args, **kwargs: pytest.fail("raw completed summary must prevent behavior replay"),
    )

    assert calls == 1
    assert resumed["status"] == "failed"
    assert len(resumed["all_attempts"]) == 1
    assert (job_dir / "report_required.json").is_file()
    assert not (job_dir / "attempts" / "attempt_002").exists()


def test_resume_recovers_successful_attempt_when_job_status_write_was_lost(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    (job_dir / "job_status.json").unlink()

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    assert calls == 1
    recovered = json.loads((job_dir / "job_status.json").read_text(encoding="utf-8"))
    assert recovered["resume_action"] == "recovered_from_attempt_result"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_resume_reconstructs_attempt_result_from_authoritative_raw_artifacts(tmp_path, damage):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    attempt_dir = job_dir / "attempts" / "attempt_001"
    attempt_result = attempt_dir / "attempt_result.json"
    (job_dir / "job_status.json").unlink()
    if damage == "missing":
        attempt_result.unlink()
    else:
        attempt_result.write_text('{"broken":', encoding="utf-8")

    resumed = _run_prepared_experiment(protocol_path, runner=fake_runner)
    repaired = json.loads(attempt_result.read_text(encoding="utf-8"))

    assert resumed["status"] == "succeeded"
    assert calls == 1
    assert repaired["formal_sample"] is True
    assert repaired["recovery_source"] == "authoritative_child_artifacts"
    assert repaired["summary"]["termination_class"] == "natural_game_over"
    if damage == "corrupt":
        assert list(attempt_dir.glob("attempt_result.corrupt.*.json"))


def test_resume_preserves_model_protocol_behavior_sample_without_replay(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_model_protocol_failure(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    attempt_result = job_dir / "attempts" / "attempt_001" / "attempt_result.json"
    (job_dir / "job_status.json").unlink()
    attempt_result.unlink()

    resumed = _run_prepared_experiment(protocol_path, runner=fake_runner)
    repaired = json.loads(attempt_result.read_text(encoding="utf-8"))

    assert resumed["status"] == "succeeded"
    assert calls == 1
    assert repaired["termination_class"] == "model_protocol_error"
    assert repaired["formal_sample"] is True
    assert repaired["recovery_source"] == "authoritative_child_artifacts"


def test_live_attempt_lease_blocks_duplicate_direct_job(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    prepare_source_lock(protocol_path, resume=False)
    out_root = tmp_path / "results"
    frozen = json.loads((out_root / "source_lock.json").read_text(encoding="utf-8"))
    protocol = load_protocol(protocol_path)
    job_dir = out_root / "runs" / "seed_a" / "replicate_01"
    direct_cli_experiment._atomic_write_json(
        job_dir / "active_attempt.json",
        {"owner_pid": os.getpid(), "child_pid": None, "attempt": 1},
    )
    calls = []

    summary = _run_source_locked_experiment_core(
        protocol=protocol,
        protocol_path=protocol_path,
        out_root=out_root,
        source_lock_record=frozen,
        runner=lambda *args, **kwargs: calls.append(args),
    )

    assert summary["status"] == "failed"
    assert summary["lanes"][0]["jobs"][0]["status"] == "active_attempt_detected"
    assert calls == []


def test_lane_exception_persists_fail_closed_final_status_and_original_error(tmp_path, monkeypatch):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)

    def fail_lane(*_args, **_kwargs):
        raise OSError("simulated lane control failure")

    monkeypatch.setattr(direct_cli_experiment, "_run_seed_lane", fail_lane)
    summary = _run_prepared_experiment(
        protocol_path,
        runner=lambda *_args, **_kwargs: pytest.fail("lane failure must occur before child execution"),
    )
    out_root = tmp_path / "results"
    persisted_summary = json.loads((out_root / "experiment_summary.json").read_text(encoding="utf-8"))
    persisted_status = json.loads((out_root / "experiment_status.json").read_text(encoding="utf-8"))
    sticky = json.loads((out_root / "report_required.json").read_text(encoding="utf-8"))

    assert summary == persisted_summary == persisted_status
    assert summary["status"] == "failed"
    assert summary["completed_runs"] == 0
    assert summary["execution_errors"][0]["phase"] == "seed_lane"
    assert summary["execution_errors"][0]["error_type"] == "OSError"
    assert summary["execution_errors"][0]["error"] == "simulated lane control failure"
    assert "fail_lane" in summary["execution_errors"][0]["traceback"]
    assert sticky["reason"] == "lane_execution_exception"
    assert sticky["execution_error"]["error"] == "simulated lane control failure"


def test_dead_attempt_lease_is_marked_orphaned_before_resume(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    prepare_source_lock(protocol_path, resume=False)
    out_root = tmp_path / "results"
    job_dir = out_root / "runs" / "seed_a" / "replicate_01"
    direct_cli_experiment._atomic_write_json(
        job_dir / "active_attempt.json",
        {"owner_pid": 2147483647, "child_pid": 2147483646, "attempt": 1},
    )
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    summary = _run_prepared_experiment(protocol_path, runner=fake_runner)

    assert summary["status"] == "succeeded"
    assert calls == 1
    orphan = json.loads((job_dir / "orphaned_attempt.json").read_text(encoding="utf-8"))
    assert orphan["state"] == "orphaned_processes_gone"
    assert not (job_dir / "active_attempt.json").exists()


def test_subprocess_runner_records_child_pid_in_attempt_lease(tmp_path):
    lease_path = tmp_path / "active_attempt.json"
    direct_cli_experiment._atomic_write_json(
        lease_path,
        {"owner_pid": os.getpid(), "child_pid": None, "state": "starting"},
    )

    returncode = direct_cli_experiment._subprocess_runner(
        [sys.executable, "-c", "print('ok')"],
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        cwd=tmp_path,
        timeout_sec=10,
        lease_path=lease_path,
    )
    lease = json.loads(lease_path.read_text(encoding="utf-8"))

    assert returncode == 0
    assert isinstance(lease["child_pid"], int)
    assert lease["child_pid"] > 0
    assert lease["state"] == "running"


def test_subprocess_runner_kills_child_if_lease_update_fails(tmp_path, monkeypatch):
    lease_path = tmp_path / "active_attempt.json"
    direct_cli_experiment._atomic_write_json(
        lease_path,
        {"owner_pid": os.getpid(), "child_pid": None, "state": "starting"},
    )

    class FakeProcess:
        pid = 424242

        @staticmethod
        def poll():
            return None

    process = FakeProcess()
    terminated = []
    original_write = direct_cli_experiment._atomic_write_json

    def failing_write(path, value):
        if path == lease_path and value.get("state") == "running":
            raise OSError("injected lease write failure")
        return original_write(path, value)

    monkeypatch.setattr(direct_cli_experiment.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(direct_cli_experiment, "_atomic_write_json", failing_write)
    monkeypatch.setattr(direct_cli_experiment, "_terminate_process_tree", terminated.append)

    with pytest.raises(OSError, match="injected lease write failure"):
        direct_cli_experiment._subprocess_runner(
            [sys.executable, "-c", "print('never executed')"],
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            cwd=tmp_path,
            timeout_sec=10,
            lease_path=lease_path,
        )

    assert terminated == [process]


def test_subprocess_runner_wait_failure_reaps_tree_and_preserves_live_child_lease(tmp_path, monkeypatch):
    lease_path = tmp_path / "active_attempt.json"
    direct_cli_experiment._atomic_write_json(
        lease_path,
        {"owner_pid": os.getpid(), "child_pid": None, "state": "starting"},
    )

    class FakeProcess:
        pid = 424242

        @staticmethod
        def wait(timeout):
            raise OSError("synthetic wait failure")

    process = FakeProcess()
    terminated = []
    monkeypatch.setattr(direct_cli_experiment.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        direct_cli_experiment,
        "_terminate_process_tree",
        lambda child: terminated.append(child) or False,
    )

    with pytest.raises(OSError, match="synthetic wait failure"):
        direct_cli_experiment._subprocess_runner(
            ["noop"],
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            cwd=tmp_path,
            timeout_sec=1.0,
            lease_path=lease_path,
        )

    monkeypatch.setattr(direct_cli_experiment, "_pid_alive", lambda pid: pid == process.pid)
    direct_cli_experiment._release_attempt_lease(lease_path)
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    assert terminated == [process]
    assert lease["child_pid"] == process.pid
    assert lease["state"] == "cleanup_required"


def test_subprocess_runner_timeout_reaps_tree_and_returns_124(tmp_path, monkeypatch):
    lease_path = tmp_path / "active_attempt.json"
    direct_cli_experiment._atomic_write_json(
        lease_path,
        {"owner_pid": os.getpid(), "child_pid": None, "state": "starting"},
    )

    class FakeProcess:
        pid = 424242

        @staticmethod
        def wait(timeout):
            raise direct_cli_experiment.subprocess.TimeoutExpired("synthetic", timeout)

    process = FakeProcess()
    terminated = []
    monkeypatch.setattr(direct_cli_experiment.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        direct_cli_experiment,
        "_terminate_process_tree",
        lambda child: terminated.append(child) or True,
    )

    returncode = direct_cli_experiment._subprocess_runner(
        ["noop"],
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        cwd=tmp_path,
        timeout_sec=1.0,
        lease_path=lease_path,
    )

    assert returncode == 124
    assert terminated == [process]
    assert "timed out after 1s" in (tmp_path / "stderr.log").read_text(encoding="utf-8")


def test_resume_never_bypasses_historical_report_required(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    (job_dir / "report_required.json").write_text(
        json.dumps({"reason": "manual review remains required"}) + "\n",
        encoding="utf-8",
    )

    resumed = _run_prepared_experiment(protocol_path, runner=fake_runner)
    assert resumed["status"] == "failed"
    assert calls == 1
    assert resumed["lanes"][0]["jobs"][0]["resume_action"] == "blocked_report_required"


def test_corrupt_job_report_fails_closed_without_losing_attempt_history(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    report_path = job_dir / "report_required.json"
    report_path.write_text('{"broken":', encoding="utf-8")

    resumed = _run_prepared_experiment(
        protocol_path,
        runner=lambda *args, **kwargs: pytest.fail("corrupt sticky report must block execution"),
    )

    assert resumed["status"] == "failed"
    assert calls == 1
    assert len(resumed["all_attempts"]) == 1
    assert resumed["issues_requiring_report"][0]["reason"] == "corrupt_job_report_required"
    assert report_path.read_text(encoding="utf-8") == '{"broken":'


def test_analysis_failure_after_natural_terminal_is_kept_without_replay(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1, max_attempts=2)
    calls = 0

    def report_failure_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        _make_analysis_pending(command)
        return 0

    result = _run_prepared_experiment(protocol_path, runner=report_failure_runner)
    job = result["lanes"][0]["jobs"][0]

    assert result["status"] == "succeeded"
    assert result["needs_attention_runs"] == 1
    assert result["analysis_complete_runs"] == 0
    assert calls == 1
    assert job["termination_class"] == "natural_game_over"
    assert job["analysis_pending"] is True
    assert job["needs_attention"] is True
    assert job["report_required"] is False


def test_present_contradictory_benchmark_trace_fails_closed(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)

    def contradictory_report_runner(command, **kwargs):
        _write_success(command, cwd=kwargs["cwd"])
        artifacts = Path(_argument(command, "--out-dir"))
        benchmark_trace = artifacts / "benchmark_trace.jsonl"
        rows = [json.loads(line) for line in benchmark_trace.read_text(encoding="utf-8").splitlines()]
        rows[-1]["result"]["extra"]["authoritative_terminal_state"]["tampered"] = True
        benchmark_trace.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary_path = artifacts / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        engine_audit = summary["analysis"]["engine_audit"]
        benchmark_audit = audit_benchmark_trace(benchmark_trace, engine_audit=engine_audit)
        assert benchmark_audit["valid"] is False
        summary["analysis"]["benchmark_audit"] = benchmark_audit
        summary["analysis"]["benchmark_trace_sha256"] = hash_file(benchmark_trace)
        summary.update(
            {
                "analysis_error": "trace closure validation failed: injected contradiction",
                "analysis_pending": True,
                "needs_attention": True,
            }
        )
        summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        events_path = artifacts / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        events[-1] = {"type": "summary", **summary}
        events_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
        return 0

    result = _run_prepared_experiment(protocol_path, runner=contradictory_report_runner)
    job = result["lanes"][0]["jobs"][0]

    assert result["status"] == "failed"
    assert job["formal_sample"] is False
    assert job["validation"]["checks"]["present_benchmark_trace_authoritative_closure"] is False
    assert job["report_required"]["reason"] == "non_retryable_behavior_artifact_failure"


def test_tampered_analysis_artifact_is_not_counted_complete_or_formal(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)

    def tampered_report_runner(command, **kwargs):
        _write_success(command, cwd=kwargs["cwd"])
        artifacts = Path(_argument(command, "--out-dir"))
        replay = artifacts / "trace_analysis" / "replay.md"
        replay.write_text(replay.read_text(encoding="utf-8") + "\ntampered\n", encoding="utf-8")
        return 0

    result = _run_prepared_experiment(protocol_path, runner=tampered_report_runner)
    job = result["lanes"][0]["jobs"][0]

    assert result["status"] == "failed"
    assert result["analysis_complete_runs"] == 0
    assert result["needs_attention_runs"] == 0
    assert job["formal_sample"] is False
    assert job["validation"]["analysis_checks"]["analysis_artifact_manifest_matches"] is False


def test_model_protocol_failure_counts_once_and_lane_continues_to_next_replicate(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=2, max_attempts=2)
    calls = 0

    def protocol_failure_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_model_protocol_failure(command, cwd=kwargs["cwd"])
        return 0

    result = _run_prepared_experiment(protocol_path, runner=protocol_failure_runner)
    job = result["lanes"][0]["jobs"][0]

    assert result["status"] == "succeeded"
    assert calls == 2
    assert result["completed_runs"] == 2
    assert not (tmp_path / "results" / "report_required.json").exists()
    for job in result["lanes"][0]["jobs"]:
        assert job["termination_class"] == "model_protocol_error"
        assert job["formal_sample"] is True
        assert job["retryable"] is False
        assert job["validation"]["request_audit"]["protocol_error_count"] == 1


def test_localization_warning_invalidates_a_formal_attempt(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)

    def missing_localization_runner(command, **kwargs):
        _write_success(command, cwd=kwargs["cwd"])
        kwargs["stderr_path"].write_text(
            "[WARN] Localization dir not found: C:\\missing\\localization_eng\n",
            encoding="utf-8",
        )
        return 0

    result = _run_prepared_experiment(protocol_path, runner=missing_localization_runner)
    attempt = result["all_attempts"][0]

    assert result["status"] == "failed"
    assert attempt["validation"]["checks"]["model_visible_localization_complete"] is False
    assert attempt["validation"]["localization_audit"]["missing_directory_warning"] is True


def test_provider_event_reordered_after_model_decision_is_rejected_without_replay(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1, max_attempts=2)
    calls = 0

    def reordered_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        events_path = Path(_argument(command, "--out-dir")) / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        events[0], events[1] = events[1], events[0]
        events_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
        return 0

    result = _run_prepared_experiment(protocol_path, runner=reordered_runner)
    job = result["lanes"][0]["jobs"][0]

    assert result["status"] == "failed"
    assert calls == 1
    assert job["validation"]["request_audit"]["event_order_valid"] is False
    assert job["report_required"] is not False


def test_resume_fails_closed_without_replaying_tampered_completed_prompt_artifact(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=1)
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    config_path = (
        tmp_path
        / "results"
        / "runs"
        / "seed_a"
        / "replicate_01"
        / "attempts"
        / "attempt_001"
        / "artifacts"
        / "config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["system_prompt"] += "\ntampered"
    config["system_prompt_sha256"] = hashlib.sha256(config["system_prompt"].encode("utf-8")).hexdigest()
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    resumed = _run_prepared_experiment(protocol_path, runner=fake_runner)
    assert resumed["status"] == "failed"
    assert calls == 1
    job_dir = tmp_path / "results" / "runs" / "seed_a" / "replicate_01"
    assert (job_dir / "report_required.json").is_file()
    assert not (job_dir / "attempts" / "attempt_002").exists()


def test_resume_fails_closed_without_replaying_completed_job_copied_to_another_replicate(tmp_path):
    protocol_path = _write_protocol(tmp_path, seeds=["seed_a"], repeats=2)
    calls = 0

    def fake_runner(command, **kwargs):
        nonlocal calls
        calls += 1
        _write_success(command, cwd=kwargs["cwd"])
        return 0

    assert _run_prepared_experiment(protocol_path, runner=fake_runner)["status"] == "succeeded"
    runs = tmp_path / "results" / "runs" / "seed_a"
    shutil.rmtree(runs / "replicate_02")
    shutil.copytree(runs / "replicate_01", runs / "replicate_02")

    resumed = _run_prepared_experiment(protocol_path, runner=fake_runner)
    assert resumed["status"] == "failed"
    assert calls == 2
    status = json.loads((runs / "replicate_02" / "job_status.json").read_text(encoding="utf-8"))
    assert status["replicate"] == 2
    assert status["attempt"] == 1
    assert status["resume_action"] == "blocked_report_required"
    assert not (runs / "replicate_02" / "attempts" / "attempt_002").exists()
