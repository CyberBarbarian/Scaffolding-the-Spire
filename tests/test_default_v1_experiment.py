import copy
import hashlib
import io
import json
import shutil
import threading
import urllib.error
from pathlib import Path

import pytest

import sts2_bench.default_v1_experiment as experiment
from sts2_bench.agents import (
    FORMAL_PROVIDER_AUDIT_ENV,
    OpenAICompatAgent,
    build_formal_generalist_system_message,
)
from sts2_bench.experiment_identity import manifest_path_for_trace
from sts2_bench.game_rules import (
    full_game_rules_meta,
    game_rules_identity,
    game_rules_meta,
    render_game_rules,
)
from sts2_bench.default_v1_source_lock import validate_protocol


def _protocol(*, repeats: int = 1, attempts: int = 3) -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "default_v1_rules_v2_posterior_r10_formal_20260725.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["experiment_id"] = "default-v1-test"
    option_index = protocol["common_args"].index("--experiment-id")
    protocol["common_args"][option_index + 1] = "default-v1-test"
    protocol["comparison"]["repeats_per_seed"] = repeats
    protocol["schedule"]["max_attempts_per_invocation"] = attempts
    protocol["schedule"]["run_timeout_sec"] = 60
    return protocol


def _argument(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _generic_protocol(tmp_path: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    template = (
        root
        / "experiments"
        / "ablation_combat_state_only_defect_template_20260726.json"
    )
    protocol = json.loads(template.read_text(encoding="utf-8"))
    protocol["template_status"] = "formal_execution_ready"
    protocol["comparison"] = {
        "baseline": "formal-generalist-defect-template-20260726",
        "matched_character": "Defect",
        "matched_ascension": 0,
        "matched_seeds": [
            "ironclad_a0_3407",
            "ironclad_a0_1007",
            "ironclad_a0_2407",
        ],
        "repeats_per_seed": 5,
        "total_runs": 15,
    }
    protocol["schedule"] = {
        "workers": 3,
        "parallel_unit": "seed_lane",
        "between_seed_lanes": "parallel",
        "within_seed_lane": "strictly_serial",
        "max_attempts_per_invocation": 3,
        "run_timeout_sec": 7200,
        "resume_unit": "seed_replicate",
    }
    path = tmp_path / "generic-formal.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return validate_protocol(path)


def test_generic_attempt_command_derives_locked_run_and_scenario_ids(
    tmp_path: Path,
) -> None:
    protocol = _generic_protocol(tmp_path)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    command = experiment._attempt_command(
        protocol,
        lane_name="seed-3407",
        seed="ironclad_a0_3407",
        replicate=1,
        attempt_dir=attempt_dir,
    )

    assert _argument(command, "--scenario-id") == "defect-a0-seed-3407"
    assert _argument(command, "--run-id") == (
        "ablation-combat-state-only-defect-template-20260726"
        "--seed-3407--replicate-01"
    )


def test_generic_resolved_config_audit_is_protocol_driven(tmp_path: Path) -> None:
    protocol = _generic_protocol(tmp_path)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    command = experiment._attempt_command(
        protocol,
        lane_name="seed-3407",
        seed="ironclad_a0_3407",
        replicate=1,
        attempt_dir=attempt_dir,
    )
    probe = experiment._validate_resolved_run_config(
        {},
        protocol=protocol,
        command=command,
        seed="ironclad_a0_3407",
        attempt_dir=attempt_dir,
    )
    run_config: dict = {}
    for dotted, value in probe["expected"].items():
        if dotted == "argv":
            run_config["argv"] = value
            continue
        current = run_config
        parts = dotted.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    valid = experiment._validate_resolved_run_config(
        {"run_config": run_config},
        protocol=protocol,
        command=command,
        seed="ironclad_a0_3407",
        attempt_dir=attempt_dir,
    )
    run_config["agent"]["rule_projection_and_recent_changes"].pop("enabled")
    missing_treatment_field = experiment._validate_resolved_run_config(
        {"run_config": run_config},
        protocol=protocol,
        command=command,
        seed="ironclad_a0_3407",
        attempt_dir=attempt_dir,
    )
    run_config["agent"]["rule_projection_and_recent_changes"]["enabled"] = True
    run_config["agent"]["model"] = "drifted-model"
    drifted = experiment._validate_resolved_run_config(
        {"run_config": run_config},
        protocol=protocol,
        command=command,
        seed="ironclad_a0_3407",
        attempt_dir=attempt_dir,
    )

    assert valid["valid"] is True
    assert missing_treatment_field["valid"] is False
    assert (
        missing_treatment_field["checks"][
            "agent.rule_projection_and_recent_changes.enabled"
        ]
        is False
    )
    assert drifted["valid"] is False
    assert drifted["checks"]["agent.model"] is False


def _model_request_audits_meta(
    *requests: list[dict[str, str]],
    provider_usages: list[dict] | None = None,
    provider_latencies: list[float] | None = None,
) -> dict:
    audits = []
    raw_response = '{"action_id":0,"reason":"test"}'
    default_usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    final_metadata = {}
    final_usage = default_usage
    for request_index, messages in enumerate(requests):
        usage = (
            provider_usages[request_index]
            if provider_usages is not None
            else default_usage
        )
        latency = (
            provider_latencies[request_index]
            if provider_latencies is not None
            else 0.0
        )
        payload = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        response = {
            "id": f"request-{request_index}",
            "model": "deepseek-v4-flash",
            "created": 1234567890 + request_index,
            "system_fingerprint": "test-fingerprint",
            "choices": [{"message": {"content": raw_response}, "finish_reason": "stop"}],
            "usage": usage,
        }
        response_body = json.dumps(response, separators=(",", ":"))
        final_metadata = {
            "model": response["model"],
            "id": response["id"],
            "created": response["created"],
            "system_fingerprint": response["system_fingerprint"],
            "finish_reason": "stop",
        }
        final_usage = usage
        audits.append(
            {
                "request_index": request_index,
                "messages": messages,
                "messages_sha256": hashlib.sha256(payload).hexdigest(),
                "provider_response_status": "succeeded",
                "provider_response_body": response_body,
                "provider_response_sha256": hashlib.sha256(response_body.encode("utf-8")).hexdigest(),
                "provider_http_status": 200,
                "provider_finish_reason": "stop",
                "provider_metadata": final_metadata,
                "provider_usage": usage,
                "provider_latency_sec": latency,
            }
        )
    result = {
        "model_request_audits": audits,
        "raw_response": raw_response,
        "usage": final_usage,
        "provider_finish_reason": "stop",
        "provider_metadata": final_metadata,
    }
    if len(audits) > 1:
        result["previous_raw_responses"] = [raw_response] * (len(audits) - 1)
    return result


def _write_attempt_artifacts(
    command: list[str],
    *,
    result: dict,
    manifest_status: str,
) -> None:
    trace_path = Path(_argument(command, "--out"))
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "type": "decision",
        "decision": "combat_play",
        "context": {"act": 1, "floor": 1, "room_type": "Monster"},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck": []},
        "combat": {"hand": [], "enemies": []},
    }
    prompt = "Choose one legal combat action.\n\n" + render_game_rules(state)
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": prompt},
    ]
    trace_path.write_text(
        json.dumps(
            {
                "type": "decision",
                "step": 0,
                "decision": "combat_play",
                "authoritative_state": state,
                "prompt": prompt,
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "type": "action",
                "step": 0,
                "decision": "combat_play",
                "agent_meta": {
                    "agent": "openai_compat",
                    "model": "deepseek-v4-flash",
                    "json_response_format": False,
                    **game_rules_meta(state, prompt=prompt),
                    **_model_request_audits_meta(messages),
                },
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps({"type": "result", "result": result}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    identity = {
        "experiment_id": "default-v1-test",
        "scenario_id": _argument(command, "--scenario-id"),
        "replicate_id": _argument(command, "--replicate-id"),
        "run_id": _argument(command, "--run-id"),
    }
    run_config = {
        "argv": command[7:],
        "agent": {
            "type": "coordinator",
            "base_url": "https://zgc.apihy.com/v1",
            "model": "deepseek-v4-flash",
            "llm_timeout": 180.0,
            "llm_max_retries": 1,
            "llm_fallback_mode": "error",
            "llm_temperature": 0.0,
            "prompt_style": "analysis",
            "include_json_state": False,
            "short_term_memory": {
                "scope": "episode_local_action_history",
                "enabled": False,
                "window": 8,
                "mode": "factual_diff",
            },
            "run_summary_enabled": False,
            "context_management": {
                "mode": "coordinator",
                "route_candidate_planner": {"objective": "none", "llm_hint": "none"},
                "ca": {
                    "mode": "combat_mdp_state",
                    "turn_process": "structured_current_turn_actions",
                    "combat_reasoning_pool": True,
                    "combat_turn_board": True,
                    "combat_skill_hints": {"enabled": False, "profile": "baseline", "max_hints": 4},
                },
                "occ": {
                    "mode": "reference_turn",
                    "run_context": "rm_rp_slice",
                    "reward_posterior_evaluator": True,
                    "reference_context": {"window": 10, "source": "rm_rp_slice_plus_short_occ_history"},
                },
            },
            "route_plan": {"enabled": True, "enforcement": "hard", "route_map_prompt_format": "canonical"},
            "diagnostics": {
                "self_consistency": {
                    "enabled": False,
                    "samples": 3,
                    "decisions": "map_select,card_reward",
                    "mode": "diagnostic",
                },
                "action_prompt_order": {"mode": "default", "seed": 0},
            },
        },
        "game": {"character": "Ironclad", "ascension": 0, "lang": "en"},
        "benchmark": {
            "seeds_file": _argument(command, "--seeds"),
            "seeds": [result["seed"]],
            "count": 1,
            "setup_actions_file": None,
            "setup_action_count": 0,
            "max_steps": 2000,
            "max_game_actions": 0,
            "include_full_map": True,
            "allow_repeat_views": False,
            "allow_custom_sequences": False,
            "custom_sequence_exposure": "safe",
            "combat_card_sequences": False,
            "precomputed_combat_card_sequences": False,
            "combat_reasoning_pool": True,
            "combat_turn_board": True,
            "combat_skill_hints": {"enabled": False, "profile": "baseline", "max_hints": 4},
            "combat_skill_lines": False,
        },
        "logging": {
            "out": _argument(command, "--out"),
            "overwrite": False,
            "log_level": "full",
            "print_prompts": False,
            "print_model_output": False,
            "trace_analysis": True,
            "trace_analysis_dir": _argument(command, "--trace-analysis-dir"),
        },
    }
    analysis_dir = Path(_argument(command, "--trace-analysis-dir"))
    analysis_index = {
        "source": str(trace_path),
        "out_dir": str(analysis_dir),
        "run_count": 1,
        "runs": [{"seed": result["seed"]}],
    }
    manifest_path_for_trace(trace_path).write_text(
        json.dumps(
            {
                "status": manifest_status,
                "identity": identity,
                "run_config": run_config,
                "results": [result],
                "trace_analysis": analysis_index,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "index.json").write_text(
        json.dumps(analysis_index, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _natural_result(seed: str = "seed-a") -> dict:
    terminal = {
        "decision": "game_over",
        "victory": False,
        "context": {"act": 1, "floor": 7, "room_type": "Monster"},
        "player": {
            "hp": 0,
            "max_hp": 80,
            "gold": 99,
            "deck_size": 1,
            "deck": [{"id": "strike", "name": "Strike", "upgraded": False}],
            "relics": [],
            "potions": [],
        },
    }
    return {
        "seed": seed,
        "decision": "game_over",
        "victory": False,
        "act": 1,
        "floor": 7,
        "room_type": "Monster",
        "hp": 0,
        "max_hp": 80,
        "gold": 99,
        "deck_size": 1,
        "error": None,
        "truncated": False,
        "extra": {"authoritative_terminal_state": terminal},
    }


def _run_job(
    tmp_path,
    monkeypatch,
    runner,
    *,
    attempts: int = 3,
    lane_name: str = "seed-a-lane",
    seed: str = "seed-a",
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(exist_ok=True)
    out_root = tmp_path / "results"
    monkeypatch.setattr(
        experiment,
        "_verify_expected_identity",
        lambda _snapshot, identity: {"identity_sha256": identity},
    )
    return experiment._run_job(
        _protocol(attempts=attempts),
        benchmark_root=snapshot,
        lock_root=snapshot,
        out_root=out_root,
        frozen_identity="frozen-test",
        identity_lock=threading.Lock(),
        lane_name=lane_name,
        seed=seed,
        replicate=1,
        runner=runner,
    )


def test_run_job_resumes_only_after_revalidating_completed_artifacts(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("ok\n", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    first = _run_job(tmp_path, monkeypatch, runner)
    second = _run_job(tmp_path, monkeypatch, runner)

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert second["resume_action"] == "skipped_completed"
    assert len(calls) == 1

    trace_path = Path(_argument(calls[0], "--out"))
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "decision", "decision": "combat_play"}) + "\n")

    third = _run_job(tmp_path, monkeypatch, runner)

    assert third["status"] == "needs_attention"
    assert third["formal_sample"] is False
    assert len(calls) == 1


def test_run_job_rejects_completed_attempt_copied_from_another_lane(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("ok\n", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        seed = json.loads(Path(_argument(command, "--seeds")).read_text(encoding="utf-8"))[0]
        _write_attempt_artifacts(
            command,
            result=_natural_result(seed),
            manifest_status="succeeded",
        )
        return 0

    first = _run_job(tmp_path, monkeypatch, runner)
    source = tmp_path / "results" / "runs" / "seed-a" / "replicate_01"
    destination = tmp_path / "results" / "runs" / "seed-b" / "replicate_01"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)

    second = _run_job(
        tmp_path,
        monkeypatch,
        runner,
        lane_name="seed-b-lane",
        seed="seed-b",
    )

    assert first["status"] == "succeeded"
    assert second["status"] == "needs_attention"
    assert second["formal_sample"] is False
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("protocol_sha256", "0" * 64),
        ("game_rules", {"ruleset_id": "wrong-rules"}),
        ("job_relative_path", "runs/another-seed/replicate_01"),
    ],
)
def test_run_job_revalidates_attempt_manifest_job_binding(
    tmp_path,
    monkeypatch,
    field,
    invalid_value,
):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("ok\n", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    first = _run_job(tmp_path, monkeypatch, runner)
    attempt_manifest_path = (
        tmp_path
        / "results"
        / "runs"
        / "seed-a"
        / "replicate_01"
        / "attempts"
        / "attempt_001"
        / "attempt_manifest.json"
    )
    attempt_manifest = json.loads(attempt_manifest_path.read_text(encoding="utf-8"))
    attempt_manifest[field] = invalid_value
    attempt_manifest_path.write_text(
        json.dumps(attempt_manifest, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    second = _run_job(tmp_path, monkeypatch, runner)

    assert first["status"] == "succeeded"
    assert second["status"] == "needs_attention"
    assert second["formal_sample"] is False
    assert len(calls) == 1


def test_attempt_command_uses_rules_v2_run_id(tmp_path):
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    command = experiment._attempt_command(
        _protocol(),
        lane_name="seed-a-lane",
        seed="seed-a",
        replicate=1,
        attempt_dir=attempt_dir,
    )

    assert _argument(command, "--run-id") == "default-v1-rules-v2--seed-a-lane--replicate-01"


def test_trace_rule_exposure_requires_prompt_level_audit_fields(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    state = {
        "type": "decision",
        "decision": "map_select",
        "context": {"act": 1, "floor": 1, "room_type": "Map"},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck": []},
    }
    prompt = "Choose a legal route action.\n\n" + render_game_rules(state)
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": prompt},
    ]
    decision = {
        "type": "decision",
        "step": 0,
        "decision": "map_select",
        "authoritative_state": state,
        "prompt": prompt,
    }
    action = {
        "type": "action",
        "step": 0,
        "decision": "map_select",
        "agent_meta": {
            **game_rules_meta(state, prompt=prompt),
            **_model_request_audits_meta(messages),
        },
    }
    trace_path.write_text(json.dumps(decision) + "\n" + json.dumps(action) + "\n", encoding="utf-8")
    expected_rules = {
        **game_rules_identity(),
        "default_delivery": "state_selected_prompt",
    }

    trace = experiment._read_trace_jsonl(trace_path)
    assert experiment._trace_rule_exposure(trace, expected_rules)["valid"] is True

    action["agent_meta"]["model_request_audits"][0]["messages_sha256"] = "b" * 64
    trace_path.write_text(json.dumps(decision) + "\n" + json.dumps(action) + "\n", encoding="utf-8")

    trace = experiment._read_trace_jsonl(trace_path)
    assert experiment._trace_rule_exposure(trace, expected_rules)["valid"] is False


def test_trace_rule_exposure_accepts_static_formal_generalist_rules() -> None:
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 1, "room_type": "Monster"},
        "player": {"hp": 70, "max_hp": 70, "gold": 99, "deck": []},
        "combat": {"hand": [], "enemies": []},
    }
    prompt = "Choose one current legal action."
    messages = [
        {"role": "system", "content": build_formal_generalist_system_message()},
        {"role": "user", "content": prompt},
    ]
    preview = "\n".join(
        [
            "===== message[0] role=system =====",
            messages[0]["content"],
            "===== message[1] role=user =====",
            messages[1]["content"],
        ]
    )
    decision = {
        "type": "decision",
        "step": 0,
        "decision": "combat_play",
        "authoritative_state": state,
        "prompt": prompt,
    }
    action = {
        "type": "action",
        "step": 0,
        "decision": "combat_play",
        "agent_meta": {
            **full_game_rules_meta(preview),
            "rule_projection_and_recent_changes_enabled": False,
            "recent_state_changes_delivery": "omitted",
            **_model_request_audits_meta(messages),
        },
    }

    audit = experiment._trace_rule_exposure(
        {"rows": [decision, action]},
        game_rules_identity(),
        projection_enabled=False,
        rules_delivery="static_full_system_prompt",
    )
    wrong_delivery = experiment._trace_rule_exposure(
        {"rows": [decision, action]},
        game_rules_identity(),
        projection_enabled=False,
        rules_delivery="state_selected_prompt",
    )

    assert audit["valid"] is True
    assert audit["request_count"] == 1
    assert wrong_delivery["valid"] is False


def _runtime_takeover_rows() -> tuple[dict, dict]:
    choice = {"col": 2, "row": 3, "type": "Monster"}
    map_action = {
        "action_id": 5,
        "kind": "map",
        "label": "go to Monster at col=2 row=3",
        "cmd": {
            "cmd": "action",
            "action": "select_map_node",
            "args": {"col": 2, "row": 3},
        },
    }
    decision = {
        "type": "decision",
        "step": 1,
        "decision": "map_select",
        "authoritative_state": {
            "type": "decision",
            "decision": "map_select",
            "choices": [choice],
        },
        "legal_actions": [
            {
                "action_id": 0,
                "kind": "view_map",
                "label": "view map and current position",
                "cmd": {"cmd": "bench_view", "view": "map"},
            },
            map_action,
        ],
        "runtime_takeover": {
            "mode": "forced_singleton_map",
            "action": map_action,
            "model_called": False,
        },
    }
    action = {
        "type": "action",
        "step": 1,
        "decision": "map_select",
        "command": map_action["cmd"],
        "action": map_action,
        "agent_meta": {
            "agent": "runtime",
            "active_agent": "runtime",
            "policy": "forced_singleton_map",
            "model_called": False,
            "route_plan_effect": {
                "mode": "forced_singleton_map",
                "selected_node": choice,
                "outcome": "no_active_plan",
            },
        },
    }
    return decision, action


def test_trace_rule_exposure_separates_strict_runtime_takeover_from_model_events():
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 1, "room_type": "Map"},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck": []},
    }
    prompt = "Choose a legal route action.\n\n" + render_game_rules(state)
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": prompt},
    ]
    model_decision = {
        "type": "decision",
        "step": 0,
        "decision": "map_select",
        "authoritative_state": state,
        "prompt": prompt,
    }
    model_action = {
        "type": "action",
        "step": 0,
        "decision": "map_select",
        "agent_meta": {
            **game_rules_meta(state, prompt=prompt),
            **_model_request_audits_meta(messages),
        },
    }
    runtime_decision, runtime_action = _runtime_takeover_rows()

    audit = experiment._trace_rule_exposure(
        {"rows": [model_decision, model_action, runtime_decision, runtime_action]},
        {**game_rules_identity(), "default_delivery": "state_selected_prompt"},
    )

    assert audit["valid"] is True
    assert audit["model_event_count"] == 1
    assert audit["request_count"] == 1
    assert audit["runtime_takeover_count"] == 1
    assert audit["invalid_runtime_takeover_indexes"] == []


def test_trace_rule_exposure_accepts_exact_no_route_runtime_takeover_shape():
    runtime_decision, runtime_action = _runtime_takeover_rows()
    runtime_action = copy.deepcopy(runtime_action)
    runtime_action["agent_meta"].pop("route_plan_effect")

    audit = experiment._runtime_takeover_audit(
        runtime_action,
        runtime_decision,
        route_plan_enabled=False,
    )

    assert audit == {"valid": True, "errors": []}


@pytest.mark.parametrize("route_plan_effect", [None, {}])
def test_trace_rule_exposure_rejects_route_plan_metadata_when_route_plan_is_disabled(
    route_plan_effect,
):
    runtime_decision, runtime_action = _runtime_takeover_rows()
    runtime_action = copy.deepcopy(runtime_action)
    runtime_action["agent_meta"]["route_plan_effect"] = route_plan_effect

    audit = experiment._trace_rule_exposure(
        {"rows": [runtime_decision, runtime_action]},
        {**game_rules_identity(), "default_delivery": "state_selected_prompt"},
        route_plan_enabled=False,
    )

    assert audit["valid"] is False
    assert audit["runtime_takeover_count"] == 1
    assert audit["invalid_runtime_takeover_indexes"] == [0]


@pytest.mark.parametrize(
    "mutation",
    [
        "prompt",
        "model_called",
        "model_metadata",
        "missing_decision_marker",
        "two_choices",
        "two_map_actions",
        "choice_mismatch",
        "rejected_action",
        "route_effect",
    ],
)
def test_trace_rule_exposure_rejects_invalid_runtime_takeover(mutation):
    runtime_decision, runtime_action = _runtime_takeover_rows()
    runtime_decision = copy.deepcopy(runtime_decision)
    runtime_action = copy.deepcopy(runtime_action)
    if mutation == "prompt":
        runtime_decision["prompt"] = "No model prompt is allowed."
    elif mutation == "model_called":
        runtime_action["agent_meta"]["model_called"] = True
    elif mutation == "model_metadata":
        runtime_action["agent_meta"]["model_request_audits"] = []
    elif mutation == "missing_decision_marker":
        runtime_decision.pop("runtime_takeover")
    elif mutation == "two_choices":
        runtime_decision["authoritative_state"]["choices"].append(
            {"col": 3, "row": 3, "type": "Unknown"}
        )
    elif mutation == "two_map_actions":
        extra = copy.deepcopy(runtime_action["action"])
        extra["action_id"] = 6
        extra["cmd"]["args"]["col"] = 3
        runtime_decision["legal_actions"].append(extra)
    elif mutation == "choice_mismatch":
        runtime_action["command"]["args"]["col"] = 4
    elif mutation == "rejected_action":
        runtime_action["type"] = "rejected_action"
    else:
        runtime_action["agent_meta"]["route_plan_effect"]["outcome"] = "invented"

    audit = experiment._trace_rule_exposure(
        {"rows": [runtime_decision, runtime_action]},
        {**game_rules_identity(), "default_delivery": "state_selected_prompt"},
    )

    assert audit["valid"] is False
    assert audit["runtime_takeover_count"] == 1
    assert audit["invalid_runtime_takeover_indexes"] == [0]


def test_default_provider_audit_preserves_success_and_protocol_failure_bodies(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            self.status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload

    responses = [
        {
            "id": "success-request",
            "model": "resolved-model",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 5},
        },
        {
            "id": "truncated-request",
            "model": "resolved-model",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
            "usage": {"total_tokens": 6},
        },
    ]
    bodies = [json.dumps(response, separators=(",", ":")).encode("utf-8") for response in responses]
    monkeypatch.setenv(FORMAL_PROVIDER_AUDIT_ENV, "1")
    monkeypatch.setattr(
        "sts2_bench.agents.urllib.request.urlopen",
        lambda request, timeout: FakeResponse(bodies.pop(0)),
    )
    agent = OpenAICompatAgent(base_url="http://127.0.0.1:8000/v1", model="test", max_retries=0)
    messages = [{"role": "user", "content": "choose"}]

    success_audits = []
    agent._begin_model_request(success_audits, messages)
    success = agent._chat(messages)
    agent._complete_model_request(success_audits, success)
    assert success_audits[0]["provider_response_status"] == "succeeded"
    assert success_audits[0]["provider_http_status"] == 200
    assert success_audits[0]["provider_finish_reason"] == "stop"
    assert json.loads(success_audits[0]["provider_response_body"])["id"] == "success-request"
    assert success_audits[0]["provider_response_sha256"] == hashlib.sha256(
        success_audits[0]["provider_response_body"].encode("utf-8")
    ).hexdigest()
    assert experiment._provider_request_audit(success_audits[0])["valid"] is True
    for invalid_http_status in (None, 500, "500", True):
        tampered_success = {**success_audits[0], "provider_http_status": invalid_http_status}
        assert experiment._provider_request_audit(tampered_success)["valid"] is False
    for invalid_latency in (None, -0.1, "0.0", True, float("inf"), float("-inf"), float("nan")):
        tampered_success = {**success_audits[0], "provider_latency_sec": invalid_latency}
        assert experiment._provider_request_audit(tampered_success)["valid"] is False

    failure_audits = []
    agent._begin_model_request(failure_audits, messages)
    truncated = agent._chat(messages)
    with pytest.raises(ValueError, match="finish_reason=length") as captured:
        agent._complete_model_request(failure_audits, truncated)
    agent._fail_model_request(failure_audits, captured.value)
    assert failure_audits[0]["provider_response_status"] == "model_protocol_error"
    assert failure_audits[0]["provider_finish_reason"] == "length"
    assert json.loads(failure_audits[0]["provider_response_body"])["id"] == "truncated-request"
    assert experiment._provider_request_audit(failure_audits[0])["valid"] is True
    failure_audits[0]["provider_metadata"]["id"] = "tampered-request"
    assert experiment._provider_request_audit(failure_audits[0])["valid"] is False


def test_default_provider_audit_binds_mutually_exclusive_protocol_error_http_status():
    response_body = '{"error":{"message":"invalid request"}}'
    protocol_error = {
        "provider_response_status": "model_protocol_error",
        "provider_response_body": response_body,
        "provider_response_sha256": hashlib.sha256(response_body.encode("utf-8")).hexdigest(),
        "provider_http_status": 200,
        "provider_metadata": {},
        "provider_finish_reason": None,
        "provider_usage": None,
        "provider_latency_sec": 0.0,
        "provider_error": "invalid provider response",
    }

    assert experiment._provider_request_audit(protocol_error)["valid"] is True
    for valid_http_status in (200, 299, 400, 401, 499):
        with_status = {**protocol_error, "provider_http_status": valid_http_status}
        assert experiment._provider_request_audit(with_status)["valid"] is True
    missing_status = dict(protocol_error)
    missing_status.pop("provider_http_status")
    assert experiment._provider_request_audit(missing_status)["valid"] is False
    for invalid_http_status in (None, 199, 300, 399, 408, 429, 500, 600, "401", True):
        tampered = {**protocol_error, "provider_http_status": invalid_http_status}
        assert experiment._provider_request_audit(tampered)["valid"] is False


def test_default_provider_audit_accepts_only_retryable_http_status_as_transport_error():
    response_body = '{"error":{"message":"provider unavailable"}}'
    transport_error = {
        "provider_response_status": "transport_error",
        "provider_response_body": response_body,
        "provider_response_sha256": hashlib.sha256(response_body.encode("utf-8")).hexdigest(),
        "provider_metadata": {},
        "provider_finish_reason": None,
        "provider_usage": None,
        "provider_latency_sec": 0.0,
        "provider_error": "provider unavailable",
    }
    for valid_http_status in (408, 429, 500, 503, 599):
        candidate = {**transport_error, "provider_http_status": valid_http_status}
        assert experiment._provider_request_audit(candidate)["valid"] is True
    for invalid_http_status in (200, 299, 400, 401, 499, "429", True):
        candidate = {**transport_error, "provider_http_status": invalid_http_status}
        assert experiment._provider_request_audit(candidate)["valid"] is False


def test_default_provider_audit_rejects_empty_stop_and_preserves_http_error_body(monkeypatch):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]}
            ).encode("utf-8")

    monkeypatch.setenv(FORMAL_PROVIDER_AUDIT_ENV, "1")
    agent = OpenAICompatAgent(base_url="http://127.0.0.1:8000/v1", model="test", max_retries=0)
    messages = [{"role": "user", "content": "choose"}]
    monkeypatch.setattr("sts2_bench.agents.urllib.request.urlopen", lambda request, timeout: FakeResponse())
    empty_audits = []
    agent._begin_model_request(empty_audits, messages)
    response = agent._chat(messages)
    with pytest.raises(ValueError, match="non-empty text") as captured:
        agent._complete_model_request(empty_audits, response)
    agent._fail_model_request(empty_audits, captured.value)
    assert empty_audits[0]["provider_response_status"] == "model_protocol_error"
    assert experiment._provider_request_audit(empty_audits[0])["valid"] is True

    error_body = b'{"error":{"message":"rate limited"}}'

    def raise_http_error(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(error_body),
        )

    monkeypatch.setattr("sts2_bench.agents.urllib.request.urlopen", raise_http_error)
    http_audits = []
    agent._begin_model_request(http_audits, messages)
    with pytest.raises(RuntimeError, match="LLM HTTP 429") as http_error:
        agent._chat(messages)
    agent._fail_model_request(http_audits, http_error.value)
    assert http_audits[0]["provider_response_status"] == "transport_error"
    assert http_audits[0]["provider_http_status"] == 429
    assert http_audits[0]["provider_response_body"] == error_body.decode("utf-8")
    assert experiment._provider_request_audit(http_audits[0])["valid"] is True


@pytest.mark.parametrize(
    "mutation",
    ["raw_response", "finish_reason", "usage", "metadata", "body_hash"],
)
def test_trace_rule_exposure_fails_closed_on_provider_evidence_drift(tmp_path, mutation):
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 1, "room_type": "Map"},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck": []},
    }
    prompt = "Choose a legal route action.\n\n" + render_game_rules(state)
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": prompt},
    ]
    decision = {
        "type": "decision",
        "step": 0,
        "decision": "map_select",
        "authoritative_state": state,
        "prompt": prompt,
    }
    action = {
        "type": "action",
        "step": 0,
        "decision": "map_select",
        "agent_meta": {
            **game_rules_meta(state, prompt=prompt),
            **_model_request_audits_meta(messages),
        },
    }
    request = action["agent_meta"]["model_request_audits"][0]
    if mutation == "raw_response":
        action["agent_meta"]["raw_response"] = '{"action_id":1}'
    elif mutation == "finish_reason":
        body = json.loads(request["provider_response_body"])
        body["choices"][0]["finish_reason"] = "length"
        request["provider_response_body"] = json.dumps(body, separators=(",", ":"))
        request["provider_response_sha256"] = hashlib.sha256(
            request["provider_response_body"].encode("utf-8")
        ).hexdigest()
        request["provider_response_status"] = "model_protocol_error"
        request["provider_finish_reason"] = "length"
        request["provider_metadata"]["finish_reason"] = "length"
        request["provider_error"] = "finish_reason=length"
        action["agent_meta"]["provider_finish_reason"] = "length"
        action["agent_meta"]["provider_metadata"]["finish_reason"] = "length"
    elif mutation == "usage":
        action["agent_meta"]["usage"] = {"total_tokens": 999}
    elif mutation == "metadata":
        action["agent_meta"]["provider_metadata"]["id"] = "another-request"
    else:
        request["provider_response_sha256"] = "0" * 64

    trace_path = tmp_path / f"{mutation}.jsonl"
    trace_path.write_text(json.dumps(decision) + "\n" + json.dumps(action) + "\n", encoding="utf-8")
    expected_rules = {**game_rules_identity(), "default_delivery": "state_selected_prompt"}
    audit = experiment._trace_rule_exposure(experiment._read_trace_jsonl(trace_path), expected_rules)

    assert audit["valid"] is False
    assert audit["invalid_provider_request_refs"]


def test_trace_rule_exposure_audits_rejected_and_final_request_sequences(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 1, "room_type": "Map"},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck": []},
    }
    prompt = "Choose a legal route action.\n\n" + render_game_rules(state)
    first_request = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": prompt},
    ]
    corrected_request = [
        *first_request,
        {"role": "user", "content": "Your previous response was invalid; choose again."},
    ]
    decision = {
        "type": "decision",
        "step": 0,
        "decision": "map_select",
        "authoritative_state": state,
        "prompt": prompt,
    }
    rule_meta = game_rules_meta(state, prompt=prompt)
    rejected = {
        "type": "rejected_action",
        "step": 0,
        "decision": "map_select",
        "agent_meta": {
            **rule_meta,
            **_model_request_audits_meta(first_request, corrected_request),
        },
    }
    action = {
        "type": "action",
        "step": 0,
        "decision": "map_select",
        "agent_meta": {
            **rule_meta,
            **_model_request_audits_meta(first_request),
        },
    }
    trace_path.write_text(
        "\n".join(json.dumps(row) for row in (decision, rejected, action)) + "\n",
        encoding="utf-8",
    )
    expected_rules = {**game_rules_identity(), "default_delivery": "state_selected_prompt"}

    audit = experiment._trace_rule_exposure(experiment._read_trace_jsonl(trace_path), expected_rules)

    assert audit["valid"] is True
    assert audit["model_event_count"] == 2
    assert audit["request_count"] == 3

    rejected["agent_meta"]["model_request_audits"][1]["messages_sha256"] = "0" * 64
    trace_path.write_text(
        "\n".join(json.dumps(row) for row in (decision, rejected, action)) + "\n",
        encoding="utf-8",
    )

    audit = experiment._trace_rule_exposure(experiment._read_trace_jsonl(trace_path), expected_rules)
    assert audit["valid"] is False
    assert audit["invalid_request_refs"] == [{"model_event_index": 0, "request_index": 1}]


def test_trace_rule_exposure_aggregates_multi_request_usage_and_latency():
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 1, "room_type": "Map"},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck": []},
    }
    prompt = "Choose a legal route action.\n\n" + render_game_rules(state)
    first_request = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": prompt},
    ]
    corrected_request = [
        *first_request,
        {"role": "user", "content": "Your previous response was invalid; choose again."},
    ]
    usages = [
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
        {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    ]
    decision = {
        "type": "decision",
        "step": 0,
        "decision": "map_select",
        "authoritative_state": state,
        "prompt": prompt,
    }
    action = {
        "type": "action",
        "step": 0,
        "decision": "map_select",
        "agent_meta": {
            **game_rules_meta(state, prompt=prompt),
            **_model_request_audits_meta(
                first_request,
                corrected_request,
                provider_usages=usages,
                provider_latencies=[0.125, 0.375],
            ),
        },
    }

    audit = experiment._trace_rule_exposure(
        {"rows": [decision, action]},
        {**game_rules_identity(), "default_delivery": "state_selected_prompt"},
    )

    assert audit["valid"] is True
    assert audit["request_count"] == 2
    assert audit["provider_usage_totals"] == {
        "prompt_tokens": 30,
        "completion_tokens": 7,
        "total_tokens": 37,
        "prompt_tokens_details": {"cached_tokens": 7},
    }
    assert audit["provider_latency_sec"] == 0.5


def test_run_job_records_malformed_trace_as_failed_attempt_then_retries(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        if len(calls) == 1:
            Path(_argument(command, "--out")).write_text(
                '{"type":"decision","step":0}\n{"type":',
                encoding="utf-8",
            )
            return 1
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=2)

    first_attempt = json.loads(
        (
            tmp_path
            / "results"
            / "runs"
            / "seed-a"
            / "replicate_01"
            / "attempts"
            / "attempt_001"
            / "attempt_result.json"
        ).read_text(encoding="utf-8")
    )
    assert first_attempt["status"] == "retryable_infrastructure"
    assert first_attempt["validation"]["checks"]["trace_jsonl_well_formed"] is False
    assert first_attempt["validation"]["trace_errors"][0]["line"] == 2
    assert status["status"] == "succeeded"
    assert status["attempt"] == 2
    assert len(calls) == 2


def test_run_job_records_malformed_result_manifest_then_retries(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        if len(calls) == 1:
            trace_path = Path(_argument(command, "--out"))
            experiment.manifest_path_for_trace(trace_path).write_bytes(b'{"status":"succeeded","bad":"\xe4')
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=2)

    first_attempt = json.loads(
        (
            tmp_path
            / "results"
            / "runs"
            / "seed-a"
            / "replicate_01"
            / "attempts"
            / "attempt_001"
            / "attempt_result.json"
        ).read_text(encoding="utf-8")
    )
    assert first_attempt["status"] == "retryable_infrastructure"
    assert first_attempt["validation"]["checks"]["manifest_succeeded"] is False
    assert status["status"] == "succeeded"
    assert status["attempt"] == 2
    assert len(calls) == 2


def test_trace_reader_reports_half_written_utf8_without_raising(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_bytes(b'{"type":"decision","step":0}\n{"type":"action","text":"\xe4')

    trace = experiment._read_trace_jsonl(trace_path)

    assert trace["well_formed"] is False
    assert trace["rows"] == [{"type": "decision", "step": 0}]
    assert trace["errors"][0]["line"] == 2
    assert trace["errors"][0]["error"].startswith("invalid UTF-8:")


def test_run_job_preserves_truncated_behavior_failure_as_replacement_required(
    tmp_path,
    monkeypatch,
):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        result = {
            "seed": "seed-a",
            "decision": "combat_play",
            "victory": False,
            "error": None,
            "truncated": True,
            "extra": {"last_observed_state": {"decision": "combat_play"}},
        }
        _write_attempt_artifacts(command, result=result, manifest_status="succeeded")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=1)

    assert status["status"] == "behavior_failure"
    assert status["formal_sample"] is False
    assert status["replacement_required"] is True
    assert status["retryable"] is True
    assert status["classification"] == "behavior_failure"
    assert status["validation"]["checks"]["not_truncated"] is False
    assert status["validation"]["checks"]["natural_game_over"] is False
    assert len(calls) == 1


def test_run_job_rejects_authoritative_terminal_engine_error(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        result = _natural_result()
        result["extra"]["authoritative_terminal_state"].update(
            {
                "engine_error": True,
                "error": "engine_error",
                "engine_error_reason": "terminal export failed",
            }
        )
        _write_attempt_artifacts(command, result=result, manifest_status="succeeded")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=2)

    assert status["status"] == "retryable_infrastructure"
    assert status["formal_sample"] is False
    assert status["validation"]["checks"]["authoritative_terminal_engine_error_free"] is False
    assert len(calls) == 2


def test_run_job_stops_and_reports_unsupported_headless_decision(tmp_path, monkeypatch):
    calls = []
    observed_state = {
        "decision": "unrecognized_state",
        "room_type": "FutureRoom",
        "message": "No supported decision adapter matched the current room or state",
    }

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("", encoding="utf-8")
        kwargs["stderr_path"].write_text("unsupported\n", encoding="utf-8")
        result = {
            "seed": "seed-a",
            "decision": "unrecognized_state",
            "victory": False,
            "error": "unsupported decision: 'unrecognized_state'; refusing implicit proceed",
            "truncated": False,
            "extra": {"last_observed_state": observed_state},
        }
        _write_attempt_artifacts(command, result=result, manifest_status="failed")
        return 1

    status = _run_job(tmp_path, monkeypatch, runner, attempts=3)

    assert status["status"] == "needs_attention"
    assert status["validation"]["report_required"] is True
    assert status["validation"]["issue"]["last_observed_state"] == observed_state
    assert len(calls) == 1
    report = json.loads(
        (tmp_path / "results" / "runs" / "seed-a" / "replicate_01" / "attempts" / "attempt_001" / "report_required.json").read_text(encoding="utf-8")
    )
    assert report["seed"] == "seed-a"
    assert report["decision"] == "unrecognized_state"


def test_run_job_stops_and_reports_recognized_decision_without_legal_actions(tmp_path, monkeypatch):
    calls = []
    observed_state = {"decision": "event_result", "can_proceed": False}

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout_path"].write_text("", encoding="utf-8")
        kwargs["stderr_path"].write_text("no legal actions\n", encoding="utf-8")
        result = {
            "seed": "seed-a",
            "decision": "event_result",
            "victory": False,
            "error": "recognized decision 'event_result' exposed no legal actions; refusing implicit proceed",
            "truncated": False,
            "extra": {"last_observed_state": observed_state},
        }
        _write_attempt_artifacts(command, result=result, manifest_status="failed")
        return 1

    status = _run_job(tmp_path, monkeypatch, runner, attempts=3)

    assert status["status"] == "needs_attention"
    assert status["validation"]["report_required"] is True
    assert status["validation"]["issue"]["last_observed_state"] == observed_state
    assert len(calls) == 1


def test_initialize_output_rejects_existing_results_from_an_old_identity(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    out_root = tmp_path / "results"
    out_root.mkdir()
    generated_suite = out_root / experiment.GENERATED_SUITE
    generated_suite.parent.mkdir(parents=True)
    generated_suite.write_text("{}\n", encoding="utf-8")
    identity_path = out_root / experiment.IDENTITY_FILE
    identity_path.write_text(json.dumps({"identity_sha256": "old"}) + "\n", encoding="utf-8")

    experiment._initialize_output(
        out_root,
        protocol_path=protocol_path,
        frozen={"identity_sha256": "old"},
    )

    identity_path.write_text(json.dumps({"identity_sha256": "new"}) + "\n", encoding="utf-8")
    with pytest.raises(experiment.DefaultV1ExperimentError, match="Source identity does not match"):
        experiment._initialize_output(
            out_root,
            protocol_path=protocol_path,
            frozen={"identity_sha256": "old"},
        )


def test_seed_lane_runs_replicates_in_strict_order(tmp_path, monkeypatch):
    order = []

    def runner(command, **kwargs):
        replicate = _argument(command, "--replicate-id")
        order.append(replicate)
        kwargs["stdout_path"].write_text("", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(
        experiment,
        "_verify_expected_identity",
        lambda _snapshot, identity: {"identity_sha256": identity},
    )
    lane = experiment._run_seed_lane(
        _protocol(repeats=3, attempts=1),
        benchmark_root=snapshot,
        lock_root=snapshot,
        out_root=tmp_path / "results",
        frozen_identity="frozen-test",
        identity_lock=threading.Lock(),
        lane={"name": "seed-a-lane", "seed": "seed-a"},
        runner=runner,
        progress=lambda: None,
    )

    assert order == ["replicate-01", "replicate-02", "replicate-03"]
    assert lane["status"] == "succeeded"
    assert lane["completed_replicates"] == 3


def test_seed_lane_stops_after_nonformal_behavior_failure(tmp_path, monkeypatch):
    order = []

    def runner(command, **kwargs):
        order.append(_argument(command, "--replicate-id"))
        kwargs["stdout_path"].write_text("", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        result = {
            "seed": "seed-a",
            "decision": "combat_play",
            "victory": False,
            "error": None,
            "truncated": True,
            "extra": {"last_observed_state": {"decision": "combat_play"}},
        }
        _write_attempt_artifacts(command, result=result, manifest_status="failed")
        return 1

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(
        experiment,
        "_verify_expected_identity",
        lambda _snapshot, identity: {"identity_sha256": identity},
    )
    lane = experiment._run_seed_lane(
        _protocol(repeats=3, attempts=1),
        benchmark_root=snapshot,
        lock_root=snapshot,
        out_root=tmp_path / "results",
        frozen_identity="frozen-test",
        identity_lock=threading.Lock(),
        lane={"name": "seed-a-lane", "seed": "seed-a"},
        runner=runner,
        progress=lambda: None,
    )

    assert order == ["replicate-01"]
    assert lane["status"] == "replacement_pending"
    assert lane["behavior_failures"] == 1
    assert lane["processed_replicates"] == 1
    assert not (tmp_path / "results" / "report_required.json").exists()


def test_seed_lane_does_not_start_new_job_after_experiment_sticky_hold(tmp_path):
    out_root = tmp_path / "results"
    experiment._atomic_write_json(
        out_root / "report_required.json",
        {"type": "unsupported_headless_decision"},
    )
    calls = []

    lane = experiment._run_seed_lane(
        _protocol(repeats=3, attempts=1),
        benchmark_root=tmp_path / "snapshot",
        lock_root=tmp_path / "snapshot",
        out_root=out_root,
        frozen_identity="frozen-test",
        identity_lock=threading.Lock(),
        lane={"name": "seed-a-lane", "seed": "seed-a"},
        runner=lambda *args, **kwargs: calls.append(args),
        progress=lambda: None,
    )

    assert lane["status"] == "needs_attention"
    assert lane["completed_replicates"] == 0
    assert lane["experiment_hold"]["type"] == "unsupported_headless_decision"
    assert calls == []


def test_concurrent_global_sticky_publish_is_first_writer_wins(tmp_path):
    out_root = tmp_path / "results"
    control_lock = threading.Lock()
    start = threading.Barrier(3)
    reports = [
        {"type": "synthetic-report", "seed": "seed-a"},
        {"type": "synthetic-report", "seed": "seed-b"},
    ]
    returned = []
    errors = []

    def publish(report):
        try:
            start.wait(timeout=2)
            returned.append(
                experiment._publish_experiment_hold(
                    out_root,
                    report,
                    control_lock=control_lock,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - preserve concurrent failure evidence
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(report,)) for report in reports]
    for thread in threads:
        thread.start()
    start.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=2)

    stored = json.loads((out_root / "report_required.json").read_text(encoding="utf-8"))
    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert stored in reports
    assert returned == [stored, stored]


def test_global_hold_published_after_lane_check_still_blocks_attempt_launch(tmp_path, monkeypatch):
    out_root = tmp_path / "results"
    protocol = _protocol(repeats=2, attempts=1)
    control_lock = threading.Lock()
    second_lane_entered_job = threading.Event()
    hold_published = threading.Event()
    lane_results = {}
    runner_calls = []
    original_run_job = experiment._run_job
    original_publish = experiment._publish_experiment_hold

    monkeypatch.setattr(
        experiment,
        "_recover_job",
        lambda *args, **kwargs: {
            "status": "pending",
            "formal_sample": False,
            "retryable": True,
            "attempt_history": [],
        },
    )

    def observed_publish(out_root, report, *, control_lock):
        result = original_publish(out_root, report, control_lock=control_lock)
        hold_published.set()
        return result

    monkeypatch.setattr(experiment, "_publish_experiment_hold", observed_publish)

    def scheduled_run_job(protocol, *, seed, replicate, **kwargs):
        if seed == "seed-a":
            assert second_lane_entered_job.wait(timeout=2)
            issue = {"type": "synthetic-report", "seed": seed, "replicate": replicate}
            return {
                "status": "needs_attention",
                "formal_sample": False,
                "identity_stable": True,
                "validation": {"report_required": True, "issue": issue},
                "issue": issue,
            }
        if replicate == 1:
            return {
                "status": "succeeded",
                "formal_sample": True,
                "identity_stable": True,
                "classification": "completed_game",
                "validation": {"report_required": False},
            }
        second_lane_entered_job.set()
        assert hold_published.wait(timeout=2)
        return original_run_job(protocol, seed=seed, replicate=replicate, **kwargs)

    monkeypatch.setattr(experiment, "_run_job", scheduled_run_job)

    def run_lane(seed, lane_name):
        lane_results[seed] = experiment._run_seed_lane(
            protocol,
            benchmark_root=tmp_path / "snapshot",
            lock_root=tmp_path / "snapshot",
            out_root=out_root,
            frozen_identity="frozen-test",
            identity_lock=threading.Lock(),
            lane={"name": lane_name, "seed": seed},
            runner=lambda *args, **kwargs: runner_calls.append(args),
            progress=lambda: None,
            protocol_sha256="protocol-test",
            control_lock=control_lock,
        )

    threads = [
        threading.Thread(target=run_lane, args=("seed-a", "seed-a-lane")),
        threading.Thread(target=run_lane, args=("seed-b", "seed-b-lane")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    seed_b_jobs = lane_results["seed-b"]["jobs"]
    assert all(not thread.is_alive() for thread in threads)
    assert runner_calls == []
    assert seed_b_jobs[1]["resume_action"] == "blocked_experiment_report_required"
    assert not (out_root / "runs" / "seed-b" / "replicate_02" / "attempts").exists()


def test_model_protocol_failure_is_one_nonretryable_nonformal_result(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        result = {
            "seed": "seed-a",
            "decision": "combat_play",
            "victory": False,
            "error": "LLM failed after 2 attempt(s): invalid JSON: expected object",
            "truncated": False,
            "extra": {"last_observed_state": {"decision": "combat_play"}},
        }
        _write_attempt_artifacts(command, result=result, manifest_status="failed")
        return 1

    status = _run_job(tmp_path, monkeypatch, runner, attempts=1)

    assert status["status"] == "behavior_failure"
    assert status["formal_sample"] is False
    assert status["replacement_required"] is True
    assert status["retryable"] is True
    assert status["classification"] == "behavior_failure"
    assert len(calls) == 1


def test_behavior_failure_is_replaced_with_natural_game_in_same_job(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            result = {
                "seed": "seed-a",
                "decision": "combat_play",
                "victory": False,
                "error": "LLM failed after 2 attempt(s): invalid JSON: expected object",
                "truncated": False,
                "extra": {"last_observed_state": {"decision": "combat_play"}},
            }
            _write_attempt_artifacts(command, result=result, manifest_status="failed")
            return 1
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=2)

    assert status["status"] == "succeeded"
    assert status["formal_sample"] is True
    assert status["attempt"] == 2
    assert [item["classification"] for item in status["attempt_history"]] == [
        "behavior_failure",
        "completed_game",
    ]
    assert len(calls) == 2
    assert not (tmp_path / "results" / "report_required.json").exists()


def test_resume_appends_natural_replacement_without_mutating_behavior_attempt(
    tmp_path,
    monkeypatch,
):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            result = {
                "seed": "seed-a",
                "decision": "combat_play",
                "victory": False,
                "error": "LLM failed after 2 attempt(s): invalid JSON: expected object",
                "truncated": False,
                "extra": {"last_observed_state": {"decision": "combat_play"}},
            }
            _write_attempt_artifacts(command, result=result, manifest_status="failed")
            return 1
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    first = _run_job(tmp_path, monkeypatch, runner, attempts=1)
    first_attempt = (
        tmp_path
        / "results"
        / "runs"
        / "seed-a"
        / "replicate_01"
        / "attempts"
        / "attempt_001"
        / "attempt_result.json"
    )
    first_hash = hashlib.sha256(first_attempt.read_bytes()).hexdigest()
    second = _run_job(tmp_path, monkeypatch, runner, attempts=1)

    assert first["status"] == "behavior_failure"
    assert second["status"] == "succeeded"
    assert second["attempt"] == 2
    assert hashlib.sha256(first_attempt.read_bytes()).hexdigest() == first_hash
    assert len(calls) == 2


def test_seed_lane_replaces_behavior_before_advancing_replicate(tmp_path, monkeypatch):
    order = []

    def runner(command, **kwargs):
        replicate = _argument(command, "--replicate-id")
        order.append(replicate)
        if order == ["replicate-01"]:
            result = {
                "seed": "seed-a",
                "decision": "combat_play",
                "victory": False,
                "error": "LLM failed after 2 attempt(s): invalid JSON: expected object",
                "truncated": False,
                "extra": {"last_observed_state": {"decision": "combat_play"}},
            }
            _write_attempt_artifacts(command, result=result, manifest_status="failed")
            return 1
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(
        experiment,
        "_verify_expected_identity",
        lambda _snapshot, identity: {"identity_sha256": identity},
    )
    lane = experiment._run_seed_lane(
        _protocol(repeats=2, attempts=2),
        benchmark_root=snapshot,
        lock_root=snapshot,
        out_root=tmp_path / "results",
        frozen_identity="frozen-test",
        identity_lock=threading.Lock(),
        lane={"name": "seed-a-lane", "seed": "seed-a"},
        runner=runner,
        progress=lambda: None,
    )

    assert order == ["replicate-01", "replicate-01", "replicate-02"]
    assert lane["status"] == "succeeded"
    assert lane["completed_replicates"] == 2
    assert lane["behavior_failures"] == 1


def test_behavior_failure_in_one_lane_does_not_hold_healthy_lane(tmp_path, monkeypatch):
    out_root = tmp_path / "results"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(
        experiment,
        "_verify_expected_identity",
        lambda _snapshot, identity: {"identity_sha256": identity},
    )
    protocol = _protocol(repeats=2, attempts=1)
    control_lock = threading.Lock()
    lane_results = {}

    def runner(command, **kwargs):
        seed = json.loads(Path(_argument(command, "--seeds")).read_text(encoding="utf-8"))[0]
        if seed == "seed-a":
            result = {
                "seed": seed,
                "decision": "combat_play",
                "victory": False,
                "error": "LLM failed after 2 attempt(s): invalid JSON: expected object",
                "truncated": False,
                "extra": {"last_observed_state": {"decision": "combat_play"}},
            }
            _write_attempt_artifacts(command, result=result, manifest_status="failed")
            return 1
        _write_attempt_artifacts(
            command,
            result=_natural_result(seed),
            manifest_status="succeeded",
        )
        return 0

    def run_lane(seed):
        lane_results[seed] = experiment._run_seed_lane(
            protocol,
            benchmark_root=snapshot,
            lock_root=snapshot,
            out_root=out_root,
            frozen_identity="frozen-test",
            identity_lock=threading.Lock(),
            lane={"name": f"{seed}-lane", "seed": seed},
            runner=runner,
            progress=lambda: None,
            control_lock=control_lock,
        )

    threads = [
        threading.Thread(target=run_lane, args=("seed-a",)),
        threading.Thread(target=run_lane, args=("seed-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert lane_results["seed-a"]["status"] == "replacement_pending"
    assert lane_results["seed-b"]["status"] == "succeeded"
    assert lane_results["seed-b"]["completed_replicates"] == 2
    assert not (out_root / "report_required.json").exists()


def test_transport_failure_is_retried_but_preserved_in_attempt_history(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            result = {
                "seed": "seed-a",
                "decision": "combat_play",
                "victory": False,
                "error": "LLM failed after 2 attempt(s): <urlopen error timed out>",
                "truncated": False,
                "extra": {"last_observed_state": {"decision": "combat_play"}},
            }
            _write_attempt_artifacts(command, result=result, manifest_status="failed")
            return 1
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=3)

    assert status["status"] == "succeeded"
    assert len(calls) == 2
    assert status["attempt_history"][0]["classification"] == "transport_infrastructure"


def test_auth_configuration_failure_is_not_automatically_retried(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        result = {
            "seed": "seed-a",
            "decision": "combat_play",
            "victory": False,
            "error": "LLM failed after 2 attempt(s): LLM HTTP 401: Invalid token",
            "truncated": False,
            "extra": {"last_observed_state": {"decision": "combat_play"}},
        }
        _write_attempt_artifacts(command, result=result, manifest_status="failed")
        return 1

    status = _run_job(tmp_path, monkeypatch, runner, attempts=3)

    assert status["status"] == "needs_attention"
    assert status["formal_sample"] is False
    assert len(calls) == 1


def test_report_required_is_sticky_across_resume(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        result = {
            "seed": "seed-a",
            "decision": "unrecognized_state",
            "victory": False,
            "error": "unsupported decision: 'unrecognized_state'; refusing implicit proceed",
            "truncated": False,
            "extra": {"last_observed_state": {"decision": "unrecognized_state"}},
        }
        _write_attempt_artifacts(command, result=result, manifest_status="failed")
        return 1

    first = _run_job(tmp_path, monkeypatch, runner)
    second = _run_job(tmp_path, monkeypatch, runner)

    assert first["status"] == second["status"] == "needs_attention"
    assert second["resume_action"] == "sticky_report_required"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "damage",
    ["corrupt_job_status", "missing_attempt_result", "corrupt_attempt_result"],
)
def test_resume_recovers_valid_attempt_from_disk_without_replay(tmp_path, monkeypatch, damage):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    first = _run_job(tmp_path, monkeypatch, runner)
    job_dir = tmp_path / "results" / "runs" / "seed-a" / "replicate_01"
    if damage == "corrupt_job_status":
        (job_dir / "job_status.json").write_text('{"broken":', encoding="utf-8")
    elif damage == "missing_attempt_result":
        (job_dir / "attempts" / "attempt_001" / "attempt_result.json").unlink()
    else:
        (job_dir / "attempts" / "attempt_001" / "attempt_result.json").write_text(
            '{"broken":', encoding="utf-8"
        )

    second = _run_job(tmp_path, monkeypatch, runner)

    assert first["formal_sample"] is True
    assert second["formal_sample"] is True
    assert second["resume_action"] == "skipped_completed"
    assert len(calls) == 1
    attempt_dir = job_dir / "attempts" / "attempt_001"
    repaired = json.loads((attempt_dir / "attempt_result.json").read_text(encoding="utf-8"))
    assert repaired["formal_sample"] is True
    assert repaired["seed"] == "seed-a"
    assert repaired["frozen_identity_sha256"] == "frozen-test"
    if damage == "missing_attempt_result":
        assert repaired["recovery_source"] == "authoritative_child_artifacts"
    if damage == "corrupt_attempt_result":
        assert repaired["recovery_source"] == "authoritative_child_artifacts"
        assert list(attempt_dir.glob("attempt_result.corrupt.*.json"))


def test_resume_rejects_drift_in_accepted_artifact_manifest(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    first = _run_job(tmp_path, monkeypatch, runner)
    attempt_dir = (
        tmp_path
        / "results"
        / "runs"
        / "seed-a"
        / "replicate_01"
        / "attempts"
        / "attempt_001"
    )
    (attempt_dir / "unexpected-after-acceptance.txt").write_text("drift\n", encoding="utf-8")

    resumed = _run_job(tmp_path, monkeypatch, runner)

    assert first["formal_sample"] is True
    assert resumed["formal_sample"] is False
    assert resumed["status"] == "needs_attention"
    assert resumed["issue"]["type"] == "accepted_artifact_drift"
    assert len(calls) == 1


def test_live_attempt_lease_blocks_duplicate_process(tmp_path, monkeypatch):
    calls = []
    job_dir = tmp_path / "results" / "runs" / "seed-a" / "replicate_01"
    experiment._atomic_write_json(
        job_dir / "active_attempt.json",
        {"owner_pid": __import__("os").getpid(), "child_pid": None, "attempt": 1},
    )

    status = _run_job(tmp_path, monkeypatch, lambda *args, **kwargs: calls.append(args))

    assert status["status"] == "active_attempt_detected"
    assert calls == []


def test_corrupt_active_attempt_lease_becomes_sticky_and_final_revalidation_continues(tmp_path):
    out_root = tmp_path / "results"
    job_dir = out_root / "runs" / "seed-a" / "replicate_01"
    lease_path = job_dir / "active_attempt.json"
    lease_path.parent.mkdir(parents=True)
    corrupt_evidence = '{"owner_pid":'
    lease_path.write_text(corrupt_evidence, encoding="utf-8")
    protocol = _protocol(repeats=1, attempts=1)
    protocol["seed_lanes"] = [{"name": "seed-a-lane", "seed": "seed-a"}]
    control_lock = threading.Lock()

    lanes = experiment._revalidate_all_jobs(
        protocol,
        out_root=out_root,
        frozen_identity="frozen-test",
        protocol_sha256="protocol-test",
        control_lock=control_lock,
    )

    job = lanes[0]["jobs"][0]
    job_report = json.loads((job_dir / "report_required.json").read_text(encoding="utf-8"))
    global_report = json.loads((out_root / "report_required.json").read_text(encoding="utf-8"))
    stored_status = json.loads((job_dir / "job_status.json").read_text(encoding="utf-8"))
    assert job["status"] == "needs_attention"
    assert job["issue"]["type"] == "corrupt_active_attempt_lease"
    assert job_report == job["issue"]
    assert global_report == job["issue"]
    assert stored_status["resume_action"] == "blocked_corrupt_active_attempt_lease"
    assert lease_path.read_text(encoding="utf-8") == corrupt_evidence


def test_subprocess_runner_reaps_child_when_child_pid_lease_write_fails(tmp_path, monkeypatch):
    class Proc:
        pid = 424242

    child = Proc()
    terminated = []
    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(
        experiment,
        "_try_read_json",
        lambda path: ({"owner_pid": 1, "child_pid": None, "state": "starting"}, None),
    )

    def fail_write(*args, **kwargs):
        raise OSError("synthetic child-pid lease write failure")

    monkeypatch.setattr(experiment, "_atomic_write_json", fail_write)
    monkeypatch.setattr(experiment, "_terminate_process_tree", lambda proc: terminated.append(proc))

    with pytest.raises(OSError, match="synthetic child-pid lease write failure"):
        experiment._subprocess_runner(
            ["noop"],
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            cwd=tmp_path,
            timeout_sec=1.0,
            lease_path=tmp_path / "active_attempt.json",
        )

    assert terminated == [child]


def test_subprocess_runner_enables_strict_provider_audit_only_in_child(tmp_path, monkeypatch):
    lease_path = tmp_path / "active_attempt.json"
    experiment._atomic_write_json(
        lease_path,
        {"owner_pid": 1, "child_pid": None, "state": "starting"},
    )
    captured = {}

    class Proc:
        pid = 424242

        @staticmethod
        def wait(timeout):
            return 0

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return Proc()

    monkeypatch.delenv(experiment.FORMAL_PROVIDER_AUDIT_ENV, raising=False)
    monkeypatch.setattr(experiment.subprocess, "Popen", fake_popen)

    returncode = experiment._subprocess_runner(
        ["noop"],
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        cwd=tmp_path,
        timeout_sec=1.0,
        lease_path=lease_path,
    )

    assert returncode == 0
    assert captured["env"][experiment.FORMAL_PROVIDER_AUDIT_ENV] == "1"
    assert experiment.FORMAL_PROVIDER_AUDIT_ENV not in experiment.os.environ


def test_subprocess_runner_wait_failure_reaps_tree_and_preserves_live_child_lease(tmp_path, monkeypatch):
    lease_path = tmp_path / "active_attempt.json"
    experiment._atomic_write_json(
        lease_path,
        {"owner_pid": 1, "child_pid": None, "state": "starting"},
    )

    class Proc:
        pid = 424242

        @staticmethod
        def wait(timeout):
            raise OSError("synthetic wait failure")

    child = Proc()
    terminated = []
    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(
        experiment,
        "_terminate_process_tree",
        lambda proc: terminated.append(proc) or False,
    )

    with pytest.raises(OSError, match="synthetic wait failure"):
        experiment._subprocess_runner(
            ["noop"],
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            cwd=tmp_path,
            timeout_sec=1.0,
            lease_path=lease_path,
        )

    monkeypatch.setattr(experiment, "_pid_alive", lambda pid: pid == child.pid)
    experiment._release_attempt_lease(lease_path)
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    assert terminated == [child]
    assert lease["child_pid"] == child.pid
    assert lease["state"] == "cleanup_required"


def test_subprocess_runner_timeout_reaps_tree_and_returns_124(tmp_path, monkeypatch):
    lease_path = tmp_path / "active_attempt.json"
    experiment._atomic_write_json(
        lease_path,
        {"owner_pid": 1, "child_pid": None, "state": "starting"},
    )

    class Proc:
        pid = 424242

        @staticmethod
        def wait(timeout):
            raise experiment.subprocess.TimeoutExpired("synthetic", timeout)

    child = Proc()
    terminated = []
    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(
        experiment,
        "_terminate_process_tree",
        lambda proc: terminated.append(proc) or True,
    )

    returncode = experiment._subprocess_runner(
        ["noop"],
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        cwd=tmp_path,
        timeout_sec=1.0,
        lease_path=lease_path,
    )

    assert returncode == 124
    assert terminated == [child]
    assert "timed out after 1s" in (tmp_path / "stderr.log").read_text(encoding="utf-8")


def test_runner_exception_is_atomic_process_failure_then_retried(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise OSError("injected launch failure")
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=2)
    first_result = (
        tmp_path / "results" / "runs" / "seed-a" / "replicate_01" / "attempts" / "attempt_001" / "attempt_result.json"
    )

    assert status["status"] == "succeeded"
    assert json.loads(first_result.read_text(encoding="utf-8"))["classification"] == "process_infrastructure"
    assert len(calls) == 2


def test_postrun_trace_analysis_failure_keeps_authoritative_game_without_replay(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        if len(calls) == 1:
            trace_path = Path(_argument(command, "--out"))
            manifest_path = experiment.manifest_path_for_trace(trace_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "failed"
            manifest["error"] = "RuntimeError: injected trace analyzer failure"
            manifest.pop("trace_analysis", None)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (Path(_argument(command, "--trace-analysis-dir")) / "index.json").unlink()
            return 1
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=2)

    assert status["status"] == "succeeded"
    assert status["classification"] == "completed_game"
    assert status["postrun_pending"] is True
    assert status["needs_attention"] is True
    assert status["attempt_history"][0]["classification"] == "completed_game"
    assert len(calls) == 1


def test_watchdog_exit_after_authoritative_game_over_does_not_replay(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        return 124

    status = _run_job(tmp_path, monkeypatch, runner, attempts=2)

    assert status["status"] == "succeeded"
    assert status["classification"] == "completed_game"
    assert status["postrun_pending"] is True
    assert len(calls) == 1


@pytest.mark.parametrize(
    "result",
    [
        {
            "seed": "seed-a",
            "decision": "combat_play",
            "victory": False,
            "error": "LLM failed after 2 attempt(s): invalid JSON: expected object",
            "truncated": False,
            "extra": {"last_observed_state": {"decision": "combat_play"}},
        },
        {
            "seed": "seed-a",
            "decision": "forfeit",
            "victory": False,
            "error": None,
            "truncated": False,
            "termination_class": "model_forfeit",
            "extra": {"last_observed_state": {"decision": "combat_play"}},
        },
    ],
)
def test_nonformal_behavior_result_with_missing_postrun_artifacts_is_still_replaced(
    tmp_path,
    monkeypatch,
    result,
):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_attempt_artifacts(command, result=result, manifest_status="failed")
        trace_path = Path(_argument(command, "--out"))
        experiment.manifest_path_for_trace(trace_path).unlink()
        (Path(_argument(command, "--trace-analysis-dir")) / "index.json").unlink()
        return 1

    status = _run_job(tmp_path, monkeypatch, runner, attempts=2)

    assert status["status"] == "behavior_failure"
    assert status["formal_sample"] is False
    assert status["classification"] == "behavior_failure"
    assert status["replacement_required"] is True
    assert status["postrun_pending"] is False
    assert len(calls) == 2


def test_trace_analysis_index_must_match_child_manifest(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        index_path = Path(_argument(command, "--trace-analysis-dir")) / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["run_count"] = 2
        index_path.write_text(json.dumps(index), encoding="utf-8")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=3)

    assert status["status"] == "needs_attention"
    assert status["classification"] == "integrity_failure"
    assert status["validation"]["checks"]["trace_analysis_complete"] is False
    assert len(calls) == 1


def test_terminal_cross_field_mismatch_fails_closed_without_retry(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        result = _natural_result()
        result["hp"] = 7
        _write_attempt_artifacts(command, result=result, manifest_status="succeeded")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=3)

    assert status["status"] == "needs_attention"
    assert status["validation"]["checks"]["result_matches_terminal"] is False
    assert len(calls) == 1


def test_resolved_model_config_drift_fails_closed_without_retry(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_attempt_artifacts(command, result=_natural_result(), manifest_status="succeeded")
        trace_path = Path(_argument(command, "--out"))
        manifest_path = experiment.manifest_path_for_trace(trace_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_config"]["agent"]["model"] = "wrong-model"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    status = _run_job(tmp_path, monkeypatch, runner, attempts=3)

    assert status["status"] == "needs_attention"
    assert status["validation"]["checks"]["resolved_run_config_matches"] is False
    assert len(calls) == 1


def test_resume_does_not_retry_process_failure_with_tampered_attempt_binding(tmp_path, monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        raise OSError("injected launch failure")

    first = _run_job(tmp_path, monkeypatch, runner, attempts=1)
    attempt_dir = tmp_path / "results" / "runs" / "seed-a" / "replicate_01" / "attempts" / "attempt_001"
    manifest_path = attempt_dir / "attempt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = "tampered-seed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    second = _run_job(tmp_path, monkeypatch, runner, attempts=3)

    assert first["classification"] == "process_infrastructure"
    assert second["status"] == "needs_attention"
    assert second["attempt_history"][0]["classification"] == "integrity_failure"
    assert len(calls) == 1


def test_malformed_attempt_directory_is_not_silently_ignored(tmp_path, monkeypatch):
    malformed = tmp_path / "results" / "runs" / "seed-a" / "replicate_01" / "attempts" / "attempt_notes"
    malformed.mkdir(parents=True)
    calls = []

    status = _run_job(tmp_path, monkeypatch, lambda *args, **kwargs: calls.append(args))

    assert status["status"] == "needs_attention"
    assert status["issue"]["type"] == "malformed_attempt_directory"
    assert calls == []


def test_formal_credentials_require_explicit_external_env_file(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    results = tmp_path / "results"
    snapshot.mkdir()
    results.mkdir()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-must-not-win")

    with pytest.raises(experiment.DefaultV1ExperimentError, match="explicit external"):
        experiment._load_and_validate_external_environment(
            None,
            base_url="https://example.invalid/v1",
            forbidden_roots=(snapshot, results),
        )

    inside_snapshot = snapshot / ".env"
    inside_snapshot.write_text("DEEPSEEK_API_KEY=inside\n", encoding="utf-8")
    with pytest.raises(experiment.DefaultV1ExperimentError, match="outside"):
        experiment._load_and_validate_external_environment(
            inside_snapshot,
            base_url="https://example.invalid/v1",
            forbidden_roots=(snapshot, results),
        )

    external = tmp_path / "credentials.env"
    external.write_text(
        "STS2_BENCH_MODEL=must-be-ignored\nOPENAI_API_KEY=external-key\n",
        encoding="utf-8",
    )
    experiment._load_and_validate_external_environment(
        external,
        base_url="https://example.invalid/v1",
        forbidden_roots=(snapshot, results),
    )

    assert "DEEPSEEK_API_KEY" not in experiment.os.environ
    assert experiment.os.environ["OPENAI_API_KEY"] == "external-key"
    assert experiment.os.environ.get("STS2_BENCH_MODEL") != "must-be-ignored"


def test_external_env_file_checks_both_lexical_and_resolved_boundaries(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    results = tmp_path / "results"
    snapshot.mkdir()
    results.mkdir()
    external = tmp_path / "external.env"
    external.write_text("DEEPSEEK_API_KEY=external\n", encoding="utf-8")
    inside = results / "inside.env"
    inside.write_text("DEEPSEEK_API_KEY=inside\n", encoding="utf-8")
    lexical_hardlink = results / "hardlink-to-external.env"
    lexical_hardlink.hardlink_to(external)
    resolved_hardlink = tmp_path / "outside-hardlink-to-result.env"
    resolved_hardlink.hardlink_to(inside)
    candidates = [lexical_hardlink, resolved_hardlink]
    resolved_alias = tmp_path / "outside-alias.env"
    try:
        resolved_alias.symlink_to(inside)
    except OSError:
        pass
    else:
        candidates.append(resolved_alias)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-remains-on-rejection")

    for candidate in candidates:
        with pytest.raises(experiment.DefaultV1ExperimentError, match="outside|single-link"):
            experiment._load_and_validate_external_environment(
                candidate,
                base_url="https://example.invalid/v1",
                forbidden_roots=(snapshot, results),
            )
        assert experiment.os.environ["DEEPSEEK_API_KEY"] == "ambient-remains-on-rejection"


def test_external_env_file_accepts_ordinary_single_link_outside_boundaries(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    results = tmp_path / "results"
    snapshot.mkdir()
    results.mkdir()
    external = tmp_path / "ordinary-external.env"
    external.write_text("DEEPSEEK_API_KEY=ordinary-external\n", encoding="utf-8")
    assert external.stat().st_nlink == 1
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-must-be-replaced")

    experiment._load_and_validate_external_environment(
        external,
        base_url="https://example.invalid/v1",
        forbidden_roots=(snapshot, results),
    )

    assert experiment.os.environ["DEEPSEEK_API_KEY"] == "ordinary-external"


def test_external_env_file_accepts_qwen_provider_credential(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    results = tmp_path / "results"
    snapshot.mkdir()
    results.mkdir()
    external = tmp_path / "qwen-external.env"
    external.write_text("QWEN_API_KEY=qwen-test-only\n", encoding="utf-8")
    monkeypatch.setenv("QWEN_API_KEY", "ambient-must-be-replaced")

    experiment._load_and_validate_external_environment(
        external,
        base_url="https://example.invalid/v1",
        forbidden_roots=(snapshot, results),
    )

    assert experiment.os.environ["QWEN_API_KEY"] == "qwen-test-only"


@pytest.mark.parametrize(
    ("failure", "existing_status"),
    [
        ("Output root already exists; pass --resume", {"status": "succeeded", "marker": "completed"}),
        ("Output root already exists; pass --resume", {"status": "running", "marker": "active"}),
        ("The external environment file does not exist", {"status": "succeeded", "marker": "env-file"}),
        ("Output root already exists; pass --resume", None),
    ],
)
def test_main_prelease_failures_never_create_or_overwrite_formal_status(
    tmp_path,
    monkeypatch,
    capsys,
    failure,
    existing_status,
):
    out_root = tmp_path / "formal-results"
    out_root.mkdir()
    (out_root / "nonempty.marker").write_text("preserve\n", encoding="utf-8")
    status_path = out_root / "experiment_status.json"
    if existing_status is not None:
        status_path.write_text(json.dumps(existing_status), encoding="utf-8")
        before = status_path.read_bytes()
    else:
        before = None

    def fail_before_lease(*args, **kwargs):
        raise experiment.DefaultV1ExperimentError(failure)

    monkeypatch.setattr(experiment, "run_experiment", fail_before_lease)
    exit_code = experiment.main(
        [
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--out-root",
            str(out_root),
            "--expected-identity",
            "0" * 64,
            "--env-file",
            str(tmp_path / "credentials.env"),
        ]
    )

    assert exit_code == 1
    assert failure in capsys.readouterr().err
    if before is None:
        assert not status_path.exists()
    else:
        assert status_path.read_bytes() == before
