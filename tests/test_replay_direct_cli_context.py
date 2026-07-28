from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sts2_bench.experiment_identity import hash_file, hash_json
from sts2_bench.direct_cli_experiment import DirectCliExperimentError
import tools.replay_direct_cli_context as replay_cli
from tools.replay_direct_cli_context import (
    _source_provenance,
    load_preflight_environment,
    reconstruct_request,
    reserve_output_path,
    validate_output_path,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _json_sha(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_source_provenance_fixture(tmp_path: Path, *, invalidated: bool = False) -> dict:
    out_root = tmp_path / "formal"
    attempt_dir = out_root / "runs" / "seed_a" / "replicate_01" / "attempts" / "attempt_001"
    artifacts = attempt_dir / "artifacts"
    artifacts.mkdir(parents=True)
    protocol_label = "benchmark/experiments/protocol.json"
    benchmark_root = tmp_path / "benchmark"
    protocol_path = benchmark_root / "experiments" / "protocol.json"
    protocol = {
        "experiment_id": "exp",
        "seeds": ["seed_a"],
        "repeats_per_seed": 1,
        "model": "model",
        "base_url": "https://example.invalid/v1",
        "temperature": 0.0,
        "max_tokens": 64,
        "request_retries": 1,
        "history_chars": 24000,
        "max_decisions": 5000,
        "character": "Ironclad",
        "ascension": 0,
        "lang": "en",
        "request_timeout_sec": 180,
        "game_rules": {"ruleset_id": "rules", "source": "rules.py"},
    }
    _write_json(protocol_path, protocol)
    protocol_sha256 = hash_file(protocol_path)
    inventory = [
        {
            "label": protocol_label,
            "path": str(protocol_path.resolve()),
            "sha256": protocol_sha256,
            "size": protocol_path.stat().st_size,
            "tracked": True,
        }
    ]
    files = [{"label": protocol_label, "sha256": protocol_sha256, "size": protocol_path.stat().st_size}]
    identity_payload = {
        "protocol_sha256": protocol_sha256,
        "runtime": {"python": "test"},
        "files": files,
    }
    source_lock = hash_json(identity_payload)
    snapshot_identity = hash_json(files)
    _write_json(
        out_root / "source_lock.json",
        {
            **identity_payload,
            "identity_sha256": source_lock,
            "inventory": inventory,
            "source_tree_identity_sha256": snapshot_identity,
            "protocol_label": protocol_label,
            "protocol_path": str(protocol_path.resolve()),
            "benchmark_root": str(benchmark_root.resolve()),
            "sts2_cli_root": str((tmp_path / "sts2-cli").resolve()),
            "execution_mode": "active_worktree",
        },
    )
    _write_json(
        attempt_dir / "attempt_manifest.json",
        {
            "experiment_id": "exp",
            "seed": "seed_a",
            "replicate": 1,
            "attempt": 1,
            "formal_sample": True,
            "source_lock_sha256": source_lock,
            "source_tree_identity_sha256": snapshot_identity,
            "protocol_sha256": protocol_sha256,
            "command": [
                "python",
                "--out-dir",
                str(artifacts),
                "--seed",
                "seed_a",
                "--source-lock-sha256",
                source_lock,
                "--source-tree-identity-sha256",
                snapshot_identity,
            ],
        },
    )

    system_prompt = "system rules"
    visible = "Floor: 1\nChoose: "
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + visible
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    messages_sha256 = _json_sha(messages)
    _write_json(
        artifacts / "config.json",
        {
            "source_lock_sha256": source_lock,
            "source_tree_identity_sha256": snapshot_identity,
            "seed": "seed_a",
            "model": "model",
            "base_url": "https://example.invalid/v1",
            "temperature": 0.0,
            "max_tokens": 64,
            "request_retries": 1,
            "history_chars": 24000,
            "max_decisions": 5000,
            "character": "Ironclad",
            "ascension": 0,
            "lang": "en",
            "timeout": 180,
            "game_rules": {"ruleset_id": "rules"},
            "system_prompt": system_prompt,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        },
    )
    (artifacts / "terminal.txt").write_text(visible + "0\n", encoding="utf-8")
    event = {
        "type": "terminal_decision",
        "decision_index": 0,
        "terminal_window_start": 0,
        "terminal_window_end": len(visible),
        "terminal_window_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
        "model_messages_sha256": messages_sha256,
    }
    (artifacts / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    _write_json(
        artifacts / "summary.json",
        {"formal_behavior_sample": True, "termination_class": "natural_game_over"},
    )
    attempt_result = {
        "experiment_id": "exp",
        "seed": "seed_a",
        "replicate": 1,
        "attempt": 1,
        "status": "succeeded",
        "formal_sample": True,
        "termination_class": "natural_game_over",
        "identity_stable": True,
        "source_tree_stable": True,
        "source_lock_sha256": source_lock,
        "source_tree_identity_sha256": snapshot_identity,
        "protocol_sha256": protocol_sha256,
        "validation": {
            "valid": True,
            "request_audit": {
                "valid": True,
                "decisions": [
                    {
                        "decision_index": 0,
                        "valid": True,
                        "reconstructed_model_messages_sha256": messages_sha256,
                    }
                ],
            },
        },
    }
    _write_json(attempt_dir / "attempt_result.json", attempt_result)
    _write_json(attempt_dir.parent.parent / "job_status.json", {**attempt_result, "attempt_history": []})
    if invalidated:
        _write_json(
            out_root / "report_required.json",
            {
                "schema_version": 1,
                "experiment_id": "exp",
                "reason": "systematic_provider_completion_failure",
                "protocol_disposition": "invalidated_do_not_resume_or_count_for_paper",
            },
        )
    return {
        "out_root": out_root,
        "attempt_dir": attempt_dir,
        "artifacts": artifacts,
        "source_lock": source_lock,
        "snapshot_identity": snapshot_identity,
        "protocol_sha256": protocol_sha256,
        "messages_sha256": messages_sha256,
        "visible": visible,
    }


@pytest.mark.parametrize("output_limit", [[], ["--max-tokens", "4096", "--omit-max-tokens"]])
def test_replay_cli_requires_exactly_one_output_limit_control(tmp_path, output_limit):
    argv = [
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--decision-index",
        "0",
        "--env-file",
        str(tmp_path / "credentials.env"),
        "--output",
        str(tmp_path / "preflight.json"),
        *output_limit,
    ]

    with pytest.raises(SystemExit) as exc_info:
        replay_cli.main(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "recorded_event",
    [
        {
            "type": "terminal_decision",
            "model_messages_sha256": "a" * 64,
            "error": None,
            "output_format_valid": True,
            "provider_metadata": {"finish_reason": "stop"},
        },
        {
            "type": "model_protocol_error",
            "model_messages_sha256": "a" * 64,
            "error": "ValueError: finish_reason=length",
            "output_format_valid": None,
            "provider_metadata": {"finish_reason": "length"},
        },
    ],
    ids=["terminal-decision", "historical-length-failure"],
)
def test_replay_cli_omit_max_tokens_uses_provider_default_and_records_control(
    tmp_path,
    monkeypatch,
    recorded_event,
):
    source_root = tmp_path / "formal"
    artifacts = source_root / "artifacts"
    artifacts.mkdir(parents=True)
    for name in ("config.json", "events.jsonl", "terminal.txt"):
        (artifacts / name).write_text("{}\n", encoding="utf-8")
    env_file = tmp_path / "credentials.env"
    env_file.write_text("DEEPSEEK_API_KEY=test-only\n", encoding="utf-8")
    output = tmp_path / "preflight.json"
    captured: dict = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def complete(self, messages):
            captured["messages"] = messages
            captured["reservation_status_during_call"] = json.loads(
                output.read_text(encoding="utf-8")
            )["status"]
            return SimpleNamespace(
                text="0",
                usage={},
                provider_metadata={"finish_reason": "stop"},
            )

    monkeypatch.setattr(
        replay_cli,
        "_source_provenance",
        lambda _artifacts, *, decision_index: (source_root, {"source": "test", "decision_index": decision_index}),
    )
    monkeypatch.setattr(
        replay_cli,
        "reconstruct_request",
        lambda _artifacts, decision_index: (
            {
                "base_url": "https://example.invalid/v1",
                "model": "model",
                "temperature": 0.0,
                "timeout": 180,
                "max_tokens": 4096,
            },
            recorded_event,
            [{"role": "user", "content": "Choose: "}],
        ),
    )
    monkeypatch.setattr(replay_cli, "load_preflight_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_cli, "_selected_api_key", lambda: "test-only")
    monkeypatch.setattr(replay_cli, "_python_runtime_identity", lambda: {"runtime": "test"})
    monkeypatch.setattr(replay_cli, "OpenAICompatTerminalModel", FakeModel)

    returncode = replay_cli.main(
        [
            "--artifacts",
            str(artifacts),
            "--decision-index",
            "0",
            "--omit-max-tokens",
            "--env-file",
            str(env_file),
            "--output",
            str(output),
        ]
    )

    assert returncode == 0
    assert captured["reservation_status_during_call"] == "provider_call_started"
    assert captured["max_tokens"] is None
    assert captured["request_retries"] == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["preflight_max_tokens"] is None
    assert result["preflight_output_limit_control"] == "provider_default_omitted"
    assert result["preflight_provider_request"] == {
        "model": "model",
        "messages": captured["messages"],
        "temperature": 0.0,
    }
    assert "max_tokens" not in result["preflight_provider_request"]
    assert result["recorded_event_type"] == recorded_event["type"]
    assert result["recorded_event_error"] == recorded_event["error"]
    assert result["recorded_output_format_valid"] == recorded_event["output_format_valid"]
    assert result["recorded_provider_metadata"] == recorded_event["provider_metadata"]


@pytest.mark.parametrize(
    "damage",
    [
        "selection_name",
        "artifacts_path",
        "decision_index",
        "output_name",
        "source_hash",
        "terminal_decision_count",
        "character",
        "source_max_tokens",
        "non_midpoint",
        "historical_not_length",
        "model",
        "base_url",
        "explicit_max_tokens",
        "paper_sample",
        "output_root",
        "call_policy",
    ],
)
def test_replay_cli_rejects_selection_plan_mismatch_before_provider_call(
    tmp_path,
    monkeypatch,
    damage,
):
    fixture = _write_source_provenance_fixture(tmp_path)
    artifacts = fixture["artifacts"]
    monkeypatch.setattr(replay_cli, "REPO_ROOT", tmp_path)
    source_id = "source_a"
    selection_name = "chosen_context"
    plan_character = "Ironclad"
    plan_source_max_tokens = 64
    plan_terminal_decision_count = 1
    plan_selection = "early"
    plan_decision_index = 0
    if damage == "non_midpoint":
        event = json.loads((artifacts / "events.jsonl").read_text(encoding="utf-8"))
        events = [
            {**event, "decision_index": decision_index}
            for decision_index in range(3)
        ]
        (artifacts / "events.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in events),
            encoding="utf-8",
        )
        plan_terminal_decision_count = 3
        plan_selection = "middle"
    elif damage == "historical_not_length":
        event = json.loads((artifacts / "events.jsonl").read_text(encoding="utf-8"))
        event.update(
            {
                "type": "model_protocol_error",
                "provider_metadata": {"finish_reason": "stop"},
            }
        )
        (artifacts / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
        plan_terminal_decision_count = 0
        plan_selection = "historical_failure"
    elif damage == "terminal_decision_count":
        plan_terminal_decision_count = 2
    elif damage == "character":
        plan_character = "Silent"
    elif damage == "source_max_tokens":
        plan_source_max_tokens = 63
    plan_path = tmp_path / "selection-plan.json"
    selection_plan = {
        "schema_version": 1,
        "selection_plan_id": "selection-v1",
        "status": "locked_before_provider_calls",
        "model": "model",
        "base_url": "https://example.invalid/v1",
        "preflight_output_limit_control": "provider_default_omitted",
        "paper_sample": False,
        "output_root": ".",
        "call_policy": {
            "max_provider_calls_per_context": 1,
            "request_retries": 0,
            "failure_disposition": "preserve_and_invalidate_plan",
        },
        "sources": [
            {
                "source_id": source_id,
                "character": plan_character,
                "artifacts_path": artifacts.relative_to(tmp_path).as_posix(),
                "config_sha256": hash_file(artifacts / "config.json"),
                "events_sha256": hash_file(artifacts / "events.jsonl"),
                "terminal_sha256": hash_file(artifacts / "terminal.txt"),
                "terminal_decision_count": plan_terminal_decision_count,
                "source_max_tokens": plan_source_max_tokens,
            }
        ],
        "contexts": [
            {
                "name": selection_name,
                "source_id": source_id,
                "character": plan_character,
                "selection": plan_selection,
                "decision_index": plan_decision_index,
            }
        ],
    }
    if damage == "model":
        selection_plan["model"] = "other-model"
    elif damage == "base_url":
        selection_plan["base_url"] = "https://other.invalid/v1"
    elif damage == "paper_sample":
        selection_plan["paper_sample"] = True
    elif damage == "output_root":
        selection_plan["output_root"] = "different-output-root"
    elif damage == "call_policy":
        selection_plan["call_policy"]["max_provider_calls_per_context"] = 2
    _write_json(plan_path, selection_plan)
    selected_artifacts = artifacts
    selected_name = selection_name
    selected_index = 0
    output = tmp_path / f"{selection_name}.json"

    if damage == "selection_name":
        selected_name = "not_in_plan"
    elif damage == "artifacts_path":
        selected_artifacts = tmp_path / "other-artifacts"
        selected_artifacts.mkdir()
    elif damage == "decision_index":
        selected_index = 1
    elif damage == "output_name":
        output = tmp_path / "wrong-output-name.json"
    elif damage == "source_hash":
        (artifacts / "terminal.txt").write_text("drifted\n", encoding="utf-8")
    output_limit_args = (
        ["--max-tokens", "4096"]
        if damage == "explicit_max_tokens"
        else ["--omit-max-tokens"]
    )

    def forbidden_after_plan_validation(*args, **kwargs):
        raise AssertionError("selection-plan mismatch reached provider preparation")

    monkeypatch.setattr(replay_cli, "_source_provenance", forbidden_after_plan_validation)
    monkeypatch.setattr(replay_cli, "OpenAICompatTerminalModel", forbidden_after_plan_validation)

    with pytest.raises(ValueError, match="[Ss]election|[Rr]eplay|[Oo]utput|hashes drifted"):
        replay_cli.main(
            [
                "--artifacts",
                str(selected_artifacts),
                "--decision-index",
                str(selected_index),
                *output_limit_args,
                "--env-file",
                str(tmp_path / "unused.env"),
                "--output",
                str(output),
                "--selection-plan",
                str(plan_path),
                "--selection-name",
                selected_name,
            ]
        )

    assert not output.exists()


def test_reconstruct_request_is_exactly_bound_to_recorded_terminal_window(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    system_prompt = "system rules"
    visible = "Floor: 1\nChoose: "
    terminal = "old\n" + visible + "0\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + visible
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    _write_json(artifacts / "config.json", {"system_prompt": system_prompt})
    (artifacts / "terminal.txt").write_text(terminal, encoding="utf-8")
    event = {
        "type": "terminal_decision",
        "decision_index": 7,
        "terminal_window_start": 4,
        "terminal_window_end": 4 + len(visible),
        "terminal_window_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
        "model_messages_sha256": _json_sha(messages),
    }
    (artifacts / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    _config, recorded, rebuilt = reconstruct_request(artifacts, decision_index=7)

    assert recorded == event
    assert rebuilt == messages


def test_validate_output_path_rejects_source_tree_and_overwrite(tmp_path):
    source_root = tmp_path / "formal"
    source_root.mkdir()

    with pytest.raises(ValueError, match="outside the source formal result root"):
        validate_output_path(source_root / "preflight.json", source_root=source_root)

    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        validate_output_path(existing, source_root=source_root)

    assert validate_output_path(tmp_path / "new.json", source_root=source_root) == (tmp_path / "new.json").resolve()


def test_output_reservation_is_exclusive_commits_once_and_cleans_up_before_provider_call(
    tmp_path,
):
    source_root = tmp_path / "formal"
    source_root.mkdir()
    output = tmp_path / "preflight.json"

    with reserve_output_path(output, source_root=source_root) as reservation:
        assert output.is_file()
        with pytest.raises(ValueError, match="overwrite"):
            reserve_output_path(output, source_root=source_root)
        reservation.mark_provider_call_started()
        with pytest.raises(ValueError, match="another provider call"):
            reservation.mark_provider_call_started()
        reservation.commit({"status": "passed"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "passed"}
    with pytest.raises(ValueError, match="overwrite"):
        reserve_output_path(output, source_root=source_root)

    aborted = tmp_path / "aborted.json"
    with pytest.raises(RuntimeError, match="abort"):
        with reserve_output_path(aborted, source_root=source_root):
            raise RuntimeError("abort")
    assert not aborted.exists()


def test_output_reservation_preserves_post_provider_start_failure_and_blocks_retry(tmp_path):
    source_root = tmp_path / "formal"
    source_root.mkdir()
    output = tmp_path / "provider-started.json"

    with pytest.raises(RuntimeError, match="abort after provider start"):
        with reserve_output_path(output, source_root=source_root) as reservation:
            reservation.mark_provider_call_started()
            raise RuntimeError("abort after provider start")

    marker = json.loads(output.read_text(encoding="utf-8"))
    assert marker["schema_version"] == 1
    assert marker["status"] == "provider_call_started"
    assert marker["pid"] > 0
    assert marker["token"]
    assert marker["started_at"]
    with pytest.raises(ValueError, match="overwrite"):
        reserve_output_path(output, source_root=source_root)


def test_preflight_environment_rejects_source_root_output_tree_and_hardlink_alias(tmp_path):
    source_root = tmp_path / "formal"
    output_parent = tmp_path / "preflight"
    source_root.mkdir()
    output_parent.mkdir()
    source_env = source_root / "credentials.env"
    source_env.write_text("DEEPSEEK_API_KEY=test-only\n", encoding="utf-8")

    with pytest.raises(DirectCliExperimentError, match="outside the formal result root"):
        load_preflight_environment(
            source_env,
            base_url="https://example.invalid/v1",
            source_root=source_root,
            output_parent=output_parent,
        )

    output_env = output_parent / "credentials.env"
    output_env.write_text("DEEPSEEK_API_KEY=test-only\n", encoding="utf-8")
    with pytest.raises(DirectCliExperimentError, match="outside the formal result root"):
        load_preflight_environment(
            output_env,
            base_url="https://example.invalid/v1",
            source_root=source_root,
            output_parent=output_parent,
        )

    hardlink_alias = tmp_path / "source-credential-alias.env"
    os.link(source_env, hardlink_alias)
    with pytest.raises(DirectCliExperimentError, match="single-link regular file"):
        load_preflight_environment(
            hardlink_alias,
            base_url="https://example.invalid/v1",
            source_root=source_root,
            output_parent=output_parent,
        )


def test_preflight_environment_rejects_resolved_alias_into_source_root(tmp_path):
    source_root = tmp_path / "formal"
    output_parent = tmp_path / "preflight"
    source_root.mkdir()
    output_parent.mkdir()
    source_env = source_root / "credentials.env"
    source_env.write_text("DEEPSEEK_API_KEY=test-only\n", encoding="utf-8")
    alias = tmp_path / "source-credential-symlink.env"
    try:
        alias.symlink_to(source_env)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(DirectCliExperimentError, match="outside the formal result root"):
        load_preflight_environment(
            alias,
            base_url="https://example.invalid/v1",
            source_root=source_root,
            output_parent=output_parent,
        )


def test_source_provenance_binds_attempt_to_source_lock(tmp_path):
    fixture = _write_source_provenance_fixture(tmp_path)
    artifacts = fixture["artifacts"]

    source_root, provenance = _source_provenance(artifacts, decision_index=0)

    assert source_root == fixture["out_root"]
    assert provenance["source_lock_sha256"] == fixture["source_lock"]
    assert provenance["source_tree_identity_sha256"] == fixture["snapshot_identity"]
    assert provenance["source_eligibility"] == "accepted_formal_sample"
    assert provenance["protocol_sha256"] == fixture["protocol_sha256"]
    assert provenance["accepted_model_messages_sha256"] == fixture["messages_sha256"]

    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    config["source_lock_sha256"] = "tampered"
    _write_json(artifacts / "config.json", config)
    with pytest.raises(ValueError, match="not bound"):
        _source_provenance(artifacts, decision_index=0)


def test_source_provenance_rejects_coordinated_prompt_and_event_hash_tampering(tmp_path):
    fixture = _write_source_provenance_fixture(tmp_path)
    artifacts = fixture["artifacts"]
    config_path = artifacts / "config.json"
    events_path = artifacts / "events.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["system_prompt"] = "tampered system rules"
    config["system_prompt_sha256"] = hashlib.sha256(config["system_prompt"].encode("utf-8")).hexdigest()
    _write_json(config_path, config)
    tampered_messages = [
        {"role": "system", "content": config["system_prompt"]},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + fixture["visible"]
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    event = json.loads(events_path.read_text(encoding="utf-8"))
    event["model_messages_sha256"] = _json_sha(tampered_messages)
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Source request"):
        _source_provenance(artifacts, decision_index=0)


def test_source_provenance_records_root_protocol_invalidation(tmp_path):
    fixture = _write_source_provenance_fixture(tmp_path, invalidated=True)

    source_root, provenance = _source_provenance(fixture["artifacts"], decision_index=0)

    report_path = fixture["out_root"] / "report_required.json"
    assert source_root == fixture["out_root"]
    assert provenance["source_eligibility"] == "invalidated_protocol_context"
    assert provenance["attempt_eligibility"] == "accepted_formal_sample"
    assert provenance["source_protocol_disposition"] == "invalidated_do_not_resume_or_count_for_paper"
    assert provenance["source_report_reason"] == "systematic_provider_completion_failure"
    assert provenance["source_report_required_sha256"] == hash_file(report_path)


def test_source_provenance_rejects_active_source_drift(tmp_path):
    out_root = tmp_path / "formal"
    attempt_dir = out_root / "runs" / "seed_a" / "replicate_01" / "attempts" / "attempt_001"
    artifacts = attempt_dir / "artifacts"
    artifacts.mkdir(parents=True)
    protocol_label = "benchmark/experiments/protocol.json"
    protocol_path = tmp_path / "benchmark" / "experiments" / "protocol.json"
    _write_json(protocol_path, {"experiment_id": "exp"})
    inventory = [
        {
            "label": protocol_label,
            "path": str(protocol_path.resolve()),
            "sha256": hash_file(protocol_path),
            "size": protocol_path.stat().st_size,
            "tracked": True,
        }
    ]
    files = [
        {"label": item["label"], "sha256": item["sha256"], "size": item["size"]}
        for item in inventory
    ]
    identity_payload = {"protocol_sha256": hash_file(protocol_path), "runtime": {}, "files": files}
    _write_json(
        out_root / "source_lock.json",
        {
            **identity_payload,
            "identity_sha256": hash_json(identity_payload),
            "inventory": inventory,
                "source_tree_identity_sha256": hash_json(files),
                "protocol_label": protocol_label,
                "protocol_path": str(protocol_path.resolve()),
                "benchmark_root": str((tmp_path / "benchmark").resolve()),
                "sts2_cli_root": str((tmp_path / "sts2-cli").resolve()),
                "execution_mode": "active_worktree",
        },
    )
    _write_json(attempt_dir / "attempt_manifest.json", {})
    _write_json(attempt_dir / "attempt_result.json", {})
    _write_json(attempt_dir.parent.parent / "job_status.json", {"attempt_history": []})
    _write_json(artifacts / "summary.json", {})
    _write_json(artifacts / "config.json", {})
    protocol_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="active worktree is invalid"):
        _source_provenance(artifacts, decision_index=0)
