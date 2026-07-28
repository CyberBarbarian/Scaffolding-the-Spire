import copy
import hashlib
import json
from pathlib import Path

import pytest

import tools.continue_direct_cli_trajectory as continuation
import sts2_bench.direct_cli_continuation as continuation_validator
from sts2_bench.direct_cli_baseline import ModelReply


def _write_engine_prefix(path):
    rows = [
        {"step": 0, "type": "state", "data": {"type": "ready"}},
        {
            "step": 1,
            "type": "state",
            "data": {
                "type": "decision",
                "decision": "combat_play",
                "context": {"act": 1, "floor": 17},
                "player": {"hp": 14, "energy": 3},
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _plan(tmp_path):
    replay_messages = [{"role": "user", "content": "decision 0"}]
    boundary_messages = [{"role": "user", "content": "decision 1"}]
    engine_trace = tmp_path / "engine_trace.jsonl"
    engine_rows = _write_engine_prefix(engine_trace)
    source_attempt = tmp_path / "source_attempt"
    source_artifacts = source_attempt / "artifacts"
    source_artifacts.mkdir(parents=True)
    source_paths = {
        "config.json": source_artifacts / "config.json",
        "events.jsonl": source_artifacts / "events.jsonl",
        "terminal.txt": source_artifacts / "terminal.txt",
        "engine_trace.jsonl": source_artifacts / "engine_trace.jsonl",
        "attempt_manifest.json": source_attempt / "attempt_manifest.json",
        "attempt_result.json": source_attempt / "attempt_result.json",
    }
    for name, path in source_paths.items():
        path.write_text(f"{name}\n", encoding="utf-8")
    config = {
        "base_url": "https://provider.invalid/v1",
        "model": "test-model",
        "temperature": 0.0,
        "max_tokens": None,
        "request_retries": 2,
    }
    plan = {
        "adapter_sha256": continuation.hash_file(Path(continuation.__file__)),
        "source_attempt": str(source_attempt),
        "source_artifacts": str(source_artifacts),
        "source_hashes": {
            name: continuation.hash_file(path) for name, path in source_paths.items()
        },
        "source_lock_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
        "config": config,
        "decisions": [
            {
                "decision_index": 0,
                "model_messages_sha256": continuation._messages_sha256(
                    replay_messages
                ),
                "model_raw": "e",
                "command": "e",
            }
        ],
        "boundary": {
            "decision_index": 1,
            "model_messages_sha256": continuation._messages_sha256(
                boundary_messages
            ),
            "provider_request_sha256": continuation._provider_request_sha256(
                boundary_messages,
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            ),
            "engine_prefix_row_count": len(engine_rows),
            "engine_prefix_sha256": continuation.hash_json(
                continuation._canonical_engine_prefix(engine_rows)
            ),
            "state_sha256": continuation.hash_json(engine_rows[-1]["data"]),
        },
    }
    source_lock = {
        "identity_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
    }
    return plan, replay_messages, boundary_messages, engine_trace, source_lock


def _protocol_retry_plan(tmp_path):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    plan["config"]["request_retries"] = 1
    retry_messages = [
        *boundary_messages,
        {
            "role": "user",
            "content": continuation.OUTPUT_RETRY_MESSAGE,
        },
    ]
    empty_body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": ""},
                    "finish_reason": "stop",
                }
            ],
            "model": plan["config"]["model"],
            "id": "empty-response",
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        },
        separators=(",", ":"),
    )
    empty_attempt = {
        "provider_attempt_index": 0,
        "model_messages_sha256": plan["boundary"]["model_messages_sha256"],
        "provider_request_sha256": plan["boundary"]["provider_request_sha256"],
        "status": "failed",
        "failure_class": "model_protocol_error",
        "error": "ValueError: LLM response content is empty",
        "latency_sec": 1.25,
        "provider_metadata": {
            "model": plan["config"]["model"],
            "id": "empty-response",
            "finish_reason": "stop",
        },
        "provider_response_sha256": hashlib.sha256(
            empty_body.encode("utf-8")
        ).hexdigest(),
        "provider_response_encoding": "utf-8",
        "provider_response_body": empty_body,
    }
    retry_attempt = {
        "provider_attempt_index": 1,
        "model_messages_sha256": continuation._messages_sha256(retry_messages),
        "provider_request_sha256": continuation._provider_request_sha256(
            retry_messages,
            model=plan["config"]["model"],
            temperature=plan["config"]["temperature"],
            max_tokens=plan["config"]["max_tokens"],
        ),
        "status": "failed",
        "failure_class": "provider_transport_error",
        "error": "RemoteDisconnected: remote closed",
        "latency_sec": 0.5,
        "provider_metadata": {},
    }
    plan["boundary"].update(
        {
            "first_live_dispatch_model_messages_sha256": retry_attempt[
                "model_messages_sha256"
            ],
            "first_live_dispatch_provider_request_sha256": retry_attempt[
                "provider_request_sha256"
            ],
            "protocol_retry_prelude": {
                "kind": "audited_empty_content_then_retry_transport_failure_v1",
                "model": plan["config"]["model"],
                "temperature": plan["config"]["temperature"],
                "max_tokens": plan["config"]["max_tokens"],
                "retry_message_sha256": hashlib.sha256(
                    continuation.OUTPUT_RETRY_MESSAGE.encode("utf-8")
                ).hexdigest(),
                "source_provider_attempts": [empty_attempt, retry_attempt],
                "source_provider_attempts_sha256": continuation.hash_json(
                    [empty_attempt, retry_attempt]
                ),
                "replayed_empty_provider_attempt": empty_attempt,
                "replayed_empty_usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                },
                "replayed_empty_latency_sec": 1.25,
            },
        }
    )
    return (
        plan,
        replay_messages,
        boundary_messages,
        retry_messages,
        engine_trace,
        source_lock,
    )


def _model(plan, engine_trace, source_lock, *, verify_only=True):
    return continuation.ReplayThenLiveModel(
        plan,
        replay_engine_trace=engine_trace,
        verify_only=verify_only,
        source_lock_verifier=lambda: source_lock,
    )


def test_verify_only_replays_locally_and_stops_with_zero_provider_calls(tmp_path):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    model = _model(plan, engine_trace, source_lock)
    live_calls = []
    model._complete_live = lambda messages: live_calls.append(messages)

    reply = model.complete(replay_messages)

    assert reply.text == "e"
    assert reply.provider_attempt_count == 0
    assert reply.provider_metadata == {}
    with pytest.raises(
        continuation.ReplayBoundaryReached,
        match="verify-only Direct continuation seam reached",
    ):
        model.complete(boundary_messages)

    assert live_calls == []
    assert model.provider_calls == 0
    assert model.replayed_decisions == 1
    assert model.seam_proof["provider_calls"] == 0
    assert model.seam_proof["credential_file_read"] is False


def test_protocol_retry_verify_only_keeps_base_seam_and_zero_provider_calls(tmp_path):
    (
        plan,
        replay_messages,
        boundary_messages,
        _retry_messages,
        engine_trace,
        source_lock,
    ) = _protocol_retry_plan(tmp_path)
    model = _model(plan, engine_trace, source_lock)
    model.complete(replay_messages)

    with pytest.raises(continuation.ReplayBoundaryReached):
        model.complete(boundary_messages)

    assert model.provider_calls == 0
    assert model.credential_file_read is False
    assert model.seam_proof["model_messages_sha256"] == plan["boundary"][
        "model_messages_sha256"
    ]
    assert model.seam_proof["provider_request_sha256"] == plan["boundary"][
        "provider_request_sha256"
    ]
    assert model.seam_proof[
        "first_live_dispatch_provider_request_sha256"
    ] == plan["boundary"]["first_live_dispatch_provider_request_sha256"]


def test_protocol_retry_live_dispatches_only_retry_request(tmp_path, monkeypatch):
    (
        plan,
        replay_messages,
        boundary_messages,
        retry_messages,
        engine_trace,
        source_lock,
    ) = _protocol_retry_plan(tmp_path)
    provider_messages = []
    observed_attempts = []
    dispatches = []

    def fake_provider(model, messages):
        provider_messages.append(copy.deepcopy(messages))
        model.attempt_observer(
            {
                "provider_attempt_index": 0,
                "model_messages_sha256": continuation._messages_sha256(messages),
                "provider_request_sha256": continuation._provider_request_sha256(
                    messages,
                    model=model.model,
                    temperature=model.temperature,
                    max_tokens=model.max_tokens,
                ),
                "status": "succeeded",
                "failure_class": None,
                "error": None,
                "latency_sec": 0.25,
                "provider_metadata": {"finish_reason": "stop"},
            }
        )
        return ModelReply(
            text="0",
            usage={"prompt_tokens": 7, "completion_tokens": 11},
            latency_sec=0.25,
            provider_attempt_count=1,
            provider_metadata={"finish_reason": "stop"},
        )

    monkeypatch.setattr(
        continuation.OpenAICompatTerminalModel,
        "complete",
        fake_provider,
    )
    model = continuation.ReplayThenLiveModel(
        plan,
        replay_engine_trace=engine_trace,
        verify_only=False,
        source_lock_verifier=lambda: source_lock,
        credential_loader=lambda: "secret",
        dispatch_persistor=dispatches.append,
    )
    model.attempt_observer = observed_attempts.append
    model.complete(replay_messages)
    reply = model.complete(boundary_messages)

    assert provider_messages == [retry_messages]
    assert boundary_messages not in provider_messages
    assert [item["provider_attempt_index"] for item in observed_attempts] == [0, 1]
    assert observed_attempts[0]["failure_class"] == "model_protocol_error"
    assert observed_attempts[1]["status"] == "succeeded"
    assert dispatches[0]["model_messages_sha256"] == continuation._messages_sha256(
        retry_messages
    )
    assert dispatches[0]["provider_request_sha256"] == plan["boundary"][
        "first_live_dispatch_provider_request_sha256"
    ]
    assert reply.provider_attempt_count == 2
    assert reply.usage == {"prompt_tokens": 10, "completion_tokens": 16}
    assert reply.latency_sec == pytest.approx(1.5)


def test_protocol_retry_once_guard_precedes_replayed_empty_attempt(
    tmp_path,
    monkeypatch,
):
    (
        plan,
        replay_messages,
        boundary_messages,
        _retry_messages,
        engine_trace,
        source_lock,
    ) = _protocol_retry_plan(tmp_path)
    provider_calls = []
    observed_attempts = []

    def fail_provider(_model, messages):
        provider_calls.append(copy.deepcopy(messages))
        raise TimeoutError("ambiguous retry failure")

    monkeypatch.setattr(
        continuation.OpenAICompatTerminalModel,
        "complete",
        fail_provider,
    )
    model = continuation.ReplayThenLiveModel(
        plan,
        replay_engine_trace=engine_trace,
        verify_only=False,
        source_lock_verifier=lambda: source_lock,
        credential_loader=lambda: "secret",
    )
    model.attempt_observer = observed_attempts.append
    model.complete(replay_messages)

    with pytest.raises(TimeoutError, match="ambiguous retry failure"):
        model.complete(boundary_messages)
    with pytest.raises(
        continuation.ContinuationError,
        match="Provider dispatch already claimed",
    ):
        model.complete(boundary_messages)

    assert len(provider_calls) == 1
    assert len(observed_attempts) == 1
    assert observed_attempts[0]["provider_attempt_index"] == 0


def test_protocol_retry_validator_binds_base_seam_and_retry_dispatch(tmp_path):
    (
        plan,
        _replay_messages,
        boundary_messages,
        _retry_messages,
        _engine_trace,
        _source_lock,
    ) = _protocol_retry_plan(tmp_path)
    request_payload = {
        "model": plan["config"]["model"],
        "messages": boundary_messages,
        "temperature": plan["config"]["temperature"],
    }

    assert continuation_validator._boundary_request_contract_valid(
        plan=plan,
        boundary_messages=boundary_messages,
        request_payload=request_payload,
    )

    tampered = copy.deepcopy(plan)
    tampered["boundary"]["first_live_dispatch_provider_request_sha256"] = (
        tampered["boundary"]["provider_request_sha256"]
    )
    assert not continuation_validator._boundary_request_contract_valid(
        plan=tampered,
        boundary_messages=boundary_messages,
        request_payload=request_payload,
    )


def test_replay_message_mismatch_fails_closed_before_provider(tmp_path):
    plan, _, _, engine_trace, source_lock = _plan(tmp_path)
    model = _model(plan, engine_trace, source_lock)
    live_calls = []
    model._complete_live = lambda messages: live_calls.append(messages)

    with pytest.raises(continuation.ContinuationError, match="Replay prompt diverged"):
        model.complete([{"role": "user", "content": "changed"}])

    assert live_calls == []
    assert model.provider_calls == 0
    assert model.seam_proof is None


def test_boundary_request_mismatch_fails_closed_before_source_or_provider(tmp_path):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    plan["boundary"]["provider_request_sha256"] = "wrong-request"
    source_checks = []
    model = continuation.ReplayThenLiveModel(
        plan,
        replay_engine_trace=engine_trace,
        verify_only=True,
        source_lock_verifier=lambda: source_checks.append(True) or source_lock,
    )
    live_calls = []
    model._complete_live = lambda messages: live_calls.append(messages)
    model.complete(replay_messages)

    with pytest.raises(
        continuation.ContinuationError,
        match="provider request seam does not match",
    ):
        model.complete(boundary_messages)

    assert source_checks == []
    assert live_calls == []
    assert model.provider_calls == 0


def test_source_lock_mismatch_fails_closed_before_engine_or_provider(tmp_path):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    drifted_lock = {**source_lock, "source_tree_identity_sha256": "drifted"}
    model = _model(plan, engine_trace, drifted_lock)
    model.complete(replay_messages)
    engine_trace.unlink()

    with pytest.raises(continuation.ContinuationError, match="Source lock drifted"):
        model.complete(boundary_messages)

    assert model.provider_calls == 0
    assert model.seam_proof is None


def test_source_artifact_mismatch_fails_closed_before_provider(tmp_path):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    model = _model(plan, engine_trace, source_lock)
    model.complete(replay_messages)
    source_terminal = Path(plan["source_artifacts"]) / "terminal.txt"
    source_terminal.write_text("drifted\n", encoding="utf-8")

    with pytest.raises(
        continuation.ContinuationError,
        match="Source attempt artifacts drifted",
    ):
        model.complete(boundary_messages)

    assert model.provider_calls == 0
    assert model.seam_proof is None


def test_engine_prefix_or_state_mismatch_fails_closed_before_provider(tmp_path):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    model = _model(plan, engine_trace, source_lock)
    model.complete(replay_messages)
    rows = _write_engine_prefix(engine_trace)
    rows[-1]["data"]["player"]["hp"] = 13
    engine_trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        continuation.ContinuationError,
        match="engine trajectory diverged",
    ):
        model.complete(boundary_messages)

    assert model.provider_calls == 0
    assert model.seam_proof is None


def test_non_verify_seam_delegates_only_after_all_checks(tmp_path):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    model = _model(
        copy.deepcopy(plan),
        engine_trace,
        source_lock,
        verify_only=False,
    )
    delegated = []

    def fake_live(messages):
        delegated.append(messages)
        model.provider_calls += 1
        return ModelReply(text="0", usage={}, latency_sec=0.0)

    model._complete_live = fake_live
    model.complete(replay_messages)
    reply = model.complete(boundary_messages)
    later_messages = [{"role": "user", "content": "decision 2"}]
    later_reply = model.complete(later_messages)

    assert reply.text == "0"
    assert later_reply.text == "0"
    assert delegated == [boundary_messages, later_messages]
    assert model.provider_calls == 2
    assert model.cursor == 3
    assert model.seam_proof["status"] == "verified"


def test_replay_restores_original_provider_evidence_and_metrics(tmp_path):
    plan, replay_messages, _, engine_trace, source_lock = _plan(tmp_path)
    provider_attempt = {
        "provider_attempt_index": 0,
        "model_messages_sha256": plan["decisions"][0][
            "model_messages_sha256"
        ],
        "provider_request_sha256": "request",
        "status": "succeeded",
        "failure_class": None,
        "error": None,
        "latency_sec": 1.25,
        "provider_metadata": {"finish_reason": "stop"},
        "provider_response_sha256": "response",
        "provider_response_body": "{\"ok\":true}",
    }
    plan["decisions"][0].update(
        {
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "latency_sec": 1.25,
            "provider_attempt_count": 1,
            "provider_metadata": {"finish_reason": "stop"},
            "provider_attempts": [provider_attempt],
        }
    )
    model = _model(plan, engine_trace, source_lock)
    observed = []
    model.attempt_observer = observed.append

    reply = model.complete(replay_messages)

    assert observed == [provider_attempt]
    assert reply.usage == {"prompt_tokens": 3, "completion_tokens": 2}
    assert reply.latency_sec == 1.25
    assert reply.provider_attempt_count == 1
    assert reply.provider_metadata == {"finish_reason": "stop"}
    assert model.provider_calls == 0


def test_live_persists_seam_before_credential_and_dispatch(tmp_path, monkeypatch):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    order = []

    def fake_provider_complete(_model, _messages):
        order.append("provider")
        return ModelReply(text="0", usage={}, latency_sec=0.1)

    monkeypatch.setattr(
        continuation.OpenAICompatTerminalModel,
        "complete",
        fake_provider_complete,
    )
    model = continuation.ReplayThenLiveModel(
        plan,
        replay_engine_trace=engine_trace,
        verify_only=False,
        source_lock_verifier=lambda: source_lock,
        seam_persistor=lambda value: order.append(
            f"seam:{value['credential_file_read']}"
        ),
        credential_loader=lambda: order.append("credential") or "secret",
        dispatch_persistor=lambda _value: order.append("dispatch"),
    )

    model.complete(replay_messages)
    reply = model.complete(boundary_messages)

    assert reply.text == "0"
    assert order == [
        "seam:False",
        "dispatch",
        "credential",
        "provider",
    ]
    assert model.seam_proof["credential_file_read"] is False
    assert model.provider_calls == 1
    assert model.credential_file_read is True


def test_failed_boundary_dispatch_cannot_be_reissued(tmp_path, monkeypatch):
    plan, replay_messages, boundary_messages, engine_trace, source_lock = _plan(
        tmp_path
    )
    provider_attempts = []

    def fail_provider(_model, _messages):
        provider_attempts.append(True)
        raise TimeoutError("ambiguous provider timeout")

    monkeypatch.setattr(
        continuation.OpenAICompatTerminalModel,
        "complete",
        fail_provider,
    )
    model = continuation.ReplayThenLiveModel(
        plan,
        replay_engine_trace=engine_trace,
        verify_only=False,
        source_lock_verifier=lambda: source_lock,
        credential_loader=lambda: "secret",
    )
    model.complete(replay_messages)

    with pytest.raises(TimeoutError, match="ambiguous provider timeout"):
        model.complete(boundary_messages)
    with pytest.raises(
        continuation.ContinuationError,
        match="Provider dispatch already claimed",
    ):
        model.complete(boundary_messages)

    assert provider_attempts == [True]
    assert model.provider_calls == 1
    assert model.cursor == 1


def test_claim_id_uses_only_stable_source_boundary_invariants(tmp_path):
    plan, _, _, _, _ = _plan(tmp_path)
    first = continuation._continuation_claim_id(plan)
    changed_verification = copy.deepcopy(plan)
    changed_verification["adapter_sha256"] = "new-adapter"
    changed_verification["verification_sha256"] = "new-verification"
    assert continuation._continuation_claim_id(changed_verification) == first

    changed_boundary = copy.deepcopy(plan)
    changed_boundary["boundary"]["provider_request_sha256"] = "different"
    assert continuation._continuation_claim_id(changed_boundary) != first


def test_atomic_claim_rejects_second_owner_without_overwriting(tmp_path):
    claim_path = tmp_path / "claims" / "stable.json"
    first = {"owner": "first", "provider_dispatch_started": False}

    continuation._atomic_create(claim_path, first)

    with pytest.raises(
        continuation.ContinuationError,
        match="Continuation claim already exists",
    ):
        continuation._atomic_create(
            claim_path,
            {"owner": "second", "provider_dispatch_started": False},
        )

    assert json.loads(claim_path.read_text(encoding="utf-8")) == first


def test_atomic_seam_rejects_rewrite_without_overwriting(tmp_path):
    seam_path = tmp_path / "continuation" / "seam_proof.json"
    first = {"status": "verified", "state_sha256": "state-a"}

    continuation._atomic_create(seam_path, first)

    with pytest.raises(
        continuation.ContinuationError,
        match="Continuation claim already exists",
    ):
        continuation._atomic_create(
            seam_path,
            {"status": "verified", "state_sha256": "state-b"},
        )

    assert json.loads(seam_path.read_text(encoding="utf-8")) == first


def test_predispatch_failure_materialization_is_idempotent(tmp_path, monkeypatch):
    attempt_dir = tmp_path / "attempts" / "attempt_002"
    continuation_dir = attempt_dir / "continuation"
    artifacts = attempt_dir / "artifacts"
    continuation_dir.mkdir(parents=True)
    artifacts.mkdir()
    claim_path = tmp_path / "claims" / "stable.json"
    claim_path.parent.mkdir()
    engine_prefix = [
        {
            "step": 0,
            "ts": "source-timestamp",
            "type": "state",
            "data": {"type": "decision", "decision": "combat_play"},
        }
    ]
    plan = {
        "source_attempt": "source-attempt",
        "accepted_decision_count": 1,
        "decisions": [
            {
                "decision_index": 0,
                "model_messages_sha256": "messages-0",
                "model_raw": "e",
                "command": "e",
                "usage": {},
                "latency_sec": 0.0,
                "provider_attempt_count": 0,
                "provider_metadata": {},
                "provider_attempts": [],
            }
        ],
        "boundary": {
            "decision_index": 1,
            "model_messages_sha256": "messages",
            "provider_request_sha256": "request",
            "engine_prefix_row_count": len(engine_prefix),
            "engine_prefix_sha256": continuation.hash_json(
                continuation._canonical_engine_prefix(engine_prefix)
            ),
            "state_sha256": continuation.hash_json(engine_prefix[-1]["data"]),
        },
    }
    claim = {
        "claim_id": "stable",
        "source_attempt": "source-attempt",
        "boundary_decision_index": 1,
        "boundary_provider_request_sha256": "request",
        "provider_dispatch_started": False,
    }
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    (attempt_dir / "attempt_manifest.json").write_text(
        json.dumps(
            {
                "attempt": 2,
                "execution_mode": "trajectory_continuation",
                "continuation": {
                    "claim_id": "stable",
                    "source_attempt": "source-attempt",
                },
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "summary.json").write_text(
        json.dumps(
            {
                "formal_behavior_sample": False,
                "termination_class": "process_infrastructure_error",
                "error": "local write failed",
            }
        ),
        encoding="utf-8",
    )
    terminal_decision = {
        "type": "terminal_decision",
        "decision_index": 0,
        "model_messages_sha256": "messages-0",
        "model_raw": "e",
        "command": "e",
        "usage": {},
        "latency_sec": 0.0,
        "provider_attempt_count": 0,
        "provider_metadata": {},
    }
    (artifacts / "events.jsonl").write_text(
        json.dumps(terminal_decision) + "\n",
        encoding="utf-8",
    )
    (artifacts / "config.json").write_text(
        json.dumps(
            {
                "history_chars": 100,
                "model": "test-model",
                "temperature": 0.0,
                "max_tokens": None,
                "request_retries": 0,
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "terminal.txt").write_text("", encoding="utf-8")
    replay_engine_rows = [
        {**engine_prefix[0], "ts": "different-replay-timestamp"},
        {"step": 0, "type": "action", "data": {"cmd": "quit"}},
        {
            "step": 1,
            "type": "state",
            "data": {"type": "quit_result", "success": True},
        },
    ]
    (artifacts / "engine_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in replay_engine_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        continuation,
        "_audit_terminal_decision_requests",
        lambda *_args, **_kwargs: {"valid": True},
    )
    (continuation_dir / "seam_proof.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "decision_index": 1,
                "replayed_decisions": 1,
                "provider_calls": 0,
                "credential_file_read": False,
                "credential_values_recorded": False,
                "model_messages_sha256": "messages",
                "provider_request_sha256": "request",
                "engine_prefix_sha256": plan["boundary"]["engine_prefix_sha256"],
                "engine_prefix_raw_sha256": "source-raw-hash-with-other-timestamps",
                "state_sha256": plan["boundary"]["state_sha256"],
            }
        ),
        encoding="utf-8",
    )

    first = continuation._predispatch_failure_result(
        attempt_dir,
        plan=plan,
        claim_path=claim_path,
    )
    first_hash = continuation.hash_file(attempt_dir / "attempt_result.json")
    second = continuation._predispatch_failure_result(
        attempt_dir,
        plan=plan,
        claim_path=claim_path,
    )

    assert second == first
    assert continuation.hash_file(attempt_dir / "attempt_result.json") == first_hash
    assert first["retryable_before_provider_dispatch"] is True
    assert first["continuation_validation"]["checks"]["no_dispatch_evidence"] is True
    assert (
        first["continuation_validation"]["current_checks"][
            "accepted_prefix_evidence_matches_plan"
        ]
        is True
    )
    assert (
        attempt_dir
        / "continuation"
        / "predispatch_retry_authorization.json"
    ).is_file()

    result_path = attempt_dir / "attempt_result.json"
    authorization_path = (
        attempt_dir / "continuation" / "predispatch_retry_authorization.json"
    )
    result_path.unlink()
    authorization_path.unlink()
    recovered = continuation._predispatch_failure_result(
        attempt_dir,
        plan=plan,
        claim_path=claim_path,
    )
    assert recovered["retryable_before_provider_dispatch"] is True
    assert result_path.is_file()
    assert authorization_path.is_file()

    historical = json.loads(result_path.read_text(encoding="utf-8"))
    historical["continuation_validation"][
        "predispatch_failure_attestation_sha256"
    ] = "historical-attestation-hash"
    result_path.write_text(json.dumps(historical), encoding="utf-8")
    authorization_path.unlink()
    reconciled = continuation._predispatch_failure_result(
        attempt_dir,
        plan=plan,
        claim_path=claim_path,
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert authorization["stored_attestation_matches_current"] is False
    assert (
        reconciled["continuation_validation"][
            "current_predispatch_failure_attestation_sha256"
        ]
        == continuation.hash_file(
            attempt_dir
            / "continuation"
            / "predispatch_failure_attestation.json"
        )
    )


def test_recoverable_provider_failure_accepts_audited_http_error_body():
    body = '{"error":{"code":"system_cpu_overloaded"}}'
    row = {
        "status": "failed",
        "failure_class": "provider_transport_error",
        "error": "RuntimeError: HTTP 503",
        "provider_metadata": {},
        "http_status": 503,
        "provider_response_encoding": "utf-8",
        "provider_response_body": body,
        "provider_response_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }

    assert continuation._recoverable_provider_failure_evidence_valid(row)
    assert not continuation._recoverable_provider_failure_evidence_valid(
        {**row, "provider_response_sha256": "wrong"}
    )
    assert not continuation._recoverable_provider_failure_evidence_valid(
        {**row, "http_status": 200}
    )


def test_recoverable_provider_failure_accepts_only_exact_arrearage_configuration_error():
    body = '{"error":{"type":"Arrearage","code":"Arrearage"}}'
    row = {
        "status": "failed",
        "failure_class": "provider_configuration_error",
        "error": "ProviderConfigurationError: HTTP 400 Arrearage",
        "provider_metadata": {},
        "http_status": 400,
        "provider_response_encoding": "utf-8",
        "provider_response_body": body,
        "provider_response_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }

    assert continuation._recoverable_provider_failure_evidence_valid(row)
    assert not continuation._recoverable_provider_failure_evidence_valid(
        {**row, "http_status": 403}
    )
    denied_body = '{"error":{"type":"permission_denied","code":"permission_denied"}}'
    assert not continuation._recoverable_provider_failure_evidence_valid(
        {
            **row,
            "provider_response_body": denied_body,
            "provider_response_sha256": hashlib.sha256(
                denied_body.encode("utf-8")
            ).hexdigest(),
        }
    )


def test_authoritative_decision_state_allows_only_exact_get_map_tail():
    decision = {
        "step": 4,
        "type": "state",
        "data": {"type": "decision", "decision": "map_select"},
    }
    query_tail = [
        {"step": 4, "type": "action", "data": {"cmd": "get_map"}},
        {"step": 5, "type": "state", "data": {"type": "map", "rows": []}},
    ]

    assert continuation._authoritative_decision_state(
        [decision, *query_tail]
    ) == decision["data"]
    with pytest.raises(
        continuation.ContinuationError,
        match="state-changing tail",
    ):
        continuation._authoritative_decision_state(
            [
                decision,
                {"step": 4, "type": "action", "data": {"cmd": "action"}},
                {"step": 5, "type": "state", "data": {"type": "map"}},
            ]
        )


def test_build_plan_rejects_protocol_error_at_transport_boundary(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    source_attempt = (
        tmp_path
        / "results"
        / "condition"
        / "live"
        / "runs"
        / "seed"
        / "replicate_01"
        / "attempts"
        / "attempt_001"
    )
    artifacts = source_attempt / "artifacts"
    artifacts.mkdir(parents=True)
    live_root = source_attempt.parents[4]
    (live_root / "source_lock.json").write_text("{}\n", encoding="utf-8")
    source_lock = {
        "identity_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
    }
    monkeypatch.setattr(
        continuation,
        "_read_and_verify_source_lock",
        lambda _root: dict(source_lock),
    )
    monkeypatch.setattr(
        continuation,
        "_audit_terminal_decision_requests",
        lambda *_args, **_kwargs: {"valid": True},
    )
    config = {
        "system_prompt": "rules",
        "history_chars": 1000,
        "model": "test-model",
        "base_url": "https://provider.invalid/v1",
        "temperature": 0.0,
        "timeout": 1,
        "max_tokens": None,
        "request_retries": 0,
        "max_decisions": 10,
        "seed": "seed",
        "character": "Ironclad",
        "ascension": 0,
        "lang": "eng",
    }
    terminal = "boundary state\n  [log] Game log saved to ignored"
    boundary_messages = [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been "
                "removed by FIFO truncation.\n\nboundary state\n\n"
                "Return only the next terminal input line."
            ),
        },
    ]
    boundary_attempt = {
        "type": "provider_request_attempt",
        "decision_index": 1,
        "provider_attempt_index": 0,
        "model_messages_sha256": continuation._messages_sha256(
            boundary_messages
        ),
        "provider_request_sha256": continuation._provider_request_sha256(
            boundary_messages,
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        ),
        "status": "failed",
        "failure_class": "provider_transport_error",
        "error": "TimeoutError: timed out",
        "provider_metadata": {},
    }
    events = [
        {
            "type": "terminal_decision",
            "decision_index": 0,
            "model_messages_sha256": "accepted",
            "model_raw": "e",
            "command": "e",
        },
        boundary_attempt,
        {
            "type": "model_protocol_error",
            "decision_index": 1,
            "error": "response was parsed and rejected",
        },
    ]
    engine_rows = [
        {
            "step": 0,
            "type": "state",
            "data": {"type": "decision", "decision": "combat_play"},
        },
        {"step": 0, "type": "action", "data": {"cmd": "quit"}},
        {"step": 1, "type": "state", "data": {"type": "quit"}},
    ]
    (source_attempt / "attempt_manifest.json").write_text(
        json.dumps(
            {
                "source_lock_sha256": "lock",
                "source_tree_identity_sha256": "tree",
                "protocol_sha256": "protocol",
                "execution_sts2_cli_root": str(tmp_path / "cli"),
            }
        ),
        encoding="utf-8",
    )
    (source_attempt / "attempt_result.json").write_text(
        json.dumps(
            {
                "termination_class": "provider_transport_error",
                "identity_stable": True,
                "source_tree_stable": True,
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (artifacts / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events),
        encoding="utf-8",
    )
    (artifacts / "terminal.txt").write_text(terminal, encoding="utf-8")
    (artifacts / "engine_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in engine_rows),
        encoding="utf-8",
    )

    with pytest.raises(
        continuation.ContinuationError,
        match="recoverable provider failure",
    ):
        continuation.build_replay_plan(source_attempt)


def _write_safe_predispatch_failure(attempt_dir, *, plan, adapter_sha256):
    continuation_dir = attempt_dir / "continuation"
    continuation_dir.mkdir(parents=True)
    attestation_path = continuation_dir / "predispatch_failure_attestation.json"
    attestation_path.write_text(
        json.dumps(
            {
                "status": "failed_before_provider_dispatch",
                "provider_calls": 0,
                "credential_file_read": False,
                "dispatch_count": 0,
                "checks": {"safe": True},
            }
        ),
        encoding="utf-8",
    )
    attestation_sha256 = continuation.hash_file(attestation_path)
    result_path = attempt_dir / "attempt_result.json"
    result_path.write_text(
        json.dumps(
            {
                "retryable_before_provider_dispatch": True,
                "termination_class": "process_infrastructure_error",
                "continuation_validation": {
                    "predispatch_failure_attestation_sha256": attestation_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    authorization_path = continuation_dir / "predispatch_retry_authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "status": "predispatch_retry_authorized",
                "source_attempt": plan["source_attempt"],
                "boundary_decision_index": plan["boundary"]["decision_index"],
                "provider_calls": 0,
                "credential_file_read": False,
                "dispatch_count": 0,
                "plan_sha256": continuation.hash_json(plan),
                "adapter_sha256": adapter_sha256,
                "attempt_result_sha256": continuation.hash_file(result_path),
                "current_attestation_sha256": attestation_sha256,
                "stored_attestation_sha256": attestation_sha256,
                "checks": {"safe": True},
            }
        ),
        encoding="utf-8",
    )
    return {
        "attempt": int(attempt_dir.name.removeprefix("attempt_")),
        "attempt_dir": str(attempt_dir.resolve()),
        "attestation_sha256": attestation_sha256,
        "current_attestation_sha256": attestation_sha256,
        "retry_authorization_sha256": continuation.hash_file(authorization_path),
    }


def test_predispatch_failure_list_cannot_omit_canonical_sibling(
    tmp_path,
    monkeypatch,
):
    attempts = tmp_path / "attempts"
    (attempts / "attempt_001").mkdir(parents=True)
    current = attempts / "attempt_004"
    current.mkdir()
    plan = {
        "source_attempt": str(attempts / "attempt_001"),
        "boundary": {"decision_index": 3},
    }
    adapter_sha256 = "adapter"
    listed = _write_safe_predispatch_failure(
        attempts / "attempt_002",
        plan=plan,
        adapter_sha256=adapter_sha256,
    )
    _write_safe_predispatch_failure(
        attempts / "attempt_003",
        plan=plan,
        adapter_sha256=adapter_sha256,
    )

    monkeypatch.setattr(
        continuation_validator,
        "_verify_predispatch_failure",
        lambda **kwargs: (
            kwargs["row"].get("attempt_dir") == str(kwargs["failure_dir"]),
            {"attempt_dir": str(kwargs["failure_dir"])},
        ),
    )
    valid, _chain = continuation_validator._predispatch_failure_chain(
            attempt_dir=current,
            claim={"predispatch_failures": [listed]},
            plan=plan,
            adapter_sha256=adapter_sha256,
    )
    assert valid is False


def test_predispatch_failure_chain_is_anchored_in_attempt_order(
    tmp_path,
    monkeypatch,
):
    attempts = tmp_path / "attempts"
    (attempts / "attempt_001").mkdir(parents=True)
    current = attempts / "attempt_004"
    current.mkdir()
    plan = {
        "source_attempt": str(attempts / "attempt_001"),
        "boundary": {"decision_index": 3},
    }
    adapter_sha256 = "adapter"
    second = _write_safe_predispatch_failure(
        attempts / "attempt_002",
        plan=plan,
        adapter_sha256=adapter_sha256,
    )
    third = _write_safe_predispatch_failure(
        attempts / "attempt_003",
        plan=plan,
        adapter_sha256=adapter_sha256,
    )

    monkeypatch.setattr(
        continuation_validator,
        "_verify_predispatch_failure",
        lambda **kwargs: (
            kwargs["row"].get("attempt_dir") == str(kwargs["failure_dir"]),
            {"attempt_dir": str(kwargs["failure_dir"])},
        ),
    )
    valid, _chain = continuation_validator._predispatch_failure_chain(
            attempt_dir=current,
            claim={"predispatch_failures": [third, second]},
            plan=plan,
            adapter_sha256=adapter_sha256,
    )
    assert valid is False
    valid, chain = continuation_validator._predispatch_failure_chain(
        attempt_dir=current,
        claim={"predispatch_failures": [second, third]},
        plan=plan,
        adapter_sha256=adapter_sha256,
    )
    assert valid is True
    assert [item["attempt_dir"] for item in chain] == [
        str((attempts / "attempt_002").resolve()),
        str((attempts / "attempt_003").resolve()),
    ]


def test_run_live_setup_exception_clears_owned_lease(tmp_path, monkeypatch):
    job_dir = tmp_path / "live" / "runs" / "seed" / "replicate_01"
    source_attempt = job_dir / "attempts" / "attempt_001"
    source_artifacts = source_attempt / "artifacts"
    source_artifacts.mkdir(parents=True)
    source_manifest = {
        "experiment_id": "integration",
        "seed": "seed",
        "replicate": 1,
        "attempt": 1,
        "source_lock_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
        "command": [
            "python",
            "-m",
            "runner",
            "--sts2-cli-root",
            str(tmp_path / "source-cli"),
            "--out-dir",
            str(source_artifacts),
        ],
    }
    (source_attempt / "attempt_manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )
    (source_attempt / "attempt_result.json").write_text("{}\n", encoding="utf-8")
    for name in ("config.json", "events.jsonl", "terminal.txt", "engine_trace.jsonl"):
        (source_artifacts / name).write_text("{}\n", encoding="utf-8")
    source_paths = {
        "config.json": source_artifacts / "config.json",
        "events.jsonl": source_artifacts / "events.jsonl",
        "terminal.txt": source_artifacts / "terminal.txt",
        "engine_trace.jsonl": source_artifacts / "engine_trace.jsonl",
        "attempt_manifest.json": source_attempt / "attempt_manifest.json",
        "attempt_result.json": source_attempt / "attempt_result.json",
    }
    plan = {
        "adapter_sha256": continuation.hash_file(Path(continuation.__file__)),
        "source_attempt": str(source_attempt),
        "source_artifacts": str(source_artifacts),
        "source_live_root": str(tmp_path / "live"),
        "source_hashes": {
            name: continuation.hash_file(path) for name, path in source_paths.items()
        },
        "source_lock_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
        "execution_sts2_cli_root": str(tmp_path / "cli"),
        "accepted_decision_count": 1,
        "config": {},
        "decisions": [],
        "boundary": {
            "decision_index": 1,
            "provider_request_sha256": "request",
            "failed_provider_attempt_count": 1,
        },
    }
    verification_root = tmp_path / "verification"
    verification_root.mkdir()
    (verification_root / "continuation_verification.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (tmp_path / "protocol.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "cli").mkdir()
    source_lock = {
        "identity_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
        "protocol_path": str(tmp_path / "protocol.json"),
        "benchmark_root": str(tmp_path),
        "sts2_cli_root": str(tmp_path / "cli"),
    }
    monkeypatch.setattr(
        continuation,
        "_load_verified_plan",
        lambda _root: (copy.deepcopy(plan), {"plan_sha256": "plan"}),
    )
    monkeypatch.setattr(
        continuation,
        "_read_and_verify_source_lock",
        lambda _root: dict(source_lock),
    )
    original_atomic_create = continuation._atomic_create
    lease_path = job_dir / "active_attempt.json"

    def fail_after_lease(path, value):
        original_atomic_create(path, value)
        if path == lease_path:
            raise RuntimeError("setup exploded")

    monkeypatch.setattr(continuation, "_atomic_create", fail_after_lease)

    with pytest.raises(RuntimeError, match="setup exploded"):
        continuation.run_live(verification_root, tmp_path / "provider.env")

    assert not lease_path.exists()


def test_run_live_creates_canonical_attempt_dispatches_once_and_validates(
    tmp_path,
    monkeypatch,
):
    job_dir = tmp_path / "live" / "runs" / "seed" / "replicate_01"
    source_attempt = job_dir / "attempts" / "attempt_001"
    source_artifacts = source_attempt / "artifacts"
    source_artifacts.mkdir(parents=True)
    replay_messages = [{"role": "user", "content": "decision 0"}]
    boundary_messages = [{"role": "user", "content": "decision 1"}]
    engine_trace = source_artifacts / "engine_trace.jsonl"
    engine_rows = _write_engine_prefix(engine_trace)
    config = {
        "base_url": "https://provider.invalid/v1",
        "model": "test-model",
        "temperature": 0.0,
        "max_tokens": None,
        "request_retries": 2,
        "seed": "seed",
        "character": "Ironclad",
        "ascension": 0,
        "lang": "eng",
        "history_chars": 10000,
        "max_decisions": 500,
    }
    source_manifest = {
        "schema_version": 1,
        "experiment_id": "integration",
        "seed": "seed",
        "replicate": 1,
        "attempt": 1,
        "source_lock_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
        "command": [
            "python",
            "-m",
            "runner",
            "--sts2-cli-root",
            str(tmp_path / "source-cli"),
            "--out-dir",
            str(source_artifacts),
        ],
    }
    (source_attempt / "attempt_manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )
    (source_attempt / "attempt_result.json").write_text(
        json.dumps({"status": "failed", "termination_class": "provider_transport_error"}),
        encoding="utf-8",
    )
    for name in ("config.json", "events.jsonl", "terminal.txt"):
        (source_artifacts / name).write_text("{}\n", encoding="utf-8")
    source_paths = {
        "config.json": source_artifacts / "config.json",
        "events.jsonl": source_artifacts / "events.jsonl",
        "terminal.txt": source_artifacts / "terminal.txt",
        "engine_trace.jsonl": engine_trace,
        "attempt_manifest.json": source_attempt / "attempt_manifest.json",
        "attempt_result.json": source_attempt / "attempt_result.json",
    }
    plan = {
        "schema_version": 2,
        "adapter_sha256": continuation.hash_file(Path(continuation.__file__)),
        "source_attempt": str(source_attempt),
        "source_artifacts": str(source_artifacts),
        "source_live_root": str(tmp_path / "live"),
        "source_hashes": {
            name: continuation.hash_file(path) for name, path in source_paths.items()
        },
        "source_lock_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
        "execution_sts2_cli_root": str(tmp_path / "cli"),
        "accepted_decision_count": 1,
        "config": config,
        "decisions": [
            {
                "decision_index": 0,
                "model_messages_sha256": continuation._messages_sha256(
                    replay_messages
                ),
                "model_raw": "e",
                "command": "e",
            }
        ],
        "boundary": {
            "decision_index": 1,
            "model_messages_sha256": continuation._messages_sha256(
                boundary_messages
            ),
            "provider_request_sha256": continuation._provider_request_sha256(
                boundary_messages,
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            ),
            "failed_provider_attempt_count": 2,
            "engine_prefix_row_count": len(engine_rows),
            "engine_prefix_sha256": continuation.hash_json(
                continuation._canonical_engine_prefix(engine_rows)
            ),
            "state_sha256": continuation.hash_json(engine_rows[-1]["data"]),
        },
    }
    verification_root = tmp_path / "verification"
    verification_root.mkdir()
    (verification_root / "continuation_verification.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (tmp_path / "protocol.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "cli").mkdir()
    env_file = tmp_path / "provider.env"
    env_file.write_text("STS2_BENCH_API_KEY=not-a-real-key\n", encoding="utf-8")
    source_lock = {
        "identity_sha256": "lock",
        "source_tree_identity_sha256": "tree",
        "protocol_sha256": "protocol",
        "protocol_path": str(tmp_path / "protocol.json"),
        "benchmark_root": str(tmp_path),
        "sts2_cli_root": str(tmp_path / "cli"),
    }
    provider_messages = []
    validator_calls = []

    monkeypatch.setattr(
        continuation,
        "_load_verified_plan",
        lambda _root: (copy.deepcopy(plan), {"plan_sha256": "plan"}),
    )
    monkeypatch.setattr(
        continuation,
        "_read_and_verify_source_lock",
        lambda _root: dict(source_lock),
    )
    monkeypatch.setattr(continuation, "load_human_cli", lambda _root: object())
    monkeypatch.setattr(
        continuation,
        "_lazy_api_key",
        lambda _path: lambda: "not-a-real-key",
    )

    def fake_provider_complete(_model, messages):
        provider_messages.append(copy.deepcopy(messages))
        return ModelReply(
            text="e",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            latency_sec=0.01,
            provider_attempt_count=1,
            provider_metadata={"finish_reason": "stop"},
        )

    monkeypatch.setattr(
        continuation.OpenAICompatTerminalModel,
        "complete",
        fake_provider_complete,
    )

    def fake_run_direct_cli(*, model, out_dir, **_kwargs):
        out_dir.mkdir(parents=True)
        _write_engine_prefix(out_dir / "engine_trace.jsonl")
        assert model.complete(replay_messages).text == "e"
        assert model.complete(boundary_messages).text == "e"
        for name in ("config.json", "events.jsonl", "terminal.txt"):
            (out_dir / name).write_text("{}\n", encoding="utf-8")
        return {"formal_behavior_sample": True, "decision": "game_over"}

    monkeypatch.setattr(continuation, "run_direct_cli", fake_run_direct_cli)

    def fake_validate(attempt_dir, **_kwargs):
        validator_calls.append(attempt_dir)
        assert (attempt_dir / "attempt_manifest.json").is_file()
        assert (attempt_dir / "continuation" / "seam_proof.json").is_file()
        assert (
            attempt_dir
            / "continuation"
            / "dispatches"
            / "decision_000001.json"
        ).is_file()
        return {
            "valid": True,
            "termination_class": "natural_game_over",
            "checks": {
                "config_game_rules_match_protocol": True,
                "terminal_decision_rule_exposure_complete": True,
            },
            "summary": {
                "error": None,
                "truncated": False,
                "decision": "game_over",
            },
        }

    monkeypatch.setattr(
        continuation,
        "_validate_attempt_artifacts",
        fake_validate,
    )

    result = continuation.run_live(verification_root, env_file)

    attempt_dir = job_dir / "attempts" / "attempt_002"
    assert result["status"] == "succeeded"
    assert result["formal_sample"] is True
    assert result["continuation_validation"]["valid"] is True
    assert result["continuation_proof"]["provider_calls"] == 1
    assert result["continuation_proof"]["credential_file_read"] is True
    assert provider_messages == [boundary_messages]
    assert validator_calls == [attempt_dir]
    assert not (job_dir / "active_attempt.json").exists()
    job_status = json.loads(
        (job_dir / "job_status.json").read_text(encoding="utf-8")
    )
    assert len(job_status["attempt_history"]) == 2
    claim_files = list((job_dir / "continuation_claims").glob("*.json"))
    assert len(claim_files) == 1
    claim = json.loads(claim_files[0].read_text(encoding="utf-8"))
    assert claim["status"] == "completed"
    assert claim["provider_dispatch_started"] is True
    assert claim["provider_calls"] == 1
    manifest = json.loads(
        (attempt_dir / "attempt_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["command"] == manifest["source_protocol_command"]
    command_root_index = manifest["command"].index("--sts2-cli-root") + 1
    assert manifest["command"][command_root_index] == str((tmp_path / "cli").resolve())
    assert manifest["wrapper_command"][1] == str(Path(continuation.__file__).resolve())
    wrapper_root_index = manifest["wrapper_command"].index("--sts2-cli-root") + 1
    assert manifest["wrapper_command"][wrapper_root_index] == str(
        (tmp_path / "cli").resolve()
    )

    (attempt_dir / "continuation" / "seam_proof.json").unlink()
    tampered = continuation._validate_continuation_evidence(
        attempt_dir,
        plan=plan,
        wrapper_command=manifest["wrapper_command"],
        execution_identity_sha256=result["continuation_validation"][
            "execution_identity_sha256"
        ],
    )
    assert tampered["valid"] is False
    assert tampered["checks"]["seam_matches_verified_boundary"] is False
    assert tampered["checks"]["attestation_matches_execution"] is False
