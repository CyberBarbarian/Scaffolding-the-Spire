from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parents[2] / ".trajectory_recovery" / "run_default_trace_continuation.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("trajectory_recovery_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeOccAgent:
    def __init__(self) -> None:
        self.reward_posterior_evaluator = object()


class _FakeCoordinator:
    def __init__(self) -> None:
        self.occ_agent = _FakeOccAgent()
        self.build_calls = 0

    def build_prompt(self, state, legal_actions):
        self.build_calls += 1
        posterior = self.occ_agent.reward_posterior_evaluator
        if type(posterior).__name__ == "_RecordedRewardPosteriorEvaluator":
            posterior.call_count = 1
            posterior.candidate_identity_verified = True
        return f"decision={state['decision']} posterior={type(posterior).__name__}"

    def reset_episode(self):
        return None


def _agent(module):
    state = {"decision": "card_reward", "cards": [{"name": "Wisp"}]}
    inner = _FakeCoordinator()
    agent = module.ReplayThenLiveAgent(
        inner,
        [],
        {
            "step": 7,
            "prompt": "old",
            "authoritative_state": state,
            "_recorded_target_action": {
                "agent_meta": {
                    "reward_posterior_evaluator": "llm",
                    "reward_posterior_evaluation": {"mode": "recorded"},
                }
            },
        },
        stop_at_recovery=True,
        rebuild_target_prompt=True,
    )
    return agent, inner, state


def test_card_reward_rebuild_reuses_recorded_posterior_and_restores_evaluator():
    module = _load_launcher()
    agent, inner, state = _agent(module)
    evaluator = inner.occ_agent.reward_posterior_evaluator

    prompt = agent.build_prompt(state, [])

    assert prompt == "decision=card_reward posterior=_RecordedRewardPosteriorEvaluator"
    assert inner.occ_agent.reward_posterior_evaluator is evaluator
    assert agent.target_prompt_proof_scope == "full_rebuilt_prompt_recorded_posterior"
    assert agent.recorded_reward_posterior_reused is True
    assert agent.recorded_reward_posterior_replay_count == 1
    with pytest.raises(RuntimeError, match=module.RECOVERY_SENTINEL):
        agent.choose(state, [], prompt=prompt)


def test_non_reward_rebuild_uses_normal_full_prompt():
    module = _load_launcher()
    state = {"decision": "map_select"}
    inner = _FakeCoordinator()
    agent = module.ReplayThenLiveAgent(
        inner,
        [],
        {"step": 8, "prompt": "old", "authoritative_state": state},
        stop_at_recovery=True,
        rebuild_target_prompt=True,
    )

    prompt = agent.build_prompt(state, [])

    assert prompt == "decision=map_select posterior=object"
    assert agent.target_prompt_proof_scope == "full_rebuilt_prompt"
    assert agent.recorded_reward_posterior_reused is False


def _recorded_posterior_meta():
    messages = [
        {"role": "system", "content": "analyze"},
        {"role": "user", "content": "reward"},
    ]
    body = json.dumps(
        {"model": "qwen/qwen3.6-flash", "choices": [{"finish_reason": "stop"}]},
        separators=(",", ":"),
    )
    return {
        "reward_posterior_attempt": 1,
        "reward_posterior_model": "qwen3.6-flash",
        "reward_posterior_model_request_audits": [
            {"request_index": 0, "provider_response_status": "transport_error"},
            {
                "request_index": 1,
                "messages": messages,
                "messages_sha256": hashlib.sha256(
                    json.dumps(
                        messages,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "provider_response_body": body,
                "provider_response_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "provider_response_status": "succeeded",
                "provider_http_status": 200,
                "provider_finish_reason": "stop",
            },
        ],
    }


def test_recorded_posterior_selects_accepted_retry_and_preserves_binding():
    module = _load_launcher()
    meta = _recorded_posterior_meta()

    accepted = module._accepted_recorded_reward_posterior_request_audit(meta)

    assert accepted["request_index"] == 1
    assert accepted["messages_sha256"]
    assert accepted["provider_response_sha256"]
    assert accepted["provider_http_status"] == 200


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("messages_sha256", "0" * 64, "messages hash mismatch"),
        ("provider_response_sha256", "0" * 64, "response hash mismatch"),
        ("provider_http_status", 500, "provider result is incomplete"),
        ("provider_response_status", "transport_error", "provider result is incomplete"),
    ],
)
def test_recorded_posterior_provider_binding_fails_closed(field, value, message):
    module = _load_launcher()
    meta = _recorded_posterior_meta()
    meta["reward_posterior_model_request_audits"][1][field] = value

    with pytest.raises(RuntimeError, match=message):
        module._accepted_recorded_reward_posterior_request_audit(meta)


def test_recorded_posterior_rejects_model_or_attempt_drift():
    module = _load_launcher()
    meta = _recorded_posterior_meta()
    meta["reward_posterior_model"] = "another-model"
    with pytest.raises(RuntimeError, match="response model mismatch"):
        module._accepted_recorded_reward_posterior_request_audit(meta)

    meta = _recorded_posterior_meta()
    meta["reward_posterior_attempt"] = 2
    with pytest.raises(RuntimeError, match="accepted request audit is not unique"):
        module._accepted_recorded_reward_posterior_request_audit(meta)
