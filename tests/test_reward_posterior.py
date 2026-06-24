import json

import pytest

from sts2_bench.actions import build_legal_actions
from sts2_bench.agents import ModelActionError
from sts2_bench.evidence import build_evidence_bundle
from sts2_bench.posterior import LLMRewardPosteriorEvaluator


def _card_reward_state():
    return {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
        "player": {
            "hp": 46,
            "max_hp": 80,
            "deck_size": 6,
            "deck": [
                {"id": "CARD.STRIKE", "name": "Strike", "type": "Attack", "cost": 1, "description": "Deal 6 damage."},
                {"id": "CARD.STRIKE", "name": "Strike", "type": "Attack", "cost": 1, "description": "Deal 6 damage."},
                {"id": "CARD.DEFEND", "name": "Defend", "type": "Skill", "cost": 1, "description": "Gain 5 Block."},
                {"id": "CARD.DEFEND", "name": "Defend", "type": "Skill", "cost": 1, "description": "Gain 5 Block."},
                {"id": "CARD.BASH", "name": "Bash", "type": "Attack", "cost": 2, "description": "Deal damage. Apply Vulnerable."},
                {"id": "CARD.POMMEL", "name": "Pommel Strike", "type": "Attack", "cost": 1, "description": "Deal damage. Draw 1 card."},
            ],
        },
        "cards": [
            {"index": 0, "name": "Twin Strike", "type": "Attack", "cost": 1},
            {"index": 1, "name": "Shrug It Off", "type": "Skill", "cost": 1},
            {"index": 2, "name": "Havoc", "type": "Skill", "cost": 1},
        ],
        "can_skip": True,
        "last_action_result": {"changes": ["previous fight lost 18 HP over 5 turns"]},
    }


class _ScriptChatClient:
    model = "script-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def _chat(self, messages):
        self.messages.append([dict(message) for message in messages])
        payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return {
            "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }


def _slot_fill_for(legal_actions):
    return {
        "mode": "reward_posterior_slot_fill",
        "candidate_analysis_updates": [
            {
                "action_id": action.action_id,
                "deck_delta": "adds one visible card",
                "immediate_current_deck_effect": "compares only against the current deck",
                "redundancy_or_dilution_risks": "noted without selecting an action",
                "visible_support": "visible deck facts only",
                "missing_or_uncertain_support": "future support is uncertain",
                "comparison_to_unchanged_deck": "candidate deck state is compared to skip baseline",
                "uncertainty": "low",
            }
            for action in legal_actions
            if action.kind != "card_reward_skip" and action.command.get("cmd") != "bench_view"
        ],
        "global_uncertainties": [],
    }


def _action_of_kind(legal_actions, kind):
    return next(action for action in legal_actions if action.kind == kind)


def _without_candidate_update(analysis, action):
    analysis["candidate_analysis_updates"] = [
        item for item in analysis["candidate_analysis_updates"] if item["action_id"] != action.action_id
    ]
    return analysis


def test_llm_reward_posterior_evaluator_builds_candidate_structure_without_choosing():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    client = _ScriptChatClient([_slot_fill_for(legal_actions)])
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=0)

    evaluation, meta = evaluator.evaluate(state, legal_actions, bundle)

    data = evaluation.to_dict()
    expected_ids = bundle.candidate_action_ids()
    actual_ids = {item["action_id"] for item in data["candidate_deck_analyses"]}
    by_id = {candidate.action_id: candidate for candidate in bundle.candidate_evidence}
    prompt = client.messages[0][-1]["content"]
    assert actual_ids == expected_ids
    assert data["mode"] == "reward_posterior_deck_analysis"
    assert all(item["label"] == by_id[item["action_id"]].label for item in data["candidate_deck_analyses"])
    assert all(item["kind"] == by_id[item["action_id"]].kind for item in data["candidate_deck_analyses"])
    assert "Do not choose a game action" in prompt
    assert "posterior_candidate_audit_no_policy" in prompt
    assert "Do not output candidate_deck_analyses" in prompt
    assert "score" not in json.dumps(data, ensure_ascii=False).lower()
    assert "recommendation" not in json.dumps(data, ensure_ascii=False).lower()
    assert meta["reward_posterior_evaluator"] == "llm"
    assert meta["reward_posterior_code_owned_candidate_ids"] == sorted(expected_ids)
    assert meta["reward_posterior_evaluation_hash"]


def test_llm_reward_posterior_evaluator_generates_skip_candidate_without_llm_update():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    skip_action = _action_of_kind(legal_actions, "card_reward_skip")
    no_skip_update = _without_candidate_update(_slot_fill_for(legal_actions), skip_action)
    client = _ScriptChatClient([no_skip_update, _slot_fill_for(legal_actions)])
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=1)

    evaluation, meta = evaluator.evaluate(state, legal_actions, bundle)

    data = evaluation.to_dict()
    actual_ids = {item["action_id"] for item in data["candidate_deck_analyses"]}
    skip_items = [item for item in data["candidate_deck_analyses"] if item["action_id"] == skip_action.action_id]
    assert actual_ids == bundle.candidate_action_ids()
    assert len(skip_items) == 1
    assert skip_items[0]["label"] == skip_action.label
    assert skip_items[0]["kind"] == skip_action.kind
    assert "unchanged" in skip_items[0]["deck_delta"]
    assert meta["reward_posterior_attempt"] == 0
    assert meta["reward_posterior_code_owned_skip_candidate_ids"] == [skip_action.action_id]
    assert len(client.messages) == 1


def test_llm_reward_posterior_prompt_exposes_code_owned_skeleton_and_fill_targets():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    evaluator = LLMRewardPosteriorEvaluator(_ScriptChatClient([]), max_retries=0)

    prompt = evaluator.build_prompt(state, legal_actions, bundle)

    skip_action = next(action for action in legal_actions if action.kind == "card_reward_skip")
    assert "Code-owned final candidate_deck_analyses skeleton:" in prompt
    assert "Required non-skip candidate_analysis_updates targets:" in prompt
    assert f'"action_id":{skip_action.action_id}' in prompt
    assert '"kind":"card_reward_skip"' in prompt
    assert "Do not output candidate_deck_analyses" in prompt
    assert "Skip is generated by code as the unchanged-deck candidate" in prompt


def test_llm_reward_posterior_evaluator_rejects_policy_fields():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    bad = _slot_fill_for(legal_actions)
    bad["candidate_analysis_updates"][0]["recommendation"] = "pick this card"
    client = _ScriptChatClient([bad])
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=0)

    with pytest.raises(ModelActionError, match="Policy field is not allowed"):
        evaluator.evaluate(state, legal_actions, bundle)


def test_llm_reward_posterior_evaluator_retries_transport_without_prompt_correction(monkeypatch):
    monkeypatch.setattr("sts2_bench.posterior.time.sleep", lambda _delay: None)
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    client = _ScriptChatClient(
        [
            RuntimeError('LLM HTTP 503: {"error":{"message":"auth_unavailable: no auth available"}}'),
            _slot_fill_for(legal_actions),
        ]
    )
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=1)

    _evaluation, meta = evaluator.evaluate(state, legal_actions, bundle)

    assert meta["reward_posterior_attempt"] == 1
    assert len(client.messages) == 2
    assert client.messages[1] == client.messages[0]
    retry_text = "\n".join(message["content"] for message in client.messages[1])
    assert "previous posterior deck analysis was invalid" not in retry_text


def test_llm_reward_posterior_evaluator_records_previous_retry_details():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    missing_action = _action_of_kind(legal_actions, "card_reward")
    incomplete = _without_candidate_update(_slot_fill_for(legal_actions), missing_action)
    client = _ScriptChatClient([incomplete, _slot_fill_for(legal_actions)])
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=1)

    _evaluation, meta = evaluator.evaluate(state, legal_actions, bundle)

    assert meta["reward_posterior_attempt"] == 1
    assert "candidate_analysis_updates must cover every non-skip candidate" in meta["reward_posterior_previous_errors"][0]
    assert "candidate_analysis_updates" in meta["reward_posterior_previous_raw_responses"][0]
    retry_text = "\n".join(message["content"] for message in client.messages[1])
    assert "previous posterior deck analysis was invalid" in retry_text


def test_llm_reward_posterior_evaluator_requires_exact_non_skip_update_coverage():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    missing_action = _action_of_kind(legal_actions, "card_reward")
    incomplete = _without_candidate_update(_slot_fill_for(legal_actions), missing_action)
    client = _ScriptChatClient([incomplete])
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=0)

    with pytest.raises(ModelActionError, match="candidate_analysis_updates must cover every non-skip candidate"):
        evaluator.evaluate(state, legal_actions, bundle)
