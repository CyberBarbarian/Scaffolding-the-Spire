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
        self.messages.append(messages)
        payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return {
            "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }


def _analysis_for(legal_actions):
    return {
        "mode": "reward_posterior_deck_analysis",
        "baseline_analysis": "The unchanged deck is the skip-card-reward baseline.",
        "candidate_deck_analyses": [
            {
                "action_id": action.action_id,
                "label": action.label,
                "deck_delta": "unchanged" if action.kind == "card_reward_skip" else "adds one visible card",
                "immediate_current_deck_effect": "compares only against the current deck",
                "redundancy_or_dilution_risks": "noted without selecting an action",
                "visible_support": "visible deck facts only",
                "missing_or_uncertain_support": "future support is uncertain",
                "comparison_to_unchanged_deck": "candidate deck state is compared to skip baseline",
                "uncertainty": "low",
            }
            for action in legal_actions
            if action.command.get("cmd") != "bench_view"
        ],
        "global_uncertainties": [],
    }


def test_llm_reward_posterior_evaluator_analyzes_every_candidate_without_choosing():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    client = _ScriptChatClient([_analysis_for(legal_actions)])
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=0)

    evaluation, meta = evaluator.evaluate(state, legal_actions, bundle)

    data = evaluation.to_dict()
    expected_ids = bundle.candidate_action_ids()
    actual_ids = {item["action_id"] for item in data["candidate_deck_analyses"]}
    prompt = client.messages[0][-1]["content"]
    assert actual_ids == expected_ids
    assert data["mode"] == "reward_posterior_deck_analysis"
    assert "Do not choose a game action" in prompt
    assert "posterior_candidate_audit_no_policy" in prompt
    assert "score" not in json.dumps(data, ensure_ascii=False).lower()
    assert "recommendation" not in json.dumps(data, ensure_ascii=False).lower()
    assert meta["reward_posterior_evaluator"] == "llm"
    assert meta["reward_posterior_evaluation_hash"]


def test_llm_reward_posterior_evaluator_rejects_policy_fields():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    bad = _analysis_for(legal_actions)
    bad["candidate_deck_analyses"][0]["recommendation"] = "pick this card"
    client = _ScriptChatClient([bad])
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=0)

    with pytest.raises(ModelActionError, match="Policy field is not allowed"):
        evaluator.evaluate(state, legal_actions, bundle)


def test_llm_reward_posterior_evaluator_requires_exact_candidate_coverage():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    bundle = build_evidence_bundle(state, legal_actions)
    incomplete = _analysis_for(legal_actions)
    incomplete["candidate_deck_analyses"] = incomplete["candidate_deck_analyses"][:-1]
    client = _ScriptChatClient([incomplete])
    evaluator = LLMRewardPosteriorEvaluator(client, max_retries=0)

    with pytest.raises(ModelActionError, match="must cover every candidate_evidence action_id"):
        evaluator.evaluate(state, legal_actions, bundle)
