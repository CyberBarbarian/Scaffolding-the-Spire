import json
import re
from types import SimpleNamespace

import pytest

from sts2_bench.actions import LegalAction
from sts2_bench.agents import (
    OpenAICompatAgent,
    _combat_boundary_facts,
    _combat_memory_snapshot,
    _combat_mdp_action_fact,
    build_formal_generalist_system_message,
)
from sts2_bench.game_rules import render_full_game_rules
from sts2_bench.run_benchmark import build_run_config, validate_agent_configuration
from sts2_bench.runner import agent_from_args


def _combat_state(round_number: int = 1) -> dict:
    return {
        "decision": "combat_play",
        "round": round_number,
        "context": {"act": 1, "floor": 3, "room_type": "Monster"},
        "player": {
            "name": "Defect",
            "hp": 40,
            "max_hp": 70,
            "block": 0,
            "gold": 99,
            "deck_size": 10,
            "deck": [
                {
                    "id": "CARD.ZAP",
                    "name": "Zap",
                    "type": "Skill",
                    "cost": 1,
                    "description": "Channel 1 Lightning.",
                    "upgraded": False,
                    "after_upgrade": {"description": "Channel 2 Lightning."},
                }
            ],
            "relics": [{"id": "CRACKED_CORE", "name": "Cracked Core"}],
            "potions": [{"id": "FIRE_POTION", "name": "Fire Potion"}],
        },
        "combat": {
            "round": round_number,
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "CARD.ZAP",
                    "name": "Zap",
                    "type": "Skill",
                    "cost": 1,
                    "description": "Channel 1 Lightning.",
                    "upgraded": False,
                }
            ],
            "enemies": [
                {
                    "index": 0,
                    "instance_id": "enemy-1",
                    "name": "Jaw Worm",
                    "hp": 40,
                    "max_hp": 40,
                    "intents": [{"damage": 11, "hits": 1}],
                    "powers": [],
                }
            ],
            "draw_pile_count": 5,
            "discard_pile_count": 4,
            "exhaust_pile_count": 0,
        },
    }


def _end_turn_action() -> LegalAction:
    return LegalAction(
        action_id=0,
        kind="combat_end_turn",
        label="end turn",
        command={"cmd": "action", "action": "end_turn"},
    )


def _accepted_response(round_number: int) -> str:
    return json.dumps(
        {
            "turn_intent": f"round {round_number} intent",
            "action_id": 0,
            "selected_action_label": "end turn",
            "reason": f"round {round_number} accepted",
        },
        separators=(",", ":"),
    )


def _response(raw: str) -> dict:
    return {
        "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _run_args(**overrides):
    values = {
        "agent": "coordinator",
        "base_url": "http://127.0.0.1:1",
        "model": "test",
        "llm_timeout": 5.0,
        "llm_max_retries": 0,
        "llm_fallback_mode": "error",
        "llm_temperature": 0.0,
        "prompt_style": "analysis",
        "include_json_state": False,
        "short_term_memory_enabled": False,
        "short_term_memory_window": 8,
        "short_term_memory_mode": "factual_diff",
        "run_summary": False,
        "context_mode": "combat_mdp_state",
        "turn_chat_window": 10,
        "combat_dialogue_history_turns": 3,
        "turn_chat_update_mode": "delta",
        "turn_chat_assistant_history": "compact",
        "turn_chat_plan": False,
        "route_plan_enabled": True,
        "route_plan_enforcement": "hard",
        "route_map_prompt_format": "canonical",
        "route_candidate_objective": "none",
        "map_route_hint": "none",
        "occ_context_mode": "reference_turn",
        "reward_posterior_evaluator": True,
        "self_consistency_enabled": False,
        "self_consistency_samples": 3,
        "self_consistency_decisions": "map_select,card_reward",
        "self_consistency_mode": "diagnostic",
        "action_prompt_order": "default",
        "action_prompt_order_seed": 0,
        "allow_custom_sequences": False,
        "custom_sequence_exposure": "safe",
        "combat_card_sequences_enabled": False,
        "precomputed_combat_card_sequences_enabled": False,
        "combat_reasoning_pool_enabled": True,
        "combat_turn_board_enabled": True,
        "combat_skill_hints_enabled": False,
        "combat_skill_hints_profile": "baseline",
        "combat_skill_hints_max_hints": 4,
        "combat_skill_lines_enabled": False,
        "rule_projection_and_recent_changes_enabled": True,
        "character": "Defect",
        "ascension": 0,
        "lang": "en",
        "seeds": None,
        "count": 0,
        "max_steps": 10,
        "max_game_actions": 0,
        "include_full_map": True,
        "allow_repeat_views": False,
        "out": "",
        "log_level": "full",
        "print_prompts": False,
        "print_model_output": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_three_combat_conditions_share_identical_current_prompt_at_first_decision():
    state = _combat_state()
    legal_actions = [_end_turn_action()]
    common = {
        "base_url": "http://127.0.0.1:1",
        "model": "test",
        "prompt_style": "analysis",
        "run_summary_enabled": False,
        "combat_reasoning_pool_enabled": True,
        "combat_turn_board_enabled": True,
        "combat_skill_hints_enabled": False,
        "combat_card_sequences_enabled": False,
    }
    managed = OpenAICompatAgent(conversation_mode="combat_mdp_state", **common)
    state_only = OpenAICompatAgent(conversation_mode="combat_state_only", **common)
    dialogue = OpenAICompatAgent(conversation_mode="combat_dialogue", **common)

    managed_prompt = managed.build_prompt(state, legal_actions)
    state_only_prompt = state_only.build_prompt(state, legal_actions)
    dialogue_preview = dialogue.build_prompt(state, legal_actions)
    dialogue_current = dialogue_preview.split("===== message[1] role=user =====\n", 1)[1]

    assert managed_prompt == state_only_prompt == dialogue_current
    assert "Combat turn process JSON:" not in state_only_prompt
    assert "last_action_result" not in state_only_prompt


def test_resolved_config_records_raw_dialogue_condition_without_changing_architecture():
    args = _run_args(context_mode="combat_dialogue")

    config = build_run_config(args, [], [])

    assert config["agent"]["architecture"] == "role_specialized"
    assert config["combat_context"] == {
        "variant": "combat_dialogue",
        "dialogue_history_turns": 3,
        "history_representation": "raw_dialogue",
    }
    assert config["agent"]["context_management"]["ca"]["history"] == "raw_dialogue"
    assert config["agent"]["context_management"]["ca"]["dialogue_history_turns"] == 3
    assert config["game_rules"]["delivery"] == "state_selected_prompt"


def test_resolved_config_and_preflight_fail_close_formal_generalist_contract():
    args = _run_args(
        agent="formal_generalist",
        context_mode="combat_state_only",
        route_plan_enabled=False,
        reward_posterior_evaluator=False,
        combat_reasoning_pool_enabled=False,
        combat_turn_board_enabled=False,
        rule_projection_and_recent_changes_enabled=False,
    )

    validate_agent_configuration(args)
    config = build_run_config(args, [], [])

    assert config["agent"]["architecture"] == "formal_generalist"
    assert config["game_rules"]["delivery"] == "static_full_system_prompt"
    assert config["combat_context"]["history_representation"] == "none"
    assert config["benchmark"]["allow_custom_sequences"] is False
    assert config["benchmark"]["combat_card_sequences"] is False

    args.llm_fallback_mode = "action"
    with pytest.raises(ValueError, match="fallback_mode"):
        validate_agent_configuration(args)


def test_combat_state_only_discards_turn_intent_and_history(monkeypatch):
    state = _combat_state()
    action = _end_turn_action()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_state_only",
        prompt_style="analysis",
        run_summary_enabled=False,
        combat_reasoning_pool_enabled=True,
        combat_turn_board_enabled=True,
        combat_card_sequences_enabled=False,
    )
    monkeypatch.setattr(agent, "_chat", lambda _messages: _response(_accepted_response(1)))

    chosen, meta = agent.choose(state, [action])
    agent.record_transition(state, chosen, meta, _combat_state(2))
    next_prompt = agent.build_prompt(_combat_state(2), [action])

    assert meta["assistant_history"] == "none"
    assert meta["history_message_count"] == 0
    assert "round 1 intent" not in next_prompt
    assert "Combat turn process JSON:" not in next_prompt


def test_combat_dialogue_keeps_raw_successes_and_last_three_completed_turns(monkeypatch):
    action = _end_turn_action()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_dialogue",
        prompt_style="analysis",
        run_summary_enabled=False,
        combat_reasoning_pool_enabled=True,
        combat_card_sequences_enabled=False,
        combat_dialogue_history_turns=3,
    )
    responses = iter(_response(_accepted_response(round_number)) for round_number in range(1, 6))
    monkeypatch.setattr(agent, "_chat", lambda _messages: next(responses))

    for round_number in range(1, 5):
        state = _combat_state(round_number)
        chosen, meta = agent.choose(state, [action])
        assert meta["raw_response"] == _accepted_response(round_number)
        agent.record_transition(state, chosen, meta, _combat_state(round_number + 1))

    preview = agent.build_prompt(_combat_state(5), [action])
    assert _accepted_response(1) not in preview
    for round_number in (2, 3, 4):
        assert _accepted_response(round_number) in preview
    assert "memory_contract" not in preview
    assert "authority" not in preview.lower()


def test_combat_dialogue_does_not_retain_failed_response_or_correction(monkeypatch):
    state = _combat_state()
    action = _end_turn_action()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_dialogue",
        prompt_style="analysis",
        run_summary_enabled=False,
        combat_reasoning_pool_enabled=True,
        combat_card_sequences_enabled=False,
        max_retries=1,
    )
    failed = '{"action_id":999,"reason":"bad retry"}'
    replies = iter([_response(failed), _response(_accepted_response(1))])
    monkeypatch.setattr(agent, "_chat", lambda _messages: next(replies))

    chosen, meta = agent.choose(state, [action])
    agent.record_transition(state, chosen, meta, _combat_state(2))
    next_preview = agent.build_prompt(_combat_state(2), [action])

    assert failed in meta["previous_raw_responses"]
    assert failed not in next_preview
    assert "Your previous response was invalid" not in next_preview
    assert _accepted_response(1) in next_preview


def test_combat_dialogue_commits_only_runner_accepted_exchange(monkeypatch):
    state = _combat_state()
    action = _end_turn_action()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_dialogue",
        prompt_style="analysis",
        run_summary_enabled=False,
        combat_reasoning_pool_enabled=True,
        combat_card_sequences_enabled=False,
    )
    rejected = _accepted_response(1)
    accepted = _accepted_response(2)
    replies = iter([_response(rejected), _response(accepted)])
    monkeypatch.setattr(agent, "_chat", lambda _messages: next(replies))
    original_prompt = agent.build_prompt(state, [action])

    first_action, first_meta = agent.choose(state, [action], prompt=original_prompt)
    agent.receive_external_feedback(
        state,
        first_action,
        first_meta,
        "The action reason failed runner validation.",
        [action],
    )
    second_action, second_meta = agent.choose(state, [action], prompt=original_prompt)
    agent.record_transition(state, second_action, second_meta, _combat_state(2))
    next_preview = agent.build_prompt(_combat_state(2), [action])

    assert rejected not in next_preview
    assert "failed runner validation" not in next_preview
    assert accepted in next_preview


def test_combat_dialogue_resets_after_leaving_combat(monkeypatch):
    state = _combat_state()
    action = _end_turn_action()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_dialogue",
        prompt_style="analysis",
        run_summary_enabled=False,
        combat_reasoning_pool_enabled=True,
        combat_card_sequences_enabled=False,
    )
    monkeypatch.setattr(agent, "_chat", lambda _messages: _response(_accepted_response(1)))
    chosen, meta = agent.choose(state, [action])
    agent.record_transition(state, chosen, meta, {"decision": "combat_reward"})

    assert agent.combat_dialogue_buffer.turns == []
    agent.reset_episode()
    assert agent.combat_dialogue_buffer.turns == []


def test_formal_generalist_factory_creates_one_bare_policy_with_fixed_contract():
    agent = agent_from_args(
        "formal_generalist",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_state_only",
        run_summary_enabled=False,
        route_plan_enabled=False,
        reward_posterior_evaluator_enabled=False,
        combat_card_sequences_enabled=False,
        combat_skill_lines_enabled=False,
        rule_projection_and_recent_changes_enabled=False,
    )

    assert isinstance(agent, OpenAICompatAgent)
    assert agent.prompt_profile == "formal_generalist"
    assert agent.active_policy_owner == "formal_generalist"
    assert not hasattr(agent, "ca_agent")
    assert not hasattr(agent, "occ_agent")
    assert not hasattr(agent, "reward_posterior_evaluator")


def test_formal_generalist_prompt_is_structured_neutral_and_static_rules_are_once():
    agent = agent_from_args(
        "formal_generalist",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_state_only",
        run_summary_enabled=False,
        route_plan_enabled=False,
        reward_posterior_evaluator_enabled=False,
        combat_card_sequences_enabled=False,
        combat_skill_lines_enabled=False,
        rule_projection_and_recent_changes_enabled=False,
    )
    prompt = agent.build_prompt(_combat_state(), [_end_turn_action()])
    system = build_formal_generalist_system_message()

    assert "Combat MDP state JSON:" in prompt
    assert "Jaw Worm" in prompt
    assert "Legal actions:" in prompt
    assert render_full_game_rules() not in prompt
    assert system.count(render_full_game_rules()) == 1
    lowered = (system + "\n" + prompt).lower()
    for forbidden in (
        r"\bcli\b",
        "baseline",
        "single-agent",
        "single agent",
        "headless benchmark",
        "persistent route planning is disabled",
        "ablation",
        "active route plan",
        "evidence context",
        "run summary",
        "recent state changes",
    ):
        assert re.search(forbidden, lowered) is None


def test_formal_generalist_choose_audits_static_rules_and_trace_context(monkeypatch):
    agent = agent_from_args(
        "formal_generalist",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_state_only",
        run_summary_enabled=False,
        route_plan_enabled=False,
        reward_posterior_evaluator_enabled=False,
        combat_card_sequences_enabled=False,
        combat_skill_lines_enabled=False,
        rule_projection_and_recent_changes_enabled=False,
        prompt_style="default",
    )
    raw = json.dumps(
        {
            "action_id": 0,
            "selected_action_label": "end turn",
            "reason": "Choose the current legal action.",
        },
        separators=(",", ":"),
    )
    captured = {}

    def fake_chat(messages):
        captured["messages"] = messages
        return _response(raw)

    monkeypatch.setattr(agent, "_chat", fake_chat)
    action, meta = agent.choose(_combat_state(), [_end_turn_action()])

    assert action.kind == "combat_end_turn"
    assert len(captured["messages"]) == 2
    assert captured["messages"][0]["content"].count(render_full_game_rules()) == 1
    assert meta["game_rule_delivery"] == "static_full_system_prompt"
    assert meta["game_rule_exposure_valid"] is True
    assert meta["active_policy_owner"] == "formal_generalist"
    assert meta["assistant_history"] == "none"
    assert len(meta["current_state_payload_sha256"]) == 64
    assert len(meta["legal_actions_sha256"]) == 64


def test_combat_card_select_keeps_actual_hand_separate_from_selection_candidates():
    state = _combat_state()
    state.update(
        {
            "decision": "card_select",
            "prompt": "Choose a card in your Draw Pile and add it into your Hand.",
            "cards": [
                {
                    "index": 0,
                    "id": "CARD.GLACIER",
                    "name": "Glacier",
                    "type": "Skill",
                    "cost": 2,
                    "description": "Gain 6 Block. Channel 2 Frost.",
                }
            ],
            "min_select": 1,
            "max_select": 1,
            "can_skip": False,
        }
    )
    state["combat"]["hand"] = list(state["combat"]["hand"])
    action = LegalAction(
        action_id=0,
        kind="card_select",
        label="select card index 0: Glacier",
        command={"cmd": "action", "action": "select_card", "args": {"indices": [0]}},
    )
    agent = agent_from_args(
        "formal_generalist",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_state_only",
        run_summary_enabled=False,
        route_plan_enabled=False,
        reward_posterior_evaluator_enabled=False,
        combat_card_sequences_enabled=False,
        combat_skill_lines_enabled=False,
        rule_projection_and_recent_changes_enabled=False,
    )

    prompt = agent.build_prompt(state, [action])
    payload_text = prompt.split("Combat MDP state JSON:\n", 1)[1].split("\n\nLegal actions:", 1)[0]
    payload = json.loads(payload_text)

    assert [card["name"] for card in payload["hand_snapshot"]] == ["Zap"]
    assert "candidates" not in payload["selection_context"]
    assert prompt.count('"name":"Glacier"') == 1


def test_managed_combat_view_does_not_replay_prior_observed_changes():
    view = LegalAction(
        action_id=0,
        kind="view_draw_pile",
        label="view draw pile",
        command={"cmd": "bench_view", "view": "draw"},
    )

    fact = _combat_mdp_action_fact(
        _combat_state(),
        view,
        {"changes": ["energy 3 -> 2", "enemy hp 40 -> 30"]},
    )

    assert fact["action_kind"] == "view_draw_pile"
    assert fact["observed_changes"] == []


def test_managed_combat_boundary_uses_enemy_instance_identity_across_reordering():
    before = _combat_memory_snapshot(
        {
            "round": 1,
            "player": {},
            "enemies": [
                {"index": 0, "instance_id": 41, "name": "Fogmog", "hp": 30, "powers": []},
                {"index": 1, "instance_id": 42, "name": "Eye", "hp": 0, "powers": []},
            ],
        }
    )
    after = _combat_memory_snapshot(
        {
            "round": 2,
            "player": {},
            "enemies": [
                {"index": 0, "instance_id": 41, "name": "Fogmog", "hp": 30, "powers": []},
            ],
        }
    )

    facts = _combat_boundary_facts(before, after)

    assert not any(
        fact.get("entity") == "enemy:instance_id:41" and fact.get("field") == "presence"
        for fact in facts
    )
    assert any(
        fact.get("entity") == "enemy:instance_id:42" and fact.get("field") == "presence"
        for fact in facts
    )


@pytest.mark.parametrize(
    ("state", "marker"),
    [
        (
            {
                "decision": "map_select",
                "context": {"act": 1, "floor": 1, "room_type": "Map"},
                "player": {"deck": [], "relics": [], "potions": []},
                "choices": [{"col": 0, "row": 1, "type": "Monster"}],
                "full_map": {
                    "type": "map",
                    "nodes": [{"col": 0, "row": 1, "type": "Monster", "edges": []}],
                },
            },
            "Canonical route map JSON:",
        ),
        (
            {
                "decision": "card_reward",
                "player": {
                    "deck": [{"id": "A", "name": "Zap", "description": "Channel 1 Lightning."}],
                    "relics": [],
                    "potions": [],
                },
                "cards": [
                    {"index": 0, "id": "B", "name": "Ball Lightning", "description": "Deal damage."}
                ],
                "can_skip": True,
            },
            "Decision state JSON:",
        ),
        (
            {
                "decision": "event",
                "player": {"deck": [], "relics": [], "potions": []},
                "event": {"id": "EVENT.TEST", "title": "Test Event"},
                "choices": [{"index": 0, "label": "Proceed"}],
            },
            "Decision state JSON:",
        ),
        (
            {
                "decision": "shop",
                "player": {"gold": 100, "deck": [], "relics": [], "potions": []},
                "shop": {"cards": [{"index": 0, "name": "Zap", "price": 50}]},
            },
            "Decision state JSON:",
        ),
        (
            {
                "decision": "rest_site",
                "player": {"hp": 20, "max_hp": 70, "deck": [], "relics": [], "potions": []},
                "choices": [{"index": 0, "label": "Rest"}],
            },
            "Decision state JSON:",
        ),
    ],
)
def test_formal_generalist_uses_structured_state_for_noncombat_decisions(state, marker):
    agent = agent_from_args(
        "formal_generalist",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_state_only",
        run_summary_enabled=False,
        route_plan_enabled=False,
        reward_posterior_evaluator_enabled=False,
        combat_card_sequences_enabled=False,
        combat_skill_lines_enabled=False,
        rule_projection_and_recent_changes_enabled=False,
    )
    legal_actions = [
        LegalAction(
            action_id=0,
            kind=str(state["decision"]),
            label="choose current option",
            command={"cmd": "action", "action": "choose"},
        ),
        LegalAction(
            action_id=1,
            kind=f"{state['decision']}_skip",
            label="skip",
            command={"cmd": "action", "action": "skip"},
        ),
    ]
    prompt = agent.build_prompt(state, legal_actions)

    assert marker in prompt
    assert "Legal actions:" in prompt
    assert "active_route_plan" not in prompt
    assert "last_action_result" not in prompt


def test_formal_generalist_rejects_scaffolding_before_any_provider_call():
    with pytest.raises(ValueError, match="incompatible settings: route_plan"):
        agent_from_args(
            "formal_generalist",
            base_url="http://127.0.0.1:1",
            model="test",
            api_key="local",
            conversation_mode="combat_state_only",
            run_summary_enabled=False,
            route_plan_enabled=True,
            reward_posterior_evaluator_enabled=False,
            combat_card_sequences_enabled=False,
            combat_skill_lines_enabled=False,
            rule_projection_and_recent_changes_enabled=False,
        )
