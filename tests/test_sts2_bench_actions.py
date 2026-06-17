import json
import os
import urllib.error

import pytest

from sts2_bench.actions import build_legal_actions
from sts2_bench.agents import (
    ModelActionError,
    OpenAICompatAgent,
    RandomAgent,
    build_compact_assistant_message,
    parse_json_object,
)
from sts2_bench.run_benchmark import _redact_argv, load_benchmark_config
from sts2_bench.runner import agent_from_args, run_one


def test_parse_json_object_accepts_trailing_text_after_first_object():
    parsed = parse_json_object('{"action_id": 2, "reason": "take card"}\nExtra rationale.')

    assert parsed == {"action_id": 2, "reason": "take card"}


def test_parse_json_object_accepts_intro_and_fenced_object():
    parsed = parse_json_object(
        'Choice:\n```json\n{"action_id": 1, "reason": "route"}\n```\nDone.'
    )

    assert parsed == {"action_id": 1, "reason": "route"}


def test_meta_view_actions_include_combat_piles():
    state = {
        "decision": "combat_play",
        "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
        "hand": [],
        "enemies": [],
    }

    actions = build_legal_actions(state)
    by_kind = {action.kind: action for action in actions}

    assert by_kind["view_deck"].command == {"cmd": "bench_view", "view": "deck"}
    assert by_kind["view_map"].command == {"cmd": "bench_view", "view": "map"}
    assert by_kind["view_draw_pile"].command == {"cmd": "bench_view", "view": "draw"}
    assert by_kind["view_discard_pile"].command == {"cmd": "bench_view", "view": "discard"}
    assert by_kind["view_exhaust_pile"].command == {"cmd": "bench_view", "view": "exhaust"}


def test_meta_view_actions_do_not_repeat_by_default():
    state = {
        "decision": "map_select",
        "view_deck": True,
        "view_draw_pile": True,
        "choices": [],
    }

    actions = build_legal_actions(state)
    kinds = {action.kind for action in actions}

    assert "view_deck" not in kinds
    assert "view_draw_pile" not in kinds
    assert "view_map" in kinds
    assert "view_discard_pile" in kinds
    assert "view_exhaust_pile" in kinds


def test_card_select_requires_multi_card_actions_when_min_select_is_two():
    state = {
        "decision": "card_select",
        "min_select": 2,
        "max_select": 2,
        "cards": [
            {"index": 0, "name": "Bloodletting"},
            {"index": 1, "name": "Setup Strike"},
            {"index": 2, "name": "Sword Boomerang"},
        ],
    }

    actions = build_legal_actions(state)
    select_actions = [action for action in actions if action.kind == "card_select"]

    assert select_actions
    assert {action.command["args"]["indices"] for action in select_actions} == {
        "0,1",
        "0,2",
        "1,2",
    }
    assert all("," in action.command["args"]["indices"] for action in select_actions)
    assert not any(action.label.startswith("select card ") for action in select_actions)


def test_card_select_keeps_single_card_actions_when_min_select_is_one():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 1,
        "cards": [
            {"index": 3, "name": "Bash"},
            {"index": 4, "name": "Armaments"},
        ],
    }

    actions = build_legal_actions(state)
    select_actions = [action for action in actions if action.kind == "card_select"]

    assert [action.command["args"]["indices"] for action in select_actions] == ["3", "4"]
    assert [action.label for action in select_actions] == [
        "select card 3: Bash",
        "select card 4: Armaments",
    ]


def test_card_select_single_card_choices_are_not_capped():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 1,
        "cards": [{"index": idx, "name": f"Card {idx}"} for idx in range(70)],
    }

    actions = build_legal_actions(state)
    select_actions = [action for action in actions if action.kind == "card_select"]

    assert len(select_actions) == 70
    assert select_actions[-1].command["args"]["indices"] == "69"
    assert select_actions[-1].label == "select card 69: Card 69"


def test_card_select_can_skip_even_when_min_select_is_positive():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 1,
        "can_skip": True,
        "cards": [{"index": 0, "name": "Strike"}],
    }

    actions = build_legal_actions(state)

    assert any(action.kind == "card_select_skip" for action in actions)


def test_bundle_select_action_labels_include_pack_card_names():
    state = {
        "decision": "bundle_select",
        "bundles": [
            {
                "index": 0,
                "cards": [
                    {"name": "Setup Strike"},
                    {"name": "Cinder"},
                    {"name": "Ashen Strike"},
                ],
            },
            {
                "index": 1,
                "cards": [
                    {"name": "Iron Wave"},
                    {"name": "Shrug It Off"},
                    {"name": "Dismantle"},
                ],
            },
        ],
    }

    actions = [action for action in build_legal_actions(state) if action.kind == "bundle_select"]

    assert [action.label for action in actions] == [
        "select bundle 0: Setup Strike, Cinder, Ashen Strike",
        "select bundle 1: Iron Wave, Shrug It Off, Dismantle",
    ]


def test_combat_reward_exposes_skip_and_potion_discard_shortcuts():
    state = {
        "decision": "combat_reward",
        "rewards": [
            {"index": 0, "kind": "potion", "can_claim": False, "can_skip": True},
            {"index": 1, "kind": "gold", "amount": 12, "can_claim": True},
        ],
        "player": {
            "potions": [
                {"index": 0, "name": "Potion A"},
                {"index": 2, "name": "Potion B"},
            ]
        },
    }

    actions = build_legal_actions(state)
    by_kind = {action.kind: action for action in actions}

    assert "combat_reward_skip" in by_kind
    assert by_kind["combat_reward_skip"].command == {
        "cmd": "action",
        "action": "skip_reward",
        "args": {"reward_index": 0},
    }
    assert not any(
        action.kind == "combat_reward" and action.command["args"]["reward_index"] == 0
        for action in actions
    )
    assert any(action.kind == "combat_reward" and action.command["args"]["reward_index"] == 1 for action in actions)
    assert {
        action.command["args"]["potion_index"]
        for action in actions
        if action.kind == "combat_reward_discard_potion"
    } == {0, 2}


def test_event_choice_exposes_leave_when_available():
    state = {
        "decision": "event_choice",
        "can_leave": True,
        "options": [{"index": 0, "title": "Search", "is_locked": False}],
    }

    actions = build_legal_actions(state)

    assert any(action.kind == "event" for action in actions)
    leave = next(action for action in actions if action.kind == "event_leave")
    assert leave.command == {"cmd": "action", "action": "leave_room"}


def test_fake_merchant_shop_exposes_relic_purchase_and_leave_actions():
    state = {
        "decision": "fake_merchant_shop",
        "can_leave": True,
        "relics": [
            {"index": 0, "name": "Bag of Marbles", "cost": 95, "is_stocked": True, "can_buy": True},
            {"index": 1, "name": "Anchor", "cost": 120, "is_stocked": True, "can_buy": False},
            {"index": 2, "name": "Lantern", "cost": 110, "is_stocked": False, "can_buy": False},
        ],
        "player": {"gold": 100},
    }

    actions = build_legal_actions(state)

    buy_actions = [action for action in actions if action.kind == "fake_merchant_buy_relic"]
    assert len(buy_actions) == 1
    assert buy_actions[0].label == "buy fake merchant relic 0: Bag of Marbles for 95 gold"
    assert buy_actions[0].command == {
        "cmd": "action",
        "action": "buy_relic",
        "args": {"relic_index": 0},
    }
    leave = next(action for action in actions if action.kind == "fake_merchant_leave")
    assert leave.command == {"cmd": "action", "action": "leave_room"}


def test_fake_merchant_shop_exposes_foul_potion_action_when_available():
    state = {
        "decision": "fake_merchant_shop",
        "can_throw_foul_potion": True,
        "foul_potion_action": {"action": "use_potion", "potion_index": 2},
        "relics": [],
        "player": {"gold": 0},
    }

    actions = build_legal_actions(state)

    foul_potion = next(action for action in actions if action.kind == "fake_merchant_throw_foul_potion")
    assert foul_potion.command == {
        "cmd": "action",
        "action": "use_potion",
        "args": {"potion_index": 2},
    }


def test_crystal_sphere_exposes_tool_and_cell_actions():
    state = {
        "decision": "crystal_sphere",
        "can_proceed": False,
        "clickable_cells": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    }

    actions = build_legal_actions(state)

    assert {action.command["args"]["tool"] for action in actions if action.kind == "crystal_sphere_tool"} == {
        "big",
        "small",
    }
    assert {
        (action.command["args"]["x"], action.command["args"]["y"])
        for action in actions
        if action.kind == "crystal_sphere_cell"
    } == {(1, 2), (3, 4)}
    assert not any(action.kind == "crystal_sphere" for action in actions)


class _ChooseEndTurnAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "test"}


class _ViewThenEndTurnAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        kind = "combat_end_turn" if state.get("view_deck") else "view_deck"
        action = next(action for action in legal_actions if action.kind == kind)
        return action, {"agent": "test"}


class _RaisingAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        raise ModelActionError("LLM failed after 1 attempt(s): action_id out of range: 99")


class _ErrorProcess:
    def start(self):
        pass

    def start_run(self, **kwargs):
        return {
            "decision": "combat_play",
            "context": {"act": 1, "floor": 1},
            "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
            "energy": 0,
            "hand": [],
            "enemies": [],
        }

    def send(self, command):
        return {"type": "error", "message": "forced engine error"}

    def close(self):
        pass


class _NonTerminalEndTurnProcess:
    def start(self):
        pass

    def start_run(self, **kwargs):
        return {
            "decision": "combat_play",
            "context": {"act": 1, "floor": 1},
            "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
            "energy": 0,
            "max_energy": 3,
            "hand": [],
            "enemies": [],
        }

    def send(self, command):
        return {
            "decision": "combat_play",
            "context": {"act": 1, "floor": 1},
            "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
            "energy": 3,
            "max_energy": 3,
            "round": 2,
            "hand": [],
            "enemies": [],
        }

    def close(self):
        pass


def test_runner_stops_after_engine_error_state():
    result = run_one(
        agent=_ChooseEndTurnAgent(),
        seed="test",
        process=_ErrorProcess(),
        max_steps=20,
    )

    assert result.error == "forced engine error"
    assert result.invalid_states == 1
    assert result.steps == 1
    assert not result.truncated


def test_runner_max_game_actions_does_not_count_view_actions():
    result = run_one(
        agent=_ViewThenEndTurnAgent(),
        seed="test",
        process=_NonTerminalEndTurnProcess(),
        max_steps=20,
        max_game_actions=1,
    )

    assert result.steps == 2
    assert result.view_deck_count == 1
    assert result.game_action_count == 1
    assert result.truncated
    assert result.extra["truncation_reason"] == "max_game_actions exceeded (1)"


def test_runner_stops_without_action_and_keeps_state_when_agent_decision_fails():
    result = run_one(
        agent=_RaisingAgent(),
        seed="test",
        process=_NonTerminalEndTurnProcess(),
        max_steps=20,
    )

    assert "action_id out of range: 99" in result.error
    assert result.steps == 0
    assert result.game_action_count == 0
    assert result.floor == 1
    assert result.hp == 80
    assert not result.truncated


def test_openai_compat_agent_prompt_includes_short_term_memory():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        memory_enabled=True,
        memory_window=2,
    )
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 120, "deck_size": 12},
        "choices": [],
    }
    action = next(
        action
        for action in build_legal_actions(
            {
                **state,
                "choices": [{"col": 1, "row": 3, "type": "Monster"}],
            }
        )
        if action.kind == "map"
    )

    agent.record_transition(
        state,
        action,
        {"parsed": {"reason": "Need one more hallway reward before resting."}},
    )
    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Episode memory:" in prompt
    assert "decision=map_select" in prompt
    assert "floor=3" in prompt
    assert "action=go to Monster at col=1 row=3" in prompt
    assert "Need one more hallway reward before resting." in prompt


def test_openai_compat_agent_prompt_includes_rule_based_run_summary():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        run_summary_enabled=True,
    )
    state = {
        "decision": "combat_play",
        "context": {
            "act": 1,
            "floor": 17,
            "room_type": "Boss",
            "boss": {"name": "Lagavulin Matriarch"},
        },
        "player": {
            "hp": 18,
            "max_hp": 85,
            "gold": 66,
            "deck_size": 22,
            "potion_slots": 4,
            "potions": [],
            "relics": [{"name": "Burning Blood"}, {"name": "Reptile Trinket"}],
        },
        "round": 9,
        "energy": 3,
        "max_energy": 3,
        "player_powers": [{"name": "Dexterity", "amount": -4, "type": "Buff"}],
        "enemies": [
            {
                "index": 0,
                "name": "Lagavulin Matriarch",
                "hp": 101,
                "max_hp": 222,
                "block": 12,
                "intents": [{"type": "Buff"}, {"type": "Debuff"}],
            }
        ],
        "hand": [],
    }

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Run summary:" in prompt
    assert "Card damage and block values shown in the state are engine preview values" in prompt
    assert "- position: act=1 floor=17 room=Boss boss=Lagavulin Matriarch" in prompt
    assert "- resources: hp=18/85 gold=66 deck_size=22 potions=0/4 relic_count=2" in prompt
    assert "low HP; survival and rest decisions need extra scrutiny" in prompt
    assert "no potions available for this high-risk fight" in prompt
    assert "negative player powers: Dexterity(-4)" in prompt
    assert "visible enemy intents include no attack damage this turn" in prompt


def test_openai_compat_agent_run_summary_can_be_disabled():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        run_summary_enabled=False,
    )
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 120, "deck_size": 12},
        "choices": [],
    }

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Run summary:" not in prompt


def test_openai_compat_agent_prompt_requires_selected_action_label_echo():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
    )
    state = _combat_turn_state()

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "selected_action_label" in prompt
    assert "copy the selected legal action label exactly" in prompt


def test_openai_compat_agent_prompt_explains_atomic_card_sequence_actions():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
    )
    state = _combat_turn_state()

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "combat_card_sequence" in prompt
    assert "engine-side queued card sequence" in prompt


def test_openai_compat_agent_factual_memory_records_transition_diffs_not_reasons():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        memory_enabled=True,
        memory_window=2,
        memory_mode="factual_diff",
    )
    old_state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 5,
        "discard_pile_count": 0,
        "hand": [{"index": 0, "name": "Strike", "cost": 1, "type": "Attack", "target_type": "AnyEnemy", "can_play": True}],
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 20, "max_hp": 40, "block": 0}],
    }
    new_state = {
        **old_state,
        "energy": 2,
        "hand": [],
        "discard_pile_count": 1,
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 14, "max_hp": 40, "block": 0}],
    }
    action = next(action for action in build_legal_actions(old_state) if action.label.startswith("play card 0"))

    agent.record_transition(
        old_state,
        action,
        {"parsed": {"reason": "This reason should not be stored."}},
        new_state,
    )
    prompt = agent.build_prompt(new_state, build_legal_actions(new_state))

    assert "Episode memory:" in prompt
    assert "action=play card 0: Strike on enemy 0: Jaw Worm" in prompt
    assert "changes=energy: 3/3 -> 2/3" in prompt
    assert "discard_pile: 0 -> 1" in prompt
    assert "enemy Jaw Worm[0]: hp 20/40 -> 14/40" in prompt
    assert "This reason should not be stored." not in prompt


def test_agent_memory_resets_between_runs():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        memory_enabled=True,
    )
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 1, "room_type": "Monster"},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck_size": 10},
        "energy": 0,
        "hand": [],
        "enemies": [],
    }
    action = next(action for action in build_legal_actions(state) if action.kind == "combat_end_turn")

    agent.record_transition(state, action, {"parsed": {"reason": "No energy."}})
    assert "Episode memory:" in agent.build_prompt(state, build_legal_actions(state))

    agent.reset_episode()

    assert "Episode memory:" not in agent.build_prompt(state, build_legal_actions(state))


def _combat_turn_state(**overrides):
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {
            "hp": 70,
            "max_hp": 80,
            "block": 0,
            "gold": 0,
            "deck_size": 10,
            "potion_slots": 3,
            "potions": [],
        },
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 5,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
        "hand": [
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            }
        ],
        "enemies": [
            {
                "index": 0,
                "name": "Jaw Worm",
                "hp": 20,
                "max_hp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 7}],
            }
        ],
    }
    state.update(overrides)
    return state


def _turn_chat_agent(responses, **kwargs):
    max_retries = kwargs.pop("max_retries", 0)
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="turn_chat",
        memory_enabled=False,
        max_retries=max_retries,
        **kwargs,
    )
    sent_messages = []
    response_iter = iter(responses)

    def fake_chat(messages):
        sent_messages.append([dict(message) for message in messages])
        response = next(response_iter)
        if isinstance(response, BaseException):
            raise response
        return {"choices": [{"message": {"content": json.dumps(response)}}]}

    agent._chat = fake_chat
    return agent, sent_messages


def _single_turn_agent(responses, **kwargs):
    max_retries = kwargs.pop("max_retries", 0)
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        max_retries=max_retries,
        **kwargs,
    )
    sent_messages = []
    response_iter = iter(responses)

    def fake_chat(messages):
        sent_messages.append([dict(message) for message in messages])
        response = next(response_iter)
        if isinstance(response, BaseException):
            raise response
        return {"choices": [{"message": {"content": json.dumps(response)}}]}

    agent._chat = fake_chat
    return agent, sent_messages


def test_openai_compat_agent_retries_invalid_action_before_strict_error():
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 99, "reason": "bad id"},
            {"action_id": 5, "reason": "corrected to legal Strike."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.kind == "combat_play_card"
    assert meta["attempt"] == 1
    assert len(sent_messages) == 2
    assert "previous response was invalid" in sent_messages[1][-1]["content"]
    assert "between 0 and" in sent_messages[1][-1]["content"]


def test_openai_compat_agent_single_turn_retries_transport_error_without_prompt_correction(monkeypatch):
    monkeypatch.setattr("sts2_bench.agents.time.sleep", lambda _seconds: None)
    state = _combat_turn_state()
    agent, sent_messages = _single_turn_agent(
        [
            urllib.error.URLError("temporary EOF"),
            {"action_id": 5, "reason": "Recovered after transport retry."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state), prompt="choose an action")

    assert action.action_id == 5
    assert meta["attempt"] == 1
    assert len(sent_messages) == 2
    assert sent_messages[1] == sent_messages[0]
    assert "previous response was invalid" not in "\n".join(message["content"] for message in sent_messages[1])


def test_openai_compat_agent_turn_chat_retries_transport_error_without_prompt_correction(monkeypatch):
    monkeypatch.setattr("sts2_bench.agents.time.sleep", lambda _seconds: None)
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [
            urllib.error.URLError("temporary EOF"),
            {"action_id": 5, "reason": "Recovered after transport retry."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.action_id == 5
    assert meta["attempt"] == 1
    assert len(sent_messages) == 2
    assert sent_messages[1] == sent_messages[0]
    assert "previous response was invalid" not in "\n".join(message["content"] for message in sent_messages[1])


def test_openai_compat_agent_remaps_when_selected_label_matches_current_action():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": end_turn.action_id,
                "selected_action_label": strike.label,
                "reason": "Play Strike now.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["attempt"] == 0
    assert meta["action_id_remapped_from"] == end_turn.action_id
    assert meta["resolved_action_id"] == strike.action_id
    assert len(sent_messages) == 1


def test_openai_compat_agent_resolves_missing_action_id_from_exact_selected_label():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "selected_action_label": strike.label,
                "reason": "Play Strike now.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["attempt"] == 0
    assert meta["action_id_resolved_from_label"] is True
    assert meta["resolved_action_id"] == strike.action_id
    assert len(sent_messages) == 1


def test_openai_compat_agent_resolves_action_alias_as_selected_label():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action": strike.label,
                "reason": "Play Strike now.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["action_id_resolved_from_label"] is True
    assert meta["resolved_action_id"] == strike.action_id
    assert len(sent_messages) == 1


def test_openai_compat_agent_retries_when_selected_label_matches_no_current_action():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": end_turn.action_id,
                "selected_action_label": "play card 99: Ghost Strike",
                "reason": "Play Strike now.",
            },
            {
                "action_id": strike.action_id,
                "selected_action_label": strike.label,
                "reason": "Play Strike now.",
            },
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["attempt"] == 1
    assert "selected_action_label" in sent_messages[1][-1]["content"]


def test_openai_compat_agent_trusts_exact_selected_label_over_discursive_reason():
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "name": "Bloodletting",
                "cost": 0,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
            }
        ],
        energy=0,
    )
    actions = build_legal_actions(state)
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": end_turn.action_id,
                "selected_action_label": end_turn.label,
                "reason": "Play Bloodletting would cost HP here, so skip it.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == end_turn
    assert meta["attempt"] == 0
    assert len(sent_messages) == 1


def test_openai_compat_agent_retries_when_reason_mentions_different_current_action():
    state = _combat_turn_state(
        view_deck=True,
        view_map=True,
        view_draw_pile=True,
        view_discard_pile=True,
        view_exhaust_pile=True,
        hand=[
            {
                "index": 4,
                "name": "Pommel Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 9},
            }
        ],
    )
    actions = build_legal_actions(state)
    pommel = next(action for action in actions if action.kind == "combat_play_card")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": end_turn.action_id,
                "reason": "Play Pommel Strike first to deal damage before ending.",
            },
            {
                "action_id": pommel.action_id,
                "reason": "Play Pommel Strike first to deal damage.",
            },
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == pommel
    assert meta["attempt"] == 1
    assert "does not match selected action" in sent_messages[1][-1]["content"]


def test_combat_play_exposes_atomic_card_sequence_actions():
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 1,
                "name": "Defend",
                "cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
    )
    actions = build_legal_actions(state)

    sequence = next(action for action in actions if action.kind == "combat_card_sequence")

    assert sequence.label == "play sequence: 0 Strike on enemy 0 -> 1 Defend"
    assert sequence.command == {
        "cmd": "bench_sequence",
        "sequence": [
            {"card_index": 0, "target_index": 0},
            {"card_index": 1},
        ],
    }


def test_combat_play_exposes_custom_card_sequence_gateway():
    actions = build_legal_actions(_combat_turn_state())

    custom = next(action for action in actions if action.kind == "combat_custom_sequence")

    assert custom.label == "propose custom deterministic card sequence"
    assert custom.command == {"cmd": "bench_custom_sequence"}


def test_custom_card_sequence_gateway_can_be_disabled_for_env_action_masks():
    actions = build_legal_actions(_combat_turn_state(), allow_custom_sequences=False)

    assert not any(action.kind == "combat_custom_sequence" for action in actions)


class _CapturingRandomAgent(RandomAgent):
    def __init__(self):
        super().__init__(seed=0)
        self.seen_legal_actions = []

    def choose(self, state, legal_actions, *, prompt=None):
        self.seen_legal_actions = list(legal_actions)
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "random"}


class _OneCombatTurnProcess:
    def start(self):
        pass

    def start_run(self, **kwargs):
        return _combat_turn_state()

    def send(self, command):
        return {"decision": "game_over", "victory": False, "player": {"hp": 70, "max_hp": 80}}

    def close(self):
        pass


def test_runner_does_not_expose_custom_sequence_gateway_to_random_agent():
    agent = _CapturingRandomAgent()

    run_one(agent=agent, seed="test", process=_OneCombatTurnProcess(), max_steps=1)

    assert agent.seen_legal_actions
    assert not any(action.kind == "combat_custom_sequence" for action in agent.seen_legal_actions)


def test_openai_compat_agent_accepts_custom_sequence_payload():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    custom = next(action for action in actions if action.kind == "combat_custom_sequence")
    agent, _sent_messages = _turn_chat_agent(
        [
            {
                "action_id": custom.action_id,
                "selected_action_label": custom.label,
                "sequence": [
                    {"card_index": 0, "target_index": 0},
                    {"card_index": 1},
                ],
                "reason": "Use a deterministic custom card sequence.",
            }
        ]
    )

    action, meta = agent.choose(state, actions)

    assert action == custom
    assert meta["parsed"]["sequence"] == [
        {"card_index": 0, "target_index": 0},
        {"card_index": 1},
    ]


class _ChooseSequenceAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        action = next(action for action in legal_actions if action.kind == "combat_card_sequence")
        return action, {"agent": "test"}


class _SequenceProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _combat_turn_state(
            hand=[
                {
                    "index": 0,
                    "instance_id": "strike-a",
                    "name": "Strike",
                    "cost": 1,
                    "energy_cost": 1,
                    "type": "Attack",
                    "target_type": "AnyEnemy",
                    "can_play": True,
                    "stats": {"damage": 6},
                },
                {
                    "index": 1,
                    "instance_id": "defend-b",
                    "name": "Defend",
                    "cost": 1,
                    "energy_cost": 1,
                    "type": "Skill",
                    "target_type": "None",
                    "can_play": True,
                    "stats": {"block": 5},
                },
            ],
            enemies=[
                {
                    "index": 0,
                    "instance_id": "jaw-worm",
                    "name": "Jaw Worm",
                    "hp": 20,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 7}],
                }
            ],
        )

    def send(self, command):
        self.sent.append(command)
        if len(self.sent) == 1:
            return _combat_turn_state(
                energy=2,
                hand=[
                    {
                        "index": 0,
                        "instance_id": "defend-b",
                        "name": "Defend",
                        "cost": 1,
                        "energy_cost": 1,
                        "type": "Skill",
                        "target_type": "None",
                        "can_play": True,
                        "stats": {"block": 5},
                    }
                ],
                enemies=[
                    {
                        "index": 0,
                        "instance_id": "jaw-worm",
                        "name": "Jaw Worm",
                        "hp": 14,
                        "max_hp": 40,
                        "block": 0,
                        "intents": [{"type": "Attack", "damage": 7}],
                    }
                ],
            )
        return _combat_turn_state(energy=1, hand=[], enemies=[])

    def close(self):
        pass


def test_runner_executes_card_sequence_as_one_benchmark_action():
    process = _SequenceProcess()

    result = run_one(
        agent=_ChooseSequenceAgent(),
        seed="test",
        process=process,
        max_steps=1,
        max_game_actions=1,
    )

    assert [command["args"] for command in process.sent] == [
        {"card_index": 0, "target_index": 0},
        {"card_index": 0},
    ]
    assert result.steps == 1
    assert result.game_action_count == 1
    assert result.extra["last_sequence_steps"] == 2


class _ChooseCustomSequenceAgent:
    def __init__(self, sequence):
        self.sequence = sequence

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        action = next(action for action in legal_actions if action.kind == "combat_custom_sequence")
        return action, {
            "agent": "test",
            "parsed": {
                "sequence": self.sequence,
                "reason": "Use a custom deterministic sequence.",
            },
        }


def _custom_sequence_state(**overrides):
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "instance_id": "energy-a",
                "name": "Energy Tap",
                "cost": 0,
                "energy_cost": 0,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"energy": 1},
            },
            {
                "index": 1,
                "instance_id": "big-hit",
                "name": "Big Hit",
                "cost": 4,
                "energy_cost": 4,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": False,
                "stats": {"damage": 20},
            },
        ],
        enemies=[
            {
                "index": 0,
                "instance_id": "jaw-worm",
                "name": "Jaw Worm",
                "hp": 30,
                "max_hp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 7}],
            }
        ],
    )
    state.update(overrides)
    return state


class _CustomSequenceProcess:
    def __init__(self, *, drift=False):
        self.sent = []
        self.drift = drift

    def start(self):
        pass

    def start_run(self, **kwargs):
        state = _custom_sequence_state()
        if self.drift:
            state["enemies"] = [{**state["enemies"][0], "hp": 99}]
        return state

    def send(self, command):
        self.sent.append(command)
        args = command.get("args") or {}
        if args.get("card_index") == 0:
            return _custom_sequence_state(
                energy=4,
                hand=[
                    {
                        "index": 0,
                        "instance_id": "big-hit",
                        "name": "Big Hit",
                        "cost": 4,
                        "energy_cost": 4,
                        "type": "Attack",
                        "target_type": "AnyEnemy",
                        "can_play": True,
                        "stats": {"damage": 20},
                    }
                ],
            )
        if args.get("card_index") == 0 and args.get("target_index") == 0:
            raise AssertionError("unreachable duplicate branch")
        return _custom_sequence_state(
            energy=0,
            hand=[],
            enemies=[
                {
                    "index": 0,
                    "instance_id": "jaw-worm",
                    "name": "Jaw Worm",
                    "hp": 10,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 7}],
                }
            ],
        )

    def close(self):
        pass


class _CustomSequenceMainProcess(_CustomSequenceProcess):
    def send(self, command):
        self.sent.append(command)
        if len(self.sent) == 1:
            return _custom_sequence_state(
                energy=4,
                hand=[
                    {
                        "index": 0,
                        "instance_id": "big-hit",
                        "name": "Big Hit",
                        "cost": 4,
                        "energy_cost": 4,
                        "type": "Attack",
                        "target_type": "AnyEnemy",
                        "can_play": True,
                        "stats": {"damage": 20},
                    }
                ],
            )
        return _custom_sequence_state(
            energy=0,
            hand=[],
            enemies=[
                {
                    "index": 0,
                    "instance_id": "jaw-worm",
                    "name": "Jaw Worm",
                    "hp": 10,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 7}],
                }
            ],
        )


def test_runner_validates_custom_sequence_with_probe_replay_before_main_execution():
    main = _CustomSequenceMainProcess()
    probe = _CustomSequenceMainProcess()
    sequence = [
        {"card_index": 0},
        {"card_index": 1, "target_index": 0},
    ]

    result = run_one(
        agent=_ChooseCustomSequenceAgent(sequence),
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=1,
        max_game_actions=1,
    )

    assert [command["args"] for command in probe.sent] == [
        {"card_index": 0},
        {"card_index": 0, "target_index": 0},
    ]
    assert [command["args"] for command in main.sent] == [
        {"card_index": 0},
        {"card_index": 0, "target_index": 0},
    ]
    assert result.error is None
    assert result.extra["last_sequence_validation"]["validated"] is True
    assert result.extra["last_sequence_steps"] == 2


def test_runner_ignores_extra_custom_sequence_target_for_non_enemy_cards():
    main = _CustomSequenceMainProcess()
    probe = _CustomSequenceMainProcess()
    sequence = [
        {"card_index": 0, "target_index": 0},
        {"card_index": 1, "target_index": 0},
    ]

    result = run_one(
        agent=_ChooseCustomSequenceAgent(sequence),
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=1,
        max_game_actions=1,
    )

    assert [command["args"] for command in probe.sent] == [
        {"card_index": 0},
        {"card_index": 0, "target_index": 0},
    ]
    assert [command["args"] for command in main.sent] == [
        {"card_index": 0},
        {"card_index": 0, "target_index": 0},
    ]
    assert result.error is None
    assert result.extra["last_sequence_validation"]["validated"] is True
    assert result.extra["last_sequence_steps"] == 2


def _early_terminal_custom_sequence_state():
    return _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "instance_id": "molten-fist",
                "name": "Molten Fist",
                "cost": 1,
                "energy_cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "description": "Deal 13 damage. Exhaust.",
                "stats": {"damage": 13},
            },
            {
                "index": 1,
                "instance_id": "strike-a",
                "name": "Strike",
                "cost": 1,
                "energy_cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 2,
                "instance_id": "strike-b",
                "name": "Strike",
                "cost": 1,
                "energy_cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
        ],
        enemies=[
            {
                "index": 0,
                "instance_id": "fuzzy-wurm-crawler",
                "name": "Fuzzy Wurm Crawler",
                "hp": 13,
                "max_hp": 56,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 11}],
            }
        ],
    )


class _EarlyTerminalCustomSequenceProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _early_terminal_custom_sequence_state()

    def send(self, command):
        self.sent.append(command)
        return {"decision": "game_over", "victory": False, "player": {"hp": 56, "max_hp": 80}}

    def close(self):
        pass


def test_runner_accepts_custom_sequence_prefix_when_combat_ends_early():
    main = _EarlyTerminalCustomSequenceProcess()
    probe = _EarlyTerminalCustomSequenceProcess()
    sequence = [
        {"card_index": 0, "target_index": 0},
        {"card_index": 1, "target_index": 0},
        {"card_index": 2, "target_index": 0},
    ]

    result = run_one(
        agent=_ChooseCustomSequenceAgent(sequence),
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=3,
    )

    assert [command["args"] for command in probe.sent] == [{"card_index": 0, "target_index": 0}]
    assert [command["args"] for command in main.sent] == [{"card_index": 0, "target_index": 0}]
    assert result.error is None
    assert result.decision == "game_over"
    assert result.extra["last_sequence_validation"]["validated"] is True
    assert result.extra["last_sequence_validation"]["steps_executed"] == 1
    assert result.extra["last_sequence_steps"] == 1


def test_runner_rejects_custom_sequence_when_probe_replay_drifts():
    main = _CustomSequenceMainProcess()
    probe = _CustomSequenceMainProcess(drift=True)
    sequence = [
        {"card_index": 0},
        {"card_index": 1, "target_index": 0},
    ]

    result = run_one(
        agent=_ChooseCustomSequenceAgent(sequence),
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=1,
    )

    assert main.sent == []
    assert "drift" in result.error
    assert result.game_action_count == 0


def test_runner_rejects_custom_sequence_over_validation_limit_without_probe():
    main = _CustomSequenceMainProcess()

    def fail_probe_factory():
        raise AssertionError("probe should not be started for overlong sequence")

    result = run_one(
        agent=_ChooseCustomSequenceAgent([{"card_index": 0}] * 5),
        seed="test",
        process=main,
        sequence_probe_factory=fail_probe_factory,
        max_steps=1,
    )

    assert main.sent == []
    assert "at most 4" in result.error


def _unsafe_custom_sequence_state():
    return _custom_sequence_state(
        hand=[
            {
                "index": 0,
                "instance_id": "chaos-a",
                "name": "Chaos",
                "cost": 0,
                "energy_cost": 0,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "description": "Add a random card to your hand.",
                "stats": {},
            },
            {
                "index": 1,
                "instance_id": "defend-b",
                "name": "Defend",
                "cost": 1,
                "energy_cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"block": 5},
            },
        ]
    )


class _UnsafeCustomSequenceProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _unsafe_custom_sequence_state()

    def send(self, command):
        self.sent.append(command)
        if len(self.sent) == 1:
            return _custom_sequence_state(
                energy=3,
                hand=[
                    {
                        "index": 0,
                        "instance_id": "defend-b",
                        "name": "Defend",
                        "cost": 1,
                        "energy_cost": 1,
                        "type": "Skill",
                        "target_type": "None",
                        "can_play": True,
                        "stats": {"block": 5},
                    }
                ],
            )
        return _custom_sequence_state(energy=2, hand=[])

    def close(self):
        pass


def test_runner_rejects_custom_sequence_with_random_or_choice_card_before_main_execution():
    main = _UnsafeCustomSequenceProcess()
    probe = _UnsafeCustomSequenceProcess()

    result = run_one(
        agent=_ChooseCustomSequenceAgent([{"card_index": 0}, {"card_index": 1}]),
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=1,
    )

    assert probe.sent == []
    assert main.sent == []
    assert "non-deterministic" in result.error


def _juggernaut_sequence_state():
    return _combat_turn_state(
        hand=[
            {
                "index": 0,
                "instance_id": "juggernaut",
                "name": "Juggernaut",
                "cost": 0,
                "energy_cost": 0,
                "type": "Power",
                "target_type": "Self",
                "can_play": True,
                "description": "Whenever you gain Block, deal 5 damage to a random enemy.",
                "stats": {"juggernautpower": 5},
            },
            {
                "index": 1,
                "instance_id": "defend-a",
                "name": "Defend",
                "cost": 1,
                "energy_cost": 1,
                "type": "Skill",
                "target_type": "Self",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
    )


class _InvalidCustomSequenceThenEndTurnAgent:
    def __init__(self, *, invalid_calls=1):
        self.calls = 0
        self.invalid_calls = invalid_calls
        self.feedback = []

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        self.calls += 1
        if self.calls <= self.invalid_calls:
            action = next(action for action in legal_actions if action.kind == "combat_custom_sequence")
            return action, {
                "agent": "test",
                "parsed": {
                    "sequence": [{"card_index": 0}, {"card_index": 1}],
                    "reason": "Try an invalid custom sequence.",
                },
            }
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "test", "parsed": {"reason": "Use a legal action after feedback."}}

    def receive_external_feedback(self, state, action, meta, feedback, legal_actions):
        self.feedback.append(feedback)


class _JuggernautSequenceProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _juggernaut_sequence_state()

    def send(self, command):
        self.sent.append(command)
        return {"decision": "game_over", "victory": False, "player": {"hp": 78, "max_hp": 80}}

    def close(self):
        pass


def test_runner_reasks_after_custom_sequence_validation_failure_before_stopping():
    agent = _InvalidCustomSequenceThenEndTurnAgent()
    main = _JuggernautSequenceProcess()
    probe = _JuggernautSequenceProcess()

    result = run_one(
        agent=agent,
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=3,
    )

    assert agent.calls == 2
    assert len(agent.feedback) == 1
    assert "custom sequence validation failed" in agent.feedback[0]
    assert "Juggernaut" in agent.feedback[0]
    assert probe.sent == []
    assert [command["action"] for command in main.sent] == ["end_turn"]
    assert result.error is None
    assert result.extra["custom_sequence_validation_retry_count"] == 1


def test_runner_reasks_twice_after_repeated_custom_sequence_validation_failure():
    agent = _InvalidCustomSequenceThenEndTurnAgent(invalid_calls=2)
    main = _JuggernautSequenceProcess()
    probe = _JuggernautSequenceProcess()

    result = run_one(
        agent=agent,
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=4,
    )

    assert agent.calls == 3
    assert len(agent.feedback) == 2
    assert "custom sequence validation failed" in agent.feedback[1]
    assert "do not predict shifted card indices" in agent.feedback[1]
    assert [command["action"] for command in main.sent] == ["end_turn"]
    assert result.error is None
    assert result.extra["custom_sequence_validation_retry_count"] == 2


def test_combat_card_sequence_omits_draw_or_random_cards():
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 1,
                "name": "Battle Trance",
                "cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"cards": 3},
                "description": "Draw 3 cards.",
            },
        ],
    )
    actions = build_legal_actions(state)

    assert not any(action.kind == "combat_card_sequence" for action in actions)


def test_openai_compat_agent_strict_mode_raises_after_invalid_action_without_fallback():
    state = _combat_turn_state()
    agent, _sent_messages = _turn_chat_agent(
        [{"action_id": 99, "reason": "bad id"}],
        max_retries=0,
        fallback_mode="error",
    )

    with pytest.raises(ModelActionError, match="action_id out of range: 99"):
        agent.choose(state, build_legal_actions(state))


def test_compact_assistant_history_omits_state_local_action_ids():
    state = _combat_turn_state()
    action = next(action for action in build_legal_actions(state) if action.kind == "combat_play_card")

    payload = json.loads(build_compact_assistant_message(action, {"reason": "Play Strike."}))

    assert "action_id" not in payload
    assert payload["action"] == action.label
    assert payload["action_kind"] == action.kind


def test_openai_compat_agent_turn_chat_keeps_history_within_combat_turn():
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        energy=2,
        hand=[],
        discard_pile_count=1,
        enemies=[
            {
                "index": 0,
                "name": "Jaw Worm",
                "hp": 14,
                "max_hp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 7}],
            }
        ],
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": ["energy: 3/3 -> 2/3"],
        },
    )
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": 5,
                "reason": "Strike now.",
                "calculations": ["full private-ish calculation should not persist"],
                "candidates": [{"action_id": 5, "label": "play card"}],
            },
            {"action_id": 5, "reason": "No useful cards remain."},
        ]
    )

    first_action, _first_meta = agent.choose(state, build_legal_actions(state))
    second_action, second_meta = agent.choose(next_state, build_legal_actions(next_state))

    assert first_action.kind == "combat_play_card"
    assert second_action.kind == "combat_end_turn"
    assert len(sent_messages[1]) == 4
    assistant_history = sent_messages[1][2]["content"]
    assert "Strike now." in assistant_history
    assert "calculations" not in assistant_history
    assert "candidates" not in assistant_history
    assert "Turn update." in sent_messages[1][3]["content"]
    assert "Current turn state:" in sent_messages[1][3]["content"]
    assert second_meta["conversation_mode"] == "turn_chat"
    assert second_meta["turn_chat_history_turns"] == 1


def test_openai_compat_agent_turn_chat_plan_is_added_to_initial_combat_prompt_and_kept_compact():
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        energy=2,
        hand=[],
        discard_pile_count=1,
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": ["energy: 3/3 -> 2/3"],
        },
    )
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "turn_plan": {
                    "objective": "Deal damage before blocking.",
                    "replan_if": ["draw changes hand"],
                },
                "action_id": 5,
                "reason": "Start with Strike.",
                "calculations": ["do not keep this verbose field"],
            },
            {"action_id": 5, "reason": "End after the plan."},
        ],
        turn_chat_plan_enabled=True,
    )

    agent.choose(state, build_legal_actions(state))
    agent.choose(next_state, build_legal_actions(next_state))

    assert "turn_plan" in sent_messages[0][0]["content"]
    assert "visible hand in detail" in sent_messages[0][0]["content"]
    assert "Turn planning instruction:" not in sent_messages[0][1]["content"]
    assistant_history = sent_messages[1][2]["content"]
    assert "turn_plan" in assistant_history
    assert "Deal damage before blocking." in assistant_history
    assert "calculations" not in assistant_history


def test_openai_compat_agent_turn_chat_omits_plan_prompt_by_default():
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent([{"action_id": 5, "reason": "Strike."}])

    agent.choose(state, build_legal_actions(state))

    assert "turn_plan" not in sent_messages[0][0]["content"]
    assert "Turn planning instruction:" not in sent_messages[0][1]["content"]
    assert "turn_plan" not in sent_messages[0][1]["content"]


def test_openai_compat_agent_turn_chat_clears_after_end_turn_transition():
    state = _combat_turn_state(energy=0, hand=[])
    next_state = _combat_turn_state(round=2, energy=3, hand=[])
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 5, "reason": "End the spent turn."},
            {"action_id": 5, "reason": "Fresh turn prompt."},
        ]
    )

    action, meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, action, meta, next_state)
    agent.choose(next_state, build_legal_actions(next_state))

    assert action.kind == "combat_end_turn"
    assert len(sent_messages[1]) == 2
    assert "End the spent turn." not in sent_messages[1][-1]["content"]
    assert "New player turn." in sent_messages[1][-1]["content"]


def test_openai_compat_agent_turn_chat_does_not_keep_noncombat_history():
    combat_state = _combat_turn_state()
    map_state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 0, "deck_size": 10},
        "choices": [],
    }
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 5, "reason": "Strike before leaving combat context."},
            {"action_id": 0, "reason": "View deck on map."},
        ]
    )

    agent.choose(combat_state, build_legal_actions(combat_state))
    agent.choose(map_state, build_legal_actions(map_state))

    assert len(sent_messages[1]) == 2
    assert "Strike before leaving combat context." not in sent_messages[1][-1]["content"]
    assert "Game state:" in sent_messages[1][-1]["content"]


def test_openai_compat_agent_turn_chat_window_trims_old_pairs():
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 5, "reason": "first action"},
            {"action_id": 5, "reason": "second action"},
            {"action_id": 5, "reason": "third action"},
        ],
        turn_chat_window=1,
    )

    actions = build_legal_actions(state)
    agent.choose(state, actions)
    agent.choose(state, actions)
    agent.choose(state, actions)

    third_transcript = "\n".join(message["content"] for message in sent_messages[2])
    assert "first action" not in third_transcript
    assert "second action" in third_transcript


def test_openai_compat_agent_turn_chat_view_update_keeps_current_turn_reminder():
    state = _combat_turn_state()
    viewed_state = _combat_turn_state(
        view_draw_pile=True,
        draw_pile=[{"index": 0, "name": "Defend", "cost": 1, "type": "Skill", "stats": {"block": 5}}],
    )
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 2, "reason": "Check draw pile."},
            {"action_id": 5, "reason": "End after checking."},
        ]
    )

    agent.choose(state, build_legal_actions(state))
    agent.choose(viewed_state, build_legal_actions(viewed_state))

    update = sent_messages[1][-1]["content"]
    assert "Requested information: draw pile." in update
    assert "Viewed draw pile:" in update
    assert "Current turn reminder:" in update
    assert "Legal actions:" in update


def test_openai_compat_agent_turn_chat_treats_combat_card_select_as_same_turn_modal():
    state = _combat_turn_state()
    modal_state = {
        "decision": "card_select",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": state["player"],
        "min_select": 1,
        "max_select": 1,
        "prompt": "Exhaust 1 card.",
        "source_card": {"name": "Burning Pact"},
        "combat": {
            "round": 1,
            "energy": 2,
            "max_energy": 3,
            "draw_pile_count": 4,
            "discard_pile_count": 1,
            "exhaust_pile_count": 0,
            "enemies": state["enemies"],
        },
        "cards": [{"index": 0, "name": "Injury", "cost": 0, "type": "Curse"}],
    }
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 5, "reason": "Play Burning Pact."},
            {"action_id": 5, "reason": "Exhaust Injury."},
        ]
    )

    agent.choose(state, build_legal_actions(state))
    agent.choose(modal_state, build_legal_actions(modal_state))

    assert len(sent_messages[1]) == 4
    assert "Play Burning Pact." in sent_messages[1][2]["content"]
    assert "Turn modal." in sent_messages[1][3]["content"]
    assert "Decision: card_select" in sent_messages[1][3]["content"]


def test_openai_compat_agent_rejects_turn_chat_with_episode_memory():
    with pytest.raises(ValueError, match="parallel"):
        OpenAICompatAgent(
            base_url="http://127.0.0.1:1",
            model="test",
            conversation_mode="turn_chat",
            memory_enabled=True,
        )


def test_run_config_redacts_api_key_cli_arg():
    assert _redact_argv(["--api-key", "secret", "--model", "m"]) == [
        "--api-key",
        "<redacted>",
        "--model",
        "m",
    ]
    assert _redact_argv(["--api-key=secret"]) == ["--api-key=<redacted>"]


def test_agent_from_args_passes_llm_temperature():
    agent = agent_from_args(
        "llm",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        llm_temperature=0.25,
    )

    assert isinstance(agent, OpenAICompatAgent)
    assert agent.temperature == 0.25


def test_agent_from_args_passes_strict_fallback_mode_by_default():
    agent = agent_from_args(
        "llm",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
    )

    assert isinstance(agent, OpenAICompatAgent)
    assert agent.fallback_mode == "error"


def test_load_benchmark_config_reads_nested_context_management(tmp_path, monkeypatch):
    pytest.importorskip("omegaconf")
    for key in (
        "STS2_BENCH_LLM_TEMPERATURE",
        "STS2_BENCH_CONTEXT_MODE",
        "STS2_BENCH_TURN_CHAT_WINDOW",
        "STS2_BENCH_TURN_CHAT_UPDATE_MODE",
        "STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY",
        "STS2_BENCH_TURN_CHAT_PLAN_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        """
llm_temperature: 0.2
context_management:
  mode: turn_chat
  turn_chat:
    window: 3
    update_mode: delta
    assistant_history: compact
    plan:
      enabled: true
""",
        encoding="utf-8",
    )

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_LLM_TEMPERATURE"] == "0.2"
    assert os.environ["STS2_BENCH_CONTEXT_MODE"] == "turn_chat"
    assert os.environ["STS2_BENCH_TURN_CHAT_WINDOW"] == "3"
    assert os.environ["STS2_BENCH_TURN_CHAT_UPDATE_MODE"] == "delta"
    assert os.environ["STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY"] == "compact"
    assert os.environ["STS2_BENCH_TURN_CHAT_PLAN_ENABLED"] == "true"
