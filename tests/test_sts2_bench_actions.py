import json
import os

import pytest

from sts2_bench.actions import build_legal_actions
from sts2_bench.agents import OpenAICompatAgent
from sts2_bench.experience import ExperienceOpenAICompatAgent
from sts2_bench.run_benchmark import _redact_argv, load_benchmark_config
from sts2_bench.runner import agent_from_args, run_one


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
    assert "printed attack intent and hand damage/block are current engine values" in prompt
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
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="turn_chat",
        memory_enabled=False,
        max_retries=0,
        **kwargs,
    )
    sent_messages = []
    response_iter = iter(responses)

    def fake_chat(messages):
        sent_messages.append([dict(message) for message in messages])
        return {"choices": [{"message": {"content": json.dumps(next(response_iter))}}]}

    agent._chat = fake_chat
    return agent, sent_messages


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


def _write_experience_store(path):
    path.write_text(
        json.dumps(
            {
                "id": "exp_jaw_worm_0001",
                "status": "active",
                "memory_type": "risk",
                "scope": {
                    "level": "encounter",
                    "character": "Ironclad",
                    "ascension_min": 0,
                    "ascension_max": 20,
                    "act": 1,
                    "enemy": "Jaw Worm",
                },
                "trigger": "Jaw Worm combat in Act 1",
                "lesson": "Jaw Worm punishes slow opening turns.",
                "recommendation": "Balance early damage with enough block when Jaw Worm is attacking.",
                "anti_pattern": "Ignoring incoming damage while only playing low-impact attacks.",
                "quality": {"confidence": 0.7},
                "retrieval_hints": {"keywords": ["jaw_worm", "combat_play", "act1"], "priority": 0.8},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_experience_agent_injects_retrieved_memory_and_records_meta(tmp_path):
    store = tmp_path / "active.jsonl"
    _write_experience_store(store)
    state = _combat_turn_state()
    agent = ExperienceOpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        max_retries=0,
        experience_memory_store_path=str(store),
    )
    agent.start_episode(seed="bench_0000", character="Ironclad", ascension=0)

    def fake_chat(_messages):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "action_id": 5,
                                "reason": "Strike while tracking incoming damage.",
                                "used_memory_ids": ["exp_jaw_worm_0001"],
                            }
                        )
                    }
                }
            ]
        }

    agent._chat = fake_chat
    prompt = agent.build_prompt(state, build_legal_actions(state))
    action, meta = agent.choose(state, build_legal_actions(state), prompt=prompt)

    assert "Experience notes, advisory only" in prompt
    assert "exp_jaw_worm_0001" in prompt
    assert action.kind == "combat_play_card"
    assert meta["experience_memory"]["retrieved_ids"] == ["exp_jaw_worm_0001"]
    assert meta["experience_memory"]["used_memory_ids"] == ["exp_jaw_worm_0001"]


def test_experience_retriever_global_scope_respects_act_and_phase(tmp_path):
    store = tmp_path / "active.jsonl"
    store.write_text(
        json.dumps(
            {
                "id": "exp_global_combat_act1",
                "status": "active",
                "memory_type": "rule",
                "scope": {
                    "level": "global",
                    "character": "Ironclad",
                    "act": 1,
                    "phase": "combat_play",
                },
                "trigger": "Safe combat scaling turn.",
                "lesson": "Play scaling on safe combat turns.",
                "recommendation": "Use combat scaling only during combat decisions.",
                "quality": {"confidence": 0.9},
                "retrieval_hints": {"keywords": ["combat_play", "scaling", "strike"], "priority": 0.8},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    agent = ExperienceOpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        max_retries=0,
        experience_memory_store_path=str(store),
    )
    agent.start_episode(seed="bench_0000", character="Ironclad", ascension=0)

    act1_combat = _combat_turn_state()
    act1_map = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 0, "deck_size": 10},
        "choices": [{"col": 1, "row": 3, "type": "Monster"}],
    }
    act2_combat = _combat_turn_state(context={"act": 2, "floor": 2, "room_type": "Monster"})

    assert "exp_global_combat_act1" in agent.build_prompt(act1_combat, build_legal_actions(act1_combat))
    assert "exp_global_combat_act1" not in agent.build_prompt(act1_map, build_legal_actions(act1_map))
    assert "exp_global_combat_act1" not in agent.build_prompt(act2_combat, build_legal_actions(act2_combat))


def test_experience_agent_writes_compact_event_trace(tmp_path):
    store = tmp_path / "active.jsonl"
    trace = tmp_path / "events.jsonl"
    _write_experience_store(store)
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        energy=2,
        enemies=[{"index": 0, "name": "Jaw Worm", "hp": 14, "max_hp": 40, "block": 0}],
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": ["energy: 3/3 -> 2/3", "enemy Jaw Worm[0]: hp 20/40 -> 14/40"],
        },
    )
    agent = ExperienceOpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        experience_memory_store_path=str(store),
        experience_event_trace_path=str(trace),
        max_retries=0,
    )
    agent.start_episode(seed="bench_0000", character="Ironclad", ascension=0)
    action = next(action for action in build_legal_actions(state) if action.kind == "combat_play_card")
    meta = {
        "parsed": {"reason": "Strike while tracking incoming damage.", "used_memory_ids": ["exp_jaw_worm_0001"]},
        "experience_memory": {
            "retrieved_ids": ["exp_jaw_worm_0001"],
            "used_memory_ids": ["exp_jaw_worm_0001"],
        },
    }

    agent.record_transition(state, action, meta, next_state)

    event = json.loads(trace.read_text(encoding="utf-8").strip())
    assert event["seed"] == "bench_0000"
    assert event["action"]["kind"] == "combat_play_card"
    assert event["outcome"]["energy_delta"] == -1
    assert event["outcome"]["enemy_hp_delta"] == -6
    assert event["agent"]["retrieved_memory_ids"] == ["exp_jaw_worm_0001"]
    assert "hand" not in event


def test_experience_event_trace_omits_enemy_hp_delta_for_non_combat_followups(tmp_path):
    store = tmp_path / "active.jsonl"
    trace = tmp_path / "events.jsonl"
    _write_experience_store(store)
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        decision="card_select",
        enemies=[],
        cards=[{"index": 0, "name": "Defend"}],
        last_action_result={
            "action_label": "play card 0: Armaments",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "card_select",
            "changes": ["energy: 3/3 -> 2/3", "hand_count: 5 -> 4"],
        },
    )
    agent = ExperienceOpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        experience_memory_store_path=str(store),
        experience_event_trace_path=str(trace),
        max_retries=0,
    )
    agent.start_episode(seed="bench_0000", character="Ironclad", ascension=0)
    action = next(action for action in build_legal_actions(state) if action.kind == "combat_play_card")
    meta = {"parsed": {"reason": "Upgrade a card."}, "experience_memory": {"retrieved_ids": [], "used_memory_ids": []}}

    agent.record_transition(state, action, meta, next_state)

    event = json.loads(trace.read_text(encoding="utf-8").strip())
    assert event["outcome"]["decision_after"] == "card_select"
    assert event["outcome"]["enemy_hp_delta"] is None


def test_experience_agent_auto_learns_at_act_end_and_refreshes_retriever(tmp_path):
    store = tmp_path / "active.jsonl"
    trace = tmp_path / "events.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        context={"act": 2, "floor": 1, "room_type": "Map"},
        enemies=[],
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "map_select",
            "changes": ["act: 1 -> 2", "enemy Jaw Worm[0]: hp 20/40 -> 0/40"],
        },
    )
    agent = ExperienceOpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        experience_memory_store_path=str(store),
        experience_event_trace_path=str(trace),
        experience_auto_learn_enabled=True,
        experience_auto_learn_candidate_path=str(candidates),
        max_retries=0,
    )
    agent.start_episode(seed="bench_0000", character="Ironclad", ascension=0)

    def fake_chat(_messages):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "candidates": [
                                    {
                                        "id": "exp_auto_act1",
                                        "memory_type": "risk",
                                        "scope": {"level": "encounter"},
                                        "trigger": "Jaw Worm act-end reflected lesson.",
                                        "lesson": "Trust engine preview values when checking lethal.",
                                        "recommendation": "Use displayed damage/block values directly.",
                                        "quality": {"confidence": 0.8},
                                        "retrieval_hints": {
                                            "keywords": ["jaw_worm", "combat_play", "preview_damage"],
                                            "priority": 0.9,
                                        },
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }

    agent._chat = fake_chat
    action = next(action for action in build_legal_actions(state) if action.kind == "combat_play_card")
    meta = {"parsed": {"reason": "End the act."}, "experience_memory": {"retrieved_ids": [], "used_memory_ids": []}}

    agent.record_transition(state, action, meta, next_state)

    assert json.loads(store.read_text(encoding="utf-8"))["id"] == "exp_auto_act1"
    assert json.loads(candidates.read_text(encoding="utf-8"))["status"] == "candidate"
    records = agent.pop_auto_learn_records()
    assert records[0]["status"] == "ok"
    assert records[0]["act"] == 1
    prompt = agent.build_prompt(_combat_turn_state(), build_legal_actions(_combat_turn_state()))
    assert "exp_auto_act1" in prompt


def test_experience_agent_auto_learns_on_death_when_configured(tmp_path):
    store = tmp_path / "active.jsonl"
    trace = tmp_path / "events.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        decision="game_over",
        player={"hp": 0, "max_hp": 80, "block": 0, "gold": 0, "deck_size": 10},
        hand=[],
        enemies=[],
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "game_over",
            "changes": ["hp: 6/80 -> 0/80", "enemy Jaw Worm[0]: hp 20/40 -> 14/40"],
        },
    )
    agent = ExperienceOpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        experience_memory_store_path=str(store),
        experience_event_trace_path=str(trace),
        experience_auto_learn_enabled=True,
        experience_auto_learn_trigger="act_end_or_death",
        experience_auto_learn_candidate_path=str(candidates),
        max_retries=0,
    )
    agent.start_episode(seed="bench_0000", character="Ironclad", ascension=0)

    def fake_chat(_messages):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "candidates": [
                                    {
                                        "id": "exp_auto_death_act1",
                                        "memory_type": "risk",
                                        "scope": {"level": "encounter"},
                                        "trigger": "Jaw Worm death reflected lesson.",
                                        "lesson": "Do not spend the turn on damage if shown incoming total is lethal.",
                                        "recommendation": "Use shown_total and shown_block directly before choosing an attack.",
                                        "quality": {"confidence": 0.82},
                                        "retrieval_hints": {
                                            "keywords": ["jaw_worm", "death", "shown_total"],
                                            "priority": 0.9,
                                        },
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }

    agent._chat = fake_chat
    action = next(action for action in build_legal_actions(state) if action.kind == "combat_play_card")
    meta = {"parsed": {"reason": "Failed lethal check."}, "experience_memory": {"retrieved_ids": [], "used_memory_ids": []}}

    agent.record_transition(state, action, meta, next_state)

    assert json.loads(candidates.read_text(encoding="utf-8"))["id"] == "exp_auto_death_act1"
    assert not store.exists()
    records = agent.pop_auto_learn_records()
    assert records[0]["status"] == "ok"
    assert records[0]["trigger"] == "act_end_or_death"
    assert records[0]["cause"] == "death"
    assert records[0]["act"] == 1
    assert records[0]["promote"] is False
    assert records[0]["summary"]["promotion"] is None


def test_agent_from_args_can_create_experience_agent(tmp_path):
    store = tmp_path / "active.jsonl"
    _write_experience_store(store)

    agent = agent_from_args(
        "llm",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        experience_memory_enabled=True,
        experience_memory_store_path=str(store),
        experience_auto_learn_enabled=True,
        experience_auto_learn_candidate_path=str(tmp_path / "candidates.jsonl"),
    )

    assert isinstance(agent, ExperienceOpenAICompatAgent)
    assert agent.experience_auto_learn_enabled is True
    assert agent.experience_auto_learn_promote_on_death is False


def test_load_benchmark_config_reads_nested_context_management(tmp_path, monkeypatch):
    pytest.importorskip("omegaconf")
    for key in (
        "STS2_BENCH_LLM_TEMPERATURE",
        "STS2_BENCH_CONTEXT_MODE",
        "STS2_BENCH_TURN_CHAT_WINDOW",
        "STS2_BENCH_TURN_CHAT_UPDATE_MODE",
        "STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY",
        "STS2_BENCH_TURN_CHAT_PLAN_ENABLED",
        "STS2_BENCH_EXPERIENCE_MEMORY_ENABLED",
        "STS2_BENCH_EXPERIENCE_MEMORY_STORE_PATH",
        "STS2_BENCH_EXPERIENCE_MEMORY_TOP_K",
        "STS2_BENCH_EXPERIENCE_EVENT_TRACE_PATH",
        "STS2_BENCH_EXPERIENCE_AUTO_LEARN_ENABLED",
        "STS2_BENCH_EXPERIENCE_AUTO_LEARN_TRIGGER",
        "STS2_BENCH_EXPERIENCE_AUTO_LEARN_CANDIDATE_PATH",
        "STS2_BENCH_EXPERIENCE_AUTO_LEARN_MAX_CANDIDATES_PER_SPAN",
        "STS2_BENCH_EXPERIENCE_AUTO_LEARN_PROMOTE_ON_DEATH",
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
experience_memory:
  enabled: true
  store_path: sts2_bench/experience/memory/active.jsonl
  top_k: 3
  event_trace_path: logs/experience_events.jsonl
  auto_learn:
    enabled: true
    trigger: act_end
    candidate_path: sts2_bench/experience/memory/candidates.jsonl
    max_candidates_per_span: 2
    promote_on_death: true
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
    assert os.environ["STS2_BENCH_EXPERIENCE_MEMORY_ENABLED"] == "true"
    assert os.environ["STS2_BENCH_EXPERIENCE_MEMORY_STORE_PATH"] == "sts2_bench/experience/memory/active.jsonl"
    assert os.environ["STS2_BENCH_EXPERIENCE_MEMORY_TOP_K"] == "3"
    assert os.environ["STS2_BENCH_EXPERIENCE_EVENT_TRACE_PATH"] == "logs/experience_events.jsonl"
    assert os.environ["STS2_BENCH_EXPERIENCE_AUTO_LEARN_ENABLED"] == "true"
    assert os.environ["STS2_BENCH_EXPERIENCE_AUTO_LEARN_TRIGGER"] == "act_end"
    assert os.environ["STS2_BENCH_EXPERIENCE_AUTO_LEARN_CANDIDATE_PATH"] == "sts2_bench/experience/memory/candidates.jsonl"
    assert os.environ["STS2_BENCH_EXPERIENCE_AUTO_LEARN_MAX_CANDIDATES_PER_SPAN"] == "2"
    assert os.environ["STS2_BENCH_EXPERIENCE_AUTO_LEARN_PROMOTE_ON_DEATH"] == "true"
