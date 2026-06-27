import json
from pathlib import Path

from sts2_bench.combat_snapshot import (
    create_save_backed_combat_snapshot,
    extract_combat_snapshot,
    replay_snapshot_prefix,
    run_combat_snapshot,
)


def test_extract_combat_snapshot_records_prefix_and_entry(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "type": "run_config",
                "config": {
                    "game": {"character": "Ironclad", "ascension": 0, "lang": "en"},
                    "benchmark": {"seeds": ["bench_0000"]},
                },
            },
            {
                "type": "start",
                "seed": "bench_0000",
                "character": "Ironclad",
                "ascension": 0,
            },
            {"type": "decision", "step": 2, "decision": "event_choice", "state": _event_summary()},
            {
                "type": "action",
                "step": 2,
                "decision": "event_choice",
                "action": {"action_id": 5, "kind": "event", "label": "choose option 0: Fight"},
            },
            {
                "type": "decision",
                "step": 3,
                "decision": "combat_play",
                "state": _combat_summary(),
                "enemy_snapshot": [{"index": 0, "name": "Punch Construct", "hp": 33, "max_hp": 55}],
            },
        ],
    )

    snapshot = extract_combat_snapshot(trace_path, start_step=3, snapshot_id="unit_snapshot")

    assert snapshot["snapshot_id"] == "unit_snapshot"
    assert snapshot["game"] == {"seed": "bench_0000", "character": "Ironclad", "ascension": 0, "lang": "en"}
    assert snapshot["prefix_actions"] == [
        {
            "step": 2,
            "decision": "event_choice",
            "action_id": 5,
            "kind": "event",
            "label": "choose option 0: Fight",
        }
    ]
    assert snapshot["target"]["step"] == 3
    assert snapshot["target"]["decision"] == "combat_play"
    assert snapshot["target"]["state_summary"]["hp"] == 51
    assert snapshot["stop_policy"]["active_decisions"] == ["combat_play", "card_select"]


def test_extract_recovers_old_custom_sequence_from_next_decision_summary(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"type": "start", "seed": "bench_0000", "character": "Ironclad", "ascension": 0},
            {"type": "decision", "step": 2, "decision": "combat_play", "state": _combat_summary()},
            {
                "type": "action",
                "step": 2,
                "decision": "combat_play",
                "action": {
                    "action_id": 9,
                    "kind": "combat_custom_sequence",
                    "label": "propose custom deterministic card sequence",
                },
                "agent_meta": {"parsed": {"reason": "legacy trace omitted sequence payload"}},
            },
            {
                "type": "decision",
                "step": 3,
                "decision": "combat_play",
                "state": {
                    **_combat_summary(),
                    "last_action_result": {
                        "action_label": "custom sequence: 0 on enemy 0 -> 1 -> 2 on enemy 0",
                        "action_kind": "combat_card_sequence",
                        "decision_before": "combat_play",
                        "decision_after": "combat_play",
                        "changes": [],
                    },
                },
            },
        ],
    )

    snapshot = extract_combat_snapshot(trace_path, start_step=3)

    assert snapshot["prefix_actions"] == [
        {
            "step": 2,
            "decision": "combat_play",
            "action_id": 9,
            "kind": "combat_custom_sequence",
            "label": "propose custom deterministic card sequence",
            "sequence": [
                {"card_index": 0, "target_index": 0},
                {"card_index": 1},
                {"card_index": 2, "target_index": 0},
            ],
        }
    ]


def test_replay_snapshot_prefix_matches_actions_by_kind_and_label():
    snapshot = _snapshot()
    process = _PrefixProcess()

    replay = replay_snapshot_prefix(snapshot, process=process)

    assert replay.state["decision"] == "combat_play"
    assert process.commands == [{"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}]
    assert replay.action_tape == process.commands


def test_replay_snapshot_prefix_replays_historical_sequence_from_label_when_mask_changed():
    snapshot = {
        "type": "combat_snapshot",
        "version": 1,
        "snapshot_id": "unsafe_sequence_snapshot",
        "game": {"seed": "bench_0000", "character": "Ironclad", "ascension": 0, "lang": "en"},
        "target": {"step": 3, "decision": "card_select", "state_summary": {}},
        "stop_policy": {"active_decisions": ["combat_play", "card_select"]},
        "prefix_actions": [
            {
                "step": 2,
                "decision": "combat_play",
                "action_id": 33,
                "kind": "combat_card_sequence",
                "label": "play sequence: 0 Strike on enemy 0 -> 1 Armaments",
            }
        ],
    }
    process = _UnsafeSequencePrefixProcess()

    replay = replay_snapshot_prefix(snapshot, process=process)

    assert replay.state["decision"] == "card_select"
    assert process.commands == [
        {"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 0}},
        {"cmd": "action", "action": "play_card", "args": {"card_index": 1}},
    ]


def test_run_combat_snapshot_stops_after_the_fight():
    process = _CombatCompletesProcess()

    result = run_combat_snapshot(
        snapshot=_snapshot(),
        agent=_EndTurnAgent(),
        process=process,
        max_steps=5,
    )

    assert result.error is None
    assert result.truncated is False
    assert result.decision == "combat_reward"
    assert result.steps == 1
    assert result.game_action_count == 1
    assert result.extra["snapshot_outcome"] == "combat_complete"
    assert result.extra["snapshot_entry_step"] == 3


def test_save_backed_snapshot_loads_save_and_replays_entry_commands(tmp_path):
    process = _SaveBackedProcess()
    save_path = tmp_path / "entry.save"
    action_tape = [
        {"cmd": "action", "action": "claim_reward", "args": {"reward_index": 0}},
        {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 2}},
    ]

    snapshot = create_save_backed_combat_snapshot(
        process=process,
        state=_combat_state(),
        action_tape=action_tape,
        save_path=save_path,
        snapshot_id="save_snapshot",
        game={"seed": "bench_0000", "character": "Ironclad", "ascension": 0, "lang": "en"},
        step=7,
    )

    replay = replay_snapshot_prefix(snapshot, process=_SaveBackedProcess())

    assert snapshot["load"]["type"] == "save"
    assert snapshot["load"]["entry_commands"] == [
        {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 2}}
    ]
    assert replay.state["decision"] == "combat_play"
    assert replay.action_tape == snapshot["load"]["entry_commands"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _snapshot() -> dict:
    return {
        "type": "combat_snapshot",
        "version": 1,
        "snapshot_id": "unit_snapshot",
        "game": {"seed": "bench_0000", "character": "Ironclad", "ascension": 0, "lang": "en"},
        "target": {
            "step": 3,
            "decision": "combat_play",
            "state_summary": _combat_summary(),
            "enemy_snapshot": [{"index": 0, "name": "Punch Construct", "hp": 33, "max_hp": 55}],
        },
        "stop_policy": {"active_decisions": ["combat_play", "card_select"]},
        "prefix_actions": [
            {
                "step": 2,
                "decision": "event_choice",
                "action_id": 5,
                "kind": "event",
                "label": "choose option 0: Fight",
            }
        ],
    }


def _event_state() -> dict:
    return {
        "decision": "event_choice",
        "context": {"act": 1, "floor": 12, "room_type": "Event"},
        "player": {"hp": 51, "max_hp": 82, "gold": 129, "deck_size": 17},
        "options": [{"index": 0, "title": "Fight"}],
    }


def _event_summary() -> dict:
    return {
        "decision": "event_choice",
        "act": 1,
        "floor": 12,
        "room_type": "Event",
        "hp": 51,
        "max_hp": 82,
        "gold": 129,
        "deck_size": 17,
        "hand_count": 0,
        "enemy_count": 0,
        "choice_count": 1,
        "card_count": 0,
        "reward_count": 0,
        "viewed": [],
    }


def _combat_state() -> dict:
    return {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 12, "room_type": "Event"},
        "player": {"hp": 51, "max_hp": 82, "gold": 129, "deck_size": 17},
        "energy": 0,
        "hand": [],
        "enemies": [{"index": 0, "name": "Punch Construct", "hp": 33, "max_hp": 55}],
    }


def _combat_summary() -> dict:
    return {
        "decision": "combat_play",
        "act": 1,
        "floor": 12,
        "room_type": "Event",
        "hp": 51,
        "max_hp": 82,
        "gold": 129,
        "deck_size": 17,
        "hand_count": 0,
        "enemy_count": 1,
        "choice_count": 0,
        "card_count": 0,
        "reward_count": 0,
        "viewed": [],
        "last_action_result": {
            "action_label": "choose option 0: Fight",
            "action_kind": "event",
            "decision_before": "event_choice",
            "decision_after": "combat_play",
            "changes": [],
        },
    }


def _reward_state() -> dict:
    return {
        "decision": "combat_reward",
        "context": {"act": 1, "floor": 12, "room_type": "Event"},
        "player": {"hp": 49, "max_hp": 82, "gold": 129, "deck_size": 17},
        "rewards": [],
    }


class _PrefixProcess:
    def __init__(self):
        self.commands = []

    def start(self):
        pass

    def start_run(self, **_kwargs):
        return _event_state()

    def send(self, command):
        self.commands.append(command)
        if command == {"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}:
            return _combat_state()
        raise AssertionError(f"unexpected command: {command!r}")

    def close(self):
        pass


class _CombatCompletesProcess(_PrefixProcess):
    def send(self, command):
        self.commands.append(command)
        if command == {"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}:
            return _combat_state()
        if command == {"cmd": "action", "action": "end_turn"}:
            return _reward_state()
        raise AssertionError(f"unexpected command: {command!r}")


class _UnsafeSequencePrefixProcess:
    def __init__(self):
        self.commands = []

    def start(self):
        pass

    def start_run(self, **_kwargs):
        return {
            "decision": "combat_play",
            "context": {"act": 1, "floor": 12, "room_type": "Monster"},
            "player": {"hp": 51, "max_hp": 82},
            "energy": 3,
            "hand": [
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
                    "name": "Armaments",
                    "cost": 1,
                    "type": "Skill",
                    "target_type": "None",
                    "can_play": True,
                    "description": "Gain 5 Block. Upgrade a card in your hand.",
                    "stats": {"block": 5},
                },
            ],
            "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 20, "max_hp": 40}],
        }

    def send(self, command):
        self.commands.append(command)
        if command == {"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 0}}:
            return {
                "decision": "combat_play",
                "context": {"act": 1, "floor": 12, "room_type": "Monster"},
                "player": {"hp": 51, "max_hp": 82},
                "energy": 2,
                "hand": [
                    {
                        "index": 1,
                        "name": "Armaments",
                        "cost": 1,
                        "type": "Skill",
                        "target_type": "None",
                        "can_play": True,
                        "description": "Gain 5 Block. Upgrade a card in your hand.",
                        "stats": {"block": 5},
                    }
                ],
                "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 14, "max_hp": 40}],
            }
        if command == {"cmd": "action", "action": "play_card", "args": {"card_index": 1}}:
            return {
                "decision": "card_select",
                "context": {"act": 1, "floor": 12, "room_type": "Monster"},
                "combat": {"round": 2},
                "player": {"hp": 51, "max_hp": 82},
                "cards": [{"index": 0, "name": "Defend"}],
            }
        raise AssertionError(f"unexpected command: {command!r}")

    def close(self):
        pass


class _EndTurnAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        del state, prompt
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "test", "parsed": {"reason": "End the turn."}}


class _SaveBackedProcess:
    def __init__(self):
        self.commands = []

    def start(self):
        pass

    def send(self, command):
        self.commands.append(command)
        if command.get("cmd") == "write_continue_save":
            return {"type": "save_result", "success": True, "path": command["path"]}
        if command.get("cmd") == "load_save":
            return _event_state()
        if command == {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 2}}:
            return _combat_state()
        raise AssertionError(f"unexpected command: {command!r}")

    def close(self):
        pass
