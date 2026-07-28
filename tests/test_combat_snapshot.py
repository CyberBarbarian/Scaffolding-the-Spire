import json
from pathlib import Path

import pytest
import sts2_bench.combat_snapshot as combat_snapshot_module

from sts2_bench.combat_snapshot import (
    SnapshotReplayError,
    create_save_backed_combat_snapshot,
    extract_combat_snapshot,
    replay_snapshot_prefix,
    run_combat_snapshot,
)
from sts2_bench.runner import JsonlLogger


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


def test_extract_combat_snapshot_summarizes_authoritative_target_state(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {"type": "start", "seed": "bench_0000", "character": "Ironclad", "ascension": 0},
            {
                "type": "decision",
                "step": 3,
                "decision": "combat_play",
                "state": {"type": "decision", **_combat_state()},
                "authoritative_state": {
                    **_combat_state(),
                    "last_action_result": _combat_summary()["last_action_result"],
                },
            },
        ],
    )

    snapshot = extract_combat_snapshot(trace_path, start_step=3)

    assert snapshot["target"]["state_summary"] == _combat_summary()


def test_extract_combat_snapshot_preserves_setup_action_prefix(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    setup_command = {"cmd": "action", "action": "choose_option", "args": {"option_index": 1}}
    _write_jsonl(
        trace_path,
        [
            {"type": "start", "seed": "bench_0000", "character": "Ironclad", "ascension": 0},
            {
                "type": "setup_action",
                "setup_index": 0,
                "command": setup_command,
                "state": _event_summary(),
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

    snapshot = extract_combat_snapshot(trace_path, start_step=3)

    assert snapshot["prefix_actions"][0] == {
        "source": "setup_action",
        "setup_index": 0,
        "command": setup_command,
    }
    assert snapshot["prefix_actions"][1]["kind"] == "event"


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
    assert snapshot["target"]["last_action_result"] == {
        "action_label": "custom sequence: 0 on enemy 0 -> 1 -> 2 on enemy 0",
        "action_kind": "combat_card_sequence",
        "decision_before": "combat_play",
        "decision_after": "combat_play",
        "changes": [],
    }


def test_replay_snapshot_prefix_rejects_last_action_result_drift():
    snapshot = _snapshot()
    snapshot["target"]["last_action_result"] = {
        "action_label": "choose option 0",
        "action_kind": "event",
        "decision_before": "event_choice",
        "decision_after": "combat_play",
        "changes": ["floor: 1 -> 2"],
    }
    process = _PrefixProcess()

    with pytest.raises(SnapshotReplayError, match="last_action_result mismatch"):
        replay_snapshot_prefix(snapshot, process=process)


def test_replay_snapshot_prefix_matches_actions_by_kind_and_label():
    snapshot = _snapshot()
    process = _PrefixProcess()

    replay = replay_snapshot_prefix(snapshot, process=process)

    assert replay.state["decision"] == "combat_play"
    assert process.commands == [{"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}]
    assert replay.action_tape == process.commands


def test_replay_snapshot_prefix_replays_raw_setup_commands_before_saved_actions():
    snapshot = _snapshot()
    setup_command = {"cmd": "action", "action": "choose_option", "args": {"option_index": 1}}
    snapshot["prefix_actions"] = [
        {"source": "setup_action", "setup_index": 0, "command": setup_command},
        *snapshot["prefix_actions"],
    ]
    process = _RawSetupPrefixProcess()

    replay = replay_snapshot_prefix(snapshot, process=process)

    assert replay.state["decision"] == "combat_play"
    assert process.commands == [
        setup_command,
        {"cmd": "action", "action": "choose_option", "args": {"option_index": 0}},
    ]
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


@pytest.mark.parametrize("failure_stage", ["start", "start_run", "send"])
def test_replay_snapshot_prefix_closes_owned_process_on_every_replay_failure(failure_stage):
    process = _FailingOwnedPrefixProcess(failure_stage)

    with pytest.raises(RuntimeError, match=failure_stage):
        replay_snapshot_prefix(_snapshot(), process_factory=lambda: process)

    assert process.closed


def test_replay_snapshot_prefix_does_not_close_borrowed_process_on_failure():
    process = _FailingOwnedPrefixProcess("start_run")

    with pytest.raises(RuntimeError, match="start_run"):
        replay_snapshot_prefix(_snapshot(), process=process)

    assert not process.closed


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


def test_run_combat_snapshot_zero_max_steps_runs_until_fight_ends():
    process = _CombatCompletesProcess()

    result = run_combat_snapshot(
        snapshot=_snapshot(),
        agent=_EndTurnAgent(),
        process=process,
        max_steps=0,
    )

    assert result.error is None
    assert result.truncated is False
    assert result.decision == "combat_reward"
    assert result.steps == 1
    assert result.extra["snapshot_outcome"] == "combat_complete"


def test_run_combat_snapshot_refuses_implicit_proceed_when_action_surface_is_empty(monkeypatch, tmp_path):
    process = _CombatCompletesProcess()
    original = combat_snapshot_module.build_legal_actions

    def empty_at_target(state, *args, **kwargs):
        if state.get("decision") == "combat_play":
            return []
        return original(state, *args, **kwargs)

    monkeypatch.setattr(combat_snapshot_module, "build_legal_actions", empty_at_target)
    trace_path = tmp_path / "trace.jsonl"
    logger = JsonlLogger(trace_path)
    try:
        result = run_combat_snapshot(
            snapshot=_snapshot(),
            agent=_EndTurnAgent(),
            process=process,
            logger=logger,
            log_level="full",
            max_steps=5,
        )
    finally:
        logger.close()

    assert "refusing implicit proceed" in result.error
    assert result.invalid_states == 1
    assert process.commands == [{"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}]
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(row.get("type") == "decision_error" and row.get("report_required") for row in rows)


def test_run_combat_snapshot_hides_precomputed_sequences_by_default_but_keeps_custom_gateway():
    process = _SequenceCombatCompletesProcess()
    agent = _CaptureKindsAgent()

    result = run_combat_snapshot(
        snapshot=_sequence_snapshot(),
        agent=agent,
        process=process,
        max_steps=5,
    )

    assert result.error is None
    assert "combat_card_sequence" not in agent.kinds
    assert "combat_custom_sequence" in agent.kinds


def test_run_combat_snapshot_can_expose_precomputed_sequences_for_diagnostics():
    process = _SequenceCombatCompletesProcess()
    agent = _CaptureKindsAgent()

    result = run_combat_snapshot(
        snapshot=_sequence_snapshot(),
        agent=agent,
        process=process,
        max_steps=5,
        precomputed_combat_card_sequences_enabled=True,
    )

    assert result.error is None
    assert "combat_card_sequence" in agent.kinds
    assert "combat_custom_sequence" in agent.kinds


def test_run_combat_snapshot_logs_custom_sequence_validation_feedback(tmp_path):
    process = _SequenceCombatCompletesProcess()
    agent = _InvalidCustomSequenceThenEndTurnAgent()
    out = tmp_path / "snapshot.jsonl"
    logger = JsonlLogger(out)

    try:
        result = run_combat_snapshot(
            snapshot=_sequence_snapshot(),
            agent=agent,
            process=process,
            logger=logger,
            max_steps=5,
        )
    finally:
        logger.close()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.error is None
    assert result.extra["custom_sequence_validation_retry_count"] == 1
    assert any(row.get("type") == "custom_sequence_validation_feedback" for row in rows)
    action_rows = [row for row in rows if row.get("type") == "action"]
    assert len(action_rows) == 1
    assert action_rows[0]["action"]["kind"] == "combat_end_turn"
    assert action_rows[0]["snapshot_id"] == "sequence_unit_snapshot"


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


def _sequence_snapshot() -> dict:
    snapshot = _snapshot()
    snapshot["snapshot_id"] = "sequence_unit_snapshot"
    snapshot["target"]["state_summary"] = {
        **_combat_summary(),
        "hand_count": 2,
    }
    return snapshot


def _event_state() -> dict:
    return {
        "decision": "event_choice",
        "context": {"act": 1, "floor": 12, "room_type": "Event"},
        "player": {"hp": 51, "max_hp": 82, "gold": 129, "deck_size": 17},
        "options": [{"index": 0, "title": "Fight", "is_locked": False}],
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


def _sequence_combat_state() -> dict:
    return {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 12, "room_type": "Event"},
        "player": {"hp": 51, "max_hp": 82, "gold": 129, "deck_size": 17},
        "energy": 2,
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
                "name": "Defend",
                "cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
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


class _FailingOwnedPrefixProcess(_PrefixProcess):
    def __init__(self, failure_stage):
        super().__init__()
        self.failure_stage = failure_stage
        self.closed = False

    def start(self):
        if self.failure_stage == "start":
            raise RuntimeError("start")

    def start_run(self, **_kwargs):
        if self.failure_stage == "start_run":
            raise RuntimeError("start_run")
        return _event_state()

    def send(self, command):
        if self.failure_stage == "send":
            raise RuntimeError("send")
        return super().send(command)

    def close(self):
        self.closed = True


class _CombatCompletesProcess(_PrefixProcess):
    def send(self, command):
        self.commands.append(command)
        if command == {"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}:
            return _combat_state()
        if command == {"cmd": "action", "action": "end_turn"}:
            return _reward_state()
        raise AssertionError(f"unexpected command: {command!r}")


class _RawSetupPrefixProcess(_PrefixProcess):
    def __init__(self):
        super().__init__()
        self.saw_setup = False

    def send(self, command):
        self.commands.append(command)
        if command == {"cmd": "action", "action": "choose_option", "args": {"option_index": 1}}:
            self.saw_setup = True
            return _event_state()
        if self.saw_setup and command == {"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}:
            return _combat_state()
        raise AssertionError(f"unexpected command: {command!r}")


class _SequenceCombatCompletesProcess(_PrefixProcess):
    def send(self, command):
        self.commands.append(command)
        if command == {"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}:
            return _sequence_combat_state()
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


class _CaptureKindsAgent(_EndTurnAgent):
    def __init__(self):
        self.kinds = []

    def choose(self, state, legal_actions, *, prompt=None):
        self.kinds = [action.kind for action in legal_actions]
        return super().choose(state, legal_actions, prompt=prompt)


class _InvalidCustomSequenceThenEndTurnAgent(_EndTurnAgent):
    def __init__(self):
        self.calls = 0

    def choose(self, state, legal_actions, *, prompt=None):
        self.calls += 1
        if self.calls == 1:
            action = next(action for action in legal_actions if action.kind == "combat_custom_sequence")
            return action, {
                "agent": "test",
                "parsed": {
                    "action_id": action.action_id,
                    "selected_action_label": action.label,
                    "sequence": [
                        {"card_index": 0, "target_index": 0},
                        {"card_index": 1},
                        {"card_index": 0, "target_index": 0},
                        {"card_index": 1},
                        {"card_index": 0, "target_index": 0},
                    ],
                    "reason": "Invalid overlong custom sequence.",
                },
            }
        return super().choose(state, legal_actions, prompt=prompt)


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
