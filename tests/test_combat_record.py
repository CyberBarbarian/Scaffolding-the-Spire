import json

from sts2_bench.combat_record import CombatRecorder
from sts2_bench.runner import run_one


def test_run_one_writes_per_combat_record_with_commands_and_hp_loss(tmp_path):
    recorder = CombatRecorder(
        root=tmp_path / "combat_records",
        run_id="unit_run",
        condition="max_rest_cot",
        seed="bench_0000",
        character="Ironclad",
        ascension=0,
    )
    process = _OneCombatProcess()

    result = run_one(
        agent=_EndTurnAgent(),
        seed="bench_0000",
        character="Ironclad",
        ascension=0,
        max_steps=1,
        process=process,
        route_plan_enabled=False,
        combat_recorder=recorder,
    )

    assert result.error is None

    index_path = tmp_path / "combat_records" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index["combats"]) == 1
    row = index["combats"][0]
    assert row["combat_id"] == "unit_run_combat_001_A1F7_Monster"
    assert row["condition"] == "max_rest_cot"
    assert row["entry_hp"] == 45
    assert row["exit_hp"] == 33
    assert row["hp_loss"] == 12
    assert row["terminal_death"] is False

    combat_dir = tmp_path / "combat_records" / "unit_run_combat_001_A1F7_Monster"
    snapshot = json.loads((combat_dir / "entry_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["entry"]["step"] == 0
    assert snapshot["entry"]["decision"] == "combat_play"
    assert snapshot["entry"]["prefix_commands"] == []
    assert snapshot["save"]["success"] is True

    room_rows = _read_jsonl(combat_dir / "room.jsonl")
    assert [row["type"] for row in room_rows] == ["combat_entry", "combat_action", "combat_exit"]
    assert room_rows[1]["action"]["kind"] == "combat_end_turn"
    assert room_rows[1]["command"] == {"cmd": "action", "action": "end_turn"}
    assert room_rows[2]["exit_hp"] == 33

    command_rows = _read_jsonl(combat_dir / "commands.jsonl")
    assert command_rows == [
        {
            "type": "command",
            "step": 0,
            "decision": "combat_play",
            "command": {"cmd": "action", "action": "end_turn"},
        }
    ]


def test_rest_fullgame_worker_command_passes_combat_record_options(tmp_path):
    from sts2_bench.rest_fullgame_experiment import CONDITIONS, _worker_command, build_parser

    args = build_parser().parse_args(
        [
            "--out-dir",
            str(tmp_path / "run"),
            "--combat-record-dir",
            str(tmp_path / "records"),
            "--no-combat-records",
        ]
    )

    cmd = _worker_command(
        args=args,
        condition=CONDITIONS[0],
        seed="bench_0000",
        out_dir=tmp_path / "run",
        trace_path=tmp_path / "run" / "traces" / "trace.jsonl",
        analysis_dir=tmp_path / "run" / "analysis" / "case",
        result_path=tmp_path / "run" / "worker_results" / "case.json",
    )

    assert "--combat-record-dir" in cmd
    assert cmd[cmd.index("--combat-record-dir") + 1] == str(tmp_path / "records")
    assert "--no-combat-records" in cmd
    assert "--combat-records" not in cmd


class _EndTurnAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        del state, prompt
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "test", "parsed": {"reason": "End the turn."}}


class _OneCombatProcess:
    def __init__(self):
        self.commands = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        del kwargs
        return _combat_state(hp=45)

    def send(self, command):
        self.commands.append(command)
        if command.get("cmd") == "write_continue_save":
            path = command["path"]
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("save")
            return {"type": "save_result", "success": True, "path": path}
        if command == {"cmd": "action", "action": "end_turn"}:
            return {
                "decision": "combat_reward",
                "context": {"act": 1, "floor": 7, "room_type": "Monster"},
                "player": {"hp": 33, "max_hp": 80, "gold": 23, "deck_size": 10},
                "rewards": [],
            }
        raise AssertionError(f"unexpected command: {command!r}")

    def close(self):
        pass


def _combat_state(*, hp):
    return {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 7, "room_type": "Monster"},
        "player": {"hp": hp, "max_hp": 80, "gold": 23, "deck_size": 10},
        "energy": 3,
        "hand": [],
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 40, "max_hp": 40, "intent": "Attack"}],
    }


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
