import json
from pathlib import Path

from sts2_bench.actions import build_legal_actions
from sts2_bench.card_choice_trace_eval import (
    extract_trace_card_choice_scenarios,
    run_card_choice_trace_replay,
)


class _PickOriginalAgent:
    def build_prompt(self, state, legal_actions):
        return f"trace_replay={state.get('trace_replay_id')}"

    def choose(self, state, legal_actions, *, prompt=None):
        del prompt
        action = next(action for action in legal_actions if action.label == state["trace_original_selected_label"])
        return action, {
            "agent": "pick_original",
            "parsed": {
                "action_id": action.action_id,
                "selected_action_label": action.label,
                "reason": "repeat original trace choice",
                "run_plan_patch": {},
                "run_memory_facts": [],
                "no_context_change_reason": "Trace replay only.",
            },
        }


class _PickSkipAgent:
    def build_prompt(self, state, legal_actions):
        return f"trace_replay={state.get('trace_replay_id')}"

    def choose(self, state, legal_actions, *, prompt=None):
        del prompt
        action = next(action for action in legal_actions if action.kind == "card_reward_skip")
        return action, {
            "agent": "pick_skip",
            "parsed": {
                "action_id": action.action_id,
                "selected_action_label": action.label,
                "reason": "skip in trace replay",
                "run_plan_patch": {},
                "run_memory_facts": [],
                "no_context_change_reason": "Trace replay only.",
            },
        }


def test_extract_trace_card_choice_scenarios_uses_full_state_when_available(tmp_path):
    trace_path = tmp_path / "full.jsonl"
    state = _full_card_reward_state()
    rows = [
        {"type": "decision", "step": 3, "decision": "card_reward", "state": state},
        {
            "type": "action",
            "step": 3,
            "decision": "card_reward",
            "action": {"action_id": 1, "kind": "card_reward", "label": "pick card 1: Shrug It Off"},
        },
    ]
    _write_jsonl(trace_path, rows)

    result = extract_trace_card_choice_scenarios([trace_path])

    assert result.summary["full_scenarios"] == 1
    assert result.summary["degraded_scenarios"] == 0
    scenario = result.scenarios[0]
    assert scenario.state["trace_evidence_complete"] is True
    assert scenario.state["trace_original_selected_label"] == "pick card 1: Shrug It Off"
    assert scenario.state["player"]["deck"][0]["name"] == "Strike"
    assert set(scenario.oracle.acceptable_labels) == {
        action.label for action in build_legal_actions(state) if action.command.get("cmd") != "bench_view"
    }


def test_extract_trace_card_choice_scenarios_skips_degraded_by_default(tmp_path):
    trace_path = tmp_path / "decisions-only.jsonl"
    rows = [
        {
            "type": "decision",
            "step": 4,
            "decision": "card_reward",
            "state": {
                "decision": "card_reward",
                "act": 1,
                "floor": 2,
                "hp": 44,
                "max_hp": 80,
                "deck_size": 12,
                "card_count": 3,
            },
            "legal_actions": [
                {"action_id": 5, "kind": "card_reward", "label": "pick card 0: Twin Strike"},
                {"action_id": 6, "kind": "card_reward_skip", "label": "skip card reward"},
            ],
        },
        {
            "type": "action",
            "step": 4,
            "decision": "card_reward",
            "action": {"action_id": 5, "kind": "card_reward", "label": "pick card 0: Twin Strike"},
        },
    ]
    _write_jsonl(trace_path, rows)

    clean = extract_trace_card_choice_scenarios([trace_path])
    degraded = extract_trace_card_choice_scenarios([trace_path], allow_degraded=True)

    assert clean.scenarios == []
    assert clean.summary["skipped_incomplete"] == 1
    assert degraded.summary["degraded_scenarios"] == 1
    assert degraded.scenarios[0].state["trace_evidence_complete"] is False
    assert degraded.scenarios[0].state["trace_degraded_reason"] == "decision trace lacks full player.deck/cards state"


def test_run_card_choice_trace_replay_reports_repeated_original_and_skip_changes(tmp_path):
    trace_path = tmp_path / "full.jsonl"
    state = _full_card_reward_state()
    rows = [
        {"type": "decision", "step": 3, "decision": "card_reward", "state": state},
        {
            "type": "action",
            "step": 3,
            "decision": "card_reward",
            "action": {"action_id": 1, "kind": "card_reward", "label": "pick card 1: Shrug It Off"},
        },
    ]
    _write_jsonl(trace_path, rows)
    scenarios = extract_trace_card_choice_scenarios([trace_path]).scenarios
    repeat_out = tmp_path / "repeat.jsonl"
    skip_out = tmp_path / "skip.jsonl"

    repeat_summary = run_card_choice_trace_replay(agent=_PickOriginalAgent(), scenarios=scenarios, out=repeat_out)
    skip_summary = run_card_choice_trace_replay(agent=_PickSkipAgent(), scenarios=scenarios, out=skip_out)

    assert repeat_summary["repeated_original_count"] == 1
    assert repeat_summary["changed_from_original_count"] == 0
    assert skip_summary["repeated_original_count"] == 0
    assert skip_summary["changed_from_original_count"] == 1
    assert skip_summary["skip_selected_count"] == 1
    rows = [json.loads(line) for line in skip_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    choice = next(row for row in rows if row["type"] == "trace_choice")
    assert choice["original_selected_label"] == "pick card 1: Shrug It Off"
    assert choice["selected_label"] == "skip card reward"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _full_card_reward_state() -> dict:
    return {
        "decision": "card_reward",
        "act": 1,
        "floor": 2,
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {
            "hp": 50,
            "max_hp": 80,
            "gold": 99,
            "deck_size": 2,
            "deck": [
                {"name": "Strike", "type": "Attack", "cost": 1, "description": "Deal 6 damage."},
                {"name": "Defend", "type": "Skill", "cost": 1, "description": "Gain 5 Block."},
            ],
        },
        "cards": [
            {"index": 0, "name": "Twin Strike", "type": "Attack", "cost": 1},
            {"index": 1, "name": "Shrug It Off", "type": "Skill", "cost": 1},
        ],
        "can_skip": True,
    }
