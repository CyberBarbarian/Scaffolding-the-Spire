from __future__ import annotations

import json

from sts2_bench.direct_cli_report import (
    DIRECT_CLI_ANALYSIS_FILES,
    analyze_direct_cli_run,
    audit_benchmark_trace,
    audit_engine_trace,
    convert_engine_trace,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_direct_cli_report_reuses_benchmark_trace_analysis_and_tracks_deck_changes(tmp_path):
    starter_deck = [
        *[{"id": "CARD.STRIKE_IRONCLAD", "name": "Strike"} for _ in range(5)],
        *[{"id": "CARD.DEFEND_IRONCLAD", "name": "Defend"} for _ in range(4)],
        {"id": "CARD.BASH", "name": "Bash"},
    ]
    upgraded_deck = [*starter_deck, {"id": "CARD.POMMEL_STRIKE", "name": "Pommel Strike"}]
    engine_trace = tmp_path / "engine_trace.jsonl"
    _write_jsonl(
        engine_trace,
        [
            {"step": 1, "type": "state", "data": {"type": "ready"}},
            {"step": 1, "type": "action", "data": {"cmd": "start_run", "seed": "fixed_seed"}},
            {
                "step": 2,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "card_reward",
                    "context": {"act": 1, "floor": 2, "room_type": "Monster"},
                    "player": {"hp": 75, "max_hp": 80, "deck_size": 10, "deck": starter_deck},
                    "cards": [{"name": "Pommel Strike"}, {"name": "Iron Wave"}],
                },
            },
            {
                "step": 2,
                "type": "action",
                "data": {"cmd": "action", "action": "take_card_reward", "args": {"card_index": 0}},
            },
            {
                "step": 3,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "map_select",
                    "context": {"act": 1, "floor": 2, "room_type": "Map"},
                    "player": {"hp": 75, "max_hp": 80, "deck_size": 11, "deck": upgraded_deck},
                    "last_action_result": {
                        "action_kind": "card_reward",
                        "action_label": "take card 0: Pommel Strike",
                        "changes": ["deck: +Pommel Strike"],
                    },
                },
            },
        ],
    )
    config = {
        "protocol": "human_cli_raw_fifo_v1",
        "seed": "fixed_seed",
        "character": "Ironclad",
        "ascension": 0,
    }
    cli_summary = {
        "outcome": "incomplete",
        "decision_limit_reached": True,
        "error": None,
    }

    converted = convert_engine_trace(engine_trace, config=config, cli_summary=cli_summary)
    action = next(row for row in converted if row["type"] == "action")
    assert action["action"]["kind"] == "card_reward"
    assert action["action"]["label"] == "take card 0: Pommel Strike"

    artifacts = analyze_direct_cli_run(
        engine_trace,
        tmp_path / "benchmark_trace.jsonl",
        tmp_path / "trace_analysis",
        config=config,
        cli_summary=cli_summary,
    )

    report = json.loads((tmp_path / "trace_analysis" / "summary.json").read_text(encoding="utf-8"))
    assert report["final_deck"]["size"] == 11
    assert report["final_deck"]["events"][0]["changes"][0]["name"] == "Pommel Strike"
    assert artifacts["deck_event_count"] == 1
    assert set(artifacts["analysis_artifact_manifest"]) == set(DIRECT_CLI_ANALYSIS_FILES)
    assert all(
        len(item["sha256"]) == 64 and item["bytes"] > 0
        for item in artifacts["analysis_artifact_manifest"].values()
    )
    assert (tmp_path / "trace_analysis" / "replay.md").is_file()
    assert (tmp_path / "trace_analysis" / "deck_and_card_rewards_zh.md").is_file()


def test_direct_cli_report_synthesizes_room_and_deck_events_from_raw_engine_states(tmp_path):
    starter_deck = [
        *[{"name": "Strike", "upgraded": False} for _ in range(5)],
        *[{"name": "Defend", "upgraded": False} for _ in range(4)],
        {"name": "Bash", "upgraded": False},
    ]
    rewarded_deck = [*starter_deck, {"name": "Pommel Strike", "upgraded": False}]
    engine_trace = tmp_path / "engine_trace.jsonl"
    _write_jsonl(
        engine_trace,
        [
            {
                "step": 1,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "map_select",
                    "context": {"act": 1, "floor": 1, "room_type": "Map"},
                    "choices": [{"col": 3, "row": 1, "type": "Monster"}],
                    "player": {"hp": 80, "max_hp": 80, "deck_size": 10, "deck": starter_deck},
                },
            },
            {
                "step": 1,
                "type": "action",
                "data": {"cmd": "get_map"},
            },
            {
                "step": 1,
                "type": "state",
                "data": {"type": "map", "map": {"nodes": []}},
            },
            {
                "step": 1,
                "type": "action",
                "data": {"cmd": "action", "action": "select_map_node", "args": {"col": 3, "row": 1}},
            },
            {
                "step": 2,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "card_reward",
                    "context": {"act": 1, "floor": 2, "room_type": "Monster"},
                    "cards": [{"index": 0, "name": "Pommel Strike"}],
                    "can_skip": True,
                    "player": {"hp": 72, "max_hp": 80, "deck_size": 10, "deck": starter_deck},
                },
            },
            {
                "step": 2,
                "type": "action",
                "data": {"cmd": "action", "action": "select_card_reward", "args": {"card_index": 0}},
            },
            {
                "step": 3,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "map_select",
                    "context": {"act": 1, "floor": 2, "room_type": "Map"},
                    "choices": [],
                    "player": {"hp": 72, "max_hp": 80, "deck_size": 11, "deck": rewarded_deck},
                },
            },
        ],
    )
    config = {"seed": "fixed", "character": "Ironclad", "ascension": 0, "protocol": "direct_cli"}
    summary = {"outcome": "incomplete", "decision_limit_reached": True, "error": None}

    artifacts = analyze_direct_cli_run(
        engine_trace,
        tmp_path / "benchmark_trace.jsonl",
        tmp_path / "trace_analysis",
        config=config,
        cli_summary=summary,
    )
    report = json.loads((tmp_path / "trace_analysis" / "summary.json").read_text(encoding="utf-8"))

    assert artifacts["room_count"] == 1
    assert artifacts["deck_event_count"] == 1
    assert report["rooms"][0]["room_type"] == "Monster"
    assert report["map_choices"][0]["selected"]["node"] == {"col": 3, "row": 1, "type": "Monster"}
    assert report["card_rewards"][0]["selected_card"] == "Pommel Strike"
    assert report["final_deck"]["size"] == 11


def test_engine_trace_audit_rejects_intermediate_error_and_unsupported_state(tmp_path):
    terminal = {
        "type": "decision",
        "decision": "game_over",
        "victory": False,
        "player": {
            "deck_size": 1,
            "deck": [{"id": "CARD.STRIKE_IRONCLAD", "name": "Strike"}],
        },
    }
    trace = tmp_path / "engine_trace.jsonl"
    _write_jsonl(
        trace,
        [
            {"step": 1, "type": "state", "data": {"type": "error", "message": "bad action"}},
            {
                "step": 2,
                "type": "state",
                "data": {"type": "decision", "decision": "event_blocked", "player": {}},
            },
            {"step": 3, "type": "state", "data": terminal},
            {"step": 3, "type": "action", "data": {"cmd": "quit"}},
        ],
    )

    audit = audit_engine_trace(
        trace,
        config={"seed": "fixed"},
        cli_summary={
            "completed": True,
            "outcome": "defeat",
            "error": None,
            "decision_limit_reached": False,
        },
    )

    assert audit["valid"] is False
    assert audit["checks"]["no_engine_error_states"] is False
    assert audit["checks"]["no_unsupported_decisions"] is False


def test_behavior_termination_preserves_authoritative_state_and_conversion_counts(tmp_path):
    player = {
        "hp": 42,
        "max_hp": 80,
        "deck_size": 1,
        "deck": [{"id": "CARD.STRIKE_IRONCLAD", "name": "Strike"}],
        "relics": [],
        "potions": [],
    }
    engine_trace = tmp_path / "engine_trace.jsonl"
    _write_jsonl(
        engine_trace,
        [
            {
                "step": 1,
                "type": "action",
                "data": {
                    "cmd": "start_run",
                    "seed": "fixed",
                    "character": "Ironclad",
                    "ascension": 0,
                    "lang": "en",
                },
            },
            {
                "step": 1,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "map_select",
                    "context": {"act": 1, "floor": 2, "room_type": "Map"},
                    "choices": [{"col": 0, "row": 1, "type": "Monster"}],
                    "player": player,
                },
            },
            {
                "step": 1,
                "type": "action",
                "data": {"cmd": "action", "action": "select_map_node", "args": {"col": 0, "row": 1}},
            },
            {
                "step": 2,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "map_select",
                    "context": {"act": 1, "floor": 3, "room_type": "Map"},
                    "choices": [],
                    "player": player,
                },
            },
        ],
    )
    config = {
        "seed": "fixed",
        "character": "Ironclad",
        "ascension": 0,
        "lang": "en",
        "protocol": "human_cli_raw_fifo_v1",
    }
    summary = {
        "termination_class": "model_forfeit",
        "formal_behavior_sample": True,
        "completed": False,
        "outcome": "forfeit",
        "decision_limit_reached": False,
        "error": None,
    }

    engine_audit = audit_engine_trace(engine_trace, config=config, cli_summary=summary)
    assert engine_audit["valid"] is True
    assert engine_audit["authoritative_terminal_state"]["decision"] == "map_select"
    assert engine_audit["max_observed_floor"] == 3
    assert engine_audit["max_observed_floor_by_act"] == {"1": 3}
    assert engine_audit["cumulative_floor_components_by_act"] == {"1": 3}
    assert engine_audit["cumulative_observed_floor"] == 3

    benchmark_trace = tmp_path / "benchmark_trace.jsonl"
    expected_rows = convert_engine_trace(engine_trace, config=config, cli_summary=summary)
    _write_jsonl(
        benchmark_trace,
        expected_rows,
    )
    benchmark_audit = audit_benchmark_trace(
        benchmark_trace,
        engine_audit=engine_audit,
        expected_rows=expected_rows,
    )
    assert benchmark_audit["valid"] is True
    converted_rows = [json.loads(line) for line in benchmark_trace.read_text(encoding="utf-8").splitlines()]
    converted_terminal_decision = [row for row in converted_rows if row["type"] == "decision"][-1]
    assert converted_terminal_decision["state"]["last_action_result"]["action_kind"] == "map"
    result = converted_rows[-1]["result"]
    assert result["extra"]["authoritative_terminal_state"] == engine_audit["authoritative_terminal_state"]
    assert "last_action_result" not in result["extra"]["authoritative_terminal_state"]

    rows = [json.loads(line) for line in benchmark_trace.read_text(encoding="utf-8").splitlines()]
    next(row for row in rows if row["type"] == "action")["action"]["label"] = "tampered"
    _write_jsonl(benchmark_trace, rows)
    tampered_audit = audit_benchmark_trace(
        benchmark_trace,
        engine_audit=engine_audit,
        expected_rows=expected_rows,
    )
    assert tampered_audit["valid"] is False
    assert tampered_audit["checks"]["rows_match_deterministic_conversion"] is False


def test_engine_audit_sums_visible_floor_progress_across_acts(tmp_path):
    player = {
        "hp": 42,
        "max_hp": 80,
        "deck_size": 1,
        "deck": [{"id": "CARD.STRIKE_IRONCLAD", "name": "Strike"}],
        "relics": [],
        "potions": [],
    }
    engine_trace = tmp_path / "engine_trace.jsonl"
    _write_jsonl(
        engine_trace,
        [
            {
                "step": 1,
                "type": "action",
                "data": {
                    "cmd": "start_run",
                    "seed": "fixed",
                    "character": "Ironclad",
                    "ascension": 0,
                    "lang": "en",
                },
            },
            {
                "step": 2,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "map_select",
                    "context": {
                        "act": 1,
                        "engine_floor": 17,
                        "display_floor": 16,
                        "room_type": "Map",
                    },
                    "choices": [],
                    "player": player,
                },
            },
            {
                "step": 3,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "map_select",
                    "context": {
                        "act": 2,
                        "engine_floor": 0,
                        "display_floor": 0,
                        "room_type": "Map",
                    },
                    "choices": [],
                    "player": player,
                },
            },
            {
                "step": 4,
                "type": "state",
                "data": {
                    "type": "decision",
                    "decision": "map_select",
                    "context": {
                        "act": 2,
                        "engine_floor": 14,
                        "display_floor": 13,
                        "room_type": "Map",
                    },
                    "choices": [],
                    "player": player,
                },
            },
        ],
    )
    audit = audit_engine_trace(
        engine_trace,
        config={
            "seed": "fixed",
            "character": "Ironclad",
            "ascension": 0,
            "lang": "en",
        },
        cli_summary={
            "termination_class": "model_forfeit",
            "completed": False,
            "outcome": "forfeit",
            "decision_limit_reached": False,
            "error": None,
        },
    )

    assert audit["valid"] is True
    assert audit["max_observed_floor"] == 16
    assert audit["max_observed_floor_by_act"] == {"1": 16, "2": 13}
    assert audit["cumulative_floor_components_by_act"] == {"1": 16, "2": 14}
    assert audit["cumulative_observed_floor"] == 30
