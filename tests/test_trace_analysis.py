import json
import re

from sts2_bench.actions import LegalAction
from sts2_bench.evidence import build_evidence_bundle
from sts2_bench.runner import _decision_log_entry
from sts2_bench.trace_analysis import analyze_trace


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _decision(step, state, legal_actions):
    return {
        "type": "decision",
        "step": step,
        "decision": state["decision"],
        "state": state,
        "legal_actions": legal_actions,
    }


def _action(step, decision, label, kind, *, action_id=0, command=None, reason=""):
    return {
        "type": "action",
        "step": step,
        "decision": decision,
        "action": {
            "action_id": action_id,
            "kind": kind,
            "label": label,
            "cmd": command or {"cmd": "action", "action": "noop"},
        },
        "agent_meta": {"parsed": {"reason": reason}},
    }


def _state(decision, *, act, floor, room, hp, max_hp=80, **extra):
    state = {
        "decision": decision,
        "act": act,
        "floor": floor,
        "context": {"act": act, "floor": floor, "room_type": room},
        "player": {
            "hp": hp,
            "max_hp": max_hp,
            "gold": 100,
            "deck_size": 12,
        },
    }
    state.update(extra)
    return state


def _map_action(action_id, label, col, row):
    return {
        "action_id": action_id,
        "kind": "map",
        "label": label,
        "cmd": {
            "cmd": "action",
            "action": "select_map_node",
            "args": {"col": col, "row": row},
        },
    }


def _sample_map_snapshot():
    return {
        "type": "map",
        "context": {"act": 1, "floor": 1, "room_type": "Event"},
        "rows": [
            [
                {
                    "col": 0,
                    "row": 0,
                    "type": "Ancient",
                    "children": [{"col": 0, "row": 1}, {"col": 2, "row": 1}],
                    "visited": True,
                    "current": True,
                }
            ],
            [
                {
                    "col": 0,
                    "row": 1,
                    "type": "Monster",
                    "children": [{"col": 1, "row": 2}],
                    "visited": False,
                    "current": False,
                },
                {
                    "col": 2,
                    "row": 1,
                    "type": "Shop",
                    "children": [{"col": 1, "row": 2}],
                    "visited": False,
                    "current": False,
                },
            ],
        ],
        "boss": {"col": 1, "row": 2, "type": "Boss", "name": "Test Boss"},
        "current_coord": {"col": 0, "row": 0},
    }


def test_decision_log_entry_records_map_snapshot_outside_state():
    state = _state(
        "map_select",
        act=1,
        floor=1,
        room="Map",
        hp=80,
        full_map=_sample_map_snapshot(),
    )

    entry = _decision_log_entry(
        log_level="decisions",
        step=0,
        state=state,
        legal_actions=[
            LegalAction(
                action_id=0,
                label="go to Monster at col=0 row=1",
                command={"cmd": "action", "action": "select_map_node", "args": {"col": 0, "row": 1}},
                kind="map",
            )
        ],
        prompt=None,
    )

    assert entry["map_snapshot"]["type"] == "map"
    assert "full_map" not in entry["state"]
    assert "map_snapshot" not in entry["state"]


def test_decision_log_entry_records_enemy_snapshot_outside_compact_state():
    state = _state(
        "combat_play",
        act=1,
        floor=2,
        room="Monster",
        hp=70,
        enemies=[
            {
                "index": 0,
                "name": "Jaw Worm",
                "hp": 33,
                "max_hp": 40,
                "block": 5,
                "move_name": "Chomp",
                "intents": [{"type": "Attack", "damage": 7, "hits": 2}],
                "powers": [{"name": "Strength", "amount": 2, "type": "Buff"}],
            }
        ],
    )

    entry = _decision_log_entry(
        log_level="decisions",
        step=0,
        state=state,
        legal_actions=[],
        prompt=None,
    )

    assert "enemies" not in entry["state"]
    assert entry["state"]["enemy_count"] == 1
    assert entry["enemy_snapshot"] == [
        {
            "index": 0,
            "name": "Jaw Worm",
            "hp": 33,
            "max_hp": 40,
            "block": 5,
            "intent": "Attack 7x2",
            "intents": [{"type": "Attack", "damage": 7, "hits": 2, "total_damage": 14, "text": "Attack 7x2"}],
            "move_name": "Chomp",
            "powers": [{"name": "Strength", "amount": 2, "type": "Buff"}],
        }
    ]


def test_decision_log_entry_records_nested_combat_enemy_snapshot():
    state = _state(
        "card_select",
        act=2,
        floor=7,
        room="Elite",
        hp=27,
        combat={
            "enemies": [
                {
                    "index": 0,
                    "name": "Entomancer",
                    "hp": 1,
                    "max_hp": 145,
                    "block": 0,
                    "move_name": "Attack",
                    "intents": [{"type": "Attack", "damage": 28, "hits": 1}],
                }
            ]
        },
    )

    entry = _decision_log_entry(
        log_level="decisions",
        step=289,
        state=state,
        legal_actions=[],
        prompt=None,
    )

    assert entry["state"]["enemy_count"] == 0
    assert entry["enemy_snapshot"][0]["name"] == "Entomancer"
    assert entry["enemy_snapshot"][0]["intent"] == "Attack 28"
    assert entry["enemy_snapshot"][0]["intents"][0]["total_damage"] == 28


def test_decision_log_entry_records_evidence_summary_flags():
    state = _state(
        "card_reward",
        act=1,
        floor=2,
        room="Monster",
        hp=70,
        cards=[{"index": 0, "name": "Twin Strike"}],
        can_skip=True,
        player={
            "hp": 70,
            "max_hp": 80,
            "gold": 100,
            "deck_size": 3,
            "deck": [
                {"id": "CARD.STRIKE", "name": "Strike", "type": "Attack", "cost": 1},
                {"id": "CARD.STRIKE", "name": "Strike", "type": "Attack", "cost": 1},
                {"id": "CARD.DEFEND", "name": "Defend", "type": "Skill", "cost": 1},
            ],
        },
    )
    legal_actions = [
        LegalAction(
            action_id=0,
            label="pick card 0: Twin Strike",
            command={"cmd": "action", "action": "select_card_reward", "args": {"card_index": 0}},
            kind="card_reward",
        ),
        LegalAction(
            action_id=1,
            label="skip card reward",
            command={"cmd": "action", "action": "skip_card_reward"},
            kind="card_reward_skip",
        ),
    ]
    evidence_bundle = build_evidence_bundle(state, legal_actions)

    entry = _decision_log_entry(
        log_level="decisions",
        step=0,
        state=state,
        legal_actions=legal_actions,
        prompt=None,
        evidence_bundle=evidence_bundle,
    )

    assert entry["evidence_bundle_hash"] == evidence_bundle.bundle_hash()
    assert entry["deck_summary_visible"] is True
    assert entry["full_deck_details_visible"] is False
    assert entry["deck_details_visible"] is False
    assert entry["skip_action_visible"] is True
    assert entry["evidence_bundle"]["candidate_count"] == 2
    assert entry["evidence_bundle"]["deck_summary_visible"] is True
    assert entry["evidence_bundle"]["deck_group_count"] == 2
    assert entry["evidence_bundle"]["decision_type"] == "card_reward"


def test_analyze_trace_writes_replay_metrics_and_high_risk_events(tmp_path):
    trace_path = tmp_path / "sample.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        {
            "type": "start",
            "seed": "seed-a",
            "character": "Ironclad",
            "ascension": 0,
        },
        _decision(
            0,
            _state(
                "map_select",
                act=1,
                floor=1,
                room="Map",
                hp=42,
                choices=[
                    {"col": 1, "row": 1, "type": "RestSite"},
                    {"col": 2, "row": 1, "type": "Elite"},
                ],
            ),
            [
                _map_action(0, "go to RestSite at col=1 row=1", 1, 1),
                _map_action(1, "go to Elite at col=2 row=1", 2, 1),
            ],
        ),
        _action(
            0,
            "map_select",
            "go to Elite at col=2 row=1",
            "map",
            action_id=1,
            command=_map_action(1, "go to Elite at col=2 row=1", 2, 1)["cmd"],
            reason="Elite rewards are worth it despite low HP.",
        ),
        _decision(
            1,
            _state("combat_play", act=1, floor=2, room="Elite", hp=42),
            [
                {
                    "action_id": 0,
                    "kind": "combat_end_turn",
                    "label": "end turn",
                    "cmd": {"cmd": "action", "action": "end_turn"},
                }
            ],
        ),
        _action(1, "combat_play", "end turn", "combat_end_turn"),
        _decision(
            2,
            _state(
                "card_reward",
                act=1,
                floor=2,
                room="Elite",
                hp=12,
                cards=[
                    {"index": 0, "name": "Strike+"},
                    {"index": 1, "name": "Shrug It Off"},
                    {"index": 2, "name": "Skip Candidate"},
                ],
            ),
            [
                {
                    "action_id": 0,
                    "kind": "combat_reward",
                    "label": "take card 1: Shrug It Off",
                    "cmd": {"cmd": "action", "action": "choose_card_reward", "args": {"index": 1}},
                },
                {
                    "action_id": 1,
                    "kind": "combat_reward_skip",
                    "label": "skip card reward",
                    "cmd": {"cmd": "action", "action": "skip_card_reward"},
                },
            ],
        ),
        _action(2, "card_reward", "take card 1: Shrug It Off", "card_reward"),
        _decision(
            3,
            _state(
                "card_reward",
                act=1,
                floor=3,
                room="Monster",
                hp=12,
                cards=[
                    {"index": 0, "name": "Anger"},
                    {"index": 1, "name": "Flex"},
                    {"index": 2, "name": "Havoc"},
                ],
            ),
            [
                {
                    "action_id": 0,
                    "kind": "combat_reward",
                    "label": "take card 0: Anger",
                    "cmd": {"cmd": "action", "action": "choose_card_reward", "args": {"index": 0}},
                },
                {
                    "action_id": 1,
                    "kind": "combat_reward_skip",
                    "label": "skip card reward",
                    "cmd": {"cmd": "action", "action": "skip_card_reward"},
                },
            ],
        ),
        _action(3, "card_reward", "skip card reward", "card_reward_skip", action_id=1),
        _decision(4, _state("map_select", act=1, floor=3, room="Map", hp=12), []),
        _decision(
            5,
            _state(
                "rest_site",
                act=1,
                floor=4,
                room="RestSite",
                hp=12,
            ),
            [
                {"action_id": 0, "kind": "rest_site", "label": "choose option 0: Rest"},
                {"action_id": 1, "kind": "rest_site", "label": "choose option 1: Smith"},
            ],
        ),
        _action(5, "rest_site", "choose option 1: Smith", "rest_site"),
        _decision(
            6,
            _state("card_select", act=1, floor=4, room="RestSite", hp=12),
            [{"action_id": 0, "kind": "card_select", "label": "select card 0: Bash"}],
        ),
        _action(6, "card_select", "select card 0: Bash", "card_select"),
        _decision(7, _state("map_select", act=1, floor=5, room="Map", hp=12), []),
        _decision(
            8,
            _state(
                "shop",
                act=1,
                floor=6,
                room="Shop",
                hp=12,
            ),
            [
                {"action_id": 0, "kind": "shop_buy_card", "label": "buy card 0: Pommel Strike for 51 gold"},
                {"action_id": 1, "kind": "shop_remove_card", "label": "remove a card for 75 gold"},
            ],
        ),
        _action(8, "shop", "buy card 0: Pommel Strike for 51 gold", "shop_buy_card"),
        _decision(
            9,
            _state("shop", act=1, floor=6, room="Shop", hp=12),
            [{"action_id": 0, "kind": "shop_remove_card", "label": "remove a card for 75 gold"}],
        ),
        _action(9, "shop", "remove a card for 75 gold", "shop_remove_card"),
        _decision(
            10,
            _state("card_select", act=1, floor=6, room="Shop", hp=12),
            [{"action_id": 0, "kind": "card_select", "label": "select card 3: Strike"}],
        ),
        _action(10, "card_select", "select card 3: Strike", "card_select"),
        {
            "type": "result",
            "result": {
                "seed": "seed-a",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 1,
                "floor": 6,
                "room_type": "Shop",
                "hp": 12,
                "max_hp": 80,
                "error": None,
                "truncated": False,
            },
        },
    ]
    _write_jsonl(trace_path, rows)

    report = analyze_trace(trace_path, out_dir)

    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "replay.md").is_file()
    assert (out_dir / "hp_by_action.svg").is_file()
    assert (out_dir / "hp_by_room.svg").is_file()
    assert (out_dir / "map_path.svg").is_file()
    assert (out_dir / "route_hp_replay.svg").is_file()
    assert (out_dir / "large_loss_rooms" / "room_001_Elite.jsonl").is_file()
    assert (out_dir / "large_loss_rooms" / "room_001_Elite.md").is_file()
    hp_by_action_svg = (out_dir / "hp_by_action.svg").read_text(encoding="utf-8")
    hp_by_room_svg = (out_dir / "hp_by_room.svg").read_text(encoding="utf-8")
    route_hp_svg = (out_dir / "route_hp_replay.svg").read_text(encoding="utf-8")

    assert report["hp_timeline"]["by_action"][0]["hp"] == 42
    assert report["rooms"][0]["room_type"] == "Elite"
    assert report["rooms"][0]["entry_hp"] == 42
    assert report["rooms"][0]["exit_hp"] == 12
    assert report["rooms"][0]["hp_loss"] == 30
    assert report["card_rewards"][0]["offered"] == ["Strike+", "Shrug It Off", "Skip Candidate"]
    assert report["card_rewards"][0]["selected"] == "take card 1: Shrug It Off"
    assert report["card_rewards"][1]["offered"] == ["Anger", "Flex", "Havoc"]
    assert report["card_rewards"][1]["skipped"] is True
    assert report["rest_sites"][0]["choice"] == "Smith"
    assert report["rest_sites"][0]["smith_card"] == "Bash"
    assert report["shops"][0]["actions"] == [
        "buy card 0: Pommel Strike for 51 gold",
        "remove a card for 75 gold -> Strike",
    ]
    assert report["shops"][0]["purchases"] == ["buy card 0: Pommel Strike for 51 gold"]
    assert report["shops"][0]["removals"] == [{"label": "remove a card for 75 gold", "card": "Strike"}]
    assert "Low HP zone" in hp_by_action_svg
    assert "A1F2 Elite" in hp_by_action_svg
    assert "Room HP loss" in hp_by_room_svg
    assert "Elite" in hp_by_room_svg
    assert "Entry 42/80" in hp_by_room_svg
    assert "Exit 12/80" in hp_by_room_svg
    assert "loss 30" in hp_by_room_svg
    assert "large loss" in hp_by_room_svg
    assert 'class="event-panel"' in hp_by_action_svg
    assert 'class="room-loss-panel"' in hp_by_room_svg
    assert "start 42/80" not in hp_by_action_svg
    assert "min 12/80" not in hp_by_action_svg
    assert "Entry 42/80 Exit 12/80 loss 30" not in hp_by_room_svg
    assert 'class="room-band"' in hp_by_action_svg
    assert 'class="room-action-segment"' in hp_by_action_svg
    assert 'class="drop-label"' in hp_by_action_svg
    assert 'class="chart-summary"' in hp_by_action_svg
    assert 'class="chart-summary" x="62" y="48"' in hp_by_action_svg
    assert 'data-plot-top="128"' in hp_by_action_svg
    assert 'data-room-band-y="72"' in hp_by_action_svg
    assert 'class="room-band" x="' in hp_by_action_svg
    assert ' y="72"' in hp_by_action_svg
    assert 'class="room-hp-segment large-loss"' in hp_by_room_svg
    assert 'class="loss-label"' in hp_by_room_svg
    assert "F2 Elite -30" in hp_by_room_svg
    assert "42-&gt;12" in hp_by_room_svg
    assert "Room #1" in hp_by_room_svg
    assert "HP axis" in hp_by_room_svg
    assert 'class="map-route-panel"' in route_hp_svg
    assert 'class="hp-route-panel"' in route_hp_svg
    assert 'class="same-floor-map-route"' in route_hp_svg
    assert 'class="same-floor-node selected' in route_hp_svg
    assert 'class="same-floor-node option' in route_hp_svg
    assert "Map route vs HP route" in route_hp_svg
    assert "F2 Elite -30" in route_hp_svg
    assert "42-&gt;12" in route_hp_svg

    event_types = {event["type"] for event in report["high_risk_events"]}
    assert "low_hp_elite_selected" in event_types
    assert "low_hp_safe_route_ignored" in event_types


def test_route_hp_replay_splits_dense_room_labels(tmp_path):
    trace_path = tmp_path / "dense_route_labels.jsonl"
    out_dir = tmp_path / "analysis"
    room_types = [
        "Monster",
        "Monster",
        "RestSite",
        "Monster",
        "Elite",
        "Treasure",
        "Monster",
        "Event",
        "Monster",
        "RestSite",
        "Monster",
        "Boss",
    ]
    losses = [4, 3, 0, 5, 8, 0, 4, 0, 7, 0, 9, 40]
    rows = [{"type": "start", "seed": "seed-a", "character": "Ironclad", "ascension": 0}]
    hp = 80
    step = 0
    for index, (room_type, loss) in enumerate(zip(room_types, losses), start=1):
        label = f"go to {room_type} at col=0 row={index}"
        map_action = _map_action(0, label, 0, index)
        rows.append(_decision(step, _state("map_select", act=1, floor=index, room="Map", hp=hp), [map_action]))
        rows.append(_action(step, "map_select", label, "map", command=map_action["cmd"]))
        rows.append(_decision(step + 1, _state("combat_play", act=1, floor=index, room=room_type, hp=hp), []))
        hp = max(0, hp - loss)
        rows.append(_action(step + 1, "combat_play", "end turn", "combat_end_turn"))
        step += 2
    rows.append(
        {
            "type": "result",
            "result": {
                "seed": "seed-a",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 1,
                "floor": len(room_types),
                "room_type": room_types[-1],
                "hp": hp,
                "max_hp": 80,
                "error": None,
                "truncated": False,
            },
        }
    )
    _write_jsonl(trace_path, rows)

    analyze_trace(trace_path, out_dir)
    route_hp_svg = (out_dir / "route_hp_replay.svg").read_text(encoding="utf-8")

    assert 'class="hp-room-label"' in route_hp_svg
    assert 'class="loss-floor-label"' in route_hp_svg
    assert 'class="loss-room-label"' in route_hp_svg
    assert 'class="loss-label" x="' not in route_hp_svg
    assert "F12 Boss -40" in route_hp_svg
    assert "40-&gt;0" in route_hp_svg


def test_hp_timeline_reads_flat_hp_from_decisions_log(tmp_path):
    trace_path = tmp_path / "flat_hp.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        {"type": "start", "seed": "seed-a", "character": "Ironclad", "ascension": 0},
        {
            "type": "decision",
            "step": 0,
            "decision": "combat_play",
            "state": {
                "decision": "combat_play",
                "act": 1,
                "floor": 2,
                "room_type": "Monster",
                "hp": 70,
                "max_hp": 80,
            },
            "legal_actions": [
                {
                    "action_id": 0,
                    "kind": "combat_end_turn",
                    "label": "end turn",
                    "cmd": {"cmd": "action", "action": "end_turn"},
                }
            ],
        },
        _action(0, "combat_play", "end turn", "combat_end_turn"),
        {
            "type": "decision",
            "step": 1,
            "decision": "combat_play",
            "state": {
                "decision": "combat_play",
                "act": 1,
                "floor": 2,
                "room_type": "Monster",
                "hp": 63,
                "max_hp": 80,
            },
            "legal_actions": [],
        },
        {
            "type": "result",
            "result": {
                "seed": "seed-a",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 1,
                "floor": 2,
                "room_type": "Monster",
                "hp": 63,
                "max_hp": 80,
                "error": None,
                "truncated": False,
            },
        },
    ]
    _write_jsonl(trace_path, rows)

    report = analyze_trace(trace_path, out_dir)
    svg = (out_dir / "hp_by_action.svg").read_text(encoding="utf-8")

    assert report["hp_timeline"]["by_action"][0]["hp"] == 70
    assert report["hp_timeline"]["by_action"][1]["hp"] == 63
    assert "HP 63/80" in svg
    assert "A1F2 Monster" in svg


def test_card_rewards_fallback_to_legal_actions_and_localize_names(tmp_path):
    trace_path = tmp_path / "compact_card_reward.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        {"type": "start", "seed": "seed-a", "character": "Ironclad", "ascension": 0},
        _decision(
            0,
            _state("card_reward", act=2, floor=4, room="Monster", hp=54, max_hp=77),
            [
                {"action_id": 0, "kind": "card_reward", "label": "pick card 0: Demesne"},
                {"action_id": 1, "kind": "card_reward", "label": "pick card 1: Hotfix"},
                {"action_id": 2, "kind": "card_reward", "label": "pick card 2: Ricochet"},
                {"action_id": 3, "kind": "card_reward_skip", "label": "skip card reward"},
            ],
        ),
        _action(0, "card_reward", "pick card 0: Demesne", "card_reward"),
        _decision(
            1,
            _state(
                "map_select",
                act=2,
                floor=4,
                room="Map",
                hp=54,
                max_hp=77,
                last_action_result={
                    "action_label": "pick card 0: Demesne",
                    "changes": ["deck_size: 25 -> 26", "deck: +Demesne"],
                },
            ),
            [],
        ),
        {
            "type": "result",
            "result": {
                "seed": "seed-a",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 2,
                "floor": 4,
                "room_type": "Monster",
                "hp": 54,
                "max_hp": 77,
                "error": None,
                "truncated": False,
            },
        },
    ]
    _write_jsonl(trace_path, rows)

    report = analyze_trace(trace_path, out_dir)
    replay = (out_dir / "replay.md").read_text(encoding="utf-8")

    assert report["card_rewards"][0]["offered"] == ["Demesne", "Hotfix", "Ricochet"]
    assert report["card_rewards"][0]["offered_zh"] == ["领域", "热修复", "连续反弹"]
    assert report["card_rewards"][0]["selected_card"] == "Demesne"
    assert report["card_rewards"][0]["selected_card_zh"] == "领域"
    assert "领域 / Demesne" in replay
    assert "热修复 / Hotfix" in replay


def test_analyze_trace_reconstructs_final_deck_and_writes_chinese_deck_report(tmp_path):
    trace_path = tmp_path / "deck_reconstruction.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        {"type": "start", "seed": "seed-a", "character": "Ironclad", "ascension": 0},
        _decision(
            0,
            _state(
                "card_select",
                act=1,
                floor=1,
                room="Event",
                hp=80,
                max_hp=80,
                player={"hp": 80, "max_hp": 80, "gold": 99, "deck_size": 3},
            ),
            [
                {"action_id": 0, "kind": "card_select", "label": "select card 0: Strike"},
                {"action_id": 1, "kind": "card_select", "label": "select card 1: Defend"},
                {"action_id": 2, "kind": "card_select", "label": "select card 2: Bash"},
            ],
        ),
        _action(0, "card_select", "select card 0: Strike", "card_select"),
        _decision(
            1,
            _state(
                "map_select",
                act=1,
                floor=1,
                room="Map",
                hp=80,
                max_hp=80,
                last_action_result={
                    "action_label": "select card 0: Strike",
                    "changes": ["deck: -Strike +Primal Force"],
                },
            ),
            [],
        ),
        _decision(
            2,
            _state("card_reward", act=1, floor=2, room="Monster", hp=80, max_hp=80),
            [
                {"action_id": 0, "kind": "card_reward", "label": "pick card 0: Drum of Battle"},
                {"action_id": 1, "kind": "card_reward", "label": "pick card 1: Stone Armor"},
            ],
        ),
        _action(2, "card_reward", "pick card 0: Drum of Battle", "card_reward"),
        _decision(
            3,
            _state(
                "shop",
                act=1,
                floor=3,
                room="Shop",
                hp=80,
                max_hp=80,
                last_action_result={
                    "action_label": "pick card 0: Drum of Battle",
                    "changes": ["deck_size: 3 -> 4", "deck: +Drum of Battle"],
                },
            ),
            [{"action_id": 0, "kind": "shop_buy_card", "label": "buy card 0: Stone Armor for 73 gold"}],
        ),
        _action(3, "shop", "buy card 0: Stone Armor for 73 gold", "shop_buy_card"),
        _decision(
            4,
            _state(
                "map_select",
                act=1,
                floor=3,
                room="Map",
                hp=80,
                max_hp=80,
                last_action_result={
                    "action_label": "buy card 0: Stone Armor for 73 gold",
                    "changes": ["deck_size: 4 -> 5", "deck: +Stone Armor"],
                },
            ),
            [],
        ),
        {
            "type": "result",
            "result": {
                "seed": "seed-a",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 1,
                "floor": 3,
                "room_type": "Shop",
                "hp": 80,
                "max_hp": 80,
                "deck_size": 5,
                "error": None,
                "truncated": False,
            },
        },
    ]
    _write_jsonl(trace_path, rows)

    report = analyze_trace(trace_path, out_dir)
    deck_report = (out_dir / "deck_and_card_rewards_zh.md").read_text(encoding="utf-8")

    assert report["final_deck"]["size"] == 5
    assert [card["name"] for card in report["final_deck"]["cards"]] == [
        "Defend",
        "Bash",
        "Primal Force",
        "Drum of Battle",
        "Stone Armor",
    ]
    assert [card["name_zh"] for card in report["final_deck"]["cards"]] == [
        "防御",
        "痛击",
        "原始力量",
        "战鼓",
        "岩石铠甲",
    ]
    assert report["final_deck"]["events"][0]["changes"][0]["operation"] == "remove"
    assert report["final_deck"]["events"][0]["changes"][1]["name_zh"] == "原始力量"
    assert "终局牌组（重构）" in deck_report
    assert "战鼓 / Drum of Battle" in deck_report
    assert "岩石铠甲 / Stone Armor" in deck_report


def test_analyze_trace_summarizes_route_plan_events(tmp_path):
    trace_path = tmp_path / "route_plan_events.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        {"type": "start", "seed": "seed-a", "character": "Ironclad", "ascension": 0},
        _decision(
            0,
            _state("map_select", act=1, floor=1, room="Map", hp=80),
            [_map_action(0, "go to Monster at col=0 row=1", 0, 1)],
        ),
        {
            "type": "route_plan_event",
            "event_type": "route_plan_created",
            "step": 0,
            "selected_node": {"col": 0, "row": 1, "type": "Monster"},
            "new_plan": [{"col": 0, "row": 1, "type": "Monster"}],
            "reason_tags": ["safe_start"],
            "reason": "Open with a hallway fight.",
            "route_validity": "topological",
            "active_plan": {"plan_id": "rp_001", "remaining": []},
        },
        _action(
            0,
            "map_select",
            "go to Monster at col=0 row=1",
            "map",
            command=_map_action(0, "go to Monster at col=0 row=1", 0, 1)["cmd"],
        ),
        {"type": "result", "result": {"seed": "seed-a", "character": "Ironclad", "ascension": 0, "victory": False}},
    ]
    _write_jsonl(trace_path, rows)

    report = analyze_trace(trace_path, out_dir)
    replay = (out_dir / "replay.md").read_text(encoding="utf-8")

    assert report["route_plan_events"][0]["event_type"] == "route_plan_created"
    assert report["route_plan_events"][0]["reason_tags"] == ["safe_start"]
    assert "Route plan events" in replay
    assert "route_plan_created" in replay
    assert "Open with a hallway fight." in replay


def test_route_hp_replay_marks_replan_stage_colors_and_reason(tmp_path):
    trace_path = tmp_path / "route_replan.jsonl"
    out_dir = tmp_path / "analysis"
    full_map = {
        "type": "map",
        "current_coord": {"col": 0, "row": 0},
        "rows": [
            [{"col": 0, "row": 0, "type": "Ancient", "children": [{"col": 0, "row": 1}]}],
            [
                {
                    "col": 0,
                    "row": 1,
                    "type": "Monster",
                    "children": [{"col": 1, "row": 2}, {"col": 2, "row": 2}],
                }
            ],
            [
                {"col": 1, "row": 2, "type": "Elite", "children": [{"col": 1, "row": 3}]},
                {"col": 2, "row": 2, "type": "RestSite", "children": [{"col": 1, "row": 3}]},
            ],
        ],
        "boss": {"col": 1, "row": 3, "type": "Boss", "name": "Test Boss"},
    }
    second_map = {**full_map, "current_coord": {"col": 0, "row": 1}}
    rows = [
        {"type": "start", "seed": "seed-a", "character": "Ironclad", "ascension": 0},
        {
            "type": "decision",
            "step": 0,
            "decision": "map_select",
            "state": _state(
                "map_select",
                act=1,
                floor=1,
                room="Map",
                hp=80,
                choices=[{"col": 0, "row": 1, "type": "Monster"}],
            ),
            "legal_actions": [_map_action(0, "go to Monster at col=0 row=1", 0, 1)],
            "map_snapshot": full_map,
        },
        {
            "type": "route_plan_event",
            "event_type": "route_plan_created",
            "step": 0,
            "selected_node": {"col": 0, "row": 1, "type": "Monster"},
            "new_plan": [
                {"col": 0, "row": 1, "type": "Monster"},
                {"col": 1, "row": 2, "type": "Elite"},
            ],
            "reason_tags": ["risky_elite_path"],
            "reason": "Try the elite branch if HP stays healthy.",
        },
        _action(
            0,
            "map_select",
            "go to Monster at col=0 row=1",
            "map",
            command=_map_action(0, "go to Monster at col=0 row=1", 0, 1)["cmd"],
        ),
        _decision(1, _state("combat_play", act=1, floor=2, room="Monster", hp=50), []),
        _action(1, "combat_play", "end turn", "combat_end_turn", reason="finish fight"),
        {
            "type": "decision",
            "step": 2,
            "decision": "map_select",
            "state": _state(
                "map_select",
                act=1,
                floor=2,
                room="Map",
                hp=35,
                choices=[
                    {"col": 1, "row": 2, "type": "Elite"},
                    {"col": 2, "row": 2, "type": "RestSite"},
                ],
            ),
            "legal_actions": [
                _map_action(0, "go to Elite at col=1 row=2", 1, 2),
                _map_action(1, "go to RestSite at col=2 row=2", 2, 2),
            ],
            "map_snapshot": second_map,
        },
        {
            "type": "route_plan_event",
            "event_type": "route_plan_updated",
            "step": 2,
            "selected_node": {"col": 2, "row": 2, "type": "RestSite"},
            "old_plan_remaining": [{"col": 1, "row": 2, "type": "Elite"}],
            "new_plan": [{"col": 2, "row": 2, "type": "RestSite"}],
            "reason_tags": ["low_hp", "avoid_elite"],
            "reason": "HP is low, so avoid the elite and take the rest site.",
        },
        _action(
            2,
            "map_select",
            "go to RestSite at col=2 row=2",
            "map",
            command=_map_action(1, "go to RestSite at col=2 row=2", 2, 2)["cmd"],
            reason="update route after HP loss",
        ),
        {
            "type": "result",
            "result": {
                "seed": "seed-a",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 1,
                "floor": 3,
                "room_type": "RestSite",
                "hp": 35,
                "max_hp": 80,
                "error": None,
                "truncated": False,
            },
        },
    ]
    _write_jsonl(trace_path, rows)

    analyze_trace(trace_path, out_dir)
    route_hp_svg = (out_dir / "route_hp_replay.svg").read_text(encoding="utf-8")

    assert 'data-route-stage="1"' in route_hp_svg
    assert 'data-route-stage="2"' in route_hp_svg
    assert "route-plan-stage-1" in route_hp_svg
    assert "route-plan-stage-2" in route_hp_svg
    assert "REPLAN R2" in route_hp_svg
    assert "low_hp, avoid_elite" in route_hp_svg
    assert "HP is low, so avoid the elite and take the rest site." in route_hp_svg


def test_analyze_trace_flags_forced_elite_after_previous_branch(tmp_path):
    trace_path = tmp_path / "forced_elite.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        _decision(
            0,
            _state(
                "map_select",
                act=2,
                floor=1,
                room="Map",
                hp=80,
                choices=[
                    {"col": 1, "row": 1, "type": "Monster"},
                    {"col": 4, "row": 1, "type": "Monster"},
                ],
            ),
            [
                _map_action(0, "go to Monster at col=1 row=1", 1, 1),
                _map_action(1, "go to Monster at col=4 row=1", 4, 1),
            ],
        ),
        _action(
            0,
            "map_select",
            "go to Monster at col=4 row=1",
            "map",
            action_id=1,
            command=_map_action(1, "go to Monster at col=4 row=1", 4, 1)["cmd"],
            reason="This branch reaches multiple elites for relic value.",
        ),
        _decision(1, _state("combat_play", act=2, floor=2, room="Monster", hp=80), []),
        _action(1, "combat_play", "end turn", "combat_end_turn"),
        _decision(2, _state("map_select", act=2, floor=3, room="Map", hp=20), []),
        _decision(
            3,
            _state(
                "map_select",
                act=2,
                floor=4,
                room="Map",
                hp=20,
                choices=[{"col": 3, "row": 6, "type": "Elite"}],
            ),
            [_map_action(0, "go to Elite at col=3 row=6", 3, 6)],
        ),
        _action(
            3,
            "map_select",
            "go to Elite at col=3 row=6",
            "map",
            command=_map_action(0, "go to Elite at col=3 row=6", 3, 6)["cmd"],
            reason="Only available node; must fight elite despite low HP.",
        ),
        {
            "type": "result",
            "result": {
                "seed": "seed-forced",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 2,
                "floor": 6,
                "room_type": "Elite",
                "hp": 0,
                "max_hp": 80,
                "error": None,
                "truncated": False,
            },
        },
    ]
    _write_jsonl(trace_path, rows)

    report = analyze_trace(trace_path, out_dir)

    forced_events = [
        event
        for event in report["high_risk_events"]
        if event["type"] == "low_hp_forced_elite_after_branch"
    ]
    assert len(forced_events) == 1
    assert forced_events[0]["branch_step"] == 0
    assert forced_events[0]["branch_selected"]["label"] == "go to Monster at col=4 row=1"


def test_map_path_svg_separates_reused_coordinates_by_act(tmp_path):
    trace_path = tmp_path / "multi_act_map.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        {"type": "start", "seed": "seed-a", "character": "Ironclad", "ascension": 0},
        _decision(
            0,
            _state(
                "map_select",
                act=1,
                floor=1,
                room="Map",
                hp=80,
                choices=[
                    {"col": 1, "row": 1, "type": "Monster"},
                    {"col": 2, "row": 1, "type": "Shop"},
                ],
            ),
            [
                _map_action(0, "go to Monster at col=1 row=1", 1, 1),
                _map_action(1, "go to Shop at col=2 row=1", 2, 1),
            ],
        ),
        _action(
            0,
            "map_select",
            "go to Monster at col=1 row=1",
            "map",
            command=_map_action(0, "go to Monster at col=1 row=1", 1, 1)["cmd"],
        ),
        _decision(
            1,
            _state(
                "map_select",
                act=2,
                floor=1,
                room="Map",
                hp=70,
                choices=[
                    {"col": 1, "row": 1, "type": "Elite"},
                    {"col": 2, "row": 1, "type": "RestSite"},
                ],
            ),
            [
                _map_action(0, "go to Elite at col=1 row=1", 1, 1),
                _map_action(1, "go to RestSite at col=2 row=1", 2, 1),
            ],
        ),
        _action(
            1,
            "map_select",
            "go to Elite at col=1 row=1",
            "map",
            command=_map_action(0, "go to Elite at col=1 row=1", 1, 1)["cmd"],
        ),
        {
            "type": "result",
            "result": {
                "seed": "seed-a",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 2,
                "floor": 1,
                "room_type": "Elite",
                "hp": 0,
                "max_hp": 80,
                "error": None,
                "truncated": False,
            },
        },
    ]
    _write_jsonl(trace_path, rows)

    report = analyze_trace(trace_path, out_dir)
    svg = (out_dir / "map_path.svg").read_text(encoding="utf-8")

    assert report["map_choices"][0]["selected"]["node"]["type"] == "Monster"
    assert report["map_choices"][1]["selected"]["node"]["type"] == "Elite"
    assert "Act 1" in svg
    assert "Act 2" in svg
    assert "act=1 col=1 row=1 type=Monster" in svg
    assert "act=2 col=1 row=1 type=Elite" in svg


def test_map_path_svg_uses_full_map_snapshot_when_available(tmp_path):
    trace_path = tmp_path / "full_map_trace.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        {"type": "start", "seed": "seed-a", "character": "Ironclad", "ascension": 0},
        {
            "type": "decision",
            "step": 0,
            "decision": "map_select",
            "state": _state(
                "map_select",
                act=1,
                floor=1,
                room="Map",
                hp=80,
                choices=[
                    {"col": 0, "row": 1, "type": "Monster"},
                    {"col": 2, "row": 1, "type": "Shop"},
                ],
            ),
            "legal_actions": [
                _map_action(0, "go to Monster at col=0 row=1", 0, 1),
                _map_action(1, "go to Shop at col=2 row=1", 2, 1),
            ],
            "map_snapshot": _sample_map_snapshot(),
        },
        _action(
            0,
            "map_select",
            "go to Monster at col=0 row=1",
            "map",
            command=_map_action(0, "go to Monster at col=0 row=1", 0, 1)["cmd"],
        ),
        {
            "type": "result",
            "result": {
                "seed": "seed-a",
                "character": "Ironclad",
                "ascension": 0,
                "victory": False,
                "decision": "game_over",
                "act": 1,
                "floor": 1,
                "room_type": "Monster",
                "hp": 0,
                "max_hp": 80,
                "error": None,
                "truncated": False,
            },
        },
    ]
    _write_jsonl(trace_path, rows)

    report = analyze_trace(trace_path, out_dir)
    svg = (out_dir / "map_path.svg").read_text(encoding="utf-8")
    route_hp_svg = (out_dir / "route_hp_replay.svg").read_text(encoding="utf-8")

    assert report["map_choices"][0]["map_snapshot"]["type"] == "map"
    assert "Full map path" in svg
    assert "class=\"map-edge\"" in svg
    assert "act=1 col=2 row=1 type=Shop" in svg
    assert "act=1 col=1 row=2 type=Boss" in svg
    assert "Played map topology" in route_hp_svg
    assert route_hp_svg.count("Played map topology") == 1
    assert ".panel-title { font-size: 26px" in route_hp_svg
    assert ".section-title { font-size: 19px" in route_hp_svg
    assert ".route-topology-act-label { font-size: 18px" in route_hp_svg
    assert ".route-node-icon { font-size: 18px" in route_hp_svg
    assert ".route-topology-coordinate { font-size: 15px" in route_hp_svg
    assert ".selected-room-badge { font-size: 18px" in route_hp_svg
    assert ".loss-floor-label { font-size: 15px" in route_hp_svg
    assert ".loss-room-label { font-size: 14px" in route_hp_svg
    assert ".hp-transition-label { font-size: 15px" in route_hp_svg
    assert ".room-index-label { font-size: 13px" in route_hp_svg
    assert "dominant-baseline: central" in route_hp_svg
    assert 'dominant-baseline="central"' in route_hp_svg
    assert 'y="672">HP route by room' in route_hp_svg
    assert 'r="20"' in route_hp_svg
    assert 'r="17"' in route_hp_svg
    assert 'r="14"' in route_hp_svg
    assert 'class="section-title"' in route_hp_svg
    assert 'class="route-topology-map"' in route_hp_svg
    assert 'data-layout="horizontal"' in route_hp_svg
    assert 'data-x-axis="row"' in route_hp_svg
    assert 'data-y-axis="col"' in route_hp_svg
    assert 'class="route-topology-panel" fill="none"' in route_hp_svg
    assert 'class="route-topology-act-label"' in route_hp_svg
    assert route_hp_svg.index('class="route-topology-panel"') < route_hp_svg.index('class="route-topology-act-label"')
    panel_match = re.search(r'class="route-topology-panel" fill="none" x="[^"]+" y="([^"]+)"', route_hp_svg)
    label_match = re.search(r'class="route-topology-act-label" x="[^"]+" y="([^"]+)"', route_hp_svg)
    assert panel_match is not None
    assert label_match is not None
    assert float(label_match.group(1)) <= float(panel_match.group(1)) - 10
    assert "#fbf8ef" not in route_hp_svg
    assert 'class="route-topology-edge"' in route_hp_svg
    assert 'class="route-topology-node selected' in route_hp_svg
    assert 'class="route-topology-node option' in route_hp_svg
    assert 'class="route-topology-path"' in route_hp_svg
    match = re.search(r'class="route-topology-path" points="([^"]+)"', route_hp_svg)
    assert match is not None
    path_points = [
        tuple(float(part) for part in point.split(","))
        for point in match.group(1).split()
    ]
    assert len(path_points) >= 2
    dx = abs(path_points[-1][0] - path_points[0][0])
    dy = abs(path_points[-1][1] - path_points[0][1])
    assert dx > dy
    assert 'class="selected-room-badge"' in route_hp_svg
    assert "act=1 floor=1 col=0 row=1 type=Monster" in route_hp_svg
    assert "act=1 floor=1 col=2 row=1 type=Shop" in route_hp_svg
    assert "type=Boss" not in route_hp_svg
    assert 'class="same-floor-map-route"' not in route_hp_svg
