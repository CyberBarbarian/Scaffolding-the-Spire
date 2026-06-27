from sts2_bench.actions import LegalAction
from sts2_bench.route_graph import build_canonical_route_map, build_route_graph


def _map_action(action_id: int, room_type: str, col: int, row: int) -> LegalAction:
    return LegalAction(
        action_id=action_id,
        label=f"go to {room_type} at col={col} row={row}",
        command={"cmd": "action", "action": "select_map_node", "args": {"col": col, "row": row}},
        kind="map",
    )


def test_build_route_graph_exposes_exact_nodes_edges_and_legal_starts():
    full_map = {
        "type": "map",
        "current_coord": {"col": 3, "row": 0},
        "rows": [
            [{"col": 3, "row": 0, "type": "Ancient", "children": [{"col": 1, "row": 1}, {"col": 3, "row": 1}]}],
            [
                {"col": 1, "row": 1, "type": "Monster", "children": [{"col": 2, "row": 2}]},
                {"col": 3, "row": 1, "type": "Monster", "children": [{"col": 4, "row": 2}]},
            ],
            [
                {"col": 2, "row": 2, "type": "RestSite", "children": [{"col": 3, "row": 3}]},
                {"col": 4, "row": 2, "type": "Elite", "children": [{"col": 3, "row": 3}]},
            ],
            [{"col": 3, "row": 3, "type": "Boss", "children": []}],
        ],
        "boss": {"col": 3, "row": 3, "type": "Boss"},
    }
    choices = [
        {"col": 1, "row": 1, "type": "Monster"},
        {"col": 3, "row": 1, "type": "Monster"},
    ]

    graph = build_route_graph(
        full_map,
        choices,
        legal_actions=[
            _map_action(5, "Monster", 1, 1),
            _map_action(6, "Monster", 3, 1),
        ],
    )

    assert graph["current"] == {"col": 3, "row": 0, "type": "Ancient", "node_key": "3:0"}
    assert graph["legal_starts"] == [
        {"col": 1, "row": 1, "type": "Monster", "node_key": "1:1", "choice_index": 0, "action_id": 5},
        {"col": 3, "row": 1, "type": "Monster", "node_key": "3:1", "choice_index": 1, "action_id": 6},
    ]
    assert graph["constraints"] == {
        "ordered_plan_required": True,
        "complete_visible_route_required": True,
        "winged_boots_max_forward_jumps": 3,
    }
    monster = next(node for node in graph["nodes"] if node["node_key"] == "1:1")
    assert monster == {
        "col": 1,
        "row": 1,
        "type": "Monster",
        "node_key": "1:1",
        "children": [{"col": 2, "row": 2, "type": "RestSite", "node_key": "2:2"}],
    }


def test_build_route_graph_omits_children_not_visible_in_snapshot():
    full_map = {
        "type": "map",
        "current_coord": {"col": 0, "row": 0},
        "rows": [
            [{"col": 0, "row": 0, "type": "Ancient", "children": [{"col": 0, "row": 1}]}],
            [{"col": 0, "row": 1, "type": "Monster", "children": [{"col": 9, "row": 9}]}],
        ],
    }

    graph = build_route_graph(full_map, [{"col": 0, "row": 1, "type": "Monster"}])

    monster = next(node for node in graph["nodes"] if node["node_key"] == "0:1")
    assert monster["children"] == []


def test_build_route_graph_can_expose_winged_boots_remaining_jumps():
    full_map = {
        "type": "map",
        "current_coord": {"col": 0, "row": 0},
        "rows": [
            [{"col": 0, "row": 0, "type": "Ancient", "children": [{"col": 0, "row": 1}]}],
            [{"col": 0, "row": 1, "type": "Monster", "children": []}],
        ],
    }

    graph = build_route_graph(
        full_map,
        [{"col": 0, "row": 1, "type": "Monster"}],
        winged_boots_remaining=2,
    )

    assert graph["constraints"]["winged_boots_max_forward_jumps"] == 3
    assert graph["constraints"]["winged_boots_remaining_forward_jumps"] == 2


def test_build_canonical_route_map_groups_nodes_by_row_and_edges_by_id():
    full_map = {
        "type": "map",
        "current_coord": {"col": 3, "row": 0},
        "rows": [
            [{"col": 3, "row": 0, "type": "Ancient", "children": [{"col": 1, "row": 1}, {"col": 3, "row": 1}]}],
            [
                {"col": 1, "row": 1, "type": "Monster", "children": [{"col": 2, "row": 2}]},
                {"col": 3, "row": 1, "type": "Monster", "children": [{"col": 4, "row": 2}]},
            ],
            [
                {"col": 2, "row": 2, "type": "RestSite", "children": [{"col": 3, "row": 3}]},
                {"col": 4, "row": 2, "type": "Elite", "children": [{"col": 3, "row": 3}]},
            ],
            [{"col": 3, "row": 3, "type": "Boss", "children": []}],
        ],
        "boss": {"col": 3, "row": 3, "type": "Boss"},
    }

    route_map = build_canonical_route_map(
        full_map,
        [{"col": 1, "row": 1, "type": "Monster"}, {"col": 3, "row": 1, "type": "Monster"}],
        legal_actions=[_map_action(5, "Monster", 1, 1), _map_action(6, "Monster", 3, 1)],
        winged_boots_remaining=1,
    )

    assert route_map["schema"] == "sts2_route_map_v1"
    assert route_map["current"] == "r00c3"
    assert route_map["legal_starts"] == [
        {"action_id": 5, "node": "r01c1", "symbol": "M", "type": "Monster", "choice_index": 0},
        {"action_id": 6, "node": "r01c3", "symbol": "M", "type": "Monster", "choice_index": 1},
    ]
    assert route_map["constraints"]["only_follow_edges"] is True
    assert route_map["constraints"]["complete_visible_route_required"] is True
    assert route_map["constraints"]["winged_boots_left"] == 1
    assert route_map["rows"][1] == {
        "row": 1,
        "nodes": [
            {"id": "r01c1", "col": 1, "row": 1, "symbol": "M", "type": "Monster", "action_id": 5},
            {"id": "r01c3", "col": 3, "row": 1, "symbol": "M", "type": "Monster", "action_id": 6},
        ],
    }
    assert route_map["edges"] == {
        "r00c3": ["r01c1", "r01c3"],
        "r01c1": ["r02c2"],
        "r01c3": ["r02c4"],
        "r02c2": ["r03c3"],
        "r02c4": ["r03c3"],
        "r03c3": [],
    }
