from sts2_bench.actions import LegalAction
from sts2_bench.route_graph import build_route_graph


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
