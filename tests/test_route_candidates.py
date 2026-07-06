from sts2_bench.actions import build_legal_actions
from sts2_bench.route_candidates import enumerate_route_candidates, rank_route_candidates, route_to_text


def _node(col: int, row: int, room_type: str, children: list[tuple[int, int]]) -> dict:
    return {
        "col": col,
        "row": row,
        "type": room_type,
        "children": [{"col": child_col, "row": child_row} for child_col, child_row in children],
    }


def _branching_map() -> dict:
    return {
        "type": "map",
        "current_coord": {"col": 1, "row": 0},
        "rows": [
            [_node(1, 0, "Ancient", [(0, 1), (2, 1)])],
            [
                _node(0, 1, "Monster", [(0, 2)]),
                _node(2, 1, "Monster", [(2, 2)]),
            ],
            [
                _node(0, 2, "RestSite", [(0, 3)]),
                _node(2, 2, "Elite", [(2, 3)]),
            ],
            [
                _node(0, 3, "RestSite", [(1, 4)]),
                _node(2, 3, "RestSite", [(1, 4)]),
            ],
        ],
        "boss": {"col": 1, "row": 4, "type": "Boss"},
    }


def test_enumerates_complete_routes_from_legal_starts():
    choices = [
        {"col": 0, "row": 1, "type": "Monster"},
        {"col": 2, "row": 1, "type": "Monster"},
    ]
    state = {"decision": "map_select", "choices": choices}
    legal_actions = build_legal_actions(state)

    candidates = enumerate_route_candidates(_branching_map(), choices, legal_actions=legal_actions)

    assert len(candidates) == 2
    assert {candidate.action_id for candidate in candidates} == {5, 6}
    assert all(candidate.nodes[-1]["type"] == "Boss" for candidate in candidates)


def test_max_restsites_prefers_more_campfires_then_fewer_elites():
    choices = [
        {"col": 0, "row": 1, "type": "Monster"},
        {"col": 2, "row": 1, "type": "Monster"},
    ]
    state = {"decision": "map_select", "choices": choices}
    legal_actions = build_legal_actions(state)

    ranked = rank_route_candidates(
        enumerate_route_candidates(_branching_map(), choices, legal_actions=legal_actions),
        objective="max_restsites",
    )

    assert ranked[0].action_id == 5
    assert ranked[0].restsite_count == 2
    assert ranked[0].elite_count == 0
    assert route_to_text(ranked[0]).startswith("Monster(0,1) -> RestSite(0,2)")


def _risk_tiebreaker_map() -> dict:
    return {
        "type": "map",
        "current_coord": {"col": 1, "row": 0},
        "rows": [
            [_node(1, 0, "Ancient", [(0, 1), (2, 1), (4, 1)])],
            [
                _node(0, 1, "Monster", [(0, 2)]),
                _node(2, 1, "Monster", [(2, 2)]),
                _node(4, 1, "Monster", [(4, 2)]),
            ],
            [
                _node(0, 2, "Monster", [(0, 3)]),
                _node(2, 2, "RestSite", [(2, 3)]),
                _node(4, 2, "Shop", [(4, 3)]),
            ],
            [
                _node(0, 3, "RestSite", [(1, 4)]),
                _node(2, 3, "Monster", [(3, 4)]),
                _node(4, 3, "Shop", [(5, 4)]),
            ],
            [
                _node(1, 4, "RestSite", [(2, 5)]),
                _node(3, 4, "RestSite", [(2, 5)]),
                _node(5, 4, "RestSite", [(2, 5)]),
            ],
        ],
        "boss": {"col": 2, "row": 5, "type": "Boss"},
    }


def test_max_restsites_low_risk_keeps_campfires_primary_then_breaks_risk_ties():
    choices = [
        {"col": 0, "row": 1, "type": "Monster"},
        {"col": 2, "row": 1, "type": "Monster"},
        {"col": 4, "row": 1, "type": "Monster"},
    ]
    state = {"decision": "map_select", "choices": choices}
    legal_actions = build_legal_actions(state)

    ranked = rank_route_candidates(
        enumerate_route_candidates(_risk_tiebreaker_map(), choices, legal_actions=legal_actions),
        objective="max_restsites_low_risk",
    )

    assert ranked[0].restsite_count == 2
    assert ranked[0].pre_first_restsite_combat_count == 1
    assert route_to_text(ranked[0]).startswith("Monster(2,1) -> RestSite(2,2)")
