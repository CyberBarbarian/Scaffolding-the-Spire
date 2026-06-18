from sts2_bench.actions import build_legal_actions
from sts2_bench.route_plan import RoutePlanTracker
from sts2_bench.route_plan_smoke import RoutePlanSmokeAgent


def _full_map() -> dict:
    return {
        "type": "map",
        "current_coord": {"col": 0, "row": 0},
        "rows": [
            [
                {
                    "col": 0,
                    "row": 0,
                    "type": "Ancient",
                    "children": [{"col": 0, "row": 1}],
                }
            ],
            [
                {
                    "col": 0,
                    "row": 1,
                    "type": "Monster",
                    "children": [{"col": 1, "row": 2}],
                }
            ],
            [{"col": 1, "row": 2, "type": "RestSite", "children": []}],
        ],
        "boss": {"col": 0, "row": 3, "type": "Boss"},
    }


def _map_state(*, choices: list[dict], full_map: dict | None = None, hp: int = 70, gold: int = 99) -> dict:
    return {
        "decision": "map_select",
        "choices": choices,
        "player": {"hp": hp, "max_hp": 80, "gold": gold, "deck_size": 10},
        "act": 1,
        "floor": 1,
        "context": {"act": 1, "floor": 1, "room_type": "Map"},
        "full_map": full_map or _full_map(),
    }


def test_smoke_agent_creates_plan_then_follows_planned_node():
    agent = RoutePlanSmokeAgent()
    first_state = _map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}])
    first_actions = build_legal_actions(first_state)

    first_action, first_meta = agent.choose(first_state, first_actions)

    route_event = first_meta["parsed"]["route_plan_event"]
    assert first_action.kind == "map"
    assert route_event["event_type"] == "route_plan_created"
    assert route_event["new_plan"] == [
        {"col": 0, "row": 1, "type": "Monster"},
        {"col": 1, "row": 2, "type": "RestSite"},
    ]

    second_state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "RestSite"}],
        full_map={
            **_full_map(),
            "current_coord": {"col": 0, "row": 1},
        },
    )
    second_actions = build_legal_actions(second_state)

    second_action, second_meta = agent.choose(second_state, second_actions)

    assert second_action.kind == "map"
    assert second_action.command["args"] == {"col": 1, "row": 2}
    assert "route_plan_event" not in second_meta["parsed"]


def _realistic_branching_map(*, current: tuple[int, int] = (3, 0)) -> dict:
    rows = [
        [
            _node(3, 0, "Ancient", [(1, 1), (3, 1), (5, 1)]),
        ],
        [
            _node(1, 1, "Monster", [(0, 2), (2, 2)]),
            _node(3, 1, "Monster", [(2, 2), (4, 2)]),
            _node(5, 1, "Shop", [(4, 2), (6, 2)]),
        ],
        [
            _node(0, 2, "Elite", [(1, 3)]),
            _node(2, 2, "Monster", [(1, 3), (3, 3)]),
            _node(4, 2, "RestSite", [(3, 3), (5, 3)]),
            _node(6, 2, "Treasure", [(5, 3)]),
        ],
        [
            _node(1, 3, "Elite", [(2, 4)]),
            _node(3, 3, "Monster", [(2, 4), (4, 4)]),
            _node(5, 3, "Shop", [(4, 4)]),
        ],
        [
            _node(2, 4, "RestSite", [(3, 5)]),
            _node(4, 4, "Monster", [(3, 5)]),
        ],
        [
            _node(3, 5, "Boss", []),
        ],
    ]
    return {
        "type": "map",
        "current_coord": {"col": current[0], "row": current[1]},
        "rows": rows,
        "boss": {"col": 3, "row": 5, "type": "Boss"},
    }


def _node(col: int, row: int, room_type: str, children: list[tuple[int, int]]) -> dict:
    return {
        "col": col,
        "row": row,
        "type": room_type,
        "children": [{"col": child_col, "row": child_row} for child_col, child_row in children],
    }


def _choose_smoke_route(state: dict):
    agent = RoutePlanSmokeAgent()
    legal_actions = build_legal_actions(state)
    action, meta = agent.choose(state, legal_actions)
    route_event = meta["parsed"]["route_plan_event"]
    result = RoutePlanTracker().validate_and_apply(
        state=state,
        legal_actions=legal_actions,
        action=action,
        meta=meta,
        step=0,
    )
    assert result.error is None
    assert result.events
    return action, route_event, result.events[0]


def test_smoke_agent_generates_validator_legal_complete_routes_from_different_map_nodes():
    starts = [
        ((3, 0), [{"col": 1, "row": 1, "type": "Monster"}, {"col": 3, "row": 1, "type": "Monster"}, {"col": 5, "row": 1, "type": "Shop"}]),
        ((3, 1), [{"col": 2, "row": 2, "type": "Monster"}, {"col": 4, "row": 2, "type": "RestSite"}]),
        ((4, 2), [{"col": 3, "row": 3, "type": "Monster"}, {"col": 5, "row": 3, "type": "Shop"}]),
    ]

    for current, choices in starts:
        state = _map_state(
            choices=choices,
            full_map=_realistic_branching_map(current=current),
            hp=64,
            gold=160,
        )

        _action, route_event, logged = _choose_smoke_route(state)

        assert route_event["new_plan"][-1] == {"col": 3, "row": 5, "type": "Boss"}
        assert logged["route_validity"] == "topological"
        assert logged["new_plan"] == route_event["new_plan"]


def test_smoke_agent_route_choice_changes_with_hp_and_gold_on_same_map():
    choices = [
        {"col": 1, "row": 1, "type": "Monster"},
        {"col": 3, "row": 1, "type": "Monster"},
        {"col": 5, "row": 1, "type": "Shop"},
    ]
    full_map = _realistic_branching_map(current=(3, 0))

    healthy_state = _map_state(choices=choices, full_map=full_map, hp=72, gold=30)
    low_hp_state = _map_state(choices=choices, full_map=full_map, hp=18, gold=30)
    rich_state = _map_state(choices=choices, full_map=full_map, hp=56, gold=220)

    healthy_action, healthy_event, _healthy_logged = _choose_smoke_route(healthy_state)
    low_hp_action, low_hp_event, _low_hp_logged = _choose_smoke_route(low_hp_state)
    rich_action, rich_event, _rich_logged = _choose_smoke_route(rich_state)

    assert healthy_action.command["args"] == {"col": 1, "row": 1}
    assert [node["type"] for node in healthy_event["new_plan"]].count("Elite") >= 2
    assert low_hp_action.command["args"] == {"col": 3, "row": 1}
    assert "RestSite" in [node["type"] for node in low_hp_event["new_plan"][:3]]
    assert rich_action.command["args"] == {"col": 5, "row": 1}
    assert rich_event["new_plan"][0]["type"] == "Shop"
