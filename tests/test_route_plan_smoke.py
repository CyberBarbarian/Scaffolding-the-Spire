from sts2_bench.actions import build_legal_actions
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


def _map_state(*, choices: list[dict], full_map: dict | None = None) -> dict:
    return {
        "decision": "map_select",
        "choices": choices,
        "player": {"hp": 70, "max_hp": 80},
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
