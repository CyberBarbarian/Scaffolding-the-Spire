import json

from sts2_bench.actions import LegalAction
from sts2_bench.agents import OpenAICompatAgent, build_llm_prompt
from sts2_bench.multi_agent import CoordinatorAgent
from sts2_bench.route_plan import ActiveRoutePlan, RoutePlanTracker
from sts2_bench.run_context import RunContextStore
from sts2_bench.runner import JsonlLogger, RouteCandidatePlannerAgent, agent_from_args, run_one


def _map_action(action_id, room_type, col, row):
    return LegalAction(
        action_id=action_id,
        label=f"go to {room_type} at col={col} row={row}",
        command={"cmd": "action", "action": "select_map_node", "args": {"col": col, "row": row}},
        kind="map",
    )


def _map_state(*, choices, full_map=None, hp=80, act=1, floor=1, relics=None):
    return {
        "decision": "map_select",
        "act": act,
        "floor": floor,
        "context": {"act": act, "floor": floor, "room_type": "Map"},
        "player": {"hp": hp, "max_hp": 80, "gold": 99, "deck_size": 10, "relics": relics or []},
        "choices": choices,
        **({"full_map": full_map} if full_map is not None else {}),
    }


def _json_block_after(prompt, marker, next_marker):
    payload = prompt.split(marker + "\n", 1)[1].split("\n\n" + next_marker, 1)[0]
    return json.loads(payload)


def _full_map(*, current=(0, 0), edges=None, nodes=None):
    edges = edges or {}
    nodes = nodes or {
        (0, 0): "Ancient",
        (0, 1): "Monster",
        (1, 2): "Elite",
        (2, 1): "Shop",
        (2, 2): "RestSite",
    }
    rows_by_row = {}
    for (col, row), room_type in nodes.items():
        rows_by_row.setdefault(row, []).append(
            {
                "col": col,
                "row": row,
                "type": room_type,
                "children": [
                    {"col": child_col, "row": child_row}
                    for child_col, child_row in edges.get((col, row), [])
                ],
            }
        )
    return {
        "type": "map",
        "rows": [rows_by_row[row] for row in sorted(rows_by_row)],
        "current_coord": {"col": current[0], "row": current[1]},
    }


def _event(event_type, plan, **extra):
    payload = {
        "type": event_type,
        "new_plan": [{"col": col, "row": row, "type": room_type} for col, row, room_type in plan],
        "reason_tags": extra.pop("reason_tags", ["test"]),
        "reason": extra.pop("reason", f"{event_type} for test"),
    }
    payload.update(extra)
    return payload


def test_route_plan_created_records_topological_plan_and_remaining_route():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]}),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )

    assert result.error is None
    assert len(result.events) == 1
    event = result.events[0]
    assert event["event_type"] == "route_plan_created"
    assert event["route_validity"] == "topological"
    assert event["selected_node"] == {"col": 0, "row": 1, "type": "Monster"}
    assert event["active_plan"]["remaining"] == [{"col": 1, "row": 2, "type": "Elite"}]


def test_forced_map_action_invalidates_active_plan_when_only_node_differs():
    tracker = RoutePlanTracker()
    tracker.active = ActiveRoutePlan(
        plan_id="rp_001",
        remaining=[
            {"col": 1, "row": 2, "type": "Elite"},
            {"col": 3, "row": 3, "type": "Boss"},
        ],
        map_revision=None,
        act=1,
    )
    state = _map_state(
        choices=[{"col": 2, "row": 2, "type": "RestSite"}],
        full_map=_full_map(
            current=(0, 1),
            nodes={
                (0, 1): "Monster",
                (2, 2): "RestSite",
                (3, 3): "Boss",
            },
            edges={(0, 1): [(2, 2)], (2, 2): [(3, 3)]},
        ),
        floor=2,
    )
    action = _map_action(0, "RestSite", 2, 2)

    effect = tracker.apply_forced_map_action(state=state, action=action)

    assert effect["outcome"] == "invalidated_active_plan"
    assert effect["invalidation_reason"] == "forced_node_differs_from_expected"
    assert effect["previous_plan_id"] == "rp_001"
    assert tracker.active is None


def test_route_candidate_replan_without_authoritative_active_plan_emits_created():
    planner = RouteCandidatePlannerAgent(
        OpenAICompatAgent(base_url="http://127.0.0.1:1/v1", model="fake"),
        objective="max_restsites",
    )
    planner.force_replan = True
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]}),
    )

    chosen, meta = planner._choose_map(state, [action])

    assert chosen == action
    assert meta["parsed"]["route_plan_event"]["event_type"] == "route_plan_created"


def test_route_candidate_follows_authoritative_active_plan_when_private_cache_is_empty():
    planner = RouteCandidatePlannerAgent(
        OpenAICompatAgent(base_url="http://127.0.0.1:1/v1", model="fake"),
        objective="max_restsites",
    )
    planner.planned_remaining = []
    elite = _map_action(0, "Elite", 1, 2)
    state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "Elite"}],
        full_map=_full_map(current=(0, 1), edges={(0, 1): [(1, 2)]}),
    )
    state["active_route_plan"] = {
        "plan_id": "rp_001",
        "expected_next": {"col": 1, "row": 2, "type": "Elite"},
        "remaining": [{"col": 1, "row": 2, "type": "Elite"}],
    }

    chosen, meta = planner._choose_map(state, [elite])

    assert chosen == elite
    assert meta["route_candidate_followed"] is True
    assert "route_plan_event" not in meta["parsed"]


def test_route_candidate_replan_preserves_map_changed_reason_from_validation_feedback():
    planner = RouteCandidatePlannerAgent(
        OpenAICompatAgent(base_url="http://127.0.0.1:1/v1", model="fake"),
        objective="max_restsites",
    )
    old_action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]}),
    )
    state["active_route_plan"] = {
        "plan_id": "rp_001",
        "expected_next": {"col": 0, "row": 1, "type": "Monster"},
        "remaining": [{"col": 0, "row": 1, "type": "Monster"}],
    }
    planner.receive_external_feedback(
        state,
        old_action,
        {},
        "route-plan validation failed [route_plan_invalid_after_map_change]",
        [old_action],
    )

    chosen, meta = planner._choose_map(state, [old_action])

    assert chosen == old_action
    event = meta["parsed"]["route_plan_event"]
    assert event["event_type"] == "route_plan_updated"
    assert "map_changed" in event["reason_tags"]


def test_route_plan_created_rejects_incomplete_visible_suffix():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]}),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster")])}},
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_incomplete"


def test_route_plan_created_rejects_partial_segment_until_visible_branch():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(
            nodes={
                (0, 0): "Ancient",
                (0, 1): "Monster",
                (0, 2): "Shop",
                (0, 3): "Unknown",
                (0, 4): "Monster",
                (1, 5): "Elite",
                (2, 5): "RestSite",
            },
            edges={
                (0, 0): [(0, 1)],
                (0, 1): [(0, 2)],
                (0, 2): [(0, 3)],
                (0, 3): [(0, 4)],
                (0, 4): [(1, 5), (2, 5)],
            },
        ),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (0, 2, "Shop")])}},
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_incomplete"


def test_route_plan_created_rejects_invalid_future_suffix_in_visible_map():
    tracker = RoutePlanTracker()
    action = _map_action(5, "Monster", 1, 1)
    state = _map_state(
        choices=[{"col": 1, "row": 1, "type": "Monster"}, {"col": 6, "row": 1, "type": "Monster"}],
        full_map=_full_map(
            current=(3, 0),
            nodes={
                (3, 0): "Ancient",
                (1, 1): "Monster",
                (6, 1): "Monster",
                (0, 2): "Unknown",
                (2, 2): "Monster",
                (6, 2): "Monster",
            },
            edges={(3, 0): [(1, 1), (6, 1)], (1, 1): [(0, 2), (2, 2)], (6, 1): [(6, 2)]},
        ),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action, _map_action(6, "Monster", 6, 1)],
        action=action,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(1, 1, "Monster"), (1, 2, "Event"), (1, 3, "Monster")],
                    reason_tags=["start_act2", "elite_hunting"],
                    reason="Plan the left path, but hallucinate future hidden nodes.",
                )
            }
        },
        step=195,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_topology_invalid"


def test_route_plan_rejects_node_type_that_does_not_match_route_graph():
    tracker = RoutePlanTracker()
    action = _map_action(5, "Monster", 1, 1)
    state = _map_state(
        choices=[{"col": 1, "row": 1, "type": "Monster"}],
        full_map=_full_map(
            current=(3, 0),
            nodes={(3, 0): "Ancient", (1, 1): "Monster", (0, 2): "Unknown"},
            edges={(3, 0): [(1, 1)], (1, 1): [(0, 2)]},
        ),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(1, 1, "Monster"), (0, 2, "Event")],
                    reason="Rename Unknown as Event.",
                )
            }
        },
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_topology_invalid"
    assert "exact type" in result.error.message


def test_route_plan_created_requires_reason():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]}),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(0, 1, "Monster"), (1, 2, "Elite")],
                    reason="",
                )
            }
        },
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_reason_required"


def test_route_plan_missing_created_event_rejected_when_no_active_plan():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]}),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={"parsed": {"reason": "go to monster"}},
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_missing_created_event"


def test_route_plan_execute_current_plan_consumes_next_node_without_logging_event():
    tracker = RoutePlanTracker()
    first_action = _map_action(0, "Monster", 0, 1)
    first_state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]}),
    )
    tracker.validate_and_apply(
        state=first_state,
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    next_action = _map_action(0, "Elite", 1, 2)
    next_state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "Elite"}],
        full_map=_full_map(current=(0, 1), edges={(0, 1): [(1, 2)]}),
        hp=62,
    )

    result = tracker.validate_and_apply(
        state=next_state,
        legal_actions=[next_action],
        action=next_action,
        meta={"parsed": {"reason": "follow plan"}},
        step=1,
    )

    assert result.error is None
    assert result.events == []
    assert tracker.active is None


def test_route_plan_requires_topological_route_without_winged_boots():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(
            nodes={(0, 0): "Ancient", (0, 1): "Monster", (2, 2): "RestSite"},
            edges={(0, 0): [(0, 1)]},
        ),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (2, 2, "RestSite")])}},
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_topology_invalid"


def test_route_plan_allows_up_to_three_forward_jumps_with_winged_boots():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        relics=[{"name": "Winged Boots"}],
        full_map=_full_map(
            nodes={
                (0, 0): "Ancient",
                (0, 1): "Monster",
                (2, 2): "RestSite",
                (1, 3): "Boss",
            },
            edges={(0, 0): [(0, 1)]},
        ),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(0, 1, "Monster"), (2, 2, "RestSite"), (1, 3, "Boss")],
                )
            }
        },
        step=0,
    )

    assert result.error is None
    assert result.events[0]["route_validity"] == "winged_boots_override"
    assert result.events[0]["winged_boots_jumps_used"] == 2
    assert result.events[0]["active_plan"]["remaining"] == [
        {"col": 2, "row": 2, "type": "RestSite"},
        {"col": 1, "row": 3, "type": "Boss"},
    ]


def test_route_plan_rejects_forward_jumps_over_winged_boots_remaining_counter():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        relics=[{"id": "winged_boots", "counter": 2}],
        full_map=_full_map(
            nodes={
                (0, 0): "Ancient",
                (0, 1): "Monster",
                (2, 2): "RestSite",
                (4, 3): "Shop",
                (1, 4): "Boss",
            },
            edges={(0, 0): [(0, 1)]},
        ),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(0, 1, "Monster"), (2, 2, "RestSite"), (4, 3, "Shop"), (1, 4, "Boss")],
                )
            }
        },
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_topology_invalid"
    assert "at most 2" in result.error.message


def test_route_plan_rejects_more_than_three_forward_jumps_with_winged_boots():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        relics=[{"id": "winged_boots"}],
        full_map=_full_map(
            nodes={
                (0, 0): "Ancient",
                (0, 1): "Monster",
                (2, 2): "RestSite",
                (4, 3): "Shop",
                (1, 4): "Treasure",
                (3, 5): "Boss",
            },
            edges={(0, 0): [(0, 1)]},
        ),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [
                        (0, 1, "Monster"),
                        (2, 2, "RestSite"),
                        (4, 3, "Shop"),
                        (1, 4, "Treasure"),
                        (3, 5, "Boss"),
                    ],
                )
            }
        },
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_topology_invalid"
    assert "at most three" in result.error.message


def test_route_plan_rejects_non_forward_winged_boots_jumps():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        relics=[{"name": "Winged Boots"}],
        full_map=_full_map(
            nodes={
                (0, 0): "Ancient",
                (0, 1): "Monster",
                (2, 1): "Shop",
                (1, 2): "Boss",
            },
            edges={(0, 0): [(0, 1)]},
        ),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(0, 1, "Monster"), (2, 1, "Shop"), (1, 2, "Boss")],
                )
            }
        },
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_topology_invalid"
    assert "move forward" in result.error.message


def test_route_plan_deviation_without_update_is_rejected():
    tracker = RoutePlanTracker()
    first_action = _map_action(0, "Monster", 0, 1)
    first_state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2), (2, 2)]}),
    )
    tracker.validate_and_apply(
        state=first_state,
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    elite = _map_action(0, "Elite", 1, 2)
    rest = _map_action(1, "RestSite", 2, 2)
    next_state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "Elite"}, {"col": 2, "row": 2, "type": "RestSite"}],
        full_map=_full_map(current=(0, 1), edges={(0, 1): [(1, 2), (2, 2)]}),
        hp=18,
    )

    result = tracker.validate_and_apply(
        state=next_state,
        legal_actions=[elite, rest],
        action=rest,
        meta={"parsed": {"reason": "rest is safer"}},
        step=1,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_deviated_without_update"
    assert "emit route_plan_updated" in result.error.feedback


def test_route_plan_update_records_reason_and_replaces_remaining_plan():
    tracker = RoutePlanTracker()
    first_action = _map_action(0, "Monster", 0, 1)
    first_state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2), (2, 2)]}),
    )
    tracker.validate_and_apply(
        state=first_state,
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    elite = _map_action(0, "Elite", 1, 2)
    rest = _map_action(1, "RestSite", 2, 2)
    next_state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "Elite"}, {"col": 2, "row": 2, "type": "RestSite"}],
        full_map=_full_map(current=(0, 1), edges={(0, 1): [(1, 2), (2, 2)]}),
        hp=18,
    )

    result = tracker.validate_and_apply(
        state=next_state,
        legal_actions=[elite, rest],
        action=rest,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_updated",
                    [(2, 2, "RestSite")],
                    reason_tags=["low_hp", "avoid_elite"],
                    reason="HP dropped to 18/80, so avoid the elite.",
                )
            }
        },
        step=1,
    )

    assert result.error is None
    event = result.events[0]
    assert event["event_type"] == "route_plan_updated"
    assert event["old_plan_remaining"] == [{"col": 1, "row": 2, "type": "Elite"}]
    assert event["reason_tags"] == ["low_hp", "avoid_elite"]
    assert event["active_plan"]["remaining"] == []


def test_route_plan_update_requires_explicit_new_plan():
    tracker = RoutePlanTracker()
    first_action = _map_action(0, "Monster", 0, 1)
    first_state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(
            nodes={(0, 0): "Ancient", (0, 1): "Monster", (1, 2): "Elite", (2, 2): "RestSite", (3, 3): "Boss"},
            edges={(0, 0): [(0, 1)], (0, 1): [(1, 2), (2, 2)], (1, 2): [(3, 3)], (2, 2): [(3, 3)]},
        ),
    )
    tracker.validate_and_apply(
        state=first_state,
        legal_actions=[first_action],
        action=first_action,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(0, 1, "Monster"), (1, 2, "Elite"), (3, 3, "Boss")],
                )
            }
        },
        step=0,
    )
    elite = _map_action(0, "Elite", 1, 2)
    rest = _map_action(1, "RestSite", 2, 2)
    next_state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "Elite"}, {"col": 2, "row": 2, "type": "RestSite"}],
        full_map=_full_map(
            current=(0, 1),
            nodes={(0, 1): "Monster", (1, 2): "Elite", (2, 2): "RestSite", (3, 3): "Boss"},
            edges={(0, 1): [(1, 2), (2, 2)], (1, 2): [(3, 3)], (2, 2): [(3, 3)]},
        ),
        hp=18,
    )

    result = tracker.validate_and_apply(
        state=next_state,
        legal_actions=[elite, rest],
        action=rest,
        meta={
            "parsed": {
                "route_plan_event": {
                    "type": "route_plan_updated",
                    "reason_tags": ["low_hp", "avoid_elite"],
                    "reason": "HP dropped to 18/80, so change to the rest route.",
                }
            }
        },
        step=1,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_empty"


def test_route_plan_update_requires_reason_for_route_change():
    tracker = RoutePlanTracker()
    first_action = _map_action(0, "Monster", 0, 1)
    first_state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2), (2, 2)]}),
    )
    tracker.validate_and_apply(
        state=first_state,
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    rest = _map_action(1, "RestSite", 2, 2)
    next_state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "Elite"}, {"col": 2, "row": 2, "type": "RestSite"}],
        full_map=_full_map(current=(0, 1), edges={(0, 1): [(1, 2), (2, 2)]}),
        hp=18,
    )

    result = tracker.validate_and_apply(
        state=next_state,
        legal_actions=[_map_action(0, "Elite", 1, 2), rest],
        action=rest,
        meta={"parsed": {"route_plan_event": {"type": "route_plan_updated", "reason_tags": ["low_hp"]}}},
        step=1,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_update_reason_required"


def test_empty_route_plan_event_placeholder_is_treated_as_omitted_when_following_active_route():
    tracker = RoutePlanTracker()
    first_action = _map_action(0, "Monster", 0, 1)
    snapshot = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]})
    tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=snapshot),
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    elite = _map_action(0, "Elite", 1, 2)

    result = tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 1, "row": 2, "type": "Elite"}], full_map=snapshot),
        legal_actions=[elite],
        action=elite,
        meta={"parsed": {"route_plan_event": {}}},
        step=1,
    )

    assert result.error is None
    assert result.events == []


def test_redundant_created_event_is_rejected_without_consuming_active_route():
    tracker = RoutePlanTracker()
    snapshot = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]})
    first_action = _map_action(0, "Monster", 0, 1)
    tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=snapshot),
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    before = tracker.active.to_dict()
    elite = _map_action(0, "Elite", 1, 2)

    result = tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 1, "row": 2, "type": "Elite"}], full_map=snapshot),
        legal_actions=[elite],
        action=elite,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(1, 2, "Elite")],
                    reason="A second creation event must not replace or silently consume the active plan.",
                )
            }
        },
        step=1,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_unexpected_event_type"
    assert tracker.active.to_dict() == before


def test_map_revision_with_legal_next_but_invalid_tail_requires_update_without_consuming():
    tracker = RoutePlanTracker()
    old = _full_map(
        nodes={(0, 0): "Ancient", (0, 1): "Monster", (1, 2): "Elite", (1, 3): "Boss"},
        edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)], (1, 2): [(1, 3)]},
    )
    first = _map_action(0, "Monster", 0, 1)
    created = tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=old),
        legal_actions=[first],
        action=first,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(0, 1, "Monster"), (1, 2, "Elite"), (1, 3, "Boss")],
                )
            }
        },
        step=0,
    )
    assert created.error is None
    before = tracker.active.to_dict()

    changed = _full_map(
        current=(0, 1),
        nodes={(0, 1): "Monster", (1, 2): "Elite", (2, 3): "Boss"},
        edges={(0, 1): [(1, 2)], (1, 2): [(2, 3)]},
    )
    elite = _map_action(0, "Elite", 1, 2)
    stale = tracker.validate_and_apply(
        state=_map_state(
            choices=[{"col": 1, "row": 2, "type": "Elite"}],
            full_map=changed,
            floor=2,
        ),
        legal_actions=[elite],
        action=elite,
        meta={"parsed": {}},
        step=1,
    )

    assert stale.error is not None
    assert stale.error.code == "route_plan_invalid_after_map_change"
    assert tracker.active.to_dict() == before

    updated = tracker.validate_and_apply(
        state=_map_state(
            choices=[{"col": 1, "row": 2, "type": "Elite"}],
            full_map=changed,
            floor=2,
        ),
        legal_actions=[elite],
        action=elite,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_updated",
                    [(1, 2, "Elite"), (2, 3, "Boss")],
                    reason_tags=["map_changed"],
                )
            }
        },
        step=1,
    )

    assert updated.error is None
    assert tracker.active.remaining == [{"col": 2, "row": 3, "type": "Boss"}]


def test_benign_map_revision_keeps_valid_silent_follow_behavior():
    tracker = RoutePlanTracker()
    old = _full_map(
        nodes={(0, 0): "Ancient", (0, 1): "Monster", (1, 2): "Elite", (1, 3): "Boss"},
        edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)], (1, 2): [(1, 3)]},
    )
    first = _map_action(0, "Monster", 0, 1)
    tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=old),
        legal_actions=[first],
        action=first,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(0, 1, "Monster"), (1, 2, "Elite"), (1, 3, "Boss")],
                )
            }
        },
        step=0,
    )
    benign = _full_map(
        current=(0, 1),
        nodes={
            (0, 1): "Monster",
            (1, 2): "Elite",
            (2, 2): "Shop",
            (1, 3): "Boss",
        },
        edges={(0, 1): [(1, 2), (2, 2)], (1, 2): [(1, 3)], (2, 2): [(1, 3)]},
    )
    elite = _map_action(0, "Elite", 1, 2)

    result = tracker.validate_and_apply(
        state=_map_state(
            choices=[{"col": 1, "row": 2, "type": "Elite"}],
            full_map=benign,
            floor=2,
        ),
        legal_actions=[elite],
        action=elite,
        meta={"parsed": {}},
        step=1,
    )

    assert result.error is None
    assert result.events == []
    assert tracker.active.remaining == [{"col": 1, "row": 3, "type": "Boss"}]


def test_blank_route_plan_event_type_is_rejected_when_following_active_route():
    tracker = RoutePlanTracker()
    first_action = _map_action(0, "Monster", 0, 1)
    snapshot = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]})
    tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=snapshot),
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    elite = _map_action(0, "Elite", 1, 2)

    result = tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 1, "row": 2, "type": "Elite"}], full_map=snapshot),
        legal_actions=[elite],
        action=elite,
        meta={"parsed": {"route_plan_event": {"type": ""}}},
        step=1,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_unexpected_event_type"


def test_route_plan_invalid_next_not_legal_rejects_when_map_revision_is_unchanged():
    tracker = RoutePlanTracker()
    snapshot = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]})
    first_action = _map_action(0, "Monster", 0, 1)
    tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=snapshot),
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    rest = _map_action(0, "RestSite", 2, 2)

    result = tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 2, "row": 2, "type": "RestSite"}], full_map=snapshot),
        legal_actions=[rest],
        action=rest,
        meta={"parsed": {"route_plan_event": _event("route_plan_updated", [(2, 2, "RestSite")])}},
        step=1,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_invalid_next_not_legal"


def test_route_plan_update_allowed_when_map_revision_changes_and_new_plan_is_legal():
    tracker = RoutePlanTracker()
    first_snapshot = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]})
    first_action = _map_action(0, "Monster", 0, 1)
    tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=first_snapshot),
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")])}},
        step=0,
    )
    changed_snapshot = _full_map(current=(0, 1), edges={(0, 1): [(2, 2)]})
    rest = _map_action(0, "RestSite", 2, 2)

    result = tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 2, "row": 2, "type": "RestSite"}], full_map=changed_snapshot),
        legal_actions=[rest],
        action=rest,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_updated",
                    [(2, 2, "RestSite")],
                    reason_tags=["map_changed", "new_route_revealed"],
                )
            }
        },
        step=1,
    )

    assert result.error is None
    assert result.events[0]["trigger"] == "map_changed"


def test_route_plan_update_requires_map_changed_tag_when_revision_invalidates_old_tail():
    tracker = RoutePlanTracker()
    original = _full_map(
        nodes={(0, 0): "Ancient", (0, 1): "Monster", (1, 2): "Elite", (1, 3): "Boss"},
        edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)], (1, 2): [(1, 3)]},
    )
    first = _map_action(0, "Monster", 0, 1)
    tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=original),
        legal_actions=[first],
        action=first,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(0, 1, "Monster"), (1, 2, "Elite"), (1, 3, "Boss")],
                )
            }
        },
        step=0,
    )
    before = tracker.active.to_dict()
    changed = _full_map(
        current=(0, 1),
        nodes={
            (0, 1): "Monster",
            (1, 2): "Elite",
            (2, 3): "Boss",
        },
        edges={(0, 1): [(1, 2)], (1, 2): [(2, 3)]},
    )
    elite = _map_action(0, "Elite", 1, 2)
    without_tag = tracker.validate_and_apply(
        state=_map_state(
            choices=[{"col": 1, "row": 2, "type": "Elite"}],
            full_map=changed,
            floor=2,
        ),
        legal_actions=[elite],
        action=elite,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_updated",
                    [(1, 2, "Elite"), (2, 3, "Boss")],
                    reason_tags=["test"],
                )
            }
        },
        step=1,
    )

    assert without_tag.error is not None
    assert without_tag.error.code == "route_plan_update_missing_map_changed_reason"
    assert tracker.active.to_dict() == before

    with_tag = tracker.validate_and_apply(
        state=_map_state(
            choices=[{"col": 1, "row": 2, "type": "Elite"}],
            full_map=changed,
            floor=2,
        ),
        legal_actions=[elite],
        action=elite,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_updated",
                    [(1, 2, "Elite"), (2, 3, "Boss")],
                    reason_tags=["map_changed"],
                )
            }
        },
        step=1,
    )

    assert with_tag.error is None


def test_route_plan_created_allowed_when_map_revision_changes_and_stale_next_coord_is_legal():
    tracker = RoutePlanTracker()
    first_snapshot = _full_map(
        current=(0, 0),
        nodes={(0, 0): "Ancient", (1, 1): "Monster", (0, 2): "Elite"},
        edges={(0, 0): [(1, 1)], (1, 1): [(0, 2)]},
    )
    first_action = _map_action(0, "Monster", 1, 1)
    tracker.validate_and_apply(
        state=_map_state(choices=[{"col": 1, "row": 1, "type": "Monster"}], full_map=first_snapshot),
        legal_actions=[first_action],
        action=first_action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(1, 1, "Monster"), (0, 2, "Elite")])}},
        step=0,
    )
    changed_snapshot = _full_map(
        current=(0, 1),
        nodes={(0, 1): "Ancient", (0, 2): "Monster", (2, 2): "Shop"},
        edges={(0, 1): [(0, 2), (2, 2)]},
    )
    selected = _map_action(5, "Shop", 2, 2)
    stale_next_is_still_legal = _map_action(6, "Monster", 0, 2)

    result = tracker.validate_and_apply(
        state=_map_state(
            choices=[{"col": 2, "row": 2, "type": "Shop"}, {"col": 0, "row": 2, "type": "Monster"}],
            full_map=changed_snapshot,
            act=2,
            floor=1,
        ),
        legal_actions=[selected, stale_next_is_still_legal],
        action=selected,
        meta={
            "parsed": {
                "route_plan_event": _event(
                    "route_plan_created",
                    [(2, 2, "Shop")],
                    reason_tags=["map_changed", "shop"],
                )
            }
        },
        step=1,
    )

    assert result.error is None
    assert result.events[0]["event_type"] == "route_plan_created"
    assert result.events[0]["old_plan_remaining"] == [{"col": 0, "row": 2, "type": "Elite"}]
    assert result.events[0]["trigger"] == "map_changed"
    assert result.events[0]["active_plan"]["remaining"] == []


def test_non_topological_but_legal_first_step_is_rejected_without_winged_boots():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Shop", 2, 1)
    state = _map_state(
        choices=[{"col": 2, "row": 1, "type": "Shop"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(2, 1, "Shop")])}},
        step=0,
    )

    assert result.error is not None
    assert result.error.code == "route_plan_topology_invalid"


def test_route_plan_event_json_is_serializable():
    tracker = RoutePlanTracker()
    action = _map_action(0, "Monster", 0, 1)
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[action],
        action=action,
        meta={"parsed": {"route_plan_event": _event("route_plan_created", [(0, 1, "Monster")])}},
        step=0,
    )

    json.dumps(result.events[0], ensure_ascii=False)


def test_map_prompt_instructs_llm_to_emit_route_plan_event():
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )
    action = _map_action(0, "Monster", 0, 1)

    prompt = build_llm_prompt(state, [action])

    assert "route_plan_event" in prompt
    assert "route_plan_created" in prompt
    assert "route_plan_updated" in prompt
    assert "route_plan_followed" not in prompt
    assert "omit route_plan_event" in prompt
    assert "new_plan" in prompt
    assert "reason_tags" in prompt
    assert 'reason_tags must include the separate string "map_changed"' in prompt
    assert "Canonical route map JSON:" in prompt
    assert "Route graph JSON:" not in prompt
    assert '"id":"r01c0"' in prompt
    assert '"symbol":"M"' in prompt
    assert "each adjacent pair in new_plan must follow an edge" in prompt
    assert "Route graph exact nodes:" not in prompt
    assert "Active route plan JSON:\nnull" in prompt
    assert "No formal active route plan exists" in prompt


def test_map_prompt_uses_canonical_route_map_representation_by_default():
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )
    action = _map_action(0, "Monster", 0, 1)

    prompt = build_llm_prompt(state, [action])

    assert "Canonical route map JSON:" in prompt
    assert "Route graph JSON:" not in prompt
    route_map = json.loads(prompt.split("Canonical route map JSON:\n", 1)[1])
    assert route_map["schema"] == "sts2_route_map_v1"
    assert route_map["legal_starts"] == [
        {"action_id": 0, "node": "r01c0", "symbol": "M", "type": "Monster", "choice_index": 0}
    ]
    assert {"id": "r01c0", "col": 0, "row": 1, "symbol": "M", "type": "Monster", "action_id": 0} in route_map["rows"][1]["nodes"]
    assert route_map["edges"]["r00c0"] == ["r01c0"]


def test_map_prompt_can_use_legacy_route_graph_representation_explicitly():
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )
    action = _map_action(0, "Monster", 0, 1)

    prompt = build_llm_prompt(state, [action], route_map_prompt_format="route_graph")

    assert "Route graph JSON:" in prompt
    assert "Canonical route map JSON:" not in prompt
    assert '"node_key":"0:1"' in prompt


def test_openai_agent_uses_canonical_route_map_prompt_format_by_default():
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )
    action = _map_action(0, "Monster", 0, 1)
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1/v1",
        model="dummy",
    )

    prompt = agent.build_prompt(state, [action])

    assert "Canonical route map JSON:" in prompt
    assert "Route graph JSON:" not in prompt


def test_agent_from_args_uses_canonical_route_map_prompt_format_by_default():
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )
    action = _map_action(0, "Monster", 0, 1)
    agent = agent_from_args(
        "llm",
        base_url="http://127.0.0.1:1/v1",
        model="dummy",
        api_key="local",
    )

    prompt = agent.build_prompt(state, [action])

    assert "Canonical route map JSON:" in prompt
    assert "Route graph JSON:" not in prompt


def test_map_prompt_can_omit_route_plan_instructions_when_disabled():
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )
    state["active_route_plan"] = {
        "plan_id": "stale",
        "remaining": [{"col": 0, "row": 1, "type": "Monster"}],
        "expected_next": {"col": 0, "row": 1, "type": "Monster"},
    }
    action = _map_action(0, "Monster", 0, 1)

    prompt = build_llm_prompt(state, [action], route_plan_enabled=False)

    assert "route_plan_event" not in prompt
    assert "route_plan_created" not in prompt
    assert "route_plan_updated" not in prompt
    assert "Canonical route map JSON:" in prompt
    assert "Persistent route planning is disabled." not in prompt
    assert "Route decision contract:" not in prompt
    assert "Active route plan JSON:" not in prompt
    assert "active_route_plan" not in prompt


def test_map_prompt_uses_structured_route_state_without_text_map_duplication():
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
        hp=18,
    )
    state["player"]["deck"] = [
        {"name": "Strike", "type": "Attack", "cost": 1},
        {"name": "Strike", "type": "Attack", "cost": 1},
        {"name": "Defend", "type": "Skill", "cost": 1},
    ]
    state["last_action_result"] = {
        "action_label": "go to Monster at col=0 row=1",
        "action_kind": "map",
        "decision_before": "map_select",
        "decision_after": "combat_play",
        "changes": ["hp: 45/80 -> 18/80", "room: Map -> Monster"],
    }
    state["active_route_plan"] = {
        "plan_id": "rp_001",
        "remaining": [{"col": 0, "row": 1, "type": "Monster"}],
        "expected_next": {"col": 0, "row": 1, "type": "Monster"},
    }
    action = _map_action(0, "Monster", 0, 1)

    prompt = build_llm_prompt(state, [action])
    route_state = _json_block_after(prompt, "Route decision state JSON:", "Legal actions:")

    assert "Route decision state JSON:" in prompt
    assert "Game state:" not in prompt
    assert "Full map:" not in prompt
    assert "Map choices (" not in prompt
    assert "\nActive route plan:\n" not in prompt
    assert "Compact state JSON:" not in prompt
    assert route_state["player"]["hp"] == 18
    assert route_state["deck_summary"]["visible"] is True
    assert route_state["deck_summary"]["groups"][0]["name"] == "Defend"
    assert route_state["recent_state_changes"]["hp_changes"] == ["hp: 45/80 -> 18/80"]


def test_map_prompt_includes_active_route_plan_context():
    state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "Elite"}, {"col": 2, "row": 2, "type": "RestSite"}],
        full_map=_full_map(current=(0, 1), edges={(0, 1): [(1, 2), (2, 2)]}),
    )
    state["active_route_plan"] = {
        "plan_id": "rp_001",
        "remaining": [{"col": 1, "row": 2, "type": "Elite"}],
        "expected_next": {"col": 1, "row": 2, "type": "Elite"},
    }

    prompt = build_llm_prompt(state, [_map_action(0, "Elite", 1, 2), _map_action(1, "RestSite", 2, 2)])

    assert "Active route plan JSON:" in prompt
    assert '"expected_next":{"col":1,"row":2,"type":"Elite"}' in prompt
    assert '"role":"advisory_current_state_feature"' in prompt
    assert "Active route plan is advisory context, not a binding commitment" in prompt
    assert "You can follow expected_next only when that exact node is currently selectable in legal_actions" in prompt
    assert 'reason_tags including the separate string "map_changed"' in prompt
    assert "choose from current state, legal_actions, route_graph, deck, relics, boss, and recent state changes" in prompt
    assert "select expected_next and omit route_plan_event" in prompt
    assert "route_plan_updated" in prompt
    assert "risk_score" not in prompt
    assert "stickiness" not in prompt


def test_map_changed_feedback_requires_explicit_map_changed_tag():
    state = _map_state(
        choices=[{"col": 1, "row": 2, "type": "Elite"}, {"col": 2, "row": 2, "type": "RestSite"}],
        full_map=_full_map(current=(0, 1), edges={(0, 1): [(1, 2), (2, 2)]}),
    )
    tracker = RoutePlanTracker()
    tracker.active = ActiveRoutePlan(
        plan_id="rp_001",
        remaining=[{"col": 3, "row": 3, "type": "Boss"}],
        map_revision="old_revision",
        act=1,
    )

    result = tracker.validate_and_apply(
        state=state,
        legal_actions=[_map_action(0, "Elite", 1, 2), _map_action(1, "RestSite", 2, 2)],
        action=_map_action(0, "Elite", 1, 2),
        meta={"parsed": {"action_id": 0, "reason": "try old route"}},
        step=7,
    )

    assert result.error is not None
    assert "route_plan_updated" in result.error.feedback
    assert 'separate string "map_changed"' in result.error.feedback


def test_runner_runtime_follows_active_route_without_second_model_decision(tmp_path):
    boss_nodes = {
        (0, 0): "Ancient",
        (0, 1): "Monster",
        (2, 1): "Shop",
        (1, 2): "Elite",
        (2, 2): "RestSite",
        (3, 3): "Boss",
    }
    first_map = _full_map(
        nodes=boss_nodes,
        edges={
            (0, 0): [(0, 1), (2, 1)],
            (0, 1): [(1, 2)],
            (2, 1): [(2, 2)],
            (1, 2): [(3, 3)],
            (2, 2): [(3, 3)],
        },
    )
    second_map = _full_map(
        current=(0, 1),
        nodes={key: value for key, value in boss_nodes.items() if key != (0, 0)},
        edges={
            (0, 1): [(1, 2)],
            (2, 1): [(2, 2)],
            (1, 2): [(3, 3)],
            (2, 2): [(3, 3)],
        },
    )
    process = _RoutePlanProcess(
        [
            _map_state(
                choices=[
                    {"col": 0, "row": 1, "type": "Monster"},
                    {"col": 2, "row": 1, "type": "Shop"},
                ],
                full_map=first_map,
            ),
            _map_state(choices=[{"col": 1, "row": 2, "type": "Elite"}], full_map=second_map, hp=62),
            {"decision": "game_over", "victory": False, "player": {"hp": 54, "max_hp": 80}},
        ]
    )
    agent = _RoutePlanScriptAgent(
        [
            {
                "action_index": 5,
                "parsed": {
                    "reason": "create route",
                    "route_plan_event": _event(
                        "route_plan_created",
                        [(0, 1, "Monster"), (1, 2, "Elite"), (3, 3, "Boss")],
                    ),
                },
            },
        ]
    )
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(
            agent=agent,
            seed="route-plan",
            process=process,
            logger=logger,
            log_level="full",
            max_steps=5,
        )
    finally:
        logger.close()

    assert result.error is None
    assert agent.calls == 1
    assert result.extra["forced_singleton_map_count"] == 1
    rows = _read_jsonl(out)
    map_decisions = [row for row in rows if row.get("type") == "decision" and row.get("decision") == "map_select"]
    assert len(map_decisions) == 2
    assert "runtime_takeover" not in map_decisions[0]
    assert map_decisions[1]["runtime_takeover"]["model_called"] is False
    assert "prompt" not in map_decisions[1]
    map_actions = [row for row in rows if row.get("type") == "action" and row.get("decision") == "map_select"]
    assert map_actions[1]["agent_meta"]["policy"] == "forced_singleton_map"
    assert map_actions[1]["agent_meta"]["route_plan_effect"]["outcome"] == "followed_active_plan"
    events = [row for row in rows if row.get("type") == "route_plan_event"]
    assert [event["event_type"] for event in events] == ["route_plan_created"]


class _RoutePlanScriptAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.feedback = []

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        response = self.responses[self.calls]
        self.calls += 1
        action = legal_actions[response["action_index"]]
        meta = {"agent": "route_plan_script", "parsed": dict(response.get("parsed") or {})}
        if "raw_response" in response:
            meta["raw_response"] = response["raw_response"]
        return action, meta

    def receive_external_feedback(self, state, action, meta, feedback, legal_actions):
        self.feedback.append(feedback)


class _RoutePlanProcess:
    def __init__(self, states):
        self.states = list(states)
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return self.states.pop(0)

    def send(self, command):
        self.sent.append(command)
        if command.get("cmd") == "get_map":
            return self.states[0]["full_map"]
        if self.states:
            return self.states.pop(0)
        return {"decision": "game_over", "victory": False, "player": {"hp": 60, "max_hp": 80}}

    def close(self):
        pass


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_runner_automatically_executes_singleton_map_without_agent_or_prompt(tmp_path):
    first_map = _full_map(edges={(0, 0): [(0, 1)]})
    process = _RoutePlanProcess(
        [
            _map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=first_map),
            {"decision": "game_over", "victory": False, "player": {"hp": 54, "max_hp": 80}},
        ]
    )
    agent = _RoutePlanScriptAgent([])
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(
            agent=agent,
            seed="route-plan",
            process=process,
            logger=logger,
            log_level="full",
            max_steps=2,
        )
    finally:
        logger.close()

    assert result.error is None
    assert agent.calls == 0
    assert process.sent == [
        {"cmd": "action", "action": "select_map_node", "args": {"col": 0, "row": 1}}
    ]
    rows = _read_jsonl(out)
    decision = next(row for row in rows if row.get("type") == "decision")
    assert decision["runtime_takeover"]["model_called"] is False
    assert "prompt" not in decision
    action = next(row for row in rows if row.get("type") == "action")
    assert action["agent_meta"]["agent"] == "runtime"
    assert action["agent_meta"]["model_called"] is False
    assert action["agent_meta"]["route_plan_effect"]["outcome"] == "no_active_plan"
    assert "parsed" not in action["agent_meta"]
    assert "run_context_patch" not in action["agent_meta"]
    events = [row for row in _read_jsonl(out) if row.get("type") == "route_plan_event"]
    assert events == []
    assert not any(row.get("type") == "route_plan_validation_error" for row in rows)


def test_runner_can_disable_route_plan_validation(tmp_path):
    full_map = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]})
    process = _RoutePlanProcess(
        [
            _map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=full_map),
            {"decision": "game_over", "victory": False, "player": {"hp": 54, "max_hp": 80}},
        ]
    )
    agent = _RoutePlanScriptAgent([{"action_index": 5, "parsed": {"reason": "choose map without plan"}}])
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(
            agent=agent,
            seed="route-plan-disabled",
            process=process,
            logger=logger,
            max_steps=2,
            route_plan_enabled=False,
        )
    finally:
        logger.close()

    assert result.error is None
    rows = _read_jsonl(out)
    assert not any(row.get("type") == "route_plan_validation_error" for row in rows)
    assert not any(row.get("type") == "route_plan_event" for row in rows)


def test_runner_reasks_when_map_choice_deviates_without_route_plan_update(tmp_path):
    boss_nodes = {
        (0, 0): "Ancient",
        (0, 1): "Monster",
        (2, 1): "Shop",
        (1, 2): "Elite",
        (2, 2): "RestSite",
        (3, 3): "Boss",
    }
    first_map = _full_map(
        nodes=boss_nodes,
        edges={
            (0, 0): [(0, 1), (2, 1)],
            (0, 1): [(1, 2), (2, 2)],
            (2, 1): [(2, 2)],
            (1, 2): [(3, 3)],
            (2, 2): [(3, 3)],
        },
    )
    second_map = _full_map(
        current=(0, 1),
        nodes={key: value for key, value in boss_nodes.items() if key != (0, 0)},
        edges={(0, 1): [(1, 2), (2, 2)], (1, 2): [(3, 3)], (2, 2): [(3, 3)]},
    )
    process = _RoutePlanProcess(
        [
            _map_state(
                choices=[
                    {"col": 0, "row": 1, "type": "Monster"},
                    {"col": 2, "row": 1, "type": "Shop"},
                ],
                full_map=first_map,
            ),
            _map_state(
                choices=[{"col": 1, "row": 2, "type": "Elite"}, {"col": 2, "row": 2, "type": "RestSite"}],
                full_map=second_map,
                hp=18,
            ),
            {"decision": "game_over", "victory": False, "player": {"hp": 38, "max_hp": 80}},
        ]
    )
    agent = _RoutePlanScriptAgent(
        [
            {
                "action_index": 5,
                "parsed": {
                    "reason": "create route",
                    "route_plan_event": _event(
                        "route_plan_created",
                        [(0, 1, "Monster"), (1, 2, "Elite"), (3, 3, "Boss")],
                    ),
                },
            },
            {"action_index": 6, "parsed": {"reason": "rest without saying why"}},
            {
                "action_index": 6,
                "parsed": {
                    "reason": "update route after feedback",
                    "route_plan_event": _event(
                        "route_plan_updated",
                        [(2, 2, "RestSite"), (3, 3, "Boss")],
                        reason_tags=["low_hp", "avoid_elite"],
                        reason="HP is low, so take the rest site instead of the elite.",
                    ),
                },
            },
        ]
    )
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(agent=agent, seed="route-plan", process=process, logger=logger, max_steps=5)
    finally:
        logger.close()

    assert result.error is None
    assert len(agent.feedback) == 1
    assert "route_plan_deviated_without_update" in agent.feedback[0]
    map_commands = [cmd for cmd in process.sent if cmd.get("action") == "select_map_node"]
    assert [cmd["args"] for cmd in map_commands] == [{"col": 0, "row": 1}, {"col": 2, "row": 2}]
    rows = _read_jsonl(out)
    assert any(row.get("type") == "route_plan_validation_error" for row in rows)
    events = [row for row in rows if row.get("type") == "route_plan_event"]
    assert [event["event_type"] for event in events] == ["route_plan_created", "route_plan_updated"]
    assert events[-1]["reason_tags"] == ["low_hp", "avoid_elite"]


def test_runner_rolls_back_rejected_context_patch_and_logs_raw_attempt(tmp_path):
    full_map = _full_map(
        nodes={
            (0, 0): "Ancient",
            (0, 1): "Monster",
            (2, 1): "Shop",
            (1, 2): "Elite",
            (3, 3): "Boss",
        },
        edges={
            (0, 0): [(0, 1), (2, 1)],
            (0, 1): [(1, 2)],
            (2, 1): [(3, 3)],
            (1, 2): [(3, 3)],
        },
    )
    process = _RoutePlanProcess(
        [
            _map_state(
                choices=[
                    {"col": 0, "row": 1, "type": "Monster"},
                    {"col": 2, "row": 1, "type": "Shop"},
                ],
                full_map=full_map,
            ),
            {"decision": "game_over", "victory": False, "player": {"hp": 70, "max_hp": 80}},
        ]
    )
    store = RunContextStore(formal_route_plan_enabled=True)
    occ = _RoutePlanScriptAgent(
        [
            {
                "action_index": 5,
                "raw_response": "RAW_INVALID_ROUTE_ATTEMPT",
                "parsed": {
                    "reason": "invalid attempt without a formal route",
                    "run_memory_facts": ["rejected fact must be rolled back"],
                },
            },
            {
                "action_index": 5,
                "raw_response": "RAW_VALID_ROUTE_ATTEMPT",
                "parsed": {
                    "reason": "create valid route after feedback",
                    "run_memory_facts": ["accepted fact remains"],
                    "route_plan_event": _event(
                        "route_plan_created",
                        [(0, 1, "Monster"), (1, 2, "Elite"), (3, 3, "Boss")],
                    ),
                },
            },
        ]
    )
    agent = CoordinatorAgent(ca_agent=_RoutePlanScriptAgent([]), occ_agent=occ, store=store)
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(
            agent=agent,
            seed="route-plan-rollback",
            process=process,
            logger=logger,
            log_level="full",
            max_steps=2,
        )
    finally:
        logger.close()

    assert result.error is None
    fact_texts = [str(item.get("text") or "") for item in store.run_memory.facts]
    assert "rejected fact must be rolled back" not in fact_texts
    assert "accepted fact remains" in fact_texts
    rows = _read_jsonl(out)
    rejected = [row for row in rows if row.get("type") == "rejected_action"]
    assert len(rejected) == 1
    assert rejected[0]["validation"]["code"] == "route_plan_missing_created_event"
    assert rejected[0]["agent_meta"]["raw_response"] == "RAW_INVALID_ROUTE_ATTEMPT"


def test_runner_can_soft_log_route_plan_validation_without_reask(tmp_path):
    boss_nodes = {
        (0, 0): "Ancient",
        (0, 1): "Monster",
        (2, 1): "Shop",
        (1, 2): "Elite",
        (2, 2): "RestSite",
        (3, 3): "Boss",
    }
    first_map = _full_map(
        nodes=boss_nodes,
        edges={
            (0, 0): [(0, 1), (2, 1)],
            (0, 1): [(1, 2), (2, 2)],
            (2, 1): [(2, 2)],
            (1, 2): [(3, 3)],
            (2, 2): [(3, 3)],
        },
    )
    second_map = _full_map(
        current=(0, 1),
        nodes={key: value for key, value in boss_nodes.items() if key != (0, 0)},
        edges={(0, 1): [(1, 2), (2, 2)], (1, 2): [(3, 3)], (2, 2): [(3, 3)]},
    )
    process = _RoutePlanProcess(
        [
            _map_state(
                choices=[
                    {"col": 0, "row": 1, "type": "Monster"},
                    {"col": 2, "row": 1, "type": "Shop"},
                ],
                full_map=first_map,
            ),
            _map_state(
                choices=[{"col": 1, "row": 2, "type": "Elite"}, {"col": 2, "row": 2, "type": "RestSite"}],
                full_map=second_map,
                hp=18,
            ),
            {"decision": "game_over", "victory": False, "player": {"hp": 38, "max_hp": 80}},
        ]
    )
    agent = _RoutePlanScriptAgent(
        [
            {
                "action_index": 5,
                "parsed": {
                    "reason": "create route",
                    "route_plan_event": _event(
                        "route_plan_created",
                        [(0, 1, "Monster"), (1, 2, "Elite"), (3, 3, "Boss")],
                    ),
                },
            },
            {"action_index": 6, "parsed": {"reason": "rest without update"}},
        ]
    )
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(
            agent=agent,
            seed="route-plan-soft",
            process=process,
            logger=logger,
            max_steps=5,
            route_plan_enforcement="soft",
        )
    finally:
        logger.close()

    assert result.error is None
    assert agent.feedback == []
    assert result.extra["route_plan_validation_soft_error_count"] == 1
    map_commands = [cmd for cmd in process.sent if cmd.get("action") == "select_map_node"]
    assert [cmd["args"] for cmd in map_commands] == [{"col": 0, "row": 1}, {"col": 2, "row": 2}]
    rows = _read_jsonl(out)
    validation_errors = [row for row in rows if row.get("type") == "route_plan_validation_error"]
    assert len(validation_errors) == 1
    assert validation_errors[0]["enforcement"] == "soft"
    events = [row for row in rows if row.get("type") == "route_plan_event"]
    assert [event["event_type"] for event in events] == ["route_plan_created"]
