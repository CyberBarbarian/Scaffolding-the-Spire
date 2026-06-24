import json

from sts2_bench.actions import LegalAction
from sts2_bench.agents import build_llm_prompt
from sts2_bench.route_plan import RoutePlanTracker
from sts2_bench.runner import JsonlLogger, run_one


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


def test_route_plan_created_requires_complete_visible_route():
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
    assert "complete visible route" in result.error.feedback


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
    assert "Route graph JSON:" in prompt
    assert '"node_key":"0:1"' in prompt
    assert "Route graph exact nodes:" not in prompt


def test_map_prompt_can_omit_route_plan_instructions_when_disabled():
    state = _map_state(
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        full_map=_full_map(edges={(0, 0): [(0, 1)]}),
    )
    action = _map_action(0, "Monster", 0, 1)

    prompt = build_llm_prompt(state, [action], route_plan_enabled=False)

    assert "route_plan_event" not in prompt
    assert "route_plan_created" not in prompt
    assert "route_plan_updated" not in prompt
    assert "Route graph JSON:" in prompt


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
        return action, {"agent": "route_plan_script", "parsed": dict(response.get("parsed") or {})}

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


def test_runner_logs_route_plan_created_and_silent_execution(tmp_path):
    first_map = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2)]})
    second_map = _full_map(current=(0, 1), edges={(0, 1): [(1, 2)]})
    process = _RoutePlanProcess(
        [
            _map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=first_map),
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
                    "route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")]),
                },
            },
            {"action_index": 5, "parsed": {"reason": "follow planned elite"}},
        ]
    )
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(agent=agent, seed="route-plan", process=process, logger=logger, max_steps=5)
    finally:
        logger.close()

    assert result.error is None
    events = [row for row in _read_jsonl(out) if row.get("type") == "route_plan_event"]
    assert [event["event_type"] for event in events] == ["route_plan_created"]


def test_runner_can_disable_route_plan_validation(tmp_path):
    first_map = _full_map(edges={(0, 0): [(0, 1)]})
    process = _RoutePlanProcess(
        [
            _map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=first_map),
            {"decision": "game_over", "victory": False, "player": {"hp": 54, "max_hp": 80}},
        ]
    )
    agent = _RoutePlanScriptAgent(
        [
            {"action_index": 5, "parsed": {"reason": "choose current best map node"}},
        ]
    )
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(
            agent=agent,
            seed="route-plan-disabled",
            process=process,
            logger=logger,
            max_steps=5,
            route_plan_enabled=False,
        )
    finally:
        logger.close()

    assert result.error is None
    assert not [row for row in _read_jsonl(out) if row.get("type") == "route_plan_validation_error"]
    assert not [row for row in _read_jsonl(out) if row.get("type") == "route_plan_event"]


def test_runner_can_soft_log_route_plan_validation_without_reask(tmp_path):
    first_map = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2), (2, 2)]})
    second_map = _full_map(current=(0, 1), edges={(0, 1): [(1, 2), (2, 2)]})
    process = _RoutePlanProcess(
        [
            _map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=first_map),
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
                    "route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")]),
                },
            },
            {"action_index": 6, "parsed": {"reason": "rest without saying why"}},
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
    errors = [row for row in rows if row.get("type") == "route_plan_validation_error"]
    assert [row["enforcement"] for row in errors] == ["soft"]
    assert errors[0]["code"] == "route_plan_deviated_without_update"
    events = [row for row in rows if row.get("type") == "route_plan_event"]
    assert [event["event_type"] for event in events] == ["route_plan_created"]


def test_runner_reasks_when_map_choice_deviates_without_route_plan_update(tmp_path):
    first_map = _full_map(edges={(0, 0): [(0, 1)], (0, 1): [(1, 2), (2, 2)]})
    second_map = _full_map(current=(0, 1), edges={(0, 1): [(1, 2), (2, 2)]})
    process = _RoutePlanProcess(
        [
            _map_state(choices=[{"col": 0, "row": 1, "type": "Monster"}], full_map=first_map),
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
                    "route_plan_event": _event("route_plan_created", [(0, 1, "Monster"), (1, 2, "Elite")]),
                },
            },
            {"action_index": 6, "parsed": {"reason": "rest without saying why"}},
            {
                "action_index": 6,
                "parsed": {
                    "reason": "update route after feedback",
                    "route_plan_event": _event(
                        "route_plan_updated",
                        [(2, 2, "RestSite")],
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
