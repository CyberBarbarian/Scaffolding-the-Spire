"""Route-plan event validation for map decisions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .actions import LegalAction
from .route_graph import (
    build_route_graph,
    node_coord as route_node_coord,
    normalize_map_node,
    route_graph_current_coord,
    route_graph_edges,
    route_graph_node_map,
)


ROUTE_PLAN_EVENT_TYPES = {"route_plan_created", "route_plan_updated"}


@dataclass
class RoutePlanValidationError:
    code: str
    message: str
    feedback: str
    event: dict[str, Any] | None = None


@dataclass
class RoutePlanValidationResult:
    events: list[dict[str, Any]]
    error: RoutePlanValidationError | None = None


@dataclass
class ActiveRoutePlan:
    plan_id: str
    remaining: list[dict[str, Any]]
    map_revision: str | None
    act: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "remaining": [dict(node) for node in self.remaining],
            "based_on_map_revision": self.map_revision,
            "act": self.act,
        }

    def to_prompt_context(self) -> dict[str, Any]:
        remaining = [dict(node) for node in self.remaining]
        return {
            "plan_id": self.plan_id,
            "role": "advisory_current_state_feature",
            "decision_basis": (
                "choose from current state, legal_actions, route_graph, deck, relics, boss, "
                "and recent state changes"
            ),
            "past_plan_semantics": "previous plan is context, not a binding commitment",
            "remaining": remaining,
            "expected_next": dict(remaining[0]) if remaining else None,
            "actions": {
                "follow": "select expected_next and omit route_plan_event",
                "change": "select a different legal map action and emit route_plan_updated with reason",
            },
            "based_on_map_revision": self.map_revision,
            "act": self.act,
        }


class RoutePlanTracker:
    """Validate explicit route-plan events emitted by map-select policies.

    The tracker is intentionally runner-local.  It does not mutate engine state
    and treats current legal actions as the final execution authority.
    """

    def __init__(self) -> None:
        self.active: ActiveRoutePlan | None = None
        self._next_plan_id = 1

    def active_context(self) -> dict[str, Any] | None:
        if self.active is None or not self.active.remaining:
            return None
        return self.active.to_prompt_context()

    def validate_and_apply(
        self,
        *,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        action: LegalAction,
        meta: dict[str, Any],
        step: int,
    ) -> RoutePlanValidationResult:
        if state.get("decision") != "map_select" or action.kind != "map":
            return RoutePlanValidationResult([])

        selected = _node_from_action(action)
        if selected is None:
            return RoutePlanValidationResult(
                [],
                self._error(
                    "route_plan_selected_node_missing",
                    "Selected map action does not expose a col,row node.",
                ),
            )

        parsed = meta.get("parsed") if isinstance(meta.get("parsed"), dict) else {}
        raw_event = parsed.get("route_plan_event") if isinstance(parsed, dict) else None
        event = _normalize_route_plan_event(raw_event)
        legal_nodes = _legal_map_nodes(state, legal_actions)
        map_snapshot = _map_snapshot(state)
        map_revision = map_revision_for_state(state)
        state_act = _state_act(state)
        winged_boots_remaining = winged_boots_remaining_for_state(state)
        act_changed = bool(self.active and self.active.act is not None and state_act is not None and self.active.act != state_act)
        map_changed = bool(
            self.active
            and self.active.map_revision
            and map_revision
            and self.active.map_revision != map_revision
        )

        if self.active and self.active.remaining:
            if act_changed:
                if event is None or event.get("event_type") not in {"route_plan_created", "route_plan_updated"}:
                    return RoutePlanValidationResult(
                        [],
                        self._error(
                            "route_plan_invalid_after_map_change",
                            "The act changed; emit route_plan_created or route_plan_updated with reason_tags containing map_changed.",
                            event=event,
                        ),
                    )
                if "map_changed" not in event.get("reason_tags", []):
                    return RoutePlanValidationResult(
                        [],
                        self._error(
                            "route_plan_update_missing_map_changed_reason",
                            "Route change after a changed map must include reason_tags containing map_changed.",
                            event=event,
                        ),
                    )
                return self._apply_created_or_updated(
                    event,
                    event_type=event.get("event_type"),
                    selected=selected,
                    legal_nodes=legal_nodes,
                    map_snapshot=map_snapshot,
                    map_revision=map_revision,
                    map_changed=True,
                    act=state_act,
                    winged_boots_remaining=winged_boots_remaining,
                    step=step,
                )

            planned_next = self.active.remaining[0]
            if _coord(planned_next) not in legal_nodes:
                if map_changed:
                    if event is None or event.get("event_type") != "route_plan_updated":
                        return RoutePlanValidationResult(
                            [],
                            self._error(
                                "route_plan_invalid_after_map_change",
                                "The map revision changed and the old plan no longer matches legal_actions; emit route_plan_updated.",
                                event=event,
                            ),
                        )
                    if "map_changed" not in event.get("reason_tags", []):
                        return RoutePlanValidationResult(
                            [],
                            self._error(
                                "route_plan_update_missing_map_changed_reason",
                                "Route update after a changed map must include reason_tags containing map_changed.",
                                event=event,
                            ),
                        )
                    return self._apply_created_or_updated(
                        event,
                        event_type="route_plan_updated",
                        selected=selected,
                        legal_nodes=legal_nodes,
                        map_snapshot=map_snapshot,
                        map_revision=map_revision,
                        map_changed=True,
                        act=state_act,
                        winged_boots_remaining=winged_boots_remaining,
                        step=step,
                    )
                return RoutePlanValidationResult(
                    [],
                    self._error(
                        "route_plan_invalid_next_not_legal",
                        "The active route plan's next node is not in current legal_actions.",
                        event=event,
                    ),
                )

        if self.active is None or not self.active.remaining:
            if event is None:
                return RoutePlanValidationResult(
                    [],
                    self._error(
                        "route_plan_missing_created_event",
                        "No active route plan exists; emit route_plan_created before selecting a map node.",
                    ),
                )
            if event.get("event_type") != "route_plan_created":
                return RoutePlanValidationResult(
                    [],
                    self._error(
                        "route_plan_missing_created_event",
                        "No active route plan exists; emit route_plan_created before selecting a map node.",
                        event=event,
                    ),
                )
            return self._apply_created_or_updated(
                event,
                event_type="route_plan_created",
                selected=selected,
                legal_nodes=legal_nodes,
                map_snapshot=map_snapshot,
                map_revision=map_revision,
                map_changed=False,
                act=state_act,
                winged_boots_remaining=winged_boots_remaining,
                step=step,
            )

        planned_next = self.active.remaining[0]
        selected_matches_plan = _coord(selected) == _coord(planned_next)
        if event is None:
            if not selected_matches_plan:
                return RoutePlanValidationResult(
                    [],
                    self._error(
                        "route_plan_deviated_without_update",
                        "Selected map node is outside the active route plan and no route_plan_updated event was emitted.",
                    ),
                )
            return self._consume_followed(map_revision=map_revision, act=state_act)

        event_type = event.get("event_type")
        if event_type == "route_plan_updated":
            return self._apply_created_or_updated(
                event,
                event_type="route_plan_updated",
                selected=selected,
                legal_nodes=legal_nodes,
                map_snapshot=map_snapshot,
                map_revision=map_revision,
                map_changed=map_changed,
                act=state_act,
                winged_boots_remaining=winged_boots_remaining,
                step=step,
            )

        return RoutePlanValidationResult(
            [],
            self._error(
                "route_plan_unexpected_event_type",
                "Active route plans can only be executed silently or changed with route_plan_updated at map_select.",
                event=event,
            ),
        )

    def _apply_created_or_updated(
        self,
        event: dict[str, Any],
        *,
        event_type: str,
        selected: dict[str, Any],
        legal_nodes: dict[tuple[int, int], dict[str, Any]],
        map_snapshot: dict[str, Any] | None,
        map_revision: str | None,
        map_changed: bool,
        act: int | None,
        winged_boots_remaining: int,
        step: int,
    ) -> RoutePlanValidationResult:
        if not event.get("reason"):
            code = "route_plan_update_reason_required" if event_type == "route_plan_updated" else "route_plan_reason_required"
            message = (
                "route_plan_updated must include a non-empty reason explaining why the active route is changing."
                if event_type == "route_plan_updated"
                else "route_plan_created must include a non-empty reason explaining the route decision."
            )
            return RoutePlanValidationResult(
                [],
                self._error(
                    code,
                    message,
                    event=event,
                ),
            )

        plan = _normalize_plan(event.get("new_plan") or event.get("plan") or event.get("route"))
        if not plan:
            return RoutePlanValidationResult(
                [],
                self._error(
                    "route_plan_empty",
                    "route_plan_event must include a complete new_plan starting from the selected map action.",
                    event=event,
                ),
            )
        first = plan[0]
        if _coord(first) != _coord(selected):
            return RoutePlanValidationResult(
                [],
                self._error(
                    "route_plan_first_node_mismatch",
                    "The first node of new_plan must match the selected map action.",
                    event=event,
                ),
            )
        if _coord(first) not in legal_nodes:
            return RoutePlanValidationResult(
                [],
                self._error(
                    "route_plan_first_node_not_legal",
                    "The first node of new_plan is not in current legal_actions.",
                    event=event,
                ),
            )

        decorated_plan = [_merge_known_node_type(node, legal_nodes) for node in plan]
        topology = _validate_plan_topology(
            decorated_plan,
            map_snapshot,
            winged_boots_remaining=winged_boots_remaining,
        )
        if topology["route_validity"] == "invalid":
            return RoutePlanValidationResult(
                [],
                self._error(
                    "route_plan_topology_invalid",
                    topology.get("reason") or "new_plan is not a valid route through the current map snapshot.",
                    event=event,
                ),
            )
        if topology["route_validity"] == "incomplete":
            return RoutePlanValidationResult(
                [],
                self._error(
                    "route_plan_incomplete",
                    topology.get("reason") or "new_plan prefix must identify a route that can be completed through the visible graph.",
                    event=event,
                ),
            )

        old_remaining = self.active.remaining if self.active else []
        plan_id = self._allocate_plan_id() if event_type == "route_plan_created" else (self.active.plan_id if self.active else self._allocate_plan_id())
        remaining = [dict(node) for node in decorated_plan[1:]]
        active_for_log = ActiveRoutePlan(plan_id=plan_id, remaining=remaining, map_revision=map_revision, act=act)
        logged = {
            "type": "route_plan_event",
            "event_type": event_type,
            "step": step,
            "selected_node": selected,
            "old_plan_remaining": [dict(node) for node in old_remaining],
            "new_plan": decorated_plan,
            "reason_tags": event.get("reason_tags", []),
            "reason": event.get("reason") or "",
            "route_validity": topology["route_validity"],
            "override_used": bool(topology["override_used"]),
            "trigger": "map_changed" if map_changed else event.get("trigger"),
            "map_revision": map_revision,
            "active_plan": active_for_log.to_dict(),
        }
        if topology.get("winged_boots_jumps_used"):
            logged["winged_boots_jumps_used"] = topology["winged_boots_jumps_used"]
            logged["winged_boots_remaining_before"] = winged_boots_remaining
            logged["winged_boots_remaining_after"] = max(
                0,
                winged_boots_remaining - int(topology["winged_boots_jumps_used"]),
            )
        self.active = active_for_log if remaining else None
        return RoutePlanValidationResult([logged])

    def _consume_followed(
        self,
        *,
        map_revision: str | None,
        act: int | None,
    ) -> RoutePlanValidationResult:
        assert self.active is not None
        remaining = [dict(node) for node in self.active.remaining[1:]]
        active_for_log = ActiveRoutePlan(
            plan_id=self.active.plan_id,
            remaining=remaining,
            map_revision=map_revision or self.active.map_revision,
            act=act if act is not None else self.active.act,
        )
        self.active = active_for_log if remaining else None
        return RoutePlanValidationResult([])

    def _allocate_plan_id(self) -> str:
        plan_id = f"rp_{self._next_plan_id:03d}"
        self._next_plan_id += 1
        return plan_id

    def _error(
        self,
        code: str,
        message: str,
        *,
        event: dict[str, Any] | None = None,
    ) -> RoutePlanValidationError:
        return RoutePlanValidationError(
            code=code,
            message=message,
            feedback=_route_plan_feedback(code, message),
            event=event,
        )


def map_revision_for_state(state: dict[str, Any]) -> str | None:
    snapshot = _map_snapshot(state)
    if snapshot is None:
        return None
    payload = {
        "rows": snapshot.get("rows"),
        "boss": snapshot.get("boss"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _state_act(state: dict[str, Any]) -> int | None:
    value = state.get("act")
    if value is None and isinstance(state.get("context"), dict):
        value = state["context"].get("act")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def winged_boots_remaining_for_state(state: dict[str, Any]) -> int:
    """Return remaining Winged Boots forward jumps exposed by the current state."""

    for relic in _state_relics(state):
        if not _relic_matches_winged_boots(relic):
            continue
        counter = _winged_boots_counter(relic)
        return max(0, min(3, counter if counter is not None else 3))
    return 0


def _state_relics(state: dict[str, Any]) -> list[Any]:
    relics: list[Any] = []
    player = state.get("player")
    if isinstance(player, dict):
        player_relics = player.get("relics")
        if isinstance(player_relics, list):
            relics.extend(player_relics)
    state_relics = state.get("relics")
    if isinstance(state_relics, list):
        relics.extend(state_relics)
    return relics


def _relic_matches_winged_boots(relic: Any) -> bool:
    values: list[Any]
    if isinstance(relic, dict):
        values = [
            relic.get("id"),
            relic.get("key"),
            relic.get("name"),
            relic.get("title"),
            relic.get("display_name"),
        ]
        for nested in list(values):
            if isinstance(nested, dict):
                values.extend(nested.values())
    else:
        values = [relic]
    for value in values:
        text = str(value or "").strip().lower()
        normalized = re.sub(r"[\s_-]+", "", text)
        if normalized == "wingedboots" or "winged boots" in text or "\u7fbd\u7ffc\u4e4b\u9774" in text:
            return True
    return False


def _winged_boots_counter(relic: Any) -> int | None:
    if not isinstance(relic, dict):
        return None
    counter_keys = (
        "charges",
        "remaining",
        "uses_remaining",
        "jumps_remaining",
        "counter",
        "show_counter",
        "display_amount",
        "amount",
    )
    for key in counter_keys:
        value = _int_value(relic.get(key))
        if value is not None:
            return value
    for nested_key in ("vars", "state", "metadata"):
        nested = relic.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in counter_keys:
            value = _int_value(nested.get(key))
            if value is not None:
                return value
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    return None


def _normalize_route_plan_event(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"event_type": str(value)}
    if not value:
        return None
    raw_event_type = value.get("type")
    if raw_event_type is None or raw_event_type == "":
        raw_event_type = value.get("event_type")
    event_type = raw_event_type.strip() if isinstance(raw_event_type, str) else raw_event_type
    payload = {k: v for k, v in value.items() if k not in {"type", "event_type"}}
    if event_type not in ROUTE_PLAN_EVENT_TYPES:
        return {"event_type": str(event_type or ""), **payload}
    event = dict(payload)
    event["event_type"] = event_type
    event["reason"] = _one_line(event.get("reason"))
    event["reason_tags"] = _string_list(event.get("reason_tags"))
    return event


def _normalize_plan(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        node = _normalize_node(item)
        if node is not None:
            result.append(node)
    return result


def _normalize_node(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    node = normalize_map_node(value)
    if node is None:
        return None
    if value.get("type") is not None:
        node["type"] = str(value.get("type"))
    if value.get("floor") is not None:
        node["floor"] = value.get("floor")
    return node


def _node_from_action(action: LegalAction) -> dict[str, Any] | None:
    command = action.command or {}
    args = command.get("args") or {}
    if command.get("action") != "select_map_node":
        return None
    node = _normalize_node(args)
    if node is None:
        return None
    room_type = _room_type_from_label(action.label)
    if room_type:
        node["type"] = room_type
    return node


def _legal_map_nodes(state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[tuple[int, int], dict[str, Any]]:
    by_coord: dict[tuple[int, int], dict[str, Any]] = {}
    for choice in state.get("choices") or []:
        node = _normalize_node(choice)
        if node is not None:
            by_coord[_coord(node)] = node
    for action in legal_actions:
        if action.kind != "map":
            continue
        node = _node_from_action(action)
        if node is not None:
            existing = by_coord.setdefault(_coord(node), node)
            if "type" not in existing and node.get("type"):
                existing["type"] = node["type"]
    return by_coord


def _merge_known_node_type(node: dict[str, Any], legal_nodes: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    merged = dict(node)
    known = legal_nodes.get(_coord(node))
    if known:
        if "type" not in merged and known.get("type"):
            merged["type"] = known["type"]
    return merged


def _validate_plan_topology(
    plan: list[dict[str, Any]],
    snapshot: dict[str, Any] | None,
    *,
    winged_boots_remaining: int,
) -> dict[str, Any]:
    base = {"override_used": False, "winged_boots_jumps_used": 0}
    if snapshot is None:
        return {"route_validity": "unknown", **base}
    node_map, edges = _snapshot_graph(snapshot)
    if not node_map:
        return {"route_validity": "unknown", **base}
    plan_coords = [_coord(node) for node in plan]
    if any(coord not in node_map for coord in plan_coords):
        return {
            "route_validity": "invalid",
            **base,
            "reason": "new_plan references nodes that are absent from the current route_graph.",
        }
    for node in plan:
        coord = _coord(node)
        known_type = node_map.get(coord, {}).get("type")
        if known_type is None:
            continue
        submitted_type = node.get("type")
        if submitted_type is None:
            return {
                "route_validity": "invalid",
                **base,
                "reason": "Every new_plan node must include the exact type from the current route_graph.",
            }
        if str(submitted_type) != str(known_type):
            return {
                "route_validity": "invalid",
                **base,
                "reason": (
                    "new_plan node type must match the exact type from the current route_graph; "
                    f"node {coord} used {submitted_type!r} but route_graph has {known_type!r}."
                ),
            }

    jumps_used = 0
    current = _current_coord(snapshot)
    if current is not None and plan_coords:
        transition = _validate_route_transition(
            current,
            plan_coords[0],
            edges,
            winged_boots_available=winged_boots_remaining > 0,
        )
        if transition["route_validity"] == "invalid":
            return transition
        jumps_used += transition["winged_boots_jumps_used"]
        if jumps_used > winged_boots_remaining:
            return _too_many_winged_boots_jumps(jumps_used, winged_boots_remaining)
    for prev, nxt in zip(plan_coords, plan_coords[1:]):
        transition = _validate_route_transition(
            prev,
            nxt,
            edges,
            winged_boots_available=winged_boots_remaining > 0,
        )
        if transition["route_validity"] == "invalid":
            return transition
        jumps_used += transition["winged_boots_jumps_used"]
        if jumps_used > winged_boots_remaining:
            return _too_many_winged_boots_jumps(jumps_used, winged_boots_remaining)

    if plan_coords:
        boss_coords = {
            coord
            for coord, node in node_map.items()
            if str(node.get("type") or "").casefold() == "boss"
        }
        if boss_coords and plan_coords[-1] not in boss_coords:
            return {
                "route_validity": "incomplete",
                "override_used": bool(jumps_used),
                "winged_boots_jumps_used": jumps_used,
                "reason": "new_plan must continue to the visible Boss node when Boss is present in route_graph.",
            }

    if plan_coords and any(child in node_map for child in edges.get(plan_coords[-1], set())):
        return {
            "route_validity": "incomplete",
            "override_used": bool(jumps_used),
            "winged_boots_jumps_used": jumps_used,
            "reason": "new_plan must be a complete visible route; it cannot stop before visible child nodes.",
        }

    return {
        "route_validity": "winged_boots_override" if jumps_used else "topological",
        "override_used": bool(jumps_used),
        "winged_boots_jumps_used": jumps_used,
    }


def _too_many_winged_boots_jumps(jumps_used: int, winged_boots_remaining: int) -> dict[str, Any]:
    jump_text = "jump" if winged_boots_remaining == 1 else "jumps"
    limit_text = "three (3)" if winged_boots_remaining == 3 else str(winged_boots_remaining)
    return {
        "route_validity": "invalid",
        "override_used": True,
        "winged_boots_jumps_used": jumps_used,
        "reason": (
            f"Winged Boots permits at most {limit_text} disconnected forward "
            f"{jump_text} in this route plan because that many charges remain."
        ),
    }


def _validate_route_transition(
    previous: tuple[int, int],
    next_coord: tuple[int, int],
    edges: dict[tuple[int, int], set[tuple[int, int]]],
    *,
    winged_boots_available: bool,
) -> dict[str, Any]:
    if next_coord in edges.get(previous, set()):
        return {"route_validity": "topological", "override_used": False, "winged_boots_jumps_used": 0}
    if not winged_boots_available:
        return {
            "route_validity": "invalid",
            "override_used": False,
            "winged_boots_jumps_used": 0,
            "reason": "new_plan contains disconnected nodes; only Winged Boots can bridge disconnected route nodes.",
        }
    if next_coord[1] <= previous[1]:
        return {
            "route_validity": "invalid",
            "override_used": True,
            "winged_boots_jumps_used": 0,
            "reason": "Winged Boots route jumps must still move forward to a later map row.",
        }
    return {"route_validity": "winged_boots_override", "override_used": True, "winged_boots_jumps_used": 1}


def _snapshot_graph(snapshot: dict[str, Any]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], set[tuple[int, int]]]]:
    graph = build_route_graph(snapshot, [])
    return route_graph_node_map(graph), route_graph_edges(graph)


def _current_coord(snapshot: dict[str, Any]) -> tuple[int, int] | None:
    return route_graph_current_coord(build_route_graph(snapshot, []))


def _map_snapshot(state: dict[str, Any]) -> dict[str, Any] | None:
    full_map = state.get("full_map")
    if isinstance(full_map, dict) and full_map.get("type") == "map":
        return full_map
    return None


def _coord(node: dict[str, Any]) -> tuple[int, int]:
    return route_node_coord(node)


def _room_type_from_label(label: str) -> str | None:
    match = re.search(r"go to (?P<type>\w+) at col=", str(label))
    return match.group("type") if match else None


def _route_plan_feedback(code: str, message: str) -> str:
    missing_created_hint = ""
    if code == "route_plan_missing_created_event":
        missing_created_hint = (
            "There is no active route to execute or update. For this retry, emit route_plan_created with a "
            "complete new_plan whose first node is the selected legal map node. "
        )
    topology_hint = ""
    if code == "route_plan_topology_invalid":
        topology_hint = (
            "Do not include inferred or remembered future nodes that are absent from the current route_graph. "
            "Copy each new_plan node's exact col, row, and type from route_graph.nodes. "
            "Every adjacent pair in new_plan must be connected by the map unless Winged Boots is available; with "
            "Winged Boots, disconnected jumps cannot exceed the remaining charges and every jump must move forward "
            "to a later row. "
        )
    incomplete_hint = ""
    if code == "route_plan_incomplete":
        incomplete_hint = (
            "The submitted new_plan ends before the visible route is complete. Continue it through the current "
            "route_graph until the selected path reaches the visible Boss node, or the visible terminal node when "
            "Boss is not present in route_graph. "
        )
    return (
        f"Your previous map choice violated route-plan validation ({code}): {message} "
        f"{missing_created_hint}"
        f"{topology_hint}"
        f"{incomplete_hint}"
        "If you have no active route plan, emit route_plan_created with a complete new_plan whose first node is the selected legal map node. "
        "If you are executing the active route, choose the planned next node and do not emit route_plan_event. "
        "If you need to change route, emit route_plan_updated with reason_tags, a non-empty reason, and a complete new_plan from the selected legal node. "
        "If the map changed, include reason_tags containing map_changed. Then choose again."
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_one_line(value)]
    if not isinstance(value, list):
        return []
    return [_one_line(item) for item in value if _one_line(item)]


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
