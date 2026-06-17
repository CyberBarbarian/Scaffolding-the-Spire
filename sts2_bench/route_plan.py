"""Route-plan event validation for map decisions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .actions import LegalAction


ROUTE_PLAN_EVENT_TYPES = {"route_plan_created", "route_plan_followed", "route_plan_updated"}


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "remaining": [dict(node) for node in self.remaining],
            "based_on_map_revision": self.map_revision,
        }


class RoutePlanTracker:
    """Validate explicit route-plan events emitted by map-select policies.

    The tracker is intentionally runner-local.  It does not mutate engine state
    and treats current legal actions as the final execution authority.
    """

    def __init__(self) -> None:
        self.active: ActiveRoutePlan | None = None
        self._next_plan_id = 1

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
        map_changed = bool(
            self.active
            and self.active.map_revision
            and map_revision
            and self.active.map_revision != map_revision
        )

        if self.active and self.active.remaining:
            planned_next = self.active.remaining[0]
            if _coord(planned_next) not in legal_nodes:
                if not map_changed:
                    return RoutePlanValidationResult(
                        [],
                        self._error(
                            "route_plan_invalid_next_not_legal",
                            "The active route plan's next node is not in current legal_actions.",
                            event=event,
                        ),
                    )
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

        if self.active is None or not self.active.remaining:
            if event is None:
                return RoutePlanValidationResult([])
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
            return self._apply_followed(selected=selected, map_revision=map_revision, step=step)

        event_type = event.get("event_type")
        if event_type == "route_plan_followed":
            if not selected_matches_plan:
                return RoutePlanValidationResult(
                    [],
                    self._error(
                        "route_plan_followed_wrong_node",
                        "route_plan_followed selected a node different from the active plan's next node.",
                        event=event,
                    ),
                )
            return self._apply_followed(selected=selected, map_revision=map_revision, step=step, source_event=event)

        if event_type == "route_plan_updated":
            return self._apply_created_or_updated(
                event,
                event_type="route_plan_updated",
                selected=selected,
                legal_nodes=legal_nodes,
                map_snapshot=map_snapshot,
                map_revision=map_revision,
                map_changed=map_changed,
                step=step,
            )

        return RoutePlanValidationResult(
            [],
            self._error(
                "route_plan_unexpected_event_type",
                "Active route plans can only be followed or updated at map_select.",
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
        step: int,
    ) -> RoutePlanValidationResult:
        plan = _normalize_plan(event.get("new_plan") or event.get("plan") or event.get("route"))
        if not plan:
            return RoutePlanValidationResult(
                [],
                self._error("route_plan_empty", f"{event_type} must include a non-empty new_plan.", event=event),
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
        original_decorated_plan = [dict(node) for node in decorated_plan]
        trimmed_invalid_suffix: list[dict[str, Any]] = []
        topology = _validate_plan_topology(decorated_plan, map_snapshot)
        if topology["route_validity"] == "invalid":
            trimmed_plan, trimmed_invalid_suffix = _trim_invalid_future_suffix(decorated_plan, map_snapshot)
            if not trimmed_plan:
                return RoutePlanValidationResult(
                    [],
                    self._error(
                        "route_plan_topology_invalid",
                        "new_plan references nodes that are absent from the current map snapshot.",
                        event=event,
                    ),
                )
            decorated_plan = trimmed_plan
            topology = _validate_plan_topology(decorated_plan, map_snapshot)
            if topology["route_validity"] == "invalid":
                return RoutePlanValidationResult(
                    [],
                    self._error(
                        "route_plan_topology_invalid",
                        "new_plan references nodes that are absent from the current map snapshot.",
                        event=event,
                    ),
                )
            topology = {**topology, "route_validity": "partial_invalid_suffix_trimmed"}

        old_remaining = self.active.remaining if self.active else []
        plan_id = self._allocate_plan_id() if event_type == "route_plan_created" else (self.active.plan_id if self.active else self._allocate_plan_id())
        remaining = [dict(node) for node in decorated_plan[1:]]
        active_for_log = ActiveRoutePlan(plan_id=plan_id, remaining=remaining, map_revision=map_revision)
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
        if trimmed_invalid_suffix:
            logged["original_new_plan"] = original_decorated_plan
            logged["trimmed_invalid_suffix"] = trimmed_invalid_suffix
            logged["trimmed_invalid_suffix_count"] = len(trimmed_invalid_suffix)
        self.active = active_for_log if remaining else None
        return RoutePlanValidationResult([logged])

    def _apply_followed(
        self,
        *,
        selected: dict[str, Any],
        map_revision: str | None,
        step: int,
        source_event: dict[str, Any] | None = None,
    ) -> RoutePlanValidationResult:
        assert self.active is not None
        old_remaining = [dict(node) for node in self.active.remaining]
        remaining = [dict(node) for node in self.active.remaining[1:]]
        active_for_log = ActiveRoutePlan(
            plan_id=self.active.plan_id,
            remaining=remaining,
            map_revision=map_revision or self.active.map_revision,
        )
        logged = {
            "type": "route_plan_event",
            "event_type": "route_plan_followed",
            "step": step,
            "selected_node": selected,
            "old_plan_remaining": old_remaining,
            "reason_tags": (source_event or {}).get("reason_tags", []),
            "reason": (source_event or {}).get("reason") or "follow active route plan",
            "map_revision": map_revision,
            "active_plan": active_for_log.to_dict(),
        }
        self.active = active_for_log if remaining else None
        return RoutePlanValidationResult([logged])

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
        "current_coord": snapshot.get("current_coord"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _normalize_route_plan_event(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    event_type = value.get("type") or value.get("event_type")
    if event_type not in ROUTE_PLAN_EVENT_TYPES:
        return {"event_type": str(event_type or ""), **{k: v for k, v in value.items() if k not in {"type", "event_type"}}}
    event = {k: v for k, v in value.items() if k not in {"type", "event_type"}}
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
    col = value.get("col")
    row = value.get("row")
    if col is None or row is None:
        return None
    try:
        node = {"col": int(col), "row": int(row)}
    except (TypeError, ValueError):
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


def _validate_plan_topology(plan: list[dict[str, Any]], snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        return {"route_validity": "unknown", "override_used": False}
    node_map, edges = _snapshot_graph(snapshot)
    if not node_map:
        return {"route_validity": "unknown", "override_used": False}
    plan_coords = [_coord(node) for node in plan]
    if any(coord not in node_map for coord in plan_coords):
        return {"route_validity": "invalid", "override_used": False}

    route_validity = "topological"
    override_used = False
    current = _current_coord(snapshot)
    if current is not None and plan_coords:
        if plan_coords[0] not in edges.get(current, set()):
            route_validity = "requires_override"
            override_used = True
    for prev, nxt in zip(plan_coords, plan_coords[1:]):
        if nxt not in edges.get(prev, set()):
            route_validity = "requires_override"
    return {"route_validity": route_validity, "override_used": override_used}


def _trim_invalid_future_suffix(
    plan: list[dict[str, Any]], snapshot: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if snapshot is None:
        return plan, []
    node_map, _edges = _snapshot_graph(snapshot)
    if not node_map:
        return plan, []
    valid_prefix: list[dict[str, Any]] = []
    for index, node in enumerate(plan):
        if _coord(node) not in node_map:
            return valid_prefix, [dict(item) for item in plan[index:]]
        valid_prefix.append(dict(node))
    return valid_prefix, []


def _snapshot_graph(snapshot: dict[str, Any]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], set[tuple[int, int]]]]:
    node_map: dict[tuple[int, int], dict[str, Any]] = {}
    edges: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for row_nodes in snapshot.get("rows") or []:
        for raw_node in row_nodes:
            node = _normalize_node(raw_node)
            if node is None:
                continue
            coord = _coord(node)
            node_map[coord] = raw_node
            for child in raw_node.get("children") or []:
                child_node = _normalize_node(child)
                if child_node is not None:
                    edges.setdefault(coord, set()).add(_coord(child_node))
    boss = _normalize_node(snapshot.get("boss") or {})
    if boss is not None:
        node_map[_coord(boss)] = snapshot.get("boss") or {}
    return node_map, edges


def _current_coord(snapshot: dict[str, Any]) -> tuple[int, int] | None:
    current = _normalize_node(snapshot.get("current_coord") or {})
    return _coord(current) if current else None


def _map_snapshot(state: dict[str, Any]) -> dict[str, Any] | None:
    full_map = state.get("full_map")
    if isinstance(full_map, dict) and full_map.get("type") == "map":
        return full_map
    return None


def _coord(node: dict[str, Any]) -> tuple[int, int]:
    return (int(node["col"]), int(node["row"]))


def _room_type_from_label(label: str) -> str | None:
    match = re.search(r"go to (?P<type>\w+) at col=", str(label))
    return match.group("type") if match else None


def _route_plan_feedback(code: str, message: str) -> str:
    missing_created_hint = ""
    if code == "route_plan_missing_created_event":
        missing_created_hint = (
            "There is no active route to follow or update. This often happens after a short one-node new_plan "
            "was consumed by the previous map move. For this retry, do not emit route_plan_followed or "
            "route_plan_updated; emit route_plan_created with a new_plan whose first and possibly only node is "
            "the selected legal map node. "
        )
    topology_hint = ""
    if code == "route_plan_topology_invalid":
        topology_hint = (
            "Do not include inferred or remembered future nodes that are absent from the current full map snapshot. "
            "If uncertain, make new_plan contain only the selected legal map node. "
        )
    return (
        f"Your previous map choice violated route-plan validation ({code}): {message} "
        f"{missing_created_hint}"
        f"{topology_hint}"
        "If you have no active route plan, emit route_plan_created with a non-empty new_plan whose first node is the selected legal map node. "
        "If you are following the active route, choose the planned next node. "
        "If you need to change route, emit route_plan_updated with old_plan_remaining, new_plan, reason_tags, and reason; "
        "the new_plan first node must be in current legal_actions. "
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
