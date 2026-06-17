"""Small scripted agent for route-plan smoke runs."""

from __future__ import annotations

import re
from typing import Any

from .actions import LegalAction


class RoutePlanSmokeAgent:
    """Deterministic policy that exercises route-plan logging on real maps."""

    def __init__(self) -> None:
        self.planned_remaining: list[dict[str, Any]] = []
        self.feedback: list[str] = []
        self.force_update = False

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        return ""

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt: str | None = None,
    ) -> tuple[LegalAction, dict[str, Any]]:
        del prompt
        if state.get("decision") == "map_select":
            return self._choose_map(state, legal_actions)
        action = _first_non_view_action(legal_actions)
        return action, {
            "agent": "route_plan_smoke",
            "parsed": {"reason": "Smoke policy selected the first non-view legal action."},
        }

    def receive_external_feedback(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        feedback: str,
        legal_actions: list[LegalAction],
    ) -> None:
        del state, action, meta, legal_actions
        self.feedback.append(feedback)
        if "route-plan validation" in feedback:
            self.force_update = True

    def _choose_map(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
    ) -> tuple[LegalAction, dict[str, Any]]:
        map_actions = [action for action in legal_actions if action.kind == "map"]
        if not map_actions:
            action = _first_non_view_action(legal_actions)
            return action, {
                "agent": "route_plan_smoke",
                "parsed": {"reason": "No map action was available; selected first non-view action."},
            }

        if self.planned_remaining and not self.force_update:
            planned_next = self.planned_remaining[0]
            matching = _matching_map_action(map_actions, planned_next)
            if matching is not None:
                self.planned_remaining = self.planned_remaining[1:]
                return matching, {
                    "agent": "route_plan_smoke",
                    "parsed": {"reason": "Follow smoke route plan node."},
                }

        selected = map_actions[0]
        selected_node = _node_from_action(selected)
        plan = _extend_plan_from_snapshot(selected_node, state.get("full_map"))
        event_type = "route_plan_updated" if self.planned_remaining or self.force_update else "route_plan_created"
        reason_tags = ["smoke_replan", "map_changed"] if event_type == "route_plan_updated" else ["smoke_start"]
        self.planned_remaining = [dict(node) for node in plan[1:]]
        self.force_update = False
        return selected, {
            "agent": "route_plan_smoke",
            "parsed": {
                "reason": "Exercise route-plan logging in a scripted smoke run.",
                "route_plan_event": {
                    "event_type": event_type,
                    "new_plan": plan,
                    "reason_tags": reason_tags,
                    "reason": "Create or repair a short deterministic route for smoke validation.",
                },
            },
        }


def _first_non_view_action(legal_actions: list[LegalAction]) -> LegalAction:
    for action in legal_actions:
        if not action.kind.startswith("view_"):
            return action
    if not legal_actions:
        raise ValueError("No legal actions available")
    return legal_actions[0]


def _matching_map_action(actions: list[LegalAction], node: dict[str, Any]) -> LegalAction | None:
    target = _coord(node)
    for action in actions:
        action_node = _node_from_action(action)
        if action_node is not None and _coord(action_node) == target:
            return action
    return None


def _extend_plan_from_snapshot(selected: dict[str, Any], snapshot: Any, *, max_nodes: int = 2) -> list[dict[str, Any]]:
    plan = [dict(selected)]
    if not isinstance(snapshot, dict) or snapshot.get("type") != "map":
        return plan
    node_map = _snapshot_nodes(snapshot)
    current = dict(selected)
    while len(plan) < max_nodes:
        node = node_map.get(_coord(current))
        children = node.get("children") if isinstance(node, dict) else None
        if not children:
            break
        child = _normalize_node(children[0])
        if child is None:
            break
        known = node_map.get(_coord(child))
        if isinstance(known, dict) and known.get("type") is not None:
            child["type"] = str(known.get("type"))
        plan.append(child)
        current = child
    return plan


def _snapshot_nodes(snapshot: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    nodes: dict[tuple[int, int], dict[str, Any]] = {}
    for row in snapshot.get("rows") or []:
        for raw_node in row:
            node = _normalize_node(raw_node)
            if node is not None:
                nodes[_coord(node)] = raw_node
    boss = snapshot.get("boss")
    node = _normalize_node(boss)
    if node is not None and isinstance(boss, dict):
        nodes[_coord(node)] = boss
    return nodes


def _node_from_action(action: LegalAction) -> dict[str, Any]:
    args = (action.command or {}).get("args") or {}
    node = _normalize_node(args) or {"col": args.get("col"), "row": args.get("row")}
    if "type" not in node:
        room_type = _room_type_from_label(action.label)
        if room_type:
            node["type"] = room_type
    return node


def _normalize_node(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        node = {"col": int(value["col"]), "row": int(value["row"])}
    except (KeyError, TypeError, ValueError):
        return None
    if value.get("type") is not None:
        node["type"] = str(value.get("type"))
    return node


def _coord(node: dict[str, Any]) -> tuple[int, int]:
    return int(node["col"]), int(node["row"])


def _room_type_from_label(label: str) -> str | None:
    match = re.search(r"go to (?P<type>\w+) at col=", str(label))
    return match.group("type") if match else None
