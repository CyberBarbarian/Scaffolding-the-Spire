"""Small scripted agent for route-plan smoke runs."""

from __future__ import annotations

import re
from typing import Any

from .actions import LegalAction
from .route_graph import build_route_graph, node_coord as route_node_coord, normalize_map_node, route_graph_node_map


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

        selected, plan, route_tags = _select_route(state, map_actions)
        event_type = "route_plan_updated" if self.planned_remaining or self.force_update else "route_plan_created"
        reason_tags = ["smoke_replan", "map_changed"] if event_type == "route_plan_updated" else ["smoke_start"]
        reason_tags.extend(route_tags)
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
                    "reason": "Create or repair a complete deterministic route for smoke validation.",
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


def _select_route(state: dict[str, Any], map_actions: list[LegalAction]) -> tuple[LegalAction, list[dict[str, Any]], list[str]]:
    best: tuple[float, int, LegalAction, list[dict[str, Any]]] | None = None
    for index, action in enumerate(map_actions):
        selected = _node_from_action(action)
        plan = _extend_plan_from_snapshot(selected, state.get("full_map"), state=state)
        score = _score_route(plan, state)
        candidate = (score, -index, action, plan)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    score, _index, action, plan = best
    del score
    return action, plan, _route_reason_tags(plan, state)


def _extend_plan_from_snapshot(
    selected: dict[str, Any],
    snapshot: Any,
    *,
    state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict) or snapshot.get("type") != "map":
        return [dict(selected)]
    route_graph = build_route_graph(snapshot, (state or {}).get("choices") or [])
    node_map = route_graph_node_map(route_graph)
    routes = _complete_routes_from_node(selected, node_map, seen={_coord(selected)})
    if not state:
        return routes[0]
    return max(routes, key=lambda route: _score_route(route, state))


def _complete_routes_from_node(
    node: dict[str, Any],
    node_map: dict[tuple[int, int], dict[str, Any]],
    *,
    seen: set[tuple[int, int]],
) -> list[list[dict[str, Any]]]:
    current = _decorate_known_node(node, node_map)
    raw_node = node_map.get(_coord(current))
    children = raw_node.get("children") if isinstance(raw_node, dict) else None
    if not children:
        return [[current]]

    routes: list[list[dict[str, Any]]] = []
    for raw_child in children:
        child = _normalize_node(raw_child)
        if child is None:
            continue
        child_coord = _coord(child)
        if child_coord in seen:
            continue
        for suffix in _complete_routes_from_node(child, node_map, seen={*seen, child_coord}):
            routes.append([current, *suffix])
    return routes or [[current]]


def _decorate_known_node(node: dict[str, Any], node_map: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    result = dict(node)
    known = node_map.get(_coord(result))
    if isinstance(known, dict) and known.get("type") is not None:
        result["type"] = str(known.get("type"))
    return result


def _score_route(plan: list[dict[str, Any]], state: dict[str, Any]) -> float:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    hp = _number(player.get("hp"), default=0.0)
    max_hp = max(_number(player.get("max_hp"), default=80.0), 1.0)
    gold = _number(player.get("gold"), default=0.0)
    hp_ratio = hp / max_hp
    low_hp = hp_ratio <= 0.35
    healthy = hp_ratio >= 0.65
    rich = gold >= 150
    score = 0.0
    for index, node in enumerate(plan):
        room_type = str(node.get("type") or "")
        early = max(0.0, 1.5 - (0.25 * index))
        if low_hp:
            score += _low_hp_room_score(room_type, gold) * early
        elif rich:
            score += _rich_room_score(room_type, hp_ratio) * early
        elif healthy:
            score += _healthy_room_score(room_type, gold) * early
        else:
            score += _balanced_room_score(room_type, gold) * early
    return score


def _low_hp_room_score(room_type: str, gold: float) -> float:
    if room_type == "Elite":
        return -10.0
    if room_type == "RestSite":
        return 8.0
    if room_type == "Shop":
        return 2.5 if gold >= 75 else -2.0
    if room_type == "Treasure":
        return 1.0
    if room_type == "Monster":
        return -1.0
    return 0.0


def _rich_room_score(room_type: str, hp_ratio: float) -> float:
    if room_type == "Shop":
        return 8.0
    if room_type == "Treasure":
        return 3.0
    if room_type == "RestSite":
        return 2.0
    if room_type == "Elite":
        return 1.5 if hp_ratio >= 0.5 else -4.0
    if room_type == "Monster":
        return 0.5
    return 0.0


def _healthy_room_score(room_type: str, gold: float) -> float:
    if room_type == "Elite":
        return 5.0
    if room_type == "Treasure":
        return 2.0
    if room_type == "Monster":
        return 1.0
    if room_type == "Shop":
        return 1.0 if gold >= 75 else -1.0
    if room_type == "RestSite":
        return -1.0
    return 0.0


def _balanced_room_score(room_type: str, gold: float) -> float:
    if room_type == "RestSite":
        return 4.0
    if room_type == "Shop":
        return 3.0 if gold >= 100 else 0.5
    if room_type == "Treasure":
        return 2.0
    if room_type == "Elite":
        return -2.0
    if room_type == "Monster":
        return 0.5
    return 0.0


def _route_reason_tags(plan: list[dict[str, Any]], state: dict[str, Any]) -> list[str]:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    hp = _number(player.get("hp"), default=0.0)
    max_hp = max(_number(player.get("max_hp"), default=80.0), 1.0)
    gold = _number(player.get("gold"), default=0.0)
    room_types = [str(node.get("type") or "") for node in plan]
    tags: list[str] = []
    if hp / max_hp <= 0.35:
        tags.append("low_hp")
    if gold >= 150 and "Shop" in room_types:
        tags.append("shop_value")
    if room_types.count("Elite") >= 2:
        tags.append("elite_hunting")
    if "RestSite" in room_types[:3]:
        tags.append("early_rest")
    return tags


def _number(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _node_from_action(action: LegalAction) -> dict[str, Any]:
    args = (action.command or {}).get("args") or {}
    node = _normalize_node(args) or {"col": args.get("col"), "row": args.get("row")}
    if "type" not in node:
        room_type = _room_type_from_label(action.label)
        if room_type:
            node["type"] = room_type
    return node


def _normalize_node(value: Any) -> dict[str, Any] | None:
    return normalize_map_node(value)


def _coord(node: dict[str, Any]) -> tuple[int, int]:
    return route_node_coord(node)


def _room_type_from_label(label: str) -> str | None:
    match = re.search(r"go to (?P<type>\w+) at col=", str(label))
    return match.group("type") if match else None
