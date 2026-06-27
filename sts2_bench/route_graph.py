"""Formal route graph representation for map planning."""

from __future__ import annotations

import re
from typing import Any

from .actions import LegalAction


Coord = tuple[int, int]


ROOM_TYPE_SYMBOLS = {
    "Ancient": "A",
    "Monster": "M",
    "Elite": "E",
    "RestSite": "R",
    "Shop": "S",
    "Treasure": "T",
    "Unknown": "?",
    "Boss": "B",
}

ROUTE_MAP_LEGEND = {
    "A": "Ancient",
    "M": "Monster",
    "E": "Elite",
    "R": "RestSite",
    "S": "Shop",
    "T": "Treasure",
    "?": "Unknown",
    "B": "Boss",
}


def build_route_graph(
    map_data: dict[str, Any],
    choices: list[dict[str, Any]] | None = None,
    *,
    legal_actions: list[LegalAction] | None = None,
    winged_boots_remaining: int | None = None,
) -> dict[str, Any]:
    """Build a machine-readable graph for route-plan prompting and checks."""

    nodes_by_coord = _snapshot_nodes(map_data)
    action_ids = _map_action_ids_by_coord(legal_actions or [])
    current = _decorate_node(normalize_map_node(map_data.get("current_coord") or {}), nodes_by_coord)
    nodes = []

    for coord in sorted(nodes_by_coord, key=lambda item: (item[1], item[0])):
        raw_node = nodes_by_coord[coord]
        node = _public_node(raw_node, coord=coord)
        children = []
        for raw_child in raw_node.get("children") or []:
            child = normalize_map_node(raw_child)
            if child is None:
                continue
            child_coord = node_coord(child)
            known_child = nodes_by_coord.get(child_coord)
            if known_child is None:
                continue
            children.append(_public_node(known_child, coord=child_coord))
        node["children"] = children
        nodes.append(node)

    legal_starts = []
    for index, raw_choice in enumerate(_visible_choices(choices or [], nodes_by_coord)):
        choice = _decorate_node(normalize_map_node(raw_choice), nodes_by_coord)
        if choice is None:
            continue
        choice["choice_index"] = index
        action_id = action_ids.get(node_coord(choice))
        if action_id is not None:
            choice["action_id"] = action_id
        legal_starts.append(choice)

    constraints = {
        "ordered_plan_required": True,
        "complete_visible_route_required": True,
        "winged_boots_max_forward_jumps": 3,
    }
    if winged_boots_remaining is not None:
        constraints["winged_boots_remaining_forward_jumps"] = max(0, min(3, int(winged_boots_remaining)))

    return {
        "current": current,
        "legal_starts": legal_starts,
        "nodes": nodes,
        "constraints": constraints,
    }


def build_canonical_route_map(
    map_data: dict[str, Any],
    choices: list[dict[str, Any]] | None = None,
    *,
    legal_actions: list[LegalAction] | None = None,
    winged_boots_remaining: int | None = None,
) -> dict[str, Any]:
    """Build an LLM-facing row/edge map without route-value features."""

    nodes_by_coord = _snapshot_nodes(map_data)
    action_ids = _map_action_ids_by_coord(legal_actions or [])
    current = normalize_map_node(map_data.get("current_coord") or {})
    rows = []
    for row, coords in _coords_grouped_by_row(nodes_by_coord).items():
        row_nodes = []
        for coord in coords:
            raw_node = nodes_by_coord[coord]
            node = _canonical_node(raw_node, coord=coord)
            action_id = action_ids.get(coord)
            if action_id is not None:
                node["action_id"] = action_id
            row_nodes.append(node)
        rows.append({"row": row, "nodes": row_nodes})

    legal_starts = []
    for index, raw_choice in enumerate(_visible_choices(choices or [], nodes_by_coord)):
        choice = _decorate_node(normalize_map_node(raw_choice), nodes_by_coord)
        if choice is None:
            continue
        coord = node_coord(choice)
        room_type = str(choice.get("type", "?"))
        start = {
            "node": canonical_node_id(coord),
            "symbol": _room_type_symbol(room_type),
            "type": room_type,
            "choice_index": index,
        }
        action_id = action_ids.get(coord)
        if action_id is not None:
            start = {"action_id": action_id, **start}
        legal_starts.append(start)

    edges = {}
    for coord in sorted(nodes_by_coord, key=lambda item: (item[1], item[0])):
        raw_node = nodes_by_coord[coord]
        children = []
        for raw_child in raw_node.get("children") or []:
            child = normalize_map_node(raw_child)
            if child is None:
                continue
            child_coord = node_coord(child)
            if child_coord in nodes_by_coord:
                children.append(canonical_node_id(child_coord))
        edges[canonical_node_id(coord)] = sorted(children, key=_canonical_node_sort_key)

    return {
        "schema": "sts2_route_map_v1",
        "legend": ROUTE_MAP_LEGEND,
        "current": canonical_node_id(node_coord(current)) if current else None,
        "legal_starts": legal_starts,
        "rows": rows,
        "edges": edges,
        "constraints": {
            "only_follow_edges": True,
            "complete_visible_route_required": True,
            "winged_boots_left": 0 if winged_boots_remaining is None else max(0, min(3, int(winged_boots_remaining))),
            "winged_boots_rule": "may jump to a later row only when a normal edge is not used",
        },
    }


def normalize_map_node(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        node = {"col": int(value["col"]), "row": int(value["row"])}
    except (KeyError, TypeError, ValueError):
        return None
    if value.get("type") is not None:
        node["type"] = str(value.get("type"))
    return node


def node_coord(node: dict[str, Any]) -> Coord:
    return int(node["col"]), int(node["row"])


def make_node_key(coord: Coord) -> str:
    return f"{coord[0]}:{coord[1]}"


def canonical_node_id(coord: Coord) -> str:
    return f"r{coord[1]:02d}c{coord[0]}"


def route_graph_node_map(route_graph: dict[str, Any]) -> dict[Coord, dict[str, Any]]:
    nodes: dict[Coord, dict[str, Any]] = {}
    for raw_node in route_graph.get("nodes") or []:
        node = normalize_map_node(raw_node)
        if node is None:
            continue
        public = dict(raw_node)
        public.setdefault("node_key", make_node_key(node_coord(node)))
        nodes[node_coord(node)] = public
    return nodes


def route_graph_edges(route_graph: dict[str, Any]) -> dict[Coord, set[Coord]]:
    edges: dict[Coord, set[Coord]] = {}
    for raw_node in route_graph.get("nodes") or []:
        node = normalize_map_node(raw_node)
        if node is None:
            continue
        coord = node_coord(node)
        for raw_child in raw_node.get("children") or []:
            child = normalize_map_node(raw_child)
            if child is not None:
                edges.setdefault(coord, set()).add(node_coord(child))
    return edges


def route_graph_current_coord(route_graph: dict[str, Any]) -> Coord | None:
    current = normalize_map_node(route_graph.get("current") or {})
    return node_coord(current) if current else None


def _snapshot_nodes(map_data: dict[str, Any]) -> dict[Coord, dict[str, Any]]:
    nodes: dict[Coord, dict[str, Any]] = {}
    for row in map_data.get("rows") or []:
        for raw_node in row:
            node = normalize_map_node(raw_node)
            if node is not None and isinstance(raw_node, dict):
                nodes[node_coord(node)] = raw_node
    boss = map_data.get("boss")
    boss_node = normalize_map_node(boss)
    if boss_node is not None and isinstance(boss, dict):
        nodes[node_coord(boss_node)] = boss
    return nodes


def _public_node(raw_node: dict[str, Any], *, coord: Coord | None = None) -> dict[str, Any]:
    node = normalize_map_node(raw_node)
    if node is None:
        if coord is None:
            raise ValueError("Cannot build public node without coordinates")
        node = {"col": coord[0], "row": coord[1]}
    coord = coord or node_coord(node)
    return {
        "col": coord[0],
        "row": coord[1],
        "type": str(raw_node.get("type", node.get("type", "?"))),
        "node_key": make_node_key(coord),
    }


def _canonical_node(raw_node: dict[str, Any], *, coord: Coord) -> dict[str, Any]:
    room_type = str(raw_node.get("type", "?"))
    return {
        "id": canonical_node_id(coord),
        "col": coord[0],
        "row": coord[1],
        "symbol": _room_type_symbol(room_type),
        "type": room_type,
    }


def _room_type_symbol(room_type: Any) -> str:
    return ROOM_TYPE_SYMBOLS.get(str(room_type), "?")


def _coords_grouped_by_row(nodes_by_coord: dict[Coord, dict[str, Any]]) -> dict[int, list[Coord]]:
    rows: dict[int, list[Coord]] = {}
    for coord in sorted(nodes_by_coord, key=lambda item: (item[1], item[0])):
        rows.setdefault(coord[1], []).append(coord)
    return rows


def _canonical_node_sort_key(node_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"r(?P<row>-?\d+)c(?P<col>-?\d+)", node_id)
    if not match:
        return (0, 0)
    return int(match.group("row")), int(match.group("col"))


def _decorate_node(node: dict[str, Any] | None, nodes_by_coord: dict[Coord, dict[str, Any]]) -> dict[str, Any] | None:
    if node is None:
        return None
    known = nodes_by_coord.get(node_coord(node), node)
    return _public_node(known, coord=node_coord(node))


def _visible_choices(choices: list[dict[str, Any]], nodes_by_coord: dict[Coord, dict[str, Any]]) -> list[dict[str, Any]]:
    if not nodes_by_coord:
        return choices
    visible_coords = set(nodes_by_coord)
    filtered = []
    for choice in choices:
        node = normalize_map_node(choice)
        if node is None:
            continue
        if node_coord(node) in visible_coords or choice.get("type") == "Ancient":
            filtered.append(choice)
    return filtered or choices


def _map_action_ids_by_coord(legal_actions: list[LegalAction]) -> dict[Coord, int]:
    by_coord: dict[Coord, int] = {}
    for action in legal_actions:
        if action.kind != "map":
            continue
        node = _node_from_action(action)
        if node is not None:
            by_coord[node_coord(node)] = action.action_id
    return by_coord


def _node_from_action(action: LegalAction) -> dict[str, Any] | None:
    command = action.command or {}
    args = command.get("args") or {}
    if command.get("action") != "select_map_node":
        return None
    node = normalize_map_node(args)
    if node is None:
        return None
    room_type = _room_type_from_label(action.label)
    if room_type:
        node["type"] = room_type
    return node


def _room_type_from_label(label: str) -> str | None:
    match = re.search(r"go to (?P<type>\w+) at col=", str(label))
    return match.group("type") if match else None
