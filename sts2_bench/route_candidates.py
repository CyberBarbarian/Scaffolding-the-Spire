"""Route enumeration and filtering helpers for map planning."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Literal

from .actions import LegalAction, build_legal_actions
from .process import Sts2Process
from .route_graph import build_route_graph, node_coord, normalize_map_node, route_graph_node_map


RouteObjective = Literal["max_restsites", "max_restsites_low_risk"]


@dataclass(frozen=True)
class RouteCandidate:
    """A complete visible route from one legal map start."""

    nodes: tuple[dict[str, Any], ...]
    choice_index: int | None = None
    action_id: int | None = None

    @property
    def start(self) -> dict[str, Any]:
        return self.nodes[0]

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            room_type = str(node.get("type") or "?")
            counts[room_type] = counts.get(room_type, 0) + 1
        return counts

    @property
    def restsite_count(self) -> int:
        return self.counts.get("RestSite", 0)

    @property
    def elite_count(self) -> int:
        return self.counts.get("Elite", 0)

    @property
    def monster_count(self) -> int:
        return self.counts.get("Monster", 0)

    @property
    def unknown_count(self) -> int:
        return self.counts.get("Unknown", 0)

    @property
    def shop_count(self) -> int:
        return self.counts.get("Shop", 0)

    @property
    def treasure_count(self) -> int:
        return self.counts.get("Treasure", 0)

    @property
    def combat_count(self) -> int:
        return self.monster_count + self.elite_count

    @property
    def pre_first_restsite_combat_count(self) -> int:
        count = 0
        for node in self.nodes:
            room_type = str(node.get("type") or "?")
            if room_type == "RestSite":
                break
            if room_type in {"Monster", "Elite"}:
                count += 1
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice_index": self.choice_index,
            "action_id": self.action_id,
            "start": dict(self.start),
            "counts": self.counts,
            "route": [dict(node) for node in self.nodes],
        }


def enumerate_route_candidates(
    full_map: dict[str, Any],
    choices: list[dict[str, Any]] | None = None,
    *,
    legal_actions: list[LegalAction] | None = None,
) -> list[RouteCandidate]:
    """Enumerate complete visible routes from every legal map choice."""

    if not isinstance(full_map, dict) or full_map.get("type") != "map":
        return []

    route_graph = build_route_graph(full_map, choices or [], legal_actions=legal_actions)
    node_map = route_graph_node_map(route_graph)
    candidates: list[RouteCandidate] = []
    for raw_start in route_graph.get("legal_starts") or []:
        start = normalize_map_node(raw_start)
        if start is None:
            continue
        starts = _complete_routes_from_node(start, node_map, seen={node_coord(start)})
        for route in starts:
            candidates.append(
                RouteCandidate(
                    nodes=tuple(_public_node(node) for node in route),
                    choice_index=_maybe_int(raw_start.get("choice_index")),
                    action_id=_maybe_int(raw_start.get("action_id")),
                )
            )
    return candidates


def rank_route_candidates(
    candidates: list[RouteCandidate],
    *,
    objective: RouteObjective = "max_restsites",
) -> list[RouteCandidate]:
    """Return candidates sorted from best to worst for the requested objective."""

    if objective == "max_restsites":
        return sorted(candidates, key=_max_restsites_key, reverse=True)
    if objective == "max_restsites_low_risk":
        return sorted(candidates, key=_max_restsites_low_risk_key, reverse=True)
    raise ValueError(f"Unsupported route objective: {objective}")


def select_route_candidate(
    candidates: list[RouteCandidate],
    *,
    objective: RouteObjective = "max_restsites",
) -> RouteCandidate | None:
    ranked = rank_route_candidates(candidates, objective=objective)
    return ranked[0] if ranked else None


def route_to_text(candidate: RouteCandidate) -> str:
    return " -> ".join(
        f"{node.get('type', '?')}({node.get('col')},{node.get('row')})" for node in candidate.nodes
    )


def _complete_routes_from_node(
    node: dict[str, Any],
    node_map: dict[tuple[int, int], dict[str, Any]],
    *,
    seen: set[tuple[int, int]],
) -> list[list[dict[str, Any]]]:
    current = _decorate_known_node(node, node_map)
    raw_node = node_map.get(node_coord(current))
    children = raw_node.get("children") if isinstance(raw_node, dict) else None
    if not children:
        return [[current]]

    routes: list[list[dict[str, Any]]] = []
    for raw_child in children:
        child = normalize_map_node(raw_child)
        if child is None:
            continue
        child_coord = node_coord(child)
        if child_coord in seen:
            continue
        for suffix in _complete_routes_from_node(child, node_map, seen={*seen, child_coord}):
            routes.append([current, *suffix])
    return routes or [[current]]


def _decorate_known_node(
    node: dict[str, Any],
    node_map: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    result = dict(node)
    known = node_map.get(node_coord(result))
    if isinstance(known, dict):
        result["type"] = str(known.get("type", result.get("type", "?")))
    return _public_node(result)


def _public_node(node: dict[str, Any]) -> dict[str, Any]:
    result = {
        "col": int(node["col"]),
        "row": int(node["row"]),
        "type": str(node.get("type", "?")),
    }
    if node.get("node_key") is not None:
        result["node_key"] = str(node.get("node_key"))
    return result


def _max_restsites_key(candidate: RouteCandidate) -> tuple[int, int, int, int, int, int]:
    # Primary objective: many campfires. Tie-breakers keep the first version simple:
    # fewer elites, more treasure, more monsters for early card rewards, fewer shops, then earlier choice.
    choice = candidate.choice_index if candidate.choice_index is not None else 999
    return (
        candidate.restsite_count,
        -candidate.elite_count,
        candidate.treasure_count,
        candidate.monster_count,
        -candidate.shop_count,
        -choice,
    )


def _max_restsites_low_risk_key(candidate: RouteCandidate) -> tuple[int, int, int, int, int, int, int, int]:
    # Keep campfire count as the primary objective. Risk terms only break ties
    # among routes with the same number of campfires.
    choice = candidate.choice_index if candidate.choice_index is not None else 999
    return (
        candidate.restsite_count,
        -candidate.elite_count,
        -candidate.pre_first_restsite_combat_count,
        -candidate.combat_count,
        -candidate.unknown_count,
        candidate.treasure_count,
        candidate.shop_count,
        -choice,
    )


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enumerate and rank visible STS2 map routes.")
    parser.add_argument("--seed", default="bench_0000")
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--neow-option", type=int, default=None)
    parser.add_argument("--card-indices", default="")
    parser.add_argument(
        "--objective",
        choices=["max_restsites", "max_restsites_low_risk"],
        default="max_restsites",
    )
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args(argv)

    proc = Sts2Process()
    try:
        proc.start()
        state = proc.start_run(
            character=args.character,
            ascension=args.ascension,
            seed=args.seed,
            lang=args.lang,
        )
        if args.neow_option is not None:
            state = proc.send({"cmd": "action", "action": "choose_option", "args": {"option_index": args.neow_option}})
        if args.card_indices:
            state = proc.send({"cmd": "action", "action": "select_cards", "args": {"indices": args.card_indices}})
        if state.get("decision") != "map_select":
            raise SystemExit(f"Current state is {state.get('decision')}; provide setup actions that reach map_select.")
        full_map = proc.send({"cmd": "get_map"})
    finally:
        proc.close()

    state = {**state, "full_map": full_map}
    legal_actions = build_legal_actions(state)
    candidates = enumerate_route_candidates(full_map, state.get("choices") or [], legal_actions=legal_actions)
    ranked = rank_route_candidates(candidates, objective=args.objective)
    selected = ranked[0] if ranked else None
    payload = {
        "seed": args.seed,
        "objective": args.objective,
        "candidate_count": len(candidates),
        "selected": selected.to_dict() if selected else None,
        "top": [candidate.to_dict() for candidate in ranked[: max(args.top, 0)]],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
