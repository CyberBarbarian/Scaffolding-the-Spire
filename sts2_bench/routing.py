"""Decision routing for the V1 coordinator agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ActiveAgent = Literal["ca", "occ", "none"]


@dataclass(frozen=True)
class RouteDecision:
    active_agent: ActiveAgent
    reason: str


def route_state(state: dict[str, Any]) -> RouteDecision:
    """Route one decision point to CombatAgent or OutOfCombatController.

    Routing is intentionally based on the exported decision surface, not on
    room labels.  ``card_select`` is the main dual-owned state: combat-generated
    modal selections stay with CA, while reward/event selections go to OCC.
    """

    decision = state.get("decision")
    if decision == "game_over":
        return RouteDecision("none", "game_over is terminal")
    if decision == "combat_play":
        return RouteDecision("ca", "combat_play is routed to CA")
    if decision == "card_select":
        if isinstance(state.get("combat"), dict) and state.get("combat"):
            return RouteDecision("ca", "card_select with combat context is routed to CA")
        return RouteDecision("occ", "card_select without combat context is routed to OCC")

    return RouteDecision("occ", f"{decision} is routed to OCC")
