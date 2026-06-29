"""Atomic card-sequence execution for benchmark actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MAX_CUSTOM_SEQUENCE_VALIDATION_LENGTH = 4


@dataclass(frozen=True)
class CardSequenceResult:
    state: dict[str, Any]
    steps_executed: int
    stopped_reason: str | None = None
    commands_executed: list[dict[str, Any]] = field(default_factory=list)


def execute_card_sequence(
    state: dict[str, Any],
    sequence: list[dict[str, Any]],
    send_fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    stop_if_card: Callable[[dict[str, Any]], str | None] | None = None,
) -> CardSequenceResult:
    """Execute a queued card sequence using initial hand/enemy bindings.

    This mirrors the interactive CLI sequence semantics: the requested card and
    target are bound to the initial state, then each play is remapped to the
    current hand/enemy index after prior cards have resolved.
    """

    current = state
    plan, plan_error = _card_sequence_plan(state, sequence)
    if plan_error:
        return CardSequenceResult(current, 0, plan_error)

    steps_executed = 0
    commands_executed: list[dict[str, Any]] = []
    for step in plan:
        if current.get("decision") != "combat_play":
            return CardSequenceResult(current, steps_executed, "manual decision is required")

        hand = current.get("hand", [])
        energy = current.get("energy", 0)
        enemies = current.get("enemies", [])
        card_index = step.get("initial_card_index")
        requested_target_index = step.get("target_index")
        target_index = _sequence_target_index(enemies, step)
        has_explicit_target = requested_target_index is not None
        card = _find_sequence_card(hand, step)
        if card is None:
            return CardSequenceResult(current, steps_executed, f"card index {card_index} is no longer in hand")
        if stop_if_card is not None:
            stopped_reason = stop_if_card(card)
            if stopped_reason:
                return CardSequenceResult(current, steps_executed, stopped_reason, commands_executed)
        if not card.get("can_play") or _card_cost_value(card) > energy:
            return CardSequenceResult(current, steps_executed, f"{_name(card.get('name'))} cannot be played now")

        args = {"card_index": card["index"]}
        if card.get("target_type") == "AnyEnemy":
            live_enemies = [enemy for enemy in enemies if enemy.get("hp", 0) > 0]
            if has_explicit_target:
                target = next((enemy for enemy in live_enemies if enemy.get("index") == target_index), None)
                if target is None:
                    return CardSequenceResult(
                        current,
                        steps_executed,
                        f"enemy target {requested_target_index} is not available",
                    )
                args["target_index"] = target_index
            elif len(live_enemies) == 1:
                args["target_index"] = live_enemies[0]["index"]
            else:
                return CardSequenceResult(current, steps_executed, f"{_name(card.get('name'))} needs a target")
        elif has_explicit_target:
            return CardSequenceResult(current, steps_executed, f"{_name(card.get('name'))} does not take an enemy target")

        command = {"cmd": "action", "action": "play_card", "args": args}
        current = send_fn(command)
        steps_executed += 1
        commands_executed.append(command)
        if not current:
            return CardSequenceResult(current or {}, steps_executed, "manual decision is required", commands_executed)
        if current.get("type") == "error":
            return CardSequenceResult(current, steps_executed, current.get("message") or "engine error", commands_executed)
        if current.get("decision") != "combat_play":
            if current.get("decision") in {"combat_reward", "event_choice", "map_select", "game_over"}:
                return CardSequenceResult(current, steps_executed, commands_executed=commands_executed)
            return CardSequenceResult(current, steps_executed, "manual decision is required", commands_executed)
    return CardSequenceResult(current, steps_executed, commands_executed=commands_executed)


def normalize_card_sequence_payload(value: Any) -> list[dict[str, int]]:
    """Validate and normalize a model-proposed sequence payload."""

    if not isinstance(value, list):
        raise ValueError("sequence must be a list of card steps")
    sequence: list[dict[str, int]] = []
    for index, raw_step in enumerate(value):
        if not isinstance(raw_step, dict):
            raise ValueError(f"sequence step {index} must be an object")
        card_index = raw_step.get("card_index")
        if type(card_index) is not int:
            raise ValueError(
                f"sequence step {index} must include top-level integer card_index. "
                "Do not wrap sequence steps in cmd, action, or args objects"
            )
        step = {"card_index": card_index}
        target_index = raw_step.get("target_index")
        if target_index is not None:
            if type(target_index) is not int:
                raise ValueError(f"sequence step {index} target_index must be an integer")
            step["target_index"] = target_index
        sequence.append(step)
    return sequence


def _card_sequence_plan(
    initial_state: dict[str, Any],
    sequence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    initial_hand = initial_state.get("hand", [])
    initial_by_index = {card.get("index"): card for card in initial_hand}
    exports_instances = any("instance_id" in card for card in initial_hand)
    initial_enemies = initial_state.get("enemies", [])
    initial_enemy_by_index = {enemy.get("index"): enemy for enemy in initial_enemies}
    exports_enemy_instances = any("instance_id" in enemy for enemy in initial_enemies)
    plan = []
    for step in sequence:
        card_index = step.get("card_index")
        target_index = step.get("target_index")
        initial_card = initial_by_index.get(card_index)
        if initial_card is None and exports_instances:
            return None, f"card index {card_index} is not in the current hand"
        if initial_card is not None and str(initial_card.get("target_type") or "") != "AnyEnemy":
            target_index = None
        initial_target = initial_enemy_by_index.get(target_index) if target_index is not None else None
        if target_index is not None and initial_target is None and exports_enemy_instances:
            return None, f"enemy target {target_index} is not in the current enemy list"
        plan.append(
            {
                "initial_card_index": card_index,
                "target_index": target_index,
                "instance_id": initial_card.get("instance_id") if initial_card is not None else None,
                "target_instance_id": initial_target.get("instance_id") if initial_target is not None else None,
            }
        )
    return plan, None


def _find_sequence_card(current_hand: list[dict[str, Any]], step: dict[str, Any]) -> dict[str, Any] | None:
    instance_id = step.get("instance_id")
    if instance_id is not None:
        return next((card for card in current_hand if card.get("instance_id") == instance_id), None)
    card_index = step.get("initial_card_index")
    return next((card for card in current_hand if card.get("index") == card_index), None)


def _sequence_target_index(current_enemies: list[dict[str, Any]], step: dict[str, Any]) -> int | None:
    target_instance_id = step.get("target_instance_id")
    if target_instance_id is not None:
        target = next(
            (
                enemy
                for enemy in current_enemies
                if enemy.get("instance_id") == target_instance_id and enemy.get("hp", 0) > 0
            ),
            None,
        )
        return target.get("index") if target is not None else None
    return step.get("target_index")


def _card_cost_value(card: dict[str, Any]) -> int:
    cost = card.get("energy_cost", card.get("cost", 99))
    if isinstance(cost, (int, float)):
        return int(cost)
    if isinstance(cost, str) and cost.upper() == "X":
        return 0
    return 99


def _name(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("en", "name", "title", "id"):
            value = obj.get(key)
            if value:
                return str(value)
        return "?"
    return str(obj) if obj is not None else "?"
