"""Engine-backed validation for model-proposed card sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .process import Sts2Process
from .sequences import (
    MAX_CUSTOM_SEQUENCE_VALIDATION_LENGTH,
    execute_card_sequence,
    normalize_card_sequence_payload,
)
from .sequence_safety import sequence_card_block_reason


@dataclass(frozen=True)
class SequenceValidationResult:
    validated: bool
    reason: str = ""
    final_state: dict[str, Any] | None = None
    steps_executed: int = 0
    commands_executed: list[dict[str, Any]] = field(default_factory=list)


def validate_custom_sequence(
    sequence_payload: Any,
    *,
    current_state: dict[str, Any],
    action_tape: list[dict[str, Any]],
    character: str,
    seed: str,
    ascension: int,
    lang: str,
    process_factory: Callable[[], Sts2Process] | None = None,
) -> SequenceValidationResult:
    try:
        sequence = normalize_card_sequence_payload(sequence_payload)
    except ValueError as exc:
        return SequenceValidationResult(False, str(exc))

    if len(sequence) < 2:
        return SequenceValidationResult(False, "custom sequence must contain at least 2 card steps")
    if len(sequence) > MAX_CUSTOM_SEQUENCE_VALIDATION_LENGTH:
        return SequenceValidationResult(
            False,
            f"custom sequence validation accepts at most {MAX_CUSTOM_SEQUENCE_VALIDATION_LENGTH} card steps",
        )
    preflight_error = _non_atomic_sequence_preflight(sequence, current_state)
    if preflight_error:
        return SequenceValidationResult(False, preflight_error)
    preflight_error = _modifier_changing_sequence_preflight(sequence, current_state)
    if preflight_error:
        return SequenceValidationResult(False, preflight_error)
    preflight_error = _target_specific_lethal_preflight(sequence, current_state)
    if preflight_error:
        return SequenceValidationResult(False, preflight_error)
    preflight_error = _one_shot_free_attack_preflight(sequence, current_state)
    if preflight_error:
        return SequenceValidationResult(False, preflight_error)

    factory = process_factory or Sts2Process
    proc = factory()
    try:
        proc.start()
        replay_state = proc.start_run(character=character, seed=seed, ascension=ascension, lang=lang)
        for command in action_tape:
            replay_state = proc.send(_copy_command(command))
            if replay_state.get("type") == "error":
                return SequenceValidationResult(
                    False,
                    f"probe replay failed: {replay_state.get('message') or 'engine error'}",
                    final_state=replay_state,
                )

        if state_fingerprint(replay_state) != state_fingerprint(current_state):
            return SequenceValidationResult(False, "probe replay drifted from current state", final_state=replay_state)

        sequence_result = execute_card_sequence(
            replay_state,
            sequence,
            proc.send,
            stop_if_card=_custom_sequence_block_reason,
        )
        if sequence_result.stopped_reason:
            return SequenceValidationResult(
                False,
                f"custom sequence stopped after {sequence_result.steps_executed} step(s): {sequence_result.stopped_reason}",
                final_state=sequence_result.state,
                steps_executed=sequence_result.steps_executed,
                commands_executed=sequence_result.commands_executed,
            )
        reason = ""
        if sequence_result.steps_executed != len(sequence):
            reason = (
                f"custom sequence completed after validated prefix "
                f"{sequence_result.steps_executed} of {len(sequence)} step(s) "
                f"because decision became {sequence_result.state.get('decision')}"
            )
        return SequenceValidationResult(
            True,
            reason=reason,
            final_state=sequence_result.state,
            steps_executed=sequence_result.steps_executed,
            commands_executed=sequence_result.commands_executed,
        )
    finally:
        close = getattr(proc, "close", None)
        if callable(close):
            close()


def state_fingerprint(state: dict[str, Any]) -> str:
    payload = _canonical_state(state)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _non_atomic_sequence_preflight(sequence: list[dict[str, int]], current_state: dict[str, Any]) -> str | None:
    hand = current_state.get("hand") if isinstance(current_state.get("hand"), list) else []
    hand_by_index = {card.get("index"): card for card in hand if isinstance(card, dict)}

    for step_number, step in enumerate(sequence, start=1):
        card = hand_by_index.get(step.get("card_index"))
        if not isinstance(card, dict):
            continue
        block_reason = sequence_card_block_reason(card, allowed_target_types={"None", "Self", "AnyEnemy", "AllEnemies"})
        if block_reason:
            return f"step {step_number}: {block_reason}. Play this card as a single action and wait for the next state"
    return None


def _target_specific_lethal_preflight(sequence: list[dict[str, int]], current_state: dict[str, Any]) -> str | None:
    hand = current_state.get("hand") if isinstance(current_state.get("hand"), list) else []
    enemies = current_state.get("enemies") if isinstance(current_state.get("enemies"), list) else []
    hand_by_index = {card.get("index"): card for card in hand if isinstance(card, dict)}
    enemies_by_index = {enemy.get("index"): enemy for enemy in enemies if isinstance(enemy, dict)}
    cumulative_by_target: dict[int, float] = {}
    lethal_step_by_target: dict[int, int] = {}

    for step_number, step in enumerate(sequence, start=1):
        card = hand_by_index.get(step.get("card_index"))
        target_index = step.get("target_index")
        if not isinstance(card, dict) or target_index is None:
            continue
        if str(card.get("target_type") or "") != "AnyEnemy":
            continue
        if target_index in lethal_step_by_target:
            return (
                "target-specific damage_by_target preview kills enemy target "
                f"{target_index} after step {lethal_step_by_target[target_index]}; "
                f"later step {step_number} still targets it"
            )
        target = enemies_by_index.get(target_index)
        hp = _numeric_value(target.get("hp") if isinstance(target, dict) else None)
        damage = _target_specific_damage(card, target_index)
        if hp is None or hp <= 0 or damage is None or damage <= 0:
            continue
        cumulative = cumulative_by_target.get(target_index, 0.0) + damage
        cumulative_by_target[target_index] = cumulative
        if cumulative >= hp:
            lethal_step_by_target[target_index] = step_number

    return None


def _one_shot_free_attack_preflight(sequence: list[dict[str, int]], current_state: dict[str, Any]) -> str | None:
    charges = _one_shot_free_attack_charges(current_state)
    if charges is None:
        return None

    hand = current_state.get("hand") if isinstance(current_state.get("hand"), list) else []
    hand_by_index = {card.get("index"): card for card in hand if isinstance(card, dict)}
    attack_steps = 0
    for step in sequence:
        card = hand_by_index.get(step.get("card_index"))
        if isinstance(card, dict) and _is_attack_card(card):
            attack_steps += 1

    if attack_steps <= charges:
        return None
    return (
        f"one-shot free attack power covers only {charges} attack step(s); "
        f"custom sequence contains {attack_steps} attack steps"
    )


def _modifier_changing_sequence_preflight(sequence: list[dict[str, int]], current_state: dict[str, Any]) -> str | None:
    hand = current_state.get("hand") if isinstance(current_state.get("hand"), list) else []
    hand_by_index = {card.get("index"): card for card in hand if isinstance(card, dict)}

    for step_number, step in enumerate(sequence, start=1):
        card = hand_by_index.get(step.get("card_index"))
        if not isinstance(card, dict):
            continue
        target_index = step.get("target_index")
        if target_index is not None and _changes_target_damage_modifiers(card):
            later_step = _later_attack_against_target(sequence, hand_by_index, start=step_number, target_index=target_index)
            if later_step is not None:
                return (
                    f"step {step_number} changes enemy target {target_index} damage modifiers; "
                    f"later step {later_step} still attacks that target. Stop the sequence at the "
                    "modifier-changing step and wait for the next state"
                )
        if _changes_player_damage_modifiers(card):
            later_step = _later_damage_or_block_step(sequence, hand_by_index, start=step_number)
            if later_step is not None:
                return (
                    f"step {step_number} changes player damage/block modifiers; later step {later_step} "
                    "depends on recalculated previews. Stop the sequence at the modifier-changing step "
                    "and wait for the next state"
                )

    return None


def _later_attack_against_target(
    sequence: list[dict[str, int]],
    hand_by_index: dict[Any, dict[str, Any]],
    *,
    start: int,
    target_index: int,
) -> int | None:
    for later_step_number, later_step in enumerate(sequence[start:], start=start + 1):
        later_card = hand_by_index.get(later_step.get("card_index"))
        if not isinstance(later_card, dict) or not _is_damage_card(later_card):
            continue
        target_type = str(later_card.get("target_type") or "")
        if target_type == "AllEnemies" or later_step.get("target_index") == target_index:
            return later_step_number
    return None


def _later_damage_or_block_step(
    sequence: list[dict[str, int]],
    hand_by_index: dict[Any, dict[str, Any]],
    *,
    start: int,
) -> int | None:
    for later_step_number, later_step in enumerate(sequence[start:], start=start + 1):
        later_card = hand_by_index.get(later_step.get("card_index"))
        if isinstance(later_card, dict) and (_is_damage_card(later_card) or _is_block_card(later_card)):
            return later_step_number
    return None


def _changes_target_damage_modifiers(card: dict[str, Any]) -> bool:
    keys = _stat_keys(card)
    return any("vulnerable" in key or "weak" in key for key in keys)


def _changes_player_damage_modifiers(card: dict[str, Any]) -> bool:
    keys = _stat_keys(card)
    return any("strength" in key or "dexterity" in key for key in keys)


def _is_damage_card(card: dict[str, Any]) -> bool:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    target_type = str(card.get("target_type") or "")
    return (
        str(card.get("type") or "") == "Attack"
        or target_type in {"AnyEnemy", "AllEnemies"}
        or any(key in stats for key in ("damage", "calculateddamage", "damage_by_target", "calculateddamage_by_target"))
    )


def _is_attack_card(card: dict[str, Any]) -> bool:
    return str(card.get("type") or "") == "Attack"


def _is_block_card(card: dict[str, Any]) -> bool:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    return any("block" in str(key).lower() for key in stats)


def _stat_keys(card: dict[str, Any]) -> set[str]:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    return {str(key).lower() for key in stats}


def _target_specific_damage(card: dict[str, Any], target_index: int) -> float | None:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    for damage_by_target_key in ("damage_by_target", "calculateddamage_by_target"):
        damage = _target_specific_damage_from_entries(stats.get(damage_by_target_key), target_index)
        if damage is not None:
            return damage
    return None


def _target_specific_damage_from_entries(damage_by_target: Any, target_index: int) -> float | None:
    entry: Any = None
    if isinstance(damage_by_target, list):
        entry = next(
            (
                item
                for item in damage_by_target
                if isinstance(item, dict) and item.get("target_index") == target_index
            ),
            None,
        )
    elif isinstance(damage_by_target, dict):
        if target_index in damage_by_target:
            entry = damage_by_target[target_index]
        elif str(target_index) in damage_by_target:
            entry = damage_by_target[str(target_index)]
    if isinstance(entry, dict):
        for key in ("unblocked_damage", "damage", "calculateddamage"):
            value = entry.get(key)
            if value is not None:
                return _numeric_value(value)
        return None
    return _numeric_value(entry)


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _one_shot_free_attack_charges(state: dict[str, Any]) -> int | None:
    for power in _player_powers(state):
        text = _power_search_text(power)
        if "free_attack_power" not in text and not (
            "next attack" in text and ("costs 0" in text or "cost 0" in text or "free" in text)
        ):
            continue
        amount = _numeric_value(power.get("amount"))
        if amount is None:
            return 1
        return max(1, int(amount))
    return None


def _player_powers(state: dict[str, Any]) -> list[dict[str, Any]]:
    direct = state.get("player_powers")
    if isinstance(direct, list):
        return [power for power in direct if isinstance(power, dict)]
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    powers = player.get("powers") if isinstance(player.get("powers"), list) else []
    return [power for power in powers if isinstance(power, dict)]


def _power_search_text(power: dict[str, Any]) -> str:
    parts = [
        str(power.get("id") or ""),
        _name(power.get("name")),
        str(power.get("description") or ""),
        str(power.get("raw_description") or ""),
    ]
    return " ".join(parts).replace("[E]", "0").lower()


def _canonical_state(state: dict[str, Any]) -> dict[str, Any]:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    return {
        "decision": state.get("decision"),
        "type": state.get("type"),
        "context": {
            "act": state.get("act") or context.get("act"),
            "floor": state.get("floor") or context.get("floor"),
            "room_type": context.get("room_type"),
        },
        "round": state.get("round"),
        "energy": state.get("energy"),
        "max_energy": state.get("max_energy"),
        "player": {
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block", 0),
            "gold": player.get("gold"),
            "deck_size": player.get("deck_size"),
            "powers": _canonical_items(state.get("player_powers") or player.get("powers") or []),
            "potions": _canonical_items(player.get("potions") or state.get("potions") or []),
            "relics": _canonical_items(player.get("relics") or []),
        },
        "piles": {
            "draw": state.get("draw_pile_count"),
            "discard": state.get("discard_pile_count"),
            "exhaust": state.get("exhaust_pile_count"),
            "draw_cards": _canonical_cards(state.get("draw_pile") or []),
            "discard_cards": _canonical_cards(state.get("discard_pile") or []),
            "exhaust_cards": _canonical_cards(state.get("exhaust_pile") or []),
        },
        "hand": _canonical_cards(state.get("hand") or []),
        "enemies": _canonical_enemies(state.get("enemies") or []),
        "cards": _canonical_cards(state.get("cards") or []),
    }


def _canonical_cards(cards: list[Any]) -> list[dict[str, Any]]:
    result = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        result.append(
            {
                "index": card.get("index"),
                "name": _name(card.get("name")),
                "cost": card.get("cost"),
                "energy_cost": card.get("energy_cost"),
                "type": card.get("type"),
                "target_type": card.get("target_type"),
                "can_play": card.get("can_play"),
                "stats": card.get("stats") or {},
            }
        )
    return result


def _canonical_enemies(enemies: list[Any]) -> list[dict[str, Any]]:
    result = []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        result.append(
            {
                "index": enemy.get("index"),
                "name": _name(enemy.get("name")),
                "hp": enemy.get("hp"),
                "max_hp": enemy.get("max_hp"),
                "block": enemy.get("block", 0),
                "intents": enemy.get("intents") or [],
                "powers": _canonical_items(enemy.get("powers") or []),
            }
        )
    return result


def _canonical_items(items: list[Any]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        if not item:
            continue
        if isinstance(item, dict):
            result.append(
                {
                    "index": item.get("index", item.get("slot_index", item.get("slot"))),
                    "name": _name(item.get("name")),
                    "id": item.get("id"),
                    "amount": item.get("amount"),
                    "counter": item.get("counter", item.get("display_amount")),
                }
            )
        else:
            result.append({"name": str(item)})
    return result


def _name(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("en", "name", "title", "id"):
            value = obj.get(key)
            if value:
                return str(value)
        return "?"
    return str(obj) if obj is not None else "?"


def _custom_sequence_block_reason(card: dict[str, Any]) -> str | None:
    return sequence_card_block_reason(card, allowed_target_types={"None", "Self", "AnyEnemy", "AllEnemies"})


def _copy_command(command: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(command, ensure_ascii=False))
