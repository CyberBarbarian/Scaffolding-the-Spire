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


_CUSTOM_SEQUENCE_BLOCKED_TEXT_MARKERS = ("random", "choose", "select", "discover")


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
                "instance_id": card.get("instance_id"),
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
                "instance_id": enemy.get("instance_id"),
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
                    "instance_id": item.get("instance_id"),
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
    text = _card_search_text(card)
    if any(marker in text for marker in _CUSTOM_SEQUENCE_BLOCKED_TEXT_MARKERS):
        return f"non-deterministic or manual-choice card is not eligible for custom sequence: {_name(card.get('name'))}"
    return None


def _card_search_text(card: dict[str, Any]) -> str:
    parts = [
        _name(card.get("name")),
        str(card.get("id") or ""),
        str(card.get("description") or ""),
        str(card.get("raw_description") or ""),
        str(card.get("text") or ""),
    ]
    keywords = card.get("keywords")
    if isinstance(keywords, list):
        parts.extend(str(keyword) for keyword in keywords)
    return " ".join(parts).lower()


def _copy_command(command: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(command, ensure_ascii=False))
