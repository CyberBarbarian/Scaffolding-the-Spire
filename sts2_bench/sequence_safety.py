"""Shared safety checks for queued combat card sequences."""

from __future__ import annotations

from typing import Any


NON_ATOMIC_SEQUENCE_TEXT_MARKERS = (
    "draw",
    "random",
    "discard",
    "exhaust 1 card",
    "exhaust a card",
    "exhaust all",
    "exhaust your",
    "exhaust the",
    "choose",
    "select",
    "shuffle",
    "generate",
    "discover",
    "upgrade",
)
NON_ATOMIC_SEQUENCE_STAT_MARKERS = (
    "cards",
    "draw",
    "discard",
    "exhaust",
    "random",
    "upgrade",
    "shuffle",
    "generate",
    "discover",
)
NON_ATOMIC_TARGET_MARKERS = ("random", "choose", "select")


def sequence_card_block_reason(
    card: dict[str, Any],
    *,
    allowed_target_types: set[str] | None = None,
) -> str | None:
    """Return why a card cannot be part of a queued sequence, or None if atomic."""

    card_name = _name(card.get("name"))
    target_type = str(card.get("target_type") or "")
    target_marker = _first_marker(target_type.lower(), NON_ATOMIC_TARGET_MARKERS)
    if target_marker is not None:
        return f"non-deterministic or non-atomic card is not eligible for sequence: {card_name} uses {target_marker} target"
    if allowed_target_types is not None and target_type not in allowed_target_types:
        return f"unsupported target type is not eligible for sequence: {card_name} target_type={target_type or '?'}"

    text = _card_search_text(card)
    marker = _first_marker(text, NON_ATOMIC_SEQUENCE_TEXT_MARKERS)
    if marker is not None:
        return f"non-deterministic or non-atomic card is not eligible for sequence: {card_name} uses {marker}"

    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    for key in stats:
        marker = _first_marker(str(key).lower(), NON_ATOMIC_SEQUENCE_STAT_MARKERS)
        if marker is not None:
            return f"non-deterministic or non-atomic card is not eligible for sequence: {card_name} uses {marker}"
    return None


def is_sequence_atomic_card(
    card: dict[str, Any],
    *,
    allowed_target_types: set[str] | None = None,
) -> bool:
    return sequence_card_block_reason(card, allowed_target_types=allowed_target_types) is None


def _first_marker(text: str, markers: tuple[str, ...]) -> str | None:
    return next((marker for marker in markers if marker in text), None)


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


def _name(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("en", "name", "title", "id"):
            value = obj.get(key)
            if value:
                return str(value)
        return "?"
    return str(obj) if obj is not None else "?"
