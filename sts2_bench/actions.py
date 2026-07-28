"""Legal-action generation for benchmark and RL policies."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Literal

from .sequence_safety import is_sequence_atomic_card


MAX_CARD_SEQUENCE_LENGTH = 3
MAX_CARD_SEQUENCE_ACTIONS = 24
MAX_MATERIALIZED_CARD_SELECT_ACTIONS = 256
CustomSequenceExposure = Literal["safe", "eligible"]


class UnsupportedDecisionError(RuntimeError):
    """Raised when Headless exposes a decision the benchmark cannot safely execute."""


@dataclass(frozen=True)
class LegalAction:
    """One action exposed to a model or policy.

    ``command`` is either the exact JSON command sent to ``Sts2Headless`` or a
    benchmark-local ``bench_view`` command. ``label`` is intentionally plain
    text because it is shown to LLMs and log readers.
    """

    action_id: int
    label: str
    command: dict[str, Any]
    kind: str

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "label": self.label,
        }

    def to_log_dict(self) -> dict[str, Any]:
        return {
            **self.to_prompt_dict(),
            "cmd": self.command,
        }


def _action(label: str, action: str, kind: str, **args: Any) -> dict[str, Any]:
    command: dict[str, Any] = {"cmd": "action", "action": action}
    if args:
        command["args"] = args
    return {"label": label, "command": command, "kind": kind}


def _card_reward_skip_label(reward_index: Any, *, already_handled: bool) -> str:
    if already_handled:
        return f"confirm skipped card reward {reward_index}"
    return f"skip unopened card reward {reward_index}"


def _view_action(label: str, view: str, kind: str) -> dict[str, Any]:
    return {
        "label": label,
        "command": {"cmd": "bench_view", "view": view},
        "kind": kind,
    }


def _sequence_action(label: str, sequence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": label,
        "command": {"cmd": "bench_sequence", "sequence": sequence},
        "kind": "combat_card_sequence",
    }


def _custom_sequence_action() -> dict[str, Any]:
    return {
        "label": "propose custom deterministic card sequence",
        "command": {"cmd": "bench_custom_sequence"},
        "kind": "combat_custom_sequence",
    }


def _name(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("en", "name", "title", "id"):
            value = obj.get(key)
            if value:
                return str(value)
        return "?"
    return str(obj) if obj is not None else "?"


def _price(item: dict[str, Any]) -> int:
    for key in ("price", "gold_cost", "cost"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 10**9


def _card_cost_value(card: dict[str, Any]) -> int:
    cost = card.get("energy_cost", card.get("cost", 99))
    if isinstance(cost, (int, float)):
        return int(cost)
    if isinstance(cost, str) and cost.upper() == "X":
        return 0
    return 99


def _single_card_kills_only_enemy(state: dict[str, Any], playable_cards: list[dict[str, Any]]) -> bool:
    live_enemies = [
        enemy
        for enemy in state.get("enemies", []) or []
        if isinstance(enemy, dict) and enemy.get("hp", 0) > 0 and enemy.get("targetable", True) is not False
    ]
    if len(live_enemies) != 1:
        return False
    enemy = live_enemies[0]
    target_index = enemy.get("index")
    hp = _numeric_value(enemy.get("hp"))
    if target_index is None or hp is None:
        return False
    for card in playable_cards:
        if str(card.get("target_type") or "") not in {"AnyEnemy", "AllEnemies"}:
            continue
        damage = _target_specific_damage(card, target_index)
        if damage is not None and damage >= hp:
            return True
    return False


def _should_expose_custom_sequence(state: dict[str, Any], playable_cards: list[dict[str, Any]]) -> bool:
    if not _has_custom_sequence_entrypoint(state, playable_cards):
        return False
    if any(not _is_sequence_safe_card(card) for card in playable_cards):
        return False
    if _has_insufficient_one_shot_free_attack_charges(state, playable_cards):
        return False
    if _single_card_kills_only_enemy(state, playable_cards):
        return False
    if _has_live_enemy_powers(state):
        return False
    if _has_modifier_changing_card(playable_cards):
        return False
    if _has_target_specific_damage_adjustment(state, playable_cards):
        return False
    if _has_cumulative_attack_lethal(state, playable_cards):
        return False
    return True


def _should_expose_custom_sequence_for_mode(
    state: dict[str, Any],
    playable_cards: list[dict[str, Any]],
    *,
    exposure: CustomSequenceExposure,
) -> bool:
    if exposure == "safe":
        return _should_expose_custom_sequence(state, playable_cards)
    if exposure == "eligible":
        return _has_custom_sequence_entrypoint(state, playable_cards)
    raise ValueError(f"Unsupported custom_sequence_exposure: {exposure}")


def _has_custom_sequence_entrypoint(state: dict[str, Any], playable_cards: list[dict[str, Any]]) -> bool:
    hand_cards = [card for card in state.get("hand", []) or [] if isinstance(card, dict)]
    if len(hand_cards) < 2:
        return False
    if len(playable_cards) >= 2:
        return True
    return any(_card_grants_energy(card) for card in playable_cards)


def _card_grants_energy(card: dict[str, Any]) -> bool:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    for key, value in stats.items():
        if "energy" not in str(key).lower():
            continue
        amount = _numeric_value(value)
        if amount is not None and amount > 0:
            return True
    return False


def _has_live_enemy_powers(state: dict[str, Any]) -> bool:
    for enemy in state.get("enemies", []) or []:
        if not isinstance(enemy, dict) or enemy.get("hp", 0) <= 0:
            continue
        if enemy.get("powers"):
            return True
    return False


def _has_insufficient_one_shot_free_attack_charges(
    state: dict[str, Any],
    playable_cards: list[dict[str, Any]],
) -> bool:
    charges = _one_shot_free_attack_charges(state)
    if charges is None:
        return False
    playable_attacks = sum(1 for card in playable_cards if _is_attack_card(card))
    return playable_attacks > charges


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


def _is_attack_card(card: dict[str, Any]) -> bool:
    return str(card.get("type") or "") == "Attack"


def _has_modifier_changing_card(playable_cards: list[dict[str, Any]]) -> bool:
    for card in playable_cards:
        stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
        keys = {str(key).lower() for key in stats}
        if any(marker in key for key in keys for marker in ("vulnerable", "weak", "strength", "dexterity")):
            return True
    return False


def _has_target_specific_damage_adjustment(state: dict[str, Any], playable_cards: list[dict[str, Any]]) -> bool:
    live_enemies = [
        enemy
        for enemy in state.get("enemies", []) or []
        if isinstance(enemy, dict) and enemy.get("hp", 0) > 0 and enemy.get("targetable", True) is not False
    ]
    for card in playable_cards:
        if str(card.get("target_type") or "") not in {"AnyEnemy", "AllEnemies"}:
            continue
        base_damage = _base_card_damage(card)
        stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
        for enemy in live_enemies:
            target_index = enemy.get("index")
            if target_index is None:
                continue
            for key in ("damage_by_target", "calculateddamage_by_target"):
                entry = _target_specific_entry(stats.get(key), target_index)
                if _target_specific_entry_is_adjusted(entry, base_damage):
                    return True
    return False


def _has_cumulative_attack_lethal(state: dict[str, Any], playable_cards: list[dict[str, Any]]) -> bool:
    live_enemies = [
        enemy
        for enemy in state.get("enemies", []) or []
        if isinstance(enemy, dict) and enemy.get("hp", 0) > 0 and enemy.get("targetable", True) is not False
    ]
    for enemy in live_enemies:
        target_index = enemy.get("index")
        hp = _numeric_value(enemy.get("hp"))
        if target_index is None or hp is None:
            continue
        total = 0.0
        for card in playable_cards:
            if str(card.get("target_type") or "") not in {"AnyEnemy", "AllEnemies"}:
                continue
            damage = _target_specific_damage(card, target_index)
            if damage is None:
                damage = _base_card_damage(card)
            if damage is not None and damage > 0:
                total += damage
        if total >= hp:
            return True
    return False


def _base_card_damage(card: dict[str, Any]) -> float | None:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    for key in ("damage", "calculateddamage"):
        value = _numeric_value(stats.get(key))
        if value is not None:
            return value
    return None


def _target_specific_entry_is_adjusted(entry: Any, base_damage: float | None) -> bool:
    if entry is None:
        return False
    if isinstance(entry, dict):
        block = _numeric_value(entry.get("block"))
        if block is not None and block > 0:
            return True
        for key in ("unblocked_damage", "damage", "calculateddamage"):
            damage = _numeric_value(entry.get(key))
            if damage is not None:
                return base_damage is not None and damage != base_damage
        return False
    damage = _numeric_value(entry)
    return damage is not None and base_damage is not None and damage != base_damage


def _target_specific_damage(card: dict[str, Any], target_index: int) -> float | None:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    for key in ("damage_by_target", "calculateddamage_by_target"):
        damage = _target_specific_damage_from_entries(stats.get(key), target_index)
        if damage is not None:
            return damage
    return None


def _target_specific_damage_from_entries(damage_by_target: Any, target_index: int) -> float | None:
    entry = _target_specific_entry(damage_by_target, target_index)
    if isinstance(entry, dict):
        for key in ("unblocked_damage", "damage", "calculateddamage"):
            value = entry.get(key)
            if value is not None:
                return _numeric_value(value)
        return None
    return _numeric_value(entry)


def _target_specific_entry(damage_by_target: Any, target_index: int) -> Any:
    if isinstance(damage_by_target, list):
        return next(
            (
                item
                for item in damage_by_target
                if isinstance(item, dict) and item.get("target_index") == target_index
            ),
            None,
        )
    if isinstance(damage_by_target, dict):
        if target_index in damage_by_target:
            return damage_by_target[target_index]
        if str(target_index) in damage_by_target:
            return damage_by_target[str(target_index)]
    return None


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_sequence_safe_card(card: dict[str, Any]) -> bool:
    return is_sequence_atomic_card(card, allowed_target_types={"None", "Self", "AnyEnemy"})


def _sequence_step_label(card: dict[str, Any], enemy: dict[str, Any] | None = None) -> str:
    label = f"{card.get('index')} {_name(card.get('name'))}"
    if enemy is not None:
        label += f" on enemy {enemy.get('index')}"
    return label


def _combat_card_sequence_actions(state: dict[str, Any], playable_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    energy = int(state.get("energy", 0) or 0)
    enemies = [enemy for enemy in state.get("enemies", []) if enemy.get("hp", 0) > 0]
    candidates: list[dict[str, Any]] = []
    for card in playable_cards:
        if not _is_sequence_safe_card(card):
            continue
        target_type = str(card.get("target_type") or "")
        if target_type == "AnyEnemy":
            for enemy in enemies:
                candidates.append(
                    {
                        "card": card,
                        "cost": _card_cost_value(card),
                        "step": {"card_index": card.get("index"), "target_index": enemy.get("index")},
                        "label": _sequence_step_label(card, enemy),
                    }
                )
        else:
            candidates.append(
                {
                    "card": card,
                    "cost": _card_cost_value(card),
                    "step": {"card_index": card.get("index")},
                    "label": _sequence_step_label(card),
                }
            )

    actions: list[dict[str, Any]] = []
    max_length = min(MAX_CARD_SEQUENCE_LENGTH, len({item["card"].get("index") for item in candidates}))
    for length in range(2, max_length + 1):
        for ordered in itertools.permutations(candidates, length):
            card_indices = [item["card"].get("index") for item in ordered]
            if len(set(card_indices)) != len(card_indices):
                continue
            if sum(int(item["cost"]) for item in ordered) > energy:
                continue
            label = "play sequence: " + " -> ".join(str(item["label"]) for item in ordered)
            actions.append(_sequence_action(label, [dict(item["step"]) for item in ordered]))
            if len(actions) >= MAX_CARD_SEQUENCE_ACTIONS:
                return actions
    return actions


def _card_select_label(cards: list[dict[str, Any]]) -> str:
    indices = ",".join(str(card.get("index")) for card in cards)
    names = " + ".join(_name(card.get("name")) for card in cards)
    return f"select cards {indices}: {names}"


def card_select_combination_count(card_count: int, min_select: int, max_select: int) -> int:
    """Return the semantic non-empty card-selection count without enumerating it."""

    upper = min(max_select, card_count)
    lower = max(1, min_select)
    if card_count <= 0 or upper < lower:
        return 0
    return sum(math.comb(card_count, count) for count in range(lower, upper + 1))


def _bundle_label(bundle: dict[str, Any]) -> str:
    index = bundle.get("index")
    title = _name(bundle.get("name") or bundle.get("title"))
    if title != "?":
        return f"select bundle {index}: {title}"

    cards = bundle.get("cards") or []
    card_names = [_name(card.get("name")) for card in cards if isinstance(card, dict)]
    card_names = [card_name for card_name in card_names if card_name != "?"]
    if card_names:
        return f"select bundle {index}: {', '.join(card_names)}"

    return f"select bundle {index}"


def _card_select_actions(cards: list[dict[str, Any]], min_select: int, max_select: int) -> list[dict[str, Any]]:
    if not cards or max_select <= 0:
        return []

    # Keep the action surface valid for fixed-size selections.  The engine
    # expects all required cards in one command, so exposing single-card picks
    # for min_select > 1 creates illegal benchmark actions.
    actions: list[dict[str, Any]] = []
    upper = min(max_select, len(cards))
    lower = max(1, min_select)
    combination_count = card_select_combination_count(len(cards), min_select, max_select)
    if combination_count > MAX_MATERIALIZED_CARD_SELECT_ACTIONS:
        # Keep one executable representative so generic/fallback policies still
        # have a safe action. LLM policies resolve any other valid subset from
        # selected_card_indices without materializing the combinatorial list.
        selected = list(cards[:lower])
        indices = ",".join(str(card.get("index")) for card in selected)
        return [
            _action(
                _card_select_label(selected),
                "select_cards",
                "card_select_space",
                indices=indices,
            )
        ]
    for count in range(lower, upper + 1):
        for combo in itertools.combinations(cards, count):
            selected = list(combo)
            indices = ",".join(str(card.get("index")) for card in selected)
            label = (
                f"select card {selected[0].get('index')}: {_name(selected[0].get('name'))}"
                if count == 1
                else _card_select_label(selected)
            )
            actions.append(_action(label, "select_cards", "card_select", indices=indices))
    return actions


def materialize_card_select_space_action(
    state: dict[str, Any],
    template: LegalAction,
    selected_indices: list[int],
) -> LegalAction:
    """Resolve one executable member of a compressed card-selection action space."""

    if template.kind != "card_select_space":
        raise ValueError("card selection template must have kind card_select_space")

    cards = [card for card in state.get("cards") or [] if isinstance(card, dict)]
    by_index: dict[int, dict[str, Any]] = {}
    for card in cards:
        raw_index = card.get("index")
        if isinstance(raw_index, bool):
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index in by_index:
            raise ValueError(f"current card selection contains duplicate candidate index {index}")
        by_index[index] = card

    selected_set = set(selected_indices)
    if any(index not in by_index for index in selected_indices):
        raise ValueError(
            "selected_card_indices do not identify current candidate cards: "
            + str(selected_indices)
        )
    min_select = max(1, int(state.get("min_select", 1) or 0))
    max_select = min(int(state.get("max_select", min_select) or min_select), len(cards))
    if len(selected_indices) < min_select or len(selected_indices) > max_select:
        raise ValueError(
            f"selected_card_indices count must be between {min_select} and {max_select}, got {len(selected_indices)}"
        )

    normalized_indices = [index for index in by_index if index in selected_set]
    selected_cards = [by_index[index] for index in normalized_indices]
    indices_text = ",".join(str(index) for index in normalized_indices)
    names = " + ".join(_name(card.get("name")) for card in selected_cards)
    label = (
        f"select card {normalized_indices[0]}: {names}"
        if len(normalized_indices) == 1
        else f"select cards {indices_text}: {names}"
    )
    return LegalAction(
        action_id=template.action_id,
        label=label,
        command={"cmd": "action", "action": "select_cards", "args": {"indices": indices_text}},
        kind="card_select",
    )


def action_matches_legal_surface(
    state: dict[str, Any],
    action: LegalAction,
    legal_actions: list[LegalAction],
) -> bool:
    """Return whether an action is a concrete member of the current legal surface."""

    if action in legal_actions:
        return True
    if action.kind != "card_select":
        return False

    templates = [
        candidate
        for candidate in legal_actions
        if candidate.kind == "card_select_space" and candidate.action_id == action.action_id
    ]
    if len(templates) != 1:
        return False

    raw_indices = (action.command.get("args") or {}).get("indices")
    if not isinstance(raw_indices, str) or not raw_indices:
        return False
    try:
        selected_indices = [int(value) for value in raw_indices.split(",") if value != ""]
    except ValueError:
        return False
    if len(selected_indices) != len(raw_indices.split(",")):
        return False

    try:
        expected = materialize_card_select_space_action(
            state,
            templates[0],
            selected_indices,
        )
    except ValueError:
        return False
    return action == expected


def _targeted_card_actions(state: dict[str, Any], card: dict[str, Any]) -> list[dict[str, Any]]:
    card_index = card.get("index")
    card_name = _name(card.get("name"))
    target_type = str(card.get("target_type") or "")
    enemies = [enemy for enemy in state.get("enemies", []) if enemy.get("hp", 0) > 0]

    if target_type == "AnyEnemy":
        return [
            _action(
                f"play card {card_index}: {card_name} on enemy {enemy.get('index')}: {_name(enemy.get('name'))}",
                "play_card",
                "combat_play_card",
                card_index=card_index,
                target_index=enemy.get("index"),
            )
            for enemy in enemies
        ]

    return [
        _action(
            f"play card {card_index}: {card_name}",
            "play_card",
            "combat_play_card",
            card_index=card_index,
        )
    ]


def build_legal_actions(
    state: dict[str, Any],
    *,
    allow_repeat_views: bool = False,
    allow_card_sequences: bool = True,
    allow_precomputed_card_sequences: bool = False,
    allow_custom_sequences: bool = True,
    custom_sequence_exposure: CustomSequenceExposure = "safe",
    blocked_card_reward_claims: set[Any] | None = None,
) -> list[LegalAction]:
    """Build legal actions from the current decision point.

    The resulting action ids are stable only for this one state.  Store the
    selected full command in logs, not just the id.
    """

    if state.get("engine_error") is True or state.get("error") == "engine_error":
        raise UnsupportedDecisionError(
            f"engine error state: {state.get('engine_error_reason') or state.get('message') or '?'}"
        )
    decision = state.get("decision")
    raw: list[dict[str, Any]] = []

    if decision != "game_over":
        if allow_repeat_views or not state.get("view_deck"):
            raw.append(_view_action("view deck", "deck", "view_deck"))
        if allow_repeat_views or not state.get("view_map"):
            raw.append(_view_action("view map and current position", "map", "view_map"))
        if allow_repeat_views or not state.get("view_draw_pile"):
            raw.append(_view_action("view draw pile", "draw", "view_draw_pile"))
        if allow_repeat_views or not state.get("view_discard_pile"):
            raw.append(_view_action("view discard pile", "discard", "view_discard_pile"))
        if allow_repeat_views or not state.get("view_exhaust_pile"):
            raw.append(_view_action("view exhaust pile", "exhaust", "view_exhaust_pile"))

    if decision == "map_select":
        for choice in state.get("choices", []):
            raw.append(
                _action(
                    f"go to {choice.get('type', '?')} at col={choice.get('col')} row={choice.get('row')}",
                    "select_map_node",
                    "map",
                    col=choice.get("col"),
                    row=choice.get("row"),
                )
            )

    elif decision == "combat_play":
        energy = state.get("energy", 0)
        playable_cards: list[dict[str, Any]] = []
        for card in state.get("hand", []):
            if card.get("can_play") is not True:
                continue
            if _card_cost_value(card) > energy:
                continue
            playable_cards.append(card)
            raw.extend(_targeted_card_actions(state, card))
        if allow_precomputed_card_sequences:
            raw.extend(_combat_card_sequence_actions(state, playable_cards))

        player = state.get("player") or {}
        for potion in player.get("potions", []) or state.get("potions", []) or []:
            if not potion:
                continue
            potion_index = potion.get("index", potion.get("slot_index"))
            if potion_index is None:
                continue
            potion_name = _name(potion.get("name"))
            target_type = str(potion.get("target_type") or "")
            if target_type == "AnyEnemy":
                for enemy in state.get("enemies", []):
                    if enemy.get("hp", 0) > 0:
                        raw.append(
                            _action(
                                f"use potion {potion_index}: {potion_name} on enemy {enemy.get('index')}: {_name(enemy.get('name'))}",
                                "use_potion",
                                "combat_use_potion",
                                potion_index=potion_index,
                                target_index=enemy.get("index"),
                            )
                        )
            else:
                raw.append(
                    _action(
                        f"use potion {potion_index}: {potion_name}",
                        "use_potion",
                        "combat_use_potion",
                        potion_index=potion_index,
                    )
                )

        raw.append(_action("end turn", "end_turn", "combat_end_turn"))
        if (
            playable_cards
            and allow_card_sequences
            and allow_custom_sequences
            and _should_expose_custom_sequence_for_mode(
                state,
                playable_cards,
                exposure=custom_sequence_exposure,
            )
        ):
            raw.append(_custom_sequence_action())

    elif decision == "combat_reward":
        rewards = state.get("rewards") or []
        blocked_card_reward_claims = blocked_card_reward_claims or set()
        for reward in rewards:
            reward_index = reward.get("index")
            reward_kind = reward.get("kind", reward.get("type_name", "?"))
            is_blocked_card_reward = reward_kind == "card_reward" and reward_index in blocked_card_reward_claims
            if reward.get("can_claim") is True and not is_blocked_card_reward:
                label = (
                    f"open card reward {reward_index}: reveal offered cards"
                    if reward_kind == "card_reward"
                    else f"claim reward {reward_index}: {reward_kind}"
                )
                raw.append(
                    _action(
                        label,
                        "claim_reward",
                        "combat_reward",
                        reward_index=reward_index,
                    )
                )
            if reward.get("can_skip") is True:
                label = (
                    _card_reward_skip_label(
                        reward_index,
                        already_handled=reward.get("can_claim") is False or is_blocked_card_reward,
                    )
                    if reward_kind == "card_reward"
                    else f"skip reward {reward_index}: {reward_kind}"
                )
                raw.append(
                    _action(
                        label,
                        "skip_reward",
                        "combat_reward_skip",
                        reward_index=reward_index,
                    )
                )
        player = state.get("player") or {}
        for potion in player.get("potions", []) or []:
            if not potion:
                continue
            potion_index = potion.get("index", potion.get("slot", potion.get("slot_index")))
            if potion_index is None:
                continue
            raw.append(
                _action(
                    f"discard potion {potion_index}: {_name(potion.get('name'))}",
                    "discard_potion",
                    "combat_reward_discard_potion",
                    potion_index=potion_index,
                )
            )
    elif decision == "card_reward":
        for card in state.get("cards", []):
            raw.append(
                _action(
                    f"pick card {card.get('index')}: {_name(card.get('name'))}",
                    "select_card_reward",
                    "card_reward",
                    card_index=card.get("index"),
                )
            )
        if state.get("can_skip") is True:
            raw.append(_action("skip card reward", "skip_card_reward", "card_reward_skip"))

    elif decision == "treasure":
        relics = state.get("relics") or []
        for relic in relics:
            raw.append(
                _action(
                    f"claim relic {relic.get('index')}: {_name(relic.get('name'))}",
                    "claim_relic",
                    "treasure",
                    relic_index=relic.get("index"),
                )
            )
        if not relics and state.get("can_proceed") is True:
            raw.append(_action("proceed", "proceed", "proceed"))

    elif decision == "rest_site":
        for option in state.get("options", []):
            if option.get("is_enabled") is not True:
                continue
            title = _name(option.get("title") or option.get("name") or option.get("description"))
            raw.append(
                _action(
                    f"choose option {option.get('index')}: {title}",
                    "choose_option",
                    "rest_site",
                    option_index=option.get("index"),
                )
            )

    elif decision == "event_choice":
        for option in state.get("options", []):
            if option.get("is_locked") is not False:
                continue
            title = _name(option.get("title") or option.get("name") or option.get("description"))
            raw.append(
                _action(
                    f"choose option {option.get('index')}: {title}",
                    "choose_option",
                    "event",
                    option_index=option.get("index"),
                )
            )
        if state.get("can_leave") is True:
            raw.append(_action("leave event", "leave_room", "event_leave"))

    elif decision == "shop":
        for card in state.get("cards", []) or []:
            if card.get("is_stocked") is True and card.get("can_buy") is True:
                raw.append(
                    _action(
                        f"buy card {card.get('index')}: {_name(card.get('name'))} for {_price(card)} gold",
                        "buy_card",
                        "shop_buy_card",
                        card_index=card.get("index"),
                    )
                )
        for relic in state.get("relics", []) or []:
            if relic.get("is_stocked") is True and relic.get("can_buy") is True:
                raw.append(
                    _action(
                        f"buy relic {relic.get('index')}: {_name(relic.get('name'))} for {_price(relic)} gold",
                        "buy_relic",
                        "shop_buy_relic",
                        relic_index=relic.get("index"),
                    )
                )
        for potion in state.get("potions", []) or []:
            if potion.get("is_stocked") is True and potion.get("can_buy") is True:
                raw.append(
                    _action(
                        f"buy potion {potion.get('index')}: {_name(potion.get('name'))} for {_price(potion)} gold",
                        "buy_potion",
                        "shop_buy_potion",
                        potion_index=potion.get("index"),
                    )
                )
        removal_cost = state.get("card_removal_cost")
        if isinstance(removal_cost, (int, float)) and state.get("can_remove_card") is True:
            raw.append(_action(f"remove a card for {int(removal_cost)} gold", "remove_card", "shop_remove_card"))
        raw.append(_action("leave shop", "leave_room", "shop_leave"))

    elif decision == "fake_merchant_shop":
        for relic in state.get("relics", []) or []:
            if not (relic.get("is_stocked") is True and relic.get("can_buy") is True):
                continue
            raw.append(
                _action(
                    f"buy fake merchant relic {relic.get('index')}: {_name(relic.get('name'))} for {_price(relic)} gold",
                    "buy_relic",
                    "fake_merchant_buy_relic",
                    relic_index=relic.get("index"),
                )
            )

        foul_potion_action = state.get("foul_potion_action")
        if state.get("can_throw_foul_potion") is True and isinstance(foul_potion_action, dict):
            action_name = foul_potion_action.get("action")
            potion_index = foul_potion_action.get("potion_index")
            if action_name and potion_index is not None:
                raw.append(
                    _action(
                        "throw foul potion at fake merchant",
                        str(action_name),
                        "fake_merchant_throw_foul_potion",
                        potion_index=potion_index,
                    )
                )

        if state.get("can_leave") is True:
            raw.append(_action("leave fake merchant", "leave_room", "fake_merchant_leave"))

    elif decision == "card_select":
        cards = state.get("cards") or []
        min_select = int(state.get("min_select", 1) or 0)
        max_select = int(state.get("max_select", min_select) or min_select)
        if state.get("can_skip") is True:
            raw.append(_action("skip selection", "skip_select", "card_select_skip"))
        raw.extend(_card_select_actions(cards, min_select, max_select))

    elif decision == "bundle_select":
        for bundle in state.get("bundles", []) or state.get("options", []) or []:
            raw.append(
                _action(
                    _bundle_label(bundle),
                    "select_bundle",
                    "bundle_select",
                    bundle_index=bundle.get("index"),
                )
            )

    elif decision == "crystal_sphere":
        if state.get("can_proceed") is True:
            raw.append(_action("proceed from crystal sphere", "crystal_sphere_proceed", "crystal_sphere"))
        if state.get("can_use_big_tool") is True:
            raw.append(_action("set crystal sphere tool: big", "crystal_sphere_set_tool", "crystal_sphere_tool", tool="big"))
        if state.get("can_use_small_tool") is True:
            raw.append(_action("set crystal sphere tool: small", "crystal_sphere_set_tool", "crystal_sphere_tool", tool="small"))
        for cell in state.get("clickable_cells", []) or []:
            raw.append(
                _action(
                    f"click crystal sphere cell {cell.get('x')},{cell.get('y')}",
                    "crystal_sphere_click_cell",
                    "crystal_sphere_cell",
                    x=cell.get("x"),
                    y=cell.get("y"),
                )
            )

    elif decision == "event_result":
        if state.get("can_proceed") is True:
            raw.append(_action("proceed from event result", "proceed", "event_result_proceed"))

    elif decision == "game_over":
        raw = []

    else:
        room_type = (state.get("context") or {}).get("room_type") or state.get("room_type")
        raise UnsupportedDecisionError(
            f"unsupported decision: {decision!r} (room_type={room_type!r}); refusing implicit proceed"
        )

    return [
        LegalAction(action_id=i, label=item["label"], command=item["command"], kind=item["kind"])
        for i, item in enumerate(raw)
    ]
