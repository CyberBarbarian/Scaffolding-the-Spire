"""LLM-facing state compaction and prompt rendering."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .route_graph import build_route_graph
from .route_plan import winged_boots_remaining_for_state


DROP_KEYS = {
    "enchantment_description",
    "affliction_description",
    "id",
    "option_id",
    "instance_id",
    "move_id",
    "draw_pile",
    "discard_pile",
    "exhaust_pile",
    "inactive_enemies",
    "full_map",
}

MAP_ICONS = {
    "Monster": "M",
    "Elite": "E",
    "Boss": "B",
    "RestSite": "R",
    "Shop": "$",
    "Treasure": "T",
    "Event": "?",
    "Unknown": "?",
    "Ancient": "A",
}

PILE_VIEW_SPECS = {
    "draw": ("view_draw_pile", "Draw pile"),
    "discard": ("view_discard_pile", "Discard pile"),
    "exhaust": ("view_exhaust_pile", "Exhaust pile"),
}


def name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("en", "name", "title", "id"):
            if value.get(key):
                return str(value[key])
        return "?"
    return str(value) if value is not None else "?"


def _compact_obj(obj: Any, *, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return "..."
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in DROP_KEYS:
                continue
            if key == "deck" and isinstance(value, list):
                out[key] = compact_deck(value)
                continue
            out[key] = _compact_obj(value, depth=depth + 1, max_depth=max_depth)
        return out
    if isinstance(obj, list):
        return [_compact_obj(item, depth=depth + 1, max_depth=max_depth) for item in obj]
    return obj


def compact_deck(deck: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, int], int] = {}
    examples: dict[tuple[str, int], dict[str, Any]] = {}
    for card in deck:
        card_name = name(card.get("name"))
        upgraded = int(card.get("upgrade_level", card.get("upgraded", 0)) or 0)
        key = (card_name, upgraded)
        counts[key] = counts.get(key, 0) + 1
        examples.setdefault(
            key,
            {
                "name": card_name,
                "upgraded": upgraded,
                "type": card.get("type"),
                "cost": card_cost_label(card),
            },
        )
    result = []
    for key, count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1])):
        item = dict(examples[key])
        item["count"] = count
        result.append(item)
    return result


def pile_cards_from_state(state: dict[str, Any], pile_name: str) -> list[dict[str, Any]]:
    direct = state.get(f"{pile_name}_pile")
    if isinstance(direct, list):
        return direct
    combat = state.get("combat")
    if isinstance(combat, dict):
        nested = combat.get(f"{pile_name}_pile")
        if isinstance(nested, list):
            return nested
    return []


def pile_count_from_state(state: dict[str, Any], pile_name: str, cards: list[dict[str, Any]] | None = None) -> int:
    count = state.get(f"{pile_name}_pile_count")
    if isinstance(count, (int, float)):
        return int(count)
    combat = state.get("combat")
    if isinstance(combat, dict):
        nested_count = combat.get(f"{pile_name}_pile_count")
        if isinstance(nested_count, (int, float)):
            return int(nested_count)
    return len(cards if cards is not None else pile_cards_from_state(state, pile_name))


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly state for logs and LLM prompts."""

    data = _compact_obj(state)
    viewed = []
    if state.get("decision") == "combat_play":
        data["combat_facts"] = combat_facts(state)
    full_map = state.get("full_map")
    if isinstance(full_map, dict) and full_map.get("type") == "map":
        data["route_graph"] = build_route_graph(
            full_map,
            state.get("choices", []) or [],
            winged_boots_remaining=winged_boots_remaining_for_state(state),
        )

    if state.get("view_deck"):
        player = state.get("player") or {}
        viewed.append("deck")
        data["viewed_deck"] = {
            "deck_size": player.get("deck_size"),
            "deck": compact_deck(player.get("deck") or []),
        }

    if state.get("view_map"):
        viewed.append("map")
        full_map = state.get("full_map")
        if isinstance(full_map, dict) and full_map.get("type") == "map":
            data["viewed_map"] = render_full_map(full_map, state.get("choices", []) or [])
        else:
            data["viewed_map_error"] = _compact_obj(state.get("view_map_error") or "Map view unavailable.")

    for pile_name, (flag, _title) in PILE_VIEW_SPECS.items():
        if not state.get(flag):
            continue
        cards = pile_cards_from_state(state, pile_name)
        viewed.append(f"{pile_name}_pile")
        data[f"viewed_{pile_name}_pile"] = {
            "count": pile_count_from_state(state, pile_name, cards),
            "cards": _compact_obj(cards),
        }

    if viewed:
        data["viewed"] = viewed

    return data


def incoming_damage(state: dict[str, Any]) -> int:
    total = 0
    for enemy in state.get("enemies", []) or []:
        for intent in enemy.get("intents") or []:
            if intent.get("type") not in {"Attack", "DeathBlow"}:
                continue
            damage = intent.get("damage") or 0
            hits = intent.get("hits") or 1
            if isinstance(damage, (int, float)) and isinstance(hits, (int, float)):
                total += int(damage) * int(hits)
    return total


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def combat_facts(state: dict[str, Any]) -> dict[str, Any]:
    """Return derived combat facts that are safer as state than prompt policy."""

    player = state.get("player") or {}
    hp = _as_int(player.get("hp"))
    max_hp = _as_int(player.get("max_hp"))
    current_block = _as_int(player.get("block")) or 0
    enemies = state.get("enemies", []) or []
    incoming_parts: list[str] = []
    non_attack_intents: list[str] = []
    incoming_total = 0
    for enemy in enemies:
        enemy_name = name(enemy.get("name"))
        enemy_index = enemy.get("index", "?")
        for intent in enemy.get("intents") or []:
            intent_type = str(intent.get("type", "?"))
            if intent_type in {"Attack", "DeathBlow"}:
                damage = _as_int(intent.get("damage"))
                hits = _as_int(intent.get("hits")) or 1
                if damage is None:
                    incoming_parts.append(f"{enemy_index}:{enemy_name}:{intent_type}=unknown")
                    continue
                incoming_total += damage * hits
                suffix = f"{damage}x{hits}" if hits > 1 else str(damage)
                incoming_parts.append(f"{enemy_index}:{enemy_name}:{intent_type} {suffix}")
            else:
                non_attack_intents.append(f"{enemy_index}:{enemy_name}:{intent_type}")

    hp_if_end = None
    if hp is not None:
        hp_if_end = hp - max(0, incoming_total - current_block)

    preview_fields = sorted(
        {
            "damage" if key == "calculateddamage" else key
            for card in state.get("hand", []) or []
            for key, value in (card.get("stats") or {}).items()
            if value is not None and key in {"damage", "calculateddamage", "block", "magic", "draw"}
        }
    )
    context = state.get("context") or {}
    room_type = context.get("room_type", state.get("room_type"))
    future_risk_reasons: list[str] = []
    if room_type in {"Boss", "Elite"}:
        future_risk_reasons.append("boss_or_elite")
    if hp is not None and max_hp is not None and hp <= max(12, int(max_hp * 0.25)):
        future_risk_reasons.append("low_hp")
    if enemies and incoming_total == 0:
        future_risk_reasons.append("no_current_attack")
    if non_attack_intents:
        future_risk_reasons.append("non_attack_intent")

    return {
        "incoming_attack_total": incoming_total,
        "incoming_attack_parts": incoming_parts,
        "current_block": current_block,
        "hp_if_end_turn_now": hp_if_end,
        "enemy_intent_damage_basis": "engine_displayed_current_values",
        "card_preview_values_basis": "engine_displayed_current_values",
        "card_preview_fields": preview_fields,
        "non_attack_intents": non_attack_intents,
        "future_risk_check_reasons": future_risk_reasons,
    }


def card_cost_label(card: dict[str, Any]) -> Any:
    """Return the CLI-facing card cost, matching python/play.py fallback order."""

    return card.get("cost", card.get("energy_cost", "?"))


def card_line(card: dict[str, Any], state: dict[str, Any] | None = None) -> str:
    stats = card.get("stats") or {}
    bits = [
        f"[{card.get('index', '?')}]",
        name(card.get("name")),
        f"cost={card_cost_label(card)}",
        str(card.get("type", "?")),
    ]
    if card.get("rarity"):
        bits.append(f"rarity={card.get('rarity')}")
    if card.get("upgraded") is not None:
        bits.append(f"upgraded={card.get('upgraded')}")
    if card.get("can_play") is not None:
        bits.append(f"can_play={card.get('can_play')}")
    for key in ("damage", "calculateddamage", "block", "magic", "draw"):
        value = stats.get(key)
        if value is not None:
            label = "damage" if key == "calculateddamage" else key
            bits.append(f"{label}={value}")
    for key in ("cards", "vulnerablepower", "weakpower", "strengthpower", "dexteritypower"):
        value = stats.get(key)
        if value is not None:
            bits.append(f"{key}={value}")
    if card.get("target_type"):
        bits.append(f"target={card.get('target_type')}")
    if card.get("star_cost"):
        bits.append(f"star_cost={card.get('star_cost')}")
    if card.get("enchantment"):
        bits.append(f"enchantment={name(card.get('enchantment'))}")
    if card.get("affliction"):
        bits.append(f"affliction={name(card.get('affliction'))}")
    keywords = card.get("keywords") or []
    if keywords:
        bits.append("keywords=" + ",".join(str(k) for k in keywords))
    return " ".join(bits)


def one_line(value: Any) -> str:
    if not value:
        return ""
    text = name(value) if isinstance(value, dict) else str(value)
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def description_of(obj: dict[str, Any]) -> str:
    """Return a one-line description if the simulator exported one."""

    description = obj.get("description")
    if not description:
        return ""
    return one_line(description)


def card_description(card: dict[str, Any]) -> str:
    return description_of(card)


def vars_line(obj: dict[str, Any]) -> str:
    vars_obj = obj.get("vars")
    if not vars_obj:
        return ""
    try:
        return json.dumps(vars_obj, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(vars_obj)


def append_combat_fact_lines(lines: list[str], state: dict[str, Any], *, prefix: str = "  ") -> None:
    facts = combat_facts(state)
    lines.append("Combat facts:")
    parts = [
        f"incoming_attack_total={facts['incoming_attack_total']}",
        f"current_block={facts['current_block']}",
        f"hp_if_end_turn_now={facts['hp_if_end_turn_now']}",
    ]
    lines.append(prefix + " ".join(parts))
    if facts["incoming_attack_parts"]:
        lines.append(prefix + "incoming_attack_parts=" + "; ".join(facts["incoming_attack_parts"]))
    if facts["non_attack_intents"]:
        lines.append(prefix + "non_attack_intents=" + "; ".join(facts["non_attack_intents"]))
    preview_fields = facts["card_preview_fields"]
    fields_text = ",".join(preview_fields) if preview_fields else "none"
    lines.append(
        prefix
        + "basis: enemy intent damage and card preview fields are engine_displayed_current_values"
        + f" card_preview_fields={fields_text}"
    )
    if facts["future_risk_check_reasons"]:
        lines.append(prefix + "future_risk_check_reasons=" + ",".join(facts["future_risk_check_reasons"]))


def combat_skill_hints(state: dict[str, Any], *, max_hints: int = 4) -> list[str]:
    """Return compact combat heuristics triggered by the visible state."""

    if state.get("decision", state.get("type")) != "combat_play":
        return []

    facts = combat_facts(state)
    hints = [
        "order: lethal > survival > reduce actions > damage/setup; one action",
    ]
    current_block = _as_int((state.get("player") or {}).get("block")) or 0
    incoming = _as_int(facts.get("incoming_attack_total")) or 0
    if incoming > current_block:
        hints.append("survive: block/kill until safe; stop overblocking")
    elif incoming == 0 and facts.get("non_attack_intents"):
        hints.append("no_attack: skip pure block; use damage/draw/scaling")

    enemies = [enemy for enemy in state.get("enemies", []) or [] if (_as_int(enemy.get("hp")) or 0) > 0]
    hand = state.get("hand", []) or []
    multi_hint = _multi_enemy_skill_hint(enemies, hand, energy=_as_int(state.get("energy")) or 0)
    if multi_hint:
        hints.append(multi_hint)

    if _hand_has_self_damage(hand):
        hints.append("self_damage: only for lethal, survival, or a concrete same-turn play")
    if _hand_has_draw(hand):
        hints.append("draw: real draw can be early if safe; view actions only reveal info")

    return hints[:max_hints]


def append_combat_skill_lines(lines: list[str], state: dict[str, Any], *, prefix: str = "  ") -> None:
    hints = combat_skill_hints(state)
    if not hints:
        return
    lines.append("Combat skills:")
    for hint in hints:
        lines.append(prefix + hint)


def _hand_has_self_damage(hand: list[dict[str, Any]]) -> bool:
    for card in hand:
        text = " ".join(
            str(part).lower()
            for part in (
                name(card.get("name")),
                card_description(card),
                card.get("raw_description"),
            )
            if part
        )
        if "lose" in text and ("hp" in text or "health" in text):
            return True
        if "blood" in text and "energy" in text:
            return True
    return False


def _hand_has_draw(hand: list[dict[str, Any]]) -> bool:
    for card in hand:
        stats = card.get("stats") or {}
        draw = _as_int(stats.get("draw") or stats.get("cards"))
        if draw and draw > 0:
            return True
        text = " ".join(
            str(part).lower()
            for part in (
                card_description(card),
                card.get("raw_description"),
            )
            if part
        )
        if "draw" in text:
            return True
    return False


def _multi_enemy_skill_hint(enemies: list[dict[str, Any]], hand: list[dict[str, Any]], *, energy: int) -> str:
    if len(enemies) <= 1:
        return ""

    best_single_damage = _best_single_target_attack_damage(hand, energy=energy)
    if best_single_damage > 0:
        killable = [
            enemy
            for enemy in enemies
            if _enemy_effective_hp(enemy) <= best_single_damage
        ]
        if killable:
            target = min(killable, key=lambda enemy: (_enemy_effective_hp(enemy), enemy.get("index", 999)))
            return f"kill_now: enemy[{target.get('index','?')}] hp={target.get('hp','?')} dies to one card; avoid spread/setup"

    if len(enemies) >= 3 and _hand_has_aoe_attack(hand, energy=energy):
        return "aoe: hit-all is strong vs 3+ unless one enemy is killable"

    target = min(enemies, key=lambda enemy: (_enemy_effective_hp(enemy), enemy.get("index", 999)))
    return f"focus: enemy[{target.get('index','?')}] lowest hp={target.get('hp','?')}; avoid spreading"


def _enemy_effective_hp(enemy: dict[str, Any]) -> int:
    return (_as_int(enemy.get("hp")) or 0) + (_as_int(enemy.get("block")) or 0)


def _best_single_target_attack_damage(hand: list[dict[str, Any]], *, energy: int) -> int:
    return max(
        (_card_damage(card) for card in hand if _is_playable_attack(card, energy=energy) and not _is_aoe_card(card)),
        default=0,
    )


def _hand_has_aoe_attack(hand: list[dict[str, Any]], *, energy: int) -> bool:
    return any(_is_playable_attack(card, energy=energy) and _is_aoe_card(card) for card in hand)


def _is_playable_attack(card: dict[str, Any], *, energy: int) -> bool:
    if card.get("can_play") is False:
        return False
    if str(card.get("type") or "").lower() != "attack":
        return False
    cost = _as_int(card_cost_label(card))
    if cost is not None and cost > energy:
        return False
    return _card_damage(card) > 0


def _card_damage(card: dict[str, Any]) -> int:
    stats = card.get("stats") or {}
    return _as_int(stats.get("damage")) or _as_int(stats.get("calculateddamage")) or 0


def _is_aoe_card(card: dict[str, Any]) -> bool:
    target_type = str(card.get("target_type") or card.get("target") or "").lower()
    if "all" in target_type:
        return True
    text = " ".join(
        str(part).lower()
        for part in (
            name(card.get("name")),
            card_description(card),
            card.get("raw_description"),
        )
        if part
    )
    return "all enemies" in text or "all enemy" in text


def append_hover_tip_lines(lines: list[str], obj: dict[str, Any], *, prefix: str = "  ") -> None:
    for tip in obj.get("hover_tips") or []:
        title = name(tip.get("name") or tip.get("title") or tip.get("id") or tip.get("kind"))
        desc = description_of(tip)
        if desc:
            lines.append(prefix + f"tip: {title}: {desc}")
        elif title and title != "?":
            lines.append(prefix + f"tip: {title}")


def _draw_conn(buf: list[str], from_col: int, to_col: int, width: int) -> None:
    """Draw a simple TUI-style connection between two map columns."""

    from_pos = from_col * width + width // 2
    to_pos = to_col * width + width // 2
    if from_pos == to_pos:
        if 0 <= from_pos < len(buf):
            buf[from_pos] = "|"
        return
    lo, hi = sorted((from_pos, to_pos))
    char = "/" if to_pos < from_pos else "\\"
    for pos in range(lo, hi + 1):
        if 0 <= pos < len(buf):
            buf[pos] = char


def map_display_choices(choices: list[dict[str, Any]], map_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Match ``python/play.py`` filtering of choices to visible map coords."""

    visible_coords: set[tuple[Any, Any]] = set()
    for row in (map_data or {}).get("rows", []):
        for node in row:
            visible_coords.add((node.get("col"), node.get("row")))

    boss = (map_data or {}).get("boss") or {}
    if "col" in boss and "row" in boss:
        visible_coords.add((boss.get("col"), boss.get("row")))

    if not visible_coords:
        return choices

    filtered = [
        choice
        for choice in choices
        if (choice.get("col"), choice.get("row")) in visible_coords
        or choice.get("type") == "Ancient"
    ]
    return filtered or choices


def render_full_map(map_data: dict[str, Any], choices: list[dict[str, Any]]) -> list[str]:
    """Render map as a text grid close to ``python/play.py``'s TUI map."""

    rows = map_data.get("rows") or []
    if not rows:
        return []

    visible_choices = map_display_choices(choices, map_data)
    choice_indices = {
        (choice.get("col"), choice.get("row")): idx
        for idx, choice in enumerate(visible_choices)
    }
    choice_nodes = {
        (choice.get("col"), choice.get("row")): choice
        for choice in visible_choices
    }

    node_map: dict[tuple[int, int], dict[str, Any]] = {}
    max_col = 0
    row_numbers: set[int] = set()
    for row in rows:
        for node in row:
            col = int(node.get("col", 0))
            row_number = int(node.get("row", 0))
            node_map[(col, row_number)] = node
            max_col = max(max_col, col)
            row_numbers.add(row_number)

    boss = map_data.get("boss") or {}
    if "col" in boss:
        max_col = max(max_col, int(boss.get("col", 0)))
    for col, _row in choice_indices:
        if isinstance(col, int):
            max_col = max(max_col, col)

    row_numbers_sorted = sorted(row_numbers)
    total_cols = max_col + 1
    cell_width = 4
    grid_width = cell_width * total_cols
    current = map_data.get("current_coord")
    out = ["Full map:"]

    if current and current.get("row", -1) not in row_numbers:
        out.append("  You are at the start")

    off_grid_choices = []
    for coord, choice_idx in choice_indices.items():
        col, row = coord
        boss_coord = (boss.get("col"), boss.get("row"))
        if coord not in node_map and coord != boss_coord:
            node = choice_nodes.get(coord, {"col": col, "row": row, "type": "?"})
            off_grid_choices.append((choice_idx, node))
    for choice_idx, node in sorted(off_grid_choices):
        out.append(f"  [{choice_idx}] {node.get('type', '?')}")

    boss_col = int(boss.get("col", 0))
    boss_row = boss.get("row", -1)
    boss_choice_idx = choice_indices.get((boss_col, boss_row))
    boss_buf = list(" " * grid_width)
    boss_center = boss_col * cell_width + cell_width // 2
    if 0 <= boss_center < len(boss_buf):
        boss_buf[boss_center] = "B"
    out.append(f"  B | {''.join(boss_buf)}")
    if boss_choice_idx is not None:
        ann = list(" " * grid_width)
        label = f"[{boss_choice_idx}]"
        start = boss_center - 1
        for offset, ch in enumerate(label):
            if 0 <= start + offset < len(ann):
                ann[start + offset] = ch
        out.append(f"    | {''.join(ann)}")

    if row_numbers_sorted:
        top_row = row_numbers_sorted[-1]
        conn = list(" " * grid_width)
        for row in rows:
            for node in row:
                if node.get("row") != top_row:
                    continue
                for child in node.get("children") or []:
                    if child.get("row") == boss_row:
                        _draw_conn(conn, int(node.get("col", 0)), int(child.get("col", 0)), cell_width)
        out.append(f"    | {''.join(conn)}")

    for idx in range(len(row_numbers_sorted) - 1, -1, -1):
        row_number = row_numbers_sorted[idx]
        node_buf = list(" " * grid_width)
        for col in range(total_cols):
            node = node_map.get((col, row_number))
            if not node:
                continue
            icon = MAP_ICONS.get(node.get("type", "?"), ".")
            center = col * cell_width + cell_width // 2
            is_current = bool(current and current.get("col") == col and current.get("row") == row_number)
            if is_current and center - 1 >= 0 and center + 1 < len(node_buf):
                node_buf[center - 1] = "["
                node_buf[center] = icon
                node_buf[center + 1] = "]"
            else:
                node_buf[center] = icon
        out.append(f"  {row_number:>2}| {''.join(node_buf)}")

        row_choices = {
            col: choice_indices[(col, row_number)]
            for col in range(total_cols)
            if (col, row_number) in choice_indices
        }
        if row_choices:
            ann = list(" " * grid_width)
            for col, choice_idx in row_choices.items():
                label = f"[{choice_idx}]"
                start = col * cell_width + cell_width // 2 - 1
                for offset, ch in enumerate(label):
                    if 0 <= start + offset < len(ann):
                        ann[start + offset] = ch
            out.append(f"    | {''.join(ann)}")

        if idx > 0:
            below_row = row_numbers_sorted[idx - 1]
            conn = list(" " * grid_width)
            for row in rows:
                for node in row:
                    if node.get("row") != below_row:
                        continue
                    for child in node.get("children") or []:
                        if child.get("row") == row_number:
                            _draw_conn(conn, int(node.get("col", 0)), int(child.get("col", 0)), cell_width)
            out.append(f"    | {''.join(conn)}")

    out.append("  Legend: M=Monster E=Elite R=Rest $=Shop T=Treasure ?=Event A=Ancient [x]=Current [n]=Choice")
    if choice_indices:
        parts = []
        inverse = {value: key for key, value in choice_indices.items()}
        for choice_idx in sorted(inverse):
            col, row = inverse[choice_idx]
            node = node_map.get((col, row))
            if not node and col == boss_col and row == boss_row:
                node = boss
            if not node:
                node = choice_nodes.get((col, row))
            if node:
                parts.append(f"{choice_idx}={node.get('type', '?')}")
        if parts:
            out.append("  Choices: " + " ".join(parts))
    return out


def append_card_lines(lines: list[str], card: dict[str, Any], *, prefix: str = "  ", suffix: str = "") -> None:
    lines.append(prefix + card_line(card) + suffix)
    description = card_description(card)
    if description:
        lines.append(prefix + f"  description: {description}")
    for key in ("enchantment_description", "affliction_description"):
        value = one_line(card.get(key))
        if value:
            lines.append(prefix + f"  {key}: {value}")
    after = card.get("after_upgrade")
    if isinstance(after, dict):
        upgrade_parts = []
        old_cost = card_cost_label(card)
        new_cost = card_cost_label(after)
        if new_cost != "?" and new_cost != old_cost:
            upgrade_parts.append(f"cost {old_cost} -> {new_cost}")
        stats = card.get("stats") or {}
        after_stats = after.get("stats") or {}
        for key in sorted(set(stats) | set(after_stats)):
            if key.endswith("_by_target"):
                continue
            old = stats.get(key)
            new = after_stats.get(key, old)
            if old != new and not isinstance(old, (dict, list)) and not isinstance(new, (dict, list)):
                upgrade_parts.append(f"{key} {old} -> {new}")
        after_desc = card_description(after)
        if upgrade_parts:
            lines.append(prefix + "  upgrade: " + "; ".join(upgrade_parts))
        if after_desc:
            lines.append(prefix + f"  upgrade_description: {after_desc}")
    append_hover_tip_lines(lines, card, prefix=prefix + "  ")


def append_deck_view(lines: list[str], player: dict[str, Any]) -> None:
    deck = player.get("deck") or []
    lines.append(f"Deck view ({len(deck)} entries, deck_size={player.get('deck_size', '?')}):")
    if not deck:
        lines.append("  No deck details available in this state.")
        return
    for idx, card in enumerate(deck):
        card_for_render = dict(card)
        if card_for_render.get("index") is None:
            card_for_render["index"] = idx
        count = card.get("count")
        suffix = f" x{count}" if count else ""
        append_card_lines(lines, card_for_render, suffix=suffix)


def append_pile_view(lines: list[str], state: dict[str, Any], pile_name: str) -> None:
    _flag, title = PILE_VIEW_SPECS[pile_name]
    cards = pile_cards_from_state(state, pile_name)
    count = pile_count_from_state(state, pile_name, cards)
    lines.append(f"{title} ({count}):")
    if not cards:
        if count == 0:
            lines.append("  Empty")
        else:
            lines.append("  Card details are only available during combat.")
        return
    for idx, card in enumerate(cards):
        card_for_render = dict(card)
        if card_for_render.get("index") is None:
            card_for_render["index"] = idx
        append_card_lines(lines, card_for_render)


def append_bundle_lines(lines: list[str], bundles: list[dict[str, Any]]) -> None:
    lines.append(f"Card packs ({len(bundles)}):")
    if not bundles:
        lines.append("  No card packs available.")
        return

    for bundle_idx, bundle in enumerate(bundles):
        index = bundle.get("index", bundle_idx)
        lines.append(f"  Pack [{index}]:")
        cards = bundle.get("cards") or []
        if not cards:
            lines.append("    No cards in this pack.")
            continue
        for card_idx, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            card_for_render = dict(card)
            if card_for_render.get("index") is None:
                card_for_render["index"] = card_idx
            append_card_lines(lines, card_for_render, prefix="    ")


def power_kind(power: dict[str, Any]) -> str:
    power_type = str(power.get("type") or power.get("power_type") or "").lower()
    amount = power.get("amount", 0)
    if isinstance(amount, (int, float)) and amount < 0:
        return "Debuff"
    if "debuff" in power_type:
        return "Debuff"
    if "buff" in power_type:
        return "Buff"
    return "Power"


def power_label(power: dict[str, Any], *, include_type: bool = False) -> str:
    amount = power.get("amount")
    suffix = f"({amount})" if amount not in (None, "", 0) else ""
    label = f"{name(power.get('name'))}{suffix}"
    if include_type:
        label = f"{power_kind(power)} {label}"
    return label


def power_description_text(power: dict[str, Any]) -> str:
    amount = power.get("amount")
    power_name = name(power.get("name")).strip().lower()
    if isinstance(amount, (int, float)) and amount < 0:
        value = abs(int(amount))
        if power_name == "dexterity":
            return f"Reduces Block gained from cards by {value}."
        if power_name == "strength":
            return f"Decreases attack damage by {value}."
    return description_of(power)


def append_power_lines(lines: list[str], powers: list[dict[str, Any]], *, prefix: str = "  ", include_type: bool = False) -> None:
    for power in powers:
        label = power_label(power, include_type=include_type)
        description = power_description_text(power)
        if description:
            lines.append(prefix + f"{label}: {description}")
        else:
            vars_text = vars_line(power)
            if vars_text:
                lines.append(prefix + f"{label} vars={vars_text}")
            else:
                lines.append(prefix + label)
        append_hover_tip_lines(lines, power, prefix=prefix + "  ")


def append_viewed_information(lines: list[str], state: dict[str, Any], player: dict[str, Any]) -> None:
    has_pile_view = any(state.get(flag) for flag, _title in PILE_VIEW_SPECS.values())
    if not (state.get("view_deck") or state.get("view_map") or has_pile_view):
        return

    lines.append("")
    lines.append("Viewed information:")
    if state.get("view_deck"):
        lines.append("Viewed deck:")
        append_deck_view(lines, player)

    if state.get("view_map"):
        lines.append("Viewed map:")
        full_map = state.get("full_map")
        if isinstance(full_map, dict) and full_map.get("type") == "map":
            lines.extend(render_full_map(full_map, state.get("choices", []) or []))
        elif state.get("view_map_error"):
            lines.append(f"  Map view unavailable: {state.get('view_map_error')}")
        else:
            lines.append("  Map view unavailable.")

    for pile_name, (flag, _title) in PILE_VIEW_SPECS.items():
        if not state.get(flag):
            continue
        lines.append(f"Viewed {pile_name} pile:")
        append_pile_view(lines, state, pile_name)


def _value_text(value: Any) -> str:
    return "?" if value is None else str(value)


def _add_change(changes: list[str], label: str, old: Any, new: Any) -> None:
    if old != new:
        changes.append(f"{label}: {_value_text(old)} -> {_value_text(new)}")


def _hp_pair(entity: dict[str, Any]) -> str:
    return f"{entity.get('hp', '?')}/{entity.get('max_hp', '?')}"


def _named_counter(items: list[dict[str, Any]], *, upgrade_sensitive: bool = False) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items:
        if not item:
            continue
        if not isinstance(item, dict):
            counter[name(item)] += 1
            continue
        label = name(item.get("name"))
        if upgrade_sensitive:
            upgraded = item.get("upgrade_level", item.get("upgraded", 0)) or 0
            if isinstance(upgraded, bool):
                upgraded = 1 if upgraded else 0
            if upgraded:
                label += "+" if upgraded == 1 else f"+{upgraded}"
        counter[label] += 1
    return counter


def _counter_delta_text(old_items: list[dict[str, Any]], new_items: list[dict[str, Any]], *, upgrade_sensitive: bool = False) -> str:
    old_counter = _named_counter(old_items, upgrade_sensitive=upgrade_sensitive)
    new_counter = _named_counter(new_items, upgrade_sensitive=upgrade_sensitive)
    removed = old_counter - new_counter
    added = new_counter - old_counter
    parts = []
    for item, count in sorted(removed.items()):
        parts.append(f"-{item}" + (f"x{count}" if count > 1 else ""))
    for item, count in sorted(added.items()):
        parts.append(f"+{item}" + (f"x{count}" if count > 1 else ""))
    return " ".join(parts)


def _combat_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    combat = state.get("combat")
    if isinstance(combat, dict) and combat:
        return combat
    if isinstance(state.get("enemies"), list):
        return state
    return {}


def _enemy_key(enemy: dict[str, Any]) -> tuple[str, Any]:
    return (name(enemy.get("name")), enemy.get("index"))


def build_last_action_result(
    old_state: dict[str, Any],
    new_state: dict[str, Any],
    *,
    action_label: str,
    action_kind: str,
) -> dict[str, Any]:
    """Build a compact transition summary for the next model-facing state."""

    changes: list[str] = []
    old_context = old_state.get("context") or {}
    new_context = new_state.get("context") or {}
    old_player = old_state.get("player") or {}
    new_player = new_state.get("player") or {}

    _add_change(changes, "act", old_state.get("act") or old_context.get("act"), new_state.get("act") or new_context.get("act"))
    _add_change(changes, "floor", old_state.get("floor") or old_context.get("floor"), new_state.get("floor") or new_context.get("floor"))
    _add_change(changes, "room", old_context.get("room_type"), new_context.get("room_type"))

    if new_player:
        if _hp_pair(old_player) != _hp_pair(new_player):
            changes.append(f"hp: {_hp_pair(old_player)} -> {_hp_pair(new_player)}")
        _add_change(changes, "block", old_player.get("block", 0), new_player.get("block", 0))
        old_gold = old_player.get("gold")
        new_gold = new_player.get("gold")
        if old_gold != new_gold:
            suffix = ""
            if isinstance(old_gold, (int, float)) and isinstance(new_gold, (int, float)):
                suffix = f" ({new_gold - old_gold:+})"
            changes.append(f"gold: {_value_text(old_gold)} -> {_value_text(new_gold)}{suffix}")
        _add_change(changes, "deck_size", old_player.get("deck_size"), new_player.get("deck_size"))

        deck_delta = _counter_delta_text(old_player.get("deck", []) or [], new_player.get("deck", []) or [], upgrade_sensitive=True)
        if deck_delta:
            changes.append(f"deck: {deck_delta}")
        relic_delta = _counter_delta_text(old_player.get("relics", []) or [], new_player.get("relics", []) or [])
        if relic_delta:
            changes.append(f"relics: {relic_delta}")
        potion_delta = _counter_delta_text(old_player.get("potions", []) or [], new_player.get("potions", []) or [])
        if potion_delta:
            changes.append(f"potions: {potion_delta}")

    old_combat = _combat_snapshot(old_state)
    new_combat = _combat_snapshot(new_state)
    if old_combat and new_combat:
        _add_change(changes, "round", old_combat.get("round"), new_combat.get("round"))
        old_energy = f"{old_combat.get('energy', '?')}/{old_combat.get('max_energy', '?')}"
        new_energy = f"{new_combat.get('energy', '?')}/{new_combat.get('max_energy', '?')}"
        if old_energy != new_energy:
            changes.append(f"energy: {old_energy} -> {new_energy}")
        for pile_name in ("draw", "discard", "exhaust"):
            old_count = pile_count_from_state(old_combat, pile_name)
            new_count = pile_count_from_state(new_combat, pile_name)
            if old_count != new_count:
                changes.append(f"{pile_name}_pile: {old_count} -> {new_count}")

        old_hand = old_combat.get("hand")
        new_hand = new_combat.get("hand")
        if isinstance(old_hand, list) and isinstance(new_hand, list) and len(old_hand) != len(new_hand):
            changes.append(f"hand_count: {len(old_hand)} -> {len(new_hand)}")

    if old_combat and new_combat:
        new_enemies = {
            _enemy_key(enemy): enemy
            for enemy in new_combat.get("enemies", []) or []
            if isinstance(enemy, dict)
        }
        for old_enemy in old_combat.get("enemies", []) or []:
            if not isinstance(old_enemy, dict):
                continue
            enemy_label = f"{name(old_enemy.get('name'))}[{old_enemy.get('index', '?')}]"
            new_enemy = new_enemies.get(_enemy_key(old_enemy))
            if not new_enemy:
                changes.append(f"enemy {enemy_label}: defeated")
                continue
            enemy_parts = []
            if _hp_pair(old_enemy) != _hp_pair(new_enemy):
                enemy_parts.append(f"hp {_hp_pair(old_enemy)} -> {_hp_pair(new_enemy)}")
            if old_enemy.get("block", 0) != new_enemy.get("block", 0):
                enemy_parts.append(f"block {old_enemy.get('block', 0)} -> {new_enemy.get('block', 0)}")
            if enemy_parts:
                changes.append(f"enemy {enemy_label}: {'; '.join(enemy_parts)}")

    if not changes:
        changes.append("no visible state changes")

    return {
        "action_label": action_label,
        "action_kind": action_kind,
        "decision_before": old_state.get("decision", old_state.get("type")),
        "decision_after": new_state.get("decision", new_state.get("type")),
        "changes": changes,
    }


def append_last_action_result(lines: list[str], result: Any) -> None:
    if not isinstance(result, dict):
        return
    lines.append("")
    lines.append("Last game action result:")
    if result.get("action_label"):
        lines.append(f"  action: {result.get('action_label')}")
    if result.get("action_kind"):
        lines.append(f"  kind: {result.get('action_kind')}")
    before = result.get("decision_before")
    after = result.get("decision_after")
    if before is not None or after is not None:
        lines.append(f"  decision: {_value_text(before)} -> {_value_text(after)}")
    for change in result.get("changes") or []:
        lines.append(f"  {change}")


def append_active_route_plan_lines(lines: list[str], active_route_plan: Any) -> None:
    if not isinstance(active_route_plan, dict):
        return
    expected_next = active_route_plan.get("expected_next")
    remaining = active_route_plan.get("remaining")
    if not isinstance(remaining, list):
        remaining = []
    lines.append("Active route plan:")
    lines.append(f"  plan_id={active_route_plan.get('plan_id', '?')} expected_next={_route_node_text(expected_next)}")
    if remaining:
        lines.append("  remaining: " + " -> ".join(_route_node_text(node) for node in remaining))
    lines.append("  follow: select expected_next and omit route_plan_event")
    lines.append("  change: choose a different legal map action and emit route_plan_updated with reason")


def _route_node_text(node: Any) -> str:
    if not isinstance(node, dict):
        return "none"
    room_type = node.get("type", "?")
    return f"{room_type}({node.get('col', '?')},{node.get('row', '?')})"


def player_summary(player: dict[str, Any]) -> str:
    if not player:
        return "unknown player"
    parts = [
        f"hp={player.get('hp', '?')}/{player.get('max_hp', '?')}",
        f"block={player.get('block', 0)}",
        f"gold={player.get('gold', '?')}",
        f"deck_size={player.get('deck_size', '?')}",
    ]
    relics = [name(relic.get("name")) for relic in player.get("relics", []) or [] if relic]
    potions = [f"{pot.get('index', pot.get('slot_index', '?'))}:{name(pot.get('name'))}" for pot in player.get("potions", []) or [] if pot]
    if relics:
        parts.append("relics=" + ", ".join(relics[:12]) + (f" (+{len(relics) - 12})" if len(relics) > 12 else ""))
    if potions:
        parts.append("potions=" + ", ".join(potions))
    return " | ".join(parts)


def append_player_details(lines: list[str], player: dict[str, Any]) -> None:
    """Render player details like the TUI's ``show_player``.

    Keep this intentionally close to ``python/play.py``: show relics and
    potions, but do not expand the full deck in ordinary decision prompts.
    Full deck expansion is available from the raw/compact JSON, but putting it
    in the human-readable section makes it too easy to confuse deck cards with
    the current hand.
    """

    relics = [relic for relic in player.get("relics", []) or [] if relic]
    if relics:
        lines.append(f"Relics ({len(relics)}):")
        for relic in relics:
            bits = [name(relic.get("name"))]
            vars_text = vars_line(relic)
            if vars_text:
                bits.append(f"vars={vars_text}")
            if relic.get("show_counter") is not None:
                bits.append(f"show_counter={relic.get('show_counter')}")
            lines.append("  " + " ".join(bits))
            desc = description_of(relic)
            if desc:
                lines.append(f"    description: {desc}")

    potions = [potion for potion in player.get("potions", []) or [] if potion]
    if potions or player.get("potion_slots") is not None:
        slots = player.get("potion_slots", "?")
        empty = player.get("potion_empty_slots", "?")
        lines.append(f"Potions ({len(potions)}/{slots}, empty={empty}):")
        for potion in potions:
            idx = potion.get("index", potion.get("slot_index", "?"))
            bits = [f"[{idx}]", name(potion.get("name"))]
            if potion.get("target_type"):
                bits.append(f"target={potion.get('target_type')}")
            vars_text = vars_line(potion)
            if vars_text:
                bits.append(f"vars={vars_text}")
            lines.append("  " + " ".join(bits))
            desc = description_of(potion)
            if desc:
                lines.append(f"    description: {desc}")


def render_state_text(state: dict[str, Any]) -> str:
    """Render a concise, stable text view inspired by ``python/play.py``."""

    decision = state.get("decision", state.get("type", "?"))
    context = state.get("context") or {}
    player = state.get("player") or {}

    lines = [
        f"Decision: {decision}",
        f"Context: act={context.get('act', state.get('act', '?'))} floor={context.get('floor', state.get('floor', '?'))} room={context.get('room_type', '?')} boss={name((context.get('boss') or {}).get('name'))}",
        f"Player: {player_summary(player)}",
    ]
    append_player_details(lines, player)
    append_last_action_result(lines, state.get("last_action_result"))

    if decision == "combat_play":
        lines.append(
            f"Combat: round={state.get('round', '?')} energy={state.get('energy', '?')}/{state.get('max_energy', '?')} "
            f"draw_pile={state.get('draw_pile_count', '?')} discard_pile={state.get('discard_pile_count', '?')}"
        )
        append_combat_fact_lines(lines, state)
        append_combat_skill_lines(lines, state)
        powers = state.get("player_powers") or []
        if powers:
            lines.append(f"Player powers ({len(powers)}):")
            append_power_lines(lines, powers, include_type=True)
        if state.get("orbs"):
            lines.append("Orbs: " + "; ".join(f"{name(o.get('name'))} passive={o.get('passive')} evoke={o.get('evoke')}" for o in state.get("orbs", [])))
        if state.get("stars") is not None:
            lines.append(f"Stars: {state.get('stars')}")

        enemies = state.get("enemies", []) or []
        lines.append(f"Enemies ({len(enemies)}):")
        for enemy in enemies:
            intents = []
            for intent in enemy.get("intents") or []:
                if intent.get("type") == "Attack":
                    if intent.get("hits", 1) and intent.get("hits", 1) > 1:
                        intents.append(f"Attack {intent.get('damage')}x{intent.get('hits')}")
                    else:
                        intents.append(f"Attack {intent.get('damage')}")
                else:
                    intents.append(str(intent.get("type")))
            powers = enemy.get("powers") or []
            power_text = ""
            if powers:
                power_text = " powers=" + ",".join(power_label(power) for power in powers)
            move = enemy.get("move_name") or "?"
            lines.append(
                f"  [{enemy.get('index')}] {name(enemy.get('name'))} hp={enemy.get('hp')}/{enemy.get('max_hp')} block={enemy.get('block', 0)} intent={','.join(intents) or 'none'} move={move}{power_text}"
            )
            if powers:
                append_power_lines(lines, powers, prefix="    ", include_type=True)

        hand = state.get("hand", []) or []
        lines.append(f"Hand ({len(hand)}):")
        for card in hand:
            append_card_lines(lines, card)

    elif decision == "map_select":
        choices = state.get("choices", []) or []
        full_map = state.get("full_map")
        if not state.get("view_map") and isinstance(full_map, dict) and full_map.get("type") == "map":
            lines.extend(render_full_map(full_map, choices))
        append_active_route_plan_lines(lines, state.get("active_route_plan"))
        lines.append(f"Map choices ({len(choices)}):")
        for choice in choices:
            lines.append(f"  col={choice.get('col')} row={choice.get('row')} type={choice.get('type')}")

    elif decision == "card_reward":
        cards = state.get("cards", []) or []
        lines.append(f"Card rewards ({len(cards)}):")
        for card in cards:
            append_card_lines(lines, card)
        lines.append(f"Can skip: {state.get('can_skip', True)}")

    elif decision == "bundle_select":
        bundles = state.get("bundles", []) or state.get("options", []) or []
        append_bundle_lines(lines, bundles)

    elif decision == "combat_reward":
        rewards = state.get("rewards", []) or []
        lines.append(f"Combat rewards ({len(rewards)}):")
        for reward in rewards:
            bits = [
                f"[{reward.get('index')}]",
                f"kind={reward.get('kind')}",
                f"type_name={reward.get('type_name', '?')}",
                f"name={name(reward.get('name'))}",
            ]
            if reward.get("amount") is not None:
                bits.append(f"amount={reward.get('amount')}")
            if reward.get("can_claim") is not None:
                bits.append(f"can_claim={reward.get('can_claim')}")
            if reward.get("can_skip") is not None:
                bits.append(f"can_skip={reward.get('can_skip')}")
            lines.append("  " + " ".join(bits))
            desc = description_of(reward)
            if desc:
                lines.append(f"    description: {desc}")
            append_hover_tip_lines(lines, reward, prefix="    ")

    elif decision == "event_choice":
        lines.append(f"Event: {name(state.get('event_name'))}")
        if state.get("description"):
            lines.append(f"Description: {state.get('description')}")
        options = state.get("options", []) or []
        lines.append(f"Options ({len(options)}):")
        for option in options:
            locked = " locked" if option.get("is_locked") else ""
            enabled = "" if option.get("is_enabled", True) else " disabled"
            vars_text = vars_line(option) or "{}"
            lines.append(f"  [{option.get('index')}] {name(option.get('title') or option.get('name'))}{locked}{enabled} vars={vars_text}")
            desc = description_of(option)
            if desc:
                lines.append(f"    description: {desc}")
            append_hover_tip_lines(lines, option, prefix="    ")

    elif decision == "rest_site":
        options = state.get("options", []) or []
        lines.append(f"Rest options ({len(options)}):")
        for option in options:
            disabled = " disabled" if option.get("is_enabled") is False else ""
            vars_text = vars_line(option) or "{}"
            lines.append(f"  [{option.get('index')}] {name(option.get('title') or option.get('name'))}{disabled} vars={vars_text}")
            desc = description_of(option)
            if desc:
                lines.append(f"    description: {desc}")
            append_hover_tip_lines(lines, option, prefix="    ")

    elif decision == "shop":
        shop_cards = state.get("cards", []) or []
        lines.append(f"Shop cards ({len(shop_cards)}):")
        for card in shop_cards:
            stocked = "" if card.get("is_stocked") else " sold"
            lines.append(f"  [{card.get('index')}] {name(card.get('name'))} price={card.get('price', card.get('gold_cost', '?'))}{stocked}")
            append_card_lines(lines, card, prefix="    ")
        shop_relics = state.get("relics", []) or []
        lines.append(f"Shop relics ({len(shop_relics)}):")
        for relic in shop_relics:
            stocked = "" if relic.get("is_stocked") else " sold"
            lines.append(f"  [{relic.get('index')}] {name(relic.get('name'))} cost={relic.get('cost')}{stocked}")
            desc = description_of(relic)
            if desc:
                lines.append(f"    description: {desc}")
            append_hover_tip_lines(lines, relic, prefix="    ")
        shop_potions = state.get("potions", []) or []
        lines.append(f"Shop potions ({len(shop_potions)}):")
        for potion in shop_potions:
            stocked = "" if potion.get("is_stocked") else " sold"
            lines.append(f"  [{potion.get('index')}] {name(potion.get('name'))} cost={potion.get('cost')}{stocked} target={potion.get('target_type', '?')}")
            desc = description_of(potion)
            if desc:
                lines.append(f"    description: {desc}")
            append_hover_tip_lines(lines, potion, prefix="    ")
        if state.get("card_removal_cost") is not None:
            lines.append(f"Card removal cost: {state.get('card_removal_cost')}")

    elif decision == "treasure":
        relics = state.get("relics", []) or []
        lines.append(f"Treasure relics ({len(relics)}):")
        for relic in relics:
            lines.append(f"  [{relic.get('index')}] {name(relic.get('name'))}")
            desc = description_of(relic)
            if desc:
                lines.append(f"    description: {desc}")
            append_hover_tip_lines(lines, relic, prefix="    ")
        if not state.get("relics"):
            lines.append(state.get("message", "No relic choices"))

    elif decision == "card_select":
        lines.append(f"Card select: min={state.get('min_select')} max={state.get('max_select')} prompt={name(state.get('prompt'))}")
        for source_key in ("source_event_option", "source_room_option", "source_potion", "source_card", "source_power"):
            source = state.get(source_key)
            if isinstance(source, dict) and source:
                lines.append(f"{source_key}: {name(source.get('title') or source.get('name') or source.get('id'))}")
                desc = description_of(source)
                if desc:
                    lines.append(f"  description: {desc}")
                append_hover_tip_lines(lines, source, prefix="  ")
        combat = state.get("combat")
        if isinstance(combat, dict) and combat:
            lines.append(
                f"Combat context: round={combat.get('round', '?')} energy={combat.get('energy', '?')}/{combat.get('max_energy', '?')} draw={combat.get('draw_pile_count', '?')} discard={combat.get('discard_pile_count', '?')} exhaust={combat.get('exhaust_pile_count', '?')}"
            )
            combat_powers = combat.get("player_powers") or state.get("player_powers") or []
            if combat_powers:
                lines.append(f"Player powers ({len(combat_powers)}):")
                append_power_lines(lines, combat_powers, include_type=True)
            for enemy in combat.get("enemies") or []:
                intents = ",".join(str(intent.get("type")) for intent in enemy.get("intents") or []) or "none"
                powers = enemy.get("powers") or []
                power_text = ""
                if powers:
                    power_text = " powers=" + ",".join(power_label(power) for power in powers)
                lines.append(f"  enemy [{enemy.get('index')}] {name(enemy.get('name'))} hp={enemy.get('hp')}/{enemy.get('max_hp')} block={enemy.get('block', 0)} intents={intents}{power_text}")
                if powers:
                    append_power_lines(lines, powers, prefix="    ", include_type=True)
        cards = state.get("cards", []) or []
        lines.append(f"Selectable cards ({len(cards)}):")
        for card in cards:
            append_card_lines(lines, card)

    elif decision == "game_over":
        lines.append(f"Game over: victory={state.get('victory')} act={state.get('act')} floor={state.get('floor')}")

    append_viewed_information(lines, state, player)
    return "\n".join(lines)
