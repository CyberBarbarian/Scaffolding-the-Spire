"""Neutral Slay the Spire 2 rules shared by structured and direct-CLI baselines.

The text in this module explains stable game mechanics only.  It must not
rank actions, recommend tactics, reproduce card-specific knowledge, or infer
hidden state.  Runtime state and legal actions remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


GAME_RULESET_ID = "sts2-official-rules-v2"
DECISION_RULES_HEADING = "Game rules for the current decision:"

# Sources are the game's English tutorial/localization strings shipped with the
# tested runtime: ftues.json, static_hover_tips.json, map.json, and
# ascension.json.  The module boundaries are benchmark context-management
# boundaries; they do not change the rules themselves.
GAME_RULE_MODULES: dict[str, tuple[str, ...]] = {
    "core_run": (
        "The goal of a run is to climb through the Acts and defeat the final Act boss. If the player's HP reaches 0, the run ends in defeat.",
        "The current engine-exported state, object descriptions, and legal actions are authoritative whenever a card, relic, potion, room, event, or modifier changes a default rule.",
    ),
    "persistent_resources": (
        "Cards make up the player's deck. Relics provide persistent effects, and potions occupy limited slots and can be consumed for their stated effects.",
        "HP, Max HP, Gold, the deck, relics, and held potions persist between rooms unless a stated effect changes them; combat-only state does not automatically persist after combat.",
    ),
    "combat_cycle": (
        "At the start of a normal turn, 5 cards are drawn and Energy replenishes to its per-turn maximum, unless current effects state otherwise.",
        "Cards require their displayed Energy cost. Ending the turn discards the remaining hand, enemies take their turns, then the next player turn begins.",
        "Attack cards resolve their stated on-play effects and count as Attacks for Attack-specific mechanics.",
        "Skill cards resolve their stated on-play effects and do not persist unless their text explicitly creates a continuing effect.",
        "Power cards enter play and establish their described persistent combat effect. Playing a Power resolves only separate text that is explicitly immediate or on-play.",
        '"At the start...", "At the end...", and "Whenever..." clauses are delayed triggers. They do not fire merely because a Power was played. A number inside such a trigger is not immediate damage or Block for the play-card action and must not be counted until that trigger occurs.',
        "If cards must be drawn while the draw pile is empty, the discard pile is shuffled into the draw pile. Exhausted cards remain outside both piles for the rest of that combat unless an effect returns them.",
        "Block reduces incoming attack damage and normally wears off at the start of the player's next turn, after enemies have acted. Enemy Intent shows the currently announced action and may change on later turns.",
    ),
    "map_navigation": (
        "A map is climbed from lower rows to higher rows. After completing a room, the next room must be one of the upward connected choices exposed by the current map and legal actions.",
        "A room already passed cannot be revisited. A special effect may change reachability; the current legal map actions are the final authority.",
    ),
    "room_glossary": (
        "Monster rooms contain normal combats. Elite rooms contain stronger combats and normally award a Relic. Rest Sites offer one available rest-site action. Merchants sell cards, relics, potions, and card removal for Gold. Treasure rooms contain treasure. Boss rooms end an Act when their boss is defeated.",
        "An Unknown (?) room hides its contents until entered and may resolve to an Event, enemies, a Merchant, or Treasure. It is not synonymous with an Event room.",
        "Ancient rooms grant their displayed bonuses, treasures, and knowledge. Event rooms present their own descriptions and options.",
    ),
    "act_transition": (
        "Defeating an Act boss allows progression to the next Act; defeating the Act 3 boss completes the run.",
        "The next floor after an Act 1 or Act 2 boss is an Ancient. At Ascension 0, an Ancient heals 100% of missing HP; higher Ascension modifiers may reduce this, so the current ascension and state remain authoritative.",
    ),
    "post_combat_rewards": (
        "After combat, rewards remain separate until claimed or skipped through the available reward actions. Claiming one reward does not imply that the other displayed rewards were claimed.",
        "A card reward is normally a pack of 3 offered cards from which at most 1 may be added to the deck; the current reward state states whether it may be skipped.",
    ),
    "treasure": (
        "A Treasure room presents the treasure choices shown by the current state. Claim treasure only through a currently legal action; the resulting state determines which choices remain and when the room is complete.",
        "A claimed Relic becomes part of the player's persistent relic collection and provides its stated effect.",
    ),
    "deckbuilding_choice": (
        "Adding a card places it in the persistent deck used in later combats. Removing, transforming, or upgrading a card changes the persistent deck exactly as the current prompt and selected operation state.",
        "A multi-card selector may instead operate only on the current combat or another stated effect. Its prompt, source object, selection limits, candidates, and legal actions define the operation; do not infer a permanent deck change when none is stated.",
    ),
    "rest_site": (
        "At a Rest Site, exactly one currently available option is chosen before moving on. Default options can include Rest, which heals 30% of Max HP rounded down, and Smith, which upgrades a card; current effects may add, remove, or change options.",
    ),
    "merchant": (
        "A Merchant sells the currently displayed cards, relics, and potions for Gold and may offer card removal. Purchases require the displayed price, and leaving ends access to that shop.",
    ),
    "event": (
        "An Event is governed by its current description and options. Options can be enabled, disabled, or locked and may lead to further event states, combat, rewards, or another selection.",
        "An event-specific interface is still part of the same event until its current state exposes a proceed or leave action.",
    ),
    "crystal_sphere": (
        "In the Crystal Sphere interface, a tool selection and a cell click are separate actions. The current tool, remaining divinations, visible cells, revealed items, clickable cells, and can_proceed flag define what can be done.",
    ),
}

_BASE_MODULES = ("core_run", "persistent_resources")
_NO_ACTION_RULE_DECISIONS = {
    "game_over",
    "unrecognized_state",
    "event_blocked",
    "unknown",  # Legacy trace compatibility; new runtimes emit unrecognized_state.
}


def _normalized_room_type(state: dict[str, Any]) -> str:
    context = state.get("context")
    value = context.get("room_type") if isinstance(context, dict) else None
    if value in (None, ""):
        value = state.get("room_type")
    token = "".join(ch for ch in str(value or "").lower() if ch.isalnum())
    aliases = {
        "combat": "monster",
        "combatroom": "monster",
        "monsterroom": "monster",
        "eliteroom": "elite",
        "bossroom": "boss",
        "rest": "restsite",
        "restroom": "restsite",
        "restsiteroom": "restsite",
        "merchant": "merchant",
        "merchantroom": "merchant",
        "shop": "merchant",
        "shoproom": "merchant",
        "eventroom": "event",
        "treasureroom": "treasure",
        "ancientroom": "ancient",
    }
    return aliases.get(token, token)


def _append_unique(target: list[str], module_ids: Iterable[str]) -> None:
    for module_id in module_ids:
        if module_id not in GAME_RULE_MODULES:
            raise KeyError(f"Unknown game-rule module: {module_id}")
        if module_id not in target:
            target.append(module_id)


def _room_modules(room_type: str) -> tuple[str, ...]:
    if room_type == "restsite":
        return ("rest_site",)
    if room_type == "merchant":
        return ("merchant",)
    if room_type == "event":
        return ("event",)
    if room_type in {"monster", "elite", "boss"}:
        return ("combat_cycle",)
    if room_type == "ancient":
        return ("act_transition",)
    return ()


def _card_select_parent_modules(state: dict[str, Any]) -> tuple[str, ...]:
    if isinstance(state.get("combat"), dict) and state.get("combat"):
        return ("combat_cycle",)
    if isinstance(state.get("source_event_option"), dict) and state.get("source_event_option"):
        return ("event",)
    room_type = _normalized_room_type(state)
    if isinstance(state.get("source_room_option"), dict) and state.get("source_room_option"):
        return _room_modules(room_type)
    if any(isinstance(state.get(key), dict) and state.get(key) for key in ("source_potion", "source_card", "source_power")):
        return _room_modules(room_type)
    return _room_modules(room_type)


def _public_source_modules(state: dict[str, Any]) -> tuple[str, ...]:
    """Map exported provenance fields to their neutral rule domains."""

    modules: list[str] = []
    room_modules = _room_modules(_normalized_room_type(state))
    has_event_option = isinstance(state.get("source_event_option"), dict) and state.get("source_event_option")
    if has_event_option or state.get("from_event") is True:
        _append_unique(modules, ("event",))
    if any(
        isinstance(state.get(key), dict) and state.get(key)
        for key in ("source_room_option", "source_potion", "source_card", "source_power")
    ):
        _append_unique(modules, room_modules)
    return tuple(modules)


def rule_module_ids_for_state(state: dict[str, Any]) -> tuple[str, ...]:
    """Select neutral rule modules using only public decision/context/source fields."""

    decision = str(state.get("decision") or "")
    if decision in _NO_ACTION_RULE_DECISIONS:
        return ()
    if decision == "event_result":
        return ("core_run",)

    modules = list(_BASE_MODULES)
    room_type = _normalized_room_type(state)

    if decision == "combat_play":
        _append_unique(modules, ("combat_cycle",))
        if room_type == "boss":
            _append_unique(modules, ("act_transition",))
    elif decision == "map_select":
        _append_unique(modules, ("map_navigation", "room_glossary", "act_transition"))
    elif decision == "combat_reward":
        _append_unique(modules, ("post_combat_rewards",))
        if room_type == "boss":
            _append_unique(modules, ("act_transition",))
    elif decision == "card_reward":
        _append_unique(modules, ("deckbuilding_choice",))
        source_modules = _public_source_modules(state)
        room_modules = _room_modules(room_type)
        scoped_room_modules = room_modules if "combat_cycle" not in room_modules else ()
        _append_unique(modules, source_modules)
        _append_unique(modules, scoped_room_modules)
        if not source_modules and not scoped_room_modules:
            _append_unique(modules, ("post_combat_rewards",))
        if room_type == "boss":
            _append_unique(modules, ("act_transition",))
    elif decision == "bundle_select":
        _append_unique(modules, ("deckbuilding_choice",))
        if room_type == "event":
            _append_unique(modules, ("event",))
    elif decision == "card_select":
        _append_unique(modules, _card_select_parent_modules(state))
        if not (isinstance(state.get("combat"), dict) and state.get("combat")):
            _append_unique(modules, ("deckbuilding_choice",))
    elif decision == "rest_site":
        _append_unique(modules, ("rest_site",))
    elif decision == "shop":
        _append_unique(modules, ("merchant",))
    elif decision == "fake_merchant_shop":
        _append_unique(modules, ("event",))
    elif decision == "event_choice":
        _append_unique(modules, ("event",))
    elif decision == "crystal_sphere":
        _append_unique(modules, ("event", "crystal_sphere"))
    elif decision == "treasure":
        _append_unique(modules, ("treasure",))

    if state.get("view_map") is True:
        _append_unique(modules, ("map_navigation", "room_glossary"))
    if any(state.get(key) is True for key in ("view_draw_pile", "view_discard_pile", "view_exhaust_pile")):
        if isinstance(state.get("combat"), dict) and state.get("combat"):
            _append_unique(modules, ("combat_cycle",))

    return tuple(modules)


def _render_modules(module_ids: Iterable[str], *, heading: str) -> str:
    lines = [heading, "Only the modules listed below apply to this prompt; current state and legal actions override defaults."]
    for module_id in module_ids:
        lines.append(f"[{module_id}]")
        lines.extend(f"- {rule}" for rule in GAME_RULE_MODULES[module_id])
    return "\n".join(lines)


def render_game_rules(state: dict[str, Any]) -> str:
    module_ids = rule_module_ids_for_state(state)
    if not module_ids:
        return ""
    return _render_modules(module_ids, heading=DECISION_RULES_HEADING)


def render_rule_modules(module_ids: Iterable[str], *, heading: str) -> str:
    """Render a caller-selected, de-duplicated rule-module sequence."""

    selected: list[str] = []
    _append_unique(selected, module_ids)
    return _render_modules(selected, heading=heading) if selected else ""


def render_full_game_rules() -> str:
    return _render_modules(GAME_RULE_MODULES, heading=f"Game rules ({GAME_RULESET_ID}):")


def _canonical_ruleset_payload() -> dict[str, Any]:
    return {
        "ruleset_id": GAME_RULESET_ID,
        "modules": [{"module_id": module_id, "rules": list(rules)} for module_id, rules in GAME_RULE_MODULES.items()],
    }


def game_rules_identity() -> dict[str, Any]:
    payload = _canonical_ruleset_payload()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "ruleset_id": GAME_RULESET_ID,
        "ruleset_sha256": hashlib.sha256(encoded).hexdigest(),
        "module_ids": list(GAME_RULE_MODULES),
    }


def audit_game_rule_prompt(
    prompt: str,
    *,
    expected_render: str,
    module_ids: Iterable[str],
    delivery: str,
    empty_render_marker: str = DECISION_RULES_HEADING,
) -> dict[str, Any]:
    """Audit an actual final prompt against the exact expected rule block."""

    expected_occurrences = 1 if expected_render else 0
    occurrences = prompt.count(expected_render) if expected_render else prompt.count(empty_render_marker)
    marker_occurrences = prompt.count(empty_render_marker)
    expected_marker_occurrences = expected_occurrences
    expected_sha = hashlib.sha256(expected_render.encode("utf-8")).hexdigest()
    return {
        "game_rule_module_ids": list(module_ids),
        "game_rule_render_sha256": expected_sha,
        "game_rule_expected_block_sha256": expected_sha,
        "game_rule_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "game_rule_occurrences": occurrences,
        "game_rule_expected_occurrences": expected_occurrences,
        "game_rule_marker_occurrences": marker_occurrences,
        "game_rule_expected_marker_occurrences": expected_marker_occurrences,
        "game_rule_exposure_valid": (
            occurrences == expected_occurrences
            and marker_occurrences == expected_marker_occurrences
        ),
        "game_rule_delivery": delivery,
    }


def game_rules_meta(
    state: dict[str, Any],
    prompt: str | None = None,
    *,
    projection_enabled: bool = True,
) -> dict[str, Any]:
    module_ids = rule_module_ids_for_state(state) if projection_enabled else []
    rendered = render_game_rules(state) if projection_enabled else ""
    delivery = "state_selected_prompt" if projection_enabled else "omitted"
    identity = game_rules_identity()
    meta = {
        "game_ruleset_id": identity["ruleset_id"],
        "game_ruleset_sha256": identity["ruleset_sha256"],
        "game_rule_module_ids": list(module_ids),
        "game_rule_render_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "game_rule_delivery": delivery,
        "game_rule_projection_enabled": projection_enabled,
    }
    if prompt is not None:
        meta.update(
            audit_game_rule_prompt(
                prompt,
                expected_render=rendered,
                module_ids=module_ids,
                delivery=delivery,
            )
        )
    return meta


def full_game_rules_meta(prompt: str | None = None) -> dict[str, Any]:
    """Return audit metadata for the static full rules-v2 system contract."""

    rendered = render_full_game_rules()
    identity = game_rules_identity()
    meta = {
        "game_ruleset_id": identity["ruleset_id"],
        "game_ruleset_sha256": identity["ruleset_sha256"],
        "game_rule_module_ids": list(identity["module_ids"]),
        "game_rule_render_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "game_rule_delivery": "static_full_system_prompt",
        "game_rule_projection_enabled": False,
    }
    if prompt is not None:
        meta.update(
            audit_game_rule_prompt(
                prompt,
                expected_render=rendered,
                module_ids=identity["module_ids"],
                delivery="static_full_system_prompt",
                empty_render_marker=f"Game rules ({GAME_RULESET_ID}):",
            )
        )
    return meta
