"""Policy agents used by benchmark runners."""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .actions import LegalAction
from .context import (
    append_last_action_result,
    append_viewed_information,
    build_last_action_result,
    card_line,
    compact_state,
    name,
    power_label,
    render_state_text,
)
from .sequences import normalize_card_sequence_payload


PromptStyle = Literal["default", "analysis"]
MemoryMode = Literal["action_reason", "factual_diff"]
ConversationMode = Literal["single_turn", "turn_chat"]
TurnChatUpdateMode = Literal["delta"]
TurnChatAssistantHistory = Literal["compact"]
FallbackMode = Literal["action", "error"]


class ModelActionError(RuntimeError):
    """Raised when an LLM cannot produce a valid legal action after retries."""


class Agent(Protocol):
    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        """Return the prompt this agent would use for this decision."""

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt: str | None = None,
    ) -> tuple[LegalAction, dict[str, Any]]:
        """Return a legal action and metadata for logging."""


@dataclass
class ShortTermMemory:
    """Small per-episode memory owned by an agent, not by state rendering."""

    max_entries: int = 8
    mode: MemoryMode = "action_reason"
    entries: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.entries.clear()

    def remember(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        if self.max_entries <= 0:
            return
        if self.mode == "factual_diff":
            entry = self._factual_entry(state, action, next_state)
        else:
            entry = self._action_reason_entry(state, action, meta)
        if not entry:
            return

        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            del self.entries[: len(self.entries) - self.max_entries]

    def _action_reason_entry(self, state: dict[str, Any], action: LegalAction, meta: dict[str, Any]) -> str:
        player = state.get("player") or {}
        parsed = meta.get("parsed") if isinstance(meta.get("parsed"), dict) else {}
        reason = _one_line(parsed.get("reason") or meta.get("reason") or "")
        parts = self._location_parts(state)
        if player:
            parts.append(f"hp={player.get('hp', '?')}/{player.get('max_hp', '?')}")
            parts.append(f"gold={player.get('gold', '?')}")
            parts.append(f"deck_size={player.get('deck_size', '?')}")
        parts.append(f"action={action.label}")
        if reason:
            parts.append(f"reason={reason}")
        return " | ".join(parts)

    def _factual_entry(
        self,
        state: dict[str, Any],
        action: LegalAction,
        next_state: dict[str, Any] | None,
    ) -> str:
        if next_state is None or action.command.get("cmd") == "bench_view":
            return ""

        result = next_state.get("last_action_result")
        if not isinstance(result, dict):
            result = build_last_action_result(
                state,
                next_state,
                action_label=action.label,
                action_kind=action.kind,
            )
        parts = self._location_parts(next_state)
        parts.append(f"action={result.get('action_label') or action.label}")
        before = result.get("decision_before")
        after = result.get("decision_after")
        if before is not None or after is not None:
            parts.append(f"transition={before} -> {after}")
        changes = [_one_line(change) for change in result.get("changes") or [] if _one_line(change)]
        if changes:
            parts.append("changes=" + "; ".join(changes))
        return " | ".join(parts)

    def _location_parts(self, state: dict[str, Any]) -> list[str]:
        context = state.get("context") or {}
        return [
            f"decision={state.get('decision', '?')}",
            f"act={state.get('act') or context.get('act', '?')}",
            f"floor={state.get('floor') or context.get('floor', '?')}",
            f"room={context.get('room_type', '?')}",
        ]

    def render(self) -> str:
        if not self.entries:
            return ""
        return "\n".join(f"- {entry}" for entry in self.entries)


@dataclass
class TurnChatBuffer:
    """Current-player-turn chat history for an LLM agent.

    This is an agent-side context-management buffer only.  It does not change
    the benchmark action surface or the headless environment command protocol.
    """

    max_turns: int
    plan_enabled: bool = False
    messages: list[dict[str, str]] = field(default_factory=list)
    active_turn_key: tuple[Any, ...] | None = None

    def reset(self) -> None:
        self.messages.clear()
        self.active_turn_key = None

    def history_turns(self) -> int:
        return sum(1 for message in self.messages if message.get("role") == "assistant")

    def messages_for_decision(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt_style: PromptStyle,
        run_summary_text: str,
        update_mode: TurnChatUpdateMode,
        record_user: bool,
    ) -> list[dict[str, str]]:
        system = {
            "role": "system",
            "content": build_turn_chat_system_message(prompt_style, turn_plan_enabled=self.plan_enabled),
        }
        turn_key = turn_chat_key(state)
        if turn_key is None:
            if record_user:
                self.reset()
            prompt = build_llm_prompt(
                state,
                legal_actions,
                include_json=False,
                prompt_style=prompt_style,
                memory_text="",
                run_summary_text=run_summary_text,
            )
            return [
                {
                    "role": "system",
                    "content": "You are a careful game-playing policy. Return only valid JSON. Never invent actions.",
                },
                {"role": "user", "content": prompt},
            ]

        history = list(self.messages) if self.active_turn_key == turn_key else []
        user_message = self._build_user_message(
            state,
            legal_actions,
            run_summary_text=run_summary_text,
            update_mode=update_mode,
            is_initial=not history,
        )
        if record_user:
            if self.active_turn_key != turn_key:
                self.reset()
                self.active_turn_key = turn_key
            self.messages.append({"role": "user", "content": user_message})
            history = list(self.messages)
        else:
            history.append({"role": "user", "content": user_message})
        return [system, *history]

    def remember_assistant(self, action: LegalAction, parsed: dict[str, Any]) -> None:
        if self.active_turn_key is None:
            return
        self.messages.append(
            {
                "role": "assistant",
                "content": build_compact_assistant_message(action, parsed),
            }
        )
        self._trim()

    def remember_user_feedback(self, feedback: str) -> None:
        if self.active_turn_key is None:
            return
        self.messages.append({"role": "user", "content": feedback})
        self._trim()

    def after_transition(self, action: LegalAction, next_state: dict[str, Any] | None) -> None:
        if action.kind == "combat_end_turn":
            self.reset()
            return
        if next_state is None:
            return
        next_turn_key = turn_chat_key(next_state)
        if next_turn_key is None:
            self.reset()
            return
        if self.active_turn_key is not None and next_turn_key != self.active_turn_key:
            self.reset()

    def _trim(self) -> None:
        if self.max_turns <= 0:
            self.messages.clear()
            return

        assistant_seen = 0
        keep_from = 0
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].get("role") == "assistant":
                assistant_seen += 1
                if assistant_seen > self.max_turns:
                    keep_from = index + 1
                    break
        if keep_from:
            del self.messages[:keep_from]

    def _build_user_message(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        run_summary_text: str,
        update_mode: TurnChatUpdateMode,
        is_initial: bool,
    ) -> str:
        if is_initial:
            return build_turn_initial_user_message(
                state,
                legal_actions,
                run_summary_text,
            )
        if update_mode != "delta":
            raise ValueError(f"Unsupported turn_chat update mode: {update_mode}")
        if _viewed_labels(state):
            return build_view_update_user_message(state, legal_actions)
        return build_turn_update_user_message(state, legal_actions)


def build_llm_prompt(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    include_json: bool = True,
    prompt_style: PromptStyle = "default",
    memory_text: str = "",
    run_summary_text: str = "",
) -> str:
    """Build the action-selection prompt used by LLM policies.

    ``context.py`` owns only state rendering.  The policy-facing task
    instruction, response schema, and legal-action wrapper live with agents so
    different agents can define different prompting methods over the same
    state text.
    """

    if prompt_style not in {"default", "analysis"}:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    parts = _prompt_header(prompt_style)
    if run_summary_text:
        parts.extend(["", "Run summary:", run_summary_text])
    if memory_text:
        parts.extend(["", "Episode memory:", memory_text])
    parts.extend(
        [
            "",
            "Game state:",
            render_state_text(state),
            "",
            "Legal actions:",
            json.dumps([action.to_prompt_dict() for action in legal_actions], ensure_ascii=False, separators=(",", ":")),
        ]
    )
    if include_json:
        parts.extend(
            [
                "",
                "Compact state JSON:",
                json.dumps(compact_state(state), ensure_ascii=False, separators=(",", ":")),
            ]
        )
    return "\n".join(parts)


def build_turn_chat_system_message(prompt_style: PromptStyle, *, turn_plan_enabled: bool = False) -> str:
    return "\n".join(_prompt_header(prompt_style, turn_plan_enabled=turn_plan_enabled))


def build_turn_initial_user_message(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    run_summary_text: str = "",
) -> str:
    parts = ["New player turn."]
    if run_summary_text:
        parts.extend(["", "Run summary:", run_summary_text])
    parts.extend(
        [
            "",
            "Game state:",
            render_state_text(state),
            "",
            "Legal actions:",
            _legal_actions_text(legal_actions),
        ]
    )
    return "\n".join(parts)


def build_turn_update_user_message(state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
    if state.get("decision") == "card_select":
        return "\n".join(
            [
                "Turn modal.",
                "",
                render_state_text(state),
                "",
                "Legal actions:",
                _legal_actions_text(legal_actions),
            ]
        )

    lines = ["Turn update."]
    last_action_lines: list[str] = []
    append_last_action_result(last_action_lines, state.get("last_action_result"))
    if last_action_lines:
        lines.extend(last_action_lines)
    lines.extend(
        [
            "",
            "Current turn state:",
            _render_current_turn_state(state),
            "",
            "Legal actions:",
            _legal_actions_text(legal_actions),
        ]
    )
    return "\n".join(lines)


def build_view_update_user_message(state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
    viewed = _viewed_labels(state)
    lines = ["Requested information: " + ", ".join(viewed) + "."]
    append_viewed_information(lines, state, state.get("player") or {})
    lines.extend(
        [
            "",
            "Current turn reminder:",
            _render_current_turn_state(state),
            "",
            "Legal actions:",
            _legal_actions_text(legal_actions),
        ]
    )
    return "\n".join(lines)


def build_compact_assistant_message(action: LegalAction, parsed: dict[str, Any]) -> str:
    reason = _one_line(parsed.get("reason") if isinstance(parsed, dict) else "")
    payload: dict[str, Any] = {
        "action": action.label,
        "action_kind": action.kind,
    }
    if isinstance(parsed, dict):
        for key in ("turn_plan", "plan_update"):
            if key in parsed:
                payload[key] = parsed[key]
        if action.kind == "combat_custom_sequence" and "sequence" in parsed:
            payload["sequence"] = parsed["sequence"]
    if reason:
        payload["reason"] = reason
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def turn_chat_key(state: dict[str, Any]) -> tuple[Any, ...] | None:
    decision = state.get("decision")
    combat = state.get("combat") if isinstance(state.get("combat"), dict) else {}
    if decision == "combat_play":
        round_number = state.get("round")
    elif decision == "card_select" and combat:
        round_number = combat.get("round")
    else:
        return None

    if round_number is None:
        return None

    context = state.get("context") or {}
    boss = context.get("boss") if isinstance(context.get("boss"), dict) else {}
    return (
        context.get("act", state.get("act")),
        context.get("floor", state.get("floor")),
        context.get("room_type"),
        name(boss.get("name")) if boss else "",
        round_number,
    )


def _legal_actions_text(legal_actions: list[LegalAction]) -> str:
    return json.dumps([action.to_prompt_dict() for action in legal_actions], ensure_ascii=False, separators=(",", ":"))


def _viewed_labels(state: dict[str, Any]) -> list[str]:
    labels = []
    for label, key in (
        ("deck", "view_deck"),
        ("map", "view_map"),
        ("draw pile", "view_draw_pile"),
        ("discard pile", "view_discard_pile"),
        ("exhaust pile", "view_exhaust_pile"),
    ):
        if state.get(key):
            labels.append(label)
    return labels


def _render_current_turn_state(state: dict[str, Any]) -> str:
    if state.get("decision") != "combat_play":
        return render_state_text(state)

    player = state.get("player") or {}
    lines = [
        (
            f"Player: hp={player.get('hp', '?')}/{player.get('max_hp', '?')} "
            f"block={player.get('block', 0)} energy={state.get('energy', '?')}/{state.get('max_energy', '?')} "
            f"potions={_potion_count_text(player)}"
        )
    ]

    powers = state.get("player_powers") or []
    if powers:
        lines.append("Player powers:")
        for power in powers:
            lines.append(f"  {power_label(power, include_type=True)}")

    enemies = state.get("enemies") or []
    lines.append("Enemies:")
    if not enemies:
        lines.append("  none")
    for enemy in enemies:
        if enemy.get("hp", 0) <= 0:
            continue
        powers_text = ""
        powers = enemy.get("powers") or []
        if powers:
            powers_text = " powers=" + ",".join(power_label(power) for power in powers)
        lines.append(
            f"  [{enemy.get('index')}] {name(enemy.get('name'))} "
            f"hp={enemy.get('hp')}/{enemy.get('max_hp')} block={enemy.get('block', 0)} "
            f"intent={_intent_text(enemy)}{powers_text}"
        )

    hand = state.get("hand") or []
    lines.append("Hand:")
    if not hand:
        lines.append("  empty")
    for card in hand:
        lines.append("  " + card_line(card))

    lines.append(
        f"Piles: draw={state.get('draw_pile_count', '?')} "
        f"discard={state.get('discard_pile_count', '?')} exhaust={state.get('exhaust_pile_count', '?')}"
    )
    return "\n".join(lines)


def _potion_count_text(player: dict[str, Any]) -> str:
    potions = [potion for potion in player.get("potions", []) or [] if potion]
    slots = player.get("potion_slots")
    return f"{len(potions)}/{slots}" if slots is not None else str(len(potions))


def _intent_text(enemy: dict[str, Any]) -> str:
    intents = []
    for intent in enemy.get("intents") or []:
        if intent.get("type") == "Attack":
            hits = intent.get("hits", 1)
            if hits and hits > 1:
                intents.append(f"Attack {intent.get('damage')}x{hits}")
            else:
                intents.append(f"Attack {intent.get('damage')}")
        else:
            intents.append(str(intent.get("type")))
    return ",".join(intents) or "none"


def _messages_char_count(messages: list[dict[str, str]]) -> int:
    return sum(len(message.get("content", "")) for message in messages)


def _messages_preview(messages: list[dict[str, str]]) -> str:
    parts = []
    for index, message in enumerate(messages):
        parts.append(f"===== message[{index}] role={message.get('role', '?')} =====")
        parts.append(message.get("content", ""))
    return "\n".join(parts)


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def build_run_summary(state: dict[str, Any]) -> str:
    """Build a small rule-based method summary without replacing state text."""

    context = state.get("context") or {}
    player = state.get("player") or {}
    lines: list[str] = []

    boss = context.get("boss") if isinstance(context.get("boss"), dict) else {}
    position = [
        f"act={state.get('act') or context.get('act', '?')}",
        f"floor={state.get('floor') or context.get('floor', '?')}",
        f"room={context.get('room_type', '?')}",
    ]
    boss_name = name(boss.get("name")) if boss else ""
    if boss_name and boss_name != "?":
        position.append(f"boss={boss_name}")
    lines.append("- position: " + " ".join(position))

    if player:
        potions = [potion for potion in player.get("potions", []) or [] if potion]
        slots = player.get("potion_slots")
        potion_text = f"{len(potions)}"
        if slots is not None:
            potion_text += f"/{slots}"
        resources = [
            f"hp={player.get('hp', '?')}/{player.get('max_hp', '?')}",
            f"gold={player.get('gold', '?')}",
            f"deck_size={player.get('deck_size', '?')}",
            f"potions={potion_text}",
        ]
        relics = [relic for relic in player.get("relics", []) or [] if relic]
        if relics:
            resources.append(f"relic_count={len(relics)}")
        lines.append("- resources: " + " ".join(resources))

    warnings = _run_summary_warnings(state)
    if warnings:
        lines.append("- warnings: " + "; ".join(warnings))

    return "\n".join(lines)


def _run_summary_warnings(state: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    context = state.get("context") or {}
    player = state.get("player") or {}

    hp = player.get("hp")
    max_hp = player.get("max_hp")
    if isinstance(hp, (int, float)) and isinstance(max_hp, (int, float)) and max_hp > 0:
        if hp / max_hp <= 0.35:
            warnings.append("low HP; survival and rest decisions need extra scrutiny")

    potions = [potion for potion in player.get("potions", []) or [] if potion]
    if context.get("room_type") in {"Boss", "Elite"} and not potions:
        warnings.append("no potions available for this high-risk fight")

    negative_powers = []
    for power in state.get("player_powers") or []:
        amount = power.get("amount")
        if isinstance(amount, (int, float)) and amount < 0:
            negative_powers.append(f"{name(power.get('name'))}({amount})")
    if negative_powers:
        warnings.append("negative player powers: " + ", ".join(negative_powers))

    if state.get("decision") == "combat_play" and _has_enemies(state) and not _has_enemy_attack_intent(state):
        warnings.append("visible enemy intents include no attack damage this turn")

    return warnings


def _has_enemies(state: dict[str, Any]) -> bool:
    return any(enemy.get("hp", 0) > 0 for enemy in state.get("enemies", []) or [])


def _has_enemy_attack_intent(state: dict[str, Any]) -> bool:
    for enemy in state.get("enemies", []) or []:
        if enemy.get("hp", 0) <= 0:
            continue
        for intent in enemy.get("intents") or []:
            if intent.get("type") in {"Attack", "DeathBlow"}:
                return True
    return False


def _prompt_header(prompt_style: PromptStyle, *, turn_plan_enabled: bool = False) -> list[str]:
    shared = [
        "Each legal action is one atomic benchmark action, not an open-ended plan. Most legal actions execute one engine command; combat_card_sequence is an engine-side queued card sequence that may play multiple listed cards before the next LLM request.",
        "Choose combat_card_sequence only for deterministic same-turn combat actions where every listed card is already in hand, currently playable, and does not rely on draw, random effects, discard/exhaust choices, generated cards, or other modal decisions.",
        "If useful deterministic compression is not listed, you may choose combat_custom_sequence and include a sequence array of card steps like {\"card_index\":0,\"target_index\":0}; the benchmark validates it with the engine before execution.",
        "For combat_custom_sequence, card_index values refer to the current visible hand snapshot; do not predict that indices will shift after earlier cards are played. If the sequence depends on post-play hand changes, use single legal actions instead.",
        "After playing a card, using a potion, choosing a reward, viewing information, or executing a card sequence, you will receive a fresh state and may act again if the game still allows actions.",
        "In combat, end turn is the action that intentionally finishes the current turn; do not choose it while useful playable cards or potions remain unless passing is strategically better.",
        "Enemy intent damage shown in the state is the engine-displayed damage after currently visible modifiers; do not add enemy Strength or other visible modifiers to that intent damage a second time.",
        "Card damage and block values shown in the state are engine preview values after currently visible player modifiers such as Strength, Dexterity, Frail, Weak, and card-specific temporary effects; do not apply those visible modifiers to shown card damage or block a second time.",
        "If all visible enemy intents are non-attack intents, enemies are not making attack damage this turn; block usually expires at end of turn, so avoid spending energy only for block unless another effect justifies it.",
        "View deck/map/pile actions do not advance the game state. They are information requests logged separately; use them when deck composition, pile contents, path context, or current position could affect the decision.",
    ]
    turn_plan_rules = _turn_plan_header_rules() if turn_plan_enabled else []

    if prompt_style == "analysis":
        schema = (
            '{"situation":"<current objective and main risk>",'
            '"calculations":["<damage/block/energy/path/reward calculation if relevant>"],'
            '"candidates":[{"action_id":<integer>,"label":"<legal action label>",'
            '"pros":"<why it helps>","cons":"<main risk or cost>"}],'
            '"action_id":<integer>,"selected_action_label":"<copy exact chosen legal action label>",'
            '"sequence":[{"card_index":<integer>,"target_index":<integer if needed>}],'
            '"reason":"<final concise reason>"}'
        )
        if turn_plan_enabled:
            schema = (
                '{"turn_plan":{"objective":"<goal for this visible hand>",'
                '"survival_check":"<incoming damage and block/kill requirement>",'
                '"tentative_sequence":["<advisory card/order idea, not commands>"],'
                '"replan_if":["<state change that invalidates the plan>"]},'
                '"situation":"<current objective and main risk>",'
                '"calculations":["<damage/block/energy/path/reward calculation if relevant>"],'
                '"candidates":[{"action_id":<integer>,"label":"<legal action label>",'
                '"pros":"<why it helps>","cons":"<main risk or cost>"}],'
                '"action_id":<integer>,"selected_action_label":"<copy exact chosen legal action label>",'
                '"sequence":[{"card_index":<integer>,"target_index":<integer if needed>}],'
                '"reason":"<final concise reason>"}'
            )
        return [
            "You are playing Slay the Spire 2 through a headless benchmark environment.",
            "Choose exactly one legal action. Use only an action_id from the legal action list.",
            "Also copy the selected legal action label exactly into selected_action_label; action ids are state-local, so never reuse an action_id from earlier messages without checking the current legal actions.",
            *shared,
            *turn_plan_rules,
            "Before choosing, write a compact public analysis in JSON. Analyze the current situation, do any needed arithmetic, compare a few plausible legal actions, then choose.",
            "Return only valid JSON with this schema:",
            schema,
            "For combat, compute enemy attacks from intents yourself and compare playable damage, block, energy, and lethal lines.",
            "For map choices, compare path rewards and risks. For rewards, compare deck impact.",
        ]

    schema = '{"action_id": <integer>, "selected_action_label": "<copy exact chosen legal action label>", "sequence": [{"card_index": <integer>, "target_index": <integer if needed>}], "reason": "<short reason>"}'
    if turn_plan_enabled:
        schema = (
            '{"turn_plan": {"objective": "<goal for this visible hand>", '
            '"survival_check": "<incoming damage and block/kill requirement>", '
            '"tentative_sequence": ["<advisory card/order idea, not commands>"], '
            '"replan_if": ["<state change that invalidates the plan>"]}, '
            '"action_id": <integer>, "selected_action_label": "<copy exact chosen legal action label>", '
            '"sequence": [{"card_index": <integer>, "target_index": <integer if needed>}], '
            '"reason": "<short reason>"}'
        )
    return [
        "You are playing Slay the Spire 2 through a headless benchmark environment.",
        "Choose exactly one legal action. Return only JSON with this schema:",
        *shared,
        "Also copy the selected legal action label exactly into selected_action_label; action ids are state-local, so never reuse an action_id from earlier messages without checking the current legal actions.",
        *turn_plan_rules,
        schema,
    ]


def _turn_plan_header_rules() -> list[str]:
    return [
        (
            'When the current user message starts with "New player turn." and the game state '
            'decision is combat_play, include a top-level "turn_plan" object.'
        ),
        (
            "The turn_plan should plan the visible hand in detail: incoming damage, lethal possibilities, "
            "block needs, energy, card order, draw/random effects, modal risks, and conditions that require replanning."
        ),
        (
            'For later messages in the same player turn, follow the prior turn_plan when it is still valid; '
            'if the latest state invalidates it, you may include a top-level "plan_update" string or object.'
        ),
        "The turn_plan is advisory history only. Still choose exactly one current legal action_id.",
    ]


class RandomAgent:
    """Simple non-LLM baseline."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        include_json_state: bool = False,
        prompt_style: PromptStyle = "default",
        memory_enabled: bool = False,
        memory_window: int = 8,
        memory_mode: MemoryMode = "action_reason",
        run_summary_enabled: bool = True,
    ) -> None:
        self.rng = random.Random(seed)
        self.include_json_state = include_json_state
        self.prompt_style = prompt_style
        self.memory = ShortTermMemory(memory_window, mode=memory_mode) if memory_enabled else None
        self.run_summary_enabled = run_summary_enabled

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        return build_llm_prompt(
            state,
            legal_actions,
            include_json=self.include_json_state,
            prompt_style=self.prompt_style,
            memory_text=self.memory.render() if self.memory else "",
            run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
        )

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt: str | None = None,
    ) -> tuple[LegalAction, dict[str, Any]]:
        action = self.rng.choice(legal_actions)
        return action, {"agent": "random"}

    def reset_episode(self) -> None:
        if self.memory:
            self.memory.reset()

    def record_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        if self.memory:
            self.memory.remember(state, action, meta, next_state)


@dataclass
class OpenAICompatAgent:
    """Local LLM agent for OpenAI-compatible servers.

    Works with Ollama, LM Studio, vLLM, llama.cpp server, and similar local
    servers that expose ``/v1/chat/completions``.
    """

    base_url: str
    model: str
    api_key: str = "local"
    temperature: float = 0.0
    timeout: float = 120.0
    include_json_state: bool = False
    max_retries: int = 2
    prompt_style: PromptStyle = "default"
    memory_enabled: bool = False
    memory_window: int = 8
    memory_mode: MemoryMode = "action_reason"
    run_summary_enabled: bool = True
    conversation_mode: ConversationMode = "single_turn"
    turn_chat_window: int = 4
    turn_chat_update_mode: TurnChatUpdateMode = "delta"
    turn_chat_assistant_history: TurnChatAssistantHistory = "compact"
    turn_chat_plan_enabled: bool = False
    fallback_mode: FallbackMode = "error"
    turn_chat_buffer: TurnChatBuffer = field(init=False)
    pending_external_feedback: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.conversation_mode not in {"single_turn", "turn_chat"}:
            raise ValueError(f"Unknown conversation mode: {self.conversation_mode}")
        if self.turn_chat_update_mode != "delta":
            raise ValueError(f"Unsupported turn_chat_update_mode: {self.turn_chat_update_mode}")
        if self.turn_chat_assistant_history != "compact":
            raise ValueError(f"Unsupported turn_chat_assistant_history: {self.turn_chat_assistant_history}")
        if self.fallback_mode not in {"action", "error"}:
            raise ValueError(f"Unsupported fallback_mode: {self.fallback_mode}")
        if self.turn_chat_window < 1:
            raise ValueError("turn_chat_window must be at least 1")
        if self.conversation_mode == "turn_chat" and self.memory_enabled:
            raise ValueError("turn_chat is a context-management method parallel to memory_enabled; disable memory_enabled when using turn_chat")
        self.memory = ShortTermMemory(self.memory_window, mode=self.memory_mode) if self.memory_enabled else None
        self.turn_chat_buffer = TurnChatBuffer(
            max_turns=self.turn_chat_window,
            plan_enabled=self.turn_chat_plan_enabled,
        )

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        if self.conversation_mode == "turn_chat":
            messages = self.turn_chat_buffer.messages_for_decision(
                state,
                legal_actions,
                prompt_style=self.prompt_style,
                run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
                update_mode=self.turn_chat_update_mode,
                record_user=False,
            )
            return _messages_preview(messages)

        return build_llm_prompt(
            state,
            legal_actions,
            include_json=self.include_json_state,
            prompt_style=self.prompt_style,
            memory_text=self.memory.render() if self.memory else "",
            run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
        )

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt: str | None = None,
    ) -> tuple[LegalAction, dict[str, Any]]:
        if not legal_actions:
            raise ValueError("No legal actions available")

        if self.conversation_mode == "turn_chat":
            return self._choose_turn_chat(state, legal_actions)

        prompt = prompt if prompt is not None else self.build_prompt(state, legal_actions)
        messages = [
            {
                "role": "system",
                "content": "You are a careful game-playing policy. Return only valid JSON. Never invent actions.",
            },
            {"role": "user", "content": prompt},
        ]
        if self.pending_external_feedback:
            messages.append({"role": "user", "content": "\n".join(self.pending_external_feedback)})
            self.pending_external_feedback.clear()

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = self._chat(messages)
                elapsed = time.time() - started
                text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(text)
                action, action_remap, resolved_from_label = _resolve_model_action(parsed, legal_actions)
                return action, {
                    "agent": "openai_compat",
                    "model": self.model,
                    "conversation_mode": self.conversation_mode,
                    "message_count": len(messages),
                    "conversation_prompt_chars": _messages_char_count(messages),
                    "attempt": attempt,
                    "elapsed_sec": elapsed,
                    "raw_response": text,
                    "parsed": parsed,
                    "usage": response.get("usage"),
                    "prompt_chars": len(prompt),
                    **_action_resolution_meta(action, action_remap, resolved_from_label),
                }
            except Exception as exc:  # noqa: BLE001 - log and retry with stricter correction
                last_error = str(exc)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid: {last_error}. "
                            f"Return only JSON with action_id between 0 and {len(legal_actions) - 1}, "
                            "selected_action_label copied exactly from that current legal action label, "
                            "sequence included only when choosing combat_custom_sequence, "
                            "and a reason describing the same selected action."
                        ),
                    }
                )

        if self.fallback_mode == "error":
            raise ModelActionError(_model_action_error_message(last_error, self.max_retries))

        # Optional deterministic fallback keeps exploratory benchmark batches
        # moving while still recording the invalid model behavior.
        fallback = first_non_view_action(legal_actions)
        return fallback, {
            "agent": "openai_compat",
            "model": self.model,
            "conversation_mode": self.conversation_mode,
            "message_count": len(messages),
            "conversation_prompt_chars": _messages_char_count(messages),
            "fallback": True,
            "error": last_error,
            "fallback_action_id": fallback.action_id,
            "prompt_chars": len(prompt),
        }

    def _choose_turn_chat(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
    ) -> tuple[LegalAction, dict[str, Any]]:
        messages = self.turn_chat_buffer.messages_for_decision(
            state,
            legal_actions,
            prompt_style=self.prompt_style,
            run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
            update_mode=self.turn_chat_update_mode,
            record_user=True,
        )
        turn_key = turn_chat_key(state)
        history_turns = self.turn_chat_buffer.history_turns()

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = self._chat(messages)
                elapsed = time.time() - started
                text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(text)
                action, action_remap, resolved_from_label = _resolve_model_action(parsed, legal_actions)
                self.turn_chat_buffer.remember_assistant(action, parsed)
                return action, {
                    "agent": "openai_compat",
                    "model": self.model,
                    "conversation_mode": self.conversation_mode,
                    "message_count": len(messages),
                    "turn_chat_history_turns": history_turns,
                    "turn_key": turn_key,
                    "conversation_prompt_chars": _messages_char_count(messages),
                    "prompt_chars": _messages_char_count(messages),
                    "attempt": attempt,
                    "elapsed_sec": elapsed,
                    "raw_response": text,
                    "parsed": parsed,
                    "usage": response.get("usage"),
                    **_action_resolution_meta(action, action_remap, resolved_from_label),
                }
            except Exception as exc:  # noqa: BLE001 - log and retry with stricter correction
                last_error = str(exc)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid: {last_error}. "
                            f"Return only JSON with action_id between 0 and {len(legal_actions) - 1}, "
                            "selected_action_label copied exactly from that current legal action label, "
                            "sequence included only when choosing combat_custom_sequence, "
                            "and a reason describing the same selected action."
                        ),
                    }
                )

        if self.fallback_mode == "error":
            raise ModelActionError(_model_action_error_message(last_error, self.max_retries))

        fallback = first_non_view_action(legal_actions)
        self.turn_chat_buffer.remember_assistant(
            fallback,
            {"reason": f"Fallback after invalid model response: {last_error}"},
        )
        return fallback, {
            "agent": "openai_compat",
            "model": self.model,
            "conversation_mode": self.conversation_mode,
            "message_count": len(messages),
            "turn_chat_history_turns": history_turns,
            "turn_key": turn_key,
            "conversation_prompt_chars": _messages_char_count(messages),
            "prompt_chars": _messages_char_count(messages),
            "fallback": True,
            "error": last_error,
            "fallback_action_id": fallback.action_id,
        }

    def reset_episode(self) -> None:
        if self.memory:
            self.memory.reset()
        self.turn_chat_buffer.reset()
        self.pending_external_feedback.clear()

    def receive_external_feedback(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        feedback: str,
        legal_actions: list[LegalAction],
    ) -> None:
        if self.conversation_mode == "turn_chat":
            self.turn_chat_buffer.remember_user_feedback(feedback)
            return
        self.pending_external_feedback.append(feedback)

    def record_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        if self.memory:
            self.memory.remember(state, action, meta, next_state)
        if self.conversation_mode == "turn_chat":
            self.turn_chat_buffer.after_transition(action, next_state)

    def _chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        base = self.base_url.rstrip("/")
        url = base + "/chat/completions"
        if not base.endswith("/v1"):
            url = base + "/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body[:500]}") from exc


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object even if the model wrapped it in fences."""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    value = _parse_first_json_object(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object, got {type(value).__name__}")
    return value


def _parse_first_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    try:
        value, _end = decoder.raw_decode(text)
        return value
    except json.JSONDecodeError as first_error:
        start = text.find("{")
        if start < 0:
            raise first_error
        value, _end = decoder.raw_decode(text[start:])
    return value


def _resolve_model_action(
    parsed: dict[str, Any],
    legal_actions: list[LegalAction],
) -> tuple[LegalAction, int | None, bool]:
    selected_label = _selected_action_label(parsed)
    label_action = _action_matching_selected_label(selected_label, legal_actions)
    raw_action_id = parsed.get("action_id")

    try:
        action_id = int(raw_action_id)
    except (TypeError, ValueError):
        if label_action is not None:
            _validate_model_action(parsed, label_action, legal_actions)
            return label_action, None, True
        raise ValueError("action_id must be an integer or selected_action_label must exactly match a current legal action")

    if action_id < 0 or action_id >= len(legal_actions):
        if label_action is not None:
            _validate_model_action(parsed, label_action, legal_actions)
            return label_action, action_id, True
        raise ValueError(f"action_id out of range: {action_id}")

    action = legal_actions[action_id]
    remapped_from: int | None = None
    if label_action is not None and label_action != action:
        remapped_from = action.action_id
        action = label_action

    _validate_model_action(parsed, action, legal_actions)
    return action, remapped_from, False


def _action_resolution_meta(
    action: LegalAction,
    remapped_from: int | None,
    resolved_from_label: bool,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if remapped_from is not None:
        meta["action_id_remapped_from"] = remapped_from
    if remapped_from is not None or resolved_from_label:
        meta["resolved_action_id"] = action.action_id
    if resolved_from_label:
        meta["action_id_resolved_from_label"] = True
    return meta


def _action_matching_selected_label(
    selected_label: str,
    legal_actions: list[LegalAction],
) -> LegalAction | None:
    if not selected_label:
        return None
    normalized_label = _normalize_for_action_match(selected_label)
    matches = [
        action
        for action in legal_actions
        if _normalize_for_action_match(action.label) == normalized_label
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _validate_model_action(parsed: dict[str, Any], action: LegalAction, legal_actions: list[LegalAction]) -> None:
    selected_label = _selected_action_label(parsed)
    if selected_label:
        if _normalize_for_action_match(selected_label) != _normalize_for_action_match(action.label):
            raise ValueError(
                "selected_action_label does not match selected action: "
                f"action_id {action.action_id} is {action.label!r}, got {selected_label!r}"
            )
        if action.kind != "combat_custom_sequence":
            return

    if action.kind == "combat_custom_sequence":
        normalize_card_sequence_payload(parsed.get("sequence"))
        return

    reason = _one_line(parsed.get("reason") if isinstance(parsed, dict) else "")
    mismatched_action = _reason_mentions_different_action(reason, action, legal_actions)
    if mismatched_action is not None:
        raise ValueError(
            "reason does not match selected action: "
            f"action_id {action.action_id} is {action.label!r}, "
            f"but reason appears to describe {mismatched_action.label!r}"
        )


def _selected_action_label(parsed: dict[str, Any]) -> str:
    for key in ("selected_action_label", "action_label", "label", "action"):
        value = parsed.get(key)
        if isinstance(value, str):
            return _one_line(value)
    return ""


def _reason_mentions_different_action(
    reason: str,
    selected_action: LegalAction,
    legal_actions: list[LegalAction],
) -> LegalAction | None:
    normalized_reason = _normalize_for_action_match(reason)
    if not normalized_reason:
        return None

    selected_phrases = _action_reason_phrases(selected_action)
    if any(_contains_positive_phrase(normalized_reason, phrase) for phrase in selected_phrases):
        return None

    for candidate in legal_actions:
        if candidate == selected_action or candidate.command.get("cmd") == "bench_view":
            continue
        for phrase in _action_reason_phrases(candidate):
            if _contains_positive_phrase(normalized_reason, phrase):
                return candidate
    return None


def _action_reason_phrases(action: LegalAction) -> set[str]:
    label = action.label
    normalized_label = _normalize_for_action_match(label)
    phrases = {normalized_label}

    if action.kind == "combat_end_turn":
        phrases.update({"end turn", "end the turn", "finish turn", "finish the turn"})

    if action.kind == "combat_card_sequence":
        phrases.add("play sequence")
        for part in label.split(":", 1)[-1].split("->"):
            normalized_part = _normalize_for_action_match(part)
            match = re.match(r"\d+\s+(.+?)(?:\s+on enemy\s+\d+)?$", normalized_part)
            if match:
                card_name = match.group(1)
                phrases.update({f"play {card_name}", f"play card {card_name}"})

    card_match = re.match(r"play card\s+\d+:\s*(.*?)(?:\s+on enemy\s+\d+:|$)", label, flags=re.IGNORECASE)
    if card_match:
        card_name = _normalize_for_action_match(card_match.group(1))
        phrases.update({f"play {card_name}", f"play card {card_name}"})

    for prefix, verbs in (
        (r"(?:pick|select) card\s+\d+:\s*(.*)$", ("pick", "select", "take")),
        (r"buy card\s+\d+:\s*(.*?)\s+for\s+\d+\s+gold$", ("buy", "purchase")),
        (r"use potion\s+\d+:\s*(.*?)(?:\s+on enemy\s+\d+:|$)", ("use", "drink")),
        (r"claim relic\s+\d+:\s*(.*)$", ("claim", "take")),
        (r"claim reward\s+\d+:\s*(.*)$", ("claim", "take")),
        (r"choose option\s+\d+:\s*(.*)$", ("choose", "select")),
    ):
        match = re.match(prefix, label, flags=re.IGNORECASE)
        if match:
            subject = _normalize_for_action_match(match.group(1))
            phrases.update(f"{verb} {subject}" for verb in verbs)

    return {phrase for phrase in phrases if len(phrase) >= 3}


def _contains_positive_phrase(normalized_text: str, phrase: str) -> bool:
    if not phrase:
        return False
    haystack = f" {normalized_text} "
    needle = f" {phrase} "
    start = haystack.find(needle)
    while start >= 0:
        prefix = haystack[max(0, start - 48) : start]
        if not _negates_action_phrase(prefix):
            return True
        start = haystack.find(needle, start + 1)
    return False


def _negates_action_phrase(prefix: str) -> bool:
    negation_markers = (
        "cannot ",
        "can t ",
        "cant ",
        "do not ",
        "don t ",
        "dont ",
        "not ",
        "no need to ",
        "avoid ",
        "skip ",
        "without ",
        "instead of ",
        "rather than ",
        "unable to ",
    )
    return any(marker in prefix for marker in negation_markers)


def _normalize_for_action_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def first_non_view_action(legal_actions: list[LegalAction]) -> LegalAction:
    return next(
        (action for action in legal_actions if action.command.get("cmd") != "bench_view"),
        legal_actions[0],
    )


def _model_action_error_message(last_error: str | None, max_retries: int) -> str:
    attempts = max_retries + 1
    reason = last_error or "unknown model action error"
    return f"LLM failed after {attempts} attempt(s): {reason}"
