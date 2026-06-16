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
    intent_label,
    name,
    power_label,
    render_state_text,
)


PromptStyle = Literal["default", "analysis"]
MemoryMode = Literal["action_reason", "factual_diff"]
ConversationMode = Literal["single_turn", "turn_chat"]
TurnChatUpdateMode = Literal["delta"]
TurnChatAssistantHistory = Literal["compact"]


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
        "action_id": action.action_id,
        "action": action.label,
    }
    if isinstance(parsed, dict):
        for key in ("turn_plan", "plan_update"):
            if key in parsed:
                payload[key] = parsed[key]
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
    lines.append(
        "Shown combat values are current engine previews. Use enemy Attack shown_total and hand "
        "shown_damage/shown_block exactly as printed; do not adjust them for visible Strength, "
        "Dexterity, Frail, Weak, or temporary powers."
    )

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
        lines.append("  " + card_line(card, preview_labels=True))

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
        intents.append(intent_label(intent))
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
        "Each legal action is one atomic command, not a full-turn plan. After playing a card, using a potion, choosing a reward, or viewing information, you will receive a fresh state and may act again if the game still allows actions.",
        "In combat, end turn is the action that intentionally finishes the current turn; do not choose it while useful playable cards or potions remain unless passing is strategically better.",
        "In combat, unused energy is not saved after ending the turn; next turn receives fresh energy from the engine.",
        "In combat, printed attack intent and hand damage/block are current engine values; do not recompute visible powers into them.",
        "If all visible enemy intents are non-attack intents, enemies are not making attack damage this turn; block usually expires at end of turn, so avoid spending energy only for block unless another effect justifies it.",
        "View deck/map/pile actions do not advance the game state. They are information requests logged separately; use them when deck composition, pile contents, path context, or current position could affect the decision.",
    ]
    turn_plan_rules = _turn_plan_header_rules() if turn_plan_enabled else []

    if prompt_style == "analysis":
        schema = (
            '{"situation":"<current objective and main risk>",'
            '"calculations":["<shown-value damage/block/energy/path/reward check if relevant>"],'
            '"candidates":[{"action_id":<integer>,"label":"<legal action label>",'
            '"pros":"<why it helps>","cons":"<main risk or cost>"}],'
            '"action_id":<integer>,"reason":"<final concise reason>"}'
        )
        if turn_plan_enabled:
            schema = (
                '{"turn_plan":{"objective":"<goal for this visible hand>",'
                '"survival_check":"<incoming damage and block/kill requirement>",'
                '"tentative_sequence":["<advisory card/order idea, not commands>"],'
                '"replan_if":["<state change that invalidates the plan>"]},'
                '"situation":"<current objective and main risk>",'
                '"calculations":["<shown-value damage/block/energy/path/reward check if relevant>"],'
                '"candidates":[{"action_id":<integer>,"label":"<legal action label>",'
                '"pros":"<why it helps>","cons":"<main risk or cost>"}],'
                '"action_id":<integer>,"reason":"<final concise reason>"}'
            )
        return [
            "You are playing Slay the Spire 2 through a headless benchmark environment.",
            "Choose exactly one legal action. Use only an action_id from the legal action list.",
            *shared,
            *turn_plan_rules,
            "Before choosing, write a compact public analysis in JSON. Analyze the current situation, do any needed arithmetic using printed current values, compare a few plausible legal actions, then choose.",
            "Return only valid JSON with this schema:",
            schema,
            "For combat, first check survival using enemy Attack shown_total, current block, HP, energy, and printed card shown_block/shown_damage. Treat shown_damage and shown_block as final current values: never reduce shown_block again for Frail/Dexterity, never add Strength/Weak/Vulnerable again to shown_damage, and never add enemy Strength again to shown_total.",
            "For map choices, compare path rewards and risks. For rewards, compare deck impact.",
        ]

    schema = '{"action_id": <integer>, "reason": "<short reason>"}'
    if turn_plan_enabled:
        schema = (
            '{"turn_plan": {"objective": "<goal for this visible hand>", '
            '"survival_check": "<incoming damage and block/kill requirement>", '
            '"tentative_sequence": ["<advisory card/order idea, not commands>"], '
            '"replan_if": ["<state change that invalidates the plan>"]}, '
            '"action_id": <integer>, "reason": "<short reason>"}'
        )
    return [
        "You are playing Slay the Spire 2 through a headless benchmark environment.",
        "Choose exactly one legal action. Return only JSON with this schema:",
        *shared,
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
            "block needs, energy, card order, draw/random effects, modal risks, and conditions that require replanning, "
            "using printed current combat values without recomputing visible powers into them."
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
    turn_chat_buffer: TurnChatBuffer = field(init=False)

    def __post_init__(self) -> None:
        if self.conversation_mode not in {"single_turn", "turn_chat"}:
            raise ValueError(f"Unknown conversation mode: {self.conversation_mode}")
        if self.turn_chat_update_mode != "delta":
            raise ValueError(f"Unsupported turn_chat_update_mode: {self.turn_chat_update_mode}")
        if self.turn_chat_assistant_history != "compact":
            raise ValueError(f"Unsupported turn_chat_assistant_history: {self.turn_chat_assistant_history}")
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

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = self._chat(messages)
                elapsed = time.time() - started
                text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(text)
                action_id = int(parsed.get("action_id"))
                if action_id < 0 or action_id >= len(legal_actions):
                    raise ValueError(f"action_id out of range: {action_id}")
                return legal_actions[action_id], {
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
                }
            except Exception as exc:  # noqa: BLE001 - log and retry with stricter correction
                last_error = str(exc)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid: {last_error}. "
                            f"Return only JSON with action_id between 0 and {len(legal_actions) - 1}."
                        ),
                    }
                )

        # Deterministic fallback keeps long benchmark batches moving while
        # still recording the invalid model behavior.  Prefer a real game
        # action so parse failures do not degenerate into repeated view actions.
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
                action_id = int(parsed.get("action_id"))
                if action_id < 0 or action_id >= len(legal_actions):
                    raise ValueError(f"action_id out of range: {action_id}")
                action = legal_actions[action_id]
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
                }
            except Exception as exc:  # noqa: BLE001 - log and retry with stricter correction
                last_error = str(exc)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid: {last_error}. "
                            f"Return only JSON with action_id between 0 and {len(legal_actions) - 1}."
                        ),
                    }
                )

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
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object, got {type(value).__name__}")
    return value


def first_non_view_action(legal_actions: list[LegalAction]) -> LegalAction:
    return next(
        (action for action in legal_actions if action.command.get("cmd") != "bench_view"),
        legal_actions[0],
    )
