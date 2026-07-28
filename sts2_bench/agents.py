"""Policy agents used by benchmark runners."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .actions import (
    LegalAction,
    card_select_combination_count,
    materialize_card_select_space_action,
)
from .context import (
    append_last_action_result,
    append_viewed_information,
    build_last_action_result,
    combat_skill_hints,
    compact_deck,
    compact_state,
    dedupe_hover_tips,
    name,
    pile_cards_from_state,
    render_state_text,
)
from .game_rules import full_game_rules_meta, game_rules_meta, render_full_game_rules, render_game_rules
from .route_graph import build_canonical_route_map, build_route_graph
from .route_plan import winged_boots_remaining_for_state
from .sequences import MAX_CUSTOM_SEQUENCE_VALIDATION_LENGTH, normalize_card_sequence_payload
from .sequence_safety import sequence_card_block_reason


PromptStyle = Literal["default", "analysis"]
ShortTermMemoryMode = Literal["action_reason", "factual_diff"]
ConversationMode = Literal[
    "single_turn",
    "turn_chat",
    "combat_mdp_state",
    "combat_state_only",
    "combat_dialogue",
]
PromptProfile = Literal["default", "formal_generalist"]
TurnChatUpdateMode = Literal["delta"]
TurnChatAssistantHistory = Literal["compact"]
FallbackMode = Literal["action", "error"]
SelfConsistencyMode = Literal["diagnostic", "majority_vote"]
ActionPromptOrderMode = Literal["default", "shuffle_map"]
RouteMapPromptFormat = Literal["route_graph", "canonical"]

FORMAL_PROVIDER_AUDIT_ENV = "STS2_BENCH_FORMAL_PROVIDER_AUDIT"
_PROVIDER_METADATA_KEYS = ("model", "id", "created", "system_fingerprint", "service_tier")
MapRouteHint = Literal["none", "max_restsites"]
CombatSkillHintsProfile = Literal["baseline", "survival", "targeting", "full"]
HttpBackend = Literal["urllib", "requests_session", "requests_session_worker"]
SELF_CONSISTENCY_MAJORITY_ALLOWED_DECISIONS = {"map_select", "card_reward"}
TURN_CHAT_CURRENT_STATE_NOTICE = (
    "Only the latest Current turn state and Legal actions are valid; earlier hands, card_index values, "
    "selected_card_indices, action_id values, powers, potions, character mechanics, and legal actions in "
    "this conversation are stale after each action."
)


class ModelActionError(RuntimeError):
    """Raised when an LLM cannot produce a valid legal action after retries."""

    def __init__(self, message: str, *, audit_meta: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.audit_meta = json.loads(json.dumps(audit_meta or {}, ensure_ascii=False))


def _requests_session_worker(connection: Any) -> None:
    """Own one persistent Session behind a parent-enforced hard deadline."""

    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util import Timeout as Urllib3Timeout
    except ImportError as exc:  # pragma: no cover - exercised only in an incomplete environment
        connection.send({"status": "worker_error", "message": str(exc)})
        connection.close()
        return
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, pool_block=True, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        while True:
            command = connection.recv()
            if command.get("type") == "close":
                connection.send({"status": "closed"})
                return
            try:
                with session.post(
                    command["url"],
                    data=command["body"],
                    headers=command["headers"],
                    timeout=Urllib3Timeout(total=float(command["timeout"])),
                ) as response:
                    response_body = response.content
                    if response.status_code >= 400:
                        connection.send(
                            {
                                "status": "http_error",
                                "code": response.status_code,
                                # Preserve the complete byte payload.  The parent
                                # process owns the canonical UTF-8 decode, hash,
                                # and formal provider audit record; truncating at
                                # this boundary would make HTTP failures
                                # unauditable.
                                "response_body": response_body,
                            }
                        )
                        continue
                    try:
                        decoded = json.loads(response_body.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        connection.send(
                            {
                                "status": "json_error",
                                "code": response.status_code,
                                "message": exc.msg,
                                "document": exc.doc,
                                "position": exc.pos,
                                "response_body": response_body,
                            }
                        )
                        continue
                    connection.send(
                        {
                            "status": "ok",
                            "code": response.status_code,
                            "response": decoded,
                            "response_body": response_body,
                        }
                    )
            except requests.exceptions.RequestException as exc:
                connection.send({"status": "request_error", "message": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                connection.send(
                    {
                        "status": "worker_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        session.close()
        connection.close()


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
    """Small per-episode action history; never a cross-run experience store."""

    max_entries: int = 8
    mode: ShortTermMemoryMode = "action_reason"
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


def _route_plan_contract_enabled(state: dict[str, Any], route_plan_enabled: bool) -> bool:
    return bool(route_plan_enabled and state.get("decision") == "map_select")


def _combat_sequence_contract_enabled(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    combat_card_sequences_enabled: bool,
) -> bool:
    return bool(
        combat_card_sequences_enabled
        and state.get("decision") == "combat_play"
        and any(action.kind == "combat_custom_sequence" for action in legal_actions)
    )


@dataclass(frozen=True)
class PromptContract:
    """Decision-scoped response contract exposed to the acting model."""

    decision_type: str = ""
    route_plan_enabled: bool = False
    combat_custom_sequence_enabled: bool = False
    card_select_indices_enabled: bool = False
    context_patch_fields_enabled: bool = True
    evidence_context_enabled: bool = True
    interface_preamble_enabled: bool = True
    turn_intent_enabled: bool = False
    turn_check_enabled: bool = False


def build_prompt_contract(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    route_plan_enabled: bool = True,
    context_patch_fields_enabled: bool = True,
    evidence_context_enabled: bool = True,
    interface_preamble_enabled: bool = True,
    combat_card_sequences_enabled: bool = True,
    turn_intent_enabled: bool = False,
    turn_check_enabled: bool = False,
) -> PromptContract:
    """Resolve global feature switches to the contract valid for this state."""

    return PromptContract(
        decision_type=str(state.get("decision") or state.get("type") or ""),
        route_plan_enabled=_route_plan_contract_enabled(state, route_plan_enabled),
        combat_custom_sequence_enabled=_combat_sequence_contract_enabled(
            state,
            legal_actions,
            combat_card_sequences_enabled,
        ),
        card_select_indices_enabled=bool(
            state.get("decision") == "card_select"
            and any(action.kind in {"card_select", "card_select_space", "card_select_skip"} for action in legal_actions)
        ),
        context_patch_fields_enabled=bool(
            context_patch_fields_enabled
            and state.get("decision") != "combat_play"
        ),
        evidence_context_enabled=bool(evidence_context_enabled),
        interface_preamble_enabled=bool(interface_preamble_enabled),
        turn_intent_enabled=bool(turn_intent_enabled and state.get("decision") == "combat_play"),
        turn_check_enabled=bool(turn_check_enabled and state.get("decision") == "combat_play"),
    )


@dataclass
class TurnChatBuffer:
    """Current-player-turn chat history for an LLM agent.

    This is an agent-side context-management buffer only.  It does not change
    the benchmark action surface or the headless environment command protocol.
    """

    max_turns: int
    plan_enabled: bool = False
    route_plan_enabled: bool = True
    route_map_prompt_format: RouteMapPromptFormat = "canonical"
    map_route_hint: MapRouteHint = "none"
    context_patch_fields_enabled: bool = True
    combat_card_sequences_enabled: bool = True
    combat_skill_lines_enabled: bool = True
    rule_projection_and_recent_changes_enabled: bool = True
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
            "content": build_turn_chat_system_message(
                prompt_style,
                state=state,
                contract=build_prompt_contract(
                    state,
                    legal_actions,
                    route_plan_enabled=False,
                    context_patch_fields_enabled=self.context_patch_fields_enabled,
                    combat_card_sequences_enabled=self.combat_card_sequences_enabled,
                    turn_check_enabled=self.plan_enabled,
                ),
                rule_projection_enabled=self.rule_projection_and_recent_changes_enabled,
            ),
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
                route_plan_enabled=self.route_plan_enabled,
                route_map_prompt_format=self.route_map_prompt_format,
                map_route_hint=self.map_route_hint,
                context_patch_fields_enabled=self.context_patch_fields_enabled,
                combat_card_sequences_enabled=self.combat_card_sequences_enabled,
                combat_skill_lines_enabled=self.combat_skill_lines_enabled,
                rule_projection_and_recent_changes_enabled=self.rule_projection_and_recent_changes_enabled,
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
                route_map_prompt_format=self.route_map_prompt_format,
                combat_skill_lines_enabled=self.combat_skill_lines_enabled,
                recent_state_changes_enabled=self.rule_projection_and_recent_changes_enabled,
            )
        if update_mode != "delta":
            raise ValueError(f"Unsupported turn_chat update mode: {update_mode}")
        if _viewed_labels(state):
            return build_view_update_user_message(
                state,
                legal_actions,
                route_map_prompt_format=self.route_map_prompt_format,
                combat_skill_lines_enabled=self.combat_skill_lines_enabled,
                recent_state_changes_enabled=self.rule_projection_and_recent_changes_enabled,
            )
        return build_turn_update_user_message(
            state,
            legal_actions,
            route_map_prompt_format=self.route_map_prompt_format,
            combat_skill_lines_enabled=self.combat_skill_lines_enabled,
            recent_state_changes_enabled=self.rule_projection_and_recent_changes_enabled,
        )


@dataclass
class CombatDialogueBuffer:
    """Raw accepted combat dialogue for the active turn and recent completed turns."""

    completed_turns_to_keep: int = 3
    reasoning_pool_enabled: bool = False
    skill_hints_enabled: bool = False
    skill_hints_profile: CombatSkillHintsProfile = "baseline"
    skill_hints_max_hints: int = 4
    rule_projection_and_recent_changes_enabled: bool = True
    active_combat_key: tuple[Any, ...] | None = None
    active_turn_key: tuple[Any, ...] | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.active_combat_key = None
        self.active_turn_key = None
        self.turns.clear()

    def _activate(self, state: dict[str, Any]) -> tuple[Any, ...] | None:
        turn_key = turn_chat_key(state)
        if turn_key is None:
            self.reset()
            return None
        combat_key = turn_key[:-1]
        if combat_key != self.active_combat_key:
            self.reset()
            self.active_combat_key = combat_key
        if turn_key != self.active_turn_key:
            self.active_turn_key = turn_key
            self.turns.append({"turn_key": turn_key, "messages": []})
            self._trim()
        return turn_key

    def messages_for_decision(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt_style: PromptStyle,
        run_summary_text: str,
        turn_check_enabled: bool,
        route_plan_enabled: bool,
        context_patch_fields_enabled: bool,
        combat_card_sequences_enabled: bool,
        record_user: bool,
    ) -> list[dict[str, str]]:
        turn_key = self._activate(state)
        if turn_key is None:
            prompt = build_llm_prompt(
                state,
                legal_actions,
                include_json=False,
                prompt_style=prompt_style,
                memory_text="",
                run_summary_text=run_summary_text,
                route_plan_enabled=route_plan_enabled,
                context_patch_fields_enabled=context_patch_fields_enabled,
                combat_card_sequences_enabled=combat_card_sequences_enabled,
                rule_projection_and_recent_changes_enabled=(
                    self.rule_projection_and_recent_changes_enabled
                ),
            )
            return [
                {"role": "system", "content": _default_policy_system_message()},
                {"role": "user", "content": prompt},
            ]

        user_message = build_combat_current_state_user_message(
            state,
            legal_actions,
            prompt_style=prompt_style,
            run_summary_text=run_summary_text,
            turn_intent_enabled=self.reasoning_pool_enabled,
            turn_check_enabled=turn_check_enabled,
            route_plan_enabled=route_plan_enabled,
            context_patch_fields_enabled=context_patch_fields_enabled,
            combat_card_sequences_enabled=combat_card_sequences_enabled,
            skill_hints_enabled=self.skill_hints_enabled,
            skill_hints_profile=self.skill_hints_profile,
            skill_hints_max_hints=self.skill_hints_max_hints,
            rule_projection_enabled=self.rule_projection_and_recent_changes_enabled,
        )
        history = [
            dict(message)
            for turn in self.turns
            for message in turn.get("messages", [])
            if isinstance(message, dict)
        ]
        if record_user:
            self.turns[-1]["messages"].append({"role": "user", "content": user_message})
        else:
            history.append({"role": "user", "content": user_message})
        if record_user:
            history = [
                dict(message)
                for turn in self.turns
                for message in turn.get("messages", [])
                if isinstance(message, dict)
            ]
        return [
            {"role": "system", "content": _default_policy_system_message()},
            *history,
        ]

    def remember_assistant_raw(self, raw_response: str) -> None:
        if self.active_turn_key is None or not self.turns:
            return
        self.turns[-1]["messages"].append({"role": "assistant", "content": raw_response})

    def remember_accepted_exchange(
        self,
        state: dict[str, Any],
        user_message: str,
        raw_response: str,
    ) -> None:
        if self._activate(state) is None or not self.turns:
            return
        self.turns[-1]["messages"].extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": raw_response},
            ]
        )

    def history_audit(self) -> dict[str, Any]:
        messages = [
            message
            for turn in self.turns
            for message in turn.get("messages", [])
            if isinstance(message, dict)
        ]
        completed = self.turns[:-1] if self.turns else []
        chars = _messages_char_count(messages)
        return {
            "history_message_count": len(messages),
            "history_completed_turns": len(completed),
            "history_rounds": [
                (turn.get("turn_key") or (None,))[-1]
                for turn in self.turns
            ],
            "history_chars": chars,
            "history_token_estimate": (chars + 3) // 4,
            "history_token_usage": {
                "kind": "estimated_chars_div_4",
                "tokens": (chars + 3) // 4,
            },
            "assistant_history": "raw",
        }

    def after_transition(self, action: LegalAction, next_state: dict[str, Any] | None) -> None:
        if not isinstance(next_state, dict):
            self.reset()
            return
        next_turn_key = turn_chat_key(next_state)
        if next_turn_key is None:
            self.reset()
            return
        if self.active_combat_key is not None and next_turn_key[:-1] != self.active_combat_key:
            self.reset()

    def _trim(self) -> None:
        maximum = max(0, self.completed_turns_to_keep) + 1
        if len(self.turns) > maximum:
            del self.turns[: len(self.turns) - maximum]


@dataclass
class CombatMdpStateBuffer:
    """Structured same-turn facts plus rolling completed-turn memory without chat replay."""

    reasoning_pool_enabled: bool = False
    turn_board_enabled: bool = False
    skill_hints_enabled: bool = False
    skill_hints_profile: CombatSkillHintsProfile = "baseline"
    skill_hints_max_hints: int = 4
    route_map_prompt_format: RouteMapPromptFormat = "canonical"
    map_route_hint: MapRouteHint = "none"
    rule_projection_and_recent_changes_enabled: bool = True
    turn_board_max_entries: int = 3
    active_combat_key: tuple[Any, ...] | None = None
    active_turn_key: tuple[Any, ...] | None = None
    turn_start_snapshot: dict[str, Any] | None = None
    actions_this_turn: list[dict[str, Any]] = field(default_factory=list)
    current_turn_intent: dict[str, Any] | None = None
    turn_board: list[dict[str, Any]] = field(default_factory=list)
    combat_summary: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.active_combat_key = None
        self.active_turn_key = None
        self._reset_current_turn()
        self.turn_board.clear()
        self.combat_summary.clear()

    def _reset_current_turn(self) -> None:
        self.turn_start_snapshot = None
        self.actions_this_turn.clear()
        self.current_turn_intent = None

    def _start_current_turn(self, state: dict[str, Any]) -> None:
        self._reset_current_turn()
        if self.turn_board_enabled:
            self.turn_start_snapshot = _combat_memory_snapshot(state)

    def actions_count(self, state: dict[str, Any]) -> int:
        turn_key = turn_chat_key(state)
        if turn_key is None or turn_key != self.active_turn_key:
            return 0
        return len(self.actions_this_turn)

    def history_audit(self, state: dict[str, Any]) -> dict[str, Any]:
        process = _combat_turn_process_for_prompt(
            state,
            self.actions_this_turn,
            self.current_turn_intent if self.reasoning_pool_enabled else None,
            self.turn_board if self.turn_board_enabled else [],
            self.combat_summary if self.turn_board_enabled else {},
        )
        has_history = bool(
            self.actions_this_turn
            or self.current_turn_intent
            or self.turn_board
            or self.combat_summary
        )
        rendered = (
            json.dumps(process, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if has_history
            else ""
        )
        summarized_turns = int(self.combat_summary.get("completed_turns") or 0)
        completed_turns = len(self.turn_board) + summarized_turns
        chars = len(rendered)
        return {
            "history_message_count": 0,
            "history_completed_turns": completed_turns,
            "history_rounds": [
                entry.get("round")
                for entry in self.turn_board
                if isinstance(entry, dict)
            ],
            "history_chars": chars,
            "history_token_estimate": (chars + 3) // 4,
            "history_token_usage": {
                "kind": "estimated_chars_div_4",
                "tokens": (chars + 3) // 4,
            },
            "assistant_history": "managed",
        }

    def build_prompt(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt_style: PromptStyle,
        run_summary_text: str,
        turn_check_enabled: bool = False,
        route_plan_enabled: bool = True,
        context_patch_fields_enabled: bool = True,
        combat_card_sequences_enabled: bool = True,
    ) -> str:
        turn_key = turn_chat_key(state)
        if turn_key is None:
            self.reset()
            return build_llm_prompt(
                state,
                legal_actions,
                include_json=False,
                prompt_style=prompt_style,
                memory_text="",
                run_summary_text=run_summary_text,
                route_plan_enabled=route_plan_enabled,
                route_map_prompt_format=self.route_map_prompt_format,
                map_route_hint=self.map_route_hint,
                context_patch_fields_enabled=context_patch_fields_enabled,
                combat_card_sequences_enabled=combat_card_sequences_enabled,
                rule_projection_and_recent_changes_enabled=self.rule_projection_and_recent_changes_enabled,
            )
        combat_key = turn_key[:-1]
        if combat_key != self.active_combat_key:
            self.active_combat_key = combat_key
            self.active_turn_key = None
            self._reset_current_turn()
            self.turn_board.clear()
            self.combat_summary.clear()
        if turn_key != self.active_turn_key:
            self.active_turn_key = turn_key
            self._start_current_turn(state)
        return build_combat_mdp_user_message(
            state,
            legal_actions,
            actions_this_turn=self.actions_this_turn,
            current_turn_intent=(
                self.current_turn_intent if self.reasoning_pool_enabled else None
            ),
            turn_board=self.turn_board if self.turn_board_enabled else [],
            combat_summary=self.combat_summary if self.turn_board_enabled else {},
            prompt_style=prompt_style,
            run_summary_text=run_summary_text,
            turn_intent_enabled=self.reasoning_pool_enabled,
            turn_check_enabled=turn_check_enabled,
            route_plan_enabled=route_plan_enabled,
            context_patch_fields_enabled=context_patch_fields_enabled,
            combat_card_sequences_enabled=combat_card_sequences_enabled,
            skill_hints_enabled=self.skill_hints_enabled,
            skill_hints_profile=self.skill_hints_profile,
            skill_hints_max_hints=self.skill_hints_max_hints,
            rule_projection_enabled=self.rule_projection_and_recent_changes_enabled,
        )

    def record_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None,
    ) -> None:
        turn_key = turn_chat_key(state)
        if turn_key is None:
            self.reset()
            return
        combat_key = turn_key[:-1]
        if combat_key != self.active_combat_key:
            self.active_combat_key = combat_key
            self.active_turn_key = None
            self._reset_current_turn()
            self.turn_board.clear()
            self.combat_summary.clear()
        if self.active_turn_key != turn_key:
            self.active_turn_key = turn_key
            self._start_current_turn(state)

        if action.kind == "combat_end_turn":
            next_turn_key = turn_chat_key(next_state) if isinstance(next_state, dict) else None
            if next_turn_key is not None and next_turn_key[:-1] == combat_key:
                if self.turn_board_enabled:
                    turn_entry = _combat_turn_board_entry(
                        self.turn_start_snapshot or _combat_memory_snapshot(state),
                        state,
                        next_state,
                        self.actions_this_turn,
                    )
                    self.turn_board.append(turn_entry)
                    while len(self.turn_board) > self.turn_board_max_entries:
                        _fold_combat_turn_summary(self.combat_summary, self.turn_board.pop(0))
                self.active_turn_key = None
                self._reset_current_turn()
            else:
                self.reset()
            return

        if not isinstance(next_state, dict):
            return
        next_turn_key = turn_chat_key(next_state)
        if next_turn_key != turn_key:
            self.reset()
            return

        result = next_state.get("last_action_result")
        if not isinstance(result, dict):
            result = build_last_action_result(
                state,
                next_state,
                action_label=action.label,
                action_kind=action.kind,
            )
        self.actions_this_turn.append(_combat_mdp_action_fact(state, action, result))
        if self.reasoning_pool_enabled:
            next_intent = _combat_mdp_turn_intent_fact(action, meta)
            if next_intent is not None:
                self.current_turn_intent = next_intent


def build_llm_prompt(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    include_json: bool = True,
    prompt_style: PromptStyle = "default",
    memory_text: str = "",
    run_summary_text: str = "",
    route_plan_enabled: bool = True,
    route_map_prompt_format: RouteMapPromptFormat = "canonical",
    map_route_hint: MapRouteHint = "none",
    context_patch_fields_enabled: bool = True,
    combat_card_sequences_enabled: bool = True,
    combat_skill_lines_enabled: bool = True,
    rule_projection_and_recent_changes_enabled: bool = True,
) -> str:
    """Build the action-selection prompt used by LLM policies.

    ``context.py`` owns only state rendering.  The policy-facing task
    instruction, response schema, and legal-action wrapper live with agents so
    different agents can define different prompting methods over the same
    state text.
    """

    if prompt_style not in {"default", "analysis"}:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    contract = build_prompt_contract(
        state,
        legal_actions,
        route_plan_enabled=route_plan_enabled,
        context_patch_fields_enabled=context_patch_fields_enabled,
        combat_card_sequences_enabled=combat_card_sequences_enabled,
    )
    parts = _prompt_header(
        prompt_style,
        contract=contract,
    )
    parts.extend(_decision_protocol_prompt_lines(state))
    parts.extend(
        _game_rules_prompt_lines(
            state,
            enabled=rule_projection_and_recent_changes_enabled,
        )
    )
    if run_summary_text:
        parts.extend(["", "Run summary:", run_summary_text])
    if memory_text:
        parts.extend(["", "Episode memory:", memory_text])
    if state.get("decision") == "map_select":
        parts.extend(
            [
                "",
                "Route decision state JSON:",
                json.dumps(
                    _route_decision_state_for_prompt(
                        state,
                        recent_state_changes_enabled=rule_projection_and_recent_changes_enabled,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
        parts.extend(
            [
                "",
                "Legal actions:",
                _legal_actions_text(legal_actions, state=state),
            ]
        )
    else:
        parts.extend(
            [
                "",
                "Game state:",
                render_state_text(
                    state,
                    combat_skill_lines_enabled=combat_skill_lines_enabled,
                    recent_state_changes_enabled=rule_projection_and_recent_changes_enabled,
                ),
                "",
                "Legal actions:",
                _legal_actions_text(legal_actions, state=state),
            ]
        )
    parts.extend(
        _route_graph_prompt_lines(
            state,
            legal_actions,
            route_map_prompt_format=route_map_prompt_format,
            route_plan_enabled=route_plan_enabled,
            recent_state_changes_enabled=rule_projection_and_recent_changes_enabled,
        )
    )
    parts.extend(_map_route_hint_lines(state, map_route_hint))
    if include_json and state.get("decision") != "map_select":
        parts.extend(
            [
                "",
                "Compact state JSON:",
                json.dumps(
                    _compact_state_for_prompt(
                        state,
                        recent_state_changes_enabled=rule_projection_and_recent_changes_enabled,
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ]
        )
    return "\n".join(parts)


def build_turn_chat_system_message(
    prompt_style: PromptStyle,
    *,
    state: dict[str, Any],
    contract: PromptContract,
    rule_projection_enabled: bool = True,
) -> str:
    parts = _prompt_header(prompt_style, contract=contract)
    parts.extend(_decision_protocol_prompt_lines(state))
    parts.extend(_game_rules_prompt_lines(state, enabled=rule_projection_enabled))
    return "\n".join(parts)


def build_turn_initial_user_message(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    run_summary_text: str = "",
    *,
    route_map_prompt_format: RouteMapPromptFormat = "canonical",
    combat_skill_lines_enabled: bool = True,
    recent_state_changes_enabled: bool = True,
) -> str:
    parts = ["New player turn."]
    if run_summary_text:
        parts.extend(["", "Run summary:", run_summary_text])
    parts.extend(
        [
            "",
            "Game state:",
            render_state_text(
                state,
                combat_skill_lines_enabled=combat_skill_lines_enabled,
                recent_state_changes_enabled=recent_state_changes_enabled,
            ),
            "",
            "Legal actions:",
            _legal_actions_text(legal_actions, state=state),
        ]
    )
    parts.extend(
        _route_graph_prompt_lines(
            state,
            legal_actions,
            route_map_prompt_format=route_map_prompt_format,
            recent_state_changes_enabled=recent_state_changes_enabled,
        )
    )
    return "\n".join(parts)


def build_turn_update_user_message(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    route_map_prompt_format: RouteMapPromptFormat = "canonical",
    combat_skill_lines_enabled: bool = True,
    recent_state_changes_enabled: bool = True,
) -> str:
    if state.get("decision") == "card_select":
        return "\n".join(
            [
                "Turn modal.",
                TURN_CHAT_CURRENT_STATE_NOTICE,
                "",
                "Current turn state:",
                render_state_text(
                    state,
                    combat_skill_lines_enabled=combat_skill_lines_enabled,
                    recent_state_changes_enabled=recent_state_changes_enabled,
                ),
                "",
                "Legal actions:",
                _legal_actions_text(legal_actions, state=state),
            ]
        )

    lines = [
        "Turn update.",
        TURN_CHAT_CURRENT_STATE_NOTICE,
    ]
    last_action_lines: list[str] = []
    if recent_state_changes_enabled:
        append_last_action_result(last_action_lines, state.get("last_action_result"))
    if last_action_lines:
        lines.extend(last_action_lines)
    lines.extend(
        [
            "",
            "Current turn state:",
            _render_current_turn_state(
                state,
                combat_skill_lines_enabled=combat_skill_lines_enabled,
                recent_state_changes_enabled=recent_state_changes_enabled,
            ),
            "",
            "Legal actions:",
            _legal_actions_text(legal_actions, state=state),
        ]
    )
    lines.extend(
        _route_graph_prompt_lines(
            state,
            legal_actions,
            route_map_prompt_format=route_map_prompt_format,
            recent_state_changes_enabled=recent_state_changes_enabled,
        )
    )
    return "\n".join(lines)


def build_combat_mdp_user_message(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    actions_this_turn: list[dict[str, Any]],
    current_turn_intent: dict[str, Any] | None = None,
    turn_board: list[dict[str, Any]] | None = None,
    combat_summary: dict[str, Any] | None = None,
    prompt_style: PromptStyle = "default",
    run_summary_text: str = "",
    turn_intent_enabled: bool = False,
    turn_check_enabled: bool = False,
    route_plan_enabled: bool = True,
    context_patch_fields_enabled: bool = True,
    combat_card_sequences_enabled: bool = True,
    skill_hints_enabled: bool = False,
    skill_hints_profile: CombatSkillHintsProfile = "baseline",
    skill_hints_max_hints: int = 4,
    rule_projection_enabled: bool = True,
) -> str:
    contract = build_prompt_contract(
        state,
        legal_actions,
        route_plan_enabled=route_plan_enabled,
        context_patch_fields_enabled=context_patch_fields_enabled,
        combat_card_sequences_enabled=combat_card_sequences_enabled,
        turn_intent_enabled=turn_intent_enabled,
        turn_check_enabled=turn_check_enabled,
    )
    parts = _prompt_header(
        prompt_style,
        contract=contract,
    )
    parts.extend(_decision_protocol_prompt_lines(state))
    parts.extend(["", "Combat MDP decision."])
    if run_summary_text:
        parts.extend(["", "Run summary:", run_summary_text])
    parts.extend(
        [
            "",
            "Combat MDP state JSON:",
            json.dumps(
                _combat_mdp_state_for_prompt(
                    state,
                    skill_hints_enabled=skill_hints_enabled,
                    skill_hints_profile=skill_hints_profile,
                    skill_hints_max_hints=skill_hints_max_hints,
                    sequence_metadata_enabled=combat_card_sequences_enabled,
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    if actions_this_turn or current_turn_intent or turn_board or combat_summary:
        parts.extend(
            [
                "",
                "Combat turn process JSON:",
                json.dumps(
                    _combat_turn_process_for_prompt(
                        state,
                        actions_this_turn,
                        current_turn_intent,
                        turn_board or [],
                        combat_summary or {},
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    # Move the single authoritative rule rendering next to the action choice.
    # This improves decision-time salience without maintaining a duplicate rule
    # capsule.
    parts.extend(_game_rules_prompt_lines(state, enabled=rule_projection_enabled))
    parts.extend(
        [
            "",
            "Legal actions:",
            _legal_actions_text(legal_actions, state=state),
        ]
    )
    return "\n".join(parts)


def build_combat_current_state_user_message(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    prompt_style: PromptStyle = "default",
    run_summary_text: str = "",
    turn_intent_enabled: bool = False,
    turn_check_enabled: bool = False,
    route_plan_enabled: bool = True,
    context_patch_fields_enabled: bool = True,
    combat_card_sequences_enabled: bool = True,
    skill_hints_enabled: bool = False,
    skill_hints_profile: CombatSkillHintsProfile = "baseline",
    skill_hints_max_hints: int = 4,
    rule_projection_enabled: bool = True,
) -> str:
    """Render the managed control's current-state portion without any history."""

    return build_combat_mdp_user_message(
        state,
        legal_actions,
        actions_this_turn=[],
        current_turn_intent=None,
        turn_board=[],
        combat_summary={},
        prompt_style=prompt_style,
        run_summary_text=run_summary_text,
        turn_intent_enabled=turn_intent_enabled,
        turn_check_enabled=turn_check_enabled,
        route_plan_enabled=route_plan_enabled,
        context_patch_fields_enabled=context_patch_fields_enabled,
        combat_card_sequences_enabled=combat_card_sequences_enabled,
        skill_hints_enabled=skill_hints_enabled,
        skill_hints_profile=skill_hints_profile,
        skill_hints_max_hints=skill_hints_max_hints,
        rule_projection_enabled=rule_projection_enabled,
    )


def _default_policy_system_message() -> str:
    return "You are a careful game-playing policy. Return only valid JSON. Never invent actions."


def build_formal_generalist_system_message() -> str:
    return "\n\n".join(
        [
            (
                "Choose one legal action from the current authoritative state. "
                "Return only valid JSON and never invent an action."
            ),
            render_full_game_rules(),
        ]
    )


def build_formal_generalist_user_message(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    prompt_style: PromptStyle = "default",
    combat_card_sequences_enabled: bool = True,
) -> str:
    contract = build_prompt_contract(
        state,
        legal_actions,
        route_plan_enabled=False,
        context_patch_fields_enabled=False,
        evidence_context_enabled=False,
        interface_preamble_enabled=False,
        combat_card_sequences_enabled=combat_card_sequences_enabled,
        turn_intent_enabled=False,
        turn_check_enabled=False,
    )
    parts = _prompt_header(prompt_style, contract=contract)
    parts.extend(_decision_protocol_prompt_lines(state))
    decision = state.get("decision")
    if turn_chat_key(state) is not None:
        label = "Combat MDP state JSON:"
        payload = _combat_mdp_state_for_prompt(
            state,
            skill_hints_enabled=False,
            sequence_metadata_enabled=combat_card_sequences_enabled,
        )
        if decision == "card_select":
            selection_context = payload.get("selection_context")
            if isinstance(selection_context, dict):
                selection_context.pop("candidates", None)
    elif decision == "map_select":
        label = "Route decision state JSON:"
        payload = _route_decision_state_for_prompt(
            state,
            recent_state_changes_enabled=False,
        )
    else:
        label = "Decision state JSON:"
        payload = _formal_generalist_state_for_prompt(state)
    parts.extend(
        [
            "",
            label,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "",
            "Legal actions:",
            _legal_actions_text(legal_actions, state=state),
        ]
    )
    if decision == "map_select":
        parts.extend(
            _route_graph_prompt_lines(
                state,
                legal_actions,
                route_map_prompt_format="canonical",
                route_plan_enabled=False,
                recent_state_changes_enabled=False,
            )
        )
    return "\n".join(parts)


_FORMAL_GENERALIST_OMITTED_STATE_KEYS = {
    "active_route_plan",
    "context_patch",
    "evidence_context",
    "last_action_result",
    "recent_state_changes",
    "route_plan",
    "route_process",
    "run_context",
    "run_summary",
    "short_term_memory",
}


def _formal_generalist_state_for_prompt(state: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _combat_mdp_visible_value(value)
        for key, value in state.items()
        if key not in _FORMAL_GENERALIST_OMITTED_STATE_KEYS
    }


def build_view_update_user_message(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    route_map_prompt_format: RouteMapPromptFormat = "canonical",
    combat_skill_lines_enabled: bool = True,
    recent_state_changes_enabled: bool = True,
) -> str:
    viewed = _viewed_labels(state)
    lines = [
        "Requested information: " + ", ".join(viewed) + ".",
        TURN_CHAT_CURRENT_STATE_NOTICE,
    ]
    append_viewed_information(lines, state, state.get("player") or {})
    lines.extend(
        [
            "",
            "Current turn reminder:",
            _render_current_turn_state(
                state,
                combat_skill_lines_enabled=combat_skill_lines_enabled,
                recent_state_changes_enabled=recent_state_changes_enabled,
            ),
            "",
            "Legal actions:",
            _legal_actions_text(legal_actions, state=state),
        ]
    )
    lines.extend(
        _route_graph_prompt_lines(
            state,
            legal_actions,
            route_map_prompt_format=route_map_prompt_format,
            recent_state_changes_enabled=recent_state_changes_enabled,
        )
    )
    return "\n".join(lines)


def _route_graph_prompt_lines(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    route_map_prompt_format: RouteMapPromptFormat = "canonical",
    route_plan_enabled: bool = True,
    recent_state_changes_enabled: bool = True,
) -> list[str]:
    full_map = state.get("full_map")
    if not isinstance(full_map, dict) or full_map.get("type") != "map":
        return []
    if route_map_prompt_format not in {"route_graph", "canonical"}:
        raise ValueError(f"Unsupported route_map_prompt_format: {route_map_prompt_format}")
    if route_map_prompt_format == "canonical":
        route_payload = build_canonical_route_map(
            full_map,
            state.get("choices", []) or [],
            legal_actions=legal_actions,
            winged_boots_remaining=winged_boots_remaining_for_state(state),
        )
        route_label = "Canonical route map JSON:"
    else:
        route_payload = build_route_graph(
            full_map,
            state.get("choices", []) or [],
            legal_actions=legal_actions,
            winged_boots_remaining=winged_boots_remaining_for_state(state),
        )
        route_label = "Route graph JSON:"
    if not route_plan_enabled:
        constraints = dict(route_payload.get("constraints") or {})
        constraints.pop("ordered_plan_required", None)
        constraints.pop("complete_visible_route_required", None)
        route_payload = {**route_payload, "constraints": constraints}
    lines: list[str] = []
    if route_plan_enabled:
        active_route_plan = state.get("active_route_plan")
        if isinstance(active_route_plan, dict):
            prompt_route_plan = _route_plan_context_for_prompt(
                active_route_plan,
                recent_state_changes_enabled=recent_state_changes_enabled,
            )
            decision_basis = "current state, legal_actions, route_graph, deck, relics, and boss"
            if recent_state_changes_enabled:
                decision_basis += ", and recent state changes"
            lines.extend(
                [
                    "",
                    "Active route plan JSON:",
                    json.dumps(prompt_route_plan, ensure_ascii=False, separators=(",", ":")),
                    (
                        "Active route plan is advisory context, not a binding commitment. Evaluate the current MDP state: "
                        f"choose from {decision_basis}. "
                        "If following active_route_plan.expected_next, select expected_next and omit route_plan_event while "
                        "still explaining the action in reason. You can follow expected_next only when that exact node is "
                        "currently selectable in legal_actions. If expected_next is absent from legal_actions, or newly "
                        "revealed map structure changes the route, select a legal map action and emit route_plan_updated "
                        'with reason_tags including the separate string "map_changed", a non-empty reason, and a complete '
                        "new_plan from the selected legal node to the visible Boss or visible terminal node."
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Active route plan JSON:",
                    "null",
                    (
                        "No formal active route plan exists. For a selected map action, emit route_plan_created with a "
                        "complete new_plan. Only this Active route plan JSON is executable route state; free-form run "
                        "plans or route intentions are not active routes."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            route_label,
            json.dumps(route_payload, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    return lines


def _route_decision_state_for_prompt(
    state: dict[str, Any],
    *,
    recent_state_changes_enabled: bool = True,
) -> dict[str, Any]:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    prompt_state = {
        "decision": state.get("decision") or state.get("type"),
        "act": state.get("act") or context.get("act"),
        "floor": state.get("floor") or context.get("floor"),
        "room_type": context.get("room_type"),
        "boss": _route_named_summary(context.get("boss")),
        # Route choices depend on exact deck, relic, and potion mechanics.  A
        # grouped deck summary is useful as an index, but it must not replace
        # the complete engine-exported objects (including upgrade previews).
        "context": _combat_mdp_visible_value(context),
        "player": _combat_mdp_visible_value(player),
        "choices": _combat_mdp_visible_value(state.get("choices") or []),
        "deck_summary": _route_deck_summary(player),
    }
    if recent_state_changes_enabled:
        prompt_state["recent_state_changes"] = _route_recent_state_changes(state.get("last_action_result"))
    return prompt_state


def _route_deck_summary(player: dict[str, Any]) -> dict[str, Any]:
    deck = player.get("deck")
    if not isinstance(deck, list):
        return {
            "visible": False,
            "deck_size": player.get("deck_size"),
        }
    return {
        "visible": True,
        "deck_size": player.get("deck_size", len(deck)),
        "groups": compact_deck([card for card in deck if isinstance(card, dict)]),
    }


def _route_recent_state_changes(last_action_result: Any) -> dict[str, Any]:
    if not isinstance(last_action_result, dict):
        return {
            "visible": False,
            "changes": [],
            "hp_changes": [],
        }
    changes = [str(item) for item in last_action_result.get("changes") or []]
    hp_changes = [item for item in changes if "hp" in item.lower()]
    return {
        "visible": True,
        "action_label": last_action_result.get("action_label"),
        "action_kind": last_action_result.get("action_kind"),
        "decision_before": last_action_result.get("decision_before"),
        "decision_after": last_action_result.get("decision_after"),
        "changes": changes,
        "hp_changes": hp_changes,
    }


def _route_named_summary(value: Any) -> dict[str, Any] | str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            "name": name(value),
            "id": value.get("id"),
        }
    return name(value)


def _route_plan_context_for_prompt(
    active_route_plan: dict[str, Any],
    *,
    recent_state_changes_enabled: bool = True,
) -> dict[str, Any]:
    context = dict(active_route_plan)
    context.setdefault("role", "advisory_current_state_feature")
    if recent_state_changes_enabled:
        context.setdefault(
            "decision_basis",
            "choose from current state, legal_actions, route_graph, deck, relics, boss, and recent state changes",
        )
    else:
        context["decision_basis"] = "choose from current state, legal_actions, route_graph, deck, relics, and boss"
    context.setdefault("past_plan_semantics", "previous plan is context, not a binding commitment")
    return context


def build_compact_assistant_message(action: LegalAction, parsed: dict[str, Any]) -> str:
    reason = _one_line(parsed.get("reason") if isinstance(parsed, dict) else "")
    payload: dict[str, Any] = {
        "action": action.label,
        "action_kind": action.kind,
    }
    if isinstance(parsed, dict):
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


def _legal_actions_text(
    legal_actions: list[LegalAction],
    *,
    state: dict[str, Any] | None = None,
) -> str:
    if isinstance(state, dict) and state.get("decision") == "card_select":
        cards = state.get("cards") or []
        selection_actions = [action for action in legal_actions if action.kind in {"card_select", "card_select_space"}]
        if selection_actions:
            min_select = int(state.get("min_select", 1) or 0)
            max_select = int(state.get("max_select", min_select) or min_select)
            nonempty_min_select = max(1, min_select)
            candidate_indices = [card.get("index") for card in cards if isinstance(card, dict)]
            legal_combination_count = card_select_combination_count(
                len(candidate_indices),
                nonempty_min_select,
                max_select,
            )
            payload = {
                "kind": "card_select",
                "prompt": state.get("prompt"),
                "candidate_indices": candidate_indices,
                "min_select": min_select,
                "max_select": max_select,
                "nonempty_min_select": nonempty_min_select,
                "can_skip": any(action.kind == "card_select_skip" for action in legal_actions),
                "indices_must_be_distinct": True,
                "selection_order": "unordered",
                "legal_combination_count": legal_combination_count,
                "selection_rule": (
                    "Any non-empty set of distinct candidate indices whose count is between nonempty_min_select and "
                    "max_select forms a legal card selection. Return it in selected_card_indices. An empty list selects "
                    "the separate skip action only when can_skip is true."
                ),
                # Preserve every engine-exported field for every candidate. The
                # optimization removes repeated combination labels, not facts.
                "candidates": list(cards),
                "non_selection_actions": [
                    action.to_prompt_dict()
                    for action in legal_actions
                    if action.kind not in {"card_select", "card_select_space"}
                ],
            }
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps([action.to_prompt_dict() for action in legal_actions], ensure_ascii=False, separators=(",", ":"))


def _map_route_hint_lines(state: dict[str, Any], map_route_hint: MapRouteHint) -> list[str]:
    if state.get("decision") != "map_select" or map_route_hint == "none":
        return []
    if map_route_hint == "max_restsites":
        return [
            "",
            "Map route experiment hint:",
            (
                "When comparing legal map routes, prioritize the complete visible path with the most RestSite rooms "
                "because each RestSite is the main planned recovery/upgrading opportunity. Treat lower immediate risk "
                "as a tie-breaker after RestSite count. Still choose a current legal map action and explain the route "
                "calculation in the JSON reason/calculations."
            ),
        ]
    raise ValueError(f"Unsupported map_route_hint: {map_route_hint}")


def _legal_actions_for_prompt(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    mode: ActionPromptOrderMode,
    rng: random.Random,
) -> list[LegalAction]:
    if mode == "default" or state.get("decision") != "map_select":
        return list(legal_actions)
    if mode != "shuffle_map":
        raise ValueError(f"Unsupported action_prompt_order: {mode}")

    map_actions = [action for action in legal_actions if action.kind == "map"]
    if len(map_actions) <= 1:
        return list(legal_actions)

    shuffled = list(map_actions)
    rng.shuffle(shuffled)
    shuffled_iter = iter(shuffled)
    return [next(shuffled_iter) if action.kind == "map" else action for action in legal_actions]


def _action_prompt_order_meta(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    display_actions: list[LegalAction],
    *,
    mode: ActionPromptOrderMode,
) -> dict[str, Any]:
    if state.get("decision") != "map_select":
        return {}

    original_map_actions = [action for action in legal_actions if action.kind == "map"]
    display_map_actions = [action for action in display_actions if action.kind == "map"]
    if not original_map_actions:
        return {}

    meta: dict[str, Any] = {
        "action_prompt_order": mode,
        "action_prompt_map_action_ids": [action.action_id for action in display_map_actions],
        "action_prompt_map_action_labels": [action.label for action in display_map_actions],
    }
    if [action.action_id for action in display_map_actions] != [action.action_id for action in original_map_actions]:
        meta["action_prompt_order_changed"] = True
    return meta


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


def _render_current_turn_state(
    state: dict[str, Any],
    *,
    combat_skill_lines_enabled: bool = True,
    recent_state_changes_enabled: bool = True,
) -> str:
    # A turn-chat transcript deliberately retains prior actions, so every new
    # user message must restate the complete *current* visible state.  Reuse
    # the canonical renderer rather than maintaining a shorter second view
    # that can leave stale hands, powers, potions, or character mechanics in
    # the conversation.
    return render_state_text(
        state,
        combat_skill_lines_enabled=combat_skill_lines_enabled,
        recent_state_changes_enabled=recent_state_changes_enabled,
    )


def _combat_mdp_state_for_prompt(
    state: dict[str, Any],
    *,
    skill_hints_enabled: bool = False,
    skill_hints_profile: CombatSkillHintsProfile = "baseline",
    skill_hints_max_hints: int = 4,
    sequence_metadata_enabled: bool = True,
) -> dict[str, Any]:
    combat = state.get("combat") if isinstance(state.get("combat"), dict) else state
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    hand = state.get("hand") if state.get("decision") == "combat_play" else combat.get("hand")
    if not isinstance(hand, list):
        hand = []
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else state.get("enemies")
    if not isinstance(enemies, list):
        enemies = []
    player_powers = state.get("player_powers")
    if not isinstance(player_powers, list):
        player_powers = combat.get("player_powers")
    if not isinstance(player_powers, list):
        player_powers = player.get("powers")
    if not isinstance(player_powers, list):
        player_powers = []
    potions = player.get("potions")
    if not isinstance(potions, list):
        potions = state.get("potions")
    if not isinstance(potions, list):
        potions = []
    relics = player.get("relics")
    if not isinstance(relics, list):
        relics = state.get("relics")
    if not isinstance(relics, list):
        relics = []
    payload = {
        "decision": state.get("decision"),
        "turn": {
            "act": context.get("act", state.get("act")),
            "floor": context.get("floor", state.get("floor")),
            "room_type": context.get("room_type"),
            "round": combat.get("round", state.get("round")),
        },
        "player": {
            "name": player.get("name"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block", 0),
            "gold": player.get("gold"),
            "deck_size": player.get("deck_size"),
            "energy": combat.get("energy", state.get("energy")),
            "max_energy": combat.get("max_energy", state.get("max_energy")),
            "potion_slots": player.get("potion_slots"),
            "potion_empty_slots": player.get("potion_empty_slots"),
            "powers": [_combat_mdp_power(power) for power in player_powers if isinstance(power, dict)],
            "potions": [_combat_mdp_item(potion) for potion in potions if isinstance(potion, dict)],
            "relics": [_combat_mdp_item(relic) for relic in relics if isinstance(relic, dict)],
        },
        "enemies": [_combat_mdp_enemy(enemy) for enemy in enemies if isinstance(enemy, dict)],
        "hand_snapshot": [
            _combat_mdp_card(card, sequence_metadata_enabled=sequence_metadata_enabled)
            for card in hand
            if isinstance(card, dict)
        ],
        "piles": {
            "draw_count": combat.get("draw_pile_count", state.get("draw_pile_count")),
            "discard_count": combat.get("discard_pile_count", state.get("discard_pile_count")),
            "exhaust_count": combat.get("exhaust_pile_count", state.get("exhaust_pile_count")),
        },
    }
    for key in ("orbs", "orb_slots", "stars", "osty"):
        if key in combat:
            payload[key] = _combat_mdp_visible_value(combat.get(key))
        elif key in state:
            payload[key] = _combat_mdp_visible_value(state.get(key))
    viewed_information = _combat_mdp_viewed_information(state, player)
    if viewed_information:
        payload["viewed_information"] = viewed_information
    if state.get("decision") == "card_select":
        source_keys = (
            "source_card",
            "source_power",
            "source_potion",
            "source_event_option",
            "source_room_option",
        )
        payload["selection_context"] = {
            "prompt": state.get("prompt"),
            "min_select": state.get("min_select"),
            "max_select": state.get("max_select"),
            "can_skip": state.get("can_skip"),
            "candidates": [
                dict(card)
                for card in state.get("cards") or []
                if isinstance(card, dict)
            ],
            "sources": {
                key: dict(state[key])
                for key in source_keys
                if isinstance(state.get(key), dict)
            },
        }
    if skill_hints_enabled:
        guidance = _combat_guidance_for_prompt(
            state,
            profile=skill_hints_profile,
            max_hints=skill_hints_max_hints,
        )
        if guidance["hints"]:
            payload["combat_guidance"] = guidance
    return payload


def _combat_guidance_for_prompt(
    state: dict[str, Any],
    *,
    profile: CombatSkillHintsProfile = "baseline",
    max_hints: int = 4,
) -> dict[str, Any]:
    if profile not in {"baseline", "survival", "targeting", "full"}:
        profile = "baseline"
    effective_max_hints = max(1, int(max_hints or 4))
    if profile == "full":
        effective_max_hints = max(effective_max_hints, 8)
    raw_hints = combat_skill_hints(state, max_hints=effective_max_hints)
    selected = [
        hint
        for hint in raw_hints
        if profile in {"baseline", "full"} or _combat_skill_hint_matches_profile(hint, profile)
    ]
    return {
        "source": "derived_visible_state",
        "mode": "structured",
        "profile": profile,
        "hints": [{"id": _combat_skill_hint_id(hint), "text": hint} for hint in selected[:effective_max_hints]],
    }


def _combat_skill_hint_matches_profile(hint: str, profile: CombatSkillHintsProfile) -> bool:
    hint_id = _combat_skill_hint_id(hint)
    if profile == "survival":
        return hint_id in {"priority_order", "survival", "no_attack", "self_damage"}
    if profile == "targeting":
        return hint_id in {"priority_order", "kill_now", "aoe", "focus"}
    return True


def _combat_skill_hint_id(hint: str) -> str:
    prefix = hint.split(":", 1)[0].strip().lower()
    mapping = {
        "order": "priority_order",
        "survive": "survival",
        "no_attack": "no_attack",
        "kill_now": "kill_now",
        "aoe": "aoe",
        "focus": "focus",
        "self_damage": "self_damage",
        "draw": "draw",
    }
    return mapping.get(prefix, re.sub(r"[^a-z0-9]+", "_", prefix).strip("_") or "hint")


def _combat_turn_process_for_prompt(
    state: dict[str, Any],
    actions_this_turn: list[dict[str, Any]],
    current_turn_intent: dict[str, Any] | None = None,
    turn_board: list[dict[str, Any]] | None = None,
    combat_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    combat = state.get("combat") if isinstance(state.get("combat"), dict) else state
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    actions = [dict(action) for action in actions_this_turn]
    process = {
        "turn_key": list(turn_chat_key(state) or []),
        "actions_this_turn": actions,
        "summary": {
            "action_count": len(actions),
            "cards_played": sum(1 for action in actions if action.get("action_kind") == "combat_play_card"),
            "current_energy": combat.get("energy", state.get("energy")),
            "current_block": player.get("block", 0),
        },
    }
    if current_turn_intent:
        process["current_turn_intent"] = dict(current_turn_intent)
        process["summary"]["current_turn_intent_present"] = True
    if turn_board:
        oldest_turn_start = turn_board[0].get("turn_start")
        if isinstance(oldest_turn_start, dict):
            process["recent_window_start"] = _combat_history_window_start_for_prompt(
                oldest_turn_start
            )
        process["recent_turns"] = [_combat_turn_for_prompt(item) for item in turn_board]
        process["summary"]["recent_turn_count"] = len(turn_board)
    if combat_summary:
        process["combat_summary"] = dict(combat_summary)
        process["summary"]["summarized_turn_count"] = combat_summary.get("completed_turns", 0)
    if turn_board or combat_summary:
        process["memory_contract"] = {
            "facts": "engine_state_only",
            "history_window": "up_to_3_completed_turns",
            "history_representation": "window_start_plus_actions_and_end_turn_deltas",
            "snapshot_diff_ordering": "not_observed",
            "legal_action_ids": "current_state_only",
        }
    return process


def _combat_turn_for_prompt(turn_entry: dict[str, Any]) -> dict[str, Any]:
    boundary = turn_entry.get("turn_boundary")
    if not isinstance(boundary, dict):
        boundary = {}
    before_end_turn = boundary.get("before_end_turn")
    if not isinstance(before_end_turn, dict):
        before_end_turn = {}
    observed_facts = boundary.get("observed_facts")
    if not isinstance(observed_facts, list):
        observed_facts = []
    return {
        "turn_key": list(turn_entry.get("turn_key") or []),
        "round": turn_entry.get("round"),
        "actions": [
            _combat_history_action_for_prompt(action)
            for action in turn_entry.get("actions") or []
            if isinstance(action, dict)
        ],
        "end_turn_transition": {
            "announced_enemy_intents": _combat_announced_enemy_intents_for_prompt(
                before_end_turn
            ),
            "observed_facts": [
                dict(fact)
                for fact in observed_facts
                if isinstance(fact, dict) and _combat_history_boundary_fact_is_relevant(fact)
            ],
            "ordering": boundary.get("ordering", "not_observed"),
            "source": boundary.get("source", "snapshot_diff"),
        },
    }


def _combat_history_window_start_for_prompt(snapshot: dict[str, Any]) -> dict[str, Any]:
    player = snapshot.get("player")
    if not isinstance(player, dict):
        player = {}
    enemies = snapshot.get("enemies")
    if not isinstance(enemies, list):
        enemies = []
    return {
        "round": snapshot.get("round"),
        "player": {
            key: player.get(key)
            for key in ("hp", "max_hp", "powers")
            if player.get(key) not in (None, [])
        },
        "enemies": [
            {
                key: enemy.get(key)
                for key in ("index", "name", "hp", "max_hp", "powers")
                if enemy.get(key) not in (None, [])
            }
            for enemy in enemies
            if isinstance(enemy, dict)
        ],
    }


def _combat_history_action_for_prompt(action: dict[str, Any]) -> dict[str, Any]:
    projected = {
        key: action.get(key)
        for key in (
            "action_kind",
            "card_name",
            "card_type",
            "card_cost_at_play",
            "engine_action",
            "target",
        )
        if action.get(key) is not None
    }
    changes = action.get("observed_changes")
    if isinstance(changes, list):
        meaningful_changes = [
            str(change)
            for change in changes
            if not str(change).startswith(
                (
                    "energy:",
                    "hand_count:",
                    "draw_pile:",
                    "discard_pile:",
                    "exhaust_pile:",
                )
            )
        ]
        if meaningful_changes:
            projected["observed_effects"] = meaningful_changes
    return projected


def _combat_history_boundary_fact_is_relevant(fact: dict[str, Any]) -> bool:
    entity = str(fact.get("entity") or "")
    field = str(fact.get("field") or "")
    if entity == "combat":
        return False
    if entity == "player" and field == "block":
        return False
    if entity.startswith("enemy:") and field == "intents":
        return False
    return True


def _combat_announced_enemy_intents_for_prompt(
    before_end_turn: dict[str, Any],
) -> list[dict[str, Any]]:
    enemies = before_end_turn.get("enemies")
    if not isinstance(enemies, list):
        return []
    announced: list[dict[str, Any]] = []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        intents = enemy.get("intents")
        if not isinstance(intents, list):
            intents = []
        announced.append(
            {
                "index": enemy.get("index"),
                "name": enemy.get("name"),
                "intents": [
                    dict(intent) for intent in intents if isinstance(intent, dict)
                ],
            }
        )
    return announced


def _combat_turn_board_entry(
    turn_start_snapshot: dict[str, Any],
    state: dict[str, Any],
    next_state: dict[str, Any],
    actions_this_turn: list[dict[str, Any]],
) -> dict[str, Any]:
    before_end_turn = _combat_memory_snapshot(state)
    next_turn_start = _combat_memory_snapshot(next_state)
    return {
        "turn_key": list(turn_chat_key(state) or []),
        "round": turn_start_snapshot.get("round", before_end_turn.get("round")),
        "turn_start": dict(turn_start_snapshot),
        "actions": [_combat_turn_board_action(action) for action in actions_this_turn],
        "turn_boundary": {
            "trigger": "combat_end_turn",
            "source": "snapshot_diff",
            "ordering": "not_observed",
            "before_end_turn": before_end_turn,
            "observed_facts": _combat_boundary_facts(before_end_turn, next_turn_start),
            "next_turn_start": next_turn_start,
        },
    }


def _combat_memory_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    combat = state.get("combat") if isinstance(state.get("combat"), dict) else state
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else state.get("enemies")
    if not isinstance(enemies, list):
        enemies = []
    hand = combat.get("hand") if isinstance(combat.get("hand"), list) else state.get("hand")
    if not isinstance(hand, list):
        hand = []
    player_powers = combat.get("player_powers")
    if not isinstance(player_powers, list):
        player_powers = state.get("player_powers")
    if not isinstance(player_powers, list):
        player_powers = player.get("powers")
    if not isinstance(player_powers, list):
        player_powers = []
    compact_enemies = [_combat_memory_enemy(enemy) for enemy in enemies if isinstance(enemy, dict)]
    return {
        "round": combat.get("round", state.get("round")),
        "player": {
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block", 0),
            "powers": [_combat_memory_power(power) for power in player_powers if isinstance(power, dict)],
        },
        "energy": combat.get("energy", state.get("energy")),
        "max_energy": combat.get("max_energy", state.get("max_energy")),
        "hand_count": len(hand),
        "hand": [name(card.get("name")) for card in hand if isinstance(card, dict)],
        "piles": {
            "draw_count": combat.get("draw_pile_count", state.get("draw_pile_count")),
            "discard_count": combat.get("discard_pile_count", state.get("discard_pile_count")),
            "exhaust_count": combat.get("exhaust_pile_count", state.get("exhaust_pile_count")),
        },
        "incoming_damage": _incoming_damage(enemies),
        "enemies": compact_enemies,
    }


def _combat_memory_enemy(enemy: dict[str, Any]) -> dict[str, Any]:
    """Keep the historical turn board compact; full facts live in the current MDP state."""

    compact = {
        "index": enemy.get("index"),
        "name": name(enemy.get("name")),
        "hp": enemy.get("hp"),
        "max_hp": enemy.get("max_hp"),
        "block": enemy.get("block", 0),
        "intents": [
            {
                "type": intent.get("type"),
                "damage": intent.get("damage"),
                "hits": intent.get("hits", 1),
            }
            for intent in enemy.get("intents") or []
            if isinstance(intent, dict)
        ],
        "powers": [
            _combat_memory_power(power)
            for power in enemy.get("powers") or []
            if isinstance(power, dict)
        ],
    }
    instance_id = enemy.get("instance_id")
    if instance_id is not None:
        compact["instance_id"] = instance_id
    return compact


def _combat_memory_power(power: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name(power.get("name")),
        "amount": power.get("amount"),
        "type": power.get("type"),
    }


def _combat_boundary_facts(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    _append_combat_fact(facts, "combat", "round", before.get("round"), after.get("round"))

    player_before = before.get("player") if isinstance(before.get("player"), dict) else {}
    player_after = after.get("player") if isinstance(after.get("player"), dict) else {}
    for field_name in ("hp", "max_hp", "block"):
        _append_combat_fact(facts, "player", field_name, player_before.get(field_name), player_after.get(field_name))
    _append_combat_power_facts(facts, "player", player_before.get("powers"), player_after.get("powers"))

    for field_name in ("energy", "max_energy", "hand_count", "hand"):
        _append_combat_fact(facts, "combat", field_name, before.get(field_name), after.get(field_name))
    piles_before = before.get("piles") if isinstance(before.get("piles"), dict) else {}
    piles_after = after.get("piles") if isinstance(after.get("piles"), dict) else {}
    for field_name in ("draw_count", "discard_count", "exhaust_count"):
        _append_combat_fact(facts, "combat", field_name, piles_before.get(field_name), piles_after.get(field_name))

    enemies_before = _combat_snapshot_enemies(before)
    enemies_after = _combat_snapshot_enemies(after)
    for enemy_key in sorted(set(enemies_before) | set(enemies_after)):
        enemy_before = enemies_before.get(enemy_key)
        enemy_after = enemies_after.get(enemy_key)
        entity = f"enemy:{enemy_key}"
        if enemy_before is None or enemy_after is None:
            _append_combat_fact(facts, entity, "presence", enemy_before, enemy_after)
            continue
        for field_name in ("hp", "max_hp", "block", "intents"):
            _append_combat_fact(facts, entity, field_name, enemy_before.get(field_name), enemy_after.get(field_name))
        _append_combat_power_facts(facts, entity, enemy_before.get("powers"), enemy_after.get("powers"))
    return facts


def _append_combat_fact(
    facts: list[dict[str, Any]],
    entity: str,
    field_name: str,
    before: Any,
    after: Any,
) -> None:
    if before == after:
        return
    fact: dict[str, Any] = {"entity": entity, "field": field_name, "before": before, "after": after}
    if _is_number(before) and _is_number(after):
        fact["delta"] = after - before
    facts.append(fact)


def _append_combat_power_facts(
    facts: list[dict[str, Any]],
    entity: str,
    before: Any,
    after: Any,
) -> None:
    before_by_name = _combat_powers_by_name(before)
    after_by_name = _combat_powers_by_name(after)
    for power_name in sorted(set(before_by_name) | set(after_by_name)):
        power_before = before_by_name.get(power_name)
        power_after = after_by_name.get(power_name)
        if power_before == power_after:
            continue
        if power_before is None:
            change = "added"
        elif power_after is None:
            change = "removed"
        else:
            change = "changed"
        facts.append(
            {
                "entity": entity,
                "field": f"power:{power_name}",
                "change": change,
                "before": power_before,
                "after": power_after,
            }
        )


def _combat_powers_by_name(powers: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(powers, list):
        return {}
    return {
        str(power.get("name") or power.get("type") or "unknown"): dict(power)
        for power in powers
        if isinstance(power, dict)
    }


def _combat_snapshot_enemies(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    enemies = snapshot.get("enemies")
    if not isinstance(enemies, list):
        return {}
    return {
        (
            f"instance_id:{enemy.get('instance_id')}"
            if enemy.get("instance_id") is not None
            else f"{enemy.get('index')}:{enemy.get('name')}"
        ): enemy
        for enemy in enemies
        if isinstance(enemy, dict)
    }


def _fold_combat_turn_summary(summary: dict[str, Any], turn_entry: dict[str, Any]) -> None:
    round_number = turn_entry.get("round")
    if not summary:
        summary.update(
            {
                "source": "deterministic_fold",
                "covered_rounds": {"first": round_number, "last": round_number, "count": 0},
                "completed_turns": 0,
                "player_hp_delta": 0,
                "boundary_player_hp_delta": 0,
                "enemy_hp_delta_by_enemy": {},
                "boundary_enemy_hp_delta_by_enemy": {},
                "cards_played": {},
                "power_event_counts": {},
            }
        )

    covered_rounds = summary["covered_rounds"]
    covered_rounds["last"] = round_number
    covered_rounds["count"] += 1
    summary["completed_turns"] += 1

    turn_start = turn_entry.get("turn_start") if isinstance(turn_entry.get("turn_start"), dict) else {}
    boundary = turn_entry.get("turn_boundary") if isinstance(turn_entry.get("turn_boundary"), dict) else {}
    before_end = boundary.get("before_end_turn") if isinstance(boundary.get("before_end_turn"), dict) else {}
    next_start = boundary.get("next_turn_start") if isinstance(boundary.get("next_turn_start"), dict) else {}
    _add_player_hp_delta(summary, "player_hp_delta", turn_start, next_start)
    _add_player_hp_delta(summary, "boundary_player_hp_delta", before_end, next_start)
    _add_enemy_hp_deltas(summary["enemy_hp_delta_by_enemy"], turn_start, next_start)
    _add_enemy_hp_deltas(summary["boundary_enemy_hp_delta_by_enemy"], before_end, next_start)

    cards_played = summary["cards_played"]
    for action in turn_entry.get("actions") or []:
        if not isinstance(action, dict) or action.get("action_kind") != "combat_play_card":
            continue
        card_name = action.get("card_name")
        if card_name:
            cards_played[str(card_name)] = cards_played.get(str(card_name), 0) + 1

    power_event_counts = summary["power_event_counts"]
    for fact in boundary.get("observed_facts") or []:
        if not isinstance(fact, dict) or not str(fact.get("field") or "").startswith("power:"):
            continue
        event_key = f"{fact.get('entity')}|{fact.get('field')}|{fact.get('change')}"
        power_event_counts[event_key] = power_event_counts.get(event_key, 0) + 1


def _add_player_hp_delta(
    summary: dict[str, Any],
    summary_key: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    player_before = before.get("player") if isinstance(before.get("player"), dict) else {}
    player_after = after.get("player") if isinstance(after.get("player"), dict) else {}
    hp_before = player_before.get("hp")
    hp_after = player_after.get("hp")
    if _is_number(hp_before) and _is_number(hp_after):
        summary[summary_key] += hp_after - hp_before


def _add_enemy_hp_deltas(
    totals: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    enemies_before = _combat_snapshot_enemies(before)
    enemies_after = _combat_snapshot_enemies(after)
    for enemy_key in set(enemies_before) | set(enemies_after):
        hp_before = enemies_before.get(enemy_key, {}).get("hp", 0)
        hp_after = enemies_after.get(enemy_key, {}).get("hp", 0)
        if not (_is_number(hp_before) and _is_number(hp_after)):
            continue
        total = totals.get(enemy_key, 0) + hp_after - hp_before
        if total:
            totals[enemy_key] = total
        else:
            totals.pop(enemy_key, None)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _combat_turn_board_action(action: dict[str, Any]) -> dict[str, Any]:
    keep: dict[str, Any] = {}
    for key in ("action_kind", "card_name", "card_type", "card_cost_at_play", "engine_action"):
        value = action.get(key)
        if value is not None:
            keep[key] = value
    target = action.get("target")
    if isinstance(target, dict):
        keep["target"] = {key: target.get(key) for key in ("index", "name") if target.get(key) is not None}
    changes = action.get("observed_changes")
    if isinstance(changes, list):
        keep["observed_changes"] = [str(item) for item in changes]
    return keep


def _combat_mdp_action_fact(
    state: dict[str, Any],
    action: LegalAction,
    result: dict[str, Any],
) -> dict[str, Any]:
    is_view_action = action.command.get("cmd") == "bench_view" or action.kind.startswith("view_")
    fact: dict[str, Any] = {
        "action_kind": action.kind,
        "observed_changes": (
            []
            if is_view_action
            else [str(item) for item in result.get("changes") or []]
        ),
    }
    command_action = action.command.get("action")
    args = action.command.get("args") if isinstance(action.command.get("args"), dict) else {}
    if command_action == "play_card":
        card = _card_by_index(state.get("hand"), args.get("card_index"))
        if card:
            fact["card_name"] = name(card.get("name"))
            fact["card_type"] = card.get("type")
            fact["card_cost_at_play"] = card.get("cost")
        target = _enemy_by_index(state.get("enemies"), args.get("target_index"))
        if target:
            fact["target"] = {
                "index": target.get("index"),
                "name": name(target.get("name")),
            }
    elif command_action == "use_potion":
        fact["potion_index"] = args.get("potion_index")
        target = _enemy_by_index(state.get("enemies"), args.get("target_index"))
        if target:
            fact["target"] = {
                "index": target.get("index"),
                "name": name(target.get("name")),
            }
    else:
        fact["engine_action"] = command_action
    return fact


def _combat_mdp_turn_intent_fact(
    action: LegalAction,
    meta: dict[str, Any],
) -> dict[str, Any] | None:
    parsed = meta.get("parsed") if isinstance(meta.get("parsed"), dict) else {}
    intent = _one_line(parsed.get("turn_intent"))
    if not intent:
        return None
    return {
        "intent": "Re-evaluate the provisional action order or resource objective against the current state.",
        "status": "provisional_recheck_against_current_state",
        "after_action_label": action.label,
    }


def _combat_mdp_card(
    card: dict[str, Any],
    *,
    sequence_metadata_enabled: bool = True,
) -> dict[str, Any]:
    compact = _combat_mdp_visible_object(card)
    if not sequence_metadata_enabled:
        return compact
    block_reasons = _combat_mdp_sequence_block_reasons(card)
    compact["sequence_eligible"] = not block_reasons
    if block_reasons:
        compact["sequence_block_reasons"] = block_reasons
    recompute = _combat_mdp_recompute_reasons(card)
    if recompute:
        compact["must_recompute_after_play"] = recompute
    return compact


def _combat_mdp_enemy(enemy: dict[str, Any]) -> dict[str, Any]:
    visible = _combat_mdp_visible_object(enemy)
    visible["intents"] = [
        _combat_mdp_intent(intent)
        for intent in enemy.get("intents") or []
        if isinstance(intent, dict)
    ]
    visible["powers"] = [
        _combat_mdp_power(power)
        for power in enemy.get("powers") or []
        if isinstance(power, dict)
    ]
    return visible


def _combat_mdp_intent(intent: dict[str, Any]) -> dict[str, Any]:
    return _combat_mdp_visible_object(intent)


def _combat_mdp_power(power: dict[str, Any]) -> dict[str, Any]:
    return _combat_mdp_visible_object(power)


def _combat_mdp_item(item: dict[str, Any]) -> dict[str, Any]:
    """Preserve the complete visible engine facts for a relic or potion."""

    return _combat_mdp_visible_object(item)


def _combat_mdp_visible_object(value: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe, near-lossless copy of an engine-exported object."""

    copied = {
        str(key): _combat_mdp_visible_value(item)
        for key, item in value.items()
    }
    raw_name = value.get("name")
    if raw_name is not None:
        normalized_name = name(raw_name)
        if isinstance(raw_name, dict):
            copied["name_data"] = _combat_mdp_visible_value(raw_name)
        copied["name"] = normalized_name
    if isinstance(value.get("hover_tips"), list):
        copied["hover_tips"] = [
            _combat_mdp_visible_value(tip)
            for tip in dedupe_hover_tips(value.get("hover_tips"))
        ]
    return copied


def _combat_mdp_visible_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_combat_mdp_visible_value(item) for item in value]
    if isinstance(value, dict):
        copied = {
            str(key): _combat_mdp_visible_value(item)
            for key, item in value.items()
        }
        if isinstance(value.get("hover_tips"), list):
            copied["hover_tips"] = [
                _combat_mdp_visible_value(tip)
                for tip in dedupe_hover_tips(value.get("hover_tips"))
            ]
        return copied
    return str(value)


def _combat_mdp_viewed_information(
    state: dict[str, Any],
    player: dict[str, Any],
) -> dict[str, Any]:
    """Expose complete card objects only for information the agent explicitly viewed."""

    viewed: dict[str, Any] = {}
    if state.get("view_deck"):
        deck = player.get("deck") if isinstance(player.get("deck"), list) else []
        viewed["deck"] = [
            _combat_mdp_visible_object(card)
            for card in deck
            if isinstance(card, dict)
        ]
    for pile_name in ("draw", "discard", "exhaust"):
        if not state.get(f"view_{pile_name}_pile"):
            continue
        viewed[f"{pile_name}_pile"] = [
            _combat_mdp_visible_object(card)
            for card in pile_cards_from_state(state, pile_name)
            if isinstance(card, dict)
        ]
    return viewed


def _combat_mdp_stats(stats: Any) -> dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in stats.items():
        stat = _combat_mdp_stat_value(value)
        if stat is not None:
            compact[str(key)] = stat
    return compact


def _combat_mdp_stat_value(value: Any, *, depth: int = 0) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 3:
        return _one_line(value)[:120]
    if isinstance(value, list):
        return [_combat_mdp_stat_value(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            stat = _combat_mdp_stat_value(item, depth=depth + 1)
            if stat is not None:
                compact[str(key)] = stat
        return compact
    return _one_line(value)[:120]


def _combat_mdp_text(value: Any) -> str:
    if isinstance(value, dict):
        text = name(value)
    else:
        text = _one_line(value)
    return text[:300]


def _combat_mdp_sequence_block_reasons(card: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if card.get("can_play") is False:
        reasons.append("card is not currently playable")
    block_reason = sequence_card_block_reason(card, allowed_target_types={"None", "Self", "AnyEnemy", "AllEnemies"})
    if block_reason:
        reasons.append(block_reason)
    return reasons


def _combat_mdp_recompute_reasons(card: dict[str, Any]) -> list[str]:
    text = _combat_mdp_card_search_text(card)
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    stat_keys = {str(key).lower() for key in stats}
    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if "draw" in text or "draw" in stat_keys or "cards" in stat_keys:
        add("draw")
    if "discard" in text or "discard" in stat_keys:
        add("hand_change")
    if "exhaust" in text or "exhaust" in stat_keys:
        add("hand_change")
    if "random" in text or "random" in stat_keys or "random" in str(card.get("target_type") or "").lower():
        add("random_effect")
    if any(marker in text or marker in stat_keys for marker in ("choose", "select", "discover", "generate")):
        add("modal_or_generated_effect")
    if "shuffle" in text or "shuffle" in stat_keys:
        add("deck_order_change")
    if "upgrade" in text or "upgrade" in stat_keys:
        add("card_state_change")
    if any(key in stat_keys for key in ("vulnerablepower", "weakpower", "strengthpower", "dexteritypower")):
        add("modifier_change")
    return reasons


def _combat_mdp_card_search_text(card: dict[str, Any]) -> str:
    parts = [
        name(card.get("name")),
        str(card.get("id") or ""),
        _combat_mdp_text(card.get("description")),
        _combat_mdp_text(card.get("raw_description")),
        _combat_mdp_text(card.get("text")),
        str(card.get("target_type") or ""),
    ]
    keywords = card.get("keywords")
    if isinstance(keywords, list):
        parts.extend(str(keyword) for keyword in keywords)
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    parts.extend(str(key) for key in stats)
    return " ".join(part for part in parts if part).lower()


def _incoming_damage(enemies: list[dict[str, Any]]) -> int:
    total = 0
    for enemy in enemies:
        if not isinstance(enemy, dict) or _number_or_zero(enemy.get("hp")) <= 0:
            continue
        for intent in enemy.get("intents") or []:
            if not isinstance(intent, dict) or intent.get("type") not in {"Attack", "DeathBlow"}:
                continue
            total += int(_number_or_zero(intent.get("damage")) * _number_or_zero(intent.get("hits") or 1))
    return total


def _card_by_index(cards: Any, card_index: Any) -> dict[str, Any] | None:
    if not isinstance(cards, list):
        return None
    for card in cards:
        if isinstance(card, dict) and card.get("index") == card_index:
            return card
    return None


def _enemy_by_index(enemies: Any, enemy_index: Any) -> dict[str, Any] | None:
    if enemy_index is None or not isinstance(enemies, list):
        return None
    for enemy in enemies:
        if isinstance(enemy, dict) and enemy.get("index") == enemy_index:
            return enemy
    return None


def _number_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _messages_char_count(messages: list[dict[str, str]]) -> int:
    return sum(len(message.get("content", "")) for message in messages)


def _messages_preview(messages: list[dict[str, str]]) -> str:
    parts = []
    for index, message in enumerate(messages):
        parts.append(f"===== message[{index}] role={message.get('role', '?')} =====")
        parts.append(message.get("content", ""))
    return "\n".join(parts)


def _canonical_payload_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _context_trace_meta(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    variant: str,
    history: dict[str, Any] | None,
    active_policy_owner: str,
    combat_card_sequences_enabled: bool,
) -> dict[str, Any]:
    if turn_chat_key(state) is not None:
        state_payload = _combat_mdp_state_for_prompt(
            state,
            skill_hints_enabled=False,
            sequence_metadata_enabled=combat_card_sequences_enabled,
        )
    else:
        state_payload = _formal_generalist_state_for_prompt(state)
    meta = {
        "combat_context_variant": variant,
        "current_state_payload_sha256": _canonical_payload_sha256(state_payload),
        "legal_actions_sha256": _canonical_payload_sha256(
            [action.to_log_dict() for action in legal_actions]
        ),
        "active_policy_owner": active_policy_owner,
    }
    if history is None:
        meta.update(
            {
                "history_message_count": 0,
                "history_completed_turns": 0,
                "history_rounds": [],
                "history_chars": 0,
                "history_token_estimate": 0,
                "history_token_usage": {
                    "kind": "estimated_chars_div_4",
                    "tokens": 0,
                },
                "assistant_history": "none",
            }
        )
    else:
        meta.update(history)
    return meta


def _model_request_audit(messages: list[dict[str, str]], *, request_index: int) -> dict[str, Any]:
    frozen_messages = [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in messages
    ]
    encoded = json.dumps(
        frozen_messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "request_index": request_index,
        "messages": frozen_messages,
        "messages_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _append_model_request_audit(
    request_audits: list[dict[str, Any]],
    messages: list[dict[str, str]],
) -> None:
    request_audits.append(_model_request_audit(messages, request_index=len(request_audits)))


def _provider_finish_reason(response: dict[str, Any]) -> Any:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("finish_reason")


def _provider_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: response[key] for key in _PROVIDER_METADATA_KEYS if response.get(key) is not None}
    finish_reason = _provider_finish_reason(response)
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    return metadata


def _provider_response_text(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalize_self_consistency_decisions(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)

    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _self_consistency_sample_entry(
    sample_index: int,
    action: LegalAction,
    parsed: dict[str, Any],
    *,
    primary: bool,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "sample_index": sample_index,
        "primary": primary,
        "action_id": action.action_id,
        "kind": action.kind,
        "label": action.label,
    }
    reason = _one_line(parsed.get("reason") if isinstance(parsed, dict) else "")
    if reason:
        entry["reason"] = reason[:300]
    return entry


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


CUSTOM_SEQUENCE_KIND_LABEL_ERROR = (
    "combat_custom_sequence is an action kind, not a selected_action_label. "
    "Use the current legal action_id for the custom sequence gateway and copy that legal action label exactly"
)

CUSTOM_SEQUENCE_PAYLOAD_RULE = (
    "For combat_custom_sequence payloads, sequence must be a list of step objects with top-level integer card_index "
    "and optional top-level integer target_index. Do not wrap sequence steps in cmd, action, or args objects."
)


def _validate_turn_check_contract_response(
    parsed: dict[str, Any],
    contract: PromptContract,
) -> None:
    """Validate only the required response shape, never gameplay arithmetic."""

    if contract.turn_intent_enabled and not _one_line(parsed.get("turn_intent")):
        raise ValueError("turn_intent is required by the current combat response contract")
    if not contract.turn_check_enabled:
        return
    turn_check = parsed.get("turn_check")
    if not isinstance(turn_check, dict):
        raise ValueError("turn_check must be an object in the current combat response contract")
    for field_name in ("observed_numbers", "incoming_damage_calculation"):
        if not isinstance(turn_check.get(field_name), dict):
            raise ValueError(f"turn_check.{field_name} must be an object")
    selected_action_calculation = turn_check.get("selected_action_calculation")
    if selected_action_calculation is not None and not isinstance(selected_action_calculation, dict):
        raise ValueError("turn_check.selected_action_calculation must be an object or null")
    end_turn_audit = turn_check.get("end_turn_audit")
    if end_turn_audit is not None and not isinstance(end_turn_audit, dict):
        raise ValueError("turn_check.end_turn_audit must be an object or null")
def _validate_turn_check_action_response(
    parsed: dict[str, Any],
    contract: PromptContract,
    action: LegalAction,
) -> None:
    """Validate that the selected action uses only its applicable scratchpad branch."""

    if not contract.turn_check_enabled:
        return
    turn_check = parsed["turn_check"]
    selected_action_calculation = turn_check.get("selected_action_calculation")
    end_turn_audit = turn_check.get("end_turn_audit")
    if action.kind == "combat_end_turn":
        if selected_action_calculation is not None:
            raise ValueError(
                "turn_check.selected_action_calculation must be null when selecting end turn"
            )
        if not isinstance(end_turn_audit, dict):
            raise ValueError("turn_check.end_turn_audit must be an object when selecting end turn")
        return
    if not isinstance(selected_action_calculation, dict):
        raise ValueError(
            "turn_check.selected_action_calculation must be an object for a non-end-turn action"
        )
    if end_turn_audit is not None:
        raise ValueError("turn_check.end_turn_audit must be null for a non-end-turn action")


def _model_action_correction_text(
    error: str,
    *,
    contract: PromptContract,
    legal_actions: list[LegalAction],
) -> str:
    if contract.card_select_indices_enabled:
        return (
            f"Your previous response was invalid: {error}. "
            "For a card selection, return only JSON with selected_card_indices as distinct current candidate index "
            "integers satisfying min_select and max_select; use [] only for a visible legal skip. To choose a listed "
            "non-selection action such as viewing information, omit selected_card_indices and return its current "
            "action_id plus exact selected_action_label. Use exactly one of these two identity branches and include a "
            "reason describing the same choice."
        )

    sequence_rule = (
        "sequence included only when choosing combat_custom_sequence. "
        f"{CUSTOM_SEQUENCE_KIND_LABEL_ERROR}. {CUSTOM_SEQUENCE_PAYLOAD_RULE} "
        if contract.combat_custom_sequence_enabled
        else "do not include a sequence field because no combat_custom_sequence gateway is available in this current state. "
    )
    required_combat_fields: list[str] = []
    if contract.turn_intent_enabled:
        required_combat_fields.append("turn_intent")
    if contract.turn_check_enabled:
        required_combat_fields.append("turn_check")
    turn_check_rule = (
        "Preserve the required top-level "
        + " and ".join(required_combat_fields)
        + " fields from the original current-state response contract. "
        if required_combat_fields
        else ""
    )
    return (
        f"Your previous response was invalid: {error}. "
        f"Return only JSON with action_id between 0 and {len(legal_actions) - 1}, "
        "selected_action_label copied exactly from that current legal action label, "
        + sequence_rule
        + turn_check_rule
        + "and a reason describing the same selected action."
    )


def _decision_protocol_prompt_lines(state: dict[str, Any]) -> list[str]:
    """Return interaction semantics that apply only to the current decision.

    These lines describe the benchmark interface, not game strategy. Keeping
    them decision-scoped avoids exposing irrelevant instructions elsewhere.
    """

    if state.get("decision") != "combat_reward":
        return []
    return [
        "",
        "Decision-specific interaction protocol:",
        "A card_reward entry is an unopened reward pack. Opening it reveals the offered cards and does not add a card; choosing or skipping an offered card occurs only in the resulting card_reward state.",
    ]


def _game_rules_prompt_lines(
    state: dict[str, Any],
    *,
    enabled: bool = True,
) -> list[str]:
    if not enabled:
        return []
    rendered = render_game_rules(state)
    return ["", rendered] if rendered else []


def _compact_state_for_prompt(
    state: dict[str, Any],
    *,
    recent_state_changes_enabled: bool,
) -> dict[str, Any]:
    if recent_state_changes_enabled or "last_action_result" not in state:
        return compact_state(state)
    prompt_state = dict(state)
    prompt_state.pop("last_action_result", None)
    return compact_state(prompt_state)


def _prompt_header(
    prompt_style: PromptStyle,
    *,
    contract: PromptContract,
) -> list[str]:
    turn_intent_enabled = contract.turn_intent_enabled
    turn_check_enabled = contract.turn_check_enabled
    route_plan_enabled = contract.route_plan_enabled
    context_patch_fields_enabled = contract.context_patch_fields_enabled
    combat_card_sequences_enabled = contract.combat_custom_sequence_enabled
    card_select_indices_enabled = contract.card_select_indices_enabled
    combat_decision = contract.decision_type == "combat_play"

    atomic_action_rule = (
        "Each listed action is atomic. In combat, choose only one current legal card, one potion use, one view action, or end turn."
        if not contract.interface_preamble_enabled
        else "Each legal action is one atomic benchmark action. In combat, choose only one current legal card, one potion use, one view action, or end turn."
    )
    sequence_rules = (
        [
            "Each legal action is one atomic benchmark action, not an open-ended plan. Most legal actions execute one engine command; combat_custom_sequence is a model-proposed deterministic card sequence validated with the engine before execution.",
            "Choose combat_custom_sequence only for deterministic same-turn combat actions where every listed card is already in hand, currently playable, and every card play is atomic. Do not include cards that draw, discard, exhaust, choose/select, shuffle, generate/discover, upgrade, or use random effects or random targets; play those cards as single legal actions and wait for the next state.",
            "combat_custom_sequence is an action kind, not a selected_action_label; when selecting it, use the current legal action_id for the custom sequence gateway and copy that gateway's exact legal label.",
            CUSTOM_SEQUENCE_PAYLOAD_RULE,
            "Before choosing combat_custom_sequence, verify it is deterministic; sum each step's current cost against current energy and do not include a card that becomes unaffordable or unplayable after earlier steps.",
            "Include a non-empty top-level sequence only when the selected legal action is combat_custom_sequence; for any other action, use an empty sequence or omit the field.",
            "For combat_custom_sequence, card_index values are bound to the initial visible hand snapshot, and every bound card must remain playable after earlier sequence steps resolve. If you cannot verify the whole sequence from the visible hand snapshot, current energy, and prior step effects, use a single legal action instead.",
            f"combat_custom_sequence may contain at most {MAX_CUSTOM_SEQUENCE_VALIDATION_LENGTH} card steps; if a useful line is longer, submit the first safe prefix and wait for the next state.",
            "If an earlier step can kill or remove a targeted enemy, do not point later steps at that enemy; stop the sequence before the removal and wait for the next state.",
            "When choosing combat_custom_sequence, include a combat_notes cost audit and lethal-target audit; for attacks, use the target-specific damage_by_target or calculateddamage_by_target entry for that target, preferring unblocked_damage when present, not the base stats.damage value.",
            "If any step applies Vulnerable, Weak, Strength, Dexterity, or another modifier that can change later damage or block previews, stop the sequence at that modifier-changing step and wait for the next state.",
            "If shown damage is lethal against that enemy's current hp before later targeted steps, stop the sequence at the lethal/removal step and wait for the next state.",
            "Track cumulative target-specific shown damage per target; once cumulative damage reaches that enemy's current hp, no later sequence step may target it.",
        ]
        if combat_decision and combat_card_sequences_enabled
        else [atomic_action_rule]
        if combat_decision
        else []
    )
    state_update_rules = (
        [
            "After playing a card, using a potion, or viewing information, you will receive a fresh state and may act again if the game still allows actions."
        ]
        if combat_decision
        else []
    )
    combat_evaluation_rules = (
        ["Write a concise COMBAT_EVALUATION using the current visible combat state and any Combat facts shown."]
        if combat_decision
        else []
    )
    card_select_rules = (
        [
            *(
                [
                    "The current decision is card_select. Every complete engine-exported candidate object is preserved without combination-level repetition; no candidate facts have been abbreviated or omitted.",
                    "To select cards, return selected_card_indices as a list of distinct current candidate index integers satisfying min_select and max_select. The benchmark resolves that list to the exact existing engine action.",
                ]
                if contract.interface_preamble_enabled
                else [
                    "The current decision is card_select.",
                    "To select cards, return selected_card_indices as a list of distinct current candidate index integers satisfying min_select and max_select.",
                ]
            ),
            "Return selected_card_indices=[] only to choose the visible skip action when can_skip is true. To choose a non-selection action such as viewing information, omit selected_card_indices and use that action's current action_id and exact selected_action_label.",
            "Use exactly one identity branch: selected_card_indices for a card subset, or action_id plus selected_action_label for a listed non-selection action.",
        ]
        if card_select_indices_enabled
        else []
    )
    shared = [
        *sequence_rules,
        *state_update_rules,
        (
            "View deck/map/pile actions do not advance the game state; use them when deck composition, pile contents, path context, or current position could affect the decision."
            if not contract.interface_preamble_enabled
            else "View deck/map/pile actions do not advance the game state. They are information requests logged separately; use them when deck composition, pile contents, path context, or current position could affect the decision."
        ),
        *(
            [
                "Evidence context, when present, is advisory only. Evidence modules do not score, rank, recommend, or choose actions; current game state and legal_actions are always authoritative."
            ]
            if contract.evidence_context_enabled
            else []
        ),
        *card_select_rules,
    ]
    route_plan_rules = [
        "Only Active route plan JSON is executable route state. A free-form run plan or route intention never permits silent route execution.",
        'For map_select decisions with a selected map action, include a top-level "route_plan_event" object only when creating or changing a route plan. When executing active_route_plan.expected_next, select that node, omit route_plan_event, and still include a top-level reason for the action.',
        'Use route_plan_event {"type":"route_plan_created","new_plan":[{"col":<int>,"row":<int>,"type":"<room type>"}],"reason_tags":["<tag>"],"reason":"<why this route>"} when no active route exists; the new_plan must be complete from the selected legal node to the visible Boss, or visible terminal node when Boss is not present.',
        'Use route_plan_event {"type":"route_plan_updated","old_plan_remaining":[{"col":<int>,"row":<int>,"type":"<room type>"}],"new_plan":[{"col":<int>,"row":<int>,"type":"<room type>"}],"reason_tags":["<tag>"],"reason":"<why not follow the old route>"} before selecting a different route; route_plan_updated must include a non-empty reason.',
        "A route_plan_event new_plan must be complete from the selected legal map node through visible connected nodes to the visible Boss, or to the visible terminal node when Boss is not present. Do not submit a prefix, partial route, or omitted new_plan.",
        "Before outputting a route_plan_event, audit the path against the route map JSON: each adjacent pair in new_plan must follow an edge unless Winged Boots is available.",
        "With Winged Boots, submitted new_plan may contain disconnected forward jumps up to the remaining Winged Boots charges shown in route map constraints; every jump must move forward to a later row.",
        "Every node you include in new_plan must be copied exactly from route map nodes: exact col, row, and type strings such as Unknown or RestSite. Do not infer hidden room types and do not rename RestSite to Rest or Unknown to Event.",
        "The first route_plan_event new_plan node, when present, must match the selected legal map action. Current legal_actions are the final execution authority. If a newly exposed map changes the route, route_plan_updated is required and reason_tags must include the separate string \"map_changed\".",
    ]
    context_patch_schema = (
        '"evidence_refs":[<candidate action_id integers from Evidence context if used>],'
        '"run_plan_patch":{},"run_memory_facts":[],'
        if context_patch_fields_enabled
        else ""
    )
    context_patch_schema_spaced = (
        '"evidence_refs": [<candidate action_id integers from Evidence context if used>], '
        '"run_plan_patch": {}, "run_memory_facts": [], '
        if context_patch_fields_enabled
        else ""
    )
    route_plan_schema = (
        '"route_plan_event":{"type":"<for route plan changes: '
        'route_plan_created|route_plan_updated; omit when executing active plan>",'
        '"new_plan":[{"col":<integer>,"row":<integer>,"type":"<exact current map type>"}],'
        '"reason_tags":["<tag>"],"reason":"<route reason>"}'
    )
    route_plan_schema_spaced = (
        '"route_plan_event": {"type": "<for route plan changes: '
        'route_plan_created|route_plan_updated; omit when executing active plan>", '
        '"new_plan": [{"col": <integer>, "row": <integer>, "type": "<exact current map type>"}], '
        '"reason_tags": ["<tag>"], "reason": "<route reason>"}'
    )
    turn_check_schema = (
        '"turn_check":{"observed_numbers":{"current_hp":<number>,"current_block":<number>,'
        '"current_energy":<number>},'
        '"incoming_damage_calculation":{"parts":["<enemy index: displayed damage x hits = subtotal>"],'
        '"total":<number>},'
        '"selected_action_calculation":{"cost":<number|null>,'
        '"displayed_damage":<number|null>,"displayed_block":<number|null>,'
        '"expected_energy_after":<number|null>,"expected_block_after":<number|null>,'
        '"expected_target_hp_after":<number|null>,'
        '"effect_summary":"<immediate effect of this one selected action>"}|null,'
        '"end_turn_audit":{"selected_action_is_end_turn":true,'
        '"playable_cards_now":<integer|null>,"remaining_hand_will_be_discarded":<true|null>,'
        '"current_energy_carries_to_next_turn":<false|null>,'
        '"expected_hp_after_enemy_attacks":<number|null>,'
        '"expected_energy_next_player_turn":<number|null>,'
        '"expected_block_next_player_turn":<number|null>,'
        '"justification":"<why ending the turn is preferable now>"}|null}'
    )
    turn_check_schema_spaced = (
        '"turn_check": {"observed_numbers": {"current_hp": <number>, "current_block": <number>, '
        '"current_energy": <number>}, '
        '"incoming_damage_calculation": {"parts": ["<enemy index: displayed damage x hits = subtotal>"], '
        '"total": <number>}, '
        '"selected_action_calculation": {"cost": <number|null>, '
        '"displayed_damage": <number|null>, "displayed_block": <number|null>, '
        '"expected_energy_after": <number|null>, "expected_block_after": <number|null>, '
        '"expected_target_hp_after": <number|null>, '
        '"effect_summary": "<immediate effect of this one selected action>"} | null, '
        '"end_turn_audit": {"selected_action_is_end_turn": true, '
        '"playable_cards_now": <integer|null>, "remaining_hand_will_be_discarded": <true|null>, '
        '"current_energy_carries_to_next_turn": <false|null>, '
        '"expected_hp_after_enemy_attacks": <number|null>, '
        '"expected_energy_next_player_turn": <number|null>, '
        '"expected_block_next_player_turn": <number|null>, '
        '"justification": "<why ending the turn is preferable now>"} | null}'
    )
    compact_combat_rules = (
        _compact_combat_header_rules(
            turn_intent_enabled=turn_intent_enabled,
            turn_check_enabled=turn_check_enabled,
            combat_card_sequences_enabled=combat_card_sequences_enabled,
        )
        if turn_intent_enabled or turn_check_enabled
        else []
    )
    turn_intent_schema_fragment = (
        '"turn_intent":"<one short provisional action-order or resource objective; no calculations, predictions, or unobserved consequences; replace after every observed action>",'
        if turn_intent_enabled
        else ""
    )
    turn_intent_schema_fragment_spaced = (
        '"turn_intent": "<one short provisional action-order or resource objective; no calculations, predictions, or unobserved consequences; replace after every observed action>", '
        if turn_intent_enabled
        else ""
    )
    turn_check_schema_fragment = f"{turn_check_schema}," if turn_check_enabled else ""
    turn_check_schema_fragment_spaced = f"{turn_check_schema_spaced}, " if turn_check_enabled else ""

    if prompt_style == "analysis":
        sequence_schema_fragment = (
            '"sequence":[<only when selected action is combat_custom_sequence; otherwise empty>],'
            if combat_card_sequences_enabled
            else ""
        )
        route_plan_schema_fragment = f"{route_plan_schema}," if route_plan_enabled else ""
        action_identity_schema = (
            '"selected_card_indices":[<for card selection only; distinct current candidate index integers; [] only for visible skip; otherwise omit>],'
            '"action_id":<for a listed non-selection action only; otherwise omit>,'
            '"selected_action_label":"<for a listed non-selection action only; exact current label; otherwise omit>",'
            if card_select_indices_enabled
            else '"action_id":<integer>,"selected_action_label":"<copy exact chosen legal action label>",'
        )
        candidate_analysis_schema = (
            '"candidates":[{"card_indices":[<candidate index integers>],'
            '"pros":"<why this subset helps>","cons":"<main risk or cost>"}],'
            if card_select_indices_enabled
            else '"candidates":[{"action_id":<integer>,"label":"<legal action label>",'
            '"pros":"<why it helps>","cons":"<main risk or cost>"}],'
        )
        if turn_intent_enabled or turn_check_enabled:
            schema = (
                "{"
                f"{turn_intent_schema_fragment}"
                f"{turn_check_schema_fragment}"
                f"{action_identity_schema}"
                f"{sequence_schema_fragment}"
                f"{context_patch_schema}"
                '"reason":"<one short current-state reason for the selected action>"}'
            )
        else:
            schema = (
                "{"
                '"situation":"<current objective and main risk>",'
                '"calculations":["<damage/block/energy/path/reward calculation if relevant>"],'
                f"{candidate_analysis_schema}"
                f"{action_identity_schema}"
                f"{sequence_schema_fragment}"
                f'{"\"combat_notes\":[\"<optional factual combat observation, no advice>\"]," if combat_decision else ""}'
                f"{route_plan_schema_fragment}"
                f"{context_patch_schema}"
                '"reason":"<final concise reason>"}'
            )
        return [
            *(
                ["You are playing Slay the Spire 2 through a headless benchmark environment."]
                if contract.interface_preamble_enabled
                else []
            ),
            (
                "Choose exactly one legal card selection through selected_card_indices, or one listed non-selection action through its current action_id and exact selected_action_label."
                if card_select_indices_enabled
                else "Choose exactly one legal action. Use only an action_id from the legal action list."
            ),
            *(
                []
                if card_select_indices_enabled
                else ["Also copy the selected legal action label exactly into selected_action_label; action ids are state-local, so never reuse an action_id from earlier messages without checking the current legal actions."]
            ),
            *shared,
            *(route_plan_rules if route_plan_enabled else []),
            *compact_combat_rules,
            *(
                ["Use the current visible combat state and any Combat facts shown, then choose the first atomic action."]
                if combat_decision
                else ["Reason briefly from the current options."]
            ),
            "Return only valid JSON with this schema:",
            schema,
            *(
                ["Compare path rewards and risks."]
                if contract.decision_type == "map_select"
                else ["Compare deck impact."]
                if contract.decision_type == "card_reward"
                else []
            ),
        ]

    sequence_schema_fragment = (
        '"sequence": [<only when selected action is combat_custom_sequence; otherwise empty>], '
        if combat_card_sequences_enabled
        else ""
    )
    route_plan_schema_fragment = f"{route_plan_schema_spaced}, " if route_plan_enabled else ""
    action_identity_schema_spaced = (
        '"selected_card_indices": [<for card selection only; distinct current candidate index integers; [] only for visible skip; otherwise omit>], '
        '"action_id": <for a listed non-selection action only; otherwise omit>, '
        '"selected_action_label": "<for a listed non-selection action only; exact current label; otherwise omit>", '
        if card_select_indices_enabled
        else '"action_id": <integer>, "selected_action_label": "<copy exact chosen legal action label>", '
    )
    if turn_intent_enabled or turn_check_enabled:
        schema = (
            "{"
            f"{turn_intent_schema_fragment_spaced}"
            f"{turn_check_schema_fragment_spaced}"
            f"{action_identity_schema_spaced}"
            f"{sequence_schema_fragment}"
            f"{context_patch_schema_spaced}"
            '"reason": "<one short current-state reason for the selected action>"}'
        )
    else:
        schema = (
            "{"
            f"{action_identity_schema_spaced}"
            f"{sequence_schema_fragment}"
            f'{"\"combat_notes\": [\"<optional factual combat observation, no advice>\"], " if combat_decision else ""}'
            f"{route_plan_schema_fragment}"
            f"{context_patch_schema_spaced}"
            '"reason": "<short reason>"}'
        )
    return [
        *(
            ["You are playing Slay the Spire 2 through a headless benchmark environment."]
            if contract.interface_preamble_enabled
            else []
        ),
        (
            "Choose exactly one legal card selection through selected_card_indices, or one listed non-selection action through its current action_id and exact selected_action_label."
            if card_select_indices_enabled
            else "Choose exactly one legal action."
        ),
        *shared,
        *(route_plan_rules if route_plan_enabled else []),
        *(
            []
            if card_select_indices_enabled
            else ["Also copy the selected legal action label exactly into selected_action_label; action ids are state-local, so never reuse an action_id from earlier messages without checking the current legal actions."]
        ),
        *compact_combat_rules,
        *combat_evaluation_rules,
        *([] if combat_decision else ["Decide briefly from the current options."]),
        f"End with the final decision on the last line as one JSON object with this schema: {schema}",
    ]


def _compact_combat_header_rules(
    *,
    turn_intent_enabled: bool,
    turn_check_enabled: bool,
    combat_card_sequences_enabled: bool = True,
) -> list[str]:
    rules: list[str] = []
    if turn_intent_enabled:
        rules.extend(
            [
                'When the game state decision is combat_play, include a top-level "turn_intent" field.',
                "turn_intent is one short, provisional turn-level intention. Re-evaluate and replace it after every observed action; never treat a past intent as authoritative state.",
                "Use turn_intent only for the intended action order or resource objective. Do not put arithmetic, lethal claims, future enemy intents, or unobserved consequences in it.",
            ]
        )
    if not turn_check_enabled:
        return rules
    sequence_rule = (
        "Do not write a tentative multi-card script in turn_check. If selecting combat_custom_sequence, justify only that currently legal sequence; otherwise justify only the single selected action."
        if combat_card_sequences_enabled
        else "Do not write a tentative multi-card script in turn_check. Justify only the single selected action."
    )
    rules.extend(
        [
        'When the game state decision is combat_play, include a top-level "turn_check" field.',
        "The turn_check is your own current-state arithmetic scratchpad. The engine does not fill it, validate it, correct it, or use it to choose an action.",
        "Copy current HP, block, and energy from the latest visible state. Independently total incoming attack parts from the raw latest enemy intents. Displayed intent damage already includes current Strength and other engine modifiers; do not apply them a second time.",
        "For a selected non-end action, include selected_action_calculation and omit end_turn_audit (or set it to null). Copy the action's current displayed cost, damage, and block when present, then calculate expected energy, block, and target HP immediately after that one action. Use null when a calculation field does not apply; do not invent an effect.",
        "For end turn, omit selected_action_calculation (or set it to null) and include end_turn_audit. Count currently playable cards from the visible hand, explicitly acknowledge that the remaining hand will be discarded and current energy will not carry over, calculate HP after the currently displayed enemy attacks and block, then state expected energy and block at the next player turn.",
        sequence_rule,
        "Recompute every number from the latest state after each atomic action, especially after draw, enemy death, energy change, random/modal effects, card_select, or last_action_result.",
        "Choose exactly one current legal action. Do not enumerate or calculate every unselected alternative.",
        ]
    )
    return rules


class RandomAgent:
    """Simple non-LLM baseline."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        include_json_state: bool = False,
        prompt_style: PromptStyle = "default",
        short_term_memory_enabled: bool = False,
        short_term_memory_window: int = 8,
        short_term_memory_mode: ShortTermMemoryMode = "action_reason",
        run_summary_enabled: bool = True,
    ) -> None:
        self.rng = random.Random(seed)
        self.include_json_state = include_json_state
        self.prompt_style = prompt_style
        self.memory = (
            ShortTermMemory(short_term_memory_window, mode=short_term_memory_mode)
            if short_term_memory_enabled
            else None
        )
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
        if action.kind == "card_select_space":
            cards = [card for card in state.get("cards") or [] if isinstance(card, dict)]
            min_select = max(1, int(state.get("min_select", 1) or 0))
            max_select = min(int(state.get("max_select", min_select) or min_select), len(cards))
            count = self.rng.randint(min_select, max_select)
            positions = sorted(self.rng.sample(range(len(cards)), count))
            action = materialize_card_select_space_action(
                state,
                action,
                [int(cards[position]["index"]) for position in positions],
            )
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
    short_term_memory_enabled: bool = False
    short_term_memory_window: int = 8
    short_term_memory_mode: ShortTermMemoryMode = "action_reason"
    run_summary_enabled: bool = True
    conversation_mode: ConversationMode = "single_turn"
    prompt_profile: PromptProfile = "default"
    active_policy_owner: str = "llm_policy"
    turn_chat_window: int = 4
    combat_dialogue_history_turns: int = 3
    turn_chat_update_mode: TurnChatUpdateMode = "delta"
    turn_chat_assistant_history: TurnChatAssistantHistory = "compact"
    turn_chat_plan_enabled: bool = False
    turn_check_enabled: bool | None = None
    route_plan_enabled: bool = True
    route_map_prompt_format: RouteMapPromptFormat = "canonical"
    map_route_hint: MapRouteHint = "none"
    context_patch_fields_enabled: bool = True
    fallback_mode: FallbackMode = "error"
    self_consistency_enabled: bool = False
    self_consistency_samples: int = 3
    self_consistency_decisions: tuple[str, ...] | str = ("map_select", "card_reward")
    self_consistency_mode: SelfConsistencyMode = "diagnostic"
    action_prompt_order: ActionPromptOrderMode = "default"
    action_prompt_order_seed: int | None = 0
    combat_card_sequences_enabled: bool = True
    combat_reasoning_pool_enabled: bool = False
    combat_turn_board_enabled: bool = False
    combat_skill_hints_enabled: bool = False
    combat_skill_hints_profile: CombatSkillHintsProfile = "baseline"
    combat_skill_hints_max_hints: int = 4
    combat_skill_lines_enabled: bool = True
    rule_projection_and_recent_changes_enabled: bool = True
    json_response_format: bool = False
    http_backend: HttpBackend = "urllib"
    turn_chat_buffer: TurnChatBuffer = field(init=False)
    combat_dialogue_buffer: CombatDialogueBuffer = field(init=False)
    combat_mdp_buffer: CombatMdpStateBuffer = field(init=False)
    action_prompt_rng: random.Random = field(init=False, repr=False)
    last_action_prompt_order_meta: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    last_action_prompt_base_text: str = field(default="", init=False, repr=False)
    pending_external_feedback: list[str] = field(default_factory=list, init=False)
    pending_dialogue_exchange: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _http_session: Any = field(default=None, init=False, repr=False)
    _http_worker_process: Any = field(default=None, init=False, repr=False)
    _http_worker_connection: Any = field(default=None, init=False, repr=False)
    _http_worker_epoch: int = field(default=0, init=False, repr=False)
    _http_worker_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _last_provider_response_audit: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.conversation_mode not in {
            "single_turn",
            "turn_chat",
            "combat_mdp_state",
            "combat_state_only",
            "combat_dialogue",
        }:
            raise ValueError(f"Unknown conversation mode: {self.conversation_mode}")
        if self.prompt_profile not in {"default", "formal_generalist"}:
            raise ValueError(f"Unknown prompt profile: {self.prompt_profile}")
        if self.turn_chat_update_mode != "delta":
            raise ValueError(f"Unsupported turn_chat_update_mode: {self.turn_chat_update_mode}")
        if self.turn_chat_assistant_history != "compact":
            raise ValueError(f"Unsupported turn_chat_assistant_history: {self.turn_chat_assistant_history}")
        if self.fallback_mode not in {"action", "error"}:
            raise ValueError(f"Unsupported fallback_mode: {self.fallback_mode}")
        if self.turn_chat_window < 1:
            raise ValueError("turn_chat_window must be at least 1")
        if self.combat_dialogue_history_turns < 0:
            raise ValueError("combat_dialogue_history_turns must be non-negative")
        if self.self_consistency_samples < 1:
            raise ValueError("self_consistency_samples must be at least 1")
        if self.self_consistency_mode not in {"diagnostic", "majority_vote"}:
            raise ValueError(f"Unsupported self_consistency_mode: {self.self_consistency_mode}")
        if self.action_prompt_order not in {"default", "shuffle_map"}:
            raise ValueError(f"Unsupported action_prompt_order: {self.action_prompt_order}")
        if self.route_map_prompt_format not in {"route_graph", "canonical"}:
            raise ValueError(f"Unsupported route_map_prompt_format: {self.route_map_prompt_format}")
        if self.combat_skill_hints_profile not in {"baseline", "survival", "targeting", "full"}:
            raise ValueError(f"Unsupported combat_skill_hints_profile: {self.combat_skill_hints_profile}")
        if self.combat_skill_hints_max_hints < 1:
            raise ValueError("combat_skill_hints_max_hints must be at least 1")
        if self.http_backend not in {"urllib", "requests_session", "requests_session_worker"}:
            raise ValueError(f"Unsupported HTTP backend: {self.http_backend}")
        self.self_consistency_decisions = _normalize_self_consistency_decisions(self.self_consistency_decisions)
        self.action_prompt_rng = random.Random(self.action_prompt_order_seed)
        effective_turn_check = self.turn_chat_plan_enabled if self.turn_check_enabled is None else self.turn_check_enabled
        self.turn_check_enabled = bool(effective_turn_check)
        if self.conversation_mode in {
            "turn_chat",
            "combat_mdp_state",
            "combat_state_only",
            "combat_dialogue",
        } and self.short_term_memory_enabled:
            raise ValueError(
                f"{self.conversation_mode} is parallel to short_term_memory_enabled; disable short-term memory when using it"
            )
        if self.prompt_profile == "formal_generalist":
            incompatible = {
                "conversation_mode": self.conversation_mode != "combat_state_only",
                "short_term_memory": self.short_term_memory_enabled,
                "run_summary": self.run_summary_enabled,
                "turn_check": bool(self.turn_check_enabled),
                "route_plan": self.route_plan_enabled,
                "context_patch_fields": self.context_patch_fields_enabled,
                "fallback_mode": self.fallback_mode != "error",
                "self_consistency": self.self_consistency_enabled,
                "combat_card_sequences": self.combat_card_sequences_enabled,
                "combat_reasoning_pool": self.combat_reasoning_pool_enabled,
                "combat_turn_board": self.combat_turn_board_enabled,
                "combat_skill_hints": self.combat_skill_hints_enabled,
                "combat_skill_lines": self.combat_skill_lines_enabled,
                "rule_projection_and_recent_changes": (
                    self.rule_projection_and_recent_changes_enabled
                ),
            }
            enabled = [name for name, present in incompatible.items() if present]
            if enabled:
                raise ValueError(
                    "formal_generalist requires its fixed formal context contract; "
                    f"incompatible settings: {', '.join(enabled)}"
                )
        self.memory = (
            ShortTermMemory(self.short_term_memory_window, mode=self.short_term_memory_mode)
            if self.short_term_memory_enabled
            else None
        )
        self.turn_chat_buffer = TurnChatBuffer(
            max_turns=self.turn_chat_window,
            plan_enabled=self.turn_check_enabled,
            route_plan_enabled=self.route_plan_enabled,
            route_map_prompt_format=self.route_map_prompt_format,
            map_route_hint=self.map_route_hint,
            context_patch_fields_enabled=self.context_patch_fields_enabled,
            combat_card_sequences_enabled=self.combat_card_sequences_enabled,
            combat_skill_lines_enabled=self.combat_skill_lines_enabled,
            rule_projection_and_recent_changes_enabled=self.rule_projection_and_recent_changes_enabled,
        )
        self.combat_dialogue_buffer = CombatDialogueBuffer(
            completed_turns_to_keep=self.combat_dialogue_history_turns,
            reasoning_pool_enabled=self.combat_reasoning_pool_enabled,
            skill_hints_enabled=self.combat_skill_hints_enabled,
            skill_hints_profile=self.combat_skill_hints_profile,
            skill_hints_max_hints=self.combat_skill_hints_max_hints,
            rule_projection_and_recent_changes_enabled=(
                self.rule_projection_and_recent_changes_enabled
            ),
        )
        self.combat_mdp_buffer = CombatMdpStateBuffer(
            reasoning_pool_enabled=self.combat_reasoning_pool_enabled,
            turn_board_enabled=self.combat_turn_board_enabled,
            skill_hints_enabled=self.combat_skill_hints_enabled,
            skill_hints_profile=self.combat_skill_hints_profile,
            skill_hints_max_hints=self.combat_skill_hints_max_hints,
            route_map_prompt_format=self.route_map_prompt_format,
            map_route_hint=self.map_route_hint,
            rule_projection_and_recent_changes_enabled=self.rule_projection_and_recent_changes_enabled,
        )

    def _formal_provider_audit_enabled(self) -> bool:
        return os.environ.get(FORMAL_PROVIDER_AUDIT_ENV) == "1"

    def _begin_model_request(
        self,
        request_audits: list[dict[str, Any]],
        messages: list[dict[str, str]],
    ) -> None:
        _append_model_request_audit(request_audits, messages)
        self._last_provider_response_audit = None

    def _capture_provider_response_body(self, response_body: bytes) -> str:
        raw_body = response_body.decode("utf-8")
        if self._formal_provider_audit_enabled():
            self._last_provider_response_audit = {
                "provider_response_body": raw_body,
                "provider_response_sha256": hashlib.sha256(response_body).hexdigest(),
            }
        return raw_body

    def _capture_provider_http_status(self, status: Any) -> None:
        if not self._formal_provider_audit_enabled():
            return
        if not isinstance(self._last_provider_response_audit, dict):
            raise ValueError("formal provider response body was not preserved before HTTP status")
        if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
            raise ValueError(f"formal provider response has invalid HTTP status: {status!r}")
        self._last_provider_response_audit["provider_http_status"] = status

    def _complete_model_request(
        self,
        request_audits: list[dict[str, Any]],
        response: dict[str, Any],
        *,
        elapsed_sec: float = 0.0,
    ) -> None:
        if not self._formal_provider_audit_enabled():
            return
        if not request_audits or not isinstance(self._last_provider_response_audit, dict):
            raise ValueError("formal provider response body was not preserved")
        request = request_audits[-1]
        request.update(self._last_provider_response_audit)
        finish_reason = _provider_finish_reason(response)
        response_text = _provider_response_text(response)
        completed_normally = (
            finish_reason == "stop"
            and isinstance(response_text, str)
            and bool(response_text.strip())
        )
        request.update(
            {
                "provider_response_status": "succeeded" if completed_normally else "model_protocol_error",
                "provider_finish_reason": finish_reason,
                "provider_metadata": _provider_response_metadata(response),
                "provider_usage": response.get("usage") if isinstance(response.get("usage"), dict) else None,
                "provider_latency_sec": float(elapsed_sec),
            }
        )
        if finish_reason != "stop":
            detail = "missing" if finish_reason is None else str(finish_reason)
            request["provider_error"] = f"finish_reason={detail}"
            raise ValueError(f"LLM response did not complete normally (finish_reason={detail})")
        if not isinstance(response_text, str) or not response_text.strip():
            request["provider_error"] = "response content is not non-empty text"
            raise ValueError("LLM response content is not non-empty text")

    def _fail_model_request(
        self,
        request_audits: list[dict[str, Any]],
        exc: Exception,
        *,
        elapsed_sec: float = 0.0,
    ) -> None:
        if not self._formal_provider_audit_enabled() or not request_audits:
            return
        request = request_audits[-1]
        if isinstance(self._last_provider_response_audit, dict):
            request.update(self._last_provider_response_audit)
        if request.get("provider_response_status") is None:
            request["provider_response_status"] = (
                "transport_error" if _is_transport_error(exc) else "model_protocol_error"
            )
        request.setdefault("provider_metadata", {})
        request.setdefault("provider_finish_reason", None)
        request.setdefault("provider_usage", None)
        request["provider_latency_sec"] = float(elapsed_sec)
        request["provider_error"] = _one_line(exc)[:500]

    @staticmethod
    def _final_provider_meta(request_audits: list[dict[str, Any]]) -> dict[str, Any]:
        if not request_audits or request_audits[-1].get("provider_response_status") != "succeeded":
            return {}
        final = request_audits[-1]
        return {
            "provider_metadata": final.get("provider_metadata") or {},
            "provider_finish_reason": final.get("provider_finish_reason"),
        }

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        if self.prompt_profile == "formal_generalist":
            return build_formal_generalist_user_message(
                state,
                legal_actions,
                prompt_style=self.prompt_style,
                combat_card_sequences_enabled=self.combat_card_sequences_enabled,
            )
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
        if self.conversation_mode == "combat_dialogue" and turn_chat_key(state) is not None:
            messages = self.combat_dialogue_buffer.messages_for_decision(
                state,
                legal_actions,
                prompt_style=self.prompt_style,
                run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
                turn_check_enabled=bool(self.turn_check_enabled),
                route_plan_enabled=self.route_plan_enabled,
                context_patch_fields_enabled=self.context_patch_fields_enabled,
                combat_card_sequences_enabled=self.combat_card_sequences_enabled,
                record_user=False,
            )
            return _messages_preview(messages)
        if self.conversation_mode == "combat_state_only" and turn_chat_key(state) is not None:
            return build_combat_current_state_user_message(
                state,
                legal_actions,
                prompt_style=self.prompt_style,
                run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
                turn_intent_enabled=self.combat_reasoning_pool_enabled,
                turn_check_enabled=bool(self.turn_check_enabled),
                route_plan_enabled=self.route_plan_enabled,
                context_patch_fields_enabled=self.context_patch_fields_enabled,
                combat_card_sequences_enabled=self.combat_card_sequences_enabled,
                skill_hints_enabled=self.combat_skill_hints_enabled,
                skill_hints_profile=self.combat_skill_hints_profile,
                skill_hints_max_hints=self.combat_skill_hints_max_hints,
                rule_projection_enabled=self.rule_projection_and_recent_changes_enabled,
            )
        if self.conversation_mode == "combat_mdp_state":
            return self.combat_mdp_buffer.build_prompt(
                state,
                legal_actions,
                prompt_style=self.prompt_style,
                run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
                turn_check_enabled=bool(self.turn_check_enabled),
                route_plan_enabled=self.route_plan_enabled,
                context_patch_fields_enabled=self.context_patch_fields_enabled,
                combat_card_sequences_enabled=self.combat_card_sequences_enabled,
            )

        display_actions = self._legal_actions_for_prompt(state, legal_actions)
        prompt = build_llm_prompt(
            state,
            display_actions,
            include_json=self.include_json_state,
            prompt_style=self.prompt_style,
            memory_text=self.memory.render() if self.memory else "",
            run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
            route_plan_enabled=self.route_plan_enabled,
            route_map_prompt_format=self.route_map_prompt_format,
            map_route_hint=self.map_route_hint,
            context_patch_fields_enabled=self.context_patch_fields_enabled,
            combat_card_sequences_enabled=self.combat_card_sequences_enabled,
            combat_skill_lines_enabled=self.combat_skill_lines_enabled,
            rule_projection_and_recent_changes_enabled=self.rule_projection_and_recent_changes_enabled,
        )
        self.last_action_prompt_base_text = prompt
        return prompt

    def _legal_actions_for_prompt(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
    ) -> list[LegalAction]:
        display_actions = _legal_actions_for_prompt(
            state,
            legal_actions,
            mode=self.action_prompt_order,
            rng=self.action_prompt_rng,
        )
        self.last_action_prompt_order_meta = _action_prompt_order_meta(
            state,
            legal_actions,
            display_actions,
            mode=self.action_prompt_order,
        )
        return display_actions

    def _action_prompt_order_meta_for_prompt(self, prompt: str) -> dict[str, Any]:
        if not self.last_action_prompt_order_meta:
            return {}
        if prompt != self.last_action_prompt_base_text and not prompt.startswith(self.last_action_prompt_base_text):
            return {}
        return dict(self.last_action_prompt_order_meta)

    def _context_trace_meta_for_state(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        dialogue_history_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.prompt_profile == "formal_generalist":
            variant = "combat_state_only"
            history = None
        elif self.conversation_mode == "combat_mdp_state":
            variant = "combat_mdp_state"
            history = self.combat_mdp_buffer.history_audit(state)
        elif self.conversation_mode == "combat_dialogue":
            variant = "combat_dialogue"
            history = dialogue_history_meta
        elif self.conversation_mode == "combat_state_only":
            variant = "combat_state_only"
            history = None
        else:
            variant = self.conversation_mode
            history = None
        return _context_trace_meta(
            state,
            legal_actions,
            variant=variant,
            history=history,
            active_policy_owner=self.active_policy_owner,
            combat_card_sequences_enabled=self.combat_card_sequences_enabled,
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
            if prompt is not None and prompt != self.build_prompt(state, legal_actions):
                raise ValueError(
                    "turn_chat cannot safely apply an external prompt override; "
                    "refusing to discard or flatten caller-provided context"
                )
            return self._choose_turn_chat(state, legal_actions)

        dialogue_active = bool(
            self.conversation_mode == "combat_dialogue"
            and turn_chat_key(state) is not None
        )
        dialogue_history_meta: dict[str, Any] | None = None
        dialogue_user_message: str | None = None
        if dialogue_active:
            self.pending_dialogue_exchange = None
            expected_prompt = self.build_prompt(state, legal_actions)
            if prompt is not None and prompt != expected_prompt:
                raise ValueError(
                    "combat_dialogue cannot safely apply an external prompt override; "
                    "refusing to discard or flatten caller-provided context"
                )
            self.combat_dialogue_buffer._activate(state)
            dialogue_history_meta = self.combat_dialogue_buffer.history_audit()
            messages = self.combat_dialogue_buffer.messages_for_decision(
                state,
                legal_actions,
                prompt_style=self.prompt_style,
                run_summary_text=build_run_summary(state) if self.run_summary_enabled else "",
                turn_check_enabled=bool(self.turn_check_enabled),
                route_plan_enabled=self.route_plan_enabled,
                context_patch_fields_enabled=self.context_patch_fields_enabled,
                combat_card_sequences_enabled=self.combat_card_sequences_enabled,
                record_user=False,
            )
            dialogue_user_message = messages[-1]["content"]
            prompt = _messages_preview(messages)
            rule_audit_prompt = messages[-1]["content"]
        else:
            prompt = prompt if prompt is not None else self.build_prompt(state, legal_actions)
            if self.prompt_profile == "formal_generalist":
                messages = [
                    {"role": "system", "content": build_formal_generalist_system_message()},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "system", "content": _default_policy_system_message()},
                    {"role": "user", "content": prompt},
                ]
            rule_audit_prompt = (
                _messages_preview(messages)
                if self.prompt_profile == "formal_generalist"
                else prompt
            )

        if self.prompt_profile == "formal_generalist":
            rule_meta = full_game_rules_meta(rule_audit_prompt)
            rule_meta.update(
                {
                    "rule_projection_and_recent_changes_enabled": False,
                    "recent_state_changes_delivery": "omitted",
                }
            )
        else:
            rule_meta = game_rules_meta(
                state,
                prompt=rule_audit_prompt,
                projection_enabled=self.rule_projection_and_recent_changes_enabled,
            )
            rule_meta.update(
                {
                    "rule_projection_and_recent_changes_enabled": self.rule_projection_and_recent_changes_enabled,
                    "recent_state_changes_delivery": (
                        "prompt" if self.rule_projection_and_recent_changes_enabled else "omitted"
                    ),
                }
            )
        if not rule_meta.get("game_rule_exposure_valid"):
            raise ValueError(
                "final decision prompt does not contain exactly the expected game-rule block; "
                "refusing to call the model"
            )
        action_prompt_order_meta = self._action_prompt_order_meta_for_prompt(prompt)
        response_contract = build_prompt_contract(
            state,
            legal_actions,
            route_plan_enabled=self.route_plan_enabled,
            context_patch_fields_enabled=self.context_patch_fields_enabled,
            evidence_context_enabled=self.prompt_profile != "formal_generalist",
            combat_card_sequences_enabled=self.combat_card_sequences_enabled,
            turn_intent_enabled=bool(
                self.conversation_mode
                in {"combat_mdp_state", "combat_state_only", "combat_dialogue"}
                and self.combat_reasoning_pool_enabled
            ),
            turn_check_enabled=bool(
                self.turn_check_enabled
                and self.conversation_mode
                in {"combat_mdp_state", "combat_state_only", "combat_dialogue"}
            ),
        )
        if self.pending_external_feedback:
            messages.append({"role": "user", "content": "\n".join(self.pending_external_feedback)})
            self.pending_external_feedback.clear()

        last_error: str | None = None
        previous_errors: list[str] = []
        previous_raw_responses: list[str] = []
        request_audits: list[dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            started = time.time()
            text: str | None = None
            try:
                self._begin_model_request(request_audits, messages)
                response = self._chat(messages)
                elapsed = time.time() - started
                self._complete_model_request(request_audits, response, elapsed_sec=elapsed)
                text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(text)
                _validate_turn_check_contract_response(parsed, response_contract)
                action, action_remap, resolved_from_label = _resolve_model_action(parsed, legal_actions, state=state)
                _validate_turn_check_action_response(parsed, response_contract, action)
                parsed = _normalized_resolved_parsed(parsed, action)
                usage = response.get("usage")
                resolution_meta = _action_resolution_meta(action, action_remap, resolved_from_label)
                self_consistency = self._sample_self_consistency(
                    state=state,
                    legal_actions=legal_actions,
                    messages=messages,
                    primary_action=action,
                    primary_parsed=parsed,
                    primary_raw_response=text,
                    primary_usage=usage,
                    primary_resolution_meta=resolution_meta,
                    request_audits=request_audits,
                    response_contract=response_contract,
                )
                if self_consistency is not None:
                    action = self_consistency["selected_action"]
                    parsed = self_consistency["selected_parsed"]
                    text = self_consistency["selected_raw_response"]
                    usage = self_consistency["selected_usage"]
                    resolution_meta = self_consistency["selected_resolution_meta"]

                if dialogue_active:
                    self.pending_dialogue_exchange = {
                        "turn_key": turn_chat_key(state),
                        "user_message": dialogue_user_message,
                        "raw_response": text,
                    }
                meta = {
                    "agent": "openai_compat",
                    "model": self.model,
                    "conversation_mode": self.conversation_mode,
                    "message_count": len(messages),
                    "conversation_prompt_chars": _messages_char_count(messages),
                    "attempt": attempt,
                    "elapsed_sec": elapsed,
                    "json_response_format": self.json_response_format,
                    "raw_response": text,
                    "parsed": parsed,
                    "usage": usage,
                    "prompt_chars": len(prompt),
                    "model_request_audits": request_audits,
                    **rule_meta,
                    **action_prompt_order_meta,
                    **resolution_meta,
                    **self._final_provider_meta(request_audits),
                    **self._context_trace_meta_for_state(
                        state,
                        legal_actions,
                        dialogue_history_meta=dialogue_history_meta,
                    ),
                }
                if self_consistency is not None:
                    meta["self_consistency"] = self_consistency["summary"]
                if self.conversation_mode == "combat_mdp_state":
                    meta["combat_mdp_turn_actions"] = self.combat_mdp_buffer.actions_count(state)
                if previous_errors:
                    meta["previous_errors"] = list(previous_errors)
                if previous_raw_responses:
                    meta["previous_raw_responses"] = list(previous_raw_responses)
                return action, meta
            except Exception as exc:  # noqa: BLE001 - log and retry with stricter correction
                self._fail_model_request(request_audits, exc, elapsed_sec=time.time() - started)
                last_error = str(exc)
                previous_errors.append(last_error)
                if text is not None:
                    previous_raw_responses.append(text)
                if _is_transport_error(exc):
                    if attempt < self.max_retries:
                        time.sleep(_transport_retry_delay(attempt, exc))
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": _model_action_correction_text(
                            last_error,
                            contract=response_contract,
                            legal_actions=legal_actions,
                        ),
                    }
                )

        if self.fallback_mode == "error":
            raise ModelActionError(
                _model_action_error_message(last_error, self.max_retries, previous_raw_responses),
                audit_meta={
                    "model": self.model,
                    "json_response_format": self.json_response_format,
                    "model_request_audits": request_audits,
                    **rule_meta,
                },
            )

        # Optional deterministic fallback keeps exploratory benchmark batches
        # moving while still recording the invalid model behavior.
        fallback = first_non_view_action(legal_actions)
        return fallback, {
            "agent": "openai_compat",
            "model": self.model,
            "conversation_mode": self.conversation_mode,
            "message_count": len(messages),
            "conversation_prompt_chars": _messages_char_count(messages),
            "json_response_format": self.json_response_format,
            "fallback": True,
            "error": last_error,
            "fallback_action_id": fallback.action_id,
            "prompt_chars": len(prompt),
            "model_request_audits": request_audits,
            **rule_meta,
            **action_prompt_order_meta,
            **self._context_trace_meta_for_state(
                state,
                legal_actions,
                dialogue_history_meta=dialogue_history_meta,
            ),
        }

    def _sample_self_consistency(
        self,
        *,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        messages: list[dict[str, str]],
        primary_action: LegalAction,
        primary_parsed: dict[str, Any],
        primary_raw_response: str,
        primary_usage: dict[str, Any] | None,
        primary_resolution_meta: dict[str, Any],
        request_audits: list[dict[str, Any]],
        response_contract: PromptContract,
    ) -> dict[str, Any] | None:
        decision = str(state.get("decision") or "")
        if (
            not self.self_consistency_enabled
            or self.self_consistency_samples <= 1
            or decision not in self.self_consistency_decisions
        ):
            return None

        started = time.time()
        samples: list[dict[str, Any]] = [
            {
                "entry": _self_consistency_sample_entry(0, primary_action, primary_parsed, primary=True),
                "action": primary_action,
                "parsed": primary_parsed,
                "raw_response": primary_raw_response,
                "usage": primary_usage,
                "resolution_meta": primary_resolution_meta,
            }
        ]
        errors: list[dict[str, Any]] = []

        for sample_index in range(1, self.self_consistency_samples):
            raw_text: str | None = None
            request_started = time.time()
            try:
                self._begin_model_request(request_audits, messages)
                response = self._chat(messages)
                self._complete_model_request(
                    request_audits,
                    response,
                    elapsed_sec=time.time() - request_started,
                )
                raw_text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(raw_text)
                _validate_turn_check_contract_response(parsed, response_contract)
                action, action_remap, resolved_from_label = _resolve_model_action(parsed, legal_actions, state=state)
                _validate_turn_check_action_response(parsed, response_contract, action)
                parsed = _normalized_resolved_parsed(parsed, action)
                samples.append(
                    {
                        "entry": _self_consistency_sample_entry(sample_index, action, parsed, primary=False),
                        "action": action,
                        "parsed": parsed,
                        "raw_response": raw_text,
                        "usage": response.get("usage"),
                        "resolution_meta": _action_resolution_meta(action, action_remap, resolved_from_label),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic samples should not block the run
                self._fail_model_request(
                    request_audits,
                    exc,
                    elapsed_sec=time.time() - request_started,
                )
                error: dict[str, Any] = {
                    "sample_index": sample_index,
                    "error": _one_line(exc)[:300],
                }
                if raw_text:
                    error["raw_response_excerpt"] = _one_line(raw_text)[:500]
                errors.append(error)

        entries = [sample["entry"] for sample in samples]
        vote_counter: Counter[int] = Counter(int(entry["action_id"]) for entry in entries)
        first_seen = {int(entry["action_id"]): index for index, entry in enumerate(entries)}
        primary_action_id = primary_action.action_id
        ordered_action_ids = sorted(
            vote_counter,
            key=lambda action_id: (
                -vote_counter[action_id],
                0 if action_id == primary_action_id else 1,
                first_seen[action_id],
            ),
        )
        vote_counts = [
            {
                "action_id": action_id,
                "label": next((entry["label"] for entry in entries if entry["action_id"] == action_id), ""),
                "kind": next((entry["kind"] for entry in entries if entry["action_id"] == action_id), ""),
                "count": vote_counter[action_id],
            }
            for action_id in ordered_action_ids
        ]
        valid_sample_count = len(samples)
        top_vote = vote_counts[0] if vote_counts else {}
        top_action_id = top_vote.get("action_id")
        top_vote_count = int(top_vote.get("count", 0) or 0)
        majority_found = top_vote_count > valid_sample_count / 2 if valid_sample_count else False
        tied_top_action_ids = [action_id for action_id, count in vote_counter.items() if count == top_vote_count]
        vote_tie = len(tied_top_action_ids) > 1

        selected_sample = samples[0]
        majority_vote_applied = False
        majority_vote_allowed = decision in SELF_CONSISTENCY_MAJORITY_ALLOWED_DECISIONS
        if (
            self.self_consistency_mode == "majority_vote"
            and majority_vote_allowed
            and majority_found
            and top_action_id != primary_action_id
        ):
            selected_sample = next(sample for sample in samples if sample["entry"]["action_id"] == top_action_id)
            majority_vote_applied = True

        selected_action = selected_sample["action"]
        summary = {
            "mode": self.self_consistency_mode,
            "decision": decision,
            "samples_requested": self.self_consistency_samples,
            "samples_completed": valid_sample_count,
            "invalid_samples": len(errors),
            "primary_action_id": primary_action_id,
            "selected_action_id": selected_action.action_id,
            "top_action_id": top_action_id,
            "top_vote_count": top_vote_count,
            "majority_found": majority_found,
            "majority_vote_allowed": majority_vote_allowed,
            "majority_vote_applied": majority_vote_applied,
            "vote_tie": vote_tie,
            "disagreement": len(vote_counts) > 1 or top_action_id != primary_action.action_id,
            "vote_counts": vote_counts,
            "samples": entries,
            "errors": errors[:5],
            "elapsed_sec": time.time() - started,
        }
        return {
            "selected_action": selected_action,
            "selected_parsed": selected_sample["parsed"],
            "selected_raw_response": selected_sample["raw_response"],
            "selected_usage": selected_sample["usage"],
            "selected_resolution_meta": selected_sample["resolution_meta"],
            "summary": summary,
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
        decision_prompt = _messages_preview(messages)
        rule_meta = game_rules_meta(
            state,
            prompt=decision_prompt,
            projection_enabled=self.rule_projection_and_recent_changes_enabled,
        )
        rule_meta.update(
            {
                "rule_projection_and_recent_changes_enabled": self.rule_projection_and_recent_changes_enabled,
                "recent_state_changes_delivery": (
                    "prompt" if self.rule_projection_and_recent_changes_enabled else "omitted"
                ),
            }
        )
        if not rule_meta.get("game_rule_exposure_valid"):
            raise ValueError(
                "final turn-chat decision prompt does not contain exactly the expected game-rule block; "
                "refusing to call the model"
            )
        turn_key = turn_chat_key(state)
        history_turns = self.turn_chat_buffer.history_turns()
        response_contract = build_prompt_contract(
            state,
            legal_actions,
            route_plan_enabled=False,
            context_patch_fields_enabled=self.context_patch_fields_enabled,
            combat_card_sequences_enabled=self.combat_card_sequences_enabled,
            turn_check_enabled=self.turn_check_enabled,
        )

        last_error: str | None = None
        previous_errors: list[str] = []
        previous_raw_responses: list[str] = []
        request_audits: list[dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            started = time.time()
            text: str | None = None
            try:
                self._begin_model_request(request_audits, messages)
                response = self._chat(messages)
                elapsed = time.time() - started
                self._complete_model_request(request_audits, response, elapsed_sec=elapsed)
                text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(text)
                _validate_turn_check_contract_response(parsed, response_contract)
                action, action_remap, resolved_from_label = _resolve_model_action(parsed, legal_actions, state=state)
                _validate_turn_check_action_response(parsed, response_contract, action)
                parsed = _normalized_resolved_parsed(parsed, action)
                self.turn_chat_buffer.remember_assistant(action, parsed)
                meta = {
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
                    "model_request_audits": request_audits,
                    **self._final_provider_meta(request_audits),
                    **rule_meta,
                    **_action_resolution_meta(action, action_remap, resolved_from_label),
                    **_context_trace_meta(
                        state,
                        legal_actions,
                        variant="turn_chat",
                        history={
                            "history_message_count": max(0, len(messages) - 2),
                            "history_completed_turns": 0,
                            "history_rounds": [turn_key[-1]] if turn_key else [],
                            "history_chars": max(
                                0,
                                _messages_char_count(messages)
                                - len(messages[0].get("content", ""))
                                - len(messages[-1].get("content", "")),
                            ),
                            "history_token_estimate": max(
                                0,
                                (
                                    _messages_char_count(messages)
                                    - len(messages[0].get("content", ""))
                                    - len(messages[-1].get("content", ""))
                                    + 3
                                )
                                // 4,
                            ),
                            "history_token_usage": {
                                "kind": "estimated_chars_div_4",
                                "tokens": max(
                                    0,
                                    (
                                        _messages_char_count(messages)
                                        - len(messages[0].get("content", ""))
                                        - len(messages[-1].get("content", ""))
                                        + 3
                                    )
                                    // 4,
                                ),
                            },
                            "assistant_history": "managed",
                        },
                        active_policy_owner=self.active_policy_owner,
                        combat_card_sequences_enabled=self.combat_card_sequences_enabled,
                    ),
                }
                if previous_errors:
                    meta["previous_errors"] = list(previous_errors)
                if previous_raw_responses:
                    meta["previous_raw_responses"] = list(previous_raw_responses)
                return action, meta
            except Exception as exc:  # noqa: BLE001 - log and retry with stricter correction
                self._fail_model_request(request_audits, exc, elapsed_sec=time.time() - started)
                last_error = str(exc)
                previous_errors.append(last_error)
                if text is not None:
                    previous_raw_responses.append(text)
                if _is_transport_error(exc):
                    if attempt < self.max_retries:
                        time.sleep(_transport_retry_delay(attempt, exc))
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": _model_action_correction_text(
                            last_error,
                            contract=response_contract,
                            legal_actions=legal_actions,
                        ),
                    }
                )

        if self.fallback_mode == "error":
            raise ModelActionError(
                _model_action_error_message(last_error, self.max_retries),
                audit_meta={
                    "model": self.model,
                    "json_response_format": self.json_response_format,
                    "model_request_audits": request_audits,
                    **rule_meta,
                },
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
            "model_request_audits": request_audits,
            **rule_meta,
        }

    def reset_episode(self) -> None:
        if self.memory:
            self.memory.reset()
        self.turn_chat_buffer.reset()
        self.combat_dialogue_buffer.reset()
        self.combat_mdp_buffer.reset()
        self.pending_external_feedback.clear()
        self.pending_dialogue_exchange = None

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
        if self.conversation_mode == "combat_dialogue":
            pending = self.pending_dialogue_exchange
            if (
                isinstance(pending, dict)
                and pending.get("turn_key") == turn_chat_key(state)
                and isinstance(pending.get("user_message"), str)
                and isinstance(pending.get("raw_response"), str)
            ):
                self.combat_dialogue_buffer.remember_accepted_exchange(
                    state,
                    pending["user_message"],
                    pending["raw_response"],
                )
            self.pending_dialogue_exchange = None
            self.combat_dialogue_buffer.after_transition(action, next_state)
        if self.conversation_mode == "combat_mdp_state":
            self.combat_mdp_buffer.record_transition(state, action, meta, next_state)

    def configure_http_backend(self, backend: str) -> None:
        if backend not in {"urllib", "requests_session", "requests_session_worker"}:
            raise ValueError(f"Unsupported HTTP backend: {backend}")
        if backend != self.http_backend:
            self.close_http_transport()
            self.http_backend = backend  # type: ignore[assignment]

    def close_http_transport(self) -> None:
        session = self._http_session
        self._http_session = None
        if session is not None:
            session.close()
        self._close_http_worker()

    def _close_http_worker(self) -> None:
        with self._http_worker_lock:
            process = self._http_worker_process
            connection = self._http_worker_connection
            self._http_worker_process = None
            self._http_worker_connection = None
            self._http_worker_epoch += 1
        if connection is not None:
            try:
                if process is not None and process.is_alive():
                    connection.send({"type": "close"})
                    connection.poll(0.5)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive() and hasattr(process, "kill"):  # pragma: no cover - defensive fallback
                process.kill()
                process.join(timeout=2.0)
            process.close()
        if connection is not None:
            connection.close()

    def _force_abort_http_worker(
        self,
        process: Any = None,
        connection: Any = None,
        epoch: int | None = None,
    ) -> None:
        """Detach immediately at a hard deadline and reap outside the caller."""

        with self._http_worker_lock:
            if process is None and connection is None:
                process = self._http_worker_process
                connection = self._http_worker_connection
                self._http_worker_process = None
                self._http_worker_connection = None
                self._http_worker_epoch += 1
            elif (
                self._http_worker_process is process
                and self._http_worker_connection is connection
                and epoch == self._http_worker_epoch
            ):
                self._http_worker_process = None
                self._http_worker_connection = None
                self._http_worker_epoch += 1
        if process is not None:
            try:
                process.terminate()
            except (OSError, ValueError):
                pass
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

        def reap() -> None:
            if process is None:
                return
            try:
                process.join(timeout=2.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=2.0)
                if not process.is_alive():
                    process.close()
            except (OSError, ValueError):
                return

        if process is not None:
            threading.Thread(target=reap, name="sts2-http-reaper", daemon=True).start()

    def _ensure_http_worker(self) -> tuple[Any, Any, int]:
        stale: tuple[Any, Any, int] | None = None
        with self._http_worker_lock:
            process = self._http_worker_process
            connection = self._http_worker_connection
            if process is not None and process.is_alive() and connection is not None:
                return process, connection, self._http_worker_epoch
            if process is not None or connection is not None:
                stale = (process, connection, self._http_worker_epoch)
                self._http_worker_process = None
                self._http_worker_connection = None
                self._http_worker_epoch += 1
            epoch = self._http_worker_epoch
        if stale is not None:
            self._force_abort_http_worker(*stale)
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        candidate_process = context.Process(
            target=_requests_session_worker,
            args=(child_connection,),
            name="sts2-http-session",
            daemon=True,
        )
        candidate_process.start()
        child_connection.close()
        with self._http_worker_lock:
            if epoch != self._http_worker_epoch:
                superseded = True
                existing = None
            else:
                existing_process = self._http_worker_process
                existing_connection = self._http_worker_connection
                if existing_process is not None and existing_process.is_alive() and existing_connection is not None:
                    superseded = False
                    existing = (existing_process, existing_connection, self._http_worker_epoch)
                else:
                    self._http_worker_process = candidate_process
                    self._http_worker_connection = parent_connection
                    return candidate_process, parent_connection, epoch
        self._force_abort_http_worker(candidate_process, parent_connection, epoch)
        if superseded:
            raise urllib.error.URLError("HTTP worker startup was superseded by a request deadline")
        assert existing is not None
        return existing

    def _chat_request_parts(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, bytes, dict[str, str]]:
        base = self.base_url.rstrip("/")
        url = base + "/chat/completions"
        if not base.endswith("/v1"):
            url = base + "/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.json_response_format:
            payload["response_format"] = {"type": "json_object"}
        return (
            url,
            json.dumps(payload).encode("utf-8"),
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Mozilla/5.0 (compatible; sts2-agent-bench/1.0)",
            },
        )

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        url, body, headers = self._chat_request_parts(messages)
        timeout = self.timeout if timeout_override is None else min(self.timeout, float(timeout_override))
        if timeout <= 0:
            raise TimeoutError("LLM request timeout budget was exhausted before the request started")
        if self.http_backend == "requests_session":
            return self._chat_requests_session(url=url, body=body, headers=headers, timeout=timeout)
        if self.http_backend == "requests_session_worker":
            return self._chat_requests_session_worker(url=url, body=body, headers=headers, timeout=timeout)
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_body = self._capture_provider_response_body(response.read())
                response_status = getattr(response, "status", None)
                if response_status is None and hasattr(response, "getcode"):
                    response_status = response.getcode()
                self._capture_provider_http_status(response_status)
                return json.loads(raw_body)
        except urllib.error.HTTPError as exc:
            body = self._capture_provider_response_body(exc.read())
            self._capture_provider_http_status(exc.code)
            raise RuntimeError(f"LLM HTTP {exc.code}: {body[:500]}") from exc

    def _chat_requests_session(
        self,
        *,
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util import Timeout as Urllib3Timeout
        except ImportError as exc:  # pragma: no cover - exercised only in an incomplete environment
            raise RuntimeError(
                "requests_session HTTP backend requires requests>=2.32; install sts2_bench/requirements.txt"
            ) from exc

        if self._http_session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, pool_block=True, max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._http_session = session
        try:
            with self._http_session.post(
                url,
                data=body,
                headers=headers,
                timeout=Urllib3Timeout(total=timeout),
            ) as response:
                response_body = response.content
                raw_body = self._capture_provider_response_body(response_body)
                self._capture_provider_http_status(response.status_code)
                if response.status_code >= 400:
                    raise RuntimeError(f"LLM HTTP {response.status_code}: {raw_body[:500]}")
                return json.loads(raw_body)
        except requests.exceptions.RequestException as exc:
            raise urllib.error.URLError(exc) from exc

    def _chat_requests_session_worker(
        self,
        *,
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        dispatch_done = threading.Event()
        dispatch_cancelled = threading.Event()
        dispatch_result: dict[str, Any] = {}

        def dispatch() -> None:
            try:
                process, connection, epoch = self._ensure_http_worker()
                dispatch_result["process"] = process
                dispatch_result["connection"] = connection
                dispatch_result["epoch"] = epoch
                if dispatch_cancelled.is_set():
                    self._force_abort_http_worker(process, connection, epoch)
                    return
                connection.send(
                    {
                        "type": "request",
                        "url": url,
                        "body": body,
                        "headers": headers,
                        "timeout": timeout,
                    }
                )
            except BaseException as exc:  # captured and re-raised on the caller thread
                dispatch_result["error"] = exc
            finally:
                dispatch_done.set()

        threading.Thread(target=dispatch, name="sts2-http-dispatch", daemon=True).start()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not dispatch_done.wait(remaining):
            dispatch_cancelled.set()
            self._force_abort_http_worker()
            raise TimeoutError(f"LLM request exceeded hard wall-clock timeout of {timeout:.3f}s")
        dispatch_error = dispatch_result.get("error")
        if dispatch_error is not None:
            self._force_abort_http_worker()
            if isinstance(dispatch_error, (BrokenPipeError, EOFError, OSError)):
                raise urllib.error.URLError(dispatch_error) from dispatch_error
            raise dispatch_error
        connection = dispatch_result.get("connection")
        process = dispatch_result.get("process")
        epoch = dispatch_result.get("epoch")
        if connection is None:
            self._force_abort_http_worker()
            raise urllib.error.URLError("HTTP worker dispatch ended without a connection")
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not connection.poll(remaining):
                self._force_abort_http_worker(process, connection, epoch)
                raise TimeoutError(f"LLM request exceeded hard wall-clock timeout of {timeout:.3f}s")
            result = connection.recv()
        except TimeoutError:
            raise
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._force_abort_http_worker(process, connection, epoch)
            raise urllib.error.URLError(exc) from exc
        status = result.get("status")
        response_body = result.get("response_body")
        if status in {"ok", "http_error", "json_error"}:
            if not isinstance(response_body, (bytes, bytearray)):
                raise RuntimeError("LLM HTTP worker omitted the raw provider response body")
            raw_body = self._capture_provider_response_body(bytes(response_body))
            self._capture_provider_http_status(result.get("code"))
        else:
            raw_body = None
        if status == "ok":
            response = result.get("response")
            if not isinstance(response, dict):
                raise RuntimeError("LLM worker returned a non-object response envelope")
            return response
        if status == "http_error":
            code = result.get("code")
            raise RuntimeError(f"LLM HTTP {code}: {(raw_body or '')[:500]}")
        if status == "json_error":
            raise json.JSONDecodeError(
                str(result.get("message") or "Invalid JSON"),
                str(result.get("document") or raw_body or ""),
                int(result.get("position") or 0),
            )
        if status == "request_error":
            raise urllib.error.URLError(str(result.get("message") or "HTTP worker request failed"))
        raise RuntimeError(
            "LLM HTTP worker failed"
            + (f" ({result.get('error_type')})" if result.get("error_type") else "")
            + f": {result.get('message', '')}"
        )


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
    *,
    state: dict[str, Any] | None = None,
) -> tuple[LegalAction, int | None, bool]:
    selected_card_action = _card_select_action_from_indices(parsed, legal_actions, state=state)
    if selected_card_action is not None:
        _validate_model_action(
            _normalized_resolved_parsed(parsed, selected_card_action),
            selected_card_action,
            legal_actions,
        )
        return selected_card_action, None, False

    selected_label = _selected_action_label(parsed)
    if _is_custom_sequence_kind_label(selected_label):
        raise ValueError(CUSTOM_SEQUENCE_KIND_LABEL_ERROR)
    label_action = _action_matching_selected_label(selected_label, legal_actions)
    raw_action_id = parsed.get("action_id")
    custom_label_error = "combat_custom_sequence requires action_id and selected_action_label to identify the same current legal action"

    try:
        action_id = int(raw_action_id)
    except (TypeError, ValueError):
        if label_action is not None:
            if label_action.kind == "combat_custom_sequence":
                raise ValueError(custom_label_error)
            _validate_model_action(parsed, label_action, legal_actions)
            return label_action, None, True
        raise ValueError("action_id must be an integer or selected_action_label must exactly match a current legal action")

    if action_id < 0 or action_id >= len(legal_actions):
        if label_action is not None:
            if label_action.kind == "combat_custom_sequence":
                raise ValueError(custom_label_error)
            _validate_model_action(parsed, label_action, legal_actions)
            return label_action, action_id, True
        raise ValueError(f"action_id out of range: {action_id}")

    action = legal_actions[action_id]
    remapped_from: int | None = None
    if label_action is not None and label_action != action:
        if label_action.kind == "combat_custom_sequence" or action.kind == "combat_custom_sequence":
            raise ValueError(custom_label_error)
        remapped_from = action.action_id
        action = label_action

    _validate_model_action(parsed, action, legal_actions)
    return action, remapped_from, False


def _card_select_action_from_indices(
    parsed: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    state: dict[str, Any] | None = None,
) -> LegalAction | None:
    if "selected_card_indices" not in parsed:
        return None

    raw_indices = parsed.get("selected_card_indices")
    if not isinstance(raw_indices, list):
        raise ValueError("selected_card_indices must be a JSON list of distinct current candidate index integers")

    selected_indices: list[int] = []
    for value in raw_indices:
        if isinstance(value, bool):
            raise ValueError("selected_card_indices entries must be integers, not booleans")
        if isinstance(value, int):
            selected_indices.append(value)
            continue
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            selected_indices.append(int(value.strip()))
            continue
        raise ValueError("selected_card_indices entries must be integers")
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("selected_card_indices must not contain duplicate candidate indices")

    if not selected_indices:
        matches = [action for action in legal_actions if action.kind == "card_select_skip"]
    else:
        selected_set = set(selected_indices)
        matches = []
        for action in legal_actions:
            if action.kind != "card_select":
                continue
            raw_action_indices = str((action.command.get("args") or {}).get("indices") or "")
            try:
                action_indices = [int(value) for value in raw_action_indices.split(",") if value != ""]
            except ValueError:
                continue
            if len(action_indices) == len(selected_indices) and set(action_indices) == selected_set:
                matches.append(action)

        if not matches and isinstance(state, dict):
            spaces = [action for action in legal_actions if action.kind == "card_select_space"]
            if len(spaces) == 1:
                matches = [materialize_card_select_space_action(state, spaces[0], selected_indices)]

    if len(matches) != 1:
        if not selected_indices:
            raise ValueError("selected_card_indices=[] is legal only when the current card selection exposes skip")
        raise ValueError(
            "selected_card_indices do not identify exactly one current legal card selection: "
            + json.dumps(selected_indices, ensure_ascii=False, separators=(",", ":"))
        )

    return matches[0]


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


def _normalized_resolved_parsed(parsed: dict[str, Any], action: LegalAction) -> dict[str, Any]:
    normalized = dict(parsed)
    normalized["action_id"] = action.action_id
    normalized["selected_action_label"] = action.label
    return normalized


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


def _is_custom_sequence_kind_label(selected_label: str) -> bool:
    return _normalize_for_action_match(selected_label) == "combat custom sequence"


def _validate_model_action(parsed: dict[str, Any], action: LegalAction, legal_actions: list[LegalAction]) -> None:
    selected_label = _selected_action_label(parsed)
    if selected_label:
        if _normalize_for_action_match(selected_label) != _normalize_for_action_match(action.label):
            raise ValueError(
                "selected_action_label does not match selected action: "
                f"action_id {action.action_id} is {action.label!r}, got {selected_label!r}"
            )

    if action.kind != "combat_custom_sequence" and _has_nonempty_sequence_payload(parsed):
        raise ValueError("sequence may only be non-empty when selecting combat_custom_sequence")

    if selected_label and action.kind != "combat_custom_sequence":
        _validate_unopened_card_reward_skip(action, legal_actions)
        return

    if action.kind == "combat_custom_sequence":
        normalize_card_sequence_payload(parsed.get("sequence"))
        return

    _validate_unopened_card_reward_skip(action, legal_actions)

    reason = _one_line(parsed.get("reason") if isinstance(parsed, dict) else "")
    mismatched_action = _reason_mentions_different_action(reason, action, legal_actions)
    if mismatched_action is not None:
        raise ValueError(
            "reason does not match selected action: "
            f"action_id {action.action_id} is {action.label!r}, "
            f"but reason appears to describe {mismatched_action.label!r}"
        )


def _has_nonempty_sequence_payload(parsed: dict[str, Any]) -> bool:
    if "sequence" not in parsed:
        return False
    sequence = parsed.get("sequence")
    if sequence is None or sequence == "":
        return False
    if isinstance(sequence, list):
        return bool(sequence)
    return True


def _validate_unopened_card_reward_skip(action: LegalAction, legal_actions: list[LegalAction]) -> None:
    if not _is_unopened_card_reward_skip_action(action):
        return
    reward_index = (action.command.get("args") or {}).get("reward_index")
    if not any(_is_matching_card_reward_claim(candidate, reward_index) for candidate in legal_actions):
        return
    raise ValueError(
        "unopened card_reward skip is not allowed while an open/reveal action exists; "
        "open the card reward first so offered cards are visible, then pick or skip in the card_reward state"
    )


def _is_unopened_card_reward_skip_action(action: LegalAction) -> bool:
    return action.kind == "combat_reward_skip" and _action_label_mentions_card_reward(action)


def _is_matching_card_reward_claim(action: LegalAction, reward_index: Any) -> bool:
    if action.kind != "combat_reward":
        return False
    if not _action_label_mentions_card_reward(action):
        return False
    return (action.command.get("args") or {}).get("reward_index") == reward_index


def _action_label_mentions_card_reward(action: LegalAction) -> bool:
    label = str(action.label or "").lower()
    normalized = _normalize_for_action_match(action.label)
    return "card_reward" in label or "card reward" in normalized


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
        (r"open card reward\s+\d+:\s*(.*)$", ("open", "reveal")),
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


def _model_action_error_message(
    last_error: str | None,
    max_retries: int,
    raw_responses: list[str] | None = None,
) -> str:
    attempts = max_retries + 1
    reason = last_error or "unknown model action error"
    message = f"LLM failed after {attempts} attempt(s): {reason}"
    if raw_responses:
        excerpt = _one_line(raw_responses[-1])[:500]
        message = f"{message}; last_raw_response_excerpt={excerpt!r}"
    return message


def _is_transport_error(exc: Exception) -> bool:
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    if _is_auth_unavailable_error(exc):
        return True
    match = re.match(r"LLM HTTP (\d{3}):", str(exc))
    return bool(match and (match.group(1).startswith("5") or match.group(1) in {"408", "429"}))


def _is_auth_unavailable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "auth_unavailable" in message or "no auth available" in message


def _transport_retry_delay(attempt: int, exc: Exception | None = None) -> float:
    if exc is not None and _is_auth_unavailable_error(exc):
        return min(10.0 * (2.0**attempt), 60.0)
    return min(2.0**attempt, 15.0)
