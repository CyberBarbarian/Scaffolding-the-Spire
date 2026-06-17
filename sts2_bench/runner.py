"""Benchmark runner for STS2 policies."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Literal

from .actions import LegalAction, build_legal_actions
from .agents import (
    Agent,
    ConversationMode,
    FallbackMode,
    MemoryMode,
    OpenAICompatAgent,
    PromptStyle,
    RandomAgent,
    TurnChatAssistantHistory,
    TurnChatUpdateMode,
)
from .context import build_last_action_result, compact_state
from .multi_agent import CoordinatorAgent, OutOfCombatContextAgent
from .process import Sts2Process
from .route_plan import RoutePlanTracker
from .sequences import execute_card_sequence
from .sequence_validation import validate_custom_sequence
from .run_context import RunContextStore


LogLevel = Literal["metrics", "decisions", "full"]
CUSTOM_SEQUENCE_VALIDATION_RETRIES = 2
ROUTE_PLAN_VALIDATION_RETRIES = 2


@dataclass
class RunResult:
    seed: str
    character: str
    ascension: int
    victory: bool = False
    decision: str | None = None
    act: int | None = None
    floor: int | None = None
    room_type: str | None = None
    hp: int | None = None
    max_hp: int | None = None
    gold: int | None = None
    deck_size: int | None = None
    steps: int = 0
    game_action_count: int = 0
    view_deck_count: int = 0
    view_map_count: int = 0
    view_draw_pile_count: int = 0
    view_discard_pile_count: int = 0
    view_exhaust_pile_count: int = 0
    repeated_view_count: int = 0
    model_fallback_count: int = 0
    invalid_states: int = 0
    truncated: bool = False
    wall_time_sec: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["extra"] = dict(self.extra)
        return data


class JsonlLogger:
    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self.handle = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", encoding="utf-8")

    def write(self, entry: dict[str, Any]) -> None:
        if not self.handle:
            return
        self.handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


def run_one(
    *,
    agent: Agent,
    seed: str,
    character: str = "Ironclad",
    ascension: int = 0,
    lang: str = "en",
    max_steps: int = 2000,
    max_game_actions: int | None = None,
    process: Sts2Process | None = None,
    logger: JsonlLogger | None = None,
    print_prompts: bool = False,
    print_model_output: bool = False,
    include_full_map: bool = False,
    allow_repeat_views: bool = False,
    log_level: LogLevel = "decisions",
    sequence_probe_factory: Callable[[], Sts2Process] | None = None,
) -> RunResult:
    """Run one game and return summary metrics."""

    owns_process = process is None
    proc = process or Sts2Process()
    started = time.time()
    result = RunResult(seed=seed, character=character, ascension=ascension)
    action_tape: list[dict[str, Any]] = []
    route_plan_tracker = RoutePlanTracker()

    state: dict[str, Any] | None = None
    try:
        proc.start()
        _reset_agent_episode(agent)
        state = proc.start_run(character=character, seed=seed, ascension=ascension, lang=lang)
        logger and logger.write(
            {
                "type": "start",
                "seed": seed,
                "character": character,
                "ascension": ascension,
                "log_level": log_level,
                "allow_repeat_views": allow_repeat_views,
            }
        )

        for step in range(max_steps):
            decision = state.get("decision")
            if decision == "game_over":
                _fill_result(result, state)
                return result

            if include_full_map and decision == "map_select" and "full_map" not in state:
                map_state = proc.send({"cmd": "get_map"})
                if map_state.get("type") == "map":
                    state = {**state, "full_map": map_state}

            legal_actions = build_legal_actions(
                state,
                allow_repeat_views=allow_repeat_views,
                allow_custom_sequences=_allow_custom_sequences_for_agent(agent),
            )
            prompt = agent.build_prompt(state, legal_actions) if print_prompts or log_level == "full" else None
            if logger:
                decision_entry = _decision_log_entry(
                    log_level=log_level,
                    step=step,
                    state=state,
                    legal_actions=legal_actions,
                    prompt=prompt,
                )
                if decision_entry:
                    logger.write(decision_entry)

            if not legal_actions:
                result.invalid_states += 1
                result.game_action_count += 1
                result.steps = step + 1
                state = proc.action("proceed")
                if _max_game_actions_reached(result, max_game_actions, state):
                    return result
                continue

            if print_prompts:
                print(f"\n===== PROMPT step={step} decision={decision} =====")
                print(prompt)
                print(f"===== END PROMPT step={step} =====\n", flush=True)

            action, meta, validation_error = _choose_action_with_runner_validation(
                agent=agent,
                state=state,
                legal_actions=legal_actions,
                prompt=prompt,
                result=result,
                logger=logger,
                log_level=log_level,
                step=step,
                decision=decision,
                print_model_output=print_model_output,
                action_tape=action_tape,
                character=character,
                seed=seed,
                ascension=ascension,
                lang=lang,
                sequence_probe_factory=sequence_probe_factory,
                route_plan_tracker=route_plan_tracker,
            )
            if validation_error:
                result.invalid_states += 1
                if validation_error.startswith("route plan validation failed:"):
                    result.error = validation_error
                else:
                    result.error = f"custom sequence validation failed: {validation_error}"
                _fill_result(result, state)
                return result

            if action.command.get("cmd") == "bench_view":
                old_state = state
                _count_view_action(result, state, action.command)
                result.steps = step + 1
                state = _apply_view_action(proc, state, action.command)
                _record_agent_transition(agent, old_state, action, meta, state)
                continue
            if action.command.get("cmd") == "bench_sequence":
                result.game_action_count += 1
                result.steps = step + 1
                old_state = state
                sequence_result = execute_card_sequence(state, action.command.get("sequence") or [], proc.send)
                next_state = sequence_result.state
                result.extra["last_sequence_steps"] = sequence_result.steps_executed
                if sequence_result.stopped_reason:
                    result.extra["last_sequence_stop"] = sequence_result.stopped_reason
                if next_state.get("type") == "error":
                    result.invalid_states += 1
                    logger and logger.write({"type": "error", "step": step, "state": next_state})
                    result.error = next_state.get("message") or "Headless engine returned an error state"
                    _fill_result(result, next_state)
                    return result
                action_tape.extend(_copy_commands(sequence_result.commands_executed))
                state = _attach_last_action_result(old_state, next_state, action)
                _record_agent_transition(agent, old_state, action, meta, state)
                if _max_game_actions_reached(result, max_game_actions, state):
                    return result
                continue
            if action.command.get("cmd") == "bench_custom_sequence":
                raise RuntimeError("unvalidated custom sequence reached execution")

            result.game_action_count += 1
            result.steps = step + 1
            old_state = state
            next_state = proc.send(action.command)
            if next_state.get("type") == "error":
                result.invalid_states += 1
                logger and logger.write({"type": "error", "step": step, "state": next_state})
                result.error = next_state.get("message") or "Headless engine returned an error state"
                _fill_result(result, next_state)
                return result
            action_tape.append(_copy_command(action.command))
            state = _attach_last_action_result(old_state, next_state, action)
            _record_agent_transition(agent, old_state, action, meta, state)
            if _max_game_actions_reached(result, max_game_actions, state):
                return result

        result.truncated = True
        result.extra["truncation_reason"] = f"max_steps exceeded ({max_steps})"
        _fill_result(result, state)
        return result
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures per seed
        result.error = str(exc)
        if isinstance(state, dict):
            _fill_result(result, state)
        return result
    finally:
        result.wall_time_sec = time.time() - started
        logger and logger.write({"type": "result", "result": result.to_dict()})
        if owns_process:
            proc.close()


def _choose_action_with_runner_validation(
    *,
    agent: Agent,
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    prompt: str | None,
    result: RunResult,
    logger: JsonlLogger | None,
    log_level: LogLevel,
    step: int,
    decision: Any,
    print_model_output: bool,
    action_tape: list[dict[str, Any]],
    character: str,
    seed: str,
    ascension: int,
    lang: str,
    sequence_probe_factory: Callable[[], Sts2Process] | None,
    route_plan_tracker: RoutePlanTracker,
) -> tuple[LegalAction, dict[str, Any], str | None]:
    validation_retries = 0
    route_plan_validation_retries = 0
    while True:
        action, meta = agent.choose(state, legal_actions, prompt=prompt)
        if not isinstance(action, LegalAction):
            raise TypeError("Agent returned a non-LegalAction")
        if meta.get("fallback"):
            result.model_fallback_count += 1

        if print_model_output:
            _print_model_output(step=step, decision=decision, action=action, meta=meta)

        if logger:
            logger.write(_action_log_entry(log_level=log_level, step=step, decision=decision, action=action, meta=meta))

        route_plan_result = route_plan_tracker.validate_and_apply(
            state=state,
            legal_actions=legal_actions,
            action=action,
            meta=meta,
            step=step,
        )
        if route_plan_result.error is not None:
            error = route_plan_result.error
            if logger:
                logger.write(
                    {
                        "type": "route_plan_validation_error",
                        "step": step,
                        "code": error.code,
                        "reason": error.message,
                        "event": error.event,
                        "retry": route_plan_validation_retries,
                        "retryable": route_plan_validation_retries < ROUTE_PLAN_VALIDATION_RETRIES,
                    }
                )
            if route_plan_validation_retries >= ROUTE_PLAN_VALIDATION_RETRIES:
                return action, meta, f"route plan validation failed: {error.code}: {error.message}"

            route_plan_validation_retries += 1
            result.extra["route_plan_validation_retry_count"] = (
                int(result.extra.get("route_plan_validation_retry_count") or 0) + 1
            )
            _send_external_feedback(agent, state, action, meta, error.feedback, legal_actions)
            continue

        if route_plan_result.events:
            for event in route_plan_result.events:
                logger and logger.write(event)
            result.extra["route_plan_event_count"] = int(result.extra.get("route_plan_event_count") or 0) + len(
                route_plan_result.events
            )

        if action.command.get("cmd") != "bench_custom_sequence":
            return action, meta, None

        action, meta, validation_error, sequence_payload = _prepare_custom_sequence_action(
            action=action,
            meta=meta,
            legal_actions=legal_actions,
            result=result,
            state=state,
            action_tape=action_tape,
            character=character,
            seed=seed,
            ascension=ascension,
            lang=lang,
            sequence_probe_factory=sequence_probe_factory,
        )
        if validation_error is None:
            return action, meta, None

        if logger:
            logger.write(
                {
                    "type": "custom_sequence_validation_error",
                    "step": step,
                    "reason": validation_error,
                    "sequence": sequence_payload,
                    "retry": validation_retries,
                    "retryable": validation_retries < CUSTOM_SEQUENCE_VALIDATION_RETRIES,
                }
            )
        if validation_retries >= CUSTOM_SEQUENCE_VALIDATION_RETRIES:
            return action, meta, validation_error

        validation_retries += 1
        result.extra["custom_sequence_validation_retry_count"] = (
            int(result.extra.get("custom_sequence_validation_retry_count") or 0) + 1
        )
        _send_external_feedback(
            agent,
            state,
            action,
            meta,
            _custom_sequence_feedback(validation_error),
            legal_actions,
        )


def _prepare_custom_sequence_action(
    *,
    action: LegalAction,
    meta: dict[str, Any],
    legal_actions: list[LegalAction],
    result: RunResult,
    state: dict[str, Any],
    action_tape: list[dict[str, Any]],
    character: str,
    seed: str,
    ascension: int,
    lang: str,
    sequence_probe_factory: Callable[[], Sts2Process] | None,
) -> tuple[LegalAction, dict[str, Any], str | None, Any]:
    parsed = meta.get("parsed") if isinstance(meta.get("parsed"), dict) else {}
    sequence_payload = parsed.get("sequence")
    existing_action = _matching_sequence_action(sequence_payload, legal_actions)
    if existing_action is not None:
        result.extra["last_sequence_validation"] = {
            "validated": True,
            "reason": "custom sequence matched existing legal action",
            "steps_executed": len(sequence_payload) if isinstance(sequence_payload, list) else 0,
        }
        return (
            existing_action,
            {**meta, "custom_sequence_resolved_to_existing_action_id": existing_action.action_id},
            None,
            sequence_payload,
        )

    validation = validate_custom_sequence(
        sequence_payload,
        current_state=state,
        action_tape=action_tape,
        character=character,
        seed=seed,
        ascension=ascension,
        lang=lang,
        process_factory=sequence_probe_factory,
    )
    result.extra["last_sequence_validation"] = {
        "validated": validation.validated,
        "reason": validation.reason,
        "steps_executed": validation.steps_executed,
    }
    if not validation.validated:
        return action, meta, validation.reason, sequence_payload

    return (
        LegalAction(
            action_id=action.action_id,
            label=_custom_sequence_label(sequence_payload),
            command={"cmd": "bench_sequence", "sequence": sequence_payload},
            kind="combat_card_sequence",
        ),
        {**meta, "custom_sequence_validated": True},
        None,
        sequence_payload,
    )


def _send_external_feedback(
    agent: Agent,
    state: dict[str, Any],
    action: LegalAction,
    meta: dict[str, Any],
    feedback: str,
    legal_actions: list[LegalAction],
) -> None:
    receive = getattr(agent, "receive_external_feedback", None)
    if callable(receive):
        receive(state, action, meta, feedback, legal_actions)


def _custom_sequence_feedback(validation_error: str) -> str:
    return (
        f"Your previous response was invalid: custom sequence validation failed: {validation_error}. "
        "Choose again from the current legal actions. Do not use combat_custom_sequence for cards with random, "
        "choose, select, discover, or other manual-choice effects; play those cards as single legal actions "
        "or choose a listed deterministic combat_card_sequence. For custom sequence payloads, card_index values "
        "come from the current visible hand snapshot; do not predict shifted card indices after earlier cards are played."
    )


def _fill_result(result: RunResult, state: dict[str, Any]) -> None:
    player = state.get("player") or {}
    context = state.get("context") or {}
    result.decision = state.get("decision")
    result.victory = bool(state.get("victory", False))
    result.act = state.get("act") or context.get("act")
    result.floor = state.get("floor") or context.get("floor")
    result.room_type = context.get("room_type")
    result.hp = player.get("hp")
    result.max_hp = player.get("max_hp")
    result.gold = player.get("gold")
    result.deck_size = player.get("deck_size")


def _max_game_actions_reached(
    result: RunResult,
    max_game_actions: int | None,
    state: dict[str, Any],
) -> bool:
    if max_game_actions is None or max_game_actions <= 0:
        return False
    if result.game_action_count < max_game_actions:
        return False
    if state.get("decision") == "game_over":
        _fill_result(result, state)
        return True
    result.truncated = True
    result.extra["truncation_reason"] = f"max_game_actions exceeded ({max_game_actions})"
    _fill_result(result, state)
    return True


def _reset_agent_episode(agent: Agent) -> None:
    reset = getattr(agent, "reset_episode", None)
    if callable(reset):
        reset()


def _allow_custom_sequences_for_agent(agent: Agent) -> bool:
    return not isinstance(agent, RandomAgent)


def _record_agent_transition(
    agent: Agent,
    state: dict[str, Any],
    action: LegalAction,
    meta: dict[str, Any],
    next_state: dict[str, Any] | None = None,
) -> None:
    record = getattr(agent, "record_transition", None)
    if callable(record):
        try:
            record(state, action, meta, next_state)
        except TypeError:
            record(state, action, meta)


def _apply_view_action(proc: Sts2Process, state: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
    view = command.get("view")
    if view == "deck":
        return {**state, "view_deck": True}
    if view == "map":
        map_state = proc.send({"cmd": "get_map"})
        if map_state.get("type") == "map":
            return {**state, "view_map": True, "full_map": map_state}
        return {**state, "view_map": True, "view_map_error": map_state}
    if view in {"draw", "discard", "exhaust"}:
        return {**state, f"view_{view}_pile": True}
    return state


def _matching_sequence_action(sequence_payload: Any, legal_actions: list[LegalAction]) -> LegalAction | None:
    sequence_key = _sequence_key(sequence_payload)
    if sequence_key is None:
        return None
    for action in legal_actions:
        if action.kind != "combat_card_sequence":
            continue
        if _sequence_key(action.command.get("sequence")) == sequence_key:
            return action
    return None


def _sequence_key(sequence_payload: Any) -> tuple[tuple[Any, Any], ...] | None:
    if not isinstance(sequence_payload, list):
        return None
    key = []
    for step in sequence_payload:
        if not isinstance(step, dict):
            return None
        if "card_index" not in step:
            return None
        key.append((step.get("card_index"), step.get("target_index")))
    return tuple(key)


def _custom_sequence_label(sequence_payload: Any) -> str:
    if not isinstance(sequence_payload, list):
        return "custom card sequence"
    parts = []
    for step in sequence_payload:
        if not isinstance(step, dict):
            continue
        label = f"{step.get('card_index')}"
        if step.get("target_index") is not None:
            label += f" on enemy {step.get('target_index')}"
        parts.append(label)
    return "custom sequence: " + " -> ".join(parts) if parts else "custom card sequence"


def _copy_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_copy_command(command) for command in commands]


def _copy_command(command: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(command, ensure_ascii=False))


def _attach_last_action_result(old_state: dict[str, Any], new_state: dict[str, Any], action: LegalAction) -> dict[str, Any]:
    return {
        **new_state,
        "last_action_result": build_last_action_result(
            old_state,
            new_state,
            action_label=action.label,
            action_kind=action.kind,
        ),
    }


def _print_model_output(*, step: int, decision: Any, action: LegalAction, meta: dict[str, Any]) -> None:
    print(f"\n===== MODEL OUTPUT step={step} decision={decision} =====")
    raw_response = meta.get("raw_response")
    if raw_response is not None:
        print("raw_response:")
        print(raw_response)
    parsed = meta.get("parsed")
    if parsed is not None:
        print("parsed:")
        print(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))
    if meta.get("fallback"):
        print("fallback: true")
    if meta.get("error"):
        print(f"error: {meta.get('error')}")
    usage = meta.get("usage")
    if usage is not None:
        print("usage:")
        print(json.dumps(usage, ensure_ascii=False, separators=(",", ":")))
    print("selected_action:")
    print(json.dumps(action.to_prompt_dict(), ensure_ascii=False, separators=(",", ":")))
    print(f"===== END MODEL OUTPUT step={step} =====\n", flush=True)


def _count_view_action(result: RunResult, state: dict[str, Any], command: dict[str, Any]) -> None:
    view = command.get("view")
    if view == "deck":
        if state.get("view_deck"):
            result.repeated_view_count += 1
        result.view_deck_count += 1
    elif view == "map":
        if state.get("view_map"):
            result.repeated_view_count += 1
        result.view_map_count += 1
    elif view == "draw":
        if state.get("view_draw_pile"):
            result.repeated_view_count += 1
        result.view_draw_pile_count += 1
    elif view == "discard":
        if state.get("view_discard_pile"):
            result.repeated_view_count += 1
        result.view_discard_pile_count += 1
    elif view == "exhaust":
        if state.get("view_exhaust_pile"):
            result.repeated_view_count += 1
        result.view_exhaust_pile_count += 1


def _decision_log_entry(
    *,
    log_level: LogLevel,
    step: int,
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    prompt: str | None,
) -> dict[str, Any] | None:
    if log_level == "metrics":
        return None

    entry: dict[str, Any] = {
        "type": "decision",
        "step": step,
        "decision": state.get("decision"),
    }
    map_snapshot = _map_snapshot_for_trace(state)
    if map_snapshot is not None:
        entry["map_snapshot"] = map_snapshot
    if log_level == "full":
        entry["state"] = compact_state(state)
        entry["legal_actions"] = [action.to_prompt_dict() for action in legal_actions]
        if prompt is not None:
            entry["prompt"] = prompt
        return entry

    entry["state"] = _state_summary(state)
    entry["legal_actions"] = [_legal_action_summary(action) for action in legal_actions]
    return entry


def _map_snapshot_for_trace(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("decision") != "map_select":
        return None
    full_map = state.get("full_map")
    if isinstance(full_map, dict) and full_map.get("type") == "map":
        return full_map
    return None


def _action_log_entry(
    *,
    log_level: LogLevel,
    step: int,
    decision: Any,
    action: LegalAction,
    meta: dict[str, Any],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "action",
        "step": step,
        "decision": decision,
    }
    if log_level == "full":
        entry["action"] = action.to_prompt_dict()
        entry["agent_meta"] = meta
        return entry

    entry["action"] = _legal_action_summary(action)
    entry["agent_meta"] = _agent_meta_summary(meta)
    return entry


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    context = state.get("context") or {}
    player = state.get("player") or {}
    summary = {
        "decision": state.get("decision"),
        "act": state.get("act") or context.get("act"),
        "floor": state.get("floor") or context.get("floor"),
        "room_type": context.get("room_type"),
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "gold": player.get("gold"),
        "deck_size": player.get("deck_size"),
        "hand_count": len(state.get("hand") or []),
        "enemy_count": len(state.get("enemies") or []),
        "choice_count": len(state.get("choices") or state.get("options") or []),
        "card_count": len(state.get("cards") or []),
        "reward_count": len(state.get("rewards") or []),
        "viewed": [
            label
            for label, key in (
                ("deck", "view_deck"),
                ("map", "view_map"),
                ("draw_pile", "view_draw_pile"),
                ("discard_pile", "view_discard_pile"),
                ("exhaust_pile", "view_exhaust_pile"),
            )
            if state.get(key)
        ],
    }
    last_action_result = _last_action_result_summary(state.get("last_action_result"))
    if last_action_result is not None:
        summary["last_action_result"] = last_action_result
    return summary


def _last_action_result_summary(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    return {
        "action_label": result.get("action_label"),
        "action_kind": result.get("action_kind"),
        "decision_before": result.get("decision_before"),
        "decision_after": result.get("decision_after"),
        "changes": result.get("changes", [])[:8],
    }


def _legal_action_summary(action: LegalAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "kind": action.kind,
        "label": action.label,
    }


def _agent_meta_summary(meta: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "agent",
        "model",
        "active_agent",
        "route_reason",
        "conversation_mode",
        "message_count",
        "turn_chat_history_turns",
        "turn_key",
        "conversation_prompt_chars",
        "attempt",
        "elapsed_sec",
        "prompt_chars",
        "occ_context_mode",
        "occ_history_turns",
        "fallback",
        "fallback_action_id",
        "error",
        "usage",
        "run_context_patch",
        "run_context_hash",
    ):
        if key in meta and meta[key] is not None:
            summary[key] = meta[key]

    parsed = meta.get("parsed")
    if isinstance(parsed, dict):
        summary["parsed"] = {
            key: parsed[key]
            for key in ("action_id", "reason", "route_plan_event")
            if key in parsed
        }
    return summary


def summarize(results: Iterable[RunResult]) -> dict[str, Any]:
    rows = list(results)
    if not rows:
        return {"runs": 0}
    complete = [row for row in rows if row.error is None and not row.truncated]
    floors = [row.floor or 0 for row in rows]
    total_steps = sum(row.steps for row in rows)
    total_view_actions = sum(
        row.view_deck_count
        + row.view_map_count
        + row.view_draw_pile_count
        + row.view_discard_pile_count
        + row.view_exhaust_pile_count
        for row in rows
    )
    total_game_actions = sum(row.game_action_count for row in rows)
    truncated = sum(1 for row in rows if row.truncated)
    errors = sum(1 for row in rows if row.error is not None)
    summary = {
        "runs": len(rows),
        "completed": len(complete),
        "victories": sum(1 for row in rows if row.victory),
        "deaths": sum(1 for row in complete if not row.victory),
        "truncated": truncated,
        "errors": errors,
        "truncation_rate": truncated / len(rows),
        "error_rate": errors / len(rows),
        "win_rate": sum(1 for row in rows if row.victory) / len(rows),
        "min_floor": min(floors),
        "max_floor": max(floors),
        "median_floor": median(floors),
        "avg_floor": sum(floors) / len(rows),
        "avg_floor_completed": (
            sum((row.floor or 0) for row in complete) / len(complete)
            if complete
            else 0.0
        ),
        "avg_steps": sum(row.steps for row in rows) / len(rows),
        "avg_game_action_count": total_game_actions / len(rows),
        "avg_view_deck_count": sum(row.view_deck_count for row in rows) / len(rows),
        "avg_view_map_count": sum(row.view_map_count for row in rows) / len(rows),
        "avg_view_draw_pile_count": sum(row.view_draw_pile_count for row in rows) / len(rows),
        "avg_view_discard_pile_count": sum(row.view_discard_pile_count for row in rows) / len(rows),
        "avg_view_exhaust_pile_count": sum(row.view_exhaust_pile_count for row in rows) / len(rows),
        "avg_view_action_count": total_view_actions / len(rows),
        "avg_repeated_view_count": sum(row.repeated_view_count for row in rows) / len(rows),
        "view_action_rate": total_view_actions / total_steps if total_steps else 0.0,
        "avg_model_fallback_count": sum(row.model_fallback_count for row in rows) / len(rows),
        "avg_invalid_states": sum(row.invalid_states for row in rows) / len(rows),
        "avg_wall_time_sec": sum(row.wall_time_sec for row in rows) / len(rows),
    }
    if truncated:
        summary["warning"] = "Some runs hit max_steps; floor is not a natural terminal result for those runs."
    return summary


def agent_from_args(
    kind: str,
    *,
    base_url: str | None,
    model: str | None,
    api_key: str,
    llm_timeout: float = 120.0,
    llm_max_retries: int = 2,
    llm_fallback_mode: FallbackMode = "error",
    llm_temperature: float = 0.0,
    include_json_state: bool = False,
    prompt_style: PromptStyle = "default",
    memory_enabled: bool = False,
    memory_window: int = 8,
    memory_mode: MemoryMode = "action_reason",
    run_summary_enabled: bool = True,
    conversation_mode: ConversationMode = "single_turn",
    turn_chat_window: int = 4,
    turn_chat_update_mode: TurnChatUpdateMode = "delta",
    turn_chat_assistant_history: TurnChatAssistantHistory = "compact",
    turn_chat_plan_enabled: bool = False,
) -> Agent:
    if kind == "random":
        if conversation_mode != "single_turn":
            raise ValueError("turn_chat context management is only supported by the llm agent")
        return RandomAgent(
            seed=0,
            include_json_state=include_json_state,
            prompt_style=prompt_style,
            memory_enabled=memory_enabled,
            memory_window=memory_window,
            memory_mode=memory_mode,
            run_summary_enabled=run_summary_enabled,
        )
    if kind in {"llm", "coordinator"}:
        if not base_url or not model:
            raise ValueError("--base-url and --model, or STS2_BENCH_BASE_URL and STS2_BENCH_MODEL, are required for LLM-backed agents")
        if api_key == "local" and not _is_local_url(base_url):
            raise ValueError(
                "Set --api-key, DEEPSEEK_API_KEY, or OPENAI_API_KEY for non-local LLM endpoints"
            )
        if kind == "coordinator":
            store = RunContextStore()
            ca_agent = OpenAICompatAgent(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
                fallback_mode=llm_fallback_mode,
                temperature=llm_temperature,
                include_json_state=include_json_state,
                prompt_style=prompt_style,
                memory_enabled=False,
                run_summary_enabled=run_summary_enabled,
                conversation_mode="turn_chat",
                turn_chat_window=turn_chat_window,
                turn_chat_update_mode=turn_chat_update_mode,
                turn_chat_assistant_history=turn_chat_assistant_history,
                turn_chat_plan_enabled=turn_chat_plan_enabled,
            )
            occ_inner = OpenAICompatAgent(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
                fallback_mode=llm_fallback_mode,
                temperature=llm_temperature,
                include_json_state=include_json_state,
                prompt_style=prompt_style,
                memory_enabled=False,
                run_summary_enabled=run_summary_enabled,
                conversation_mode="single_turn",
                turn_chat_window=turn_chat_window,
                turn_chat_update_mode=turn_chat_update_mode,
                turn_chat_assistant_history=turn_chat_assistant_history,
                turn_chat_plan_enabled=False,
            )
            return CoordinatorAgent(
                ca_agent=ca_agent,
                occ_agent=OutOfCombatContextAgent(occ_inner, store, history_window=turn_chat_window),
                store=store,
                fallback_mode=llm_fallback_mode,
            )
        return OpenAICompatAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
            fallback_mode=llm_fallback_mode,
            temperature=llm_temperature,
            include_json_state=include_json_state,
            prompt_style=prompt_style,
            memory_enabled=memory_enabled,
            memory_window=memory_window,
            memory_mode=memory_mode,
            run_summary_enabled=run_summary_enabled,
            conversation_mode=conversation_mode,
            turn_chat_window=turn_chat_window,
            turn_chat_update_mode=turn_chat_update_mode,
            turn_chat_assistant_history=turn_chat_assistant_history,
            turn_chat_plan_enabled=turn_chat_plan_enabled,
        )
    raise ValueError(f"Unknown agent kind: {kind}")


def _is_local_url(url: str) -> bool:
    lowered = url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered or "0.0.0.0" in lowered
