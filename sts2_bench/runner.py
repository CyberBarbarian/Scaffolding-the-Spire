"""Benchmark runner for STS2 policies."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Literal

from .actions import LegalAction, build_legal_actions
from .agents import (
    Agent,
    ConversationMode,
    MemoryMode,
    OpenAICompatAgent,
    PromptStyle,
    RandomAgent,
    TurnChatAssistantHistory,
    TurnChatUpdateMode,
)
from .context import build_last_action_result, compact_state
from .experience import ExperienceOpenAICompatAgent
from .process import Sts2Process


LogLevel = Literal["metrics", "decisions", "full"]


@dataclass
class RunResult:
    seed: str
    character: str
    ascension: int
    victory: bool = False
    act: int | None = None
    floor: int | None = None
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
    process: Sts2Process | None = None,
    logger: JsonlLogger | None = None,
    print_prompts: bool = False,
    print_model_output: bool = False,
    include_full_map: bool = False,
    allow_repeat_views: bool = False,
    log_level: LogLevel = "decisions",
) -> RunResult:
    """Run one game and return summary metrics."""

    owns_process = process is None
    proc = process or Sts2Process()
    started = time.time()
    result = RunResult(seed=seed, character=character, ascension=ascension)

    try:
        proc.start()
        _start_agent_episode(agent, seed=seed, character=character, ascension=ascension)
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

            legal_actions = build_legal_actions(state, allow_repeat_views=allow_repeat_views)
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
                continue

            if print_prompts:
                print(f"\n===== PROMPT step={step} decision={decision} =====")
                print(prompt)
                print(f"===== END PROMPT step={step} =====\n", flush=True)

            action, meta = agent.choose(state, legal_actions, prompt=prompt)
            if not isinstance(action, LegalAction):
                raise TypeError("Agent returned a non-LegalAction")
            if meta.get("fallback"):
                result.model_fallback_count += 1

            if print_model_output:
                _print_model_output(step=step, decision=decision, action=action, meta=meta)

            logger and logger.write(_action_log_entry(log_level=log_level, step=step, decision=decision, action=action, meta=meta))

            if action.command.get("cmd") == "bench_view":
                old_state = state
                _count_view_action(result, state, action.command)
                result.steps = step + 1
                state = _apply_view_action(proc, state, action.command)
                _record_agent_transition(agent, old_state, action, meta, state)
                _drain_agent_auto_learn_records(agent, logger, step)
                continue

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
            state = _attach_last_action_result(old_state, next_state, action)
            _record_agent_transition(agent, old_state, action, meta, state)
            _drain_agent_auto_learn_records(agent, logger, step)

        result.truncated = True
        result.extra["truncation_reason"] = f"max_steps exceeded ({max_steps})"
        _fill_result(result, state)
        return result
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures per seed
        result.error = str(exc)
        return result
    finally:
        result.wall_time_sec = time.time() - started
        logger and logger.write({"type": "result", "result": result.to_dict()})
        if owns_process:
            proc.close()


def _fill_result(result: RunResult, state: dict[str, Any]) -> None:
    player = state.get("player") or {}
    context = state.get("context") or {}
    result.victory = bool(state.get("victory", False))
    result.act = state.get("act") or context.get("act")
    result.floor = state.get("floor") or context.get("floor")
    result.hp = player.get("hp")
    result.max_hp = player.get("max_hp")
    result.gold = player.get("gold")
    result.deck_size = player.get("deck_size")


def _reset_agent_episode(agent: Agent) -> None:
    reset = getattr(agent, "reset_episode", None)
    if callable(reset):
        reset()


def _start_agent_episode(agent: Agent, *, seed: str, character: str, ascension: int) -> None:
    start = getattr(agent, "start_episode", None)
    if callable(start):
        try:
            start(seed=seed, character=character, ascension=ascension)
        except TypeError:
            start()
        return
    _reset_agent_episode(agent)


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


def _drain_agent_auto_learn_records(agent: Agent, logger: JsonlLogger | None, step: int) -> None:
    pop_records = getattr(agent, "pop_auto_learn_records", None)
    if not callable(pop_records):
        return
    records = pop_records()
    for record in records:
        logger and logger.write({"type": "experience_auto_learn", "step": step, **record})


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
    if log_level == "full":
        entry["state"] = compact_state(state)
        entry["legal_actions"] = [action.to_prompt_dict() for action in legal_actions]
        if prompt is not None:
            entry["prompt"] = prompt
        return entry

    entry["state"] = _state_summary(state)
    entry["legal_actions"] = [_legal_action_summary(action) for action in legal_actions]
    return entry


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
        "conversation_mode",
        "message_count",
        "turn_chat_history_turns",
        "turn_key",
        "conversation_prompt_chars",
        "attempt",
        "elapsed_sec",
        "prompt_chars",
        "fallback",
        "fallback_action_id",
        "error",
        "usage",
        "experience_memory",
    ):
        if key in meta and meta[key] is not None:
            summary[key] = meta[key]

    parsed = meta.get("parsed")
    if isinstance(parsed, dict):
        summary["parsed"] = {
            key: parsed[key]
            for key in ("action_id", "reason")
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
    experience_memory_enabled: bool = False,
    experience_memory_mode: str = "frozen",
    experience_memory_store_path: str = "sts2_bench/experience/memory/active.jsonl",
    experience_memory_top_k: int = 5,
    experience_memory_token_budget: int = 700,
    experience_memory_min_confidence: float = 0.35,
    experience_memory_include_global: bool = True,
    experience_memory_allow_cross_character: bool = False,
    experience_memory_allow_cross_ascension: bool = True,
    experience_event_trace_path: str = "",
    experience_auto_learn_enabled: bool = False,
    experience_auto_learn_trigger: str = "act_end",
    experience_auto_learn_candidate_path: str = "sts2_bench/experience/memory/candidates.jsonl",
    experience_auto_learn_max_candidates_per_span: int = 2,
    experience_auto_learn_max_events_per_span: int = 40,
    experience_auto_learn_min_confidence: float = 0.35,
    experience_auto_learn_promote: bool = True,
    experience_auto_learn_promote_on_death: bool = False,
) -> Agent:
    if kind == "random":
        if conversation_mode != "single_turn":
            raise ValueError("turn_chat context management is only supported by the llm agent")
        if experience_memory_enabled:
            raise ValueError("experience_memory is only supported by the llm agent")
        return RandomAgent(
            seed=0,
            include_json_state=include_json_state,
            prompt_style=prompt_style,
            memory_enabled=memory_enabled,
            memory_window=memory_window,
            memory_mode=memory_mode,
            run_summary_enabled=run_summary_enabled,
        )
    if kind == "llm":
        if not base_url or not model:
            raise ValueError("--base-url and --model, or STS2_BENCH_BASE_URL and STS2_BENCH_MODEL, are required for --agent llm")
        if api_key == "local" and not _is_local_url(base_url):
            raise ValueError(
                "Set --api-key, DEEPSEEK_API_KEY, or OPENAI_API_KEY for non-local LLM endpoints"
            )
        agent_class = ExperienceOpenAICompatAgent if experience_memory_enabled else OpenAICompatAgent
        extra_kwargs: dict[str, Any] = {}
        if experience_memory_enabled:
            extra_kwargs = {
                "experience_memory_mode": experience_memory_mode,
                "experience_memory_store_path": experience_memory_store_path,
                "experience_memory_top_k": experience_memory_top_k,
                "experience_memory_token_budget": experience_memory_token_budget,
                "experience_memory_min_confidence": experience_memory_min_confidence,
                "experience_memory_include_global": experience_memory_include_global,
                "experience_memory_allow_cross_character": experience_memory_allow_cross_character,
                "experience_memory_allow_cross_ascension": experience_memory_allow_cross_ascension,
                "experience_event_trace_path": experience_event_trace_path,
                "experience_auto_learn_enabled": experience_auto_learn_enabled,
                "experience_auto_learn_trigger": experience_auto_learn_trigger,
                "experience_auto_learn_candidate_path": experience_auto_learn_candidate_path,
                "experience_auto_learn_max_candidates_per_span": experience_auto_learn_max_candidates_per_span,
                "experience_auto_learn_max_events_per_span": experience_auto_learn_max_events_per_span,
                "experience_auto_learn_min_confidence": experience_auto_learn_min_confidence,
                "experience_auto_learn_promote": experience_auto_learn_promote,
                "experience_auto_learn_promote_on_death": experience_auto_learn_promote_on_death,
            }
        return agent_class(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
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
            **extra_kwargs,
        )
    raise ValueError(f"Unknown agent kind: {kind}")


def _is_local_url(url: str) -> bool:
    lowered = url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered or "0.0.0.0" in lowered
