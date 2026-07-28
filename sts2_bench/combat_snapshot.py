"""Replay-prefix combat snapshots for focused high-pressure ablations."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .actions import CustomSequenceExposure, LegalAction, UnsupportedDecisionError, build_legal_actions
from .agents import Agent
from .evidence import build_evidence_bundle
from .process import Sts2Process
from .route_plan import RoutePlanTracker
from .runner import (
    JsonlLogger,
    LogLevel,
    RoutePlanEnforcement,
    RunResult,
    _agent_run_context_store,
    _allow_card_sequences_for_agent,
    _allow_custom_sequences_for_agent,
    _apply_view_action,
    _attach_last_action_result,
    _choose_action_with_runner_validation,
    _copy_command,
    _copy_commands,
    _count_view_action,
    _decision_log_entry,
    _enemy_snapshot_for_trace,
    _fill_result,
    _log_executed_action,
    _max_game_actions_reached,
    _record_agent_transition,
    _reset_agent_episode,
    _state_summary,
    agent_from_args,
)
from .sequences import execute_card_sequence


SNAPSHOT_VERSION = 1
DEFAULT_ACTIVE_DECISIONS = ("combat_play", "card_select")
SAVE_LOAD_TYPE = "save"


class CombatSnapshotError(RuntimeError):
    """Base error for snapshot extraction and replay failures."""


class SnapshotReplayError(CombatSnapshotError):
    """Raised when a saved prefix no longer replays to the target state."""


@dataclass
class PrefixReplayResult:
    state: dict[str, Any]
    action_tape: list[dict[str, Any]]
    prefix_actions_executed: int
    process: Any


def extract_combat_snapshot(
    trace_path: Path | str,
    *,
    start_step: int,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Extract a replay-prefix snapshot ending at ``start_step`` from a JSONL trace."""

    trace_path = Path(trace_path)
    rows = _read_jsonl(trace_path)
    start_row = _first_row(rows, "start")
    run_config = next((row.get("config") for row in rows if row.get("type") == "run_config"), {})
    target_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("type") == "decision" and row.get("step") == start_step
        ),
        None,
    )
    target_row = rows[target_index] if target_index is not None else None
    if target_row is None:
        raise CombatSnapshotError(f"No decision row found at step {start_step} in {trace_path}")

    target_decision = target_row.get("decision")
    if target_decision not in DEFAULT_ACTIVE_DECISIONS:
        raise CombatSnapshotError(
            f"Snapshot start step {start_step} is {target_decision!r}; expected a combat entry decision"
        )

    prefix_actions = []
    for index, row in enumerate(rows[:target_index]):
        if row.get("type") == "setup_action":
            command = row.get("command")
            if isinstance(command, dict):
                prefix_actions.append(
                    {
                        "source": "setup_action",
                        "setup_index": row.get("setup_index"),
                        "command": _copy_command(command),
                    }
                )
            continue
        if row.get("type") != "action":
            continue
        step = row.get("step")
        if not isinstance(step, int) or step >= start_step:
            continue
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        prefix_item = {
            "step": step,
            "decision": row.get("decision"),
            "action_id": action.get("action_id"),
            "kind": action.get("kind"),
            "label": action.get("label"),
        }
        sequence = (
            _sequence_from_action_row(row)
            or _sequence_from_action_label(action.get("label"))
            or _sequence_from_next_decision(rows, index)
        )
        if action.get("kind") == "combat_custom_sequence" and sequence is None:
            continue
        if sequence is not None:
            prefix_item["sequence"] = sequence
        prefix_actions.append(prefix_item)

    game = _game_from_rows(start_row, run_config)
    authoritative_target_state = target_row.get("authoritative_state")
    target_state = (
        authoritative_target_state
        if isinstance(authoritative_target_state, dict)
        else target_row.get("state")
    )
    if not isinstance(target_state, dict):
        target_state = {}
    target_state_summary = (
        _state_summary(target_state)
        if isinstance(authoritative_target_state, dict)
        else target_state
    )
    target_last_action_result = target_state.get("last_action_result")
    if not isinstance(target_last_action_result, dict):
        target_last_action_result = None
    return {
        "type": "combat_snapshot",
        "version": SNAPSHOT_VERSION,
        "snapshot_id": snapshot_id or f"{trace_path.stem}_step_{start_step}",
        "source": {
            "trace_path": str(trace_path),
            "start_step": start_step,
            "trace_log_level": start_row.get("log_level"),
        },
        "game": game,
        "target": {
            "step": start_step,
            "decision": target_decision,
            "state_summary": target_state_summary,
            "enemy_snapshot": target_row.get("enemy_snapshot"),
            "last_action_result": target_last_action_result,
        },
        "stop_policy": {
            "active_decisions": list(DEFAULT_ACTIVE_DECISIONS),
            "description": "Run from this entry until the decision leaves combat_play/card_select or reaches game_over.",
        },
        "prefix_actions": prefix_actions,
    }


def write_combat_snapshot(snapshot: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_save_backed_combat_snapshot(
    *,
    process: Any,
    state: dict[str, Any],
    action_tape: list[dict[str, Any]],
    save_path: Path | str,
    snapshot_id: str,
    game: dict[str, Any],
    step: int | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a combat snapshot backed by a real engine save plus room-entry commands."""

    target_decision = state.get("decision")
    if target_decision not in DEFAULT_ACTIVE_DECISIONS:
        raise CombatSnapshotError(
            f"Save-backed snapshot target is {target_decision!r}; expected a combat decision"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_result = process.send({"cmd": "write_continue_save", "path": str(save_path)})
    if not (isinstance(save_result, dict) and save_result.get("success") is True):
        raise CombatSnapshotError(f"Could not write combat checkpoint: {save_result!r}")

    entry_commands = _entry_commands_since_room_entry(action_tape)
    return {
        "type": "combat_snapshot",
        "version": SNAPSHOT_VERSION,
        "snapshot_id": snapshot_id,
        "source": {
            "kind": SAVE_LOAD_TYPE,
            "checkpoint_path": str(save_path),
            "save_result": save_result,
            **(source or {}),
        },
        "game": dict(game),
        "target": {
            "step": step,
            "decision": target_decision,
            "state_summary": _state_summary(state),
            "enemy_snapshot": _enemy_snapshot_for_trace(state),
        },
        "stop_policy": {
            "active_decisions": list(DEFAULT_ACTIVE_DECISIONS),
            "description": "Load the pre-room save, replay entry commands, then run until combat exits.",
        },
        "prefix_actions": [],
        "load": {
            "type": SAVE_LOAD_TYPE,
            "path": str(save_path),
            "entry_commands": entry_commands,
        },
    }


def load_combat_snapshot(path: Path | str) -> dict[str, Any]:
    snapshot = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    _validate_snapshot(snapshot)
    return snapshot


def replay_snapshot_prefix(
    snapshot: dict[str, Any],
    *,
    process: Any | None = None,
    process_factory: Callable[[], Any] | None = None,
    allow_repeat_views: bool = False,
    allow_card_sequences: bool = True,
    allow_precomputed_card_sequences: bool = False,
    allow_custom_sequences: bool = True,
    custom_sequence_exposure: CustomSequenceExposure = "safe",
) -> PrefixReplayResult:
    """Start a run and replay saved labels to the snapshot entry."""

    _validate_snapshot(snapshot)
    owns_process = process is None
    proc = process if process is not None else (process_factory or Sts2Process)()
    try:
        return _replay_snapshot_prefix_with_process(
            snapshot,
            proc=proc,
            allow_repeat_views=allow_repeat_views,
            allow_card_sequences=allow_card_sequences,
            allow_precomputed_card_sequences=allow_precomputed_card_sequences,
            allow_custom_sequences=allow_custom_sequences,
            custom_sequence_exposure=custom_sequence_exposure,
        )
    except BaseException:
        if owns_process:
            close = getattr(proc, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        raise


def _replay_snapshot_prefix_with_process(
    snapshot: dict[str, Any],
    *,
    proc: Any,
    allow_repeat_views: bool,
    allow_card_sequences: bool,
    allow_precomputed_card_sequences: bool,
    allow_custom_sequences: bool,
    custom_sequence_exposure: CustomSequenceExposure,
) -> PrefixReplayResult:
    game = snapshot["game"]
    proc.start()
    if _is_save_backed_snapshot(snapshot):
        state = proc.send(
            {
                "cmd": "load_save",
                "path": snapshot["load"]["path"],
                "lang": game.get("lang", "en"),
            }
        )
        if state.get("type") == "error":
            raise SnapshotReplayError(state.get("message") or "Headless engine returned an error state")
        action_tape: list[dict[str, Any]] = []
        for command in snapshot["load"].get("entry_commands") or []:
            state = proc.send(command)
            if state.get("type") == "error":
                raise SnapshotReplayError(state.get("message") or "Headless engine returned an error state")
            action_tape.append(_copy_command(command))
        _assert_target_matches(snapshot, state)
        return PrefixReplayResult(
            state=state,
            action_tape=action_tape,
            prefix_actions_executed=len(action_tape),
            process=proc,
        )

    state = proc.start_run(
        character=game.get("character", "Ironclad"),
        seed=game.get("seed"),
        ascension=int(game.get("ascension", 0) or 0),
        lang=game.get("lang", "en"),
    )
    action_tape: list[dict[str, Any]] = []
    prefix_actions = snapshot.get("prefix_actions") or []

    for item in prefix_actions:
        raw_command = item.get("command")
        if isinstance(raw_command, dict):
            command = _copy_command(raw_command)
            state = proc.send(command)
            if isinstance(state, dict) and state.get("type") == "error":
                raise SnapshotReplayError(state.get("message") or "Headless engine returned an error state")
            action_tape.append(command)
            continue
        legal_actions = build_legal_actions(
            state,
            allow_repeat_views=allow_repeat_views,
            allow_card_sequences=allow_card_sequences,
            allow_precomputed_card_sequences=allow_precomputed_card_sequences,
            allow_custom_sequences=allow_custom_sequences,
            custom_sequence_exposure=custom_sequence_exposure,
        )
        action = _match_saved_action(item, legal_actions, state)
        state = _execute_action(
            proc=proc,
            state=state,
            action=action,
            action_tape=action_tape,
            agent=None,
            meta={},
            result=None,
        )

    _assert_target_matches(snapshot, state)
    return PrefixReplayResult(
        state=state,
        action_tape=action_tape,
        prefix_actions_executed=len(prefix_actions),
        process=proc,
    )


def run_combat_snapshot(
    *,
    snapshot: dict[str, Any],
    agent: Agent,
    process: Any | None = None,
    process_factory: Callable[[], Any] | None = None,
    logger: JsonlLogger | None = None,
    max_steps: int | None = 200,
    max_game_actions: int | None = None,
    allow_repeat_views: bool = False,
    allow_custom_sequences: bool | None = None,
    custom_sequence_exposure: CustomSequenceExposure = "safe",
    combat_card_sequences_enabled: bool = True,
    precomputed_combat_card_sequences_enabled: bool = False,
    log_level: LogLevel = "decisions",
    print_prompts: bool = False,
    print_model_output: bool = False,
    route_plan_enabled: bool = False,
    route_plan_enforcement: RoutePlanEnforcement = "hard",
    sequence_probe_factory: Callable[[], Sts2Process] | None = None,
) -> RunResult:
    """Run one agent from a saved combat entry until the combat is over."""

    if route_plan_enforcement not in {"hard", "soft"}:
        raise ValueError(f"Unsupported route_plan_enforcement: {route_plan_enforcement}")

    _validate_snapshot(snapshot)
    owns_process = process is None
    proc = process or (process_factory or Sts2Process)()
    game = snapshot["game"]
    seed = str(game.get("seed") or "")
    character = str(game.get("character") or "Ironclad")
    ascension = int(game.get("ascension", 0) or 0)
    lang = str(game.get("lang") or "en")
    result = RunResult(seed=seed, character=character, ascension=ascension)
    result.extra.update(
        {
            "snapshot_id": snapshot.get("snapshot_id"),
            "snapshot_entry_step": snapshot.get("target", {}).get("step"),
            "snapshot_prefix_actions": len(snapshot.get("prefix_actions") or []),
        }
    )
    started = time.time()
    state: dict[str, Any] | None = None

    custom_sequences_enabled = (
        _allow_custom_sequences_for_agent(agent) if allow_custom_sequences is None else allow_custom_sequences
    )
    card_sequences_enabled = combat_card_sequences_enabled and _allow_card_sequences_for_agent(agent)
    precomputed_card_sequences_enabled = (
        precomputed_combat_card_sequences_enabled and _allow_card_sequences_for_agent(agent)
    )
    effective_sequence_probe_factory = sequence_probe_factory
    if effective_sequence_probe_factory is None and _is_save_backed_snapshot(snapshot):
        effective_sequence_probe_factory = _save_backed_sequence_probe_factory(snapshot)
    active_decisions = set((snapshot.get("stop_policy") or {}).get("active_decisions") or DEFAULT_ACTIVE_DECISIONS)
    route_plan_tracker = RoutePlanTracker() if route_plan_enabled else None

    try:
        _reset_agent_episode(agent)
        replay = replay_snapshot_prefix(
            snapshot,
            process=proc,
            allow_repeat_views=allow_repeat_views,
            allow_card_sequences=card_sequences_enabled,
            allow_precomputed_card_sequences=precomputed_card_sequences_enabled,
            allow_custom_sequences=custom_sequences_enabled,
            custom_sequence_exposure=custom_sequence_exposure,
        )
        state = replay.state
        action_tape = replay.action_tape
        logger and logger.write(
            {
                "type": "snapshot_entry",
                "snapshot_id": snapshot.get("snapshot_id"),
                "entry_step": snapshot.get("target", {}).get("step"),
                "prefix_actions_executed": replay.prefix_actions_executed,
                "state": _state_summary(state),
                "enemy_snapshot": _enemy_snapshot_for_trace(state),
            }
        )

        step_limit = None if max_steps is None or max_steps <= 0 else max_steps
        step = 0
        while step_limit is None or step < step_limit:
            decision = state.get("decision")
            if decision == "game_over":
                result.extra["snapshot_outcome"] = "game_over"
                _fill_result(result, state)
                return result
            if decision not in active_decisions:
                result.extra["snapshot_outcome"] = "combat_complete"
                _fill_result(result, state)
                return result

            try:
                legal_actions = build_legal_actions(
                    state,
                    allow_repeat_views=allow_repeat_views,
                    allow_card_sequences=card_sequences_enabled,
                    allow_precomputed_card_sequences=precomputed_card_sequences_enabled,
                    allow_custom_sequences=custom_sequences_enabled,
                    custom_sequence_exposure=custom_sequence_exposure,
                )
            except UnsupportedDecisionError as exc:
                result.invalid_states += 1
                if logger:
                    decision_entry = _decision_log_entry(
                        log_level=log_level,
                        step=step,
                        state=state,
                        legal_actions=[],
                        prompt=None,
                    )
                    if decision_entry:
                        decision_entry["snapshot_id"] = snapshot.get("snapshot_id")
                        decision_entry["snapshot_entry_step"] = snapshot.get("target", {}).get("step")
                        decision_entry["decision_error"] = str(exc)
                        decision_entry["report_required"] = True
                        logger.write(decision_entry)
                raise
            evidence_bundle = build_evidence_bundle(
                state,
                legal_actions,
                store=_agent_run_context_store(agent),
            )
            prompt = agent.build_prompt(state, legal_actions) if print_prompts or log_level == "full" else None
            if logger:
                decision_entry = _decision_log_entry(
                    log_level=log_level,
                    step=step,
                    state=state,
                    legal_actions=legal_actions,
                    prompt=prompt,
                    evidence_bundle=evidence_bundle,
                )
                if decision_entry:
                    decision_entry["snapshot_id"] = snapshot.get("snapshot_id")
                    decision_entry["snapshot_entry_step"] = snapshot.get("target", {}).get("step")
                    logger.write(decision_entry)

            if not legal_actions:
                result.invalid_states += 1
                error = (
                    f"recognized snapshot decision {decision!r} exposed no legal actions; "
                    "refusing implicit proceed"
                )
                logger and logger.write(
                    {
                        "type": "decision_error",
                        "step": step,
                        "decision": decision,
                        "snapshot_id": snapshot.get("snapshot_id"),
                        "snapshot_entry_step": snapshot.get("target", {}).get("step"),
                        "error": error,
                        "report_required": True,
                    }
                )
                raise UnsupportedDecisionError(error)

            if print_prompts:
                print(f"\n===== SNAPSHOT PROMPT step={step} decision={decision} =====")
                print(prompt)
                print(f"===== END SNAPSHOT PROMPT step={step} =====\n", flush=True)

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
                sequence_probe_factory=effective_sequence_probe_factory,
                route_plan_tracker=route_plan_tracker,
                route_plan_enabled=route_plan_enabled,
                route_plan_enforcement=route_plan_enforcement,
                evidence_bundle=evidence_bundle,
            )
            if validation_error:
                result.invalid_states += 1
                result.error = validation_error
                _fill_result(result, state)
                return result

            result.steps = step + 1
            state = _execute_action(
                proc=proc,
                state=state,
                action=action,
                action_tape=action_tape,
                agent=agent,
                meta=meta,
                result=result,
            )
            _log_executed_action(
                logger=logger,
                log_level=log_level,
                step=step,
                decision=decision,
                action=action,
                meta=meta,
                extra={
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "snapshot_entry_step": snapshot.get("target", {}).get("step"),
                },
            )
            if _max_game_actions_reached(result, max_game_actions, state):
                return result
            step += 1

        result.truncated = True
        result.extra["truncation_reason"] = f"max_steps exceeded ({max_steps})"
        _fill_result(result, state)
        return result
    except Exception as exc:  # noqa: BLE001 - micro-benchmarks should report per-run failures.
        result.error = str(exc)
        if isinstance(state, dict):
            _fill_result(result, state)
        return result
    finally:
        result.wall_time_sec = time.time() - started
        logger and logger.write({"type": "snapshot_result", "result": result.to_dict()})
        if owns_process:
            close = getattr(proc, "close", None)
            if callable(close):
                close()


def _execute_action(
    *,
    proc: Any,
    state: dict[str, Any],
    action: LegalAction,
    action_tape: list[dict[str, Any]],
    agent: Agent | None,
    meta: dict[str, Any],
    result: RunResult | None,
) -> dict[str, Any]:
    command_type = action.command.get("cmd")
    old_state = state
    if command_type == "bench_view":
        if result is not None:
            _count_view_action(result, state, action.command)
        next_state = _apply_view_action(proc, state, action.command)
    elif command_type == "bench_sequence":
        if result is not None:
            result.game_action_count += 1
        sequence_result = execute_card_sequence(state, action.command.get("sequence") or [], proc.send)
        next_state = sequence_result.state
        action_tape.extend(_copy_commands(sequence_result.commands_executed))
        if result is not None:
            result.extra["last_sequence_steps"] = sequence_result.steps_executed
            if sequence_result.stopped_reason:
                result.extra["last_sequence_stop"] = sequence_result.stopped_reason
        if sequence_result.stopped_reason and not sequence_result.commands_executed:
            raise SnapshotReplayError(sequence_result.stopped_reason)
        if next_state.get("type") == "error":
            raise SnapshotReplayError(next_state.get("message") or "Headless engine returned an error state")
        next_state = _attach_last_action_result(old_state, next_state, action)
    elif command_type == "bench_custom_sequence":
        raise SnapshotReplayError("unvalidated custom sequence reached execution")
    else:
        if result is not None:
            result.game_action_count += 1
        next_state = proc.send(action.command)
        if next_state.get("type") == "error":
            raise SnapshotReplayError(next_state.get("message") or "Headless engine returned an error state")
        action_tape.append(_copy_command(action.command))
        next_state = _attach_last_action_result(old_state, next_state, action)

    if agent is not None:
        _record_agent_transition(agent, old_state, action, meta, next_state)
    return next_state


def _save_backed_sequence_probe_factory(snapshot: dict[str, Any]) -> Callable[[], Any]:
    load = snapshot.get("load") if isinstance(snapshot.get("load"), dict) else {}
    path = str(load.get("path") or "")
    lang = str((snapshot.get("game") or {}).get("lang") or "en")

    class _SaveBackedSequenceProbe:
        def __init__(self) -> None:
            self._process = Sts2Process()

        def start(self) -> Any:
            return self._process.start()

        def start_run(self, **_kwargs: Any) -> dict[str, Any]:
            state = self._process.send({"cmd": "load_save", "path": path, "lang": lang})
            if state.get("type") == "error":
                return state
            return state

        def send(self, command: dict[str, Any]) -> dict[str, Any]:
            return self._process.send(command)

        def close(self) -> None:
            close = getattr(self._process, "close", None)
            if callable(close):
                close()

    return _SaveBackedSequenceProbe


def _match_saved_action(item: dict[str, Any], legal_actions: list[LegalAction], state: dict[str, Any]) -> LegalAction:
    del state
    sequence = item.get("sequence")
    if not isinstance(sequence, list):
        sequence = _sequence_from_action_label(item.get("label"))
    if isinstance(sequence, list) and item.get("kind") in {"combat_card_sequence", "combat_custom_sequence"}:
        return LegalAction(
            action_id=int(item.get("action_id") or -1),
            label=_custom_sequence_label(sequence),
            command={"cmd": "bench_sequence", "sequence": sequence},
            kind="combat_card_sequence",
        )

    exact = [
        action
        for action in legal_actions
        if action.kind == item.get("kind") and action.label == item.get("label")
    ]
    if len(exact) == 1:
        return exact[0]

    label_only = [action for action in legal_actions if action.label == item.get("label")]
    if len(label_only) == 1:
        return label_only[0]

    semantic = _semantic_action_match(item, legal_actions)
    if semantic is not None:
        return semantic

    action_id = item.get("action_id")
    by_id = [
        action
        for action in legal_actions
        if action.kind == item.get("kind") and action.action_id == action_id
    ]
    if len(by_id) == 1:
        return by_id[0]

    legal = "; ".join(f"{action.action_id}:{action.kind}:{action.label}" for action in legal_actions)
    raise SnapshotReplayError(
        "Could not replay saved action "
        f"step={item.get('step')} kind={item.get('kind')} label={item.get('label')!r}. "
        f"Current legal actions: {legal}"
    )


def _semantic_action_match(item: dict[str, Any], legal_actions: list[LegalAction]) -> LegalAction | None:
    kind = item.get("kind")
    label = str(item.get("label") or "")
    if kind not in {"combat_reward", "combat_reward_skip"}:
        return None
    match = re.search(r"reward\s+(\d+)", label)
    if not match:
        return None
    reward_index = int(match.group(1))
    expected_action = "claim_reward" if kind == "combat_reward" else "skip_reward"
    matches = []
    for action in legal_actions:
        if action.kind != kind:
            continue
        command = action.command
        args = command.get("args") if isinstance(command.get("args"), dict) else {}
        if command.get("action") == expected_action and args.get("reward_index") == reward_index:
            matches.append(action)
    return matches[0] if len(matches) == 1 else None


def _sequence_from_action_label(label: Any) -> list[dict[str, Any]] | None:
    text = str(label or "")
    prefix = next((item for item in ("play sequence: ", "custom sequence: ") if text.startswith(item)), None)
    if prefix is None:
        return None
    steps: list[dict[str, Any]] = []
    for part in text[len(prefix) :].split(" -> "):
        card_match = re.match(r"\s*(\d+)\b", part)
        if not card_match:
            return None
        step: dict[str, Any] = {"card_index": int(card_match.group(1))}
        target_match = re.search(r"\bon enemy\s+(\d+)\b", part)
        if target_match:
            step["target_index"] = int(target_match.group(1))
        steps.append(step)
    return steps or None


def _sequence_from_next_decision(rows: list[dict[str, Any]], action_index: int) -> list[dict[str, Any]] | None:
    for row in rows[action_index + 1 :]:
        if row.get("type") != "decision":
            continue
        state = row.get("state") if isinstance(row.get("state"), dict) else {}
        result = state.get("last_action_result") if isinstance(state.get("last_action_result"), dict) else {}
        label = result.get("action_label")
        return _sequence_from_action_label(label)
    return None


def _assert_target_matches(snapshot: dict[str, Any], state: dict[str, Any]) -> None:
    target = snapshot.get("target") or {}
    expected_decision = target.get("decision")
    if state.get("decision") != expected_decision:
        raise SnapshotReplayError(
            f"Prefix replay ended at decision {state.get('decision')!r}, expected {expected_decision!r}"
        )

    expected_summary = target.get("state_summary") if isinstance(target.get("state_summary"), dict) else {}
    actual_summary = _state_summary(state)
    mismatches = []
    for key, expected_value in expected_summary.items():
        if key == "last_action_result":
            continue
        if actual_summary.get(key) != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_summary.get(key)!r}")
    if mismatches:
        raise SnapshotReplayError("Prefix replay target state mismatch: " + "; ".join(mismatches[:8]))

    if "last_action_result" in target:
        expected_last_action_result = target.get("last_action_result")
        actual_last_action_result = state.get("last_action_result")
        if actual_last_action_result != expected_last_action_result:
            raise SnapshotReplayError(
                "Prefix replay last_action_result mismatch: "
                f"expected {expected_last_action_result!r}, got {actual_last_action_result!r}"
            )

    expected_enemies = target.get("enemy_snapshot")
    if expected_enemies is not None:
        actual_enemies = _enemy_snapshot_for_trace(state)
        if not _enemy_snapshot_matches(expected_enemies, actual_enemies):
            raise SnapshotReplayError(
                "Prefix replay enemy snapshot mismatch: "
                f"expected {expected_enemies!r}, got {actual_enemies!r}"
            )


def _enemy_snapshot_matches(expected: Any, actual: Any) -> bool:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return expected == actual
    if len(expected) != len(actual):
        return False
    for expected_enemy, actual_enemy in zip(expected, actual, strict=False):
        if not isinstance(expected_enemy, dict) or not isinstance(actual_enemy, dict):
            if expected_enemy != actual_enemy:
                return False
            continue
        for key, value in expected_enemy.items():
            if actual_enemy.get(key) != value:
                return False
    return True


def _sequence_from_action_row(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    command = action.get("cmd") if isinstance(action.get("cmd"), dict) else action.get("command")
    if isinstance(command, dict) and command.get("cmd") == "bench_sequence" and isinstance(command.get("sequence"), list):
        return command["sequence"]

    meta = row.get("agent_meta") if isinstance(row.get("agent_meta"), dict) else {}
    parsed = meta.get("parsed") if isinstance(meta.get("parsed"), dict) else {}
    sequence = parsed.get("sequence")
    if isinstance(sequence, list):
        return sequence
    return None


def _custom_sequence_label(sequence_payload: list[dict[str, Any]]) -> str:
    parts = []
    for step in sequence_payload:
        label = f"{step.get('card_index')}"
        if step.get("target_index") is not None:
            label += f" on enemy {step.get('target_index')}"
        parts.append(label)
    return "custom sequence: " + " -> ".join(parts) if parts else "custom card sequence"


def _entry_commands_since_room_entry(action_tape: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index in range(len(action_tape) - 1, -1, -1):
        command = action_tape[index]
        if command.get("cmd") == "action" and command.get("action") == "select_map_node":
            return _copy_commands(action_tape[index:])
    return _copy_commands(action_tape)


def _is_save_backed_snapshot(snapshot: dict[str, Any]) -> bool:
    load = snapshot.get("load")
    return isinstance(load, dict) and load.get("type") == SAVE_LOAD_TYPE


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _first_row(rows: list[dict[str, Any]], row_type: str) -> dict[str, Any]:
    for row in rows:
        if row.get("type") == row_type:
            return row
    raise CombatSnapshotError(f"Trace does not contain a {row_type!r} row")


def _game_from_rows(start_row: dict[str, Any], run_config: Any) -> dict[str, Any]:
    config_game = run_config.get("game") if isinstance(run_config, dict) else {}
    if not isinstance(config_game, dict):
        config_game = {}
    benchmark = run_config.get("benchmark") if isinstance(run_config, dict) else {}
    if not isinstance(benchmark, dict):
        benchmark = {}
    seeds = benchmark.get("seeds") if isinstance(benchmark, dict) else None
    seed = start_row.get("seed")
    if seed is None and isinstance(seeds, list) and seeds:
        seed = seeds[0]
    return {
        "seed": seed,
        "character": start_row.get("character") or config_game.get("character") or "Ironclad",
        "ascension": start_row.get("ascension", config_game.get("ascension", 0)),
        "lang": start_row.get("lang") or config_game.get("lang") or "en",
    }


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    if snapshot.get("type") != "combat_snapshot":
        raise CombatSnapshotError("Snapshot type must be 'combat_snapshot'")
    if snapshot.get("version") != SNAPSHOT_VERSION:
        raise CombatSnapshotError(f"Unsupported combat snapshot version: {snapshot.get('version')!r}")
    if not isinstance(snapshot.get("game"), dict):
        raise CombatSnapshotError("Snapshot is missing game metadata")
    if not isinstance(snapshot.get("target"), dict):
        raise CombatSnapshotError("Snapshot is missing target metadata")
    if not isinstance(snapshot.get("prefix_actions"), list):
        raise CombatSnapshotError("Snapshot prefix_actions must be a list")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract and run replay-prefix combat snapshots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract a combat entry snapshot from a JSONL trace")
    extract_parser.add_argument("--trace", required=True)
    extract_parser.add_argument("--start-step", required=True, type=int)
    extract_parser.add_argument("--snapshot-id")
    extract_parser.add_argument("--out", required=True)

    verify_parser = subparsers.add_parser("verify", help="Replay a snapshot prefix and verify the target state")
    verify_parser.add_argument("--snapshot", required=True)
    verify_parser.add_argument("--allow-repeat-views", action=argparse.BooleanOptionalAction, default=False)
    verify_parser.add_argument("--allow-custom-sequences", action=argparse.BooleanOptionalAction, default=True)
    verify_parser.add_argument("--custom-sequence-exposure", choices=["safe", "eligible"], default="safe")
    verify_parser.add_argument("--combat-card-sequences", action=argparse.BooleanOptionalAction, default=True)
    verify_parser.add_argument("--precomputed-combat-card-sequences", action=argparse.BooleanOptionalAction, default=False)

    run_parser = subparsers.add_parser("run", help="Run one agent from a combat snapshot until the fight ends")
    run_parser.add_argument("--snapshot", required=True)
    _add_run_args(run_parser)

    args = parser.parse_args(argv)
    if args.command == "extract":
        snapshot = extract_combat_snapshot(args.trace, start_step=args.start_step, snapshot_id=args.snapshot_id)
        write_combat_snapshot(snapshot, args.out)
        print(json.dumps({"snapshot": args.out, "prefix_actions": len(snapshot["prefix_actions"])}, ensure_ascii=False))
        return 0

    if args.command == "verify":
        snapshot = load_combat_snapshot(args.snapshot)
        replay = replay_snapshot_prefix(
            snapshot,
            allow_repeat_views=args.allow_repeat_views,
            allow_card_sequences=args.combat_card_sequences,
            allow_precomputed_card_sequences=args.precomputed_combat_card_sequences,
            allow_custom_sequences=args.allow_custom_sequences,
            custom_sequence_exposure=args.custom_sequence_exposure,
        )
        close = getattr(replay.process, "close", None)
        if callable(close):
            close()
        print(
            json.dumps(
                {
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "verified": True,
                    "prefix_actions": replay.prefix_actions_executed,
                    "state": _state_summary(replay.state),
                    "enemy_snapshot": _enemy_snapshot_for_trace(replay.state),
                },
                ensure_ascii=False,
            )
        )
        return 0

    snapshot = load_combat_snapshot(args.snapshot)
    logger = JsonlLogger(args.out)
    try:
        logger.write({"type": "snapshot_run_config", "config": _snapshot_run_config(args, snapshot)})
        conversation_mode = "single_turn" if args.agent == "random" else args.context_mode
        agent = agent_from_args(
            args.agent,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            llm_timeout=args.llm_timeout,
            llm_max_retries=args.llm_max_retries,
            llm_fallback_mode=args.llm_fallback_mode,
            llm_temperature=args.llm_temperature,
            include_json_state=args.include_json_state,
            prompt_style=args.prompt_style,
            short_term_memory_enabled=args.short_term_memory_enabled,
            short_term_memory_window=args.short_term_memory_window,
            short_term_memory_mode=args.short_term_memory_mode,
            run_summary_enabled=args.run_summary,
            conversation_mode=conversation_mode,
            turn_chat_window=args.turn_chat_window,
            combat_dialogue_history_turns=args.combat_dialogue_history_turns,
            turn_chat_update_mode="delta",
            turn_chat_assistant_history="compact",
            turn_chat_plan_enabled=args.turn_check,
            turn_check_enabled=args.turn_check,
            route_plan_enabled=args.route_plan_enabled,
            route_plan_enforcement=args.route_plan_enforcement,
            reward_posterior_evaluator_enabled=args.reward_posterior_evaluator,
            self_consistency_enabled=args.self_consistency_enabled,
            self_consistency_samples=args.self_consistency_samples,
            self_consistency_decisions=args.self_consistency_decisions,
            self_consistency_mode=args.self_consistency_mode,
            action_prompt_order=args.action_prompt_order,
            action_prompt_order_seed=args.action_prompt_order_seed,
            combat_card_sequences_enabled=args.combat_card_sequences,
            combat_reasoning_pool_enabled=args.combat_reasoning_pool,
            combat_turn_board_enabled=args.combat_turn_board,
            combat_skill_hints_enabled=args.combat_skill_hints,
            combat_skill_hints_profile=args.combat_skill_hints_profile,
            combat_skill_hints_max_hints=args.combat_skill_hints_max_hints,
            combat_skill_lines_enabled=args.combat_skill_lines,
            rule_projection_and_recent_changes_enabled=args.rule_projection_and_recent_changes_enabled,
        )
        result = run_combat_snapshot(
            snapshot=snapshot,
            agent=agent,
            logger=logger,
            max_steps=args.max_steps,
            max_game_actions=args.max_game_actions or None,
            allow_repeat_views=args.allow_repeat_views,
            allow_custom_sequences=args.allow_custom_sequences,
            custom_sequence_exposure=args.custom_sequence_exposure,
            combat_card_sequences_enabled=args.combat_card_sequences,
            precomputed_combat_card_sequences_enabled=args.precomputed_combat_card_sequences,
            log_level=args.log_level,
            print_prompts=args.print_prompts,
            print_model_output=args.print_model_output,
            route_plan_enabled=args.route_plan_enabled,
            route_plan_enforcement=args.route_plan_enforcement,
        )
        print(json.dumps({"type": "snapshot_result", "result": result.to_dict()}, ensure_ascii=False), flush=True)
        return 1 if result.error else 0
    finally:
        logger.close()


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    from .run_benchmark import env_bool, env_float, env_int, env_str, load_benchmark_env

    load_benchmark_env()
    parser.add_argument(
        "--agent",
        choices=["random", "llm", "coordinator", "formal_generalist"],
        default=env_str("STS2_BENCH_AGENT", "random"),
    )
    parser.add_argument("--base-url", default=env_str("STS2_BENCH_BASE_URL"))
    parser.add_argument("--model", default=env_str("STS2_BENCH_MODEL"))
    parser.add_argument("--api-key", default=env_str("DEEPSEEK_API_KEY", "local", fallbacks=("OPENAI_API_KEY",)))
    parser.add_argument("--llm-timeout", type=float, default=env_float("STS2_BENCH_LLM_TIMEOUT", 120.0))
    parser.add_argument("--llm-max-retries", type=int, default=env_int("STS2_BENCH_LLM_MAX_RETRIES", 2))
    parser.add_argument("--llm-fallback-mode", choices=["error", "action"], default=env_str("STS2_BENCH_LLM_FALLBACK_MODE", "error"))
    parser.add_argument("--llm-temperature", type=float, default=env_float("STS2_BENCH_LLM_TEMPERATURE", 0.0))
    parser.add_argument("--prompt-style", choices=["default", "analysis"], default=env_str("STS2_BENCH_PROMPT_STYLE", "default"))
    parser.add_argument("--include-json-state", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_INCLUDE_JSON_STATE", False))
    parser.add_argument("--short-term-memory", action=argparse.BooleanOptionalAction, dest="short_term_memory_enabled", default=env_bool("STS2_BENCH_SHORT_TERM_MEMORY_ENABLED", False))
    parser.add_argument("--short-term-memory-window", type=int, default=env_int("STS2_BENCH_SHORT_TERM_MEMORY_WINDOW", 8))
    parser.add_argument("--short-term-memory-mode", choices=["action_reason", "factual_diff"], default=env_str("STS2_BENCH_SHORT_TERM_MEMORY_MODE", "action_reason"))
    parser.add_argument("--run-summary", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_RUN_SUMMARY_ENABLED", True))
    parser.add_argument(
        "--context-mode",
        choices=[
            "single_turn",
            "turn_chat",
            "combat_mdp_state",
            "combat_state_only",
            "combat_dialogue",
        ],
        default=env_str("STS2_BENCH_CONTEXT_MODE", "combat_mdp_state"),
    )
    parser.add_argument("--turn-chat-window", type=int, default=env_int("STS2_BENCH_TURN_CHAT_WINDOW", 4))
    parser.add_argument(
        "--combat-dialogue-history-turns",
        type=int,
        default=env_int("STS2_BENCH_COMBAT_DIALOGUE_HISTORY_TURNS", 3),
    )
    parser.add_argument("--turn-check", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_TURN_CHECK_ENABLED", False))
    parser.add_argument("--route-plan", action=argparse.BooleanOptionalAction, dest="route_plan_enabled", default=env_bool("STS2_BENCH_ROUTE_PLAN_ENABLED", False))
    parser.add_argument("--route-plan-enforcement", choices=["hard", "soft"], default=env_str("STS2_BENCH_ROUTE_PLAN_ENFORCEMENT", "hard"))
    parser.add_argument("--reward-posterior-evaluator", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_REWARD_POSTERIOR_EVALUATOR", True))
    parser.add_argument(
        "--rule-projection-and-recent-changes",
        action=argparse.BooleanOptionalAction,
        dest="rule_projection_and_recent_changes_enabled",
        default=env_bool("STS2_BENCH_RULE_PROJECTION_AND_RECENT_CHANGES", True),
    )
    parser.add_argument("--self-consistency", action=argparse.BooleanOptionalAction, dest="self_consistency_enabled", default=env_bool("STS2_BENCH_SELF_CONSISTENCY", False))
    parser.add_argument("--self-consistency-samples", type=int, default=env_int("STS2_BENCH_SELF_CONSISTENCY_SAMPLES", 3))
    parser.add_argument("--self-consistency-decisions", default=env_str("STS2_BENCH_SELF_CONSISTENCY_DECISIONS", "combat_play"))
    parser.add_argument("--self-consistency-mode", choices=["diagnostic", "majority_vote"], default=env_str("STS2_BENCH_SELF_CONSISTENCY_MODE", "diagnostic"))
    parser.add_argument("--action-prompt-order", choices=["default", "shuffle_map"], default=env_str("STS2_BENCH_ACTION_PROMPT_ORDER", "default"))
    parser.add_argument("--action-prompt-order-seed", type=int, default=env_int("STS2_BENCH_ACTION_PROMPT_ORDER_SEED", 0))
    parser.add_argument("--allow-repeat-views", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_ALLOW_REPEAT_VIEWS", False))
    parser.add_argument("--allow-custom-sequences", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_ALLOW_CUSTOM_SEQUENCES", True))
    parser.add_argument(
        "--custom-sequence-exposure",
        choices=["safe", "eligible"],
        default=env_str("STS2_BENCH_CUSTOM_SEQUENCE_EXPOSURE", "safe"),
        help="safe keeps conservative gateway gating; eligible exposes combat_custom_sequence whenever the hand has a possible custom sequence entrypoint",
    )
    parser.add_argument(
        "--combat-card-sequences",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_COMBAT_CARD_SEQUENCES", True),
        help="Allow model-generated combat_custom_sequence payloads and prompt rules",
    )
    parser.add_argument(
        "--precomputed-combat-card-sequences",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_PRECOMPUTED_COMBAT_CARD_SEQUENCES", False),
        help="Expose harness-precomputed combat_card_sequence legal actions",
    )
    parser.add_argument("--combat-reasoning-pool", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_COMBAT_REASONING_POOL", False))
    parser.add_argument("--combat-turn-board", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_COMBAT_TURN_BOARD", False))
    parser.add_argument("--combat-skill-hints", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_COMBAT_SKILL_HINTS", False))
    parser.add_argument("--combat-skill-hints-profile", choices=["baseline", "survival", "targeting", "full"], default=env_str("STS2_BENCH_COMBAT_SKILL_HINTS_PROFILE", "baseline"))
    parser.add_argument("--combat-skill-hints-max-hints", type=int, default=env_int("STS2_BENCH_COMBAT_SKILL_HINTS_MAX_HINTS", 4))
    parser.add_argument("--combat-skill-lines", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_COMBAT_SKILL_LINES", True))
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--no-max-steps",
        dest="max_steps",
        action="store_const",
        const=0,
        help="Run until natural combat termination without a combat step cap.",
    )
    parser.add_argument("--max-game-actions", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--log-level", choices=["metrics", "decisions", "full"], default=env_str("STS2_BENCH_LOG_LEVEL", "decisions"))
    parser.add_argument("--print-prompts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-model-output", action=argparse.BooleanOptionalAction, default=False)


def _snapshot_run_config(args: argparse.Namespace, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_source": snapshot.get("source"),
        "snapshot_game": snapshot.get("game"),
        "agent": {
            "type": args.agent,
            "base_url": args.base_url,
            "model": args.model,
            "llm_timeout": args.llm_timeout,
            "llm_max_retries": args.llm_max_retries,
            "context_mode": args.context_mode,
            "combat_context": {
                "variant": args.context_mode,
                "dialogue_history_turns": args.combat_dialogue_history_turns,
                "history_representation": {
                    "combat_state_only": "none",
                    "combat_dialogue": "raw_dialogue",
                    "combat_mdp_state": "managed_structured",
                    "turn_chat": "legacy_turn_chat",
                    "single_turn": "legacy_none",
                }[args.context_mode],
            },
            "turn_check": args.turn_check,
            "route_plan_enabled": args.route_plan_enabled,
            "route_plan_enforcement": args.route_plan_enforcement,
            "self_consistency_enabled": args.self_consistency_enabled,
            "action_prompt_order": args.action_prompt_order,
            "combat_reasoning_pool": args.combat_reasoning_pool,
            "combat_turn_board": args.combat_turn_board,
            "combat_skill_lines": args.combat_skill_lines,
            "rule_projection_and_recent_changes": {
                "enabled": args.rule_projection_and_recent_changes_enabled,
            },
            "combat_skill_hints": {
                "enabled": args.combat_skill_hints,
                "profile": args.combat_skill_hints_profile,
                "max_hints": args.combat_skill_hints_max_hints,
            },
        },
        "runner": {
            "allow_custom_sequences": args.allow_custom_sequences,
            "custom_sequence_exposure": args.custom_sequence_exposure,
            "combat_card_sequences": args.combat_card_sequences,
            "precomputed_combat_card_sequences": args.precomputed_combat_card_sequences,
            "allow_repeat_views": args.allow_repeat_views,
            "max_steps": args.max_steps,
            "max_game_actions": args.max_game_actions,
            "log_level": args.log_level,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
