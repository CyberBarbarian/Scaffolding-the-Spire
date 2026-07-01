"""Save/load assisted full-game runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .actions import CustomSequenceExposure, LegalAction, build_legal_actions
from .agents import Agent
from .combat_snapshot import (
    SAVE_LOAD_TYPE,
    SNAPSHOT_VERSION,
    _execute_action,
    _save_backed_sequence_probe_factory,
    create_save_backed_combat_snapshot,
    replay_snapshot_prefix,
    write_combat_snapshot,
)
from .evidence import build_evidence_bundle
from .process import Sts2Process
from .route_plan import RoutePlanTracker
from .run_benchmark import env_bool, env_float, env_int, env_str, load_benchmark_env, load_seeds
from .runner import (
    JsonlLogger,
    LogLevel,
    RoutePlanEnforcement,
    RunResult,
    _action_log_entry,
    _agent_run_context_store,
    _allow_card_sequences_for_agent,
    _allow_custom_sequences_for_agent,
    _attach_route_plan_context,
    _choose_action_with_runner_validation,
    _copy_command,
    _decision_log_entry,
    _enemy_snapshot_for_trace,
    _fill_result,
    _max_game_actions_reached,
    _reset_agent_episode,
    agent_from_args,
    summarize,
)


COMBAT_DECISIONS = {"combat_play", "card_select"}


@dataclass(frozen=True)
class ReplayAttempt:
    attempt_id: str
    outcome: str
    hp: int | None
    steps: int
    commands: list[dict[str, Any]]


@dataclass
class _ActiveCheckpoint:
    checkpoint_id: str
    directory: Path
    snapshot: dict[str, Any]
    entry_hp: int | None
    entry_step: int
    run_tape_at_entry: list[dict[str, Any]]


@dataclass
class _ReplayRun:
    attempt: ReplayAttempt
    state: dict[str, Any]
    process: Sts2Process
    replay_tape: list[dict[str, Any]]
    entry_command_count: int
    result: RunResult
    trace_path: Path


@dataclass
class _ResumeReplay:
    state: dict[str, Any]
    action_tape: list[dict[str, Any]]
    route_plan_tracker: RoutePlanTracker | None
    next_step: int
    checkpoint_count: int


def build_snapshot_from_combat_entry(
    entry_snapshot: dict[str, Any],
    *,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Convert a CombatRecorder entry snapshot into a combat-snapshot payload."""

    run = entry_snapshot.get("run") if isinstance(entry_snapshot.get("run"), dict) else {}
    entry = entry_snapshot.get("entry") if isinstance(entry_snapshot.get("entry"), dict) else {}
    save = entry_snapshot.get("save") if isinstance(entry_snapshot.get("save"), dict) else {}
    save_path = str(save.get("path") or "")
    combat_id = str(entry_snapshot.get("combat_id") or "combat")
    prefix_commands = entry.get("prefix_commands") if isinstance(entry.get("prefix_commands"), list) else []
    entry_commands = _entry_commands_since_room_entry(prefix_commands)

    return {
        "type": "combat_snapshot",
        "version": SNAPSHOT_VERSION,
        "snapshot_id": snapshot_id or combat_id,
        "source": {
            "kind": SAVE_LOAD_TYPE,
            "checkpoint_path": save_path,
            "from_combat_entry": combat_id,
        },
        "game": {
            "seed": run.get("seed"),
            "character": run.get("character") or "Ironclad",
            "ascension": int(run.get("ascension") or 0),
            "lang": run.get("lang") or "en",
        },
        "target": {
            "step": entry.get("step"),
            "decision": entry.get("decision"),
            "state_summary": _copy_json(entry.get("state") or {}),
            "enemy_snapshot": _copy_json(entry.get("enemies") or []),
        },
        "stop_policy": {
            "active_decisions": ["combat_play", "card_select"],
            "description": "Load the combat-entry save, replay room-entry commands, then run until combat exits.",
        },
        "prefix_actions": [],
        "load": {
            "type": SAVE_LOAD_TYPE,
            "path": save_path,
            "entry_commands": entry_commands,
        },
    }


def select_best_attempt(attempts: list[ReplayAttempt]) -> ReplayAttempt:
    if not attempts:
        raise ValueError("attempts must not be empty")
    return max(attempts, key=_attempt_score)


def merge_replay_commands_into_run_tape(
    run_tape_at_entry: list[dict[str, Any]],
    replay_tape: list[dict[str, Any]],
    *,
    replay_entry_command_count: int,
) -> list[dict[str, Any]]:
    """Append replayed combat commands without duplicating room-entry commands."""

    if replay_entry_command_count < 0:
        raise ValueError("replay_entry_command_count must be non-negative")
    return [
        *_copy_json(run_tape_at_entry),
        *_copy_json(replay_tape[replay_entry_command_count:]),
    ]


def _sequence_validation_tape_for_checkpoint(
    checkpoint: _ActiveCheckpoint,
    action_tape: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    load = checkpoint.snapshot.get("load") if isinstance(checkpoint.snapshot.get("load"), dict) else {}
    entry_commands = load.get("entry_commands") if isinstance(load.get("entry_commands"), list) else []
    combat_commands = action_tape[len(checkpoint.run_tape_at_entry) :]
    return [
        *_copy_json(entry_commands),
        *_copy_json(combat_commands),
    ]


def run_sl_one(
    *,
    agent_factory: Callable[[], Agent],
    seed: str,
    character: str = "Ironclad",
    ascension: int = 0,
    lang: str = "en",
    max_steps: int = 2000,
    max_game_actions: int | None = None,
    logger: JsonlLogger | None = None,
    sl_dir: Path | str,
    sl_hp_loss_threshold: int = 20,
    sl_retries: int = 2,
    sl_max_combat_steps: int = 120,
    log_level: LogLevel = "decisions",
    include_full_map: bool = False,
    allow_repeat_views: bool = False,
    allow_custom_sequences: bool | None = None,
    custom_sequence_exposure: CustomSequenceExposure = "safe",
    combat_card_sequences_enabled: bool = True,
    precomputed_combat_card_sequences_enabled: bool = False,
    route_plan_enabled: bool = True,
    route_plan_enforcement: RoutePlanEnforcement = "hard",
    resume_trace: Path | str | None = None,
    resume_before_step: int | None = None,
) -> RunResult:
    """Run one full game, replaying high-loss combats from saved entry checkpoints."""

    if route_plan_enforcement not in {"hard", "soft"}:
        raise ValueError(f"Unsupported route_plan_enforcement: {route_plan_enforcement}")

    sl_root = Path(sl_dir)
    sl_root.mkdir(parents=True, exist_ok=True)
    agent = agent_factory()
    proc = Sts2Process()
    result = RunResult(seed=seed, character=character, ascension=ascension)
    started = time.time()
    action_tape: list[dict[str, Any]] = []
    route_plan_tracker = RoutePlanTracker() if route_plan_enabled else None
    checkpoint: _ActiveCheckpoint | None = None
    checkpoint_count = 0
    state: dict[str, Any] | None = None

    try:
        proc.start()
        _reset_agent_episode(agent)
        state = proc.start_run(character=character, seed=seed, ascension=ascension, lang=lang)
        start_step = 0
        logger and logger.write(
            {
                "type": "start",
                "seed": seed,
                "character": character,
                "ascension": ascension,
                "lang": lang,
                "sl_hp_loss_threshold": sl_hp_loss_threshold,
                "sl_retries": sl_retries,
                "allow_card_sequences": combat_card_sequences_enabled,
                "allow_custom_sequences": allow_custom_sequences,
                "custom_sequence_exposure": custom_sequence_exposure,
                "route_plan_enabled": route_plan_enabled,
                "route_plan_enforcement": route_plan_enforcement,
            }
        )
        if resume_trace is not None:
            replay = _replay_trace_prefix_for_resume(
                process=proc,
                state=state,
                trace_path=Path(resume_trace),
                before_step=resume_before_step,
                route_plan_enabled=route_plan_enabled,
                action_tape=action_tape,
                result=result,
                character=character,
                seed=seed,
                ascension=ascension,
                lang=lang,
            )
            state = replay.state
            action_tape = replay.action_tape
            route_plan_tracker = replay.route_plan_tracker
            start_step = replay.next_step
            checkpoint_count = replay.checkpoint_count
            logger and logger.write(
                {
                    "type": "resume_replay",
                    "source_trace": str(resume_trace),
                    "next_step": replay.next_step,
                    "action_tape_length": len(action_tape),
                    "checkpoint_count": checkpoint_count,
                    "route_plan_active": bool(route_plan_tracker and route_plan_tracker.active),
                }
            )

        for step in range(start_step, max_steps):
            decision = state.get("decision")
            if decision == "game_over":
                _fill_result(result, state)
                return result

            if include_full_map and decision == "map_select" and "full_map" not in state:
                map_state = proc.send({"cmd": "get_map"})
                if map_state.get("type") == "map":
                    state = {**state, "full_map": map_state}
            if route_plan_tracker is not None:
                state = _attach_route_plan_context(state, route_plan_tracker)

            if checkpoint is None and _is_combat_state(state):
                checkpoint_count += 1
                checkpoint = _create_checkpoint(
                    process=proc,
                    state=state,
                    action_tape=action_tape,
                    sl_root=sl_root,
                    checkpoint_index=checkpoint_count,
                    seed=seed,
                    character=character,
                    ascension=ascension,
                    lang=lang,
                    step=step,
                    logger=logger,
                    result=result,
                )

            legal_actions = build_legal_actions(
                state,
                allow_repeat_views=allow_repeat_views,
                allow_card_sequences=combat_card_sequences_enabled and _allow_card_sequences_for_agent(agent),
                allow_precomputed_card_sequences=(
                    precomputed_combat_card_sequences_enabled and _allow_card_sequences_for_agent(agent)
                ),
                allow_custom_sequences=(
                    _allow_custom_sequences_for_agent(agent) if allow_custom_sequences is None else allow_custom_sequences
                ),
                custom_sequence_exposure=custom_sequence_exposure,
            )
            evidence_bundle = build_evidence_bundle(
                state,
                legal_actions,
                store=_agent_run_context_store(agent),
            )
            if logger:
                decision_entry = _decision_log_entry(
                    log_level=log_level,
                    step=step,
                    state=state,
                    legal_actions=legal_actions,
                    prompt=None,
                    evidence_bundle=evidence_bundle,
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

            sequence_probe_factory = (
                _save_backed_sequence_probe_factory(checkpoint.snapshot)
                if checkpoint is not None
                else None
            )
            sequence_validation_tape = (
                _sequence_validation_tape_for_checkpoint(checkpoint, action_tape)
                if checkpoint is not None
                else action_tape
            )
            action, meta, validation_error = _choose_action_with_runner_validation(
                agent=agent,
                state=state,
                legal_actions=legal_actions,
                prompt=None,
                result=result,
                logger=logger,
                log_level=log_level,
                step=step,
                decision=decision,
                print_model_output=False,
                action_tape=sequence_validation_tape,
                character=character,
                seed=seed,
                ascension=ascension,
                lang=lang,
                sequence_probe_factory=sequence_probe_factory,
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

            if checkpoint is not None and not _is_combat_state(state):
                replay = _maybe_replay_checkpoint(
                    checkpoint=checkpoint,
                    original_state=state,
                    original_tape=action_tape,
                    agent_factory=agent_factory,
                    character=character,
                    seed=seed,
                    ascension=ascension,
                    lang=lang,
                    allow_repeat_views=allow_repeat_views,
                    allow_custom_sequences=allow_custom_sequences,
                    custom_sequence_exposure=custom_sequence_exposure,
                    combat_card_sequences_enabled=combat_card_sequences_enabled,
                    precomputed_combat_card_sequences_enabled=precomputed_combat_card_sequences_enabled,
                    log_level=log_level,
                    route_plan_enabled=route_plan_enabled,
                    route_plan_enforcement=route_plan_enforcement,
                    sl_hp_loss_threshold=sl_hp_loss_threshold,
                    sl_retries=sl_retries,
                    sl_max_combat_steps=sl_max_combat_steps,
                    logger=logger,
                    result=result,
                )
                checkpoint = None
                if replay is not None:
                    proc.close()
                    proc = replay.process
                    state = replay.state
                    action_tape = merge_replay_commands_into_run_tape(
                        replay.attempt.commands,
                        replay.replay_tape,
                        replay_entry_command_count=replay.entry_command_count,
                    )

            if _max_game_actions_reached(result, max_game_actions, state):
                return result

        result.truncated = True
        result.extra["truncation_reason"] = f"max_steps exceeded ({max_steps})"
        _fill_result(result, state)
        return result
    except Exception as exc:  # noqa: BLE001 - experiment runner reports per-run failure.
        result.error = str(exc)
        if isinstance(state, dict):
            _fill_result(result, state)
        return result
    finally:
        result.wall_time_sec = time.time() - started
        logger and logger.write({"type": "result", "result": result.to_dict()})
        proc.close()


def _replay_trace_prefix_for_resume(
    *,
    process: Sts2Process,
    state: dict[str, Any],
    trace_path: Path,
    before_step: int | None,
    route_plan_enabled: bool,
    action_tape: list[dict[str, Any]],
    result: RunResult,
    character: str,
    seed: str,
    ascension: int,
    lang: str,
) -> _ResumeReplay:
    rows = _read_trace_rows(trace_path)
    cut_step = before_step if before_step is not None else _infer_resume_cut_step(rows)
    route_plan_tracker = RoutePlanTracker() if route_plan_enabled else None
    replay_tape = action_tape
    checkpoint_count = sum(1 for row in rows if row.get("type") == "sl_checkpoint" and _row_step(row) < cut_step)

    for row in rows:
        if row.get("type") != "action":
            continue
        step = _row_step(row)
        if step >= cut_step:
            continue
        if route_plan_tracker is not None:
            state = _attach_route_plan_context(state, route_plan_tracker)
        legal_actions = build_legal_actions(
            state,
            allow_repeat_views=True,
            allow_card_sequences=True,
            allow_precomputed_card_sequences=False,
            allow_custom_sequences=True,
            custom_sequence_exposure="eligible",
        )
        action = _action_from_trace_row(row, state=state, legal_actions=legal_actions)
        meta = {"parsed": ((row.get("agent_meta") or {}).get("parsed") or {})}
        if route_plan_tracker is not None and state.get("decision") == "map_select" and action.kind == "map":
            route_plan_result = route_plan_tracker.validate_and_apply(
                state=state,
                legal_actions=legal_actions,
                action=action,
                meta=meta,
                step=step,
            )
            if route_plan_result.error is not None:
                error = route_plan_result.error
                raise RuntimeError(f"resume route-plan replay failed at step {step}: {error.code}: {error.message}")
        state = _execute_action(
            proc=process,
            state=state,
            action=action,
            action_tape=replay_tape,
            agent=None,
            meta=meta,
            result=result,
        )
        result.steps = max(result.steps, step + 1)

    return _ResumeReplay(
        state=state,
        action_tape=replay_tape,
        route_plan_tracker=route_plan_tracker,
        next_step=cut_step,
        checkpoint_count=checkpoint_count,
    )


def _read_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _infer_resume_cut_step(rows: list[dict[str, Any]]) -> int:
    for row in rows:
        if row.get("type") == "route_plan_validation_error":
            return _row_step(row)
    for row in rows:
        if row.get("type") == "result":
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            steps = result.get("steps")
            if isinstance(steps, int):
                return steps
    raise ValueError("Could not infer resume cut step from trace")


def _row_step(row: dict[str, Any]) -> int:
    value = row.get("step")
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _action_from_trace_row(
    row: dict[str, Any],
    *,
    state: dict[str, Any],
    legal_actions: list[LegalAction],
) -> LegalAction:
    command = row.get("command") if isinstance(row.get("command"), dict) else {}
    action_payload = row.get("action") if isinstance(row.get("action"), dict) else {}
    if command.get("cmd") == "bench_custom_sequence":
        parsed = (row.get("agent_meta") or {}).get("parsed") if isinstance(row.get("agent_meta"), dict) else {}
        sequence = parsed.get("sequence") if isinstance(parsed, dict) else None
        if not isinstance(sequence, list):
            raise RuntimeError(f"Trace custom sequence at step {row.get('step')} has no sequence payload")
        return LegalAction(
            action_id=int(action_payload.get("action_id") or 0),
            label=str(action_payload.get("label") or "replay custom sequence"),
            command={"cmd": "bench_sequence", "sequence": sequence},
            kind="combat_card_sequence",
        )
    for action in legal_actions:
        if _commands_equal(action.command, command):
            return action
    label = action_payload.get("label")
    if isinstance(label, str):
        for action in legal_actions:
            if action.label == label:
                return action
    raise RuntimeError(
        f"Could not match trace action at step {row.get('step')} decision {state.get('decision')}: {command}"
    )


def _commands_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _create_checkpoint(
    *,
    process: Sts2Process,
    state: dict[str, Any],
    action_tape: list[dict[str, Any]],
    sl_root: Path,
    checkpoint_index: int,
    seed: str,
    character: str,
    ascension: int,
    lang: str,
    step: int,
    logger: JsonlLogger | None,
    result: RunResult,
) -> _ActiveCheckpoint | None:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    act = state.get("act") or context.get("act") or 0
    floor = state.get("floor") or context.get("floor") or 0
    room_type = str(context.get("room_type") or state.get("room_type") or "Combat")
    checkpoint_id = f"sl_ckpt_{checkpoint_index:03d}_A{act}F{floor}_{_safe_id(room_type)}"
    directory = sl_root / checkpoint_id
    save_path = directory / "entry.save"
    try:
        snapshot = create_save_backed_combat_snapshot(
            process=process,
            state=state,
            action_tape=action_tape,
            save_path=save_path,
            snapshot_id=checkpoint_id,
            game={"seed": seed, "character": character, "ascension": ascension, "lang": lang},
            step=step,
            source={"sl_checkpoint": True},
        )
        write_combat_snapshot(snapshot, directory / "snapshot.json")
    except Exception as exc:  # noqa: BLE001 - checkpointing should not kill the run.
        result.extra["sl_checkpoint_error_count"] = int(result.extra.get("sl_checkpoint_error_count") or 0) + 1
        logger and logger.write(
            {
                "type": "sl_checkpoint_error",
                "step": step,
                "error": str(exc),
                "state": {"act": act, "floor": floor, "room_type": room_type, "hp": _hp(state)},
            }
        )
        return None

    result.extra["sl_checkpoint_count"] = int(result.extra.get("sl_checkpoint_count") or 0) + 1
    logger and logger.write(
        {
            "type": "sl_checkpoint",
            "checkpoint_id": checkpoint_id,
            "step": step,
            "act": act,
            "floor": floor,
            "room_type": room_type,
            "entry_hp": _hp(state),
            "snapshot": str(directory / "snapshot.json"),
            "save": str(save_path),
        }
    )
    return _ActiveCheckpoint(
        checkpoint_id=checkpoint_id,
        directory=directory,
        snapshot=snapshot,
        entry_hp=_hp(state),
        entry_step=step,
        run_tape_at_entry=_copy_json(action_tape),
    )


def _maybe_replay_checkpoint(
    *,
    checkpoint: _ActiveCheckpoint,
    original_state: dict[str, Any],
    original_tape: list[dict[str, Any]],
    agent_factory: Callable[[], Agent],
    character: str,
    seed: str,
    ascension: int,
    lang: str,
    allow_repeat_views: bool,
    allow_custom_sequences: bool | None,
    custom_sequence_exposure: CustomSequenceExposure,
    combat_card_sequences_enabled: bool,
    precomputed_combat_card_sequences_enabled: bool,
    log_level: LogLevel,
    route_plan_enabled: bool,
    route_plan_enforcement: RoutePlanEnforcement,
    sl_hp_loss_threshold: int,
    sl_retries: int,
    sl_max_combat_steps: int,
    logger: JsonlLogger | None,
    result: RunResult,
) -> _ReplayRun | None:
    exit_hp = _hp(original_state)
    hp_loss = (
        checkpoint.entry_hp - exit_hp
        if checkpoint.entry_hp is not None and exit_hp is not None
        else None
    )
    terminal_death = original_state.get("decision") == "game_over" and original_state.get("victory") is not True
    if not terminal_death and (hp_loss is None or hp_loss < sl_hp_loss_threshold):
        return None

    result.extra["sl_trigger_count"] = int(result.extra.get("sl_trigger_count") or 0) + 1
    logger and logger.write(
        {
            "type": "sl_trigger",
            "checkpoint_id": checkpoint.checkpoint_id,
            "entry_hp": checkpoint.entry_hp,
            "exit_hp": exit_hp,
            "hp_loss": hp_loss,
            "terminal_death": terminal_death,
            "retries": sl_retries,
        }
    )
    original = ReplayAttempt(
        attempt_id="original",
        outcome=_combat_exit_outcome(original_state),
        hp=exit_hp,
        steps=max(0, result.steps - checkpoint.entry_step),
        commands=_copy_json(checkpoint.run_tape_at_entry),
    )
    attempts: list[ReplayAttempt] = [original]
    replay_runs: dict[str, _ReplayRun] = {}
    for retry_index in range(1, sl_retries + 1):
        replay = _run_replay_attempt(
            checkpoint=checkpoint,
            attempt_index=retry_index,
            agent_factory=agent_factory,
            character=character,
            seed=seed,
            ascension=ascension,
            lang=lang,
            allow_repeat_views=allow_repeat_views,
            allow_custom_sequences=allow_custom_sequences,
            custom_sequence_exposure=custom_sequence_exposure,
            combat_card_sequences_enabled=combat_card_sequences_enabled,
            precomputed_combat_card_sequences_enabled=precomputed_combat_card_sequences_enabled,
            log_level=log_level,
            route_plan_enabled=route_plan_enabled,
            route_plan_enforcement=route_plan_enforcement,
            max_steps=sl_max_combat_steps,
        )
        attempts.append(replay.attempt)
        replay_runs[replay.attempt.attempt_id] = replay
        logger and logger.write(
            {
                "type": "sl_attempt",
                "checkpoint_id": checkpoint.checkpoint_id,
                "attempt": replay.attempt.attempt_id,
                "outcome": replay.attempt.outcome,
                "hp": replay.attempt.hp,
                "steps": replay.attempt.steps,
                "trace": str(replay.trace_path),
                "error": replay.result.error,
                "fallbacks": replay.result.model_fallback_count,
            }
        )

    best = select_best_attempt(attempts)
    if best.attempt_id == "original":
        for replay in replay_runs.values():
            replay.process.close()
        logger and logger.write(
            {
                "type": "sl_adopt",
                "checkpoint_id": checkpoint.checkpoint_id,
                "adopted": "original",
                "original_hp": original.hp,
                "best_hp": best.hp,
            }
        )
        return None

    adopted = replay_runs[best.attempt_id]
    for attempt_id, replay in replay_runs.items():
        if attempt_id != best.attempt_id:
            replay.process.close()
    result.extra["sl_adoption_count"] = int(result.extra.get("sl_adoption_count") or 0) + 1
    logger and logger.write(
        {
            "type": "sl_adopt",
            "checkpoint_id": checkpoint.checkpoint_id,
            "adopted": best.attempt_id,
            "original_hp": original.hp,
            "best_hp": best.hp,
            "best_outcome": best.outcome,
            "trace": str(adopted.trace_path),
        }
    )
    return adopted


def _run_replay_attempt(
    *,
    checkpoint: _ActiveCheckpoint,
    attempt_index: int,
    agent_factory: Callable[[], Agent],
    character: str,
    seed: str,
    ascension: int,
    lang: str,
    allow_repeat_views: bool,
    allow_custom_sequences: bool | None,
    custom_sequence_exposure: CustomSequenceExposure,
    combat_card_sequences_enabled: bool,
    precomputed_combat_card_sequences_enabled: bool,
    log_level: LogLevel,
    route_plan_enabled: bool,
    route_plan_enforcement: RoutePlanEnforcement,
    max_steps: int,
) -> _ReplayRun:
    attempt_id = f"retry_{attempt_index:02d}"
    trace_path = checkpoint.directory / f"{attempt_id}.jsonl"
    retry_logger = JsonlLogger(trace_path)
    retry_proc = Sts2Process()
    retry_agent = agent_factory()
    retry_result = RunResult(seed=seed, character=character, ascension=ascension)
    started = time.time()
    state: dict[str, Any] | None = None
    replay_tape: list[dict[str, Any]] = []
    entry_command_count = len((checkpoint.snapshot.get("load") or {}).get("entry_commands") or [])
    route_plan_tracker = RoutePlanTracker() if route_plan_enabled else None
    try:
        _reset_agent_episode(retry_agent)
        replay = replay_snapshot_prefix(
            checkpoint.snapshot,
            process=retry_proc,
            allow_repeat_views=allow_repeat_views,
            allow_card_sequences=combat_card_sequences_enabled and _allow_card_sequences_for_agent(retry_agent),
            allow_precomputed_card_sequences=(
                precomputed_combat_card_sequences_enabled and _allow_card_sequences_for_agent(retry_agent)
            ),
            allow_custom_sequences=(
                _allow_custom_sequences_for_agent(retry_agent)
                if allow_custom_sequences is None
                else allow_custom_sequences
            ),
            custom_sequence_exposure=custom_sequence_exposure,
        )
        state = replay.state
        replay_tape = replay.action_tape
        retry_logger.write(
            {
                "type": "sl_retry_entry",
                "checkpoint_id": checkpoint.checkpoint_id,
                "attempt": attempt_id,
                "entry_command_count": entry_command_count,
                "state": {"decision": state.get("decision"), "hp": _hp(state)},
                "enemy_snapshot": _enemy_snapshot_for_trace(state),
            }
        )
        for step in range(max_steps):
            decision = state.get("decision")
            if decision == "game_over" or not _is_combat_state(state):
                _fill_result(retry_result, state)
                return _replay_run(
                    checkpoint=checkpoint,
                    attempt_id=attempt_id,
                    result=retry_result,
                    state=state,
                    process=retry_proc,
                    replay_tape=replay_tape,
                    entry_command_count=entry_command_count,
                    trace_path=trace_path,
                )
            if route_plan_tracker is not None:
                state = _attach_route_plan_context(state, route_plan_tracker)
            legal_actions = build_legal_actions(
                state,
                allow_repeat_views=allow_repeat_views,
                allow_card_sequences=combat_card_sequences_enabled and _allow_card_sequences_for_agent(retry_agent),
                allow_precomputed_card_sequences=(
                    precomputed_combat_card_sequences_enabled and _allow_card_sequences_for_agent(retry_agent)
                ),
                allow_custom_sequences=(
                    _allow_custom_sequences_for_agent(retry_agent)
                    if allow_custom_sequences is None
                    else allow_custom_sequences
                ),
                custom_sequence_exposure=custom_sequence_exposure,
            )
            evidence_bundle = build_evidence_bundle(
                state,
                legal_actions,
                store=_agent_run_context_store(retry_agent),
            )
            decision_entry = _decision_log_entry(
                log_level=log_level,
                step=step,
                state=state,
                legal_actions=legal_actions,
                prompt=None,
                evidence_bundle=evidence_bundle,
            )
            if decision_entry:
                retry_logger.write(decision_entry)
            action, meta, validation_error = _choose_action_with_runner_validation(
                agent=retry_agent,
                state=state,
                legal_actions=legal_actions,
                prompt=None,
                result=retry_result,
                logger=retry_logger,
                log_level=log_level,
                step=step,
                decision=decision,
                print_model_output=False,
                action_tape=replay_tape,
                character=character,
                seed=seed,
                ascension=ascension,
                lang=lang,
                sequence_probe_factory=_save_backed_sequence_probe_factory(checkpoint.snapshot),
                route_plan_tracker=route_plan_tracker,
                route_plan_enabled=route_plan_enabled,
                route_plan_enforcement=route_plan_enforcement,
                evidence_bundle=evidence_bundle,
            )
            if validation_error:
                retry_result.invalid_states += 1
                retry_result.error = validation_error
                _fill_result(retry_result, state)
                return _replay_run(
                    checkpoint=checkpoint,
                    attempt_id=attempt_id,
                    result=retry_result,
                    state=state,
                    process=retry_proc,
                    replay_tape=replay_tape,
                    entry_command_count=entry_command_count,
                    trace_path=trace_path,
                )
            retry_result.steps = step + 1
            state = _execute_action(
                proc=retry_proc,
                state=state,
                action=action,
                action_tape=replay_tape,
                agent=retry_agent,
                meta=meta,
                result=retry_result,
            )
        retry_result.truncated = True
        retry_result.extra["truncation_reason"] = f"max_steps exceeded ({max_steps})"
        if isinstance(state, dict):
            _fill_result(retry_result, state)
        return _replay_run(
            checkpoint=checkpoint,
            attempt_id=attempt_id,
            result=retry_result,
            state=state or {},
            process=retry_proc,
            replay_tape=replay_tape,
            entry_command_count=entry_command_count,
            trace_path=trace_path,
        )
    except Exception as exc:  # noqa: BLE001 - retry failures are compared as failed attempts.
        retry_result.error = str(exc)
        if isinstance(state, dict):
            _fill_result(retry_result, state)
        return _replay_run(
            checkpoint=checkpoint,
            attempt_id=attempt_id,
            result=retry_result,
            state=state or {},
            process=retry_proc,
            replay_tape=replay_tape,
            entry_command_count=entry_command_count,
            trace_path=trace_path,
        )
    finally:
        retry_result.wall_time_sec = time.time() - started
        retry_logger.write({"type": "sl_retry_result", "result": retry_result.to_dict()})
        retry_logger.close()


def _replay_run(
    *,
    checkpoint: _ActiveCheckpoint,
    attempt_id: str,
    result: RunResult,
    state: dict[str, Any],
    process: Sts2Process,
    replay_tape: list[dict[str, Any]],
    entry_command_count: int,
    trace_path: Path,
) -> _ReplayRun:
    return _ReplayRun(
        attempt=ReplayAttempt(
            attempt_id=attempt_id,
            outcome=_combat_exit_outcome(state),
            hp=_hp(state),
            steps=result.steps,
            commands=_copy_json(checkpoint.run_tape_at_entry),
        ),
        state=state,
        process=process,
        replay_tape=_copy_json(replay_tape),
        entry_command_count=entry_command_count,
        result=result,
        trace_path=trace_path,
    )


def _attempt_score(attempt: ReplayAttempt) -> tuple[int, int, int, int]:
    survived = 0 if attempt.outcome == "game_over" else 1
    completed = 1 if attempt.outcome == "combat_complete" else 0
    hp = attempt.hp if isinstance(attempt.hp, int) else -1
    return (survived, completed, hp, -attempt.steps)


def _entry_commands_since_room_entry(action_tape: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index in range(len(action_tape) - 1, -1, -1):
        command = action_tape[index]
        if command.get("cmd") == "action" and command.get("action") == "select_map_node":
            return _copy_json(action_tape[index:])
    return _copy_json(action_tape)


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _is_combat_state(state: dict[str, Any]) -> bool:
    decision = state.get("decision")
    if decision == "card_select":
        return isinstance(state.get("combat"), dict)
    return decision in COMBAT_DECISIONS


def _hp(state: dict[str, Any]) -> int | None:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    value = player.get("hp", state.get("hp"))
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _combat_exit_outcome(state: dict[str, Any]) -> str:
    if state.get("decision") == "game_over":
        return "game_over" if state.get("victory") is not True else "victory"
    if not _is_combat_state(state):
        return "combat_complete"
    return "active"


def _safe_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe.strip("_") or "value"


def _api_key_default() -> str:
    return env_str("STS2_BENCH_API_KEY") or env_str("DEEPSEEK_API_KEY") or env_str("OPENAI_API_KEY") or "local"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one save/load-assisted full STS2 benchmark game.")
    parser.add_argument("--agent", default=env_str("STS2_BENCH_AGENT", "llm"), choices=["random", "llm", "coordinator"])
    parser.add_argument("--base-url", default=env_str("STS2_BENCH_BASE_URL"))
    parser.add_argument("--model", default=env_str("STS2_BENCH_MODEL"))
    parser.add_argument("--api-key", default=_api_key_default())
    parser.add_argument("--llm-timeout", type=float, default=env_float("STS2_BENCH_LLM_TIMEOUT", 120.0))
    parser.add_argument("--llm-max-retries", type=int, default=env_int("STS2_BENCH_LLM_MAX_RETRIES", 1))
    parser.add_argument("--llm-fallback-mode", default=env_str("STS2_BENCH_LLM_FALLBACK_MODE", "action"), choices=["error", "action"])
    parser.add_argument("--llm-temperature", type=float, default=env_float("STS2_BENCH_LLM_TEMPERATURE", 0.0))
    parser.add_argument("--prompt-style", default=env_str("STS2_BENCH_PROMPT_STYLE", "default"), choices=["default", "analysis"])
    parser.add_argument("--context-mode", default=env_str("STS2_BENCH_CONTEXT_MODE", "combat_mdp_state"), choices=["single_turn", "turn_chat", "combat_mdp_state"])
    parser.add_argument("--run-summary", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_RUN_SUMMARY_ENABLED", False))
    parser.add_argument("--memory-enabled", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_MEMORY_ENABLED", False))
    parser.add_argument("--route-plan", action=argparse.BooleanOptionalAction, dest="route_plan_enabled", default=env_bool("STS2_BENCH_ROUTE_PLAN_ENABLED", True))
    parser.add_argument("--route-plan-enforcement", default=env_str("STS2_BENCH_ROUTE_PLAN_ENFORCEMENT", "hard"), choices=["hard", "soft"])
    parser.add_argument("--route-map-prompt-format", default=env_str("STS2_BENCH_ROUTE_MAP_PROMPT_FORMAT", "canonical"), choices=["route_graph", "canonical"])
    parser.add_argument("--character", default=env_str("STS2_BENCH_CHARACTER", "Ironclad"))
    parser.add_argument("--ascension", type=int, default=env_int("STS2_BENCH_ASCENSION", 0))
    parser.add_argument("--lang", default=env_str("STS2_BENCH_LANG", "en"), choices=["en", "zh"])
    parser.add_argument("--seeds", default=env_str("STS2_BENCH_SEEDS"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=env_int("STS2_BENCH_MAX_STEPS", 2000))
    parser.add_argument("--max-game-actions", type=int, default=env_int("STS2_BENCH_MAX_GAME_ACTIONS", 0))
    parser.add_argument("--out", required=True)
    parser.add_argument("--sl-dir", required=True)
    parser.add_argument("--sl-hp-loss-threshold", type=int, default=20)
    parser.add_argument("--sl-retries", type=int, default=2)
    parser.add_argument("--sl-max-combat-steps", type=int, default=120)
    parser.add_argument("--log-level", default=env_str("STS2_BENCH_LOG_LEVEL", "decisions"), choices=["metrics", "decisions", "full"])
    parser.add_argument("--include-full-map", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_INCLUDE_FULL_MAP", False))
    parser.add_argument("--allow-repeat-views", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_ALLOW_REPEAT_VIEWS", False))
    parser.add_argument("--allow-custom-sequences", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_ALLOW_CUSTOM_SEQUENCES", True))
    parser.add_argument("--custom-sequence-exposure", default=env_str("STS2_BENCH_CUSTOM_SEQUENCE_EXPOSURE", "eligible"), choices=["safe", "eligible"])
    parser.add_argument("--combat-card-sequences", action=argparse.BooleanOptionalAction, dest="combat_card_sequences_enabled", default=env_bool("STS2_BENCH_COMBAT_CARD_SEQUENCES", True))
    parser.add_argument("--precomputed-combat-card-sequences", action=argparse.BooleanOptionalAction, dest="precomputed_combat_card_sequences_enabled", default=env_bool("STS2_BENCH_PRECOMPUTED_COMBAT_CARD_SEQUENCES", False))
    parser.add_argument("--self-consistency", action=argparse.BooleanOptionalAction, dest="self_consistency_enabled", default=env_bool("STS2_BENCH_SELF_CONSISTENCY", False))
    parser.add_argument("--reflection-verifier", action=argparse.BooleanOptionalAction, dest="reflection_enabled", default=env_bool("STS2_BENCH_REFLECTION_VERIFIER", False))
    parser.add_argument("--reward-posterior-evaluator", action=argparse.BooleanOptionalAction, default=env_bool("STS2_BENCH_REWARD_POSTERIOR_EVALUATOR", False))
    parser.add_argument("--combat-reasoning-pool", action=argparse.BooleanOptionalAction, dest="combat_reasoning_pool_enabled", default=env_bool("STS2_BENCH_COMBAT_REASONING_POOL", False))
    parser.add_argument("--combat-turn-board", action=argparse.BooleanOptionalAction, dest="combat_turn_board_enabled", default=env_bool("STS2_BENCH_COMBAT_TURN_BOARD", False))
    parser.add_argument("--resume-trace", help="Replay executed actions from a previous trace before continuing.")
    parser.add_argument("--resume-before-step", type=int, help="Replay actions with step less than this value.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_benchmark_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    seeds = load_seeds(args.seeds, args.count)
    if len(seeds) != 1:
        parser.error("sl_full_run currently expects exactly one seed")

    def make_agent() -> Agent:
        return agent_from_args(
            args.agent,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            llm_timeout=args.llm_timeout,
            llm_max_retries=args.llm_max_retries,
            llm_fallback_mode=args.llm_fallback_mode,
            llm_temperature=args.llm_temperature,
            include_json_state=False,
            prompt_style=args.prompt_style,
            memory_enabled=args.memory_enabled,
            run_summary_enabled=args.run_summary,
            conversation_mode=args.context_mode,
            route_plan_enabled=args.route_plan_enabled,
            route_map_prompt_format=args.route_map_prompt_format,
            route_plan_enforcement=args.route_plan_enforcement,
            reward_posterior_evaluator_enabled=args.reward_posterior_evaluator,
            self_consistency_enabled=args.self_consistency_enabled,
            reflection_enabled=args.reflection_enabled,
            combat_card_sequences_enabled=args.combat_card_sequences_enabled,
            combat_reasoning_pool_enabled=args.combat_reasoning_pool_enabled,
            combat_turn_board_enabled=args.combat_turn_board_enabled,
        )

    logger = JsonlLogger(args.out)
    try:
        run_config = {
            "type": "sl_run_config",
            "config": {
                "seed": seeds[0],
                "agent": args.agent,
                "model": args.model,
                "context_mode": args.context_mode,
                "prompt_style": args.prompt_style,
                "sl_hp_loss_threshold": args.sl_hp_loss_threshold,
                "sl_retries": args.sl_retries,
                "combat_card_sequences": args.combat_card_sequences_enabled,
                "custom_sequence_exposure": args.custom_sequence_exposure,
                "argv": list(argv if argv is not None else sys.argv[1:]),
            },
        }
        print(json.dumps(run_config, ensure_ascii=False), flush=True)
        logger.write(run_config)
        result = run_sl_one(
            agent_factory=make_agent,
            seed=seeds[0],
            character=args.character,
            ascension=args.ascension,
            lang=args.lang,
            max_steps=args.max_steps,
            max_game_actions=args.max_game_actions or None,
            logger=logger,
            sl_dir=args.sl_dir,
            sl_hp_loss_threshold=args.sl_hp_loss_threshold,
            sl_retries=args.sl_retries,
            sl_max_combat_steps=args.sl_max_combat_steps,
            log_level=args.log_level,
            include_full_map=args.include_full_map,
            allow_repeat_views=args.allow_repeat_views,
            allow_custom_sequences=args.allow_custom_sequences,
            custom_sequence_exposure=args.custom_sequence_exposure,
            combat_card_sequences_enabled=args.combat_card_sequences_enabled,
            precomputed_combat_card_sequences_enabled=args.precomputed_combat_card_sequences_enabled,
            route_plan_enabled=args.route_plan_enabled,
            route_plan_enforcement=args.route_plan_enforcement,
            resume_trace=args.resume_trace,
            resume_before_step=args.resume_before_step,
        )
        print(json.dumps({"seed": seeds[0], "result": result.to_dict()}, ensure_ascii=False), flush=True)
        print(json.dumps({"summary": summarize([result])}, ensure_ascii=False, indent=2), flush=True)
        return 0 if result.error is None else 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
