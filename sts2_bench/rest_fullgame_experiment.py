"""Run full-game RestSite route-injection comparisons.

This module is intentionally thin: it keeps the normal benchmark runner,
validators, action surface, and OpenAI-compatible agent, and only appends
map-select prompt instructions for the two route-planning conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .actions import LegalAction
from .agents import OpenAICompatAgent
from .combat_record import CombatRecorder
from .run_benchmark import env_bool, env_float, env_int, env_str, load_benchmark_env
from .runner import JsonlLogger, RunResult, run_one
from .seeds import DEFAULT_SEEDS_FILE, read_seed_file
from .trace_analysis import analyze_trace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THIRD_SEED = "ironclad_a0_20260627"

COMMON_ROUTE_COT_INJECTION = """

Route-planning reasoning instruction for map_select only:
- If there is no active_route_plan, create a route plan and explicitly compare the visible connected route suffixes before choosing.
- If you are changing away from active_route_plan.expected_next, update the route plan and explicitly compare the visible connected route suffixes before choosing.
- For route creation or route change, add a top-level "route_rest_comparison" array. For each non-view legal map action, include action_id, selected node, best visible route, RestSite count on that route, and the main risk tradeoff.
- If following active_route_plan.expected_next, do not replan and do not add route_rest_comparison. Select expected_next, omit route_plan_event, and use only a short reason such as "following active_route_plan.expected_next".
- If you create or update route_plan_event, its new_plan must match the route you compared and selected.
- The route comparison is required for route-planning accuracy on creation/update, but the final choice may still use normal game strategy unless another instruction says RestSite count is primary.
- Do not choose a view action only to count RestSites when the route map JSON is already present.
""".rstrip()

MAX_REST_STYLE_INJECTION = """

RestSite-priority route style for this run:
- When creating or updating a route plan, choose a legal map action whose complete visible route can contain the maximum possible number of RestSite nodes before the visible Boss or terminal node.
- When following an existing active_route_plan.expected_next, continue following it unless the current state gives a concrete reason to replan.
- RestSite count is the primary route objective for this experiment because future monster and elite attrition can kill the run.
- If several routes tie on RestSite count, break ties by survival, avoiding elites when weak, then useful shops/events.
- In route_plan_event.reason, state the chosen route's RestSite count and why no legal visible alternative gives more RestSites.
""".rstrip()


@dataclass(frozen=True)
class Condition:
    name: str
    injection: str


CONDITIONS = (
    Condition("max_rest_cot", COMMON_ROUTE_COT_INJECTION + "\n" + MAX_REST_STYLE_INJECTION),
    Condition("control_cot", COMMON_ROUTE_COT_INJECTION),
)


def _selected_conditions(names: str | None) -> tuple[Condition, ...]:
    if not names:
        return CONDITIONS
    requested = [name.strip() for name in names.split(",") if name.strip()]
    known = {condition.name: condition for condition in CONDITIONS}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(f"Unknown condition(s): {', '.join(unknown)}")
    return tuple(known[name] for name in requested)


class MapPromptInjectionAgent:
    """Delegate agent that appends route instructions only at map_select."""

    def __init__(self, base: OpenAICompatAgent, *, condition: str, injection: str) -> None:
        self.base = base
        self.condition = condition
        self.injection = injection

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        prompt = self.base.build_prompt(state, legal_actions)
        if state.get("decision") == "map_select":
            return prompt + self.injection
        return prompt

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt: str | None = None,
    ) -> tuple[LegalAction, dict[str, Any]]:
        effective_prompt = prompt
        if effective_prompt is None:
            effective_prompt = self.build_prompt(state, legal_actions)
        action, meta = self.base.choose(state, legal_actions, prompt=effective_prompt)
        if state.get("decision") == "map_select":
            meta = dict(meta)
            meta["route_prompt_condition"] = self.condition
            meta["route_common_cot_injected"] = True
            meta["route_max_rest_injected"] = self.condition == "max_rest_cot"
        return action, meta

    def reset_episode(self) -> None:
        reset = getattr(self.base, "reset_episode", None)
        if callable(reset):
            reset()

    def receive_external_feedback(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        feedback: str,
        legal_actions: list[LegalAction],
    ) -> None:
        receive = getattr(self.base, "receive_external_feedback", None)
        if callable(receive):
            receive(state, action, meta, feedback, legal_actions)

    def record_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        record = getattr(self.base, "record_transition", None)
        if callable(record):
            record(state, action, meta, next_state)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_fixed_seeds(path: str | None, count: int, third_seed: str) -> list[str]:
    if path:
        seeds = read_seed_file(path)
    else:
        seeds = read_seed_file(DEFAULT_SEEDS_FILE)
        if len(seeds) < count and third_seed not in seeds:
            seeds = [*seeds, third_seed]
    if len(seeds) < count:
        raise ValueError(f"Need {count} fixed seeds, got {len(seeds)}")
    return seeds[:count]


def _api_key(api_key_env: str | None = None) -> str:
    if api_key_env:
        value = os.environ.get(api_key_env)
        if value:
            return value
        raise ValueError(f"{api_key_env} is not set after loading benchmark env")
    return env_str("DEEPSEEK_API_KEY", "local", fallbacks=("OPENAI_API_KEY",)) or "local"


def _new_agent(args: argparse.Namespace) -> MapPromptInjectionAgent:
    base_url = env_str("STS2_BENCH_BASE_URL")
    model = env_str("STS2_BENCH_MODEL")
    if not base_url:
        raise ValueError("STS2_BENCH_BASE_URL is required for the LLM experiment")
    if not model:
        raise ValueError("STS2_BENCH_MODEL is required for the LLM experiment")

    condition = next(item for item in CONDITIONS if item.name == args.condition)
    base = OpenAICompatAgent(
        base_url=base_url,
        model=model,
        api_key=_api_key(getattr(args, "api_key_env", "")),
        temperature=env_float("STS2_BENCH_LLM_TEMPERATURE", 0.0),
        timeout=args.llm_timeout,
        include_json_state=env_bool("STS2_BENCH_INCLUDE_JSON_STATE", False),
        max_retries=args.llm_max_retries,
        prompt_style="analysis",
        memory_enabled=env_bool("STS2_BENCH_MEMORY_ENABLED", False),
        memory_window=env_int("STS2_BENCH_MEMORY_WINDOW", 8),
        memory_mode=env_str("STS2_BENCH_MEMORY_MODE", "action_reason") or "action_reason",
        run_summary_enabled=env_bool("STS2_BENCH_RUN_SUMMARY_ENABLED", False),
        conversation_mode=env_str(
            "STS2_BENCH_CONTEXT_MODE",
            "combat_mdp_state",
            fallbacks=("STS2_BENCH_CONVERSATION_MODE",),
        )
        or "combat_mdp_state",
        turn_chat_window=env_int("STS2_BENCH_TURN_CHAT_WINDOW", 10),
        turn_chat_update_mode=env_str("STS2_BENCH_TURN_CHAT_UPDATE_MODE", "delta") or "delta",
        turn_chat_assistant_history=env_str("STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY", "compact") or "compact",
        turn_chat_plan_enabled=env_bool(
            "STS2_BENCH_TURN_CHAT_PLAN_ENABLED",
            env_bool("STS2_BENCH_TURN_CHECK_ENABLED", False),
        ),
        turn_check_enabled=env_bool("STS2_BENCH_TURN_CHECK_ENABLED", False),
        route_plan_enabled=True,
        route_map_prompt_format=args.route_map_prompt_format,
        context_patch_fields_enabled=True,
        fallback_mode=env_str("STS2_BENCH_LLM_FALLBACK_MODE", "error") or "error",
        self_consistency_enabled=env_bool("STS2_BENCH_SELF_CONSISTENCY", False),
        self_consistency_samples=env_int("STS2_BENCH_SELF_CONSISTENCY_SAMPLES", 3),
        self_consistency_decisions=env_str("STS2_BENCH_SELF_CONSISTENCY_DECISIONS", "map_select,card_reward")
        or "map_select,card_reward",
        self_consistency_mode=env_str("STS2_BENCH_SELF_CONSISTENCY_MODE", "diagnostic") or "diagnostic",
        reflection_enabled=env_bool("STS2_BENCH_REFLECTION_VERIFIER", False),
        reflection_decisions=env_str("STS2_BENCH_REFLECTION_DECISIONS", "card_reward") or "card_reward",
        reflection_mode=env_str("STS2_BENCH_REFLECTION_MODE", "diagnostic") or "diagnostic",
        reflection_max_options=env_int("STS2_BENCH_REFLECTION_MAX_OPTIONS", 8),
        action_prompt_order=env_str("STS2_BENCH_ACTION_PROMPT_ORDER", "default") or "default",
        action_prompt_order_seed=env_int("STS2_BENCH_ACTION_PROMPT_ORDER_SEED", 0),
        combat_card_sequences_enabled=env_bool("STS2_BENCH_COMBAT_CARD_SEQUENCES", True),
        combat_reasoning_pool_enabled=env_bool("STS2_BENCH_COMBAT_REASONING_POOL", False),
        combat_turn_board_enabled=env_bool("STS2_BENCH_COMBAT_TURN_BOARD", False),
        json_response_format=args.json_response_format,
    )
    return MapPromptInjectionAgent(base, condition=condition.name, injection=condition.injection)


def _first_act_passed(result: dict[str, Any]) -> bool:
    act = result.get("act")
    if result.get("victory"):
        return True
    return isinstance(act, int) and act >= 2


def _count_planned_rest_sites(route_plan_events: list[dict[str, Any]]) -> list[int]:
    counts: list[int] = []
    for event in route_plan_events:
        plan = event.get("new_plan")
        if not isinstance(plan, list):
            continue
        counts.append(sum(1 for node in plan if isinstance(node, dict) and node.get("type") == "RestSite"))
    return counts


def _selected_rest_node_count(map_choices: list[dict[str, Any]]) -> int:
    count = 0
    for choice in map_choices:
        selected = choice.get("selected") or {}
        node = selected.get("node") if isinstance(selected, dict) else None
        if isinstance(node, dict) and node.get("type") == "RestSite":
            count += 1
    return count


def _trace_metrics(trace_path: Path, analysis_dir: Path) -> dict[str, Any]:
    report = analyze_trace(trace_path, analysis_dir)
    map_choices = report.get("map_choices") or []
    rest_sites = report.get("rest_sites") or []
    route_errors = report.get("route_plan_validation_errors") or []
    route_events = report.get("route_plan_events") or []
    rooms = report.get("rooms") or []
    terminal_deaths = [room for room in rooms if room.get("terminal_death")]
    return {
        "map_action_count": len(map_choices),
        "selected_rest_node_count": _selected_rest_node_count(map_choices),
        "rest_site_action_count": len(rest_sites),
        "rest_site_choices": [item.get("choice") for item in rest_sites],
        "route_plan_event_count": len(route_events),
        "route_plan_planned_rest_counts": _count_planned_rest_sites(route_events),
        "route_plan_validation_error_count": len(route_errors),
        "large_loss_room_count": sum(1 for room in rooms if room.get("large_loss")),
        "terminal_death_room": terminal_deaths[-1] if terminal_deaths else None,
    }


def _result_row(
    *,
    condition: str,
    seed: str,
    trace_path: Path,
    result: RunResult,
    elapsed_sec: float,
    analysis_dir: Path,
) -> dict[str, Any]:
    result_dict = result.to_dict()
    metrics = _trace_metrics(trace_path, analysis_dir)
    return {
        "condition": condition,
        "seed": seed,
        "elapsed_sec": round(elapsed_sec, 3),
        "trace_path": str(trace_path),
        "analysis_dir": str(analysis_dir),
        "result": result_dict,
        "first_act_pass": _first_act_passed(result_dict),
        "trace_metrics": metrics,
    }


def _read_results(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)

    aggregate: dict[str, Any] = {}
    for condition, rows in by_condition.items():
        floors = [(row.get("result") or {}).get("floor") or 0 for row in rows]
        aggregate[condition] = {
            "runs": len(rows),
            "natural_complete_count": sum(
                1
                for row in rows
                if not (row.get("result") or {}).get("error") and not (row.get("result") or {}).get("truncated")
            ),
            "first_act_pass_count": sum(1 for row in rows if row.get("first_act_pass")),
            "victory_count": sum(1 for row in rows if (row.get("result") or {}).get("victory")),
            "avg_floor": round(sum(floors) / len(floors), 2) if floors else 0,
            "max_floor": max(floors) if floors else 0,
            "total_selected_rest_nodes": sum(
                ((row.get("trace_metrics") or {}).get("selected_rest_node_count") or 0) for row in rows
            ),
            "total_rest_site_actions": sum(
                ((row.get("trace_metrics") or {}).get("rest_site_action_count") or 0) for row in rows
            ),
            "total_route_plan_errors": sum(
                ((row.get("trace_metrics") or {}).get("route_plan_validation_error_count") or 0) for row in rows
            ),
            "total_fallbacks": sum(((row.get("result") or {}).get("model_fallback_count") or 0) for row in rows),
            "error_count": sum(1 for row in rows if (row.get("result") or {}).get("error")),
            "truncated_count": sum(1 for row in rows if (row.get("result") or {}).get("truncated")),
        }
    return aggregate


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["records"]
    lines = [
        "# RestSite Full-Game CoT Route Injection Experiment",
        "",
        f"- output: `{payload['out_dir']}`",
        f"- model: `{payload['model']}`",
        f"- base_url: `{payload['base_url']}`",
        f"- seeds: {', '.join(payload['seeds'])}",
        f"- route map prompt format: `{payload['route_map_prompt_format']}`",
        f"- prompt style: `analysis`",
        f"- context mode: `{payload['context_mode']}`",
        f"- route plan enforcement: `{payload['route_plan_enforcement']}`",
        f"- LLM timeout: `{payload['llm_timeout_sec']}` sec",
        f"- LLM max retries: `{payload['llm_max_retries']}`",
        f"- JSON response format: `{payload['json_response_format']}`",
        f"- decision watchdog: `{payload['decision_watchdog_sec']}` sec without trace progress",
        f"- max game actions: none",
        f"- max steps safety cap: `{payload['max_steps']}`",
        "",
        "Enabled conditions require explicit route RestSite comparison CoT when creating or updating a route plan. When following `active_route_plan.expected_next`, conditions keep the existing lightweight follow behavior. Only `max_rest_cot` additionally makes maximum visible RestSite count the route objective during creation/update.",
        "",
        "## Runs",
        "",
        "| condition | seed | act | floor | decision | HP | first Act passed | complete | truncated | error | map actions | selected RestSite nodes | RestSite actions | route errors | fallbacks | elapsed sec |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        result = record.get("result") or {}
        metrics = record.get("trace_metrics") or {}
        complete = not result.get("error") and not result.get("truncated")
        hp = result.get("hp")
        max_hp = result.get("max_hp")
        hp_text = "" if hp is None else f"{hp}/{max_hp}"
        error = result.get("error") or ""
        if len(error) > 80:
            error = error[:77] + "..."
        lines.append(
            "| {condition} | {seed} | {act} | {floor} | {decision} | {hp} | {pass_act} | {complete} | {truncated} | {error} | {map_actions} | {selected_rest} | {rest_actions} | {route_errors} | {fallbacks} | {elapsed} |".format(
                condition=record.get("condition"),
                seed=record.get("seed"),
                act=result.get("act"),
                floor=result.get("floor"),
                decision=result.get("decision"),
                hp=hp_text,
                pass_act=record.get("first_act_pass"),
                complete=complete,
                truncated=result.get("truncated"),
                error=error.replace("|", "/"),
                map_actions=metrics.get("map_action_count"),
                selected_rest=metrics.get("selected_rest_node_count"),
                rest_actions=metrics.get("rest_site_action_count"),
                route_errors=metrics.get("route_plan_validation_error_count"),
                fallbacks=result.get("model_fallback_count"),
                elapsed=record.get("elapsed_sec"),
            )
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "```json",
            json.dumps(payload["aggregate"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Prompt Injections",
            "",
            "### Common Route CoT",
            "",
            "```text",
            COMMON_ROUTE_COT_INJECTION.strip(),
            "```",
            "",
            "### Max RestSite Style",
            "",
            "```text",
            MAX_REST_STYLE_INJECTION.strip(),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _payload(args: argparse.Namespace, out_dir: Path, seeds: list[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate(records)
    return {
        "created_at": args.created_at,
        "out_dir": str(out_dir),
        "model": env_str("STS2_BENCH_MODEL"),
        "base_url": env_str("STS2_BENCH_BASE_URL"),
        "seeds": seeds,
        "default_seed_file": str(DEFAULT_SEEDS_FILE),
        "third_seed_if_needed": args.third_seed,
        "route_map_prompt_format": args.route_map_prompt_format,
        "route_plan_enforcement": args.route_plan_enforcement,
        "prompt_style": "analysis",
        "context_mode": env_str("STS2_BENCH_CONTEXT_MODE", "combat_mdp_state", fallbacks=("STS2_BENCH_CONVERSATION_MODE",)),
        "llm_timeout_sec": args.llm_timeout,
        "llm_max_retries": args.llm_max_retries,
        "json_response_format": args.json_response_format,
        "decision_watchdog_sec": args.decision_watchdog_sec,
        "combat_records": args.combat_records,
        "combat_record_dir": str(_combat_record_root(args, out_dir)),
        "max_steps": args.max_steps,
        "conditions": [
            {
                "name": item.name,
                "common_route_cot": True,
                "max_rest_injection": item.name == "max_rest_cot",
                "injection": item.injection,
            }
            for item in _selected_conditions(args.conditions)
        ],
        "records": records,
        "aggregate": aggregate,
    }


def _persist(args: argparse.Namespace, out_dir: Path, seeds: list[str], records: list[dict[str, Any]]) -> None:
    payload = _payload(args, out_dir, seeds, records)
    _write_json(out_dir / "results.json", payload)
    _write_summary(out_dir / "summary.md", payload)


def _execute_single_run(
    *,
    args: argparse.Namespace,
    condition: Condition,
    seed: str,
    trace_path: Path,
    analysis_dir: Path,
) -> dict[str, Any]:
    run_args = argparse.Namespace(**vars(args))
    run_args.condition = condition.name
    agent = _new_agent(run_args)
    logger = JsonlLogger(trace_path)
    combat_record_root = _combat_record_root(args, trace_path.parent.parent)
    combat_recorder = (
        CombatRecorder(
            root=combat_record_root,
            run_id=f"{condition.name}_{seed}",
            condition=condition.name,
            seed=seed,
            character=args.character,
            ascension=args.ascension,
            lang=args.lang,
        )
        if args.combat_records
        else None
    )
    logger.write(
        {
            "type": "experiment_config",
            "condition": condition.name,
            "seed": seed,
            "common_route_cot": True,
            "max_rest_injection": condition.name == "max_rest_cot",
            "route_prompt_injection": condition.injection,
            "combat_records": bool(combat_recorder),
            "combat_record_dir": str(combat_record_root),
        }
    )
    started = time.time()
    try:
        print(
            json.dumps(
                {
                    "start": True,
                    "condition": condition.name,
                    "seed": seed,
                    "trace": str(trace_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        result = run_one(
            agent=agent,
            seed=seed,
            character=args.character,
            ascension=args.ascension,
            lang=args.lang,
            max_steps=args.max_steps,
            max_game_actions=None,
            logger=logger,
            print_prompts=False,
            print_model_output=False,
            include_full_map=True,
            allow_repeat_views=False,
            allow_custom_sequences=True,
            combat_card_sequences_enabled=env_bool("STS2_BENCH_COMBAT_CARD_SEQUENCES", True),
            log_level=args.log_level,
            route_plan_enabled=True,
            route_plan_enforcement=args.route_plan_enforcement,
            combat_recorder=combat_recorder,
        )
    finally:
        logger.close()
    elapsed = time.time() - started
    row = _result_row(
        condition=condition.name,
        seed=seed,
        trace_path=trace_path,
        result=result,
        elapsed_sec=elapsed,
        analysis_dir=analysis_dir,
    )
    print(
        json.dumps({"finish": True, "condition": condition.name, "seed": seed, "result": result.to_dict()}, ensure_ascii=False),
        flush=True,
    )
    return row


def _trace_marker(trace_path: Path) -> tuple[int, int] | None:
    try:
        stat = trace_path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _last_state_from_trace(trace_path: Path) -> dict[str, Any]:
    if not trace_path.is_file():
        return {}
    last_state: dict[str, Any] = {}
    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "decision" and isinstance(obj.get("state"), dict):
            last_state = obj["state"]
    return last_state


def _count_trace_actions(trace_path: Path) -> tuple[int, int]:
    if not trace_path.is_file():
        return 0, 0
    steps = 0
    game_actions = 0
    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "action":
            continue
        steps += 1
        action = obj.get("action") or {}
        if action.get("kind") not in {"view_deck", "view_map", "view_draw_pile", "view_discard_pile", "view_exhaust_pile"}:
            game_actions += 1
    return steps, game_actions


def _partial_result(
    *,
    args: argparse.Namespace,
    seed: str,
    trace_path: Path,
    error: str,
    elapsed_sec: float,
) -> RunResult:
    state = _last_state_from_trace(trace_path)
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    steps, game_actions = _count_trace_actions(trace_path)
    result = RunResult(seed=seed, character=args.character, ascension=args.ascension)
    result.decision = state.get("decision")
    result.act = state.get("act") or context.get("act")
    result.floor = state.get("floor") or context.get("floor")
    result.room_type = context.get("room_type") or state.get("room_type")
    result.hp = player.get("hp") if player else state.get("hp")
    result.max_hp = player.get("max_hp") if player else state.get("max_hp")
    result.gold = player.get("gold") if player else state.get("gold")
    result.deck_size = state.get("deck_size")
    result.steps = steps
    result.game_action_count = game_actions
    result.truncated = True
    result.wall_time_sec = elapsed_sec
    result.error = error
    result.extra["watchdog_timeout"] = True
    result.extra["watchdog_timeout_sec"] = args.decision_watchdog_sec
    return result


def _append_result_to_trace(trace_path: Path, result: RunResult) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "result", "result": result.to_dict()}, ensure_ascii=False) + "\n")


def _timeout_row(
    *,
    args: argparse.Namespace,
    condition: Condition,
    seed: str,
    trace_path: Path,
    analysis_dir: Path,
    elapsed_sec: float,
    reason: str,
) -> dict[str, Any]:
    result = _partial_result(args=args, seed=seed, trace_path=trace_path, error=reason, elapsed_sec=elapsed_sec)
    _append_result_to_trace(trace_path, result)
    return _result_row(
        condition=condition.name,
        seed=seed,
        trace_path=trace_path,
        result=result,
        elapsed_sec=elapsed_sec,
        analysis_dir=analysis_dir,
    )


def _worker_command(
    *,
    args: argparse.Namespace,
    condition: Condition,
    seed: str,
    out_dir: Path,
    trace_path: Path,
    analysis_dir: Path,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "sts2_bench.rest_fullgame_experiment",
        "--worker",
        "--out-dir",
        str(out_dir),
        "--worker-condition",
        condition.name,
        "--worker-seed",
        seed,
        "--worker-trace",
        str(trace_path),
        "--worker-analysis-dir",
        str(analysis_dir),
        "--worker-result",
        str(result_path),
        "--character",
        args.character,
        "--ascension",
        str(args.ascension),
        "--lang",
        args.lang,
        "--max-steps",
        str(args.max_steps),
        "--llm-timeout",
        str(args.llm_timeout),
        "--llm-max-retries",
        str(args.llm_max_retries),
        "--api-key-env",
        args.api_key_env,
        "--json-response-format" if args.json_response_format else "--no-json-response-format",
        "--route-plan-enforcement",
        args.route_plan_enforcement,
        "--route-map-prompt-format",
        args.route_map_prompt_format,
        "--log-level",
        args.log_level,
        "--combat-record-dir",
        str(_combat_record_root(args, out_dir)),
        "--combat-records" if args.combat_records else "--no-combat-records",
    ]


def _combat_record_root(args: argparse.Namespace, out_dir: Path) -> Path:
    return Path(args.combat_record_dir) if args.combat_record_dir else out_dir / "combat_records"


def _run_one_supervised(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    condition: Condition,
    seed: str,
) -> dict[str, Any]:
    trace_path = out_dir / "traces" / f"{condition.name}_{seed}.jsonl"
    analysis_dir = out_dir / "analysis" / f"{condition.name}_{seed}"
    result_path = out_dir / "worker_results" / f"{condition.name}_{seed}.json"
    worker_log_dir = out_dir / "worker_logs"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = worker_log_dir / f"{condition.name}_{seed}.out.log"
    stderr_path = worker_log_dir / f"{condition.name}_{seed}.err.log"
    for path in (result_path, stdout_path, stderr_path):
        path.unlink(missing_ok=True)

    cmd = _worker_command(
        args=args,
        condition=condition,
        seed=seed,
        out_dir=out_dir,
        trace_path=trace_path,
        analysis_dir=analysis_dir,
        result_path=result_path,
    )
    print(json.dumps({"start": True, "condition": condition.name, "seed": seed, "trace": str(trace_path)}, ensure_ascii=False), flush=True)
    started = time.time()
    last_marker = _trace_marker(trace_path)
    last_progress = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=stdout, stderr=stderr)
        while proc.poll() is None:
            time.sleep(2.0)
            marker = _trace_marker(trace_path)
            if marker != last_marker:
                last_marker = marker
                last_progress = time.time()
            if time.time() - last_progress > args.decision_watchdog_sec:
                proc.kill()
                proc.wait(timeout=10)
                elapsed = time.time() - started
                reason = (
                    f"decision_watchdog_timeout: no trace progress for "
                    f"{args.decision_watchdog_sec:.1f}s"
                )
                row = _timeout_row(
                    args=args,
                    condition=condition,
                    seed=seed,
                    trace_path=trace_path,
                    analysis_dir=analysis_dir,
                    elapsed_sec=elapsed,
                    reason=reason,
                )
                print(
                    json.dumps(
                        {
                            "finish": True,
                            "condition": condition.name,
                            "seed": seed,
                            "timeout": True,
                            "result": row["result"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return row

    elapsed = time.time() - started
    if result_path.is_file():
        payload = _read_results(result_path) or {}
        row = payload.get("row")
        if isinstance(row, dict):
            print(
                json.dumps(
                    {
                        "finish": True,
                        "condition": condition.name,
                        "seed": seed,
                        "result": row.get("result"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return row

    stderr_tail = ""
    if stderr_path.is_file():
        stderr_tail = "\n".join(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-5:])
    reason = f"worker_failed_without_result: exit_code={proc.returncode}"
    if stderr_tail:
        reason += f"; stderr_tail={stderr_tail[:500]}"
    row = _timeout_row(
        args=args,
        condition=condition,
        seed=seed,
        trace_path=trace_path,
        analysis_dir=analysis_dir,
        elapsed_sec=elapsed,
        reason=reason,
    )
    print(
        json.dumps(
            {"finish": True, "condition": condition.name, "seed": seed, "worker_failed": True, "result": row["result"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return row


def run_worker(args: argparse.Namespace) -> int:
    load_benchmark_env()
    if args.llm_max_retries is None:
        args.llm_max_retries = env_int("STS2_BENCH_LLM_MAX_RETRIES", 2)
    condition = next(item for item in CONDITIONS if item.name == args.worker_condition)
    row = _execute_single_run(
        args=args,
        condition=condition,
        seed=args.worker_seed,
        trace_path=Path(args.worker_trace),
        analysis_dir=Path(args.worker_analysis_dir),
    )
    _write_json(Path(args.worker_result), {"row": row})
    return 0


def run(args: argparse.Namespace) -> int:
    load_benchmark_env()
    if args.llm_max_retries is None:
        args.llm_max_retries = env_int("STS2_BENCH_LLM_MAX_RETRIES", 2)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"rest_fullgame_cot_fixed_seeds_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _load_fixed_seeds(args.seeds, args.count, args.third_seed)
    conditions = _selected_conditions(args.conditions)
    (out_dir / "fixed_seeds.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    results_path = out_dir / "results.json"
    previous = _read_results(results_path) if args.resume else None
    records: list[dict[str, Any]] = list((previous or {}).get("records") or [])
    done = {(record.get("condition"), record.get("seed")) for record in records}

    if args.dry_run:
        payload = _payload(args, out_dir, seeds, records)
        _write_json(results_path, payload)
        _write_summary(out_dir / "summary.md", payload)
        print(json.dumps({"dry_run": True, "out_dir": str(out_dir), "seeds": seeds}, ensure_ascii=False), flush=True)
        return 0

    for condition in conditions:
        for seed in seeds:
            if args.resume and not args.force and (condition.name, seed) in done:
                print(json.dumps({"skip": True, "condition": condition.name, "seed": seed}, ensure_ascii=False), flush=True)
                continue

            row = _run_one_supervised(args=args, out_dir=out_dir, condition=condition, seed=seed)
            records.append(row)
            done.add((condition.name, seed))
            _persist(args, out_dir, seeds, records)

    _persist(args, out_dir, seeds, records)
    failures = [row for row in records if (row.get("result") or {}).get("error")]
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RestSite route CoT full-game comparison.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--conditions", default="", help="Comma-separated subset: max_rest_cot,control_cot")
    parser.add_argument("--third-seed", default=DEFAULT_THIRD_SEED)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-retries", type=int, default=None)
    parser.add_argument("--api-key-env", default="")
    parser.add_argument(
        "--json-response-format",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_JSON_RESPONSE_FORMAT", False),
    )
    parser.add_argument("--decision-watchdog-sec", type=float, default=60.0)
    parser.add_argument("--route-plan-enforcement", choices=["hard", "soft"], default="hard")
    parser.add_argument("--route-map-prompt-format", choices=["canonical", "route_graph"], default="canonical")
    parser.add_argument("--log-level", choices=["metrics", "decisions", "full"], default="decisions")
    parser.add_argument("--combat-record-dir", default="")
    parser.add_argument("--combat-records", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-condition", choices=[item.name for item in CONDITIONS], default="")
    parser.add_argument("--worker-seed", default="")
    parser.add_argument("--worker-trace", default="")
    parser.add_argument("--worker-analysis-dir", default="")
    parser.add_argument("--worker-result", default="")
    parser.set_defaults(created_at=datetime.now().isoformat(timespec="seconds"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    if args.worker:
        return run_worker(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
