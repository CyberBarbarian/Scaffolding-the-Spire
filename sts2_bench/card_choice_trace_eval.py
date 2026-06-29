"""Replay card-reward decisions extracted from benchmark traces."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .actions import LegalAction, build_legal_actions
from .agents import Agent
from .card_choice_eval import CardChoiceOracle, CardChoiceScenario, EvalLogLevel
from .context import compact_state
from .evidence import build_evidence_bundle
from .runner import agent_from_args


@dataclass(frozen=True)
class TraceExtractionResult:
    scenarios: list[CardChoiceScenario]
    summary: dict[str, Any]
    skipped: list[dict[str, Any]] = field(default_factory=list)


def extract_trace_card_choice_scenarios(
    trace_paths: Iterable[Path | str],
    *,
    allow_degraded: bool = False,
) -> TraceExtractionResult:
    paths = [Path(path) for path in trace_paths]
    scenarios: list[CardChoiceScenario] = []
    skipped: list[dict[str, Any]] = []
    decision_rows_seen = 0
    action_rows_seen = 0
    full_count = 0
    degraded_count = 0

    for path in paths:
        decisions_by_step: dict[Any, dict[str, Any]] = {}
        for row in _read_jsonl(path):
            if row.get("type") == "decision" and row.get("decision") == "card_reward":
                decision_rows_seen += 1
                decisions_by_step[row.get("step")] = row
                continue
            if row.get("type") != "action" or row.get("decision") != "card_reward":
                continue
            action_rows_seen += 1
            step = row.get("step")
            decision = decisions_by_step.get(step)
            if decision is None:
                skipped.append({"trace": str(path), "step": step, "reason": "missing matching decision row"})
                continue
            scenario = _scenario_from_trace_pair(path, decision, row, allow_degraded=allow_degraded)
            if scenario is None:
                skipped.append(
                    {
                        "trace": str(path),
                        "step": step,
                        "reason": "decision trace lacks full player.deck/cards state",
                    }
                )
                continue
            scenarios.append(scenario)
            if scenario.state.get("trace_evidence_complete"):
                full_count += 1
            else:
                degraded_count += 1

    return TraceExtractionResult(
        scenarios=scenarios,
        skipped=skipped,
        summary={
            "trace_count": len(paths),
            "decision_rows_seen": decision_rows_seen,
            "action_rows_seen": action_rows_seen,
            "scenarios": len(scenarios),
            "full_scenarios": full_count,
            "degraded_scenarios": degraded_count,
            "skipped_incomplete": len([item for item in skipped if item.get("reason") == "decision trace lacks full player.deck/cards state"]),
            "skipped_total": len(skipped),
            "allow_degraded": allow_degraded,
        },
    )


def run_card_choice_trace_replay(
    *,
    agent: Agent,
    scenarios: Iterable[CardChoiceScenario],
    out: Path | str | None = None,
    log_level: EvalLogLevel = "decisions",
) -> dict[str, Any]:
    if log_level not in {"decisions", "full"}:
        raise ValueError("log_level must be decisions or full")
    rows = list(scenarios)
    writer = _JsonlWriter(out)
    choice_rows: list[dict[str, Any]] = []
    try:
        writer.write(
            {
                "type": "trace_replay_start",
                "scenario_count": len(rows),
                "log_level": log_level,
                "evidence_complete_count": sum(1 for scenario in rows if scenario.state.get("trace_evidence_complete")),
                "degraded_count": sum(1 for scenario in rows if not scenario.state.get("trace_evidence_complete")),
            }
        )
        for index, scenario in enumerate(rows):
            _reset_agent(agent)
            state = _copy_json(scenario.state)
            legal_actions = build_legal_actions(state)
            evidence_bundle = build_evidence_bundle(state, legal_actions)
            prompt = agent.build_prompt(state, legal_actions) if log_level == "full" else None
            writer.write(
                _trace_scenario_row(
                    index=index,
                    scenario=scenario,
                    state=state,
                    legal_actions=legal_actions,
                    evidence_bundle_hash=evidence_bundle.bundle_hash(),
                    prompt=prompt,
                    log_level=log_level,
                )
            )
            choice = _choose_trace_action(
                agent=agent,
                scenario=scenario,
                state=state,
                legal_actions=legal_actions,
                evidence_bundle_hash=evidence_bundle.bundle_hash(),
                prompt=prompt,
            )
            writer.write(choice)
            choice_rows.append(choice)
        summary = summarize_trace_replay_choices(choice_rows)
        writer.write({"type": "trace_replay_summary", **summary})
        return summary
    finally:
        writer.close()


def summarize_trace_replay_choices(choice_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(choice_rows)
    errors = sum(1 for row in choice_rows if row.get("verdict") == "error")
    invalid = sum(1 for row in choice_rows if row.get("verdict") == "invalid")
    repeated = sum(1 for row in choice_rows if row.get("repeated_original"))
    changed = sum(1 for row in choice_rows if row.get("changed_from_original"))
    skip_selected = sum(1 for row in choice_rows if row.get("selected_kind") == "card_reward_skip")
    view_selected = sum(1 for row in choice_rows if str(row.get("selected_kind") or "").startswith("view_"))
    complete = sum(1 for row in choice_rows if row.get("trace_evidence_complete"))
    return {
        "total": total,
        "errors": errors,
        "invalid": invalid,
        "evidence_complete_count": complete,
        "degraded_count": total - complete,
        "repeated_original_count": repeated,
        "changed_from_original_count": changed,
        "skip_selected_count": skip_selected,
        "view_selected_count": view_selected,
        "repeat_original_rate": _safe_ratio(repeated, total - errors - invalid),
        "skip_selection_rate": _safe_ratio(skip_selected, total - errors - invalid),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay card reward choices extracted from trace JSONL files.")
    parser.add_argument("--trace", action="append", required=True, help="Trace JSONL path. May be passed multiple times.")
    parser.add_argument("--allow-degraded", action="store_true", help="Build minimal replay scenarios when full deck state is missing.")
    parser.add_argument("--agent", default="coordinator", choices=["coordinator", "llm", "random"])
    parser.add_argument("--base-url", default=os.environ.get("STS2_BENCH_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("STS2_BENCH_MODEL"))
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "local")
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-fallback-mode", default="error", choices=["error", "action"])
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--prompt-style", default="default", choices=["default", "analysis"])
    parser.add_argument("--include-json-state", action="store_true")
    parser.add_argument("--no-run-summary", action="store_true")
    parser.add_argument("--turn-chat-window", type=int, default=4)
    parser.add_argument("--reward-posterior-evaluator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", choices=["decisions", "full"], default="decisions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    extraction = extract_trace_card_choice_scenarios(args.trace, allow_degraded=args.allow_degraded)
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
        run_summary_enabled=not args.no_run_summary,
        turn_chat_window=args.turn_chat_window,
        reward_posterior_evaluator_enabled=args.reward_posterior_evaluator,
    )
    summary = run_card_choice_trace_replay(
        agent=agent,
        scenarios=extraction.scenarios,
        out=args.out,
        log_level=args.log_level,
    )
    result = {"extraction": extraction.summary, "replay": summary}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _scenario_from_trace_pair(
    path: Path,
    decision_row: dict[str, Any],
    action_row: dict[str, Any],
    *,
    allow_degraded: bool,
) -> CardChoiceScenario | None:
    state = decision_row.get("state") if isinstance(decision_row.get("state"), dict) else {}
    original = _action_summary(action_row)
    if not original.get("label"):
        return None
    evidence_complete = _has_full_card_reward_state(state)
    if evidence_complete:
        scenario_state = _copy_json(state)
    elif allow_degraded:
        scenario_state = _minimal_card_reward_state(decision_row)
    else:
        return None
    scenario_state["trace_replay_id"] = _trace_replay_id(path, decision_row.get("step"))
    scenario_state["eval_scenario_id"] = scenario_state["trace_replay_id"]
    scenario_state["trace_source"] = str(path)
    scenario_state["trace_step"] = decision_row.get("step")
    scenario_state["trace_original_selected_label"] = original["label"]
    scenario_state["trace_original_selected_kind"] = original.get("kind")
    scenario_state["trace_evidence_complete"] = evidence_complete
    if not evidence_complete:
        scenario_state["trace_degraded_reason"] = "decision trace lacks full player.deck/cards state"

    legal_actions = build_legal_actions(scenario_state)
    acceptable = [action.label for action in legal_actions if not action.command.get("cmd") == "bench_view"]
    if not acceptable:
        acceptable = [action.label for action in legal_actions]
    return CardChoiceScenario(
        id=scenario_state["trace_replay_id"],
        category="trace_replay",
        state=scenario_state,
        oracle=CardChoiceOracle(
            acceptable_labels=acceptable,
            bad_labels=[],
            expected_behavior="trace_replay",
        ),
        tags=[
            "trace_replay",
            f"evidence_complete:{str(evidence_complete).lower()}",
            f"original_kind:{original.get('kind')}",
        ],
        notes_for_humans=f"Original trace selected {original['label']} at step {decision_row.get('step')}.",
    )


def _choose_trace_action(
    *,
    agent: Agent,
    scenario: CardChoiceScenario,
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    evidence_bundle_hash: str,
    prompt: str | None,
) -> dict[str, Any]:
    original_label = str(state.get("trace_original_selected_label") or "")
    try:
        action, meta = agent.choose(state, legal_actions, prompt=prompt)
    except Exception as exc:  # noqa: BLE001 - score per replay scenario
        return {
            "type": "trace_choice",
            "scenario_id": scenario.id,
            "trace_source": state.get("trace_source"),
            "trace_step": state.get("trace_step"),
            "trace_evidence_complete": state.get("trace_evidence_complete"),
            "original_selected_label": original_label,
            "selected_label": None,
            "selected_kind": None,
            "evidence_bundle_hash": evidence_bundle_hash,
            "verdict": "error",
            "error": str(exc),
        }
    if not isinstance(action, LegalAction) or action not in legal_actions:
        return {
            "type": "trace_choice",
            "scenario_id": scenario.id,
            "trace_source": state.get("trace_source"),
            "trace_step": state.get("trace_step"),
            "trace_evidence_complete": state.get("trace_evidence_complete"),
            "original_selected_label": original_label,
            "selected_label": getattr(action, "label", None),
            "selected_kind": getattr(action, "kind", None),
            "evidence_bundle_hash": evidence_bundle_hash,
            "agent_meta": _safe_json(meta if isinstance(meta, dict) else {}),
            "verdict": "invalid",
        }
    repeated = action.label == original_label
    return {
        "type": "trace_choice",
        "scenario_id": scenario.id,
        "trace_source": state.get("trace_source"),
        "trace_step": state.get("trace_step"),
        "trace_evidence_complete": state.get("trace_evidence_complete"),
        "original_selected_label": original_label,
        "original_selected_kind": state.get("trace_original_selected_kind"),
        "selected_action_id": action.action_id,
        "selected_label": action.label,
        "selected_kind": action.kind,
        "evidence_bundle_hash": evidence_bundle_hash,
        "agent_meta": _safe_json(meta if isinstance(meta, dict) else {}),
        "verdict": "ok",
        "repeated_original": repeated,
        "changed_from_original": not repeated,
    }


def _trace_scenario_row(
    *,
    index: int,
    scenario: CardChoiceScenario,
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    evidence_bundle_hash: str,
    prompt: str | None,
    log_level: EvalLogLevel,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "trace_scenario",
        "index": index,
        "scenario_id": scenario.id,
        "trace_source": state.get("trace_source"),
        "trace_step": state.get("trace_step"),
        "trace_evidence_complete": state.get("trace_evidence_complete"),
        "trace_degraded_reason": state.get("trace_degraded_reason"),
        "original_selected_label": state.get("trace_original_selected_label"),
        "state": {
            "decision": state.get("decision"),
            "act": state.get("act") or (state.get("context") or {}).get("act"),
            "floor": state.get("floor") or (state.get("context") or {}).get("floor"),
            "hp": (state.get("player") or {}).get("hp"),
            "max_hp": (state.get("player") or {}).get("max_hp"),
            "deck_size": (state.get("player") or {}).get("deck_size"),
            "card_count": len(state.get("cards") or []),
        },
        "legal_actions": [{"action_id": action.action_id, "kind": action.kind, "label": action.label} for action in legal_actions],
        "evidence_bundle_hash": evidence_bundle_hash,
    }
    if log_level == "full":
        row["state"] = compact_state(state)
        row["legal_actions"] = [action.to_log_dict() for action in legal_actions]
        row["prompt"] = prompt
    return row


def _minimal_card_reward_state(decision_row: dict[str, Any]) -> dict[str, Any]:
    summary = decision_row.get("state") if isinstance(decision_row.get("state"), dict) else {}
    cards = _cards_from_legal_action_summaries(decision_row.get("legal_actions") or [])
    return {
        "decision": "card_reward",
        "act": summary.get("act"),
        "floor": summary.get("floor"),
        "context": {
            "act": summary.get("act"),
            "floor": summary.get("floor"),
            "room_type": summary.get("room_type"),
        },
        "player": {
            "hp": summary.get("hp"),
            "max_hp": summary.get("max_hp"),
            "gold": summary.get("gold"),
            "deck_size": summary.get("deck_size"),
        },
        "cards": cards,
        "can_skip": True,
        "last_action_result": summary.get("last_action_result"),
    }


def _cards_from_legal_action_summaries(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for action in actions:
        if action.get("kind") != "card_reward":
            continue
        match = re.search(r"pick card (?P<index>\d+): (?P<name>.+)$", str(action.get("label") or ""))
        if not match:
            continue
        cards.append(
            {
                "index": int(match.group("index")),
                "name": match.group("name").strip(),
            }
        )
    return cards


def _has_full_card_reward_state(state: dict[str, Any]) -> bool:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    return (
        state.get("decision") == "card_reward"
        and isinstance(player.get("deck"), list)
        and isinstance(state.get("cards"), list)
    )


def _action_summary(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    return {
        "label": action.get("label") or row.get("label"),
        "kind": action.get("kind") or row.get("kind"),
        "action_id": action.get("action_id") if "action_id" in action else row.get("action_id"),
    }


def _trace_replay_id(path: Path, step: Any) -> str:
    safe_step = str(step).replace(" ", "_")
    return f"{path.stem}:step_{safe_step}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _reset_agent(agent: Agent) -> None:
    reset = getattr(agent, "reset_episode", None)
    if callable(reset):
        reset()


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


class _JsonlWriter:
    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self.handle = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        if self.handle:
            self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.handle.flush()

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
