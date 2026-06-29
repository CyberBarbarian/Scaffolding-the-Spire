"""Synthetic card-reward choice evaluation for LLM policies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .actions import LegalAction, build_legal_actions
from .agents import Agent
from .context import compact_state
from .evidence import EvidenceBundle, build_evidence_bundle
from .multi_agent import CoordinatorAgent, OutOfCombatContextAgent
from .runner import LogLevel, agent_from_args
from .run_context import RunContextStore


CARD_CHOICE_CATEGORIES = {
    "defense_gap_indirect",
    "skip_is_best",
    "synergy_pick",
    "anti_synergy_trap",
    "overweight_attack",
    "resource_curve",
    "redundancy",
    "ambiguous_good_pick",
}

EvalLogLevel = Literal["decisions", "full"]


@dataclass(frozen=True)
class CardChoiceOracle:
    acceptable_labels: list[str]
    bad_labels: list[str] = field(default_factory=list)
    expected_behavior: str = ""
    rationale_keywords: list[str] = field(default_factory=list)

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "acceptable_labels": list(self.acceptable_labels),
            "bad_labels": list(self.bad_labels),
            "expected_behavior": self.expected_behavior,
            "rationale_keywords": list(self.rationale_keywords),
        }


@dataclass(frozen=True)
class CardChoiceScenario:
    id: str
    category: str
    state: dict[str, Any]
    oracle: CardChoiceOracle
    tags: list[str] = field(default_factory=list)
    notes_for_humans: str = ""

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "tags": list(self.tags),
            "notes_for_humans": self.notes_for_humans,
        }


@dataclass(frozen=True)
class CardChoiceVerdict:
    verdict: str
    failure_tags: list[str]
    matched_expected_behavior: str
    rationale_keyword_hits: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "failure_tags": list(self.failure_tags),
            "matched_expected_behavior": self.matched_expected_behavior,
            "rationale_keyword_hits": list(self.rationale_keyword_hits),
        }


class _JsonlWriter:
    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self.handle = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        if not self.handle:
            return
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


def load_card_choice_scenarios() -> list[CardChoiceScenario]:
    return list(_build_scenarios())


def choose_scenarios(
    scenarios: Iterable[CardChoiceScenario] | None = None,
    *,
    scenario_ids: list[str] | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[CardChoiceScenario]:
    selected = list(scenarios if scenarios is not None else load_card_choice_scenarios())
    if scenario_ids:
        wanted = set(scenario_ids)
        selected = [scenario for scenario in selected if scenario.id in wanted]
        missing = wanted - {scenario.id for scenario in selected}
        if missing:
            raise ValueError(f"Unknown scenario ids: {', '.join(sorted(missing))}")
    if categories:
        wanted_categories = set(categories)
        unknown = wanted_categories - CARD_CHOICE_CATEGORIES
        if unknown:
            raise ValueError(f"Unknown categories: {', '.join(sorted(unknown))}")
        selected = [scenario for scenario in selected if scenario.category in wanted_categories]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected = selected[:limit]
    return selected


def run_card_choice_eval(
    *,
    agent: Agent,
    scenarios: Iterable[CardChoiceScenario] | None = None,
    out: Path | str | None = None,
    log_level: EvalLogLevel = "decisions",
) -> dict[str, Any]:
    if log_level not in {"decisions", "full"}:
        raise ValueError("log_level must be decisions or full")

    rows = list(scenarios if scenarios is not None else load_card_choice_scenarios())
    writer = _JsonlWriter(out)
    choice_rows: list[dict[str, Any]] = []
    try:
        writer.write(
            {
                "type": "start",
                "scenario_count": len(rows),
                "log_level": log_level,
                "categories": sorted({scenario.category for scenario in rows}),
            }
        )
        for index, scenario in enumerate(rows):
            _reset_agent(agent)
            state = _copy_json(scenario.state)
            state["eval_scenario_id"] = scenario.id
            legal_actions = build_legal_actions(state)
            evidence_bundle = build_evidence_bundle(
                state,
                legal_actions,
                store=_agent_run_context_store(agent),
            )
            prompt = agent.build_prompt(state, legal_actions) if log_level == "full" else None
            writer.write(
                _scenario_log_row(
                    index=index,
                    scenario=scenario,
                    state=state,
                    legal_actions=legal_actions,
                    evidence_bundle=evidence_bundle,
                    prompt=prompt,
                    log_level=log_level,
                )
            )
            choice_row = _choose_and_score(
                agent=agent,
                scenario=scenario,
                state=state,
                legal_actions=legal_actions,
                evidence_bundle=evidence_bundle,
                prompt=prompt,
            )
            writer.write(choice_row)
            choice_rows.append(choice_row)

        summary = summarize_card_choice_results(choice_rows)
        writer.write({"type": "summary", **summary})
        return summary
    finally:
        writer.close()


def summarize_card_choice_results(choice_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(choice_rows)
    correct = sum(1 for row in choice_rows if row.get("verdict") == "correct")
    incorrect = sum(1 for row in choice_rows if row.get("verdict") == "incorrect")
    errors = sum(1 for row in choice_rows if row.get("verdict") == "error")
    invalid = sum(1 for row in choice_rows if row.get("verdict") == "invalid")
    by_category: dict[str, dict[str, Any]] = {}
    for row in choice_rows:
        category = str(row.get("category") or "?")
        bucket = by_category.setdefault(category, {"total": 0, "correct": 0, "accuracy": 0.0})
        bucket["total"] += 1
        if row.get("verdict") == "correct":
            bucket["correct"] += 1
    for bucket in by_category.values():
        bucket["accuracy"] = _safe_ratio(bucket["correct"], bucket["total"])

    skip_expected = [row for row in choice_rows if row.get("expected_behavior") == "skip"]
    skip_selected = [row for row in choice_rows if row.get("selected_kind") == "card_reward_skip"]
    skip_correct = [row for row in skip_selected if row.get("verdict") == "correct"]
    trap_rows = [row for row in choice_rows if row.get("category") == "anti_synergy_trap"]
    trap_failures = [
        row
        for row in trap_rows
        if row.get("verdict") != "correct" and "bad_label" in (row.get("failure_tags") or [])
    ]

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "invalid": invalid,
        "accuracy": _safe_ratio(correct, total),
        "category_accuracy": by_category,
        "skip_precision": _safe_ratio(len(skip_correct), len(skip_selected)),
        "skip_recall": _safe_ratio(
            sum(1 for row in skip_expected if row.get("verdict") == "correct"),
            len(skip_expected),
        ),
        "trap_failure_rate": _safe_ratio(len(trap_failures), len(trap_rows)) if trap_rows else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic card reward choice evaluation.")
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
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-level", choices=["decisions", "full"], default="decisions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    scenarios = choose_scenarios(
        scenario_ids=args.scenario_id or None,
        categories=args.category or None,
        limit=args.limit,
    )
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
    summary = run_card_choice_eval(agent=agent, scenarios=scenarios, out=args.out, log_level=args.log_level)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _choose_and_score(
    *,
    agent: Agent,
    scenario: CardChoiceScenario,
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    evidence_bundle: EvidenceBundle,
    prompt: str | None,
) -> dict[str, Any]:
    try:
        action, meta = agent.choose(state, legal_actions, prompt=prompt)
    except Exception as exc:  # noqa: BLE001 - eval should score per scenario
        verdict = CardChoiceVerdict(
            verdict="error",
            failure_tags=["agent_error"],
            matched_expected_behavior=scenario.oracle.expected_behavior,
            rationale_keyword_hits=[],
        )
        return {
            "type": "choice",
            "scenario_id": scenario.id,
            "category": scenario.category,
            "selected_label": None,
            "selected_kind": None,
            "evidence_bundle_hash": evidence_bundle.bundle_hash(),
            "error": str(exc),
            **verdict.to_dict(),
            "expected_behavior": scenario.oracle.expected_behavior,
        }

    if not isinstance(action, LegalAction) or action not in legal_actions:
        selected_label = getattr(action, "label", None)
        verdict = CardChoiceVerdict(
            verdict="invalid",
            failure_tags=["illegal_action"],
            matched_expected_behavior=scenario.oracle.expected_behavior,
            rationale_keyword_hits=[],
        )
        return {
            "type": "choice",
            "scenario_id": scenario.id,
            "category": scenario.category,
            "selected_label": selected_label,
            "selected_kind": getattr(action, "kind", None),
            "evidence_bundle_hash": evidence_bundle.bundle_hash(),
            "agent_meta": _safe_json(meta if isinstance(meta, dict) else {}),
            **verdict.to_dict(),
            "expected_behavior": scenario.oracle.expected_behavior,
        }

    meta = dict(meta)
    meta.setdefault("evidence_bundle_hash", evidence_bundle.bundle_hash())
    verdict = _score_choice(scenario, action, meta)
    return {
        "type": "choice",
        "scenario_id": scenario.id,
        "category": scenario.category,
        "selected_action_id": action.action_id,
        "selected_label": action.label,
        "selected_kind": action.kind,
        "evidence_bundle_hash": evidence_bundle.bundle_hash(),
        "agent_meta": _safe_json(meta),
        **verdict.to_dict(),
        "expected_behavior": scenario.oracle.expected_behavior,
    }


def _score_choice(
    scenario: CardChoiceScenario,
    action: LegalAction,
    meta: dict[str, Any],
) -> CardChoiceVerdict:
    selected = action.label
    correct = selected in scenario.oracle.acceptable_labels
    failure_tags: list[str] = []
    if not correct:
        failure_tags.append(f"category:{scenario.category}")
        failure_tags.append(f"expected:{scenario.oracle.expected_behavior}")
        if selected in scenario.oracle.bad_labels:
            failure_tags.append("bad_label")
        if action.kind == "card_reward_skip":
            failure_tags.append("unexpected_skip")
        elif "skip card reward" in scenario.oracle.acceptable_labels:
            failure_tags.append("missed_skip")

    reason = ""
    parsed = meta.get("parsed")
    if isinstance(parsed, dict):
        reason = str(parsed.get("reason") or "")
    reason_lower = reason.lower()
    keyword_hits = [
        keyword
        for keyword in scenario.oracle.rationale_keywords
        if keyword.lower() in reason_lower
    ]
    return CardChoiceVerdict(
        verdict="correct" if correct else "incorrect",
        failure_tags=failure_tags,
        matched_expected_behavior=scenario.oracle.expected_behavior,
        rationale_keyword_hits=keyword_hits,
    )


def _scenario_log_row(
    *,
    index: int,
    scenario: CardChoiceScenario,
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    evidence_bundle: EvidenceBundle,
    prompt: str | None,
    log_level: EvalLogLevel,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "scenario",
        "index": index,
        "scenario_id": scenario.id,
        "category": scenario.category,
        "tags": list(scenario.tags),
        "notes_for_humans": scenario.notes_for_humans,
        "state": _state_summary(state),
        "legal_actions": [_legal_action_summary(action) for action in legal_actions],
        "evidence_bundle_hash": evidence_bundle.bundle_hash(),
        "evidence_bundle": evidence_bundle.summary_dict(),
    }
    if log_level == "full":
        row["state"] = compact_state(state)
        row["legal_actions"] = [action.to_log_dict() for action in legal_actions]
        row["evidence_bundle"] = evidence_bundle.to_dict()
        row["prompt"] = prompt
    return row


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player") or {}
    context = state.get("context") or {}
    return {
        "decision": state.get("decision"),
        "act": state.get("act") or context.get("act"),
        "floor": state.get("floor") or context.get("floor"),
        "room_type": context.get("room_type"),
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "deck_size": player.get("deck_size"),
        "card_count": len(state.get("cards") or []),
        "can_skip": state.get("can_skip"),
        "last_action_result": state.get("last_action_result"),
    }


def _legal_action_summary(action: LegalAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "kind": action.kind,
        "label": action.label,
    }


def _agent_run_context_store(agent: Agent) -> RunContextStore | None:
    if isinstance(agent, CoordinatorAgent):
        return agent.store
    if isinstance(agent, OutOfCombatContextAgent):
        return agent.store
    store = getattr(agent, "store", None)
    return store if isinstance(store, RunContextStore) else None


def _reset_agent(agent: Agent) -> None:
    reset = getattr(agent, "reset_episode", None)
    if callable(reset):
        reset()


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _build_scenarios() -> list[CardChoiceScenario]:
    specs = [
        _spec(
            "cc_001",
            "defense_gap_indirect",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Twin Strike", "Blood Wall"],
            offers=["Twin Strike", "Shrug It Off", "Havoc"],
            acceptable=["Shrug It Off"],
            hp=48,
            last_loss=18,
            tags=["moderate_hp_feedback", "slight_attack_bias"],
            notes="Defense is only one card behind attacks; the model must value a premium block-and-draw card without an obvious gap.",
        ),
        _spec(
            "cc_002",
            "defense_gap_indirect",
            deck=["Strike"] * 4 + ["Defend"] * 3 + ["Bash", "Pommel Strike", "True Grit", "Stone Armor"],
            offers=["Stone Armor", "Cinder", "Havoc"],
            acceptable=["Stone Armor"],
            hp=45,
            last_loss=20,
            tags=["moderate_hp_feedback", "defense_quality_gap"],
            notes="The deck has some defense, but another high-quality defensive card is better than speculative exhaust attacks.",
        ),
        _spec(
            "cc_003",
            "defense_gap_indirect",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Body Slam", "Bash", "True Grit"],
            offers=["Flame Barrier", "Bludgeon", "Havoc"],
            acceptable=["Flame Barrier"],
            hp=42,
            last_loss=22,
            tags=["body_slam_needs_better_block", "moderate_hp_feedback"],
            notes="Body Slam exists, but the deck still wants a stronger block source rather than another damage card.",
        ),
        _spec(
            "cc_004",
            "skip_is_best",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Shrug It Off", "Stone Armor", "Pommel Strike", "Bloodletting"],
            offers=["Strike", "Defend", "Havoc"],
            acceptable=["Skip"],
            tags=["medium_quality_deck", "low_impact_offer"],
            notes="Medium-sized deck already has draw, block, and energy; the offer only adds basics or unreliable top-deck play.",
        ),
        _spec(
            "cc_005",
            "skip_is_best",
            deck=["Strike"] * 5
            + ["Defend"] * 5
            + ["Shrug It Off"] * 2
            + ["Stone Armor"] * 2
            + ["Pommel Strike"] * 2
            + ["Bloodletting", "Offering", "Bash", "Bludgeon"],
            offers=["Strike", "Defend", "Havoc"],
            acceptable=["Skip"],
            tags=["large_high_quality_deck", "duplicate_offer"],
            notes="Large but high-quality deck where every offered card is below the current card-quality bar.",
        ),
        _spec(
            "cc_006",
            "skip_is_best",
            deck=["Strike"] * 4
            + ["Defend"] * 4
            + ["Barricade", "Impervious", "Blood Wall", "True Grit", "Armaments", "Bash"],
            offers=["Strike", "Havoc", "Cinder"],
            acceptable=["Skip"],
            tags=["clear_block_plan", "off_plan_offer"],
            notes="Clear block-retention plan already exists; the offer does not improve that plan.",
        ),
        _spec(
            "cc_007",
            "synergy_pick",
            deck=["Strike"] * 7 + ["Defend"] * 4 + ["Bash", "Twin Strike", "Pommel Strike"],
            offers=["Perfected Strike", "Havoc", "Stone Armor"],
            acceptable=["Perfected Strike"],
            tags=["strike_density"],
            notes="Many Strike-name attacks make Perfected Strike contextually attractive.",
        ),
        _spec(
            "cc_008",
            "synergy_pick",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Shrug It Off"] * 2 + ["Stone Armor"] * 2 + ["Entrench"],
            offers=["Body Slam", "Havoc", "Twin Strike"],
            acceptable=["Body Slam"],
            tags=["block_payoff"],
            notes="Block density supports Body Slam.",
        ),
        _spec(
            "cc_009",
            "synergy_pick",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Bludgeon", "Barricade", "Impervious"],
            offers=["Bloodletting", "Drum of Battle", "Havoc"],
            acceptable=["Bloodletting"],
            tags=["expensive_payoff", "energy_patch"],
            notes="Expensive cards make Bloodletting attractive; Drum of Battle lacks exhaust payoff support here.",
        ),
        _spec(
            "cc_010",
            "anti_synergy_trap",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Twin Strike", "Pommel Strike", "Blood Wall"],
            offers=["Bludgeon", "Shrug It Off", "Cinder"],
            acceptable=["Shrug It Off"],
            hp=44,
            last_loss=19,
            tags=["strong_single_card_trap", "moderate_hp_feedback"],
            notes="Bludgeon is a strong single card, but the existing curve and recent damage make Shrug It Off a better fit.",
        ),
        _spec(
            "cc_011",
            "anti_synergy_trap",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Pommel Strike", "Anger", "Bloodletting"],
            offers=["Barricade", "Pommel Strike", "Havoc"],
            acceptable=["Pommel Strike"],
            tags=["rare_power_trap", "insufficient_retention_support"],
            notes="Barricade is attractive, but the deck does not yet make enough persistent block to justify it.",
        ),
        _spec(
            "cc_012",
            "anti_synergy_trap",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Barricade", "Impervious", "Blood Wall", "True Grit"],
            offers=["Bludgeon", "Armaments", "Bloodletting"],
            acceptable=["Armaments"],
            tags=["rare_attack_trap", "upgrade_support"],
            notes="Several unupgraded block-plan cards become much stronger when upgraded, so Armaments beats another payoff card.",
        ),
        _spec(
            "cc_013",
            "overweight_attack",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Twin Strike"] * 2 + ["Pommel Strike", "Bash", "True Grit", "Blood Wall"],
            offers=["Twin Strike", "Shrug It Off", "Cinder"],
            acceptable=["Shrug It Off"],
            tags=["attack_overweight", "low_quality_block"],
            notes="The deck has defense, but attacks still outnumber quality block sources.",
        ),
        _spec(
            "cc_014",
            "overweight_attack",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Anger"] * 2 + ["Twin Strike", "Pommel Strike", "Bash", "True Grit"],
            offers=["Stone Armor", "Anger", "Havoc"],
            acceptable=["Stone Armor"],
            tags=["attack_overweight", "block_patch"],
            notes="Attacks are only moderately ahead, but the best incremental improvement is still a defensive card.",
        ),
        _spec(
            "cc_015",
            "overweight_attack",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Twin Strike"] * 2 + ["Pommel Strike", "Anger", "Bash", "Blood Wall"],
            offers=["Armaments", "Bludgeon", "Cinder"],
            acceptable=["Armaments"],
            tags=["attack_overweight", "upgrade_many_basics"],
            notes="The deck can use improvement, but another attack is less useful than upgrading existing basics and block.",
        ),
        _spec(
            "cc_016",
            "resource_curve",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Bludgeon", "Barricade", "Impervious"],
            offers=["Bloodletting", "Bludgeon", "Barricade"],
            acceptable=["Bloodletting"],
            tags=["too_many_expensive_cards", "low_cost_patch"],
            notes="High-cost curve makes an energy card more valuable than another expensive payoff.",
        ),
        _spec(
            "cc_017",
            "resource_curve",
            deck=["Strike"] * 5 + ["Defend"] * 5 + ["Anger"] * 2 + ["Pommel Strike"],
            offers=["Bludgeon", "True Grit", "Havoc"],
            acceptable=["Bludgeon"],
            tags=["needs_payoff", "low_curve"],
            notes="Low-curve deck can afford one large damage payoff.",
        ),
        _spec(
            "cc_018",
            "resource_curve",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bludgeon"] * 2 + ["Bash", "Barricade"],
            offers=["Pommel Strike", "Bludgeon", "Havoc"],
            acceptable=["Pommel Strike"],
            tags=["expensive_duplicates", "cycle_patch"],
            notes="Another expensive card is less useful than cheap cycle.",
        ),
        _spec(
            "cc_019",
            "redundancy",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Shrug It Off"] * 3 + ["Stone Armor"] * 2 + ["Bash"],
            offers=["Shrug It Off", "Twin Strike", "Havoc"],
            acceptable=["Skip"],
            tags=["duplicate_good_card", "large_enough_defense"],
            notes="Even a normally good card can be redundant here.",
        ),
        _spec(
            "cc_020",
            "redundancy",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bloodletting"] * 2 + ["Offering", "Pommel Strike", "Bash"],
            offers=["Bloodletting", "Havoc", "Anger"],
            acceptable=["Skip"],
            tags=["duplicate_energy", "low_impact_offer"],
            notes="Multiple HP-for-energy cards reduce the value of another Bloodletting.",
        ),
        _spec(
            "cc_021",
            "redundancy",
            deck=["Strike"] * 7 + ["Twin Strike"] * 3 + ["Pommel Strike"] * 2 + ["Defend"] * 4 + ["Bash"],
            offers=["Twin Strike", "Strike", "Pommel Strike"],
            acceptable=["Skip"],
            tags=["attack_duplicates"],
            notes="Every offered attack duplicates an already crowded role.",
        ),
        _spec(
            "cc_022",
            "ambiguous_good_pick",
            deck=["Strike"] * 6 + ["Defend"] * 2 + ["Bash", "Twin Strike"],
            offers=["Shrug It Off", "Stone Armor", "Twin Strike"],
            acceptable=["Shrug It Off", "Stone Armor"],
            hp=37,
            last_loss=21,
            tags=["two_defense_answers"],
            notes="Two defensive cards are both acceptable improvements.",
        ),
        _spec(
            "cc_023",
            "ambiguous_good_pick",
            deck=["Strike"] * 7 + ["Defend"] * 4 + ["Twin Strike", "Bash"],
            offers=["Perfected Strike", "Pommel Strike", "Havoc"],
            acceptable=["Perfected Strike", "Pommel Strike"],
            tags=["strike_synergy_or_cycle"],
            notes="Both direct Strike payoff and cheap cycle are defensible.",
        ),
        _spec(
            "cc_024",
            "ambiguous_good_pick",
            deck=["Strike"] * 5 + ["Defend"] * 5 + ["Bash", "Armaments", "Pommel Strike"],
            offers=["Shrug It Off", "Pommel Strike", "Havoc"],
            acceptable=["Shrug It Off", "Pommel Strike"],
            tags=["balanced_pick"],
            notes="Both draw/block and draw/attack are acceptable; Havoc is the trap.",
        ),
    ]
    return [_scenario_from_spec(item) for item in specs]


def _spec(
    scenario_id: str,
    category: str,
    *,
    deck: list[str],
    offers: list[str],
    acceptable: list[str],
    hp: int = 64,
    last_loss: int | None = None,
    tags: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": scenario_id,
        "category": category,
        "deck": deck,
        "offers": offers,
        "acceptable": acceptable,
        "hp": hp,
        "last_loss": last_loss,
        "tags": list(tags or []),
        "notes": notes,
    }


def _scenario_from_spec(spec: dict[str, Any]) -> CardChoiceScenario:
    offer_cards = [_card(name, index=index) for index, name in enumerate(spec["offers"])]
    state = {
        "decision": "card_reward",
        "act": 1,
        "floor": 6,
        "context": {"act": 1, "floor": 6, "room_type": "Monster"},
        "player": {
            "hp": spec["hp"],
            "max_hp": 80,
            "gold": 84,
            "deck_size": len(spec["deck"]),
            "relics": [],
            "potions": [],
            "deck": [_card(name) for name in spec["deck"]],
        },
        "cards": offer_cards,
        "can_skip": True,
    }
    if spec.get("last_loss") is not None:
        state["last_action_result"] = {
            "action_label": "previous combat report",
            "action_kind": "combat_report",
            "decision_before": "combat_play",
            "decision_after": "card_reward",
            "changes": [
                f"previous fight lost {spec['last_loss']} HP over 5 turns",
                "combat ended with card reward available",
            ],
        }
    acceptable_labels = [_label_for_offer(name, offer_cards) for name in spec["acceptable"]]
    bad_labels = [
        action_label
        for action_label in [_label_for_offer(name, offer_cards) for name in [*spec["offers"], "Skip"]]
        if action_label not in acceptable_labels
    ]
    expected_behavior = "skip" if acceptable_labels == ["skip card reward"] else spec["category"]
    oracle = CardChoiceOracle(
        acceptable_labels=acceptable_labels,
        bad_labels=bad_labels,
        expected_behavior=expected_behavior,
        rationale_keywords=_rationale_keywords(spec["category"]),
    )
    return CardChoiceScenario(
        id=spec["id"],
        category=spec["category"],
        state=state,
        oracle=oracle,
        tags=spec["tags"],
        notes_for_humans=spec["notes"],
    )


def _label_for_offer(name: str, offer_cards: list[dict[str, Any]]) -> str:
    if name == "Skip":
        return "skip card reward"
    for card in offer_cards:
        if card["name"] == name:
            return f"pick card {card['index']}: {name}"
    raise ValueError(f"Offer {name!r} is not present")


def _rationale_keywords(category: str) -> list[str]:
    return {
        "defense_gap_indirect": ["block", "damage", "hp"],
        "skip_is_best": ["skip", "deck", "low impact"],
        "synergy_pick": ["synergy", "deck", "payoff"],
        "anti_synergy_trap": ["synergy", "cost", "support"],
        "overweight_attack": ["attack", "block", "redundant"],
        "resource_curve": ["cost", "energy", "curve"],
        "redundancy": ["duplicate", "redundant", "skip"],
        "ambiguous_good_pick": ["both", "acceptable", "draw"],
    }.get(category, [])


def _card(name: str, *, index: int | None = None) -> dict[str, Any]:
    base = dict(_CARD_LIBRARY[name])
    if index is not None:
        base["index"] = index
    return base


_CARD_LIBRARY: dict[str, dict[str, Any]] = {
    "Strike": {
        "id": "CARD.STRIKE_IRONCLAD",
        "name": "Strike",
        "type": "Attack",
        "cost": 1,
        "rarity": "Starter",
        "description": "Deal 6 damage.",
        "stats": {"damage": 6},
    },
    "Defend": {
        "id": "CARD.DEFEND_IRONCLAD",
        "name": "Defend",
        "type": "Skill",
        "cost": 1,
        "rarity": "Starter",
        "description": "Gain 5 Block.",
        "stats": {"block": 5},
    },
    "Bash": {
        "id": "CARD.BASH",
        "name": "Bash",
        "type": "Attack",
        "cost": 2,
        "rarity": "Starter",
        "description": "Deal 8 damage. Apply 2 Vulnerable.",
        "stats": {"damage": 8},
    },
    "Twin Strike": {
        "id": "CARD.TWIN_STRIKE",
        "name": "Twin Strike",
        "type": "Attack",
        "cost": 1,
        "rarity": "Common",
        "description": "Deal 5 damage twice.",
        "stats": {"damage": 10},
    },
    "Shrug It Off": {
        "id": "CARD.SHRUG_IT_OFF",
        "name": "Shrug It Off",
        "type": "Skill",
        "cost": 1,
        "rarity": "Common",
        "description": "Gain 8 Block. Draw 1 card.",
        "stats": {"block": 8, "cards": 1},
    },
    "Stone Armor": {
        "id": "CARD.STONE_ARMOR",
        "name": "Stone Armor",
        "type": "Skill",
        "cost": 1,
        "rarity": "Common",
        "description": "Gain Plating.",
        "stats": {"plating": 4},
    },
    "Flame Barrier": {
        "id": "CARD.FLAME_BARRIER",
        "name": "Flame Barrier",
        "type": "Skill",
        "cost": 2,
        "rarity": "Uncommon",
        "description": "Gain 12 Block. Deal damage back when attacked.",
        "stats": {"block": 12},
    },
    "Blood Wall": {
        "id": "CARD.BLOOD_WALL",
        "name": "Blood Wall",
        "type": "Skill",
        "cost": 1,
        "rarity": "Common",
        "description": "Lose HP. Gain Block.",
        "stats": {"block": 14, "hp_loss": 2},
    },
    "True Grit": {
        "id": "CARD.TRUE_GRIT",
        "name": "True Grit",
        "type": "Skill",
        "cost": 1,
        "rarity": "Common",
        "description": "Gain Block. Exhaust 1 card at random.",
        "stats": {"block": 7, "exhaust": 1},
    },
    "Impervious": {
        "id": "CARD.IMPERVIOUS",
        "name": "Impervious",
        "type": "Skill",
        "cost": 2,
        "rarity": "Rare",
        "description": "Gain a large amount of Block.",
        "stats": {"block": 30},
    },
    "Bloodletting": {
        "id": "CARD.BLOODLETTING",
        "name": "Bloodletting",
        "type": "Skill",
        "cost": 0,
        "rarity": "Uncommon",
        "description": "Lose HP. Gain Energy.",
        "stats": {"hp_loss": 3, "energy": 2},
    },
    "Offering": {
        "id": "CARD.OFFERING",
        "name": "Offering",
        "type": "Skill",
        "cost": 0,
        "rarity": "Rare",
        "description": "Lose HP. Gain Energy. Draw cards.",
        "stats": {"hp_loss": 6, "energy": 2, "cards": 3},
    },
    "Havoc": {
        "id": "CARD.HAVOC",
        "name": "Havoc",
        "type": "Skill",
        "cost": 1,
        "rarity": "Common",
        "description": "Play the top card of your draw pile and exhaust it.",
    },
    "Anger": {
        "id": "CARD.ANGER",
        "name": "Anger",
        "type": "Attack",
        "cost": 0,
        "rarity": "Common",
        "description": "Deal 6 damage. Add a copy of this card to your discard pile.",
        "stats": {"damage": 6},
    },
    "Cinder": {
        "id": "CARD.CINDER",
        "name": "Cinder",
        "type": "Attack",
        "cost": 1,
        "rarity": "Common",
        "description": "Deal damage. Exhaust 1 card at random.",
        "stats": {"damage": 10, "exhaust": 1},
    },
    "Pommel Strike": {
        "id": "CARD.POMMEL_STRIKE",
        "name": "Pommel Strike",
        "type": "Attack",
        "cost": 1,
        "rarity": "Common",
        "description": "Deal 9 damage. Draw 1 card.",
        "stats": {"damage": 9, "cards": 1},
    },
    "Perfected Strike": {
        "id": "CARD.PERFECTED_STRIKE",
        "name": "Perfected Strike",
        "type": "Attack",
        "cost": 2,
        "rarity": "Common",
        "description": "Deal more damage for each card containing Strike in your deck.",
        "stats": {"damage": 18},
    },
    "Body Slam": {
        "id": "CARD.BODY_SLAM",
        "name": "Body Slam",
        "type": "Attack",
        "cost": 1,
        "rarity": "Common",
        "description": "Deal damage equal to your Block.",
    },
    "Entrench": {
        "id": "CARD.ENTRENCH",
        "name": "Entrench",
        "type": "Skill",
        "cost": 2,
        "rarity": "Uncommon",
        "description": "Double your Block.",
    },
    "Drum of Battle": {
        "id": "CARD.DRUM_OF_BATTLE",
        "name": "Drum of Battle",
        "type": "Power",
        "cost": 1,
        "rarity": "Uncommon",
        "description": "Draw cards. At the start of your turn, exhaust the top card of your draw pile.",
        "stats": {"cards": 2, "exhaust_top_each_turn": 1},
    },
    "Bludgeon": {
        "id": "CARD.BLUDGEON",
        "name": "Bludgeon",
        "type": "Attack",
        "cost": 3,
        "rarity": "Rare",
        "description": "Deal 32 damage.",
        "stats": {"damage": 32},
    },
    "Barricade": {
        "id": "CARD.BARRICADE",
        "name": "Barricade",
        "type": "Power",
        "cost": 3,
        "rarity": "Rare",
        "description": "Block is not removed at start of turn.",
    },
    "Armaments": {
        "id": "CARD.ARMAMENTS",
        "name": "Armaments",
        "type": "Skill",
        "cost": 1,
        "rarity": "Common",
        "description": "Gain 5 Block. Upgrade a card in your hand for the combat.",
        "stats": {"block": 5},
    },
}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
