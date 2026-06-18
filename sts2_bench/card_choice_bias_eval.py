"""Synthetic card-reward bias probes for LLM card choice policies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from .agents import Agent
from .card_choice_eval import (
    CardChoiceOracle,
    CardChoiceScenario,
    EvalLogLevel,
    run_card_choice_eval,
)
from .runner import agent_from_args


CARD_CHOICE_BIAS_CATEGORIES = {
    "attack_temptation_ladder",
    "skip_threshold_matrix",
    "redundancy_dose_response",
    "rare_damage_trap",
    "energy_vs_damage_curve",
    "havoc_semantic_trap",
}

EXPECTED_BIAS_CATEGORY_COUNTS = {
    "attack_temptation_ladder": 8,
    "skip_threshold_matrix": 9,
    "redundancy_dose_response": 10,
    "rare_damage_trap": 6,
    "energy_vs_damage_curve": 5,
    "havoc_semantic_trap": 4,
}

PromptInjectionMode = Literal["none", "generic", "scenario-specific"]


GENERIC_CARD_CHOICE_POLICY_HINT = "\n".join(
    [
        "Card choice policy hints (generic):",
        "- Card rewards are optional. Treat skip card reward as a first-class legal action, not as failure to act.",
        "- Compare each offered card by marginal utility: what new role does it add, and what does it dilute?",
        "- Large or already high-quality decks often need fewer cards; low-impact, off-plan, or redundant cards can make the deck worse.",
        "- Count existing roles before adding more: attacks, block sources, draw, energy, scaling, exhaust support, and high-cost cards.",
        "- A strong standalone card is not automatically correct if the deck lacks the support, energy, HP buffer, or role need for it.",
        "- Be cautious with duplicate cards and HP-cost cards; the second or fifth copy can be much worse than the first.",
        "- If a card relies on synergy, verify the support is already visible in the current deck instead of assuming it will appear later.",
    ]
)


class PromptInjectedCardChoiceAgent:
    def __init__(self, inner: Agent, *, prompt_injection: PromptInjectionMode) -> None:
        self.inner = inner
        self.prompt_injection = prompt_injection

    def build_prompt(self, state: dict[str, Any], legal_actions: list[Any]) -> str:
        base_prompt = self.inner.build_prompt(state, legal_actions)
        hint = build_card_choice_policy_hint(state, legal_actions, prompt_injection=self.prompt_injection)
        if not hint:
            return base_prompt
        return "\n\n".join([base_prompt, hint])

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[Any],
        *,
        prompt: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        effective_prompt = prompt if prompt is not None else self.build_prompt(state, legal_actions)
        action, meta = self.inner.choose(state, legal_actions, prompt=effective_prompt)
        meta = dict(meta)
        hint = build_card_choice_policy_hint(state, legal_actions, prompt_injection=self.prompt_injection)
        meta["prompt_injection_variant"] = self.prompt_injection
        meta["prompt_injection_chars"] = len(hint)
        return action, meta

    def reset_episode(self) -> None:
        reset = getattr(self.inner, "reset_episode", None)
        if callable(reset):
            reset()

    def receive_external_feedback(
        self,
        state: dict[str, Any],
        action: Any,
        meta: dict[str, Any],
        feedback: str,
        legal_actions: list[Any],
    ) -> None:
        receive = getattr(self.inner, "receive_external_feedback", None)
        if callable(receive):
            receive(state, action, meta, feedback, legal_actions)

    def record_transition(
        self,
        state: dict[str, Any],
        action: Any,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        record = getattr(self.inner, "record_transition", None)
        if callable(record):
            record(state, action, meta, next_state)


def wrap_card_choice_prompt_agent(
    agent: Agent,
    *,
    prompt_injection: PromptInjectionMode = "none",
) -> Agent:
    if prompt_injection == "none":
        return agent
    if prompt_injection not in {"generic", "scenario-specific"}:
        raise ValueError("prompt_injection must be none, generic, or scenario-specific")
    return PromptInjectedCardChoiceAgent(agent, prompt_injection=prompt_injection)


def build_card_choice_policy_hint(
    state: dict[str, Any],
    legal_actions: list[Any],
    *,
    prompt_injection: PromptInjectionMode = "none",
) -> str:
    if prompt_injection == "none":
        return ""
    if prompt_injection == "generic":
        return GENERIC_CARD_CHOICE_POLICY_HINT
    if prompt_injection != "scenario-specific":
        raise ValueError("prompt_injection must be none, generic, or scenario-specific")

    return "\n".join(
        [
            "Card choice policy hints (scenario-specific):",
            "Use only the visible current state, visible deck summary, visible offered cards, and current legal actions.",
            "Do not infer that taking a card is mandatory; if skip card reward is legal, compare it directly against each pick.",
            "General checks:",
            "- Evaluate marginal utility, dilution cost, duplicate count, curve pressure, role balance, and visible synergy support.",
            "- Prefer support only when the support cards or payoff cards are already visible in the current deck.",
            "Visible state checks:",
            *_visible_card_choice_checks(state, legal_actions),
        ]
    )



def load_card_choice_bias_scenarios() -> list[CardChoiceScenario]:
    return list(_build_bias_scenarios())


def choose_bias_scenarios(
    scenarios: Iterable[CardChoiceScenario] | None = None,
    *,
    scenario_ids: list[str] | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[CardChoiceScenario]:
    selected = list(scenarios if scenarios is not None else load_card_choice_bias_scenarios())
    if scenario_ids:
        wanted = set(scenario_ids)
        selected = [scenario for scenario in selected if scenario.id in wanted]
        missing = wanted - {scenario.id for scenario in selected}
        if missing:
            raise ValueError(f"Unknown scenario ids: {', '.join(sorted(missing))}")
    if categories:
        wanted_categories = set(categories)
        unknown = wanted_categories - CARD_CHOICE_BIAS_CATEGORIES
        if unknown:
            raise ValueError(f"Unknown categories: {', '.join(sorted(unknown))}")
        selected = [scenario for scenario in selected if scenario.category in wanted_categories]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected = selected[:limit]
    return selected


def run_card_choice_bias_eval(
    *,
    agent: Agent,
    scenarios: Iterable[CardChoiceScenario] | None = None,
    out: Path | str | None = None,
    log_level: EvalLogLevel = "decisions",
) -> dict[str, Any]:
    rows = list(scenarios if scenarios is not None else load_card_choice_bias_scenarios())
    scenario_by_id = {scenario.id: scenario for scenario in rows}
    output_path = Path(out) if out else None
    temp_path: Path | None = None
    if output_path is None:
        fd, raw_path = tempfile.mkstemp(prefix="card-choice-bias-", suffix=".jsonl")
        os.close(fd)
        temp_path = Path(raw_path)
        output_path = temp_path

    base_summary = run_card_choice_eval(
        agent=agent,
        scenarios=rows,
        out=output_path,
        log_level=log_level,
    )
    jsonl_rows = _read_jsonl(output_path)
    choice_rows = [row for row in jsonl_rows if row.get("type") == "choice"]
    bias_summary = summarize_card_choice_bias_results(choice_rows, scenario_by_id)
    if out:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "bias_summary", **bias_summary}, ensure_ascii=False) + "\n")
    if temp_path:
        temp_path.unlink(missing_ok=True)
    return {**base_summary, "bias_summary": bias_summary}


def summarize_card_choice_bias_results(
    choice_rows: list[dict[str, Any]],
    scenarios_by_id: dict[str, CardChoiceScenario],
) -> dict[str, Any]:
    total = len(choice_rows)
    attack_offer_count = 0
    attack_selected_count = 0
    attack_trap_count = 0
    attack_trap_attack_selected = 0
    defensive_answer_missed = 0
    skip_expected_count = 0
    skip_selected_count = 0
    skip_correct_count = 0
    redundant_offer_count = 0
    redundant_selected_count = 0
    rare_attack_trap_count = 0
    rare_attack_selected_count = 0
    energy_expected_count = 0
    energy_selected_count = 0
    high_curve_damage_selected_count = 0
    havoc_offer_count = 0
    havoc_selected_count = 0
    supported_havoc_selected_count = 0
    unsupported_havoc_selected_count = 0
    by_redundancy_count: dict[str, dict[str, Any]] = {}
    category_counts: dict[str, dict[str, Any]] = {}

    for row in choice_rows:
        scenario = scenarios_by_id.get(str(row.get("scenario_id")))
        if scenario is None:
            continue
        tags = set(scenario.tags)
        selected_name = _selected_card_name(row)
        selected_card = _offer_card_by_name(scenario, selected_name)
        selected_type = selected_card.get("type") if selected_card else None
        selected_is_attack = selected_type == "Attack"
        selected_is_skip = row.get("selected_kind") == "card_reward_skip"
        selected_is_energy = _card_has_energy(selected_card)
        category_bucket = category_counts.setdefault(
            scenario.category,
            {"total": 0, "correct": 0, "accuracy": 0.0},
        )
        category_bucket["total"] += 1
        if row.get("verdict") == "correct":
            category_bucket["correct"] += 1

        if _has_offer_type(scenario, "Attack"):
            attack_offer_count += 1
        if selected_is_attack:
            attack_selected_count += 1

        if "attack_trap" in tags:
            attack_trap_count += 1
            if selected_is_attack:
                attack_trap_attack_selected += 1
                if row.get("verdict") != "correct":
                    defensive_answer_missed += 1

        if "expected_skip" in tags or scenario.oracle.acceptable_labels == ["skip card reward"]:
            skip_expected_count += 1
            if selected_is_skip and row.get("verdict") == "correct":
                skip_correct_count += 1
        if selected_is_skip:
            skip_selected_count += 1

        redundancy_count = _int_tag(tags, "redundancy_count:")
        redundant_offer = _str_tag(tags, "redundant_offer:")
        if redundancy_count is not None:
            bucket = by_redundancy_count.setdefault(
                str(redundancy_count),
                {"total": 0, "selected": 0, "selection_rate": 0.0},
            )
            bucket["total"] += 1
            if redundant_offer:
                redundant_offer_count += 1
                if selected_name == redundant_offer:
                    redundant_selected_count += 1
                    bucket["selected"] += 1

        if "rare_attack_trap" in tags:
            rare_attack_trap_count += 1
            if selected_is_attack:
                rare_attack_selected_count += 1

        if "expected_energy" in tags:
            energy_expected_count += 1
            if selected_is_energy:
                energy_selected_count += 1
            if "high_curve" in tags and selected_is_attack:
                high_curve_damage_selected_count += 1

        if _offer_card_by_name(scenario, "Havoc"):
            havoc_offer_count += 1
            if selected_name == "Havoc":
                havoc_selected_count += 1
                if "havoc_supported" in tags:
                    supported_havoc_selected_count += 1
                if "havoc_unsupported" in tags:
                    unsupported_havoc_selected_count += 1

    for bucket in by_redundancy_count.values():
        bucket["selection_rate"] = _safe_ratio(bucket["selected"], bucket["total"])
    for bucket in category_counts.values():
        bucket["accuracy"] = _safe_ratio(bucket["correct"], bucket["total"])

    return {
        "total": total,
        "category_accuracy": category_counts,
        "attack_bias": {
            "attack_offer_count": attack_offer_count,
            "attack_selected_count": attack_selected_count,
            "attack_selection_rate": _safe_ratio(attack_selected_count, attack_offer_count),
            "attack_trap_count": attack_trap_count,
            "attack_trap_attack_selected_count": attack_trap_attack_selected,
            "attack_trap_attack_selection_rate": _safe_ratio(attack_trap_attack_selected, attack_trap_count),
            "defensive_answer_missed_count": defensive_answer_missed,
        },
        "skip_bias": {
            "skip_expected_count": skip_expected_count,
            "skip_selected_count": skip_selected_count,
            "skip_correct_count": skip_correct_count,
            "skip_miss_count": skip_expected_count - skip_correct_count,
            "skip_precision": _safe_ratio(skip_correct_count, skip_selected_count),
            "skip_recall": _safe_ratio(skip_correct_count, skip_expected_count),
        },
        "redundancy_bias": {
            "redundant_offer_count": redundant_offer_count,
            "redundant_selected_count": redundant_selected_count,
            "redundant_selection_rate": _safe_ratio(redundant_selected_count, redundant_offer_count),
            "by_redundancy_count": dict(sorted(by_redundancy_count.items())),
        },
        "rare_damage_bias": {
            "rare_attack_trap_count": rare_attack_trap_count,
            "rare_attack_selected_count": rare_attack_selected_count,
            "rare_attack_selection_rate": _safe_ratio(rare_attack_selected_count, rare_attack_trap_count),
        },
        "energy_curve_bias": {
            "energy_expected_count": energy_expected_count,
            "energy_selected_count": energy_selected_count,
            "energy_selection_rate": _safe_ratio(energy_selected_count, energy_expected_count),
            "high_curve_damage_selected_count": high_curve_damage_selected_count,
        },
        "havoc_bias": {
            "havoc_offer_count": havoc_offer_count,
            "havoc_selected_count": havoc_selected_count,
            "havoc_selection_rate": _safe_ratio(havoc_selected_count, havoc_offer_count),
            "supported_havoc_selected_count": supported_havoc_selected_count,
            "unsupported_havoc_selected_count": unsupported_havoc_selected_count,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic card reward bias probes.")
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
    parser.add_argument("--prompt-injection", choices=["none", "generic", "scenario-specific"], default="none")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    scenarios = choose_bias_scenarios(
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
    agent = wrap_card_choice_prompt_agent(agent, prompt_injection=args.prompt_injection)
    summary = run_card_choice_bias_eval(
        agent=agent,
        scenarios=scenarios,
        out=args.out,
        log_level=args.log_level,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _build_bias_scenarios() -> list[CardChoiceScenario]:
    specs = [
        _spec(
            "bias_at_001",
            "attack_temptation_ladder",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Twin Strike", "Pommel Strike"],
            offers=["Twin Strike", "Shrug It Off", "Havoc"],
            acceptable=["Shrug It Off"],
            hp=50,
            last_loss=16,
            tags=["bias_axis:attack_overpick", "attack_trap", "attack_lure:mild"],
            notes="Mild attack lure against a block-and-draw answer after a moderately costly fight.",
        ),
        _spec(
            "bias_at_002",
            "attack_temptation_ladder",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Anger", "Pommel Strike", "Blood Wall"],
            offers=["Anger", "Stone Armor", "Havoc"],
            acceptable=["Stone Armor"],
            hp=47,
            last_loss=18,
            tags=["bias_axis:attack_overpick", "attack_trap", "attack_lure:zero_cost"],
            notes="Zero-cost damage is tempting, but the deck wants durable defensive texture.",
        ),
        _spec(
            "bias_at_003",
            "attack_temptation_ladder",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Twin Strike", "True Grit"],
            offers=["Pommel Strike", "Shrug It Off", "Cinder"],
            acceptable=["Shrug It Off"],
            hp=45,
            last_loss=20,
            tags=["bias_axis:attack_overpick", "attack_trap", "attack_lure:cantrip_damage"],
            notes="Cantrip attack competes with a better stabilization card.",
        ),
        _spec(
            "bias_at_004",
            "attack_temptation_ladder",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Twin Strike", "Pommel Strike"],
            offers=["Perfected Strike", "Flame Barrier", "Havoc"],
            acceptable=["Flame Barrier"],
            hp=42,
            last_loss=22,
            tags=["bias_axis:attack_overpick", "attack_trap", "attack_lure:synergy_damage"],
            notes="Strike synergy is real, but recent HP loss and weak block quality make Flame Barrier the target.",
        ),
        _spec(
            "bias_at_005",
            "attack_temptation_ladder",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Blood Wall", "Pommel Strike", "Anger"],
            offers=["Cinder", "Shrug It Off", "Bloodletting"],
            acceptable=["Shrug It Off"],
            hp=43,
            last_loss=21,
            tags=["bias_axis:attack_overpick", "attack_trap", "attack_lure:exhaust_damage"],
            notes="Cinder's damage and exhaust text should not crowd out plain block and draw.",
        ),
        _spec(
            "bias_at_006",
            "attack_temptation_ladder",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Twin Strike", "True Grit", "Armaments"],
            offers=["Bludgeon", "Stone Armor", "Pommel Strike"],
            acceptable=["Stone Armor"],
            hp=41,
            last_loss=24,
            tags=["bias_axis:attack_overpick", "attack_trap", "attack_lure:rare_damage"],
            notes="A rare attack is offered, but another stable defensive card better fits the evidence.",
        ),
        _spec(
            "bias_at_007",
            "attack_temptation_ladder",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Body Slam", "Twin Strike", "Blood Wall"],
            offers=["Twin Strike", "Flame Barrier", "Cinder"],
            acceptable=["Flame Barrier"],
            hp=44,
            last_loss=19,
            tags=["bias_axis:attack_overpick", "attack_trap", "attack_lure:duplicate_attack"],
            notes="Body Slam gives block payoff, so high block is more valuable than another attack.",
        ),
        _spec(
            "bias_at_008",
            "attack_temptation_ladder",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Pommel Strike", "Anger", "Bloodletting"],
            offers=["Pommel Strike", "Armaments", "Havoc"],
            acceptable=["Armaments"],
            hp=46,
            last_loss=17,
            tags=["bias_axis:attack_overpick", "attack_trap", "attack_lure:second_cantrip"],
            notes="The deck has many basics and can use upgrades more than another attack cantrip.",
        ),
        _spec(
            "bias_sk_001",
            "skip_threshold_matrix",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Shrug It Off", "Stone Armor", "Pommel Strike", "Bloodletting"],
            offers=["Strike", "Defend", "Havoc"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "deck_quality:medium", "offer_quality:low"],
            notes="Medium deck with basics and unreliable Havoc in the offer.",
        ),
        _spec(
            "bias_sk_002",
            "skip_threshold_matrix",
            deck=["Strike"] * 5 + ["Defend"] * 5 + ["Shrug It Off"] * 2 + ["Stone Armor"] * 2 + ["Pommel Strike"] * 2 + ["Bloodletting", "Offering", "Bash", "Bludgeon"],
            offers=["Strike", "Defend", "Havoc"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "deck_size:large", "offer_quality:low"],
            notes="Larger high-quality deck with no offered card above the bar.",
        ),
        _spec(
            "bias_sk_003",
            "skip_threshold_matrix",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Barricade", "Impervious", "Blood Wall", "True Grit", "Armaments", "Bash"],
            offers=["Strike", "Havoc", "Cinder"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "plan:block", "offer_quality:off_plan"],
            notes="Clear block-retention plan with off-plan attack offers.",
        ),
        _spec(
            "bias_sk_004",
            "skip_threshold_matrix",
            deck=["Strike"] * 5 + ["Defend"] * 5 + ["Shrug It Off"] * 2 + ["Blood Wall", "Armaments", "Pommel Strike", "Bash"],
            offers=["Defend", "Twin Strike", "Anger"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "deck_quality:solid", "offer_quality:pollution"],
            notes="Solid deck where every offered card duplicates a weak or crowded role.",
        ),
        _spec(
            "bias_sk_005",
            "skip_threshold_matrix",
            deck=["Strike"] * 5 + ["Defend"] * 5 + ["Shrug It Off", "Stone Armor", "Bloodletting", "Offering", "Pommel Strike", "Bash", "Armaments"],
            offers=["Bloodletting", "Strike", "Defend"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "hp_cost_duplicate", "offer_quality:low"],
            notes="The deck already has HP-for-energy, so the duplicate is not worth more HP pressure.",
        ),
        _spec(
            "bias_sk_006",
            "skip_threshold_matrix",
            deck=["Strike"] * 4 + ["Defend"] * 5 + ["Shrug It Off"] * 2 + ["Flame Barrier", "Blood Wall", "Barricade", "Impervious", "Bash"],
            offers=["Stone Armor", "Defend", "Havoc"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "plan:block", "offer_quality:redundant"],
            notes="Block plan is already saturated; additional mediocre block dilutes key cards.",
        ),
        _spec(
            "bias_sk_007",
            "skip_threshold_matrix",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Shrug It Off", "Stone Armor", "Pommel Strike", "Armaments", "Bash", "Bludgeon"],
            offers=["Twin Strike", "Cinder", "Havoc"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "deck_quality:focused", "offer_quality:attack_noise"],
            notes="Focused deck does not need another attack-only card.",
        ),
        _spec(
            "bias_sk_008",
            "skip_threshold_matrix",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Pommel Strike"] * 2 + ["Shrug It Off"] * 2 + ["Offering", "Bloodletting", "Armaments", "Bash"],
            offers=["Pommel Strike", "Strike", "Havoc"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "deck_quality:draw_ready", "offer_quality:redundant"],
            notes="Draw and energy are already present; another Pommel Strike is not needed.",
        ),
        _spec(
            "bias_sk_009",
            "skip_threshold_matrix",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Feel No Pain", "Dark Embrace", "True Grit", "Second Wind", "Burning Pact", "Bash"],
            offers=["Havoc", "Strike", "Defend"],
            acceptable=["Skip"],
            tags=["bias_axis:skip_aversion", "expected_skip", "plan:exhaust", "offer_quality:unreliable"],
            notes="Even an exhaust deck can skip an unreliable Havoc when the support package is already strong.",
        ),
        _spec(
            "bias_rd_001",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Pommel Strike"],
            offers=["Shrug It Off", "Twin Strike", "Havoc"],
            acceptable=["Shrug It Off"],
            tags=["bias_axis:redundancy", "redundancy_count:0", "redundant_offer:Shrug It Off"],
            notes="No existing Shrug It Off, so taking the first premium block draw is acceptable.",
        ),
        _spec(
            "bias_rd_002",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Shrug It Off", "Twin Strike"],
            offers=["Shrug It Off", "Stone Armor", "Havoc"],
            acceptable=["Shrug It Off", "Stone Armor"],
            tags=["bias_axis:redundancy", "redundancy_count:1", "redundant_offer:Shrug It Off"],
            notes="One existing copy still leaves another high-quality defense defensible.",
        ),
        _spec(
            "bias_rd_003",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Shrug It Off"] * 2 + ["Stone Armor"],
            offers=["Shrug It Off", "Pommel Strike", "Havoc"],
            acceptable=["Pommel Strike"],
            tags=["bias_axis:redundancy", "redundancy_count:2", "redundant_offer:Shrug It Off"],
            notes="At two copies, the deck can value attack-cycle more than another duplicate block cantrip.",
        ),
        _spec(
            "bias_rd_004",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"] + ["Shrug It Off"] * 3 + ["Stone Armor"],
            offers=["Shrug It Off", "Twin Strike", "Havoc"],
            acceptable=["Skip"],
            tags=["bias_axis:redundancy", "redundancy_count:3", "redundant_offer:Shrug It Off", "expected_skip"],
            notes="Three copies make another copy too redundant for a stable deck.",
        ),
        _spec(
            "bias_rd_005",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"] + ["Shrug It Off"] * 4 + ["Stone Armor"],
            offers=["Shrug It Off", "Pommel Strike", "Defend"],
            acceptable=["Skip"],
            tags=["bias_axis:redundancy", "redundancy_count:4", "redundant_offer:Shrug It Off", "expected_skip"],
            notes="Four copies should strongly push away from another duplicate.",
        ),
        _spec(
            "bias_rd_006",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Bloodletting", "Bludgeon"],
            offers=["Bloodletting", "Stone Armor", "Havoc"],
            acceptable=["Stone Armor"],
            tags=["bias_axis:redundancy", "redundancy_count:1", "redundant_offer:Bloodletting", "hp_cost_duplicate"],
            notes="One HP-for-energy card is enough for this curve; take the defensive card.",
        ),
        _spec(
            "bias_rd_007",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Bloodletting"] * 2 + ["Offering", "Pommel Strike"],
            offers=["Bloodletting", "Anger", "Havoc"],
            acceptable=["Skip"],
            tags=["bias_axis:redundancy", "redundancy_count:2", "redundant_offer:Bloodletting", "expected_skip"],
            notes="Multiple HP-for-energy effects raise the bar for taking another one.",
        ),
        _spec(
            "bias_rd_008",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"] + ["Pommel Strike"] * 2 + ["Shrug It Off"],
            offers=["Pommel Strike", "Stone Armor", "Cinder"],
            acceptable=["Stone Armor"],
            tags=["bias_axis:redundancy", "redundancy_count:2", "redundant_offer:Pommel Strike"],
            notes="Two attack cantrips make a defensive texture card better.",
        ),
        _spec(
            "bias_rd_009",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"] + ["Pommel Strike"] * 3 + ["Shrug It Off", "Bloodletting"],
            offers=["Pommel Strike", "True Grit", "Strike"],
            acceptable=["True Grit"],
            tags=["bias_axis:redundancy", "redundancy_count:3", "redundant_offer:Pommel Strike"],
            notes="Three Pommel Strikes make another copy less valuable than defensive utility.",
        ),
        _spec(
            "bias_rd_010",
            "redundancy_dose_response",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"] + ["Twin Strike"] * 4 + ["Pommel Strike", "Shrug It Off"],
            offers=["Twin Strike", "Strike", "Havoc"],
            acceptable=["Skip"],
            tags=["bias_axis:redundancy", "redundancy_count:4", "redundant_offer:Twin Strike", "expected_skip"],
            notes="Attack redundancy is the entire point of the probe.",
        ),
        _spec(
            "bias_rt_001",
            "rare_damage_trap",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Pommel Strike", "Blood Wall", "True Grit"],
            offers=["Bludgeon", "Shrug It Off", "Cinder"],
            acceptable=["Shrug It Off"],
            hp=44,
            last_loss=20,
            tags=["bias_axis:rare_damage_trap", "attack_trap", "rare_attack_trap"],
            notes="Rare damage competes with premium block draw after a costly fight.",
        ),
        _spec(
            "bias_rt_002",
            "rare_damage_trap",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Twin Strike", "Armaments", "Blood Wall"],
            offers=["Bludgeon", "Stone Armor", "Anger"],
            acceptable=["Stone Armor"],
            hp=43,
            last_loss=19,
            tags=["bias_axis:rare_damage_trap", "attack_trap", "rare_attack_trap"],
            notes="The expensive attack is strong, but defensive scaling is the missing role.",
        ),
        _spec(
            "bias_rt_003",
            "rare_damage_trap",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Pommel Strike", "Twin Strike", "Barricade", "Impervious", "Blood Wall", "True Grit"],
            offers=["Bludgeon", "Armaments", "Bloodletting"],
            acceptable=["Armaments"],
            tags=["bias_axis:rare_damage_trap", "attack_trap", "rare_attack_trap", "upgrade_support"],
            notes="Upgrade support improves several plan cards more than another rare attack does.",
        ),
        _spec(
            "bias_rt_004",
            "rare_damage_trap",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Bludgeon", "Barricade", "Impervious"],
            offers=["Bludgeon", "Bloodletting", "Shrug It Off"],
            acceptable=["Bloodletting"],
            tags=["bias_axis:rare_damage_trap", "attack_trap", "rare_attack_trap", "expected_energy", "high_curve"],
            notes="A second Bludgeon is less important than energy for the existing expensive cards.",
        ),
        _spec(
            "bias_rt_005",
            "rare_damage_trap",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Body Slam", "Entrench", "Blood Wall", "True Grit"],
            offers=["Bludgeon", "Flame Barrier", "Cinder"],
            acceptable=["Flame Barrier"],
            hp=45,
            last_loss=18,
            tags=["bias_axis:rare_damage_trap", "attack_trap", "rare_attack_trap", "block_payoff"],
            notes="Block payoff already exists; the rare attack is off-plan.",
        ),
        _spec(
            "bias_rt_006",
            "rare_damage_trap",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Pommel Strike", "Shrug It Off", "Bloodletting"],
            offers=["Bludgeon", "True Grit", "Anger"],
            acceptable=["True Grit"],
            hp=41,
            last_loss=23,
            tags=["bias_axis:rare_damage_trap", "attack_trap", "rare_attack_trap"],
            notes="Recent damage and crowded attacks make defensive utility the better answer.",
        ),
        _spec(
            "bias_ec_001",
            "energy_vs_damage_curve",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Bludgeon", "Barricade", "Impervious"],
            offers=["Bloodletting", "Bludgeon", "Twin Strike"],
            acceptable=["Bloodletting"],
            tags=["bias_axis:energy_curve", "expected_energy", "high_curve"],
            notes="Expensive curve needs energy more than another attack.",
        ),
        _spec(
            "bias_ec_002",
            "energy_vs_damage_curve",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Bludgeon", "Impervious", "Flame Barrier"],
            offers=["Offering", "Pommel Strike", "Cinder"],
            acceptable=["Offering"],
            tags=["bias_axis:energy_curve", "expected_energy", "high_curve"],
            notes="High curve plus draw makes Offering valuable despite HP cost.",
        ),
        _spec(
            "bias_ec_003",
            "energy_vs_damage_curve",
            deck=["Strike"] * 5 + ["Defend"] * 5 + ["Anger"] * 2 + ["Pommel Strike"],
            offers=["Bludgeon", "Bloodletting", "Havoc"],
            acceptable=["Bludgeon"],
            tags=["bias_axis:energy_curve", "low_curve", "needs_payoff"],
            notes="Low-cost deck can take a large payoff instead of more energy.",
        ),
        _spec(
            "bias_ec_004",
            "energy_vs_damage_curve",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Bludgeon", "Barricade", "Impervious", "Shrug It Off"],
            offers=["Bloodletting", "Cinder", "Stone Armor"],
            acceptable=["Bloodletting"],
            tags=["bias_axis:energy_curve", "expected_energy", "high_curve"],
            notes="Energy patch is the clean answer for a costly deck.",
        ),
        _spec(
            "bias_ec_005",
            "energy_vs_damage_curve",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Offering", "Bloodletting", "Bludgeon"],
            offers=["Bloodletting", "Shrug It Off", "Pommel Strike"],
            acceptable=["Shrug It Off"],
            hp=39,
            last_loss=20,
            tags=["bias_axis:energy_curve", "hp_cost_saturation"],
            notes="Energy is already covered and HP has been pressured, so block draw is better.",
        ),
        _spec(
            "bias_hv_001",
            "havoc_semantic_trap",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Bash", "Bludgeon", "Barricade", "Impervious"],
            offers=["Havoc", "Shrug It Off", "Twin Strike"],
            acceptable=["Shrug It Off"],
            tags=["bias_axis:havoc_semantics", "havoc_unsupported"],
            notes="Havoc can randomly consume expensive payoff cards without enough exhaust support.",
        ),
        _spec(
            "bias_hv_002",
            "havoc_semantic_trap",
            deck=["Strike"] * 4 + ["Defend"] * 4 + ["Feel No Pain", "Dark Embrace", "True Grit", "Second Wind", "Burning Pact", "Bash"],
            offers=["Havoc", "Cinder", "Stone Armor"],
            acceptable=["Havoc", "Stone Armor"],
            tags=["bias_axis:havoc_semantics", "havoc_supported"],
            notes="Exhaust support makes Havoc defensible rather than automatically bad.",
        ),
        _spec(
            "bias_hv_003",
            "havoc_semantic_trap",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash", "Pommel Strike", "Bloodletting", "Armaments"],
            offers=["Havoc", "Shrug It Off", "Anger"],
            acceptable=["Shrug It Off"],
            hp=45,
            last_loss=18,
            tags=["bias_axis:havoc_semantics", "havoc_unsupported"],
            notes="No exhaust payoff and recent damage make the reliable defensive card superior.",
        ),
        _spec(
            "bias_hv_004",
            "havoc_semantic_trap",
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Feel No Pain", "True Grit", "Second Wind", "Cinder", "Bash"],
            offers=["Havoc", "Burning Pact", "Stone Armor"],
            acceptable=["Burning Pact", "Stone Armor"],
            tags=["bias_axis:havoc_semantics", "havoc_mixed_support"],
            notes="Partial exhaust support exists, but controlled exhaust or stable defense is better than random top-deck play.",
        ),
    ]
    _assert_expected_counts(specs)
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
        "floor": 7,
        "context": {"act": 1, "floor": 7, "room_type": "Monster"},
        "player": {
            "hp": spec["hp"],
            "max_hp": 80,
            "gold": 90,
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


def _assert_expected_counts(specs: list[dict[str, Any]]) -> None:
    counts = Counter(spec["category"] for spec in specs)
    if set(counts) != CARD_CHOICE_BIAS_CATEGORIES:
        raise AssertionError(f"Unexpected bias categories: {dict(counts)}")
    if dict(counts) != EXPECTED_BIAS_CATEGORY_COUNTS:
        raise AssertionError(f"Unexpected bias category counts: {dict(counts)}")


def _label_for_offer(name: str, offer_cards: list[dict[str, Any]]) -> str:
    if name == "Skip":
        return "skip card reward"
    for card in offer_cards:
        if card["name"] == name:
            return f"pick card {card['index']}: {name}"
    raise ValueError(f"Offer {name!r} is not present")


def _rationale_keywords(category: str) -> list[str]:
    return {
        "attack_temptation_ladder": ["attack", "block", "hp"],
        "skip_threshold_matrix": ["skip", "deck", "quality"],
        "redundancy_dose_response": ["duplicate", "redundant", "count"],
        "rare_damage_trap": ["rare", "damage", "support"],
        "energy_vs_damage_curve": ["energy", "cost", "curve"],
        "havoc_semantic_trap": ["havoc", "exhaust", "top"],
    }.get(category, [])


def _visible_card_choice_checks(state: dict[str, Any], legal_actions: list[Any]) -> list[str]:
    player = state.get("player") or {}
    deck = [card for card in player.get("deck") or [] if isinstance(card, dict)]
    offers = [card for card in state.get("cards") or [] if isinstance(card, dict)]
    deck_size = player.get("deck_size")
    if deck_size is None:
        deck_size = len(deck)

    deck_names = Counter(str(card.get("name") or "?") for card in deck)
    attack_count = sum(1 for card in deck if card.get("type") == "Attack")
    skill_count = sum(1 for card in deck if card.get("type") == "Skill")
    power_count = sum(1 for card in deck if card.get("type") == "Power")
    block_source_count = sum(1 for card in deck if _card_block_value(card) > 0)
    energy_card_count = sum(1 for card in deck if _card_energy_value(card) > 0)
    hp_cost_count = sum(1 for card in deck if _card_hp_loss_value(card) > 0)
    high_cost_count = sum(1 for card in deck if _card_cost_value(card) is not None and _card_cost_value(card) >= 2)
    skip_visible = any("skip" in str(action.kind).lower() or "skip" in str(action.label).lower() for action in legal_actions)

    lines = [
        (
            f"- deck_size={deck_size}; visible deck type counts: attacks={attack_count}, "
            f"skills={skill_count}, powers={power_count}, block_sources={block_source_count}, "
            f"energy_cards={energy_card_count}, hp_cost_cards={hp_cost_count}, high_cost_cards={high_cost_count}."
        )
    ]
    if skip_visible:
        lines.append("- skip card reward is currently legal; include its zero-dilution value in the comparison.")
    if offers:
        lines.append("- offered cards: " + "; ".join(_offer_summary(card, deck_names) for card in offers) + ".")
    lines.append("- duplicate offer counts: " + _duplicate_offer_text(offers, deck_names) + ".")

    if isinstance(deck_size, int) and deck_size >= 16:
        lines.append("- large deck check: adding a card must beat the cost of diluting the current best cards.")
    if attack_count > block_source_count:
        lines.append("- role balance check: attacks currently outnumber visible block/plating sources.")
    if high_cost_count >= 3:
        lines.append("- curve check: the visible deck already has several cost-2-or-more cards; consider energy and draw before adding more cost.")
    if energy_card_count >= 2:
        lines.append("- energy saturation check: the visible deck already has multiple energy sources; compare more energy against HP cost and current needs.")
    if any(_card_hp_loss_value(card) > 0 for card in offers):
        lines.append("- HP-cost offer check: HP loss is a real cost, especially after recent combat damage.")
    if _offer_named(offers, "Havoc"):
        lines.append("- Havoc check: verify controlled exhaust/top-deck support before treating random top-card play as a benefit.")
    if any(card.get("rarity") == "Rare" and card.get("type") == "Attack" for card in offers):
        lines.append("- rare attack check: rarity and burst damage do not remove the need to compare deck fit, curve, and survival.")
    return lines


def _offer_summary(card: dict[str, Any], deck_names: Counter[str]) -> str:
    name = str(card.get("name") or "?")
    cost = card.get("cost", "?")
    stats = []
    if _card_block_value(card) > 0:
        stats.append(f"block={_card_block_value(card)}")
    if _card_energy_value(card) > 0:
        stats.append(f"energy={_card_energy_value(card)}")
    if _card_hp_loss_value(card) > 0:
        stats.append(f"hp_loss={_card_hp_loss_value(card)}")
    if card.get("stats", {}).get("cards"):
        stats.append(f"draw={card.get('stats', {}).get('cards')}")
    stats_text = f" stats={','.join(stats)}" if stats else ""
    return f"{name} type={card.get('type', '?')} cost={cost} deck_copies={deck_names.get(name, 0)}{stats_text}"


def _duplicate_offer_text(offers: list[dict[str, Any]], deck_names: Counter[str]) -> str:
    parts = [f"{card.get('name', '?')}={deck_names.get(str(card.get('name') or '?'), 0)}" for card in offers]
    return ", ".join(parts) if parts else "no visible offers"


def _offer_named(offers: list[dict[str, Any]], name: str) -> bool:
    return any(card.get("name") == name for card in offers)


def _card_cost_value(card: dict[str, Any]) -> int | None:
    value = card.get("cost")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _card_block_value(card: dict[str, Any]) -> int:
    stats = card.get("stats") or {}
    return _int_value(stats.get("block")) + _int_value(stats.get("plating"))


def _card_energy_value(card: dict[str, Any]) -> int:
    stats = card.get("stats") or {}
    return _int_value(stats.get("energy"))


def _card_hp_loss_value(card: dict[str, Any]) -> int:
    stats = card.get("stats") or {}
    return _int_value(stats.get("hp_loss"))


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _card(name: str, *, index: int | None = None) -> dict[str, Any]:
    base = dict(_CARD_LIBRARY[name])
    if index is not None:
        base["index"] = index
    return base


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _selected_card_name(row: dict[str, Any]) -> str | None:
    label = row.get("selected_label")
    if not isinstance(label, str) or not label.startswith("pick card "):
        return None
    if ": " not in label:
        return None
    return label.split(": ", 1)[1]


def _offer_card_by_name(scenario: CardChoiceScenario, name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    for card in scenario.state.get("cards") or []:
        if card.get("name") == name:
            return card
    return None


def _has_offer_type(scenario: CardChoiceScenario, card_type: str) -> bool:
    return any(card.get("type") == card_type for card in scenario.state.get("cards") or [])


def _card_has_energy(card: dict[str, Any] | None) -> bool:
    if not card:
        return False
    stats = card.get("stats") or {}
    return bool(stats.get("energy"))


def _int_tag(tags: set[str], prefix: str) -> int | None:
    raw = _str_tag(tags, prefix)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _str_tag(tags: set[str], prefix: str) -> str | None:
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


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
    "Feel No Pain": {
        "id": "CARD.FEEL_NO_PAIN",
        "name": "Feel No Pain",
        "type": "Power",
        "cost": 1,
        "rarity": "Uncommon",
        "description": "Whenever a card is Exhausted, gain Block.",
    },
    "Dark Embrace": {
        "id": "CARD.DARK_EMBRACE",
        "name": "Dark Embrace",
        "type": "Power",
        "cost": 2,
        "rarity": "Uncommon",
        "description": "Whenever a card is Exhausted, draw a card.",
    },
    "Second Wind": {
        "id": "CARD.SECOND_WIND",
        "name": "Second Wind",
        "type": "Skill",
        "cost": 1,
        "rarity": "Uncommon",
        "description": "Exhaust all non-Attack cards in your hand. Gain Block for each card Exhausted.",
        "stats": {"block": 5, "exhaust": 1},
    },
    "Burning Pact": {
        "id": "CARD.BURNING_PACT",
        "name": "Burning Pact",
        "type": "Skill",
        "cost": 1,
        "rarity": "Uncommon",
        "description": "Exhaust 1 card. Draw cards.",
        "stats": {"cards": 2, "exhaust": 1},
    },
}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
