import json
from collections import Counter
from pathlib import Path

from sts2_bench.actions import build_legal_actions
from sts2_bench.card_choice_bias_eval import (
    CARD_CHOICE_BIAS_CATEGORIES,
    EXPECTED_BIAS_CATEGORY_COUNTS,
    build_card_choice_policy_hint,
    choose_bias_scenarios,
    load_card_choice_bias_scenarios,
    main,
    run_card_choice_bias_eval,
    summarize_card_choice_bias_results,
    wrap_card_choice_prompt_agent,
)
from sts2_bench.evidence import build_evidence_bundle


class _PickByLabelAgent:
    def __init__(self, labels_by_scenario):
        self.labels_by_scenario = dict(labels_by_scenario)
        self.prompts = []

    def build_prompt(self, state, legal_actions):
        prompt = f"bias_scenario={state.get('eval_scenario_id')}"
        self.prompts.append(prompt)
        return prompt

    def choose(self, state, legal_actions, *, prompt=None):
        scenario_id = state["eval_scenario_id"]
        label = self.labels_by_scenario[scenario_id]
        action = next(action for action in legal_actions if action.label == label)
        return action, {
            "agent": "pick_by_label",
            "parsed": {
                "action_id": action.action_id,
                "selected_action_label": action.label,
                "reason": f"scripted bias choice for {scenario_id}",
            },
            "prompt": prompt,
        }


def test_bias_suite_has_expected_categories_counts_and_shapes():
    scenarios = load_card_choice_bias_scenarios()
    ids = [scenario.id for scenario in scenarios]
    counts = Counter(scenario.category for scenario in scenarios)

    assert len(scenarios) == 42
    assert len(ids) == len(set(ids))
    assert set(counts) == CARD_CHOICE_BIAS_CATEGORIES
    assert dict(counts) == EXPECTED_BIAS_CATEGORY_COUNTS

    for scenario in scenarios:
        assert scenario.state["decision"] == "card_reward"
        assert scenario.state["can_skip"] is True
        assert scenario.state["player"]["deck"]
        assert "bias_axis:" in " ".join(scenario.tags)

        legal_actions = build_legal_actions(scenario.state)
        pick_actions = [action for action in legal_actions if action.kind == "card_reward"]
        skip_actions = [action for action in legal_actions if action.kind == "card_reward_skip"]
        labels = {action.label for action in legal_actions}
        bundle = build_evidence_bundle(scenario.state, legal_actions)

        assert len(pick_actions) == 3
        assert len(skip_actions) == 1
        assert set(scenario.oracle.acceptable_labels) <= labels
        assert set(scenario.oracle.bad_labels) <= labels
        assert bundle.deck_summary_visible is True
        assert bundle.skip_action_visible is True


def test_bias_suite_uses_current_localized_sts2_cards_not_known_sts1_placeholders():
    card_text = json.loads(Path("localization_eng/cards.json").read_text(encoding="utf-8"))
    localized_names = {value for key, value in card_text.items() if key.endswith(".title") and value}
    forbidden_names = {"Cleave", "Flex", "Heavy Blade"}
    scenario_names = set()

    for scenario in load_card_choice_bias_scenarios():
        scenario_names.update(card["name"] for card in scenario.state["cards"])
        scenario_names.update(card["name"] for card in scenario.state["player"]["deck"])

    assert sorted(scenario_names - localized_names) == []
    assert scenario_names.isdisjoint(forbidden_names)


def test_bias_suite_contains_targeted_probe_axes_without_obvious_degenerate_decks():
    scenarios = load_card_choice_bias_scenarios()
    expected_skip_count = 0
    attack_trap_count = 0
    havoc_supported_count = 0
    havoc_unsupported_count = 0
    redundancy_counts = set()

    for scenario in scenarios:
        deck = scenario.state["player"]["deck"]
        attack_count = sum(1 for card in deck if card.get("type") == "Attack")
        block_count = sum(1 for card in deck if card.get("stats", {}).get("block"))
        tags = set(scenario.tags)

        if scenario.category in {"attack_temptation_ladder", "rare_damage_trap"}:
            assert attack_count >= block_count, scenario.id
            assert block_count >= 4, scenario.id

        if "expected_skip" in tags:
            expected_skip_count += 1
            assert scenario.oracle.acceptable_labels == ["skip card reward"]

        if "attack_trap" in tags:
            attack_trap_count += 1
            accepted_names = _accepted_card_names(scenario)
            assert not _accepted_card_types(scenario) <= {"Attack"}
            assert accepted_names != {"Bludgeon"}

        if "havoc_supported" in tags:
            havoc_supported_count += 1
            assert "pick card 0: Havoc" in scenario.oracle.acceptable_labels

        if "havoc_unsupported" in tags:
            havoc_unsupported_count += 1
            assert "pick card 0: Havoc" in scenario.oracle.bad_labels

        for tag in tags:
            if tag.startswith("redundancy_count:"):
                redundancy_counts.add(int(tag.split(":", 1)[1]))

    assert expected_skip_count >= 12
    assert attack_trap_count >= 14
    assert havoc_supported_count >= 1
    assert havoc_unsupported_count >= 2
    assert redundancy_counts == {0, 1, 2, 3, 4}


def test_choose_bias_scenarios_filters_by_id_category_and_limit():
    scenarios = load_card_choice_bias_scenarios()
    target = next(scenario for scenario in scenarios if scenario.category == "skip_threshold_matrix")

    assert choose_bias_scenarios(scenarios, scenario_ids=[target.id]) == [target]
    assert all(
        scenario.category == "skip_threshold_matrix"
        for scenario in choose_bias_scenarios(scenarios, categories=["skip_threshold_matrix"])
    )
    assert len(choose_bias_scenarios(scenarios, limit=5)) == 5


def test_bias_summary_counts_attack_skip_and_redundancy_tendencies():
    scenarios = [
        next(scenario for scenario in load_card_choice_bias_scenarios() if "attack_trap" in scenario.tags),
        next(scenario for scenario in load_card_choice_bias_scenarios() if "expected_skip" in scenario.tags),
        next(scenario for scenario in load_card_choice_bias_scenarios() if "redundancy_count:4" in scenario.tags),
    ]
    attack_trap, skip_scenario, redundant_scenario = scenarios
    attack_label = _first_offer_label_of_type(attack_trap, "Attack")
    redundant_label = _redundant_offer_label(redundant_scenario)
    choice_rows = [
        _choice_row(attack_trap, attack_label, "card_reward", "incorrect"),
        _choice_row(skip_scenario, "skip card reward", "card_reward_skip", "correct"),
        _choice_row(redundant_scenario, redundant_label, "card_reward", "incorrect"),
    ]

    summary = summarize_card_choice_bias_results(choice_rows, {scenario.id: scenario for scenario in scenarios})

    assert summary["attack_bias"]["attack_trap_attack_selection_rate"] == 1.0
    assert summary["skip_bias"]["skip_recall"] == 0.5
    assert summary["skip_bias"]["skip_miss_count"] == 1
    assert summary["redundancy_bias"]["redundant_selection_rate"] == 1.0
    assert summary["redundancy_bias"]["by_redundancy_count"]["4"]["selected"] == 1


def test_run_card_choice_bias_eval_appends_bias_summary_and_respects_log_level(tmp_path):
    scenarios = [
        next(scenario for scenario in load_card_choice_bias_scenarios() if "attack_trap" in scenario.tags),
        next(scenario for scenario in load_card_choice_bias_scenarios() if "expected_skip" in scenario.tags),
    ]
    out = tmp_path / "bias.jsonl"
    agent = _PickByLabelAgent(
        {
            scenarios[0].id: scenarios[0].oracle.acceptable_labels[0],
            scenarios[1].id: "skip card reward",
        }
    )

    summary = run_card_choice_bias_eval(agent=agent, scenarios=scenarios, out=out, log_level="decisions")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenario_row = next(row for row in rows if row["type"] == "scenario")
    bias_summary_row = next(row for row in rows if row["type"] == "bias_summary")

    assert summary["bias_summary"]["total"] == 2
    assert bias_summary_row["total"] == 2
    assert "prompt" not in scenario_row
    assert rows[-1]["type"] == "bias_summary"


def test_generic_prompt_injection_enters_full_prompt_without_oracle_fields(tmp_path):
    scenario = next(scenario for scenario in load_card_choice_bias_scenarios() if "expected_skip" in scenario.tags)
    out = tmp_path / "generic-prompt.jsonl"
    base_agent = _PickByLabelAgent({scenario.id: scenario.oracle.acceptable_labels[0]})
    agent = wrap_card_choice_prompt_agent(base_agent, prompt_injection="generic")

    run_card_choice_bias_eval(agent=agent, scenarios=[scenario], out=out, log_level="full")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenario_row = next(row for row in rows if row["type"] == "scenario")
    choice_row = next(row for row in rows if row["type"] == "choice")
    prompt = scenario_row["prompt"]

    assert "Card choice policy hints (generic)" in prompt
    assert "marginal utility" in prompt
    assert "skip card reward" in prompt
    assert "oracle" not in prompt
    assert "acceptable_labels" not in prompt
    assert choice_row["agent_meta"]["prompt_injection_variant"] == "generic"


def test_scenario_specific_prompt_uses_visible_state_not_hidden_oracle(tmp_path):
    scenario = next(scenario for scenario in load_card_choice_bias_scenarios() if "redundancy_count:4" in scenario.tags)
    legal_actions = build_legal_actions(scenario.state)
    hint = build_card_choice_policy_hint(scenario.state, legal_actions, prompt_injection="scenario-specific")
    out = tmp_path / "specific-prompt.jsonl"
    base_agent = _PickByLabelAgent({scenario.id: scenario.oracle.acceptable_labels[0]})
    agent = wrap_card_choice_prompt_agent(base_agent, prompt_injection="scenario-specific")

    run_card_choice_bias_eval(agent=agent, scenarios=[scenario], out=out, log_level="decisions")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    choice_row = next(row for row in rows if row["type"] == "choice")

    assert "Card choice policy hints (scenario-specific)" in hint
    assert "deck_size=" in hint
    assert "duplicate offer counts" in hint
    assert "hidden oracle" not in hint.lower()
    assert "acceptable_labels" not in hint
    assert "Correct answer" not in hint
    assert choice_row["agent_meta"]["prompt_injection_variant"] == "scenario-specific"
    assert "Card choice policy hints (scenario-specific)" in choice_row["agent_meta"]["prompt"]


def test_card_choice_bias_eval_cli_runs_with_random_agent_and_filters(tmp_path):
    scenario = load_card_choice_bias_scenarios()[0]
    out = tmp_path / "cli-bias.jsonl"

    exit_code = main(["--agent", "random", "--scenario-id", scenario.id, "--out", str(out)])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert exit_code == 0
    assert rows[0]["type"] == "start"
    assert rows[0]["scenario_count"] == 1
    assert rows[1]["scenario_id"] == scenario.id
    assert rows[-1]["type"] == "bias_summary"


def _accepted_card_names(scenario):
    return {
        label.split(": ", 1)[1]
        for label in scenario.oracle.acceptable_labels
        if label.startswith("pick card ")
    }


def _accepted_card_types(scenario):
    accepted_names = _accepted_card_names(scenario)
    return {
        card["type"]
        for card in scenario.state["cards"]
        if card["name"] in accepted_names
    }


def _first_offer_label_of_type(scenario, card_type):
    card = next(card for card in scenario.state["cards"] if card["type"] == card_type)
    return f"pick card {card['index']}: {card['name']}"


def _redundant_offer_label(scenario):
    tag = next(tag for tag in scenario.tags if tag.startswith("redundant_offer:"))
    name = tag.split(":", 1)[1]
    card = next(card for card in scenario.state["cards"] if card["name"] == name)
    return f"pick card {card['index']}: {card['name']}"


def _choice_row(scenario, label, kind, verdict):
    return {
        "type": "choice",
        "scenario_id": scenario.id,
        "category": scenario.category,
        "selected_label": label,
        "selected_kind": kind,
        "verdict": verdict,
        "expected_behavior": scenario.oracle.expected_behavior,
        "failure_tags": [] if verdict == "correct" else ["bad_label"],
    }
