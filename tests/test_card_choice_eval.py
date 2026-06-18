import json
from pathlib import Path

from sts2_bench.actions import build_legal_actions
from sts2_bench.card_choice_eval import (
    CARD_CHOICE_CATEGORIES,
    _CARD_LIBRARY,
    choose_scenarios,
    load_card_choice_scenarios,
    main,
    run_card_choice_eval,
)
from sts2_bench.evidence import build_evidence_bundle


class _PickByLabelAgent:
    def __init__(self, labels_by_scenario):
        self.labels_by_scenario = dict(labels_by_scenario)
        self.prompts = []

    def build_prompt(self, state, legal_actions):
        prompt = f"scenario={state.get('eval_scenario_id')}"
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
                "reason": f"scripted choice for {scenario_id}",
            },
            "prompt": prompt,
        }


def test_scenarios_have_valid_card_reward_shape_and_oracles():
    scenarios = load_card_choice_scenarios()
    ids = [scenario.id for scenario in scenarios]

    assert len(scenarios) >= 24
    assert len(ids) == len(set(ids))
    assert CARD_CHOICE_CATEGORIES <= {scenario.category for scenario in scenarios}

    for scenario in scenarios:
        assert scenario.category
        assert scenario.state["decision"] == "card_reward"
        assert scenario.state["can_skip"] is True
        assert scenario.state["player"]["deck"]
        assert "notes_for_humans" in scenario.to_metadata_dict()

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


def test_scenario_cards_exist_in_local_sts2_card_text():
    card_text = json.loads(Path("localization_eng/cards.json").read_text(encoding="utf-8"))
    localized_names = {value for key, value in card_text.items() if key.endswith(".title") and value}
    scenario_names = set(_CARD_LIBRARY)
    for scenario in load_card_choice_scenarios():
        scenario_names.update(card["name"] for card in scenario.state["cards"])
        scenario_names.update(card["name"] for card in scenario.state["player"]["deck"])

    missing = sorted(name for name in scenario_names if name not in localized_names)

    assert missing == []


def test_key_scenario_shapes_are_not_degenerate():
    scenarios = load_card_choice_scenarios()

    for scenario in scenarios:
        deck = scenario.state["player"]["deck"]
        attack_count = sum(1 for card in deck if card.get("type") == "Attack")
        block_count = sum(1 for card in deck if card.get("stats", {}).get("block"))

        if scenario.category == "defense_gap_indirect":
            assert attack_count > block_count
            assert attack_count - block_count <= 2, scenario.id

        if scenario.category == "overweight_attack":
            assert attack_count > block_count
            assert block_count >= 4, scenario.id
            assert attack_count - block_count <= 4, scenario.id

        if scenario.category == "skip_is_best":
            deck_names = {card["name"] for card in deck}
            offer_names = {card["name"] for card in scenario.state["cards"]}
            assert deck_names & {"Shrug It Off", "Stone Armor", "Bloodletting", "Impervious", "Armaments"}, scenario.id
            assert not (offer_names & {"Anger", "Bloodletting", "Offering", "Shrug It Off"}), scenario.id


def test_contextual_payoff_cards_require_supporting_context():
    energy_names = {"Bloodletting", "Offering", "Adrenaline", "Fuel"}
    exhaust_support_names = {
        "Bloodletting",
        "Burning Pact",
        "Brand",
        "Cinder",
        "Havoc",
        "Scavenge",
        "Second Wind",
        "True Grit",
    }
    resource_curve_has_energy_pick = False

    for scenario in load_card_choice_scenarios():
        deck_names = {card["name"] for card in scenario.state["player"]["deck"]}
        accepted_names = {
            label.split(": ", 1)[1]
            for label in scenario.oracle.acceptable_labels
            if label.startswith("pick card ")
        }
        if "Drum of Battle" in accepted_names:
            assert deck_names & exhaust_support_names, scenario.id
        if scenario.category == "resource_curve" and accepted_names & energy_names:
            resource_curve_has_energy_pick = True

    assert resource_curve_has_energy_pick is True


def test_choose_scenarios_filters_by_id_category_and_limit():
    scenarios = load_card_choice_scenarios()
    target = next(scenario for scenario in scenarios if scenario.category == "skip_is_best")

    assert choose_scenarios(scenarios, scenario_ids=[target.id]) == [target]
    assert all(
        scenario.category == "skip_is_best"
        for scenario in choose_scenarios(scenarios, categories=["skip_is_best"])
    )
    assert len(choose_scenarios(scenarios, limit=3)) == 3


def test_run_card_choice_eval_scores_correct_and_incorrect_choices(tmp_path):
    scenarios = [
        next(scenario for scenario in load_card_choice_scenarios() if scenario.category == "skip_is_best"),
        next(scenario for scenario in load_card_choice_scenarios() if scenario.category == "defense_gap_indirect"),
    ]
    correct = scenarios[0]
    incorrect = scenarios[1]
    bad_label = incorrect.oracle.bad_labels[0]
    out = tmp_path / "eval.jsonl"
    agent = _PickByLabelAgent(
        {
            correct.id: correct.oracle.acceptable_labels[0],
            incorrect.id: bad_label,
        }
    )

    summary = run_card_choice_eval(agent=agent, scenarios=scenarios, out=out, log_level="decisions")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenario_rows = [row for row in rows if row["type"] == "scenario"]
    choice_rows = [row for row in rows if row["type"] == "choice"]

    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert summary["incorrect"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["skip_recall"] == 1.0
    assert summary["trap_failure_rate"] == 0.0
    assert {row["verdict"] for row in choice_rows} == {"correct", "incorrect"}
    assert choice_rows[1]["failure_tags"]
    assert "oracle" not in json.dumps(scenario_rows, ensure_ascii=False)
    assert "acceptable_labels" not in json.dumps(scenario_rows, ensure_ascii=False)
    assert "evidence_bundle" in scenario_rows[0]
    assert "prompt" not in scenario_rows[0]
    assert rows[-1]["type"] == "summary"


def test_run_card_choice_eval_full_log_includes_prompt_and_full_evidence(tmp_path):
    scenario = next(scenario for scenario in load_card_choice_scenarios() if scenario.category == "synergy_pick")
    out = tmp_path / "full.jsonl"
    agent = _PickByLabelAgent({scenario.id: scenario.oracle.acceptable_labels[0]})

    run_card_choice_eval(agent=agent, scenarios=[scenario], out=out, log_level="full")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenario_row = next(row for row in rows if row["type"] == "scenario")
    choice_row = next(row for row in rows if row["type"] == "choice")

    assert scenario_row["prompt"] == f"scenario={scenario.id}"
    assert scenario_row["evidence_bundle"]["deck_summary"]["representation"] == "complete_grouped_inventory"
    assert choice_row["selected_label"] == scenario.oracle.acceptable_labels[0]


def test_card_choice_eval_cli_runs_with_random_agent_and_filters(tmp_path):
    scenario = load_card_choice_scenarios()[0]
    out = tmp_path / "cli.jsonl"

    exit_code = main(["--agent", "random", "--scenario-id", scenario.id, "--out", str(out)])

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert exit_code == 0
    assert rows[0]["type"] == "start"
    assert rows[0]["scenario_count"] == 1
    assert rows[1]["scenario_id"] == scenario.id
    assert rows[-1]["type"] == "summary"
