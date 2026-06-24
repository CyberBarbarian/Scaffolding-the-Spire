import pytest

from sts2_bench.actions import LegalAction
from sts2_bench.agents import ModelActionError
from sts2_bench.evidence import build_evidence_bundle
from sts2_bench.multi_agent import CoordinatorAgent, OutOfCombatContextAgent
from sts2_bench.routing import route_state
from sts2_bench.run_benchmark import build_run_config
from sts2_bench.run_context import RunContextStore
from sts2_bench.runner import agent_from_args


def _action(action_id=0, kind="map", label="go to Monster"):
    return LegalAction(
        action_id=action_id,
        kind=kind,
        label=label,
        command={"cmd": "action", "action": "select_map_node", "args": {"col": 0, "row": 1}},
    )


class _FakeAgent:
    def __init__(self, name, *, parsed=None):
        self.name = name
        self.parsed = parsed or {"action_id": 0, "reason": f"{name} chose"}
        self.choices = []
        self.prompts = []
        self.transitions = []

    def build_prompt(self, state, legal_actions):
        self.prompts.append((state, legal_actions))
        return f"{self.name} prompt"

    def choose(self, state, legal_actions, *, prompt=None):
        self.choices.append((state, legal_actions, prompt))
        return legal_actions[0], {"agent": self.name, "parsed": dict(self.parsed)}

    def reset_episode(self):
        self.transitions.clear()

    def record_transition(self, state, action, meta, next_state=None):
        self.transitions.append((state, action, meta, next_state))


class _StaleActionAgent(_FakeAgent):
    def choose(self, state, legal_actions, *, prompt=None):
        self.choices.append((state, legal_actions, prompt))
        stale = LegalAction(
            action_id=99,
            kind="map",
            label="stale action",
            command={"cmd": "action", "action": "select_map_node", "args": {"col": 9, "row": 9}},
        )
        return stale, {"agent": self.name, "parsed": {"action_id": 99, "reason": "stale"}}


class _FakeRewardPosteriorEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, state, legal_actions, evidence_bundle):
        self.calls.append((state, legal_actions, evidence_bundle))
        return (
            {
                "mode": "reward_posterior_deck_analysis",
                "baseline_analysis": "Skipping keeps the current deck unchanged.",
                "candidate_deck_analyses": [
                    {
                        "action_id": candidate.action_id,
                        "label": candidate.label,
                        "deck_delta": "unchanged" if candidate.kind == "card_reward_skip" else "adds one visible card",
                        "immediate_current_deck_effect": "no new card enters the current deck",
                        "redundancy_or_dilution_risks": "no dilution",
                        "visible_support": "current deck remains visible",
                        "missing_or_uncertain_support": "none for the unchanged deck",
                        "comparison_to_unchanged_deck": "this is the unchanged baseline",
                        "uncertainty": "low",
                    }
                    for candidate in evidence_bundle.candidate_evidence
                ],
                "global_uncertainties": [],
            },
            {
                "reward_posterior_evaluator": "fake",
                "reward_posterior_evaluation_hash": "abc123",
                "reward_posterior_prompt_chars": 111,
            },
        )


def test_route_state_assigns_combat_and_combat_modal_to_ca():
    assert route_state({"decision": "combat_play"}).active_agent == "ca"
    assert route_state({"decision": "card_select", "combat": {"round": 2}}).active_agent == "ca"


@pytest.mark.parametrize(
    "decision",
    [
        "map_select",
        "combat_reward",
        "card_reward",
        "bundle_select",
        "rest_site",
        "event_choice",
        "event_result",
        "shop",
        "fake_merchant_shop",
        "treasure",
        "crystal_sphere",
    ],
)
def test_route_state_assigns_out_of_combat_decisions_to_occ(decision):
    assert route_state({"decision": decision}).active_agent == "occ"


def test_route_state_assigns_noncombat_card_select_to_occ():
    routed = route_state({"decision": "card_select", "cards": []})

    assert routed.active_agent == "occ"
    assert "without combat" in routed.reason


def test_run_context_store_accepts_whitelisted_patch_and_round_trips():
    store = RunContextStore()

    summary = store.apply_agent_patch(
        {
            "run_plan_patch": {
                "route_plan": "Prefer a shop path before the Act 1 boss.",
                "build_plan": "Take efficient attacks before speculative scaling.",
                "resource_policy": "Rest below 35 percent HP.",
                "open_threads": ["Remove a Strike at the next shop."],
            },
            "run_memory_facts": [
                {
                    "kind": "decision_summary",
                    "text": "Skipped a low-impact skill reward.",
                    "floor": 3,
                }
            ],
        },
        owner="occ",
        state={"decision": "card_reward", "context": {"floor": 3}},
        action=_action(kind="card_reward_skip", label="skip card reward"),
    )

    assert summary["applied"] is True
    assert summary["run_plan_updates"] == ["route_plan", "build_plan", "resource_policy", "open_threads"]
    data = store.to_dict()
    restored = RunContextStore.from_dict(data)
    assert restored.run_plan.route_plan == "Prefer a shop path before the Act 1 boss."
    assert restored.run_plan.open_threads == ["Remove a Strike at the next shop."]
    assert restored.run_memory.facts[0]["owner"] == "occ"


def test_run_context_store_rejects_unknown_patch_keys():
    store = RunContextStore()

    with pytest.raises(ValueError, match="Unsupported run_plan_patch keys"):
        store.apply_agent_patch(
            {"run_plan_patch": {"freeform_strategy_blob": "do everything"}},
            owner="occ",
            state={"decision": "map_select"},
            action=_action(),
        )


def test_out_of_combat_prompt_includes_rm_rp_slice_and_patch_schema():
    store = RunContextStore()
    store.apply_agent_patch(
        {"run_plan_patch": {"route_plan": "Path toward shop.", "open_threads": ["Remove Strike."]}},
        owner="occ",
        state={"decision": "map_select"},
        action=_action(),
    )
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store)

    prompt = wrapped.build_prompt({"decision": "map_select"}, [_action()])

    assert "Out-of-combat controller context:" in prompt
    assert "Evidence context" in prompt
    assert "advisory only" in prompt
    assert "route_plan: Path toward shop." in prompt
    assert "Remove Strike." in prompt
    assert "run_plan_patch" in prompt
    assert "run_memory_facts" in prompt
    assert "no_context_change_reason" in prompt
    assert "do not score, rank, recommend, or choose actions" in prompt
    assert "Do not leave run_plan_patch empty for a real OCC game action" not in prompt
    assert "occ prompt" in prompt


def test_out_of_combat_prompt_can_hide_run_plan_route_plan():
    store = RunContextStore()
    store.apply_agent_patch(
        {
            "run_plan_patch": {
                "route_plan": "Path toward shop.",
                "build_plan": "Prefer efficient attacks.",
            }
        },
        owner="occ",
        state={"decision": "map_select"},
        action=_action(),
    )
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store, run_plan_route_plan_enabled=False)

    prompt = wrapped.build_prompt({"decision": "map_select"}, [_action()])

    assert "route_plan: Path toward shop." not in prompt
    assert "Path toward shop." not in prompt
    assert "Allowed run_plan_patch keys: build_plan, resource_policy" in prompt
    assert "Allowed run_plan_patch keys: route_plan" not in prompt
    assert "build_plan: Prefer efficient attacks." in prompt


def test_out_of_combat_prompt_includes_short_reference_turn_history_after_transition():
    store = RunContextStore()
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store, history_window=2)
    action = _action(label="go to Shop")
    meta = {
        "active_agent": "occ",
        "parsed": {"reason": "Use early gold to remove a Strike."},
    }
    next_state = {
        "decision": "shop",
        "context": {"floor": 2},
        "last_action_result": {
            "changes": ["room: Map -> Shop", "gold: 99 -> 99"],
        },
    }

    wrapped.record_transition({"decision": "map_select", "context": {"floor": 1}}, action, meta, next_state)
    prompt = wrapped.build_prompt({"decision": "shop"}, [_action(label="leave shop")])

    assert "OCC short turn history" in prompt
    assert "reference only" in prompt
    assert "Use early gold to remove a Strike." in prompt
    assert "go to Shop" in prompt
    assert "room: Map -> Shop" in prompt


def test_card_reward_prompt_uses_current_facts_without_run_plan_or_occ_history():
    store = RunContextStore()
    store.apply_agent_patch(
        {
            "run_plan_patch": {
                "route_plan": "Go to a shop.",
                "build_plan": "Old speculative exhaust plan.",
                "open_threads": ["Old reward reason."],
            },
            "run_memory_facts": ["Previous card reward reason should not be replayed."],
        },
        owner="occ",
        state={"decision": "map_select", "context": {"floor": 1}},
        action=_action(),
    )
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store, history_window=2)
    wrapped.record_transition(
        {"decision": "map_select", "context": {"floor": 1}},
        _action(label="go to Monster"),
        {"parsed": {"reason": "Past route reason."}},
        {"decision": "card_reward", "last_action_result": {"changes": ["room: Map -> Monster"]}},
    )
    state = {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {
            "hp": 55,
            "max_hp": 80,
            "deck_size": 3,
            "deck": [
                {"name": "Strike", "type": "Attack", "cost": 1, "description": "Deal 6 damage."},
                {"name": "Defend", "type": "Skill", "cost": 1, "description": "Gain 5 Block."},
                {"name": "Bash", "type": "Attack", "cost": 2, "description": "Deal damage. Apply Vulnerable."},
            ],
        },
        "cards": [{"index": 0, "name": "Shrug It Off", "type": "Skill", "cost": 1}],
        "can_skip": True,
    }

    prompt = wrapped.build_prompt(state, [_action(kind="card_reward", label="pick card 0: Shrug It Off")])

    assert "Reward decision context: history-free current facts only" in prompt
    assert "complete grouped deck inventory is already included" in prompt
    assert "unchanged current deck as the baseline" in prompt
    assert "marginal improvement over skip" in prompt
    assert "complete_grouped_inventory" in prompt
    assert "Old speculative exhaust plan" not in prompt
    assert "Previous card reward reason" not in prompt
    assert "OCC short turn history" not in prompt
    assert "Out-of-combat controller context" not in prompt


def test_card_reward_prompt_includes_reward_posterior_deck_analysis_when_enabled():
    store = RunContextStore()
    evaluator = _FakeRewardPosteriorEvaluator()
    wrapped = OutOfCombatContextAgent(
        _FakeAgent("occ"),
        store,
        reward_posterior_evaluator=evaluator,
    )
    state = {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {
            "hp": 55,
            "max_hp": 80,
            "deck_size": 2,
            "deck": [
                {"name": "Strike", "type": "Attack", "cost": 1, "description": "Deal 6 damage."},
                {"name": "Defend", "type": "Skill", "cost": 1, "description": "Gain 5 Block."},
            ],
        },
        "cards": [{"index": 0, "name": "Twin Strike", "type": "Attack", "cost": 1}],
        "can_skip": True,
    }
    legal_actions = [
        _action(action_id=0, kind="card_reward", label="pick card 0: Twin Strike"),
        LegalAction(
            action_id=1,
            kind="card_reward_skip",
            label="skip card reward",
            command={"cmd": "action", "action": "skip_card_reward"},
        ),
    ]

    prompt = wrapped.build_prompt(state, legal_actions)
    chosen, meta = wrapped.choose(state, legal_actions, prompt=prompt)

    assert chosen is legal_actions[0]
    assert evaluator.calls
    assert isinstance(evaluator.calls[0][2], type(build_evidence_bundle(state, legal_actions, store=store)))
    assert "Reward posterior deck-state analysis" in prompt
    assert "comparison_to_unchanged_deck" in prompt
    assert "Skipping keeps the current deck unchanged." in prompt
    assert meta["reward_posterior_evaluation_visible"] is True
    assert meta["reward_posterior_evaluation_hash"] == "abc123"
    assert meta["reward_posterior_evaluator"] == "fake"


def test_card_reward_does_not_apply_build_plan_patch_or_auto_reason_fact():
    store = RunContextStore()
    occ = _FakeAgent(
        "occ",
        parsed={
            "action_id": 0,
            "selected_action_label": "pick card 0: Shrug It Off",
            "reason": "Do not persist this card reward narrative.",
            "run_plan_patch": {"build_plan": "Become a block deck."},
            "run_memory_facts": ["This reward reason should not persist."],
        },
    )
    coordinator = CoordinatorAgent(ca_agent=_FakeAgent("ca"), occ_agent=occ, store=store)
    action = _action(kind="card_reward", label="pick card 0: Shrug It Off")

    chosen, meta = coordinator.choose({"decision": "card_reward"}, [action])

    assert chosen is action
    assert store.run_plan.build_plan == ""
    assert store.run_memory.facts == []
    assert meta["run_context_patch"]["applied"] is False
    assert meta["run_context_patch"]["card_reward_history_free"] is True


def test_coordinator_records_combat_report_from_true_transition_and_ca_notes():
    store = RunContextStore()
    coordinator = CoordinatorAgent(ca_agent=_FakeAgent("ca"), occ_agent=_FakeAgent("occ"), store=store)
    state = {
        "decision": "combat_play",
        "round": 3,
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
        "player": {"hp": 42, "max_hp": 80},
    }
    next_state = {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
        "player": {"hp": 34, "max_hp": 80},
        "last_action_result": {"changes": ["hp: 42 -> 34", "combat ended"]},
    }
    meta = {"active_agent": "ca", "parsed": {"combat_notes": ["Multiple turns had poor block availability."]}}

    coordinator.record_transition(state, _action(kind="combat_end_turn", label="end turn"), meta, next_state)

    assert len(store.run_memory.combat_reports) == 1
    report = store.run_memory.combat_reports[0]
    assert report["kind"] == "combat_fact_board"
    assert report["hp_before_final_transition"] == 42
    assert report["hp_after_combat"] == 34
    assert report["hp_delta_final_transition"] == -8
    assert report["final_round"] == 3
    assert report["ca_observations"] == ["Multiple turns had poor block availability."]
    assert "recommendation" not in report


def test_coordinator_routes_to_active_agent_and_commits_occ_patch():
    store = RunContextStore()
    ca = _FakeAgent("ca")
    occ = _FakeAgent(
        "occ",
        parsed={
            "action_id": 0,
            "reason": "Shop path supports removals.",
            "run_plan_patch": {"route_plan": "Take the shop path."},
        },
    )
    coordinator = CoordinatorAgent(ca_agent=ca, occ_agent=occ, store=store)
    action = _action()

    chosen, meta = coordinator.choose({"decision": "map_select"}, [action])

    assert chosen is action
    assert occ.choices and not ca.choices
    assert meta["active_agent"] == "occ"
    assert meta["route_reason"] == "map_select is routed to OCC"
    assert meta["run_context_patch"]["applied"] is True
    assert meta["run_context_patch"]["facts_added"] == 1
    assert store.run_plan.route_plan == "Take the shop path."
    assert store.run_memory.facts[0]["text"] == "Shop path supports removals."


def test_coordinator_can_ignore_run_plan_route_plan_patch_when_disabled():
    store = RunContextStore()
    occ = _FakeAgent(
        "occ",
        parsed={
            "action_id": 0,
            "reason": "Choose from current map only.",
            "run_plan_patch": {
                "route_plan": "Take the elite path.",
                "resource_policy": "Rest below half HP.",
            },
        },
    )
    coordinator = CoordinatorAgent(
        ca_agent=_FakeAgent("ca"),
        occ_agent=occ,
        store=store,
        run_plan_route_plan_enabled=False,
    )
    action = _action()

    chosen, meta = coordinator.choose({"decision": "map_select"}, [action])

    assert chosen is action
    assert store.run_plan.route_plan == ""
    assert store.run_plan.resource_policy == "Rest below half HP."
    assert meta["run_context_patch"]["run_plan_updates"] == ["resource_policy"]
    assert meta["run_context_patch"]["ignored_run_plan_updates"] == ["route_plan"]


def test_coordinator_strict_guard_raises_for_stale_delegate_action():
    coordinator = CoordinatorAgent(
        ca_agent=_FakeAgent("ca"),
        occ_agent=_StaleActionAgent("occ"),
    )

    with pytest.raises(ModelActionError, match="outside the current legal action list"):
        coordinator.choose({"decision": "map_select"}, [_action()])


def test_coordinator_records_transition_on_selected_child_agent():
    ca = _FakeAgent("ca")
    occ = _FakeAgent("occ")
    coordinator = CoordinatorAgent(ca_agent=ca, occ_agent=occ)
    state = {"decision": "combat_play"}
    action = _action(kind="combat_end_turn", label="end turn")
    meta = {"active_agent": "ca"}
    next_state = {"decision": "combat_reward"}

    coordinator.record_transition(state, action, meta, next_state)

    assert ca.transitions == [(state, action, meta, next_state)]
    assert occ.transitions == []


def test_agent_from_args_can_build_coordinator_agent():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        llm_timeout=5.0,
        llm_max_retries=0,
        prompt_style="analysis",
        turn_chat_window=3,
        turn_chat_plan_enabled=True,
        self_consistency_enabled=True,
        self_consistency_samples=5,
        self_consistency_decisions="map_select,card_reward",
        self_consistency_mode="majority_vote",
        reflection_enabled=True,
        reflection_decisions="noncombat_non_map",
        reflection_mode="verifier_vote",
        reflection_max_options=6,
        action_prompt_order="shuffle_map",
        action_prompt_order_seed=7,
    )

    assert isinstance(agent, CoordinatorAgent)
    assert agent.fallback_mode == "error"
    assert agent.ca_agent.conversation_mode == "turn_chat"
    assert agent.ca_agent.turn_chat_plan_enabled is True
    assert agent.ca_agent.route_plan_enabled is False
    assert agent.ca_agent.context_patch_fields_enabled is False
    assert agent.ca_agent.fallback_mode == "error"
    assert isinstance(agent.occ_agent, OutOfCombatContextAgent)
    assert agent.occ_agent.history_window == 3
    assert agent.occ_agent.inner.fallback_mode == "error"
    assert agent.occ_agent.inner.self_consistency_enabled is True
    assert agent.occ_agent.inner.self_consistency_samples == 5
    assert agent.occ_agent.inner.self_consistency_decisions == ("map_select", "card_reward")
    assert agent.occ_agent.inner.self_consistency_mode == "majority_vote"
    assert agent.occ_agent.inner.reflection_enabled is True
    assert agent.occ_agent.inner.reflection_decisions == ("noncombat_non_map",)
    assert agent.occ_agent.inner.reflection_mode == "verifier_vote"
    assert agent.occ_agent.inner.reflection_max_options == 6
    assert agent.occ_agent.inner.action_prompt_order == "shuffle_map"
    assert agent.occ_agent.inner.action_prompt_order_seed == 7
    assert agent.ca_agent.self_consistency_enabled is False
    assert agent.ca_agent.reflection_enabled is False
    assert agent.occ_agent.reward_posterior_evaluator is not None


def test_run_config_records_coordinator_context_split():
    class Args:
        agent = "coordinator"
        base_url = "http://127.0.0.1:1"
        model = "test"
        llm_timeout = 5.0
        llm_max_retries = 0
        llm_fallback_mode = "error"
        llm_temperature = 0.0
        prompt_style = "analysis"
        include_json_state = False
        memory_enabled = False
        memory_window = 8
        memory_mode = "factual_diff"
        run_summary = True
        context_mode = "single_turn"
        turn_chat_window = 3
        turn_chat_update_mode = "delta"
        turn_chat_assistant_history = "compact"
        turn_check = True
        combat_card_sequences = True
        route_plan_enabled = False
        route_plan_enforcement = "soft"
        run_plan_route_plan_enabled = False
        self_consistency_enabled = True
        self_consistency_samples = 5
        self_consistency_decisions = "map_select,card_reward"
        self_consistency_mode = "majority_vote"
        reflection_enabled = True
        reflection_decisions = "noncombat_non_map"
        reflection_mode = "verifier_vote"
        reflection_max_options = 6
        action_prompt_order = "shuffle_map"
        action_prompt_order_seed = 7
        reward_posterior_evaluator = True
        character = "Ironclad"
        ascension = 0
        lang = "en"
        seeds = None
        count = 0
        max_steps = 10
        max_game_actions = 0
        include_full_map = True
        allow_repeat_views = False
        out = ""
        log_level = "metrics"
        print_prompts = False
        print_model_output = False

    config = build_run_config(Args, [], ["--api-key", "secret"])

    context = config["agent"]["context_management"]
    assert config["agent"]["llm_fallback_mode"] == "error"
    assert context["mode"] == "coordinator"
    assert context["route_plan"]["enabled"] is False
    assert context["route_plan"]["enforcement"] == "soft"
    assert context["run_plan"]["route_plan"]["enabled"] is False
    assert context["ca"]["mode"] == "turn_chat"
    assert context["ca"]["turn_chat"]["window"] == 3
    assert context["ca"]["turn_chat"]["turn_check"]["enabled"] is True
    assert context["occ"]["mode"] == "reference_turn"
    assert context["occ"]["reference_context"]["window"] == 3
    assert context["occ"]["reference_context"]["source"] == "rm_rp_slice_plus_short_occ_history"
    assert context["occ"]["reward_posterior_evaluator"] is True
    assert config["agent"]["diagnostics"]["self_consistency"] == {
        "enabled": True,
        "samples": 5,
        "decisions": "map_select,card_reward",
        "mode": "majority_vote",
    }
    assert config["agent"]["diagnostics"]["reflection"] == {
        "enabled": True,
        "decisions": "noncombat_non_map",
        "mode": "verifier_vote",
        "max_options": 6,
    }
    assert config["agent"]["diagnostics"]["action_prompt_order"] == {
        "mode": "shuffle_map",
        "seed": 7,
    }
