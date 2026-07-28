import json

import pytest

from sts2_bench.actions import (
    LegalAction,
    build_legal_actions,
    materialize_card_select_space_action,
)
from sts2_bench.agents import ModelActionError, OpenAICompatAgent
from sts2_bench.evidence import build_evidence_bundle
from sts2_bench.multi_agent import CoordinatorAgent, OutOfCombatContextAgent
from sts2_bench.routing import route_state
from sts2_bench.run_benchmark import build_run_config
from sts2_bench.run_context import RunContextStore
from sts2_bench.runner import agent_from_args, run_one


def _action(action_id=0, kind="map", label="go to Monster"):
    return LegalAction(
        action_id=action_id,
        kind=kind,
        label=label,
        command={"cmd": "action", "action": "select_map_node", "args": {"col": 0, "row": 1}},
    )


def _json_block_after(prompt, marker):
    payload = prompt.split(marker + "\n", 1)[1].split("\n\n", 1)[0]
    return json.loads(payload)


def test_occ_card_select_prompt_keeps_complete_cards_without_repeating_subset_evidence():
    cards = [
        {
            "index": index,
            "name": f"Card {index}",
            "description": f"Complete effect {index}.",
            "after_upgrade": {"description": f"Complete upgraded effect {index}."},
            "engine_extension": {"must_remain_visible": index},
        }
        for index in range(12)
    ]
    state = {
        "decision": "card_select",
        "min_select": 4,
        "max_select": 4,
        "can_skip": False,
        "cards": cards,
    }
    legal_actions = build_legal_actions(state)
    inner = OpenAICompatAgent(base_url="http://127.0.0.1:1", model="test")
    wrapped = OutOfCombatContextAgent(inner, RunContextStore())

    prompt = wrapped.build_prompt(state, legal_actions)

    action_space = _json_block_after(prompt, "Legal actions:")
    assert action_space["candidates"] == cards
    assert action_space["legal_combination_count"] == 495
    assert "select cards 0,1,2,3" not in prompt
    evidence = _json_block_after(prompt, "Evidence modules do not score, rank, recommend, or choose actions; the LLM policy must use the current decision's declared legal selection protocol.")
    assert evidence["candidate_evidence"] == []
    assert "card_select_legal_combination_count=495" in evidence["state_facts"]


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


class _MaterializedCardSelectAgent(_FakeAgent):
    def __init__(self, name, selected_indices):
        super().__init__(name)
        self.selected_indices = selected_indices

    def choose(self, state, legal_actions, *, prompt=None):
        self.choices.append((state, legal_actions, prompt))
        template = next(action for action in legal_actions if action.kind == "card_select_space")
        action = materialize_card_select_space_action(
            state,
            template,
            self.selected_indices,
        )
        return action, {
            "agent": self.name,
            "parsed": {
                "selected_card_indices": list(self.selected_indices),
                "reason": "Select the requested cards.",
            },
        }


class _ForgedMaterializedCardSelectAgent(_FakeAgent):
    def choose(self, state, legal_actions, *, prompt=None):
        self.choices.append((state, legal_actions, prompt))
        template = next(action for action in legal_actions if action.kind == "card_select_space")
        forged = LegalAction(
            action_id=template.action_id,
            kind="card_select",
            label="select cards 0,1,99: Card 0 + Card 1 + Missing",
            command={
                "cmd": "action",
                "action": "select_cards",
                "args": {"indices": "0,1,99"},
            },
        )
        return forged, {
            "agent": self.name,
            "parsed": {
                "selected_card_indices": [0, 1, 99],
                "reason": "Forge a stale card selection.",
            },
        }


class _CardSelectSpaceProcess:
    def __init__(self, state):
        self.state = state
        self.sent_commands = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return self.state

    def send(self, command):
        self.sent_commands.append(command)
        return {
            "decision": "game_over",
            "victory": False,
            "context": {"act": 2, "floor": 1, "room_type": "Event"},
            "player": {
                "hp": 80,
                "max_hp": 80,
                "gold": 0,
                "deck_size": len(self.state["cards"]),
                "deck": list(self.state["cards"]),
                "relics": [],
                "potions": [],
            },
        }

    def close(self):
        pass


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


def test_coordinator_records_forced_map_fact_without_invoking_model_delegates():
    ca = _FakeAgent("ca")
    occ = _FakeAgent("occ")
    store = RunContextStore(formal_route_plan_enabled=True)
    coordinator = CoordinatorAgent(ca_agent=ca, occ_agent=occ, store=store)
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 2, "room_type": "Map"},
        "player": {"hp": 76, "max_hp": 80},
        "choices": [{"col": 1, "row": 2, "type": "Elite"}],
        "active_route_plan": {
            "plan_id": "rp_001",
            "expected_next": {"col": 1, "row": 2, "type": "Elite"},
            "remaining": [{"col": 1, "row": 2, "type": "Elite"}],
        },
    }
    action = LegalAction(
        action_id=0,
        label="go to Elite at col=1 row=2",
        command={
            "cmd": "action",
            "action": "select_map_node",
            "args": {"col": 1, "row": 2},
        },
        kind="map",
    )
    next_state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 2, "room_type": "Elite"},
        "player": {"hp": 76, "max_hp": 80},
    }

    coordinator.record_runtime_transition(
        state,
        action,
        {
            "agent": "runtime",
            "policy": "forced_singleton_map",
            "route_plan_effect": {"outcome": "followed_active_plan"},
        },
        next_state,
    )

    assert ca.choices == []
    assert occ.choices == []
    assert ca.transitions == []
    assert occ.transitions == []
    assert store.run_memory.facts == []
    assert len(store.run_memory.route_steps) == 1
    assert store.run_memory.route_steps[0]["selected_matches_active_plan"] is True


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


def test_run_context_store_rejects_invalid_patch_atomically():
    store = RunContextStore()
    store.run_plan.build_plan = "Original build plan."
    before = store.to_dict()
    before_hash = store.snapshot_hash()

    with pytest.raises(ValueError, match="open_threads must be a string or list of strings"):
        store.apply_agent_patch(
            {
                "run_plan_patch": {
                    "build_plan": "This valid-looking field must not leak through.",
                    "open_threads": 123,
                },
                "run_memory_facts": [{"text": "This fact must not be appended."}],
            },
            owner="occ",
            state={"decision": "card_reward", "context": {"floor": 3}},
            action=_action(kind="card_reward_skip", label="skip card reward"),
        )

    assert store.to_dict() == before
    assert store.snapshot_hash() == before_hash


def test_coordinator_rejects_split_brain_occ_context_store():
    coordinator_store = RunContextStore()
    occ_store = RunContextStore()
    occ = OutOfCombatContextAgent(_FakeAgent("occ"), occ_store)

    with pytest.raises(ValueError, match="must share the same RunContextStore"):
        CoordinatorAgent(
            ca_agent=_FakeAgent("ca"),
            occ_agent=occ,
            store=coordinator_store,
        )


def test_out_of_combat_prompt_includes_rm_rp_slice_and_patch_schema():
    store = RunContextStore()
    store.apply_agent_patch(
        {"run_plan_patch": {"route_plan": "Path toward shop.", "open_threads": ["Remove Strike."]}},
        owner="occ",
        state={"decision": "event_choice"},
        action=_action(kind="event_choice", label="choose option"),
    )
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store)

    prompt = wrapped.build_prompt({"decision": "event_choice"}, [_action(kind="event_choice", label="choose option")])

    assert "Out-of-combat controller context:" in prompt
    assert "Evidence context" in prompt
    assert "advisory only" in prompt
    assert "route_plan: Path toward shop." in prompt
    assert "Remove Strike." in prompt
    assert "run_plan_patch" in prompt
    assert "run_memory_facts" in prompt
    assert "no_context_change_reason" not in prompt
    assert "do not score, rank, recommend, or choose actions" in prompt
    assert "Do not leave run_plan_patch empty for a real OCC game action" not in prompt
    assert "occ prompt" in prompt


def test_formal_route_plan_store_rejects_legacy_route_plan_patch():
    store = RunContextStore(formal_route_plan_enabled=True)

    with pytest.raises(ValueError, match="Unsupported run_plan_patch keys: route_plan"):
        store.apply_agent_patch(
            {"run_plan_patch": {"route_plan": ["r00c3", "r01c1"]}},
            owner="occ",
            state={"decision": "event_choice"},
            action=_action(kind="event_choice", label="choose option"),
        )

    assert store.run_plan.route_plan == ""


def test_formal_route_plan_map_prompt_hides_legacy_draft_and_exposes_null_active_plan():
    store = RunContextStore(formal_route_plan_enabled=True)
    store.run_plan.route_plan = "stale unverified route draft"
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store)
    state = {
        "decision": "map_select",
        "player": {"hp": 80, "max_hp": 80, "deck": []},
        "choices": [{"col": 1, "row": 1, "type": "Monster"}],
    }

    prompt = wrapped.build_prompt(state, [_action(label="go to Monster")])
    route_context = _json_block_after(prompt, "Route OCC context JSON:")

    assert "stale unverified route draft" not in prompt
    assert "route_plan" not in route_context["run_plan"]
    assert route_context["active_route_plan"] is None
    assert route_context["route_plan_authority"] == "active_route_plan_only"
    assert "route_plan" not in route_context["output_contract"]["allowed_run_plan_patch_keys"]


def test_disabled_route_plan_map_prompt_has_no_persistent_route_channel():
    store = RunContextStore(
        formal_route_plan_enabled=False,
        route_plan_enabled=False,
    )
    store.run_plan.route_plan = "stale route draft must stay hidden"
    store.run_memory.route_steps = [
        {
            "kind": "route_step",
            "act": 1,
            "floor": 1,
            "selected": {"col": 1, "row": 1, "type": "Monster"},
            "selected_matches_active_plan": None,
            "route_plan_event_type": None,
            "next_decision": "combat_play",
        }
    ]
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store)
    state = {
        "decision": "map_select",
        "player": {"hp": 80, "max_hp": 80, "deck": []},
        "choices": [{"col": 1, "row": 1, "type": "Monster"}],
    }

    prompt = wrapped.build_prompt(state, [_action(label="go to Monster")])
    route_context = _json_block_after(prompt, "Route OCC context JSON:")

    assert "stale route draft must stay hidden" not in prompt
    assert "run_plan" not in route_context
    assert "active_route_plan" not in route_context
    assert "route_plan_authority" not in route_context
    assert "route_plan" not in route_context["output_contract"]["allowed_run_plan_patch_keys"]
    assert "route_decision_contract" not in route_context
    assert "selected_matches_active_plan" not in route_context["route_process"]["steps"][0]
    assert "route_plan_event_type" not in route_context["route_process"]["steps"][0]


def test_map_out_of_combat_prompt_uses_structured_route_context_without_candidate_duplication():
    store = RunContextStore()
    store.run_plan.route_plan = "Prefer the rest path if HP remains low."
    store.run_memory.route_steps = [
        {
            "kind": "route_step",
            "act": 1,
            "floor": 4,
            "selected": {"col": 0, "row": 1, "type": "Monster"},
            "hp_at_selection": 42,
            "max_hp": 80,
            "next_decision": "combat_play",
        }
    ]
    store.run_memory.combat_reports.append(
        {
            "kind": "combat_fact_board",
            "act": 1,
            "floor": 4,
            "hp_before_final_transition": 42,
            "hp_after_combat": 34,
            "hp_delta_final_transition": -8,
            "final_round": 3,
            "room_type": "Monster",
        }
    )
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store)
    state = {
        "decision": "map_select",
        "player": {
            "hp": 34,
            "max_hp": 80,
            "gold": 99,
            "deck_size": 3,
            "deck": [
                {"name": "Strike", "type": "Attack", "cost": 1},
                {"name": "Defend", "type": "Skill", "cost": 1},
                {"name": "Bash", "type": "Attack", "cost": 2},
            ],
        },
        "choices": [{"col": 0, "row": 1, "type": "RestSite"}],
    }

    prompt = wrapped.build_prompt(state, [_action(label="go to RestSite")])
    route_context = _json_block_after(prompt, "Route OCC context JSON:")

    assert "Route OCC context JSON:" in prompt
    assert "Out-of-combat controller context:" not in prompt
    assert "Evidence context" not in prompt
    assert "OCC short turn history" not in prompt
    assert "candidate_evidence" not in json.dumps(route_context, ensure_ascii=False)
    assert route_context["run_plan"]["route_plan"] == "Prefer the rest path if HP remains low."
    assert "run_memory" not in route_context
    assert route_context["route_process"]["steps"][0]["selected"] == {"col": 0, "row": 1, "type": "Monster"}
    assert route_context["route_process"]["steps"][0]["hp_at_selection"] == 42
    assert route_context["route_process"]["summary"]["visited_room_types"] == ["Monster"]
    assert route_context["hp_loss_trajectory"]["segments"][0]["hp_loss"] == 8
    assert route_context["hp_loss_trajectory"]["segments"][0]["room_type"] == "Monster"
    assert route_context["hp_loss_trajectory"]["summary"]["total_hp_loss"] == 8
    assert route_context["hp_loss_trajectory"]["summary"]["current_hp"] == 34
    assert route_context["evidence_summary"]["deck_summary_visible"] is True
    assert route_context["output_contract"]["required_keys"] == [
        "action_id",
        "selected_action_label",
        "reason",
        "evidence_refs",
        "run_plan_patch",
        "run_memory_facts",
    ]
    assert "run_plan_patch" in prompt


def test_out_of_combat_prompt_includes_room_process_after_entering_room():
    store = RunContextStore()
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), store, history_window=2)
    action = _action(label="go to Shop")
    meta = {
        "active_agent": "occ",
        "parsed": {"reason": "Use early gold to remove a Strike."},
    }
    next_state = {
        "decision": "shop",
        "context": {"act": 1, "floor": 2, "room_type": "Shop"},
        "last_action_result": {
            "changes": ["room: Map -> Shop", "gold: 99 -> 99"],
        },
    }

    wrapped.record_transition({"decision": "map_select", "context": {"floor": 1}}, action, meta, next_state)
    prompt = wrapped.build_prompt(
        {"decision": "shop", "context": {"act": 1, "floor": 2, "room_type": "Shop"}},
        [_action(label="leave shop")],
    )
    room_context = _json_block_after(prompt, "Room process context JSON:")

    assert "OCC short turn history" not in prompt
    assert room_context["room_key"] == {"act": 1, "floor": 2, "room_type": "Shop"}
    assert room_context["actions"][0]["action_kind"] == "map"
    assert room_context["actions"][0]["reason"] == "Use early gold to remove a Strike."
    assert room_context["actions"][0]["outcome"] == ["room: Map -> Shop", "gold: 99 -> 99"]


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

    assert "Card reward decision context: current facts only" in prompt
    assert "complete grouped deck inventory is already included" in prompt
    assert "complete_grouped_inventory" in prompt
    assert "posterior" not in prompt.lower()
    assert "posterior_audit" not in prompt
    assert "candidate_evidence" not in prompt
    assert "Evidence context" not in prompt
    assert "combat_notes" not in prompt
    assert "COMBAT_EVALUATION" not in prompt
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
    assert meta["reward_posterior_evaluator_enabled"] is True
    assert meta["reward_posterior_evaluation_visible"] is True
    assert meta["reward_posterior_evaluation_hash"] == "abc123"
    assert meta["reward_posterior_evaluator"] == "fake"


def test_card_reward_trace_meta_explicitly_records_posterior_off():
    wrapped = OutOfCombatContextAgent(_FakeAgent("occ"), RunContextStore())
    state = {
        "decision": "card_reward",
        "player": {"deck": [{"name": "Strike"}]},
        "cards": [{"index": 0, "name": "Zap"}],
        "can_skip": True,
    }
    legal_actions = [_action(kind="card_reward", label="pick card 0: Zap")]

    prompt = wrapped.build_prompt(state, legal_actions)
    _chosen, meta = wrapped.choose(state, legal_actions, prompt=prompt)

    assert meta["reward_posterior_evaluator_enabled"] is False
    assert meta["reward_posterior_evaluation_visible"] is False


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


def test_room_process_context_records_visible_card_reward_skip_without_stale_ids():
    store = RunContextStore()
    occ = OutOfCombatContextAgent(_FakeAgent("occ"), store)
    coordinator = CoordinatorAgent(ca_agent=_FakeAgent("ca"), occ_agent=occ, store=store)
    combat_state = {
        "decision": "combat_play",
        "round": 2,
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
        "player": {"hp": 42, "max_hp": 80},
    }
    reward_state = {
        "decision": "combat_reward",
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
        "player": {"hp": 37, "max_hp": 80},
        "rewards": [{"index": 0, "kind": "card_reward"}],
    }
    open_card_reward = LegalAction(
        action_id=7,
        kind="combat_reward",
        label="open card reward 0: reveal offered cards",
        command={"cmd": "action", "action": "claim_reward", "args": {"reward_index": 0}},
    )
    visible_card_state = {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
        "player": {"hp": 37, "max_hp": 80},
        "cards": [
            {"index": 0, "name": "Rupture", "type": "Power", "cost": 1},
            {"index": 1, "name": "Twin Strike", "type": "Attack", "cost": 1},
            {"index": 2, "name": "Clash", "type": "Attack", "cost": 0},
        ],
        "can_skip": True,
    }
    skip_visible_cards = LegalAction(
        action_id=3,
        kind="card_reward_skip",
        label="skip card reward",
        command={"cmd": "action", "action": "skip_card_reward"},
    )

    coordinator.record_transition(
        combat_state,
        _action(kind="combat_end_turn", label="end turn"),
        {"active_agent": "ca", "parsed": {"combat_notes": ["Took damage to finish the fight."]}},
        reward_state,
    )
    coordinator.record_transition(
        reward_state,
        open_card_reward,
        {"active_agent": "occ", "parsed": {"reason": "Reveal the offered cards before deciding."}},
        visible_card_state,
    )
    coordinator.record_transition(
        visible_card_state,
        skip_visible_cards,
        {"active_agent": "occ", "parsed": {"reason": "Do not add these low-impact attacks."}},
        reward_state,
    )

    prompt = coordinator.build_prompt(reward_state, [open_card_reward])
    room_context = _json_block_after(prompt, "Room process context JSON:")
    room_payload = json.dumps(room_context, ensure_ascii=False, sort_keys=True)

    assert room_context["room_key"] == {"act": 1, "floor": 4, "room_type": "Monster"}
    assert any(item["action_kind"] == "combat_end_turn" for item in room_context["actions"])
    assert room_context["inspected_card_rewards"][0]["candidates"] == ["Rupture", "Twin Strike", "Clash"]
    assert room_context["inspected_card_rewards"][0]["last_decision"] == "skipped_visible_candidates"
    assert room_context["inspected_card_rewards"][0]["still_available_after_skip"] is True
    assert "action_id" not in room_payload
    assert "card_index" not in room_payload
    assert "reward_index" not in room_payload


def test_room_process_context_resets_when_current_room_changes():
    store = RunContextStore()
    occ = OutOfCombatContextAgent(_FakeAgent("occ"), store)
    coordinator = CoordinatorAgent(ca_agent=_FakeAgent("ca"), occ_agent=occ, store=store)
    reward_state = {
        "decision": "combat_reward",
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
        "rewards": [{"index": 0, "kind": "card_reward"}],
    }
    visible_card_state = {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 4, "room_type": "Monster"},
        "cards": [{"index": 0, "name": "Rupture"}],
        "can_skip": True,
    }
    open_card_reward = LegalAction(
        action_id=7,
        kind="combat_reward",
        label="open card reward 0: reveal offered cards",
        command={"cmd": "action", "action": "claim_reward", "args": {"reward_index": 0}},
    )

    coordinator.record_transition(
        reward_state,
        open_card_reward,
        {"active_agent": "occ", "parsed": {"reason": "Reveal the offered cards."}},
        visible_card_state,
    )

    next_room_state = {
        "decision": "event_choice",
        "context": {"act": 1, "floor": 5, "room_type": "Event"},
    }
    prompt = coordinator.build_prompt(next_room_state, [_action(kind="event_choice", label="choose option")])

    assert "Room process context JSON:" not in prompt
    assert "Rupture" not in prompt


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


def test_coordinator_records_route_step_from_map_transition():
    store = RunContextStore()
    coordinator = CoordinatorAgent(ca_agent=_FakeAgent("ca"), occ_agent=_FakeAgent("occ"), store=store)
    state = {
        "decision": "map_select",
        "act": 1,
        "floor": 4,
        "context": {"act": 1, "floor": 4, "room_type": "Map"},
        "player": {"hp": 42, "max_hp": 80, "gold": 99, "deck_size": 12},
        "choices": [{"col": 2, "row": 5, "type": "RestSite"}],
        "active_route_plan": {
            "expected_next": {"col": 1, "row": 5, "type": "Elite"},
        },
    }
    next_state = {
        "decision": "rest_site",
        "context": {"act": 1, "floor": 5, "room_type": "RestSite"},
        "player": {"hp": 42, "max_hp": 80, "gold": 99, "deck_size": 12},
    }
    action = LegalAction(
        action_id=0,
        kind="map",
        label="go to RestSite",
        command={"cmd": "action", "action": "select_map_node", "args": {"col": 2, "row": 5}},
    )
    meta = {
        "active_agent": "occ",
        "parsed": {
            "route_plan_event": {
                "type": "route_plan_updated",
                "reason": "Low HP needs the rest site.",
            }
        },
    }

    coordinator.record_transition(state, action, meta, next_state)

    assert len(getattr(store.run_memory, "route_steps", [])) == 1
    step = store.run_memory.route_steps[0]
    assert step["kind"] == "route_step"
    assert step["selected"] == {"col": 2, "row": 5, "type": "RestSite"}
    assert step["hp_at_selection"] == 42
    assert step["max_hp"] == 80
    assert step["selected_matches_active_plan"] is False
    assert step["route_plan_event_type"] == "route_plan_updated"
    assert step["next_decision"] == "rest_site"


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


def test_coordinator_strict_guard_raises_for_stale_delegate_action():
    coordinator = CoordinatorAgent(
        ca_agent=_FakeAgent("ca"),
        occ_agent=_StaleActionAgent("occ"),
    )

    with pytest.raises(ModelActionError, match="outside the current legal action list"):
        coordinator.choose({"decision": "map_select"}, [_action()])


def test_coordinator_accepts_valid_materialized_card_select_space_action():
    state = {
        "decision": "card_select",
        "min_select": 3,
        "max_select": 3,
        "can_skip": False,
        "cards": [{"index": index, "name": f"Card {index}"} for index in range(20)],
    }
    legal_actions = build_legal_actions(state)
    coordinator = CoordinatorAgent(
        ca_agent=_FakeAgent("ca"),
        occ_agent=_MaterializedCardSelectAgent("occ", [0, 1, 2]),
    )

    chosen, meta = coordinator.choose(state, legal_actions)

    assert chosen.kind == "card_select"
    assert chosen.command["args"]["indices"] == "0,1,2"
    assert meta["active_agent"] == "occ"


def test_coordinator_rejects_forged_materialized_card_select_space_action():
    state = {
        "decision": "card_select",
        "min_select": 3,
        "max_select": 3,
        "can_skip": False,
        "cards": [{"index": index, "name": f"Card {index}"} for index in range(20)],
    }
    legal_actions = build_legal_actions(state)
    coordinator = CoordinatorAgent(
        ca_agent=_FakeAgent("ca"),
        occ_agent=_ForgedMaterializedCardSelectAgent("occ"),
    )

    with pytest.raises(ModelActionError, match="outside the current legal action list"):
        coordinator.choose(state, legal_actions)


def test_runner_executes_materialized_card_select_space_action_end_to_end():
    state = {
        "decision": "card_select",
        "context": {"act": 2, "floor": 1, "room_type": "Event"},
        "min_select": 3,
        "max_select": 3,
        "can_skip": False,
        "cards": [{"index": index, "name": f"Card {index}"} for index in range(20)],
    }
    process = _CardSelectSpaceProcess(state)
    coordinator = CoordinatorAgent(
        ca_agent=_FakeAgent("ca"),
        occ_agent=_MaterializedCardSelectAgent("occ", [0, 1, 2]),
    )

    result = run_one(
        agent=coordinator,
        seed="card-select-space-mini",
        process=process,
        max_steps=2,
    )

    assert result.error is None
    assert result.decision == "game_over"
    assert process.sent_commands == [
        {
            "cmd": "action",
            "action": "select_cards",
            "args": {"indices": "0,1,2"},
        }
    ]


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
    )

    assert isinstance(agent, CoordinatorAgent)
    assert agent.fallback_mode == "error"
    assert agent.ca_agent.conversation_mode == "single_turn"
    assert agent.ca_agent.turn_chat_plan_enabled is True
    assert agent.ca_agent.fallback_mode == "error"
    assert isinstance(agent.occ_agent, OutOfCombatContextAgent)
    assert agent.occ_agent.history_window == 3
    assert agent.occ_agent.inner.fallback_mode == "error"
    assert agent.occ_agent.reward_posterior_evaluator is not None


def test_agent_from_args_can_build_coordinator_with_combat_mdp_ca():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        llm_timeout=5.0,
        llm_max_retries=0,
        conversation_mode="combat_mdp_state",
        turn_chat_window=3,
    )

    assert isinstance(agent, CoordinatorAgent)
    assert agent.ca_agent.conversation_mode == "combat_mdp_state"
    assert agent.ca_agent.short_term_memory_enabled is False
    assert isinstance(agent.occ_agent, OutOfCombatContextAgent)
    assert agent.occ_agent.inner.conversation_mode == "single_turn"


def test_agent_from_args_keeps_single_turn_ca_as_minimal_combat_context():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="single_turn",
    )

    assert isinstance(agent, CoordinatorAgent)
    assert agent.ca_agent.conversation_mode == "single_turn"
    assert agent.ca_agent.turn_chat_buffer.messages == []


@pytest.mark.parametrize("posterior_enabled", [True, False])
def test_single_agent_reuses_history_free_reward_posterior_switch(posterior_enabled):
    agent = agent_from_args(
        "llm",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        reward_posterior_evaluator_enabled=posterior_enabled,
    )

    assert isinstance(agent, OutOfCombatContextAgent)
    assert agent.reward_only is True
    assert isinstance(agent.inner, OpenAICompatAgent)
    assert (agent.reward_posterior_evaluator is not None) is posterior_enabled


def test_agent_from_args_propagates_paired_context_switch_to_both_delegates():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="single_turn",
        rule_projection_and_recent_changes_enabled=False,
    )

    assert isinstance(agent, CoordinatorAgent)
    assert agent.ca_agent.rule_projection_and_recent_changes_enabled is False
    assert agent.occ_agent.inner.rule_projection_and_recent_changes_enabled is False
    assert agent.store.formal_route_plan_enabled is True


def test_all_four_ablation_controls_compose_without_provider_call_or_prompt_leak(monkeypatch):
    agent = agent_from_args(
        "llm",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="single_turn",
        reward_posterior_evaluator_enabled=False,
        rule_projection_and_recent_changes_enabled=False,
    )
    assert isinstance(agent, OutOfCombatContextAgent)
    assert isinstance(agent.inner, OpenAICompatAgent)
    monkeypatch.setattr(
        agent.inner,
        "_chat",
        lambda _messages: pytest.fail("prompt-only switch validation must not call a provider"),
    )
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 3, "room_type": "Monster"},
        "energy": 3,
        "max_energy": 3,
        "player": {
            "name": "Defect",
            "hp": 40,
            "max_hp": 70,
            "block": 0,
            "deck": [
                {
                    "id": "CARD.ZAP",
                    "name": "Zap",
                    "type": "Skill",
                    "cost": 1,
                    "description": "Channel 1 Lightning.",
                    "upgraded": False,
                }
            ],
            "relics": [{"id": "CRACKED_CORE", "name": "Cracked Core"}],
            "potions": [{"id": "SWIFT_POTION", "name": "Swift Potion"}],
        },
        "hand": [
            {
                "index": 0,
                "id": "CARD.ZAP",
                "name": "Zap",
                "type": "Skill",
                "cost": 1,
                "description": "Channel 1 Lightning.",
            }
        ],
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 40, "block": 0}],
        "last_action_result": {
            "action_label": "play Defend",
            "action_kind": "combat_card",
            "changes": ["block: 0 -> 5"],
        },
    }
    legal_actions = [
        LegalAction(
            action_id=0,
            kind="combat_end_turn",
            label="end turn",
            command={"cmd": "action", "action": "end_turn"},
        )
    ]

    prompt = agent.build_prompt(state, legal_actions)

    assert agent.reward_only is True
    assert agent.reward_posterior_evaluator is None
    assert agent.inner.conversation_mode == "single_turn"
    assert "Game rules for the current decision:" not in prompt
    assert "Last action result:" not in prompt
    assert '"last_action_result"' not in prompt
    assert "Zap" in prompt
    assert "Cracked Core" in prompt
    assert "Swift Potion" in prompt
    assert "Jaw Worm" in prompt
    assert "Legal actions:" in prompt
    assert "disabled" not in prompt.lower()
    assert "ablation" not in prompt.lower()
    for treatment_hint in (
        "control group",
        "treatment group",
        "minimal formal",
        "minimal context",
        "simplified context",
        "no persistent route",
        "without route plan",
    ):
        assert treatment_hint not in prompt.lower()


def test_explicit_default_switch_values_preserve_coordinator_prompt_exactly():
    common = {
        "base_url": "http://127.0.0.1:1",
        "model": "test",
        "api_key": "local",
        "conversation_mode": "combat_mdp_state",
        "reward_posterior_evaluator_enabled": True,
    }
    implicit = agent_from_args("coordinator", **common)
    explicit = agent_from_args(
        "coordinator",
        **common,
        rule_projection_and_recent_changes_enabled=True,
    )
    assert isinstance(implicit, CoordinatorAgent)
    assert isinstance(explicit, CoordinatorAgent)
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 3, "room_type": "Monster"},
        "player": {"hp": 40, "max_hp": 70, "energy": 3, "deck": [], "relics": [], "potions": []},
        "combat": {"hand": [], "enemies": []},
    }
    legal_actions = [
        LegalAction(
            action_id=0,
            kind="combat_end_turn",
            label="end turn",
            command={"cmd": "action", "action": "end_turn"},
        )
    ]

    assert implicit.build_prompt(state, legal_actions) == explicit.build_prompt(state, legal_actions)


def test_agent_from_args_disables_formal_and_legacy_route_plan_channels_together():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_mdp_state",
        route_plan_enabled=False,
    )

    assert isinstance(agent, CoordinatorAgent)
    assert agent.store.formal_route_plan_enabled is False
    assert agent.store.route_plan_enabled is False
    assert "route_plan" not in agent.store.allowed_plan_patch_keys()


def test_agent_from_args_rejects_short_term_memory_for_coordinator():
    with pytest.raises(ValueError, match="short_term_memory.*not supported by coordinator"):
        agent_from_args(
            "coordinator",
            base_url="http://127.0.0.1:1",
            model="test",
            api_key="local",
            short_term_memory_enabled=True,
        )


def test_agent_from_args_passes_combat_reasoning_pool_to_coordinator_ca():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_mdp_state",
        combat_reasoning_pool_enabled=True,
    )

    assert agent.ca_agent.combat_reasoning_pool_enabled is True
    assert agent.ca_agent.combat_mdp_buffer.reasoning_pool_enabled is True
    assert agent.occ_agent.inner.combat_reasoning_pool_enabled is False


def test_agent_from_args_passes_combat_turn_board_to_coordinator_ca():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_mdp_state",
        combat_turn_board_enabled=True,
    )

    assert agent.ca_agent.combat_turn_board_enabled is True
    assert agent.ca_agent.combat_mdp_buffer.turn_board_enabled is True
    assert agent.occ_agent.inner.combat_turn_board_enabled is False


def test_agent_from_args_passes_combat_skill_hints_to_coordinator_ca_only():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="combat_mdp_state",
        combat_skill_hints_enabled=True,
    )

    assert agent.ca_agent.combat_skill_hints_enabled is True
    assert agent.ca_agent.combat_mdp_buffer.skill_hints_enabled is True
    assert agent.occ_agent.inner.combat_skill_hints_enabled is False


def test_agent_from_args_passes_combat_skill_lines_to_both_coordinator_delegates():
    agent = agent_from_args(
        "coordinator",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        conversation_mode="turn_chat",
        combat_skill_lines_enabled=False,
    )

    assert agent.ca_agent.combat_skill_lines_enabled is False
    assert agent.ca_agent.turn_chat_buffer.combat_skill_lines_enabled is False
    assert agent.occ_agent.inner.combat_skill_lines_enabled is False


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
        short_term_memory_enabled = False
        short_term_memory_window = 8
        short_term_memory_mode = "factual_diff"
        run_summary = True
        context_mode = "single_turn"
        turn_chat_window = 3
        turn_chat_update_mode = "delta"
        turn_chat_assistant_history = "compact"
        turn_chat_plan = True
        route_plan_enabled = False
        route_plan_enforcement = "soft"
        route_map_prompt_format = "canonical"
        route_candidate_objective = "max_restsites"
        occ_context_mode = "room_process"
        self_consistency_enabled = True
        self_consistency_samples = 5
        self_consistency_decisions = "map_select,card_reward"
        self_consistency_mode = "majority_vote"
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
        allow_custom_sequences = False
        combat_reasoning_pool_enabled = False
        combat_turn_board_enabled = False
        out = ""
        log_level = "metrics"
        print_prompts = False
        print_model_output = False

    config = build_run_config(Args, [], ["--api-key", "secret"])

    context = config["agent"]["context_management"]
    assert config["agent"]["llm_fallback_mode"] == "error"
    assert context["mode"] == "coordinator"
    assert context["route_candidate_planner"]["objective"] == "max_restsites"
    assert config["agent"]["route_plan"]["enabled"] is False
    assert config["agent"]["route_plan"]["enforcement"] == "soft"
    assert config["agent"]["rule_projection_and_recent_changes"]["enabled"] is True
    assert context["ca"]["mode"] == "single_turn"
    assert context["ca"]["history"] == "none"
    assert context["occ"]["mode"] == "room_process"
    assert context["occ"]["room_context"]["scope"] == "current_room"
    assert context["occ"]["room_context"]["source"] == "rm_rp_slice_plus_room_process_ledger"
    assert context["occ"]["room_context"]["reset"] == "room_change_or_map"
    assert context["occ"]["reward_posterior_evaluator"] is True
    assert config["benchmark"]["allow_custom_sequences"] is False


def test_run_config_records_coordinator_combat_mdp_ca_mode():
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
        short_term_memory_enabled = False
        short_term_memory_window = 8
        short_term_memory_mode = "factual_diff"
        run_summary = True
        context_mode = "combat_mdp_state"
        turn_chat_window = 3
        turn_chat_update_mode = "delta"
        turn_chat_assistant_history = "compact"
        turn_chat_plan = False
        occ_context_mode = "room_process"
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
        allow_custom_sequences = False
        combat_reasoning_pool_enabled = False
        combat_turn_board_enabled = True
        out = ""
        log_level = "metrics"
        print_prompts = False
        print_model_output = False

    config = build_run_config(Args, [], [])
    context = config["agent"]["context_management"]

    assert context["mode"] == "coordinator"
    assert context["ca"]["mode"] == "combat_mdp_state"
    assert context["ca"]["turn_process"] == "structured_current_turn_actions"
    assert context["ca"]["combat_turn_board"] is True
