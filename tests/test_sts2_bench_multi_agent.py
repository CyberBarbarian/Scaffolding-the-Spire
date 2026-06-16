import pytest

from sts2_bench.actions import LegalAction
from sts2_bench.agents import ModelActionError
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
    assert "route_plan: Path toward shop." in prompt
    assert "Remove Strike." in prompt
    assert "run_plan_patch" in prompt
    assert "run_memory_facts" in prompt
    assert "include run_plan_patch" in prompt
    assert "Do not leave run_plan_patch empty for a real OCC game action" in prompt
    assert "occ prompt" in prompt


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
    assert agent.ca_agent.conversation_mode == "turn_chat"
    assert agent.ca_agent.turn_chat_plan_enabled is True
    assert agent.ca_agent.fallback_mode == "error"
    assert isinstance(agent.occ_agent, OutOfCombatContextAgent)
    assert agent.occ_agent.history_window == 3
    assert agent.occ_agent.inner.fallback_mode == "error"


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
        turn_chat_plan = True
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
    assert context["ca"]["mode"] == "turn_chat"
    assert context["ca"]["turn_chat"]["window"] == 3
    assert context["occ"]["mode"] == "reference_turn"
    assert context["occ"]["reference_context"]["window"] == 3
    assert context["occ"]["reference_context"]["source"] == "rm_rp_slice_plus_short_occ_history"
