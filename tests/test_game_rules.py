from __future__ import annotations

from sts2_bench.actions import LegalAction
from sts2_bench.agents import (
    PromptContract,
    build_combat_mdp_user_message,
    build_llm_prompt,
    build_turn_chat_system_message,
)
from sts2_bench.context import render_state_text
from sts2_bench.game_rules import (
    GAME_RULE_MODULES,
    GAME_RULESET_ID,
    game_rules_identity,
    game_rules_meta,
    render_game_rules,
    render_full_game_rules,
    rule_module_ids_for_state,
)


def _state(decision: str, room_type: str = "Monster", **extra):
    return {
        "type": "decision",
        "decision": decision,
        "context": {"act": 1, "floor": 1, "room_type": room_type},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck_size": 10},
        **extra,
    }


def _action() -> LegalAction:
    return LegalAction(0, "proceed", {"cmd": "action", "action": "proceed"}, "proceed")


def test_decisions_receive_only_their_rule_modules():
    combat = rule_module_ids_for_state(_state("combat_play", enemies=[], hand=[]))
    map_modules = rule_module_ids_for_state(_state("map_select", choices=[]))
    crystal = rule_module_ids_for_state(_state("crystal_sphere", "Event"))
    fake_shop = rule_module_ids_for_state(_state("fake_merchant_shop", "Event"))

    assert combat == ("core_run", "persistent_resources", "combat_cycle")
    assert "map_navigation" not in combat and "merchant" not in combat
    assert map_modules == (
        "core_run",
        "persistent_resources",
        "map_navigation",
        "room_glossary",
        "act_transition",
    )
    assert "combat_cycle" not in map_modules
    assert crystal == ("core_run", "persistent_resources", "event", "crystal_sphere")
    assert fake_shop == ("core_run", "persistent_resources", "event")
    assert "merchant" not in fake_shop


def test_boss_and_reward_rules_are_conditionally_scoped():
    assert "act_transition" in rule_module_ids_for_state(_state("combat_play", "Boss", enemies=[], hand=[]))
    assert "act_transition" in rule_module_ids_for_state(_state("combat_reward", "Boss", rewards=[]))
    assert "act_transition" not in rule_module_ids_for_state(_state("combat_reward", "Monster", rewards=[]))
    assert rule_module_ids_for_state(_state("event_result", "Event")) == ("core_run",)
    for decision in ("game_over", "unrecognized_state", "unknown", "event_blocked"):
        assert rule_module_ids_for_state(_state(decision)) == ()


def test_card_select_inherits_public_source_and_marks_noncombat_selection_as_persistent():
    combat = _state("card_select", combat={"round": 1})
    event = _state("card_select", "Event", source_event_option={"id": "OPTION"})
    rest = _state("card_select", "RestSite", source_room_option={"id": "SMITH"})
    merchant = _state("card_select", "Shop", source_room_option={"id": "REMOVE"})
    potion = _state("card_select", "Event", source_potion={"id": "POTION"})

    assert "combat_cycle" in rule_module_ids_for_state(combat)
    assert "event" in rule_module_ids_for_state(event)
    assert "rest_site" in rule_module_ids_for_state(rest)
    assert "merchant" in rule_module_ids_for_state(merchant)
    assert "event" in rule_module_ids_for_state(potion)
    assert "deckbuilding_choice" not in rule_module_ids_for_state(combat)
    for state in (event, rest, merchant, potion):
        assert "deckbuilding_choice" in rule_module_ids_for_state(state)


def test_treasure_and_card_reward_use_public_provenance_modules():
    treasure = rule_module_ids_for_state(_state("treasure", "Treasure", relics=[]))
    combat_reward = rule_module_ids_for_state(_state("card_reward", "Monster", cards=[]))
    event_reward = rule_module_ids_for_state(
        _state("card_reward", "Monster", cards=[], source_event_option={"id": "OPTION"})
    )
    ancient_reward = rule_module_ids_for_state(
        _state("card_reward", "Ancient", cards=[], from_event=True)
    )
    rest_reward = rule_module_ids_for_state(
        _state("card_reward", "RestSite", cards=[], source_room_option={"id": "OPTION"})
    )

    assert treasure == ("core_run", "persistent_resources", "treasure")
    assert "post_combat_rewards" in combat_reward
    assert "event" in event_reward and "post_combat_rewards" not in event_reward
    assert "act_transition" in ancient_reward
    assert "event" in ancient_reward and "post_combat_rewards" not in ancient_reward
    assert "rest_site" in rest_reward and "post_combat_rewards" not in rest_reward


def test_bundle_selection_does_not_hardcode_event_source():
    event_modules = rule_module_ids_for_state(_state("bundle_select", "Event", bundles=[]))
    ancient_modules = rule_module_ids_for_state(_state("bundle_select", "Ancient", bundles=[]))

    assert "deckbuilding_choice" in event_modules and "event" in event_modules
    assert "deckbuilding_choice" in ancient_modules and "event" not in ancient_modules


def test_rule_identity_and_render_are_stable_and_nonduplicated():
    identity = game_rules_identity()
    text = render_full_game_rules()

    assert identity["ruleset_id"] == GAME_RULESET_ID
    assert identity["module_ids"] == list(GAME_RULE_MODULES)
    assert len(identity["ruleset_sha256"]) == 64
    for module_id in GAME_RULE_MODULES:
        assert text.count(f"[{module_id}]") == 1


def test_formal_prompt_shapes_receive_the_same_state_selected_rule_block():
    state = _state(
        "combat_play",
        enemies=[],
        hand=[],
        round=1,
        energy=3,
        max_energy=3,
        draw_pile_count=5,
        discard_pile_count=0,
        exhaust_pile_count=0,
    )
    actions = [_action()]
    single = build_llm_prompt(state, actions, include_json=False)
    mdp = build_combat_mdp_user_message(state, actions, actions_this_turn=[])
    system = build_turn_chat_system_message("default", state=state, contract=PromptContract())

    for prompt in (single, mdp, system):
        assert "Game rules for the current decision:" in prompt
        assert "[combat_cycle]" in prompt
        assert "Power cards enter play and establish their described persistent combat effect." in prompt
        assert "A number inside such a trigger is not immediate damage or Block" in prompt
        assert "[map_navigation]" not in prompt
        assert prompt.index("You are playing Slay the Spire 2") < prompt.index("Game rules for the current decision:")


def test_card_type_timing_rules_are_combat_scoped_and_action_adjacent():
    combat_state = _state(
        "combat_play",
        enemies=[],
        hand=[],
        round=1,
        energy=3,
        max_energy=3,
        draw_pile_count=5,
        discard_pile_count=0,
        exhaust_pile_count=0,
    )
    combat_prompt = build_combat_mdp_user_message(combat_state, [_action()], actions_this_turn=[])
    map_prompt = build_llm_prompt(_state("map_select", choices=[]), [_action()], include_json=False)

    timing_rule = '"At the start...", "At the end...", and "Whenever..." clauses are delayed triggers.'
    rendered_rules = render_game_rules(combat_state)
    assert combat_prompt.count(timing_rule) == 1
    assert combat_prompt.index(timing_rule) < combat_prompt.index("Legal actions:")
    assert timing_rule not in map_prompt
    assert "Inferno" not in rendered_rules


def test_card_reward_opening_protocol_is_decision_scoped():
    reward = _state("combat_reward", rewards=[])
    combat = _state("combat_play", enemies=[], hand=[])

    reward_prompt = build_llm_prompt(reward, [_action()], include_json=False)
    combat_prompt = build_llm_prompt(combat, [_action()], include_json=False)

    assert "Decision-specific interaction protocol:" in reward_prompt
    assert "unopened reward pack" in reward_prompt
    assert "open it first" not in reward_prompt
    assert "unopened reward pack" not in combat_prompt
    assert "open it first" not in combat_prompt


def test_shared_context_renders_unknown_and_missing_decision_state():
    map_prompt = render_state_text(
        _state(
            "map_select",
            choices=[],
            full_map={
                "type": "map",
                "current_coord": {"col": 0, "row": 0},
                "rows": [[{"col": 0, "row": 0, "type": "Ancient", "children": []}]],
                "boss": {"col": 0, "row": 0, "type": "Boss"},
            },
        )
    )
    crystal_prompt = render_state_text(
        _state(
            "crystal_sphere",
            "Event",
            event_name="Crystal Sphere",
            grid_width=3,
            grid_height=2,
            divinations_remaining=2,
            tool="small",
            can_use_big_tool=True,
            can_use_small_tool=True,
            can_proceed=False,
            cells=[{"x": 0, "y": 0, "is_hidden": False}],
            clickable_cells=[{"x": 1, "y": 0}],
            visible_items=[],
            revealed_items=[],
        )
    )
    event_result_prompt = render_state_text(
        _state(
            "event_result",
            "Event",
            event_id="TEST_EVENT",
            event_name="Test Event",
            description="Ended.",
            vars={"gold": 20},
            options=[{"index": 0, "title": "Proceed"}],
            can_proceed=True,
        )
    )

    assert "?=Unknown" in map_prompt and "?=Event" not in map_prompt
    assert "divinations_remaining=2" in crystal_prompt
    assert 'Crystal Sphere clickable_cells: [{"x":1,"y":0}]' in crystal_prompt
    assert "Event id: TEST_EVENT" in event_result_prompt
    assert "Can proceed: True" in event_result_prompt


def test_rule_meta_binds_selected_render_to_ruleset():
    state = _state("map_select", choices=[])
    meta = game_rules_meta(state)

    assert meta["game_ruleset_id"] == GAME_RULESET_ID
    assert meta["game_ruleset_sha256"] == game_rules_identity()["ruleset_sha256"]
    assert meta["game_rule_module_ids"] == list(rule_module_ids_for_state(state))
    assert len(meta["game_rule_render_sha256"]) == 64
    assert meta["game_rule_delivery"] == "state_selected_prompt"


def test_rule_meta_audits_the_actual_decision_prompt_exactly_once():
    state = _state("map_select", choices=[])
    rendered = render_game_rules(state)
    prompt = "header\n\n" + rendered + "\n\nstate"

    valid = game_rules_meta(state, prompt=prompt)
    missing = game_rules_meta(state, prompt="header\n\nstate")
    duplicated = game_rules_meta(state, prompt=prompt + "\n\n" + rendered)

    assert valid["game_rule_expected_block_sha256"] == valid["game_rule_render_sha256"]
    assert len(valid["game_rule_prompt_sha256"]) == 64
    assert valid["game_rule_occurrences"] == 1
    assert valid["game_rule_expected_occurrences"] == 1
    assert valid["game_rule_marker_occurrences"] == 1
    assert valid["game_rule_expected_marker_occurrences"] == 1
    assert valid["game_rule_exposure_valid"] is True
    assert missing["game_rule_occurrences"] == 0
    assert missing["game_rule_exposure_valid"] is False
    assert duplicated["game_rule_occurrences"] == 2
    assert duplicated["game_rule_exposure_valid"] is False

    mutated_duplicate = game_rules_meta(
        state,
        prompt=prompt + "\n\nGame rules for the current decision:\n[unexpected]\n- corrupted",
    )
    assert mutated_duplicate["game_rule_occurrences"] == 1
    assert mutated_duplicate["game_rule_marker_occurrences"] == 2
    assert mutated_duplicate["game_rule_exposure_valid"] is False


def test_rule_meta_requires_zero_rule_blocks_for_no_rule_decision():
    state = _state("game_over")

    valid = game_rules_meta(state, prompt="terminal state")
    invalid = game_rules_meta(
        state,
        prompt="Game rules for the current decision:\nOnly the modules listed below apply to this prompt.",
    )

    assert valid["game_rule_occurrences"] == 0
    assert valid["game_rule_expected_occurrences"] == 0
    assert valid["game_rule_marker_occurrences"] == 0
    assert valid["game_rule_expected_marker_occurrences"] == 0
    assert valid["game_rule_exposure_valid"] is True
    assert invalid["game_rule_occurrences"] == 1
    assert invalid["game_rule_exposure_valid"] is False


def test_paired_context_switch_omits_rule_projection_and_recent_changes_only():
    state = _state(
        "combat_play",
        energy=2,
        max_energy=3,
        hand=[],
        enemies=[],
        last_action_result={
            "action_label": "play Defend",
            "action_kind": "combat_card",
            "changes": ["block: 0 -> 5"],
        },
    )
    prompt = build_llm_prompt(
        state,
        [_action()],
        include_json=True,
        rule_projection_and_recent_changes_enabled=False,
    )
    audit = game_rules_meta(state, prompt=prompt, projection_enabled=False)

    assert "Game rules for the current decision:" not in prompt
    assert "Last action result:" not in prompt
    assert '"last_action_result"' not in prompt
    assert "Decision: combat_play" in prompt
    assert "Legal actions:" in prompt
    assert audit["game_rule_delivery"] == "omitted"
    assert audit["game_rule_projection_enabled"] is False
    assert audit["game_rule_expected_occurrences"] == 0
    assert audit["game_rule_exposure_valid"] is True


def test_paired_context_switch_omits_route_recent_state_changes_without_placeholder():
    state = _state(
        "map_select",
        choices=[{"col": 0, "row": 1, "type": "Monster"}],
        last_action_result={"changes": ["hp: 40/80 -> 35/80"]},
    )
    action = LegalAction(
        0,
        "map",
        {"cmd": "action", "action": "select_map_node", "args": {"col": 0, "row": 1}},
        "go to Monster",
    )
    prompt = build_llm_prompt(
        state,
        [action],
        route_plan_enabled=False,
        rule_projection_and_recent_changes_enabled=False,
    )

    assert '"recent_state_changes"' not in prompt
    assert "Game rules for the current decision:" not in prompt
    assert '"choices":[{"col":0,"row":1,"type":"Monster"}]' in prompt
