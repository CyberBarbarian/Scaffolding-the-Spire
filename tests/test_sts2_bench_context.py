from sts2_bench.context import build_last_action_result, combat_facts, combat_skill_hints, compact_state, render_full_map, render_state_text


def test_combat_facts_never_export_attack_only_hp_projection():
    state = {
        "decision": "combat_play",
        "player": {"hp": 18, "max_hp": 80, "block": 3},
        "enemies": [
            {
                "index": 0,
                "name": "Test Enemy",
                "intents": [{"type": "Attack", "damage": 7, "hits": 2}],
            }
        ],
        "hand": [],
    }

    assert "hp_if_end_turn_now" not in combat_facts(state)
    assert "hp_if_end_turn_now" not in render_state_text(state)


def test_view_deck_uses_full_card_rendering():
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 1, "room_type": "Monster"},
        "player": {
            "hp": 80,
            "max_hp": 80,
            "gold": 99,
            "deck_size": 2,
            "deck": [
                {
                    "name": "Strike",
                    "cost": 1,
                    "type": "Attack",
                    "rarity": "Basic",
                    "upgraded": False,
                    "stats": {"damage": 6},
                    "target_type": "AnyEnemy",
                    "description": "Deal 6 damage.",
                    "after_upgrade": {
                        "cost": 1,
                        "stats": {"damage": 9},
                        "description": "Deal 9 damage.",
                    },
                    "hover_tips": [
                        {
                            "name": "Attack",
                            "description": "Attacks are played against enemies.",
                        }
                    ],
                    "count": 2,
                }
            ],
        },
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "enemies": [],
        "hand": [],
        "view_deck": True,
    }

    text = render_state_text(state)

    assert "Deck view (1 entries, deck_size=2):" in text
    assert "[0] Strike cost=1 Attack rarity=Basic upgraded=False damage=6 target=AnyEnemy x2" in text
    assert "description: Deal 6 damage." in text
    assert "upgrade: damage 6 -> 9" in text
    assert "upgrade_description: Deal 9 damage." in text
    assert "tip: Attack: Attacks are played against enemies." in text


def test_combat_card_select_restates_current_hand_and_character_mechanics():
    state = {
        "decision": "card_select",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {
            "hp": 70,
            "max_hp": 80,
            "gold": 0,
            "deck_size": 10,
            "potions": [{"index": 0, "name": "Fire Potion", "description": "Deal 20 damage."}],
        },
        "min_select": 1,
        "max_select": 1,
        "can_skip": False,
        "prompt": "Choose a card.",
        "cards": [{"index": 0, "name": "Candidate", "description": "Selectable card."}],
        "combat": {
            "round": 1,
            "energy": 2,
            "max_energy": 3,
            "draw_pile_count": 4,
            "discard_pile_count": 1,
            "exhaust_pile_count": 0,
            "player_powers": [{"name": "Strength", "amount": 2, "description": "Power mechanic."}],
            "hand": [
                {
                    "index": 0,
                    "name": "Current Hand Card",
                    "description": "Current hand mechanic.",
                    "after_upgrade": {"description": "Current hand upgraded mechanic."},
                }
            ],
            "enemies": [],
            "orbs": [],
            "orb_slots": 3,
            "stars": 1,
            "osty": {
                "name": "Osty",
                "hp": 12,
                "max_hp": 20,
                "block": 2,
                "alive": True,
                "powers": [{"name": "Osty Power", "description": "Osty mechanic."}],
            },
        },
    }

    text = render_state_text(state)

    assert "description: Deal 20 damage." in text
    assert "Strength(2): Power mechanic." in text
    assert "Current combat hand (1):" in text
    assert "description: Current hand mechanic." in text
    assert "upgrade_description: Current hand upgraded mechanic." in text
    assert "Orbs: none" in text
    assert "Orb slots: 3" in text
    assert "Stars: 1" in text
    assert "Osty powers (1):" in text
    assert "Osty Power: Osty mechanic." in text


def test_combat_skill_hints_are_state_triggered_and_rendered():
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 6, "room_type": "Elite"},
        "player": {"hp": 34, "max_hp": 80, "block": 2, "gold": 10, "deck_size": 12},
        "round": 2,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 5,
        "discard_pile_count": 2,
        "enemies": [
            {
                "index": 0,
                "name": "Cultist",
                "hp": 10,
                "max_hp": 48,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 6, "hits": 1}],
            },
            {
                "index": 1,
                "name": "Louse",
                "hp": 8,
                "max_hp": 14,
                "block": 0,
                "intents": [{"type": "Buff"}],
            },
        ],
        "hand": [
            {"index": 0, "name": "Bloodletting", "type": "Skill", "cost": 0, "description": "Lose 3 HP. Gain 2 Energy."},
            {"index": 1, "name": "Pommel Strike", "type": "Attack", "cost": 1, "stats": {"damage": 9, "draw": 1}},
        ],
    }

    hints = combat_skill_hints(state)
    text = render_state_text(state)

    assert hints == [
        "order: lethal > survival > reduce actions > damage/setup; one action",
        "survive: block/kill until safe; stop overblocking",
        "kill_now: enemy[1] hp=8 dies to one card; avoid spread/setup",
        "self_damage: only for lethal, survival, or a concrete same-turn play",
    ]
    assert "Combat skills:" in text
    assert "kill_now: enemy[1] hp=8 dies to one card" in text
    assert "self_damage: only for lethal, survival" in text


def test_combat_skill_hints_no_attack_discourage_pure_block():
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 17, "room_type": "Boss"},
        "player": {"hp": 54, "max_hp": 86, "block": 0, "gold": 143, "deck_size": 21},
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 14,
        "discard_pile_count": 0,
        "enemies": [
            {
                "index": 0,
                "name": "Lagavulin Matriarch",
                "hp": 222,
                "max_hp": 222,
                "block": 12,
                "intents": [{"type": "Sleep"}],
            }
        ],
        "hand": [{"index": 0, "name": "Defend", "type": "Skill", "cost": 1, "stats": {"block": 5}}],
    }

    hints = combat_skill_hints(state)

    assert "no_attack: skip pure block; use damage/draw/scaling" in hints


def test_combat_skill_hints_focus_lowest_enemy_when_no_kill_is_available():
    state = {
        "decision": "combat_play",
        "context": {"act": 2, "floor": 6, "room_type": "Monster"},
        "player": {"hp": 60, "max_hp": 93, "block": 0, "gold": 41, "deck_size": 28},
        "round": 3,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 12,
        "discard_pile_count": 7,
        "enemies": [
            {"index": 0, "name": "Myte", "hp": 48, "max_hp": 66, "block": 0, "intents": [{"type": "Attack", "damage": 7}]},
            {"index": 1, "name": "Myte", "hp": 23, "max_hp": 66, "block": 0, "intents": [{"type": "Attack", "damage": 7}]},
        ],
        "hand": [
            {"index": 0, "name": "Strike", "type": "Attack", "cost": 1, "stats": {"damage": 9}, "target_type": "AnyEnemy"},
            {"index": 1, "name": "Defend", "type": "Skill", "cost": 1, "stats": {"block": 5}},
        ],
    }

    hints = combat_skill_hints(state)

    assert "focus: enemy[1] lowest hp=23; avoid spreading" in hints


def test_combat_power_descriptions_are_rendered_without_target_damage_preview():
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 17, "room_type": "Boss"},
        "player": {"hp": 28, "max_hp": 80, "gold": 203, "deck_size": 21},
        "round": 6,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 1,
        "discard_pile_count": 7,
        "player_powers": [
            {
                "name": "Weak",
                "amount": 2,
                "type": "Debuff",
                "description": "Attacks deal 25% less damage.",
            },
            {
                "name": "Strength",
                "amount": -4,
                "type": "Buff",
                "description": "Strength adds additional damage to Attacks.",
            },
            {
                "name": "Dexterity",
                "amount": -2,
                "type": "Buff",
                "description": "Dexterity improves Block gained from cards.",
            },
        ],
        "enemies": [
            {
                "index": 0,
                "name": "Vantom",
                "hp": 35,
                "max_hp": 140,
                "block": 0,
                "move_name": "Attack",
                "intents": [{"type": "Attack", "damage": 27, "hits": 1}],
                "powers": [
                    {
                        "name": "Slippery",
                        "amount": 6,
                        "type": "Buff",
                        "description": "The next 6 times Vantom loses HP, it only loses 1 HP instead.",
                    }
                ],
            }
        ],
        "hand": [
            {
                "index": 0,
                "name": "Bash",
                "cost": 2,
                "type": "Attack",
                "can_play": True,
                "target_type": "AnyEnemy",
                "stats": {"damage": 12, "damage_by_target": {"0": 1}},
                "description": "Deal 12 damage. Apply 2 Vulnerable.",
            }
        ],
    }

    text = render_state_text(state)

    assert "Player powers (3):" in text
    assert "Debuff Weak(2): Attacks deal 25% less damage." in text
    assert "Debuff Strength(-4): Decreases attack damage by 4." in text
    assert "Debuff Dexterity(-2): Reduces Block gained from cards by 2." in text
    assert "powers=Slippery(6)" in text
    assert "Buff Slippery(6): The next 6 times Vantom loses HP, it only loses 1 HP instead." in text
    assert "damage_by_target" not in text
    assert "target_damage" not in text


def test_view_draw_pile_uses_card_rendering_and_compact_state_keeps_requested_pile():
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {"hp": 70, "max_hp": 80, "gold": 12, "deck_size": 10},
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 1,
        "discard_pile_count": 0,
        "enemies": [],
        "hand": [],
        "draw_pile": [
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "stats": {"damage": 6},
                "description": "Deal 6 damage.",
            }
        ],
        "view_draw_pile": True,
    }

    text = render_state_text(state)
    compact = compact_state(state)

    assert "Viewed draw pile:" in text
    assert "Draw pile (1):" in text
    assert "[0] Strike cost=1 Attack damage=6" in text
    assert "description: Deal 6 damage." in text
    assert "draw_pile" not in compact
    assert compact["viewed_draw_pile"]["count"] == 1
    assert compact["viewed_draw_pile"]["cards"][0]["name"] == "Strike"
    assert "draw_pile" in compact["viewed"]


def test_bundle_select_renders_visible_card_pack_details():
    state = {
        "decision": "bundle_select",
        "context": {"act": 1, "floor": 1, "room_type": "Event"},
        "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 11},
        "bundles": [
            {
                "index": 0,
                "cards": [
                    {
                        "name": "Setup Strike",
                        "cost": 1,
                        "type": "Attack",
                        "rarity": "Common",
                        "stats": {"damage": 7, "strengthpower": 2},
                        "description": "Deal 7 damage.\nGain 2 Strength this turn.",
                        "after_upgrade": {
                            "cost": 1,
                            "stats": {"damage": 10, "strengthpower": 2},
                            "description": "Deal 10 damage.\nGain 2 Strength this turn.",
                        },
                    }
                ],
            },
            {
                "index": 1,
                "cards": [
                    {
                        "name": "Shrug It Off",
                        "cost": 1,
                        "type": "Skill",
                        "rarity": "Common",
                        "stats": {"block": 8, "cards": 1},
                        "description": "Gain 8 Block.\nDraw 1 card.",
                    }
                ],
            },
        ],
    }

    text = render_state_text(state)

    assert "Card packs (2):" in text
    assert "Pack [0]:" in text
    assert "[0] Setup Strike cost=1 Attack rarity=Common damage=7 strengthpower=2" in text
    assert "description: Deal 7 damage. Gain 2 Strength this turn." in text
    assert "upgrade: damage 7 -> 10" in text
    assert "upgrade_description: Deal 10 damage. Gain 2 Strength this turn." in text
    assert "Pack [1]:" in text
    assert "[0] Shrug It Off cost=1 Skill rarity=Common block=8 cards=1" in text
    assert "description: Gain 8 Block. Draw 1 card." in text


def test_combat_text_renders_all_public_character_mechanics():
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {"hp": 70, "max_hp": 80, "block": 0, "gold": 0, "deck_size": 10},
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 5,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
        "hand": [],
        "enemies": [],
        "orbs": [
            {
                "index": 0,
                "name": "Lightning",
                "passive": 3,
                "evoke": 8,
                "position_label": "rightmost",
            }
        ],
        "orb_slots": 3,
        "stars": 2,
        "osty": {
            "name": "Osty",
            "hp": 18,
            "max_hp": 24,
            "block": 4,
            "alive": True,
        },
    }

    text = render_state_text(state)

    assert "Orb slots: 3" in text
    assert "position=rightmost" in text
    assert "Stars: 2" in text
    assert "Osty: Osty hp=18/24 block=4 alive=True" in text


def test_event_choice_text_preserves_event_vars_and_structured_option_identity():
    state = {
        "decision": "event_choice",
        "context": {"act": 1, "floor": 4, "room_type": "Event"},
        "player": {"hp": 70, "max_hp": 80, "gold": 20, "deck_size": 10},
        "event_name": "Relic Trader",
        "description": "Trade one relic for another.",
        "vars": {"gold_cost": 20},
        "options": [
            {
                "index": 0,
                "title": "Trade",
                "description": "Lose Anchor. Gain Lantern.",
                "text_key": "RELIC_TRADE",
                "relic_trade": {
                    "lose": {"id": "RELIC.ANCHOR", "name": "Anchor"},
                    "gain": {"id": "RELIC.LANTERN", "name": "Lantern"},
                },
            }
        ],
    }

    text = render_state_text(state)

    assert 'Event vars: {"gold_cost":20}' in text
    assert "text_key=RELIC_TRADE" in text
    assert 'relic_trade={"gain":{"id":"RELIC.LANTERN","name":"Lantern"},"lose":{"id":"RELIC.ANCHOR","name":"Anchor"}}' in text


def test_fake_merchant_text_preserves_event_context_and_foul_potion_availability():
    state = {
        "decision": "fake_merchant_shop",
        "context": {"act": 1, "floor": 5, "room_type": "Event"},
        "player": {"hp": 70, "max_hp": 80, "gold": 100, "deck_size": 10},
        "event_name": "Fake Merchant",
        "description": "The merchant's wares look suspicious.",
        "vars": {"discount": 25},
        "relics": [],
        "can_leave": True,
        "can_throw_foul_potion": True,
        "foul_potion_action": {"action": "use_potion", "potion_index": 2},
    }

    text = render_state_text(state)

    assert "Fake merchant event: Fake Merchant" in text
    assert "Description: The merchant's wares look suspicious." in text
    assert 'Event vars: {"discount":25}' in text
    assert "can_throw_foul_potion=True" in text
    assert '"potion_index":2' in text


def test_standard_shop_text_preserves_engine_affordances():
    state = {
        "decision": "shop",
        "player": {"hp": 70, "max_hp": 80, "gold": 100, "deck_size": 10},
        "cards": [
            {"index": 0, "name": "Card", "price": 50, "is_stocked": True, "can_buy": True},
        ],
        "relics": [
            {"index": 1, "name": "Relic", "cost": 120, "is_stocked": True, "can_buy": False},
        ],
        "potions": [
            {
                "index": 2,
                "name": "Potion",
                "cost": 50,
                "is_stocked": True,
                "can_buy": False,
                "target_type": "None",
            },
        ],
        "card_removal_cost": 75,
        "can_remove_card": True,
    }

    text = render_state_text(state)

    assert "[0] Card price=50 is_stocked=True can_buy=True" in text
    assert "[1] Relic cost=120 is_stocked=True can_buy=False" in text
    assert "[2] Potion cost=50 is_stocked=True can_buy=False" in text
    assert "Card removal cost: 75 can_remove_card=True" in text


def test_card_reward_text_does_not_invent_missing_skip_affordance():
    text = render_state_text(
        {
            "decision": "card_reward",
            "player": {"hp": 70, "max_hp": 80, "gold": 100, "deck_size": 10},
            "cards": [{"index": 0, "name": "Strike"}],
        }
    )

    assert "Can skip: None" in text
    assert "Can skip: True" not in text


def test_view_discard_pile_with_hidden_details_matches_cli_message():
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 12, "deck_size": 10},
        "choices": [],
        "discard_pile_count": 4,
        "view_discard_pile": True,
    }

    text = render_state_text(state)

    assert "Viewed discard pile:" in text
    assert "Discard pile (4):" in text
    assert "Card details are only available during combat." in text


def test_full_map_render_keeps_ascii_separate_from_route_graph_json():
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 1, "room_type": "Map"},
        "player": {"hp": 72, "max_hp": 80, "gold": 30, "deck_size": 10},
        "choices": [
            {"col": 1, "row": 1, "type": "Monster"},
            {"col": 3, "row": 1, "type": "Monster"},
        ],
        "full_map": {
            "type": "map",
            "current_coord": {"col": 3, "row": 0},
            "rows": [
                [
                    {
                        "col": 3,
                        "row": 0,
                        "type": "Ancient",
                        "children": [{"col": 1, "row": 1}, {"col": 3, "row": 1}],
                    }
                ],
                [
                    {"col": 1, "row": 1, "type": "Monster", "children": [{"col": 2, "row": 2}]},
                    {"col": 3, "row": 1, "type": "Monster", "children": [{"col": 4, "row": 2}]},
                ],
                [
                    {"col": 2, "row": 2, "type": "RestSite", "children": [{"col": 3, "row": 3}]},
                    {"col": 4, "row": 2, "type": "Elite", "children": [{"col": 3, "row": 3}]},
                ],
                [{"col": 3, "row": 3, "type": "Boss", "children": []}],
            ],
            "boss": {"col": 3, "row": 3, "type": "Boss"},
        },
        "view_map": True,
    }

    text = render_state_text(state)
    ascii_map = "\n".join(render_full_map(state["full_map"], state["choices"]))
    compact = compact_state(state)

    assert "Full map:" in text
    assert "Route graph exact nodes:" not in text
    assert "Route graph exact nodes:" not in ascii_map
    assert "route_graph" in compact
    assert compact["route_graph"]["legal_starts"][0]["node_key"] == "1:1"
    assert compact["route_graph"]["nodes"][1]["children"] == [
        {"col": 2, "row": 2, "type": "RestSite", "node_key": "2:2"}
    ]


def test_last_action_result_is_rendered_as_transition_diff():
    old_state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 6, "room_type": "Elite"},
        "player": {"hp": 60, "max_hp": 80, "gold": 10, "deck_size": 10},
        "round": 2,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 2,
        "discard_pile_count": 3,
        "exhaust_pile_count": 0,
        "hand": [{"index": 0, "name": "Strike"}],
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 20, "max_hp": 40, "block": 0}],
    }
    new_state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 6, "room_type": "Elite"},
        "player": {"hp": 60, "max_hp": 80, "gold": 10, "deck_size": 10},
        "round": 2,
        "energy": 2,
        "max_energy": 3,
        "draw_pile_count": 2,
        "discard_pile_count": 3,
        "exhaust_pile_count": 1,
        "hand": [],
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 14, "max_hp": 40, "block": 0}],
    }
    state = {
        **new_state,
        "last_action_result": build_last_action_result(
            old_state,
            new_state,
            action_label="play card 0: Strike on enemy 0: Jaw Worm",
            action_kind="combat_play_card",
        ),
    }

    text = render_state_text(state)

    assert "Last game action result:" in text
    assert "action: play card 0: Strike on enemy 0: Jaw Worm" in text
    assert "kind: combat_play_card" in text
    assert "decision: combat_play -> combat_play" in text
    assert "energy: 3/3 -> 2/3" in text
    assert "exhaust_pile: 0 -> 1" in text
    assert "hand_count: 1 -> 0" in text
    assert "enemy Jaw Worm[0]: hp 20/40 -> 14/40" in text


def test_last_action_result_tracks_surviving_enemy_by_instance_id_after_reindex():
    old_state = {
        "decision": "combat_play",
        "round": 4,
        "energy": 1,
        "max_energy": 3,
        "player": {"hp": 2, "max_hp": 80},
        "hand": [{"index": 0, "name": "Strike"}],
        "enemies": [
            {
                "index": 0,
                "combat_index": 0,
                "instance_id": 14,
                "name": "Leaf Slime (M)",
                "hp": 1,
                "max_hp": 32,
                "block": 0,
            },
            {
                "index": 1,
                "combat_index": 1,
                "instance_id": 15,
                "name": "Flyconid",
                "hp": 26,
                "max_hp": 47,
                "block": 0,
            },
        ],
    }
    new_state = {
        "decision": "combat_play",
        "round": 4,
        "energy": 0,
        "max_energy": 3,
        "player": {"hp": 2, "max_hp": 80},
        "hand": [],
        "enemies": [
            {
                "index": 0,
                "combat_index": 0,
                "instance_id": 15,
                "name": "Flyconid",
                "hp": 20,
                "max_hp": 47,
                "block": 0,
            }
        ],
    }

    result = build_last_action_result(
        old_state,
        new_state,
        action_label="play card 0: Strike on enemy 0: Leaf Slime (M)",
        action_kind="combat_play_card",
    )

    assert "enemy Leaf Slime (M)[0]: defeated" in result["changes"]
    assert "enemy Flyconid[1]: defeated" not in result["changes"]
    assert "enemy Flyconid[1]: hp 26/47 -> 20/47" in result["changes"]


def test_last_action_result_distinguishes_same_named_enemies_by_instance_id():
    old_state = {
        "decision": "combat_play",
        "round": 2,
        "energy": 1,
        "max_energy": 3,
        "player": {"hp": 20, "max_hp": 80},
        "hand": [{"index": 0, "name": "Strike"}],
        "enemies": [
            {
                "index": 0,
                "instance_id": 21,
                "name": "Wriggler",
                "hp": 1,
                "max_hp": 18,
                "block": 0,
            },
            {
                "index": 1,
                "instance_id": 22,
                "name": "Wriggler",
                "hp": 14,
                "max_hp": 18,
                "block": 0,
            },
        ],
    }
    new_state = {
        "decision": "combat_play",
        "round": 2,
        "energy": 0,
        "max_energy": 3,
        "player": {"hp": 20, "max_hp": 80},
        "hand": [],
        "enemies": [
            {
                "index": 0,
                "instance_id": 22,
                "name": "Wriggler",
                "hp": 14,
                "max_hp": 18,
                "block": 0,
            }
        ],
    }

    result = build_last_action_result(
        old_state,
        new_state,
        action_label="play card 0: Strike on enemy 0: Wriggler",
        action_kind="combat_play_card",
    )

    assert result["changes"].count("enemy Wriggler[0]: defeated") == 1
    assert "enemy Wriggler[1]: defeated" not in result["changes"]


def test_map_legend_treats_question_mark_as_unknown():
    full_map = {
        "type": "map",
        "current_coord": {"col": 0, "row": 0},
        "rows": [[{"col": 0, "row": 0, "type": "Unknown", "children": []}]],
        "boss": {"col": 0, "row": 0, "type": "Boss"},
    }

    text = "\n".join(render_full_map(full_map, []))

    assert "?=Unknown" in text
    assert "?=Event" not in text


def test_crystal_sphere_preserves_all_authoritative_decision_fields():
    state = {
        "decision": "crystal_sphere",
        "context": {"act": 1, "floor": 4, "room_type": "Event"},
        "player": {"hp": 70, "max_hp": 80, "gold": 20, "deck_size": 11},
        "event_name": "Crystal Sphere",
        "grid_width": 3,
        "grid_height": 2,
        "divinations_remaining": 2,
        "tool": "small",
        "can_use_big_tool": True,
        "can_use_small_tool": False,
        "can_proceed": False,
        "cells": [{"x": 0, "y": 0, "hidden": False}],
        "clickable_cells": [{"x": 1, "y": 0}],
        "visible_items": [{"x": 0, "y": 0, "kind": "gem"}],
        "revealed_items": [],
    }

    text = render_state_text(state)

    assert "width=3 height=2 divinations_remaining=2 tool=small" in text
    assert "can_use_big_tool=True can_use_small_tool=False can_proceed=False" in text
    assert 'Crystal Sphere cells: [{"hidden":false,"x":0,"y":0}]' in text
    assert 'Crystal Sphere clickable_cells: [{"x":1,"y":0}]' in text
    assert 'Crystal Sphere visible_items: [{"kind":"gem","x":0,"y":0}]' in text
    assert "Crystal Sphere revealed_items: []" in text


def test_event_result_preserves_identity_result_and_proceed_state():
    state = {
        "decision": "event_result",
        "context": {"act": 1, "floor": 4, "room_type": "Event"},
        "player": {"hp": 70, "max_hp": 80, "gold": 20, "deck_size": 11},
        "event_id": "TEST_EVENT",
        "event_name": "Test Event",
        "description": "The event is complete.",
        "vars": {"gold": 20},
        "options": [{"index": 0, "title": "Proceed"}],
        "can_proceed": True,
    }

    text = render_state_text(state)

    assert "Event result: Test Event" in text
    assert "Event id: TEST_EVENT" in text
    assert "Description: The event is complete." in text
    assert 'Event vars: {"gold":20}' in text
    assert 'Event result options: [{"index":0,"title":"Proceed"}]' in text
    assert "Can proceed: True" in text
