from sts2_bench.context import build_last_action_result, compact_state, render_full_map, render_state_text


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
