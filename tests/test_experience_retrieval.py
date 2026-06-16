import json

from sts2_bench.actions import LegalAction
from sts2_bench.experience.agent import ExperienceMemoryRetriever


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _action(action_id, label, kind="combat_play_card"):
    return LegalAction(
        action_id=action_id,
        label=label,
        kind=kind,
        command={"cmd": "action", "action": "play_card", "args": {"card_index": action_id}},
    )


def _lagavulin_state(*, hp=9, round_number=8, intent_damage=14, hand=None):
    if hand is None:
        hand = [
            {"index": 0, "name": "Strike", "type": "Attack", "cost": 1, "can_play": True, "stats": {"damage": 7}},
            {
                "index": 1,
                "name": "Setup Strike",
                "type": "Attack",
                "cost": 1,
                "can_play": True,
                "stats": {"damage": 8, "strengthpower": 2},
            },
            {
                "index": 2,
                "name": "Shrug It Off",
                "type": "Skill",
                "cost": 1,
                "can_play": True,
                "stats": {"block": 6, "cards": 1},
            },
        ]
    return {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 17, "room_type": "Boss"},
        "act": 1,
        "floor": 17,
        "round": round_number,
        "energy": 1,
        "max_energy": 3,
        "player": {"hp": hp, "max_hp": 90, "deck_size": 23, "relics": [], "potions": []},
        "player_powers": [{"name": "Dexterity", "amount": -2, "type": "Debuff"}],
        "enemies": [
            {
                "index": 0,
                "name": "Lagavulin Matriarch",
                "hp": 67,
                "max_hp": 222,
                "block": 0,
                "intents": [{"type": "Attack", "damage": intent_damage}],
                "powers": [{"name": "Strength", "amount": 2, "type": "Buff"}],
            }
        ],
        "hand": hand,
    }


def test_retrieval_filters_out_timing_and_boundary_mismatches(tmp_path):
    store = tmp_path / "active.jsonl"
    _write_jsonl(
        store,
        [
            {
                "id": "first_turn_lagavulin",
                "status": "active",
                "memory_type": "rule",
                "scope": {"level": "combat", "character": "Ironclad", "act": 1, "enemy": "Lagavulin Matriarch", "phase": "first_turn"},
                "trigger": "when facing a stunned Lagavulin on the first turn",
                "lesson": "Use the free damage window.",
                "recommendation": "Play all attacks while the enemy is stunned.",
                "counterfactual": "Not applicable if the enemy is not stunned.",
                "quality": {"confidence": 0.9},
                "retrieval_hints": {"keywords": ["lagavulin", "stun", "first_turn"], "priority": 0.8},
            },
            {
                "id": "bloodletting_high_hp",
                "status": "active",
                "memory_type": "risk",
                "scope": {"level": "combat", "character": "Ironclad", "act": 1, "enemy": "Lagavulin Matriarch", "phase": "combat_play"},
                "trigger": "when considering using Bloodletting against Lagavulin while HP is above 50%",
                "lesson": "Bloodletting can convert safe HP into energy.",
                "recommendation": "Use Bloodletting when you have 3+ energy worth of attacks in hand.",
                "counterfactual": "Do not use if HP is below 50% or if you lack sufficient attacks.",
                "quality": {"confidence": 0.85},
                "retrieval_hints": {"keywords": ["bloodletting", "lagavulin"], "priority": 0.7},
            },
            {
                "id": "global_high_hp_block",
                "status": "active",
                "memory_type": "rule",
                "scope": {"level": "global", "character": "Ironclad", "act": 1, "enemy": "Living Fog", "phase": "combat_play"},
                "trigger": "When deciding whether to play a block card early in combat with high HP",
                "lesson": "With high HP above 75%, damage may beat over-blocking.",
                "recommendation": "Avoid over-blocking early when HP is high.",
                "counterfactual": "Do not apply if enemy has a large attack incoming or if you have low HP.",
                "quality": {"confidence": 0.85},
                "retrieval_hints": {"keywords": ["block", "high_hp", "early_combat"], "priority": 0.7},
            },
        ],
    )
    retriever = ExperienceMemoryRetriever(store_path=store)

    result = retriever.retrieve(
        _lagavulin_state(),
        [
            _action(5, "play card 0: Strike on enemy 0: Lagavulin Matriarch"),
            _action(6, "play card 1: Setup Strike on enemy 0: Lagavulin Matriarch"),
            _action(7, "play card 2: Shrug It Off"),
        ],
        character="Ironclad",
        ascension=0,
    )

    assert result.rendered_ids == []


def test_retrieval_allows_first_turn_stun_memory_only_when_state_matches(tmp_path):
    store = tmp_path / "active.jsonl"
    _write_jsonl(
        store,
        [
            {
                "id": "first_turn_stun",
                "status": "active",
                "memory_type": "rule",
                "scope": {"level": "combat", "character": "Ironclad", "act": 1, "enemy": "Lagavulin Matriarch", "phase": "first_turn"},
                "trigger": "when facing a stunned Lagavulin on the first turn",
                "lesson": "Use the free damage window.",
                "recommendation": "Play high damage attacks during the stun turn.",
                "counterfactual": "Not applicable if the enemy is not stunned.",
                "quality": {"confidence": 0.9},
                "retrieval_hints": {"keywords": ["lagavulin", "stun", "first_turn"], "priority": 0.8},
            }
        ],
    )
    retriever = ExperienceMemoryRetriever(store_path=store)
    state = _lagavulin_state(round_number=1, intent_damage=0)
    state["enemies"][0]["intents"] = [{"type": "Stun"}]

    result = retriever.retrieve(
        state,
        [_action(5, "play card 0: Strike on enemy 0: Lagavulin Matriarch")],
        character="Ironclad",
        ascension=0,
    )

    assert result.rendered_ids == ["first_turn_stun"]


def test_global_retrieval_requires_specific_context_overlap(tmp_path):
    store = tmp_path / "active.jsonl"
    _write_jsonl(
        store,
        [
            {
                "id": "unrelated_global",
                "status": "active",
                "memory_type": "rule",
                "scope": {"level": "global", "character": "Ironclad", "act": 1, "phase": "combat_play"},
                "trigger": "When using a very specific relic line.",
                "lesson": "Only relevant with an unrelated relic.",
                "recommendation": "Use the unrelated relic line.",
                "quality": {"confidence": 0.95},
                "retrieval_hints": {"keywords": ["unrelated_relic_line"], "priority": 1.0},
            }
        ],
    )
    retriever = ExperienceMemoryRetriever(store_path=store)

    result = retriever.retrieve(
        _lagavulin_state(),
        [_action(7, "play card 2: Shrug It Off")],
        character="Ironclad",
        ascension=0,
    )

    assert result.rendered_ids == []
