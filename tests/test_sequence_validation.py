from sts2_bench.sequence_validation import state_fingerprint


def test_state_fingerprint_ignores_process_local_instance_ids():
    base_state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 5, "room_type": "Monster"},
        "energy": 3,
        "max_energy": 3,
        "player": {
            "hp": 66,
            "max_hp": 80,
            "gold": 17,
            "deck_size": 14,
            "relics": [{"id": "starter", "name": "Burning Blood", "instance_id": "relic-a"}],
        },
        "hand": [
            {"index": 0, "instance_id": 12, "name": "Strike", "cost": 1, "type": "Attack", "target_type": "AnyEnemy"},
            {"index": 1, "instance_id": 13, "name": "Defend", "cost": 1, "type": "Skill", "target_type": "Self"},
        ],
        "enemies": [{"index": 0, "instance_id": 2, "name": "Shrinker Beetle", "hp": 40, "max_hp": 40}],
    }
    reloaded_state = {
        **base_state,
        "player": {
            **base_state["player"],
            "relics": [{"id": "starter", "name": "Burning Blood", "instance_id": "relic-b"}],
        },
        "hand": [
            {**base_state["hand"][0], "instance_id": 1},
            {**base_state["hand"][1], "instance_id": 2},
        ],
        "enemies": [{**base_state["enemies"][0], "instance_id": 1}],
    }

    assert state_fingerprint(base_state) == state_fingerprint(reloaded_state)
