import json

from sts2_bench.experience.reflection import combat_spans, learn_from_trace, promote_candidates, reflect_trace


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _combat_event(step, *, action_label, enemy_hp_delta=0, decision_after="combat_play"):
    return {
        "seed": "bench_0000",
        "step": step,
        "character": "Ironclad",
        "ascension": 0,
        "act": 1,
        "floor": 1,
        "decision": "combat_play",
        "room_type": "Monster",
        "encounter": "Jaw Worm",
        "hp_bucket_before": "0.75-1.00",
        "hp_bucket_after": "0.50-0.75",
        "deck_size_bucket": "0-12",
        "action": {
            "kind": "combat_play_card",
            "label": action_label,
        },
        "outcome": {
            "decision_after": decision_after,
            "hp_delta": -4 if decision_after != "combat_play" else 0,
            "block_delta": 0,
            "energy_delta": -1,
            "enemy_hp_delta": enemy_hp_delta,
            "important_changes": ["enemy Jaw Worm hp changed"],
        },
        "agent": {
            "reason": "Compare damage with incoming attack.",
            "fallback": False,
            "retrieved_memory_ids": ["exp_old"],
            "used_memory_ids": ["exp_old"],
        },
    }


def _reward_event(step):
    return {
        "seed": "bench_0000",
        "step": step,
        "character": "Ironclad",
        "ascension": 0,
        "act": 1,
        "floor": 1,
        "decision": "combat_reward",
        "room_type": "Monster",
        "encounter": "Jaw Worm",
        "hp_bucket_before": "0.50-0.75",
        "hp_bucket_after": "0.50-0.75",
        "deck_size_bucket": "0-12",
        "action": {
            "kind": "combat_reward",
            "label": "claim reward 0: gold",
        },
        "outcome": {
            "decision_after": "map_select",
            "hp_delta": 0,
            "block_delta": 0,
            "energy_delta": None,
            "enemy_hp_delta": None,
            "important_changes": ["gold: 0 -> 12"],
        },
        "agent": {
            "reason": "Claim gold.",
            "fallback": False,
            "retrieved_memory_ids": [],
            "used_memory_ids": [],
        },
    }


def test_combat_spans_exclude_post_combat_reward_events():
    spans = combat_spans(
        [
            _combat_event(0, action_label="play card 0: Bash on enemy 0: Jaw Worm", enemy_hp_delta=-8),
            _combat_event(1, action_label="end turn", enemy_hp_delta=0, decision_after="combat_reward"),
            _reward_event(2),
        ]
    )

    assert len(spans) == 1
    assert [event["step"] for event in spans[0].events] == [0, 1]


def test_reflect_trace_writes_normalized_candidates(tmp_path):
    trace = tmp_path / "events.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        trace,
        [
            _combat_event(0, action_label="play card 0: Bash on enemy 0: Jaw Worm", enemy_hp_delta=-8),
            _combat_event(
                1,
                action_label="end turn",
                enemy_hp_delta=0,
                decision_after="combat_reward",
            ),
        ],
    )
    spans = combat_spans([json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()])
    assert len(spans) == 1

    def fake_chat(messages):
        assert "Compact trace span:" in messages[-1]["content"]
        return json.dumps(
            {
                "candidates": [
                    {
                        "memory_type": "risk",
                        "scope": {"level": "encounter", "phase": "first_turn"},
                        "trigger": "Act 1 Jaw Worm with incoming damage pressure.",
                        "lesson": "Card count is not the objective; compare damage lines with survival.",
                        "recommendation": "When HP is pressured, verify block or lethal before spending energy on attacks.",
                        "anti_pattern": "Playing more cards without checking incoming damage.",
                        "counterfactual": "If lethal is guaranteed, damage can override block.",
                        "quality": {"confidence": 0.72},
                        "retrieval_hints": {"keywords": ["jaw_worm", "block_vs_damage"], "priority": 0.8},
                    }
                ]
            }
        )

    summary = reflect_trace(
        trace_path=trace,
        candidate_out=candidates,
        chat_fn=fake_chat,
        overwrite=True,
    )

    assert summary["candidates_written"] == 1
    candidate = json.loads(candidates.read_text(encoding="utf-8"))
    assert candidate["status"] == "candidate"
    assert candidate["memory_type"] == "risk"
    assert candidate["scope"]["character"] == "Ironclad"
    assert candidate["scope"]["enemy"] == "Jaw Worm"
    assert candidate["scope"]["phase"] == "combat_play"
    assert candidate["source"]["span_type"] == "combat"
    assert candidate["source"]["used_memory_ids"] == ["exp_old"]
    assert "jaw_worm" in candidate["retrieval_hints"]["keywords"]


def test_promote_candidates_appends_active_entries_and_deduplicates(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    active = tmp_path / "active.jsonl"
    candidate = {
        "id": "exp_test_0001",
        "status": "candidate",
        "memory_type": "risk",
        "scope": {"level": "encounter", "character": "Ironclad", "act": 1, "enemy": "Jaw Worm", "phase": "first_turn"},
        "trigger": "Act 1 Jaw Worm with incoming damage pressure.",
        "lesson": "Compare damage with survival.",
        "recommendation": "Verify block or lethal before spending all energy on attacks.",
        "quality": {"confidence": 0.72},
        "retrieval_hints": {"keywords": ["jaw_worm", "combat_play"], "priority": 0.8},
        "source": {"span_type": "combat"},
    }
    _write_jsonl(candidates, [candidate])

    summary = promote_candidates(candidates_path=candidates, active_path=active)
    assert summary["promoted"] == 1
    active_entry = json.loads(active.read_text(encoding="utf-8"))
    assert active_entry["status"] == "active"
    assert active_entry["scope"]["phase"] == "combat_play"
    assert active_entry["quality"]["confidence"] == 0.72
    assert "promoted_at" in active_entry["quality"]

    second_summary = promote_candidates(candidates_path=candidates, active_path=active)
    assert second_summary["promoted"] == 0
    assert second_summary["skipped"][0]["reason"] == "duplicate_id"


def test_learn_from_trace_reflects_and_promotes_only_new_candidates(tmp_path):
    trace = tmp_path / "events.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    active = tmp_path / "active.jsonl"
    old_candidate = {
        "id": "exp_old_unreviewed",
        "status": "candidate",
        "memory_type": "risk",
        "scope": {"level": "encounter", "character": "Ironclad", "act": 1, "enemy": "Jaw Worm"},
        "trigger": "Old candidate should stay inactive.",
        "lesson": "Old lesson.",
        "recommendation": "Do not promote unrelated old candidates in a learn run.",
        "quality": {"confidence": 0.9},
        "retrieval_hints": {"keywords": ["old"], "priority": 0.5},
    }
    _write_jsonl(candidates, [old_candidate])
    _write_jsonl(
        trace,
        [
            _combat_event(0, action_label="play card 0: Bash on enemy 0: Jaw Worm", enemy_hp_delta=-8),
            _combat_event(
                1,
                action_label="end turn",
                enemy_hp_delta=0,
                decision_after="combat_reward",
            ),
        ],
    )

    def fake_chat(_messages):
        return json.dumps(
            {
                "candidates": [
                    {
                        "id": "exp_new_from_trace",
                        "memory_type": "risk",
                        "scope": {"level": "encounter"},
                        "trigger": "New reflected candidate.",
                        "lesson": "Use engine preview values instead of recalculating card damage.",
                        "recommendation": "Trust shown damage/block values when checking lethal and survival.",
                        "quality": {"confidence": 0.8},
                        "retrieval_hints": {"keywords": ["preview_damage"], "priority": 0.7},
                    }
                ]
            }
        )

    summary = learn_from_trace(
        trace_path=trace,
        candidate_out=candidates,
        active_path=active,
        chat_fn=fake_chat,
    )

    assert summary["reflection"]["candidate_ids"] == ["exp_new_from_trace"]
    assert summary["promotion"]["promoted"] == 1
    active_ids = [json.loads(line)["id"] for line in active.read_text(encoding="utf-8").splitlines()]
    assert active_ids == ["exp_new_from_trace"]
    candidate_ids = [json.loads(line)["id"] for line in candidates.read_text(encoding="utf-8").splitlines()]
    assert candidate_ids == ["exp_old_unreviewed", "exp_new_from_trace"]
