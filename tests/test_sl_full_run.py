from sts2_bench import sl_full_run
from sts2_bench.actions import LegalAction
from sts2_bench.combat_snapshot import PrefixReplayResult
from sts2_bench.sl_full_run import (
    _ActiveCheckpoint,
    _run_replay_attempt,
    ReplayAttempt,
    build_snapshot_from_combat_entry,
    merge_replay_commands_into_run_tape,
    run_sl_one,
    select_best_attempt,
)


def test_build_snapshot_from_combat_entry_uses_room_entry_commands_only(tmp_path):
    save_path = tmp_path / "entry.save"
    entry = {
        "combat_id": "run_combat_001_A1F6_Monster",
        "run": {
            "seed": "bench_seed",
            "character": "Ironclad",
            "ascension": 0,
            "lang": "en",
        },
        "entry": {
            "step": 12,
            "decision": "combat_play",
            "state": {"decision": "combat_play", "hp": 42, "enemy_count": 1},
            "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 40}],
            "prefix_commands": [
                {"cmd": "action", "action": "claim_reward", "args": {"reward_index": 0}},
                {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 3}},
                {"cmd": "action", "action": "choose_option", "args": {"index": 0}},
            ],
        },
        "save": {"success": True, "path": str(save_path)},
    }

    snapshot = build_snapshot_from_combat_entry(entry, snapshot_id="sl_retry_001")

    assert snapshot["type"] == "combat_snapshot"
    assert snapshot["snapshot_id"] == "sl_retry_001"
    assert snapshot["game"] == {
        "seed": "bench_seed",
        "character": "Ironclad",
        "ascension": 0,
        "lang": "en",
    }
    assert snapshot["load"] == {
        "type": "save",
        "path": str(save_path),
        "entry_commands": [
            {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 3}},
            {"cmd": "action", "action": "choose_option", "args": {"index": 0}},
        ],
    }


def test_select_best_attempt_prefers_survival_then_hp_then_shorter_combat():
    original = ReplayAttempt(
        attempt_id="original",
        outcome="combat_complete",
        hp=12,
        steps=20,
        commands=[],
    )
    death = ReplayAttempt(
        attempt_id="retry_death",
        outcome="game_over",
        hp=0,
        steps=8,
        commands=[],
    )
    high_hp = ReplayAttempt(
        attempt_id="retry_high",
        outcome="combat_complete",
        hp=30,
        steps=18,
        commands=[],
    )
    same_hp_shorter = ReplayAttempt(
        attempt_id="retry_short",
        outcome="combat_complete",
        hp=30,
        steps=12,
        commands=[],
    )

    assert select_best_attempt([original, death, high_hp, same_hp_shorter]).attempt_id == "retry_short"


def test_merge_replay_commands_into_run_tape_does_not_duplicate_room_entry_commands():
    run_tape_at_entry = [
        {"cmd": "action", "action": "claim_reward", "args": {"reward_index": 0}},
        {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 3}},
    ]
    replay_tape = [
        {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 3}},
        {"cmd": "action", "action": "play_card", "args": {"card_index": 0}},
        {"cmd": "action", "action": "end_turn"},
    ]

    merged = merge_replay_commands_into_run_tape(
        run_tape_at_entry,
        replay_tape,
        replay_entry_command_count=1,
    )

    assert merged == [
        {"cmd": "action", "action": "claim_reward", "args": {"reward_index": 0}},
        {"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 3}},
        {"cmd": "action", "action": "play_card", "args": {"card_index": 0}},
        {"cmd": "action", "action": "end_turn"},
    ]


def test_sequence_validation_tape_for_checkpoint_uses_room_entry_and_current_combat_only(tmp_path):
    early_map = {"cmd": "action", "action": "select_map_node", "args": {"col": 2, "row": 1}}
    room_entry = {"cmd": "action", "action": "select_map_node", "args": {"col": 3, "row": 4}}
    combat_action = {"cmd": "action", "action": "play_card", "args": {"card_index": 2, "target_index": 0}}
    checkpoint = _ActiveCheckpoint(
        checkpoint_id="sl_ckpt_002_A1F5_Monster",
        directory=tmp_path,
        snapshot={
            "load": {
                "type": "save",
                "path": str(tmp_path / "entry.save"),
                "entry_commands": [room_entry],
            }
        },
        entry_hp=55,
        entry_step=23,
        run_tape_at_entry=[early_map, room_entry],
    )

    tape = sl_full_run._sequence_validation_tape_for_checkpoint(
        checkpoint,
        [early_map, room_entry, combat_action],
    )

    assert tape == [room_entry, combat_action]


def test_replay_attempt_uses_save_backed_probe_factory_for_sequence_validation(tmp_path, monkeypatch):
    entry_command = {"cmd": "action", "action": "select_map_node", "args": {"col": 3, "row": 16}}
    snapshot = {
        "load": {
            "type": "save",
            "path": str(tmp_path / "entry.save"),
            "entry_commands": [entry_command],
        }
    }
    checkpoint = _ActiveCheckpoint(
        checkpoint_id="sl_ckpt_001_A1F17_Boss",
        directory=tmp_path,
        snapshot=snapshot,
        entry_hp=79,
        entry_step=115,
        run_tape_at_entry=[entry_command],
    )
    sentinel_factory = object()
    captured = {}

    def fake_save_backed_sequence_probe_factory(snapshot_arg):
        captured["snapshot"] = snapshot_arg
        return sentinel_factory

    class DummyProcess:
        def close(self):
            pass

    class DummyEvidenceBundle:
        deck_summary_visible = False
        full_deck_details_visible = False
        deck_details_visible = False
        skip_action_visible = False

        def bundle_hash(self):
            return "test-hash"

        def summary_dict(self):
            return {"decision_type": "combat_play"}

    def fake_replay_snapshot_prefix(snapshot_arg, **kwargs):
        return PrefixReplayResult(
            state={"decision": "combat_play", "player": {"hp": 79}},
            action_tape=[entry_command],
            prefix_actions_executed=1,
            process=kwargs["process"],
        )

    legal_action = LegalAction(
        action_id=1,
        label="end turn",
        command={"cmd": "action", "action": "end_turn"},
        kind="combat_end_turn",
    )

    def fake_choose_action_with_runner_validation(**kwargs):
        captured["sequence_probe_factory"] = kwargs["sequence_probe_factory"]
        return legal_action, {}, "forced-stop"

    monkeypatch.setattr(sl_full_run, "_save_backed_sequence_probe_factory", fake_save_backed_sequence_probe_factory)
    monkeypatch.setattr(sl_full_run, "Sts2Process", DummyProcess)
    monkeypatch.setattr(sl_full_run, "replay_snapshot_prefix", fake_replay_snapshot_prefix)
    monkeypatch.setattr(sl_full_run, "build_legal_actions", lambda *args, **kwargs: [legal_action])
    monkeypatch.setattr(sl_full_run, "build_evidence_bundle", lambda *args, **kwargs: DummyEvidenceBundle())
    monkeypatch.setattr(sl_full_run, "_choose_action_with_runner_validation", fake_choose_action_with_runner_validation)

    replay = _run_replay_attempt(
        checkpoint=checkpoint,
        attempt_index=1,
        agent_factory=lambda: object(),
        character="Ironclad",
        seed="bench_seed",
        ascension=0,
        lang="en",
        allow_repeat_views=False,
        allow_custom_sequences=True,
        custom_sequence_exposure="safe",
        combat_card_sequences_enabled=True,
        precomputed_combat_card_sequences_enabled=False,
        log_level="decisions",
        route_plan_enabled=False,
        route_plan_enforcement="hard",
        max_steps=1,
    )

    assert captured["snapshot"] is snapshot
    assert captured["sequence_probe_factory"] is sentinel_factory
    assert replay.result.error == "forced-stop"


def test_main_sl_run_uses_active_checkpoint_probe_factory_for_sequence_validation(tmp_path, monkeypatch):
    snapshot = {
        "load": {
            "type": "save",
            "path": str(tmp_path / "entry.save"),
            "entry_commands": [],
        }
    }
    sentinel_factory = object()
    captured = {}

    class DummyProcess:
        def start(self):
            return None

        def start_run(self, **kwargs):
            return {
                "decision": "combat_play",
                "act": 1,
                "floor": 3,
                "room_type": "Monster",
                "player": {"hp": 70},
            }

        def close(self):
            pass

    class DummyEvidenceBundle:
        deck_summary_visible = False
        full_deck_details_visible = False
        deck_details_visible = False
        skip_action_visible = False

        def bundle_hash(self):
            return "test-hash"

        def summary_dict(self):
            return {"decision_type": "combat_play"}

    legal_action = LegalAction(
        action_id=1,
        label="end turn",
        command={"cmd": "action", "action": "end_turn"},
        kind="combat_end_turn",
    )

    def fake_create_save_backed_combat_snapshot(**kwargs):
        captured["checkpoint_state"] = kwargs["state"]
        return snapshot

    def fake_save_backed_sequence_probe_factory(snapshot_arg):
        captured["snapshot"] = snapshot_arg
        return sentinel_factory

    def fake_choose_action_with_runner_validation(**kwargs):
        captured["sequence_probe_factory"] = kwargs["sequence_probe_factory"]
        return legal_action, {}, "forced-stop"

    monkeypatch.setattr(sl_full_run, "Sts2Process", DummyProcess)
    monkeypatch.setattr(sl_full_run, "create_save_backed_combat_snapshot", fake_create_save_backed_combat_snapshot)
    monkeypatch.setattr(sl_full_run, "_save_backed_sequence_probe_factory", fake_save_backed_sequence_probe_factory)
    monkeypatch.setattr(sl_full_run, "build_legal_actions", lambda *args, **kwargs: [legal_action])
    monkeypatch.setattr(sl_full_run, "build_evidence_bundle", lambda *args, **kwargs: DummyEvidenceBundle())
    monkeypatch.setattr(sl_full_run, "_choose_action_with_runner_validation", fake_choose_action_with_runner_validation)

    result = run_sl_one(
        agent_factory=lambda: object(),
        seed="bench_seed",
        sl_dir=tmp_path,
        max_steps=1,
        allow_custom_sequences=True,
        route_plan_enabled=False,
    )

    assert captured["checkpoint_state"]["decision"] == "combat_play"
    assert captured["snapshot"] is snapshot
    assert captured["sequence_probe_factory"] is sentinel_factory
    assert result.error == "forced-stop"
