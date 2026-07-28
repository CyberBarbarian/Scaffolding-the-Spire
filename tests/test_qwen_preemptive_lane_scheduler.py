import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "plan_qwen_preemptive_lanes.py"
SPEC = importlib.util.spec_from_file_location("plan_qwen_preemptive_lanes", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def track(track_id, state, *, module="B2", alive=False, filler=False, ready=False, safe=False):
    return {
        "id": track_id,
        "paper_slot": f"slot/{track_id}",
        "output_root": f"results/{track_id}",
        "module": module,
        "state": state,
        "process_alive": alive,
        "is_filler": filler,
        "launch_ready": ready,
        "safe_checkpoint": safe,
        "provider_in_flight": False,
        "last_accepted_action": 8,
        "next_decision": 9,
        "zero_api_replay_ready": safe,
    }


def state(*tracks, cap=2):
    return {"schema_version": MODULE.SCHEMA_VERSION, "lane_cap": cap, "tracks": list(tracks)}


def test_quarantined_audit_releases_compute_and_starts_filler():
    suspect = track("suspect", "QUARANTINED", alive=True, safe=True)
    suspect["audit_pending"] = True
    filler = track("filler", "READY_TO_START", filler=True, ready=True)
    result = MODULE.plan_schedule(state(suspect, filler, cap=1))
    assert [item["action"] for item in result["actions"]] == [
        "RELEASE_QUARANTINED",
        "START_FILLER",
    ]
    assert result["audit_pending"] == 1
    assert result["projected_occupied"] == 1


def test_ready_priority_track_pauses_safe_lower_priority_filler():
    filler = track("filler", "ACTIVE", module="ABLATION", alive=True, filler=True, safe=True)
    original = track("original", "READY_TO_RESUME", module="B1", ready=True)
    result = MODULE.plan_schedule(state(filler, original, cap=1))
    assert result["actions"] == [
        {"action": "PAUSE_FILLER", "track_id": "filler", "for_track_id": "original"},
        {"action": "RESUME_PRIORITY", "track_id": "original"},
    ]
    assert result["projected_occupied"] == 1


def test_inflight_quarantined_process_is_not_killed_or_overcommitted():
    suspect = track("suspect", "QUARANTINED", alive=True, safe=False)
    suspect["provider_in_flight"] = True
    filler = track("filler", "READY_TO_START", filler=True, ready=True)
    result = MODULE.plan_schedule(state(suspect, filler, cap=1))
    assert result["actions"] == [
        {"action": "WAIT_SAFE_BOUNDARY", "track_id": "suspect", "provider_in_flight": True}
    ]
    assert result["projected_occupied"] == 1


def test_audit_only_track_does_not_consume_compute_lane():
    suspect = track("suspect", "QUARANTINED", alive=False)
    suspect["audit_pending"] = True
    filler = track("filler", "READY_TO_START", filler=True, ready=True)
    result = MODULE.plan_schedule(state(suspect, filler, cap=1))
    assert result["observed_occupied"] == 0
    assert result["projected_occupied"] == 1
    assert result["actions"] == [{"action": "START_FILLER", "track_id": "filler"}]


def test_duplicate_active_paper_slot_fails_closed():
    first = track("first", "ACTIVE", alive=True)
    second = track("second", "ACTIVE", alive=True)
    second["paper_slot"] = first["paper_slot"]
    with pytest.raises(MODULE.SchedulerError, match="paper_slot"):
        MODULE.plan_schedule(state(first, second, cap=2))
