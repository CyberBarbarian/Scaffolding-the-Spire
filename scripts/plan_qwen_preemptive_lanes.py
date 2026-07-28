"""Plan safe preemption for the Qwen formal-experiment compute lanes.

This module does not stop or launch processes.  It separates slow paper-track
audit state from scarce model-compute occupancy and emits deterministic actions
that existing audited launchers can execute.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-preemptive-scheduler-v1"
TRACK_STATES = {
    "ACTIVE",
    "QUARANTINED",
    "READY_TO_RESUME",
    "READY_TO_START",
    "PAUSED_FILLER",
    "COUNTED",
    "EXCLUDED",
    "BLOCKED",
}
DEFAULT_PRIORITIES = {"B1": 400, "B2": 300, "ABLATION": 200}


class SchedulerError(ValueError):
    pass


def _priority(track: dict[str, Any]) -> int:
    explicit = track.get("priority")
    if isinstance(explicit, int):
        return explicit
    return DEFAULT_PRIORITIES.get(str(track.get("module") or ""), 0)


def _safe_checkpoint(track: dict[str, Any]) -> bool:
    return (
        track.get("safe_checkpoint") is True
        and track.get("provider_in_flight") is False
        and isinstance(track.get("next_decision"), int)
        and isinstance(track.get("last_accepted_action"), int)
        and track["next_decision"] == track["last_accepted_action"] + 1
        and track.get("zero_api_replay_ready") is True
    )


def _validate(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise SchedulerError(f"schema_version must be {SCHEMA_VERSION}")
    lane_cap = state.get("lane_cap")
    if not isinstance(lane_cap, int) or lane_cap < 1:
        raise SchedulerError("lane_cap must be a positive integer")
    tracks = state.get("tracks")
    if not isinstance(tracks, list):
        raise SchedulerError("tracks must be a list")

    seen_ids: set[str] = set()
    occupied_slots: set[str] = set()
    occupied_roots: set[str] = set()
    for track in tracks:
        if not isinstance(track, dict):
            raise SchedulerError("every track must be an object")
        track_id = str(track.get("id") or "")
        if not track_id or track_id in seen_ids:
            raise SchedulerError(f"track id is empty or duplicated: {track_id!r}")
        seen_ids.add(track_id)
        status = str(track.get("state") or "")
        if status not in TRACK_STATES:
            raise SchedulerError(f"invalid state for {track_id}: {status!r}")
        if track.get("process_alive") is True:
            paper_slot = str(track.get("paper_slot") or "")
            output_root = os.path.normcase(str(track.get("output_root") or ""))
            if not paper_slot or paper_slot in occupied_slots:
                raise SchedulerError(f"active paper_slot is empty or duplicated: {paper_slot!r}")
            if not output_root or output_root in occupied_roots:
                raise SchedulerError(f"active output_root is empty or duplicated: {output_root!r}")
            occupied_slots.add(paper_slot)
            occupied_roots.add(output_root)
    return lane_cap, tracks


def plan_schedule(state: dict[str, Any]) -> dict[str, Any]:
    lane_cap, tracks = _validate(state)
    actions: list[dict[str, Any]] = []
    projected_alive = {str(track["id"]) for track in tracks if track.get("process_alive") is True}

    # A quarantined paper track keeps all evidence and priority, but its compute
    # process is released as soon as the next accepted-action seam is durable.
    for track in tracks:
        if track.get("state") != "QUARANTINED" or track.get("process_alive") is not True:
            continue
        if _safe_checkpoint(track):
            actions.append({"action": "RELEASE_QUARANTINED", "track_id": track["id"]})
            projected_alive.remove(str(track["id"]))
        else:
            actions.append(
                {
                    "action": "WAIT_SAFE_BOUNDARY",
                    "track_id": track["id"],
                    "provider_in_flight": track.get("provider_in_flight"),
                }
            )

    def free_lanes() -> int:
        return lane_cap - len(projected_alive)

    ready_to_resume = sorted(
        (track for track in tracks if track.get("state") == "READY_TO_RESUME"),
        key=lambda track: (-_priority(track), str(track["id"])),
    )
    for track in ready_to_resume:
        if free_lanes() == 0:
            fillers = sorted(
                (
                    candidate
                    for candidate in tracks
                    if str(candidate["id"]) in projected_alive
                    and candidate.get("is_filler") is True
                    and _safe_checkpoint(candidate)
                    and _priority(candidate) < _priority(track)
                ),
                key=lambda candidate: (_priority(candidate), str(candidate["id"])),
            )
            if fillers:
                filler = fillers[0]
                actions.append(
                    {
                        "action": "PAUSE_FILLER",
                        "track_id": filler["id"],
                        "for_track_id": track["id"],
                    }
                )
                projected_alive.remove(str(filler["id"]))
        if free_lanes() > 0 and track.get("launch_ready") is True:
            actions.append({"action": "RESUME_PRIORITY", "track_id": track["id"]})
            projected_alive.add(str(track["id"]))

    fillers = sorted(
        (
            track
            for track in tracks
            if track.get("state") in {"READY_TO_START", "PAUSED_FILLER"}
            and track.get("launch_ready") is True
            and str(track["id"]) not in projected_alive
        ),
        key=lambda track: (-_priority(track), str(track["id"])),
    )
    for track in fillers:
        if free_lanes() <= 0:
            break
        action = "RESUME_FILLER" if track.get("state") == "PAUSED_FILLER" else "START_FILLER"
        actions.append({"action": action, "track_id": track["id"]})
        projected_alive.add(str(track["id"]))

    blockers = [
        {
            "track_id": track["id"],
            "state": track["state"],
            "blocker": track.get("blocker"),
        }
        for track in tracks
        if track.get("state") == "BLOCKED"
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "lane_cap": lane_cap,
        "observed_occupied": sum(track.get("process_alive") is True for track in tracks),
        "projected_occupied": len(projected_alive),
        "projected_free": free_lanes(),
        "audit_pending": sum(track.get("audit_pending") is True for track in tracks),
        "actions": actions,
        "blockers": blockers,
    }
    if result["projected_occupied"] > lane_cap:
        raise SchedulerError("projected model-lane usage exceeds lane_cap")
    return result


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".scheduler-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = plan_schedule(state)
    if args.output:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
