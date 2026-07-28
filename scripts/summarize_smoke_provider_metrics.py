"""Summarize audited provider calls from nightly smoke result roots."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return round(ordered[index], 6)


def _summary(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(calls)
    successful = [item for item in items if item["successful"]]
    latencies = [float(item["latency_sec"]) for item in successful]
    statuses: dict[str, int] = {}
    errors: dict[str, int] = {}
    for item in items:
        status = str(item.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if item["successful"]:
            continue
        error = str(item.get("error") or "unknown")[:240]
        errors[error] = errors.get(error, 0) + 1
    return {
        "provider_calls": len(items),
        "successful_calls": len(successful),
        "failed_or_retry_calls": len(items) - len(successful),
        "status_counts": statuses,
        "error_counts": errors,
        "latency_mean_sec": round(statistics.fmean(latencies), 6) if latencies else None,
        "latency_p50_sec": _percentile(latencies, 0.50),
        "latency_p95_sec": _percentile(latencies, 0.95),
        "latency_max_sec": round(max(latencies), 6) if latencies else None,
    }


def _direct_cli_calls(root: Path) -> list[dict[str, Any]]:
    events = _rows(root / "B0_direct_cli" / "artifacts" / "events.jsonl")
    calls: list[dict[str, Any]] = []
    for row in events:
        if row.get("type") != "terminal_decision":
            continue
        calls.append(
            {
                "kind": "policy",
                "successful": bool(row.get("output_format_valid")),
                "status": "succeeded" if row.get("output_format_valid") else "model_protocol_error",
                "latency_sec": float(row.get("provider_latency_sec") or 0.0),
                "error": None if row.get("output_format_valid") else "invalid terminal output",
            }
        )
    return calls


def _audited_calls(audits: Any, *, kind: str) -> list[dict[str, Any]]:
    if not isinstance(audits, list):
        return []
    calls: list[dict[str, Any]] = []
    for audit in audits:
        if not isinstance(audit, dict):
            continue
        status = str(audit.get("provider_response_status") or "unknown")
        calls.append(
            {
                "kind": kind,
                "successful": status == "succeeded",
                "status": status,
                "latency_sec": float(audit.get("provider_latency_sec") or 0.0),
                "error": audit.get("provider_error"),
            }
        )
    return calls


def _agent_lane_calls(trace: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for row in _rows(trace):
        if row.get("type") != "action":
            continue
        meta = row.get("agent_meta")
        if not isinstance(meta, dict):
            continue
        calls.extend(_audited_calls(meta.get("model_request_audits"), kind="policy"))
        calls.extend(
            _audited_calls(
                meta.get("reward_posterior_model_request_audits"),
                kind="reward_posterior",
            )
        )
    return calls


def summarize_root(root: Path) -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    all_calls: list[dict[str, Any]] = []
    for lane in sorted(path for path in root.iterdir() if path.is_dir()):
        if lane.name == "B0_direct_cli":
            calls = _direct_cli_calls(root)
        elif (lane / "trace.jsonl").is_file():
            calls = _agent_lane_calls(lane / "trace.jsonl")
        else:
            continue
        all_calls.extend(calls)
        lane_summary = _summary(calls)
        lane_summary["policy"] = _summary(item for item in calls if item["kind"] == "policy")
        lane_summary["reward_posterior"] = _summary(
            item for item in calls if item["kind"] == "reward_posterior"
        )
        lanes[lane.name] = lane_summary
    return {
        "root": str(root.resolve()),
        "lanes": lanes,
        "overall": _summary(all_calls),
        "overall_policy": _summary(item for item in all_calls if item["kind"] == "policy"),
        "overall_reward_posterior": _summary(
            item for item in all_calls if item["kind"] == "reward_posterior"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {"roots": [summarize_root(root.resolve()) for root in args.root]}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
