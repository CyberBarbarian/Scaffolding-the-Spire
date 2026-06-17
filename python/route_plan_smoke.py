#!/usr/bin/env python3
"""Run a short real-game route-plan smoke trace."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_bench.process import Sts2Process  # noqa: E402
from sts2_bench.route_plan_smoke import RoutePlanSmokeAgent  # noqa: E402
from sts2_bench.runner import JsonlLogger, run_one  # noqa: E402
from sts2_bench.trace_analysis import analyze_trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="route-plan-smoke")
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-game-actions", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--verbose-engine", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "artifacts" / f"route_plan_smoke_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    analysis_dir = out_dir / "analysis"

    logger = JsonlLogger(trace_path)
    process = Sts2Process(build_if_needed=True, stderr=None if args.verbose_engine else subprocess.DEVNULL)
    try:
        result = run_one(
            agent=RoutePlanSmokeAgent(),
            seed=args.seed,
            character=args.character,
            ascension=args.ascension,
            lang=args.lang,
            max_steps=args.max_steps,
            max_game_actions=args.max_game_actions,
            process=process,
            logger=logger,
            include_full_map=True,
            log_level="decisions",
        )
    finally:
        logger.close()
        process.close()

    report = analyze_trace(trace_path, analysis_dir)
    route_events = report.get("route_plan_events") or []
    validation_errors = report.get("route_plan_validation_errors") or []
    payload = {
        "result": result.to_dict(),
        "trace": str(trace_path),
        "analysis_dir": str(analysis_dir),
        "replay": str(analysis_dir / "replay.md"),
        "route_plan_event_count": len(route_events),
        "route_plan_validation_error_count": len(validation_errors),
        "route_plan_event_types": [event.get("event_type") for event in route_events],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if result.error or not route_events or validation_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
