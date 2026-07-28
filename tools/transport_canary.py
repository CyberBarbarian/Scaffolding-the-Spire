"""Small real-endpoint latency canary for byte-equivalent HTTP backends."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sts2_bench.agents import OpenAICompatAgent
from sts2_bench.experiment_identity import hash_file, hash_json
from sts2_bench.full_reflection_pairs import DEFAULT_EXPERIMENT, ROOT, load_experiment
from sts2_bench.reflection_probe import _resolve_api_key


def _agent(spec: dict[str, Any], api_key: str, backend: str) -> OpenAICompatAgent:
    return OpenAICompatAgent(
        base_url=str(spec["base_url"]),
        model=str(spec["model"]),
        api_key=api_key,
        temperature=float(spec.get("llm_temperature") or 0),
        timeout=float(spec.get("llm_timeout") or 180),
        max_retries=0,
        run_summary_enabled=False,
        json_response_format=True,
        http_backend=backend,  # type: ignore[arg-type]
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [float(row["elapsed_sec"]) for row in rows if row.get("status") == "ok"]
    return {
        "attempts": len(rows),
        "successes": len(successful),
        "errors": len(rows) - len(successful),
        "mean_sec": statistics.fmean(successful) if successful else None,
        "median_sec": statistics.median(successful) if successful else None,
        "min_sec": min(successful) if successful else None,
        "max_sec": max(successful) if successful else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    config, source = load_experiment(args.experiment)
    spec = config["baseline"]
    api_key = _resolve_api_key(spec, root=ROOT)
    messages = [
        {
            "role": "system",
            "content": "Return one valid JSON object only. Do not use markdown.",
        },
        {
            "role": "user",
            "content": 'Transport canary. Return exactly the semantic object {"ok": true}.',
        },
    ]
    candidate_backend = str(config["run"]["transport_reliability"]["http_backend"])
    agents = {backend: _agent(spec, api_key, backend) for backend in ("urllib", candidate_backend)}
    urllib_body = agents["urllib"]._chat_request_parts(messages)[1]
    candidate_body = agents[candidate_backend]._chat_request_parts(messages)[1]
    if urllib_body != candidate_body:
        raise RuntimeError("HTTP backends produced different request bytes")

    rows: list[dict[str, Any]] = []
    try:
        for backend, agent in agents.items():
            started = time.monotonic()
            response = agent._chat(copy.deepcopy(messages))
            rows.append(
                {
                    "phase": "warmup",
                    "backend": backend,
                    "status": "ok",
                    "elapsed_sec": time.monotonic() - started,
                    "response_hash": hash_json(response),
                }
            )
        for repetition in range(args.repetitions):
            order = ("urllib", candidate_backend) if repetition % 2 == 0 else (candidate_backend, "urllib")
            for backend in order:
                started = time.monotonic()
                try:
                    response = agents[backend]._chat(copy.deepcopy(messages))
                    rows.append(
                        {
                            "phase": "measured",
                            "repetition": repetition + 1,
                            "backend": backend,
                            "status": "ok",
                            "elapsed_sec": time.monotonic() - started,
                            "response_hash": hash_json(response),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "phase": "measured",
                            "repetition": repetition + 1,
                            "backend": backend,
                            "status": "error",
                            "elapsed_sec": time.monotonic() - started,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        }
                    )
    finally:
        for agent in agents.values():
            agent.close_http_transport()

    measured = [row for row in rows if row["phase"] == "measured"]
    document = {
        "schema_version": 1,
        "experiment": str(source),
        "experiment_file_sha256": hash_file(source),
        "experiment_resolved_hash": hash_json(config),
        "model": spec["model"],
        "request_body_sha256": hashlib.sha256(urllib_body).hexdigest(),
        "request_bytes_equal": True,
        "repetitions": args.repetitions,
        "summaries": {
            backend: _summary([row for row in measured if row["backend"] == backend])
            for backend in agents
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "ok" for row in measured) else 1


if __name__ == "__main__":
    raise SystemExit(main())
