"""Measure OpenAI-compatible provider concurrency with short and real prompts.

The probe never writes API keys. It records the provider model catalog, request
hashes, response hashes, latency, HTTP status, usage, and sanitized error text.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_json(
    *,
    url: str,
    api_key: str,
    body: dict[str, Any] | None,
    timeout_sec: float,
) -> tuple[int | None, dict[str, Any] | None, str, float]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET" if body is None else "POST",
    )
    started = time.perf_counter()
    status: int | None = None
    raw = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            status = int(response.status)
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # transport errors are evidence, not probe crashes
        return None, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - started
    elapsed = time.perf_counter() - started
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return status, None, raw[:1000], elapsed
    return status, parsed, raw, elapsed


def _load_long_messages(
    trace_path: Path,
    *,
    decision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[int, int, list[dict[str, Any]], str]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") != "action" or row.get("decision") != decision:
            continue
        meta = row.get("agent_meta")
        if not isinstance(meta, dict):
            continue
        audits = meta.get("model_request_audits")
        if not isinstance(audits, list) or not audits:
            continue
        messages = audits[0].get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        candidates.append(
            (
                int(meta.get("prompt_chars") or 0),
                int(row.get("step") or 0),
                messages,
                str(audits[0].get("messages_sha256") or ""),
            )
        )
    if not candidates:
        raise RuntimeError(f"No audited {decision} model request in {trace_path}")
    prompt_chars, step, messages, recorded_hash = max(candidates, key=lambda item: item[0])
    actual_hash = _sha256_text(_canonical(messages))
    if recorded_hash and recorded_hash != actual_hash:
        raise RuntimeError(
            f"Recorded message hash mismatch at step {step}: {recorded_hash} != {actual_hash}"
        )
    return messages, {
        "source_trace": str(trace_path.resolve()),
        "source_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "source_step": step,
        "source_decision": decision,
        "prompt_chars": prompt_chars,
        "messages_sha256": actual_hash,
        "message_count": len(messages),
    }


def _one_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_sec: float,
    barrier: threading.Barrier,
    request_index: int,
    scenario: str,
    level: int,
    round_index: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    barrier.wait(timeout=30)
    status, parsed, raw, elapsed = _request_json(
        url=f"{base_url.rstrip('/')}/chat/completions",
        api_key=api_key,
        body=body,
        timeout_sec=timeout_sec,
    )
    content: Any = (
        parsed["choices"][0]["message"].get("content")
        if isinstance(parsed, dict)
        and isinstance(parsed.get("choices"), list)
        and parsed["choices"]
        and isinstance(parsed["choices"][0], dict)
        and isinstance(parsed["choices"][0].get("message"), dict)
        else None
    )
    choice_present = bool(
        isinstance(parsed, dict)
        and isinstance(parsed.get("choices"), list)
        and parsed["choices"]
        and isinstance(parsed["choices"][0], dict)
        and isinstance(parsed["choices"][0].get("message"), dict)
        and content is not None
    )
    json_content_valid = False
    if isinstance(content, str):
        try:
            json_content_valid = isinstance(json.loads(content), dict)
        except json.JSONDecodeError:
            pass
    error_value: Any = parsed.get("error") if isinstance(parsed, dict) else raw
    if isinstance(error_value, (dict, list)):
        error_text = _canonical(error_value)
    else:
        error_text = str(error_value or "")
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    return {
        "scenario": scenario,
        "level": level,
        "round": round_index,
        "request_index": request_index,
        "http_status": status,
        "success": bool(
            status is not None
            and 200 <= status < 300
            and choice_present
            and json_content_valid
        ),
        "choice_present": choice_present,
        "json_content_valid": json_content_valid,
        "latency_sec": round(elapsed, 6),
        "response_sha256": _sha256_text(raw),
        "response_chars": len(raw),
        "usage": usage if isinstance(usage, dict) else None,
        "error": error_text[:1000] or None,
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return round(ordered[index], 6)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["scenario"]), int(row["level"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (scenario, level), items in sorted(groups.items()):
        latencies = [float(item["latency_sec"]) for item in items]
        errors: dict[str, int] = {}
        for item in items:
            if item["success"]:
                continue
            key = f"http={item['http_status']} error={item['error']}"
            errors[key] = errors.get(key, 0) + 1
        summary.append(
            {
                "scenario": scenario,
                "level": level,
                "requests": len(items),
                "successes": sum(bool(item["success"]) for item in items),
                "success_rate": sum(bool(item["success"]) for item in items) / len(items),
                "stable": all(bool(item["success"]) for item in items),
                "latency_mean_sec": round(statistics.fmean(latencies), 6),
                "latency_p50_sec": _percentile(latencies, 0.50),
                "latency_p95_sec": _percentile(latencies, 0.95),
                "errors": errors,
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--long-decision",
        default="combat_play",
        help="Decision type whose largest audited request supplies the long scenario",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", default="1,2,3,4,5")
    parser.add_argument(
        "--scenarios",
        default="short_json,long_real_combat",
        help="Comma-separated subset of short_json,long_real_combat",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--cooldown-sec", type=float, default=3.0)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable {args.api_key_env}")
    levels = [int(value) for value in args.levels.split(",") if value.strip()]
    if not levels or any(level <= 0 for level in levels):
        raise RuntimeError("Concurrency levels must be positive integers")
    requested_scenarios = [
        value.strip() for value in args.scenarios.split(",") if value.strip()
    ]
    valid_scenarios = {"short_json", "long_real_combat"}
    if not requested_scenarios or not set(requested_scenarios) <= valid_scenarios:
        raise RuntimeError(
            f"Scenarios must be a non-empty subset of {sorted(valid_scenarios)}"
        )

    status, catalog_payload, raw_catalog, catalog_elapsed = _request_json(
        url=f"{args.base_url.rstrip('/')}/models",
        api_key=api_key,
        body=None,
        timeout_sec=args.timeout_sec,
    )
    catalog_ids = sorted(
        str(item.get("id"))
        for item in (
            catalog_payload.get("data", [])
            if isinstance(catalog_payload, dict)
            else []
        )
        if isinstance(item, dict) and item.get("id")
    )
    if status != 200 or args.model not in catalog_ids:
        raise RuntimeError(
            f"Catalog gate failed: status={status}, model_present={args.model in catalog_ids}, "
            f"body_sha256={_sha256_text(raw_catalog)}"
        )

    long_messages, long_source = _load_long_messages(
        args.trace.resolve(),
        decision=args.long_decision,
    )
    short_messages = [
        {
            "role": "system",
            "content": "Return one valid JSON object and no surrounding text.",
        },
        {
            "role": "user",
            "content": 'Return exactly {"probe":"ok"}.',
        },
    ]
    all_scenarios = {
        "short_json": short_messages,
        "long_real_combat": long_messages,
    }
    scenarios = {name: all_scenarios[name] for name in requested_scenarios}

    rows: list[dict[str, Any]] = []
    for scenario, messages in scenarios.items():
        for level in levels:
            for round_index in range(1, args.rounds + 1):
                barrier = threading.Barrier(level)
                with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                    futures = [
                        executor.submit(
                            _one_completion,
                            base_url=args.base_url,
                            api_key=api_key,
                            model=args.model,
                            messages=messages,
                            timeout_sec=args.timeout_sec,
                            barrier=barrier,
                            request_index=index,
                            scenario=scenario,
                            level=level,
                            round_index=round_index,
                        )
                        for index in range(level)
                    ]
                    rows.extend(future.result() for future in futures)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                partial = {
                    "schema_version": 1,
                    "status": "running",
                    "channel_name": args.channel_name,
                    "base_url": args.base_url,
                    "model": args.model,
                    "catalog_http_status": status,
                    "catalog_latency_sec": round(catalog_elapsed, 6),
                    "catalog_model_ids": catalog_ids,
                    "long_prompt_source": long_source,
                    "short_messages_sha256": _sha256_text(_canonical(short_messages)),
                    "levels": levels,
                    "rounds": args.rounds,
                    "results": rows,
                    "summary": _summarize(rows),
                }
                args.out.write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                time.sleep(args.cooldown_sec)

    summary = _summarize(rows)
    highest_stable: dict[str, int | None] = {}
    for scenario in scenarios:
        highest: int | None = None
        by_level = {
            int(item["level"]): bool(item["stable"])
            for item in summary
            if item["scenario"] == scenario
        }
        for level in sorted(levels):
            if not by_level.get(level, False):
                break
            highest = level
        highest_stable[scenario] = highest
    final = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "channel_name": args.channel_name,
        "base_url": args.base_url,
        "model": args.model,
        "catalog_http_status": status,
        "catalog_latency_sec": round(catalog_elapsed, 6),
        "catalog_model_ids": catalog_ids,
        "long_prompt_source": long_source,
        "short_messages_sha256": _sha256_text(_canonical(short_messages)),
        "levels": levels,
        "rounds": args.rounds,
        "highest_zero_error_concurrency": highest_stable,
        "results": rows,
        "summary": summary,
        "api_key_recorded": False,
    }
    args.out.write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "channel": args.channel_name,
                "model": args.model,
                "highest_zero_error_concurrency": highest_stable,
                "out": str(args.out.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
