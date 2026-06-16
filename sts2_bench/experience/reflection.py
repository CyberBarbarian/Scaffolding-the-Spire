"""Offline reflection utilities for learnable experience memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..agents import OpenAICompatAgent, parse_json_object
from .agent import ExperienceMemoryEntry


ChatFn = Callable[[list[dict[str, str]]], str]
ALLOWED_MEMORY_TYPES = {"rule", "risk", "episode", "playbook", "counterfactual"}
DEFAULT_CANDIDATE_PATH = "sts2_bench/experience/memory/candidates.jsonl"
DEFAULT_ACTIVE_PATH = "sts2_bench/experience/memory/active.jsonl"
NON_COMBAT_PHASE_TAGS = {
    "combat_reward",
    "card_reward",
    "bundle_select",
    "card_select",
    "event_choice",
    "map_select",
    "shop",
    "rest",
}


@dataclass(frozen=True)
class TraceSpan:
    span_id: str
    span_type: str
    events: list[dict[str, Any]]

    @property
    def first_event(self) -> dict[str, Any]:
        return self.events[0]

    @property
    def last_event(self) -> dict[str, Any]:
        return self.events[-1]


def load_event_trace(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{target}:{line_number} must be a JSON object")
        events.append(payload)
    return events


def combat_spans(events: list[dict[str, Any]]) -> list[TraceSpan]:
    spans: list[TraceSpan] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[Any, ...] | None = None

    for event in events:
        if not _is_combat_event(event):
            if current:
                spans.append(_make_combat_span(current))
                current = []
                current_key = None
            continue

        key = _combat_key(event)
        if current and key != current_key:
            spans.append(_make_combat_span(current))
            current = []
        current.append(event)
        current_key = key

        decision_after = _outcome(event).get("decision_after")
        if decision_after and str(decision_after) != "combat_play":
            spans.append(_make_combat_span(current))
            current = []
            current_key = None

    if current:
        spans.append(_make_combat_span(current))
    return spans


def build_reflection_messages(span: TraceSpan, *, max_candidates: int, max_events: int) -> list[dict[str, str]]:
    prompt_payload = _span_prompt_payload(span, max_events=max_events)
    user = "\n".join(
        [
            "Reflect on this compact Slay the Spire 2 agent trace span.",
            "Extract only reusable decision lessons that could improve future agent play.",
            "Do not restate full state data, card text, or obvious facts already visible in future states.",
            "Prefer lessons about action selection, risk boundaries, sequencing, and anti-patterns.",
            "For combat spans, ignore post-combat reward/path choices; only extract combat-play lessons.",
            "If the span has no reusable lesson, return an empty candidates list.",
            f"Return at most {max_candidates} candidates.",
            "Return only valid JSON with this schema:",
            (
                '{"candidates":[{"memory_type":"rule|risk|episode|playbook|counterfactual",'
                '"scope":{"level":"global|encounter|act|combat","character":"<character if specific>",'
                '"act":<integer if specific>,"enemy":"<enemy if specific>","phase":"combat_play for combat spans"},'
                '"trigger":"<when this lesson should be retrieved>",'
                '"lesson":"<short reusable lesson>",'
                '"recommendation":"<actionable advice>",'
                '"anti_pattern":"<mistake to avoid, optional>",'
                '"counterfactual":"<when not to apply, optional>",'
                '"quality":{"confidence":<0.0-1.0>},'
                '"retrieval_hints":{"keywords":["<search tag>"],"priority":<0.0-1.0>}}]}'
            ),
            "Compact trace span:",
            "For timing such as first turn, safe turn, stunned, high HP, or low HP, write it in trigger/counterfactual; do not invent a free-form phase.",
            json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You write compact, reusable experience memories for an STS2 benchmark agent. "
                "Be conservative: weak or redundant lessons should be omitted."
            ),
        },
        {"role": "user", "content": user},
    ]


def reflect_trace(
    *,
    trace_path: str | Path,
    candidate_out: str | Path,
    chat_fn: ChatFn,
    max_spans: int = 0,
    max_candidates_per_span: int = 2,
    max_events_per_span: int = 40,
    overwrite: bool = False,
) -> dict[str, Any]:
    events = load_event_trace(trace_path)
    return reflect_events(
        events,
        trace_path=str(trace_path),
        candidate_out=candidate_out,
        chat_fn=chat_fn,
        max_spans=max_spans,
        max_candidates_per_span=max_candidates_per_span,
        max_events_per_span=max_events_per_span,
        overwrite=overwrite,
    )


def reflect_events(
    events: list[dict[str, Any]],
    *,
    trace_path: str,
    candidate_out: str | Path,
    chat_fn: ChatFn,
    max_spans: int = 0,
    max_candidates_per_span: int = 2,
    max_events_per_span: int = 40,
    overwrite: bool = False,
) -> dict[str, Any]:
    spans = combat_spans(events)
    if max_spans > 0:
        spans = spans[:max_spans]

    written: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for span in spans:
        try:
            candidates = reflect_span(
                span,
                chat_fn=chat_fn,
                trace_path=str(trace_path),
                max_candidates=max_candidates_per_span,
                max_events=max_events_per_span,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports span-level failures and continues.
            errors.append({"span_id": span.span_id, "error": str(exc)})
            continue
        written.extend(candidates)

    append_jsonl(candidate_out, written, overwrite=overwrite)
    return {
        "trace_path": str(trace_path),
        "candidate_out": str(candidate_out),
        "events": len(events),
        "spans": len(spans),
        "candidates_written": len(written),
        "candidate_ids": [str(candidate.get("id")) for candidate in written if candidate.get("id")],
        "errors": errors,
    }


def reflect_span(
    span: TraceSpan,
    *,
    chat_fn: ChatFn,
    trace_path: str,
    max_candidates: int,
    max_events: int,
) -> list[dict[str, Any]]:
    messages = build_reflection_messages(span, max_candidates=max_candidates, max_events=max_events)
    text = chat_fn(messages)
    payload = parse_json_object(text)
    raw_candidates = payload.get("candidates") or []
    if not isinstance(raw_candidates, list):
        raise ValueError("reflection response field candidates must be a list")

    candidates: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(raw_candidates[:max_candidates]):
        if not isinstance(raw_candidate, dict):
            continue
        candidate = normalize_candidate(raw_candidate, span=span, trace_path=trace_path, index=index)
        ExperienceMemoryEntry.from_dict({**candidate, "status": "active"})
        candidates.append(candidate)
    return candidates


def normalize_candidate(
    candidate: dict[str, Any],
    *,
    span: TraceSpan,
    trace_path: str,
    index: int,
) -> dict[str, Any]:
    first = span.first_event
    memory_type = _one_line(candidate.get("memory_type") or candidate.get("type") or "rule")
    if memory_type not in ALLOWED_MEMORY_TYPES:
        memory_type = "rule"

    scope = candidate.get("scope") if isinstance(candidate.get("scope"), dict) else {}
    normalized_scope = dict(scope)
    normalized_scope.setdefault("level", "encounter" if first.get("encounter") else "combat")
    if first.get("character"):
        normalized_scope.setdefault("character", first.get("character"))
    if first.get("act") is not None:
        normalized_scope.setdefault("act", first.get("act"))
    if first.get("encounter"):
        normalized_scope.setdefault("enemy", first.get("encounter"))
    if span.span_type == "combat":
        normalized_scope["phase"] = "combat_play"
    elif first.get("decision"):
        normalized_scope.setdefault("phase", first.get("decision"))
    if span.span_type == "combat" and _slug(normalized_scope.get("phase")) in NON_COMBAT_PHASE_TAGS:
        normalized_scope["phase"] = "combat_play"

    quality = candidate.get("quality") if isinstance(candidate.get("quality"), dict) else {}
    confidence = _clamp_float(quality.get("confidence"), default=0.5)
    normalized_quality = dict(quality)
    normalized_quality["confidence"] = confidence

    hints = candidate.get("retrieval_hints") if isinstance(candidate.get("retrieval_hints"), dict) else {}
    raw_keywords = hints.get("keywords")
    if not isinstance(raw_keywords, list):
        raw_keywords = []
    keywords = _keyword_list(
        [
            *raw_keywords,
            memory_type,
            first.get("character"),
            first.get("decision"),
            first.get("room_type"),
            first.get("encounter"),
            f"act{first.get('act')}" if first.get("act") is not None else "",
        ]
    )
    normalized_hints = dict(hints)
    normalized_hints["keywords"] = keywords
    normalized_hints["priority"] = _clamp_float(hints.get("priority"), default=0.5)

    payload: dict[str, Any] = {
        "id": _candidate_id(candidate, span=span, index=index),
        "status": "candidate",
        "memory_type": memory_type,
        "scope": normalized_scope,
        "trigger": _one_line(candidate.get("trigger")),
        "lesson": _one_line(candidate.get("lesson")),
        "recommendation": _one_line(candidate.get("recommendation")),
        "anti_pattern": _one_line(candidate.get("anti_pattern")),
        "counterfactual": _one_line(candidate.get("counterfactual")),
        "quality": normalized_quality,
        "retrieval_hints": normalized_hints,
        "source": {
            "type": "reflection",
            "trace_path": trace_path,
            "span_id": span.span_id,
            "span_type": span.span_type,
            "seed": first.get("seed"),
            "step_start": first.get("step"),
            "step_end": span.last_event.get("step"),
            "event_count": len(span.events),
            "retrieved_memory_ids": _unique_ids(
                memory_id
                for event in span.events
                for memory_id in (_agent(event).get("retrieved_memory_ids") or [])
            ),
            "used_memory_ids": _unique_ids(
                memory_id for event in span.events for memory_id in (_agent(event).get("used_memory_ids") or [])
            ),
        },
    }
    if not payload["recommendation"] and payload["lesson"]:
        payload["recommendation"] = payload["lesson"]
    return payload


def promote_candidates(
    *,
    candidates_path: str | Path,
    active_path: str | Path,
    min_confidence: float = 0.35,
    limit: int = 0,
    reviewed_only: bool = False,
    dry_run: bool = False,
    include_ids: set[str] | None = None,
) -> dict[str, Any]:
    candidates = _read_jsonl(candidates_path, missing_ok=False)
    active = _read_jsonl(active_path, missing_ok=True)
    existing_ids = {str(entry.get("id")) for entry in active if isinstance(entry, dict) and entry.get("id")}
    existing_fingerprints = {_entry_fingerprint(entry) for entry in active if isinstance(entry, dict)}

    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("id") or "")
        if include_ids is not None and candidate_id not in include_ids:
            continue
        status = str(candidate.get("status") or "candidate")
        if reviewed_only and status != "reviewed":
            skipped.append({"id": candidate_id, "reason": "not_reviewed"})
            continue

        active_entry = dict(candidate)
        active_entry["status"] = "active"
        active_entry["scope"] = _canonical_scope_for_promotion(active_entry)
        active_entry.setdefault("quality", {})
        if not isinstance(active_entry["quality"], dict):
            active_entry["quality"] = {}
        active_entry["quality"]["promoted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            entry = ExperienceMemoryEntry.from_dict(active_entry)
        except Exception as exc:  # noqa: BLE001 - candidate validation result is part of summary.
            skipped.append({"id": candidate_id, "reason": f"invalid:{exc}"})
            continue
        if entry.confidence < min_confidence:
            skipped.append({"id": candidate_id, "reason": "low_confidence"})
            continue
        if entry.id in existing_ids:
            skipped.append({"id": entry.id, "reason": "duplicate_id"})
            continue
        fingerprint = _entry_fingerprint(active_entry)
        if fingerprint in existing_fingerprints:
            skipped.append({"id": entry.id, "reason": "duplicate_content"})
            continue

        promoted.append(active_entry)
        existing_ids.add(entry.id)
        existing_fingerprints.add(fingerprint)
        if limit > 0 and len(promoted) >= limit:
            break

    if not dry_run:
        append_jsonl(active_path, promoted, overwrite=False)

    return {
        "candidates_path": str(candidates_path),
        "active_path": str(active_path),
        "promoted": len(promoted),
        "skipped": skipped,
        "dry_run": dry_run,
        "include_ids": sorted(include_ids) if include_ids is not None else None,
    }


def _canonical_scope_for_promotion(entry: dict[str, Any]) -> dict[str, Any]:
    scope = entry.get("scope") if isinstance(entry.get("scope"), dict) else {}
    normalized = dict(scope)
    source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
    if source.get("span_type") == "combat":
        normalized["phase"] = "combat_play"
    return normalized


def learn_from_trace(
    *,
    trace_path: str | Path,
    candidate_out: str | Path,
    active_path: str | Path,
    chat_fn: ChatFn,
    max_spans: int = 0,
    max_candidates_per_span: int = 2,
    max_events_per_span: int = 40,
    overwrite_candidates: bool = False,
    promote: bool = True,
    min_confidence: float = 0.35,
    promote_limit: int = 0,
    reviewed_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    events = load_event_trace(trace_path)
    return learn_from_events(
        events,
        trace_path=str(trace_path),
        candidate_out=candidate_out,
        active_path=active_path,
        chat_fn=chat_fn,
        max_spans=max_spans,
        max_candidates_per_span=max_candidates_per_span,
        max_events_per_span=max_events_per_span,
        overwrite_candidates=overwrite_candidates,
        promote=promote,
        min_confidence=min_confidence,
        promote_limit=promote_limit,
        reviewed_only=reviewed_only,
        dry_run=dry_run,
    )


def learn_from_events(
    events: list[dict[str, Any]],
    *,
    trace_path: str,
    candidate_out: str | Path,
    active_path: str | Path,
    chat_fn: ChatFn,
    max_spans: int = 0,
    max_candidates_per_span: int = 2,
    max_events_per_span: int = 40,
    overwrite_candidates: bool = False,
    promote: bool = True,
    min_confidence: float = 0.35,
    promote_limit: int = 0,
    reviewed_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    reflection = reflect_events(
        events,
        trace_path=trace_path,
        candidate_out=candidate_out,
        chat_fn=chat_fn,
        max_spans=max_spans,
        max_candidates_per_span=max_candidates_per_span,
        max_events_per_span=max_events_per_span,
        overwrite=overwrite_candidates,
    )
    candidate_ids = {str(candidate_id) for candidate_id in reflection.get("candidate_ids", []) if candidate_id}
    promotion: dict[str, Any] | None = None
    if promote:
        promotion = promote_candidates(
            candidates_path=candidate_out,
            active_path=active_path,
            min_confidence=min_confidence,
            limit=promote_limit,
            reviewed_only=reviewed_only,
            dry_run=dry_run,
            include_ids=candidate_ids,
        )
    return {
        "trace_path": str(trace_path),
        "candidate_out": str(candidate_out),
        "active_path": str(active_path),
        "reflection": reflection,
        "promotion": promotion,
    }


def append_jsonl(path: str | Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with target.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def reflect_main(argv: list[str] | None = None) -> int:
    _load_benchmark_env_if_available()
    parser = argparse.ArgumentParser(description="Generate experience-memory candidates from compact event traces.")
    parser.add_argument("--trace", required=True, help="Compact experience event trace JSONL path")
    parser.add_argument("--candidate-out", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--base-url", default=_env_str("STS2_BENCH_BASE_URL"))
    parser.add_argument("--model", default=_env_str("STS2_BENCH_MODEL"))
    parser.add_argument("--api-key", default=_env_str("DEEPSEEK_API_KEY", "local", "OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=float(_env_str("STS2_BENCH_LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--timeout", type=float, default=float(_env_str("STS2_BENCH_LLM_TIMEOUT", "120.0")))
    parser.add_argument("--max-retries", type=int, default=int(_env_str("STS2_BENCH_LLM_MAX_RETRIES", "2")))
    parser.add_argument("--max-spans", type=int, default=0, help="0 means all spans")
    parser.add_argument("--max-candidates-per-span", type=int, default=2)
    parser.add_argument("--max-events-per-span", type=int, default=40)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-prompts", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    events = load_event_trace(args.trace)
    spans = combat_spans(events)
    if args.max_spans > 0:
        spans = spans[: args.max_spans]

    if args.dry_run:
        for span in spans:
            messages = build_reflection_messages(
                span,
                max_candidates=args.max_candidates_per_span,
                max_events=args.max_events_per_span,
            )
            print(json.dumps({"span_id": span.span_id, "messages": messages}, ensure_ascii=False), flush=True)
        print(json.dumps({"trace_path": args.trace, "spans": len(spans), "dry_run": True}, ensure_ascii=False))
        return 0

    if not args.base_url or not args.model:
        parser.error("--base-url and --model are required unless --dry-run is used")

    chat_fn = _openai_chat_fn(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    if args.print_prompts:
        chat_fn = _printing_chat_fn(chat_fn)

    summary = reflect_trace(
        trace_path=args.trace,
        candidate_out=args.candidate_out,
        chat_fn=chat_fn,
        max_spans=args.max_spans,
        max_candidates_per_span=args.max_candidates_per_span,
        max_events_per_span=args.max_events_per_span,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not summary["errors"] else 1


def promote_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote reviewed experience candidates into the active memory store.")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--active", default=DEFAULT_ACTIVE_PATH)
    parser.add_argument("--ids", nargs="*", help="Optional candidate ids to promote")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--reviewed-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    summary = promote_candidates(
        candidates_path=args.candidates,
        active_path=args.active,
        min_confidence=args.min_confidence,
        limit=args.limit,
        reviewed_only=args.reviewed_only,
        dry_run=args.dry_run,
        include_ids=set(args.ids) if args.ids else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def learn_main(argv: list[str] | None = None) -> int:
    _load_benchmark_env_if_available()
    parser = argparse.ArgumentParser(description="Reflect a compact trace and optionally promote new memories.")
    parser.add_argument("--trace", required=True, help="Compact experience event trace JSONL path")
    parser.add_argument("--candidate-out", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--active", default=DEFAULT_ACTIVE_PATH)
    parser.add_argument("--base-url", default=_env_str("STS2_BENCH_BASE_URL"))
    parser.add_argument("--model", default=_env_str("STS2_BENCH_MODEL"))
    parser.add_argument("--api-key", default=_env_str("DEEPSEEK_API_KEY", "local", "OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=float(_env_str("STS2_BENCH_LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--timeout", type=float, default=float(_env_str("STS2_BENCH_LLM_TIMEOUT", "120.0")))
    parser.add_argument("--max-retries", type=int, default=int(_env_str("STS2_BENCH_LLM_MAX_RETRIES", "2")))
    parser.add_argument("--max-spans", type=int, default=0, help="0 means all spans")
    parser.add_argument("--max-candidates-per-span", type=int, default=2)
    parser.add_argument("--max-events-per-span", type=int, default=40)
    parser.add_argument("--overwrite-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--promote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--promote-limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--reviewed-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-prompts", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    events = load_event_trace(args.trace)
    spans = combat_spans(events)
    if args.max_spans > 0:
        spans = spans[: args.max_spans]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "trace_path": args.trace,
                    "candidate_out": args.candidate_out,
                    "active_path": args.active,
                    "events": len(events),
                    "spans": len(spans),
                    "promote": args.promote,
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    if not args.base_url or not args.model:
        parser.error("--base-url and --model are required unless --dry-run is used")

    chat_fn = _openai_chat_fn(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    if args.print_prompts:
        chat_fn = _printing_chat_fn(chat_fn)

    summary = learn_from_trace(
        trace_path=args.trace,
        candidate_out=args.candidate_out,
        active_path=args.active,
        chat_fn=chat_fn,
        max_spans=args.max_spans,
        max_candidates_per_span=args.max_candidates_per_span,
        max_events_per_span=args.max_events_per_span,
        overwrite_candidates=args.overwrite_candidates,
        promote=args.promote,
        min_confidence=args.min_confidence,
        promote_limit=args.promote_limit,
        reviewed_only=args.reviewed_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    errors = summary["reflection"].get("errors") or []
    return 0 if not errors else 1


def _load_benchmark_env_if_available() -> None:
    try:
        from ..run_benchmark import load_benchmark_env
    except Exception:
        return
    try:
        load_benchmark_env()
    except Exception:
        return


def _openai_chat_fn(
    *,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> ChatFn:
    agent = OpenAICompatAgent(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=0,
    )

    def chat(messages: list[dict[str, str]]) -> str:
        request_messages = list(messages)
        last_error = ""
        for _attempt in range(max_retries + 1):
            try:
                response = agent._chat(request_messages)
                text = response["choices"][0]["message"]["content"]
                parse_json_object(text)
                return text
            except Exception as exc:  # noqa: BLE001 - correction loop mirrors action agent behavior.
                last_error = str(exc)
                request_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid: {last_error}. "
                            "Return only valid JSON matching the requested schema."
                        ),
                    }
                )
        raise ValueError(f"reflection LLM response stayed invalid: {last_error}")

    return chat


def _printing_chat_fn(chat_fn: ChatFn) -> ChatFn:
    def chat(messages: list[dict[str, str]]) -> str:
        print(json.dumps({"reflection_prompt": messages}, ensure_ascii=False), flush=True)
        return chat_fn(messages)

    return chat


def _span_prompt_payload(span: TraceSpan, *, max_events: int) -> dict[str, Any]:
    events = span.events[:max_events]
    return {
        "span_id": span.span_id,
        "span_type": span.span_type,
        "truncated": len(span.events) > len(events),
        "summary": _span_summary(span),
        "events": events,
    }


def _span_summary(span: TraceSpan) -> dict[str, Any]:
    first = span.first_event
    last = span.last_event
    return {
        "seed": first.get("seed"),
        "character": first.get("character"),
        "ascension": first.get("ascension"),
        "act": first.get("act"),
        "floor": first.get("floor"),
        "room_type": first.get("room_type"),
        "encounter": first.get("encounter"),
        "step_start": first.get("step"),
        "step_end": last.get("step"),
        "hp_bucket_before": first.get("hp_bucket_before"),
        "hp_bucket_after": last.get("hp_bucket_after"),
        "final_decision": _outcome(last).get("decision_after"),
        "actions": [_action(event).get("label") for event in span.events],
        "fallback_count": sum(1 for event in span.events if _agent(event).get("fallback")),
    }


def _make_combat_span(events: list[dict[str, Any]]) -> TraceSpan:
    first = events[0]
    span_id = ":".join(
        [
            "combat",
            _slug(first.get("seed") or "unknown_seed"),
            str(first.get("act") or "act_unknown"),
            str(first.get("floor") or "floor_unknown"),
            _slug(first.get("encounter") or "unknown_encounter"),
            str(first.get("step") or 0),
            str(events[-1].get("step") or 0),
        ]
    )
    return TraceSpan(span_id=span_id, span_type="combat", events=list(events))


def _is_combat_event(event: dict[str, Any]) -> bool:
    decision = str(event.get("decision") or "")
    return decision == "combat_play"


def _combat_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("seed"),
        event.get("character"),
        event.get("ascension"),
        event.get("act"),
        event.get("floor"),
        event.get("room_type"),
        event.get("encounter"),
    )


def _candidate_id(candidate: dict[str, Any], *, span: TraceSpan, index: int) -> str:
    explicit = _one_line(candidate.get("id") or candidate.get("memory_id"))
    if explicit:
        return explicit
    basis = {
        "span_id": span.span_id,
        "index": index,
        "memory_type": candidate.get("memory_type"),
        "trigger": candidate.get("trigger"),
        "lesson": candidate.get("lesson"),
        "recommendation": candidate.get("recommendation"),
    }
    digest = hashlib.sha1(json.dumps(basis, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    first = span.first_event
    character = _slug(first.get("character") or "global")
    return f"exp_{character}_{digest}"


def _read_jsonl(path: str | Path, *, missing_ok: bool) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        if missing_ok:
            return []
        raise FileNotFoundError(str(target))
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{target}:{line_number} must be a JSON object")
        rows.append(payload)
    return rows


def _entry_fingerprint(entry: dict[str, Any]) -> str:
    payload = {
        "memory_type": entry.get("memory_type"),
        "scope": entry.get("scope") or {},
        "trigger": _one_line(entry.get("trigger")).lower(),
        "recommendation": _one_line(entry.get("recommendation")).lower(),
        "anti_pattern": _one_line(entry.get("anti_pattern")).lower(),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _action(event: dict[str, Any]) -> dict[str, Any]:
    action = event.get("action")
    return action if isinstance(action, dict) else {}


def _agent(event: dict[str, Any]) -> dict[str, Any]:
    agent = event.get("agent")
    return agent if isinstance(agent, dict) else {}


def _outcome(event: dict[str, Any]) -> dict[str, Any]:
    outcome = event.get("outcome")
    return outcome if isinstance(outcome, dict) else {}


def _keyword_list(values: list[Any]) -> list[str]:
    keywords: list[str] = []
    for value in values:
        keyword = _slug(value)
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    return keywords[:16]


def _unique_ids(values: Any) -> list[str]:
    unique: list[str] = []
    for value in values:
        text = _one_line(value)
        if text and text not in unique:
            unique.append(text)
    return unique


def _clamp_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "_", _one_line(value).lower()).strip("_")


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _env_str(name: str, default: str | None = None, fallback: str | None = None) -> str | None:
    for key in (name, fallback):
        if key and os.environ.get(key):
            return os.environ[key]
    return default


def _main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m sts2_bench.experience.reflection {reflect,promote} ...", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "reflect":
        return reflect_main(rest)
    if command == "promote":
        return promote_main(rest)
    if command == "learn":
        return learn_main(rest)
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
