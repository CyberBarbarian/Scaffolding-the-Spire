"""Verify and continue a Direct-CLI trajectory without replaying model calls.

``--verify-only`` starts the same source-locked engine/CLI, feeds the already
accepted model outputs from the source attempt locally, and stops at the first
unresolved provider decision.  It never loads a credential or calls a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_bench.direct_cli_baseline import (  # noqa: E402
    ModelReply,
    OpenAICompatTerminalModel,
    OUTPUT_RETRY_MESSAGE,
    _load_env_file,
    _merge_provider_usage,
    _provider_response_metadata,
    load_human_cli,
    run_direct_cli,
)
from sts2_bench.direct_cli_experiment import (  # noqa: E402
    _audit_terminal_decision_requests,
    _read_and_verify_source_lock,
    _validate_attempt_artifacts,
    build_source_lock,
)
from sts2_bench.direct_cli_continuation import (  # noqa: E402
    continuation_execution_identity as _formal_continuation_execution_identity,
    _predispatch_failure_chain as _formal_predispatch_failure_chain,
    validate_continuation_evidence as _formal_validate_continuation_evidence,
)
from sts2_bench.experiment_identity import hash_file, hash_json  # noqa: E402


class ContinuationError(RuntimeError):
    pass


class ReplayBoundaryReached(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ContinuationError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContinuationError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _read_and_verify_source_lock_for_execution(
    live_root: Path,
    execution_sts2_cli_root: Path | None = None,
) -> dict[str, Any]:
    """Verify the original lock, or an exact isolated relocation of its CLI bytes."""

    freeze_path = live_root.resolve(strict=True) / "source_lock.json"
    if not freeze_path.is_file():
        frozen = _read_and_verify_source_lock(live_root)
        if (
            execution_sts2_cli_root is not None
            and execution_sts2_cli_root.resolve(strict=True)
            != Path(str(frozen.get("sts2_cli_root") or "")).resolve()
        ):
            raise ContinuationError(
                "Requested sts2-cli root does not match the verified source lock"
            )
        return frozen
    frozen = _read_json(freeze_path)
    recorded_root = Path(str(frozen.get("sts2_cli_root") or "")).resolve()
    if execution_sts2_cli_root is None:
        execution_sts2_cli_root = recorded_root
    execution_root = execution_sts2_cli_root.resolve(strict=True)
    if execution_root == recorded_root:
        return _read_and_verify_source_lock(live_root)

    protocol_path = Path(str(frozen.get("protocol_path") or "")).resolve(strict=True)
    rebuilt = build_source_lock(protocol_path, sts2_cli_root=execution_root)
    source_tree_identity = hash_json(
        [
            {
                "label": item["label"],
                "sha256": item["sha256"],
                "size": item["size"],
            }
            for item in rebuilt["inventory"]
            if item.get("tracked") is True
        ]
    )
    invariant_keys = (
        "schema_version",
        "identity_sha256",
        "protocol_sha256",
        "runtime",
        "files",
        "game_rules",
    )
    if (
        any(rebuilt.get(key) != frozen.get(key) for key in invariant_keys)
        or source_tree_identity != frozen.get("source_tree_identity_sha256")
    ):
        raise ContinuationError(
            "Alternate sts2-cli root does not reproduce the locked source identity"
        )

    materialization_path = execution_root.parent / "recovery_materialization.json"
    materialization = _read_json(materialization_path)
    if (
        materialization.get("source_lock_path") != str(freeze_path)
        or materialization.get("source_lock_sha256") != frozen.get("identity_sha256")
        or materialization.get("source_tree_identity_sha256")
        != frozen.get("source_tree_identity_sha256")
        or Path(
            str(materialization.get("materialized_sts2_cli_root") or "")
        ).resolve()
        != execution_root
        or materialization.get("verification_mismatch_count") != 0
        or materialization.get("formal_sts2_cli_root_modified") is not False
        or materialization.get("provider_calls") != 0
        or materialization.get("credential_file_read") is not False
    ):
        raise ContinuationError("Alternate sts2-cli materialization record is invalid")

    return {
        **frozen,
        "inventory": rebuilt["inventory"],
        "sts2_cli_root": str(execution_root),
        "source_relocation": {
            "status": "verified_exact_locked_bytes",
            "recorded_sts2_cli_root": str(recorded_root),
            "execution_sts2_cli_root": str(execution_root),
            "materialization_path": str(materialization_path.resolve()),
            "materialization_sha256": hash_file(materialization_path),
            "source_lock_record_sha256": hash_file(freeze_path),
        },
    }


def _raw_jsonl_prefix_sha256(path: Path, row_count: int) -> str:
    if row_count < 0:
        raise ContinuationError("Raw JSONL prefix row count cannot be negative")
    selected: list[bytes] = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.strip():
            continue
        selected.append(line)
        if len(selected) == row_count:
            break
    if len(selected) != row_count:
        raise ContinuationError(f"JSONL has fewer than {row_count} non-empty rows: {path}")
    return hashlib.sha256(b"".join(selected)).hexdigest()


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    return hash_json(messages)


def _provider_request_sha256(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return hash_json(payload)


def _canonical_engine_prefix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": row.get("step"),
            "type": row.get("type"),
            "data": row.get("data"),
        }
        for row in rows
    ]


def _recoverable_provider_failure_evidence_valid(row: dict[str, Any]) -> bool:
    failure_class = row.get("failure_class")
    if (
        row.get("status") != "failed"
        or failure_class
        not in {"provider_transport_error", "provider_configuration_error"}
        or not isinstance(row.get("error"), str)
        or not row["error"]
        or row.get("provider_metadata") not in (None, {})
    ):
        return False
    http_status = row.get("http_status")
    response_body = row.get("provider_response_body")
    response_sha256 = row.get("provider_response_sha256")
    response_encoding = row.get("provider_response_encoding")
    if http_status is None:
        return (
            response_body is None
            and response_sha256 is None
            and response_encoding is None
        )
    if (
        not isinstance(http_status, int)
        or not 400 <= http_status <= 599
        or not isinstance(response_body, str)
        or not isinstance(response_sha256, str)
    ):
        return False
    encoding = str(response_encoding or "utf-8")
    try:
        encoded = response_body.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    if hashlib.sha256(encoded).hexdigest() != response_sha256:
        return False
    if failure_class == "provider_configuration_error":
        if http_status != 400:
            return False
        try:
            payload = json.loads(response_body)
        except json.JSONDecodeError:
            return False
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return False
        markers = {
            str(error.get("code") or "").strip().lower(),
            str(error.get("type") or "").strip().lower(),
        }
        return "arrearage" in markers
    return True


def _canonical_provider_attempt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"type", "decision_index", "timestamp"}
    }


def _empty_content_protocol_attempt_valid(
    row: dict[str, Any],
    *,
    request_model: str,
) -> tuple[bool, dict[str, Any]]:
    if (
        row.get("status") != "failed"
        or row.get("failure_class") != "model_protocol_error"
        or row.get("error") != "ValueError: LLM response content is empty"
        or row.get("http_status") is not None
        or not isinstance(row.get("provider_metadata"), dict)
        or not isinstance(row.get("provider_response_body"), str)
        or not isinstance(row.get("provider_response_sha256"), str)
        or row.get("provider_response_encoding") != "utf-8"
        or not isinstance(row.get("latency_sec"), (int, float))
        or isinstance(row.get("latency_sec"), bool)
        or float(row["latency_sec"]) < 0
    ):
        return False, {}
    body_text = str(row["provider_response_body"])
    if hashlib.sha256(body_text.encode("utf-8")).hexdigest() != row[
        "provider_response_sha256"
    ]:
        return False, {}
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError:
        return False, {}
    if not isinstance(body, dict):
        return False, {}
    choices = body.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") != "stop"
        or not isinstance(choices[0].get("message"), dict)
        or choices[0]["message"].get("content") != ""
        or body.get("model") != request_model
        or _provider_response_metadata(body) != row["provider_metadata"]
    ):
        return False, {}
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return False, {}
    return True, usage


def _first_live_dispatch_messages(
    messages: list[dict[str, str]],
    boundary: dict[str, Any],
) -> list[dict[str, str]]:
    prelude = boundary.get("protocol_retry_prelude")
    if not isinstance(prelude, dict):
        return messages
    if (
        prelude.get("kind")
        != "audited_empty_content_then_retry_transport_failure_v1"
        or prelude.get("retry_message_sha256")
        != hashlib.sha256(OUTPUT_RETRY_MESSAGE.encode("utf-8")).hexdigest()
    ):
        raise ContinuationError("Protocol retry prelude identity is invalid")
    retry_messages = [
        *messages,
        {"role": "user", "content": OUTPUT_RETRY_MESSAGE},
    ]
    if (
        _messages_sha256(retry_messages)
        != boundary.get("first_live_dispatch_model_messages_sha256")
        or _provider_request_sha256(
            retry_messages,
            model=str(prelude["model"]),
            temperature=float(prelude["temperature"]),
            max_tokens=prelude.get("max_tokens"),
        )
        != boundary.get("first_live_dispatch_provider_request_sha256")
    ):
        raise ContinuationError("Protocol retry dispatch request cannot be reconstructed")
    return retry_messages


def _first_live_dispatch_request_sha256(boundary: dict[str, Any]) -> str:
    return str(
        boundary.get("first_live_dispatch_provider_request_sha256")
        or boundary["provider_request_sha256"]
    )


def _seam_first_dispatch_matches(
    seam: dict[str, Any],
    boundary: dict[str, Any],
) -> bool:
    if (
        "first_live_dispatch_model_messages_sha256" not in boundary
        and "first_live_dispatch_provider_request_sha256" not in boundary
    ):
        return True
    return (
        seam.get("first_live_dispatch_model_messages_sha256")
        == boundary.get("first_live_dispatch_model_messages_sha256")
        and seam.get("first_live_dispatch_provider_request_sha256")
        == boundary.get("first_live_dispatch_provider_request_sha256")
    )


def _canonical_replayed_decisions(
    terminal_decisions: list[dict[str, Any]],
    provider_attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts_by_decision: dict[int, list[dict[str, Any]]] = {}
    for attempt in provider_attempts:
        attempts_by_decision.setdefault(int(attempt["decision_index"]), []).append(attempt)
    decisions: list[dict[str, Any]] = []
    for row in terminal_decisions:
        decision_index = int(row["decision_index"])
        attempts = [
            _canonical_provider_attempt(attempt)
            for attempt in attempts_by_decision.get(decision_index, [])
        ]
        decisions.append(
            {
                "decision_index": decision_index,
                "model_messages_sha256": row["model_messages_sha256"],
                "model_raw": row["model_raw"],
                "command": row["command"],
                "usage": row.get("usage") if isinstance(row.get("usage"), dict) else {},
                "latency_sec": float(row.get("latency_sec") or 0.0),
                "provider_attempt_count": int(row.get("provider_attempt_count") or 0),
                "provider_metadata": (
                    row.get("provider_metadata")
                    if isinstance(row.get("provider_metadata"), dict)
                    else {}
                ),
                "provider_attempts": attempts,
            }
        )
    return decisions


def _authoritative_decision_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_indexes = [
        index
        for index, row in enumerate(rows)
        if row.get("type") == "state"
        and isinstance(row.get("data"), dict)
        and row["data"].get("type") == "decision"
    ]
    if not decision_indexes:
        raise ContinuationError("Engine boundary has no authoritative decision state")
    decision_index = decision_indexes[-1]
    trailing = rows[decision_index + 1 :]
    if len(trailing) % 2:
        raise ContinuationError("Engine boundary has an incomplete informational query")
    for offset in range(0, len(trailing), 2):
        action = trailing[offset]
        response = trailing[offset + 1]
        if (
            action.get("type") != "action"
            or not isinstance(action.get("data"), dict)
            or action["data"].get("cmd") != "get_map"
            or response.get("type") != "state"
            or not isinstance(response.get("data"), dict)
            or response["data"].get("type") != "map"
        ):
            raise ContinuationError(
                "Engine boundary contains a state-changing tail after its decision"
            )
    return dict(rows[decision_index]["data"])


def _find_live_root(source_attempt: Path) -> Path:
    for parent in source_attempt.parents:
        if (parent / "source_lock.json").is_file():
            return parent
    raise ContinuationError("Source attempt is not beneath a Direct-CLI source lock")


def _boundary_messages(
    *,
    config: dict[str, Any],
    terminal: str,
    expected_sha256: str,
) -> tuple[int, list[dict[str, str]]]:
    marker = "\n  [log] Game log saved to"
    boundary_end = terminal.rfind(marker)
    if boundary_end < 0:
        raise ContinuationError("Cannot isolate the pre-cleanup terminal boundary")
    history_limit = int(config["history_chars"])
    visible = terminal[max(0, boundary_end - history_limit) : boundary_end]
    messages = [
        {"role": "system", "content": str(config["system_prompt"])},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + visible
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    if _messages_sha256(messages) != expected_sha256:
        raise ContinuationError("Terminal cleanup boundary does not reconstruct the failed request")
    return boundary_end, messages


def build_replay_plan(
    source_attempt: Path,
    execution_sts2_cli_root: Path | None = None,
) -> dict[str, Any]:
    source_attempt = source_attempt.resolve(strict=True)
    results_root = (ROOT / "results").resolve()
    if not source_attempt.is_relative_to(results_root):
        raise ContinuationError("Source attempt must remain beneath benchmark results")
    artifacts = source_attempt / "artifacts"
    required = (
        source_attempt / "attempt_manifest.json",
        source_attempt / "attempt_result.json",
        artifacts / "config.json",
        artifacts / "events.jsonl",
        artifacts / "terminal.txt",
        artifacts / "engine_trace.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ContinuationError(f"Source continuation artifacts are missing: {missing}")

    attempt_manifest = _read_json(source_attempt / "attempt_manifest.json")
    attempt_result = _read_json(source_attempt / "attempt_result.json")
    config = _read_json(artifacts / "config.json")
    events = _read_jsonl(artifacts / "events.jsonl")
    engine_rows = _read_jsonl(artifacts / "engine_trace.jsonl")
    terminal = (artifacts / "terminal.txt").read_text(encoding="utf-8")
    live_root = _find_live_root(source_attempt)
    source_lock = _read_and_verify_source_lock_for_execution(
        live_root,
        execution_sts2_cli_root,
    )
    if (
        attempt_result.get("termination_class")
        not in {"provider_transport_error", "provider_configuration_error"}
        or attempt_result.get("identity_stable") is not True
        or attempt_result.get("source_tree_stable") is not True
        or attempt_manifest.get("source_lock_sha256") != source_lock.get("identity_sha256")
        or attempt_manifest.get("source_tree_identity_sha256")
        != source_lock.get("source_tree_identity_sha256")
        or attempt_manifest.get("protocol_sha256") != source_lock.get("protocol_sha256")
    ):
        raise ContinuationError("Attempt is not an identity-stable recoverable provider failure")

    terminal_decisions = [
        row for row in events if row.get("type") == "terminal_decision"
    ]
    boundary_index = len(terminal_decisions)
    if [row.get("decision_index") for row in terminal_decisions] != list(
        range(boundary_index)
    ):
        raise ContinuationError("Accepted terminal decisions are not one contiguous prefix")
    prefix_attempts = [
        row
        for row in events
        if row.get("type") == "provider_request_attempt"
        and isinstance(row.get("decision_index"), int)
        and row["decision_index"] < boundary_index
    ]
    prefix_audit = _audit_terminal_decision_requests(
        terminal_decisions,
        provider_attempts=prefix_attempts,
        protocol_errors=[],
        terminal=terminal,
        history_limit=int(config["history_chars"]),
        request_model=str(config["model"]),
        request_temperature=float(config["temperature"]),
        request_max_tokens=config.get("max_tokens"),
        request_retries=int(config["request_retries"]),
        require_prompt_contract=True,
    )
    if prefix_audit.get("valid") is not True:
        raise ContinuationError("Accepted Direct-CLI request prefix failed reconstruction audit")

    boundary_attempts = [
        row
        for row in events
        if row.get("type") == "provider_request_attempt"
        and row.get("decision_index") == boundary_index
    ]
    boundary_has_terminal_decision = any(
        row.get("type") == "terminal_decision"
        and row.get("decision_index") == boundary_index
        for row in events
    )
    boundary_has_protocol_event = any(
        row.get("type") == "model_protocol_error"
        and (
            not isinstance(row.get("decision_index"), int)
            or row["decision_index"] >= boundary_index
        )
        for row in events
    )
    protocol_retry_prelude: dict[str, Any] | None = None
    boundary_end = -1
    boundary_messages: list[dict[str, str]] = []
    if (
        len(boundary_attempts) == 2
        and [row.get("provider_attempt_index") for row in boundary_attempts] == [0, 1]
        and int(config["request_retries"]) == 1
        and not boundary_has_terminal_decision
        and not boundary_has_protocol_event
    ):
        empty_attempt, retry_failure = boundary_attempts
        empty_valid, empty_usage = _empty_content_protocol_attempt_valid(
            empty_attempt,
            request_model=str(config["model"]),
        )
        if (
            empty_valid
            and retry_failure.get("failure_class") == "provider_transport_error"
            and _recoverable_provider_failure_evidence_valid(retry_failure)
        ):
            try:
                candidate_end, candidate_messages = _boundary_messages(
                    config=config,
                    terminal=terminal,
                    expected_sha256=str(empty_attempt["model_messages_sha256"]),
                )
            except (ContinuationError, KeyError, TypeError, ValueError):
                candidate_end, candidate_messages = -1, []
            candidate_request_sha256 = (
                _provider_request_sha256(
                    candidate_messages,
                    model=str(config["model"]),
                    temperature=float(config["temperature"]),
                    max_tokens=config.get("max_tokens"),
                )
                if candidate_messages
                else None
            )
            retry_messages = [
                *candidate_messages,
                {"role": "user", "content": OUTPUT_RETRY_MESSAGE},
            ]
            retry_messages_sha256 = (
                _messages_sha256(retry_messages) if candidate_messages else None
            )
            retry_request_sha256 = (
                _provider_request_sha256(
                    retry_messages,
                    model=str(config["model"]),
                    temperature=float(config["temperature"]),
                    max_tokens=config.get("max_tokens"),
                )
                if candidate_messages
                else None
            )
            if (
                candidate_request_sha256
                == empty_attempt.get("provider_request_sha256")
                and retry_messages_sha256
                == retry_failure.get("model_messages_sha256")
                and retry_request_sha256
                == retry_failure.get("provider_request_sha256")
            ):
                boundary_end = candidate_end
                boundary_messages = candidate_messages
                protocol_retry_prelude = {
                    "kind": (
                        "audited_empty_content_then_retry_transport_failure_v1"
                    ),
                    "model": str(config["model"]),
                    "temperature": float(config["temperature"]),
                    "max_tokens": config.get("max_tokens"),
                    "retry_message_sha256": hashlib.sha256(
                        OUTPUT_RETRY_MESSAGE.encode("utf-8")
                    ).hexdigest(),
                    "source_provider_attempts": [
                        _canonical_provider_attempt(row) for row in boundary_attempts
                    ],
                    "source_provider_attempts_sha256": hash_json(
                        [
                            _canonical_provider_attempt(row)
                            for row in boundary_attempts
                        ]
                    ),
                    "replayed_empty_provider_attempt": (
                        _canonical_provider_attempt(empty_attempt)
                    ),
                    "replayed_empty_usage": empty_usage,
                    "replayed_empty_latency_sec": float(
                        empty_attempt["latency_sec"]
                    ),
                }
    if protocol_retry_prelude is None:
        if (
            not boundary_attempts
            or [row.get("provider_attempt_index") for row in boundary_attempts]
            != list(range(len(boundary_attempts)))
            or any(
                not _recoverable_provider_failure_evidence_valid(row)
                for row in boundary_attempts
            )
            or boundary_has_terminal_decision
            or boundary_has_protocol_event
        ):
            raise ContinuationError(
                "Unresolved boundary is not a recoverable provider failure"
            )
        message_hashes = {
            row.get("model_messages_sha256") for row in boundary_attempts
        }
        request_hashes = {
            row.get("provider_request_sha256") for row in boundary_attempts
        }
        if (
            len(message_hashes) != 1
            or None in message_hashes
            or len(request_hashes) != 1
            or None in request_hashes
        ):
            raise ContinuationError(
                "Boundary provider attempts do not bind one exact request"
            )
        boundary_messages_sha256 = str(next(iter(message_hashes)))
        boundary_request_sha256 = str(next(iter(request_hashes)))
        boundary_end, boundary_messages = _boundary_messages(
            config=config,
            terminal=terminal,
            expected_sha256=boundary_messages_sha256,
        )
        reconstructed_request_sha256 = _provider_request_sha256(
            boundary_messages,
            model=str(config["model"]),
            temperature=float(config["temperature"]),
            max_tokens=config.get("max_tokens"),
        )
        if reconstructed_request_sha256 != boundary_request_sha256:
            raise ContinuationError(
                "Boundary provider request hash cannot be reconstructed"
            )
        first_live_dispatch_messages_sha256 = boundary_messages_sha256
        first_live_dispatch_request_sha256 = boundary_request_sha256
    else:
        boundary_messages_sha256 = str(
            boundary_attempts[0]["model_messages_sha256"]
        )
        boundary_request_sha256 = str(
            boundary_attempts[0]["provider_request_sha256"]
        )
        first_live_dispatch_messages_sha256 = str(
            boundary_attempts[1]["model_messages_sha256"]
        )
        first_live_dispatch_request_sha256 = str(
            boundary_attempts[1]["provider_request_sha256"]
        )

    quit_indexes = [
        index
        for index, row in enumerate(engine_rows)
        if row.get("type") == "action"
        and isinstance(row.get("data"), dict)
        and row["data"].get("cmd") == "quit"
    ]
    if len(quit_indexes) != 1 or quit_indexes[0] != len(engine_rows) - 2:
        raise ContinuationError("Engine trace does not end in one isolated cleanup quit")
    engine_boundary_index = quit_indexes[0]
    engine_prefix = engine_rows[:engine_boundary_index]
    if not engine_prefix:
        raise ContinuationError("Engine boundary is empty")
    boundary_state = _authoritative_decision_state(engine_prefix)

    decisions = _canonical_replayed_decisions(terminal_decisions, prefix_attempts)
    return {
        "schema_version": 2,
        "mode": "direct_cli_trajectory_continuation",
        "adapter_sha256": hash_file(Path(__file__).resolve()),
        "source_attempt": str(source_attempt),
        "source_artifacts": str(artifacts),
        "source_live_root": str(live_root),
        "source_hashes": {
            path.name: hash_file(path)
            for path in (
                artifacts / "config.json",
                artifacts / "events.jsonl",
                artifacts / "terminal.txt",
                artifacts / "engine_trace.jsonl",
                source_attempt / "attempt_manifest.json",
                source_attempt / "attempt_result.json",
            )
        },
        "source_lock_sha256": source_lock["identity_sha256"],
        "source_tree_identity_sha256": source_lock["source_tree_identity_sha256"],
        "protocol_sha256": source_lock["protocol_sha256"],
        "config": {
            key: config.get(key)
            for key in (
                "model",
                "base_url",
                "temperature",
                "timeout",
                "max_tokens",
                "request_retries",
                "history_chars",
                "max_decisions",
                "seed",
                "character",
                "ascension",
                "lang",
            )
        },
        "execution_sts2_cli_root": source_lock["sts2_cli_root"],
        "original_execution_sts2_cli_root": attempt_manifest[
            "execution_sts2_cli_root"
        ],
        "source_relocation": source_lock.get("source_relocation"),
        "accepted_decision_count": boundary_index,
        "accepted_provider_attempt_count": len(prefix_attempts),
        "decisions": decisions,
        "boundary": {
            "decision_index": boundary_index,
            "failed_provider_attempt_count": len(boundary_attempts),
            "model_messages_sha256": boundary_messages_sha256,
            "provider_request_sha256": boundary_request_sha256,
            "first_live_dispatch_model_messages_sha256": (
                first_live_dispatch_messages_sha256
            ),
            "first_live_dispatch_provider_request_sha256": (
                first_live_dispatch_request_sha256
            ),
            "protocol_retry_prelude": protocol_retry_prelude,
            "terminal_end": boundary_end,
            "terminal_prefix_sha256": hashlib.sha256(
                terminal[:boundary_end].encode("utf-8")
            ).hexdigest(),
            "engine_prefix_row_count": len(engine_prefix),
            "engine_prefix_sha256": hash_json(
                _canonical_engine_prefix(engine_prefix)
            ),
            "source_engine_prefix_raw_sha256": _raw_jsonl_prefix_sha256(
                artifacts / "engine_trace.jsonl",
                len(engine_prefix),
            ),
            "state_sha256": hash_json(boundary_state),
        },
        "credential_values_recorded": False,
    }


class ReplayThenLiveModel(OpenAICompatTerminalModel):
    """Return accepted source replies locally, then stop or delegate at the seam."""

    def __init__(
        self,
        plan: dict[str, Any],
        *,
        replay_engine_trace: Path,
        verify_only: bool,
        source_lock_verifier: Callable[[], dict[str, Any]],
        seam_persistor: Callable[[dict[str, Any]], None] | None = None,
        credential_loader: Callable[[], str] | None = None,
        dispatch_persistor: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        config = plan["config"]
        super().__init__(
            base_url=str(config["base_url"]),
            model=str(config["model"]),
            api_key="VERIFY_ONLY_NO_CREDENTIAL",
            temperature=float(config["temperature"]),
            timeout=float(config.get("timeout") or 1.0),
            max_tokens=config.get("max_tokens"),
            request_retries=int(config["request_retries"]),
        )
        self.plan = plan
        self.replay_engine_trace = replay_engine_trace
        self.verify_only = verify_only
        self.source_lock_verifier = source_lock_verifier
        self.seam_persistor = seam_persistor
        self.credential_loader = credential_loader
        self.dispatch_persistor = dispatch_persistor
        self.cursor = 0
        self.replayed_decisions = 0
        self.provider_calls = 0
        self.credential_file_read = False
        self.seam_proof: dict[str, Any] | None = None
        self.dispatched_decisions: set[int] = set()

    def _verify_adapter_and_source_artifacts(self) -> None:
        if hash_file(Path(__file__).resolve()) != self.plan["adapter_sha256"]:
            raise ContinuationError("Continuation adapter drifted before the seam")
        source_attempt = Path(self.plan["source_attempt"])
        artifacts = Path(self.plan["source_artifacts"])
        current_paths = {
            "config.json": artifacts / "config.json",
            "events.jsonl": artifacts / "events.jsonl",
            "terminal.txt": artifacts / "terminal.txt",
            "engine_trace.jsonl": artifacts / "engine_trace.jsonl",
            "attempt_manifest.json": source_attempt / "attempt_manifest.json",
            "attempt_result.json": source_attempt / "attempt_result.json",
        }
        expected_hashes = self.plan["source_hashes"]
        if set(expected_hashes) != set(current_paths) or any(
            not path.is_file() or hash_file(path) != expected_hashes[name]
            for name, path in current_paths.items()
        ):
            raise ContinuationError("Source attempt artifacts drifted before the seam")

    def _verify_source_lock(self) -> None:
        value = self.source_lock_verifier()
        expected = self.plan
        if (
            value.get("identity_sha256") != expected["source_lock_sha256"]
            or value.get("source_tree_identity_sha256")
            != expected["source_tree_identity_sha256"]
            or value.get("protocol_sha256") != expected["protocol_sha256"]
        ):
            raise ContinuationError("Source lock drifted before the continuation seam")

    def _verify_engine_seam(self) -> str:
        rows = _read_jsonl(self.replay_engine_trace)
        boundary = self.plan["boundary"]
        authoritative_state = _authoritative_decision_state(rows)
        if (
            len(rows) != boundary["engine_prefix_row_count"]
            or hash_json(_canonical_engine_prefix(rows))
            != boundary["engine_prefix_sha256"]
            or hash_json(authoritative_state) != boundary["state_sha256"]
        ):
            raise ContinuationError("Replayed engine trajectory diverged before provider use")
        return _raw_jsonl_prefix_sha256(self.replay_engine_trace, len(rows))

    def _complete_live(self, messages: list[dict[str, str]]) -> ModelReply:
        decision_index = self.cursor
        if decision_index in self.dispatched_decisions:
            raise ContinuationError(
                f"Provider dispatch already claimed for decision {decision_index}"
            )
        request_hash = _provider_request_sha256(
            messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        dispatch = {
            "status": "provider_dispatch_started",
            "decision_index": decision_index,
            "logical_provider_call_index": self.provider_calls,
            "model_messages_sha256": _messages_sha256(messages),
            "provider_request_sha256": request_hash,
            "started_at": _utc_now(),
        }
        if self.dispatch_persistor is not None:
            self.dispatch_persistor(dispatch)
        self.dispatched_decisions.add(decision_index)
        self.provider_calls += 1
        if not self.credential_file_read:
            if self.credential_loader is None:
                raise ContinuationError("Live continuation has no credential loader")
            api_key = self.credential_loader()
            if not api_key:
                raise ContinuationError("Live continuation credential is empty")
            self.api_key = api_key
            self.credential_file_read = True
        return super().complete(messages)

    def _complete_live_after_protocol_retry(
        self,
        messages: list[dict[str, str]],
        boundary: dict[str, Any],
    ) -> ModelReply:
        prelude = boundary.get("protocol_retry_prelude")
        if not isinstance(prelude, dict):
            raise ContinuationError("Protocol retry prelude is missing")
        if self.cursor in self.dispatched_decisions:
            raise ContinuationError(
                f"Provider dispatch already claimed for decision {self.cursor}"
            )
        retry_messages = _first_live_dispatch_messages(messages, boundary)
        replayed_attempt = prelude.get("replayed_empty_provider_attempt")
        if not isinstance(replayed_attempt, dict):
            raise ContinuationError("Replayed empty provider attempt is missing")
        original_observer = self.attempt_observer
        if original_observer is not None:
            original_observer(dict(replayed_attempt))

        def shifted_observer(value: dict[str, Any]) -> None:
            if value.get("provider_attempt_index") != 0:
                raise ContinuationError(
                    "Protocol retry live provider emitted an unexpected attempt index"
                )
            shifted = dict(value)
            shifted["provider_attempt_index"] = 1
            if original_observer is not None:
                original_observer(shifted)

        original_retries = self.request_retries
        self.request_retries = 0
        self.attempt_observer = shifted_observer
        try:
            reply = self._complete_live(retry_messages)
        finally:
            self.request_retries = original_retries
            self.attempt_observer = original_observer
        return ModelReply(
            text=reply.text,
            usage=_merge_provider_usage(
                dict(prelude.get("replayed_empty_usage") or {}),
                reply.usage,
            ),
            latency_sec=(
                float(prelude.get("replayed_empty_latency_sec") or 0.0)
                + reply.latency_sec
            ),
            provider_attempt_count=2,
            provider_metadata=dict(reply.provider_metadata or {}),
        )

    def complete(self, messages: list[dict[str, str]]) -> ModelReply:
        messages_hash = _messages_sha256(messages)
        decisions = self.plan["decisions"]
        if self.cursor < len(decisions):
            expected = decisions[self.cursor]
            if (
                expected.get("decision_index") != self.cursor
                or expected.get("model_messages_sha256") != messages_hash
            ):
                raise ContinuationError(
                    f"Replay prompt diverged at decision {self.cursor}"
                )
            self.cursor += 1
            self.replayed_decisions += 1
            if self.attempt_observer is not None:
                for attempt in expected.get("provider_attempts") or []:
                    if not isinstance(attempt, dict):
                        raise ContinuationError(
                            f"Replay provider evidence is malformed at decision {self.cursor - 1}"
                        )
                    self.attempt_observer(dict(attempt))
            return ModelReply(
                text=str(expected["model_raw"]),
                usage=dict(expected.get("usage") or {}),
                latency_sec=float(expected.get("latency_sec") or 0.0),
                provider_attempt_count=int(expected.get("provider_attempt_count") or 0),
                provider_metadata=dict(expected.get("provider_metadata") or {}),
            )

        boundary = self.plan["boundary"]
        if self.cursor > boundary["decision_index"]:
            reply = self._complete_live(messages)
            self.cursor += 1
            return reply
        request_hash = _provider_request_sha256(
            messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if (
            self.cursor != boundary["decision_index"]
            or messages_hash != boundary["model_messages_sha256"]
            or request_hash != boundary["provider_request_sha256"]
        ):
            raise ContinuationError("Unresolved provider request seam does not match")
        self._verify_adapter_and_source_artifacts()
        self._verify_source_lock()
        engine_prefix_raw_sha256 = self._verify_engine_seam()
        self.seam_proof = {
            "status": "verified",
            "decision_index": self.cursor,
            "replayed_decisions": self.replayed_decisions,
            "model_messages_sha256": messages_hash,
            "provider_request_sha256": request_hash,
            "first_live_dispatch_model_messages_sha256": boundary.get(
                "first_live_dispatch_model_messages_sha256",
                boundary["model_messages_sha256"],
            ),
            "first_live_dispatch_provider_request_sha256": (
                _first_live_dispatch_request_sha256(boundary)
            ),
            "engine_prefix_sha256": boundary["engine_prefix_sha256"],
            "engine_prefix_raw_sha256": engine_prefix_raw_sha256,
            "state_sha256": boundary["state_sha256"],
            "source_lock_sha256": self.plan["source_lock_sha256"],
            "source_tree_identity_sha256": self.plan[
                "source_tree_identity_sha256"
            ],
            "provider_calls": self.provider_calls,
            "credential_file_read": False,
            "credential_values_recorded": False,
            "verified_at": _utc_now(),
        }
        if self.seam_persistor is not None:
            self.seam_persistor(dict(self.seam_proof))
        if self.verify_only:
            raise ReplayBoundaryReached("verify-only Direct continuation seam reached")
        if isinstance(boundary.get("protocol_retry_prelude"), dict):
            reply = self._complete_live_after_protocol_retry(messages, boundary)
        else:
            reply = self._complete_live(messages)
        self.cursor += 1
        return reply


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The canonical Direct result roots already approach MAX_PATH on Windows.
    # A temp name derived from the 64-character claim filename can therefore
    # fail before ``os.replace`` even though the canonical destination works.
    # Keep the sibling name short while retaining enough randomness for
    # concurrent writers in the same directory.
    temporary = path.with_name(f".cw-{uuid.uuid4().hex[:12]}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ContinuationError(f"Continuation claim already exists: {path}") from exc


def _predispatch_failure_result(
    attempt_dir: Path,
    *,
    plan: dict[str, Any],
    claim_path: Path,
) -> dict[str, Any]:
    """Validate and materialize a local failure that happened before dispatch.

    The dispatch record is persisted before credentials are loaded or the HTTP
    call begins.  Consequently, an absent dispatch directory plus a verified
    zero-call seam is the fail-closed proof that retrying the same boundary
    cannot duplicate a provider request.
    """

    manifest_path = attempt_dir / "attempt_manifest.json"
    summary_path = attempt_dir / "artifacts" / "summary.json"
    events_path = attempt_dir / "artifacts" / "events.jsonl"
    config_path = attempt_dir / "artifacts" / "config.json"
    terminal_path = attempt_dir / "artifacts" / "terminal.txt"
    engine_trace_path = attempt_dir / "artifacts" / "engine_trace.jsonl"
    seam_path = attempt_dir / "continuation" / "seam_proof.json"
    required_paths = [
        manifest_path,
        summary_path,
        events_path,
        config_path,
        terminal_path,
        seam_path,
    ]
    if "engine_prefix_row_count" in plan["boundary"]:
        required_paths.append(engine_trace_path)
    if not all(path.is_file() for path in required_paths):
        raise ContinuationError(
            f"Existing continuation attempt is not a complete pre-dispatch failure: {attempt_dir}"
        )
    if (attempt_dir / "continuation" / "continuation_attestation.json").exists():
        raise ContinuationError("Existing continuation already has an execution attestation")
    dispatch_paths = sorted(
        (attempt_dir / "continuation" / "dispatches").glob("decision_*.json")
    )
    if dispatch_paths:
        raise ContinuationError("Existing continuation may already have dispatched a provider call")

    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    seam = _read_json(seam_path)
    claim = _read_json(claim_path)
    config = _read_json(config_path)
    terminal = terminal_path.read_text(encoding="utf-8")
    continuation = manifest.get("continuation") or {}
    accepted_count = int(plan["accepted_decision_count"])
    event_rows = _read_jsonl(events_path)
    decision_rows = [
        row
        for row in event_rows
        if row.get("type") == "terminal_decision"
    ]
    all_provider_attempt_rows = [
        row for row in event_rows if row.get("type") == "provider_request_attempt"
    ]
    provider_attempt_rows = [
        row for row in all_provider_attempt_rows
        if isinstance(row.get("decision_index"), int)
        and 0 <= row["decision_index"] < accepted_count
    ]
    expected_provider_attempt_count = int(
        plan.get(
            "accepted_provider_attempt_count",
            sum(
                len(decision.get("provider_attempts") or [])
                for decision in plan["decisions"]
            ),
        )
    )
    prefix_audit = _audit_terminal_decision_requests(
        decision_rows,
        provider_attempts=provider_attempt_rows,
        protocol_errors=[],
        terminal=terminal,
        history_limit=int(config["history_chars"]),
        request_model=str(config["model"]),
        request_temperature=float(config["temperature"]),
        request_max_tokens=config.get("max_tokens"),
        request_retries=int(config["request_retries"]),
        require_prompt_contract=True,
    )
    canonical_decisions = _canonical_replayed_decisions(
        decision_rows,
        provider_attempt_rows,
    )
    terminal_boundary_valid = True
    if "terminal_end" in plan["boundary"]:
        try:
            boundary_end, boundary_messages = _boundary_messages(
                config=config,
                terminal=terminal,
                expected_sha256=plan["boundary"]["model_messages_sha256"],
            )
            terminal_boundary_valid = (
                boundary_end == plan["boundary"]["terminal_end"]
                and hashlib.sha256(
                    terminal[:boundary_end].encode("utf-8")
                ).hexdigest()
                == plan["boundary"]["terminal_prefix_sha256"]
                and _provider_request_sha256(
                    boundary_messages,
                    model=str(config["model"]),
                    temperature=float(config["temperature"]),
                    max_tokens=config.get("max_tokens"),
                )
                == plan["boundary"]["provider_request_sha256"]
            )
        except (ContinuationError, KeyError, TypeError, ValueError):
            terminal_boundary_valid = False
    engine_boundary_valid = True
    engine_tail_valid = True
    if "engine_prefix_row_count" in plan["boundary"]:
        try:
            engine_rows = _read_jsonl(engine_trace_path)
            engine_row_count = int(plan["boundary"]["engine_prefix_row_count"])
            engine_prefix = engine_rows[:engine_row_count]
            engine_tail = engine_rows[engine_row_count:]
            engine_boundary_valid = (
                len(engine_prefix) == engine_row_count
                and hash_json(_canonical_engine_prefix(engine_prefix))
                == plan["boundary"]["engine_prefix_sha256"]
                and hash_json(_authoritative_decision_state(engine_prefix))
                == plan["boundary"]["state_sha256"]
            )
            engine_tail_valid = (
                not engine_tail
                or (
                    len(engine_tail) == 2
                    and engine_tail[0].get("type") == "action"
                    and isinstance(engine_tail[0].get("data"), dict)
                    and engine_tail[0]["data"].get("cmd") == "quit"
                    and engine_tail[1].get("type") == "state"
                    and isinstance(engine_tail[1].get("data"), dict)
                    and engine_tail[1]["data"].get("type") == "quit_result"
                    and engine_tail[1]["data"].get("success") is True
                )
            )
        except (ContinuationError, KeyError, TypeError, ValueError):
            engine_boundary_valid = False
            engine_tail_valid = False
    expected_indices = list(range(accepted_count))
    checks = {
        "execution_mode_is_continuation": (
            manifest.get("execution_mode") == "trajectory_continuation"
        ),
        "source_attempt_matches": (
            continuation.get("source_attempt") == plan["source_attempt"]
        ),
        "claim_matches": (
            continuation.get("claim_id") == claim.get("claim_id")
            and claim.get("source_attempt") == plan["source_attempt"]
            and claim.get("boundary_decision_index")
            == plan["boundary"]["decision_index"]
            and claim.get("boundary_provider_request_sha256")
            == _first_live_dispatch_request_sha256(plan["boundary"])
        ),
        "claim_has_no_dispatch": (
            claim.get("provider_dispatch_started") is False
            and not claim.get("last_dispatch")
        ),
        "seam_is_verified_zero_call": (
            seam.get("status") == "verified"
            and seam.get("decision_index") == plan["boundary"]["decision_index"]
            and seam.get("replayed_decisions") == accepted_count
            and seam.get("provider_calls") == 0
            and seam.get("credential_file_read") is False
            and seam.get("credential_values_recorded") is False
            and seam.get("model_messages_sha256")
            == plan["boundary"]["model_messages_sha256"]
            and seam.get("provider_request_sha256")
            == plan["boundary"]["provider_request_sha256"]
            and _seam_first_dispatch_matches(seam, plan["boundary"])
            and seam.get("engine_prefix_sha256")
            == plan["boundary"]["engine_prefix_sha256"]
            and seam.get("state_sha256") == plan["boundary"]["state_sha256"]
        ),
        "replayed_decisions_are_exact_prefix": (
            [row.get("decision_index") for row in decision_rows] == expected_indices
        ),
        "provider_rows_are_strictly_before_boundary": (
            len(provider_attempt_rows) == len(all_provider_attempt_rows)
        ),
        "provider_row_count_matches_plan": (
            len(all_provider_attempt_rows) == expected_provider_attempt_count
        ),
        "accepted_prefix_request_audit_valid": prefix_audit.get("valid") is True,
        "accepted_prefix_evidence_matches_plan": (
            canonical_decisions == plan["decisions"]
        ),
        "terminal_boundary_recomputed": terminal_boundary_valid,
        "engine_boundary_recomputed": engine_boundary_valid,
        "engine_tail_is_cleanup_only": engine_tail_valid,
        "summary_is_local_infrastructure_failure": (
            summary.get("formal_behavior_sample") is False
            and summary.get("termination_class") == "process_infrastructure_error"
            and isinstance(summary.get("error"), str)
            and bool(summary["error"])
        ),
        "no_dispatch_evidence": not dispatch_paths,
    }
    if not all(checks.values()):
        failed = [name for name, valid in checks.items() if not valid]
        raise ContinuationError(
            "Existing continuation is not safely retryable before provider dispatch: "
            + ", ".join(failed)
        )

    attestation_path = (
        attempt_dir / "continuation" / "predispatch_failure_attestation.json"
    )
    result_path = attempt_dir / "attempt_result.json"
    authorization_path = (
        attempt_dir / "continuation" / "predispatch_retry_authorization.json"
    )
    attestation_core = {
        "schema_version": 1,
        "status": "failed_before_provider_dispatch",
        "attempt": manifest.get("attempt"),
        "claim_id": claim.get("claim_id"),
        "source_attempt": plan["source_attempt"],
        "boundary_decision_index": plan["boundary"]["decision_index"],
        "replayed_decision_count": accepted_count,
        "provider_calls": 0,
        "credential_file_read": False,
        "dispatch_count": 0,
        "checks": checks,
        "summary_sha256": hash_file(summary_path),
        "seam_proof_sha256": hash_file(seam_path),
    }
    if attestation_path.is_file():
        attestation = _read_json(attestation_path)
        attestation_checks = attestation.get("checks")
        if (
            not isinstance(attestation.get("recorded_at"), str)
            or not isinstance(attestation_checks, dict)
            or not attestation_checks
            or any(value is not True for value in attestation_checks.values())
            or any(checks.get(name) is not True for name in attestation_checks)
            or any(
                attestation.get(key) != value
                for key, value in attestation_core.items()
                if key != "checks"
            )
        ):
            raise ContinuationError("Stored pre-dispatch failure attestation conflicts")
    else:
        if result_path.is_file():
            raise ContinuationError(
                "Stored pre-dispatch failure result has no attestation"
            )
        attestation = {**attestation_core, "recorded_at": _utc_now()}
        _atomic_create(attestation_path, attestation)

    current_attestation_sha256 = hash_file(attestation_path)
    if result_path.is_file():
        stored = _read_json(result_path)
        stored_validation = stored.get("continuation_validation") or {}
        stored_checks = stored_validation.get("checks")
        if (
            stored.get("retryable_before_provider_dispatch") is not True
            or not isinstance(stored_checks, dict)
            or not stored_checks
            or any(value is not True for value in stored_checks.values())
            or any(checks.get(name) is not True for name in stored_checks)
        ):
            raise ContinuationError("Stored pre-dispatch failure result conflicts")
    else:
        stored = {
            **manifest,
            "status": "failed",
            "formal_sample": False,
            "returncode": 2,
            "identity_stable": True,
            "source_tree_stable": True,
            "termination_class": "process_infrastructure_error",
            "retryable": True,
            "retryable_before_provider_dispatch": True,
            "strict_continuation_claim_blocks_replay": False,
            "summary": summary,
            "continuation_validation": {
                "valid": False,
                "classification": "failed_before_provider_dispatch",
                "checks": checks,
                "predispatch_failure_attestation_sha256": (
                    current_attestation_sha256
                ),
            },
            "finished_at": _utc_now(),
            "resume_action": "trajectory_continuation_predispatch_retry",
        }
        _atomic_create(result_path, stored)

    stored_validation = stored.get("continuation_validation") or {}
    stored_attestation_sha256 = stored_validation.get(
        "predispatch_failure_attestation_sha256"
    )
    authorization_core = {
        "schema_version": 1,
        "status": "predispatch_retry_authorized",
        "attempt": manifest.get("attempt"),
        "claim_id": claim.get("claim_id"),
        "source_attempt": plan["source_attempt"],
        "boundary_decision_index": plan["boundary"]["decision_index"],
        "provider_calls": 0,
        "credential_file_read": False,
        "dispatch_count": 0,
        "checks": checks,
        "plan_sha256": hash_json(plan),
        "adapter_sha256": hash_file(Path(__file__).resolve()),
        "attempt_result_sha256": hash_file(result_path),
        "current_attestation_sha256": current_attestation_sha256,
        "stored_attestation_sha256": stored_attestation_sha256,
        "stored_attestation_matches_current": (
            stored_attestation_sha256 == current_attestation_sha256
        ),
    }
    if authorization_path.is_file():
        authorization = _read_json(authorization_path)
        if (
            not isinstance(authorization.get("recorded_at"), str)
            or any(
                authorization.get(key) != value
                for key, value in authorization_core.items()
            )
        ):
            raise ContinuationError("Stored pre-dispatch retry authorization conflicts")
    else:
        authorization = {**authorization_core, "recorded_at": _utc_now()}
        _atomic_create(authorization_path, authorization)

    return {
        **stored,
        "predispatch_retry_authorization_sha256": hash_file(authorization_path),
        "continuation_validation": {
            **stored_validation,
            "current_predispatch_failure_attestation_sha256": (
                current_attestation_sha256
            ),
            "predispatch_retry_authorization_sha256": hash_file(
                authorization_path
            ),
            "current_checks": checks,
        },
    }


def _load_verified_plan(verification_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verification_root = verification_root.resolve(strict=True)
    results_root = (ROOT / "results").resolve()
    if not verification_root.is_relative_to(results_root):
        raise ContinuationError("Verification root must remain beneath benchmark results")
    plan_path = verification_root / "replay_plan.json"
    verification_path = verification_root / "continuation_verification.json"
    if not plan_path.is_file() or not verification_path.is_file():
        raise ContinuationError("Verification root is incomplete")
    plan = _read_json(plan_path)
    verification = _read_json(verification_path)
    adapter_sha256 = hash_file(Path(__file__).resolve())
    seam = verification.get("seam") or {}
    boundary = plan.get("boundary") or {}
    if (
        plan.get("schema_version") != 2
        or verification.get("status") != "verified"
        or verification.get("mode") != "verify_only"
        or verification.get("provider_calls") != 0
        or verification.get("credential_file_read") is not False
        or verification.get("adapter_sha256") != adapter_sha256
        or plan.get("adapter_sha256") != adapter_sha256
        or verification.get("plan_sha256") != hash_file(plan_path)
        or seam.get("status") != "verified"
        or seam.get("decision_index") != boundary.get("decision_index")
        or seam.get("replayed_decisions") != plan.get("accepted_decision_count")
        or seam.get("model_messages_sha256")
        != boundary.get("model_messages_sha256")
        or seam.get("provider_request_sha256")
        != boundary.get("provider_request_sha256")
        or not _seam_first_dispatch_matches(seam, boundary)
        or seam.get("engine_prefix_sha256")
        != boundary.get("engine_prefix_sha256")
        or seam.get("state_sha256") != boundary.get("state_sha256")
        or seam.get("source_lock_sha256") != plan.get("source_lock_sha256")
        or seam.get("source_tree_identity_sha256")
        != plan.get("source_tree_identity_sha256")
        or seam.get("provider_calls") != 0
        or seam.get("credential_file_read") is not False
        or seam.get("credential_values_recorded") is not False
    ):
        raise ContinuationError("Verification root is not a current zero-call proof")
    rebuilt = build_replay_plan(
        Path(str(plan["source_attempt"])),
        Path(str(plan["execution_sts2_cli_root"])),
    )
    if hash_json(rebuilt) != hash_json(plan):
        raise ContinuationError("Verified replay plan no longer matches source artifacts")
    return plan, verification


def _continuation_claim_id(plan: dict[str, Any]) -> str:
    boundary = plan["boundary"]
    return hash_json(
        {
            "source_lock_sha256": plan["source_lock_sha256"],
            "source_hashes": plan["source_hashes"],
            "boundary_decision_index": boundary["decision_index"],
            "boundary_provider_request_sha256": (
                _first_live_dispatch_request_sha256(boundary)
            ),
        }
    )


def _continuation_execution_identity(
    plan: dict[str, Any],
    verification: dict[str, Any],
    *,
    adapter_sha256: str,
) -> str:
    validator_sha256 = hash_file(
        Path(sys.modules[_formal_continuation_execution_identity.__module__].__file__)
    )
    return _formal_continuation_execution_identity(
        plan,
        verification,
        adapter_sha256=adapter_sha256,
        validator_sha256=validator_sha256,
    )


def _validate_continuation_evidence(
    attempt_dir: Path,
    *,
    plan: dict[str, Any],
    wrapper_command: list[str],
    execution_identity_sha256: str,
) -> dict[str, Any]:
    validator_sha256 = hash_file(
        Path(sys.modules[_formal_validate_continuation_evidence.__module__].__file__)
    )
    return _formal_validate_continuation_evidence(
        attempt_dir,
        plan=plan,
        wrapper_command=wrapper_command,
        execution_identity_sha256=execution_identity_sha256,
        validator_sha256=validator_sha256,
    )


def _replace_command_option(command: list[str], option: str, value: str) -> list[str]:
    updated = list(command)
    try:
        index = updated.index(option)
    except ValueError as exc:
        raise ContinuationError(f"Source attempt command is missing {option}") from exc
    if index + 1 >= len(updated):
        raise ContinuationError(f"Source attempt command has no value for {option}")
    updated[index + 1] = value
    return updated


def _lazy_api_key(env_file: Path) -> Callable[[], str]:
    env_file = env_file.resolve(strict=True)

    def load() -> str:
        _load_env_file(env_file, override=True)
        return str(
            os.environ.get("STS2_BENCH_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )

    return load


def run_verify_only(
    source_attempt: Path,
    output_root: Path,
    execution_sts2_cli_root: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    results_root = (ROOT / "results").resolve()
    if not output_root.is_relative_to(results_root):
        raise ContinuationError("Verification output must remain beneath benchmark results")
    if output_root.exists():
        raise ContinuationError("Verification output root must not already exist")
    plan = build_replay_plan(source_attempt, execution_sts2_cli_root)
    output_root.mkdir(parents=True)
    _atomic_write(output_root / "replay_plan.json", plan)
    replay_artifacts = output_root / "replay_artifacts"
    live_root = Path(plan["source_live_root"])
    model = ReplayThenLiveModel(
        plan,
        replay_engine_trace=replay_artifacts / "engine_trace.jsonl",
        verify_only=True,
        source_lock_verifier=lambda: _read_and_verify_source_lock_for_execution(
            live_root,
            Path(plan["execution_sts2_cli_root"]),
        ),
    )
    config = plan["config"]
    human_cli = load_human_cli(Path(plan["execution_sts2_cli_root"]).resolve())
    run_direct_cli(
        human_cli=human_cli,
        model=model,
        out_dir=replay_artifacts,
        seed=str(config["seed"]),
        character=str(config["character"]),
        ascension=int(config["ascension"]),
        lang=str(config["lang"]),
        history_chars=int(config["history_chars"]),
        max_decisions=int(config["max_decisions"]),
        echo_terminal=False,
        source_lock_sha256=plan["source_lock_sha256"],
        source_tree_identity_sha256=plan["source_tree_identity_sha256"],
    )
    if (
        model.seam_proof is None
        or model.provider_calls != 0
        or model.replayed_decisions != plan["accepted_decision_count"]
    ):
        raise ContinuationError("Verify-only run did not stop at a zero-call seam")
    result = {
        "schema_version": 1,
        "status": "verified",
        "mode": "verify_only",
        "source_attempt": plan["source_attempt"],
        "adapter_sha256": hash_file(Path(__file__).resolve()),
        "plan_sha256": hash_file(output_root / "replay_plan.json"),
        "seam": model.seam_proof,
        "provider_calls": 0,
        "credential_file_read": False,
        "credential_values_recorded": False,
    }
    _atomic_write(output_root / "continuation_verification.json", result)
    return result


def _run_live_inner(
    verification_root: Path,
    env_file: Path,
    execution_sts2_cli_root: Path | None = None,
) -> dict[str, Any]:
    plan, verification = _load_verified_plan(verification_root)
    planned_execution_root = Path(str(plan["execution_sts2_cli_root"])).resolve()
    if (
        execution_sts2_cli_root is not None
        and execution_sts2_cli_root.resolve(strict=True) != planned_execution_root
    ):
        raise ContinuationError(
            "Requested sts2-cli root does not match the verified replay plan"
        )
    verification = {
        **verification,
        "verification_sha256": hash_file(
            verification_root.resolve() / "continuation_verification.json"
        ),
    }
    source_attempt = Path(str(plan["source_attempt"])).resolve(strict=True)
    source_manifest = _read_json(source_attempt / "attempt_manifest.json")
    live_root = Path(str(plan["source_live_root"])).resolve(strict=True)
    source_lock = _read_and_verify_source_lock_for_execution(
        live_root,
        planned_execution_root,
    )
    protocol_path = Path(str(source_lock.get("protocol_path") or "")).resolve(strict=True)
    protocol = _read_json(protocol_path)
    job_dir = source_attempt.parent.parent
    if job_dir.name != f"replicate_{int(source_manifest['replicate']):02d}":
        raise ContinuationError("Source attempt is not in its declared replicate job")
    existing_attempts = sorted(
        path
        for path in (job_dir / "attempts").glob("attempt_*")
        if path.is_dir()
    )
    source_attempt_index = int(source_manifest["attempt"])
    if (
        source_attempt.name != f"attempt_{source_attempt_index:03d}"
        or source_attempt not in [path.resolve() for path in existing_attempts]
    ):
        raise ContinuationError("Source attempt is not a canonical attempt")
    lease_path = job_dir / "active_attempt.json"
    if lease_path.exists():
        raise ContinuationError("Canonical job already has an active attempt")
    prior_job_status = (
        _read_json(job_dir / "job_status.json")
        if (job_dir / "job_status.json").is_file()
        else {}
    )
    prior_history = prior_job_status.get("attempt_history")
    if not isinstance(prior_history, list):
        prior_history = [
            _read_json(path)
            for path in sorted((job_dir / "attempts").glob("attempt_*/attempt_result.json"))
        ]

    adapter_sha256 = hash_file(Path(__file__).resolve())
    validator_path = Path(
        sys.modules[_formal_validate_continuation_evidence.__module__].__file__
    ).resolve()
    validator_sha256 = hash_file(validator_path)
    execution_identity_sha256 = _continuation_execution_identity(
        plan,
        verification,
        adapter_sha256=adapter_sha256,
    )
    claim_id = _continuation_claim_id(plan)
    # Canonical B0 job roots can already approach MAX_PATH on Windows.  The
    # complete claim id remains inside the claim; a 128-bit filename prefix is
    # sufficient to keep its stable on-disk location collision-safe and usable.
    claim_path = job_dir / "continuation_claims" / f"{claim_id[:32]}.json"
    later_attempts = [
        path
        for path in existing_attempts
        if int(path.name.removeprefix("attempt_")) > source_attempt_index
    ]
    predispatch_failures: list[dict[str, Any]] = []
    if later_attempts:
        if not claim_path.is_file():
            raise ContinuationError(
                "Later continuation attempts exist without their stable claim"
            )
        for path in later_attempts:
            failure = _predispatch_failure_result(
                path,
                plan=plan,
                claim_path=claim_path,
            )
            predispatch_failures.append(
                {
                    "attempt": failure.get("attempt"),
                    "attempt_dir": str(path.resolve()),
                    "finished_at": failure.get("finished_at"),
                    "attestation_sha256": failure.get(
                        "continuation_validation", {}
                    ).get("predispatch_failure_attestation_sha256"),
                    "current_attestation_sha256": failure.get(
                        "continuation_validation", {}
                    ).get("current_predispatch_failure_attestation_sha256"),
                    "retry_authorization_sha256": failure.get(
                        "predispatch_retry_authorization_sha256"
                    ),
                }
            )
            if not any(
                row.get("attempt") == failure.get("attempt")
                for row in prior_history
                if isinstance(row, dict)
            ):
                prior_history.append(failure)
        latest_failure = _read_json(later_attempts[-1] / "attempt_result.json")
        _atomic_write(
            job_dir / "job_status.json",
            {**latest_failure, "attempt_history": prior_history},
        )
    elif existing_attempts[-1].resolve() != source_attempt:
        raise ContinuationError("Source attempt is not the latest canonical attempt")

    attempt_index = (
        max(
            int(path.name.removeprefix("attempt_"))
            for path in existing_attempts
        )
        + 1
    )
    attempt_dir = job_dir / "attempts" / f"attempt_{attempt_index:03d}"
    artifacts = attempt_dir / "artifacts"
    if attempt_dir.exists():
        raise ContinuationError("Canonical continuation attempt already exists")
    claim = {
        "schema_version": 1,
        "claim_id": claim_id,
        "status": "reserved",
        "source_attempt": str(source_attempt),
        "source_lock_sha256": plan["source_lock_sha256"],
        "source_hashes": plan["source_hashes"],
        "boundary_decision_index": plan["boundary"]["decision_index"],
        "boundary_provider_request_sha256": (
            _first_live_dispatch_request_sha256(plan["boundary"])
        ),
        "verification_root": str(verification_root.resolve()),
        "verification_sha256": verification["verification_sha256"],
        "plan_sha256": verification["plan_sha256"],
        "adapter_sha256": adapter_sha256,
        "validator_sha256": validator_sha256,
        "continuation_execution_identity_sha256": execution_identity_sha256,
        "provider_dispatch_started": False,
        "created_at": _utc_now(),
    }
    started_at = _utc_now()
    try:
        _atomic_create(
            lease_path,
            {
                "schema_version": 1,
                "experiment_id": source_manifest["experiment_id"],
                "seed": source_manifest["seed"],
                "replicate": source_manifest["replicate"],
                "attempt": attempt_index,
                "owner_pid": os.getpid(),
                "child_pid": None,
                "state": "trajectory_continuation",
                "started_at": started_at,
                "claim_id": claim_id,
            },
        )
    except Exception:
        try:
            lease = _read_json(lease_path)
        except (FileNotFoundError, json.JSONDecodeError, ContinuationError):
            lease = {}
        if (
            lease.get("owner_pid") == os.getpid()
            and lease.get("claim_id") == claim_id
        ):
            lease_path.unlink(missing_ok=True)
        raise
    try:
        if predispatch_failures:
            stored_claim = _read_json(claim_path)
            invariant_keys = (
                "claim_id",
                "source_attempt",
                "source_lock_sha256",
                "source_hashes",
                "boundary_decision_index",
                "boundary_provider_request_sha256",
            )
            if any(
                stored_claim.get(key) != claim.get(key) for key in invariant_keys
            ):
                raise ContinuationError("Stored continuation claim invariants conflict")
            if (
                stored_claim.get("provider_dispatch_started") is not False
                or stored_claim.get("last_dispatch")
            ):
                raise ContinuationError("Stored continuation claim may have dispatched")
            claim = {
                **claim,
                "created_at": stored_claim.get("created_at") or claim["created_at"],
                "status": "reserved_after_predispatch_failure",
                "predispatch_failures": predispatch_failures,
                "updated_at": _utc_now(),
            }
            _atomic_write(claim_path, claim)
        else:
            _atomic_create(claim_path, claim)
    except Exception:
        try:
            lease = _read_json(lease_path)
        except (FileNotFoundError, json.JSONDecodeError, ContinuationError):
            lease = {}
        if (
            lease.get("owner_pid") == os.getpid()
            and lease.get("claim_id") == claim_id
        ):
            lease_path.unlink(missing_ok=True)
        raise
    attempt_dir.mkdir(parents=True, exist_ok=False)
    continuation_dir = attempt_dir / "continuation"
    continuation_dir.mkdir()
    _atomic_write(continuation_dir / "claim_snapshot.json", claim)
    source_protocol_command = _replace_command_option(
        list(source_manifest["command"]),
        "--out-dir",
        str(artifacts.resolve()),
    )
    source_protocol_command = _replace_command_option(
        source_protocol_command,
        "--sts2-cli-root",
        str(Path(plan["execution_sts2_cli_root"]).resolve()),
    )
    wrapper_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--live",
        "--verification-root",
        str(verification_root.resolve()),
        "--env-file",
        str(env_file.resolve()),
        "--sts2-cli-root",
        str(Path(plan["execution_sts2_cli_root"]).resolve()),
    ]
    attempt_manifest = {
        **source_manifest,
        "attempt": attempt_index,
        # ``command`` remains the logical Direct-CLI invocation consumed by the
        # normal B0 artifact validator.  ``wrapper_command`` records the actual
        # OS process that replayed the immutable prefix before invoking it.
        "command": source_protocol_command,
        "source_protocol_command": source_protocol_command,
        "wrapper_command": wrapper_command,
        "command_role": "validated_direct_cli_invocation",
        "wrapper_command_role": "actual_os_process",
        "started_at": started_at,
        "execution_mode": "trajectory_continuation",
        "continuation": {
            "claim_id": claim_id,
            "claim_path": str(claim_path),
            "source_attempt": str(source_attempt),
            "verification_root": str(verification_root.resolve()),
            "verification_sha256": claim["verification_sha256"],
            "plan_sha256": claim["plan_sha256"],
            "adapter_sha256": claim["adapter_sha256"],
            "validator_path": str(validator_path),
            "validator_sha256": validator_sha256,
            "execution_identity_sha256": execution_identity_sha256,
            "replayed_decision_count": plan["accepted_decision_count"],
            "prior_failed_provider_attempts": plan["boundary"][
                "failed_provider_attempt_count"
            ],
        },
    }
    _atomic_write(attempt_dir / "attempt_manifest.json", attempt_manifest)
    (attempt_dir / "stdout.log").write_text("", encoding="utf-8")
    (attempt_dir / "stderr.log").write_text("", encoding="utf-8")

    def persist_seam(value: dict[str, Any]) -> None:
        nonlocal claim
        _atomic_create(continuation_dir / "seam_proof.json", value)
        claim = {
            **claim,
            "status": "seam_verified",
            "seam_proof_sha256": hash_file(continuation_dir / "seam_proof.json"),
            "credential_file_read": value.get("credential_file_read") is True,
            "updated_at": _utc_now(),
        }
        _atomic_write(claim_path, claim)

    def persist_dispatch(value: dict[str, Any]) -> None:
        nonlocal claim
        decision_index = int(value["decision_index"])
        dispatch_path = (
            continuation_dir
            / "dispatches"
            / f"decision_{decision_index:06d}.json"
        )
        _atomic_create(dispatch_path, value)
        claim = {
            **claim,
            "status": "provider_dispatch_started",
            "provider_dispatch_started": True,
            "last_dispatch": value,
            "last_dispatch_sha256": hash_file(dispatch_path),
            "updated_at": _utc_now(),
        }
        _atomic_write(claim_path, claim)

    model = ReplayThenLiveModel(
        plan,
        replay_engine_trace=artifacts / "engine_trace.jsonl",
        verify_only=False,
        source_lock_verifier=lambda: _read_and_verify_source_lock_for_execution(
            live_root,
            planned_execution_root,
        ),
        seam_persistor=persist_seam,
        credential_loader=_lazy_api_key(env_file),
        dispatch_persistor=persist_dispatch,
    )
    config = plan["config"]
    human_cli = load_human_cli(Path(plan["execution_sts2_cli_root"]).resolve())
    try:
        summary = run_direct_cli(
            human_cli=human_cli,
            model=model,
            out_dir=artifacts,
            seed=str(config["seed"]),
            character=str(config["character"]),
            ascension=int(config["ascension"]),
            lang=str(config["lang"]),
            history_chars=int(config["history_chars"]),
            max_decisions=int(config["max_decisions"]),
            echo_terminal=False,
            source_lock_sha256=plan["source_lock_sha256"],
            source_tree_identity_sha256=plan["source_tree_identity_sha256"],
        )
        post_source_lock = _read_and_verify_source_lock_for_execution(
            live_root,
            planned_execution_root,
        )
        if (
            post_source_lock.get("identity_sha256") != plan["source_lock_sha256"]
            or post_source_lock.get("source_tree_identity_sha256")
            != plan["source_tree_identity_sha256"]
            or post_source_lock.get("protocol_sha256") != plan["protocol_sha256"]
        ):
            raise ContinuationError("Source lock drifted during live continuation")
        if (
            model.seam_proof is None
            or model.replayed_decisions != plan["accepted_decision_count"]
            or model.provider_calls < 1
            or model.credential_file_read is not True
        ):
            raise ContinuationError(
                "Live continuation did not complete its strict replay seam"
            )
        dispatch_files = sorted(
            (continuation_dir / "dispatches").glob("decision_*.json")
        )
        if (
            len(dispatch_files) != model.provider_calls
            or not dispatch_files
            or int(
                _read_json(dispatch_files[0]).get("decision_index", -1)
            )
            != int(plan["boundary"]["decision_index"])
        ):
            raise ContinuationError("Live continuation dispatch evidence is incomplete")
        failure_chain_valid, failure_chain = _formal_predispatch_failure_chain(
            attempt_dir=attempt_dir,
            claim=claim,
            plan=plan,
            adapter_sha256=adapter_sha256,
        )
        if not failure_chain_valid:
            raise ContinuationError("Pre-dispatch failure chain is not formally anchored")
        attestation = {
            "schema_version": 1,
            "status": "execution_complete",
            "claim_id": claim_id,
            "source_attempt": str(source_attempt),
            "source_lock_sha256": plan["source_lock_sha256"],
            "source_tree_identity_sha256": plan["source_tree_identity_sha256"],
            "protocol_sha256": plan["protocol_sha256"],
            "verification_sha256": claim["verification_sha256"],
            "plan_sha256": claim["plan_sha256"],
            "adapter_sha256": claim["adapter_sha256"],
            "validator_sha256": validator_sha256,
            "continuation_execution_identity_sha256": execution_identity_sha256,
            "claim_snapshot_sha256": hash_file(
                continuation_dir / "claim_snapshot.json"
            ),
            "predispatch_failure_chain_sha256": hash_json(failure_chain),
            "replayed_decision_count": model.replayed_decisions,
            "provider_calls": model.provider_calls,
            "credential_file_read": model.credential_file_read,
            "boundary_decision_index": plan["boundary"]["decision_index"],
            "seam_proof_sha256": hash_file(
                continuation_dir / "seam_proof.json"
            ),
            "dispatch_sha256": [hash_file(path) for path in dispatch_files],
            "post_run_source_lock": {
                "identity_sha256": post_source_lock.get("identity_sha256"),
                "source_tree_identity_sha256": post_source_lock.get(
                    "source_tree_identity_sha256"
                ),
                "protocol_sha256": post_source_lock.get("protocol_sha256"),
            },
            "completed_at": _utc_now(),
        }
        _atomic_write(
            continuation_dir / "continuation_attestation.json",
            attestation,
        )
        claim = {
            **claim,
            "status": "execution_complete_pending_validation",
            "provider_calls": model.provider_calls,
            "credential_file_read": model.credential_file_read,
            "attestation_sha256": hash_file(
                continuation_dir / "continuation_attestation.json"
            ),
            "updated_at": _utc_now(),
        }
        _atomic_write(claim_path, claim)
        returncode = 0 if summary.get("formal_behavior_sample") is True else 1
        validation = _validate_attempt_artifacts(
            attempt_dir,
            protocol=protocol,
            protocol_path=protocol_path,
            execution_benchmark_root=Path(
                str(source_lock["benchmark_root"])
            ).resolve(),
            execution_sts2_cli_root=Path(
                str(plan["execution_sts2_cli_root"])
            ).resolve(),
            source_lock=str(plan["source_lock_sha256"]),
            source_tree_identity=str(plan["source_tree_identity_sha256"]),
            expected_seed=str(source_manifest["seed"]),
            expected_replicate=int(source_manifest["replicate"]),
            returncode=returncode,
        )
        continuation_validation = _validate_continuation_evidence(
            attempt_dir,
            plan=plan,
            wrapper_command=wrapper_command,
            execution_identity_sha256=execution_identity_sha256,
        )
        succeeded = (
            validation.get("valid") is True
            and continuation_validation.get("valid") is True
        )
        claim = {
            **claim,
            "status": "completed" if succeeded else "formal_validation_rejected",
            "provider_calls": model.provider_calls,
            "credential_file_read": model.credential_file_read,
            "attestation_sha256": hash_file(
                continuation_dir / "continuation_attestation.json"
            ),
            "validation_sha256": hash_json(validation),
            "continuation_validation_sha256": hash_json(continuation_validation),
            "finished_at": _utc_now(),
        }
        _atomic_write(claim_path, claim)
        result = {
            **attempt_manifest,
            "status": "succeeded" if succeeded else "failed",
            "formal_sample": succeeded,
            "returncode": returncode,
            "identity_stable": True,
            "source_tree_stable": True,
            "rules_exposure_valid": bool(
                validation.get("checks", {}).get(
                    "config_game_rules_match_protocol"
                )
                and validation.get("checks", {}).get(
                    "terminal_decision_rule_exposure_complete"
                )
            ),
            "termination_class": validation.get("termination_class"),
            "retryable": False,
            "strict_continuation_claim_blocks_replay": not succeeded,
            "validation": validation,
            "continuation_validation": continuation_validation,
            "artifacts": str(artifacts),
            "summary": validation.get("summary") or summary,
            "accepted_summary_sha256": (
                hash_file(artifacts / "summary.json")
                if (artifacts / "summary.json").is_file()
                else None
            ),
            "accepted_analysis_artifact_manifest": (
                (
                    (validation.get("summary") or {})
                    .get("analysis", {})
                    .get("analysis_artifact_manifest")
                )
                if isinstance((validation.get("summary") or {}).get("analysis"), dict)
                else None
            ),
            "finished_at": _utc_now(),
            "resume_action": "trajectory_continuation",
            "continuation_proof": {
                **attempt_manifest["continuation"],
                "provider_calls": model.provider_calls,
                "credential_file_read": model.credential_file_read,
                "seam_proof_sha256": (
                    hash_file(continuation_dir / "seam_proof.json")
                    if (continuation_dir / "seam_proof.json").is_file()
                    else None
                ),
                "attestation_sha256": claim["attestation_sha256"],
                "claim_status": claim["status"],
            },
        }
        _atomic_write(attempt_dir / "attempt_result.json", result)
        _atomic_write(
            job_dir / "job_status.json",
            {**result, "attempt_history": [*prior_history, result]},
        )
        return result
    finally:
        try:
            lease = _read_json(lease_path)
        except (FileNotFoundError, json.JSONDecodeError, ContinuationError):
            lease = {}
        if lease.get("owner_pid") == os.getpid() and lease.get("claim_id") == claim_id:
            lease_path.unlink(missing_ok=True)


def run_live(
    verification_root: Path,
    env_file: Path,
    execution_sts2_cli_root: Path | None = None,
) -> dict[str, Any]:
    """Run live with a fail-safe lease release covering all post-lease setup."""
    try:
        return _run_live_inner(
            verification_root,
            env_file,
            execution_sts2_cli_root,
        )
    finally:
        try:
            plan = _read_json(verification_root.resolve() / "replay_plan.json")
            source_attempt = Path(str(plan["source_attempt"])).resolve()
            lease_path = source_attempt.parent.parent / "active_attempt.json"
            claim_id = _continuation_claim_id(plan)
            lease = _read_json(lease_path)
        except (FileNotFoundError, json.JSONDecodeError, ContinuationError, KeyError):
            lease = {}
        if (
            lease.get("owner_pid") == os.getpid()
            and lease.get("claim_id") == claim_id
        ):
            lease_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--source-attempt", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--verification-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--sts2-cli-root", type=Path)
    args = parser.parse_args(argv)
    if args.verify_only and (args.source_attempt is None or args.out_root is None):
        parser.error("--verify-only requires --source-attempt and --out-root")
    if args.verify_only and (args.verification_root is not None or args.env_file is not None):
        parser.error("--verify-only refuses --verification-root and --env-file")
    if args.live and (args.verification_root is None or args.env_file is None):
        parser.error("--live requires --verification-root and --env-file")
    if args.live and (args.source_attempt is not None or args.out_root is not None):
        parser.error("--live derives the canonical attempt from --verification-root")
    try:
        result = (
            run_verify_only(
                args.source_attempt,
                args.out_root,
                args.sts2_cli_root,
            )
            if args.verify_only
            else run_live(
                args.verification_root,
                args.env_file,
                args.sts2_cli_root,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"verified", "succeeded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
