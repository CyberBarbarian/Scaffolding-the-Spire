"""Formal evidence validation for source-locked Direct-CLI continuations.

The original gameplay source lock remains immutable.  A continuation adds a
second, explicit execution identity that binds the old source, exact replay
boundary, zero-call verification, continuation adapter, and this validator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .direct_cli_baseline import OUTPUT_RETRY_MESSAGE
from .direct_cli_experiment import _audit_terminal_decision_requests
from .experiment_identity import hash_file, hash_json


class DirectCliContinuationEvidenceError(RuntimeError):
    pass


def _first_live_dispatch_request_sha256(boundary: dict[str, Any]) -> str:
    return str(
        boundary.get("first_live_dispatch_provider_request_sha256")
        or boundary["provider_request_sha256"]
    )


def _boundary_request_contract_valid(
    *,
    plan: dict[str, Any],
    boundary_messages: list[dict[str, str]],
    request_payload: dict[str, Any],
) -> bool:
    boundary = plan["boundary"]
    if (
        hash_json(boundary_messages) != boundary["model_messages_sha256"]
        or hash_json(request_payload) != boundary["provider_request_sha256"]
    ):
        return False
    prelude = boundary.get("protocol_retry_prelude")
    if not isinstance(prelude, dict):
        return (
            boundary.get(
                "first_live_dispatch_model_messages_sha256",
                boundary["model_messages_sha256"],
            )
            == boundary["model_messages_sha256"]
            and _first_live_dispatch_request_sha256(boundary)
            == boundary["provider_request_sha256"]
        )
    source_attempts = prelude.get("source_provider_attempts")
    if (
        prelude.get("kind")
        != "audited_empty_content_then_retry_transport_failure_v1"
        or prelude.get("retry_message_sha256")
        != hashlib.sha256(OUTPUT_RETRY_MESSAGE.encode("utf-8")).hexdigest()
        or not isinstance(source_attempts, list)
        or len(source_attempts) != 2
        or hash_json(source_attempts)
        != prelude.get("source_provider_attempts_sha256")
        or [item.get("provider_attempt_index") for item in source_attempts]
        != [0, 1]
        or source_attempts[0].get("failure_class") != "model_protocol_error"
        or source_attempts[0].get("error")
        != "ValueError: LLM response content is empty"
        or source_attempts[1].get("failure_class") != "provider_transport_error"
    ):
        return False
    retry_messages = [
        *boundary_messages,
        {"role": "user", "content": OUTPUT_RETRY_MESSAGE},
    ]
    retry_payload = dict(request_payload)
    retry_payload["messages"] = retry_messages
    return (
        hash_json(retry_messages)
        == boundary.get("first_live_dispatch_model_messages_sha256")
        and hash_json(retry_payload)
        == boundary.get("first_live_dispatch_provider_request_sha256")
        and source_attempts[0].get("model_messages_sha256")
        == boundary["model_messages_sha256"]
        and source_attempts[0].get("provider_request_sha256")
        == boundary["provider_request_sha256"]
        and source_attempts[1].get("model_messages_sha256")
        == boundary.get("first_live_dispatch_model_messages_sha256")
        and source_attempts[1].get("provider_request_sha256")
        == boundary.get("first_live_dispatch_provider_request_sha256")
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DirectCliContinuationEvidenceError(f"Expected JSON object: {path}")
    return value


def continuation_execution_identity(
    plan: dict[str, Any],
    verification: dict[str, Any],
    *,
    adapter_sha256: str,
    validator_sha256: str,
) -> str:
    """Bind an immutable gameplay source to the continuation-only executor."""
    return hash_json(
        {
            "schema_version": 1,
            "source_lock_sha256": plan["source_lock_sha256"],
            "source_tree_identity_sha256": plan["source_tree_identity_sha256"],
            "protocol_sha256": plan["protocol_sha256"],
            "source_hashes": plan["source_hashes"],
            "boundary": plan["boundary"],
            "verification_sha256": verification["verification_sha256"],
            "plan_sha256": verification["plan_sha256"],
            "adapter_sha256": adapter_sha256,
            "validator_sha256": validator_sha256,
        }
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DirectCliContinuationEvidenceError(
                    f"Expected JSON object in {path}"
                )
            rows.append(value)
    return rows


def _canonical_replayed_decisions(
    terminal_decisions: list[dict[str, Any]],
    provider_attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts_by_decision: dict[int, list[dict[str, Any]]] = {}
    for attempt in provider_attempts:
        attempts_by_decision.setdefault(int(attempt["decision_index"]), []).append(
            attempt
        )
    return [
        {
            "decision_index": int(row["decision_index"]),
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
            "provider_attempts": [
                {
                    key: value
                    for key, value in attempt.items()
                    if key not in {"type", "decision_index", "timestamp"}
                }
                for attempt in attempts_by_decision.get(int(row["decision_index"]), [])
            ],
        }
        for row in terminal_decisions
    ]


def _canonical_engine_prefix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"step": row.get("step"), "type": row.get("type"), "data": row.get("data")}
        for row in rows
    ]


def _authoritative_decision_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexes = [
        index
        for index, row in enumerate(rows)
        if row.get("type") == "state"
        and isinstance(row.get("data"), dict)
        and row["data"].get("type") == "decision"
    ]
    if not indexes:
        raise DirectCliContinuationEvidenceError("Missing authoritative decision state")
    index = indexes[-1]
    trailing = rows[index + 1 :]
    if len(trailing) % 2:
        raise DirectCliContinuationEvidenceError("Incomplete informational tail")
    for offset in range(0, len(trailing), 2):
        action, response = trailing[offset : offset + 2]
        if (
            action.get("type") != "action"
            or not isinstance(action.get("data"), dict)
            or action["data"].get("cmd") != "get_map"
            or response.get("type") != "state"
            or not isinstance(response.get("data"), dict)
            or response["data"].get("type") != "map"
        ):
            raise DirectCliContinuationEvidenceError(
                "State-changing engine tail after boundary"
            )
    return dict(rows[index]["data"])


def _attempt_index(path: Path) -> int:
    if not path.name.startswith("attempt_"):
        raise DirectCliContinuationEvidenceError(f"Non-canonical attempt: {path}")
    index = int(path.name.removeprefix("attempt_"))
    if path.name != f"attempt_{index:03d}":
        raise DirectCliContinuationEvidenceError(f"Non-canonical attempt: {path}")
    return index


def _recorded_predispatch_checks_are_safe(checks: Any) -> bool:
    if not isinstance(checks, dict) or not checks:
        return False
    superseded = {
        "engine_raw_prefix_matches_seam",
        "engine_tail_is_cleanup_only",
    }
    return all(
        value is True for name, value in checks.items() if name not in superseded
    )


def _verify_predispatch_failure(
    *,
    failure_dir: Path,
    row: dict[str, Any],
    claim_id: str | None,
    plan: dict[str, Any],
    adapter_sha256: str | None,
) -> tuple[bool, dict[str, Any]]:
    try:
        attempt_index = _attempt_index(failure_dir)
        manifest_path = failure_dir / "attempt_manifest.json"
        result_path = failure_dir / "attempt_result.json"
        config_path = failure_dir / "artifacts" / "config.json"
        events_path = failure_dir / "artifacts" / "events.jsonl"
        terminal_path = failure_dir / "artifacts" / "terminal.txt"
        engine_path = failure_dir / "artifacts" / "engine_trace.jsonl"
        summary_path = failure_dir / "artifacts" / "summary.json"
        seam_path = failure_dir / "continuation" / "seam_proof.json"
        attestation_path = (
            failure_dir / "continuation" / "predispatch_failure_attestation.json"
        )
        authorization_path = (
            failure_dir / "continuation" / "predispatch_retry_authorization.json"
        )
        required = (
            manifest_path,
            result_path,
            config_path,
            events_path,
            terminal_path,
            engine_path,
            summary_path,
            seam_path,
            attestation_path,
            authorization_path,
        )
        if not all(path.is_file() for path in required):
            return False, {}
        if (
            (failure_dir / "continuation" / "continuation_attestation.json").exists()
            or any(
                (failure_dir / "continuation" / "dispatches").glob(
                    "decision_*.json"
                )
            )
        ):
            return False, {}
        manifest = _read_json(manifest_path)
        result = _read_json(result_path)
        config = _read_json(config_path)
        summary = _read_json(summary_path)
        seam = _read_json(seam_path)
        attestation = _read_json(attestation_path)
        authorization = _read_json(authorization_path)
        events = _read_jsonl(events_path)
        engine_rows = _read_jsonl(engine_path)
        terminal = terminal_path.read_text(encoding="utf-8")
        accepted_count = int(plan["accepted_decision_count"])
        terminal_decisions = [
            item for item in events if item.get("type") == "terminal_decision"
        ]
        provider_attempts = [
            item for item in events if item.get("type") == "provider_request_attempt"
        ]
        protocol_errors = [
            item for item in events if item.get("type") == "model_protocol_error"
        ]
        provider_indices_valid = all(
            isinstance(item.get("decision_index"), int)
            and not isinstance(item.get("decision_index"), bool)
            and 0 <= item["decision_index"] < accepted_count
            for item in provider_attempts
        )
        prefix_audit = _audit_terminal_decision_requests(
            terminal_decisions,
            provider_attempts=provider_attempts,
            protocol_errors=protocol_errors,
            terminal=terminal,
            history_limit=int(config["history_chars"]),
            request_model=str(config["model"]),
            request_temperature=float(config["temperature"]),
            request_max_tokens=config.get("max_tokens"),
            request_retries=int(config["request_retries"]),
            require_prompt_contract=True,
        )
        marker = "\n  [log] Game log saved to"
        boundary_end = terminal.rfind(marker)
        visible = (
            terminal[max(0, boundary_end - int(config["history_chars"])) : boundary_end]
            if boundary_end >= 0
            else ""
        )
        boundary_messages = [
            {"role": "system", "content": str(config.get("system_prompt") or "")},
            {
                "role": "user",
                "content": (
                    "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                    + visible
                    + "\n\nReturn only the next terminal input line."
                ),
            },
        ]
        request_payload: dict[str, Any] = {
            "model": str(config["model"]),
            "messages": boundary_messages,
            "temperature": float(config["temperature"]),
        }
        if config.get("max_tokens") is not None:
            request_payload["max_tokens"] = config["max_tokens"]
        quit_indexes = [
            index
            for index, item in enumerate(engine_rows)
            if item.get("type") == "action"
            and isinstance(item.get("data"), dict)
            and item["data"].get("cmd") == "quit"
        ]
        engine_shape_valid = (
            len(quit_indexes) == 1
            and quit_indexes[0] == len(engine_rows) - 2
            and engine_rows[-1].get("type") == "state"
            and isinstance(engine_rows[-1].get("data"), dict)
            and engine_rows[-1]["data"].get("type") == "quit_result"
            and engine_rows[-1]["data"].get("success") is True
        )
        engine_prefix = engine_rows[: quit_indexes[0]] if engine_shape_valid else []
        engine_state = (
            _authoritative_decision_state(engine_prefix) if engine_prefix else {}
        )
        result_validation = result.get("continuation_validation") or {}
        result_attestation_sha256 = result_validation.get(
            "predispatch_failure_attestation_sha256"
        )
        authorization_checks = authorization.get("checks")
        attestation_checks = attestation.get("checks")
        recomputed_checks = {
            "canonical_attempt_identity": (
                manifest.get("attempt") == attempt_index
                and row.get("attempt") == attempt_index
                and row.get("attempt_dir") == str(failure_dir)
            ),
            "manifest_and_result_classification": (
                manifest.get("execution_mode") == "trajectory_continuation"
                and (manifest.get("continuation") or {}).get("source_attempt")
                == plan["source_attempt"]
                and result.get("status") == "failed"
                and result.get("formal_sample") is False
                and result.get("retryable_before_provider_dispatch") is True
                and result.get("termination_class") == "process_infrastructure_error"
                and result.get("summary") == summary
            ),
            "accepted_prefix_exact": (
                [item.get("decision_index") for item in terminal_decisions]
                == list(range(accepted_count))
                and not protocol_errors
                and provider_indices_valid
                and len(provider_attempts)
                == int(plan.get("accepted_provider_attempt_count") or 0)
                and prefix_audit.get("valid") is True
                and _canonical_replayed_decisions(
                    terminal_decisions, provider_attempts
                )
                == plan["decisions"]
            ),
            "boundary_reconstructed": (
                boundary_end >= 0
                and _boundary_request_contract_valid(
                    plan=plan,
                    boundary_messages=boundary_messages,
                    request_payload=request_payload,
                )
            ),
            "engine_boundary_reconstructed": (
                engine_shape_valid
                and len(engine_prefix)
                == plan["boundary"]["engine_prefix_row_count"]
                and hash_json(_canonical_engine_prefix(engine_prefix))
                == plan["boundary"]["engine_prefix_sha256"]
                and hash_json(engine_state) == plan["boundary"]["state_sha256"]
            ),
            "zero_call_seam": (
                seam.get("status") == "verified"
                and seam.get("decision_index")
                == plan["boundary"]["decision_index"]
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
                and seam.get("state_sha256")
                == plan["boundary"]["state_sha256"]
            ),
            "summary_is_predispatch_infrastructure_failure": (
                summary.get("formal_behavior_sample") is False
                and summary.get("termination_class") == "process_infrastructure_error"
                and isinstance(summary.get("error"), str)
                and bool(summary["error"])
            ),
        }
        valid = all(recomputed_checks.values()) and not (
            result.get("retryable_before_provider_dispatch") is not True
            or attestation.get("status") != "failed_before_provider_dispatch"
            or attestation.get("attempt") != attempt_index
            or attestation.get("claim_id") != claim_id
            or attestation.get("source_attempt") != plan["source_attempt"]
            or attestation.get("boundary_decision_index")
            != plan["boundary"]["decision_index"]
            or attestation.get("replayed_decision_count") != accepted_count
            or attestation.get("provider_calls") != 0
            or attestation.get("credential_file_read") is not False
            or attestation.get("dispatch_count") != 0
            or attestation.get("summary_sha256") != hash_file(summary_path)
            or attestation.get("seam_proof_sha256") != hash_file(seam_path)
            or not _recorded_predispatch_checks_are_safe(attestation_checks)
            or authorization.get("status") != "predispatch_retry_authorized"
            or authorization.get("attempt") != attempt_index
            or authorization.get("claim_id") != claim_id
            or authorization.get("source_attempt") != plan["source_attempt"]
            or authorization.get("boundary_decision_index")
            != plan["boundary"]["decision_index"]
            or authorization.get("provider_calls") != 0
            or authorization.get("credential_file_read") is not False
            or authorization.get("dispatch_count") != 0
            or authorization.get("plan_sha256") != hash_json(plan)
            or authorization.get("adapter_sha256") != adapter_sha256
            or authorization.get("attempt_result_sha256") != hash_file(result_path)
            or authorization.get("current_attestation_sha256")
            != hash_file(attestation_path)
            or authorization.get("stored_attestation_sha256")
            != result_attestation_sha256
            or authorization.get("stored_attestation_matches_current")
            != (result_attestation_sha256 == hash_file(attestation_path))
            or not _recorded_predispatch_checks_are_safe(authorization_checks)
            or row.get("finished_at") != result.get("finished_at")
            or row.get("attestation_sha256") != result_attestation_sha256
            or row.get("current_attestation_sha256")
            != hash_file(attestation_path)
            or row.get("retry_authorization_sha256")
            != hash_file(authorization_path)
        )
        evidence = {
            "attempt": attempt_index,
            "attempt_dir": str(failure_dir),
            "attempt_manifest_sha256": hash_file(manifest_path),
            "attempt_result_sha256": hash_file(result_path),
            "config_sha256": hash_file(config_path),
            "events_sha256": hash_file(events_path),
            "terminal_sha256": hash_file(terminal_path),
            "engine_trace_sha256": hash_file(engine_path),
            "summary_sha256": hash_file(summary_path),
            "seam_proof_sha256": hash_file(seam_path),
            "stored_attestation_sha256": result_attestation_sha256,
            "current_attestation_sha256": hash_file(attestation_path),
            "retry_authorization_sha256": hash_file(authorization_path),
            "recomputed_checks": recomputed_checks,
        }
        return valid, evidence
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, {}


def _predispatch_failure_chain(
    *,
    attempt_dir: Path,
    claim: dict[str, Any],
    plan: dict[str, Any],
    adapter_sha256: str | None,
) -> tuple[bool, list[dict[str, Any]]]:
    try:
        current_dir = attempt_dir.resolve(strict=True)
        source_dir = Path(str(plan["source_attempt"])).resolve(strict=True)
        attempts_root = current_dir.parent
        if source_dir.parent != attempts_root:
            return False, []
        source_index = _attempt_index(source_dir)
        current_index = _attempt_index(current_dir)
        expected_dirs = [
            attempts_root / f"attempt_{index:03d}"
            for index in range(source_index + 1, current_index)
        ]
        if not all(path.is_dir() for path in expected_dirs):
            return False, []
        rows = claim.get("predispatch_failures")
        if not expected_dirs:
            return rows in (None, []), []
        if not isinstance(rows, list) or len(rows) != len(expected_dirs):
            return False, []
        chain: list[dict[str, Any]] = []
        for row, failure_dir in zip(rows, expected_dirs, strict=True):
            if not isinstance(row, dict):
                return False, []
            valid, evidence = _verify_predispatch_failure(
                failure_dir=failure_dir.resolve(),
                row=row,
                claim_id=claim.get("claim_id"),
                plan=plan,
                adapter_sha256=adapter_sha256,
            )
            if not valid:
                return False, []
            chain.append(evidence)
        return True, chain
    except (OSError, TypeError, ValueError):
        return False, []


def validate_continuation_evidence(
    attempt_dir: Path,
    *,
    plan: dict[str, Any],
    wrapper_command: list[str],
    execution_identity_sha256: str,
    validator_sha256: str,
) -> dict[str, Any]:
    """Validate continuation provenance in addition to the normal B0 checks."""
    manifest = _read_json(attempt_dir / "attempt_manifest.json")
    continuation = manifest.get("continuation")
    if not isinstance(continuation, dict):
        continuation = {}
    continuation_dir = attempt_dir / "continuation"
    seam_path = continuation_dir / "seam_proof.json"
    attestation_path = continuation_dir / "continuation_attestation.json"
    snapshot_path = continuation_dir / "claim_snapshot.json"
    claim_path_value = continuation.get("claim_path")
    claim_path = (
        Path(str(claim_path_value)).resolve()
        if isinstance(claim_path_value, str) and claim_path_value
        else Path()
    )
    seam = _read_json(seam_path) if seam_path.is_file() else {}
    attestation = _read_json(attestation_path) if attestation_path.is_file() else {}
    snapshot = _read_json(snapshot_path) if snapshot_path.is_file() else {}
    claim = _read_json(claim_path) if claim_path.is_file() else {}
    seam_sha256 = hash_file(seam_path) if seam_path.is_file() else None
    dispatch_paths = sorted((continuation_dir / "dispatches").glob("decision_*.json"))
    dispatches = [_read_json(path) for path in dispatch_paths]
    boundary_index = int(plan["boundary"]["decision_index"])
    expected_dispatch_indices = list(
        range(boundary_index, boundary_index + len(dispatches))
    )
    source_attempt = Path(str(plan["source_attempt"]))
    source_artifacts = Path(str(plan["source_artifacts"]))
    source_paths = {
        "config.json": source_artifacts / "config.json",
        "events.jsonl": source_artifacts / "events.jsonl",
        "terminal.txt": source_artifacts / "terminal.txt",
        "engine_trace.jsonl": source_artifacts / "engine_trace.jsonl",
        "attempt_manifest.json": source_attempt / "attempt_manifest.json",
        "attempt_result.json": source_attempt / "attempt_result.json",
    }
    source_hashes_match = (
        set(source_paths) == set(plan["source_hashes"])
        and all(
            path.is_file() and hash_file(path) == plan["source_hashes"][name]
            for name, path in source_paths.items()
        )
    )
    immutable_claim_matches = all(
        claim.get(key) == snapshot.get(key)
        for key in (
            "claim_id",
            "source_attempt",
            "source_lock_sha256",
            "source_hashes",
            "boundary_decision_index",
            "boundary_provider_request_sha256",
            "verification_root",
            "verification_sha256",
            "plan_sha256",
            "adapter_sha256",
            "validator_sha256",
            "continuation_execution_identity_sha256",
        )
    )
    predispatch_failures_are_safe, predispatch_failure_chain = (
        _predispatch_failure_chain(
        attempt_dir=attempt_dir,
        claim=claim,
        plan=plan,
        adapter_sha256=continuation.get("adapter_sha256"),
        )
    )
    claim_snapshot_sha256 = hash_file(snapshot_path) if snapshot_path.is_file() else None
    predispatch_failure_chain_sha256 = hash_json(predispatch_failure_chain)
    checks = {
        "execution_mode_is_trajectory_continuation": (
            manifest.get("execution_mode") == "trajectory_continuation"
        ),
        "normal_validator_command_is_source_protocol_command": (
            manifest.get("command") == manifest.get("source_protocol_command")
        ),
        "wrapper_command_provenance_matches": (
            manifest.get("wrapper_command") == wrapper_command
        ),
        "validator_identity_matches": (
            continuation.get("validator_sha256") == validator_sha256
            and claim.get("validator_sha256") == validator_sha256
            and attestation.get("validator_sha256") == validator_sha256
        ),
        "execution_identity_matches": (
            continuation.get("execution_identity_sha256")
            == execution_identity_sha256
            and claim.get("continuation_execution_identity_sha256")
            == execution_identity_sha256
            and attestation.get("continuation_execution_identity_sha256")
            == execution_identity_sha256
        ),
        "source_artifacts_remain_immutable": source_hashes_match,
        "claim_exists_and_matches_manifest": (
            claim_path.is_file()
            and continuation.get("claim_id") == claim.get("claim_id")
            and claim.get("source_attempt") == plan["source_attempt"]
            and claim.get("source_lock_sha256") == plan["source_lock_sha256"]
            and claim.get("source_hashes") == plan["source_hashes"]
            and claim.get("boundary_decision_index") == boundary_index
            and claim.get("boundary_provider_request_sha256")
            == _first_live_dispatch_request_sha256(plan["boundary"])
        ),
        "claim_snapshot_preserves_immutable_fields": immutable_claim_matches,
        "claim_snapshot_preserves_predispatch_chain": (
            snapshot.get("predispatch_failures")
            == claim.get("predispatch_failures")
        ),
        "seam_matches_verified_boundary": (
            seam_path.is_file()
            and seam.get("status") == "verified"
            and seam.get("decision_index") == boundary_index
            and seam.get("replayed_decisions") == plan["accepted_decision_count"]
            and seam.get("model_messages_sha256")
            == plan["boundary"]["model_messages_sha256"]
            and seam.get("provider_request_sha256")
            == plan["boundary"]["provider_request_sha256"]
            and _seam_first_dispatch_matches(seam, plan["boundary"])
            and seam.get("engine_prefix_sha256")
            == plan["boundary"]["engine_prefix_sha256"]
            and seam.get("state_sha256") == plan["boundary"]["state_sha256"]
            and seam.get("source_lock_sha256") == plan["source_lock_sha256"]
            and seam.get("source_tree_identity_sha256")
            == plan["source_tree_identity_sha256"]
            and seam.get("provider_calls") == 0
            and seam.get("credential_file_read") is False
            and seam.get("credential_values_recorded") is False
            and claim.get("seam_proof_sha256") == seam_sha256
        ),
        "predispatch_failures_are_safe": predispatch_failures_are_safe,
        "dispatches_are_complete_and_contiguous": (
            bool(dispatches)
            and [row.get("decision_index") for row in dispatches]
            == expected_dispatch_indices
            and [row.get("logical_provider_call_index") for row in dispatches]
            == list(range(len(dispatches)))
            and all(row.get("status") == "provider_dispatch_started" for row in dispatches)
            and (
                not dispatches
                or (
                    dispatches[0].get("model_messages_sha256")
                    == plan["boundary"].get(
                        "first_live_dispatch_model_messages_sha256",
                        plan["boundary"]["model_messages_sha256"],
                    )
                    and dispatches[0].get("provider_request_sha256")
                    == _first_live_dispatch_request_sha256(plan["boundary"])
                )
            )
        ),
        "attestation_matches_execution": (
            attestation_path.is_file()
            and attestation.get("status") == "execution_complete"
            and attestation.get("claim_id") == claim.get("claim_id")
            and attestation.get("source_attempt") == plan["source_attempt"]
            and attestation.get("source_lock_sha256") == plan["source_lock_sha256"]
            and attestation.get("source_tree_identity_sha256")
            == plan["source_tree_identity_sha256"]
            and attestation.get("protocol_sha256") == plan["protocol_sha256"]
            and attestation.get("replayed_decision_count")
            == plan["accepted_decision_count"]
            and attestation.get("provider_calls") == len(dispatches)
            and attestation.get("credential_file_read") is True
            and attestation.get("claim_snapshot_sha256")
            == claim_snapshot_sha256
            and attestation.get("predispatch_failure_chain_sha256")
            == predispatch_failure_chain_sha256
            and attestation.get("boundary_decision_index") == boundary_index
            and attestation.get("seam_proof_sha256") == seam_sha256
            and attestation.get("dispatch_sha256")
            == [hash_file(path) for path in dispatch_paths]
            and attestation.get("post_run_source_lock")
            == {
                "identity_sha256": plan["source_lock_sha256"],
                "source_tree_identity_sha256": plan[
                    "source_tree_identity_sha256"
                ],
                "protocol_sha256": plan["protocol_sha256"],
            }
            and claim.get("credential_file_read") is True
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "execution_identity_sha256": execution_identity_sha256,
        "claim_path": str(claim_path),
        "claim_snapshot_sha256": claim_snapshot_sha256,
        "predispatch_failure_chain": predispatch_failure_chain,
        "predispatch_failure_chain_sha256": predispatch_failure_chain_sha256,
        "seam_proof_sha256": seam_sha256,
        "attestation_sha256": (
            hash_file(attestation_path) if attestation_path.is_file() else None
        ),
        "dispatch_sha256": [hash_file(path) for path in dispatch_paths],
    }
