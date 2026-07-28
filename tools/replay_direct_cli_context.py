"""Replay one recorded Direct-CLI request to preflight a larger output budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sts2_bench.direct_cli_baseline import (
    ModelProtocolError,
    OpenAICompatTerminalModel,
    ProviderConfigurationError,
    ProviderTransportError,
    _python_runtime_identity,
    terminal_command_from_response,
    terminal_output_format_valid,
)
from sts2_bench.direct_cli_experiment import (
    DirectCliExperimentError,
    _load_and_validate_external_environment,
    _verify_live_source_inventory,
)
from sts2_bench.experiment_identity import hash_file, hash_json


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def _selection_plan_binding(
    plan_path: Path,
    *,
    selection_name: str,
    artifacts: Path,
    decision_index: int,
    output: Path,
    preflight_max_tokens: int | None,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    if not plan_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Selection plan must remain inside the benchmark repository")
    plan = _read_json(plan_path)
    sources = plan.get("sources")
    contexts = plan.get("contexts")
    call_policy = plan.get("call_policy")
    actual_output_control = (
        "provider_default_omitted" if preflight_max_tokens is None else "explicit_max_tokens"
    )
    output_root_relative = plan.get("output_root")
    if (
        plan.get("schema_version") != 1
        or plan.get("status") != "locked_before_provider_calls"
        or plan.get("paper_sample") is not False
        or plan.get("preflight_output_limit_control") != actual_output_control
        or (actual_output_control == "provider_default_omitted" and preflight_max_tokens is not None)
        or not isinstance(output_root_relative, str)
        or not output_root_relative
        or Path(output_root_relative).is_absolute()
        or not isinstance(call_policy, dict)
        or call_policy.get("max_provider_calls_per_context") != 1
        or call_policy.get("request_retries") != 0
        or call_policy.get("failure_disposition") != "preserve_and_invalidate_plan"
        or not isinstance(sources, list)
        or not isinstance(contexts, list)
    ):
        raise ValueError("Selection plan contract is invalid")
    expected_output_root = (REPO_ROOT / output_root_relative).resolve()
    if (
        not expected_output_root.is_relative_to(REPO_ROOT.resolve())
        or output.parent.resolve() != expected_output_root
    ):
        raise ValueError("Preflight output does not match the plan-locked output root")
    matching_contexts = [
        value for value in contexts if isinstance(value, dict) and value.get("name") == selection_name
    ]
    if len(matching_contexts) != 1:
        raise ValueError("Selection name is not unique in the locked plan")
    context = matching_contexts[0]
    matching_sources = [
        value
        for value in sources
        if isinstance(value, dict) and value.get("source_id") == context.get("source_id")
    ]
    if len(matching_sources) != 1:
        raise ValueError("Selection source is not unique in the locked plan")
    source = matching_sources[0]
    relative_artifacts = source.get("artifacts_path")
    if (
        not isinstance(relative_artifacts, str)
        or not relative_artifacts
        or Path(relative_artifacts).is_absolute()
    ):
        raise ValueError("Selection source path is invalid")
    expected_artifacts = (REPO_ROOT / relative_artifacts).resolve(strict=True)
    if not expected_artifacts.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Selection source path escapes the benchmark repository")
    if expected_artifacts != artifacts.resolve():
        raise ValueError("Replay artifacts do not match the locked selection plan")
    if context.get("decision_index") != decision_index:
        raise ValueError("Replay decision index does not match the locked selection plan")
    if output.stem != selection_name:
        raise ValueError("Preflight output name does not match the locked selection name")
    expected_hashes = {
        "config_sha256": hash_file(artifacts / "config.json"),
        "events_sha256": hash_file(artifacts / "events.jsonl"),
        "terminal_sha256": hash_file(artifacts / "terminal.txt"),
    }
    if any(source.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("Selection source hashes drifted after the plan was locked")
    config = _read_json(artifacts / "config.json")
    events = [
        json.loads(line)
        for line in (artifacts / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terminal_decisions = [
        value for value in events if isinstance(value, dict) and value.get("type") == "terminal_decision"
    ]
    selected_events = [
        value
        for value in events
        if isinstance(value, dict)
        and value.get("type") in {"terminal_decision", "model_protocol_error"}
        and value.get("decision_index") == decision_index
    ]
    if (
        source.get("character") != context.get("character")
        or config.get("character") != source.get("character")
        or config.get("model") != plan.get("model")
        or config.get("base_url") != plan.get("base_url")
        or config.get("max_tokens") != source.get("source_max_tokens")
        or source.get("terminal_decision_count") != len(terminal_decisions)
        or len(selected_events) != 1
    ):
        raise ValueError("Selection source semantics do not match the locked plan")
    selection = context.get("selection")
    terminal_count = len(terminal_decisions)
    if selection in {"early", "middle", "late"}:
        expected_index = {
            "early": 0,
            "middle": (terminal_count - 1) // 2,
            "late": terminal_count - 1,
        }[selection]
        if decision_index != expected_index or selected_events[0].get("type") != "terminal_decision":
            raise ValueError("Standard replay violates the locked endpoint/midpoint selection rule")
    elif selection == "historical_failure":
        metadata = selected_events[0].get("provider_metadata")
        if (
            selected_events[0].get("type") != "model_protocol_error"
            or not isinstance(metadata, dict)
            or metadata.get("finish_reason") != "length"
        ):
            raise ValueError("Historical replay does not match the locked length-failure ledger")
    else:
        raise ValueError("Selection plan contains an unsupported selection class")
    return {
        "selection_plan_id": plan.get("selection_plan_id"),
        "selection_plan_path": plan_path.relative_to(REPO_ROOT.resolve()).as_posix(),
        "selection_plan_sha256": hash_file(plan_path),
        "selection_name": selection_name,
        "output_path": output.relative_to(REPO_ROOT.resolve()).as_posix(),
        "selection": selection,
        "source_id": source.get("source_id"),
        "source_artifacts_path": relative_artifacts,
        "source_terminal_decision_count": source.get("terminal_decision_count"),
        "source_max_tokens": source.get("source_max_tokens"),
    }


def reconstruct_request(artifacts: Path, *, decision_index: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    config_path = artifacts / "config.json"
    events_path = artifacts / "events.jsonl"
    terminal_path = artifacts / "terminal.txt"
    config = _read_json(config_path)
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = [
        row
        for row in events
        if isinstance(row, dict)
        and row.get("type") in {"terminal_decision", "model_protocol_error"}
        and row.get("decision_index") == decision_index
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one recorded request at decision_index={decision_index}")
    event = candidates[0]
    start = event.get("terminal_window_start")
    end = event.get("terminal_window_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError("Recorded terminal window offsets are invalid")
    terminal = terminal_path.read_text(encoding="utf-8")
    if end > len(terminal):
        raise ValueError("Recorded terminal window exceeds terminal.txt")
    visible = terminal[start:end]
    if hashlib.sha256(visible.encode("utf-8")).hexdigest() != event.get("terminal_window_sha256"):
        raise ValueError("Recorded terminal window hash does not match terminal.txt")
    system_prompt = config.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("config.json does not contain the recorded system prompt")
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                + visible
                + "\n\nReturn only the next terminal input line."
            ),
        },
    ]
    if _hash_json(messages) != event.get("model_messages_sha256"):
        raise ValueError("Reconstructed request does not match the recorded model_messages_sha256")
    return config, event, messages


def _selected_api_key() -> str:
    for name in ("STS2_BENCH_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("No API key was loaded")


class _OutputReservation:
    """Own an output path from before the provider call until atomic commit."""

    def __init__(self, path: Path, marker: bytes, token: str) -> None:
        self.path = path
        self._marker = marker
        self._token = token
        self._committed = False
        self._provider_call_started = False

    def _still_owned(self) -> bool:
        try:
            return self.path.read_bytes() == self._marker
        except FileNotFoundError:
            return False

    def commit(self, value: dict[str, Any]) -> None:
        if self._committed or not self._still_owned():
            raise ValueError("Preflight output reservation was lost before commit")
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if not self._still_owned():
                raise ValueError("Preflight output reservation was lost before commit")
            os.replace(temporary, self.path)
            self._committed = True
        finally:
            temporary.unlink(missing_ok=True)

    def mark_provider_call_started(self) -> None:
        if self._committed or self._provider_call_started or not self._still_owned():
            raise ValueError("Preflight output reservation cannot start another provider call")
        started_marker = (
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "provider_call_started",
                    "pid": os.getpid(),
                    "token": self._token,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        with self.path.open("r+b") as handle:
            if handle.read() != self._marker:
                raise ValueError("Preflight output reservation was lost before provider call")
            handle.seek(0)
            handle.truncate()
            handle.write(started_marker)
            handle.flush()
            os.fsync(handle.fileno())
        self._marker = started_marker
        self._provider_call_started = True

    def release(self) -> None:
        if not self._committed and not self._provider_call_started and self._still_owned():
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> _OutputReservation:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _source_provenance(artifacts: Path, *, decision_index: int) -> tuple[Path, dict[str, Any]]:
    attempt_dir = artifacts.parent
    attempt_manifest = _read_json(attempt_dir / "attempt_manifest.json")
    attempt_result = _read_json(attempt_dir / "attempt_result.json")
    job_status_path = attempt_dir.parent.parent / "job_status.json"
    job_status = _read_json(job_status_path)
    job_status_result = dict(job_status)
    job_status_result.pop("attempt_history", None)
    if job_status_result != attempt_result:
        raise ValueError("Source attempt_result does not match its accepted job_status copy")
    summary_path = artifacts / "summary.json"
    summary = _read_json(summary_path)
    out_root = next(
        (ancestor for ancestor in artifacts.parents if (ancestor / "source_lock.json").is_file()),
        None,
    )
    if out_root is None:
        raise ValueError("Source artifacts are not inside a source-locked experiment root")
    source_lock_record = _read_json(out_root / "source_lock.json")
    identity_payload = {
        "protocol_sha256": source_lock_record.get("protocol_sha256"),
        "runtime": source_lock_record.get("runtime"),
        "files": source_lock_record.get("files"),
    }
    if source_lock_record.get("identity_sha256") != hash_json(identity_payload):
        raise ValueError("Source identity record is invalid")
    inventory = source_lock_record.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Source identity record has no inventory")
    normalized_inventory = [
        {"label": item.get("label"), "sha256": item.get("sha256"), "size": item.get("size")}
        for item in inventory
        if isinstance(item, dict)
    ]
    if (
        len(normalized_inventory) != len(inventory)
        or normalized_inventory != source_lock_record.get("files")
        or source_lock_record.get("execution_mode") != "active_worktree"
        or not all(item.get("tracked") is True for item in inventory if isinstance(item, dict))
        or not all(
            isinstance(item.get("label"), str)
            and isinstance(item.get("path"), str)
            and Path(item["path"]).resolve().is_file()
            for item in inventory
            if isinstance(item, dict)
        )
    ):
        raise ValueError("Source active-worktree identity inventory is invalid")
    try:
        verified_source_identity = _verify_live_source_inventory(source_lock_record)
    except DirectCliExperimentError as exc:
        raise ValueError(f"Source active worktree is invalid: {exc}") from exc
    source_tree_identity = source_lock_record.get("source_tree_identity_sha256")
    if verified_source_identity != source_tree_identity:
        raise ValueError("Source active-worktree identity does not match its record")

    protocol_label = source_lock_record.get("protocol_label")
    protocol_items = [item for item in inventory if item.get("label") == protocol_label]
    if not isinstance(protocol_label, str) or len(protocol_items) != 1:
        raise ValueError("Source protocol is not uniquely identified")
    protocol_path = Path(str(source_lock_record.get("protocol_path") or "")).resolve()
    if not protocol_path.is_file() or Path(str(protocol_items[0].get("path") or "")).resolve() != protocol_path:
        raise ValueError("Source protocol path is invalid")
    protocol_sha256 = hash_file(protocol_path)
    if (
        protocol_sha256 != source_lock_record.get("protocol_sha256")
        or protocol_items[0].get("sha256") != protocol_sha256
    ):
        raise ValueError("Source protocol does not match its record")
    protocol = _read_json(protocol_path)

    config = _read_json(artifacts / "config.json")
    system_prompt = config.get("system_prompt")
    if (
        not isinstance(system_prompt, str)
        or not system_prompt
        or config.get("system_prompt_sha256")
        != hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    ):
        raise ValueError("Source system prompt does not match its recorded hash")
    source_lock = source_lock_record.get("identity_sha256")
    experiment_id = attempt_manifest.get("experiment_id")
    seed = attempt_manifest.get("seed")
    replicate = attempt_manifest.get("replicate")
    attempt = attempt_manifest.get("attempt")
    expected_attempt_dir = (
        out_root
        / "runs"
        / str(seed)
        / f"replicate_{replicate:02d}"
        / "attempts"
        / f"attempt_{attempt:03d}"
        if isinstance(replicate, int)
        and not isinstance(replicate, bool)
        and isinstance(attempt, int)
        and not isinstance(attempt, bool)
        else None
    )
    protocol_config_matches = all(
        config.get(config_key) == protocol.get(protocol_key)
        for config_key, protocol_key in (
            ("model", "model"),
            ("base_url", "base_url"),
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("history_chars", "history_chars"),
            ("max_decisions", "max_decisions"),
            ("character", "character"),
            ("ascension", "ascension"),
            ("lang", "lang"),
            ("timeout", "request_timeout_sec"),
        )
    ) and (
        config.get("request_retries") == protocol.get("request_retries")
        if "request_retries" in protocol
        else config.get("transport_retries") == protocol.get("transport_retries")
    )
    protocol_rules = protocol.get("game_rules")
    expected_config_rules = (
        {key: value for key, value in protocol_rules.items() if key != "source"}
        if isinstance(protocol_rules, dict)
        else None
    )
    command = attempt_manifest.get("command")

    def command_option(name: str) -> str | None:
        if not isinstance(command, list):
            return None
        try:
            index = command.index(name)
        except ValueError:
            return None
        return str(command[index + 1]) if index + 1 < len(command) else None

    command_artifacts = command_option("--out-dir")
    common_binding_valid = (
        config.get("source_lock_sha256") == source_lock
        and config.get("source_tree_identity_sha256") == source_tree_identity
        and config.get("seed") == seed
        and config.get("game_rules") == expected_config_rules
        and protocol_config_matches
        and experiment_id == protocol.get("experiment_id")
        and attempt_manifest.get("formal_sample") is True
        and isinstance(protocol.get("seeds"), list)
        and seed in protocol["seeds"]
        and isinstance(replicate, int)
        and not isinstance(replicate, bool)
        and 1 <= replicate <= int(protocol.get("repeats_per_seed") or 0)
        and isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and attempt >= 1
        and expected_attempt_dir is not None
        and attempt_dir.resolve() == expected_attempt_dir.resolve()
        and isinstance(command_artifacts, str)
        and Path(command_artifacts).resolve() == artifacts.resolve()
        and command_option("--seed") == seed
        and command_option("--source-lock-sha256") == source_lock
        and command_option("--source-tree-identity-sha256") == source_tree_identity
        and attempt_manifest.get("source_lock_sha256") == source_lock
        and attempt_manifest.get("source_tree_identity_sha256") == source_tree_identity
        and attempt_manifest.get("protocol_sha256") == protocol_sha256
        and attempt_result.get("experiment_id") == experiment_id
        and attempt_result.get("seed") == seed
        and attempt_result.get("replicate") == replicate
        and attempt_result.get("attempt") == attempt
        and attempt_result.get("source_lock_sha256") == source_lock
        and attempt_result.get("source_tree_identity_sha256") == source_tree_identity
        and attempt_result.get("protocol_sha256") == protocol_sha256
        and attempt_result.get("identity_stable") is True
        and attempt_result.get("source_tree_stable") is True
    )
    if not common_binding_valid:
        raise ValueError("Source attempt is not bound to its source lock")

    _, selected_event, selected_messages = reconstruct_request(
        artifacts,
        decision_index=decision_index,
    )
    accepted_request_audit = attempt_result.get("validation", {}).get("request_audit", {})
    accepted_decisions = accepted_request_audit.get("decisions")
    accepted_matches = [
        item
        for item in accepted_decisions
        if isinstance(item, dict) and item.get("decision_index") == decision_index
    ] if isinstance(accepted_decisions, list) else []
    selected_messages_sha256 = _hash_json(selected_messages)
    if (
        accepted_request_audit.get("valid") is not True
        or len(accepted_matches) != 1
        or accepted_matches[0].get("valid") is not True
        or accepted_matches[0].get("reconstructed_model_messages_sha256") != selected_messages_sha256
        or selected_event.get("model_messages_sha256") != selected_messages_sha256
    ):
        raise ValueError("Source request does not match the request accepted by the formal audit")

    if (
        attempt_result.get("status") == "succeeded"
        and attempt_result.get("formal_sample") is True
        and isinstance(attempt_result.get("validation"), dict)
        and attempt_result["validation"].get("valid") is True
    ):
        attempt_eligibility = "accepted_formal_sample"
    elif (
        attempt_result.get("status") == "failed"
        and attempt_result.get("formal_sample") is False
        and attempt_result.get("termination_class") == "model_protocol_error"
        and summary.get("formal_behavior_sample") is True
        and summary.get("termination_class") == "model_protocol_error"
    ):
        attempt_eligibility = "recorded_model_protocol_error"
    else:
        raise ValueError("Source attempt is neither an accepted formal sample nor a recorded model protocol error")
    root_report_path = out_root / "report_required.json"
    root_report: dict[str, Any] | None = None
    if root_report_path.is_file():
        root_report = _read_json(root_report_path)
        if root_report.get("experiment_id") != experiment_id:
            raise ValueError("Source root report_required does not match the attempt experiment")
    protocol_disposition = root_report.get("protocol_disposition") if root_report is not None else None
    source_eligibility = (
        "invalidated_protocol_context"
        if protocol_disposition == "invalidated_do_not_resume_or_count_for_paper"
        else attempt_eligibility
    )
    return out_root, {
        "experiment_id": experiment_id,
        "seed": seed,
        "character": config.get("character"),
        "replicate": replicate,
        "attempt": attempt,
        "source_eligibility": source_eligibility,
        "attempt_eligibility": attempt_eligibility,
        "source_protocol_disposition": protocol_disposition,
        "source_report_reason": root_report.get("reason") if root_report is not None else None,
        "source_report_required_sha256": hash_file(root_report_path) if root_report is not None else None,
        "attempt_status": attempt_result.get("status"),
        "formal_sample": attempt_result.get("formal_sample"),
        "identity_stable": attempt_result.get("identity_stable"),
        "source_lock_sha256": source_lock,
        "source_tree_identity_sha256": source_tree_identity,
        "protocol_label": protocol_label,
        "protocol_sha256": protocol_sha256,
        "source_python_runtime": source_lock_record.get("runtime"),
        "attempt_manifest_sha256": hash_file(attempt_dir / "attempt_manifest.json"),
        "attempt_result_sha256": hash_file(attempt_dir / "attempt_result.json"),
        "job_status_sha256": hash_file(job_status_path),
        "accepted_model_messages_sha256": selected_messages_sha256,
        "source_summary_sha256": hash_file(summary_path),
        "source_lock_record_sha256": hash_file(out_root / "source_lock.json"),
    }


def validate_output_path(output: Path, *, source_root: Path) -> Path:
    resolved = output.resolve()
    if _path_is_within(resolved, source_root.resolve()):
        raise ValueError("Preflight output must be outside the source formal result root")
    if resolved.exists():
        raise ValueError("Refusing to overwrite an existing preflight output")
    return resolved


def reserve_output_path(output: Path, *, source_root: Path) -> _OutputReservation:
    resolved = validate_output_path(output, source_root=source_root)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    marker = (
        json.dumps(
            {
                "schema_version": 1,
                "status": "reserved",
                "pid": os.getpid(),
                "token": token,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Refusing to overwrite an existing preflight output") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(marker)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise
    return _OutputReservation(resolved, marker, token)


def load_preflight_environment(
    env_file: Path,
    *,
    base_url: str,
    source_root: Path,
    output_parent: Path,
) -> None:
    """Load credentials only from outside both the formal source and output trees."""

    _load_and_validate_external_environment(
        env_file,
        base_url=base_url,
        forbidden_roots=(source_root, output_parent),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--decision-index", type=int, required=True)
    output_limit = parser.add_mutually_exclusive_group(required=True)
    output_limit.add_argument("--max-tokens", type=int)
    output_limit.add_argument("--omit-max-tokens", action="store_true")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--selection-name")
    args = parser.parse_args(argv)
    if args.decision_index < 0 or (not args.omit_max_tokens and args.max_tokens < 1):
        parser.error("decision-index must be non-negative and max-tokens must be positive when provided")

    if (args.selection_plan is None) != (args.selection_name is None):
        parser.error("--selection-plan and --selection-name must be provided together")
    artifacts = args.artifacts.resolve(strict=True)
    requested_output = args.output.resolve()
    selection_binding = (
        _selection_plan_binding(
            args.selection_plan,
            selection_name=args.selection_name,
            artifacts=artifacts,
            decision_index=args.decision_index,
            output=requested_output,
            preflight_max_tokens=None if args.omit_max_tokens else args.max_tokens,
        )
        if args.selection_plan is not None
        else None
    )
    source_root, source_provenance = _source_provenance(
        artifacts,
        decision_index=args.decision_index,
    )
    with reserve_output_path(args.output, source_root=source_root) as output_reservation:
        output = output_reservation.path
        config, event, messages = reconstruct_request(artifacts, decision_index=args.decision_index)
        load_preflight_environment(
            args.env_file,
            base_url=str(config["base_url"]),
            source_root=source_root,
            output_parent=output.parent,
        )
        attempts: list[dict[str, Any]] = []
        model = OpenAICompatTerminalModel(
            base_url=str(config["base_url"]),
            model=str(config["model"]),
            api_key=_selected_api_key(),
            temperature=float(config["temperature"]),
            timeout=float(config["timeout"]),
            max_tokens=None if args.omit_max_tokens else args.max_tokens,
            request_retries=0,
            attempt_observer=attempts.append,
        )
        preflight_provider_request: dict[str, Any] = {
            "model": config["model"],
            "messages": messages,
            "temperature": config["temperature"],
        }
        if not args.omit_max_tokens:
            preflight_provider_request["max_tokens"] = args.max_tokens
        base_result = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_artifacts": str(artifacts),
            "source_config_sha256": hash_file(artifacts / "config.json"),
            "source_events_sha256": hash_file(artifacts / "events.jsonl"),
            "source_terminal_sha256": hash_file(artifacts / "terminal.txt"),
            "source_provenance": source_provenance,
            "selection_plan_binding": selection_binding,
            "decision_index": args.decision_index,
            "recorded_model_messages_sha256": event["model_messages_sha256"],
            "recorded_event_type": event.get("type"),
            "recorded_event_error": event.get("error"),
            "recorded_output_format_valid": event.get("output_format_valid"),
            "recorded_provider_metadata": event.get("provider_metadata"),
            "recorded_max_tokens": config["max_tokens"],
            "preflight_max_tokens": None if args.omit_max_tokens else args.max_tokens,
            "preflight_output_limit_control": (
                "provider_default_omitted" if args.omit_max_tokens else "explicit_max_tokens"
            ),
            "model": config["model"],
            "base_url": config["base_url"],
            "preflight_provider_request": preflight_provider_request,
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": hash_file(Path(__file__).resolve()),
            "python_runtime": _python_runtime_identity(),
        }
        try:
            output_reservation.mark_provider_call_started()
            reply = model.complete(messages)
            command = terminal_command_from_response(reply.text)
            format_valid = terminal_output_format_valid(reply.text)
            finish_reason = reply.provider_metadata.get("finish_reason")
            passed = bool(format_valid and finish_reason == "stop")
            result = {
                **base_result,
                "status": "passed" if passed else "failed",
                "finish_reason": finish_reason,
                "output_format_valid": format_valid,
                "command": command,
                "response_text": reply.text,
                "response_text_sha256": hashlib.sha256(reply.text.encode("utf-8")).hexdigest(),
                "usage": reply.usage,
                "provider_metadata": reply.provider_metadata,
                "provider_attempts": attempts,
            }
        except (ModelProtocolError, ProviderConfigurationError, ProviderTransportError, ValueError) as exc:
            result = {
                **base_result,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "provider_attempts": attempts,
            }
            passed = False
        output_reservation.commit(result)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
