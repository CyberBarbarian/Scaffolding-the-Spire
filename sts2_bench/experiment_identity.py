"""Stable experiment identities and provenance for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_SCHEMA_VERSION = 2
PROMPT_SOURCE_PATHS = (
    "sts2_bench/actions.py",
    "sts2_bench/agents.py",
    "sts2_bench/context.py",
    "sts2_bench/evidence.py",
    "sts2_bench/feedback.py",
    "sts2_bench/multi_agent.py",
    "sts2_bench/posterior.py",
    "sts2_bench/route_candidates.py",
    "sts2_bench/route_graph.py",
    "sts2_bench/route_plan.py",
    "sts2_bench/run_context.py",
    "sts2_bench/runner.py",
    "sts2_bench/routing.py",
    "sts2_bench/sequence_safety.py",
    "sts2_bench/sequence_validation.py",
    "sts2_bench/sequences.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def hash_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str, text: bool = True) -> str | bytes | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            check=False,
            text=text,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def repo_provenance(root: str | Path = ROOT) -> dict[str, Any]:
    repo_root = Path(root)
    commit = _git_output(repo_root, "rev-parse", "HEAD")
    status = _git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git_output(repo_root, "diff", "--binary", "HEAD", "--", text=False)
    untracked = _git_output(repo_root, "ls-files", "--others", "--exclude-standard")
    status_known = isinstance(status, str)
    diff_known = isinstance(diff, bytes)
    untracked_known = isinstance(untracked, str)
    status_text = status.strip() if status_known else ""
    diff_bytes = diff if diff_known else b""
    untracked_hashes = []
    if isinstance(untracked, str):
        for relative_path in sorted(path for path in untracked.splitlines() if path):
            untracked_hashes.append({"path": relative_path, "sha256": hash_file(repo_root / relative_path)})
    dirty_payload = (
        status_text.encode("utf-8")
        + b"\0"
        + diff_bytes
        + b"\0"
        + canonical_json(untracked_hashes).encode("utf-8")
    )
    provenance_known = status_known and diff_known and untracked_known
    dirty = bool(status_text) if provenance_known else None
    return {
        "git_commit": commit.strip() if isinstance(commit, str) else None,
        "git_dirty": dirty,
        "git_status_known": provenance_known,
        "git_probe_status": {
            "commit": isinstance(commit, str),
            "status": status_known,
            "diff": diff_known,
            "untracked": untracked_known,
        },
        "git_dirty_hash": hashlib.sha256(dirty_payload).hexdigest() if dirty else None,
        "untracked_file_hashes": untracked_hashes,
    }


def prompt_source_hash(root: str | Path = ROOT) -> str:
    repo_root = Path(root)
    payload: list[dict[str, str | None]] = []
    for relative_path in PROMPT_SOURCE_PATHS:
        payload.append({"path": relative_path, "sha256": hash_file(repo_root / relative_path)})
    return hash_json(payload)


def safe_identifier(value: str, *, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("._-").lower()
    return safe or fallback


def _identity_config(run_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(run_config)
    config.pop("argv", None)
    config.pop("logging", None)
    config.pop("identity", None)
    config.pop("provenance", None)
    game = config.pop("game", {})
    benchmark = config.get("benchmark")
    if isinstance(benchmark, dict):
        for key in ("seeds", "seeds_file", "setup_actions_file", "setup_action_count"):
            benchmark.pop(key, None)
    return {"agent": config.get("agent"), "benchmark": benchmark, "game": game}


def build_run_identity(
    run_config: dict[str, Any],
    *,
    seeds: list[str],
    experiment_id: str | None = None,
    scenario_id: str | None = None,
    replicate_id: str | None = None,
    run_id: str | None = None,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    game = run_config.get("game") if isinstance(run_config.get("game"), dict) else {}
    benchmark = run_config.get("benchmark") if isinstance(run_config.get("benchmark"), dict) else {}
    agent = run_config.get("agent") if isinstance(run_config.get("agent"), dict) else {}
    seed_pool_hash = hash_json(list(seeds))
    setup_actions_file = benchmark.get("setup_actions_file")
    setup_actions_hash = hash_file(setup_actions_file)
    config_id = hash_json(
        {
            "config": _identity_config(run_config),
            "seed_pool_hash": seed_pool_hash,
            "setup_actions_hash": setup_actions_hash,
        }
    )
    default_scenario = f"{game.get('character') or 'unknown'}-a{game.get('ascension', 0)}"
    if benchmark.get("setup_action_count"):
        default_scenario += f"-setup-{(setup_actions_hash or 'unknown')[:12]}"
    default_replicate = (
        f"seed-{safe_identifier(str(seeds[0]), fallback='unknown')}"
        if len(seeds) == 1
        else f"seedpool-{seed_pool_hash[:12]}"
    )
    model_payload = {
        "base_url": agent.get("base_url"),
        "model": agent.get("model"),
        "agent_type": agent.get("type"),
    }
    model_fingerprint = hash_json(model_payload)
    resolved_experiment_id = safe_identifier(experiment_id or "sts2-bench", fallback="sts2-bench")
    resolved_scenario_id = safe_identifier(scenario_id or default_scenario, fallback="scenario")
    resolved_replicate_id = safe_identifier(replicate_id or default_replicate, fallback="replicate")
    resolved_run_id = safe_identifier(
        run_id
        or (
            f"{resolved_scenario_id}--{resolved_replicate_id}"
            f"--cfg-{config_id[:12]}--model-{model_fingerprint[:12]}"
        ),
        fallback=config_id[:16],
    )
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "experiment_id": resolved_experiment_id,
        "config_id": config_id,
        "scenario_id": resolved_scenario_id,
        "replicate_id": resolved_replicate_id,
        "model_fingerprint": model_fingerprint,
        "run_id": resolved_run_id,
        "hashes": {
            "resolved_config": hash_json(run_config),
            "seed_pool": seed_pool_hash,
            "tracked_benchmark_config": hash_file(Path(root) / "sts2_bench" / "benchmark.yaml"),
            "prompt_sources": prompt_source_hash(root),
            "setup_actions": setup_actions_hash,
            "experience_memory_snapshot": None,
        },
    }


def default_trace_path(identity: dict[str, Any], root: str | Path = "results") -> Path:
    return Path(root) / str(identity["experiment_id"]) / str(identity["run_id"]) / "trace.jsonl"


def manifest_path_for_trace(trace_path: str | Path) -> Path:
    trace = Path(trace_path)
    return trace.with_name(f"{trace.stem}.manifest.json")
