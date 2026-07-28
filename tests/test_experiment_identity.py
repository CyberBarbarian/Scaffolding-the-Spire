import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sts2_bench import run_benchmark
from sts2_bench.artifact_lease import ArtifactLeaseGroup
from sts2_bench.experiment_identity import (
    build_run_identity,
    default_trace_path,
    manifest_path_for_trace,
    prompt_source_hash,
    repo_provenance,
)
from sts2_bench.run_benchmark import _ensure_fresh_artifacts, _write_manifest
from sts2_bench.runner import RunResult
from sts2_bench.trace_analysis import TRACE_ANALYSIS_MARKER, TRACE_ANALYSIS_MARKER_CONTENT


def _run_config(out):
    return {
        "argv": ["--api-key", "<redacted>"],
        "agent": {
            "type": "coordinator",
            "base_url": "https://example.invalid",
            "model": "test-model",
            "prompt_style": "analysis",
        },
        "game": {"character": "Ironclad", "ascension": 0, "lang": "en"},
        "benchmark": {
            "seeds": ["SEED-A"],
            "seeds_file": None,
            "max_steps": 900,
            "setup_actions_file": None,
            "setup_action_count": 0,
        },
        "logging": {"out": out, "log_level": "decisions"},
    }


def test_run_identity_is_stable_across_artifact_paths():
    first = build_run_identity(_run_config("results/a.jsonl"), seeds=["SEED-A"])
    second = build_run_identity(_run_config("results/b.jsonl"), seeds=["SEED-A"])

    assert first["config_id"] == second["config_id"]
    assert first["run_id"] == second["run_id"]
    assert first["hashes"]["resolved_config"] != second["hashes"]["resolved_config"]
    assert first["scenario_id"] == "ironclad-a0"
    assert first["replicate_id"] == "seed-seed-a"


def test_default_trace_and_manifest_paths_share_a_run_directory():
    identity = build_run_identity(
        _run_config(None),
        seeds=["SEED-A"],
        experiment_id="route ablation",
    )

    trace = default_trace_path(identity)

    assert trace == Path("results") / identity["experiment_id"] / identity["run_id"] / "trace.jsonl"
    assert manifest_path_for_trace(trace) == trace.with_name("trace.manifest.json")


def test_setup_action_content_is_part_of_the_default_scenario_identity(tmp_path):
    setup = tmp_path / "setup.json"
    setup.write_text('[{"cmd":"action","action":"proceed"}]', encoding="utf-8")
    config = _run_config(None)
    config["benchmark"]["setup_actions_file"] = str(setup)
    config["benchmark"]["setup_action_count"] = 1

    first = build_run_identity(config, seeds=["SEED-A"])
    setup.write_text('[{"cmd":"action","action":"end_turn"}]', encoding="utf-8")
    second = build_run_identity(config, seeds=["SEED-A"])

    assert first["scenario_id"] != second["scenario_id"]
    assert first["run_id"] != second["run_id"]


def test_explicit_labels_do_not_hide_game_or_seed_identity_changes():
    first_config = _run_config(None)
    second_config = _run_config(None)
    second_config["game"]["character"] = "Silent"

    first = build_run_identity(
        first_config,
        seeds=["SEED-A"],
        scenario_id="shared-scenario",
        replicate_id="shared-replicate",
    )
    changed_game = build_run_identity(
        second_config,
        seeds=["SEED-A"],
        scenario_id="shared-scenario",
        replicate_id="shared-replicate",
    )
    changed_seed = build_run_identity(
        first_config,
        seeds=["SEED-B"],
        scenario_id="shared-scenario",
        replicate_id="shared-replicate",
    )

    assert first["run_id"] != changed_game["run_id"]
    assert first["run_id"] != changed_seed["run_id"]


def test_prompt_source_hash_includes_action_contract_sources(tmp_path):
    source = tmp_path / "sts2_bench" / "actions.py"
    source.parent.mkdir(parents=True)
    source.write_text("ACTION = 1\n", encoding="utf-8")
    first = prompt_source_hash(tmp_path)
    source.write_text("ACTION = 2\n", encoding="utf-8")
    second = prompt_source_hash(tmp_path)

    assert first != second


def test_prompt_source_hash_includes_agent_routing_source(tmp_path):
    source = tmp_path / "sts2_bench" / "routing.py"
    source.parent.mkdir(parents=True)
    source.write_text("ROUTE = 'ca'\n", encoding="utf-8")
    first = prompt_source_hash(tmp_path)
    source.write_text("ROUTE = 'occ'\n", encoding="utf-8")
    second = prompt_source_hash(tmp_path)

    assert first != second


def test_git_probe_failure_is_unknown_not_clean(monkeypatch, tmp_path):
    def fake_git_output(_root, *args, text=True):
        if args[:2] == ("rev-parse", "HEAD"):
            return "abc\n"
        if args and args[0] == "status":
            return None
        return b"" if not text else ""

    monkeypatch.setattr("sts2_bench.experiment_identity._git_output", fake_git_output)
    provenance = repo_provenance(tmp_path)

    assert provenance["git_dirty"] is None
    assert provenance["git_status_known"] is False


def test_existing_trace_requires_explicit_overwrite(tmp_path):
    trace = tmp_path / "trace.jsonl"
    manifest = tmp_path / "trace.manifest.json"
    trace.write_text("existing", encoding="utf-8")

    try:
        _ensure_fresh_artifacts(trace_path=trace, manifest_path=manifest, overwrite=False)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("existing trace should have been rejected")

    _ensure_fresh_artifacts(trace_path=trace, manifest_path=manifest, overwrite=True)


def test_run_artifact_lease_allows_only_one_concurrent_writer(tmp_path):
    trace = tmp_path / "trace.jsonl"
    manifest = tmp_path / "trace.manifest.json"
    start = threading.Barrier(2)
    attempted = threading.Barrier(3)
    release = threading.Event()

    def claim():
        start.wait()
        try:
            lease = run_benchmark._acquire_run_artifact_lease(
                trace_path=trace,
                manifest_path=manifest,
                trace_analysis_dir=None,
                overwrite=False,
            )
        except FileExistsError:
            attempted.wait()
            return "blocked"
        attempted.wait()
        release.wait(timeout=5)
        lease.release()
        return "acquired"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim) for _ in range(2)]
        attempted.wait(timeout=5)
        assert not trace.exists()
        assert not manifest.exists()
        release.set()
        outcomes = [future.result(timeout=5) for future in futures]

    assert sorted(outcomes) == ["acquired", "blocked"]


def test_run_artifact_lease_protects_shared_analysis_namespace(tmp_path):
    analysis = tmp_path / "shared-analysis"
    start = threading.Barrier(2)
    attempted = threading.Barrier(3)
    release = threading.Event()

    def claim(index):
        trace = tmp_path / f"trace-{index}.jsonl"
        manifest = tmp_path / f"trace-{index}.manifest.json"
        start.wait()
        try:
            lease = run_benchmark._acquire_run_artifact_lease(
                trace_path=trace,
                manifest_path=manifest,
                trace_analysis_dir=analysis,
                overwrite=False,
            )
        except FileExistsError:
            attempted.wait()
            return "blocked"
        attempted.wait()
        release.wait(timeout=5)
        lease.release()
        return "acquired"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim, index) for index in range(2)]
        attempted.wait(timeout=5)
        assert not analysis.exists()
        release.set()
        outcomes = [future.result(timeout=5) for future in futures]

    assert sorted(outcomes) == ["acquired", "blocked"]


def test_artifact_lease_group_releases_every_member_after_one_release_failure(tmp_path):
    released = []

    class FakeLease:
        def __init__(self, name, *, fail=False):
            self.path = tmp_path / name
            self.fail = fail

        def acquire(self):
            return self

        def release(self):
            released.append(self.path.name)
            if self.fail:
                raise OSError("synthetic release failure")

    first = FakeLease("a.lock")
    failing = FakeLease("b.lock", fail=True)
    group = ArtifactLeaseGroup([first, failing])
    group.acquire()

    with pytest.raises(OSError, match="synthetic release failure"):
        group.release()

    assert released == ["b.lock", "a.lock"]


def test_overwrite_removes_stale_trace_analysis_directory(tmp_path):
    trace = tmp_path / "trace.jsonl"
    manifest = tmp_path / "trace.manifest.json"
    analysis = tmp_path / "trace_analysis"
    stale = analysis / "old-seed" / "stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")
    (analysis / TRACE_ANALYSIS_MARKER).write_text(
        TRACE_ANALYSIS_MARKER_CONTENT,
        encoding="utf-8",
    )

    _ensure_fresh_artifacts(
        trace_path=trace,
        manifest_path=manifest,
        trace_analysis_dir=analysis,
        overwrite=True,
    )

    assert not analysis.exists()


def test_overwrite_rejects_analysis_directory_containing_trace_and_preserves_sentinel(tmp_path):
    analysis = tmp_path / "run"
    trace = analysis / "trace.jsonl"
    manifest = analysis / "trace.manifest.json"
    analysis.mkdir()
    sentinel = analysis / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    (analysis / TRACE_ANALYSIS_MARKER).write_text(
        TRACE_ANALYSIS_MARKER_CONTENT,
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="contains the trace or manifest"):
        _ensure_fresh_artifacts(
            trace_path=trace,
            manifest_path=manifest,
            trace_analysis_dir=analysis,
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_overwrite_rejects_project_root_analysis_directory(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    sentinel = project_root / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    (project_root / TRACE_ANALYSIS_MARKER).write_text(
        TRACE_ANALYSIS_MARKER_CONTENT,
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    monkeypatch.setattr(run_benchmark, "ROOT", project_root)

    with pytest.raises(FileExistsError, match="project root or its ancestor"):
        _ensure_fresh_artifacts(
            trace_path=output_root / "trace.jsonl",
            manifest_path=output_root / "trace.manifest.json",
            trace_analysis_dir=project_root,
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_overwrite_rejects_symlinked_analysis_directory(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    (target / TRACE_ANALYSIS_MARKER).write_text(
        TRACE_ANALYSIS_MARKER_CONTENT,
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis-link"
    try:
        analysis.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink is unavailable: {exc}")

    with pytest.raises(FileExistsError, match="linked trace analysis output"):
        _ensure_fresh_artifacts(
            trace_path=tmp_path / "output" / "trace.jsonl",
            manifest_path=tmp_path / "output" / "trace.manifest.json",
            trace_analysis_dir=analysis,
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_overwrite_rejects_directory_at_trace_output_before_manifest_write(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.mkdir()
    manifest = tmp_path / "trace.manifest.json"

    with pytest.raises(FileExistsError, match="Trace output exists and is not a file"):
        _ensure_fresh_artifacts(
            trace_path=trace,
            manifest_path=manifest,
            overwrite=True,
        )

    assert not manifest.exists()


def test_atomic_manifest_write_preserves_previous_file_when_replace_fails(tmp_path, monkeypatch):
    manifest = tmp_path / "trace.manifest.json"
    manifest.write_text('{"status":"old"}', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(run_benchmark.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        _write_manifest(manifest, {"status": "new"})

    assert manifest.read_text(encoding="utf-8") == '{"status":"old"}'
    assert not list(tmp_path.glob(".trace.manifest.json.*.tmp"))


def test_run_main_writes_manifest_and_refuses_silent_overwrite(tmp_path, monkeypatch):
    trace = tmp_path / "run" / "trace.jsonl"
    monkeypatch.setattr(run_benchmark, "load_benchmark_env", lambda: None)
    monkeypatch.setattr(run_benchmark, "load_seeds", lambda _path, _count: ["SEED-A"])
    monkeypatch.setattr(run_benchmark, "load_setup_actions", lambda _path: [])
    monkeypatch.setattr(run_benchmark, "repo_provenance", lambda _root: {"git_commit": "abc", "git_dirty": False, "git_dirty_hash": None})
    monkeypatch.setattr(run_benchmark, "agent_from_args", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        run_benchmark,
        "run_one",
        lambda **_kwargs: RunResult(
            seed="SEED-A",
            character="Ironclad",
            ascension=0,
            decision="game_over",
        ),
    )
    argv = [
        "--agent",
        "random",
        "--count",
        "1",
        "--out",
        str(trace),
        "--no-trace-analysis",
    ]

    assert run_benchmark.main(argv) == 0

    manifest_path = trace.with_name("trace.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["identity"]["scenario_id"] == "ironclad-a0"
    assert manifest["identity"]["replicate_id"] == "seed-seed-a"
    assert manifest["artifacts"]["trace"] == str(trace)
    assert manifest["provenance"]["git_commit"] == "abc"

    with pytest.raises(SystemExit) as exc_info:
        run_benchmark.main(argv)
    assert exc_info.value.code == 2

    assert run_benchmark.main([*argv, "--overwrite"]) == 0
