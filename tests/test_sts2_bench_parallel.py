import json
import sys
from pathlib import Path

import pytest

from sts2_bench.parallel_benchmark import SuiteJob, load_suite, main, run_suite


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_suite_accepts_generic_jobs(tmp_path):
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "jobs": [
                {"name": "default", "args": ["--agent", "random", "--count", "1"]},
                {"name": "cot map", "args": ["--agent", "coordinator", "--route-plan"]},
            ]
        },
    )

    jobs = load_suite(suite)

    assert jobs == [
        SuiteJob(name="default", args=["--agent", "random", "--count", "1"]),
        SuiteJob(name="cot map", args=["--agent", "coordinator", "--route-plan"]),
    ]


def test_load_suite_accepts_verified_pressure_combats(tmp_path):
    suite = tmp_path / "verified_pressure_suite.json"
    _write_json(
        suite,
        {
            "combats": [
                {
                    "id": "act1_boss_lowhp",
                    "run_command": [
                        "python",
                        "-X",
                        "utf8",
                        "-m",
                        "sts2_bench.run_benchmark",
                        "--model",
                        "deepseek-v4-flash",
                        "--setup-actions",
                        "setup.json",
                        "--out",
                        "old.jsonl",
                    ],
                }
            ]
        },
    )

    jobs = load_suite(suite)

    assert jobs == [
        SuiteJob(
            name="act1_boss_lowhp",
            args=[
                "--model",
                "deepseek-v4-flash",
                "--setup-actions",
                "setup.json",
                "--out",
                "old.jsonl",
            ],
        )
    ]


def test_load_suite_accepts_unified_combat_snapshot_run_command(tmp_path):
    suite = tmp_path / "verified_pressure_suite.json"
    _write_json(
        suite,
        {
            "combats": [
                {
                    "id": "act1_boss_lowhp",
                    "run_command": [
                        "python",
                        "-X",
                        "utf8",
                        "-m",
                        "sts2_bench",
                        "combat-snapshot",
                        "run",
                        "--snapshot",
                        "snapshot.json",
                        "--out",
                        "old.jsonl",
                    ],
                }
            ]
        },
    )

    jobs = load_suite(suite)

    assert jobs == [
        SuiteJob(
            name="act1_boss_lowhp",
            args=[
                "combat-snapshot",
                "run",
                "--snapshot",
                "snapshot.json",
                "--out",
                "old.jsonl",
            ],
        )
    ]


def test_dry_run_keeps_combat_snapshot_jobs_on_unified_cli(tmp_path):
    suite = tmp_path / "suite.json"
    out_root = tmp_path / "parallel_out"
    _write_json(
        suite,
        {
            "jobs": [
                {
                    "name": "snapshot job",
                    "args": [
                        "combat-snapshot",
                        "run",
                        "--snapshot",
                        "snapshot.json",
                        "--model",
                        "old-model",
                        "--out",
                        "old.jsonl",
                    ],
                }
            ]
        },
    )

    assert main(["--suite", str(suite), "--out-root", str(out_root), "--model", "fixed-model", "--dry-run"]) == 0

    plan = json.loads((out_root / "dry_run_plan.json").read_text(encoding="utf-8"))
    job = plan["jobs"][0]
    assert job["args"] == [
        "combat-snapshot",
        "run",
        "--snapshot",
        "snapshot.json",
        "--model",
        "fixed-model",
        "--out",
        str(out_root / "snapshot_job" / "trace.jsonl"),
    ]
    assert "--trace-analysis-dir" not in job["args"]
    assert job["command"] == [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "sts2_bench",
        "combat-snapshot",
        "run",
        "--snapshot",
        "snapshot.json",
        "--model",
        "fixed-model",
        "--out",
        str(out_root / "snapshot_job" / "trace.jsonl"),
    ]


def test_dry_run_writes_plan_with_isolated_trace_paths_and_model_override(tmp_path):
    suite = tmp_path / "suite.json"
    out_root = tmp_path / "parallel_out"
    _write_json(
        suite,
        {
            "jobs": [
                {
                    "name": "job one",
                    "args": ["--agent", "random", "--model", "old-model", "--out", "old.jsonl"],
                }
            ]
        },
    )

    assert main(["--suite", str(suite), "--out-root", str(out_root), "--model", "fixed-model", "--dry-run"]) == 0

    plan = json.loads((out_root / "dry_run_plan.json").read_text(encoding="utf-8"))
    job = plan["jobs"][0]
    assert job["name"] == "job one"
    assert job["job_dir"] == str(out_root / "job_one")
    assert job["trace"] == str(out_root / "job_one" / "trace.jsonl")
    assert job["args"] == [
        "--agent",
        "random",
        "--model",
        "fixed-model",
        "--out",
        str(out_root / "job_one" / "trace.jsonl"),
        "--trace-analysis-dir",
        str(out_root / "job_one" / "trace_analysis"),
    ]
    assert job["command"] == [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "sts2_bench",
        "run",
        "--agent",
        "random",
        "--model",
        "fixed-model",
        "--out",
        str(out_root / "job_one" / "trace.jsonl"),
        "--trace-analysis-dir",
        str(out_root / "job_one" / "trace_analysis"),
    ]
    assert job["identity"]["scenario_id"] == "job_one"
    assert job["identity"]["replicate_id"] == "replicate-001"
    assert len(job["identity"]["config_id"]) == 64
    assert len(job["identity"]["model_fingerprint"]) == 64
    assert job["identity"]["run_id"].startswith("job_one--replicate-001--cfg-")


def test_run_suite_writes_job_results_and_summary(tmp_path):
    out_root = tmp_path / "parallel_out"
    calls = []

    def fake_runner(command, *, stdout_path, stderr_path, cwd, timeout_sec):
        calls.append((command, stdout_path, stderr_path, cwd, timeout_sec))
        stdout_path.write_text('{"summary": {"runs": 1}}\n', encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return 0

    exit_code = run_suite(
        [
            SuiteJob(name="job a", args=["--agent", "random"]),
            SuiteJob(name="job b", args=["--agent", "random"]),
        ],
        out_root=out_root,
        workers=2,
        runner=fake_runner,
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert (out_root / "job_a" / "launcher_result.json").is_file()
    assert (out_root / "job_b" / "launcher_result.json").is_file()
    assert (out_root / "suite_manifest.json").is_file()

    summary = json.loads((out_root / "suite_summary.json").read_text(encoding="utf-8"))
    assert len(summary["suite_id"]) == 64
    assert "git_commit" in summary["provenance"]
    assert summary["job_count"] == 2
    assert summary["succeeded_count"] == 2
    assert summary["failed_count"] == 0
    assert [job["name"] for job in summary["jobs"]] == ["job a", "job b"]


def test_run_suite_refuses_non_empty_output_root(tmp_path):
    out_root = tmp_path / "parallel_out"
    out_root.mkdir()
    (out_root / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_suite(
            [SuiteJob(name="job", args=["--agent", "random"])],
            out_root=out_root,
            workers=1,
            runner=lambda *_args, **_kwargs: 0,
        )
