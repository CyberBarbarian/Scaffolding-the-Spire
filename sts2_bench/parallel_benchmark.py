"""Parallel suite runner for STS2 benchmark experiments."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .experiment_identity import hash_file, hash_json, prompt_source_hash, repo_provenance, safe_identifier


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_CLI_COMMANDS = {
    "run",
    "card-choice",
    "card-choice-bias",
    "card-choice-trace",
    "combat-snapshot",
    "parallel",
    "trace-analysis",
    "process-smoke",
}


@dataclass(frozen=True)
class SuiteJob:
    name: str
    args: list[str]
    scenario_id: str | None = None
    replicate_id: str | None = None


@dataclass(frozen=True)
class JobPlan:
    name: str
    safe_name: str
    args: list[str]
    command: list[str]
    job_dir: Path
    trace_path: Path
    trace_analysis_dir: Path
    stdout_path: Path
    stderr_path: Path
    launcher_config_path: Path
    launcher_result_path: Path
    config_id: str
    scenario_id: str
    replicate_id: str
    model_fingerprint: str
    run_id: str
    input_hashes: dict[str, str | None]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _coerce_args(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{context} must be a list of strings")
    return list(value)


def _args_from_run_command(command: list[str]) -> list[str]:
    for index in range(len(command) - 1):
        if command[index] == "-m" and command[index + 1] == "sts2_bench.run_benchmark":
            return command[index + 2 :]
        if command[index] == "-m" and command[index + 1] == "sts2_bench.combat_snapshot":
            return ["combat-snapshot", *command[index + 2 :]]
        if command[index] == "-m" and command[index + 1] == "sts2_bench":
            rest = command[index + 2 :]
            if rest and rest[0] == "run":
                return rest[1:]
            return rest
    return command


def load_suite(path: str | Path) -> list[SuiteJob]:
    suite_path = Path(path)
    data = _read_json(suite_path)
    jobs: list[SuiteJob] = []

    if "jobs" in data:
        entries = data["jobs"]
        if not isinstance(entries, list):
            raise ValueError(f"{suite_path} field 'jobs' must be a list")
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"{suite_path} jobs[{index - 1}] must be an object")
            name = str(entry.get("name") or entry.get("id") or f"job_{index:03d}")
            raw_args = entry.get("args", entry.get("argv"))
            if raw_args is None and "run_command" in entry:
                raw_args = _args_from_run_command(_coerce_args(entry["run_command"], context=f"jobs[{index - 1}].run_command"))
            jobs.append(
                SuiteJob(
                    name=name,
                    args=_coerce_args(raw_args, context=f"jobs[{index - 1}].args"),
                    scenario_id=str(entry["scenario_id"]) if entry.get("scenario_id") else None,
                    replicate_id=str(entry["replicate_id"]) if entry.get("replicate_id") else None,
                )
            )
    elif "combats" in data:
        entries = data["combats"]
        if not isinstance(entries, list):
            raise ValueError(f"{suite_path} field 'combats' must be a list")
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"{suite_path} combats[{index - 1}] must be an object")
            name = str(entry.get("name") or entry.get("id") or f"combat_{index:03d}")
            command = _coerce_args(entry.get("run_command"), context=f"combats[{index - 1}].run_command")
            jobs.append(
                SuiteJob(
                    name=name,
                    args=_args_from_run_command(command),
                    scenario_id=str(entry["scenario_id"]) if entry.get("scenario_id") else None,
                    replicate_id=str(entry["replicate_id"]) if entry.get("replicate_id") else None,
                )
            )
    else:
        raise ValueError(f"{suite_path} must contain either 'jobs' or 'combats'")

    if not jobs:
        raise ValueError(f"{suite_path} contains no jobs")
    return jobs


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-").lower()
    return safe or "job"


def _replace_option(args: list[str], option: str, value: str) -> list[str]:
    replaced = False
    output: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item == option:
            output.extend([option, value])
            index += 2 if index + 1 < len(args) else 1
            replaced = True
        elif item.startswith(f"{option}="):
            output.extend([option, value])
            index += 1
            replaced = True
        else:
            output.append(item)
            index += 1
    if not replaced:
        output.extend([option, value])
    return output


def _uses_unified_cli(args: list[str]) -> bool:
    return bool(args and args[0] in UNIFIED_CLI_COMMANDS)


def _option_value(args: list[str], option: str) -> str | None:
    for index, item in enumerate(args):
        if item == option and index + 1 < len(args):
            return args[index + 1]
        if item.startswith(f"{option}="):
            return item.split("=", 1)[1]
    return None


def _identity_args(args: list[str]) -> list[str]:
    omitted = {"--out", "--trace-analysis-dir"}
    output: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item in omitted:
            index += 2 if index + 1 < len(args) else 1
            continue
        if any(item.startswith(f"{option}=") for option in omitted):
            index += 1
            continue
        output.append(item)
        index += 1
    return output


def build_job_plans(
    jobs: list[SuiteJob],
    *,
    out_root: Path,
    model: str | None = None,
    python_executable: str | None = None,
) -> list[JobPlan]:
    executable = python_executable or sys.executable
    seen: dict[str, int] = {}
    plans: list[JobPlan] = []
    for job in jobs:
        base_name = _safe_name(job.name)
        seen[base_name] = seen.get(base_name, 0) + 1
        safe_name = base_name if seen[base_name] == 1 else f"{base_name}_{seen[base_name]:02d}"
        job_dir = out_root / safe_name
        trace_path = job_dir / "trace.jsonl"
        trace_analysis_dir = job_dir / "trace_analysis"
        planned_args = list(job.args)
        if model:
            planned_args = _replace_option(planned_args, "--model", model)
        planned_args = _replace_option(planned_args, "--out", str(trace_path))
        if _uses_unified_cli(planned_args):
            command_args = planned_args
        else:
            planned_args = _replace_option(planned_args, "--trace-analysis-dir", str(trace_analysis_dir))
            command_args = ["run", *planned_args]
        input_hashes = {
            "snapshot": hash_file(_option_value(planned_args, "--snapshot")),
            "setup_actions": hash_file(_option_value(planned_args, "--setup-actions")),
            "seed_file": hash_file(_option_value(planned_args, "--seeds")),
            "experience_memory_snapshot": None,
        }
        config_id = hash_json({"args": _identity_args(planned_args), "input_hashes": input_hashes})
        scenario_id = safe_identifier(job.scenario_id or base_name, fallback="scenario")
        replicate_id = safe_identifier(
            job.replicate_id or f"replicate-{seen[base_name]:03d}",
            fallback="replicate",
        )
        model_fingerprint = hash_json(
            {
                "model": _option_value(planned_args, "--model"),
                "base_url": _option_value(planned_args, "--base-url"),
            }
        )
        run_id = safe_identifier(
            f"{scenario_id}--{replicate_id}--cfg-{config_id[:12]}--model-{model_fingerprint[:12]}",
            fallback=config_id[:16],
        )
        command = [executable, "-X", "utf8", "-m", "sts2_bench", *command_args]
        plans.append(
            JobPlan(
                name=job.name,
                safe_name=safe_name,
                args=planned_args,
                command=command,
                job_dir=job_dir,
                trace_path=trace_path,
                trace_analysis_dir=trace_analysis_dir,
                stdout_path=job_dir / "stdout.txt",
                stderr_path=job_dir / "stderr.txt",
                launcher_config_path=job_dir / "launcher_config.json",
                launcher_result_path=job_dir / "launcher_result.json",
                config_id=config_id,
                scenario_id=scenario_id,
                replicate_id=replicate_id,
                model_fingerprint=model_fingerprint,
                run_id=run_id,
                input_hashes=input_hashes,
            )
        )
    return plans


def _plan_to_dict(plan: JobPlan) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": plan.name,
        "safe_name": plan.safe_name,
        "identity": {
            "config_id": plan.config_id,
            "scenario_id": plan.scenario_id,
            "replicate_id": plan.replicate_id,
            "model_fingerprint": plan.model_fingerprint,
            "run_id": plan.run_id,
            "input_hashes": plan.input_hashes,
        },
        "args": plan.args,
        "command": plan.command,
        "job_dir": str(plan.job_dir),
        "trace": str(plan.trace_path),
        "trace_analysis_dir": str(plan.trace_analysis_dir),
        "stdout": str(plan.stdout_path),
        "stderr": str(plan.stderr_path),
    }


def _subprocess_runner(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path,
    timeout_sec: float,
) -> int:
    timeout = None if timeout_sec <= 0 else timeout_sec
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            timeout=timeout,
            check=False,
        )
    return int(completed.returncode)


def _run_plan(
    plan: JobPlan,
    *,
    runner: Callable[..., int],
    cwd: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    plan.job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(plan.launcher_config_path, _plan_to_dict(plan))
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.monotonic()
    error: str | None = None
    returncode = 1
    try:
        returncode = int(
            runner(
                plan.command,
                stdout_path=plan.stdout_path,
                stderr_path=plan.stderr_path,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )
        )
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        error = f"timed out after {timeout_sec:g}s: {exc}"
        plan.stderr_path.write_text(error + "\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive guard for launcher failures
        returncode = 1
        error = f"{type(exc).__name__}: {exc}"
        plan.stderr_path.write_text(error + "\n", encoding="utf-8")
    finished_at = datetime.now().isoformat(timespec="seconds")
    result = {
        **_plan_to_dict(plan),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": round(time.monotonic() - started, 3),
        "returncode": returncode,
        "status": "succeeded" if returncode == 0 and error is None else "failed",
        "error": error,
    }
    _write_json(plan.launcher_result_path, result)
    return result


def _write_status(path: Path, *, status: str, plans: list[JobPlan], results: list[dict[str, Any]]) -> None:
    completed = {result["safe_name"]: result for result in results}
    jobs = []
    for plan in plans:
        result = completed.get(plan.safe_name)
        jobs.append(result if result is not None else {**_plan_to_dict(plan), "status": "pending"})
    _write_json(path, {"status": status, "job_count": len(plans), "completed_count": len(results), "jobs": jobs})


def run_suite(
    jobs: list[SuiteJob],
    *,
    out_root: str | Path,
    workers: int,
    model: str | None = None,
    python_executable: str | None = None,
    timeout_sec: float = 0,
    runner: Callable[..., int] = _subprocess_runner,
    cwd: str | Path = ROOT,
) -> int:
    out_root_path = Path(out_root)
    if out_root_path.exists() and any(out_root_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty suite directory: {out_root_path}")
    plans = build_job_plans(jobs, out_root=out_root_path, model=model, python_executable=python_executable)
    out_root_path.mkdir(parents=True, exist_ok=True)
    status_path = out_root_path / "suite_status.json"
    summary_path = out_root_path / "suite_summary.json"
    manifest_path = out_root_path / "suite_manifest.json"
    provenance = repo_provenance(Path(cwd))
    suite_prompt_hash = prompt_source_hash(Path(cwd))
    suite_id = hash_json([_plan_to_dict(plan)["identity"] for plan in plans])
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "suite_id": suite_id,
            "provenance": provenance,
            "prompt_sources_hash": suite_prompt_hash,
            "job_count": len(plans),
            "jobs": [_plan_to_dict(plan) for plan in plans],
        },
    )
    results: list[dict[str, Any]] = []
    _write_status(status_path, status="running", plans=plans, results=results)

    max_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_plan, plan, runner=runner, cwd=Path(cwd), timeout_sec=timeout_sec)
            for plan in plans
        ]
        for future in as_completed(futures):
            results.append(future.result())
            _write_status(status_path, status="running", plans=plans, results=results)

    results_by_safe_name = {result["safe_name"]: result for result in results}
    ordered_results = [results_by_safe_name[plan.safe_name] for plan in plans]
    failed = [result for result in ordered_results if result["status"] != "succeeded"]
    summary = {
        "schema_version": 1,
        "suite_id": suite_id,
        "provenance": provenance,
        "prompt_sources_hash": suite_prompt_hash,
        "status": "succeeded" if not failed else "failed",
        "job_count": len(ordered_results),
        "succeeded_count": len(ordered_results) - len(failed),
        "failed_count": len(failed),
        "workers": max_workers,
        "out_root": str(out_root_path),
        "jobs": ordered_results,
    }
    _write_json(summary_path, summary)
    _write_status(status_path, status=summary["status"], plans=plans, results=ordered_results)
    return 0 if not failed else 1


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"parallel_{stamp}"


def _write_dry_run_plan(
    *,
    suite: Path,
    out_root: Path,
    jobs: list[SuiteJob],
    workers: int,
    model: str | None,
    python_executable: str | None,
) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty suite directory: {out_root}")
    plans = build_job_plans(jobs, out_root=out_root, model=model, python_executable=python_executable)
    _write_json(
        out_root / "dry_run_plan.json",
        {
            "schema_version": 1,
            "suite_id": hash_json([_plan_to_dict(plan)["identity"] for plan in plans]),
            "suite": str(suite),
            "suite_sha256": hash_file(suite),
            "provenance": repo_provenance(ROOT),
            "prompt_sources_hash": prompt_source_hash(ROOT),
            "out_root": str(out_root),
            "workers": workers,
            "jobs": [_plan_to_dict(plan) for plan in plans],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run STS2 benchmark suites in parallel.")
    parser.add_argument("--suite", required=True, help="JSON suite with jobs[] or verified-pressure combats[].")
    parser.add_argument("--out-root", default=None, help="Directory for per-job traces and launcher artifacts.")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent benchmark processes.")
    parser.add_argument("--model", default=None, help="Override --model for every job in the suite.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child benchmark processes.")
    parser.add_argument("--timeout-sec", type=float, default=0, help="Per-job timeout in seconds; 0 disables timeouts.")
    parser.add_argument("--dry-run", action="store_true", help="Write dry_run_plan.json without launching jobs.")
    args = parser.parse_args(argv)

    suite_path = Path(args.suite)
    out_root = Path(args.out_root) if args.out_root else _default_out_root()
    jobs = load_suite(suite_path)
    if args.dry_run:
        _write_dry_run_plan(
            suite=suite_path,
            out_root=out_root,
            jobs=jobs,
            workers=max(1, args.workers),
            model=args.model,
            python_executable=args.python,
        )
        print(json.dumps({"dry_run_plan": str(out_root / "dry_run_plan.json")}, ensure_ascii=False), flush=True)
        return 0

    return run_suite(
        jobs,
        out_root=out_root,
        workers=args.workers,
        model=args.model,
        python_executable=args.python,
        timeout_sec=args.timeout_sec,
    )


if __name__ == "__main__":
    raise SystemExit(main())
