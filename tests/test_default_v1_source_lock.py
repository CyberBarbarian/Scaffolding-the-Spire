from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from sts2_bench.default_v1_source_lock import (
    BENCHMARK_EXECUTION_FILES,
    DEFAULT_PROTOCOL,
    GENERATED_SUITE,
    IDENTITY_FILE,
    DefaultV1SourceLockError,
    build_ablation_template_lock,
    build_generated_suite,
    create_or_verify_source_lock,
    protocol_run_id_prefix,
    protocol_scenario_id,
    resolve_protocol_execution_configuration,
    validate_ablation_switch_configuration,
    validate_protocol,
    verify_source_lock,
)


def test_ablation_switch_configuration_binds_protocol_defaults_to_cli_args():
    protocol = {
        "default_configuration": {
            "agent": "llm",
            "coordination": "single_agent",
            "context_mode": "single_turn",
            "reward_posterior_evaluator": False,
            "rule_projection_and_recent_changes": False,
        },
        "common_args": [
            "--agent",
            "llm",
            "--context-mode",
            "single_turn",
            "--no-reward-posterior-evaluator",
            "--no-rule-projection-and-recent-changes",
        ],
    }

    assert validate_ablation_switch_configuration(protocol) == {
        "agent": "llm",
        "coordination": "single_agent",
        "context_mode": "single_turn",
        "reward_posterior_evaluator": False,
        "rule_projection_and_recent_changes": False,
    }


def test_ablation_switch_configuration_rejects_cli_default_mismatch():
    protocol = {
        "default_configuration": {
            "agent": "coordinator",
            "coordination": "OCC/CA",
            "context_mode": "turn_chat",
            "reward_posterior_evaluator": True,
            "rule_projection_and_recent_changes": False,
        },
        "common_args": [
            "--agent",
            "coordinator",
            "--context-mode",
            "turn_chat",
            "--reward-posterior-evaluator",
            "--rule-projection-and-recent-changes",
        ],
    }

    with pytest.raises(DefaultV1SourceLockError, match="--no-rule-projection-and-recent-changes"):
        validate_ablation_switch_configuration(protocol)


@pytest.mark.parametrize(
    ("filename", "context_mode"),
    [
        ("ablation_combat_state_only_defect_template_20260726.json", "combat_state_only"),
        ("ablation_combat_dialogue_defect_template_20260726.json", "combat_dialogue"),
    ],
)
def test_combat_context_protocol_templates_bind_new_modes(filename, context_mode):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    protocol = json.loads(path.read_text(encoding="utf-8"))

    resolved = validate_ablation_switch_configuration(protocol)

    assert resolved["agent"] == "coordinator"
    assert resolved["context_mode"] == context_mode
    assert resolved["combat_dialogue_history_turns"] == 3
    assert resolved["architecture"] == "role_specialized"
    assert resolved["llm_fallback_mode"] == "error"
    assert resolved["combat_card_sequences"] is False
    assert resolved["game_rules_delivery"] == "state_selected_prompt"
    assert protocol["formal_experiment_started"] is False


def test_formal_generalist_protocol_locks_every_scaffolding_switch_off():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "formal_generalist_defect_template_20260726.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    resolved = validate_ablation_switch_configuration(protocol)

    assert resolved["agent"] == "formal_generalist"
    assert resolved["architecture"] == "formal_generalist"
    assert resolved["game_rules_delivery"] == "static_full_system_prompt"
    assert resolved["reward_posterior_evaluator"] is False
    assert resolved["rule_projection_and_recent_changes"] is False
    assert protocol["formal_experiment_started"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda protocol: protocol["default_configuration"].__setitem__(
                "combat_reasoning_pool", False
            ),
            "--no-combat-reasoning-pool",
        ),
        (
            lambda protocol: protocol["default_configuration"].__setitem__(
                "llm_fallback_mode", "action"
            ),
            "llm_fallback_mode",
        ),
        (
            lambda protocol: protocol["game_rules"].__setitem__(
                "ruleset_sha256", "0" * 64
            ),
            "game_rules identity",
        ),
    ],
)
def test_ablation_template_source_lock_rejects_single_sided_drift(mutator, message):
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "ablation_combat_state_only_defect_template_20260726.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    mutator(protocol)

    with pytest.raises(DefaultV1SourceLockError, match=message):
        validate_ablation_switch_configuration(protocol)


@pytest.mark.parametrize(
    "filename",
    [
        "ablation_combat_state_only_defect_template_20260726.json",
        "ablation_combat_dialogue_defect_template_20260726.json",
        "formal_generalist_defect_template_20260726.json",
    ],
)
def test_ablation_template_lock_is_zero_api_and_content_addressed(filename):
    path = Path(__file__).resolve().parents[1] / "experiments" / filename

    first = build_ablation_template_lock(path)
    second = build_ablation_template_lock(path)

    assert first == second
    assert first["provider_calls"] == 0
    assert first["formal_experiment_started"] is False
    assert len(first["identity_sha256"]) == 64
    assert any(item["path"] == "sts2_bench/agents.py" for item in first["files"])


def test_bounded_smoke_keeps_strict_switch_validation(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "ablation_combat_state_only_defect_template_20260726.json"
    )
    protocol = json.loads(source.read_text(encoding="utf-8"))
    protocol["template_status"] = "bounded_smoke"
    path = tmp_path / "bounded_smoke.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    lock = build_ablation_template_lock(path)

    assert lock["execution_mode"] == "active_worktree_bounded_smoke"
    assert lock["provider_calls"] == 0
    assert lock["resolved_ablation_configuration"]["context_mode"] == "combat_state_only"


ROOT = Path(__file__).resolve().parents[1]


def _generic_formal_protocol() -> dict:
    path = ROOT / "experiments" / "ablation_combat_state_only_defect_template_20260726.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["template_status"] = "formal_execution_ready"
    protocol["comparison"] = {
        "baseline": "formal-generalist-defect-template-20260726",
        "matched_character": "Defect",
        "matched_ascension": 0,
        "matched_seeds": [
            "ironclad_a0_3407",
            "ironclad_a0_1007",
            "ironclad_a0_2407",
        ],
        "repeats_per_seed": 5,
        "total_runs": 15,
    }
    protocol["schedule"] = {
        "workers": 3,
        "parallel_unit": "seed_lane",
        "between_seed_lanes": "parallel",
        "within_seed_lane": "strictly_serial",
        "max_attempts_per_invocation": 3,
        "run_timeout_sec": 7200,
        "resume_unit": "seed_replicate",
    }
    return protocol


def test_zero_api_template_cannot_be_started_by_formal_runner() -> None:
    path = ROOT / "experiments" / "ablation_combat_state_only_defect_template_20260726.json"

    with pytest.raises(DefaultV1SourceLockError, match="formal_execution_ready"):
        validate_protocol(path)


def test_generic_formal_protocol_reuses_template_lock_and_derives_safe_ids(
    tmp_path: Path,
) -> None:
    protocol = _generic_formal_protocol()
    path = tmp_path / "generic-formal.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    validated = validate_protocol(path)
    runtime = resolve_protocol_execution_configuration(validated)
    suite = build_generated_suite(validated)

    assert validated["_protocol_kind"] == "generic_ablation_formal"
    assert validated["_ablation_template_lock"]["resolved_ablation_configuration"][
        "context_mode"
    ] == "combat_state_only"
    assert runtime["model"] == "qwen3.6-flash"
    assert runtime["character"] == "Defect"
    assert protocol_run_id_prefix(validated) == protocol["experiment_id"]
    assert protocol_scenario_id(validated, "seed-3407") == "defect-a0-seed-3407"
    assert suite["schema_version"] == 3
    assert suite["run_id_prefix"] == protocol["experiment_id"]
    assert suite["jobs"][0]["run_id"] == (
        f"{protocol['experiment_id']}--seed-3407--replicate-01"
    )
    assert suite["jobs"][0]["scenario_id"] == "defect-a0-seed-3407"


def test_generic_formal_protocol_rejects_unsafe_run_id_prefix(tmp_path: Path) -> None:
    protocol = _generic_formal_protocol()
    protocol["experiment_id"] = "../unsafe"
    option_index = protocol["common_args"].index("--experiment-id")
    protocol["common_args"][option_index + 1] = "../unsafe"
    path = tmp_path / "generic-formal.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(DefaultV1SourceLockError, match="safe lowercase run-id"):
        validate_protocol(path)


def test_generic_formal_protocol_rejects_default_cli_alias_drift(
    tmp_path: Path,
) -> None:
    protocol = _generic_formal_protocol()
    protocol["default_configuration"]["llm_fallback_mode"] = "action"
    path = tmp_path / "generic-formal.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(DefaultV1SourceLockError, match="llm_fallback_mode"):
        validate_protocol(path)


@pytest.mark.parametrize(
    "condition",
    [
        "B1_formal_generalist",
        "B2_full_bg_agent",
        "A1_no_route_plan",
        "A2_combat_state_only",
        "A3_combat_raw_dialogue",
        "A4_no_reward_posterior",
    ],
)
def test_formal_matrix_conditions_share_generic_protocol_validator(
    tmp_path: Path,
    condition: str,
) -> None:
    filename = (
        "formal_generalist_defect_template_20260726.json"
        if condition == "B1_formal_generalist"
        else "ablation_combat_dialogue_defect_template_20260726.json"
        if condition == "A3_combat_raw_dialogue"
        else "ablation_combat_state_only_defect_template_20260726.json"
    )
    protocol = json.loads((ROOT / "experiments" / filename).read_text(encoding="utf-8"))
    protocol["template_status"] = "formal_execution_ready"
    protocol["experiment_id"] = f"formal-{condition.lower().replace('_', '-')}-20260726"
    args = protocol["common_args"]
    args[args.index("--experiment-id") + 1] = protocol["experiment_id"]
    if condition in {
        "B2_full_bg_agent",
        "A1_no_route_plan",
        "A4_no_reward_posterior",
    }:
        protocol["default_configuration"]["context_mode"] = "combat_mdp_state"
        args[args.index("--context-mode") + 1] = "combat_mdp_state"
    if condition == "A1_no_route_plan":
        protocol["default_configuration"]["route_plan"] = False
        args[args.index("--route-plan")] = "--no-route-plan"
    if condition == "A4_no_reward_posterior":
        protocol["default_configuration"]["reward_posterior_evaluator"] = False
        protocol["exposure_control"]["reward_posterior_evaluator"] = False
        args[args.index("--reward-posterior-evaluator")] = (
            "--no-reward-posterior-evaluator"
        )
    protocol["comparison"] = {
        "baseline": "matched-formal-control",
        "matched_character": "Defect",
        "matched_ascension": 0,
        "matched_seeds": [
            "ironclad_a0_3407",
            "ironclad_a0_1007",
            "ironclad_a0_2407",
        ],
        "repeats_per_seed": 5,
        "total_runs": 15,
    }
    protocol["schedule"] = {
        "workers": 3,
        "parallel_unit": "seed_lane",
        "between_seed_lanes": "parallel",
        "within_seed_lane": "strictly_serial",
        "max_attempts_per_invocation": 1,
        "run_timeout_sec": 7200,
        "resume_unit": "seed_replicate",
    }
    path = tmp_path / f"{condition}.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    validated = validate_protocol(path)

    assert validated["_protocol_kind"] == "generic_ablation_formal"
    assert validated["schedule"]["max_attempts_per_invocation"] == 1


def _set_source_lock_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "STS2_CLI_ROOT"):
        monkeypatch.delenv(name, raising=False)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _build_source_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "benchmark"
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    relative_files = [
        *BENCHMARK_EXECUTION_FILES,
        f"experiments/{DEFAULT_PROTOCOL.name}",
        "scripts/run_default_v1_rules_v2_formal.ps1",
        *(str(lane["seed_file"]) for lane in protocol["seed_lanes"]),
    ]
    for relative in relative_files:
        _copy_file(ROOT / relative, source / relative)
    (source / "experiments" / DEFAULT_PROTOCOL.name).write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )

    cli = tmp_path / "sts2-cli"
    for relative in ("python/game_log.py", "python/play.py"):
        destination = cli / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("VALUE = 1\n", encoding="utf-8")
    for relative in (
        "src/Sts2Headless/Program.cs",
        "src/Sts2Headless/RunSimulator.cs",
        "src/Sts2Headless/Sts2Headless.csproj",
        "src/GodotStubs/Core.cs",
        "src/GodotStubs/GodotStubs.csproj",
    ):
        destination = cli / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("// source\n", encoding="utf-8")
    runtime = cli / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0"
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("Sts2Headless.dll", "GodotSharp.dll"):
        (runtime / name).write_bytes(name.encode("utf-8"))
    (runtime / "Sts2Headless.runtimeconfig.json").write_text("{}", encoding="utf-8")
    library = cli / "lib"
    library.mkdir(parents=True, exist_ok=True)
    (library / "sts2.dll").write_bytes(b"game")
    for language in ("localization_eng", "localization_zhs"):
        localization = cli / language / "cards.json"
        localization.parent.mkdir(parents=True, exist_ok=True)
        localization.write_text('{"ok": true}\n', encoding="utf-8")
        (localization.parent / "_done.txt").write_text("complete\n", encoding="utf-8")
    dotnet = cli / ".tools" / "dotnet"
    (dotnet / "host" / "fxr" / "9.0.16").mkdir(parents=True, exist_ok=True)
    (dotnet / "shared" / "Microsoft.NETCore.App" / "9.0.16").mkdir(parents=True, exist_ok=True)
    (dotnet / "dotnet.exe").write_bytes(b"dotnet")
    (dotnet / "host" / "fxr" / "9.0.16" / "hostfxr.dll").write_bytes(b"host")
    (
        dotnet / "shared" / "Microsoft.NETCore.App" / "9.0.16" / "System.Private.CoreLib.dll"
    ).write_bytes(b"core")
    newest_input = max(
        path.stat().st_mtime
        for root in (cli / "src" / "Sts2Headless", cli / "src" / "GodotStubs", cli / "lib")
        for path in root.rglob("*")
        if path.is_file() and "bin" not in path.parts
    )
    for name in ("Sts2Headless.dll", "GodotSharp.dll"):
        os.utime(runtime / name, (newest_input + 5, newest_input + 5))
    return source, cli, source / "experiments" / DEFAULT_PROTOCOL.name


def test_protocol_and_generated_suite_keep_three_serial_seed_lanes() -> None:
    protocol = validate_protocol(DEFAULT_PROTOCOL)
    suite = build_generated_suite(protocol)
    assert protocol["schedule"]["workers"] == 3
    assert protocol["schedule"]["within_seed_lane"] == "strictly_serial"
    assert len(suite["jobs"]) == 15


def test_protocol_rejects_stale_direct_cli_baseline(tmp_path: Path) -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    protocol["comparison"]["baseline"] = "direct-cli-rules-v2-active-ironclad-r1-20260723"
    path = tmp_path / DEFAULT_PROTOCOL.name
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(DefaultV1SourceLockError, match="comparison.baseline"):
        validate_protocol(path)


def test_protocol_requires_two_stage_reward_evaluation(tmp_path: Path) -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    protocol["exposure_control"]["reward_posterior_evaluator"] = False
    option_index = protocol["common_args"].index("--reward-posterior-evaluator")
    protocol["common_args"][option_index] = "--no-reward-posterior-evaluator"
    path = tmp_path / DEFAULT_PROTOCOL.name
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(DefaultV1SourceLockError, match="must include --reward-posterior-evaluator"):
        validate_protocol(path)


def test_source_lock_tracks_active_files_without_copying_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_source_lock_environment(monkeypatch)
    source, cli, protocol = _build_source_fixture(tmp_path)
    lock_root = tmp_path / "results"

    created = create_or_verify_source_lock(
        lock_root,
        protocol_path=protocol,
        source_root=source,
        sts2_cli_root=cli,
    )

    identity = json.loads((lock_root / IDENTITY_FILE).read_text(encoding="utf-8"))
    labels = {item["label"] for item in identity["files"]}
    assert created["execution_mode"] == "active_worktree"
    assert any(label.startswith("sts2-cli/localization_eng/") for label in labels)
    assert any(label.startswith("sts2-cli/localization_zhs/") for label in labels)
    assert not (lock_root / "code_snapshot").exists()
    assert not (lock_root / "runtime").exists()
    assert not (lock_root / "run_frozen.ps1").exists()
    assert (lock_root / GENERATED_SUITE).is_file()


def test_generic_source_lock_binds_protocol_run_id_and_template_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_source_lock_environment(monkeypatch)
    source, cli, _legacy_protocol = _build_source_fixture(tmp_path)
    protocol = _generic_formal_protocol()
    protocol_path = source / "experiments" / "generic-formal.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    lock_root = tmp_path / "generic-results"

    created = create_or_verify_source_lock(
        lock_root,
        protocol_path=protocol_path,
        source_root=source,
        sts2_cli_root=cli,
    )
    identity = json.loads((lock_root / IDENTITY_FILE).read_text(encoding="utf-8"))
    suite = json.loads((lock_root / GENERATED_SUITE).read_text(encoding="utf-8"))
    labels = {item["label"] for item in identity["files"]}

    assert created["status"] == "verified"
    assert identity["protocol_kind"] == "generic_ablation_formal"
    assert identity["run_id_prefix"] == protocol["experiment_id"]
    assert identity["ablation_template_lock"]["identity_sha256"] == suite[
        "ablation_template_identity_sha256"
    ]
    assert "benchmark/experiments/generic-formal.json" in labels
    assert verify_source_lock(lock_root)["identity_sha256"] == identity["identity_sha256"]


def test_source_lock_detects_active_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_source_lock_environment(monkeypatch)
    source, cli, protocol = _build_source_fixture(tmp_path)
    lock_root = tmp_path / "results"
    create_or_verify_source_lock(
        lock_root,
        protocol_path=protocol,
        source_root=source,
        sts2_cli_root=cli,
    )

    (cli / "localization_eng" / "cards.json").write_text('{"drift": true}\n', encoding="utf-8")

    with pytest.raises(DefaultV1SourceLockError, match="Locked active source file drifted"):
        verify_source_lock(lock_root)


def test_source_lock_rejects_legacy_snapshot_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_source_lock_environment(monkeypatch)
    lock_root = tmp_path / "legacy"
    lock_root.mkdir()
    (lock_root / IDENTITY_FILE).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "execution_mode": "isolated_snapshot",
                "code_snapshot_created": True,
                "files": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DefaultV1SourceLockError, match="active-worktree"):
        verify_source_lock(lock_root)


def test_source_lock_refuses_ambient_cli_redirect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_source_lock_environment(monkeypatch)
    source, cli, protocol = _build_source_fixture(tmp_path)
    monkeypatch.setenv("STS2_CLI_ROOT", str(tmp_path / "other-cli"))

    with pytest.raises(DefaultV1SourceLockError, match="may not redirect"):
        create_or_verify_source_lock(
            tmp_path / "results",
            protocol_path=protocol,
            source_root=source,
            sts2_cli_root=cli,
        )


def test_source_lock_contains_no_credential_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_source_lock_environment(monkeypatch)
    source, cli, protocol = _build_source_fixture(tmp_path)
    (source / ".env").write_text("DEEPSEEK_API_KEY=secret\n", encoding="utf-8")
    lock_root = tmp_path / "results"
    create_or_verify_source_lock(
        lock_root,
        protocol_path=protocol,
        source_root=source,
        sts2_cli_root=cli,
    )
    text = (lock_root / IDENTITY_FILE).read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY" not in text
    assert "secret" not in text
    assert os.path.abspath(source / ".env") not in text
