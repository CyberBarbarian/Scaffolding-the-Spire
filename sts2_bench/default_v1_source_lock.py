"""Create and verify the Default-v1 active-worktree source lock."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .experiment_identity import hash_file, hash_json
from .game_rules import game_rules_identity
from .seeds import read_seed_file


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILENAME = "default_v1_rules_v2_posterior_r10_formal_20260725.json"
EXPECTED_EXPERIMENT_ID = "default-v1-rules-v2-posterior-r10-ironclad-20260725"
EXPECTED_DIRECT_CLI_BASELINE_ID = "direct-cli-rules-v2-active-ironclad-r5-20260725"
LEGACY_RUN_ID_PREFIX = "default-v1-rules-v2"
FORMAL_EXECUTION_READY = "formal_execution_ready"
DEFAULT_PROTOCOL = ROOT / "experiments" / PROTOCOL_FILENAME
GENERATED_SUITE = Path("experiments/default_v1_rules_v2_generated_suite.json")
IDENTITY_FILE = Path("source_lock.json")
VERIFICATION_FILE = Path("source_lock_verification.json")
DOTNET_RUNTIME_VERSION = "9.0.16"
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
STS2_CLI_LABEL_ROOT = Path("sts2-cli")
BENCHMARK_EXECUTION_FILES = (
    "sts2_bench/__init__.py",
    "sts2_bench/__main__.py",
    "sts2_bench/actions.py",
    "sts2_bench/agents.py",
    "sts2_bench/artifact_lease.py",
    "sts2_bench/benchmark.yaml",
    "sts2_bench/context.py",
    "sts2_bench/default_v1_experiment.py",
    "sts2_bench/default_v1_source_lock.py",
    "sts2_bench/evidence.py",
    "sts2_bench/experiment_identity.py",
    "sts2_bench/feedback.py",
    "sts2_bench/game_rules.py",
    "sts2_bench/localization_integrity.py",
    "sts2_bench/multi_agent.py",
    "sts2_bench/posterior.py",
    "sts2_bench/process.py",
    "sts2_bench/route_candidates.py",
    "sts2_bench/route_graph.py",
    "sts2_bench/route_plan.py",
    "sts2_bench/routing.py",
    "sts2_bench/run_benchmark.py",
    "sts2_bench/run_context.py",
    "sts2_bench/runner.py",
    "sts2_bench/seeds.py",
    "sts2_bench/sequence_safety.py",
    "sts2_bench/sequence_validation.py",
    "sts2_bench/sequences.py",
    "sts2_bench/trace_analysis.py",
)
FORBIDDEN_BENCHMARK_SOURCE_FILES = {
    "sts2_bench/direct_cli_baseline.py",
    "sts2_bench/direct_cli_experiment.py",
    "sts2_bench/direct_cli_report.py",
    "sts2_bench/full_reflection_pairs.py",
    "sts2_bench/gated_memory_experiment.py",
    "sts2_bench/layered_experience.py",
    "sts2_bench/minimal_gated_experience.py",
    "sts2_bench/reflection_candidate_r1.py",
    "sts2_bench/reflection_candidates.py",
    "sts2_bench/reflection_execution.py",
    "sts2_bench/reflection_extension.py",
    "sts2_bench/reflection_five_block_report.py",
    "sts2_bench/reflection_pairs.py",
    "sts2_bench/reflection_probe.py",
    "sts2_bench/reflection_r1.py",
    "sts2_bench/reflection_state_fork.py",
    "sts2_bench/reflection_triads.py",
    "sts2_bench/segmented_experience_extract.py",
    "sts2_bench/sequential_online_memory.py",
}
SENSITIVE_FILENAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "secrets",
    "secrets.json",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
DEPENDENCY_DISTRIBUTIONS = (
    "omegaconf",
    "requests",
    "antlr4-python3-runtime",
    "PyYAML",
    "certifi",
    "charset-normalizer",
    "idna",
    "urllib3",
)
SANITIZED_PYTHON_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUTF8": "1",
}
FORBIDDEN_BASE_PROMPT_FRAGMENTS = {
    "do not choose it while useful playable cards or potions remain",
    "do not add enemy Strength or other visible modifiers",
    "do not apply those visible modifiers to shown card damage or block",
    "avoid spending energy only for block",
}
SAFE_ID_COMPONENT = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")


class DefaultV1SourceLockError(RuntimeError):
    """Raised when the formal Default-v1 source lock cannot be trusted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DefaultV1SourceLockError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


VALUE_OPTIONS = {
    "--agent", "--base-url", "--model", "--llm-timeout", "--llm-max-retries",
    "--llm-fallback-mode", "--llm-temperature", "--prompt-style",
    "--short-term-memory-window", "--short-term-memory-mode", "--context-mode",
    "--turn-chat-window", "--turn-chat-update-mode", "--turn-chat-assistant-history",
    "--combat-dialogue-history-turns",
    "--route-plan-enforcement", "--route-map-prompt-format", "--route-candidate-objective",
    "--map-route-hint", "--occ-context-mode", "--self-consistency-samples",
    "--self-consistency-decisions", "--self-consistency-mode", "--action-prompt-order",
    "--action-prompt-order-seed", "--character", "--ascension", "--lang", "--max-steps",
    "--max-game-actions", "--experiment-id", "--log-level", "--custom-sequence-exposure",
    "--combat-skill-hints-profile", "--combat-skill-hints-max-hints",
}
FLAG_OPTIONS = {
    "--no-include-json-state", "--no-short-term-memory", "--no-run-summary", "--no-turn-check",
    "--route-plan", "--no-route-plan",
    "--reward-posterior-evaluator", "--no-reward-posterior-evaluator",
    "--no-self-consistency",
    "--include-full-map", "--no-allow-repeat-views", "--no-allow-custom-sequences",
    "--no-combat-card-sequences", "--no-precomputed-combat-card-sequences",
    "--combat-reasoning-pool", "--no-combat-reasoning-pool",
    "--combat-turn-board", "--no-combat-turn-board", "--no-combat-skill-hints",
    "--no-combat-skill-lines", "--no-print-prompts", "--no-print-model-output", "--trace-analysis",
    "--rule-projection-and-recent-changes", "--no-rule-projection-and-recent-changes",
}
FORBIDDEN_OPTIONS = {
    "--setup-actions", "--seeds", "--count", "--scenario-id", "--replicate-id", "--run-id",
    "--out", "--trace-analysis-dir", "--api-key", "--overwrite",
}


def _parse_common_args(args: list[str]) -> dict[str, str | bool]:
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(args):
        token = args[index]
        option, separator, inline_value = token.partition("=")
        if not option.startswith("--"):
            raise DefaultV1SourceLockError(f"Unexpected positional common_arg: {token}")
        if option in FORBIDDEN_OPTIONS:
            raise DefaultV1SourceLockError(f"Forbidden or lane-specific option in common_args: {option}")
        if option in parsed:
            raise DefaultV1SourceLockError(f"Duplicate common_args option: {option}")
        if option in VALUE_OPTIONS:
            if separator:
                value = inline_value
            else:
                index += 1
                if index >= len(args) or args[index].startswith("--"):
                    raise DefaultV1SourceLockError(f"Missing value for common_args option: {option}")
                value = args[index]
            parsed[option] = value
        elif option in FLAG_OPTIONS:
            if separator:
                raise DefaultV1SourceLockError(f"Boolean flag cannot use an inline value: {option}")
            parsed[option] = True
        else:
            raise DefaultV1SourceLockError(f"Unexpected Default-v1 common_args option: {option}")
        index += 1
    for option in parsed:
        opposite = f"--no-{option[2:]}" if not option.startswith("--no-") else f"--{option[5:]}"
        if opposite in parsed:
            raise DefaultV1SourceLockError(f"Conflicting common_args flags: {option} and {opposite}")
    return parsed


def _required_arg_value(
    parsed: dict[str, str | bool],
    option: str,
) -> str:
    value = parsed.get(option)
    if not isinstance(value, str) or not value:
        raise DefaultV1SourceLockError(f"Formal protocol common_args must define {option}")
    return value


def _required_flag_value(
    parsed: dict[str, str | bool],
    *,
    enabled: str,
    disabled: str,
) -> bool:
    if parsed.get(enabled) is True:
        return True
    if parsed.get(disabled) is True:
        return False
    raise DefaultV1SourceLockError(
        f"Formal protocol common_args must define exactly one of {enabled} or {disabled}"
    )


def _as_int(value: str, *, option: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise DefaultV1SourceLockError(f"{option} must be an integer") from exc


def _as_float(value: str, *, option: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise DefaultV1SourceLockError(f"{option} must be numeric") from exc


def resolve_protocol_execution_configuration(protocol: dict[str, Any]) -> dict[str, Any]:
    """Resolve the audited runtime contract from executable common_args.

    The protocol defaults remain a human-readable design record. Runtime values
    come from the actual CLI arguments, and explicit legacy/template aliases are
    checked below so the two representations cannot silently diverge.
    """

    common_args = protocol.get("common_args")
    defaults = protocol.get("default_configuration")
    if not isinstance(common_args, list) or not all(isinstance(item, str) for item in common_args):
        raise DefaultV1SourceLockError("Protocol common_args must be a list of strings")
    if not isinstance(defaults, dict):
        raise DefaultV1SourceLockError("Protocol default_configuration must be an object")
    parsed = _parse_common_args(common_args)
    resolved = {
        "agent": _required_arg_value(parsed, "--agent"),
        "base_url": _required_arg_value(parsed, "--base-url"),
        "model": _required_arg_value(parsed, "--model"),
        "timeout_sec": _as_float(
            _required_arg_value(parsed, "--llm-timeout"),
            option="--llm-timeout",
        ),
        "retries": _as_int(
            _required_arg_value(parsed, "--llm-max-retries"),
            option="--llm-max-retries",
        ),
        "fallback": _required_arg_value(parsed, "--llm-fallback-mode"),
        "temperature": _as_float(
            _required_arg_value(parsed, "--llm-temperature"),
            option="--llm-temperature",
        ),
        "prompt_style": _required_arg_value(parsed, "--prompt-style"),
        "include_json_state": False,
        "short_term_memory": _required_flag_value(
            parsed,
            enabled="--short-term-memory",
            disabled="--no-short-term-memory",
        ),
        "short_term_memory_window": _as_int(
            _required_arg_value(parsed, "--short-term-memory-window"),
            option="--short-term-memory-window",
        ),
        "short_term_memory_mode": _required_arg_value(parsed, "--short-term-memory-mode"),
        "run_summary": _required_flag_value(
            parsed,
            enabled="--run-summary",
            disabled="--no-run-summary",
        ),
        "context_mode": _required_arg_value(parsed, "--context-mode"),
        "combat_dialogue_history_turns": _as_int(
            _required_arg_value(parsed, "--combat-dialogue-history-turns"),
            option="--combat-dialogue-history-turns",
        )
        if "--combat-dialogue-history-turns" in parsed
        else 3,
        "turn_chat_window": _as_int(
            _required_arg_value(parsed, "--turn-chat-window"),
            option="--turn-chat-window",
        ),
        "turn_chat_update_mode": _required_arg_value(parsed, "--turn-chat-update-mode"),
        "turn_chat_assistant_history": _required_arg_value(
            parsed, "--turn-chat-assistant-history"
        ),
        "turn_check": _required_flag_value(
            parsed,
            enabled="--turn-check",
            disabled="--no-turn-check",
        ),
        "route_plan": _required_flag_value(
            parsed,
            enabled="--route-plan",
            disabled="--no-route-plan",
        ),
        "route_plan_enforcement": _required_arg_value(parsed, "--route-plan-enforcement"),
        "route_map_prompt_format": _required_arg_value(parsed, "--route-map-prompt-format"),
        "reward_posterior_evaluator": _required_flag_value(
            parsed,
            enabled="--reward-posterior-evaluator",
            disabled="--no-reward-posterior-evaluator",
        ),
        "route_candidate_objective": _required_arg_value(
            parsed, "--route-candidate-objective"
        ),
        "map_route_hint": _required_arg_value(parsed, "--map-route-hint"),
        "occ_context_mode": _required_arg_value(parsed, "--occ-context-mode"),
        "self_consistency": _required_flag_value(
            parsed,
            enabled="--self-consistency",
            disabled="--no-self-consistency",
        ),
        "self_consistency_samples": _as_int(
            _required_arg_value(parsed, "--self-consistency-samples"),
            option="--self-consistency-samples",
        ),
        "self_consistency_decisions": _required_arg_value(
            parsed, "--self-consistency-decisions"
        ),
        "self_consistency_mode": _required_arg_value(parsed, "--self-consistency-mode"),
        "action_prompt_order": _required_arg_value(parsed, "--action-prompt-order"),
        "action_prompt_order_seed": _as_int(
            _required_arg_value(parsed, "--action-prompt-order-seed"),
            option="--action-prompt-order-seed",
        ),
        "character": _required_arg_value(parsed, "--character"),
        "ascension": _as_int(
            _required_arg_value(parsed, "--ascension"),
            option="--ascension",
        ),
        "language": _required_arg_value(parsed, "--lang"),
        "max_steps": _as_int(
            _required_arg_value(parsed, "--max-steps"),
            option="--max-steps",
        ),
        "max_game_actions": _as_int(
            _required_arg_value(parsed, "--max-game-actions"),
            option="--max-game-actions",
        ),
        "experiment_id": _required_arg_value(parsed, "--experiment-id"),
        "log_level": _required_arg_value(parsed, "--log-level"),
        "include_full_map": parsed.get("--include-full-map") is True,
        "allow_repeat_views": not (parsed.get("--no-allow-repeat-views") is True),
        "allow_custom_sequences": not (parsed.get("--no-allow-custom-sequences") is True),
        "custom_sequence_exposure": _required_arg_value(
            parsed, "--custom-sequence-exposure"
        ),
        "combat_card_sequences": not (
            parsed.get("--no-combat-card-sequences") is True
        ),
        "precomputed_combat_card_sequences": not (
            parsed.get("--no-precomputed-combat-card-sequences") is True
        ),
        "combat_reasoning_pool": parsed.get("--combat-reasoning-pool") is True,
        "combat_turn_board": parsed.get("--combat-turn-board") is True,
        "combat_skill_hints": not (parsed.get("--no-combat-skill-hints") is True),
        "combat_skill_hints_profile": _required_arg_value(
            parsed, "--combat-skill-hints-profile"
        ),
        "combat_skill_hints_max_hints": _as_int(
            _required_arg_value(parsed, "--combat-skill-hints-max-hints"),
            option="--combat-skill-hints-max-hints",
        ),
        "combat_skill_lines": not (parsed.get("--no-combat-skill-lines") is True),
        "rule_projection_and_recent_changes": _required_flag_value(
            parsed,
            enabled="--rule-projection-and-recent-changes",
            disabled="--no-rule-projection-and-recent-changes",
        )
        if (
            "--rule-projection-and-recent-changes" in parsed
            or "--no-rule-projection-and-recent-changes" in parsed
        )
        else True,
        "print_prompts": not (parsed.get("--no-print-prompts") is True),
        "print_model_output": not (parsed.get("--no-print-model-output") is True),
        "trace_analysis": parsed.get("--trace-analysis") is True,
    }
    aliases = (
        ("agent", "agent", str),
        ("base_url", "base_url", str),
        ("model", "model", str),
        ("timeout_sec", "request_timeout_sec", float),
        ("retries", "request_retries", int),
        ("fallback", "fallback_after_retries", str),
        ("fallback", "llm_fallback_mode", str),
        ("temperature", "temperature", float),
        ("prompt_style", "prompt_style", str),
        ("context_mode", "context_mode", str),
        ("combat_dialogue_history_turns", "combat_dialogue_history_turns", int),
        ("character", "character", str),
        ("ascension", "ascension", int),
        ("language", "language", str),
        ("max_steps", "max_decision_steps", int),
        ("max_game_actions", "max_game_actions", int),
        ("include_full_map", "include_full_map", bool),
        ("allow_repeat_views", "allow_repeat_views", bool),
        ("allow_custom_sequences", "allow_custom_sequences", bool),
        ("combat_card_sequences", "combat_card_sequences", bool),
        (
            "precomputed_combat_card_sequences",
            "precomputed_combat_card_sequences",
            bool,
        ),
        ("combat_reasoning_pool", "combat_reasoning_pool", bool),
        ("combat_turn_board", "combat_turn_board", bool),
        ("combat_skill_hints", "combat_skill_hints", bool),
        ("combat_skill_lines", "combat_skill_lines", bool),
        ("reward_posterior_evaluator", "reward_posterior_evaluator", bool),
        (
            "rule_projection_and_recent_changes",
            "rule_projection_and_recent_changes",
            bool,
        ),
        ("route_plan", "route_plan", bool),
        ("self_consistency", "self_consistency", bool),
        ("short_term_memory", "short_term_memory", bool),
        ("run_summary", "run_summary", bool),
        ("turn_check", "turn_check", bool),
        ("log_level", "log_level", str),
    )
    for resolved_key, default_key, converter in aliases:
        if default_key not in defaults:
            continue
        try:
            default_value = converter(defaults[default_key])
        except (TypeError, ValueError) as exc:
            raise DefaultV1SourceLockError(
                f"default_configuration.{default_key} has an invalid value"
            ) from exc
        if default_value != resolved[resolved_key]:
            raise DefaultV1SourceLockError(
                f"default_configuration.{default_key} and common_args differ"
            )
    if resolved["experiment_id"] != protocol.get("experiment_id"):
        raise DefaultV1SourceLockError("experiment_id and --experiment-id differ")
    return resolved


def protocol_run_id_prefix(protocol: dict[str, Any]) -> str:
    """Return the source-locked safe run-id prefix for this protocol."""

    locked = protocol.get("_run_id_prefix")
    if isinstance(locked, str):
        return locked
    if protocol.get("template_status") != FORMAL_EXECUTION_READY:
        return LEGACY_RUN_ID_PREFIX
    experiment_id = str(protocol.get("experiment_id") or "")
    if SAFE_ID_COMPONENT.fullmatch(experiment_id) is None:
        raise DefaultV1SourceLockError(
            "Generic formal experiment_id must be a safe lowercase run-id component"
        )
    return experiment_id


def protocol_scenario_id(protocol: dict[str, Any], lane_name: str) -> str:
    """Derive a safe scenario id without hard-coding character or ascension."""

    if SAFE_ID_COMPONENT.fullmatch(lane_name) is None:
        raise DefaultV1SourceLockError("Seed lane name is not a safe identifier")
    if (
        protocol.get("_protocol_kind") != "generic_ablation_formal"
        and protocol.get("template_status") != FORMAL_EXECUTION_READY
    ):
        return f"ironclad-a0-{lane_name}"
    runtime = protocol.get("_resolved_execution_configuration")
    if not isinstance(runtime, dict):
        runtime = resolve_protocol_execution_configuration(protocol)
    character = str(runtime["character"]).strip().lower()
    ascension = int(runtime["ascension"])
    if SAFE_ID_COMPONENT.fullmatch(character) is None or ascension < 0:
        raise DefaultV1SourceLockError("Character/ascension cannot form a safe scenario id")
    return f"{character}-a{ascension}-{lane_name}"


def _require_flag(args: dict[str, str | bool], flag: str) -> None:
    if args.get(flag) is not True:
        raise DefaultV1SourceLockError(f"Default-v1 common_args must include {flag}")


def _bind_ablation_boolean(
    defaults: dict[str, Any],
    parsed: dict[str, str | bool],
    *,
    key: str,
    enabled_flag: str,
    disabled_flag: str,
) -> bool:
    value = defaults.get(key)
    if not isinstance(value, bool):
        raise DefaultV1SourceLockError(
            f"default_configuration.{key} must be boolean"
        )
    _require_flag(parsed, enabled_flag if value else disabled_flag)
    return value


def validate_ablation_switch_configuration(protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate the four reusable ablation controls without weakening Default-v1.

    Character/seed schedules and the exact single-factor comparison remain the
    responsibility of a treatment-specific adapter. This helper keeps the
    protocol ``default_configuration`` and executable ``common_args`` as one
    source of truth for the shared switches.
    """

    defaults = protocol.get("default_configuration")
    common_args = protocol.get("common_args")
    if not isinstance(defaults, dict):
        raise DefaultV1SourceLockError("Ablation protocol must define default_configuration")
    if not isinstance(common_args, list) or not all(isinstance(item, str) for item in common_args):
        raise DefaultV1SourceLockError("Ablation protocol common_args must be a list of strings")
    parsed = _parse_common_args(common_args)

    agent = defaults.get("agent")
    if agent not in {"llm", "coordinator", "formal_generalist"}:
        raise DefaultV1SourceLockError(
            "Ablation agent must be llm, coordinator, or formal_generalist"
        )
    expected_coordination = (
        "formal_generalist"
        if agent == "formal_generalist"
        else "single_agent"
        if agent == "llm"
        else "OCC/CA"
    )
    if defaults.get("coordination") != expected_coordination:
        raise DefaultV1SourceLockError(
            f"default_configuration.coordination must be {expected_coordination!r}"
        )
    if parsed.get("--agent") != agent:
        raise DefaultV1SourceLockError("default_configuration.agent and --agent differ")

    context_mode = defaults.get("context_mode")
    if context_mode not in {
        "single_turn",
        "turn_chat",
        "combat_mdp_state",
        "combat_state_only",
        "combat_dialogue",
    }:
        raise DefaultV1SourceLockError("Unsupported ablation context_mode")
    if parsed.get("--context-mode") != context_mode:
        raise DefaultV1SourceLockError("default_configuration.context_mode and --context-mode differ")

    posterior_enabled = defaults.get("reward_posterior_evaluator")
    if not isinstance(posterior_enabled, bool):
        raise DefaultV1SourceLockError(
            "default_configuration.reward_posterior_evaluator must be boolean"
        )
    expected_posterior_flag = (
        "--reward-posterior-evaluator"
        if posterior_enabled
        else "--no-reward-posterior-evaluator"
    )
    _require_flag(parsed, expected_posterior_flag)

    paired_enabled = defaults.get("rule_projection_and_recent_changes")
    if not isinstance(paired_enabled, bool):
        raise DefaultV1SourceLockError(
            "default_configuration.rule_projection_and_recent_changes must be boolean"
        )
    expected_paired_flag = (
        "--rule-projection-and-recent-changes"
        if paired_enabled
        else "--no-rule-projection-and-recent-changes"
    )
    _require_flag(parsed, expected_paired_flag)

    result = {
        "agent": agent,
        "coordination": expected_coordination,
        "context_mode": context_mode,
        "reward_posterior_evaluator": posterior_enabled,
        "rule_projection_and_recent_changes": paired_enabled,
    }
    strict_template = bool(
        protocol.get("schema_version") == 3
        and protocol.get("template_status")
        in {"locked_zero_api_only", "bounded_smoke", FORMAL_EXECUTION_READY}
    )
    if strict_template:
        architecture = defaults.get("architecture")
        expected_architecture = (
            "formal_generalist" if agent == "formal_generalist" else "role_specialized"
        )
        if architecture != expected_architecture:
            raise DefaultV1SourceLockError(
                f"default_configuration.architecture must be {expected_architecture!r}"
            )
        if parsed.get("--llm-fallback-mode") != defaults.get("llm_fallback_mode"):
            raise DefaultV1SourceLockError(
                "default_configuration.llm_fallback_mode and common_args differ"
            )
        if parsed.get("--prompt-style") != defaults.get("prompt_style"):
            raise DefaultV1SourceLockError(
                "default_configuration.prompt_style and common_args differ"
            )
        bound_switches = {
            "short_term_memory": _bind_ablation_boolean(
                defaults,
                parsed,
                key="short_term_memory",
                enabled_flag="--short-term-memory",
                disabled_flag="--no-short-term-memory",
            ),
            "run_summary": _bind_ablation_boolean(
                defaults,
                parsed,
                key="run_summary",
                enabled_flag="--run-summary",
                disabled_flag="--no-run-summary",
            ),
            "turn_check": _bind_ablation_boolean(
                defaults,
                parsed,
                key="turn_check",
                enabled_flag="--turn-check",
                disabled_flag="--no-turn-check",
            ),
            "route_plan": _bind_ablation_boolean(
                defaults,
                parsed,
                key="route_plan",
                enabled_flag="--route-plan",
                disabled_flag="--no-route-plan",
            ),
            "self_consistency": _bind_ablation_boolean(
                defaults,
                parsed,
                key="self_consistency",
                enabled_flag="--self-consistency",
                disabled_flag="--no-self-consistency",
            ),
            "allow_custom_sequences": _bind_ablation_boolean(
                defaults,
                parsed,
                key="allow_custom_sequences",
                enabled_flag="--allow-custom-sequences",
                disabled_flag="--no-allow-custom-sequences",
            ),
            "combat_card_sequences": _bind_ablation_boolean(
                defaults,
                parsed,
                key="combat_card_sequences",
                enabled_flag="--combat-card-sequences",
                disabled_flag="--no-combat-card-sequences",
            ),
            "precomputed_combat_card_sequences": _bind_ablation_boolean(
                defaults,
                parsed,
                key="precomputed_combat_card_sequences",
                enabled_flag="--precomputed-combat-card-sequences",
                disabled_flag="--no-precomputed-combat-card-sequences",
            ),
            "combat_reasoning_pool": _bind_ablation_boolean(
                defaults,
                parsed,
                key="combat_reasoning_pool",
                enabled_flag="--combat-reasoning-pool",
                disabled_flag="--no-combat-reasoning-pool",
            ),
            "combat_turn_board": _bind_ablation_boolean(
                defaults,
                parsed,
                key="combat_turn_board",
                enabled_flag="--combat-turn-board",
                disabled_flag="--no-combat-turn-board",
            ),
            "combat_skill_hints": _bind_ablation_boolean(
                defaults,
                parsed,
                key="combat_skill_hints",
                enabled_flag="--combat-skill-hints",
                disabled_flag="--no-combat-skill-hints",
            ),
            "combat_skill_lines": _bind_ablation_boolean(
                defaults,
                parsed,
                key="combat_skill_lines",
                enabled_flag="--combat-skill-lines",
                disabled_flag="--no-combat-skill-lines",
            ),
            "include_full_map": _bind_ablation_boolean(
                defaults,
                parsed,
                key="include_full_map",
                enabled_flag="--include-full-map",
                disabled_flag="--no-include-full-map",
            ),
        }
        rules = protocol.get("game_rules")
        identity = game_rules_identity()
        if not isinstance(rules, dict):
            raise DefaultV1SourceLockError("Ablation protocol must bind game_rules")
        if (
            rules.get("ruleset_id") != identity["ruleset_id"]
            or rules.get("ruleset_sha256") != identity["ruleset_sha256"]
        ):
            raise DefaultV1SourceLockError(
                "Ablation protocol game_rules identity does not match the active rules-v2 source"
            )
        delivery = defaults.get("game_rules_delivery")
        if rules.get("delivery") != delivery:
            raise DefaultV1SourceLockError(
                "default_configuration.game_rules_delivery and game_rules.delivery differ"
            )
        result.update(
            {
                "architecture": architecture,
                "llm_fallback_mode": defaults.get("llm_fallback_mode"),
                "prompt_style": defaults.get("prompt_style"),
                "game_rules_delivery": delivery,
                "game_ruleset_id": identity["ruleset_id"],
                "game_ruleset_sha256": identity["ruleset_sha256"],
                **bound_switches,
            }
        )
    if context_mode == "combat_dialogue" or strict_template:
        history_turns = defaults.get("combat_dialogue_history_turns")
        if isinstance(history_turns, bool) or not isinstance(history_turns, int) or history_turns < 0:
            raise DefaultV1SourceLockError(
                "combat_dialogue_history_turns must be a non-negative integer"
            )
        if parsed.get("--combat-dialogue-history-turns") != str(history_turns):
            raise DefaultV1SourceLockError(
                "default_configuration.combat_dialogue_history_turns and common_args differ"
            )
        result["combat_dialogue_history_turns"] = history_turns
    if agent == "formal_generalist":
        if context_mode != "combat_state_only":
            raise DefaultV1SourceLockError(
                "formal_generalist must use combat_state_only"
            )
        if defaults.get("architecture") != "formal_generalist":
            raise DefaultV1SourceLockError(
                "formal_generalist architecture must be explicit"
            )
        if defaults.get("game_rules_delivery") != "static_full_system_prompt":
            raise DefaultV1SourceLockError(
                "formal_generalist must use static_full_system_prompt"
            )
        for flag in (
            "--no-short-term-memory",
            "--no-run-summary",
            "--no-turn-check",
            "--no-route-plan",
            "--no-reward-posterior-evaluator",
            "--no-self-consistency",
            "--no-combat-reasoning-pool",
            "--no-combat-turn-board",
            "--no-combat-skill-hints",
            "--no-combat-skill-lines",
            "--no-rule-projection-and-recent-changes",
            "--include-full-map",
        ):
            _require_flag(parsed, flag)
        result["architecture"] = "formal_generalist"
        result["game_rules_delivery"] = "static_full_system_prompt"
    return result


def build_ablation_template_lock(
    protocol_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Build a read-only identity for a prepared ablation protocol without running it."""

    project_root = Path(root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = _read_object(protocol_file)
    resolved = validate_ablation_switch_configuration(protocol)
    files = []
    for relative in BENCHMARK_EXECUTION_FILES:
        path = project_root / relative
        if not path.is_file():
            raise DefaultV1SourceLockError(
                f"Ablation source-lock file is missing: {relative}"
            )
        files.append(
            {
                "path": relative,
                "sha256": hash_file(path),
            }
        )
    template_status = str(protocol.get("template_status") or "")
    payload = {
        "schema_version": 1,
        "execution_mode": (
            "active_worktree_bounded_smoke"
            if template_status == "bounded_smoke"
            else "active_worktree_formal_execution_ready"
            if template_status == FORMAL_EXECUTION_READY
            else "active_worktree_zero_api_template"
        ),
        "protocol_path": str(protocol_file),
        "protocol_sha256": hash_file(protocol_file),
        "resolved_ablation_configuration": resolved,
        "game_rules": game_rules_identity(),
        "files": files,
    }
    return {
        **payload,
        "identity_sha256": hash_json(payload),
        "provider_calls": 0,
        "formal_experiment_started": False,
    }


def _validate_generic_formal_protocol(
    protocol_file: Path,
    *,
    project_root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if protocol.get("template_status") != FORMAL_EXECUTION_READY:
        raise DefaultV1SourceLockError(
            "Only template_status='formal_execution_ready' generic protocols may use the formal runner"
        )
    if protocol.get("schema_version") != 3:
        raise DefaultV1SourceLockError("Generic formal protocols must use schema_version 3")
    if protocol.get("formal_experiment_started") is not False:
        raise DefaultV1SourceLockError(
            "A generic formal protocol must be locked before its first execution"
        )

    template_lock = build_ablation_template_lock(protocol_file, root=project_root)
    resolved = resolve_protocol_execution_configuration(protocol)
    comparison = protocol.get("comparison")
    schedule = protocol.get("schedule")
    exposure = protocol.get("exposure_control")
    lanes = protocol.get("seed_lanes")
    if not isinstance(comparison, dict) or not isinstance(schedule, dict):
        raise DefaultV1SourceLockError(
            "Generic formal protocol must define comparison and schedule objects"
        )
    if not isinstance(exposure, dict):
        raise DefaultV1SourceLockError(
            "Generic formal protocol must define exposure_control"
        )
    if not isinstance(lanes, list) or len(lanes) != 3:
        raise DefaultV1SourceLockError(
            "Generic formal protocol must define exactly three seed lanes"
        )
    expected_schedule = {
        "workers": 3,
        "parallel_unit": "seed_lane",
        "between_seed_lanes": "parallel",
        "within_seed_lane": "strictly_serial",
        "resume_unit": "seed_replicate",
    }
    for key, expected in expected_schedule.items():
        if schedule.get(key) != expected:
            raise DefaultV1SourceLockError(
                f"Generic formal schedule.{key} must be {expected!r}"
            )
    if int(schedule.get("max_attempts_per_invocation") or 0) < 1:
        raise DefaultV1SourceLockError(
            "max_attempts_per_invocation must be at least 1"
        )
    if float(schedule.get("run_timeout_sec") or 0) <= 0:
        raise DefaultV1SourceLockError(
            "run_timeout_sec must be a positive hard watchdog"
        )

    repeats = int(comparison.get("repeats_per_seed") or 0)
    if repeats < 1:
        raise DefaultV1SourceLockError("repeats_per_seed must be positive")
    if int(comparison.get("total_runs") or 0) != len(lanes) * repeats:
        raise DefaultV1SourceLockError(
            "comparison.total_runs must equal three lanes times repeats_per_seed"
        )
    if comparison.get("matched_character") != resolved["character"]:
        raise DefaultV1SourceLockError(
            "comparison.matched_character and executable character differ"
        )
    if int(comparison.get("matched_ascension", -1)) != resolved["ascension"]:
        raise DefaultV1SourceLockError(
            "comparison.matched_ascension and executable ascension differ"
        )
    if not isinstance(comparison.get("baseline"), str) or not comparison["baseline"].strip():
        raise DefaultV1SourceLockError(
            "Generic formal comparison.baseline must be explicit"
        )

    for key in (
        "reflection",
        "cross_run_experience_memory",
        "human_authored_combat_skill_hints",
        "self_consistency",
    ):
        if exposure.get(key) is not False:
            raise DefaultV1SourceLockError(
                f"Generic formal exposure control {key} must be false"
            )
    if exposure.get("reward_posterior_evaluator") is not resolved[
        "reward_posterior_evaluator"
    ]:
        raise DefaultV1SourceLockError(
            "Exposure control reward_posterior_evaluator and common_args differ"
        )
    sample_contract = protocol.get("formal_sample_contract")
    if not isinstance(sample_contract, dict) or sample_contract.get(
        "natural_game_over"
    ) != "completed_game":
        raise DefaultV1SourceLockError(
            "Generic formal sample contract must define natural_game_over='completed_game'"
        )

    parsed = _parse_common_args(protocol["common_args"])
    for flag in (
        "--no-include-json-state",
        "--no-print-prompts",
        "--no-print-model-output",
        "--no-allow-repeat-views",
        "--trace-analysis",
    ):
        _require_flag(parsed, flag)
    if resolved["fallback"] != "error":
        raise DefaultV1SourceLockError(
            "Generic formal execution requires llm_fallback_mode='error'"
        )
    if resolved["allow_custom_sequences"]:
        raise DefaultV1SourceLockError(
            "Generic formal execution must disable custom sequences"
        )
    if resolved["combat_card_sequences"]:
        raise DefaultV1SourceLockError(
            "Generic formal execution must disable combat card sequences"
        )
    if resolved["precomputed_combat_card_sequences"]:
        raise DefaultV1SourceLockError(
            "Generic formal execution must disable precomputed combat card sequences"
        )

    observed_seeds: list[str] = []
    observed_names: list[str] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            raise DefaultV1SourceLockError("Every seed lane must be an object")
        lane_name = str(lane.get("name") or "")
        seed = str(lane.get("seed") or "")
        if SAFE_ID_COMPONENT.fullmatch(lane_name) is None:
            raise DefaultV1SourceLockError(
                "Seed lane name must be a safe lowercase identifier"
            )
        if SAFE_ID_COMPONENT.fullmatch(seed) is None:
            raise DefaultV1SourceLockError(
                "Seed value must be a safe lowercase identifier"
            )
        seed_file = project_root / str(lane.get("seed_file") or "")
        if read_seed_file(seed_file) != [seed] * repeats:
            raise DefaultV1SourceLockError(
                f"Seed lane {lane_name} must contain {repeats} copies of {seed}"
            )
        observed_names.append(lane_name)
        observed_seeds.append(seed)
    if comparison.get("matched_seeds") != observed_seeds:
        raise DefaultV1SourceLockError(
            "Seed lane order must match comparison.matched_seeds"
        )
    if len(set(observed_names)) != 3 or len(set(observed_seeds)) != 3:
        raise DefaultV1SourceLockError(
            "Seed lane names and seeds must each be unique"
        )

    run_id_prefix = str(protocol.get("experiment_id") or "")
    if SAFE_ID_COMPONENT.fullmatch(run_id_prefix) is None:
        raise DefaultV1SourceLockError(
            "Generic formal experiment_id must be a safe lowercase run-id component"
        )
    for lane_name in observed_names:
        protocol_scenario_id(
            {
                **protocol,
                "_protocol_kind": "generic_ablation_formal",
                "_resolved_execution_configuration": resolved,
            },
            lane_name,
        )
    protocol["_protocol_kind"] = "generic_ablation_formal"
    protocol["_protocol_path"] = str(protocol_file)
    protocol["_run_id_prefix"] = run_id_prefix
    protocol["_resolved_execution_configuration"] = resolved
    protocol["_ablation_template_lock"] = template_lock
    return protocol


def validate_protocol(protocol_path: str | Path, *, root: str | Path = ROOT) -> dict[str, Any]:
    """Validate the paper design and return the parsed protocol."""

    protocol_file = Path(protocol_path).resolve()
    project_root = Path(root).resolve()
    protocol = _read_object(protocol_file)
    if protocol.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        return _validate_generic_formal_protocol(
            protocol_file,
            project_root=project_root,
            protocol=protocol,
        )

    comparison = protocol.get("comparison")
    schedule = protocol.get("schedule")
    exposure = protocol.get("exposure_control")
    common_args = protocol.get("common_args")
    lanes = protocol.get("seed_lanes")
    if not isinstance(comparison, dict) or not isinstance(schedule, dict) or not isinstance(exposure, dict):
        raise DefaultV1SourceLockError("Protocol must define comparison, schedule, and exposure_control objects")
    if not isinstance(common_args, list) or not all(isinstance(item, str) for item in common_args):
        raise DefaultV1SourceLockError("Protocol common_args must be a list of strings")
    parsed_args = _parse_common_args(common_args)
    if not isinstance(lanes, list) or len(lanes) != 3:
        raise DefaultV1SourceLockError("Default-v1 must define exactly three seed lanes")
    if int(schedule.get("workers") or 0) != len(lanes):
        raise DefaultV1SourceLockError("workers must equal the number of seed lanes")
    if schedule.get("within_seed_lane") != "strictly_serial":
        raise DefaultV1SourceLockError("Each seed lane must remain strictly serial")
    if schedule.get("resume_unit") != "seed_replicate":
        raise DefaultV1SourceLockError("Default-v1 resume_unit must be seed_replicate")
    if int(schedule.get("max_attempts_per_invocation") or 0) < 1:
        raise DefaultV1SourceLockError("max_attempts_per_invocation must be at least 1")
    if float(schedule.get("run_timeout_sec") or 0) <= 0:
        raise DefaultV1SourceLockError("run_timeout_sec must be a positive hard watchdog")
    repeats_per_seed = int(comparison.get("repeats_per_seed") or 0)
    total_runs = int(comparison.get("total_runs") or 0)
    if protocol.get("continuation") is not None:
        raise DefaultV1SourceLockError(
            "The two-stage Default-v1 treatment must start from a clean fifteen-run protocol"
        )
    if repeats_per_seed != 5:
        raise DefaultV1SourceLockError("Default-v1 requires five repeats per seed")
    if total_runs != 15:
        raise DefaultV1SourceLockError("Default-v1 requires exactly fifteen runs")
    if comparison.get("baseline") != EXPECTED_DIRECT_CLI_BASELINE_ID:
        raise DefaultV1SourceLockError(
            f"comparison.baseline must be {EXPECTED_DIRECT_CLI_BASELINE_ID!r}"
        )

    required_values = {
        "--agent": "coordinator",
        "--base-url": "https://zgc.apihy.com/v1",
        "--model": "deepseek-v4-flash",
        "--llm-timeout": "180",
        "--llm-max-retries": "1",
        "--llm-fallback-mode": "error",
        "--llm-temperature": "0",
        "--prompt-style": "analysis",
        "--short-term-memory-window": "8",
        "--short-term-memory-mode": "factual_diff",
        "--context-mode": "combat_mdp_state",
        "--turn-chat-window": "10",
        "--turn-chat-update-mode": "delta",
        "--turn-chat-assistant-history": "compact",
        "--route-plan-enforcement": "hard",
        "--route-map-prompt-format": "canonical",
        "--route-candidate-objective": "none",
        "--map-route-hint": "none",
        "--occ-context-mode": "reference_turn",
        "--self-consistency-samples": "3",
        "--self-consistency-decisions": "map_select,card_reward",
        "--self-consistency-mode": "diagnostic",
        "--action-prompt-order": "default",
        "--action-prompt-order-seed": "0",
        "--character": "Ironclad",
        "--ascension": "0",
        "--lang": "en",
        "--max-steps": "2000",
        "--max-game-actions": "0",
        "--experiment-id": EXPECTED_EXPERIMENT_ID,
        "--log-level": "full",
        "--custom-sequence-exposure": "safe",
        "--combat-skill-hints-profile": "baseline",
        "--combat-skill-hints-max-hints": "4",
    }
    for option, expected in required_values.items():
        actual = parsed_args.get(option)
        if actual != expected:
            raise DefaultV1SourceLockError(f"{option} must be {expected!r}, got {actual!r}")

    required_flags = {
        "--no-include-json-state",
        "--no-short-term-memory",
        "--no-run-summary",
        "--no-turn-check",
        "--route-plan",
        "--reward-posterior-evaluator",
        "--no-self-consistency",
        "--include-full-map",
        "--no-allow-repeat-views",
        "--no-allow-custom-sequences",
        "--no-combat-card-sequences",
        "--no-precomputed-combat-card-sequences",
        "--combat-reasoning-pool",
        "--combat-turn-board",
        "--no-combat-skill-hints",
        "--no-combat-skill-lines",
        "--no-print-prompts",
        "--no-print-model-output",
        "--trace-analysis",
    }
    for flag in sorted(required_flags):
        _require_flag(parsed_args, flag)
    expected_options = set(required_values) | required_flags
    unexpected_options = sorted(set(parsed_args) - expected_options)
    if unexpected_options:
        raise DefaultV1SourceLockError(
            "Default-v1 common_args contains non-protocol options: " + ", ".join(unexpected_options)
        )

    false_exposure_keys = {
        "reflection",
        "cross_run_experience_memory",
        "human_authored_combat_skill_hints",
        "self_consistency",
        "generic_short_term_action_history",
    }
    for key in false_exposure_keys:
        if exposure.get(key) is not False:
            raise DefaultV1SourceLockError(f"Exposure control {key} must be false")
    if exposure.get("reward_posterior_evaluator") is not True:
        raise DefaultV1SourceLockError(
            "Exposure control reward_posterior_evaluator must be true for the two-stage treatment"
        )
    if protocol.get("formal_sample_contract", {}).get("two_stage_card_reward_required") is not True:
        raise DefaultV1SourceLockError(
            "The formal sample contract must require two-stage card-reward evaluation"
        )
    if exposure.get("episode_local_structured_run_context") is not True:
        raise DefaultV1SourceLockError("The default episode-local structured run context must remain enabled")
    if exposure.get("occ_reference_turn_history") is not True:
        raise DefaultV1SourceLockError("OCC reference-turn history must remain enabled")

    expected_rules = {
        **game_rules_identity(),
        "default_delivery": "state_selected_prompt",
        "direct_cli_delivery": "static_full_system_prompt",
        "source": "sts2_bench/game_rules.py",
    }
    if protocol.get("game_rules") != expected_rules:
        raise DefaultV1SourceLockError("Protocol game_rules identity or delivery contract does not match the implementation")

    sample_contract = protocol.get("formal_sample_contract")
    if sample_contract != {
        "natural_game_over": "completed_game",
        "model_protocol_evidence_route_or_budget_failure": "preserved_non_sample_replacement_attempt",
        "automatic_retry": "transport_or_process_infrastructure_only",
        "unsupported_environment_state": "sticky_report_required",
        "audited_game_rule_exposure_required_for_every_model_request": True,
        "two_stage_card_reward_required": True,
    }:
        raise DefaultV1SourceLockError("Default-v1 formal_sample_contract does not match the runner taxonomy")

    expected_seeds = comparison.get("matched_seeds")
    if not isinstance(expected_seeds, list) or len(expected_seeds) != 3:
        raise DefaultV1SourceLockError("comparison.matched_seeds must contain exactly three seeds")
    observed_seeds: list[str] = []
    observed_names: list[str] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            raise DefaultV1SourceLockError("Every seed lane must be an object")
        seed = str(lane.get("seed") or "")
        seed_file = project_root / str(lane.get("seed_file") or "")
        values = read_seed_file(seed_file)
        if values != [seed] * 5:
            raise DefaultV1SourceLockError(f"Seed lane {lane.get('name')} must contain five copies of {seed}")
        observed_seeds.append(seed)
        observed_names.append(str(lane.get("name") or ""))
    if observed_seeds != expected_seeds:
        raise DefaultV1SourceLockError("Seed lane order must match comparison.matched_seeds")
    if len(set(observed_seeds)) != len(observed_seeds) or len(set(observed_names)) != len(observed_names):
        raise DefaultV1SourceLockError("Seed lane names and seeds must each be unique")
    default_configuration = protocol.get("default_configuration")
    if not isinstance(default_configuration, dict):
        raise DefaultV1SourceLockError("Protocol default_configuration must be an object")
    if default_configuration.get("allow_custom_sequences") is not False:
        raise DefaultV1SourceLockError("Default-v1 must disable custom sequences")
    if default_configuration.get("combat_card_sequences") is not False:
        raise DefaultV1SourceLockError("Default-v1 must disable combat card sequences")
    if default_configuration.get("precomputed_combat_card_sequences") is not False:
        raise DefaultV1SourceLockError("Default-v1 must disable precomputed combat card sequences")
    protocol["_protocol_kind"] = "legacy_default_v1"
    protocol["_protocol_path"] = str(protocol_file)
    protocol["_run_id_prefix"] = LEGACY_RUN_ID_PREFIX
    protocol["_resolved_execution_configuration"] = resolve_protocol_execution_configuration(
        protocol
    )
    return protocol


def build_generated_suite(protocol: dict[str, Any]) -> dict[str, Any]:
    """Expand the formal design into independently identified replicate jobs."""

    jobs = []
    for lane in protocol["seed_lanes"]:
        lane_name = str(lane["name"])
        seed = str(lane["seed"])
        scenario_id = protocol_scenario_id(protocol, lane_name)
        for replicate in range(1, int(protocol["comparison"]["repeats_per_seed"]) + 1):
            replicate_id = f"replicate-{replicate:02d}"
            jobs.append(
                {
                    "name": f"{lane_name}-{replicate_id}",
                    "lane": lane_name,
                    "lane_order": replicate,
                    "seed": seed,
                    "scenario_id": scenario_id,
                    "replicate_id": replicate_id,
                }
            )
    suite = {
        "schema_version": 2,
        "generated_from": f"experiments/{PROTOCOL_FILENAME}",
        "experiment_id": protocol["experiment_id"],
        "game_rules": protocol["game_rules"],
        "schedule": protocol["schedule"],
        "jobs": jobs,
    }
    if protocol.get("_protocol_kind") == "generic_ablation_formal":
        protocol_path = Path(str(protocol["_protocol_path"]))
        try:
            generated_from = protocol_path.relative_to(ROOT).as_posix()
        except ValueError:
            generated_from = str(protocol_path)
        suite.update(
            {
                "schema_version": 3,
                "generated_from": generated_from,
                "run_id_prefix": protocol_run_id_prefix(protocol),
                "ablation_template_identity_sha256": protocol[
                    "_ablation_template_lock"
                ]["identity_sha256"],
            }
        )
        for job in jobs:
            job["run_id"] = (
                f"{protocol_run_id_prefix(protocol)}--{job['lane']}--"
                f"{job['replicate_id']}"
            )
    return suite


def _include_source_path(relative: Path) -> bool:
    if any(part in EXCLUDED_NAMES for part in relative.parts):
        return False
    return relative.suffix.lower() not in EXCLUDED_SUFFIXES


def _files_under(path: Path, *, root: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise DefaultV1SourceLockError(f"Required source-lock input is missing: {path}")
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and _include_source_path(candidate.relative_to(root)):
            yield candidate


def _canonical_sts2_cli_root(
    benchmark_root: Path,
    explicit_root: str | Path | None = None,
) -> Path:
    sibling = (benchmark_root.parent / "sts2-cli").resolve()
    selected = Path(explicit_root).expanduser().resolve() if explicit_root is not None else sibling
    ambient = os.environ.get("STS2_CLI_ROOT")
    if ambient and Path(ambient).expanduser().resolve() != selected:
        raise DefaultV1SourceLockError(
            "Ambient STS2_CLI_ROOT may not redirect source locking; pass --sts2-cli-root explicitly"
        )
    return selected


def _source_files(
    root: Path,
    canonical_cli: Path,
    protocol: dict[str, Any],
    protocol_file: Path,
) -> list[tuple[str, Path]]:
    benchmark_inputs = [root / relative for relative in BENCHMARK_EXECUTION_FILES]
    benchmark_inputs.extend(
        [
            protocol_file,
            *(root / str(lane["seed_file"]) for lane in protocol["seed_lanes"]),
        ]
    )
    if protocol.get("_protocol_kind") != "generic_ablation_formal":
        benchmark_inputs.append(
            root / "scripts" / "run_default_v1_rules_v2_formal.ps1"
        )
    runtime_inputs = [
        canonical_cli / "python" / "game_log.py",
        canonical_cli / "python" / "play.py",
        canonical_cli / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0",
        canonical_cli / "lib",
        canonical_cli / "localization_eng",
        canonical_cli / "localization_zhs",
        canonical_cli / ".tools" / "dotnet" / "dotnet.exe",
        canonical_cli / ".tools" / "dotnet" / "host",
        canonical_cli / ".tools" / "dotnet" / "shared" / "Microsoft.NETCore.App" / DOTNET_RUNTIME_VERSION,
    ]
    files: dict[str, tuple[str, Path]] = {}
    for item in benchmark_inputs:
        for file_path in _files_under(item, root=root):
            relative = file_path.relative_to(root)
            label = f"benchmark/{relative.as_posix()}"
            files[label] = (label, file_path.resolve())
    for item in runtime_inputs:
        for file_path in _files_under(item, root=canonical_cli):
            relative = STS2_CLI_LABEL_ROOT / file_path.relative_to(canonical_cli)
            label = relative.as_posix()
            files[label] = (label, file_path.resolve())
    for source_root in (
        canonical_cli / "src" / "Sts2Headless",
        canonical_cli / "src" / "GodotStubs",
    ):
        for file_path in sorted(source_root.rglob("*")):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in {".cs", ".csproj"}
                and "bin" not in file_path.parts
                and "obj" not in file_path.parts
            ):
                relative = STS2_CLI_LABEL_ROOT / file_path.relative_to(canonical_cli)
                label = relative.as_posix()
                files[label] = (label, file_path.resolve())
    selected = [files[key] for key in sorted(files)]
    selected_labels = {label.removeprefix("benchmark/") for label, _ in selected if label.startswith("benchmark/")}
    forbidden = sorted(selected_labels & FORBIDDEN_BENCHMARK_SOURCE_FILES)
    if forbidden:
        raise DefaultV1SourceLockError(
            "Default-v1 source lock includes forbidden Direct/reflection/experience modules: "
            + ", ".join(forbidden)
        )
    return selected


def _require_fresh_headless_build(canonical_cli: Path) -> None:
    headless_root = canonical_cli / "src" / "Sts2Headless"
    stubs_root = canonical_cli / "src" / "GodotStubs"
    headless_dll = headless_root / "bin" / "Debug" / "net9.0" / "Sts2Headless.dll"
    stubs_dll = headless_root / "bin" / "Debug" / "net9.0" / "GodotSharp.dll"
    game_dll = canonical_cli / "lib" / "sts2.dll"
    if not headless_dll.is_file() or not stubs_dll.is_file() or not game_dll.is_file():
        raise DefaultV1SourceLockError(
            "A prebuilt headless runtime, Godot stubs, and lib/sts2.dll are required before source locking"
        )
    headless_built_at = headless_dll.stat().st_mtime
    stubs_built_at = stubs_dll.stat().st_mtime
    stale_headless_sources = [
        path
        for path in headless_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".cs", ".csproj"}
        and "bin" not in path.parts
        and "obj" not in path.parts
        and path.stat().st_mtime > headless_built_at
    ]
    stale_stub_sources = [
        path
        for path in stubs_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".cs", ".csproj"}
        and "bin" not in path.parts
        and "obj" not in path.parts
        and path.stat().st_mtime > stubs_built_at
    ]
    if game_dll.stat().st_mtime > headless_built_at or stale_headless_sources or stale_stub_sources:
        raise DefaultV1SourceLockError("Headless runtime is stale; build it before locking the active sources")


def _inventory(source_files: Iterable[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "path": str(path.resolve()),
            "sha256": hash_file(path),
            "size": path.stat().st_size,
        }
        for label, path in source_files
    ]


def _verify_inventory(expected: list[dict[str, Any]]) -> None:
    for item in expected:
        if not isinstance(item, dict):
            raise DefaultV1SourceLockError("Source lock contains an invalid inventory entry")
        path = Path(str(item.get("path") or "")).resolve()
        actual = {
            "label": item.get("label"),
            "path": str(path),
            "sha256": hash_file(path),
            "size": path.stat().st_size if path.is_file() else None,
        }
        if actual != item:
            raise DefaultV1SourceLockError(f"Locked active source file drifted: {item.get('label')}")


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_executable": str(executable),
        "python_executable_sha256": hash_file(executable),
        "python_dependencies": _dependency_versions(),
        "dotnet_framework": "Microsoft.NETCore.App",
        "dotnet_version": DOTNET_RUNTIME_VERSION,
        "execution_mode": "active_worktree",
        "sanitized_python_environment": SANITIZED_PYTHON_ENV,
    }


def _require_sanitized_python_environment() -> None:
    for name, value in SANITIZED_PYTHON_ENV.items():
        if os.environ.get(name) != value:
            raise DefaultV1SourceLockError(f"Source-lock operations require {name}={value}")
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        if os.environ.get(name):
            raise DefaultV1SourceLockError(f"Source-lock operations require {name} to be unset")


def _identity_payload(
    inventory: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    generated_suite_sha256: str,
    benchmark_root: Path,
    sts2_cli_root: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": 4,
        "execution_mode": "active_worktree",
        "benchmark_root": str(benchmark_root.resolve()),
        "sts2_cli_root": str(sts2_cli_root.resolve()),
        "runtime": _runtime_identity(),
        "protocol_sha256": protocol_sha256,
        "generated_suite_sha256": generated_suite_sha256,
        "game_rules": protocol["game_rules"],
        "exposure_control": protocol["exposure_control"],
        "formal_sample_contract": protocol["formal_sample_contract"],
        "credential_delivery": "explicit_external_env_file",
        "files": inventory,
    }
    if protocol.get("_protocol_kind") == "generic_ablation_formal":
        payload.update(
            {
                "protocol_kind": "generic_ablation_formal",
                "run_id_prefix": protocol_run_id_prefix(protocol),
                "resolved_execution_configuration": protocol[
                    "_resolved_execution_configuration"
                ],
                "ablation_template_lock": protocol["_ablation_template_lock"],
            }
        )
    return payload


def _require_source_import_smoke(root: Path) -> None:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        environment.pop(name, None)
    environment.update(SANITIZED_PYTHON_ENV)
    commands = (
        [
            sys.executable,
            "-B",
            "-s",
            "-X",
            "utf8",
            "-c",
            (
                "import sts2_bench.default_v1_source_lock; "
                "import sts2_bench.default_v1_experiment; "
                "import sts2_bench.run_benchmark"
            ),
        ],
        [sys.executable, "-B", "-s", "-X", "utf8", "-m", "sts2_bench", "run", "--help"],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise DefaultV1SourceLockError(f"Active-worktree Python closure smoke failed: {diagnostic}")


def create_or_verify_source_lock(
    lock_root: str | Path,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    source_root: str | Path = ROOT,
    sts2_cli_root: str | Path | None = None,
) -> dict[str, Any]:
    """Lock the active benchmark/runtime files without copying an execution tree."""

    source = Path(source_root).resolve()
    destination = Path(lock_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    _require_sanitized_python_environment()
    protocol = validate_protocol(protocol_file, root=source)
    canonical_cli = _canonical_sts2_cli_root(source, sts2_cli_root)
    _require_fresh_headless_build(canonical_cli)
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / IDENTITY_FILE
    if identity_path.is_file():
        verification = verify_source_lock(destination)
        frozen = _read_object(identity_path)
        if (
            Path(str(frozen.get("benchmark_root") or "")).resolve() != source
            or Path(str(frozen.get("sts2_cli_root") or "")).resolve() != canonical_cli
            or Path(str(frozen.get("protocol_path") or "")).resolve() != protocol_file
        ):
            raise DefaultV1SourceLockError("Existing source lock points to different active roots")
        return verification

    generated_suite = build_generated_suite(protocol)
    generated_suite_path = destination / GENERATED_SUITE
    _write_json(generated_suite_path, generated_suite)
    inventory = _inventory(
        _source_files(source, canonical_cli, protocol, protocol_file)
    )
    payload = _identity_payload(
        inventory,
        protocol=protocol,
        protocol_sha256=hash_file(protocol_file),
        generated_suite_sha256=hash_file(generated_suite_path),
        benchmark_root=source,
        sts2_cli_root=canonical_cli,
    )
    frozen = {
        **payload,
        "identity_sha256": hash_json(payload),
        "created_at": _utc_now(),
        "protocol_path": str(protocol_file),
        "generated_suite": str(generated_suite_path),
        "lock_root": str(destination),
    }
    _write_json(identity_path, frozen)
    verification = verify_source_lock(destination)
    _write_json(destination / VERIFICATION_FILE, verification)
    return verification


def verify_source_lock(lock_root: str | Path) -> dict[str, Any]:
    """Verify the active worktrees and the no-reflection experiment contract."""

    root = Path(lock_root).resolve()
    _require_sanitized_python_environment()
    identity_path = root / IDENTITY_FILE
    if not identity_path.is_file():
        raise DefaultV1SourceLockError(f"Source identity is missing: {identity_path}")
    frozen = _read_object(identity_path)
    expected_files = frozen.get("files")
    if not isinstance(expected_files, list):
        raise DefaultV1SourceLockError("Source identity has no file inventory")

    benchmark_root = Path(str(frozen.get("benchmark_root") or "")).resolve()
    sts2_cli_root = Path(str(frozen.get("sts2_cli_root") or "")).resolve()
    protocol_path = Path(str(frozen.get("protocol_path") or "")).resolve()
    if frozen.get("execution_mode") != "active_worktree":
        raise DefaultV1SourceLockError("Default-v1 source lock is not an active-worktree identity")
    protocol = validate_protocol(protocol_path, root=benchmark_root)
    expected_suite = build_generated_suite(protocol)
    actual_suite = _read_object(root / GENERATED_SUITE)
    if actual_suite != expected_suite:
        raise DefaultV1SourceLockError("Generated Default-v1 suite does not match the locked protocol")

    payload = _identity_payload(
        expected_files,
        protocol=protocol,
        protocol_sha256=hash_file(protocol_path),
        generated_suite_sha256=hash_file(root / GENERATED_SUITE),
        benchmark_root=benchmark_root,
        sts2_cli_root=sts2_cli_root,
    )
    expected_identity = str(frozen.get("identity_sha256") or "")
    actual_identity = hash_json(payload)
    if actual_identity != expected_identity:
        raise DefaultV1SourceLockError("Source identity document was modified or is internally inconsistent")
    for key, value in payload.items():
        if frozen.get(key) != value:
            raise DefaultV1SourceLockError(f"Source identity metadata drifted at {key}")

    _verify_inventory(expected_files)

    benchmark_files = {
        str(item["label"]).removeprefix("benchmark/")
        for item in expected_files
        if isinstance(item, dict) and str(item.get("label") or "").startswith("benchmark/sts2_bench/")
    }
    expected_benchmark_files = set(BENCHMARK_EXECUTION_FILES)
    if benchmark_files != expected_benchmark_files:
        missing = sorted(expected_benchmark_files - benchmark_files)
        unexpected = sorted(benchmark_files - expected_benchmark_files)
        raise DefaultV1SourceLockError(
            "Locked benchmark source closure drifted; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    forbidden = sorted(benchmark_files & FORBIDDEN_BENCHMARK_SOURCE_FILES)
    if forbidden:
        raise DefaultV1SourceLockError(
            "Locked benchmark closure exposes forbidden Direct/reflection/experience modules: "
            + ", ".join(forbidden)
        )
    localization_labels = {
        str(item.get("label") or "")
        for item in expected_files
        if isinstance(item, dict)
    }
    for prefix in ("sts2-cli/localization_eng/", "sts2-cli/localization_zhs/"):
        if not any(label.startswith(prefix) for label in localization_labels):
            raise DefaultV1SourceLockError(f"Source lock is missing localization resources under {prefix}")
        if f"{prefix}_done.txt" not in localization_labels:
            raise DefaultV1SourceLockError(
                f"Source lock is missing the localization completion marker {prefix}_done.txt"
            )
        if not any(label.startswith(prefix) and label.endswith(".json") for label in localization_labels):
            raise DefaultV1SourceLockError(f"Source lock has no localization JSON resources under {prefix}")
    _require_source_import_smoke(benchmark_root)

    common_args = protocol["common_args"]
    forbidden_modules = protocol["exposure_control"].get("forbidden_runtime_modules") or []
    joined_args = " ".join(common_args).lower()
    exposed = [name for name in forbidden_modules if str(name).lower() in joined_args]
    if exposed:
        raise DefaultV1SourceLockError(f"Reflection module exposed by runtime arguments: {', '.join(exposed)}")

    agents_text = (benchmark_root / "sts2_bench" / "agents.py").read_text(encoding="utf-8")
    prompt_leaks = sorted(fragment for fragment in FORBIDDEN_BASE_PROMPT_FRAGMENTS if fragment in agents_text)
    if prompt_leaks:
        raise DefaultV1SourceLockError(
            "Human-authored combat policy remains in the base prompt: " + ", ".join(prompt_leaks)
        )

    verification = {
        "status": "verified",
        "verified_at": _utc_now(),
        "identity_sha256": expected_identity,
        "file_count": len(expected_files),
        "protocol_sha256": hash_file(protocol_path),
        "generated_suite_sha256": hash_file(root / GENERATED_SUITE),
        "game_rules": protocol["game_rules"],
        "reflection_exposed": False,
        "cross_run_experience_memory_exposed": False,
        "seed_lanes": len(protocol["seed_lanes"]),
        "repeats_per_seed": int(protocol["comparison"]["repeats_per_seed"]),
        "total_runs": int(protocol["comparison"]["total_runs"]),
        "execution_mode": "active_worktree",
        "benchmark_root": str(benchmark_root),
        "sts2_cli_root": str(sts2_cli_root),
    }
    return verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or verify the Default-v1 active-worktree source lock.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock", help="Create or verify a source lock.")
    lock.add_argument("--lock-root", required=True)
    lock.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    lock.add_argument("--source-root", default=str(ROOT))
    lock.add_argument("--sts2-cli-root")
    verify = subparsers.add_parser("verify", help="Verify an existing source lock.")
    verify.add_argument("--lock-root", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "lock":
            result = create_or_verify_source_lock(
                args.lock_root,
                protocol_path=args.protocol,
                source_root=args.source_root,
                sts2_cli_root=args.sts2_cli_root,
            )
        else:
            result = verify_source_lock(args.lock_root)
    except DefaultV1SourceLockError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
