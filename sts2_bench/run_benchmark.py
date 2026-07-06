"""Command-line runner for STS2 benchmark policies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .runner import JsonlLogger, agent_from_args, run_one, summarize
from .seeds import DEFAULT_SEEDS_FILE, read_seed_file
from .trace_analysis import analyze_benchmark_trace


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV = ROOT / ".env"
TRACKED_CONFIG = ROOT / "sts2_bench" / "benchmark.yaml"
CONFIG_ENV_KEYS = {
    "agent": "STS2_BENCH_AGENT",
    "base_url": "STS2_BENCH_BASE_URL",
    "model": "STS2_BENCH_MODEL",
    "llm_timeout": "STS2_BENCH_LLM_TIMEOUT",
    "llm_max_retries": "STS2_BENCH_LLM_MAX_RETRIES",
    "llm_fallback_mode": "STS2_BENCH_LLM_FALLBACK_MODE",
    "llm_temperature": "STS2_BENCH_LLM_TEMPERATURE",
    "prompt_style": "STS2_BENCH_PROMPT_STYLE",
    "include_json_state": "STS2_BENCH_INCLUDE_JSON_STATE",
    "memory_enabled": "STS2_BENCH_MEMORY_ENABLED",
    "memory_window": "STS2_BENCH_MEMORY_WINDOW",
    "memory_mode": "STS2_BENCH_MEMORY_MODE",
    "run_summary_enabled": "STS2_BENCH_RUN_SUMMARY_ENABLED",
    "character": "STS2_BENCH_CHARACTER",
    "ascension": "STS2_BENCH_ASCENSION",
    "lang": "STS2_BENCH_LANG",
    "seeds": "STS2_BENCH_SEEDS",
    "count": "STS2_BENCH_COUNT",
    "max_steps": "STS2_BENCH_MAX_STEPS",
    "max_game_actions": "STS2_BENCH_MAX_GAME_ACTIONS",
    "out": "STS2_BENCH_OUT",
    "log_level": "STS2_BENCH_LOG_LEVEL",
    "print_prompts": "STS2_BENCH_PRINT_PROMPTS",
    "print_model_output": "STS2_BENCH_PRINT_MODEL_OUTPUT",
    "include_full_map": "STS2_BENCH_INCLUDE_FULL_MAP",
    "allow_repeat_views": "STS2_BENCH_ALLOW_REPEAT_VIEWS",
    "allow_custom_sequences": "STS2_BENCH_ALLOW_CUSTOM_SEQUENCES",
    "custom_sequence_exposure": "STS2_BENCH_CUSTOM_SEQUENCE_EXPOSURE",
    "combat_card_sequences_enabled": "STS2_BENCH_COMBAT_CARD_SEQUENCES",
    "precomputed_combat_card_sequences_enabled": "STS2_BENCH_PRECOMPUTED_COMBAT_CARD_SEQUENCES",
    "combat_reasoning_pool_enabled": "STS2_BENCH_COMBAT_REASONING_POOL",
    "combat_turn_board_enabled": "STS2_BENCH_COMBAT_TURN_BOARD",
    "route_plan_enabled": "STS2_BENCH_ROUTE_PLAN_ENABLED",
    "route_plan_enforcement": "STS2_BENCH_ROUTE_PLAN_ENFORCEMENT",
    "reward_posterior_evaluator": "STS2_BENCH_REWARD_POSTERIOR_EVALUATOR",
    "self_consistency_enabled": "STS2_BENCH_SELF_CONSISTENCY",
    "self_consistency_samples": "STS2_BENCH_SELF_CONSISTENCY_SAMPLES",
    "self_consistency_decisions": "STS2_BENCH_SELF_CONSISTENCY_DECISIONS",
    "self_consistency_mode": "STS2_BENCH_SELF_CONSISTENCY_MODE",
    "reflection_enabled": "STS2_BENCH_REFLECTION_VERIFIER",
    "reflection_decisions": "STS2_BENCH_REFLECTION_DECISIONS",
    "reflection_mode": "STS2_BENCH_REFLECTION_MODE",
    "reflection_max_options": "STS2_BENCH_REFLECTION_MAX_OPTIONS",
    "action_prompt_order": "STS2_BENCH_ACTION_PROMPT_ORDER",
    "action_prompt_order_seed": "STS2_BENCH_ACTION_PROMPT_ORDER_SEED",
    "trace_analysis": "STS2_BENCH_TRACE_ANALYSIS",
    "trace_analysis_dir": "STS2_BENCH_TRACE_ANALYSIS_DIR",
}
NESTED_CONFIG_ENV_KEYS = {
    ("context_management", "mode"): "STS2_BENCH_CONTEXT_MODE",
    ("context_management", "turn_chat", "window"): "STS2_BENCH_TURN_CHAT_WINDOW",
    ("context_management", "turn_chat", "update_mode"): "STS2_BENCH_TURN_CHAT_UPDATE_MODE",
    ("context_management", "turn_chat", "assistant_history"): "STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY",
    ("context_management", "turn_chat", "plan", "enabled"): "STS2_BENCH_TURN_CHAT_PLAN_ENABLED",
    ("context_management", "turn_check", "enabled"): "STS2_BENCH_TURN_CHECK_ENABLED",
    ("route_plan", "enabled"): "STS2_BENCH_ROUTE_PLAN_ENABLED",
    ("route_plan", "enforcement"): "STS2_BENCH_ROUTE_PLAN_ENFORCEMENT",
    ("diagnostics", "self_consistency", "enabled"): "STS2_BENCH_SELF_CONSISTENCY",
    ("diagnostics", "self_consistency", "samples"): "STS2_BENCH_SELF_CONSISTENCY_SAMPLES",
    ("diagnostics", "self_consistency", "decisions"): "STS2_BENCH_SELF_CONSISTENCY_DECISIONS",
    ("diagnostics", "self_consistency", "mode"): "STS2_BENCH_SELF_CONSISTENCY_MODE",
    ("diagnostics", "reflection_verifier", "enabled"): "STS2_BENCH_REFLECTION_VERIFIER",
    ("diagnostics", "reflection_verifier", "decisions"): "STS2_BENCH_REFLECTION_DECISIONS",
    ("diagnostics", "reflection_verifier", "mode"): "STS2_BENCH_REFLECTION_MODE",
    ("diagnostics", "reflection_verifier", "max_options"): "STS2_BENCH_REFLECTION_MAX_OPTIONS",
    ("diagnostics", "action_prompt_order", "mode"): "STS2_BENCH_ACTION_PROMPT_ORDER",
    ("diagnostics", "action_prompt_order", "seed"): "STS2_BENCH_ACTION_PROMPT_ORDER_SEED",
    ("context_management", "combat_reasoning_pool", "enabled"): "STS2_BENCH_COMBAT_REASONING_POOL",
    ("context_management", "combat_turn_board", "enabled"): "STS2_BENCH_COMBAT_TURN_BOARD",
}


def load_seeds(path: str | None, count: int) -> list[str]:
    if not path:
        seeds = read_seed_file(DEFAULT_SEEDS_FILE)
        if count > len(seeds):
            raise ValueError(
                f"Requested count={count}, but {DEFAULT_SEEDS_FILE} contains only "
                f"{len(seeds)} preconfigured fixed seeds. Pass --seeds to declare an explicit seed file."
            )
        return seeds[:count]
    return read_seed_file(path)


def load_env_file(path: Path) -> None:
    """Load a simple env file without adding a runtime dependency.

    Existing environment variables win over file values. CLI arguments then win
    over both because argparse uses these values only as defaults.
    """

    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()

        os.environ.setdefault(key, value)


def load_dotenv(path: Path = LOCAL_ENV) -> None:
    """Load local secrets from .env.

    Kept as a small compatibility wrapper for scripts that imported the old
    helper directly.
    """

    load_env_file(path)


def _parse_config_scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    normalized = value.lower()
    if normalized in {"true", "false"}:
        return normalized == "true"
    if normalized in {"null", "none", "~"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml_mapping(path: Path) -> dict[str, object]:
    """Parse the tracked benchmark config without requiring a YAML dependency.

    The committed config intentionally uses only mapping nodes and scalar
    values, so a small fallback parser keeps basic commands such as `--help`
    usable before optional benchmark dependencies are installed.
    """

    data: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, data)]
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip(" "))]:
            raise ValueError(f"{path}:{lineno} must use spaces for indentation")

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{lineno} must use 'key: value' syntax")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{lineno} has an empty key")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"{path}:{lineno} has invalid indentation")
        parent = stack[-1][1]

        value = raw_value.strip()
        if not value:
            child: dict[str, object] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        if value and not value.startswith(("'", '"')):
            value = value.split(" #", 1)[0].strip()
        parent[key] = _parse_config_scalar(value)
    return data


def load_benchmark_config(path: Path = TRACKED_CONFIG) -> None:
    """Load tracked benchmark defaults from YAML.

    Values are mirrored into the existing STS2_BENCH_* environment names so
    CLI defaults and shell overrides stay backward compatible.
    """

    if not path.is_file():
        return

    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        data = _load_simple_yaml_mapping(path)
    else:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if data is None:
        return
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")

    for key, env_key in CONFIG_ENV_KEYS.items():
        if key not in data or data[key] is None:
            continue

        value = data[key]
        _set_config_env(env_key, value, f"{path}:{key}")

    for key_path, env_key in NESTED_CONFIG_ENV_KEYS.items():
        found, value = _nested_config_value(data, key_path)
        if not found or value is None:
            continue
        _set_config_env(env_key, value, f"{path}:{'.'.join(key_path)}")


def _nested_config_value(data: dict[str, object], key_path: tuple[str, ...]) -> tuple[bool, object]:
    current: object = data
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _set_config_env(env_key: str, value: object, source: str) -> None:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (str, int, float)):
        text = str(value)
    else:
        raise ValueError(f"{source} must be a scalar value, got {type(value).__name__}")
    os.environ.setdefault(env_key, text)


def load_benchmark_env() -> None:
    """Load local secrets first, then tracked benchmark defaults.

    Precedence is: CLI arguments, shell environment variables, .env, tracked
    benchmark config, code defaults.
    """

    load_env_file(LOCAL_ENV)
    load_benchmark_config(TRACKED_CONFIG)


def env_str(name: str, default: str | None = None, *, fallbacks: tuple[str, ...] = ()) -> str | None:
    for key in (name, *fallbacks):
        value = os.environ.get(key)
        if value:
            return value
    return default


def env_int(name: str, default: int) -> int:
    value = env_str(name)
    return int(value) if value is not None else default


def env_float(name: str, default: float) -> float:
    value = env_str(name)
    return float(value) if value is not None else default


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _default_trace_analysis_dir(out: str | Path) -> Path:
    trace_path = Path(out)
    return Path("artifacts") / "trace_analysis" / trace_path.stem


def _resolve_trace_analysis_dir(
    *,
    out: str | Path | None,
    trace_analysis_enabled: bool,
    trace_analysis_dir: str | Path | None,
) -> Path | None:
    if not trace_analysis_enabled:
        return None
    if trace_analysis_dir:
        return Path(trace_analysis_dir)
    if not out:
        return None
    return _default_trace_analysis_dir(out)


def _redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    secret_flags = {"--api-key"}
    for token in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if token in secret_flags:
            redacted.append(token)
            redact_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in secret_flags):
            flag, _value = token.split("=", 1)
            redacted.append(f"{flag}=<redacted>")
            continue
        redacted.append(token)
    return redacted


def build_run_config(args: argparse.Namespace, seeds: list[str], argv: list[str]) -> dict[str, object]:
    """Build a redacted, resolved benchmark config for logs and trace files."""

    turn_check_enabled = bool(getattr(args, "turn_check", getattr(args, "turn_chat_plan", False)))
    turn_chat_config = {
        "window": args.turn_chat_window,
        "update_mode": args.turn_chat_update_mode,
        "assistant_history": args.turn_chat_assistant_history,
        "turn_check": {
            "enabled": turn_check_enabled,
        },
    }
    route_candidate_config = {
        "objective": getattr(args, "route_candidate_objective", "none"),
    }
    if args.agent == "coordinator":
        ca_mode = "combat_mdp_state" if args.context_mode == "combat_mdp_state" else "turn_chat"
        ca_context: dict[str, object] = {"mode": ca_mode}
        if ca_mode == "combat_mdp_state":
            ca_context["turn_process"] = "structured_current_turn_actions"
            ca_context["combat_reasoning_pool"] = bool(getattr(args, "combat_reasoning_pool_enabled", False))
            ca_context["combat_turn_board"] = bool(getattr(args, "combat_turn_board_enabled", False))
        else:
            ca_context["turn_chat"] = turn_chat_config
        context_management: dict[str, object] = {
            "mode": "coordinator",
            "route_candidate_planner": route_candidate_config,
            "ca": ca_context,
            "occ": {
                "mode": "room_process",
                "run_context": "rm_rp_slice",
                "reward_posterior_evaluator": bool(getattr(args, "reward_posterior_evaluator", True)),
                "room_context": {
                    "scope": "current_room",
                    "source": "rm_rp_slice_plus_room_process_ledger",
                    "reset": "room_change_or_map",
                },
            },
        }
    else:
        context_management = {
            "mode": args.context_mode,
            "route_candidate_planner": route_candidate_config,
            "turn_chat": turn_chat_config,
            "combat_reasoning_pool": {
                "enabled": bool(getattr(args, "combat_reasoning_pool_enabled", False)),
            },
            "combat_turn_board": {
                "enabled": bool(getattr(args, "combat_turn_board_enabled", False)),
            },
        }

    return {
        "argv": _redact_argv(argv),
        "agent": {
            "type": args.agent,
            "base_url": args.base_url,
            "model": args.model,
            "llm_timeout": args.llm_timeout,
            "llm_max_retries": args.llm_max_retries,
            "llm_fallback_mode": args.llm_fallback_mode,
            "llm_temperature": args.llm_temperature,
            "prompt_style": args.prompt_style,
            "include_json_state": args.include_json_state,
            "memory": {
                "enabled": args.memory_enabled,
                "window": args.memory_window,
                "mode": args.memory_mode,
            },
            "run_summary_enabled": args.run_summary,
            "context_management": context_management,
            "route_plan": {
                "enabled": bool(getattr(args, "route_plan_enabled", True)),
                "enforcement": getattr(args, "route_plan_enforcement", "hard"),
                "route_map_prompt_format": getattr(args, "route_map_prompt_format", "canonical"),
            },
            "diagnostics": {
                "self_consistency": {
                    "enabled": bool(getattr(args, "self_consistency_enabled", False)),
                    "samples": getattr(args, "self_consistency_samples", 3),
                    "decisions": getattr(args, "self_consistency_decisions", "map_select,card_reward"),
                    "mode": getattr(args, "self_consistency_mode", "diagnostic"),
                },
                "reflection_verifier": {
                    "enabled": bool(getattr(args, "reflection_enabled", False)),
                    "decisions": getattr(args, "reflection_decisions", "card_reward"),
                    "mode": getattr(args, "reflection_mode", "diagnostic"),
                    "max_options": getattr(args, "reflection_max_options", 8),
                },
                "action_prompt_order": {
                    "mode": getattr(args, "action_prompt_order", "default"),
                    "seed": getattr(args, "action_prompt_order_seed", 0),
                },
            },
        },
        "game": {
            "character": args.character,
            "ascension": args.ascension,
            "lang": args.lang,
        },
        "benchmark": {
            "seeds_file": args.seeds,
            "seeds": list(seeds),
            "count": args.count,
            "max_steps": args.max_steps,
            "max_game_actions": args.max_game_actions,
            "include_full_map": args.include_full_map,
            "allow_repeat_views": args.allow_repeat_views,
            "allow_custom_sequences": bool(getattr(args, "allow_custom_sequences", True)),
            "custom_sequence_exposure": getattr(args, "custom_sequence_exposure", "safe"),
            "combat_card_sequences": bool(getattr(args, "combat_card_sequences_enabled", True)),
            "precomputed_combat_card_sequences": bool(
                getattr(args, "precomputed_combat_card_sequences_enabled", False)
            ),
            "combat_reasoning_pool": bool(getattr(args, "combat_reasoning_pool_enabled", False)),
            "combat_turn_board": bool(getattr(args, "combat_turn_board_enabled", False)),
        },
        "logging": {
            "out": args.out,
            "log_level": args.log_level,
            "print_prompts": args.print_prompts,
            "print_model_output": args.print_model_output,
            "trace_analysis": bool(getattr(args, "trace_analysis", True)),
            "trace_analysis_dir": getattr(args, "trace_analysis_dir", None),
        },
    }


def main(argv: list[str] | None = None) -> int:
    load_benchmark_env()

    parser = argparse.ArgumentParser(description="Run STS2 headless benchmark policies.")
    parser.add_argument("--agent", choices=["random", "llm", "coordinator"], default=env_str("STS2_BENCH_AGENT", "random"))
    parser.add_argument(
        "--base-url",
        default=env_str("STS2_BENCH_BASE_URL"),
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1",
    )
    parser.add_argument("--model", default=env_str("STS2_BENCH_MODEL"), help="Model name for --agent llm")
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=env_float("STS2_BENCH_LLM_TIMEOUT", 120.0),
        help="Timeout in seconds for one LLM HTTP request",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=env_int("STS2_BENCH_LLM_MAX_RETRIES", 2),
        help="Number of retries after an LLM request or parsing failure",
    )
    parser.add_argument(
        "--llm-fallback-mode",
        choices=["error", "action"],
        default=env_str("STS2_BENCH_LLM_FALLBACK_MODE", "error"),
        help="What to do after all LLM retries fail: error stops the run; action uses deterministic first-action fallback",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=env_float("STS2_BENCH_LLM_TEMPERATURE", 0.0),
        help="Sampling temperature for LLM chat completions",
    )
    parser.add_argument(
        "--api-key",
        default=env_str("DEEPSEEK_API_KEY", "local", fallbacks=("OPENAI_API_KEY",)),
    )
    parser.add_argument(
        "--prompt-style",
        choices=["default", "analysis"],
        default=env_str("STS2_BENCH_PROMPT_STYLE", "default"),
        help="LLM prompt style. 'analysis' asks for structured situation analysis and calculations.",
    )
    parser.add_argument(
        "--include-json-state",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_INCLUDE_JSON_STATE", False),
        help="Include compact state JSON in LLM prompts",
    )
    parser.add_argument(
        "--memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_MEMORY_ENABLED", False),
        help="Include a short per-episode action memory in LLM prompts",
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=env_int("STS2_BENCH_MEMORY_WINDOW", 8),
        help="Number of recent selected actions to keep in agent memory",
    )
    parser.add_argument(
        "--memory-mode",
        choices=["action_reason", "factual_diff"],
        default=env_str("STS2_BENCH_MEMORY_MODE", "action_reason"),
        help="Agent memory content: previous action reasons or factual state-transition diffs",
    )
    parser.add_argument(
        "--run-summary",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_RUN_SUMMARY_ENABLED", True),
        help="Include a small rule-based run summary in the LLM prompt",
    )
    parser.add_argument(
        "--context-mode",
        choices=["single_turn", "turn_chat", "combat_mdp_state"],
        default=env_str("STS2_BENCH_CONTEXT_MODE", "single_turn", fallbacks=("STS2_BENCH_CONVERSATION_MODE",)),
        help=(
            "Agent context-management mode. turn_chat keeps short chat history only within a player combat turn; "
            "combat_mdp_state uses structured current-turn combat facts without chat-history replay."
        ),
    )
    parser.add_argument(
        "--turn-chat-window",
        type=int,
        default=env_int("STS2_BENCH_TURN_CHAT_WINDOW", 4),
        help="Number of recent user/assistant turn-chat pairs to keep inside the current player turn",
    )
    parser.add_argument(
        "--turn-chat-update-mode",
        choices=["delta"],
        default=env_str("STS2_BENCH_TURN_CHAT_UPDATE_MODE", "delta"),
        help="How turn_chat renders follow-up messages after the first full state",
    )
    parser.add_argument(
        "--turn-chat-assistant-history",
        choices=["compact"],
        default=env_str("STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY", "compact"),
        help="How turn_chat stores assistant messages in history",
    )
    parser.add_argument(
        "--turn-check",
        "--turn-chat-plan",
        dest="turn_chat_plan",
        action=argparse.BooleanOptionalAction,
        default=env_bool(
            "STS2_BENCH_TURN_CHECK_ENABLED",
            env_bool("STS2_BENCH_TURN_CHAT_PLAN_ENABLED", False),
        ),
        help="Ask combat prompts to include a current-state turn_check tactical guardrail",
    )
    parser.add_argument(
        "--route-plan",
        action=argparse.BooleanOptionalAction,
        dest="route_plan_enabled",
        default=env_bool("STS2_BENCH_ROUTE_PLAN_ENABLED", True),
        help="Enable route-plan prompting and runner validation",
    )
    parser.add_argument(
        "--route-plan-enforcement",
        choices=["hard", "soft"],
        default=env_str("STS2_BENCH_ROUTE_PLAN_ENFORCEMENT", "hard"),
        help="Route-plan validation behavior: hard retries/stops; soft logs diagnostics only",
    )
    parser.add_argument(
        "--route-map-prompt-format",
        choices=["route_graph", "canonical"],
        default=env_str("STS2_BENCH_ROUTE_MAP_PROMPT_FORMAT", "canonical"),
        help="Map graph rendering used in LLM prompts; route_graph preserves the legacy nested JSON, canonical uses row-wise nodes plus an edge map",
    )
    parser.add_argument(
        "--reward-posterior-evaluator",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_REWARD_POSTERIOR_EVALUATOR", True),
        help="For coordinator card rewards, run an evidence-only LLM deck-state posterior evaluator before OCC chooses",
    )
    parser.add_argument(
        "--route-candidate-objective",
        choices=["none", "max_restsites", "max_restsites_low_risk"],
        default=env_str("STS2_BENCH_ROUTE_CANDIDATE_OBJECTIVE", "none"),
        help="Optionally auto-select map routes from enumerated complete visible route candidates instead of calling the LLM for map_select",
    )
    parser.add_argument(
        "--self-consistency",
        action=argparse.BooleanOptionalAction,
        dest="self_consistency_enabled",
        default=env_bool("STS2_BENCH_SELF_CONSISTENCY", False),
        help="Run extra diagnostic samples for selected decision types",
    )
    parser.add_argument(
        "--self-consistency-samples",
        type=int,
        default=env_int("STS2_BENCH_SELF_CONSISTENCY_SAMPLES", 3),
        help="Number of total self-consistency samples including the primary response",
    )
    parser.add_argument(
        "--self-consistency-decisions",
        default=env_str("STS2_BENCH_SELF_CONSISTENCY_DECISIONS", "map_select,card_reward"),
        help="Comma-separated decisions for self-consistency diagnostics",
    )
    parser.add_argument(
        "--self-consistency-mode",
        choices=["diagnostic", "majority_vote"],
        default=env_str("STS2_BENCH_SELF_CONSISTENCY_MODE", "diagnostic"),
        help="Whether self-consistency only logs votes or can apply majority_vote",
    )
    parser.add_argument(
        "--reflection-verifier",
        action=argparse.BooleanOptionalAction,
        dest="reflection_enabled",
        default=env_bool("STS2_BENCH_REFLECTION_VERIFIER", False),
        help="Run strict per-option reflection verifier diagnostics",
    )
    parser.add_argument(
        "--reflection-decisions",
        default=env_str("STS2_BENCH_REFLECTION_DECISIONS", "card_reward"),
        help="Comma-separated decisions for reflection verifier diagnostics",
    )
    parser.add_argument(
        "--reflection-mode",
        choices=["diagnostic", "verifier_vote"],
        default=env_str("STS2_BENCH_REFLECTION_MODE", "diagnostic"),
        help="Whether reflection only logs or can apply verifier_vote",
    )
    parser.add_argument(
        "--reflection-max-options",
        type=int,
        default=env_int("STS2_BENCH_REFLECTION_MAX_OPTIONS", 8),
        help="Maximum number of non-view candidate actions eligible for reflection",
    )
    parser.add_argument(
        "--action-prompt-order",
        choices=["default", "shuffle_map"],
        default=env_str("STS2_BENCH_ACTION_PROMPT_ORDER", "default"),
        help="Diagnostic legal-action display ordering mode",
    )
    parser.add_argument(
        "--action-prompt-order-seed",
        type=int,
        default=env_int("STS2_BENCH_ACTION_PROMPT_ORDER_SEED", 0),
        help="Seed for diagnostic legal-action display ordering",
    )
    parser.add_argument(
        "--character",
        default=env_str("STS2_BENCH_CHARACTER", "Ironclad"),
        choices=["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"],
    )
    parser.add_argument("--ascension", type=int, default=env_int("STS2_BENCH_ASCENSION", 0))
    parser.add_argument("--lang", default=env_str("STS2_BENCH_LANG", "en"), choices=["en", "zh"])
    parser.add_argument("--seeds", default=env_str("STS2_BENCH_SEEDS"), help="JSON seed file")
    parser.add_argument("--count", type=int, default=env_int("STS2_BENCH_COUNT", 1))
    parser.add_argument(
        "--max-steps",
        type=int,
        default=env_int("STS2_BENCH_MAX_STEPS", 2000),
        help="Safety cap on decision steps per run; set high for floor benchmark",
    )
    parser.add_argument(
        "--max-game-actions",
        type=int,
        default=env_int("STS2_BENCH_MAX_GAME_ACTIONS", 0),
        help="Optional cap on real game actions per run; benchmark-local view actions do not count",
    )
    parser.add_argument("--out", default=env_str("STS2_BENCH_OUT"), help="JSONL trace output path")
    parser.add_argument(
        "--log-level",
        choices=["metrics", "decisions", "full"],
        default=env_str("STS2_BENCH_LOG_LEVEL", "decisions"),
        help="JSONL detail level. metrics is smallest; full includes full state, prompts, and raw responses.",
    )
    parser.add_argument(
        "--print-prompts",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_PRINT_PROMPTS", False),
        help="Print the LLM prompt for every decision before acting",
    )
    parser.add_argument(
        "--print-model-output",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_PRINT_MODEL_OUTPUT", False),
        help="Print the raw agent/model output and selected action for every decision",
    )
    parser.add_argument(
        "--include-full-map",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_INCLUDE_FULL_MAP", False),
        help="Always fetch and render the full map at map_select decisions",
    )
    parser.add_argument(
        "--allow-repeat-views",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_ALLOW_REPEAT_VIEWS", False),
        help="Allow view deck/map actions again after that information has already been viewed in the current decision",
    )
    parser.add_argument(
        "--allow-custom-sequences",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_ALLOW_CUSTOM_SEQUENCES", True),
        help="Expose the model-proposed combat_custom_sequence gateway for LLM agents",
    )
    parser.add_argument(
        "--custom-sequence-exposure",
        choices=["safe", "eligible"],
        default=env_str("STS2_BENCH_CUSTOM_SEQUENCE_EXPOSURE", "safe"),
        help="safe keeps conservative gateway gating; eligible exposes combat_custom_sequence whenever the hand has a possible custom sequence entrypoint",
    )
    parser.add_argument(
        "--combat-card-sequences",
        action=argparse.BooleanOptionalAction,
        dest="combat_card_sequences_enabled",
        default=env_bool("STS2_BENCH_COMBAT_CARD_SEQUENCES", True),
        help="Allow model-generated combat_custom_sequence payloads and prompt rules",
    )
    parser.add_argument(
        "--precomputed-combat-card-sequences",
        action=argparse.BooleanOptionalAction,
        dest="precomputed_combat_card_sequences_enabled",
        default=env_bool("STS2_BENCH_PRECOMPUTED_COMBAT_CARD_SEQUENCES", False),
        help="Expose harness-precomputed combat_card_sequence legal actions",
    )
    parser.add_argument(
        "--combat-reasoning-pool",
        action=argparse.BooleanOptionalAction,
        dest="combat_reasoning_pool_enabled",
        default=env_bool("STS2_BENCH_COMBAT_REASONING_POOL", False),
        help="For combat_mdp_state CA prompts, keep compact same-turn reason/combat_notes/turn_check in the structured turn process",
    )
    parser.add_argument(
        "--combat-turn-board",
        action=argparse.BooleanOptionalAction,
        dest="combat_turn_board_enabled",
        default=env_bool("STS2_BENCH_COMBAT_TURN_BOARD", False),
        help="For combat_mdp_state CA prompts, keep compact engine-fact summaries from previous turns in the same combat",
    )
    parser.add_argument(
        "--trace-analysis-dir",
        default=env_str("STS2_BENCH_TRACE_ANALYSIS_DIR"),
        help=(
            "Directory for replay/report artifacts. Defaults to artifacts/trace_analysis/<trace stem> "
            "when trace analysis is enabled."
        ),
    )
    parser.add_argument(
        "--trace-analysis",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_TRACE_ANALYSIS", True),
        help="After the run, split the JSONL trace per seed and write replay/report artifacts",
    )
    args = parser.parse_args(argv)
    resolved_trace_analysis_dir = _resolve_trace_analysis_dir(
        out=args.out,
        trace_analysis_enabled=args.trace_analysis,
        trace_analysis_dir=args.trace_analysis_dir,
    )
    if args.trace_analysis and resolved_trace_analysis_dir is None:
        parser.error("--trace-analysis requires --out so the replay/report analyzer has a trace to read")
    args.trace_analysis_dir = str(resolved_trace_analysis_dir) if resolved_trace_analysis_dir is not None else None

    seeds = load_seeds(args.seeds, args.count)
    run_config = build_run_config(args, seeds, list(argv if argv is not None else sys.argv[1:]))
    logger = JsonlLogger(args.out)
    results = []
    try:
        run_config_event = {"type": "run_config", "config": run_config}
        print(json.dumps(run_config_event, ensure_ascii=False), flush=True)
        logger.write(run_config_event)

        agent = agent_from_args(
            args.agent,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            llm_timeout=args.llm_timeout,
            llm_max_retries=args.llm_max_retries,
            llm_fallback_mode=args.llm_fallback_mode,
            llm_temperature=args.llm_temperature,
            include_json_state=args.include_json_state,
            prompt_style=args.prompt_style,
            memory_enabled=args.memory_enabled,
            memory_window=args.memory_window,
            memory_mode=args.memory_mode,
            run_summary_enabled=args.run_summary,
            conversation_mode=args.context_mode,
            turn_chat_window=args.turn_chat_window,
            turn_chat_update_mode=args.turn_chat_update_mode,
            turn_chat_assistant_history=args.turn_chat_assistant_history,
            turn_chat_plan_enabled=args.turn_chat_plan,
            turn_check_enabled=args.turn_chat_plan,
            route_plan_enabled=args.route_plan_enabled,
            route_plan_enforcement=args.route_plan_enforcement,
            route_map_prompt_format=args.route_map_prompt_format,
            reward_posterior_evaluator_enabled=args.reward_posterior_evaluator,
            self_consistency_enabled=args.self_consistency_enabled,
            self_consistency_samples=args.self_consistency_samples,
            self_consistency_decisions=args.self_consistency_decisions,
            self_consistency_mode=args.self_consistency_mode,
            reflection_enabled=args.reflection_enabled,
            reflection_decisions=args.reflection_decisions,
            reflection_mode=args.reflection_mode,
            reflection_max_options=args.reflection_max_options,
            action_prompt_order=args.action_prompt_order,
            action_prompt_order_seed=args.action_prompt_order_seed,
            combat_card_sequences_enabled=args.combat_card_sequences_enabled,
            combat_reasoning_pool_enabled=args.combat_reasoning_pool_enabled,
            combat_turn_board_enabled=args.combat_turn_board_enabled,
            route_candidate_objective=(
                None if args.route_candidate_objective == "none" else args.route_candidate_objective
            ),
        )
        for seed in seeds:
            result = run_one(
                agent=agent,
                seed=seed,
                character=args.character,
                ascension=args.ascension,
                lang=args.lang,
                max_steps=args.max_steps,
                max_game_actions=args.max_game_actions or None,
                logger=logger,
                print_prompts=args.print_prompts,
                print_model_output=args.print_model_output,
                include_full_map=args.include_full_map,
                allow_repeat_views=args.allow_repeat_views,
                allow_custom_sequences=args.allow_custom_sequences,
                custom_sequence_exposure=args.custom_sequence_exposure,
                combat_card_sequences_enabled=args.combat_card_sequences_enabled,
                precomputed_combat_card_sequences_enabled=args.precomputed_combat_card_sequences_enabled,
                log_level=args.log_level,
                route_plan_enabled=args.route_plan_enabled,
                route_plan_enforcement=args.route_plan_enforcement,
                route_candidate_objective=(
                    None if args.route_candidate_objective == "none" else args.route_candidate_objective
                ),
            )
            results.append(result)
            print(json.dumps({"seed": seed, "result": result.to_dict()}, ensure_ascii=False), flush=True)
        exit_code = 0 if all(result.error is None for result in results) else 1
        print(json.dumps({"summary": summarize(results)}, ensure_ascii=False, indent=2))
    finally:
        logger.close()
    if args.trace_analysis_dir:
        analysis_index = analyze_benchmark_trace(args.out, args.trace_analysis_dir)
        print(json.dumps({"trace_analysis": analysis_index}, ensure_ascii=False, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
