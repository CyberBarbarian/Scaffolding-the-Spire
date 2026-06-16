"""Command-line runner for STS2 benchmark policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .runner import JsonlLogger, agent_from_args, run_one, summarize


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV = ROOT / ".env"
TRACKED_CONFIG = ROOT / "sts2_bench" / "benchmark.yaml"
CONFIG_ENV_KEYS = {
    "agent": "STS2_BENCH_AGENT",
    "base_url": "STS2_BENCH_BASE_URL",
    "model": "STS2_BENCH_MODEL",
    "llm_timeout": "STS2_BENCH_LLM_TIMEOUT",
    "llm_max_retries": "STS2_BENCH_LLM_MAX_RETRIES",
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
    "out": "STS2_BENCH_OUT",
    "log_level": "STS2_BENCH_LOG_LEVEL",
    "print_prompts": "STS2_BENCH_PRINT_PROMPTS",
    "print_model_output": "STS2_BENCH_PRINT_MODEL_OUTPUT",
    "include_full_map": "STS2_BENCH_INCLUDE_FULL_MAP",
    "allow_repeat_views": "STS2_BENCH_ALLOW_REPEAT_VIEWS",
}
NESTED_CONFIG_ENV_KEYS = {
    ("context_management", "mode"): "STS2_BENCH_CONTEXT_MODE",
    ("context_management", "turn_chat", "window"): "STS2_BENCH_TURN_CHAT_WINDOW",
    ("context_management", "turn_chat", "update_mode"): "STS2_BENCH_TURN_CHAT_UPDATE_MODE",
    ("context_management", "turn_chat", "assistant_history"): "STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY",
    ("context_management", "turn_chat", "plan", "enabled"): "STS2_BENCH_TURN_CHAT_PLAN_ENABLED",
    ("experience_memory", "enabled"): "STS2_BENCH_EXPERIENCE_MEMORY_ENABLED",
    ("experience_memory", "mode"): "STS2_BENCH_EXPERIENCE_MEMORY_MODE",
    ("experience_memory", "store_path"): "STS2_BENCH_EXPERIENCE_MEMORY_STORE_PATH",
    ("experience_memory", "top_k"): "STS2_BENCH_EXPERIENCE_MEMORY_TOP_K",
    ("experience_memory", "token_budget"): "STS2_BENCH_EXPERIENCE_MEMORY_TOKEN_BUDGET",
    ("experience_memory", "min_confidence"): "STS2_BENCH_EXPERIENCE_MEMORY_MIN_CONFIDENCE",
    ("experience_memory", "include_global"): "STS2_BENCH_EXPERIENCE_MEMORY_INCLUDE_GLOBAL",
    ("experience_memory", "allow_cross_character"): "STS2_BENCH_EXPERIENCE_MEMORY_ALLOW_CROSS_CHARACTER",
    ("experience_memory", "allow_cross_ascension"): "STS2_BENCH_EXPERIENCE_MEMORY_ALLOW_CROSS_ASCENSION",
    ("experience_memory", "event_trace_path"): "STS2_BENCH_EXPERIENCE_EVENT_TRACE_PATH",
    ("experience_memory", "auto_learn", "enabled"): "STS2_BENCH_EXPERIENCE_AUTO_LEARN_ENABLED",
    ("experience_memory", "auto_learn", "trigger"): "STS2_BENCH_EXPERIENCE_AUTO_LEARN_TRIGGER",
    ("experience_memory", "auto_learn", "candidate_path"): "STS2_BENCH_EXPERIENCE_AUTO_LEARN_CANDIDATE_PATH",
    (
        "experience_memory",
        "auto_learn",
        "max_candidates_per_span",
    ): "STS2_BENCH_EXPERIENCE_AUTO_LEARN_MAX_CANDIDATES_PER_SPAN",
    ("experience_memory", "auto_learn", "max_events_per_span"): "STS2_BENCH_EXPERIENCE_AUTO_LEARN_MAX_EVENTS_PER_SPAN",
    ("experience_memory", "auto_learn", "min_confidence"): "STS2_BENCH_EXPERIENCE_AUTO_LEARN_MIN_CONFIDENCE",
    ("experience_memory", "auto_learn", "promote"): "STS2_BENCH_EXPERIENCE_AUTO_LEARN_PROMOTE",
    ("experience_memory", "auto_learn", "promote_on_death"): "STS2_BENCH_EXPERIENCE_AUTO_LEARN_PROMOTE_ON_DEATH",
}


def load_seeds(path: str | None, count: int) -> list[str]:
    if not path:
        return [f"bench_{i:04d}" for i in range(count)]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        seeds = []
        for item in data:
            if isinstance(item, str):
                seeds.append(item)
            elif isinstance(item, dict) and item.get("seed"):
                seeds.append(str(item["seed"]))
            else:
                raise ValueError(f"Unsupported seed entry: {item!r}")
        return seeds
    raise ValueError("Seed file must be a JSON list of strings or objects with a seed field")


def load_env_file(path: Path) -> None:
    """Load a simple env file without adding a runtime dependency.

    Existing environment variables win over file values. CLI arguments then win
    over both because argparse uses these values only as defaults.
    """

    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
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


def load_benchmark_config(path: Path = TRACKED_CONFIG) -> None:
    """Load tracked benchmark defaults from YAML.

    Values are mirrored into the existing STS2_BENCH_* environment names so
    CLI defaults and shell overrides stay backward compatible.
    """

    if not path.is_file():
        return

    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OmegaConf is required to read sts2_bench/benchmark.yaml. "
            "Install it with `python3 -m pip install omegaconf`."
        ) from exc

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

    return {
        "argv": _redact_argv(argv),
        "agent": {
            "type": args.agent,
            "base_url": args.base_url,
            "model": args.model,
            "llm_timeout": args.llm_timeout,
            "llm_max_retries": args.llm_max_retries,
            "llm_temperature": args.llm_temperature,
            "prompt_style": args.prompt_style,
            "include_json_state": args.include_json_state,
            "memory": {
                "enabled": args.memory_enabled,
                "window": args.memory_window,
                "mode": args.memory_mode,
            },
            "run_summary_enabled": args.run_summary,
            "context_management": {
                "mode": args.context_mode,
                "turn_chat": {
                    "window": args.turn_chat_window,
                    "update_mode": args.turn_chat_update_mode,
                    "assistant_history": args.turn_chat_assistant_history,
                    "plan": {
                        "enabled": args.turn_chat_plan,
                    },
                },
            },
            "experience_memory": {
                "enabled": args.experience_memory_enabled,
                "mode": args.experience_memory_mode,
                "store_path": args.experience_memory_store_path,
                "store_digest": _file_sha256(args.experience_memory_store_path),
                "top_k": args.experience_memory_top_k,
                "token_budget": args.experience_memory_token_budget,
                "min_confidence": args.experience_memory_min_confidence,
                "include_global": args.experience_memory_include_global,
                "allow_cross_character": args.experience_memory_allow_cross_character,
                "allow_cross_ascension": args.experience_memory_allow_cross_ascension,
                "event_trace_path": args.experience_event_trace_path,
                "auto_learn": {
                    "enabled": args.experience_auto_learn_enabled,
                    "trigger": args.experience_auto_learn_trigger,
                    "candidate_path": args.experience_auto_learn_candidate_path,
                    "max_candidates_per_span": args.experience_auto_learn_max_candidates_per_span,
                    "max_events_per_span": args.experience_auto_learn_max_events_per_span,
                    "min_confidence": args.experience_auto_learn_min_confidence,
                    "promote": args.experience_auto_learn_promote,
                    "promote_on_death": args.experience_auto_learn_promote_on_death,
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
            "include_full_map": args.include_full_map,
            "allow_repeat_views": args.allow_repeat_views,
        },
        "logging": {
            "out": args.out,
            "log_level": args.log_level,
            "print_prompts": args.print_prompts,
            "print_model_output": args.print_model_output,
        },
    }


def _file_sha256(path: str | None) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    load_benchmark_env()

    parser = argparse.ArgumentParser(description="Run STS2 headless benchmark policies.")
    parser.add_argument("--agent", choices=["random", "llm"], default=env_str("STS2_BENCH_AGENT", "random"))
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
        choices=["single_turn", "turn_chat"],
        default=env_str("STS2_BENCH_CONTEXT_MODE", "single_turn", fallbacks=("STS2_BENCH_CONVERSATION_MODE",)),
        help="Agent context-management mode. turn_chat keeps short chat history only within a player combat turn.",
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
        "--turn-chat-plan",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_TURN_CHAT_PLAN_ENABLED", False),
        help="Ask the first turn_chat response in each player turn to include an advisory turn_plan",
    )
    parser.add_argument(
        "--experience-memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_EXPERIENCE_MEMORY_ENABLED", False),
        help="Use frozen experience-memory retrieval in the LLM prompt",
    )
    parser.add_argument(
        "--experience-memory-mode",
        choices=["frozen", "observe"],
        default=env_str("STS2_BENCH_EXPERIENCE_MEMORY_MODE", "frozen"),
    )
    parser.add_argument(
        "--experience-memory-store-path",
        default=env_str("STS2_BENCH_EXPERIENCE_MEMORY_STORE_PATH", "sts2_bench/experience/memory/active.jsonl"),
    )
    parser.add_argument(
        "--experience-memory-top-k",
        type=int,
        default=env_int("STS2_BENCH_EXPERIENCE_MEMORY_TOP_K", 5),
    )
    parser.add_argument(
        "--experience-memory-token-budget",
        type=int,
        default=env_int("STS2_BENCH_EXPERIENCE_MEMORY_TOKEN_BUDGET", 700),
    )
    parser.add_argument(
        "--experience-memory-min-confidence",
        type=float,
        default=env_float("STS2_BENCH_EXPERIENCE_MEMORY_MIN_CONFIDENCE", 0.35),
    )
    parser.add_argument(
        "--experience-memory-include-global",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_EXPERIENCE_MEMORY_INCLUDE_GLOBAL", True),
    )
    parser.add_argument(
        "--experience-memory-allow-cross-character",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_EXPERIENCE_MEMORY_ALLOW_CROSS_CHARACTER", False),
    )
    parser.add_argument(
        "--experience-memory-allow-cross-ascension",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_EXPERIENCE_MEMORY_ALLOW_CROSS_ASCENSION", True),
    )
    parser.add_argument(
        "--experience-event-trace-path",
        default=env_str("STS2_BENCH_EXPERIENCE_EVENT_TRACE_PATH", ""),
        help="Optional compact event trace JSONL path for offline reflection",
    )
    parser.add_argument(
        "--experience-auto-learn-enabled",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_EXPERIENCE_AUTO_LEARN_ENABLED", False),
        help="Automatically reflect and update experience memory during the run",
    )
    parser.add_argument(
        "--experience-auto-learn-trigger",
        choices=["act_end", "act_end_or_death"],
        default=env_str("STS2_BENCH_EXPERIENCE_AUTO_LEARN_TRIGGER", "act_end"),
        help="When automatic experience learning runs",
    )
    parser.add_argument(
        "--experience-auto-learn-candidate-path",
        default=env_str("STS2_BENCH_EXPERIENCE_AUTO_LEARN_CANDIDATE_PATH", "sts2_bench/experience/memory/candidates.jsonl"),
    )
    parser.add_argument(
        "--experience-auto-learn-max-candidates-per-span",
        type=int,
        default=env_int("STS2_BENCH_EXPERIENCE_AUTO_LEARN_MAX_CANDIDATES_PER_SPAN", 2),
    )
    parser.add_argument(
        "--experience-auto-learn-max-events-per-span",
        type=int,
        default=env_int("STS2_BENCH_EXPERIENCE_AUTO_LEARN_MAX_EVENTS_PER_SPAN", 40),
    )
    parser.add_argument(
        "--experience-auto-learn-min-confidence",
        type=float,
        default=env_float("STS2_BENCH_EXPERIENCE_AUTO_LEARN_MIN_CONFIDENCE", 0.35),
    )
    parser.add_argument(
        "--experience-auto-learn-promote",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_EXPERIENCE_AUTO_LEARN_PROMOTE", True),
        help="Promote candidates generated by automatic learning into the active store",
    )
    parser.add_argument(
        "--experience-auto-learn-promote-on-death",
        action=argparse.BooleanOptionalAction,
        default=env_bool("STS2_BENCH_EXPERIENCE_AUTO_LEARN_PROMOTE_ON_DEATH", False),
        help="Also promote death-triggered automatic learning into the active store",
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
    args = parser.parse_args(argv)

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
            experience_memory_enabled=args.experience_memory_enabled,
            experience_memory_mode=args.experience_memory_mode,
            experience_memory_store_path=args.experience_memory_store_path,
            experience_memory_top_k=args.experience_memory_top_k,
            experience_memory_token_budget=args.experience_memory_token_budget,
            experience_memory_min_confidence=args.experience_memory_min_confidence,
            experience_memory_include_global=args.experience_memory_include_global,
            experience_memory_allow_cross_character=args.experience_memory_allow_cross_character,
            experience_memory_allow_cross_ascension=args.experience_memory_allow_cross_ascension,
            experience_event_trace_path=args.experience_event_trace_path,
            experience_auto_learn_enabled=args.experience_auto_learn_enabled,
            experience_auto_learn_trigger=args.experience_auto_learn_trigger,
            experience_auto_learn_candidate_path=args.experience_auto_learn_candidate_path,
            experience_auto_learn_max_candidates_per_span=args.experience_auto_learn_max_candidates_per_span,
            experience_auto_learn_max_events_per_span=args.experience_auto_learn_max_events_per_span,
            experience_auto_learn_min_confidence=args.experience_auto_learn_min_confidence,
            experience_auto_learn_promote=args.experience_auto_learn_promote,
            experience_auto_learn_promote_on_death=args.experience_auto_learn_promote_on_death,
        )
        for seed in seeds:
            result = run_one(
                agent=agent,
                seed=seed,
                character=args.character,
                ascension=args.ascension,
                lang=args.lang,
                max_steps=args.max_steps,
                logger=logger,
                print_prompts=args.print_prompts,
                print_model_output=args.print_model_output,
                include_full_map=args.include_full_map,
                allow_repeat_views=args.allow_repeat_views,
                log_level=args.log_level,
            )
            results.append(result)
            print(json.dumps({"seed": seed, "result": result.to_dict()}, ensure_ascii=False), flush=True)
        print(json.dumps({"summary": summarize(results)}, ensure_ascii=False, indent=2))
        return 0 if all(result.error is None for result in results) else 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
