"""Unified command-line entrypoint for STS2 benchmark experiments."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Command:
    module: str
    summary: str


COMMANDS: dict[str, Command] = {
    "run": Command("sts2_bench.run_benchmark", "Run headless benchmark policies."),
    "card-choice": Command("sts2_bench.card_choice_eval", "Run synthetic card reward choice evaluation."),
    "card-choice-bias": Command("sts2_bench.card_choice_bias_eval", "Run card reward bias probes."),
    "card-choice-trace": Command("sts2_bench.card_choice_trace_eval", "Replay card reward choices from traces."),
    "combat-snapshot": Command("sts2_bench.combat_snapshot", "Extract, verify, or run combat snapshots."),
    "trace-analysis": Command("sts2_bench.trace_analysis", "Generate replay and failure-analysis artifacts."),
    "process-smoke": Command("sts2_bench.process", "Start the headless process and print its ready message."),
}


def _build_parser() -> argparse.ArgumentParser:
    command_help = "\n".join(f"  {name:<18} {command.summary}" for name, command in COMMANDS.items())
    return argparse.ArgumentParser(
        prog="python -m sts2_bench",
        description="Unified STS2 benchmark experiment runner.",
        epilog="Commands:\n" + command_help + "\n\nUse `python -m sts2_bench <command> --help` for command options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def _run_module_main(module_name: str, argv: list[str]) -> int:
    module = import_module(module_name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise RuntimeError(f"{module_name} does not expose main(argv)")

    main_module = sys.modules.get("__main__")
    old_spec = getattr(main_module, "__spec__", None) if main_module is not None else None
    if main_module is not None:
        main_module.__spec__ = None
    try:
        return int(main(argv) or 0)
    finally:
        if main_module is not None:
            main_module.__spec__ = old_spec


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command_name = args[0]
    command = COMMANDS.get(command_name)
    if command is None:
        parser.error(f"unknown command {command_name!r}; choose one of: {', '.join(COMMANDS)}")

    old_argv = sys.argv
    sys.argv = [f"{parser.prog} {command_name}", *args[1:]]
    try:
        return _run_module_main(command.module, args[1:])
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
