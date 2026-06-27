import pytest


def test_unified_cli_dispatches_run_command(monkeypatch):
    from sts2_bench import __main__ as cli

    calls = []

    def fake_run_module_main(module_name, argv):
        calls.append((module_name, argv))
        return 37

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)

    assert cli.main(["run", "--agent", "random", "--count", "1"]) == 37
    assert calls == [
        (
            "sts2_bench.run_benchmark",
            ["--agent", "random", "--count", "1"],
        )
    ]


def test_unified_cli_sets_subcommand_prog(monkeypatch):
    import sys

    from sts2_bench import __main__ as cli

    seen_prog = []

    def fake_run_module_main(module_name, argv):
        del module_name, argv
        seen_prog.append(sys.argv[0])
        return 0

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)

    assert cli.main(["run", "--help"]) == 0
    assert seen_prog == ["python -m sts2_bench run"]


def test_run_module_main_uses_subcommand_prog_for_argparse(monkeypatch):
    import argparse
    import sys
    from types import SimpleNamespace

    from sts2_bench import __main__ as cli

    seen_prog = []

    def fake_main(argv):
        del argv
        seen_prog.append(argparse.ArgumentParser().prog)
        return 0

    class PackageMainSpec:
        name = "sts2_bench.__main__"

    monkeypatch.setitem(sys.modules, "fake_sts2_cli_module", SimpleNamespace(main=fake_main))
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", PackageMainSpec(), raising=False)
    monkeypatch.setattr(sys, "argv", ["python -m sts2_bench run"])

    assert cli._run_module_main("fake_sts2_cli_module", []) == 0
    assert seen_prog == ["python -m sts2_bench run"]


def test_unified_cli_dispatches_experiment_commands(monkeypatch):
    from sts2_bench import __main__ as cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_run_module_main",
        lambda module_name, argv: calls.append((module_name, argv)) or 0,
    )

    assert cli.main(["card-choice", "--out", "results/card_choice.jsonl"]) == 0
    assert cli.main(["card-choice-bias", "--out", "results/bias.jsonl"]) == 0
    assert cli.main(["card-choice-trace", "--trace", "trace.jsonl", "--out", "results/replay.jsonl"]) == 0
    assert cli.main(["combat-snapshot", "extract", "--trace", "trace.jsonl", "--start-step", "3", "--out", "snapshot.json"]) == 0
    assert cli.main(["trace-analysis", "trace.jsonl", "--out-dir", "artifacts/trace_analysis/run"]) == 0
    assert cli.main(["route-smoke", "--seed", "smoke"]) == 0
    assert cli.main(["process-smoke"]) == 0

    assert calls == [
        ("sts2_bench.card_choice_eval", ["--out", "results/card_choice.jsonl"]),
        ("sts2_bench.card_choice_bias_eval", ["--out", "results/bias.jsonl"]),
        ("sts2_bench.card_choice_trace_eval", ["--trace", "trace.jsonl", "--out", "results/replay.jsonl"]),
        ("sts2_bench.combat_snapshot", ["extract", "--trace", "trace.jsonl", "--start-step", "3", "--out", "snapshot.json"]),
        ("sts2_bench.trace_analysis", ["trace.jsonl", "--out-dir", "artifacts/trace_analysis/run"]),
        ("sts2_bench.route_plan_smoke", ["--seed", "smoke"]),
        ("sts2_bench.process", []),
    ]


def test_unified_cli_help_lists_public_commands(capsys):
    from sts2_bench import __main__ as cli

    assert cli.main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "run" in output
    assert "card-choice" in output
    assert "combat-snapshot" in output
    assert "process-smoke" in output
    assert "route-smoke" in output


def test_unified_cli_rejects_unknown_command():
    from sts2_bench import __main__ as cli

    with pytest.raises(SystemExit):
        cli.main(["unknown-command"])


def test_process_smoke_help_does_not_start_runtime(monkeypatch, capsys):
    from sts2_bench import process

    class FailingProcess:
        def __init__(self, *args, **kwargs):
            raise AssertionError("help should not start Sts2Process")

    monkeypatch.setattr(process, "Sts2Process", FailingProcess)

    assert process.main(["--help"]) == 0
    assert "process-smoke" in capsys.readouterr().out


def test_route_plan_smoke_module_exposes_cli_main():
    from sts2_bench import route_plan_smoke

    assert callable(route_plan_smoke.main)
