from pathlib import Path

from sts2_bench import process


BENCH_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RUNTIME = BENCH_ROOT.parent / "sts2-cli"


def test_benchmark_has_no_duplicate_executable_runtime():
    forbidden = [
        BENCH_ROOT / "python" / "play.py",
        BENCH_ROOT / "python" / "play_full_run.py",
        BENCH_ROOT / "src" / "Sts2Headless" / "RunSimulator.cs",
        BENCH_ROOT / "launch.py",
        BENCH_ROOT / "sts2-cli.bat",
        BENCH_ROOT / "sts2-cli-zh.bat",
        BENCH_ROOT / "agent" / "sts2_bridge.py",
        BENCH_ROOT / "sts2_bench" / "sl_full_run.py",
    ]

    assert all(not path.exists() for path in forbidden)


def test_process_transport_targets_authoritative_runtime():
    assert process.ROOT == CANONICAL_RUNTIME.resolve()
    assert process.PROJECT == CANONICAL_RUNTIME.resolve() / "src" / "Sts2Headless" / "Sts2Headless.csproj"
    assert process.PROJECT.is_file()
