"""Regression tests for repo-local toolchain discovery."""

from pathlib import Path

from conftest import LOCAL_DOTNET_DIR, STS2_CLI_ROOT
from sts2_bench import process as bench_process


def test_pytest_fixture_uses_sts2_cli_local_tools():
    assert LOCAL_DOTNET_DIR == STS2_CLI_ROOT / ".tools" / "dotnet"


def test_benchmark_process_uses_sts2_cli_local_tools():
    root = Path(__file__).resolve().parents[1]
    assert bench_process.LOCAL_DOTNET_DIR == root.parent / "sts2-cli" / ".tools" / "dotnet"
