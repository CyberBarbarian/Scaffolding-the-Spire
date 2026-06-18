"""Regression tests for Godot presentation stubs."""

from pathlib import Path
import subprocess

from conftest import DOTNET, STS2_CLI_ROOT


GODOT_STUBS_PROJECT = str(STS2_CLI_ROOT / "src" / "GodotStubs" / "GodotStubs.csproj")


def test_gpu_particles_2d_exposes_known_vfx_properties():
    source = Path(__file__).resolve().parents[1] / "src" / "GodotStubs" / "UI.cs"
    text = source.read_text(encoding="utf-8")

    assert "public int Amount { get; set; }" in text
    assert "public double Lifetime { get; set; }" in text
    assert "public float Explosiveness { get; set; }" in text


def test_godot_stubs_build_without_known_stub_warnings():
    result = subprocess.run(
        [DOTNET, "build", GODOT_STUBS_PROJECT, "--no-restore", "--no-incremental"],
        cwd=STS2_CLI_ROOT,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    output = result.stdout

    assert result.returncode == 0, output
    assert "CS0108" not in output
    assert "CS8625" not in output
    assert "CS0067" not in output
