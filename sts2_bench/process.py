"""Subprocess transport for the STS2 headless JSON protocol.

This module intentionally does not use the HTTP bridge in ``agent/``.  It
talks directly to ``Sts2Headless`` over stdin/stdout so benchmark and RL loops
can own process lifetime, logging, reset, and parallel workers cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .seeds import DEFAULT_TEST_SEED


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "src" / "Sts2Headless" / "Sts2Headless.csproj"
LIB_DIR = ROOT / "lib"
HEADLESS_DLL = ROOT / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0" / "Sts2Headless.dll"
LOCAL_DOTNET_DIR = ROOT / ".tools" / "dotnet"
LOCAL_DOTNET = LOCAL_DOTNET_DIR / ("dotnet.exe" if os.name == "nt" else "dotnet")
MAC_ARM_DOTNET = Path(os.path.expanduser("~/.dotnet-arm64/dotnet"))


class Sts2ProtocolError(RuntimeError):
    """Raised when the headless process exits or returns invalid protocol data."""


def find_dotnet() -> str:
    """Find a dotnet executable using the same local-first policy as tests."""

    candidates = [
        LOCAL_DOTNET,
        MAC_ARM_DOTNET,
        Path(os.path.expanduser("~/.dotnet/dotnet")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    dotnet = shutil.which("dotnet")
    if dotnet:
        return dotnet
    return str(MAC_ARM_DOTNET)


class Sts2Process:
    """Owns one headless STS2 process.

    The process can be reused across runs by calling ``reset`` then
    ``start_run``.  For heavy benchmark parallelism, create one instance per
    worker process rather than sharing this object across threads.
    """

    def __init__(
        self,
        *,
        dotnet: str | None = None,
        cwd: Path | str = ROOT,
        stderr: int | None = subprocess.DEVNULL,
        build_if_needed: bool = False,
    ) -> None:
        self.dotnet = dotnet or find_dotnet()
        self.cwd = Path(cwd)
        self.stderr = stderr
        self.build_if_needed = build_if_needed
        self.proc: subprocess.Popen[str] | None = None
        self.ready: dict[str, Any] | None = None

    def __enter__(self) -> "Sts2Process":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if LOCAL_DOTNET.is_file():
            env["DOTNET_ROOT"] = str(LOCAL_DOTNET_DIR)
            env["PATH"] = str(LOCAL_DOTNET_DIR) + os.pathsep + env.get("PATH", "")
        if (LIB_DIR / "sts2.dll").is_file():
            env.setdefault("STS2_LIB", str(LIB_DIR))
            env.setdefault("STS2_GAME_DIR", str(LIB_DIR))
        return env

    def _command(self) -> list[str]:
        if HEADLESS_DLL.is_file():
            return [self.dotnet, str(HEADLESS_DLL)]
        if self.build_if_needed:
            return [self.dotnet, "run", "--project", str(PROJECT)]
        return [self.dotnet, "run", "--no-build", "--project", str(PROJECT)]

    def start(self) -> dict[str, Any]:
        if self.proc and self.proc.poll() is None:
            return self.ready or {"type": "ready"}

        self.proc = subprocess.Popen(
            self._command(),
            cwd=self.cwd,
            env=self._env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        ready = self._read_json()
        if ready.get("type") != "ready":
            raise Sts2ProtocolError(f"Expected ready, got {ready!r}")
        self.ready = ready
        return ready

    def _read_json(self) -> dict[str, Any]:
        if not self.proc or not self.proc.stdout:
            raise Sts2ProtocolError("Process is not started")

        while True:
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                raise Sts2ProtocolError(f"EOF from Sts2Headless process, returncode={code}")
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise Sts2ProtocolError(f"Invalid JSON from Sts2Headless: {line[:500]}") from exc

    def send(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.proc or self.proc.poll() is not None:
            self.start()
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._read_json()

    def start_run(
        self,
        *,
        character: str = "Ironclad",
        seed: str | None = DEFAULT_TEST_SEED,
        ascension: int = 0,
        lang: str = "en",
    ) -> dict[str, Any]:
        command: dict[str, Any] = {
            "cmd": "start_run",
            "character": character,
            "ascension": ascension,
            "lang": lang,
        }
        if seed is not None:
            command["seed"] = seed
        return self.send(command)

    def action(self, action: str, **args: Any) -> dict[str, Any]:
        command: dict[str, Any] = {"cmd": "action", "action": action}
        if args:
            command["args"] = args
        return self.send(command)

    def reset(self) -> dict[str, Any]:
        return self.send({"cmd": "reset"})

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        self.ready = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write('{"cmd":"quit"}\n')
                    proc.stdin.flush()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    """Small smoke-test entry point: start, print ready, quit."""

    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m sts2_bench process-smoke",
        description="Start Sts2Headless, print its ready message, and quit.",
    )
    if any(arg in {"-h", "--help"} for arg in args):
        parser.print_help()
        return 0
    parser.parse_args(args)

    with Sts2Process(stderr=None) as proc:
        print(json.dumps(proc.ready, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
