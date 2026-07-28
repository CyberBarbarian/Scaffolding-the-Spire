"""Subprocess transport for the STS2 headless JSON protocol.

It talks directly to the authoritative ``sts2-cli`` ``Sts2Headless`` runtime
over stdin/stdout so benchmark and RL loops can own process lifetime, logging,
reset, and parallel workers cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .seeds import DEFAULT_TEST_SEED


BENCH_ROOT = Path(__file__).resolve().parents[1]
STS2_CLI_ROOT_ENV = "STS2_CLI_ROOT"


def _runtime_root() -> Path:
    configured = os.environ.get(STS2_CLI_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (BENCH_ROOT.parent / "sts2-cli").resolve()


ROOT = _runtime_root()
PROJECT = ROOT / "src" / "Sts2Headless" / "Sts2Headless.csproj"
LIB_DIR = ROOT / "lib"
HEADLESS_DLL = ROOT / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0" / "Sts2Headless.dll"
LOCAL_DOTNET_DIR = ROOT / ".tools" / "dotnet"
LOCAL_DOTNET = LOCAL_DOTNET_DIR / ("dotnet.exe" if os.name == "nt" else "dotnet")
MAC_ARM_DOTNET = Path(os.path.expanduser("~/.dotnet-arm64/dotnet"))
FROZEN_RUNTIME_ENV = "STS2_BENCH_FROZEN_RUNTIME"
DEFAULT_STARTUP_TIMEOUT_SEC = 30.0
DEFAULT_RESPONSE_TIMEOUT_SEC = 90.0
_STDOUT_EOF = object()


class Sts2ProtocolError(RuntimeError):
    """Raised when the headless process exits or returns invalid protocol data."""


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    """Force-stop ``proc`` and any child process created by ``dotnet run``."""

    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
        else:  # pragma: no cover - Windows is the active development platform
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass

    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


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
        stderr: int | None = None,
        build_if_needed: bool = False,
        startup_timeout_sec: float = DEFAULT_STARTUP_TIMEOUT_SEC,
        response_timeout_sec: float = DEFAULT_RESPONSE_TIMEOUT_SEC,
    ) -> None:
        if startup_timeout_sec <= 0 or response_timeout_sec <= 0:
            raise ValueError("Process protocol timeouts must be positive")
        self.dotnet = dotnet or find_dotnet()
        self.cwd = Path(cwd)
        self.stderr = stderr
        self.build_if_needed = build_if_needed
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.response_timeout_sec = float(response_timeout_sec)
        self.proc: subprocess.Popen[str] | None = None
        self.ready: dict[str, Any] | None = None
        self._stdout_queue: queue.Queue[Any] | None = None
        self._stdout_reader: threading.Thread | None = None

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
        frozen_runtime = env.get(FROZEN_RUNTIME_ENV) == "1"
        if frozen_runtime and not (LIB_DIR / "sts2.dll").is_file():
            raise Sts2ProtocolError(f"Frozen game runtime is missing: {LIB_DIR / 'sts2.dll'}")
        if frozen_runtime:
            env["STS2_LIB"] = str(LIB_DIR)
            env["STS2_GAME_DIR"] = str(LIB_DIR)
        elif (LIB_DIR / "sts2.dll").is_file():
            env.setdefault("STS2_LIB", str(LIB_DIR))
            env.setdefault("STS2_GAME_DIR", str(LIB_DIR))
        return env

    def _command(self) -> list[str]:
        if HEADLESS_DLL.is_file():
            return [self.dotnet, str(HEADLESS_DLL)]
        if os.environ.get(FROZEN_RUNTIME_ENV) == "1":
            raise Sts2ProtocolError(f"Frozen headless runtime is missing: {HEADLESS_DLL}")
        if self.build_if_needed:
            return [self.dotnet, "run", "--project", str(PROJECT)]
        return [self.dotnet, "run", "--no-build", "--project", str(PROJECT)]

    def start(self) -> dict[str, Any]:
        if self.proc:
            if self.proc.poll() is None:
                return self.ready or {"type": "ready"}
            self.close()

        if not ROOT.is_dir():
            raise Sts2ProtocolError(
                f"Canonical sts2-cli runtime is missing: {ROOT}. "
                f"Set {STS2_CLI_ROOT_ENV} to the authoritative checkout or frozen runtime."
            )

        creation: dict[str, Any] = {}
        if os.name == "nt":
            creation["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:  # pragma: no cover - Windows is the active development platform
            creation["start_new_session"] = True
        try:
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
                **creation,
            )
            self._start_stdout_reader(self.proc)
            ready = self._read_json(timeout_sec=self.startup_timeout_sec, phase="ready")
            if ready.get("type") != "ready":
                raise Sts2ProtocolError(f"Expected ready, got {ready!r}")
            self.ready = ready
            return ready
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _start_stdout_reader(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            raise Sts2ProtocolError("Sts2Headless stdout pipe is missing")
        output: queue.Queue[Any] = queue.Queue()

        def pump() -> None:
            try:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        output.put(_STDOUT_EOF)
                        return
                    output.put(line)
            except BaseException as exc:  # noqa: BLE001 - surface reader failure to the owner
                output.put(exc)

        reader = threading.Thread(
            target=pump,
            name=f"sts2-stdout-{getattr(proc, 'pid', 'unknown')}",
            daemon=True,
        )
        self._stdout_queue = output
        self._stdout_reader = reader
        reader.start()

    def _read_json(self, *, timeout_sec: float, phase: str) -> dict[str, Any]:
        if not self.proc or not self.proc.stdout:
            raise Sts2ProtocolError("Process is not started")
        if self._stdout_queue is None:
            self._start_stdout_reader(self.proc)

        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Sts2ProtocolError(
                    f"Timed out waiting for Sts2Headless {phase} after {timeout_sec:g}s"
                )
            try:
                item = self._stdout_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise Sts2ProtocolError(
                    f"Timed out waiting for Sts2Headless {phase} after {timeout_sec:g}s"
                ) from exc
            if item is _STDOUT_EOF:
                code = self.proc.poll()
                raise Sts2ProtocolError(f"EOF from Sts2Headless process, returncode={code}")
            if isinstance(item, BaseException):
                raise Sts2ProtocolError(f"Failed reading Sts2Headless stdout: {item}") from item
            line = str(item).strip()
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
        try:
            self.proc.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            return self._read_json(timeout_sec=self.response_timeout_sec, phase="response")
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

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
        reader = self._stdout_reader
        self.proc = None
        self.ready = None
        self._stdout_queue = None
        self._stdout_reader = None
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
        except (OSError, subprocess.TimeoutExpired):
            _terminate_process_tree(proc)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            if reader is not None and reader is not threading.current_thread():
                reader.join(timeout=5)
                if reader.is_alive():
                    raise Sts2ProtocolError("Sts2Headless stdout reader did not stop during process cleanup")


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
