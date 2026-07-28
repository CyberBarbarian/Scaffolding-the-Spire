import threading

import pytest

import sts2_bench.process as process
from sts2_bench.default_v1_experiment import (
    ACTIVE_STS2_CLI_ENV,
    DefaultV1ExperimentError,
    PYTHON_ENV_CONTRACT,
    _require_active_runtime_binding,
)


def test_default_formal_runtime_requires_active_worktree_library_paths(tmp_path, monkeypatch):
    expected_cli = tmp_path / "sts2-cli"
    expected_cli.mkdir(parents=True)
    expected_lib = expected_cli / "lib"
    expected_lib.mkdir(parents=True)
    monkeypatch.setenv(ACTIVE_STS2_CLI_ENV, str(expected_cli))
    monkeypatch.setenv("STS2_LIB", str(expected_lib))
    monkeypatch.setenv("STS2_GAME_DIR", str(expected_lib))
    for name, value in PYTHON_ENV_CONTRACT.items():
        monkeypatch.setenv(name, value)

    _require_active_runtime_binding(expected_cli)

    monkeypatch.setenv("STS2_LIB", str(tmp_path / "external"))
    with pytest.raises(DefaultV1ExperimentError, match="STS2_LIB"):
        _require_active_runtime_binding(expected_cli)


def test_frozen_process_overrides_ambient_runtime_paths(tmp_path, monkeypatch):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "sts2.dll").write_bytes(b"runtime")
    monkeypatch.setattr(process, "LIB_DIR", lib_dir)
    monkeypatch.setattr(process, "LOCAL_DOTNET", tmp_path / "missing-dotnet")
    monkeypatch.setenv(process.FROZEN_RUNTIME_ENV, "1")
    monkeypatch.setenv("STS2_LIB", "C:/external/lib")
    monkeypatch.setenv("STS2_GAME_DIR", "C:/external/game")

    env = process.Sts2Process()._env()

    assert env["STS2_LIB"] == str(lib_dir)
    assert env["STS2_GAME_DIR"] == str(lib_dir)


def test_frozen_process_refuses_missing_prebuilt_headless_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(process, "HEADLESS_DLL", tmp_path / "missing" / "Sts2Headless.dll")
    monkeypatch.setenv(process.FROZEN_RUNTIME_ENV, "1")

    with pytest.raises(process.Sts2ProtocolError, match="Frozen headless runtime is missing"):
        process.Sts2Process()._command()


def test_close_reaps_process_and_closes_all_owned_pipes():
    class Pipe:
        def __init__(self):
            self.closed = False

        def write(self, _value):
            return None

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class Proc:
        def __init__(self):
            self.stdin = Pipe()
            self.stdout = Pipe()
            self.stderr = Pipe()

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    owner = process.Sts2Process()
    child = Proc()
    owner.proc = child

    owner.close()

    assert owner.proc is None
    assert child.stdin.closed
    assert child.stdout.closed
    assert child.stderr.closed


def test_start_failure_reaps_spawned_process_and_clears_owner(monkeypatch):
    class Pipe:
        def __init__(self, line=""):
            self.line = line
            self.closed = False

        def readline(self):
            line, self.line = self.line, ""
            return line

        def write(self, _value):
            return None

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class Proc:
        def __init__(self):
            self.stdin = Pipe()
            self.stdout = Pipe('{"type":"unexpected"}\n')
            self.stderr = Pipe()
            self.waited = False

        def poll(self):
            return None

        def wait(self, timeout):
            self.waited = True
            return 0

    child = Proc()
    owner = process.Sts2Process()
    monkeypatch.setattr(owner, "_command", lambda: ["dotnet", "headless.dll"])
    monkeypatch.setattr(owner, "_env", lambda: {})
    monkeypatch.setattr(process.subprocess, "Popen", lambda *_args, **_kwargs: child)

    with pytest.raises(process.Sts2ProtocolError, match="Expected ready"):
        with owner:
            pytest.fail("failed __enter__ must not yield the process")

    assert owner.proc is None
    assert owner.ready is None
    assert child.waited
    assert child.stdin.closed
    assert child.stdout.closed
    assert child.stderr.closed


def test_start_timeout_reaps_process_and_joins_stdout_reader(monkeypatch):
    release = threading.Event()

    class Pipe:
        def __init__(self, *, blocking=False):
            self.blocking = blocking
            self.closed = False

        def readline(self):
            if self.blocking:
                release.wait(timeout=2)
            return ""

        def write(self, _value):
            return None

        def flush(self):
            return None

        def close(self):
            self.closed = True
            release.set()

    class Proc:
        pid = 424242

        def __init__(self):
            self.stdin = Pipe()
            self.stdout = Pipe(blocking=True)
            self.stderr = Pipe()

        def poll(self):
            return None

        def wait(self, timeout):
            raise process.subprocess.TimeoutExpired("headless", timeout)

    child = Proc()
    terminated = []
    owner = process.Sts2Process(startup_timeout_sec=0.01)
    monkeypatch.setattr(owner, "_command", lambda: ["dotnet", "headless.dll"])
    monkeypatch.setattr(owner, "_env", lambda: {})
    monkeypatch.setattr(process.subprocess, "Popen", lambda *_args, **_kwargs: child)

    def terminate(proc):
        terminated.append(proc)
        release.set()

    monkeypatch.setattr(process, "_terminate_process_tree", terminate)

    with pytest.raises(process.Sts2ProtocolError, match="Timed out waiting for Sts2Headless ready"):
        owner.start()

    assert terminated == [child]
    assert owner.proc is None
    assert owner._stdout_reader is None
    assert child.stdin.closed
    assert child.stdout.closed
    assert child.stderr.closed


def test_response_timeout_reaps_process_and_joins_stdout_reader(monkeypatch):
    release = threading.Event()

    class Pipe:
        def __init__(self, *, lines=None, blocking=False):
            self.lines = list(lines or [])
            self.blocking = blocking
            self.closed = False

        def readline(self):
            if self.lines:
                return self.lines.pop(0)
            if self.blocking:
                release.wait(timeout=2)
            return ""

        def write(self, _value):
            return None

        def flush(self):
            return None

        def close(self):
            self.closed = True
            release.set()

    class Proc:
        pid = 424242

        def __init__(self):
            self.stdin = Pipe()
            self.stdout = Pipe(lines=['{"type":"ready"}\n'], blocking=True)
            self.stderr = Pipe()

        def poll(self):
            return None

        def wait(self, timeout):
            raise process.subprocess.TimeoutExpired("headless", timeout)

    child = Proc()
    terminated = []
    owner = process.Sts2Process(response_timeout_sec=0.01)
    monkeypatch.setattr(owner, "_command", lambda: ["dotnet", "headless.dll"])
    monkeypatch.setattr(owner, "_env", lambda: {})
    monkeypatch.setattr(process.subprocess, "Popen", lambda *_args, **_kwargs: child)

    def terminate(proc):
        terminated.append(proc)
        release.set()

    monkeypatch.setattr(process, "_terminate_process_tree", terminate)

    assert owner.start() == {"type": "ready"}
    reader = owner._stdout_reader
    with pytest.raises(process.Sts2ProtocolError, match="Timed out waiting for Sts2Headless response"):
        owner.send({"cmd": "reset"})

    assert terminated == [child]
    assert reader is not None and not reader.is_alive()
    assert owner.proc is None
    assert owner._stdout_reader is None


def test_start_closes_pipes_from_an_exited_process_before_replacement(monkeypatch):
    class Pipe:
        def __init__(self, line=""):
            self.line = line
            self.closed = False

        def readline(self):
            line, self.line = self.line, ""
            return line

        def write(self, _value):
            return None

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class Proc:
        def __init__(self, *, returncode, line=""):
            self.returncode = returncode
            self.stdin = Pipe()
            self.stdout = Pipe(line)
            self.stderr = Pipe()

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            return self.returncode or 0

    old = Proc(returncode=1)
    new = Proc(returncode=None, line='{"type":"ready"}\n')
    owner = process.Sts2Process()
    owner.proc = old
    monkeypatch.setattr(owner, "_command", lambda: ["dotnet", "headless.dll"])
    monkeypatch.setattr(owner, "_env", lambda: {})
    monkeypatch.setattr(process.subprocess, "Popen", lambda *_args, **_kwargs: new)

    assert owner.start() == {"type": "ready"}

    assert old.stdin.closed
    assert old.stdout.closed
    assert old.stderr.closed
    assert owner.proc is new
    owner.close()


@pytest.mark.skipif(process.os.name != "nt", reason="Windows taskkill contract")
def test_windows_tree_termination_targets_the_entire_process_tree(monkeypatch):
    class Proc:
        pid = 4321

        def __init__(self):
            self.polls = 0
            self.waited = False
            self.killed = False

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

        def wait(self, timeout):
            self.waited = True
            return 0

        def kill(self):
            self.killed = True

    calls = []
    child = Proc()
    monkeypatch.setattr(
        process.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    process._terminate_process_tree(child)

    assert calls[0][0] == ["taskkill", "/PID", "4321", "/T", "/F"]
    assert child.waited
    assert not child.killed


def test_close_uses_process_tree_termination_after_wait_timeout(monkeypatch):
    class Pipe:
        def __init__(self):
            self.closed = False

        def write(self, _value):
            return None

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class Proc:
        def __init__(self):
            self.stdin = Pipe()
            self.stdout = Pipe()
            self.stderr = Pipe()

        def poll(self):
            return None

        def wait(self, timeout):
            raise process.subprocess.TimeoutExpired("headless", timeout)

    terminated = []
    owner = process.Sts2Process()
    child = Proc()
    owner.proc = child
    monkeypatch.setattr(process, "_terminate_process_tree", lambda proc: terminated.append(proc))

    owner.close()

    assert terminated == [child]
    assert child.stdin.closed
    assert child.stdout.closed
    assert child.stderr.closed
