"""Cross-process leases for benchmark artifact writers."""

from __future__ import annotations

import json
import os
import socket
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


class ArtifactLeaseError(RuntimeError):
    """Raised when another process already owns an artifact lease."""


_ACTIVE_PATHS: set[str] = set()
_ACTIVE_PATHS_GUARD = threading.Lock()


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _lock_file(handle: BinaryIO) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ArtifactLease:
    """Hold an exclusive OS-backed lease for one artifact namespace."""

    def __init__(self, path: str | Path, *, label: str) -> None:
        self.path = Path(path)
        self.label = label
        self.token = uuid.uuid4().hex
        self._handle: BinaryIO | None = None
        self._key: str | None = None

    def acquire(self) -> "ArtifactLease":
        if self._handle is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = _path_key(self.path)
        with _ACTIVE_PATHS_GUARD:
            if key in _ACTIVE_PATHS:
                raise ArtifactLeaseError(f"Another writer already owns {self.label}: {self.path}")
            _ACTIVE_PATHS.add(key)
        handle: BinaryIO | None = None
        try:
            handle = self.path.open("a+b")
            _lock_file(handle)
            self._handle = handle
            self._key = key
            self._write_metadata("running")
            return self
        except (OSError, BlockingIOError) as exc:
            if handle is not None:
                handle.close()
            with _ACTIVE_PATHS_GUARD:
                _ACTIVE_PATHS.discard(key)
            raise ArtifactLeaseError(f"Another writer already owns {self.label}: {self.path}") from exc

    def _write_metadata(self, status: str) -> None:
        if self._handle is None:
            return
        payload = {
            "schema_version": 1,
            "status": status,
            "label": self.label,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "token": self.token,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(encoded)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.seek(0)

    def release(self) -> None:
        handle = self._handle
        key = self._key
        if handle is None:
            return
        try:
            try:
                self._write_metadata("released")
            finally:
                _unlock_file(handle)
        finally:
            handle.close()
            self._handle = None
            self._key = None
            if key is not None:
                with _ACTIVE_PATHS_GUARD:
                    _ACTIVE_PATHS.discard(key)

    def __enter__(self) -> "ArtifactLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class ArtifactLeaseGroup:
    """Acquire several artifact namespaces in stable order as one lease."""

    def __init__(self, leases: list[ArtifactLease]) -> None:
        unique: dict[str, ArtifactLease] = {}
        for lease in leases:
            unique.setdefault(_path_key(lease.path), lease)
        self.leases = [unique[key] for key in sorted(unique)]
        self._acquired: list[ArtifactLease] = []

    def acquire(self) -> "ArtifactLeaseGroup":
        try:
            for lease in self.leases:
                lease.acquire()
                self._acquired.append(lease)
        except BaseException:
            try:
                self.release()
            except BaseException:
                pass
            raise
        return self

    def release(self) -> None:
        first_error: BaseException | None = None
        while self._acquired:
            try:
                self._acquired.pop().release()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "ArtifactLeaseGroup":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
