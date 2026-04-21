"""
Crash-safe I/O utilities for GEM segment managers.

Provides:
- atomic_json_write: write JSON with fsync + rename (crash-safe)
- FileLock: exclusive file lock for multi-process safety
- RWLock: reader-writer lock for concurrent reads
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


def atomic_json_write(path: Path, data: Any) -> None:
    """Write JSON atomically: write to tmp file, fsync, rename over target."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".json_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class FileLock:
    """
    Exclusive file lock using fcntl.flock.
    Prevents multiple processes from opening the same shard.
    """

    def __init__(self, lock_path: Path):
        self._path = Path(lock_path)
        self._fd = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self._path, "w")
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as e:
            self._fd.close()
            self._fd = None
            import errno
            if getattr(e, 'errno', None) in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise RuntimeError(
                    f"Shard is locked by another process: {self._path}"
                ) from e
            raise RuntimeError(
                f"Cannot acquire lock on {self._path}: {e}"
            ) from e

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self._fd.close()
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class RWLock:
    """
    Reader-writer lock for concurrent reads with exclusive writes.

    NOT reentrant. Do NOT attempt to upgrade a read lock to a write lock
    from the same thread — this will deadlock. Each read_acquire must be
    paired with exactly one read_release.
    """

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._write_waiters = 0
        self._writer_id: int | None = None

    def read_acquire(self) -> None:
        with self._cond:
            current = threading.get_ident()
            if self._writer and self._writer_id == current:
                raise RuntimeError(
                    "RWLock: cannot acquire read lock while holding write lock "
                    "(would deadlock on release)"
                )
            while self._writer or self._write_waiters > 0:
                self._cond.wait()
            self._readers += 1

    def read_release(self) -> None:
        with self._cond:
            if self._readers <= 0:
                raise RuntimeError("RWLock: read_release without matching read_acquire")
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def write_acquire(self) -> None:
        with self._cond:
            current = threading.get_ident()
            if self._writer and self._writer_id == current:
                raise RuntimeError("RWLock: write lock is not reentrant")
            self._write_waiters += 1
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._write_waiters -= 1
            self._writer = True
            self._writer_id = current

    def write_release(self) -> None:
        with self._cond:
            if not self._writer:
                raise RuntimeError("RWLock: write_release without matching write_acquire")
            self._writer = False
            self._writer_id = None
            self._cond.notify_all()

    class _ReadContext:
        def __init__(self, lock: "RWLock"):
            self._lock = lock

        def __enter__(self):
            self._lock.read_acquire()
            return self

        def __exit__(self, *args):
            self._lock.read_release()

    class _WriteContext:
        def __init__(self, lock: "RWLock"):
            self._lock = lock

        def __enter__(self):
            self._lock.write_acquire()
            return self

        def __exit__(self, *args):
            self._lock.write_release()

    def read_lock(self) -> "_ReadContext":
        return self._ReadContext(self)

    def write_lock(self) -> "_WriteContext":
        return self._WriteContext(self)
