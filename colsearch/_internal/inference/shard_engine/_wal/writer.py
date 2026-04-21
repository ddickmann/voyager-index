"""Append-only writer for the shard-engine WAL."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .common import _BATCH_FLUSH_THRESHOLD
from .types import WalEntry, WalOp
from .reader import WalReader

class WalWriter:
    """Append-only WAL writer with per-entry CRC32.

    Supports an optional *batch mode* (``begin_batch`` / ``end_batch``)
    that buffers writes and defers ``flush`` until the batch ends or the
    buffer exceeds ``_BATCH_FLUSH_THRESHOLD`` entries.

    Can also be used as a context manager::

        with WalWriter(path).open() as w:
            w.log_insert(doc_id, vectors)
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._fd = None
        self._n_entries = 0
        self._batch_mode = False
        self._batch_count = 0

    def __enter__(self) -> "WalWriter":
        if self._fd is None:
            self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self) -> "WalWriter":
        is_new = not self._path.exists()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self._path, "ab" if not is_new else "wb")
        if not is_new:
            self._count_existing()
        return self

    def _count_existing(self):
        try:
            entries = WalReader(self._path).replay()
            self._n_entries = len(entries)
            if self._n_entries == 0:
                logger.debug("WAL exists but contains no valid entries: %s", self._path)
        except Exception as e:
            logger.warning("WAL corrupt or unreadable (%s); starting with n_entries=0: %s",
                           self._path, e)
            self._n_entries = 0

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------

    def begin_batch(self) -> None:
        """Enter batch mode: writes are buffered and flushed on ``end_batch``."""
        self._batch_mode = True
        self._batch_count = 0

    def end_batch(self) -> None:
        """Flush buffered writes and leave batch mode."""
        if self._fd is not None:
            self._fd.flush()
        self._batch_mode = False
        self._batch_count = 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _write_entry(self, op: WalOp, doc_id: int,
                     vectors: Optional[np.ndarray] = None,
                     payload: Optional[dict] = None):
        if self._fd is None:
            raise RuntimeError("WAL not open")

        buf = struct.pack("<Bq", op.value, doc_id)

        if op == WalOp.UPDATE_PAYLOAD:
            if payload is not None:
                pj = json.dumps(payload).encode("utf-8")
                buf += struct.pack("<I", len(pj))
                buf += pj
        else:
            if vectors is not None and op != WalOp.DELETE:
                arr = np.asarray(vectors, dtype=np.float32)
                n_vecs, dim = arr.shape
                buf += struct.pack("<II", n_vecs, dim)
                buf += arr.tobytes()

            if payload is not None and op != WalOp.DELETE:
                pj = json.dumps(payload).encode("utf-8")
                buf += struct.pack("<I", len(pj))
                buf += pj

        header = struct.pack(HEADER_FMT, WAL_MAGIC, WAL_VERSION, len(buf))
        crc = zlib.crc32(header + buf) & 0xFFFFFFFF
        self._fd.write(header)
        self._fd.write(buf)
        self._fd.write(struct.pack("<I", crc))

        self._n_entries += 1

        if self._batch_mode:
            self._batch_count += 1
            if self._batch_count >= _BATCH_FLUSH_THRESHOLD:
                self._fd.flush()
                self._batch_count = 0
        else:
            self._fd.flush()

    def log_insert(self, doc_id: int, vectors: np.ndarray,
                   payload: Optional[dict] = None):
        self._write_entry(WalOp.INSERT, doc_id, vectors, payload)

    def log_delete(self, doc_id: int):
        self._write_entry(WalOp.DELETE, doc_id)

    def log_upsert(self, doc_id: int, vectors: Optional[np.ndarray] = None,
                   payload: Optional[dict] = None):
        self._write_entry(WalOp.UPSERT, doc_id, vectors, payload)

    def log_update_payload(self, doc_id: int, payload: dict):
        self._write_entry(WalOp.UPDATE_PAYLOAD, doc_id, payload=payload)

    @property
    def n_entries(self) -> int:
        return self._n_entries

    def sync(self):
        """Flush and fsync the WAL to disk without closing."""
        if self._fd is not None:
            self._fd.flush()
            os.fsync(self._fd.fileno())

    def close(self):
        if self._fd is not None:
            self._fd.flush()
            os.fsync(self._fd.fileno())
            self._fd.close()
            self._fd = None

    def truncate(self):
        if self._fd is not None:
            self._fd.flush()
            self._fd.close()
            self._fd = None

        parent = self._path.parent
        fd_num, tmp_path = tempfile.mkstemp(dir=parent, prefix=".wal_trunc_")
        try:
            os.close(fd_num)
            os.rename(tmp_path, str(self._path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._n_entries = 0
        self._fd = open(self._path, "ab")

