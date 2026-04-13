"""WAL (Write-Ahead Log) for the shard engine.

Binary format per entry:
  [HEADER: magic(4) + version(1) + entry_len(4)] [PAYLOAD] [CRC32(4)]

The shard WAL uses its own CRC scope (header + payload) which differs from
gem_wal.py (payload-only CRC).  The two formats are not wire-compatible.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

WAL_MAGIC = b"SWAL"
WAL_VERSION = 1

HEADER_FMT = "<4sBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CRC_SIZE = 4

_BATCH_FLUSH_THRESHOLD = 1000
_REPLAY_CHUNK_SIZE = 1 << 20  # 1 MiB


class WalOp(IntEnum):
    INSERT = 0
    DELETE = 1
    UPSERT = 2
    UPDATE_PAYLOAD = 3


@dataclass
class WalEntry:
    op: WalOp
    doc_id: int
    vectors: Optional[np.ndarray] = None
    payload: Optional[dict] = None


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


class WalReader:
    """Reads and replays WAL entries, skipping corrupted ones."""

    def __init__(self, path: Path):
        self._path = Path(path)

    def replay(self) -> List[WalEntry]:
        if not self._path.exists():
            return []

        entries: List[WalEntry] = []
        corrupted = 0
        buf = b""

        with open(self._path, "rb") as fd:
            while True:
                chunk = fd.read(_REPLAY_CHUNK_SIZE)
                if not chunk and not buf:
                    break
                buf += chunk
                buf, new_entries, new_corrupt = self._process_buffer(buf, eof=not chunk)
                entries.extend(new_entries)
                corrupted += new_corrupt
                if not chunk:
                    break

        if corrupted > 0:
            logger.warning("WAL replay: %d corrupted entries skipped", corrupted)

        return entries

    @staticmethod
    def _process_buffer(data: bytes, eof: bool = False):
        """Parse as many complete entries as possible from *data*.

        Returns ``(remaining_bytes, entries, corrupt_count)``.
        """
        entries: List[WalEntry] = []
        corrupted = 0
        pos = 0

        while pos + HEADER_SIZE <= len(data):
            header = data[pos:pos + HEADER_SIZE]
            magic, version, entry_len = struct.unpack(HEADER_FMT, header)

            if magic != WAL_MAGIC or version != WAL_VERSION:
                corrupted += 1
                nxt = data.find(WAL_MAGIC, pos + 1)
                if nxt == -1:
                    if eof:
                        pos = len(data)
                    break
                pos = nxt
                continue

            payload_end = pos + HEADER_SIZE + entry_len
            crc_end = payload_end + CRC_SIZE
            if crc_end > len(data):
                if eof:
                    corrupted += 1
                    pos = len(data)
                break

            payload = data[pos + HEADER_SIZE:payload_end]
            expected_crc = struct.unpack("<I", data[payload_end:crc_end])[0]
            actual_crc = zlib.crc32(header + payload) & 0xFFFFFFFF

            if expected_crc != actual_crc:
                corrupted += 1
                logger.warning("WAL CRC mismatch at offset %d", pos)
                nxt = data.find(WAL_MAGIC, pos + 1)
                if nxt == -1:
                    if eof:
                        pos = len(data)
                    break
                pos = nxt
                continue

            entry = WalReader._parse_entry(payload)
            if entry is not None:
                entries.append(entry)
            pos = crc_end

        return data[pos:], entries, corrupted

    @staticmethod
    def _parse_entry(payload: bytes) -> Optional[WalEntry]:
        if len(payload) < 9:
            return None

        op_byte = payload[0]
        doc_id = struct.unpack_from("<q", payload, 1)[0]
        try:
            op = WalOp(op_byte)
        except ValueError:
            logger.warning("WAL: unknown op byte %d at doc_id %d", op_byte, doc_id)
            return None

        vectors = None
        entry_payload = None
        rest = payload[9:]

        if op == WalOp.UPDATE_PAYLOAD:
            if len(rest) >= 4:
                pld_len = struct.unpack_from("<I", rest, 0)[0]
                if len(rest) >= 4 + pld_len:
                    try:
                        entry_payload = json.loads(
                            rest[4:4 + pld_len].decode("utf-8"),
                        )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
        elif op != WalOp.DELETE and len(rest) >= 8:
            n_vecs, dim = struct.unpack_from("<II", rest, 0)
            expected = n_vecs * dim * 4
            vec_data = rest[8:]
            if len(vec_data) >= expected:
                vectors = np.frombuffer(
                    vec_data[:expected], dtype=np.float32,
                ).reshape(n_vecs, dim).copy()
                trailing = vec_data[expected:]
                if len(trailing) >= 4:
                    pld_len = struct.unpack_from("<I", trailing, 0)[0]
                    if len(trailing) >= 4 + pld_len:
                        try:
                            entry_payload = json.loads(
                                trailing[4:4 + pld_len].decode("utf-8"),
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

        return WalEntry(op=op, doc_id=doc_id, vectors=vectors, payload=entry_payload)
