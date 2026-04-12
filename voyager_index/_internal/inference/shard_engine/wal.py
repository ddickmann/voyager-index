"""WAL (Write-Ahead Log) for the shard engine.

Reuses the same binary format as gem_wal.py:
  [HEADER: magic(4) + version(1) + entry_len(4)] [PAYLOAD] [CRC32(4)]

This ensures format compatibility and crash recovery semantics.
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


class WalOp(IntEnum):
    INSERT = 0
    DELETE = 1
    UPSERT = 2


@dataclass
class WalEntry:
    op: WalOp
    doc_id: int
    vectors: Optional[np.ndarray] = None
    payload: Optional[dict] = None


class WalWriter:
    """Append-only WAL writer with per-entry CRC32."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._fd = None
        self._n_entries = 0

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
        except Exception as e:
            logger.warning("WAL entry count failed: %s", e)
            self._n_entries = 0

    def _write_entry(self, op: WalOp, doc_id: int,
                     vectors: Optional[np.ndarray] = None,
                     payload: Optional[dict] = None):
        if self._fd is None:
            raise RuntimeError("WAL not open")

        buf = struct.pack("<Bq", op.value, doc_id)

        if vectors is not None and op != WalOp.DELETE:
            n_vecs, dim = vectors.shape
            buf += struct.pack("<II", n_vecs, dim)
            buf += vectors.astype(np.float32).tobytes()

        if payload is not None and op != WalOp.DELETE:
            pj = json.dumps(payload).encode("utf-8")
            buf += struct.pack("<I", len(pj))
            buf += pj

        crc = zlib.crc32(buf) & 0xFFFFFFFF
        header = struct.pack(HEADER_FMT, WAL_MAGIC, WAL_VERSION, len(buf))
        self._fd.write(header)
        self._fd.write(buf)
        self._fd.write(struct.pack("<I", crc))
        self._fd.flush()
        self._n_entries += 1

    def log_insert(self, doc_id: int, vectors: np.ndarray,
                   payload: Optional[dict] = None):
        self._write_entry(WalOp.INSERT, doc_id, vectors, payload)

    def log_delete(self, doc_id: int):
        self._write_entry(WalOp.DELETE, doc_id)

    def log_upsert(self, doc_id: int, vectors: np.ndarray,
                   payload: Optional[dict] = None):
        self._write_entry(WalOp.UPSERT, doc_id, vectors, payload)

    @property
    def n_entries(self) -> int:
        return self._n_entries

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

        with open(self._path, "rb") as fd:
            data = fd.read()

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
                    break
                pos = nxt
                continue

            payload_end = pos + HEADER_SIZE + entry_len
            crc_end = payload_end + CRC_SIZE
            if crc_end > len(data):
                corrupted += 1
                break

            payload = data[pos + HEADER_SIZE:payload_end]
            expected_crc = struct.unpack("<I", data[payload_end:crc_end])[0]
            actual_crc = zlib.crc32(payload) & 0xFFFFFFFF

            if expected_crc != actual_crc:
                corrupted += 1
                logger.warning("WAL CRC mismatch at offset %d", pos)
                nxt = data.find(WAL_MAGIC, pos + 1)
                if nxt == -1:
                    break
                pos = nxt
                continue

            entry = self._parse_entry(payload)
            if entry is not None:
                entries.append(entry)
            pos = crc_end

        if corrupted > 0:
            logger.warning("WAL replay: %d corrupted entries skipped", corrupted)

        return entries

    @staticmethod
    def _parse_entry(payload: bytes) -> Optional[WalEntry]:
        if len(payload) < 9:
            return None

        op_byte = payload[0]
        doc_id = struct.unpack_from("<q", payload, 1)[0]
        try:
            op = WalOp(op_byte)
        except ValueError:
            return None

        vectors = None
        entry_payload = None
        rest = payload[9:]

        if op != WalOp.DELETE and len(rest) >= 8:
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
