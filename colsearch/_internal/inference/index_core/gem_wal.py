"""
Write-Ahead Log (WAL) for GEM Native Segment Manager.

Binary append-only log with CRC32 integrity checks for crash recovery.

Format per entry:
    [4 bytes magic 'GWAL']
    [1 byte version]
    [4 bytes entry_length (payload only)]
    [payload bytes]
    [4 bytes CRC32 of payload]

Payload layout:
    [1 byte op: 0=INSERT, 1=DELETE, 2=UPSERT]
    [8 bytes external_id (u64 LE)]
    [optional vector data for INSERT/UPSERT:
        4 bytes n_vecs (u32 LE)
        4 bytes dim (u32 LE)
        n_vecs*dim*4 bytes float32 LE
    ]
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import tempfile
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

WAL_MAGIC = b"GWAL"
WAL_VERSION = 1

HEADER_FMT = "<4sBI"  # magic(4) + version(1) + entry_len(4)
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CRC_SIZE = 4


class WalOp(IntEnum):
    INSERT = 0
    DELETE = 1
    UPSERT = 2
    UPDATE_PAYLOAD = 3


@dataclass
class WalEntry:
    op: WalOp
    external_id: int
    vectors: Optional[np.ndarray] = None
    payload: Optional[dict] = None


class WalWriter:
    """Append-only WAL writer with per-entry CRC32."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._fd = None
        self._n_entries = 0

    def open(self):
        is_new = not self._path.exists()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self._path, "ab" if not is_new else "wb")
        if not is_new:
            self._count_existing_entries()
        return self

    def _count_existing_entries(self):
        try:
            reader = WalReader(self._path)
            entries = reader.replay()
            self._n_entries = len(entries)
        except Exception as e:
            logger.warning("WAL entry count failed (treating as 0): %s", e)
            self._n_entries = 0

    def _write_entry(
        self,
        op: WalOp,
        external_id: int,
        vectors: Optional[np.ndarray] = None,
        doc_payload: Optional[dict] = None,
    ):
        if self._fd is None:
            raise RuntimeError("WAL not open")

        buf = struct.pack("<Bq", op.value, external_id)

        if vectors is not None and op != WalOp.DELETE:
            n_vecs, dim = vectors.shape
            buf += struct.pack("<II", n_vecs, dim)
            buf += vectors.astype(np.float32).tobytes()

        if doc_payload is not None and op != WalOp.DELETE:
            payload_json = json.dumps(doc_payload).encode("utf-8")
            buf += struct.pack("<I", len(payload_json))
            buf += payload_json

        entry_bytes = buf
        crc = zlib.crc32(entry_bytes) & 0xFFFFFFFF

        header = struct.pack(HEADER_FMT, WAL_MAGIC, WAL_VERSION, len(entry_bytes))
        self._fd.write(header)
        self._fd.write(entry_bytes)
        self._fd.write(struct.pack("<I", crc))
        self._fd.flush()
        self._n_entries += 1

    def log_insert(self, external_id: int, vectors: np.ndarray, doc_payload: Optional[dict] = None):
        self._write_entry(WalOp.INSERT, external_id, vectors, doc_payload)

    def log_delete(self, external_id: int):
        self._write_entry(WalOp.DELETE, external_id)

    def log_upsert(self, external_id: int, vectors: np.ndarray, doc_payload: Optional[dict] = None):
        self._write_entry(WalOp.UPSERT, external_id, vectors, doc_payload)

    def log_update_payload(self, external_id: int, payload: dict):
        """WAL-log a payload update so it survives crashes."""
        if self._fd is None:
            raise RuntimeError("WAL not open")
        payload_json = json.dumps(payload).encode("utf-8")
        buf = struct.pack("<Bq", WalOp.UPDATE_PAYLOAD.value, external_id)
        buf += struct.pack("<I", len(payload_json))
        buf += payload_json
        crc = zlib.crc32(buf) & 0xFFFFFFFF
        header = struct.pack(HEADER_FMT, WAL_MAGIC, WAL_VERSION, len(buf))
        self._fd.write(header)
        self._fd.write(buf)
        self._fd.write(struct.pack("<I", crc))
        self._fd.flush()
        self._n_entries += 1

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
        """Atomic clear: close fd, write empty file via tmp + rename, reopen."""
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

        with open(self._path, "rb") as fd:
            file_data = fd.read()

        pos = 0
        while pos + HEADER_SIZE <= len(file_data):
            header = file_data[pos : pos + HEADER_SIZE]
            magic, version, entry_len = struct.unpack(HEADER_FMT, header)

            if magic != WAL_MAGIC or version != WAL_VERSION:
                corrupted += 1
                next_pos = file_data.find(WAL_MAGIC, pos + 1)
                if next_pos == -1:
                    break
                pos = next_pos
                continue

            payload_end = pos + HEADER_SIZE + entry_len
            crc_end = payload_end + CRC_SIZE
            if crc_end > len(file_data):
                corrupted += 1
                break

            payload = file_data[pos + HEADER_SIZE : payload_end]
            expected_crc = struct.unpack("<I", file_data[payload_end:crc_end])[0]
            actual_crc = zlib.crc32(payload) & 0xFFFFFFFF

            if expected_crc != actual_crc:
                corrupted += 1
                logger.warning("WAL CRC mismatch at offset %d, scanning for next entry", pos)
                next_pos = file_data.find(WAL_MAGIC, pos + 1)
                if next_pos == -1:
                    break
                pos = next_pos
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
        external_id = struct.unpack_from("<q", payload, 1)[0]
        try:
            op = WalOp(op_byte)
        except ValueError:
            return None

        vectors = None
        entry_payload = None
        rest = payload[9:]

        if op == WalOp.UPDATE_PAYLOAD and len(rest) >= 4:
            json_len = struct.unpack_from("<I", rest, 0)[0]
            json_data = rest[4 : 4 + json_len]
            try:
                entry_payload = json.loads(json_data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
        elif op != WalOp.DELETE and len(rest) >= 8:
            n_vecs, dim = struct.unpack_from("<II", rest, 0)
            expected = n_vecs * dim * 4
            vec_data = rest[8:]
            if len(vec_data) >= expected:
                vectors = np.frombuffer(vec_data[:expected], dtype=np.float32).reshape(
                    n_vecs, dim
                )
                trailing = vec_data[expected:]
                if len(trailing) >= 4:
                    pld_len = struct.unpack_from("<I", trailing, 0)[0]
                    if len(trailing) >= 4 + pld_len:
                        try:
                            entry_payload = json.loads(
                                trailing[4 : 4 + pld_len].decode("utf-8")
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

        return WalEntry(op=op, external_id=external_id, vectors=vectors, payload=entry_payload)


class CheckpointManager:
    """Atomic checkpoint for the active mutable segment state."""

    def __init__(self, checkpoint_dir: Path):
        self._dir = Path(checkpoint_dir)

    def save(
        self,
        doc_vectors: List[np.ndarray],
        doc_ids: List[int],
        next_doc_id: int,
        codebook_trained: bool,
        sealed_deleted_ids: Optional[set] = None,
    ):
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = self._dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "next_doc_id": next_doc_id,
            "codebook_trained": codebook_trained,
            "sealed_deleted_ids": sorted(sealed_deleted_ids) if sealed_deleted_ids else [],
            "n_docs": len(doc_ids),
        }
        meta_path = tmp_dir / "meta.json"
        meta_bytes = json.dumps(meta).encode("utf-8")
        with open(meta_path, "wb") as f:
            f.write(meta_bytes)

        ids_path = tmp_dir / "doc_ids.npy"
        np.save(ids_path, np.array(doc_ids, dtype=np.int64))

        if doc_vectors:
            shapes = [(v.shape[0], v.shape[1]) for v in doc_vectors]
            np.save(tmp_dir / "shapes.npy", np.array(shapes, dtype=np.int64))
            all_flat = np.vstack(doc_vectors) if doc_vectors else np.zeros((0, 1), dtype=np.float32)
            np.save(tmp_dir / "vectors.npy", all_flat)
        else:
            np.save(tmp_dir / "shapes.npy", np.zeros((0, 2), dtype=np.int64))
            np.save(tmp_dir / "vectors.npy", np.zeros((0, 1), dtype=np.float32))

        final_dir = self._dir / "current"
        old_dir = self._dir / ".old"
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if final_dir.exists():
            os.rename(str(final_dir), str(old_dir))
        os.rename(str(tmp_dir), str(final_dir))
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)

    def load(
        self,
    ) -> Optional[
        Tuple[List[np.ndarray], List[int], int, bool, set]
    ]:
        current = self._dir / "current"
        meta_path = current / "meta.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        next_doc_id = meta["next_doc_id"]
        codebook_trained = meta.get("codebook_trained", False)
        sealed_deleted_ids = set(meta.get("sealed_deleted_ids", []))

        ids_path = current / "doc_ids.npy"
        doc_ids = np.load(ids_path).tolist()

        shapes_path = current / "shapes.npy"
        vectors_path = current / "vectors.npy"

        shapes = np.load(shapes_path)
        flat = np.load(vectors_path)

        doc_vectors = []
        offset = 0
        for n_vecs, dim in shapes:
            n = int(n_vecs)
            dim = int(dim)
            mat = flat[offset : offset + n]
            doc_vectors.append(mat.astype(np.float32))
            offset += n

        return doc_vectors, doc_ids, next_doc_id, codebook_trained, sealed_deleted_ids

    def exists(self) -> bool:
        return (self._dir / "current" / "meta.json").exists()

    def clear(self):
        current = self._dir / "current"
        if current.exists():
            shutil.rmtree(current)
