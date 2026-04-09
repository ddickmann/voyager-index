"""Crash-fuzz and WAL correctness tests for GEM WAL."""
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from voyager_index._internal.inference.index_core.gem_wal import (
    WalWriter, WalReader, WalEntry, WalOp,
    WAL_MAGIC, WAL_VERSION, HEADER_FMT, HEADER_SIZE, CRC_SIZE,
)


def _make_vectors(n_vecs=4, dim=8, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState(42)
    return rng.randn(n_vecs, dim).astype(np.float32)


def _entry_boundaries(raw: bytes):
    """Return list of (start, payload_end, crc_end) for each valid entry."""
    boundaries = []
    pos = 0
    while pos + HEADER_SIZE <= len(raw):
        magic, version, entry_len = struct.unpack(HEADER_FMT, raw[pos:pos + HEADER_SIZE])
        if magic != WAL_MAGIC or version != WAL_VERSION:
            break
        payload_end = pos + HEADER_SIZE + entry_len
        crc_end = payload_end + CRC_SIZE
        if crc_end > len(raw):
            break
        boundaries.append((pos, payload_end, crc_end))
        pos = crc_end
    return boundaries


class TestWalRoundtrip:
    """Basic write-read roundtrip tests."""

    def test_write_read_roundtrip(self):
        """Write 10 INSERT entries, replay, verify all 10 recovered."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            writer = WalWriter(wal_path)
            writer.open()
            vectors = [_make_vectors(seed=i) for i in range(10)]
            for i in range(10):
                writer.log_insert(i, vectors[i])
            writer.close()

            entries = WalReader(wal_path).replay()
            assert len(entries) == 10
            for i, entry in enumerate(entries):
                assert entry.op == WalOp.INSERT
                assert entry.external_id == i
                assert entry.vectors is not None
                assert entry.vectors.shape == (4, 8)
                np.testing.assert_array_almost_equal(entry.vectors, vectors[i])

    def test_delete_roundtrip(self):
        """Write INSERT then DELETE for same id. Replay returns both entries."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            writer = WalWriter(wal_path)
            writer.open()
            vecs = _make_vectors(seed=99)
            writer.log_insert(42, vecs)
            writer.log_delete(42)
            writer.close()

            entries = WalReader(wal_path).replay()
            assert len(entries) == 2
            assert entries[0].op == WalOp.INSERT
            assert entries[0].external_id == 42
            np.testing.assert_array_almost_equal(entries[0].vectors, vecs)
            assert entries[1].op == WalOp.DELETE
            assert entries[1].external_id == 42
            assert entries[1].vectors is None

    def test_upsert_roundtrip(self):
        """Write INSERT then UPSERT for same id. Replay returns both with correct vectors."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            writer = WalWriter(wal_path)
            writer.open()
            vecs_old = _make_vectors(seed=10)
            vecs_new = _make_vectors(seed=20)
            writer.log_insert(7, vecs_old)
            writer.log_upsert(7, vecs_new)
            writer.close()

            entries = WalReader(wal_path).replay()
            assert len(entries) == 2
            assert entries[0].op == WalOp.INSERT
            assert entries[0].external_id == 7
            np.testing.assert_array_almost_equal(entries[0].vectors, vecs_old)
            assert entries[1].op == WalOp.UPSERT
            assert entries[1].external_id == 7
            np.testing.assert_array_almost_equal(entries[1].vectors, vecs_new)


class TestWalTruncation:
    """Simulate crash by truncating the WAL file at various points."""

    def _write_five_entries(self, wal_path: Path):
        writer = WalWriter(wal_path)
        writer.open()
        for i in range(5):
            writer.log_insert(i, _make_vectors(seed=i))
        writer.close()

    def test_truncation_mid_header(self):
        """Truncate within the last entry's CRC region. Should recover first 4."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            self._write_five_entries(wal_path)

            raw = wal_path.read_bytes()
            truncated = raw[: len(raw) - 3]
            wal_path.write_bytes(truncated)

            entries = WalReader(wal_path).replay()
            assert len(entries) == 4
            for i, entry in enumerate(entries):
                assert entry.external_id == i

    def test_truncation_mid_payload(self):
        """Truncate in the middle of the last entry's payload. Should recover first 4."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            self._write_five_entries(wal_path)

            raw = wal_path.read_bytes()
            bounds = _entry_boundaries(raw)
            assert len(bounds) == 5
            last_start = bounds[4][0]
            cut_pos = last_start + HEADER_SIZE + 2
            wal_path.write_bytes(raw[:cut_pos])

            entries = WalReader(wal_path).replay()
            assert len(entries) == 4
            for i, entry in enumerate(entries):
                assert entry.external_id == i

    def test_truncation_mid_crc(self):
        """Truncate midway through the last entry's CRC. Should recover first 4."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            self._write_five_entries(wal_path)

            raw = wal_path.read_bytes()
            bounds = _entry_boundaries(raw)
            assert len(bounds) == 5
            last_payload_end = bounds[4][1]
            cut_pos = last_payload_end + 2  # 2 bytes into the 4-byte CRC
            wal_path.write_bytes(raw[:cut_pos])

            entries = WalReader(wal_path).replay()
            assert len(entries) == 4
            for i, entry in enumerate(entries):
                assert entry.external_id == i


class TestWalCorruption:
    """Simulate on-disk corruption by flipping bytes."""

    def test_corrupted_crc(self):
        """Flip a byte in the 3rd entry's CRC. Should recover entries 1,2 then 4,5."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            writer = WalWriter(wal_path)
            writer.open()
            for i in range(5):
                writer.log_insert(i, _make_vectors(seed=i))
            writer.close()

            raw = bytearray(wal_path.read_bytes())
            bounds = _entry_boundaries(bytes(raw))
            assert len(bounds) == 5

            crc_offset = bounds[2][1]  # start of 3rd entry's CRC
            raw[crc_offset] ^= 0xFF
            wal_path.write_bytes(bytes(raw))

            entries = WalReader(wal_path).replay()
            assert len(entries) == 4
            recovered_ids = [e.external_id for e in entries]
            assert recovered_ids == [0, 1, 3, 4]


class TestWalEdgeCases:
    """Edge cases: empty WAL, payload roundtrip, truncate method."""

    def test_empty_wal(self):
        """Replay of a zero-byte WAL returns an empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            wal_path.write_bytes(b"")

            entries = WalReader(wal_path).replay()
            assert entries == []

    def test_payload_roundtrip(self):
        """Write INSERT with a payload dict, replay, verify payload recovered."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            writer = WalWriter(wal_path)
            writer.open()
            vecs = _make_vectors(seed=0)
            payload = {"color": "blue", "score": 3.14, "tags": ["a", "b"]}
            writer.log_insert(100, vecs, doc_payload=payload)
            writer.close()

            entries = WalReader(wal_path).replay()
            assert len(entries) == 1
            assert entries[0].external_id == 100
            assert entries[0].payload == payload
            np.testing.assert_array_almost_equal(entries[0].vectors, vecs)

    def test_truncate_method(self):
        """writer.truncate() clears WAL; new writes are visible after."""
        with tempfile.TemporaryDirectory() as tmp:
            wal_path = Path(tmp) / "test.wal"
            writer = WalWriter(wal_path)
            writer.open()
            for i in range(5):
                writer.log_insert(i, _make_vectors(seed=i))

            assert writer.n_entries == 5
            writer.truncate()
            assert writer.n_entries == 0

            entries = WalReader(wal_path).replay()
            assert entries == []

            for i in range(100, 103):
                writer.log_insert(i, _make_vectors(seed=i))
            writer.close()

            entries = WalReader(wal_path).replay()
            assert len(entries) == 3
            assert [e.external_id for e in entries] == [100, 101, 102]
