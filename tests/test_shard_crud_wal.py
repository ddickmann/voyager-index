"""Tests for shard engine CRUD, WAL, and crash recovery (Chunk 3)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from colsearch._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)
from colsearch._internal.inference.shard_engine.wal import (
    WalEntry,
    WalOp,
    WalReader,
    WalWriter,
)
from colsearch._internal.inference.shard_engine.memtable import MemTable


def _make_corpus(n_docs: int = 100, dim: int = 64, min_tok: int = 8, max_tok: int = 24):
    rng = np.random.RandomState(42)
    vectors = []
    for _ in range(n_docs):
        n_tok = rng.randint(min_tok, max_tok + 1)
        v = rng.randn(n_tok, dim).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vectors.append(v)
    return vectors


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="shard_crud_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------
# WAL unit tests
# ------------------------------------------------------------------

class TestWal:
    def test_write_and_replay(self, tmp_dir):
        wal_path = tmp_dir / "test.wal"
        writer = WalWriter(wal_path).open()

        v1 = np.random.randn(10, 64).astype(np.float32)
        writer.log_insert(1, v1, {"text": "hello"})
        writer.log_insert(2, np.random.randn(5, 64).astype(np.float32))
        writer.log_delete(1)
        writer.log_upsert(3, np.random.randn(8, 64).astype(np.float32), {"text": "world"})
        assert writer.n_entries == 4
        writer.close()

        entries = WalReader(wal_path).replay()
        assert len(entries) == 4
        assert entries[0].op == WalOp.INSERT
        assert entries[0].doc_id == 1
        assert entries[0].payload == {"text": "hello"}
        np.testing.assert_allclose(entries[0].vectors, v1, atol=1e-6)
        assert entries[2].op == WalOp.DELETE
        assert entries[2].doc_id == 1
        assert entries[3].op == WalOp.UPSERT

    def test_truncate(self, tmp_dir):
        wal_path = tmp_dir / "test.wal"
        writer = WalWriter(wal_path).open()
        writer.log_insert(1, np.random.randn(5, 64).astype(np.float32))
        writer.truncate()
        assert writer.n_entries == 0
        writer.log_insert(2, np.random.randn(5, 64).astype(np.float32))
        writer.close()

        entries = WalReader(wal_path).replay()
        assert len(entries) == 1
        assert entries[0].doc_id == 2

    def test_corrupt_entry_skipped(self, tmp_dir):
        wal_path = tmp_dir / "test.wal"
        writer = WalWriter(wal_path).open()
        writer.log_insert(1, np.random.randn(5, 64).astype(np.float32))
        writer.log_insert(2, np.random.randn(5, 64).astype(np.float32))
        writer.close()

        data = wal_path.read_bytes()
        corrupted = data[:20] + b"\xff" * 4 + data[24:]
        wal_path.write_bytes(corrupted)

        entries = WalReader(wal_path).replay()
        assert len(entries) <= 2

    def test_empty_wal_replay(self, tmp_dir):
        wal_path = tmp_dir / "nonexistent.wal"
        entries = WalReader(wal_path).replay()
        assert entries == []


# ------------------------------------------------------------------
# MemTable unit tests
# ------------------------------------------------------------------

class TestMemTable:
    def test_insert_and_search(self):
        mt = MemTable(dim=64, device="cpu")
        v = np.random.randn(10, 64).astype(np.float32)
        mt.insert(1, v)
        mt.insert(2, np.random.randn(8, 64).astype(np.float32))
        assert mt.size == 2

        q = np.random.randn(5, 64).astype(np.float32)
        results = mt.search(q, k=2)
        assert len(results) == 2

    def test_delete_creates_tombstone(self):
        mt = MemTable(dim=64, device="cpu")
        mt.insert(1, np.random.randn(10, 64).astype(np.float32))
        mt.delete(1)
        assert mt.size == 0
        assert mt.is_tombstoned(1)

    def test_upsert_replaces(self):
        mt = MemTable(dim=64, device="cpu")
        v1 = np.random.randn(10, 64).astype(np.float32)
        v2 = np.random.randn(5, 64).astype(np.float32)
        mt.insert(1, v1)
        mt.upsert(1, v2)
        assert mt.size == 1
        docs, _, _ = mt.snapshot()
        np.testing.assert_array_equal(docs[1], v2)

    def test_drain(self):
        mt = MemTable(dim=64, device="cpu")
        mt.insert(1, np.random.randn(10, 64).astype(np.float32))
        mt.delete(2)
        docs, payloads, tombstones = mt.drain()
        assert 1 in docs
        assert 2 in tombstones
        assert mt.size == 0


# ------------------------------------------------------------------
# Manager CRUD integration
# ------------------------------------------------------------------

class TestManagerCRUD:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_add_after_build(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=100)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=50, dim=dim)
        mgr.build(vectors, list(range(50)))

        new_vecs = _make_corpus(n_docs=10, dim=dim)
        mgr.add_multidense(new_vecs, list(range(50, 60)))

        q = np.random.RandomState(7).randn(8, dim).astype(np.float32)
        results = mgr.search_multivector(q, k=10)
        assert len(results) == 10
        mgr.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_delete_filters_results(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=30, dim=dim)
        mgr.build(vectors, list(range(30)))

        q = np.random.RandomState(7).randn(8, dim).astype(np.float32)
        results_before = mgr.search_multivector(q, k=5)
        top_id = results_before[0][0]

        mgr.delete([top_id])
        results_after = mgr.search_multivector(q, k=5)
        result_ids_after = [r[0] for r in results_after]
        assert top_id not in result_ids_after
        mgr.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_upsert(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=20, dim=dim)
        mgr.build(vectors, list(range(20)))

        new_vec = [np.random.randn(10, dim).astype(np.float32)]
        mgr.upsert_multidense(new_vec, [0])

        assert mgr._memtable.size == 1
        mgr.close()


# ------------------------------------------------------------------
# Crash recovery
# ------------------------------------------------------------------

class TestCrashRecovery:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_wal_replay_on_reload(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=100)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=40, dim=dim)
        mgr.build(vectors, list(range(40)))

        new_vecs = _make_corpus(n_docs=5, dim=dim)
        mgr.add_multidense(new_vecs, list(range(40, 45)))
        mgr.delete([0])
        assert mgr._wal_writer.n_entries == 6  # 5 inserts + 1 delete

        mgr._wal_writer.close()
        mgr._wal_writer = None
        mgr._memtable = None
        mgr._is_built = False

        mgr2 = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        assert mgr2._memtable.size == 5
        assert mgr2._memtable.is_tombstoned(0)

        q = np.random.RandomState(99).randn(8, dim).astype(np.float32)
        results = mgr2.search_multivector(q, k=5)
        result_ids = [r[0] for r in results]
        assert 0 not in result_ids
        mgr2.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flush_preserves_memtable(self, tmp_dir):
        """flush() syncs WAL but does NOT drain memtable (safe against data loss)."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=20, dim=dim)
        mgr.build(vectors, list(range(20)))

        new_vecs = _make_corpus(n_docs=3, dim=dim)
        mgr.add_multidense(new_vecs, list(range(20, 23)))
        assert mgr._wal_writer.n_entries == 3

        mgr.flush()
        assert mgr._wal_writer.n_entries == 3
        assert mgr._memtable.size == 3
        mgr.close()
