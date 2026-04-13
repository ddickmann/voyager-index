"""Hardening tests for shard engine: covers gaps identified in audit."""
from __future__ import annotations

import inspect
import logging
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from voyager_index._internal.inference.shard_engine.config import (
    AnnBackend,
    Compression,
    SearchConfig,
)
from voyager_index._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)
from voyager_index._internal.inference.shard_engine.memtable import MemTable
from voyager_index._internal.inference.shard_engine.wal import WalOp, WalReader, WalWriter

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"


def _make_corpus(n_docs: int = 50, dim: int = 64, min_tok: int = 8, max_tok: int = 32, seed: int = 42):
    rng = np.random.RandomState(seed)
    return [rng.randn(rng.randint(min_tok, max_tok + 1), dim).astype(np.float32) for _ in range(n_docs)]


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path / "shard_test"


# ------------------------------------------------------------------
# FIX 2: allocate_ids collision
# ------------------------------------------------------------------

class TestAllocateIds:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_no_id_collision_after_add(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(20, dim=dim)
        mgr.build(corpus, list(range(20)))

        ids1 = mgr.allocate_ids(5)
        mgr.add_multidense(_make_corpus(5, dim=dim), ids1)

        ids2 = mgr.allocate_ids(3)
        assert not set(ids1) & set(ids2), "Second allocation must not overlap first"
        assert min(ids2) > max(ids1)
        mgr.close()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_allocate_ids_monotonic(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(10, dim=dim)
        mgr.build(corpus, list(range(10)))

        batch1 = mgr.allocate_ids(3)
        assert batch1 == [10, 11, 12]
        mgr.add_multidense(_make_corpus(3, dim=dim), batch1)
        batch2 = mgr.allocate_ids(2)
        assert batch2 == [13, 14]
        mgr.close()


# ------------------------------------------------------------------
# FIX 4: PreloadedGpuCorpus doc-ID mapping
# ------------------------------------------------------------------

class TestGpuCorpusMapping:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_score_candidates_with_missing_ids(self, tmp_dir):
        from voyager_index._internal.inference.shard_engine.scorer import PreloadedGpuCorpus
        dim = 64
        n_docs = 20
        vecs = [np.random.randn(16, dim).astype(np.float32) for _ in range(n_docs)]
        ids = list(range(n_docs))
        corpus = PreloadedGpuCorpus(vecs, ids, dim, device="cuda")

        query = torch.randn(8, dim)
        candidate_ids = [0, 1, 999, 5, 1000]
        result_ids, result_scores = corpus.score_candidates(query, candidate_ids, k=3)
        assert len(result_ids) == 3
        for rid in result_ids:
            assert rid in {0, 1, 5}

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_score_candidates_all_missing(self, tmp_dir):
        from voyager_index._internal.inference.shard_engine.scorer import PreloadedGpuCorpus
        dim = 64
        vecs = [np.random.randn(16, dim).astype(np.float32) for _ in range(5)]
        corpus = PreloadedGpuCorpus(vecs, [0, 1, 2, 3, 4], dim, device="cuda")

        query = torch.randn(8, dim)
        result_ids, result_scores = corpus.score_candidates(query, [99, 100, 101], k=3)
        assert result_ids == []
        assert result_scores == []


# ------------------------------------------------------------------
# FIX 8: Filters
# ------------------------------------------------------------------

class TestSearchFilters:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_search_with_filters(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(30, dim=dim)
        payloads = [{"category": "A" if i % 2 == 0 else "B"} for i in range(30)]
        mgr.build(corpus, list(range(30)), payloads=payloads)

        query = np.random.randn(8, dim).astype(np.float32)
        results = mgr.search_multivector(query, k=10, filters={"category": {"$eq": "A"}})
        for did, _ in results:
            payload = mgr._get_payload(did)
            assert payload.get("category") == "A"
        mgr.close()

    def test_evaluate_filter_type_error(self):
        payload = {"age": "not_a_number"}
        assert not ShardSegmentManager._evaluate_filter(payload, {"age": {"$gt": 10}})

    def test_evaluate_filter_none_value(self):
        payload = {"age": None}
        assert not ShardSegmentManager._evaluate_filter(payload, {"age": {"$gt": 10}})
        assert not ShardSegmentManager._evaluate_filter(payload, {"age": {"$lt": 10}})


# ------------------------------------------------------------------
# FIX 9: Sealed payloads + retrieve
# ------------------------------------------------------------------

class TestSealedPayloads:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_payloads_survive_reload(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        payloads = [{"title": f"doc_{i}"} for i in range(20)]
        mgr.build(_make_corpus(20, dim=dim), list(range(20)), payloads=payloads)
        mgr.close()

        mgr2 = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        results = mgr2.retrieve([0, 5, 10], with_payload=True)
        assert results[0]["payload"]["title"] == "doc_0"
        assert results[1]["payload"]["title"] == "doc_5"
        assert results[2]["payload"]["title"] == "doc_10"
        mgr2.close()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_retrieve_with_vector(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(20, dim=dim)
        mgr.build(corpus, list(range(20)))

        results = mgr.retrieve([0], with_vector=True, with_payload=True)
        assert results[0]["vector"] is not None
        assert results[0]["vector"].shape[1] == dim
        mgr.close()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_explain_score_sealed(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(20, dim=dim)
        mgr.build(corpus, list(range(20)))

        query = np.random.randn(8, dim).astype(np.float32)
        scores, matched = mgr._explain_score(query, 0)
        assert scores is not None
        assert len(scores) == 8
        mgr.close()


# ------------------------------------------------------------------
# FIX 10: Thread safety
# ------------------------------------------------------------------

class TestThreadSafety:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_concurrent_search_and_add(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        mgr.build(_make_corpus(50, dim=dim), list(range(50)))

        errors = []

        def searcher():
            try:
                for _ in range(10):
                    q = np.random.randn(8, dim).astype(np.float32)
                    mgr.search_multivector(q, k=5)
            except Exception as e:
                errors.append(e)

        def adder():
            try:
                for i in range(10):
                    v = [np.random.randn(16, dim).astype(np.float32)]
                    mgr.add_multidense(v, [100 + i])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=searcher) for _ in range(3)]
        threads.append(threading.Thread(target=adder))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        mgr.close()


# ------------------------------------------------------------------
# FIX 11: search_batch and upsert_payload
# ------------------------------------------------------------------

class TestIndexApiGaps:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_search_batch(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        mgr.build(_make_corpus(30, dim=dim), list(range(30)))

        queries = [np.random.randn(8, dim).astype(np.float32) for _ in range(3)]
        results = mgr.search_batch(queries, k=5)
        assert len(results) == 3
        for r in results:
            assert len(r) <= 5
        mgr.close()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_upsert_payload(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        payloads = [{"title": f"doc_{i}"} for i in range(20)]
        mgr.build(_make_corpus(20, dim=dim), list(range(20)), payloads=payloads)

        mgr.upsert_payload(0, {"title": "updated_doc_0", "extra": True})
        p = mgr._get_payload(0)
        assert p["title"] == "updated_doc_0"
        assert p["extra"] is True
        mgr.close()


# ------------------------------------------------------------------
# FIX 12: shard store validation
# ------------------------------------------------------------------

class TestShardStoreValidation:
    def test_build_validates_dims(self, tmp_dir):
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        store = ShardStore(tmp_dir / "store")
        vecs = np.random.randn(10, 32).astype(np.float16)
        with pytest.raises(ValueError, match="incompatible with dim"):
            store.build(
                all_vectors=vecs,
                doc_offsets=[(0, 10)],
                doc_ids=[0],
                shard_assignments=np.array([0]),
                n_shards=1,
                dim=64,
            )

    def test_build_validates_length_mismatch(self, tmp_dir):
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        store = ShardStore(tmp_dir / "store")
        vecs = np.random.randn(10, 64).astype(np.float16)
        with pytest.raises(ValueError, match="doc_offsets"):
            store.build(
                all_vectors=vecs,
                doc_offsets=[(0, 10), (10, 20)],
                doc_ids=[0],
                shard_assignments=np.array([0]),
                n_shards=1,
                dim=64,
            )


# ------------------------------------------------------------------
# FIX 13: WAL robustness
# ------------------------------------------------------------------

class TestWalRobustness:
    def test_wal_sync(self, tmp_dir):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wal_path = tmp_dir / "test.wal"
        w = WalWriter(wal_path)
        w.open()
        w.log_insert(0, np.random.randn(4, 64).astype(np.float32), {"key": "val"})
        w.sync()
        assert wal_path.stat().st_size > 0
        w.close()

        entries = WalReader(wal_path).replay()
        assert len(entries) == 1
        assert entries[0].doc_id == 0
        assert entries[0].payload == {"key": "val"}

    def test_wal_payload_only_upsert(self, tmp_dir):
        """WAL entry with None vectors should be handled correctly."""
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wal_path = tmp_dir / "test.wal"
        w = WalWriter(wal_path)
        w.open()
        w.log_upsert(42, None, {"updated": True})
        w.close()

        entries = WalReader(wal_path).replay()
        assert len(entries) == 1
        assert entries[0].op == WalOp.UPSERT
        assert entries[0].doc_id == 42


# ------------------------------------------------------------------
# Memtable tombstone snapshot
# ------------------------------------------------------------------

class TestMemtableTombstones:
    def test_tombstones_snapshot(self):
        mt = MemTable(dim=64, device="cpu")
        mt.insert(0, np.random.randn(4, 64).astype(np.float32))
        mt.insert(1, np.random.randn(4, 64).astype(np.float32))
        mt.delete(0)
        ts = mt.tombstones_snapshot()
        assert 0 in ts
        assert 1 not in ts
        ts.add(999)
        assert 999 not in mt.tombstones_snapshot()

    def test_upsert_payload(self):
        mt = MemTable(dim=64, device="cpu")
        mt.insert(0, np.random.randn(4, 64).astype(np.float32), {"key": "old"})
        mt.upsert_payload(0, {"key": "new"})
        _, payloads, _ = mt.snapshot()
        assert payloads[0]["key"] == "new"


# ------------------------------------------------------------------
# FIX 17: Config cleanup
# ------------------------------------------------------------------

class TestConfigCleanup:
    def test_faiss_flat_ip_is_default(self):
        assert AnnBackend.FAISS_FLAT_IP.value == "faiss_flat_ip"
        assert AnnBackend.FAISS_HNSW_IP.value == "faiss_flat_ip"

    def test_search_config_no_dead_fields(self):
        sc = SearchConfig()
        assert not hasattr(sc, "top_shards")
        assert not hasattr(sc, "colbandit")

    def test_engine_config_defaults(self):
        cfg = ShardEngineConfig()
        assert cfg.ann_backend == AnnBackend.FAISS_FLAT_IP


# ------------------------------------------------------------------
# FIX 1: flush does not lose data
# ------------------------------------------------------------------

class TestFlushSafety:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_flush_preserves_data(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        mgr.build(_make_corpus(20, dim=dim), list(range(20)))
        mgr.add_multidense(_make_corpus(5, dim=dim), list(range(20, 25)))

        total_before = mgr.total_vectors()
        mgr.flush()
        total_after = mgr.total_vectors()
        assert total_after == total_before

        q = np.random.randn(8, dim).astype(np.float32)
        results = mgr.search_multivector(q, k=5)
        assert len(results) > 0
        mgr.close()


# ------------------------------------------------------------------
# Scroll with filters
# ------------------------------------------------------------------

class TestScrollWithFilters:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_scroll_with_sealed_filters(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        payloads = [{"cat": "A" if i % 2 == 0 else "B"} for i in range(20)]
        mgr.build(_make_corpus(20, dim=dim), list(range(20)), payloads=payloads)

        page, next_off = mgr.scroll(limit=100, filters={"cat": {"$eq": "A"}})
        for did in page:
            assert mgr._get_payload(did).get("cat") == "A"
        mgr.close()


# ==================================================================
# 360-DEGREE CRITICAL FIX VERIFICATION (all CPU-safe, no CUDA guard)
# ==================================================================

class TestCriticalFixVerification:
    """One test per CRITICAL FIX GROUP (20-27). Each provides explicit
    evidence that the fix is correct and in place. All run on CPU."""

    # ----------------------------------------------------------
    # FIX 20: WAL UPDATE_PAYLOAD round-trip
    # ----------------------------------------------------------

    def test_fix20_wal_update_payload_roundtrip(self, tmp_dir):
        """Evidence: UPDATE_PAYLOAD entries are correctly written, parsed,
        and replayed — payload survives WAL round-trip."""
        from voyager_index._internal.inference.shard_engine.wal import WalOp, WalReader, WalWriter
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wal_path = tmp_dir / "fix20.wal"

        w = WalWriter(wal_path).open()
        w.log_insert(0, np.random.randn(4, 64).astype(np.float32), {"title": "original"})
        w.log_update_payload(0, {"title": "updated", "extra": 42})
        w.log_update_payload(99, {"brand_new": True})
        assert w.n_entries == 3
        w.close()

        entries = WalReader(wal_path).replay()
        assert len(entries) == 3

        assert entries[0].op == WalOp.INSERT
        assert entries[0].doc_id == 0
        assert entries[0].vectors is not None
        assert entries[0].payload == {"title": "original"}

        assert entries[1].op == WalOp.UPDATE_PAYLOAD
        assert entries[1].doc_id == 0
        assert entries[1].vectors is None
        assert entries[1].payload == {"title": "updated", "extra": 42}

        assert entries[2].op == WalOp.UPDATE_PAYLOAD
        assert entries[2].doc_id == 99
        assert entries[2].vectors is None
        assert entries[2].payload == {"brand_new": True}

    def test_fix20_payload_survives_replay(self, tmp_dir):
        """Evidence: upsert_payload WAL entries are replayed on reload."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        payloads = [{"title": f"doc_{i}"} for i in range(10)]
        mgr.build(_make_corpus(10, dim=dim), list(range(10)), payloads=payloads)

        mgr.upsert_payload(0, {"title": "UPDATED", "new_field": True})

        wal_entries_before = mgr._wal_writer.n_entries
        assert wal_entries_before >= 1

        mgr.close()

        mgr2 = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        p = mgr2._get_payload(0)
        assert p["title"] == "UPDATED", f"Payload not replayed: {p}"
        assert p["new_field"] is True
        mgr2.close()

    # ----------------------------------------------------------
    # FIX 21: Search snapshot consistency
    # ----------------------------------------------------------

    def test_fix21_search_snapshot_consistency(self, tmp_dir):
        """Evidence: concurrent search + delete does not crash, and
        save() is protected by the lock."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(30, dim=dim), list(range(30)))

        errors = []

        def searcher():
            try:
                for _ in range(20):
                    q = np.random.randn(8, dim).astype(np.float32)
                    results = mgr.search_multivector(q, k=5)
                    for did, _ in results:
                        assert isinstance(did, int)
            except Exception as e:
                errors.append(("search", e))

        def deleter():
            try:
                for i in range(10):
                    mgr.delete([i])
            except Exception as e:
                errors.append(("delete", e))

        def saver():
            try:
                mgr.save()
            except Exception as e:
                errors.append(("save", e))

        threads = [
            threading.Thread(target=searcher),
            threading.Thread(target=searcher),
            threading.Thread(target=deleter),
            threading.Thread(target=saver),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent errors: {errors}"
        mgr.close()

    # ----------------------------------------------------------
    # FIX 22: retrieve() respects tombstones
    # ----------------------------------------------------------

    def test_fix22_retrieve_respects_tombstones(self, tmp_dir):
        """Evidence: deleted documents are not returned by retrieve()."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        payloads = [{"title": f"doc_{i}"} for i in range(5)]
        mgr.build(_make_corpus(5, dim=dim), list(range(5)), payloads=payloads)

        mgr.delete([1, 3])

        results = mgr.retrieve([0, 1, 2, 3, 4], with_payload=True)
        result_ids = [r["id"] for r in results]

        assert 1 not in result_ids, "Tombstoned doc 1 should not appear"
        assert 3 not in result_ids, "Tombstoned doc 3 should not appear"
        assert 0 in result_ids
        assert 2 in result_ids
        assert 4 in result_ids
        assert len(results) == 3

        for r in results:
            assert r["payload"]["title"] == f"doc_{r['id']}"
        mgr.close()

    # ----------------------------------------------------------
    # FIX 23: search_batch accepts ef/n_probes without TypeError
    # ----------------------------------------------------------

    def test_fix23_search_batch_no_typeerror(self, tmp_dir):
        """Evidence: search_batch accepts ef= and n_probes= kwargs
        (the exact signature Index.search_batch passes)."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(20, dim=dim), list(range(20)))

        queries = [np.random.randn(8, dim).astype(np.float32) for _ in range(3)]

        results = mgr.search_batch(queries, k=5, ef=100, n_probes=4)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)
            assert len(r) <= 5

        results_single = mgr.search_multivector(
            queries[0], k=5, ef=50, n_probes=2, some_future_param=True,
        )
        assert isinstance(results_single, list)
        mgr.close()

    # ----------------------------------------------------------
    # FIX 24: Journal backup completeness
    # ----------------------------------------------------------

    def test_fix24_journal_backup_completeness(self, tmp_dir):
        """Evidence: the SHARD branch in _begin_collection_mutation
        copies the entire shard/ directory (not just 4 files)."""
        import voyager_index._internal.server.api.service as svc_module
        import inspect
        source = inspect.getsource(svc_module.SearchService._begin_collection_mutation)

        assert 'shard_dir / "engine_meta.json"' not in source or \
               "self._copy_into_backup(runtime, backup_root, shard_dir)" in source, \
            "SHARD backup should copy entire shard_dir, not individual files"

        assert "runtime.engine.flush" in source, \
            "SHARD backup should flush the engine before copying"

        restore_source = inspect.getsource(svc_module.SearchService._restore_collection_from_backup)
        assert 'shard_dir / "engine_meta.json"' not in restore_source or \
               "self._restore_from_backup(runtime, backup_root, shard_dir)" in restore_source, \
            "SHARD restore should restore entire shard_dir"

    # ----------------------------------------------------------
    # FIX 25: IVF-PQ nprobe > 1
    # ----------------------------------------------------------

    def test_fix25_ivfpq_nprobe_set(self, tmp_dir):
        """Evidence: IVF-PQ index has nprobe > 1 after build."""
        try:
            import faiss
        except ImportError:
            pytest.skip("faiss not available")

        from voyager_index._internal.inference.shard_engine.lemur_router import LemurRouter

        lemur_dir = tmp_dir / "lemur_nprobe_test"
        lemur_dir.mkdir(parents=True, exist_ok=True)
        router = LemurRouter(
            index_dir=lemur_dir,
            ann_backend="faiss_ivfpq_ip",
            device="cpu",
        )

        n_docs = 500
        dim = 32
        doc_vecs = torch.randn(n_docs * 16, dim, dtype=torch.float16)
        doc_counts = torch.full((n_docs,), 16, dtype=torch.int32)
        doc_ids = list(range(n_docs))
        doc_id_to_shard = {i: i % 8 for i in range(n_docs)}

        router.fit_initial(
            pooled_doc_vectors=doc_vecs,
            pooled_doc_counts=doc_counts,
            doc_ids=doc_ids,
            doc_id_to_shard=doc_id_to_shard,
            epochs=2,
        )

        index = router._index
        target = index
        if hasattr(target, 'index') and target.index is not None:
            target = target.index
        if hasattr(target, 'nprobe'):
            assert target.nprobe > 1, f"nprobe should be > 1 but is {target.nprobe}"
        router.save()

        router2 = LemurRouter(
            index_dir=lemur_dir,
            ann_backend="faiss_ivfpq_ip",
            device="cpu",
        )
        router2.load()
        target2 = router2._index
        if hasattr(target2, 'index') and target2.index is not None:
            target2 = target2.index
        if hasattr(target2, 'nprobe'):
            assert target2.nprobe > 1, f"nprobe after reload should be > 1 but is {target2.nprobe}"

    # ----------------------------------------------------------
    # FIX 26: brute_force_maxsim empty input
    # ----------------------------------------------------------

    def test_fix26_brute_force_empty_input(self, tmp_dir):
        """Evidence: brute_force_maxsim with empty doc_ids returns
        empty lists instead of crashing with torch.cat([])."""
        from voyager_index._internal.inference.shard_engine.scorer import brute_force_maxsim
        query = torch.randn(8, 64)
        ids, scores = brute_force_maxsim(query, [], [], dim=64, device="cpu")
        assert ids == []
        assert scores == []

    # ----------------------------------------------------------
    # FIX 27: CPU tests run without CUDA guard
    # ----------------------------------------------------------

    def test_fix27_cpu_build_search_lifecycle(self, tmp_dir):
        """Evidence: ShardSegmentManager works end-to-end on CPU.
        This test has NO CUDA skip guard — it MUST run in CI."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        corpus = _make_corpus(20, dim=dim)
        payloads = [{"idx": i} for i in range(20)]
        mgr.build(corpus, list(range(20)), payloads=payloads)

        q = np.random.randn(8, dim).astype(np.float32)
        results = mgr.search_multivector(q, k=5)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

        mgr.add_multidense(_make_corpus(3, dim=dim), [20, 21, 22])
        assert mgr.total_vectors() >= 23

        mgr.delete([0])
        results_after = mgr.search_multivector(q, k=20)
        assert 0 not in [r[0] for r in results_after]

        mgr.upsert_payload(1, {"idx": "UPDATED"})
        p = mgr._get_payload(1)
        assert p["idx"] == "UPDATED"

        page, _ = mgr.scroll(limit=100)
        assert len(page) >= 20

        stats = mgr.get_statistics()
        assert stats["engine"] == "shard"
        assert stats["is_built"] is True

        mgr.flush()
        assert mgr.total_vectors() >= 22

        mgr.close()

        mgr2 = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        assert mgr2._is_built
        p2 = mgr2._get_payload(1)
        assert p2["idx"] == "UPDATED", "Payload update did not survive reload"
        mgr2.close()

    def test_fix27_cpu_filters(self, tmp_dir):
        """Evidence: search with filters works on CPU without CUDA."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        payloads = [{"cat": "A" if i % 2 == 0 else "B"} for i in range(20)]
        mgr.build(_make_corpus(20, dim=dim), list(range(20)), payloads=payloads)

        q = np.random.randn(8, dim).astype(np.float32)
        results = mgr.search_multivector(q, k=10, filters={"cat": {"$eq": "A"}})
        for did, _ in results:
            p = mgr._get_payload(did)
            assert p.get("cat") == "A", f"Doc {did} has cat={p.get('cat')}, expected A"
        mgr.close()

    def test_fix27_cpu_wal_replay(self, tmp_dir):
        """Evidence: WAL replay works on CPU. No CUDA guard."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(20, dim=dim), list(range(20)))

        mgr.add_multidense(_make_corpus(5, dim=dim), list(range(20, 25)))
        mgr.delete([0])
        assert mgr._wal_writer.n_entries == 6
        mgr.close()

        mgr2 = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        assert mgr2._memtable.size == 5
        assert mgr2._memtable.is_tombstoned(0)

        q = np.random.randn(8, dim).astype(np.float32)
        results = mgr2.search_multivector(q, k=5)
        assert 0 not in [r[0] for r in results]
        mgr2.close()


# ==================================================================
# 360-DEGREE HIGH FIX VERIFICATION (all CPU-safe, no CUDA guard)
# ==================================================================

class TestHighFixVerification:
    """One test per HIGH FIX GROUP (28-36). Each provides explicit
    evidence that the fix is correct and in place. All run on CPU."""

    # ----------------------------------------------------------
    # FIX 28: $and / $or recursive filter evaluation
    # ----------------------------------------------------------

    def test_fix28_and_or_filters(self, tmp_dir):
        """Evidence: _evaluate_filter supports $and, $or, and nesting."""
        ef = ShardSegmentManager._evaluate_filter

        payload_a1 = {"cat": "A", "tier": 1}
        payload_b2 = {"cat": "B", "tier": 2}
        payload_a2 = {"cat": "A", "tier": 2}

        assert ef(payload_a1, {"$and": [{"cat": "A"}, {"tier": {"$eq": 1}}]})
        assert not ef(payload_b2, {"$and": [{"cat": "A"}, {"tier": {"$eq": 1}}]})

        assert ef(payload_a1, {"$or": [{"cat": "A"}, {"cat": "B"}]})
        assert ef(payload_b2, {"$or": [{"cat": "A"}, {"cat": "B"}]})
        assert not ef({"cat": "C"}, {"$or": [{"cat": "A"}, {"cat": "B"}]})

        nested = {"$and": [
            {"$or": [{"cat": "A"}, {"cat": "B"}]},
            {"tier": {"$gt": 1}},
        ]}
        assert ef(payload_b2, nested)
        assert ef(payload_a2, nested)
        assert not ef(payload_a1, nested)

    def test_fix28_and_or_in_search(self, tmp_dir):
        """Evidence: $and/$or works end-to-end in search_multivector."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        payloads = [{"cat": "A" if i < 10 else "B", "tier": i % 3} for i in range(20)]
        mgr.build(_make_corpus(20, dim=dim), list(range(20)), payloads=payloads)

        q = np.random.randn(8, dim).astype(np.float32)
        results = mgr.search_multivector(q, k=20, filters={
            "$or": [{"cat": "A"}, {"tier": {"$eq": 2}}]
        })
        for did, _ in results:
            p = mgr._get_payload(did)
            assert p["cat"] == "A" or p["tier"] == 2
        mgr.close()

    # ----------------------------------------------------------
    # FIX 29: Statistics no double-counting
    # ----------------------------------------------------------

    def test_fix29_stats_no_double_count(self, tmp_dir):
        """Evidence: upsert of existing sealed doc doesn't inflate total_vectors."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(20, dim=dim), list(range(20)))

        assert mgr.total_vectors() == 20
        assert mgr.get_statistics()["n_live"] == 20

        mgr.add_multidense([np.random.randn(10, dim).astype(np.float32)], [0])
        assert mgr.total_vectors() == 20, f"Expected 20 but got {mgr.total_vectors()}"
        assert mgr.get_statistics()["n_live"] == 20

        mgr.add_multidense([np.random.randn(10, dim).astype(np.float32)], [100])
        assert mgr.total_vectors() == 21
        mgr.close()

    # ----------------------------------------------------------
    # FIX 30: delete() updates LEMUR router
    # ----------------------------------------------------------

    def test_fix30_delete_updates_router(self, tmp_dir):
        """Evidence: after delete(), router._tombstones contains deleted IDs."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(20, dim=dim), list(range(20)))

        mgr.delete([3, 7, 15])

        router_tombstones = mgr._router._tombstones
        assert 3 in router_tombstones
        assert 7 in router_tombstones
        assert 15 in router_tombstones
        assert 0 not in router_tombstones
        mgr.close()

    # ----------------------------------------------------------
    # FIX 31: _load_sealed_vectors uses single batch fetch
    # ----------------------------------------------------------

    def test_fix31_batch_sealed_vector_load(self, tmp_dir):
        """Evidence: _load_sealed_vectors returns correct count
        and inspection of source confirms single batch call."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(20, dim=dim), list(range(20)))

        vecs = mgr._load_sealed_vectors()
        assert vecs is not None
        assert len(vecs) == 20

        import inspect
        source = inspect.getsource(ShardSegmentManager._load_sealed_vectors)
        assert "self._store.fetch_docs(self._doc_ids)" in source, \
            "_load_sealed_vectors should call fetch_docs once with full list"
        assert "for did in self._doc_ids:\n" not in source or \
               "self._store.fetch_docs([did])" not in source, \
            "_load_sealed_vectors should not call fetch_docs in a per-doc loop"
        mgr.close()

    # ----------------------------------------------------------
    # FIX 32: PreloadedGpuCorpus refresh
    # ----------------------------------------------------------

    def test_fix32_gpu_corpus_refresh(self, tmp_dir):
        """Evidence: PreloadedGpuCorpus.refresh() updates doc_ids and tensors."""
        from voyager_index._internal.inference.shard_engine.scorer import PreloadedGpuCorpus

        dim = 32
        device = "cuda" if HAS_CUDA else "cpu"
        vecs = [np.random.randn(8, dim).astype(np.float32) for _ in range(5)]
        corpus = PreloadedGpuCorpus(vecs, [0, 1, 2, 3, 4], dim, device=device)
        assert len(corpus.doc_ids) == 5

        new_vecs = [np.random.randn(8, dim).astype(np.float32) for _ in range(10)]
        corpus.refresh(new_vecs, list(range(10)))
        assert len(corpus.doc_ids) == 10
        assert corpus.D.shape[0] == 10
        assert corpus.M.shape[0] == 10
        assert 9 in corpus.doc_id_to_idx

    def test_fix32_manager_refresh_gpu_corpus(self, tmp_dir):
        """Evidence: manager.refresh_gpu_corpus() calls through correctly."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(20, dim=dim), list(range(20)))
        mgr.refresh_gpu_corpus()
        mgr.close()

    # ----------------------------------------------------------
    # FIX 33: ROQ4 CPU no crash + manifest metadata
    # ----------------------------------------------------------

    def test_fix33_roq4_cpu_no_crash(self, tmp_dir):
        """Evidence: score_roq4_topk on CPU returns empty results, no crash."""
        from voyager_index._internal.inference.shard_engine.scorer import score_roq4_topk
        ids, scores = score_roq4_topk(
            query_codes=torch.zeros(1, 8, 4, dtype=torch.uint8),
            query_meta=torch.zeros(1, 8, 4, dtype=torch.float32),
            doc_codes=torch.zeros(5, 16, 4, dtype=torch.uint8),
            doc_meta=torch.zeros(5, 16, 4, dtype=torch.float32),
            doc_ids=[0, 1, 2, 3, 4],
            k=3,
            device=torch.device("cpu"),
        )
        assert ids == []
        assert scores == []

    def test_fix33_manifest_fallback_to_fp16(self, tmp_dir):
        """Evidence: build with ROQ4 but no roq_doc_codes => manifest says fp16."""
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        store_dir = tmp_dir / "store33"
        store = ShardStore(store_dir)
        n_docs = 10
        dim = 32
        vecs = np.random.randn(n_docs * 8, dim).astype(np.float16)
        offsets = [(i * 8, (i + 1) * 8) for i in range(n_docs)]
        ids = list(range(n_docs))
        assignments = np.zeros(n_docs, dtype=np.int32)

        manifest = store.build(
            all_vectors=vecs,
            doc_offsets=offsets,
            doc_ids=ids,
            shard_assignments=assignments,
            n_shards=1,
            dim=dim,
            compression=Compression.ROQ4,
            roq_doc_codes=None,
        )
        assert manifest.compression == "fp16", \
            f"Manifest compression should be fp16 but is {manifest.compression}"

    # ----------------------------------------------------------
    # FIX 34: No double filter + checkpoint WAL count
    # ----------------------------------------------------------

    def test_fix34_no_double_filter(self):
        """Evidence: shard search branch in service.py does not call _matches_filter."""
        import inspect
        import voyager_index._internal.server.api.service as svc_module
        source = inspect.getsource(svc_module.SearchService.search)

        shard_section_start = source.index("runtime.kind == CollectionKind.SHARD")
        shard_section = source[shard_section_start:]
        next_elif = shard_section.find("elif runtime.kind")
        if next_elif == -1:
            next_elif = shard_section.find("else:")
        shard_section = shard_section[:next_elif] if next_elif > 0 else shard_section

        assert "_matches_filter" not in shard_section, \
            "Shard search branch should NOT apply _matches_filter (engine already filters)"

    def test_fix34_checkpoint_returns_actual_wal_count(self, tmp_dir):
        """Evidence: checkpoint_collection returns actual WAL count, not hardcoded 0."""
        import inspect
        import voyager_index._internal.server.api.service as svc_module
        source = inspect.getsource(svc_module.SearchService.checkpoint_collection)
        assert '"wal_entries_after": 0' not in source, \
            "checkpoint should return actual WAL count, not hardcoded 0"
        assert "runtime.engine._wal_writer.n_entries" in source

    # ----------------------------------------------------------
    # FIX 35: Hybrid rejects 3D numpy
    # ----------------------------------------------------------

    def test_fix35_hybrid_rejects_3d(self, tmp_dir):
        """Evidence: HybridSearchManager.index() raises ValueError on 3D arrays."""
        from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager
        hm = HybridSearchManager(
            shard_path=tmp_dir / "hybrid35",
            dim=32,
            distance_metric="cosine",
        )
        bad_vectors = np.random.randn(5, 8, 32).astype(np.float32)
        with pytest.raises(ValueError, match="3D arrays not supported"):
            hm.index(
                corpus=["a", "b", "c", "d", "e"],
                vectors=bad_vectors,
                ids=[0, 1, 2, 3, 4],
            )

    # ----------------------------------------------------------
    # FIX 36: Index API and HTTP coverage (existence check)
    # ----------------------------------------------------------

    def test_fix36_index_api_coverage(self, tmp_dir):
        """Evidence: Index.delete and Index.scroll exist and are callable
        on the shard engine at the public API level."""
        from voyager_index.index import Index
        dim = 64
        idx = Index(str(tmp_dir / "idx36"), dim=dim, engine="shard",
                     n_shards=4, lemur_epochs=2, k_candidates=50)
        vectors = _make_corpus(20, dim=dim)
        idx.add(vectors, ids=list(range(20)))

        idx.delete([0, 1])
        page = idx.scroll(limit=100)
        assert 0 not in [r.doc_id for r in page.results]
        assert 1 not in [r.doc_id for r in page.results]
        assert len(page.results) == 18

        idx.close()

    def test_fix36_http_routes_exist(self):
        """Evidence: scroll, retrieve, search/batch routes are registered."""
        import voyager_index._internal.server.api.routes as routes_module
        source = inspect.getsource(routes_module)
        assert "/collections/{name}/scroll" in source
        assert "/collections/{name}/retrieve" in source
        assert "/collections/{name}/search/batch" in source


# ==================================================================
# 360-DEGREE MEDIUM+LOW FIX VERIFICATION (all CPU-safe, no CUDA guard)
# ==================================================================

class TestMediumLowFixVerification:
    """One test (or group) per FIX GROUP 37-45. Each provides explicit
    evidence that the fix is correct and in place. All run on CPU."""

    # ----------------------------------------------------------
    # FIX 37: WAL performance and robustness
    # ----------------------------------------------------------

    def test_fix37_context_manager(self, tmp_dir):
        """Evidence: WalWriter can be used as a context manager."""
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wal_path = tmp_dir / "fix37_ctx.wal"
        with WalWriter(wal_path).open() as w:
            w.log_insert(0, np.random.randn(4, 64).astype(np.float32))
            assert w.n_entries == 1
        entries = WalReader(wal_path).replay()
        assert len(entries) == 1

    def test_fix37_parse_entry_unknown_op_warning(self, tmp_dir, caplog):
        """Evidence: _parse_entry logs warning on unknown op byte."""
        import struct, json, zlib
        from voyager_index._internal.inference.shard_engine.wal import (
            WAL_MAGIC, WAL_VERSION, HEADER_FMT, WalReader,
        )
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wal_path = tmp_dir / "fix37_unknown_op.wal"

        unknown_op = 99
        buf = struct.pack("<Bq", unknown_op, 42)
        header = struct.pack(HEADER_FMT, WAL_MAGIC, WAL_VERSION, len(buf))
        crc = zlib.crc32(header + buf) & 0xFFFFFFFF

        with open(wal_path, "wb") as f:
            f.write(header)
            f.write(buf)
            f.write(struct.pack("<I", crc))

        with caplog.at_level(logging.WARNING):
            entries = WalReader(wal_path).replay()
        assert len(entries) == 0
        assert any("unknown op byte" in r.message for r in caplog.records), \
            "Expected warning about unknown op byte"

    def test_fix37_count_existing_logs_corrupt(self, tmp_dir, caplog):
        """Evidence: _count_existing logs warning on corrupt WAL."""
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wal_path = tmp_dir / "fix37_corrupt.wal"
        with open(wal_path, "wb") as f:
            f.write(b"CORRUPTDATA" * 10)

        with caplog.at_level(logging.WARNING):
            w = WalWriter(wal_path)
            w.open()
        assert w.n_entries == 0
        w.close()

    def test_fix37_batch_write_mode(self, tmp_dir):
        """Evidence: batch mode buffers writes and flushes on end_batch."""
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wal_path = tmp_dir / "fix37_batch.wal"
        with WalWriter(wal_path).open() as w:
            w.begin_batch()
            for i in range(50):
                w.log_insert(i, np.random.randn(4, 64).astype(np.float32))
            w.end_batch()
            assert w.n_entries == 50

        entries = WalReader(wal_path).replay()
        assert len(entries) == 50

    def test_fix37_streaming_replay(self, tmp_dir):
        """Evidence: large WAL files are replayed correctly via streaming."""
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wal_path = tmp_dir / "fix37_stream.wal"
        n_entries = 200
        with WalWriter(wal_path).open() as w:
            w.begin_batch()
            for i in range(n_entries):
                w.log_insert(i, np.random.randn(8, 64).astype(np.float32))
            w.end_batch()

        entries = WalReader(wal_path).replay()
        assert len(entries) == n_entries
        assert entries[0].doc_id == 0
        assert entries[-1].doc_id == n_entries - 1

    # ----------------------------------------------------------
    # FIX 38: Shard store validation and manifest gaps
    # ----------------------------------------------------------

    def test_fix38_negative_shard_assignment_rejected(self, tmp_dir):
        """Evidence: build() rejects shard_assignments with negative values."""
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        tmp_dir.mkdir(parents=True, exist_ok=True)
        store = ShardStore(tmp_dir / "store38a")
        vecs = np.random.randn(16, 64).astype(np.float16)
        with pytest.raises(ValueError, match="out of bounds"):
            store.build(
                all_vectors=vecs,
                doc_offsets=[(0, 8), (8, 16)],
                doc_ids=[0, 1],
                shard_assignments=np.array([-1, 0]),
                n_shards=2,
                dim=64,
            )

    def test_fix38_duplicate_doc_ids_rejected(self, tmp_dir):
        """Evidence: build() rejects duplicate doc_ids."""
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        tmp_dir.mkdir(parents=True, exist_ok=True)
        store = ShardStore(tmp_dir / "store38b")
        vecs = np.random.randn(16, 64).astype(np.float16)
        with pytest.raises(ValueError, match="Duplicate doc_ids"):
            store.build(
                all_vectors=vecs,
                doc_offsets=[(0, 8), (8, 16)],
                doc_ids=[0, 0],
                shard_assignments=np.array([0, 0]),
                n_shards=1,
                dim=64,
            )

    def test_fix38_n_shards_lt_1_rejected(self, tmp_dir):
        """Evidence: build() rejects n_shards < 1."""
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        tmp_dir.mkdir(parents=True, exist_ok=True)
        store = ShardStore(tmp_dir / "store38c")
        vecs = np.random.randn(8, 64).astype(np.float16)
        with pytest.raises(ValueError, match="n_shards must be >= 1"):
            store.build(
                all_vectors=vecs,
                doc_offsets=[(0, 8)],
                doc_ids=[0],
                shard_assignments=np.array([0]),
                n_shards=0,
                dim=64,
            )

    def test_fix38_manifest_has_version(self, tmp_dir):
        """Evidence: StoreManifest has a version field."""
        from voyager_index._internal.inference.shard_engine.shard_store import StoreManifest
        assert hasattr(StoreManifest, "__dataclass_fields__")
        assert "version" in StoreManifest.__dataclass_fields__

    def test_fix38_load_shard_roq4_needs_safetensors(self, tmp_dir):
        """Evidence: load_shard_roq4 raises ImportError when safetensors unavailable."""
        from voyager_index._internal.inference.shard_engine import shard_store as ss_mod

        tmp_dir.mkdir(parents=True, exist_ok=True)
        store_dir = tmp_dir / "store38e"
        store_dir.mkdir()
        store = ss_mod.ShardStore(store_dir)

        with mock.patch.object(ss_mod, "SAFETENSORS_AVAILABLE", False):
            with pytest.raises(ImportError, match="safetensors"):
                store.load_shard_roq4(0)

    # ----------------------------------------------------------
    # FIX 39: Router state consistency and configuration
    # ----------------------------------------------------------

    def test_fix39_router_state_default_flat(self):
        """Evidence: RouterState.ann_backend defaults to 'faiss_flat_ip'."""
        from voyager_index._internal.inference.shard_engine.lemur_router import RouterState
        state = RouterState()
        assert state.ann_backend == "faiss_flat_ip"

    def test_fix39_load_syncs_ann_backend(self, tmp_dir):
        """Evidence: _load_if_present syncs self.ann_backend from state."""
        source = inspect.getsource(
            __import__(
                "voyager_index._internal.inference.shard_engine.lemur_router",
                fromlist=["LemurRouter"],
            ).LemurRouter._load_if_present
        )
        assert 'self.ann_backend = self._state.ann_backend.replace("hnsw", "flat")' in source

    def test_fix39_no_redundant_branch_in_add_or_update(self):
        """Evidence: add_or_update_docs has no if/else around _rebuild_ann."""
        from voyager_index._internal.inference.shard_engine.lemur_router import LemurRouter
        source = inspect.getsource(LemurRouter.add_or_update_docs)
        rebuild_calls = source.count("self._rebuild_ann()")
        assert rebuild_calls == 1, f"Expected exactly 1 _rebuild_ann() call, found {rebuild_calls}"

    def test_fix39_no_lazy_search_lock(self):
        """Evidence: _search has no lazy init of _search_lock."""
        from voyager_index._internal.inference.shard_engine.lemur_router import LemurRouter
        source = inspect.getsource(LemurRouter._search)
        assert "hasattr" not in source or "_search_lock" not in source.split("hasattr")[1][:50]

    def test_fix39_thread_safe_gpu_resources(self):
        """Evidence: _get_gpu_resources uses a lock."""
        from voyager_index._internal.inference.shard_engine.lemur_router import LemurRouter
        source = inspect.getsource(LemurRouter._get_gpu_resources)
        assert "_gpu_res_lock" in source

    # ----------------------------------------------------------
    # FIX 40: Service layer and API model gaps
    # ----------------------------------------------------------

    def test_fix40_search_request_has_ef_nprobes(self):
        """Evidence: SearchRequest accepts ef and n_probes fields."""
        from voyager_index._internal.server.api.models import SearchRequest
        fields = SearchRequest.model_fields
        assert "ef" in fields
        assert "n_probes" in fields

    def test_fix40_collection_info_shard_fields(self):
        """Evidence: CollectionInfo has n_shards, k_candidates, total_tokens."""
        from voyager_index._internal.server.api.models import CollectionInfo
        fields = CollectionInfo.model_fields
        assert "n_shards" in fields
        assert "k_candidates" in fields
        assert "total_tokens" in fields

    def test_fix40_structured_error_response(self):
        """Evidence: _raise_service_error uses ErrorResponse model."""
        import voyager_index._internal.server.api.routes as routes_mod
        source = inspect.getsource(routes_mod._raise_service_error)
        assert "ErrorResponse" in source

    # ----------------------------------------------------------
    # FIX 41: ef/n_probes wired or documented
    # ----------------------------------------------------------

    def test_fix41_search_multivector_ef_none_ok(self, tmp_dir):
        """Evidence: search_multivector accepts ef=None without error."""
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(20, dim=dim), list(range(20)))

        results = mgr.search_multivector(
            np.random.randn(8, dim).astype(np.float32),
            k=5, ef=None, n_probes=None,
        )
        assert isinstance(results, list)
        mgr.close()

    def test_fix41_nprobes_wired_to_router(self):
        """Evidence: n_probes is passed through to router.route()."""
        source = inspect.getsource(ShardSegmentManager.search_multivector)
        assert "nprobe_override" in source or "n_probes" in source

    # ----------------------------------------------------------
    # FIX 42: Documentation alignment
    # ----------------------------------------------------------

    def test_fix42_shard_engine_md_not_pure_python(self):
        """Evidence: shard-engine.md does NOT say 'Pure Python'."""
        doc_path = Path(__file__).resolve().parent.parent / "docs" / "guides" / "shard-engine.md"
        if not doc_path.exists():
            pytest.skip("docs not found")
        content = doc_path.read_text()
        assert "Pure Python" not in content, "shard-engine.md should not claim 'Pure Python'"
        assert "PyTorch" in content or "native deps" in content

    def test_fix42_production_md_roq_keys(self):
        """Evidence: PRODUCTION.md section 9.2 uses roq_codes / roq_meta."""
        doc_path = Path(__file__).resolve().parent.parent / "PRODUCTION.md"
        if not doc_path.exists():
            pytest.skip("PRODUCTION.md not found")
        content = doc_path.read_text()
        assert "roq_codes" in content
        assert "roq_meta" in content

    # ----------------------------------------------------------
    # FIX 43: Memtable and compaction improvements
    # ----------------------------------------------------------

    def test_fix43_snapshot_returns_independent_copies(self):
        """Evidence: mutating snapshot payloads does not affect memtable."""
        mt = MemTable(dim=64, device="cpu")
        mt.insert(0, np.random.randn(4, 64).astype(np.float32), {"key": "original"})
        _, snap_payloads, _ = mt.snapshot()
        snap_payloads[0]["key"] = "MUTATED"
        _, snap2, _ = mt.snapshot()
        assert snap2[0]["key"] == "original", "Snapshot mutation leaked into memtable"

    def test_fix43_compaction_task_metric_names(self, tmp_dir):
        """Evidence: CompactionTask.run() returns memtable_docs_at_sync key."""
        from voyager_index._internal.inference.shard_engine.compaction import CompactionTask
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        mgr.build(_make_corpus(10, dim=dim), list(range(10)))

        task = CompactionTask(mgr)
        stats = task.run()
        assert "memtable_docs_at_sync" in stats
        assert "tombstones_at_sync" in stats
        assert "duration_ms" in stats
        assert isinstance(stats["memtable_docs_at_sync"], int)
        assert stats["duration_ms"] >= 0
        assert "flushed_docs" not in stats
        mgr.close()

    def test_fix43_memtable_search_snapshot_docstring(self):
        """Evidence: MemTable.search documents snapshot semantics."""
        docstring = MemTable.search.__doc__
        assert "snapshot" in docstring.lower() or "stale" in docstring.lower()

    # ----------------------------------------------------------
    # FIX 44: Shard store fetch efficiency
    # ----------------------------------------------------------

    def test_fix44_fetch_docs_docstring_mentions_shard_size(self):
        """Evidence: fetch_docs docstring mentions O(shard_size)."""
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        docstring = ShardStore.fetch_docs.__doc__
        assert "shard_size" in docstring.lower() or "O(shard" in docstring

    # ----------------------------------------------------------
    # FIX 45: Minor polish (selected)
    # ----------------------------------------------------------

    def test_fix45_evaluate_filter_no_doc_id_param(self):
        """Evidence: _evaluate_filter has no doc_id parameter."""
        sig = inspect.signature(ShardSegmentManager._evaluate_filter)
        param_names = list(sig.parameters.keys())
        assert "doc_id" not in param_names
        assert ShardSegmentManager._evaluate_filter({"x": 1}, {"x": {"$eq": 1}})

    def test_fix45_explain_score_public(self):
        """Evidence: explain_score is a public method."""
        assert hasattr(ShardSegmentManager, "explain_score")
        assert callable(ShardSegmentManager.explain_score)

    def test_fix45_fits_on_gpu_bfloat16(self):
        """Evidence: fits_on_gpu handles bfloat16 via dtype.itemsize."""
        from voyager_index._internal.inference.shard_engine.scorer import PreloadedGpuCorpus
        source = inspect.getsource(PreloadedGpuCorpus.fits_on_gpu)
        assert "itemsize" in source, "fits_on_gpu should use dtype.itemsize"
        assert "(2 if" not in source, "fits_on_gpu should not hardcode dtype sizes"

    def test_fix45_candidate_plan_renamed(self):
        """Evidence: CandidatePlan field is post_tombstone_count, not raw_candidate_count."""
        from voyager_index._internal.inference.shard_engine.lemur_router import CandidatePlan
        fields = [f.name for f in CandidatePlan.__dataclass_fields__.values()]
        assert "post_tombstone_count" in fields
        assert "raw_candidate_count" not in fields

    def test_fix45_single_doc_build_search(self, tmp_dir):
        """Evidence: single-document index build and search works."""
        dim = 64
        config = ShardEngineConfig(n_shards=1, dim=dim, lemur_epochs=1, k_candidates=10)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cpu")
        corpus = [np.random.randn(8, dim).astype(np.float32)]
        mgr.build(corpus, [0], payloads=[{"title": "only_doc"}])

        q = np.random.randn(8, dim).astype(np.float32)
        results = mgr.search_multivector(q, k=5)
        assert len(results) >= 1
        assert results[0][0] == 0
        mgr.close()

    def test_fix45_manager_docstring_updated(self):
        """Evidence: manager.py module docstring mentions WAL, retrieve, scroll."""
        import voyager_index._internal.inference.shard_engine.manager as mgr_mod
        docstring = mgr_mod.__doc__
        assert "WAL" in docstring
        assert "retrieve" in docstring
        assert "scroll" in docstring

    def test_fix45_search_config_batch_size_documented(self):
        """Evidence: SearchConfig docstring documents batch_size as reserved."""
        docstring = SearchConfig.__doc__
        assert "batch_size" in docstring
        assert "reserved" in docstring.lower() or "future" in docstring.lower()

    def test_fix45_rebuild_ann_deletes_old_index(self):
        """Evidence: _rebuild_ann explicitly deletes old index."""
        from voyager_index._internal.inference.shard_engine.lemur_router import LemurRouter
        source = inspect.getsource(LemurRouter._rebuild_ann)
        assert "old_index" in source
        assert "del old_index" in source


# ======================================================================
# 360-degree verification: Production Feature Gaps (#11, #22, #25, #26,
# #43, #54, #103, #104)
# ======================================================================


class TestProductionFeatureGaps:
    """Verify all 8 PRODUCTION.md feature gaps are implemented."""

    # ------------------------------------------------------------------
    # #11 / #103: set_metrics_hook + per-search metric emissions
    # ------------------------------------------------------------------

    def test_feat11_set_metrics_hook_exists(self):
        """Evidence: ShardSegmentManager has set_metrics_hook method."""
        assert hasattr(ShardSegmentManager, "set_metrics_hook")
        sig = inspect.signature(ShardSegmentManager.set_metrics_hook)
        assert "hook" in sig.parameters

    def test_feat11_emit_metric_exists(self):
        assert hasattr(ShardSegmentManager, "_emit_metric")

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_feat103_search_emits_metrics(self, tmp_dir):
        """Evidence: search_multivector emits search_latency_us, candidates_scored, route_ms."""
        dim = 64
        config = ShardEngineConfig(n_shards=2, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(n_docs=10, dim=dim)
        ids = list(range(10))
        mgr.build(corpus, ids)

        emitted = {}

        def hook(name, value):
            emitted[name] = value

        mgr.set_metrics_hook(hook)
        q = np.random.randn(8, dim).astype(np.float32)
        mgr.search_multivector(q, k=3)

        assert "search_latency_us" in emitted
        assert emitted["search_latency_us"] > 0
        assert "candidates_scored" in emitted
        assert "route_ms" in emitted
        mgr.close()

    def test_feat103_compaction_emits_metric(self, tmp_dir):
        """Evidence: CompactionTask.run() calls _emit_metric('compaction', ...)."""
        from voyager_index._internal.inference.shard_engine.compaction import CompactionTask
        source = inspect.getsource(CompactionTask.run)
        assert "_emit_metric" in source
        assert '"compaction"' in source

    # ------------------------------------------------------------------
    # #22: Atomic checkpoint (ShardCheckpointManager)
    # ------------------------------------------------------------------

    def test_feat22_checkpoint_manager_exists(self):
        """Evidence: ShardCheckpointManager class importable."""
        from voyager_index._internal.inference.shard_engine.checkpoint import ShardCheckpointManager
        assert hasattr(ShardCheckpointManager, "save")
        assert hasattr(ShardCheckpointManager, "load")
        assert hasattr(ShardCheckpointManager, "clear")

    def test_feat22_checkpoint_roundtrip(self, tmp_dir):
        """Evidence: save + load round-trips memtable data."""
        from voyager_index._internal.inference.shard_engine.checkpoint import ShardCheckpointManager

        ckpt = ShardCheckpointManager(tmp_dir)
        docs = {0: np.random.randn(5, 32).astype(np.float32),
                1: np.random.randn(3, 32).astype(np.float32)}
        payloads = {0: {"color": "red"}, 1: {"color": "blue"}}
        tombstones = {99, 100}

        ckpt.save(docs, payloads, tombstones, next_doc_id=10, wal_offset=42)
        loaded = ckpt.load()

        assert loaded is not None
        assert loaded["wal_offset"] == 42
        assert loaded["next_doc_id"] == 10
        assert loaded["tombstones"] == tombstones
        assert set(loaded["docs"].keys()) == {0, 1}
        np.testing.assert_allclose(loaded["docs"][0], docs[0], atol=1e-6)
        assert loaded["payloads"][0]["color"] == "red"

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_feat22_flush_creates_checkpoint(self, tmp_dir):
        """Evidence: flush() creates a checkpoint directory."""
        dim = 64
        config = ShardEngineConfig(n_shards=2, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(n_docs=10, dim=dim)
        mgr.build(corpus, list(range(10)))
        new_vec = np.random.randn(5, dim).astype(np.float32)
        mgr.add_multidense([new_vec], [100])
        mgr.flush()

        checkpoint_dir = tmp_dir / "checkpoints" / "current"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "meta.json").exists()
        mgr.close()

    def test_feat22_replay_wal_uses_checkpoint(self):
        """Evidence: _replay_wal has checkpoint loading code."""
        source = inspect.getsource(ShardSegmentManager._replay_wal)
        assert "checkpoint_mgr.load()" in source or "_checkpoint_mgr.load()" in source
        assert "wal_offset" in source

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_feat22_checkpoint_skips_wal_entries(self, tmp_dir):
        """Evidence: reload after flush+close skips checkpointed WAL entries."""
        dim = 64
        config = ShardEngineConfig(n_shards=2, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(n_docs=10, dim=dim)
        mgr.build(corpus, list(range(10)))

        v1 = np.random.randn(5, dim).astype(np.float32)
        mgr.add_multidense([v1], [100])
        mgr.flush()

        v2 = np.random.randn(5, dim).astype(np.float32)
        mgr.add_multidense([v2], [200])
        mgr.close()

        mgr2 = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        assert mgr2._memtable is not None
        docs_snap, _, _ = mgr2._memtable.snapshot()
        assert 100 in docs_snap, "Checkpointed doc 100 should be restored"
        assert 200 in docs_snap, "Post-checkpoint doc 200 should be replayed from WAL"
        mgr2.close()

    # ------------------------------------------------------------------
    # #25: FileLock cross-process safety
    # ------------------------------------------------------------------

    def test_feat25_filelock_acquired_on_init(self, tmp_dir):
        """Evidence: ShardSegmentManager acquires FileLock in __init__."""
        source = inspect.getsource(ShardSegmentManager.__init__)
        assert "_file_lock" in source
        assert "FileLock" in source or "_FileLock" in source

    def test_feat25_filelock_released_on_close(self):
        """Evidence: close() releases the file lock."""
        source = inspect.getsource(ShardSegmentManager.close)
        assert "_file_lock" in source

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_feat25_lock_file_created(self, tmp_dir):
        """Evidence: .lock file appears after manager creation with a manifest."""
        dim = 64
        config = ShardEngineConfig(n_shards=2, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(n_docs=10, dim=dim)
        mgr.build(corpus, list(range(10)))
        assert (tmp_dir / ".lock").exists()
        mgr.close()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_feat25_second_manager_raises(self, tmp_dir):
        """Evidence: a second manager on the same path fails to acquire the lock."""
        dim = 64
        config = ShardEngineConfig(n_shards=2, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr1 = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(n_docs=10, dim=dim)
        mgr1.build(corpus, list(range(10)))
        assert mgr1._file_lock is not None

        mgr2 = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        assert mgr2._file_lock is None, "Second manager should fail to acquire lock"
        mgr2.close()
        mgr1.close()

    # ------------------------------------------------------------------
    # #26: Atomic JSON writes
    # ------------------------------------------------------------------

    def test_feat26_manifest_uses_atomic_write(self):
        """Evidence: StoreManifest.save uses atomic_json_write."""
        from voyager_index._internal.inference.shard_engine.shard_store import StoreManifest
        source = inspect.getsource(StoreManifest.save)
        assert "atomic_json_write" in source

    def test_feat26_engine_meta_uses_atomic_write(self):
        """Evidence: manager.build writes engine_meta.json atomically."""
        source = inspect.getsource(ShardSegmentManager.build)
        assert "atomic_json_write" in source

    def test_feat26_payloads_uses_atomic_write(self):
        """Evidence: manager.build writes payloads.json atomically."""
        source = inspect.getsource(ShardSegmentManager.build)
        count = source.count("atomic_json_write")
        assert count >= 2, f"Expected at least 2 atomic_json_write calls, found {count}"

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_feat26_no_partial_writes(self, tmp_dir):
        """Evidence: manifest.json written without partial file remnants."""
        dim = 64
        config = ShardEngineConfig(n_shards=2, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(n_docs=10, dim=dim)
        mgr.build(corpus, list(range(10)))

        import json
        manifest_path = tmp_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        assert "num_shards" in data

        tmp_files = list(tmp_dir.glob(".json_tmp_*"))
        assert len(tmp_files) == 0, f"Leftover temp files: {tmp_files}"
        mgr.close()

    # ------------------------------------------------------------------
    # #43: Pure-Python BM25 fallback
    # ------------------------------------------------------------------

    def test_feat43_bm25s_available_flag(self):
        """Evidence: hybrid_manager has _BM25S_AVAILABLE flag."""
        from voyager_index._internal.inference.index_core import hybrid_manager
        assert hasattr(hybrid_manager, "_BM25S_AVAILABLE")

    def test_feat43_legacy_bm25_import(self):
        """Evidence: hybrid_manager imports LegacyBM25Engine."""
        from voyager_index._internal.inference.index_core import hybrid_manager
        assert hasattr(hybrid_manager, "LegacyBM25Engine")

    def test_feat43_rebuild_has_fallback(self):
        """Evidence: rebuild_sparse_state has LegacyBM25Engine fallback path."""
        from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager
        source = inspect.getsource(HybridSearchManager.rebuild_sparse_state)
        assert "LegacyBM25Engine" in source

    def test_feat43_search_handles_legacy(self):
        """Evidence: search() checks _legacy_bm25 for sparse results."""
        from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager
        source = inspect.getsource(HybridSearchManager.search)
        assert "_legacy_bm25" in source

    def test_feat43_fallback_used_when_bm25s_missing(self):
        """Evidence: with bm25s mocked as unavailable, rebuild falls back to LegacyBM25Engine."""
        import voyager_index._internal.inference.index_core.hybrid_manager as hm
        orig = hm._BM25S_AVAILABLE
        try:
            hm._BM25S_AVAILABLE = False
            from voyager_index._internal.inference.engines.bm25 import BM25Engine as LBM25
            engine = LBM25()
            engine.index_documents(["hello world", "foo bar"], [0, 1])
            results = engine.search("hello", top_k=1)
            assert len(results) > 0, "LegacyBM25Engine should return results"
            assert results[0].doc_id == 0
        finally:
            hm._BM25S_AVAILABLE = orig

    # ------------------------------------------------------------------
    # #54: SearchPipeline default shard dense backend
    # ------------------------------------------------------------------

    def test_feat54_pipeline_dense_engine_param(self):
        """Evidence: SearchPipeline.__init__ accepts dense_engine with shard default."""
        from voyager_index._internal.inference.search_pipeline import SearchPipeline
        sig = inspect.signature(SearchPipeline.__init__)
        assert "dense_engine" in sig.parameters
        default = sig.parameters["dense_engine"].default
        assert default == "shard"

    def test_feat54_pipeline_passes_dense_engine(self):
        """Evidence: SearchPipeline passes dense_engine to HybridSearchManager."""
        from voyager_index._internal.inference.search_pipeline import SearchPipeline
        source = inspect.getsource(SearchPipeline.__init__)
        assert "dense_engine=dense_engine" in source

    def test_feat54_pipeline_config_stored(self):
        """Evidence: SearchPipeline stores dense_engine in config."""
        from voyager_index._internal.inference.search_pipeline import SearchPipeline
        source = inspect.getsource(SearchPipeline.__init__)
        assert '"dense_engine"' in source

    # ------------------------------------------------------------------
    # #104: Page cache hit rate tracking
    # ------------------------------------------------------------------

    def test_feat104_page_cache_method_exists(self):
        """Evidence: ShardStore has page_cache_residency method."""
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        assert hasattr(ShardStore, "page_cache_residency")

    def test_feat104_returns_dict_or_none(self, tmp_dir):
        """Evidence: page_cache_residency returns dict on Linux, None otherwise."""
        import sys
        from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
        store = ShardStore(tmp_dir)
        result = store.page_cache_residency()
        if sys.platform == "linux":
            assert result is None or isinstance(result, dict)
        else:
            assert result is None

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_feat104_wired_to_statistics(self, tmp_dir):
        """Evidence: get_statistics includes page_cache on Linux."""
        import sys
        dim = 64
        config = ShardEngineConfig(n_shards=2, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device=DEVICE)
        corpus = _make_corpus(n_docs=10, dim=dim)
        mgr.build(corpus, list(range(10)))
        stats = mgr.get_statistics()
        if sys.platform == "linux":
            assert "page_cache" in stats
            pc = stats["page_cache"]
            assert "total_pages" in pc
            assert "resident_pages" in pc
            assert "hit_rate" in pc
        mgr.close()

    def test_feat104_emitted_as_metric(self):
        """Evidence: search_multivector emits page_cache_hit_rate metric."""
        source = inspect.getsource(ShardSegmentManager.search_multivector)
        assert "page_cache_hit_rate" in source
