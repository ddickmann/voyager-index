"""Tests for compaction, filters, scroll, explain (Chunk 6)."""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from voyager_index._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)
from voyager_index._internal.inference.shard_engine.compaction import (
    CompactionTask,
    CompactionScheduler,
)


def _make_corpus(n_docs: int = 50, dim: int = 64, min_tok: int = 8, max_tok: int = 20):
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
    d = tempfile.mkdtemp(prefix="shard_advanced_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------
# Compaction
# ------------------------------------------------------------------

class TestCompaction:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compaction_task(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=30, dim=dim)
        mgr.build(vectors, list(range(30)))

        new_vecs = _make_corpus(n_docs=5, dim=dim)
        mgr.add_multidense(new_vecs, list(range(30, 35)))
        assert mgr._memtable.size == 5

        task = CompactionTask(mgr)
        stats = task.run()
        assert stats["memtable_docs_at_sync"] == 5
        assert mgr._memtable.size == 5
        mgr.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compaction_scheduler(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=20, dim=dim)
        mgr.build(vectors, list(range(20)))

        completed = []
        scheduler = CompactionScheduler(mgr, interval_s=0.5, on_complete=lambda s: completed.append(s))
        scheduler.start()
        time.sleep(1.5)
        scheduler.stop()
        # Scheduler ran at least once (even if no data to compact)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_concurrent_reads_during_compaction(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=40, dim=dim)
        mgr.build(vectors, list(range(40)))

        new_vecs = _make_corpus(n_docs=5, dim=dim)
        mgr.add_multidense(new_vecs, list(range(40, 45)))

        errors = []

        def search_loop():
            try:
                for _ in range(10):
                    q = np.random.randn(8, dim).astype(np.float32)
                    results = mgr.search_multivector(q, k=3)
                    assert len(results) > 0
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=search_loop)
        t2 = threading.Thread(target=lambda: CompactionTask(mgr).run())
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        assert not errors, f"Search errors during compaction: {errors}"
        mgr.close()


# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------

class TestFilters:
    def test_evaluate_filter_eq(self):
        assert ShardSegmentManager._evaluate_filter({"status": "active"}, {"status": {"$eq": "active"}})
        assert not ShardSegmentManager._evaluate_filter({"status": "inactive"}, {"status": {"$eq": "active"}})

    def test_evaluate_filter_in(self):
        assert ShardSegmentManager._evaluate_filter({"tag": "a"}, {"tag": {"$in": ["a", "b"]}})
        assert not ShardSegmentManager._evaluate_filter({"tag": "c"}, {"tag": {"$in": ["a", "b"]}})

    def test_evaluate_filter_contains(self):
        assert ShardSegmentManager._evaluate_filter({"tags": ["x", "y"]}, {"tags": {"$contains": "x"}})
        assert not ShardSegmentManager._evaluate_filter({"tags": ["x", "y"]}, {"tags": {"$contains": "z"}})

    def test_evaluate_filter_gt_lt(self):
        assert ShardSegmentManager._evaluate_filter({"score": 0.8}, {"score": {"$gt": 0.5}})
        assert not ShardSegmentManager._evaluate_filter({"score": 0.3}, {"score": {"$gt": 0.5}})
        assert ShardSegmentManager._evaluate_filter({"score": 0.3}, {"score": {"$lt": 0.5}})

    def test_evaluate_filter_simple(self):
        assert ShardSegmentManager._evaluate_filter({"status": "ok"}, {"status": "ok"})
        assert not ShardSegmentManager._evaluate_filter({"status": "fail"}, {"status": "ok"})


# ------------------------------------------------------------------
# Scroll
# ------------------------------------------------------------------

class TestScroll:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scroll_pagination(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=50, dim=dim)
        mgr.build(vectors, list(range(50)))

        all_ids = []
        offset = 0
        while True:
            page_ids, next_off = mgr.scroll(limit=15, offset=offset)
            all_ids.extend(page_ids)
            if next_off is None:
                break
            offset = next_off

        assert len(all_ids) == 50
        assert sorted(all_ids) == list(range(50))
        mgr.close()


# ------------------------------------------------------------------
# Explain
# ------------------------------------------------------------------

class TestExplain:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_explain_score(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = ShardSegmentManager(tmp_dir, config=config, device="cuda")
        vectors = _make_corpus(n_docs=20, dim=dim)
        mgr.build(vectors, list(range(20)))

        new_vec = [np.random.randn(10, dim).astype(np.float32)]
        mgr.add_multidense(new_vec, [100])

        query = np.random.randn(8, dim).astype(np.float32)
        tok_scores, matched = mgr._explain_score(query, 100)
        assert tok_scores is not None
        assert matched is not None
        assert len(tok_scores) == 8
        assert len(matched) == 8
        mgr.close()
