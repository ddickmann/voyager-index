"""Tests for the shard engine production module (Chunk 1)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from voyager_index._internal.inference.shard_engine import ShardSegmentManager
from voyager_index._internal.inference.shard_engine.config import (
    Compression,
    RouterType,
    StorageLayout,
)
from voyager_index._internal.inference.shard_engine.manager import ShardEngineConfig
from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
from voyager_index._internal.inference.shard_engine.scorer import (
    _get_maxsim,
    brute_force_maxsim,
    score_all_docs_topk,
)


def _make_corpus(n_docs: int = 200, dim: int = 128, min_tok: int = 10, max_tok: int = 60):
    """Generate a synthetic multi-vector corpus."""
    rng = np.random.RandomState(42)
    vectors = []
    ids = list(range(n_docs))
    for _ in range(n_docs):
        n_tok = rng.randint(min_tok, max_tok + 1)
        v = rng.randn(n_tok, dim).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vectors.append(v)
    return vectors, ids, dim


@pytest.fixture
def tmp_index_dir():
    d = tempfile.mkdtemp(prefix="shard_engine_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------
# ShardStore round-trip
# ------------------------------------------------------------------

class TestShardStore:
    def test_build_and_load(self, tmp_index_dir):
        vectors, ids, dim = _make_corpus(n_docs=100)
        all_vecs = np.concatenate(vectors, axis=0).astype(np.float16)
        offsets = []
        off = 0
        for v in vectors:
            offsets.append((off, off + v.shape[0]))
            off += v.shape[0]

        assignments = np.array([i % 8 for i in range(100)], dtype=np.int32)
        store = ShardStore(tmp_index_dir)
        store.build(
            all_vectors=all_vecs,
            doc_offsets=offsets,
            doc_ids=ids,
            shard_assignments=assignments,
            n_shards=8,
            dim=dim,
            compression=Compression.FP16,
            uniform_shard_tokens=True,
        )

        assert store.manifest is not None
        assert store.manifest.num_docs == 100
        assert store.manifest.num_shards == 8

        store2 = ShardStore(tmp_index_dir)
        assert store2.manifest is not None
        assert store2.manifest.num_docs == 100
        assert len(store2.all_doc_ids()) == 100

    def test_doc_selective_fetch(self, tmp_index_dir):
        vectors, ids, dim = _make_corpus(n_docs=50)
        all_vecs = np.concatenate(vectors, axis=0).astype(np.float16)
        offsets = []
        off = 0
        for v in vectors:
            offsets.append((off, off + v.shape[0]))
            off += v.shape[0]

        assignments = np.array([i % 4 for i in range(50)], dtype=np.int32)
        store = ShardStore(tmp_index_dir)
        store.build(
            all_vectors=all_vecs,
            doc_offsets=offsets,
            doc_ids=ids,
            shard_assignments=assignments,
            n_shards=4,
            dim=dim,
        )

        shard_0_ids = [i for i in range(50) if assignments[i] == 0]
        flat_emb, doc_offsets_out, fetched_ids = store.load_docs_from_shard(0, shard_0_ids)
        assert len(fetched_ids) > 0
        assert flat_emb.shape[1] == dim


# ------------------------------------------------------------------
# Scorer correctness
# ------------------------------------------------------------------

class TestScorer:
    def test_maxsim_kernel_loads(self):
        fn = _get_maxsim()
        assert fn is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_brute_force_maxsim(self):
        vectors, ids, dim = _make_corpus(n_docs=30, dim=64)
        query = np.random.randn(32, 64).astype(np.float32)
        result_ids, result_scores = brute_force_maxsim(
            query, vectors, ids, dim, k=5, device="cuda",
        )
        assert len(result_ids) == 5
        assert len(result_scores) == 5
        assert result_scores == sorted(result_scores, reverse=True)


# ------------------------------------------------------------------
# ShardSegmentManager integration
# ------------------------------------------------------------------

class TestShardSegmentManager:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_build_and_search(self, tmp_index_dir):
        vectors, ids, dim = _make_corpus(n_docs=200, dim=128)
        # The new ShardEngineConfig default is rroq158 K=8192, which needs
        # >= 8192 tokens to train the codebook. This test uses 200 docs of
        # ~35 tokens (~7000 tokens) so we explicitly fall back to FP16 — the
        # codec under test here is the engine plumbing, not the codec.
        config = ShardEngineConfig(
            n_shards=8,
            dim=dim,
            compression=Compression.FP16,
            lemur_epochs=3,
            k_candidates=200,
        )
        mgr = ShardSegmentManager(tmp_index_dir, config=config, device="cuda")
        mgr.build(vectors, ids)

        assert mgr.total_vectors() == 200

        query = np.random.RandomState(99).randn(32, dim).astype(np.float32)
        results = mgr.search_multivector(query, k=10)

        assert len(results) == 10
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert isinstance(score, float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_recall_at_10(self, tmp_index_dir):
        """Build 500 docs, check R@10 > 0.5 vs brute-force ground truth."""
        vectors, ids, dim = _make_corpus(n_docs=500, dim=128)
        config = ShardEngineConfig(
            n_shards=16,
            dim=dim,
            # 500 docs × ~35 tokens (~17.5k tokens) is borderline for K=8192;
            # use fp16 here so the recall assertion measures the engine, not
            # the codec's small-corpus floor.
            compression=Compression.FP16,
            lemur_epochs=5,
            k_candidates=500,
        )
        mgr = ShardSegmentManager(tmp_index_dir, config=config, device="cuda")
        mgr.build(vectors, ids)

        n_queries = 20
        rng = np.random.RandomState(123)
        recalls = []
        for _ in range(n_queries):
            q = rng.randn(32, dim).astype(np.float32)

            gt_ids, _ = brute_force_maxsim(q, vectors, ids, dim, k=10, device="cuda")
            gt_set = set(gt_ids[:10])

            results = mgr.search_multivector(q, k=10)
            result_set = set(r[0] for r in results)
            recall = len(gt_set & result_set) / max(len(gt_set), 1)
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        assert mean_recall > 0.5, f"Mean R@10 = {mean_recall:.3f}, expected > 0.5"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_save_and_reload(self, tmp_index_dir):
        vectors, ids, dim = _make_corpus(n_docs=100, dim=64)
        config = ShardEngineConfig(
            n_shards=4, dim=dim,
            compression=Compression.FP16,  # tiny corpus; can't train K=8192
            lemur_epochs=2, k_candidates=100,
        )
        mgr = ShardSegmentManager(tmp_index_dir, config=config, device="cuda")
        mgr.build(vectors, ids)

        query = np.random.RandomState(7).randn(16, dim).astype(np.float32)
        results_before = mgr.search_multivector(query, k=5)
        mgr.close()

        mgr2 = ShardSegmentManager(tmp_index_dir, config=config, device="cuda")
        assert mgr2._is_built
        stats = mgr2.get_statistics()
        assert stats["num_docs"] == 100
        assert stats["engine"] == "shard"
        mgr2.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_crud_on_unbuilt_index(self, tmp_index_dir):
        """add_multidense on unbuilt index delegates to build(); empty corpus raises."""
        config = ShardEngineConfig(dim=64, compression=Compression.FP16)
        mgr = ShardSegmentManager(tmp_index_dir, config=config, device="cuda")
        with pytest.raises(ValueError, match="empty corpus"):
            mgr.add_multidense([], [])
        mgr.delete([1])

    def test_context_manager(self, tmp_index_dir):
        config = ShardEngineConfig(dim=64, compression=Compression.FP16)
        with ShardSegmentManager(tmp_index_dir, config=config, device="cpu") as mgr:
            assert mgr is not None
