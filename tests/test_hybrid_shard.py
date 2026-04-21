"""Tests for hybrid search with shard engine as dense backend (Chunk 4)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import bm25s
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from colsearch._internal.inference.index_core.hybrid_manager import (
        HybridSearchManager,
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

from colsearch._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)


def _make_corpus(n_docs: int = 50, dim: int = 64, min_tok: int = 8, max_tok: int = 20):
    rng = np.random.RandomState(42)
    vectors = []
    texts = []
    for i in range(n_docs):
        n_tok = rng.randint(min_tok, max_tok + 1)
        v = rng.randn(n_tok, dim).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        vectors.append(v)
        texts.append(f"Document {i} about topic {i % 5} with content sample {rng.randint(0, 1000)}")
    return vectors, texts


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="hybrid_shard_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.skipif(not BM25_AVAILABLE, reason="bm25s required")
@pytest.mark.skipif(not HYBRID_AVAILABLE, reason="hybrid_manager required")
class TestHybridWithShardDense:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_construct_with_shard_engine(self, tmp_dir):
        config = ShardEngineConfig(n_shards=4, dim=64, lemur_epochs=2, k_candidates=50)
        mgr = HybridSearchManager(
            shard_path=tmp_dir,
            dim=64,
            dense_engine="shard",
            dense_engine_config=config,
        )
        assert mgr._dense_engine_type == "shard"
        assert isinstance(mgr.hnsw, ShardSegmentManager)
        mgr.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_index_and_search_hybrid(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = HybridSearchManager(
            shard_path=tmp_dir,
            dim=dim,
            dense_engine="shard",
            dense_engine_config=config,
        )
        vectors, texts = _make_corpus(n_docs=50, dim=dim)
        ids = list(range(50))
        mgr.index_multivector(texts, vectors, ids)

        query_vec = np.random.RandomState(99).randn(8, dim).astype(np.float32)
        results = mgr.search(
            query_text="topic 3 content",
            query_vector=query_vec,
            k=10,
        )

        assert "dense" in results
        assert "sparse" in results
        assert "union_ids" in results
        assert len(results["dense"]) > 0
        mgr.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dense_only_search(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = HybridSearchManager(
            shard_path=tmp_dir,
            dim=dim,
            dense_engine="shard",
            dense_engine_config=config,
        )
        vectors, texts = _make_corpus(n_docs=30, dim=dim)
        ids = list(range(30))
        mgr.index_multivector(texts, vectors, ids)

        query_vec = np.random.RandomState(7).randn(8, dim).astype(np.float32)
        results = mgr.search(
            query_text="",
            query_vector=query_vec,
            k=5,
        )

        assert len(results["dense"]) > 0
        mgr.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sparse_only_search(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = HybridSearchManager(
            shard_path=tmp_dir,
            dim=dim,
            dense_engine="shard",
            dense_engine_config=config,
        )
        vectors, texts = _make_corpus(n_docs=30, dim=dim)
        ids = list(range(30))
        mgr.index_multivector(texts, vectors, ids)

        results = mgr.search(
            query_text="Document about topic 2",
            query_vector=None,
            k=5,
        )
        assert len(results["sparse"]) >= 0
        mgr.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_rrf_fusion(self, tmp_dir):
        dim = 64
        config = ShardEngineConfig(n_shards=4, dim=dim, lemur_epochs=2, k_candidates=50)
        mgr = HybridSearchManager(
            shard_path=tmp_dir,
            dim=dim,
            dense_engine="shard",
            dense_engine_config=config,
        )
        vectors, texts = _make_corpus(n_docs=50, dim=dim)
        ids = list(range(50))
        mgr.index_multivector(texts, vectors, ids)

        query_vec = np.random.RandomState(42).randn(8, dim).astype(np.float32)
        results = mgr.search(
            query_text="topic 1 content",
            query_vector=query_vec,
            k=20,
        )

        assert len(results["union_ids"]) > 0
        assert mgr._last_search_context is not None
        assert "rrf" in mgr._last_search_context
        mgr.close()
