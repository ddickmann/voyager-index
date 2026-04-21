"""Tests for shard engine public API via Index and IndexBuilder (Chunk 2)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from colsearch.index import Index, IndexBuilder


def _make_corpus(n_docs: int = 100, dim: int = 128, min_tok: int = 10, max_tok: int = 50):
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
    d = tempfile.mkdtemp(prefix="shard_api_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestIndexShardEngine:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_index_constructor(self, tmp_dir):
        idx = Index(tmp_dir, dim=128, engine="shard")
        assert idx.engine == "shard"
        assert repr(idx).endswith("open)")
        idx.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_add_and_search(self, tmp_dir):
        dim = 64
        idx = Index(tmp_dir, dim=dim, engine="shard", k_candidates=100, n_shards=4, lemur_epochs=2)
        vectors = _make_corpus(n_docs=100, dim=dim, min_tok=8, max_tok=32)
        ids = list(range(100))
        idx.add(vectors, ids=ids)

        query = np.random.RandomState(99).randn(16, dim).astype(np.float32)
        results = idx.search(query, k=5)
        assert len(results) == 5
        for r in results:
            assert hasattr(r, "doc_id")
            assert hasattr(r, "score")
            assert isinstance(r.score, float)
        idx.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_stats(self, tmp_dir):
        dim = 64
        idx = Index(tmp_dir, dim=dim, engine="shard", n_shards=4, lemur_epochs=2, k_candidates=50)
        vectors = _make_corpus(n_docs=50, dim=dim, min_tok=8, max_tok=24)
        idx.add(vectors, ids=list(range(50)))
        stats = idx.stats()
        assert stats.engine == "shard"
        assert stats.total_documents == 50
        idx.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reopen_index(self, tmp_dir):
        dim = 64
        idx = Index(tmp_dir, dim=dim, engine="shard", n_shards=4, lemur_epochs=2, k_candidates=100)
        vectors = _make_corpus(n_docs=80, dim=dim, min_tok=8, max_tok=24)
        idx.add(vectors, ids=list(range(80)))
        idx.close()

        idx2 = Index(tmp_dir, dim=dim, engine="shard", k_candidates=100)
        stats = idx2.stats()
        assert stats.total_documents == 80

        query = np.random.RandomState(7).randn(16, dim).astype(np.float32)
        results = idx2.search(query, k=3)
        assert len(results) == 3
        idx2.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_context_manager(self, tmp_dir):
        dim = 64
        with Index(tmp_dir, dim=dim, engine="shard", n_shards=4, lemur_epochs=2, k_candidates=50) as idx:
            vectors = _make_corpus(n_docs=30, dim=dim, min_tok=8, max_tok=24)
            idx.add(vectors, ids=list(range(30)))
            results = idx.search(
                np.random.RandomState(1).randn(8, dim).astype(np.float32), k=5,
            )
            assert len(results) > 0


class TestIndexBuilder:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_builder_with_shard(self, tmp_dir):
        idx = (
            IndexBuilder(tmp_dir, dim=64)
            .with_shard(n_shards=4, lemur_epochs=2, k_candidates=50)
            .build()
        )
        assert idx.engine == "shard"
        vectors = _make_corpus(n_docs=40, dim=64, min_tok=8, max_tok=20)
        idx.add(vectors, ids=list(range(40)))
        stats = idx.stats()
        assert stats.total_documents == 40
        idx.close()

    def test_unknown_engine_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="Unknown engine"):
            Index(tmp_dir, dim=64, engine="nonexistent")


class TestIndexPublicAPIShard:
    """Index-level public API tests for shard engine (FIX 36)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_delete_excludes_from_search(self, tmp_dir):
        dim = 64
        idx = Index(tmp_dir, dim=dim, engine="shard", n_shards=4, lemur_epochs=2, k_candidates=100)
        vectors = _make_corpus(n_docs=50, dim=dim, min_tok=8, max_tok=24)
        ids = list(range(50))
        idx.add(vectors, ids=ids)
        idx.delete([0, 1, 2])
        results = idx.search(vectors[0], k=50)
        returned_ids = {r.doc_id for r in results}
        assert 0 not in returned_ids
        assert 1 not in returned_ids
        assert 2 not in returned_ids
        idx.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_upsert_searchable(self, tmp_dir):
        dim = 64
        idx = Index(tmp_dir, dim=dim, engine="shard", n_shards=4, lemur_epochs=2, k_candidates=100)
        vectors = _make_corpus(n_docs=30, dim=dim, min_tok=8, max_tok=24)
        idx.add(vectors, ids=list(range(30)))
        new_vec = np.random.RandomState(999).randn(12, dim).astype(np.float32)
        idx.upsert([new_vec], ids=[0], payloads=[{"updated": True}])
        results = idx.search(new_vec, k=5)
        found_ids = [r.doc_id for r in results]
        assert 0 in found_ids
        idx.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scroll_paginated(self, tmp_dir):
        dim = 64
        idx = Index(tmp_dir, dim=dim, engine="shard", n_shards=4, lemur_epochs=2, k_candidates=100)
        vectors = _make_corpus(n_docs=50, dim=dim, min_tok=8, max_tok=24)
        idx.add(vectors, ids=list(range(50)))
        all_ids = []
        offset = 0
        while True:
            page = idx.scroll(limit=20, offset=offset)
            all_ids.extend(r.doc_id for r in page.results)
            if page.next_offset is None:
                break
            offset = page.next_offset
        assert len(all_ids) == 50
        assert len(set(all_ids)) == 50
        idx.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_search_with_filters(self, tmp_dir):
        dim = 64
        idx = Index(tmp_dir, dim=dim, engine="shard", n_shards=4, lemur_epochs=2, k_candidates=100)
        vectors = _make_corpus(n_docs=40, dim=dim, min_tok=8, max_tok=24)
        payloads = [{"color": "red" if i < 20 else "blue"} for i in range(40)]
        idx.add(vectors, ids=list(range(40)), payloads=payloads)
        results = idx.search(vectors[0], k=40, filters={"color": "red"})
        for r in results:
            assert r.payload["color"] == "red"
        idx.close()
