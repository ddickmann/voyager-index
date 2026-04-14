"""Tests for the Rust GEM router and GemScreeningIndex wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from latence_gem_router import PyGemRouter
    GEM_AVAILABLE = True
except ImportError:
    GEM_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_router not installed")


def _synthetic_corpus(n_docs=30, dim=16, vecs_per_doc=8, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_docs * vecs_per_doc, dim)).astype(np.float32)
    doc_ids = list(range(n_docs))
    offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]
    return data, doc_ids, offsets


class TestPyGemRouter:
    def test_build_and_route(self):
        dim = 16
        data, doc_ids, offsets = _synthetic_corpus(n_docs=30, dim=dim)
        router = PyGemRouter(dim=dim)
        router.build(data, doc_ids, offsets, n_fine=16, n_coarse=4)

        assert router.is_ready()
        assert router.n_docs() == 30
        assert router.n_fine() == 16
        assert router.n_coarse() == 4

        query = np.random.randn(4, dim).astype(np.float32)
        results = router.route_query(query, n_probes=4, max_candidates=10)
        assert len(results) > 0
        assert len(results) <= 10
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert score > 0

    def test_cluster_entries(self):
        dim = 16
        data, doc_ids, offsets = _synthetic_corpus(n_docs=20, dim=dim)
        router = PyGemRouter(dim=dim)
        router.build(data, doc_ids, offsets, n_fine=8, n_coarse=4)

        query = np.random.randn(3, dim).astype(np.float32)
        entries = router.get_cluster_entries(query, n_probes=4)
        assert isinstance(entries, list)

    def test_query_profile(self):
        dim = 16
        data, doc_ids, offsets = _synthetic_corpus(dim=dim)
        router = PyGemRouter(dim=dim)
        router.build(data, doc_ids, offsets, n_fine=16, n_coarse=4)

        query = np.random.randn(5, dim).astype(np.float32)
        cids, ctop = router.compute_query_profile(query, n_probes=4)
        assert len(cids) == 5
        assert len(ctop) > 0
        for c in cids:
            assert 0 <= c < 16

    def test_add_documents(self):
        dim = 16
        data, doc_ids, offsets = _synthetic_corpus(n_docs=20, dim=dim)
        router = PyGemRouter(dim=dim)
        router.build(data, doc_ids, offsets, n_fine=8, n_coarse=4)
        assert router.n_docs() == 20

        new_data, _, new_offsets = _synthetic_corpus(n_docs=10, dim=dim, seed=99)
        new_ids = list(range(20, 30))
        router.add_documents(new_data, new_ids, new_offsets)
        assert router.n_docs() == 30

    def test_save_load_roundtrip(self):
        dim = 16
        data, doc_ids, offsets = _synthetic_corpus(dim=dim)
        router = PyGemRouter(dim=dim)
        router.build(data, doc_ids, offsets, n_fine=8, n_coarse=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "router.gemr")
            router.save(path)

            router2 = PyGemRouter(dim=dim)
            router2.load(path)
            assert router2.is_ready()
            assert router2.n_docs() == router.n_docs()
            assert router2.n_fine() == router.n_fine()
            assert router2.n_coarse() == router.n_coarse()

            query = np.random.randn(3, dim).astype(np.float32)
            r1 = router.route_query(query, n_probes=4, max_candidates=10)
            r2 = router2.route_query(query, n_probes=4, max_candidates=10)
            assert len(r1) == len(r2)

    def test_dimension_mismatch_raises(self):
        router = PyGemRouter(dim=16)
        data = np.random.randn(100, 32).astype(np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            router.build(data, [0, 1], [(0, 50), (50, 100)], n_fine=8, n_coarse=4)

    def test_route_before_build_raises(self):
        router = PyGemRouter(dim=16)
        query = np.random.randn(3, 16).astype(np.float32)
        with pytest.raises(ValueError, match="not built"):
            router.route_query(query)

    def test_cluster_overlaps(self):
        dim = 16
        data, doc_ids, offsets = _synthetic_corpus(dim=dim)
        router = PyGemRouter(dim=dim)
        router.build(data, doc_ids, offsets, n_fine=16, n_coarse=4)

        query = np.random.randn(5, dim).astype(np.float32)
        overlaps = router.query_cluster_overlaps(query, n_probes=4)
        assert len(overlaps) == 30
        for doc_id, overlap in overlaps:
            assert isinstance(doc_id, int)
            assert overlap >= 0


class TestGemScreeningIndex:
    def test_gem_screening_build_and_search(self):
        from voyager_index._internal.inference.index_core.gem_screening import (
            GemScreeningIndex,
            GEM_ROUTER_AVAILABLE,
        )
        if not GEM_ROUTER_AVAILABLE:
            pytest.skip("gem_router not available")

        dim = 16
        n_docs = 25
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = GemScreeningIndex(Path(tmpdir) / "gem", dim=dim)
            emb = np.random.randn(n_docs, 8, dim).astype(np.float32)
            idx.rebuild(doc_ids=list(range(n_docs)), embeddings=emb)

            assert idx.gem_enabled
            assert idx._doc_count == n_docs

            query = np.random.randn(4, dim).astype(np.float32)
            results = idx.search(query, top_k=5)
            assert len(results) > 0
            assert idx.last_search_profile["gem_active"]

    def test_gem_screening_persistence(self):
        from voyager_index._internal.inference.index_core.gem_screening import (
            GemScreeningIndex,
            GEM_ROUTER_AVAILABLE,
        )
        if not GEM_ROUTER_AVAILABLE:
            pytest.skip("gem_router not available")

        dim = 16
        n_docs = 20
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "gem"
            idx = GemScreeningIndex(root, dim=dim)
            emb = np.random.randn(n_docs, 6, dim).astype(np.float32)
            idx.rebuild(doc_ids=list(range(n_docs)), embeddings=emb)

            idx2 = GemScreeningIndex(root, dim=dim, load_if_exists=True)
            assert idx2.gem_enabled
            assert idx2._doc_count == n_docs
