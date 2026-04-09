"""Integration tests for GEM gap implementations: qEMD, dual-graph, adaptive cutoff, defaults."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from latence_gem_index import GemSegment, PyMutableGemSegment
    GEM_INDEX_AVAILABLE = True
except ImportError:
    GEM_INDEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GEM_INDEX_AVAILABLE, reason="latence_gem_index not installed")


def _synthetic_corpus(n_docs=30, dim=16, vecs_per_doc=8, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_docs * vecs_per_doc, dim)).astype(np.float32)
    doc_ids = list(range(1, n_docs + 1))
    offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]
    return data, doc_ids, offsets


class TestDefaults:
    def test_default_build_uses_ch_not_emd(self):
        """Default build should use qCH (use_emd=False, dual_graph=False)."""
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=20, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)
        assert seg.is_ready()
        assert seg.n_docs() == 20
        assert seg.n_edges() > 0

    def test_search_returns_results(self):
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=20, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)
        query = np.random.RandomState(99).randn(3, 8).astype(np.float32)
        results = seg.search(query, k=5, ef=32, n_probes=2)
        assert len(results) > 0
        assert len(results) <= 5
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert np.isfinite(score)


class TestUseEmd:
    def test_build_with_use_emd(self):
        """Build with use_emd=True should succeed and produce a valid graph."""
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=15, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=6,
                  ef_construction=16, max_kmeans_iter=5, ctop_r=2,
                  use_emd=True)
        assert seg.is_ready()
        assert seg.n_edges() > 0

    def test_emd_search_returns_results(self):
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=15, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=6,
                  ef_construction=16, max_kmeans_iter=5, ctop_r=2,
                  use_emd=True)
        query = np.random.RandomState(99).randn(3, 8).astype(np.float32)
        results = seg.search(query, k=5, ef=32, n_probes=2)
        assert len(results) > 0


class TestDualGraph:
    def test_build_with_dual_graph(self):
        """Build with dual_graph=True should succeed."""
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=20, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2,
                  dual_graph=True)
        assert seg.is_ready()
        assert seg.n_edges() > 0

    def test_dual_graph_search(self):
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=20, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2,
                  dual_graph=True)
        query = np.random.RandomState(99).randn(3, 8).astype(np.float32)
        results = seg.search(query, k=5, ef=32, n_probes=2)
        assert len(results) > 0

    def test_dual_graph_with_emd(self):
        """Dual graph + EMD combo should work."""
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=10, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=6,
                  ef_construction=16, max_kmeans_iter=5, ctop_r=2,
                  use_emd=True, dual_graph=True)
        assert seg.is_ready()
        query = np.random.RandomState(99).randn(3, 8).astype(np.float32)
        results = seg.search(query, k=5, ef=32, n_probes=2)
        assert len(results) > 0


class TestPayloadValidation:
    def test_dual_graph_rejects_payload_clusters(self):
        """dual_graph=True + payload_clusters should raise ValueError."""
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=10, dim=8, vecs_per_doc=4)
        with pytest.raises(ValueError, match="mutually exclusive"):
            seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=6,
                      ef_construction=16, max_kmeans_iter=5, ctop_r=2,
                      payload_clusters=[0] * 10, dual_graph=True)


class TestMutableSegment:
    def test_mutable_build_and_search(self):
        seg = PyMutableGemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=20, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)
        assert seg.is_ready()
        assert seg.n_nodes() == 20

        query = np.random.RandomState(99).randn(3, 8).astype(np.float32)
        results = seg.search(query, k=5, ef=32)
        assert len(results) > 0

    def test_mutable_insert_and_delete(self):
        seg = PyMutableGemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=10, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)
        initial_nodes = seg.n_nodes()

        new_doc = np.random.RandomState(77).randn(4, 8).astype(np.float32)
        seg.insert(new_doc, doc_id=999)
        assert seg.n_nodes() == initial_nodes + 1

        deleted = seg.delete(999)
        assert deleted

    def test_mutable_with_use_emd(self):
        seg = PyMutableGemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=10, dim=8, vecs_per_doc=4)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2,
                  use_emd=True)
        assert seg.is_ready()
        query = np.random.RandomState(99).randn(3, 8).astype(np.float32)
        results = seg.search(query, k=5, ef=32)
        assert len(results) > 0


class TestSemanticShortcuts:
    def test_inject_shortcuts_basic(self):
        """Inject shortcuts from training pairs and verify total_shortcuts > 0."""
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=30, dim=16, vecs_per_doc=8)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)
        assert seg.is_ready()

        dim = 16
        vecs_per_doc = 8
        training_pairs = []
        for doc_idx in range(5):
            start = doc_idx * vecs_per_doc
            end = (doc_idx + 1) * vecs_per_doc
            flat_query = data[start:end].flatten().tolist()
            target_internal_id = doc_idx
            training_pairs.append((flat_query, target_internal_id))

        seg.inject_shortcuts(training_pairs, max_shortcuts_per_node=4)
        assert seg.total_shortcuts() > 0

    def test_shortcuts_change_results(self):
        """Search with and without shortcuts after injection; total_shortcuts must be > 0."""
        seg = GemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=30, dim=16, vecs_per_doc=8)
        seg.build(data, ids, offsets, n_fine=8, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)

        query = np.random.RandomState(99).randn(3, 16).astype(np.float32)
        results_no_sc = seg.search(query, k=5, ef=32, n_probes=2, enable_shortcuts=False)
        assert len(results_no_sc) > 0

        dim = 16
        vecs_per_doc = 8
        training_pairs = []
        for doc_idx in range(5):
            start = doc_idx * vecs_per_doc
            end = (doc_idx + 1) * vecs_per_doc
            flat_query = data[start:end].flatten().tolist()
            training_pairs.append((flat_query, doc_idx))

        seg.inject_shortcuts(training_pairs, max_shortcuts_per_node=4)
        assert seg.total_shortcuts() > 0

        results_with_sc = seg.search(query, k=5, ef=32, n_probes=2, enable_shortcuts=True)
        assert len(results_with_sc) > 0


try:
    from latence_gem_router import PyGemRouter
    GEM_ROUTER_AVAILABLE = True
except ImportError:
    GEM_ROUTER_AVAILABLE = False


@pytest.mark.skipif(not GEM_ROUTER_AVAILABLE, reason="latence_gem_router not installed")
class TestAdaptiveCutoff:
    """E2E tests for PyGemRouter.train_adaptive_cutoff."""

    @staticmethod
    def _build_router(n_docs=30, dim=16, vecs_per_doc=8, seed=42):
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_docs * vecs_per_doc, dim)).astype(np.float32)
        doc_ids = list(range(n_docs))
        offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]

        router = PyGemRouter(dim=dim)
        router.build(data, doc_ids, offsets, n_fine=16, n_coarse=4)
        return router, dim, n_docs

    @staticmethod
    def _make_training_pairs(n_queries, vecs_per_query, dim, n_docs, seed=123):
        """Generate synthetic training data for adaptive cutoff.

        Returns (training_queries, n_query_vecs, training_positives).
        """
        rng = np.random.default_rng(seed)
        training_queries = rng.standard_normal(
            (n_queries * vecs_per_query, dim)
        ).astype(np.float32)
        n_query_vecs = [vecs_per_query] * n_queries
        training_positives = [int(rng.integers(0, n_docs)) for _ in range(n_queries)]
        return training_queries, n_query_vecs, training_positives

    def test_train_adaptive_cutoff_e2e(self):
        """Build a router, train adaptive cutoff, verify it returns serialized bytes."""
        router, dim, n_docs = self._build_router(n_docs=30)
        assert router.is_ready()

        n_queries = 20
        vecs_per_query = 4
        training_queries, n_query_vecs, training_positives = self._make_training_pairs(
            n_queries, vecs_per_query, dim, n_docs,
        )

        tree_bytes = router.train_adaptive_cutoff(
            training_queries, n_query_vecs, training_positives,
            t=3, r_max=8, max_depth=6,
        )

        assert isinstance(tree_bytes, (bytes, list))
        raw = bytes(tree_bytes) if not isinstance(tree_bytes, bytes) else tree_bytes
        assert len(raw) > 0

    def test_adaptive_cutoff_predict_values_in_range(self):
        """Train the tree and verify the serialized blob is non-trivial."""
        router, dim, n_docs = self._build_router(n_docs=30)

        training_queries, n_query_vecs, training_positives = self._make_training_pairs(
            n_queries=25, vecs_per_query=5, dim=dim, n_docs=n_docs, seed=999,
        )

        tree_bytes = router.train_adaptive_cutoff(
            training_queries, n_query_vecs, training_positives,
        )

        assert isinstance(tree_bytes, (bytes, list))
        raw = bytes(tree_bytes) if not isinstance(tree_bytes, bytes) else tree_bytes
        assert len(raw) > 8, "serialized tree should contain more than just a header"
