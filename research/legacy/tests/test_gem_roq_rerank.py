"""Integration tests for ROQ 4-bit reranking wired into the GEM native path."""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest

try:
    from latence_gem_index import GemSegment
    GEM_AVAILABLE = True
except ImportError:
    GEM_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from voyager_index._internal.kernels.roq import ROQ_TRITON_AVAILABLE
except ImportError:
    ROQ_TRITON_AVAILABLE = False

try:
    from voyager_index._internal.inference.index_core.gem_manager import GemNativeSegmentManager
    MANAGER_AVAILABLE = True
except ImportError:
    MANAGER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (GEM_AVAILABLE and TORCH_AVAILABLE and MANAGER_AVAILABLE),
    reason="GEM index, torch, or gem_manager not available",
)

DIM = 32
N_DOCS = 60
TOKENS_PER_DOC = 8
SEED = 42


def _make_data(n_docs=N_DOCS, dim=DIM, tpd=TOKENS_PER_DOC, seed=SEED):
    rng = np.random.default_rng(seed)
    vectors = []
    ids = list(range(1, n_docs + 1))
    for _ in range(n_docs):
        vecs = rng.standard_normal((tpd, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vectors.append(vecs)
    return vectors, ids


def _make_query(dim=DIM, n_tokens=4, seed=99):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_tokens, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    return q


class TestRoqRerankBasic:
    """Test that the ROQ rerank path produces valid results."""

    @pytest.mark.skipif(not ROQ_TRITON_AVAILABLE, reason="Triton ROQ kernels not available")
    def test_build_search_with_roq(self):
        tmp = tempfile.mkdtemp()
        try:
            mgr = GemNativeSegmentManager(
                tmp, DIM,
                n_fine=16, n_coarse=4, max_degree=8,
                gem_ef_construction=32, max_kmeans_iter=5,
                ctop_r=2, seed_batch_size=N_DOCS,
                enable_wal=True, roq_rerank=True,
            )
            vectors, ids = _make_data()
            mgr.add_multidense(vectors, ids)
            mgr.seal_active_segment()

            assert len(mgr._doc_roq) > 0, "ROQ data should be populated"
            assert len(mgr._doc_vectors) == 0, "FP32 vectors should not be stored in ROQ mode"

            q = _make_query()
            results = mgr.search_multivector(q, k=10, ef=32, n_probes=2)

            assert len(results) > 0, "Search should return results"
            assert len(results) <= 10
            for doc_id, score in results:
                assert isinstance(doc_id, int)
                assert np.isfinite(score), f"Score for doc {doc_id} is not finite: {score}"

            mgr.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.skipif(not ROQ_TRITON_AVAILABLE, reason="Triton ROQ kernels not available")
    def test_save_load_roq_rerank(self):
        """ROQ reranking should work after save/load cycle."""
        tmp = tempfile.mkdtemp()
        try:
            mgr = GemNativeSegmentManager(
                tmp, DIM,
                n_fine=16, n_coarse=4, max_degree=8,
                gem_ef_construction=32, max_kmeans_iter=5,
                ctop_r=2, seed_batch_size=N_DOCS,
                enable_wal=True, roq_rerank=True,
            )
            vectors, ids = _make_data()
            mgr.add_multidense(vectors, ids)
            mgr.seal_active_segment()

            q = _make_query()
            results_before = mgr.search_multivector(q, k=10, ef=32, n_probes=2)
            mgr.close()

            mgr2 = GemNativeSegmentManager(
                tmp, DIM,
                n_fine=16, n_coarse=4, max_degree=8,
                gem_ef_construction=32, max_kmeans_iter=5,
                ctop_r=2, seed_batch_size=N_DOCS,
                enable_wal=True, roq_rerank=True,
            )

            assert len(mgr2._doc_roq) > 0, "ROQ data should be restored on load"

            results_after = mgr2.search_multivector(q, k=10, ef=32, n_probes=2)
            assert len(results_after) > 0

            ids_before = {r[0] for r in results_before}
            ids_after = {r[0] for r in results_after}
            overlap = len(ids_before & ids_after) / max(len(ids_before), 1)
            assert overlap > 0.5, f"Results should be similar after reload, overlap={overlap:.2f}"

            mgr2.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestRoqVsFp32Recall:
    """Compare ROQ 4-bit recall against FP32 MaxSim recall."""

    @pytest.mark.skipif(not ROQ_TRITON_AVAILABLE, reason="Triton ROQ kernels not available")
    def test_recall_close_to_fp32(self):
        """ROQ rerank recall should be within a reasonable margin of FP32 rerank."""
        vectors, ids = _make_data(n_docs=50, dim=DIM, tpd=TOKENS_PER_DOC)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tmp_fp32 = tempfile.mkdtemp()
        tmp_roq = tempfile.mkdtemp()
        try:
            mgr_fp32 = GemNativeSegmentManager(
                tmp_fp32, DIM,
                n_fine=16, n_coarse=4, max_degree=8,
                gem_ef_construction=32, max_kmeans_iter=5,
                ctop_r=2, seed_batch_size=50,
                enable_wal=True, rerank_device=device,
            )
            mgr_fp32.add_multidense(vectors, ids)
            mgr_fp32.seal_active_segment()

            mgr_roq = GemNativeSegmentManager(
                tmp_roq, DIM,
                n_fine=16, n_coarse=4, max_degree=8,
                gem_ef_construction=32, max_kmeans_iter=5,
                ctop_r=2, seed_batch_size=50,
                enable_wal=True, roq_rerank=True,
            )
            mgr_roq.add_multidense(vectors, ids)
            mgr_roq.seal_active_segment()

            n_queries = 10
            k = 10
            overlaps = []
            for qi in range(n_queries):
                q = _make_query(seed=200 + qi)
                res_fp32 = mgr_fp32.search_multivector(q, k=k, ef=32, n_probes=2)
                res_roq = mgr_roq.search_multivector(q, k=k, ef=32, n_probes=2)

                ids_fp32 = {r[0] for r in res_fp32}
                ids_roq = {r[0] for r in res_roq}
                if ids_fp32:
                    overlaps.append(len(ids_fp32 & ids_roq) / len(ids_fp32))

            if overlaps:
                mean_overlap = np.mean(overlaps)
                assert mean_overlap > 0.3, (
                    f"ROQ recall overlap with FP32 too low: {mean_overlap:.2f}"
                )

            mgr_fp32.close()
            mgr_roq.close()
        finally:
            shutil.rmtree(tmp_fp32, ignore_errors=True)
            shutil.rmtree(tmp_roq, ignore_errors=True)


class TestRoqMemoryFootprint:
    """Verify ROQ uses less memory than FP32 for the same documents."""

    @pytest.mark.skipif(not ROQ_TRITON_AVAILABLE, reason="Triton ROQ kernels not available")
    def test_roq_smaller_than_fp32(self):
        vectors, ids = _make_data(n_docs=30, dim=DIM, tpd=TOKENS_PER_DOC)

        tmp_fp32 = tempfile.mkdtemp()
        tmp_roq = tempfile.mkdtemp()
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            mgr_fp32 = GemNativeSegmentManager(
                tmp_fp32, DIM,
                n_fine=16, n_coarse=4, max_degree=8,
                gem_ef_construction=32, max_kmeans_iter=5,
                ctop_r=2, seed_batch_size=30,
                enable_wal=True, rerank_device=device,
            )
            mgr_fp32.add_multidense(vectors, ids)

            mgr_roq = GemNativeSegmentManager(
                tmp_roq, DIM,
                n_fine=16, n_coarse=4, max_degree=8,
                gem_ef_construction=32, max_kmeans_iter=5,
                ctop_r=2, seed_batch_size=30,
                enable_wal=True, roq_rerank=True,
            )
            mgr_roq.add_multidense(vectors, ids)

            fp32_bytes = sum(v.nbytes for v in mgr_fp32._doc_vectors.values())
            roq_bytes = sum(c.nbytes + m.nbytes for c, m in mgr_roq._doc_roq.values())

            assert roq_bytes < fp32_bytes, (
                f"ROQ ({roq_bytes} bytes) should be smaller than FP32 ({fp32_bytes} bytes)"
            )
            ratio = fp32_bytes / max(roq_bytes, 1)
            assert ratio > 2.0, f"Expected at least 2x compression, got {ratio:.1f}x"

            mgr_fp32.close()
            mgr_roq.close()
        finally:
            shutil.rmtree(tmp_fp32, ignore_errors=True)
            shutil.rmtree(tmp_roq, ignore_errors=True)


class TestFallbackWithoutTriton:
    """When Triton is not available, roq_rerank=True should degrade gracefully."""

    def test_roq_false_when_unavailable(self):
        """If ROQ is explicitly requested but unavailable, it falls back to proxy."""
        tmp = tempfile.mkdtemp()
        try:
            mgr = GemNativeSegmentManager(
                tmp, DIM,
                n_fine=16, n_coarse=4, max_degree=8,
                gem_ef_construction=32, max_kmeans_iter=5,
                ctop_r=2, seed_batch_size=N_DOCS,
                enable_wal=True,
                roq_rerank=not ROQ_TRITON_AVAILABLE,
            )
            vectors, ids = _make_data()
            mgr.add_multidense(vectors, ids)
            mgr.seal_active_segment()

            q = _make_query()
            results = mgr.search_multivector(q, k=5, ef=32, n_probes=2)
            assert len(results) > 0
            mgr.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
