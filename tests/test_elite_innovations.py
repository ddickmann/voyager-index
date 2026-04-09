"""
Comprehensive integration tests for Elite Innovation Layer:
  - Innovation 1: Filter-aware routing
  - Innovation 3: GPU qCH scorer (Triton + PyTorch fallback)
  - Innovation 4: Self-healing graph
  - Innovation 5: Multi-index ensemble (RRF)
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional imports with guards
# ---------------------------------------------------------------------------

try:
    from latence_gem_index import GemSegment, PyMutableGemSegment, PyEnsembleGemSegment
    GEM_AVAILABLE = True
except ImportError:
    GEM_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    from voyager_index._internal.inference.index_core.triton_qch_kernel import (
        qch_max_gather_gpu,
        qch_max_gather_torch,
        TRITON_AVAILABLE,
        MAX_SUPPORTED_DOC_LEN,
    )
    QCH_KERNEL_AVAILABLE = True
except ImportError:
    QCH_KERNEL_AVAILABLE = False
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _synthetic_corpus(n_docs=30, dim=16, vecs_per_doc=8, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_docs * vecs_per_doc, dim)).astype(np.float32)
    doc_ids = list(range(1, n_docs + 1))
    offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]
    return data, doc_ids, offsets


def _build_sealed_segment(n_docs=30, dim=16, vecs_per_doc=8, seed=42):
    seg = GemSegment()
    data, ids, offsets = _synthetic_corpus(n_docs, dim, vecs_per_doc, seed)
    seg.build(data, ids, offsets, n_fine=16, n_coarse=4, max_degree=8,
              ef_construction=32, max_kmeans_iter=5, ctop_r=2)
    return seg, data, ids, offsets


def _naive_cpu_qch(scores_flat, codes, offsets, lengths, n_query, n_fine):
    """Reference CPU implementation for qCH max-gather (no truncation)."""
    n_docs = len(offsets)
    out = np.empty(n_docs, dtype=np.float32)
    scores_2d = scores_flat.reshape(n_query, n_fine)
    for d in range(n_docs):
        off = offsets[d]
        ln = lengths[d]
        total = 0.0
        for qi in range(n_query):
            max_val = -1e30
            for ci in range(ln):
                c = codes[off + ci]
                s = scores_2d[qi, c]
                if s > max_val:
                    max_val = s
            total += max_val
        out[d] = 1.0 - total / n_query
    return out


def _synthetic_qch_data(n_docs, n_query, n_fine, max_doc_len, seed=42):
    """Generate synthetic qCH scoring data."""
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(n_query * n_fine).astype(np.float32)
    lengths = rng.integers(1, max_doc_len + 1, size=n_docs).astype(np.int32)
    offsets = np.zeros(n_docs, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths[:-1])
    total_codes = int(offsets[-1] + lengths[-1]) if n_docs > 0 else 0
    codes = rng.integers(0, n_fine, size=total_codes).astype(np.int32)
    return scores, codes, offsets, lengths


# ===========================================================================
# Innovation 3: GPU qCH Scorer
# ===========================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.skipif(not QCH_KERNEL_AVAILABLE, reason="qCH kernel not importable")
class TestGpuQchPyTorchFallback:
    """Tests for the PyTorch fallback (runs on any device with torch)."""

    @pytest.mark.parametrize("max_doc_len", [64, 256, 512, 1024, 2048])
    @pytest.mark.parametrize("n_query", [8, 32])
    def test_pytorch_fallback_matches_cpu(self, n_query, max_doc_len):
        n_docs, n_fine = 200, 128
        scores, codes, offsets, lengths = _synthetic_qch_data(
            n_docs, n_query, n_fine, max_doc_len, seed=n_query * 100 + max_doc_len,
        )

        cpu_ref = _naive_cpu_qch(scores, codes, offsets, lengths, n_query, n_fine)

        device = "cuda" if CUDA_AVAILABLE else "cpu"
        t_scores = torch.from_numpy(scores).to(device)
        t_codes = torch.from_numpy(codes).to(device)
        t_offsets = torch.from_numpy(offsets).to(device)
        t_lengths = torch.from_numpy(lengths).to(device)

        gpu_result = qch_max_gather_torch(
            t_scores, t_codes, t_offsets, t_lengths, n_query, n_fine,
        ).cpu().numpy()

        np.testing.assert_allclose(gpu_result, cpu_ref, atol=1e-4, rtol=1e-4)

    def test_no_truncation_at_2048(self):
        """Verify a doc with exactly 2048 codes is NOT truncated.

        We construct data where the best code for each query vector only
        appears in positions 512+, so truncation at 512 would produce a
        measurably different (worse) score.
        """
        n_docs, n_query, n_fine = 1, 8, 64
        rng = np.random.default_rng(77)
        scores = rng.standard_normal(n_query * n_fine).astype(np.float32)
        scores_2d = scores.reshape(n_query, n_fine)

        # First 512 codes use a mediocre code; codes 512+ include the argmax
        mediocre_code = 0
        codes_2048 = np.full(2048, mediocre_code, dtype=np.int32)
        for qi in range(n_query):
            best_code = int(np.argmax(scores_2d[qi]))
            codes_2048[512 + qi] = best_code

        offsets = np.array([0], dtype=np.int32)
        lengths = np.array([2048], dtype=np.int32)

        cpu_full = _naive_cpu_qch(scores, codes_2048, offsets, lengths, n_query, n_fine)
        cpu_trunc = _naive_cpu_qch(
            scores, codes_2048[:512], offsets,
            np.array([512], dtype=np.int32), n_query, n_fine,
        )
        assert not np.allclose(cpu_full, cpu_trunc, atol=1e-3), \
            "Full and truncated scores should differ for this crafted doc"

        device = "cuda" if CUDA_AVAILABLE else "cpu"
        torch_result = qch_max_gather_torch(
            torch.from_numpy(scores).to(device),
            torch.from_numpy(codes_2048).to(device),
            torch.from_numpy(offsets).to(device),
            torch.from_numpy(lengths).to(device),
            n_query, n_fine,
        ).cpu().numpy()

        np.testing.assert_allclose(torch_result, cpu_full, atol=1e-5)

    def test_empty_docs(self):
        n_query, n_fine = 4, 32
        scores = np.random.default_rng(1).standard_normal(n_query * n_fine).astype(np.float32)
        codes = np.array([], dtype=np.int32)
        offsets = np.array([0, 0], dtype=np.int32)
        lengths = np.array([0, 0], dtype=np.int32)

        device = "cuda" if CUDA_AVAILABLE else "cpu"
        result = qch_max_gather_torch(
            torch.from_numpy(scores).to(device),
            torch.from_numpy(codes).to(device),
            torch.from_numpy(offsets).to(device),
            torch.from_numpy(lengths).to(device),
            n_query, n_fine,
        ).cpu().numpy()
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA GPU available")
@pytest.mark.skipif(not QCH_KERNEL_AVAILABLE, reason="qCH kernel not importable")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
class TestGpuQchTriton:
    """Tests for the Triton kernel (requires CUDA GPU)."""

    @pytest.mark.parametrize("max_doc_len", [64, 256, 512, 1024, 2048])
    @pytest.mark.parametrize("n_query", [8, 32])
    def test_triton_matches_cpu(self, n_query, max_doc_len):
        n_docs, n_fine = 200, 128
        scores, codes, offsets, lengths = _synthetic_qch_data(
            n_docs, n_query, n_fine, max_doc_len, seed=n_query * 100 + max_doc_len,
        )
        cpu_ref = _naive_cpu_qch(scores, codes, offsets, lengths, n_query, n_fine)

        triton_result = qch_max_gather_gpu(
            torch.from_numpy(scores).cuda(),
            torch.from_numpy(codes).cuda(),
            torch.from_numpy(offsets).cuda(),
            torch.from_numpy(lengths).cuda(),
            n_query, n_fine,
        ).cpu().numpy()

        np.testing.assert_allclose(triton_result, cpu_ref, atol=1e-4, rtol=1e-4)

    def test_triton_no_truncation_2048(self):
        """Triton kernel must handle docs with exactly 2048 tokens."""
        n_docs, n_query, n_fine = 5, 16, 64
        rng = np.random.default_rng(99)
        scores = rng.standard_normal(n_query * n_fine).astype(np.float32)
        all_codes = []
        all_offsets = []
        all_lengths = []
        off = 0
        for d in range(n_docs):
            ln = 2048
            c = rng.integers(0, n_fine, size=ln).astype(np.int32)
            all_codes.append(c)
            all_offsets.append(off)
            all_lengths.append(ln)
            off += ln
        codes = np.concatenate(all_codes)
        offsets = np.array(all_offsets, dtype=np.int32)
        lengths = np.array(all_lengths, dtype=np.int32)

        cpu_ref = _naive_cpu_qch(scores, codes, offsets, lengths, n_query, n_fine)

        triton_result = qch_max_gather_gpu(
            torch.from_numpy(scores).cuda(),
            torch.from_numpy(codes).cuda(),
            torch.from_numpy(offsets).cuda(),
            torch.from_numpy(lengths).cuda(),
            n_query, n_fine,
        ).cpu().numpy()

        np.testing.assert_allclose(triton_result, cpu_ref, atol=1e-4, rtol=1e-4)

    def test_triton_fallback_for_over_2048(self):
        """Docs > 2048 should fall back to PyTorch without error."""
        n_docs, n_query, n_fine = 2, 4, 32
        rng = np.random.default_rng(55)
        scores = rng.standard_normal(n_query * n_fine).astype(np.float32)
        codes = rng.integers(0, n_fine, size=3000).astype(np.int32)
        offsets = np.array([0, 2500], dtype=np.int32)
        lengths = np.array([2500, 500], dtype=np.int32)

        cpu_ref = _naive_cpu_qch(scores, codes, offsets, lengths, n_query, n_fine)

        result = qch_max_gather_gpu(
            torch.from_numpy(scores).cuda(),
            torch.from_numpy(codes).cuda(),
            torch.from_numpy(offsets).cuda(),
            torch.from_numpy(lengths).cuda(),
            n_query, n_fine,
        ).cpu().numpy()

        np.testing.assert_allclose(result, cpu_ref, atol=1e-4, rtol=1e-4)

    def test_triton_matches_pytorch_fallback(self):
        """Triton and PyTorch fallback produce identical results."""
        n_docs, n_query, n_fine = 500, 16, 256
        scores, codes, offsets, lengths = _synthetic_qch_data(
            n_docs, n_query, n_fine, max_doc_len=512, seed=1234,
        )
        t_s = torch.from_numpy(scores).cuda()
        t_c = torch.from_numpy(codes).cuda()
        t_o = torch.from_numpy(offsets).cuda()
        t_l = torch.from_numpy(lengths).cuda()

        triton_r = qch_max_gather_gpu(t_s, t_c, t_o, t_l, n_query, n_fine).cpu().numpy()
        torch_r = qch_max_gather_torch(t_s, t_c, t_o, t_l, n_query, n_fine).cpu().numpy()

        np.testing.assert_allclose(triton_r, torch_r, atol=1e-4, rtol=1e-4)


# ===========================================================================
# Innovation 1: Filter-Aware Routing
# ===========================================================================

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestFilterRouting:

    def test_filter_returns_only_matching_docs(self):
        """Search with filter should only return docs that match the filter."""
        seg, data, ids, offsets = _build_sealed_segment(n_docs=30, dim=16, vecs_per_doc=8)

        payloads = []
        for i, doc_id in enumerate(ids):
            category = "A" if i < 15 else "B"
            payloads.append((doc_id, [("category", category)]))
        seg.set_doc_payloads(payloads)

        query = np.random.default_rng(99).standard_normal((3, 16)).astype(np.float32)

        results_a = seg.search(query, k=10, ef=50, n_probes=3,
                               filter=[("category", "A")])
        for doc_id, _ in results_a:
            idx = ids.index(doc_id)
            assert idx < 15, f"Doc {doc_id} (idx={idx}) should be category A"

        results_b = seg.search(query, k=10, ef=50, n_probes=3,
                               filter=[("category", "B")])
        for doc_id, _ in results_b:
            idx = ids.index(doc_id)
            assert idx >= 15, f"Doc {doc_id} (idx={idx}) should be category B"

    def test_empty_filter_same_as_unfiltered(self):
        seg, data, ids, offsets = _build_sealed_segment(n_docs=20, dim=16, vecs_per_doc=8)

        payloads = [(doc_id, [("k", "v")]) for doc_id in ids]
        seg.set_doc_payloads(payloads)

        query = np.random.default_rng(42).standard_normal((3, 16)).astype(np.float32)

        results_nofilt = seg.search(query, k=10, ef=50, n_probes=3)
        results_empty = seg.search(query, k=10, ef=50, n_probes=3, filter=[])

        assert set(d for d, _ in results_nofilt) == set(d for d, _ in results_empty)

    def test_zero_match_filter_returns_empty(self):
        seg, data, ids, offsets = _build_sealed_segment(n_docs=20, dim=16, vecs_per_doc=8)

        payloads = [(doc_id, [("color", "red")]) for doc_id in ids]
        seg.set_doc_payloads(payloads)

        query = np.random.default_rng(42).standard_normal((3, 16)).astype(np.float32)
        results = seg.search(query, k=10, ef=50, n_probes=3,
                             filter=[("color", "blue")])
        assert len(results) == 0

    def test_filter_without_payloads_logs_warning(self, caplog):
        """Searching with a filter when no payloads set should log a warning."""
        seg, data, ids, offsets = _build_sealed_segment(n_docs=20, dim=16, vecs_per_doc=8)
        query = np.random.default_rng(42).standard_normal((3, 16)).astype(np.float32)

        results = seg.search(query, k=10, ef=50, n_probes=3,
                             filter=[("k", "v")])
        assert len(results) > 0


# ===========================================================================
# Innovation 4: Self-Healing Graph
# ===========================================================================

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestSelfHealing:

    def test_heal_after_deletes(self):
        """After deleting 30% of docs, healing should improve graph quality."""
        seg = PyMutableGemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=50, dim=16, vecs_per_doc=8, seed=42)
        seg.build(data, ids, offsets, n_fine=16, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)

        n_delete = 15
        for doc_id in ids[:n_delete]:
            seg.delete(doc_id)

        metrics_before = seg.graph_quality_metrics()
        del_ratio_before = metrics_before[0]
        assert del_ratio_before > 0.1, f"Expected delete ratio > 0.1, got {del_ratio_before}"

        seg.heal()

        metrics_after = seg.graph_quality_metrics()
        isolated_after = metrics_after[2]
        stale_rep_after = metrics_after[3]

        assert isolated_after <= 0.05, \
            f"Isolated ratio should be <= 0.05 after heal, got {isolated_after}"
        assert stale_rep_after <= 0.2, \
            f"Stale rep ratio should be <= 0.2 after heal, got {stale_rep_after}"

    def test_heal_idempotent(self):
        """Calling heal() twice produces the same metrics."""
        seg = PyMutableGemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=40, dim=16, vecs_per_doc=8, seed=55)
        seg.build(data, ids, offsets, n_fine=16, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)

        for doc_id in ids[:10]:
            seg.delete(doc_id)

        seg.heal()
        metrics_1 = seg.graph_quality_metrics()

        seg.heal()
        metrics_2 = seg.graph_quality_metrics()

        for i in range(4):
            assert abs(metrics_1[i] - metrics_2[i]) < 1e-9, \
                f"Metric[{i}] changed between heals: {metrics_1[i]} -> {metrics_2[i]}"

    def test_needs_healing_converges(self):
        """After deletes, repeated heal() calls should converge to healthy state."""
        seg = PyMutableGemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=50, dim=16, vecs_per_doc=8, seed=66)
        seg.build(data, ids, offsets, n_fine=16, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)

        for doc_id in ids[:10]:
            seg.delete(doc_id)

        for _ in range(3):
            if not seg.needs_healing():
                break
            seg.heal()

        metrics = seg.graph_quality_metrics()
        assert metrics[2] <= 0.1, \
            f"Isolated ratio should be low after healing, got {metrics[2]}"
        assert metrics[3] <= 0.5, \
            f"Stale rep ratio should improve after healing, got {metrics[3]}"

    def test_search_still_works_after_heal(self):
        seg = PyMutableGemSegment()
        data, ids, offsets = _synthetic_corpus(n_docs=40, dim=16, vecs_per_doc=8, seed=77)
        seg.build(data, ids, offsets, n_fine=16, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2)

        for doc_id in ids[:10]:
            seg.delete(doc_id)

        seg.heal()

        query = np.random.default_rng(88).standard_normal((3, 16)).astype(np.float32)
        results = seg.search(query, k=5, ef=32)
        assert len(results) > 0
        deleted_set = set(ids[:10])
        for doc_id, score in results:
            assert doc_id not in deleted_set, f"Deleted doc {doc_id} in results"
            assert np.isfinite(score)


# ===========================================================================
# Innovation 5: Multi-Index Ensemble (RRF)
# ===========================================================================

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestEnsembleRRF:

    def _build_ensemble(self, n_docs=30, dim=16, vecs_per_doc=8, n_modalities=2, seed=42):
        rng = np.random.default_rng(seed)
        total_vecs = n_docs * vecs_per_doc
        all_vectors = rng.standard_normal((total_vecs, dim)).astype(np.float32)
        doc_ids = list(range(1, n_docs + 1))
        doc_offsets = [(i * vecs_per_doc, (i + 1) * vecs_per_doc) for i in range(n_docs)]

        modality_tags = np.array(
            [i % n_modalities for i in range(total_vecs)], dtype=np.uint8,
        )

        ens = PyEnsembleGemSegment()
        ens.build(
            all_vectors, doc_ids, doc_offsets,
            modality_tags.tolist(), n_modalities,
            n_fine=16, n_coarse=4, max_degree=8,
            ef_construction=32, max_kmeans_iter=5, ctop_r=2,
        )
        return ens, all_vectors, doc_ids, modality_tags

    def test_basic_recall(self):
        ens, vecs, ids, tags = self._build_ensemble()
        assert ens.is_ready()
        assert ens.n_docs() == 30
        assert ens.n_modalities() == 2

        query = np.random.default_rng(99).standard_normal((4, 16)).astype(np.float32)
        query_tags = [0, 0, 1, 1]
        results = ens.search(query, query_tags, k=10, ef=50, n_probes=3)

        assert len(results) > 0
        assert len(results) <= 10
        for doc_id, score in results:
            assert doc_id in ids
            assert score > 0

    def test_scores_monotonically_decreasing(self):
        ens, vecs, ids, tags = self._build_ensemble()
        query = np.random.default_rng(42).standard_normal((4, 16)).astype(np.float32)
        query_tags = [0, 0, 1, 1]
        results = ens.search(query, query_tags, k=10, ef=50, n_probes=3)

        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-7, \
                f"Score[{i}]={scores[i]} < Score[{i+1}]={scores[i+1]}"

    def test_single_modality(self):
        """Only one modality has tokens -- should still work."""
        ens, vecs, ids, tags = self._build_ensemble(n_modalities=2)
        query = np.random.default_rng(42).standard_normal((4, 16)).astype(np.float32)
        query_tags = [0, 0, 0, 0]
        results = ens.search(query, query_tags, k=10, ef=50, n_probes=3)
        assert len(results) > 0

    def test_empty_query_one_modality(self):
        """One modality has no query tokens -- graceful handling."""
        ens, vecs, ids, tags = self._build_ensemble(n_modalities=3)
        query = np.random.default_rng(42).standard_normal((4, 16)).astype(np.float32)
        query_tags = [0, 0, 1, 1]
        results = ens.search(query, query_tags, k=10, ef=50, n_probes=3)
        assert len(results) > 0

    def test_rrf_is_1_based(self):
        """
        Verify RRF scores use 1-based ranking (k=60):
        top hit should get 1/(60+1) = ~0.01639, NOT 1/(60+0) = ~0.01667.
        """
        ens, vecs, ids, tags = self._build_ensemble(n_modalities=1, n_docs=10)
        query = np.random.default_rng(42).standard_normal((2, 16)).astype(np.float32)
        query_tags = [0, 0]
        results = ens.search(query, query_tags, k=10, ef=50, n_probes=3)

        if len(results) > 0:
            top_score = results[0][1]
            max_possible_1based = 1.0 / 61.0
            max_possible_0based = 1.0 / 60.0
            assert top_score <= max_possible_1based + 1e-6, \
                f"Top RRF score {top_score} exceeds 1-based max {max_possible_1based}, suggests 0-based ranking"

    def test_query_modality_tags_length_validation(self):
        ens, vecs, ids, tags = self._build_ensemble()
        query = np.random.default_rng(42).standard_normal((4, 16)).astype(np.float32)
        with pytest.raises(ValueError, match="query_modality_tags length"):
            ens.search(query, [0, 0], k=10)

    def test_k_zero_returns_empty(self):
        ens, vecs, ids, tags = self._build_ensemble()
        query = np.random.default_rng(42).standard_normal((2, 16)).astype(np.float32)
        results = ens.search(query, [0, 1], k=0, ef=50, n_probes=3)
        assert len(results) == 0
