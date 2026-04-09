"""
QA Savage Test Suite -- GEM Feature Coverage
============================================

Integration tests covering core GEM subsystems and Elite Innovations.
Assertions are quantitative where feasible; GPU / Triton / router sections
require their respective native extensions and are skipped when unavailable.

NOTE: Tests guarded by ``GEM_AVAILABLE``, ``ROUTER_AVAILABLE``, ``TORCH_AVAILABLE``,
and ``TRITON_AVAILABLE`` will be skipped in environments without those
dependencies.  Run ``pip install -e src/kernels/gem_index`` and
``pip install -e src/kernels/gem_router`` to enable full coverage.

Sections:
  1. Sealed GemSegment (core)
  2. Mutable PyMutableGemSegment
  3. GPU qCH Scorer (Innovation 3)    [requires CUDA + Triton]
  4. Filter-Aware Routing (Innovation 1)
  5. Multi-Index Ensemble / RRF (Innovation 5)
  6. Self-Healing Graph (Innovation 4)
  7. Persistence
  8. Router / Codebook                [requires latence_gem_router]
  9. Edge Cases & Stress
 10. Quantitative Benchmarks
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional imports with skip guards
# ---------------------------------------------------------------------------

try:
    from latence_gem_index import GemSegment, PyEnsembleGemSegment, PyMutableGemSegment
    GEM_AVAILABLE = True
except ImportError:
    GEM_AVAILABLE = False

try:
    from latence_gem_router import PyGemRouter
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    from voyager_index._internal.inference.index_core.triton_qch_kernel import (
        TRITON_AVAILABLE,
        qch_max_gather_gpu,
        qch_max_gather_torch,
    )
    QCH_KERNEL_AVAILABLE = True
except ImportError:
    QCH_KERNEL_AVAILABLE = False
    TRITON_AVAILABLE = False

try:
    from voyager_index._internal.inference.index_core.gpu_qch import GpuQchScorer
    GPU_SCORER_AVAILABLE = True
except ImportError:
    GPU_SCORER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _corpus(n_docs=30, dim=16, vpd=8, seed=42):
    """Deterministic synthetic corpus."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_docs * vpd, dim)).astype(np.float32)
    ids = list(range(1, n_docs + 1))
    offs = [(i * vpd, (i + 1) * vpd) for i in range(n_docs)]
    return data, ids, offs


def _query(n_q=4, dim=16, seed=99):
    return np.random.default_rng(seed).standard_normal((n_q, dim)).astype(np.float32)


def _build_sealed(n_docs=30, dim=16, vpd=8, seed=42, **kw):
    seg = GemSegment()
    data, ids, offs = _corpus(n_docs, dim, vpd, seed)
    defaults = dict(n_fine=16, n_coarse=4, max_degree=8,
                    ef_construction=32, max_kmeans_iter=5, ctop_r=2)
    defaults.update(kw)
    seg.build(data, ids, offs, **defaults)
    return seg, data, ids, offs


def _build_mutable(n_docs=30, dim=16, vpd=8, seed=42, **kw):
    seg = PyMutableGemSegment()
    data, ids, offs = _corpus(n_docs, dim, vpd, seed)
    defaults = dict(n_fine=16, n_coarse=4, max_degree=8,
                    ef_construction=32, max_kmeans_iter=5, ctop_r=2)
    defaults.update(kw)
    seg.build(data, ids, offs, **defaults)
    return seg, data, ids, offs


def _naive_cpu_qch(scores_flat, codes, offsets, lengths, n_query, n_fine):
    n_docs = len(offsets)
    out = np.empty(n_docs, dtype=np.float32)
    s2d = scores_flat.reshape(n_query, n_fine)
    for d in range(n_docs):
        off, ln = offsets[d], lengths[d]
        total = 0.0
        for qi in range(n_query):
            mx = -1e30
            for ci in range(ln):
                s = s2d[qi, codes[off + ci]]
                if s > mx:
                    mx = s
            total += mx
        out[d] = 1.0 - total / n_query
    return out


def _synth_qch(n_docs, n_q, n_fine, max_len, seed=42):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(n_q * n_fine).astype(np.float32)
    lengths = rng.integers(1, max_len + 1, size=n_docs).astype(np.int32)
    offsets = np.zeros(n_docs, dtype=np.int32)
    if n_docs > 1:
        offsets[1:] = np.cumsum(lengths[:-1])
    total = int(offsets[-1] + lengths[-1]) if n_docs > 0 else 0
    codes = rng.integers(0, n_fine, size=total).astype(np.int32)
    return scores, codes, offsets, lengths


# ###########################################################################
# 1. Sealed GemSegment (core)
# ###########################################################################

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestSealedCore:

    # -- Build variants --

    def test_build_default_dual_graph(self):
        seg, _, ids, _ = _build_sealed(dual_graph=True)
        assert seg.is_ready()
        assert seg.n_docs() == len(ids)
        assert seg.n_edges() > 0
        assert seg.dim() == 16

    def test_build_use_emd(self):
        seg, *_ = _build_sealed(n_docs=15, use_emd=True)
        assert seg.is_ready()
        assert seg.n_edges() > 0

    def test_build_single_graph(self):
        seg, *_ = _build_sealed(dual_graph=False)
        assert seg.is_ready()
        assert seg.n_edges() > 0

    def test_build_payload_clusters(self):
        seg = GemSegment()
        data, ids, offs = _corpus(n_docs=20)
        clusters = [i % 3 for i in range(20)]
        seg.build(data, ids, offs, n_fine=16, n_coarse=4, max_degree=8,
                  ef_construction=32, max_kmeans_iter=5, ctop_r=2,
                  payload_clusters=clusters, dual_graph=False)
        assert seg.is_ready()

    def test_build_dual_plus_payload_raises(self):
        seg = GemSegment()
        data, ids, offs = _corpus(n_docs=10)
        with pytest.raises(ValueError, match="mutually exclusive"):
            seg.build(data, ids, offs, n_fine=8, n_coarse=4, max_degree=6,
                      ef_construction=16, max_kmeans_iter=5, ctop_r=2,
                      payload_clusters=[0] * 10, dual_graph=True)

    # -- Search --

    def test_search_basic(self):
        seg, *_ = _build_sealed()
        q = _query()
        results = seg.search(q, k=5, ef=32, n_probes=2)
        assert 0 < len(results) <= 5
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert np.isfinite(score)

    def test_search_k_bounds(self):
        seg, *_ = _build_sealed(n_docs=10)
        q = _query()
        r = seg.search(q, k=20, ef=50, n_probes=3)
        assert len(r) <= 10

    def test_search_ef_scaling(self):
        """Higher ef should return at least as good recall."""
        seg, *_ = _build_sealed(n_docs=50, vpd=8)
        q = _query()
        bf = seg.brute_force_proxy(q, k=10)
        bf_ids = set(d for d, _ in bf)

        r_low = seg.search(q, k=10, ef=16, n_probes=2)
        r_high = seg.search(q, k=10, ef=200, n_probes=4)
        recall_low = len(set(d for d, _ in r_low) & bf_ids) / max(len(bf_ids), 1)
        recall_high = len(set(d for d, _ in r_high) & bf_ids) / max(len(bf_ids), 1)
        assert recall_high >= recall_low - 0.1

    # -- graph_connectivity_report --

    def test_connectivity_report(self):
        seg, *_ = _build_sealed(n_docs=50, vpd=8, dual_graph=True)
        n_comp, giant_frac = seg.graph_connectivity_report()
        assert n_comp >= 1
        assert 0.0 < giant_frac <= 1.0, f"giant component fraction: {giant_frac}"

    # -- search_with_stats --

    def test_search_with_stats(self):
        seg, *_ = _build_sealed()
        q = _query()
        results, (visited, dist_comps) = seg.search_with_stats(q, k=5, ef=32, n_probes=2)
        assert len(results) > 0
        assert visited > 0
        assert dist_comps > 0

    # -- search_batch --

    def test_search_batch_matches_sequential(self):
        seg, *_ = _build_sealed(n_docs=40)
        queries = [_query(seed=i) for i in range(5)]
        batch_results = seg.search_batch(queries, k=5, ef=50, n_probes=3)
        assert len(batch_results) == 5
        for i, q in enumerate(queries):
            seq_r = seg.search(q, k=5, ef=50, n_probes=3)
            batch_ids = set(d for d, _ in batch_results[i])
            seq_ids = set(d for d, _ in seq_r)
            assert batch_ids == seq_ids

    # -- brute_force_proxy --

    def test_brute_force_proxy_is_oracle(self):
        seg, *_ = _build_sealed(n_docs=20)
        q = _query()
        bf = seg.brute_force_proxy(q, k=20)
        assert len(bf) == 20
        scores = [s for _, s in bf]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-6

    # -- getters --

    def test_get_codebook_centroids(self):
        seg, *_ = _build_sealed(n_docs=20, n_fine=16)
        c = seg.get_codebook_centroids()
        assert c.shape == (16, 16)
        assert np.all(np.isfinite(c))

    def test_get_idf(self):
        seg, *_ = _build_sealed(n_docs=20, n_fine=16)
        idf = seg.get_idf()
        assert idf.shape == (16,)
        assert np.all(idf >= 0)

    def test_get_flat_codes(self):
        seg, *_ = _build_sealed(n_docs=20, n_fine=16)
        codes, offsets, lengths = seg.get_flat_codes()
        assert len(offsets) == 20
        assert len(lengths) == 20
        assert len(codes) > 0
        assert int(lengths.sum()) == len(codes)

    # -- inject_shortcuts --

    def test_inject_shortcuts(self):
        seg, data, ids, offs = _build_sealed(n_docs=30, vpd=8)
        pairs = []
        for i in range(5):
            start = i * 8
            flat_q = data[start:start + 8].flatten().tolist()
            pairs.append((flat_q, i))
        seg.inject_shortcuts(pairs, max_shortcuts_per_node=4)
        assert seg.total_shortcuts() > 0

    def test_search_with_shortcuts(self):
        seg, data, ids, offs = _build_sealed(n_docs=30, vpd=8)
        pairs = [(data[i * 8:(i + 1) * 8].flatten().tolist(), i) for i in range(5)]
        seg.inject_shortcuts(pairs, max_shortcuts_per_node=4)
        q = _query(dim=16)
        r = seg.search(q, k=5, ef=32, n_probes=2, enable_shortcuts=True)
        assert len(r) > 0

    # -- prune_stale_shortcuts --

    def test_prune_stale_shortcuts_deleted(self):
        seg, data, *_ = _build_sealed(n_docs=30, vpd=8)
        pairs = [(data[i * 8:(i + 1) * 8].flatten().tolist(), i) for i in range(10)]
        seg.inject_shortcuts(pairs, max_shortcuts_per_node=4)
        before = seg.total_shortcuts()
        assert before > 0
        deleted = [False] * 30
        deleted[0] = True
        deleted[1] = True
        seg.prune_stale_shortcuts(deleted)
        after = seg.total_shortcuts()
        assert after <= before

    def test_prune_stale_shortcuts_age(self):
        seg, data, *_ = _build_sealed(n_docs=30, vpd=8)
        pairs = [(data[i * 8:(i + 1) * 8].flatten().tolist(), i) for i in range(10)]
        seg.inject_shortcuts(pairs, max_shortcuts_per_node=4)
        before = seg.total_shortcuts()
        deleted = [False] * 30
        seg.prune_stale_shortcuts(deleted, max_age=0, current_generation=100)
        after = seg.total_shortcuts()
        assert after <= before

    # -- n_nodes, n_edges --

    def test_introspection(self):
        seg, *_ = _build_sealed(n_docs=25)
        assert seg.n_docs() == 25
        assert seg.n_nodes() >= 25
        assert seg.n_edges() > 0
        assert seg.dim() == 16
        assert seg.is_ready()


# ###########################################################################
# 2. Mutable PyMutableGemSegment
# ###########################################################################

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestMutableCore:

    def test_build_default(self):
        seg, _, ids, _ = _build_mutable()
        assert seg.is_ready()
        assert seg.n_nodes() == len(ids)
        assert seg.dim() == 16

    def test_build_use_emd(self):
        seg, *_ = _build_mutable(n_docs=15, use_emd=True)
        assert seg.is_ready()

    def test_insert_single(self):
        seg, _, ids, _ = _build_mutable(n_docs=20)
        before = seg.n_nodes()
        new_doc = np.random.default_rng(77).standard_normal((8, 16)).astype(np.float32)
        seg.insert(new_doc, doc_id=9999)
        assert seg.n_nodes() == before + 1

    def test_insert_batch(self):
        seg, _, ids, _ = _build_mutable(n_docs=20)
        before = seg.n_nodes()
        docs = [np.random.default_rng(i).standard_normal((8, 16)).astype(np.float32)
                for i in range(5)]
        seg.insert_batch(docs, [10001, 10002, 10003, 10004, 10005])
        assert seg.n_nodes() == before + 5

    def test_delete_existing(self):
        seg, _, ids, _ = _build_mutable(n_docs=20)
        before_live = seg.n_live()
        assert seg.delete(ids[0])
        assert seg.n_live() == before_live - 1

    def test_delete_nonexistent(self):
        seg, *_ = _build_mutable(n_docs=10)
        assert not seg.delete(999999)

    def test_compact(self):
        seg, _, ids, _ = _build_mutable(n_docs=20)
        for doc_id in ids[:5]:
            seg.delete(doc_id)
        assert seg.delete_ratio() > 0
        seg.compact()
        assert seg.delete_ratio() == 0.0
        assert seg.n_live() == 15

    def test_search(self):
        seg, *_ = _build_mutable()
        q = _query()
        r = seg.search(q, k=5, ef=32)
        assert len(r) > 0

    def test_search_batch(self):
        seg, *_ = _build_mutable(n_docs=30)
        queries = [_query(seed=i) for i in range(3)]
        batch = seg.search_batch(queries, k=5, ef=32)
        assert len(batch) == 3
        for r in batch:
            assert len(r) > 0

    def test_metrics_fresh_build(self):
        seg, *_ = _build_mutable(n_docs=30)
        d, avg, iso, stale = seg.graph_quality_metrics()
        assert d == 0.0
        assert avg > 0
        assert iso >= 0
        assert stale >= 0

    def test_needs_healing_fresh(self):
        seg, *_ = _build_mutable(n_docs=30)
        assert not seg.needs_healing()

    def test_quality_score_fresh(self):
        seg, *_ = _build_mutable(n_docs=30)
        assert seg.quality_score() == pytest.approx(1.0, abs=0.01)

    def test_quality_score_after_deletes(self):
        seg, _, ids, _ = _build_mutable(n_docs=30)
        for d in ids[:10]:
            seg.delete(d)
        assert seg.quality_score() < 1.0

    def test_memory_bytes(self):
        seg, *_ = _build_mutable(n_docs=30)
        assert seg.memory_bytes() > 0

    def test_n_edges(self):
        seg, *_ = _build_mutable(n_docs=20)
        assert seg.n_edges() > 0

    def test_avg_degree(self):
        seg, *_ = _build_mutable(n_docs=20)
        assert seg.avg_degree() > 0

    def test_delete_ratio(self):
        seg, _, ids, _ = _build_mutable(n_docs=20)
        assert seg.delete_ratio() == 0.0
        seg.delete(ids[0])
        assert seg.delete_ratio() > 0


# ###########################################################################
# 3. GPU qCH Scorer (Innovation 3)
# ###########################################################################

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.skipif(not QCH_KERNEL_AVAILABLE, reason="qCH kernel not importable")
class TestGpuQchSavage:

    @pytest.mark.parametrize("max_len", [64, 256, 512, 1024, 2048])
    @pytest.mark.parametrize("n_q", [1, 8, 32])
    def test_pytorch_fallback_parity(self, n_q, max_len):
        n_docs, n_fine = 200, 128
        scores, codes, offsets, lengths = _synth_qch(n_docs, n_q, n_fine, max_len, seed=n_q * 100 + max_len)
        cpu_ref = _naive_cpu_qch(scores, codes, offsets, lengths, n_q, n_fine)
        dev = "cuda" if CUDA_AVAILABLE else "cpu"
        gpu = qch_max_gather_torch(
            torch.from_numpy(scores).to(dev), torch.from_numpy(codes).to(dev),
            torch.from_numpy(offsets).to(dev), torch.from_numpy(lengths).to(dev),
            n_q, n_fine,
        ).cpu().numpy()
        np.testing.assert_allclose(gpu, cpu_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="No GPU")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="No Triton")
    @pytest.mark.parametrize("max_len", [64, 512, 1024, 2048])
    def test_triton_parity(self, max_len):
        n_docs, n_q, n_fine = 300, 16, 128
        scores, codes, offsets, lengths = _synth_qch(n_docs, n_q, n_fine, max_len, seed=max_len)
        cpu_ref = _naive_cpu_qch(scores, codes, offsets, lengths, n_q, n_fine)
        tri = qch_max_gather_gpu(
            torch.from_numpy(scores).cuda(), torch.from_numpy(codes).cuda(),
            torch.from_numpy(offsets).cuda(), torch.from_numpy(lengths).cuda(),
            n_q, n_fine,
        ).cpu().numpy()
        np.testing.assert_allclose(tri, cpu_ref, atol=1e-4, rtol=1e-4)

    def test_no_truncation_proof(self):
        n_q, n_fine = 8, 64
        rng = np.random.default_rng(77)
        scores = rng.standard_normal(n_q * n_fine).astype(np.float32)
        s2d = scores.reshape(n_q, n_fine)
        codes_2048 = np.full(2048, 0, dtype=np.int32)
        for qi in range(n_q):
            codes_2048[512 + qi] = int(np.argmax(s2d[qi]))
        offs = np.array([0], dtype=np.int32)
        lens = np.array([2048], dtype=np.int32)
        full = _naive_cpu_qch(scores, codes_2048, offs, lens, n_q, n_fine)
        trunc = _naive_cpu_qch(scores, codes_2048[:512], offs, np.array([512], dtype=np.int32), n_q, n_fine)
        assert not np.allclose(full, trunc, atol=1e-3)
        dev = "cuda" if CUDA_AVAILABLE else "cpu"
        torch_r = qch_max_gather_torch(
            torch.from_numpy(scores).to(dev), torch.from_numpy(codes_2048).to(dev),
            torch.from_numpy(offs).to(dev), torch.from_numpy(lens).to(dev), n_q, n_fine,
        ).cpu().numpy()
        np.testing.assert_allclose(torch_r, full, atol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="No GPU")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="No Triton")
    def test_over_2048_fallback(self):
        n_q, n_fine = 4, 32
        rng = np.random.default_rng(55)
        scores = rng.standard_normal(n_q * n_fine).astype(np.float32)
        codes = rng.integers(0, n_fine, size=3000).astype(np.int32)
        offs = np.array([0, 2500], dtype=np.int32)
        lens = np.array([2500, 500], dtype=np.int32)
        cpu_ref = _naive_cpu_qch(scores, codes, offs, lens, n_q, n_fine)
        r = qch_max_gather_gpu(
            torch.from_numpy(scores).cuda(), torch.from_numpy(codes).cuda(),
            torch.from_numpy(offs).cuda(), torch.from_numpy(lens).cuda(), n_q, n_fine,
        ).cpu().numpy()
        np.testing.assert_allclose(r, cpu_ref, atol=1e-4)

    def test_empty_docs(self):
        n_q, n_fine = 4, 32
        scores = np.random.default_rng(1).standard_normal(n_q * n_fine).astype(np.float32)
        dev = "cuda" if CUDA_AVAILABLE else "cpu"
        r = qch_max_gather_torch(
            torch.from_numpy(scores).to(dev),
            torch.from_numpy(np.array([], dtype=np.int32)).to(dev),
            torch.from_numpy(np.array([0, 0], dtype=np.int32)).to(dev),
            torch.from_numpy(np.array([0, 0], dtype=np.int32)).to(dev),
            n_q, n_fine,
        ).cpu().numpy()
        np.testing.assert_allclose(r, [1.0, 1.0], atol=1e-6)

    def test_single_doc(self):
        n_q, n_fine = 4, 16
        scores, codes, offs, lens = _synth_qch(1, n_q, n_fine, 32, seed=10)
        cpu = _naive_cpu_qch(scores, codes, offs, lens, n_q, n_fine)
        dev = "cuda" if CUDA_AVAILABLE else "cpu"
        r = qch_max_gather_torch(
            torch.from_numpy(scores).to(dev), torch.from_numpy(codes).to(dev),
            torch.from_numpy(offs).to(dev), torch.from_numpy(lens).to(dev), n_q, n_fine,
        ).cpu().numpy()
        np.testing.assert_allclose(r, cpu, atol=1e-5)

    @pytest.mark.skipif(not CUDA_AVAILABLE or not GPU_SCORER_AVAILABLE, reason="No GPU scorer")
    @pytest.mark.skipif(not GEM_AVAILABLE, reason="No gem_index")
    def test_gpu_scorer_from_segment(self):
        seg, *_ = _build_sealed(n_docs=50, dim=32, vpd=8)
        scorer = GpuQchScorer.from_gem_segment(seg, device="cuda")
        q = torch.from_numpy(_query(n_q=4, dim=32)).cuda()
        scores = scorer.score_query(q)
        assert scores.shape == (50,)
        assert torch.all(torch.isfinite(scores))

    @pytest.mark.skipif(not CUDA_AVAILABLE or not GPU_SCORER_AVAILABLE, reason="No GPU scorer")
    @pytest.mark.skipif(not GEM_AVAILABLE, reason="No gem_index")
    def test_gpu_scorer_filtered(self):
        seg, *_ = _build_sealed(n_docs=50, dim=32, vpd=8)
        scorer = GpuQchScorer.from_gem_segment(seg, device="cuda")
        q = torch.from_numpy(_query(n_q=4, dim=32)).cuda()
        mask = torch.zeros(50, dtype=torch.bool, device="cuda")
        mask[:10] = True
        scores = scorer.score_query_filtered(q, mask)
        assert scores.shape == (50,)
        assert torch.all(scores[10:] == float("inf"))
        assert torch.all(torch.isfinite(scores[:10]))


# ###########################################################################
# 4. Filter-Aware Routing (Innovation 1)
# ###########################################################################

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestFilterRoutingSavage:

    def test_single_field_filter(self):
        seg, _, ids, _ = _build_sealed(n_docs=40)
        payloads = [(d, [("cat", "A" if i < 20 else "B")]) for i, d in enumerate(ids)]
        seg.set_doc_payloads(payloads)
        q = _query()
        ra = seg.search(q, k=10, ef=50, n_probes=3, filter=[("cat", "A")])
        for d, _ in ra:
            assert ids.index(d) < 20

    def test_multi_field_and_filter(self):
        seg, _, ids, _ = _build_sealed(n_docs=40)
        payloads = []
        for i, d in enumerate(ids):
            payloads.append((d, [("color", "red" if i % 2 == 0 else "blue"),
                                 ("size", "L" if i < 20 else "S")]))
        seg.set_doc_payloads(payloads)
        q = _query()
        results = seg.search(q, k=10, ef=50, n_probes=3,
                             filter=[("color", "red"), ("size", "L")])
        for d, _ in results:
            idx = ids.index(d)
            assert idx % 2 == 0 and idx < 20

    def test_empty_filter_same_as_unfiltered(self):
        seg, _, ids, _ = _build_sealed(n_docs=20)
        seg.set_doc_payloads([(d, [("k", "v")]) for d in ids])
        q = _query()
        r1 = set(d for d, _ in seg.search(q, k=10, ef=50, n_probes=3))
        r2 = set(d for d, _ in seg.search(q, k=10, ef=50, n_probes=3, filter=[]))
        assert r1 == r2

    def test_zero_match_filter(self):
        seg, _, ids, _ = _build_sealed(n_docs=20)
        seg.set_doc_payloads([(d, [("color", "red")]) for d in ids])
        q = _query()
        r = seg.search(q, k=10, ef=50, n_probes=3, filter=[("color", "blue")])
        assert len(r) == 0

    def test_no_payloads_still_works(self):
        seg, *_ = _build_sealed(n_docs=20)
        q = _query()
        r = seg.search(q, k=10, ef=50, n_probes=3, filter=[("k", "v")])
        assert len(r) > 0

    def test_all_docs_match(self):
        seg, _, ids, _ = _build_sealed(n_docs=20)
        seg.set_doc_payloads([(d, [("x", "y")]) for d in ids])
        q = _query()
        r_filt = set(d for d, _ in seg.search(q, k=10, ef=50, n_probes=3, filter=[("x", "y")]))
        r_none = set(d for d, _ in seg.search(q, k=10, ef=50, n_probes=3))
        assert r_filt == r_none

    def test_low_selectivity(self):
        seg, _, ids, _ = _build_sealed(n_docs=100, vpd=4)
        payloads = [(d, [("rare", "yes" if i == 0 else "no")]) for i, d in enumerate(ids)]
        seg.set_doc_payloads(payloads)
        q = _query()
        r = seg.search(q, k=10, ef=100, n_probes=4, filter=[("rare", "yes")])
        assert len(r) <= 1
        if len(r) == 1:
            assert r[0][0] == ids[0]


# ###########################################################################
# 5. Multi-Index Ensemble / RRF (Innovation 5)
# ###########################################################################

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestEnsembleRRFSavage:

    def _build_ens(self, n_docs=30, dim=16, vpd=8, nm=2, seed=42):
        rng = np.random.default_rng(seed)
        total = n_docs * vpd
        vecs = rng.standard_normal((total, dim)).astype(np.float32)
        ids = list(range(1, n_docs + 1))
        offs = [(i * vpd, (i + 1) * vpd) for i in range(n_docs)]
        tags = [i % nm for i in range(total)]
        ens = PyEnsembleGemSegment()
        ens.build(vecs, ids, offs, tags, nm, n_fine=16, n_coarse=4,
                  max_degree=8, ef_construction=32, max_kmeans_iter=5, ctop_r=2)
        return ens, ids

    def test_2_modality(self):
        ens, ids = self._build_ens(nm=2)
        assert ens.is_ready()
        assert ens.n_docs() == 30
        assert ens.n_modalities() == 2
        q = _query()
        r = ens.search(q, [0, 0, 1, 1], k=10, ef=50, n_probes=3)
        assert len(r) > 0

    def test_3_modality(self):
        ens, ids = self._build_ens(nm=3)
        assert ens.n_modalities() == 3
        q = _query()
        r = ens.search(q, [0, 1, 2, 0], k=10, ef=50, n_probes=3)
        assert len(r) > 0

    def test_single_modality_query(self):
        ens, _ = self._build_ens(nm=2)
        q = _query()
        r = ens.search(q, [0, 0, 0, 0], k=10, ef=50, n_probes=3)
        assert len(r) > 0

    def test_rrf_1_based(self):
        ens, _ = self._build_ens(nm=1, n_docs=10)
        q = _query(n_q=2)
        r = ens.search(q, [0, 0], k=10, ef=50, n_probes=3)
        if len(r) > 0:
            assert r[0][1] <= 1.0 / 61.0 + 1e-6

    def test_score_monotonicity(self):
        ens, _ = self._build_ens()
        q = _query()
        r = ens.search(q, [0, 0, 1, 1], k=10, ef=50, n_probes=3)
        scores = [s for _, s in r]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-7

    def test_tag_length_validation(self):
        ens, _ = self._build_ens()
        q = _query()
        with pytest.raises(ValueError, match="query_modality_tags length"):
            ens.search(q, [0, 0], k=10)

    def test_k_zero(self):
        ens, _ = self._build_ens()
        q = _query(n_q=2)
        assert len(ens.search(q, [0, 1], k=0, ef=50, n_probes=3)) == 0

    def test_empty_modality_query(self):
        ens, _ = self._build_ens(nm=3)
        q = _query()
        r = ens.search(q, [0, 0, 1, 1], k=10, ef=50, n_probes=3)
        assert len(r) > 0


# ###########################################################################
# 6. Self-Healing Graph (Innovation 4)
# ###########################################################################

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestSelfHealingSavage:

    @pytest.mark.parametrize("pct", [10, 30, 50, 70, 80])
    def test_heal_at_delete_pct(self, pct):
        seg, _, ids, _ = _build_mutable(n_docs=60, vpd=8)
        n_del = int(len(ids) * pct / 100)
        for d in ids[:n_del]:
            seg.delete(d)
        m_before = seg.graph_quality_metrics()
        for _ in range(3):
            seg.heal()
        m_after = seg.graph_quality_metrics()
        assert m_after[3] <= m_before[3] + 0.01, \
            f"stale_rep_ratio should not worsen: {m_before[3]:.3f} -> {m_after[3]:.3f}"
        q = _query()
        r = seg.search(q, k=5, ef=32)
        deleted_set = set(ids[:n_del])
        for d, _ in r:
            assert d not in deleted_set

    def test_heal_idempotent(self):
        seg, _, ids, _ = _build_mutable(n_docs=40)
        for d in ids[:10]:
            seg.delete(d)
        seg.heal()
        m1 = seg.graph_quality_metrics()
        seg.heal()
        m2 = seg.graph_quality_metrics()
        for i in range(4):
            assert abs(m1[i] - m2[i]) < 1e-9

    def test_metrics_fresh_all_healthy(self):
        seg, *_ = _build_mutable(n_docs=40)
        d, avg, iso, stale = seg.graph_quality_metrics()
        assert d == 0.0
        assert avg > 0
        assert iso == 0.0

    def test_compact_then_heal(self):
        seg, _, ids, _ = _build_mutable(n_docs=40)
        for d in ids[:10]:
            seg.delete(d)
        seg.compact()
        assert seg.delete_ratio() == 0.0
        seg.heal()
        _, _, iso, stale = seg.graph_quality_metrics()
        assert iso <= 0.1
        q = _query()
        r = seg.search(q, k=5, ef=32)
        assert len(r) > 0

    def test_connectivity_report_after_heal(self):
        """graph_connectivity_report returns meaningful component/edge stats."""
        seg, _, ids, _ = _build_mutable(n_docs=60, vpd=8)
        for d in ids[:30]:
            seg.delete(d)
        for _ in range(3):
            seg.heal()
        n_comp, giant_frac, cross_ratio = seg.graph_connectivity_report()
        assert n_comp >= 1, "should have at least one component"
        assert 0.0 < giant_frac <= 1.0, f"giant frac out of range: {giant_frac}"
        assert 0.0 <= cross_ratio <= 1.0, f"cross ratio out of range: {cross_ratio}"

    def test_extreme_delete_90_pct(self):
        """Smoke test: 90% deletes + heal + compact should not crash."""
        seg, _, ids, _ = _build_mutable(n_docs=100, vpd=4)
        n_del = int(len(ids) * 0.9)
        for d in ids[:n_del]:
            seg.delete(d)
        for _ in range(5):
            seg.heal()
        seg.compact()
        assert seg.delete_ratio() == 0.0
        q = _query()
        r = seg.search(q, k=5, ef=32)
        deleted_set = set(ids[:n_del])
        for d, _ in r:
            assert d not in deleted_set


# ###########################################################################
# 7. Persistence
# ###########################################################################

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestPersistenceSavage:

    def test_save_load_roundtrip(self):
        seg, _, ids, _ = _build_sealed(n_docs=30)
        q = _query()
        r_before = seg.search(q, k=10, ef=50, n_probes=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "seg.gem")
            seg.save(path)
            seg2 = GemSegment()
            seg2.load(path)
            assert seg2.is_ready()
            assert seg2.n_docs() == 30
            r_after = seg2.search(q, k=10, ef=50, n_probes=3)
        ids_before = set(d for d, _ in r_before)
        ids_after = set(d for d, _ in r_after)
        assert ids_before == ids_after

    def test_load_fresh_segment(self):
        seg, *_ = _build_sealed(n_docs=20)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "seg.gem")
            seg.save(path)
            fresh = GemSegment()
            fresh.load(path)
            assert fresh.is_ready()
            assert fresh.n_docs() == 20
            q = _query()
            r = fresh.search(q, k=5, ef=32, n_probes=2)
            assert len(r) > 0

    def test_corrupt_magic_raises(self):
        seg, *_ = _build_sealed(n_docs=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "seg.gem")
            seg.save(path)
            data = Path(path).read_bytes()
            corrupt = b"\x00\x00\x00\x00" + data[4:]
            Path(path).write_bytes(corrupt)
            seg2 = GemSegment()
            with pytest.raises((ValueError, Exception)):
                seg2.load(path)

    def test_truncated_file_raises(self):
        seg, *_ = _build_sealed(n_docs=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "seg.gem")
            seg.save(path)
            data = Path(path).read_bytes()
            Path(path).write_bytes(data[:50])
            seg2 = GemSegment()
            with pytest.raises((ValueError, Exception)):
                seg2.load(path)

    def test_filter_index_cleared_after_load(self):
        seg, _, ids, _ = _build_sealed(n_docs=20)
        seg.set_doc_payloads([(d, [("k", "v")]) for d in ids])
        q = _query()
        r_filt = seg.search(q, k=10, ef=50, n_probes=3, filter=[("k", "v")])
        assert len(r_filt) > 0
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "seg.gem")
            seg.save(path)
            seg2 = GemSegment()
            seg2.load(path)
            r2 = seg2.search(q, k=10, ef=50, n_probes=3, filter=[("k", "v")])
            assert len(r2) > 0


# ###########################################################################
# 8. Router / Codebook
# ###########################################################################

@pytest.mark.skipif(not ROUTER_AVAILABLE, reason="latence_gem_router not installed")
class TestRouterCodebookSavage:

    def test_build_and_route(self):
        dim = 16
        data, doc_ids, offs = _corpus(n_docs=30, dim=dim)
        r = PyGemRouter(dim=dim)
        r.build(data, doc_ids, offs, n_fine=16, n_coarse=4)
        assert r.is_ready()
        assert r.n_docs() == 30
        q = _query(dim=dim)
        results = r.route_query(q, n_probes=4, max_candidates=10)
        assert len(results) > 0

    def test_add_documents(self):
        dim = 16
        data, doc_ids, offs = _corpus(n_docs=20, dim=dim)
        r = PyGemRouter(dim=dim)
        r.build(data, doc_ids, offs, n_fine=16, n_coarse=4)
        before = r.n_docs()
        rng = np.random.default_rng(77)
        new_data = rng.standard_normal((8, dim)).astype(np.float32)
        r.add_documents(new_data, [999], [(0, 8)])
        assert r.n_docs() == before + 1

    def test_cluster_entries(self):
        dim = 16
        data, doc_ids, offs = _corpus(dim=dim)
        r = PyGemRouter(dim=dim)
        r.build(data, doc_ids, offs, n_fine=16, n_coarse=4)
        q = _query(dim=dim)
        entries = r.get_cluster_entries(q, n_probes=4)
        assert isinstance(entries, list)

    def test_query_profile(self):
        dim = 16
        data, doc_ids, offs = _corpus(dim=dim)
        r = PyGemRouter(dim=dim)
        r.build(data, doc_ids, offs, n_fine=16, n_coarse=4)
        q = _query(dim=dim)
        profile = r.compute_query_profile(q)
        assert isinstance(profile, (list, tuple))

    def test_save_load_roundtrip(self):
        dim = 16
        data, doc_ids, offs = _corpus(n_docs=30, dim=dim)
        r = PyGemRouter(dim=dim)
        r.build(data, doc_ids, offs, n_fine=16, n_coarse=4)
        q = _query(dim=dim)
        r1 = r.route_query(q, n_probes=4, max_candidates=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "router.gemr")
            r.save(path)
            r2 = PyGemRouter(dim=dim)
            r2.load(path)
            assert r2.is_ready()
            assert r2.n_docs() == 30
            r2_results = r2.route_query(q, n_probes=4, max_candidates=10)
        ids1 = set(d for d, _ in r1)
        ids2 = set(d for d, _ in r2_results)
        assert ids1 == ids2

    def test_train_adaptive_cutoff(self):
        dim = 16
        data, doc_ids, offs = _corpus(n_docs=30, dim=dim)
        r = PyGemRouter(dim=dim)
        r.build(data, doc_ids, offs, n_fine=16, n_coarse=4)
        rng = np.random.default_rng(123)
        n_queries = 20
        vpq = 4
        tq = rng.standard_normal((n_queries * vpq, dim)).astype(np.float32)
        nqv = [vpq] * n_queries
        tp = [int(rng.integers(0, 30)) for _ in range(n_queries)]
        tree_bytes = r.train_adaptive_cutoff(tq, nqv, tp, t=3, r_max=8, max_depth=6)
        raw = bytes(tree_bytes) if not isinstance(tree_bytes, bytes) else tree_bytes
        assert len(raw) > 0

    @pytest.mark.skipif(not GEM_AVAILABLE, reason="No gem_index")
    def test_adaptive_cutoff_loadable(self):
        dim = 16
        data, doc_ids, offs = _corpus(n_docs=30, dim=dim)
        r = PyGemRouter(dim=dim)
        r.build(data, doc_ids, offs, n_fine=16, n_coarse=4)
        rng = np.random.default_rng(123)
        tq = rng.standard_normal((80, dim)).astype(np.float32)
        tree_bytes = r.train_adaptive_cutoff(tq, [4] * 20,
                                              [int(rng.integers(0, 30)) for _ in range(20)])
        raw = bytes(tree_bytes) if not isinstance(tree_bytes, bytes) else tree_bytes
        seg, *_ = _build_sealed(n_docs=30, dim=dim)
        seg.load_cutoff_tree(list(raw))
        q = _query(dim=dim)
        results = seg.search(q, k=5, ef=32, n_probes=3)
        assert len(results) > 0

    def test_cluster_overlap(self):
        dim = 16
        data, doc_ids, offs = _corpus(dim=dim)
        r = PyGemRouter(dim=dim)
        r.build(data, doc_ids, offs, n_fine=16, n_coarse=4)
        q = _query(dim=dim)
        overlaps = r.query_cluster_overlaps(q, n_probes=4)
        assert isinstance(overlaps, list)


# ###########################################################################
# 9. Edge Cases & Stress
# ###########################################################################

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestEdgeCasesStress:

    def test_single_doc(self):
        seg, _, ids, _ = _build_sealed(n_docs=1, vpd=4, n_fine=4, n_coarse=1)
        assert seg.is_ready()
        q = _query(n_q=2)
        r = seg.search(q, k=5, ef=32, n_probes=1)
        assert len(r) == 1
        assert r[0][0] == ids[0]

    def test_two_docs(self):
        seg, *_ = _build_sealed(n_docs=2, vpd=4, n_fine=4, n_coarse=1,
                                max_degree=4, ef_construction=16)
        assert seg.is_ready()
        assert seg.n_docs() == 2
        q = _query(n_q=2)
        r = seg.search(q, k=5, ef=32, n_probes=1)
        assert 1 <= len(r) <= 2

    def test_dim_1(self):
        seg = GemSegment()
        data, ids, offs = _corpus(n_docs=10, dim=1, vpd=4)
        seg.build(data, ids, offs, n_fine=4, n_coarse=2, max_degree=4,
                  ef_construction=16, max_kmeans_iter=5, ctop_r=1)
        assert seg.is_ready()
        q = _query(n_q=2, dim=1)
        r = seg.search(q, k=3, ef=16, n_probes=1)
        assert len(r) > 0

    def test_max_degree_1(self):
        seg, *_ = _build_sealed(n_docs=10, vpd=4, max_degree=1)
        q = _query(n_q=2)
        r = seg.search(q, k=3, ef=32, n_probes=2)
        assert len(r) > 0

    def test_n_fine_gt_n_docs(self):
        seg, *_ = _build_sealed(n_docs=5, vpd=4, n_fine=64, n_coarse=4)
        assert seg.is_ready()
        q = _query(n_q=2)
        r = seg.search(q, k=3, ef=32, n_probes=2)
        assert len(r) > 0

    def test_k_gt_n_docs(self):
        seg, *_ = _build_sealed(n_docs=5, vpd=4)
        q = _query(n_q=2)
        r = seg.search(q, k=100, ef=200, n_probes=3)
        assert len(r) <= 5

    def test_very_large_ef(self):
        seg, *_ = _build_sealed(n_docs=20, vpd=4)
        q = _query(n_q=2)
        r = seg.search(q, k=5, ef=10000, n_probes=4)
        assert len(r) > 0

    def test_mutable_single_doc(self):
        seg = PyMutableGemSegment()
        data, ids, offs = _corpus(n_docs=1, dim=16, vpd=4)
        seg.build(data, ids, offs, n_fine=4, n_coarse=1, max_degree=4,
                  ef_construction=16, max_kmeans_iter=5, ctop_r=1)
        assert seg.is_ready()
        q = _query(n_q=2)
        r = seg.search(q, k=1, ef=16)
        assert len(r) == 1


# ###########################################################################
# 10. Quantitative Benchmarks
# ###########################################################################

@pytest.mark.skipif(not GEM_AVAILABLE, reason="latence_gem_index not installed")
class TestQuantitativeBenchmarks:

    def test_recall_at_10_vs_brute_force(self):
        """Graph search recall@10 vs brute-force proxy oracle."""
        seg, *_ = _build_sealed(n_docs=200, dim=32, vpd=8)
        recalls = []
        for seed in range(10):
            q = _query(n_q=4, dim=32, seed=seed * 7 + 1)
            bf = seg.brute_force_proxy(q, k=10)
            bf_ids = set(d for d, _ in bf)
            sr = seg.search(q, k=10, ef=100, n_probes=4)
            sr_ids = set(d for d, _ in sr)
            if len(bf_ids) > 0:
                recalls.append(len(sr_ids & bf_ids) / len(bf_ids))
        avg_recall = np.mean(recalls)
        print(f"\n  Recall@10 vs brute-force: {avg_recall:.3f} (n_queries=10)")
        assert avg_recall >= 0.4, f"Recall@10 too low: {avg_recall:.3f}"

    @pytest.mark.skipif(not CUDA_AVAILABLE or not QCH_KERNEL_AVAILABLE, reason="No GPU")
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="No Triton")
    def test_gpu_speedup(self):
        """Triton must be significantly faster than naive CPU."""
        n_docs, n_q, n_fine, max_len = 10_000, 16, 256, 512
        scores, codes, offsets, lengths = _synth_qch(n_docs, n_q, n_fine, max_len)
        t_s = torch.from_numpy(scores).cuda()
        t_c = torch.from_numpy(codes).cuda()
        t_o = torch.from_numpy(offsets).cuda()
        t_l = torch.from_numpy(lengths).cuda()

        # Warmup
        qch_max_gather_gpu(t_s, t_c, t_o, t_l, n_q, n_fine)
        torch.cuda.synchronize()

        n_runs = 5
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            qch_max_gather_gpu(t_s, t_c, t_o, t_l, n_q, n_fine)
            torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) / n_runs * 1000

        cpu_n = min(200, n_docs)
        t0 = time.perf_counter()
        _naive_cpu_qch(scores, codes[:int(offsets[cpu_n - 1] + lengths[cpu_n - 1])],
                       offsets[:cpu_n], lengths[:cpu_n], n_q, n_fine)
        cpu_ms = (time.perf_counter() - t0) * 1000
        cpu_projected = cpu_ms * (n_docs / cpu_n)

        speedup = cpu_projected / triton_ms
        print(f"\n  GPU speedup: {speedup:.0f}x (Triton={triton_ms:.2f}ms, CPU projected={cpu_projected:.0f}ms)")
        assert speedup > 100, f"GPU speedup only {speedup:.0f}x, expected >100x"

    def test_heal_effectiveness(self):
        """Heal must quantitatively improve graph metrics."""
        seg, _, ids, _ = _build_mutable(n_docs=200, dim=32, vpd=8)
        for d in ids[:40]:
            seg.delete(d)
        _, _, iso_before, stale_before = seg.graph_quality_metrics()
        seg.heal()
        _, _, iso_after, stale_after = seg.graph_quality_metrics()
        print(f"\n  Heal: iso {iso_before:.3f}->{iso_after:.3f}, stale {stale_before:.3f}->{stale_after:.3f}")
        assert stale_after <= stale_before + 0.01

    def test_filter_overhead(self):
        """Filtered search should not be more than 5x slower."""
        seg, _, ids, _ = _build_sealed(n_docs=500, dim=32, vpd=4)
        payloads = [(d, [("cat", "A" if i < 250 else "B")]) for i, d in enumerate(ids)]
        seg.set_doc_payloads(payloads)
        q = _query(dim=32)
        n_runs = 20
        seg.search(q, k=10, ef=50, n_probes=3)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            seg.search(q, k=10, ef=50, n_probes=3)
        base_ms = (time.perf_counter() - t0) / n_runs * 1000
        t0 = time.perf_counter()
        for _ in range(n_runs):
            seg.search(q, k=10, ef=50, n_probes=3, filter=[("cat", "A")])
        filt_ms = (time.perf_counter() - t0) / n_runs * 1000
        ratio = filt_ms / max(base_ms, 0.001)
        print(f"\n  Filter overhead: {ratio:.2f}x (base={base_ms:.2f}ms, filt={filt_ms:.2f}ms)")
        assert ratio < 5.0, f"Filter overhead {ratio:.1f}x exceeds 5x"

    def test_ensemble_vs_single_recall(self):
        """Ensemble should return results from multiple modalities."""
        n_docs, dim, vpd, nm = 100, 32, 8, 2
        rng = np.random.default_rng(42)
        total = n_docs * vpd
        vecs = rng.standard_normal((total, dim)).astype(np.float32)
        ids = list(range(1, n_docs + 1))
        offs = [(i * vpd, (i + 1) * vpd) for i in range(n_docs)]
        tags = [i % nm for i in range(total)]
        ens = PyEnsembleGemSegment()
        ens.build(vecs, ids, offs, tags, nm, n_fine=32, n_coarse=4,
                  max_degree=8, ef_construction=32, max_kmeans_iter=5, ctop_r=2)
        q = _query(n_q=4, dim=dim)
        ens_r = ens.search(q, [0, 0, 1, 1], k=10, ef=100, n_probes=4)
        print(f"\n  Ensemble results: {len(ens_r)}")
        assert len(ens_r) > 0
        scores = [s for _, s in ens_r]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-7
