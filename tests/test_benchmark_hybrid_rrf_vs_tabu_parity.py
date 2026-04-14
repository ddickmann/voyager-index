"""Parity checks for tools/benchmarks/benchmark_hybrid_rrf_vs_tabu.py (no dataset download)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_BENCH = _REPO / "tools" / "benchmarks" / "benchmark_hybrid_rrf_vs_tabu.py"
if _BENCH.is_file():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_rrf_tabu", _BENCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_rrf_tabu"] = mod
    spec.loader.exec_module(mod)
else:  # pragma: no cover
    mod = None


@pytest.mark.skipif(mod is None, reason="benchmark script missing")
def test_rrf_fusion_matches_rrf_ranked_ids() -> None:
    """fuse_results_rrf must match the legacy reciprocal-rank sum ordering."""
    dense = [(10, 0.9), (20, 0.8), (30, 0.7)]
    sparse = [(20, 0.55), (40, 0.4)]
    fused = mod.fuse_results_rrf(dense, sparse, rrf_k=60, top_k=500)
    legacy = mod.rrf_ranked_ids(dense, sparse, rrf_k=60.0)
    assert fused[:20] == legacy[:20]


@pytest.mark.skipif(mod is None, reason="benchmark script missing")
def test_ranking_metrics_ndcg_perfect_order() -> None:
    rel = {1: 1.0, 2: 1.0}
    m = mod._ranking_metrics([1, 2, 99], rel, k=3)
    assert m["ndcg"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx(1.0)
    assert m["support_coverage"] == pytest.approx(1.0)
    assert m["answer_utility"] == pytest.approx(1.0)
    assert m["hit"] == pytest.approx(1.0)


@pytest.mark.skipif(mod is None, reason="benchmark script missing")
def test_tabu_ranking_smoke() -> None:
    pytest.importorskip("latence_solver")
    assert mod is not None
    cand = [0, 1, 2]
    emb = np.eye(3, 8, dtype=np.float32)
    texts = ["a b c", "d e f", "g h i"]
    q = np.ones(8, dtype=np.float32)
    dmap = {0: 0.9, 1: 0.5, 2: 0.1}
    bmap = {0: 0.2, 1: 0.8, 2: 0.3}
    dense_l = [(0, 0.9), (1, 0.5), (2, 0.1)]
    sparse_l = [(1, 0.8), (0, 0.2), (2, 0.3)]
    from latence_solver import SolverConfig, SolverConstraints

    out = mod.tabu_ranking(
        cand,
        emb,
        texts,
        q,
        dmap,
        bmap,
        dense_l,
        sparse_l,
        rrf_k=60.0,
        solver_config=SolverConfig(iterations=15, random_seed=3, use_gpu=False),
        constraints=SolverConstraints(max_tokens=2000, max_chunks=3, max_per_cluster=2),
    )
    assert len(out) == 3
    assert set(out) == {0, 1, 2}


@pytest.mark.skipif(mod is None, reason="benchmark script missing")
def test_solver_max_chunks_auto_large_pool() -> None:
    assert mod._solver_max_chunks_for_pool(200, large_chunk_pool=True, explicit_cap=None, eval_k=10) == 64
    assert mod._solver_max_chunks_for_pool(40, large_chunk_pool=True, explicit_cap=None, eval_k=10) == 20
    assert mod._solver_max_chunks_for_pool(100, large_chunk_pool=False, explicit_cap=None, eval_k=10) == 6
    assert mod._solver_max_chunks_for_pool(100, large_chunk_pool=False, explicit_cap=12, eval_k=10) == 12


@pytest.mark.skipif(mod is None, reason="benchmark script missing")
def test_tabu_ce_lambda0_uses_precomputed_scores_and_keeps_ce_order(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyOutput:
        selected_indices = [2, 0]

    class DummySolver:
        def __init__(self, config) -> None:
            self.config = config

        def solve_precomputed_numpy(self, *args, **kwargs):
            return DummyOutput()

    fake_module = types.SimpleNamespace(TabuSearchSolver=DummySolver)
    monkeypatch.setitem(sys.modules, "latence_solver", fake_module)

    cand = [0, 1, 2]
    emb = np.eye(3, 4, dtype=np.float32)
    texts = ["a", "b", "c"]
    q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dense_l = [(0, 0.6), (1, 0.5), (2, 0.4)]
    sparse_l = [(2, 0.7), (0, 0.3), (1, 0.2)]
    ce_scores = {0: 0.95, 1: 0.10, 2: 0.80}

    out = mod.tabu_ranking(
        cand,
        emb,
        texts,
        q,
        {0: 0.6, 1: 0.5, 2: 0.4},
        {0: 0.3, 1: 0.2, 2: 0.7},
        dense_l,
        sparse_l,
        rrf_k=60.0,
        solver_config=types.SimpleNamespace(lambda_=0.0),
        constraints=object(),
        relevance_source="ce",
        ce_scores=ce_scores,
        tail_order="ce",
    )

    assert out == [0, 2, 1]


@pytest.mark.skipif(mod is None, reason="benchmark script missing")
def test_tabu_ranking_details_include_coverage_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyOutput:
        selected_indices = [1, 0]
        objective_score = 1.25
        fulfilment_total = 0.75
        redundancy_penalty = 0.10
        total_tokens = 96
        num_selected = 2
        solve_time_ms = 0.05
        constraints_satisfied = True

    class DummySolver:
        def __init__(self, config) -> None:
            self.config = config

        def solve_precomputed_numpy(self, *args, **kwargs):
            captured["coverage_matrix"] = kwargs.get("coverage_matrix")
            captured["query_token_weights"] = kwargs.get("query_token_weights")
            return DummyOutput()

    fake_module = types.SimpleNamespace(TabuSearchSolver=DummySolver)
    monkeypatch.setitem(sys.modules, "latence_solver", fake_module)

    details = mod.tabu_ranking(
        [0, 1, 2],
        np.eye(3, 4, dtype=np.float32),
        ["invoice total due", "invoice duplicate total", "board report"],
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        {0: 0.9, 1: 0.8, 2: 0.1},
        {0: 0.3, 1: 0.2, 2: 0.1},
        [(0, 0.9), (1, 0.8), (2, 0.1)],
        [(1, 0.4), (0, 0.3), (2, 0.1)],
        rrf_k=60.0,
        solver_config=types.SimpleNamespace(lambda_=0.3),
        constraints=object(),
        query_text="invoice total due",
        return_details=True,
    )

    assert isinstance(captured["coverage_matrix"], np.ndarray)
    assert captured["coverage_matrix"].shape[0] >= 1
    assert isinstance(captured["query_token_weights"], np.ndarray)
    assert float(np.sum(captured["query_token_weights"])) == pytest.approx(1.0)
    assert details["ranked_ids"][:2] == [1, 0]
    assert details["solver_output"]["fulfilment_total"] == pytest.approx(0.75)
    assert "uncovered_mass" in details["diagnostics"]
    assert "semantic_query" in details["query_aspects"]
