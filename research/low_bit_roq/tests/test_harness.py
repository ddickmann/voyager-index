"""
Sanity tests for the Phase-0 harness — uses a fake SearchRunner so it is
fully CPU-only and fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.low_bit_roq import harness, progress_md


def _make_runner_factory(seed_jitter: float = 0.0):
    rng = np.random.default_rng(0)

    def factory(dataset: str, seed: int):
        query_ids = [f"{dataset}-q{i}" for i in range(20)]
        relevance: dict[str, list[int]] = {qid: [i] for i, qid in enumerate(query_ids)}

        def runner(qid: str, _seed: int):
            rel = relevance[qid]
            retrieved = list(np.random.default_rng(seed + hash(qid) % 1024).integers(0, 1000, size=10))
            if rng.random() > 0.3:
                retrieved[0] = rel[0]
            candidate_ids = {
                k: list(set(retrieved) | set(rng.integers(0, 1000, size=k).tolist()))
                for k in (500, 1000, 2000, 4000)
            }
            return harness.QueryResult(
                query_id=qid,
                retrieved_ids=retrieved,
                candidate_ids_per_k=candidate_ids,
                relevant=rel,
                latency_ms=5.0 + rng.normal() + seed_jitter,
                n_score_evals=128,
                n_ann_probes=4,
                n_roq_kernel_calls=1,
                bytes_fetched=12_345,
            )

        return runner, query_ids

    return factory


def test_run_sweep_aggregates_per_dataset_and_macro():
    cfg = harness.HarnessConfig(
        experiment_id="harness-smoke",
        summary="harness smoke test",
        datasets=("ArguAna",),
        seeds=(0, 1),
        cold_warm=False,
        drop_caches_between_seeds=False,
    )
    agg = harness.run_sweep(cfg, _make_runner_factory())
    assert agg["id"] == "harness-smoke"
    assert "ArguAna" in agg["per_dataset"]
    assert agg["macro"]["recall_at_10"] >= 0
    assert agg["macro"]["warm_p95_ms"] > 0


def test_paired_bootstrap_pvalue_is_low_for_clear_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.05, size=200)
    treat = base + 0.1
    p, ci = harness.paired_bootstrap_pvalue(base.tolist(), treat.tolist())
    assert p < 0.01
    assert ci[0] > 0


def test_paired_bootstrap_pvalue_high_for_no_difference():
    rng = np.random.default_rng(1)
    base = rng.normal(0.5, 0.05, size=200)
    treat = base + rng.normal(0, 1e-4, size=200)
    p, _ = harness.paired_bootstrap_pvalue(base.tolist(), treat.tolist())
    assert p > 0.05


def test_emit_progress_stub_writes_report_and_appends_md(tmp_path: Path):
    cfg = harness.HarnessConfig(
        experiment_id="harness-report",
        summary="emits stub",
        datasets=("ArguAna",),
        seeds=(0,),
        cold_warm=False,
        drop_caches_between_seeds=False,
    )
    agg = harness.run_sweep(cfg, _make_runner_factory())

    progress_file = tmp_path / "PROGRESS.md"
    progress_file.write_text(
        "# PROGRESS\n\n## Current State\n- **Phase:** test\n\n## Promoted\n_(empty)_\n\n## Killed\n_(empty)_\n\n## Open `[VERDICT-PENDING]` entries\n_(empty)_\n\n---\n",
        encoding="utf-8",
    )
    report_path = harness.emit_progress_stub(
        agg,
        progress_path=progress_file,
        reports_dir=tmp_path / "reports",
    )
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["id"] == "harness-report"
    md = progress_file.read_text(encoding="utf-8")
    assert "harness-report" in md


def test_recall_and_ndcg_at_k_correct():
    retrieved = [10, 20, 30, 40, 50]
    relevant = [40, 50]
    assert harness.recall_at_k(retrieved, set(relevant), 5) == pytest.approx(1.0)
    assert harness.recall_at_k(retrieved, set(relevant), 2) == pytest.approx(0.0)
    assert harness.ndcg_at_k(retrieved, relevant, 5) > 0
