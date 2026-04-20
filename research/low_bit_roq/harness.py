"""
Phase 0 harness extensions.

This module is intentionally a *thin* wrapper on top of
``benchmarks/beir_benchmark.py`` and the existing
``benchmarks/shard_bench/metrics.py`` — the plan explicitly says
"glue, don't rewrite". It adds the four pieces the existing runner is
missing:

1. Multi-seed sweep + paired bootstrap aggregator.
2. Candidate-recall logging at k_candidates ∈ {500, 1000, 2000, 4000} pulled
   from the engine's existing ``inspect_query_pipeline()``.
3. Cold-cache vs warm-cache p95 split (drops OS page cache between runs).
4. Compute accounting: #score-evals/q, ANN probes/q, ROQ kernel calls/q,
   bytes-fetched/q.

Plus the small glue that emits the JSON report and appends a stub entry to
PROGRESS.md.

This module has zero hard dependencies on a particular quantizer / router /
dataset — it consumes per-query output of an arbitrary "search runner"
callable. That keeps the same harness usable for A1 cells, A6 gate runs,
B-phase runs, and the final C-phase replay.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import progress_md

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Per-query output produced by a SearchRunner.

    The harness only needs the four fields below to compute every metric in
    the gate; everything else lives in ``extras`` for ad-hoc plots.
    """

    query_id: str
    retrieved_ids: Sequence[int]
    candidate_ids_per_k: Mapping[int, Sequence[int]]
    relevant: Sequence[int] | Mapping[int, int]
    latency_ms: float
    n_score_evals: int = 0
    n_ann_probes: int = 0
    n_roq_kernel_calls: int = 0
    bytes_fetched: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


SearchRunner = Callable[[str, int], QueryResult]
"""Signature: ``(query_id, seed) -> QueryResult``.

The runner is given a query_id (from the dataset) and a seed (so that any
randomized index initialization can be re-rolled per seed). The runner is
fully responsible for calling the engine through the LEMUR -> ANN ->
router -> ROQ rerank lane and returning per-query timings + candidate IDs.
"""


@dataclass
class DatasetSeedResult:
    dataset: str
    seed: int
    per_query: list[QueryResult]
    cold_p95_ms: float
    warm_p95_ms: float
    qps: float
    vram_mb: float | None = None
    disk_bytes: int | None = None


# ---------------------------------------------------------------------------
# Cold-cache helper
# ---------------------------------------------------------------------------


def drop_os_page_cache() -> None:
    """Best-effort cold-cache: ``sync && echo 3 > /proc/sys/vm/drop_caches``.

    Silently no-ops if we lack permission, so this works on dev laptops and
    on the fixed benchmark box.
    """
    try:
        subprocess.run(["sync"], check=False)
        with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as f:
            f.write("3\n")
    except (PermissionError, OSError, FileNotFoundError):
        pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _to_relevant_set(relevant: Sequence[int] | Mapping[int, int]) -> set[int]:
    if isinstance(relevant, dict):
        return {doc_id for doc_id, grade in relevant.items() if grade > 0}
    return set(relevant)


def recall_at_k(retrieved: Sequence[int], relevant_set: set[int], k: int) -> float:
    if not relevant_set:
        return 1.0
    hit = len(set(retrieved[:k]) & relevant_set)
    return hit / min(k, len(relevant_set))


def ndcg_at_k(
    retrieved: Sequence[int],
    relevant: Sequence[int] | Mapping[int, int],
    k: int,
) -> float:
    grades = relevant if isinstance(relevant, dict) else {d: 1 for d in relevant}
    if not grades:
        return 0.0
    gains = []
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = grades.get(int(doc_id), 0)
        if rel > 0:
            gains.append((2 ** rel - 1) / np.log2(rank + 1))
    dcg = float(np.sum(gains)) if gains else 0.0
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = float(np.sum([(2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal_grades)]))
    return dcg / idcg if idcg > 0 else 0.0


def candidate_recall_at_k(
    candidate_ids: Sequence[int], relevant_set: set[int], k: int, eval_k: int = 10
) -> float:
    """Fraction of the top-``eval_k`` ground-truth docs that survived into the
    candidate shortlist of size ``k``. This is the diagnostic that tells you
    whether the failure is at routing time or at rerank time.
    """
    if not relevant_set:
        return 1.0
    shortlist = set(candidate_ids[:k])
    target = min(eval_k, len(relevant_set))
    if target == 0:
        return 1.0
    return len(shortlist & relevant_set) / target


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap_pvalue(
    baseline_per_query: Sequence[float],
    treatment_per_query: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    two_sided: bool = True,
) -> tuple[float, tuple[float, float]]:
    """Returns (p_value, ci95) for the difference treatment - baseline,
    paired by query index. Resampling is over query IDs.
    """
    if len(baseline_per_query) != len(treatment_per_query):
        raise ValueError("baseline / treatment lengths must match (paired)")
    deltas = np.asarray(treatment_per_query, dtype=np.float64) - np.asarray(
        baseline_per_query, dtype=np.float64
    )
    n = deltas.size
    if n == 0:
        return 1.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = deltas[idx].mean(axis=1)
    observed = float(deltas.mean())

    centered = resampled_means - resampled_means.mean()
    if two_sided:
        p = float(np.mean(np.abs(centered) >= abs(observed)))
    else:
        p = float(np.mean(centered <= -abs(observed)))
    ci_lo, ci_hi = np.quantile(resampled_means, [0.025, 0.975])
    return p, (float(ci_lo), float(ci_hi))


# ---------------------------------------------------------------------------
# Compute accounting
# ---------------------------------------------------------------------------


@dataclass
class ComputeAccounting:
    score_evals_per_q_mean: float
    ann_probes_per_q_mean: float
    roq_kernel_calls_per_q_mean: float
    bytes_fetched_per_q_mean: float

    @classmethod
    def from_query_results(cls, results: Sequence[QueryResult]) -> "ComputeAccounting":
        if not results:
            return cls(0.0, 0.0, 0.0, 0.0)
        return cls(
            score_evals_per_q_mean=float(np.mean([r.n_score_evals for r in results])),
            ann_probes_per_q_mean=float(np.mean([r.n_ann_probes for r in results])),
            roq_kernel_calls_per_q_mean=float(
                np.mean([r.n_roq_kernel_calls for r in results])
            ),
            bytes_fetched_per_q_mean=float(np.mean([r.bytes_fetched for r in results])),
        )


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    experiment_id: str
    summary: str
    datasets: Sequence[str]
    seeds: Sequence[int]
    k_candidates_grid: Sequence[int] = (500, 1000, 2000, 4000)
    eval_k: int = 10
    bootstrap_resamples: int = 10_000
    cold_warm: bool = True
    drop_caches_between_seeds: bool = True
    config_snapshot: Mapping[str, Any] = field(default_factory=dict)
    baseline_name: str = "fp16"
    gate: str = "informs nothing"


def run_sweep(
    cfg: HarnessConfig,
    runner_factory: Callable[[str, int], tuple[SearchRunner, Sequence[str]]],
) -> dict[str, Any]:
    """Run the multi-seed, multi-dataset sweep.

    ``runner_factory(dataset, seed) -> (runner, query_ids)`` is called once
    per (dataset, seed). It returns a SearchRunner closure (which the
    harness then calls per-query) and the list of query_ids to iterate.
    The factory is also responsible for index build / load + warm-up; the
    harness only times the actual queries.
    """
    per_dataset_results: dict[str, list[DatasetSeedResult]] = {}

    for dataset in cfg.datasets:
        per_dataset_results[dataset] = []
        for seed in cfg.seeds:
            log.info("Running %s seed=%d", dataset, seed)
            if cfg.drop_caches_between_seeds:
                drop_os_page_cache()
                gc.collect()

            runner, query_ids = runner_factory(dataset, seed)

            cold_latencies: list[float] = []
            warm_latencies: list[float] = []
            per_query: list[QueryResult] = []

            cold_n = max(1, len(query_ids) // 10) if cfg.cold_warm else 0
            for i, qid in enumerate(query_ids):
                t0 = time.perf_counter()
                qr = runner(qid, seed)
                t1 = time.perf_counter()
                wall_ms = (t1 - t0) * 1000.0
                if qr.latency_ms <= 0.0:
                    qr.latency_ms = wall_ms
                per_query.append(qr)
                if i < cold_n:
                    cold_latencies.append(qr.latency_ms)
                else:
                    warm_latencies.append(qr.latency_ms)

            cold_p95 = float(np.percentile(cold_latencies, 95)) if cold_latencies else float("nan")
            warm_p95 = (
                float(np.percentile(warm_latencies, 95)) if warm_latencies else cold_p95
            )

            total_s = sum(r.latency_ms for r in per_query) / 1000.0
            qps = len(per_query) / total_s if total_s > 0 else 0.0

            per_dataset_results[dataset].append(
                DatasetSeedResult(
                    dataset=dataset,
                    seed=seed,
                    per_query=per_query,
                    cold_p95_ms=cold_p95,
                    warm_p95_ms=warm_p95,
                    qps=qps,
                )
            )

    return _aggregate(cfg, per_dataset_results)


# ---------------------------------------------------------------------------
# Aggregation + report writing
# ---------------------------------------------------------------------------


def _aggregate(
    cfg: HarnessConfig,
    per_dataset_results: dict[str, list[DatasetSeedResult]],
) -> dict[str, Any]:
    """Macro-averaged + per-dataset summary that gets dumped to JSON."""
    per_dataset_summary: dict[str, Any] = {}
    macro_recall10 = []
    macro_ndcg10 = []
    macro_warm_p95 = []
    macro_cold_p95 = []

    for dataset, runs in per_dataset_results.items():
        dataset_recall10 = []
        dataset_ndcg10 = []
        dataset_warm_p95 = []
        dataset_cold_p95 = []
        candidate_recall_per_k: dict[int, list[float]] = {
            k: [] for k in cfg.k_candidates_grid
        }
        compute_acc: list[ComputeAccounting] = []

        for run in runs:
            r10s = []
            n10s = []
            for qr in run.per_query:
                rel_set = _to_relevant_set(qr.relevant)
                r10s.append(recall_at_k(qr.retrieved_ids, rel_set, cfg.eval_k))
                n10s.append(ndcg_at_k(qr.retrieved_ids, qr.relevant, cfg.eval_k))
                for k in cfg.k_candidates_grid:
                    candidate_ids = qr.candidate_ids_per_k.get(k, qr.retrieved_ids)
                    candidate_recall_per_k[k].append(
                        candidate_recall_at_k(candidate_ids, rel_set, k, eval_k=cfg.eval_k)
                    )
            dataset_recall10.append(float(np.mean(r10s)))
            dataset_ndcg10.append(float(np.mean(n10s)))
            dataset_warm_p95.append(run.warm_p95_ms)
            dataset_cold_p95.append(run.cold_p95_ms)
            compute_acc.append(ComputeAccounting.from_query_results(run.per_query))

        per_dataset_summary[dataset] = {
            "recall_at_10_mean": float(np.mean(dataset_recall10)),
            "recall_at_10_std": float(np.std(dataset_recall10, ddof=1) if len(dataset_recall10) > 1 else 0.0),
            "ndcg_at_10_mean": float(np.mean(dataset_ndcg10)),
            "ndcg_at_10_std": float(np.std(dataset_ndcg10, ddof=1) if len(dataset_ndcg10) > 1 else 0.0),
            "warm_p95_ms_mean": float(np.mean(dataset_warm_p95)),
            "cold_p95_ms_mean": float(np.mean(dataset_cold_p95)),
            "candidate_recall_per_k": {
                str(k): float(np.mean(v)) if v else 0.0
                for k, v in candidate_recall_per_k.items()
            },
            "compute_accounting": {
                "score_evals_per_q": float(np.mean([c.score_evals_per_q_mean for c in compute_acc])),
                "ann_probes_per_q": float(np.mean([c.ann_probes_per_q_mean for c in compute_acc])),
                "roq_kernel_calls_per_q": float(
                    np.mean([c.roq_kernel_calls_per_q_mean for c in compute_acc])
                ),
                "bytes_fetched_per_q": float(np.mean([c.bytes_fetched_per_q_mean for c in compute_acc])),
            },
        }
        macro_recall10.append(per_dataset_summary[dataset]["recall_at_10_mean"])
        macro_ndcg10.append(per_dataset_summary[dataset]["ndcg_at_10_mean"])
        macro_warm_p95.append(per_dataset_summary[dataset]["warm_p95_ms_mean"])
        macro_cold_p95.append(per_dataset_summary[dataset]["cold_p95_ms_mean"])

    macro_summary = {
        "recall_at_10": float(np.mean(macro_recall10)),
        "ndcg_at_10": float(np.mean(macro_ndcg10)),
        "warm_p95_ms": float(np.mean(macro_warm_p95)),
        "cold_p95_ms": float(np.mean(macro_cold_p95)),
    }

    return {
        "id": cfg.experiment_id,
        "summary": cfg.summary,
        "config": dict(cfg.config_snapshot),
        "datasets": list(cfg.datasets),
        "seeds": len(list(cfg.seeds)),
        "baseline": cfg.baseline_name,
        "macro": macro_summary,
        "per_dataset": per_dataset_summary,
        "env": _capture_env(),
        "gate": cfg.gate,
    }


def _capture_env() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def write_report(
    aggregated: dict[str, Any],
    *,
    reports_dir: Path | str = progress_md.REPORTS_DIR_DEFAULT,
) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{aggregated['id']}.json"
    path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    return path


def _baseline_compare_metric_rows(
    aggregated: dict[str, Any], baseline: dict[str, Any]
) -> list[progress_md.MetricRow]:
    rows: list[progress_md.MetricRow] = []
    a_macro = aggregated["macro"]
    b_macro = baseline["macro"]

    base_per_query = baseline.get("per_query_recall_at_10", [])
    treat_per_query = aggregated.get("per_query_recall_at_10", [])
    p_recall = ""
    if base_per_query and treat_per_query and len(base_per_query) == len(treat_per_query):
        p, _ = paired_bootstrap_pvalue(base_per_query, treat_per_query)
        p_recall = p

    rows.append(
        progress_md.MetricRow(
            name="Recall@10 (rerank)",
            baseline=b_macro["recall_at_10"],
            this=a_macro["recall_at_10"],
            delta=a_macro["recall_at_10"] - b_macro["recall_at_10"],
            p_value=p_recall,
        )
    )
    rows.append(
        progress_md.MetricRow(
            name="NDCG@10  (rerank)",
            baseline=b_macro["ndcg_at_10"],
            this=a_macro["ndcg_at_10"],
            delta=a_macro["ndcg_at_10"] - b_macro["ndcg_at_10"],
        )
    )
    if b_macro.get("warm_p95_ms"):
        delta_pct = (a_macro["warm_p95_ms"] - b_macro["warm_p95_ms"]) / b_macro["warm_p95_ms"]
        rows.append(
            progress_md.MetricRow(
                name="end-to-end p95",
                baseline=f"{b_macro['warm_p95_ms']:.0f}ms",
                this=f"{a_macro['warm_p95_ms']:.0f}ms",
                delta=progress_md.MetricRow.fmt(delta_pct, kind="pct"),
            )
        )
    return rows


def emit_progress_stub(
    aggregated: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    progress_path: Path | str = progress_md.PROGRESS_PATH_DEFAULT,
    reports_dir: Path | str = progress_md.REPORTS_DIR_DEFAULT,
) -> Path:
    """Write the JSON report and append the auto-stub PROGRESS.md entry."""
    report_path = write_report(aggregated, reports_dir=reports_dir)
    metrics: list[progress_md.MetricRow] = []
    if baseline is not None:
        metrics = _baseline_compare_metric_rows(aggregated, baseline)
    else:
        macro = aggregated["macro"]
        metrics = [
            progress_md.MetricRow(name="Recall@10", baseline="—", this=macro["recall_at_10"]),
            progress_md.MetricRow(name="NDCG@10", baseline="—", this=macro["ndcg_at_10"]),
            progress_md.MetricRow(name="warm p95", baseline="—", this=f"{macro['warm_p95_ms']:.0f}ms"),
            progress_md.MetricRow(name="cold p95", baseline="—", this=f"{macro['cold_p95_ms']:.0f}ms"),
        ]
    stub = progress_md.StubEntry(
        experiment_id=aggregated["id"],
        summary=aggregated["summary"],
        config=aggregated["config"],
        datasets=aggregated["datasets"],
        seeds=aggregated["seeds"],
        baseline_name=aggregated["baseline"],
        metrics=metrics,
        artifacts=[report_path],
        gate=aggregated["gate"],
    )
    progress_md.append_stub(stub, progress_path=progress_path)
    return report_path
