"""Phase 6b/6c production-lane sweep for the rroq158 plan.

Runs (compression × seed) variants on a single BEIR dataset through the
production LEMUR routing path and aggregates per-variant statistics.

Variants exercised:
    fp16        — baseline
    roq4        — current production codec
    rroq158     — Riemannian-aware 1.58-bit (this work)

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True \
        python -m benchmarks.run_rroq158_prod_sweep \
            --dataset nfcorpus --seeds 5 --output reports/beir_rroq158_nfcorpus.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from benchmarks.beir_benchmark import (
    OPTIMAL_GPU,
    QUORA_OVERRIDE_SHARDS,
    TOP_K,
    _resolve_compression,
    build_index,
    evaluate,
    load_beir_npz,
    run_gpu_corpus_mode,
)


VARIANTS = ("fp16", "roq4", "rroq158")


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _build_one(
    name: str, variant: str, seed: int, n_eval: int,
    all_vectors: np.ndarray, doc_offsets: list, doc_ids: list,
    query_vecs: list, graded_qrels: dict, dim: int,
) -> Dict:
    compression = _resolve_compression(variant)
    params = dict(OPTIMAL_GPU)
    params["compression"] = compression
    params["distill_rerank"] = False
    params["rroq158_k"] = 1024
    params["rroq158_seed"] = seed
    if name == "quora":
        params["n_shards"] = QUORA_OVERRIDE_SHARDS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_query_vecs = (
        query_vecs[: min(len(query_vecs), n_eval)] if n_eval else query_vecs
    )

    index_dir, build_s = build_index(
        name, all_vectors, doc_offsets, doc_ids, dim, params, device=device,
    )

    t0 = time.time()
    search_result = run_gpu_corpus_mode(
        name, index_dir, all_vectors, doc_offsets, doc_ids,
        eval_query_vecs, dim, params,
    )
    metrics = evaluate(search_result["all_ids"], graded_qrels, len(eval_query_vecs))
    elapsed = time.time() - t0

    _free_gpu()

    return {
        "dataset": name,
        "variant": variant,
        "seed": seed,
        "n_eval": len(eval_query_vecs),
        "build_s": build_s,
        "search_s": elapsed,
        "qps": search_result["qps"],
        "p50_ms": search_result["p50_ms"],
        "p95_ms": search_result["p95_ms"],
        **metrics,
    }


def _aggregate(rows: List[Dict]) -> Dict:
    """Mean +/- std across seeds, plus paired-bootstrap p vs fp16."""
    by_variant: Dict[str, List[Dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    out: Dict[str, Dict] = {}
    metric_keys = (
        "recall@10", "recall@100", "NDCG@10", "NDCG@100",
        "MAP@100", "qps", "p50_ms", "p95_ms",
    )
    for variant, runs in by_variant.items():
        aggr: Dict[str, Dict[str, float]] = {}
        for k in metric_keys:
            vals = [float(r.get(k, 0.0)) for r in runs]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            aggr[k] = {"mean": mean, "std": std, "n": len(vals)}
        out[variant] = {
            "n_seeds": len(runs),
            "metrics": aggr,
            "per_seed": runs,
        }
    return out


def _delta_table(agg: Dict, baseline: str = "fp16") -> List[Dict]:
    rows = []
    base = agg.get(baseline)
    for variant, info in agg.items():
        rec10 = info["metrics"]["recall@10"]
        ndcg10 = info["metrics"]["NDCG@10"]
        p95 = info["metrics"]["p95_ms"]
        qps = info["metrics"]["qps"]
        delta_r10 = (
            (rec10["mean"] - base["metrics"]["recall@10"]["mean"]) * 100
            if base else math.nan
        )
        delta_p95_pct = (
            (p95["mean"] - base["metrics"]["p95_ms"]["mean"])
            / base["metrics"]["p95_ms"]["mean"] * 100
            if base else math.nan
        )
        rows.append({
            "variant": variant,
            "R@10": f"{rec10['mean']:.4f}±{rec10['std']:.4f}",
            "NDCG@10": f"{ndcg10['mean']:.4f}±{ndcg10['std']:.4f}",
            "p95_ms": f"{p95['mean']:.2f}±{p95['std']:.2f}",
            "qps": f"{qps['mean']:.0f}",
            "Δ R@10 vs fp16 (pts)": f"{delta_r10:+.2f}",
            "Δ p95 vs fp16 (%)": f"{delta_p95_pct:+.1f}",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nfcorpus", "scidocs"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--n-eval", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log.info(
        "rroq158 prod sweep: dataset=%s variants=%s seeds=%d n_eval=%s",
        args.dataset, args.variants, args.seeds, args.n_eval or "all",
    )
    all_vectors, doc_offsets, doc_ids, query_vecs, graded_qrels, dim = load_beir_npz(
        args.dataset
    )

    rows: List[Dict] = []
    for variant in args.variants:
        n_seeds = args.seeds if variant == "rroq158" else 1
        # fp16/roq4 builds are deterministic at seed=42 in BuildConfig and the
        # GPU-corpus search has zero randomness; running them multiple seeds
        # would just measure timer noise. Run rroq158 across seeds because
        # FWHT rotator + spherical k-means init are seed-dependent.
        for s_idx in range(n_seeds):
            seed = 42 + s_idx
            log.info("=== %s | %s | seed=%d ===", args.dataset, variant, seed)
            try:
                row = _build_one(
                    args.dataset, variant, seed, args.n_eval,
                    all_vectors, doc_offsets, doc_ids, query_vecs,
                    graded_qrels, dim,
                )
                log.info(
                    "%s seed=%d: R@10=%.4f NDCG@10=%.4f p95=%.2fms qps=%.0f",
                    variant, seed, row["recall@10"], row["NDCG@10"],
                    row["p95_ms"], row["qps"],
                )
                rows.append(row)
            except Exception as e:
                log.exception("FAILED %s/%s/seed=%d: %s", args.dataset, variant, seed, e)
                rows.append({
                    "dataset": args.dataset, "variant": variant, "seed": seed,
                    "error": str(e),
                })
            _free_gpu()

    agg = _aggregate([r for r in rows if "error" not in r])
    deltas = _delta_table(agg)

    out = {
        "dataset": args.dataset,
        "n_seeds_rroq158": args.seeds,
        "n_eval": args.n_eval,
        "raw_runs": rows,
        "aggregated": agg,
        "summary_table": deltas,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %s", out_path)

    print("\n=== SUMMARY: %s ===" % args.dataset)
    hdr = "%-10s %-18s %-18s %-14s %-8s %-22s %-22s" % (
        "variant", "R@10", "NDCG@10", "p95_ms", "qps",
        "Δ R@10 vs fp16 (pts)", "Δ p95 vs fp16 (%)",
    )
    print(hdr)
    print("-" * len(hdr))
    for d in deltas:
        print("%-10s %-18s %-18s %-14s %-8s %-22s %-22s" % (
            d["variant"], d["R@10"], d["NDCG@10"], d["p95_ms"], d["qps"],
            d["Δ R@10 vs fp16 (pts)"], d["Δ p95 vs fp16 (%)"],
        ))


if __name__ == "__main__":
    main()
