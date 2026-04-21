#!/usr/bin/env python3
"""Quick k_candidates sweep on 7.5k dataset — build once, sweep k values."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from colsearch._internal.inference.shard_engine.builder import DEFAULT_NPZ, _index_dir, build, load_corpus
from colsearch._internal.inference.shard_engine.colbandit_reranker import ColBanditConfig as _CBConfig, ColBanditReranker
from colsearch._internal.inference.shard_engine.config import (
    BenchmarkConfig,
    BuildConfig,
    Compression,
    RouterType,
    SearchConfig,
    StorageLayout,
    TransferMode,
)
from colsearch._internal.inference.shard_engine.fetch_pipeline import FetchPipeline, PinnedBufferPool
from colsearch._internal.inference.shard_engine.lemur_router import LemurRouter
from colsearch._internal.inference.shard_engine.scorer import brute_force_maxsim, score_all_docs_topk, warmup_maxsim
from benchmarks.shard_bench.metrics import compute_all_metrics
from colsearch._internal.inference.shard_engine.profiler import QueryProfile, aggregate_profiles
from benchmarks.shard_bench.run_benchmark import compute_ground_truth, search_shard_routed
from colsearch._internal.inference.shard_engine.shard_store import ShardStore

CORPUS_SIZE = 7500
N_EVAL = 100
K_VALUES = [100, 200, 300, 500, 750, 1000, 1500, 2000]
DEVICE = "cuda"


def main():
    npz_path = Path(DEFAULT_NPZ)
    all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim = load_corpus(npz_path, max_docs=CORPUS_SIZE)
    doc_vecs = [all_vectors[s:e] for s, e in doc_offsets]
    n_eval = min(N_EVAL, len(query_vecs))

    gt_cache = Path.home() / ".cache" / "shard-bench" / f"gt_{CORPUS_SIZE}.npz"
    ground_truth = compute_ground_truth(
        query_vecs, doc_vecs, doc_ids, dim, n_eval, k=100,
        cache_path=gt_cache, device=DEVICE,
    )

    # Build index once
    bcfg = BuildConfig(
        corpus_size=CORPUS_SIZE,
        compression=Compression.FP16,
        layout=StorageLayout.PROXY_GROUPED,
        router_type=RouterType.LEMUR,
    )
    bcfg.lemur.enabled = True
    bcfg.lemur.device = DEVICE
    build(bcfg, npz_path=npz_path, device=DEVICE)

    index_dir = _index_dir(bcfg)
    store = ShardStore(index_dir)
    router = LemurRouter(
        index_dir / "lemur",
        ann_backend=bcfg.lemur.ann_backend.value,
        device=bcfg.lemur.device,
    )
    router.load()

    pool = PinnedBufferPool(max_tokens=50_000, dim=dim, n_buffers=3)
    pipeline = FetchPipeline(store=store, mode=TransferMode.PINNED, pinned_pool=pool, device=DEVICE)

    # Warmup
    representative_token_counts = [
        int(s.p95_tokens) for s in store.manifest.shards if s.p95_tokens > 0
    ]
    if not representative_token_counts:
        representative_token_counts = [128, 256]
    warmup_maxsim(dim=dim, doc_token_counts=representative_token_counts, device=DEVICE)
    print("Kernel warmup done")

    results = []
    for k_cand in K_VALUES:
        scfg = SearchConfig(
            max_docs_exact=min(k_cand, CORPUS_SIZE),
            transfer_mode=TransferMode.PINNED,
            k_candidates=k_cand,
        )

        profiles = []
        for qi in range(n_eval):
            qv = torch.from_numpy(query_vecs[qi]).float()
            prof = search_shard_routed(qv, router, pipeline, scfg, RouterType.LEMUR, k=100, device=DEVICE)
            profiles.append(prof)

        all_retrieved = [p.retrieved_ids for p in profiles]
        quality = compute_all_metrics(all_retrieved, ground_truth[:n_eval], ks=(10, 100))
        latency = aggregate_profiles(profiles)
        total_s = sum(p.total_ms for p in profiles) / 1000.0
        qps = n_eval / total_s if total_s > 0 else 0.0

        row = {
            "k_candidates": k_cand,
            "recall_at_10": quality.get("recall_at_10", 0),
            "recall_at_100": quality.get("recall_at_100", 0),
            "mrr_at_10": quality.get("mrr_at_10", 0),
            "p50_total_ms": latency.get("p50_total_ms", 0),
            "p95_total_ms": latency.get("p95_total_ms", 0),
            "p50_maxsim_ms": latency.get("p50_maxsim_ms", 0),
            "p50_routing_ms": latency.get("p50_routing_ms", 0),
            "p50_fetch_ms": latency.get("p50_fetch_ms", 0),
            "qps": qps,
        }
        results.append(row)
        print(
            f"k={k_cand:>5d}  R@10={row['recall_at_10']:.4f}  R@100={row['recall_at_100']:.4f}  "
            f"MRR@10={row['mrr_at_10']:.4f}  p50={row['p50_total_ms']:.1f}ms  "
            f"p95={row['p95_total_ms']:.1f}ms  QPS={row['qps']:.1f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    out = Path(__file__).resolve().parent / "results" / "k_sweep_7500.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out}")

    # Summary table
    print(f"\n{'k':>6}  {'R@10':>7}  {'R@100':>7}  {'MRR@10':>7}  {'p50ms':>7}  {'p95ms':>7}  {'QPS':>7}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['k_candidates']:>6}  {r['recall_at_10']:>7.4f}  {r['recall_at_100']:>7.4f}  "
            f"{r['mrr_at_10']:>7.4f}  {r['p50_total_ms']:>7.1f}  {r['p95_total_ms']:>7.1f}  {r['qps']:>7.1f}"
        )


if __name__ == "__main__":
    main()
