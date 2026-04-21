"""Main benchmark harness for shard_bench."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.shard_bench.baselines import BaselineDenseSingleVector, BaselineGpuMaxSim
from colsearch._internal.inference.shard_engine.builder import DEFAULT_NPZ, _index_dir, build, load_corpus
from colsearch._internal.inference.shard_engine.centroid_router import CentroidRouter
from colsearch._internal.inference.shard_engine.colbandit_reranker import ColBanditConfig as _CBConfig, ColBanditReranker
from colsearch._internal.inference.shard_engine.config import (
    BenchmarkConfig,
    BuildConfig,
    Compression,
    RouterType,
    SearchConfig,
    StorageLayout,
    TransferMode,
    SWEEP_K_CANDIDATES,
    SWEEP_MAX_DOCS_EXACT,
    SWEEP_TOP_SHARDS,
    SWEEP_TRANSFER,
)
from colsearch._internal.inference.shard_engine.fetch_pipeline import FetchPipeline, PinnedBufferPool
from colsearch._internal.inference.shard_engine.lemur_router import LemurRouter
from colsearch._internal.inference.shard_engine.scorer import brute_force_maxsim, score_all_docs_topk, warmup_maxsim, PreloadedGpuCorpus
from benchmarks.shard_bench.metrics import compute_all_metrics
from colsearch._internal.inference.shard_engine.profiler import QueryProfile, Timer, aggregate_profiles, memory_snapshot
from colsearch._internal.inference.shard_engine.shard_store import ShardStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _mem_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1.0


# ======================================================================
# Ground truth
# ======================================================================

def compute_ground_truth(
    query_vecs: list,
    doc_vecs: list,
    doc_ids: List[int],
    dim: int,
    n_eval: int,
    k: int = 100,
    cache_path: Optional[Path] = None,
    device: str = "cuda",
) -> List[List[int]]:
    if cache_path and cache_path.exists():
        log.info("Loading cached ground truth from %s", cache_path)
        data = np.load(str(cache_path), allow_pickle=True)
        gt = [row.tolist() for row in data["gt_ids"][:n_eval]]
        if len(gt) >= n_eval:
            return gt

    log.info("Computing brute-force ground truth for %d queries (k=%d)...", n_eval, k)
    t0 = time.time()
    gts: List[List[int]] = []
    for qi in range(n_eval):
        ids, _scores = brute_force_maxsim(
            query_vecs[qi], doc_vecs, doc_ids, dim, k=k, device=device,
        )
        gts.append(ids)
        if (qi + 1) % 20 == 0:
            elapsed = time.time() - t0
            log.info("  GT %d/%d (%.1fs, %.1f q/s)", qi + 1, n_eval, elapsed, (qi + 1) / elapsed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info("Ground truth computed in %.1fs", time.time() - t0)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        max_len = max(len(g) for g in gts) if gts else 0
        gt_arr = np.full((len(gts), max_len), -1, dtype=np.int64)
        for i, g in enumerate(gts):
            gt_arr[i, :len(g)] = g
        np.savez_compressed(str(cache_path), gt_ids=gt_arr)

    return gts


# ======================================================================
# Shard-routed search (centroid + LEMUR paths)
# ======================================================================

def search_shard_routed(
    query: torch.Tensor,
    router,
    pipeline: FetchPipeline,
    search_cfg: SearchConfig,
    router_type: RouterType,
    k: int = 10,
    device: str = "cuda",
    gpu_corpus: PreloadedGpuCorpus = None,
) -> QueryProfile:
    prof = QueryProfile()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    with Timer(sync_cuda=True) as t_route:
        if router_type == RouterType.CENTROID:
            routed = router.route(query, top_shards=search_cfg.top_shards, max_docs=search_cfg.max_docs_exact)
        else:
            routed = router.route(
                query,
                k_candidates=search_cfg.k_candidates,
                prefetch_doc_cap=search_cfg.max_docs_exact,
            )
    prof.routing_ms = t_route.elapsed_ms

    # ---- GEM fast path: GPU-resident corpus, zero fetch ----
    if gpu_corpus is not None and router_type == RouterType.LEMUR:
        candidate_ids = routed.doc_ids[:search_cfg.max_docs_exact]
        prof.num_docs_scored = len(candidate_ids)
        prof.num_shards_fetched = 0
        prof.fetch_ms = 0.0
        prof.h2d_bytes = 0

        ids, scores, score_stats = gpu_corpus.score_candidates(
            query, candidate_ids, k=k, return_stats=True,
        )
        prof.exact_ms = score_stats.get("exact_ms", 0.0)
        prof.h2d_ms = score_stats.get("h2d_ms", 0.0)
        prof.maxsim_ms = score_stats.get("maxsim_ms", prof.exact_ms)
        prof.topk_ms = score_stats.get("topk_ms", 0.0)
        prof.retrieved_ids = ids
        prof.retrieved_scores = scores
        prof.total_ms = prof.routing_ms + prof.exact_ms
        return prof

    # ---- Shard-fetch path (disk-backed, for large corpora) ----
    if router_type == RouterType.CENTROID:
        prof.num_shards_fetched = len(routed)
        shard_chunks, fetch_stats = pipeline.fetch_per_shard(routed, max_docs=search_cfg.max_docs_exact)
    else:
        prof.num_shards_fetched = len(routed.shard_ids)
        shard_chunks, fetch_stats = pipeline.fetch_candidate_docs(
            routed.by_shard, max_docs=search_cfg.max_docs_exact,
        )

    prof.fetch_ms = fetch_stats.get("fetch_ms", 0.0)
    prof.h2d_bytes = fetch_stats.get("h2d_bytes", 0)
    prof.num_docs_scored = fetch_stats.get("num_docs", 0)

    if prof.num_docs_scored > 0:
        with Timer(sync_cuda=True) as t_score:
            if search_cfg.use_colbandit and router_type == RouterType.LEMUR:
                reranker = ColBanditReranker(search_cfg.colbandit)
                ids, scores, _stats = reranker.rerank_shard_chunks(query, shard_chunks, k=k, device=dev)
                prof.exact_ms = t_score.elapsed_ms
                prof.maxsim_ms = t_score.elapsed_ms
            else:
                ids, scores, score_stats = score_all_docs_topk(
                    query,
                    shard_chunks,
                    k=k,
                    device=dev,
                    variable_length_strategy=search_cfg.variable_length_strategy,
                    return_stats=True,
                )
                prof.exact_ms = score_stats.get("exact_ms", t_score.elapsed_ms)
                prof.h2d_ms = score_stats.get("h2d_ms", 0.0)
                prof.maxsim_ms = score_stats.get("maxsim_ms", t_score.elapsed_ms)
                prof.topk_ms = score_stats.get("topk_ms", 0.0)
        prof.retrieved_ids = ids
        prof.retrieved_scores = scores

    prof.total_ms = prof.routing_ms + prof.prune_ms + prof.fetch_ms + prof.exact_ms
    return prof


# ======================================================================
# Single sweep configuration
# ======================================================================

def run_single_config(
    cfg: BenchmarkConfig,
    query_vecs: list,
    doc_vecs: list,
    doc_ids: List[int],
    ground_truth: List[List[int]],
    dim: int,
    device: str = "cuda",
    force_no_gpu_corpus: bool = False,
) -> dict:
    bcfg = cfg.build
    scfg = cfg.search
    index_dir = _index_dir(bcfg)

    if not (index_dir / "manifest.json").exists():
        build(bcfg, device=device)

    store = ShardStore(index_dir)

    if bcfg.router_type == RouterType.CENTROID:
        router = CentroidRouter.load(index_dir / "router", device=device)
    else:
        router = LemurRouter(
            index_dir / "lemur",
            ann_backend=bcfg.lemur.ann_backend.value,
            device=bcfg.lemur.device,
        )
        router.load()

    pool = None
    if scfg.transfer_mode in (TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED):
        pool = PinnedBufferPool(
            max_tokens=scfg.pinned_buffer_max_tokens,
            dim=dim,
            n_buffers=scfg.pinned_pool_buffers,
        )
    pipeline = FetchPipeline(store=store, mode=scfg.transfer_mode, pinned_pool=pool, device=device)

    n_eval = min(cfg.n_eval_queries, len(query_vecs), len(ground_truth))
    profiles: List[QueryProfile] = []

    gpu_corpus = None
    if force_no_gpu_corpus:
        log.info("GPU corpus DISABLED (force_no_gpu_corpus=True) — using shard fetch")
    else:
        max_tok_est = max((v.shape[0] for v in doc_vecs), default=1)
        if PreloadedGpuCorpus.fits_on_gpu(len(doc_vecs), max_tok_est, dim):
            gpu_corpus = PreloadedGpuCorpus(doc_vecs, doc_ids, dim, device=device)
        else:
            log.info("Corpus too large for GPU (%d docs × %d tok) — using shard fetch", len(doc_vecs), max_tok_est)

    log.info(
        "Running %d queries: router=%s top_shards=%s k_candidates=%s max_docs=%d transfer=%s compression=%s layout=%s gpu_corpus=%s",
        n_eval, bcfg.router_type.value,
        scfg.top_shards if bcfg.router_type == RouterType.CENTROID else "-",
        scfg.k_candidates if bcfg.router_type == RouterType.LEMUR else "-",
        scfg.max_docs_exact, scfg.transfer_mode.value,
        bcfg.compression.value, bcfg.layout.value,
        "yes" if gpu_corpus else "no",
    )

    # Singleton warmup: pre-warm Triton autotune for expected shapes
    if store.manifest:
        representative_token_counts = [
            int(s.p95_tokens) for s in store.manifest.shards if s.p95_tokens > 0
        ]
        if not representative_token_counts:
            representative_token_counts = [128, 256]
    else:
        representative_token_counts = [128, 256]
    warmup_maxsim(dim=dim, doc_token_counts=representative_token_counts, device=device)
    log.info("  Kernel warmup done")

    for qi in range(n_eval):
        qv = torch.from_numpy(query_vecs[qi]).float()
        prof = search_shard_routed(qv, router, pipeline, scfg, bcfg.router_type, k=cfg.top_k_recall, device=device, gpu_corpus=gpu_corpus)
        profiles.append(prof)
        if (qi + 1) % 50 == 0:
            log.info("  Query %d/%d  p50_total=%.1fms", qi + 1, n_eval,
                     np.median([p.total_ms for p in profiles]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_retrieved = [p.retrieved_ids for p in profiles]
    gt_slice = ground_truth[:n_eval]
    quality = compute_all_metrics(all_retrieved, gt_slice, ks=(10, 100))
    latency = aggregate_profiles(profiles)
    mem = memory_snapshot()
    total_time_s = sum(p.total_ms for p in profiles) / 1000.0
    qps = n_eval / total_time_s if total_time_s > 0 else 0.0

    result = {
        "router_type": bcfg.router_type.value,
        "corpus_size": bcfg.corpus_size,
        "compression": bcfg.compression.value,
        "layout": bcfg.layout.value,
        "n_centroids": bcfg.n_centroids,
        "n_shards": bcfg.n_shards,
        "top_shards": scfg.top_shards if bcfg.router_type == RouterType.CENTROID else None,
        "k_candidates": scfg.k_candidates if bcfg.router_type == RouterType.LEMUR else None,
        "max_docs_exact": scfg.max_docs_exact,
        "transfer_mode": scfg.transfer_mode.value,
        "use_colbandit": scfg.use_colbandit and bcfg.router_type == RouterType.LEMUR,
        "pooling_enabled": bcfg.pooling.enabled,
        "pool_factor": bcfg.pooling.pool_factor if bcfg.pooling.enabled else None,
        "n_eval": n_eval,
        **quality,
        **latency,
        "qps": qps,
        **mem,
    }

    log.info(
        "RESULT: R@10=%.4f R@100=%.4f MRR@10=%.4f p50=%.1fms p95=%.1fms QPS=%.1f",
        quality.get("recall_at_10", 0), quality.get("recall_at_100", 0),
        quality.get("mrr_at_10", 0),
        latency.get("p50_total_ms", 0), latency.get("p95_total_ms", 0),
        qps,
    )

    return result


# ======================================================================
# Baselines
# ======================================================================

def run_baselines(
    query_vecs: list,
    doc_vecs: list,
    doc_ids: List[int],
    ground_truth: List[List[int]],
    dim: int,
    n_eval: int,
    max_docs_gpu: int = 10_000,
    device: str = "cuda",
) -> List[dict]:
    results: List[dict] = []

    log.info("=== Baseline A: GPU-only MaxSim ===")
    try:
        baseline_a = BaselineGpuMaxSim(doc_vecs, doc_ids, dim, device=device, max_docs=max_docs_gpu)
        profiles_a = []
        for qi in range(n_eval):
            qv = torch.from_numpy(query_vecs[qi]).float()
            profiles_a.append(baseline_a.search(qv, k=100))
        all_ret_a = [p.retrieved_ids for p in profiles_a]
        quality_a = compute_all_metrics(all_ret_a, ground_truth[:n_eval], ks=(10, 100))
        latency_a = aggregate_profiles(profiles_a)
        total_s = sum(p.total_ms for p in profiles_a) / 1000.0
        results.append({
            "pipeline": "baseline_a_gpu_maxsim",
            "corpus_size": max_docs_gpu,
            **quality_a, **latency_a,
            "qps": n_eval / total_s if total_s > 0 else 0,
            **memory_snapshot(),
        })
        log.info("Baseline A: R@10=%.4f p50=%.1fms", quality_a.get("recall_at_10", 0), latency_a.get("p50_total_ms", 0))
        del baseline_a
    except Exception as e:
        log.warning("Baseline A failed: %s", e)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("=== Baseline C: Dense single-vector ===")
    try:
        baseline_c = BaselineDenseSingleVector(doc_vecs, doc_ids, dim)
        profiles_c = []
        for qi in range(n_eval):
            qv = torch.from_numpy(query_vecs[qi]).float()
            profiles_c.append(baseline_c.search(qv, k=100))
        all_ret_c = [p.retrieved_ids for p in profiles_c]
        quality_c = compute_all_metrics(all_ret_c, ground_truth[:n_eval], ks=(10, 100))
        latency_c = aggregate_profiles(profiles_c)
        total_s = sum(p.total_ms for p in profiles_c) / 1000.0
        results.append({
            "pipeline": "baseline_c_dense_single_vector",
            "corpus_size": len(doc_ids),
            **quality_c, **latency_c,
            "qps": n_eval / total_s if total_s > 0 else 0,
            **memory_snapshot(),
        })
        log.info("Baseline C: R@10=%.4f p50=%.1fms", quality_c.get("recall_at_10", 0), latency_c.get("p50_total_ms", 0))
        del baseline_c
    except Exception as e:
        log.warning("Baseline C failed: %s", e)

    gc.collect()
    return results


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Shard Benchmark Harness")
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="Minimal sweep for smoke testing")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-gt", action="store_true")
    parser.add_argument("--cold-reads", action="store_true", help="Drop page cache before each run")
    parser.add_argument("--router", choices=[x.value for x in RouterType], default=RouterType.LEMUR.value)
    parser.add_argument("--enable-pooling", action="store_true")
    parser.add_argument("--pool-factor", type=int, default=2)
    parser.add_argument("--use-colbandit", action="store_true")
    parser.add_argument("--k-candidates", type=int, default=2000)
    parser.add_argument("--no-gpu-corpus", action="store_true", help="Force shard-fetch path, skip GPU corpus preloading")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim = load_corpus(
        Path(args.npz), max_docs=args.corpus_size,
    )
    doc_vecs = [all_vectors[s:e] for s, e in doc_offsets]
    n_eval = min(args.n_eval, len(query_vecs))

    gt_cache = Path.home() / ".cache" / "shard-bench" / f"gt_{args.corpus_size}.npz"
    if args.skip_gt and gt_cache.exists():
        data = np.load(str(gt_cache), allow_pickle=True)
        ground_truth = [row[row >= 0].tolist() for row in data["gt_ids"][:n_eval]]
    else:
        ground_truth = compute_ground_truth(
            query_vecs, doc_vecs, doc_ids, dim, n_eval, k=100,
            cache_path=gt_cache, device=device,
        )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"bench_{args.corpus_size}.jsonl"

    all_results: List[dict] = []

    if not args.skip_baselines:
        baseline_results = run_baselines(
            query_vecs, doc_vecs, doc_ids, ground_truth, dim, n_eval,
            max_docs_gpu=min(args.corpus_size, 20_000),
            device=device,
        )
        all_results.extend(baseline_results)

    router_type = RouterType(args.router)

    if args.quick:
        sweep_shards = [8]
        sweep_docs = [5_000]
        sweep_transfer = [TransferMode.PINNED]
        sweep_compression = [Compression.FP16]
        sweep_layout = [StorageLayout.PROXY_GROUPED if router_type == RouterType.LEMUR else StorageLayout.CENTROID_GROUPED]
        sweep_kcand = [args.k_candidates]
    else:
        sweep_shards = SWEEP_TOP_SHARDS
        sweep_docs = SWEEP_MAX_DOCS_EXACT
        sweep_transfer = SWEEP_TRANSFER
        sweep_compression = [Compression.FP16, Compression.INT8]
        sweep_layout = [StorageLayout.PROXY_GROUPED, StorageLayout.RANDOM] if router_type == RouterType.LEMUR else [StorageLayout.CENTROID_GROUPED, StorageLayout.RANDOM]
        sweep_kcand = SWEEP_K_CANDIDATES if router_type == RouterType.LEMUR else [args.k_candidates]

    for compression in sweep_compression:
        for layout in sweep_layout:
            bcfg = BuildConfig(
                corpus_size=args.corpus_size,
                compression=compression,
                layout=layout,
                router_type=router_type,
            )
            bcfg.pooling.enabled = bool(args.enable_pooling)
            bcfg.pooling.pool_factor = int(args.pool_factor)
            bcfg.lemur.enabled = bcfg.router_type == RouterType.LEMUR
            bcfg.lemur.device = device
            build(bcfg, npz_path=Path(args.npz), device=device)

            for top_shards in sweep_shards:
                for max_docs in sweep_docs:
                    for transfer in sweep_transfer:
                        for k_cand in sweep_kcand:
                            scfg = SearchConfig(
                                top_shards=top_shards,
                                max_docs_exact=max_docs,
                                transfer_mode=transfer,
                                k_candidates=k_cand,
                                use_colbandit=args.use_colbandit,
                            )
                            if args.use_colbandit:
                                scfg.colbandit = _CBConfig()
                            cfg = BenchmarkConfig(
                                build=bcfg,
                                search=scfg,
                                n_eval_queries=n_eval,
                            )
                            if args.cold_reads:
                                idx_dir = _index_dir(bcfg)
                                ShardStore(idx_dir).drop_all_page_cache()

                            try:
                                result = run_single_config(
                                    cfg, query_vecs, doc_vecs, doc_ids,
                                    ground_truth, dim, device=device,
                                    force_no_gpu_corpus=args.no_gpu_corpus,
                                )
                                result["pipeline"] = f"shard_routed_{bcfg.router_type.value}"
                                all_results.append(result)
                            except Exception as e:
                                log.error("Config failed: %s", e, exc_info=True)

                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

    with open(results_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    log.info("Results written to %s (%d entries)", results_file, len(all_results))
    log.info("Final RSS: %.1f GB", _mem_gb())

    _print_summary(all_results)


def _print_summary(results: List[dict]) -> None:
    W = 140
    print("\n" + "=" * W)
    print(
        f"{'Pipeline':<28} {'Router':<8} {'Compr':<6} {'Layout':<14} {'Shards':>6} "
        f"{'kCand':>6} {'MaxDocs':>8} {'R@10':>7} {'R@100':>7} {'p50ms':>7} {'p95ms':>7} {'QPS':>7}"
    )
    print("-" * W)
    for r in results:
        print(
            f"{r.get('pipeline', '?'):<28} "
            f"{r.get('router_type', '-'):<8} "
            f"{r.get('compression', '-'):<6} "
            f"{r.get('layout', '-'):<14} "
            f"{str(r.get('top_shards', '-')):>6} "
            f"{str(r.get('k_candidates', '-')):>6} "
            f"{str(r.get('max_docs_exact', '-')):>8} "
            f"{r.get('recall_at_10', 0):>7.4f} "
            f"{r.get('recall_at_100', 0):>7.4f} "
            f"{r.get('p50_total_ms', 0):>7.1f} "
            f"{r.get('p95_total_ms', 0):>7.1f} "
            f"{r.get('qps', 0):>7.1f}"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
