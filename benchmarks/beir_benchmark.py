"""
BEIR Benchmark Suite for voyager-index

Runs 6 standard BEIR datasets in two modes:
  1. GPU-corpus  -- full corpus preloaded into VRAM, LEMUR routing, Triton MaxSim
  2. CPU-8-worker -- shard-fetch from mmap, 8-worker parallel, CPU scoring

Reports: NDCG@10, MAP@100, Recall@100, search-only QPS (encoding excluded),
         indexing throughput (encoding excluded).

Reference hardware: NVIDIA RTX A5000 (24 GB VRAM) / consumer-grade CPU.

Usage:
    python benchmarks/beir_benchmark.py
    python benchmarks/beir_benchmark.py --datasets fiqa scifact
    python benchmarks/beir_benchmark.py --modes gpu cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.shard_bench.metrics import compute_all_metrics, ndcg_at_k, map_at_k, recall_at_k
from voyager_index._internal.inference.shard_engine.builder import build, load_corpus, _index_dir
from voyager_index._internal.inference.shard_engine.config import (
    AnnBackend,
    BuildConfig,
    Compression,
    LemurConfig,
    RouterType,
    SearchConfig,
    StorageLayout,
    TransferMode,
)
from voyager_index._internal.inference.shard_engine.fetch_pipeline import FetchPipeline, PinnedBufferPool
from voyager_index._internal.inference.shard_engine.lemur_router import LemurRouter
from voyager_index._internal.inference.shard_engine.scorer import (
    PreloadedGpuCorpus,
    brute_force_maxsim,
    score_all_docs_topk,
    warmup_maxsim,
)
from voyager_index._internal.inference.shard_engine.shard_store import ShardStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BEIR_CACHE = Path.home() / ".cache" / "voyager-qa" / "beir"
INDEX_CACHE = Path.home() / ".cache" / "shard-bench" / "beir"

DATASETS = ["arguana", "fiqa", "nfcorpus", "quora", "scidocs", "scifact"]

TOP_K = 100

OPTIMAL_GPU = dict(
    n_shards=32,
    compression=Compression.FP16,
    layout=StorageLayout.PROXY_GROUPED,
    router_type=RouterType.LEMUR,
    k_candidates=2000,
    max_docs_exact=2000,
    n_full_scores=4096,
    transfer_mode=TransferMode.PINNED,
    lemur_epochs=10,
    ann_backend=AnnBackend.FAISS_FLAT_IP,
)

OPTIMAL_CPU = dict(
    n_shards=32,
    compression=Compression.FP16,
    layout=StorageLayout.PROXY_GROUPED,
    router_type=RouterType.LEMUR,
    k_candidates=2000,
    max_docs_exact=2000,
    n_full_scores=4096,
    transfer_mode=TransferMode.PINNED,
    lemur_epochs=10,
    ann_backend=AnnBackend.FAISS_FLAT_IP,
)

QUORA_OVERRIDE_SHARDS = 32  # quora is ~23k docs, no need for extra shards


def _mem_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1.0


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────

def load_beir_npz(name: str) -> Tuple[np.ndarray, list, list, list, Dict[int, Dict[int, int]], int]:
    """Load a BEIR dataset NPZ. Returns graded qrels as {qi: {doc_idx: grade}}."""
    npz_path = BEIR_CACHE / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dataset {name} not prepared. Run: python benchmarks/data/prepare_beir_datasets.py --datasets {name}"
        )

    npz = np.load(str(npz_path), allow_pickle=True)
    doc_offsets_arr = npz["doc_offsets"]
    n_docs = int(npz["n_docs"])
    dim = int(npz["dim"])
    last_vec = int(doc_offsets_arr[-1][1])
    all_vectors = npz["doc_vectors"][:last_vec]
    doc_offsets = [(int(s), int(e)) for s, e in doc_offsets_arr]
    doc_ids = list(range(n_docs))

    query_offsets = npz["query_offsets"]
    all_q = npz["query_vectors"]
    query_vecs = [all_q[int(s):int(e)].astype(np.float32) for s, e in query_offsets]

    qrels_mat = npz["qrels"]
    grades_mat = npz["qrel_grades"]
    graded_qrels: Dict[int, Dict[int, int]] = {}
    for qi in range(qrels_mat.shape[0]):
        rels: Dict[int, int] = {}
        for ri in range(qrels_mat.shape[1]):
            doc_idx = int(qrels_mat[qi, ri])
            if 0 <= doc_idx < n_docs:
                grade = int(grades_mat[qi, ri])
                rels[doc_idx] = max(grade, 1)
        if rels:
            graded_qrels[qi] = rels

    tok_counts = [e - s for s, e in doc_offsets]
    log.info(
        "%s: %d docs, %d queries, dim=%d, tokens/doc p50=%.0f p95=%.0f",
        name, n_docs, len(query_vecs), dim,
        np.median(tok_counts), np.percentile(tok_counts, 95),
    )
    return all_vectors, doc_offsets, doc_ids, query_vecs, graded_qrels, dim


# ─────────────────────────────────────────────────────────────
# Index building (encoding-free, measures only shard construction)
# ─────────────────────────────────────────────────────────────

def build_index(
    name: str,
    all_vectors: np.ndarray,
    doc_offsets: list,
    doc_ids: list,
    dim: int,
    params: dict,
    device: str = "cuda",
) -> Tuple[Path, float]:
    n_shards = params["n_shards"]
    if name == "quora":
        n_shards = QUORA_OVERRIDE_SHARDS

    bcfg = BuildConfig(
        corpus_size=len(doc_ids),
        n_shards=n_shards,
        dim=dim,
        compression=params["compression"],
        layout=params["layout"],
        router_type=params["router_type"],
        uniform_shard_tokens=True,
        seed=42,
    )
    bcfg.lemur = LemurConfig(
        enabled=params["router_type"] == RouterType.LEMUR,
        device=device,
        ann_backend=params["ann_backend"],
        epochs=params["lemur_epochs"],
        k_candidates=params["k_candidates"],
    )

    index_dir = INDEX_CACHE / name / f"s{n_shards}_{params['compression'].value}_{params['layout'].value}"
    if (index_dir / "manifest.json").exists():
        log.info("Index for %s already built at %s", name, index_dir)
        return index_dir, 0.0

    npz_path = BEIR_CACHE / f"{name}.npz"
    t0 = time.time()
    built_dir = build(bcfg, npz_path=npz_path, device=device)
    build_elapsed = time.time() - t0

    if built_dir != index_dir:
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        if index_dir.exists():
            shutil.rmtree(index_dir)
        shutil.copytree(built_dir, index_dir)

    log.info("Built index for %s in %.1fs at %s", name, build_elapsed, index_dir)
    return index_dir, build_elapsed


# ─────────────────────────────────────────────────────────────
# Search modes
# ─────────────────────────────────────────────────────────────

def _single_query_search(
    query: torch.Tensor,
    router: LemurRouter,
    pipeline: FetchPipeline,
    params: dict,
    k: int,
    device: str,
    gpu_corpus: Optional[PreloadedGpuCorpus] = None,
) -> Tuple[List[int], List[float], float]:
    """Execute a single query. Returns (ids, scores, elapsed_ms)."""
    t0 = time.perf_counter()

    routed = router.route(
        query,
        k_candidates=params["k_candidates"],
        prefetch_doc_cap=params["max_docs_exact"],
    )

    if gpu_corpus is not None:
        candidate_ids = routed.doc_ids[:params["max_docs_exact"]]
        ids, scores, _ = gpu_corpus.score_candidates(
            query, candidate_ids, k=k, return_stats=True,
        )
    else:
        shard_chunks, _ = pipeline.fetch_candidate_docs(
            routed.by_shard, max_docs=params["max_docs_exact"],
        )
        dev = torch.device(device)
        ids, scores, _ = score_all_docs_topk(
            query, shard_chunks, k=k, device=dev,
            variable_length_strategy="bucketed", return_stats=True,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return ids, scores, elapsed_ms


def run_gpu_corpus_mode(
    name: str,
    index_dir: Path,
    all_vectors: np.ndarray,
    doc_offsets: list,
    doc_ids: list,
    query_vecs: list,
    dim: int,
    params: dict,
    n_warmup: int = 5,
) -> Dict[str, Any]:
    """GPU-corpus mode: entire corpus preloaded into VRAM."""
    device = "cuda"
    log.info("[GPU-corpus] %s: loading index ...", name)

    store = ShardStore(index_dir)
    router = LemurRouter(
        index_dir / "lemur",
        ann_backend=params["ann_backend"].value,
        device=device,
    )
    router.load()

    doc_vecs = [all_vectors[s:e] for s, e in doc_offsets]
    gpu_corpus = PreloadedGpuCorpus(doc_vecs, doc_ids, dim, device=device)
    warmup_maxsim(dim=dim, doc_token_counts=[128, 256], device=device)

    pool = PinnedBufferPool(max_tokens=50_000, dim=dim, n_buffers=3)
    pipeline = FetchPipeline(store=store, mode=TransferMode.PINNED, pinned_pool=pool, device=device)

    for i in range(min(n_warmup, len(query_vecs))):
        qv = torch.from_numpy(query_vecs[i]).float()
        _single_query_search(qv, router, pipeline, params, TOP_K, device, gpu_corpus=gpu_corpus)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    log.info("[GPU-corpus] %s: running %d queries ...", name, len(query_vecs))
    all_ids = []
    all_elapsed = []

    for qi in range(len(query_vecs)):
        qv = torch.from_numpy(query_vecs[qi]).float()
        ids, _, elapsed = _single_query_search(qv, router, pipeline, params, TOP_K, device, gpu_corpus=gpu_corpus)
        all_ids.append(ids)
        all_elapsed.append(elapsed)

    total_s = sum(all_elapsed) / 1000.0
    qps = len(query_vecs) / total_s if total_s > 0 else 0
    p50 = float(np.median(all_elapsed))
    p95 = float(np.percentile(all_elapsed, 95))

    del gpu_corpus, pipeline, pool, store, router
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": "GPU-corpus",
        "dataset": name,
        "n_queries": len(query_vecs),
        "all_ids": all_ids,
        "qps": qps,
        "p50_ms": p50,
        "p95_ms": p95,
    }


def _cpu_worker_search(args):
    """Worker function for CPU parallel search."""
    index_dir, query_vec_bytes, query_shape, params, dim, k = args
    query_vec = np.frombuffer(query_vec_bytes, dtype=np.float32).reshape(query_shape)

    store = ShardStore(index_dir)
    router = LemurRouter(
        index_dir / "lemur",
        ann_backend=params["ann_backend"].value,
        device="cpu",
    )
    router.load()
    pipeline = FetchPipeline(store=store, mode=TransferMode.PAGEABLE, pinned_pool=None, device="cpu")

    qv = torch.from_numpy(query_vec).float()
    ids, scores, elapsed = _single_query_search(qv, router, pipeline, params, k, "cpu")
    return ids, elapsed


def run_cpu_multiworker_mode(
    name: str,
    index_dir: Path,
    query_vecs: list,
    dim: int,
    params: dict,
    n_workers: int = 8,
    n_warmup: int = 3,
) -> Dict[str, Any]:
    """CPU multi-worker mode: shard-fetch with parallel workers."""
    device = "cpu"
    log.info("[CPU-%dw] %s: running %d queries ...", n_workers, name, len(query_vecs))

    store = ShardStore(index_dir)
    router = LemurRouter(
        index_dir / "lemur",
        ann_backend=params["ann_backend"].value,
        device=device,
    )
    router.load()
    pipeline = FetchPipeline(store=store, mode=TransferMode.PAGEABLE, pinned_pool=None, device=device)
    warmup_maxsim(dim=dim, doc_token_counts=[128, 256], device=device)

    for i in range(min(n_warmup, len(query_vecs))):
        qv = torch.from_numpy(query_vecs[i]).float()
        _single_query_search(qv, router, pipeline, params, TOP_K, device)

    all_ids = []
    all_elapsed = []

    def _search_one(qi):
        qv = torch.from_numpy(query_vecs[qi]).float()
        ids, _, elapsed = _single_query_search(qv, router, pipeline, params, TOP_K, device)
        return ids, elapsed

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_search_one, qi) for qi in range(len(query_vecs))]
        for f in futures:
            ids, elapsed = f.result()
            all_ids.append(ids)
            all_elapsed.append(elapsed)

    total_s = sum(all_elapsed) / 1000.0
    qps = len(query_vecs) / total_s * n_workers if total_s > 0 else 0
    p50 = float(np.median(all_elapsed))
    p95 = float(np.percentile(all_elapsed, 95))

    del pipeline, store, router
    gc.collect()

    return {
        "mode": f"CPU-{n_workers}w",
        "dataset": name,
        "n_queries": len(query_vecs),
        "all_ids": all_ids,
        "qps": qps,
        "p50_ms": p50,
        "p95_ms": p95,
    }


# ─────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────

def evaluate(
    all_ids: List[List[int]],
    graded_qrels: Dict[int, Dict[int, int]],
    n_queries: int,
) -> Dict[str, float]:
    all_relevant = []
    valid_retrieved = []
    for qi in range(n_queries):
        if qi in graded_qrels:
            all_relevant.append(graded_qrels[qi])
            valid_retrieved.append(all_ids[qi] if qi < len(all_ids) else [])

    return compute_all_metrics(valid_retrieved, all_relevant, ks=(10, 100))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_dataset(name: str, modes: List[str], n_workers: int = 8) -> List[Dict[str, Any]]:
    all_vectors, doc_offsets, doc_ids, query_vecs, graded_qrels, dim = load_beir_npz(name)
    doc_vecs = [all_vectors[s:e] for s, e in doc_offsets]
    n_docs = len(doc_ids)

    results = []

    for mode in modes:
        if mode == "gpu":
            params = dict(OPTIMAL_GPU)
            if name == "quora":
                params["n_shards"] = QUORA_OVERRIDE_SHARDS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            index_dir, build_s = build_index(name, all_vectors, doc_offsets, doc_ids, dim, params, device=device)
            indexing_throughput = n_docs / build_s if build_s > 0 else float("inf")

            search_result = run_gpu_corpus_mode(
                name, index_dir, all_vectors, doc_offsets, doc_ids,
                query_vecs, dim, params,
            )
            metrics = evaluate(search_result["all_ids"], graded_qrels, len(query_vecs))

            results.append({
                **{k: v for k, v in search_result.items() if k != "all_ids"},
                **metrics,
                "indexing_docs_per_sec": indexing_throughput,
                "n_docs": n_docs,
                "dim": dim,
                "top_k": TOP_K,
                "params": {k: str(v) for k, v in params.items()},
            })

        elif mode == "cpu":
            params = dict(OPTIMAL_CPU)
            if name == "quora":
                params["n_shards"] = QUORA_OVERRIDE_SHARDS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            index_dir, build_s = build_index(name, all_vectors, doc_offsets, doc_ids, dim, params, device=device)
            indexing_throughput = n_docs / build_s if build_s > 0 else float("inf")

            search_result = run_cpu_multiworker_mode(
                name, index_dir, query_vecs, dim, params, n_workers=n_workers,
            )
            metrics = evaluate(search_result["all_ids"], graded_qrels, len(query_vecs))

            results.append({
                **{k: v for k, v in search_result.items() if k != "all_ids"},
                **metrics,
                "indexing_docs_per_sec": indexing_throughput,
                "n_docs": n_docs,
                "dim": dim,
                "top_k": TOP_K,
                "params": {k: str(v) for k, v in params.items()},
            })

    del all_vectors, doc_offsets, doc_ids, doc_vecs, query_vecs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def format_results_table(all_results: List[Dict[str, Any]]) -> str:
    lines = []
    hdr = (
        f"{'Dataset':<12} {'Mode':<14} {'n_docs':>8} "
        f"{'NDCG@10':>8} {'MAP@100':>8} {'R@100':>8} "
        f"{'QPS':>8} {'p50ms':>8} {'p95ms':>8} "
        f"{'Idx d/s':>10}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in sorted(all_results, key=lambda x: (x["dataset"], x["mode"])):
        lines.append(
            f"{r['dataset']:<12} {r['mode']:<14} {r['n_docs']:>8} "
            f"{r.get('NDCG@10', 0):>8.4f} {r.get('MAP@100', 0):>8.4f} {r.get('recall@100', 0):>8.4f} "
            f"{r['qps']:>8.1f} {r['p50_ms']:>8.1f} {r['p95_ms']:>8.1f} "
            f"{r['indexing_docs_per_sec']:>10.0f}"
        )

    return "\n".join(lines)


def format_markdown_table(all_results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("| Dataset | Mode | Docs | NDCG@10 | MAP@100 | Recall@100 | QPS | p50 (ms) | p95 (ms) |")
    lines.append("|---------|------|-----:|--------:|--------:|-----------:|----:|---------:|---------:|")

    for r in sorted(all_results, key=lambda x: (x["dataset"], x["mode"])):
        lines.append(
            f"| {r['dataset']} | {r['mode']} | {r['n_docs']:,} "
            f"| {r.get('NDCG@10', 0):.4f} | {r.get('MAP@100', 0):.4f} | {r.get('recall@100', 0):.4f} "
            f"| {r['qps']:.1f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="BEIR Benchmark Suite for voyager-index")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--modes", nargs="*", default=["gpu", "cpu"], choices=["gpu", "cpu"])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="benchmarks/beir_results.jsonl")
    args = parser.parse_args()

    all_results = []

    for name in args.datasets:
        log.info("=" * 70)
        log.info("DATASET: %s", name)
        log.info("=" * 70)

        try:
            ds_results = run_dataset(name, args.modes, n_workers=args.n_workers)
            all_results.extend(ds_results)
        except Exception as e:
            log.error("Failed on %s: %s", name, e, exc_info=True)
            continue

    print("\n" + "=" * 100)
    print("BEIR BENCHMARK RESULTS — voyager-index")
    print(f"Model: lightonai/GTE-ModernColBERT-v1 | top_k={TOP_K}")
    print("=" * 100)
    print(format_results_table(all_results))
    print()
    print("Markdown:")
    print(format_markdown_table(all_results))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    log.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
