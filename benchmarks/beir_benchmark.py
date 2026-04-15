"""
BEIR Benchmark Suite for voyager-index

Runs 6 standard BEIR datasets in two modes:
  1. GPU-corpus  -- full corpus preloaded into VRAM, LEMUR routing, Triton MaxSim
  2. CPU-8-worker -- shard-fetch from mmap, 8-worker parallel, CPU scoring

Reports: search-only BEIR metrics and latency/throughput tables.

Reference hardware: NVIDIA RTX A5000 (24 GB VRAM) / consumer-grade CPU.

Usage:
    python benchmarks/beir_benchmark.py
    python benchmarks/beir_benchmark.py --n-eval 100   # quick sample
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
from concurrent.futures import ThreadPoolExecutor
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


def ensure_merged_layout(
    index_dir: Path,
    all_vectors: np.ndarray,
    doc_offsets: list,
    doc_ids: list,
    dim: int,
) -> None:
    """Persist merged mmap files for native Rust exact CPU scoring."""
    emb_path = index_dir / "merged_embeddings.bin"
    offsets_path = index_dir / "merged_offsets.bin"
    doc_map_path = index_dir / "merged_doc_map.bin"

    if emb_path.exists() and offsets_path.exists() and doc_map_path.exists():
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = int(all_vectors.shape[0])
    f16_vectors = np.ascontiguousarray(all_vectors.astype(np.float16, copy=False))
    merged_offsets = np.array(
        [int(s) for s, _e in doc_offsets] + [int(doc_offsets[-1][1]) if doc_offsets else 0],
        dtype=np.int64,
    )
    merged_doc_ids = np.array(doc_ids, dtype=np.uint64)

    with open(emb_path, "wb") as f:
        f.write(np.array([total_tokens], dtype=np.int64).tobytes())
        f.write(np.array([dim], dtype=np.int64).tobytes())
        f.write(f16_vectors.tobytes())

    with open(offsets_path, "wb") as f:
        f.write(np.array([len(merged_offsets)], dtype=np.int64).tobytes())
        f.write(merged_offsets.tobytes())

    with open(doc_map_path, "wb") as f:
        f.write(np.array([len(merged_doc_ids)], dtype=np.int64).tobytes())
        f.write(merged_doc_ids.tobytes())

    log.info(
        "Merged mmap files written at %s (%d docs, %d tokens)",
        index_dir,
        len(doc_ids),
        total_tokens,
    )


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
    n_warmup: int = 10,
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
    tok_counts = sorted({e - s for s, e in doc_offsets})
    warmup_maxsim(dim=dim, doc_token_counts=tok_counts, device=device)

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


class _RustCpuWorker:
    """Independent CPU worker using native Rust exact scoring."""

    def __init__(self, worker_id: int, index_dir: Path, dim: int, ann_backend: AnnBackend):
        import latence_shard_engine

        runtime_dir = index_dir / f"_rust_cpu_runtime_w{worker_id}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "shard.wal").touch(exist_ok=True)

        self._rust_idx = latence_shard_engine.ShardIndex(str(runtime_dir), dim)
        self._rust_idx.load_merged(str(index_dir))
        self._router = LemurRouter(
            index_dir / "lemur",
            ann_backend=ann_backend.value,
            device="cpu",
        )
        self._router.load()

    def search(self, query_vec: np.ndarray, params: dict, k: int) -> Tuple[List[int], List[float], float]:
        t0 = time.perf_counter()
        q_np = np.ascontiguousarray(query_vec, dtype=np.float32)
        q_t = torch.from_numpy(q_np).float()
        routed = self._router.route(
            q_t,
            k_candidates=params["k_candidates"],
            prefetch_doc_cap=params["max_docs_exact"],
        )
        candidate_ids = [int(doc_id) for doc_id in routed.doc_ids[: params["max_docs_exact"]]]
        if candidate_ids:
            ids, scores = self._rust_idx.score_candidates_exact(q_np, candidate_ids, k)
            out_ids = ids.tolist()
            out_scores = scores.tolist()
        else:
            out_ids = []
            out_scores = []
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return out_ids, out_scores, elapsed_ms


def _run_cpu_fallback_mode(
    name: str,
    index_dir: Path,
    query_vecs: list,
    dim: int,
    params: dict,
    n_workers: int = 8,
    n_warmup: int = 5,
) -> Dict[str, Any]:
    """Sequential fallback when the native Rust CPU path is unavailable."""
    device = "cpu"
    log.info("[CPU-%dw] %s: native CPU path unavailable, using fallback path ...", n_workers, name)

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

    log.info("[CPU-%dw] %s: running %d queries (sequential fallback) ...", n_workers, name, len(query_vecs))
    all_ids = []
    all_elapsed = []

    for qi in range(len(query_vecs)):
        qv = torch.from_numpy(query_vecs[qi]).float()
        ids, _, elapsed = _single_query_search(qv, router, pipeline, params, TOP_K, device)
        all_ids.append(ids)
        all_elapsed.append(elapsed)

    total_s = sum(all_elapsed) / 1000.0
    qps = len(query_vecs) / total_s if total_s > 0 else 0
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


def run_cpu_multiworker_mode(
    name: str,
    index_dir: Path,
    all_vectors: np.ndarray,
    doc_offsets: list,
    doc_ids: list,
    query_vecs: list,
    dim: int,
    params: dict,
    n_workers: int = 8,
    n_warmup: int = 2,
) -> Dict[str, Any]:
    """CPU multi-worker mode using native Rust exact scoring over merged mmap."""
    log.info("[CPU-%dw] %s: preparing native CPU runtime ...", n_workers, name)

    try:
        ensure_merged_layout(index_dir, all_vectors, doc_offsets, doc_ids, dim)
        worker_count = max(1, min(n_workers, len(query_vecs)))
        workers = [
            _RustCpuWorker(i, index_dir, dim, params["ann_backend"])
            for i in range(worker_count)
        ]
    except Exception as exc:
        log.warning("Native CPU runtime init failed for %s: %s", name, exc)
        return _run_cpu_fallback_mode(
            name,
            index_dir,
            query_vecs,
            dim,
            params,
            n_workers=n_workers,
            n_warmup=max(n_warmup, 5),
        )

    query_partitions = [list(range(i, len(query_vecs), worker_count)) for i in range(worker_count)]

    for worker, indices in zip(workers, query_partitions):
        for qi in indices[: min(n_warmup, len(indices))]:
            worker.search(query_vecs[qi], params, TOP_K)

    log.info("[CPU-%dw] %s: running %d queries (native exact, %d workers) ...",
             worker_count, name, len(query_vecs), worker_count)

    def _run_partition(worker: _RustCpuWorker, indices: List[int]) -> List[Tuple[int, List[int], float]]:
        out = []
        for qi in indices:
            ids, _scores, elapsed = worker.search(query_vecs[qi], params, TOP_K)
            out.append((qi, ids, elapsed))
        return out

    wall_t0 = time.perf_counter()
    all_results = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_run_partition, worker, indices)
            for worker, indices in zip(workers, query_partitions)
            if indices
        ]
        for future in futures:
            all_results.extend(future.result())
    wall_s = time.perf_counter() - wall_t0

    ordered_ids = [[] for _ in range(len(query_vecs))]
    all_elapsed = []
    for qi, ids, elapsed in all_results:
        ordered_ids[qi] = ids
        all_elapsed.append(elapsed)

    qps = len(query_vecs) / wall_s if wall_s > 0 else 0
    p50 = float(np.median(all_elapsed)) if all_elapsed else 0.0
    p95 = float(np.percentile(all_elapsed, 95)) if all_elapsed else 0.0

    log.info("[CPU-%dw] %s: QPS=%.1f  p50=%.1fms  p95=%.1fms",
             worker_count, name, qps, p50, p95)

    del workers
    gc.collect()

    return {
        "mode": f"CPU-{worker_count}w",
        "dataset": name,
        "n_queries": len(query_vecs),
        "all_ids": ordered_ids,
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

def run_dataset(
    name: str,
    modes: List[str],
    n_workers: int = 8,
    n_eval: Optional[int] = 0,
) -> List[Dict[str, Any]]:
    all_vectors, doc_offsets, doc_ids, query_vecs, graded_qrels, dim = load_beir_npz(name)
    doc_vecs = [all_vectors[s:e] for s, e in doc_offsets]
    n_docs = len(doc_ids)
    eval_query_vecs = query_vecs[: min(len(query_vecs), n_eval)] if n_eval else query_vecs

    log.info(
        "%s: evaluating %d/%d queries to mirror prior shard-bench runs",
        name,
        len(eval_query_vecs),
        len(query_vecs),
    )

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
                eval_query_vecs, dim, params,
            )
            metrics = evaluate(search_result["all_ids"], graded_qrels, len(eval_query_vecs))

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
                name,
                index_dir,
                all_vectors,
                doc_offsets,
                doc_ids,
                eval_query_vecs,
                dim,
                params,
                n_workers=n_workers,
            )
            metrics = evaluate(search_result["all_ids"], graded_qrels, len(eval_query_vecs))

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


def _merge_gpu_cpu(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group results by dataset and merge GPU/CPU rows into one per dataset."""
    by_ds: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        ds = r["dataset"]
        if ds not in by_ds:
            by_ds[ds] = {"dataset": ds, "n_docs": r.get("n_docs", 0)}
        row = by_ds[ds]
        if "GPU" in r.get("mode", ""):
            row["gpu_qps"] = r["qps"]
            row["gpu_p95"] = r["p95_ms"]
            for k in ("NDCG@10", "NDCG@100", "MAP@100", "recall@10", "recall@100"):
                if k in r:
                    row[k] = r[k]
        elif "CPU" in r.get("mode", ""):
            row["cpu_qps"] = r["qps"]
            row["cpu_p95"] = r["p95_ms"]
            for k in ("NDCG@10", "NDCG@100", "MAP@100", "recall@10", "recall@100"):
                if k not in row and k in r:
                    row[k] = r[k]
    return sorted(by_ds.values(), key=lambda x: x["dataset"])


def format_results_table(all_results: List[Dict[str, Any]]) -> str:
    lines = []
    hdr = (
        f"{'Dataset':<12} {'Docs':>8} "
        f"{'MAP@100':>8} {'NDCG@10':>8} {'NDCG@100':>9} {'R@10':>7} {'R@100':>7} "
        f"{'GPU QPS':>9} {'GPU P95':>9} {'CPU QPS':>9} {'CPU P95':>9}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for row in _merge_gpu_cpu(all_results):
        lines.append(
            f"{row['dataset']:<12} {row['n_docs']:>8} "
            f"{row.get('MAP@100', 0):>8.4f} {row.get('NDCG@10', 0):>8.4f} {row.get('NDCG@100', 0):>9.4f} "
            f"{row.get('recall@10', 0):>7.4f} {row.get('recall@100', 0):>7.4f} "
            f"{row.get('gpu_qps', 0):>9.1f} {row.get('gpu_p95', 0):>9.1f} "
            f"{row.get('cpu_qps', 0):>9.1f} {row.get('cpu_p95', 0):>9.1f}"
        )

    return "\n".join(lines)


def format_markdown_table(all_results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(
        "| Dataset | Documents | MAP@100 | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 "
        "| GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |"
    )
    lines.append(
        "|---------|----------:|--------:|--------:|---------:|----------:|-----------:"
        "|--------:|-------------:|--------:|-------------:|"
    )

    for row in _merge_gpu_cpu(all_results):
        lines.append(
            f"| {row['dataset']} | {row['n_docs']:,} "
            f"| {row.get('MAP@100', 0):.4f} | {row.get('NDCG@10', 0):.4f} | {row.get('NDCG@100', 0):.4f} "
            f"| {row.get('recall@10', 0):.4f} | {row.get('recall@100', 0):.4f} "
            f"| {row.get('gpu_qps', 0):.1f} | {row.get('gpu_p95', 0):.1f} "
            f"| {row.get('cpu_qps', 0):.1f} | {row.get('cpu_p95', 0):.1f} |"
        )

    return "\n".join(lines)


def format_comparison_table(all_results: List[Dict[str, Any]]) -> str:
    """Compact GPU vs CPU speedup comparison."""
    lines = []
    lines.append("| Dataset | GPU QPS | CPU QPS | Speedup | GPU P95 (ms) | CPU P95 (ms) | Latency Ratio |")
    lines.append("|---------|--------:|--------:|--------:|-------------:|-------------:|--------------:|")

    for row in _merge_gpu_cpu(all_results):
        gpu_q = row.get("gpu_qps", 0)
        cpu_q = row.get("cpu_qps", 0)
        gpu_p = row.get("gpu_p95", 0)
        cpu_p = row.get("cpu_p95", 0)
        speedup = gpu_q / cpu_q if cpu_q > 0 else float("inf")
        lat_ratio = cpu_p / gpu_p if gpu_p > 0 else float("inf")
        lines.append(
            f"| {row['dataset']} | {gpu_q:.1f} | {cpu_q:.1f} | {speedup:.1f}x "
            f"| {gpu_p:.1f} | {cpu_p:.1f} | {lat_ratio:.1f}x |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="BEIR Benchmark Suite for voyager-index")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--modes", nargs="*", default=["gpu", "cpu"], choices=["gpu", "cpu"])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument(
        "--n-eval",
        type=int,
        default=0,
        help="Number of queries to evaluate per dataset. Use 0 for the full query set.",
    )
    parser.add_argument("--output", type=str, default="benchmarks/beir_results.jsonl")
    args = parser.parse_args()

    all_results = []

    for name in args.datasets:
        log.info("=" * 70)
        log.info("DATASET: %s", name)
        log.info("=" * 70)

        try:
            ds_results = run_dataset(name, args.modes, n_workers=args.n_workers, n_eval=args.n_eval)
            all_results.extend(ds_results)
        except Exception as e:
            log.error("Failed on %s: %s", name, e, exc_info=True)
            continue

    print("\n" + "=" * 120)
    print("BEIR BENCHMARK RESULTS — voyager-index (search-only, encoding excluded)")
    print(f"Model: lightonai/GTE-ModernColBERT-v1 | top_k={TOP_K}")
    print("=" * 120)
    print(format_results_table(all_results))
    print()
    print("Markdown (next-plaid style):")
    print(format_markdown_table(all_results))
    print()
    print("GPU vs CPU comparison:")
    print(format_comparison_table(all_results))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    log.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
