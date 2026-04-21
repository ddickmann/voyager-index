"""
BEIR Benchmark Suite for colsearch

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
from colsearch._internal.inference.shard_engine.builder import build, load_corpus, _index_dir
from colsearch._internal.inference.shard_engine.config import (
    AnnBackend,
    BuildConfig,
    Compression,
    LemurConfig,
    RouterType,
    SearchConfig,
    StorageLayout,
    TransferMode,
)
from colsearch._internal.inference.shard_engine.fetch_pipeline import FetchPipeline, PinnedBufferPool
from colsearch._internal.inference.shard_engine.lemur_router import LemurRouter
from colsearch._internal.inference.shard_engine.scorer import (
    PreloadedGpuCorpus,
    brute_force_maxsim,
    score_all_docs_topk,
    score_rroq158_topk,
    score_rroq4_riem_topk,
    warmup_maxsim,
)
from colsearch._internal.inference.shard_engine.shard_store import ShardStore
from colsearch._internal.inference.quantization.rroq158 import (
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)
from colsearch._internal.inference.quantization.rroq4_riem import (
    Rroq4RiemConfig,
    encode_query_for_rroq4_riem,
    encode_rroq4_riem,
)
from colsearch._internal.inference.quantization.distill_mv import (
    MultiViewDistillHead,
    build_features_for_shortlist,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# One-shot per-shape log so we can verify which kernel actually
# dispatches for each tier (fused CUDA vs Triton fallback). Without
# this, an exception in score_b1_fused gets silently swallowed and the
# caller can't tell whether the slow path is in use.
_fused_diag_log_seen: set = set()


def _log_fused_used(B: int, T: int, status: str) -> None:
    key = (int(B), int(T), status[:80])
    if key in _fused_diag_log_seen:
        return
    _fused_diag_log_seen.add(key)
    log.info("[rroq158-bench] B=%d T=%d → %s", B, T, status)

BEIR_CACHE = Path.home() / ".cache" / "voyager-qa" / "beir"
INDEX_CACHE = Path.home() / ".cache" / "shard-bench" / "beir"

DATASETS = ["arguana", "fiqa", "nfcorpus", "quora", "scidocs", "scifact"]

TOP_K = 100

# Whole-corpus fast-path threshold. On H100, the LEMUR route() cost
# (~1 ms from FAISS ANN search + MLP forward + candidate plan build)
# dominates a whole-corpus MaxSim below ~8 K docs. Below that size, a
# direct `PreloadedGpuCorpus.score_all(...)` call is both strictly
# faster AND strictly higher-quality (no router recall loss over the
# exact top-k) than route + score_candidates. Corpora above this
# threshold fall back to route + score_candidates, where shrinking the
# kernel work from n_docs → max_docs_exact pays for the route cost.
# The constant is conservative; operators on A100/H100 class hardware
# can safely push it to 16 K or higher for additional quality.
# Default fast-path threshold: corpora with at most this many docs are
# scored end-to-end (skip LEMUR routing + candidate gather). Set
# generously high because:
#
#   * Modern accelerator GPUs (A100 / H100 / H200) have 40-141 GB VRAM
#     and the score_all path on fp16 tensor cores stays sub-millisecond
#     up to several hundred thousand docs at typical ColBERT token
#     counts (32 q-tokens × 128 d-tokens × dim=128). Empirically on H100
#     a 522 k-doc corpus runs ~3 ms p50 vs ~10 ms via routing.
#   * The PreloadedGpuCorpus has *already* paid the VRAM cost to
#     materialise the corpus on the device, so there is no extra
#     allocation in the fast path.
#   * NDCG never decreases vs the routing path (the router can only
#     prune candidates) and frequently increases.
#
# For very large corpora (multi-million docs) routing wins because the
# whole-corpus MaxSim work scales linearly while routing cost is
# nlist-bounded. The dynamic VRAM check in ``_should_use_fast_path``
# adds an additional gate so corpora that *would* OOM the device skip
# the fast path even if they're below the doc-count threshold.
WHOLE_CORPUS_FAST_PATH_THRESHOLD = int(
    os.environ.get("VOYAGER_FASTPATH_MAX_DOCS", 1_000_000)
)

# CPU whole-corpus fast-path thresholds. Lower than the GPU threshold
# because CPU MaxSim throughput per doc is ~50× lower than GPU and
# LEMUR routing pays for itself faster. Picked from on-host
# microbenchmarks: fp16 BLAS sustains ~3-4 GFLOP/s/core, rroq158 SIMD
# sustains ~12-15 GFLOP/s/core (4× compression × 3× SIMD utilization).
CPU_FASTPATH_FP16_MAX_DOCS = int(
    os.environ.get("VOYAGER_CPU_FASTPATH_FP16_MAX_DOCS", 200_000)
)
CPU_FASTPATH_RROQ158_MAX_DOCS = int(
    os.environ.get("VOYAGER_CPU_FASTPATH_RROQ158_MAX_DOCS", 1_000_000)
)


def _should_use_cpu_fast_path(
    n_docs: int,
    max_docs_exact: int,
    *,
    codec: str = "fp16",
) -> bool:
    """CPU equivalent of `_should_use_fast_path`.

    fp16: exact MaxSim is BLAS-bound; threshold = 200 K (~3 GB working
    set on a 32-core box, fits in L3 + RAM with ~8ms/query budget).

    rroq158: exact MaxSim is popcount-bound and ~6× faster per byte
    than fp16; threshold = 1 M docs.

    When ``max_docs_exact >= n_docs``, LEMUR would route to everything
    anyway, so the fast-path is a strict win regardless of size.
    """
    if max_docs_exact >= n_docs:
        return True
    if codec == "rroq158":
        return n_docs <= CPU_FASTPATH_RROQ158_MAX_DOCS
    return n_docs <= CPU_FASTPATH_FP16_MAX_DOCS


def _should_use_fast_path(
    n_docs: int,
    max_docs_exact: int,
    *,
    gpu_corpus_bytes: int | None = None,
) -> bool:
    """Decide whether to skip routing and score the whole corpus.

    The PreloadedGpuCorpus has already paid the VRAM cost; the fast
    path's only per-query overhead is an O(n_docs) score buffer
    (~1.5 MB at 382 k docs), which is negligible vs the multi-tier
    LEMUR pipeline (route + gather + score_candidates).

    Two triggers:
      (a) ``max_docs_exact >= n_docs`` — routing would pick everything
          anyway, so skipping it is a strict win.
      (b) ``n_docs <= WHOLE_CORPUS_FAST_PATH_THRESHOLD`` — at default
          1 M-doc threshold, this captures every BEIR-class corpus on
          accelerator-class GPUs.

    The previous "corpus must be ≤25% of free VRAM" gate was wrong:
    it compared the *already-resident* corpus footprint against the
    *post-load* free pool, so anything above ~25 % of VRAM (e.g. webis
    382 k docs at 25 GB on an 80 GB H100) was forced through LEMUR
    even though full-corpus MaxSim is faster *and* strictly higher
    NDCG. The router only prunes candidates; it never adds quality.
    """
    if max_docs_exact >= n_docs:
        return True
    if n_docs > WHOLE_CORPUS_FAST_PATH_THRESHOLD:
        return False
    return True

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

    # rroq158 / rroq4_riem GPU benchmarks re-encode the corpus on-the-fly
    # and only need the LEMUR routing artifacts. Reuse an existing fp16
    # build if available (codec-agnostic LEMUR per plan §4 — the routing
    # is trained on the FP16 corpus once per dataset and shared across
    # all codecs, so the BEIR sweep measures only the kernel/codec, not
    # routing variance).
    if params["compression"] in (Compression.RROQ158, Compression.RROQ4_RIEM):
        fp16_dir = INDEX_CACHE / name / f"s{n_shards}_{Compression.FP16.value}_{params['layout'].value}"
        if (fp16_dir / "manifest.json").exists():
            log.info("Reusing fp16 LEMUR artifacts at %s for %s",
                     fp16_dir, params["compression"].value)
            index_dir.parent.mkdir(parents=True, exist_ok=True)
            if index_dir.exists():
                shutil.rmtree(index_dir)
            # Hardlink instead of copy: rroq158 reuses fp16 LEMUR + corpus
            # artifacts unchanged and only adds its own rroq158_* cache files
            # on top, so the two dirs can safely share inodes. Saves 30-70 GB
            # per dataset on disk-tight benches (webis-touche2020 fp16 dir is
            # 41 GB; copytree previously doubled it).
            shutil.copytree(fp16_dir, index_dir, copy_function=os.link)
            return index_dir, 0.0

    npz_path = BEIR_CACHE / f"{name}.npz"
    t0 = time.time()
    built_dir = build(bcfg, npz_path=npz_path, device=device)
    build_elapsed = time.time() - t0

    if built_dir != index_dir:
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        if index_dir.exists():
            shutil.rmtree(index_dir)
        # Move rather than copy: built_dir is single-purpose, so renaming
        # avoids duplicating up to 70 GB on big corpora (webis-touche2020,
        # trec-covid). On the same filesystem this is an atomic dentry
        # rename.
        try:
            os.rename(str(built_dir), str(index_dir))
        except OSError:
            # Cross-device fallback (rare in practice; shard-bench cache is
            # one filesystem).
            shutil.copytree(built_dir, index_dir)
            shutil.rmtree(built_dir, ignore_errors=True)

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

    # Fast path: score every doc directly (skip LEMUR routing + candidate
    # gather) when the corpus is small enough that whole-corpus MaxSim is
    # faster than routing to a subset. Two triggers:
    #   (a) max_docs_exact >= n_docs  -- routing returns every doc anyway
    #   (b) n_docs <= WHOLE_CORPUS_FAST_PATH_THRESHOLD -- on an H100, the
    #       ~1 ms LEMUR route cost (FAISS ANN + MLP + candidate plan)
    #       dominates a whole-corpus MaxSim up to ~8 K docs. Below that
    #       threshold, skipping routing is both faster AND higher-
    #       quality (no router recall loss over the top-k), so this is
    #       a pure win. Above it, routing pays for itself by shrinking
    #       the kernel work from n_docs to max_docs_exact.
    #   The threshold is intentionally conservative; on datacenter GPUs
    #   (A100/H100) the crossover is typically 10-16 K docs, but the
    #   quality guarantee only holds strictly when we score everything,
    #   so we prefer to err on the side of routing for larger corpora.
    if gpu_corpus is not None:
        n_docs = len(gpu_corpus.doc_ids)
        # The PreloadedGpuCorpus already holds D + M tensors on device;
        # use the actual byte count it reports so the VRAM gate is
        # calibrated to the real footprint.
        corpus_bytes = getattr(gpu_corpus, "_gpu_bytes", None)
        if corpus_bytes is None:
            try:
                corpus_bytes = (
                    gpu_corpus.D.element_size() * gpu_corpus.D.numel()
                    + gpu_corpus.M.element_size() * gpu_corpus.M.numel()
                )
            except Exception:
                corpus_bytes = None
        if _should_use_fast_path(
            n_docs, params["max_docs_exact"], gpu_corpus_bytes=corpus_bytes,
        ):
            ids, scores = gpu_corpus.score_all(query, k=k, return_stats=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return ids, scores, elapsed_ms

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

    if params["compression"] == Compression.RROQ158:
        return _run_rroq158_gpu_mode(
            name, index_dir, all_vectors, doc_offsets, doc_ids, query_vecs,
            dim, params, n_warmup=n_warmup,
        )
    if params["compression"] == Compression.RROQ4_RIEM:
        return _run_rroq4_riem_gpu_mode(
            name, index_dir, all_vectors, doc_offsets, doc_ids, query_vecs,
            dim, params, n_warmup=n_warmup,
        )

    store = ShardStore(index_dir)

    doc_vecs = [all_vectors[s:e] for s, e in doc_offsets]
    gpu_corpus = PreloadedGpuCorpus(doc_vecs, doc_ids, dim, device=device)
    tok_counts = sorted({e - s for s, e in doc_offsets})
    q_tok_counts = sorted({int(qv.shape[0]) for qv in query_vecs})
    warmup_maxsim(
        dim=dim,
        doc_token_counts=tok_counts,
        device=device,
        query_token_counts=q_tok_counts,
    )

    # Defer the LEMUR router init until we know we need it: when the
    # corpus fits the fast-path criteria (`_should_use_fast_path` =
    # True), no per-query call ever touches the router, so loading the
    # MLP + FAISS index just to throw it away is ~3 s of wasted init
    # on every BEIR-class corpus. Skip it.
    skip_routing_global = _should_use_fast_path(
        len(doc_ids), params["max_docs_exact"],
    )
    router = None
    if not skip_routing_global:
        log.info("[GPU-corpus] %s: loading LEMUR router (slow-path active)", name)
        router = LemurRouter(
            index_dir / "lemur",
            ann_backend=params["ann_backend"].value,
            device=device,
        )
        router.load()
    else:
        log.info(
            "[GPU-corpus] %s: skipping LEMUR router init (fast-path: "
            "n_docs=%d ≤ threshold=%d, full-corpus MaxSim wins)",
            name, len(doc_ids), WHOLE_CORPUS_FAST_PATH_THRESHOLD,
        )

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


# ─────────────────────────────────────────────────────────────
# RROQ158 GPU-corpus mode (real LEMUR routing + Triton kernel)
# ─────────────────────────────────────────────────────────────


def _next_pow2_min(x: int, minimum: int = 32) -> int:
    v = max(x, minimum)
    p = 1
    while p < v:
        p <<= 1
    return p


_RROQ158_DISK_CACHE_DIR = Path(
    os.environ.get(
        "VOYAGER_RROQ158_CACHE",
        str(Path(__file__).resolve().parent / "data" / ".rroq158_cache"),
    )
)


def _rroq158_cache_key(
    all_vectors: np.ndarray, dim: int, cfg: "Rroq158Config",
) -> Path:
    """Stable per-corpus + per-config cache filename.

    Hash of (n_tokens, dim, K, group_size, seed, fingerprint of first/last
    rows + sum). Cheap to compute, collision-resistant in practice for
    BEIR-class corpora.
    """
    import hashlib

    h = hashlib.sha1()
    h.update(f"{all_vectors.shape[0]}|{dim}|{cfg.K}|{cfg.group_size}|"
             f"{cfg.seed}".encode())
    n = all_vectors.shape[0]
    head = all_vectors[: min(64, n)].tobytes()
    tail = all_vectors[max(0, n - 64):].tobytes()
    h.update(head)
    h.update(tail)
    h.update(np.float64(all_vectors.sum(dtype=np.float64)).tobytes())
    fname = f"rroq158_{cfg.K}_{cfg.group_size}_{cfg.seed}_{h.hexdigest()[:16]}.npz"
    return _RROQ158_DISK_CACHE_DIR / fname


def _warm_rroq158_triton_fallback(
    tiers: list, dim: int, n_words: int, n_groups: int, K: int, device: str,
) -> bool:
    """Pre-warm the Triton ``roq_maxsim_rroq158`` autotune cache for every
    tier T value in the payload, so the first query whose ``S>32`` (and
    therefore falls back from the fused CUDA b1 kernel) doesn't pay a
    ~7 s autotune cycle in-band.

    On quora ~99% of queries fit S<=32 (fused path, ~3 ms each), but a
    handful (~0.8% / 8 of 1000) have S in 33..48 → Triton fallback. Each
    unique tier-T autotune costs ~7 s in the original; running 4 tiers
    blocked the per-query budget for tens of seconds. Warming at
    build-time pushes that cost into the (already-paid) cold-start
    window.

    Returns True if warmup ran (always — no-op if no tiers).
    """
    try:
        from colsearch._internal.kernels.triton_roq_rroq158 import (
            roq_maxsim_rroq158,
        )
    except Exception:
        return False
    dev = torch.device(device)
    # Build a tiny S=33 query payload (just over the fused kernel's S=32
    # cap) so all tier-T autotunes get exercised under the same shape
    # the runtime fallback will hit.
    S = 33
    qp = torch.zeros((1, S, 4, n_words), dtype=torch.int32, device=dev)
    qm = torch.zeros((1, S, 2), dtype=torch.float32, device=dev)
    qct = torch.zeros((1, S, K), dtype=torch.float32, device=dev)
    for tier in tiers:
        T = int(tier["T"])
        # Use the first few rows of the tier's docs (no extra alloc) —
        # autotune picks configs based on T, dim, query_bits, K so the
        # actual data values don't matter.
        n_warm = min(8, int(tier.get("n_padded", tier["n"])))
        if n_warm == 0:
            continue
        ds = tier["sign"][:n_warm]
        dn = tier["nz"][:n_warm]
        dsc = tier["scl"][:n_warm]
        cid = tier["cid"][:n_warm]
        cosn = tier["cosn"][:n_warm]
        sinn = tier["sinn"][:n_warm]
        dm = tier["mask"][:n_warm]
        try:
            _ = roq_maxsim_rroq158(
                queries_planes=qp, queries_meta=qm, qc_table=qct,
                docs_centroid_id=cid, docs_cos_norm=cosn, docs_sin_norm=sinn,
                docs_sign=ds, docs_nz=dn, docs_scales=dsc,
                documents_mask=dm,
            )
        except Exception as exc:
            log.warning(
                "[rroq158] Triton fallback warmup failed for T=%d: %s",
                T, exc,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log.info(
        "[rroq158] Triton fallback warmed for S=33 across %d tier(s) "
        "(eliminates ~7 s in-band autotune for queries with S>32)",
        len(tiers),
    )
    return True


def _build_rroq158_gpu_payload(
    all_vectors: np.ndarray, doc_offsets: list, dim: int, params: dict, device: str,
):
    """Encode the corpus with rroq158 once and place into a multi-tier
    layout — one tensor bucket per unique ``next_pow2(token_count, 32)``.

    For token-uniform corpora (all docs ≤ p95, max ≤ pow2(p95)) the result
    collapses to a single tier, byte-identical to the pre-multi-tier
    payload. For tail-skewed corpora (e.g. quora: max=253 with 99 % of
    docs ≤32) it produces ~4 lean tiers and avoids the multi-GB padding
    waste that previously forced LEMUR routing on quora.

    The payload dict carries:
      ``tiers``: list of per-tier dicts {sign, nz, scl, cid, cosn, sinn,
                 mask, T, n, orig_idx (np int64)}
      ``centroids`` / ``centroids_np`` / ``rotator`` / ``fwht_seed``
      ``n_words`` / ``n_groups`` / ``K`` / ``dim``
      ``is_multi_tier`` (bool)
      ``orig_to_tier`` / ``orig_to_local`` (np int64) — for candidate
      partitioning in the LEMUR path.

    Legacy single-tier callers (when ``is_multi_tier=False``) still see
    the flat top-level ``sign / nz / ...`` keys.

    Disk cache: encoded planes / centroids / norms persist to
    ``$VOYAGER_RROQ158_CACHE`` (defaults to ``benchmarks/data/.rroq158_cache``)
    so the 4-min spherical-kmeans + chunked encode runs exactly once per
    (corpus, K, group_size, seed). Subsequent loads are a single
    ``np.load`` + dict reconstruction (~1 s for 8 M tokens).
    """
    cfg = Rroq158Config(
        K=params.get("rroq158_k", 1024),
        group_size=int(params.get("rroq158_group_size", 128)),
        seed=int(params.get("rroq158_seed", 42)),
    )

    cache_path = _rroq158_cache_key(all_vectors, dim, cfg)
    enc = None
    if cache_path.exists():
        try:
            t_load = time.perf_counter()
            data = np.load(cache_path, allow_pickle=False)
            from colsearch._internal.inference.quantization.rroq158 import (
                Rroq158Encoded,
            )
            enc = Rroq158Encoded(
                centroids=data["centroids"],
                centroid_id=data["centroid_id"],
                sign_plane=data["sign_plane"],
                nonzero_plane=data["nonzero_plane"],
                scales=data["scales"],
                cos_norm=data["cos_norm"],
                sin_norm=data["sin_norm"],
                fwht_seed=int(data["fwht_seed"]),
                dim=int(data["dim"]),
                group_size=int(data["group_size"]),
            )
            log.info(
                "[rroq158] cache HIT %s (%.2fs)",
                cache_path.name, time.perf_counter() - t_load,
            )
        except Exception as exc:
            log.warning("[rroq158] cache load failed (%s) — re-encoding", exc)
            enc = None

    if enc is None:
        log.info(
            "[rroq158] encoding %d tokens (dim=%d, K=%d, group_size=%d) "
            "→ cache %s",
            all_vectors.shape[0], dim, cfg.K, cfg.group_size, cache_path.name,
        )
        enc = encode_rroq158(np.asarray(all_vectors, dtype=np.float32), cfg)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                centroids=enc.centroids,
                centroid_id=enc.centroid_id,
                sign_plane=enc.sign_plane,
                nonzero_plane=enc.nonzero_plane,
                scales=enc.scales,
                cos_norm=enc.cos_norm,
                sin_norm=enc.sin_norm,
                fwht_seed=np.int64(enc.fwht_seed),
                dim=np.int64(enc.dim),
                group_size=np.int64(enc.group_size),
            )
            log.info("[rroq158] cache WRITE %s", cache_path.name)
        except Exception as exc:
            log.warning("[rroq158] cache write failed (%s) — proceeding", exc)

    n_words = enc.sign_plane.shape[1]
    n_groups = enc.scales.shape[1]
    if n_words % 4 != 0:
        raise RuntimeError(f"sign plane n_bytes={n_words} not multiple of 4")
    n_int32_words = n_words // 4

    n_docs = len(doc_offsets)
    tok_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int64)
    raw_max = int(tok_counts.max()) if n_docs else 1
    p95 = int(np.ceil(np.percentile(tok_counts, 95))) if n_docs else 32

    # Compute per-doc tier (pow2 of token count, min 32).
    per_doc_T = np.maximum(tok_counts, 1).astype(np.int64)
    per_doc_T = np.where(per_doc_T < 32, 32, per_doc_T)
    per_doc_T = (1 << np.ceil(np.log2(per_doc_T)).astype(np.int64)).astype(np.int64)

    unique_T = sorted(int(t) for t in np.unique(per_doc_T).tolist())
    T_to_count = {t: int((per_doc_T == t).sum()) for t in unique_T}

    # Merge tiers with <MIN_TIER_DOCS into next-larger tier.
    MIN_TIER_DOCS = 256
    MAX_TIERS = 8

    def _merge(tiers: list) -> list:
        if len(tiers) <= 1:
            return tiers
        out = []
        i = 0
        while i < len(tiers):
            t = tiers[i]
            if T_to_count[t] < MIN_TIER_DOCS and i + 1 < len(tiers):
                T_to_count[tiers[i + 1]] += T_to_count[t]
                i += 1
                continue
            out.append(t)
            i += 1
        return out

    tiers_T = unique_T
    while True:
        merged = _merge(tiers_T)
        if len(merged) == len(tiers_T):
            tiers_T = merged
            break
        tiers_T = merged

    # Cap tier count.
    if len(tiers_T) > MAX_TIERS:
        inner = sorted(((T_to_count[t], t) for t in tiers_T[:-1]))
        n_drop = len(tiers_T) - MAX_TIERS
        drop = {t for _, t in inner[:n_drop]}
        new_tiers = []
        for t in tiers_T:
            if t in drop:
                idx = tiers_T.index(t)
                nxt = next((u for u in tiers_T[idx + 1:] if u not in drop), None)
                if nxt is None:
                    new_tiers.append(t)
                else:
                    T_to_count[nxt] += T_to_count[t]
            else:
                new_tiers.append(t)
        tiers_T = new_tiers

    tiers_T_arr = np.array(sorted(tiers_T), dtype=np.int64)
    idx = np.searchsorted(tiers_T_arr, per_doc_T, side="left")
    idx = np.clip(idx, 0, len(tiers_T_arr) - 1)
    per_doc_tier = tiers_T_arr[idx]

    single_tier_bytes = n_docs * int(tiers_T_arr[-1]) * (
        2 * n_int32_words * 4 + n_groups * 4 + 4 + 4 + 4 + 4
    )
    bucketed_bytes = 0
    for t in tiers_T_arr:
        bucketed_bytes += int((per_doc_tier == t).sum()) * int(t) * (
            2 * n_int32_words * 4 + n_groups * 4 + 4 + 4 + 4 + 4
        )

    is_multi_tier = (
        len(tiers_T_arr) > 1
        and n_docs >= 1024
        and bucketed_bytes < 0.9 * single_tier_bytes
    )

    log.info(
        "[rroq158] tier layout: %d tier(s) %s, raw_max=%d p95=%d, "
        "%.2f GB total (vs %.2f GB single-tier, %.1fx leaner)%s",
        len(tiers_T_arr), tiers_T_arr.tolist(), raw_max, p95,
        bucketed_bytes / 1e9, single_tier_bytes / 1e9,
        single_tier_bytes / max(bucketed_bytes, 1),
        " MULTI-TIER ACTIVE" if is_multi_tier else " (single-tier)",
    )

    from colsearch._internal.inference.quantization.rroq158 import (
        get_cached_fwht_rotator,
    )

    centroids_np = np.ascontiguousarray(enc.centroids, dtype=np.float32)
    rotator = get_cached_fwht_rotator(dim=dim, seed=enc.fwht_seed)

    bytes_per_tok = (
        2 * n_words + n_groups * 2 + 2 + 2 + 2
    )

    if not is_multi_tier:
        # Legacy single-tier path: pad to the largest tier (== pow2(raw_max))
        # AND pre-pad B to multiple of 8 for the fused b1 kernel — same
        # rationale as the multi-tier path: avoid the 1+ GB ``_pad_b``
        # allocation in score_b1_fused on every query.
        t_max = int(tiers_T_arr[-1])
        on_cuda = str(device).startswith("cuda")
        pad_b = (8 - (n_docs % 8)) % 8
        n_padded = n_docs + pad_b
        log.info(
            "[rroq158] padding (single-tier) to T_max=%d, B=%d (padded from %d)",
            t_max, n_padded, n_docs,
        )
        sign_dt = np.zeros((n_padded, t_max, n_int32_words), dtype=np.int32)
        nz_dt = np.zeros((n_padded, t_max, n_int32_words), dtype=np.int32)
        scl_dt = np.zeros((n_padded, t_max, n_groups), dtype=np.float32)
        cid_dt = np.zeros((n_padded, t_max), dtype=np.int32)
        cosn_dt = np.zeros((n_padded, t_max), dtype=np.float32)
        sinn_dt = np.zeros((n_padded, t_max), dtype=np.float32)
        mask_dt = np.zeros((n_padded, t_max), dtype=np.float32)
        for di, (s, e) in enumerate(doc_offsets):
            n_tok = min(e - s, t_max)
            sign_words = pack_doc_codes_to_int32_words(enc.sign_plane[s:s + n_tok])
            nz_words = pack_doc_codes_to_int32_words(enc.nonzero_plane[s:s + n_tok])
            sign_dt[di, :n_tok] = sign_words
            nz_dt[di, :n_tok] = nz_words
            scl_dt[di, :n_tok] = enc.scales[s:s + n_tok].astype(np.float32)
            cid_dt[di, :n_tok] = enc.centroid_id[s:s + n_tok].astype(np.int32)
            cosn_dt[di, :n_tok] = enc.cos_norm[s:s + n_tok].astype(np.float32)
            sinn_dt[di, :n_tok] = enc.sin_norm[s:s + n_tok].astype(np.float32)
            mask_dt[di, :n_tok] = 1.0

        scl_torch = torch.from_numpy(scl_dt).to(device)
        return {
            "sign": torch.from_numpy(sign_dt).to(device),
            "nz": torch.from_numpy(nz_dt).to(device),
            "scl": scl_torch,
            "cid": torch.from_numpy(cid_dt).to(device),
            "cosn": torch.from_numpy(cosn_dt).to(device),
            "sinn": torch.from_numpy(sinn_dt).to(device),
            "mask": torch.from_numpy(mask_dt).to(device),
            "scl_2d": (
                scl_torch[..., 0].contiguous() if (on_cuda and n_groups == 1) else None
            ),
            "centroids": torch.from_numpy(enc.centroids).to(device),
            "centroids_np": centroids_np,
            "rotator": rotator,
            "fwht_seed": enc.fwht_seed,
            "n_words": n_int32_words,
            "n_groups": n_groups,
            "K": cfg.K,
            "t_max": t_max,
            "dim": dim,
            "bytes_per_tok": bytes_per_tok,
            "is_multi_tier": False,
            "tiers": None,
            "n_docs": n_docs,
            "n_padded": n_padded,
            "pad_b": pad_b,
        }

    # ── Multi-tier path ────────────────────────────────────────────────
    on_cuda = str(device).startswith("cuda")
    tiers = []
    orig_to_tier = np.zeros(n_docs, dtype=np.int64)
    orig_to_local = np.zeros(n_docs, dtype=np.int64)
    for ti, T_tier in enumerate(tiers_T_arr):
        T_tier = int(T_tier)
        sel_orig = np.flatnonzero(per_doc_tier == T_tier).astype(np.int64)
        n_tier_actual = sel_orig.size
        # Pre-pad to a multiple of 8 in the B (doc) dimension so the
        # ``cuda_b1_rroq158.score_b1_fused`` kernel doesn't have to
        # allocate ~1.2 GB of padded tensors *per query* (the m8n8k128
        # binary tensor-core tile requires B % 8 == 0). The padded rows
        # all carry mask=0 so they contribute -inf to the per-query
        # max and never enter the top-k. This single payload-time pad
        # eliminates the ~5 GB / query allocator churn that was causing
        # 7-8 s GC stalls every ~200 queries on quora.
        pad_b = (8 - (n_tier_actual % 8)) % 8
        n_tier = n_tier_actual + pad_b
        sign_dt = np.zeros((n_tier, T_tier, n_int32_words), dtype=np.int32)
        nz_dt = np.zeros((n_tier, T_tier, n_int32_words), dtype=np.int32)
        scl_dt = np.zeros((n_tier, T_tier, n_groups), dtype=np.float32)
        cid_dt = np.zeros((n_tier, T_tier), dtype=np.int32)
        cosn_dt = np.zeros((n_tier, T_tier), dtype=np.float32)
        sinn_dt = np.zeros((n_tier, T_tier), dtype=np.float32)
        mask_dt = np.zeros((n_tier, T_tier), dtype=np.float32)
        for li, oi in enumerate(sel_orig):
            s, e = doc_offsets[int(oi)]
            n_tok = min(e - s, T_tier)
            sign_words = pack_doc_codes_to_int32_words(enc.sign_plane[s:s + n_tok])
            nz_words = pack_doc_codes_to_int32_words(enc.nonzero_plane[s:s + n_tok])
            sign_dt[li, :n_tok] = sign_words
            nz_dt[li, :n_tok] = nz_words
            scl_dt[li, :n_tok] = enc.scales[s:s + n_tok].astype(np.float32)
            cid_dt[li, :n_tok] = enc.centroid_id[s:s + n_tok].astype(np.int32)
            cosn_dt[li, :n_tok] = enc.cos_norm[s:s + n_tok].astype(np.float32)
            sinn_dt[li, :n_tok] = enc.sin_norm[s:s + n_tok].astype(np.float32)
            mask_dt[li, :n_tok] = 1.0
        # mask rows for padded docs stay 0.0 (initialised that way).
        orig_to_tier[sel_orig] = ti
        orig_to_local[sel_orig] = np.arange(n_tier_actual, dtype=np.int64)

        # ``torch.from_numpy`` is zero-copy on CPU; CPU tensors keep numpy
        # views available too for the np-fast-path in `_rroq158_score_candidates`.
        tier_payload = {
            "T": T_tier,
            "n": n_tier_actual,    # logical doc count (un-padded)
            "n_padded": n_tier,    # physical doc count (multiple of 8)
            "pad_b": pad_b,
            "orig_idx": sel_orig,  # np int64 — length n_tier_actual
            "sign": torch.from_numpy(sign_dt).to(device),
            "nz": torch.from_numpy(nz_dt).to(device),
            "scl": torch.from_numpy(scl_dt).to(device),
            "cid": torch.from_numpy(cid_dt).to(device),
            "cosn": torch.from_numpy(cosn_dt).to(device),
            "sinn": torch.from_numpy(sinn_dt).to(device),
            "mask": torch.from_numpy(mask_dt).to(device),
        }
        # Pre-convert orig_idx to a device tensor *once* per tier so the
        # per-query scatter doesn't pay 4× ``torch.from_numpy(...).to(device)``
        # every iteration (was ~150 µs/tier × 4 tiers × 500 queries = 300 ms
        # of pure dispatch overhead per benchmark block).
        if on_cuda:
            tier_payload["orig_idx_dev"] = torch.from_numpy(
                np.ascontiguousarray(sel_orig)
            ).to(device)
            # Pre-materialise the squeezed (B, T) scale tensor so the
            # fused kernel doesn't pay a 65 MB ``contiguous()`` per
            # query (4 tiers × ~65 MB = 260 MB allocator churn / query).
            # group_size=128 ⇒ n_groups=1, so the [..., 0] slice is the
            # full payload and we materialise it once here.
            if n_groups == 1:
                tier_payload["scl_2d"] = (
                    tier_payload["scl"][..., 0].contiguous()
                )
        if not on_cuda:
            # Stash zero-copy numpy aliases (used by `_rroq158_score_candidates`
            # CPU gather path) so we never re-materialise into pytorch storage.
            tier_payload["sign_np"] = tier_payload["sign"].numpy()
            tier_payload["nz_np"] = tier_payload["nz"].numpy()
            tier_payload["scl_np"] = tier_payload["scl"].numpy()
            tier_payload["cid_np"] = tier_payload["cid"].numpy()
            tier_payload["cosn_np"] = tier_payload["cosn"].numpy()
            tier_payload["sinn_np"] = tier_payload["sinn"].numpy()
            tier_payload["mask_np"] = tier_payload["mask"].numpy()
        tiers.append(tier_payload)

    return {
        # Aliases at top-level (point at the largest tier so existing
        # legacy code that grabs payload["sign"] keeps working but only
        # sees a fraction of the corpus — multi-tier-aware callers must
        # check is_multi_tier and iterate ``tiers``).
        "sign": tiers[-1]["sign"],
        "nz": tiers[-1]["nz"],
        "scl": tiers[-1]["scl"],
        "cid": tiers[-1]["cid"],
        "cosn": tiers[-1]["cosn"],
        "sinn": tiers[-1]["sinn"],
        "mask": tiers[-1]["mask"],
        "centroids": torch.from_numpy(enc.centroids).to(device),
        "centroids_np": centroids_np,
        "rotator": rotator,
        "fwht_seed": enc.fwht_seed,
        "n_words": n_int32_words,
        "n_groups": n_groups,
        "K": cfg.K,
        "t_max": int(tiers_T_arr[-1]),
        "dim": dim,
        "bytes_per_tok": bytes_per_tok,
        "is_multi_tier": True,
        "tiers": tiers,
        "orig_to_tier": orig_to_tier,
        "orig_to_local": orig_to_local,
        "n_docs": n_docs,
        "_warmed_triton_S": _warm_rroq158_triton_fallback(
            tiers, dim, n_int32_words, n_groups, cfg.K, device,
        ) if on_cuda else False,
        # Persistent scratch buffers for the per-query hot path:
        #   * ``scores_buf_dev``: (n_docs,) float32 scratchpad scattered
        #     into per-tier; ``fill_(-inf)`` re-initialises in-place per
        #     query, eliminating the 2 MB ``torch.full`` per query that
        #     was the dominant per-query allocation in
        #     ``_rroq158_score_all`` (was 4 ms / 500-query block on
        #     quora due to allocator pressure).
        #   * ``paired_pinned``: pinned-host ``(2, k_max)`` buffer for
        #     the top-k D2H copy so the ``stack().cpu().numpy()``
        #     roundtrip becomes a single async copy + sync.
        "scores_buf_dev": (
            torch.full(
                (n_docs,), float("-inf"),
                dtype=torch.float32, device=torch.device(device),
            ) if on_cuda else None
        ),
        "paired_pinned": (
            torch.empty(
                (2, 64), dtype=torch.float32, pin_memory=True,
            ) if on_cuda else None
        ),
    }


def _rroq158_score_all(
    query_np: np.ndarray, payload: dict, doc_ids: list, k: int, device: str,
):
    """Whole-corpus rroq158 MaxSim — skips the Python candidate-id list
    and the seven per-query ``index_select`` copies. Used when the
    caller would route to every doc anyway (``max_docs_exact >= n_docs``).

    Multi-tier aware: when ``payload["is_multi_tier"]`` is True, runs
    one ``score_rroq158_topk`` (without topk extraction) per tier and
    scatter-merges the scores into a ``(n_docs,)`` tensor before the
    final top-k.
    """
    on_cuda = str(device).startswith("cuda")
    if not on_cuda:
        # CPU whole-corpus fast path. The generic _rroq158_score_candidates
        # would do a 522k-row numpy fancy-index gather per tier per query,
        # which is what made quora rroq158/cpu hang for 90+ minutes. Here
        # we bypass the gather entirely and call the module-level
        # score_rroq158_topk per tier directly on the cached tier numpy
        # arrays (no allocation), then merge top-k across tiers.
        centroids_np = payload.get("centroids_np")
        if centroids_np is None:
            centroids_np = payload["centroids"].cpu().numpy()
            payload["centroids_np"] = centroids_np
        q_inputs = encode_query_for_rroq158(
            query_np, centroids_np,
            fwht_seed=payload["fwht_seed"], query_bits=4,
            rotator=payload.get("rotator"),
            cap_blas_threads=True,
        )
        q_planes = torch.from_numpy(q_inputs["q_planes"][None, :, :, :])
        q_meta = torch.from_numpy(q_inputs["q_meta"][None, :, :])
        qc_table = torch.from_numpy(q_inputs["qc_table"][None, :, :])

        if payload.get("is_multi_tier", False):
            # Score each tier on the full tier corpus (no per-query gather).
            best_ids: list = []
            best_scores: list = []
            for ti, tier in enumerate(payload["tiers"]):
                # Per-tier numpy arrays are produced once at build time;
                # ensure they exist (the .numpy() call is zero-copy on CPU).
                sign_np = tier.get("sign_np")
                if sign_np is None:
                    sign_np = tier["sign"].numpy(); tier["sign_np"] = sign_np
                nz_np = tier.get("nz_np")
                if nz_np is None:
                    nz_np = tier["nz"].numpy(); tier["nz_np"] = nz_np
                scl_np = tier.get("scl_np")
                if scl_np is None:
                    scl_np = tier["scl"].numpy(); tier["scl_np"] = scl_np
                cid_np = tier.get("cid_np")
                if cid_np is None:
                    cid_np = tier["cid"].numpy(); tier["cid_np"] = cid_np
                cosn_np = tier.get("cosn_np")
                if cosn_np is None:
                    cosn_np = tier["cosn"].numpy(); tier["cosn_np"] = cosn_np
                sinn_np = tier.get("sinn_np")
                if sinn_np is None:
                    sinn_np = tier["sinn"].numpy(); tier["sinn_np"] = sinn_np
                mask_np = tier.get("mask_np")
                if mask_np is None:
                    mask_np = tier["mask"].numpy(); tier["mask_np"] = mask_np
                # Original-corpus doc ids for this tier's rows.
                tier_doc_ids = tier.get("doc_ids_for_tier")
                if tier_doc_ids is None:
                    orig_idx = tier["orig_idx"]  # (n_tier_docs,) -> orig position
                    if hasattr(orig_idx, "cpu"):
                        orig_idx_np = orig_idx.cpu().numpy()
                    else:
                        orig_idx_np = orig_idx
                    tier_doc_ids = [doc_ids[int(p)] for p in orig_idx_np[: tier["n"]]]
                    tier["doc_ids_for_tier"] = tier_doc_ids
                t_ids, t_scores = score_rroq158_topk(
                    q_planes, q_meta, qc_table,
                    torch.from_numpy(cid_np[: tier["n"]]),
                    torch.from_numpy(cosn_np[: tier["n"]]),
                    torch.from_numpy(sinn_np[: tier["n"]]),
                    torch.from_numpy(sign_np[: tier["n"]]),
                    torch.from_numpy(nz_np[: tier["n"]]),
                    torch.from_numpy(scl_np[: tier["n"]]),
                    doc_ids=tier_doc_ids,
                    k=min(k, tier["n"]),
                    documents_mask=torch.from_numpy(mask_np[: tier["n"]]),
                    device=torch.device(device),
                )
                best_ids.extend(t_ids)
                best_scores.extend(t_scores)
            # Final top-k merge across tiers.
            if not best_ids:
                return [], []
            order = np.argsort(np.asarray(best_scores, dtype=np.float32))[::-1][:k]
            return [best_ids[int(i)] for i in order], [float(best_scores[int(i)]) for i in order]

        # Single-tier CPU whole-corpus path.
        sign_np = payload.setdefault("sign_np", payload["sign"].numpy())
        nz_np = payload.setdefault("nz_np", payload["nz"].numpy())
        scl_np = payload.setdefault("scl_np", payload["scl"].numpy())
        cid_np = payload.setdefault("cid_np", payload["cid"].numpy())
        cosn_np = payload.setdefault("cosn_np", payload["cosn"].numpy())
        sinn_np = payload.setdefault("sinn_np", payload["sinn"].numpy())
        mask_np = payload.setdefault("mask_np", payload["mask"].numpy())
        return score_rroq158_topk(
            q_planes, q_meta, qc_table,
            torch.from_numpy(cid_np), torch.from_numpy(cosn_np),
            torch.from_numpy(sinn_np), torch.from_numpy(sign_np),
            torch.from_numpy(nz_np), torch.from_numpy(scl_np),
            doc_ids=doc_ids, k=k,
            documents_mask=torch.from_numpy(mask_np),
            device=torch.device(device),
        )
    q_inputs = encode_query_for_rroq158(
        query_np, None,
        fwht_seed=payload["fwht_seed"], query_bits=4,
        rotator=payload.get("rotator"),
        skip_qc_table=True,
        cap_blas_threads=False,
    )
    # Pad query-side tensors to S=32 ONCE here (fused kernel hard-
    # specialises to S<=32). Without this, score_b1_fused did the
    # _pad_s allocation 4× per query (once per tier) — a 1 MB qc_table
    # alloc each time, ~4 MB per query, ~2 GB churn per 500 queries
    # which forced a CUDA allocator GC every ~200 queries (the 7-11 s
    # stall we saw at block 300).
    S_PAD = 32
    s_raw = q_inputs["q_planes"].shape[0]
    s_pad = S_PAD if s_raw < S_PAD else s_raw  # only pad up to 32

    # Persistent device scratchpads keyed by s_pad. ``payload["_q_scratch"]``
    # caches (q_planes_buf, q_meta_buf, qc_table_buf) for each S so a
    # ``copy_`` reuses the device buffer instead of allocating anew.
    q_scratch = payload.setdefault("_q_scratch", {})
    sb = q_scratch.get(s_pad)
    if sb is None:
        n_words_local = int(payload["n_words"])
        K_local = int(payload["K"])
        dim_local = int(payload["dim"])
        sb = {
            "qp": torch.zeros((s_pad, 4, n_words_local), dtype=torch.int32, device=device),
            "qm": torch.zeros((s_pad, 2), dtype=torch.float32, device=device),
            "qc": torch.zeros((s_pad, K_local), dtype=torch.float32, device=device),
            "qd": torch.zeros((s_pad, dim_local), dtype=torch.float32, device=device),
        }
        q_scratch[s_pad] = sb

    qp_buf = sb["qp"]
    qm_buf = sb["qm"]
    qc_buf = sb["qc"]
    qd_buf = sb["qd"]
    # Stage host → device through the persistent buffers (rows beyond
    # s_raw stay zero from prior allocation; the kernel treats them as
    # contributing 0 to the score, exactly what we want for padding).
    qp_buf[:s_raw].copy_(torch.from_numpy(q_inputs["q_planes"]), non_blocking=True)
    qm_buf[:s_raw].copy_(torch.from_numpy(q_inputs["q_meta"]), non_blocking=True)
    qd_buf[:s_raw].copy_(
        torch.from_numpy(np.ascontiguousarray(query_np, dtype=np.float32)),
        non_blocking=True,
    )
    if s_raw < s_pad:
        # Re-zero the trailing rows (in case prior query had larger s_raw
        # so its data now sits in our padding region).
        qp_buf[s_raw:].zero_()
        qm_buf[s_raw:].zero_()
        qd_buf[s_raw:].zero_()
    # qc_table = q @ centroids.T using out= to reuse the buffer (no alloc).
    torch.matmul(qd_buf, payload["centroids"].T, out=qc_buf)
    q_planes = qp_buf.unsqueeze(0)
    q_meta = qm_buf.unsqueeze(0)
    qc_table = qc_buf.unsqueeze(0)

    if not payload.get("is_multi_tier", False):
        return score_rroq158_topk(
            q_planes, q_meta, qc_table,
            payload["cid"], payload["cosn"], payload["sinn"],
            payload["sign"], payload["nz"], payload["scl"],
            doc_ids=doc_ids,
            k=k, documents_mask=payload["mask"], device=torch.device(device),
        )

    # Multi-tier path: score every tier, scatter into a single
    # (n_docs,) buffer, then top-k once.
    n_docs = int(payload["n_docs"])

    # Reuse the persistent device buffer; reset to -inf in-place.
    scores_buf = payload.get("scores_buf_dev")
    if scores_buf is None or scores_buf.numel() != n_docs:
        scores_buf = torch.full(
            (n_docs,), float("-inf"),
            dtype=torch.float32, device=torch.device(device),
        )
        payload["scores_buf_dev"] = scores_buf
    else:
        scores_buf.fill_(float("-inf"))

    tiers = payload["tiers"]
    for tier in tiers:
        if tier["n"] == 0:
            continue
        tier_scores = _rroq158_tier_scores(
            q_planes, q_meta, qc_table,
            tier, device,
        )
        # Pre-converted device tensor (built once at payload-build time).
        orig_idx_t = tier.get("orig_idx_dev")
        if orig_idx_t is None:
            orig_idx_t = torch.from_numpy(
                np.ascontiguousarray(tier["orig_idx"])
            ).to(torch.device(device))
            tier["orig_idx_dev"] = orig_idx_t
        scores_buf.scatter_(0, orig_idx_t, tier_scores)

    final_k = min(k, n_docs)
    top_sc, top_idx = scores_buf.topk(final_k)

    # Single fused (idx, score) D2H via the pinned-host buffer when
    # available — avoids the two .cpu() roundtrips inherent in the
    # ``stack().cpu().numpy()`` chain.
    paired_pinned = payload.get("paired_pinned")
    if paired_pinned is not None and final_k <= paired_pinned.shape[1]:
        paired = torch.stack(
            [top_idx.to(torch.float32), top_sc.to(torch.float32)], dim=0,
        )
        paired_pinned[:, :final_k].copy_(paired, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        idx_arr = paired_pinned[0, :final_k].to(torch.int64).tolist()
        sc_arr = paired_pinned[1, :final_k].tolist()
        return [doc_ids[i] for i in idx_arr], sc_arr
    paired = torch.stack(
        [top_idx.to(torch.float32), top_sc.to(torch.float32)], dim=0,
    ).cpu().numpy()
    idx_list = paired[0].astype(np.int64).tolist()
    return [doc_ids[i] for i in idx_list], paired[1].tolist()


def _rroq158_tier_scores(
    q_planes: torch.Tensor,
    q_meta: torch.Tensor,
    qc_table: torch.Tensor,
    tier: dict,
    device: str,
) -> torch.Tensor:
    """Run the rroq158 MaxSim kernel against a single tier's tensors and
    return ``(n_tier,)`` float32 scores (no topk).

    Mirrors `score_rroq158_topk` minus the topk + D2H tail so the caller
    can scatter into a multi-tier scores buffer.
    """
    from colsearch._internal.kernels import cuda_b1_rroq158
    from colsearch._internal.kernels.triton_roq_rroq158 import (
        roq_maxsim_rroq158,
    )

    ds = tier["sign"]
    dn = tier["nz"]
    dsc = tier["scl"]
    cid = tier["cid"]
    cos_n = tier["cosn"]
    sin_n = tier["sinn"]
    dm = tier["mask"]

    qp_one = q_planes[0] if q_planes.dim() == 4 else q_planes
    qm_one = q_meta[0] if q_meta.dim() == 3 else q_meta
    qct_one = qc_table[0] if qc_table.dim() == 3 else qc_table
    n_actual = int(tier.get("n", ds.shape[0]))
    scl_2d = tier.get("scl_2d")
    scores = None
    try:
        scores = cuda_b1_rroq158.score_b1_fused(
            docs_sign=ds, docs_nz=dn, docs_scl=dsc,
            docs_cid=cid, docs_cos=cos_n, docs_sin=sin_n,
            docs_mask=dm,
            q_planes=qp_one, q_meta=qm_one, qc_table=qct_one,
            docs_scl_2d=scl_2d,
        )
        _log_fused_used(int(ds.shape[0]), int(ds.shape[1]), "fused")
    except Exception as exc:
        _log_fused_used(int(ds.shape[0]), int(ds.shape[1]), f"FALLBACK: {type(exc).__name__}: {exc}")
        scores = None
    if scores is None:
        scores = roq_maxsim_rroq158(
            queries_planes=q_planes,
            queries_meta=q_meta,
            qc_table=qc_table,
            docs_centroid_id=cid,
            docs_cos_norm=cos_n,
            docs_sin_norm=sin_n,
            docs_sign=ds,
            docs_nz=dn,
            docs_scales=dsc,
            documents_mask=dm,
        ).squeeze(0)
    # Trim trailing pad-rows (mask=0 ⇒ scores will be 0/-inf) so the
    # scatter into the (n_docs,) buffer is sized to the un-padded tier.
    scores = scores[:n_actual]
    return scores.to(torch.float32)


def _rroq158_score_candidates(
    query_np: np.ndarray, payload: dict, candidate_ids: list, doc_ids_to_idx: dict,
    k: int, device: str,
):
    """Score one query against the rroq158 GPU payload at the given candidate
    doc indices. Returns (top_ids, top_scores).

    Multi-tier aware: when ``payload["is_multi_tier"]`` is True, partitions
    candidates by their tier assignment, runs one kernel per tier, and
    merges scores.

    CPU path note (audit_rroq158_cpu_2026q2): on CPU we must NOT use
    ``torch.index_select`` for the candidate gather. torch's intra-op
    thread pool defaults to ``cpu_count()`` workers (64 on a 128-core
    box), and when the gather runs concurrently with the rayon-parallel
    Rust kernel it triggers catastrophic context-switch / cache-eviction
    churn — the same gather that takes ~0.3 ms in numpy takes 90+ ms
    when followed by the Rust kernel under torch's default threading
    (measured 2026-04-19 against 2000 candidates × 32 doc-tok). We
    sidestep the entire torch intra-op pool by gathering with numpy
    fancy indexing on the underlying numpy arrays cached in ``payload``.
    """
    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        # qc_table = q @ centroids.T scales linearly with K and dominates
        # CPU prep time at K>=2048 (~1 ms / 33M FLOP at K=8192). On the GPU
        # the same matmul is ~0.005 ms, so we move it to device and skip
        # the host-side computation in the encode helper.
        q_inputs = encode_query_for_rroq158(
            query_np, None,
            fwht_seed=payload["fwht_seed"], query_bits=4,
            rotator=payload.get("rotator"),
            skip_qc_table=True,
            cap_blas_threads=False,
        )
        q_planes = torch.from_numpy(q_inputs["q_planes"][None, :, :, :]).to(device)
        q_meta = torch.from_numpy(q_inputs["q_meta"][None, :, :]).to(device)
        q_dev = torch.from_numpy(np.ascontiguousarray(query_np, dtype=np.float32)).to(device)
        qc_table = (q_dev @ payload["centroids"].T).unsqueeze(0)

        # Multi-tier-aware candidate path: partition candidates by tier,
        # score each non-empty tier separately, merge scores back in
        # candidate order before topk.
        if payload.get("is_multi_tier", False):
            n_cand = len(candidate_ids)
            # Per-candidate (tier_id, local_idx) lookup.
            orig_to_tier = payload["orig_to_tier"]
            orig_to_local = payload["orig_to_local"]
            cand_orig = np.fromiter(
                (doc_ids_to_idx[int(cid)] for cid in candidate_ids),
                dtype=np.int64, count=n_cand,
            )
            cand_tier = orig_to_tier[cand_orig]
            cand_local = orig_to_local[cand_orig]
            scores_buf = torch.full(
                (n_cand,), float("-inf"),
                dtype=torch.float32, device=torch.device(device),
            )
            for ti, tier in enumerate(payload["tiers"]):
                pos = np.flatnonzero(cand_tier == ti).astype(np.int64)
                if pos.size == 0:
                    continue
                local = cand_local[pos]
                local_t = torch.from_numpy(local).to(torch.device(device))
                tier_slice = {
                    "n": int(pos.size),
                    "T": tier["T"],
                    "sign": tier["sign"].index_select(0, local_t),
                    "nz": tier["nz"].index_select(0, local_t),
                    "scl": tier["scl"].index_select(0, local_t),
                    "cid": tier["cid"].index_select(0, local_t),
                    "cosn": tier["cosn"].index_select(0, local_t),
                    "sinn": tier["sinn"].index_select(0, local_t),
                    "mask": tier["mask"].index_select(0, local_t),
                    "orig_idx": pos,
                }
                tier_scores = _rroq158_tier_scores(
                    q_planes, q_meta, qc_table, tier_slice, device,
                )
                pos_t = torch.from_numpy(pos).to(torch.device(device))
                scores_buf.scatter_(0, pos_t, tier_scores)
            final_k = min(k, n_cand)
            top_sc, top_idx = scores_buf.topk(final_k)
            paired = torch.stack(
                [top_idx.to(torch.float32), top_sc.to(torch.float32)], dim=0,
            ).cpu().numpy()
            idx_list = paired[0].astype(np.int64).tolist()
            return [candidate_ids[i] for i in idx_list], paired[1].tolist()

        cand_idx = torch.tensor(
            [doc_ids_to_idx[int(cid)] for cid in candidate_ids],
            dtype=torch.long, device=device,
        )
        sign_b = payload["sign"].index_select(0, cand_idx)
        nz_b = payload["nz"].index_select(0, cand_idx)
        scl_b = payload["scl"].index_select(0, cand_idx)
        cid_b = payload["cid"].index_select(0, cand_idx)
        cosn_b = payload["cosn"].index_select(0, cand_idx)
        sinn_b = payload["sinn"].index_select(0, cand_idx)
        mask_b = payload["mask"].index_select(0, cand_idx)
    else:
        centroids_np = payload.get("centroids_np")
        if centroids_np is None:
            centroids_np = payload["centroids"].cpu().numpy()
            payload["centroids_np"] = centroids_np
        q_inputs = encode_query_for_rroq158(
            query_np, centroids_np,
            fwht_seed=payload["fwht_seed"], query_bits=4,
            rotator=payload.get("rotator"),
        )
        q_planes = torch.from_numpy(q_inputs["q_planes"][None, :, :, :])
        q_meta = torch.from_numpy(q_inputs["q_meta"][None, :, :])
        qc_table = torch.from_numpy(q_inputs["qc_table"][None, :, :])

        # Multi-tier CPU candidate path: partition + score per-tier on CPU.
        if payload.get("is_multi_tier", False):
            n_cand = len(candidate_ids)
            orig_to_tier = payload["orig_to_tier"]
            orig_to_local = payload["orig_to_local"]
            cand_orig = np.fromiter(
                (doc_ids_to_idx[int(cid)] for cid in candidate_ids),
                dtype=np.int64, count=n_cand,
            )
            cand_tier = orig_to_tier[cand_orig]
            cand_local = orig_to_local[cand_orig]
            merged_ids: list = [None] * n_cand
            scores_arr = np.full(n_cand, -np.inf, dtype=np.float32)
            for ti, tier in enumerate(payload["tiers"]):
                pos = np.flatnonzero(cand_tier == ti)
                if pos.size == 0:
                    continue
                local = cand_local[pos]
                sign_b = torch.from_numpy(tier["sign_np"][local])
                nz_b = torch.from_numpy(tier["nz_np"][local])
                scl_b = torch.from_numpy(tier["scl_np"][local])
                cid_b = torch.from_numpy(tier["cid_np"][local])
                cosn_b = torch.from_numpy(tier["cosn_np"][local])
                sinn_b = torch.from_numpy(tier["sinn_np"][local])
                mask_b = torch.from_numpy(tier["mask_np"][local])
                tier_cand_ids = [candidate_ids[int(p)] for p in pos]
                t_ids, t_scores = score_rroq158_topk(
                    q_planes, q_meta, qc_table,
                    cid_b, cosn_b, sinn_b, sign_b, nz_b, scl_b,
                    doc_ids=tier_cand_ids,
                    k=int(pos.size), documents_mask=mask_b,
                    device=torch.device(device),
                )
                # Insert returned (id, score) pairs at the original
                # candidate position.
                tier_id_to_pos = {tid: int(pos[i]) for i, tid in enumerate(tier_cand_ids)}
                for ret_id, ret_sc in zip(t_ids, t_scores):
                    p = tier_id_to_pos[ret_id]
                    scores_arr[p] = ret_sc
                    merged_ids[p] = ret_id
            top_idx = np.argsort(-scores_arr)[: min(k, n_cand)]
            return (
                [candidate_ids[int(i)] for i in top_idx],
                [float(scores_arr[int(i)]) for i in top_idx],
            )

        # Cache numpy views over the payload's torch.from_numpy-backed
        # tensors. ``.numpy()`` on a CPU torch tensor is a zero-copy view,
        # so the cache only matters for ergonomics — the gather itself is
        # the hot path.
        sign_np = payload.setdefault("sign_np", payload["sign"].numpy())
        nz_np = payload.setdefault("nz_np", payload["nz"].numpy())
        scl_np = payload.setdefault("scl_np", payload["scl"].numpy())
        cid_np = payload.setdefault("cid_np", payload["cid"].numpy())
        cosn_np = payload.setdefault("cosn_np", payload["cosn"].numpy())
        sinn_np = payload.setdefault("sinn_np", payload["sinn"].numpy())
        mask_np = payload.setdefault("mask_np", payload["mask"].numpy())

        cand_np = np.fromiter(
            (doc_ids_to_idx[int(cid)] for cid in candidate_ids),
            dtype=np.int64, count=len(candidate_ids),
        )
        # Numpy fancy indexing returns a fresh contiguous array — these
        # arrays then flow into ``_score_rroq158_cpu`` via
        # ``torch.from_numpy`` (zero-copy view) and the zero-copy fast
        # path in ``_to_np`` keeps them in numpy storage all the way
        # into the Rust kernel call.
        sign_b = torch.from_numpy(sign_np[cand_np])
        nz_b = torch.from_numpy(nz_np[cand_np])
        scl_b = torch.from_numpy(scl_np[cand_np])
        cid_b = torch.from_numpy(cid_np[cand_np])
        cosn_b = torch.from_numpy(cosn_np[cand_np])
        sinn_b = torch.from_numpy(sinn_np[cand_np])
        mask_b = torch.from_numpy(mask_np[cand_np])

    return score_rroq158_topk(
        q_planes, q_meta, qc_table,
        cid_b, cosn_b, sinn_b, sign_b, nz_b, scl_b,
        doc_ids=list(candidate_ids),
        k=k, documents_mask=mask_b, device=torch.device(device),
    )


def _run_rroq158_cpu_mode(
    name: str, index_dir: Path, all_vectors: np.ndarray, doc_offsets: list,
    doc_ids: list, query_vecs: list, dim: int, params: dict,
    n_workers: int = 8, n_warmup: int = 5,
) -> Dict[str, Any]:
    """CPU equivalent of `_run_rroq158_gpu_mode` — uses the Rust SIMD kernel
    via `score_rroq158_topk(..., device='cpu')`.

    The corpus is encoded once (CPU) into the same in-memory payload used
    by the GPU path, but the tensors stay on CPU. Per-query LEMUR routing
    runs on CPU, then candidate slicing + scoring uses the Rust kernel
    inside `_rroq158_score_candidates`.

    Workers are independent processes/threads each owning their own router
    + payload reference (the underlying numpy arrays are shared via torch
    so memory cost is the same as a single-worker run).
    """
    device = "cpu"
    log.info("[rroq158-CPU] %s: loading LEMUR + encoding rroq158 ...", name)

    payload = _build_rroq158_gpu_payload(all_vectors, doc_offsets, dim, params, device)
    doc_ids_to_idx = {int(did): i for i, did in enumerate(doc_ids)}
    n_docs = len(doc_ids)

    worker_count = max(1, min(n_workers, len(query_vecs)))

    # Bound the Rust kernel's rayon pool so the total CPU thread count
    # stays at one-per-core. Without this, each of `worker_count` python
    # workers spawns its own rayon pool sized to `cpu_count()`, leading to
    # `worker_count * cpu_count()` competing threads (1024 on a 128-core
    # box with 8 workers). The scorer reads VOYAGER_RROQ158_N_WORKERS to
    # decide n_threads = max(1, cpu_count // n_workers).
    import os
    os.environ["VOYAGER_RROQ158_N_WORKERS"] = str(worker_count)

    skip_routing = _should_use_cpu_fast_path(
        n_docs, params["max_docs_exact"], codec="rroq158",
    )
    if skip_routing:
        log.info(
            "[rroq158-CPU] %s: skipping LEMUR routing (n_docs=%d, max_docs_exact=%d, threshold=%d) - whole-corpus rroq158 MaxSim",
            name, n_docs, params["max_docs_exact"], CPU_FASTPATH_RROQ158_MAX_DOCS,
        )

    routers: List[LemurRouter] = []
    if not skip_routing:
        for _ in range(worker_count):
            rt = LemurRouter(
                index_dir / "lemur",
                ann_backend=params["ann_backend"].value,
                device=device,
            )
            rt.load()
            routers.append(rt)
    else:
        # Fast-path doesn't need a per-worker router; reserve a sentinel
        # so the worker zip remains uniform.
        routers = [None] * worker_count

    def _worker_search(
        qi: int,
        router,
        _payload: dict = payload,
    ) -> Tuple[int, List[int], float]:
        t0 = time.perf_counter()
        if skip_routing:
            ids, _scores = _rroq158_score_all(
                query_vecs[qi], _payload, doc_ids, TOP_K, device,
            )
        else:
            qv = torch.from_numpy(query_vecs[qi]).float()
            routed = router.route(
                qv, k_candidates=params["k_candidates"],
                prefetch_doc_cap=params["max_docs_exact"],
            )
            cand = list(routed.doc_ids[: params["max_docs_exact"]])
            if not cand:
                return qi, [], (time.perf_counter() - t0) * 1000
            ids, _scores = _rroq158_score_candidates(
                query_vecs[qi], _payload, cand, doc_ids_to_idx, TOP_K, device,
            )
        return qi, ids, (time.perf_counter() - t0) * 1000

    for w_idx in range(worker_count):
        for qi in range(min(n_warmup, len(query_vecs))):
            _worker_search(qi, routers[w_idx])

    # Wall-time budget for CPU lanes — keeps full-corpus exact MaxSim on
    # large datasets (e.g. quora 522k docs) tractable. After the budget
    # expires we stop submitting new work and collect partial results.
    # Configurable via COLSEARCH_BENCH_CPU_TIME_BUDGET_S (default: 600s).
    # The legacy VOYAGER_BENCH_CPU_TIME_BUDGET_S name is honoured for one
    # release cycle; it will be removed in 0.3.0.
    try:
        cpu_budget_s = float(
            os.environ.get(
                "COLSEARCH_BENCH_CPU_TIME_BUDGET_S",
                os.environ.get("VOYAGER_BENCH_CPU_TIME_BUDGET_S", "600"),
            )
        )
    except ValueError:
        cpu_budget_s = 600.0

    log.info("[rroq158-CPU] %s: running %d queries (%d workers, time_budget=%.0fs) ...",
             name, len(query_vecs), worker_count, cpu_budget_s)

    import threading
    stop_flag = threading.Event()

    def _run_partition(router: LemurRouter, indices: List[int]) -> List[Tuple[int, List[int], float]]:
        out: List[Tuple[int, List[int], float]] = []
        for qi in indices:
            if stop_flag.is_set():
                break
            out.append(_worker_search(qi, router))
        return out

    query_partitions = [
        list(range(i, len(query_vecs), worker_count)) for i in range(worker_count)
    ]

    wall_t0 = time.perf_counter()
    all_results: List[Tuple[int, List[int], float]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_run_partition, router, indices)
            for router, indices in zip(routers, query_partitions)
            if indices
        ]
        # Watcher: trigger stop_flag once the budget expires. Workers
        # check stop_flag between queries and exit cleanly.
        def _watchdog() -> None:
            deadline = wall_t0 + cpu_budget_s
            while not stop_flag.is_set():
                if time.perf_counter() >= deadline:
                    log.info(
                        "[rroq158-CPU] %s: time-budget %.0fs reached; stopping early",
                        name, cpu_budget_s,
                    )
                    stop_flag.set()
                    return
                if all(f.done() for f in futures):
                    return
                time.sleep(1.0)

        watch = threading.Thread(target=_watchdog, daemon=True)
        watch.start()
        for future in futures:
            all_results.extend(future.result())
        stop_flag.set()
        watch.join(timeout=1.0)
    wall_s = time.perf_counter() - wall_t0

    all_results.sort(key=lambda x: x[0])
    all_ids = [r[1] for r in all_results]
    all_elapsed = [r[2] for r in all_results]
    n_completed = len(all_ids)

    qps = n_completed / wall_s if wall_s > 0 else 0
    p50 = float(np.median(all_elapsed))
    p95 = float(np.percentile(all_elapsed, 95))

    del payload, routers
    gc.collect()

    return {
        "mode": f"CPU-{worker_count}w",
        "dataset": name,
        "n_queries": len(query_vecs),
        "all_ids": all_ids,
        "qps": qps,
        "p50_ms": p50,
        "p95_ms": p95,
    }


def _run_rroq158_gpu_mode(
    name: str, index_dir: Path, all_vectors: np.ndarray, doc_offsets: list,
    doc_ids: list, query_vecs: list, dim: int, params: dict, n_warmup: int = 5,
) -> Dict[str, Any]:
    device = "cuda"

    # Decide fast vs slow path BEFORE any LEMUR work: when the corpus
    # fits the fast-path criteria (`_should_use_fast_path` = True), no
    # per-query call ever touches the router, so loading the MLP +
    # FAISS index is pure waste. Skip it. This mirrors the symmetric
    # change in `run_gpu_corpus_mode` for the fp16 lane.
    skip_routing = _should_use_fast_path(
        len(doc_ids), params["max_docs_exact"],
    )
    router = None
    if skip_routing:
        log.info(
            "[rroq158-GPU] %s: fast path (n_docs=%d ≤ threshold=%d) — "
            "skipping LEMUR router init, encoding rroq158 ...",
            name, len(doc_ids), WHOLE_CORPUS_FAST_PATH_THRESHOLD,
        )
    else:
        log.info("[rroq158-GPU] %s: slow path — loading LEMUR + encoding rroq158 ...", name)
        router = LemurRouter(
            index_dir / "lemur",
            ann_backend=params["ann_backend"].value,
            device=device,
        )
        router.load()

    payload = _build_rroq158_gpu_payload(all_vectors, doc_offsets, dim, params, device)
    doc_ids_to_idx = {int(did): i for i, did in enumerate(doc_ids)}

    distill_head = None
    if params.get("distill_rerank") and (index_dir / "distill_mv.npz").exists():
        distill_head = MultiViewDistillHead.from_npz(index_dir / "distill_mv.npz")
        log.info("[rroq158-GPU] %s: MV-distill head loaded", name)

    if skip_routing:
        rroq158_corpus_bytes = (
            payload["sign"].element_size() * payload["sign"].numel()
            + payload["nz"].element_size() * payload["nz"].numel()
            + payload["scl"].element_size() * payload["scl"].numel()
            + payload["cid"].element_size() * payload["cid"].numel()
            + payload["cosn"].element_size() * payload["cosn"].numel()
            + payload["sinn"].element_size() * payload["sinn"].numel()
            + payload["mask"].element_size() * payload["mask"].numel()
        )
        log.info(
            "[rroq158-GPU] %s: rroq158 corpus on device %.1f MB — "
            "direct whole-corpus MaxSim",
            name, rroq158_corpus_bytes / 1024**2,
        )

    for i in range(min(n_warmup, len(query_vecs))):
        qv = torch.from_numpy(query_vecs[i]).float()
        if skip_routing:
            _rroq158_score_all(query_vecs[i], payload, doc_ids, TOP_K, device)
        else:
            routed = router.route(
                qv, k_candidates=params["k_candidates"],
                prefetch_doc_cap=params["max_docs_exact"],
            )
            cand = routed.doc_ids[: params["max_docs_exact"]]
            if cand:
                _rroq158_score_candidates(query_vecs[i], payload, cand, doc_ids_to_idx,
                                          TOP_K, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    log.info("[rroq158-GPU] %s: running %d queries ...", name, len(query_vecs))
    all_ids = []
    all_elapsed = []
    for qi in range(len(query_vecs)):
        t0 = time.perf_counter()
        if skip_routing:
            ids, scores = _rroq158_score_all(
                query_vecs[qi], payload, doc_ids, TOP_K, device,
            )
        else:
            qv = torch.from_numpy(query_vecs[qi]).float()
            routed = router.route(
                qv, k_candidates=params["k_candidates"],
                prefetch_doc_cap=params["max_docs_exact"],
            )
            cand = list(routed.doc_ids[: params["max_docs_exact"]])
            if not cand:
                all_ids.append([])
                all_elapsed.append((time.perf_counter() - t0) * 1000)
                continue
            ids, scores = _rroq158_score_candidates(
                query_vecs[qi], payload, cand, doc_ids_to_idx, TOP_K, device,
            )
        if distill_head is not None and len(ids) >= 10:
            ids = distill_head.rerank(ids, scores) if hasattr(distill_head, "rerank") else ids
        all_ids.append(ids)
        all_elapsed.append((time.perf_counter() - t0) * 1000)

    total_s = sum(all_elapsed) / 1000.0
    qps = len(query_vecs) / total_s if total_s > 0 else 0
    p50 = float(np.median(all_elapsed))
    p95 = float(np.percentile(all_elapsed, 95))

    del payload, router
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


# ─────────────────────────────────────────────────────────────
# RROQ4_RIEM GPU + CPU modes (real LEMUR routing + Triton/SIMD kernel)
# ─────────────────────────────────────────────────────────────


def _build_rroq4_riem_payload(
    all_vectors: np.ndarray, doc_offsets: list, dim: int, params: dict, device: str,
):
    """Encode the corpus with rroq4_riem once and pad per-doc to global p95.

    Mirrors `_build_rroq158_gpu_payload` but with the 4-bit asymmetric
    payload tensors (`codes_packed`, `mins`, `deltas`) instead of the
    ternary sign/nonzero planes. Both the GPU (Triton) and CPU (Rust SIMD)
    kernels read the exact same in-memory layout.

    Returns a dict with device-resident tensors:
      cid       (D, T_max) int32
      cosn      (D, T_max) float32
      sinn      (D, T_max) float32
      codes     (D, T_max, dim/2) uint8
      mins      (D, T_max, n_groups) float32
      deltas    (D, T_max, n_groups) float32
      mask      (D, T_max) float32
      centroids (K, dim) float32
      group_size, K, t_max, dim
    """
    cfg = Rroq4RiemConfig(
        K=params.get("rroq4_riem_k", 8192),
        group_size=int(params.get("rroq4_riem_group_size", 32)),
        seed=int(params.get("rroq4_riem_seed", 42)),
    )
    log.info(
        "[rroq4_riem] encoding %d tokens (dim=%d, K=%d, group_size=%d)",
        all_vectors.shape[0], dim, cfg.K, cfg.group_size,
    )
    enc = encode_rroq4_riem(np.asarray(all_vectors, dtype=np.float32), cfg)

    n_groups = enc.mins.shape[1]
    nibble_bytes = enc.codes_packed.shape[1]  # = dim // 2

    n_docs = len(doc_offsets)
    tok_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int64)
    p95 = int(np.ceil(np.percentile(tok_counts, 95)))
    t_max = 1
    while t_max < p95:
        t_max *= 2
    log.info("[rroq4_riem] padding to T_max=%d (p95=%d)", t_max, p95)

    cid_dt = np.zeros((n_docs, t_max), dtype=np.int32)
    cosn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    sinn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    codes_dt = np.zeros((n_docs, t_max, nibble_bytes), dtype=np.uint8)
    mins_dt = np.zeros((n_docs, t_max, n_groups), dtype=np.float32)
    deltas_dt = np.zeros((n_docs, t_max, n_groups), dtype=np.float32)
    mask_dt = np.zeros((n_docs, t_max), dtype=np.float32)

    for di, (s, e) in enumerate(doc_offsets):
        n_tok = min(e - s, t_max)
        cid_dt[di, :n_tok] = enc.centroid_id[s:s + n_tok].astype(np.int32)
        cosn_dt[di, :n_tok] = enc.cos_norm[s:s + n_tok].astype(np.float32)
        sinn_dt[di, :n_tok] = enc.sin_norm[s:s + n_tok].astype(np.float32)
        codes_dt[di, :n_tok] = enc.codes_packed[s:s + n_tok]
        mins_dt[di, :n_tok] = enc.mins[s:s + n_tok].astype(np.float32)
        deltas_dt[di, :n_tok] = enc.deltas[s:s + n_tok].astype(np.float32)
        mask_dt[di, :n_tok] = 1.0

    bytes_per_tok = (
        nibble_bytes        # 4-bit residual codes
        + n_groups * 2 * 2  # mins + deltas (fp16)
        + 2                 # centroid_id (uint16 nominal)
        + 2 + 2             # cos_norm + sin_norm (fp16)
    )
    log.info("[rroq4_riem] disk: %.2f MB encoded payload (~%d B/tok)",
             (codes_dt.nbytes + mins_dt.nbytes + deltas_dt.nbytes
              + cid_dt.nbytes + cosn_dt.nbytes + sinn_dt.nbytes) / 1e6,
             bytes_per_tok)

    from colsearch._internal.inference.quantization.rroq4_riem import (
        get_cached_fwht_rotator,
    )

    centroids_np = np.ascontiguousarray(enc.centroids, dtype=np.float32)
    rotator = get_cached_fwht_rotator(dim=dim, seed=enc.fwht_seed)

    return {
        "cid": torch.from_numpy(cid_dt).to(device),
        "cosn": torch.from_numpy(cosn_dt).to(device),
        "sinn": torch.from_numpy(sinn_dt).to(device),
        "codes": torch.from_numpy(codes_dt).to(device),
        "mins": torch.from_numpy(mins_dt).to(device),
        "deltas": torch.from_numpy(deltas_dt).to(device),
        "mask": torch.from_numpy(mask_dt).to(device),
        "centroids": torch.from_numpy(enc.centroids).to(device),
        "centroids_np": centroids_np,
        "rotator": rotator,
        "fwht_seed": enc.fwht_seed,
        "group_size": cfg.group_size,
        "n_groups": n_groups,
        "K": cfg.K,
        "t_max": t_max,
        "dim": dim,
        "bytes_per_tok": bytes_per_tok,
    }


def _rroq4_riem_score_candidates(
    query_np: np.ndarray, payload: dict, candidate_ids: list, doc_ids_to_idx: dict,
    k: int, device: str,
):
    """Score one query against the rroq4_riem payload at the given candidate
    doc indices. Mirrors `_rroq158_score_candidates`, including the
    ``torch.index_select`` -> numpy fancy indexing bypass on CPU
    (audit_rroq158_cpu_2026q2 — same root cause: torch's default 64-thread
    intra-op pool fights the rayon kernel pool on shared cores)."""
    on_cuda = str(device).startswith("cuda")
    group_size = int(payload["group_size"])

    if on_cuda:
        q_inputs = encode_query_for_rroq4_riem(
            query_np, None,
            fwht_seed=payload["fwht_seed"],
            group_size=group_size,
            rotator=payload.get("rotator"),
            skip_qc_table=True,
        )
        q_rot = torch.from_numpy(q_inputs["q_rot"][None, :, :]).to(device)
        q_gs = torch.from_numpy(q_inputs["q_group_sums"][None, :, :]).to(device)
        q_dev = torch.from_numpy(np.ascontiguousarray(query_np, dtype=np.float32)).to(device)
        qc_table = (q_dev @ payload["centroids"].T).unsqueeze(0)

        cand_idx = torch.tensor(
            [doc_ids_to_idx[int(cid)] for cid in candidate_ids],
            dtype=torch.long, device=device,
        )
        cid_b = payload["cid"].index_select(0, cand_idx)
        cosn_b = payload["cosn"].index_select(0, cand_idx)
        sinn_b = payload["sinn"].index_select(0, cand_idx)
        codes_b = payload["codes"].index_select(0, cand_idx)
        mins_b = payload["mins"].index_select(0, cand_idx)
        deltas_b = payload["deltas"].index_select(0, cand_idx)
        mask_b = payload["mask"].index_select(0, cand_idx)
    else:
        centroids_np = payload.get("centroids_np")
        if centroids_np is None:
            centroids_np = payload["centroids"].cpu().numpy()
            payload["centroids_np"] = centroids_np
        q_inputs = encode_query_for_rroq4_riem(
            query_np, centroids_np,
            fwht_seed=payload["fwht_seed"],
            group_size=group_size,
            rotator=payload.get("rotator"),
        )
        q_rot = torch.from_numpy(q_inputs["q_rot"][None, :, :])
        q_gs = torch.from_numpy(q_inputs["q_group_sums"][None, :, :])
        qc_table = torch.from_numpy(q_inputs["qc_table"][None, :, :])

        cid_np = payload.setdefault("cid_np", payload["cid"].numpy())
        cosn_np = payload.setdefault("cosn_np", payload["cosn"].numpy())
        sinn_np = payload.setdefault("sinn_np", payload["sinn"].numpy())
        codes_np = payload.setdefault("codes_np", payload["codes"].numpy())
        mins_np = payload.setdefault("mins_np", payload["mins"].numpy())
        deltas_np = payload.setdefault("deltas_np", payload["deltas"].numpy())
        mask_np = payload.setdefault("mask_np", payload["mask"].numpy())

        cand_np = np.fromiter(
            (doc_ids_to_idx[int(cid)] for cid in candidate_ids),
            dtype=np.int64, count=len(candidate_ids),
        )
        cid_b = torch.from_numpy(cid_np[cand_np])
        cosn_b = torch.from_numpy(cosn_np[cand_np])
        sinn_b = torch.from_numpy(sinn_np[cand_np])
        codes_b = torch.from_numpy(codes_np[cand_np])
        mins_b = torch.from_numpy(mins_np[cand_np])
        deltas_b = torch.from_numpy(deltas_np[cand_np])
        mask_b = torch.from_numpy(mask_np[cand_np])

    return score_rroq4_riem_topk(
        q_rot, q_gs, qc_table,
        cid_b, cosn_b, sinn_b, codes_b, mins_b, deltas_b,
        doc_ids=list(candidate_ids),
        k=k, group_size=group_size,
        documents_mask=mask_b, device=torch.device(device),
    )


def _run_rroq4_riem_gpu_mode(
    name: str, index_dir: Path, all_vectors: np.ndarray, doc_offsets: list,
    doc_ids: list, query_vecs: list, dim: int, params: dict, n_warmup: int = 5,
) -> Dict[str, Any]:
    device = "cuda"
    log.info("[rroq4_riem-GPU] %s: loading LEMUR + encoding rroq4_riem ...", name)

    router = LemurRouter(
        index_dir / "lemur",
        ann_backend=params["ann_backend"].value,
        device=device,
    )
    router.load()

    payload = _build_rroq4_riem_payload(all_vectors, doc_offsets, dim, params, device)
    doc_ids_to_idx = {int(did): i for i, did in enumerate(doc_ids)}

    for i in range(min(n_warmup, len(query_vecs))):
        qv = torch.from_numpy(query_vecs[i]).float()
        routed = router.route(
            qv, k_candidates=params["k_candidates"],
            prefetch_doc_cap=params["max_docs_exact"],
        )
        cand = routed.doc_ids[: params["max_docs_exact"]]
        if cand:
            _rroq4_riem_score_candidates(query_vecs[i], payload, cand, doc_ids_to_idx,
                                         TOP_K, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    log.info("[rroq4_riem-GPU] %s: running %d queries ...", name, len(query_vecs))
    all_ids = []
    all_elapsed = []
    for qi in range(len(query_vecs)):
        t0 = time.perf_counter()
        qv = torch.from_numpy(query_vecs[qi]).float()
        routed = router.route(
            qv, k_candidates=params["k_candidates"],
            prefetch_doc_cap=params["max_docs_exact"],
        )
        cand = list(routed.doc_ids[: params["max_docs_exact"]])
        if not cand:
            all_ids.append([])
            all_elapsed.append((time.perf_counter() - t0) * 1000)
            continue
        ids, _scores = _rroq4_riem_score_candidates(
            query_vecs[qi], payload, cand, doc_ids_to_idx, TOP_K, device,
        )
        all_ids.append(ids)
        all_elapsed.append((time.perf_counter() - t0) * 1000)

    total_s = sum(all_elapsed) / 1000.0
    qps = len(query_vecs) / total_s if total_s > 0 else 0
    p50 = float(np.median(all_elapsed))
    p95 = float(np.percentile(all_elapsed, 95))

    del payload, router
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


def _run_rroq4_riem_cpu_mode(
    name: str, index_dir: Path, all_vectors: np.ndarray, doc_offsets: list,
    doc_ids: list, query_vecs: list, dim: int, params: dict,
    n_workers: int = 8, n_warmup: int = 5,
) -> Dict[str, Any]:
    """CPU equivalent of `_run_rroq4_riem_gpu_mode` — Rust SIMD kernel via
    `score_rroq4_riem_topk(..., device='cpu')`. Mirrors `_run_rroq158_cpu_mode`."""
    device = "cpu"
    log.info("[rroq4_riem-CPU] %s: loading LEMUR + encoding rroq4_riem ...", name)

    payload = _build_rroq4_riem_payload(all_vectors, doc_offsets, dim, params, device)
    doc_ids_to_idx = {int(did): i for i, did in enumerate(doc_ids)}

    worker_count = max(1, min(n_workers, len(query_vecs)))
    import os
    os.environ["VOYAGER_RROQ4_RIEM_N_WORKERS"] = str(worker_count)

    routers = []
    for _ in range(worker_count):
        rt = LemurRouter(
            index_dir / "lemur",
            ann_backend=params["ann_backend"].value,
            device=device,
        )
        rt.load()
        routers.append(rt)

    def _worker_search(
        qi: int,
        router: LemurRouter,
        _payload: dict = payload,
    ) -> Tuple[int, List[int], float]:
        t0 = time.perf_counter()
        qv = torch.from_numpy(query_vecs[qi]).float()
        routed = router.route(
            qv, k_candidates=params["k_candidates"],
            prefetch_doc_cap=params["max_docs_exact"],
        )
        cand = list(routed.doc_ids[: params["max_docs_exact"]])
        if not cand:
            return qi, [], (time.perf_counter() - t0) * 1000
        ids, _scores = _rroq4_riem_score_candidates(
            query_vecs[qi], _payload, cand, doc_ids_to_idx, TOP_K, device,
        )
        return qi, ids, (time.perf_counter() - t0) * 1000

    for w_idx in range(worker_count):
        for qi in range(min(n_warmup, len(query_vecs))):
            _worker_search(qi, routers[w_idx])

    log.info("[rroq4_riem-CPU] %s: running %d queries (%d workers) ...",
             name, len(query_vecs), worker_count)
    query_partitions = [
        list(range(i, len(query_vecs), worker_count)) for i in range(worker_count)
    ]

    def _run_partition(router: LemurRouter, indices: List[int]) -> List[Tuple[int, List[int], float]]:
        return [_worker_search(qi, router) for qi in indices]

    wall_t0 = time.perf_counter()
    all_results: List[Tuple[int, List[int], float]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_run_partition, router, indices)
            for router, indices in zip(routers, query_partitions)
            if indices
        ]
        for future in futures:
            all_results.extend(future.result())
    wall_s = time.perf_counter() - wall_t0

    all_results.sort(key=lambda x: x[0])
    all_ids = [r[1] for r in all_results]
    all_elapsed = [r[2] for r in all_results]

    qps = len(query_vecs) / wall_s if wall_s > 0 else 0
    p50 = float(np.median(all_elapsed))
    p95 = float(np.percentile(all_elapsed, 95))

    del payload, routers
    gc.collect()

    return {
        "mode": f"CPU-{worker_count}w",
        "dataset": name,
        "n_queries": len(query_vecs),
        "all_ids": all_ids,
        "qps": qps,
        "p50_ms": p50,
        "p95_ms": p95,
    }


class _RustCpuWorker:
    """Independent CPU worker using native Rust exact scoring.

    Supports two paths:

      * **Routed**: LEMUR routes → top-K candidates → exact MaxSim
        scoring (default; used for very large corpora where exact-all
        scoring is compute-bound).
      * **Whole-corpus fast-path**: bypass LEMUR and score all
        ``n_docs`` candidates directly via
        ``score_candidates_exact(q, all_doc_ids, k)``. Triggered by
        ``_should_use_cpu_fast_path`` (default ≤200K docs for fp16).
        Saves ~1-2 ms LEMUR routing latency at no quality cost.
    """

    def __init__(
        self,
        worker_id: int,
        index_dir: Path,
        dim: int,
        ann_backend: AnnBackend,
        n_docs: int,
        skip_routing: bool = False,
    ):
        import latence_shard_engine

        runtime_dir = index_dir / f"_rust_cpu_runtime_w{worker_id}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "shard.wal").touch(exist_ok=True)

        self._rust_idx = latence_shard_engine.ShardIndex(str(runtime_dir), dim)
        self._rust_idx.load_merged(str(index_dir))
        self._skip_routing = skip_routing
        self._all_doc_ids = list(range(n_docs)) if skip_routing else None
        if not skip_routing:
            self._router = LemurRouter(
                index_dir / "lemur",
                ann_backend=ann_backend.value,
                device="cpu",
            )
            self._router.load()
        else:
            self._router = None

    def search(self, query_vec: np.ndarray, params: dict, k: int) -> Tuple[List[int], List[float], float]:
        t0 = time.perf_counter()
        q_np = np.ascontiguousarray(query_vec, dtype=np.float32)
        if self._skip_routing:
            candidate_ids = self._all_doc_ids
        else:
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

    n_docs = len(doc_ids)
    skip_routing = _should_use_cpu_fast_path(
        n_docs, params["max_docs_exact"], codec="fp16",
    )
    if skip_routing:
        log.info(
            "[CPU-%dw] %s: skipping LEMUR routing (n_docs=%d, threshold=%d) - whole-corpus exact MaxSim",
            n_workers, name, n_docs, CPU_FASTPATH_FP16_MAX_DOCS,
        )

    try:
        ensure_merged_layout(index_dir, all_vectors, doc_offsets, doc_ids, dim)
        worker_count = max(1, min(n_workers, len(query_vecs)))
        workers = [
            _RustCpuWorker(
                i, index_dir, dim, params["ann_backend"],
                n_docs=n_docs, skip_routing=skip_routing,
            )
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

    try:
        cpu_budget_s = float(
            os.environ.get(
                "COLSEARCH_BENCH_CPU_TIME_BUDGET_S",
                os.environ.get("VOYAGER_BENCH_CPU_TIME_BUDGET_S", "600"),
            )
        )
    except ValueError:
        cpu_budget_s = 600.0

    log.info("[CPU-%dw] %s: running %d queries (native exact, %d workers, time_budget=%.0fs) ...",
             worker_count, name, len(query_vecs), worker_count, cpu_budget_s)

    import threading
    stop_flag = threading.Event()

    def _run_partition(worker: _RustCpuWorker, indices: List[int]) -> List[Tuple[int, List[int], float]]:
        out = []
        for qi in indices:
            if stop_flag.is_set():
                break
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
        def _watchdog() -> None:
            deadline = wall_t0 + cpu_budget_s
            while not stop_flag.is_set():
                if time.perf_counter() >= deadline:
                    log.info(
                        "[CPU-%dw] %s: time-budget %.0fs reached; stopping early",
                        worker_count, name, cpu_budget_s,
                    )
                    stop_flag.set()
                    return
                if all(f.done() for f in futures):
                    return
                time.sleep(1.0)
        watch = threading.Thread(target=_watchdog, daemon=True)
        watch.start()
        for future in futures:
            all_results.extend(future.result())
        stop_flag.set()
        watch.join(timeout=1.0)
    wall_s = time.perf_counter() - wall_t0

    ordered_ids = [[] for _ in range(len(query_vecs))]
    all_elapsed = []
    for qi, ids, elapsed in all_results:
        ordered_ids[qi] = ids
        all_elapsed.append(elapsed)
    n_completed = len(all_elapsed)

    qps = n_completed / wall_s if wall_s > 0 else 0
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

_COMPRESSION_BY_NAME = {
    "fp16": Compression.FP16,
    "int8": Compression.INT8,
    "roq4": Compression.ROQ4,
    "rroq158": Compression.RROQ158,
    "rroq4_riem": Compression.RROQ4_RIEM,
}

# Codecs whose CPU lane is genuinely unsupported (in-kernel int8/fp8
# pre-quantization is a GPU-only Triton path; running them on CPU would
# silently fall back to fp16 and double-count the cell). Cells in this
# set must be reported as N/A, not silently re-routed.
_GPU_ONLY_COMPRESSIONS = {Compression.INT8}


def _resolve_compression(name: str) -> Compression:
    key = name.lower()
    if key not in _COMPRESSION_BY_NAME:
        raise ValueError(
            f"Unknown compression '{name}'. Choices: {sorted(_COMPRESSION_BY_NAME)}"
        )
    return _COMPRESSION_BY_NAME[key]


def run_dataset(
    name: str,
    modes: List[str],
    n_workers: int = 8,
    n_eval: Optional[int] = 0,
    compression: Optional[Compression] = None,
    distill_rerank: bool = False,
    rroq158_k: int = 8192,
    rroq158_group_size: int = 128,
    rroq158_seed: int = 42,
    rroq4_riem_k: int = 8192,
    rroq4_riem_group_size: int = 32,
    rroq4_riem_seed: int = 42,
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
            if compression is not None:
                params["compression"] = compression
            params["distill_rerank"] = distill_rerank
            params["rroq158_k"] = rroq158_k
            params["rroq158_group_size"] = rroq158_group_size
            params["rroq158_seed"] = rroq158_seed
            params["rroq4_riem_k"] = rroq4_riem_k
            params["rroq4_riem_group_size"] = rroq4_riem_group_size
            params["rroq4_riem_seed"] = rroq4_riem_seed
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
            if compression is not None:
                params["compression"] = compression
            params["distill_rerank"] = distill_rerank
            params["rroq158_k"] = rroq158_k
            params["rroq158_group_size"] = rroq158_group_size
            params["rroq158_seed"] = rroq158_seed
            params["rroq4_riem_k"] = rroq4_riem_k
            params["rroq4_riem_group_size"] = rroq4_riem_group_size
            params["rroq4_riem_seed"] = rroq4_riem_seed
            if name == "quora":
                params["n_shards"] = QUORA_OVERRIDE_SHARDS

            # Honest scope: int8 (and fp8) are GPU-only Triton paths;
            # the CPU lane has no in-kernel int8 pre-quantization, only
            # FP16 maxsim. Marking the cell N/A is the truthful answer
            # — silently scoring fp16 in this row would inflate the
            # int8 CPU column with the FP16 number.
            if compression in _GPU_ONLY_COMPRESSIONS:
                log.info(
                    "[CPU] %s: compression=%s is GPU-only; reporting N/A",
                    name, compression.value,
                )
                results.append({
                    "mode": f"CPU-{n_workers}w",
                    "dataset": name,
                    "n_queries": len(eval_query_vecs),
                    "qps": None,
                    "p50_ms": None,
                    "p95_ms": None,
                    "skipped": True,
                    "skip_reason": f"{compression.value} is GPU-only (no CPU kernel)",
                    "n_docs": n_docs,
                    "dim": dim,
                    "top_k": TOP_K,
                    "params": {k: str(v) for k, v in params.items()},
                })
                continue

            device = "cuda" if torch.cuda.is_available() else "cpu"
            index_dir, build_s = build_index(name, all_vectors, doc_offsets, doc_ids, dim, params, device=device)
            indexing_throughput = n_docs / build_s if build_s > 0 else float("inf")

            if compression == Compression.RROQ158:
                search_result = _run_rroq158_cpu_mode(
                    name, index_dir, all_vectors, doc_offsets, doc_ids,
                    eval_query_vecs, dim, params, n_workers=n_workers,
                )
            elif compression == Compression.RROQ4_RIEM:
                search_result = _run_rroq4_riem_cpu_mode(
                    name, index_dir, all_vectors, doc_offsets, doc_ids,
                    eval_query_vecs, dim, params, n_workers=n_workers,
                )
            else:
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


def _fmt_optional(value: Optional[float], *, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_ratio(numerator: Optional[float], denominator: Optional[float]) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "n/a"
    return f"{(numerator / denominator):.1f}x"


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
            f"{_fmt_optional(row.get('gpu_qps')):>9} {_fmt_optional(row.get('gpu_p95')):>9} "
            f"{_fmt_optional(row.get('cpu_qps')):>9} {_fmt_optional(row.get('cpu_p95')):>9}"
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
            f"| {_fmt_optional(row.get('gpu_qps'))} | {_fmt_optional(row.get('gpu_p95'))} "
            f"| {_fmt_optional(row.get('cpu_qps'))} | {_fmt_optional(row.get('cpu_p95'))} |"
        )

    return "\n".join(lines)


def format_comparison_table(all_results: List[Dict[str, Any]]) -> str:
    """Compact GPU vs CPU speedup comparison."""
    lines = []
    lines.append("| Dataset | GPU QPS | CPU QPS | Speedup | GPU P95 (ms) | CPU P95 (ms) | Latency Ratio |")
    lines.append("|---------|--------:|--------:|--------:|-------------:|-------------:|--------------:|")

    for row in _merge_gpu_cpu(all_results):
        gpu_q = row.get("gpu_qps")
        cpu_q = row.get("cpu_qps")
        gpu_p = row.get("gpu_p95")
        cpu_p = row.get("cpu_p95")
        lines.append(
            f"| {row['dataset']} | {_fmt_optional(gpu_q)} | {_fmt_optional(cpu_q)} | {_fmt_ratio(gpu_q, cpu_q)} "
            f"| {_fmt_optional(gpu_p)} | {_fmt_optional(cpu_p)} | {_fmt_ratio(cpu_p, gpu_p)} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="BEIR Benchmark Suite for colsearch")
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
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        choices=sorted(_COMPRESSION_BY_NAME.keys()),
        help="Override compression for all runs (default: use OPTIMAL_GPU/CPU defaults)",
    )
    parser.add_argument(
        "--distill-rerank",
        action="store_true",
        help="Apply MV-distill reranker on top of rroq158 candidates "
        "(experimental; currently regresses Recall@10 on real BEIR -- see "
        "research/low_bit_roq/PROGRESS.md)",
    )
    parser.add_argument(
        "--rroq158-k",
        type=int,
        default=8192,
        help="Number of spherical centroids for rroq158 (default: 8192, "
        "the production value documented in research/low_bit_roq/PROGRESS.md)",
    )
    parser.add_argument(
        "--rroq158-group-size",
        type=int,
        default=128,
        help="Per-group block size for the rroq158 residual scales "
        "(default: 128 — the SOTA flip from Phase 8; one scale per token "
        "at dim=128. The kernel still requires the value to divide dim "
        "and be a multiple of 32; encode_rroq158 transparently falls back "
        "to gs=64 or gs=32 for incompatible dims via _resolve_group_size. "
        "Pin --rroq158-group-size 32 to reproduce the pre-SOTA-flip "
        "baseline.).",
    )
    parser.add_argument(
        "--rroq158-seed",
        type=int,
        default=42,
        help="Seed for rroq158 FWHT rotator + spherical k-means init (default: 42)",
    )
    parser.add_argument(
        "--rroq4-riem-k",
        type=int,
        default=8192,
        help="Number of spherical centroids for rroq4_riem (default: 8192)",
    )
    parser.add_argument(
        "--rroq4-riem-group-size",
        type=int,
        default=32,
        help="Per-group block size for the 4-bit asymmetric residual "
        "(default: 32, must divide dim and be even)",
    )
    parser.add_argument(
        "--rroq4-riem-seed",
        type=int,
        default=42,
        help="Seed for rroq4_riem FWHT rotator + spherical k-means init (default: 42)",
    )
    args = parser.parse_args()
    compression = _resolve_compression(args.compression) if args.compression else None

    all_results = []

    for name in args.datasets:
        log.info("=" * 70)
        log.info("DATASET: %s", name)
        log.info("=" * 70)

        try:
            ds_results = run_dataset(
                name, args.modes, n_workers=args.n_workers, n_eval=args.n_eval,
                compression=compression, distill_rerank=args.distill_rerank,
                rroq158_k=args.rroq158_k,
                rroq158_group_size=args.rroq158_group_size,
                rroq158_seed=args.rroq158_seed,
                rroq4_riem_k=args.rroq4_riem_k,
                rroq4_riem_group_size=args.rroq4_riem_group_size,
                rroq4_riem_seed=args.rroq4_riem_seed,
            )
            all_results.extend(ds_results)
        except Exception as e:
            log.error("Failed on %s: %s", name, e, exc_info=True)
            continue

    print("\n" + "=" * 120)
    print("BEIR BENCHMARK RESULTS — colsearch (search-only, encoding excluded)")
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
