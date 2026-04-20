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
    score_rroq158_topk,
    score_rroq4_riem_topk,
    warmup_maxsim,
)
from voyager_index._internal.inference.shard_engine.shard_store import ShardStore
from voyager_index._internal.inference.quantization.rroq158 import (
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)
from voyager_index._internal.inference.quantization.rroq4_riem import (
    Rroq4RiemConfig,
    encode_query_for_rroq4_riem,
    encode_rroq4_riem,
)
from voyager_index._internal.inference.quantization.distill_mv import (
    MultiViewDistillHead,
    build_features_for_shortlist,
)

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
            shutil.copytree(fp16_dir, index_dir)
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


# ─────────────────────────────────────────────────────────────
# RROQ158 GPU-corpus mode (real LEMUR routing + Triton kernel)
# ─────────────────────────────────────────────────────────────


def _build_rroq158_gpu_payload(
    all_vectors: np.ndarray, doc_offsets: list, dim: int, params: dict, device: str,
):
    """Encode the corpus with rroq158 once and pad per-doc to global p95.

    Returns a dict with GPU-resident tensors:
      sign  (D, T_max, n_words) int32
      nz    (D, T_max, n_words) int32
      scl   (D, T_max, n_groups) float32
      cid   (D, T_max) int32
      cosn  (D, T_max) float32
      sinn  (D, T_max) float32
      mask  (D, T_max) float32
      centroids (K, dim) float32
      n_words, n_groups, K
    """
    cfg = Rroq158Config(
        K=params.get("rroq158_k", 1024),
        group_size=int(params.get("rroq158_group_size", 128)),
        seed=int(params.get("rroq158_seed", 42)),
    )
    log.info("[rroq158] encoding %d tokens (dim=%d, K=%d, group_size=%d)",
             all_vectors.shape[0], dim, cfg.K, cfg.group_size)
    enc = encode_rroq158(np.asarray(all_vectors, dtype=np.float32), cfg)

    n_words = enc.sign_plane.shape[1]
    n_groups = enc.scales.shape[1]
    if n_words % 4 != 0:
        raise RuntimeError(f"sign plane n_bytes={n_words} not multiple of 4")
    n_int32_words = n_words // 4

    n_docs = len(doc_offsets)
    tok_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int64)
    p95 = int(np.ceil(np.percentile(tok_counts, 95)))
    t_max = 1
    while t_max < p95:
        t_max *= 2
    log.info("[rroq158] padding to T_max=%d (p95=%d)", t_max, p95)

    sign_dt = np.zeros((n_docs, t_max, n_int32_words), dtype=np.int32)
    nz_dt = np.zeros((n_docs, t_max, n_int32_words), dtype=np.int32)
    scl_dt = np.zeros((n_docs, t_max, n_groups), dtype=np.float32)
    cid_dt = np.zeros((n_docs, t_max), dtype=np.int32)
    cosn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    sinn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    mask_dt = np.zeros((n_docs, t_max), dtype=np.float32)

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

    bytes_per_tok = (
        2 * n_words + n_groups * 2 + 2 + 2 + 2  # sign+nz + scales(fp16) + cid(int16 nominal) + cos+sin
    )
    log.info("[rroq158] disk: %.2f MB encoded payload (~%d B/tok)",
             (sign_dt.nbytes + nz_dt.nbytes + scl_dt.nbytes
              + cid_dt.nbytes + cosn_dt.nbytes + sinn_dt.nbytes) / 1e6,
             bytes_per_tok)

    from voyager_index._internal.inference.quantization.rroq158 import (
        get_cached_fwht_rotator,
    )

    centroids_np = np.ascontiguousarray(enc.centroids, dtype=np.float32)
    rotator = get_cached_fwht_rotator(dim=dim, seed=enc.fwht_seed)

    return {
        "sign": torch.from_numpy(sign_dt).to(device),
        "nz": torch.from_numpy(nz_dt).to(device),
        "scl": torch.from_numpy(scl_dt).to(device),
        "cid": torch.from_numpy(cid_dt).to(device),
        "cosn": torch.from_numpy(cosn_dt).to(device),
        "sinn": torch.from_numpy(sinn_dt).to(device),
        "mask": torch.from_numpy(mask_dt).to(device),
        "centroids": torch.from_numpy(enc.centroids).to(device),
        # Cached host-side aliases so per-query encode does not have to do a
        # GPU->CPU copy of the centroid table or rebuild the FWHT rotator.
        "centroids_np": centroids_np,
        "rotator": rotator,
        "fwht_seed": enc.fwht_seed,
        "n_words": n_int32_words,
        "n_groups": n_groups,
        "K": cfg.K,
        "t_max": t_max,
        "dim": dim,
        "bytes_per_tok": bytes_per_tok,
    }


def _rroq158_score_candidates(
    query_np: np.ndarray, payload: dict, candidate_ids: list, doc_ids_to_idx: dict,
    k: int, device: str,
):
    """Score one query against the rroq158 GPU payload at the given candidate
    doc indices. Returns (top_ids, top_scores).

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
        )
        q_planes = torch.from_numpy(q_inputs["q_planes"][None, :, :, :]).to(device)
        q_meta = torch.from_numpy(q_inputs["q_meta"][None, :, :]).to(device)
        q_dev = torch.from_numpy(np.ascontiguousarray(query_np, dtype=np.float32)).to(device)
        qc_table = (q_dev @ payload["centroids"].T).unsqueeze(0)

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

    worker_count = max(1, min(n_workers, len(query_vecs)))

    # Bound the Rust kernel's rayon pool so the total CPU thread count
    # stays at one-per-core. Without this, each of `worker_count` python
    # workers spawns its own rayon pool sized to `cpu_count()`, leading to
    # `worker_count * cpu_count()` competing threads (1024 on a 128-core
    # box with 8 workers). The scorer reads VOYAGER_RROQ158_N_WORKERS to
    # decide n_threads = max(1, cpu_count // n_workers).
    import os
    os.environ["VOYAGER_RROQ158_N_WORKERS"] = str(worker_count)

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
        _payload: dict = payload,  # bind via default arg → ruff-friendly closure
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
        ids, _scores = _rroq158_score_candidates(
            query_vecs[qi], _payload, cand, doc_ids_to_idx, TOP_K, device,
        )
        return qi, ids, (time.perf_counter() - t0) * 1000

    for w_idx in range(worker_count):
        for qi in range(min(n_warmup, len(query_vecs))):
            _worker_search(qi, routers[w_idx])

    log.info("[rroq158-CPU] %s: running %d queries (%d workers) ...",
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


def _run_rroq158_gpu_mode(
    name: str, index_dir: Path, all_vectors: np.ndarray, doc_offsets: list,
    doc_ids: list, query_vecs: list, dim: int, params: dict, n_warmup: int = 5,
) -> Dict[str, Any]:
    device = "cuda"
    log.info("[rroq158-GPU] %s: loading LEMUR + encoding rroq158 ...", name)

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

    for i in range(min(n_warmup, len(query_vecs))):
        qv = torch.from_numpy(query_vecs[i]).float()
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

    from voyager_index._internal.inference.quantization.rroq4_riem import (
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
