"""Per-dataset fast-path validation across the BEIR datasets.

Times both ``voyager_fp16`` (PreloadedGpuCorpus.score_all) and
``voyager_rroq158_gs128`` (_rroq158_score_all) using the **full corpus
in VRAM**, exactly the same code path the production search uses when
``_should_use_fast_path`` returns True. Reports QPS / p50 / p95 and the
estimated GPU footprint per codec.
"""
from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beir_benchmark import (  # noqa: E402
    OPTIMAL_GPU,
    _build_rroq158_gpu_payload,
    _rroq158_score_all,
    build_index,
    load_beir_npz,
)
from voyager_index._internal.inference.shard_engine.config import Compression
from voyager_index._internal.inference.shard_engine.scorer import (
    PreloadedGpuCorpus,
    warmup_maxsim,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def time_fp16(corpus, query_vecs, n_warmup=50, n_meas=200):
    n = min(n_meas, len(query_vecs))
    for i in range(n_warmup):
        qv = torch.from_numpy(query_vecs[i % len(query_vecs)]).float()
        corpus.score_all(qv, k=10, return_stats=False)
    torch.cuda.synchronize()
    times = []
    for i in range(n):
        qv = torch.from_numpy(query_vecs[i]).float()
        t0 = time.perf_counter()
        corpus.score_all(qv, k=10, return_stats=False)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def time_rroq158(payload, doc_ids, query_vecs, n_warmup=50, n_meas=200):
    n = min(n_meas, len(query_vecs))
    for i in range(n_warmup):
        _rroq158_score_all(query_vecs[i % len(query_vecs)], payload, doc_ids, 10, "cuda")
    torch.cuda.synchronize()
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        _rroq158_score_all(query_vecs[i], payload, doc_ids, 10, "cuda")
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def free_gb():
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


def run_dataset(dataset: str):
    print(f"\n{'='*78}\nDATASET: {dataset}\n{'='*78}")
    all_vectors, doc_offsets, doc_ids, query_vecs, _, dim = load_beir_npz(dataset)
    n_docs = len(doc_ids)
    p95 = int(np.percentile([e - s for s, e in doc_offsets], 95))
    print(f"  {n_docs:>7d} docs  |  {len(query_vecs):>5d} queries  |  dim={dim}  |  T_p95={p95}")

    # ---- FP16 ----
    free_b, _ = free_gb()
    params_fp16 = dict(OPTIMAL_GPU); params_fp16["compression"] = Compression.FP16
    build_index(dataset, all_vectors, doc_offsets, doc_ids, dim, params_fp16, device="cuda")
    doc_vecs = [all_vectors[s:e] for s, e in doc_offsets]
    corpus = PreloadedGpuCorpus(doc_vecs, doc_ids, dim, device="cuda")
    fp16_bytes = corpus.D.element_size() * corpus.D.numel() + corpus.M.element_size() * corpus.M.numel()
    tok_counts = sorted({e - s for s, e in doc_offsets})
    q_tok = sorted({int(qv.shape[0]) for qv in query_vecs})
    warmup_maxsim(dim=dim, doc_token_counts=tok_counts, device="cuda", query_token_counts=q_tok)

    fp16_t = time_fp16(corpus, query_vecs)
    free_a, _ = free_gb()
    print(f"\n  fp16 score_all  | corpus={fp16_bytes/1024**2:7.1f} MB  free VRAM {free_a:.1f}/{free_b:.1f} GB  ")
    print(f"    p50 = {np.median(fp16_t):.3f} ms  p95 = {np.percentile(fp16_t, 95):.3f} ms  -> QPS = {1000.0/float(np.median(fp16_t)):.0f}")

    del corpus, doc_vecs
    gc.collect(); torch.cuda.empty_cache()

    # ---- RROQ158 ----
    free_b, _ = free_gb()
    params_r = dict(OPTIMAL_GPU); params_r["compression"] = Compression.RROQ158; params_r["rroq158_group_size"] = 128
    build_index(dataset, all_vectors, doc_offsets, doc_ids, dim, params_r, device="cuda")
    payload = _build_rroq158_gpu_payload(all_vectors, doc_offsets, dim, params_r, "cuda")
    rroq_bytes = sum(t.element_size() * t.numel() for k, t in payload.items()
                     if isinstance(t, torch.Tensor))

    rroq_t = time_rroq158(payload, doc_ids, query_vecs)
    free_a, _ = free_gb()
    print(f"\n  rroq158 score_all | corpus={rroq_bytes/1024**2:7.1f} MB  free VRAM {free_a:.1f}/{free_b:.1f} GB")
    print(f"    p50 = {np.median(rroq_t):.3f} ms  p95 = {np.percentile(rroq_t, 95):.3f} ms  -> QPS = {1000.0/float(np.median(rroq_t)):.0f}")

    print(f"\n  storage ratio (fp16 / rroq158): {fp16_bytes / rroq_bytes:.2f}x")
    print(f"  speed   ratio (fp16 / rroq158): {np.median(rroq_t) / np.median(fp16_t):.2f}x  (fp16 is faster)")

    del payload
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    targets = sys.argv[1:] or ["arguana", "nfcorpus", "scidocs", "trec-covid", "webis-touche2020"]
    for ds in targets:
        run_dataset(ds)
    print(f"\n{'='*78}\nDONE\n{'='*78}")
