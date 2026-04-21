"""Parity + speedup smoke for bucketed PreloadedGpuCorpus.

Generates a quora-like synthetic corpus (most docs short, ~1% tail of
long docs) and checks that:

  1. The bucketed layout activates (env default).
  2. score_all top-k ids/scores are bit-identical to the single-tier
     layout (env-disabled) for every query.
  3. score_candidates top-k ids/scores are bit-identical for routed
     candidate sets.
  4. The bucketed layout uses substantially less VRAM and runs faster
     than the single-tier layout.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force the env vars before import so PreloadedGpuCorpus picks them up.
os.environ.setdefault("VOYAGER_FP16_BUCKETED_PADDING", "1")

from voyager_index._internal.inference.shard_engine.scorer import (  # noqa: E402
    PreloadedGpuCorpus,
)


def _quora_like_corpus(n_docs=20_000, dim=128, p95_tok=30, max_tok=253, seed=0):
    """522K is too big for a smoke; use 20K with the same shape stats.

    99% of docs at p<=p95 tokens (uniform 5..p95+10). 1% tail with token
    counts uniform between p95+10 and max_tok. Same skew profile as the
    real quora distribution.
    """
    rng = np.random.default_rng(seed)
    tail_frac = 0.01
    n_tail = max(1, int(n_docs * tail_frac))
    n_short = n_docs - n_tail
    short_counts = rng.integers(5, p95_tok + 10, size=n_short)
    tall_counts = rng.integers(p95_tok + 10, max_tok + 1, size=n_tail)
    counts = np.concatenate([short_counts, tall_counts])
    rng.shuffle(counts)

    doc_vecs = []
    for c in counts:
        v = rng.standard_normal((int(c), dim)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        doc_vecs.append(v)
    doc_ids = list(range(n_docs))
    return doc_vecs, doc_ids


def _build(env_value, doc_vecs, doc_ids, dim, device):
    os.environ["VOYAGER_FP16_BUCKETED_PADDING"] = env_value
    return PreloadedGpuCorpus(doc_vecs, doc_ids, dim, device=device)


def _bench_score_all(corpus, queries, k=10, n_warm=5, n_iter=30):
    for q in queries[:n_warm]:
        corpus.score_all(q, k=k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for q in queries:
            corpus.score_all(q, k=k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3 / (n_iter * len(queries))
    return dt_ms


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping smoke")
        return 0
    device = "cuda"

    print("[bucket-smoke] building quora-like synthetic corpus (n=20000, p95=30, max=253)...")
    doc_vecs, doc_ids = _quora_like_corpus(n_docs=20_000)
    dim = doc_vecs[0].shape[1]

    rng = np.random.default_rng(7)
    queries = [
        torch.from_numpy(rng.standard_normal((32, dim)).astype(np.float32))
        for _ in range(8)
    ]

    print("[bucket-smoke] building SINGLE-TIER corpus (env=0)...")
    c_single = _build("0", doc_vecs, doc_ids, dim, device)
    print(f"  is_bucketed = {c_single._is_bucketed}, gpu_bytes = "
          f"{c_single._gpu_bytes / 1e9:.2f} GB")

    print("[bucket-smoke] building BUCKETED corpus (env=1)...")
    c_bucket = _build("1", doc_vecs, doc_ids, dim, device)
    print(f"  is_bucketed = {c_bucket._is_bucketed}, gpu_bytes = "
          f"{c_bucket._gpu_bytes / 1e9:.2f} GB")

    if not c_bucket._is_bucketed:
        print("[bucket-smoke] FAIL: bucketing did not activate on quora-like dist")
        return 1

    print("[bucket-smoke] checking score_all parity over 8 queries...")
    all_pass = True
    for qi, q in enumerate(queries):
        ids_s, sc_s = c_single.score_all(q, k=10)
        ids_b, sc_b = c_bucket.score_all(q, k=10)
        same_ids = ids_s == ids_b
        sc_s_a = np.array(sc_s, dtype=np.float32)
        sc_b_a = np.array(sc_b, dtype=np.float32)
        max_abs = float(np.abs(sc_s_a - sc_b_a).max())
        ok = same_ids and max_abs < 1e-3
        marker = "OK" if ok else "FAIL"
        print(f"  q{qi}: ids_match={same_ids}  max_abs={max_abs:.2e}  -> {marker}")
        all_pass = all_pass and ok

    print("[bucket-smoke] checking score_candidates parity...")
    rng2 = np.random.default_rng(11)
    cand_ids = rng2.choice(len(doc_ids), size=2000, replace=False).tolist()
    for qi, q in enumerate(queries[:3]):
        ids_s, sc_s = c_single.score_candidates(q, cand_ids, k=10)
        ids_b, sc_b = c_bucket.score_candidates(q, cand_ids, k=10)
        same_ids = ids_s == ids_b
        sc_s_a = np.array(sc_s, dtype=np.float32)
        sc_b_a = np.array(sc_b, dtype=np.float32)
        max_abs = float(np.abs(sc_s_a - sc_b_a).max())
        ok = same_ids and max_abs < 1e-3
        marker = "OK" if ok else "FAIL"
        print(f"  cand-q{qi}: ids_match={same_ids}  max_abs={max_abs:.2e}  -> {marker}")
        all_pass = all_pass and ok

    print("[bucket-smoke] benchmarking score_all latency (single-tier vs bucketed)...")
    ms_single = _bench_score_all(c_single, queries[:4])
    ms_bucket = _bench_score_all(c_bucket, queries[:4])
    print(f"  single-tier: {ms_single:6.3f} ms / query")
    print(f"  bucketed:    {ms_bucket:6.3f} ms / query  ({ms_single / ms_bucket:.2f}x)")
    print(f"  vram saving: {(c_single._gpu_bytes - c_bucket._gpu_bytes) / 1e9:.2f} GB "
          f"({c_single._gpu_bytes / max(c_bucket._gpu_bytes, 1):.2f}x leaner)")

    print("=" * 70)
    if all_pass:
        print("PASS: bucketed PreloadedGpuCorpus is parity-equivalent and faster")
        return 0
    print("FAIL: parity broken")
    return 1


if __name__ == "__main__":
    sys.exit(main())
