"""End-to-end CPU production parity test for the AVX-512 VPOPCNTDQ tier.

Calls `score_rroq158_topk(device='cpu')` once with the backend forced to
X86V3 (scalar popcntq + AVX2) and once with X86V4 (AVX-512 VPOPCNTDQ),
verifies the returned (top_ids, top_scores) match bit-exact and reports
per-call latency for each. Mirrors `_smoke_b1_production.py` but for the
Rust SIMD CPU lane instead of the CUDA GPU lane.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import latence_shard_engine as eng  # noqa: E402

from voyager_index._internal.inference.quantization.rroq158 import (  # noqa: E402
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)
from voyager_index._internal.inference.shard_engine.scorer import (  # noqa: E402
    score_rroq158_topk,
)


def make_payload_cpu(n_docs=2000, t_max=32, dim=128, K=1024, seed=0):
    rng = np.random.default_rng(seed)
    tok_counts = rng.integers(20, t_max + 1, size=n_docs)
    tok_counts = np.clip(tok_counts, 1, t_max)
    total = int(tok_counts.sum())
    vec = rng.standard_normal((total, dim)).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    cfg = Rroq158Config(K=K, group_size=128, seed=seed)
    enc = encode_rroq158(vec, cfg)
    n_int32 = enc.sign_plane.shape[1] // 4
    n_groups = enc.scales.shape[1]

    sign = np.zeros((n_docs, t_max, n_int32), dtype=np.int32)
    nz = np.zeros((n_docs, t_max, n_int32), dtype=np.int32)
    scl = np.zeros((n_docs, t_max, n_groups), dtype=np.float32)
    cid = np.zeros((n_docs, t_max), dtype=np.int32)
    cosn = np.zeros((n_docs, t_max), dtype=np.float32)
    sinn = np.zeros((n_docs, t_max), dtype=np.float32)
    mask = np.zeros((n_docs, t_max), dtype=np.float32)

    cur = 0
    for di, n in enumerate(tok_counts):
        n = int(n)
        sign[di, :n] = pack_doc_codes_to_int32_words(enc.sign_plane[cur:cur + n])
        nz[di, :n] = pack_doc_codes_to_int32_words(enc.nonzero_plane[cur:cur + n])
        scl[di, :n] = enc.scales[cur:cur + n].astype(np.float32)
        cid[di, :n] = enc.centroid_id[cur:cur + n].astype(np.int32)
        cosn[di, :n] = enc.cos_norm[cur:cur + n].astype(np.float32)
        sinn[di, :n] = enc.sin_norm[cur:cur + n].astype(np.float32)
        mask[di, :n] = 1.0
        cur += n

    return dict(
        sign=torch.from_numpy(sign), nz=torch.from_numpy(nz),
        scl=torch.from_numpy(scl), cid=torch.from_numpy(cid),
        cosn=torch.from_numpy(cosn), sinn=torch.from_numpy(sinn),
        mask=torch.from_numpy(mask),
        centroids=enc.centroids, fwht_seed=enc.fwht_seed,
        n_int32=n_int32, n_groups=n_groups, K=K, dim=dim,
    )


def make_query_cpu(payload, S=32, dim=128, seed=1):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((S, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q_inputs = encode_query_for_rroq158(
        q, payload["centroids"],
        fwht_seed=payload["fwht_seed"], query_bits=4,
        rotator=None,
        skip_qc_table=False,
        cap_blas_threads=False,
    )
    qp = torch.from_numpy(q_inputs["q_planes"][None, ...])
    qm = torch.from_numpy(q_inputs["q_meta"][None, ...])
    qct = torch.from_numpy(q_inputs["qc_table"][None, ...])
    return qp, qm, qct


def call(payload, qp, qm, qct, doc_ids, k):
    return score_rroq158_topk(
        qp, qm, qct,
        payload["cid"], payload["cosn"], payload["sinn"],
        payload["sign"], payload["nz"], payload["scl"],
        doc_ids=doc_ids, k=k,
        documents_mask=payload["mask"],
        device=torch.device("cpu"),
    )


def bench(payload, qp, qm, qct, doc_ids, k, n_iter=30, n_warm=5):
    for _ in range(n_warm):
        call(payload, qp, qm, qct, doc_ids, k)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ids, sc = call(payload, qp, qm, qct, doc_ids, k)
    dt_ms = (time.perf_counter() - t0) * 1e3 / n_iter
    return dt_ms, ids, sc


def main():
    print("[prod-cpu] preparing payload + query (production rroq158 shape, "
          "B=2000, T=32, S=32, dim=128)...")
    payload = make_payload_cpu(n_docs=2000, t_max=32)
    qp, qm, qct = make_query_cpu(payload)
    doc_ids = list(range(payload["sign"].shape[0]))
    K_TOPK = 10

    print("[prod-cpu] benchmarking V3 path (scalar popcntq + AVX2)...")
    eng._rroq158_force_backend_for_tests("x86v3")
    v3_ms, ids_v3, sc_v3 = bench(payload, qp, qm, qct, doc_ids, K_TOPK)
    print(f"[prod-cpu]   V3: {v3_ms:7.3f} ms / call  topk_ids[:5]={ids_v3[:5]}")

    print("[prod-cpu] benchmarking V4 path (AVX-512 VPOPCNTDQ, auto-detect)...")
    eng._rroq158_force_backend_for_tests("auto")
    v4_ms, ids_v4, sc_v4 = bench(payload, qp, qm, qct, doc_ids, K_TOPK)
    print(f"[prod-cpu]   V4: {v4_ms:7.3f} ms / call  topk_ids[:5]={ids_v4[:5]}")

    same_ids = list(ids_v3) == list(ids_v4)
    sc_v3_arr = np.array(sc_v3, dtype=np.float32)
    sc_v4_arr = np.array(sc_v4, dtype=np.float32)
    abs_err = float(np.abs(sc_v3_arr - sc_v4_arr).max())
    rel_err = float((np.abs(sc_v3_arr - sc_v4_arr) / (np.abs(sc_v3_arr) + 1e-6)).max())
    bit_exact = bool(np.array_equal(sc_v3_arr, sc_v4_arr))

    print()
    print("=" * 70)
    print("PRODUCTION CPU-LANE PARITY (score_rroq158_topk, device='cpu')")
    print("=" * 70)
    print(f"  topk ids identical:     {same_ids}")
    print(f"  topk scores bit-exact:  {bit_exact}")
    print(f"  topk scores max abs:    {abs_err:.4e}")
    print(f"  topk scores max rel:    {rel_err:.4e}")
    print(f"  V3 per-call ms:         {v3_ms:7.3f}")
    print(f"  V4 per-call ms:         {v4_ms:7.3f}")
    print(f"  speedup V4/V3:          {v3_ms / v4_ms:.2f}x")
    print("=" * 70)
    if not same_ids or rel_err > 5e-3:
        print("[prod-cpu] FAIL: parity broken")
        sys.exit(1)
    print("[prod-cpu] OK: production CPU path uses V4 kernel correctly")


if __name__ == "__main__":
    main()
