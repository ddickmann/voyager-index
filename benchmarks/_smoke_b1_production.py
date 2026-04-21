"""End-to-end parity test for the fused b1 kernel via production wiring.

Calls `score_rroq158_topk` once with VOYAGER_RROQ158_USE_B1_FUSED=0 (Triton)
and once with =1 (fused). Checks the returned (top_ids, top_scores) match
within fp32 tolerance and reports the per-call latency for each path.
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


from voyager_index._internal.inference.quantization.rroq158 import (  # noqa: E402
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    pack_doc_codes_to_int32_words,
)
from voyager_index._internal.inference.shard_engine.scorer import (  # noqa: E402
    score_rroq158_topk,
)


def make_payload(n_docs=2048, t_max=288, dim=128, K=1024, seed=0):
    rng = np.random.default_rng(seed)
    tok_counts = rng.integers(60, max(t_max - 10, 280), size=n_docs)
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
        sign_w = pack_doc_codes_to_int32_words(enc.sign_plane[cur:cur + n])
        nz_w = pack_doc_codes_to_int32_words(enc.nonzero_plane[cur:cur + n])
        sign[di, :n] = sign_w
        nz[di, :n] = nz_w
        scl[di, :n] = enc.scales[cur:cur + n].astype(np.float32)
        cid[di, :n] = enc.centroid_id[cur:cur + n].astype(np.int32)
        cosn[di, :n] = enc.cos_norm[cur:cur + n].astype(np.float32)
        sinn[di, :n] = enc.sin_norm[cur:cur + n].astype(np.float32)
        mask[di, :n] = 1.0
        cur += n
    return dict(
        sign=sign, nz=nz, scl=scl, cid=cid, cosn=cosn, sinn=sinn, mask=mask,
        centroids=enc.centroids, fwht_seed=enc.fwht_seed,
        n_int32=n_int32, n_groups=n_groups, K=K, dim=dim,
    )


def make_query(payload, S=32, dim=128, seed=1):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((S, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q_inputs = encode_query_for_rroq158(
        q, None,
        fwht_seed=payload["fwht_seed"], query_bits=4,
        rotator=None,
        skip_qc_table=True,
        cap_blas_threads=False,
    )
    return q, q_inputs


def to_gpu(payload, query_np, q_inputs, device):
    qp = torch.from_numpy(q_inputs["q_planes"][None, ...]).to(device)
    qm = torch.from_numpy(q_inputs["q_meta"][None, ...]).to(device)
    q_dev = torch.from_numpy(query_np).to(device)
    centroids = torch.from_numpy(payload["centroids"]).to(device)
    qct = (q_dev @ centroids.T).unsqueeze(0)
    return dict(
        qp=qp, qm=qm, qct=qct,
        cid=torch.from_numpy(payload["cid"]).to(device),
        cosn=torch.from_numpy(payload["cosn"]).to(device),
        sinn=torch.from_numpy(payload["sinn"]).to(device),
        sign=torch.from_numpy(payload["sign"]).to(device),
        nz=torch.from_numpy(payload["nz"]).to(device),
        scl=torch.from_numpy(payload["scl"]).to(device),
        mask=torch.from_numpy(payload["mask"]).to(device),
    )


def call(g, doc_ids, k, device):
    return score_rroq158_topk(
        g["qp"], g["qm"], g["qct"],
        g["cid"], g["cosn"], g["sinn"],
        g["sign"], g["nz"], g["scl"],
        doc_ids=doc_ids, k=k,
        documents_mask=g["mask"], device=device,
    )


def bench(g, doc_ids, k, device, n_iter=50):
    # Warm
    for _ in range(5):
        call(g, doc_ids, k, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ids, sc = call(g, doc_ids, k, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3 / n_iter
    return dt_ms, ids, sc


def main():
    device = torch.device("cuda:0")
    print("[prod] preparing payload + query...")
    payload = make_payload(n_docs=2048)
    query_np, q_inputs = make_query(payload)
    g = to_gpu(payload, query_np, q_inputs, device)
    doc_ids = list(range(payload["sign"].shape[0]))

    K_TOPK = 10

    print("[prod] benchmarking Triton path (VOYAGER_RROQ158_USE_B1_FUSED=0)...")
    os.environ["VOYAGER_RROQ158_USE_B1_FUSED"] = "0"
    # Force re-import path decision: re-import the kernels module
    import importlib
    from voyager_index._internal.kernels import cuda_b1_rroq158
    importlib.reload(cuda_b1_rroq158)
    triton_ms, ids_t, sc_t = bench(g, doc_ids, K_TOPK, device)
    print(f"[prod]   Triton: {triton_ms:.3f} ms / call  topk_ids[:5]={ids_t[:5]}")

    print("[prod] benchmarking FUSED path  (VOYAGER_RROQ158_USE_B1_FUSED=1)...")
    os.environ["VOYAGER_RROQ158_USE_B1_FUSED"] = "1"
    importlib.reload(cuda_b1_rroq158)
    fused_ms, ids_f, sc_f = bench(g, doc_ids, K_TOPK, device)
    print(f"[prod]   FUSED:  {fused_ms:.3f} ms / call  topk_ids[:5]={ids_f[:5]}")

    # Parity (top-k ids and scores)
    same_ids = list(ids_t) == list(ids_f)
    sc_t_arr = np.array(sc_t, dtype=np.float32)
    sc_f_arr = np.array(sc_f, dtype=np.float32)
    abs_err = float(np.abs(sc_t_arr - sc_f_arr).max())
    rel_err = float((np.abs(sc_t_arr - sc_f_arr) / (np.abs(sc_t_arr) + 1e-6)).max())

    print()
    print("=" * 70)
    print("PRODUCTION-WIRING PARITY")
    print("=" * 70)
    print(f"  topk ids identical:   {same_ids}")
    print(f"  topk scores max abs:  {abs_err:.4e}")
    print(f"  topk scores max rel:  {rel_err:.4e}")
    print(f"  speedup FUSED/Triton: {triton_ms / fused_ms:.2f}x")
    print(f"  triton  per-call ms:  {triton_ms:.3f}")
    print(f"  fused   per-call ms:  {fused_ms:.3f}")
    print("=" * 70)
    if not same_ids or rel_err > 5e-3:
        print("[prod] FAIL: parity broken")
        sys.exit(1)
    print("[prod] OK: production path uses fused kernel correctly")


if __name__ == "__main__":
    main()
