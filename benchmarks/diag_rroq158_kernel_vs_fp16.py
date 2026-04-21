"""Isolate rroq158 kernel quality vs fp16 brute-force MaxSim — no routing.

Computes NDCG@10 on arguana for two scoring paths against the IDENTICAL
8674-document candidate set per query:

  A) fp16 brute-force MaxSim (the truth ranking)
  B) rroq158 reference Python MaxSim (the lossy approximation)

If A and B agree closely, the −2.7 NDCG@10 we measured in the production
sweep is a routing / wrapper artifact. If A − B ≈ −2.7 pt here too, the
1.58-bit codec genuinely loses that quality on arguana and the historical
"K=8192 closes the gap" claim was over-stated.

Run:
    python benchmarks/diag_rroq158_kernel_vs_fp16.py --dataset arguana \
           --n-queries 200 --rroq158-k 8192
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beir_benchmark import load_beir_npz, evaluate  # noqa: E402
from colsearch._internal.inference.quantization.rroq158 import (  # noqa: E402
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    get_cached_fwht_rotator,
    pack_doc_codes_to_int32_words,
)
from colsearch._internal.inference.quantization.rroq4_riem import (  # noqa: E402
    Rroq4RiemConfig,
    encode_query_for_rroq4_riem,
    encode_rroq4_riem,
)
from colsearch._internal.inference.quantization.rroq4_riem import (  # noqa: E402
    get_cached_fwht_rotator as get_cached_fwht_rotator_riem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("diag_rroq158")


# ---------------------------------------------------------------------------
# fp16 brute-force MaxSim on GPU (the truth ranking — no codec, no routing)
# ---------------------------------------------------------------------------

def fp16_maxsim_topk_gpu(
    query_vecs: list, doc_vecs: list, top_k: int, device: str,
) -> list[list[int]]:
    """For each query, score against ALL docs with exact fp16 MaxSim, return
    top-k doc indices. This is the brute-force ground truth ranking."""
    docs_tensor = []
    doc_lens = []
    for d in doc_vecs:
        t = torch.from_numpy(np.ascontiguousarray(d)).to(device).float()
        docs_tensor.append(t)
        doc_lens.append(t.shape[0])

    all_ids: list[list[int]] = []
    n_docs = len(doc_vecs)
    for qi, q in enumerate(query_vecs):
        q_t = torch.from_numpy(q).to(device).float()  # (Sq, dim)
        scores = torch.zeros(n_docs, device=device, dtype=torch.float32)
        for di, d_t in enumerate(docs_tensor):
            sim = q_t @ d_t.T  # (Sq, Td)
            per_q_max = sim.max(dim=1).values  # (Sq,)
            scores[di] = per_q_max.sum()
        topv, topi = torch.topk(scores, k=min(top_k, n_docs))
        all_ids.append([int(x) for x in topi.cpu().numpy().tolist()])
    return all_ids


def fp16_maxsim_topk_gpu_batched(
    query_vecs: list, all_vectors: np.ndarray, doc_offsets: list,
    top_k: int, device: str,
) -> list[list[int]]:
    """Faster fp16 brute-force MaxSim using a flat (n_tokens, dim) tensor and
    a per-query gather. Runs all 8674 docs in a single matmul."""
    flat = torch.from_numpy(np.ascontiguousarray(all_vectors)).to(device).float()
    n_tok = flat.shape[0]
    n_docs = len(doc_offsets)
    starts = torch.tensor([s for s, _ in doc_offsets], device=device, dtype=torch.long)
    ends = torch.tensor([e for _, e in doc_offsets], device=device, dtype=torch.long)
    doc_id_per_tok = torch.zeros(n_tok, device=device, dtype=torch.long)
    for di, (s, e) in enumerate(doc_offsets):
        doc_id_per_tok[s:e] = di

    all_ids: list[list[int]] = []
    for q in query_vecs:
        q_t = torch.from_numpy(q).to(device).float()  # (Sq, dim)
        sim = q_t @ flat.T  # (Sq, n_tok)
        per_q_max = torch.full(
            (q_t.shape[0], n_docs), -1e30, device=device, dtype=torch.float32,
        )
        # scatter_reduce_ amax over tokens belonging to each doc
        idx = doc_id_per_tok.unsqueeze(0).expand(q_t.shape[0], -1)
        per_q_max.scatter_reduce_(1, idx, sim, reduce="amax", include_self=True)
        scores = per_q_max.sum(dim=0)  # (n_docs,)
        topv, topi = torch.topk(scores, k=min(top_k, n_docs))
        all_ids.append([int(x) for x in topi.cpu().numpy().tolist()])
    return all_ids


# ---------------------------------------------------------------------------
# rroq158 brute-force MaxSim using the production Triton kernel
# ---------------------------------------------------------------------------

def rroq158_maxsim_topk_gpu(
    query_vecs: list, all_vectors: np.ndarray, doc_offsets: list,
    top_k: int, device: str, rroq158_k: int, seed: int = 42,
) -> tuple[list[list[int]], dict]:
    """Encode the corpus once with rroq158, then for each query score against
    ALL docs using the Triton kernel. No LEMUR routing."""
    from colsearch._internal.kernels.triton_roq_rroq158 import roq_maxsim_rroq158

    cfg = Rroq158Config(K=rroq158_k, group_size=32, seed=seed)
    log.info("rroq158 encoding %d tokens, K=%d", all_vectors.shape[0], rroq158_k)
    enc = encode_rroq158(np.asarray(all_vectors, dtype=np.float32), cfg)

    # Pad per-doc to global max token count (NOT p95) — eliminate truncation
    # artifact entirely.
    n_docs = len(doc_offsets)
    tok_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int64)
    t_max_real = int(tok_counts.max())
    t_max = 1
    while t_max < t_max_real:
        t_max *= 2

    n_words = enc.sign_plane.shape[1]
    if n_words % 4 != 0:
        raise RuntimeError("sign plane n_bytes not multiple of 4")
    n_int32_words = n_words // 4
    n_groups = enc.scales.shape[1]

    sign_dt = np.zeros((n_docs, t_max, n_int32_words), dtype=np.int32)
    nz_dt = np.zeros((n_docs, t_max, n_int32_words), dtype=np.int32)
    scl_dt = np.zeros((n_docs, t_max, n_groups), dtype=np.float32)
    cid_dt = np.zeros((n_docs, t_max), dtype=np.int32)
    cosn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    sinn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    mask_dt = np.zeros((n_docs, t_max), dtype=np.float32)

    for di, (s, e) in enumerate(doc_offsets):
        n_tok = e - s
        if n_tok > t_max:
            raise RuntimeError(f"doc {di} has {n_tok} > t_max={t_max}")
        sign_dt[di, :n_tok] = pack_doc_codes_to_int32_words(enc.sign_plane[s:e])
        nz_dt[di, :n_tok] = pack_doc_codes_to_int32_words(enc.nonzero_plane[s:e])
        scl_dt[di, :n_tok] = enc.scales[s:e].astype(np.float32)
        cid_dt[di, :n_tok] = enc.centroid_id[s:e].astype(np.int32)
        cosn_dt[di, :n_tok] = enc.cos_norm[s:e].astype(np.float32)
        sinn_dt[di, :n_tok] = enc.sin_norm[s:e].astype(np.float32)
        mask_dt[di, :n_tok] = 1.0

    sign_g = torch.from_numpy(sign_dt).to(device)
    nz_g = torch.from_numpy(nz_dt).to(device)
    scl_g = torch.from_numpy(scl_dt).to(device)
    cid_g = torch.from_numpy(cid_dt).to(device)
    cosn_g = torch.from_numpy(cosn_dt).to(device)
    sinn_g = torch.from_numpy(sinn_dt).to(device)
    mask_g = torch.from_numpy(mask_dt).to(device)
    centroids_g = torch.from_numpy(enc.centroids).to(device).float()
    rotator = get_cached_fwht_rotator(dim=enc.dim, seed=enc.fwht_seed)

    log.info("rroq158 packed: docs=%d t_max=%d (max real=%d) n_groups=%d",
             n_docs, t_max, t_max_real, n_groups)

    all_ids: list[list[int]] = []
    for qi, q in enumerate(query_vecs):
        q_inputs = encode_query_for_rroq158(
            q, centroids=enc.centroids, fwht_seed=enc.fwht_seed,
            query_bits=4, rotator=rotator, skip_qc_table=False,
        )
        q_planes = torch.from_numpy(q_inputs["q_planes"]).to(device).unsqueeze(0)
        q_meta = torch.from_numpy(q_inputs["q_meta"]).to(device).unsqueeze(0)
        qc_table = torch.from_numpy(q_inputs["qc_table"]).to(device).unsqueeze(0)
        q_mask = torch.ones(1, q.shape[0], device=device, dtype=torch.float32)

        scores = roq_maxsim_rroq158(
            q_planes, q_meta, qc_table,
            cid_g, cosn_g, sinn_g, sign_g, nz_g, scl_g,
            queries_mask=q_mask, documents_mask=mask_g,
        )  # (1, n_docs)
        scores = scores.squeeze(0)
        topv, topi = torch.topk(scores, k=min(top_k, n_docs))
        all_ids.append([int(x) for x in topi.cpu().numpy().tolist()])

    info = {
        "K": rroq158_k,
        "n_groups": n_groups,
        "t_max": t_max,
        "t_max_real": int(t_max_real),
    }
    return all_ids, info


# ---------------------------------------------------------------------------
# rroq4_riem brute-force MaxSim using the production Triton kernel
# ---------------------------------------------------------------------------

def rroq4_riem_maxsim_topk_gpu(
    query_vecs: list, all_vectors: np.ndarray, doc_offsets: list,
    top_k: int, device: str, k_centroids: int = 8192,
    group_size: int = 32, seed: int = 42,
) -> tuple[list[list[int]], dict]:
    from colsearch._internal.kernels.triton_roq_rroq4_riem import (
        roq_maxsim_rroq4_riem,
    )

    cfg = Rroq4RiemConfig(K=k_centroids, group_size=group_size, seed=seed)
    log.info("rroq4_riem encoding %d tokens, K=%d, gs=%d",
             all_vectors.shape[0], k_centroids, group_size)
    enc = encode_rroq4_riem(np.asarray(all_vectors, dtype=np.float32), cfg)

    n_docs = len(doc_offsets)
    tok_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int64)
    t_max_real = int(tok_counts.max())
    t_max = 1
    while t_max < t_max_real:
        t_max *= 2

    n_groups = enc.mins.shape[1]
    nibble_bytes = enc.codes_packed.shape[1]

    cid_dt = np.zeros((n_docs, t_max), dtype=np.int32)
    cosn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    sinn_dt = np.zeros((n_docs, t_max), dtype=np.float32)
    codes_dt = np.zeros((n_docs, t_max, nibble_bytes), dtype=np.uint8)
    mins_dt = np.zeros((n_docs, t_max, n_groups), dtype=np.float32)
    deltas_dt = np.zeros((n_docs, t_max, n_groups), dtype=np.float32)
    mask_dt = np.zeros((n_docs, t_max), dtype=np.float32)

    for di, (s, e) in enumerate(doc_offsets):
        n_tok = e - s
        cid_dt[di, :n_tok] = enc.centroid_id[s:e].astype(np.int32)
        cosn_dt[di, :n_tok] = enc.cos_norm[s:e].astype(np.float32)
        sinn_dt[di, :n_tok] = enc.sin_norm[s:e].astype(np.float32)
        codes_dt[di, :n_tok] = enc.codes_packed[s:e]
        mins_dt[di, :n_tok] = enc.mins[s:e].astype(np.float32)
        deltas_dt[di, :n_tok] = enc.deltas[s:e].astype(np.float32)
        mask_dt[di, :n_tok] = 1.0

    cid_g = torch.from_numpy(cid_dt).to(device)
    cosn_g = torch.from_numpy(cosn_dt).to(device)
    sinn_g = torch.from_numpy(sinn_dt).to(device)
    codes_g = torch.from_numpy(codes_dt).to(device)
    mins_g = torch.from_numpy(mins_dt).to(device)
    deltas_g = torch.from_numpy(deltas_dt).to(device)
    mask_g = torch.from_numpy(mask_dt).to(device)
    rotator = get_cached_fwht_rotator_riem(dim=enc.dim, seed=enc.fwht_seed)

    log.info("rroq4_riem packed: docs=%d t_max=%d n_groups=%d",
             n_docs, t_max, n_groups)

    all_ids: list[list[int]] = []
    for q in query_vecs:
        q_inputs = encode_query_for_rroq4_riem(
            q, centroids=enc.centroids, fwht_seed=enc.fwht_seed,
            group_size=group_size, rotator=rotator, skip_qc_table=False,
        )
        q_rot = torch.from_numpy(q_inputs["q_rot"]).to(device).unsqueeze(0)
        q_gs = torch.from_numpy(q_inputs["q_group_sums"]).to(device).unsqueeze(0)
        qc_table = torch.from_numpy(q_inputs["qc_table"]).to(device).unsqueeze(0)
        q_mask = torch.ones(1, q.shape[0], device=device, dtype=torch.float32)

        scores = roq_maxsim_rroq4_riem(
            q_rot, q_gs, qc_table,
            cid_g, cosn_g, sinn_g, codes_g, mins_g, deltas_g,
            queries_mask=q_mask, documents_mask=mask_g,
            group_size=group_size,
        ).squeeze(0)
        topv, topi = torch.topk(scores, k=min(top_k, n_docs))
        all_ids.append([int(x) for x in topi.cpu().numpy().tolist()])

    info = {"K": k_centroids, "group_size": group_size,
            "n_groups": n_groups, "t_max": t_max,
            "t_max_real": int(t_max_real)}
    return all_ids, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="arguana")
    parser.add_argument("--n-queries", type=int, default=200,
                        help="Subset of queries to evaluate (full set is slow on GPU).")
    parser.add_argument("--rroq158-k", type=int, default=8192)
    parser.add_argument("--rroq4-riem-k", type=int, default=8192)
    parser.add_argument("--rroq4-riem-group-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-rroq158", action="store_true")
    parser.add_argument("--skip-rroq4-riem", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim = load_beir_npz(args.dataset)
    log.info("%s: %d docs, %d queries, dim=%d", args.dataset,
             len(doc_offsets), len(query_vecs), dim)

    # Use queries that have qrels (deterministic order)
    eval_qis = sorted([qi for qi in qrels.keys() if qi < len(query_vecs)])[:args.n_queries]
    eval_queries = [query_vecs[qi] for qi in eval_qis]
    log.info("Evaluating %d queries (subset of %d total)",
             len(eval_qis), len(query_vecs))

    # ---- A: fp16 brute-force MaxSim ground truth ----
    log.info("=" * 60)
    log.info("A) fp16 brute-force MaxSim on ALL %d docs", len(doc_offsets))
    log.info("=" * 60)
    t0 = time.time()
    fp_ids = fp16_maxsim_topk_gpu_batched(
        eval_queries, all_vectors, doc_offsets, args.top_k, device,
    )
    fp_time = time.time() - t0
    log.info("fp16 brute-force took %.1fs (%.2f q/s)", fp_time, len(eval_queries) / fp_time)

    # Re-key qrels to subset query indices for evaluate()
    sub_qrels = {i: qrels[qi] for i, qi in enumerate(eval_qis) if qi in qrels}
    fp_metrics = evaluate(fp_ids, sub_qrels, len(eval_queries))
    log.info("FP16 brute-force NDCG@10: %.4f  Recall@100: %.4f  Recall@10: %.4f",
             fp_metrics["NDCG@10"], fp_metrics["recall@100"], fp_metrics["recall@10"])

    rq_metrics = None
    rq_ids = None
    if not args.skip_rroq158:
        log.info("=" * 60)
        log.info("B) rroq158 brute-force MaxSim on ALL %d docs (K=%d)",
                 len(doc_offsets), args.rroq158_k)
        log.info("=" * 60)
        t0 = time.time()
        rq_ids, info = rroq158_maxsim_topk_gpu(
            eval_queries, all_vectors, doc_offsets, args.top_k, device,
            rroq158_k=args.rroq158_k, seed=args.seed,
        )
        rq_time = time.time() - t0
        log.info("rroq158 brute-force took %.1fs (%.2f q/s)  info=%s",
                 rq_time, len(eval_queries) / rq_time, info)
        rq_metrics = evaluate(rq_ids, sub_qrels, len(eval_queries))
        log.info("RROQ158 brute-force NDCG@10: %.4f  Recall@100: %.4f  Recall@10: %.4f",
                 rq_metrics["NDCG@10"], rq_metrics["recall@100"], rq_metrics["recall@10"])

    r4_metrics = None
    r4_ids = None
    if not args.skip_rroq4_riem:
        log.info("=" * 60)
        log.info("C) rroq4_riem brute-force MaxSim on ALL %d docs (K=%d, gs=%d)",
                 len(doc_offsets), args.rroq4_riem_k, args.rroq4_riem_group_size)
        log.info("=" * 60)
        t0 = time.time()
        r4_ids, info4 = rroq4_riem_maxsim_topk_gpu(
            eval_queries, all_vectors, doc_offsets, args.top_k, device,
            k_centroids=args.rroq4_riem_k, group_size=args.rroq4_riem_group_size,
            seed=args.seed,
        )
        r4_time = time.time() - t0
        log.info("rroq4_riem brute-force took %.1fs (%.2f q/s)  info=%s",
                 r4_time, len(eval_queries) / r4_time, info4)
        r4_metrics = evaluate(r4_ids, sub_qrels, len(eval_queries))
        log.info("RROQ4_RIEM brute-force NDCG@10: %.4f  Recall@100: %.4f  Recall@10: %.4f",
                 r4_metrics["NDCG@10"], r4_metrics["recall@100"], r4_metrics["recall@10"])

    def _overlap(a, b, k=10):
        if a is None or b is None:
            return None
        pct = []
        for x, y in zip(a, b):
            sx, sy = set(x[:k]), set(y[:k])
            pct.append(len(sx & sy) / k)
        return float(np.mean(pct)) if pct else 0.0

    print()
    print("=" * 78)
    print(f"DIAGNOSTIC: {args.dataset} | n_queries={len(eval_qis)} | top_k={args.top_k}")
    print("=" * 78)
    print(f"  fp16 brute-force                NDCG@10={fp_metrics['NDCG@10']:.4f}  R@100={fp_metrics['recall@100']:.4f}")
    if rq_metrics is not None:
        d_ndcg = rq_metrics["NDCG@10"] - fp_metrics["NDCG@10"]
        d_r100 = rq_metrics["recall@100"] - fp_metrics["recall@100"]
        print(f"  rroq158 K={args.rroq158_k:<5d}                  NDCG@10={rq_metrics['NDCG@10']:.4f}  R@100={rq_metrics['recall@100']:.4f}  Δ={d_ndcg:+.4f}/{d_r100:+.4f}  top10∩fp16={_overlap(fp_ids, rq_ids)*100:.1f}%")
    if r4_metrics is not None:
        d_ndcg = r4_metrics["NDCG@10"] - fp_metrics["NDCG@10"]
        d_r100 = r4_metrics["recall@100"] - fp_metrics["recall@100"]
        print(f"  rroq4_riem K={args.rroq4_riem_k:<5d} gs={args.rroq4_riem_group_size:<3d}        NDCG@10={r4_metrics['NDCG@10']:.4f}  R@100={r4_metrics['recall@100']:.4f}  Δ={d_ndcg:+.4f}/{d_r100:+.4f}  top10∩fp16={_overlap(fp_ids, r4_ids)*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
