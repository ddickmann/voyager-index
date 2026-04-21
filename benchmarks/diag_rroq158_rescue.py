"""rroq158 rescue experiment matrix on arguana brute-force MaxSim.

Tests four orthogonal levers for closing the rroq158 → fp16 NDCG@10 gap:

  1) K_centroids ∈ {8192, 16384}      — finer-grained tangent residuals
  2) query_bits ∈ {4, 6, 8}           — query-side fidelity (cheap; no disk hit)
  3) fp16 rerank top-{0, 32}          — rescore the rroq158 shortlist with fp16
  4) (skipped: group_size — dim=128 forces gs >= 32 because the popcount
      kernel needs >= 1 32-bit word per scale group)

Methodology:
  * brute-force MaxSim on ALL 8674 arguana docs per query (no LEMUR routing,
    no 2000-cap candidate set). isolates pure kernel fidelity from any
    wrapper / routing artifact.
  * fp16 brute-force scoring is the truth ranking; deltas are reported
    against it.
  * top-32 fp16 rerank: take the rroq158 top-32, rescore those 32 docs with
    full-fidelity fp16 MaxSim, re-sort. Rerank cost is bounded
    (32 * dim_max * Sq) — tiny vs the full-corpus rroq158 sweep.

Run:
    python benchmarks/diag_rroq158_rescue.py --n-queries 400 --output /tmp/rescue.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beir_benchmark import evaluate, load_beir_npz  # noqa: E402
from colsearch._internal.inference.quantization.rroq158 import (  # noqa: E402
    Rroq158Config,
    encode_query_for_rroq158,
    encode_rroq158,
    get_cached_fwht_rotator,
    pack_doc_codes_to_int32_words,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rescue")


# ---------------------------------------------------------------------------
# fp16 brute-force MaxSim (truth ranking) and per-candidate rerank
# ---------------------------------------------------------------------------

def fp16_maxsim_full_topk(
    query_vecs, all_vectors, doc_offsets, top_k, device,
):
    flat = torch.from_numpy(np.ascontiguousarray(all_vectors)).to(device).float()
    n_tok = flat.shape[0]
    n_docs = len(doc_offsets)
    doc_id_per_tok = torch.zeros(n_tok, device=device, dtype=torch.long)
    for di, (s, e) in enumerate(doc_offsets):
        doc_id_per_tok[s:e] = di

    all_ids = []
    for q in query_vecs:
        q_t = torch.from_numpy(q).to(device).float()
        sim = q_t @ flat.T
        per_q_max = torch.full(
            (q_t.shape[0], n_docs), -1e30, device=device, dtype=torch.float32,
        )
        idx = doc_id_per_tok.unsqueeze(0).expand(q_t.shape[0], -1)
        per_q_max.scatter_reduce_(1, idx, sim, reduce="amax", include_self=True)
        scores = per_q_max.sum(dim=0)
        topv, topi = torch.topk(scores, k=min(top_k, n_docs))
        all_ids.append([int(x) for x in topi.cpu().numpy().tolist()])
    return all_ids


def fp16_two_stage(
    query_vecs, base_ranking_per_query, all_vectors, doc_offsets,
    rerank_depth, final_top_k, device,
):
    """Two-stage retrieval: rescore the top ``rerank_depth`` docs of the base
    ranking with full fp16 MaxSim, then concatenate the rroq158 ranks
    [rerank_depth, final_top_k) unchanged. This preserves R@100 (presence)
    while fixing the within-top-K ordering."""
    flat = torch.from_numpy(np.ascontiguousarray(all_vectors)).to(device).float()
    starts = [s for s, _ in doc_offsets]
    ends = [e for _, e in doc_offsets]

    all_ids = []
    for q, base in zip(query_vecs, base_ranking_per_query):
        if not base:
            all_ids.append([])
            continue
        q_t = torch.from_numpy(q).to(device).float()
        head = base[:rerank_depth]
        tail = base[rerank_depth:final_top_k]

        head_scores = torch.empty(len(head), device=device, dtype=torch.float32)
        for i, di in enumerate(head):
            d_t = flat[starts[di]:ends[di]]
            sim = q_t @ d_t.T
            head_scores[i] = sim.max(dim=1).values.sum()
        order = torch.argsort(head_scores, descending=True)
        head_sorted = [int(head[i.item()]) for i in order]
        all_ids.append(head_sorted + list(tail))
    return all_ids


# ---------------------------------------------------------------------------
# rroq158 brute-force MaxSim, parameterised by K and query_bits
# ---------------------------------------------------------------------------

def _build_payload(all_vectors, doc_offsets, K, group_size, seed, device):
    cfg = Rroq158Config(K=K, group_size=group_size, seed=seed)
    log.info("encode rroq158: K=%d gs=%d, %d tokens", K, group_size, all_vectors.shape[0])
    t0 = time.time()
    enc = encode_rroq158(np.asarray(all_vectors, dtype=np.float32), cfg)
    encode_s = time.time() - t0
    log.info("encode took %.1fs", encode_s)

    n_docs = len(doc_offsets)
    tok_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int64)
    t_max_real = int(tok_counts.max())
    t_max = 1
    while t_max < t_max_real:
        t_max *= 2

    n_words = enc.sign_plane.shape[1]
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
        sign_dt[di, :n_tok] = pack_doc_codes_to_int32_words(enc.sign_plane[s:e])
        nz_dt[di, :n_tok] = pack_doc_codes_to_int32_words(enc.nonzero_plane[s:e])
        scl_dt[di, :n_tok] = enc.scales[s:e].astype(np.float32)
        cid_dt[di, :n_tok] = enc.centroid_id[s:e].astype(np.int32)
        cosn_dt[di, :n_tok] = enc.cos_norm[s:e].astype(np.float32)
        sinn_dt[di, :n_tok] = enc.sin_norm[s:e].astype(np.float32)
        mask_dt[di, :n_tok] = 1.0

    payload = {
        "sign_g": torch.from_numpy(sign_dt).to(device),
        "nz_g": torch.from_numpy(nz_dt).to(device),
        "scl_g": torch.from_numpy(scl_dt).to(device),
        "cid_g": torch.from_numpy(cid_dt).to(device),
        "cosn_g": torch.from_numpy(cosn_dt).to(device),
        "sinn_g": torch.from_numpy(sinn_dt).to(device),
        "mask_g": torch.from_numpy(mask_dt).to(device),
        "centroids": np.ascontiguousarray(enc.centroids, dtype=np.float32),
        "fwht_seed": enc.fwht_seed,
        "dim": enc.dim,
        "K": K,
        "group_size": group_size,
        "encode_s": encode_s,
        "n_groups": n_groups,
    }
    return payload


def rroq158_full_topk(query_vecs, payload, top_k, device, query_bits):
    from colsearch._internal.kernels.triton_roq_rroq158 import roq_maxsim_rroq158

    rotator = get_cached_fwht_rotator(dim=payload["dim"], seed=payload["fwht_seed"])
    n_docs = payload["sign_g"].shape[0]

    all_ids = []
    for q in query_vecs:
        q_inputs = encode_query_for_rroq158(
            q, centroids=payload["centroids"], fwht_seed=payload["fwht_seed"],
            query_bits=query_bits, rotator=rotator, skip_qc_table=False,
        )
        q_planes = torch.from_numpy(q_inputs["q_planes"]).to(device).unsqueeze(0)
        q_meta = torch.from_numpy(q_inputs["q_meta"]).to(device).unsqueeze(0)
        qc_table = torch.from_numpy(q_inputs["qc_table"]).to(device).unsqueeze(0)
        q_mask = torch.ones(1, q.shape[0], device=device, dtype=torch.float32)

        scores = roq_maxsim_rroq158(
            q_planes, q_meta, qc_table,
            payload["cid_g"], payload["cosn_g"], payload["sinn_g"],
            payload["sign_g"], payload["nz_g"], payload["scl_g"],
            queries_mask=q_mask, documents_mask=payload["mask_g"],
        ).squeeze(0)
        topv, topi = torch.topk(scores, k=min(top_k, n_docs))
        all_ids.append([int(x) for x in topi.cpu().numpy().tolist()])
    return all_ids


def _bytes_per_tok(K, dim, group_size):
    n_groups = dim // group_size
    n_words = (dim + 7) // 8
    return 2 * n_words + n_groups * 2 + 2 + 2 + 2  # signs+nz + scales(fp16) + cid(uint16) + cos+sin


def _overlap10(a, b):
    if a is None or b is None:
        return None
    return float(np.mean([len(set(x[:10]) & set(y[:10])) / 10.0 for x, y in zip(a, b)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="arguana")
    parser.add_argument("--n-queries", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/tmp/rescue.json"))
    parser.add_argument("--ks", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--query-bits", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--rerank-tops", type=int, nargs="+", default=[0, 32])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    all_vectors, doc_offsets, _, query_vecs, qrels, dim = load_beir_npz(args.dataset)
    log.info("%s: %d docs, %d total queries, dim=%d",
             args.dataset, len(doc_offsets), len(query_vecs), dim)

    eval_qis = sorted([qi for qi in qrels.keys() if qi < len(query_vecs)])[:args.n_queries]
    eval_queries = [query_vecs[qi] for qi in eval_qis]
    sub_qrels = {i: qrels[qi] for i, qi in enumerate(eval_qis) if qi in qrels}
    log.info("Evaluating %d queries", len(eval_qis))

    log.info("=" * 70)
    log.info("Truth ranking: fp16 brute-force MaxSim on ALL %d docs", len(doc_offsets))
    log.info("=" * 70)
    t0 = time.time()
    fp_ids = fp16_maxsim_full_topk(eval_queries, all_vectors, doc_offsets, 100, device)
    fp_time = time.time() - t0
    fp_metrics = evaluate(fp_ids, sub_qrels, len(eval_queries))
    log.info("FP16 brute-force NDCG@10=%.4f R@100=%.4f  (%.1fs, %.1f q/s)",
             fp_metrics["NDCG@10"], fp_metrics["recall@100"], fp_time,
             len(eval_queries) / fp_time)

    cells = []
    cells.append({
        "label": "fp16_brute_force",
        "K": None, "query_bits": None, "rerank_top": 0,
        "ndcg_at_10": fp_metrics["NDCG@10"],
        "recall_at_10": fp_metrics["recall@10"],
        "recall_at_100": fp_metrics["recall@100"],
        "delta_ndcg_vs_fp16": 0.0,
        "top10_overlap_with_fp16": 1.0,
        "bytes_per_tok": dim * 2,
    })

    # Cache payloads per K so we don't re-encode for each query_bits sweep
    payloads = {}
    for K in args.ks:
        payloads[K] = _build_payload(
            all_vectors, doc_offsets, K=K, group_size=32, seed=args.seed,
            device=device,
        )

    for K in args.ks:
        payload = payloads[K]
        for qb in args.query_bits:
            log.info("=" * 70)
            log.info("CELL: K=%d query_bits=%d", K, qb)
            log.info("=" * 70)
            t0 = time.time()
            rq_ids = rroq158_full_topk(eval_queries, payload, 100, device, query_bits=qb)
            rq_time = time.time() - t0
            rq_metrics = evaluate(rq_ids, sub_qrels, len(eval_queries))
            d_ndcg = rq_metrics["NDCG@10"] - fp_metrics["NDCG@10"]
            ovr = _overlap10(fp_ids, rq_ids)
            log.info("  NDCG@10=%.4f (Δ=%+.4f) R@100=%.4f overlap10=%.1f%%  (%.1fs)",
                     rq_metrics["NDCG@10"], d_ndcg, rq_metrics["recall@100"],
                     ovr * 100, rq_time)
            cells.append({
                "label": f"K={K}_qb={qb}",
                "K": K, "query_bits": qb, "rerank_top": 0,
                "ndcg_at_10": rq_metrics["NDCG@10"],
                "recall_at_10": rq_metrics["recall@10"],
                "recall_at_100": rq_metrics["recall@100"],
                "delta_ndcg_vs_fp16": d_ndcg,
                "top10_overlap_with_fp16": ovr,
                "search_s": rq_time,
                "qps_kernel_only": len(eval_queries) / rq_time,
                "bytes_per_tok": _bytes_per_tok(K, dim, 32),
            })

            for rerank_top in args.rerank_tops:
                if rerank_top == 0:
                    continue
                t1 = time.time()
                rerank_ids = fp16_two_stage(
                    eval_queries, rq_ids, all_vectors, doc_offsets,
                    rerank_depth=rerank_top, final_top_k=100, device=device,
                )
                rerank_time = time.time() - t1
                rerank_metrics = evaluate(rerank_ids, sub_qrels, len(eval_queries))
                d_ndcg2 = rerank_metrics["NDCG@10"] - fp_metrics["NDCG@10"]
                ovr2 = _overlap10(fp_ids, rerank_ids)
                log.info("    + fp16 rerank top-%d → NDCG@10=%.4f (Δ=%+.4f) R@100=%.4f overlap10=%.1f%% (+%.1fs)",
                         rerank_top, rerank_metrics["NDCG@10"], d_ndcg2,
                         rerank_metrics["recall@100"], ovr2 * 100, rerank_time)
                cells.append({
                    "label": f"K={K}_qb={qb}_rerank{rerank_top}",
                    "K": K, "query_bits": qb, "rerank_top": rerank_top,
                    "ndcg_at_10": rerank_metrics["NDCG@10"],
                    "recall_at_10": rerank_metrics["recall@10"],
                    "recall_at_100": rerank_metrics["recall@100"],
                    "delta_ndcg_vs_fp16": d_ndcg2,
                    "top10_overlap_with_fp16": ovr2,
                    "search_s": rq_time + rerank_time,
                    "qps_with_rerank": len(eval_queries) / (rq_time + rerank_time),
                    "rerank_only_s": rerank_time,
                    "bytes_per_tok": _bytes_per_tok(K, dim, 32),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump({
            "dataset": args.dataset,
            "n_queries": len(eval_qis),
            "fp16_baseline": {
                "ndcg_at_10": fp_metrics["NDCG@10"],
                "recall_at_100": fp_metrics["recall@100"],
            },
            "cells": cells,
        }, fh, indent=2)

    print()
    print("=" * 90)
    print(f"RESCUE MATRIX: {args.dataset} | n_queries={len(eval_qis)}")
    print("=" * 90)
    print(f"{'cell':<32} {'NDCG@10':>9} {'Δ vs fp16':>10} {'R@100':>8} "
          f"{'top10∩fp16':>11} {'B/tok':>7}")
    print("-" * 90)
    for c in cells:
        print(f"{c['label']:<32} {c['ndcg_at_10']:>9.4f} {c['delta_ndcg_vs_fp16']:>+10.4f} "
              f"{c['recall_at_100']:>8.4f} {c['top10_overlap_with_fp16']*100:>10.1f}% "
              f"{c['bytes_per_tok']:>7d}")
    print()
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
