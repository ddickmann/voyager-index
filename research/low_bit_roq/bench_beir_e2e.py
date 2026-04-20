"""
End-to-end BEIR retrieval test: ternary + rroq158 + MV distill vs the
production fp16 baseline, on a real BEIR dataset.

What this measures:
  - Real Recall@10 / NDCG@10 / MAP@10 against the qrels for each compression
  - Disk footprint per token for each
  - The actual production loss from going ternary

What this skips (intentionally):
  - LEMUR routing (uses brute-force MaxSim on the full corpus)
  - Multi-shard fetch pipeline
  - Triton kernels (uses numpy/torch reference scoring; same ranking output)

The skipped pieces are what Phase A5 will plumb. They affect *latency* but
not retrieval quality: with brute-force scoring on the full corpus, the
Recall numbers are the upper bound of what the same compression will give
once routed.

Memory: nfcorpus is the smallest BEIR fixture (3.6k docs / 864k tokens /
220 MB fp16). Peak working set ~1 GB, well under the 24 GB cap.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.shard_bench.metrics import compute_all_metrics  # noqa: E402

log = logging.getLogger(__name__)

BEIR_CACHE = Path("/root/.cache/voyager-qa/beir")


@dataclass
class BeirRow:
    method: str
    bits_per_coord: float
    bytes_per_token: float
    extra_bytes_per_token: float
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    map_at_10: float
    mrr_at_10: float
    avg_score_ms: float
    notes: str


# ---------------------------------------------------------------------------
# Data loader (mirrors benchmarks/beir_benchmark.py.load_beir_npz)
# ---------------------------------------------------------------------------


def load_nfcorpus(npz_path: Path):
    npz = np.load(str(npz_path), allow_pickle=True)
    n_docs = int(npz["n_docs"])
    dim = int(npz["dim"])
    doc_offsets_arr = npz["doc_offsets"]
    last_vec = int(doc_offsets_arr[-1][1])
    all_vectors = npz["doc_vectors"][:last_vec]
    doc_offsets = [(int(s), int(e)) for s, e in doc_offsets_arr]
    query_offsets = npz["query_offsets"]
    all_q = npz["query_vectors"]
    queries = [all_q[int(s):int(e)].astype(np.float32) for s, e in query_offsets]
    qrels_mat = npz["qrels"]
    grades_mat = npz["qrel_grades"]
    qrels = {}
    for qi in range(qrels_mat.shape[0]):
        rels = {}
        for ri in range(qrels_mat.shape[1]):
            doc_idx = int(qrels_mat[qi, ri])
            if 0 <= doc_idx < n_docs:
                grade = int(grades_mat[qi, ri])
                rels[doc_idx] = max(grade, 1)
        if rels:
            qrels[qi] = rels
    return all_vectors.astype(np.float32), doc_offsets, queries, qrels, dim


# ---------------------------------------------------------------------------
# MaxSim brute-force (numpy)
# ---------------------------------------------------------------------------


def maxsim_score(query_tokens: np.ndarray, doc_tok_matrix: np.ndarray,
                 doc_offsets: list[tuple[int, int]]) -> np.ndarray:
    """Returns score[doc_id] = sum_q max_d <q, doc_tok_d>.

    query_tokens: (n_q_tokens, dim)
    doc_tok_matrix: (n_total_tokens, dim) — concatenation
    doc_offsets: list of (start, end) per doc
    """
    sims = query_tokens @ doc_tok_matrix.T                    # (n_q_tok, n_total_tok)
    scores = np.empty(len(doc_offsets), dtype=np.float32)
    for di, (s, e) in enumerate(doc_offsets):
        if s == e:
            scores[di] = -1e9
            continue
        max_per_q = sims[:, s:e].max(axis=1)
        scores[di] = max_per_q.sum()
    return scores


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def _ternary_encode_decode(tokens: np.ndarray, *, group_size: int, seed: int):
    """Returns dict with: rotated tokens, decoded (in rotated space), rotator fn."""
    from research.low_bit_roq.ternary import TernaryConfig, TernaryQuantizer
    cfg = TernaryConfig(
        dim=tokens.shape[1], group_size=group_size, rotate=True,
        tau_frac=0.5, seed=seed,
    )
    q = TernaryQuantizer(cfg)
    enc = q.quantize(tokens)
    decoded = q.decode(enc)        # in rotated space
    return {"rotator": q._rotate, "decoded_rot": decoded, "rotated": enc["rotated"]}


def _roq4_decode(tokens: np.ndarray, *, group_size: int, seed: int):
    from voyager_index._internal.inference.quantization.rotational import (
        RoQConfig, RotationalQuantizer,
    )
    import torch
    cfg = RoQConfig(dim=tokens.shape[1], num_bits=4, num_rounds=3,
                    seed=seed, group_size=group_size)
    q = RotationalQuantizer(cfg)
    res = q.quantize(tokens, store=False)
    decoded = q.decode(res["codes"], res["scales"], res["offsets"]).cpu().numpy()
    decoded = decoded[:, : tokens.shape[1]]
    decoded /= np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-12
    return decoded.astype(np.float32) * np.linalg.norm(tokens, axis=1, keepdims=True)


def _rroq158_encode(
    tokens: np.ndarray, *, K: int, group_size: int, seed: int,
    fit_sample_cap: int = 100_000, encode_chunk: int = 32_768,
):
    """Returns rroq158 encoded payload + reconstructed ambient docs.

    Memory-conscious variant: fits centroids on a subsample, assigns + encodes
    the full corpus in chunks of ``encode_chunk`` tokens. Peak extra
    allocation = ``encode_chunk * K * 4`` bytes (128 MB at K=1024,
    chunk=32K).
    """
    from research.low_bit_roq.spherical_kmeans import (
        SphericalKMeansConfig, cluster, l2_normalize,
    )
    from research.low_bit_roq.tangent_query import log_map_unit_sphere
    from research.low_bit_roq.anisotropic import fit_anisotropic_min_max  # noqa: F401
    from research.low_bit_roq.ternary import TernaryConfig, TernaryQuantizer as TQ

    rng = np.random.default_rng(seed)
    n = tokens.shape[0]
    fit_idx = rng.choice(n, size=min(fit_sample_cap, n), replace=False)
    fit_tokens = tokens[fit_idx].astype(np.float32)
    log.info("  rroq158 fit: %d/%d tokens, K=%d", fit_tokens.shape[0], n, K)

    sk_cfg = SphericalKMeansConfig(n_clusters=K, n_iter=15, seed=seed)
    centroids, _ = cluster(fit_tokens, sk_cfg)
    del fit_tokens; gc.collect()

    tq = TQ(TernaryConfig(dim=tokens.shape[1], group_size=group_size,
                          rotate=False, fit_method="anisotropic"))

    centroid_id = np.empty(n, dtype=np.int32)
    norms = np.linalg.norm(tokens, axis=1).astype(np.float32)
    log.info("  rroq158 encode: chunked assign+quantize, chunk=%d", encode_chunk)
    sign_planes = []; nonzero_planes = []; scales_all = []; r_amb_chunks = []
    for s in range(0, n, encode_chunk):
        e = min(s + encode_chunk, n)
        chunk = tokens[s:e].astype(np.float32)
        chunk_n = l2_normalize(chunk)
        sims_chunk = chunk_n @ centroids.T              # (chunk, K)
        cid_chunk = sims_chunk.argmax(axis=1).astype(np.int32)
        centroid_id[s:e] = cid_chunk
        del sims_chunk
        c_per_tok = centroids[cid_chunk]
        tangent = log_map_unit_sphere(c_per_tok, chunk_n)
        enc = tq.quantize(tangent.astype(np.float32))
        sign_planes.append(enc["sign_plane"])
        nonzero_planes.append(enc["nonzero_plane"])
        scales_all.append(enc["scales"])
        r_amb_chunks.append(tq.decode({"sign_plane": enc["sign_plane"],
                                       "nonzero_plane": enc["nonzero_plane"],
                                       "scales": enc["scales"]}))
    code_payload = {
        "sign_plane": np.concatenate(sign_planes, axis=0),
        "nonzero_plane": np.concatenate(nonzero_planes, axis=0),
        "scales": np.concatenate(scales_all, axis=0),
    }
    r_amb = np.concatenate(r_amb_chunks, axis=0)
    del sign_planes, nonzero_planes, scales_all, r_amb_chunks; gc.collect()

    log.info("  rroq158 reconstruct ambient docs")
    c_per_doc = centroids[centroid_id]
    r_norm = np.linalg.norm(r_amb, axis=1, keepdims=True) + 1e-12
    cos_t = np.cos(r_norm); sin_t = np.sinc(r_norm / np.pi)
    recon_n = cos_t * c_per_doc + sin_t * r_amb
    recon = recon_n * norms[:, None]
    del c_per_doc, recon_n; gc.collect()
    return {
        "centroids": centroids,
        "centroid_id": centroid_id,
        "r_amb": r_amb,
        "norms": norms,
        "recon": recon.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Multi-view distill scoring (rroq158 score + ternary score → MLP rerank)
# ---------------------------------------------------------------------------


def _build_mv_features_for_doc_tokens(
    queries_n: np.ndarray,            # (n_q_tok, dim) L2-normalized
    rroq_payload: dict,
    ternary_decoded_rot: np.ndarray,  # (n_total_tok, dim) ternary decoded in rotated space
    ternary_rotator,
    queries_raw: np.ndarray,          # (n_q_tok, dim) raw queries (for ternary rotation)
) -> np.ndarray:
    """Per-query-token × per-doc-token features.

    F=6: [score_rroq, qc, qr, ||r̂||, |qd-qc|, score_ternary]

    Shape (n_q_tok, n_total_tok, 6).
    Memory: at 32 q tokens × 864K doc tokens × 6 × 4B = 660 MB — too much!

    To stay in budget we compute features in chunks of doc tokens.
    """
    raise NotImplementedError("Use _mv_score_for_doc_tokens chunked variant")


def _train_mv_distill_on_pairs(
    train_queries: list[np.ndarray],   # per query: (q_tok, dim)
    rroq_payload: dict,
    ternary_decoded_rot: np.ndarray,
    ternary_rotator,
    doc_offsets: list[tuple[int, int]],
    qrels: dict,
    all_vecs: np.ndarray,
    *,
    rerank_topk: int = 200,
    epochs: int = 6,
    hidden: int = 32,
    seed: int = 0,
):
    """Train on what production sees: the rroq158 top-K shortlist for each
    train query, with binary labels from qrels (positive if doc in qrels).

    This makes train and eval distributions match. The earlier
    "pos vs random-neg" training broke at inference because the eval
    candidates are all "rroq158 top-200" — never random.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    # Build the dense token matrices we score against.
    rroq_recon = rroq_payload["recon"]                     # (n_tok, dim)
    centroids_per_doc_tok = rroq_payload["centroids"][rroq_payload["centroid_id"]]
    r_amb = rroq_payload["r_amb"]
    r_norm_tok = np.linalg.norm(r_amb, axis=1).astype(np.float32)

    examples = []
    for qi, q in enumerate(train_queries):
        if qi not in qrels:
            continue
        pos_set = set(qrels[qi].keys())
        if not pos_set:
            continue
        # Build the rroq158 shortlist for THIS query (mirrors production).
        scores_full = maxsim_score(q, rroq_recon, doc_offsets)
        topk = np.argpartition(-scores_full, kth=min(rerank_topk - 1, len(scores_full) - 1))
        topk = topk[:rerank_topk]
        # Skip queries whose positives don't overlap the shortlist (nothing to learn).
        if not (pos_set & set(topk.tolist())):
            continue

        q_n = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        q_rot = ternary_rotator(q)
        q_rot_n = q_rot / (np.linalg.norm(q_rot, axis=1, keepdims=True) + 1e-12)

        for d in topk:
            label = 1.0 if int(d) in pos_set else 0.0
            s, e = doc_offsets[int(d)]
            if s == e:
                continue
            doc_recon = rroq_recon[s:e]
            doc_recon_n = doc_recon / (np.linalg.norm(doc_recon, axis=1, keepdims=True) + 1e-12)
            score_rroq = (q_n @ doc_recon_n.T).max(axis=1).sum()
            doc_ter = ternary_decoded_rot[s:e]
            doc_ter_n = doc_ter / (np.linalg.norm(doc_ter, axis=1, keepdims=True) + 1e-12)
            score_ter = (q_rot_n @ doc_ter_n.T).max(axis=1).sum()
            qc = (q_n @ centroids_per_doc_tok[s:e].T).max(axis=1).sum()
            r_norm_avg = float(r_norm_tok[s:e].mean())
            disagree = float(abs(score_rroq - score_ter))
            examples.append((float(score_rroq), float(score_ter), float(qc),
                             r_norm_avg, disagree, label))
    if not examples:
        log.warning("no training examples — falling back to no MV head")
        return None

    feats = np.array([[e[0], e[1], e[2], e[3], e[4]] for e in examples], dtype=np.float32)
    labels = np.array([e[5] for e in examples], dtype=np.float32)
    log.info("MV distill: %d training pairs (positives=%d, negatives=%d)",
             len(examples), int(labels.sum()), int((1 - labels).sum()))

    f_t = torch.from_numpy(feats); l_t = torch.from_numpy(labels)
    mean = f_t.mean(dim=0); std = f_t.std(dim=0).clamp_min(1e-6)
    f_n = (f_t - mean) / std

    model = nn.Sequential(
        nn.Linear(5, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    n = f_n.shape[0]; bsz = 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for s in range(0, n, bsz):
            idx = perm[s:s+bsz]
            logits = model(f_n[idx]).squeeze(-1)
            loss = bce(logits, l_t[idx])
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += float(loss.detach()) * idx.numel()
        log.info("    distill epoch %d  bce=%.4f", ep, ep_loss / n)

    def score_fn(features_np: np.ndarray) -> np.ndarray:
        ft = torch.from_numpy(features_np)
        ft = (ft - mean) / std
        with torch.no_grad():
            return model(ft).squeeze(-1).cpu().numpy()
    return score_fn


def _mv_rerank_topk(
    query_n: np.ndarray, query_rot_n: np.ndarray,
    candidate_doc_ids: np.ndarray,
    rroq_payload: dict,
    ternary_decoded_rot: np.ndarray,
    doc_offsets: list[tuple[int, int]],
    score_fn,
    base_score_per_doc: np.ndarray,
) -> np.ndarray:
    """For each candidate, build the 5-feature pair and rerank by MV score.
    Returns reranked doc ids, length == len(candidate_doc_ids).
    """
    rroq_recon = rroq_payload["recon"]
    centroids_per_doc_tok = rroq_payload["centroids"][rroq_payload["centroid_id"]]
    r_amb = rroq_payload["r_amb"]
    r_norm_tok = np.linalg.norm(r_amb, axis=1).astype(np.float32)

    feats = np.empty((len(candidate_doc_ids), 5), dtype=np.float32)
    for i, d in enumerate(candidate_doc_ids):
        s, e = doc_offsets[int(d)]
        if s == e:
            feats[i] = 0.0
            continue
        doc_recon = rroq_recon[s:e]
        doc_recon_n = doc_recon / (np.linalg.norm(doc_recon, axis=1, keepdims=True) + 1e-12)
        score_rroq = (query_n @ doc_recon_n.T).max(axis=1).sum()
        doc_ter = ternary_decoded_rot[s:e]
        doc_ter_n = doc_ter / (np.linalg.norm(doc_ter, axis=1, keepdims=True) + 1e-12)
        score_ter = (query_rot_n @ doc_ter_n.T).max(axis=1).sum()
        qc = (query_n @ centroids_per_doc_tok[s:e].T).max(axis=1).sum()
        r_norm_avg = float(r_norm_tok[s:e].mean())
        disagree = float(abs(score_rroq - score_ter))
        feats[i] = [score_rroq, score_ter, qc, r_norm_avg, disagree]

    if score_fn is None:
        order = np.argsort(-base_score_per_doc[candidate_doc_ids])
    else:
        scores = score_fn(feats)
        order = np.argsort(-scores)
    return candidate_doc_ids[order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--n-train-queries", type=int, default=200)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--rerank-topk", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("research/low_bit_roq/reports/beir_e2e.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    npz_path = BEIR_CACHE / f"{args.dataset}.npz"
    log.info("loading %s", npz_path)
    all_vecs, doc_offsets, queries, qrels, dim = load_nfcorpus(npz_path)
    log.info("%s: %d docs, %d query-token sets, dim=%d, %d total doc tokens",
             args.dataset, len(doc_offsets), len(queries), dim, all_vecs.shape[0])

    n_eval_queries = len(queries) - args.n_train_queries
    log.info("split: %d train, %d eval queries", args.n_train_queries, n_eval_queries)
    train_queries = queries[: args.n_train_queries]
    eval_queries = queries[args.n_train_queries :]
    eval_qids = list(range(args.n_train_queries, len(queries)))
    eval_relevant = [qrels.get(qi, {}) for qi in eval_qids]

    rows: list[BeirRow] = []

    # ----- fp16 baseline (no compression) -----
    log.info("scoring fp16 baseline on %d eval queries", n_eval_queries)
    t0 = time.perf_counter()
    fp16_topk_per_query = []
    for q in eval_queries:
        scores = maxsim_score(q, all_vecs, doc_offsets)
        top100 = np.argpartition(-scores, kth=99)[:100]
        top100 = top100[np.argsort(-scores[top100])]
        fp16_topk_per_query.append(top100.tolist())
    fp16_elapsed = time.perf_counter() - t0
    fp16_metrics = compute_all_metrics(fp16_topk_per_query, eval_relevant, ks=(10, 100))
    rows.append(BeirRow(
        method="fp16-bruteforce", bits_per_coord=16.0,
        bytes_per_token=dim * 2, extra_bytes_per_token=0,
        recall_at_10=fp16_metrics["recall@10"], recall_at_100=fp16_metrics["recall@100"],
        ndcg_at_10=fp16_metrics["NDCG@10"], map_at_10=fp16_metrics["MAP@10"],
        mrr_at_10=fp16_metrics["MRR@10"],
        avg_score_ms=1000 * fp16_elapsed / n_eval_queries,
        notes="reference ceiling (matches 0.340 published Recall@10 if all 323 queries used)",
    ))
    log.info("  fp16: R@10=%.4f NDCG@10=%.4f MAP@10=%.4f  %.1f ms/query",
             rows[-1].recall_at_10, rows[-1].ndcg_at_10, rows[-1].map_at_10,
             rows[-1].avg_score_ms)

    # ----- ROQ4 baseline -----
    log.info("encoding with roq4")
    roq4_decoded = _roq4_decode(all_vecs, group_size=args.group_size, seed=args.seed)
    log.info("scoring roq4 on %d eval queries", n_eval_queries)
    t0 = time.perf_counter()
    roq4_topk_per_query = []
    for q in eval_queries:
        scores = maxsim_score(q, roq4_decoded, doc_offsets)
        top100 = np.argpartition(-scores, kth=99)[:100]
        top100 = top100[np.argsort(-scores[top100])]
        roq4_topk_per_query.append(top100.tolist())
    roq4_elapsed = time.perf_counter() - t0
    roq4_metrics = compute_all_metrics(roq4_topk_per_query, eval_relevant, ks=(10, 100))
    rows.append(BeirRow(
        method="roq4", bits_per_coord=4.0,
        bytes_per_token=dim * 4 / 8, extra_bytes_per_token=0,
        recall_at_10=roq4_metrics["recall@10"], recall_at_100=roq4_metrics["recall@100"],
        ndcg_at_10=roq4_metrics["NDCG@10"], map_at_10=roq4_metrics["MAP@10"],
        mrr_at_10=roq4_metrics["MRR@10"],
        avg_score_ms=1000 * roq4_elapsed / n_eval_queries,
        notes="current production-grade quantization",
    ))
    log.info("  roq4: R@10=%.4f NDCG@10=%.4f MAP@10=%.4f  %.1f ms/query",
             rows[-1].recall_at_10, rows[-1].ndcg_at_10, rows[-1].map_at_10,
             rows[-1].avg_score_ms)
    del roq4_decoded; gc.collect()

    # ----- ternary baseline -----
    log.info("encoding with ternary (1.58-bit)")
    ter = _ternary_encode_decode(all_vecs, group_size=args.group_size, seed=args.seed)
    log.info("scoring ternary on %d eval queries", n_eval_queries)
    t0 = time.perf_counter()
    ter_topk_per_query = []
    for q in eval_queries:
        q_rot = ter["rotator"](q)
        scores = maxsim_score(q_rot, ter["decoded_rot"], doc_offsets)
        top100 = np.argpartition(-scores, kth=99)[:100]
        top100 = top100[np.argsort(-scores[top100])]
        ter_topk_per_query.append(top100.tolist())
    ter_elapsed = time.perf_counter() - t0
    ter_metrics = compute_all_metrics(ter_topk_per_query, eval_relevant, ks=(10, 100))
    rows.append(BeirRow(
        method="ternary", bits_per_coord=1.58,
        bytes_per_token=dim * 1.58 / 8, extra_bytes_per_token=0,
        recall_at_10=ter_metrics["recall@10"], recall_at_100=ter_metrics["recall@100"],
        ndcg_at_10=ter_metrics["NDCG@10"], map_at_10=ter_metrics["MAP@10"],
        mrr_at_10=ter_metrics["MRR@10"],
        avg_score_ms=1000 * ter_elapsed / n_eval_queries,
        notes="A2.5 candidate (1.58-bit asymmetric kernel)",
    ))
    log.info("  ternary: R@10=%.4f NDCG@10=%.4f MAP@10=%.4f  %.1f ms/query",
             rows[-1].recall_at_10, rows[-1].ndcg_at_10, rows[-1].map_at_10,
             rows[-1].avg_score_ms)

    # ----- rroq158-K* -----
    log.info("encoding with rroq158-K%d", args.K)
    rroq_payload = _rroq158_encode(all_vecs, K=args.K, group_size=args.group_size, seed=args.seed)
    log.info("scoring rroq158 on %d eval queries", n_eval_queries)
    t0 = time.perf_counter()
    rroq_topk_per_query = []
    rroq_scores_per_query = []                    # (n_eval, n_docs) — needed for X1 rerank
    for q in eval_queries:
        scores = maxsim_score(q, rroq_payload["recon"], doc_offsets)
        rroq_scores_per_query.append(scores)
        top100 = np.argpartition(-scores, kth=99)[:100]
        top100 = top100[np.argsort(-scores[top100])]
        rroq_topk_per_query.append(top100.tolist())
    rroq_elapsed = time.perf_counter() - t0
    rroq_metrics = compute_all_metrics(rroq_topk_per_query, eval_relevant, ks=(10, 100))
    extra_b = int(np.ceil(np.log2(args.K))) / 8
    rows.append(BeirRow(
        method=f"rroq158-K{args.K}", bits_per_coord=1.58,
        bytes_per_token=dim * 1.58 / 8, extra_bytes_per_token=extra_b,
        recall_at_10=rroq_metrics["recall@10"], recall_at_100=rroq_metrics["recall@100"],
        ndcg_at_10=rroq_metrics["NDCG@10"], map_at_10=rroq_metrics["MAP@10"],
        mrr_at_10=rroq_metrics["MRR@10"],
        avg_score_ms=1000 * rroq_elapsed / n_eval_queries,
        notes=f"K={args.K} centroid + tangent residual; +{extra_b:.2f} B/tok",
    ))
    log.info("  rroq158-K%d: R@10=%.4f NDCG@10=%.4f MAP@10=%.4f  %.1f ms/query",
             args.K, rows[-1].recall_at_10, rows[-1].ndcg_at_10, rows[-1].map_at_10,
             rows[-1].avg_score_ms)

    # ----- ternary + rroq158 + MV distill rerank -----
    log.info("training MV distill on %d train queries with positives from qrels",
             args.n_train_queries)
    score_fn = _train_mv_distill_on_pairs(
        train_queries, rroq_payload, ter["decoded_rot"], ter["rotator"],
        doc_offsets, qrels, all_vecs,
        rerank_topk=args.rerank_topk, epochs=6, seed=args.seed,
    )

    log.info("rerank: rroq158 top-%d → MV distill", args.rerank_topk)
    t0 = time.perf_counter()
    mv_topk_per_query = []
    for qi_local, q in enumerate(eval_queries):
        candidates = np.array(rroq_topk_per_query[qi_local][: args.rerank_topk])
        if len(candidates) < 100:
            mv_topk_per_query.append(rroq_topk_per_query[qi_local])
            continue
        q_n = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        q_rot = ter["rotator"](q)
        q_rot_n = q_rot / (np.linalg.norm(q_rot, axis=1, keepdims=True) + 1e-12)
        reranked = _mv_rerank_topk(
            q_n, q_rot_n, candidates, rroq_payload, ter["decoded_rot"],
            doc_offsets, score_fn, rroq_scores_per_query[qi_local],
        )
        mv_topk_per_query.append(reranked[:100].tolist())
    mv_elapsed = time.perf_counter() - t0
    mv_metrics = compute_all_metrics(mv_topk_per_query, eval_relevant, ks=(10, 100))
    rows.append(BeirRow(
        method=f"rroq158-K{args.K}+ternary+MV-distill", bits_per_coord=1.58,
        bytes_per_token=dim * 1.58 / 8, extra_bytes_per_token=extra_b,
        recall_at_10=mv_metrics["recall@10"], recall_at_100=mv_metrics["recall@100"],
        ndcg_at_10=mv_metrics["NDCG@10"], map_at_10=mv_metrics["MAP@10"],
        mrr_at_10=mv_metrics["MRR@10"],
        avg_score_ms=1000 * (rroq_elapsed + mv_elapsed) / n_eval_queries,
        notes=f"top-{args.rerank_topk} from rroq158, MV MLP rerank trained on qrels",
    ))
    log.info("  MV-rerank: R@10=%.4f NDCG@10=%.4f MAP@10=%.4f  %.1f ms/query (incl. base)",
             rows[-1].recall_at_10, rows[-1].ndcg_at_10, rows[-1].map_at_10,
             rows[-1].avg_score_ms)

    # ----- summary table -----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"dataset": args.dataset,
                                    "n_eval_queries": n_eval_queries,
                                    "rows": [asdict(r) for r in rows]},
                                   indent=2, default=str), encoding="utf-8")

    fp16_r10 = rows[0].recall_at_10
    roq4_r10 = rows[1].recall_at_10
    log.info("=== %s end-to-end Recall@10 (n_eval=%d) ===", args.dataset, n_eval_queries)
    log.info("%-40s %8s %8s %8s %8s %12s %12s",
             "method", "R@10", "R@100", "NDCG@10", "MAP@10", "B/tok", "ms/query")
    for r in rows:
        delta_fp16 = r.recall_at_10 - fp16_r10
        delta_roq4 = r.recall_at_10 - roq4_r10
        log.info("%-40s %8.4f %8.4f %8.4f %8.4f %12.2f %12.2f   (fp16Δ%+ .4f roq4Δ%+ .4f)",
                 r.method, r.recall_at_10, r.recall_at_100, r.ndcg_at_10,
                 r.map_at_10, r.bytes_per_token + r.extra_bytes_per_token,
                 r.avg_score_ms, delta_fp16, delta_roq4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
