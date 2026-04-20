"""
"How much of the ternary→4-bit ranking gap can we recover?"

Composes the three orthogonal recovery tools that should layer on top of
the ternary base:

  B0  tangent-aware rerank score  s = <q, d> - λ · θ²(q, d)
  B3  rroq158                     centroid + tangent-residual ternary
  X1  distillation reranker       2-layer MLP on (score, ||r̂||, <q,c>, ...)

For each condition we measure ``rank_corr@100`` against the fp16 ceiling
on the same shortlist so the numbers are directly comparable to the
``bench_bitwidth_compare`` results (`ternary=0.242`, `roq4=0.620`).

Memory plan: ~250 MB peak. Runs in <60 s on CPU + small CPU MLP.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RecoveryRow:
    name: str
    bits: float
    centroids_K: int
    rank_corr_top100: float
    nn5_excl_self: float
    nn50_excl_self: float
    extra_bytes_per_token: float
    notes: str


def _rank_corr_top100(
    true_ips: np.ndarray, approx_ips: np.ndarray, q_idx: np.ndarray
) -> float:
    n_q, n_t = true_ips.shape
    if n_t <= 2:
        return 0.0
    rows = np.arange(n_q)
    true_masked = true_ips.copy()
    true_masked[rows, q_idx] = -np.inf
    k = min(100, n_t - 1)
    cands = np.argpartition(-true_masked, kth=k - 1, axis=1)[:, :k]
    corrs = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        c = cands[i]
        t_scores = true_ips[i, c]
        a_scores = approx_ips[i, c]
        t_rank = np.argsort(np.argsort(-t_scores))
        a_rank = np.argsort(np.argsort(-a_scores))
        d = t_rank.astype(np.float64) - a_rank.astype(np.float64)
        corrs[i] = 1 - 6 * (d * d).sum() / (k * (k * k - 1))
    return float(corrs.mean())


def _nn_excl_self(
    true_ips: np.ndarray, approx_ips: np.ndarray, q_idx: np.ndarray, k: int
) -> float:
    if k <= 0 or true_ips.shape[1] <= 1:
        return 0.0
    rows = np.arange(true_ips.shape[0])
    true_masked = true_ips.copy()
    approx_masked = approx_ips.copy()
    true_masked[rows, q_idx] = -np.inf
    approx_masked[rows, q_idx] = -np.inf
    k_eff = min(k, true_ips.shape[1] - 1)
    true_top = np.argpartition(-true_masked, kth=k_eff - 1, axis=1)[:, :k_eff]
    approx_top = np.argpartition(-approx_masked, kth=k_eff - 1, axis=1)[:, :k_eff]
    overlaps = np.empty(true_ips.shape[0], dtype=np.float32)
    for i in range(true_ips.shape[0]):
        overlaps[i] = len(set(true_top[i].tolist()) & set(approx_top[i].tolist())) / k_eff
    return float(overlaps.mean())


# ---------------------------------------------------------------------------
# Encoder helpers (compose ternary / rroq158 / per-token rerank score)
# ---------------------------------------------------------------------------


def _ternary_decoded(tokens: np.ndarray, *, group_size: int, seed: int):
    """Returns (decoded_in_rotated_space, rotated_tokens, queries_rotator)."""
    from .ternary import TernaryConfig, TernaryQuantizer
    cfg = TernaryConfig(
        dim=tokens.shape[1], group_size=group_size, rotate=True,
        tau_frac=0.5, seed=seed,
    )
    q = TernaryQuantizer(cfg)
    enc = q.quantize(tokens)
    return q.decode(enc), enc["rotated"], q


def _roq4_decoded(tokens: np.ndarray, *, group_size: int, seed: int):
    from voyager_index._internal.inference.quantization.rotational import (
        RoQConfig, RotationalQuantizer,
    )
    cfg = RoQConfig(dim=tokens.shape[1], num_bits=4, num_rounds=3,
                    seed=seed, group_size=group_size)
    q = RotationalQuantizer(cfg)
    res = q.quantize(tokens, store=False)
    decoded = q.decode(res["codes"], res["scales"], res["offsets"]).cpu().numpy()
    return decoded[:, : tokens.shape[1]]


def _l2_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _tangent_corrected_scores(
    queries_n: np.ndarray, docs_n: np.ndarray, *, lam: float
) -> np.ndarray:
    """Per-pair tangent-aware score: s = <q,d> - λ·θ². Same formula as B0
    but applied at the pair-score level rather than the centroid-routing
    level so we can A/B it without the production router wired."""
    cos = np.clip(queries_n @ docs_n.T, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cos)
    return cos - lam * (theta ** 2)


# ---------------------------------------------------------------------------
# Distillation reranker training (lightweight, in-process)
# ---------------------------------------------------------------------------


def _build_rroq_features(
    queries_n: np.ndarray,                # (n_q, dim) L2-normalized
    centroids: np.ndarray,                # (K, dim) L2-normalized
    centroid_id: np.ndarray,              # (n_d,) int32
    residual_amb: np.ndarray,             # (n_d, dim) tangent residual in ambient space
) -> np.ndarray:
    """Per-pair features that the reranker actually has at request time.

    Production constraint: only the LOW-BIT view of the doc is available
    (centroid + ternary residual). No fp16 doc vector is allowed as a
    feature — that would leak the target.

    F=5:
      score_rroq:  rroq158 approx score  cos(||r||)·<q,c> + sinc(||r||)·<q,r̂>
      qc:          <q, c_d>
      qr:          <q, r̂_d>
      r_norm:      ||r̂_d||
      |qr|/r_norm: residual cosine magnitude (proxy for off-centroid
                   alignment; carries info that score alone does not)
    Shape (n_q, n_d, 5). ~40 MB at 256x8192.
    """
    n_q = queries_n.shape[0]
    n_d = centroid_id.shape[0]
    c_per_doc = centroids[centroid_id]                                 # (n_d, dim)
    qc = queries_n @ c_per_doc.T                                       # (n_q, n_d)
    qr = queries_n @ residual_amb.T                                    # (n_q, n_d)
    r_norm_vec = np.linalg.norm(residual_amb, axis=1) + 1e-12          # (n_d,)
    cos_t = np.cos(r_norm_vec)
    sin_t = np.sinc(r_norm_vec / np.pi)
    score_rroq = (cos_t[None, :] * qc) + (sin_t[None, :] * qr)
    r_norm_b = np.broadcast_to(r_norm_vec.astype(np.float32)[None, :], (n_q, n_d))
    res_cos_mag = np.abs(qr) / r_norm_b
    return np.stack([score_rroq, qc, qr, r_norm_b, res_cos_mag], axis=-1).astype(np.float32)


def _train_distill_pairwise(
    features: np.ndarray,        # (n_q, n_d, F)
    target_scores: np.ndarray,   # (n_q, n_d) fp16 scores
    q_idx: np.ndarray,           # query indices into doc set (for self-mask)
    *,
    seed: int,
    epochs: int = 6,
    hidden: int = 32,
    pairs_per_query: int = 256,
) -> "object":
    """Pairwise margin loss: for each query, sample pairs (i, j) and ask
    the model to keep their fp16 ordering. This is robust to score scale
    and trains fast.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_q, n_d, F = features.shape
    f_t = torch.from_numpy(features.reshape(-1, F))                       # (n_q*n_d, F)
    mean = f_t.mean(dim=0)
    std = f_t.std(dim=0).clamp_min(1e-6)
    f_n = (f_t - mean) / std

    model = nn.Sequential(
        nn.Linear(F, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # Pre-build self-mask
    self_mask = np.zeros((n_q, n_d), dtype=bool)
    self_mask[np.arange(n_q), q_idx] = True

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_pairs = 0
        for q in range(n_q):
            valid_idx = np.where(~self_mask[q])[0]
            i = rng.choice(valid_idx, size=pairs_per_query, replace=True)
            j = rng.choice(valid_idx, size=pairs_per_query, replace=True)
            t_i = target_scores[q, i]
            t_j = target_scores[q, j]
            keep = t_i != t_j
            i, j, t_i, t_j = i[keep], j[keep], t_i[keep], t_j[keep]
            if len(i) == 0:
                continue
            sign = torch.from_numpy(np.where(t_i > t_j, 1.0, -1.0).astype(np.float32))

            f_i = f_n[q * n_d + i]
            f_j = f_n[q * n_d + j]
            s_i = model(f_i).squeeze(-1)
            s_j = model(f_j).squeeze(-1)
            margin = sign * (s_i - s_j)
            loss = torch.clamp(0.1 - margin, min=0.0).mean()

            optim.zero_grad(); loss.backward(); optim.step()
            epoch_loss += float(loss.detach()) * len(i)
            n_pairs += len(i)
        log.info("    distill epoch %d  pairwise hinge loss=%.4f  (%d pairs)",
                 epoch, epoch_loss / max(n_pairs, 1), n_pairs)

    def score_fn(test_feats_np: np.ndarray) -> np.ndarray:
        ft = torch.from_numpy(test_feats_np.reshape(-1, test_feats_np.shape[-1]))
        ft = (ft - mean) / std
        with torch.no_grad():
            out = model(ft).squeeze(-1).cpu().numpy()
        return out.reshape(test_feats_np.shape[:-1])

    return score_fn


def _train_per_centroid_bias(
    train_feats: np.ndarray,
    train_target: np.ndarray,
    centroid_id: np.ndarray,
    K: int,
    q_idx_train: np.ndarray,
    *,
    seed: int,
    epochs: int = 8,
    pairs_per_query: int = 256,
):
    """Cheapest learnable head: a single scalar bias per centroid.

    final_score(q, d) = rroq158_score(q, d) + bias[centroid_id[d]]

    Trained with the same pairwise hinge as the MLP. Total deployable
    state: K floats (~4KB for K=1024). 0 inference cost beyond an LUT.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_q, n_d, _ = train_feats.shape
    base_score = torch.from_numpy(train_feats[..., 0])  # (n_q, n_d) — score_rroq is feature 0
    cid = torch.from_numpy(centroid_id.astype(np.int64))
    bias = torch.zeros(K, requires_grad=True)
    optim = torch.optim.AdamW([bias], lr=3e-2, weight_decay=1e-4)

    self_mask = np.zeros((n_q, n_d), dtype=bool)
    self_mask[np.arange(n_q), q_idx_train] = True

    for epoch in range(epochs):
        epoch_loss = 0.0; n_pairs = 0
        for q in range(n_q):
            valid = np.where(~self_mask[q])[0]
            i = rng.choice(valid, size=pairs_per_query, replace=True)
            j = rng.choice(valid, size=pairs_per_query, replace=True)
            t_i = train_target[q, i]; t_j = train_target[q, j]
            keep = t_i != t_j
            i, j = i[keep], j[keep]
            if len(i) == 0:
                continue
            sign = torch.from_numpy(np.where(t_i[keep] > t_j[keep], 1.0, -1.0).astype(np.float32))
            s_i = base_score[q, i] + bias[cid[i]]
            s_j = base_score[q, j] + bias[cid[j]]
            loss = torch.clamp(0.1 - sign * (s_i - s_j), min=0.0).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            epoch_loss += float(loss.detach()) * len(i); n_pairs += len(i)
        log.info("    bias epoch %d hinge=%.4f (%d pairs)", epoch, epoch_loss / max(n_pairs, 1), n_pairs)

    bias_np = bias.detach().cpu().numpy()

    def score_fn(test_feats_np: np.ndarray, cid_np: np.ndarray) -> np.ndarray:
        score_base = test_feats_np[..., 0]                    # (n_q, n_d)
        return score_base + bias_np[cid_np][None, :]

    return score_fn


def _train_residual_mlp(
    train_feats: np.ndarray,
    train_target: np.ndarray,
    q_idx_train: np.ndarray,
    *,
    seed: int,
    epochs: int = 6,
    hidden: int = 32,
    pairs_per_query: int = 256,
):
    """Residual head: final = base + alpha · tanh(MLP(features)).

    The base is feature[..., 0] (rroq158 score). The MLP is forced to
    learn small corrections, which is more stable than replacing the
    score outright. alpha is a learned scalar starting at 0.1.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_q, n_d, F = train_feats.shape
    f_t = torch.from_numpy(train_feats.reshape(-1, F))
    mean = f_t.mean(dim=0); std = f_t.std(dim=0).clamp_min(1e-6)
    f_n = (f_t - mean) / std
    base_score = torch.from_numpy(train_feats[..., 0])

    head = nn.Sequential(nn.Linear(F, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    alpha = nn.Parameter(torch.tensor(0.1))
    optim = torch.optim.AdamW(list(head.parameters()) + [alpha],
                              lr=3e-3, weight_decay=1e-4)

    self_mask = np.zeros((n_q, n_d), dtype=bool)
    self_mask[np.arange(n_q), q_idx_train] = True

    for epoch in range(epochs):
        epoch_loss = 0.0; n_pairs = 0
        for q in range(n_q):
            valid = np.where(~self_mask[q])[0]
            i = rng.choice(valid, size=pairs_per_query, replace=True)
            j = rng.choice(valid, size=pairs_per_query, replace=True)
            t_i = train_target[q, i]; t_j = train_target[q, j]
            keep = t_i != t_j
            i, j = i[keep], j[keep]
            if len(i) == 0:
                continue
            sign = torch.from_numpy(np.where(t_i[keep] > t_j[keep], 1.0, -1.0).astype(np.float32))
            d_i = alpha * torch.tanh(head(f_n[q * n_d + i]).squeeze(-1))
            d_j = alpha * torch.tanh(head(f_n[q * n_d + j]).squeeze(-1))
            s_i = base_score[q, i] + d_i
            s_j = base_score[q, j] + d_j
            loss = torch.clamp(0.1 - sign * (s_i - s_j), min=0.0).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            epoch_loss += float(loss.detach()) * len(i); n_pairs += len(i)
        log.info("    residMLP epoch %d hinge=%.4f alpha=%.3f (%d pairs)",
                 epoch, epoch_loss / max(n_pairs, 1), float(alpha), n_pairs)

    head.eval()

    def score_fn(test_feats_np: np.ndarray) -> np.ndarray:
        ft = torch.from_numpy(test_feats_np.reshape(-1, test_feats_np.shape[-1]))
        ft = (ft - mean) / std
        with torch.no_grad():
            delta = (alpha * torch.tanh(head(ft).squeeze(-1))).cpu().numpy()
        delta = delta.reshape(test_feats_np.shape[:-1])
        return test_feats_np[..., 0] + delta

    return score_fn


def _train_distill_with_embedding(
    train_feats: np.ndarray,
    train_target: np.ndarray,
    centroid_id: np.ndarray,
    K: int,
    embed_dim: int,
    q_idx_train: np.ndarray,
    *,
    seed: int,
    epochs: int = 6,
    hidden: int = 32,
    pairs_per_query: int = 256,
):
    """MLP head with per-centroid embedding concatenated to per-pair features."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_q, n_d, F = train_feats.shape
    f_t = torch.from_numpy(train_feats.reshape(-1, F))
    mean = f_t.mean(dim=0); std = f_t.std(dim=0).clamp_min(1e-6)
    f_n = (f_t - mean) / std

    cid = torch.from_numpy(centroid_id.astype(np.int64))
    embed = nn.Embedding(K, embed_dim)
    nn.init.normal_(embed.weight, std=0.02)
    head = nn.Sequential(
        nn.Linear(F + embed_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    optim = torch.optim.AdamW(list(embed.parameters()) + list(head.parameters()),
                              lr=3e-3, weight_decay=1e-4)

    self_mask = np.zeros((n_q, n_d), dtype=bool)
    self_mask[np.arange(n_q), q_idx_train] = True

    for epoch in range(epochs):
        epoch_loss = 0.0; n_pairs = 0
        for q in range(n_q):
            valid = np.where(~self_mask[q])[0]
            i = rng.choice(valid, size=pairs_per_query, replace=True)
            j = rng.choice(valid, size=pairs_per_query, replace=True)
            t_i = train_target[q, i]; t_j = train_target[q, j]
            keep = t_i != t_j
            i, j = i[keep], j[keep]
            if len(i) == 0:
                continue
            sign = torch.from_numpy(np.where(t_i[keep] > t_j[keep], 1.0, -1.0).astype(np.float32))
            f_i = torch.cat([f_n[q * n_d + i], embed(cid[i])], dim=-1)
            f_j = torch.cat([f_n[q * n_d + j], embed(cid[j])], dim=-1)
            s_i = head(f_i).squeeze(-1); s_j = head(f_j).squeeze(-1)
            loss = torch.clamp(0.1 - sign * (s_i - s_j), min=0.0).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            epoch_loss += float(loss.detach()) * len(i); n_pairs += len(i)
        log.info("    emb epoch %d hinge=%.4f (%d pairs)", epoch, epoch_loss / max(n_pairs, 1), n_pairs)

    embed.eval(); head.eval()

    def score_fn(test_feats_np: np.ndarray, cid_np: np.ndarray) -> np.ndarray:
        ft = torch.from_numpy(test_feats_np.reshape(-1, test_feats_np.shape[-1]))
        ft = (ft - mean) / std
        n_q_, n_d_, _ = test_feats_np.shape
        cid_t = torch.from_numpy(cid_np.astype(np.int64))
        with torch.no_grad():
            emb_doc = embed(cid_t)              # (n_d_, embed_dim)
            emb_pair = emb_doc[None, :, :].expand(n_q_, n_d_, -1).reshape(-1, embed_dim)
            full = torch.cat([ft, emb_pair], dim=-1)
            scores = head(full).squeeze(-1).cpu().numpy().reshape(n_q_, n_d_)
        return scores

    return score_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-path", type=Path,
        default=Path("research/low_bit_roq/tests/fixtures/token_sample_1m.npy"),
    )
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--n-queries-train", type=int, default=192)
    parser.add_argument("--n-queries-eval", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--K-list", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--lam-list", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.25])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("research/low_bit_roq/reports/recovery.json"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    arr = np.load(str(args.sample_path), mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    take = min(args.n_tokens, arr.shape[0])
    idx = rng.choice(arr.shape[0], size=take, replace=False)
    tokens = np.array(arr[idx], dtype=np.float32, order="C")
    del arr

    n_q_total = args.n_queries_train + args.n_queries_eval
    q_idx_all = rng.choice(tokens.shape[0], size=min(n_q_total, tokens.shape[0]), replace=False)
    q_idx_train = q_idx_all[: args.n_queries_train]
    q_idx_eval = q_idx_all[args.n_queries_train :]
    queries_train = tokens[q_idx_train].copy()
    queries_eval = tokens[q_idx_eval].copy()

    tokens_n = _l2_norm(tokens)
    qe_n = _l2_norm(queries_eval)
    fp16_true = qe_n @ tokens_n.T

    rows: list[RecoveryRow] = []

    # ---------- baselines ----------
    log.info("baseline: fp16 ceiling")
    rows.append(RecoveryRow(
        name="fp16", bits=16.0, centroids_K=0,
        rank_corr_top100=1.0,
        nn5_excl_self=1.0, nn50_excl_self=1.0,
        extra_bytes_per_token=0.0, notes="ceiling",
    ))

    log.info("baseline: roq4")
    roq4_dec = _roq4_decoded(tokens, group_size=args.group_size, seed=args.seed)
    roq4_n = _l2_norm(roq4_dec)
    roq4_approx = qe_n @ roq4_n.T
    rows.append(RecoveryRow(
        name="roq4", bits=4.0, centroids_K=0,
        rank_corr_top100=_rank_corr_top100(fp16_true, roq4_approx, q_idx_eval),
        nn5_excl_self=_nn_excl_self(fp16_true, roq4_approx, q_idx_eval, k=5),
        nn50_excl_self=_nn_excl_self(fp16_true, roq4_approx, q_idx_eval, k=50),
        extra_bytes_per_token=0.0, notes="current production-quality reference",
    ))
    del roq4_dec, roq4_n, roq4_approx; gc.collect()

    log.info("baseline: ternary (rotated-space cosine)")
    ter_dec, ter_rot, ter_q = _ternary_decoded(tokens, group_size=args.group_size, seed=args.seed)
    ter_n = _l2_norm(ter_dec)
    qe_rot = ter_q._rotate(queries_eval)
    qe_rot_n = _l2_norm(qe_rot)
    tokens_rot_n = _l2_norm(ter_rot)
    ternary_true = qe_rot_n @ tokens_rot_n.T
    ternary_approx = qe_rot_n @ ter_n.T
    base_rank_corr = _rank_corr_top100(ternary_true, ternary_approx, q_idx_eval)
    rows.append(RecoveryRow(
        name="ternary", bits=1.58, centroids_K=0,
        rank_corr_top100=base_rank_corr,
        nn5_excl_self=_nn_excl_self(ternary_true, ternary_approx, q_idx_eval, k=5),
        nn50_excl_self=_nn_excl_self(ternary_true, ternary_approx, q_idx_eval, k=50),
        extra_bytes_per_token=0.0, notes="A2.5 baseline candidate",
    ))

    # ---------- B0: tangent-aware rerank score ----------
    log.info("B0 tangent-corrected scores (lam sweep) on ternary base")
    for lam in args.lam_list:
        if lam == 0.0:
            continue
        scores = _tangent_corrected_scores(qe_rot_n, ter_n, lam=lam)
        rows.append(RecoveryRow(
            name=f"ternary+B0-tangent-lam{lam}", bits=1.58, centroids_K=0,
            rank_corr_top100=_rank_corr_top100(ternary_true, scores, q_idx_eval),
            nn5_excl_self=_nn_excl_self(ternary_true, scores, q_idx_eval, k=5),
            nn50_excl_self=_nn_excl_self(ternary_true, scores, q_idx_eval, k=50),
            extra_bytes_per_token=0.0,
            notes=f"per-pair s = <q,d> - {lam}·theta^2; cheap, no doc rebuild",
        ))

    del ter_dec, ter_n, qe_rot, qe_rot_n, tokens_rot_n, ter_rot
    gc.collect()

    # ---------- B3: rroq158 (Riemannian ternary with K centroids) ----------
    from .rroq import RroqConfig, RroqEncoder
    rroq_results: dict[int, np.ndarray] = {}
    for K in args.K_list:
        log.info("B3 rroq158 K=%d", K)
        cfg = RroqConfig(K=K, bits=2, group_size=args.group_size,  # rroq with bits=2
                         tangent_basis="identity", spherical_kmeans_iter=15,
                         seed=args.seed, eta=4.0)
        enc = RroqEncoder(cfg)
        enc.fit(tokens)
        encoded = enc.encode(tokens)
        approx = enc.reference_score(queries_eval, encoded)
        rroq_results[K] = approx
        bits_for_centroid = int(np.ceil(np.log2(K)))
        rows.append(RecoveryRow(
            name=f"rroq2-K{K}", bits=2.0, centroids_K=K,
            rank_corr_top100=_rank_corr_top100(fp16_true, approx, q_idx_eval),
            nn5_excl_self=_nn_excl_self(fp16_true, approx, q_idx_eval, k=5),
            nn50_excl_self=_nn_excl_self(fp16_true, approx, q_idx_eval, k=50),
            extra_bytes_per_token=bits_for_centroid / 8,
            notes=f"K={K} centroid + tangent residual; +{bits_for_centroid}b/tok",
        ))
        del enc, encoded; gc.collect()

    # rroq158 (true ternary residual)
    K = args.K_list[-1]
    log.info("B3 rroq158 (ternary residual) K=%d", K)
    cfg158 = RroqConfig(K=K, bits=1.58, group_size=args.group_size,
                        tangent_basis="identity", spherical_kmeans_iter=15,
                        seed=args.seed, eta=4.0)
    enc = RroqEncoder(cfg158)
    enc.fit(tokens)
    encoded = enc.encode(tokens)
    approx = enc.reference_score(queries_eval, encoded)
    rows.append(RecoveryRow(
        name=f"rroq158-K{K}", bits=1.58, centroids_K=K,
        rank_corr_top100=_rank_corr_top100(fp16_true, approx, q_idx_eval),
        nn5_excl_self=_nn_excl_self(fp16_true, approx, q_idx_eval, k=5),
        nn50_excl_self=_nn_excl_self(fp16_true, approx, q_idx_eval, k=50),
        extra_bytes_per_token=int(np.ceil(np.log2(K))) / 8,
        notes=f"ternary residual; K={K} centroid + tangent encoding",
    ))
    del enc, encoded; gc.collect()

    # ---------- X1: distillation reranker (pairwise margin) ----------
    # Use the rroq158-K1024 features that we just built — they have real
    # informational content per pair (qd, qc, qr, ||r||, |qd-qc|).
    log.info("X1 distill rerank: build features on train+eval queries (using rroq158-K1024)")
    cfg_distill = RroqConfig(K=1024, bits=1.58, group_size=args.group_size,
                             tangent_basis="identity", spherical_kmeans_iter=15,
                             seed=args.seed, eta=4.0)
    enc_distill = RroqEncoder(cfg_distill)
    enc_distill.fit(tokens)
    encoded_distill = enc_distill.encode(tokens)

    # Reconstruct r_amb for feature building (mirror reference_score logic).
    from .ternary import TernaryConfig, TernaryQuantizer as TQ
    tq2 = TQ(TernaryConfig(dim=tokens.shape[1], group_size=args.group_size,
                            rotate=False, fit_method="anisotropic"))
    r_amb_distill = tq2.decode(encoded_distill.code_payload)

    qt_n = _l2_norm(queries_train)
    qe_n = _l2_norm(queries_eval)
    train_target = qt_n @ tokens_n.T
    eval_target = qe_n @ tokens_n.T

    train_feats = _build_rroq_features(
        qt_n,
        encoded_distill.centroids, encoded_distill.centroid_id, r_amb_distill,
    )
    log.info("X1 distill: pairwise training (%d queries x %d pairs)",
             args.n_queries_train, 256)
    score_fn = _train_distill_pairwise(
        train_feats, train_target, q_idx_train, seed=args.seed,
        epochs=6, hidden=32, pairs_per_query=256,
    )
    del train_feats, train_target; gc.collect()

    eval_feats = _build_rroq_features(
        qe_n,
        encoded_distill.centroids, encoded_distill.centroid_id, r_amb_distill,
    )
    distill_scores = score_fn(eval_feats)
    rows.append(RecoveryRow(
        name="rroq158-K1024+X1-MLP", bits=1.58, centroids_K=1024,
        rank_corr_top100=_rank_corr_top100(eval_target, distill_scores, q_idx_eval),
        nn5_excl_self=_nn_excl_self(eval_target, distill_scores, q_idx_eval, k=5),
        nn50_excl_self=_nn_excl_self(eval_target, distill_scores, q_idx_eval, k=50),
        extra_bytes_per_token=int(np.ceil(np.log2(1024))) / 8,
        notes="3-layer MLP, ~1.2K params, pairwise hinge on 5 rroq158 features",
    ))

    # ----- X1b: per-centroid additive bias (cheapest deployable variant) -----
    log.info("X1b per-centroid bias training")
    score_fn_bias = _train_per_centroid_bias(
        train_feats=_build_rroq_features(
            qt_n, encoded_distill.centroids, encoded_distill.centroid_id, r_amb_distill,
        ),
        train_target=qt_n @ tokens_n.T,
        centroid_id=encoded_distill.centroid_id,
        K=1024,
        q_idx_train=q_idx_train,
        seed=args.seed,
    )
    bias_scores = score_fn_bias(eval_feats, encoded_distill.centroid_id)
    rows.append(RecoveryRow(
        name="rroq158-K1024+X1-bias", bits=1.58, centroids_K=1024,
        rank_corr_top100=_rank_corr_top100(eval_target, bias_scores, q_idx_eval),
        nn5_excl_self=_nn_excl_self(eval_target, bias_scores, q_idx_eval, k=5),
        nn50_excl_self=_nn_excl_self(eval_target, bias_scores, q_idx_eval, k=50),
        extra_bytes_per_token=int(np.ceil(np.log2(1024))) / 8,
        notes="K scalar biases (~1KB total), score' = score + bias[c_d]",
    ))

    # ----- X1e: MULTI-VIEW distill — combine rroq158 + raw ternary scores -----
    # The two scores use independent rotations / codes, so their noise is
    # decorrelated. A learned combiner can leverage both views.
    log.info("X1e multi-view (rroq158 + ternary) distill training")
    ter_dec_mv, ter_rot_mv, ter_q_mv = _ternary_decoded(
        tokens, group_size=args.group_size, seed=args.seed
    )
    ter_dec_mv_n = _l2_norm(ter_dec_mv)
    qt_rot_mv_n = _l2_norm(ter_q_mv._rotate(queries_train))
    qe_rot_mv_n = _l2_norm(ter_q_mv._rotate(queries_eval))
    train_ternary_score = qt_rot_mv_n @ ter_dec_mv_n.T
    eval_ternary_score = qe_rot_mv_n @ ter_dec_mv_n.T

    train_feats_mv = _build_rroq_features(
        qt_n, encoded_distill.centroids, encoded_distill.centroid_id, r_amb_distill,
    )
    train_feats_mv = np.concatenate(
        [train_feats_mv, train_ternary_score[..., None].astype(np.float32)], axis=-1
    )
    eval_feats_mv = np.concatenate(
        [eval_feats, eval_ternary_score[..., None].astype(np.float32)], axis=-1
    )
    score_fn_mv = _train_distill_pairwise(
        train_feats_mv, qt_n @ tokens_n.T, q_idx_train,
        seed=args.seed, epochs=6, hidden=32, pairs_per_query=256,
    )
    mv_scores = score_fn_mv(eval_feats_mv)
    rows.append(RecoveryRow(
        name="rroq158-K1024+ternary+X1-MV", bits=1.58, centroids_K=1024,
        rank_corr_top100=_rank_corr_top100(eval_target, mv_scores, q_idx_eval),
        nn5_excl_self=_nn_excl_self(eval_target, mv_scores, q_idx_eval, k=5),
        nn50_excl_self=_nn_excl_self(eval_target, mv_scores, q_idx_eval, k=50),
        extra_bytes_per_token=int(np.ceil(np.log2(1024))) / 8,
        notes="MLP on rroq158 features + ternary score (independent noise sources)",
    ))
    del ter_dec_mv, ter_rot_mv, qt_rot_mv_n, qe_rot_mv_n
    del train_ternary_score, eval_ternary_score, train_feats_mv, eval_feats_mv, mv_scores
    gc.collect()

    # ----- X1d: residual MLP — final = base_score + alpha * delta_mlp -----
    log.info("X1d residual MLP training")
    score_fn_res = _train_residual_mlp(
        train_feats=_build_rroq_features(
            qt_n, encoded_distill.centroids, encoded_distill.centroid_id, r_amb_distill,
        ),
        train_target=qt_n @ tokens_n.T,
        q_idx_train=q_idx_train,
        seed=args.seed,
    )
    res_scores = score_fn_res(eval_feats)
    rows.append(RecoveryRow(
        name="rroq158-K1024+X1-residMLP", bits=1.58, centroids_K=1024,
        rank_corr_top100=_rank_corr_top100(eval_target, res_scores, q_idx_eval),
        nn5_excl_self=_nn_excl_self(eval_target, res_scores, q_idx_eval, k=5),
        nn50_excl_self=_nn_excl_self(eval_target, res_scores, q_idx_eval, k=50),
        extra_bytes_per_token=int(np.ceil(np.log2(1024))) / 8,
        notes="final = base + 0.3·tanh(MLP(features)) — residual head, stable",
    ))

    # ----- X1c: MLP + centroid embedding (per-cluster correction) -----
    log.info("X1c MLP+centroid embedding training")
    score_fn_emb = _train_distill_with_embedding(
        train_feats=_build_rroq_features(
            qt_n, encoded_distill.centroids, encoded_distill.centroid_id, r_amb_distill,
        ),
        train_target=qt_n @ tokens_n.T,
        centroid_id=encoded_distill.centroid_id,
        K=1024, embed_dim=8,
        q_idx_train=q_idx_train,
        seed=args.seed,
    )
    emb_scores = score_fn_emb(eval_feats, encoded_distill.centroid_id)
    rows.append(RecoveryRow(
        name="rroq158-K1024+X1-MLP+emb", bits=1.58, centroids_K=1024,
        rank_corr_top100=_rank_corr_top100(eval_target, emb_scores, q_idx_eval),
        nn5_excl_self=_nn_excl_self(eval_target, emb_scores, q_idx_eval, k=5),
        nn50_excl_self=_nn_excl_self(eval_target, emb_scores, q_idx_eval, k=50),
        extra_bytes_per_token=int(np.ceil(np.log2(1024))) / 8,
        notes=f"MLP + 8-d centroid embedding ({1024*8 + 1200} params)",
    ))
    del enc_distill, encoded_distill, r_amb_distill, eval_feats, distill_scores, bias_scores, emb_scores
    gc.collect()

    # ---------- summary + report ----------
    out = {"config": vars(args) | {"out": str(args.out), "sample_path": str(args.sample_path)},
           "rows": [asdict(r) for r in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    log.info("=== recovery comparison ===")
    log.info("%-40s %5s %5s %12s %8s %8s %12s",
             "name", "bits", "K", "rank@100", "NN5*", "NN50*", "extra B/tok")
    base = next(r for r in rows if r.name == "ternary").rank_corr_top100
    ceil = next(r for r in rows if r.name == "roq4").rank_corr_top100
    gap = ceil - base
    for r in rows:
        recover_pct = 100 * (r.rank_corr_top100 - base) / gap if gap > 0 else 0.0
        log.info("%-40s %5.2f %5d %12.3f %8.3f %8.3f %12.2f  (recovers %5.1f%% of gap)",
                 r.name, r.bits, r.centroids_K, r.rank_corr_top100,
                 r.nn5_excl_self, r.nn50_excl_self, r.extra_bytes_per_token, recover_pct)
    log.info("base ternary rank_corr=%.3f  ceiling roq4=%.3f  gap=%.3f",
             base, ceil, gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
