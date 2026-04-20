"""
Side-by-side bit-width comparison for the "how much worse is ternary?" Q.

Runs each of {1-bit, 1.58-bit, 2-bit, 4-bit} through the existing
distortion bench on the same fixture and reports both the offline
distortion AND a self-match-corrected NN-preservation proxy. The
proxy excludes the diagonal in the (queries, tokens) similarity matrix
before argpartition, so NN1 is no longer dominated by "every query is
its own top-1 neighbour" and becomes a meaningful Recall@K predictor.

Memory plan: ~50 MB peak. Runs in <5 s on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import distortion_bench

log = logging.getLogger(__name__)


@dataclass
class CompareRow:
    quantizer: str
    bits: float
    angular_p50_deg: float
    angular_p90_deg: float
    cos_preservation: float
    rank_corr_top100: float
    nn1_excl_self: float
    nn5_excl_self: float
    nn50_excl_self: float
    bytes_per_token: float


def _scale_invariant_ips(decoded: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """L2-normalize the decoded vectors before computing inner products.

    This sidesteps the existing RotationalQuantizer scale-blowup (its
    ``decode`` returns vectors ~14x the original norm because the codes
    {0,1,2,3} are scaled by per-token range ÷ levels — fine for the
    affine-corrected kernel scorer, useless for a direct ambient IP).
    Cosine-similarity-based ranking is the right proxy because both
    MaxSim and the kernel's asymmetric scorer are rank-equivariant
    under per-doc-token scaling.
    """
    dn = decoded / (np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-12)
    return queries @ dn.T


def _nn_excl_self(true_ips: np.ndarray, approx_ips: np.ndarray, q_idx: np.ndarray, k: int) -> float:
    if k <= 0 or true_ips.shape[1] <= 1:
        return 0.0
    n_q = true_ips.shape[0]
    rows = np.arange(n_q)
    true_masked = true_ips.copy()
    approx_masked = approx_ips.copy()
    true_masked[rows, q_idx] = -np.inf
    approx_masked[rows, q_idx] = -np.inf
    k_eff = min(k, true_ips.shape[1] - 1)
    true_top = np.argpartition(-true_masked, kth=k_eff - 1, axis=1)[:, :k_eff]
    approx_top = np.argpartition(-approx_masked, kth=k_eff - 1, axis=1)[:, :k_eff]
    overlaps = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        overlaps[i] = len(set(true_top[i].tolist()) & set(approx_top[i].tolist())) / k_eff
    return float(overlaps.mean())


def _rank_correlation_top100(true_ips: np.ndarray, approx_ips: np.ndarray, q_idx: np.ndarray) -> float:
    """Average Spearman correlation over each query's true top-100 candidates.

    This is the cleanest "does the bit-width preserve the rerank ranking?"
    metric — much closer to BEIR Recall@10 than raw NN preservation
    because BEIR Recall@10 depends on the *ordering* among the top
    candidates, not on whether the absolute top-1 changed.
    """
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-path", type=Path,
        default=Path("research/low_bit_roq/tests/fixtures/token_sample_1m.npy"),
    )
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--n-queries", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("research/low_bit_roq/reports/bitwidth_compare.json"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    arr = np.load(str(args.sample_path), mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    take = min(args.n_tokens, arr.shape[0])
    idx = rng.choice(arr.shape[0], size=take, replace=False)
    tokens = np.array(arr[idx], dtype=np.float32, order="C")
    del arr
    q_idx = rng.choice(tokens.shape[0], size=min(args.n_queries, tokens.shape[0]), replace=False)
    queries = tokens[q_idx].copy()

    bit_specs = [
        ("roq1", 1.0),
        ("ternary", 1.58),
        ("roq2", 2.0),
        ("roq4", 4.0),
    ]

    rows: list[CompareRow] = []
    tokens_n = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-12)
    queries_n = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    true_cos = queries_n @ tokens_n.T

    for name, bits in bit_specs:
        if bits == 1.58:
            decoded, ref = distortion_bench._ternary_quantize(
                tokens, group_size=args.group_size, fwht=True, tau_frac=0.5, seed=args.seed,
            )
            from .ternary import TernaryConfig, TernaryQuantizer
            tq = TernaryQuantizer(TernaryConfig(
                dim=tokens.shape[1], group_size=args.group_size,
                rotate=True, tau_frac=0.5, seed=args.seed,
            ))
            queries_rot = tq._rotate(queries)
            tokens_rot = tq._rotate(tokens)
            qn = queries_rot / (np.linalg.norm(queries_rot, axis=1, keepdims=True) + 1e-12)
            true_for_this = qn @ (tokens_rot / (np.linalg.norm(tokens_rot, axis=1, keepdims=True) + 1e-12)).T
            approx_cos = _scale_invariant_ips(decoded, qn)
        else:
            decoded, ref = distortion_bench._existing_roq_quantize(
                tokens, int(bits), args.group_size, fwht=True, seed=args.seed,
            )
            true_for_this = true_cos
            approx_cos = _scale_invariant_ips(decoded, queries_n)
        if decoded.shape != ref.shape:
            d = min(decoded.shape[1], ref.shape[1])
            decoded, ref = decoded[:, :d], ref[:, :d]

        ang = distortion_bench._angular_error_deg(ref, decoded)
        cos_pres = float((1 - np.sqrt(((true_for_this - approx_cos) ** 2).mean()) /
                         (np.sqrt((true_for_this ** 2).mean()) + 1e-12)))

        rows.append(CompareRow(
            quantizer=name, bits=bits,
            angular_p50_deg=float(np.percentile(ang, 50)),
            angular_p90_deg=float(np.percentile(ang, 90)),
            cos_preservation=cos_pres,
            rank_corr_top100=_rank_correlation_top100(true_for_this, approx_cos, q_idx),
            nn1_excl_self=_nn_excl_self(true_for_this, approx_cos, q_idx, k=1),
            nn5_excl_self=_nn_excl_self(true_for_this, approx_cos, q_idx, k=5),
            nn50_excl_self=_nn_excl_self(true_for_this, approx_cos, q_idx, k=50),
            bytes_per_token=distortion_bench._bytes_per_token_for(
                bits, tokens.shape[1], args.group_size
            ),
        ))
        del decoded, ref, ang, approx_cos
        gc.collect()

    out = {"config": vars(args) | {"out": str(args.out), "sample_path": str(args.sample_path)},
           "rows": [asdict(r) for r in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("=== bit-width comparison (group_size=%d, FWHT on, %d tokens, %d queries) ===",
             args.group_size, tokens.shape[0], queries.shape[0])
    log.info("%-9s %5s %10s %10s %8s %10s %8s %8s %9s %9s",
             "quant", "bits", "angle_p50", "angle_p90", "cos_pres",
             "rank_top100", "NN1*", "NN5*", "NN50*", "B/token")
    for r in rows:
        log.info("%-9s %5.2f %10.2f %10.2f %8.3f %10.3f %8.3f %8.3f %9.3f %9.1f",
                 r.quantizer, r.bits, r.angular_p50_deg, r.angular_p90_deg,
                 r.cos_preservation, r.rank_corr_top100,
                 r.nn1_excl_self, r.nn5_excl_self, r.nn50_excl_self,
                 r.bytes_per_token)
    log.info("(* NN excludes self-matches; metrics use L2-normed decoded tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
