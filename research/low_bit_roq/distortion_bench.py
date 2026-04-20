"""
Offline distortion benchmark.

Plan reference: Phase 0 deliverable + Phase A1 mechanics + Phase B3 hard-stop
gate. Runs entirely on a saved token sample (no BEIR download, no GPU
required). Produces:

- IP / cosine error histogram per quantizer
- angular error percentiles (p50, p90, p99)
- top-K NN preservation (K ∈ {1, 5, 100})

Usable for any quantizer that exposes ``quantize`` + ``decode`` (existing
RotationalQuantizer, the new TernaryQuantizer, and rroq2).

Run:

    python -m research.low_bit_roq.distortion_bench \
      --sample-path tests/fixtures/token_sample_1m.npy \
      --bits 1 1.58 2 4 \
      --group-size 16 \
      --out research/low_bit_roq/reports/distortion_phase0.json

The 1-million-token sample comes from a held-out shard. If you don't have
one, ``--synthetic`` generates a normalized iid sample for smoke testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DistortionRow:
    quantizer: str
    bits: float
    group_size: int | None
    fwht: bool
    n_tokens: int
    angular_error_p50_deg: float
    angular_error_p90_deg: float
    angular_error_p99_deg: float
    ip_error_rms: float
    cosine_error_rms: float
    nn1_preservation: float
    nn5_preservation: float
    nn100_preservation: float
    bytes_per_token: float
    notes: str = ""


def _angular_error_deg(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Per-row angular error in degrees between x and x_hat."""
    x_n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    x_hat_n = x_hat / (np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-12)
    cos = np.clip((x_n * x_hat_n).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _ip_error_from_matrices(true_ips: np.ndarray, approx_ips: np.ndarray) -> float:
    """RMS of the asymmetric IP error <q,x> - <q,x_hat> across (q,x) pairs.

    ScaNN-style metric: this is what asymmetric distance computation in the
    rerank kernel actually approximates. RMS not MAE because the optimization
    target in A3 is L2 on the IP gap.
    """
    diff = true_ips - approx_ips
    return float(np.sqrt((diff * diff).mean()))


def _cosine_error(x: np.ndarray, x_hat: np.ndarray) -> float:
    angles_rad = np.radians(_angular_error_deg(x, x_hat))
    return float(np.sqrt((1 - np.cos(angles_rad)).mean()))


def _nn_preservation_from_matrices(
    true_ips: np.ndarray, approx_ips: np.ndarray, k: int
) -> float:
    """Fraction of queries for which the top-K neighbours by true IP have
    overlap with the top-K neighbours by approx IP.

    Both ``true_ips`` and ``approx_ips`` are precomputed (n_queries, n_tokens)
    matrices — passing them in (rather than recomputing) keeps the per-cell
    peak memory at one matrix instead of two, which matters for the A1 sweep
    on the 24 GB benchmark box.
    """
    if k <= 0:
        return 0.0
    k_eff = min(k, true_ips.shape[1])
    true_top = np.argpartition(-true_ips, kth=k_eff - 1, axis=1)[:, :k_eff]
    approx_top = np.argpartition(-approx_ips, kth=k_eff - 1, axis=1)[:, :k_eff]
    n_q = true_ips.shape[0]
    overlaps = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        overlaps[i] = len(set(true_top[i].tolist()) & set(approx_top[i].tolist())) / k_eff
    return float(overlaps.mean())


# ---------------------------------------------------------------------------
# Quantizer adapters
# ---------------------------------------------------------------------------


def _bytes_per_token_for(bits: float, dim: int, group_size: int | None) -> float:
    """Conservative byte budget per token: payload + per-group meta."""
    if bits == 1.0:
        return dim / 8
    if bits == 1.58:
        return 2 * dim / 8
    body = bits * dim / 8
    if group_size:
        n_groups = max(1, dim // group_size)
        body += n_groups * 8
    else:
        body += 8
    return body


def _existing_roq_quantize(
    x: np.ndarray,
    bits: int,
    group_size: int | None,
    fwht: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap the existing RotationalQuantizer.

    ``RotationalQuantizer.decode`` always returns vectors in ambient
    (un-rotated) space, regardless of whether FWHT was applied at encode
    time. So we always compare ``decoded`` against the original ``x``.
    """
    from voyager_index._internal.inference.quantization.rotational import (
        RoQConfig,
        RotationalQuantizer,
    )

    cfg_kwargs = dict(
        dim=x.shape[1],
        num_bits=int(bits),
        num_rounds=3 if fwht else 0,
        seed=seed,
    )
    if int(bits) in (2, 4) and group_size is not None:
        cfg_kwargs["group_size"] = group_size
    cfg = RoQConfig(**cfg_kwargs)
    q = RotationalQuantizer(cfg)
    res = q.quantize(x, store=False)
    decoded = q.decode(res["codes"], res["scales"], res["offsets"]).cpu().numpy()
    decoded = decoded[:, : x.shape[1]]
    return decoded, x


def _ternary_quantize(
    x: np.ndarray,
    group_size: int | None,
    fwht: bool,
    tau_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from .ternary import TernaryConfig, TernaryQuantizer

    cfg = TernaryConfig(
        dim=x.shape[1],
        group_size=group_size,
        rotate=fwht,
        tau_frac=tau_frac,
        seed=seed,
    )
    q = TernaryQuantizer(cfg)
    res = q.quantize(x)
    decoded = q.decode(res)
    return decoded, res["rotated"]


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def sweep(
    tokens: np.ndarray,
    queries: np.ndarray,
    *,
    bits: Sequence[float],
    group_size: int | None,
    fwht: bool = True,
    seed: int = 0,
) -> list[DistortionRow]:
    """Run a distortion measurement for each entry in ``bits``.

    Memory plan (each cell, then released):

    1. quantize+decode ``tokens``                  ~ 2 * n_tokens * dim * 4 B
    2. per-row angular error                       ~ n_tokens * 8 B
    3. ``true_ips`` matrix (queries @ ref.T)       ~ n_queries * n_tokens * 4 B
    4. ``approx_ips`` matrix (queries @ decoded.T) ~ n_queries * n_tokens * 4 B
    5. (3) and (4) reused across IP error / NN1 / NN5 / NN100 — formerly
       reallocated per metric, which was the leading OOM contributor.
    6. ``decoded``, ``ref``, both ``ips`` matrices, and torch tensor cache
       are explicitly freed before the next bit-width.

    For the 24 GB benchmark box the recommended cell size is
    ``n_tokens <= 20_000`` and ``n_queries <= 256``: ~80 MB peak per cell.
    """
    import gc as _gc

    rows: list[DistortionRow] = []
    dim = tokens.shape[1]
    for b in bits:
        if b == 1.58:
            decoded, ref = _ternary_quantize(
                tokens, group_size=group_size, fwht=fwht, tau_frac=0.5, seed=seed
            )
            quantizer_name = "ternary"
        else:
            decoded, ref = _existing_roq_quantize(
                tokens, int(b), group_size, fwht=fwht, seed=seed
            )
            quantizer_name = f"roq{int(b)}"
        if decoded.shape != ref.shape:
            min_d = min(decoded.shape[1], ref.shape[1])
            decoded = decoded[:, :min_d]
            ref = ref[:, :min_d]
        ang = _angular_error_deg(ref, decoded)
        true_ips = queries @ ref.T
        approx_ips = queries @ decoded.T
        rows.append(
            DistortionRow(
                quantizer=quantizer_name,
                bits=float(b),
                group_size=group_size,
                fwht=fwht,
                n_tokens=int(tokens.shape[0]),
                angular_error_p50_deg=float(np.percentile(ang, 50)),
                angular_error_p90_deg=float(np.percentile(ang, 90)),
                angular_error_p99_deg=float(np.percentile(ang, 99)),
                ip_error_rms=_ip_error_from_matrices(true_ips, approx_ips),
                cosine_error_rms=_cosine_error(ref, decoded),
                nn1_preservation=_nn_preservation_from_matrices(true_ips, approx_ips, k=1),
                nn5_preservation=_nn_preservation_from_matrices(true_ips, approx_ips, k=5),
                nn100_preservation=_nn_preservation_from_matrices(
                    true_ips, approx_ips, k=min(100, ref.shape[0])
                ),
                bytes_per_token=_bytes_per_token_for(b, dim, group_size),
            )
        )
        del decoded, ref, ang, true_ips, approx_ips
        _gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_sample(sample_path: Path | None, synthetic: bool, n_tokens: int, dim: int) -> np.ndarray:
    if synthetic or sample_path is None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n_tokens, dim)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x
    return np.load(sample_path).astype(np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-path", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=16384)
    parser.add_argument("--n-queries", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--bits", type=float, nargs="+", default=[1.0, 1.58, 2.0, 4.0])
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--no-fwht", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("research/low_bit_roq/reports/distortion.json"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tokens = _load_sample(args.sample_path, args.synthetic, args.n_tokens, args.dim)
    rng = np.random.default_rng(args.seed + 1)
    q_idx = rng.choice(tokens.shape[0], size=min(args.n_queries, tokens.shape[0]), replace=False)
    queries = tokens[q_idx]

    rows = sweep(
        tokens,
        queries,
        bits=args.bits,
        group_size=args.group_size,
        fwht=not args.no_fwht,
        seed=args.seed,
    )

    out = {
        "config": {
            "n_tokens": int(tokens.shape[0]),
            "n_queries": int(queries.shape[0]),
            "dim": int(tokens.shape[1]),
            "bits": list(args.bits),
            "group_size": args.group_size,
            "fwht": not args.no_fwht,
        },
        "rows": [asdict(r) for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
