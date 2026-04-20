"""
Phase A3 — anisotropic codebook A/B.

For each codebook fitting method (uniform vs anisotropic with η ∈ {2, 4, 8}),
encode a sample of real ColBERT tokens, decode, and measure the angular
error and IP error. Anisotropic fitting should reduce the *parallel*
component of the residual most — so the IP error (which is what the
asymmetric kernel actually pays for at rerank time) should drop more
than the global angular error.

Memory plan: ~30 MB peak. Runs in <2 s on CPU. No GPU, no Triton.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import anisotropic
from .ternary import TernaryConfig, TernaryQuantizer

log = logging.getLogger(__name__)


@dataclass
class AnisotropicRow:
    quantizer: str
    method: str
    eta: float
    n_tokens: int
    dim: int
    group_size: int
    angular_p50_deg: float
    angular_p90_deg: float
    ip_error_rms: float
    ip_error_parallel_rms: float
    ip_error_perp_rms: float


def _angular_deg(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    xhn = x_hat / (np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip((xn * xhn).sum(axis=1), -1.0, 1.0)))


def _ip_error_components(
    tokens: np.ndarray, decoded: np.ndarray, queries: np.ndarray
) -> tuple[float, float, float]:
    """Decompose the IP error into parallel-to-token and perpendicular-to-token
    components, summed over (q, t) pairs.

    The parallel component is exactly what the anisotropic loss penalises
    (η > 1) and is the leading cause of rerank ranking errors; the
    perpendicular component is incoherent across tokens and averages out
    across the MaxSim sum over query tokens.
    """
    residual = tokens - decoded
    norms = np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-12
    u = tokens / norms
    parallel_per_token = (residual * u).sum(axis=1, keepdims=True) * u
    perp_per_token = residual - parallel_per_token

    par_ips = queries @ parallel_per_token.T
    perp_ips = queries @ perp_per_token.T
    total_ips = par_ips + perp_ips
    return (
        float(np.sqrt((total_ips ** 2).mean())),
        float(np.sqrt((par_ips ** 2).mean())),
        float(np.sqrt((perp_ips ** 2).mean())),
    )


def _ternary_decoded(
    tokens: np.ndarray, *, group_size: int, eta: float, fit_method: str, seed: int
) -> np.ndarray:
    cfg = TernaryConfig(
        dim=tokens.shape[1],
        group_size=group_size,
        rotate=False,
        tau_frac=0.5,
        seed=seed,
        fit_method=fit_method,
    )
    if fit_method == "anisotropic":
        anisotropic.ETA_DEFAULT = eta
    q = TernaryQuantizer(cfg)
    enc = q.quantize(tokens)
    return q.decode(enc)


def _two_bit_decoded(
    tokens: np.ndarray, *, group_size: int, eta: float, seed: int
) -> np.ndarray:
    """Anisotropic 2-bit decode using ``fit_anisotropic_min_max`` directly.

    This is the right baseline for A3 — comparing it against
    ``RotationalQuantizer(num_bits=2)`` (which uses uniform per-group
    min/max) isolates the η>1 contribution from the rotation/grouping.
    """
    n, dim = tokens.shape
    n_groups = dim // group_size
    grouped = tokens.reshape(n, n_groups, group_size)
    bits = 2
    levels = (1 << bits) - 1
    scales, offsets = anisotropic.fit_anisotropic_min_max(
        grouped, bits=bits, eta=eta, n_newton=8
    )
    codes = np.round((grouped - offsets[..., None]) / scales[..., None]).clip(0, levels)
    decoded = (codes * scales[..., None] + offsets[..., None]).reshape(n, dim).astype(np.float32)
    return decoded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-path", type=Path,
        default=Path("research/low_bit_roq/tests/fixtures/token_sample_1m.npy"),
    )
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--n-queries", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--etas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument(
        "--out", type=Path,
        default=Path("research/low_bit_roq/reports/a3_anisotropic.json"),
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

    rows: list[AnisotropicRow] = []

    log.info("ternary uniform vs anisotropic ...")
    decoded = _ternary_decoded(
        tokens, group_size=args.group_size, eta=1.0, fit_method="tau_frac", seed=args.seed
    )
    ang = _angular_deg(tokens, decoded)
    total, par, perp = _ip_error_components(tokens, decoded, queries)
    rows.append(AnisotropicRow(
        quantizer="ternary", method="tau_frac", eta=1.0,
        n_tokens=tokens.shape[0], dim=tokens.shape[1], group_size=args.group_size,
        angular_p50_deg=float(np.percentile(ang, 50)),
        angular_p90_deg=float(np.percentile(ang, 90)),
        ip_error_rms=total, ip_error_parallel_rms=par, ip_error_perp_rms=perp,
    ))
    for eta in args.etas:
        if eta == 1.0:
            continue
        decoded = _ternary_decoded(
            tokens, group_size=args.group_size, eta=eta, fit_method="anisotropic", seed=args.seed
        )
        ang = _angular_deg(tokens, decoded)
        total, par, perp = _ip_error_components(tokens, decoded, queries)
        rows.append(AnisotropicRow(
            quantizer="ternary", method="anisotropic", eta=eta,
            n_tokens=tokens.shape[0], dim=tokens.shape[1], group_size=args.group_size,
            angular_p50_deg=float(np.percentile(ang, 50)),
            angular_p90_deg=float(np.percentile(ang, 90)),
            ip_error_rms=total, ip_error_parallel_rms=par, ip_error_perp_rms=perp,
        ))

    log.info("2-bit uniform vs anisotropic ...")
    for eta in args.etas:
        decoded = _two_bit_decoded(
            tokens, group_size=args.group_size, eta=eta, seed=args.seed
        )
        ang = _angular_deg(tokens, decoded)
        total, par, perp = _ip_error_components(tokens, decoded, queries)
        rows.append(AnisotropicRow(
            quantizer="roq2", method="anisotropic" if eta > 1.0 else "uniform", eta=eta,
            n_tokens=tokens.shape[0], dim=tokens.shape[1], group_size=args.group_size,
            angular_p50_deg=float(np.percentile(ang, 50)),
            angular_p90_deg=float(np.percentile(ang, 90)),
            ip_error_rms=total, ip_error_parallel_rms=par, ip_error_perp_rms=perp,
        ))

    out = {"config": vars(args) | {"out": str(args.out), "sample_path": str(args.sample_path)},
           "rows": [asdict(r) for r in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for r in rows:
        log.info(
            "%s/%s eta=%.1f angular_p50=%.2f IP_RMS=%.4f par=%.4f perp=%.4f",
            r.quantizer, r.method, r.eta, r.angular_p50_deg,
            r.ip_error_rms, r.ip_error_parallel_rms, r.ip_error_perp_rms,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
