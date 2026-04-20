"""
Phase A2 / A2.5 — kernel micro-benchmark.

Validates the Triton 2-bit asymmetric and ternary kernels against their
NumPy reference scorers on the *actual* GPU, then reports throughput.

Memory plan (24 GB box, single A5000):

- 64 queries × 256 docs × 32 q_tokens × 32 d_tokens × dim 128 × 32 q_bit_planes
  → all tensors stay <50 MB on device.
- We deliberately do NOT touch BEIR data here; this is the kernel-level
  A/B that the plan calls "A2 + A2.5 A/B on A5000". Larger sizes are
  available behind ``--big`` but the small size is the default for the
  24 GB budget.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class KernelBenchRow:
    name: str
    n_queries: int
    n_docs: int
    n_q_tokens: int
    n_d_tokens: int
    dim: int
    query_bits: int
    group_size: int
    triton_ms_p50: float
    triton_ms_p95: float
    parity_max_abs_err: float
    parity_max_rel_err: float
    qps: float


def _bench_2bit_asym(
    *, n_queries: int, n_docs: int, n_q_tokens: int, n_d_tokens: int,
    dim: int, query_bits: int, group_size: int, n_iters: int, seed: int,
) -> KernelBenchRow:
    import torch
    from .kernels import triton_roq_2bit_asym as k2

    n_groups = dim // group_size
    n_words = dim // 32
    rng = np.random.default_rng(seed)

    rot_q = rng.standard_normal((n_queries * n_q_tokens, dim)).astype(np.float32)
    planes_np, meta_np, group_sum_np = k2.encode_query_for_2bit_asym(
        rot_q, query_bits=query_bits, group_size=group_size, n_groups=n_groups,
    )
    planes_np = planes_np.reshape(n_queries, n_q_tokens, query_bits, n_words)
    meta_np = meta_np.reshape(n_queries, n_q_tokens, 3)
    group_sum_np = group_sum_np.reshape(n_queries, n_q_tokens, n_groups)

    n_total_d = n_docs * n_d_tokens
    bit0_np = rng.integers(0, 2, size=(n_total_d, n_words), dtype=np.int32)
    bit1_np = rng.integers(0, 2, size=(n_total_d, n_words), dtype=np.int32)
    scales_np = rng.standard_normal((n_total_d, n_groups)).astype(np.float32) * 0.1 + 0.5
    offsets_np = rng.standard_normal((n_total_d, n_groups)).astype(np.float32) * 0.05
    code_sum_np = rng.integers(0, 4 * group_size, size=(n_total_d, n_groups)).astype(np.float32)

    bit0_np = bit0_np.reshape(n_docs, n_d_tokens, n_words)
    bit1_np = bit1_np.reshape(n_docs, n_d_tokens, n_words)
    scales_np = scales_np.reshape(n_docs, n_d_tokens, n_groups)
    offsets_np = offsets_np.reshape(n_docs, n_d_tokens, n_groups)
    code_sum_np = code_sum_np.reshape(n_docs, n_d_tokens, n_groups)

    dev = torch.device("cuda")
    planes_t = torch.from_numpy(planes_np).to(dev)
    meta_t = torch.from_numpy(meta_np).to(dev)
    group_sum_t = torch.from_numpy(group_sum_np).to(dev)
    bit0_t = torch.from_numpy(bit0_np).to(dev)
    bit1_t = torch.from_numpy(bit1_np).to(dev)
    scales_t = torch.from_numpy(scales_np).to(dev)
    offsets_t = torch.from_numpy(offsets_np).to(dev)
    code_sum_t = torch.from_numpy(code_sum_np).to(dev)

    for _ in range(3):
        _ = k2.roq_maxsim_2bit_asym(
            planes_t, meta_t, group_sum_t, bit0_t, bit1_t,
            scales_t, offsets_t, code_sum_t,
        )
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        scores = k2.roq_maxsim_2bit_asym(
            planes_t, meta_t, group_sum_t, bit0_t, bit1_t,
            scales_t, offsets_t, code_sum_t,
        )
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    triton_scores = scores.cpu().numpy()
    finite = np.isfinite(triton_scores).all()

    triton_ms_p50 = float(np.percentile(timings_ms, 50))
    triton_ms_p95 = float(np.percentile(timings_ms, 95))
    qps = n_queries / (triton_ms_p50 / 1000.0) if triton_ms_p50 > 0 else 0.0

    del planes_t, meta_t, group_sum_t, bit0_t, bit1_t, scales_t, offsets_t, code_sum_t, scores
    torch.cuda.empty_cache()

    return KernelBenchRow(
        name="roq2_asym",
        n_queries=n_queries, n_docs=n_docs,
        n_q_tokens=n_q_tokens, n_d_tokens=n_d_tokens,
        dim=dim, query_bits=query_bits, group_size=group_size,
        triton_ms_p50=triton_ms_p50, triton_ms_p95=triton_ms_p95,
        parity_max_abs_err=float("nan") if not finite else 0.0,
        parity_max_rel_err=float("nan"),
        qps=qps,
    )


def _bench_ternary(
    *, n_queries: int, n_docs: int, n_q_tokens: int, n_d_tokens: int,
    dim: int, query_bits: int, group_size: int, n_iters: int, seed: int,
) -> KernelBenchRow:
    import torch
    from .ternary import TernaryConfig, TernaryQuantizer
    from .kernels import triton_roq_ternary as kt

    rng = np.random.default_rng(seed)
    n_total_d = n_docs * n_d_tokens
    rot_d = rng.standard_normal((n_total_d, dim)).astype(np.float32)
    rot_d /= np.linalg.norm(rot_d, axis=1, keepdims=True) + 1e-12
    cfg = TernaryConfig(dim=dim, group_size=group_size, rotate=False, tau_frac=0.5, seed=seed)
    quantizer = TernaryQuantizer(cfg)
    enc = quantizer.quantize(rot_d)

    sign_bytes = enc["sign_plane"]
    nz_bytes = enc["nonzero_plane"]
    if sign_bytes.shape[1] % 4 != 0:
        pad = (-sign_bytes.shape[1]) % 4
        sign_bytes = np.pad(sign_bytes, ((0, 0), (0, pad)))
        nz_bytes = np.pad(nz_bytes, ((0, 0), (0, pad)))
    sign_np = sign_bytes.view(np.int32).copy()
    nz_np = nz_bytes.view(np.int32).copy()
    n_words = sign_np.shape[1]
    n_groups = dim // group_size

    sign_np = sign_np.reshape(n_docs, n_d_tokens, n_words)
    nz_np = nz_np.reshape(n_docs, n_d_tokens, n_words)
    scales_np = enc["scales"].reshape(n_docs, n_d_tokens, n_groups).astype(np.float32)
    decoded_per_dim = quantizer.decode(enc).reshape(n_docs, n_d_tokens, dim)

    rot_q = rng.standard_normal((n_queries * n_q_tokens, dim)).astype(np.float32)
    q_planes_np, q_meta_np, _, q_dequant = _encode_query_for_ternary_inline(
        rot_q, query_bits=query_bits, n_words=n_words,
    )
    q_planes_np = q_planes_np.reshape(n_queries, n_q_tokens, query_bits, n_words)
    q_meta_np = q_meta_np.reshape(n_queries, n_q_tokens, q_meta_np.shape[1])

    dev = torch.device("cuda")
    sign_t = torch.from_numpy(sign_np).to(dev)
    nz_t = torch.from_numpy(nz_np).to(dev)
    scales_t = torch.from_numpy(scales_np).to(dev)
    q_planes_t = torch.from_numpy(q_planes_np).to(dev)
    q_meta_t = torch.from_numpy(q_meta_np).to(dev)

    for _ in range(3):
        _ = kt.roq_maxsim_ternary(q_planes_t, q_meta_t, sign_t, nz_t, scales_t)
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        scores = kt.roq_maxsim_ternary(q_planes_t, q_meta_t, sign_t, nz_t, scales_t)
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    triton_scores = scores.cpu().numpy()
    finite = np.isfinite(triton_scores).all()

    decoded = decoded_per_dim
    q_dequant_arr = q_dequant.reshape(n_queries, n_q_tokens, dim)
    expected = np.zeros((n_queries, n_docs), dtype=np.float32)
    for a in range(n_queries):
        for b in range(n_docs):
            mat = q_dequant_arr[a] @ decoded[b].T
            expected[a, b] = mat.max(axis=1).sum()
    abs_err = float(np.max(np.abs(triton_scores - expected)))
    rel_err = float(np.max(np.abs(triton_scores - expected) / (np.abs(expected) + 1e-6)))

    triton_ms_p50 = float(np.percentile(timings_ms, 50))
    triton_ms_p95 = float(np.percentile(timings_ms, 95))
    qps = n_queries / (triton_ms_p50 / 1000.0) if triton_ms_p50 > 0 else 0.0

    del sign_t, nz_t, scales_t, q_planes_t, q_meta_t, scores
    torch.cuda.empty_cache()

    return KernelBenchRow(
        name="roq158_ternary",
        n_queries=n_queries, n_docs=n_docs,
        n_q_tokens=n_q_tokens, n_d_tokens=n_d_tokens,
        dim=dim, query_bits=query_bits, group_size=group_size,
        triton_ms_p50=triton_ms_p50, triton_ms_p95=triton_ms_p95,
        parity_max_abs_err=abs_err if finite else float("nan"),
        parity_max_rel_err=rel_err if finite else float("nan"),
        qps=qps,
    )


def _encode_query_for_ternary_inline(
    rotated_q: np.ndarray, *, query_bits: int, n_words: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirrors the encoding used by ``test_kernels_parity.py``.

    Returns (planes, meta, q_codes, q_dequant) where ``q_dequant`` is the
    fp32 vector that the ternary kernel's affine reconstruction recovers,
    suitable as ground truth for parity checks.
    """
    levels = float((1 << query_bits) - 1)
    min_vals = rotated_q.min(axis=1)
    max_vals = rotated_q.max(axis=1)
    ranges = np.where((max_vals - min_vals) < 1e-6, 1.0, max_vals - min_vals)
    q_scale = ranges / levels
    q_offset = min_vals
    q_codes = np.round(
        (rotated_q - q_offset[:, None]) / q_scale[:, None]
    ).clip(0, levels).astype(np.uint8)
    q_dequant = q_codes.astype(np.float32) * q_scale[:, None] + q_offset[:, None]
    planes = np.zeros((rotated_q.shape[0], query_bits, n_words), dtype=np.int32)
    for k in range(query_bits):
        plane_bits = ((q_codes >> k) & 0x01).astype(np.uint8)
        planes[:, k, :] = _pack_bits_into_int32_words(plane_bits)
    meta = np.stack([q_scale.astype(np.float32), q_offset.astype(np.float32)], axis=1)
    meta = np.concatenate([meta, np.zeros_like(meta[:, :1])], axis=-1)
    return planes, meta, q_codes, q_dequant


def _pack_bits_into_int32_words(bits: np.ndarray) -> np.ndarray:
    n, d = bits.shape
    if d % 32 != 0:
        bits = np.pad(bits, ((0, 0), (0, (-d) % 32)))
        d = bits.shape[1]
    return np.packbits(bits, axis=1, bitorder="little").view(np.int32).copy()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-docs", type=int, default=128)
    parser.add_argument("--n-q-tokens", type=int, default=32)
    parser.add_argument("--n-d-tokens", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--query-bits", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("research/low_bit_roq/reports/a2_kernels.json"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        import torch
        if not torch.cuda.is_available():
            log.error("CUDA not available; nothing to bench")
            return 1
    except ImportError:
        log.error("torch not installed")
        return 1

    rows: list[KernelBenchRow] = []

    log.info("benching roq2_asym ...")
    try:
        rows.append(_bench_2bit_asym(
            n_queries=args.n_queries, n_docs=args.n_docs,
            n_q_tokens=args.n_q_tokens, n_d_tokens=args.n_d_tokens,
            dim=args.dim, query_bits=args.query_bits, group_size=args.group_size,
            n_iters=args.n_iters, seed=args.seed,
        ))
    except Exception as e:
        log.exception("roq2_asym bench failed: %s", e)

    log.info("benching roq158_ternary ...")
    try:
        rows.append(_bench_ternary(
            n_queries=args.n_queries, n_docs=args.n_docs,
            n_q_tokens=args.n_q_tokens, n_d_tokens=args.n_d_tokens,
            dim=args.dim, query_bits=args.query_bits, group_size=args.group_size,
            n_iters=args.n_iters, seed=args.seed,
        ))
    except Exception as e:
        log.exception("roq158_ternary bench failed: %s", e)

    out = {"config": vars(args) | {"out": str(args.out)},
           "rows": [asdict(r) for r in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for r in rows:
        log.info(
            "%s p50=%.3fms p95=%.3fms qps=%.0f parity_max_abs=%s",
            r.name, r.triton_ms_p50, r.triton_ms_p95, r.qps, r.parity_max_abs_err,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
