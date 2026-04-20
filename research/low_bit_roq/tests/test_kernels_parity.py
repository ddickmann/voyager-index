"""
Numpy-only parity tests for the kernel reference implementations.

These tests use the python ``reference_score_ternary`` shipped with the
ternary kernel module. We do *not* require Triton or CUDA — the goal is to
exercise the bit-level decomposition we implement on the host so that the
GPU kernel has a known-good comparison target. Triton-on-GPU parity
(``test_triton_kernels.py``) is a separate file and is ``skipif``-gated.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _maybe_import(mod: str):
    try:
        return importlib.import_module(mod)
    except Exception:
        return None


def test_ternary_reference_matches_decoded_dot_product():
    np.random.seed(0)
    dim = 128
    group_size = 32
    n_groups = dim // group_size
    n_words = dim // 32
    n_docs = 5
    n_queries = 3

    rotated_doc = np.random.standard_normal((n_docs, dim)).astype(np.float32)
    rotated_doc /= np.linalg.norm(rotated_doc, axis=1, keepdims=True) + 1e-12
    rotated_q = np.random.standard_normal((n_queries, dim)).astype(np.float32)

    grouped = rotated_doc.reshape(n_docs, n_groups, group_size)
    std = grouped.std(axis=2, ddof=0) + 1e-8
    tau = std * 0.5
    sign = (rotated_doc > 0).astype(np.uint8)
    nonzero = (np.abs(rotated_doc) > np.repeat(tau, group_size, axis=1)).astype(np.uint8)
    abs_g = np.abs(grouped)
    mask_init = abs_g > tau[..., None]
    num = (abs_g * mask_init).sum(axis=2)
    den = mask_init.sum(axis=2).clip(min=1)
    scales = (num / den).astype(np.float32)

    decoded = (
        (2.0 * sign.astype(np.float32) - 1.0)
        * nonzero.astype(np.float32)
        * np.repeat(scales, group_size, axis=1)
    )

    query_bits = 4
    levels = float((1 << query_bits) - 1)
    min_vals = rotated_q.min(axis=1)
    max_vals = rotated_q.max(axis=1)
    ranges = np.where((max_vals - min_vals) < 1e-6, 1.0, max_vals - min_vals)
    q_scale = ranges / levels
    q_offset = min_vals
    q_codes = np.round((rotated_q - q_offset[:, None]) / q_scale[:, None]).clip(0, levels).astype(np.uint8)
    queries_dequant = q_codes.astype(np.float32) * q_scale[:, None] + q_offset[:, None]

    expected = queries_dequant @ decoded.T

    sign_words = _pack_into_int32_words(sign)
    nz_words = _pack_into_int32_words(nonzero)
    q_planes = np.zeros((n_queries, query_bits, n_words), dtype=np.int32)
    for k in range(query_bits):
        plane = ((q_codes >> k) & 0x01).astype(np.uint8)
        q_planes[:, k, :] = _pack_into_int32_words(plane)

    queries_planes = q_planes[:, None, :, :]
    queries_meta = np.stack([q_scale, q_offset], axis=1).astype(np.float32)[:, None, :]
    queries_meta = np.concatenate([queries_meta, np.zeros_like(queries_meta[..., :1])], axis=-1)
    docs_sign = sign_words[:, None, :]
    docs_nz = nz_words[:, None, :]
    docs_scales = scales[:, None, :]

    kernels = _maybe_import("research.low_bit_roq.kernels.triton_roq_ternary")
    if kernels is None:
        pytest.skip("triton not installed")
    ref = kernels.reference_score_ternary(
        queries_planes,
        queries_meta,
        docs_sign,
        docs_nz,
        docs_scales,
    )

    np.testing.assert_allclose(ref, expected, rtol=1e-3, atol=1e-3)


def _pack_into_int32_words(bits: np.ndarray) -> np.ndarray:
    """Pack ``(N, dim)`` 0/1 array into ``(N, dim/32)`` int32, little-endian."""
    n, d = bits.shape
    if d % 32 != 0:
        bits = np.pad(bits, ((0, 0), (0, (-d) % 32)))
        d = bits.shape[1]
    packed_bytes = np.packbits(bits, axis=1, bitorder="little")
    return packed_bytes.view(np.int32).copy()


def test_split_2bit_codes_to_bit_planes_matches_dequant():
    """Take some packed 2-bit codes, split into bit-planes, and verify the
    bit0+2*bit1 reconstruction matches the original codes."""
    kernels = _maybe_import("research.low_bit_roq.kernels.triton_roq_2bit_asym")
    if kernels is None:
        pytest.skip("triton not installed")

    rng = np.random.default_rng(0)
    n = 8
    dim = 64
    codes = rng.integers(0, 4, size=(n, dim)).astype(np.uint8)
    packed = np.zeros((n, dim // 4), dtype=np.uint8)
    packed[:, :] = (
        (codes[:, 0::4] << 6)
        | (codes[:, 1::4] << 4)
        | (codes[:, 2::4] << 2)
        | codes[:, 3::4]
    )
    bit0_words, bit1_words = kernels.split_2bit_codes_to_bit_planes(packed)
    bit0 = np.unpackbits(bit0_words.view(np.uint8), axis=1, bitorder="little")[:, :dim]
    bit1 = np.unpackbits(bit1_words.view(np.uint8), axis=1, bitorder="little")[:, :dim]
    reconstructed = bit0.astype(np.uint8) + 2 * bit1.astype(np.uint8)
    np.testing.assert_array_equal(reconstructed, codes)


def test_encode_query_for_2bit_asym_round_trip_matches_dequant():
    kernels = _maybe_import("research.low_bit_roq.kernels.triton_roq_2bit_asym")
    if kernels is None:
        pytest.skip("triton not installed")
    rng = np.random.default_rng(1)
    rotated = rng.standard_normal((4, 128)).astype(np.float32)
    planes, meta, group_sum = kernels.encode_query_for_2bit_asym(
        rotated, query_bits=4, group_size=16, n_groups=8
    )
    assert planes.shape == (4, 4, 4)  # n=4, query_bits=4, n_words=128/32=4
    assert meta.shape == (4, 3)
    assert group_sum.shape == (4, 8)

    levels = (1 << 4) - 1
    min_v = rotated.min(axis=1)
    max_v = rotated.max(axis=1)
    rng_ = np.where((max_v - min_v) < 1e-6, 1.0, max_v - min_v)
    scales = rng_ / levels
    quant = np.round((rotated - min_v[:, None]) / scales[:, None]).clip(0, levels).astype(np.uint8)
    expected_group_sum = quant.reshape(4, 8, 16).sum(axis=2).astype(np.float32)
    np.testing.assert_array_equal(group_sum.astype(np.float32), expected_group_sum)
