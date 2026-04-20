"""
Numpy-only parity / sanity tests for the ternary (1.58-bit) encoder.

These tests are intentionally GPU-free so they run in CI on every commit.
Triton-kernel parity is in ``test_triton_kernels.py`` and is skipped when
no GPU is available.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")


def _has_voyager_index() -> bool:
    try:
        import voyager_index._internal.inference.quantization.rotational  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_voyager_index(), reason="voyager_index not installed"
)


def _make_data(n: int = 256, dim: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def test_ternary_decode_shape_and_values():
    from research.low_bit_roq.ternary import TernaryConfig, TernaryQuantizer

    x = _make_data(64, 128)
    q = TernaryQuantizer(TernaryConfig(dim=128, group_size=32, rotate=False))
    enc = q.quantize(x)
    dec = q.decode(enc)

    assert dec.shape == (64, 128)
    unique = np.unique(np.round(dec / (np.abs(dec[dec != 0]).min() + 1e-12)))
    assert set(unique).issubset({-1.0, 0.0, 1.0, -2.0, 2.0}) or len(unique) <= 64


def test_ternary_round_trip_recovers_inner_product_within_band():
    """Median |IP_true - IP_decoded| should be a small fraction of ||x|| ||y||."""
    from research.low_bit_roq.ternary import TernaryConfig, TernaryQuantizer

    x = _make_data(256, 128, seed=1)
    queries = _make_data(128, 128, seed=2)
    q = TernaryQuantizer(TernaryConfig(dim=128, group_size=32, rotate=False))
    enc = q.quantize(x)
    dec = q.decode(enc)

    ip_true = queries @ x.T
    ip_dec = queries @ dec.T
    err = np.abs(ip_true - ip_dec)
    median_err = float(np.median(err))
    assert median_err < 0.30, f"median IP error {median_err} too large"


def test_ternary_anisotropic_fit_runs():
    from research.low_bit_roq.ternary import TernaryConfig, TernaryQuantizer

    x = _make_data(64, 128)
    q = TernaryQuantizer(
        TernaryConfig(dim=128, group_size=32, rotate=False, fit_method="anisotropic")
    )
    enc = q.quantize(x)
    assert enc["scales"].shape == (64, 128 // 32)
    assert enc["tau"].shape == (64, 128 // 32)
    assert (enc["scales"] >= 0).all()


def test_ternary_query_bit_planes_layout():
    from research.low_bit_roq.ternary import TernaryConfig, TernaryQuantizer

    queries = _make_data(8, 128, seed=3)
    q = TernaryQuantizer(TernaryConfig(dim=128, group_size=32, rotate=False))
    planes, scales, offsets = q.encode_query_bit_planes(queries, query_bits=4)
    assert planes.shape == (8, 4, 128 // 8)
    assert scales.shape == (8,)
    assert offsets.shape == (8,)
    assert planes.dtype == np.uint8


def test_ternary_rejects_invalid_group_size():
    import pytest as _pt
    from research.low_bit_roq.ternary import TernaryConfig, TernaryQuantizer

    with _pt.raises(ValueError):
        TernaryQuantizer(TernaryConfig(dim=128, group_size=16, rotate=False))
