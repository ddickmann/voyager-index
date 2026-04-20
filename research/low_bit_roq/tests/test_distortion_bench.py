"""Smoke tests for the offline distortion benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _has_voyager_index() -> bool:
    try:
        import voyager_index._internal.inference.quantization.rotational  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_voyager_index(), reason="voyager_index not installed"
)


def test_sweep_returns_one_row_per_bit_setting():
    from research.low_bit_roq import distortion_bench

    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((512, 128)).astype(np.float32)
    tokens /= np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-12
    queries = tokens[:64]
    rows = distortion_bench.sweep(
        tokens, queries, bits=[1.0, 1.58, 2.0, 4.0], group_size=32, fwht=True, seed=0
    )
    assert len(rows) == 4
    bit_set = sorted({r.bits for r in rows})
    assert bit_set == [1.0, 1.58, 2.0, 4.0]
    for r in rows:
        assert 0.0 <= r.angular_error_p50_deg <= 180.0
        assert 0.0 <= r.nn1_preservation <= 1.0


def test_sweep_main_writes_report(tmp_path: Path):
    from research.low_bit_roq import distortion_bench

    out = tmp_path / "distortion.json"
    rc = distortion_bench.main(
        [
            "--synthetic",
            "--n-tokens", "256",
            "--n-queries", "32",
            "--dim", "64",
            "--bits", "1.0", "1.58", "2.0",
            "--group-size", "32",
            "--out", str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "rows" in data and len(data["rows"]) == 3
