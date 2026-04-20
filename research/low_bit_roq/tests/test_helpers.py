"""
Tests for the small numpy-only helpers that don't need GPU or
voyager-index: salience pruning, mixed-precision promotion, spherical
k-means, tangent query, anisotropic codebook fitting, per-cluster PCA,
shortcut edges, and filter-aware probing.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_compute_salience_norm_signal():
    from research.low_bit_roq import salience

    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((128, 32)).astype(np.float32)
    sal = salience.compute_salience(tokens, signal="norm")
    assert sal.shape == (128,)
    assert (sal >= 0).all()
    np.testing.assert_allclose(sal, np.linalg.norm(tokens, axis=1), rtol=1e-5)


def test_compute_salience_idf_requires_inputs():
    from research.low_bit_roq import salience

    with pytest.raises(ValueError):
        salience.compute_salience(np.zeros((4, 4)), signal="idf")


def test_prune_by_quantile_respects_floor():
    from research.low_bit_roq import salience

    tokens = np.tile(np.arange(20, dtype=np.float32)[:, None], (1, 4))
    sal = np.arange(20, dtype=np.float32)
    offsets = np.array([0, 10, 20], dtype=np.int64)
    cfg = salience.SalienceConfig(prune_quantile=0.8, min_tokens_per_doc=4)
    kept, new_off = salience.prune_by_quantile(tokens, sal, offsets, cfg)
    counts = np.diff(new_off)
    assert (counts >= 4).all(), counts
    assert kept.shape[0] == counts.sum()


def test_prune_by_quantile_zero_returns_input():
    from research.low_bit_roq import salience

    tokens = np.zeros((6, 4), dtype=np.float32)
    sal = np.arange(6, dtype=np.float32)
    off = np.array([0, 3, 6])
    cfg = salience.SalienceConfig(prune_quantile=0.0)
    kept, new_off = salience.prune_by_quantile(tokens, sal, off, cfg)
    assert kept is tokens or kept.shape == tokens.shape
    np.testing.assert_array_equal(new_off, off)


def test_select_promoted_tokens_picks_top_quantile():
    from research.low_bit_roq.mixed_precision import (
        MixedPrecisionConfig,
        select_promoted_tokens,
    )

    sal = np.arange(100, dtype=np.float32)
    cfg = MixedPrecisionConfig(promote_fraction=0.10, promote_signal="salience")
    mask = select_promoted_tokens(salience=sal, cfg=cfg)
    assert mask.sum() == 10
    assert mask[-10:].all()
    assert not mask[:-10].any()


def test_spherical_kmeans_recovers_clusters_for_separated_blobs():
    from research.low_bit_roq.spherical_kmeans import (
        SphericalKMeansConfig,
        cluster,
        l2_normalize,
    )

    rng = np.random.default_rng(0)
    centers = np.eye(3, dtype=np.float32)
    pts = []
    truth = []
    for c_idx in range(3):
        block = centers[c_idx] + 0.05 * rng.standard_normal((50, 3)).astype(np.float32)
        pts.append(block)
        truth.extend([c_idx] * 50)
    x = np.concatenate(pts, axis=0)
    cfg = SphericalKMeansConfig(n_clusters=3, n_iter=20, seed=42)
    centroids, assign = cluster(x, cfg)
    assert centroids.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-4)
    purity = 0
    for c in range(3):
        members = np.array(truth)[assign == c]
        if members.size:
            purity += np.bincount(members).max()
    assert purity / x.shape[0] > 0.9


def test_tangent_geodesic_score_matches_manual_formula():
    from research.low_bit_roq import tangent_query

    rng = np.random.default_rng(7)
    centroids = rng.standard_normal((10, 16)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    q = rng.standard_normal(16).astype(np.float32)
    q /= np.linalg.norm(q)
    score = tangent_query.tangent_geodesic_score(q, centroids, lam=0.25)
    cos_t = np.clip(centroids @ q, -1 + 1e-7, 1 - 1e-7)
    theta = np.arccos(cos_t)
    expected = cos_t - 0.25 * theta ** 2
    np.testing.assert_allclose(score, expected, rtol=1e-5)


def test_log_map_unit_sphere_is_orthogonal_to_centroid():
    from research.low_bit_roq.tangent_query import log_map_unit_sphere

    rng = np.random.default_rng(0)
    c = rng.standard_normal((5, 8)).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    q = rng.standard_normal((5, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    tan = log_map_unit_sphere(c, q)
    inner = (c * tan).sum(axis=-1)
    np.testing.assert_allclose(inner, 0.0, atol=1e-5)


def test_anisotropic_min_max_returns_finite_scales():
    from research.low_bit_roq.anisotropic import fit_anisotropic_min_max

    rng = np.random.default_rng(0)
    grouped = rng.standard_normal((16, 8, 16)).astype(np.float32)
    scales, offsets = fit_anisotropic_min_max(grouped, bits=2, eta=4.0)
    assert scales.shape == (16, 8)
    assert np.isfinite(scales).all()
    assert (scales > 0).all()


def test_fit_ternary_codebook_picks_finite_tau():
    from research.low_bit_roq.anisotropic import fit_ternary_codebook

    rng = np.random.default_rng(0)
    grouped = rng.standard_normal((8, 4, 16)).astype(np.float32)
    tau, scale = fit_ternary_codebook(grouped)
    assert tau.shape == (8, 4)
    assert scale.shape == (8, 4)
    assert (tau >= 0).all()
    assert (scale >= 0).all()
