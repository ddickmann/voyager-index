"""Tests for cross-cut helpers (shortcut edges, filter-aware probing)."""

from __future__ import annotations

import numpy as np

from research.low_bit_roq import filter_aware, shortcut_edges


def test_shortcut_edges_only_above_threshold():
    cfg = shortcut_edges.ShortcutConfig(rank_disagreement_threshold=20)
    fp16 = np.array([0, 1, 2, 3, 4, 100, 6, 7])
    lowbit = np.array([0, 1, 2, 90, 4, 5, 6, 7])
    edges = shortcut_edges.mine_disagreements(
        {0: lowbit}, {0: fp16}, cfg=cfg
    )
    assert any(e.src_doc == 3 for e in edges)
    assert any(e.dst_doc == 5 for e in edges)


def test_shortcut_edges_aging_drops_old():
    cfg = shortcut_edges.ShortcutConfig(rank_disagreement_threshold=10)
    edges = [shortcut_edges.ShortcutEdge(0, 1, 1.0, age=3)]
    aged = shortcut_edges.age_edges(edges, ttl=4)
    assert aged == []


def test_filter_sketch_per_cluster_fraction_passing():
    cfg = filter_aware.FilterSketchConfig(n_buckets=64, n_hashes=2)
    sketch = filter_aware.PerClusterFilterSketch(n_clusters=4, cfg=cfg)

    doc_to_cluster = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    doc_to_filter = np.array([7, 7, 8, 9, 7, 11, 12, 13, 14, 15])
    sketch.fit(doc_to_cluster, doc_to_filter)
    assert sketch.fraction_passing(0, 7) >= 2 / 3
    assert sketch.fraction_passing(2, 11) >= 1 / 4


def test_filter_sketch_select_clusters_drops_empty():
    cfg = filter_aware.FilterSketchConfig(
        n_buckets=64, n_hashes=2, skip_cluster_threshold=0.9
    )
    sketch = filter_aware.PerClusterFilterSketch(n_clusters=3, cfg=cfg)
    sketch.fit(np.array([0, 0, 0, 1, 1, 1]), np.array([1, 1, 1, 5, 5, 5]))
    kept = sketch.select_clusters([0, 1, 2], query_filter=1)
    assert 0 in kept
    assert 1 not in kept and 2 not in kept
