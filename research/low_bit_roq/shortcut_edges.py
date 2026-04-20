"""
Shortcut edges from quantization disagreement.

Plan reference: cross-cut 2 (also Innovation 1 / 4 from
``voyager_index_inno.md``). Mines pairs where lowbit-rank is far from
fp16-rank on a held-out query set, builds a small directed-edge sketch
that the router can use to boost candidates for similar query patterns.

Gates:
- bridge-edge-degree cap (per-doc and per-query-cluster)
- edge aging (TTL = N rebuilds)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ShortcutConfig:
    rank_disagreement_threshold: int = 50
    """Add an edge only when (lowbit_rank - fp16_rank) > threshold."""
    max_out_degree_per_doc: int = 8
    max_out_degree_per_query_cluster: int = 16
    edge_ttl_rebuilds: int = 4


@dataclass
class ShortcutEdge:
    src_doc: int
    dst_doc: int
    weight: float
    age: int = 0


def mine_disagreements(
    lowbit_ranks_per_query: dict[int, np.ndarray],
    fp16_ranks_per_query: dict[int, np.ndarray],
    *,
    cfg: ShortcutConfig,
) -> list[ShortcutEdge]:
    edges: list[ShortcutEdge] = []
    for q, lowbit_ranks in lowbit_ranks_per_query.items():
        fp16_ranks = fp16_ranks_per_query.get(q)
        if fp16_ranks is None or fp16_ranks.shape != lowbit_ranks.shape:
            continue
        diffs = lowbit_ranks - fp16_ranks
        bad_mask = diffs > cfg.rank_disagreement_threshold
        good_mask = diffs < -cfg.rank_disagreement_threshold
        bad_docs = np.where(bad_mask)[0]
        good_docs = np.where(good_mask)[0]
        for src in bad_docs[: cfg.max_out_degree_per_query_cluster]:
            for dst in good_docs[: cfg.max_out_degree_per_doc]:
                weight = float(diffs[src] - diffs[dst])
                edges.append(ShortcutEdge(int(src), int(dst), weight))
    log.info("mined %d shortcut edges", len(edges))
    return edges


def age_edges(edges: list[ShortcutEdge], *, ttl: int) -> list[ShortcutEdge]:
    return [
        ShortcutEdge(e.src_doc, e.dst_doc, e.weight, e.age + 1)
        for e in edges
        if e.age + 1 < ttl
    ]
