"""
LEMUR / ANN / router production-lane integration glue (A5).

The plan requires every Phase-A and Phase-B winner to "light up" through
the full LEMUR -> ANN -> router -> ROQ rerank -> optional exact MaxSim
lane on a real shard build, not a synthetic kernel benchmark. This module
is the thin glue layer that gets new compression / kernel variants into
the lane without forking the production code.

Touchpoints (per the plan A5 section):

- ``shard_engine/_builder/pipeline.py`` — wire ``RoQConfig(num_bits=2,
  group_size=...)`` into the LEMUR path; add new layout constants if
  required.
- ``shard_engine/serving_config.py`` — extend Compression / StorageLayout
  via ``register_research_compression`` below (additive, no edits to the
  enum file).
- ``shard_engine/centroid_router.py`` — add normalize-on-route as an A1
  axis (already supported via ``patch_centroid_router_normalize``).
- ``shard_engine/_lemur/ann.py`` — IndexFlatIP keeps fp16 proxy weights
  for Phase A.
- ``shard_engine/_lemur/backends.py`` — for B2 LEMUR-tangent, emit
  ``LemurTangentFeaturizer.featurize`` instead of the default token
  embeddings.
- ``shard_engine/scorer.py`` — add ``score_roq2_topk`` /
  ``score_ternary_topk`` / ``score_rroq2_topk`` paralleling the existing
  ``score_roq4_topk``.
- Persistence migration: read-old-write-new, gated behind a config flag.

This module exposes small wrapper / register / patch utilities so the
research code stays additive — none of the production .py files need to
be edited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compression / StorageLayout registry (additive)
# ---------------------------------------------------------------------------


@dataclass
class ResearchCompression:
    """Description of a research compression mode that the build pipeline
    can resolve to (encoder, scorer) at runtime.

    ``encode_fn`` runs at index time over the per-shard token tensor.
    ``score_fn`` runs at query time for the rerank tier; signature should
    mirror ``score_roq4_topk``.
    ``persist_fn`` writes the per-shard payload (codes + meta + new fields)
    to the shard directory.
    """

    name: str
    bits: float
    encode_fn: Callable[..., dict[str, Any]]
    score_fn: Callable[..., tuple[list[int], list[float]]]
    persist_fn: Callable[..., None]
    config_defaults: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ResearchCompression] = {}


def register_research_compression(c: ResearchCompression) -> None:
    if c.name in _REGISTRY:
        raise ValueError(f"compression {c.name!r} already registered")
    _REGISTRY[c.name] = c
    log.info("registered research compression %s (bits=%.2f)", c.name, c.bits)


def get_research_compression(name: str) -> ResearchCompression:
    if name not in _REGISTRY:
        raise KeyError(f"unknown research compression {name!r}; known: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_research_compressions() -> list[str]:
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# 2-bit asymmetric persistence layout
# ---------------------------------------------------------------------------


def persist_2bit_asym_layout(
    shard_dir: Path,
    *,
    quantized: dict[str, Any],
    group_size: int,
) -> None:
    """Write the on-disk layout the asymmetric 2-bit kernel reads.

    Layout (one .npy per field, mmap-friendly):

    - ``roq2_asym_bit0.npy`` — (N, n_words) int32
    - ``roq2_asym_bit1.npy`` — (N, n_words) int32
    - ``roq2_asym_scales.npy`` — (N, n_groups) float32
    - ``roq2_asym_offsets.npy`` — (N, n_groups) float32
    - ``roq2_asym_code_sums.npy`` — (N, n_groups) float32
    """
    from .kernels.triton_roq_2bit_asym import split_2bit_codes_to_bit_planes

    bit0, bit1 = split_2bit_codes_to_bit_planes(quantized["codes"])
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "roq2_asym_bit0.npy", bit0)
    np.save(shard_dir / "roq2_asym_bit1.npy", bit1)
    np.save(shard_dir / "roq2_asym_scales.npy", quantized["scales"].astype(np.float32))
    np.save(shard_dir / "roq2_asym_offsets.npy", quantized["offsets"].astype(np.float32))
    np.save(
        shard_dir / "roq2_asym_code_sums.npy",
        quantized["code_sums"].astype(np.float32),
    )


def persist_ternary_layout(
    shard_dir: Path,
    *,
    encoded: dict[str, np.ndarray],
) -> None:
    """Write ternary encoder output to disk.

    Layout:
    - ``roq158_sign.npy``    — (N, n_words) int32
    - ``roq158_nonzero.npy`` — (N, n_words) int32
    - ``roq158_scales.npy``  — (N, n_groups) float32
    - ``roq158_tau.npy``     — (N, n_groups) float32 (debug; not used at score-time)
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    sign = encoded["sign_plane"]
    nonzero = encoded["nonzero_plane"]
    if sign.dtype == np.uint8:
        sign = sign.view(np.int32) if sign.shape[1] % 4 == 0 else _pack_uint8_to_int32(sign)
        nonzero = nonzero.view(np.int32) if nonzero.shape[1] % 4 == 0 else _pack_uint8_to_int32(nonzero)
    np.save(shard_dir / "roq158_sign.npy", sign)
    np.save(shard_dir / "roq158_nonzero.npy", nonzero)
    np.save(shard_dir / "roq158_scales.npy", encoded["scales"].astype(np.float32))
    np.save(shard_dir / "roq158_tau.npy", encoded["tau"].astype(np.float32))


def _pack_uint8_to_int32(arr: np.ndarray) -> np.ndarray:
    n, d = arr.shape
    pad = (-d) % 4
    if pad:
        arr = np.pad(arr, ((0, 0), (0, pad)))
    return arr.view(np.int32)


# ---------------------------------------------------------------------------
# Score functions (research)
# ---------------------------------------------------------------------------


def score_roq2_asym_topk(
    queries_planes,
    queries_meta,
    queries_group_sum,
    docs_bit0,
    docs_bit1,
    docs_scales,
    docs_offsets,
    docs_code_sum,
    doc_ids: Sequence[int],
    k: int = 10,
    documents_mask=None,
):
    from .kernels.triton_roq_2bit_asym import roq_maxsim_2bit_asym

    scores = roq_maxsim_2bit_asym(
        queries_planes,
        queries_meta,
        queries_group_sum,
        docs_bit0,
        docs_bit1,
        docs_scales,
        docs_offsets,
        docs_code_sum,
        documents_mask=documents_mask,
    ).squeeze(0)
    final_k = min(k, scores.shape[0])
    top_sc, top_idx = scores.topk(final_k)
    idx_list = top_idx.cpu().tolist()
    return [doc_ids[i] for i in idx_list], top_sc.cpu().tolist()


def score_ternary_topk(
    queries_planes,
    queries_meta,
    docs_sign,
    docs_nz,
    docs_scales,
    doc_ids: Sequence[int],
    k: int = 10,
    documents_mask=None,
):
    from .kernels.triton_roq_ternary import roq_maxsim_ternary

    scores = roq_maxsim_ternary(
        queries_planes,
        queries_meta,
        docs_sign,
        docs_nz,
        docs_scales,
        documents_mask=documents_mask,
    ).squeeze(0)
    final_k = min(k, scores.shape[0])
    top_sc, top_idx = scores.topk(final_k)
    idx_list = top_idx.cpu().tolist()
    return [doc_ids[i] for i in idx_list], top_sc.cpu().tolist()


# ---------------------------------------------------------------------------
# CentroidRouter normalize-on patch (A1 axis)
# ---------------------------------------------------------------------------


def patch_centroid_router_normalize(centroid_router) -> None:
    """Force the router to L2-normalize both centroids and queries before
    inner product. Reversible: stash original ``route``."""
    import torch
    import torch.nn.functional as F

    if getattr(centroid_router, "_normalize_patched", False):
        return
    original_route = type(centroid_router).route

    def normalized_route(self, q_tokens, *args, **kwargs):
        q_tokens = F.normalize(q_tokens, dim=-1)
        if hasattr(self, "centroid_table"):
            self.centroid_table = F.normalize(self.centroid_table, dim=-1)
        return original_route(self, q_tokens, *args, **kwargs)

    centroid_router.route = normalized_route.__get__(
        centroid_router, type(centroid_router)
    )
    centroid_router._normalize_patched = True


# ---------------------------------------------------------------------------
# Migration: read-old-write-new
# ---------------------------------------------------------------------------


@dataclass
class MigrationPlan:
    src_compression: str
    dst_compression: str
    src_dir: Path
    dst_dir: Path
    verify_checksums: bool = True


def migrate_shard(plan: MigrationPlan) -> None:
    """Read an existing shard with one compression layout and write a new
    layout. Intended for the C3 production replay test.

    The function is intentionally stateless — it does not modify the
    in-place shard. Hardware engineering can run it offline against the
    production indices, then point serving at the new directory.
    """
    log.info(
        "migrating shard %s -> %s  (%s -> %s)",
        plan.src_dir,
        plan.dst_dir,
        plan.src_compression,
        plan.dst_compression,
    )
    plan.dst_dir.mkdir(parents=True, exist_ok=True)
    src = get_research_compression(plan.src_compression)
    dst = get_research_compression(plan.dst_compression)
    raise NotImplementedError(
        "Concrete migrate logic depends on the corpus checkpoint format; "
        "wire to ``shard_engine/_builder/pipeline.py:rebuild_shard`` once "
        "C3 needs to land on a real cluster."
    )
