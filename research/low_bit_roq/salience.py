"""
Token salience pruning at index time.

Plan reference: Phase A4. Drops bottom-X% doc tokens before quantization.
At 50% prune + best-A1 2-bit, total disk is ~3.5% of fp16 — that may end
up being the headline number rather than bit-reduction.

Four salience signals (sweep in A4):

1. ``norm`` — ``||token||`` after pre-encoder L2. Cheapest. For
   ColBERT-style models token norm correlates with informativeness because
   pad/CLS/SEP tokens have small norms post-FFN.
2. ``idf`` — corpus-level token-position frequency. Use only for models
   with a public vocabulary mapping (ColBERTv2). Skip otherwise.
3. ``attention_mass`` — sum of MaxSim contributions over a held-out 5k
   training-query set. Most expensive (one full retrieval pass) but
   strongest signal.
4. ``self_attention_norm`` — encoder-internal attention norms if available
   without re-encoding. Skip if only the embedding outputs are persisted.

This module exposes ``compute_salience`` and ``prune_by_quantile`` so the
shard build pipeline can compose either one with the existing
``RotationalQuantizer`` step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)


SALIENCE_SIGNALS = ("norm", "idf", "attention_mass", "self_attention_norm")


@dataclass
class SalienceConfig:
    signal: str = "norm"
    prune_quantile: float = 0.0  # 0..0.5
    """Fraction of tokens to drop. 0.0 = keep all."""
    min_tokens_per_doc: int = 4
    """Floor on retained tokens per document. Prevents documents losing all
    tokens on the long tail of the salience distribution."""


def compute_salience(
    tokens: np.ndarray,
    *,
    signal: str = "norm",
    idf_table: dict[int, float] | None = None,
    token_ids: np.ndarray | None = None,
    attention_mass_table: np.ndarray | None = None,
) -> np.ndarray:
    """Per-token salience score (higher = more important).

    ``tokens`` shape (N, D). Optional ``token_ids`` (N,) is required for
    the IDF signal. ``attention_mass_table`` (N,) is required for the
    attention-mass signal.
    """
    if signal == "norm":
        return np.linalg.norm(tokens, axis=1).astype(np.float32)
    if signal == "idf":
        if idf_table is None or token_ids is None:
            raise ValueError("idf signal requires idf_table and token_ids")
        return np.array([idf_table.get(int(tid), 0.0) for tid in token_ids], dtype=np.float32)
    if signal == "attention_mass":
        if attention_mass_table is None:
            raise ValueError("attention_mass signal requires attention_mass_table")
        return np.asarray(attention_mass_table, dtype=np.float32)
    if signal == "self_attention_norm":
        raise NotImplementedError(
            "self_attention_norm requires encoder hooks; skip per plan A4 if "
            "embedding outputs are the only persisted artifact"
        )
    raise ValueError(f"unknown salience signal: {signal}")


def prune_by_quantile(
    tokens: np.ndarray,
    salience: np.ndarray,
    doc_offsets: np.ndarray,
    cfg: SalienceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop bottom-``prune_quantile`` of tokens corpus-wide, then re-densify
    per-doc offsets.

    ``doc_offsets`` of shape (n_docs + 1,) follows the existing voyager-index
    flat-tokens-with-offsets layout.

    Returns (kept_tokens, new_doc_offsets).
    """
    if cfg.prune_quantile <= 0.0:
        return tokens, doc_offsets
    threshold = float(np.quantile(salience, cfg.prune_quantile))
    keep_mask = salience > threshold

    n_docs = doc_offsets.shape[0] - 1
    new_offsets = np.zeros_like(doc_offsets)
    kept_indices: list[int] = []
    for d in range(n_docs):
        start, end = int(doc_offsets[d]), int(doc_offsets[d + 1])
        doc_keep = keep_mask[start:end].copy()
        if doc_keep.sum() < cfg.min_tokens_per_doc:
            local_sal = salience[start:end]
            top_idx = np.argpartition(-local_sal, kth=min(cfg.min_tokens_per_doc - 1, len(local_sal) - 1))[: cfg.min_tokens_per_doc]
            doc_keep = np.zeros_like(doc_keep)
            doc_keep[top_idx] = True
        kept_indices.extend(int(start + i) for i in np.nonzero(doc_keep)[0])
        new_offsets[d + 1] = new_offsets[d] + int(doc_keep.sum())
    kept = tokens[np.asarray(kept_indices, dtype=np.int64)]
    log.info(
        "prune_by_quantile: kept %d / %d tokens (%.1f%% dropped)",
        kept.shape[0],
        tokens.shape[0],
        100 * (1 - kept.shape[0] / tokens.shape[0]),
    )
    return kept, new_offsets


def sweep_prune_rates(
    tokens: np.ndarray,
    doc_offsets: np.ndarray,
    *,
    signal: str,
    rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5),
    eval_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
    **signal_kwargs,
) -> list[dict[str, float]]:
    """A4 sweep entry point.

    ``eval_fn(tokens, offsets) -> {recall_at_10: ..., ...}`` is called for
    each prune rate. The harness wraps this so the per-rate result rolls up
    into a single PROGRESS.md entry.
    """
    sal = compute_salience(tokens, signal=signal, **signal_kwargs)
    rows = []
    for rate in rates:
        cfg = SalienceConfig(signal=signal, prune_quantile=rate)
        kept_tokens, kept_offsets = prune_by_quantile(tokens, sal, doc_offsets, cfg)
        row: dict[str, float] = {
            "prune_rate": float(rate),
            "kept_tokens": int(kept_tokens.shape[0]),
            "disk_fraction": float(kept_tokens.shape[0] / max(tokens.shape[0], 1)),
        }
        if eval_fn is not None:
            row.update(eval_fn(kept_tokens, kept_offsets))
        rows.append(row)
    return rows
