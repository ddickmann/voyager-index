"""Offline fusion-weight grid sweep for groundedness (Phase J1).

Once the four peer channels (reverse_context_calibrated, literal_guarded,
nli_aggregate, optional semantic_entropy / structured_source_guarded)
are trustworthy on their own, the remaining win comes from picking
fusion weights that keep *every stratum* above the precision floor -
not just the easy ones. This script does that offline.

Design:

- We encode and score each (context, positive, negative) triple *once*
  against the production scorer, recording the raw channel values and
  the stratum label.
- For each weight combination in a deterministic grid we re-fuse in
  Python with :func:`fuse_groundedness_v2` and compute the per-stratum
  F1 at a shared threshold. The headline objective is the **minimum
  per-stratum F1** (the weakest link rule): a weight combination is
  only a win if it keeps every stratum honest.
- We persist the chosen weights as an env-var defaults JSON that the
  runtime service reads at startup, so the sweep never ships a binary
  change without review.

Usage:

    python -m research.triangular_maxsim.sweep_fusion_weights \
        --pairs-per-stratum 80 \
        --threshold 0.55 \
        --out research/triangular_maxsim/reports/fusion_weights.json

This script intentionally does **not** overwrite the default weights in
``groundedness_nli.py``; the JSON output is meant to be committed and
loaded via ``VOYAGER_GROUNDEDNESS_FUSION_W_*`` env vars. That keeps
production defaults reversible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from research.triangular_maxsim.groundedness_external_eval import (  # noqa: E402
    _encode_pair_side,
    _load_nli_provider,
    _load_provider,
    _load_reranker,
    _score_pair_side,
    build_null_bank_embeddings,
)
from research.triangular_maxsim.groundedness_minimal_pairs import (  # noqa: E402
    MinimalPair,
    build_minimal_pairs,
)
from voyager_index._internal.server.api.groundedness_nli import (  # noqa: E402
    fuse_groundedness_v2,
)


# ----------------------------------------------------------------------
# Scoring utilities
# ----------------------------------------------------------------------


def _channels_from_score(scored: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract the four fusion channels from a single scored response."""

    scores = scored["scores"]
    return {
        "calibrated": (
            float(scores["reverse_context_calibrated"])
            if scores.get("reverse_context_calibrated") is not None
            else float(scores.get("reverse_context") or 0.0)
        ),
        "literal": (
            float(scores["literal_guarded"]) if scores.get("literal_guarded") is not None else None
        ),
        "nli": (
            float(scores["nli_aggregate"]) if scores.get("nli_aggregate") is not None else None
        ),
        "semantic_entropy": (
            float(scores["semantic_entropy_aggregate"])
            if scores.get("semantic_entropy_aggregate") is not None
            else None
        ),
        "structured": (
            float(scores["structured_source_guarded"])
            if scores.get("structured_source_guarded") is not None
            else None
        ),
    }


def _collect_channel_samples(
    *,
    pairs: Sequence[MinimalPair],
    provider,
    null_bank_embeddings: Optional[Sequence[torch.Tensor]],
    nli_provider,
    nli_reranker,
    nli_concat_premises: Optional[bool],
    nli_use_atomic_claims: Optional[bool],
) -> List[Dict[str, Any]]:
    """Score every pair once and return the raw channel table.

    We only call the expensive scorer once per side; the weight grid
    only replays :func:`fuse_groundedness_v2` over these cached rows.
    """

    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        pos_materials = _encode_pair_side(
            provider=provider, context=pair.context, response=pair.positive
        )
        pos_scored = _score_pair_side(
            materials=pos_materials,
            response_text=pair.positive,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_use_atomic_claims=nli_use_atomic_claims,
        )
        neg_materials = _encode_pair_side(
            provider=provider, context=pair.context, response=pair.negative
        )
        neg_scored = _score_pair_side(
            materials=neg_materials,
            response_text=pair.negative,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_use_atomic_claims=nli_use_atomic_claims,
        )
        rows.append(
            {
                "stratum": pair.stratum,
                "pos_channels": _channels_from_score(pos_scored),
                "neg_channels": _channels_from_score(neg_scored),
            }
        )
    return rows


# ----------------------------------------------------------------------
# Grid sweep
# ----------------------------------------------------------------------


def _fuse(weights: Dict[str, float], channels: Dict[str, Optional[float]]) -> Optional[float]:
    return fuse_groundedness_v2(
        reverse_context_calibrated=channels.get("calibrated"),
        literal_guarded=channels.get("literal"),
        nli_aggregate=channels.get("nli"),
        semantic_entropy=channels.get("semantic_entropy"),
        structured_source_guarded=channels.get("structured"),
        weights=weights,
    )


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return (2 * tp) / float(denom)


def _evaluate_weights(
    weights: Dict[str, float],
    rows: Sequence[Dict[str, Any]],
    *,
    threshold: float,
) -> Dict[str, Any]:
    """Compute per-stratum F1 and the worst-case (minimum) F1."""

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for row in rows:
        stratum = row["stratum"]
        pos_score = _fuse(weights, row["pos_channels"])
        neg_score = _fuse(weights, row["neg_channels"])
        if pos_score is None or neg_score is None:
            continue
        if pos_score >= threshold:
            counts[stratum]["tp"] += 1
        else:
            counts[stratum]["fn"] += 1
        if neg_score >= threshold:
            counts[stratum]["fp"] += 1
        else:
            counts[stratum]["tn"] += 1

    per_stratum: Dict[str, float] = {}
    for stratum, c in counts.items():
        per_stratum[stratum] = _f1(c["tp"], c["fp"], c["fn"])
    min_f1 = min(per_stratum.values()) if per_stratum else 0.0
    macro_f1 = sum(per_stratum.values()) / float(len(per_stratum)) if per_stratum else 0.0
    return {
        "weights": weights,
        "per_stratum_f1": per_stratum,
        "min_f1": float(min_f1),
        "macro_f1": float(macro_f1),
        "counts": {k: dict(v) for k, v in counts.items()},
    }


def _build_weight_grid(
    *,
    steps: int,
    include_semantic_entropy: bool,
    include_structured: bool,
) -> List[Dict[str, float]]:
    """Deterministic grid over the simplex.

    Steps controls the resolution: ``steps=5`` generates values at
    ``[0, .25, .5, .75, 1.0]`` for each active channel. We prune points
    that sum to zero (trivial) and normalize each surviving point to
    sum to 1 so the grid stays on the unit simplex.
    """

    raw_values = [i / float(steps) for i in range(steps + 1)]
    channels: List[str] = ["calibrated", "literal", "nli"]
    if include_semantic_entropy:
        channels.append("semantic_entropy")
    if include_structured:
        channels.append("structured")

    grid: List[Dict[str, float]] = []
    seen = set()
    for combo in itertools.product(raw_values, repeat=len(channels)):
        total = sum(combo)
        if total <= 0:
            continue
        weights = {
            name: float(value) / float(total) for name, value in zip(channels, combo)
        }
        # De-duplicate by a coarsely rounded tuple so numerically
        # identical points do not flood the grid.
        fingerprint = tuple(round(weights[c], 3) for c in channels)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        # Pad with zero for channels not in the grid so downstream code
        # can still read dict keys uniformly.
        if not include_semantic_entropy:
            weights["semantic_entropy"] = 0.0
        if not include_structured:
            weights["structured"] = 0.0
        grid.append(weights)
    return grid


def sweep(
    rows: Sequence[Dict[str, Any]],
    *,
    steps: int,
    threshold: float,
    include_semantic_entropy: bool,
    include_structured: bool,
) -> Dict[str, Any]:
    """Run the grid sweep and return the ranked leaderboard."""

    grid = _build_weight_grid(
        steps=steps,
        include_semantic_entropy=include_semantic_entropy,
        include_structured=include_structured,
    )
    leaderboard = [_evaluate_weights(weights, rows, threshold=threshold) for weights in grid]
    leaderboard.sort(key=lambda item: (item["min_f1"], item["macro_f1"]), reverse=True)
    best = leaderboard[0] if leaderboard else None
    return {
        "threshold": float(threshold),
        "grid_points": len(grid),
        "pair_count": len(rows),
        "best": best,
        "top_10": leaderboard[:10],
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase J: sweep fusion weights")
    parser.add_argument("--model", default=os.environ.get("VOYAGER_GROUNDEDNESS_MODEL"))
    parser.add_argument("--pairs-per-stratum", type=int, default=40)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--enable-nli",
        action="store_true",
        help="Enable the NLI lane when scoring (otherwise the NLI channel stays at zero).",
    )
    parser.add_argument(
        "--nli-model",
        default=os.environ.get(
            "VOYAGER_GROUNDEDNESS_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.environ.get("VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL"),
    )
    parser.add_argument(
        "--concat-premises",
        dest="concat_premises",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-concat-premises",
        dest="concat_premises",
        action="store_false",
    )
    parser.add_argument(
        "--atomic-claims",
        dest="atomic_claims",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-atomic-claims",
        dest="atomic_claims",
        action="store_false",
    )
    parser.add_argument(
        "--include-semantic-entropy",
        action="store_true",
        help="Include the semantic_entropy axis in the grid sweep.",
    )
    parser.add_argument(
        "--include-structured",
        action="store_true",
        help="Include the structured_source axis in the grid sweep.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_HERE / "reports" / "fusion_weights.json",
    )
    args = parser.parse_args(argv)

    provider = _load_provider(args.model)
    pairs = build_minimal_pairs(pairs_per_stratum=args.pairs_per_stratum)
    null_bank = build_null_bank_embeddings(provider)
    nli_provider = _load_nli_provider(args.nli_model) if args.enable_nli else None
    nli_reranker = _load_reranker(args.reranker_model) if args.enable_nli else None

    rows = _collect_channel_samples(
        pairs=pairs,
        provider=provider,
        null_bank_embeddings=null_bank,
        nli_provider=nli_provider,
        nli_reranker=nli_reranker,
        nli_concat_premises=args.concat_premises,
        nli_use_atomic_claims=args.atomic_claims,
    )
    report = sweep(
        rows,
        steps=args.steps,
        threshold=args.threshold,
        include_semantic_entropy=args.include_semantic_entropy,
        include_structured=args.include_structured,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    best = report.get("best") or {}
    print(
        json.dumps(
            {
                "threshold": report["threshold"],
                "pair_count": report["pair_count"],
                "grid_points": report["grid_points"],
                "best_weights": best.get("weights"),
                "best_min_f1": best.get("min_f1"),
                "best_macro_f1": best.get("macro_f1"),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
