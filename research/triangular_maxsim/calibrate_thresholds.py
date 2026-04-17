"""Offline threshold calibration for the groundedness tracker (Phase H).

Sweeps thresholds over the groundedness_v2 / reverse_context_calibrated
scores on a held-out set of minimal pairs and (optionally) external
benchmarks, selects the smallest threshold that achieves a
configurable precision floor per failure stratum, and writes the
result as a JSON constants file consumed by the runtime risk-band
classifier.

Design:

- Calibration runs fully offline. We re-use the evaluation harness's
  ``_encode_pair_side`` / ``_score_pair_side`` for honesty with the
  production scorer; the only difference is we record the raw
  positive/negative score per pair across a broader sampling budget
  than the headline harness uses.
- Thresholds are chosen *per stratum* so the risk bands can calibrate
  separately for easy (lexical) vs hard (role_swap / negation /
  partial) failure modes.
- Output schema (a superset of the runtime loader's contract):

      {
        "schema_version": 1,
        "headline": "groundedness_v2" | "reverse_context_calibrated",
        "precision_target": 0.75,
        "strata": {
          "<stratum>": {
            "green_min": float,
            "amber_min": float,
            "sample_count": int,
            "precision_at_green": float,
            "recall_at_green": float
          },
          ...
        }
      }

    ``green_min`` is the smallest headline score for which, on this
    stratum, positives outnumber negatives at the configured precision
    floor. ``amber_min`` is one standard step below ``green_min`` so
    the amber band is non-empty even when the headline is noisy; red
    is everything else.

Usage:

    python -m research.triangular_maxsim.calibrate_thresholds \
        --pairs-per-stratum 80 \
        --out voyager_index/_internal/server/api/groundedness_thresholds.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
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
    _resolve_headline,
    _score_pair_side,
    build_null_bank_embeddings,
)
from research.triangular_maxsim.groundedness_minimal_pairs import (  # noqa: E402
    MinimalPair,
    build_minimal_pairs,
)


def _headline_field_for(scores: Dict[str, Any], nli_enabled: bool) -> Tuple[float, str]:
    return _resolve_headline(scores, nli_enabled=nli_enabled)


def _precision_recall_at_threshold(
    positives: Sequence[float],
    negatives: Sequence[float],
    threshold: float,
) -> Tuple[float, float]:
    """Precision/recall treating positives (grounded) as label 1."""

    tp = sum(1 for s in positives if s >= threshold)
    fp = sum(1 for s in negatives if s >= threshold)
    fn = sum(1 for s in positives if s < threshold)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def _select_green_threshold(
    positives: Sequence[float],
    negatives: Sequence[float],
    *,
    precision_target: float,
    coarse_step: float = 0.005,
) -> Tuple[float, float, float]:
    """Smallest threshold satisfying the precision floor on this stratum.

    Returns ``(green_min, precision, recall)``. When no threshold meets
    the precision floor we fall back to the median positive score and
    flag the stratum via a zero precision so the runtime can display it
    as amber only.
    """

    if not positives or not negatives:
        return 1.0, 0.0, 0.0
    lo = min(min(positives), min(negatives))
    hi = max(max(positives), max(negatives))
    if lo == hi:
        return float(lo), 0.0, 0.0
    grid = []
    x = lo
    while x <= hi:
        grid.append(float(x))
        x += coarse_step
    grid.append(float(hi))
    candidates: List[Tuple[float, float, float]] = []
    for thr in grid:
        precision, recall = _precision_recall_at_threshold(positives, negatives, thr)
        if precision >= precision_target:
            candidates.append((thr, precision, recall))
    if candidates:
        candidates.sort(key=lambda item: (-item[2], item[0]))
        return candidates[0]
    median_pos = float(statistics.median(positives))
    precision, recall = _precision_recall_at_threshold(positives, negatives, median_pos)
    return median_pos, precision, recall


def calibrate(
    pairs: Sequence[MinimalPair],
    provider,
    *,
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    nli_provider=None,
    nli_reranker=None,
    nli_concat_premises: Optional[bool] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    precision_target: float = 0.75,
) -> Dict[str, Any]:
    """Run the calibration sweep and return the threshold dict."""

    nli_enabled = nli_provider is not None
    buckets: Dict[str, Dict[str, List[float]]] = {}
    headline_fields: Dict[str, int] = {}
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
        pos_score, pos_field = _headline_field_for(pos_scored["scores"], nli_enabled)
        neg_score, neg_field = _headline_field_for(neg_scored["scores"], nli_enabled)
        headline_fields[pos_field] = headline_fields.get(pos_field, 0) + 1
        headline_fields[neg_field] = headline_fields.get(neg_field, 0) + 1
        bucket = buckets.setdefault(pair.stratum, {"pos": [], "neg": []})
        bucket["pos"].append(float(pos_score))
        bucket["neg"].append(float(neg_score))

    strata_output: Dict[str, Any] = {}
    for stratum, samples in sorted(buckets.items()):
        positives = samples["pos"]
        negatives = samples["neg"]
        green_min, precision, recall = _select_green_threshold(
            positives, negatives, precision_target=precision_target
        )
        spread = max(1e-4, float(statistics.pstdev(positives + negatives)))
        amber_min = max(0.0, green_min - spread)
        strata_output[stratum] = {
            "green_min": float(green_min),
            "amber_min": float(amber_min),
            "sample_count": int(len(positives) + len(negatives)),
            "precision_at_green": float(precision),
            "recall_at_green": float(recall),
            "positive_median": float(statistics.median(positives)) if positives else 0.0,
            "negative_median": float(statistics.median(negatives)) if negatives else 0.0,
        }

    # Picks the most frequently used headline across all pairs so the
    # runtime can verify the thresholds target the expected score.
    headline = (
        max(headline_fields, key=headline_fields.get) if headline_fields else "reverse_context_calibrated"
    )
    return {
        "schema_version": 1,
        "headline": headline,
        "precision_target": float(precision_target),
        "nli_enabled": bool(nli_enabled),
        "pair_count": int(len(pairs)),
        "strata": strata_output,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase H: calibrate risk-band thresholds")
    parser.add_argument("--model", default=os.environ.get("VOYAGER_GROUNDEDNESS_MODEL"))
    parser.add_argument("--pairs-per-stratum", type=int, default=60)
    parser.add_argument("--precision-target", type=float, default=0.75)
    parser.add_argument(
        "--enable-nli",
        action="store_true",
        help="Include the NLI lane when calibrating (thresholds target groundedness_v2 instead of calibrated).",
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
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "voyager_index/_internal/server/api/groundedness_thresholds.json",
    )
    args = parser.parse_args(argv)

    provider = _load_provider(args.model)
    pairs = build_minimal_pairs(pairs_per_stratum=args.pairs_per_stratum)
    null_bank_embeddings = build_null_bank_embeddings(provider)
    nli_provider = _load_nli_provider(args.nli_model) if args.enable_nli else None
    nli_reranker = _load_reranker(args.reranker_model) if args.enable_nli else None

    report = calibrate(
        pairs,
        provider,
        null_bank_embeddings=null_bank_embeddings,
        nli_provider=nli_provider,
        nli_reranker=nli_reranker,
        nli_concat_premises=args.concat_premises,
        nli_use_atomic_claims=args.atomic_claims,
        precision_target=args.precision_target,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "headline": report["headline"],
                "precision_target": report["precision_target"],
                "nli_enabled": report["nli_enabled"],
                "pair_count": report["pair_count"],
                "strata": {
                    stratum: {
                        "green_min": info["green_min"],
                        "amber_min": info["amber_min"],
                        "precision_at_green": info["precision_at_green"],
                    }
                    for stratum, info in sorted(report["strata"].items())
                },
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
