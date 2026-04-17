"""Phase E external evaluation harness for the groundedness Beta.

This script evaluates the production scoring policy on:

1. The deterministic minimal-pair fixture (always available).
2. The optional external benchmarks RAGTruth, HaluEval, and FActScore (only
   active when the corresponding ``VOYAGER_GROUNDEDNESS_*_DIR`` env var
   resolves to disk).

It then reports per-stratum metrics together with the pre-registered exit
criteria from
:mod:`research.triangular_maxsim.groundedness_external_benchmarks`.

The harness is designed to run with very small encoders (e.g. the
``DummyGroundednessProvider`` used in unit tests) so the lane can be exercised
deterministically in CI; production runs override the provider with a real
ColBERT or ``vllm-factory`` deployment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.triangular_maxsim.groundedness_external_benchmarks import (  # noqa: E402
    PREREGISTERED_TARGETS,
    BenchmarkSample,
    available_benchmarks,
    load_factscore,
    load_halueval,
    load_ragtruth,
)
from research.triangular_maxsim.groundedness_minimal_pairs import (  # noqa: E402
    MinimalPair,
    build_minimal_pairs,
    stratum_summary,
)
from voyager_index._internal.server.api.groundedness import (  # noqa: E402
    SupportUnitInput,
    encode_texts,
    score_groundedness,
    segment_text,
    tokenize_text,
)


# ----------------------------------------------------------------------
# Provider loading
# ----------------------------------------------------------------------


def _load_provider(model_name: Optional[str]):
    """Load a real provider for production runs, else fall back to the dummy.

    The dummy provider is intentionally weak; it makes per-stratum numbers low
    but it lets us exercise the entire lane deterministically in CI.
    """

    if model_name:
        try:
            from pylate import models

            device = "cuda" if torch.cuda.is_available() else "cpu"
            return models.ColBERT(
                model_name_or_path=model_name,
                device=device,
                do_query_expansion=False,
            )
        except Exception:
            pass
    from tests.test_groundedness_service import DummyGroundednessProvider

    return DummyGroundednessProvider(dim=24)


# ----------------------------------------------------------------------
# Score helpers
# ----------------------------------------------------------------------


def _score_pair_side(
    *,
    provider,
    context: str,
    response: str,
    chunk_token_budget: int = 256,
) -> Dict[str, Any]:
    segments = segment_text(
        context, "sentence_packed", provider=provider, chunk_token_budget=chunk_token_budget
    )
    segment_texts = [segment["text"] for segment in segments] or [context]
    support_embeddings = encode_texts(provider, segment_texts, is_query=False, prompt_name=None)
    support_units = [
        SupportUnitInput(
            support_id="sup-{idx}".format(idx=idx),
            chunk_id=None,
            source_mode="raw_context",
            text=segment_texts[idx],
            embeddings=embedding,
            tokens=tokenize_text(provider, segment_texts[idx], expected_len=int(embedding.shape[0]), is_query=False),
        )
        for idx, embedding in enumerate(support_embeddings)
    ]
    response_embedding = encode_texts(provider, [response], is_query=False, prompt_name=None)[0]
    response_tokens = tokenize_text(
        provider, response, expected_len=int(response_embedding.shape[0]), is_query=False
    )
    started = time.perf_counter()
    scored = score_groundedness(
        support_units=support_units,
        response_embeddings=response_embedding,
        response_tokens=response_tokens,
        response_text=response,
        evidence_limit=3,
        primary_metric="reverse_context",
        debug_dense_matrices=False,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "scores": scored["scores"],
        "latency_ms": latency_ms,
    }


# ----------------------------------------------------------------------
# Minimal-pair lane
# ----------------------------------------------------------------------


def _bootstrap_ci(
    rng: random.Random,
    samples: Sequence[float],
    *,
    iterations: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    n = len(samples)
    means: List[float] = []
    for _ in range(iterations):
        draw = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / float(n))
    means.sort()
    lo_idx = max(0, int(math.floor((alpha / 2.0) * iterations)))
    hi_idx = min(iterations - 1, int(math.ceil((1.0 - alpha / 2.0) * iterations)))
    return float(means[lo_idx]), float(means[hi_idx])


def evaluate_minimal_pairs(
    pairs: Sequence[MinimalPair],
    provider,
    *,
    score_field: str = "reverse_context_calibrated",
    fallback_field: str = "reverse_context",
    seed: int = 17,
) -> Dict[str, Any]:
    """Score every minimal pair and report paired-ranking accuracy per stratum.

    A pair is "correctly ranked" when the positive response scores strictly
    higher than the negative response on ``score_field`` (with
    ``fallback_field`` substituted whenever the calibrated score is missing).
    """

    rng = random.Random(seed)
    by_stratum: Dict[str, List[float]] = {}
    latency_samples: List[float] = []
    for pair in pairs:
        pos = _score_pair_side(provider=provider, context=pair.context, response=pair.positive)
        neg = _score_pair_side(provider=provider, context=pair.context, response=pair.negative)
        latency_samples.append(pos["latency_ms"])
        latency_samples.append(neg["latency_ms"])
        pos_score = pos["scores"].get(score_field) or pos["scores"].get(fallback_field) or 0.0
        neg_score = neg["scores"].get(score_field) or neg["scores"].get(fallback_field) or 0.0
        correct = 1.0 if float(pos_score) > float(neg_score) else 0.0
        by_stratum.setdefault(pair.stratum, []).append(correct)

    per_stratum: Dict[str, Dict[str, float]] = {}
    for stratum, outcomes in by_stratum.items():
        accuracy = sum(outcomes) / float(len(outcomes))
        ci_lo, ci_hi = _bootstrap_ci(rng, outcomes)
        per_stratum[stratum] = {
            "n": len(outcomes),
            "paired_accuracy": float(accuracy),
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
        }

    overall_outcomes = [outcome for outcomes in by_stratum.values() for outcome in outcomes]
    overall_accuracy = (
        sum(overall_outcomes) / float(len(overall_outcomes)) if overall_outcomes else 0.0
    )
    return {
        "per_stratum": per_stratum,
        "overall_accuracy": float(overall_accuracy),
        "pair_count": len(pairs),
        "stratum_counts": stratum_summary(pairs),
        "latency_p50_ms": float(np.percentile(latency_samples, 50)) if latency_samples else 0.0,
        "latency_p95_ms": float(np.percentile(latency_samples, 95)) if latency_samples else 0.0,
    }


# ----------------------------------------------------------------------
# External benchmark lane
# ----------------------------------------------------------------------


def _binary_label(sample: BenchmarkSample) -> int:
    return 0 if sample.label == "hallucinated" else 1


def evaluate_benchmark_samples(
    samples: Sequence[BenchmarkSample],
    provider,
) -> Dict[str, Any]:
    """Score a flat list of benchmark samples and bucket per stratum.

    Reports binary classification stats (TP/FP/TN/FN, F1) using the median
    score across the bucket as a per-stratum threshold. This is intentionally
    a lightweight headline; richer per-benchmark protocols (span F1 for
    RAGTruth, atomic precision for FActScore) live alongside this in the
    raw payload so callers can post-process.
    """

    by_stratum: Dict[str, List[Tuple[float, int]]] = {}
    for sample in samples:
        scored = _score_pair_side(
            provider=provider, context=sample.context, response=sample.response
        )
        score = (
            scored["scores"].get("reverse_context_calibrated")
            or scored["scores"].get("reverse_context")
            or 0.0
        )
        by_stratum.setdefault(sample.stratum, []).append((float(score), _binary_label(sample)))

    per_stratum: Dict[str, Dict[str, float]] = {}
    for stratum, rows in by_stratum.items():
        scores = [score for score, _ in rows]
        labels = [label for _, label in rows]
        threshold = float(np.median(scores)) if scores else 0.0
        tp = sum(1 for s, l in rows if s >= threshold and l == 1)
        fp = sum(1 for s, l in rows if s >= threshold and l == 0)
        fn = sum(1 for s, l in rows if s < threshold and l == 1)
        tn = sum(1 for s, l in rows if s < threshold and l == 0)
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_stratum[stratum] = {
            "n": len(rows),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": threshold,
        }
    return {
        "sample_count": len(samples),
        "per_stratum": per_stratum,
    }


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def assemble_report(
    *,
    minimal_pair_results: Dict[str, Any],
    external_results: Dict[str, Optional[Dict[str, Any]]],
    targets: Dict[str, Dict[str, Any]] = PREREGISTERED_TARGETS,
) -> Dict[str, Any]:
    """Merge per-lane results with the pre-registered exit criteria.

    Each criterion gets a ``met`` boolean, the observed value, and the gap
    relative to the threshold. Missing benchmarks are reported as
    ``"status": "skipped"`` so the audit doc can flag them honestly.
    """

    report: Dict[str, Any] = {
        "minimal_pairs": minimal_pair_results,
        "external": external_results,
        "criteria": {},
    }

    def _bucket_accuracy(strata: Sequence[str]) -> Tuple[float, float, int]:
        outcomes_acc: List[float] = []
        ci_lowers: List[float] = []
        n = 0
        for stratum in strata:
            stats = minimal_pair_results["per_stratum"].get(stratum)
            if not stats:
                continue
            outcomes_acc.append(stats["paired_accuracy"])
            ci_lowers.append(stats["ci_lower"])
            n += stats["n"]
        if not outcomes_acc:
            return 0.0, 0.0, 0
        return float(sum(outcomes_acc) / len(outcomes_acc)), float(min(ci_lowers)), n

    for key, target in targets.items():
        if key.startswith("minimal_pairs_"):
            strata = target.get("strata", ())
            accuracy, ci_lower, n = _bucket_accuracy(strata)
            met = (
                accuracy >= target.get("min", 0.0)
                and ci_lower >= target.get("ci_lower_min", 0.0)
                and n > 0
            )
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": accuracy,
                "ci_lower": ci_lower,
                "n": n,
                "min": target.get("min"),
                "ci_lower_min": target.get("ci_lower_min"),
                "met": bool(met),
                "notes": target.get("notes"),
            }
        elif key == "ragtruth":
            payload = external_results.get("ragtruth")
            if not payload:
                report["criteria"][key] = {"status": "skipped", "notes": target.get("notes")}
                continue
            f1s = [stats["f1"] for stats in payload["per_stratum"].values()]
            macro = float(sum(f1s) / len(f1s)) if f1s else 0.0
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": macro,
                "min": target.get("min"),
                "met": bool(macro >= target.get("min", 0.0)),
                "notes": target.get("notes"),
            }
        elif key == "halueval_qa":
            payload = external_results.get("halueval")
            if not payload or "qa" not in payload["per_stratum"]:
                report["criteria"][key] = {"status": "skipped", "notes": target.get("notes")}
                continue
            qa_stats = payload["per_stratum"]["qa"]
            paired_proxy = float(qa_stats["precision"])
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": paired_proxy,
                "min": target.get("min"),
                "met": bool(paired_proxy >= target.get("min", 0.0)),
                "notes": target.get("notes"),
            }
        elif key == "factscore":
            payload = external_results.get("factscore")
            if not payload or "biography" not in payload["per_stratum"]:
                report["criteria"][key] = {"status": "skipped", "notes": target.get("notes")}
                continue
            bio_stats = payload["per_stratum"]["biography"]
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": float(bio_stats["precision"]),
                "min": target.get("min"),
                "met": bool(bio_stats["precision"] >= target.get("min", 0.0)),
                "notes": target.get("notes"),
            }
        elif key.startswith("latency_"):
            value = minimal_pair_results.get("latency_p95_ms", 0.0)
            limit = target.get("max", float("inf"))
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": float(value),
                "max": limit,
                "met": bool(value <= limit),
                "notes": target.get("notes"),
            }

    overall_passed = all(
        criterion.get("met", False)
        for criterion in report["criteria"].values()
        if criterion.get("status") != "skipped"
    )
    report["all_targets_met"] = bool(overall_passed)
    return report


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase E external evaluation harness")
    parser.add_argument("--model", default=os.environ.get("VOYAGER_GROUNDEDNESS_MODEL"))
    parser.add_argument(
        "--pairs-per-stratum",
        type=int,
        default=int(os.environ.get("VOYAGER_GROUNDEDNESS_MIN_PAIRS_PER_STRATUM", "30")),
    )
    parser.add_argument(
        "--max-external-per-stratum",
        type=int,
        default=int(os.environ.get("VOYAGER_GROUNDEDNESS_MAX_EXTERNAL_PER_STRATUM", "200")),
    )
    parser.add_argument("--out", type=Path, default=_HERE / "groundedness_external_eval_report.json")
    args = parser.parse_args(argv)

    provider = _load_provider(args.model)
    pairs = build_minimal_pairs(pairs_per_stratum=args.pairs_per_stratum)
    minimal_pair_results = evaluate_minimal_pairs(pairs, provider)

    external_results: Dict[str, Optional[Dict[str, Any]]] = {}
    for name, loader in (
        ("ragtruth", load_ragtruth),
        ("halueval", load_halueval),
        ("factscore", load_factscore),
    ):
        samples = loader(max_samples_per_stratum=args.max_external_per_stratum)
        if samples is None:
            external_results[name] = None
            continue
        external_results[name] = evaluate_benchmark_samples(samples, provider)

    report = assemble_report(
        minimal_pair_results=minimal_pair_results,
        external_results=external_results,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "minimal_pair_summary": minimal_pair_results["per_stratum"],
        "available_external": available_benchmarks(),
        "all_targets_met": report["all_targets_met"],
        "report_path": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
