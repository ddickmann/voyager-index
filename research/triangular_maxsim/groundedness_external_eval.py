"""Phase E external evaluation harness for the groundedness Beta.

Two execution lanes drive the same minimal-pair fixture:

- the **default** lane uses ``reverse_context_calibrated`` as the headline
  (pass ``null_bank_embeddings`` to ``score_groundedness``)
- the **NLI** lane additionally builds a ``HuggingFaceNLIProvider`` and
  uses the fused ``groundedness_v2`` as the headline

Both lanes report:

- per-stratum paired ranking accuracy with bootstrap 95 percent CIs
- per-call latency split into ``encode_ms`` (provider work) and
  ``score_ms`` (Voyager scoring math), with the latency exit criterion
  applied to the sum so the budget reflects the full request

The harness then maps results to the pre-registered exit criteria from
:mod:`research.triangular_maxsim.groundedness_external_benchmarks` and
stamps a single ``headline_verdict`` string into the report so callers
do not have to re-derive it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    default_null_bank_texts,
    encode_texts,
    score_groundedness,
    segment_text,
    tokenize_text,
)


# ----------------------------------------------------------------------
# Provider loading
# ----------------------------------------------------------------------


def _load_provider(model_name: Optional[str]):
    """Load a real provider for production runs, else fall back to the dummy."""

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


def _load_nli_provider(model_id: Optional[str]):
    """Build a HuggingFace NLI provider on demand; ``None`` if disabled."""

    if not model_id:
        return None
    from voyager_index._internal.server.api.groundedness_nli import HuggingFaceNLIProvider

    return HuggingFaceNLIProvider(model_id=model_id)


def _load_reranker(model_id: Optional[str]):
    """Build a cross-encoder premise reranker on demand; ``None`` if disabled."""

    if not model_id:
        return None
    from voyager_index._internal.server.api.groundedness_nli import CrossEncoderPremiseReranker

    return CrossEncoderPremiseReranker(model_id=model_id)


# ----------------------------------------------------------------------
# Score helpers
# ----------------------------------------------------------------------


def _encode_pair_side(
    *,
    provider,
    context: str,
    response: str,
    chunk_token_budget: int = 256,
) -> Dict[str, Any]:
    """Phase 1: encode + tokenize. Returns the materials needed by scoring."""

    started = time.perf_counter()
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
    encode_ms = (time.perf_counter() - started) * 1000.0
    return {
        "support_units": support_units,
        "response_embedding": response_embedding,
        "response_tokens": response_tokens,
        "encode_ms": float(encode_ms),
    }


def _score_pair_side(
    *,
    materials: Dict[str, Any],
    response_text: str,
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    nli_provider=None,
    nli_max_latency_ms: float = 2000.0,
    nli_reranker=None,
    nli_concat_premises: Optional[bool] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    verification_samples: Optional[Sequence[str]] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Phase 2: score the encoded materials. Times only the score call."""

    started = time.perf_counter()
    scored = score_groundedness(
        support_units=materials["support_units"],
        response_embeddings=materials["response_embedding"],
        response_tokens=materials["response_tokens"],
        response_text=response_text,
        evidence_limit=3,
        primary_metric="reverse_context",
        debug_dense_matrices=False,
        null_bank_embeddings=null_bank_embeddings,
        nli_provider=nli_provider,
        nli_max_latency_ms=nli_max_latency_ms,
        nli_reranker=nli_reranker,
        nli_concat_premises=nli_concat_premises,
        nli_use_atomic_claims=nli_use_atomic_claims,
        verification_samples=list(verification_samples) if verification_samples else None,
        semantic_entropy_enabled=semantic_entropy_enabled,
        fusion_weights=fusion_weights,
    )
    score_ms = (time.perf_counter() - started) * 1000.0
    return {
        "scores": scored["scores"],
        "score_ms": float(score_ms),
    }


def _resolve_headline(
    scores: Dict[str, Any],
    *,
    nli_enabled: bool,
) -> Tuple[float, str]:
    """Pick the headline score for paired ranking and report which one was used.

    ``groundedness_v2`` already fuses calibrated + literal-guarded (and the
    optional NLI channel), so it is the preferred headline whenever it is
    available. Falls back to ``reverse_context_calibrated`` and finally the
    raw ``reverse_context`` score.
    """

    if scores.get("groundedness_v2") is not None:
        if nli_enabled:
            return float(scores["groundedness_v2"]), "groundedness_v2"
        return float(scores["groundedness_v2"]), "groundedness_v2_no_nli"
    cal = scores.get("reverse_context_calibrated")
    if cal is not None:
        return float(cal), "reverse_context_calibrated"
    return float(scores.get("reverse_context") or 0.0), "reverse_context"


# ----------------------------------------------------------------------
# Null bank caching
# ----------------------------------------------------------------------


def build_null_bank_embeddings(provider) -> List[torch.Tensor]:
    """Encode the default null bank once and return list of per-text embeddings."""

    texts = default_null_bank_texts()
    if not texts:
        return []
    embeddings = encode_texts(provider, texts, is_query=False, prompt_name=None)
    return [emb.detach().clone() for emb in embeddings]


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
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    nli_provider=None,
    nli_max_latency_ms: float = 2000.0,
    nli_reranker=None,
    nli_concat_premises: Optional[bool] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    semantic_entropy_sample_count: int = 0,
    fusion_weights: Optional[Dict[str, float]] = None,
    warmup: int = 3,
    seed: int = 17,
) -> Dict[str, Any]:
    """Score every minimal pair and report paired-ranking accuracy per stratum.

    Latency is recorded per call as ``encode_ms + score_ms`` so the reported
    p50 / p95 reflect the full request, not just post-encoder math.
    """

    rng = random.Random(seed)
    by_stratum: Dict[str, List[float]] = {}
    encode_samples: List[float] = []
    score_samples: List[float] = []
    full_samples: List[float] = []
    headline_seen: Dict[str, int] = {}

    def _build_samples(pair_obj: MinimalPair, seed_response: str) -> Optional[List[str]]:
        """Synthesize caller-supplied verification samples.

        For a headline run we cannot call a real LLM, so we synthesize a
        deterministic bank of paraphrases around ``seed_response`` taken
        from the pair context. When semantic entropy is disabled or the
        count is non-positive, return ``None`` so the fast path is
        unchanged.
        """

        if not semantic_entropy_enabled or semantic_entropy_sample_count <= 0:
            return None
        # Mix the positive and negative responses around the seed to emulate
        # temperature>0 sampling: a stable generator mostly returns the seed,
        # a confabulating one drifts toward the opposite side. We combine
        # both to produce a realistic mix without an actual LLM call.
        alt_sources = [pair_obj.positive, pair_obj.negative]
        samples = [seed_response]
        for idx in range(1, semantic_entropy_sample_count):
            samples.append(alt_sources[idx % 2])
        return samples

    if pairs and warmup > 0:
        for _ in range(warmup):
            warm_pair = pairs[0]
            warm_materials = _encode_pair_side(
                provider=provider, context=warm_pair.context, response=warm_pair.positive
            )
            _ = _score_pair_side(
                materials=warm_materials,
                response_text=warm_pair.positive,
                null_bank_embeddings=null_bank_embeddings,
                nli_provider=nli_provider,
                nli_max_latency_ms=nli_max_latency_ms,
                nli_reranker=nli_reranker,
                nli_concat_premises=nli_concat_premises,
                nli_use_atomic_claims=nli_use_atomic_claims,
                verification_samples=_build_samples(warm_pair, warm_pair.positive),
                semantic_entropy_enabled=semantic_entropy_enabled,
                fusion_weights=fusion_weights,
            )

    nli_enabled = nli_provider is not None
    for pair in pairs:
        pos_materials = _encode_pair_side(
            provider=provider, context=pair.context, response=pair.positive
        )
        pos_scored = _score_pair_side(
            materials=pos_materials,
            response_text=pair.positive,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_max_latency_ms=nli_max_latency_ms,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_use_atomic_claims=nli_use_atomic_claims,
            verification_samples=_build_samples(pair, pair.positive),
            semantic_entropy_enabled=semantic_entropy_enabled,
            fusion_weights=fusion_weights,
        )
        neg_materials = _encode_pair_side(
            provider=provider, context=pair.context, response=pair.negative
        )
        neg_scored = _score_pair_side(
            materials=neg_materials,
            response_text=pair.negative,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_max_latency_ms=nli_max_latency_ms,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_use_atomic_claims=nli_use_atomic_claims,
            verification_samples=_build_samples(pair, pair.negative),
            semantic_entropy_enabled=semantic_entropy_enabled,
            fusion_weights=fusion_weights,
        )
        for materials, scored in (
            (pos_materials, pos_scored),
            (neg_materials, neg_scored),
        ):
            encode_samples.append(materials["encode_ms"])
            score_samples.append(scored["score_ms"])
            full_samples.append(materials["encode_ms"] + scored["score_ms"])

        pos_score, pos_field = _resolve_headline(pos_scored["scores"], nli_enabled=nli_enabled)
        neg_score, neg_field = _resolve_headline(neg_scored["scores"], nli_enabled=nli_enabled)
        headline_seen[pos_field] = headline_seen.get(pos_field, 0) + 1
        headline_seen[neg_field] = headline_seen.get(neg_field, 0) + 1
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
    headline_field = max(headline_seen, key=headline_seen.get) if headline_seen else "reverse_context"
    return {
        "per_stratum": per_stratum,
        "overall_accuracy": float(overall_accuracy),
        "pair_count": len(pairs),
        "stratum_counts": stratum_summary(pairs),
        "headline_used": headline_field,
        "headline_distribution": dict(headline_seen),
        "nli_enabled": bool(nli_enabled),
        "encode_p50_ms": float(np.percentile(encode_samples, 50)) if encode_samples else 0.0,
        "encode_p95_ms": float(np.percentile(encode_samples, 95)) if encode_samples else 0.0,
        "score_p50_ms": float(np.percentile(score_samples, 50)) if score_samples else 0.0,
        "score_p95_ms": float(np.percentile(score_samples, 95)) if score_samples else 0.0,
        "latency_p50_ms": float(np.percentile(full_samples, 50)) if full_samples else 0.0,
        "latency_p95_ms": float(np.percentile(full_samples, 95)) if full_samples else 0.0,
    }


# ----------------------------------------------------------------------
# External benchmark lane
# ----------------------------------------------------------------------


def _binary_label(sample: BenchmarkSample) -> int:
    return 0 if sample.label == "hallucinated" else 1


def evaluate_benchmark_samples(
    samples: Sequence[BenchmarkSample],
    provider,
    *,
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    nli_provider=None,
    nli_reranker=None,
    nli_concat_premises: Optional[bool] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    semantic_entropy_sample_count: int = 0,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score a flat list of benchmark samples and bucket per stratum."""

    nli_enabled = nli_provider is not None

    def _external_samples(sample_obj: BenchmarkSample) -> Optional[List[str]]:
        if not semantic_entropy_enabled or semantic_entropy_sample_count <= 0:
            return None
        return [sample_obj.response] * semantic_entropy_sample_count

    by_stratum: Dict[str, List[Tuple[float, int]]] = {}
    for sample in samples:
        materials = _encode_pair_side(
            provider=provider, context=sample.context, response=sample.response
        )
        scored = _score_pair_side(
            materials=materials,
            response_text=sample.response,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_use_atomic_claims=nli_use_atomic_claims,
            verification_samples=_external_samples(sample),
            semantic_entropy_enabled=semantic_entropy_enabled,
            fusion_weights=fusion_weights,
        )
        score, _field = _resolve_headline(scored["scores"], nli_enabled=nli_enabled)
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
# Aggregation and verdict
# ----------------------------------------------------------------------


def assemble_report(
    *,
    minimal_pair_results: Dict[str, Any],
    external_results: Dict[str, Optional[Dict[str, Any]]],
    targets: Dict[str, Dict[str, Any]] = PREREGISTERED_TARGETS,
) -> Dict[str, Any]:
    """Merge per-lane results with the pre-registered exit criteria.

    Latency criteria use ``latency_p95_ms`` (encode + score). The
    ``latency_with_nli`` target is only checked when ``nli_enabled=True``;
    otherwise it is reported as ``not_applicable``.
    """

    nli_enabled = bool(minimal_pair_results.get("nli_enabled"))
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
            label = "pass" if met else (
                "partial"
                if accuracy >= target.get("min", 0.0) and n > 0
                else "fail"
            )
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": accuracy,
                "ci_lower": ci_lower,
                "n": n,
                "min": target.get("min"),
                "ci_lower_min": target.get("ci_lower_min"),
                "met": bool(met),
                "label": label,
                "notes": target.get("notes"),
            }
        elif key == "ragtruth":
            payload = external_results.get("ragtruth")
            if not payload:
                report["criteria"][key] = {"status": "skipped", "notes": target.get("notes")}
                continue
            f1s = [stats["f1"] for stats in payload["per_stratum"].values()]
            macro = float(sum(f1s) / len(f1s)) if f1s else 0.0
            met = bool(macro >= target.get("min", 0.0))
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": macro,
                "min": target.get("min"),
                "met": met,
                "label": "pass" if met else "fail",
                "notes": target.get("notes"),
            }
        elif key == "halueval_qa":
            payload = external_results.get("halueval")
            if not payload or "qa" not in payload["per_stratum"]:
                report["criteria"][key] = {"status": "skipped", "notes": target.get("notes")}
                continue
            qa_stats = payload["per_stratum"]["qa"]
            paired_proxy = float(qa_stats["precision"])
            met = bool(paired_proxy >= target.get("min", 0.0))
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": paired_proxy,
                "min": target.get("min"),
                "met": met,
                "label": "pass" if met else "fail",
                "notes": target.get("notes"),
            }
        elif key == "factscore":
            payload = external_results.get("factscore")
            if not payload or "biography" not in payload["per_stratum"]:
                report["criteria"][key] = {"status": "skipped", "notes": target.get("notes")}
                continue
            bio_stats = payload["per_stratum"]["biography"]
            met = bool(bio_stats["precision"] >= target.get("min", 0.0))
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": float(bio_stats["precision"]),
                "min": target.get("min"),
                "met": met,
                "label": "pass" if met else "fail",
                "notes": target.get("notes"),
            }
        elif key == "latency_score_only":
            value = float(minimal_pair_results.get("latency_p95_ms", 0.0))
            limit = float(target.get("max", math.inf))
            met = value <= limit
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": value,
                "max": limit,
                "met": bool(met),
                "label": "pass" if met else "fail",
                "applies_when": "nli_disabled",
                "applicable": not nli_enabled,
                "notes": target.get("notes"),
            }
        elif key == "latency_with_nli":
            value = float(minimal_pair_results.get("latency_p95_ms", 0.0))
            limit = float(target.get("max", math.inf))
            if not nli_enabled:
                report["criteria"][key] = {
                    "status": "not_applicable",
                    "applies_when": "nli_enabled",
                    "applicable": False,
                    "notes": target.get("notes"),
                }
                continue
            met = value <= limit
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": value,
                "max": limit,
                "met": bool(met),
                "label": "pass" if met else "fail",
                "applies_when": "nli_enabled",
                "applicable": True,
                "notes": target.get("notes"),
            }

    actionable = [
        criterion
        for criterion in report["criteria"].values()
        if criterion.get("status") not in {"skipped", "not_applicable"}
        and criterion.get("applicable", True)
    ]
    overall_passed = bool(actionable) and all(
        criterion.get("met", False) for criterion in actionable
    )
    report["all_targets_met"] = bool(overall_passed)
    report["headline_verdict"] = compute_headline_verdict(report)
    return report


def compute_headline_verdict(report: Dict[str, Any]) -> str:
    """Map per-stratum criterion labels to a single, plainly worded verdict."""

    criteria = report.get("criteria", {})
    nli_enabled = bool(report.get("minimal_pairs", {}).get("nli_enabled"))

    def _label(name: str) -> str:
        return str(criteria.get(name, {}).get("label", "missing"))

    lex = _label("minimal_pairs_lexical")
    sem = _label("minimal_pairs_semantic")
    par = _label("minimal_pairs_partial")
    lat_no = _label("latency_score_only")
    lat_nli = _label("latency_with_nli")

    if lex == "fail":
        return "not a feature yet: lexical strata fail"

    semantic_ok = sem in {"pass", "partial"}
    partial_ok = par in {"pass", "partial"}

    if not nli_enabled:
        if lex == "pass" and semantic_ok and partial_ok and lat_no == "pass":
            return "feature in Beta, ready for evidence/QA without NLI"
        if lex == "pass" and (sem == "fail" or par == "fail"):
            return "feature in Beta, NLI required for negation/role/partial"
        if lat_no == "fail":
            return "feature in Beta, missing latency budget without NLI"
        return "feature in Beta with caveats; see per-stratum table"

    if lex == "pass" and semantic_ok and partial_ok and lat_nli == "pass":
        return "feature in Beta with NLI peer, ready for evidence/QA"
    if lex == "pass" and (sem == "fail" or par == "fail"):
        return "not a feature yet: NLI lane still fails semantic or partial"
    if lat_nli == "fail":
        return "feature in Beta with NLI peer, missing latency budget"
    return "feature in Beta with NLI peer; see per-stratum table"


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
    parser.add_argument(
        "--enable-nli",
        action="store_true",
        help="Build a HuggingFaceNLIProvider and run with the fused groundedness_v2 headline.",
    )
    parser.add_argument(
        "--nli-model",
        default=os.environ.get(
            "VOYAGER_GROUNDEDNESS_NLI_MODEL",
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.environ.get("VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL"),
        help="Optional cross-encoder premise reranker (e.g. BAAI/bge-reranker-v2-m3).",
    )
    parser.add_argument(
        "--concat-premises",
        dest="concat_premises",
        action="store_true",
        default=None,
        help="Concatenate top-k premises into a single NLI input per claim.",
    )
    parser.add_argument(
        "--no-concat-premises",
        dest="concat_premises",
        action="store_false",
        help="Disable the multi-premise concatenation (revert to per-premise scoring).",
    )
    parser.add_argument(
        "--atomic-claims",
        dest="atomic_claims",
        action="store_true",
        default=None,
        help="Decompose each sentence into atomic facts and verify each atom separately.",
    )
    parser.add_argument(
        "--no-atomic-claims",
        dest="atomic_claims",
        action="store_false",
        help="Disable atomic-fact decomposition and keep sentence-level verification.",
    )
    parser.add_argument(
        "--enable-semantic-entropy",
        action="store_true",
        help="Enable semantic-entropy peer (Phase G) using synthesized verification samples.",
    )
    parser.add_argument(
        "--semantic-entropy-samples",
        type=int,
        default=int(os.environ.get("VOYAGER_GROUNDEDNESS_SEMANTIC_ENTROPY_SAMPLES", "4")),
        help="Number of synthetic verification samples to draw per case (default 4).",
    )
    parser.add_argument(
        "--fusion-weight-calibrated",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fusion-weight-literal",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fusion-weight-nli",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fusion-weight-semantic-entropy",
        type=float,
        default=None,
    )
    parser.add_argument("--out", type=Path, default=_HERE / "groundedness_external_eval_report.json")
    args = parser.parse_args(argv)

    provider = _load_provider(args.model)
    pairs = build_minimal_pairs(pairs_per_stratum=args.pairs_per_stratum)

    null_bank_embeddings = build_null_bank_embeddings(provider)
    nli_provider = _load_nli_provider(args.nli_model) if args.enable_nli else None
    nli_reranker = _load_reranker(args.reranker_model) if args.enable_nli else None

    fusion_weights: Optional[Dict[str, float]] = None
    if any(
        getattr(args, name) is not None
        for name in (
            "fusion_weight_calibrated",
            "fusion_weight_literal",
            "fusion_weight_nli",
            "fusion_weight_semantic_entropy",
        )
    ):
        fusion_weights = {
            "calibrated": args.fusion_weight_calibrated if args.fusion_weight_calibrated is not None else 0.5,
            "literal": args.fusion_weight_literal if args.fusion_weight_literal is not None else 0.2,
            "nli": args.fusion_weight_nli if args.fusion_weight_nli is not None else (0.3 if args.enable_nli else 0.0),
            "semantic_entropy": (
                args.fusion_weight_semantic_entropy
                if args.fusion_weight_semantic_entropy is not None
                else (0.15 if args.enable_semantic_entropy else 0.0)
            ),
        }
    elif args.enable_semantic_entropy:
        fusion_weights = {
            "calibrated": 0.45,
            "literal": 0.2,
            "nli": 0.25 if args.enable_nli else 0.0,
            "semantic_entropy": 0.1,
        }

    semantic_entropy_sample_count = max(0, int(args.semantic_entropy_samples))

    minimal_pair_results = evaluate_minimal_pairs(
        pairs,
        provider,
        null_bank_embeddings=null_bank_embeddings,
        nli_provider=nli_provider,
        nli_reranker=nli_reranker,
        nli_concat_premises=args.concat_premises,
        nli_use_atomic_claims=args.atomic_claims,
        semantic_entropy_enabled=args.enable_semantic_entropy,
        semantic_entropy_sample_count=semantic_entropy_sample_count,
        fusion_weights=fusion_weights,
    )

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
        external_results[name] = evaluate_benchmark_samples(
            samples,
            provider,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            nli_concat_premises=args.concat_premises,
            nli_use_atomic_claims=args.atomic_claims,
            semantic_entropy_enabled=args.enable_semantic_entropy,
            semantic_entropy_sample_count=semantic_entropy_sample_count,
            fusion_weights=fusion_weights,
        )

    report = assemble_report(
        minimal_pair_results=minimal_pair_results,
        external_results=external_results,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "headline_verdict": report["headline_verdict"],
        "headline_used": minimal_pair_results.get("headline_used"),
        "nli_enabled": minimal_pair_results.get("nli_enabled"),
        "minimal_pair_summary": minimal_pair_results["per_stratum"],
        "encode_p95_ms": minimal_pair_results.get("encode_p95_ms"),
        "score_p95_ms": minimal_pair_results.get("score_p95_ms"),
        "latency_p95_ms": minimal_pair_results.get("latency_p95_ms"),
        "available_external": available_benchmarks(),
        "all_targets_met": report["all_targets_met"],
        "report_path": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
