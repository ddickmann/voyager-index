"""External benchmark adapters for Phase E groundedness evaluation.

This module wraps the file-on-disk surfaces of three public benchmarks
without bundling the data itself. Each adapter:

- looks up its data directory from a ``VOYAGER_GROUNDEDNESS_*`` env var
- returns ``None`` when the data is unavailable so the eval lane can skip
  cleanly instead of raising
- emits a homogeneous list of :class:`BenchmarkSample` instances, each with a
  per-stratum label so downstream metrics can be reported per stratum

The adapters cover three benchmarks chosen for stratum coverage:

- **RAGTruth** (`https://github.com/ParticleMedia/RAGTruth`): span-level
  faithfulness with QA, summarization, and data-to-text strata.
- **HaluEval** (`https://github.com/RUCAIBox/HaluEval`): binary halluc /
  factual labels for QA, summarization, and dialogue.
- **FActScore** (`https://github.com/shmsw25/FActScore`): per-claim atomic
  precision over biographies.

Only the file shapes that are stable across releases are touched; richer
fields (e.g. atomic-fact lists for FActScore) are kept in the raw payload so
callers can extend metrics later without re-parsing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------


@dataclass
class BenchmarkSample:
    """One scoring task for the external evaluation lane."""

    benchmark: str
    sample_id: str
    stratum: str
    context: str
    response: str
    label: str
    query: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() else None


def _read_jsonl(path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj
            if limit is not None and idx + 1 >= limit:
                break


# ----------------------------------------------------------------------
# RAGTruth
# ----------------------------------------------------------------------


def load_ragtruth(
    *,
    max_samples_per_stratum: Optional[int] = None,
) -> Optional[List[BenchmarkSample]]:
    """Load RAGTruth samples if the data directory is available.

    Expected layout (mirrors the upstream repo's ``dataset`` folder):

    ``$VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR/{qa,summarization,data2text}/test.jsonl``

    Each line carries ``source_info``, ``response``, and ``labels`` with
    span-level annotations. We surface the response-level binary label as
    ``label`` so the standard binary lane works out of the box, and keep the
    full annotation under ``raw`` for span-level metrics.
    """

    base = _env_path("VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR")
    if base is None:
        return None
    samples: List[BenchmarkSample] = []
    for stratum in ("qa", "summarization", "data2text"):
        stratum_dir = base / stratum
        candidates = [stratum_dir / "test.jsonl", stratum_dir / "test.json"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        emitted = 0
        for obj in _read_jsonl(path):
            sample_id = str(obj.get("id") or obj.get("sample_id") or f"{stratum}-{emitted}")
            response = obj.get("response") or obj.get("answer") or ""
            context = obj.get("source_info") or obj.get("context") or ""
            if isinstance(context, dict):
                context = json.dumps(context, ensure_ascii=False)
            labels = obj.get("labels") or []
            label = "hallucinated" if labels else "faithful"
            samples.append(
                BenchmarkSample(
                    benchmark="ragtruth",
                    sample_id=sample_id,
                    stratum=stratum,
                    context=str(context),
                    response=str(response),
                    label=label,
                    query=obj.get("query"),
                    raw=obj,
                )
            )
            emitted += 1
            if max_samples_per_stratum is not None and emitted >= max_samples_per_stratum:
                break
    return samples or None


# ----------------------------------------------------------------------
# HaluEval
# ----------------------------------------------------------------------


def load_halueval(
    *,
    max_samples_per_stratum: Optional[int] = None,
) -> Optional[List[BenchmarkSample]]:
    """Load HaluEval QA / summarization / dialogue samples.

    Expected layout (mirrors the upstream repo):

    ``$VOYAGER_GROUNDEDNESS_HALUEVAL_DIR/{qa,summarization,dialogue}_data.jsonl``

    Each line typically contains paired ``right_answer`` and
    ``hallucinated_answer`` fields against a shared ``knowledge`` /
    ``document`` / ``dialogue_history``. We emit two samples per line (one
    positive, one negative) labelled ``faithful`` / ``hallucinated``.
    """

    base = _env_path("VOYAGER_GROUNDEDNESS_HALUEVAL_DIR")
    if base is None:
        return None
    samples: List[BenchmarkSample] = []
    for stratum, source_field, query_field, right_field, wrong_field in (
        ("qa", "knowledge", "question", "right_answer", "hallucinated_answer"),
        ("summarization", "document", None, "right_summary", "hallucinated_summary"),
        ("dialogue", "dialogue_history", None, "right_response", "hallucinated_response"),
    ):
        candidates = [
            base / "{stratum}_data.jsonl".format(stratum=stratum),
            base / stratum / "data.jsonl",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        emitted = 0
        for obj in _read_jsonl(path):
            context = obj.get(source_field) or ""
            query = obj.get(query_field) if query_field else None
            sample_id_base = str(obj.get("id") or obj.get("sample_id") or "{stratum}-{idx}".format(stratum=stratum, idx=emitted))
            right = obj.get(right_field)
            wrong = obj.get(wrong_field)
            if right:
                samples.append(
                    BenchmarkSample(
                        benchmark="halueval",
                        sample_id="{base}-pos".format(base=sample_id_base),
                        stratum=stratum,
                        context=str(context),
                        response=str(right),
                        label="faithful",
                        query=str(query) if query else None,
                        raw=obj,
                    )
                )
            if wrong:
                samples.append(
                    BenchmarkSample(
                        benchmark="halueval",
                        sample_id="{base}-neg".format(base=sample_id_base),
                        stratum=stratum,
                        context=str(context),
                        response=str(wrong),
                        label="hallucinated",
                        query=str(query) if query else None,
                        raw=obj,
                    )
                )
            emitted += 1
            if max_samples_per_stratum is not None and emitted >= max_samples_per_stratum:
                break
    return samples or None


# ----------------------------------------------------------------------
# FActScore
# ----------------------------------------------------------------------


def load_factscore(
    *,
    max_samples_per_stratum: Optional[int] = None,
) -> Optional[List[BenchmarkSample]]:
    """Load FActScore biographies if available.

    Expected layout:

    ``$VOYAGER_GROUNDEDNESS_FACTSCORE_DIR/biographies.jsonl``

    Each line carries ``topic`` / ``output`` / ``annotations`` with per-claim
    support labels. We aggregate per-claim labels into a single response-level
    proportion that downstream metrics can threshold.
    """

    base = _env_path("VOYAGER_GROUNDEDNESS_FACTSCORE_DIR")
    if base is None:
        return None
    candidates = [base / "biographies.jsonl", base / "factscore.jsonl"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    samples: List[BenchmarkSample] = []
    emitted = 0
    for obj in _read_jsonl(path):
        topic = str(obj.get("topic") or "biography")
        output = obj.get("output") or obj.get("response") or ""
        annotations = obj.get("annotations") or []
        if isinstance(output, list):
            output = " ".join(str(part) for part in output)
        supported = sum(1 for ann in annotations if ann.get("is_supported"))
        total = max(1, len(annotations))
        precision = float(supported) / float(total)
        label = "faithful" if precision >= 0.65 else "hallucinated"
        samples.append(
            BenchmarkSample(
                benchmark="factscore",
                sample_id="{topic}-{idx}".format(topic=topic, idx=emitted),
                stratum="biography",
                context=str(obj.get("context") or ""),
                response=str(output),
                label=label,
                query=str(topic),
                raw={"precision": precision, **obj},
            )
        )
        emitted += 1
        if max_samples_per_stratum is not None and emitted >= max_samples_per_stratum:
            break
    return samples or None


# ----------------------------------------------------------------------
# Pre-registered exit criteria
# ----------------------------------------------------------------------


PREREGISTERED_TARGETS: Dict[str, Dict[str, Any]] = {
    "ragtruth": {
        "metric": "span_f1_macro",
        "min": 0.55,
        "ci_lower_min": 0.50,
        "notes": "Macro span-F1 across qa/summarization/data2text strata.",
    },
    "halueval_qa": {
        "metric": "paired_accuracy",
        "min": 0.70,
        "ci_lower_min": 0.65,
        "notes": "Paired ranking accuracy on HaluEval QA pairs.",
    },
    "factscore": {
        "metric": "claim_precision",
        "min": 0.65,
        "ci_lower_min": 0.60,
        "notes": "Per-claim atomic precision on FActScore biographies.",
    },
    "minimal_pairs_lexical": {
        "metric": "paired_accuracy",
        "min": 0.80,
        "ci_lower_min": 0.75,
        "notes": "Pairs in entity_swap, date_swap, number_swap, unit_swap.",
        "strata": ("entity_swap", "date_swap", "number_swap", "unit_swap"),
    },
    "minimal_pairs_semantic": {
        "metric": "paired_accuracy",
        "min": 0.70,
        "ci_lower_min": 0.65,
        "notes": "Pairs in negation and role_swap (need NLI).",
        "strata": ("negation", "role_swap"),
    },
    "minimal_pairs_partial": {
        "metric": "paired_accuracy",
        "min": 0.65,
        "ci_lower_min": 0.60,
        "notes": "Partial-support pairs (one clause grounded, one not).",
        "strata": ("partial",),
    },
    "latency_score_only": {
        "metric": "p95_ms",
        "max": 100.0,
        "notes": "Score-only request budget without NLI.",
    },
    "latency_with_nli": {
        "metric": "p95_ms",
        "max": 250.0,
        "notes": "Full-request budget with NLI verifier enabled.",
    },
}


def available_benchmarks() -> List[str]:
    """Return the names of benchmarks whose data directories resolve to disk."""

    out: List[str] = []
    if _env_path("VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR") is not None:
        out.append("ragtruth")
    if _env_path("VOYAGER_GROUNDEDNESS_HALUEVAL_DIR") is not None:
        out.append("halueval")
    if _env_path("VOYAGER_GROUNDEDNESS_FACTSCORE_DIR") is not None:
        out.append("factscore")
    return out


__all__ = [
    "BenchmarkSample",
    "PREREGISTERED_TARGETS",
    "available_benchmarks",
    "load_factscore",
    "load_halueval",
    "load_ragtruth",
]
