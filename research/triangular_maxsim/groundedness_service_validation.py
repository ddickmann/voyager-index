"""Validate the production groundedness scoring policy on the handcrafted suite.

This script reuses the cached embeddings from the triangular MaxSim research
cases but scores them through the production service policy:

* primary metric: reverse-context MaxSim
* optional diagnostics: reverse-query-context and triangular groundedness
* sparse evidence payloads derived from the same helper used by the API
"""

from __future__ import annotations

import argparse
import json
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

from research.triangular_maxsim.cases import CASES  # noqa: E402
from research.triangular_maxsim.dataset import DEFAULT_MODEL, load_or_build  # noqa: E402
from research.triangular_maxsim.groundedness_long_ambiguous_cases import LONG_AMBIGUOUS_CASES  # noqa: E402
from voyager_index._internal.server.api.groundedness import (  # noqa: E402
    SupportUnitInput,
    encode_texts,
    provider_token_limit,
    score_groundedness,
    score_groundedness_chunked,
    segment_text,
    tokenize_text,
)

GO_NO_GO = {
    "min_anchor_auroc_reverse_context": 0.90,
    "max_p95_latency_ms": 25.0,
}
CASE_BY_ID = {case.id: case for case in CASES}
DEFAULT_RAW_CONTEXT_CHUNK_TOKENS = 1024
PREVIOUS_LONG_AMBIGUOUS_REPORT = _HERE / "groundedness_service_validation__long_ambiguous_beta.json"


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUROC with ties handled by average rank."""

    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    ranks = [0.0] * len(pairs)
    idx = 0
    while idx < len(pairs):
        jdx = idx + 1
        while jdx < len(pairs) and pairs[jdx][0] == pairs[idx][0]:
            jdx += 1
        avg_rank = (idx + jdx + 1) / 2.0
        for rank_idx in range(idx, jdx):
            ranks[rank_idx] = avg_rank
        idx = jdx

    sum_pos = sum(rank for rank, (_score, label) in zip(ranks, pairs) if label == 1)
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _maybe_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    return auroc(scores, labels)


def _load_provider(model_name: str, use_prompts: bool):
    from pylate import models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    provider = models.ColBERT(
        model_name_or_path=model_name,
        device=device,
        do_query_expansion=False,
    )
    query_prompt_name = "query" if use_prompts else None
    document_prompt_name = "document" if use_prompts else None
    return provider, query_prompt_name, document_prompt_name


def _provider_token_count(provider, text: str) -> int:
    encoded = provider.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def _format_support_section(case_id: str) -> str:
    case = CASE_BY_ID[case_id]
    return f"[{case.id} | {case.source}] {case.context}"


def _build_long_context_sections(provider, case_spec) -> tuple[List[str], int]:
    core_sections = [_format_support_section(case_id) for case_id in case_spec.core_case_ids]
    distractor_sections = [_format_support_section(case_id) for case_id in case_spec.distractor_case_ids]
    if not distractor_sections:
        merged = "\n\n".join(core_sections)
        return core_sections, _provider_token_count(provider, merged)

    prefix: List[str] = []
    suffix: List[str] = []
    token_count = sum(_provider_token_count(provider, section) for section in core_sections)
    distractor_token_counts = [_provider_token_count(provider, section) for section in distractor_sections]
    idx = 0
    add_to_prefix = True
    safety_limit = max(len(distractor_sections) * 32, 64)

    while token_count < case_spec.token_budget and idx < safety_limit:
        section_idx = idx % len(distractor_sections)
        section = distractor_sections[section_idx]
        if add_to_prefix:
            prefix.append(section)
        else:
            suffix.append(section)
        add_to_prefix = not add_to_prefix
        idx += 1
        token_count += distractor_token_counts[section_idx]

    return prefix + core_sections + suffix, token_count


def _render_long_context(provider, case_spec) -> tuple[str, int]:
    section_texts, context_token_count = _build_long_context_sections(provider, case_spec)
    return "\n\n".join(section_texts), context_token_count


def _build_packed_support_units(
    *,
    case_id: str,
    raw_context: str,
    provider,
    document_prompt_name: Optional[str],
    chunk_token_budget: int,
) -> List[SupportUnitInput]:
    segments = segment_text(
        raw_context,
        "sentence_packed",
        provider=provider,
        chunk_token_budget=chunk_token_budget,
    )
    segment_texts = [segment["text"] for segment in segments]
    support_embeddings = encode_texts(
        provider,
        segment_texts,
        is_query=False,
        prompt_name=document_prompt_name,
    )
    return [
        SupportUnitInput(
            support_id=f"{case_id}-packed-{idx}",
            chunk_id=None,
            source_mode="raw_context",
            text=segment["text"],
            embeddings=embedding,
            tokens=tokenize_text(provider, segment["text"], expected_len=int(embedding.shape[0])),
            offset_start=int(segment["offset_start"]),
            offset_end=int(segment["offset_end"]),
        )
        for idx, (segment, embedding) in enumerate(zip(segments, support_embeddings))
    ]


def _select_hardest_previous_case_row() -> Optional[Dict[str, Any]]:
    if not PREVIOUS_LONG_AMBIGUOUS_REPORT.exists():
        return None
    payload = json.loads(PREVIOUS_LONG_AMBIGUOUS_REPORT.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        return None

    non_grounded = [row for row in rows if row.get("label") != "grounded"]
    candidates = non_grounded or rows
    return max(
        candidates,
        key=lambda row: (
            float(row["scores"]["reverse_context"]),
            int(row.get("support_unit_count", 0)),
        ),
    )


def _verify_chunked_merge(
    *,
    support_units: Sequence[SupportUnitInput],
    response_embeddings: torch.Tensor,
    response_tokens: Sequence[str],
    query_embeddings: Optional[torch.Tensor],
    query_tokens: Optional[Sequence[str]],
) -> Dict[str, Any]:
    chunked = score_groundedness_chunked(
        support_batches=[[unit] for unit in support_units],
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        query_embeddings=query_embeddings,
        query_tokens=query_tokens,
        evidence_limit=3,
        primary_metric="reverse_context",
        debug_dense_matrices=False,
    )
    reference = score_groundedness(
        support_units=support_units,
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        query_embeddings=query_embeddings,
        query_tokens=query_tokens,
        evidence_limit=3,
        primary_metric="reverse_context",
        debug_dense_matrices=False,
    )

    token_diffs = [
        abs(chunked_row["reverse_context"] - reference_row["reverse_context"])
        for chunked_row, reference_row in zip(chunked["response_tokens"], reference["response_tokens"])
    ]
    evidence_matches = all(
        chunked_row["support_unit_index"] == reference_row["support_unit_index"]
        and chunked_row["support_token_index"] == reference_row["support_token_index"]
        for chunked_row, reference_row in zip(chunked["response_tokens"], reference["response_tokens"])
    )
    return {
        "per_token_max_abs_diff": float(max(token_diffs) if token_diffs else 0.0),
        "scalar_abs_diff": float(
            abs(chunked["scores"]["reverse_context"] - reference["scores"]["reverse_context"])
        ),
        "evidence_mappings_match": evidence_matches,
        "top_evidence_matches": [
            (
                evidence["response_token_index"],
                evidence["support_unit_index"],
                evidence["support_token_index"],
            )
            for evidence in chunked["top_evidence"]
        ]
        == [
            (
                evidence["response_token_index"],
                evidence["support_unit_index"],
                evidence["support_token_index"],
            )
            for evidence in reference["top_evidence"]
        ],
    }


def score_embedded_case(embedded_case) -> Dict[str, Any]:
    support_units = [
        SupportUnitInput(
            support_id=embedded_case.case.id,
            chunk_id=embedded_case.case.id,
            source_mode="chunk_ids",
            text=embedded_case.case.context,
            embeddings=embedded_case.C,
            tokens=embedded_case.c_tokens,
        )
    ]
    start = time.perf_counter()
    result = score_groundedness(
        support_units=support_units,
        response_embeddings=embedded_case.R,
        response_tokens=embedded_case.r_tokens,
        query_embeddings=embedded_case.Q,
        query_tokens=embedded_case.q_tokens,
        evidence_limit=3,
        primary_metric="reverse_context",
        debug_dense_matrices=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "id": embedded_case.case.id,
        "source": embedded_case.case.source,
        "label": embedded_case.case.label,
        "subcategory": embedded_case.case.subcategory,
        "query": embedded_case.case.query,
        "response": embedded_case.case.response,
        "scores": result["scores"],
        "top_evidence": result["top_evidence"],
        "warnings": result["warnings"],
        "latency_ms": latency_ms,
        "notes": embedded_case.case.notes,
        "context_token_count": int(embedded_case.C.shape[0]),
        "support_unit_count": 1,
    }


def _score_packed_long_context_case(
    case_spec,
    provider,
    *,
    query_prompt_name: Optional[str],
    document_prompt_name: Optional[str],
    chunk_token_budget: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_context, context_token_count = _render_long_context(provider, case_spec)
    support_units = _build_packed_support_units(
        case_id=case_spec.id,
        raw_context=raw_context,
        provider=provider,
        document_prompt_name=document_prompt_name,
        chunk_token_budget=chunk_token_budget,
    )

    query_embeddings = encode_texts(
        provider,
        [case_spec.query],
        is_query=True,
        prompt_name=query_prompt_name,
    )[0]
    response_embeddings = encode_texts(
        provider,
        [case_spec.response],
        is_query=False,
        prompt_name=document_prompt_name,
    )[0]
    query_tokens = tokenize_text(provider, case_spec.query, expected_len=int(query_embeddings.shape[0]))
    response_tokens = tokenize_text(provider, case_spec.response, expected_len=int(response_embeddings.shape[0]))

    start = time.perf_counter()
    result = score_groundedness_chunked(
        support_batches=[[unit] for unit in support_units],
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        query_embeddings=query_embeddings,
        query_tokens=query_tokens,
        evidence_limit=3,
        primary_metric="reverse_context",
        debug_dense_matrices=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    row = {
        "id": case_spec.id,
        "source": ", ".join(CASE_BY_ID[case_id].source for case_id in case_spec.core_case_ids),
        "label": case_spec.label,
        "subcategory": case_spec.subcategory,
        "query": case_spec.query,
        "response": case_spec.response,
        "scores": result["scores"],
        "top_evidence": result["top_evidence"],
        "warnings": result["warnings"],
        "latency_ms": latency_ms,
        "notes": case_spec.notes,
        "context_token_count": context_token_count,
        "support_unit_count": len(support_units),
        "raw_context_chunk_tokens": chunk_token_budget,
    }
    details = {
        "support_units": support_units,
        "query_embeddings": query_embeddings,
        "query_tokens": query_tokens,
        "response_embeddings": response_embeddings,
        "response_tokens": response_tokens,
    }
    return row, details


def score_long_context_case(
    case_spec,
    provider,
    *,
    query_prompt_name: Optional[str],
    document_prompt_name: Optional[str],
    chunk_token_budget: int,
) -> Dict[str, Any]:
    row, _details = _score_packed_long_context_case(
        case_spec,
        provider,
        query_prompt_name=query_prompt_name,
        document_prompt_name=document_prompt_name,
        chunk_token_budget=chunk_token_budget,
    )
    return row


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    anchors = [row for row in rows if row["label"] in {"grounded", "ungrounded"}]
    labels = [1 if row["label"] == "grounded" else 0 for row in anchors]

    metrics = {
        "anchor_count": len(anchors),
        "AUROC_reverse_context": _maybe_auroc([row["scores"]["reverse_context"] for row in anchors], labels),
        "AUROC_reverse_query_context": _maybe_auroc(
            [row["scores"]["reverse_query_context"] for row in anchors],
            labels,
        ),
        "AUROC_triangular": _maybe_auroc([row["scores"]["triangular"] for row in anchors], labels),
        "latency_p50_ms": float(np.percentile([row["latency_ms"] for row in rows], 50)),
        "latency_p95_ms": float(np.percentile([row["latency_ms"] for row in rows], 95)),
        "mean_context_tokens": float(np.mean([row["context_token_count"] for row in rows])),
        "max_context_tokens": int(max(row["context_token_count"] for row in rows)),
        "mean_support_units": float(np.mean([row["support_unit_count"] for row in rows])),
    }
    metrics["gate_anchor_auroc_pass"] = (
        metrics["AUROC_reverse_context"] is not None
        and metrics["AUROC_reverse_context"] >= GO_NO_GO["min_anchor_auroc_reverse_context"]
    )
    metrics["gate_latency_pass"] = metrics["latency_p95_ms"] <= GO_NO_GO["max_p95_latency_ms"]
    metrics["go_for_user_facing"] = bool(metrics["gate_anchor_auroc_pass"] and metrics["gate_latency_pass"])

    by_subcategory: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        subcategory = row["subcategory"] or row["label"]
        bucket = by_subcategory.setdefault(
            subcategory,
            {
                "count": 0,
                "mean_reverse_context": 0.0,
                "mean_triangular": 0.0,
                "mean_context_tokens": 0.0,
                "examples": [],
            },
        )
        bucket["count"] += 1
        bucket["mean_reverse_context"] += float(row["scores"]["reverse_context"])
        bucket["mean_triangular"] += float(row["scores"]["triangular"] or 0.0)
        bucket["mean_context_tokens"] += float(row["context_token_count"])
        if len(bucket["examples"]) < 2:
            bucket["examples"].append(
                {
                    "id": row["id"],
                    "response": row["response"],
                    "top_evidence": row["top_evidence"],
                    "notes": row["notes"],
                }
            )
    for bucket in by_subcategory.values():
        bucket["mean_reverse_context"] /= max(bucket["count"], 1)
        bucket["mean_triangular"] /= max(bucket["count"], 1)
        bucket["mean_context_tokens"] /= max(bucket["count"], 1)

    return {
        "metrics": metrics,
        "by_subcategory": by_subcategory,
        "go_no_go": GO_NO_GO,
    }


def _fmt_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_markdown(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    *,
    model_name: str,
    use_prompts: bool,
    profile: str,
    raw_context_chunk_tokens: int,
    encoder_token_limit: Optional[int],
    hardest_case_comparison: Optional[Dict[str, Any]] = None,
) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Groundedness Service Validation",
        "",
        f"- profile: `{profile}`",
        f"- model: `{model_name}`",
        f"- prompts: `{use_prompts}`",
        f"- packed raw_context chunk tokens: `{raw_context_chunk_tokens}`",
        f"- encoder token limit: `{encoder_token_limit}`",
        f"- anchor count: `{metrics['anchor_count']}`",
        f"- anchor AUROC (reverse_context): `{_fmt_metric(metrics['AUROC_reverse_context'])}`",
        f"- anchor AUROC (reverse_query_context): `{_fmt_metric(metrics['AUROC_reverse_query_context'])}`",
        f"- anchor AUROC (triangular): `{_fmt_metric(metrics['AUROC_triangular'])}`",
        f"- latency p50/p95 ms: `{metrics['latency_p50_ms']:.2f}` / `{metrics['latency_p95_ms']:.2f}`",
        f"- mean/max context tokens: `{metrics['mean_context_tokens']:.0f}` / `{metrics['max_context_tokens']}`",
        f"- mean packed support units: `{metrics['mean_support_units']:.1f}`",
        f"- user-facing go/no-go: `{metrics['go_for_user_facing']}`",
        "",
        "## Difficulty Summary",
        "",
        "| bucket | count | mean reverse_context | mean triangular | mean context tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, bucket in sorted(summary["by_subcategory"].items()):
        lines.append(
            f"| {name} | {bucket['count']} | {bucket['mean_reverse_context']:.4f} | {bucket['mean_triangular']:.4f} | {bucket['mean_context_tokens']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Hardest Previous Case Rerun",
            "",
        ]
    )
    if encoder_token_limit is not None and raw_context_chunk_tokens > encoder_token_limit:
        lines.extend(
            [
                "- note: the requested packed budget exceeds this encoder's tokenizer limit, so support windows can be truncated during encoding on this model.",
                "",
            ]
        )
    if hardest_case_comparison is None:
        lines.append("- No previous long-ambiguous baseline report was available for comparison.")
        lines.append("")
    else:
        previous_row = hardest_case_comparison["previous"]
        rerun_row = hardest_case_comparison["rerun"]
        verification = hardest_case_comparison["verification"]
        lines.extend(
            [
                f"- selected case: `{previous_row['id']}` ({previous_row['label']})",
                f"- rationale: highest previous non-grounded reverse_context score (`{previous_row['scores']['reverse_context']:.4f}`) in the earlier long-context report",
                f"- before: reverse_context `{previous_row['scores']['reverse_context']:.4f}`, support_units `{previous_row['support_unit_count']}`",
                f"- after packed-{raw_context_chunk_tokens}: reverse_context `{rerun_row['scores']['reverse_context']:.4f}`, support_units `{rerun_row['support_unit_count']}`",
                f"- verification per-token max abs diff: `{verification['per_token_max_abs_diff']:.8f}`",
                f"- verification scalar abs diff: `{verification['scalar_abs_diff']:.8f}`",
                f"- evidence mapping exact match: `{verification['evidence_mappings_match']}`",
                f"- top-evidence exact match: `{verification['top_evidence_matches']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Example Evidence",
            "",
        ]
    )
    for name, bucket in sorted(summary["by_subcategory"].items()):
        lines.append(f"### {name}")
        for example in bucket["examples"]:
            lines.append(f"- `{example['id']}`: {example['response']}")
            lines.append(f"  - notes: {example['notes']}")
            for evidence in example["top_evidence"]:
                lines.append(
                    f"  - token `{evidence['response_token']}` -> `{evidence['support_token']}` score `{evidence['score']:.4f}`"
                )
        lines.append("")

    lines.extend(
        [
            "## Per-Case Scores",
            "",
            "| id | label | subcategory | context_tokens | support_units | reverse_context | reverse_query_context | triangular | latency_ms |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {id} | {label} | {subcategory} | {context_tokens} | {support_units} | {rc:.4f} | {qc:.4f} | {tri:.4f} | {latency:.2f} |".format(
                id=row["id"],
                label=row["label"],
                subcategory=row["subcategory"] or "-",
                context_tokens=row["context_token_count"],
                support_units=row["support_unit_count"],
                rc=row["scores"]["reverse_context"],
                qc=row["scores"]["reverse_query_context"],
                tri=row["scores"]["triangular"],
                latency=row["latency_ms"],
            )
        )

    return "\n".join(lines) + "\n"


def run(
    model_name: str,
    use_prompts: bool,
    rebuild: bool,
    tag: str,
    profile: str,
    raw_context_chunk_tokens: int,
) -> Tuple[Dict[str, Any], Path, Path]:
    hardest_case_comparison: Optional[Dict[str, Any]] = None
    encoder_token_limit: Optional[int] = None
    if profile == "default":
        embedded = load_or_build(rebuild=rebuild, model_name=model_name, use_prompts=use_prompts)
        if embedded:
            score_embedded_case(embedded[0])
        rows = [score_embedded_case(case) for case in embedded]
    elif profile == "long_ambiguous":
        provider, query_prompt_name, document_prompt_name = _load_provider(model_name, use_prompts)
        encoder_token_limit = provider_token_limit(provider)
        if LONG_AMBIGUOUS_CASES:
            score_long_context_case(
                LONG_AMBIGUOUS_CASES[0],
                provider,
                query_prompt_name=query_prompt_name,
                document_prompt_name=document_prompt_name,
                chunk_token_budget=raw_context_chunk_tokens,
            )
        rows = [
            score_long_context_case(
                case_spec,
                provider,
                query_prompt_name=query_prompt_name,
                document_prompt_name=document_prompt_name,
                chunk_token_budget=raw_context_chunk_tokens,
            )
            for case_spec in LONG_AMBIGUOUS_CASES
        ]
        previous_hardest = _select_hardest_previous_case_row()
        if previous_hardest is not None:
            hardest_case_id = previous_hardest["id"]
            matching_case = next((case for case in LONG_AMBIGUOUS_CASES if case.id == hardest_case_id), None)
            if matching_case is not None:
                rerun_row, details = _score_packed_long_context_case(
                    matching_case,
                    provider,
                    query_prompt_name=query_prompt_name,
                    document_prompt_name=document_prompt_name,
                    chunk_token_budget=raw_context_chunk_tokens,
                )
                hardest_case_comparison = {
                    "previous": previous_hardest,
                    "rerun": rerun_row,
                    "verification": _verify_chunked_merge(
                        support_units=details["support_units"],
                        response_embeddings=details["response_embeddings"],
                        response_tokens=details["response_tokens"],
                        query_embeddings=details["query_embeddings"],
                        query_tokens=details["query_tokens"],
                    ),
                }
    else:
        raise ValueError(f"Unknown validation profile: {profile}")

    summary = summarize(rows)
    payload = {
        "profile": profile,
        "model": model_name,
        "use_prompts": use_prompts,
        "raw_context_chunk_tokens": raw_context_chunk_tokens,
        "encoder_token_limit": encoder_token_limit,
        "summary": summary,
        "rows": rows,
        "hardest_case_comparison": hardest_case_comparison,
    }
    json_path = _HERE / f"groundedness_service_validation__{tag}.json"
    md_path = _HERE / f"groundedness_service_validation__{tag}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            summary,
            rows,
            model_name=model_name,
            use_prompts=use_prompts,
            profile=profile,
            raw_context_chunk_tokens=raw_context_chunk_tokens,
            encoder_token_limit=encoder_token_limit,
            hardest_case_comparison=hardest_case_comparison,
        ),
        encoding="utf-8",
    )
    return payload, json_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--profile", default="default", choices=["default", "long_ambiguous"])
    parser.add_argument("--tag", default="default")
    parser.add_argument("--raw-context-chunk-tokens", type=int, default=DEFAULT_RAW_CONTEXT_CHUNK_TOKENS)
    args = parser.parse_args()

    payload, json_path, md_path = run(
        model_name=args.model,
        use_prompts=args.prompts,
        rebuild=args.rebuild,
        tag=args.tag,
        profile=args.profile,
        raw_context_chunk_tokens=args.raw_context_chunk_tokens,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(payload["summary"]["metrics"], indent=2))
