"""Shared groundedness helpers for the reference API."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from voyager_index._internal.kernels.triton_triangular_maxsim import (
    grounded_coverage,
    naive_reverse_maxsim_qc,
    triangular_maxsim,
    weighted_groundedness,
)

_TOKEN_FALLBACK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.UNICODE)
_SPECIAL_TOKENS = {
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "<s>",
    "</s>",
    "<pad>",
    "<bos>",
    "<eos>",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
_MAX_DEBUG_MATRIX_ELEMENTS = 32_768


@dataclass
class SupportUnitInput:
    """Normalized support unit used by the groundedness scorer."""

    support_id: str
    text: str
    embeddings: torch.Tensor
    tokens: List[str]
    chunk_id: Optional[Any] = None
    source_mode: str = "chunk_ids"
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), p=2, dim=-1)


def _strip_marker(token: str) -> str:
    out = token
    for prefix in ("Ġ", "▁"):
        while out.startswith(prefix):
            out = out[len(prefix):]
    while out.startswith("##"):
        out = out[2:]
    return out


def token_weights(tokens: Sequence[str]) -> torch.Tensor:
    """Rule-based token weighting that downweights filler and punctuation."""

    weights: List[float] = []
    for token in tokens:
        if token in _SPECIAL_TOKENS:
            weights.append(0.0)
            continue
        stripped = _strip_marker(token).strip()
        if not stripped:
            weights.append(0.0)
            continue
        if all(ch in string.punctuation for ch in stripped):
            weights.append(0.0)
            continue
        if stripped.lower() in _STOPWORDS:
            weights.append(0.0)
            continue
        if any(ch.isdigit() for ch in stripped):
            weights.append(1.5)
            continue
        weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def _fallback_tokens(text: str) -> List[str]:
    return _TOKEN_FALLBACK_RE.findall(text)


def align_tokens(tokens: Sequence[str], expected_len: int) -> List[str]:
    base = list(tokens)
    if expected_len <= 0:
        return []
    if len(base) > expected_len:
        return base[:expected_len]
    if len(base) < expected_len:
        base.extend(f"tok_{idx}" for idx in range(len(base), expected_len))
    return base


def tokenize_text(provider: Any, text: str, *, expected_len: Optional[int] = None) -> List[str]:
    tokens: List[str] = []

    tokenize_method = getattr(provider, "tokenize", None)
    if callable(tokenize_method):
        try:
            tokens = list(tokenize_method(text))
        except Exception:
            tokens = []

    if not tokens and hasattr(provider, "tokenizer"):
        tokenizer = provider.tokenizer
        try:
            encoded = tokenizer(text, add_special_tokens=True)
            input_ids = encoded["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
            else:
                tokens = [str(item) for item in input_ids]
        except Exception:
            tokens = []

    if not tokens:
        tokens = _fallback_tokens(text)

    if expected_len is not None:
        tokens = align_tokens(tokens, expected_len)
    return tokens


def _to_tensor_list(output: Any, *, expected_items: int) -> List[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        if output.ndim == 1:
            return [output.reshape(1, -1).float()]
        if output.ndim == 2:
            return [output.float()]
        if output.ndim == 3:
            return [output[idx].float() for idx in range(output.shape[0])]
    if isinstance(output, np.ndarray):
        if output.ndim == 1:
            return [torch.from_numpy(output.reshape(1, -1)).float()]
        if output.ndim == 2:
            return [torch.from_numpy(output).float()]
        if output.ndim == 3:
            return [torch.from_numpy(output[idx]).float() for idx in range(output.shape[0])]
    if isinstance(output, list):
        if not output:
            return []
        if len(output) == expected_items and all(
            isinstance(item, (list, np.ndarray, torch.Tensor)) for item in output
        ):
            tensors = []
            for item in output:
                tensor = torch.as_tensor(item, dtype=torch.float32)
                if tensor.ndim == 1:
                    tensor = tensor.reshape(1, -1)
                tensors.append(tensor)
            return tensors
        if expected_items == 1 and output and not isinstance(output[0], list):
            return [torch.as_tensor(output, dtype=torch.float32).reshape(1, -1)]
        if expected_items == 1 and output and isinstance(output[0], list) and (
            not output[0] or not isinstance(output[0][0], list)
        ):
            return [torch.as_tensor(output, dtype=torch.float32)]
        tensors = [torch.as_tensor(item, dtype=torch.float32) for item in output]
        if len(tensors) == expected_items:
            return tensors
        if len(tensors) == 1 and expected_items == 1:
            return tensors
    raise TypeError("Unsupported encoder output shape for groundedness scoring")


def encode_texts(
    provider: Any,
    texts: Sequence[str],
    *,
    is_query: bool,
    prompt_name: Optional[str] = None,
) -> List[torch.Tensor]:
    """Encode texts into multi-vector tensors.

    Prefers batching when supported and falls back to one-by-one encoding.
    """

    items = list(texts)
    if not items:
        return []

    kwargs: Dict[str, Any] = {"is_query": is_query}
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name

    attempts = [dict(kwargs)]
    if "prompt_name" in kwargs:
        attempts.append({"is_query": is_query})
    attempts.append({})

    for candidate_kwargs in attempts:
        try:
            return _to_tensor_list(provider.encode(items, **candidate_kwargs), expected_items=len(items))
        except TypeError:
            continue
        except Exception:
            break

    encoded: List[torch.Tensor] = []
    for text in items:
        tensor: Optional[torch.Tensor] = None
        for candidate_kwargs in attempts:
            try:
                output = provider.encode([text], **candidate_kwargs)
                tensor = _to_tensor_list(output, expected_items=1)[0]
                break
            except TypeError:
                continue
            except Exception:
                tensor = None
                break
        if tensor is None:
            raise TypeError("Provider cannot encode text for groundedness scoring")
        encoded.append(tensor)
    return encoded


def segment_text(text: str, mode: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []

    if mode == "paragraph":
        spans: List[Dict[str, Any]] = []
        cursor = 0
        for part in re.split(r"\n\s*\n", text):
            start = text.find(part, cursor)
            if start < 0:
                continue
            end = start + len(part)
            cursor = end
            segment = part.strip()
            if segment:
                spans.append({"text": segment, "offset_start": start, "offset_end": end})
        return spans or [{"text": stripped, "offset_start": text.find(stripped), "offset_end": text.find(stripped) + len(stripped)}]

    spans = []
    for match in _SENTENCE_RE.finditer(text):
        segment = match.group(0).strip()
        if segment:
            spans.append({"text": segment, "offset_start": match.start(), "offset_end": match.end()})
    return spans or [{"text": stripped, "offset_start": text.find(stripped), "offset_end": text.find(stripped) + len(stripped)}]


def score_groundedness(
    *,
    support_units: Sequence[SupportUnitInput],
    response_embeddings: torch.Tensor,
    response_tokens: Sequence[str],
    query_embeddings: Optional[torch.Tensor] = None,
    query_tokens: Optional[Sequence[str]] = None,
    evidence_limit: int = 8,
    primary_metric: str = "reverse_context",
    debug_dense_matrices: bool = False,
) -> Dict[str, Any]:
    """Score response groundedness against support units.

    Returns a plain python payload suitable for Pydantic response models.
    """

    if not support_units:
        raise ValueError("At least one support unit is required")

    support_tensors = [unit.embeddings.float() for unit in support_units]
    support_tokens_nested = [align_tokens(unit.tokens, int(unit.embeddings.shape[0])) for unit in support_units]
    response_tokens_aligned = align_tokens(response_tokens, int(response_embeddings.shape[0]))

    C = torch.cat(support_tensors, dim=0)
    R = response_embeddings.float()
    C_norm = _normalize(C)
    R_norm = _normalize(R)
    sim_rc = R_norm @ C_norm.T
    rc_values, rc_indices = sim_rc.max(dim=1)

    weights = token_weights(response_tokens_aligned).to(rc_values.device, dtype=rc_values.dtype)
    reverse_context_score = weighted_groundedness(rc_values, weights)

    reverse_query_context_score: Optional[float] = None
    reverse_query_context_values: Optional[torch.Tensor] = None
    triangular_score: Optional[float] = None
    triangular_values: Optional[torch.Tensor] = None
    triangular_indices: Optional[torch.Tensor] = None
    echo_values: Optional[torch.Tensor] = None
    echo_mean: Optional[float] = None
    grounded_coverage_score: Optional[float] = None
    query_tokens_aligned: Optional[List[str]] = None
    query_token_rows: Optional[List[Dict[str, Any]]] = None
    sim_rq: Optional[torch.Tensor] = None
    sim_tri: Optional[torch.Tensor] = None

    if query_embeddings is not None:
        Q = query_embeddings.float()
        Q_norm = _normalize(Q)
        sim_rq = R_norm @ Q_norm.T
        query_tokens_aligned = align_tokens(query_tokens or [], int(Q.shape[0]))
        reverse_query_context_score, reverse_query_context_values = naive_reverse_maxsim_qc(
            Q_norm,
            C_norm,
            R_norm,
            weights=weights,
            normalize=False,
        )
        tri = triangular_maxsim(Q_norm, C_norm, R_norm, normalize=False)
        triangular_score = weighted_groundedness(tri.g, weights)
        triangular_values = tri.g
        triangular_indices = tri.jstar.to(torch.long)
        echo_values = tri.e
        echo_mean = weighted_groundedness(tri.e, weights)
        grounded_coverage_score = grounded_coverage(tri.u)
        query_token_rows = [
            {
                "index": idx,
                "token": query_tokens_aligned[idx],
                "coverage": float(tri.u[idx].item()),
            }
            for idx in range(len(query_tokens_aligned))
        ]
        if debug_dense_matrices:
            sim_tri = torch.minimum(sim_rc, tri.a[None, :])

    metric_name = primary_metric
    metric_values = rc_values
    metric_indices = rc_indices
    if primary_metric == "triangular":
        if triangular_values is None or triangular_indices is None:
            raise ValueError("triangular primary metric requires query embeddings")
        metric_values = triangular_values
        metric_indices = triangular_indices

    support_token_unit: List[int] = []
    support_token_local_idx: List[int] = []
    support_tokens_flat: List[str] = []
    support_units_payload: List[Dict[str, Any]] = []
    support_token_scores: List[List[float]] = []
    for unit_idx, unit in enumerate(support_units):
        tokens = support_tokens_nested[unit_idx]
        support_units_payload.append(
            {
                "index": unit_idx,
                "support_id": unit.support_id,
                "chunk_id": unit.chunk_id,
                "source_mode": unit.source_mode,
                "text": unit.text,
                "offset_start": unit.offset_start,
                "offset_end": unit.offset_end,
                "token_count": len(tokens),
                "tokens": tokens,
                "token_scores": [0.0] * len(tokens),
                "score": 0.0,
                "matched_response_tokens": 0,
            }
        )
        support_token_scores.append([0.0] * len(tokens))
        for token_idx, token in enumerate(tokens):
            support_token_unit.append(unit_idx)
            support_token_local_idx.append(token_idx)
            support_tokens_flat.append(token)

    support_score_numerators = [0.0 for _ in support_units]
    support_score_denominators = [0.0 for _ in support_units]

    response_token_rows: List[Dict[str, Any]] = []
    evidence_candidates: List[Dict[str, Any]] = []
    for idx, token in enumerate(response_tokens_aligned):
        support_flat_idx = int(metric_indices[idx].item()) if metric_indices.numel() else -1
        support_unit_idx = None
        support_token_idx = None
        support_token = None
        chunk_id = None
        if 0 <= support_flat_idx < len(support_token_unit):
            support_unit_idx = support_token_unit[support_flat_idx]
            support_token_idx = support_token_local_idx[support_flat_idx]
            support_token = support_tokens_flat[support_flat_idx]
            chunk_id = support_units_payload[support_unit_idx]["chunk_id"]
            score_value = float(metric_values[idx].item())
            support_token_scores[support_unit_idx][support_token_idx] = max(
                support_token_scores[support_unit_idx][support_token_idx],
                score_value,
            )
            support_score_numerators[support_unit_idx] += float(weights[idx].item()) * score_value
            support_score_denominators[support_unit_idx] += float(weights[idx].item())
            support_units_payload[support_unit_idx]["matched_response_tokens"] += 1
            if float(weights[idx].item()) > 0:
                evidence_candidates.append(
                    {
                        "response_token_index": idx,
                        "response_token": token,
                        "support_unit_index": support_unit_idx,
                        "support_token_index": support_token_idx,
                        "support_token": support_token,
                        "chunk_id": chunk_id,
                        "metric": metric_name,
                        "score": score_value,
                        "_rank": float(weights[idx].item()) * score_value,
                    }
                )
        response_token_rows.append(
            {
                "index": idx,
                "token": token,
                "weight": float(weights[idx].item()),
                "reverse_context": float(rc_values[idx].item()),
                "reverse_query_context": (
                    float(reverse_query_context_values[idx].item())
                    if reverse_query_context_values is not None
                    else None
                ),
                "triangular": float(triangular_values[idx].item()) if triangular_values is not None else None,
                "echo": float(echo_values[idx].item()) if echo_values is not None else None,
                "support_unit_index": support_unit_idx,
                "support_token_index": support_token_idx,
                "support_token": support_token,
                "chunk_id": chunk_id,
                "heatmap_score": float(metric_values[idx].item()),
            }
        )

    for unit_idx, payload in enumerate(support_units_payload):
        payload["token_scores"] = support_token_scores[unit_idx]
        denom = max(support_score_denominators[unit_idx], 1e-9)
        payload["score"] = float(support_score_numerators[unit_idx] / denom) if support_score_denominators[unit_idx] else 0.0

    evidence_candidates.sort(key=lambda item: item["_rank"], reverse=True)
    top_evidence = [
        {key: value for key, value in evidence.items() if key != "_rank"}
        for evidence in evidence_candidates[: max(1, evidence_limit)]
    ]

    debug_payload: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    if debug_dense_matrices:
        debug_payload = {}
        rc_elements = int(sim_rc.shape[0] * sim_rc.shape[1])
        if rc_elements <= _MAX_DEBUG_MATRIX_ELEMENTS:
            debug_payload["response_to_support"] = sim_rc.detach().cpu().tolist()
        else:
            warnings.append("response_to_support matrix omitted because it exceeds the debug size limit")
        if sim_rq is not None:
            rq_elements = int(sim_rq.shape[0] * sim_rq.shape[1])
            if rq_elements <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["response_to_query"] = sim_rq.detach().cpu().tolist()
            else:
                warnings.append("response_to_query matrix omitted because it exceeds the debug size limit")
        if sim_tri is not None:
            tri_elements = int(sim_tri.shape[0] * sim_tri.shape[1])
            if tri_elements <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["triangular_gated"] = sim_tri.detach().cpu().tolist()
            else:
                warnings.append("triangular_gated matrix omitted because it exceeds the debug size limit")
        if not debug_payload:
            debug_payload = None

    scores = {
        "primary_name": metric_name,
        "primary_score": float(triangular_score if metric_name == "triangular" else reverse_context_score),
        "reverse_context": float(reverse_context_score),
        "reverse_query_context": (
            float(reverse_query_context_score) if reverse_query_context_score is not None else None
        ),
        "triangular": float(triangular_score) if triangular_score is not None else None,
        "echo_mean": float(echo_mean) if echo_mean is not None else None,
        "grounded_coverage": float(grounded_coverage_score) if grounded_coverage_score is not None else None,
    }

    return {
        "scores": scores,
        "response_tokens": response_token_rows,
        "support_units": support_units_payload,
        "top_evidence": top_evidence,
        "query_tokens": query_token_rows,
        "debug": debug_payload,
        "warnings": warnings,
    }

