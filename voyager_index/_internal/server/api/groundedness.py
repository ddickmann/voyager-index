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


def _tokens_from_provider_tokenize(output: Any) -> List[str]:
    if isinstance(output, dict):
        return []
    if isinstance(output, str):
        return [output]
    if isinstance(output, (list, tuple)):
        if output and all(isinstance(item, str) for item in output):
            return list(output)
        return []
    try:
        items = list(output)
    except TypeError:
        return []
    if items and all(isinstance(item, str) for item in items):
        return items
    return []


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
            tokens = _tokens_from_provider_tokenize(tokenize_method(text))
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


def count_text_tokens(provider: Any, text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0

    if hasattr(provider, "tokenizer"):
        tokenizer = provider.tokenizer
        try:
            encoded = tokenizer(text, add_special_tokens=False, truncation=False)
            input_ids = encoded["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return len(input_ids)
        except Exception:
            pass

    tokenize_method = getattr(provider, "tokenize", None)
    if callable(tokenize_method):
        try:
            tokens = _tokens_from_provider_tokenize(tokenize_method(text))
            if tokens:
                return len(tokens)
        except Exception:
            pass

    return len(_fallback_tokens(text))


def provider_token_limit(provider: Any) -> Optional[int]:
    for attr in ("doc_maxlen", "max_length", "model_max_length"):
        value = getattr(provider, attr, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value

    tokenizer = getattr(provider, "tokenizer", None)
    if tokenizer is None:
        return None
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 1_000_000:
        return value
    return None


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


def _fallback_segment(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    start = text.find(stripped)
    return [{"text": stripped, "offset_start": start, "offset_end": start + len(stripped)}]


def _trimmed_span(fragment: str, start: int, end: int) -> Optional[Dict[str, Any]]:
    stripped = fragment.strip()
    if not stripped:
        return None
    leading = len(fragment) - len(fragment.lstrip())
    trailing = len(fragment) - len(fragment.rstrip())
    return {
        "text": stripped,
        "offset_start": start + leading,
        "offset_end": end - trailing,
    }


def _paragraph_spans(text: str) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    cursor = 0
    for part in re.split(r"\n\s*\n", text):
        start = text.find(part, cursor)
        if start < 0:
            continue
        end = start + len(part)
        cursor = end
        span = _trimmed_span(part, start, end)
        if span is not None:
            spans.append(span)
    return spans or _fallback_segment(text)


def _sentence_spans(text: str) -> List[Dict[str, Any]]:
    spans = []
    for match in _SENTENCE_RE.finditer(text):
        span = _trimmed_span(match.group(0), match.start(), match.end())
        if span is not None:
            spans.append(span)
    return spans or _fallback_segment(text)


def _pack_sentence_spans(
    text: str,
    spans: Sequence[Dict[str, Any]],
    *,
    provider: Any,
    chunk_token_budget: int,
) -> List[Dict[str, Any]]:
    if provider is None:
        raise ValueError("sentence_packed segmentation requires a provider for token counting")
    if chunk_token_budget <= 0:
        raise ValueError("chunk_token_budget must be positive")

    packed: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        start = int(current[0]["offset_start"])
        end = int(current[-1]["offset_end"])
        packed.append(
            {
                "text": text[start:end].strip(),
                "offset_start": start,
                "offset_end": end,
                "token_count": current_tokens,
            }
        )
        current = []
        current_tokens = 0

    for span in spans:
        span_tokens = max(count_text_tokens(provider, span["text"]), 1)
        if current and current_tokens + span_tokens > chunk_token_budget:
            flush()
        current.append(span)
        current_tokens += span_tokens
        if current_tokens >= chunk_token_budget:
            flush()

    flush()
    return packed or _fallback_segment(text)


def segment_text(
    text: str,
    mode: str,
    *,
    provider: Any | None = None,
    chunk_token_budget: int = 1024,
) -> List[Dict[str, Any]]:
    if not text.strip():
        return []

    if mode == "paragraph":
        return _paragraph_spans(text)
    if mode == "sentence_packed":
        return _pack_sentence_spans(
            text,
            _sentence_spans(text),
            provider=provider,
            chunk_token_budget=chunk_token_budget,
        )
    return _sentence_spans(text)


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


def score_groundedness_chunked(
    *,
    support_batches: Sequence[Sequence[SupportUnitInput]],
    response_embeddings: torch.Tensor,
    response_tokens: Sequence[str],
    query_embeddings: Optional[torch.Tensor] = None,
    query_tokens: Optional[Sequence[str]] = None,
    evidence_limit: int = 8,
    primary_metric: str = "reverse_context",
    debug_dense_matrices: bool = False,
) -> Dict[str, Any]:
    """Score chunked support windows and merge them by per-token maxima.

    This is mathematically exact for the shipped naive reverse-context score
    because the support-token partition only changes how the maxima are computed,
    not the underlying support-token union.
    """

    batches = [list(batch) for batch in support_batches if batch]
    if not batches:
        raise ValueError("At least one non-empty support batch is required")

    flat_support_units = [unit for batch in batches for unit in batch]
    response_tokens_aligned = align_tokens(response_tokens, int(response_embeddings.shape[0]))
    weights = token_weights(response_tokens_aligned).to(dtype=torch.float32)

    support_units_payload: List[Dict[str, Any]] = []
    support_token_scores: List[List[float]] = []
    for unit_idx, unit in enumerate(flat_support_units):
        tokens = align_tokens(unit.tokens, int(unit.embeddings.shape[0]))
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

    batch_offsets: List[int] = []
    batch_results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    response_to_support_parts: List[np.ndarray] = []
    triangular_gated_parts: List[np.ndarray] = []
    response_to_query_matrix: Optional[List[List[float]]] = None

    support_offset = 0
    for batch in batches:
        batch_offsets.append(support_offset)
        support_offset += len(batch)
        batch_result = score_groundedness(
            support_units=batch,
            response_embeddings=response_embeddings,
            response_tokens=response_tokens_aligned,
            query_embeddings=query_embeddings,
            query_tokens=query_tokens,
            evidence_limit=evidence_limit,
            primary_metric=primary_metric,
            debug_dense_matrices=debug_dense_matrices,
        )
        batch_results.append(batch_result)
        warnings.extend(batch_result.get("warnings", []))
        if debug_dense_matrices:
            debug_payload = batch_result.get("debug") or {}
            if debug_payload.get("response_to_support") is not None:
                response_to_support_parts.append(np.asarray(debug_payload["response_to_support"], dtype=np.float32))
            if debug_payload.get("triangular_gated") is not None:
                triangular_gated_parts.append(np.asarray(debug_payload["triangular_gated"], dtype=np.float32))
            if response_to_query_matrix is None and debug_payload.get("response_to_query") is not None:
                response_to_query_matrix = debug_payload["response_to_query"]

    token_count = len(response_tokens_aligned)
    merged_reverse_context = [float("-inf")] * token_count
    merged_reverse_query_context: Optional[List[float]] = None
    merged_triangular: Optional[List[float]] = None
    merged_echo: Optional[List[float]] = None
    merged_query_coverages: Optional[List[float]] = None
    merged_query_tokens: Optional[List[str]] = None

    winner_scores = [float("-inf")] * token_count
    winner_support_units: List[Optional[int]] = [None] * token_count
    winner_support_tokens: List[Optional[int]] = [None] * token_count
    winner_support_token_text: List[Optional[str]] = [None] * token_count
    winner_chunk_ids: List[Optional[Any]] = [None] * token_count

    for batch_idx, batch_result in enumerate(batch_results):
        support_unit_offset = batch_offsets[batch_idx]
        for row in batch_result["response_tokens"]:
            token_idx = int(row["index"])
            reverse_context_value = float(row["reverse_context"])
            if reverse_context_value > merged_reverse_context[token_idx]:
                merged_reverse_context[token_idx] = reverse_context_value

            reverse_query_context_value = row.get("reverse_query_context")
            if reverse_query_context_value is not None:
                if merged_reverse_query_context is None:
                    merged_reverse_query_context = [float("-inf")] * token_count
                reverse_query_context_value = float(reverse_query_context_value)
                if reverse_query_context_value > merged_reverse_query_context[token_idx]:
                    merged_reverse_query_context[token_idx] = reverse_query_context_value

            triangular_value = row.get("triangular")
            if triangular_value is not None:
                if merged_triangular is None:
                    merged_triangular = [float("-inf")] * token_count
                triangular_value = float(triangular_value)
                if triangular_value > merged_triangular[token_idx]:
                    merged_triangular[token_idx] = triangular_value

            echo_value = row.get("echo")
            if echo_value is not None:
                if merged_echo is None:
                    merged_echo = [float("-inf")] * token_count
                echo_value = float(echo_value)
                if echo_value > merged_echo[token_idx]:
                    merged_echo[token_idx] = echo_value

            metric_value = (
                float(row["triangular"])
                if primary_metric == "triangular" and row.get("triangular") is not None
                else reverse_context_value
            )
            if metric_value > winner_scores[token_idx]:
                winner_scores[token_idx] = metric_value
                local_support_unit = row.get("support_unit_index")
                winner_support_units[token_idx] = (
                    support_unit_offset + int(local_support_unit) if local_support_unit is not None else None
                )
                winner_support_tokens[token_idx] = (
                    int(row["support_token_index"]) if row.get("support_token_index") is not None else None
                )
                winner_support_token_text[token_idx] = row.get("support_token")
                winner_chunk_ids[token_idx] = row.get("chunk_id")

        query_rows = batch_result.get("query_tokens") or []
        if query_rows:
            if merged_query_coverages is None:
                merged_query_coverages = [float("-inf")] * len(query_rows)
                merged_query_tokens = [str(row["token"]) for row in query_rows]
            for row in query_rows:
                query_idx = int(row["index"])
                coverage_value = float(row["coverage"])
                if coverage_value > merged_query_coverages[query_idx]:
                    merged_query_coverages[query_idx] = coverage_value

    reverse_context_values = torch.tensor(merged_reverse_context, dtype=torch.float32)
    reverse_context_score = weighted_groundedness(reverse_context_values, weights)

    reverse_query_context_score: Optional[float] = None
    reverse_query_context_values: Optional[torch.Tensor] = None
    if merged_reverse_query_context is not None:
        reverse_query_context_values = torch.tensor(merged_reverse_query_context, dtype=torch.float32)
        reverse_query_context_score = weighted_groundedness(reverse_query_context_values, weights)

    triangular_score: Optional[float] = None
    triangular_values: Optional[torch.Tensor] = None
    if merged_triangular is not None:
        triangular_values = torch.tensor(merged_triangular, dtype=torch.float32)
        triangular_score = weighted_groundedness(triangular_values, weights)

    echo_mean: Optional[float] = None
    echo_values: Optional[torch.Tensor] = None
    if merged_echo is not None:
        echo_values = torch.tensor(merged_echo, dtype=torch.float32)
        echo_mean = weighted_groundedness(echo_values, weights)

    grounded_coverage_score: Optional[float] = None
    query_token_rows: Optional[List[Dict[str, Any]]] = None
    if merged_query_coverages is not None and merged_query_tokens is not None:
        query_coverage_tensor = torch.tensor(merged_query_coverages, dtype=torch.float32)
        grounded_coverage_score = grounded_coverage(query_coverage_tensor)
        query_token_rows = [
            {
                "index": idx,
                "token": merged_query_tokens[idx],
                "coverage": float(query_coverage_tensor[idx].item()),
            }
            for idx in range(len(merged_query_tokens))
        ]

    metric_name = primary_metric
    metric_values = triangular_values if metric_name == "triangular" else reverse_context_values
    if metric_name == "triangular" and metric_values is None:
        raise ValueError("triangular primary metric requires query embeddings")

    support_score_numerators = [0.0 for _ in flat_support_units]
    support_score_denominators = [0.0 for _ in flat_support_units]
    response_token_rows: List[Dict[str, Any]] = []
    evidence_candidates: List[Dict[str, Any]] = []
    for token_idx, token in enumerate(response_tokens_aligned):
        support_unit_idx = winner_support_units[token_idx]
        support_token_idx = winner_support_tokens[token_idx]
        support_token = winner_support_token_text[token_idx]
        chunk_id = winner_chunk_ids[token_idx]
        score_value = float(metric_values[token_idx].item())
        if support_unit_idx is not None and support_token_idx is not None:
            support_token_scores[support_unit_idx][support_token_idx] = max(
                support_token_scores[support_unit_idx][support_token_idx],
                score_value,
            )
            support_score_numerators[support_unit_idx] += float(weights[token_idx].item()) * score_value
            support_score_denominators[support_unit_idx] += float(weights[token_idx].item())
            support_units_payload[support_unit_idx]["matched_response_tokens"] += 1
            if float(weights[token_idx].item()) > 0:
                evidence_candidates.append(
                    {
                        "response_token_index": token_idx,
                        "response_token": token,
                        "support_unit_index": support_unit_idx,
                        "support_token_index": support_token_idx,
                        "support_token": support_token,
                        "chunk_id": chunk_id,
                        "metric": metric_name,
                        "score": score_value,
                        "_rank": float(weights[token_idx].item()) * score_value,
                    }
                )
        response_token_rows.append(
            {
                "index": token_idx,
                "token": token,
                "weight": float(weights[token_idx].item()),
                "reverse_context": float(reverse_context_values[token_idx].item()),
                "reverse_query_context": (
                    float(reverse_query_context_values[token_idx].item())
                    if reverse_query_context_values is not None
                    else None
                ),
                "triangular": float(triangular_values[token_idx].item()) if triangular_values is not None else None,
                "echo": float(echo_values[token_idx].item()) if echo_values is not None else None,
                "support_unit_index": support_unit_idx,
                "support_token_index": support_token_idx,
                "support_token": support_token,
                "chunk_id": chunk_id,
                "heatmap_score": score_value,
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
    if debug_dense_matrices:
        debug_payload = {}
        if len(response_to_support_parts) == len(batches) and response_to_support_parts:
            merged_response_to_support = np.concatenate(response_to_support_parts, axis=1)
            if merged_response_to_support.size <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["response_to_support"] = merged_response_to_support.tolist()
            else:
                warnings.append("response_to_support matrix omitted because the merged debug matrix exceeds the size limit")
        if response_to_query_matrix is not None:
            debug_payload["response_to_query"] = response_to_query_matrix
        if len(triangular_gated_parts) == len(batches) and triangular_gated_parts:
            merged_triangular_gated = np.concatenate(triangular_gated_parts, axis=1)
            if merged_triangular_gated.size <= _MAX_DEBUG_MATRIX_ELEMENTS:
                debug_payload["triangular_gated"] = merged_triangular_gated.tolist()
            else:
                warnings.append("triangular_gated matrix omitted because the merged debug matrix exceeds the size limit")
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
        "warnings": list(dict.fromkeys(warnings)),
    }

