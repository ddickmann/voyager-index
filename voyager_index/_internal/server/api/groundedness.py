"""Shared groundedness helpers for the reference API."""

from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from voyager_index._internal.kernels.triton_triangular_maxsim import (
    grounded_coverage,
    naive_reverse_maxsim_qc,
    triangular_maxsim,
    weighted_groundedness,
)
from voyager_index._internal.server.api.groundedness_nli import (
    ClaimVerification,
    NLIProvider,
    aggregate_nli_score,
    fuse_groundedness_v2,
    project_claim_scores_to_tokens,
    verify_claims,
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
_DEFAULT_CHUNK_TOKEN_BUDGET = 256
_CONSENSUS_THRESHOLD = 0.85
_CONSENSUS_ALPHA = 20.0
_CONSENSUS_PENALTY_SCALE = 0.03
_CONSENSUS_UNIT_SCALE = 4.0

_CALIBRATION_MIN_STD = 1e-3
_CALIBRATION_TEMPERATURE = 1.0

# Diverse, short, topically unrelated text spans used to build the null
# distribution for per-token calibration. Mixing domains (history, science,
# geography, biology) keeps the bank from being adversarially close to any
# single response and gives a stable mean/std per response token.
DEFAULT_NULL_BANK_TEXTS: Tuple[str, ...] = (
    "The cat sat on the mat by the window.",
    "In 1492 Christopher Columbus sailed across the Atlantic Ocean.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Eiffel Tower was completed in 1889 on the Champ de Mars in Paris.",
    "Quantum entanglement allows two particles to share a single quantum state.",
    "Water boils at one hundred degrees Celsius at standard sea level pressure.",
    "Shakespeare wrote roughly thirty-seven plays during his lifetime in England.",
    "The Great Wall of China stretches over thirteen thousand miles across Asia.",
    "DNA molecules carry the genetic instructions used by all known organisms.",
    "The first crewed Moon landing took place on the twentieth of July 1969.",
    "Ludwig van Beethoven composed nine symphonies despite progressive deafness.",
    "Light travels at roughly two hundred ninety-nine million meters per second.",
    "Magnesium burns with a bright white flame in the presence of oxygen.",
    "The Pacific Ocean covers more surface area than all of Earth's continents combined.",
    "Penicillin was discovered by Alexander Fleming in 1928 from a stray mould.",
    "The Nile River flows northward through northeastern Africa for over six thousand kilometers.",
)


def default_null_bank_texts() -> List[str]:
    """Return a copy of the default null bank text list used by calibration."""

    return list(DEFAULT_NULL_BANK_TEXTS)


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


def _is_content_token(token: str) -> bool:
    """Mirror of :func:`token_weights`'s positive cases used for support-side masking.

    A token counts as content-bearing when it is not a special/whitespace marker,
    not pure punctuation, and not in the curated stopword list. Numeric, alphabetic,
    and mixed-content tokens all count as content.
    """

    if token in _SPECIAL_TOKENS:
        return False
    stripped = _strip_marker(token).strip()
    if not stripped:
        return False
    if all(ch in string.punctuation for ch in stripped):
        return False
    if stripped.lower() in _STOPWORDS:
        return False
    return True


def support_content_mask(tokens: Sequence[str]) -> torch.Tensor:
    """Return a boolean mask marking content-bearing support tokens.

    ``True`` means the token is allowed to participate in MaxSim attribution and
    breadth statistics; ``False`` excludes filler such as punctuation, special
    tokens, and curated stopwords from being chosen as the "best supporting"
    token for any response token.
    """

    if not tokens:
        return torch.zeros((0,), dtype=torch.bool)
    return torch.tensor([_is_content_token(token) for token in tokens], dtype=torch.bool)


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


def tokenize_text(
    provider: Any,
    text: str,
    *,
    expected_len: Optional[int] = None,
    is_query: bool = False,
) -> List[str]:
    tokens: List[str] = []

    tokenize_method = getattr(provider, "tokenize", None)
    if callable(tokenize_method):
        try:
            tokens = _tokens_from_provider_tokenize(tokenize_method(text, is_query=is_query))
        except TypeError:
            try:
                tokens = _tokens_from_provider_tokenize(tokenize_method(text))
            except Exception:
                tokens = []
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


def count_text_tokens(provider: Any, text: str, *, is_query: bool = False) -> int:
    """Count document-side tokens for support-unit packing.

    Prefers a strict ``encoded_token_count(text, is_query=...)`` hook on the
    provider, which lets vLLM-style providers report the exact post-tokenizer
    sequence length (including specials and ``[D]/[Q]`` prefix). Falls back to
    the bare tokenizer when that hook is not available so older providers still
    work.
    """

    stripped = text.strip()
    if not stripped:
        return 0

    encoded_count_method = getattr(provider, "encoded_token_count", None)
    if callable(encoded_count_method):
        try:
            value = encoded_count_method(text, is_query=is_query)
            if isinstance(value, int) and value >= 0:
                return value
        except TypeError:
            try:
                value = encoded_count_method(text)
                if isinstance(value, int) and value >= 0:
                    return value
            except Exception:
                pass
        except Exception:
            pass

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
            tokens = _tokens_from_provider_tokenize(tokenize_method(text, is_query=is_query))
            if tokens:
                return len(tokens)
        except TypeError:
            try:
                tokens = _tokens_from_provider_tokenize(tokenize_method(text))
                if tokens:
                    return len(tokens)
            except Exception:
                pass
        except Exception:
            pass

    return len(_fallback_tokens(text))


def provider_token_limit(provider: Any, *, is_query: bool = False) -> Optional[int]:
    role_attrs = ("query_maxlen", "query_length") if is_query else ("doc_maxlen", "document_length")
    for attr in (*role_attrs, "max_length", "model_max_length"):
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


def partition_support_units(
    support_units: Sequence[SupportUnitInput],
    *,
    batch_size: int,
) -> List[List[SupportUnitInput]]:
    size = max(1, int(batch_size))
    return [list(support_units[idx : idx + size]) for idx in range(0, len(support_units), size)]


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
        span_tokens = max(count_text_tokens(provider, span["text"], is_query=False), 1)
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
    chunk_token_budget: int = _DEFAULT_CHUNK_TOKEN_BUDGET,
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


def _support_unit_maxima(
    similarity: torch.Tensor,
    support_tokens_nested: Sequence[Sequence[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    unit_values: List[torch.Tensor] = []
    unit_indices: List[torch.Tensor] = []
    offset = 0
    for tokens in support_tokens_nested:
        token_count = len(tokens)
        if token_count <= 0:
            unit_values.append(torch.full((similarity.shape[0],), float("-inf"), dtype=similarity.dtype, device=similarity.device))
            unit_indices.append(torch.full((similarity.shape[0],), -1, dtype=torch.long, device=similarity.device))
            continue
        unit_slice = similarity[:, offset : offset + token_count]
        values, indices = unit_slice.max(dim=1)
        unit_values.append(values)
        unit_indices.append(indices.to(torch.long))
        offset += token_count

    if not unit_values:
        empty = torch.empty((similarity.shape[0], 0), dtype=similarity.dtype, device=similarity.device)
        empty_idx = torch.empty((similarity.shape[0], 0), dtype=torch.long, device=similarity.device)
        return empty, empty_idx
    return torch.stack(unit_values, dim=1), torch.stack(unit_indices, dim=1)


def _normalize_text_for_signature(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def support_unit_signature(unit: SupportUnitInput) -> str:
    """Return a stable signature for de-duplicating near-duplicate support units.

    Preference order: explicit ``chunk_id`` (string), else a SHA1 of the
    normalized support text. Empty texts hash to a stable bucket, so a request
    sending a single empty placeholder twice is still treated as one source.
    """

    if unit.chunk_id is not None:
        return f"chunk:{unit.chunk_id}"
    normalized = _normalize_text_for_signature(unit.text)
    if not normalized:
        return "text:__empty__"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"text:{digest}"


def _dedup_unit_maxima(
    unit_maxima: torch.Tensor,
    signatures: Optional[Sequence[str]],
) -> Tuple[torch.Tensor, int]:
    """Cluster columns of a ``(R, U)`` unit-maxima matrix by signature.

    Returns the deduped matrix (one column per unique signature, taking the
    column-wise max within each cluster) and the count of duplicates removed.
    Falls back to the original matrix when signatures are missing or already
    unique, so the path stays a no-op for the common single-source case.
    """

    if unit_maxima.numel() == 0:
        return unit_maxima, 0
    if not signatures or len(signatures) != int(unit_maxima.shape[1]):
        return unit_maxima, 0

    seen: Dict[str, int] = {}
    cluster_columns: List[List[int]] = []
    for column_idx, signature in enumerate(signatures):
        bucket = seen.get(signature)
        if bucket is None:
            seen[signature] = len(cluster_columns)
            cluster_columns.append([column_idx])
        else:
            cluster_columns[bucket].append(column_idx)

    duplicates_removed = int(unit_maxima.shape[1]) - len(cluster_columns)
    if duplicates_removed <= 0:
        return unit_maxima, 0

    deduped = torch.stack(
        [unit_maxima[:, indices].max(dim=1).values for indices in cluster_columns],
        dim=1,
    )
    return deduped, duplicates_removed


def compute_null_distribution(
    response_embeddings: torch.Tensor,
    null_bank_embeddings: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Per-response-token null mean and std from a bank of unrelated support units.

    For each null bank entry ``b``, computes ``g_t^(b) = max_j sim(r_t, b_j)``
    over the bank's tokens. Aggregates ``{g_t^(b)}`` across the bank into a per
    response token mean and standard deviation. The third return value is the
    number of bank entries that actually contributed (non-empty embeddings).
    """

    if response_embeddings.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.float32)
        return empty, empty, 0
    R_norm = _normalize(response_embeddings.float())
    per_unit_max_values: List[torch.Tensor] = []
    for bank_embedding in null_bank_embeddings:
        if bank_embedding is None or bank_embedding.numel() == 0:
            continue
        B_norm = _normalize(bank_embedding.float())
        sim = R_norm @ B_norm.T
        max_per_response, _indices = sim.max(dim=1)
        per_unit_max_values.append(max_per_response.detach().to(R_norm.device))

    response_token_count = int(response_embeddings.shape[0])
    if not per_unit_max_values:
        zeros = torch.zeros((response_token_count,), dtype=torch.float32)
        ones = torch.ones((response_token_count,), dtype=torch.float32)
        return zeros, ones, 0

    stack = torch.stack(per_unit_max_values, dim=0)
    mean = stack.mean(dim=0)
    if stack.shape[0] > 1:
        std = stack.std(dim=0, unbiased=False)
    else:
        std = torch.full_like(mean, _CALIBRATION_MIN_STD)
    std = std.clamp(min=_CALIBRATION_MIN_STD)
    return mean.to(torch.float32), std.to(torch.float32), int(stack.shape[0])


def calibrate_per_token_scores(
    rc_values: torch.Tensor,
    null_mean: torch.Tensor,
    null_std: torch.Tensor,
    *,
    temperature: float = _CALIBRATION_TEMPERATURE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standardize raw per-token reverse-context scores using a null distribution.

    Returns ``(z, p_grounded)`` where ``z = (g_t - mu_null_t) / sigma_null_t``
    is the per-token z-score against the null bank and
    ``p_grounded = sigmoid(z / temperature)`` is a probability-style
    aggregation-friendly score in ``(0, 1)``.
    """

    if rc_values.numel() == 0:
        empty = torch.zeros_like(rc_values)
        return empty, empty
    safe_std = null_std.clamp(min=_CALIBRATION_MIN_STD)
    z = (rc_values - null_mean) / safe_std
    prob = torch.sigmoid(z / max(float(temperature), _CALIBRATION_MIN_STD))
    return z, prob


# ----------------------------------------------------------------------
# Phase C: narrow-scope literal extraction and guardrails
# ----------------------------------------------------------------------

# Per-mismatch penalty applied to the literal-guarded secondary score.
# Each unmatched response literal multiplies the score by ``(1 - rate)``
# down to a configurable floor.
_LITERAL_PENALTY_RATE = 0.15
_LITERAL_PENALTY_FLOOR = 0.0

# Identity literals like "GTE-3.5-Turbo" or product codes mix letters and
# digits; URLs must end on a non-punctuation character to avoid trailing
# commas/periods leaking into the literal.
_LITERAL_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    # ISO date YYYY-MM-DD
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    # Numeric date DD/MM/YYYY or MM/DD/YYYY
    ("date", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    # Long-form date "20 July 1981" / "20 Jul 1981"
    (
        "date",
        re.compile(
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
            r"Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    # "July 20, 1981" / "Jul 20 1981"
    (
        "date",
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
            r"Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}(?:,)?\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    # Bare 4-digit year (filtered later if also captured by another pattern).
    ("year", re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")),
    # Currency amounts with $/€/£ prefix and optional decimals/commas.
    ("currency", re.compile(r"(?:\$|€|£)\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?")),
    # Percentages.
    ("percent", re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%")),
    # Numeric measurements with common units.
    (
        "measurement",
        re.compile(
            r"\b\d{1,4}(?:\.\d+)?\s?(?:kg|g|mg|km|m|cm|mm|mph|kph|kmh|hours?|"
            r"minutes?|seconds?|days?|years?|months?|weeks?|MB|GB|TB|KB|"
            r"liters?|litres?|ml|gallons?|miles?|feet|inches?|in|ft|lb|lbs|"
            r"oz|°C|°F)\b",
            re.IGNORECASE,
        ),
    ),
    # Standalone numbers (integers / decimals / thousands separators).
    ("number", re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")),
    # URLs (HTTP(S)).
    ("url", re.compile(r"https?://[^\s<>\"']+[^\s<>\"'.,;:!?]")),
    # Email addresses.
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # Mixed-case identifiers ("ABC123", "GTE-3.5", "section-7b") - require at
    # least one letter and one digit so it doesn't fire on prose words or
    # bare numbers (already covered above).
    (
        "identifier",
        re.compile(r"\b(?=[A-Za-z0-9._/-]{3,40}\b)[A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*\b"),
    ),
)


def _normalize_literal_value(kind: str, value: str) -> str:
    """Canonicalize a literal so equivalent surface forms collide on a hash."""

    text = value.strip().lower()
    if kind == "currency":
        text = text.replace(" ", "")
    if kind == "percent":
        text = text.replace(" ", "")
    if kind == "measurement":
        text = re.sub(r"\s+", " ", text)
    if kind == "date":
        text = re.sub(r"[,]", "", text)
        text = re.sub(r"\s+", " ", text)
    if kind == "number":
        text = text.replace(",", "")
    return text


def extract_literals(text: str) -> List[Dict[str, Any]]:
    """Extract narrow-scope literals (dates, numbers, units, identifiers).

    Each literal is represented as ``{"kind": str, "value": str,
    "normalized": str, "start": int, "end": int}``. Overlapping spans are
    resolved greedily by length so the most informative literal wins (e.g. a
    full date beats an embedded year).
    """

    if not text:
        return []

    raw_hits: List[Dict[str, Any]] = []
    for kind, pattern in _LITERAL_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            raw_hits.append(
                {
                    "kind": kind,
                    "value": value,
                    "normalized": _normalize_literal_value(kind, value),
                    "start": int(match.start()),
                    "end": int(match.end()),
                }
            )

    if not raw_hits:
        return []

    raw_hits.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"]))
    accepted: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []
    for hit in raw_hits:
        span = (hit["start"], hit["end"])
        if any(span[0] < end and start < span[1] for start, end in occupied):
            continue
        accepted.append(hit)
        occupied.append(span)
    accepted.sort(key=lambda item: item["start"])
    return accepted


def collect_support_literal_set(support_units: Sequence[Any]) -> Dict[str, set]:
    """Build a per-kind set of normalized literals across all support units."""

    bucket: Dict[str, set] = {}
    for unit in support_units:
        text = getattr(unit, "text", "") or ""
        for literal in extract_literals(text):
            bucket.setdefault(literal["kind"], set()).add(literal["normalized"])
    return bucket


def diff_literals(
    response_text: str,
    support_units: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compare response literals against the support union.

    Returns ``(response_literals, mismatches, matches)``. A response literal is
    counted as matched when the same kind+normalized value appears anywhere in
    the support set, OR (for ``year`` literals) when any support ``date``
    literal contains the same year. The fall-through helps avoid double-counting
    when the support carries the full long-form date but the response only
    surfaces the year.
    """

    response_literals = extract_literals(response_text or "")
    support_buckets = collect_support_literal_set(support_units)
    support_dates = support_buckets.get("date", set())

    matches: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    for literal in response_literals:
        kind = literal["kind"]
        norm = literal["normalized"]
        bucket = support_buckets.get(kind, set())
        is_match = norm in bucket
        if not is_match and kind == "year":
            is_match = any(norm in support_date for support_date in support_dates)
        if not is_match and kind == "number":
            for measurement in support_buckets.get("measurement", set()):
                if measurement.startswith(norm + " ") or measurement == norm:
                    is_match = True
                    break
        if is_match:
            matches.append(literal)
        else:
            mismatches.append(literal)
    return response_literals, mismatches, matches


def literal_guarded_score(
    base_score: float,
    mismatches: Sequence[Dict[str, Any]],
    *,
    rate: float = _LITERAL_PENALTY_RATE,
    floor: float = _LITERAL_PENALTY_FLOOR,
) -> float:
    """Apply a multiplicative penalty per literal mismatch with a hard floor."""

    if base_score <= 0 or not mismatches:
        return float(base_score)
    penalty = (1.0 - max(0.0, min(1.0, rate))) ** len(mismatches)
    guarded = float(base_score) * float(penalty)
    if floor > 0:
        guarded = max(guarded, float(floor))
    return guarded


def _consensus_statistics(
    unit_maxima: torch.Tensor,
    reverse_context_values: torch.Tensor,
    weights: torch.Tensor,
    *,
    unit_signatures: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor | float | int]:
    if unit_maxima.numel() == 0:
        zeros = torch.zeros_like(reverse_context_values)
        return {
            "hits": zeros.to(torch.long),
            "soft_breadth": zeros,
            "effective_support_units": zeros,
            "consensus_values": reverse_context_values,
            "consensus_score": float(weighted_groundedness(reverse_context_values, weights)),
            "duplicates_removed": 0,
            "effective_unit_count": 0,
        }

    deduped_maxima, duplicates_removed = _dedup_unit_maxima(unit_maxima, unit_signatures)
    positive_mass = torch.clamp(deduped_maxima - _CONSENSUS_THRESHOLD, min=0.0)
    hits = (deduped_maxima >= _CONSENSUS_THRESHOLD).sum(dim=1).to(torch.long)
    soft_breadth = torch.sigmoid((deduped_maxima - _CONSENSUS_THRESHOLD) * _CONSENSUS_ALPHA).sum(dim=1)
    mass_sum = positive_mass.sum(dim=1)
    mass_sq_sum = (positive_mass**2).sum(dim=1)
    effective_support_units = torch.where(
        mass_sq_sum > 0,
        (mass_sum**2) / torch.clamp(mass_sq_sum, min=1e-9),
        torch.zeros_like(mass_sum),
    )
    breadth_factor = 1.0 - torch.exp(
        -torch.clamp(effective_support_units - 1.0, min=0.0) / _CONSENSUS_UNIT_SCALE
    )
    consensus_values = torch.clamp(
        reverse_context_values * (1.0 - (_CONSENSUS_PENALTY_SCALE * (1.0 - breadth_factor))),
        min=0.0,
    )
    return {
        "hits": hits,
        "soft_breadth": soft_breadth,
        "effective_support_units": effective_support_units,
        "consensus_values": consensus_values,
        "consensus_score": float(weighted_groundedness(consensus_values, weights)),
        "duplicates_removed": int(duplicates_removed),
        "effective_unit_count": int(deduped_maxima.shape[1]),
    }


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
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    response_text: Optional[str] = None,
    nli_provider: Optional[NLIProvider] = None,
    nli_max_claims: Optional[int] = None,
    nli_top_k_premises: Optional[int] = None,
    nli_max_batch: Optional[int] = None,
    nli_max_latency_ms: Optional[float] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
    _emit_dedup_warning: bool = True,
) -> Dict[str, Any]:
    """Score response groundedness against support units.

    Returns a plain python payload suitable for Pydantic response models.
    """

    if not support_units:
        raise ValueError("At least one support unit is required")

    support_tensors = [unit.embeddings.float() for unit in support_units]
    support_tokens_nested = [align_tokens(unit.tokens, int(unit.embeddings.shape[0])) for unit in support_units]
    response_tokens_aligned = align_tokens(response_tokens, int(response_embeddings.shape[0]))
    support_signatures = [support_unit_signature(unit) for unit in support_units]

    C = torch.cat(support_tensors, dim=0)
    R = response_embeddings.float()
    C_norm = _normalize(C)
    R_norm = _normalize(R)
    sim_rc_raw = R_norm @ C_norm.T

    flat_support_tokens = [token for tokens in support_tokens_nested for token in tokens]
    support_content_mask_flat = support_content_mask(flat_support_tokens).to(sim_rc_raw.device)
    if support_content_mask_flat.numel() > 0 and bool(support_content_mask_flat.any()):
        masked_sim_rc = sim_rc_raw.masked_fill(~support_content_mask_flat.unsqueeze(0), -1e4)
    else:
        masked_sim_rc = sim_rc_raw
    sim_rc = masked_sim_rc
    rc_values, rc_indices = sim_rc.max(dim=1)

    weights = token_weights(response_tokens_aligned).to(rc_values.device, dtype=rc_values.dtype)
    reverse_context_score = weighted_groundedness(rc_values, weights)
    reverse_context_unit_values, _reverse_context_unit_token_indices = _support_unit_maxima(
        sim_rc,
        support_tokens_nested,
    )
    consensus = _consensus_statistics(
        reverse_context_unit_values,
        rc_values,
        weights,
        unit_signatures=support_signatures,
    )
    consensus_values = consensus["consensus_values"]
    consensus_hardened_score = float(consensus["consensus_score"])
    support_unit_hits = consensus["hits"]
    support_unit_soft_breadth = consensus["soft_breadth"]
    effective_support_units = consensus["effective_support_units"]
    consensus_duplicates_removed = int(consensus.get("duplicates_removed", 0))
    consensus_effective_unit_count = int(consensus.get("effective_unit_count", reverse_context_unit_values.shape[1]))

    null_mean, null_std, null_bank_size = compute_null_distribution(
        response_embeddings,
        list(null_bank_embeddings) if null_bank_embeddings else [],
    )
    null_mean = null_mean.to(rc_values.device, dtype=rc_values.dtype)
    null_std = null_std.to(rc_values.device, dtype=rc_values.dtype)
    rc_z_values, p_grounded_values = calibrate_per_token_scores(rc_values, null_mean, null_std)
    if null_bank_size > 0:
        reverse_context_calibrated_score = weighted_groundedness(p_grounded_values, weights)
    else:
        reverse_context_calibrated_score = float(reverse_context_score)

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
            sim_tri = torch.minimum(sim_rc_raw, tri.a[None, :])

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
                "reverse_context_calibrated": float(p_grounded_values[idx].item()) if null_bank_size > 0 else None,
                "reverse_context_z": float(rc_z_values[idx].item()) if null_bank_size > 0 else None,
                "null_mean": float(null_mean[idx].item()) if null_bank_size > 0 else None,
                "null_std": float(null_std[idx].item()) if null_bank_size > 0 else None,
                "consensus_hardened": float(consensus_values[idx].item()),
                "support_unit_hits_above_threshold": int(support_unit_hits[idx].item()),
                "support_unit_soft_breadth": float(support_unit_soft_breadth[idx].item()),
                "effective_support_units": float(effective_support_units[idx].item()),
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
    if _emit_dedup_warning and consensus_duplicates_removed > 0:
        warnings.append(
            "support_unit_dedup: {removed} duplicate support unit(s) collapsed into {kept} unique source(s) for breadth statistics".format(
                removed=consensus_duplicates_removed,
                kept=consensus_effective_unit_count,
            )
        )
    if null_bank_size == 0:
        warnings.append(
            "calibration_disabled: no null bank embeddings supplied; reverse_context_calibrated falls back to raw reverse_context"
        )
    if debug_dense_matrices:
        debug_payload = {}
        rc_elements = int(sim_rc_raw.shape[0] * sim_rc_raw.shape[1])
        if rc_elements <= _MAX_DEBUG_MATRIX_ELEMENTS:
            debug_payload["response_to_support"] = sim_rc_raw.detach().cpu().tolist()
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

    response_literals, literal_mismatches, literal_matches = diff_literals(
        response_text or "", support_units
    )
    base_for_guard = (
        reverse_context_calibrated_score if null_bank_size > 0 else reverse_context_score
    )
    literal_guarded_value = literal_guarded_score(float(base_for_guard), literal_mismatches)
    if literal_mismatches:
        warnings.append(
            "literal_mismatch: {n} response literal(s) not present in support: {sample}".format(
                n=len(literal_mismatches),
                sample=", ".join(
                    "{kind}={value}".format(kind=item["kind"], value=item["value"])
                    for item in literal_mismatches[:5]
                ),
            )
        )

    response_token_total = len(response_tokens_aligned)
    nli_payload = _maybe_run_nli(
        response_text=response_text,
        response_tokens=response_tokens_aligned,
        support_units=support_units,
        nli_provider=nli_provider,
        nli_max_claims=nli_max_claims,
        nli_top_k_premises=nli_top_k_premises,
        nli_max_batch=nli_max_batch,
        nli_max_latency_ms=nli_max_latency_ms,
    )
    if nli_payload is not None:
        warnings.extend(nli_payload.pop("warnings", []))
    nli_aggregate = nli_payload["aggregate_score"] if nli_payload else None
    nli_per_token = nli_payload["per_token"] if nli_payload else [None] * response_token_total
    groundedness_v2 = fuse_groundedness_v2(
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score) if null_bank_size > 0 else float(reverse_context_score)
        ),
        literal_guarded=float(literal_guarded_value),
        nli_aggregate=nli_aggregate,
        weights=fusion_weights,
    )
    for token_idx, row in enumerate(response_token_rows):
        row["nli_score"] = nli_per_token[token_idx] if token_idx < len(nli_per_token) else None

    scores = {
        "primary_name": metric_name,
        "primary_score": float(triangular_score if metric_name == "triangular" else reverse_context_score),
        "reverse_context": float(reverse_context_score),
        "reverse_context_calibrated": float(reverse_context_calibrated_score) if null_bank_size > 0 else None,
        "literal_guarded": float(literal_guarded_value),
        "literal_mismatch_count": int(len(literal_mismatches)),
        "literal_match_count": int(len(literal_matches)),
        "literal_total_count": int(len(response_literals)),
        "nli_aggregate": float(nli_aggregate) if nli_aggregate is not None else None,
        "nli_claim_count": (
            int(nli_payload["claim_count"]) if nli_payload is not None else 0
        ),
        "nli_skipped_count": (
            int(nli_payload["skipped_count"]) if nli_payload is not None else 0
        ),
        "groundedness_v2": float(groundedness_v2) if groundedness_v2 is not None else None,
        "consensus_hardened": consensus_hardened_score,
        "reverse_query_context": (
            float(reverse_query_context_score) if reverse_query_context_score is not None else None
        ),
        "triangular": float(triangular_score) if triangular_score is not None else None,
        "echo_mean": float(echo_mean) if echo_mean is not None else None,
        "grounded_coverage": float(grounded_coverage_score) if grounded_coverage_score is not None else None,
        "null_bank_size": null_bank_size,
    }

    return {
        "scores": scores,
        "response_tokens": response_token_rows,
        "support_units": support_units_payload,
        "top_evidence": top_evidence,
        "query_tokens": query_token_rows,
        "debug": debug_payload,
        "warnings": warnings,
        "literal_diagnostics": {
            "response_literals": response_literals,
            "matches": literal_matches,
            "mismatches": literal_mismatches,
        },
        "nli_diagnostics": (
            None if nli_payload is None else {
                "claims": nli_payload["claim_records"],
                "aggregate_score": nli_payload["aggregate_score"],
            }
        ),
        "_internals": {
            "reverse_context_unit_values": reverse_context_unit_values,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_bank_size": null_bank_size,
        },
    }


def _maybe_run_nli(
    *,
    response_text: Optional[str],
    response_tokens: Sequence[str],
    support_units: Sequence[Any],
    nli_provider: Optional[NLIProvider],
    nli_max_claims: Optional[int],
    nli_top_k_premises: Optional[int],
    nli_max_batch: Optional[int],
    nli_max_latency_ms: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Run claim-level NLI and project to tokens; return ``None`` when disabled."""

    if nli_provider is None:
        return None
    text = response_text or ""
    if not text.strip() or not list(support_units):
        return {
            "claim_records": [],
            "claim_count": 0,
            "skipped_count": 0,
            "aggregate_score": None,
            "per_token": [None] * len(response_tokens),
            "warnings": [],
        }
    verifications, warnings = verify_claims(
        response_text=text,
        support_units=support_units,
        nli_provider=nli_provider,
        max_claims=nli_max_claims if nli_max_claims is not None else 16,
        top_k_premises=nli_top_k_premises if nli_top_k_premises is not None else 3,
        max_batch=nli_max_batch if nli_max_batch is not None else 16,
        max_latency_ms=nli_max_latency_ms if nli_max_latency_ms is not None else 2000.0,
    )
    aggregate = aggregate_nli_score(verifications)
    per_token = project_claim_scores_to_tokens(response_tokens, text, verifications)
    claim_records = [_claim_to_dict(v) for v in verifications]
    skipped = sum(1 for v in verifications if v.skipped)
    return {
        "claim_records": claim_records,
        "claim_count": len(verifications),
        "skipped_count": int(skipped),
        "aggregate_score": aggregate,
        "per_token": per_token,
        "warnings": warnings,
    }


def _claim_to_dict(verification: ClaimVerification) -> Dict[str, Any]:
    return {
        "index": int(verification.claim.index),
        "text": verification.claim.text,
        "char_start": int(verification.claim.char_start),
        "char_end": int(verification.claim.char_end),
        "entailment": float(verification.entailment),
        "neutral": float(verification.neutral),
        "contradiction": float(verification.contradiction),
        "score": float(verification.score),
        "skipped": bool(verification.skipped),
        "skip_reason": verification.skip_reason,
        "premise_count": int(len(verification.premises)),
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
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    response_text: Optional[str] = None,
    nli_provider: Optional[NLIProvider] = None,
    nli_max_claims: Optional[int] = None,
    nli_top_k_premises: Optional[int] = None,
    nli_max_batch: Optional[int] = None,
    nli_max_latency_ms: Optional[float] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score chunked support windows and merge them by per-token maxima.

    This is mathematically exact for the shipped naive reverse-context score
    because the support-token partition only changes how the maxima are computed,
    not the underlying support-token union.
    """

    batches = [list(batch) for batch in support_batches if batch]
    if not batches:
        raise ValueError("At least one non-empty support batch is required")
    if len(batches) == 1:
        return score_groundedness(
            support_units=batches[0],
            response_embeddings=response_embeddings,
            response_tokens=response_tokens,
            query_embeddings=query_embeddings,
            query_tokens=query_tokens,
            evidence_limit=evidence_limit,
            primary_metric=primary_metric,
            debug_dense_matrices=debug_dense_matrices,
            null_bank_embeddings=null_bank_embeddings,
            response_text=response_text,
            nli_provider=nli_provider,
            nli_max_claims=nli_max_claims,
            nli_top_k_premises=nli_top_k_premises,
            nli_max_batch=nli_max_batch,
            nli_max_latency_ms=nli_max_latency_ms,
            fusion_weights=fusion_weights,
        )

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
    reverse_context_unit_value_parts: List[torch.Tensor] = []

    support_offset = 0
    for batch_idx, batch in enumerate(batches):
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
            null_bank_embeddings=null_bank_embeddings if batch_idx == 0 else None,
            response_text=None,
            _emit_dedup_warning=False,
        )
        batch_results.append(batch_result)
        for warning_msg in batch_result.get("warnings", []):
            if warning_msg.startswith("calibration_disabled"):
                continue
            if warning_msg.startswith("literal_mismatch"):
                continue
            warnings.append(warning_msg)
        batch_internals = batch_result.get("_internals") or {}
        if batch_internals.get("reverse_context_unit_values") is not None:
            reverse_context_unit_value_parts.append(batch_internals["reverse_context_unit_values"])
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
    reverse_context_unit_values = (
        torch.cat(reverse_context_unit_value_parts, dim=1)
        if reverse_context_unit_value_parts
        else torch.empty((token_count, 0), dtype=torch.float32)
    )

    first_internals = batch_results[0].get("_internals") or {} if batch_results else {}
    null_mean_tensor = first_internals.get("null_mean")
    null_std_tensor = first_internals.get("null_std")
    null_bank_size = int(first_internals.get("null_bank_size", 0) or 0)
    if null_bank_size > 0 and null_mean_tensor is not None and null_std_tensor is not None:
        null_mean_tensor = null_mean_tensor.detach().to(dtype=torch.float32)
        null_std_tensor = null_std_tensor.detach().to(dtype=torch.float32)
        rc_z_values, p_grounded_values = calibrate_per_token_scores(
            reverse_context_values,
            null_mean_tensor,
            null_std_tensor,
            temperature=_CALIBRATION_TEMPERATURE,
        )
        reverse_context_calibrated_score = weighted_groundedness(p_grounded_values, weights)
    else:
        rc_z_values = torch.zeros_like(reverse_context_values)
        p_grounded_values = reverse_context_values.clone()
        reverse_context_calibrated_score = float(reverse_context_score)
        null_mean_tensor = None
        null_std_tensor = None
    if null_bank_size == 0:
        warnings.append(
            "calibration_disabled: no null bank embeddings supplied; reverse_context_calibrated falls back to raw reverse_context"
        )
    flat_unit_signatures = [support_unit_signature(unit) for unit in flat_support_units]
    consensus = _consensus_statistics(
        reverse_context_unit_values,
        reverse_context_values,
        weights,
        unit_signatures=flat_unit_signatures,
    )
    consensus_values = consensus["consensus_values"]
    consensus_hardened_score = float(consensus["consensus_score"])
    support_unit_hits = consensus["hits"]
    support_unit_soft_breadth = consensus["soft_breadth"]
    effective_support_units = consensus["effective_support_units"]
    consensus_duplicates_removed = int(consensus.get("duplicates_removed", 0))
    consensus_effective_unit_count = int(consensus.get("effective_unit_count", reverse_context_unit_values.shape[1]))
    if consensus_duplicates_removed > 0:
        warnings.append(
            "support_unit_dedup: {removed} duplicate support unit(s) collapsed into {kept} unique source(s) for breadth statistics".format(
                removed=consensus_duplicates_removed,
                kept=consensus_effective_unit_count,
            )
        )

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
                "reverse_context_calibrated": (
                    float(p_grounded_values[token_idx].item()) if null_bank_size > 0 else None
                ),
                "reverse_context_z": (
                    float(rc_z_values[token_idx].item()) if null_bank_size > 0 else None
                ),
                "null_mean": (
                    float(null_mean_tensor[token_idx].item())
                    if null_mean_tensor is not None
                    else None
                ),
                "null_std": (
                    float(null_std_tensor[token_idx].item())
                    if null_std_tensor is not None
                    else None
                ),
                "consensus_hardened": float(consensus_values[token_idx].item()),
                "support_unit_hits_above_threshold": int(support_unit_hits[token_idx].item()),
                "support_unit_soft_breadth": float(support_unit_soft_breadth[token_idx].item()),
                "effective_support_units": float(effective_support_units[token_idx].item()),
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

    response_literals, literal_mismatches, literal_matches = diff_literals(
        response_text or "", flat_support_units
    )
    base_for_guard = (
        reverse_context_calibrated_score if null_bank_size > 0 else reverse_context_score
    )
    literal_guarded_value = literal_guarded_score(float(base_for_guard), literal_mismatches)
    if literal_mismatches:
        warnings.append(
            "literal_mismatch: {n} response literal(s) not present in support: {sample}".format(
                n=len(literal_mismatches),
                sample=", ".join(
                    "{kind}={value}".format(kind=item["kind"], value=item["value"])
                    for item in literal_mismatches[:5]
                ),
            )
        )

    nli_payload = _maybe_run_nli(
        response_text=response_text,
        response_tokens=response_tokens_aligned,
        support_units=flat_support_units,
        nli_provider=nli_provider,
        nli_max_claims=nli_max_claims,
        nli_top_k_premises=nli_top_k_premises,
        nli_max_batch=nli_max_batch,
        nli_max_latency_ms=nli_max_latency_ms,
    )
    if nli_payload is not None:
        warnings.extend(nli_payload.pop("warnings", []))
    nli_aggregate = nli_payload["aggregate_score"] if nli_payload else None
    nli_per_token_chunked = nli_payload["per_token"] if nli_payload else [None] * token_count
    groundedness_v2 = fuse_groundedness_v2(
        reverse_context_calibrated=(
            float(reverse_context_calibrated_score) if null_bank_size > 0 else float(reverse_context_score)
        ),
        literal_guarded=float(literal_guarded_value),
        nli_aggregate=nli_aggregate,
        weights=fusion_weights,
    )
    for token_idx, row in enumerate(response_token_rows):
        row["nli_score"] = (
            nli_per_token_chunked[token_idx]
            if token_idx < len(nli_per_token_chunked)
            else None
        )

    scores = {
        "primary_name": metric_name,
        "primary_score": float(triangular_score if metric_name == "triangular" else reverse_context_score),
        "reverse_context": float(reverse_context_score),
        "reverse_context_calibrated": (
            float(reverse_context_calibrated_score) if null_bank_size > 0 else None
        ),
        "literal_guarded": float(literal_guarded_value),
        "literal_mismatch_count": int(len(literal_mismatches)),
        "literal_match_count": int(len(literal_matches)),
        "literal_total_count": int(len(response_literals)),
        "nli_aggregate": float(nli_aggregate) if nli_aggregate is not None else None,
        "nli_claim_count": (
            int(nli_payload["claim_count"]) if nli_payload is not None else 0
        ),
        "nli_skipped_count": (
            int(nli_payload["skipped_count"]) if nli_payload is not None else 0
        ),
        "groundedness_v2": float(groundedness_v2) if groundedness_v2 is not None else None,
        "consensus_hardened": consensus_hardened_score,
        "reverse_query_context": (
            float(reverse_query_context_score) if reverse_query_context_score is not None else None
        ),
        "triangular": float(triangular_score) if triangular_score is not None else None,
        "echo_mean": float(echo_mean) if echo_mean is not None else None,
        "grounded_coverage": float(grounded_coverage_score) if grounded_coverage_score is not None else None,
        "null_bank_size": null_bank_size,
    }

    return {
        "scores": scores,
        "response_tokens": response_token_rows,
        "support_units": support_units_payload,
        "top_evidence": top_evidence,
        "query_tokens": query_token_rows,
        "debug": debug_payload,
        "warnings": list(dict.fromkeys(warnings)),
        "literal_diagnostics": {
            "response_literals": response_literals,
            "matches": literal_matches,
            "mismatches": literal_mismatches,
        },
        "nli_diagnostics": (
            None if nli_payload is None else {
                "claims": nli_payload["claim_records"],
                "aggregate_score": nli_payload["aggregate_score"],
            }
        ),
        "_internals": {
            "reverse_context_unit_values": reverse_context_unit_values,
            "null_mean": null_mean_tensor,
            "null_std": null_std_tensor,
            "null_bank_size": null_bank_size,
        },
    }

