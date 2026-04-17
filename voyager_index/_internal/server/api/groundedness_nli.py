"""NLI / claim-level verification for groundedness Real Hardening Phase D.

This module is intentionally pluggable: the heavy entailment model is loaded
lazily and behind a feature flag, while the orchestration logic (claim
splitting, premise selection, per-token projection, score fusion) is pure
Python and unit-testable with a fake provider.

The module never raises during normal operation; on any failure it returns
``None`` so callers can keep returning the embedding-only headline.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Public constants and tunables
# ----------------------------------------------------------------------

# Default fusion weights for groundedness_v2. Must be non-negative and will
# be re-normalized at fusion time so callers can disable a channel by setting
# its weight to 0 without re-tuning the others.
_DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
    "calibrated": 0.5,
    "literal": 0.2,
    "nli": 0.3,
}

_DEFAULT_MAX_CLAIMS = 16
_DEFAULT_TOP_K_PREMISES = 3
_DEFAULT_NLI_MAX_BATCH = 16
_DEFAULT_NLI_MAX_LATENCY_MS = 2000.0

_CLAIM_SPLIT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.UNICODE)
_CONJUNCTION_SPLIT_RE = re.compile(r"\s*(?:;|\bbut\b|\bhowever\b|\bwhereas\b)\s+", re.IGNORECASE)
_TOKEN_FALLBACK_RE = re.compile(r"\w+", re.UNICODE)
_STRIP_PREFIXES = ("Ġ", "▁", "##")


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------


@dataclass
class Claim:
    """A response sub-statement we want to verify against the support union."""

    index: int
    text: str
    char_start: int
    char_end: int


@dataclass
class ClaimVerification:
    """Per-claim NLI score and bookkeeping."""

    claim: Claim
    entailment: float
    neutral: float
    contradiction: float
    score: float
    premises: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None


class NLIProvider(Protocol):
    """Minimal contract for an entailment backend.

    Implementations receive aligned premise/hypothesis pairs and must return
    ``(entail_p, neutral_p, contradict_p)`` triples summing to roughly ``1``.
    """

    def entail(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> List[Tuple[float, float, float]]:
        ...


# ----------------------------------------------------------------------
# Claim splitting
# ----------------------------------------------------------------------


def split_claims(response_text: str, *, max_claims: int = _DEFAULT_MAX_CLAIMS) -> List[Claim]:
    """Split a response into a small set of declarative claims.

    The splitter is intentionally conservative: it segments by sentence
    punctuation first, then optionally splits long sentences on strong
    conjunctions (``;``, ``but``, ``however``, ``whereas``). Each claim keeps
    its character offsets in the original response so callers can project
    NLI scores back to specific response tokens later.
    """

    if not response_text or not response_text.strip():
        return []

    raw_spans: List[Tuple[int, int, str]] = []
    for match in _CLAIM_SPLIT_RE.finditer(response_text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        sentence_start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
        sentence_end = sentence_start + len(sentence)
        raw_spans.append((sentence_start, sentence_end, sentence))
    if not raw_spans:
        text = response_text.strip()
        offset = response_text.find(text)
        raw_spans.append((max(0, offset), max(0, offset) + len(text), text))

    refined: List[Tuple[int, int, str]] = []
    for start, end, sentence in raw_spans:
        if len(sentence) <= 220:
            refined.append((start, end, sentence))
            continue
        cursor = start
        for sub in _CONJUNCTION_SPLIT_RE.split(sentence):
            sub_clean = sub.strip()
            if not sub_clean:
                continue
            sub_start = response_text.find(sub_clean, cursor, end)
            if sub_start < 0:
                sub_start = cursor
            sub_end = sub_start + len(sub_clean)
            refined.append((sub_start, sub_end, sub_clean))
            cursor = sub_end

    if max_claims > 0 and len(refined) > max_claims:
        refined = refined[:max_claims]

    return [
        Claim(index=idx, text=text, char_start=start, char_end=end)
        for idx, (start, end, text) in enumerate(refined)
    ]


# ----------------------------------------------------------------------
# Premise selection (lexical overlap)
# ----------------------------------------------------------------------


def _tokenize_for_overlap(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_FALLBACK_RE.findall(text or "")]


def _content_set(text: str) -> set:
    tokens = _tokenize_for_overlap(text)
    return {token for token in tokens if len(token) > 2}


def _select_premises_for_claim(
    claim: Claim,
    support_units: Sequence[Any],
    *,
    top_k: int,
    fallback_join: bool,
) -> List[str]:
    """Pick the top-k support unit texts most lexically overlapping with the claim.

    Falls back to a single concatenated premise if no overlap is found and
    ``fallback_join`` is True so the entailment model still sees the full
    context (large concat may exceed the model's context window — caller
    should keep this small).
    """

    claim_terms = _content_set(claim.text)
    if not claim_terms:
        return []

    scored: List[Tuple[float, int, str]] = []
    for idx, unit in enumerate(support_units):
        unit_text = getattr(unit, "text", "") or ""
        if not unit_text:
            continue
        unit_terms = _content_set(unit_text)
        if not unit_terms:
            continue
        overlap = len(claim_terms & unit_terms)
        if overlap == 0:
            continue
        scored.append((-float(overlap), idx, unit_text))

    if not scored:
        if not fallback_join:
            return []
        joined = " ".join(getattr(u, "text", "") or "" for u in support_units if getattr(u, "text", ""))
        return [joined] if joined else []

    scored.sort()
    return [text for _score, _idx, text in scored[: max(1, top_k)]]


# ----------------------------------------------------------------------
# Per-token projection
# ----------------------------------------------------------------------


def _strip_token_marker(token: str) -> str:
    out = token
    for prefix in _STRIP_PREFIXES:
        while out.startswith(prefix):
            out = out[len(prefix):]
    return out


def project_claim_scores_to_tokens(
    response_tokens: Sequence[str],
    response_text: str,
    verifications: Sequence[ClaimVerification],
) -> List[Optional[float]]:
    """Project per-claim NLI scores back onto response tokens.

    Walks the response tokens left-to-right, advancing a character cursor
    through ``response_text`` by stripped token surface, and assigns each
    token the score of whichever claim span contains the cursor. Tokens that
    fall outside any claim span (whitespace-only, special tokens, padding)
    receive ``None``.
    """

    n = len(response_tokens)
    if n == 0 or not response_text:
        return [None] * n

    spans = [(v.claim.char_start, v.claim.char_end, float(v.score)) for v in verifications]
    if not spans:
        return [None] * n

    out: List[Optional[float]] = [None] * n
    cursor = 0
    text_len = len(response_text)
    for idx, token in enumerate(response_tokens):
        surface = _strip_token_marker(token).strip()
        if not surface:
            continue
        match_at = response_text.find(surface, cursor, text_len) if cursor < text_len else -1
        if match_at < 0:
            match_at = response_text.lower().find(surface.lower(), cursor, text_len) if cursor < text_len else -1
        if match_at < 0:
            continue
        cursor = match_at + len(surface)
        for start, end, score in spans:
            if start <= match_at < end:
                out[idx] = score
                break
    return out


# ----------------------------------------------------------------------
# NLI orchestration
# ----------------------------------------------------------------------


def _aggregate_premise_scores(per_premise_scores: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    """Aggregate per-premise NLI triples into ``(entail, neutral, contradict, score)``.

    ``entailment`` and ``contradiction`` take the maximum across premises so
    that any single supporting/refuting premise can carry the claim. ``score``
    is the signed margin ``entailment - contradiction`` clamped to ``[-1, 1]``.
    """

    if not per_premise_scores:
        return 0.0, 0.0, 0.0, 0.0
    entail = max(s[0] for s in per_premise_scores)
    contradict = max(s[2] for s in per_premise_scores)
    neutral = max(s[1] for s in per_premise_scores)
    score = max(-1.0, min(1.0, entail - contradict))
    return entail, neutral, contradict, score


def verify_claims(
    response_text: str,
    support_units: Sequence[Any],
    nli_provider: NLIProvider,
    *,
    max_claims: int = _DEFAULT_MAX_CLAIMS,
    top_k_premises: int = _DEFAULT_TOP_K_PREMISES,
    max_batch: int = _DEFAULT_NLI_MAX_BATCH,
    max_latency_ms: float = _DEFAULT_NLI_MAX_LATENCY_MS,
) -> Tuple[List[ClaimVerification], List[str]]:
    """Run NLI verification on every claim in ``response_text``.

    Returns ``(verifications, warnings)``. Verification is robust: any
    individual claim that fails to resolve premises or whose backend call
    raises is marked ``skipped=True`` rather than aborting the whole batch.
    """

    warnings: List[str] = []
    claims = split_claims(response_text, max_claims=max_claims)
    if not claims:
        return [], warnings

    pairs: List[Tuple[int, str, str]] = []
    claim_premise_index: Dict[int, List[int]] = {}
    selected_premises: Dict[int, List[str]] = {}
    skipped: Dict[int, str] = {}

    for claim in claims:
        premises = _select_premises_for_claim(
            claim,
            support_units,
            top_k=top_k_premises,
            fallback_join=True,
        )
        if not premises:
            skipped[claim.index] = "no_premises"
            continue
        selected_premises[claim.index] = premises
        for premise in premises:
            pair_idx = len(pairs)
            pairs.append((claim.index, premise, claim.text))
            claim_premise_index.setdefault(claim.index, []).append(pair_idx)

    if not pairs:
        return [
            ClaimVerification(
                claim=claim,
                entailment=0.0,
                neutral=0.0,
                contradiction=0.0,
                score=0.0,
                premises=[],
                skipped=True,
                skip_reason=skipped.get(claim.index, "no_premises"),
            )
            for claim in claims
        ], warnings

    triples: List[Tuple[float, float, float]] = []
    started = time.perf_counter()
    try:
        for batch_start in range(0, len(pairs), max(1, max_batch)):
            batch = pairs[batch_start : batch_start + max_batch]
            batch_premises = [premise for _, premise, _ in batch]
            batch_hypotheses = [hypothesis for _, _, hypothesis in batch]
            batch_triples = nli_provider.entail(batch_premises, batch_hypotheses)
            if len(batch_triples) != len(batch):
                raise RuntimeError(
                    "NLI provider returned {got} triples for batch size {expected}".format(
                        got=len(batch_triples), expected=len(batch)
                    )
                )
            triples.extend(batch_triples)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms > max_latency_ms:
                warnings.append(
                    "nli_budget_exceeded: stopped after {ms:.0f} ms (budget {budget:.0f} ms); "
                    "remaining claims fall back to embedding-only scoring".format(
                        ms=elapsed_ms, budget=max_latency_ms
                    )
                )
                break
    except Exception as exc:
        logger.warning("nli_provider_failed", extra={"error": str(exc)})
        warnings.append("nli_unavailable: claim-level verification failed and was skipped")
        return [
            ClaimVerification(
                claim=claim,
                entailment=0.0,
                neutral=0.0,
                contradiction=0.0,
                score=0.0,
                premises=selected_premises.get(claim.index, []),
                skipped=True,
                skip_reason="nli_provider_error",
            )
            for claim in claims
        ], warnings

    completed_pair_count = len(triples)
    verifications: List[ClaimVerification] = []
    for claim in claims:
        if claim.index in skipped:
            verifications.append(
                ClaimVerification(
                    claim=claim,
                    entailment=0.0,
                    neutral=0.0,
                    contradiction=0.0,
                    score=0.0,
                    premises=[],
                    skipped=True,
                    skip_reason=skipped[claim.index],
                )
            )
            continue
        per_premise_scores: List[Tuple[float, float, float]] = []
        for pair_idx in claim_premise_index.get(claim.index, []):
            if pair_idx >= completed_pair_count:
                continue
            per_premise_scores.append(triples[pair_idx])
        if not per_premise_scores:
            verifications.append(
                ClaimVerification(
                    claim=claim,
                    entailment=0.0,
                    neutral=0.0,
                    contradiction=0.0,
                    score=0.0,
                    premises=selected_premises.get(claim.index, []),
                    skipped=True,
                    skip_reason="latency_budget",
                )
            )
            continue
        entail, neutral, contradict, score = _aggregate_premise_scores(per_premise_scores)
        verifications.append(
            ClaimVerification(
                claim=claim,
                entailment=entail,
                neutral=neutral,
                contradiction=contradict,
                score=score,
                premises=selected_premises.get(claim.index, []),
            )
        )
    return verifications, warnings


def aggregate_nli_score(verifications: Sequence[ClaimVerification]) -> Optional[float]:
    """Aggregate per-claim signed scores into a probability-style scalar in (0, 1)."""

    scored = [v for v in verifications if not v.skipped]
    if not scored:
        return None
    mean_signed = sum(v.score for v in scored) / float(len(scored))
    return 0.5 + 0.5 * float(mean_signed)


def fuse_groundedness_v2(
    *,
    reverse_context_calibrated: Optional[float],
    literal_guarded: Optional[float],
    nli_aggregate: Optional[float],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Convex-combination fusion of the three primary peer scores.

    Skips channels whose value is ``None``; remaining weights are renormalized
    so the output is well-defined even when only one or two channels are
    available. Returns ``None`` if no channel can contribute.
    """

    weights = dict(weights) if weights else dict(_DEFAULT_FUSION_WEIGHTS)
    channels = [
        ("calibrated", reverse_context_calibrated, weights.get("calibrated", 0.0)),
        ("literal", literal_guarded, weights.get("literal", 0.0)),
        ("nli", nli_aggregate, weights.get("nli", 0.0)),
    ]
    contributing = [
        (value, max(0.0, weight))
        for _name, value, weight in channels
        if value is not None and weight > 0
    ]
    if not contributing:
        return None
    total_weight = sum(weight for _value, weight in contributing)
    if total_weight <= 0:
        return None
    fused = sum(float(value) * float(weight) for value, weight in contributing) / total_weight
    return max(0.0, min(1.0, float(fused)))


# ----------------------------------------------------------------------
# HuggingFace-backed NLI provider (lazy)
# ----------------------------------------------------------------------


class HuggingFaceNLIProvider:
    """Thin wrapper around a HuggingFace transformers MNLI/ANLI model.

    The model is loaded on the first ``entail`` call and re-used for the
    process lifetime. We assume the model exposes the standard 3-class
    head ordering (contradiction=0, neutral=1, entailment=2 for most
    DeBERTa-MNLI variants); :py:func:`_resolve_label_indices` will look the
    real ordering up from ``id2label`` if present.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        max_length: int = 384,
    ) -> None:
        self.model_id = model_id
        self.device = device or self._default_device()
        self.max_length = int(max_length)
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._label_indices: Optional[Tuple[int, int, int]] = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch  # noqa: WPS433

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
            self._label_indices = self._resolve_label_indices()

    def _resolve_label_indices(self) -> Tuple[int, int, int]:
        id2label = getattr(self._model.config, "id2label", None) or {}
        normalized = {str(idx): str(label).lower() for idx, label in id2label.items()}
        contradiction_idx = neutral_idx = entailment_idx = -1
        for idx_str, label in normalized.items():
            idx = int(idx_str)
            if "contradiction" in label:
                contradiction_idx = idx
            elif "neutral" in label:
                neutral_idx = idx
            elif "entailment" in label:
                entailment_idx = idx
        if contradiction_idx < 0 or neutral_idx < 0 or entailment_idx < 0:
            return 0, 1, 2
        return contradiction_idx, neutral_idx, entailment_idx

    def entail(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> List[Tuple[float, float, float]]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must align")
        if not premises:
            return []
        self._ensure_loaded()
        import torch  # noqa: WPS433

        encoded = self._tokenizer(
            list(premises),
            list(hypotheses),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        if self._label_indices is None:
            self._label_indices = self._resolve_label_indices()
        c_idx, n_idx, e_idx = self._label_indices
        return [
            (
                float(row[e_idx]),
                float(row[n_idx]),
                float(row[c_idx]),
            )
            for row in probs
        ]


# ----------------------------------------------------------------------
# Service-side helpers
# ----------------------------------------------------------------------


def is_enabled() -> bool:
    """Return True when the NLI verifier is enabled via env flag."""

    return os.environ.get("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "").lower() in {"1", "true", "yes"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def resolve_default_provider() -> Optional[NLIProvider]:
    """Build the default HuggingFace NLI provider when enabled.

    Returns ``None`` if the feature flag is off or ``transformers`` is missing.
    Failures inside :py:class:`HuggingFaceNLIProvider` itself are deferred to
    the first entail call so we don't pay model-load cost for unused services.
    """

    if not is_enabled():
        return None
    model_id = os.environ.get(
        "VOYAGER_GROUNDEDNESS_NLI_MODEL",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    )
    try:
        return HuggingFaceNLIProvider(
            model_id=model_id,
            max_length=env_int("VOYAGER_GROUNDEDNESS_NLI_MAX_TOKENS", 384),
        )
    except Exception as exc:
        logger.warning("nli_provider_init_failed", extra={"error": str(exc)})
        return None


def fusion_weights_from_env() -> Dict[str, float]:
    """Read fusion weights from the environment with safe defaults."""

    return {
        "calibrated": env_float("VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED", _DEFAULT_FUSION_WEIGHTS["calibrated"]),
        "literal": env_float("VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL", _DEFAULT_FUSION_WEIGHTS["literal"]),
        "nli": env_float("VOYAGER_GROUNDEDNESS_FUSION_W_NLI", _DEFAULT_FUSION_WEIGHTS["nli"]),
    }


def default_max_claims() -> int:
    return max(1, env_int("VOYAGER_GROUNDEDNESS_NLI_MAX_CLAIMS", _DEFAULT_MAX_CLAIMS))


def default_top_k_premises() -> int:
    return max(1, env_int("VOYAGER_GROUNDEDNESS_NLI_TOP_K", _DEFAULT_TOP_K_PREMISES))


def default_max_batch() -> int:
    return max(1, env_int("VOYAGER_GROUNDEDNESS_NLI_BATCH", _DEFAULT_NLI_MAX_BATCH))


def default_max_latency_ms() -> float:
    return max(1.0, env_float("VOYAGER_GROUNDEDNESS_NLI_LATENCY_MS", _DEFAULT_NLI_MAX_LATENCY_MS))


__all__ = [
    "Claim",
    "ClaimVerification",
    "HuggingFaceNLIProvider",
    "NLIProvider",
    "aggregate_nli_score",
    "default_max_batch",
    "default_max_claims",
    "default_max_latency_ms",
    "default_top_k_premises",
    "fuse_groundedness_v2",
    "fusion_weights_from_env",
    "is_enabled",
    "project_claim_scores_to_tokens",
    "resolve_default_provider",
    "split_claims",
    "verify_claims",
]
