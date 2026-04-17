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
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from voyager_index._internal.server.api.groundedness_claims import (
    AtomicClaim,
    decompose_sentence_into_atoms,
    is_atomic_enabled,
)

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
    "semantic_entropy": 0.0,
    "structured": 0.0,
}

_DEFAULT_MAX_CLAIMS = 16
_DEFAULT_TOP_K_PREMISES = 3
_DEFAULT_NLI_MAX_BATCH = 16
_DEFAULT_NLI_MAX_LATENCY_MS = 2000.0
_DEFAULT_PREMISE_CONCAT_BUDGET = 384  # tokens approximated as words

_PREMISE_JOIN_SEPARATOR = " \u2022 "  # bullet keeps sentence boundaries visible

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
class AtomicVerification:
    """Per-atom NLI score within a parent claim."""

    atom: AtomicClaim
    entailment: float
    neutral: float
    contradiction: float
    score: float
    premises: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None


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
    atoms: List[AtomicVerification] = field(default_factory=list)


class NLIProvider(Protocol):
    """Minimal contract for an entailment backend.

    Implementations receive aligned premise/hypothesis pairs and must return
    ``(entail_p, neutral_p, contradict_p)`` triples summing to roughly ``1``.
    """

    def entail(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> List[Tuple[float, float, float]]:
        ...


class PremiseReranker(Protocol):
    """Score a sequence of (claim, candidate-premise) pairs.

    Implementations return one float per pair; higher = more entailment-relevant.
    They must be deterministic for a given input.
    """

    def score(
        self, claim: str, candidate_premises: Sequence[str]
    ) -> List[float]:
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


def _candidate_premise_texts(support_units: Sequence[Any]) -> List[str]:
    """Pull deduplicated, non-empty premise texts out of support units."""

    seen: set = set()
    out: List[str] = []
    for unit in support_units:
        text = getattr(unit, "text", "") or ""
        if not text:
            continue
        # Some support units repeat verbatim across batches — dedupe to
        # avoid feeding identical premises to the cross-encoder.
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _lexical_rank(claim_text: str, candidate_texts: Sequence[str]) -> List[Tuple[float, int, str]]:
    """Lexical overlap ranking; higher count first, stable on idx."""

    claim_terms = _content_set(claim_text)
    if not claim_terms:
        return []
    scored: List[Tuple[float, int, str]] = []
    for idx, text in enumerate(candidate_texts):
        text_terms = _content_set(text)
        if not text_terms:
            continue
        overlap = len(claim_terms & text_terms)
        if overlap == 0:
            continue
        scored.append((-float(overlap), idx, text))
    scored.sort()
    return scored


def _select_premises_for_claim(
    claim: Any,
    support_units: Sequence[Any],
    *,
    top_k: int,
    fallback_join: bool,
    reranker: Optional[PremiseReranker] = None,
) -> List[str]:
    """Pick the top-k support unit texts most likely to entail the claim.

    Selection priority:

    1. Cross-encoder reranker (if provided) — direct ``(claim, premise)``
       relevance score; falls back to lexical if reranker call fails.
    2. Lexical content-token overlap — same heuristic shipped in Phase D.
    3. Concatenated fallback when neither method finds any candidate.

    The function accepts either ``Claim`` or ``AtomicClaim`` so callers can
    reuse the same selector for atomic-fact decomposition.
    """

    claim_text = getattr(claim, "text", "") or ""
    if not claim_text:
        return []

    candidate_texts = _candidate_premise_texts(support_units)
    if not candidate_texts:
        return []

    if reranker is not None:
        try:
            scores = reranker.score(claim_text, candidate_texts)
        except Exception as exc:  # noqa: BLE001 — fall back deterministically
            logger.warning("premise_reranker_failed", extra={"error": str(exc)})
            scores = []
        if scores and len(scores) == len(candidate_texts):
            indexed = sorted(
                ((-float(score), idx) for idx, score in enumerate(scores)),
                key=lambda item: (item[0], item[1]),
            )
            ranked = [candidate_texts[idx] for _score, idx in indexed]
            return ranked[: max(1, top_k)]

    lexical = _lexical_rank(claim_text, candidate_texts)
    if lexical:
        return [text for _score, _idx, text in lexical[: max(1, top_k)]]

    if not fallback_join:
        return []
    joined = " ".join(candidate_texts)
    return [joined] if joined else []


def is_premise_concat_enabled() -> bool:
    """Return True when multi-premise concatenated NLI is enabled."""

    raw = os.environ.get("VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT", "1")
    return raw.lower() in {"1", "true", "yes"}


def _concat_premises_for_nli(
    premises: Sequence[str],
    *,
    token_budget: int,
    separator: str = _PREMISE_JOIN_SEPARATOR,
) -> str:
    """Concatenate top-k premises into one composite premise within budget.

    ``token_budget`` is approximated by whitespace word count to avoid a
    second tokenizer dependency. Premises preserve their order; if the
    budget is exceeded the trailing premise is truncated word-wise rather
    than dropped, so the most relevant premise (rank 1) is always retained.
    """

    if not premises:
        return ""
    if token_budget <= 0:
        return separator.join(premises)
    out_words: List[str] = []
    used = 0
    sep_words = len(separator.split())
    for idx, premise in enumerate(premises):
        if not premise:
            continue
        words = premise.split()
        if not words:
            continue
        budget_left = token_budget - used - (sep_words if out_words else 0)
        if budget_left <= 0:
            break
        if len(words) > budget_left:
            words = words[:budget_left]
        if out_words:
            out_words.extend(separator.split())
            used += sep_words
        out_words.extend(words)
        used += len(words)
        if used >= token_budget:
            break
        _ = idx  # noqa: F841 — cursor only
    return " ".join(out_words)


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


def _atom_min_aggregate(atoms: Sequence[AtomicVerification]) -> Tuple[float, float, float, float]:
    """Aggregate atom-level NLI scores into a single (entail, neutral, contradict, score).

    The aggregation is conservative: ``entail`` takes the **min** across atoms
    (one unsupported atom drags the parent claim down) while ``contradict``
    takes the **max** (any single contradictory atom flags the parent).
    ``score`` recomputes the entail-contradict margin clipped to ``[-1, 1]``.
    """

    scored = [a for a in atoms if not a.skipped]
    if not scored:
        return 0.0, 0.0, 0.0, 0.0
    entail = min(a.entailment for a in scored)
    contradict = max(a.contradiction for a in scored)
    neutral = max(a.neutral for a in scored)
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
    reranker: Optional[PremiseReranker] = None,
    concat_premises: Optional[bool] = None,
    premise_concat_word_budget: int = _DEFAULT_PREMISE_CONCAT_BUDGET,
    use_atomic_claims: Optional[bool] = None,
) -> Tuple[List[ClaimVerification], List[str]]:
    """Run NLI verification on every claim in ``response_text``.

    Phase F upgrades the original Phase D pipeline with three orthogonal
    levers. Each is independently toggleable so we can A/B them.

    - ``reranker``: cross-encoder ``(claim, premise)`` reranker. When
      provided it replaces the lexical premise selector with a much
      stronger entailment-relevance signal. Falls back to lexical
      selection if the reranker call raises.
    - ``concat_premises``: when True the top-k premises are concatenated
      into one composite premise per claim (single NLI call per claim
      instead of k). When False each premise is scored individually and
      aggregated via ``_aggregate_premise_scores`` (Phase D behavior).
    - ``use_atomic_claims``: when True every parent sentence is decomposed
      into atomic propositions; each atom is verified independently and
      aggregated to the parent claim with conservative ``min``-of-entail.

    Returns ``(verifications, warnings)``. Verification is robust: any
    individual claim that fails to resolve premises or whose backend call
    raises is marked ``skipped=True`` rather than aborting the whole batch.
    """

    warnings: List[str] = []
    claims = split_claims(response_text, max_claims=max_claims)
    if not claims:
        return [], warnings

    if concat_premises is None:
        concat_premises = is_premise_concat_enabled()
    if use_atomic_claims is None:
        use_atomic_claims = is_atomic_enabled()

    # -------- Build the atom graph (one entry per parent claim) -------- #

    atoms_per_claim: Dict[int, List[AtomicClaim]] = {}
    if use_atomic_claims:
        for claim in claims:
            atoms = decompose_sentence_into_atoms(
                claim.text,
                parent_index=claim.index,
                parent_start=claim.char_start,
            )
            atoms_per_claim[claim.index] = atoms or [
                AtomicClaim(
                    parent_index=claim.index,
                    atom_index=0,
                    text=claim.text,
                    char_start=claim.char_start,
                    char_end=claim.char_end,
                )
            ]
    else:
        for claim in claims:
            atoms_per_claim[claim.index] = [
                AtomicClaim(
                    parent_index=claim.index,
                    atom_index=0,
                    text=claim.text,
                    char_start=claim.char_start,
                    char_end=claim.char_end,
                )
            ]

    pairs: List[Tuple[int, int, str, str]] = []  # (claim_idx, atom_idx, premise, hypothesis)
    selected_premises: Dict[Tuple[int, int], List[str]] = {}
    skipped: Dict[Tuple[int, int], str] = {}

    for claim in claims:
        for atom in atoms_per_claim[claim.index]:
            premises = _select_premises_for_claim(
                atom,
                support_units,
                top_k=top_k_premises,
                fallback_join=True,
                reranker=reranker,
            )
            if not premises:
                skipped[(claim.index, atom.atom_index)] = "no_premises"
                continue
            selected_premises[(claim.index, atom.atom_index)] = premises
            if concat_premises:
                composite = _concat_premises_for_nli(
                    premises,
                    token_budget=premise_concat_word_budget,
                )
                if not composite:
                    skipped[(claim.index, atom.atom_index)] = "no_premises"
                    continue
                pairs.append((claim.index, atom.atom_index, composite, atom.text))
            else:
                for premise in premises:
                    pairs.append((claim.index, atom.atom_index, premise, atom.text))

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
                skip_reason="no_premises",
                atoms=[
                    AtomicVerification(
                        atom=atom,
                        entailment=0.0,
                        neutral=0.0,
                        contradiction=0.0,
                        score=0.0,
                        premises=[],
                        skipped=True,
                        skip_reason=skipped.get((claim.index, atom.atom_index), "no_premises"),
                    )
                    for atom in atoms_per_claim[claim.index]
                ],
            )
            for claim in claims
        ], warnings

    triples: List[Tuple[float, float, float]] = []
    started = time.perf_counter()
    try:
        for batch_start in range(0, len(pairs), max(1, max_batch)):
            batch = pairs[batch_start : batch_start + max_batch]
            batch_premises = [premise for _, _, premise, _ in batch]
            batch_hypotheses = [hypothesis for _, _, _, hypothesis in batch]
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
                premises=selected_premises.get((claim.index, 0), []),
                skipped=True,
                skip_reason="nli_provider_error",
                atoms=[
                    AtomicVerification(
                        atom=atom,
                        entailment=0.0,
                        neutral=0.0,
                        contradiction=0.0,
                        score=0.0,
                        premises=selected_premises.get((claim.index, atom.atom_index), []),
                        skipped=True,
                        skip_reason="nli_provider_error",
                    )
                    for atom in atoms_per_claim[claim.index]
                ],
            )
            for claim in claims
        ], warnings

    # -------- Stitch triples back to (claim, atom) -------- #

    completed_pair_count = len(triples)
    atom_pair_index: Dict[Tuple[int, int], List[int]] = {}
    for pair_idx, (claim_idx, atom_idx, _premise, _hypothesis) in enumerate(pairs):
        atom_pair_index.setdefault((claim_idx, atom_idx), []).append(pair_idx)

    verifications: List[ClaimVerification] = []
    for claim in claims:
        atom_records: List[AtomicVerification] = []
        for atom in atoms_per_claim[claim.index]:
            atom_key = (claim.index, atom.atom_index)
            if atom_key in skipped:
                atom_records.append(
                    AtomicVerification(
                        atom=atom,
                        entailment=0.0,
                        neutral=0.0,
                        contradiction=0.0,
                        score=0.0,
                        premises=[],
                        skipped=True,
                        skip_reason=skipped[atom_key],
                    )
                )
                continue
            per_premise_scores: List[Tuple[float, float, float]] = []
            for pair_idx in atom_pair_index.get(atom_key, []):
                if pair_idx >= completed_pair_count:
                    continue
                per_premise_scores.append(triples[pair_idx])
            if not per_premise_scores:
                atom_records.append(
                    AtomicVerification(
                        atom=atom,
                        entailment=0.0,
                        neutral=0.0,
                        contradiction=0.0,
                        score=0.0,
                        premises=selected_premises.get(atom_key, []),
                        skipped=True,
                        skip_reason="latency_budget",
                    )
                )
                continue
            entail, neutral, contradict, score = _aggregate_premise_scores(per_premise_scores)
            atom_records.append(
                AtomicVerification(
                    atom=atom,
                    entailment=entail,
                    neutral=neutral,
                    contradiction=contradict,
                    score=score,
                    premises=selected_premises.get(atom_key, []),
                )
            )
        if not atom_records or all(a.skipped for a in atom_records):
            verifications.append(
                ClaimVerification(
                    claim=claim,
                    entailment=0.0,
                    neutral=0.0,
                    contradiction=0.0,
                    score=0.0,
                    premises=selected_premises.get((claim.index, 0), []),
                    skipped=True,
                    skip_reason=(
                        atom_records[0].skip_reason
                        if atom_records else "no_premises"
                    ),
                    atoms=atom_records,
                )
            )
            continue
        # Aggregate atoms to parent claim with conservative ``min``-of-entail.
        if len(atom_records) == 1 and not atom_records[0].skipped:
            base = atom_records[0]
            premises = base.premises
            entail, neutral, contradict, score = (
                base.entailment,
                base.neutral,
                base.contradiction,
                base.score,
            )
        else:
            entail, neutral, contradict, score = _atom_min_aggregate(atom_records)
            # Surface the union of all premises that fed this claim's atoms.
            seen_premise: set = set()
            premises = []
            for a in atom_records:
                for p in a.premises:
                    if p in seen_premise:
                        continue
                    seen_premise.add(p)
                    premises.append(p)
        verifications.append(
            ClaimVerification(
                claim=claim,
                entailment=entail,
                neutral=neutral,
                contradiction=contradict,
                score=score,
                premises=premises,
                atoms=atom_records,
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
    semantic_entropy: Optional[float] = None,
    structured_source_guarded: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Convex-combination fusion of the available peer scores.

    Skips channels whose value is ``None`` or whose weight is ``0``;
    remaining weights are renormalized so the output is well-defined even
    when only one channel is available. Returns ``None`` if no channel
    can contribute.

    The semantic-entropy and structured-source channels are opt-in: their
    default weights are ``0`` and they renormalize the others when their
    value is provided and a non-zero weight has been configured.
    """

    weights = dict(weights) if weights else dict(_DEFAULT_FUSION_WEIGHTS)
    channels = [
        ("calibrated", reverse_context_calibrated, weights.get("calibrated", 0.0)),
        ("literal", literal_guarded, weights.get("literal", 0.0)),
        ("nli", nli_aggregate, weights.get("nli", 0.0)),
        ("semantic_entropy", semantic_entropy, weights.get("semantic_entropy", 0.0)),
        ("structured", structured_source_guarded, weights.get("structured", 0.0)),
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
# Cross-encoder premise reranker (lazy)
# ----------------------------------------------------------------------


class CrossEncoderPremiseReranker:
    """Score (claim, premise) pairs with a HuggingFace cross-encoder reranker.

    The default model is ``BAAI/bge-reranker-v2-m3`` (off-the-shelf, no
    weights trained at runtime). Loading is lazy and thread-safe; failures
    leave the reranker silently inert so the surrounding pipeline falls
    back to lexical selection.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.device = device or self._default_device()
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._unavailable = False

    @staticmethod
    def _default_device() -> str:
        try:
            import torch  # noqa: WPS433

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_loaded(self) -> bool:
        if self._unavailable:
            return False
        if self._model is not None and self._tokenizer is not None:
            return True
        with self._lock:
            if self._unavailable:
                return False
            if self._model is not None and self._tokenizer is not None:
                return True
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: WPS433

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
                self._model.to(self.device)
                self._model.eval()
                return True
            except Exception as exc:
                logger.warning(
                    "premise_reranker_init_failed",
                    extra={"model": self.model_id, "error": str(exc)},
                )
                self._unavailable = True
                return False

    def score(self, claim: str, candidate_premises: Sequence[str]) -> List[float]:
        if not candidate_premises:
            return []
        if not self._ensure_loaded():
            return []
        import torch  # noqa: WPS433

        scores: List[float] = []
        for batch_start in range(0, len(candidate_premises), self.batch_size):
            batch = candidate_premises[batch_start : batch_start + self.batch_size]
            pairs = [[claim, premise] for premise in batch]
            encoded = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
            # bge-reranker-v2-m3 uses 1-class regression head where the
            # logit is the relevance score; some 2-class rerankers use
            # softmax(positive). Handle both shapes.
            if logits.shape[-1] == 1:
                values = logits.squeeze(-1).float().cpu().tolist()
            else:
                probs = torch.softmax(logits, dim=-1)
                values = probs[:, -1].float().cpu().tolist()
            scores.extend(float(v) for v in values)
        return scores


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
        "semantic_entropy": env_float(
            "VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY",
            _DEFAULT_FUSION_WEIGHTS["semantic_entropy"],
        ),
        "structured": env_float(
            "VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED",
            _DEFAULT_FUSION_WEIGHTS["structured"],
        ),
    }


def resolve_default_reranker() -> Optional[PremiseReranker]:
    """Build the default cross-encoder reranker when configured.

    Returns ``None`` if no model is configured. The model is loaded lazily
    on the first ``score`` call so unused services pay no cost.
    """

    model_id = os.environ.get("VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL")
    if not model_id:
        return None
    try:
        return CrossEncoderPremiseReranker(
            model_id=model_id,
            max_length=env_int("VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MAX_TOKENS", 512),
            batch_size=env_int("VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_BATCH", 32),
        )
    except Exception as exc:
        logger.warning("premise_reranker_resolve_failed", extra={"error": str(exc)})
        return None


def default_premise_concat_word_budget() -> int:
    return max(64, env_int("VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT_BUDGET", _DEFAULT_PREMISE_CONCAT_BUDGET))


def default_max_claims() -> int:
    return max(1, env_int("VOYAGER_GROUNDEDNESS_NLI_MAX_CLAIMS", _DEFAULT_MAX_CLAIMS))


def default_top_k_premises() -> int:
    return max(1, env_int("VOYAGER_GROUNDEDNESS_NLI_TOP_K", _DEFAULT_TOP_K_PREMISES))


def default_max_batch() -> int:
    return max(1, env_int("VOYAGER_GROUNDEDNESS_NLI_BATCH", _DEFAULT_NLI_MAX_BATCH))


def default_max_latency_ms() -> float:
    return max(1.0, env_float("VOYAGER_GROUNDEDNESS_NLI_LATENCY_MS", _DEFAULT_NLI_MAX_LATENCY_MS))


__all__ = [
    "AtomicVerification",
    "Claim",
    "ClaimVerification",
    "CrossEncoderPremiseReranker",
    "HuggingFaceNLIProvider",
    "NLIProvider",
    "PremiseReranker",
    "aggregate_nli_score",
    "default_max_batch",
    "default_max_claims",
    "default_max_latency_ms",
    "default_premise_concat_word_budget",
    "default_top_k_premises",
    "fuse_groundedness_v2",
    "fusion_weights_from_env",
    "is_atomic_enabled",
    "is_enabled",
    "is_premise_concat_enabled",
    "project_claim_scores_to_tokens",
    "resolve_default_provider",
    "resolve_default_reranker",
    "split_claims",
    "verify_claims",
]
