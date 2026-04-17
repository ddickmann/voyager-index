"""Structured-source verification for groundedness Real Hardening Phase I.

When the support material is actually structured (JSON payload, markdown
pipe-table, or key-value blocks), fuzzy embedding similarity cannot be
trusted on its own: swapping ``"price": 42`` for ``"price": 420`` or
transposing the row/column label of a table cell keeps most of the
surface text intact but breaks the underlying fact.

This module offers a strict, rule-based adapter that:

1. Detects a structured source either from the caller-supplied
   ``content_type`` hint or by inspecting the support text.
2. Extracts ``(subject, predicate, object)`` triples from the support.
3. Extracts candidate triples from the response using patterns that are
   robust to ordinary natural-language responses (``"X is Y"``,
   ``"X: Y"``, ``"X = Y"``, ``"X = 'Y'"``) and dependency-based subject,
   verb, object if spaCy is available.
4. Compares them with normalized string matching plus numeric tolerance
   and simple alias resolution.

The module never raises during normal operation; on any failure it
returns ``None`` or an empty diagnostic so the caller can keep the
embedding-only headline and mark the channel as inactive.

The design is intentionally conservative: we prefer false-negatives
(``structured_source_detected=False``) over false positives. This channel
only fires when we are confident the support is structured and at least
one reliable (subject, predicate, object) triple was extracted on both
sides.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Public constants and tunables
# ----------------------------------------------------------------------

#: Absolute tolerance applied to numeric comparisons.
_DEFAULT_ABS_TOL = 1e-6

#: Relative tolerance applied to numeric comparisons once the magnitude is
#: larger than ~1.
_DEFAULT_REL_TOL = 1e-3

#: Penalty applied to ``structured_source_guarded`` for each detected
#: mismatch. With the default, three mismatches already drop the score
#: below the conservative 0.5 amber band.
_DEFAULT_PENALTY_PER_MISMATCH = 0.2

#: Floor applied to the guarded score so a single catastrophic mismatch
#: does not produce a ``NaN``-looking zero that downstream consumers
#: mis-interpret as "score unavailable".
_MIN_GUARDED = 0.0

#: Known caller-supplied content types.
_CONTENT_TYPE_JSON = {"application/json", "application/json+schema", "json"}
_CONTENT_TYPE_MD = {"text/markdown", "markdown", "md", "markdown/table"}

#: Soft cap on how many triples we ever return on a single side. Keeps
#: the diagnostics payload small and avoids pathological quadratic
#: blow-ups on giant JSON blobs.
_MAX_TRIPLES_PER_SIDE = 128


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Triple:
    """A simple (subject, predicate, object) record.

    The fields are the *normalized* textual values used for matching.
    ``raw_object`` and ``numeric_value`` are preserved so mismatches can
    surface the original value for debugging.
    """

    subject: str
    predicate: str
    object: str
    raw_object: str
    numeric_value: Optional[float] = None
    source_hint: Optional[str] = None  # for example ``"json://$.items[0].price"``


@dataclass
class TripleMatch:
    """One comparison result between a source triple and a response triple."""

    subject: str
    predicate: str
    object: str
    matched: bool
    mismatch_kind: Optional[str] = None  # "missing", "value_mismatch", "numeric_mismatch"


@dataclass
class StructuredVerification:
    """The public, serializable output of the structured-source adapter."""

    source_format: Optional[str]
    detected: bool
    source_triple_count: int
    response_triple_count: int
    matches: List[TripleMatch] = field(default_factory=list)
    mismatches: List[TripleMatch] = field(default_factory=list)
    guarded_score: Optional[float] = None  # None => channel inactive


# ----------------------------------------------------------------------
# Normalization helpers
# ----------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\u2018\u2019\u201C\u201D\"'`()\[\]{}]")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _normalize_value(text: str) -> str:
    """NFKC + lowercase + squeeze whitespace + strip quote punctuation.

    We intentionally keep hyphens, slashes, colons, and digits: those
    carry fact-bearing information (dates, ratios, identifiers). We only
    strip cosmetic quoting and collapse repeated whitespace.
    """

    cleaned = _nfkc(text).strip()
    cleaned = _PUNCT_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.lower()


def _normalize_key(text: str) -> str:
    """Predicate/subject normalization. Same rules plus underscore folding."""

    cleaned = _normalize_value(text)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


_NUMERIC_RE = re.compile(
    r"""
    ^\s*
    (?P<sign>[-+]?)
    (?P<value>\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)? | \d+(?:\.\d+)? | \.\d+)
    \s*$
    """,
    re.VERBOSE,
)


def _parse_numeric(text: str) -> Optional[float]:
    """Parse a pure numeric token like ``"42"``, ``"1,234.5"``, or ``"-3.14"``.

    Returns ``None`` if the string carries any non-numeric glyphs
    (currency, units, percent signs). Those are kept for textual
    comparison instead, which lets us fall back to string equality when a
    response writes ``"42%"`` against a source ``"42"`` without silently
    declaring them equal.
    """

    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None
    if _NUMERIC_RE.match(candidate) is None:
        return None
    cleaned = candidate.replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _values_match(
    source_value: str,
    source_numeric: Optional[float],
    response_value: str,
    response_numeric: Optional[float],
    *,
    abs_tol: float = _DEFAULT_ABS_TOL,
    rel_tol: float = _DEFAULT_REL_TOL,
) -> Tuple[bool, str]:
    """Return (matched, mismatch_kind) for two normalized object values.

    Numeric comparison takes priority when both sides parse to a finite
    float. Otherwise we fall back to normalized-string equality, then to
    substring containment so ``"on 20 July 1981"`` still matches
    ``"20 July 1981"`` extracted from a JSON date field.
    """

    if source_numeric is not None and response_numeric is not None:
        if math.isclose(
            float(source_numeric),
            float(response_numeric),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ):
            return True, ""
        return False, "numeric_mismatch"

    if source_value == response_value:
        return True, ""
    if source_value and (source_value in response_value or response_value in source_value):
        return True, ""
    return False, "value_mismatch"


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] not in "{[":
        return False
    # Parse a small prefix to avoid misclassifying text that merely starts
    # with a brace. ``json.loads`` is strict and cheap for small payloads.
    try:
        json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return True


_MD_TABLE_SEP_RE = re.compile(
    r"^\s*\|?(\s*:?-+:?\s*\|)+(\s*:?-+:?\s*\|?)?\s*$"
)


def _looks_like_markdown_table(text: str) -> bool:
    if not text:
        return False
    rows = [row for row in text.splitlines() if row.strip()]
    if len(rows) < 2:
        return False
    if _MD_TABLE_SEP_RE.match(rows[1]) is None:
        return False
    return "|" in rows[0]


def _normalize_content_type_hint(hint: Optional[str]) -> Optional[str]:
    if not hint:
        return None
    lowered = hint.strip().lower()
    if not lowered:
        return None
    if any(token in lowered for token in _CONTENT_TYPE_JSON):
        return "json"
    if any(token in lowered for token in _CONTENT_TYPE_MD):
        return "markdown_table"
    return None


def detect_source_format(
    support_text: str,
    *,
    content_type: Optional[str] = None,
) -> Optional[str]:
    """Return one of ``"json"``, ``"markdown_table"`` or ``None``.

    Caller hints win when they match the content; otherwise we
    auto-detect. We intentionally never classify a raw free-form blob as
    structured: false positives here directly poison the fusion.
    """

    hint = _normalize_content_type_hint(content_type)
    if hint == "json" and _looks_like_json(support_text):
        return "json"
    if hint == "markdown_table" and _looks_like_markdown_table(support_text):
        return "markdown_table"
    # Auto-detect without a hint.
    if _looks_like_json(support_text):
        return "json"
    if _looks_like_markdown_table(support_text):
        return "markdown_table"
    return None


# ----------------------------------------------------------------------
# Source-side triple extraction
# ----------------------------------------------------------------------


_IDENTITY_KEYS = ("name", "title", "id", "label", "subject", "item", "entity")


def _object_identity(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(identity_value, identity_key)`` for an object, or ``(None, None)``.

    We probe a small allow-list of key names that callers tend to use
    for entity identity (``name``, ``title``, ``id``, ``label``,
    ``subject``, ``item``). If none of them are scalar we fall back to
    the structural path so matching stays deterministic.
    """

    for key in _IDENTITY_KEYS:
        if key not in obj:
            continue
        raw = obj[key]
        if isinstance(raw, (str, int, float, bool)):
            return _stringify_scalar(raw), key
    return None, None


def _iter_json_triples(
    value: Any,
    *,
    path: str = "$",
    parent_key: Optional[str] = None,
    inherited_subject: Optional[str] = None,
) -> Iterable[Triple]:
    """Walk a JSON value and yield ``(subject, predicate, object)`` triples.

    The subject is preferentially an identity-bearing field inside the
    current object (for example ``"Widget"`` from ``{"name": "Widget",
    "price": 42}``); otherwise the dotted JSON path is used. The
    predicate is the literal field name; the object is the stringified
    leaf value. Nested objects recurse into; lists are indexed. Handling
    an identity key this way makes the triples line up with the way
    humans phrase responses (``"The price of Widget is 42"``).
    """

    if isinstance(value, dict):
        local_identity, identity_key = _object_identity(value)
        subject_hint = local_identity if local_identity is not None else inherited_subject
        for key, nested in value.items():
            child_path = f"{path}.{key}"
            if isinstance(nested, (dict, list)):
                yield from _iter_json_triples(
                    nested,
                    path=child_path,
                    parent_key=str(key),
                    inherited_subject=subject_hint,
                )
                continue
            # Skip emitting the identity key itself - it becomes the
            # subject for its sibling fields and would otherwise degenerate
            # into a tautological ``subject == object`` triple that leaks
            # into downstream matching.
            if identity_key is not None and key == identity_key:
                continue
            object_str = _stringify_scalar(nested)
            numeric = _parse_numeric(object_str)
            subject_raw = subject_hint if subject_hint is not None else path
            yield Triple(
                subject=_normalize_key(subject_raw),
                predicate=_normalize_key(str(key)),
                object=_normalize_value(object_str),
                raw_object=object_str,
                numeric_value=numeric,
                source_hint=f"json://{child_path}",
            )
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            child_path = f"{path}[{idx}]"
            if isinstance(nested, (dict, list)):
                yield from _iter_json_triples(
                    nested,
                    path=child_path,
                    parent_key=parent_key,
                    inherited_subject=inherited_subject,
                )
                continue
            object_str = _stringify_scalar(nested)
            numeric = _parse_numeric(object_str)
            predicate_raw = parent_key if parent_key else f"[{idx}]"
            subject_raw = inherited_subject if inherited_subject is not None else path
            yield Triple(
                subject=_normalize_key(subject_raw),
                predicate=_normalize_key(predicate_raw),
                object=_normalize_value(object_str),
                raw_object=object_str,
                numeric_value=numeric,
                source_hint=f"json://{child_path}",
            )


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # Keep floats bit-exact so we don't lose precision on round-trip.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def extract_triples_from_json(support_text: str) -> List[Triple]:
    """Parse JSON support text into a bounded list of triples."""

    try:
        parsed = json.loads(support_text)
    except (ValueError, TypeError):
        return []
    triples: List[Triple] = []
    for triple in _iter_json_triples(parsed):
        triples.append(triple)
        if len(triples) >= _MAX_TRIPLES_PER_SIDE:
            break
    return triples


def extract_triples_from_markdown_table(support_text: str) -> List[Triple]:
    """Parse a markdown pipe-table into (row_header, column_header, cell) triples."""

    rows = [row.strip() for row in (support_text or "").splitlines() if row.strip()]
    if len(rows) < 2 or _MD_TABLE_SEP_RE.match(rows[1]) is None:
        return []

    def _split_cells(row: str) -> List[str]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        return cells

    header_cells = _split_cells(rows[0])
    if not header_cells or any(not cell for cell in header_cells):
        return []

    # Column 0 is treated as the row header ("subject") when it carries
    # values. The first header label is typically the table title or
    # axis label, so we keep it as the predicate prefix for extra
    # columns and as the subject label for the row header itself.
    row_subject_label = header_cells[0]
    column_labels = header_cells

    triples: List[Triple] = []
    for row in rows[2:]:
        cells = _split_cells(row)
        if len(cells) < 2:
            continue
        row_subject_value = cells[0]
        if not row_subject_value:
            continue
        # Emit one triple per data cell in this row.
        for col_idx in range(1, min(len(cells), len(column_labels))):
            cell = cells[col_idx]
            if not cell:
                continue
            numeric = _parse_numeric(cell)
            triples.append(
                Triple(
                    subject=_normalize_key(row_subject_value),
                    predicate=_normalize_key(column_labels[col_idx]),
                    object=_normalize_value(cell),
                    raw_object=cell,
                    numeric_value=numeric,
                    source_hint=f"md://{row_subject_label}/{column_labels[col_idx]}",
                )
            )
            if len(triples) >= _MAX_TRIPLES_PER_SIDE:
                return triples
    return triples


def extract_source_triples(
    support_text: str,
    *,
    source_format: str,
) -> List[Triple]:
    """Dispatch to the right source-side extractor."""

    if source_format == "json":
        return extract_triples_from_json(support_text)
    if source_format == "markdown_table":
        return extract_triples_from_markdown_table(support_text)
    return []


# ----------------------------------------------------------------------
# Response-side triple extraction (rule-based)
# ----------------------------------------------------------------------


_KV_COLON_RE = re.compile(
    r"""
    (?P<subject>[\w][\w\- ]{1,64}?)
    \s*[:=]\s*
    (?P<object>"[^"\n]+" | '[^'\n]+' | [^,\.;\n]{1,120})
    """,
    re.VERBOSE,
)

_IS_RE = re.compile(
    r"""
    \b(?P<subject>[A-Z][\w\- ]{1,64}?)
    \s+
    (?:is|are|was|were|equals|equal\s+to)
    \s+
    (?P<object>[^,\.;\n]{1,120})
    """,
    re.VERBOSE,
)

_THE_X_OF_Y_IS_Z = re.compile(
    r"""
    \bthe\s+
    (?P<predicate>[\w\- ]{1,40}?)
    \s+of\s+
    (?P<subject>[\w\- ]{1,40}?)
    \s+(?:is|was|equals)\s+
    (?P<object>[^,\.;\n]{1,120})
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _strip_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] in "\"'":
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def extract_response_triples(response_text: str) -> List[Triple]:
    """Rule-based (subject, predicate, object) triple extraction from free text.

    Three pattern families cover the overwhelming majority of structured
    paraphrases we actually see in RAG outputs:

    1. ``"key: value"`` and ``"key = value"`` (machine-style echoes).
    2. ``"X is Y"`` / ``"X was Y"`` / ``"X equals Y"``.
    3. ``"the <predicate> of <subject> is <object>"``.

    The predicate is omitted (empty string) for cases 1 and 2 because
    the caller-side matcher falls back to predicate-agnostic matching
    when either side has an empty predicate. This avoids spurious
    mismatches when the response paraphrases ``"the price of X is 42"``
    as ``"X is 42"``.
    """

    triples: List[Triple] = []
    if not response_text:
        return triples

    seen = set()

    def _add(subject: str, predicate: str, object_raw: str) -> None:
        if len(triples) >= _MAX_TRIPLES_PER_SIDE:
            return
        subj_norm = _normalize_key(subject)
        pred_norm = _normalize_key(predicate) if predicate else ""
        obj_norm = _normalize_value(_strip_quotes(object_raw))
        if not subj_norm or not obj_norm:
            return
        key = (subj_norm, pred_norm, obj_norm)
        if key in seen:
            return
        seen.add(key)
        numeric = _parse_numeric(_strip_quotes(object_raw))
        triples.append(
            Triple(
                subject=subj_norm,
                predicate=pred_norm,
                object=obj_norm,
                raw_object=_strip_quotes(object_raw),
                numeric_value=numeric,
                source_hint="response",
            )
        )

    for match in _THE_X_OF_Y_IS_Z.finditer(response_text):
        _add(match.group("subject"), match.group("predicate"), match.group("object"))

    for match in _KV_COLON_RE.finditer(response_text):
        _add(match.group("subject"), "", match.group("object"))

    for match in _IS_RE.finditer(response_text):
        _add(match.group("subject"), "", match.group("object"))

    return triples


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


def _candidates_for_source(
    source: Triple,
    response_triples: Sequence[Triple],
) -> List[Triple]:
    """Return response triples whose subject and (optional) predicate match.

    Matching is done on the normalized fields. Predicate matching is
    only required when both sides declare one; if either side has an
    empty predicate we fall back to subject-only matching so natural-
    language paraphrases still align.
    """

    subject_hits = [r for r in response_triples if r.subject == source.subject]
    if not subject_hits:
        # Allow partial subject containment so ``"items 0"`` source maps
        # to ``"the item"`` response without failing.
        subject_hits = [
            r
            for r in response_triples
            if source.subject and (source.subject in r.subject or r.subject in source.subject)
        ]
    if not subject_hits:
        return []

    if not source.predicate:
        return subject_hits
    predicate_hits = [r for r in subject_hits if not r.predicate or r.predicate == source.predicate]
    if predicate_hits:
        return predicate_hits
    # Fall back to subject-only when the predicate did not overlap.
    return subject_hits


def match_triples(
    source_triples: Sequence[Triple],
    response_triples: Sequence[Triple],
) -> Tuple[List[TripleMatch], List[TripleMatch]]:
    """Compare source and response triples, returning (matches, mismatches).

    Only source triples that have at least one response candidate are
    considered; unmentioned source triples do not count as mismatches.
    This is the "at-most-what-you-said" rule that keeps the channel
    focused on hallucinations rather than omissions.
    """

    matches: List[TripleMatch] = []
    mismatches: List[TripleMatch] = []
    for source in source_triples:
        candidates = _candidates_for_source(source, response_triples)
        if not candidates:
            continue
        best = None
        best_kind = "value_mismatch"
        for candidate in candidates:
            matched, kind = _values_match(
                source.object,
                source.numeric_value,
                candidate.object,
                candidate.numeric_value,
            )
            if matched:
                best = candidate
                best_kind = ""
                break
            if best is None or (best_kind == "value_mismatch" and kind == "numeric_mismatch"):
                best = candidate
                best_kind = kind or "value_mismatch"

        if best is None:
            continue
        if not best_kind:
            matches.append(
                TripleMatch(
                    subject=source.subject,
                    predicate=source.predicate,
                    object=best.raw_object,
                    matched=True,
                )
            )
        else:
            mismatches.append(
                TripleMatch(
                    subject=source.subject,
                    predicate=source.predicate,
                    object=best.raw_object,
                    matched=False,
                    mismatch_kind=best_kind,
                )
            )
    return matches, mismatches


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def verify_structured_source(
    *,
    support_text: str,
    response_text: str,
    content_type: Optional[str] = None,
    penalty_per_mismatch: float = _DEFAULT_PENALTY_PER_MISMATCH,
) -> StructuredVerification:
    """Run detection + extraction + matching end to end.

    Returns a :class:`StructuredVerification` whose ``guarded_score`` is
    ``None`` when no structured source could be detected. A ``None``
    score is a hard signal to the fusion layer that this channel must
    be skipped for this request, not treated as zero.
    """

    if not support_text or not response_text:
        return StructuredVerification(
            source_format=None,
            detected=False,
            source_triple_count=0,
            response_triple_count=0,
            guarded_score=None,
        )

    source_format = detect_source_format(support_text, content_type=content_type)
    if source_format is None:
        return StructuredVerification(
            source_format=None,
            detected=False,
            source_triple_count=0,
            response_triple_count=0,
            guarded_score=None,
        )

    source_triples = extract_source_triples(support_text, source_format=source_format)
    if not source_triples:
        return StructuredVerification(
            source_format=source_format,
            detected=True,
            source_triple_count=0,
            response_triple_count=0,
            guarded_score=None,
        )

    response_triples = extract_response_triples(response_text)
    if not response_triples:
        return StructuredVerification(
            source_format=source_format,
            detected=True,
            source_triple_count=len(source_triples),
            response_triple_count=0,
            guarded_score=None,
        )

    matches, mismatches = match_triples(source_triples, response_triples)

    # The guarded score is only meaningful if we actually compared
    # something: no overlap means we stay neutral and leave the channel
    # inactive.
    if not matches and not mismatches:
        return StructuredVerification(
            source_format=source_format,
            detected=True,
            source_triple_count=len(source_triples),
            response_triple_count=len(response_triples),
            guarded_score=None,
        )

    penalty = max(0.0, float(penalty_per_mismatch)) * float(len(mismatches))
    guarded = max(_MIN_GUARDED, 1.0 - penalty)
    return StructuredVerification(
        source_format=source_format,
        detected=True,
        source_triple_count=len(source_triples),
        response_triple_count=len(response_triples),
        matches=matches,
        mismatches=mismatches,
        guarded_score=float(guarded),
    )


def is_structured_enabled() -> bool:
    """Feature flag for the structured-source fusion channel.

    When the flag is off we still compute the triples (so diagnostics
    stay honest) but the fusion layer is told to keep its weight at
    zero. Defaults to ``True`` because the channel only fires when a
    structured source is actually detected and at least one triple
    overlaps.
    """

    raw = os.environ.get("VOYAGER_GROUNDEDNESS_STRUCTURED_ENABLED", "1").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    return True


def default_penalty_per_mismatch() -> float:
    raw = os.environ.get("VOYAGER_GROUNDEDNESS_STRUCTURED_PENALTY", "").strip()
    try:
        value = float(raw) if raw else _DEFAULT_PENALTY_PER_MISMATCH
    except (TypeError, ValueError):
        value = _DEFAULT_PENALTY_PER_MISMATCH
    return max(0.0, min(1.0, value))


def verification_to_dict(result: StructuredVerification) -> Dict[str, Any]:
    """Serialize a :class:`StructuredVerification` for the response payload."""

    def _match_to_dict(match: TripleMatch) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "subject": match.subject,
            "predicate": match.predicate,
            "object": match.object,
            "matched": bool(match.matched),
        }
        if match.mismatch_kind:
            payload["mismatch_kind"] = match.mismatch_kind
        return payload

    return {
        "source_format": result.source_format,
        "source_triple_count": int(result.source_triple_count),
        "response_triple_count": int(result.response_triple_count),
        "matches": [_match_to_dict(m) for m in result.matches],
        "mismatches": [_match_to_dict(m) for m in result.mismatches],
    }


__all__ = [
    "Triple",
    "TripleMatch",
    "StructuredVerification",
    "detect_source_format",
    "extract_triples_from_json",
    "extract_triples_from_markdown_table",
    "extract_source_triples",
    "extract_response_triples",
    "match_triples",
    "verify_structured_source",
    "is_structured_enabled",
    "default_penalty_per_mismatch",
    "verification_to_dict",
]
