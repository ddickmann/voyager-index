"""Atomic-fact decomposition for the Groundedness NLI lane (Phase F3).

This module provides a deterministic, rule-based atomic-claim splitter that
breaks a sentence into single-predicate atomic propositions. The splitter
prefers spaCy dependency parsing when ``en_core_web_sm`` is available and
falls back to a regex-based approximation otherwise.

Design principles:

- **Deterministic.** Two calls on the same input always produce the same
  atomic sequence, so harness reports stay reproducible.
- **Conservative.** When in doubt, emit the original sentence as a single
  atom — atomic decomposition that drops content is worse than no
  decomposition at all.
- **Offset-preserving.** Each atom keeps its character offsets in the
  source text so per-token NLI projection remains correct.
- **No trained weights.** Pure dependency / regex rules. Optional spaCy
  pipeline is the off-the-shelf ``en_core_web_sm`` checkpoint.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------


@dataclass
class AtomicClaim:
    """A single-predicate atomic proposition extracted from a sentence."""

    parent_index: int
    atom_index: int
    text: str
    char_start: int
    char_end: int


# ----------------------------------------------------------------------
# Regex-based splitter (always available)
# ----------------------------------------------------------------------

# Coordinating conjunctions to split on at the clause level. These are
# intentionally conservative: only conjunctions clearly separating two
# independent assertions are honored. We keep "or" and short clauses
# attached to avoid over-splitting numeric ranges or alternatives.
_COORD_PATTERN = re.compile(
    r"\s*(?:,\s+(?:and|but|while|whereas)\s+|;\s+|\s+(?:and|but|while|whereas)\s+)",
    re.IGNORECASE,
)

# Relative-clause introducers we will optionally split on when the parent
# clause is otherwise complete. We avoid restrictive "that"-clauses to keep
# noun-phrase modifiers attached.
_RELATIVE_PATTERN = re.compile(
    r",\s+(?:which|who|where|when)\s+",
    re.IGNORECASE,
)

# Parenthetical insertions ", X, " that look appositive — heuristic split.
_APPOSITIVE_PATTERN = re.compile(
    r",\s+(?:also\s+known\s+as|aka|i\.e\.,?|e\.g\.,?)\s+",
    re.IGNORECASE,
)

_SUBJECT_HINT_RE = re.compile(r"^(?:and|but|while|whereas|which|who|where|when)\s+", re.IGNORECASE)
_MIN_ATOM_LEN = 8


def _looks_independent(fragment: str) -> bool:
    """Heuristic: does ``fragment`` look like a standalone proposition?"""

    fragment = fragment.strip()
    if len(fragment) < _MIN_ATOM_LEN:
        return False
    # If the fragment starts with a conjunction or relative pronoun, it
    # likely needs the parent clause's subject to be standalone.
    if _SUBJECT_HINT_RE.match(fragment):
        return False
    return True


def _regex_split(sentence: str, parent_start: int) -> List[Tuple[int, int, str]]:
    """Split a sentence on coordinations / relative clauses / appositives.

    Returns a list of ``(absolute_start, absolute_end, text)`` tuples whose
    union covers the input sentence (after trimming whitespace).
    """

    if not sentence or not sentence.strip():
        return []

    pieces: List[Tuple[int, int, str]] = []
    cursor = 0
    text = sentence

    def _slice_at(positions: Sequence[int]) -> List[Tuple[int, str]]:
        if not positions:
            return [(0, text)]
        ordered = sorted(set(positions))
        out: List[Tuple[int, str]] = []
        last = 0
        for pos in ordered:
            chunk = text[last:pos]
            if chunk.strip():
                out.append((last, chunk))
            last = pos
        tail = text[last:]
        if tail.strip():
            out.append((last, tail))
        return out

    coord_starts = [match.end() for match in _COORD_PATTERN.finditer(text)]
    rel_starts = [match.end() for match in _RELATIVE_PATTERN.finditer(text)]
    appo_starts = [match.end() for match in _APPOSITIVE_PATTERN.finditer(text)]
    candidate_positions = coord_starts + rel_starts + appo_starts

    fragments = _slice_at(candidate_positions)

    # Drop the leading conjunction/relative tokens from each fragment.
    cleaned: List[Tuple[int, str]] = []
    for offset, frag in fragments:
        clean = _SUBJECT_HINT_RE.sub("", frag).strip()
        if not clean:
            continue
        new_start = offset + (frag.find(clean) if clean in frag else 0)
        cleaned.append((new_start, clean))

    if len(cleaned) <= 1:
        # Nothing meaningful to split on — return the whole sentence.
        s = sentence.strip()
        offset = sentence.find(s)
        return [(parent_start + max(0, offset), parent_start + max(0, offset) + len(s), s)]

    # Each cleaned fragment must look independent; otherwise stitch back to
    # the previous one. This keeps relative clauses without explicit
    # subjects attached.
    merged: List[Tuple[int, str]] = []
    for offset, frag in cleaned:
        if merged and not _looks_independent(frag):
            prev_offset, prev_text = merged[-1]
            merged[-1] = (prev_offset, (prev_text + " " + frag).strip())
            continue
        merged.append((offset, frag))

    if len(merged) <= 1:
        s = sentence.strip()
        offset = sentence.find(s)
        return [(parent_start + max(0, offset), parent_start + max(0, offset) + len(s), s)]

    pieces = []
    for offset, frag in merged:
        start = parent_start + offset
        end = start + len(frag)
        pieces.append((start, end, frag))

    return pieces


# ----------------------------------------------------------------------
# spaCy-based splitter (when available)
# ----------------------------------------------------------------------


class _SpacySplitter:
    """Lazy wrapper around an off-the-shelf spaCy English pipeline.

    We only use the dependency parser to identify clause boundaries; no
    weights are trained at runtime. The pipeline is loaded once per
    process and re-used.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._lock = threading.Lock()
        self._nlp = None
        self._unavailable = False

    def _ensure_loaded(self) -> bool:
        if self._unavailable:
            return False
        if self._nlp is not None:
            return True
        with self._lock:
            if self._unavailable:
                return False
            if self._nlp is not None:
                return True
            try:
                import spacy  # noqa: WPS433

                self._nlp = spacy.load(self.model_name, disable=["ner", "lemmatizer"])
                return True
            except Exception as exc:  # broad: spaCy missing, model missing, etc.
                logger.info(
                    "atomic_claims_spacy_unavailable",
                    extra={"model": self.model_name, "error": str(exc)},
                )
                self._unavailable = True
                return False

    def split(self, sentence: str, parent_start: int) -> List[Tuple[int, int, str]]:
        if not self._ensure_loaded():
            return _regex_split(sentence, parent_start)

        doc = self._nlp(sentence)
        # Walk root verbs / clausal heads. Each head plus its dependent
        # subtree spans an atomic clause. We enumerate clausal heads by
        # checking ``conj``, ``ccomp``, ``advcl``, ``relcl`` deps.
        clause_heads = []
        for token in doc:
            if token.dep_ in {"ROOT"}:
                clause_heads.append(token)
            elif token.dep_ in {"conj", "ccomp", "advcl", "relcl"} and token.pos_ in {"VERB", "AUX"}:
                clause_heads.append(token)

        if len(clause_heads) <= 1:
            return _regex_split(sentence, parent_start)

        clause_spans: List[Tuple[int, int]] = []
        for head in clause_heads:
            indices = [t.i for t in head.subtree]
            if not indices:
                continue
            min_i = min(indices)
            max_i = max(indices)
            start_char = doc[min_i].idx
            end_char = doc[max_i].idx + len(doc[max_i].text)
            clause_spans.append((start_char, end_char))

        if not clause_spans:
            return _regex_split(sentence, parent_start)

        clause_spans.sort()

        # Merge overlapping spans (e.g. nested ccomp inside ROOT).
        merged: List[Tuple[int, int]] = []
        for start, end in clause_spans:
            if merged and start <= merged[-1][1]:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
                continue
            merged.append((start, end))

        # spaCy gives us nested clauses; we want the outer-most independent
        # atoms. Drop any span fully contained in another.
        outer: List[Tuple[int, int]] = []
        for start, end in merged:
            contained = False
            for o_start, o_end in outer:
                if o_start <= start and end <= o_end:
                    contained = True
                    break
            if not contained:
                outer.append((start, end))

        if len(outer) <= 1:
            return _regex_split(sentence, parent_start)

        atoms: List[Tuple[int, int, str]] = []
        for start, end in outer:
            text = sentence[start:end].strip()
            if not text or len(text) < _MIN_ATOM_LEN:
                continue
            adjusted_start = parent_start + start
            adjusted_end = adjusted_start + len(text)
            atoms.append((adjusted_start, adjusted_end, text))

        if not atoms:
            return _regex_split(sentence, parent_start)
        return atoms


_SPACY_SPLITTER: Optional[_SpacySplitter] = None


def _get_spacy_splitter() -> Optional[_SpacySplitter]:
    global _SPACY_SPLITTER
    if _SPACY_SPLITTER is not None:
        return _SPACY_SPLITTER
    model_name = os.environ.get("VOYAGER_GROUNDEDNESS_NLI_SPACY_MODEL", "en_core_web_sm")
    _SPACY_SPLITTER = _SpacySplitter(model_name)
    return _SPACY_SPLITTER


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def is_atomic_enabled() -> bool:
    """Return True when atomic-claim decomposition is enabled via env flag."""

    raw = os.environ.get("VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS", "")
    return raw.lower() in {"1", "true", "yes"}


def decompose_sentence_into_atoms(
    sentence: str,
    parent_index: int,
    parent_start: int,
    *,
    use_spacy: bool = True,
    max_atoms_per_sentence: int = 6,
) -> List[AtomicClaim]:
    """Decompose a single sentence into a small set of atomic claims.

    Returns a list with at least one entry; if no decomposition is found,
    the original sentence is returned as a single atom. Character offsets
    are absolute (i.e. relative to the original response text).
    """

    if not sentence or not sentence.strip():
        return []

    pieces: List[Tuple[int, int, str]] = []
    if use_spacy:
        splitter = _get_spacy_splitter()
        if splitter is not None:
            pieces = splitter.split(sentence, parent_start)
    if not pieces:
        pieces = _regex_split(sentence, parent_start)

    if not pieces:
        clean = sentence.strip()
        offset = sentence.find(clean)
        pieces = [
            (
                parent_start + max(0, offset),
                parent_start + max(0, offset) + len(clean),
                clean,
            )
        ]

    if max_atoms_per_sentence > 0 and len(pieces) > max_atoms_per_sentence:
        pieces = pieces[:max_atoms_per_sentence]

    return [
        AtomicClaim(
            parent_index=int(parent_index),
            atom_index=int(idx),
            text=text,
            char_start=int(start),
            char_end=int(end),
        )
        for idx, (start, end, text) in enumerate(pieces)
    ]


__all__ = [
    "AtomicClaim",
    "decompose_sentence_into_atoms",
    "is_atomic_enabled",
]
