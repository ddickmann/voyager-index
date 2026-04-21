"""Shared public models for the `colsearch.index` facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """A single search result with document ID, score, and optional payload."""

    doc_id: int
    score: float
    payload: Optional[Dict[str, Any]] = None
    token_scores: Optional[List[float]] = None
    matched_tokens: Optional[List[int]] = None

    def __repr__(self) -> str:
        pay = f", payload={self.payload}" if self.payload is not None else ""
        tok = f", token_scores=[{len(self.token_scores)} tokens]" if self.token_scores is not None else ""
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}{pay}{tok})"


@dataclass
class ScrollPage:
    """A page of results from scroll iteration."""

    results: List[SearchResult]
    next_offset: Optional[int] = None


@dataclass
class IndexStats:
    """Summary statistics for an Index."""

    total_documents: int = 0
    sealed_segments: int = 0
    active_documents: int = 0
    dim: int = 0
    engine: str = ""


__all__ = ["IndexStats", "ScrollPage", "SearchResult"]
