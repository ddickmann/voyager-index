"""Structural protocols for shard-engine routing, storage, and exact scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class RouterProtocol(Protocol):
    def fit_initial(
        self,
        pooled_doc_vectors: torch.Tensor,
        pooled_doc_counts: torch.Tensor,
        doc_ids: Sequence[int],
        doc_id_to_shard: Dict[int, int],
        epochs: int = 10,
    ) -> None: ...

    def route(
        self,
        query_vectors: torch.Tensor | np.ndarray,
        k_candidates: int = 2000,
        prefetch_doc_cap: int = 0,
        nprobe_override: Optional[int] = None,
        search_k_cap: Optional[int] = None,
    ) -> Any: ...

    def route_batch(
        self,
        queries: Sequence[torch.Tensor | np.ndarray],
        k_candidates: int = 2000,
        prefetch_doc_cap: int = 0,
        nprobe_override: Optional[int] = None,
        search_k_cap: Optional[int] = None,
    ) -> List[Any]: ...

    def add_or_update_docs(
        self,
        pooled_doc_vectors: torch.Tensor,
        pooled_doc_counts: torch.Tensor,
        doc_ids: Sequence[int],
        doc_id_to_shard: Dict[int, int],
    ) -> None: ...

    def delete_docs(self, doc_ids: Sequence[int]) -> None: ...
    def should_full_retrain(
        self,
        retrain_every_ops: int = 50_000,
        retrain_dirty_doc_ratio: float = 0.05,
        retrain_dirty_shard_ratio: float = 0.10,
    ) -> bool: ...
    def save(self) -> None: ...
    def capability_snapshot(self) -> Dict[str, object]: ...


@runtime_checkable
class StoreProtocol(Protocol):
    manifest: Any

    def build(self, *args: Any, **kwargs: Any) -> Any: ...
    def fetch_docs(self, doc_ids: List[int]) -> Dict[int, np.ndarray]: ...
    def load_shard(
        self, shard_id: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]: ...
    def load_docs_from_shard(
        self, shard_id: int, doc_ids: List[int], device: str = "cpu"
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]: ...
    def load_shard_to_pinned(self, shard_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def page_cache_residency(self) -> Optional[Dict[str, Any]]: ...
    def persist_merged_layout(
        self, all_vectors: np.ndarray, doc_offsets: List[Tuple[int, int]], doc_ids: List[int], dim: int
    ) -> None: ...
    def doc_shard_id(self, doc_id: int) -> int: ...


@runtime_checkable
class FetchPipelineProtocol(Protocol):
    def fetch_per_shard(self, shard_ids: List[int], max_docs: int = 0) -> Tuple[List[Any], dict]: ...
    def fetch_candidate_docs(self, docs_by_shard: Dict[int, List[int]]) -> Tuple[List[Any], dict]: ...
    def pipelined_search(self, *args: Any, **kwargs: Any) -> Tuple[List[int], List[float], Dict[str, float]]: ...
    def capability_snapshot(self) -> Dict[str, object]: ...


@runtime_checkable
class CandidateScorerProtocol(Protocol):
    def score_candidates(
        self, query: torch.Tensor, candidate_ids: Sequence[int], k: int = 10
    ) -> tuple[List[int], List[float]]: ...


@runtime_checkable
class NativeExactBackendProtocol(Protocol):
    def score_candidates_exact(self, *args: Any, **kwargs: Any) -> tuple[List[int], List[float]]: ...
    def score_candidates_approx(self, *args: Any, **kwargs: Any) -> tuple[List[int], List[float]]: ...
    def fetch_candidate_embeddings(self, *args: Any, **kwargs: Any) -> tuple[bytes, Any, Any]: ...
    def close(self) -> None: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    def rerank_shard_chunks(
        self,
        query: torch.Tensor,
        shard_chunks: List[Any],
        k: int = 10,
        device: Optional[torch.device] = None,
        quantization_mode: str = "",
        variable_length_strategy: str = "bucketed",
    ) -> tuple[List[int], List[float], Dict[str, float]]: ...


__all__ = [
    "CandidateScorerProtocol",
    "FetchPipelineProtocol",
    "NativeExactBackendProtocol",
    "RerankerProtocol",
    "RouterProtocol",
    "StoreProtocol",
]
