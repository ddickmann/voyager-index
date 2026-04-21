from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CentroidSearchBudget:
    candidate_budget: int
    query_token_count: int
    query_dispersion: float


class CentroidScreeningIndex:
    """
    GPU-friendly lightweight screening index for approximate candidate generation.

    This keeps a compact (N_docs, K, H) centroid tensor as derived data and uses
    a single batched matmul to score the whole corpus before exact reranking.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        dim: int,
        default_doc_prototypes: int = 4,
        device: str = "cuda",
        load_if_exists: bool = True,
    ) -> None:
        self.root_path = Path(root_path)
        self.dim = int(dim)
        self.default_doc_prototypes = max(1, int(default_doc_prototypes))
        self.device = str(device)
        self.manifest_path = self.root_path / "manifest.json"
        self.state_path = self.root_path / "centroids.pt"
        self.last_search_profile: Dict[str, Any] = {}
        self._doc_count = 0
        self._prototype_count = 0
        self._doc_ids: List[Any] = []
        self._doc_index_by_id: Dict[Any, int] = {}
        self._centroids: Optional[torch.Tensor] = None
        self._centroid_mask: Optional[torch.Tensor] = None
        self._active_rows: Optional[torch.Tensor] = None

        if load_if_exists and self.manifest_path.exists() and self.state_path.exists():
            self._load()

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return matrix / norms

    @classmethod
    def extract_centroids(
        cls,
        embedding: np.ndarray,
        *,
        max_centroids: int,
    ) -> np.ndarray:
        if embedding.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got {embedding.shape}")
        matrix = cls._normalize_rows(np.asarray(embedding, dtype=np.float32))
        mean = cls._normalize_rows(matrix.mean(axis=0, keepdims=True))[0]
        if max_centroids <= 1 or matrix.shape[0] == 1:
            return mean.reshape(1, -1).astype(np.float32, copy=False)

        target = min(int(max_centroids), matrix.shape[0] + 1)
        selected_indices: List[int] = []
        used = np.zeros((matrix.shape[0],), dtype=bool)
        sim_to_mean = matrix @ mean
        first_idx = int(np.argmin(sim_to_mean))
        selected_indices.append(first_idx)
        used[first_idx] = True

        while len(selected_indices) < target - 1:
            selected = matrix[selected_indices]
            similarities = matrix @ selected.T
            closest = similarities.max(axis=1)
            closest[used] = 1.0
            next_idx = int(np.argmin(closest))
            if used[next_idx]:
                break
            selected_indices.append(next_idx)
            used[next_idx] = True

        coverage = (
            matrix[selected_indices]
            if selected_indices
            else np.empty((0, matrix.shape[1]), dtype=np.float32)
        )
        return np.concatenate([mean.reshape(1, -1), coverage], axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _coerce_embeddings(
        embeddings: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
        lengths: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        if isinstance(embeddings, torch.Tensor):
            matrix = embeddings.detach().cpu().numpy()
        else:
            matrix = embeddings

        if isinstance(matrix, np.ndarray):
            if matrix.ndim != 3:
                raise ValueError(f"Expected embeddings with shape (docs, tokens, dim), got {matrix.shape}")
            docs = [matrix[idx] for idx in range(matrix.shape[0])]
        else:
            docs = [np.asarray(item, dtype=np.float32) for item in matrix]

        trimmed: List[np.ndarray] = []
        expected_dim = None
        for idx, doc in enumerate(docs):
            if doc.ndim != 2 or doc.shape[1] <= 0:
                raise ValueError(f"Expected 2D multivector embedding at {idx}, got {doc.shape}")
            expected_dim = expected_dim or int(doc.shape[1])
            if int(doc.shape[1]) != expected_dim:
                raise ValueError(
                    f"Expected consistent hidden dim {expected_dim}, got {doc.shape[1]} at {idx}"
                )
            limit = int(lengths[idx]) if lengths is not None else doc.shape[0]
            limit = max(1, min(limit, doc.shape[0]))
            trimmed.append(np.asarray(doc[:limit], dtype=np.float32))
        return trimmed

    def _target_device(self) -> str:
        requested = self.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA centroid screening requested but CUDA is unavailable; falling back to CPU")
            return "cpu"
        return requested

    def _target_dtype(self) -> torch.dtype:
        return torch.float16 if self._target_device().startswith("cuda") else torch.float32

    def reset(self) -> None:
        self.close()
        if self.root_path.exists():
            shutil.rmtree(self.root_path, ignore_errors=True)
        self._doc_count = 0
        self._prototype_count = 0
        self._doc_ids = []
        self._doc_index_by_id = {}
        self.last_search_profile = {}

    def close(self) -> None:
        self._centroids = None
        self._centroid_mask = None
        self._active_rows = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _write_manifest(self) -> None:
        payload = {
            "version": 1,
            "dim": self.dim,
            "default_doc_prototypes": self.default_doc_prototypes,
            "doc_count": int(self._doc_count),
            "stored_doc_count": int(len(self._doc_ids)),
            "inactive_doc_count": int(max(len(self._doc_ids) - self._doc_count, 0)),
            "prototype_count": int(self._prototype_count),
            "backend": "centroid",
        }
        self.root_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.root_path,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, self.manifest_path)
        parent_fd = os.open(self.root_path, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _write_state(self) -> None:
        if self._centroids is None or self._centroid_mask is None or self._active_rows is None:
            raise RuntimeError("Cannot persist centroid screener before build()")
        payload = {
            "doc_ids": list(self._doc_ids),
            "centroids": self._centroids.detach().to("cpu"),
            "centroid_mask": self._centroid_mask.detach().to("cpu"),
            "active_rows": self._active_rows.detach().to("cpu"),
        }
        self.root_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.root_path,
            prefix=".centroids.",
            suffix=".pt",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            torch.save(payload, temp_path)
            with open(temp_path, "rb") as state_handle:
                os.fsync(state_handle.fileno())
            os.replace(temp_path, self.state_path)
            parent_fd = os.open(self.root_path, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            temp_path.unlink(missing_ok=True)

    def _load(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("version", 0)) != 1:
            raise RuntimeError(f"Unsupported centroid screener manifest version: {manifest.get('version')!r}")
        if manifest.get("backend") not in {None, "centroid"}:
            raise RuntimeError(f"Unexpected centroid screener backend marker: {manifest.get('backend')!r}")
        state = torch.load(self.state_path, map_location="cpu", weights_only=False)
        target_device = self._target_device()
        target_dtype = self._target_dtype()
        self.default_doc_prototypes = int(
            manifest.get("default_doc_prototypes", self.default_doc_prototypes)
        )
        self.dim = int(manifest.get("dim", self.dim))
        self._doc_ids = list(state.get("doc_ids") or [])
        centroids = state["centroids"]
        centroid_mask = state["centroid_mask"]
        active_rows = state.get("active_rows")
        if active_rows is None:
            active_rows = torch.ones((len(self._doc_ids),), dtype=torch.bool)
        if centroids.ndim != 3 or int(centroids.shape[-1]) != self.dim:
            raise RuntimeError(f"Unexpected persisted centroid tensor shape: {tuple(centroids.shape)!r}")
        if centroid_mask.shape != centroids.shape[:2]:
            raise RuntimeError(
                f"Unexpected persisted centroid mask shape {tuple(centroid_mask.shape)!r} for centroids {tuple(centroids.shape)!r}"
            )
        if int(centroids.shape[0]) != len(self._doc_ids):
            raise RuntimeError(
                f"Persisted doc count mismatch: manifest/state doc_ids={len(self._doc_ids)} centroids={centroids.shape[0]}"
            )
        if active_rows.ndim != 1 or int(active_rows.shape[0]) != len(self._doc_ids):
            raise RuntimeError(
                f"Unexpected persisted active row shape {tuple(active_rows.shape)!r} for {len(self._doc_ids)} doc ids"
            )
        self._centroids = centroids.to(device=target_device, dtype=target_dtype)
        self._centroid_mask = centroid_mask.to(device=target_device, dtype=target_dtype)
        self._active_rows = active_rows.to(device=target_device, dtype=torch.bool)
        self._doc_index_by_id = {
            doc_id: idx
            for idx, doc_id in enumerate(self._doc_ids)
            if bool(self._active_rows[idx].item())
        }
        self._doc_count = len(self._doc_index_by_id)
        self._prototype_count = (
            int(self._centroid_mask[self._active_rows].sum().item())
            if self._doc_count > 0
            else 0
        )

    def build(
        self,
        doc_ids: Sequence[Any],
        embeddings: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
        lengths: Optional[Sequence[int]] = None,
    ) -> None:
        target_device = self._target_device()
        target_dtype = self._target_dtype()
        doc_list = self._coerce_embeddings(embeddings, lengths)
        if len(doc_ids) != len(doc_list):
            raise ValueError(f"Expected {len(doc_list)} doc_ids, got {len(doc_ids)}")
        n_docs = len(doc_list)
        k = self.default_doc_prototypes
        centroids = np.zeros((n_docs, k, self.dim), dtype=np.float32)
        mask = np.zeros((n_docs, k), dtype=np.float32)

        for idx, doc in enumerate(doc_list):
            if int(doc.shape[1]) != self.dim:
                raise ValueError(f"Expected embedding dim {self.dim}, got {doc.shape[1]} for document {idx}")
            extracted = self.extract_centroids(doc, max_centroids=k)
            actual = min(extracted.shape[0], k)
            centroids[idx, :actual] = extracted[:actual]
            mask[idx, :actual] = 1.0

        self._doc_ids = list(doc_ids)
        self._doc_index_by_id = {doc_id: idx for idx, doc_id in enumerate(self._doc_ids)}
        self._doc_count = n_docs
        self._prototype_count = int(mask.sum())
        self._centroids = torch.from_numpy(centroids).to(device=target_device, dtype=target_dtype)
        self._centroid_mask = torch.from_numpy(mask).to(device=target_device, dtype=target_dtype)
        self._active_rows = torch.ones((n_docs,), device=target_device, dtype=torch.bool)
        self._write_manifest()
        self._write_state()
        self.last_search_profile = {
            "mode": "centroid_screening_rebuild",
            **self.get_statistics(),
        }

    def rebuild(
        self,
        *,
        doc_ids: Sequence[Any],
        embeddings: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
        lengths: Optional[Sequence[int]] = None,
        max_prototypes: Optional[int] = None,
    ) -> None:
        if max_prototypes is not None:
            self.default_doc_prototypes = max(1, int(max_prototypes))
        self.build(doc_ids=doc_ids, embeddings=embeddings, lengths=lengths)

    def append(
        self,
        *,
        doc_ids: Sequence[Any],
        embeddings: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
        lengths: Optional[Sequence[int]] = None,
        max_prototypes: Optional[int] = None,
    ) -> None:
        if not doc_ids:
            return
        if max_prototypes is not None:
            self.default_doc_prototypes = max(1, int(max_prototypes))
        duplicates = sorted({doc_id for doc_id in doc_ids if doc_id in self._doc_index_by_id})
        if duplicates:
            raise ValueError(f"Duplicate doc_ids are not allowed in centroid sidecar append: {duplicates!r}")
        if self._centroids is None or self._centroid_mask is None or self._active_rows is None or not self._doc_ids:
            self.build(doc_ids=doc_ids, embeddings=embeddings, lengths=lengths)
            return

        target_device = self._target_device()
        target_dtype = self._target_dtype()
        doc_list = self._coerce_embeddings(embeddings, lengths)
        if len(doc_ids) != len(doc_list):
            raise ValueError(f"Expected {len(doc_list)} doc_ids, got {len(doc_ids)}")
        n_docs = len(doc_list)
        k = self.default_doc_prototypes
        centroids = np.zeros((n_docs, k, self.dim), dtype=np.float32)
        mask = np.zeros((n_docs, k), dtype=np.float32)
        start_index = len(self._doc_ids)

        for idx, doc in enumerate(doc_list):
            if int(doc.shape[1]) != self.dim:
                raise ValueError(f"Expected embedding dim {self.dim}, got {doc.shape[1]} for document {idx}")
            extracted = self.extract_centroids(doc, max_centroids=k)
            actual = min(extracted.shape[0], k)
            centroids[idx, :actual] = extracted[:actual]
            mask[idx, :actual] = 1.0

        self._doc_ids.extend(list(doc_ids))
        for offset, doc_id in enumerate(doc_ids):
            self._doc_index_by_id[doc_id] = start_index + offset
        self._doc_count += len(doc_ids)
        self._prototype_count += int(mask.sum())
        self._centroids = torch.cat(
            [
                self._centroids,
                torch.from_numpy(centroids).to(device=target_device, dtype=target_dtype),
            ],
            dim=0,
        )
        self._centroid_mask = torch.cat(
            [
                self._centroid_mask,
                torch.from_numpy(mask).to(device=target_device, dtype=target_dtype),
            ],
            dim=0,
        )
        self._active_rows = torch.cat(
            [
                self._active_rows,
                torch.ones((len(doc_ids),), device=target_device, dtype=torch.bool),
            ],
            dim=0,
        )
        self._write_manifest()
        self._write_state()

    def delete(self, doc_ids: Iterable[Any]) -> int:
        if self._active_rows is None or self._centroid_mask is None:
            return 0
        candidate_indices = [
            int(self._doc_index_by_id.pop(doc_id))
            for doc_id in doc_ids
            if doc_id in self._doc_index_by_id
        ]
        if not candidate_indices:
            return 0
        index_tensor = torch.as_tensor(candidate_indices, device=self._active_rows.device, dtype=torch.long)
        active_mask = self._active_rows.index_select(0, index_tensor)
        live_indices = index_tensor[active_mask]
        if live_indices.numel() == 0:
            return 0
        removed = int(live_indices.numel())
        removed_prototypes = int(
            self._centroid_mask.index_select(0, live_indices.to(self._centroid_mask.device)).sum().item()
        )
        self._active_rows.index_fill_(0, live_indices, False)
        self._doc_count = max(0, self._doc_count - removed)
        self._prototype_count = max(0, self._prototype_count - removed_prototypes)
        self._write_manifest()
        self._write_state()
        return removed

    def plan_budget(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        *,
        top_k: int,
        candidate_budget: Optional[int] = None,
    ) -> CentroidSearchBudget:
        if isinstance(query_embedding, torch.Tensor):
            matrix = query_embedding.detach().to("cpu", dtype=torch.float32).numpy()
        else:
            matrix = np.asarray(query_embedding, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D query embedding, got {matrix.shape}")
        normalized = self._normalize_rows(matrix)
        mean = self._normalize_rows(normalized.mean(axis=0, keepdims=True))[0]
        dispersion = float(1.0 - np.clip((normalized @ mean).mean(), -1.0, 1.0))
        budget = int(candidate_budget or min(max(top_k * 8, 32), 256))
        return CentroidSearchBudget(
            candidate_budget=max(1, int(budget)),
            query_token_count=int(matrix.shape[0]),
            query_dispersion=dispersion,
        )

    def get_doc_indices(self, doc_ids: Iterable[Any]) -> torch.Tensor:
        indices = [
            int(self._doc_index_by_id[doc_id])
            for doc_id in doc_ids
            if doc_id in self._doc_index_by_id
        ]
        return torch.tensor(indices, device=self._target_device(), dtype=torch.long)

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        *,
        top_k: int,
        candidate_budget: Optional[int] = None,
        max_query_prototypes: Optional[int] = None,
        allowed_doc_ids: Optional[Iterable[Any]] = None,
    ) -> List[Any]:
        if (
            self._centroids is None
            or self._centroid_mask is None
            or self._active_rows is None
            or self._doc_count <= 0
        ):
            self.last_search_profile = {
                "mode": "centroid_screening",
                "candidate_count": 0,
                "reason": "sidecar_empty",
            }
            return []

        budget = self.plan_budget(
            query_embedding,
            top_k=top_k,
            candidate_budget=candidate_budget,
        )
        search_started = time.perf_counter()
        target_device = self._target_device()
        target_dtype = self._target_dtype()
        if isinstance(query_embedding, np.ndarray):
            query = torch.from_numpy(query_embedding).to(device=target_device, dtype=target_dtype)
        else:
            query = query_embedding.to(device=target_device, dtype=target_dtype)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if query.dim() != 2:
            raise ValueError(f"Expected 2D query embedding, got {tuple(query.shape)!r}")
        if int(query.shape[-1]) != self.dim:
            raise ValueError(f"Expected query embedding dim {self.dim}, got {query.shape[-1]}")
        query = F.normalize(query.to(dtype=torch.float32), p=2, dim=-1).to(dtype=target_dtype)

        centroids = self._centroids
        mask = self._centroid_mask
        active_rows = self._active_rows
        docs_seen = int(self._doc_count)
        allowed_indices = None
        if allowed_doc_ids is not None:
            allowed_indices = self.get_doc_indices(allowed_doc_ids)
            if allowed_indices.numel() == 0:
                self.last_search_profile = {
                    "mode": "centroid_screening",
                    "candidate_count": 0,
                    "candidate_budget": budget.candidate_budget,
                    "query_token_count": budget.query_token_count,
                    "query_dispersion": budget.query_dispersion,
                    "docs_seen": 0,
                    "screening_backend": "centroid",
                    "screening_mode": "multi_centroid",
                    "reason": "allowed_doc_ids_empty",
                }
                return []
            centroids = centroids.index_select(0, allowed_indices)
            mask = mask.index_select(0, allowed_indices)
            active_rows = active_rows.index_select(0, allowed_indices)
            docs_seen = int(allowed_indices.numel())

        n_docs, centroids_per_doc, hidden = centroids.shape
        flat_centroids = centroids.reshape(n_docs * centroids_per_doc, hidden)
        sim = query @ flat_centroids.t()
        sim = sim.view(query.shape[0], n_docs, centroids_per_doc)
        valid_mask = active_rows.unsqueeze(1) & (mask > 0.5)
        sim = sim.masked_fill(valid_mask.unsqueeze(0) == 0, float("-inf"))
        max_sim = sim.max(dim=2).values
        max_sim = torch.nan_to_num(max_sim, neginf=0.0)
        scores = max_sim.sum(dim=0)
        k = min(int(budget.candidate_budget), int(scores.shape[0]))
        topk_scores, topk_idx = scores.topk(k)
        if allowed_indices is not None:
            topk_idx = allowed_indices.index_select(0, topk_idx)
        selected_indices = topk_idx.detach().to("cpu", dtype=torch.long).tolist()
        selected = [self._doc_ids[idx] for idx in selected_indices]
        elapsed_ms = (time.perf_counter() - search_started) * 1000.0
        self.last_search_profile = {
            "mode": "centroid_screening",
            "screening_backend": "centroid",
            "screening_mode": "multi_centroid",
            "doc_count": self._doc_count,
            "prototype_count": self._prototype_count,
            "centroids_per_doc": self.default_doc_prototypes,
            "candidate_budget": budget.candidate_budget,
            "candidate_count": len(selected),
            "query_token_count": budget.query_token_count,
            "query_dispersion": budget.query_dispersion,
            "docs_seen": docs_seen,
            "elapsed_ms": elapsed_ms,
            "score_max": float(topk_scores.max().item()) if topk_scores.numel() > 0 else None,
        }
        return selected

    def get_statistics(self) -> Dict[str, Any]:
        storage_bytes = 0
        if self.root_path.exists():
            storage_bytes = sum(
                path.stat().st_size
                for path in self.root_path.rglob("*")
                if path.is_file()
            )
        return {
            "engine": "centroid_screening",
            "doc_count": int(self._doc_count),
            "stored_doc_count": int(len(self._doc_ids)),
            "inactive_doc_count": int(max(len(self._doc_ids) - self._doc_count, 0)),
            "prototype_count": int(self._prototype_count),
            "avg_prototypes_per_doc": (
                float(self._prototype_count) / float(self._doc_count)
                if self._doc_count > 0
                else 0.0
            ),
            "storage_mb": storage_bytes / (1024 * 1024),
        }
