from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .hnsw_manager import HnswSegmentManager

logger = logging.getLogger(__name__)

MIN_PROTOTYPES_FOR_CODEBOOK = 16
DEFAULT_FINE_CODEBOOK_RATIO = 0.25
MAX_FINE_CENTROIDS = 512
DEFAULT_COARSE_CLUSTERS = 8
KMEANS_MAX_ITER = 30
DEFAULT_CTOP_R = 3


def _simple_kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int = KMEANS_MAX_ITER,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run k-means on (N, D) data, return (centroids, labels)."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    k = min(k, n)
    indices = rng.choice(n, size=k, replace=False)
    centroids = data[indices].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = _pairwise_l2(data, centroids)
        new_labels = dists.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = data[mask].mean(axis=0)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    centroids = centroids / norms
    return centroids.astype(np.float32), labels


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise squared L2 distances between rows of a and b."""
    a_sq = (a ** 2).sum(axis=1, keepdims=True)
    b_sq = (b ** 2).sum(axis=1, keepdims=True)
    return np.maximum(a_sq + b_sq.T - 2.0 * (a @ b.T), 0.0)


@dataclass(frozen=True)
class PrototypeSearchBudget:
    query_prototypes: int
    per_prototype_k: int
    candidate_budget: int
    query_token_count: int
    query_dispersion: float


class PrototypeScreeningIndex:
    """
    Compact lightweight screening index over a few prototypes per document.

    This index is intentionally derived data: it can be rebuilt from the primary
    multivector store, which keeps the implementation simple while we smoke-test
    whether multi-prototype screening is worth productizing further.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        dim: int,
        distance_metric: str = "cosine",
        default_doc_prototypes: int = 4,
        default_query_prototypes: int = 4,
        on_disk: bool = True,
        load_if_exists: bool = True,
    ) -> None:
        self.root_path = Path(root_path)
        self.dim = int(dim)
        self.distance_metric = distance_metric
        self.default_doc_prototypes = max(1, int(default_doc_prototypes))
        self.default_query_prototypes = max(1, int(default_query_prototypes))
        self.on_disk = bool(on_disk)
        self.manifest_path = self.root_path / "manifest.json"
        self.hnsw_path = self.root_path / "hnsw"
        self.last_search_profile: Dict[str, Any] = {}
        self._doc_count = 0
        self._prototype_count = 0
        self._prototype_doc_ids: Dict[int, Any] = {}
        self._doc_prototype_ids: Dict[Any, List[int]] = {}
        self._next_prototype_id = 0
        self._manager: Optional[HnswSegmentManager] = None

        self._gem_cquant: Optional[np.ndarray] = None
        self._gem_cindex_labels: Optional[np.ndarray] = None
        self._gem_n_coarse: int = 0
        self._gem_centroid_dists: Optional[np.ndarray] = None
        self._gem_idf: Optional[np.ndarray] = None
        self._gem_doc_centroid_ids: Dict[Any, List[int]] = {}
        self._gem_doc_ctop: Dict[Any, List[int]] = {}

        if load_if_exists and self.manifest_path.exists():
            self._load_manifest()
        self._open_manager()

    def _open_manager(self) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._manager = HnswSegmentManager(
            self.hnsw_path,
            dim=self.dim,
            distance_metric=self.distance_metric,
            on_disk=self.on_disk,
        )

    @property
    def manager(self) -> HnswSegmentManager:
        if self._manager is None:
            self._open_manager()
        return self._manager  # type: ignore[return-value]

    def _write_manifest(self) -> None:
        payload = {
            "version": 1,
            "dim": self.dim,
            "distance_metric": self.distance_metric,
            "default_doc_prototypes": self.default_doc_prototypes,
            "default_query_prototypes": self.default_query_prototypes,
            "doc_count": int(self._doc_count),
            "prototype_count": int(self._prototype_count),
            "prototype_doc_ids": [
                {
                    "prototype_id": int(prototype_id),
                    "doc_id": doc_id,
                }
                for prototype_id, doc_id in sorted(self._prototype_doc_ids.items())
            ],
            "doc_prototype_ids": [
                {
                    "doc_id": doc_id,
                    "prototype_ids": [int(prototype_id) for prototype_id in prototype_ids],
                }
                for doc_id, prototype_ids in self._doc_prototype_ids.items()
            ],
            "next_prototype_id": int(self._next_prototype_id),
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

    def _load_manifest(self) -> None:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._doc_count = int(payload.get("doc_count", 0))
        self._prototype_count = int(payload.get("prototype_count", 0))
        self._prototype_doc_ids = {
            int(item["prototype_id"]): item.get("doc_id")
            for item in payload.get("prototype_doc_ids", [])
        }
        if payload.get("doc_prototype_ids"):
            self._doc_prototype_ids = {
                item.get("doc_id"): [int(prototype_id) for prototype_id in item.get("prototype_ids", [])]
                for item in payload.get("doc_prototype_ids", [])
            }
        else:
            doc_map: Dict[Any, List[int]] = {}
            for prototype_id, doc_id in self._prototype_doc_ids.items():
                doc_map.setdefault(doc_id, []).append(int(prototype_id))
            self._doc_prototype_ids = doc_map
        self._next_prototype_id = int(
            payload.get(
                "next_prototype_id",
                (max(self._prototype_doc_ids.keys()) + 1) if self._prototype_doc_ids else 0,
            )
        )
        self._load_gem_state()

    @property
    def gem_codebook_path(self) -> Path:
        return self.root_path / "gem_codebook.npz"

    @property
    def gem_doc_profiles_path(self) -> Path:
        return self.root_path / "gem_doc_profiles.json"

    @property
    def gem_enabled(self) -> bool:
        return self._gem_cquant is not None and self._gem_centroid_dists is not None

    def _collect_all_prototype_vectors(self) -> Optional[np.ndarray]:
        """Gather all prototype vectors currently in the HNSW manager."""
        if self._prototype_count <= 0 or self.manager.total_vectors() <= 0:
            return None
        proto_ids = sorted(self._prototype_doc_ids.keys())
        retrieved = self.manager.retrieve(proto_ids)
        id_to_vec: Dict[int, np.ndarray] = {}
        for item in retrieved:
            pid = item.get("id")
            vec = item.get("vector")
            if pid is not None and vec is not None:
                id_to_vec[int(pid)] = np.asarray(vec, dtype=np.float32)
        vectors = [id_to_vec[pid] for pid in proto_ids if pid in id_to_vec]
        if not vectors:
            return None
        return np.stack(vectors).astype(np.float32)

    def _build_gem_codebook(self, prototype_vectors: Optional[np.ndarray] = None) -> None:
        """Build the two-stage GEM-lite codebook from all prototype vectors."""
        if prototype_vectors is None:
            prototype_vectors = self._collect_all_prototype_vectors()
        if prototype_vectors is None or prototype_vectors.shape[0] < MIN_PROTOTYPES_FOR_CODEBOOK:
            self._gem_cquant = None
            self._gem_cindex_labels = None
            self._gem_n_coarse = 0
            self._gem_centroid_dists = None
            self._gem_idf = None
            return

        norms = np.linalg.norm(prototype_vectors, axis=1, keepdims=True) + 1e-8
        normalized = prototype_vectors / norms

        n_fine = max(4, min(int(normalized.shape[0] * DEFAULT_FINE_CODEBOOK_RATIO), MAX_FINE_CENTROIDS))
        cquant, _ = _simple_kmeans(normalized, k=n_fine)

        n_coarse = max(2, min(DEFAULT_COARSE_CLUSTERS, n_fine // 2))
        _, cindex_labels = _simple_kmeans(cquant, k=n_coarse)

        dists_sq = _pairwise_l2(cquant, cquant)
        centroid_dists = np.sqrt(np.maximum(dists_sq, 0.0)).astype(np.float32)

        all_labels = _pairwise_l2(normalized, cquant).argmin(axis=1).astype(np.int32)
        n_docs = len(set(self._prototype_doc_ids.values()))
        df = np.zeros(n_fine, dtype=np.float32)
        doc_centroids: Dict[Any, set] = {}
        proto_ids_sorted = sorted(self._prototype_doc_ids.keys())
        for idx, proto_id in enumerate(proto_ids_sorted):
            if idx >= all_labels.shape[0]:
                break
            doc_id = self._prototype_doc_ids[proto_id]
            c_id = int(all_labels[idx])
            doc_centroids.setdefault(doc_id, set()).add(c_id)
        for centroid_set in doc_centroids.values():
            for c_id in centroid_set:
                df[c_id] += 1.0
        idf = np.log((max(n_docs, 1) + 1.0) / (df + 1.0)).astype(np.float32)

        self._gem_cquant = cquant
        self._gem_cindex_labels = cindex_labels
        self._gem_n_coarse = int(n_coarse)
        self._gem_centroid_dists = centroid_dists
        self._gem_idf = idf

        self._rebuild_doc_cluster_profiles()

    def _rebuild_doc_cluster_profiles(self) -> None:
        """Assign centroid IDs and C_top coarse clusters to every document."""
        if not self.gem_enabled:
            self._gem_doc_centroid_ids = {}
            self._gem_doc_ctop = {}
            return

        all_proto_vectors = self._collect_all_prototype_vectors()
        if all_proto_vectors is None:
            return

        norms = np.linalg.norm(all_proto_vectors, axis=1, keepdims=True) + 1e-8
        normalized = all_proto_vectors / norms
        all_labels = _pairwise_l2(normalized, self._gem_cquant).argmin(axis=1).astype(np.int32)

        proto_ids_sorted = sorted(self._prototype_doc_ids.keys())
        doc_centroid_ids: Dict[Any, List[int]] = {}
        for idx, proto_id in enumerate(proto_ids_sorted):
            if idx >= all_labels.shape[0]:
                break
            doc_id = self._prototype_doc_ids[proto_id]
            doc_centroid_ids.setdefault(doc_id, []).append(int(all_labels[idx]))

        doc_ctop: Dict[Any, List[int]] = {}
        for doc_id, c_ids in doc_centroid_ids.items():
            coarse_scores: Dict[int, float] = {}
            for c_id in c_ids:
                coarse_label = int(self._gem_cindex_labels[c_id])
                coarse_scores[coarse_label] = coarse_scores.get(coarse_label, 0.0) + float(self._gem_idf[c_id])
            sorted_coarse = sorted(coarse_scores.items(), key=lambda x: x[1], reverse=True)
            r = min(DEFAULT_CTOP_R, len(sorted_coarse))
            doc_ctop[doc_id] = [c for c, _ in sorted_coarse[:r]]

        self._gem_doc_centroid_ids = doc_centroid_ids
        self._gem_doc_ctop = doc_ctop

    def _save_gem_state(self) -> None:
        """Persist GEM-lite codebook and per-doc profiles."""
        if not self.gem_enabled:
            self.gem_codebook_path.unlink(missing_ok=True)
            self.gem_doc_profiles_path.unlink(missing_ok=True)
            return

        self.root_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.gem_codebook_path,
            cquant=self._gem_cquant,
            cindex_labels=self._gem_cindex_labels,
            centroid_dists=self._gem_centroid_dists,
            idf=self._gem_idf,
            n_coarse=np.array([self._gem_n_coarse], dtype=np.int32),
        )
        profiles = {
            "doc_centroid_ids": {
                str(doc_id): ids for doc_id, ids in self._gem_doc_centroid_ids.items()
            },
            "doc_ctop": {
                str(doc_id): ctop for doc_id, ctop in self._gem_doc_ctop.items()
            },
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.root_path,
            prefix=".gem_profiles.", suffix=".tmp", delete=False,
        ) as handle:
            json.dump(profiles, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, self.gem_doc_profiles_path)

    def _load_gem_state(self) -> None:
        """Load GEM-lite codebook and per-doc profiles from disk."""
        if not self.gem_codebook_path.exists():
            return
        try:
            data = np.load(self.gem_codebook_path, allow_pickle=False)
            self._gem_cquant = data["cquant"].astype(np.float32)
            self._gem_cindex_labels = data["cindex_labels"].astype(np.int32)
            self._gem_centroid_dists = data["centroid_dists"].astype(np.float32)
            self._gem_idf = data["idf"].astype(np.float32)
            self._gem_n_coarse = int(data["n_coarse"][0])
        except Exception:
            logger.warning("Failed to load GEM codebook; falling back to plain prototype screening")
            self._gem_cquant = None
            return

        if self.gem_doc_profiles_path.exists():
            try:
                profiles = json.loads(self.gem_doc_profiles_path.read_text(encoding="utf-8"))
                raw_cids = profiles.get("doc_centroid_ids", {})
                raw_ctop = profiles.get("doc_ctop", {})
                self._gem_doc_centroid_ids = {}
                self._gem_doc_ctop = {}
                for doc_id in self._doc_prototype_ids:
                    key = str(doc_id)
                    if key in raw_cids:
                        self._gem_doc_centroid_ids[doc_id] = [int(x) for x in raw_cids[key]]
                    if key in raw_ctop:
                        self._gem_doc_ctop[doc_id] = [int(x) for x in raw_ctop[key]]
            except Exception:
                logger.warning("Failed to load GEM doc profiles; will rebuild on next search")
                self._gem_doc_centroid_ids = {}
                self._gem_doc_ctop = {}

    def _compute_query_gem_profile(
        self, query_embedding: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Compute query centroid IDs and coarse cluster profile."""
        norms = np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
        normalized = query_embedding / norms
        dists = _pairwise_l2(normalized, self._gem_cquant)
        centroid_ids = dists.argmin(axis=1).astype(np.int32).tolist()

        coarse_scores: Dict[int, float] = {}
        for c_id in centroid_ids:
            coarse_label = int(self._gem_cindex_labels[c_id])
            coarse_scores[coarse_label] = coarse_scores.get(coarse_label, 0.0) + float(self._gem_idf[c_id])
        sorted_coarse = sorted(coarse_scores.items(), key=lambda x: x[1], reverse=True)
        r = min(DEFAULT_CTOP_R, len(sorted_coarse))
        query_ctop = [c for c, _ in sorted_coarse[:r]]
        return centroid_ids, query_ctop

    def _compute_qch(self, query_centroid_ids: List[int], doc_centroid_ids: List[int]) -> float:
        """Compute quantized Chamfer: sum over q of min over p of dist(NN(q), NN(p))."""
        if not query_centroid_ids or not doc_centroid_ids:
            return float("inf")
        q_arr = np.array(query_centroid_ids, dtype=np.int32)
        p_arr = np.array(doc_centroid_ids, dtype=np.int32)
        sub_matrix = self._gem_centroid_dists[np.ix_(q_arr, p_arr)]
        return float(sub_matrix.min(axis=1).sum())

    def _cluster_overlap(self, query_ctop: List[int], doc_ctop: List[int]) -> int:
        """Count overlapping coarse clusters between query and document."""
        return len(set(query_ctop) & set(doc_ctop))

    def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None

    def reset(self) -> None:
        self.close()
        if self.root_path.exists():
            shutil.rmtree(self.root_path, ignore_errors=True)
        self._doc_count = 0
        self._prototype_count = 0
        self._prototype_doc_ids = {}
        self._doc_prototype_ids = {}
        self._next_prototype_id = 0
        self._gem_cquant = None
        self._gem_cindex_labels = None
        self._gem_n_coarse = 0
        self._gem_centroid_dists = None
        self._gem_idf = None
        self._gem_doc_centroid_ids = {}
        self._gem_doc_ctop = {}
        self._open_manager()
        self._write_manifest()

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
        for idx, doc in enumerate(docs):
            if doc.ndim != 2 or doc.shape[1] <= 0:
                raise ValueError(f"Expected 2D multivector embedding at {idx}, got {doc.shape}")
            limit = int(lengths[idx]) if lengths is not None else doc.shape[0]
            limit = max(1, min(limit, doc.shape[0]))
            trimmed.append(np.asarray(doc[:limit], dtype=np.float32))
        return trimmed

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return matrix / norms

    @classmethod
    def extract_prototypes(
        cls,
        embedding: Union[np.ndarray, torch.Tensor],
        *,
        max_prototypes: int,
    ) -> np.ndarray:
        if isinstance(embedding, torch.Tensor):
            matrix = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            matrix = np.asarray(embedding, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got {matrix.shape}")

        token_matrix = cls._normalize_rows(matrix)
        mean = token_matrix.mean(axis=0, keepdims=True)
        mean = cls._normalize_rows(mean)[0]
        if max_prototypes <= 1 or token_matrix.shape[0] == 1:
            return np.asarray([mean], dtype=np.float32)

        target = min(int(max_prototypes), token_matrix.shape[0] + 1)
        selected_indices: List[int] = []
        used = np.zeros((token_matrix.shape[0],), dtype=bool)
        similarity_to_mean = token_matrix @ mean
        first_idx = int(np.argmin(similarity_to_mean))
        selected_indices.append(first_idx)
        used[first_idx] = True

        while len(selected_indices) < target - 1:
            selected = token_matrix[selected_indices]
            similarities = token_matrix @ selected.T
            closest = similarities.max(axis=1)
            closest[used] = 1.0
            next_idx = int(np.argmin(closest))
            if used[next_idx]:
                break
            selected_indices.append(next_idx)
            used[next_idx] = True

        coverage = token_matrix[selected_indices] if selected_indices else np.empty((0, token_matrix.shape[1]), dtype=np.float32)
        return np.concatenate([mean.reshape(1, -1), coverage], axis=0).astype(np.float32, copy=False)

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
        self.reset()
        self.append(
            doc_ids=doc_ids,
            embeddings=embeddings,
            lengths=lengths,
            max_prototypes=max_prototypes,
        )
        self._build_gem_codebook()
        self._save_gem_state()

    def append(
        self,
        *,
        doc_ids: Sequence[Any],
        embeddings: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
        lengths: Optional[Sequence[int]] = None,
        max_prototypes: Optional[int] = None,
    ) -> None:
        if max_prototypes is not None:
            self.default_doc_prototypes = max(1, int(max_prototypes))
        trimmed_docs = self._coerce_embeddings(embeddings, lengths)
        if len(doc_ids) != len(trimmed_docs):
            raise ValueError(f"Expected {len(trimmed_docs)} doc_ids, got {len(doc_ids)}")
        duplicates = sorted({doc_id for doc_id in doc_ids if doc_id in self._doc_prototype_ids})
        if duplicates:
            raise ValueError(f"Duplicate doc_ids are not allowed in prototype sidecar append: {duplicates!r}")

        max_proto = max(1, int(self.default_doc_prototypes))
        doc_vectors: List[np.ndarray] = []
        payloads: List[Dict[str, Any]] = []
        ids: List[int] = []

        for doc_id, matrix in zip(doc_ids, trimmed_docs):
            prototypes = self.extract_prototypes(matrix, max_prototypes=max_proto)
            prototype_ids: List[int] = []
            for slot, prototype in enumerate(prototypes):
                prototype_id = int(self._next_prototype_id)
                self._next_prototype_id += 1
                prototype_ids.append(prototype_id)
                doc_vectors.append(prototype)
                payloads.append(
                    {
                        "doc_id": doc_id,
                        "prototype_slot": int(slot),
                        "prototype_kind": "global_mean" if slot == 0 else "coverage_medoid",
                    }
                )
                ids.append(prototype_id)
                self._prototype_doc_ids[prototype_id] = doc_id
            self._doc_prototype_ids[doc_id] = prototype_ids

        if doc_vectors:
            stacked = np.stack(doc_vectors).astype(np.float32, copy=False)
            self.manager.add(stacked, ids=ids, payloads=payloads)
            self.manager.flush()
        self._doc_count += len(doc_ids)
        self._prototype_count += len(doc_vectors)
        self._write_manifest()
        if self._prototype_count >= MIN_PROTOTYPES_FOR_CODEBOOK and not self.gem_enabled:
            self._build_gem_codebook()
            self._save_gem_state()

    def delete(self, doc_ids: Iterable[Any]) -> int:
        prototype_ids: List[int] = []
        removed_docs = 0
        for doc_id in doc_ids:
            owned = self._doc_prototype_ids.pop(doc_id, None)
            if not owned:
                continue
            removed_docs += 1
            prototype_ids.extend(int(prototype_id) for prototype_id in owned)
        if not prototype_ids:
            return 0
        deleted = int(self.manager.delete(prototype_ids))
        self.manager.flush()
        for prototype_id in prototype_ids:
            self._prototype_doc_ids.pop(int(prototype_id), None)
        self._doc_count = max(0, self._doc_count - removed_docs)
        self._prototype_count = max(0, self._prototype_count - deleted)
        self._write_manifest()
        return removed_docs

    def plan_budget(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        *,
        top_k: int,
        candidate_budget: Optional[int] = None,
        max_query_prototypes: Optional[int] = None,
    ) -> PrototypeSearchBudget:
        if isinstance(query_embedding, torch.Tensor):
            matrix = query_embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            matrix = np.asarray(query_embedding, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D query embedding, got {matrix.shape}")

        normalized = self._normalize_rows(matrix)
        mean = self._normalize_rows(normalized.mean(axis=0, keepdims=True))[0]
        dispersion = float(1.0 - np.clip((normalized @ mean).mean(), -1.0, 1.0))
        token_count = int(matrix.shape[0])

        query_prototypes = min(max_query_prototypes or self.default_query_prototypes, token_count)
        if token_count <= 4:
            query_prototypes = min(query_prototypes, 1)
        elif token_count <= 16 and dispersion < 0.12:
            query_prototypes = min(query_prototypes, 2)
        elif token_count <= 64 and dispersion < 0.20:
            query_prototypes = min(query_prototypes, 3)

        budget = int(candidate_budget or min(max(top_k * 8, 32), 256))
        per_prototype_k = max(int(budget), 32)
        return PrototypeSearchBudget(
            query_prototypes=max(1, int(query_prototypes)),
            per_prototype_k=max(1, int(per_prototype_k)),
            candidate_budget=max(1, int(budget)),
            query_token_count=token_count,
            query_dispersion=dispersion,
        )

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        *,
        top_k: int,
        candidate_budget: Optional[int] = None,
        max_query_prototypes: Optional[int] = None,
        allowed_doc_ids: Optional[Iterable[Any]] = None,
    ) -> List[Any]:
        if self._prototype_count <= 0 or self.manager.total_vectors() <= 0:
            self.last_search_profile = {
                "mode": "prototype_screening",
                "candidate_count": 0,
                "reason": "sidecar_empty",
            }
            return []

        budget = self.plan_budget(
            query_embedding,
            top_k=top_k,
            candidate_budget=candidate_budget,
            max_query_prototypes=max_query_prototypes,
        )
        query_prototypes = self.extract_prototypes(
            query_embedding,
            max_prototypes=budget.query_prototypes,
        )
        allowed = set(allowed_doc_ids) if allowed_doc_ids is not None else None
        score_map: Dict[Any, float] = {}
        max_score_map: Dict[Any, float] = {}
        hit_count_map: Dict[Any, int] = {}
        docs_seen = 0
        search_started = time.perf_counter()

        for prototype in query_prototypes:
            results = self.manager.search(prototype, k=budget.per_prototype_k)
            if not results:
                continue
            for prototype_id, score in results:
                doc_id = self._prototype_doc_ids.get(int(prototype_id))
                if doc_id is None:
                    continue
                if allowed is not None and doc_id not in allowed:
                    continue
                docs_seen += 1
                score_map[doc_id] = score_map.get(doc_id, 0.0) + float(score)
                max_score_map[doc_id] = max(max_score_map.get(doc_id, float("-inf")), float(score))
                hit_count_map[doc_id] = hit_count_map.get(doc_id, 0) + 1

        use_gem = self.gem_enabled and self._gem_doc_centroid_ids
        gem_profile: Dict[str, Any] = {"gem_active": False}
        if use_gem:
            if isinstance(query_embedding, torch.Tensor):
                q_matrix = query_embedding.detach().cpu().numpy().astype(np.float32)
            else:
                q_matrix = np.asarray(query_embedding, dtype=np.float32)
            if q_matrix.ndim != 2:
                use_gem = False
            else:
                query_centroid_ids, query_ctop = self._compute_query_gem_profile(q_matrix)
                query_ctop_set = set(query_ctop)
                gem_profile = {
                    "gem_active": True,
                    "n_fine_centroids": int(self._gem_cquant.shape[0]),
                    "n_coarse_clusters": self._gem_n_coarse,
                    "query_ctop": query_ctop,
                }

        if use_gem and score_map:
            combined_scores: Dict[Any, float] = {}
            for doc_id, hnsw_score in score_map.items():
                doc_cids = self._gem_doc_centroid_ids.get(doc_id)
                doc_ctop = self._gem_doc_ctop.get(doc_id, [])
                overlap = len(query_ctop_set & set(doc_ctop))
                if doc_cids:
                    qch = self._compute_qch(query_centroid_ids, doc_cids)
                    qch_bonus = 1.0 / (1.0 + qch)
                else:
                    qch_bonus = 0.0
                overlap_bonus = float(overlap) / max(len(query_ctop_set), 1)
                combined = hnsw_score + 0.3 * qch_bonus + 0.2 * overlap_bonus
                combined_scores[doc_id] = combined

            ranked = sorted(
                combined_scores.items(),
                key=lambda item: (
                    item[1],
                    max_score_map.get(item[0], float("-inf")),
                    hit_count_map.get(item[0], 0),
                ),
                reverse=True,
            )
        else:
            ranked = sorted(
                score_map.items(),
                key=lambda item: (
                    item[1],
                    max_score_map.get(item[0], float("-inf")),
                    hit_count_map.get(item[0], 0),
                ),
                reverse=True,
            )

        selected = [doc_id for doc_id, _ in ranked[: budget.candidate_budget]]
        elapsed_ms = (time.perf_counter() - search_started) * 1000.0
        self.last_search_profile = {
            "mode": "prototype_screening",
            "doc_count": self._doc_count,
            "prototype_count": self._prototype_count,
            "query_prototypes": budget.query_prototypes,
            "query_token_count": budget.query_token_count,
            "query_dispersion": budget.query_dispersion,
            "per_prototype_k": budget.per_prototype_k,
            "candidate_budget": budget.candidate_budget,
            "candidate_count": len(selected),
            "docs_seen": docs_seen,
            "elapsed_ms": elapsed_ms,
            **gem_profile,
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
        stats: Dict[str, Any] = {
            "engine": "prototype_screening",
            "doc_count": int(self._doc_count),
            "prototype_count": int(self._prototype_count),
            "avg_prototypes_per_doc": (
                float(self._prototype_count) / float(self._doc_count)
                if self._doc_count > 0
                else 0.0
            ),
            "storage_mb": storage_bytes / (1024 * 1024),
            "gem_lite_enabled": self.gem_enabled,
        }
        if self.gem_enabled:
            stats["gem_fine_centroids"] = int(self._gem_cquant.shape[0])
            stats["gem_coarse_clusters"] = self._gem_n_coarse
            stats["gem_docs_with_profiles"] = len(self._gem_doc_ctop)
        return stats
