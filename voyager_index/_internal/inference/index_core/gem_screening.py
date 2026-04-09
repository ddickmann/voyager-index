"""
GEM Router Screening Backend
=============================

Rust-backed screening index that uses GEM-style codebook routing for
candidate generation. This replaces the Python-only GEM-lite logic with
a native Rust implementation for full speed.

Designed to be used as a drop-in screening backend for ColPaliEngine
(via the screening_backend="gem_router" config) and as a candidate
generator for ColbertIndex's balanced-mode search.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from latence_gem_router import PyGemRouter
    GEM_ROUTER_AVAILABLE = True
except ImportError:
    PyGemRouter = None
    GEM_ROUTER_AVAILABLE = False
    logger.debug("latence_gem_router not available; GemScreeningIndex will use fallback")

try:
    from .gpu_qch import GpuQchScorer
    GPU_QCH_AVAILABLE = True
except ImportError:
    GpuQchScorer = None
    GPU_QCH_AVAILABLE = False


DEFAULT_N_FINE_RATIO = 0.5
MAX_FINE_CENTROIDS = 4096
DEFAULT_N_COARSE = 32
DEFAULT_CTOP_R = 3
DEFAULT_N_PROBES = 4
MIN_DOCS_FOR_ROUTER = 8


class GemScreeningIndex:
    """
    Native Rust screening index using GEM-style codebook routing.

    Accepts the same build/search contract as PrototypeScreeningIndex so it
    can serve as a plug-in replacement inside ColPaliEngine.

    The router uses sequential integer indices internally. The mapping between
    these internal indices and the caller's external doc IDs is maintained in
    ``_doc_ids`` (position = internal index, value = external doc ID).
    Deletions mark entries as ``None`` to preserve positional stability.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        dim: int,
        n_fine: Optional[int] = None,
        n_coarse: int = DEFAULT_N_COARSE,
        ctop_r: int = DEFAULT_CTOP_R,
        n_probes: int = DEFAULT_N_PROBES,
        load_if_exists: bool = True,
        distance_metric: str = "cosine",
        default_doc_prototypes: int = 4,
        default_query_prototypes: int = 4,
        on_disk: bool = True,
    ) -> None:
        self.root_path = Path(root_path)
        self.dim = dim
        self.n_fine_override = n_fine
        self.n_coarse = n_coarse
        self.ctop_r = ctop_r
        self.n_probes = n_probes
        self.state_path = self.root_path / "gem_router.gemr"
        self.manifest_path = self.root_path / "gem_manifest.json"
        self.last_search_profile: Dict[str, Any] = {}

        self._router: Optional[Any] = None
        self._doc_count = 0
        # Position = router internal index, value = external doc ID (None = deleted)
        self._doc_ids: List[Any] = []
        self._doc_id_set: set = set()
        self._deleted_internal: set = set()

        if GEM_ROUTER_AVAILABLE:
            self._router = PyGemRouter(dim=dim)
        else:
            logger.warning(
                "latence_gem_router unavailable. "
                "GemScreeningIndex will not produce candidates."
            )

        if load_if_exists and self.state_path.exists():
            self._load()

    @property
    def gem_enabled(self) -> bool:
        return self._router is not None and self._router.is_ready()

    def _load(self) -> None:
        if self._router is None:
            return
        try:
            self._router.load(str(self.state_path))
            if self.manifest_path.exists():
                manifest = json.loads(self.manifest_path.read_text("utf-8"))
                self._doc_count = manifest.get("doc_count", self._router.n_docs())
                self._doc_ids = manifest.get("doc_ids", [])
                self._doc_id_set = {d for d in self._doc_ids if d is not None}
                self._deleted_internal = set(manifest.get("deleted_internal", []))
            else:
                self._doc_count = self._router.n_docs()
            logger.info(
                "Loaded GEM router: %d docs, %d fine centroids, %d coarse clusters",
                self._doc_count,
                self._router.n_fine(),
                self._router.n_coarse(),
            )
        except Exception:
            logger.warning("Failed to load GEM router state; will rebuild")

    def _save(self) -> None:
        if self._router is None or not self._router.is_ready():
            return
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._router.save(str(self.state_path))
        manifest = {
            "doc_count": self._doc_count,
            "doc_ids": list(self._doc_ids),
            "deleted_internal": sorted(self._deleted_internal),
            "n_fine": self._router.n_fine(),
            "n_coarse": self._router.n_coarse(),
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.root_path,
            prefix=".gem_manifest.", suffix=".tmp", delete=False,
        ) as handle:
            json.dump(manifest, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, self.manifest_path)

    def close(self) -> None:
        self._router = None

    def reset(self) -> None:
        self._doc_count = 0
        self._doc_ids = []
        self._doc_id_set = set()
        self._deleted_internal = set()
        if self._router is not None:
            self._router = PyGemRouter(dim=self.dim)
        if self.root_path.exists():
            shutil.rmtree(self.root_path, ignore_errors=True)
        self.root_path.mkdir(parents=True, exist_ok=True)

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
                raise ValueError(f"Expected 3D embeddings, got {matrix.shape}")
            docs = [matrix[idx] for idx in range(matrix.shape[0])]
        else:
            docs = [np.asarray(item, dtype=np.float32) for item in matrix]

        trimmed: List[np.ndarray] = []
        for idx, doc in enumerate(docs):
            if doc.ndim != 2:
                raise ValueError(f"Expected 2D at {idx}, got {doc.shape}")
            limit = int(lengths[idx]) if lengths is not None else doc.shape[0]
            limit = max(1, min(limit, doc.shape[0]))
            trimmed.append(np.asarray(doc[:limit], dtype=np.float32))
        return trimmed

    def rebuild(
        self,
        *,
        doc_ids: Sequence[Any],
        embeddings: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
        lengths: Optional[Sequence[int]] = None,
        max_prototypes: Optional[int] = None,
    ) -> None:
        self.reset()
        if self._router is None or len(doc_ids) < MIN_DOCS_FOR_ROUTER:
            return

        trimmed_docs = self._coerce_embeddings(embeddings, lengths)
        all_vecs, offsets, ext_ids = self._flatten_docs(trimmed_docs, start_idx=0)
        if all_vecs.shape[0] == 0:
            return

        n_fine = self.n_fine_override or min(
            MAX_FINE_CENTROIDS,
            max(4, int(all_vecs.shape[0] * DEFAULT_N_FINE_RATIO)),
        )
        n_coarse = min(self.n_coarse, n_fine // 2)

        self._router.build(
            all_vecs,
            ext_ids,
            offsets,
            n_fine=n_fine,
            n_coarse=n_coarse,
            max_kmeans_iter=30,
            ctop_r=self.ctop_r,
        )

        self._doc_ids = list(doc_ids)
        self._doc_id_set = set(self._doc_ids)
        self._deleted_internal = set()
        self._doc_count = len(doc_ids)
        self._save()

    def append(
        self,
        *,
        doc_ids: Sequence[Any],
        embeddings: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
        lengths: Optional[Sequence[int]] = None,
        max_prototypes: Optional[int] = None,
    ) -> None:
        if self._router is None or not self._router.is_ready():
            return

        duplicates = sorted({d for d in doc_ids if d in self._doc_id_set})
        if duplicates:
            raise ValueError(f"Duplicate doc_ids in GEM router append: {duplicates!r}")

        trimmed_docs = self._coerce_embeddings(embeddings, lengths)
        start_idx = len(self._doc_ids)
        all_vecs, offsets, ext_ids = self._flatten_docs(trimmed_docs, start_idx=start_idx)
        if all_vecs.shape[0] == 0:
            return

        self._router.add_documents(all_vecs, ext_ids, offsets)
        self._doc_ids.extend(doc_ids)
        self._doc_id_set.update(doc_ids)
        self._doc_count += len(doc_ids)
        self._save()

    def delete(self, doc_ids: Iterable[Any]) -> int:
        to_delete = set(doc_ids) & self._doc_id_set
        if not to_delete:
            return 0

        for internal_idx, ext_id in enumerate(self._doc_ids):
            if ext_id in to_delete:
                self._deleted_internal.add(internal_idx)
                self._doc_id_set.discard(ext_id)

        self._doc_count = max(0, self._doc_count - len(to_delete))
        self._save()
        return len(to_delete)

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        *,
        top_k: int,
        candidate_budget: Optional[int] = None,
        max_query_prototypes: Optional[int] = None,
        allowed_doc_ids: Optional[Iterable[Any]] = None,
    ) -> List[Any]:
        if not self.gem_enabled:
            self.last_search_profile = {
                "mode": "gem_router_screening",
                "candidate_count": 0,
                "reason": "router_not_ready",
            }
            return []

        if isinstance(query_embedding, torch.Tensor):
            q_matrix = query_embedding.detach().cpu().numpy().astype(np.float32)
        else:
            q_matrix = np.asarray(query_embedding, dtype=np.float32)

        if q_matrix.ndim != 2:
            self.last_search_profile = {
                "mode": "gem_router_screening",
                "candidate_count": 0,
                "reason": "invalid_query_shape",
            }
            return []

        budget = candidate_budget or min(max(top_k * 8, 32), 512)
        search_started = time.perf_counter()

        results = self._router.route_query(
            q_matrix,
            n_probes=self.n_probes,
            max_candidates=budget + len(self._deleted_internal),
        )

        allowed = set(allowed_doc_ids) if allowed_doc_ids is not None else None
        selected = []
        for doc_id, score in results:
            internal_idx = int(doc_id)
            if internal_idx in self._deleted_internal:
                continue
            if internal_idx >= len(self._doc_ids):
                logger.debug("GEM router returned out-of-range index %d; skipping", internal_idx)
                continue
            ext_doc_id = self._doc_ids[internal_idx]
            if ext_doc_id is None:
                continue
            if allowed is not None and ext_doc_id not in allowed:
                continue
            selected.append(ext_doc_id)
            if len(selected) >= budget:
                break

        elapsed_ms = (time.perf_counter() - search_started) * 1000.0
        self.last_search_profile = {
            "mode": "gem_router_screening",
            "gem_active": True,
            "n_fine_centroids": self._router.n_fine(),
            "n_coarse_clusters": self._router.n_coarse(),
            "n_probes": self.n_probes,
            "candidate_budget": budget,
            "candidate_count": len(selected),
            "elapsed_ms": elapsed_ms,
        }
        return selected

    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "engine": "gem_router",
            "doc_count": self._doc_count,
            "gem_router_ready": self.gem_enabled,
        }
        if self.gem_enabled:
            stats["n_fine_centroids"] = self._router.n_fine()
            stats["n_coarse_clusters"] = self._router.n_coarse()
        return stats

    def _flatten_docs(
        self,
        trimmed_docs: List[np.ndarray],
        start_idx: int = 0,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], List[int]]:
        """Flatten per-doc matrices into a single (N, D) array with offset ranges.

        ``start_idx`` is the router-internal index for the first document in this
        batch. For ``rebuild()`` this is 0; for ``append()`` it must be
        ``len(self._doc_ids)`` so that internal indices are globally unique.
        """
        all_vecs = []
        offsets = []
        ext_ids = []
        offset = 0
        for local_idx, matrix in enumerate(trimmed_docs):
            n_vecs = matrix.shape[0]
            all_vecs.append(matrix)
            offsets.append((offset, offset + n_vecs))
            ext_ids.append(start_idx + local_idx)
            offset += n_vecs

        if not all_vecs:
            return np.empty((0, self.dim), dtype=np.float32), [], []

        stacked = np.concatenate(all_vecs, axis=0).astype(np.float32, copy=False)
        return stacked, offsets, ext_ids
