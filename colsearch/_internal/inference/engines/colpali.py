"""
ColPali Visual Document Search Engine

ColPali is ColBERT for visual documents - using the same MaxSim late interaction
scoring but with vision-language embeddings. This enables multi-modal retrieval
where documents are indexed as images (PDFs, screenshots, scanned documents).

Key differences from ColBERT:
1. Higher embedding dimension (768 or 1024 vs 128)
2. Image patches as tokens (vs text tokens)
3. Same MaxSim scoring mechanism

The engine supports parallel indexing alongside ColBERT for hybrid text+visual search.

Usage:
    # Parallel indexing with ColBERT
    colbert_engine = ColBERTEngine(...)
    colpali_engine = ColPaliEngine(...)

    # Index same documents with both
    colbert_engine.index(embeddings=colbert_embs, ...)
    colpali_engine.index(embeddings=colpali_embs, ...)

    # Search both and combine
    text_results = colbert_engine.search(query_text_emb)
    visual_results = colpali_engine.search(query_visual_emb)

Author: Latence Team
License: Apache-2.0
"""

import heapq
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ...kernels.maxsim import fast_colbert_scores
from ..config import IndexConfig
from ..engines.base import DenseSearchEngine, SearchResult
from ..index_core.centroid_screening import CentroidScreeningIndex
from ..index_core.gem_screening import GEM_ROUTER_AVAILABLE, GemScreeningIndex
from ..index_core.prototype_screening import PrototypeScreeningIndex
from ..index_core.screening_sidecar import (
    SCREENING_HEALTH_DEGRADED,
    SCREENING_HEALTH_DISABLED,
    SCREENING_HEALTH_HEALTHY,
    SCREENING_HEALTH_WARMING,
    ScreeningCalibrationSummary,
    default_screening_state,
    normalize_screening_state,
)

logger = logging.getLogger(__name__)


class ColPaliConfig:
    """Configuration for ColPali engine."""
    SUPPORTED_SCREENING_BACKENDS = {"prototype_hnsw", "centroid", "gem_router"}

    def __init__(
        self,
        embed_dim: int = 768,           # ColPali uses 768 or 1024
        max_patches: int = 196,         # 14x14 patches for 224x224 image
        device: str = "cuda",
        batch_size: int = 16,           # Smaller due to larger embeddings
        use_quantization: bool = True,
        storage_type: str = "mmap",     # "memory", "mmap", "hdf5"
        use_prototype_screening: bool = True,
        prototype_doc_prototypes: int = 4,
        prototype_query_prototypes: int = 4,
        screening_backend: str = "prototype_hnsw",
    ):
        self.embed_dim = embed_dim
        self.max_patches = max_patches
        self.device = device
        self.batch_size = batch_size
        self.use_quantization = use_quantization
        self.storage_type = storage_type
        self.use_prototype_screening = use_prototype_screening
        self.prototype_doc_prototypes = prototype_doc_prototypes
        self.prototype_query_prototypes = prototype_query_prototypes
        if screening_backend not in self.SUPPORTED_SCREENING_BACKENDS:
            raise ValueError(
                f"Unsupported screening_backend {screening_backend!r}; "
                f"expected one of {sorted(self.SUPPORTED_SCREENING_BACKENDS)!r}"
            )
        self.screening_backend = screening_backend


class ColPaliEngine(DenseSearchEngine):
    """
    ColPali visual document search engine.

    Uses MaxSim late interaction scoring on vision-language embeddings.
    Compatible with ColBERT for parallel multi-modal indexing.

    Features:
    - Dense vector search with MaxSim scoring (same as ColBERT)
    - Designed for image/PDF/visual document embeddings
    - Supports parallel indexing with ColBERT for hybrid search
    - GPU-accelerated with Triton MaxSim kernel

    Example:
        >>> engine = ColPaliEngine(index_path="/data/colpali_index")
        >>>
        >>> # Index visual documents
        >>> engine.index(embeddings=visual_embeddings, doc_ids=doc_ids)
        >>>
        >>> # Search with visual query
        >>> results = engine.search(query_embedding=visual_query_emb, top_k=10)
        >>>
        >>> # Or use with Latence client
        >>> visual_emb = await latence.colpali.embed_image(image_path)
        >>> results = engine.search(query_embedding=visual_emb, top_k=10)
    """

    SCREENING_CALIBRATION_SAMPLE_SIZE = 8
    SCREENING_CALIBRATION_TOP_K = 5
    SCREENING_CALIBRATION_MIN_TOP1 = 0.75
    SCREENING_CALIBRATION_MIN_TOPK = 0.60
    SCREENING_RISKY_QUERY_DISPERSION = 0.28
    SCREENING_RISKY_QUERY_TOKEN_COUNT = 96

    def __init__(
        self,
        index_path: Union[str, Path],
        config: Optional[ColPaliConfig] = None,
        device: str = "cuda",
        load_if_exists: bool = True,
    ):
        """
        Initialize ColPali engine.

        Args:
            index_path: Path to store/load the index
            config: ColPali configuration
            device: Compute device ('cuda', 'cpu')
            load_if_exists: Load existing index if found
        """
        super().__init__(engine_name="colpali")

        self.index_path = Path(index_path)
        self.config = config or ColPaliConfig(device=device)
        self.device = device
        self.manifest_path = self.index_path / "manifest.json"
        self.screening_state_path = self.index_path / "screening_state.json"
        self.chunks_path = self.index_path / "chunks"
        self.legacy_index_path = self.index_path / "colpali_index.pt"

        # Compatibility surface
        self._embeddings: Optional[torch.Tensor] = None
        self._doc_ids: List[Any] = []
        self._indexed = False
        self._scales: Optional[torch.Tensor] = None
        self._chunks: List[Dict[str, Any]] = []
        self._deleted_ids: set[Any] = set()
        self._doc_locations: Dict[Any, Tuple[int, int]] = {}
        self._next_chunk_id = 0
        self._last_chunk_load_ms = 0.0
        self.last_search_profile: Dict[str, Any] = {}
        self.last_write_profile: Dict[str, Any] = {}
        self.last_screening_profile: Dict[str, Any] = {}
        self._screening_query_count = 0
        self._screening_shadow_every = int(
            os.environ.get("VOYAGER_MULTIMODAL_SCREENING_SHADOW_EVERY", "0") or "0"
        )
        self.screening_state: Dict[str, Any] = default_screening_state(
            health=SCREENING_HEALTH_DISABLED,
            reason="screening_disabled",
        )
        self.screening_index = None
        self.prototype_screening = None

        if load_if_exists:
            if self.legacy_index_path.exists() and not self.manifest_path.exists():
                raise RuntimeError(
                    "Legacy ColPali `.pt` indexes are no longer supported. "
                    "Rebuild the collection from a trusted source to migrate it."
                )
            if self._index_exists():
                self._load_index()
            else:
                self._configure_screening_index(load_if_exists=load_if_exists)
        else:
            self._configure_screening_index(load_if_exists=load_if_exists)

    def _index_exists(self) -> bool:
        """Check if index exists on disk."""
        return self.manifest_path.exists()

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _atomic_save_npz(self, path: Path, **arrays: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            saver = np.savez if self.config.storage_type in {"mmap", "fast"} else np.savez_compressed
            saver(temp_path, **arrays)
            generated = temp_path if temp_path.exists() else temp_path.with_suffix(temp_path.suffix + ".npz")
            with open(generated, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(generated, path)
            parent_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            for candidate in (temp_path, temp_path.with_suffix(temp_path.suffix + ".npz")):
                if candidate.exists():
                    candidate.unlink(missing_ok=True)

    def _create_screening_index(self, *, load_if_exists: bool):
        if self.config.screening_backend == "centroid":
            return CentroidScreeningIndex(
                self.index_path / "centroid_sidecar",
                dim=self.config.embed_dim,
                default_doc_prototypes=self.config.prototype_doc_prototypes,
                device=self.device,
                load_if_exists=load_if_exists,
            )
        if self.config.screening_backend == "gem_router":
            if not GEM_ROUTER_AVAILABLE:
                logger.warning(
                    "gem_router screening backend requested but latence_gem_router "
                    "is not installed; falling back to prototype_hnsw"
                )
            else:
                return GemScreeningIndex(
                    self.index_path / "gem_router_sidecar",
                    dim=self.config.embed_dim,
                    load_if_exists=load_if_exists,
                )
        return PrototypeScreeningIndex(
            self.index_path / "prototype_sidecar",
            dim=self.config.embed_dim,
            default_doc_prototypes=self.config.prototype_doc_prototypes,
            default_query_prototypes=self.config.prototype_query_prototypes,
            load_if_exists=load_if_exists,
        )

    def _configure_screening_index(self, *, load_if_exists: bool) -> None:
        if self.screening_index is not None:
            self.screening_index.close()
        if self.config.use_prototype_screening:
            self.screening_index = self._create_screening_index(load_if_exists=load_if_exists)
        else:
            self.screening_index = None
        self.prototype_screening = self.screening_index

    def _screening_state_timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _load_screening_state(self) -> None:
        if not self.screening_state_path.exists():
            default_health = (
                SCREENING_HEALTH_DISABLED
                if self.screening_index is None
                else SCREENING_HEALTH_WARMING
            )
            default_reason = (
                "screening_disabled"
                if self.screening_index is None
                else "awaiting_calibration"
            )
            self.screening_state = default_screening_state(
                health=default_health,
                reason=default_reason,
                updated_at=self._screening_state_timestamp(),
            )
            return
        payload = json.loads(self.screening_state_path.read_text(encoding="utf-8"))
        self.screening_state = normalize_screening_state(payload)
        if self.screening_index is None:
            self.screening_state = default_screening_state(
                health=SCREENING_HEALTH_DISABLED,
                reason="screening_disabled",
                updated_at=self._screening_state_timestamp(),
            )

    def _save_screening_state(self) -> None:
        self._atomic_write_json(self.screening_state_path, dict(self.screening_state))

    def _set_screening_state(
        self,
        *,
        health: str,
        reason: str,
        calibration: Optional[ScreeningCalibrationSummary] = None,
    ) -> None:
        self.screening_state = default_screening_state(
            health=health,
            reason=reason,
            updated_at=self._screening_state_timestamp(),
            calibration=calibration,
        )
        self._save_screening_state()

    def _apply_persisted_config(self, persisted: Dict[str, Any]) -> None:
        if not persisted:
            return
        self.config.embed_dim = int(persisted.get("embed_dim", self.config.embed_dim))
        self.config.max_patches = int(persisted.get("max_patches", self.config.max_patches))
        self.config.use_quantization = bool(
            persisted.get("use_quantization", self.config.use_quantization)
        )
        self.config.storage_type = str(persisted.get("storage_type", self.config.storage_type))
        self.config.use_prototype_screening = bool(
            persisted.get("use_prototype_screening", self.config.use_prototype_screening)
        )
        self.config.prototype_doc_prototypes = int(
            persisted.get("prototype_doc_prototypes", self.config.prototype_doc_prototypes)
        )
        self.config.prototype_query_prototypes = int(
            persisted.get("prototype_query_prototypes", self.config.prototype_query_prototypes)
        )
        screening_backend = persisted.get("screening_backend")
        if screening_backend is not None:
            screening_backend = str(screening_backend)
            if screening_backend not in ColPaliConfig.SUPPORTED_SCREENING_BACKENDS:
                raise RuntimeError(f"Unsupported persisted screening backend: {screening_backend!r}")
            self.config.screening_backend = screening_backend

    def _rebuild_doc_ids(self) -> None:
        deleted = set(self._deleted_ids)
        self._doc_ids = []
        self._doc_locations = {}
        for chunk_index, chunk in enumerate(self._chunks):
            for row_index, doc_id in enumerate(chunk["doc_ids"]):
                if doc_id in deleted:
                    continue
                self._doc_ids.append(doc_id)
                self._doc_locations[doc_id] = (chunk_index, row_index)
        self._indexed = bool(self._doc_ids)

    def _save_manifest(self) -> None:
        manifest = {
            "version": 1,
            "config": {
                "embed_dim": self.config.embed_dim,
                "max_patches": self.config.max_patches,
                "use_quantization": self.config.use_quantization,
                "storage_type": self.config.storage_type,
                "use_prototype_screening": self.config.use_prototype_screening,
                "prototype_doc_prototypes": self.config.prototype_doc_prototypes,
                "prototype_query_prototypes": self.config.prototype_query_prototypes,
                "screening_backend": self.config.screening_backend,
            },
            "next_chunk_id": self._next_chunk_id,
            "deleted_ids": list(self._deleted_ids),
            "chunks": self._chunks,
        }
        self._atomic_write_json(self.manifest_path, manifest)

    def _reset_storage(self) -> None:
        if self.index_path.exists():
            for path in self.index_path.iterdir():
                if path.is_dir():
                    for child in path.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink(missing_ok=True)
                    path.rmdir()
                else:
                    path.unlink(missing_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.chunks_path.mkdir(parents=True, exist_ok=True)
        self._chunks = []
        self._deleted_ids = set()
        self._doc_locations = {}
        self._next_chunk_id = 0
        self._doc_ids = []
        self._indexed = False
        if self.screening_index is not None:
            self.screening_index.reset()
            self.screening_state = default_screening_state(
                health=SCREENING_HEALTH_WARMING,
                reason="awaiting_calibration",
                updated_at=self._screening_state_timestamp(),
            )
        else:
            self.screening_state = default_screening_state(
                health=SCREENING_HEALTH_DISABLED,
                reason="screening_disabled",
                updated_at=self._screening_state_timestamp(),
            )

    def _normalize_doc_ids(self, doc_ids: Optional[List[Any]], num_docs: int) -> List[Any]:
        if doc_ids is None:
            start_id = len(self._doc_ids) + len(self._deleted_ids)
            doc_ids = list(range(start_id, start_id + num_docs))
        if len(doc_ids) != num_docs:
            raise ValueError("doc_ids must match the number of embeddings")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids must be unique within a single write")
        existing = set(self._doc_ids).union(self._deleted_ids)
        duplicates = existing.intersection(doc_ids)
        if duplicates:
            raise ValueError(f"Duplicate doc_ids are not allowed: {sorted(duplicates)!r}")
        return list(doc_ids)

    def _validate_lengths(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[List[int]] = None,
    ) -> np.ndarray:
        num_docs, max_patches, _ = embeddings.shape
        if lengths is None:
            return np.full((num_docs,), max_patches, dtype=np.int32)
        if len(lengths) != num_docs:
            raise ValueError("lengths must match the number of embeddings")
        array = np.asarray(lengths, dtype=np.int32)
        if np.any(array <= 0) or np.any(array > max_patches):
            raise ValueError("Each length must be between 1 and the padded patch count")
        return array

    def _append_chunk(
        self,
        embeddings: torch.Tensor,
        doc_ids: List[Any],
        lengths: np.ndarray,
    ) -> None:
        chunk_name = f"chunk_{self._next_chunk_id:06d}.npz"
        self._next_chunk_id += 1
        chunk_path = self.chunks_path / chunk_name

        if self.config.use_quantization:
            stored_embeddings, scales = self._quantize_int8(embeddings)
            embeddings_array = stored_embeddings.detach().cpu().numpy().astype(np.int8, copy=False)
            scales_array = scales.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        else:
            embeddings_array = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
            scales_array = np.empty((0,), dtype=np.float32)

        save_started = time.perf_counter()
        self._atomic_save_npz(
            chunk_path,
            embeddings=embeddings_array,
            lengths=lengths.astype(np.int32, copy=False),
            scales=scales_array,
        )
        save_ms = (time.perf_counter() - save_started) * 1000.0
        self._chunks.append(
            {
                "file": chunk_name,
                "doc_ids": list(doc_ids),
                "lengths": lengths.astype(np.int32, copy=False).tolist(),
                "quantized": bool(self.config.use_quantization),
                "shape": list(embeddings_array.shape),
            }
        )
        self._rebuild_doc_ids()
        self._save_manifest()
        self.last_write_profile = {
            "mode": "append_chunk",
            "doc_count": len(doc_ids),
            "quantized": bool(self.config.use_quantization),
            "storage_type": self.config.storage_type,
            "save_ms": save_ms,
        }

    def _load_index(self):
        """Load index from disk."""
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._apply_persisted_config(dict(manifest.get("config", {}) or {}))
            self._configure_screening_index(load_if_exists=True)
            self._chunks = list(manifest.get("chunks", []))
            self._deleted_ids = set(manifest.get("deleted_ids", []))
            self._next_chunk_id = int(manifest.get("next_chunk_id", len(self._chunks)))
            self._rebuild_doc_ids()
            self._load_screening_state()
            if self.screening_index is not None and self._doc_ids:
                stats = self.screening_index.get_statistics()
                if int(stats.get("doc_count", 0)) != len(self._doc_ids):
                    self.rebuild_prototype_screening_index()
                elif self.screening_state.get("health") == SCREENING_HEALTH_WARMING:
                    self._calibrate_screening_sidecar(reason="startup_recalibration")
            elif self.screening_index is None:
                self._set_screening_state(
                    health=SCREENING_HEALTH_DISABLED,
                    reason="screening_disabled",
                )
            else:
                self._set_screening_state(
                    health=SCREENING_HEALTH_DISABLED,
                    reason="empty_index",
                )
            logger.info(f"Loaded ColPali index with {len(self._doc_ids)} documents")
        except Exception as e:
            raise RuntimeError(f"Failed to load ColPali index at {self.index_path}: {e}") from e

    def _save_index(self):
        """Persist manifest state to disk."""
        self._save_manifest()
        logger.info("Saved ColPali manifest to %s", self.index_path)

    def _load_chunk_arrays(self, chunk: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        start = time.perf_counter()
        data = np.load(self.chunks_path / chunk["file"], allow_pickle=False)
        embeddings = data["embeddings"]
        lengths = data["lengths"].astype(np.int32, copy=False)
        scales = data["scales"].astype(np.float32, copy=False) if "scales" in data and data["scales"].size > 0 else None
        self._last_chunk_load_ms = (time.perf_counter() - start) * 1000.0
        return embeddings, lengths, scales

    def _iter_active_chunk_views(
        self,
        requested_doc_ids: Optional[set[Any]] = None,
    ):
        for chunk in self._chunks:
            positions = []
            doc_ids = []
            lengths = []
            for idx, doc_id in enumerate(chunk["doc_ids"]):
                if doc_id in self._deleted_ids:
                    continue
                if requested_doc_ids is not None and doc_id not in requested_doc_ids:
                    continue
                positions.append(idx)
                doc_ids.append(doc_id)
                lengths.append(int(chunk["lengths"][idx]))
            if not positions:
                continue
            embeddings, _, scales = self._load_chunk_arrays(chunk)
            yield chunk, doc_ids, np.asarray(positions, dtype=np.int64), np.asarray(lengths, dtype=np.int32), embeddings, scales

    def _iter_requested_chunk_views(
        self,
        requested_doc_ids: Iterable[Any],
    ):
        grouped_positions: Dict[int, List[int]] = {}
        seen: set[Any] = set()
        for doc_id in requested_doc_ids:
            if doc_id in seen or doc_id in self._deleted_ids:
                continue
            seen.add(doc_id)
            location = self._doc_locations.get(doc_id)
            if location is None:
                continue
            chunk_index, row_index = location
            grouped_positions.setdefault(int(chunk_index), []).append(int(row_index))

        for chunk_index, positions_list in grouped_positions.items():
            if not positions_list:
                continue
            chunk = self._chunks[int(chunk_index)]
            positions = np.asarray(sorted(set(positions_list)), dtype=np.int64)
            doc_ids = [chunk["doc_ids"][idx] for idx in positions]
            lengths = np.asarray(
                [int(chunk["lengths"][idx]) for idx in positions],
                dtype=np.int32,
            )
            embeddings, _, scales = self._load_chunk_arrays(chunk)
            yield chunk, doc_ids, positions, lengths, embeddings, scales

    def _materialize_chunk_embeddings(
        self,
        embeddings: np.ndarray,
        positions: np.ndarray,
        scales: Optional[np.ndarray],
    ) -> torch.Tensor:
        target_dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        chunk_embeddings = torch.from_numpy(embeddings[positions]).to(self.device)
        if scales is not None and len(scales) > 0:
            chunk_scales = torch.from_numpy(scales[positions]).to(self.device)
            return self._dequantize(chunk_embeddings, chunk_scales).to(target_dtype)
        return chunk_embeddings.to(target_dtype)

    def _build_documents_mask(
        self,
        lengths: np.ndarray,
        max_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        token_positions = torch.arange(max_tokens, device=device).unsqueeze(0)
        valid_lengths = torch.as_tensor(lengths, device=device, dtype=torch.int64).unsqueeze(1)
        return (token_positions < valid_lengths).to(torch.float32)

    def get_document_embeddings(self, doc_ids: List[Any]) -> Dict[Any, np.ndarray]:
        results: Dict[Any, np.ndarray] = {}
        for _, chunk_doc_ids, positions, lengths, embeddings, scales in self._iter_requested_chunk_views(doc_ids):
            chunk_tensors = self._materialize_chunk_embeddings(embeddings, positions, scales).detach().cpu().numpy()
            for idx, doc_id in enumerate(chunk_doc_ids):
                results[doc_id] = chunk_tensors[idx][:int(lengths[idx])]
        return results

    def _screening_budget(
        self,
        *,
        top_k: int,
        candidate_budget: Optional[int],
    ) -> int:
        if not self._doc_ids:
            return 0
        if candidate_budget is not None:
            return max(1, min(len(self._doc_ids), int(candidate_budget)))
        default_budget = min(len(self._doc_ids), max(top_k * 16, 64))
        return max(1, min(len(self._doc_ids), default_budget, 512))

    def _query_screening_features(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        if isinstance(query_embedding, torch.Tensor):
            matrix = query_embedding.detach().to("cpu", dtype=torch.float32).numpy()
        else:
            matrix = np.asarray(query_embedding, dtype=np.float32)
        if matrix.ndim == 3:
            if matrix.shape[0] != 1:
                raise ValueError("Expected a single query embedding when computing screening features")
            matrix = matrix[0]
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D query embedding, got {matrix.shape}")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        normalized = matrix / norms
        mean = normalized.mean(axis=0, keepdims=True)
        mean /= np.linalg.norm(mean, axis=1, keepdims=True) + 1e-8
        dispersion = float(1.0 - np.clip((normalized @ mean[0]).mean(), -1.0, 1.0))
        return {
            "query_token_count": int(matrix.shape[0]),
            "query_dispersion": dispersion,
        }

    @staticmethod
    def _merge_candidate_lanes(
        *candidate_lanes: Iterable[Any],
        limit: int,
    ) -> List[Any]:
        merged: List[Any] = []
        seen: set[Any] = set()
        for lane in candidate_lanes:
            for doc_id in lane:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(doc_id)
                if len(merged) >= limit:
                    return merged
        return merged

    def _sample_calibration_doc_ids(self, *, limit: int) -> List[Any]:
        if limit <= 0 or not self._doc_ids:
            return []
        if len(self._doc_ids) <= limit:
            return list(self._doc_ids)
        stride = max(1, len(self._doc_ids) // limit)
        sample = [self._doc_ids[idx] for idx in range(0, len(self._doc_ids), stride)]
        return sample[:limit]

    def _calibrate_screening_sidecar(self, *, reason: str) -> Dict[str, Any]:
        if self.screening_index is None:
            self._set_screening_state(
                health=SCREENING_HEALTH_DISABLED,
                reason="screening_disabled",
            )
            return dict(self.screening_state)
        if not self._doc_ids:
            self._set_screening_state(
                health=SCREENING_HEALTH_DISABLED,
                reason="empty_index",
            )
            return dict(self.screening_state)

        sample_doc_ids = self._sample_calibration_doc_ids(
            limit=min(self.SCREENING_CALIBRATION_SAMPLE_SIZE, len(self._doc_ids))
        )
        doc_embeddings = self.get_document_embeddings(sample_doc_ids)
        if not doc_embeddings:
            self._set_screening_state(
                health=SCREENING_HEALTH_DEGRADED,
                reason="calibration_documents_missing",
            )
            return dict(self.screening_state)

        candidate_budget = self._screening_budget(
            top_k=self.SCREENING_CALIBRATION_TOP_K,
            candidate_budget=None,
        )
        top_k = min(self.SCREENING_CALIBRATION_TOP_K, len(self._doc_ids))
        top1_hits = 0.0
        topk_hits = 0.0
        evaluated = 0

        for doc_id in sample_doc_ids:
            query_embedding = doc_embeddings.get(doc_id)
            if query_embedding is None:
                continue
            exact_results = self.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
            exact_ids = [result.doc_id for result in exact_results]
            sidecar_ids = self.screening_index.search(
                query_embedding,
                top_k=top_k,
                candidate_budget=candidate_budget,
            )
            exact_set = set(exact_ids)
            sidecar_set = set(sidecar_ids)
            if exact_ids:
                top1_hits += float(exact_ids[0] in sidecar_set)
                topk_hits += len(exact_set & sidecar_set) / float(len(exact_set))
            else:
                top1_hits += 1.0
                topk_hits += 1.0
            evaluated += 1

        if evaluated <= 0:
            self._set_screening_state(
                health=SCREENING_HEALTH_DEGRADED,
                reason="calibration_empty",
            )
            return dict(self.screening_state)

        calibration = ScreeningCalibrationSummary(
            sample_size=evaluated,
            top_k=top_k,
            candidate_budget=candidate_budget,
            top1_retention=top1_hits / float(evaluated),
            topk_retention=topk_hits / float(evaluated),
        )
        health = (
            SCREENING_HEALTH_HEALTHY
            if (
                calibration.top1_retention >= self.SCREENING_CALIBRATION_MIN_TOP1
                and calibration.topk_retention >= self.SCREENING_CALIBRATION_MIN_TOPK
            )
            else SCREENING_HEALTH_DEGRADED
        )
        state_reason = reason if health == SCREENING_HEALTH_HEALTHY else f"{reason}_failed"
        self._set_screening_state(
            health=health,
            reason=state_reason,
            calibration=calibration,
        )
        return dict(self.screening_state)

    def _screening_decision(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        *,
        top_k: int,
        candidate_ids: Optional[List[Any]],
        candidate_budget: Optional[int],
    ) -> Tuple[bool, str, Dict[str, float], int]:
        features = self._query_screening_features(query_embedding)
        effective_budget = self._screening_budget(top_k=top_k, candidate_budget=candidate_budget)
        if self.screening_index is None:
            return False, "screening_disabled", features, effective_budget
        health = str(self.screening_state.get("health", SCREENING_HEALTH_WARMING))
        if health != SCREENING_HEALTH_HEALTHY:
            return False, f"health_{health}", features, effective_budget
        if not self._doc_ids:
            return False, "empty_index", features, effective_budget
        if features["query_dispersion"] >= self.SCREENING_RISKY_QUERY_DISPERSION:
            return False, "risky_query_dispersion", features, effective_budget
        if features["query_token_count"] >= self.SCREENING_RISKY_QUERY_TOKEN_COUNT:
            return False, "risky_query_token_count", features, effective_budget
        if candidate_ids is not None and len(candidate_ids) <= effective_budget:
            return False, "exact_filtered_subset", features, effective_budget
        return True, "healthy", features, effective_budget

    def rebuild_prototype_screening_index(self) -> None:
        if self.screening_index is None:
            self._set_screening_state(
                health=SCREENING_HEALTH_DISABLED,
                reason="screening_disabled",
            )
            return
        doc_ids: List[Any] = []
        lengths: List[int] = []
        doc_embeddings: List[np.ndarray] = []
        for _, chunk_doc_ids, positions, chunk_lengths, embeddings, scales in self._iter_active_chunk_views():
            chunk_tensors = self._materialize_chunk_embeddings(embeddings, positions, scales).detach().cpu().numpy()
            for idx, doc_id in enumerate(chunk_doc_ids):
                doc_ids.append(doc_id)
                lengths.append(int(chunk_lengths[idx]))
                doc_embeddings.append(chunk_tensors[idx])
        if not doc_ids:
            self.screening_index.reset()
            self._set_screening_state(
                health=SCREENING_HEALTH_DISABLED,
                reason="empty_index",
            )
            self.last_screening_profile = {
                "mode": "screening_rebuild",
                "screening_backend": self.config.screening_backend,
                "doc_count": 0,
                "prototype_count": 0,
                "sidecar_health": self.screening_state.get("health"),
            }
            return
        stacked = np.stack(doc_embeddings).astype(np.float32, copy=False)
        self.screening_index.rebuild(
            doc_ids=doc_ids,
            embeddings=stacked,
            lengths=lengths,
            max_prototypes=self.config.prototype_doc_prototypes,
        )
        self._calibrate_screening_sidecar(reason="bootstrap_calibration")
        self.last_screening_profile = {
            "mode": "screening_rebuild",
            "screening_backend": self.config.screening_backend,
            "sidecar_health": self.screening_state.get("health"),
            **self.screening_index.get_statistics(),
        }

    def _append_to_screening_index(
        self,
        *,
        doc_ids: List[Any],
        embeddings: torch.Tensor,
        lengths: np.ndarray,
    ) -> None:
        if self.screening_index is None or not doc_ids:
            return
        self.screening_index.append(
            doc_ids=doc_ids,
            embeddings=embeddings.detach().cpu().numpy().astype(np.float32, copy=False),
            lengths=lengths.tolist(),
            max_prototypes=self.config.prototype_doc_prototypes,
        )
        self._calibrate_screening_sidecar(reason="delta_recalibration")
        self.last_screening_profile = {
            "mode": "screening_append",
            "screening_backend": self.config.screening_backend,
            "sidecar_health": self.screening_state.get("health"),
            **self.screening_index.get_statistics(),
        }

    def _delete_from_screening_index(self, doc_ids: List[Any]) -> None:
        if self.screening_index is None or not doc_ids:
            return
        removed = int(self.screening_index.delete(doc_ids))
        if self._doc_ids:
            self._calibrate_screening_sidecar(reason="delta_recalibration")
        else:
            self.screening_index.reset()
            self._set_screening_state(
                health=SCREENING_HEALTH_DISABLED,
                reason="empty_index",
            )
        self.last_screening_profile = {
            "mode": "screening_delete",
            "screening_backend": self.config.screening_backend,
            "deleted_doc_count": removed,
            "sidecar_health": self.screening_state.get("health"),
            **self.screening_index.get_statistics(),
        }

    def screen_candidates(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        *,
        top_k: int,
        candidate_ids: Optional[List[Any]] = None,
        candidate_budget: Optional[int] = None,
        max_query_prototypes: Optional[int] = None,
    ) -> Optional[List[Any]]:
        should_screen, reason, features, effective_budget = self._screening_decision(
            query_embedding,
            top_k=top_k,
            candidate_ids=candidate_ids,
            candidate_budget=candidate_budget,
        )
        if not should_screen:
            self.last_screening_profile = {
                "mode": "candidate_factory",
                "screening_backend": self.config.screening_backend,
                "reason": reason,
                "sidecar_health": self.screening_state.get("health"),
                "candidate_budget": effective_budget,
                "candidate_count": 0,
                **features,
            }
            return None
        factory_started = time.perf_counter()
        sidecar_selected = self.screening_index.search(
            query_embedding,
            top_k=top_k,
            candidate_budget=effective_budget,
            max_query_prototypes=max_query_prototypes,
            allowed_doc_ids=candidate_ids,
        )
        metadata_lane = (
            list(candidate_ids)
            if candidate_ids is not None and len(candidate_ids) <= max(effective_budget * 2, 128)
            else []
        )
        selected = self._merge_candidate_lanes(
            sidecar_selected,
            metadata_lane,
            limit=effective_budget,
        )
        if not selected:
            self.last_screening_profile = {
                "mode": "candidate_factory",
                "screening_backend": self.config.screening_backend,
                "reason": "empty_sidecar_lane",
                "sidecar_health": self.screening_state.get("health"),
                "candidate_budget": effective_budget,
                "candidate_count": 0,
                **features,
            }
            return None
        shadow_status = "disabled"
        if self._screening_shadow_every > 0:
            self._screening_query_count += 1
            if self._screening_query_count % self._screening_shadow_every == 0:
                shadow_status = "evaluated"
                exact_results = self.search(
                    query_embedding=query_embedding,
                    top_k=min(top_k, len(self._doc_ids)),
                )
                exact_ids = [result.doc_id for result in exact_results]
                if exact_ids and exact_ids[0] not in set(selected):
                    self._set_screening_state(
                        health=SCREENING_HEALTH_DEGRADED,
                        reason="shadow_mismatch",
                    )
                    shadow_status = "demoted"
        self.last_screening_profile = {
            "mode": "candidate_factory",
            "screening_backend": self.config.screening_backend,
            "sidecar_health": self.screening_state.get("health"),
            "candidate_budget": effective_budget,
            "candidate_count": len(selected),
            "lanes": [
                {
                    "name": "lightweight_sidecar",
                    "candidate_count": len(sidecar_selected),
                },
                {
                    "name": "metadata_filter_lane",
                    "candidate_count": len(metadata_lane),
                },
            ],
            "elapsed_ms": (time.perf_counter() - factory_started) * 1000.0,
            "shadow_evaluation": shadow_status,
            "sidecar_profile": dict(self.screening_index.last_search_profile),
            **features,
        }
        return selected

    def index(
        self,
        documents: Optional[List[str]] = None,
        doc_ids: Optional[List[Any]] = None,
        collection_name: str = "default",
        embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        """
        Index visual documents.

        Args:
            documents: Not used (embeddings required for visual)
            doc_ids: Document IDs
            collection_name: Collection name
            embeddings: Pre-computed visual embeddings (N, P, D)
                        P = patches, D = embed_dim

        Raises:
            ValueError: If embeddings not provided
        """
        if embeddings is None:
            raise ValueError(
                "ColPali requires pre-computed visual embeddings. "
                "Use latence.colpali.embed_images() first."
            )

        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)

        embeddings = embeddings.to(torch.float32)

        # Validate shape
        if embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D embeddings (N, patches, dim), got {embeddings.dim()}D"
            )

        num_docs = embeddings.shape[0]
        self._reset_storage()
        normalized_doc_ids = self._normalize_doc_ids(doc_ids, num_docs)
        lengths = self._validate_lengths(embeddings, kwargs.get("lengths"))
        self._append_chunk(embeddings, normalized_doc_ids, lengths)
        self.rebuild_prototype_screening_index()

        logger.info(
            f"Indexed {num_docs} visual documents "
            f"(shape: {tuple(embeddings.shape)}, quantized: {self.config.use_quantization})"
        )

    def _quantize_int8(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize embeddings to INT8.

        Args:
            embeddings: Float embeddings (N, P, D)

        Returns:
            Tuple of (quantized_embeddings, scales)
        """
        # Compute per-document scales
        max_vals = embeddings.abs().amax(dim=(1, 2), keepdim=True)
        scales = max_vals / 127.0

        # Quantize
        quantized = (embeddings / (scales + 1e-8)).round().clamp(-128, 127).to(torch.int8)

        return quantized, scales.squeeze()

    def _dequantize(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dequantize INT8 embeddings to float."""
        if indices is not None:
            selected_scales = scales[indices].unsqueeze(-1).unsqueeze(-1)
        else:
            selected_scales = scales.unsqueeze(-1).unsqueeze(-1)

        return quantized.float() * selected_scales

    def search(
        self,
        query: Optional[str] = None,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        query_embedding: Optional[torch.Tensor] = None,
        candidate_ids: Optional[List[Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for similar visual documents.

        Args:
            query: Not used (query_embedding required)
            top_k: Number of results
            collection_name: Collection to search
            query_embedding: Visual query embedding (P, D) or (1, P, D)

        Returns:
            List of SearchResult objects

        Raises:
            RuntimeError: If index not built
            ValueError: If query_embedding not provided
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        if query_embedding is None:
            raise ValueError(
                "ColPali requires query_embedding. "
                "Use latence.colpali.embed_query() first."
            )

        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding)

        query_embedding = query_embedding.to(self.device, dtype=torch.float32)

        if query_embedding.dim() not in {2, 3}:
            raise ValueError("query_embedding must have shape (tokens, dim) or (1, tokens, dim)")
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(0)
        elif query_embedding.shape[0] != 1:
            raise ValueError("ColPaliEngine.search currently supports exactly one query at a time")

        if query_embedding.shape[-1] != self.config.embed_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.config.embed_dim}, "
                f"got {query_embedding.shape[-1]}"
            )

        direct_gather = candidate_ids is not None
        heap: list[tuple[float, Any]] = []
        load_ms = 0.0
        materialize_ms = 0.0
        mask_ms = 0.0
        score_ms = 0.0
        heap_ms = 0.0
        chunk_count = 0
        docs_scanned = 0
        chunk_views = (
            self._iter_requested_chunk_views(candidate_ids)
            if direct_gather
            else self._iter_active_chunk_views()
        )
        with torch.inference_mode():
            for _, chunk_doc_ids, positions, lengths, embeddings, scales in chunk_views:
                chunk_count += 1
                docs_scanned += len(chunk_doc_ids)
                load_ms += self._last_chunk_load_ms

                materialize_start = time.perf_counter()
                chunk_embeddings = self._materialize_chunk_embeddings(embeddings, positions, scales)
                materialize_ms += (time.perf_counter() - materialize_start) * 1000.0

                mask_start = time.perf_counter()
                documents_mask = self._build_documents_mask(
                    lengths=lengths,
                    max_tokens=int(chunk_embeddings.shape[1]),
                    device=chunk_embeddings.device,
                )
                mask_ms += (time.perf_counter() - mask_start) * 1000.0

                score_start = time.perf_counter()
                scores = self._compute_maxsim(
                    query_embedding,
                    chunk_embeddings,
                    documents_mask=documents_mask,
                )
                score_ms += (time.perf_counter() - score_start) * 1000.0

                heap_start = time.perf_counter()
                local_k = min(top_k, len(chunk_doc_ids))
                local_scores, local_indices = torch.topk(scores, k=local_k)
                for score, local_idx in zip(local_scores.detach().cpu().tolist(), local_indices.detach().cpu().tolist()):
                    candidate = (float(score), chunk_doc_ids[int(local_idx)])
                    if len(heap) < top_k:
                        heapq.heappush(heap, candidate)
                    elif candidate[0] > heap[0][0]:
                        heapq.heapreplace(heap, candidate)
                heap_ms += (time.perf_counter() - heap_start) * 1000.0

        sorted_hits = sorted(heap, key=lambda item: item[0], reverse=True)
        self.last_search_profile = {
            "mode": "colpali_search",
            "chunk_count": chunk_count,
            "doc_count": docs_scanned,
            "load_ms": load_ms,
            "materialize_ms": materialize_ms,
            "mask_ms": mask_ms,
            "score_ms": score_ms,
            "heap_ms": heap_ms,
            "direct_gather": direct_gather,
            "requested_candidate_count": len(candidate_ids) if candidate_ids is not None else None,
            "quantized_storage": bool(self.config.use_quantization),
            "storage_type": self.config.storage_type,
        }
        results = [
            SearchResult(
                doc_id=doc_id,
                score=score,
                rank=0,
                source="colpali",
                metadata={"collection": collection_name},
            )
            for score, doc_id in sorted_hits
        ]
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        return results

    def _compute_maxsim(
        self,
        query: torch.Tensor,
        documents: torch.Tensor,
        documents_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MaxSim scores between query and documents.

        MaxSim = Σ_q max_d(sim(q, d))

        Args:
            query: Query embedding (1, Q, D)
            documents: Document embeddings (N, P, D)

        Returns:
            Scores (N,)
        """
        target_dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        return fast_colbert_scores(
            query.to(target_dtype),
            documents.to(target_dtype),
            documents_mask=documents_mask,
            use_quantization=False,
        )[0]

    def add_documents(
        self,
        embeddings: torch.Tensor,
        doc_ids: Optional[List[Any]] = None,
        lengths: Optional[List[int]] = None,
    ) -> None:
        """
        Add documents to existing index.

        Args:
            embeddings: Visual embeddings to add
            doc_ids: Document IDs
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)

        embeddings = embeddings.to(torch.float32)
        if embeddings.dim() != 3:
            raise ValueError("Expected 3D embeddings (N, patches, dim)")
        normalized_doc_ids = self._normalize_doc_ids(doc_ids, len(embeddings))
        lengths = self._validate_lengths(embeddings, lengths)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.chunks_path.mkdir(parents=True, exist_ok=True)
        self._append_chunk(embeddings, normalized_doc_ids, lengths)
        self._append_to_screening_index(
            doc_ids=normalized_doc_ids,
            embeddings=embeddings,
            lengths=lengths,
        )
        logger.info(f"Added {len(embeddings)} visual documents")

    def delete_documents(self, doc_ids: List[Any]) -> None:
        """
        Delete documents from index.

        Args:
            doc_ids: Document IDs to delete
        """
        known_ids = set(self._doc_ids)
        to_delete = [doc_id for doc_id in doc_ids if doc_id in known_ids]
        if not to_delete:
            logger.warning("No documents matched for deletion")
            return
        self._deleted_ids.update(to_delete)
        self._rebuild_doc_ids()
        self._save_manifest()
        self._delete_from_screening_index(to_delete)
        logger.info("Deleted %s visual documents", len(to_delete))

    def compact(self) -> None:
        active_embeddings: List[np.ndarray] = []
        active_doc_ids: List[Any] = []
        active_lengths: List[int] = []
        for _, chunk_doc_ids, positions, lengths, embeddings, scales in self._iter_active_chunk_views():
            chunk_tensors = self._materialize_chunk_embeddings(embeddings, positions, scales).detach().cpu().numpy()
            for idx, doc_id in enumerate(chunk_doc_ids):
                active_doc_ids.append(doc_id)
                active_lengths.append(int(lengths[idx]))
                active_embeddings.append(chunk_tensors[idx])

        if not active_doc_ids:
            self._reset_storage()
            self._set_screening_state(
                health=SCREENING_HEALTH_DISABLED,
                reason="empty_index",
            )
            return

        stacked = torch.from_numpy(np.stack(active_embeddings).astype(np.float32, copy=False))
        lengths_array = np.asarray(active_lengths, dtype=np.int32)
        self._reset_storage()
        self._append_chunk(stacked, active_doc_ids, lengths_array)
        self.rebuild_prototype_screening_index()
        logger.info("Compacted ColPali storage to %s active documents", len(active_doc_ids))

    def get_statistics(self) -> dict:
        """Get engine statistics."""
        if not self._chunks:
            return {
                "engine": "colpali",
                "indexed": False,
                "screening_backend": self.config.screening_backend,
                "screening_state": dict(self.screening_state),
                "direct_candidate_gather": True,
            }

        active_lengths = [
            int(length)
            for chunk in self._chunks
            for doc_id, length in zip(chunk["doc_ids"], chunk["lengths"])
            if doc_id not in self._deleted_ids
        ]
        storage_bytes = sum(
            (self.chunks_path / chunk["file"]).stat().st_size
            for chunk in self._chunks
            if (self.chunks_path / chunk["file"]).exists()
        )
        return {
            "engine": "colpali",
            "indexed": self._indexed,
            "num_documents": len(self._doc_ids),
            "embed_dim": self.config.embed_dim,
            "max_patches": max(active_lengths) if active_lengths else 0,
            "quantized": self.config.use_quantization,
            "storage_mb": storage_bytes / (1024 * 1024),
            "direct_candidate_gather": True,
            "screening_backend": self.config.screening_backend,
            "screening_state": dict(self.screening_state),
            "screening": (
                self.screening_index.get_statistics()
                if self.screening_index is not None
                else {"engine": "screening", "disabled": True}
            ),
            "prototype_screening": (
                self.screening_index.get_statistics()
                if self.screening_index is not None
                else {"engine": "prototype_screening", "disabled": True}
            ),
        }

    def embed_query(self, query: str, model: Any = None) -> torch.Tensor:
        """
        Convert query to visual embedding.

        Note: For ColPali, visual queries require an image or the ColPali model.
        This method is a placeholder - use latence.colpali.embed_query() instead.
        """
        raise NotImplementedError(
            "ColPali requires visual query embedding. "
            "Use latence.colpali.embed_query(image) instead."
        )

    def embed_documents(self, documents: List[str], model: Any = None) -> torch.Tensor:
        """
        Convert documents to visual embeddings.

        Note: For ColPali, document embedding requires images/PDFs.
        This method is a placeholder - use latence.colpali.embed_documents() instead.
        """
        raise NotImplementedError(
            "ColPali requires visual document embedding. "
            "Use latence.colpali.embed_documents(images) instead."
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        self._embeddings = None
        self._scales = None
        if self.screening_index is not None:
            self.screening_index.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ColPali engine cleanup complete")

    def close(self) -> None:
        self.cleanup()


class MultiModalEngine:
    """
    Multi-modal search engine combining ColBERT (text) and ColPali (visual).

    Enables hybrid search across text and visual documents using
    the same MaxSim scoring mechanism with different embedding models.

    Example:
        >>> engine = MultiModalEngine(
        ...     colbert_path="/data/colbert",
        ...     colpali_path="/data/colpali"
        ... )
        >>>
        >>> # Index documents with both modalities
        >>> engine.index(
        ...     text_embeddings=colbert_embs,
        ...     visual_embeddings=colpali_embs,
        ...     doc_ids=doc_ids
        ... )
        >>>
        >>> # Search with text query
        >>> results = engine.search(
        ...     text_embedding=text_query,
        ...     visual_embedding=visual_query,
        ...     fusion_weight=0.5  # 0=text only, 1=visual only
        ... )
    """

    def __init__(
        self,
        colbert_path: Union[str, Path],
        colpali_path: Union[str, Path],
        colbert_config: Optional[IndexConfig] = None,
        colpali_config: Optional[ColPaliConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize multi-modal engine.

        Args:
            colbert_path: Path for ColBERT text index
            colpali_path: Path for ColPali visual index
            colbert_config: ColBERT configuration
            colpali_config: ColPali configuration
            device: Compute device
        """
        from .colbert import ColBERTEngine

        self.colbert = ColBERTEngine(colbert_path, colbert_config, device)
        self.colpali = ColPaliEngine(colpali_path, colpali_config, device)
        self.device = device

    def index(
        self,
        text_embeddings: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        doc_ids: Optional[List[Any]] = None,
        collection_name: str = "default",
    ) -> None:
        """
        Index documents with text and/or visual embeddings.

        Args:
            text_embeddings: ColBERT text embeddings
            visual_embeddings: ColPali visual embeddings
            doc_ids: Document IDs
            collection_name: Collection name
        """
        if text_embeddings is not None:
            self.colbert.index(
                documents=text_embeddings,
                doc_ids=doc_ids,
                collection_name=collection_name,
                embeddings=text_embeddings,
            )

        if visual_embeddings is not None:
            self.colpali.index(
                embeddings=visual_embeddings,
                doc_ids=doc_ids,
                collection_name=collection_name,
            )

    def search(
        self,
        text_embedding: Optional[torch.Tensor] = None,
        visual_embedding: Optional[torch.Tensor] = None,
        top_k: int = 10,
        fusion_weight: float = 0.5,
        fusion_method: str = "weighted",
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Multi-modal search combining text and visual results.

        Args:
            text_embedding: ColBERT query embedding
            visual_embedding: ColPali query embedding
            top_k: Number of results
            fusion_weight: Weight for visual (0=text only, 1=visual only)
            fusion_method: "weighted" or "rrf"
            collection_name: Collection to search

        Returns:
            List of SearchResult objects
        """
        text_results = []
        visual_results = []

        # Get text results
        if text_embedding is not None and self.colbert._indexed:
            text_results = self.colbert.search(
                query_embedding=text_embedding,
                top_k=top_k * 2,  # Get more for fusion
                collection_name=collection_name,
            )

        # Get visual results
        if visual_embedding is not None and self.colpali._indexed:
            visual_results = self.colpali.search(
                query_embedding=visual_embedding,
                top_k=top_k * 2,
                collection_name=collection_name,
            )

        # Fusion
        if fusion_method == "weighted":
            return self._weighted_fusion(
                text_results, visual_results,
                fusion_weight, top_k
            )
        else:
            return self._rrf_fusion(text_results, visual_results, top_k)

    def _weighted_fusion(
        self,
        text_results: List[SearchResult],
        visual_results: List[SearchResult],
        visual_weight: float,
        top_k: int,
    ) -> List[SearchResult]:
        """Combine results using weighted score fusion."""
        text_weight = 1.0 - visual_weight

        # Build score map
        scores = {}

        for r in text_results:
            scores[r.doc_id] = scores.get(r.doc_id, 0) + text_weight * r.score

        for r in visual_results:
            scores[r.doc_id] = scores.get(r.doc_id, 0) + visual_weight * r.score

        # Sort and build results
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank + 1,
                source="multimodal",
                metadata={"fusion_weight": visual_weight}
            ))

        return results

    def _rrf_fusion(
        self,
        text_results: List[SearchResult],
        visual_results: List[SearchResult],
        top_k: int,
        k: int = 60,
    ) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion."""
        scores = {}

        for r in text_results:
            scores[r.doc_id] = scores.get(r.doc_id, 0) + 1.0 / (k + r.rank)

        for r in visual_results:
            scores[r.doc_id] = scores.get(r.doc_id, 0) + 1.0 / (k + r.rank)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank + 1,
                source="multimodal-rrf",
                metadata={}
            ))

        return results

    def get_statistics(self) -> dict:
        """Get combined statistics."""
        return {
            "colbert": self.colbert.get_statistics(),
            "colpali": self.colpali.get_statistics(),
        }


__all__ = [
    'ColPaliEngine',
    'ColPaliConfig',
    'MultiModalEngine',
]

