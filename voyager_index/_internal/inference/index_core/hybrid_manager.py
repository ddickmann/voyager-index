"""
Hybrid Search Manager
=====================

Orchestrates fusion between HNSW (Dense) and BM25 (Sparse) retrieval.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bm25s
    import Stemmer

    _BM25S_AVAILABLE = True
except ImportError:
    bm25s = None  # type: ignore[assignment]
    Stemmer = None  # type: ignore[assignment]
    _BM25S_AVAILABLE = False

try:
    from voyager_index._internal.inference.engines.bm25 import BM25Engine as LegacyBM25Engine
except ImportError:
    LegacyBM25Engine = None

from voyager_index._internal.inference.stateless_optimizer import (
    GpuFulfilmentPipeline,
)

logger = logging.getLogger(__name__)
DEFAULT_REFINE_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_REFINE_NLI_MODEL = "cross-encoder/nli-distilroberta-base"

try:
    from .hnsw_manager import HnswSegmentManager
except ImportError:
    HnswSegmentManager = None
    logger.warning("HnswSegmentManager not available, hybrid search will fail.")

try:
    import latence_solver
    from latence_solver import SolverConfig, SolverConstraints, TabuSearchSolver
except ImportError as e:
    latence_solver = None
    SolverConfig = None
    TabuSearchSolver = None
    SolverConstraints = None
    logger.warning(f"latence_solver not available, refinement will fail. Error: {e}")


class HybridSearchManager:
    """
    Manages both HNSW and BM25 indices for hybrid retrieval.

    Architecture:
    - Dense: HnswSegmentManager (RocksDB/Mmap)
    - Sparse: bm25s (Memory-mapped)
    - Refiner: canonical stateless fulfilment optimizer backed by `latence_solver`

    Fusion Strategy:
    - Parallel Retrieval: Get top-k candidates from both.
    - Candidate Union: Merge candidate IDs.
    - Solver Refinement: Build the canonical optimizer request and select an optimal subset
      from the fused candidate pool.
    """

    def __init__(
        self,
        shard_path: Path,
        dim: int,
        stemmer_lang: str = "english",
        on_disk: bool = True,
        multivector_comparator: Optional[str] = None,
        roq_bits: Optional[int] = None,
        distance_metric: str = "cosine",
        m: int = 16,
        ef_construct: int = 100,
        dense_engine: str = "hnsw",
        dense_engine_config: Optional[Any] = None,
    ):
        self.shard_path = Path(shard_path)
        self.shard_path.mkdir(parents=True, exist_ok=True)
        self.roq_bits = roq_bits
        self.distance_metric = distance_metric
        self.m = m
        self.ef_construct = ef_construct
        self._dense_engine_type = dense_engine

        # Dense Index
        if dense_engine == "shard":
            from voyager_index._internal.inference.shard_engine.manager import (
                ShardEngineConfig,
                ShardSegmentManager,
            )

            shard_config = dense_engine_config or ShardEngineConfig(dim=dim)
            device = getattr(shard_config, "device", "cuda") if dense_engine_config else "cuda"
            self.hnsw = ShardSegmentManager(
                path=self.shard_path / "shard_dense",
                config=shard_config,
                device=device,
            )
        else:
            if HnswSegmentManager is None:
                raise ImportError(
                    "HybridSearchManager requires HnswSegmentManager. "
                    "Install the native HNSW support or use a build that includes it."
                )
            self.hnsw = HnswSegmentManager(
                self.shard_path / "hnsw",
                dim=dim,
                distance_metric=distance_metric,
                m=m,
                ef_construct=ef_construct,
                on_disk=on_disk,
                multivector_comparator=multivector_comparator,
                roq_bits=roq_bits,
            )

        # Sparse Index
        self.bm25_path = self.shard_path / "bm25"
        self.bm25_params_path = self.bm25_path / "params.index.json"
        self.stemmer = Stemmer.Stemmer(stemmer_lang) if Stemmer is not None else None
        self.retriever = None
        self._legacy_bm25: Any = None
        self.sparse_error: Optional[str] = None
        self.sparse_index_present = self.bm25_params_path.exists()
        self.sparse_dirty = False

        # Buffer for real-time updates (bm25s is static)
        # TODO: Implement dynamic buffer or periodic re-indexing
        self.corpus_buffer: List[str] = []
        self.ids_buffer: List[int] = []
        self.payload_buffer: List[Dict[str, Any]] = []

        self._load_bm25()

        # Solver
        if TabuSearchSolver:
            self.solver_backend_status = (
                latence_solver.backend_status()
                if latence_solver is not None and hasattr(latence_solver, "backend_status")
                else {"cpu_reference_available": True}
            )
            self.solver = TabuSearchSolver(
                SolverConfig(
                    alpha=1.0,
                    beta=0.24,
                    gamma=0.18,
                    delta=0.12,
                    epsilon=0.14,
                    lambda_=0.32,
                    iterations=48,
                    tabu_tenure=14,
                    early_stopping_patience=18,
                    use_gpu=False,
                    random_seed=11,
                )
            )
        else:
            self.solver_backend_status = {}
            self.solver = None
        self.solver_available = self.solver is not None
        self._optimizer_pipeline: Optional[GpuFulfilmentPipeline] = None
        self._last_search_context: Optional[Dict[str, Any]] = None
        self._last_refine_context: Optional[Dict[str, Any]] = None
        self._cross_encoder_models: Dict[str, Any] = {}
        self._nli_models: Dict[str, Any] = {}

    def _load_bm25(self):
        """Load BM25 index if exists, falling back to LegacyBM25Engine."""
        self.sparse_index_present = self.bm25_params_path.exists()
        if not self.sparse_index_present:
            self.retriever = None
            self.sparse_error = None
            return
        try:
            json.loads(self.bm25_params_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.retriever = None
            self._legacy_bm25 = None
            self.sparse_error = f"bm25 metadata corrupted: {exc}"
            return
        if _BM25S_AVAILABLE:
            try:
                self.retriever = bm25s.BM25.load(str(self.bm25_path), load_corpus=True)
                self.sparse_error = None
                logger.info("Loaded BM25 index (bm25s)")
                return
            except Exception as e:
                logger.warning("bm25s load failed, trying legacy fallback: %s", e)
        if LegacyBM25Engine is not None and getattr(self, "corpus_buffer", None):
            try:
                self._legacy_bm25 = LegacyBM25Engine()
                self._legacy_bm25.index_documents(self.corpus_buffer, self.ids_buffer)
                self.sparse_error = None
                logger.info("Loaded BM25 index (legacy fallback)")
                return
            except Exception as e2:
                logger.error("Legacy BM25 fallback also failed: %s", e2)
        self.retriever = None
        if not _BM25S_AVAILABLE and not self.corpus_buffer:
            self.sparse_error = "bm25s unavailable; corpus not yet loaded for legacy fallback"
        else:
            self.sparse_error = "bm25s unavailable and legacy fallback failed"

    def _write_bm25_metadata(self, backend: str) -> None:
        self.bm25_path.mkdir(parents=True, exist_ok=True)
        if self.bm25_params_path.exists():
            return
        self.bm25_params_path.write_text(
            json.dumps({"backend": backend, "doc_count": len(self.ids_buffer)}),
            encoding="utf-8",
        )

    def _swap_bm25_generation(self, staged_path: Path) -> None:
        current_path = self.bm25_path
        replacement_path = current_path.with_name(f"{current_path.name}.replace")
        backup_path = current_path.with_name(f"{current_path.name}.backup")

        if replacement_path.exists():
            shutil.rmtree(replacement_path, ignore_errors=True)
        staged_path.rename(replacement_path)

        try:
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            if current_path.exists():
                current_path.rename(backup_path)
            replacement_path.rename(current_path)
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
        except Exception:
            if current_path.exists():
                shutil.rmtree(current_path, ignore_errors=True)
            if backup_path.exists():
                backup_path.rename(current_path)
            raise

    def _ensure_sparse_ready(self) -> None:
        if self.sparse_dirty:
            self.rebuild_sparse_state(self.corpus_buffer, self.ids_buffer, self.payload_buffer)

    def mark_sparse_dirty(
        self,
        corpus: List[str],
        ids: List[int],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.restore_buffers(corpus, ids, payloads)
        self.sparse_dirty = True

    @staticmethod
    def _matches_filter(payload: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if payload.get(key) != value:
                return False
        return True

    @staticmethod
    def _sanitize_sparse_text(text: str, doc_id: int) -> str:
        normalized = (text or "").strip()
        if normalized:
            return normalized
        return f"voyager-doc-{doc_id}"

    def restore_buffers(
        self,
        corpus: List[str],
        ids: List[int],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.corpus_buffer = list(corpus)
        self.ids_buffer = [int(item_id) for item_id in ids]
        self.payload_buffer = [dict(payload or {}) for payload in (payloads or [])]
        if len(self.payload_buffer) < len(self.ids_buffer):
            self.payload_buffer.extend({} for _ in range(len(self.ids_buffer) - len(self.payload_buffer)))

    def rebuild_sparse_state(
        self,
        corpus: List[str],
        ids: List[int],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.restore_buffers(corpus, ids, payloads)
        if not self.ids_buffer:
            self.retriever = None
            self.sparse_error = None
            self.sparse_index_present = False
            self.sparse_dirty = False
            if self.bm25_path.exists():
                shutil.rmtree(self.bm25_path, ignore_errors=True)
            return

        sanitized_corpus = [
            self._sanitize_sparse_text(text, doc_id) for text, doc_id in zip(self.corpus_buffer, self.ids_buffer)
        ]

        if _BM25S_AVAILABLE:
            staged_dir: Optional[Path] = None
            try:
                corpus_tokens = bm25s.tokenize(sanitized_corpus, stemmer=self.stemmer)
                staged_dir = Path(tempfile.mkdtemp(prefix="voyager-bm25-", dir=str(self.shard_path)))
                next_retriever = bm25s.BM25()
                next_retriever.index(corpus_tokens)
                next_retriever.save(str(staged_dir))
                self._swap_bm25_generation(staged_dir)
                self._write_bm25_metadata("bm25s")
                self.retriever = bm25s.BM25.load(str(self.bm25_path), load_corpus=True)
                self._legacy_bm25 = None
                self.sparse_error = None
                self.sparse_index_present = True
                self.sparse_dirty = False
                return
            except Exception as exc:
                logger.warning("bm25s rebuild failed, trying legacy: %s", exc)
            finally:
                if staged_dir is not None and staged_dir.exists():
                    shutil.rmtree(staged_dir, ignore_errors=True)

        if LegacyBM25Engine is not None:
            try:
                engine = LegacyBM25Engine()
                engine.index_documents(sanitized_corpus, self.ids_buffer)
                self._legacy_bm25 = engine
                self.retriever = None
                self._write_bm25_metadata("legacy")
                self.sparse_error = None
                self.sparse_index_present = True
                self.sparse_dirty = False
                logger.info("BM25 rebuilt with legacy fallback (%d docs)", len(sanitized_corpus))
                return
            except Exception as exc:
                logger.error("Legacy BM25 rebuild also failed: %s", exc)

        self.retriever = None
        self._legacy_bm25 = None
        self.sparse_error = "Both bm25s and legacy BM25 are unavailable"
        raise RuntimeError(self.sparse_error)

    def index(
        self, corpus: List[str], vectors: np.ndarray, ids: List[int], payloads: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Index documents into both HNSW and BM25.

        Note: BM25s currently requires full re-indexing for optimization.
        For real-time, we append to buffer and re-index periodically.
        """
        if isinstance(vectors, np.ndarray) and vectors.ndim == 3:
            raise ValueError("3D arrays not supported; pass a list of 2D arrays for multi-vector docs")
        if self._dense_engine_type == "shard":
            if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
                vecs_list = [vectors[i : i + 1] for i in range(vectors.shape[0])]
            elif isinstance(vectors, list):
                vecs_list = vectors
            else:
                vecs_list = [vectors]
            self.hnsw.add_multidense(vecs_list, ids=ids, payloads=payloads)
        else:
            self.hnsw.add(vectors, ids=ids, payloads=payloads)
        next_corpus = [*self.corpus_buffer, *corpus]
        next_ids = [*self.ids_buffer, *[int(item_id) for item_id in ids]]
        next_payloads = [*self.payload_buffer, *[dict(payload or {}) for payload in (payloads or [])]]
        self.mark_sparse_dirty(next_corpus, next_ids, next_payloads)

    def index_multivector(
        self,
        corpus: List[str],
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Index multi-vector late-interaction documents into HNSW and BM25.
        """
        self.hnsw.add_multidense(vectors, ids=ids, payloads=payloads)
        next_corpus = [*self.corpus_buffer, *corpus]
        next_ids = [*self.ids_buffer, *[int(item_id) for item_id in ids]]
        next_payloads = [*self.payload_buffer, *[dict(payload or {}) for payload in (payloads or [])]]
        self.mark_sparse_dirty(next_corpus, next_ids, next_payloads)

    def search(
        self,
        query_text: str,
        query_vector: Optional[np.ndarray],
        k: int = 50,
        filters: Optional[Dict[str, Any]] = None,
        dense_search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform Hybrid Search.

        Returns:
            Dict containing:
            - 'dense': List[(id, score)]
            - 'sparse': List[(id, score)]
            - 'union_ids': List[int]
        """
        dense_results = []
        dense_search_kwargs = dict(dense_search_kwargs or {})
        if query_vector is not None:
            dense_query = query_vector
            if self._dense_engine_type == "shard" and isinstance(query_vector, np.ndarray) and query_vector.ndim == 1:
                dense_query = query_vector.reshape(1, -1)
            if self._dense_engine_type == "shard" and hasattr(self.hnsw, "search_multivector"):
                dense_results = self.hnsw.search_multivector(
                    dense_query,
                    k=k,
                    filters=filters,
                    **dense_search_kwargs,
                )
            else:
                dense_results = self.hnsw.search(dense_query, k=k, filters=filters)
        dense_ids = {rid for rid, _ in dense_results}

        sparse_results = []
        if self.retriever and query_text.strip() and _BM25S_AVAILABLE:
            self._ensure_sparse_ready()
            query_tokens = bm25s.tokenize([query_text], stemmer=self.stemmer)
            docs, scores = self.retriever.retrieve(query_tokens, k=min(k, len(self.ids_buffer)))
            if docs.shape[1] > 0:
                for doc_idx, score in zip(docs[0], scores[0]):
                    if doc_idx >= len(self.ids_buffer):
                        continue
                    payload = self.payload_buffer[doc_idx] if doc_idx < len(self.payload_buffer) else {}
                    if not self._matches_filter(payload, filters):
                        continue
                    external_id = self.ids_buffer[doc_idx]
                    sparse_results.append((external_id, float(score)))
        elif self._legacy_bm25 and query_text.strip():
            self._ensure_sparse_ready()
            results = self._legacy_bm25.search(query_text, top_k=k)
            payload_by_id = {
                int(doc_id): (self.payload_buffer[idx] if idx < len(self.payload_buffer) else {})
                for idx, doc_id in enumerate(self.ids_buffer)
            }
            for sr in results:
                payload = payload_by_id.get(int(sr.doc_id), {})
                if not self._matches_filter(payload, filters):
                    continue
                sparse_results.append((sr.doc_id, sr.score))
        elif query_text.strip() and self.sparse_dirty:
            self._ensure_sparse_ready()
            return self.search(query_text=query_text, query_vector=query_vector, k=k, filters=filters)

        sparse_ids = {rid for rid, _ in sparse_results}

        # 3. Union
        union_ids = list(dense_ids.union(sparse_ids))

        dense_meta = {
            int(doc_id): {"dense_score": float(score), "dense_rank": rank}
            for rank, (doc_id, score) in enumerate(dense_results, start=1)
        }
        sparse_meta = {
            int(doc_id): {"sparse_score": float(score), "sparse_rank": rank}
            for rank, (doc_id, score) in enumerate(sparse_results, start=1)
        }
        rrf_meta: Dict[int, float] = {}
        rrf_k = 60.0
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_meta[int(doc_id)] = rrf_meta.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            rrf_meta[int(doc_id)] = rrf_meta.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
        self._last_search_context = {
            "dense": dense_meta,
            "sparse": sparse_meta,
            "rrf": rrf_meta,
            "query_text": query_text,
        }

        return {
            "dense": dense_results,
            "sparse": sparse_results,
            "union_ids": union_ids,
            "sparse_error": self.sparse_error,
        }

    @staticmethod
    def _resolve_refine_options(
        query_payload: Optional[Dict[str, Any]] = None,
        refine_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        if isinstance(query_payload, dict) and isinstance(query_payload.get("refine_options"), dict):
            resolved.update(dict(query_payload["refine_options"]))
        if isinstance(refine_options, dict):
            resolved.update(dict(refine_options))
        resolved["use_cross_encoder"] = bool(
            resolved.get("use_cross_encoder") or resolved.get("refine_use_cross_encoder")
        )
        resolved["cross_encoder_model"] = str(resolved.get("cross_encoder_model") or DEFAULT_REFINE_CROSS_ENCODER_MODEL)
        resolved["cross_encoder_top_k"] = max(
            1,
            int(resolved.get("cross_encoder_top_k") or resolved.get("rerank_top_k") or 96),
        )
        resolved["cross_encoder_batch_size"] = max(
            1,
            int(resolved.get("cross_encoder_batch_size") or 16),
        )
        resolved["use_nli"] = bool(resolved.get("use_nli") or resolved.get("refine_use_nli"))
        resolved["nli_model"] = str(resolved.get("nli_model") or DEFAULT_REFINE_NLI_MODEL)
        resolved["nli_top_k"] = max(
            1,
            int(
                resolved.get("nli_top_k") or resolved.get("utility_top_k") or resolved.get("cross_encoder_top_k") or 64
            ),
        )
        resolved["nli_batch_size"] = max(
            1,
            int(resolved.get("nli_batch_size") or resolved.get("cross_encoder_batch_size") or 16),
        )
        resolved["nli_promote_base_relevance"] = bool(
            resolved.get("nli_promote_base_relevance") or resolved.get("promote_base_relevance")
        )
        resolved["confidence_gating"] = bool(
            resolved.get("confidence_gating") or resolved.get("refine_confidence_gating")
        )
        resolved["confidence_gap_threshold"] = float(resolved.get("confidence_gap_threshold") or 0.20)
        resolved["confidence_min_candidates"] = max(
            2,
            int(resolved.get("confidence_min_candidates") or 16),
        )
        gate_mode = str(resolved.get("solver_gate_mode") or resolved.get("production_gate_mode") or "auto").strip().lower()
        if gate_mode not in {"auto", "always", "off", "disabled"}:
            gate_mode = "auto"
        resolved["solver_gate_mode"] = gate_mode
        resolved["solver_gate_min_candidates"] = max(
            1,
            int(resolved.get("solver_gate_min_candidates") or 24),
        )
        resolved["solver_gate_max_candidates"] = max(
            int(resolved["solver_gate_min_candidates"]),
            int(resolved.get("solver_gate_max_candidates") or 96),
        )
        resolved["solver_gate_min_headroom"] = max(
            0,
            int(resolved.get("solver_gate_min_headroom") or 12),
        )
        resolved["solver_gate_min_budget_pressure"] = max(
            0.0,
            float(resolved.get("solver_gate_min_budget_pressure") or 1.35),
        )
        resolved["solver_gate_score_gap_threshold"] = min(
            1.0,
            max(0.0, float(resolved.get("solver_gate_score_gap_threshold") or 0.18)),
        )
        resolved["production_lambda_cap"] = min(
            1.0,
            max(0.0, float(resolved.get("production_lambda_cap") or 0.05)),
        )
        resolved["production_lambda_large_candidate_threshold"] = max(
            1,
            int(resolved.get("production_lambda_large_candidate_threshold") or 48),
        )
        return resolved

    @staticmethod
    def _cross_encoder_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _get_cross_encoder(self, model_name: str) -> Any:
        cache = getattr(self, "_cross_encoder_models", None)
        if cache is None:
            cache = {}
            self._cross_encoder_models = cache
        if model_name not in cache:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on optional dependency
                raise ImportError("Cross-encoder refinement requires sentence-transformers to be installed") from exc
            device = self._cross_encoder_device()
            logger.info("Loading cross-encoder %s for hybrid refine (device=%s)", model_name, device)
            cache[model_name] = CrossEncoder(model_name, device=device)
        return cache[model_name]

    def _get_nli_model(self, model_name: str) -> Dict[str, Any]:
        cache = getattr(self, "_nli_models", None)
        if cache is None:
            cache = {}
            self._nli_models = cache
        if model_name not in cache:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError("NLI refinement requires transformers to be installed") from exc
            device_name = self._cross_encoder_device()
            model_device = torch.device("cpu" if device_name == "cpu" else device_name)
            logger.info("Loading NLI model %s for hybrid refine (device=%s)", model_name, model_device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(model_device)
            model.eval()
            cache[model_name] = {
                "tokenizer": tokenizer,
                "model": model,
                "device": model_device,
                "id2label": {
                    int(idx): str(label) for idx, label in dict(getattr(model.config, "id2label", {}) or {}).items()
                },
            }
        return cache[model_name]

    @staticmethod
    def _nli_probability_summary(probabilities: np.ndarray, id2label: Dict[int, str]) -> tuple[float, float, float]:
        entailment = 0.0
        contradiction = 0.0
        neutral = 0.0
        if probabilities.ndim != 1:
            probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        for idx, probability in enumerate(probabilities.tolist()):
            label = str(id2label.get(idx, f"label_{idx}")).lower()
            if any(token in label for token in ("entail", "support")):
                entailment = max(entailment, float(probability))
            elif any(token in label for token in ("contrad", "refute")):
                contradiction = max(contradiction, float(probability))
            elif any(token in label for token in ("neutral", "unknown", "other", "nei")):
                neutral = max(neutral, float(probability))
        if entailment <= 0.0 and contradiction <= 0.0 and neutral <= 0.0 and probabilities.size > 0:
            entailment = float(np.max(probabilities))
        return entailment, contradiction, neutral

    @staticmethod
    def _confidence_gate_summary(
        valid_items: List[Dict[str, Any]],
        retrieval_features: Dict[int, Dict[str, Any]],
        refine_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(refine_options.get("confidence_gating")):
            return {"enabled": False, "applied": False, "reason": "disabled"}
        if len(valid_items) < int(refine_options.get("confidence_min_candidates", 16)):
            return {
                "enabled": True,
                "applied": False,
                "reason": "too_few_candidates",
                "candidate_count": len(valid_items),
            }
        ordered_scores = sorted(
            (
                float(retrieval_features.get(int(item["id"]), {}).get("base_relevance", 0.0) or 0.0)
                for item in valid_items
            ),
            reverse=True,
        )
        if len(ordered_scores) < 2:
            return {
                "enabled": True,
                "applied": False,
                "reason": "insufficient_scores",
                "candidate_count": len(valid_items),
            }
        top_score = float(ordered_scores[0])
        runner_up = float(ordered_scores[1])
        gap = max(top_score - runner_up, 0.0)
        applied = gap >= float(refine_options.get("confidence_gap_threshold", 0.20))
        return {
            "enabled": True,
            "applied": bool(applied),
            "reason": "gap_threshold" if applied else "below_threshold",
            "candidate_count": len(valid_items),
            "top_score": top_score,
            "runner_up_score": runner_up,
            "gap": gap,
            "threshold": float(refine_options.get("confidence_gap_threshold", 0.20)),
        }

    @staticmethod
    def _normalize_signal(values: List[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        lo = float(arr.min())
        hi = float(arr.max())
        if hi - lo <= 1e-8:
            return np.full(arr.shape, 0.5, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32, copy=False)

    @staticmethod
    def _has_advanced_query_utility(query_payload: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(query_payload, dict):
            return False
        return any(
            key in query_payload
            for key in (
                "query_token_weights",
                "query_aspect_weights",
                "query_aspects",
                "query_aspect_vectors",
                "query_expansion_vectors",
                "teacher_token_weights",
                "teacher_utility_score",
            )
        )

    @staticmethod
    def _is_lambda_override_explicit(
        solver_config: Optional[Any],
        *,
        default_solver_config: Any = None,
    ) -> bool:
        if solver_config is None:
            return False
        if isinstance(solver_config, dict):
            return "lambda" in solver_config or "lambda_" in solver_config
        if default_solver_config is not None and solver_config is default_solver_config:
            return False
        return any(getattr(solver_config, key, None) is not None for key in ("lambda_", "lambda"))

    def _fallback_candidate_order(
        self,
        solver_candidates: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not solver_candidates:
            return [], {"signal": "none", "normalized_primary_scores": []}

        has_rerank = any(
            isinstance(
                candidate.get(
                    "rerank_score",
                    candidate.get("cross_encoder_score", candidate.get("ce_score", candidate.get("ltr_score"))),
                ),
                (int, float),
            )
            for candidate in solver_candidates
        )
        has_utility = any(
            isinstance(candidate.get("utility_score", candidate.get("nli_utility_score")), (int, float))
            for candidate in solver_candidates
        )
        has_rrf = any(isinstance(candidate.get("rrf_score"), (int, float)) for candidate in solver_candidates)

        signal_name = "base_relevance"
        if has_rerank:
            signal_name = "rerank"
        elif has_utility:
            signal_name = "utility_composite"
        elif has_rrf:
            signal_name = "rrf"

        primary_scores: List[float] = []
        ranked_rows: List[tuple[tuple[float, float, float, float, float, float, float, float], Dict[str, Any]]] = []
        for index, candidate in enumerate(solver_candidates):
            rerank_score = float(
                candidate.get(
                    "rerank_score",
                    candidate.get("cross_encoder_score", candidate.get("ce_score", candidate.get("ltr_score", 0.0))),
                )
                or 0.0
            )
            utility_score = float(candidate.get("utility_score", candidate.get("nli_utility_score", 0.0)) or 0.0)
            entailment = float(candidate.get("nli_entailment", 0.0) or 0.0)
            contradiction = float(candidate.get("nli_contradiction", 0.0) or 0.0)
            base_relevance = float(candidate.get("base_relevance", 0.0) or 0.0)
            rrf_score = float(candidate.get("rrf_score", 0.0) or 0.0)
            dense_score = float(candidate.get("dense_score", 0.0) or 0.0)
            sparse_score = float(candidate.get("sparse_score", 0.0) or 0.0)

            if has_rerank:
                primary = rerank_score
            elif has_utility:
                primary = (0.70 * base_relevance) + (0.30 * utility_score) - (0.20 * contradiction)
            elif has_rrf:
                primary = rrf_score
            else:
                primary = base_relevance
            primary_scores.append(float(primary))
            ranked_rows.append(
                (
                    (
                        float(primary),
                        utility_score,
                        entailment,
                        -contradiction,
                        base_relevance,
                        rrf_score,
                        dense_score,
                        sparse_score - (index * 1e-9),
                    ),
                    candidate,
                )
            )

        normalized_primary = self._normalize_signal(primary_scores)
        normalized_by_id = {
            id(candidate): float(normalized_primary[idx]) for idx, candidate in enumerate(solver_candidates)
        }
        ranked = [candidate for _, candidate in sorted(ranked_rows, key=lambda item: item[0], reverse=True)]
        return ranked, {
            "signal": signal_name,
            "normalized_primary_scores": [normalized_by_id[id(candidate)] for candidate in ranked],
        }

    def _production_solver_gate(
        self,
        *,
        query_text: str,
        query_payload: Optional[Dict[str, Any]],
        solver_candidates: List[Dict[str, Any]],
        constraints_payload: Dict[str, Any],
        refine_options: Dict[str, Any],
        fallback_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        gate_mode = str(refine_options.get("solver_gate_mode", "auto"))
        candidate_count = len(solver_candidates)
        selected_cap = min(
            int(constraints_payload.get("max_chunks", min(10, max(candidate_count, 1)))),
            candidate_count,
        )
        total_candidate_tokens = int(
            sum(int(candidate.get("token_count", 0) or 0) for candidate in solver_candidates)
        )
        max_tokens = int(constraints_payload.get("max_tokens", 0) or 0)
        budget_pressure = (
            float(total_candidate_tokens / float(max_tokens))
            if max_tokens > 0
            else float(candidate_count)
        )
        headroom = int(max(candidate_count - selected_cap, 0))
        normalized_scores = list(fallback_summary.get("normalized_primary_scores", []))
        if len(normalized_scores) >= 2:
            top_gap = float(max(normalized_scores[0] - normalized_scores[1], 0.0))
        elif normalized_scores:
            top_gap = float(normalized_scores[0])
        else:
            top_gap = 0.0
        advanced_query_utility = self._has_advanced_query_utility(query_payload)

        if gate_mode in {"off", "disabled"}:
            return {
                "enabled": False,
                "applied": False,
                "run_solver": True,
                "reason": "disabled",
                "fallback_signal": fallback_summary.get("signal", "none"),
                "candidate_count": candidate_count,
                "selected_cap": selected_cap,
            }
        if gate_mode == "always":
            return {
                "enabled": True,
                "applied": False,
                "run_solver": True,
                "reason": "forced",
                "fallback_signal": fallback_summary.get("signal", "none"),
                "candidate_count": candidate_count,
                "selected_cap": selected_cap,
            }

        reasons: List[str] = []
        if candidate_count < int(refine_options.get("solver_gate_min_candidates", 24)):
            reasons.append("candidate_count_below_min")
        if candidate_count > int(refine_options.get("solver_gate_max_candidates", 96)):
            reasons.append("candidate_count_above_max")
        if headroom < int(refine_options.get("solver_gate_min_headroom", 12)):
            reasons.append("insufficient_choice_headroom")
        if not query_text.strip() and not advanced_query_utility:
            reasons.append("missing_query_signal")
        if (
            budget_pressure < float(refine_options.get("solver_gate_min_budget_pressure", 1.35))
            and not advanced_query_utility
        ):
            reasons.append("weak_budget_pressure")
        if (
            top_gap >= float(refine_options.get("solver_gate_score_gap_threshold", 0.18))
            and not advanced_query_utility
        ):
            reasons.append("confident_head")

        return {
            "enabled": True,
            "applied": bool(reasons),
            "run_solver": not reasons,
            "reasons": reasons,
            "reason": reasons[0] if reasons else "passed",
            "fallback_signal": fallback_summary.get("signal", "none"),
            "candidate_count": candidate_count,
            "selected_cap": selected_cap,
            "total_candidate_tokens": total_candidate_tokens,
            "max_tokens": max_tokens,
            "budget_pressure": budget_pressure,
            "choice_headroom": headroom,
            "top_score_gap": top_gap,
            "advanced_query_utility": advanced_query_utility,
        }

    @staticmethod
    def _apply_production_solver_profile(
        *,
        solver_config_payload: Dict[str, Any],
        candidate_count: int,
        refine_options: Dict[str, Any],
        explicit_lambda_override: bool,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        summary = {
            "applied": False,
            "explicit_lambda_override": bool(explicit_lambda_override),
            "candidate_count": int(candidate_count),
        }
        if explicit_lambda_override:
            summary["reason"] = "explicit_lambda_override"
            return solver_config_payload, summary
        threshold = int(refine_options.get("production_lambda_large_candidate_threshold", 48))
        if candidate_count < threshold:
            summary["reason"] = "candidate_count_below_threshold"
            return solver_config_payload, summary
        current_lambda = float(
            solver_config_payload.get(
                "lambda",
                solver_config_payload.get("lambda_", 0.32),
            )
        )
        lambda_cap = float(refine_options.get("production_lambda_cap", 0.05))
        if current_lambda <= lambda_cap:
            summary["reason"] = "lambda_already_within_cap"
            summary["lambda"] = current_lambda
            return solver_config_payload, summary
        updated = dict(solver_config_payload)
        updated["lambda"] = lambda_cap
        summary.update(
            {
                "applied": True,
                "reason": "lambda_cap",
                "previous_lambda": current_lambda,
                "lambda": lambda_cap,
                "candidate_threshold": threshold,
            }
        )
        return updated, summary

    @staticmethod
    def _payload_rerank_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for key in (
            "base_relevance",
            "rerank_score",
            "cross_encoder_score",
            "ce_score",
            "ltr_score",
            "utility_score",
            "nli_utility_score",
            "nli_entailment",
            "nli_contradiction",
            "nli_neutral",
            "teacher_utility_score",
            "distilled_utility_score",
            "rrf_score",
            "dense_score",
            "sparse_score",
        ):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                metadata[key] = float(value)
        if "base_relevance" not in metadata:
            for key in ("rerank_score", "cross_encoder_score", "ce_score", "ltr_score"):
                if key in metadata:
                    metadata["base_relevance"] = float(metadata[key])
                    break
        for key in ("relevance_source", "rerank_source"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value.strip().lower()
        return metadata

    def _apply_cross_encoder_scores(
        self,
        *,
        valid_items: List[Dict[str, Any]],
        query_text: str,
        retrieval_features: Dict[int, Dict[str, Any]],
        model_name: str,
        batch_size: int,
        top_k: int,
    ) -> Dict[str, Any]:
        if not query_text.strip() or not valid_items:
            return {
                "enabled": False,
                "applied": False,
                "reason": "missing_query_text_or_candidates",
                "model": model_name,
                "reranked_count": 0,
            }
        ordered_ids = sorted(
            (int(item["id"]) for item in valid_items),
            key=lambda doc_id: float(retrieval_features.get(doc_id, {}).get("base_relevance", 0.0)),
            reverse=True,
        )[: min(max(top_k, 1), len(valid_items))]
        if not ordered_ids:
            return {
                "enabled": True,
                "applied": False,
                "reason": "empty_ordered_ids",
                "model": model_name,
                "reranked_count": 0,
            }
        by_id = {int(item["id"]): item for item in valid_items}
        scorer = self._get_cross_encoder(model_name)
        pairs = [(query_text, by_id[doc_id]["text"]) for doc_id in ordered_ids]
        scores = np.asarray(
            scorer.predict(pairs, batch_size=batch_size, show_progress_bar=False),
            dtype=np.float32,
        ).reshape(-1)
        for rank, (doc_id, score) in enumerate(zip(ordered_ids, scores.tolist()), start=1):
            metadata = retrieval_features.setdefault(int(doc_id), {})
            previous_base = float(metadata.get("base_relevance", 0.0) or 0.0)
            metadata.setdefault("pre_rerank_base_relevance", previous_base)
            metadata["cross_encoder_score"] = float(score)
            metadata["rerank_score"] = float(score)
            metadata["cross_encoder_rank"] = rank
            metadata["base_relevance"] = float(score)
            metadata["relevance_source"] = "cross_encoder"
        return {
            "enabled": True,
            "applied": True,
            "model": model_name,
            "reranked_count": len(ordered_ids),
            "batch_size": int(batch_size),
            "top_k": int(top_k),
        }

    def _apply_nli_scores(
        self,
        *,
        valid_items: List[Dict[str, Any]],
        query_text: str,
        retrieval_features: Dict[int, Dict[str, Any]],
        model_name: str,
        batch_size: int,
        top_k: int,
        promote_base_relevance: bool = False,
    ) -> Dict[str, Any]:
        if not query_text.strip() or not valid_items:
            return {
                "enabled": False,
                "applied": False,
                "reason": "missing_query_text_or_candidates",
                "model": model_name,
                "scored_count": 0,
            }
        ordered_ids = sorted(
            (int(item["id"]) for item in valid_items),
            key=lambda doc_id: float(retrieval_features.get(doc_id, {}).get("base_relevance", 0.0)),
            reverse=True,
        )[: min(max(top_k, 1), len(valid_items))]
        if not ordered_ids:
            return {
                "enabled": True,
                "applied": False,
                "reason": "empty_ordered_ids",
                "model": model_name,
                "scored_count": 0,
            }
        bundle = self._get_nli_model(model_name)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        device = bundle["device"]
        id2label = bundle["id2label"]
        by_id = {int(item["id"]): item for item in valid_items}
        utility_values: List[float] = []
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("NLI refinement requires torch to be installed") from exc

        for offset in range(0, len(ordered_ids), max(batch_size, 1)):
            batch_ids = ordered_ids[offset : offset + max(batch_size, 1)]
            batch_documents = [by_id[doc_id]["text"] for doc_id in batch_ids]
            encoded = tokenizer(
                [query_text] * len(batch_documents),
                batch_documents,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for doc_id, row in zip(batch_ids, probabilities):
                entailment, contradiction, neutral = self._nli_probability_summary(
                    np.asarray(row, dtype=np.float32),
                    id2label,
                )
                utility = self._clamp01(entailment * (1.0 - contradiction))
                utility_values.append(float(utility))
                metadata = retrieval_features.setdefault(int(doc_id), {})
                metadata["nli_entailment"] = float(entailment)
                metadata["nli_contradiction"] = float(contradiction)
                metadata["nli_neutral"] = float(neutral)
                metadata["nli_utility_score"] = float(utility)
                metadata["utility_score"] = float(utility)
                metadata["utility_source"] = "nli"
                if promote_base_relevance:
                    previous_base = float(metadata.get("base_relevance", 0.0) or 0.0)
                    metadata.setdefault("pre_nli_base_relevance", previous_base)
                    metadata["base_relevance"] = float(0.70 * previous_base + 0.30 * utility)
        return {
            "enabled": True,
            "applied": True,
            "model": model_name,
            "scored_count": len(ordered_ids),
            "batch_size": int(batch_size),
            "top_k": int(top_k),
            "promote_base_relevance": bool(promote_base_relevance),
            "mean_utility": float(np.mean(utility_values)) if utility_values else 0.0,
            "max_utility": float(np.max(utility_values)) if utility_values else 0.0,
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a_norm <= 1e-8 or b_norm <= 1e-8:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    @staticmethod
    def _pool_vector(vector: Any) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim == 1:
            return array
        if array.ndim == 2:
            return array.mean(axis=0)
        return array.reshape(-1)

    def _decode_roq_vector(self, payload: Dict[str, Any]) -> Optional[np.ndarray]:
        if not self.roq_bits or self.hnsw is None or getattr(self.hnsw, "quantizer", None) is None:
            return None
        if "roq_codes" not in payload:
            return None
        try:
            codes = np.asarray(payload["roq_codes"], dtype=np.uint8)
            scale = np.asarray(payload["roq_scale"], dtype=np.float32)
            offset = np.asarray(payload["roq_offset"], dtype=np.float32)
            if codes.ndim == 1:
                codes = codes.reshape(1, -1)
            if scale.ndim == 0:
                scale = scale.reshape(1, 1)
            elif scale.ndim == 1:
                scale = scale.reshape(1, -1) if codes.shape[0] == 1 and scale.size > 1 else scale.reshape(-1, 1)
            if offset.ndim == 0:
                offset = offset.reshape(1, 1)
            elif offset.ndim == 1:
                offset = offset.reshape(1, -1) if codes.shape[0] == 1 and offset.size > 1 else offset.reshape(-1, 1)
            decoded = self.hnsw.quantizer.decode(codes, scale, offset)
            if hasattr(decoded, "cpu"):
                decoded = decoded.cpu().numpy()
            return np.asarray(decoded, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive decode path
            logger.warning("Failed to decode RoQ payload for solver refinement: %s", exc)
            return None

    @staticmethod
    def _estimate_token_count(payload: Dict[str, Any], text: str) -> int:
        if isinstance(payload.get("token_count"), (int, float)):
            return max(1, int(payload["token_count"]))
        token_count = len(re.findall(r"\w+", text))
        return max(16, min(512, int(token_count * 1.35) + 4))

    @staticmethod
    def _text_overlap_score(query_text: str, text: str) -> float:
        query_terms = set(re.findall(r"\w+", (query_text or "").lower()))
        text_terms = set(re.findall(r"\w+", (text or "").lower()))
        if not query_terms or not text_terms:
            return 0.0
        return len(query_terms & text_terms) / float(len(query_terms))

    @staticmethod
    def _normalized_text(text: str) -> str:
        return " ".join(re.findall(r"\w+", (text or "").lower()))

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"\w+", (text or "").lower()))

    @staticmethod
    def _float_list(values: Any) -> list[float]:
        if not isinstance(values, list):
            return []
        result: list[float] = []
        for value in values:
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    @classmethod
    def _ontology_feature_scores(
        cls,
        payload: Dict[str, Any],
        *,
        query_text: str = "",
        query_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        ontology_terms = [str(term).strip() for term in (payload.get("ontology_terms") or []) if str(term).strip()]
        ontology_labels = {
            str(label).strip().lower() for label in (payload.get("ontology_labels") or []) if str(label).strip()
        }
        ontology_confidences = cls._float_list(payload.get("ontology_confidences"))
        ontology_evidence = cls._float_list(payload.get("ontology_evidence_counts"))
        match_count = max(0, int(payload.get("ontology_match_count", len(ontology_terms)) or 0))
        confidence = cls._clamp01(
            float(payload.get("ontology_confidence"))
            if isinstance(payload.get("ontology_confidence"), (int, float))
            else (sum(ontology_confidences) / len(ontology_confidences) if ontology_confidences else 0.0)
        )
        concept_density = cls._clamp01(
            float(payload.get("ontology_concept_density"))
            if isinstance(payload.get("ontology_concept_density"), (int, float))
            else min(1.0, match_count / 4.0)
        )
        relation_density = cls._clamp01(
            float(payload.get("ontology_relation_density"))
            if isinstance(payload.get("ontology_relation_density"), (int, float))
            else min(1.0, (sum(ontology_evidence) / float(max(match_count * 8, 1))) if match_count else 0.0)
        )

        normalized_terms = {cls._normalized_text(term) for term in ontology_terms if term}
        ontology_tokens = set().union(*(cls._token_set(term) for term in ontology_terms)) if ontology_terms else set()
        query_terms = cls._token_set(query_text)
        exact_query_match = 1.0 if cls._normalized_text(query_text) in normalized_terms and query_text else 0.0
        entity_coverage = (
            len(query_terms & ontology_tokens) / float(len(query_terms)) if query_terms and ontology_tokens else 0.0
        )

        extra_query_terms: list[str] = []
        query_label = ""
        if query_payload:
            extra_query_terms = [
                str(term).strip() for term in (query_payload.get("ontology_terms") or []) if str(term).strip()
            ]
            raw_label = query_payload.get("label") or query_payload.get("query_label")
            if isinstance(raw_label, str):
                query_label = raw_label.strip().lower()
        if extra_query_terms and ontology_terms:
            extra_term_matches = {
                cls._normalized_text(term)
                for term in extra_query_terms
                if cls._normalized_text(term) in normalized_terms
            }
            if extra_term_matches:
                exact_query_match = 1.0
        type_match = 1.0 if query_label and query_label in ontology_labels else 0.0
        query_match = cls._clamp01(
            0.55 * max(entity_coverage, exact_query_match)
            + 0.20 * type_match
            + 0.15 * confidence
            + 0.10 * relation_density
        )
        return {
            "match_count": float(match_count),
            "confidence": confidence,
            "concept_density": concept_density,
            "relation_density": relation_density,
            "entity_coverage": cls._clamp01(max(entity_coverage, exact_query_match)),
            "type_match": cls._clamp01(type_match),
            "query_match": query_match,
        }

    @classmethod
    def _density_score(cls, payload: Dict[str, Any], text: str, overlap: float) -> float:
        if isinstance(payload.get("fact_density"), (int, float)):
            return cls._clamp01(float(payload["fact_density"]))
        tokens = re.findall(r"\w+", text)
        if not tokens:
            return 0.3 + (0.2 * overlap)
        unique_ratio = len(set(token.lower() for token in tokens)) / float(len(tokens))
        numeric_ratio = sum(any(ch.isdigit() for ch in token) for token in tokens) / float(len(tokens))
        return cls._clamp01(0.25 + (0.35 * unique_ratio) + (0.2 * numeric_ratio) + (0.2 * overlap))

    @classmethod
    def _recency_score(cls, payload: Dict[str, Any]) -> float:
        for key in ("recency_score", "freshness_score"):
            if isinstance(payload.get(key), (int, float)):
                return cls._clamp01(float(payload[key]))
        for key in ("timestamp", "updated_at", "created_at", "date", "document_date"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                timestamp = float(value)
                if timestamp > 1e11:
                    timestamp /= 1000.0
                age_days = max(0.0, (datetime.now(timezone.utc).timestamp() - timestamp) / 86400.0)
                return cls._clamp01(1.0 / (1.0 + age_days / 30.0))
            if isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
                age_days = max(
                    0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 86400.0
                )
                return cls._clamp01(1.0 / (1.0 + age_days / 30.0))
        return 0.5

    @staticmethod
    def _infer_role(payload: Dict[str, Any], text: str) -> str:
        role = payload.get("rhetorical_role")
        if isinstance(role, str) and role.strip():
            return role.strip().lower()
        lowered = (text or "").lower()
        if any(token in lowered for token in ("table", "row", "column")):
            return "data_table"
        if any(token in lowered for token in ("step", "procedure", "workflow")):
            return "procedure"
        if any(token in lowered for token in ("risk", "warning", "hazard")):
            return "risk"
        if any(token in lowered for token in ("must", "shall", "constraint")):
            return "constraint"
        if any(token in lowered for token in ("because", "evidence", "study", "data")):
            return "evidence"
        if any(token in lowered for token in ("means", "defined", "definition")):
            return "definition"
        if any(token in lowered for token in ("for example", "example", "e.g.")):
            return "example"
        if any(token in lowered for token in ("therefore", "summary", "conclusion")):
            return "conclusion"
        return "unknown"

    def _cluster_assignments(self, vectors: List[np.ndarray], threshold: float = 0.92) -> List[int]:
        centroids: List[np.ndarray] = []
        assignments: List[int] = []
        for vector in vectors:
            best_idx = -1
            best_score = -1.0
            for idx, centroid in enumerate(centroids):
                score = self._cosine_similarity(vector, centroid)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx >= 0 and best_score >= threshold:
                centroids[best_idx] = (centroids[best_idx] + vector) / 2.0
                assignments.append(best_idx)
            else:
                centroids.append(vector.copy())
                assignments.append(len(centroids) - 1)
        return assignments

    def _selected_order(
        self,
        query_vector: np.ndarray,
        solver_candidates: List[Dict[str, Any]],
        selected_indices: List[int],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[int]:
        if len(selected_indices) <= 1:
            return list(selected_indices)

        query = self._pool_vector(query_vector)
        remaining = list(selected_indices)
        ordered: List[int] = []
        weights = weights or {
            "beta": 0.24,
            "gamma": 0.18,
            "delta": 0.12,
            "epsilon": 0.14,
            "lambda": 0.32,
        }

        while remaining:
            best_idx = remaining[0]
            best_score = float("-inf")
            for idx in remaining:
                candidate = solver_candidates[idx]
                vector = np.asarray(candidate["embedding"], dtype=np.float32)
                marginal = self._cosine_similarity(query, vector)
                marginal += float(weights["beta"]) * float(candidate["fact_density"])
                marginal += float(weights["gamma"]) * float(candidate["centrality_score"])
                marginal += float(weights["delta"]) * float(candidate["recency_score"])
                marginal += float(weights["epsilon"]) * float(candidate["auxiliary_score"])
                if ordered:
                    redundancy = max(
                        self._cosine_similarity(
                            vector,
                            np.asarray(solver_candidates[chosen]["embedding"], dtype=np.float32),
                        )
                        for chosen in ordered
                    )
                    marginal -= float(weights["lambda"]) * redundancy
                if marginal > best_score:
                    best_score = marginal
                    best_idx = idx
            ordered.append(best_idx)
            remaining.remove(best_idx)

        return ordered

    def _build_solver_candidates(
        self,
        query_vector: np.ndarray,
        candidate_ids: List[int],
        query_text: str = "",
        query_payload: Optional[Dict[str, Any]] = None,
        retrieval_features: Optional[Dict[int, Dict[str, Any]]] = None,
        refine_options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        retrieved_data = self.hnsw.retrieve(candidate_ids)
        valid_items: List[Dict[str, Any]] = []
        pooled_vectors: List[np.ndarray] = []

        for item in retrieved_data:
            if item is None:
                continue
            payload = dict(item.get("payload", {}) or {})
            vector = item.get("vector")
            if vector is None:
                vector = self._decode_roq_vector(payload)
            if vector is None:
                continue
            pooled = self._pool_vector(vector)
            if pooled.size == 0:
                continue
            text = str(payload.get("text") or payload.get("content") or "")
            valid_items.append(
                {
                    "id": int(item.get("id")),
                    "payload": payload,
                    "text": text,
                    "pooled": pooled,
                }
            )
            pooled_vectors.append(pooled)

        if not valid_items:
            return []

        merged_retrieval_features: Dict[int, Dict[str, Any]] = {
            int(doc_id): dict(values) for doc_id, values in dict(retrieval_features or {}).items()
        }
        for item in valid_items:
            doc_id = int(item["id"])
            merged = merged_retrieval_features.setdefault(doc_id, {})
            payload_metadata = self._payload_rerank_metadata(item["payload"])
            for key, value in payload_metadata.items():
                merged.setdefault(key, value)
            if any(key in payload_metadata for key in ("rerank_score", "cross_encoder_score", "ce_score", "ltr_score")):
                merged["base_relevance"] = float(
                    payload_metadata.get("base_relevance", merged.get("base_relevance", 0.0) or 0.0)
                )
                merged.setdefault("relevance_source", payload_metadata.get("relevance_source", "payload_rerank"))

        resolved_refine_options = self._resolve_refine_options(
            query_payload=query_payload,
            refine_options=refine_options,
        )
        rerank_summary: Optional[Dict[str, Any]] = None
        nli_summary: Optional[Dict[str, Any]] = None
        confidence_gate = self._confidence_gate_summary(
            valid_items=valid_items,
            retrieval_features=merged_retrieval_features,
            refine_options=resolved_refine_options,
        )
        if not confidence_gate.get("applied"):
            if resolved_refine_options.get("use_cross_encoder"):
                rerank_summary = self._apply_cross_encoder_scores(
                    valid_items=valid_items,
                    query_text=query_text,
                    retrieval_features=merged_retrieval_features,
                    model_name=str(resolved_refine_options["cross_encoder_model"]),
                    batch_size=int(resolved_refine_options["cross_encoder_batch_size"]),
                    top_k=int(resolved_refine_options["cross_encoder_top_k"]),
                )
            if resolved_refine_options.get("use_nli"):
                nli_summary = self._apply_nli_scores(
                    valid_items=valid_items,
                    query_text=query_text,
                    retrieval_features=merged_retrieval_features,
                    model_name=str(resolved_refine_options["nli_model"]),
                    batch_size=int(resolved_refine_options["nli_batch_size"]),
                    top_k=int(resolved_refine_options["nli_top_k"]),
                    promote_base_relevance=bool(resolved_refine_options.get("nli_promote_base_relevance", False)),
                )
        else:
            if resolved_refine_options.get("use_cross_encoder"):
                rerank_summary = {
                    "enabled": True,
                    "applied": False,
                    "reason": "confidence_gate_skip",
                    "model": str(resolved_refine_options["cross_encoder_model"]),
                    "reranked_count": 0,
                }
            if resolved_refine_options.get("use_nli"):
                nli_summary = {
                    "enabled": True,
                    "applied": False,
                    "reason": "confidence_gate_skip",
                    "model": str(resolved_refine_options["nli_model"]),
                    "scored_count": 0,
                }
        if hasattr(self, "_last_refine_context"):
            self._last_refine_context = {
                **dict(getattr(self, "_last_refine_context") or {}),
                "rerank": rerank_summary,
                "nli": nli_summary,
                "confidence_gate": confidence_gate,
            }

        centroid = np.mean(np.stack(pooled_vectors, axis=0), axis=0)
        dynamic_clusters = self._cluster_assignments(pooled_vectors)

        query = self._pool_vector(query_vector)
        solver_candidates: List[Dict[str, Any]] = []
        for idx, item in enumerate(valid_items):
            payload = item["payload"]
            text = item["text"]
            vector = item["pooled"]
            query_similarity = self._cosine_similarity(query, vector)
            centroid_similarity = self._cosine_similarity(vector, centroid)
            overlap = self._text_overlap_score(query_text, text)
            ontology_features = self._ontology_feature_scores(
                payload,
                query_text=query_text,
                query_payload=query_payload,
            )
            max_peer_similarity = max(
                (
                    self._cosine_similarity(vector, other)
                    for peer_idx, other in enumerate(pooled_vectors)
                    if peer_idx != idx
                ),
                default=0.0,
            )
            uniqueness = self._clamp01(
                0.5 * (1.0 - max(0.0, max_peer_similarity)) + 0.5 * (1.0 - max(0.0, centroid_similarity))
            )
            fact_density = self._clamp01(
                0.70 * self._density_score(payload, text, overlap)
                + 0.20 * ontology_features["concept_density"]
                + 0.10 * ontology_features["relation_density"]
            )
            payload_centrality = payload.get("centrality_score")
            centrality = (
                self._clamp01(float(payload_centrality))
                if isinstance(payload_centrality, (int, float))
                else self._clamp01(
                    0.50 * (0.5 * (query_similarity + 1.0) * 0.5 + 0.5 * (centroid_similarity + 1.0) * 0.5)
                    + 0.25 * ontology_features["confidence"]
                    + 0.25 * ontology_features["query_match"]
                )
            )
            recency = self._recency_score(payload)
            payload_auxiliary = payload.get("auxiliary_score")
            auxiliary = self._clamp01(
                (
                    0.70 * float(payload_auxiliary)
                    + 0.20 * ontology_features["query_match"]
                    + 0.10 * ontology_features["relation_density"]
                )
                if isinstance(payload_auxiliary, (int, float))
                else (
                    (0.20 * uniqueness)
                    + (0.15 * overlap)
                    + (0.15 * max(0.0, query_similarity))
                    + (0.20 * ontology_features["entity_coverage"])
                    + (0.15 * ontology_features["type_match"])
                    + (0.15 * ontology_features["confidence"])
                )
            )
            cluster_id = payload.get("cluster_id")
            if not isinstance(cluster_id, int):
                cluster_id = dynamic_clusters[idx]
            retrieval_metadata = dict(merged_retrieval_features.get(int(item["id"]), {}))

            solver_candidates.append(
                {
                    "chunk_id": str(item["id"]),
                    "content": text,
                    "embedding": vector.tolist(),
                    "token_count": self._estimate_token_count(payload, text),
                    "fact_density": fact_density,
                    "centrality_score": centrality,
                    "recency_score": recency,
                    "auxiliary_score": auxiliary,
                    "rhetorical_role": self._infer_role(payload, text),
                    "cluster_id": int(cluster_id) if isinstance(cluster_id, int) else -1,
                    "ontology_entity_coverage": ontology_features["entity_coverage"],
                    "ontology_type_match": ontology_features["type_match"],
                    "ontology_query_match": ontology_features["query_match"],
                    "ontology_concept_density": ontology_features["concept_density"],
                    "ontology_relation_density": ontology_features["relation_density"],
                    "ontology_confidence": ontology_features["confidence"],
                    **retrieval_metadata,
                }
            )

        return solver_candidates

    def refine(
        self,
        query_vector: np.ndarray,
        candidate_ids: List[int],
        query_text: str = "",
        query_payload: Optional[Dict[str, Any]] = None,
        solver_config: Optional[Any] = None,
        constraints: Optional[Any] = None,
        optimizer_policy: Optional[Any] = None,
        refine_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._last_refine_context = {}
        if not self.solver:
            fallback_ids = candidate_ids[: min(10, len(candidate_ids))]
            return {
                "solver_output": {
                    "selected_indices": list(range(len(fallback_ids))),
                    "objective_score": None,
                    "num_selected": len(fallback_ids),
                    "solve_time_ms": 0.0,
                    "constraints_satisfied": True,
                    "constraint_violations": [],
                    "fallback": True,
                },
                "selected_ids": [str(item_id) for item_id in fallback_ids],
                "selected_internal_ids": [int(item_id) for item_id in fallback_ids],
                "backend_kind": "fallback",
            }
        retrieval_features: Dict[int, Dict[str, Any]] = {}
        if self._last_search_context:
            dense_meta = self._last_search_context.get("dense", {})
            sparse_meta = self._last_search_context.get("sparse", {})
            rrf_meta = self._last_search_context.get("rrf", {})
            for candidate_id in candidate_ids:
                dense_entry = dict(dense_meta.get(int(candidate_id), {}))
                sparse_entry = dict(sparse_meta.get(int(candidate_id), {}))
                combined = {
                    **dense_entry,
                    **sparse_entry,
                }
                if int(candidate_id) in rrf_meta:
                    combined["rrf_score"] = float(rrf_meta[int(candidate_id)])
                    combined["base_relevance"] = float(rrf_meta[int(candidate_id)])
                elif "dense_score" in combined:
                    combined["base_relevance"] = float(combined["dense_score"])
                elif "sparse_score" in combined:
                    combined["base_relevance"] = float(combined["sparse_score"])
                retrieval_features[int(candidate_id)] = combined
        resolved_refine_options = self._resolve_refine_options(
            query_payload=query_payload,
            refine_options=refine_options,
        )
        effective_optimizer_policy = optimizer_policy
        advanced_query_utility = self._has_advanced_query_utility(query_payload)
        if effective_optimizer_policy is None:
            if resolved_refine_options.get("use_nli") or advanced_query_utility:
                effective_optimizer_policy = "frontier_v1"
            elif resolved_refine_options.get("use_cross_encoder"):
                effective_optimizer_policy = "post_rerank_v1"
        solver_candidates = self._build_solver_candidates(
            query_vector,
            candidate_ids,
            query_text=query_text,
            query_payload=query_payload,
            retrieval_features=retrieval_features,
            refine_options=resolved_refine_options,
        )
        if not solver_candidates:
            return {
                "solver_output": {
                    "selected_indices": [],
                    "objective_score": 0.0,
                    "num_selected": 0,
                    "solve_time_ms": 0.0,
                    "constraints_satisfied": True,
                    "constraint_violations": [],
                },
                "selected_ids": [],
                "selected_internal_ids": [],
                "backend_kind": "cpu_reference",
            }
        if constraints is None:
            token_budget = max(
                256,
                min(4096, int(sum(candidate["token_count"] for candidate in solver_candidates) * 0.45)),
            )
            constraints_payload = {
                "max_tokens": token_budget,
                "max_chunks": min(10, len(solver_candidates)),
                "max_per_cluster": 2,
            }
        else:
            if isinstance(constraints, dict):
                constraints_payload = dict(constraints)
            else:
                constraints_payload = {
                    "max_tokens": int(getattr(constraints, "max_tokens")),
                    "min_tokens": int(getattr(constraints, "min_tokens", 0)),
                    "min_chunks": int(getattr(constraints, "min_chunks", 1)),
                    "max_chunks": int(getattr(constraints, "max_chunks")),
                    "max_per_cluster": int(getattr(constraints, "max_per_cluster", 2)),
                    "must_include_roles": list(getattr(constraints, "must_include_roles", [])),
                    "excluded_chunks": list(getattr(constraints, "excluded_chunks", [])),
                    "required_chunks": list(getattr(constraints, "required_chunks", [])),
                }

        policy_name = (
            str(effective_optimizer_policy.get("name", "baseline_v1"))
            if isinstance(effective_optimizer_policy, dict)
            else str(effective_optimizer_policy or "baseline_v1")
        )
        fallback_ranked_candidates, fallback_summary = self._fallback_candidate_order(solver_candidates)
        gate_summary = self._production_solver_gate(
            query_text=query_text,
            query_payload=query_payload,
            solver_candidates=solver_candidates,
            constraints_payload=constraints_payload,
            refine_options=resolved_refine_options,
            fallback_summary=fallback_summary,
        )
        if hasattr(self, "_last_refine_context"):
            self._last_refine_context = {
                **dict(getattr(self, "_last_refine_context") or {}),
                "solver_gate": gate_summary,
            }

        candidate_by_id = {candidate["chunk_id"]: candidate for candidate in solver_candidates}

        def _mean_feature(items: List[Dict[str, Any]], key: str) -> float:
            if not items:
                return 0.0
            return float(np.mean([float(item.get(key, 0.0)) for item in items]))

        def _feature_summary(
            selected_candidates: List[Dict[str, Any]],
            extra: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            summary: Dict[str, Any] = {
                "requested_optimizer_policy": policy_name,
                "refine_rerank": dict((self._last_refine_context or {}).get("rerank") or {}),
                "refine_nli": dict((self._last_refine_context or {}).get("nli") or {}),
                "refine_confidence_gate": dict((self._last_refine_context or {}).get("confidence_gate") or {}),
                "refine_solver_gate": dict(gate_summary),
                "fallback_ordering_signal": fallback_summary.get("signal"),
                "rerank_metadata_present": any(
                    any(
                        key in candidate
                        for key in (
                            "cross_encoder_score",
                            "ce_score",
                            "ltr_score",
                            "rerank_score",
                            "utility_score",
                            "nli_utility_score",
                            "nli_entailment",
                            "nli_contradiction",
                        )
                    )
                    for candidate in solver_candidates
                ),
                "nli_metadata_present": any(
                    any(
                        key in candidate
                        for key in ("utility_score", "nli_utility_score", "nli_entailment", "nli_contradiction")
                    )
                    for candidate in solver_candidates
                ),
                "candidate_count": len(solver_candidates),
                "query_text_used": bool(query_text.strip()),
                "query_label": (
                    str(query_payload.get("label") or query_payload.get("query_label"))
                    if isinstance(query_payload, dict)
                    else None
                ),
                "candidate_avg_base_relevance": _mean_feature(solver_candidates, "base_relevance"),
                "selected_avg_base_relevance": _mean_feature(selected_candidates, "base_relevance"),
                "candidate_avg_utility_score": _mean_feature(solver_candidates, "utility_score"),
                "selected_avg_utility_score": _mean_feature(selected_candidates, "utility_score"),
                "candidate_avg_nli_entailment": _mean_feature(solver_candidates, "nli_entailment"),
                "selected_avg_nli_entailment": _mean_feature(selected_candidates, "nli_entailment"),
                "candidate_avg_nli_contradiction": _mean_feature(solver_candidates, "nli_contradiction"),
                "selected_avg_nli_contradiction": _mean_feature(selected_candidates, "nli_contradiction"),
                "candidate_avg_ontology_query_match": _mean_feature(solver_candidates, "ontology_query_match"),
                "selected_avg_ontology_query_match": _mean_feature(selected_candidates, "ontology_query_match"),
                "candidate_avg_ontology_entity_coverage": _mean_feature(solver_candidates, "ontology_entity_coverage"),
                "selected_avg_ontology_entity_coverage": _mean_feature(selected_candidates, "ontology_entity_coverage"),
                "candidate_avg_ontology_type_match": _mean_feature(solver_candidates, "ontology_type_match"),
                "selected_avg_ontology_type_match": _mean_feature(selected_candidates, "ontology_type_match"),
            }
            if isinstance(extra, dict):
                summary.update(extra)
            return summary

        if gate_summary.get("applied") and not gate_summary.get("run_solver"):
            selected_cap = int(gate_summary.get("selected_cap", min(10, len(fallback_ranked_candidates))))
            selected_candidates = fallback_ranked_candidates[:selected_cap]
            selected_ids = [candidate["chunk_id"] for candidate in selected_candidates]
            total_tokens = int(sum(int(candidate.get("token_count", 0) or 0) for candidate in selected_candidates))
            return {
                "solver_output": {
                    "selected_indices": list(range(len(selected_candidates))),
                    "objective_score": 0.0,
                    "num_selected": len(selected_candidates),
                    "total_tokens": total_tokens,
                    "solve_time_ms": 0.0,
                    "constraints_satisfied": True,
                    "constraint_violations": [],
                    "fallback": True,
                    "skipped_by_gate": True,
                    "gate": gate_summary,
                },
                "selected_ids": selected_ids,
                "selected_internal_ids": [int(item_id) for item_id in selected_ids if str(item_id).isdigit()],
                "selected_vectors": {
                    int(item_id): candidate_by_id[item_id]["embedding"]
                    for item_id in selected_ids
                    if item_id in candidate_by_id and str(item_id).isdigit()
                },
                "backend_kind": "rrf_gate",
                "execution_mode": "gate_skip",
                "solver_backend_kind": None,
                "feature_summary": _feature_summary(
                    selected_candidates,
                    {
                        "skipped_by_gate": True,
                        "selected_count": len(selected_candidates),
                        "selected_total_tokens": total_tokens,
                    },
                ),
            }

        effective_solver_config = solver_config or getattr(self.solver, "config", None)
        solver_config_payload: Dict[str, Any] = {}
        if effective_solver_config is not None:
            if isinstance(effective_solver_config, dict):
                solver_config_payload = dict(effective_solver_config)
            else:
                for field in (
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "epsilon",
                    "mu",
                    "support_secondary_discount",
                    "support_quorum_bonus",
                    "support_quorum_threshold",
                    "support_quorum_cap",
                    "iterations",
                    "tabu_tenure",
                    "early_stopping_patience",
                    "use_gpu",
                    "random_seed",
                    "enable_gpu_move_evaluation",
                    "enable_path_relinking",
                    "enable_destroy_repair",
                    "enable_reactive_tenure",
                    "enable_exact_window",
                    "exact_window_size",
                    "exact_window_time_ms",
                ):
                    value = getattr(effective_solver_config, field, None)
                    if value is not None:
                        solver_config_payload[field] = value
                lambda_value = getattr(effective_solver_config, "lambda_", None)
                if lambda_value is not None:
                    solver_config_payload["lambda"] = lambda_value
        explicit_lambda_override = self._is_lambda_override_explicit(
            solver_config,
            default_solver_config=getattr(self.solver, "config", None),
        )
        solver_config_payload, production_profile = self._apply_production_solver_profile(
            solver_config_payload=solver_config_payload,
            candidate_count=len(solver_candidates),
            refine_options=resolved_refine_options,
            explicit_lambda_override=explicit_lambda_override,
        )
        if effective_optimizer_policy is not None:
            solver_config_payload["optimizer_policy"] = effective_optimizer_policy
        rerank_applied = bool(((self._last_refine_context or {}).get("rerank") or {}).get("applied"))
        nli_applied = bool(((self._last_refine_context or {}).get("nli") or {}).get("applied"))
        logger.info(
            "Hybrid refine candidates=%d policy=%s rerank=%s nli=%s",
            len(solver_candidates),
            policy_name,
            rerank_applied,
            nli_applied,
        )

        request_candidates = []
        for candidate in solver_candidates:
            payload = {
                "text": candidate["content"],
                "token_count": candidate["token_count"],
                "fact_density": candidate["fact_density"],
                "centrality_score": candidate["centrality_score"],
                "recency_score": candidate["recency_score"],
                "auxiliary_score": candidate["auxiliary_score"],
                "rhetorical_role": candidate["rhetorical_role"],
                "cluster_id": None if int(candidate["cluster_id"]) < 0 else int(candidate["cluster_id"]),
            }
            for key, value in candidate.items():
                if key in {
                    "chunk_id",
                    "content",
                    "embedding",
                    "token_count",
                    "fact_density",
                    "centrality_score",
                    "recency_score",
                    "auxiliary_score",
                    "rhetorical_role",
                    "cluster_id",
                }:
                    continue
                payload[key] = value
            request_candidates.append(
                {
                    "id": candidate["chunk_id"],
                    "vector": candidate["embedding"],
                    "payload": payload,
                }
            )

        if self._optimizer_pipeline is None:
            self._optimizer_pipeline = GpuFulfilmentPipeline()
        result = self._optimizer_pipeline.optimize_in_process(
            query_vectors=query_vector,
            candidate_items=request_candidates,
            query_text=query_text,
            constraints=constraints_payload,
            solver_config=solver_config_payload,
            metadata={
                "query_payload": dict(query_payload or {}),
                "optimizer_policy": effective_optimizer_policy,
            },
        )
        selected_ids = [str(item_id) for item_id in result.get("selected_ids", [])]
        selected_candidates = [candidate_by_id[item_id] for item_id in selected_ids if item_id in candidate_by_id]
        return {
            "solver_output": result.get("solver_output", {}),
            "selected_ids": selected_ids,
            "selected_internal_ids": [int(item_id) for item_id in selected_ids if str(item_id).isdigit()],
            "selected_vectors": {
                int(item_id): candidate_by_id[item_id]["embedding"]
                for item_id in selected_ids
                if item_id in candidate_by_id and str(item_id).isdigit()
            },
            "backend_kind": result.get("solver_backend_kind") or result.get("backend_kind", "cpu_reference"),
            "execution_mode": result.get("backend_kind"),
            "solver_backend_kind": result.get("solver_backend_kind"),
            "feature_summary": _feature_summary(
                selected_candidates,
                {
                    **result.get("feature_summary", {}),
                    "production_solver_profile": production_profile,
                },
            ),
        }

    def close(self) -> None:
        """Release native HNSW resources for in-process restarts."""
        if self.hnsw is not None and hasattr(self.hnsw, "close"):
            self.hnsw.close()
