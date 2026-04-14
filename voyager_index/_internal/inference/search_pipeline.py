
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .index_core.hybrid_manager import HybridSearchManager

logger = logging.getLogger(__name__)

class SearchPipeline:
    """
    Vector-first local hybrid retrieval pipeline.

    Public OSS behavior focuses on:
    1. Dense retrieval (shard-based late-interaction by default, or HNSW).
    2. Canonical `bm25s` sparse retrieval via `HybridSearchManager`,
       with pure-Python BM25 fallback when bm25s is unavailable.
    3. Optional local/native refinement when a compatible solver is available.

    This pipeline does not embed raw text into dense late-interaction queries.
    String input is handled as sparse-only retrieval.
    Remote compute-side productization is intentionally out of scope for this
    class.
    """

    def __init__(
        self,
        shard_path: str,
        dim: int = 128,
        use_roq: bool = True,
        roq_bits: int = 4,
        on_disk: bool = True,
        dense_engine: str = "shard",
        dense_engine_config: Optional[Any] = None,
    ):
        """
        Initialize the search pipeline.

        Args:
            dense_engine: Dense backend to use (``"shard"`` or ``"hnsw"``).
                Defaults to ``"shard"`` for late-interaction retrieval.
            dense_engine_config: Optional config object forwarded to the
                dense backend constructor.
        """
        self.config = {
            "dim": dim,
            "use_roq": use_roq,
            "roq_bits": roq_bits,
            "on_disk": on_disk,
            "dense_engine": dense_engine,
        }

        self.manager = HybridSearchManager(
            shard_path=Path(shard_path),
            dim=dim,
            on_disk=on_disk,
            roq_bits=roq_bits if use_roq else None,
            dense_engine=dense_engine,
            dense_engine_config=dense_engine_config,
        )

        logger.info(f"SearchPipeline initialized at {shard_path} (dense={dense_engine}, RoQ={use_roq})")

    def index(
        self,
        corpus: List[str],
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[int],
        payloads: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Ingest documents into the pipeline.
        Supports both Single-Vector and Multi-Vector (ColBERT) inputs.
        """
        if isinstance(vectors, list):
            logger.info(f"Indexing {len(corpus)} documents (Multi-Vector)...")
            self.manager.index_multivector(corpus, vectors, ids, payloads)
        else:
            # Standard single vector
            logger.info(f"Indexing {len(corpus)} documents (Single-Vector)...")
            self.manager.index(corpus, vectors, ids, payloads)

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k_retrieval: int = 100,
        enable_refinement: bool = False,
        query_text: str = "",
        query_payload: Optional[Dict[str, Any]] = None,
        solver_config: Optional[Dict[str, Any]] = None,
        optimizer_policy: Optional[Any] = None,
        refine_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute retrieval with optional solver refinement.

        Args:
            query: Query vector for dense retrieval, or query text for sparse-only retrieval.
            top_k_retrieval: Number of candidates from HNSW (Map).
            enable_refinement: Whether to run optional local/native solver
                refinement when available.
            query_text: Optional sparse text / rerank text for dense-query refinement.
            query_payload: Optional query-side metadata passed into refinement.
            solver_config: Optional solver config overrides for the optimizer path.
            optimizer_policy: Optional optimizer policy preset or override dict.
            refine_options: Optional refinement options, including cross-encoder settings.
        """
        if isinstance(query, str):
            search_output = self.manager.search(
                query_text=query,
                query_vector=None,
                k=top_k_retrieval,
            )
            retrieval_ids = search_output.get('union_ids', [])
            ordered_ids = retrieval_ids
            return {
                "retrieval": search_output,
                "retrieval_count": len(retrieval_ids),
                "solver_output": None,
                "selected_ids": ordered_ids[: min(10, len(ordered_ids))],
            }

        query_vector = np.asarray(query, dtype=np.float32)
        if query_vector.ndim > 1:
            raise ValueError(
                "SearchPipeline.search expects a single dense query vector. "
                "Late-interaction multi-vector queries should use ColbertIndex directly."
            )

        search_output = self.manager.search(
            query_text=query_text,
            query_vector=query_vector,
            k=top_k_retrieval
        )

        retrieval_ids = search_output.get('union_ids', [])
        dense_ids = [doc_id for doc_id, _ in search_output.get("dense", [])]
        sparse_ids = [doc_id for doc_id, _ in search_output.get("sparse", [])]
        ordered_ids = dense_ids + [doc_id for doc_id in sparse_ids if doc_id not in dense_ids]

        # Local refinement is optional and remains distinct from any future
        # remote compute productization.
        if not enable_refinement or not getattr(self.manager, "solver_available", False):
            return {
                "retrieval": search_output,
                "retrieval_count": len(retrieval_ids),
                "solver_output": None,
                "selected_ids": ordered_ids[: min(10, len(ordered_ids))],
            }

        refine_results = self.manager.refine(
            query_vector=query_vector,
            query_text=query_text,
            query_payload=query_payload,
            candidate_ids=retrieval_ids,
            solver_config=solver_config,
            optimizer_policy=optimizer_policy,
            refine_options=refine_options,
        )

        return {
            "retrieval": search_output,
            "retrieval_count": len(retrieval_ids),
            "solver_output": refine_results.get('solver_output'),
            "selected_ids": refine_results.get('selected_ids'),
            "solver_backend": refine_results.get("backend_kind"),
        }
