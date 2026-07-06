"""
ColBERT Search Engine

Wrapper for the enterprise ColBERT index that implements the BaseSearchEngine interface.
This allows ColBERT to be used in the multi-engine search framework.

Features three modes based on corpus size:
1. REAL-TIME (<1K docs): Pure Triton (all in VRAM)
2. HIGH QUALITY (1K-50K docs): Triton + mmap
3. BALANCED (>50K docs): Fast-Plaid + Triton reranking

The BALANCED mode uses the Rust-based fast-plaid library for scalable
multi-vector search with product quantization.

Author: Latence Team
License: Apache-2.0
"""

import logging
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import torch

from ..config import IndexConfig
from ..engines.base import DenseSearchEngine, SearchResult
from ..index_core.index import ColbertIndex

logger = logging.getLogger(__name__)

# Try to import fast-plaid
_FAST_PLAID_AVAILABLE = False
try:
    from fast_plaid import FastPlaid
    _FAST_PLAID_AVAILABLE = True
    logger.info("fast-plaid library available for BALANCED mode")
except ImportError:
    logger.debug("fast-plaid not installed, BALANCED mode will use internal implementation")


class ColBERTEngine(DenseSearchEngine):
    """
    ColBERT dense retrieval engine.

    Wraps the enterprise ColBERT index with the standard search engine interface.
    Supports automatic scaling (Triton, Triton+mmap, PLAID+Triton) based on corpus size.

    Features:
    - Dense vector search with MaxSim scoring
    - Automatic strategy selection (small/medium/large)
    - Full CRUD operations
    - Collection management
    - Memory-efficient streaming

    Attributes:
        index: ColBERT index instance.
        config: Index configuration.
        device: Compute device ('cuda' or 'cpu').

    Example:
        >>> engine = ColBERTEngine(
        ...     index_path="/data/colbert_index",
        ...     config=IndexConfig()
        ... )
        >>>
        >>> # Index documents
        >>> engine.index(documents, doc_ids=list(range(len(documents))))
        >>>
        >>> # Search
        >>> results = engine.search("machine learning", top_k=10)
        >>> for result in results:
        ...     print(f"{result.doc_id}: {result.score:.4f}")
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        config: Optional[IndexConfig] = None,
        device: str = 'cuda',
        load_if_exists: bool = True,
        use_fast_plaid: bool = True,
    ):
        """
        Initialize ColBERT engine.

        Args:
            index_path: Path to store/load the index.
            config: Index configuration (uses defaults if None).
            device: Compute device ('cuda' or 'cpu').
            load_if_exists: Load existing index if found.
            use_fast_plaid: Use fast-plaid for large indexes (>50K docs).

        Raises:
            ValueError: If index_path is invalid.
        """
        super().__init__(engine_name='colbert')

        self.index_path = Path(index_path)
        self.config = config or IndexConfig(device=device)
        self.device = device
        self.use_fast_plaid = use_fast_plaid and _FAST_PLAID_AVAILABLE

        # Fast-plaid index (for BALANCED mode >50K docs)
        self._fast_plaid_index: Optional[Any] = None
        self._fast_plaid_active = False

        # Initialize or load index (use _index to avoid conflict with index() method)
        if load_if_exists and (self.index_path / "config.json").exists():
            logger.info(f"Loading existing ColBERT index from {index_path}")
            self._colbert_index = ColbertIndex.load(self.index_path, device=device)
            self._indexed = True

            # Check if fast-plaid index exists
            fast_plaid_path = self.index_path / "fast_plaid"
            if self.use_fast_plaid and fast_plaid_path.exists():
                self._load_fast_plaid_index(fast_plaid_path)
        else:
            logger.info(f"Creating new ColBERT index at {index_path}")
            self._colbert_index = ColbertIndex(
                self.index_path,
                self.config,
                create_if_missing=True
            )
            self._indexed = False

    def _load_fast_plaid_index(self, index_path: Path):
        """Load fast-plaid index if available."""
        if not _FAST_PLAID_AVAILABLE:
            return

        try:
            from fast_plaid import FastPlaid
            self._fast_plaid_index = FastPlaid.load(str(index_path), device=self.device)
            self._fast_plaid_active = True
            logger.info(f"Loaded fast-plaid index from {index_path}")
        except Exception as e:
            logger.warning(f"Failed to load fast-plaid index: {e}")
            self._fast_plaid_active = False

    def _build_fast_plaid_index(
        self,
        embeddings: torch.Tensor,
        nbits: int = 2,
        kmeans_niters: int = 10,
    ):
        """Build fast-plaid index for BALANCED mode."""
        if not _FAST_PLAID_AVAILABLE:
            logger.warning("fast-plaid not available, skipping BALANCED mode index")
            return

        try:
            from fast_plaid import FastPlaid

            fast_plaid_path = self.index_path / "fast_plaid"
            fast_plaid_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Building fast-plaid index for {embeddings.shape[0]} documents")

            # Build fast-plaid index
            self._fast_plaid_index = FastPlaid.create(
                embeddings=embeddings,
                index_path=str(fast_plaid_path),
                nbits=nbits,
                kmeans_niters=kmeans_niters,
                device=self.device,
            )
            self._fast_plaid_active = True

            logger.info(f"Fast-plaid index built at {fast_plaid_path}")

        except Exception as e:
            logger.error(f"Failed to build fast-plaid index: {e}")
            self._fast_plaid_active = False

    def index(
        self,
        documents: Union[List[str], torch.Tensor],
        doc_ids: Optional[List[Any]] = None,
        collection_name: str = "default",
        embeddings: Optional[torch.Tensor] = None,
        max_tokens: Optional[int] = None,
        build_fast_plaid: Optional[bool] = None,
        **kwargs
    ) -> None:
        """
        Index documents for search.

        Automatically selects indexing strategy based on corpus size:
        - <1K docs: Pure Triton (REAL-TIME mode)
        - 1K-50K docs: Triton + mmap (HIGH QUALITY mode)
        - >50K docs: Fast-Plaid + Triton (BALANCED mode)

        Args:
            documents: List of documents (texts or pre-computed embeddings).
            doc_ids: Optional document IDs (auto-generated if None).
            collection_name: Collection name for grouping.
            embeddings: Pre-computed embeddings (if documents are texts).
            max_tokens: Maximum token length (for streaming efficiency).
            build_fast_plaid: Force fast-plaid build (auto if None).
            **kwargs: Additional parameters.

        Raises:
            ValueError: If input format is invalid.
            RuntimeError: If indexing fails.

        Note:
            If documents are texts, you must provide embeddings or an embedding model.
        """
        logger.info(f"Indexing {len(documents)} documents to collection '{collection_name}'")

        # Determine embeddings
        if embeddings is not None:
            emb_tensor = embeddings
        elif isinstance(documents, torch.Tensor):
            emb_tensor = documents
        else:
            raise ValueError(
                "ColBERT requires pre-computed embeddings. "
                "Use embed_documents() first or pass embeddings parameter."
            )

        # Index with internal ColbertIndex
        self._index_embeddings(emb_tensor, collection_name, max_tokens)

        # Determine if we should build fast-plaid
        num_docs = emb_tensor.shape[0]
        balanced_threshold = getattr(self.config, 'balanced_threshold', 50000)

        should_build_fast_plaid = (
            build_fast_plaid if build_fast_plaid is not None
            else (self.use_fast_plaid and num_docs > balanced_threshold)
        )

        if should_build_fast_plaid:
            logger.info(f"Corpus size ({num_docs}) exceeds balanced threshold ({balanced_threshold}), building fast-plaid index")
            self._build_fast_plaid_index(emb_tensor)

        self._indexed = True
        strategy = "fast-plaid+triton" if self._fast_plaid_active else self.get_statistics().get('strategy', 'unknown')
        logger.info(f"Indexing complete. Strategy: {strategy}")

    def _index_embeddings(
        self,
        embeddings: torch.Tensor,
        collection_name: str,
        max_tokens: Optional[int]
    ) -> None:
        """
        Internal method to index embeddings.

        Args:
            embeddings: Document embeddings [num_docs, num_tokens, embed_dim].
            collection_name: Collection name.
            max_tokens: Maximum token length.
        """
        # Create batch generator
        batch_size = self.config.batch_size
        num_docs = embeddings.shape[0]

        def batch_gen() -> Generator[Tuple[torch.Tensor, None], None, None]:
            for i in range(0, num_docs, batch_size):
                end_idx = min(i + batch_size, num_docs)
                yield embeddings[i:end_idx], None

        # Build index
        self._colbert_index.build_from_batches(
            batch_gen(),
            collection_name=collection_name,
            max_tokens=max_tokens
        )

    def search(
        self,
        query: Union[str, torch.Tensor],
        top_k: int = 10,
        collection_name: Optional[str] = None,
        query_embedding: Optional[torch.Tensor] = None,
        use_rerank: bool = True,
        rerank_factor: int = 2,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        For BALANCED mode (fast-plaid active), performs two-stage search:
        1. Candidate retrieval with fast-plaid (approximate)
        2. Reranking with Triton MaxSim kernel (exact)

        Args:
            query: Query text or pre-computed embedding.
            top_k: Number of results to return.
            collection_name: Optional collection to search.
            query_embedding: Pre-computed query embedding.
            use_rerank: Rerank fast-plaid candidates with Triton (default True).
            rerank_factor: Retrieve rerank_factor * top_k candidates for reranking.
            **kwargs: Additional parameters.

        Returns:
            List of SearchResult objects, sorted by score (descending).

        Raises:
            RuntimeError: If index is not built.
            ValueError: If query format is invalid.

        Example:
            >>> results = engine.search("machine learning", top_k=10)
            >>> print(results[0].doc_id, results[0].score)
        """
        if not self._indexed:
            raise RuntimeError("Index must be built before searching. Call index() first.")

        self.validate_query(query)
        self.validate_top_k(top_k)

        # Get query embedding
        if query_embedding is not None:
            q_emb = query_embedding
        elif isinstance(query, torch.Tensor):
            q_emb = query
        else:
            raise ValueError(
                "ColBERT requires pre-computed query embedding. "
                "Use embed_query() first or pass query_embedding parameter."
            )

        # Ensure query has batch dimension
        if q_emb.dim() == 2:
            q_emb = q_emb.unsqueeze(0)

        # Use fast-plaid for BALANCED mode if active
        if self._fast_plaid_active and self._fast_plaid_index is not None:
            return self._search_fast_plaid(
                q_emb,
                top_k,
                collection_name,
                use_rerank=use_rerank,
                rerank_factor=rerank_factor,
            )

        # Standard ColBERT search
        scores, indices = self._colbert_index.search(
            queries=q_emb,
            top_k=top_k,
            collection_name=collection_name
        )

        # Convert to SearchResult format
        results = []
        for rank, (doc_id, score) in enumerate(zip(indices[0].cpu().numpy(), scores[0].cpu().numpy())):
            results.append(SearchResult(
                doc_id=int(doc_id),
                score=float(score),
                rank=rank + 1,
                source='colbert',
                metadata={'collection': collection_name}
            ))

        return results

    def _search_fast_plaid(
        self,
        query_embedding: torch.Tensor,
        top_k: int,
        collection_name: Optional[str],
        use_rerank: bool = True,
        rerank_factor: int = 2,
    ) -> List[SearchResult]:
        """
        Two-stage search with fast-plaid + Triton reranking.

        Args:
            query_embedding: Query embedding (1, S, D)
            top_k: Number of final results
            collection_name: Collection to search
            use_rerank: Whether to rerank with Triton
            rerank_factor: Candidate retrieval multiplier

        Returns:
            List of SearchResult objects
        """
        # Stage 1: Retrieve candidates with fast-plaid
        num_candidates = top_k * rerank_factor if use_rerank else top_k

        # Fast-plaid search
        results = self._fast_plaid_index.search(
            query=query_embedding.squeeze(0).cpu().numpy(),
            k=num_candidates,
        )

        candidate_ids = results['doc_ids']
        candidate_scores = results['scores']

        if not use_rerank or len(candidate_ids) == 0:
            # Return fast-plaid results directly
            return [
                SearchResult(
                    doc_id=int(doc_id),
                    score=float(score),
                    rank=rank + 1,
                    source='colbert-fastplaid',
                    metadata={'collection': collection_name, 'reranked': False}
                )
                for rank, (doc_id, score) in enumerate(zip(candidate_ids[:top_k], candidate_scores[:top_k]))
            ]

        # Stage 2: Rerank with Triton MaxSim kernel
        try:
            from ...kernels.maxsim import fast_colbert_scores

            # Get candidate embeddings
            candidate_embeddings = self._colbert_index.get_embeddings(candidate_ids)

            # Compute exact MaxSim scores
            rerank_scores = fast_colbert_scores(
                query_embedding.to(self.device),
                candidate_embeddings.to(self.device),
                use_quantization=True,
            )

            # Sort by reranked scores
            sorted_indices = torch.argsort(rerank_scores[0], descending=True)

            results = []
            for rank, idx in enumerate(sorted_indices[:top_k].cpu().numpy()):
                results.append(SearchResult(
                    doc_id=int(candidate_ids[idx]),
                    score=float(rerank_scores[0, idx].cpu()),
                    rank=rank + 1,
                    source='colbert-fastplaid-reranked',
                    metadata={'collection': collection_name, 'reranked': True}
                ))

            return results

        except Exception as e:
            logger.warning(f"Reranking failed, returning fast-plaid results: {e}")
            return [
                SearchResult(
                    doc_id=int(doc_id),
                    score=float(score),
                    rank=rank + 1,
                    source='colbert-fastplaid',
                    metadata={'collection': collection_name, 'reranked': False}
                )
                for rank, (doc_id, score) in enumerate(zip(candidate_ids[:top_k], candidate_scores[:top_k]))
            ]

    def embed_query(self, query: str, model: Any = None) -> torch.Tensor:
        """
        Convert query text to embedding.

        Args:
            query: Query text.
            model: Embedding model (required).

        Returns:
            Query embedding tensor.

        Raises:
            ValueError: If model is not provided.

        Note:
            You must provide your own embedding model (e.g., ColBERT model, VLLM, etc.)
        """
        if model is None:
            raise ValueError("Embedding model must be provided")

        # This is a placeholder - actual implementation depends on the model
        logger.warning("Using placeholder embed_query - provide actual model")
        return model.encode([query])[0]

    def embed_documents(self, documents: List[str], model: Any = None) -> torch.Tensor:
        """
        Convert documents to embeddings.

        Args:
            documents: List of document texts.
            model: Embedding model (required).

        Returns:
            Document embeddings tensor.

        Raises:
            ValueError: If model is not provided.

        Note:
            You must provide your own embedding model (e.g., ColBERT model, VLLM, etc.)
        """
        if model is None:
            raise ValueError("Embedding model must be provided")

        # This is a placeholder - actual implementation depends on the model
        logger.warning("Using placeholder embed_documents - provide actual model")
        return model.encode(documents)

    def add_documents(
        self,
        embeddings: torch.Tensor,
        collection_name: str = "default"
    ) -> None:
        """
        Add documents to existing index.

        Args:
            embeddings: Document embeddings to add.
            collection_name: Collection name.
        """
        self._colbert_index.add_documents(embeddings, collection_name)
        logger.info(f"Added {embeddings.shape[0]} documents to collection '{collection_name}'")

    def delete_documents(self, doc_ids: List[int]) -> None:
        """
        Delete documents from index.

        Args:
            doc_ids: Document IDs to delete.
        """
        self._colbert_index.delete_documents(doc_ids)
        logger.info(f"Deleted {len(doc_ids)} documents")

    def update_documents(
        self,
        doc_ids: List[int],
        embeddings: torch.Tensor
    ) -> None:
        """
        Update documents in index.

        Args:
            doc_ids: Document IDs to update.
            embeddings: New embeddings.
        """
        self._colbert_index.update_documents(doc_ids, embeddings)
        logger.info(f"Updated {len(doc_ids)} documents")

    def create_collection(self, name: str, doc_ids: List[int]) -> None:
        """
        Create a collection.

        Args:
            name: Collection name.
            doc_ids: Document IDs in collection.
        """
        self._colbert_index.create_collection(name, doc_ids)
        logger.info(f"Created collection '{name}' with {len(doc_ids)} documents")

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.

        Args:
            name: Collection name.
        """
        self._colbert_index.delete_collection(name)
        logger.info(f"Deleted collection '{name}'")

    def compact(self) -> None:
        """Compact index (remove deleted documents)."""
        self._colbert_index.compact()
        logger.info("Index compacted")

    def get_statistics(self) -> dict:
        """
        Get engine statistics.

        Returns:
            Dictionary with statistics (num_documents, strategy, storage, etc.).
        """
        stats = self._colbert_index.get_statistics()
        return {
            'engine': 'colbert',
            'num_documents': stats.num_documents,
            'num_collections': stats.num_collections,
            'strategy': stats.strategy,
            'storage_mb': stats.storage_size_mb,
            'avg_tokens_per_doc': stats.avg_tokens_per_doc,
            'embed_dim': stats.embed_dim
        }

    def cleanup(self) -> None:
        """Clean up engine resources."""
        # Index cleanup is handled by the index itself
        logger.info("ColBERT engine cleanup complete")

    def __repr__(self) -> str:
        stats = self.get_statistics() if self._indexed else {}
        status = f"indexed ({stats.get('num_documents', 0)} docs)" if self._indexed else "not indexed"
        return f"ColBERTEngine(path='{self.index_path}', status='{status}')"


__all__ = ['ColBERTEngine']

