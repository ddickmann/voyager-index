"""
Base Search Engine Interface

Defines the abstract interface that all search engines must implement.
This allows for unified handling of ColBERT, BM25, Neo4j, and future engines.

Author: ColBERT Team
License: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch


@dataclass
class SearchResult:
    """
    Unified search result format across all engines.

    Attributes:
        doc_id: Document identifier (int or string).
        score: Relevance score (higher is better).
        rank: Rank in the result list (1-based).
        source: Engine that produced this result ('colbert', 'bm25', 'neo4j').
        metadata: Optional additional information.

    Example:
        >>> result = SearchResult(
        ...     doc_id=42,
        ...     score=0.95,
        ...     rank=1,
        ...     source='colbert'
        ... )
    """
    doc_id: Any
    score: float
    rank: int
    source: str
    metadata: Optional[dict] = None

    def __repr__(self) -> str:
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, rank={self.rank}, source={self.source})"


class BaseSearchEngine(ABC):
    """
    Abstract base class for all search engines.

    All engines (ColBERT, BM25, Neo4j) must implement this interface.
    This ensures consistent behavior and easy integration into the fusion system.

    Methods to implement:
        - index(): Index documents
        - search(): Search for relevant documents
        - get_statistics(): Return engine statistics
        - cleanup(): Clean up resources

    Example:
        >>> class MyEngine(BaseSearchEngine):
        ...     def index(self, documents, **kwargs):
        ...         # Implementation
        ...         pass
        ...
        ...     def search(self, query, top_k=10):
        ...         # Implementation
        ...         return results
    """

    def __init__(self, engine_name: str):
        """
        Initialize base search engine.

        Args:
            engine_name: Name of the engine ('colbert', 'bm25', 'neo4j').
        """
        self.engine_name = engine_name
        self._indexed = False

    @abstractmethod
    def index(
        self,
        documents: List[Any],
        doc_ids: Optional[List[Any]] = None,
        **kwargs
    ) -> None:
        """
        Index documents for search.

        Args:
            documents: List of documents to index.
            doc_ids: Optional list of document IDs (auto-generated if None).
            **kwargs: Engine-specific parameters.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: Any,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Search query (text or embedding).
            top_k: Number of results to return.
            **kwargs: Engine-specific parameters.

        Returns:
            List of SearchResult objects, sorted by score (descending).

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> dict:
        """
        Get engine statistics.

        Returns:
            Dictionary with engine statistics (num_documents, strategy, etc.).

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up engine resources.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass

    def is_indexed(self) -> bool:
        """
        Check if engine has been indexed.

        Returns:
            True if indexed, False otherwise.
        """
        return self._indexed

    def validate_query(self, query: Any) -> None:
        """
        Validate query format (can be overridden by subclasses).

        Args:
            query: Query to validate.

        Raises:
            ValueError: If query is invalid.
        """
        if query is None:
            raise ValueError(f"{self.engine_name}: Query cannot be None")

    def validate_top_k(self, top_k: int) -> None:
        """
        Validate top_k parameter.

        Args:
            top_k: Number of results requested.

        Raises:
            ValueError: If top_k is invalid.
        """
        if top_k <= 0:
            raise ValueError(f"{self.engine_name}: top_k must be positive, got {top_k}")

    def __repr__(self) -> str:
        status = "indexed" if self._indexed else "not indexed"
        return f"{self.__class__.__name__}(name='{self.engine_name}', status='{status}')"


class DenseSearchEngine(BaseSearchEngine):
    """
    Base class for dense retrieval engines (ColBERT, etc.).

    These engines work with dense vector representations.
    """

    @abstractmethod
    def embed_query(self, query: str) -> torch.Tensor:
        """
        Convert query text to embedding.

        Args:
            query: Query text.

        Returns:
            Query embedding tensor.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> torch.Tensor:
        """
        Convert documents to embeddings.

        Args:
            documents: List of document texts.

        Returns:
            Document embeddings tensor.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass


class SparseSearchEngine(BaseSearchEngine):
    """
    Base class for sparse retrieval engines (BM25, etc.).

    These engines work with sparse representations (bag-of-words, TF-IDF, etc.).
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.

        Args:
            text: Input text.

        Returns:
            List of tokens.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass


class GraphSearchEngine(BaseSearchEngine):
    """
    Base class for graph-based search engines (Neo4j, etc.).

    These engines leverage graph structure for retrieval.
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to graph database.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection to graph database.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass


__all__ = [
    'BaseSearchEngine',
    'DenseSearchEngine',
    'SparseSearchEngine',
    'GraphSearchEngine',
    'SearchResult'
]



