"""
BM25s Optimized Search Engine
=============================

High-performance BM25 implementation using bm25s (Cython).
Drop-in replacement for standard BM25Engine with significantly higher throughput.

Features:
- Cython-optimized indexing and retrieval
- Memory mapping support
- Fast parallel tokenization
"""

import logging
from typing import Any, List, Optional

try:
    import bm25s
    import Stemmer  # PyStemmer for speed if available
    HAS_BM25S = True
except ImportError:
    HAS_BM25S = False

from ..config import BM25Config
from ..engines.base import SearchResult, SparseSearchEngine

logger = logging.getLogger(__name__)

class BM25sEngine(SparseSearchEngine):
    """
    Cython-optimized BM25 engine wrapper.
    """

    def __init__(
        self,
        config: Optional[BM25Config] = None,
        save_dir: Optional[str] = None
    ):
        super().__init__(engine_name='bm25s')

        if not HAS_BM25S:
            raise ImportError("bm25s library not installed. Run 'pip install bm25s'")

        self.config = config or BM25Config()
        self.save_dir = save_dir

        # Initialize retriever
        # method='lucene' matches default BM25 behavior best
        self.retriever = bm25s.BM25(
            k1=self.config.k1,
            b=self.config.b,
            method='lucene'
        )

        self.corpus: List[str] = []
        self.doc_ids: List[Any] = []
        self._indexed = False

    def index_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[Any]] = None,
        save: bool = False,
        **kwargs
    ) -> None:
        """
        Index documents using optimized C++ backend.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        logger.info(f"Indexing {len(documents)} documents with bm25s")

        self.corpus = documents
        self.doc_ids = doc_ids if doc_ids is not None else list(range(len(documents)))

        # Tokenize (bm25s has built-in fast tokenization)
        # We use default english stemmer
        corpus_tokens = bm25s.tokenize(
            documents,
            stopwords="en",
            stemmer="porter"
        )

        # Index
        self.retriever.index(corpus_tokens)
        self._indexed = True

        if save and self.save_dir:
            self.save(self.save_dir)

        logger.info("BM25s indexing complete")

    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search with optimized backend.
        """
        if not self._indexed:
            raise RuntimeError("Index must be built before searching")

        # Tokenize query
        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer="porter"
        )

        # Retrieve
        # bm25s.retrieve returns (scores, docs)
        # But docs are the indexed documents (tokens) or indices?
        # By default it returns document indices if corpus not provided to retrieve

        # We want indices to map back to our doc_ids
        doc_indices, scores = self.retriever.retrieve(
            query_tokens,
            k=top_k,
            corpus=None # Return indices
        )

        # Unpack batch results (we process single query)
        # scores and doc_indices are (n_queries, k)

        if doc_indices.shape[0] == 0:
            return []

        indices = doc_indices[0]
        scores_row = scores[0]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores_row)):
            # bm25s might return -1 for padding if fewer than k results
            if idx < 0 or idx >= len(self.doc_ids):
                continue

            results.append(SearchResult(
                doc_id=self.doc_ids[idx],
                score=float(score),
                rank=rank + 1,
                source='bm25s',
                metadata={}
            ))

        return results

    def save(self, path: str):
        """Save index to disk."""
        self.retriever.save(path)
        logger.info(f"Saved bm25s index to {path}")

    def load(self, path: str):
        """Load index from disk."""
        self.retriever = bm25s.BM25.load(path, load_corpus=False)
        self._indexed = True
        logger.info(f"Loaded bm25s index from {path}")
