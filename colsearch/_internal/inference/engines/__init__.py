"""
Search Engines Module

Provides implementations for various search engines:
- ColBERT: Dense retrieval with MaxSim (+ fast-plaid for large indexes)
- ColPali: Visual document retrieval with MaxSim (images/PDFs)
- BM25: Sparse retrieval with canonical `bm25s` OSS path
- Neo4j: Graph-based search with Cypher
- HNSW: Approximate nearest neighbor with hnswlib
- MultiModal: Combined text + visual search

All engines implement the BaseSearchEngine interface.
"""

from ..engines.base import BaseSearchEngine, DenseSearchEngine, GraphSearchEngine, SearchResult, SparseSearchEngine
from ..engines.bm25 import BM25Engine as LegacyBM25Engine
from ..engines.bm25 import BM25Tokenizer, InvertedIndex
from ..engines.bm25s_engine import BM25sEngine
from ..engines.colbert import ColBERTEngine
from ..engines.colpali import ColPaliConfig, ColPaliEngine, MultiModalEngine
from ..engines.neo4j import Neo4jEngine

# Canonical sparse engine surface for OSS callers.
BM25Engine = BM25sEngine

# HNSW is optional (requires hnswlib)
try:
    from ..engines.hnsw import HNSWConfig, HNSWEngine
    _HNSW_AVAILABLE = True
except ImportError:
    HNSWEngine = None
    HNSWConfig = None
    _HNSW_AVAILABLE = False

__all__ = [
    # Base
    'BaseSearchEngine',
    'DenseSearchEngine',
    'SparseSearchEngine',
    'GraphSearchEngine',
    'SearchResult',
    # Dense engines
    'ColBERTEngine',
    'ColPaliEngine',
    'ColPaliConfig',
    'MultiModalEngine',
    # Sparse engines
    'BM25Engine',
    'BM25sEngine',
    'LegacyBM25Engine',
    'InvertedIndex',
    'BM25Tokenizer',
    # Graph engines
    'Neo4jEngine',
    # ANN engines
    'HNSWEngine',
    'HNSWConfig',
]



