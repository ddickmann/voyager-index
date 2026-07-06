"""
Configuration Module

Defines all configuration classes for the ColBERT search system.
Production-ready with validation and sensible defaults.

Author: ColBERT Team
License: Apache-2.0
"""

import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional


@dataclass
class IndexConfig:
    """
    Configuration for ColBERT index behavior and optimization.

    Three Modes for Different Use Cases:

    🚀 REAL-TIME MODE (< realtime_threshold, default 1000 docs):
        - Pure Triton, cached in VRAM
        - ColBERT search on steroids
        - Performance: 50+ QPS
        - Perfect for:
            * Real-time apps with dynamic indexes
            * Deep research and context engineering
            * Short-lived indexes where performance is critical
        - Use when: Index has limited half-life, speed > all

    💎 HIGH QUALITY MODE (realtime_threshold to balanced_threshold):
        - Triton + mmap streaming
        - No compromises from candidate generation
        - Performance: 0.3-1 QPS, exact search
        - Perfect for:
            * Healthcare, legal, industry where quality decides over time
            * Critical applications requiring 100% accuracy
            * Can be optimized with better hardware and storage
        - Use when: Quality > speed, no approximation acceptable

    ⚖️ BALANCED MODE (> balanced_threshold, default 50K docs):
        - PLAID + Triton hybrid (approximate candidates + exact reranking)
        - Long-term index with optimized speed-quality ratio
        - Performance: 0.5-2 QPS
        - Perfect for:
            * Production systems with large document collections
            * Long-term indexes requiring both speed and quality
            * General-purpose search at scale
        - Use when: Need both speed and quality, default for production

    Attributes:
        realtime_threshold: Documents below this use REAL-TIME MODE (default: 1000).
        balanced_threshold: Documents above this use BALANCED MODE (default: 50000).
        batch_size: Batch size for index creation.
        chunk_size: Number of documents per chunk for mmap streaming.
        cache_realtime_index: Whether to cache real-time indexes in VRAM.
        compression: HDF5 compression algorithm ('gzip', 'lzf', None).
        compression_level: Compression level (1-9 for gzip).
        device: Device to use ('cuda' or 'cpu').
        plaid_nbits: Bits for PLAID quantization (2, 4, or 8).
        plaid_n_ivf_probe: Number of IVF probes for PLAID search.
        plaid_n_full_scores: Number of candidates to fully score (internal).

    Example:
        >>> # Real-time mode for apps
        >>> config = IndexConfig(realtime_threshold=1000)
        >>>
        >>> # High quality mode for healthcare
        >>> config = IndexConfig(realtime_threshold=100, balanced_threshold=1000000)
        >>>
        >>> # Balanced mode for production
        >>> config = IndexConfig(realtime_threshold=1000, balanced_threshold=50000)
    """

    # Scaling thresholds (user-configurable)
    realtime_threshold: int = 1000  # Below: REAL-TIME MODE (in-memory)
    balanced_threshold: int = 50000  # Above: BALANCED MODE (PLAID + Triton)
    # Between thresholds: HIGH QUALITY MODE (Triton + mmap)

    # Legacy names (deprecated but supported for backward compatibility)
    small_threshold: Optional[int] = None  # Use realtime_threshold instead
    large_threshold: Optional[int] = None  # Use balanced_threshold instead

    # Memory management
    batch_size: int = 100
    chunk_size: int = 500
    cache_realtime_index: bool = True  # Cache real-time indexes in VRAM
    cache_small_index: Optional[bool] = None  # Deprecated: use cache_realtime_index

    # Storage
    compression: Optional[Literal['gzip', 'lzf']] = 'gzip'
    compression_level: int = 4

    # Device
    device: Literal['cuda', 'cpu'] = 'cuda'

    # PLAID configuration (for BALANCED MODE)
    plaid_nbits: Literal[2, 4, 8] = 4
    plaid_n_ivf_probe: int = 8
    plaid_n_full_scores: int = 4096  # Internal parameter, not user-facing

    def __post_init__(self) -> None:
        """Validate configuration parameters and handle backward compatibility."""

        # Backward compatibility: support old parameter names
        if self.small_threshold is not None:
            self.realtime_threshold = self.small_threshold
        if self.large_threshold is not None:
            self.balanced_threshold = self.large_threshold
        if self.cache_small_index is not None:
            self.cache_realtime_index = self.cache_small_index

        # Validation
        if self.realtime_threshold <= 0:
            raise ValueError("realtime_threshold must be positive")
        if self.balanced_threshold <= self.realtime_threshold:
            raise ValueError("balanced_threshold must be > realtime_threshold")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.compression_level < 1 or self.compression_level > 9:
            raise ValueError("compression_level must be between 1 and 9")
        if self.plaid_nbits not in [2, 4, 8]:
            raise ValueError("plaid_nbits must be 2, 4, or 8")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'IndexConfig':
        """Create configuration from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'IndexConfig':
        """Create configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FusionConfig:
    """
    Configuration for multi-engine fusion and ranking.

    Controls how results from multiple search engines (ColBERT, BM25, Neo4j)
    are combined and ranked.

    Attributes:
        strategy: Fusion strategy to use.
        weights: Optional weights for each engine (must sum to 1.0).
        normalization: Score normalization method.
        top_k: Number of final results to return.
        min_score: Minimum score threshold for results.

    Example:
        >>> config = FusionConfig(
        ...     strategy='rrf',
        ...     weights={'colbert': 0.6, 'bm25': 0.3, 'neo4j': 0.1}
        ... )
    """

    # Fusion strategy
    strategy: Literal['weighted_sum', 'rrf', 'reciprocal_rank', 'max', 'min'] = 'rrf'

    # Engine weights (optional, used by weighted_sum)
    weights: Optional[dict[str, float]] = None

    # Normalization method
    normalization: Literal['min_max', 'z_score', 'softmax', 'none'] = 'min_max'

    # Output parameters
    top_k: int = 10
    min_score: Optional[float] = None

    # RRF specific
    rrf_k: int = 60  # Constant for reciprocal rank fusion

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.weights is not None:
            total = sum(self.weights.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point error
                raise ValueError(f"Weights must sum to 1.0, got {total}")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'FusionConfig':
        """Create configuration from dictionary."""
        return cls(**data)


@dataclass
class BM25Config:
    """
    Configuration for BM25 search engine.

    Attributes:
        k1: Term frequency saturation parameter (typically 1.2-2.0).
        b: Length normalization parameter (typically 0.75).
        epsilon: Floor value for IDF (prevents division by zero).

    Example:
        >>> config = BM25Config(k1=1.5, b=0.75)
    """

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not (0 <= self.b <= 1):
            raise ValueError("b must be between 0 and 1")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")


@dataclass
class Neo4jConfig:
    """
    Configuration for Neo4j graph search engine.

    Attributes:
        uri: Neo4j database URI.
        username: Database username.
        password: Database password.
        database: Database name.
        max_hop_distance: Maximum number of hops for graph traversal.
        relationship_types: Relationship types to consider (None = all).

    Example:
        >>> config = Neo4jConfig(
        ...     uri="bolt://localhost:7687",
        ...     username="neo4j",
        ...     password="password"
        ... )
    """

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_hop_distance: int = 3
    relationship_types: Optional[list[str]] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.uri:
            raise ValueError("uri cannot be empty")
        if self.max_hop_distance <= 0:
            raise ValueError("max_hop_distance must be positive")


__all__ = [
    'IndexConfig',
    'FusionConfig',
    'BM25Config',
    'Neo4jConfig'
]


