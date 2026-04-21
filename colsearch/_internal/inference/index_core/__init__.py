"""
Core Module

Core functionality for ColBERT indexing and search.
Contains storage layer and index logic separated for clean architecture.

Modules:
- storage: HDF5-based storage with mmap support
- index: High-level index interface with automatic scaling
- feature_bridge: Intelligence to solver feature transformation
- embedder: API gateway integration for embeddings
"""

from .index import ColbertIndex
from .storage import IndexStatistics, Storage

__all__ = [
    'Storage',
    'IndexStatistics',
    'ColbertIndex',
]

try:
    from .feature_bridge import FeatureBridge, MaxSimBridge
except Exception:  # pragma: no cover - optional while solver boundary is normalized
    FeatureBridge = None
    MaxSimBridge = None
else:
    __all__.extend([
        'FeatureBridge',
        'MaxSimBridge',
    ])

try:
    from .embedder import ColBERTEmbedder, ColBERTTokenEmbedder
except Exception:  # pragma: no cover - optional external embedding dependencies
    ColBERTEmbedder = None
    ColBERTTokenEmbedder = None
else:
    __all__.extend([
        'ColBERTEmbedder',
        'ColBERTTokenEmbedder',
    ])
