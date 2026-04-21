"""
Storage Factory: Choose Sync or Async Storage

Provides a clean interface to select the appropriate storage backend.
"""

from pathlib import Path
from typing import Literal

from .async_storage import AsyncPipelineStorage
from .storage import Storage

StorageMode = Literal['sync', 'async']

def create_storage(
    path: Path,
    config,
    mode: StorageMode = 'sync',
    **kwargs
) -> 'Storage | AsyncPipelineStorage':
    """
    Create storage backend (sync or async).

    Args:
        path: Storage directory path
        config: Index configuration
        mode: 'sync' for synchronous (simple, reliable) or
              'async' for asynchronous (2x faster, GPU-optimized)
        **kwargs: Additional arguments for async storage
            - max_batch_size: Maximum batch size (default: 100)
            - num_workers: Number of I/O workers (default: 2)

    Returns:
        Storage instance

    Examples:
        # Synchronous (default, reliable)
        >>> storage = create_storage(path, config, mode='sync')

        # Asynchronous (faster, GPU-optimized)
        >>> storage = create_storage(path, config, mode='async')

    Notes:
        - Sync: Simple, reliable, good for <10K docs
        - Async: 2x faster, GPU-optimized, best for >10K docs
        - Both are production-ready
    """
    if mode == 'sync':
        return Storage(path, config)
    elif mode == 'async':
        max_batch_size = kwargs.get('max_batch_size', 100)
        num_workers = kwargs.get('num_workers', 2)
        return AsyncPipelineStorage(path, config, max_batch_size, num_workers)
    else:
        raise ValueError(f"Invalid storage mode: {mode}. Choose 'sync' or 'async'")


__all__ = ['create_storage', 'StorageMode']
