"""
Enterprise Storage Layer

HDF5-based storage with mmap support for ColBERT embeddings.
Handles all disk I/O, CRUD operations, and metadata management.

Features:
- Batch streaming to disk (no RAM spikes)
- Compression (gzip)
- Mmap support for large-scale access
- Full CRUD operations
- Collection management
- Lazy deletion with compaction

Author: ColBERT Team
License: Apache-2.0
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class IndexStatistics:
    """
    Index statistics for monitoring.

    Attributes:
        num_documents: Number of active documents
        num_deleted: Number of deleted documents (soft delete)
        num_collections: Number of collections
        total_tokens: Total token count across all documents
        avg_tokens_per_doc: Average tokens per document
        max_tokens: Maximum token length (padding size)
        embed_dim: Embedding dimension
        storage_size_mb: Storage size in megabytes
        strategy: Current search strategy ('triton', 'triton_mmap', 'plaid_triton')
        created_at: Creation timestamp
        last_modified: Last modification timestamp
    """
    num_documents: int = 0
    num_deleted: int = 0
    num_collections: int = 0
    collections: List[str] = None
    total_tokens: int = 0
    avg_tokens_per_doc: float = 0.0
    max_tokens: int = 0
    embed_dim: int = 0
    storage_size_mb: float = 0.0
    strategy: str = "none"
    created_at: str = ""
    last_modified: str = ""

    def __post_init__(self):
        """Initialize mutable default."""
        if self.collections is None:
            self.collections = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'IndexStatistics':
        """Create from dictionary."""
        return cls(**d)


class Storage:
    """
    Production-grade storage with HDF5 + mmap.

    Handles all disk operations for ColBERT embeddings:
    - Create index from batches (true streaming)
    - Add/delete/update documents
    - Collection management
    - Compaction (remove deleted documents)
    - Statistics and metadata

    Storage Format:
        embeddings.h5: HDF5 file with embeddings dataset
            Shape: [num_docs, max_tokens, embed_dim]
            Dtype: float16 (2 bytes per value)
            Compression: gzip (configurable)
        metadata.json: JSON file with collections, doc metadata, etc.

    Attributes:
        storage_path: Path to storage directory
        config: Storage configuration
        collections: Dictionary mapping collection names to doc IDs
        doc_metadata: Per-document metadata
        doc_lengths: Actual token lengths (before padding)
        deleted_ids: Set of deleted document IDs
        num_docs: Total number of documents (including deleted)
        max_tokens: Maximum token length
        embed_dim: Embedding dimension
        next_doc_id: Next available document ID

    Example:
        >>> storage = Storage(Path("/data/storage"), config)
        >>>
        >>> # Create from batches
        >>> def batch_gen():
        ...     for batch in batches:
        ...         yield embeddings, metadata
        >>> doc_ids = storage.create_from_batches(batch_gen(), max_tokens=512)
        >>>
        >>> # Add documents
        >>> new_ids = storage.add_documents(new_embeddings)
        >>>
        >>> # Load documents
        >>> docs = storage.load_documents(doc_ids=[0, 1, 2])
    """

    def __init__(self, storage_path: Path, config):
        """
        Initialize storage.

        Args:
            storage_path: Directory for storage files
            config: Configuration object with storage settings
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.config = config

        # File paths
        self.data_file = self.storage_path / "embeddings.h5"
        self.metadata_file = self.storage_path / "metadata.json"

        # Metadata
        self.collections: Dict[str, List[int]] = {}
        self.doc_metadata: Dict[int, dict] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.deleted_ids: set = set()

        # Index info
        self.num_docs = 0
        self.max_tokens = 0
        self.embed_dim = 0
        self.next_doc_id = 0
        self.last_write_profile: Dict[str, object] = {}
        self.last_read_profile: Dict[str, object] = {}

        logger.info(f"Initialized storage at {storage_path}")

    def _compression_kwargs(self) -> Dict[str, object]:
        if self.config.compression is None:
            return {}
        kwargs: Dict[str, object] = {"compression": self.config.compression}
        if self.config.compression == "gzip":
            kwargs["compression_opts"] = self.config.compression_level
        return kwargs

    def create_from_batches(
        self,
        batch_generator: Generator[Tuple[torch.Tensor, Optional[List[dict]]], None, None],
        collection_name: str = "default",
        show_progress: bool = True,
        max_tokens: Optional[int] = None
    ) -> List[int]:
        """
        Create index from batch generator (TRUE STREAMING - no RAM spike!).

        This method processes embeddings in batches, writing directly to disk
        without accumulating data in RAM. For true streaming, provide max_tokens
        upfront. Otherwise, a lightweight first pass determines max_tokens.

        Args:
            batch_generator: Yields (embeddings, metadata) tuples
                embeddings: [batch_size, num_tokens, embed_dim]
                metadata: Optional list of metadata dicts (one per document)
            collection_name: Collection name for grouping documents
            show_progress: Whether to print progress messages
            max_tokens: Maximum token length (enables true single-pass streaming)

        Returns:
            List of assigned document IDs

        Raises:
            ValueError: If batch format is invalid

        Note:
            If max_tokens is not provided, the generator will be consumed in a
            lightweight first pass to determine max_tokens. You'll need to provide
            a new generator or call with max_tokens specified.

        Example:
            >>> def batch_gen():
            ...     for i in range(0, len(docs), batch_size):
            ...         yield doc_embeddings[i:i+batch_size], None
            >>>
            >>> doc_ids = storage.create_from_batches(
            ...     batch_gen(),
            ...     collection_name="main",
            ...     max_tokens=512  # Recommended for true streaming!
            ... )
        """
        logger.info(f"Creating index from batches (collection: {collection_name})")

        assigned_ids = []
        batch_idx = 0
        total_write_ms = 0.0

        # If max_tokens not provided, do lightweight first pass
        if max_tokens is None:
            logger.info("First pass: Determining max_tokens (shape only, no data retained)...")
            max_tokens_seen = 0
            embed_dim = None
            num_batches = 0

            for embeddings, metadata in batch_generator:
                _, num_tokens, emb_dim = embeddings.shape
                max_tokens_seen = max(max_tokens_seen, num_tokens)
                if embed_dim is None:
                    embed_dim = emb_dim
                num_batches += 1

                # Don't keep embeddings in memory!
                del embeddings
                if metadata:
                    del metadata

            self.embed_dim = embed_dim
            self.max_tokens = max_tokens_seen

            logger.warning(f"⚠️  First pass complete ({num_batches} batches scanned)")
            logger.warning(f"⚠️  max_tokens={self.max_tokens}, embed_dim={self.embed_dim}")
            logger.warning("⚠️  NOTE: Generator consumed! Provide new generator or specify max_tokens upfront.")

            return assigned_ids
        else:
            self.max_tokens = max_tokens
            logger.info(f"Using provided max_tokens={self.max_tokens}")

        # TRUE STREAMING: Process batches one at a time
        logger.info("Streaming batches to disk...")

        dataset_created = False

        with h5py.File(self.data_file, 'a') as f:
            for embeddings, metadata in batch_generator:
                batch_size, num_tokens, embed_dim = embeddings.shape

                # Initialize embed_dim from first batch
                if self.embed_dim == 0:
                    self.embed_dim = embed_dim

                # Create dataset on first batch
                if not dataset_created and 'embeddings' not in f:
                    compression_kwargs = self._compression_kwargs()
                    f.create_dataset(
                        'embeddings',
                        shape=(0, self.max_tokens, self.embed_dim),
                        maxshape=(None, self.max_tokens, self.embed_dim),
                        dtype='float16',
                        chunks=(self.config.batch_size, self.max_tokens, self.embed_dim),
                        **compression_kwargs,
                    )
                    dataset_created = True

                dataset = f['embeddings']

                # Pad to max_tokens if needed
                if num_tokens < self.max_tokens:
                    padding = torch.zeros(
                        batch_size,
                        self.max_tokens - num_tokens,
                        self.embed_dim,
                        dtype=embeddings.dtype,
                        device=embeddings.device
                    )
                    embeddings = torch.cat([embeddings, padding], dim=1)

                # Write to disk immediately
                write_start = datetime.now().timestamp()
                start_idx = dataset.shape[0]
                dataset.resize((start_idx + batch_size, self.max_tokens, self.embed_dim))
                dataset[start_idx:start_idx + batch_size] = embeddings.cpu().numpy().astype(np.float16)
                total_write_ms += (datetime.now().timestamp() - write_start) * 1000.0

                # Record metadata
                for i in range(batch_size):
                    doc_id = self.next_doc_id
                    self.next_doc_id += 1
                    assigned_ids.append(doc_id)

                    self.doc_lengths[doc_id] = num_tokens
                    if metadata:
                        self.doc_metadata[doc_id] = metadata[i]

                del embeddings

                batch_idx += 1
                if show_progress and batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Wrote {len(assigned_ids)} documents total")

        # Update collection
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        self.collections[collection_name].extend(assigned_ids)

        self.num_docs = len(assigned_ids)

        # Save metadata
        self._save_metadata()
        self.last_write_profile = {
            "mode": "create_from_batches",
            "doc_count": len(assigned_ids),
            "batch_count": batch_idx,
            "write_ms": total_write_ms,
        }

        logger.info(f"Created index: {self.num_docs} documents in collection '{collection_name}'")
        return assigned_ids

    def add_documents(
        self,
        embeddings: torch.Tensor,
        collection_name: str = "default",
        metadata: Optional[List[dict]] = None
    ) -> List[int]:
        """
        Add documents to existing index.

        Args:
            embeddings: Document embeddings [num_docs, num_tokens, embed_dim]
            collection_name: Collection name
            metadata: Optional metadata per document

        Returns:
            List of assigned document IDs

        Raises:
            ValueError: If embeddings have more tokens than max_tokens

        Example:
            >>> new_ids = storage.add_documents(
            ...     new_embeddings,
            ...     collection_name="updates"
            ... )
        """
        num_docs, num_tokens, embed_dim = embeddings.shape
        logger.info(f"Adding {num_docs} documents to collection '{collection_name}'")

        # Initialize if first add
        if not hasattr(self, 'max_tokens') or self.max_tokens == 0:
            self.max_tokens = num_tokens
            self.embed_dim = embed_dim

        # Pad to max_tokens if needed
        if num_tokens < self.max_tokens:
            padding = torch.zeros(
                num_docs,
                self.max_tokens - num_tokens,
                embed_dim,
                dtype=embeddings.dtype,
                device=embeddings.device
            )
            embeddings = torch.cat([embeddings, padding], dim=1)
        elif num_tokens > self.max_tokens:
            raise ValueError(
                f"Cannot add documents with {num_tokens} tokens to index with max_tokens={self.max_tokens}"
            )

        # Add to HDF5
        assigned_ids = []

        with h5py.File(self.data_file, 'a') as f:
            dataset = f['embeddings']

            write_start = datetime.now().timestamp()
            start_idx = dataset.shape[0]
            dataset.resize((start_idx + num_docs, self.max_tokens, self.embed_dim))
            dataset[start_idx:start_idx + num_docs] = embeddings.cpu().numpy().astype(np.float16)
            write_ms = (datetime.now().timestamp() - write_start) * 1000.0

            # Record metadata
            for i in range(num_docs):
                doc_id = self.next_doc_id
                self.next_doc_id += 1
                assigned_ids.append(doc_id)

                self.doc_lengths[doc_id] = num_tokens
                if metadata:
                    self.doc_metadata[doc_id] = metadata[i]

        # Update collection
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        self.collections[collection_name].extend(assigned_ids)

        self.num_docs += num_docs
        self._save_metadata()
        self.last_write_profile = {
            "mode": "add_documents",
            "doc_count": num_docs,
            "batch_count": 1,
            "write_ms": write_ms,
        }

        logger.info(f"Added {num_docs} documents")
        return assigned_ids

    def delete_documents(self, doc_ids: List[int]) -> None:
        """
        Mark documents as deleted (lazy deletion).

        Documents are not physically removed until compact() is called.

        Args:
            doc_ids: Document IDs to delete

        Example:
            >>> storage.delete_documents([0, 1, 2])
            >>> storage.compact()  # Physical removal
        """
        self.deleted_ids.update(doc_ids)
        logger.info(f"Marked {len(doc_ids)} documents as deleted")
        self._save_metadata()

    def update_documents(
        self,
        doc_ids: List[int],
        embeddings: torch.Tensor
    ) -> None:
        """
        Update documents (in-place replacement).

        Args:
            doc_ids: Document IDs to update
            embeddings: New embeddings [len(doc_ids), max_tokens, embed_dim]

        Example:
            >>> storage.update_documents([0, 1], updated_embeddings)
        """
        # Delete old
        self.delete_documents(doc_ids)

        # Update in-place
        with h5py.File(self.data_file, 'a') as f:
            dataset = f['embeddings']
            for i, doc_id in enumerate(doc_ids):
                if doc_id < dataset.shape[0]:
                    dataset[doc_id] = embeddings[i].cpu().numpy().astype(np.float16)
                    self.deleted_ids.discard(doc_id)

        logger.info(f"Updated {len(doc_ids)} documents")
        self._save_metadata()

    def create_collection(self, name: str, doc_ids: List[int]) -> None:
        """
        Create a new collection.

        Args:
            name: Collection name
            doc_ids: Document IDs in collection
        """
        self.collections[name] = doc_ids
        logger.info(f"Created collection '{name}' with {len(doc_ids)} documents")
        self._save_metadata()

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.

        Args:
            name: Collection name
        """
        if name in self.collections:
            del self.collections[name]
            logger.info(f"Deleted collection '{name}'")
            self._save_metadata()

    def compact(self) -> None:
        """
        Compact storage by physically removing deleted documents.

        Creates a new HDF5 file with only active documents, then replaces the old file.
        This operation can be slow for large indexes.

        Example:
            >>> storage.delete_documents([0, 1, 2])
            >>> storage.compact()  # Physically remove deleted docs
        """
        logger.info("Compacting storage...")

        # Get active documents
        active_ids = [i for i in range(self.num_docs) if i not in self.deleted_ids]

        if len(active_ids) == self.num_docs:
            logger.info("No compaction needed")
            return

        # Create new file
        new_file = self.storage_path / "embeddings_new.h5"

        with h5py.File(self.data_file, 'r') as f_old:
            old_dataset = f_old['embeddings']

            with h5py.File(new_file, 'w') as f_new:
                # Adjust chunk size for small datasets
                chunk_size = min(self.config.batch_size, len(active_ids))
                compression_kwargs = self._compression_kwargs()
                new_dataset = f_new.create_dataset(
                    'embeddings',
                    shape=(len(active_ids), old_dataset.shape[1], old_dataset.shape[2]),
                    dtype='float16',
                    chunks=(chunk_size, old_dataset.shape[1], old_dataset.shape[2]),
                    **compression_kwargs,
                )

                # Copy active documents
                for new_idx, old_idx in enumerate(active_ids):
                    new_dataset[new_idx] = old_dataset[old_idx]

        # Replace old file
        self.data_file.unlink()
        new_file.rename(self.data_file)

        # Update metadata
        self.num_docs = len(active_ids)
        self.deleted_ids.clear()

        logger.info(f"Compaction complete: {len(active_ids)} active documents")
        self._save_metadata()

    def load_documents(
        self,
        doc_ids: Optional[List[int]] = None,
        collection_name: Optional[str] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Load documents from storage.

        Args:
            doc_ids: Specific document IDs to load (None = all active)
            collection_name: Load entire collection
            device: Target device ('cuda' or 'cpu')

        Returns:
            Tensor of shape [num_docs, max_tokens, embed_dim]

        Example:
            >>> docs = storage.load_documents(doc_ids=[0, 1, 2], device='cuda')
            >>> print(docs.shape)  # [3, max_tokens, embed_dim]
        """
        if collection_name:
            doc_ids = self.collections.get(collection_name, [])
            doc_ids = [i for i in doc_ids if i not in self.deleted_ids]

        if not doc_ids:
            doc_ids = [i for i in range(self.num_docs) if i not in self.deleted_ids]

        # Filter deleted
        doc_ids = [i for i in doc_ids if i not in self.deleted_ids]

        with h5py.File(self.data_file, 'r') as f:
            dataset = f['embeddings']

            # Load in chunks to avoid RAM spike
            read_start = datetime.now().timestamp()
            chunks = []
            for i in range(0, len(doc_ids), self.config.chunk_size):
                chunk_ids = doc_ids[i:i + self.config.chunk_size]
                order = np.argsort(np.asarray(chunk_ids, dtype=np.int64))
                sorted_ids = [chunk_ids[idx] for idx in order.tolist()]
                chunk_data = dataset[sorted_ids]
                if order.tolist() != list(range(len(chunk_ids))):
                    inverse_order = np.argsort(order)
                    chunk_data = chunk_data[inverse_order]
                chunks.append(torch.from_numpy(chunk_data))

            # Concatenate and move to device
            all_docs = torch.cat(chunks, dim=0).to(device)
            self.last_read_profile = {
                "mode": "load_documents",
                "doc_count": len(doc_ids),
                "chunk_count": len(chunks),
                "elapsed_ms": (datetime.now().timestamp() - read_start) * 1000.0,
            }

        return all_docs

    def load_documents_streaming(
        self,
        doc_ids: Optional[List[int]] = None,
        device: str = 'cuda'
    ) -> Generator[torch.Tensor, None, None]:
        """
        Stream documents in chunks (for large-scale processing).

        Yields documents in chunks to avoid loading entire dataset into memory.

        Args:
            doc_ids: Document IDs to stream (None = all active)
            device: Target device

        Yields:
            Document chunks [chunk_size, max_tokens, embed_dim]

        Example:
            >>> for chunk in storage.load_documents_streaming(device='cuda'):
            ...     scores = compute_scores(queries, chunk)
            ...     # Process chunk
        """
        if not doc_ids:
            doc_ids = [i for i in range(self.num_docs) if i not in self.deleted_ids]

        doc_ids = [i for i in doc_ids if i not in self.deleted_ids]

        with h5py.File(self.data_file, 'r') as f:
            dataset = f['embeddings']

            for i in range(0, len(doc_ids), self.config.chunk_size):
                chunk_ids = doc_ids[i:i + self.config.chunk_size]
                order = np.argsort(np.asarray(chunk_ids, dtype=np.int64))
                sorted_ids = [chunk_ids[idx] for idx in order.tolist()]
                chunk_data = dataset[sorted_ids]
                if order.tolist() != list(range(len(chunk_ids))):
                    inverse_order = np.argsort(order)
                    chunk_data = chunk_data[inverse_order]
                yield torch.from_numpy(chunk_data).to(device)

    def get_statistics(self, small_threshold: int, large_threshold: int) -> IndexStatistics:
        """
        Get index statistics.

        Args:
            small_threshold: Threshold for REAL-TIME MODE (pure Triton, in-memory)
            large_threshold: Threshold for BALANCED MODE (PLAID + Triton)

        Returns:
            IndexStatistics object with mode selection
        """
        active_docs = [i for i in range(self.num_docs) if i not in self.deleted_ids]

        total_tokens = sum(self.doc_lengths.get(i, 0) for i in active_docs)
        avg_tokens = total_tokens / len(active_docs) if active_docs else 0

        storage_size = self.data_file.stat().st_size / 1024**2 if self.data_file.exists() else 0

        # Determine strategy based on corpus size
        num_active = len(active_docs)
        if num_active < small_threshold:
            strategy = "realtime"  # REAL-TIME MODE: Pure Triton (in-memory)
        elif num_active < large_threshold:
            strategy = "high_quality"  # HIGH QUALITY MODE: Triton + mmap (exact)
        else:
            strategy = "balanced"  # BALANCED MODE: PLAID + Triton (hybrid)

        return IndexStatistics(
            num_documents=len(active_docs),
            num_deleted=len(self.deleted_ids),
            num_collections=len(self.collections),
            collections=list(self.collections.keys()),
            total_tokens=total_tokens,
            avg_tokens_per_doc=avg_tokens,
            max_tokens=self.max_tokens,
            embed_dim=self.embed_dim,
            storage_size_mb=storage_size,
            strategy=strategy,
            created_at=self.doc_metadata.get(-1, {}).get('created_at', ''),
            last_modified=datetime.now().isoformat()
        )

    def _save_metadata(self) -> None:
        """Save metadata to JSON."""
        metadata = {
            'num_docs': self.num_docs,
            'max_tokens': self.max_tokens,
            'embed_dim': self.embed_dim,
            'next_doc_id': self.next_doc_id,
            'collections': self.collections,
            'doc_metadata': {str(k): v for k, v in self.doc_metadata.items()},
            'doc_lengths': {str(k): v for k, v in self.doc_lengths.items()},
            'deleted_ids': list(self.deleted_ids),
            'last_modified': datetime.now().isoformat()
        }

        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.metadata_file.with_name(f".{self.metadata_file.name}.tmp")
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, self.metadata_file)
        parent_fd = os.open(self.metadata_file.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def load_metadata(self) -> None:
        """Load metadata from JSON."""
        if not self.metadata_file.exists():
            return

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        self.num_docs = metadata['num_docs']
        self.max_tokens = metadata['max_tokens']
        self.embed_dim = metadata['embed_dim']
        self.next_doc_id = metadata['next_doc_id']
        self.collections = metadata['collections']
        self.doc_metadata = {int(k): v for k, v in metadata['doc_metadata'].items()}
        self.doc_lengths = {int(k): v for k, v in metadata['doc_lengths'].items()}
        self.deleted_ids = set(metadata['deleted_ids'])


__all__ = ['Storage', 'IndexStatistics']

