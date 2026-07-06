"""
Async Enterprise Storage Layer

High-performance async storage with GPU-optimized pipeline.
Implements proper async I/O, CUDA streams, and pinned memory for world-class performance.

Features:
- Async GPU→CPU transfers (non-blocking)
- Pinned memory for fast DMA transfers
- Async I/O with thread pool
- Triple-buffering pipeline
- Zero CPU RAM accumulation
- Scales to millions of documents

Author: ColBERT Team
License: Apache-2.0
"""

import json
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


class AsyncWriter:
    """
    Async HDF5 writer with pipelining.

    Implements triple-buffering:
    - Buffer 1: GPU compute
    - Buffer 2: GPU→CPU transfer
    - Buffer 3: CPU→Disk write

    This allows overlapping compute, transfer, and I/O for maximum throughput.
    """

    def __init__(self, h5_file: h5py.File, dataset_name: str, max_queue_size: int = 3):
        """
        Initialize async writer.

        Args:
            h5_file: Open HDF5 file
            dataset_name: Dataset name to write to
            max_queue_size: Maximum queue size (controls memory usage)
        """
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.write_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.error = None

        # Start writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

        logger.info(f"AsyncWriter started for dataset '{dataset_name}'")

    def _writer_loop(self):
        """Background thread that writes to HDF5."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get next write task (with timeout to check stop event)
                    item = self.write_queue.get(timeout=0.1)
                    if item is None:  # Sentinel value
                        break

                    start_idx, data = item
                    dataset = self.h5_file[self.dataset_name]

                    # Resize and write (blocking HDF5 operation)
                    batch_size = data.shape[0]
                    current_size = dataset.shape[0]

                    if start_idx >= current_size:
                        dataset.resize((start_idx + batch_size, dataset.shape[1], dataset.shape[2]))

                    dataset[start_idx:start_idx + batch_size] = data

                    # Mark task as done
                    self.write_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    self.error = e
                    logger.error(f"AsyncWriter error: {e}")
                    break
        except Exception as e:
            self.error = e
            logger.error(f"AsyncWriter fatal error: {e}")

    def write_async(self, start_idx: int, data: np.ndarray):
        """
        Queue data for async writing.

        Args:
            start_idx: Starting index in dataset
            data: NumPy array to write

        Raises:
            RuntimeError: If writer thread encountered an error
        """
        if self.error:
            raise RuntimeError(f"AsyncWriter failed: {self.error}")

        # Block if queue is full (backpressure)
        self.write_queue.put((start_idx, data))

    def finish(self):
        """Wait for all writes to complete."""
        # Send sentinel
        self.write_queue.put(None)

        # Wait for thread to finish
        self.writer_thread.join()

        if self.error:
            raise RuntimeError(f"AsyncWriter failed: {self.error}")

        logger.info("AsyncWriter finished")


class PinnedMemoryPool:
    """
    Pool of pinned (page-locked) CPU memory for fast GPU↔CPU transfers.

    Pinned memory enables Direct Memory Access (DMA) for 2-3x faster transfers.
    """

    def __init__(self, buffer_size: Tuple[int, int, int], num_buffers: int = 3, dtype=np.float16):
        """
        Initialize pinned memory pool.

        Args:
            buffer_size: Shape of each buffer (batch_size, tokens, dim)
            num_buffers: Number of buffers in pool
            dtype: Data type
        """
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.available = queue.Queue()

        # Allocate pinned memory buffers
        logger.info(f"Allocating {num_buffers} pinned memory buffers of shape {buffer_size}")

        for _ in range(num_buffers):
            # Allocate pinned memory on CPU
            buffer = torch.empty(buffer_size, dtype=torch.float16, pin_memory=True)
            self.available.put(buffer)

        logger.info(f"PinnedMemoryPool ready with {num_buffers} buffers")

    def get(self) -> torch.Tensor:
        """Get a free buffer (blocks if none available)."""
        return self.available.get()

    def release(self, buffer: torch.Tensor):
        """Release buffer back to pool."""
        self.available.put(buffer)


class AsyncPipelineStorage:
    """
    Async storage with GPU-optimized pipeline.

    Pipeline stages:
    1. GPU compute (embedding generation)
    2. GPU→CPU transfer (async, pinned memory)
    3. CPU→Disk write (async, thread pool)

    All stages overlap for maximum throughput.
    """

    def __init__(
        self,
        storage_path: Path,
        config,
        max_batch_size: int = 100,
        num_workers: int = 2
    ):
        """
        Initialize async pipeline storage.

        Args:
            storage_path: Storage directory
            config: Index configuration
            max_batch_size: Maximum batch size for memory allocation
            num_workers: Number of I/O worker threads
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.max_batch_size = max_batch_size

        # Paths
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

        # Async components
        self.memory_pool: Optional[PinnedMemoryPool] = None
        self.async_writer: Optional[AsyncWriter] = None
        self.cuda_stream: Optional[torch.cuda.Stream] = None

        # Thread pool for I/O
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        logger.info(f"Initialized AsyncPipelineStorage at {storage_path}")

    def create_from_batches_async(
        self,
        batch_generator: Generator[Tuple[torch.Tensor, Optional[List[dict]]], None, None],
        collection_name: str = "default",
        show_progress: bool = True,
        max_tokens: Optional[int] = None
    ) -> List[int]:
        """
        Create index with async pipeline (WORLD-CLASS PERFORMANCE).

        Pipeline:
        - Stage 1: GPU compute (embedding on GPU)
        - Stage 2: GPU→CPU (async copy to pinned memory)
        - Stage 3: CPU→Disk (async HDF5 write)

        All stages overlap for maximum throughput!

        Args:
            batch_generator: Yields (embeddings, metadata) on GPU
            collection_name: Collection name
            show_progress: Show progress
            max_tokens: Max token length

        Returns:
            List of document IDs
        """
        logger.info(f"Creating index with ASYNC PIPELINE (collection: {collection_name})")

        assigned_ids = []
        batch_idx = 0

        # Determine max_tokens if not provided
        if max_tokens is None:
            logger.info("First pass: Determining max_tokens...")
            max_tokens_seen = 0
            embed_dim = None

            for embeddings, metadata in batch_generator:
                _, num_tokens, emb_dim = embeddings.shape
                max_tokens_seen = max(max_tokens_seen, num_tokens)
                if embed_dim is None:
                    embed_dim = emb_dim
                del embeddings
                if metadata:
                    del metadata

            self.embed_dim = embed_dim
            self.max_tokens = max_tokens_seen

            logger.warning(f"⚠️  max_tokens={self.max_tokens}, embed_dim={self.embed_dim}")
            logger.warning("⚠️  Provide new generator or specify max_tokens upfront")
            return assigned_ids
        else:
            self.max_tokens = max_tokens
            logger.info(f"Using provided max_tokens={self.max_tokens}")

        # Initialize async components
        if torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()
            logger.info("✅ CUDA stream created for async transfers")

        # We need to peek at first batch to get embed_dim, but we can't exhaust the generator
        # Solution: Use itertools.tee to create two independent iterators
        import itertools
        peek_iter, main_iter = itertools.tee(batch_generator, 2)

        # Peek at first batch to determine embed_dim
        try:
            first_embeddings, _ = next(peek_iter)
            if self.embed_dim == 0:
                self.embed_dim = first_embeddings.shape[2]
                logger.info(f"Determined embed_dim={self.embed_dim} from first batch")
            del peek_iter  # Free the peek iterator
            del first_embeddings
        except StopIteration:
            logger.error("No batches provided!")
            return []

        # Initialize pinned memory pool
        # We need 3 buffers for triple-buffering
        buffer_shape = (self.max_batch_size, self.max_tokens, self.embed_dim)
        self.memory_pool = PinnedMemoryPool(buffer_shape, num_buffers=3)
        logger.info("✅ Pinned memory pool initialized")

        # Open HDF5 file and create async writer
        with h5py.File(self.data_file, 'a') as f:
            # Create dataset
            if 'embeddings' not in f:
                f.create_dataset(
                    'embeddings',
                    shape=(0, self.max_tokens, self.embed_dim),
                    maxshape=(None, self.max_tokens, self.embed_dim),
                    dtype='float16',
                    chunks=(self.config.batch_size, self.max_tokens, self.embed_dim),
                    compression=self.config.compression,
                    compression_opts=self.config.compression_level
                )
                logger.info("✅ HDF5 dataset created")

            # Initialize async writer
            self.async_writer = AsyncWriter(f, 'embeddings', max_queue_size=3)
            logger.info("✅ Async writer started")

            # Process batches with async pipeline (TRUE STREAMING!)
            pending_transfers = []

            for embeddings, metadata in main_iter:
                batch_size, num_tokens, embed_dim = embeddings.shape

                # Initialize embed_dim from first batch
                if self.embed_dim == 0:
                    self.embed_dim = embed_dim

                # Ensure embeddings are on GPU
                if not embeddings.is_cuda:
                    embeddings = embeddings.cuda()

                # Pad to max_tokens if needed (on GPU!)
                if num_tokens < self.max_tokens:
                    padding = torch.zeros(
                        batch_size,
                        self.max_tokens - num_tokens,
                        self.embed_dim,
                        dtype=embeddings.dtype,
                        device=embeddings.device
                    )
                    embeddings = torch.cat([embeddings, padding], dim=1)

                # Convert to FP16 on GPU (if not already)
                if embeddings.dtype != torch.float16:
                    embeddings = embeddings.half()

                # Get pinned memory buffer from pool
                pinned_buffer = self.memory_pool.get()

                # ASYNC GPU→CPU transfer (non-blocking!)
                with torch.cuda.stream(self.cuda_stream):
                    # Slice buffer to actual batch size
                    target_buffer = pinned_buffer[:batch_size]
                    target_buffer.copy_(embeddings, non_blocking=True)

                # Store for later synchronization
                start_idx = len(assigned_ids)
                pending_transfers.append((start_idx, target_buffer, pinned_buffer, batch_size))

                # Record metadata
                for i in range(batch_size):
                    doc_id = self.next_doc_id
                    self.next_doc_id += 1
                    assigned_ids.append(doc_id)
                    self.doc_lengths[doc_id] = num_tokens
                    if metadata:
                        self.doc_metadata[doc_id] = metadata[i]

                # Free GPU memory immediately (transfer is async!)
                del embeddings

                # Process completed transfers (if any)
                if len(pending_transfers) >= 2:  # Keep 1-2 transfers in flight
                    self._process_transfer(pending_transfers.pop(0))

                batch_idx += 1
                if show_progress and batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Queued {len(assigned_ids)} documents")

            # Process remaining transfers
            logger.info("Finishing pending transfers...")
            for transfer in pending_transfers:
                self._process_transfer(transfer)

            # Wait for all async writes to complete
            logger.info("Waiting for async writes to complete...")
            self.async_writer.finish()
            logger.info("✅ All writes complete!")

        # Update collection
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        self.collections[collection_name].extend(assigned_ids)

        self.num_docs = len(assigned_ids)
        self._save_metadata()

        logger.info(f"✅ Index created: {self.num_docs} documents with ASYNC PIPELINE")
        return assigned_ids

    def _process_transfer(self, transfer_info: Tuple):
        """
        Process a completed GPU→CPU transfer.

        Synchronizes the CUDA stream and queues data for async write.
        """
        start_idx, target_buffer, full_buffer, batch_size = transfer_info

        # Wait for GPU→CPU transfer to complete
        self.cuda_stream.synchronize()

        # Convert to numpy (CPU-side, already in pinned memory)
        numpy_data = target_buffer.cpu().numpy()

        # Queue for async write to disk
        self.async_writer.write_async(start_idx, numpy_data)

        # Release buffer back to pool
        self.memory_pool.release(full_buffer)

    def _save_metadata(self):
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

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self):
        """Load metadata from disk."""
        if not self.metadata_file.exists():
            return

        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)

        self.num_docs = metadata.get('num_docs', 0)
        self.max_tokens = metadata.get('max_tokens', 0)
        self.embed_dim = metadata.get('embed_dim', 0)
        self.next_doc_id = metadata.get('next_doc_id', 0)
        self.collections = metadata.get('collections', {})
        self.doc_metadata = {int(k): v for k, v in metadata.get('doc_metadata', {}).items()}
        self.doc_lengths = {int(k): v for k, v in metadata.get('doc_lengths', {}).items()}
        self.deleted_ids = set(metadata.get('deleted_ids', []))

        logger.info(f"Loaded metadata: {self.num_docs} docs, {len(self.deleted_ids)} deleted")

    def get_statistics(self, small_threshold: int, large_threshold: int) -> 'IndexStatistics':
        """
        Get index statistics.

        Args:
            small_threshold: Threshold for REAL-TIME MODE (pure Triton, in-memory)
            large_threshold: Threshold for BALANCED MODE (PLAID + Triton)

        Returns:
            IndexStatistics object with mode selection
        """
        from .storage import IndexStatistics

        active_docs = [
            doc_id for doc_id in range(self.next_doc_id)
            if doc_id not in self.deleted_ids
        ]

        storage_size_mb = 0
        if self.data_file.exists():
            storage_size_mb = self.data_file.stat().st_size / 1024**2

        # Determine strategy based on corpus size
        num_docs = len(active_docs)
        if num_docs < small_threshold:
            strategy = "realtime"  # REAL-TIME MODE: Pure Triton (in-memory)
        elif num_docs < large_threshold:
            strategy = "high_quality"  # HIGH QUALITY MODE: Triton + mmap (exact)
        else:
            strategy = "balanced"  # BALANCED MODE: PLAID + Triton (hybrid)

        # Calculate total tokens
        total_tokens = sum(self.doc_lengths.get(doc_id, 0) for doc_id in active_docs)
        avg_tokens = total_tokens / num_docs if num_docs > 0 else 0

        return IndexStatistics(
            num_documents=num_docs,
            num_deleted=len(self.deleted_ids),
            num_collections=len(self.collections),
            collections=list(self.collections.keys()),
            total_tokens=total_tokens,
            avg_tokens_per_doc=avg_tokens,
            max_tokens=self.max_tokens,
            embed_dim=self.embed_dim,
            storage_size_mb=storage_size_mb,
            strategy=strategy,
            created_at="",
            last_modified=""
        )

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
        """
        import h5py

        # Determine which docs to load
        if doc_ids is None:
            if collection_name:
                doc_ids = [d for d in self.collections.get(collection_name, []) if d not in self.deleted_ids]
            else:
                doc_ids = [d for d in range(self.next_doc_id) if d not in self.deleted_ids]

        if not doc_ids:
            return torch.empty((0, self.max_tokens, self.embed_dim), device=device)

        # Load from HDF5
        with h5py.File(self.data_file, 'r') as f:
            dataset = f['embeddings']
            embeddings = torch.from_numpy(dataset[doc_ids])

        return embeddings.to(device)

    def load_documents_streaming(
        self,
        doc_ids: Optional[List[int]] = None,
        device: str = 'cuda',
        chunk_size: int = 1000
    ):
        """
        Stream documents in chunks (for large-scale processing).

        Args:
            doc_ids: Document IDs to stream (None = all active)
            device: Target device
            chunk_size: Number of documents per chunk

        Yields:
            Document chunks [chunk_size, max_tokens, embed_dim]
        """
        import h5py

        # Determine which docs to load
        if doc_ids is None:
            doc_ids = [d for d in range(self.next_doc_id) if d not in self.deleted_ids]

        # Stream in chunks
        with h5py.File(self.data_file, 'r') as f:
            dataset = f['embeddings']

            for i in range(0, len(doc_ids), chunk_size):
                chunk_ids = doc_ids[i:i+chunk_size]
                chunk = torch.from_numpy(dataset[chunk_ids]).to(device)
                yield chunk

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
        """
        import h5py

        num_docs, num_tokens, embed_dim = embeddings.shape

        # Initialize dimensions if needed
        if self.max_tokens == 0:
            self.max_tokens = num_tokens
        if self.embed_dim == 0:
            self.embed_dim = embed_dim

        # Pad if needed
        if num_tokens < self.max_tokens:
            padding = torch.zeros(
                num_docs,
                self.max_tokens - num_tokens,
                self.embed_dim,
                dtype=embeddings.dtype,
                device=embeddings.device
            )
            embeddings = torch.cat([embeddings, padding], dim=1)

        # Assign IDs
        doc_ids = []
        for i in range(num_docs):
            doc_id = self.next_doc_id
            self.next_doc_id += 1
            doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = num_tokens
            if metadata and i < len(metadata):
                self.doc_metadata[doc_id] = metadata[i]

        # Write to HDF5
        with h5py.File(self.data_file, 'a') as f:
            dataset = f['embeddings']
            current_size = dataset.shape[0]
            dataset.resize((current_size + num_docs, self.max_tokens, self.embed_dim))
            dataset[current_size:current_size + num_docs] = embeddings.cpu().numpy().astype(np.float16)

        # Update collection
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        self.collections[collection_name].extend(doc_ids)

        self.num_docs += num_docs
        self._save_metadata()

        logger.info(f"Added {num_docs} documents to collection '{collection_name}'")
        return doc_ids

    def update_documents(self, doc_ids: List[int], embeddings: torch.Tensor) -> None:
        """
        Update existing documents.

        Args:
            doc_ids: Document IDs to update
            embeddings: New embeddings [len(doc_ids), num_tokens, embed_dim]
        """
        import h5py

        num_docs, num_tokens, _ = embeddings.shape

        # Pad if needed
        if num_tokens < self.max_tokens:
            padding = torch.zeros(
                num_docs,
                self.max_tokens - num_tokens,
                self.embed_dim,
                dtype=embeddings.dtype,
                device=embeddings.device
            )
            embeddings = torch.cat([embeddings, padding], dim=1)

        # Update lengths
        for i, doc_id in enumerate(doc_ids):
            self.doc_lengths[doc_id] = num_tokens

        # Write to HDF5
        with h5py.File(self.data_file, 'a') as f:
            dataset = f['embeddings']
            for i, doc_id in enumerate(doc_ids):
                dataset[doc_id] = embeddings[i].cpu().numpy().astype(np.float16)

        self._save_metadata()
        logger.info(f"Updated {len(doc_ids)} documents")

    def delete_documents(self, doc_ids: List[int]) -> None:
        """
        Mark documents as deleted (soft delete).

        Args:
            doc_ids: Document IDs to delete
        """
        for doc_id in doc_ids:
            if doc_id not in self.deleted_ids:
                self.deleted_ids.add(doc_id)
                self.num_docs -= 1

        self._save_metadata()
        logger.info(f"Deleted {len(doc_ids)} documents (soft delete)")

    def create_collection(self, name: str, doc_ids: List[int]) -> None:
        """
        Create a new collection.

        Args:
            name: Collection name
            doc_ids: Document IDs to include
        """
        self.collections[name] = doc_ids
        self._save_metadata()
        logger.info(f"Created collection '{name}' with {len(doc_ids)} documents")

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.

        Args:
            name: Collection name to delete
        """
        if name in self.collections:
            del self.collections[name]
            self._save_metadata()
            logger.info(f"Deleted collection '{name}'")

    def compact(self) -> None:
        """
        Compact storage by removing deleted documents (hard delete).

        This creates a new HDF5 file with only active documents.
        """
        import h5py

        logger.info("Starting compaction...")

        # Get active document IDs
        active_ids = [
            doc_id for doc_id in range(self.next_doc_id)
            if doc_id not in self.deleted_ids
        ]

        if not active_ids:
            logger.warning("No active documents to compact")
            return

        # Create mapping from old IDs to new IDs
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(active_ids)}

        # Load active documents
        with h5py.File(self.data_file, 'r') as f:
            old_dataset = f['embeddings']
            active_embeddings = old_dataset[active_ids]

        # Create new file
        temp_file = self.data_file.with_suffix('.tmp.h5')

        with h5py.File(temp_file, 'w') as f:
            chunk_size = min(self.config.batch_size, len(active_ids))
            f.create_dataset(
                'embeddings',
                data=active_embeddings,
                chunks=(chunk_size, self.max_tokens, self.embed_dim),
                compression=self.config.compression,
                compression_opts=self.config.compression_level
            )

        # Replace old file
        self.data_file.unlink()
        temp_file.rename(self.data_file)

        # Update metadata
        self.doc_lengths = {old_to_new[old_id]: length for old_id, length in self.doc_lengths.items() if old_id in old_to_new}
        self.doc_metadata = {old_to_new[old_id]: meta for old_id, meta in self.doc_metadata.items() if old_id in old_to_new}
        self.collections = {
            name: [old_to_new[doc_id] for doc_id in doc_ids if doc_id in old_to_new]
            for name, doc_ids in self.collections.items()
        }
        self.deleted_ids = set()
        self.next_doc_id = len(active_ids)
        self.num_docs = len(active_ids)

        self._save_metadata()
        logger.info(f"Compaction complete: {len(active_ids)} active documents")

    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("AsyncPipelineStorage cleanup complete")


__all__ = ['AsyncPipelineStorage', 'PinnedMemoryPool', 'AsyncWriter']

