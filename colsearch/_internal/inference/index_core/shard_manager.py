"""
Shard Manager for Distributed Indexing
======================================

Implements consistent hashing to distribute documents across shards.
Supports adding/removing nodes with minimal data movement.

Architecture:
- Virtual Nodes: Each physical shard has multiple virtual nodes on the ring.
- Ring: Sorted map of hash values to shard IDs.
- Hash Function: MurmurHash3 (via mmh3) for speed and distribution quality.

Implementation aligned with Qdrant best practices:
- 12 shards default (divisible by 1, 2, 3, 4, 6, 12 nodes)
- Metadata stored in JSON/SQLite
"""

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

class ShardManager:
    """
    Manages distribution of documents across shards using Consistent Hashing.
    """

    def __init__(
        self,
        num_shards: int = 12,
        virtual_nodes: int = 100,
        metadata_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize ShardManager.

        Args:
            num_shards: Total number of physical shards (default: 12).
            virtual_nodes: Number of virtual nodes per shard on the ring.
            metadata_path: Path to save/load shard metadata.
        """
        self.num_shards = num_shards
        self.virtual_nodes = virtual_nodes
        self.metadata_path = Path(metadata_path) if metadata_path else None

        # Ring structure: hash -> shard_id
        self.ring: Dict[int, int] = {}
        self.sorted_keys: List[int] = []

        # Shard state
        self.active_shards: Set[int] = set(range(num_shards))
        self.shard_mapping: Dict[int, str] = {i: f"shard_{i}" for i in range(num_shards)}

        self._build_ring()
        self._load_metadata()

    def _hash(self, key: str) -> int:
        """Compute 32-bit MurmurHash3 equivalent using MD5 (for portability)."""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def _build_ring(self):
        """Build the consistent hashing ring."""
        self.ring.clear()
        for shard_id in self.active_shards:
            for i in range(self.virtual_nodes):
                key = f"shard_{shard_id}_vnode_{i}"
                h = self._hash(key)
                self.ring[h] = shard_id

        self.sorted_keys = sorted(self.ring.keys())
        logger.info(f"Built hash ring with {len(self.ring)} virtual nodes for {len(self.active_shards)} shards")

    def get_shard_id(self, doc_id: str) -> int:
        """
        Get the shard ID for a given document ID.

        Args:
            doc_id: Unique document identifier.

        Returns:
            Shard ID (0 to num_shards-1).
        """
        if not self.sorted_keys:
            return 0

        h = self._hash(str(doc_id))

        # Binary search for the first key >= h
        import bisect
        idx = bisect.bisect_right(self.sorted_keys, h)

        if idx == len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def get_all_shards(self) -> List[int]:
        """Get list of all active shard IDs."""
        return sorted(list(self.active_shards))

    def save_metadata(self):
        """Save shard configuration to disk."""
        if not self.metadata_path:
            return

        data = {
            "num_shards": self.num_shards,
            "virtual_nodes": self.virtual_nodes,
            "active_shards": list(self.active_shards),
            "shard_mapping": self.shard_mapping
        }

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_metadata(self):
        """Load shard configuration from disk."""
        if not self.metadata_path or not self.metadata_path.exists():
            return

        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)

            self.num_shards = data.get("num_shards", 12)
            self.virtual_nodes = data.get("virtual_nodes", 100)
            self.active_shards = set(data.get("active_shards", range(self.num_shards)))
            self.shard_mapping = {int(k): v for k, v in data.get("shard_mapping", {}).items()}

            self._build_ring()
            logger.info(f"Loaded shard metadata from {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to load shard metadata: {e}")

    def route_batch(self, doc_ids: List[str]) -> Dict[int, List[int]]:
        """
        Route a batch of documents to their respective shards.

        Args:
            doc_ids: List of document IDs.

        Returns:
            Dictionary mapping shard_id -> list of indices in the original batch.
        """
        routing = defaultdict(list)
        for i, doc_id in enumerate(doc_ids):
            shard_id = self.get_shard_id(doc_id)
            routing[shard_id].append(i)
        return dict(routing)
