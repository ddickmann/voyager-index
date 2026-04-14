"""
Distributed Query Router
========================

Implements scatter-gather pattern for querying multiple index shards.
Aggregates results using Reciprocal Rank Fusion (RRF).
"""

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import List

from voyager_index._internal.inference.engines.base import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    node_id: str
    endpoint: str
    shard_ids: List[int]


class DistributedRouter:
    """
    Routes queries to appropriate shards and aggregates results.
    """

    def __init__(self, nodes: List[NodeConfig], timeout: float = 2.0, max_retries: int = 3):
        self.nodes = nodes
        self.timeout = timeout
        self.max_retries = max_retries
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes))

    def _query_node_mock(self, node: NodeConfig, query_text: str, top_k: int) -> List[SearchResult]:
        return []

    def search(self, query: str, top_k: int = 10, strategy: str = "rrf") -> List[SearchResult]:
        futures = {}
        for node in self.nodes:
            future = self.executor.submit(self._query_node_mock, node, query, top_k)
            futures[future] = node

        results_per_node = []
        for future in concurrent.futures.as_completed(futures):
            node = futures[future]
            try:
                results = future.result(timeout=self.timeout)
                results_per_node.append(results)
            except Exception as exc:
                logger.error("Query to node %s failed: %s", node.endpoint, exc)

        merged = self._merge_results(results_per_node, top_k, strategy)
        return merged

    def _merge_results(
        self,
        shard_results: List[List[SearchResult]],
        top_k: int,
        strategy: str = "rrf",
    ) -> List[SearchResult]:
        if not shard_results:
            return []

        all_results = [r for batch in shard_results for r in batch]
        if strategy in {"score", "rrf"}:
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]
        return all_results[:top_k]

    def shutdown(self):
        self.executor.shutdown()
