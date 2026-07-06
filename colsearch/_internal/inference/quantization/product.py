"""
Product Quantization for Embeddings

Divides embedding dimensions into subspaces and learns codebooks
for each subspace. Provides high compression with good accuracy.

Use case: Memory-efficient storage for large indexes.

Reference: arXiv:2405.12497 (Product Quantization for Similarity Search)

Author: Latence Team
License: Apache-2.0
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PQCodebooks:
    """Learned codebooks for product quantization."""
    codebooks: np.ndarray  # (M, K, D/M) - M subspaces, K centroids each
    num_subspaces: int     # M
    num_centroids: int     # K
    dim_per_subspace: int  # D/M


def train_codebooks(
    embeddings: np.ndarray,
    num_subspaces: int = 8,
    num_centroids: int = 256,
    num_iterations: int = 20,
    sample_size: Optional[int] = None,
) -> PQCodebooks:
    """
    Train product quantization codebooks using k-means.

    Args:
        embeddings: Training embeddings (N, D)
        num_subspaces: Number of subspaces (M)
        num_centroids: Centroids per subspace (K)
        num_iterations: K-means iterations
        sample_size: Use subset for training (None = all)

    Returns:
        PQCodebooks with learned centroids
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    N, D = embeddings.shape

    if D % num_subspaces != 0:
        raise ValueError(f"Dimension {D} must be divisible by num_subspaces {num_subspaces}")

    dim_per_subspace = D // num_subspaces

    # Sample if too large
    if sample_size and N > sample_size:
        indices = np.random.choice(N, sample_size, replace=False)
        embeddings = embeddings[indices]

    codebooks = np.zeros((num_subspaces, num_centroids, dim_per_subspace), dtype=np.float32)

    logger.info(f"Training PQ: {num_subspaces} subspaces x {num_centroids} centroids")

    for m in range(num_subspaces):
        # Extract subspace
        start = m * dim_per_subspace
        end = start + dim_per_subspace
        subvectors = embeddings[:, start:end]

        # K-means clustering
        centroids = _kmeans(subvectors, num_centroids, num_iterations)
        codebooks[m] = centroids

    return PQCodebooks(
        codebooks=codebooks,
        num_subspaces=num_subspaces,
        num_centroids=num_centroids,
        dim_per_subspace=dim_per_subspace,
    )


def _kmeans(
    data: np.ndarray,
    k: int,
    iterations: int,
) -> np.ndarray:
    """Simple k-means clustering."""
    N, D = data.shape

    # Initialize centroids randomly
    indices = np.random.choice(N, k, replace=False)
    centroids = data[indices].copy()

    for _ in range(iterations):
        # Assign points to nearest centroid
        distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)

        # Update centroids
        for j in range(k):
            mask = assignments == j
            if np.any(mask):
                centroids[j] = data[mask].mean(axis=0)

    return centroids


def pq_encode(
    embeddings: np.ndarray,
    codebooks: PQCodebooks,
) -> np.ndarray:
    """
    Encode embeddings using product quantization.

    Args:
        embeddings: Float embeddings (N, D)
        codebooks: Trained codebooks

    Returns:
        PQ codes (N, M) as uint8
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    N = embeddings.shape[0]
    M = codebooks.num_subspaces
    d = codebooks.dim_per_subspace

    codes = np.zeros((N, M), dtype=np.uint8)

    for m in range(M):
        # Extract subspace
        start = m * d
        end = start + d
        subvectors = embeddings[:, start:end]

        # Find nearest centroid
        centroids = codebooks.codebooks[m]  # (K, d)
        distances = np.sum((subvectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        codes[:, m] = np.argmin(distances, axis=1).astype(np.uint8)

    return codes


def pq_decode(
    codes: np.ndarray,
    codebooks: PQCodebooks,
) -> np.ndarray:
    """
    Decode PQ codes back to approximate embeddings.

    Args:
        codes: PQ codes (N, M)
        codebooks: Trained codebooks

    Returns:
        Reconstructed embeddings (N, D)
    """
    N = codes.shape[0]
    M = codebooks.num_subspaces
    d = codebooks.dim_per_subspace
    D = M * d

    embeddings = np.zeros((N, D), dtype=np.float32)

    for m in range(M):
        start = m * d
        end = start + d
        embeddings[:, start:end] = codebooks.codebooks[m, codes[:, m]]

    return embeddings


def pq_distance_table(
    query: np.ndarray,
    codebooks: PQCodebooks,
) -> np.ndarray:
    """
    Precompute distance table for fast search.

    Args:
        query: Query embedding (D,)
        codebooks: Trained codebooks

    Returns:
        Distance table (M, K) - squared distances to each centroid
    """
    M = codebooks.num_subspaces
    K = codebooks.num_centroids
    d = codebooks.dim_per_subspace

    table = np.zeros((M, K), dtype=np.float32)

    for m in range(M):
        start = m * d
        end = start + d
        subquery = query[start:end]

        centroids = codebooks.codebooks[m]  # (K, d)
        table[m] = np.sum((subquery - centroids) ** 2, axis=1)

    return table


def pq_search(
    query: np.ndarray,
    codes: np.ndarray,
    codebooks: PQCodebooks,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search PQ index using asymmetric distance computation.

    Args:
        query: Query embedding (D,)
        codes: PQ codes (N, M)
        codebooks: Trained codebooks
        top_k: Number of results

    Returns:
        Tuple of (indices, distances)
    """
    # Precompute distance table
    table = pq_distance_table(query, codebooks)

    # Compute distances using lookup
    N, M = codes.shape
    distances = np.zeros(N, dtype=np.float32)

    for m in range(M):
        distances += table[m, codes[:, m]]

    # Get top-k (smallest distances)
    if N <= top_k:
        indices = np.argsort(distances)
    else:
        indices = np.argpartition(distances, top_k)[:top_k]
        sorted_order = np.argsort(distances[indices])
        indices = indices[sorted_order]

    return indices, distances[indices]


class ProductQuantizer:
    """
    Product quantizer for embedding storage and search.

    Provides memory-efficient storage with configurable accuracy.
    Compression ratio: D * 4 bytes -> M bytes (e.g., 128x4 -> 8 = 64x)

    Example:
        >>> pq = ProductQuantizer(num_subspaces=8, num_centroids=256)
        >>>
        >>> # Train on embeddings
        >>> embeddings = np.random.randn(10000, 128).astype(np.float32)
        >>> pq.fit(embeddings)
        >>>
        >>> # Encode and search
        >>> pq.add(embeddings)
        >>> indices, scores = pq.search(query, top_k=100)
    """

    def __init__(
        self,
        num_subspaces: int = 8,
        num_centroids: int = 256,
        normalize: bool = True,
    ):
        """
        Initialize product quantizer.

        Args:
            num_subspaces: Number of subspaces (M)
            num_centroids: Centroids per subspace (K)
            normalize: L2-normalize embeddings
        """
        self.num_subspaces = num_subspaces
        self.num_centroids = num_centroids
        self.normalize = normalize

        self._codebooks: Optional[PQCodebooks] = None
        self._codes: Optional[np.ndarray] = None
        self._dim: Optional[int] = None

    def fit(
        self,
        embeddings: np.ndarray,
        num_iterations: int = 20,
        sample_size: Optional[int] = 50000,
    ):
        """
        Train codebooks from embeddings.

        Args:
            embeddings: Training embeddings (N, D)
            num_iterations: K-means iterations
            sample_size: Use subset for training
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self._dim = embeddings.shape[1]

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        self._codebooks = train_codebooks(
            embeddings,
            num_subspaces=self.num_subspaces,
            num_centroids=self.num_centroids,
            num_iterations=num_iterations,
            sample_size=sample_size,
        )

        logger.info(f"Trained PQ codebooks: {self.num_subspaces}x{self.num_centroids}")

    def add(self, embeddings: np.ndarray):
        """
        Encode and add embeddings to index.

        Args:
            embeddings: Embeddings to add (N, D)
        """
        if self._codebooks is None:
            raise RuntimeError("Codebooks not trained. Call fit() first.")

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        codes = pq_encode(embeddings, self._codebooks)

        if self._codes is None:
            self._codes = codes
        else:
            self._codes = np.vstack([self._codes, codes])

        logger.info(f"Added {len(embeddings)} vectors, total: {len(self._codes)}")

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.

        Args:
            query: Query embedding (D,)
            top_k: Number of results

        Returns:
            Tuple of (indices, scores)
        """
        if self._codebooks is None or self._codes is None:
            raise RuntimeError("Index not built. Call fit() and add() first.")

        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()

        if self.normalize:
            query = query / (np.linalg.norm(query) + 1e-8)

        indices, distances = pq_search(query, self._codes, self._codebooks, top_k)

        # Convert distance to similarity (higher = better)
        scores = 1.0 / (1.0 + distances)

        return indices, scores

    def reconstruct(self, indices: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate embeddings from codes.

        Args:
            indices: Vector indices

        Returns:
            Reconstructed embeddings
        """
        if self._codebooks is None or self._codes is None:
            raise RuntimeError("Index not built")

        return pq_decode(self._codes[indices], self._codebooks)

    @property
    def memory_usage_bytes(self) -> int:
        """Get memory usage of the PQ index."""
        usage = 0
        if self._codebooks is not None:
            usage += self._codebooks.codebooks.nbytes
        if self._codes is not None:
            usage += self._codes.nbytes
        return usage

    def save(self, path: str):
        """Save quantizer to disk."""
        np.savez(
            path,
            codebooks=self._codebooks.codebooks if self._codebooks else None,
            num_subspaces=self.num_subspaces,
            num_centroids=self.num_centroids,
            codes=self._codes,
            dim=self._dim,
            normalize=self.normalize,
        )

    @classmethod
    def load(cls, path: str) -> 'ProductQuantizer':
        """Load quantizer from disk."""
        data = np.load(path, allow_pickle=False)

        pq = cls(
            num_subspaces=int(data['num_subspaces']),
            num_centroids=int(data['num_centroids']),
            normalize=bool(data['normalize']),
        )

        if data['codebooks'] is not None:
            codebooks_array = data['codebooks']
            pq._codebooks = PQCodebooks(
                codebooks=codebooks_array,
                num_subspaces=int(data['num_subspaces']),
                num_centroids=int(data['num_centroids']),
                dim_per_subspace=codebooks_array.shape[2],
            )

        if 'codes' in data and data['codes'] is not None:
            pq._codes = data['codes']

        pq._dim = int(data['dim']) if data['dim'] is not None else None

        return pq


__all__ = [
    'ProductQuantizer',
    'PQCodebooks',
    'train_codebooks',
    'pq_encode',
    'pq_decode',
    'pq_search',
    'pq_distance_table',
]

