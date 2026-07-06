"""
Binary Quantization for Embeddings

Reduces 32-bit float embeddings to 1-bit binary vectors.
Achieves 32x memory reduction with fast Hamming distance search.

Use case: First-stage retrieval where speed is critical.
Follow with full-precision reranking for accuracy.

Reference: arXiv:2409.09913 (Binary Embedding for Retrieval)

Author: Latence Team
License: Apache-2.0
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def binary_quantize(embeddings: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Quantize float embeddings to binary (sign bits).

    Each dimension becomes 1 bit: positive -> 1, negative -> 0.
    Resulting binary vectors are packed into uint8 arrays.

    Args:
        embeddings: Float embeddings (N, D)

    Returns:
        Binary embeddings packed as uint8 (N, D//8)

    Example:
        >>> emb = np.random.randn(1000, 128)
        >>> binary = binary_quantize(emb)
        >>> print(binary.shape)  # (1000, 16)
        >>> print(emb.nbytes / binary.nbytes)  # 32x reduction
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Get sign bits (1 if positive, 0 if negative)
    bits = (embeddings > 0).astype(np.uint8)

    # Pack bits into bytes
    N, D = bits.shape

    # Pad dimension to multiple of 8
    if D % 8 != 0:
        pad_size = 8 - (D % 8)
        bits = np.pad(bits, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
        D = bits.shape[1]

    # Reshape and pack
    bits_reshaped = bits.reshape(N, D // 8, 8)

    # Pack 8 bits into 1 byte
    packed = np.packbits(bits_reshaped, axis=2).squeeze(axis=2)

    return packed


def binary_dequantize(
    binary: np.ndarray,
    original_dim: int,
) -> np.ndarray:
    """
    Expand binary embeddings back to float (for compatibility).

    Note: This loses precision. Use original embeddings for reranking.

    Args:
        binary: Packed binary embeddings (N, D//8)
        original_dim: Original embedding dimension

    Returns:
        Float embeddings (N, original_dim) with values -1 or 1
    """
    binary.shape[0]

    # Unpack bits
    bits = np.unpackbits(binary, axis=1)[:, :original_dim]

    # Convert 0/1 to -1/+1
    float_emb = bits.astype(np.float32) * 2 - 1

    return float_emb


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute Hamming distance between binary vectors.

    Uses population count (popcount) for efficiency.

    Args:
        a: Binary vector(s) (N, D//8) or (D//8,)
        b: Binary vector(s) (M, D//8) or (D//8,)

    Returns:
        Hamming distances (N, M) or (N,) or scalar
    """
    # XOR gives bits that differ
    xor = np.bitwise_xor(a[:, None, :] if a.ndim == 2 else a[None, None, :],
                         b[None, :, :] if b.ndim == 2 else b[None, :])

    # Count differing bits
    # Use lookup table for popcount
    _popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    distances = _popcount_table[xor].sum(axis=-1)

    return distances


def binary_search(
    query: np.ndarray,
    index: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast approximate nearest neighbor search using Hamming distance.

    Args:
        query: Binary query (D//8,) or (1, D//8)
        index: Binary index (N, D//8)
        top_k: Number of results

    Returns:
        Tuple of (indices, distances)
    """
    if query.ndim == 1:
        query = query[None, :]

    # Compute Hamming distances
    distances = hamming_distance(query, index).squeeze()

    # Get top-k (smallest distances)
    if len(distances) <= top_k:
        indices = np.argsort(distances)
    else:
        indices = np.argpartition(distances, top_k)[:top_k]
        # Sort the top-k
        sorted_order = np.argsort(distances[indices])
        indices = indices[sorted_order]

    return indices, distances[indices]


class BinaryQuantizer:
    """
    Binary quantizer for embedding storage and search.

    Provides 32x memory reduction compared to float32 embeddings.
    Best used for first-stage retrieval with full-precision reranking.

    Example:
        >>> quantizer = BinaryQuantizer()
        >>>
        >>> # Index documents
        >>> embeddings = np.random.randn(10000, 128).astype(np.float32)
        >>> quantizer.fit(embeddings)
        >>>
        >>> # Search
        >>> query = np.random.randn(128).astype(np.float32)
        >>> indices, scores = quantizer.search(query, top_k=100)
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize binary quantizer.

        Args:
            normalize: L2-normalize embeddings before quantization
        """
        self.normalize = normalize
        self._index: Optional[np.ndarray] = None
        self._original_embeddings: Optional[np.ndarray] = None
        self._dim: Optional[int] = None

    def fit(
        self,
        embeddings: np.ndarray,
        store_original: bool = False,
    ):
        """
        Fit quantizer and build binary index.

        Args:
            embeddings: Float embeddings (N, D)
            store_original: Keep original embeddings for reranking
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self._dim = embeddings.shape[1]

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        self._index = binary_quantize(embeddings)

        if store_original:
            self._original_embeddings = embeddings.copy()

        logger.info(f"Built binary index: {len(embeddings)} vectors, {self._index.nbytes / 1024:.1f} KB")

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors using Hamming distance.

        Args:
            query: Query embedding (D,)
            top_k: Number of results

        Returns:
            Tuple of (indices, distances)
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call fit() first.")

        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()

        if self.normalize:
            query = query / (np.linalg.norm(query) + 1e-8)

        query_binary = binary_quantize(query.reshape(1, -1))

        indices, distances = binary_search(query_binary, self._index, top_k)

        # Convert Hamming distance to similarity score
        max_hamming = self._dim  # Max possible Hamming distance
        scores = 1.0 - (distances.astype(np.float32) / max_hamming)

        return indices, scores

    def rerank(
        self,
        query: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rerank candidates using full-precision embeddings.

        Args:
            query: Query embedding (D,)
            candidate_indices: Candidate indices to rerank
            top_k: Number of final results

        Returns:
            Tuple of (reranked_indices, scores)
        """
        if self._original_embeddings is None:
            raise RuntimeError("Original embeddings not stored. Use store_original=True in fit().")

        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()

        if self.normalize:
            query = query / (np.linalg.norm(query) + 1e-8)

        # Get candidate embeddings
        candidates = self._original_embeddings[candidate_indices]

        # Compute cosine similarity
        scores = candidates @ query

        # Sort by score
        sorted_order = np.argsort(scores)[::-1]

        if top_k:
            sorted_order = sorted_order[:top_k]

        return candidate_indices[sorted_order], scores[sorted_order]

    @property
    def memory_usage_bytes(self) -> int:
        """Get memory usage of the binary index."""
        if self._index is None:
            return 0
        return self._index.nbytes

    def save(self, path: str):
        """Save quantizer to disk."""
        np.savez(
            path,
            index=self._index,
            dim=self._dim,
            normalize=self.normalize,
            original=self._original_embeddings,
        )

    @classmethod
    def load(cls, path: str) -> 'BinaryQuantizer':
        """Load quantizer from disk."""
        data = np.load(path, allow_pickle=False)

        quantizer = cls(normalize=bool(data['normalize']))
        quantizer._index = data['index']
        quantizer._dim = int(data['dim'])

        if 'original' in data and data['original'] is not None:
            quantizer._original_embeddings = data['original']

        return quantizer


__all__ = [
    'BinaryQuantizer',
    'binary_quantize',
    'binary_dequantize',
    'binary_search',
    'hamming_distance',
]

