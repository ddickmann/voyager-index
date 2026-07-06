"""
Scalar Quantization for Embeddings

Simple per-dimension quantization to INT8 or INT4.
Provides good balance of compression and accuracy.

Author: Latence Team
License: Apache-2.0
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def int8_quantize(
    embeddings: np.ndarray,
    per_dim: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize float embeddings to INT8.

    Uses symmetric quantization with per-dimension or per-tensor scales.

    Args:
        embeddings: Float embeddings (N, D)
        per_dim: Use per-dimension scales (more accurate) vs per-tensor

    Returns:
        Tuple of (quantized, scales, zero_points)
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    embeddings = embeddings.astype(np.float32)

    if per_dim:
        # Per-dimension scaling
        scales = embeddings.max(axis=0) - embeddings.min(axis=0)
        scales = np.where(scales == 0, 1.0, scales)
        scales = scales / 255.0  # Map to [0, 255]

        zero_points = embeddings.min(axis=0)

        quantized = ((embeddings - zero_points) / scales).round().clip(0, 255).astype(np.uint8)
    else:
        # Per-tensor scaling
        scale = (embeddings.max() - embeddings.min()) / 255.0
        if scale == 0:
            scale = 1.0

        zero_point = embeddings.min()

        quantized = ((embeddings - zero_point) / scale).round().clip(0, 255).astype(np.uint8)
        scales = np.array([scale])
        zero_points = np.array([zero_point])

    return quantized, scales, zero_points


def int8_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
) -> np.ndarray:
    """
    Dequantize INT8 embeddings back to float32.

    Args:
        quantized: INT8 embeddings (N, D)
        scales: Scale factors
        zero_points: Zero points

    Returns:
        Float32 embeddings (N, D)
    """
    quantized = quantized.astype(np.float32)

    if scales.shape[0] == 1:
        # Per-tensor
        return quantized * scales[0] + zero_points[0]
    else:
        # Per-dimension
        return quantized * scales + zero_points


def int4_quantize(
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize float embeddings to INT4.

    Packs two INT4 values into one byte for 2x compression vs INT8.

    Args:
        embeddings: Float embeddings (N, D)

    Returns:
        Tuple of (quantized, scales, zero_points)
        quantized shape is (N, D//2) with packed values
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    embeddings = embeddings.astype(np.float32)
    N, D = embeddings.shape

    # Per-dimension scaling to [0, 15]
    mins = embeddings.min(axis=0)
    maxs = embeddings.max(axis=0)
    scales = (maxs - mins) / 15.0
    scales = np.where(scales == 0, 1.0, scales)

    # Quantize to 4 bits
    q4 = ((embeddings - mins) / scales).round().clip(0, 15).astype(np.uint8)

    # Pack two values per byte
    if D % 2 != 0:
        # Pad to even
        q4 = np.pad(q4, ((0, 0), (0, 1)), mode='constant', constant_values=0)

    packed = (q4[:, 0::2] << 4) | q4[:, 1::2]

    return packed, scales, mins


def int4_dequantize(
    packed: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    original_dim: int,
) -> np.ndarray:
    """
    Dequantize INT4 embeddings back to float32.

    Args:
        packed: Packed INT4 values (N, D//2)
        scales: Scale factors (D,)
        zero_points: Zero points (D,)
        original_dim: Original dimension

    Returns:
        Float32 embeddings (N, D)
    """
    N = packed.shape[0]

    # Unpack
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F

    # Interleave
    unpacked = np.zeros((N, high.shape[1] * 2), dtype=np.float32)
    unpacked[:, 0::2] = high
    unpacked[:, 1::2] = low

    # Trim to original dimension
    unpacked = unpacked[:, :original_dim]

    # Dequantize
    return unpacked * scales + zero_points


class ScalarQuantizer:
    """
    Scalar quantizer for embedding compression.

    Supports INT8 (4x compression) and INT4 (8x compression).

    Example:
        >>> sq = ScalarQuantizer(bits=8)
        >>>
        >>> # Quantize embeddings
        >>> embeddings = np.random.randn(1000, 128).astype(np.float32)
        >>> sq.fit(embeddings)
        >>>
        >>> # Encode
        >>> codes = sq.encode(embeddings)
        >>>
        >>> # Decode
        >>> reconstructed = sq.decode(codes)
    """

    def __init__(
        self,
        bits: int = 8,
        per_dim: bool = True,
    ):
        """
        Initialize scalar quantizer.

        Args:
            bits: Quantization bits (8 or 4)
            per_dim: Use per-dimension scales
        """
        if bits not in [4, 8]:
            raise ValueError("bits must be 4 or 8")

        self.bits = bits
        self.per_dim = per_dim

        self._scales: Optional[np.ndarray] = None
        self._zero_points: Optional[np.ndarray] = None
        self._dim: Optional[int] = None

    def fit(self, embeddings: np.ndarray):
        """
        Compute quantization parameters from data.

        Args:
            embeddings: Training embeddings (N, D)
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self._dim = embeddings.shape[1]

        if self.bits == 8:
            _, self._scales, self._zero_points = int8_quantize(embeddings, self.per_dim)
        else:
            _, self._scales, self._zero_points = int4_quantize(embeddings)

        logger.info(f"Fitted ScalarQuantizer: {self.bits}-bit, dim={self._dim}")

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode embeddings to quantized format.

        Args:
            embeddings: Float embeddings (N, D)

        Returns:
            Quantized embeddings
        """
        if self._scales is None:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        if self.bits == 8:
            if self.per_dim:
                quantized = ((embeddings - self._zero_points) / self._scales).round().clip(0, 255).astype(np.uint8)
            else:
                quantized = ((embeddings - self._zero_points[0]) / self._scales[0]).round().clip(0, 255).astype(np.uint8)
        else:
            q4 = ((embeddings - self._zero_points) / self._scales).round().clip(0, 15).astype(np.uint8)
            if embeddings.shape[1] % 2 != 0:
                q4 = np.pad(q4, ((0, 0), (0, 1)), mode='constant', constant_values=0)
            quantized = (q4[:, 0::2] << 4) | q4[:, 1::2]

        return quantized

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        """
        Decode quantized embeddings back to float.

        Args:
            quantized: Quantized embeddings

        Returns:
            Float32 embeddings
        """
        if self._scales is None:
            raise RuntimeError("Quantizer not fitted.")

        if self.bits == 8:
            return int8_dequantize(quantized, self._scales, self._zero_points)
        else:
            return int4_dequantize(quantized, self._scales, self._zero_points, self._dim)

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio vs float32."""
        if self.bits == 8:
            return 4.0  # 32 / 8
        else:
            return 8.0  # 32 / 4

    def save(self, path: str):
        """Save quantizer to disk."""
        np.savez(
            path,
            scales=self._scales,
            zero_points=self._zero_points,
            dim=self._dim,
            bits=self.bits,
            per_dim=self.per_dim,
        )

    @classmethod
    def load(cls, path: str) -> 'ScalarQuantizer':
        """Load quantizer from disk."""
        data = np.load(path, allow_pickle=False)

        sq = cls(bits=int(data['bits']), per_dim=bool(data['per_dim']))
        sq._scales = data['scales']
        sq._zero_points = data['zero_points']
        sq._dim = int(data['dim'])

        return sq


__all__ = [
    'ScalarQuantizer',
    'int8_quantize',
    'int8_dequantize',
    'int4_quantize',
    'int4_dequantize',
]

