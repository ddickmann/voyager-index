"""
Embedding Quantization Module

Implements various quantization techniques for memory-efficient
embedding storage and fast approximate search.

Quantization methods:
1. Binary Quantization: 32x memory reduction, fastest search
2. Product Quantization: Learned codebooks for high accuracy
3. Scalar Quantization: Simple per-dimension quantization
4. Rotational Quantization (RoQ): Learned rotation for optimal quantization

References:
- Binary with Rotation: arXiv:2409.09913
- Matryoshka (MRL): arXiv:2405.12497

Author: Latence Team
License: Apache-2.0
"""

from .binary import (
    BinaryQuantizer,
    binary_quantize,
    binary_search,
    hamming_distance,
)
from .product import (
    ProductQuantizer,
    pq_decode,
    pq_encode,
    train_codebooks,
)
from .rotational import (
    FastWalshHadamard,
    RoQConfig,
    RotationalQuantizer,
)
from .scalar import (
    ScalarQuantizer,
    int8_dequantize,
    int8_quantize,
)

__all__ = [
    # Binary
    'BinaryQuantizer',
    'binary_quantize',
    'binary_search',
    'hamming_distance',
    # Product
    'ProductQuantizer',
    'train_codebooks',
    'pq_encode',
    'pq_decode',
    # Scalar
    'ScalarQuantizer',
    'int8_quantize',
    'int8_dequantize',
    # Rotational (RoQ)
    'RotationalQuantizer',
    'RoQConfig',
    'FastWalshHadamard',
]

