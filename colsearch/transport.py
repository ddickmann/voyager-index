"""
Public helpers for base64 vector transport payloads.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from colsearch._internal.inference.stateless_optimizer import (
    VectorPayload,
)
from colsearch._internal.inference.stateless_optimizer import (
    decode_vector_payload as _decode_vector_payload,
)
from colsearch._internal.inference.stateless_optimizer import (
    encode_float_vectors as _encode_float_vectors,
)
from colsearch._internal.inference.stateless_optimizer import (
    encode_roq_vectors as _encode_roq_vectors,
)


def encode_vector_payload(vectors: Any, *, dtype: str = "float32") -> Dict[str, Any]:
    """Encode vectors into a JSON-ready base64 payload."""
    return _encode_float_vectors(vectors, dtype=dtype).to_dict()


def encode_roq_payload(
    vectors: Any,
    *,
    num_bits: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Encode vectors into a JSON-ready RoQ payload."""
    return _encode_roq_vectors(vectors, num_bits=num_bits, seed=seed).to_dict()


def decode_payload(payload: VectorPayload | Dict[str, Any]) -> np.ndarray:
    """Decode a vector payload back into a float32 numpy array."""
    return _decode_vector_payload(payload)


__all__ = [
    "VectorPayload",
    "decode_payload",
    "encode_roq_payload",
    "encode_vector_payload",
]
