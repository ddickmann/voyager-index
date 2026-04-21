"""
GPU-first stateless fulfilment optimizer.

This module defines a transport contract for dense and multivector bundles and a
GPU-oriented optimization pipeline that:

1. Accepts base64-encoded float or RoQ payloads.
2. Preserves float payloads end-to-end and accepts pre-quantized RoQ payloads.
3. Builds query-token coverage and centroid redundancy tensors.
4. Calls the knapsack solver with precomputed relevance / fulfilment features.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from colsearch._internal.inference.quantization.rotational import RoQConfig, RotationalQuantizer
from colsearch._internal.kernels.maxsim import compute_maxsim_token_coverage_matrix

logger = logging.getLogger(__name__)
HYBRID_EXECUTION_MODE = "gpu_precompute_cpu_search"
END_TO_END_GPU_EXECUTION_MODE = "end_to_end_gpu_search"
DEFAULT_OPTIMIZER_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


class OptimizerRequestError(ValueError):
    """Raised when the public optimizer request is structurally invalid."""


class OptimizerPayloadTooLargeError(OptimizerRequestError):
    """Raised when an optimizer request exceeds the configured payload budget."""


def default_optimizer_require_gpu() -> bool:
    """When unset, require a CUDA device only if one is available (strict for prod GPU hosts)."""
    v = os.environ.get("VOYAGER_OPTIMIZER_REQUIRE_GPU", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    return bool(torch.cuda.is_available())


def optimizer_max_payload_bytes() -> int:
    raw = os.environ.get("VOYAGER_OPTIMIZER_MAX_PAYLOAD_BYTES", "").strip()
    if not raw:
        return DEFAULT_OPTIMIZER_MAX_PAYLOAD_BYTES
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - misconfiguration guard
        raise RuntimeError("VOYAGER_OPTIMIZER_MAX_PAYLOAD_BYTES must be an integer") from exc
    return max(value, 0)


def _enforce_optimizer_payload_limit(payload_bytes: int) -> None:
    max_payload_bytes = optimizer_max_payload_bytes()
    if max_payload_bytes <= 0:
        return
    if payload_bytes > max_payload_bytes:
        raise OptimizerPayloadTooLargeError(
            "Optimizer request payload exceeds the configured limit: "
            f"{payload_bytes} bytes > {max_payload_bytes} bytes. "
            "Reduce candidate/query vector payloads or raise VOYAGER_OPTIMIZER_MAX_PAYLOAD_BYTES."
        )


def _as_float32_matrix(vectors: Any) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D vector matrix, got shape {array.shape}")
    return array


def _b64encode_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def _b64decode_array(data_b64: str, dtype: np.dtype, shape: Sequence[int]) -> np.ndarray:
    raw = base64.b64decode(data_b64.encode("ascii"))
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise ValueError(
            f"Decoded payload has {len(raw)} bytes, expected {expected} for {tuple(shape)} {dtype}"
        )
    # Make decoded transport payloads writable before Torch wraps them.
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _placeholder_vector_payload(array: np.ndarray) -> "VectorPayload":
    matrix = _as_float32_matrix(array)
    return VectorPayload(
        encoding="float32",
        shape=list(matrix.shape),
        data_b64="",
        dtype="float32",
    )


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-8, None)


def _centroid(array: np.ndarray) -> np.ndarray:
    return _normalize_rows(array).mean(axis=0, keepdims=False).astype(np.float32)


def _deep_merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in dict(overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _default_optimizer_policy() -> Dict[str, Any]:
    return {
        "name": "baseline_v1",
        "query_bundle": {
            "cluster_mix_lexical": 0.55,
            "cluster_mix_cluster": 0.45,
            "coverage_bridge_retrieval": 0.60,
            "coverage_bridge_lexical": 0.40,
            "centroid_base_weight": 0.58,
            "centroid_candidate_weight": 0.42,
            "facet_similarity_threshold": 0.985,
            "max_extra_facets_single": 4,
            "max_extra_facets_multi": 2,
            "facet_top_k": 3,
            "fallback_mix_lexical": 0.45,
            "fallback_mix_retrieval": 0.35,
            "fallback_mix_cluster": 0.20,
            "fallback_base_weight": 0.30,
            "fallback_candidate_weight": 0.70,
            "fallback_diversity_top_k": 8,
            "fallback_similarity_threshold": 0.9995,
        },
        "token_weights": {
            "best_mass_weight": 0.65,
            "margin_weight": 0.35,
            "floor": 1e-3,
            "metadata_blend": 0.60,
        },
        "retrieval": {
            "base": 0.30,
            "rrf": 0.25,
            "dense": 0.15,
            "sparse": 0.10,
            "dense_rr": 0.10,
            "sparse_rr": 0.05,
            "agreement": 0.10,
            "disagreement_penalty": 0.05,
            "rerank": 0.0,
            "rerank_rr": 0.0,
            "rerank_agreement": 0.0,
            "rerank_disagreement_gate": 0.0,
            "utility": 0.0,
            "utility_rr": 0.0,
            "entailment": 0.0,
            "teacher_utility": 0.0,
            "contradiction_penalty": 0.0,
        },
        "payload_signals": {
            "anti_shadow_frontier_weight": 0.60,
            "anti_shadow_uniqueness_weight": 0.40,
            "ambiguity_hard_coverage_weight": 0.50,
            "ambiguity_relevance_weight": 0.50,
            "ambiguity_contradiction_weight": 0.40,
            "support_quorum_floor": 0.50,
            "support_quorum_uniqueness_weight": 0.50,
            "support_quorum_utility_weight": 0.35,
            "support_quorum_contradiction_penalty": 0.35,
            "pack_value_frontier_weight": 0.40,
            "pack_value_quorum_weight": 0.35,
            "pack_value_uniqueness_weight": 0.25,
            "pack_value_utility_weight": 0.25,
        },
        "redundancy": {
            "centroid": 0.55,
            "lexical": 0.15,
            "containment": 0.10,
            "aspect": 0.15,
            "same_source": 0.05,
        },
        "controller": {
            "name": "auto_v1",
            "ambiguity_p90_weight": 0.50,
            "high_ambiguity_threshold": 0.30,
            "large_candidate_count": 96.0,
            "small_candidate_count": 32.0,
            "low_uncovered_threshold": 0.10,
            "high_uncovered_threshold": 0.18,
            "retrieval_agreement_margin": 0.10,
            "metadata_coverage_threshold": 0.20,
            "redundancy_density_threshold": 0.35,
            "slack_ratio_threshold": 0.20,
            "frontier_gain_p90_threshold": 0.25,
            "pack_density_threshold": 0.04,
            "budget_avg_chunk_ratio_threshold": 3.0,
            "high_iterations": 90,
            "low_iterations": 48,
            "very_high_iterations": 110,
            "high_exact_window_time_ms": 25,
            "low_exact_window_time_ms": 12,
            "small_exact_window_size": 12,
            "large_exact_window_size": 16,
            "uncovered_mu_floor": 1.15,
            "slack_mu_floor": 1.22,
            "redundancy_lambda_floor": 0.65,
            "slack_lambda_cap": 0.48,
            "hard_coverage_relevance_boost": 0.06,
            "frontier_gain_fulfilment_boost": 0.08,
            "support_quorum_fulfilment_boost": 0.05,
            "retrieval_relevance_boost": 0.05,
            "geometry_relevance_penalty": 0.05,
            "metadata_retrieval_penalty": 0.10,
            "metadata_geometry_boost": 0.07,
            "metadata_lexical_boost": 0.03,
            "uniqueness_aux_boost": 0.05,
            "pack_value_fulfilment_boost": 0.07,
            "pack_value_aux_boost": 0.06,
            "anti_shadow_aux_boost": 0.03,
            "rerank_coverage_threshold": 0.15,
            "rerank_agreement_threshold": 0.35,
            "post_rerank_iterations_floor": 72,
            "post_rerank_lambda_floor": 0.58,
            "post_rerank_retrieval_boost": 0.10,
            "post_rerank_geometry_penalty": 0.05,
            "post_rerank_pack_value_boost": 0.04,
            "utility_signal_threshold": 0.20,
            "contradiction_signal_threshold": 0.18,
            "utility_relevance_boost": 0.06,
            "utility_support_quorum_boost": 0.05,
            "contradiction_lambda_floor": 0.72,
            "contradiction_iterations_floor": 96,
        },
        "blend_weights": {
            "relevance": {
                "geometry": 0.34,
                "retrieval": 0.18,
                "lexical": 0.12,
                "winner_mass": 0.10,
                "hard_coverage": 0.10,
                "uniqueness": 0.08,
                "support_quorum": 0.08,
            },
            "fulfilment": {
                "fulfilment": 0.34,
                "margin": 0.16,
                "winner_mass": 0.12,
                "frontier_gain": 0.18,
                "support_quorum": 0.12,
                "pack_value": 0.08,
            },
            "auxiliary": {
                "auxiliary": 0.20,
                "lexical": 0.12,
                "specificity": 0.12,
                "cluster": 0.08,
                "uniqueness": 0.15,
                "ambiguity": 0.13,
                "anti_shadow": 0.10,
                "pack_value": 0.10,
            },
        },
        "ranking": {
            "relevance": 0.50,
            "fulfilment": 0.35,
            "auxiliary": 0.15,
        },
    }


def _post_rerank_optimizer_policy() -> Dict[str, Any]:
    return _deep_merge_dict(
        _default_optimizer_policy(),
        {
            "name": "post_rerank_v1",
            "retrieval": {
                "base": 0.42,
                "rrf": 0.14,
                "dense": 0.08,
                "sparse": 0.05,
                "dense_rr": 0.05,
                "sparse_rr": 0.03,
                "agreement": 0.04,
                "disagreement_penalty": 0.02,
                "rerank": 0.28,
                "rerank_rr": 0.08,
                "rerank_agreement": 0.12,
                "rerank_disagreement_gate": 0.85,
                "utility": 0.06,
                "entailment": 0.05,
                "teacher_utility": 0.03,
                "contradiction_penalty": 0.06,
            },
            "controller": {
                "name": "post_rerank_v1",
                "rerank_coverage_threshold": 0.05,
                "rerank_agreement_threshold": 0.20,
                "post_rerank_iterations_floor": 84,
                "post_rerank_lambda_floor": 0.64,
                "post_rerank_retrieval_boost": 0.12,
                "post_rerank_geometry_penalty": 0.06,
                "post_rerank_pack_value_boost": 0.05,
            },
            "blend_weights": {
                "relevance": {
                    "geometry": 0.26,
                    "retrieval": 0.30,
                    "lexical": 0.10,
                    "winner_mass": 0.08,
                    "hard_coverage": 0.10,
                    "uniqueness": 0.07,
                    "support_quorum": 0.09,
                },
                "fulfilment": {
                    "fulfilment": 0.32,
                    "margin": 0.14,
                    "winner_mass": 0.10,
                    "frontier_gain": 0.18,
                    "support_quorum": 0.14,
                    "pack_value": 0.12,
                },
            },
            "ranking": {
                "relevance": 0.55,
                "fulfilment": 0.30,
                "auxiliary": 0.15,
            },
        },
    )


def _frontier_optimizer_policy() -> Dict[str, Any]:
    return _deep_merge_dict(
        _post_rerank_optimizer_policy(),
        {
            "name": "frontier_v1",
            "token_weights": {
                "metadata_blend": 0.78,
            },
            "retrieval": {
                "base": 0.24,
                "rrf": 0.10,
                "dense": 0.08,
                "sparse": 0.04,
                "dense_rr": 0.04,
                "sparse_rr": 0.02,
                "agreement": 0.04,
                "disagreement_penalty": 0.03,
                "rerank": 0.14,
                "rerank_rr": 0.05,
                "rerank_agreement": 0.08,
                "utility": 0.18,
                "utility_rr": 0.06,
                "entailment": 0.10,
                "teacher_utility": 0.12,
                "contradiction_penalty": 0.16,
            },
            "payload_signals": {
                "ambiguity_contradiction_weight": 0.65,
                "support_quorum_utility_weight": 0.55,
                "support_quorum_contradiction_penalty": 0.45,
                "pack_value_utility_weight": 0.40,
            },
            "redundancy": {
                "centroid": 0.40,
                "lexical": 0.18,
                "containment": 0.12,
                "aspect": 0.20,
                "same_source": 0.10,
            },
            "controller": {
                "name": "frontier_v1",
                "post_rerank_iterations_floor": 92,
                "post_rerank_lambda_floor": 0.68,
                "utility_signal_threshold": 0.16,
                "contradiction_signal_threshold": 0.12,
                "utility_relevance_boost": 0.08,
                "utility_support_quorum_boost": 0.07,
                "contradiction_lambda_floor": 0.78,
                "contradiction_iterations_floor": 108,
            },
            "blend_weights": {
                "relevance": {
                    "geometry": 0.24,
                    "retrieval": 0.32,
                    "lexical": 0.10,
                    "winner_mass": 0.06,
                    "hard_coverage": 0.12,
                    "uniqueness": 0.06,
                    "support_quorum": 0.10,
                },
                "fulfilment": {
                    "fulfilment": 0.32,
                    "margin": 0.12,
                    "winner_mass": 0.08,
                    "frontier_gain": 0.18,
                    "support_quorum": 0.18,
                    "pack_value": 0.12,
                },
            },
        },
    )


def _optimizer_policy_presets() -> Dict[str, Dict[str, Any]]:
    return {
        "baseline_v1": _default_optimizer_policy(),
        "post_rerank_v1": _post_rerank_optimizer_policy(),
        "frontier_v1": _frontier_optimizer_policy(),
    }


def _default_edge_case_fallback_config() -> Dict[str, Any]:
    return {
        "name": "high_effort_v1",
        "enabled": True,
        "min_candidate_count": 48,
        "min_risk_signals": 1,
        "thresholds": {
            "uncovered_mass": 0.15,
            "retrieval_disagreement_mean": 0.50,
            "ambiguity_p90": 0.90,
            "metadata_coverage": 0.20,
            "slack_ratio": 0.20,
            "frontier_gain_p90": 0.25,
            "pack_density": 0.04,
        },
        "solver_overrides": {
            "iterations": 160,
            "early_stopping_patience": 32,
            "enable_exact_window": True,
            "exact_window_size": 18,
            "exact_window_time_ms": 60,
            "enable_path_relinking": True,
            "enable_destroy_repair": True,
            "enable_reactive_tenure": True,
        },
        "blend_overrides": {
            "relevance": {
                "retrieval": 0.08,
                "geometry": 0.30,
                "lexical": 0.18,
                "hard_coverage": 0.16,
                "support_quorum": 0.12,
                "uniqueness": 0.10,
                "winner_mass": 0.06,
            },
            "fulfilment": {
                "fulfilment": 0.30,
                "margin": 0.14,
                "winner_mass": 0.08,
                "frontier_gain": 0.20,
                "support_quorum": 0.18,
                "pack_value": 0.10,
            },
        },
    }


@dataclass
class VectorPayload:
    encoding: str
    shape: List[int]
    data_b64: str
    dtype: str = "float32"
    num_bits: Optional[int] = None
    block_size: Optional[int] = None
    num_rounds: Optional[int] = None
    seed: int = 42
    scales_b64: Optional[str] = None
    offsets_b64: Optional[str] = None
    norms_sq_b64: Optional[str] = None
    code_sums_b64: Optional[str] = None
    code_shape: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def approx_bytes(self) -> int:
        fields = [
            self.data_b64,
            self.scales_b64,
            self.offsets_b64,
            self.norms_sq_b64,
            self.code_sums_b64,
        ]
        return sum(len(field or "") for field in fields)


@dataclass
class OptimizerCandidate:
    chunk_id: str
    text: str
    token_count: int
    vectors: VectorPayload
    fact_density: float = 0.5
    centrality_score: float = 0.5
    recency_score: float = 0.5
    auxiliary_score: float = 0.0
    rhetorical_role: str = "unknown"
    cluster_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["vectors"] = self.vectors.to_dict()
        return payload


@dataclass
class OptimizerRequest:
    query_text: str
    query_vectors: VectorPayload
    candidates: List[OptimizerCandidate]
    constraints: Dict[str, Any] = field(default_factory=dict)
    solver_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "query_vectors": self.query_vectors.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "constraints": dict(self.constraints),
            "solver_config": dict(self.solver_config),
            "metadata": dict(self.metadata),
        }

    def approx_bytes(self) -> int:
        return self.query_vectors.approx_bytes() + sum(
            candidate.vectors.approx_bytes() for candidate in self.candidates
        )


def encode_float_vectors(vectors: Any, *, dtype: str = "float16") -> VectorPayload:
    matrix = _as_float32_matrix(vectors)
    if dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported float payload dtype: {dtype}")
    array = matrix.astype(np.float16 if dtype == "float16" else np.float32, copy=False)
    return VectorPayload(
        encoding=dtype,
        dtype=dtype,
        shape=list(array.shape),
        data_b64=_b64encode_array(array),
    )


def encode_roq_vectors(
    vectors: Any,
    *,
    num_bits: int = 8,
    num_rounds: int = 3,
    seed: int = 42,
) -> VectorPayload:
    matrix = _as_float32_matrix(vectors)
    quantizer = RotationalQuantizer(
        RoQConfig(dim=matrix.shape[1], num_bits=num_bits, num_rounds=num_rounds, seed=seed)
    )
    quantized = quantizer.quantize(matrix, store=False)
    return VectorPayload(
        encoding=f"roq{num_bits}",
        dtype="uint8",
        shape=list(matrix.shape),
        data_b64=_b64encode_array(np.asarray(quantized["codes"], dtype=np.uint8)),
        num_bits=num_bits,
        block_size=int(quantizer.config.block_size),
        num_rounds=num_rounds,
        seed=seed,
        scales_b64=_b64encode_array(np.asarray(quantized["scales"], dtype=np.float32)),
        offsets_b64=_b64encode_array(np.asarray(quantized["offsets"], dtype=np.float32)),
        norms_sq_b64=_b64encode_array(np.asarray(quantized["norms_sq"], dtype=np.float32)),
        code_sums_b64=_b64encode_array(np.asarray(quantized["code_sums"], dtype=np.float32)),
        code_shape=list(np.asarray(quantized["codes"]).shape),
    )


def _roq_code_shape(payload: VectorPayload) -> List[int]:
    if payload.code_shape is not None:
        return list(payload.code_shape)
    dim = int(payload.shape[1])
    num_bits = int(payload.num_bits or 8)
    if num_bits == 8:
        packed_dim = dim
    elif num_bits == 4:
        packed_dim = math.ceil(dim / 2)
    elif num_bits == 2:
        packed_dim = math.ceil(dim / 4)
    elif num_bits == 1:
        packed_dim = math.ceil(dim / 8)
    else:
        raise ValueError(f"Unsupported RoQ bit width: {num_bits}")
    return [int(payload.shape[0]), packed_dim]


def decode_vector_payload(payload: VectorPayload | Dict[str, Any]) -> np.ndarray:
    if isinstance(payload, dict):
        payload = VectorPayload(**payload)

    if payload.encoding in {"float16", "float32"}:
        dtype = np.float16 if payload.encoding == "float16" else np.float32
        return _b64decode_array(payload.data_b64, dtype, payload.shape).astype(np.float32, copy=False)

    if payload.encoding.startswith("roq"):
        if payload.scales_b64 is None or payload.offsets_b64 is None:
            raise ValueError("RoQ payload is missing affine metadata")
        quantizer = RotationalQuantizer(
            RoQConfig(
                dim=int(payload.shape[1]),
                num_bits=int(payload.num_bits or payload.encoding.removeprefix("roq")),
                num_rounds=int(payload.num_rounds or 3),
                block_size=payload.block_size,
                seed=int(payload.seed),
            )
        )
        codes = _b64decode_array(payload.data_b64, np.uint8, _roq_code_shape(payload))
        scales = _b64decode_array(payload.scales_b64, np.float32, [int(payload.shape[0])])
        offsets = _b64decode_array(payload.offsets_b64, np.float32, [int(payload.shape[0])])
        return quantizer.decode(codes, scales, offsets).cpu().numpy().astype(np.float32, copy=False)

    raise ValueError(f"Unsupported vector encoding: {payload.encoding}")


def quantize_payload_on_ingress(
    payload: VectorPayload | Dict[str, Any],
    *,
    num_bits: int = 8,
) -> VectorPayload:
    if isinstance(payload, dict):
        payload = VectorPayload(**payload)
    if payload.encoding.startswith("roq"):
        return payload
    vectors = decode_vector_payload(payload)
    matrix = _as_float32_matrix(vectors)
    quantizer = RotationalQuantizer(RoQConfig(dim=matrix.shape[1], num_bits=num_bits))
    quantized = quantizer.quantize(matrix, store=False)
    return VectorPayload(
        encoding=f"roq{num_bits}",
        dtype="uint8",
        shape=list(matrix.shape),
        data_b64=_b64encode_array(np.asarray(quantized["codes"], dtype=np.uint8)),
        num_bits=num_bits,
        block_size=int(quantizer.config.block_size),
        num_rounds=int(quantizer.config.num_rounds),
        seed=int(quantizer.config.seed),
        scales_b64=_b64encode_array(np.asarray(quantized["scales"], dtype=np.float32)),
        offsets_b64=_b64encode_array(np.asarray(quantized["offsets"], dtype=np.float32)),
        norms_sq_b64=_b64encode_array(np.asarray(quantized["norms_sq"], dtype=np.float32)),
        code_sums_b64=_b64encode_array(np.asarray(quantized["code_sums"], dtype=np.float32)),
        code_shape=list(np.asarray(quantized["codes"]).shape),
    )


def build_optimizer_request(
    *,
    query_vectors: Any,
    candidate_items: Iterable[Any],
    query_text: str = "",
    constraints: Optional[Dict[str, Any]] = None,
    solver_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    prefer_quantized: bool = True,
    quantized_bits: int = 8,
) -> OptimizerRequest:
    query_payload = (
        encode_roq_vectors(query_vectors, num_bits=quantized_bits)
        if prefer_quantized
        else encode_float_vectors(query_vectors)
    )

    candidates: List[OptimizerCandidate] = []
    for item in candidate_items:
        if isinstance(item, tuple):
            item_id, vectors, payload = item
        else:
            item_id = item.get("id")
            vectors = item.get("vector")
            payload = item.get("payload", {})
        payload = dict(payload or {})
        vector_payload = (
            encode_roq_vectors(vectors, num_bits=quantized_bits)
            if prefer_quantized
            else encode_float_vectors(vectors)
        )
        matrix = _as_float32_matrix(vectors)
        candidates.append(
            OptimizerCandidate(
                chunk_id=str(item_id),
                text=payload.get("text", ""),
                token_count=int(payload.get("token_count", matrix.shape[0])),
                vectors=vector_payload,
                fact_density=float(payload.get("fact_density", 0.5)),
                centrality_score=float(payload.get("centrality_score", 0.5)),
                recency_score=float(payload.get("recency_score", 0.5)),
                auxiliary_score=float(payload.get("auxiliary_score", 0.0)),
                rhetorical_role=str(payload.get("rhetorical_role", "unknown")),
                cluster_id=payload.get("cluster_id"),
                metadata={
                    key: value
                    for key, value in payload.items()
                    if key not in {
                        "text",
                        "token_count",
                        "fact_density",
                        "centrality_score",
                        "recency_score",
                        "auxiliary_score",
                        "rhetorical_role",
                        "cluster_id",
                    }
                },
            )
        )

    return OptimizerRequest(
        query_text=query_text,
        query_vectors=query_payload,
        candidates=candidates,
        constraints=dict(constraints or {}),
        solver_config=dict(solver_config or {}),
        metadata=dict(metadata or {}),
    )


class GpuFulfilmentPipeline:
    """
    Stateless GPU-oriented optimizer for dense and multivector bundles.

    The local search loop still runs through the solver binding, but this class
    ensures the expensive fulfilment/relevance tensor construction remains
    multivector-preserving and GPU-first.
    """

    def __init__(self, *, require_gpu: bool | None = None, ingress_quantization_bits: int = 8):
        self.require_gpu = default_optimizer_require_gpu() if require_gpu is None else bool(require_gpu)
        self.ingress_quantization_bits = ingress_quantization_bits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._warm_dimensions: set[int] = set()
        self._last_self_test: Optional[Dict[str, Any]] = None

    def _resolve_optimizer_policy(self, policy_override: Any = None) -> Dict[str, Any]:
        presets = _optimizer_policy_presets()
        if policy_override is None:
            resolved = copy.deepcopy(presets["baseline_v1"])
        elif isinstance(policy_override, str):
            resolved = copy.deepcopy(presets.get(policy_override, presets["baseline_v1"]))
            if policy_override not in presets:
                resolved["name"] = policy_override
        elif isinstance(policy_override, dict):
            policy_name = str(policy_override.get("name", "baseline_v1"))
            resolved = copy.deepcopy(presets.get(policy_name, presets["baseline_v1"]))
            overrides = {key: copy.deepcopy(value) for key, value in policy_override.items() if key != "name"}
            resolved = _deep_merge_dict(resolved, overrides)
            resolved["name"] = policy_name
        else:
            raise TypeError("optimizer_policy must be None, a preset name, or a dict override")
        blend_weights = resolved.get("blend_weights", {})
        resolved["blend_weights"] = {
            branch: self._normalize_weight_map(dict(weights))
            for branch, weights in dict(blend_weights).items()
        }
        if "ranking" in resolved:
            resolved["ranking"] = self._normalize_weight_map(dict(resolved["ranking"]))
        return resolved

    def _resolve_edge_case_fallback_config(self, fallback_override: Any = None) -> Dict[str, Any]:
        resolved = _default_edge_case_fallback_config()
        if fallback_override in (None, True):
            return resolved
        if fallback_override in (False, "disabled"):
            resolved["enabled"] = False
            return resolved
        if isinstance(fallback_override, str):
            resolved["name"] = str(fallback_override)
            return resolved
        if isinstance(fallback_override, dict):
            return _deep_merge_dict(resolved, fallback_override)
        return resolved

    def _edge_case_fallback_decision(
        self,
        *,
        controller_features: Dict[str, float],
        fallback_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(fallback_config.get("enabled", True)):
            return {
                "enabled": False,
                "applied": False,
                "name": str(fallback_config.get("name", "disabled")),
                "candidate_count_gate": False,
                "risk_signal_count": 0,
                "min_risk_signals": int(fallback_config.get("min_risk_signals", 0)),
                "activation_reasons": [],
                "signals": [],
                "triggered_signals": [],
                "solver_overrides": {},
            }

        thresholds = dict(fallback_config.get("thresholds", {}))
        signal_specs = [
            ("uncovered_mass", ">=", float(thresholds.get("uncovered_mass", 0.18))),
            (
                "retrieval_disagreement_mean",
                ">=",
                float(thresholds.get("retrieval_disagreement_mean", 0.08)),
            ),
            ("ambiguity_p90", ">=", float(thresholds.get("ambiguity_p90", 0.30))),
            ("metadata_coverage", "<=", float(thresholds.get("metadata_coverage", 0.20))),
            ("slack_ratio", ">=", float(thresholds.get("slack_ratio", 0.20))),
            ("frontier_gain_p90", ">=", float(thresholds.get("frontier_gain_p90", 0.25))),
            ("pack_density", "<=", float(thresholds.get("pack_density", 0.04))),
        ]
        signals = []
        triggered_signals: List[str] = []
        for name, operator, threshold in signal_specs:
            value = float(controller_features.get(name, 0.0))
            triggered = value >= threshold if operator == ">=" else value <= threshold
            signals.append(
                {
                    "name": name,
                    "operator": operator,
                    "value": value,
                    "threshold": threshold,
                    "triggered": bool(triggered),
                }
            )
            if triggered:
                triggered_signals.append(name)
        candidate_count_gate = float(controller_features.get("candidate_count", 0.0)) >= float(
            fallback_config.get("min_candidate_count", 48)
        )
        min_risk_signals = int(fallback_config.get("min_risk_signals", 3))
        high_uncovered = float(controller_features.get("uncovered_mass", 0.0)) >= float(
            thresholds.get("uncovered_mass", 0.15)
        )
        high_retrieval_ambiguity = (
            float(controller_features.get("retrieval_disagreement_mean", 0.0))
            >= float(thresholds.get("retrieval_disagreement_mean", 0.50))
            and float(controller_features.get("ambiguity_p90", 0.0))
            >= float(thresholds.get("ambiguity_p90", 0.90))
        )
        metadata_conflict = (
            float(controller_features.get("metadata_coverage", 0.0))
            <= float(thresholds.get("metadata_coverage", 0.20))
            and high_retrieval_ambiguity
        )
        activation_reasons = []
        if high_uncovered:
            activation_reasons.append("high_uncovered_mass")
        if high_retrieval_ambiguity:
            activation_reasons.append("high_retrieval_ambiguity")
        if metadata_conflict:
            activation_reasons.append("metadata_conflict")
        risk_signal_gate = len(triggered_signals) >= max(min_risk_signals, 1)
        applied = candidate_count_gate and (bool(activation_reasons) or risk_signal_gate)
        return {
            "enabled": True,
            "applied": bool(applied),
            "name": str(fallback_config.get("name", "high_effort_v1")),
            "candidate_count_gate": bool(candidate_count_gate),
            "risk_signal_count": len(triggered_signals),
            "risk_signal_gate": bool(risk_signal_gate),
            "min_risk_signals": min_risk_signals,
            "activation_reasons": activation_reasons,
            "signals": signals,
            "triggered_signals": triggered_signals,
            "solver_overrides": {},
        }

    def _apply_edge_case_fallback_profile(
        self,
        *,
        solver_config_payload: Dict[str, Any],
        blend_weights: Dict[str, Dict[str, float]],
        fallback_config: Dict[str, Any],
        fallback_summary: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Any]]:
        updated_solver = dict(solver_config_payload)
        updated_blend = {branch: dict(weights) for branch, weights in blend_weights.items()}
        if not fallback_summary.get("applied", False):
            return updated_solver, updated_blend, fallback_summary
        applied_overrides: Dict[str, Any] = {}
        for key, override_value in dict(fallback_config.get("solver_overrides", {})).items():
            current_value = updated_solver.get(key)
            new_value = override_value
            if isinstance(override_value, bool):
                new_value = bool(current_value) or bool(override_value)
            elif isinstance(override_value, int) and isinstance(current_value, (int, float)):
                new_value = max(int(current_value), int(override_value))
            elif isinstance(override_value, float) and isinstance(current_value, (int, float)):
                new_value = max(float(current_value), float(override_value))
            updated_solver[key] = new_value
            if current_value != new_value:
                applied_overrides[key] = {"old": current_value, "new": new_value}
        applied_blend_overrides: Dict[str, Any] = {}
        for branch, overrides in dict(fallback_config.get("blend_overrides", {})).items():
            if branch not in updated_blend:
                continue
            branch_changes = {}
            for key, value in dict(overrides).items():
                if key not in updated_blend[branch]:
                    continue
                current_value = updated_blend[branch][key]
                new_value = float(value)
                updated_blend[branch][key] = new_value
                if current_value != new_value:
                    branch_changes[key] = {"old": current_value, "new": new_value}
            if branch_changes:
                applied_blend_overrides[branch] = branch_changes
        updated_blend = {
            branch: self._normalize_weight_map(weights) for branch, weights in updated_blend.items()
        }
        updated_summary = dict(fallback_summary)
        updated_summary["solver_overrides"] = applied_overrides
        updated_summary["blend_overrides"] = applied_blend_overrides
        return updated_solver, updated_blend, updated_summary

    def _ensure_gpu(self) -> None:
        if self.require_gpu and self.device.type != "cuda":
            raise RuntimeError("CUDA is required for the GPU-first fulfilment pipeline")

    def backend_health(self) -> Dict[str, Any]:
        solver_health = self._solver_backend_health()
        precompute_ready = self.device.type == "cuda" or not self.require_gpu
        return {
            "available": precompute_ready and solver_health["available"],
            "execution_mode": self._execution_mode(
                solver_health["requested_use_gpu"],
                solver_health.get("backend_kind"),
            ),
            "device": str(self.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "require_gpu": self.require_gpu,
            "ingress_quantization_bits": self.ingress_quantization_bits,
            "solver_backend": solver_health,
            "self_test": self._last_self_test,
        }

    def _target_solver_use_gpu(self) -> bool:
        try:
            from latence_solver import backend_status

            status = backend_status()
        except Exception:
            return False
        return bool(
            status.get("premium_backend_available")
            or status.get("experimental_cuda_available")
            or status.get("experimental_gpu_available")
        )

    def _execution_mode(self, use_gpu: bool, backend_kind: str | None = None) -> str:
        backend_kind = backend_kind or ""
        actual_gpu = use_gpu and backend_kind not in {"", "cpu_reference", "unknown"}
        return END_TO_END_GPU_EXECUTION_MODE if actual_gpu else HYBRID_EXECUTION_MODE

    def _solver_backend_health(self) -> Dict[str, Any]:
        try:
            from latence_solver import SolverConfig, TabuSearchSolver, backend_status

            use_gpu = self._target_solver_use_gpu()
            solver = TabuSearchSolver(SolverConfig(iterations=2, use_gpu=use_gpu))
            return {
                "available": True,
                "backend_kind": solver.backend_kind(),
                "requested_use_gpu": use_gpu,
                "status": backend_status(),
            }
        except Exception as exc:
            return {
                "available": False,
                "requested_use_gpu": self._target_solver_use_gpu(),
                "reason": str(exc),
            }

    def request_path_self_test(self) -> Dict[str, Any]:
        candidate_items = [
            (
                0,
                np.asarray([1.0, 0.0], dtype=np.float32),
                {"text": "alpha", "token_count": 32, "fact_density": 0.6, "cluster_id": 0},
            ),
            (
                1,
                np.asarray([0.0, 1.0], dtype=np.float32),
                {"text": "beta", "token_count": 32, "fact_density": 0.2, "cluster_id": 1},
            ),
        ]
        try:
            request = build_optimizer_request(
                query_vectors=np.asarray([1.0, 0.0], dtype=np.float32),
                candidate_items=candidate_items,
                query_text="alpha",
                constraints={"max_tokens": 32, "max_chunks": 1},
                solver_config={"iterations": 8, "mu": 1.0, "random_seed": 7},
                prefer_quantized=True,
                quantized_bits=self.ingress_quantization_bits,
            )
            result = self.optimize(request)
            self._last_self_test = {
                "ok": result["selected_ids"] == ["0"],
                "selected_ids": list(result["selected_ids"]),
                "backend_kind": result["backend_kind"],
                "solver_backend_kind": result.get("solver_backend_kind"),
            }
        except Exception as exc:
            self._last_self_test = {"ok": False, "reason": str(exc)}
        return dict(self._last_self_test)

    def warmup(self, dim: int) -> None:
        if dim in self._warm_dimensions:
            return
        if self.device.type != "cuda":
            self._warm_dimensions.add(dim)
            return
        self._ensure_gpu()
        sample = torch.randn(1, dim, device=self.device, dtype=torch.float32)
        _ = sample @ sample.T
        torch.cuda.synchronize()
        self._warm_dimensions.add(dim)

    def _decode_request(self, request: OptimizerRequest | Dict[str, Any]) -> tuple[OptimizerRequest, np.ndarray, List[np.ndarray]]:
        if isinstance(request, dict):
            request = OptimizerRequest(
                query_text=request.get("query_text", ""),
                query_vectors=VectorPayload(**request["query_vectors"]),
                candidates=[
                    OptimizerCandidate(
                        **{
                            **candidate,
                            "vectors": VectorPayload(**candidate["vectors"]),
                        }
                    )
                    for candidate in request.get("candidates", [])
                ],
                constraints=dict(request.get("constraints", {})),
                solver_config=dict(request.get("solver_config", {})),
                metadata=dict(request.get("metadata", {})),
            )
        _enforce_optimizer_payload_limit(request.approx_bytes())

        query_payload = (
            request.query_vectors
            if str(request.query_vectors.encoding).startswith("roq")
            else request.query_vectors
        )
        query_vectors = _normalize_rows(decode_vector_payload(query_payload))
        query_vectors = self._augment_query_vectors(query_vectors, request.metadata)

        candidate_vectors: List[np.ndarray] = []
        normalized_candidates: List[OptimizerCandidate] = []
        for candidate in request.candidates:
            stored_vectors = (
                candidate.vectors
                if str(candidate.vectors.encoding).startswith("roq")
                else candidate.vectors
            )
            decoded = _normalize_rows(decode_vector_payload(stored_vectors))
            candidate_vectors.append(decoded)
            normalized_candidates.append(
                OptimizerCandidate(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    token_count=int(candidate.token_count),
                    vectors=stored_vectors,
                    fact_density=float(candidate.fact_density),
                    centrality_score=float(candidate.centrality_score),
                    recency_score=float(candidate.recency_score),
                    auxiliary_score=float(candidate.auxiliary_score),
                    rhetorical_role=str(candidate.rhetorical_role),
                    cluster_id=candidate.cluster_id,
                    metadata=dict(candidate.metadata),
                )
            )

        normalized_request = OptimizerRequest(
            query_text=request.query_text,
            query_vectors=query_payload,
            candidates=normalized_candidates,
            constraints=dict(request.constraints),
            solver_config=dict(request.solver_config),
            metadata=dict(request.metadata),
        )
        return normalized_request, query_vectors, candidate_vectors

    def optimize_in_process(
        self,
        *,
        query_vectors: Any,
        candidate_items: Iterable[Any],
        query_text: str = "",
        constraints: Optional[Dict[str, Any]] = None,
        solver_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        solver: Any | None = None,
    ) -> Dict[str, Any]:
        self._ensure_gpu()
        metadata_payload = dict(metadata or {})
        normalized_query_vectors = _normalize_rows(_as_float32_matrix(query_vectors))
        normalized_query_vectors = self._augment_query_vectors(normalized_query_vectors, metadata_payload)

        normalized_candidates: List[OptimizerCandidate] = []
        candidate_vectors: List[np.ndarray] = []
        raw_payload_bytes = int(normalized_query_vectors.nbytes)
        for item in candidate_items:
            if isinstance(item, tuple):
                item_id, vectors, payload = item
            else:
                item_id = item.get("id")
                vectors = item.get("vectors")
                if vectors is None:
                    vectors = item.get("vector")
                payload = item.get("payload", {})
            matrix = _normalize_rows(_as_float32_matrix(vectors))
            raw_payload_bytes += int(matrix.nbytes)
            payload_dict = dict(payload or {})
            candidate_vectors.append(matrix)
            normalized_candidates.append(
                OptimizerCandidate(
                    chunk_id=str(item_id),
                    text=str(payload_dict.get("text", "")),
                    token_count=int(payload_dict.get("token_count", matrix.shape[0])),
                    vectors=_placeholder_vector_payload(matrix),
                    fact_density=float(payload_dict.get("fact_density", 0.5)),
                    centrality_score=float(payload_dict.get("centrality_score", 0.5)),
                    recency_score=float(payload_dict.get("recency_score", 0.5)),
                    auxiliary_score=float(payload_dict.get("auxiliary_score", 0.0)),
                    rhetorical_role=str(payload_dict.get("rhetorical_role", "unknown")),
                    cluster_id=payload_dict.get("cluster_id"),
                    metadata={
                        key: value
                        for key, value in payload_dict.items()
                        if key
                        not in {
                            "text",
                            "token_count",
                            "fact_density",
                            "centrality_score",
                            "recency_score",
                            "auxiliary_score",
                            "rhetorical_role",
                            "cluster_id",
                        }
                    },
                )
            )

        normalized_request = OptimizerRequest(
            query_text=query_text,
            query_vectors=_placeholder_vector_payload(normalized_query_vectors),
            candidates=normalized_candidates,
            constraints=dict(constraints or {}),
            solver_config=dict(solver_config or {}),
            metadata=metadata_payload,
        )
        _enforce_optimizer_payload_limit(raw_payload_bytes)
        return self._optimize_decoded(
            normalized_request,
            normalized_query_vectors,
            candidate_vectors,
            solver=solver,
            payload_bytes=raw_payload_bytes,
        )

    def _augment_query_vectors(self, query_vectors: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        nested_query_payload = metadata.get("query_payload", {})
        if not isinstance(nested_query_payload, dict):
            nested_query_payload = {}
        expansion_vectors = (
            metadata.get("query_expansion_vectors")
            or metadata.get("query_aspect_vectors")
            or metadata.get("query_focus_vectors")
            or nested_query_payload.get("query_expansion_vectors")
            or nested_query_payload.get("query_aspect_vectors")
            or nested_query_payload.get("query_focus_vectors")
        )
        if expansion_vectors is None:
            return query_vectors
        expansion_matrix = _as_float32_matrix(expansion_vectors)
        if expansion_matrix.shape[1] != query_vectors.shape[1]:
            raise ValueError("query_expansion_vectors must match query embedding dimension")
        combined = np.concatenate([query_vectors, _normalize_rows(expansion_matrix)], axis=0)
        return _normalize_rows(combined.astype(np.float32, copy=False))

    def _derive_query_bundle(
        self,
        *,
        query_vectors: np.ndarray,
        query_text: str,
        candidates: Sequence[OptimizerCandidate],
        candidate_vectors: Sequence[np.ndarray],
        policy: Dict[str, Any],
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if query_vectors.size == 0 or not candidate_vectors:
            return query_vectors, {"base_query_rows": int(query_vectors.shape[0]), "derived_query_rows": 0}

        base_vectors = _normalize_rows(query_vectors.astype(np.float32, copy=False))
        if not candidates:
            return base_vectors, {"base_query_rows": int(base_vectors.shape[0]), "derived_query_rows": 0}

        candidate_centroids = _normalize_rows(
            np.stack([_centroid(vector) for vector in candidate_vectors]).astype(np.float32, copy=False)
        )
        base_centroid = _centroid(base_vectors).astype(np.float32, copy=False)
        retrieval = self._retrieval_signal_scores(candidates, policy=policy)
        lexical = self._lexical_overlap_scores(query_text, candidates)
        cluster = self._cluster_novelty_scores(candidates)
        disagreement = retrieval["disagreement"]
        query_bundle_policy = policy["query_bundle"]

        facet_sources: list[tuple[str, np.ndarray]] = [
            ("lexical", lexical),
            ("dense", retrieval["dense"]),
            ("sparse", retrieval["sparse"]),
            ("disagreement", disagreement),
            (
                "cluster",
                self._normalize_feature(
                    query_bundle_policy["cluster_mix_lexical"] * lexical
                    + query_bundle_policy["cluster_mix_cluster"] * cluster
                ),
            ),
        ]
        if base_vectors.shape[0] > 1:
            facet_sources.append(
                (
                    "coverage_bridge",
                    self._normalize_feature(
                        query_bundle_policy["coverage_bridge_retrieval"] * retrieval["composite"]
                        + query_bundle_policy["coverage_bridge_lexical"] * lexical
                    ),
                )
            )

        facets: list[np.ndarray] = [row.astype(np.float32, copy=False) for row in base_vectors]
        added_sources: list[str] = []
        max_extra_facets = (
            int(query_bundle_policy["max_extra_facets_single"])
            if base_vectors.shape[0] <= 1
            else int(query_bundle_policy["max_extra_facets_multi"])
        )

        for source_name, scores in facet_sources:
            if len(added_sources) >= max_extra_facets or scores.size == 0 or float(np.max(scores)) <= 1e-6:
                continue
            order = np.argsort(-scores, kind="stable")
            top_k = order[: min(int(query_bundle_policy["facet_top_k"]), len(order))]
            weights = scores[top_k].astype(np.float32, copy=False)
            if float(np.sum(weights)) <= 1e-6:
                continue
            centroid = np.average(candidate_centroids[top_k], axis=0, weights=weights).astype(
                np.float32, copy=False
            )
            blended = _normalize_rows(
                (
                    query_bundle_policy["centroid_base_weight"] * base_centroid.reshape(1, -1)
                    + query_bundle_policy["centroid_candidate_weight"] * centroid.reshape(1, -1)
                ).astype(
                    np.float32, copy=False
                )
            )[0]
            similarity_to_existing = max(float(np.dot(blended, existing)) for existing in facets)
            if similarity_to_existing >= float(query_bundle_policy["facet_similarity_threshold"]):
                continue
            facets.append(blended.astype(np.float32, copy=False))
            added_sources.append(source_name)

        if base_vectors.shape[0] == 1 and not added_sources and candidate_centroids.shape[0] > 1:
            fallback_scores = self._normalize_feature(
                query_bundle_policy["fallback_mix_lexical"] * lexical
                + query_bundle_policy["fallback_mix_retrieval"] * retrieval["composite"]
                + query_bundle_policy["fallback_mix_cluster"] * cluster
            )
            order = np.argsort(-fallback_scores, kind="stable")
            base_row = facets[0]
            selected_fallback: np.ndarray | None = None
            best_diversity = -1.0
            for idx in order[: min(int(query_bundle_policy["fallback_diversity_top_k"]), len(order))]:
                candidate_row = candidate_centroids[int(idx)]
                diversity = 1.0 - float(np.dot(base_row, candidate_row))
                if diversity > best_diversity:
                    best_diversity = diversity
                    selected_fallback = candidate_row
            if selected_fallback is not None:
                fallback_blended = _normalize_rows(
                    (
                        query_bundle_policy["fallback_base_weight"] * base_centroid.reshape(1, -1)
                        + query_bundle_policy["fallback_candidate_weight"] * selected_fallback.reshape(1, -1)
                    ).astype(np.float32, copy=False)
                )[0]
                if float(np.dot(fallback_blended, base_row)) < float(
                    query_bundle_policy["fallback_similarity_threshold"]
                ):
                    facets.append(fallback_blended.astype(np.float32, copy=False))
                    added_sources.append("fallback_diverse")

        bundle = _normalize_rows(np.stack(facets).astype(np.float32, copy=False))
        return bundle, {
            "base_query_rows": int(base_vectors.shape[0]),
            "derived_query_rows": int(max(bundle.shape[0] - base_vectors.shape[0], 0)),
            "facet_sources": added_sources,
        }

    def _query_token_weights(self, coverage: np.ndarray, *, policy: Dict[str, Any]) -> np.ndarray:
        if coverage.size == 0:
            return np.zeros((0,), dtype=np.float32)
        token_policy = policy["token_weights"]
        best = coverage.max(axis=1)
        if coverage.shape[1] > 1:
            second = np.partition(coverage, coverage.shape[1] - 2, axis=1)[:, coverage.shape[1] - 2]
        else:
            second = np.zeros_like(best, dtype=np.float32)
        hardness = token_policy["best_mass_weight"] * (1.0 - best) + token_policy["margin_weight"] * (
            1.0 - np.clip(best - second, 0.0, 1.0)
        )
        weights = hardness + float(token_policy["floor"])
        weights /= weights.sum()
        return weights.astype(np.float32, copy=False)

    @staticmethod
    def _query_payload_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        nested = metadata.get("query_payload", {})
        return nested if isinstance(nested, dict) else {}

    @staticmethod
    def _coerce_weight_sequence(raw: Any) -> list[float]:
        if isinstance(raw, np.ndarray):
            return [float(value) for value in raw.reshape(-1).tolist()]
        if isinstance(raw, (list, tuple)):
            values: list[float] = []
            for item in raw:
                if isinstance(item, (int, float)):
                    values.append(float(item))
                elif isinstance(item, dict) and isinstance(item.get("weight"), (int, float)):
                    values.append(float(item["weight"]))
            return values
        return []

    def _metadata_query_token_weights(
        self,
        metadata: Dict[str, Any] | None,
        *,
        expected_rows: int,
    ) -> np.ndarray | None:
        if expected_rows <= 0:
            return None
        nested = self._query_payload_metadata(metadata)
        for key in (
            "query_token_weights",
            "query_aspect_weights",
            "teacher_token_weights",
            "query_aspects",
        ):
            raw = metadata.get(key) if isinstance(metadata, dict) else None
            if raw is None:
                raw = nested.get(key)
            values = self._coerce_weight_sequence(raw)
            if not values:
                continue
            if len(values) == expected_rows - 1 and expected_rows > 1:
                values = [float(np.mean(values))] + values
            if len(values) != expected_rows:
                continue
            arr = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
            total = float(arr.sum())
            if total <= 1e-8:
                arr = np.full((expected_rows,), 1.0 / float(expected_rows), dtype=np.float32)
            else:
                arr /= total
            return arr.astype(np.float32, copy=False)
        return None

    def _metadata_coverage_row_gains(
        self,
        metadata: Dict[str, Any] | None,
        *,
        expected_rows: int,
    ) -> np.ndarray | None:
        if expected_rows <= 0:
            return None
        nested = self._query_payload_metadata(metadata)
        for key in ("coverage_row_gains", "query_row_gains", "query_aspect_gains"):
            raw = metadata.get(key) if isinstance(metadata, dict) else None
            if raw is None:
                raw = nested.get(key)
            values = self._coerce_weight_sequence(raw)
            if not values:
                continue
            if len(values) == expected_rows - 1 and expected_rows > 1:
                values = [1.0] + values
            if len(values) != expected_rows:
                continue
            arr = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
            if float(arr.max()) <= 1e-8:
                return None
            return arr.astype(np.float32, copy=False)
        return None

    @staticmethod
    def _normalize_feature(values: np.ndarray) -> np.ndarray:
        feature = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if feature.size == 0:
            return feature
        lo = float(np.quantile(feature, 0.05))
        hi = float(np.quantile(feature, 0.95))
        if hi - lo <= 1e-8:
            order = np.argsort(feature, kind="stable")
            ranks = np.empty_like(feature, dtype=np.float32)
            if feature.size == 1:
                ranks[order[0]] = 1.0
            else:
                scale = float(feature.size + 1)
                for rank, idx in enumerate(order):
                    ranks[idx] = (rank + 1) / scale
            return ranks.astype(np.float32, copy=False)
        scaled = (feature - lo) / (hi - lo)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _reciprocal_rank_from_scores(values: np.ndarray) -> np.ndarray:
        scores = np.asarray(values, dtype=np.float32)
        if scores.size == 0:
            return np.zeros((0,), dtype=np.float32)
        ranks = np.zeros_like(scores, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        for rank, idx in enumerate(order, start=1):
            if scores[idx] <= 0.0:
                continue
            ranks[idx] = 1.0 / float(rank)
        return ranks

    @staticmethod
    def _text_terms(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) > 1
        }

    @staticmethod
    def _text_phrases(text: str, *, max_terms: int = 3) -> set[str]:
        tokens = [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 1]
        phrases: set[str] = set()
        for width in range(2, max_terms + 1):
            for offset in range(len(tokens) - width + 1):
                phrases.add(" ".join(tokens[offset : offset + width]))
        return phrases

    @staticmethod
    def _source_identity(metadata: Dict[str, Any] | None) -> str | None:
        if not isinstance(metadata, dict):
            return None
        for key in ("source_id", "document_id", "doc_id", "parent_id", "url", "path", "source"):
            value = metadata.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value).strip().lower()
        return None

    def _lexical_overlap_scores(
        self,
        query_text: str,
        candidates: Sequence[OptimizerCandidate],
    ) -> np.ndarray:
        query_terms = self._text_terms(query_text)
        query_phrases = self._text_phrases(query_text)
        if not query_terms and not query_phrases:
            return np.zeros((len(candidates),), dtype=np.float32)

        values = []
        for candidate in candidates:
            candidate_terms = set(self._text_terms(candidate.text))
            candidate_phrases = self._text_phrases(candidate.text)
            ontology_terms = candidate.metadata.get("ontology_terms", []) if candidate.metadata else []
            for term in ontology_terms:
                candidate_terms.update(self._text_terms(str(term)))
                candidate_phrases.update(self._text_phrases(str(term)))
            if not candidate_terms and not candidate_phrases:
                values.append(0.0)
                continue
            term_overlap = (
                len(query_terms & candidate_terms) / float(len(query_terms))
                if query_terms
                else 0.0
            )
            phrase_overlap = (
                len(query_phrases & candidate_phrases) / float(len(query_phrases))
                if query_phrases
                else 0.0
            )
            lexical_overlap = 0.7 * term_overlap + 0.3 * phrase_overlap
            values.append(lexical_overlap)
        return self._normalize_feature(np.asarray(values, dtype=np.float32))

    def _retrieval_signal_scores(
        self, candidates: Sequence[OptimizerCandidate], *, policy: Dict[str, Any] | None = None
    ) -> Dict[str, np.ndarray]:
        retrieval_policy = (policy or _default_optimizer_policy())["retrieval"]
        base_values = []
        rrf_values = []
        dense_values = []
        sparse_values = []
        rerank_values = []
        utility_values = []
        entailment_values = []
        contradiction_values = []
        teacher_utility_values = []
        rerank_presence = []
        for candidate in candidates:
            metadata = candidate.metadata or {}
            base_values.append(float(metadata.get("base_relevance", 0.0) or 0.0))
            rrf_values.append(float(metadata.get("rrf_score", 0.0) or 0.0))
            dense_values.append(float(metadata.get("dense_score", 0.0) or 0.0))
            sparse_values.append(float(metadata.get("sparse_score", 0.0) or 0.0))
            rerank_raw: float | None = None
            for key in ("rerank_score", "cross_encoder_score", "ce_score", "ltr_score"):
                value = metadata.get(key)
                if isinstance(value, (int, float)):
                    rerank_raw = float(value) if rerank_raw is None else max(rerank_raw, float(value))
            rerank_values.append(float(rerank_raw) if rerank_raw is not None else 0.0)
            rerank_presence.append(1.0 if rerank_raw is not None else 0.0)
            utility_raw: float | None = None
            for key in ("utility_score", "nli_utility_score"):
                value = metadata.get(key)
                if isinstance(value, (int, float)):
                    utility_raw = float(value) if utility_raw is None else max(utility_raw, float(value))
            utility_values.append(float(utility_raw) if utility_raw is not None else 0.0)
            entailment_values.append(float(metadata.get("nli_entailment", 0.0) or 0.0))
            contradiction_values.append(float(metadata.get("nli_contradiction", 0.0) or 0.0))
            teacher_raw: float | None = None
            for key in ("teacher_utility_score", "distilled_utility_score"):
                value = metadata.get(key)
                if isinstance(value, (int, float)):
                    teacher_raw = float(value) if teacher_raw is None else max(teacher_raw, float(value))
            teacher_utility_values.append(float(teacher_raw) if teacher_raw is not None else 0.0)

        base = self._normalize_feature(np.asarray(base_values, dtype=np.float32))
        rrf = self._normalize_feature(np.asarray(rrf_values, dtype=np.float32))
        dense = self._normalize_feature(np.asarray(dense_values, dtype=np.float32))
        sparse = self._normalize_feature(np.asarray(sparse_values, dtype=np.float32))
        rerank = self._normalize_feature(np.asarray(rerank_values, dtype=np.float32))
        utility = self._normalize_feature(np.asarray(utility_values, dtype=np.float32))
        entailment = self._normalize_feature(np.asarray(entailment_values, dtype=np.float32))
        contradiction = self._normalize_feature(np.asarray(contradiction_values, dtype=np.float32))
        teacher_utility = self._normalize_feature(np.asarray(teacher_utility_values, dtype=np.float32))
        dense_rr = self._reciprocal_rank_from_scores(dense)
        sparse_rr = self._reciprocal_rank_from_scores(sparse)
        rerank_rr = self._reciprocal_rank_from_scores(rerank)
        utility_rr = self._reciprocal_rank_from_scores(utility)
        agreement = np.sqrt(np.clip(dense * sparse, 0.0, 1.0)).astype(np.float32, copy=False)
        rerank_anchor = np.maximum(base, np.maximum(dense, sparse)).astype(np.float32, copy=False)
        rerank_agreement = np.sqrt(np.clip(rerank * rerank_anchor, 0.0, 1.0)).astype(
            np.float32, copy=False
        )
        disagreement = np.abs(dense - sparse).astype(np.float32, copy=False)
        disagreement_effective = (
            disagreement
            * (
                1.0
                - float(retrieval_policy.get("rerank_disagreement_gate", 0.0))
                * rerank_agreement
            )
        ).astype(np.float32, copy=False)
        composite = (
            retrieval_policy["base"] * base
            + retrieval_policy["rrf"] * rrf
            + retrieval_policy["dense"] * dense
            + retrieval_policy["sparse"] * sparse
            + retrieval_policy["dense_rr"] * dense_rr
            + retrieval_policy["sparse_rr"] * sparse_rr
            + retrieval_policy["agreement"] * agreement
            + float(retrieval_policy.get("rerank", 0.0)) * rerank
            + float(retrieval_policy.get("rerank_rr", 0.0)) * rerank_rr
            + float(retrieval_policy.get("rerank_agreement", 0.0)) * rerank_agreement
            + float(retrieval_policy.get("utility", 0.0)) * utility
            + float(retrieval_policy.get("utility_rr", 0.0)) * utility_rr
            + float(retrieval_policy.get("entailment", 0.0)) * entailment
            + float(retrieval_policy.get("teacher_utility", 0.0)) * teacher_utility
            - retrieval_policy["disagreement_penalty"] * disagreement_effective
            - float(retrieval_policy.get("contradiction_penalty", 0.0)) * contradiction
        ).astype(np.float32, copy=False)
        return {
            "base": base,
            "rrf": rrf,
            "dense": dense,
            "sparse": sparse,
            "rerank": rerank,
            "rerank_rr": rerank_rr,
            "rerank_agreement": rerank_agreement,
            "rerank_presence": np.asarray(rerank_presence, dtype=np.float32),
            "utility": utility,
            "utility_rr": utility_rr,
            "entailment": entailment,
            "contradiction": contradiction,
            "teacher_utility": teacher_utility,
            "agreement": agreement,
            "disagreement": disagreement,
            "disagreement_effective": disagreement_effective,
            "composite": self._normalize_feature(composite),
        }

    def _embedding_novelty_scores(
        self,
        coverage: np.ndarray,
        token_weights: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        if coverage.size == 0:
            zeros = np.zeros((coverage.shape[1] if coverage.ndim == 2 else 0,), dtype=np.float32)
            return {"margin": zeros, "winner_mass": zeros, "specificity": zeros}

        query_count, candidate_count = coverage.shape
        weights = (
            token_weights.astype(np.float32, copy=False)
            if token_weights.size
            else np.full((query_count,), 1.0 / float(max(query_count, 1)), dtype=np.float32)
        )
        top_idx = np.argmax(coverage, axis=1)
        top_vals = coverage[np.arange(query_count), top_idx]
        if candidate_count > 1:
            second_vals = np.partition(coverage, candidate_count - 2, axis=1)[:, candidate_count - 2]
        else:
            second_vals = np.zeros((query_count,), dtype=np.float32)

        margin = np.zeros((candidate_count,), dtype=np.float32)
        winner_mass = np.zeros((candidate_count,), dtype=np.float32)
        for token_index, winner in enumerate(top_idx):
            weight = float(weights[token_index])
            winner_mass[winner] += weight * float(top_vals[token_index])
            margin[winner] += weight * float(max(top_vals[token_index] - second_vals[token_index], 0.0))

        specificity = np.zeros((candidate_count,), dtype=np.float32)
        if query_count > 1:
            normalizer = math.log(float(query_count))
            for candidate_index in range(candidate_count):
                distribution = coverage[:, candidate_index].astype(np.float64, copy=False)
                total = float(distribution.sum())
                if total <= 1e-8:
                    continue
                distribution = distribution / total
                entropy = -float(np.sum(distribution * np.log(np.clip(distribution, 1e-8, None)))) / normalizer
                specificity[candidate_index] = float(max(0.0, 1.0 - entropy))

        return {
            "margin": self._normalize_feature(margin),
            "winner_mass": self._normalize_feature(winner_mass),
            "specificity": self._normalize_feature(specificity),
        }

    def _cluster_novelty_scores(self, candidates: Sequence[OptimizerCandidate]) -> np.ndarray:
        counts: Dict[int, int] = {}
        for candidate in candidates:
            if candidate.cluster_id is None:
                continue
            counts[int(candidate.cluster_id)] = counts.get(int(candidate.cluster_id), 0) + 1
        values = []
        for candidate in candidates:
            if candidate.cluster_id is None:
                values.append(0.5)
                continue
            values.append(1.0 / float(counts.get(int(candidate.cluster_id), 1)))
        return self._normalize_feature(np.asarray(values, dtype=np.float32))

    def _payload_grounded_signal_scores(
        self,
        coverage_matrix: np.ndarray,
        query_token_weights: np.ndarray,
        relevance_scores: np.ndarray,
        retrieval_signals: Dict[str, np.ndarray],
        redundancy_matrix: np.ndarray,
        token_costs: np.ndarray,
        policy: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        signal_policy = policy["payload_signals"]
        candidate_count = int(relevance_scores.shape[0])
        zeros = np.zeros((candidate_count,), dtype=np.float32)
        if candidate_count == 0:
            return {
                "hard_coverage": zeros,
                "frontier_gain": zeros,
                "uniqueness": zeros,
                "anti_shadow": zeros,
                "ambiguity": zeros,
                "score_cliff": zeros,
                "support_quorum": zeros,
                "pack_value": zeros,
            }

        if coverage_matrix.size == 0 or query_token_weights.size == 0:
            hard_coverage = zeros
            frontier_gain = zeros
        else:
            mean_token_coverage = coverage_matrix.mean(axis=1).astype(np.float32, copy=False)
            token_hardness = (query_token_weights * (1.0 - mean_token_coverage)).astype(
                np.float32, copy=False
            )
            token_frontier = (
                query_token_weights * (1.0 - coverage_matrix.max(axis=1))
            ).astype(np.float32, copy=False)
            hard_coverage = self._normalize_feature((coverage_matrix * token_hardness[:, None]).sum(axis=0))
            frontier_gain = self._normalize_feature((coverage_matrix * token_frontier[:, None]).sum(axis=0))

        if redundancy_matrix.size == 0:
            uniqueness = np.ones((candidate_count,), dtype=np.float32)
        else:
            max_redundancy = np.max(redundancy_matrix, axis=1).astype(np.float32, copy=False)
            uniqueness = self._normalize_feature(1.0 - np.clip(max_redundancy, 0.0, 1.0))

        anti_shadow = self._normalize_feature(
            signal_policy["anti_shadow_frontier_weight"] * frontier_gain
            + signal_policy["anti_shadow_uniqueness_weight"] * uniqueness
        )
        contradiction = retrieval_signals.get("contradiction", zeros)
        utility = retrieval_signals.get("utility", zeros)
        ambiguity = self._normalize_feature(
            (
                retrieval_signals["disagreement"]
                + float(signal_policy.get("ambiguity_contradiction_weight", 0.0)) * contradiction
            )
            * (
                signal_policy["ambiguity_hard_coverage_weight"] * hard_coverage
                + signal_policy["ambiguity_relevance_weight"] * self._normalize_feature(relevance_scores)
            )
        )
        support_quorum = self._normalize_feature(
            np.minimum(hard_coverage, frontier_gain)
            * (
                signal_policy["support_quorum_floor"]
                + signal_policy["support_quorum_uniqueness_weight"] * uniqueness
                + float(signal_policy.get("support_quorum_utility_weight", 0.0)) * utility
            )
            * (
                1.0
                - float(signal_policy.get("support_quorum_contradiction_penalty", 0.0))
                * contradiction
            )
        )

        score_cliff = np.zeros((candidate_count,), dtype=np.float32)
        composite = retrieval_signals["composite"]
        if composite.size > 1:
            order = np.argsort(-composite, kind="stable")
            for rank, idx in enumerate(order[:-1]):
                next_idx = order[rank + 1]
                score_cliff[idx] = float(max(composite[idx] - composite[next_idx], 0.0))
            score_cliff = self._normalize_feature(score_cliff)

        cost_scale = np.sqrt(np.clip(token_costs.astype(np.float32, copy=False), 1.0, None))
        pack_value = self._normalize_feature(
            (
                signal_policy["pack_value_frontier_weight"] * frontier_gain
                + signal_policy["pack_value_quorum_weight"] * support_quorum
                + signal_policy["pack_value_uniqueness_weight"] * uniqueness
                + float(signal_policy.get("pack_value_utility_weight", 0.0)) * utility
            )
            / cost_scale
        )

        return {
            "hard_coverage": hard_coverage,
            "frontier_gain": frontier_gain,
            "uniqueness": uniqueness,
            "anti_shadow": anti_shadow,
            "ambiguity": ambiguity,
            "score_cliff": score_cliff,
            "support_quorum": support_quorum,
            "pack_value": pack_value,
        }

    @staticmethod
    def _normalize_weight_map(weights: Dict[str, float]) -> Dict[str, float]:
        sanitized = {key: max(float(value), 0.0) for key, value in weights.items()}
        total = float(sum(sanitized.values()))
        if total <= 1e-8:
            count = max(len(sanitized), 1)
            return {key: 1.0 / float(count) for key in sanitized}
        return {key: value / total for key, value in sanitized.items()}

    def _controller_features(
        self,
        *,
        coverage_matrix: np.ndarray,
        query_token_weights: np.ndarray,
        relevance_scores: np.ndarray,
        retrieval_signals: Dict[str, np.ndarray],
        redundancy_matrix: np.ndarray,
        payload_signals: Dict[str, np.ndarray],
        token_costs: np.ndarray,
        max_tokens: int,
        candidate_count: int,
        query_token_count: int,
    ) -> Dict[str, float]:
        uncovered_mass = 0.0
        if coverage_matrix.size and query_token_weights.size:
            uncovered_mass = float(
                np.sum(query_token_weights * (1.0 - coverage_matrix.max(axis=1)))
            )
        redundancy_density = 0.0
        if redundancy_matrix.size:
            mask = ~np.eye(redundancy_matrix.shape[0], dtype=bool)
            if np.any(mask):
                redundancy_density = float(np.mean(redundancy_matrix[mask]))
        avg_chunk_cost = float(np.mean(token_costs)) if token_costs.size else 0.0
        min_chunk_cost = float(np.min(token_costs)) if token_costs.size else 0.0
        support_density = float(
            np.mean(relevance_scores / np.sqrt(np.clip(token_costs.astype(np.float32, copy=False), 1.0, None)))
        ) if token_costs.size and relevance_scores.size else 0.0
        metadata_coverage = 0.0
        if candidate_count > 0:
            metadata_coverage = float(
                np.mean(
                    (
                        retrieval_signals["base"]
                        + retrieval_signals["rrf"]
                        + retrieval_signals["dense"]
                        + retrieval_signals["sparse"]
                        + retrieval_signals["rerank"]
                        + retrieval_signals["utility"]
                        + retrieval_signals["teacher_utility"]
                    )
                    > 0.0
                )
            )
        estimated_pack = float(query_token_count) * max(avg_chunk_cost, min_chunk_cost, 1.0)
        slack_ratio = 0.0
        if max_tokens > 0:
            slack_ratio = float(max(max_tokens - min(max_tokens, estimated_pack), 0.0) / float(max_tokens))
        return {
            "candidate_count": float(candidate_count),
            "query_token_count": float(query_token_count),
            "max_tokens": float(max_tokens),
            "avg_chunk_cost": avg_chunk_cost,
            "min_chunk_cost": min_chunk_cost,
            "slack_ratio": slack_ratio,
            "budget_to_avg_chunk_ratio": float(max_tokens / max(avg_chunk_cost, 1.0)) if max_tokens > 0 else 0.0,
            "pack_density": support_density,
            "metadata_coverage": metadata_coverage,
            "uncovered_mass": float(uncovered_mass),
            "retrieval_agreement_mean": float(np.mean(retrieval_signals["agreement"])) if retrieval_signals["agreement"].size else 0.0,
            "retrieval_disagreement_mean": float(np.mean(retrieval_signals["disagreement"])) if retrieval_signals["disagreement"].size else 0.0,
            "rerank_signal_mean": float(np.mean(retrieval_signals["rerank"])) if retrieval_signals["rerank"].size else 0.0,
            "rerank_coverage": float(np.mean(retrieval_signals["rerank_presence"])) if retrieval_signals["rerank_presence"].size else 0.0,
            "rerank_agreement_mean": float(np.mean(retrieval_signals["rerank_agreement"])) if retrieval_signals["rerank_agreement"].size else 0.0,
            "utility_signal_mean": float(np.mean(retrieval_signals["utility"])) if retrieval_signals["utility"].size else 0.0,
            "teacher_utility_signal_mean": float(np.mean(retrieval_signals["teacher_utility"])) if retrieval_signals["teacher_utility"].size else 0.0,
            "contradiction_signal_mean": float(np.mean(retrieval_signals["contradiction"])) if retrieval_signals["contradiction"].size else 0.0,
            "redundancy_density": float(redundancy_density),
            "frontier_gain_mean": float(np.mean(payload_signals["frontier_gain"])) if payload_signals["frontier_gain"].size else 0.0,
            "frontier_gain_p90": float(np.quantile(payload_signals["frontier_gain"], 0.9)) if payload_signals["frontier_gain"].size else 0.0,
            "ambiguity_mean": float(np.mean(payload_signals["ambiguity"])) if payload_signals["ambiguity"].size else 0.0,
            "ambiguity_p90": float(np.quantile(payload_signals["ambiguity"], 0.9)) if payload_signals["ambiguity"].size else 0.0,
        }

    def _apply_controller_policy(
        self,
        *,
        solver_config_payload: Dict[str, Any],
        blend_weights: Dict[str, Dict[str, float]],
        controller_features: Dict[str, float],
        metadata: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Any]]:
        controller_defaults = policy["controller"]
        controller_spec = (
            solver_config_payload.get("controller")
            or metadata.get("controller")
            or {"name": controller_defaults.get("name", "auto_v1")}
        )
        if controller_spec in (False, None, "disabled"):
            return solver_config_payload, blend_weights, {"applied": False, "name": "disabled"}
        if isinstance(controller_spec, str):
            controller_spec = {"name": controller_spec}

        name = str(controller_spec.get("name", "auto_v1"))
        updated_solver = dict(solver_config_payload)
        updated_blend = {branch: dict(weights) for branch, weights in blend_weights.items()}

        if name in {"auto_v1", "post_rerank_v1"}:
            ambiguity_pressure = (
                controller_features["uncovered_mass"]
                + controller_features["retrieval_disagreement_mean"]
                + controller_defaults["ambiguity_p90_weight"] * controller_features["ambiguity_p90"]
            )
            if ambiguity_pressure > controller_defaults["high_ambiguity_threshold"] or controller_features["candidate_count"] >= controller_defaults["large_candidate_count"]:
                updated_solver["iterations"] = max(int(updated_solver.get("iterations", 100)), int(controller_defaults["high_iterations"]))
                updated_solver["enable_destroy_repair"] = True
                updated_solver["enable_path_relinking"] = True
                updated_solver["exact_window_time_ms"] = max(
                    int(updated_solver.get("exact_window_time_ms", 25)),
                    int(controller_defaults["high_exact_window_time_ms"]),
                )
            if controller_features["candidate_count"] <= controller_defaults["small_candidate_count"] and controller_features["uncovered_mass"] < controller_defaults["low_uncovered_threshold"]:
                updated_solver["iterations"] = min(int(updated_solver.get("iterations", 100)), int(controller_defaults["low_iterations"]))
                updated_solver["exact_window_time_ms"] = min(
                    int(updated_solver.get("exact_window_time_ms", 25)),
                    int(controller_defaults["low_exact_window_time_ms"]),
                )
                updated_solver["exact_window_size"] = min(
                    int(updated_solver.get("exact_window_size", 14)),
                    int(controller_defaults["small_exact_window_size"]),
                )
            if controller_features["uncovered_mass"] > controller_defaults["high_uncovered_threshold"]:
                updated_solver["mu"] = max(float(updated_solver.get("mu", 1.0)), float(controller_defaults["uncovered_mu_floor"]))
                updated_blend["relevance"]["hard_coverage"] += controller_defaults["hard_coverage_relevance_boost"]
                updated_blend["fulfilment"]["frontier_gain"] += controller_defaults["frontier_gain_fulfilment_boost"]
                updated_blend["fulfilment"]["support_quorum"] += controller_defaults["support_quorum_fulfilment_boost"]
            if controller_features["retrieval_agreement_mean"] > controller_features["retrieval_disagreement_mean"] + controller_defaults["retrieval_agreement_margin"]:
                updated_blend["relevance"]["retrieval"] += controller_defaults["retrieval_relevance_boost"]
                updated_blend["relevance"]["geometry"] = max(
                    updated_blend["relevance"]["geometry"] - controller_defaults["geometry_relevance_penalty"],
                    0.0,
                )
            if controller_features["metadata_coverage"] < controller_defaults["metadata_coverage_threshold"]:
                updated_blend["relevance"]["retrieval"] = max(
                    updated_blend["relevance"]["retrieval"] - controller_defaults["metadata_retrieval_penalty"],
                    0.0,
                )
                updated_blend["relevance"]["geometry"] += controller_defaults["metadata_geometry_boost"]
                updated_blend["relevance"]["lexical"] += controller_defaults["metadata_lexical_boost"]
            if controller_features["redundancy_density"] > controller_defaults["redundancy_density_threshold"]:
                updated_solver["lambda_"] = max(
                    float(updated_solver.get("lambda_", 0.5)),
                    float(controller_defaults["redundancy_lambda_floor"]),
                )
                updated_blend["auxiliary"]["uniqueness"] += controller_defaults["uniqueness_aux_boost"]
            if controller_features["slack_ratio"] > controller_defaults["slack_ratio_threshold"] and controller_features["frontier_gain_p90"] > controller_defaults["frontier_gain_p90_threshold"]:
                updated_solver["mu"] = max(float(updated_solver.get("mu", 1.0)), float(controller_defaults["slack_mu_floor"]))
                updated_solver["lambda_"] = min(
                    float(updated_solver.get("lambda_", 0.5)),
                    float(controller_defaults["slack_lambda_cap"]),
                )
                updated_blend["fulfilment"]["pack_value"] += controller_defaults["pack_value_fulfilment_boost"]
                updated_blend["auxiliary"]["pack_value"] += controller_defaults["pack_value_aux_boost"]
                updated_blend["auxiliary"]["anti_shadow"] += controller_defaults["anti_shadow_aux_boost"]
            if controller_features["pack_density"] < controller_defaults["pack_density_threshold"] and controller_features["budget_to_avg_chunk_ratio"] >= controller_defaults["budget_avg_chunk_ratio_threshold"]:
                updated_solver["iterations"] = max(
                    int(updated_solver.get("iterations", 100)),
                    int(controller_defaults["very_high_iterations"]),
                )
                updated_solver["exact_window_size"] = max(
                    int(updated_solver.get("exact_window_size", 14)),
                    int(controller_defaults["large_exact_window_size"]),
                )
            if (
                controller_features["rerank_coverage"] >= controller_defaults["rerank_coverage_threshold"]
                and controller_features["rerank_agreement_mean"] >= controller_defaults["rerank_agreement_threshold"]
            ):
                updated_solver["iterations"] = max(
                    int(updated_solver.get("iterations", 100)),
                    int(controller_defaults["post_rerank_iterations_floor"]),
                )
                updated_solver["lambda_"] = max(
                    float(updated_solver.get("lambda_", 0.5)),
                    float(controller_defaults["post_rerank_lambda_floor"]),
                )
                updated_blend["relevance"]["retrieval"] += controller_defaults["post_rerank_retrieval_boost"]
                updated_blend["relevance"]["geometry"] = max(
                    updated_blend["relevance"]["geometry"] - controller_defaults["post_rerank_geometry_penalty"],
                    0.0,
                )
                updated_blend["fulfilment"]["pack_value"] += controller_defaults["post_rerank_pack_value_boost"]
            if controller_features.get("utility_signal_mean", 0.0) >= controller_defaults.get("utility_signal_threshold", 1.0):
                updated_blend["relevance"]["retrieval"] += controller_defaults.get("utility_relevance_boost", 0.0)
                updated_blend["fulfilment"]["support_quorum"] += controller_defaults.get("utility_support_quorum_boost", 0.0)
            if controller_features.get("contradiction_signal_mean", 0.0) >= controller_defaults.get("contradiction_signal_threshold", 1.0):
                updated_solver["lambda_"] = max(
                    float(updated_solver.get("lambda_", 0.5)),
                    float(controller_defaults.get("contradiction_lambda_floor", 0.5)),
                )
                updated_solver["iterations"] = max(
                    int(updated_solver.get("iterations", 100)),
                    int(controller_defaults.get("contradiction_iterations_floor", 100)),
                )

        for key, value in dict(controller_spec.get("solver_overrides", {})).items():
            updated_solver[key] = value
        for branch, overrides in dict(controller_spec.get("blend_overrides", {})).items():
            if branch not in updated_blend:
                continue
            for key, value in dict(overrides).items():
                if key in updated_blend[branch]:
                    updated_blend[branch][key] = float(value)

        normalized_blend = {
            branch: self._normalize_weight_map(weights) for branch, weights in updated_blend.items()
        }
        return updated_solver, normalized_blend, {
            "applied": True,
            "name": name,
            "features": controller_features,
        }

    @staticmethod
    def _default_blend_weights(policy: Dict[str, Any] | None = None) -> Dict[str, Dict[str, float]]:
        resolved = policy or _default_optimizer_policy()
        return {
            branch: dict(weights)
            for branch, weights in dict(resolved.get("blend_weights", {})).items()
        }

    def _build_feature_context(
        self,
        *,
        query_text: str,
        candidates: Sequence[OptimizerCandidate],
        coverage_matrix: np.ndarray,
        query_token_weights: np.ndarray,
        relevance_scores: np.ndarray,
        fulfilment_scores: np.ndarray,
        auxiliary_scores: np.ndarray,
        redundancy_matrix: np.ndarray | None = None,
        token_costs: np.ndarray | None = None,
        max_tokens: int = 0,
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        geometry = self._normalize_feature(relevance_scores)
        fulfilment = self._normalize_feature(fulfilment_scores)
        auxiliary = self._normalize_feature(auxiliary_scores)
        retrieval = self._retrieval_signal_scores(candidates, policy=policy)
        lexical = self._lexical_overlap_scores(query_text, candidates)
        novelty = self._embedding_novelty_scores(coverage_matrix, query_token_weights)
        cluster = self._cluster_novelty_scores(candidates)
        effective_redundancy = np.asarray(
            redundancy_matrix if redundancy_matrix is not None else np.zeros((len(candidates), len(candidates))),
            dtype=np.float32,
        )
        effective_token_costs = np.asarray(
            token_costs if token_costs is not None else np.ones((len(candidates),), dtype=np.float32),
            dtype=np.float32,
        )
        payload_signals = self._payload_grounded_signal_scores(
            coverage_matrix,
            query_token_weights,
            relevance_scores,
            retrieval,
            effective_redundancy,
            effective_token_costs,
            policy,
        )

        controller_features = self._controller_features(
            coverage_matrix=coverage_matrix,
            query_token_weights=query_token_weights,
            relevance_scores=relevance_scores,
            retrieval_signals=retrieval,
            redundancy_matrix=effective_redundancy,
            payload_signals=payload_signals,
            token_costs=effective_token_costs,
            max_tokens=max_tokens,
            candidate_count=len(candidates),
            query_token_count=int(coverage_matrix.shape[0]) if coverage_matrix.ndim == 2 else 0,
        )

        return {
            "geometry": geometry,
            "fulfilment": fulfilment,
            "auxiliary": auxiliary,
            "retrieval": retrieval,
            "lexical": lexical,
            "novelty": novelty,
            "cluster": cluster,
            "payload_signals": payload_signals,
            "controller_features": controller_features,
        }

    def _compose_candidate_features(
        self,
        *,
        query_text: str,
        candidates: Sequence[OptimizerCandidate],
        coverage_matrix: np.ndarray,
        query_token_weights: np.ndarray,
        relevance_scores: np.ndarray,
        fulfilment_scores: np.ndarray,
        auxiliary_scores: np.ndarray,
        redundancy_matrix: np.ndarray | None = None,
        blend_weights: Dict[str, Dict[str, float]] | None = None,
        feature_context: Dict[str, Any] | None = None,
        token_costs: np.ndarray | None = None,
        max_tokens: int = 0,
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = feature_context or self._build_feature_context(
            query_text=query_text,
            candidates=candidates,
            coverage_matrix=coverage_matrix,
            query_token_weights=query_token_weights,
            relevance_scores=relevance_scores,
            fulfilment_scores=fulfilment_scores,
            auxiliary_scores=auxiliary_scores,
            redundancy_matrix=redundancy_matrix,
            token_costs=token_costs,
            max_tokens=max_tokens,
            policy=policy,
        )
        weights = blend_weights or self._default_blend_weights(policy)

        composed_relevance = self._normalize_feature(
            weights["relevance"]["geometry"] * context["geometry"]
            + weights["relevance"]["retrieval"] * context["retrieval"]["composite"]
            + weights["relevance"]["lexical"] * context["lexical"]
            + weights["relevance"]["winner_mass"] * context["novelty"]["winner_mass"]
            + weights["relevance"]["hard_coverage"] * context["payload_signals"]["hard_coverage"]
            + weights["relevance"]["uniqueness"] * context["payload_signals"]["uniqueness"]
            + weights["relevance"]["support_quorum"] * context["payload_signals"]["support_quorum"]
        )
        composed_fulfilment = self._normalize_feature(
            weights["fulfilment"]["fulfilment"] * context["fulfilment"]
            + weights["fulfilment"]["margin"] * context["novelty"]["margin"]
            + weights["fulfilment"]["winner_mass"] * context["novelty"]["winner_mass"]
            + weights["fulfilment"]["frontier_gain"] * context["payload_signals"]["frontier_gain"]
            + weights["fulfilment"]["support_quorum"] * context["payload_signals"]["support_quorum"]
            + weights["fulfilment"]["pack_value"] * context["payload_signals"]["pack_value"]
        )
        composed_auxiliary = self._normalize_feature(
            weights["auxiliary"]["auxiliary"] * context["auxiliary"]
            + weights["auxiliary"]["lexical"] * context["lexical"]
            + weights["auxiliary"]["specificity"] * context["novelty"]["specificity"]
            + weights["auxiliary"]["cluster"] * context["cluster"]
            + weights["auxiliary"]["uniqueness"] * context["payload_signals"]["uniqueness"]
            + weights["auxiliary"]["ambiguity"] * context["payload_signals"]["ambiguity"]
            + weights["auxiliary"]["anti_shadow"] * context["payload_signals"]["anti_shadow"]
            + weights["auxiliary"]["pack_value"] * context["payload_signals"]["pack_value"]
        )

        return {
            "relevance_scores": composed_relevance,
            "fulfilment_scores": composed_fulfilment,
            "auxiliary_scores": composed_auxiliary,
            "controller_features": context["controller_features"],
            "summary": {
                "mean_geometry_signal": float(np.mean(context["geometry"])) if context["geometry"].size else 0.0,
                "mean_retrieval_signal": float(np.mean(context["retrieval"]["composite"])) if context["retrieval"]["composite"].size else 0.0,
                "mean_rerank_signal": float(np.mean(context["retrieval"]["rerank"])) if context["retrieval"]["rerank"].size else 0.0,
                "mean_rerank_agreement_signal": float(np.mean(context["retrieval"]["rerank_agreement"])) if context["retrieval"]["rerank_agreement"].size else 0.0,
                "mean_utility_signal": float(np.mean(context["retrieval"]["utility"])) if context["retrieval"]["utility"].size else 0.0,
                "mean_teacher_utility_signal": float(np.mean(context["retrieval"]["teacher_utility"])) if context["retrieval"]["teacher_utility"].size else 0.0,
                "mean_contradiction_signal": float(np.mean(context["retrieval"]["contradiction"])) if context["retrieval"]["contradiction"].size else 0.0,
                "rerank_coverage": float(np.mean(context["retrieval"]["rerank_presence"])) if context["retrieval"]["rerank_presence"].size else 0.0,
                "mean_lexical_overlap": float(np.mean(context["lexical"])) if context["lexical"].size else 0.0,
                "mean_margin_signal": float(np.mean(context["novelty"]["margin"])) if context["novelty"]["margin"].size else 0.0,
                "mean_cluster_novelty": float(np.mean(context["cluster"])) if context["cluster"].size else 0.0,
                "mean_hard_coverage_signal": float(np.mean(context["payload_signals"]["hard_coverage"])) if context["payload_signals"]["hard_coverage"].size else 0.0,
                "mean_frontier_gain_signal": float(np.mean(context["payload_signals"]["frontier_gain"])) if context["payload_signals"]["frontier_gain"].size else 0.0,
                "mean_support_quorum_signal": float(np.mean(context["payload_signals"]["support_quorum"])) if context["payload_signals"]["support_quorum"].size else 0.0,
                "mean_pack_value_signal": float(np.mean(context["payload_signals"]["pack_value"])) if context["payload_signals"]["pack_value"].size else 0.0,
                "mean_uniqueness_signal": float(np.mean(context["payload_signals"]["uniqueness"])) if context["payload_signals"]["uniqueness"].size else 0.0,
                "mean_ambiguity_signal": float(np.mean(context["payload_signals"]["ambiguity"])) if context["payload_signals"]["ambiguity"].size else 0.0,
            },
        }

    def _coverage_and_relevance(
        self,
        query_vectors: np.ndarray,
        candidate_vectors: Sequence[np.ndarray],
        *,
        metadata: Dict[str, Any] | None,
        policy: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not candidate_vectors:
            return (
                np.zeros((query_vectors.shape[0], 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        coverage_cpu = compute_maxsim_token_coverage_matrix(
            query_vectors,
            candidate_vectors,
            device=self.device,
        )
        row_gains = self._metadata_coverage_row_gains(
            metadata,
            expected_rows=int(coverage_cpu.shape[0]),
        )
        if row_gains is not None:
            coverage_cpu = np.clip(
                coverage_cpu.astype(np.float32, copy=False) * row_gains[:, None],
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
        token_weights = self._query_token_weights(coverage_cpu, policy=policy)
        metadata_token_weights = self._metadata_query_token_weights(
            metadata,
            expected_rows=int(coverage_cpu.shape[0]),
        )
        if metadata_token_weights is not None:
            blend = float(policy["token_weights"].get("metadata_blend", 0.60))
            token_weights = (
                (1.0 - blend) * token_weights.astype(np.float32, copy=False)
                + blend * metadata_token_weights.astype(np.float32, copy=False)
            ).astype(np.float32, copy=False)
            token_weights /= float(max(token_weights.sum(), 1e-8))
        relevance = (coverage_cpu * token_weights[:, None]).sum(axis=0).astype(np.float32, copy=False)
        fulfilment = relevance.copy()
        return coverage_cpu, token_weights, relevance, fulfilment

    def _redundancy_matrix(
        self,
        candidates: Sequence[OptimizerCandidate],
        candidate_vectors: Sequence[np.ndarray],
        *,
        coverage_matrix: np.ndarray,
        policy: Dict[str, Any],
    ) -> np.ndarray:
        if not candidate_vectors:
            return np.zeros((0, 0), dtype=np.float32)
        centroids = np.stack([_centroid(vector) for vector in candidate_vectors]).astype(np.float32)
        centroids = _normalize_rows(centroids)
        centroid_similarity = np.clip(centroids @ centroids.T, 0.0, 1.0).astype(np.float32, copy=False)
        np.fill_diagonal(centroid_similarity, 0.0)
        n = len(candidates)
        lexical_similarity = np.zeros((n, n), dtype=np.float32)
        containment_similarity = np.zeros((n, n), dtype=np.float32)
        source_similarity = np.zeros((n, n), dtype=np.float32)
        term_sets = [self._text_terms(candidate.text) for candidate in candidates]
        source_ids = [self._source_identity(candidate.metadata) for candidate in candidates]
        for i in range(n):
            for j in range(i + 1, n):
                a = term_sets[i]
                b = term_sets[j]
                inter = len(a & b) if a and b else 0
                union = len(a | b) if a or b else 0
                lexical = float(inter / union) if union else 0.0
                containment = 0.0
                if inter > 0:
                    containment = max(
                        float(inter / float(max(len(a), 1))),
                        float(inter / float(max(len(b), 1))),
                    )
                lexical_similarity[i, j] = lexical_similarity[j, i] = lexical
                containment_similarity[i, j] = containment_similarity[j, i] = containment
                same_source = (
                    source_ids[i] is not None
                    and source_ids[j] is not None
                    and source_ids[i] == source_ids[j]
                )
                if same_source:
                    source_similarity[i, j] = source_similarity[j, i] = 1.0
        aspect_similarity = np.zeros((n, n), dtype=np.float32)
        if coverage_matrix.ndim == 2 and coverage_matrix.shape[0] > 1:
            aspect_profiles = _normalize_rows(coverage_matrix.T.astype(np.float32, copy=False))
            aspect_similarity = np.clip(
                aspect_profiles @ aspect_profiles.T,
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
            np.fill_diagonal(aspect_similarity, 0.0)
        redundancy_policy = self._normalize_weight_map(policy.get("redundancy", {}))
        similarity = (
            redundancy_policy.get("centroid", 0.0) * centroid_similarity
            + redundancy_policy.get("lexical", 0.0) * lexical_similarity
            + redundancy_policy.get("containment", 0.0) * containment_similarity
            + redundancy_policy.get("aspect", 0.0) * aspect_similarity
            + redundancy_policy.get("same_source", 0.0) * source_similarity
        ).astype(np.float32, copy=False)
        np.fill_diagonal(similarity, 0.0)
        return np.clip(similarity, 0.0, 1.0).astype(np.float32, copy=False)

    def _optimize_decoded(
        self,
        normalized_request: OptimizerRequest,
        query_vectors: np.ndarray,
        candidate_vectors: Sequence[np.ndarray],
        *,
        solver: Any | None = None,
        payload_bytes: int | None = None,
    ) -> Dict[str, Any]:
        candidate_count = len(candidate_vectors)
        if candidate_count == 0:
            execution_mode = self._execution_mode(False, "cpu_reference")
            return {
                "selected_ids": [],
                "selected_chunks": [],
                "solver_output": {
                    "selected_indices": [],
                    "objective_score": 0.0,
                    "fulfilment_total": 0.0,
                    "num_selected": 0,
                    "solve_time_ms": 0.0,
                    "constraints_satisfied": True,
                    "constraint_violations": [],
                    "exact_window_used": False,
                    "exact_window_core_size": 0,
                    "exact_window_nodes": 0,
                    "exact_window_exhaustive": False,
                    "exact_window_gap": 0.0,
                    "exact_window_fixed_in": 0,
                    "exact_window_fixed_out": 0,
                },
                "backend_kind": execution_mode,
                "solver_backend_kind": "cpu_reference",
                "feature_summary": {"candidate_count": 0, "payload_bytes": 0},
            }

        from latence_solver import SolverConfig, SolverConstraints, TabuSearchSolver

        solver_use_gpu = bool(normalized_request.solver_config.get("use_gpu", self._target_solver_use_gpu()))
        lambda_value = normalized_request.solver_config.get(
            "lambda_",
            normalized_request.solver_config.get("lambda", 0.5),
        )
        solver_config_payload: Dict[str, Any] = {
            "mu": float(normalized_request.solver_config.get("mu", 1.0)),
            "alpha": float(normalized_request.solver_config.get("alpha", 1.0)),
            "beta": float(normalized_request.solver_config.get("beta", 0.3)),
            "gamma": float(normalized_request.solver_config.get("gamma", 0.2)),
            "delta": float(normalized_request.solver_config.get("delta", 0.1)),
            "epsilon": float(normalized_request.solver_config.get("epsilon", 0.0)),
            "support_secondary_discount": float(
                normalized_request.solver_config.get("support_secondary_discount", 0.35)
            ),
            "support_quorum_bonus": float(
                normalized_request.solver_config.get("support_quorum_bonus", 0.18)
            ),
            "support_quorum_threshold": float(
                normalized_request.solver_config.get("support_quorum_threshold", 0.55)
            ),
            "support_quorum_cap": int(
                normalized_request.solver_config.get("support_quorum_cap", 4)
            ),
            "lambda_": float(lambda_value),
            "iterations": int(normalized_request.solver_config.get("iterations", 100)),
            "tabu_tenure": int(normalized_request.solver_config.get("tabu_tenure", 10)),
            "early_stopping_patience": int(
                normalized_request.solver_config.get("early_stopping_patience", 20)
            ),
            "use_gpu": solver_use_gpu,
            "random_seed": normalized_request.solver_config.get("random_seed"),
            "enable_gpu_move_evaluation": bool(
                normalized_request.solver_config.get("enable_gpu_move_evaluation", True)
            ),
            "enable_path_relinking": bool(
                normalized_request.solver_config.get("enable_path_relinking", True)
            ),
            "enable_destroy_repair": bool(
                normalized_request.solver_config.get("enable_destroy_repair", True)
            ),
            "enable_reactive_tenure": bool(
                normalized_request.solver_config.get("enable_reactive_tenure", True)
            ),
            "enable_exact_window": bool(
                normalized_request.solver_config.get("enable_exact_window", True)
            ),
            "exact_window_size": int(
                normalized_request.solver_config.get("exact_window_size", 14)
            ),
            "exact_window_time_ms": int(
                normalized_request.solver_config.get("exact_window_time_ms", 25)
            ),
            "controller": normalized_request.solver_config.get(
                "controller", normalized_request.metadata.get("controller")
            ),
        }
        optimizer_policy = self._resolve_optimizer_policy(
            normalized_request.solver_config.get("optimizer_policy")
            or normalized_request.metadata.get("optimizer_policy")
        )
        max_tokens = int(normalized_request.constraints.get("max_tokens", 8192))
        constraints = SolverConstraints(
            max_tokens=max_tokens,
            min_tokens=int(normalized_request.constraints.get("min_tokens", 0)),
            min_chunks=int(normalized_request.constraints.get("min_chunks", 1)),
            max_chunks=int(normalized_request.constraints.get("max_chunks", min(50, candidate_count))),
            max_per_cluster=int(normalized_request.constraints.get("max_per_cluster", 3)),
            must_include_roles=normalized_request.constraints.get("must_include_roles"),
            excluded_chunks=normalized_request.constraints.get("excluded_chunks"),
            required_chunks=normalized_request.constraints.get("required_chunks"),
        )

        query_vectors, query_bundle_summary = self._derive_query_bundle(
            query_vectors=query_vectors,
            query_text=normalized_request.query_text,
            candidates=normalized_request.candidates,
            candidate_vectors=candidate_vectors,
            policy=optimizer_policy,
        )
        self.warmup(int(query_vectors.shape[1]))

        precompute_start = time.perf_counter()
        coverage_matrix, query_token_weights, relevance_scores, fulfilment_scores = self._coverage_and_relevance(
            query_vectors,
            candidate_vectors,
            metadata=normalized_request.metadata,
            policy=optimizer_policy,
        )
        redundancy_matrix = self._redundancy_matrix(
            normalized_request.candidates,
            candidate_vectors,
            coverage_matrix=coverage_matrix,
            policy=optimizer_policy,
        )
        centroid_vectors = np.stack([_centroid(vector) for vector in candidate_vectors]).astype(np.float32)
        token_costs = np.asarray(
            [candidate.token_count for candidate in normalized_request.candidates],
            dtype=np.uint32,
        )
        density_scores = np.asarray(
            [candidate.fact_density for candidate in normalized_request.candidates],
            dtype=np.float32,
        )
        centrality_scores = np.asarray(
            [candidate.centrality_score for candidate in normalized_request.candidates],
            dtype=np.float32,
        )
        recency_scores = np.asarray(
            [candidate.recency_score for candidate in normalized_request.candidates],
            dtype=np.float32,
        )
        auxiliary_scores = np.asarray(
            [candidate.auxiliary_score for candidate in normalized_request.candidates],
            dtype=np.float32,
        )
        feature_context = self._build_feature_context(
            query_text=normalized_request.query_text,
            candidates=normalized_request.candidates,
            coverage_matrix=coverage_matrix,
            query_token_weights=query_token_weights,
            relevance_scores=relevance_scores,
            fulfilment_scores=fulfilment_scores,
            auxiliary_scores=auxiliary_scores,
            redundancy_matrix=redundancy_matrix,
            token_costs=token_costs.astype(np.float32, copy=False),
            max_tokens=max_tokens,
            policy=optimizer_policy,
        )
        blend_weights = self._default_blend_weights(optimizer_policy)
        solver_config_payload, blend_weights, controller_summary = self._apply_controller_policy(
            solver_config_payload=solver_config_payload,
            blend_weights=blend_weights,
            controller_features=feature_context["controller_features"],
            metadata=normalized_request.metadata,
            policy=optimizer_policy,
        )
        edge_case_fallback_override = (
            normalized_request.solver_config["edge_case_fallback"]
            if "edge_case_fallback" in normalized_request.solver_config
            else normalized_request.metadata.get("edge_case_fallback")
        )
        edge_case_fallback_config = self._resolve_edge_case_fallback_config(
            edge_case_fallback_override
        )
        edge_case_fallback_summary = self._edge_case_fallback_decision(
            controller_features=feature_context["controller_features"],
            fallback_config=edge_case_fallback_config,
        )
        solver_config_payload, blend_weights, edge_case_fallback_summary = self._apply_edge_case_fallback_profile(
            solver_config_payload=solver_config_payload,
            blend_weights=blend_weights,
            fallback_config=edge_case_fallback_config,
            fallback_summary=edge_case_fallback_summary,
        )
        composed_features = self._compose_candidate_features(
            query_text=normalized_request.query_text,
            candidates=normalized_request.candidates,
            coverage_matrix=coverage_matrix,
            query_token_weights=query_token_weights,
            relevance_scores=relevance_scores,
            fulfilment_scores=fulfilment_scores,
            auxiliary_scores=auxiliary_scores,
            redundancy_matrix=redundancy_matrix,
            blend_weights=blend_weights,
            feature_context=feature_context,
            token_costs=token_costs.astype(np.float32, copy=False),
            max_tokens=max_tokens,
            policy=optimizer_policy,
        )
        relevance_scores = composed_features["relevance_scores"]
        fulfilment_scores = composed_features["fulfilment_scores"]
        auxiliary_scores = composed_features["auxiliary_scores"]
        roles = np.asarray(
            [
                {
                    "definition": 0,
                    "example": 1,
                    "evidence": 2,
                    "conclusion": 3,
                    "risk": 4,
                    "constraint": 5,
                    "data_table": 6,
                    "procedure": 7,
                }.get(candidate.rhetorical_role.lower(), 255)
                for candidate in normalized_request.candidates
            ],
            dtype=np.uint8,
        )
        cluster_ids = np.asarray(
            [candidate.cluster_id if candidate.cluster_id is not None else -1 for candidate in normalized_request.candidates],
            dtype=np.int32,
        )

        if payload_bytes is None:
            payload_bytes = normalized_request.query_vectors.approx_bytes() + sum(
                candidate.vectors.approx_bytes() for candidate in normalized_request.candidates
            )

        precompute_ms = (time.perf_counter() - precompute_start) * 1000.0
        solver_config = SolverConfig(
            mu=float(solver_config_payload.get("mu", 1.0)),
            alpha=float(solver_config_payload.get("alpha", 1.0)),
            beta=float(solver_config_payload.get("beta", 0.3)),
            gamma=float(solver_config_payload.get("gamma", 0.2)),
            delta=float(solver_config_payload.get("delta", 0.1)),
            epsilon=float(solver_config_payload.get("epsilon", 0.0)),
            support_secondary_discount=float(
                solver_config_payload.get("support_secondary_discount", 0.35)
            ),
            support_quorum_bonus=float(
                solver_config_payload.get("support_quorum_bonus", 0.18)
            ),
            support_quorum_threshold=float(
                solver_config_payload.get("support_quorum_threshold", 0.55)
            ),
            support_quorum_cap=int(
                solver_config_payload.get("support_quorum_cap", 4)
            ),
            lambda_=float(solver_config_payload.get("lambda_", 0.5)),
            iterations=int(solver_config_payload.get("iterations", 100)),
            tabu_tenure=int(solver_config_payload.get("tabu_tenure", 10)),
            early_stopping_patience=int(solver_config_payload.get("early_stopping_patience", 20)),
            use_gpu=bool(solver_config_payload.get("use_gpu", solver_use_gpu)),
            random_seed=solver_config_payload.get("random_seed"),
            enable_gpu_move_evaluation=bool(
                solver_config_payload.get("enable_gpu_move_evaluation", True)
            ),
            enable_path_relinking=bool(solver_config_payload.get("enable_path_relinking", True)),
            enable_destroy_repair=bool(solver_config_payload.get("enable_destroy_repair", True)),
            enable_reactive_tenure=bool(solver_config_payload.get("enable_reactive_tenure", True)),
            enable_exact_window=bool(solver_config_payload.get("enable_exact_window", True)),
            exact_window_size=int(solver_config_payload.get("exact_window_size", 14)),
            exact_window_time_ms=int(solver_config_payload.get("exact_window_time_ms", 25)),
        )
        solver_instance = solver or TabuSearchSolver(solver_config)
        solver_backend_kind = (
            solver_instance.backend_kind() if hasattr(solver_instance, "backend_kind") else "unknown"
        )
        execution_mode = self._execution_mode(solver_use_gpu, solver_backend_kind)

        solve_start = time.perf_counter()
        solver_output = solver_instance.solve_precomputed_numpy(
            centroid_vectors,
            token_costs,
            density_scores,
            centrality_scores,
            recency_scores,
            auxiliary_scores,
            roles,
            cluster_ids,
            relevance_scores,
            similarity_matrix=redundancy_matrix,
            fulfilment_scores=fulfilment_scores,
            coverage_matrix=coverage_matrix,
            query_token_weights=query_token_weights,
            query_embedding=_centroid(query_vectors).astype(np.float32),
            constraints=constraints,
        )
        solve_elapsed_ms = (time.perf_counter() - solve_start) * 1000.0

        selected_indices = [int(index) for index in solver_output.selected_indices]
        selected_candidates = [normalized_request.candidates[index] for index in selected_indices]
        solver_output_payload = (
            solver_output.to_dict() if hasattr(solver_output, "to_dict") else {
                "selected_indices": selected_indices,
                "objective_score": float(solver_output.objective_score),
                "fulfilment_total": float(getattr(solver_output, "fulfilment_total", 0.0)),
                "num_selected": int(solver_output.num_selected),
                "solve_time_ms": float(solver_output.solve_time_ms),
                "constraints_satisfied": bool(solver_output.constraints_satisfied),
                "constraint_violations": list(solver_output.constraint_violations),
                "exact_window_used": bool(getattr(solver_output, "exact_window_used", False)),
                "exact_window_core_size": int(getattr(solver_output, "exact_window_core_size", 0)),
                "exact_window_nodes": int(getattr(solver_output, "exact_window_nodes", 0)),
                "exact_window_exhaustive": bool(getattr(solver_output, "exact_window_exhaustive", False)),
                "exact_window_gap": float(getattr(solver_output, "exact_window_gap", 0.0)),
                "exact_window_fixed_in": int(getattr(solver_output, "exact_window_fixed_in", 0)),
                "exact_window_fixed_out": int(getattr(solver_output, "exact_window_fixed_out", 0)),
            }
        )
        solver_output_payload["edge_case_fallback_applied"] = bool(
            edge_case_fallback_summary.get("applied", False)
        )
        solver_output_payload["edge_case_fallback"] = edge_case_fallback_summary

        return {
            "selected_ids": [candidate.chunk_id for candidate in selected_candidates],
            "selected_chunks": [
                {
                    "chunk_id": candidate.chunk_id,
                    "text": candidate.text,
                    "token_count": candidate.token_count,
                    "metadata": candidate.metadata,
                }
                for candidate in selected_candidates
            ],
            "solver_output": solver_output_payload,
            "backend_kind": execution_mode,
            "solver_backend_kind": solver_backend_kind,
            "feature_summary": {
                "candidate_count": candidate_count,
                "query_tokens": int(query_vectors.shape[0]),
                "vector_dim": int(query_vectors.shape[1]),
                "payload_bytes": int(payload_bytes),
                "requested_solver_use_gpu": bool(solver_use_gpu),
                "execution_mode": execution_mode,
                "precompute_time_ms": float(precompute_ms),
                "measured_solver_time_ms": float(solve_elapsed_ms),
                "mean_query_token_weight": float(query_token_weights.mean()) if query_token_weights.size else 0.0,
                "mean_candidate_fulfilment": float(np.mean(fulfilment_scores)) if fulfilment_scores.size else 0.0,
                "mean_relevance_score": float(np.mean(relevance_scores)) if relevance_scores.size else 0.0,
                "query_bundle": query_bundle_summary,
                "optimizer_policy_name": str(optimizer_policy.get("name", "baseline_v1")),
                "optimizer_policy": optimizer_policy,
                "edge_case_fallback": edge_case_fallback_summary,
                "effective_solver_config": {
                    key: value
                    for key, value in solver_config_payload.items()
                    if key != "controller"
                },
                "controller": controller_summary,
                **composed_features["summary"],
            },
        }

    def optimize(
        self,
        request: OptimizerRequest | Dict[str, Any],
        *,
        solver: Any | None = None,
    ) -> Dict[str, Any]:
        self._ensure_gpu()
        normalized_request, query_vectors, candidate_vectors = self._decode_request(request)
        return self._optimize_decoded(
            normalized_request,
            query_vectors,
            candidate_vectors,
            solver=solver,
        )


def create_stateless_optimizer_app(
    *,
    require_gpu: bool | None = None,
    strict_startup: bool | None = None,
) -> Any:
    """
    Standalone FastAPI app for the stateless optimizer (same contract as the
    canonical OSS reference service. Use :func:`default_optimizer_require_gpu` when
    ``require_gpu`` is None (honours ``VOYAGER_OPTIMIZER_REQUIRE_GPU``).
    """
    from fastapi import FastAPI, HTTPException

    pipeline = GpuFulfilmentPipeline(require_gpu=require_gpu)
    if strict_startup is None:
        strict_startup = bool(pipeline.require_gpu)

    app = FastAPI(
        title="ColSearch Fulfilment Optimizer",
        version="0.1.0",
        description="Stateless GPU-first context packing endpoint for fulfilment-aware knapsack solving.",
    )

    @app.on_event("startup")
    def _startup() -> None:
        health = pipeline.backend_health()
        if strict_startup:
            if not health["available"]:
                raise RuntimeError("GPU fulfilment optimizer failed startup health check")
            result = pipeline.request_path_self_test()
            if not result.get("ok", False):
                raise RuntimeError(
                    f"GPU fulfilment optimizer self-test failed: {result.get('reason', 'unknown')}"
                )
        else:
            if not health["available"]:
                logger.warning(
                    "Stateless optimizer startup: backend not fully available (%s); "
                    "set VOYAGER_OPTIMIZER_REQUIRE_GPU=1 for strict checks.",
                    health.get("solver_backend", {}).get("reason", "unknown"),
                )
            pipeline.request_path_self_test()

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return pipeline.backend_health()

    @app.post("/optimize")
    def optimize(request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return pipeline.optimize(request)
        except OptimizerPayloadTooLargeError as exc:  # pragma: no cover - integration surface
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ImportError as exc:  # pragma: no cover - integration surface
            raise HTTPException(
                status_code=400,
                detail="Stateless optimize requires the `latence_solver` native package to be installed.",
            ) from exc
        except (OptimizerRequestError, KeyError, TypeError, ValueError) as exc:  # pragma: no cover - integration surface
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - integration surface
            logger.exception("Optimizer request failed")
            raise HTTPException(status_code=500, detail="Optimizer request failed unexpectedly") from exc

    return app


def dump_request_json(request: OptimizerRequest) -> str:
    return json.dumps(request.to_dict(), indent=2, sort_keys=True)


__all__ = [
    "GpuFulfilmentPipeline",
    "OptimizerCandidate",
    "OptimizerPayloadTooLargeError",
    "OptimizerRequestError",
    "OptimizerRequest",
    "VectorPayload",
    "build_optimizer_request",
    "create_stateless_optimizer_app",
    "decode_vector_payload",
    "default_optimizer_require_gpu",
    "dump_request_json",
    "encode_float_vectors",
    "encode_roq_vectors",
    "quantize_payload_on_ingress",
]
