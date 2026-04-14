from __future__ import annotations

import numpy as np

from voyager_index._internal.inference.stateless_optimizer import (
    GpuFulfilmentPipeline,
    OptimizerCandidate,
    encode_float_vectors,
)


def _candidate(chunk_id: str, text: str, *, source_id: str) -> OptimizerCandidate:
    return OptimizerCandidate(
        chunk_id=chunk_id,
        text=text,
        token_count=64,
        vectors=encode_float_vectors(np.asarray([1.0, 0.0], dtype=np.float32), dtype="float32"),
        metadata={"source_id": source_id},
    )


def test_query_payload_aspect_weights_override_default_token_weights() -> None:
    pipeline = GpuFulfilmentPipeline(require_gpu=False)
    policy = pipeline._resolve_optimizer_policy("frontier_v1")
    query_vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    candidate_vectors = [
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    ]

    _, default_weights, default_relevance, _ = pipeline._coverage_and_relevance(
        query_vectors,
        candidate_vectors,
        metadata={},
        policy=policy,
    )
    _, boosted_weights, boosted_relevance, _ = pipeline._coverage_and_relevance(
        query_vectors,
        candidate_vectors,
        metadata={"query_payload": {"query_aspect_weights": [0.1, 0.9]}},
        policy=policy,
    )

    assert default_weights[0] == default_weights[1]
    assert boosted_weights[1] > boosted_weights[0]
    assert boosted_relevance[1] > boosted_relevance[0]
    assert default_relevance[0] == default_relevance[1]


def test_semantic_redundancy_blends_source_and_aspect_signals() -> None:
    pipeline = GpuFulfilmentPipeline(require_gpu=False)
    policy = pipeline._resolve_optimizer_policy("frontier_v1")
    candidates = [
        _candidate("a", "invoice total due", source_id="doc-1"),
        _candidate("b", "invoice total payable", source_id="doc-1"),
        _candidate("c", "shipment status tracking", source_id="doc-2"),
    ]
    candidate_vectors = [
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    ]
    coverage_matrix = np.asarray(
        [
            [1.0, 0.95, 0.10],
            [0.10, 0.15, 0.98],
        ],
        dtype=np.float32,
    )

    redundancy = pipeline._redundancy_matrix(
        candidates,
        candidate_vectors,
        coverage_matrix=coverage_matrix,
        policy=policy,
    )

    assert redundancy.shape == (3, 3)
    assert redundancy[0, 1] > redundancy[0, 2]
    assert redundancy[1, 0] == redundancy[0, 1]
    assert float(redundancy[0, 0]) == 0.0
