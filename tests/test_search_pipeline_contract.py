from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from colsearch._internal.inference.index_core.hybrid_manager import HybridSearchManager
from colsearch._internal.inference.search_pipeline import SearchPipeline


def test_search_pipeline_supports_sparse_only_text_queries(tmp_path: Path) -> None:
    pipeline = SearchPipeline(str(tmp_path / "pipeline"), dim=4, use_roq=False, on_disk=False)
    pipeline.index(
        corpus=["alpha document", "beta document"],
        vectors=np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        ids=[1, 2],
        payloads=[{"label": "alpha"}, {"label": "beta"}],
    )

    result = pipeline.search("alpha", top_k_retrieval=2)

    assert result["retrieval_count"] >= 1
    assert 1 in result["selected_ids"]


def test_search_pipeline_rejects_multivector_dense_queries(tmp_path: Path) -> None:
    pipeline = SearchPipeline(str(tmp_path / "pipeline"), dim=4, use_roq=False, on_disk=False)

    with pytest.raises(ValueError, match="single dense query vector"):
        pipeline.search(np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32))


def test_search_pipeline_reports_solver_backend_when_refining(tmp_path: Path) -> None:
    pytest.importorskip("latence_solver")

    pipeline = SearchPipeline(str(tmp_path / "pipeline"), dim=4, use_roq=False, on_disk=False)
    pipeline.index(
        corpus=["alpha voyager", "beta support", "gamma noise"],
        vectors=np.asarray(
            [[1, 0, 0, 0], [0.8, 0.2, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        ),
        ids=[1, 2, 3],
        payloads=[
            {"text": "alpha voyager", "token_count": 80},
            {"text": "beta support", "token_count": 90},
            {"text": "gamma noise", "token_count": 120},
        ],
    )

    result = pipeline.search(np.asarray([1, 0, 0, 0], dtype=np.float32), top_k_retrieval=3, enable_refinement=True)

    assert result["solver_output"] is not None
    assert result["solver_backend"] is not None


def test_search_pipeline_forwards_refine_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = SearchPipeline(str(tmp_path / "pipeline"), dim=4, use_roq=False, on_disk=False)
    captured: dict[str, object] = {}

    def fake_search(*, query_text, query_vector, k):
        assert query_text == "alpha voyager"
        assert k == 3
        return {
            "dense": [(1, 0.9), (2, 0.8)],
            "sparse": [(2, 0.7)],
            "union_ids": [1, 2],
            "sparse_error": None,
        }

    def fake_refine(**kwargs):
        captured.update(kwargs)
        return {
            "solver_output": {"objective_score": 1.0},
            "selected_ids": ["1", "2"],
            "backend_kind": "cpu_reference",
        }

    monkeypatch.setattr(pipeline.manager, "search", fake_search)
    monkeypatch.setattr(pipeline.manager, "solver_available", True)
    monkeypatch.setattr(pipeline.manager, "refine", fake_refine)

    result = pipeline.search(
        np.asarray([1, 0, 0, 0], dtype=np.float32),
        top_k_retrieval=3,
        enable_refinement=True,
        query_text="alpha voyager",
        query_payload={"label": "evidence"},
        solver_config={"iterations": 24},
        optimizer_policy="post_rerank_v1",
        refine_options={"use_cross_encoder": True, "cross_encoder_top_k": 8},
    )

    assert result["solver_output"] == {"objective_score": 1.0}
    assert captured["query_text"] == "alpha voyager"
    assert captured["query_payload"] == {"label": "evidence"}
    assert captured["solver_config"] == {"iterations": 24}
    assert captured["optimizer_policy"] == "post_rerank_v1"
    assert captured["refine_options"] == {"use_cross_encoder": True, "cross_encoder_top_k": 8}


def test_hybrid_manager_orders_selected_candidates_by_marginal_gain() -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    solver_candidates = [
        {
            "embedding": [0.6, 0.8],
            "fact_density": 0.2,
            "centrality_score": 0.3,
            "recency_score": 0.2,
            "auxiliary_score": 0.1,
        },
        {
            "embedding": [1.0, 0.0],
            "fact_density": 0.9,
            "centrality_score": 0.9,
            "recency_score": 0.8,
            "auxiliary_score": 0.6,
        },
    ]

    ordered = manager._selected_order(query, solver_candidates, [0, 1])

    assert ordered == [1, 0]


def test_hybrid_manager_builds_solver_candidates_from_roq_payload_when_vector_missing() -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.roq_bits = 4

    class DummyDecoded:
        def __init__(self, array):
            self._array = array

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class DummyQuantizer:
        def decode(self, codes, scale, offset):
            assert codes.shape[0] == 1
            return DummyDecoded(np.asarray([[1.0, 0.0]], dtype=np.float32))

    manager.hnsw = types.SimpleNamespace(
        quantizer=DummyQuantizer(),
        retrieve=lambda ids: [
            {
                "id": ids[0],
                "vector": None,
                "payload": {
                    "text": "voyager example",
                    "roq_codes": [1, 2, 3, 4],
                    "roq_scale": 1.0,
                    "roq_offset": 0.0,
                },
            }
        ],
    )

    candidates = manager._build_solver_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        [7],
        query_text="voyager",
    )

    assert len(candidates) == 1
    assert candidates[0]["chunk_id"] == "7"
    assert candidates[0]["embedding"] == [1.0, 0.0]


def test_hybrid_manager_forwards_shard_dense_search_kwargs() -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    captured: dict[str, object] = {}

    def fake_search_multivector(query, *, k, filters=None, **kwargs):
        captured["query_shape"] = tuple(query.shape)
        captured["k"] = k
        captured["filters"] = filters
        captured.update(kwargs)
        return [(7, 0.8)]

    manager._dense_engine_type = "shard"
    manager.hnsw = types.SimpleNamespace(search_multivector=fake_search_multivector)
    manager.retriever = None
    manager._legacy_bm25 = None
    manager.sparse_dirty = False
    manager.sparse_error = None
    manager._last_search_context = None

    result = manager.search(
        query_text="",
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        k=5,
        filters={"tenant": "acme"},
        dense_search_kwargs={"quantization_mode": "fp8", "use_colbandit": True},
    )

    assert captured["query_shape"] == (1, 2)
    assert captured["k"] == 5
    assert captured["filters"] == {"tenant": "acme"}
    assert captured["quantization_mode"] == "fp8"
    assert captured["use_colbandit"] is True
    assert result["dense"] == [(7, 0.8)]
    assert result["union_ids"] == [7]


def test_hybrid_manager_uses_ontology_query_payload_for_solver_features() -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.roq_bits = None
    shared_vector = [1.0, 0.0]
    manager.hnsw = types.SimpleNamespace(
        retrieve=lambda ids: [
            {
                "id": 11,
                "vector": shared_vector,
                "payload": {
                    "text": "neutral content",
                    "ontology_terms": ["invoice total"],
                    "ontology_labels": ["table"],
                    "ontology_confidences": [0.9],
                    "ontology_evidence_counts": [4],
                    "ontology_match_count": 1,
                    "ontology_confidence": 0.9,
                    "ontology_concept_density": 0.6,
                    "ontology_relation_density": 0.5,
                },
            },
            {
                "id": 12,
                "vector": shared_vector,
                "payload": {
                    "text": "neutral content",
                },
            },
        ],
    )

    candidates = manager._build_solver_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        [11, 12],
        query_text="invoice total",
        query_payload={"ontology_terms": ["invoice total"], "label": "table"},
    )

    by_id = {candidate["chunk_id"]: candidate for candidate in candidates}
    assert by_id["11"]["ontology_query_match"] > by_id["12"]["ontology_query_match"]
    assert by_id["11"]["ontology_entity_coverage"] > by_id["12"]["ontology_entity_coverage"]
    assert by_id["11"]["centrality_score"] > by_id["12"]["centrality_score"]
    assert by_id["11"]["auxiliary_score"] > by_id["12"]["auxiliary_score"]


def test_hybrid_manager_cross_encoder_rerank_promotes_base_relevance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.roq_bits = None
    manager._last_refine_context = {}
    shared_vector = [1.0, 0.0]
    manager.hnsw = types.SimpleNamespace(
        retrieve=lambda ids: [
            {
                "id": 11,
                "vector": shared_vector,
                "payload": {"text": "invoice total due"},
            },
            {
                "id": 12,
                "vector": shared_vector,
                "payload": {"text": "board report summary"},
            },
        ]
    )

    class DummyCrossEncoder:
        def predict(self, pairs, batch_size=32, show_progress_bar=False):
            assert len(pairs) == 2
            return np.asarray(
                [0.95 if "invoice" in document else 0.10 for _, document in pairs],
                dtype=np.float32,
            )

    monkeypatch.setattr(manager, "_get_cross_encoder", lambda model_name: DummyCrossEncoder())

    candidates = manager._build_solver_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        [11, 12],
        query_text="invoice total due",
        retrieval_features={
            11: {"base_relevance": 0.20},
            12: {"base_relevance": 0.80},
        },
        refine_options={
            "use_cross_encoder": True,
            "cross_encoder_model": "dummy-cross-encoder",
            "cross_encoder_top_k": 2,
            "cross_encoder_batch_size": 8,
        },
    )

    by_id = {candidate["chunk_id"]: candidate for candidate in candidates}
    assert by_id["11"]["cross_encoder_score"] == pytest.approx(0.95)
    assert by_id["11"]["base_relevance"] == pytest.approx(0.95)
    assert by_id["12"]["cross_encoder_score"] == pytest.approx(0.10)
    assert by_id["12"]["base_relevance"] == pytest.approx(0.10)
    assert manager._last_refine_context["rerank"]["applied"] is True


def test_hybrid_manager_nli_scores_surface_utility_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    manager = HybridSearchManager.__new__(HybridSearchManager)

    class DummyTokenizer:
        def __call__(self, left, right, **kwargs):
            batch = len(right)
            return {
                "input_ids": torch.ones((batch, 4), dtype=torch.long),
                "attention_mask": torch.ones((batch, 4), dtype=torch.long),
            }

    class DummyModel:
        def __call__(self, **kwargs):
            _ = kwargs
            logits = torch.tensor(
                [
                    [5.0, 1.0, 0.2],  # entailment-heavy
                    [0.2, 1.0, 5.0],  # contradiction-heavy
                ],
                dtype=torch.float32,
            )
            return types.SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        manager,
        "_get_nli_model",
        lambda model_name: {
            "tokenizer": DummyTokenizer(),
            "model": DummyModel(),
            "device": torch.device("cpu"),
            "id2label": {0: "entailment", 1: "neutral", 2: "contradiction"},
        },
    )

    retrieval_features = {
        11: {"base_relevance": 0.7},
        12: {"base_relevance": 0.6},
    }
    summary = manager._apply_nli_scores(
        valid_items=[
            {"id": 11, "text": "claim is supported"},
            {"id": 12, "text": "claim is contradicted"},
        ],
        query_text="supported claim",
        retrieval_features=retrieval_features,
        model_name="dummy-nli",
        batch_size=2,
        top_k=2,
    )

    assert summary["applied"] is True
    assert retrieval_features[11]["utility_score"] > retrieval_features[12]["utility_score"]
    assert retrieval_features[11]["nli_entailment"] > retrieval_features[12]["nli_entailment"]
    assert retrieval_features[12]["nli_contradiction"] > retrieval_features[11]["nli_contradiction"]


def test_hybrid_manager_confidence_gate_skips_expensive_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.roq_bits = None
    manager._last_refine_context = {}
    shared_vector = [1.0, 0.0]
    manager.hnsw = types.SimpleNamespace(
        retrieve=lambda ids: [
            {
                "id": 11,
                "vector": shared_vector,
                "payload": {"text": "top candidate"},
            },
            {
                "id": 12,
                "vector": shared_vector,
                "payload": {"text": "runner up"},
            },
        ]
    )

    def fail_cross_encoder(**kwargs):
        raise AssertionError("cross-encoder should have been skipped by confidence gate")

    def fail_nli(**kwargs):
        raise AssertionError("nli scorer should have been skipped by confidence gate")

    monkeypatch.setattr(manager, "_apply_cross_encoder_scores", fail_cross_encoder)
    monkeypatch.setattr(manager, "_apply_nli_scores", fail_nli)

    candidates = manager._build_solver_candidates(
        np.asarray([1.0, 0.0], dtype=np.float32),
        [11, 12],
        query_text="top candidate",
        retrieval_features={
            11: {"base_relevance": 0.95},
            12: {"base_relevance": 0.10},
        },
        refine_options={
            "use_cross_encoder": True,
            "use_nli": True,
            "confidence_gating": True,
            "confidence_min_candidates": 2,
            "confidence_gap_threshold": 0.20,
        },
    )

    assert len(candidates) == 2
    assert manager._last_refine_context["confidence_gate"]["applied"] is True
    assert manager._last_refine_context["rerank"]["reason"] == "confidence_gate_skip"
    assert manager._last_refine_context["nli"]["reason"] == "confidence_gate_skip"


def test_hybrid_manager_refine_skips_solver_when_production_gate_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.solver = types.SimpleNamespace(
        config=types.SimpleNamespace(lambda_=0.32, iterations=48),
    )
    manager._optimizer_pipeline = types.SimpleNamespace(
        optimize_in_process=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("solver should not run when production gate skips refinement")
        )
    )
    manager._last_search_context = None
    manager._last_refine_context = {}

    monkeypatch.setattr(
        manager,
        "_build_solver_candidates",
        lambda *args, **kwargs: [
            {
                "chunk_id": "11",
                "content": "alpha target",
                "embedding": [1.0, 0.0],
                "token_count": 80,
                "fact_density": 0.4,
                "centrality_score": 0.6,
                "recency_score": 0.2,
                "auxiliary_score": 0.3,
                "rhetorical_role": "evidence",
                "cluster_id": 0,
                "base_relevance": 0.95,
                "rrf_score": 0.95,
            },
            {
                "chunk_id": "12",
                "content": "beta support",
                "embedding": [0.8, 0.2],
                "token_count": 90,
                "fact_density": 0.3,
                "centrality_score": 0.4,
                "recency_score": 0.2,
                "auxiliary_score": 0.2,
                "rhetorical_role": "support",
                "cluster_id": 1,
                "base_relevance": 0.10,
                "rrf_score": 0.10,
            },
        ],
    )

    refined = manager.refine(
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        candidate_ids=[11, 12],
        query_text="alpha target",
        constraints={"max_tokens": 220, "max_chunks": 1},
    )

    assert refined["backend_kind"] == "rrf_gate"
    assert refined["selected_internal_ids"] == [11]
    assert refined["solver_output"]["skipped_by_gate"] is True
    assert "candidate_count_below_min" in refined["solver_output"]["gate"]["reasons"]
    assert refined["feature_summary"]["skipped_by_gate"] is True


def test_hybrid_manager_refine_caps_lambda_for_large_production_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.solver = types.SimpleNamespace(
        config=types.SimpleNamespace(lambda_=0.32, iterations=48),
    )
    manager._last_search_context = None
    manager._last_refine_context = {}
    captured: dict[str, object] = {}

    def fake_optimize_in_process(**kwargs):
        captured["solver_config"] = dict(kwargs["solver_config"])
        return {
            "selected_ids": ["100", "101"],
            "solver_output": {"objective_score": 1.25, "total_tokens": 160},
            "feature_summary": {"candidate_count": 60},
            "backend_kind": "cpu_reference",
            "solver_backend_kind": "cpu_reference",
        }

    manager._optimizer_pipeline = types.SimpleNamespace(optimize_in_process=fake_optimize_in_process)

    def build_candidates(*args, **kwargs):
        candidates = []
        for idx in range(60):
            candidates.append(
                {
                    "chunk_id": str(100 + idx),
                    "content": f"candidate {idx}",
                    "embedding": [1.0, 0.0],
                    "token_count": 64,
                    "fact_density": 0.4,
                    "centrality_score": 0.5,
                    "recency_score": 0.2,
                    "auxiliary_score": 0.3,
                    "rhetorical_role": "evidence",
                    "cluster_id": idx % 4,
                    "base_relevance": 1.0 - (0.001 * idx),
                    "rrf_score": 1.0 - (0.001 * idx),
                }
            )
        return candidates

    monkeypatch.setattr(manager, "_build_solver_candidates", build_candidates)

    refined = manager.refine(
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        candidate_ids=list(range(100, 160)),
        query_text="voyager target",
        constraints={"max_tokens": 1000, "max_chunks": 8},
    )

    assert refined["backend_kind"] == "cpu_reference"
    assert captured["solver_config"]["lambda"] == pytest.approx(0.05)
    assert refined["feature_summary"]["production_solver_profile"]["applied"] is True
    assert refined["feature_summary"]["refine_solver_gate"]["run_solver"] is True
