from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from colsearch._internal.inference.index_core.hybrid_manager import HybridSearchManager
from colsearch._internal.inference.search_pipeline import SearchPipeline


def _build_pipeline(tmp_path: Path) -> SearchPipeline:
    pipeline = SearchPipeline(str(tmp_path / "graph-pipeline"), dim=2, use_roq=False, on_disk=False)
    corpus = [
        "Team Alpha owns Service B and manages escalation.",
        "Service B causes Service C outage during dependency failures.",
        "Service C reliability overview and operating posture.",
        "Reliability playbook for recovery and incident support.",
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.05, 0.95],
            [0.85, 0.15],
            [0.10, 0.88],
        ],
        dtype=np.float32,
    )
    payloads = [
        {
            "text": corpus[0],
            "token_count": 72,
            "ontology_terms": ["Team Alpha", "Service B"],
            "relations": [{"source": "Team Alpha", "relation": "owns", "target": "Service B", "confidence": 0.96}],
            "concepts": ["ownership"],
        },
        {
            "text": corpus[1],
            "token_count": 78,
            "ontology_terms": ["Service B", "Service C"],
            "relations": [{"source": "Service B", "relation": "causes", "target": "Service C", "confidence": 0.98}],
            "concepts": ["incident"],
        },
        {
            "text": corpus[2],
            "token_count": 74,
            "ontology_terms": ["Service C"],
            "concepts": ["reliability"],
        },
        {
            "text": corpus[3],
            "token_count": 84,
            "ontology_terms": ["Playbook"],
            "concepts": ["reliability"],
        },
    ]
    pipeline.index(corpus=corpus, vectors=vectors, ids=[1, 2, 3, 4], payloads=payloads)
    return pipeline


def test_graph_policy_auto_only_triggers_for_graph_shaped_queries(tmp_path: Path) -> None:
    pipeline = _build_pipeline(tmp_path)

    ordinary = pipeline.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k_retrieval=1,
        graph_mode="auto",
        query_payload={},
    )
    graph_shaped = pipeline.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k_retrieval=1,
        graph_mode="auto",
        query_payload={"ontology_terms": ["Team Alpha", "Service C"]},
    )

    assert ordinary["retrieval"]["graph_policy"]["applied"] is False
    assert graph_shaped["retrieval"]["graph_policy"]["applied"] is True


def test_local_graph_expansion_recovers_missing_evidence(tmp_path: Path) -> None:
    pipeline = _build_pipeline(tmp_path)

    baseline = pipeline.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k_retrieval=1,
        graph_mode="off",
    )
    graph_enabled = pipeline.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k_retrieval=1,
        graph_mode="force",
        graph_options={"local_budget": 2, "community_budget": 0, "evidence_budget": 2, "max_hops": 2},
        query_payload={"ontology_terms": ["Team Alpha", "Service B"]},
    )

    assert baseline["selected_ids"] == [1]
    assert graph_enabled["selected_ids"][:2] == [1, 2]
    assert graph_enabled["retrieval"]["graph_summary"]["added_candidate_count"] >= 1
    assert graph_enabled["retrieval"]["graph_summary"]["merge_mode"] == "additive"
    assert graph_enabled["retrieval"]["graph_summary"]["invoked_after_first_stage"] is True
    assert "graph_local" in graph_enabled["retrieval"]["graph_provenance"]["2"]["lanes"]


def test_community_graph_retrieval_recovers_shared_theme_evidence(tmp_path: Path) -> None:
    pipeline = _build_pipeline(tmp_path)

    baseline = pipeline.search(
        "service c overview",
        top_k_retrieval=1,
        graph_mode="off",
    )
    graph_enabled = pipeline.search(
        "service c overview",
        top_k_retrieval=1,
        graph_mode="force",
        graph_options={"local_budget": 0, "community_budget": 2, "evidence_budget": 0, "max_hops": 2},
        query_payload={"ontology_terms": ["Service C", "support path"]},
    )

    assert baseline["selected_ids"] == [3]
    assert graph_enabled["selected_ids"][0] == 3
    assert 4 in graph_enabled["selected_ids"]
    assert graph_enabled["retrieval"]["graph_summary"]["added_candidate_count"] >= 1
    assert "graph_community" in graph_enabled["retrieval"]["graph_provenance"]["4"]["lanes"]


def test_evidence_budget_operates_independently_of_local_budget(tmp_path: Path) -> None:
    pipeline = _build_pipeline(tmp_path)

    graph_enabled = pipeline.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k_retrieval=1,
        graph_mode="force",
        graph_options={"local_budget": 0, "community_budget": 0, "evidence_budget": 2, "max_hops": 2},
        query_payload={"ontology_terms": ["Team Alpha", "Service B"]},
    )

    assert graph_enabled["selected_ids"][:2] == [1, 2]
    assert graph_enabled["retrieval"]["graph_summary"]["local_rescue_count"] == 0
    assert graph_enabled["retrieval"]["graph_summary"]["evidence_rescue_count"] >= 1
    assert "graph_local" in graph_enabled["retrieval"]["graph_provenance"]["2"]["lanes"]


def test_graph_features_change_refine_selection_when_enabled() -> None:
    manager = HybridSearchManager.__new__(HybridSearchManager)
    manager.roq_bits = None
    manager.graph_sidecar = None
    manager.graph_policy = None
    manager._last_refine_context = {}
    manager._cross_encoder_models = {}
    manager._nli_models = {}
    manager.solver = object()
    manager.hnsw = types.SimpleNamespace(
        retrieve=lambda ids: [
            {
                "id": 11,
                "vector": [1.0, 0.0],
                "payload": {"text": "baseline candidate", "token_count": 90},
            },
            {
                "id": 12,
                "vector": [1.0, 0.0],
                "payload": {"text": "graph-supported candidate", "token_count": 90},
            },
        ]
    )
    manager._last_search_context = {
        "dense": {},
        "sparse": {},
        "rrf": {},
        "graph": {
            12: {
                "graph_distance": 1.0,
                "graph_relation_confidence": 0.95,
                "graph_path_coherence": 0.90,
                "graph_support_count": 3.0,
                "graph_provenance_strength": 0.92,
                "graph_score": 0.96,
            }
        },
    }

    class FakePipeline:
        def optimize_in_process(self, *, candidate_items, **kwargs):
            ranked = sorted(
                candidate_items,
                key=lambda item: float(item["payload"].get("graph_score", 0.0))
                + float(item["payload"].get("centrality_score", 0.0))
                + float(item["payload"].get("auxiliary_score", 0.0)),
                reverse=True,
            )
            top = ranked[0]
            return {
                "selected_ids": [top["id"]],
                "solver_output": {"objective_score": float(top["payload"].get("graph_score", 0.0)), "total_tokens": 90},
                "feature_summary": {},
                "backend_kind": "cpu_reference",
            }

    manager._optimizer_pipeline = FakePipeline()

    result = manager.refine(
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        candidate_ids=[11, 12],
        query_text="",
        query_payload=None,
        refine_options={"solver_gate_mode": "always"},
    )

    assert result["selected_ids"] == ["12"]
    assert result["feature_summary"]["selected_avg_graph_score"] > result["feature_summary"]["candidate_avg_graph_score"] / 2.0


def test_graph_force_falls_back_cleanly_when_sidecar_is_degraded(tmp_path: Path) -> None:
    pipeline = _build_pipeline(tmp_path)
    pipeline.manager.graph_sidecar.health = "degraded"
    pipeline.manager.graph_sidecar.reason = "corrupted_state"

    result = pipeline.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k_retrieval=1,
        graph_mode="force",
        query_payload={"ontology_terms": ["Team Alpha", "Service C"]},
    )

    assert result["selected_ids"] == [1]
    assert result["retrieval"]["graph_policy"]["reason"] == "graph_unavailable"


def test_graph_policy_can_force_for_mandatory_workflows(tmp_path: Path) -> None:
    pipeline = _build_pipeline(tmp_path)

    result = pipeline.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k_retrieval=1,
        graph_mode="auto",
        query_payload={"workflow_type": "compliance"},
    )

    assert result["retrieval"]["graph_policy"]["applied"] is True
    assert result["retrieval"]["graph_policy"]["mandatory_reason"] == "workflow:compliance"
