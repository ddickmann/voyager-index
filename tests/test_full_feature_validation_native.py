from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "full_feature_validation",
    Path(__file__).resolve().parents[1] / "scripts" / "full_feature_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
ffv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ffv)


@pytest.mark.parametrize("module_name", ["latence_shard_engine", "latence_solver"])
def test_probe_native_package_reports_installed_when_importable(tmp_path, monkeypatch, module_name):
    original_find_spec = ffv.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == module_name:
            return object()
        return original_find_spec(name)

    monkeypatch.setattr(ffv.importlib.util, "find_spec", fake_find_spec)

    result = ffv.probe_native_package(module_name, Path(tmp_path), bootstrap=False)

    assert result["status"] == "installed_and_importable"
    assert result["source_present"] is True


@pytest.mark.parametrize("module_name", ["latence_shard_engine", "latence_solver"])
def test_probe_native_package_reports_source_present_but_not_built(tmp_path, monkeypatch, module_name):
    original_find_spec = ffv.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == module_name:
            return None
        return original_find_spec(name)

    monkeypatch.setattr(ffv.importlib.util, "find_spec", fake_find_spec)

    result = ffv.probe_native_package(module_name, Path(tmp_path), bootstrap=False)

    assert result["status"] == "source_present_but_not_built"
    assert result["reason"] == "module_not_importable_in_active_environment"
    assert result["source_present"] is True


@pytest.mark.parametrize("module_name", ["latence_shard_engine", "latence_solver"])
def test_probe_native_package_reports_missing_toolchain(tmp_path, monkeypatch, module_name):
    original_find_spec = ffv.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == module_name:
            return None
        return original_find_spec(name)

    monkeypatch.setattr(ffv.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(ffv.shutil, "which", lambda _: None)

    result = ffv.probe_native_package(module_name, Path(tmp_path), bootstrap=True)

    assert result["status"] == "build_failed"
    assert result["reason"] == "missing_rust_toolchain"
    assert result["missing_tools"] == ["cargo", "rustc"]


def test_hnsw_fallback_probe_runs_without_native_package(tmp_path):
    result = ffv._run_hnsw_probe(force_fallback=True, output_dir=Path(tmp_path), log_name="fallback")

    assert result["status"] == "passed"
    assert result["summary"]["flushed_before_reopen"] is True
    assert result["summary"]["top_ids"][0] == 100
    assert result["summary"]["filtered_top_ids"][0] == 101


def test_ranking_metrics_capture_rank_quality():
    metrics = ffv._ranking_metrics([5, 2, 9], {2: 3.0, 9: 1.0}, k=3)

    assert metrics["mrr"] == pytest.approx(0.5)
    assert 0.0 < metrics["ndcg"] <= 1.0
    assert metrics["recall"] == pytest.approx(1.0)


def test_build_evaluation_corpus_matches_rendered_text_to_ontology_terms():
    corpus = ffv.build_evaluation_corpus(
        [
            {
                "source": "/tmp/seedbox-resolution.pdf",
                "image": "/tmp/seedbox-resolution-page-1.png",
                "page": 1,
                "renderer": "pymupdf",
                "text": "Seedbox Ventures GmbH board resolution for 2026.",
            }
        ],
        [
            {
                "entity_id": "entity-seedbox",
                "canonical_name": "Seedbox Ventures",
                "label": "organization",
                "confidence": 0.97,
                "document_ids": ["doc-1"],
            }
        ],
    )

    assert corpus["summary"]["records_with_ontology_matches"] == 1
    assert corpus["records"][0]["ontology_terms"] == ["Seedbox Ventures"]
    assert corpus["queries"][0]["text"] == "Seedbox Ventures"
    assert corpus["queries"][0]["source"] == "ontology_entity"


def test_prepare_evaluation_bundle_uses_synthetic_embeddings_when_model_unavailable(tmp_path, monkeypatch):
    original_find_spec = ffv.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "sauerkrautlm_colpali":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(ffv.importlib.util, "find_spec", fake_find_spec)

    bundle = ffv.prepare_evaluation_bundle(
        [
            {
                "source": "/tmp/seedbox-resolution.pdf",
                "image": "/tmp/seedbox-resolution-page-1.png",
                "page": 1,
                "renderer": "pymupdf",
                "text": "Seedbox Ventures GmbH board resolution for 2026.",
            }
        ],
        [
            {
                "entity_id": "entity-seedbox",
                "canonical_name": "Seedbox Ventures",
                "label": "organization",
                "confidence": 0.97,
                "document_ids": ["doc-1"],
            }
        ],
        Path(tmp_path),
    )

    assert bundle["embedding_summary"]["embedding_source"] == "synthetic_fallback"
    assert bundle["corpus_summary"]["record_count"] == 1
    assert bundle["records"][0]["multivector"].shape == (8, 32)
    assert bundle["queries"][0]["dense_vector"].shape == (32,)


def test_computed_feature_pack_populates_explicit_solver_fields():
    records = [
        {
            "id": "doc-1",
            "dense_vector": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "enriched_text": "Board resolution with evidence table and summary.",
            "base_text": "Board resolution.",
            "token_count": 32,
            "ontology_terms": ["Board Resolution"],
            "ontology_concept_density": 0.8,
            "ontology_relation_density": 0.4,
            "ontology_confidence": 0.9,
            "document_id": "a",
            "source_name": "alpha.pdf",
            "page": 1,
        },
        {
            "id": "doc-2",
            "dense_vector": np.asarray([0.9, 0.1, 0.0], dtype=np.float32),
            "enriched_text": "Procedure step with supporting evidence.",
            "base_text": "Procedure step.",
            "token_count": 28,
            "ontology_terms": [],
            "ontology_concept_density": 0.1,
            "ontology_relation_density": 0.0,
            "ontology_confidence": 0.2,
            "document_id": "a",
            "source_name": "alpha.pdf",
            "page": 2,
        },
    ]

    feature_pack = ffv._computed_feature_pack(records)

    assert len(feature_pack) == 2
    for feature_row in feature_pack:
        assert set(feature_row) == {
            "fact_density",
            "centrality_score",
            "recency_score",
            "auxiliary_score",
            "cluster_id",
            "rhetorical_role",
        }
        assert 0.0 <= feature_row["fact_density"] <= 1.0
        assert 0.0 <= feature_row["centrality_score"] <= 1.0
        assert 0.0 <= feature_row["recency_score"] <= 1.0
        assert 0.0 <= feature_row["auxiliary_score"] <= 1.0

    assert feature_pack[0]["cluster_id"] == feature_pack[1]["cluster_id"]
    assert feature_pack[0]["rhetorical_role"] in ffv.ROLE_TO_ID or feature_pack[0]["rhetorical_role"] == "unknown"


def test_build_value_summary_exposes_key_deltas():
    summary = ffv.build_value_summary(
        {
            "lanes": {
                "maxsim": {"summary": {"fast_colbert_cpu": {"elapsed_ms": 10.0}, "triton_cuda": {"status": "passed", "elapsed_ms": 5.0, "parity": True}}},
                "quantization": {"summary": {"roq_1bit": {"recall_at_k": 0.1}, "roq_4bit": {"recall_at_k": 0.3, "top_1_agreement": 0.9, "rank_metrics": {"ndcg": 0.8}, "ndcg_delta_vs_full_precision": -0.02, "speedup_vs_full_precision": 1.6, "compression_vs_fp16": 3.2, "candidate_sweep": {"crossover_candidate_count": 128, "counts": [{"candidate_count": 32, "speedup_vs_full_precision": 0.7}, {"candidate_count": 64, "speedup_vs_full_precision": 0.95}, {"candidate_count": 128, "speedup_vs_full_precision": 1.1}]}}, "prototype_sidecar_screening": {"fallback_rate": 0.125, "direct_gather_rate": 0.875, "bootstrap_calibration_passed": True, "candidate_pool_retention": [{"candidate_count": 128, "top1_retention": 0.8, "topk_retention": 0.9}]}, "centroid_screening": {"recall_at_k": 0.88, "speedup_vs_full_precision": 1.9, "top_1_agreement": 0.92, "full_recovery_candidate_count": 64, "fallback_rate": 0.25, "direct_gather_rate": 0.75, "bootstrap_calibration_passed": True, "candidate_pool_retention": [{"candidate_count": 128, "top1_retention": 0.7, "topk_retention": 0.85}]}, "centroid_screening_scale_harness": {"speedup_vs_full_precision": 8.5}, "centroid_screening_decision_gate": {"promotion_recommendation": "promote_to_optimized"}, "multimodal_int8": {"recall_at_k": 0.8, "elapsed_ms_full_precision": 20.0, "elapsed_ms_quantized": 5.0}}},
                "solver": {"summary": {"bm25": {"avg_rank_metrics": {"ndcg": 0.2}}, "dense_only": {"avg_rank_metrics": {"ndcg": 0.4}}, "hybrid_rrf": {"avg_rank_metrics": {"ndcg": 0.5}}, "solver_refined": {"avg_rank_metrics": {"ndcg": 0.7}, "avg_latency_ms": 12.0}, "comparison": {"hybrid_rrf_delta": {"ndcg": 0.01}, "solver_refined_delta": {"ndcg": 0.08}, "solver_avg_latency_delta_ms": 1.5, "solver_avg_objective_delta": 0.4, "solver_avg_total_tokens_delta": -24.0}}},
                "ontology_variant": {"summary": {"hybrid_rrf_delta": {"ndcg": 0.05}, "solver_refined_delta": {"ndcg": 0.02}}},
                "api_crud": {"summary": {"search": {"dense_vector_ms": 1.0, "dense_text_ms": 2.0, "dense_hybrid_ms": 3.0, "dense_optimized_ms": 4.0, "late_interaction_ms": 5.0, "multimodal_ms": 6.0}, "restart": {"cold_start_ms": 7.0}}},
            }
        }
    )

    assert summary["maxsim"]["triton_speedup_vs_cpu"] == pytest.approx(2.0)
    assert summary["solver"]["solver_minus_rrf_ndcg"] == pytest.approx(0.2)
    assert summary["quantization"]["multimodal_int8_speedup_vs_full_precision"] == pytest.approx(4.0)
    assert summary["quantization"]["centroid_screening_recall_at_k"] == pytest.approx(0.88)
    assert summary["quantization"]["centroid_screening_promotion_recommendation"] == "promote_to_optimized"
    assert summary["quantization"]["centroid_screening_fallback_rate"] == pytest.approx(0.25)
    assert summary["quantization"]["centroid_screening_top1_retention_at_128"] == pytest.approx(0.7)
    assert summary["quantization"]["centroid_screening_topk_retention_at_128"] == pytest.approx(0.85)
    assert summary["quantization"]["prototype_sidecar_direct_gather_rate"] == pytest.approx(0.875)
    assert summary["quantization"]["prototype_sidecar_top1_retention_at_128"] == pytest.approx(0.8)
    assert summary["quantization"]["prototype_sidecar_topk_retention_at_128"] == pytest.approx(0.9)
    assert summary["quantization"]["roq_4bit_top_1_agreement"] == pytest.approx(0.9)
    assert summary["quantization"]["roq_4bit_compression_vs_fp16"] == pytest.approx(3.2)
    assert summary["quantization"]["roq_4bit_crossover_candidate_count"] == 128
    assert summary["quantization"]["roq_4bit_best_sweep_speedup"] == pytest.approx(1.1)
    assert summary["solver_enrichment"]["solver_ndcg_delta"] == pytest.approx(0.08)
    assert summary["solver_enrichment"]["solver_total_tokens_delta"] == pytest.approx(-24.0)


def test_retention_at_budget_uses_next_comparable_bucket():
    summary = {
        "candidate_pool_retention": [
            {"candidate_count": 32, "topk_retention": 0.4},
            {"candidate_count": 64, "topk_retention": 0.8},
        ]
    }

    assert ffv._retention_at_budget(summary, 40, "topk_retention") == pytest.approx(0.8)
    assert ffv._retention_at_budget(summary, 64, "topk_retention") == pytest.approx(0.8)


def test_evaluate_screened_multimodal_engine_tracks_exact_fallback():
    class FakeEngine:
        def __init__(self):
            self.device = "cpu"
            self.last_screening_profile = {}
            self.last_search_profile = {}

        def screen_candidates(self, **kwargs):
            self.last_screening_profile = {"elapsed_ms": 0.1, "reason": "health_degraded"}
            return None

        def search(self, **kwargs):
            self.last_search_profile = {
                "direct_gather": kwargs.get("candidate_ids") is not None,
                "load_ms": 0.0,
                "materialize_ms": 0.0,
                "mask_ms": 0.0,
                "score_ms": 0.0,
                "heap_ms": 0.0,
            }
            return [type("Result", (), {"doc_id": "doc-a"})()]

        def get_statistics(self):
            return {
                "screening_backend": "prototype_hnsw",
                "screening": {},
                "screening_state": {"health": "degraded"},
            }

    summary = ffv._evaluate_screened_multimodal_engine(
        engine=FakeEngine(),
        queries=[{"query_id": "q-1", "text": "alpha", "relevance_map": {"doc-a": 1.0}}],
        query_vectors=[np.asarray([[1.0, 0.0]], dtype=np.float32)],
        doc_ids=["doc-a"],
        doc_id_to_index={"doc-a": 0},
        exact_mv_topk=np.asarray([[0]], dtype=np.int32),
        full_precision_benchmark={"median_ms": 1.0},
        top_k=1,
        candidate_budgets=[1],
        default_budget=1,
    )

    assert summary["fallback_query_count"] == 1
    assert summary["fallback_routes_exact"] is True
    assert summary["bootstrap_calibration_passed"] is False


def test_pick_multimodal_ordering_winner_prefers_fastest_variant_within_ndcg_band():
    winner = ffv._pick_multimodal_ordering_winner(
        {
            "maxsim_only": {
                "avg_rank_metrics": {"ndcg": 0.91, "recall": 0.91, "mrr": 0.91},
                "avg_latency_ms": 40.0,
            },
            "solver_prefilter_maxsim": {
                "avg_rank_metrics": {"ndcg": 0.905, "recall": 0.90, "mrr": 0.90},
                "avg_latency_ms": 18.0,
            },
            "maxsim_then_solver": {
                "avg_rank_metrics": {"ndcg": 0.92, "recall": 0.92, "mrr": 0.92},
                "avg_latency_ms": 65.0,
            },
        }
    )

    assert winner["winner"] == "solver_prefilter_maxsim"
    assert winner["reason"] == "lowest_latency_within_quality_band"


def test_native_hnsw_segment_reopens_after_flush():
    latence_hnsw = pytest.importorskip("latence_hnsw")
    np = pytest.importorskip("numpy")

    with tempfile.TemporaryDirectory() as tmpdir:
        segment_path = Path(tmpdir) / "segment"
        segment = latence_hnsw.HnswSegment(str(segment_path), 4, "cosine", 16, 100, True)
        segment.add(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            [7],
            [{"text": "alpha"}],
        )
        segment.flush()
        del segment
        gc.collect()

        reopened = latence_hnsw.HnswSegment(str(segment_path), 4, "cosine", 16, 100, True)
        results = reopened.search(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=3)

        assert reopened.len() == 1
        assert results[0][0] == 7
