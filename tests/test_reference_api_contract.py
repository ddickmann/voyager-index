from __future__ import annotations

import base64
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from voyager_index import __version__ as package_version
from voyager_index import encode_vector_payload
from voyager_index._internal.inference.engines.base import SearchResult
from voyager_index._internal.server.api.models import SearchRequest
from voyager_index._internal.server.api.service import SearchService
from voyager_index._internal.server.main import create_app


def _create_client(index_path: Path, version: str = "0.1.0") -> TestClient:
    return TestClient(create_app(index_path=str(index_path), version=version))


def _optimizer_vector_payload(vectors) -> dict[str, object]:
    return encode_vector_payload(vectors, dtype="float32")


def _write_png(path: Path) -> Path:
    path.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAn8B9pVHtVQAAAAASUVORK5CYII=")
    )
    return path


def test_health_uses_app_version_and_metrics_are_prometheus_counters(tmp_path: Path) -> None:
    with _create_client(tmp_path, version="9.9.9") as client:
        health = client.get("/health")
        metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["version"] == "9.9.9"

    assert metrics.status_code == 200
    body = metrics.text
    assert "voyager_search_latency_seconds_sum" in body
    assert "voyager_search_latency_seconds_count" in body
    assert "\nvoyager_search_latency_seconds " not in body
    assert "voyager_collection_status" in body


def test_default_health_version_matches_public_package_version(tmp_path: Path) -> None:
    with TestClient(create_app(index_path=str(tmp_path))) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["version"] == package_version


def test_reference_optimize_health_is_exposed(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        response = client.get("/reference/optimize/health")

    assert response.status_code == 200
    payload = response.json()
    assert "available" in payload
    assert "execution_mode" in payload
    assert "solver_backend" in payload


def test_reference_optimize_openapi_schema_is_typed(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    optimize_schema = schema["paths"]["/reference/optimize"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert optimize_schema["$ref"].endswith("/ReferenceOptimizeRequest")


def test_shard_search_override_helper_includes_extended_runtime_knobs() -> None:
    request = SearchRequest(
        vector=[1.0, 0.0],
        top_k=1,
        n_full_scores=128,
        variable_length_strategy="bucketed",
        pinned_pool_buffers=4,
        pinned_buffer_max_tokens=12345,
    )

    overrides = SearchService._shard_search_overrides(request)

    assert overrides["n_full_scores"] == 128
    assert overrides["variable_length_strategy"] == "bucketed"
    assert overrides["pinned_pool_buffers"] == 4
    assert overrides["pinned_buffer_max_tokens"] == 12345


def test_reference_preprocess_documents_renders_page_bundle(tmp_path: Path) -> None:
    source = _write_png(tmp_path / "page.png")
    output_dir = tmp_path / "rendered-pages"

    with _create_client(tmp_path / "index") as client:
        response = client.post(
            "/reference/preprocess/documents",
            json={
                "source_paths": [str(source)],
                "output_dir": str(output_dir),
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["documents_rendered"] == 1
    assert payload["summary"]["pages_rendered"] == 1
    assert payload["bundles"][0]["pages"][0]["page_id"].startswith("page-")
    assert Path(payload["bundles"][0]["pages"][0]["image_path"]).exists()


def test_reference_preprocess_documents_requires_sources(tmp_path: Path) -> None:
    with _create_client(tmp_path / "index") as client:
        response = client.post("/reference/preprocess/documents", json={})

    assert response.status_code == 422
    assert "source_paths" in response.text or "source_dir" in response.text


def test_reference_optimize_dense_contract(tmp_path: Path) -> None:
    pytest.importorskip("latence_solver")

    with _create_client(tmp_path) as client:
        response = client.post(
            "/reference/optimize",
            json={
                "query_text": "invoice total due",
                "query_vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                "candidates": [
                    {
                        "chunk_id": "invoice",
                        "text": "invoice total due",
                        "token_count": 64,
                        "vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                        "metadata": {"dense_score": 1.0, "sparse_score": 2.5, "rrf_score": 0.03},
                    },
                    {
                        "chunk_id": "report",
                        "text": "board report summary",
                        "token_count": 96,
                        "vectors": _optimizer_vector_payload([0.0, 1.0, 0.0, 0.0]),
                        "metadata": {"dense_score": 0.1, "sparse_score": 0.2, "rrf_score": 0.01},
                    },
                ],
                "constraints": {"max_tokens": 96, "max_chunks": 1, "max_per_cluster": 1},
                "solver_config": {"iterations": 16},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_ids"] == ["invoice"]
    assert payload["solver_output"]["num_selected"] == 1
    assert payload["solver_output"]["constraints_satisfied"] is True


def test_reference_optimize_accepts_lambda_alias(tmp_path: Path) -> None:
    pytest.importorskip("latence_solver")

    with _create_client(tmp_path) as client:
        response = client.post(
            "/reference/optimize",
            json={
                "query_text": "invoice total due",
                "query_vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                "candidates": [
                    {
                        "chunk_id": "invoice",
                        "text": "invoice total due",
                        "token_count": 64,
                        "vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                    },
                    {
                        "chunk_id": "report",
                        "text": "board report summary",
                        "token_count": 96,
                        "vectors": _optimizer_vector_payload([0.0, 1.0, 0.0, 0.0]),
                    },
                ],
                "constraints": {"max_tokens": 96, "max_chunks": 1, "max_per_cluster": 1},
                "solver_config": {"iterations": 16, "lambda": 0.2},
            },
        )

    assert response.status_code == 200
    assert response.json()["solver_output"]["constraints_satisfied"] is True


def test_reference_optimize_accepts_named_post_rerank_policy(tmp_path: Path) -> None:
    def fake_optimize(self, body):
        assert body["solver_config"]["optimizer_policy"] == "post_rerank_v1"
        return {
            "selected_ids": ["invoice"],
            "solver_output": {
                "selected_indices": [0],
                "objective_score": 1.0,
                "num_selected": 1,
                "solve_time_ms": 0.1,
                "constraints_satisfied": True,
                "constraint_violations": [],
            },
            "feature_summary": {"optimizer_policy_name": "post_rerank_v1"},
        }

    with patch(
        "voyager_index._internal.inference.stateless_optimizer.GpuFulfilmentPipeline.optimize",
        new=fake_optimize,
    ):
        with _create_client(tmp_path) as client:
            response = client.post(
                "/reference/optimize",
                json={
                    "query_text": "invoice total due",
                    "query_vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                    "candidates": [
                        {
                            "chunk_id": "invoice",
                            "text": "invoice total due",
                            "token_count": 64,
                            "vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                            "metadata": {
                                "dense_score": 0.2,
                                "sparse_score": 0.2,
                                "rrf_score": 0.01,
                                "cross_encoder_score": 0.95,
                                "base_relevance": 0.95,
                            },
                        },
                        {
                            "chunk_id": "report",
                            "text": "board report summary",
                            "token_count": 96,
                            "vectors": _optimizer_vector_payload([0.0, 1.0, 0.0, 0.0]),
                            "metadata": {
                                "dense_score": 0.9,
                                "sparse_score": 0.8,
                                "rrf_score": 0.05,
                                "cross_encoder_score": 0.05,
                                "base_relevance": 0.05,
                            },
                        },
                    ],
                    "constraints": {"max_tokens": 96, "max_chunks": 1, "max_per_cluster": 1},
                    "solver_config": {"iterations": 16, "optimizer_policy": "post_rerank_v1"},
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_ids"] == ["invoice"]
    assert payload["feature_summary"]["optimizer_policy_name"] == "post_rerank_v1"


def test_reference_optimize_payload_limit_returns_413(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"VOYAGER_OPTIMIZER_MAX_PAYLOAD_BYTES": "1"}):
        with _create_client(tmp_path) as client:
            response = client.post(
                "/reference/optimize",
                json={
                    "query_text": "invoice total due",
                    "query_vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                    "candidates": [
                        {
                            "chunk_id": "invoice",
                            "text": "invoice total due",
                            "token_count": 64,
                            "vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                        }
                    ],
                },
            )

    assert response.status_code == 413
    assert "configured limit" in response.json()["detail"]


def test_collection_optimize_route_points_to_reference_optimize(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        response = client.post(
            "/collections/demo/optimize",
            json={"vector": [1.0, 0.0], "query_text": "invoice", "top_k": 1},
        )

    assert response.status_code == 501
    assert "/reference/optimize" in response.json()["detail"]


def test_scroll_and_retrieve_require_shard_collections(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )

        scroll = client.post("/collections/dense/scroll", json={"limit": 10, "offset": 0})
        retrieve = client.post("/collections/dense/retrieve", json={"ids": [1]})

    assert scroll.status_code == 400
    assert scroll.json()["code"] == "validation_error"
    assert "not a shard collection" in scroll.json()["detail"]
    assert retrieve.status_code == 400
    assert retrieve.json()["code"] == "validation_error"
    assert "not a shard collection" in retrieve.json()["detail"]


def test_shard_collection_info_exposes_extended_runtime_knobs(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        create_response = client.post(
            "/collections/shard-info",
            json={
                "dimension": 4,
                "kind": "shard",
                "n_shards": 8,
                "max_docs_exact": 222,
                "n_full_scores": 333,
                "lemur_search_k_cap": 444,
                "pinned_pool_buffers": 5,
                "pinned_buffer_max_tokens": 12345,
                "transfer_mode": "double_buffered",
                "variable_length_strategy": "bucketed",
            },
        )
        assert create_response.status_code == 200

        response = client.get("/collections/shard-info/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["max_docs_exact"] == 222
    assert payload["n_full_scores"] == 333
    assert payload["lemur_search_k_cap"] == 444
    assert payload["pinned_pool_buffers"] == 5
    assert payload["pinned_buffer_max_tokens"] == 12345
    assert payload["transfer_mode"] == "double_buffered"
    assert payload["variable_length_strategy"] == "bucketed"


def test_dense_collection_applies_dot_distance_metric(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dot",
                json={"dimension": 2, "kind": "dense", "distance": "dot"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dot/points",
                json={
                    "points": [
                        {"id": "wide", "vector": [10, 0], "payload": {"text": "wide alpha"}},
                        {"id": "unit", "vector": [1, 1], "payload": {"text": "unit beta"}},
                    ]
                },
            ).status_code
            == 200
        )

        response = client.post(
            "/collections/dot/search",
            json={"vector": [1, 1], "top_k": 2},
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "wide"


def test_dense_text_only_search_is_sparse_only_and_filtered(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": "keep",
                            "vector": [1, 0],
                            "payload": {"text": "beta ontology keep", "label": "keep"},
                        },
                        {
                            "id": "drop",
                            "vector": [0, 1],
                            "payload": {"text": "beta ontology drop", "label": "drop"},
                        },
                    ]
                },
            ).status_code
            == 200
        )

        response = client.post(
            "/collections/dense/search",
            json={"query_text": "beta", "filter": {"label": "keep"}, "top_k": 5},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["results"][0]["id"] == "keep"
    assert payload["results"][0]["payload"]["label"] == "keep"


def test_dense_collection_accepts_base64_vector_transport(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": "alpha",
                            "vector": _optimizer_vector_payload([1.0, 0.0]),
                            "payload": {"text": "alpha"},
                        }
                    ]
                },
            ).status_code
            == 200
        )

        response = client.post(
            "/collections/dense/search",
            json={"vector": _optimizer_vector_payload([1.0, 0.0]), "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "alpha"


def test_dense_optimized_search_returns_solver_metadata(tmp_path: Path) -> None:
    latence_solver = pytest.importorskip("latence_solver")
    _ = latence_solver

    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": "alpha",
                            "vector": [1, 0],
                            "payload": {"text": "voyager target alpha", "token_count": 80},
                        },
                        {
                            "id": "beta",
                            "vector": [0.8, 0.2],
                            "payload": {"text": "voyager supporting beta", "token_count": 90},
                        },
                        {"id": "gamma", "vector": [0, 1], "payload": {"text": "unrelated gamma", "token_count": 120}},
                    ]
                },
            ).status_code
            == 200
        )

        response = client.post(
            "/collections/dense/search",
            json={
                "vector": [1, 0],
                "query_text": "voyager target alpha",
                "top_k": 3,
                "strategy": "optimized",
                "max_tokens": 220,
                "max_chunks": 2,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["id"] == "alpha"
    assert payload["objective_score"] is not None
    assert payload["total_tokens"] is not None
    assert payload["total"] <= 2


def test_dense_optimized_search_gate_skips_solver_for_small_live_pools(tmp_path: Path) -> None:
    latence_solver = pytest.importorskip("latence_solver")
    _ = latence_solver

    class FailPipeline:
        def optimize_in_process(self, **kwargs):
            _ = kwargs
            raise AssertionError("production gate should skip solver on small candidate pools")

    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": "alpha",
                            "vector": [1, 0],
                            "payload": {"text": "voyager target alpha", "token_count": 80},
                        },
                        {
                            "id": "beta",
                            "vector": [0.8, 0.2],
                            "payload": {"text": "voyager supporting beta", "token_count": 90},
                        },
                        {"id": "gamma", "vector": [0, 1], "payload": {"text": "unrelated gamma", "token_count": 120}},
                    ]
                },
            ).status_code
            == 200
        )

        runtime = client.app.state.search_service.get_collection("dense")
        runtime.engine._optimizer_pipeline = FailPipeline()

        response = client.post(
            "/collections/dense/search",
            json={
                "vector": [1, 0],
                "query_text": "voyager target alpha",
                "top_k": 3,
                "strategy": "optimized",
                "max_tokens": 220,
                "max_chunks": 2,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["id"] == "alpha"
    assert payload["objective_score"] == pytest.approx(0.0)
    assert payload["total_tokens"] is not None
    assert payload["total"] <= 2


def test_dense_optimized_search_accepts_policy_and_solver_overrides(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": "alpha",
                            "vector": [1, 0],
                            "payload": {"text": "voyager target alpha", "token_count": 80},
                        },
                        {
                            "id": "beta",
                            "vector": [0.8, 0.2],
                            "payload": {"text": "voyager supporting beta", "token_count": 90},
                        },
                        {"id": "gamma", "vector": [0, 1], "payload": {"text": "unrelated gamma", "token_count": 120}},
                    ]
                },
            ).status_code
            == 200
        )

        runtime = client.app.state.search_service.get_collection("dense")
        original_solver_available = runtime.engine.solver_available
        runtime.engine.solver_available = True

        def fake_refine(
            *,
            query_vector,
            query_text,
            query_payload,
            candidate_ids,
            solver_config,
            constraints,
            optimizer_policy,
            refine_options,
        ):
            assert query_text == "voyager target alpha"
            assert query_payload == {"label": "evidence"}
            assert solver_config["iterations"] == 24
            assert solver_config["lambda"] == 0.2
            assert optimizer_policy == "post_rerank_v1"
            assert refine_options is None
            return {
                "selected_internal_ids": [candidate_ids[0], candidate_ids[1]],
                "solver_output": {
                    "objective_score": 1.5,
                    "selected_indices": [0, 1],
                    "num_selected": 2,
                    "solve_time_ms": 0.2,
                    "constraints_satisfied": True,
                    "constraint_violations": [],
                    "total_tokens": 170,
                },
                "backend_kind": "cpu_reference",
            }

        with patch.object(runtime.engine, "refine", side_effect=fake_refine):
            response = client.post(
                "/collections/dense/search",
                json={
                    "vector": [1, 0],
                    "query_text": "voyager target alpha",
                    "query_payload": {"label": "evidence"},
                    "top_k": 3,
                    "strategy": "optimized",
                    "solver_config": {"iterations": 24, "lambda": 0.2},
                    "optimizer_policy": "post_rerank_v1",
                },
            )
        runtime.engine.solver_available = original_solver_available

    assert response.status_code == 200
    payload = response.json()
    assert payload["objective_score"] is not None
    assert payload["total"] <= 2


def test_dense_optimized_search_accepts_cross_encoder_refine_flag(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": "alpha",
                            "vector": [1, 0],
                            "payload": {"text": "voyager target alpha", "token_count": 80},
                        },
                        {
                            "id": "beta",
                            "vector": [0.8, 0.2],
                            "payload": {"text": "voyager supporting beta", "token_count": 90},
                        },
                        {"id": "gamma", "vector": [0, 1], "payload": {"text": "unrelated gamma", "token_count": 120}},
                    ]
                },
            ).status_code
            == 200
        )

        runtime = client.app.state.search_service.get_collection("dense")
        original_solver_available = runtime.engine.solver_available
        runtime.engine.solver_available = True

        def fake_refine(
            *,
            query_vector,
            query_text,
            query_payload,
            candidate_ids,
            solver_config,
            constraints,
            optimizer_policy,
            refine_options,
        ):
            assert query_text == "voyager target alpha"
            assert optimizer_policy is None
            assert refine_options == {
                "use_cross_encoder": True,
                "cross_encoder_top_k": 3,
                "cross_encoder_batch_size": 2,
            }
            return {
                "selected_internal_ids": [candidate_ids[0], candidate_ids[1]],
                "solver_output": {
                    "objective_score": 2.0,
                    "selected_indices": [0, 1],
                    "num_selected": 2,
                    "solve_time_ms": 0.3,
                    "constraints_satisfied": True,
                    "constraint_violations": [],
                    "total_tokens": 170,
                },
                "backend_kind": "cpu_reference",
            }

        with patch.object(runtime.engine, "refine", side_effect=fake_refine):
            response = client.post(
                "/collections/dense/search",
                json={
                    "vector": [1, 0],
                    "query_text": "voyager target alpha",
                    "top_k": 3,
                    "strategy": "optimized",
                    "refine_use_cross_encoder": True,
                    "refine_cross_encoder_top_k": 3,
                    "refine_cross_encoder_batch_size": 2,
                },
            )
        runtime.engine.solver_available = original_solver_available

    assert response.status_code == 200
    payload = response.json()
    assert payload["objective_score"] is not None
    assert payload["results"][0]["id"] == "alpha"


def test_dense_optimized_search_accepts_nli_refine_flags(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": "alpha",
                            "vector": [1, 0],
                            "payload": {"text": "voyager target alpha", "token_count": 80},
                        },
                        {
                            "id": "beta",
                            "vector": [0.8, 0.2],
                            "payload": {"text": "voyager supporting beta", "token_count": 90},
                        },
                        {"id": "gamma", "vector": [0, 1], "payload": {"text": "unrelated gamma", "token_count": 120}},
                    ]
                },
            ).status_code
            == 200
        )

        runtime = client.app.state.search_service.get_collection("dense")
        original_solver_available = runtime.engine.solver_available
        runtime.engine.solver_available = True

        def fake_refine(
            *,
            query_vector,
            query_text,
            query_payload,
            candidate_ids,
            solver_config,
            constraints,
            optimizer_policy,
            refine_options,
        ):
            assert query_text == "voyager target alpha"
            assert optimizer_policy is None
            assert refine_options == {
                "use_nli": True,
                "nli_model": "dummy-nli-model",
                "nli_top_k": 4,
                "nli_batch_size": 2,
                "nli_promote_base_relevance": True,
                "confidence_gating": True,
                "confidence_gap_threshold": 0.25,
                "confidence_min_candidates": 2,
            }
            return {
                "selected_internal_ids": [candidate_ids[0], candidate_ids[1]],
                "solver_output": {
                    "objective_score": 2.1,
                    "selected_indices": [0, 1],
                    "num_selected": 2,
                    "solve_time_ms": 0.3,
                    "constraints_satisfied": True,
                    "constraint_violations": [],
                    "total_tokens": 170,
                },
                "backend_kind": "cpu_reference",
            }

        with patch.object(runtime.engine, "refine", side_effect=fake_refine):
            response = client.post(
                "/collections/dense/search",
                json={
                    "vector": [1, 0],
                    "query_text": "voyager target alpha",
                    "top_k": 3,
                    "strategy": "optimized",
                    "refine_use_nli": True,
                    "refine_nli_model": "dummy-nli-model",
                    "refine_nli_top_k": 4,
                    "refine_nli_batch_size": 2,
                    "refine_nli_promote_base_relevance": True,
                    "refine_confidence_gating": True,
                    "refine_confidence_gap_threshold": 0.25,
                    "refine_confidence_min_candidates": 2,
                },
            )
        runtime.engine.solver_available = original_solver_available

    assert response.status_code == 200
    payload = response.json()
    assert payload["objective_score"] is not None
    assert payload["results"][0]["id"] == "alpha"


def test_dense_optimized_search_uses_wider_candidate_pool(tmp_path: Path) -> None:
    pytest.importorskip("latence_solver")

    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": f"doc-{idx}",
                            "vector": [1.0, 0.0] if idx == 0 else [0.1 * idx, 1.0],
                            "payload": {"text": f"voyager {idx}", "token_count": 64 + idx},
                        }
                        for idx in range(8)
                    ]
                },
            ).status_code
            == 200
        )

        service = client.app.state.search_service
        runtime = service.get_collection("dense")
        internal_ids = [int(runtime.meta["records"][f"doc-{idx}"]["internal_id"]) for idx in range(8)]
        original_solver_available = runtime.engine.solver_available
        runtime.engine.solver_available = True

        def fake_search(*, query_text, query_vector, k, filters):
            assert k > 2
            return {
                "dense": [(doc_id, 1.0 - rank * 0.01) for rank, doc_id in enumerate(internal_ids[:6])],
                "sparse": [],
                "union_ids": internal_ids[:6],
                "sparse_error": None,
            }

        def fake_refine(*, query_vector, query_text, candidate_ids, constraints, **kwargs):
            assert len(candidate_ids) > 2
            return {
                "selected_internal_ids": [candidate_ids[1], candidate_ids[0]],
                "solver_output": {
                    "selected_indices": [1, 0],
                    "objective_score": 1.23,
                    "num_selected": 2,
                    "solve_time_ms": 0.5,
                    "constraints_satisfied": True,
                    "constraint_violations": [],
                    "total_tokens": 130,
                },
                "backend_kind": "cpu_reference",
            }

        with (
            patch.object(runtime.engine, "search", side_effect=fake_search),
            patch.object(runtime.engine, "refine", side_effect=fake_refine),
        ):
            response = client.post(
                "/collections/dense/search",
                json={
                    "vector": [1.0, 0.0],
                    "query_text": "voyager",
                    "top_k": 2,
                    "strategy": "optimized",
                },
            )

        runtime.engine.solver_available = original_solver_available

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["objective_score"] == pytest.approx(1.23)


def test_late_interaction_filter_and_with_vector_are_truthful(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/li",
                json={"dimension": 2, "kind": "late_interaction"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/li/points",
                json={
                    "points": [
                        {
                            "id": "keep",
                            "vectors": [[1, 0], [1, 0]],
                            "payload": {"label": "keep", "text": "alpha keep"},
                        },
                        {
                            "id": "drop",
                            "vectors": [[0, 1], [0, 1]],
                            "payload": {"label": "drop", "text": "beta drop"},
                        },
                    ]
                },
            ).status_code
            == 200
        )

        service = client.app.state.search_service
        runtime = service.get_collection("li")
        keep_internal = int(runtime.meta["records"]["keep"]["internal_id"])

        def fake_search(query_tensor, top_k, collection_name=None, doc_ids=None):
            assert doc_ids == [keep_internal]
            assert top_k == 2
            return (
                torch.tensor([[0.99]], dtype=torch.float32),
                torch.tensor([[keep_internal]], dtype=torch.long),
            )

        with patch.object(runtime.engine, "search", side_effect=fake_search):
            response = client.post(
                "/collections/li/search",
                json={
                    "vectors": [[1, 0], [1, 0]],
                    "filter": {"label": "keep"},
                    "with_vector": True,
                    "top_k": 2,
                },
            )

    assert response.status_code == 200
    result = response.json()["results"]
    assert len(result) == 1
    assert result[0]["id"] == "keep"
    assert result[0]["payload"]["label"] == "keep"
    assert result[0]["vectors"] == [[1.0, 0.0], [1.0, 0.0]]


def test_late_interaction_accepts_base64_multivector_transport(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/li",
                json={"dimension": 2, "kind": "late_interaction"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/li/points",
                json={
                    "points": [
                        {
                            "id": "keep",
                            "vectors": _optimizer_vector_payload([[1.0, 0.0], [1.0, 0.0]]),
                            "payload": {"label": "keep", "text": "alpha keep"},
                        }
                    ]
                },
            ).status_code
            == 200
        )

        response = client.post(
            "/collections/li/search",
            json={
                "vectors": _optimizer_vector_payload([[1.0, 0.0], [1.0, 0.0]]),
                "with_vector": True,
                "top_k": 1,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload["results"]] == ["keep"]
    assert payload["results"][0]["vectors"] == [[1.0, 0.0], [1.0, 0.0]]


def test_multimodal_filter_and_with_vector_are_truthful(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/mm",
                json={"dimension": 2, "kind": "multimodal"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/mm/points",
                json={
                    "points": [
                        {
                            "id": "keep",
                            "vectors": [[1, 0], [1, 0]],
                            "payload": {"label": "keep", "page_number": 1},
                        },
                        {
                            "id": "drop",
                            "vectors": [[0, 1], [0, 1]],
                            "payload": {"label": "drop", "page_number": 2},
                        },
                    ]
                },
            ).status_code
            == 200
        )

        service = client.app.state.search_service
        runtime = service.get_collection("mm")
        keep_internal = int(runtime.meta["records"]["keep"]["internal_id"])

        def fake_search(*, query_embedding, top_k, collection_name=None, candidate_ids=None):
            assert candidate_ids == [keep_internal]
            assert top_k == 2
            return [
                SearchResult(
                    doc_id=keep_internal,
                    score=0.99,
                    rank=1,
                    source="colpali",
                    metadata={"collection": collection_name},
                )
            ]

        with patch.object(runtime.engine, "search", side_effect=fake_search):
            response = client.post(
                "/collections/mm/search",
                json={
                    "vectors": [[1, 0], [1, 0]],
                    "filter": {"label": "keep"},
                    "with_vector": True,
                    "top_k": 2,
                },
            )

    assert response.status_code == 200
    result = response.json()["results"]
    assert len(result) == 1
    assert result[0]["id"] == "keep"
    assert result[0]["payload"]["label"] == "keep"
    assert result[0]["vectors"] == [[1.0, 0.0], [1.0, 0.0]]


def test_multimodal_solver_prefilter_mode_applies_solver_subset(tmp_path: Path) -> None:
    class FakePipeline:
        def __init__(self, selected_id: int):
            self.selected_id = selected_id

        def optimize_in_process(self, **kwargs):
            _ = kwargs
            return {
                "selected_ids": [str(self.selected_id)],
                "solver_output": {"objective_score": 1.5},
                "feature_summary": {"candidate_count": 2},
            }

    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/mm",
                json={"dimension": 2, "kind": "multimodal"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/mm/points",
                json={
                    "points": [
                        {
                            "id": "doc-a",
                            "vectors": [[1, 0], [1, 0]],
                            "payload": {"text": "alpha keep", "token_count": 64},
                        },
                        {
                            "id": "doc-b",
                            "vectors": [[0, 1], [0, 1]],
                            "payload": {"text": "beta forced", "token_count": 72},
                        },
                    ]
                },
            ).status_code
            == 200
        )

        service = client.app.state.search_service
        runtime = service.get_collection("mm")
        doc_b_internal = int(runtime.meta["records"]["doc-b"]["internal_id"])
        fake_pipeline = FakePipeline(doc_b_internal)

        with patch.object(service, "_get_multimodal_optimizer_pipeline", return_value=fake_pipeline):
            response = client.post(
                "/collections/mm/search",
                json={
                    "vectors": [[1, 0], [1, 0]],
                    "top_k": 1,
                    "strategy": "optimized",
                    "multimodal_optimize_mode": "solver_prefilter_maxsim",
                    "multimodal_prefilter_k": 1,
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["id"] == "doc-b"
    assert payload["objective_score"] == pytest.approx(1.5)
    profile = runtime.engine.last_search_profile
    assert profile["optimization"]["mode"] == "solver_prefilter_maxsim"
    assert profile["optimization"]["solver_selected_count"] == 1


def test_multimodal_maxsim_then_solver_mode_packs_exact_frontier(tmp_path: Path) -> None:
    class FakePipeline:
        def __init__(self, selected_id: int):
            self.selected_id = selected_id

        def optimize_in_process(self, **kwargs):
            _ = kwargs
            return {
                "selected_ids": [str(self.selected_id)],
                "solver_output": {"objective_score": 2.25},
                "feature_summary": {"candidate_count": 2},
            }

    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/mm",
                json={"dimension": 2, "kind": "multimodal"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/mm/points",
                json={
                    "points": [
                        {
                            "id": "doc-a",
                            "vectors": [[1, 0], [1, 0]],
                            "payload": {"text": "alpha top", "token_count": 64},
                        },
                        {
                            "id": "doc-b",
                            "vectors": [[0.8, 0.2], [0.8, 0.2]],
                            "payload": {"text": "beta packed", "token_count": 72},
                        },
                    ]
                },
            ).status_code
            == 200
        )

        service = client.app.state.search_service
        runtime = service.get_collection("mm")
        doc_b_internal = int(runtime.meta["records"]["doc-b"]["internal_id"])
        fake_pipeline = FakePipeline(doc_b_internal)

        with patch.object(service, "_get_multimodal_optimizer_pipeline", return_value=fake_pipeline):
            response = client.post(
                "/collections/mm/search",
                json={
                    "vectors": [[1, 0], [1, 0]],
                    "top_k": 1,
                    "strategy": "optimized",
                    "multimodal_optimize_mode": "maxsim_then_solver",
                    "multimodal_maxsim_frontier_k": 2,
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["id"] == "doc-b"
    assert payload["objective_score"] == pytest.approx(2.25)
    profile = runtime.engine.last_search_profile
    assert profile["optimization"]["mode"] == "maxsim_then_solver"
    assert profile["optimization"]["exact_frontier_size"] == 2


def test_multimodal_optimized_auto_falls_back_to_maxsim_only_without_solver(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/mm",
                json={"dimension": 2, "kind": "multimodal"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/mm/points",
                json={
                    "points": [
                        {
                            "id": "doc-a",
                            "vectors": [[1, 0], [1, 0]],
                            "payload": {"text": "alpha top"},
                        },
                        {
                            "id": "doc-b",
                            "vectors": [[0, 1], [0, 1]],
                            "payload": {"text": "beta"},
                        },
                    ]
                },
            ).status_code
            == 200
        )

        service = client.app.state.search_service
        runtime = service.get_collection("mm")
        with patch.object(service, "_multimodal_solver_available", return_value=False):
            response = client.post(
                "/collections/mm/search",
                json={
                    "vectors": [[1, 0], [1, 0]],
                    "top_k": 1,
                    "strategy": "optimized",
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["id"] == "doc-a"
    assert payload["objective_score"] is None
    assert runtime.engine.last_search_profile["optimization"]["mode"] == "maxsim_only"


def test_multimodal_optimized_defaults_to_measured_maxsim_only_path(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/mm",
                json={"dimension": 2, "kind": "multimodal"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/mm/points",
                json={
                    "points": [
                        {
                            "id": "doc-a",
                            "vectors": [[1, 0], [1, 0]],
                            "payload": {"text": "alpha top"},
                        },
                        {
                            "id": "doc-b",
                            "vectors": [[0, 1], [0, 1]],
                            "payload": {"text": "beta"},
                        },
                    ]
                },
            ).status_code
            == 200
        )

        service = client.app.state.search_service
        runtime = service.get_collection("mm")
        with patch.object(
            service,
            "_get_multimodal_optimizer_pipeline",
            side_effect=AssertionError("solver pipeline should not be used for default multimodal optimized search"),
        ):
            response = client.post(
                "/collections/mm/search",
                json={
                    "vectors": [[1, 0], [1, 0]],
                    "top_k": 1,
                    "strategy": "optimized",
                },
            )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "doc-a"
    assert runtime.engine.last_search_profile["optimization"]["mode"] == "maxsim_only"


def test_create_collection_rejects_unsupported_kind_specific_options(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        response = client.post(
            "/collections/mm",
            json={"dimension": 4, "kind": "multimodal", "distance": "dot"},
        )

    assert response.status_code == 422


def test_dense_search_rejects_multivector_queries(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 2, "kind": "dense"},
            ).status_code
            == 200
        )

        response = client.post(
            "/collections/dense/search",
            json={"vectors": [[1, 0], [1, 0]], "top_k": 2},
        )

    assert response.status_code == 400
    assert "Dense search requires 'vector'" in response.json()["detail"]


def test_scan_ceiling_rejections_surface_in_metrics_and_ready(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/li",
                json={"dimension": 2, "kind": "late_interaction"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/li/points",
                json={
                    "points": [
                        {
                            "id": "a",
                            "vectors": [[1, 0], [1, 0]],
                            "payload": {"group": "same", "text": "alpha one"},
                        },
                        {
                            "id": "b",
                            "vectors": [[1, 0], [1, 0]],
                            "payload": {"group": "same", "text": "alpha two"},
                        },
                    ]
                },
            ).status_code
            == 200
        )

        client.app.state.search_service.filter_scan_limit = 1
        response = client.post(
            "/collections/li/search",
            json={"vectors": [[1, 0], [1, 0]], "filter": {"group": "same"}, "top_k": 1},
        )
        metrics = client.get("/metrics")
        ready = client.get("/ready")

    assert response.status_code == 503
    assert "scan ceiling" in response.json()["detail"]
    assert "voyager_filter_scan_limit_hits_total 1" in metrics.text
    assert ready.status_code == 503
    assert ready.json()["status"] == "degraded"
    assert any(issue["reason"] == "filter_scan_limit_hits" for issue in ready.json()["issues"])
