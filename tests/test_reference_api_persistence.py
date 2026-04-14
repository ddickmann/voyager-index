from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from voyager_index._internal.server.api.models import (
    CollectionKind,
    CreateCollectionRequest,
    PointVector,
    SearchRequest,
)
from voyager_index._internal.server.api.service import SearchService
from voyager_index._internal.server.main import create_app


def _create_client(index_path: Path) -> TestClient:
    return TestClient(create_app(index_path=str(index_path)))


def test_dense_collection_persists_across_restart(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/dense",
                json={"dimension": 4, "kind": "dense"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/dense/points",
                json={"points": [{"id": "doc-1", "vector": [1, 0, 0, 0], "payload": {"text": "alpha"}}]},
            ).status_code
            == 200
        )

    with _create_client(tmp_path) as client:
        info = client.get("/collections/dense/info")
        search = client.post(
            "/collections/dense/search",
            json={"vector": [1, 0, 0, 0], "top_k": 1},
        )
        sparse = client.post(
            "/collections/dense/search",
            json={"query_text": "alpha", "top_k": 1},
        )
        assert info.status_code == 200
        assert info.json()["num_points"] == 1
        assert search.status_code == 200
        assert search.json()["results"][0]["id"] == "doc-1"
        assert sparse.status_code == 200
        assert sparse.json()["results"][0]["id"] == "doc-1"


def test_late_interaction_collection_persists_across_restart(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/li",
                json={"dimension": 4, "kind": "late_interaction"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/li/points",
                json={
                    "points": [
                        {
                            "id": "doc-1",
                            "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                            "payload": {"text": "alpha"},
                        }
                    ]
                },
            ).status_code
            == 200
        )

    with _create_client(tmp_path) as client:
        info = client.get("/collections/li/info")
        search = client.post(
            "/collections/li/search",
            json={"vectors": [[1, 0, 0, 0], [1, 0, 0, 0]], "top_k": 1},
        )
        assert info.status_code == 200
        assert info.json()["num_points"] == 1
        assert search.status_code == 200
        assert search.json()["results"][0]["id"] == "doc-1"


def test_dense_mutations_become_visible_across_service_instances(tmp_path: Path) -> None:
    service_a = SearchService(str(tmp_path))
    service_a.create_collection("dense", CreateCollectionRequest(dimension=4, kind="dense"))
    service_b = SearchService(str(tmp_path))

    try:
        assert (
            service_a.add_points(
                "dense",
                [PointVector(id="doc-1", vector=[1, 0, 0, 0], payload={"text": "alpha"})],
            )
            == 1
        )

        first = service_b.search(
            "dense",
            SearchRequest(query_text="alpha", top_k=1),
        )
        assert [item.id for item in first.results] == ["doc-1"]

        assert (
            service_a.add_points(
                "dense",
                [PointVector(id="doc-2", vector=[0, 1, 0, 0], payload={"text": "beta"})],
            )
            == 1
        )

        second = service_b.search(
            "dense",
            SearchRequest(query_text="beta", top_k=1),
        )
        assert [item.id for item in second.results] == ["doc-2"]
    finally:
        service_a.close()
        service_b.close()


def test_multimodal_collection_persists_across_restart(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/mm",
                json={"dimension": 4, "kind": "multimodal"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/mm/points",
                json={
                    "points": [
                        {
                            "id": "page-1",
                            "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                            "payload": {"page_number": 1},
                        }
                    ]
                },
            ).status_code
            == 200
        )

    with _create_client(tmp_path) as client:
        info = client.get("/collections/mm/info")
        search = client.post(
            "/collections/mm/search",
            json={"vectors": [[1, 0, 0, 0], [1, 0, 0, 0]], "top_k": 1},
        )
        assert info.status_code == 200
        assert info.json()["num_points"] == 1
        assert search.status_code == 200
        assert search.json()["results"][0]["id"] == "page-1"


def test_corrupted_multimodal_screening_state_surfaces_as_load_failure(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/mm",
                json={"dimension": 4, "kind": "multimodal"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/mm/points",
                json={
                    "points": [
                        {
                            "id": "page-1",
                            "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                            "payload": {"page_number": 1},
                        }
                    ]
                },
            ).status_code
            == 200
        )

    screening_state_path = tmp_path / "mm" / "colpali" / "screening_state.json"
    screening_state_path.write_text("{not-valid-json", encoding="utf-8")

    with _create_client(tmp_path) as client:
        ready = client.get("/ready")
        info = client.get("/collections/mm/info")

    assert ready.status_code == 503
    assert ready.json()["status"] == "degraded"
    assert "mm" in ready.json()["failed_collection_loads"]
    assert any(
        issue["reason"] == "collection_load_failed" and issue["name"] == "mm" for issue in ready.json()["issues"]
    )
    assert info.status_code == 404


def test_late_interaction_with_vector_handles_unsorted_internal_ids(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert (
            client.post(
                "/collections/li",
                json={"dimension": 4, "kind": "late_interaction"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/collections/li/points",
                json={
                    "points": [
                        {
                            "id": "doc-1",
                            "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                            "payload": {"text": "alpha"},
                        },
                        {
                            "id": "doc-2",
                            "vectors": [[0, 1, 0, 0], [0, 1, 0, 0]],
                            "payload": {"text": "beta"},
                        },
                    ]
                },
            ).status_code
            == 200
        )

    with _create_client(tmp_path) as client:
        response = client.post(
            "/collections/li/search",
            json={"vectors": [[0, 1, 0, 0], [0, 1, 0, 0]], "top_k": 2, "with_vector": True},
        )

    assert response.status_code == 200
    payload = response.json()["results"]
    assert [item["id"] for item in payload] == ["doc-2", "doc-1"]
    assert payload[0]["vectors"] == [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]


def test_uncommitted_dense_mutation_recovers_on_restart(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("dense", CreateCollectionRequest(dimension=2, kind="dense"))
    assert (
        service.add_points(
            "dense",
            [PointVector(id="doc-1", vector=[1, 0], payload={"text": "alpha"})],
        )
        == 1
    )

    runtime = service.get_collection("dense")
    backup = service._begin_collection_mutation(runtime, operation="add_points")
    internal_id = service._next_internal_id(runtime)
    payload = {"text": "beta", "external_id": "doc-2"}
    runtime.engine.hnsw.add(np.asarray([[0.0, 1.0]], dtype=np.float32), ids=[internal_id], payloads=[payload])
    runtime.meta["records"]["doc-2"] = {
        "external_id": "doc-2",
        "internal_id": internal_id,
        "payload": payload,
        "text": "beta",
    }
    service._sync_dense_sparse_state(runtime, rebuild_sparse=False)
    service._write_meta(runtime)
    assert (backup.backup_root / "journal.json").exists()

    service.close()
    recovered = SearchService(str(tmp_path))
    info = recovered.collection_info("dense")
    response = recovered.search(
        "dense",
        SearchRequest(vector=[1, 0], top_k=5),
    )

    assert info.num_points == 1
    assert [item.id for item in response.results] == ["doc-1"]
    assert not backup.backup_root.exists()


def test_uncommitted_late_interaction_mutation_recovers_on_restart(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("li", CreateCollectionRequest(dimension=2, kind="late_interaction"))
    assert (
        service.add_points(
            "li",
            [PointVector(id="doc-1", vectors=[[1, 0], [1, 0]], payload={"text": "alpha"})],
        )
        == 1
    )

    runtime = service.get_collection("li")
    backup = service._begin_collection_mutation(runtime, operation="add_points")
    tensor = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
    assigned_ids = runtime.engine.add_documents(tensor, metadata=[{"text": "beta", "external_id": "doc-2"}])
    runtime.meta["records"]["doc-2"] = {
        "external_id": "doc-2",
        "internal_id": int(assigned_ids[0]),
        "payload": {"text": "beta", "external_id": "doc-2"},
        "text": "beta",
    }
    runtime.meta["next_internal_id"] = max(int(runtime.meta["next_internal_id"]), int(assigned_ids[0]) + 1)
    service._refresh_runtime_indexes(runtime)
    service._write_meta(runtime)

    service.close()
    recovered = SearchService(str(tmp_path))
    info = recovered.collection_info("li")
    response = recovered.search(
        "li",
        SearchRequest(vectors=[[1, 0], [1, 0]], top_k=5),
    )

    assert info.num_points == 1
    assert [item.id for item in response.results] == ["doc-1"]
    assert not backup.backup_root.exists()


def test_uncommitted_multimodal_mutation_recovers_on_restart(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("mm", CreateCollectionRequest(dimension=2, kind="multimodal"))
    assert (
        service.add_points(
            "mm",
            [PointVector(id="page-1", vectors=[[1, 0], [1, 0]], payload={"page_number": 1})],
        )
        == 1
    )

    runtime = service.get_collection("mm")
    backup = service._begin_collection_mutation(runtime, operation="add_points")
    internal_id = service._next_internal_id(runtime)
    runtime.engine.add_documents(
        np.asarray([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32),
        doc_ids=[internal_id],
    )
    runtime.meta["records"]["page-2"] = {
        "external_id": "page-2",
        "internal_id": internal_id,
        "payload": {"page_number": 2, "external_id": "page-2"},
        "text": "",
    }
    service._refresh_runtime_indexes(runtime)
    service._write_meta(runtime)

    service.close()
    recovered = SearchService(str(tmp_path))
    info = recovered.collection_info("mm")
    response = recovered.search(
        "mm",
        SearchRequest(vectors=[[1, 0], [1, 0]], top_k=5),
    )

    assert info.num_points == 1
    assert [item.id for item in response.results] == ["page-1"]
    assert not backup.backup_root.exists()


def test_late_interaction_runtime_profiles_are_populated(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("li", CreateCollectionRequest(dimension=2, kind="late_interaction"))
    assert (
        service.add_points(
            "li",
            [PointVector(id="doc-1", vectors=[[1, 0], [1, 0]], payload={"text": "alpha"})],
        )
        == 1
    )

    runtime = service.get_collection("li")
    response = service.search(
        "li",
        SearchRequest(vectors=[[1, 0], [1, 0]], top_k=1),
    )

    assert [item.id for item in response.results] == ["doc-1"]
    assert runtime.engine.last_write_profile["mode"] == "build_from_batches"
    assert runtime.engine.last_write_profile["doc_count"] == 1
    assert runtime.engine.last_search_profile["mode"] in {"triton_cache", "triton_load", "triton_mmap"}
    assert runtime.engine.last_search_profile["doc_count"] == 1


def test_async_task_status_is_shared_across_clients(tmp_path: Path) -> None:
    with _create_client(tmp_path) as writer, _create_client(tmp_path) as reader:
        assert (
            writer.post(
                "/collections/dense",
                json={"dimension": 4, "kind": "dense"},
            ).status_code
            == 200
        )

        accepted = writer.post(
            "/collections/dense/points/async",
            json={"points": [{"id": "doc-1", "vector": [1, 0, 0, 0], "payload": {"text": "alpha"}}]},
        )
        assert accepted.status_code == 202
        task_id = accepted.json()["task_id"]

        status_payload = None
        for _ in range(40):
            status = reader.get(f"/tasks/{task_id}")
            assert status.status_code == 200
            status_payload = status.json()
            if status_payload["status"] == "completed":
                break
            time.sleep(0.05)

        assert status_payload is not None
        assert status_payload["status"] == "completed"
        assert status_payload["result"]["added"] == 1

        search = reader.post(
            "/collections/dense/search",
            json={"query_text": "alpha", "top_k": 1},
        )
        assert search.status_code == 200
        assert search.json()["results"][0]["id"] == "doc-1"


def test_uncommitted_collection_delete_recovers_on_restart(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("docs", CreateCollectionRequest(dimension=2, kind="dense"))
    backup = service._begin_delete_collection_mutation(service.get_collection("docs"))
    staged_path = backup.backup_root / "staged_collection"
    service._collection_path("docs").rename(staged_path)

    service.close()
    recovered = SearchService(str(tmp_path))

    assert recovered.collection_info("docs").name == "docs"
    assert not backup.backup_root.exists()


def test_uncommitted_collection_create_is_removed_on_restart(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    backup = service._begin_create_collection_mutation("pending", CollectionKind.DENSE)
    collection_path = service._collection_path("pending")
    collection_path.mkdir(parents=True, exist_ok=True)
    service._write_json_atomic(
        collection_path / "collection.json",
        {
            "name": "pending",
            "kind": "dense",
            "dimension": 2,
            "distance": "cosine",
            "m": 16,
            "ef_construction": 200,
            "storage_mode": "sync",
            "sparse_dirty": False,
            "created_at": service._now(),
            "updated_at": service._now(),
            "next_internal_id": 0,
            "records": {},
        },
    )

    SearchService(str(tmp_path))

    assert not collection_path.exists()
    assert not backup.backup_root.exists()
