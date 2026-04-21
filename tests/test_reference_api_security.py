from __future__ import annotations

from pathlib import Path
import shutil
from unittest.mock import patch

from fastapi.testclient import TestClient

from colsearch._internal.server.api.service import SearchService, ValidationError
from colsearch._internal.server.api.models import CreateCollectionRequest
from colsearch._internal.server.main import create_app


def _create_client(index_path: Path, *, raise_server_exceptions: bool = True) -> TestClient:
    return TestClient(
        create_app(index_path=str(index_path)),
        raise_server_exceptions=raise_server_exceptions,
    )


def test_collection_name_rejects_path_traversal(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        hidden = client.post("/collections/.hidden", json={"dimension": 4, "kind": "dense"})

    assert hidden.status_code == 400

    service = SearchService(str(tmp_path))
    try:
        service.create_collection("../evil", CreateCollectionRequest(dimension=4, kind="dense"))
    except ValidationError as exc:
        assert "single non-empty path segment" in exc.detail
    else:
        raise AssertionError("Path traversal names should be rejected")


def test_failed_dense_ingest_rolls_back_runtime_state(tmp_path: Path) -> None:
    with _create_client(tmp_path, raise_server_exceptions=False) as client:
        assert client.post(
            "/collections/dense",
            json={"dimension": 2, "kind": "dense"},
        ).status_code == 200

        service = client.app.state.search_service
        original_flush_runtime_engine = service._flush_runtime_engine

        def explode(*args, **kwargs):
            original_flush_runtime_engine(*args, **kwargs)
            raise RuntimeError("forced rollback")

        with patch.object(service, "_flush_runtime_engine", side_effect=explode):
            failed = client.post(
                "/collections/dense/points",
                json={
                    "points": [
                        {
                            "id": "doc-1",
                            "vector": [1, 0],
                            "payload": {"text": "rollback alpha"},
                        }
                    ]
                },
            )

        info = client.get("/collections/dense/info")
        search = client.post(
            "/collections/dense/search",
            json={"vector": [1, 0], "top_k": 5},
        )

    assert failed.status_code == 500
    assert info.status_code == 200
    assert info.json()["num_points"] == 0
    assert search.status_code == 200
    assert search.json()["total"] == 0


def test_delete_collection_failure_restores_collection(tmp_path: Path) -> None:
    with _create_client(tmp_path, raise_server_exceptions=False) as client:
        assert client.post(
            "/collections/docs",
            json={"dimension": 2, "kind": "dense"},
        ).status_code == 200

        service = client.app.state.search_service
        runtime = service.get_collection("docs")
        original_rmtree = shutil.rmtree

        def guarded_rmtree(path, *args, **kwargs):
            target = Path(path)
            if target.name == "staged_collection":
                raise OSError("forced delete failure")
            return original_rmtree(path, *args, **kwargs)

        with patch("colsearch._internal.server.api.service.shutil.rmtree", side_effect=guarded_rmtree):
            failed = client.delete("/collections/docs")

        info = client.get("/collections/docs/info")

    assert failed.status_code == 500
    assert info.status_code == 200
    assert info.json()["name"] == "docs"


def test_bm25_corruption_marks_readiness_degraded(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        assert client.post(
            "/collections/dense",
            json={"dimension": 2, "kind": "dense"},
        ).status_code == 200
        assert client.post(
            "/collections/dense/points",
            json={
                "points": [
                    {"id": "doc-1", "vector": [1, 0], "payload": {"text": "alpha beta"}}
                ]
            },
        ).status_code == 200
        assert client.post(
            "/collections/dense/search",
            json={"query_text": "alpha", "top_k": 1},
        ).status_code == 200

    bm25_index = tmp_path / "dense" / "hybrid" / "bm25" / "params.index.json"
    bm25_index.write_text("{broken", encoding="utf-8")

    with _create_client(tmp_path) as client:
        ready = client.get("/ready")

    assert ready.status_code == 503
    payload = ready.json()
    assert payload["status"] == "degraded"
    assert payload["degraded_collections"] == ["dense"]
    assert any(issue["reason"] == "sparse_load_failed" for issue in payload["issues"])
