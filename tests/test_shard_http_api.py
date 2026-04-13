"""HTTP integration tests for shard engine collections.

Tests the full FastAPI route → service → ShardSegmentManager path for:
  - Collection CRUD (create, info, delete)
  - Point ingestion and search (multi-vector)
  - Shard admin endpoints (compact, shards, wal/status, checkpoint)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voyager_index._internal.server.main import create_app

DIM = 32
N_TOKENS = 8
N_SHARDS = 4


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(index_path=str(tmp_path)))


def _random_multivec(n_docs: int, n_tokens: int = N_TOKENS, dim: int = DIM):
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal((n_tokens, dim)).astype(np.float32).tolist()
        for _ in range(n_docs)
    ]


def _create_shard_collection(client: TestClient, name: str = "test_shard"):
    resp = client.post(
        f"/collections/{name}",
        json={"dimension": DIM, "kind": "shard", "n_shards": N_SHARDS},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _add_points(client: TestClient, name: str, n: int = 50):
    vecs = _random_multivec(n)
    points = [
        {"id": f"doc_{i}", "vectors": vecs[i], "payload": {"tag": f"t{i % 5}"}}
        for i in range(n)
    ]
    resp = client.post(f"/collections/{name}/points", json={"points": points})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------
# Collection lifecycle
# ------------------------------------------------------------------


def test_create_shard_collection(tmp_path):
    with _client(tmp_path) as c:
        result = _create_shard_collection(c)
        assert result["status"] == "ok"

        info = c.get("/collections/test_shard/info")
        assert info.status_code == 200
        body = info.json()
        assert body["kind"] == "shard"
        assert body["dimension"] == DIM


def test_create_and_delete_shard_collection(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        resp = c.delete("/collections/test_shard")
        assert resp.status_code == 200

        collections = c.get("/collections").json()["collections"]
        assert "test_shard" not in collections


def test_duplicate_collection_rejected(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        resp = c.post(
            "/collections/test_shard",
            json={"dimension": DIM, "kind": "shard"},
        )
        assert resp.status_code == 409


# ------------------------------------------------------------------
# Points: add + search
# ------------------------------------------------------------------


def test_add_and_search_points(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        add_resp = _add_points(c, "test_shard", n=50)
        assert add_resp["added"] == 50

        query = _random_multivec(1)[0]
        search_resp = c.post(
            "/collections/test_shard/search",
            json={"vectors": query, "top_k": 5},
        )
        assert search_resp.status_code == 200
        body = search_resp.json()
        assert body["total"] <= 5
        assert len(body["results"]) <= 5
        for r in body["results"]:
            assert "id" in r
            assert "score" in r


def test_search_empty_collection(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        query = _random_multivec(1)[0]
        resp = c.post(
            "/collections/test_shard/search",
            json={"vectors": query, "top_k": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


def test_search_with_payload(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=20)
        query = _random_multivec(1)[0]
        resp = c.post(
            "/collections/test_shard/search",
            json={"vectors": query, "top_k": 3, "with_payload": True},
        )
        body = resp.json()
        for r in body["results"]:
            assert r["payload"] is not None


def test_delete_points(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=10)

        resp = c.request(
            "DELETE",
            "/collections/test_shard/points",
            json={"ids": ["doc_0", "doc_1"]},
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2


def test_search_rejects_query_text(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=10)
        resp = c.post(
            "/collections/test_shard/search",
            json={"query_text": "hello", "top_k": 3},
        )
        assert resp.status_code == 422 or resp.status_code == 400


# ------------------------------------------------------------------
# Shard admin endpoints
# ------------------------------------------------------------------


def test_list_shards(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=50)
        resp = c.get("/collections/test_shard/shards")
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection"] == "test_shard"
        assert body["num_shards"] == N_SHARDS
        assert len(body["shards"]) == N_SHARDS


def test_get_shard_detail(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=50)
        resp = c.get("/collections/test_shard/shards/0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["shard_id"] == 0
        assert body["num_docs"] > 0


def test_get_shard_detail_not_found(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=50)
        resp = c.get("/collections/test_shard/shards/9999")
        assert resp.status_code == 404


def test_wal_status(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=50)
        resp = c.get("/collections/test_shard/wal/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection"] == "test_shard"
        assert "wal_entries" in body
        assert "memtable_docs" in body


def test_compact(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=50)
        resp = c.post("/collections/test_shard/compact")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


def test_checkpoint(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=50)
        resp = c.post("/collections/test_shard/checkpoint")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert isinstance(body["wal_entries_after"], int)
        assert body["wal_entries_before"] >= body["wal_entries_after"]


def test_admin_on_non_shard_rejected(tmp_path):
    with _client(tmp_path) as c:
        c.post("/collections/dense_col", json={"dimension": DIM, "kind": "dense"})
        resp = c.get("/collections/dense_col/shards")
        assert resp.status_code in (400, 422)


# ------------------------------------------------------------------
# Strategy flag
# ------------------------------------------------------------------


def test_shard_routed_strategy_accepted(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=20)
        query = _random_multivec(1)[0]
        resp = c.post(
            "/collections/test_shard/search",
            json={"vectors": query, "top_k": 3, "strategy": "shard_routed"},
        )
        assert resp.status_code == 200


# ------------------------------------------------------------------
# HTTP observability endpoints (FIX 36)
# ------------------------------------------------------------------


def test_health_endpoint(tmp_path):
    with _client(tmp_path) as c:
        resp = c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "collections" in body
        assert "gpu_available" in body


def test_ready_endpoint(tmp_path):
    with _client(tmp_path) as c:
        resp = c.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body


def test_metrics_endpoint(tmp_path):
    with _client(tmp_path) as c:
        resp = c.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "voyager_search_requests_total" in text
        assert "voyager_collections_total" in text
        assert "voyager_points_total" in text


def test_scroll_endpoint(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=30)
        resp = c.post(
            "/collections/test_shard/scroll",
            json={"limit": 10, "offset": 0},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "ids" in body
        assert len(body["ids"]) <= 10


def test_retrieve_endpoint(tmp_path):
    with _client(tmp_path) as c:
        _create_shard_collection(c)
        _add_points(c, "test_shard", n=10)
        scroll_resp = c.post(
            "/collections/test_shard/scroll",
            json={"limit": 5, "offset": 0},
        )
        ids = scroll_resp.json()["ids"]
        resp = c.post(
            "/collections/test_shard/retrieve",
            json={"ids": ids, "with_payload": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "points" in body
        assert len(body["points"]) == len(ids)
