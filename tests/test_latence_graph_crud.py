from __future__ import annotations

from pathlib import Path

from voyager_index._internal.inference.index_core.graph_contract import GraphContractClass
from voyager_index._internal.server.api.models import CollectionKind, CreateCollectionRequest, PointVector, SearchRequest
from voyager_index._internal.server.api.service import SearchService


def _dense_points() -> list[PointVector]:
    return [
        PointVector(
            id="doc-1",
            vector=[1.0, 0.0],
            payload={
                "text": "Team Alpha owns Service B.",
                "token_count": 64,
                "ontology_terms": ["Team Alpha", "Service B"],
                "relations": [{"source": "Team Alpha", "relation": "owns", "target": "Service B", "confidence": 0.96}],
                "concepts": ["ownership"],
            },
        ),
        PointVector(
            id="doc-2",
            vector=[0.0, 1.0],
            payload={
                "text": "Service B causes Service C outage.",
                "token_count": 64,
                "ontology_terms": ["Service B", "Service C"],
                "relations": [{"source": "Service B", "relation": "causes", "target": "Service C", "confidence": 0.98}],
                "concepts": ["incident"],
            },
        ),
    ]


def _graph_sidecar(service: SearchService, name: str = "dense"):
    runtime = service.get_collection(name)
    return runtime.graph_sidecar or getattr(runtime.engine, "graph_sidecar", None)


def _multivector_points() -> list[PointVector]:
    return [
        PointVector(
            id="doc-1",
            vectors=[[1.0, 0.0], [1.0, 0.0]],
            payload={
                "text": "Team Alpha owns Service B.",
                "token_count": 64,
                "ontology_terms": ["Team Alpha", "Service B"],
                "relations": [{"source": "Team Alpha", "relation": "owns", "target": "Service B", "confidence": 0.96}],
                "concepts": ["ownership"],
            },
        ),
        PointVector(
            id="doc-2",
            vectors=[[0.0, 1.0], [0.0, 1.0]],
            payload={
                "text": "Service B causes Service C outage.",
                "token_count": 64,
                "ontology_terms": ["Service B", "Service C"],
                "relations": [{"source": "Service B", "relation": "causes", "target": "Service C", "confidence": 0.98}],
                "concepts": ["incident"],
            },
        ),
    ]


def test_dense_graph_state_persists_and_searches_across_restart(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("dense", CreateCollectionRequest(dimension=2, kind=CollectionKind.DENSE))
    assert service.add_points("dense", _dense_points()) == 2

    sidecar = _graph_sidecar(service)
    assert sidecar.get_statistics()["health"] == "healthy"
    assert sidecar.get_statistics()["num_targets"] == 2
    info = service.collection_info("dense")
    assert info.graph_health == "healthy"
    assert info.graph_contract_version == "1"
    assert info.graph_sync_status is not None

    first = service.search(
        "dense",
        SearchRequest(
            vector=[1.0, 0.0],
            top_k=2,
            graph_mode="force",
            graph_local_budget=2,
            query_payload={"ontology_terms": ["Team Alpha", "Service C"]},
        ),
    )
    assert [item.id for item in first.results] == ["doc-1", "doc-2"]
    service.close()

    reloaded = SearchService(str(tmp_path))
    second = reloaded.search(
        "dense",
        SearchRequest(
            vector=[1.0, 0.0],
            top_k=2,
            graph_mode="force",
            graph_local_budget=2,
            query_payload={"ontology_terms": ["Team Alpha", "Service C"]},
        ),
    )
    reloaded_sidecar = _graph_sidecar(reloaded)
    reloaded_info = reloaded.collection_info("dense")
    assert [item.id for item in second.results] == ["doc-1", "doc-2"]
    assert reloaded_sidecar.get_statistics()["num_targets"] == 2
    assert reloaded_info.graph_health == "healthy"
    assert reloaded_info.graph_contract_version == "1"


def test_shard_graph_search_recovers_related_evidence(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection(
        "shard",
        CreateCollectionRequest(
            dimension=2,
            kind=CollectionKind.SHARD,
            n_shards=8,
            k_candidates=32,
        ),
    )
    points = [
        PointVector(
            id="doc-1",
            vectors=[[1.0, 0.0], [1.0, 0.0]],
            payload={
                "text": "Team Alpha owns Service B.",
                "ontology_terms": ["Team Alpha", "Service B"],
                "relations": [{"source": "Team Alpha", "relation": "owns", "target": "Service B", "confidence": 0.96}],
            },
        ),
        PointVector(
            id="doc-2",
            vectors=[[0.0, 1.0], [0.0, 1.0]],
            payload={
                "text": "Service B causes Service C outage.",
                "ontology_terms": ["Service B", "Service C"],
                "relations": [{"source": "Service B", "relation": "causes", "target": "Service C", "confidence": 0.98}],
            },
        ),
    ]
    assert service.add_points("shard", points) == 2

    response = service.search(
        "shard",
        SearchRequest(
            vectors=[[1.0, 0.0]],
            top_k=2,
            graph_mode="force",
            graph_local_budget=2,
            query_payload={"ontology_terms": ["Team Alpha", "Service C"]},
        ),
    )

    assert [item.id for item in response.results] == ["doc-1", "doc-2"]
    assert response.metadata["graph"]["graph_applied"] is True


def test_late_interaction_graph_search_preserves_base_order_and_adds_rescues(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("li", CreateCollectionRequest(dimension=2, kind=CollectionKind.LATE_INTERACTION))
    assert service.add_points("li", _multivector_points()) == 2

    baseline = service.search(
        "li",
        SearchRequest(
            vectors=[[1.0, 0.0]],
            top_k=1,
            graph_mode="off",
        ),
    )
    response = service.search(
        "li",
        SearchRequest(
            vectors=[[1.0, 0.0]],
            top_k=1,
            graph_mode="force",
            graph_local_budget=2,
            query_payload={"ontology_terms": ["Team Alpha", "Service C"]},
            graph_explain=True,
        ),
    )

    assert [item.id for item in baseline.results] == ["doc-1"]
    assert [item.id for item in response.results] == ["doc-1", "doc-2"]
    assert response.metadata["graph"]["graph_applied"] is True
    assert response.metadata["graph"]["summary"]["merge_mode"] == "additive"
    assert response.metadata["graph"]["summary"]["base_order_preserved"] is True
    assert "graph_local" in response.metadata["graph"]["provenance"]["doc-2"]["lanes"]


def test_multimodal_graph_search_preserves_base_order_and_adds_rescues(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("mm", CreateCollectionRequest(dimension=2, kind=CollectionKind.MULTIMODAL))
    assert service.add_points("mm", _multivector_points()) == 2

    baseline = service.search(
        "mm",
        SearchRequest(
            vectors=[[1.0, 0.0]],
            top_k=1,
            graph_mode="off",
        ),
    )
    response = service.search(
        "mm",
        SearchRequest(
            vectors=[[1.0, 0.0]],
            top_k=1,
            graph_mode="force",
            graph_local_budget=2,
            query_payload={"ontology_terms": ["Team Alpha", "Service C"]},
            graph_explain=True,
        ),
    )

    assert [item.id for item in baseline.results] == ["doc-1"]
    assert [item.id for item in response.results] == ["doc-1", "doc-2"]
    assert response.metadata["graph"]["graph_applied"] is True
    assert response.metadata["graph"]["summary"]["merge_mode"] == "additive"
    assert response.metadata["graph"]["summary"]["base_order_preserved"] is True
    assert "graph_local" in response.metadata["graph"]["provenance"]["doc-2"]["lanes"]


def test_dense_graph_upsert_replaces_old_node_mapping(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("dense", CreateCollectionRequest(dimension=2, kind=CollectionKind.DENSE))
    assert service.add_points("dense", [_dense_points()[0]]) == 1

    sidecar = _graph_sidecar(service)
    old_nodes = set(sidecar.target_to_nodes["doc-1"])
    team_alpha_node = GraphContractClass._node_id("Team Alpha", prefix="entity")
    assert team_alpha_node in old_nodes

    replacement = PointVector(
        id="doc-1",
        vector=[1.0, 0.0],
        payload={
            "text": "Team Beta owns Service D.",
            "token_count": 64,
            "ontology_terms": ["Team Beta", "Service D"],
            "relations": [{"source": "Team Beta", "relation": "owns", "target": "Service D", "confidence": 0.96}],
            "concepts": ["ownership"],
        },
    )
    assert service.add_points("dense", [replacement]) == 1

    updated = _graph_sidecar(service)
    assert team_alpha_node not in updated.target_to_nodes["doc-1"]
    assert "doc-1" not in updated.node_to_targets.get(team_alpha_node, set())


def test_dense_graph_delete_removes_targets_and_survives_restart(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("dense", CreateCollectionRequest(dimension=2, kind=CollectionKind.DENSE))
    assert service.add_points("dense", _dense_points()) == 2
    assert service.delete_points("dense", ["doc-2"]) == 1

    sidecar = _graph_sidecar(service)
    assert "doc-2" not in sidecar.targets
    assert "doc-2" not in sidecar.target_contracts
    assert all("doc-2" not in values for values in sidecar.node_to_targets.values())
    service.close()

    reloaded = SearchService(str(tmp_path))
    reloaded_sidecar = _graph_sidecar(reloaded)
    assert "doc-2" not in reloaded_sidecar.targets
    response = reloaded.search("dense", SearchRequest(vector=[1.0, 0.0], top_k=2))
    assert [item.id for item in response.results] == ["doc-1"]


def test_append_dataset_delta_updates_only_targeted_graph_state(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("dense", CreateCollectionRequest(dimension=2, kind=CollectionKind.DENSE))
    assert service.add_points("dense", [_dense_points()[0]]) == 1

    sidecar = _graph_sidecar(service)
    delta = {
        "bundle_version": "1",
        "target_kind": "document",
        "targets": [
            {
                "target_id": "doc-2",
                "entities": ["Service B", "Service C"],
                "relations": [{"source": "Service B", "relation": "causes", "target": "Service C", "confidence": 0.98}],
                "concepts": ["incident"],
                "metadata": {"internal_id": 2, "external_id": "doc-2"},
            }
        ],
    }
    result = sidecar.append_dataset_delta(delta, dataset_id="ds_demo")
    info = service.collection_info("dense")

    assert result["changed_targets"] == ["doc-2"]
    assert sidecar.dataset_id == "ds_demo"
    assert sidecar.get_statistics()["sync_status"] == "ok"
    assert sidecar.get_statistics()["last_successful_sync_at"] is not None
    assert "doc-1" in sidecar.targets
    assert "doc-2" in sidecar.targets
    assert info.graph_dataset_id == "ds_demo"
    assert info.graph_sync_status == "ok"
    assert info.graph_last_sync_at is not None
    assert info.graph_last_successful_sync_at is not None
    service.close()

    reloaded = SearchService(str(tmp_path))
    reloaded_sidecar = _graph_sidecar(reloaded)
    reloaded_info = reloaded.collection_info("dense")
    assert reloaded_sidecar.dataset_id == "ds_demo"
    assert "doc-1" in reloaded_sidecar.targets
    assert "doc-2" in reloaded_sidecar.targets
    assert reloaded_info.graph_dataset_id == "ds_demo"
    assert reloaded_info.graph_sync_status == "ok"


def test_degraded_graph_state_falls_back_without_breaking_dense_search(tmp_path: Path) -> None:
    service = SearchService(str(tmp_path))
    service.create_collection("dense", CreateCollectionRequest(dimension=2, kind=CollectionKind.DENSE))
    assert service.add_points("dense", _dense_points()) == 2
    sidecar_path = _graph_sidecar(service).state_path
    service.close()
    sidecar_path.write_text("{not-valid-json", encoding="utf-8")

    degraded = SearchService(str(tmp_path))
    report = degraded.readiness_report()
    info = degraded.collection_info("dense")
    response = degraded.search(
        "dense",
        SearchRequest(
            vector=[1.0, 0.0],
            top_k=1,
            graph_mode="force",
            graph_local_budget=2,
            query_payload={"ontology_terms": ["Team Alpha", "Service C"]},
        ),
    )

    assert any(issue["reason"] == "latence_graph_degraded" for issue in report["issues"])
    assert info.graph_health == "degraded"
    assert [item.id for item in response.results] == ["doc-1"]
    assert response.metadata["graph"]["policy"]["reason"] == "graph_unavailable"
