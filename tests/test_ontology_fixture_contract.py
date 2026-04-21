from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from colsearch._internal.server.main import create_app


FALLBACK_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "dataset_di_fixture.json"
ONTOLOGY_FIXTURE_NAME = "dataset_di_825cbaae40335bc4265a3726.json"


def _fixture_path() -> Path:
    configured = os.environ.get("VOYAGER_ONTOLOGY_FIXTURE")
    if configured:
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "tmp_data" / ONTOLOGY_FIXTURE_NAME,
        repo_root.parent / "tmp_data" / ONTOLOGY_FIXTURE_NAME,
        Path.cwd() / "tmp_data" / ONTOLOGY_FIXTURE_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return FALLBACK_FIXTURE_PATH


def _create_client(index_path: Path) -> TestClient:
    return TestClient(create_app(index_path=str(index_path)))


def test_dataset_intelligence_fixture_roundtrips_as_searchable_payload(tmp_path: Path) -> None:
    fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
    entities = fixture["data"]["entities"][:3]
    points = []
    for idx, entity in enumerate(entities):
        points.append(
            {
                "id": entity["entity_id"],
                "vector": [1.0 if i == idx else 0.0 for i in range(4)],
                "payload": {
                    "entity_id": entity["entity_id"],
                    "label": entity["label"],
                    "canonical_name": entity["canonical_name"],
                    "source": entity["source"],
                    "properties": entity["properties"],
                    "text": f'{entity["canonical_name"]} {entity["label"]} ontology entity',
                },
            }
        )

    with _create_client(tmp_path) as client:
        assert client.post(
            "/collections/ontology",
            json={"dimension": 4, "kind": "dense"},
        ).status_code == 200
        assert client.post(
            "/collections/ontology/points",
            json={"points": points},
        ).status_code == 200

        response = client.post(
            "/collections/ontology/search",
            json={
                "query_text": entities[0]["canonical_name"],
                "filter": {"entity_id": entities[0]["entity_id"]},
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["id"] == entities[0]["entity_id"]
    assert result["payload"]["entity_id"] == entities[0]["entity_id"]
    assert result["payload"]["properties"] == entities[0]["properties"]
