"""
End-to-end happy-path reference API example for colsearch.

Run the reference server first:

    colsearch-server
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


BASE_URL = os.environ.get("VOYAGER_BASE_URL", "http://127.0.0.1:8080")


class SupportsHttpMethods(Protocol):
    def get(self, path: str, **kwargs: Any): ...
    def post(self, path: str, **kwargs: Any): ...
    def delete(self, path: str, **kwargs: Any): ...


def _raise_for_status(response) -> None:
    response.raise_for_status()


def _delete_if_present(client: SupportsHttpMethods, name: str) -> None:
    response = client.delete(f"/collections/{name}")
    if response.status_code not in {200, 404}:
        _raise_for_status(response)


def run_happy_path(client: SupportsHttpMethods, *, prefix: str = "happy-path") -> dict[str, Any]:
    dense = f"{prefix}-dense"
    li = f"{prefix}-li"
    mm = f"{prefix}-mm"

    for collection in (dense, li, mm):
        _delete_if_present(client, collection)

    _raise_for_status(client.post(f"/collections/{dense}", json={"dimension": 4, "kind": "dense"}))
    _raise_for_status(
        client.post(
            f"/collections/{dense}/points",
            json={
                "points": [
                    {
                        "id": "invoice",
                        "vector": [1, 0, 0, 0],
                        "payload": {"text": "invoice total due", "doc_type": "invoice", "token_count": 64},
                    },
                    {
                        "id": "report",
                        "vector": [0, 1, 0, 0],
                        "payload": {"text": "board report summary", "doc_type": "report", "token_count": 90},
                    },
                    {
                        "id": "receipt",
                        "vector": [0.8, 0.2, 0, 0],
                        "payload": {"text": "receipt invoice backup", "doc_type": "invoice", "token_count": 40},
                    },
                ]
            },
        )
    )
    dense_response = client.post(
        f"/collections/{dense}/search",
        json={
            "vector": [1, 0, 0, 0],
            "query_text": "invoice",
            "filter": {"doc_type": "invoice"},
            "top_k": 2,
            "with_payload": True,
        },
    )
    _raise_for_status(dense_response)

    _raise_for_status(client.post(f"/collections/{li}", json={"dimension": 4, "kind": "late_interaction"}))
    _raise_for_status(
        client.post(
            f"/collections/{li}/points",
            json={
                "points": [
                    {
                        "id": "li-1",
                        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                        "payload": {"text": "invoice total due", "label": "invoice"},
                    },
                    {
                        "id": "li-2",
                        "vectors": [[0, 1, 0, 0], [0, 1, 0, 0]],
                        "payload": {"text": "meeting minutes", "label": "meeting"},
                    },
                ]
            },
        )
    )
    li_response = client.post(
        f"/collections/{li}/search",
        json={
            "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
            "filter": {"label": "invoice"},
            "with_vector": True,
            "top_k": 2,
        },
    )
    _raise_for_status(li_response)

    _raise_for_status(client.post(f"/collections/{mm}", json={"dimension": 4, "kind": "multimodal"}))
    _raise_for_status(
        client.post(
            f"/collections/{mm}/points",
            json={
                "points": [
                    {
                        "id": "page-1",
                        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                        "payload": {"doc_id": "invoice.pdf", "page_number": 1, "kind": "invoice"},
                    },
                    {
                        "id": "page-2",
                        "vectors": [[0, 1, 0, 0], [0, 1, 0, 0]],
                        "payload": {"doc_id": "report.pdf", "page_number": 2, "kind": "report"},
                    },
                ]
            },
        )
    )
    mm_response = client.post(
        f"/collections/{mm}/search",
        json={
            "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
            "filter": {"kind": "invoice"},
            "with_vector": True,
            "top_k": 2,
        },
    )
    _raise_for_status(mm_response)

    collections_response = client.get("/collections")
    _raise_for_status(collections_response)

    return {
        "collections": collections_response.json()["collections"],
        "dense_search": dense_response.json(),
        "late_interaction_search": li_response.json(),
        "multimodal_search": mm_response.json(),
        "notes": {
            "optimized_dense_search": "Optional when latence_solver is installed locally.",
            "vllm_pooling_provider": "Optional external embedding provider flow; see examples/vllm_pooling_provider.py.",
            "optimize_endpoint": "Use /reference/optimize for the canonical solver API when latence_solver is installed.",
        },
    }


def main() -> None:
    import httpx

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        summary = run_happy_path(client)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
