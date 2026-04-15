from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np

from voyager_index import __version__ as package_version
from voyager_index import SearchPipeline
from voyager_index.server import create_app


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Required module '{name}' is not importable in the packaged install environment")


def _build_pipeline(root: Path) -> SearchPipeline:
    pipeline = SearchPipeline(str(root / "graph-pipeline"), dim=2, use_roq=False, on_disk=False)
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


def main() -> None:
    for module_name in ("latence_shard_engine", "latence_solver", "latence"):
        _require_module(module_name)

    tmp_root = Path(tempfile.mkdtemp(prefix="voyager-packaged-smoke-"))

    with TestClient(create_app(index_path=str(tmp_root / "api"), version=package_version)) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        if health.status_code != 200:
            raise RuntimeError(f"/health failed: {health.status_code} {health.text}")
        if ready.status_code != 200:
            raise RuntimeError(f"/ready failed: {ready.status_code} {ready.text}")

    pipeline = _build_pipeline(tmp_root)
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

    result_ids = list(graph_enabled["selected_ids"])
    if baseline["selected_ids"] != [1]:
        raise RuntimeError(f"Unexpected baseline graph-off result: {baseline['selected_ids']}")
    if result_ids[:2] != [1, 2]:
        raise RuntimeError(f"Unexpected graph-enabled result order: {result_ids}")
    graph_summary = graph_enabled["retrieval"]["graph_summary"]
    if graph_summary["merge_mode"] != "additive":
        raise RuntimeError(f"Expected additive graph merge, got {graph_summary['merge_mode']}")
    if graph_summary["added_candidate_count"] < 1:
        raise RuntimeError("Expected graph-aware search to add at least one candidate")

    print(
        json.dumps(
            {
                "voyager_index_version": package_version,
                "result_ids": result_ids,
                "graph_merge_mode": graph_summary["merge_mode"],
                "added_candidate_count": graph_summary["added_candidate_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
