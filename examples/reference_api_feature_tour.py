"""
Advanced feature tour for the colsearch reference API.

Run the reference server first:

    colsearch-server

Then execute:

    python examples/reference_api_feature_tour.py --output-json feature-tour-report.json
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

BASE_URL = os.environ.get("VOYAGER_BASE_URL", "http://127.0.0.1:8080")


class SupportsHttpMethods(Protocol):
    def request(self, method: str, path: str, **kwargs: Any): ...
    def get(self, path: str, **kwargs: Any): ...
    def post(self, path: str, **kwargs: Any): ...
    def delete(self, path: str, **kwargs: Any): ...


@dataclass
class StepResult:
    index: int
    name: str
    status: str
    elapsed_ms: float
    detail: str
    data: dict[str, Any] | None = None


def _response_payload(response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"text": response.text}


def _extract_error_detail(response) -> str:
    payload = _response_payload(response)
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail is not None:
            return str(detail)
    return str(payload)


def _request_json(
    client: SupportsHttpMethods,
    method: str,
    path: str,
    *,
    expected_statuses: set[int] | None = None,
    json_body: dict[str, Any] | None = None,
):
    request_kwargs = {"json": json_body} if json_body is not None else {}
    response = client.request(method.upper(), path, **request_kwargs)
    if expected_statuses is None:
        expected_statuses = {200}
    if response.status_code not in expected_statuses:
        raise RuntimeError(f"{method} {path} failed with {response.status_code}: {_extract_error_detail(response)}")
    return response


def _delete_if_present(client: SupportsHttpMethods, name: str) -> None:
    response = client.delete(f"/collections/{name}")
    if response.status_code not in {200, 404}:
        raise RuntimeError(f"DELETE /collections/{name} failed with {response.status_code}: {_extract_error_detail(response)}")


def _search_summary(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    top = results[0] if results else {}
    return {
        "total": payload.get("total"),
        "time_ms": payload.get("time_ms"),
        "top_id": top.get("id"),
        "top_score": top.get("score"),
        "objective_score": payload.get("objective_score"),
        "total_tokens": payload.get("total_tokens"),
    }


def _optimizer_vector_payload(vectors: Any) -> dict[str, Any]:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return {
        "encoding": "float32",
        "shape": list(array.shape),
        "dtype": "float32",
        "data_b64": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii"),
    }


def _configure_logger(log_level: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("colsearch.feature_tour")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_feature_tour(
    client: SupportsHttpMethods,
    *,
    prefix: str = "feature-tour",
    base_url: str = BASE_URL,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or _configure_logger("INFO")
    total_steps = 23
    step_index = 0
    steps: list[StepResult] = []

    dense = f"{prefix}-dense"
    li = f"{prefix}-li"
    mm = f"{prefix}-mm"

    report: dict[str, Any] = {
        "base_url": base_url,
        "prefix": prefix,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "collections": {"dense": dense, "late_interaction": li, "multimodal": mm},
        "searches": {},
        "checks": {},
        "steps": [],
    }

    def run_step(name: str, detail: str, fn, *, optional_if=None) -> Any:
        nonlocal step_index
        step_index += 1
        logger.info("[%02d/%02d] %s", step_index, total_steps, detail)
        start = time.perf_counter()
        try:
            data = fn()
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            message = str(exc)
            should_skip = bool(optional_if(exc)) if callable(optional_if) else False
            status = "skipped" if should_skip else "failed"
            logger.warning("[%02d/%02d] %s: %s", step_index, total_steps, status, message)
            result = StepResult(
                index=step_index,
                name=name,
                status=status,
                elapsed_ms=elapsed_ms,
                detail=message,
            )
            steps.append(result)
            if should_skip:
                return None
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info("[%02d/%02d] passed in %.1f ms", step_index, total_steps, elapsed_ms)
        result = StepResult(
            index=step_index,
            name=name,
            status="passed",
            elapsed_ms=elapsed_ms,
            detail=detail,
            data=data if isinstance(data, dict) else None,
        )
        steps.append(result)
        return data

    try:
        run_step(
            "cleanup_existing_collections",
            "Cleaning up previous feature-tour collections",
            lambda: [_delete_if_present(client, collection) for collection in (dense, li, mm)],
        )

        report["checks"]["system"] = run_step(
            "system_endpoints",
            "Checking /health, /ready, and /metrics",
            lambda: {
                "health": _request_json(client, "GET", "/health").json(),
                "ready": _request_json(client, "GET", "/ready").json(),
                "metrics_has_request_counter": "voyager_search_requests_total"
                in _request_json(client, "GET", "/metrics").text,
            },
        )

        run_step(
            "create_dense_collection",
            "Creating dense collection",
            lambda: _request_json(
                client,
                "POST",
                f"/collections/{dense}",
                json_body={"dimension": 4, "kind": "dense"},
            ).json(),
        )

        run_step(
            "add_dense_points",
            "Adding dense points for vector, BM25, and hybrid search",
            lambda: _request_json(
                client,
                "POST",
                f"/collections/{dense}/points",
                json_body={
                    "points": [
                        {
                            "id": "doc-1",
                            "vector": [1, 0, 0, 0],
                            "payload": {"text": "invoice total due", "doc_type": "invoice", "tenant": "acme", "token_count": 64},
                        },
                        {
                            "id": "doc-2",
                            "vector": [0.85, 0.15, 0, 0],
                            "payload": {"text": "invoice backup receipt", "doc_type": "invoice", "tenant": "acme", "token_count": 48},
                        },
                        {
                            "id": "doc-3",
                            "vector": [0, 1, 0, 0],
                            "payload": {"text": "board report summary", "doc_type": "report", "tenant": "beta", "token_count": 90},
                        },
                    ]
                },
            ).json(),
        )

        report["searches"]["dense_vector"] = run_step(
            "dense_vector_search",
            "Running dense vector-only search",
            lambda: _search_summary(
                _request_json(
                    client,
                    "POST",
                    f"/collections/{dense}/search",
                    json_body={"vector": [1, 0, 0, 0], "top_k": 3},
                ).json()
            ),
        )

        report["searches"]["dense_bm25"] = run_step(
            "dense_bm25_search",
            "Running BM25-only search via query_text",
            lambda: _search_summary(
                _request_json(
                    client,
                    "POST",
                    f"/collections/{dense}/search",
                    json_body={"query_text": "invoice", "top_k": 3},
                ).json()
            ),
        )

        report["searches"]["dense_hybrid_filtered"] = run_step(
            "dense_hybrid_filtered_search",
            "Running dense + BM25 hybrid search with payload filters",
            lambda: _search_summary(
                _request_json(
                    client,
                    "POST",
                    f"/collections/{dense}/search",
                    json_body={
                        "vector": [1, 0, 0, 0],
                        "query_text": "invoice",
                        "filter": {"tenant": "acme"},
                        "with_vector": True,
                        "top_k": 2,
                    },
                ).json()
            ),
        )

        run_step(
            "delete_dense_point",
            "Deleting one dense point to demonstrate point-level CRUD",
            lambda: _request_json(
                client,
                "DELETE",
                f"/collections/{dense}/points",
                json_body={"ids": ["doc-2"]},
            ).json(),
        )

        report["checks"]["dense_info_after_delete"] = run_step(
            "dense_collection_info",
            "Inspecting dense collection info after deletion",
            lambda: _request_json(client, "GET", f"/collections/{dense}/info").json(),
        )

        report["searches"]["dense_optimized"] = run_step(
            "dense_optimized_search",
            "Trying optional optimized dense search with latence_solver",
            lambda: _search_summary(
                _request_json(
                    client,
                    "POST",
                    f"/collections/{dense}/search",
                    json_body={
                        "vector": [1, 0, 0, 0],
                        "query_text": "invoice",
                        "strategy": "optimized",
                        "max_tokens": 160,
                        "max_chunks": 2,
                        "top_k": 2,
                    },
                ).json()
            ),
            optional_if=lambda exc: "latence_solver native package" in str(exc),
        )

        run_step(
            "create_late_interaction_collection",
            "Creating late-interaction collection",
            lambda: _request_json(
                client,
                "POST",
                f"/collections/{li}",
                json_body={"dimension": 4, "kind": "late_interaction", "storage_mode": "sync"},
            ).json(),
        )

        run_step(
            "add_late_interaction_points",
            "Adding late-interaction multivector points",
            lambda: _request_json(
                client,
                "POST",
                f"/collections/{li}/points",
                json_body={
                    "points": [
                        {
                            "id": "li-1",
                            "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                            "payload": {"text": "invoice total due", "label": "invoice"},
                        },
                        {
                            "id": "li-2",
                            "vectors": [[0, 1, 0, 0], [0, 1, 0, 0]],
                            "payload": {"text": "meeting notes", "label": "meeting"},
                        },
                    ]
                },
            ).json(),
        )

        report["searches"]["late_interaction"] = run_step(
            "late_interaction_search",
            "Running late-interaction search with filters and stored vectors",
            lambda: _search_summary(
                _request_json(
                    client,
                    "POST",
                    f"/collections/{li}/search",
                    json_body={
                        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                        "filter": {"label": "invoice"},
                        "with_vector": True,
                        "top_k": 2,
                    },
                ).json()
            ),
        )

        report["checks"]["late_interaction_info"] = run_step(
            "late_interaction_info",
            "Inspecting late-interaction collection info",
            lambda: _request_json(client, "GET", f"/collections/{li}/info").json(),
        )

        run_step(
            "create_multimodal_collection",
            "Creating multimodal collection",
            lambda: _request_json(
                client,
                "POST",
                f"/collections/{mm}",
                json_body={"dimension": 4, "kind": "multimodal"},
            ).json(),
        )

        run_step(
            "add_multimodal_points",
            "Adding multimodal patch embeddings",
            lambda: _request_json(
                client,
                "POST",
                f"/collections/{mm}/points",
                json_body={
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
            ).json(),
        )

        report["searches"]["multimodal_exact"] = run_step(
            "multimodal_exact_search",
            "Running multimodal exact search",
            lambda: _search_summary(
                _request_json(
                    client,
                    "POST",
                    f"/collections/{mm}/search",
                    json_body={
                        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                        "filter": {"kind": "invoice"},
                        "with_vector": True,
                        "top_k": 2,
                    },
                ).json()
            ),
        )

        report["searches"]["multimodal_optimized"] = run_step(
            "multimodal_optimized_search",
            "Running multimodal optimized search to exercise screening",
            lambda: _search_summary(
                _request_json(
                    client,
                    "POST",
                    f"/collections/{mm}/search",
                    json_body={
                        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
                        "strategy": "optimized",
                        "top_k": 2,
                    },
                ).json()
            ),
        )

        report["checks"]["multimodal_info"] = run_step(
            "multimodal_info",
            "Inspecting multimodal collection info",
            lambda: _request_json(client, "GET", f"/collections/{mm}/info").json(),
        )

        report["checks"]["collections_list"] = run_step(
            "list_collections",
            "Listing all collections created by the feature tour",
            lambda: _request_json(client, "GET", "/collections").json(),
        )

        report["checks"]["reference_optimize_health"] = run_step(
            "reference_optimize_health",
            "Inspecting the shared stateless optimizer health endpoint",
            lambda: _request_json(client, "GET", "/reference/optimize/health").json(),
        )

        report["searches"]["reference_optimize_dense"] = run_step(
            "reference_optimize_dense",
            "Calling the public stateless optimize endpoint with dense candidates",
            lambda: _request_json(
                client,
                "POST",
                "/reference/optimize",
                json_body={
                    "query_text": "invoice total due",
                    "query_vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                    "candidates": [
                        {
                            "chunk_id": "invoice-opt",
                            "text": "invoice total due",
                            "token_count": 64,
                            "vectors": _optimizer_vector_payload([1.0, 0.0, 0.0, 0.0]),
                            "metadata": {"dense_score": 1.0, "sparse_score": 2.5, "rrf_score": 0.03},
                        },
                        {
                            "chunk_id": "report-opt",
                            "text": "board report summary",
                            "token_count": 90,
                            "vectors": _optimizer_vector_payload([0.0, 1.0, 0.0, 0.0]),
                            "metadata": {"dense_score": 0.1, "sparse_score": 0.2, "rrf_score": 0.01},
                        },
                    ],
                    "constraints": {"max_tokens": 96, "max_chunks": 1, "max_per_cluster": 1},
                    "solver_config": {"iterations": 16},
                },
            ).json(),
            optional_if=lambda exc: "latence_solver" in str(exc),
        )

    finally:
        run_step(
            "cleanup_feature_tour",
            "Deleting feature-tour collections",
            lambda: [_delete_if_present(client, collection) for collection in (dense, li, mm)],
            optional_if=lambda exc: True,
        )

    report["steps"] = [
        {
            "index": step.index,
            "name": step.name,
            "status": step.status,
            "elapsed_ms": round(step.elapsed_ms, 3),
            "detail": step.detail,
            "data": step.data,
        }
        for step in steps
    ]
    passed = sum(1 for step in steps if step.status == "passed")
    skipped = sum(1 for step in steps if step.status == "skipped")
    failed = sum(1 for step in steps if step.status == "failed")
    report["summary"] = {
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "status": "passed" if failed == 0 and skipped == 0 else "passed_with_skips" if failed == 0 else "failed",
    }
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the colsearch reference API feature tour.")
    parser.add_argument("--base-url", default=BASE_URL, help="Reference API base URL")
    parser.add_argument("--prefix", default="feature-tour", help="Collection prefix for the demo run")
    parser.add_argument("--output-json", default=None, help="Optional path to write the JSON report")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logger = _configure_logger(args.log_level, args.log_file)

    import httpx

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        report = run_feature_tour(
            client,
            prefix=args.prefix,
            base_url=args.base_url,
            logger=logger,
        )

    if args.output_json:
        path = write_report(report, args.output_json)
        logger.info("Wrote feature tour report to %s", path)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
