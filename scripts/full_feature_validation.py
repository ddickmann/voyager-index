from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voyager_index._internal.inference.quantization.rotational import RoQConfig, RotationalQuantizer
from voyager_index._internal.inference.quantization.scalar import int4_dequantize, int4_quantize
from voyager_index._internal.kernels.maxsim import fast_colbert_scores
from voyager_index._internal.kernels.roq import (
    ROQ_TRITON_AVAILABLE,
    roq_maxsim_1bit,
    roq_maxsim_2bit,
    roq_maxsim_4bit,
)
from voyager_index.preprocessing import (
    enumerate_renderable_documents as shared_enumerate_renderable_documents,
    render_documents as shared_render_documents,
)


MODEL_ID = "VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1"
ONTOLOGY_JSON_NAME = "dataset_di_825cbaae40335bc4265a3726.json"
ROLE_TO_ID = {
    "definition": 0,
    "example": 1,
    "evidence": 2,
    "conclusion": 3,
    "risk": 4,
    "constraint": 5,
    "data_table": 6,
    "procedure": 7,
}
NATIVE_PACKAGE_SPECS = {
    "latence_solver": {
        "distribution": "latence-solver",
        "source_dir": REPO_ROOT / "src" / "kernels" / "knapsack_solver",
        "verify_script": None,
    },
}


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_path = repo_root / "benchmarks" / "oss_reference_benchmark.py"
    spec = importlib.util.spec_from_file_location("oss_reference_benchmark", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {benchmark_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def lane(status: str, **payload: Any) -> dict[str, Any]:
    result = {"status": status}
    result.update(payload)
    return result


def probe_native_package(module_name: str, output_dir: Path, bootstrap: bool) -> dict[str, Any]:
    spec = NATIVE_PACKAGE_SPECS[module_name]
    source_dir = Path(spec["source_dir"])
    result: dict[str, Any] = {
        "module": module_name,
        "distribution": spec["distribution"],
        "source_dir": str(source_dir),
        "source_present": source_dir.exists(),
        "cargo_available": shutil.which("cargo") is not None,
        "rustc_available": shutil.which("rustc") is not None,
        "installed_version": safe_version(spec["distribution"]),
    }
    importlib.invalidate_caches()
    if importlib.util.find_spec(module_name) is not None:
        result["status"] = "installed_and_importable"
        return result
    if not source_dir.exists():
        result["status"] = "source_missing"
        result["reason"] = "source_tree_missing"
        return result
    result["status"] = "source_present_but_not_built"
    result["reason"] = "module_not_importable_in_active_environment"
    if not bootstrap:
        return result

    missing_tools = [tool for tool in ("cargo", "rustc") if shutil.which(tool) is None]
    if missing_tools:
        result["status"] = "build_failed"
        result["reason"] = "missing_rust_toolchain"
        result["missing_tools"] = missing_tools
        return result

    log_path = output_dir / f"{module_name}_bootstrap.log"
    command = [sys.executable, "-m", "pip", "install", "--no-deps", str(source_dir)]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        "\n".join(
            [
                f"COMMAND: {' '.join(command)}",
                f"RETURN_CODE: {completed.returncode}",
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    result["bootstrap_command"] = " ".join(command)
    result["bootstrap_log"] = str(log_path)
    result["bootstrap_returncode"] = completed.returncode
    if completed.returncode != 0:
        result["status"] = "build_failed"
        result["reason"] = "pip_install_failed"
        result["stderr_tail"] = completed.stderr.splitlines()[-20:]
        return result

    importlib.invalidate_caches()
    result["installed_version"] = safe_version(spec["distribution"])
    if importlib.util.find_spec(module_name) is None:
        result["status"] = "build_failed"
        result["reason"] = "pip_install_completed_but_module_missing"
        return result

    result["status"] = "installed_and_importable"
    result["reason"] = "bootstrapped_from_source"
    return result


def bootstrap_native_packages(output_dir: Path, bootstrap: bool) -> dict[str, dict[str, Any]]:
    return {
        module_name: probe_native_package(module_name, output_dir, bootstrap=bootstrap)
        for module_name in NATIVE_PACKAGE_SPECS
    }


def _run_logged_script(command: list[str], output_dir: Path, log_name: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    log_path = output_dir / f"{log_name}.log"
    log_path.write_text(
        "\n".join(
            [
                f"COMMAND: {' '.join(command)}",
                f"RETURN_CODE: {completed.returncode}",
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "log_path": str(log_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _extract_last_json_line(text: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in subprocess output")


def _run_hnsw_probe(force_fallback: bool, output_dir: Path, log_name: str) -> dict[str, Any]:
    script = """
import builtins
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import gc

import numpy as np

repo_root = Path(os.environ["VOYAGER_REPO_ROOT"])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if os.environ.get("VOYAGER_FORCE_HNSW_FALLBACK") == "1":
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "latence_hnsw":
            raise ImportError("forced fallback for validation")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocked_import

from voyager_index._internal.inference.index_core.hnsw_manager import HnswSegmentManager

vectors = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
    ],
    dtype=np.float32,
)
payloads = [
    {"bucket": "alpha", "text": "native primary"},
    {"bucket": "beta", "text": "native filtered"},
    {"bucket": "alpha", "text": "fallback secondary"},
    {"bucket": "beta", "text": "fallback tertiary"},
]
ids = [100, 101, 102, 103]

with tempfile.TemporaryDirectory() as tmpdir:
    shard_path = Path(tmpdir) / "segments"
    manager = HnswSegmentManager(shard_path, dim=4, on_disk=True)
    manager.add(vectors, ids=ids, payloads=payloads)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    start = time.perf_counter()
    top = manager.search(query, k=3)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    filtered = manager.search(query, k=2, filters={"bucket": "beta"})
    active_segment_class = type(manager.active_segment).__name__
    active_segment_module = type(manager.active_segment).__module__
    manager.flush()
    del manager
    gc.collect()

    reopened = HnswSegmentManager(shard_path, dim=4, on_disk=True)
    reopened_top = reopened.search(query, k=3)

    summary = {
        "active_segment_class": active_segment_class,
        "active_segment_module": active_segment_module,
        "top_ids": [item[0] for item in top],
        "filtered_top_ids": [item[0] for item in filtered],
        "reopened_top_ids": [item[0] for item in reopened_top],
        "elapsed_ms": elapsed_ms,
        "flushed_before_reopen": True,
    }
    print(json.dumps(summary, sort_keys=True))
"""
    env = os.environ.copy()
    env["VOYAGER_REPO_ROOT"] = str(REPO_ROOT)
    if force_fallback:
        env["VOYAGER_FORCE_HNSW_FALLBACK"] = "1"
    execution = _run_logged_script([sys.executable, "-c", script], output_dir, log_name, env=env)
    if execution["status"] != "passed":
        return execution
    return {
        **execution,
        "summary": _extract_last_json_line(execution["stdout"]),
    }


def _normalise_solver_entities(entities: list[dict[str, Any]], target_size: int = 8) -> list[dict[str, Any]]:
    normalised = list(entities[:target_size])
    while len(normalised) < target_size:
        idx = len(normalised)
        normalised.append(
            {
                "entity_id": f"synthetic-{idx}",
                "canonical_name": f"Synthetic Entity {idx}",
                "label": f"Synthetic Label {idx}",
            }
        )
    return normalised


def _build_solver_fixture(entities: list[dict[str, Any]]) -> dict[str, Any]:
    source_entities = _normalise_solver_entities(entities)
    ids = list(range(len(source_entities)))
    vectors = np.stack(
        [synthetic_vector(f"solver::{entity['entity_id']}", 8) for entity in source_entities],
        axis=0,
    ).astype(np.float32)
    payloads = []
    corpus = []
    for idx, entity in enumerate(source_entities):
        focus_terms = "voyager target alpha priority" if idx == 0 else f"background context {idx}"
        text = f"{entity['canonical_name']} {entity['label']} {focus_terms}"
        corpus.append(text)
        if idx == 0:
            fact_density = 1.0
            centrality = 0.95
            recency = 0.9
            auxiliary = 1.0
        elif idx == 1:
            fact_density = 0.92
            centrality = 0.88
            recency = 0.84
            auxiliary = 0.72
        else:
            fact_density = max(0.05, 0.22 - idx * 0.015)
            centrality = max(0.05, 0.18 - idx * 0.01)
            recency = max(0.05, 0.14 - idx * 0.01)
            auxiliary = max(0.0, 0.08 - idx * 0.01)
        payloads.append(
            {
                "text": text,
                "token_count": 90 + idx * 7,
                "fact_density": fact_density,
                "centrality_score": centrality,
                "recency_score": recency,
                "auxiliary_score": auxiliary,
                "cluster_id": idx // 2,
            }
        )
    return {
        "ids": ids,
        "vectors": vectors,
        "payloads": payloads,
        "corpus": corpus,
        "query_text": "voyager target alpha priority",
        "query_vector": vectors[0],
        "expected_primary_id": ids[0],
        "relevance_map": {
            ids[0]: 3.0,
            ids[1]: 2.0 if len(ids) > 1 else 0.0,
        },
    }


def _solver_overlap(selected_ids: list[str], baseline_ids: list[int]) -> float:
    if not selected_ids or not baseline_ids:
        return 0.0
    selected = set(selected_ids)
    baseline = {str(item_id) for item_id in baseline_ids}
    return len(selected & baseline) / float(min(len(selected), len(baseline)))


def _ranking_metrics(ranked_ids: list[int | str], relevance_map: dict[int, float], k: int = 5) -> dict[str, float]:
    if not ranked_ids:
        return {"mrr": 0.0, "ndcg": 0.0, "recall": 0.0}

    ranked = [int(item_id) for item_id in ranked_ids[:k]]
    positives = {doc_id for doc_id, relevance in relevance_map.items() if relevance > 0.0}
    rr = 0.0
    dcg = 0.0
    hits = 0
    for rank, doc_id in enumerate(ranked, start=1):
        relevance = float(relevance_map.get(doc_id, 0.0))
        if rr == 0.0 and relevance > 0.0:
            rr = 1.0 / float(rank)
        if relevance > 0.0:
            hits += 1
            dcg += ((2.0 ** relevance) - 1.0) / math.log2(rank + 1.0)
    ideal_relevances = sorted((value for value in relevance_map.values() if value > 0.0), reverse=True)[:k]
    idcg = sum(((2.0 ** relevance) - 1.0) / math.log2(rank + 1.0) for rank, relevance in enumerate(ideal_relevances, start=1))
    return {
        "mrr": rr,
        "ndcg": (dcg / idcg) if idcg > 0.0 else 0.0,
        "recall": hits / float(len(positives)) if positives else 0.0,
    }


def _summarise_solver_output(
    status: str,
    refined: dict[str, Any],
    elapsed_ms: float,
    expected_primary_id: int,
    bm25_ids: list[int],
    dense_ids: list[int],
    hybrid_ids: list[int],
    relevance_map: dict[int, float],
) -> dict[str, Any]:
    selected_ids = list(refined.get("selected_ids", []))
    solver_output = refined.get("solver_output", {})
    selected_as_ints = [int(item_id) for item_id in selected_ids]
    return {
        "status": status,
        "selected_ids": selected_ids,
        "selected_primary_hit": str(expected_primary_id) in selected_ids,
        "overlap_with_bm25": _solver_overlap(selected_ids, bm25_ids),
        "overlap_with_dense": _solver_overlap(selected_ids, dense_ids),
        "overlap_with_hybrid_union": _solver_overlap(selected_ids, hybrid_ids),
        "rank_metrics": _ranking_metrics(selected_as_ints, relevance_map),
        "objective_score": solver_output.get("objective_score"),
        "num_selected": solver_output.get("num_selected"),
        "solve_time_ms": solver_output.get("solve_time_ms"),
        "measured_elapsed_ms": elapsed_ms,
        "constraints_satisfied": solver_output.get("constraints_satisfied"),
    }


def resolve_ontology_fixture(tmp_data_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    configured = os.environ.get("VOYAGER_ONTOLOGY_FIXTURE")
    if configured:
        return Path(configured)
    candidate = tmp_data_dir / ONTOLOGY_JSON_NAME
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dataset_di_fixture.json"


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def synthetic_vector(text: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(text))
    vector = rng.normal(size=(dim,)).astype(np.float32)
    norm = np.linalg.norm(vector) + 1e-8
    return vector / norm


def synthetic_multivector(text: str, tokens: int, dim: int) -> np.ndarray:
    base = synthetic_vector(text, dim)
    rng = np.random.default_rng(stable_seed(f"{text}:tokens"))
    matrix = np.stack(
        [
            (base + rng.normal(scale=0.05, size=(dim,)).astype(np.float32))
            for _ in range(tokens)
        ],
        axis=0,
    ).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return matrix / norms


def enumerate_source_documents(tmp_data_dir: Path, ontology_fixture: Path) -> dict[str, Any]:
    inventory = shared_enumerate_renderable_documents(tmp_data_dir, exclude_paths=[ontology_fixture])
    skipped = list(inventory["skipped"])
    for skipped_item in skipped:
        if skipped_item["path"] == str(ontology_fixture.resolve()) and skipped_item["reason"] == "excluded":
            skipped_item["reason"] = "ontology_fixture"
    return {
        "documents": inventory["documents"],
        "skipped": skipped,
    }


def render_documents(documents: list[Path], output_dir: Path) -> dict[str, Any]:
    return shared_render_documents(documents, output_dir)


def load_ontology_entities(fixture_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    entities = list(payload["data"]["entities"])
    if limit is None:
        return entities
    return entities[:limit]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.casefold())).strip()


def _visible_tokens(text: str) -> list[str]:
    return [token for token in _normalize_text(text).split(" ") if token]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _infer_rhetorical_role(text: str) -> str:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("table", "row", "column")):
        return "data_table"
    if any(token in lowered for token in ("step", "procedure", "workflow")):
        return "procedure"
    if any(token in lowered for token in ("risk", "warning", "hazard")):
        return "risk"
    if any(token in lowered for token in ("must", "shall", "constraint")):
        return "constraint"
    if any(token in lowered for token in ("because", "evidence", "study", "data")):
        return "evidence"
    if any(token in lowered for token in ("means", "defined", "definition")):
        return "definition"
    if any(token in lowered for token in ("for example", "example", "e.g.")):
        return "example"
    if any(token in lowered for token in ("therefore", "summary", "conclusion")):
        return "conclusion"
    return "unknown"


def _computed_feature_pack(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    dense_vectors = np.stack([record["dense_vector"] for record in records]).astype(np.float32, copy=False)
    dense_norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    dense_norms = np.clip(dense_norms, 1e-8, None)
    normalized_vectors = dense_vectors / dense_norms
    centroid = normalized_vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 1e-8:
        centroid = centroid / centroid_norm

    max_page_by_doc: dict[str, int] = defaultdict(int)
    cluster_ids_by_doc: dict[str, int] = {}
    for record in records:
        doc_key = str(record.get("document_id") or record.get("source_name") or record["id"])
        max_page_by_doc[doc_key] = max(max_page_by_doc[doc_key], int(record.get("page") or 1))

    feature_pack: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        enriched_text = str(record.get("enriched_text") or record.get("base_text") or "")
        tokens = _visible_tokens(enriched_text)
        unique_ratio = len(set(tokens)) / float(len(tokens)) if tokens else 0.0
        numeric_ratio = (
            sum(any(char.isdigit() for char in token) for token in tokens) / float(len(tokens))
            if tokens
            else 0.0
        )
        ontology_term_count = len(record.get("ontology_terms") or [])
        ontology_concept_density = float(record.get("ontology_concept_density", 0.0) or 0.0)
        ontology_relation_density = float(record.get("ontology_relation_density", 0.0) or 0.0)
        ontology_confidence = float(record.get("ontology_confidence", 0.0) or 0.0)

        centroid_similarity = (1.0 + _cosine_similarity(normalized_vectors[index], centroid)) * 0.5
        max_peer_similarity = max(
            (
                _cosine_similarity(normalized_vectors[index], other)
                for other_index, other in enumerate(normalized_vectors)
                if other_index != index
            ),
            default=0.0,
        )
        uniqueness = _clamp01(1.0 - max(0.0, max_peer_similarity))

        doc_key = str(record.get("document_id") or record.get("source_name") or record["id"])
        if doc_key not in cluster_ids_by_doc:
            cluster_ids_by_doc[doc_key] = len(cluster_ids_by_doc)
        max_page = max(max_page_by_doc.get(doc_key, 1), 1)
        page_number = max(1, int(record.get("page") or 1))
        front_page_bias = 1.0 if max_page <= 1 else 1.0 - ((page_number - 1) / float(max_page - 1))

        feature_pack.append(
            {
                "fact_density": _clamp01(
                    0.50 * unique_ratio
                    + 0.15 * numeric_ratio
                    + 0.20 * ontology_concept_density
                    + 0.15 * ontology_relation_density
                ),
                "centrality_score": _clamp01(
                    0.55 * centroid_similarity
                    + 0.25 * ontology_confidence
                    + 0.20 * ontology_concept_density
                ),
                "recency_score": _clamp01(
                    0.70 * front_page_bias
                    + 0.20 * ontology_confidence
                    + 0.10 * ontology_relation_density
                ),
                "auxiliary_score": _clamp01(
                    0.35 * ontology_confidence
                    + 0.20 * ontology_relation_density
                    + 0.20 * min(1.0, ontology_term_count / 4.0)
                    + 0.25 * uniqueness
                ),
                "cluster_id": int(cluster_ids_by_doc[doc_key]),
                "rhetorical_role": _infer_rhetorical_role(enriched_text),
            }
        )

    return feature_pack


def _safe_property_text(entity: dict[str, Any]) -> str:
    properties = entity.get("properties") or {}
    values = []
    for key in sorted(properties):
        value = properties[key]
        if isinstance(value, (str, int, float, bool)):
            values.append(f"{key}={value}")
        elif isinstance(value, list):
            joined = ", ".join(str(item) for item in value[:4])
            if joined:
                values.append(f"{key}={joined}")
    return "; ".join(values[:6])


def _entity_rank_key(entity: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(entity.get("evidence_count") or 0.0),
        float(entity.get("confidence") or 0.0),
        len(entity.get("document_ids") or []),
        len(str(entity.get("canonical_name") or "")),
    )


def _candidate_ontology_entities(entities: list[dict[str, Any]], limit: int = 512) -> list[dict[str, Any]]:
    filtered = []
    for entity in entities:
        canonical_name = str(entity.get("canonical_name") or "").strip()
        if not canonical_name:
            continue
        has_alpha = bool(re.search(r"[A-Za-zÄÖÜäöüß]", canonical_name))
        if not has_alpha and str(entity.get("label") or "").casefold() in {"date", "number", "ordinal"}:
            continue
        if len(canonical_name) < 4 and not has_alpha:
            continue
        filtered.append(entity)
    filtered.sort(key=_entity_rank_key, reverse=True)
    return filtered[:limit]


def _entity_terms(entity: dict[str, Any]) -> list[str]:
    terms = [str(entity.get("canonical_name") or "").strip()]
    aliases = entity.get("aliases") or []
    for alias in aliases[:4]:
        alias_text = str(alias).strip()
        if alias_text:
            terms.append(alias_text)
    return [term for term in dict.fromkeys(terms) if term]


def _match_entities_for_text(
    normalized_text: str,
    entities: list[dict[str, Any]],
    *,
    max_matches: int = 8,
) -> list[dict[str, Any]]:
    matches = []
    padded = f" {normalized_text} "
    for entity in entities:
        matched_term = None
        matched_length = 0
        for term in _entity_terms(entity):
            normalized_term = _normalize_text(term)
            if len(normalized_term) < 4:
                continue
            if f" {normalized_term} " in padded:
                matched_term = term
                matched_length = max(matched_length, len(normalized_term))
        if matched_term is None:
            continue
        matches.append(
            {
                "entity_id": str(entity.get("entity_id")),
                "canonical_name": str(entity.get("canonical_name") or matched_term),
                "label": str(entity.get("label") or "entity"),
                "confidence": float(entity.get("confidence") or 0.0),
                "evidence_count": int(entity.get("evidence_count") or 0),
                "document_frequency": len(entity.get("document_ids") or []),
                "matched_term": matched_term,
                "property_text": _safe_property_text(entity),
                "_matched_length": matched_length,
            }
        )
    matches.sort(
        key=lambda item: (
            item["_matched_length"],
            item["evidence_count"],
            item["confidence"],
            len(item["canonical_name"]),
        ),
        reverse=True,
    )
    return [{key: value for key, value in item.items() if not key.startswith("_")} for item in matches[:max_matches]]


def _record_id_for_render(source: str, page: int) -> tuple[str, str]:
    document_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return document_id, f"{document_id}-p{page:03d}"


def build_evaluation_corpus(
    rendered_images: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_entities = _candidate_ontology_entities(entities)
    records: list[dict[str, Any]] = []
    match_frequencies: Counter[str] = Counter()
    entity_lookup: dict[str, dict[str, Any]] = {}

    for render in rendered_images:
        source = str(render["source"])
        source_path = Path(source)
        page = int(render.get("page", 1))
        document_id, record_id = _record_id_for_render(source, page)
        base_text = " ".join(
            part
            for part in [
                source_path.stem.replace("_", " "),
                str(render.get("text") or "").strip(),
                f"page {page}",
            ]
            if part
        ).strip()
        normalized_text = _normalize_text(base_text)
        matches = _match_entities_for_text(normalized_text, candidate_entities)
        match_frequencies.update(match["entity_id"] for match in matches)
        for match in matches:
            entity_lookup[match["entity_id"]] = match
        ontology_terms = [match["canonical_name"] for match in matches]
        ontology_labels = [match["label"] for match in matches]
        ontology_confidences = [float(match["confidence"]) for match in matches]
        ontology_evidence_counts = [int(match["evidence_count"]) for match in matches]
        ontology_match_count = len(matches)
        base_token_count = max(1, len(_visible_tokens(base_text)))
        ontology_confidence = float(np.mean(ontology_confidences)) if ontology_confidences else 0.0
        property_match_count = sum(1 for match in matches if match["property_text"])
        ontology_concept_density = min(1.0, ontology_match_count / float(max(1.0, base_token_count / 10.0)))
        ontology_relation_density = min(
            1.0,
            (0.5 * (property_match_count / float(max(ontology_match_count, 1))))
            + (0.5 * (sum(ontology_evidence_counts) / float(max(ontology_match_count * 8, 1)))),
        )
        ontology_text = " ; ".join(
            f"{match['canonical_name']} [{match['label']}]"
            + (f" {match['property_text']}" if match["property_text"] else "")
            for match in matches
        )
        records.append(
            {
                "id": record_id,
                "document_id": document_id,
                "source": source,
                "source_name": source_path.name,
                "page": page,
                "image_path": str(render["image"]),
                "renderer": render.get("renderer"),
                "base_text": base_text or source_path.stem,
                "enriched_text": (
                    f"{base_text} ontology {ontology_text}".strip()
                    if ontology_text
                    else (base_text or source_path.stem)
                ),
                "ontology_matches": matches,
                "ontology_terms": ontology_terms,
                "ontology_labels": ontology_labels,
                "ontology_confidences": ontology_confidences,
                "ontology_evidence_counts": ontology_evidence_counts,
                "ontology_match_count": ontology_match_count,
                "ontology_confidence": ontology_confidence,
                "ontology_concept_density": ontology_concept_density,
                "ontology_relation_density": ontology_relation_density,
            }
        )

    entity_to_record_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for match in record["ontology_matches"]:
            entity_to_record_ids[match["entity_id"]].append(record["id"])

    query_specs = []
    for entity_id, record_ids in entity_to_record_ids.items():
        entity = entity_lookup[entity_id]
        if not record_ids:
            continue
        query_specs.append(
            {
                "query_id": entity_id,
                "text": entity["canonical_name"],
                "label": entity["label"],
                "relevance_map": {record_id: 3.0 for record_id in sorted(set(record_ids))},
                "source": "ontology_entity",
                "match_count": len(set(record_ids)),
                "confidence": entity["confidence"],
            }
        )
    query_specs.sort(
        key=lambda item: (
            item["match_count"],
            item["confidence"],
            len(item["text"]),
        ),
        reverse=True,
    )
    query_specs = query_specs[:8]

    if not query_specs:
        for index, record in enumerate(records[:4], start=1):
            tokens = _visible_tokens(record["base_text"])[:6]
            query_specs.append(
                {
                    "query_id": f"fallback-{index}",
                    "text": " ".join(tokens) or record["source_name"],
                    "label": "fallback",
                    "relevance_map": {record["id"]: 3.0},
                    "source": "rendered_document_fallback",
                    "match_count": 1,
                    "confidence": 1.0,
                }
            )

    return {
        "records": records,
        "queries": query_specs,
        "summary": {
            "record_count": len(records),
            "documents": len({record["document_id"] for record in records}),
            "records_with_ontology_matches": sum(1 for record in records if record["ontology_matches"]),
            "matched_entity_count": len(entity_to_record_ids),
            "query_count": len(query_specs),
            "top_ontology_terms": [
                {
                    "entity_id": entity_id,
                    "canonical_name": entity_lookup[entity_id]["canonical_name"],
                    "record_count": count,
                }
                for entity_id, count in match_frequencies.most_common(8)
                if entity_id in entity_lookup
            ],
        },
    }


def _pad_multivectors(vectors: list[np.ndarray], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not vectors:
        raise ValueError("Expected at least one multivector")
    max_tokens = max(vector.shape[0] for vector in vectors)
    dim = vectors[0].shape[1]
    padded = torch.zeros((len(vectors), max_tokens, dim), dtype=torch.float32, device=device)
    mask = torch.zeros((len(vectors), max_tokens), dtype=torch.float32, device=device)
    for index, vector in enumerate(vectors):
        length = vector.shape[0]
        padded[index, :length] = torch.from_numpy(vector).to(device=device, dtype=torch.float32)
        mask[index, :length] = 1.0
    return padded, mask


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return matrix / norms


def _pool_multivector(vector: np.ndarray) -> np.ndarray:
    pooled = vector.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(pooled) + 1e-8
    return pooled / norm


def _mean_metric(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"mrr": 0.0, "ndcg": 0.0, "recall": 0.0}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _score_correlation(reference: np.ndarray, approx: np.ndarray) -> float:
    if reference.size == 0 or approx.size == 0:
        return 0.0
    ref = reference.reshape(-1)
    app = approx.reshape(-1)
    if np.allclose(ref.std(), 0.0) or np.allclose(app.std(), 0.0):
        return 0.0
    return float(np.corrcoef(ref, app)[0, 1])


def _benchmark_repeated(
    fn,
    *,
    synchronize_device: str | None = None,
    warmup_runs: int = 1,
    measure_runs: int = 5,
):
    last_result = None
    warmup_elapsed: list[float] = []
    for _ in range(max(warmup_runs, 0)):
        if synchronize_device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        last_result = fn()
        if synchronize_device == "cuda":
            torch.cuda.synchronize()
        warmup_elapsed.append((time.perf_counter() - start) * 1000.0)

    measured: list[float] = []
    for _ in range(max(measure_runs, 1)):
        if synchronize_device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        last_result = fn()
        if synchronize_device == "cuda":
            torch.cuda.synchronize()
        measured.append((time.perf_counter() - start) * 1000.0)

    return last_result, {
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "first_warmup_ms": warmup_elapsed[0] if warmup_elapsed else None,
        "median_ms": float(np.median(np.asarray(measured, dtype=np.float64))),
        "min_ms": float(min(measured)),
        "max_ms": float(max(measured)),
        "runs_ms": measured,
    }


def _candidate_sweep_counts(base_count: int, max_count: int = 1536) -> list[int]:
    if base_count <= 0:
        return []
    counts = [base_count]
    while counts[-1] < max_count:
        counts.append(counts[-1] * 2)
    return counts


def _repeat_candidate_tensors(
    docs: torch.Tensor,
    docs_mask: torch.Tensor,
    target_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _repeat_candidate_rows(docs, target_count), _repeat_candidate_rows(docs_mask, target_count)


def _repeat_candidate_rows(values: torch.Tensor, target_count: int) -> torch.Tensor:
    base_count = int(values.shape[0])
    if target_count <= base_count:
        return values[:target_count]
    indices = torch.arange(target_count, device=values.device) % base_count
    return values.index_select(0, indices)


def _load_collfm2_runtime() -> tuple[Any, Any, str]:
    from sauerkrautlm_colpali.models import ColLFM2, ColLFM2Processor  # type: ignore
    from transformers import AutoModelForImageTextToText

    processor = ColLFM2Processor.from_pretrained(MODEL_ID)
    try:
        model = ColLFM2.from_pretrained(MODEL_ID).to("cuda:0").eval()
        return processor, model, "flash_attention_2"
    except Exception as exc:
        if "FlashAttention2" not in str(exc):
            raise
        original_loader = AutoModelForImageTextToText.from_pretrained

        def _patched_loader(*args: Any, **kwargs: Any):
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = "sdpa"
            return original_loader(*args, **kwargs)

        AutoModelForImageTextToText.from_pretrained = _patched_loader
        try:
            model = ColLFM2.from_pretrained(MODEL_ID).to("cuda:0").eval()
        finally:
            AutoModelForImageTextToText.from_pretrained = original_loader
        return processor, model, "sdpa_monkeypatch"


def _embed_real_corpus(
    records: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from PIL import Image

    processor, model, attention_backend = _load_collfm2_runtime()
    batch_size = 2
    document_vectors: list[np.ndarray] = []
    query_vectors: list[np.ndarray] = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        batch_images = []
        for record in batch_records:
            with Image.open(record["image_path"]) as image:
                batch_images.append(image.convert("RGB"))
        batch_inputs = processor.process_images(batch_images).to(model.device)
        attention_mask = batch_inputs.get("attention_mask")
        with torch.no_grad():
            batch_embeddings = model(**batch_inputs).to(torch.float32)
        for index in range(batch_embeddings.shape[0]):
            length = int(attention_mask[index].sum().item()) if attention_mask is not None else int(batch_embeddings.shape[1])
            document_vectors.append(batch_embeddings[index, :length].detach().cpu().numpy())

    query_texts = [query["text"] for query in queries]
    for start in range(0, len(query_texts), batch_size):
        batch_queries = query_texts[start:start + batch_size]
        batch_inputs = processor.process_queries(batch_queries).to(model.device)
        attention_mask = batch_inputs.get("attention_mask")
        with torch.no_grad():
            batch_embeddings = model(**batch_inputs).to(torch.float32)
        for index in range(batch_embeddings.shape[0]):
            length = int(attention_mask[index].sum().item()) if attention_mask is not None else int(batch_embeddings.shape[1])
            query_vectors.append(batch_embeddings[index, :length].detach().cpu().numpy())

    summary = {
        "embedding_source": "real_model",
        "attention_backend": attention_backend,
        "document_count": len(document_vectors),
        "query_count": len(query_vectors),
        "document_token_min": min(vector.shape[0] for vector in document_vectors),
        "document_token_max": max(vector.shape[0] for vector in document_vectors),
        "query_token_min": min(vector.shape[0] for vector in query_vectors),
        "query_token_max": max(vector.shape[0] for vector in query_vectors),
        "embed_dim": int(document_vectors[0].shape[1]) if document_vectors else 0,
    }
    manifest_path = output_dir / "corpus_embedding_summary.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "summary": summary,
        "document_vectors": document_vectors,
        "query_vectors": query_vectors,
        "processor": processor,
        "summary_path": str(manifest_path),
    }


def _embed_synthetic_corpus(records: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
    document_vectors = [
        synthetic_multivector(record["base_text"], tokens=8, dim=32).astype(np.float32)
        for record in records
    ]
    query_vectors = [
        synthetic_multivector(query["text"], tokens=6, dim=32).astype(np.float32)
        for query in queries
    ]
    return {
        "summary": {
            "embedding_source": "synthetic_fallback",
            "document_count": len(document_vectors),
            "query_count": len(query_vectors),
            "embed_dim": 32,
        },
        "document_vectors": document_vectors,
        "query_vectors": query_vectors,
        "processor": None,
        "summary_path": None,
    }


def prepare_evaluation_bundle(
    rendered_images: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    active_renders = list(rendered_images)
    if not active_renders:
        synthetic_image = _synthetic_model_image(output_dir, entities)
        active_renders = [
            {
                "source": str(synthetic_image),
                "image": str(synthetic_image),
                "page": 1,
                "renderer": "synthetic",
                "text": "Synthetic fallback image built from ontology metadata.",
            }
        ]

    corpus = build_evaluation_corpus(active_renders, entities)
    records = corpus["records"]
    queries = corpus["queries"]

    package_installed = importlib.util.find_spec("sauerkrautlm_colpali") is not None
    if package_installed and torch.cuda.is_available():
        embedding_bundle = _embed_real_corpus(records, queries, output_dir)
    else:
        embedding_bundle = _embed_synthetic_corpus(records, queries)

    for record, vectors in zip(records, embedding_bundle["document_vectors"]):
        record["multivector"] = vectors
        record["dense_vector"] = _pool_multivector(vectors)
        record["token_count"] = max(len(_visible_tokens(record["enriched_text"])), int(vectors.shape[0] * 8))
    for query, vectors in zip(queries, embedding_bundle["query_vectors"]):
        query["multivector"] = vectors
        query["dense_vector"] = _pool_multivector(vectors)

    return {
        "records": records,
        "queries": queries,
        "corpus_summary": corpus["summary"],
        "embedding_summary": {
            "package_installed": package_installed,
            "package_version": safe_version("sauerkrautlm-colpali"),
            "model_id": MODEL_ID,
            "requested_device": "cuda:0",
            **embedding_bundle["summary"],
        },
        "processor": embedding_bundle["processor"],
        "summary_path": embedding_bundle["summary_path"],
    }


def run_oss_benchmark_bundle(output_dir: Path) -> dict[str, Any]:
    benchmark_module = _load_benchmark_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        results = {
            "maxsim": benchmark_module.benchmark_maxsim("cpu", queries=2, docs=16, tokens=6, dim=8),
            "reference_api_ingest": benchmark_module.benchmark_reference_api_ingest(root / "ingest", points=8),
            "reference_api_search": benchmark_module.benchmark_reference_api_search(root / "search", top_k=3),
            "multimodal_search": benchmark_module.benchmark_multimodal_search(root / "mm", top_k=3),
        }
    path = output_dir / "benchmark_smoke.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return lane("passed", path=str(path), summary=results)


def _topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(top_k, scores.shape[1])
    order = np.argsort(-scores, axis=1)
    return order[:, :top_k]


def _mean_recall_at_k(expected: np.ndarray, predicted: np.ndarray, top_k: int) -> float:
    if expected.size == 0 or predicted.size == 0:
        return 0.0
    recalls = []
    for expected_row, predicted_row in zip(expected[:, :top_k], predicted[:, :top_k]):
        expected_set = set(int(item) for item in expected_row.tolist())
        predicted_set = set(int(item) for item in predicted_row.tolist())
        recalls.append(len(expected_set & predicted_set) / float(max(len(expected_set), 1)))
    return float(np.mean(recalls))


def _candidate_retention(expected: np.ndarray, candidate_pool: np.ndarray, top_k: int) -> float:
    if expected.size == 0 or candidate_pool.size == 0:
        return 0.0
    retained = []
    for expected_row, candidate_row in zip(expected[:, :top_k], candidate_pool):
        expected_set = set(int(item) for item in expected_row.tolist())
        candidate_set = set(int(item) for item in candidate_row.tolist())
        retained.append(len(expected_set & candidate_set) / float(max(len(expected_set), 1)))
    return float(np.mean(retained))


def _screening_budget_summary(expected_topk: np.ndarray, approx_scores: np.ndarray, budgets: list[int], *, top_k: int) -> list[dict[str, Any]]:
    if approx_scores.size == 0:
        return []
    rows = []
    total_docs = approx_scores.shape[1]
    for budget in budgets:
        effective_budget = min(int(budget), total_docs)
        if effective_budget <= 0:
            continue
        candidate_pool = _topk_indices(approx_scores, top_k=effective_budget)
        rows.append(
            {
                "candidate_count": effective_budget,
                "top1_retention": _candidate_retention(expected_topk[:, :1], candidate_pool, top_k=1),
                "topk_retention": _candidate_retention(expected_topk, candidate_pool, top_k=top_k),
            }
        )
    return rows


def _candidate_retention_from_rows(expected: np.ndarray, candidate_rows: list[list[int]], top_k: int) -> float:
    if expected.size == 0 or not candidate_rows:
        return 0.0
    retained = []
    for expected_row, candidate_row in zip(expected[:, :top_k], candidate_rows):
        expected_set = set(int(item) for item in expected_row.tolist())
        candidate_set = set(int(item) for item in candidate_row)
        retained.append(len(expected_set & candidate_set) / float(max(len(expected_set), 1)))
    return float(np.mean(retained))


def _first_full_recovery_candidate_count(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if row.get("top1_retention", 0.0) >= 1.0 and row.get("topk_retention", 0.0) >= 1.0:
            return int(row["candidate_count"])
    return None


def _evaluate_screened_multimodal_engine(
    *,
    engine: Any,
    queries: list[dict[str, Any]],
    query_vectors: list[np.ndarray],
    doc_ids: list[Any],
    doc_id_to_index: dict[Any, int],
    exact_mv_topk: np.ndarray,
    full_precision_benchmark: dict[str, Any],
    top_k: int,
    candidate_budgets: list[int],
    default_budget: int,
) -> dict[str, Any]:
    padded_rankings = np.full((len(query_vectors), top_k), -1, dtype=np.int32)
    qualitative = []
    rank_rows = []
    candidate_pool_rows: list[dict[str, Any]] = []
    screening_profile_ms = 0.0
    exact_profile_ms = 0.0
    fallback_query_count = 0
    direct_gather_query_count = 0

    for candidate_budget in candidate_budgets:
        candidate_rows: list[list[int]] = []
        for query_vector in query_vectors:
            screened_ids = engine.screen_candidates(
                query_embedding=query_vector,
                top_k=top_k,
                candidate_budget=candidate_budget,
            )
            effective_ids = list(doc_ids) if screened_ids is None else list(screened_ids)
            candidate_rows.append(
                [
                    int(doc_id_to_index[doc_id])
                    for doc_id in effective_ids
                    if doc_id in doc_id_to_index
                ]
            )
        candidate_pool_rows.append(
            {
                "candidate_count": int(candidate_budget),
                "top1_retention": _candidate_retention_from_rows(exact_mv_topk[:, :1], candidate_rows, top_k=1),
                "topk_retention": _candidate_retention_from_rows(exact_mv_topk, candidate_rows, top_k=top_k),
            }
        )

    for query_index, (query, query_vector) in enumerate(zip(queries, query_vectors)):
        screened_ids = engine.screen_candidates(
            query_embedding=query_vector,
            top_k=top_k,
            candidate_budget=default_budget,
        )
        screening_profile_ms += float(getattr(engine, "last_screening_profile", {}).get("elapsed_ms", 0.0))
        if screened_ids is None:
            fallback_query_count += 1
            screened_results = engine.search(
                query_embedding=query_vector,
                top_k=top_k,
            )
            effective_ids = list(doc_ids)
        else:
            screened_results = engine.search(
                query_embedding=query_vector,
                top_k=top_k,
                candidate_ids=screened_ids,
            )
            effective_ids = list(screened_ids)
        search_profile = dict(getattr(engine, "last_search_profile", {}) or {})
        direct_gather_query_count += int(bool(search_profile.get("direct_gather")))
        exact_profile_ms += sum(
            float(search_profile.get(key, 0.0))
            for key in ("load_ms", "materialize_ms", "mask_ms", "score_ms", "heap_ms")
        )

        ranked_ids = [
            int(doc_id_to_index[result.doc_id])
            for result in screened_results
            if result.doc_id in doc_id_to_index
        ]
        padded_rankings[query_index, : min(len(ranked_ids), top_k)] = ranked_ids[:top_k]
        exact_ids = [int(item) for item in exact_mv_topk[query_index].tolist()]
        overlap = len(set(exact_ids) & set(ranked_ids)) / float(max(len(exact_ids), 1))
        exact_top_id = doc_ids[exact_ids[0]] if exact_ids else None
        screened_top_id = doc_ids[ranked_ids[0]] if ranked_ids else None
        qualitative.append(
            {
                "query_id": query["query_id"],
                "text": query["text"],
                "full_precision_top_id": exact_top_id,
                "screened_top_id": screened_top_id,
                "top1_match": bool(exact_ids and ranked_ids and exact_ids[0] == ranked_ids[0]),
                "topk_overlap": overlap,
                "candidate_pool_size": len(effective_ids),
            }
        )
        relevance_map = {
            doc_id_to_index[external_id]: value
            for external_id, value in query["relevance_map"].items()
            if external_id in doc_id_to_index
        }
        rank_rows.append(_ranking_metrics(ranked_ids, relevance_map, k=top_k))

    qualitative.sort(key=lambda item: (not item["top1_match"], -item["topk_overlap"]), reverse=True)

    def _screened_search_once():
        rows = np.full((len(query_vectors), top_k), -1, dtype=np.int32)
        for query_index, query_vector in enumerate(query_vectors):
            screened_ids = engine.screen_candidates(
                query_embedding=query_vector,
                top_k=top_k,
                candidate_budget=default_budget,
            )
            if screened_ids is None:
                screened_results = engine.search(
                    query_embedding=query_vector,
                    top_k=top_k,
                )
            else:
                screened_results = engine.search(
                    query_embedding=query_vector,
                    top_k=top_k,
                    candidate_ids=screened_ids,
                )
            ranked_ids = [
                int(doc_id_to_index[result.doc_id])
                for result in screened_results
                if result.doc_id in doc_id_to_index
            ]
            rows[query_index, : min(len(ranked_ids), top_k)] = ranked_ids[:top_k]
        return rows

    def _full_precision_search_once():
        rows = np.full((len(query_vectors), top_k), -1, dtype=np.int32)
        for query_index, query_vector in enumerate(query_vectors):
            exact_results = engine.search(
                query_embedding=query_vector,
                top_k=top_k,
            )
            ranked_ids = [
                int(doc_id_to_index[result.doc_id])
                for result in exact_results
                if result.doc_id in doc_id_to_index
            ]
            rows[query_index, : min(len(ranked_ids), top_k)] = ranked_ids[:top_k]
        return rows

    _, full_precision_engine_benchmark = _benchmark_repeated(
        _full_precision_search_once,
        synchronize_device="cuda" if str(getattr(engine, "device", "cpu")).startswith("cuda") else None,
        warmup_runs=1,
        measure_runs=5,
    )
    _, screened_benchmark = _benchmark_repeated(
        _screened_search_once,
        synchronize_device="cuda" if str(getattr(engine, "device", "cpu")).startswith("cuda") else None,
        warmup_runs=1,
        measure_runs=5,
    )
    engine_stats = engine.get_statistics()
    screening_stats = engine_stats.get("screening") or engine_stats.get("prototype_screening", {})
    screening_state = dict(engine_stats.get("screening_state") or {})
    return {
        "screening_backend": engine_stats.get("screening_backend"),
        "screening_state": screening_state,
        "recall_at_1": _mean_recall_at_k(exact_mv_topk[:, :1], padded_rankings[:, :1], top_k=1),
        "recall_at_k": _mean_recall_at_k(exact_mv_topk, padded_rankings, top_k=top_k),
        "top_1_agreement": float(np.mean(exact_mv_topk[:, 0] == padded_rankings[:, 0])),
        "rank_metrics": _mean_metric(rank_rows),
        "candidate_pool_retention": candidate_pool_rows,
        "full_recovery_candidate_count": _first_full_recovery_candidate_count(candidate_pool_rows),
        "fallback_query_count": int(fallback_query_count),
        "fallback_rate": float(fallback_query_count) / float(max(len(query_vectors), 1)),
        "fallback_routes_exact": bool(fallback_query_count > 0),
        "direct_gather_query_count": int(direct_gather_query_count),
        "direct_gather_rate": float(direct_gather_query_count) / float(max(len(query_vectors), 1)),
        "bootstrap_calibration_passed": screening_state.get("health") == "healthy",
        "elapsed_ms": screened_benchmark["median_ms"],
        "full_precision_elapsed_ms": full_precision_engine_benchmark["median_ms"],
        "speedup_vs_full_precision": (
            full_precision_engine_benchmark["median_ms"] / screened_benchmark["median_ms"]
            if screened_benchmark["median_ms"] > 0.0
            else None
        ),
        "screening_profile_ms": screening_profile_ms,
        "exact_profile_ms": exact_profile_ms,
        "benchmark": {
            "full_precision_kernel": full_precision_benchmark,
            "full_precision_engine": full_precision_engine_benchmark,
            "screened_exact": screened_benchmark,
        },
        "storage_mb": screening_stats.get("storage_mb"),
        "avg_prototypes_per_doc": screening_stats.get("avg_prototypes_per_doc"),
        "prototype_count": screening_stats.get("prototype_count"),
        "qualitative_examples": qualitative[:6],
    }


def _benchmark_multimodal_sidecar_maintenance(
    *,
    embed_dim: int,
    screening_backend: str,
    prototype_doc_prototypes: int,
    prototype_query_prototypes: int,
    padded_docs: torch.Tensor,
    doc_ids: list[Any],
    doc_lengths: list[int],
    device: str,
) -> dict[str, Any]:
    if len(doc_ids) < 2:
        return {
            "status": "skipped",
            "reason": "requires_at_least_two_documents",
        }

    from voyager_index._internal.inference.engines.colpali import ColPaliConfig, ColPaliEngine

    base_count = len(doc_ids) - 1
    add_ids = [doc_ids[-1]]
    add_lengths = [int(doc_lengths[-1])]
    add_embeddings = padded_docs[-1:].detach().cpu().numpy()
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ColPaliEngine(
            Path(tmpdir) / "maintenance",
            config=ColPaliConfig(
                embed_dim=embed_dim,
                device=device,
                use_quantization=False,
                use_prototype_screening=True,
                prototype_doc_prototypes=prototype_doc_prototypes,
                prototype_query_prototypes=prototype_query_prototypes,
                screening_backend=screening_backend,
            ),
            device=device,
            load_if_exists=False,
        )
        engine.index(
            embeddings=padded_docs[:base_count].detach().cpu().numpy(),
            doc_ids=doc_ids[:base_count],
            lengths=doc_lengths[:base_count],
        )
        start = time.perf_counter()
        engine.add_documents(
            embeddings=add_embeddings,
            doc_ids=add_ids,
            lengths=add_lengths,
        )
        add_ms = (time.perf_counter() - start) * 1000.0
        add_state = dict(engine.get_statistics().get("screening_state") or {})
        start = time.perf_counter()
        engine.delete_documents(add_ids)
        delete_ms = (time.perf_counter() - start) * 1000.0
        delete_state = dict(engine.get_statistics().get("screening_state") or {})
        start = time.perf_counter()
        engine.compact()
        compact_ms = (time.perf_counter() - start) * 1000.0
        compact_state = dict(engine.get_statistics().get("screening_state") or {})
        engine.close()

    return {
        "status": "passed",
        "add_ms": add_ms,
        "delete_ms": delete_ms,
        "compact_ms": compact_ms,
        "post_add_health": add_state.get("health"),
        "post_delete_health": delete_state.get("health"),
        "post_compact_health": compact_state.get("health"),
    }


def _retention_at_budget(
    summary: dict[str, Any],
    candidate_budget: int,
    field: str,
) -> float | None:
    rows = list(summary.get("candidate_pool_retention", []))
    if not rows:
        return None
    requested = int(candidate_budget)
    exact_row = next(
        (row for row in rows if int(row.get("candidate_count", -1)) == requested),
        None,
    )
    if exact_row is None:
        exact_row = next(
            (row for row in rows if int(row.get("candidate_count", -1)) >= requested),
            rows[-1],
        )
    if exact_row is not None:
        value = exact_row.get(field)
        return float(value) if value is not None else None
    return None


def _centroid_promotion_gate(
    *,
    candidate_budget: int,
    centroid_summary: dict[str, Any],
    pooled_summary: dict[str, Any],
    prototype_summary: dict[str, Any],
    roq1_summary: dict[str, Any],
    roq2_summary: dict[str, Any],
) -> dict[str, Any]:
    centroid_topk_retention = _retention_at_budget(centroid_summary, candidate_budget, "topk_retention")
    centroid_top1_retention = _retention_at_budget(centroid_summary, candidate_budget, "top1_retention")
    lowbit_topk_baseline = max(
        value
        for value in (
            _retention_at_budget(roq1_summary, candidate_budget, "topk_retention"),
            _retention_at_budget(roq2_summary, candidate_budget, "topk_retention"),
        )
        if value is not None
    ) if any(
        value is not None
        for value in (
            _retention_at_budget(roq1_summary, candidate_budget, "topk_retention"),
            _retention_at_budget(roq2_summary, candidate_budget, "topk_retention"),
        )
    ) else None
    lowbit_top1_baseline = max(
        value
        for value in (
            _retention_at_budget(roq1_summary, candidate_budget, "top1_retention"),
            _retention_at_budget(roq2_summary, candidate_budget, "top1_retention"),
        )
        if value is not None
    ) if any(
        value is not None
        for value in (
            _retention_at_budget(roq1_summary, candidate_budget, "top1_retention"),
            _retention_at_budget(roq2_summary, candidate_budget, "top1_retention"),
        )
    ) else None

    end_to_end_wins = (
        centroid_summary.get("elapsed_ms") is not None
        and prototype_summary.get("elapsed_ms") is not None
        and float(centroid_summary["elapsed_ms"]) < float(prototype_summary["elapsed_ms"])
    )
    materially_better_than_lowbit = (
        centroid_topk_retention is not None
        and centroid_top1_retention is not None
        and lowbit_topk_baseline is not None
        and lowbit_top1_baseline is not None
        and centroid_topk_retention >= (lowbit_topk_baseline + 0.10)
        and centroid_top1_retention >= lowbit_top1_baseline
    )
    pooled_quality = pooled_summary.get("recall_at_k")
    centroid_quality = centroid_summary.get("recall_at_k")
    competitive_with_pooled = (
        centroid_quality is not None
        and pooled_quality is not None
        and float(centroid_quality) >= (float(pooled_quality) - 0.05)
        and float(centroid_summary.get("elapsed_ms", float("inf"))) < float(pooled_summary.get("elapsed_ms", float("inf")))
    )
    contract_ready = all(
        summary.get("screening_backend") is not None
        for summary in (centroid_summary, pooled_summary, prototype_summary)
    )
    promote = bool(
        end_to_end_wins
        and materially_better_than_lowbit
        and competitive_with_pooled
        and contract_ready
    )
    return {
        "candidate_budget": int(candidate_budget),
        "criteria": {
            "beats_current_optimized_end_to_end": end_to_end_wins,
            "beats_lowbit_on_candidate_preservation": materially_better_than_lowbit,
            "competitive_with_pooled_and_faster": competitive_with_pooled,
            "contract_ready": contract_ready,
        },
        "comparisons": {
            "centroid_elapsed_ms": centroid_summary.get("elapsed_ms"),
            "current_optimized_elapsed_ms": prototype_summary.get("elapsed_ms"),
            "pooled_elapsed_ms": pooled_summary.get("elapsed_ms"),
            "centroid_topk_retention": centroid_topk_retention,
            "best_lowbit_topk_retention": lowbit_topk_baseline,
            "centroid_top1_retention": centroid_top1_retention,
            "best_lowbit_top1_retention": lowbit_top1_baseline,
            "centroid_recall_at_k": centroid_quality,
            "pooled_recall_at_k": pooled_quality,
        },
        "promotion_recommendation": "promote_to_optimized" if promote else "keep_experimental",
    }


def _benchmark_centroid_scale_harness(
    *,
    dim: int,
    n_docs: int = 10000,
    doc_tokens: int = 64,
    query_tokens: int = 32,
    top_k: int = 10,
    candidate_budget: int = 500,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "cuda_unavailable"}

    from voyager_index._internal.inference.index_core.centroid_screening import CentroidScreeningIndex

    rng = np.random.default_rng(42)
    n_topics = 128
    topic_centroids = rng.standard_normal((n_topics, dim)).astype(np.float32)
    topic_centroids /= np.linalg.norm(topic_centroids, axis=1, keepdims=True) + 1e-8
    doc_ids = [f"synthetic-doc-{idx}" for idx in range(n_docs)]
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    docs = np.zeros((n_docs, doc_tokens, dim), dtype=np.float32)
    doc_topics: list[np.ndarray] = []
    for doc_index in range(n_docs):
        topics = rng.choice(n_topics, size=rng.integers(1, 4), replace=False)
        doc_topics.append(np.asarray(topics, dtype=np.int32))
        for token_index in range(doc_tokens):
            topic = int(rng.choice(topics))
            noise = rng.standard_normal(dim).astype(np.float32) * 0.25
            vector = topic_centroids[topic] + noise
            docs[doc_index, token_index] = vector / (np.linalg.norm(vector) + 1e-8)

    query_topics = [doc_topics[idx] for idx in rng.choice(n_docs, size=20, replace=False)]
    queries = np.zeros((len(query_topics), query_tokens, dim), dtype=np.float32)
    for query_index, topics in enumerate(query_topics):
        for token_index in range(query_tokens):
            topic = int(rng.choice(topics))
            noise = rng.standard_normal(dim).astype(np.float32) * 0.20
            vector = topic_centroids[topic] + noise
            queries[query_index, token_index] = vector / (np.linalg.norm(vector) + 1e-8)

    query_tensor = torch.from_numpy(queries).to("cuda", dtype=torch.float16)
    docs_tensor = torch.from_numpy(docs).to("cuda", dtype=torch.float16)
    documents_mask = torch.ones((n_docs, doc_tokens), device="cuda", dtype=torch.float32)
    screener = CentroidScreeningIndex(
        Path(tempfile.mkdtemp(prefix="centroid-scale-")) / "centroid",
        dim=dim,
        default_doc_prototypes=4,
        device="cuda",
        load_if_exists=False,
    )
    try:
        screener.build(doc_ids=doc_ids, embeddings=docs)

        def _full_precision_once():
            scores = fast_colbert_scores(
                query_tensor,
                docs_tensor,
                documents_mask=documents_mask,
                use_quantization=False,
            )
            return torch.topk(scores, k=top_k, dim=1).indices

        exact_topk, exact_bench = _benchmark_repeated(
            _full_precision_once,
            synchronize_device="cuda",
            warmup_runs=1,
            measure_runs=3,
        )
        exact_np = exact_topk.detach().cpu().numpy()
        candidate_rows = []

        def _centroid_once():
            rows = []
            for query in queries:
                screened_ids = screener.search(
                    query_embedding=query,
                    top_k=top_k,
                    candidate_budget=candidate_budget,
                )
                rows.append([int(doc_id_to_index[doc_id]) for doc_id in screened_ids if doc_id in doc_id_to_index])
            return rows

        screened_rows, screened_bench = _benchmark_repeated(
            _centroid_once,
            synchronize_device="cuda",
            warmup_runs=1,
            measure_runs=3,
        )
        candidate_rows = screened_rows
        recall_at_k = _candidate_retention_from_rows(exact_np, candidate_rows, top_k=top_k)
        recall_at_1 = _candidate_retention_from_rows(exact_np[:, :1], candidate_rows, top_k=1)
        return {
            "status": "passed",
            "n_docs": n_docs,
            "candidate_budget": candidate_budget,
            "top_k": top_k,
            "recall_at_1": recall_at_1,
            "recall_at_k": recall_at_k,
            "full_precision_elapsed_ms": exact_bench["median_ms"],
            "centroid_elapsed_ms": screened_bench["median_ms"],
            "speedup_vs_full_precision": (
                exact_bench["median_ms"] / screened_bench["median_ms"]
                if screened_bench["median_ms"] > 0.0
                else None
            ),
        }
    finally:
        screener.reset()
        shutil.rmtree(screener.root_path.parent, ignore_errors=True)


def run_quantization_validation(output_dir: Path, evaluation_bundle: dict[str, Any]) -> dict[str, Any]:
    records = evaluation_bundle["records"]
    queries = evaluation_bundle["queries"]
    if not records or not queries:
        return lane("skipped", reason="empty_evaluation_bundle", summary={})

    dense_docs = _normalize_matrix(np.stack([record["dense_vector"] for record in records]).astype(np.float32))
    dense_queries = _normalize_matrix(np.stack([query["dense_vector"] for query in queries]).astype(np.float32))
    top_k = min(5, len(records))
    exact_start = time.perf_counter()
    exact_dense_scores = dense_queries @ dense_docs.T
    exact_elapsed_ms = (time.perf_counter() - exact_start) * 1000.0
    exact_topk = _topk_indices(exact_dense_scores, top_k=top_k)

    results: dict[str, Any] = {
        "corpus_points": len(records),
        "query_count": len(queries),
        "embedding_source": evaluation_bundle["embedding_summary"]["embedding_source"],
    }

    quantizer_1bit = RotationalQuantizer(RoQConfig(dim=dense_docs.shape[1], num_bits=1))
    quantizer_1bit.quantize(dense_docs, store=True)
    start = time.perf_counter()
    indices_1bit, _ = quantizer_1bit.search(dense_queries, top_k=top_k)
    indices_1bit = np.atleast_2d(np.asarray(indices_1bit, dtype=np.int32))
    elapsed_1bit_ms = (time.perf_counter() - start) * 1000.0
    results["roq_1bit"] = {
        "recall_at_1": _mean_recall_at_k(exact_topk[:, :1], indices_1bit[:, :1], top_k=1),
        "recall_at_k": _mean_recall_at_k(exact_topk, indices_1bit, top_k=top_k),
        "elapsed_ms": elapsed_1bit_ms,
    }

    quantizer_4bit = RotationalQuantizer(RoQConfig(dim=dense_docs.shape[1], num_bits=4))
    quantized_4bit = quantizer_4bit.quantize(dense_docs, store=False)
    start = time.perf_counter()
    approx_scores = quantizer_4bit.approximate_scores(
        dense_queries,
        quantized_4bit["codes"],
        quantized_4bit["scales"],
        quantized_4bit["offsets"],
    )
    approx_topk = _topk_indices(approx_scores, top_k=top_k)
    elapsed_4bit_ms = (time.perf_counter() - start) * 1000.0
    quantized_queries_4bit = quantizer_4bit.quantize(dense_queries, store=False)
    symmetric_scores = quantizer_4bit.symmetric_scores(
        quantized_queries_4bit["codes"],
        quantized_queries_4bit["scales"],
        quantized_queries_4bit["offsets"],
        quantized_4bit["codes"],
        quantized_4bit["scales"],
        quantized_4bit["offsets"],
    )
    symmetric_topk = _topk_indices(symmetric_scores, top_k=top_k)
    results["roq_4bit_python_reference"] = {
        "meta_layout": "scalar",
        "score_mode": "asymmetric_fp_query_x_quant_doc",
        "recall_at_1": _mean_recall_at_k(exact_topk[:, :1], approx_topk[:, :1], top_k=1),
        "recall_at_k": _mean_recall_at_k(exact_topk, approx_topk, top_k=top_k),
        "elapsed_ms": elapsed_4bit_ms,
        "full_precision_elapsed_ms": exact_elapsed_ms,
        "score_correlation": _score_correlation(exact_dense_scores, approx_scores),
    }
    results["roq_4bit_symmetric"] = {
        "meta_layout": "scalar",
        "score_mode": "symmetric_quant_query_x_quant_doc",
        "recall_at_1": _mean_recall_at_k(exact_topk[:, :1], symmetric_topk[:, :1], top_k=1),
        "recall_at_k": _mean_recall_at_k(exact_topk, symmetric_topk, top_k=top_k),
        "score_correlation": _score_correlation(exact_dense_scores, symmetric_scores),
    }

    grouped_quantizer_4bit = RotationalQuantizer(RoQConfig(dim=dense_docs.shape[1], num_bits=4, group_size=16))
    grouped_quantized_4bit = grouped_quantizer_4bit.quantize(dense_docs, store=False)
    grouped_scores = grouped_quantizer_4bit.approximate_scores(
        dense_queries,
        grouped_quantized_4bit["codes"],
        grouped_quantized_4bit["scales"],
        grouped_quantized_4bit["offsets"],
    )
    grouped_topk = _topk_indices(grouped_scores, top_k=top_k)
    grouped_queries_4bit = grouped_quantizer_4bit.quantize(dense_queries, store=False)
    grouped_symmetric_scores = grouped_quantizer_4bit.symmetric_scores(
        grouped_queries_4bit["codes"],
        grouped_queries_4bit["scales"],
        grouped_queries_4bit["offsets"],
        grouped_quantized_4bit["codes"],
        grouped_quantized_4bit["scales"],
        grouped_quantized_4bit["offsets"],
    )
    grouped_symmetric_topk = _topk_indices(grouped_symmetric_scores, top_k=top_k)
    results["roq_4bit_grouped_diagnostic"] = {
        "group_size": 16,
        "meta_layout": "grouped",
        "score_mode": "asymmetric_fp_query_x_quant_doc",
        "recall_at_1": _mean_recall_at_k(exact_topk[:, :1], grouped_topk[:, :1], top_k=1),
        "recall_at_k": _mean_recall_at_k(exact_topk, grouped_topk, top_k=top_k),
        "score_correlation": _score_correlation(exact_dense_scores, grouped_scores),
    }
    results["roq_4bit_grouped_symmetric_diagnostic"] = {
        "group_size": 16,
        "meta_layout": "grouped",
        "score_mode": "symmetric_quant_query_x_quant_doc",
        "recall_at_1": _mean_recall_at_k(exact_topk[:, :1], grouped_symmetric_topk[:, :1], top_k=1),
        "recall_at_k": _mean_recall_at_k(exact_topk, grouped_symmetric_topk, top_k=top_k),
        "score_correlation": _score_correlation(exact_dense_scores, grouped_symmetric_scores),
    }

    scalar_codes, scalar_scales, scalar_offsets = int4_quantize(dense_docs)
    scalar_decoded = _normalize_matrix(int4_dequantize(scalar_codes, scalar_scales, scalar_offsets, dense_docs.shape[1]))
    scalar_scores = dense_queries @ scalar_decoded.T
    scalar_topk = _topk_indices(scalar_scores, top_k=top_k)
    results["scalar_int4_baseline"] = {
        "recall_at_1": _mean_recall_at_k(exact_topk[:, :1], scalar_topk[:, :1], top_k=1),
        "recall_at_k": _mean_recall_at_k(exact_topk, scalar_topk, top_k=top_k),
        "score_correlation": _score_correlation(exact_dense_scores, scalar_scores),
    }

    from voyager_index._internal.inference.engines.colpali import ColPaliConfig, ColPaliEngine

    doc_ids = [record["id"] for record in records]
    doc_vectors = [record["multivector"] for record in records]
    query_vectors = [query["multivector"] for query in queries]
    padded_docs, doc_mask = _pad_multivectors(doc_vectors, device="cpu")
    doc_lengths = [int(length) for length in doc_mask.sum(dim=1).tolist()]

    use_triton_roq = (
        evaluation_bundle["embedding_summary"]["embedding_source"] == "real_model"
        and torch.cuda.is_available()
        and ROQ_TRITON_AVAILABLE
    )
    if use_triton_roq:
        padded_queries_cuda, queries_mask_cuda = _pad_multivectors(query_vectors, device="cuda")
        padded_docs_cuda, docs_mask_cuda = _pad_multivectors(doc_vectors, device="cuda")
        production_quantizer = RotationalQuantizer(RoQConfig(dim=int(padded_docs_cuda.shape[-1]), num_bits=4))
        quantized_queries = production_quantizer.quantize(
            padded_queries_cuda.reshape(-1, padded_queries_cuda.shape[-1]),
            store=False,
        )
        quantized_docs = production_quantizer.quantize(
            padded_docs_cuda.reshape(-1, padded_docs_cuda.shape[-1]),
            store=False,
        )
        q_codes, q_meta = production_quantizer.stack_triton_inputs(
            quantized_queries,
            batch_size=int(padded_queries_cuda.shape[0]),
            item_count=int(padded_queries_cuda.shape[1]),
            device=padded_queries_cuda.device,
            include_norm_sq=False,
        )
        d_codes, d_meta = production_quantizer.stack_triton_inputs(
            quantized_docs,
            batch_size=int(padded_docs_cuda.shape[0]),
            item_count=int(padded_docs_cuda.shape[1]),
            device=padded_docs_cuda.device,
            include_norm_sq=False,
        )
        exact_mv_scores, exact_mv_bench = _benchmark_repeated(
            lambda: fast_colbert_scores(
                padded_queries_cuda,
                padded_docs_cuda,
                queries_mask=queries_mask_cuda,
                documents_mask=docs_mask_cuda,
            ),
            synchronize_device="cuda",
            warmup_runs=1,
            measure_runs=5,
        )
        roq_mv_scores, roq_mv_bench = _benchmark_repeated(
            lambda: roq_maxsim_4bit(
                q_codes,
                q_meta,
                d_codes,
                d_meta,
                queries_mask=queries_mask_cuda,
                documents_mask=docs_mask_cuda,
            ),
            synchronize_device="cuda",
            warmup_runs=1,
            measure_runs=5,
        )
        exact_mv_scores_np = exact_mv_scores.detach().to("cpu", dtype=torch.float32).numpy()
        roq_mv_scores_np = roq_mv_scores.detach().to("cpu", dtype=torch.float32).numpy()
        exact_mv_topk = _topk_indices(exact_mv_scores_np, top_k=top_k)
        roq_mv_topk = _topk_indices(roq_mv_scores_np, top_k=top_k)
        external_to_internal = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        exact_rank_rows = []
        roq_rank_rows = []
        qualitative = []
        top1_harm_count = 0
        for query_index, query in enumerate(queries):
            internal_relevance_map = {
                external_to_internal[external_id]: value
                for external_id, value in query["relevance_map"].items()
                if external_id in external_to_internal
            }
            exact_ids = [int(item) for item in exact_mv_topk[query_index].tolist()]
            roq_ids = [int(item) for item in roq_mv_topk[query_index].tolist()]
            exact_metrics = _ranking_metrics(exact_ids, internal_relevance_map, k=top_k)
            roq_metrics = _ranking_metrics(roq_ids, internal_relevance_map, k=top_k)
            exact_rank_rows.append(exact_metrics)
            roq_rank_rows.append(roq_metrics)
            exact_top_relevance = float(internal_relevance_map.get(exact_ids[0], 0.0)) if exact_ids else 0.0
            roq_top_relevance = float(internal_relevance_map.get(roq_ids[0], 0.0)) if roq_ids else 0.0
            top1_changed = bool(exact_ids and roq_ids and exact_ids[0] != roq_ids[0])
            harmful_change = top1_changed and exact_top_relevance > 0.0 and roq_top_relevance <= 0.0
            if harmful_change:
                top1_harm_count += 1
            overlap = len(set(exact_ids) & set(roq_ids)) / float(max(len(exact_ids), 1))
            qualitative.append(
                {
                    "query_id": query["query_id"],
                    "text": query["text"],
                    "top1_changed": top1_changed,
                    "harmful_top1_change": harmful_change,
                    "full_precision_top_id": doc_ids[exact_ids[0]] if exact_ids else None,
                    "roq_top_id": doc_ids[roq_ids[0]] if roq_ids else None,
                    "full_precision_top_relevance": exact_top_relevance,
                    "roq_top_relevance": roq_top_relevance,
                    "topk_overlap": overlap,
                    "ndcg_delta": roq_metrics["ndcg"] - exact_metrics["ndcg"],
                }
            )
        qualitative.sort(
            key=lambda item: (
                item["harmful_top1_change"],
                item["top1_changed"],
                abs(item["ndcg_delta"]),
            ),
            reverse=True,
        )
        exact_avg_rank = _mean_metric(exact_rank_rows)
        roq_avg_rank = _mean_metric(roq_rank_rows)
        bytes_per_token_roq = int(d_codes.shape[-1]) + (int(d_meta.shape[-1]) * 4)
        bytes_per_token_fp32 = int(padded_docs_cuda.shape[-1]) * 4
        bytes_per_token_fp16 = int(padded_docs_cuda.shape[-1]) * 2
        total_roq_bytes = int(d_codes.numel() + (d_meta.numel() * 4))
        total_fp32_bytes = int(padded_docs_cuda.numel() * 4)
        total_fp16_bytes = int(padded_docs_cuda.numel() * 2)
        ndcg_delta = roq_avg_rank["ndcg"] - exact_avg_rank["ndcg"]
        sweep_rows = []
        crossover_candidate_count = None
        for candidate_count in _candidate_sweep_counts(len(doc_ids)):
            sweep_docs_cuda, sweep_docs_mask_cuda = _repeat_candidate_tensors(
                padded_docs_cuda,
                docs_mask_cuda,
                candidate_count,
            )
            sweep_d_codes = _repeat_candidate_rows(d_codes, candidate_count)
            sweep_d_meta = _repeat_candidate_rows(d_meta, candidate_count)
            _, sweep_fp_bench = _benchmark_repeated(
                lambda: fast_colbert_scores(
                    padded_queries_cuda,
                    sweep_docs_cuda,
                    queries_mask=queries_mask_cuda,
                    documents_mask=sweep_docs_mask_cuda,
                ),
                synchronize_device="cuda",
                warmup_runs=1,
                measure_runs=5,
            )
            _, sweep_roq_bench = _benchmark_repeated(
                lambda: roq_maxsim_4bit(
                    q_codes,
                    q_meta,
                    sweep_d_codes,
                    sweep_d_meta,
                    queries_mask=queries_mask_cuda,
                    documents_mask=sweep_docs_mask_cuda,
                ),
                synchronize_device="cuda",
                warmup_runs=1,
                measure_runs=5,
            )
            fp_ms = float(sweep_fp_bench["median_ms"])
            roq_ms = float(sweep_roq_bench["median_ms"])
            speedup = (fp_ms / roq_ms) if roq_ms > 0.0 else None
            row = {
                "candidate_count": int(candidate_count),
                "total_document_tokens": int(sweep_docs_mask_cuda.sum().item()),
                "full_precision_elapsed_ms": fp_ms,
                "roq4_elapsed_ms": roq_ms,
                "speedup_vs_full_precision": speedup,
                "roq4_storage_bytes": int(sweep_d_codes.numel() + (sweep_d_meta.numel() * 4)),
                "fp16_storage_bytes": int(sweep_docs_cuda.numel() * 2),
                "fp32_storage_bytes": int(sweep_docs_cuda.numel() * 4),
            }
            sweep_rows.append(row)
            if crossover_candidate_count is None and speedup is not None and speedup >= 1.0:
                crossover_candidate_count = int(candidate_count)

        results["roq_4bit"] = {
            "status": "passed",
            "implementation": "triton_cuda_scalar_meta",
            "meta_layout": "scalar",
            "recall_at_1": _mean_recall_at_k(exact_mv_topk[:, :1], roq_mv_topk[:, :1], top_k=1),
            "recall_at_k": _mean_recall_at_k(exact_mv_topk, roq_mv_topk, top_k=top_k),
            "top_1_agreement": float(np.mean(exact_mv_topk[:, 0] == roq_mv_topk[:, 0])),
            "top_1_harm_count": top1_harm_count,
            "top_1_harm_rate": top1_harm_count / float(max(len(queries), 1)),
            "full_precision_rank_metrics": exact_avg_rank,
            "rank_metrics": roq_avg_rank,
            "ndcg_delta_vs_full_precision": ndcg_delta,
            "recall_delta_vs_full_precision": roq_avg_rank["recall"] - exact_avg_rank["recall"],
            "elapsed_ms": roq_mv_bench["median_ms"],
            "full_precision_elapsed_ms": exact_mv_bench["median_ms"],
            "speedup_vs_full_precision": (
                exact_mv_bench["median_ms"] / roq_mv_bench["median_ms"]
                if roq_mv_bench["median_ms"] > 0.0
                else None
            ),
            "benchmark": {
                "full_precision": exact_mv_bench,
                "triton_roq4": roq_mv_bench,
            },
            "bytes_per_token": {
                "roq4": bytes_per_token_roq,
                "fp16": bytes_per_token_fp16,
                "fp32": bytes_per_token_fp32,
            },
            "compression_vs_fp16": bytes_per_token_fp16 / float(bytes_per_token_roq),
            "compression_vs_fp32": bytes_per_token_fp32 / float(bytes_per_token_roq),
            "storage_bytes": {
                "roq4": total_roq_bytes,
                "fp16": total_fp16_bytes,
                "fp32": total_fp32_bytes,
            },
            "candidate_sweep": {
                "crossover_candidate_count": crossover_candidate_count,
                "counts": sweep_rows,
            },
            "quality_gate": {
                "passed": top1_harm_count == 0 and ndcg_delta >= -0.05,
                "max_allowed_ndcg_drop": -0.05,
            },
            "qualitative_examples": qualitative[:6],
        }

        screening_budgets = sorted(
            {
                int(budget)
                for budget in (top_k, 8, 16, 32, 64, 128, 256, 512, 1024, len(doc_ids))
                if 0 < int(budget) <= len(doc_ids)
            }
        )

        onebit_quantizer = RotationalQuantizer(RoQConfig(dim=int(padded_docs_cuda.shape[-1]), num_bits=1))
        onebit_query_codes, onebit_query_meta = onebit_quantizer.build_1bit_query_triton_inputs(
            padded_queries_cuda.reshape(-1, padded_queries_cuda.shape[-1]),
            batch_size=int(padded_queries_cuda.shape[0]),
            item_count=int(padded_queries_cuda.shape[1]),
            device=padded_queries_cuda.device,
            include_norm_sq=False,
        )
        onebit_quantized_docs = onebit_quantizer.quantize(
            padded_docs_cuda.reshape(-1, padded_docs_cuda.shape[-1]),
            store=False,
        )
        onebit_doc_codes = onebit_quantizer.build_1bit_doc_triton_inputs(
            onebit_quantized_docs,
            batch_size=int(padded_docs_cuda.shape[0]),
            item_count=int(padded_docs_cuda.shape[1]),
            device=padded_docs_cuda.device,
        )
        onebit_scores, onebit_bench = _benchmark_repeated(
            lambda: roq_maxsim_1bit(
                onebit_query_codes,
                onebit_doc_codes,
                onebit_query_meta,
                queries_mask=queries_mask_cuda,
                documents_mask=docs_mask_cuda,
            ),
            synchronize_device="cuda",
            warmup_runs=1,
            measure_runs=5,
        )
        onebit_scores_np = onebit_scores.detach().to("cpu", dtype=torch.float32).numpy()
        onebit_topk = _topk_indices(onebit_scores_np, top_k=top_k)
        onebit_sweep_rows = []
        onebit_crossover_candidate_count = None
        for candidate_count in _candidate_sweep_counts(len(doc_ids)):
            sweep_docs_cuda, sweep_docs_mask_cuda = _repeat_candidate_tensors(
                padded_docs_cuda,
                docs_mask_cuda,
                candidate_count,
            )
            sweep_doc_codes = _repeat_candidate_rows(onebit_doc_codes, candidate_count)
            _, sweep_fp_bench = _benchmark_repeated(
                lambda: fast_colbert_scores(
                    padded_queries_cuda,
                    sweep_docs_cuda,
                    queries_mask=queries_mask_cuda,
                    documents_mask=sweep_docs_mask_cuda,
                ),
                synchronize_device="cuda",
                warmup_runs=1,
                measure_runs=3,
            )
            _, sweep_onebit_bench = _benchmark_repeated(
                lambda: roq_maxsim_1bit(
                    onebit_query_codes,
                    sweep_doc_codes,
                    onebit_query_meta,
                    queries_mask=queries_mask_cuda,
                    documents_mask=sweep_docs_mask_cuda,
                ),
                synchronize_device="cuda",
                warmup_runs=1,
                measure_runs=3,
            )
            fp_ms = float(sweep_fp_bench["median_ms"])
            roq_ms = float(sweep_onebit_bench["median_ms"])
            speedup = (fp_ms / roq_ms) if roq_ms > 0.0 else None
            onebit_sweep_rows.append(
                {
                    "candidate_count": int(candidate_count),
                    "total_document_tokens": int(sweep_docs_mask_cuda.sum().item()),
                    "full_precision_elapsed_ms": fp_ms,
                    "roq1_elapsed_ms": roq_ms,
                    "speedup_vs_full_precision": speedup,
                    "roq1_storage_bytes": int(sweep_doc_codes.numel() * 4),
                    "fp16_storage_bytes": int(sweep_docs_cuda.numel() * 2),
                    "fp32_storage_bytes": int(sweep_docs_cuda.numel() * 4),
                }
            )
            if onebit_crossover_candidate_count is None and speedup is not None and speedup >= 1.0:
                onebit_crossover_candidate_count = int(candidate_count)
        bytes_per_token_roq1 = int(onebit_doc_codes.shape[-1] * 4)
        total_roq1_bytes = int(onebit_doc_codes.numel() * 4)
        results["roq_1bit_screening"] = {
            "status": "passed",
            "implementation": "triton_cuda_asymmetric_query_quantized",
            "meta_layout": "binary_screening",
            "query_bits": onebit_quantizer.config.query_bits,
            "recall_at_1": _mean_recall_at_k(exact_mv_topk[:, :1], onebit_topk[:, :1], top_k=1),
            "recall_at_k": _mean_recall_at_k(exact_mv_topk, onebit_topk, top_k=top_k),
            "top_1_agreement": float(np.mean(exact_mv_topk[:, 0] == onebit_topk[:, 0])),
            "score_correlation": _score_correlation(exact_mv_scores_np, onebit_scores_np),
            "elapsed_ms": onebit_bench["median_ms"],
            "full_precision_elapsed_ms": exact_mv_bench["median_ms"],
            "speedup_vs_full_precision": (
                exact_mv_bench["median_ms"] / onebit_bench["median_ms"]
                if onebit_bench["median_ms"] > 0.0
                else None
            ),
            "benchmark": {
                "full_precision": exact_mv_bench,
                "triton_roq1": onebit_bench,
            },
            "bytes_per_token": {
                "roq1": bytes_per_token_roq1,
                "fp16": bytes_per_token_fp16,
                "fp32": bytes_per_token_fp32,
            },
            "compression_vs_fp16": bytes_per_token_fp16 / float(bytes_per_token_roq1),
            "compression_vs_fp32": bytes_per_token_fp32 / float(bytes_per_token_roq1),
            "storage_bytes": {
                "roq1": total_roq1_bytes,
                "fp16": total_fp16_bytes,
                "fp32": total_fp32_bytes,
            },
            "candidate_pool_retention": _screening_budget_summary(
                exact_mv_topk,
                onebit_scores_np,
                screening_budgets,
                top_k=top_k,
            ),
            "candidate_sweep": {
                "crossover_candidate_count": onebit_crossover_candidate_count,
                "counts": onebit_sweep_rows,
            },
        }

        twobit_quantizer = RotationalQuantizer(
            RoQConfig(dim=int(padded_docs_cuda.shape[-1]), num_bits=2, group_size=int(padded_docs_cuda.shape[-1]))
        )
        twobit_quantized_queries = twobit_quantizer.quantize(
            padded_queries_cuda.reshape(-1, padded_queries_cuda.shape[-1]),
            store=False,
        )
        twobit_quantized_docs = twobit_quantizer.quantize(
            padded_docs_cuda.reshape(-1, padded_docs_cuda.shape[-1]),
            store=False,
        )
        twobit_query_codes, twobit_query_meta = twobit_quantizer.stack_triton_inputs(
            twobit_quantized_queries,
            batch_size=int(padded_queries_cuda.shape[0]),
            item_count=int(padded_queries_cuda.shape[1]),
            device=padded_queries_cuda.device,
            include_norm_sq=False,
        )
        twobit_doc_codes, twobit_doc_meta = twobit_quantizer.stack_triton_inputs(
            twobit_quantized_docs,
            batch_size=int(padded_docs_cuda.shape[0]),
            item_count=int(padded_docs_cuda.shape[1]),
            device=padded_docs_cuda.device,
            include_norm_sq=False,
        )
        twobit_scores, twobit_bench = _benchmark_repeated(
            lambda: roq_maxsim_2bit(
                twobit_query_codes,
                twobit_query_meta,
                twobit_doc_codes,
                twobit_doc_meta,
                queries_mask=queries_mask_cuda,
                documents_mask=docs_mask_cuda,
            ),
            synchronize_device="cuda",
            warmup_runs=1,
            measure_runs=5,
        )
        twobit_scores_np = twobit_scores.detach().to("cpu", dtype=torch.float32).numpy()
        twobit_topk = _topk_indices(twobit_scores_np, top_k=top_k)
        twobit_sweep_rows = []
        twobit_crossover_candidate_count = None
        for candidate_count in _candidate_sweep_counts(len(doc_ids)):
            sweep_docs_cuda, sweep_docs_mask_cuda = _repeat_candidate_tensors(
                padded_docs_cuda,
                docs_mask_cuda,
                candidate_count,
            )
            sweep_doc_codes = _repeat_candidate_rows(twobit_doc_codes, candidate_count)
            sweep_doc_meta = _repeat_candidate_rows(twobit_doc_meta, candidate_count)
            _, sweep_fp_bench = _benchmark_repeated(
                lambda: fast_colbert_scores(
                    padded_queries_cuda,
                    sweep_docs_cuda,
                    queries_mask=queries_mask_cuda,
                    documents_mask=sweep_docs_mask_cuda,
                ),
                synchronize_device="cuda",
                warmup_runs=1,
                measure_runs=3,
            )
            _, sweep_twobit_bench = _benchmark_repeated(
                lambda: roq_maxsim_2bit(
                    twobit_query_codes,
                    twobit_query_meta,
                    sweep_doc_codes,
                    sweep_doc_meta,
                    queries_mask=queries_mask_cuda,
                    documents_mask=sweep_docs_mask_cuda,
                ),
                synchronize_device="cuda",
                warmup_runs=1,
                measure_runs=3,
            )
            fp_ms = float(sweep_fp_bench["median_ms"])
            roq_ms = float(sweep_twobit_bench["median_ms"])
            speedup = (fp_ms / roq_ms) if roq_ms > 0.0 else None
            twobit_sweep_rows.append(
                {
                    "candidate_count": int(candidate_count),
                    "total_document_tokens": int(sweep_docs_mask_cuda.sum().item()),
                    "full_precision_elapsed_ms": fp_ms,
                    "roq2_elapsed_ms": roq_ms,
                    "speedup_vs_full_precision": speedup,
                    "roq2_storage_bytes": int(sweep_doc_codes.numel() + (sweep_doc_meta.numel() * 4)),
                    "fp16_storage_bytes": int(sweep_docs_cuda.numel() * 2),
                    "fp32_storage_bytes": int(sweep_docs_cuda.numel() * 4),
                }
            )
            if twobit_crossover_candidate_count is None and speedup is not None and speedup >= 1.0:
                twobit_crossover_candidate_count = int(candidate_count)
        bytes_per_token_roq2 = int(twobit_doc_codes.shape[-1]) + (int(twobit_doc_meta.shape[-1]) * 4)
        total_roq2_bytes = int(twobit_doc_codes.numel() + (twobit_doc_meta.numel() * 4))
        results["roq_2bit_screening"] = {
            "status": "passed",
            "implementation": "triton_cuda_scalar_meta",
            "meta_layout": "scalar",
            "recall_at_1": _mean_recall_at_k(exact_mv_topk[:, :1], twobit_topk[:, :1], top_k=1),
            "recall_at_k": _mean_recall_at_k(exact_mv_topk, twobit_topk, top_k=top_k),
            "top_1_agreement": float(np.mean(exact_mv_topk[:, 0] == twobit_topk[:, 0])),
            "score_correlation": _score_correlation(exact_mv_scores_np, twobit_scores_np),
            "elapsed_ms": twobit_bench["median_ms"],
            "full_precision_elapsed_ms": exact_mv_bench["median_ms"],
            "speedup_vs_full_precision": (
                exact_mv_bench["median_ms"] / twobit_bench["median_ms"]
                if twobit_bench["median_ms"] > 0.0
                else None
            ),
            "benchmark": {
                "full_precision": exact_mv_bench,
                "triton_roq2": twobit_bench,
            },
            "bytes_per_token": {
                "roq2": bytes_per_token_roq2,
                "fp16": bytes_per_token_fp16,
                "fp32": bytes_per_token_fp32,
            },
            "compression_vs_fp16": bytes_per_token_fp16 / float(bytes_per_token_roq2),
            "compression_vs_fp32": bytes_per_token_fp32 / float(bytes_per_token_roq2),
            "storage_bytes": {
                "roq2": total_roq2_bytes,
                "fp16": total_fp16_bytes,
                "fp32": total_fp32_bytes,
            },
            "candidate_pool_retention": _screening_budget_summary(
                exact_mv_topk,
                twobit_scores_np,
                screening_budgets,
                top_k=top_k,
            ),
            "candidate_sweep": {
                "crossover_candidate_count": twobit_crossover_candidate_count,
                "counts": twobit_sweep_rows,
            },
        }

        default_screen_budget = min(max(top_k * 16, 64), len(doc_ids))
        with tempfile.TemporaryDirectory() as prototype_tmpdir:
            pooled_engine = ColPaliEngine(
                Path(prototype_tmpdir) / "pooled-screening",
                config=ColPaliConfig(
                    embed_dim=int(padded_docs.shape[-1]),
                    device="cuda",
                    use_quantization=False,
                    use_prototype_screening=True,
                    prototype_doc_prototypes=1,
                    prototype_query_prototypes=1,
                    screening_backend="prototype_hnsw",
                ),
                device="cuda",
                load_if_exists=False,
            )
            prototype_engine = ColPaliEngine(
                Path(prototype_tmpdir) / "prototype-screening",
                config=ColPaliConfig(
                    embed_dim=int(padded_docs.shape[-1]),
                    device="cuda",
                    use_quantization=False,
                    use_prototype_screening=True,
                    prototype_doc_prototypes=4,
                    prototype_query_prototypes=4,
                    screening_backend="prototype_hnsw",
                ),
                device="cuda",
                load_if_exists=False,
            )
            centroid_engine = ColPaliEngine(
                Path(prototype_tmpdir) / "centroid-screening",
                config=ColPaliConfig(
                    embed_dim=int(padded_docs.shape[-1]),
                    device="cuda",
                    use_quantization=False,
                    use_prototype_screening=True,
                    prototype_doc_prototypes=4,
                    prototype_query_prototypes=4,
                    screening_backend="centroid",
                ),
                device="cuda",
                load_if_exists=False,
            )
            pooled_engine.index(embeddings=padded_docs.cpu().numpy(), doc_ids=doc_ids, lengths=doc_lengths)
            prototype_engine.index(embeddings=padded_docs.cpu().numpy(), doc_ids=doc_ids, lengths=doc_lengths)
            centroid_engine.index(embeddings=padded_docs.cpu().numpy(), doc_ids=doc_ids, lengths=doc_lengths)

            pooled_summary = _evaluate_screened_multimodal_engine(
                engine=pooled_engine,
                queries=queries,
                query_vectors=query_vectors,
                doc_ids=doc_ids,
                doc_id_to_index=external_to_internal,
                exact_mv_topk=exact_mv_topk,
                full_precision_benchmark=exact_mv_bench,
                top_k=top_k,
                candidate_budgets=screening_budgets,
                default_budget=default_screen_budget,
            )
            prototype_summary = _evaluate_screened_multimodal_engine(
                engine=prototype_engine,
                queries=queries,
                query_vectors=query_vectors,
                doc_ids=doc_ids,
                doc_id_to_index=external_to_internal,
                exact_mv_topk=exact_mv_topk,
                full_precision_benchmark=exact_mv_bench,
                top_k=top_k,
                candidate_budgets=screening_budgets,
                default_budget=default_screen_budget,
            )
            centroid_summary = _evaluate_screened_multimodal_engine(
                engine=centroid_engine,
                queries=queries,
                query_vectors=query_vectors,
                doc_ids=doc_ids,
                doc_id_to_index=external_to_internal,
                exact_mv_topk=exact_mv_topk,
                full_precision_benchmark=exact_mv_bench,
                top_k=top_k,
                candidate_budgets=screening_budgets,
                default_budget=default_screen_budget,
            )
            pooled_maintenance = _benchmark_multimodal_sidecar_maintenance(
                embed_dim=int(padded_docs.shape[-1]),
                screening_backend="prototype_hnsw",
                prototype_doc_prototypes=1,
                prototype_query_prototypes=1,
                padded_docs=padded_docs,
                doc_ids=doc_ids,
                doc_lengths=doc_lengths,
                device="cuda",
            )
            prototype_maintenance = _benchmark_multimodal_sidecar_maintenance(
                embed_dim=int(padded_docs.shape[-1]),
                screening_backend="prototype_hnsw",
                prototype_doc_prototypes=4,
                prototype_query_prototypes=4,
                padded_docs=padded_docs,
                doc_ids=doc_ids,
                doc_lengths=doc_lengths,
                device="cuda",
            )
            centroid_maintenance = _benchmark_multimodal_sidecar_maintenance(
                embed_dim=int(padded_docs.shape[-1]),
                screening_backend="centroid",
                prototype_doc_prototypes=4,
                prototype_query_prototypes=4,
                padded_docs=padded_docs,
                doc_ids=doc_ids,
                doc_lengths=doc_lengths,
                device="cuda",
            )
            pooled_engine.close()
            prototype_engine.close()
            centroid_engine.close()

        results["pooled_dense_screening"] = {
            "status": "passed",
            "implementation": "prototype_sidecar_mean_only",
            "candidate_budget": default_screen_budget,
            "incremental_maintenance": pooled_maintenance,
            **pooled_summary,
        }
        results["prototype_sidecar_screening"] = {
            "status": "passed",
            "implementation": "prototype_sidecar_mean_plus_coverage",
            "candidate_budget": default_screen_budget,
            "incremental_maintenance": prototype_maintenance,
            **prototype_summary,
        }
        results["centroid_screening"] = {
            "status": "passed",
            "implementation": "gpu_centroid_sidecar",
            "candidate_budget": default_screen_budget,
            "incremental_maintenance": centroid_maintenance,
            **centroid_summary,
        }
        results["centroid_screening_scale_harness"] = _benchmark_centroid_scale_harness(
            dim=int(padded_docs.shape[-1]),
            top_k=top_k,
            candidate_budget=min(max(top_k * 50, 100), 500),
        )
        results["centroid_screening_decision_gate"] = _centroid_promotion_gate(
            candidate_budget=default_screen_budget,
            centroid_summary=centroid_summary,
            pooled_summary=pooled_summary,
            prototype_summary=prototype_summary,
            roq1_summary=results["roq_1bit_screening"],
            roq2_summary=results["roq_2bit_screening"],
        )
    else:
        results["roq_4bit"] = {
            **results["roq_4bit_python_reference"],
            "status": "skipped",
            "reason": "triton_cuda_roq4_unavailable",
        }
        results["roq_1bit_screening"] = {
            "status": "skipped",
            "reason": "triton_cuda_real_model_screening_unavailable",
        }
        results["roq_2bit_screening"] = {
            "status": "skipped",
            "reason": "triton_cuda_real_model_screening_unavailable",
        }
        results["pooled_dense_screening"] = {
            "status": "skipped",
            "reason": "prototype_screening_requires_real_model_cuda",
        }
        results["prototype_sidecar_screening"] = {
            "status": "skipped",
            "reason": "prototype_screening_requires_real_model_cuda",
        }
        results["centroid_screening"] = {
            "status": "skipped",
            "reason": "centroid_screening_requires_real_model_cuda",
        }
        results["centroid_screening_scale_harness"] = {
            "status": "skipped",
            "reason": "centroid_screening_requires_cuda",
        }
        results["centroid_screening_decision_gate"] = {
            "promotion_recommendation": "keep_experimental",
            "reason": "real_model_cuda_screening_unavailable",
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        fp_engine = ColPaliEngine(
            Path(tmpdir) / "fp",
            config=ColPaliConfig(embed_dim=int(padded_docs.shape[-1]), device="cpu", use_quantization=False),
            device="cpu",
            load_if_exists=False,
        )
        quantized_engine = ColPaliEngine(
            Path(tmpdir) / "int8",
            config=ColPaliConfig(embed_dim=int(padded_docs.shape[-1]), device="cpu", use_quantization=True),
            device="cpu",
            load_if_exists=False,
        )
        fp_engine.index(embeddings=padded_docs.cpu().numpy(), doc_ids=doc_ids, lengths=doc_lengths)
        quantized_engine.index(embeddings=padded_docs.cpu().numpy(), doc_ids=doc_ids, lengths=doc_lengths)

        fp_rankings = []
        quantized_rankings = []
        fp_elapsed_ms = 0.0
        quantized_elapsed_ms = 0.0
        for query_vector in query_vectors:
            start = time.perf_counter()
            fp_results = fp_engine.search(query_embedding=query_vector, top_k=top_k)
            fp_elapsed_ms += (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            quantized_results = quantized_engine.search(query_embedding=query_vector, top_k=top_k)
            quantized_elapsed_ms += (time.perf_counter() - start) * 1000.0
            fp_rankings.append([doc_ids.index(result.doc_id) for result in fp_results])
            quantized_rankings.append([doc_ids.index(result.doc_id) for result in quantized_results])

    fp_rankings_arr = np.asarray(fp_rankings, dtype=np.int32)
    quantized_rankings_arr = np.asarray(quantized_rankings, dtype=np.int32)
    results["multimodal_int8"] = {
        "recall_at_k": _mean_recall_at_k(fp_rankings_arr, quantized_rankings_arr, top_k=top_k),
        "elapsed_ms_full_precision": fp_elapsed_ms,
        "elapsed_ms_quantized": quantized_elapsed_ms,
    }

    path = output_dir / "quantization.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return lane("passed", path=str(path), summary=results)


def run_maxsim_validation(output_dir: Path, evaluation_bundle: dict[str, Any]) -> dict[str, Any]:
    records = evaluation_bundle["records"][: min(32, len(evaluation_bundle["records"]))]
    queries = evaluation_bundle["queries"][: min(8, len(evaluation_bundle["queries"]))]
    if not records or not queries:
        return lane("skipped", reason="empty_evaluation_bundle", summary={})

    query_vectors = [query["multivector"] for query in queries]
    doc_vectors = [record["multivector"] for record in records]
    if evaluation_bundle["embedding_summary"]["embedding_source"] == "real_model" and len(doc_vectors) < 96:
        multiplier = max(1, math.ceil(96 / float(len(doc_vectors))))
        doc_vectors = (doc_vectors * multiplier)[:96]
    padded_queries_cpu, queries_mask_cpu = _pad_multivectors(query_vectors, device="cpu")
    padded_docs_cpu, docs_mask_cpu = _pad_multivectors(doc_vectors, device="cpu")

    benchmark_module = _load_benchmark_module()
    use_cuda_inputs = (
        evaluation_bundle["embedding_summary"]["embedding_source"] == "real_model"
        and torch.cuda.is_available()
    )
    benchmark_queries = padded_queries_cpu.to("cuda") if use_cuda_inputs else padded_queries_cpu
    benchmark_docs = padded_docs_cpu.to("cuda") if use_cuda_inputs else padded_docs_cpu
    benchmark_queries_mask = queries_mask_cpu.to("cuda") if use_cuda_inputs else queries_mask_cpu
    benchmark_docs_mask = docs_mask_cpu.to("cuda") if use_cuda_inputs else docs_mask_cpu
    processor = evaluation_bundle.get("processor")
    processor_score_fn = None
    if processor is not None:
        processor_score_fn = lambda q, d: processor.score(q, d)

    result = benchmark_module.benchmark_maxsim_tensors(
        benchmark_queries,
        benchmark_docs,
        queries_mask=benchmark_queries_mask,
        documents_mask=benchmark_docs_mask,
        processor_score_fn=processor_score_fn,
    )
    result["corpus_points"] = len(evaluation_bundle["records"])
    result["benchmark_docs_source_count"] = len(records)
    result["benchmark_docs_effective_count"] = int(padded_docs_cpu.shape[0])
    result["query_count"] = len(queries)
    result["embedding_source"] = evaluation_bundle["embedding_summary"]["embedding_source"]

    path = output_dir / "maxsim.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    overall_ok = result["fast_colbert_cpu"]["parity"] and result["triton_cuda"]["status"] != "failed"
    return lane("passed" if overall_ok else "failed", path=str(path), summary=result)


def run_hybrid_storage_validation(output_dir: Path, entities: list[dict[str, Any]]) -> dict[str, Any]:
    from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager

    vectors = np.stack([synthetic_vector(entity["entity_id"], 8) for entity in entities], axis=0)
    ids = [idx for idx in range(len(entities))]
    payloads = [{"entity_id": entity["entity_id"], "text": entity["canonical_name"]} for entity in entities]
    query = vectors[0]
    report: dict[str, Any] = {}

    for on_disk in (False, True):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = HybridSearchManager(Path(tmpdir), dim=8, on_disk=on_disk)
            manager.index(
                corpus=[payload["text"] for payload in payloads],
                vectors=vectors,
                ids=ids,
                payloads=payloads,
            )
            dense = manager.search(query_text="", query_vector=query, k=2)
            sparse = manager.search(query_text=payloads[0]["text"], query_vector=None, k=2)
        report["on_disk" if on_disk else "in_memory"] = {
            "dense_top_id": dense["dense"][0][0] if dense["dense"] else None,
            "sparse_top_id": sparse["sparse"][0][0] if sparse["sparse"] else None,
        }

    path = output_dir / "hybrid_storage.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return lane("passed", path=str(path), summary=report)


def _dense_point_payloads(records: list[dict[str, Any]], *, text_key: str) -> list[dict[str, Any]]:
    points = []
    for record in records:
        points.append(
            {
                "id": record["id"],
                "vector": record["dense_vector"].tolist(),
                "payload": {
                    "text": record[text_key],
                    "source_name": record["source_name"],
                    "document_id": record["document_id"],
                    "page_number": record["page"],
                    "token_count": record["token_count"],
                    "ontology_terms": record["ontology_terms"],
                    "ontology_labels": record["ontology_labels"],
                },
            }
        )
    return points


def _pad_record_multivectors(records: list[dict[str, Any]]) -> tuple[list[np.ndarray], int]:
    max_tokens = max(record["multivector"].shape[0] for record in records)
    padded = []
    for record in records:
        vector = record["multivector"]
        if vector.shape[0] == max_tokens:
            padded.append(vector)
            continue
        pad = np.zeros((max_tokens, vector.shape[1]), dtype=np.float32)
        pad[: vector.shape[0]] = vector
        padded.append(pad)
    return padded, max_tokens


def _multivector_points(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    padded_vectors, max_tokens = _pad_record_multivectors(records)
    points = []
    for record, padded_vector in zip(records, padded_vectors):
        points.append(
            {
                "id": record["id"],
                "vectors": padded_vector.tolist(),
                "payload": {
                    "text": record["enriched_text"],
                    "source_name": record["source_name"],
                    "document_id": record["document_id"],
                    "page_number": record["page"],
                },
            }
        )
    return points, max_tokens


def _pad_query_multivector(query_vector: np.ndarray, max_tokens: int) -> list[list[float]]:
    if query_vector.shape[0] == max_tokens:
        return query_vector.tolist()
    padded = np.zeros((max_tokens, query_vector.shape[1]), dtype=np.float32)
    padded[: query_vector.shape[0]] = query_vector
    return padded.tolist()


def _timed_client_call(fn) -> tuple[Any, float]:
    start = time.perf_counter()
    response = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return response, elapsed_ms


def _timed_client_call_median(fn, *, warmup_runs: int = 1, measure_runs: int = 3) -> tuple[Any, float]:
    response = None
    for _ in range(max(warmup_runs, 0)):
        response = fn()
    elapsed = []
    for _ in range(max(measure_runs, 1)):
        start = time.perf_counter()
        response = fn()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    assert response is not None
    return response, float(np.median(elapsed))


def run_api_validation(
    output_dir: Path,
    evaluation_bundle: dict[str, Any],
    native_packages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from voyager_index._internal.server.main import create_app

    records = evaluation_bundle["records"]
    queries = evaluation_bundle["queries"]
    if not records or not queries:
        return lane("skipped", reason="empty_evaluation_bundle", summary={})

    top_k = min(3, len(records))
    dense_base_points = _dense_point_payloads(records, text_key="base_text")
    dense_enriched_points = _dense_point_payloads(records, text_key="enriched_text")
    multivector_points, multivector_tokens = _multivector_points(records)
    primary_query = queries[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "api"
        app = create_app(index_path=str(index_path))
        results: dict[str, Any] = {}
        with TestClient(app) as client:
            dense_base_collection = client.post(
                "/collections/dense-base",
                json={"dimension": len(dense_base_points[0]["vector"]), "kind": "dense"},
            )
            dense_ontology_collection = client.post(
                "/collections/dense-ontology",
                json={"dimension": len(dense_enriched_points[0]["vector"]), "kind": "dense"},
            )
            li_collection = client.post(
                "/collections/li",
                json={"dimension": len(multivector_points[0]["vectors"][0]), "kind": "late_interaction", "storage_mode": "sync"},
            )
            mm_collection = client.post(
                "/collections/mm",
                json={"dimension": len(multivector_points[0]["vectors"][0]), "kind": "multimodal"},
            )

            dense_ingest, dense_ingest_ms = _timed_client_call(
                lambda: client.post("/collections/dense-base/points", json={"points": dense_base_points})
            )
            dense_ontology_ingest, dense_ontology_ingest_ms = _timed_client_call(
                lambda: client.post("/collections/dense-ontology/points", json={"points": dense_enriched_points})
            )
            li_ingest, li_ingest_ms = _timed_client_call(
                lambda: client.post("/collections/li/points", json={"points": multivector_points})
            )
            mm_ingest, mm_ingest_ms = _timed_client_call(
                lambda: client.post("/collections/mm/points", json={"points": multivector_points})
            )

            dense_vector, dense_vector_ms = _timed_client_call_median(
                lambda: client.post(
                    "/collections/dense-base/search",
                    json={"vector": primary_query["dense_vector"].tolist(), "top_k": top_k},
                )
            )
            dense_text, dense_text_ms = _timed_client_call_median(
                lambda: client.post(
                    "/collections/dense-base/search",
                    json={"query_text": primary_query["text"], "top_k": top_k},
                )
            )
            dense_hybrid, dense_hybrid_ms = _timed_client_call_median(
                lambda: client.post(
                    "/collections/dense-base/search",
                    json={
                        "vector": primary_query["dense_vector"].tolist(),
                        "query_text": primary_query["text"],
                        "top_k": top_k,
                    },
                )
            )
            dense_ontology_search, dense_ontology_search_ms = _timed_client_call_median(
                lambda: client.post(
                    "/collections/dense-ontology/search",
                    json={
                        "vector": primary_query["dense_vector"].tolist(),
                        "query_text": primary_query["text"],
                        "top_k": top_k,
                    },
                )
            )
            dense_optimized = None
            dense_optimized_ms = None
            if native_packages["latence_solver"]["status"] == "installed_and_importable":
                dense_optimized, dense_optimized_ms = _timed_client_call_median(
                    lambda: client.post(
                        "/collections/dense-base/search",
                        json={
                            "vector": primary_query["dense_vector"].tolist(),
                            "query_text": primary_query["text"],
                            "top_k": top_k,
                            "strategy": "optimized",
                            "max_tokens": max(256, int(np.mean([record["token_count"] for record in records]) * 2)),
                            "max_chunks": top_k,
                        },
                    )
                )

            update_payload = dense_base_points[0].copy()
            update_payload["payload"] = {
                **update_payload["payload"],
                "text": f"{update_payload['payload']['text']} updated evidence",
                "updated": True,
            }
            dense_update, dense_update_ms = _timed_client_call(
                lambda: client.post("/collections/dense-base/points", json={"points": [update_payload]})
            )
            dense_delete, dense_delete_ms = _timed_client_call(
                lambda: client.request(
                    "DELETE",
                    "/collections/dense-base/points",
                    json={"ids": [records[-1]["id"]]},
                )
            )
            li_search, li_search_ms = _timed_client_call_median(
                lambda: client.post(
                    "/collections/li/search",
                    json={"vectors": _pad_query_multivector(primary_query["multivector"], multivector_tokens), "top_k": top_k, "with_vector": True},
                )
            )
            mm_search, mm_search_ms = _timed_client_call_median(
                lambda: client.post(
                    "/collections/mm/search",
                    json={"vectors": _pad_query_multivector(primary_query["multivector"], multivector_tokens), "top_k": top_k, "with_vector": True},
                )
            )
            mm_search_profile = dict(getattr(app.state.search_service.collections["mm"].engine, "last_search_profile", {}))
            mm_search_optimized, mm_search_optimized_ms = _timed_client_call_median(
                lambda: client.post(
                    "/collections/mm/search",
                    json={
                        "vectors": _pad_query_multivector(primary_query["multivector"], multivector_tokens),
                        "top_k": top_k,
                        "with_vector": True,
                        "strategy": "optimized",
                    },
                )
            )
            service = app.state.search_service
            li_runtime = service.collections["li"]
            mm_runtime = service.collections["mm"]

            results["collections"] = {
                "dense_base": dense_base_collection.status_code,
                "dense_ontology": dense_ontology_collection.status_code,
                "late_interaction": li_collection.status_code,
                "multimodal": mm_collection.status_code,
            }
            results["ingest"] = {
                "dense_base_ms": dense_ingest_ms,
                "dense_ontology_ms": dense_ontology_ingest_ms,
                "late_interaction_ms": li_ingest_ms,
                "multimodal_ms": mm_ingest_ms,
            }
            results["search"] = {
                "dense_vector_ms": dense_vector_ms,
                "dense_vector_first_id": dense_vector.json()["results"][0]["id"],
                "dense_text_ms": dense_text_ms,
                "dense_text_first_id": dense_text.json()["results"][0]["id"],
                "dense_hybrid_ms": dense_hybrid_ms,
                "dense_hybrid_first_id": dense_hybrid.json()["results"][0]["id"],
                "dense_ontology_ms": dense_ontology_search_ms,
                "dense_ontology_first_id": dense_ontology_search.json()["results"][0]["id"],
                "dense_optimized_ms": dense_optimized_ms,
                "dense_optimized_first_id": (
                    dense_optimized.json()["results"][0]["id"]
                    if dense_optimized is not None and dense_optimized.status_code == 200 and dense_optimized.json()["results"]
                    else None
                ),
                "dense_optimized_objective": (
                    dense_optimized.json().get("objective_score")
                    if dense_optimized is not None and dense_optimized.status_code == 200
                    else None
                ),
                "late_interaction_ms": li_search_ms,
                "late_interaction_first_id": li_search.json()["results"][0]["id"],
                "multimodal_ms": mm_search_ms,
                "multimodal_first_id": mm_search.json()["results"][0]["id"],
                "multimodal_optimized_ms": mm_search_optimized_ms,
                "multimodal_optimized_first_id": mm_search_optimized.json()["results"][0]["id"],
            }
            results["crud"] = {
                "dense_update_status": dense_update.status_code,
                "dense_update_ms": dense_update_ms,
                "dense_delete_status": dense_delete.status_code,
                "dense_delete_ms": dense_delete_ms,
            }
            results["profiles"] = {
                "late_interaction_ingest": getattr(li_runtime.engine, "last_write_profile", {}),
                "late_interaction_storage_write": getattr(li_runtime.engine.storage, "last_write_profile", {}),
                "late_interaction_search": getattr(li_runtime.engine, "last_search_profile", {}),
                "late_interaction_vector_load": getattr(li_runtime.engine.storage, "last_read_profile", {}),
                "multimodal_ingest": getattr(mm_runtime.engine, "last_write_profile", {}),
                "multimodal_search": mm_search_profile,
                "multimodal_search_optimized": getattr(mm_runtime.engine, "last_search_profile", {}),
                "multimodal_screening": getattr(mm_runtime.engine, "last_screening_profile", {}),
            }

        cold_start_start = time.perf_counter()
        restarted_app = create_app(index_path=str(index_path))
        with TestClient(restarted_app) as restarted_client:
            dense_info = restarted_client.get("/collections/dense-base/info")
            dense_ontology_info = restarted_client.get("/collections/dense-ontology/info")
            li_info = restarted_client.get("/collections/li/info")
            mm_info = restarted_client.get("/collections/mm/info")
            restarted_dense = restarted_client.post(
                "/collections/dense-base/search",
                json={"query_text": primary_query["text"], "top_k": top_k},
            )
            restarted_ontology = restarted_client.post(
                "/collections/dense-ontology/search",
                json={"query_text": primary_query["text"], "top_k": top_k},
            )
        cold_start_ms = (time.perf_counter() - cold_start_start) * 1000.0
        restarted_dense_payload = restarted_dense.json()
        restarted_ontology_payload = restarted_ontology.json()

    results["restart"] = {
        "cold_start_ms": cold_start_ms,
        "dense_info": dense_info.status_code,
        "dense_info_points": dense_info.json()["num_points"],
        "dense_ontology_info": dense_ontology_info.status_code,
        "dense_ontology_info_points": dense_ontology_info.json()["num_points"],
        "li_info": li_info.status_code,
        "mm_info": mm_info.status_code,
        "dense_restart_search": restarted_dense.status_code,
        "dense_restart_total": restarted_dense_payload.get("total", 0),
        "dense_restart_first_id": (
            restarted_dense_payload["results"][0]["id"]
            if restarted_dense_payload.get("results")
            else None
        ),
        "dense_ontology_restart_search": restarted_ontology.status_code,
        "dense_ontology_restart_total": restarted_ontology_payload.get("total", 0),
        "dense_ontology_restart_first_id": (
            restarted_ontology_payload["results"][0]["id"]
            if restarted_ontology_payload.get("results")
            else None
        ),
    }
    path = output_dir / "api_validation.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return lane("passed", path=str(path), summary=results)


def _summarize_multimodal_ordering_variant(
    *,
    mode: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "mode": mode,
            "query_count": 0,
            "avg_rank_metrics": {"mrr": 0.0, "ndcg": 0.0, "recall": 0.0},
            "avg_latency_ms": 0.0,
        }
    return {
        "mode": mode,
        "query_count": len(rows),
        "avg_rank_metrics": _mean_metric([row["rank_metrics"] for row in rows]),
        "avg_latency_ms": float(np.mean([row["latency_ms"] for row in rows])),
        "avg_exact_score_ms": float(np.mean([row["exact_score_ms"] for row in rows])),
        "avg_solver_time_ms": float(np.mean([row["solver_time_ms"] for row in rows])),
        "avg_candidate_pool_size": float(np.mean([row["candidate_pool_size"] for row in rows])),
        "avg_exact_frontier_size": float(np.mean([row["exact_frontier_size"] for row in rows])),
        "avg_final_candidate_count": float(np.mean([row["final_candidate_count"] for row in rows])),
        "avg_total_tokens": float(np.mean([row["total_tokens"] for row in rows])),
        "queries": rows,
    }


def _pick_multimodal_ordering_winner(
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not variants:
        return {"winner": None, "reason": "no_variants"}
    best_ndcg = max(
        float(summary.get("avg_rank_metrics", {}).get("ndcg", 0.0))
        for summary in variants.values()
    )
    eligible: list[tuple[str, dict[str, Any]]] = []
    for mode, summary in variants.items():
        ndcg = float(summary.get("avg_rank_metrics", {}).get("ndcg", 0.0))
        if ndcg >= (best_ndcg - 0.02):
            eligible.append((mode, summary))
    if not eligible:
        eligible = list(variants.items())
    winner_mode, winner_summary = min(
        eligible,
        key=lambda item: (
            float(item[1].get("avg_latency_ms", float("inf"))),
            -float(item[1].get("avg_rank_metrics", {}).get("ndcg", 0.0)),
        ),
    )
    baseline = variants.get("maxsim_only", {})
    return {
        "winner": winner_mode,
        "reason": "lowest_latency_within_quality_band",
        "quality_band_ndcg_delta": 0.02,
        "winner_summary": {
            "avg_latency_ms": winner_summary.get("avg_latency_ms"),
            "avg_rank_metrics": winner_summary.get("avg_rank_metrics"),
        },
        "vs_maxsim_only": {
            "latency_ms_delta": (
                float(winner_summary.get("avg_latency_ms", 0.0))
                - float(baseline.get("avg_latency_ms", 0.0))
            ) if baseline else None,
            "ndcg_delta": (
                float(winner_summary.get("avg_rank_metrics", {}).get("ndcg", 0.0))
                - float(baseline.get("avg_rank_metrics", {}).get("ndcg", 0.0))
            ) if baseline else None,
            "recall_delta": (
                float(winner_summary.get("avg_rank_metrics", {}).get("recall", 0.0))
                - float(baseline.get("avg_rank_metrics", {}).get("recall", 0.0))
            ) if baseline else None,
        },
    }


def run_multimodal_ordering_validation(
    output_dir: Path,
    evaluation_bundle: dict[str, Any],
    native_packages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from fastapi.testclient import TestClient
    from voyager_index._internal.server.main import create_app

    records = evaluation_bundle["records"]
    queries = evaluation_bundle["queries"]
    if not records or not queries:
        return lane("skipped", reason="empty_evaluation_bundle", summary={})

    if native_packages["latence_solver"]["status"] != "installed_and_importable":
        return lane(
            "skipped",
            reason="latence_solver_unavailable",
            summary={"native_package": native_packages["latence_solver"]},
        )

    multivector_points, multivector_tokens = _multivector_points(records)
    external_to_position = {record["id"]: index for index, record in enumerate(records)}
    top_k = min(5, len(records))
    candidate_budget = min(len(records), max(top_k * 16, 64))
    prefilter_k = min(len(records), max(top_k * 6, 24))
    frontier_k = min(len(records), max(top_k * 8, 32))
    mode_specs = {
        "maxsim_only": {
            "multimodal_optimize_mode": "maxsim_only",
        },
        "solver_prefilter_maxsim": {
            "multimodal_optimize_mode": "solver_prefilter_maxsim",
            "multimodal_prefilter_k": prefilter_k,
        },
        "maxsim_then_solver": {
            "multimodal_optimize_mode": "maxsim_then_solver",
            "multimodal_maxsim_frontier_k": frontier_k,
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(index_path=tmpdir)
        with TestClient(app) as client:
            create = client.post(
                "/collections/mm",
                json={"dimension": len(multivector_points[0]["vectors"][0]), "kind": "multimodal"},
            )
            if create.status_code != 200:
                raise RuntimeError(f"Failed to create multimodal collection: {create.text}")
            ingest = client.post("/collections/mm/points", json={"points": multivector_points})
            if ingest.status_code != 200:
                raise RuntimeError(f"Failed to ingest multimodal points: {ingest.text}")

            runtime = client.app.state.search_service.get_collection("mm")
            summaries: dict[str, dict[str, Any]] = {}
            qualitative: dict[str, list[dict[str, Any]]] = {}
            for mode, mode_payload in mode_specs.items():
                rows: list[dict[str, Any]] = []
                for query in queries:
                    response, elapsed_ms = _timed_client_call(
                        lambda payload={
                            "vectors": _pad_query_multivector(query["multivector"], multivector_tokens),
                            "top_k": top_k,
                            "strategy": "optimized",
                            "multimodal_candidate_budget": candidate_budget,
                            **mode_payload,
                        }: client.post("/collections/mm/search", json=payload)
                    )
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Multimodal ordering validation failed for mode={mode}: "
                            f"{response.status_code} {response.text}"
                        )
                    body = response.json()
                    result_ids = [result["id"] for result in body["results"]]
                    ranked_positions = [
                        int(external_to_position[result_id])
                        for result_id in result_ids
                        if result_id in external_to_position
                    ]
                    relevance_map = {
                        int(external_to_position[external_id]): value
                        for external_id, value in query["relevance_map"].items()
                        if external_id in external_to_position
                    }
                    rank_metrics = _ranking_metrics(ranked_positions, relevance_map, k=top_k)
                    profile = dict(runtime.engine.last_search_profile or {})
                    optimization = dict(profile.get("optimization") or {})
                    rows.append(
                        {
                            "query_id": query["query_id"],
                            "text": query["text"],
                            "top_ids": result_ids,
                            "rank_metrics": rank_metrics,
                            "latency_ms": float(elapsed_ms),
                            "exact_score_ms": float(profile.get("score_ms", 0.0) or 0.0),
                            "solver_time_ms": float(optimization.get("solver_time_ms", 0.0) or 0.0),
                            "candidate_pool_size": float(
                                optimization.get(
                                    "candidate_pool_size",
                                    (profile.get("screening") or {}).get("candidate_count", len(records)),
                                ) or 0.0
                            ),
                            "exact_frontier_size": float(
                                optimization.get(
                                    "exact_frontier_size",
                                    len(result_ids),
                                ) or 0.0
                            ),
                            "final_candidate_count": float(
                                optimization.get(
                                    "solver_selected_count",
                                    len(result_ids),
                                ) or 0.0
                            ),
                            "total_tokens": float(body.get("total_tokens") or 0.0),
                            "objective_score": body.get("objective_score"),
                            "screening": profile.get("screening"),
                            "optimization": optimization,
                        }
                    )
                summaries[mode] = _summarize_multimodal_ordering_variant(mode=mode, rows=rows)

            baseline_rows = {
                row["query_id"]: row
                for row in summaries["maxsim_only"]["queries"]
            }
            for mode, summary in summaries.items():
                if mode == "maxsim_only":
                    continue
                deltas = []
                for row in summary["queries"]:
                    baseline_row = baseline_rows.get(row["query_id"])
                    if baseline_row is None:
                        continue
                    deltas.append(
                        {
                            "query_id": row["query_id"],
                            "text": row["text"],
                            "ndcg_delta_vs_maxsim_only": (
                                row["rank_metrics"]["ndcg"] - baseline_row["rank_metrics"]["ndcg"]
                            ),
                            "latency_delta_ms_vs_maxsim_only": row["latency_ms"] - baseline_row["latency_ms"],
                            "top_ids": row["top_ids"],
                            "baseline_top_ids": baseline_row["top_ids"],
                        }
                    )
                deltas.sort(
                    key=lambda item: abs(item["ndcg_delta_vs_maxsim_only"]),
                    reverse=True,
                )
                qualitative[mode] = deltas[:6]

    result = {
        "candidate_budget": candidate_budget,
        "prefilter_k": prefilter_k,
        "frontier_k": frontier_k,
        "variants": summaries,
        "winner": _pick_multimodal_ordering_winner(summaries),
        "qualitative_examples": qualitative,
    }
    path = output_dir / "multimodal_ordering.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return lane("passed", path=str(path), summary=result)


def _synthetic_model_image(output_dir: Path, entities: list[dict[str, Any]]) -> Path:
    from PIL import Image, ImageDraw

    image_path = output_dir / "synthetic-model-input.png"
    first = entities[0] if entities else {"canonical_name": "Voyager Index", "label": "Validation"}
    image = Image.new("RGB", (1200, 800), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 80), f"Entity: {first['canonical_name']}", fill="black")
    draw.text((60, 140), f"Label: {first['label']}", fill="black")
    draw.text((60, 220), "Synthetic fallback image because tmp_data has no non-JSON renderable files.", fill="black")
    image.save(image_path)
    return image_path


def run_model_lane(evaluation_bundle: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    summary = {
        **evaluation_bundle["corpus_summary"],
        **evaluation_bundle["embedding_summary"],
    }
    path = output_dir / "model_lane.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if summary["embedding_source"] == "real_model":
        return lane("passed", path=str(path), summary=summary)
    return lane(
        "skipped",
        reason="real_model_unavailable_using_synthetic_fallback",
        path=str(path),
        summary=summary,
    )


def _rrf_ranked_ids(search_result: dict[str, Any]) -> list[int]:
    fused_scores: dict[int, float] = {}
    for rank, (doc_id, _) in enumerate(search_result.get("dense", []), start=1):
        fused_scores[int(doc_id)] = fused_scores.get(int(doc_id), 0.0) + 1.0 / (60.0 + rank)
    for rank, (doc_id, _) in enumerate(search_result.get("sparse", []), start=1):
        fused_scores[int(doc_id)] = fused_scores.get(int(doc_id), 0.0) + 1.0 / (60.0 + rank)
    return [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)]


def _solver_feature_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    token_counts = np.asarray([record["token_count"] for record in records], dtype=np.uint32)
    feature_pack = _computed_feature_pack(records)
    fact_density = np.asarray(
        [feature["fact_density"] for feature in feature_pack],
        dtype=np.float32,
    )
    centrality = np.asarray(
        [feature["centrality_score"] for feature in feature_pack],
        dtype=np.float32,
    )
    recency = np.asarray(
        [feature["recency_score"] for feature in feature_pack],
        dtype=np.float32,
    )
    auxiliary = np.asarray(
        [feature["auxiliary_score"] for feature in feature_pack],
        dtype=np.float32,
    )
    rhetorical_roles = np.asarray(
        [ROLE_TO_ID.get(str(feature["rhetorical_role"]).lower(), 255) for feature in feature_pack],
        dtype=np.uint8,
    )
    cluster_ids = np.asarray(
        [int(feature["cluster_id"]) for feature in feature_pack],
        dtype=np.int32,
    )
    return {
        "token_counts": token_counts,
        "fact_density": fact_density,
        "centrality": centrality,
        "recency": recency,
        "auxiliary": auxiliary,
        "rhetorical_roles": rhetorical_roles,
        "cluster_ids": cluster_ids,
    }


def _aggregate_variant_rows(query_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not query_rows:
        return {
            "query_count": 0,
            "avg_rank_metrics": {"mrr": 0.0, "ndcg": 0.0, "recall": 0.0},
            "avg_latency_ms": 0.0,
        }
    metrics_rows = [row[key]["rank_metrics"] for row in query_rows]
    latency_rows = [row[key]["elapsed_ms"] for row in query_rows]
    return {
        "query_count": len(query_rows),
        "avg_rank_metrics": _mean_metric(metrics_rows),
        "avg_latency_ms": float(np.mean(latency_rows)),
    }


def _evaluate_dense_retrieval_variant(
    records: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    text_key: str,
    payload_text_key: str | None = None,
    payload_feature_pack: list[dict[str, Any]] | None = None,
    solver_enabled: bool,
) -> dict[str, Any]:
    from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager

    dense_vectors = np.stack([record["dense_vector"] for record in records]).astype(np.float32)
    internal_to_external = {index: record["id"] for index, record in enumerate(records)}
    external_to_internal = {record["id"]: index for index, record in enumerate(records)}
    payloads = [
        {
            **{
                "text": record[payload_text_key or text_key],
                "token_count": record["token_count"],
                "source_name": record["source_name"],
                "document_id": record["document_id"],
                "ontology_terms": record["ontology_terms"],
                "ontology_labels": record["ontology_labels"],
                "ontology_confidences": record.get("ontology_confidences", []),
                "ontology_evidence_counts": record.get("ontology_evidence_counts", []),
                "ontology_match_count": record.get("ontology_match_count", 0),
                "ontology_confidence": record.get("ontology_confidence", 0.0),
                "ontology_concept_density": record.get("ontology_concept_density", 0.0),
                "ontology_relation_density": record.get("ontology_relation_density", 0.0),
            },
            **(payload_feature_pack[index] if payload_feature_pack is not None else {}),
        }
        for index, record in enumerate(records)
    ]
    query_rows = []
    top_k = min(5, len(records))

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = HybridSearchManager(Path(tmpdir), dim=dense_vectors.shape[1], on_disk=False)
        manager.index(
            corpus=[record[text_key] for record in records],
            vectors=dense_vectors,
            ids=list(range(len(records))),
            payloads=payloads,
        )

        try:
            from latence_solver import SolverConstraints  # type: ignore
        except Exception:  # pragma: no cover - depends on optional package
            SolverConstraints = None  # type: ignore[assignment]

        for query in queries:
            internal_relevance_map = {
                external_to_internal[external_id]: value
                for external_id, value in query["relevance_map"].items()
                if external_id in external_to_internal
            }

            start = time.perf_counter()
            bm25 = manager.search(query_text=query["text"], query_vector=None, k=top_k)
            bm25_elapsed_ms = (time.perf_counter() - start) * 1000.0
            bm25_ids = [int(item_id) for item_id, _ in bm25.get("sparse", [])]

            start = time.perf_counter()
            dense = manager.search(query_text="", query_vector=query["dense_vector"], k=top_k)
            dense_elapsed_ms = (time.perf_counter() - start) * 1000.0
            dense_ids = [int(item_id) for item_id, _ in dense.get("dense", [])]

            start = time.perf_counter()
            hybrid = manager.search(query_text=query["text"], query_vector=query["dense_vector"], k=top_k)
            hybrid_elapsed_ms = (time.perf_counter() - start) * 1000.0
            hybrid_ids = _rrf_ranked_ids(hybrid)[:top_k]

            query_row: dict[str, Any] = {
                "query_id": query["query_id"],
                "text": query["text"],
                "label": query["label"],
                "bm25": {
                    "top_ids": [internal_to_external[item_id] for item_id in bm25_ids],
                    "rank_metrics": _ranking_metrics(bm25_ids, internal_relevance_map, k=top_k),
                    "elapsed_ms": bm25_elapsed_ms,
                },
                "dense_only": {
                    "top_ids": [internal_to_external[item_id] for item_id in dense_ids],
                    "rank_metrics": _ranking_metrics(dense_ids, internal_relevance_map, k=top_k),
                    "elapsed_ms": dense_elapsed_ms,
                },
                "hybrid_rrf": {
                    "top_ids": [internal_to_external[item_id] for item_id in hybrid_ids],
                    "rank_metrics": _ranking_metrics(hybrid_ids, internal_relevance_map, k=top_k),
                    "elapsed_ms": hybrid_elapsed_ms,
                },
            }

            if solver_enabled and SolverConstraints is not None and getattr(manager, "solver_available", False):
                start = time.perf_counter()
                refined = manager.refine(
                    query_vector=query["dense_vector"],
                    query_text=query["text"],
                    query_payload={"label": query.get("label"), "ontology_terms": [query["text"]]},
                    candidate_ids=hybrid.get("union_ids", []),
                    constraints=SolverConstraints(
                        max_tokens=max(256, int(np.mean([record["token_count"] for record in records]) * top_k)),
                        max_chunks=top_k,
                    ),
                )
                solver_elapsed_ms = (time.perf_counter() - start) * 1000.0
                solver_ids = [int(item_id) for item_id in refined.get("selected_internal_ids", [])]
                query_row["solver_refined"] = {
                    "top_ids": [internal_to_external[item_id] for item_id in solver_ids],
                    "rank_metrics": _ranking_metrics(solver_ids, internal_relevance_map, k=top_k),
                    "elapsed_ms": solver_elapsed_ms,
                    "objective_score": refined.get("solver_output", {}).get("objective_score"),
                    "total_tokens": refined.get("solver_output", {}).get("total_tokens"),
                    "backend_kind": refined.get("backend_kind"),
                    "feature_summary": refined.get("feature_summary"),
                }
            else:
                query_row["solver_refined"] = {
                    "top_ids": [],
                    "rank_metrics": {"mrr": 0.0, "ndcg": 0.0, "recall": 0.0},
                    "elapsed_ms": 0.0,
                    "objective_score": None,
                    "total_tokens": None,
                    "backend_kind": None,
                    "status": "skipped",
                }

            query_rows.append(query_row)

        if hasattr(manager, "close"):
            manager.close()

    result = {
        "text_key": text_key,
        "feature_pack": "explicit" if payload_feature_pack is not None else "heuristic",
        "query_count": len(query_rows),
        "bm25": _aggregate_variant_rows(query_rows, "bm25"),
        "dense_only": _aggregate_variant_rows(query_rows, "dense_only"),
        "hybrid_rrf": _aggregate_variant_rows(query_rows, "hybrid_rrf"),
        "solver_refined": _aggregate_variant_rows(query_rows, "solver_refined"),
        "queries": query_rows,
    }
    if query_rows:
        result["solver_refined"]["backend_kinds"] = sorted(
            {
                row["solver_refined"].get("backend_kind")
                for row in query_rows
                if row["solver_refined"].get("backend_kind")
            }
        )
    return result


def _compare_dense_variants(
    baseline: dict[str, Any],
    enriched: dict[str, Any],
) -> dict[str, Any]:
    def _delta(path_key: str) -> dict[str, float]:
        before = baseline[path_key]["avg_rank_metrics"]
        after = enriched[path_key]["avg_rank_metrics"]
        return {
            metric: float(after[metric] - before[metric])
            for metric in before
        }

    qualitative = []
    objective_deltas: list[float] = []
    token_deltas: list[float] = []
    for baseline_row, enriched_row in zip(baseline["queries"], enriched["queries"]):
        solver_delta = (
            enriched_row["solver_refined"]["rank_metrics"]["ndcg"]
            - baseline_row["solver_refined"]["rank_metrics"]["ndcg"]
        )
        hybrid_delta = (
            enriched_row["hybrid_rrf"]["rank_metrics"]["ndcg"]
            - baseline_row["hybrid_rrf"]["rank_metrics"]["ndcg"]
        )
        baseline_objective = baseline_row["solver_refined"].get("objective_score")
        enriched_objective = enriched_row["solver_refined"].get("objective_score")
        baseline_tokens = baseline_row["solver_refined"].get("total_tokens")
        enriched_tokens = enriched_row["solver_refined"].get("total_tokens")
        if baseline_objective is not None and enriched_objective is not None:
            objective_deltas.append(float(enriched_objective) - float(baseline_objective))
        if baseline_tokens is not None and enriched_tokens is not None:
            token_deltas.append(float(enriched_tokens) - float(baseline_tokens))
        qualitative.append(
            {
                "query_id": baseline_row["query_id"],
                "text": baseline_row["text"],
                "hybrid_ndcg_delta": float(hybrid_delta),
                "solver_ndcg_delta": float(solver_delta),
                "baseline_top_id": (
                    baseline_row["solver_refined"]["top_ids"][0]
                    if baseline_row["solver_refined"]["top_ids"]
                    else None
                ),
                "enriched_top_id": (
                    enriched_row["solver_refined"]["top_ids"][0]
                    if enriched_row["solver_refined"]["top_ids"]
                    else None
                ),
                "baseline_objective_score": baseline_objective,
                "enriched_objective_score": enriched_objective,
                "baseline_total_tokens": baseline_tokens,
                "enriched_total_tokens": enriched_tokens,
                "baseline_feature_summary": baseline_row["solver_refined"].get("feature_summary"),
                "enriched_feature_summary": enriched_row["solver_refined"].get("feature_summary"),
            }
        )
    qualitative.sort(key=lambda item: abs(item["solver_ndcg_delta"]), reverse=True)

    return {
        "bm25_delta": _delta("bm25"),
        "dense_only_delta": _delta("dense_only"),
        "hybrid_rrf_delta": _delta("hybrid_rrf"),
        "solver_refined_delta": _delta("solver_refined"),
        "solver_avg_latency_delta_ms": float(
            enriched["solver_refined"]["avg_latency_ms"] - baseline["solver_refined"]["avg_latency_ms"]
        ),
        "solver_avg_objective_delta": float(np.mean(objective_deltas)) if objective_deltas else None,
        "solver_avg_total_tokens_delta": float(np.mean(token_deltas)) if token_deltas else None,
        "qualitative_examples": qualitative[:6],
    }


def run_solver_lane(
    output_dir: Path,
    native_packages: dict[str, dict[str, Any]],
    evaluation_bundle: dict[str, Any],
) -> dict[str, Any]:
    package_status = native_packages["latence_solver"]
    results: dict[str, Any] = {"native_package": package_status}
    if package_status["status"] != "installed_and_importable":
        return lane(package_status["status"], reason=package_status.get("reason"), summary=results)

    import latence_solver  # type: ignore
    from latence_solver import SolverConfig, SolverConstraints, TabuSearchSolver  # type: ignore

    records = evaluation_bundle["records"]
    queries = evaluation_bundle["queries"]
    if not records or not queries:
        return lane("skipped", reason="empty_evaluation_bundle", summary=results)

    baseline = _evaluate_dense_retrieval_variant(
        records,
        queries,
        text_key="base_text",
        payload_text_key="base_text",
        solver_enabled=True,
    )
    enriched_feature_pack = _computed_feature_pack(records)
    full_feature_pack = _evaluate_dense_retrieval_variant(
        records,
        queries,
        text_key="base_text",
        payload_text_key="base_text",
        payload_feature_pack=enriched_feature_pack,
        solver_enabled=True,
    )
    comparison = _compare_dense_variants(baseline, full_feature_pack)
    results["backend_status"] = latence_solver.backend_status() if hasattr(latence_solver, "backend_status") else {}
    results["gpu_available"] = bool(latence_solver.gpu_available())
    results["rust_backend_available"] = bool(latence_solver.is_rust_available()) if hasattr(latence_solver, "is_rust_available") else True
    results["baseline"] = baseline
    results["full_feature_pack"] = full_feature_pack
    results["comparison"] = comparison
    results["computed_feature_pack_preview"] = enriched_feature_pack[: min(3, len(enriched_feature_pack))]
    results["bm25"] = baseline["bm25"]
    results["dense_only"] = baseline["dense_only"]
    results["hybrid_rrf"] = baseline["hybrid_rrf"]
    results["solver_refined"] = baseline["solver_refined"]
    results["queries"] = baseline["queries"]

    feature_arrays = _solver_feature_arrays(records)
    dense_vectors = np.stack([record["dense_vector"] for record in records]).astype(np.float32)
    top_k = min(5, len(records))
    primary_query = queries[0]
    external_to_position = {record["id"]: index for index, record in enumerate(records)}
    relevance_map = {
        external_to_position[external_id]: value
        for external_id, value in primary_query["relevance_map"].items()
        if external_id in external_to_position
    }

    def _run_backend_probe(use_gpu: bool) -> tuple[dict[str, Any], Any]:
        solver = TabuSearchSolver(SolverConfig(iterations=30, random_seed=11, use_gpu=use_gpu))
        start = time.perf_counter()
        output = solver.solve_numpy(
            dense_vectors,
            primary_query["dense_vector"],
            feature_arrays["token_counts"],
            feature_arrays["fact_density"],
            feature_arrays["centrality"],
            feature_arrays["recency"],
            feature_arrays["auxiliary"],
            feature_arrays["rhetorical_roles"],
            feature_arrays["cluster_ids"],
            SolverConstraints(
                max_tokens=max(256, int(np.mean(feature_arrays["token_counts"]) * top_k)),
                max_chunks=top_k,
            ),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        selected_indices = [int(index) for index in output.selected_indices]
        return (
            {
                "selected_ids": [records[index]["id"] for index in selected_indices],
                "rank_metrics": _ranking_metrics(selected_indices, relevance_map, k=top_k),
                "objective_score": output.objective_score,
                "num_selected": output.num_selected,
                "solve_time_ms": output.solve_time_ms,
                "measured_elapsed_ms": elapsed_ms,
                "constraints_satisfied": output.constraints_satisfied,
                "backend_kind": solver.backend_kind() if hasattr(solver, "backend_kind") else ("gpu" if use_gpu else "cpu_reference"),
            },
            output,
        )

    backend_probe_cpu, cpu_output = _run_backend_probe(use_gpu=False)
    results["backend_probe_cpu"] = backend_probe_cpu

    if latence_solver.gpu_available():
        backend_probe_gpu, gpu_output = _run_backend_probe(use_gpu=True)
        objective_delta = abs(float(cpu_output.objective_score) - float(gpu_output.objective_score))
        parity_ok = (
            list(cpu_output.selected_indices) == list(gpu_output.selected_indices)
            and cpu_output.constraints_satisfied == gpu_output.constraints_satisfied
            and objective_delta <= 1e-5
        )
        backend_probe_gpu["parity_with_cpu"] = {
            "status": "passed" if parity_ok else "failed",
            "objective_delta": objective_delta,
            "selected_indices_match": list(cpu_output.selected_indices) == list(gpu_output.selected_indices),
        }
        results["accelerated"] = backend_probe_gpu
    else:
        results["accelerated"] = {
            "status": "not_configured",
            "reason": "premium_or_experimental_backend_unavailable",
        }

    path = output_dir / "solver.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    overall_ok = results["solver_refined"]["query_count"] > 0 and results["accelerated"].get("parity_with_cpu", {}).get("status", "passed") == "passed"
    return lane("passed" if overall_ok else "failed", path=str(path), summary=results)


def run_ontology_variant_evaluation(
    output_dir: Path,
    native_packages: dict[str, dict[str, Any]],
    evaluation_bundle: dict[str, Any],
) -> dict[str, Any]:
    records = evaluation_bundle["records"]
    queries = [query for query in evaluation_bundle["queries"] if query.get("source") == "ontology_entity"]
    if not records or not queries or not any(record["ontology_terms"] for record in records):
        return lane("skipped", reason="no_ontology_grounded_queries", summary={})

    solver_enabled = native_packages["latence_solver"]["status"] == "installed_and_importable"
    without_ontology = _evaluate_dense_retrieval_variant(records, queries, text_key="base_text", payload_text_key="base_text", solver_enabled=solver_enabled)
    with_ontology = _evaluate_dense_retrieval_variant(records, queries, text_key="enriched_text", payload_text_key="base_text", solver_enabled=solver_enabled)

    def _delta(path_key: str) -> dict[str, float]:
        before = without_ontology[path_key]["avg_rank_metrics"]
        after = with_ontology[path_key]["avg_rank_metrics"]
        return {
            metric: float(after[metric] - before[metric])
            for metric in before
        }

    qualitative = []
    for before_row, after_row in zip(without_ontology["queries"], with_ontology["queries"]):
        delta = after_row["hybrid_rrf"]["rank_metrics"]["ndcg"] - before_row["hybrid_rrf"]["rank_metrics"]["ndcg"]
        solver_delta = (
            after_row["solver_refined"]["rank_metrics"]["ndcg"]
            - before_row["solver_refined"]["rank_metrics"]["ndcg"]
        )
        qualitative.append(
            {
                "query_id": before_row["query_id"],
                "text": before_row["text"],
                "hybrid_ndcg_delta": float(delta),
                "solver_ndcg_delta": float(solver_delta),
                "without_top_id": before_row["hybrid_rrf"]["top_ids"][0] if before_row["hybrid_rrf"]["top_ids"] else None,
                "with_top_id": after_row["hybrid_rrf"]["top_ids"][0] if after_row["hybrid_rrf"]["top_ids"] else None,
                "solver_feature_summary": after_row["solver_refined"].get("feature_summary"),
            }
        )
    qualitative.sort(key=lambda item: abs(item["hybrid_ndcg_delta"]), reverse=True)

    results = {
        "without_ontology": without_ontology,
        "with_ontology": with_ontology,
        "hybrid_rrf_delta": _delta("hybrid_rrf"),
        "solver_refined_delta": _delta("solver_refined"),
        "qualitative_examples": qualitative[:6],
    }
    path = output_dir / "ontology_variant.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return lane("passed", path=str(path), summary=results)


def run_native_hnsw_lane(output_dir: Path, native_packages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    package_status = native_packages["latence_hnsw"]
    summary: dict[str, Any] = {"native_package": package_status}
    if package_status["status"] != "installed_and_importable":
        return lane(package_status["status"], reason=package_status.get("reason"), summary=summary)

    verify_script = NATIVE_PACKAGE_SPECS["latence_hnsw"]["verify_script"]
    if verify_script is not None and Path(verify_script).exists():
        summary["verify_install"] = _run_logged_script(
            [sys.executable, str(verify_script)],
            output_dir,
            "latence_hnsw_verify_install",
        )
    else:
        summary["verify_install"] = {"status": "skipped", "reason": "verify_script_missing"}

    native_probe = _run_hnsw_probe(force_fallback=False, output_dir=output_dir, log_name="latence_hnsw_native_probe")
    fallback_probe = _run_hnsw_probe(force_fallback=True, output_dir=output_dir, log_name="latence_hnsw_fallback_probe")
    summary["native_probe"] = native_probe
    summary["fallback_probe"] = fallback_probe

    native_summary = native_probe.get("summary", {})
    fallback_summary = fallback_probe.get("summary", {})
    summary["top_id_match"] = native_summary.get("top_ids") == fallback_summary.get("top_ids")
    summary["filtered_match"] = native_summary.get("filtered_top_ids") == fallback_summary.get("filtered_top_ids")
    summary["reopen_match"] = native_summary.get("reopened_top_ids") == fallback_summary.get("reopened_top_ids")
    summary["native_elapsed_ms"] = native_summary.get("elapsed_ms")
    summary["fallback_elapsed_ms"] = fallback_summary.get("elapsed_ms")

    path = output_dir / "native_hnsw.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    overall_ok = (
        summary["verify_install"].get("status") == "passed"
        and native_probe.get("status") == "passed"
        and fallback_probe.get("status") == "passed"
        and summary["top_id_match"]
        and summary["filtered_match"]
        and summary["reopen_match"]
    )
    return lane("passed" if overall_ok else "failed", path=str(path), summary=summary)


def build_value_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    maxsim = report["lanes"].get("maxsim", {}).get("summary", {})
    if maxsim:
        cpu_ms = float(maxsim.get("fast_colbert_cpu", {}).get("elapsed_ms") or 0.0)
        triton_ms = float(maxsim.get("triton_cuda", {}).get("elapsed_ms") or 0.0)
        if cpu_ms > 0.0 and maxsim.get("triton_cuda", {}).get("status") == "passed":
            summary["maxsim"] = {
                "cpu_ms": cpu_ms,
                "triton_ms": triton_ms,
                "triton_speedup_vs_cpu": (cpu_ms / triton_ms) if triton_ms > 0.0 else None,
                "parity": maxsim.get("triton_cuda", {}).get("parity"),
            }

    quant = report["lanes"].get("quantization", {}).get("summary", {})
    if quant:
        summary["quantization"] = {
            "roq_1bit_recall_at_k": quant.get("roq_1bit", {}).get("recall_at_k"),
            "roq_1bit_screening_recall_at_k": quant.get("roq_1bit_screening", {}).get("recall_at_k"),
            "roq_1bit_screening_speedup_vs_full_precision": quant.get("roq_1bit_screening", {}).get("speedup_vs_full_precision"),
            "roq_1bit_screening_top_1_agreement": quant.get("roq_1bit_screening", {}).get("top_1_agreement"),
            "roq_1bit_screening_top1_retention_at_128": next(
                (
                    row.get("top1_retention")
                    for row in quant.get("roq_1bit_screening", {}).get("candidate_pool_retention", [])
                    if row.get("candidate_count") == 128
                ),
                None,
            ),
            "roq_2bit_screening_recall_at_k": quant.get("roq_2bit_screening", {}).get("recall_at_k"),
            "roq_2bit_screening_speedup_vs_full_precision": quant.get("roq_2bit_screening", {}).get("speedup_vs_full_precision"),
            "roq_2bit_screening_top_1_agreement": quant.get("roq_2bit_screening", {}).get("top_1_agreement"),
            "pooled_dense_screening_recall_at_k": quant.get("pooled_dense_screening", {}).get("recall_at_k"),
            "pooled_dense_screening_speedup_vs_full_precision": quant.get("pooled_dense_screening", {}).get("speedup_vs_full_precision"),
            "prototype_sidecar_screening_recall_at_k": quant.get("prototype_sidecar_screening", {}).get("recall_at_k"),
            "prototype_sidecar_screening_speedup_vs_full_precision": quant.get("prototype_sidecar_screening", {}).get("speedup_vs_full_precision"),
            "prototype_sidecar_full_recovery_candidate_count": quant.get("prototype_sidecar_screening", {}).get("full_recovery_candidate_count"),
            "prototype_sidecar_avg_prototypes_per_doc": quant.get("prototype_sidecar_screening", {}).get("avg_prototypes_per_doc"),
            "prototype_sidecar_top1_retention_at_128": next(
                (
                    row.get("top1_retention")
                    for row in quant.get("prototype_sidecar_screening", {}).get("candidate_pool_retention", [])
                    if row.get("candidate_count") == 128
                ),
                None,
            ),
            "prototype_sidecar_topk_retention_at_128": next(
                (
                    row.get("topk_retention")
                    for row in quant.get("prototype_sidecar_screening", {}).get("candidate_pool_retention", [])
                    if row.get("candidate_count") == 128
                ),
                None,
            ),
            "prototype_sidecar_fallback_rate": quant.get("prototype_sidecar_screening", {}).get("fallback_rate"),
            "prototype_sidecar_direct_gather_rate": quant.get("prototype_sidecar_screening", {}).get("direct_gather_rate"),
            "prototype_sidecar_bootstrap_calibration_passed": quant.get("prototype_sidecar_screening", {}).get("bootstrap_calibration_passed"),
            "centroid_screening_recall_at_k": quant.get("centroid_screening", {}).get("recall_at_k"),
            "centroid_screening_speedup_vs_full_precision": quant.get("centroid_screening", {}).get("speedup_vs_full_precision"),
            "centroid_screening_top_1_agreement": quant.get("centroid_screening", {}).get("top_1_agreement"),
            "centroid_screening_full_recovery_candidate_count": quant.get("centroid_screening", {}).get("full_recovery_candidate_count"),
            "centroid_screening_top1_retention_at_128": next(
                (
                    row.get("top1_retention")
                    for row in quant.get("centroid_screening", {}).get("candidate_pool_retention", [])
                    if row.get("candidate_count") == 128
                ),
                None,
            ),
            "centroid_screening_topk_retention_at_128": next(
                (
                    row.get("topk_retention")
                    for row in quant.get("centroid_screening", {}).get("candidate_pool_retention", [])
                    if row.get("candidate_count") == 128
                ),
                None,
            ),
            "centroid_screening_fallback_rate": quant.get("centroid_screening", {}).get("fallback_rate"),
            "centroid_screening_direct_gather_rate": quant.get("centroid_screening", {}).get("direct_gather_rate"),
            "centroid_screening_bootstrap_calibration_passed": quant.get("centroid_screening", {}).get("bootstrap_calibration_passed"),
            "centroid_screening_scale_speedup": quant.get("centroid_screening_scale_harness", {}).get("speedup_vs_full_precision"),
            "centroid_screening_promotion_recommendation": quant.get("centroid_screening_decision_gate", {}).get("promotion_recommendation"),
            "roq_4bit_recall_at_k": quant.get("roq_4bit", {}).get("recall_at_k"),
            "roq_4bit_score_correlation": quant.get("roq_4bit", {}).get("score_correlation"),
            "roq_4bit_symmetric_recall_at_k": quant.get("roq_4bit_symmetric", {}).get("recall_at_k"),
            "roq_4bit_top_1_agreement": quant.get("roq_4bit", {}).get("top_1_agreement"),
            "roq_4bit_ndcg": quant.get("roq_4bit", {}).get("rank_metrics", {}).get("ndcg"),
            "roq_4bit_ndcg_delta_vs_full_precision": quant.get("roq_4bit", {}).get("ndcg_delta_vs_full_precision"),
            "roq_4bit_speedup_vs_full_precision": quant.get("roq_4bit", {}).get("speedup_vs_full_precision"),
            "roq_4bit_compression_vs_fp16": quant.get("roq_4bit", {}).get("compression_vs_fp16"),
            "roq_4bit_crossover_candidate_count": quant.get("roq_4bit", {}).get("candidate_sweep", {}).get("crossover_candidate_count"),
            "roq_4bit_best_sweep_speedup": max(
                (
                    row.get("speedup_vs_full_precision")
                    for row in quant.get("roq_4bit", {}).get("candidate_sweep", {}).get("counts", [])
                    if row.get("speedup_vs_full_precision") is not None
                ),
                default=None,
            ),
            "scalar_int4_baseline_recall_at_k": quant.get("scalar_int4_baseline", {}).get("recall_at_k"),
            "multimodal_int8_recall_at_k": quant.get("multimodal_int8", {}).get("recall_at_k"),
            "multimodal_int8_speedup_vs_full_precision": (
                (quant.get("multimodal_int8", {}).get("elapsed_ms_full_precision") or 0.0)
                / max(quant.get("multimodal_int8", {}).get("elapsed_ms_quantized") or 1e-8, 1e-8)
            ) if quant.get("multimodal_int8") else None,
        }

    solver = report["lanes"].get("solver", {}).get("summary", {})
    if solver:
        bm25_ndcg = solver.get("bm25", {}).get("avg_rank_metrics", {}).get("ndcg")
        dense_ndcg = solver.get("dense_only", {}).get("avg_rank_metrics", {}).get("ndcg")
        hybrid_ndcg = solver.get("hybrid_rrf", {}).get("avg_rank_metrics", {}).get("ndcg")
        solver_ndcg = solver.get("solver_refined", {}).get("avg_rank_metrics", {}).get("ndcg")
        summary["solver"] = {
            "bm25_ndcg": bm25_ndcg,
            "dense_ndcg": dense_ndcg,
            "hybrid_rrf_ndcg": hybrid_ndcg,
            "solver_ndcg": solver_ndcg,
            "solver_minus_rrf_ndcg": (solver_ndcg - hybrid_ndcg) if solver_ndcg is not None and hybrid_ndcg is not None else None,
            "solver_minus_bm25_ndcg": (solver_ndcg - bm25_ndcg) if solver_ndcg is not None and bm25_ndcg is not None else None,
            "solver_latency_ms": solver.get("solver_refined", {}).get("avg_latency_ms"),
        }
        summary["solver_enrichment"] = {
            "solver_ndcg_delta": solver.get("comparison", {}).get("solver_refined_delta", {}).get("ndcg"),
            "hybrid_rrf_ndcg_delta": solver.get("comparison", {}).get("hybrid_rrf_delta", {}).get("ndcg"),
            "solver_latency_delta_ms": solver.get("comparison", {}).get("solver_avg_latency_delta_ms"),
            "solver_objective_delta": solver.get("comparison", {}).get("solver_avg_objective_delta"),
            "solver_total_tokens_delta": solver.get("comparison", {}).get("solver_avg_total_tokens_delta"),
        }

    ontology = report["lanes"].get("ontology_variant", {}).get("summary", {})
    if ontology:
        summary["ontology"] = {
            "hybrid_rrf_ndcg_delta": ontology.get("hybrid_rrf_delta", {}).get("ndcg"),
            "solver_ndcg_delta": ontology.get("solver_refined_delta", {}).get("ndcg"),
        }

    api = report["lanes"].get("api_crud", {}).get("summary", {})
    if api:
        summary["api"] = {
            "dense_vector_ms": api.get("search", {}).get("dense_vector_ms"),
            "dense_text_ms": api.get("search", {}).get("dense_text_ms"),
            "dense_hybrid_ms": api.get("search", {}).get("dense_hybrid_ms"),
            "dense_optimized_ms": api.get("search", {}).get("dense_optimized_ms"),
            "late_interaction_ms": api.get("search", {}).get("late_interaction_ms"),
            "late_interaction_score_ms": api.get("profiles", {}).get("late_interaction_search", {}).get("score_ms"),
            "late_interaction_vector_load_ms": api.get("profiles", {}).get("late_interaction_vector_load", {}).get("elapsed_ms"),
            "multimodal_ms": api.get("search", {}).get("multimodal_ms"),
            "multimodal_score_ms": api.get("profiles", {}).get("multimodal_search", {}).get("score_ms"),
            "multimodal_load_ms": api.get("profiles", {}).get("multimodal_search", {}).get("load_ms"),
            "multimodal_optimized_ms": api.get("search", {}).get("multimodal_optimized_ms"),
            "multimodal_optimized_score_ms": api.get("profiles", {}).get("multimodal_search_optimized", {}).get("score_ms"),
            "multimodal_screening_candidate_count": api.get("profiles", {}).get("multimodal_screening", {}).get("candidate_count"),
            "multimodal_screening_elapsed_ms": api.get("profiles", {}).get("multimodal_screening", {}).get("elapsed_ms"),
            "restart_cold_start_ms": api.get("restart", {}).get("cold_start_ms"),
        }

    multimodal_ordering = report["lanes"].get("multimodal_ordering", {}).get("summary", {})
    if multimodal_ordering:
        winner = multimodal_ordering.get("winner", {})
        summary["multimodal_ordering"] = {
            "winner": winner.get("winner"),
            "winner_reason": winner.get("reason"),
            "winner_latency_delta_vs_maxsim_only": winner.get("vs_maxsim_only", {}).get("latency_ms_delta"),
            "winner_ndcg_delta_vs_maxsim_only": winner.get("vs_maxsim_only", {}).get("ndcg_delta"),
            "winner_recall_delta_vs_maxsim_only": winner.get("vs_maxsim_only", {}).get("recall_delta"),
            "maxsim_only_ndcg": multimodal_ordering.get("variants", {}).get("maxsim_only", {}).get("avg_rank_metrics", {}).get("ndcg"),
            "solver_prefilter_ndcg": multimodal_ordering.get("variants", {}).get("solver_prefilter_maxsim", {}).get("avg_rank_metrics", {}).get("ndcg"),
            "maxsim_then_solver_ndcg": multimodal_ordering.get("variants", {}).get("maxsim_then_solver", {}).get("avg_rank_metrics", {}).get("ndcg"),
            "maxsim_only_latency_ms": multimodal_ordering.get("variants", {}).get("maxsim_only", {}).get("avg_latency_ms"),
            "solver_prefilter_latency_ms": multimodal_ordering.get("variants", {}).get("solver_prefilter_maxsim", {}).get("avg_latency_ms"),
            "maxsim_then_solver_latency_ms": multimodal_ordering.get("variants", {}).get("maxsim_then_solver", {}).get("avg_latency_ms"),
        }

    return summary


def run_gem_router_lane(output_dir: Path) -> dict[str, Any]:
    """Validate the Rust GEM router with synthetic data."""
    lane: dict[str, Any] = {"status": "skipped", "checks": []}
    try:
        from latence_gem_router import PyGemRouter
    except ImportError:
        lane["reason"] = "latence_gem_router not installed"
        return lane

    rng = np.random.default_rng(42)
    dim = 128
    n_docs = 200
    doc_tokens = 32
    n_queries = 20

    doc_embeddings = rng.standard_normal((n_docs, doc_tokens, dim)).astype(np.float32)
    for i in range(n_docs):
        norms = np.linalg.norm(doc_embeddings[i], axis=1, keepdims=True) + 1e-8
        doc_embeddings[i] /= norms

    all_vecs = doc_embeddings.reshape(-1, dim)
    offsets = [(i * doc_tokens, (i + 1) * doc_tokens) for i in range(n_docs)]
    doc_ids = list(range(n_docs))

    build_start = time.perf_counter()
    router = PyGemRouter(dim=dim)
    router.build(all_vecs, doc_ids, offsets, n_fine=64, n_coarse=16)
    build_ms = (time.perf_counter() - build_start) * 1000.0
    lane["checks"].append({"name": "build", "build_ms": build_ms, "n_fine": router.n_fine(), "n_coarse": router.n_coarse()})

    query_embeddings = rng.standard_normal((n_queries, 16, dim)).astype(np.float32)
    for q in range(n_queries):
        norms = np.linalg.norm(query_embeddings[q], axis=1, keepdims=True) + 1e-8
        query_embeddings[q] /= norms

    route_latencies = []
    candidate_counts = []
    for q in range(n_queries):
        q_vecs = query_embeddings[q]
        start = time.perf_counter()
        results = router.route_query(q_vecs, n_probes=4, max_candidates=100)
        route_latencies.append((time.perf_counter() - start) * 1000.0)
        candidate_counts.append(len(results))

    avg_latency = sum(route_latencies) / len(route_latencies) if route_latencies else 0
    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0

    lane["checks"].append({
        "name": "route_query",
        "n_queries": n_queries,
        "avg_latency_ms": avg_latency,
        "avg_candidates": avg_candidates,
    })

    # Save/load roundtrip
    save_path = output_dir / "gem_router_test.gemr"
    router.save(str(save_path))
    router2 = PyGemRouter(dim=dim)
    router2.load(str(save_path))
    lane["checks"].append({
        "name": "save_load_roundtrip",
        "n_docs_original": router.n_docs(),
        "n_docs_loaded": router2.n_docs(),
        "match": router.n_docs() == router2.n_docs(),
    })

    # Cluster entries
    entries = router.get_cluster_entries(query_embeddings[0], n_probes=4)
    lane["checks"].append({
        "name": "cluster_entries",
        "n_entries": len(entries),
        "entries": [int(e) for e in entries[:8]],
    })

    all_passed = (
        router.is_ready()
        and avg_candidates > 0
        and router.n_docs() == router2.n_docs()
    )
    lane["status"] = "passed" if all_passed else "failed"
    lane["reason"] = None if all_passed else "one or more checks failed"
    return lane


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full-feature validation matrix.")
    parser.add_argument("--tmp-data-dir", default="tmp_data")
    parser.add_argument("--ontology-fixture")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--native-bootstrap",
        choices=("auto", "off"),
        default="auto",
        help="Attempt to build repo-local native packages before running native lanes.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tmp_data_dir = Path(args.tmp_data_dir).resolve()
    ontology_fixture = resolve_ontology_fixture(tmp_data_dir, Path(args.ontology_fixture) if args.ontology_fixture else None)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else repo_root / "validation-reports" / f"full-feature-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = output_dir / "rendered-corpus"
    native_packages = bootstrap_native_packages(output_dir, bootstrap=args.native_bootstrap == "auto")

    corpus_inventory = enumerate_source_documents(tmp_data_dir, ontology_fixture)
    rendered_manifest = render_documents(corpus_inventory["documents"], rendered_dir) if corpus_inventory["documents"] else lane("skipped", reason="no_non_json_tmp_data_documents", rendered=[], skipped=corpus_inventory["skipped"])
    entities = load_ontology_entities(ontology_fixture, limit=1024)
    evaluation_bundle = prepare_evaluation_bundle(rendered_manifest.get("rendered", []), entities, output_dir)

    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "model_request": {
            "package": "sauerkrautlm-colpali",
            "model_id": MODEL_ID,
            "requested_device": "cuda:0",
        },
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "rust_toolchain": {
                "cargo_available": shutil.which("cargo") is not None,
                "rustc_available": shutil.which("rustc") is not None,
            },
            "packages": {
                "sauerkrautlm-colpali": safe_version("sauerkrautlm-colpali"),
                "httpx": safe_version("httpx"),
                "latence_solver": safe_version(NATIVE_PACKAGE_SPECS["latence_solver"]["distribution"]),
            },
        },
        "inputs": {
            "tmp_data_dir": str(tmp_data_dir),
            "ontology_fixture": str(ontology_fixture),
            "corpus_documents": [str(path) for path in corpus_inventory["documents"]],
            "corpus_skipped": corpus_inventory["skipped"],
        },
        "native_packages": native_packages,
        "corpus_rendering": rendered_manifest,
        "evaluation_corpus": {
            "records": evaluation_bundle["corpus_summary"]["record_count"],
            "documents": evaluation_bundle["corpus_summary"]["documents"],
            "queries": evaluation_bundle["corpus_summary"]["query_count"],
            "records_with_ontology_matches": evaluation_bundle["corpus_summary"]["records_with_ontology_matches"],
            "embedding_source": evaluation_bundle["embedding_summary"]["embedding_source"],
        },
        "lanes": {},
    }

    report["lanes"]["benchmark_smoke"] = run_oss_benchmark_bundle(output_dir)
    report["lanes"]["quantization"] = run_quantization_validation(output_dir, evaluation_bundle)
    report["lanes"]["maxsim"] = run_maxsim_validation(output_dir, evaluation_bundle)
    report["lanes"]["hybrid_storage"] = run_hybrid_storage_validation(output_dir, entities[:8])
    report["lanes"]["api_crud"] = run_api_validation(output_dir, evaluation_bundle, native_packages)
    report["lanes"]["multimodal_ordering"] = run_multimodal_ordering_validation(
        output_dir,
        evaluation_bundle,
        native_packages,
    )
    report["lanes"]["model"] = run_model_lane(evaluation_bundle, output_dir)
    report["lanes"]["solver"] = run_solver_lane(output_dir, native_packages, evaluation_bundle)
    report["lanes"]["ontology_variant"] = run_ontology_variant_evaluation(output_dir, native_packages, evaluation_bundle)

    blockers = []
    if not corpus_inventory["documents"]:
        blockers.append("tmp_data contains no non-JSON source documents to render")
    if not torch.cuda.is_available():
        blockers.append("CUDA is unavailable in this environment, so the requested ColLFM2 cuda:0 lane and Triton GPU lanes cannot run")
    if report["lanes"]["model"]["status"] != "passed":
        blockers.append(
            f"model lane status={report['lanes']['model']['status']} reason={report['lanes']['model'].get('reason')}"
        )
    if report["lanes"]["solver"]["status"] != "passed":
        blockers.append(
            f"latence_solver lane status={report['lanes']['solver']['status']} reason={report['lanes']['solver'].get('reason')}"
        )
    if report["lanes"]["ontology_variant"]["status"] != "passed":
        blockers.append(
            f"ontology lane status={report['lanes']['ontology_variant']['status']} reason={report['lanes']['ontology_variant'].get('reason')}"
        )
    report["blockers"] = blockers
    report["value_summary"] = build_value_summary(report)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
