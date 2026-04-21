from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from colsearch.server import create_app
from examples.reference_api_feature_tour import run_feature_tour, write_report
from examples.reference_api_happy_path import run_happy_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_curated_top_level_layout_hides_root_clutter() -> None:
    unexpected_root_entries = {
        "benchmark_colbert_mps.py",
        "benchmark_comparison.py",
        "benchmark_hnsw_retrieval.py",
        "benchmark_m1.py",
        "benchmark_pipeline.py",
        "distributed_benchmark.py",
        "check_data.py",
        "decompress_zst.py",
        "deploy.py",
        "diag_pilot.py",
        "inspect_zst.py",
        "test_model.py",
        "util_inspect_data.py",
        "verify_chunking.py",
        "verify_enriched_mode.py",
        "verify_enrichment_qualitative.py",
        "verify_enterprise_features.py",
        "verify_hybrid_fusion.py",
        "verify_intelligence_offline.py",
        "verify_intelligence_real.py",
        "verify_knapsack_auxiliary.py",
        "verify_multidense.py",
        "verify_retrieval.py",
        "verify_roq_integration.py",
        "verify_search_pipeline.py",
        "ADAPTER_CONTRACTS.md",
        "HARDENING_PLAN_V2.md",
        "MULTIMODAL_FOUNDATION.md",
        "OSS_FOUNDATION.md",
        "QDRANT_VENDORING.md",
        "RECOVERY_REFERENCE.json",
        "ROADMAP.md",
        "SCREENING_PROMOTION_DECISION_MEMO.md",
        "gem_native_benchmark_results.json",
        "gem_quality_benchmark_results.json",
        "verify_recovery.py",
        "sandbox",
        "validation-centroid",
        "validation-centroid-targeted",
        "validation-screening-audit",
        "validation-sidecar",
        "validation-sidecar-slice",
    }
    for entry in unexpected_root_entries:
        assert not (REPO_ROOT / entry).exists(), entry

    expected_release_paths = {
        "docs/getting-started/quickstart.md",
        "docs/api/python.md",
        "docs/guides/scaling.md",
        "README.md",
        "BENCHMARKS.md",
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "PRODUCTION.md",
        "SECURITY.md",
        "internal/README.md",
    }
    for entry in expected_release_paths:
        assert (REPO_ROOT / entry).exists(), entry


def test_readme_routes_users_to_polished_release_entrypoints() -> None:
    payload = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/getting-started/quickstart.md" in payload or "Quickstart" in payload
    assert "docs/api/python.md" in payload or "API Reference" in payload
    assert "docs/guides/scaling.md" in payload or "Scaling Guide" in payload
    assert "Tabu Search" in payload
    assert "http://127.0.0.1:8080/docs" in payload
    assert "multi-vector native" in payload.lower()


def test_public_examples_and_benchmark_do_not_reach_into_internal_modules() -> None:
    public_files = [
        REPO_ROOT / "examples" / "reference_api_feature_tour.py",
        REPO_ROOT / "examples" / "reference_api_happy_path.py",
        REPO_ROOT / "examples" / "reference_api_late_interaction.py",
        REPO_ROOT / "examples" / "reference_api_multimodal.py",
        REPO_ROOT / "examples" / "vllm_pooling_provider.py",
        REPO_ROOT / "benchmarks" / "oss_reference_benchmark.py",
    ]
    for path in public_files:
        payload = path.read_text(encoding="utf-8")
        assert "colsearch._internal" not in payload, path


def test_docs_quickstart_and_api_ref_exist() -> None:
    quickstart = REPO_ROOT / "docs" / "getting-started" / "quickstart.md"
    api_ref = REPO_ROOT / "docs" / "api" / "python.md"
    assert quickstart.exists(), "quickstart.md missing"
    assert api_ref.exists(), "python.md API ref missing"
    qs_text = quickstart.read_text(encoding="utf-8")
    assert "colsearch" in qs_text
    assert "Index" in qs_text


def test_reference_api_happy_path_example_smoke(tmp_path: Path) -> None:
    with TestClient(create_app(index_path=str(tmp_path))) as client:
        summary = run_happy_path(client, prefix="smoke")

    assert "smoke-dense" in summary["collections"]
    assert summary["dense_search"]["results"][0]["id"] == "invoice"
    assert summary["late_interaction_search"]["results"][0]["id"] == "li-1"
    assert summary["multimodal_search"]["results"][0]["id"] == "page-1"


def test_reference_api_feature_tour_smoke_and_report(tmp_path: Path) -> None:
    with TestClient(create_app(index_path=str(tmp_path / "index"))) as client:
        report = run_feature_tour(client, prefix="tour", base_url="http://testserver")

    assert report["summary"]["failed"] == 0
    assert report["summary"]["status"] in {"passed", "passed_with_skips"}
    assert report["searches"]["dense_vector"]["top_id"] == "doc-1"
    assert report["searches"]["dense_bm25"]["top_id"] in {"doc-1", "doc-2"}
    assert report["searches"]["dense_bm25"]["total"] >= 1
    assert report["searches"]["late_interaction"]["top_id"] == "li-1"
    assert report["searches"]["multimodal_exact"]["top_id"] == "page-1"
    assert report["checks"]["reference_optimize_health"]["available"] in {True, False}
    assert "execution_mode" in report["checks"]["reference_optimize_health"]

    output_path = write_report(report, tmp_path / "feature-tour-report.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["failed"] == 0
    assert any(step["name"] == "dense_optimized_search" for step in payload["steps"])
