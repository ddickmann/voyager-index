from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_BENCH = _REPO / "tools" / "benchmarks" / "benchmark_latence_graph_quality.py"
if _BENCH.is_file():
    spec = importlib.util.spec_from_file_location("bench_latence_graph_quality", _BENCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_latence_graph_quality"] = mod
    spec.loader.exec_module(mod)
else:  # pragma: no cover
    mod = None


@pytest.mark.skipif(mod is None, reason="graph quality benchmark missing")
def test_graph_quality_benchmark_improves_graph_shaped_queries(tmp_path: Path) -> None:
    report = mod.run_graph_quality_benchmark(tmp_path / "quality")
    representative = report["representative"]

    assert representative["comparison"]["graph_shaped"]["recall"] > 0.0
    assert representative["comparison"]["graph_shaped"]["ndcg"] > 0.0
    assert representative["comparison"]["graph_shaped"]["support_coverage"] > 0.0
    assert representative["graph"]["route_checks"]["all_passed"] is True


@pytest.mark.skipif(mod is None, reason="graph quality benchmark missing")
def test_graph_quality_benchmark_preserves_ordinary_queries(tmp_path: Path) -> None:
    report = mod.run_graph_quality_benchmark(tmp_path / "quality")
    representative = report["representative"]

    assert representative["comparison"]["ordinary"]["recall"] >= -0.01
    assert representative["comparison"]["ordinary"]["ndcg"] >= -0.01
    assert representative["comparison"]["ordinary"]["support_coverage"] >= -0.01
    assert representative["graph"]["graph_applied_rate"] < 1.0
    assert representative["graph"]["latency"]["overall"]["p95_ms"] >= 0.0


@pytest.mark.skipif(mod is None, reason="graph quality benchmark missing")
def test_graph_quality_ablation_separates_local_and_community_uplift(tmp_path: Path) -> None:
    report = mod.run_graph_quality_ablation(tmp_path / "ablation")

    local_only_queries = {item["query_id"]: item for item in report["local_only"]["synthetic"]["graph"]["queries"]}
    community_only_queries = {item["query_id"]: item for item in report["community_only"]["synthetic"]["graph"]["queries"]}
    full_queries = {item["query_id"]: item for item in report["full"]["synthetic"]["graph"]["queries"]}

    assert 2 in local_only_queries["local_chain"]["full_ranked_ids"]
    assert 4 not in local_only_queries["community_support"]["full_ranked_ids"]
    assert 4 in community_only_queries["community_support"]["full_ranked_ids"]
    assert 4 in full_queries["community_support"]["full_ranked_ids"]
