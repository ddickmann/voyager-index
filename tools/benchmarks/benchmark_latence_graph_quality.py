"""Quality benchmark for the optional Latence graph lane."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from voyager_index._internal.inference.search_pipeline import SearchPipeline

_REPRESENTATIVE_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "latence_graph_quality_eval.json"


@dataclass
class GraphQualityQuery:
    query_id: str
    subset: str
    query_input: Any
    query_text: str
    query_payload: Dict[str, Any]
    gold_ids: List[int]
    top_k: int = 1
    expected_graph_auto: Optional[bool] = None
    expected_lane: Optional[str] = None


@dataclass
class GraphQualityFixture:
    name: str
    description: str
    corpus: List[str]
    vectors: np.ndarray
    ids: List[int]
    payloads: List[Dict[str, Any]]
    queries: List[GraphQualityQuery]


def _ranking_metrics(ranked_ids: List[int], gold_ids: List[int], *, k: int = 2) -> Dict[str, float]:
    ranked = list(ranked_ids[:k])
    gold = list(gold_ids)
    if not gold:
        return {"recall": 0.0, "ndcg": 0.0, "support_coverage": 0.0}
    recall = len(set(ranked) & set(gold)) / float(len(set(gold)))
    dcg = 0.0
    for idx, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            dcg += 1.0 / np.log2(idx + 1.0)
    ideal = sum(1.0 / np.log2(idx + 1.0) for idx in range(1, min(len(gold), k) + 1))
    ndcg = dcg / ideal if ideal > 0.0 else 0.0
    return {
        "recall": float(recall),
        "ndcg": float(ndcg),
        "support_coverage": float(recall),
    }


def _synthetic_regression_fixture() -> GraphQualityFixture:
    corpus = [
        "Team Alpha owns Service B and manages escalation.",
        "Service B causes Service C outage during dependency failures.",
        "Service C reliability overview and operating posture.",
        "Reliability playbook for Service C recovery and incident support.",
        "Invoice operations manual and approval workflow.",
        "General FAQ for employee navigation.",
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.05, 0.95],
            [0.85, 0.15],
            [0.10, 0.88],
            [0.25, 0.10],
            [0.0, 0.20],
        ],
        dtype=np.float32,
    )
    ids = [1, 2, 3, 4, 5, 6]
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
            "concepts": [],
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
        {
            "text": corpus[4],
            "token_count": 68,
            "ontology_terms": ["Invoice Manual"],
            "concepts": ["operations"],
        },
        {
            "text": corpus[5],
            "token_count": 60,
            "ontology_terms": ["FAQ"],
            "concepts": ["navigation"],
        },
    ]
    queries = [
        GraphQualityQuery(
            query_id="local_chain",
            subset="graph_shaped",
            query_input="team alpha",
            query_text="how is team alpha related to service c",
            query_payload={"ontology_terms": ["Team Alpha", "Service B"]},
            gold_ids=[2],
            top_k=2,
            expected_graph_auto=True,
            expected_lane="graph_local",
        ),
        GraphQualityQuery(
            query_id="root_cause",
            subset="graph_shaped",
            query_input="team alpha dependency",
            query_text="why did service c outage happen",
            query_payload={"ontology_terms": ["Service B", "Service C"]},
            gold_ids=[2],
            top_k=2,
            expected_graph_auto=True,
            expected_lane="graph_local",
        ),
        GraphQualityQuery(
            query_id="community_support",
            subset="graph_shaped",
            query_input="overview posture",
            query_text="what reliability asset is connected to service c",
            query_payload={"ontology_terms": ["Service C", "support path"]},
            gold_ids=[4],
            top_k=3,
            expected_graph_auto=True,
            expected_lane="graph_community",
        ),
        GraphQualityQuery(
            query_id="invoice_manual",
            subset="ordinary",
            query_input="invoice manual",
            query_text="invoice manual",
            query_payload={},
            gold_ids=[5],
            top_k=1,
            expected_graph_auto=False,
        ),
        GraphQualityQuery(
            query_id="service_playbook_direct",
            subset="ordinary",
            query_input="reliability playbook",
            query_text="reliability playbook",
            query_payload={},
            gold_ids=[4],
            top_k=1,
            expected_graph_auto=False,
        ),
    ]
    return GraphQualityFixture(
        name="synthetic_regression",
        description="Tiny deterministic regression fixture focused on additive rescue and community tagging.",
        corpus=corpus,
        vectors=vectors,
        ids=ids,
        payloads=payloads,
        queries=queries,
    )


def _fixture_from_json(path: Path) -> GraphQualityFixture:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = list(payload.get("documents") or [])
    queries_payload = list(payload.get("queries") or [])
    return GraphQualityFixture(
        name=str(payload.get("name") or path.stem),
        description=str(payload.get("description") or ""),
        corpus=[str(item.get("text") or "") for item in documents],
        vectors=np.asarray([item.get("vector") or [] for item in documents], dtype=np.float32),
        ids=[int(item.get("id")) for item in documents],
        payloads=[dict(item.get("payload") or {}) for item in documents],
        queries=[
            GraphQualityQuery(
                query_id=str(item["query_id"]),
                subset=str(item["subset"]),
                query_input=item["query_input"],
                query_text=str(item.get("query_text") or item["query_input"]),
                query_payload=dict(item.get("query_payload") or {}),
                gold_ids=[int(value) for value in list(item.get("gold_ids") or [])],
                top_k=int(item.get("top_k") or 1),
                expected_graph_auto=(
                    bool(item["expected_graph_auto"]) if "expected_graph_auto" in item else None
                ),
                expected_lane=str(item["expected_lane"]) if item.get("expected_lane") else None,
            )
            for item in queries_payload
        ],
    )


def _representative_fixture() -> GraphQualityFixture:
    return _fixture_from_json(_REPRESENTATIVE_FIXTURE_PATH)


def _build_pipeline(root: Path, fixture: GraphQualityFixture) -> tuple[SearchPipeline, List[GraphQualityQuery]]:
    pipeline = SearchPipeline(str(root), dim=2, use_roq=False, on_disk=False)
    pipeline.index(corpus=fixture.corpus, vectors=fixture.vectors, ids=fixture.ids, payloads=fixture.payloads)
    return pipeline, fixture.queries


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return float((ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight))


def _run_mode(
    pipeline: SearchPipeline,
    queries: List[GraphQualityQuery],
    *,
    graph_mode: str,
    local_budget: int,
    community_budget: int,
    evidence_budget: int,
    enable_refinement: bool = False,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    subset_buckets: Dict[str, List[Dict[str, float]]] = {"graph_shaped": [], "ordinary": []}
    graph_applied_count = 0
    latencies_by_subset: Dict[str, List[float]] = {"graph_shaped": [], "ordinary": []}
    candidate_growth_by_subset: Dict[str, List[float]] = {"graph_shaped": [], "ordinary": []}
    solver_overhead_by_subset: Dict[str, List[float]] = {"graph_shaped": [], "ordinary": []}
    route_checks: List[Dict[str, Any]] = []

    for query in queries:
        search_start = time.perf_counter()
        result = pipeline.search(
            query.query_input,
            top_k_retrieval=max(query.top_k, 1),
            enable_refinement=bool(enable_refinement and getattr(pipeline.manager, "solver_available", False)),
            query_text=query.query_text if not isinstance(query.query_input, str) else "",
            query_payload=dict(query.query_payload),
            graph_mode=graph_mode,
            graph_options={
                "local_budget": local_budget,
                "community_budget": community_budget,
                "evidence_budget": evidence_budget,
                "max_hops": 2,
                "explain": True,
            },
        )
        elapsed_ms = (time.perf_counter() - search_start) * 1000.0
        ranked_ids = [int(doc_id) for doc_id in list(result.get("selected_ids") or [])]
        ranked_ids_at_k = ranked_ids[: query.top_k]
        top_k_metrics = _ranking_metrics(ranked_ids, query.gold_ids, k=query.top_k)
        metrics = _ranking_metrics(ranked_ids, query.gold_ids, k=max(len(ranked_ids), query.top_k))
        graph_summary = dict((result.get("retrieval") or {}).get("graph_summary") or {})
        graph_policy = dict((result.get("retrieval") or {}).get("graph_policy") or {})
        graph_provenance = dict((result.get("retrieval") or {}).get("graph_provenance") or {})
        solver_output = dict(result.get("solver_output") or {})
        solver_overhead_ms = float(solver_output.get("solve_time_ms", 0.0) or 0.0)
        graph_applied = bool((graph_summary.get("graph_result_count") or 0) > 0 or graph_policy.get("applied"))
        if graph_applied:
            graph_applied_count += 1
        subset_buckets[query.subset].append(metrics)
        latencies_by_subset[query.subset].append(elapsed_ms)
        candidate_growth_by_subset[query.subset].append(float(graph_summary.get("added_candidate_count", 0) or 0.0))
        solver_overhead_by_subset[query.subset].append(solver_overhead_ms)

        expected_lane_record = {}
        if query.expected_lane and query.gold_ids:
            expected_lane_record = dict(graph_provenance.get(str(query.gold_ids[0])) or {})
        route_ok = True
        route_reasons: List[str] = []
        if query.expected_graph_auto is not None and graph_mode == "auto":
            actual_auto = bool(graph_policy.get("applied"))
            if actual_auto != bool(query.expected_graph_auto):
                route_ok = False
                route_reasons.append("unexpected_auto_policy")
        if graph_applied and graph_summary.get("invoked_after_first_stage") is not True:
            route_ok = False
            route_reasons.append("graph_invoked_before_first_stage")
        if graph_applied and graph_summary.get("merge_mode") != "additive":
            route_ok = False
            route_reasons.append("graph_merge_not_additive")
        if query.expected_lane:
            actual_lanes = list(expected_lane_record.get("lanes") or [])
            if query.expected_lane not in actual_lanes:
                route_ok = False
                route_reasons.append("missing_expected_lane_tag")
        route_checks.append(
            {
                "query_id": query.query_id,
                "subset": query.subset,
                "passed": route_ok,
                "reasons": route_reasons,
                "expected_graph_auto": query.expected_graph_auto,
                "expected_lane": query.expected_lane,
                "actual_graph_auto": bool(graph_policy.get("applied")),
                "actual_lanes": list(expected_lane_record.get("lanes") or []),
                "merge_mode": graph_summary.get("merge_mode"),
                "invoked_after_first_stage": graph_summary.get("invoked_after_first_stage"),
            }
        )
        details.append(
            {
                "query_id": query.query_id,
                "subset": query.subset,
                "ranked_ids": ranked_ids_at_k,
                "full_ranked_ids": ranked_ids,
                "gold_ids": list(query.gold_ids),
                "metrics": metrics,
                "top_k_metrics": top_k_metrics,
                "graph_applied": graph_applied,
                "graph_policy": graph_policy,
                "graph_summary": graph_summary,
                "graph_provenance": graph_provenance,
                "latency_ms": elapsed_ms,
                "solver_overhead_ms": solver_overhead_ms,
            }
        )

    def _average(items: List[Dict[str, float]]) -> Dict[str, float]:
        if not items:
            return {"recall": 0.0, "ndcg": 0.0, "support_coverage": 0.0}
        return {
            key: float(sum(item[key] for item in items) / float(len(items)))
            for key in ("recall", "ndcg", "support_coverage")
        }

    def _latency_summary(values: List[float]) -> Dict[str, float]:
        return {
            "p50_ms": _percentile(values, 50.0),
            "p95_ms": _percentile(values, 95.0),
            "mean_ms": float(sum(values) / float(len(values))) if values else 0.0,
        }

    return {
        "queries": details,
        "graph_applied_rate": float(graph_applied_count / float(max(len(queries), 1))),
        "subsets": {
            "graph_shaped": _average(subset_buckets["graph_shaped"]),
            "ordinary": _average(subset_buckets["ordinary"]),
        },
        "latency": {
            "overall": _latency_summary(latencies_by_subset["graph_shaped"] + latencies_by_subset["ordinary"]),
            "graph_shaped": _latency_summary(latencies_by_subset["graph_shaped"]),
            "ordinary": _latency_summary(latencies_by_subset["ordinary"]),
        },
        "candidate_growth": {
            "overall_avg_added": float(
                sum(candidate_growth_by_subset["graph_shaped"] + candidate_growth_by_subset["ordinary"])
                / float(max(len(candidate_growth_by_subset["graph_shaped"] + candidate_growth_by_subset["ordinary"]), 1))
            ),
            "graph_shaped_avg_added": float(
                sum(candidate_growth_by_subset["graph_shaped"]) / float(max(len(candidate_growth_by_subset["graph_shaped"]), 1))
            ),
            "ordinary_avg_added": float(
                sum(candidate_growth_by_subset["ordinary"]) / float(max(len(candidate_growth_by_subset["ordinary"]), 1))
            ),
        },
        "solver_overhead": {
            "overall_p95_ms": _percentile(
                solver_overhead_by_subset["graph_shaped"] + solver_overhead_by_subset["ordinary"],
                95.0,
            ),
            "graph_shaped_p95_ms": _percentile(solver_overhead_by_subset["graph_shaped"], 95.0),
            "ordinary_p95_ms": _percentile(solver_overhead_by_subset["ordinary"], 95.0),
        },
        "route_checks": {
            "passed": sum(1 for item in route_checks if item["passed"]),
            "total": len(route_checks),
            "all_passed": all(item["passed"] for item in route_checks),
            "queries": route_checks,
        },
    }


def _comparison_payload(baseline: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    for subset in ("graph_shaped", "ordinary"):
        comparison[subset] = {
            metric: graph["subsets"][subset][metric] - baseline["subsets"][subset][metric]
            for metric in ("recall", "ndcg", "support_coverage")
        }
    baseline_p95 = float(baseline["latency"]["overall"]["p95_ms"] or 0.0)
    graph_p95 = float(graph["latency"]["overall"]["p95_ms"] or 0.0)
    comparison["latency"] = {
        "p95_ms_delta": graph_p95 - baseline_p95,
        "p95_relative_overhead": ((graph_p95 - baseline_p95) / baseline_p95) if baseline_p95 > 0.0 else 0.0,
    }
    return comparison


def _tier_report(
    fixture: GraphQualityFixture,
    root: Path,
    *,
    graph_mode: str,
    local_budget: int,
    community_budget: int,
    evidence_budget: int,
) -> Dict[str, Any]:
    pipeline, queries = _build_pipeline(root, fixture)
    baseline = _run_mode(
        pipeline,
        queries,
        graph_mode="off",
        local_budget=0,
        community_budget=0,
        evidence_budget=0,
    )
    graph = _run_mode(
        pipeline,
        queries,
        graph_mode=graph_mode,
        local_budget=local_budget,
        community_budget=community_budget,
        evidence_budget=evidence_budget,
    )
    return {
        "name": fixture.name,
        "description": fixture.description,
        "baseline": baseline,
        "graph": graph,
        "comparison": _comparison_payload(baseline, graph),
    }


def run_graph_quality_benchmark(
    root: Optional[Path] = None,
    *,
    graph_mode: str = "auto",
    local_budget: int = 4,
    community_budget: int = 4,
    evidence_budget: Optional[int] = None,
) -> Dict[str, Any]:
    if root is None:
        with TemporaryDirectory(prefix="voyager-graph-quality-") as temp_dir:
            return run_graph_quality_benchmark(
                Path(temp_dir),
                graph_mode=graph_mode,
                local_budget=local_budget,
                community_budget=community_budget,
                evidence_budget=evidence_budget,
            )
    effective_evidence_budget = 8 if evidence_budget is None else int(evidence_budget)
    synthetic = _tier_report(
        _synthetic_regression_fixture(),
        Path(root) / "synthetic",
        graph_mode=graph_mode,
        local_budget=local_budget,
        community_budget=community_budget,
        evidence_budget=effective_evidence_budget,
    )
    representative = _tier_report(
        _representative_fixture(),
        Path(root) / "representative",
        graph_mode=graph_mode,
        local_budget=local_budget,
        community_budget=community_budget,
        evidence_budget=effective_evidence_budget,
    )
    return {
        "baseline": representative["baseline"],
        "graph": representative["graph"],
        "comparison": representative["comparison"],
        "synthetic": synthetic,
        "representative": representative,
        "tiers": {
            "synthetic": synthetic,
            "representative": representative,
        },
        "methodology": {
            "synthetic_fixture": "Tiny deterministic regression suite for additive rescue and lane-tag validation.",
            "representative_fixture": str(_REPRESENTATIVE_FIXTURE_PATH.relative_to(_REPO_ROOT)),
            "metrics": [
                "recall",
                "ndcg",
                "support_coverage",
                "latency_p50_p95",
                "candidate_growth",
                "solver_overhead",
            ],
            "route_checks": [
                "ordinary queries stay graph-off in auto",
                "graph-shaped and compliance queries activate the graph lane in auto",
                "graph execution is post-retrieval only",
                "graph merge is additive rather than rank-replacing",
                "graph candidates carry graph_local or graph_community provenance tags",
            ],
        },
        "config": {
            "graph_mode": graph_mode,
            "local_budget": local_budget,
            "community_budget": community_budget,
            "evidence_budget": effective_evidence_budget,
        },
    }


def run_graph_quality_ablation(root: Optional[Path] = None) -> Dict[str, Any]:
    if root is None:
        with TemporaryDirectory(prefix="voyager-graph-ablation-") as temp_dir:
            return run_graph_quality_ablation(Path(temp_dir))

    def _mode_report(name: str, *, graph_mode: str, local_budget: int, community_budget: int, evidence_budget: int) -> Dict[str, Any]:
        payload = run_graph_quality_benchmark(
            Path(root) / name,
            graph_mode=graph_mode,
            local_budget=local_budget,
            community_budget=community_budget,
            evidence_budget=evidence_budget,
        )
        return {
            "synthetic": payload["synthetic"],
            "representative": payload["representative"],
            "graph": payload["graph"],
            "comparison": payload["comparison"],
        }

    return {
        "baseline": _mode_report(
            "baseline",
            graph_mode="off",
            local_budget=0,
            community_budget=0,
            evidence_budget=0,
        ),
        "local_only": _mode_report(
            "local",
            graph_mode="force",
            local_budget=4,
            community_budget=0,
            evidence_budget=4,
        ),
        "community_only": _mode_report(
            "community",
            graph_mode="force",
            local_budget=0,
            community_budget=4,
            evidence_budget=0,
        ),
        "full": _mode_report(
            "full",
            graph_mode="force",
            local_budget=4,
            community_budget=4,
            evidence_budget=8,
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run the Latence graph quality benchmark.")
    parser.add_argument("--mode", choices=("benchmark", "ablation"), default="benchmark")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--graph-mode", choices=("off", "auto", "force"), default="auto")
    parser.add_argument("--local-budget", type=int, default=4)
    parser.add_argument("--community-budget", type=int, default=4)
    parser.add_argument("--evidence-budget", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "ablation":
        report = run_graph_quality_ablation()
    else:
        report = run_graph_quality_benchmark(
            graph_mode=args.graph_mode,
            local_budget=args.local_budget,
            community_budget=args.community_budget,
            evidence_budget=args.evidence_budget,
        )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload)
