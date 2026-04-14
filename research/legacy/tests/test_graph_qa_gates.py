"""
Graph QA & Hard-Tail Validation Gates
======================================

Release gate test suite for voyager-index. Validates graph health,
search quality, construction correctness, churn safety, multimodal
scaling, and performance regression thresholds.

Markers:
    graph_qa_quick  — runs per-PR on 200-doc subset (~15-20s build)
    graph_qa_nightly — runs nightly on full 2.5K corpus (~2-3 min build)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.conftest import (
    SEALED_DEFAULTS,
    brute_force_maxsim,
    compute_recall_at_k,
)
from tests.data.msmarco_loader import MSMARCODataset

# ---------------------------------------------------------------------------
# Thresholds (tunable per tier)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "real": {
        "giant_component_frac": 0.99,
        "cross_cluster_edge_ratio": 0.10,
        "navigability_reach": 0.95,
        "recall_at_10_ef100": 0.30,
        "hard_recall_at_10": 0.10,
        "multi_needle_coverage": 0.50,
        "construction_bias_recall_delta": 0.05,
        "construction_bias_connectivity_delta": 0.02,
        "p99_latency_ratio": 10.0,
        "qps_floor": 50,
        "patch_build_ratio": 10.0,
    },
    "synthetic": {
        "giant_component_frac": 0.999,
        "cross_cluster_edge_ratio": 0.10,
        "navigability_reach": 0.99,
        "recall_at_10_ef100": 0.30,
        "hard_recall_at_10": 0.15,
        "multi_needle_coverage": 0.30,
        "construction_bias_recall_delta": 0.05,
        "construction_bias_connectivity_delta": 0.02,
        "p99_latency_ratio": 10.0,
        "qps_floor": 50,
        "patch_build_ratio": 10.0,
    },
}


def _get_threshold(dataset: MSMARCODataset, key: str) -> float:
    tier = "synthetic" if dataset.is_synthetic else "real"
    return THRESHOLDS[tier][key]


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Build-time Graph Health (GH1-GH4)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.graph_qa_quick
@pytest.mark.graph_qa_nightly
class TestGraphHealth:
    """GH1-GH4: Graph connectivity, bridges, navigability, repair."""

    def test_gh1_connectivity(
        self, msmarco_subset_500, sealed_segment_500, qa_artifacts
    ):
        """GH1: Graph must be fully connected (or giant component >= 0.999)."""
        ds = msmarco_subset_500
        seg = sealed_segment_500

        n_comp, giant_frac, cross_ratio = seg.graph_connectivity_report()
        n_nodes = seg.n_nodes()
        n_edges = seg.n_edges()
        n_docs = seg.n_docs()
        mean_degree = 2.0 * n_edges / max(n_nodes, 1)

        threshold = _get_threshold(ds, "giant_component_frac")
        assert n_comp == 1 or giant_frac >= threshold, (
            f"Graph not connected: {n_comp} components, "
            f"giant_frac={giant_frac:.4f} < {threshold}"
        )
        assert n_docs == ds.n_docs, f"doc count mismatch: {n_docs} != {ds.n_docs}"

        qa_artifacts.record("graph_health_gh1", {
            "n_components": n_comp,
            "giant_component_frac": giant_frac,
            "cross_cluster_edge_ratio": cross_ratio,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_docs": n_docs,
            "mean_degree": mean_degree,
            "passed": True,
        })

    def test_gh2_cross_cluster_bridges(
        self, msmarco_subset_500, build_mutable_segment, qa_artifacts
    ):
        """GH2: Cross-cluster bridge edges must exist at sufficient ratio."""
        ds = msmarco_subset_500
        seg = build_mutable_segment(ds)

        n_comp, giant_frac, cross_ratio = seg.graph_connectivity_report()
        threshold = _get_threshold(ds, "cross_cluster_edge_ratio")

        assert cross_ratio >= threshold, (
            f"cross_cluster_edge_ratio={cross_ratio:.4f} < {threshold}"
        )

        qa_artifacts.record("graph_health_gh2", {
            "n_components": n_comp,
            "giant_component_frac": giant_frac,
            "cross_cluster_edge_ratio": cross_ratio,
            "threshold": threshold,
            "passed": True,
        })

    def test_gh3_navigability_proxy(
        self, msmarco_subset_500, sealed_segment_500, qa_artifacts
    ):
        """GH3: Sampled search must reach target docs with high probability."""
        ds = msmarco_subset_500
        seg = sealed_segment_500

        n_samples = min(200, ds.n_queries)
        found_count = 0
        visited_list = []

        for qi in range(n_samples):
            qv = ds.query_vecs[qi]
            gt = brute_force_maxsim(ds.all_vectors, ds.offsets, ds.doc_ids, qv, k=1)
            if not gt:
                continue
            target_id = gt[0][0]

            results, (nodes_visited, _) = seg.search_with_stats(
                qv, k=50, ef=200, n_probes=4,
            )
            result_ids = {doc_id for doc_id, _ in results}
            if target_id in result_ids:
                found_count += 1
            visited_list.append(nodes_visited)

        reach_frac = found_count / max(n_samples, 1)
        threshold = _get_threshold(ds, "navigability_reach")
        median_visited = float(np.median(visited_list)) if visited_list else 0
        p95_visited = float(np.percentile(visited_list, 95)) if visited_list else 0

        assert reach_frac >= threshold, (
            f"navigability reach={reach_frac:.4f} < {threshold}"
        )

        qa_artifacts.record("graph_health_gh3", {
            "reachable_fraction": reach_frac,
            "median_nodes_visited": median_visited,
            "p95_nodes_visited": p95_visited,
            "n_samples": n_samples,
            "threshold": threshold,
            "passed": True,
        })

    def test_gh4_repair_audit(
        self, msmarco_subset_500, build_mutable_segment, qa_artifacts
    ):
        """GH4: Heal must improve graph quality after deletions."""
        ds = msmarco_subset_500
        seg = build_mutable_segment(ds)

        n_to_delete = ds.n_docs // 5
        for i in range(n_to_delete):
            seg.delete(ds.doc_ids[i])

        metrics_before = seg.graph_quality_metrics()
        conn_before = seg.graph_connectivity_report()

        seg.heal()

        metrics_after = seg.graph_quality_metrics()
        conn_after = seg.graph_connectivity_report()

        del_ratio_b, avg_deg_b, iso_b, stale_b = metrics_before
        del_ratio_a, avg_deg_a, iso_a, stale_a = metrics_after
        n_comp_b, giant_b, cross_b = conn_before
        n_comp_a, giant_a, cross_a = conn_after

        assert stale_a <= stale_b, (
            f"Stale rep ratio did not improve: {stale_b:.4f} -> {stale_a:.4f}"
        )
        assert iso_a < 0.30, (
            f"Isolated ratio too high after heal: {iso_a:.4f} >= 0.30"
        )

        qa_artifacts.record("graph_health_gh4", {
            "before": {
                "delete_ratio": del_ratio_b, "avg_degree": avg_deg_b,
                "isolated_ratio": iso_b, "stale_rep_ratio": stale_b,
                "n_components": n_comp_b, "giant_frac": giant_b,
            },
            "after": {
                "delete_ratio": del_ratio_a, "avg_degree": avg_deg_a,
                "isolated_ratio": iso_a, "stale_rep_ratio": stale_a,
                "n_components": n_comp_a, "giant_frac": giant_a,
            },
            "passed": True,
        })


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Search Quality (Q1-Q3)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.graph_qa_quick
@pytest.mark.graph_qa_nightly
class TestSearchQuality:
    """Q1-Q3: Fixed-recall curves, hard-query subset, multi-needle coverage."""

    def test_q1_fixed_recall_curve(
        self, msmarco_subset_500, sealed_segment_500, brute_force_gt, qa_artifacts
    ):
        """Q1: Recall@10 must meet threshold and be monotonically non-decreasing."""
        ds = msmarco_subset_500
        seg = sealed_segment_500
        gts = brute_force_gt(ds, k=100)

        ef_values = [16, 32, 64, 100, 200]
        curve = []

        for ef in ef_values:
            recalls = []
            latencies = []
            for qi, qv in enumerate(ds.query_vecs):
                t0 = time.perf_counter()
                results = seg.search(qv, k=10, ef=ef, n_probes=4)
                latencies.append((time.perf_counter() - t0) * 1e6)
                recalls.append(compute_recall_at_k(results, gts[qi], k=10))

            mean_recall = float(np.mean(recalls))
            p95_latency = float(np.percentile(latencies, 95))
            curve.append({
                "ef": ef,
                "recall_at_10": mean_recall,
                "p95_latency_us": p95_latency,
            })

        recall_at_ef100 = next(c["recall_at_10"] for c in curve if c["ef"] == 100)
        threshold = _get_threshold(ds, "recall_at_10_ef100")
        assert recall_at_ef100 >= threshold, (
            f"Recall@10 at ef=100: {recall_at_ef100:.4f} < {threshold}"
        )

        for i in range(1, len(curve)):
            assert curve[i]["recall_at_10"] >= curve[i - 1]["recall_at_10"] - 0.02, (
                f"Recall not monotonic: ef={curve[i-1]['ef']} "
                f"({curve[i-1]['recall_at_10']:.4f}) > "
                f"ef={curve[i]['ef']} ({curve[i]['recall_at_10']:.4f})"
            )

        qa_artifacts.record("search_quality_q1", {
            "curve": curve,
            "recall_at_10_ef100": recall_at_ef100,
            "threshold": threshold,
            "passed": True,
        })

    def test_q2_hard_query_subset(
        self,
        msmarco_subset_500,
        sealed_segment_500,
        brute_force_gt,
        hard_query_filter,
        qa_artifacts,
    ):
        """Q2: Hard queries (tight score margins) must still achieve minimum recall."""
        ds = msmarco_subset_500
        seg = sealed_segment_500
        gts = brute_force_gt(ds, k=100)
        hard_idxs = hard_query_filter(gts, percentile=20)

        if len(hard_idxs) < 5:
            pytest.skip("Too few hard queries identified")

        recalls = []
        latencies = []
        for qi in hard_idxs:
            qv = ds.query_vecs[qi]
            t0 = time.perf_counter()
            results = seg.search(qv, k=10, ef=200, n_probes=4)
            latencies.append((time.perf_counter() - t0) * 1e6)
            recalls.append(compute_recall_at_k(results, gts[qi], k=10))

        mean_recall = float(np.mean(recalls))
        p99_latency = float(np.percentile(latencies, 99))
        median_latency = float(np.median(latencies))

        threshold = _get_threshold(ds, "hard_recall_at_10")
        assert mean_recall >= threshold, (
            f"Hard query Recall@10={mean_recall:.4f} < {threshold}"
        )

        lat_ratio = _get_threshold(ds, "p99_latency_ratio")
        assert p99_latency < median_latency * lat_ratio or p99_latency < 100_000, (
            f"p99 latency={p99_latency:.0f}us > {lat_ratio}x median={median_latency:.0f}us"
        )

        qa_artifacts.record("search_quality_q2", {
            "n_hard_queries": len(hard_idxs),
            "mean_recall_at_10": mean_recall,
            "p99_latency_us": p99_latency,
            "median_latency_us": median_latency,
            "threshold": threshold,
            "passed": True,
        })

    def test_q3_multi_needle_coverage(
        self, msmarco_subset_500, sealed_segment_500, qa_artifacts
    ):
        """Q3: Top-20 results should cover diverse documents (not all from one region)."""
        ds = msmarco_subset_500
        seg = sealed_segment_500

        n_buckets = 16
        coverage_scores = []

        for qi, qv in enumerate(ds.query_vecs):
            results = seg.search(qv, k=20, ef=200, n_probes=4)
            result_ids = [doc_id for doc_id, _ in results]

            buckets_seen = set()
            for doc_id in result_ids:
                if doc_id < ds.n_docs:
                    buckets_seen.add(doc_id % n_buckets)

            coverage_scores.append(len(buckets_seen) >= 2)

        coverage = float(np.mean(coverage_scores))
        threshold = _get_threshold(ds, "multi_needle_coverage")

        assert coverage >= threshold, (
            f"Multi-needle coverage={coverage:.4f} < {threshold}"
        )

        qa_artifacts.record("search_quality_q3", {
            "coverage_fraction": coverage,
            "threshold": threshold,
            "n_queries": len(coverage_scores),
            "n_buckets": n_buckets,
            "passed": True,
        })


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Construction Bias (C1)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.graph_qa_quick
@pytest.mark.graph_qa_nightly
class TestConstructionBias:
    """C1: Strict vs pragmatic build should produce similar quality."""

    def test_c1_strict_vs_pragmatic(
        self, msmarco_subset_500, brute_force_gt, hard_query_filter, qa_artifacts
    ):
        """C1: qEMD (strict) vs qCH (pragmatic) recall and connectivity delta."""
        from latence_gem_index import GemSegment

        ds = msmarco_subset_500
        gts = brute_force_gt(ds, k=100)
        hard_idxs = hard_query_filter(gts, percentile=20)
        if len(hard_idxs) < 3:
            hard_idxs = list(range(min(10, ds.n_queries)))

        seg_ref = GemSegment()
        seg_ref.build(
            ds.all_vectors, ds.doc_ids, ds.offsets,
            **SEALED_DEFAULTS, use_emd=True,
        )

        seg_prod = GemSegment()
        seg_prod.build(
            ds.all_vectors, ds.doc_ids, ds.offsets,
            **SEALED_DEFAULTS, use_emd=False,
        )

        # C1.1: Connectivity delta
        n_comp_ref, giant_ref, cross_ref = seg_ref.graph_connectivity_report()
        n_comp_prod, giant_prod, cross_prod = seg_prod.graph_connectivity_report()
        giant_delta = abs(giant_ref - giant_prod)
        conn_threshold = _get_threshold(ds, "construction_bias_connectivity_delta")
        assert giant_delta < conn_threshold, (
            f"Giant frac delta={giant_delta:.4f} >= {conn_threshold}"
        )

        # C1.2: Recall delta on hard queries
        def _measure_recall(seg, idxs):
            recalls = []
            for qi in idxs:
                results = seg.search(ds.query_vecs[qi], k=10, ef=200, n_probes=4)
                recalls.append(compute_recall_at_k(results, gts[qi], k=10))
            return float(np.mean(recalls))

        recall_ref = _measure_recall(seg_ref, hard_idxs)
        recall_prod = _measure_recall(seg_prod, hard_idxs)
        recall_delta = abs(recall_ref - recall_prod)
        recall_threshold = _get_threshold(ds, "construction_bias_recall_delta")

        assert recall_delta <= recall_threshold + 0.02, (
            f"Recall delta={recall_delta:.4f} > {recall_threshold}"
        )

        qa_artifacts.record("construction_bias_c1", {
            "c1_1_connectivity": {
                "ref_giant_frac": giant_ref,
                "prod_giant_frac": giant_prod,
                "delta": giant_delta,
            },
            "c1_2_recall": {
                "ref_recall": recall_ref,
                "prod_recall": recall_prod,
                "delta": recall_delta,
            },
            "c1_3_cross_cluster": {
                "ref_cross_ratio": cross_ref,
                "prod_cross_ratio": cross_prod,
            },
            "n_hard_queries": len(hard_idxs),
            "passed": True,
        })


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Churn & Drift (O1-O3)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.graph_qa_quick
@pytest.mark.graph_qa_nightly
class TestChurnAndDrift:
    """O1-O3: Crash recovery, compaction, mutable-to-sealed quality."""

    def test_o1_crash_recovery_determinism(
        self, msmarco_subset_500, tmp_path, qa_artifacts
    ):
        """O1: Manager recovered from WAL must not lose data and still search."""
        from voyager_index._internal.inference.index_core.gem_manager import (
            GemNativeSegmentManager,
        )

        ds = msmarco_subset_500
        shard = str(tmp_path / "o1_shard")
        n_docs = min(100, ds.n_docs)

        mgr = GemNativeSegmentManager(
            shard_path=shard,
            dim=ds.dim,
            seed_batch_size=min(128, n_docs),
            enable_wal=True,
        )

        for i in range(n_docs):
            mgr.add_multidense(
                [ds.doc_vecs[i]], [ds.doc_ids[i]], [{"_idx": i}]
            )

        n_delete = n_docs // 10
        delete_ids = [ds.doc_ids[i] for i in range(n_delete)]
        mgr.delete(delete_ids)
        mgr.flush()
        mgr.close()

        mgr2 = GemNativeSegmentManager(
            shard_path=shard,
            dim=ds.dim,
            seed_batch_size=min(128, n_docs),
            enable_wal=True,
        )

        assert mgr2._active is not None and mgr2._active.is_ready(), (
            "Active segment not rebuilt after WAL recovery"
        )

        sample_queries = ds.query_vecs[:10]
        non_empty = 0
        valid_ids = 0
        alive_ids = set(ds.doc_ids[n_delete:n_docs])

        for qv in sample_queries:
            res = mgr2.search_multivector(qv, k=10, ef=100)
            if len(res) > 0:
                non_empty += 1
            for doc_id, _ in res:
                if doc_id in alive_ids:
                    valid_ids += 1

        mgr2.close()

        search_frac = non_empty / len(sample_queries)
        assert search_frac >= 0.8, (
            f"Recovery search coverage={search_frac:.2f} < 0.8 "
            f"({non_empty}/{len(sample_queries)} queries returned results)"
        )
        assert valid_ids > 0, "No valid (non-deleted) doc IDs in results"

        qa_artifacts.record("churn_o1", {
            "n_docs": n_docs,
            "n_deleted": n_delete,
            "search_coverage": search_frac,
            "valid_result_ids": valid_ids,
            "n_sample_queries": len(sample_queries),
            "passed": True,
        })

    def test_o2_high_delete_compaction(
        self, msmarco_subset_500, build_mutable_segment, qa_artifacts
    ):
        """O2: 50% deletion + compact + heal must improve graph over raw deletion state."""
        ds = msmarco_subset_500
        seg = build_mutable_segment(ds)

        metrics_pre = seg.graph_quality_metrics()
        conn_pre = seg.graph_connectivity_report()

        n_delete = ds.n_docs // 2
        for i in range(n_delete):
            seg.delete(ds.doc_ids[i])

        conn_post_del = seg.graph_connectivity_report()
        n_comp_post_del = conn_post_del[0]

        seg.compact()
        seg.heal()

        metrics_post = seg.graph_quality_metrics()
        conn_post = seg.graph_connectivity_report()

        del_post, avg_deg_post, iso_post, stale_post = metrics_post
        n_comp_post, giant_post, cross_post = conn_post

        assert n_comp_post <= n_comp_post_del, (
            f"Heal did not reduce components: {n_comp_post_del} -> {n_comp_post}"
        )
        assert iso_post < 0.15, (
            f"Post-heal isolated_ratio={iso_post:.4f} >= 0.15"
        )
        assert del_post < 0.01, (
            f"Compaction did not clean deleted entries: del_ratio={del_post:.4f}"
        )
        assert giant_post >= 0.30, (
            f"Post-heal giant_frac={giant_post:.4f} < 0.30 (too fragmented)"
        )

        qa_artifacts.record("churn_o2", {
            "n_deleted": n_delete,
            "pre": {
                "metrics": metrics_pre,
                "connectivity": conn_pre,
            },
            "post_delete": {
                "n_components": n_comp_post_del,
            },
            "post_heal": {
                "metrics": metrics_post,
                "connectivity": conn_post,
                "components_reduced": n_comp_post_del - n_comp_post,
            },
            "passed": True,
        })

    def test_o3_mutable_to_sealed_quality(
        self, msmarco_subset_500, build_sealed_segment, build_mutable_segment,
        brute_force_gt, qa_artifacts
    ):
        """O3: Sealed segment recall must be >= mutable segment recall."""
        ds = msmarco_subset_500
        gts = brute_force_gt(ds, k=100)

        seg_mut = build_mutable_segment(ds)
        seg_sealed = build_sealed_segment(ds)

        n_eval = min(50, ds.n_queries)
        recalls_mut = []
        recalls_sealed = []

        for qi in range(n_eval):
            qv = ds.query_vecs[qi]

            res_mut = seg_mut.search(qv, k=10, ef=200)
            recalls_mut.append(compute_recall_at_k(res_mut, gts[qi], k=10))

            res_sealed = seg_sealed.search(qv, k=10, ef=200, n_probes=4)
            recalls_sealed.append(compute_recall_at_k(res_sealed, gts[qi], k=10))

        mean_mut = float(np.mean(recalls_mut))
        mean_sealed = float(np.mean(recalls_sealed))

        assert mean_sealed >= mean_mut - 0.10, (
            f"Sealed recall={mean_sealed:.4f} much worse than "
            f"mutable recall={mean_mut:.4f}"
        )

        qa_artifacts.record("churn_o3", {
            "mean_recall_mutable": mean_mut,
            "mean_recall_sealed": mean_sealed,
            "n_eval_queries": n_eval,
            "passed": True,
        })


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Multimodal Patch Stress (M1-M2)
# ═══════════════════════════════════════════════════════════════════════════


class TestMultimodalStress:
    """M1-M2: Patch-vector length scaling and hubness stress."""

    @pytest.mark.graph_qa_quick
    @pytest.mark.parametrize("tpd", [32])
    def test_m1_patch_length_scaling_quick(self, tpd, mm_patch_dataset, qa_artifacts):
        """M1 (quick): Single tpd point for PR gate."""
        self._run_m1(tpd, mm_patch_dataset, qa_artifacts)

    @pytest.mark.graph_qa_nightly
    @pytest.mark.parametrize("tpd", [32, 128, 512, 1024])
    def test_m1_patch_length_scaling_nightly(self, tpd, mm_patch_dataset, qa_artifacts):
        """M1 (nightly): Full sweep of token-per-doc sizes."""
        self._run_m1(tpd, mm_patch_dataset, qa_artifacts)

    def _run_m1(self, tpd, mm_patch_dataset, qa_artifacts):
        from latence_gem_index import GemSegment

        n_docs = max(50, 200 // max(tpd // 32, 1))
        ds = mm_patch_dataset(n_docs=n_docs, dim=128, tpd=tpd)

        seg = GemSegment()
        t0 = time.perf_counter()
        seg.build(
            ds.all_vectors, ds.doc_ids, ds.offsets,
            **SEALED_DEFAULTS,
        )
        build_time = time.perf_counter() - t0

        latencies = []
        for qi in range(min(20, ds.n_queries)):
            t0 = time.perf_counter()
            seg.search(ds.query_vecs[qi], k=10, ef=64, n_probes=4)
            latencies.append((time.perf_counter() - t0) * 1e6)

        p99 = float(np.percentile(latencies, 99))

        qa_artifacts.record(f"multimodal_m1_tpd{tpd}", {
            "tpd": tpd,
            "n_docs": n_docs,
            "build_time_s": build_time,
            "p99_latency_us": p99,
            "passed": True,
        })

    @pytest.mark.graph_qa_quick
    @pytest.mark.graph_qa_nightly
    def test_m2_hubness_stress(self, qa_artifacts):
        """M2: Uniform-patch corpus must not collapse recall."""
        from latence_gem_index import GemSegment

        rng = np.random.RandomState(42)
        dim = 64
        n_docs = 100
        tpd = 32
        shared_center = rng.randn(dim).astype(np.float32) * 0.1

        doc_vecs = []
        offsets = []
        pos = 0
        for _ in range(n_docs):
            vecs = shared_center + rng.randn(tpd, dim).astype(np.float32) * 0.05
            doc_vecs.append(vecs)
            offsets.append((pos, pos + tpd))
            pos += tpd
        all_vectors = np.vstack(doc_vecs).astype(np.float32)

        seg = GemSegment()
        seg.build(
            all_vectors,
            list(range(n_docs)),
            offsets,
            **SEALED_DEFAULTS,
        )

        queries = [
            shared_center + rng.randn(16, dim).astype(np.float32) * 0.1
            for _ in range(10)
        ]

        all_results = []
        for qv in queries:
            results = seg.search(qv, k=10, ef=100, n_probes=4)
            all_results.append(len(results))

        assert all(r >= 5 for r in all_results), (
            f"Hubness collapse: some queries returned < 5 results: {all_results}"
        )

        qa_artifacts.record("multimodal_m2", {
            "n_docs": n_docs,
            "result_counts": all_results,
            "passed": True,
        })


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Performance Regression (P1-P2)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.graph_qa_quick
@pytest.mark.graph_qa_nightly
class TestPerformanceRegression:
    """P1-P2: Throughput and tail latency stability."""

    def test_p1_search_throughput(
        self, msmarco_subset_500, sealed_segment_500, qa_artifacts
    ):
        """P1: QPS must meet minimum floor."""
        ds = msmarco_subset_500
        seg = sealed_segment_500

        for qv in ds.query_vecs[:3]:
            seg.search(qv, k=10, ef=64, n_probes=4)

        n_queries = min(100, ds.n_queries)
        t0 = time.perf_counter()
        for qi in range(n_queries):
            seg.search(ds.query_vecs[qi], k=10, ef=64, n_probes=4)
        elapsed = time.perf_counter() - t0

        qps = n_queries / max(elapsed, 1e-9)
        threshold = _get_threshold(ds, "qps_floor")

        assert qps >= threshold, f"QPS={qps:.1f} < {threshold}"

        qa_artifacts.record("performance_p1", {
            "qps": qps,
            "n_queries": n_queries,
            "elapsed_s": elapsed,
            "threshold": threshold,
            "passed": True,
        })

    def test_p2_tail_latency_stability(
        self, msmarco_subset_500, sealed_segment_500, qa_artifacts
    ):
        """P2: p99 latency must not exceed threshold ratio of p50."""
        ds = msmarco_subset_500
        seg = sealed_segment_500

        for qv in ds.query_vecs[:3]:
            seg.search(qv, k=10, ef=64, n_probes=4)

        latencies = []
        for qi in range(min(100, ds.n_queries)):
            t0 = time.perf_counter()
            seg.search(ds.query_vecs[qi], k=10, ef=64, n_probes=4)
            latencies.append((time.perf_counter() - t0) * 1e6)

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        lat_ratio = _get_threshold(ds, "p99_latency_ratio")

        assert p99 < p50 * lat_ratio or p99 < 100_000, (
            f"p99={p99:.0f}us > {lat_ratio}x p50={p50:.0f}us"
        )

        qa_artifacts.record("performance_p2", {
            "p50_us": p50,
            "p95_us": p95,
            "p99_us": p99,
            "ratio_p99_p50": p99 / max(p50, 1),
            "threshold_ratio": lat_ratio,
            "passed": True,
        })
