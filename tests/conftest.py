"""
Shared fixtures for Graph QA test suite.

Provides dataset loading, segment builders, brute-force ground truth,
hard-query filtering, and artifact collection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from tests.data.msmarco_loader import (
    MSMARCODataset,
    generate_synthetic_dataset,
    load_combined_dataset,
    load_msmarco_dataset,
)

RESULTS_DIR = Path("benchmarks/results")


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def msmarco_dataset() -> MSMARCODataset:
    """Load combined MS MARCO + CQADupStack corpus; falls back to MS MARCO only."""
    return load_combined_dataset()


@pytest.fixture(scope="session")
def msmarco_subset_500(msmarco_dataset: MSMARCODataset) -> MSMARCODataset:
    """500-doc random subset for quick CI."""
    return msmarco_dataset.subset(500, seed=42)


def _make_synthetic(
    n_docs: int,
    dim: int,
    tokens_per_doc: int,
    n_queries: int = 50,
    tokens_per_query: int = 32,
    n_clusters: int = 8,
    seed: int = 42,
) -> MSMARCODataset:
    return generate_synthetic_dataset(
        n_docs=n_docs,
        dim=dim,
        tokens_per_doc=tokens_per_doc,
        n_queries=n_queries,
        tokens_per_query=tokens_per_query,
        n_clusters=n_clusters,
        seed=seed,
    )


@pytest.fixture
def mm_patch_dataset():
    """Factory for synthetic multimodal patch datasets with variable token counts."""

    def _factory(n_docs: int = 200, dim: int = 128, tpd: int = 128, seed: int = 42):
        return _make_synthetic(n_docs, dim, tpd, n_queries=20, seed=seed)

    return _factory


@pytest.fixture
def churn_mix_dataset():
    """Factory for synthetic churn-test datasets."""

    def _factory(n_docs: int = 500, dim: int = 64, seed: int = 42):
        return _make_synthetic(n_docs, dim, tokens_per_doc=32, n_queries=30, seed=seed)

    return _factory


# ---------------------------------------------------------------------------
# Segment builder helpers
# ---------------------------------------------------------------------------


GEM_DEFAULTS = dict(
    n_coarse=32,
    max_degree=16,
    ef_construction=100,
    ctop_r=3,
)

SEALED_DEFAULTS = dict(
    **GEM_DEFAULTS,
    n_fine=0,
    max_kmeans_iter=15,
)

MUTABLE_DEFAULTS = dict(
    **GEM_DEFAULTS,
    n_fine=128,
    max_kmeans_iter=15,
    n_probes=4,
)


def _build_sealed(dataset: MSMARCODataset, **kwargs):
    """Build a sealed GemSegment from a dataset (not a fixture — callable)."""
    from latence_gem_index import GemSegment

    params = {**SEALED_DEFAULTS, **kwargs}
    seg = GemSegment()
    seg.build(
        dataset.all_vectors,
        dataset.doc_ids,
        dataset.offsets,
        **params,
    )
    return seg


def _build_mutable(dataset: MSMARCODataset, **kwargs):
    """Build a PyMutableGemSegment from a dataset (not a fixture — callable)."""
    from latence_gem_index import PyMutableGemSegment

    params = {**MUTABLE_DEFAULTS, **kwargs}
    seg = PyMutableGemSegment()
    seg.build(
        dataset.all_vectors,
        dataset.doc_ids,
        dataset.offsets,
        **params,
    )
    return seg


@pytest.fixture(scope="session")
def sealed_segment_500(msmarco_subset_500):
    """Session-scoped sealed segment on 500-doc subset. Shared across read-only tests."""
    return _build_sealed(msmarco_subset_500)


@pytest.fixture
def build_sealed_segment():
    """Factory that builds a fresh sealed GemSegment from a dataset."""
    return _build_sealed


@pytest.fixture
def build_mutable_segment():
    """Factory that builds a fresh PyMutableGemSegment from a dataset."""
    return _build_mutable


@pytest.fixture
def build_manager(tmp_path):
    """Factory that builds a GemNativeSegmentManager with WAL."""

    def _build(dataset: MSMARCODataset, **kwargs):
        from voyager_index._internal.inference.index_core.gem_manager import (
            GemNativeSegmentManager,
        )

        shard_path = str(tmp_path / "shard")
        mgr = GemNativeSegmentManager(
            shard_path=shard_path,
            dim=dataset.dim,
            seed_batch_size=min(256, dataset.n_docs),
            **kwargs,
        )

        for i, doc_id in enumerate(dataset.doc_ids):
            mgr.add_multidense(
                [dataset.doc_vecs[i]],
                [doc_id],
                [{"_idx": i}],
            )

        return mgr

    return _build


# ---------------------------------------------------------------------------
# Ground truth and metrics
# ---------------------------------------------------------------------------


def brute_force_maxsim(
    all_vectors: np.ndarray,
    offsets: List[Tuple[int, int]],
    doc_ids: List[int],
    query: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """Exact MaxSim ground truth. Returns (doc_id, neg_score) sorted best-first."""
    scores = []
    for i, (start, end) in enumerate(offsets):
        doc_vecs = all_vectors[start:end]
        sims = query @ doc_vecs.T
        avg_max_sim = sims.max(axis=1).mean()
        scores.append((doc_ids[i], -avg_max_sim))
    scores.sort(key=lambda x: x[1])
    return scores[:k]


def compute_recall_at_k(
    results: List[Tuple[int, float]],
    ground_truth: List[Tuple[int, float]],
    k: int,
) -> float:
    gt_ids = {doc_id for doc_id, _ in ground_truth[:k]}
    result_ids = {doc_id for doc_id, _ in results[:k]}
    if not gt_ids:
        return 1.0
    return len(gt_ids & result_ids) / len(gt_ids)


def compute_mrr(
    results: List[Tuple[int, float]],
    relevant_ids: set,
) -> float:
    for rank, (doc_id, _) in enumerate(results, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


@pytest.fixture(scope="session")
def brute_force_gt():
    """Session-scoped brute-force ground truth computer."""

    _cache: Dict[int, List[List[Tuple[int, float]]]] = {}

    def _compute(dataset: MSMARCODataset, k: int = 100) -> List[List[Tuple[int, float]]]:
        cache_key = id(dataset)
        if cache_key in _cache:
            return _cache[cache_key]

        gts = []
        for qv in dataset.query_vecs:
            gt = brute_force_maxsim(dataset.all_vectors, dataset.offsets, dataset.doc_ids, qv, k)
            gts.append(gt)
        _cache[cache_key] = gts
        return gts

    return _compute


# ---------------------------------------------------------------------------
# Hard-query filter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hard_query_filter():
    """Identify 'hard' queries: bottom percentile by score margin."""

    def _filter(
        gts: List[List[Tuple[int, float]]],
        percentile: int = 20,
    ) -> List[int]:
        margins = []
        for gt in gts:
            if len(gt) < 10:
                margins.append(float("inf"))
                continue
            top1_score = gt[0][1]
            top10_score = gt[9][1]
            margins.append(abs(top10_score - top1_score))

        threshold = np.percentile(margins, percentile)
        return [i for i, m in enumerate(margins) if m <= threshold]

    return _filter


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------


class QAReportCollector:
    """Collects JSON metrics from test gates and writes combined report."""

    def __init__(self):
        self.sections: Dict[str, Any] = {}

    def record(self, section: str, metrics: Dict[str, Any]):
        self.sections[section] = metrics

    def save(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for section, data in self.sections.items():
            path = RESULTS_DIR / f"graph_qa_{section}.json"
            path.write_text(json.dumps(data, indent=2, default=_json_default))

        combined = RESULTS_DIR / "graph_qa_report.json"
        combined.write_text(
            json.dumps(self.sections, indent=2, default=_json_default)
        )


def _json_default(obj):
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@pytest.fixture(scope="session")
def qa_artifacts():
    """Session-scoped artifact collector; writes reports at session end."""
    collector = QAReportCollector()
    yield collector
    collector.save()
