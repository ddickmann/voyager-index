"""Semantic entropy peer for the Groundedness feature (Phase G3).

Implements Farquhar et al. 2024's semantic-entropy construction: given
multiple responses drawn from the same LLM at non-zero sampling
temperature, cluster them by meaning-preserving equivalence (bidirectional
entailment) and compute the Shannon entropy over cluster sizes. A lower
entropy signals the LLM "knows the answer" (consistent samples); a
higher entropy signals confabulation.

Design choices that keep the implementation cheap and tractable:

- **Equivalence = mutual entailment.** Two samples ``s_i`` and ``s_j``
  are in the same cluster iff ``entail(s_i, s_j) > tau_entail`` and
  ``entail(s_j, s_i) > tau_entail`` and neither direction is
  contradicted. This is symmetric by construction (guaranteed by the
  ``and`` over both directions) and transitive enough for small sample
  counts (N<=8) that equivalence-class collapse is rare in practice.
- **Reuses the existing NLI provider.** We do not instantiate a new
  model; instead we consume the same ``NLIProvider`` that drives the
  claim-level verifier. This keeps semantic entropy latency
  predictable and proportional to ``N*(N-1)`` NLI calls.
- **Shannon entropy is normalised** so the returned aggregate is in
  ``[0, 1]``: ``1 - H / log(N)``. High cluster agreement -> 1.0;
  maximally diverse samples -> 0.0.
- **Graceful degradation.** With fewer than two samples the aggregate
  is ``None``; callers should disable the channel in that case.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_ENTAILMENT_THRESHOLD = 0.55
_DEFAULT_CONTRADICTION_THRESHOLD = 0.5
_DEFAULT_MAX_SAMPLES = 8


@dataclass
class SemanticEntropyCluster:
    """An equivalence class of samples that mutually entail each other."""

    cluster_id: int
    members: List[int]
    representative: str


@dataclass
class SemanticEntropyResult:
    """Aggregate and diagnostics for one semantic-entropy computation."""

    aggregate: Optional[float]
    entropy_raw: Optional[float]
    cluster_count: int
    sample_count: int
    clusters: List[SemanticEntropyCluster]
    skipped_reason: Optional[str] = None
    latency_ms: float = 0.0


def is_semantic_entropy_enabled() -> bool:
    """True when semantic-entropy fusion weight is configured > 0."""

    raw = os.environ.get("VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY", "0")
    try:
        return float(raw) > 0
    except ValueError:
        return False


def default_max_samples() -> int:
    try:
        return max(2, int(os.environ.get("VOYAGER_GROUNDEDNESS_SEMANTIC_ENTROPY_MAX_SAMPLES", _DEFAULT_MAX_SAMPLES)))
    except ValueError:
        return _DEFAULT_MAX_SAMPLES


def default_entailment_threshold() -> float:
    try:
        return float(os.environ.get("VOYAGER_GROUNDEDNESS_SEMANTIC_ENTROPY_ENTAIL_THRESHOLD", _DEFAULT_ENTAILMENT_THRESHOLD))
    except ValueError:
        return _DEFAULT_ENTAILMENT_THRESHOLD


def default_contradiction_threshold() -> float:
    try:
        return float(os.environ.get("VOYAGER_GROUNDEDNESS_SEMANTIC_ENTROPY_CONTRADICT_THRESHOLD", _DEFAULT_CONTRADICTION_THRESHOLD))
    except ValueError:
        return _DEFAULT_CONTRADICTION_THRESHOLD


def _union_find(n: int) -> List[int]:
    return list(range(n))


def _find(parents: List[int], idx: int) -> int:
    while parents[idx] != idx:
        parents[idx] = parents[parents[idx]]
        idx = parents[idx]
    return idx


def _union(parents: List[int], a: int, b: int) -> None:
    root_a = _find(parents, a)
    root_b = _find(parents, b)
    if root_a != root_b:
        # Attach the larger-indexed root to the smaller-indexed one so the
        # cluster's representative is stable and deterministic.
        if root_a < root_b:
            parents[root_b] = root_a
        else:
            parents[root_a] = root_b


def _pairs_to_probe(samples: Sequence[str]) -> List[Tuple[int, int]]:
    """All ordered distinct pairs; bidirectional entailment needs both directions."""

    n = len(samples)
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def cluster_samples_by_entailment(
    samples: Sequence[str],
    nli_provider: Any,
    *,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> List[SemanticEntropyCluster]:
    """Cluster ``samples`` by mutual entailment using ``nli_provider``.

    Returns a deterministic list of clusters. The representative of a
    cluster is the member with the lowest original index.
    """

    n = len(samples)
    if n == 0:
        return []
    if n == 1:
        return [SemanticEntropyCluster(cluster_id=0, members=[0], representative=samples[0])]

    pairs = _pairs_to_probe(samples)
    premises = [samples[i] for i, _ in pairs]
    hypotheses = [samples[j] for _, j in pairs]
    triples = nli_provider.entail(premises, hypotheses)
    if len(triples) != len(pairs):
        raise RuntimeError(
            "NLI provider returned {got} triples for pair count {expected}".format(
                got=len(triples), expected=len(pairs)
            )
        )

    # Build a directed entailment boolean grid.
    entails = [[False] * n for _ in range(n)]
    contradicts = [[False] * n for _ in range(n)]
    for (i, j), (entail, _neutral, contradict) in zip(pairs, triples):
        entails[i][j] = entail >= entailment_threshold and contradict < contradiction_threshold
        contradicts[i][j] = contradict >= contradiction_threshold

    parents = _union_find(n)
    for i in range(n):
        for j in range(i + 1, n):
            mutual_entail = entails[i][j] and entails[j][i]
            if mutual_entail and not (contradicts[i][j] or contradicts[j][i]):
                _union(parents, i, j)

    cluster_by_root: Dict[int, List[int]] = {}
    for idx in range(n):
        root = _find(parents, idx)
        cluster_by_root.setdefault(root, []).append(idx)

    clusters: List[SemanticEntropyCluster] = []
    for cluster_id, (root, members) in enumerate(sorted(cluster_by_root.items())):
        members_sorted = sorted(members)
        clusters.append(
            SemanticEntropyCluster(
                cluster_id=cluster_id,
                members=members_sorted,
                representative=samples[members_sorted[0]],
            )
        )
    return clusters


def _shannon_entropy(probs: Sequence[float]) -> float:
    total = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        total -= p * math.log(p)
    return total


def _aggregate_from_clusters(
    clusters: Sequence[SemanticEntropyCluster],
    sample_count: int,
) -> Tuple[float, float]:
    """Return ``(normalized_aggregate, raw_entropy_nats)``.

    ``normalized_aggregate`` is ``1 - H / log(N)`` clipped to ``[0, 1]``.
    When ``sample_count <= 1`` the aggregate is ``1.0`` by convention
    (a single sample can't disagree with itself, but we shouldn't call
    this path anyway).
    """

    n = int(sample_count)
    if n <= 1 or not clusters:
        return 1.0, 0.0
    sizes = [float(len(c.members)) for c in clusters]
    total = sum(sizes) or 1.0
    probs = [size / total for size in sizes]
    raw = _shannon_entropy(probs)
    cap = math.log(max(2, n))
    normalized = 1.0 - raw / cap
    normalized = max(0.0, min(1.0, normalized))
    return float(normalized), float(raw)


def compute_semantic_entropy(
    samples: Sequence[str],
    nli_provider: Optional[Any],
    *,
    entailment_threshold: Optional[float] = None,
    contradiction_threshold: Optional[float] = None,
    max_samples: Optional[int] = None,
) -> SemanticEntropyResult:
    """End-to-end semantic-entropy computation for one request.

    The function is robust: it returns an empty result with
    ``skipped_reason`` populated whenever the NLI provider fails or the
    inputs are insufficient. Callers should not raise on error from
    here; they should fall back to the embedding-only headline.
    """

    cleaned = [s.strip() for s in samples if s and s.strip()]
    started = time.perf_counter()
    if nli_provider is None:
        return SemanticEntropyResult(
            aggregate=None,
            entropy_raw=None,
            cluster_count=0,
            sample_count=0,
            clusters=[],
            skipped_reason="no_nli_provider",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    if len(cleaned) < 2:
        return SemanticEntropyResult(
            aggregate=None,
            entropy_raw=None,
            cluster_count=max(0, len(cleaned)),
            sample_count=len(cleaned),
            clusters=[],
            skipped_reason="not_enough_samples",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    cap = max_samples if max_samples is not None else default_max_samples()
    if cap > 0 and len(cleaned) > cap:
        cleaned = cleaned[:cap]

    entail_thr = (
        entailment_threshold
        if entailment_threshold is not None
        else default_entailment_threshold()
    )
    contradict_thr = (
        contradiction_threshold
        if contradiction_threshold is not None
        else default_contradiction_threshold()
    )
    try:
        clusters = cluster_samples_by_entailment(
            cleaned,
            nli_provider,
            entailment_threshold=entail_thr,
            contradiction_threshold=contradict_thr,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_entropy_cluster_failed", extra={"error": str(exc)})
        return SemanticEntropyResult(
            aggregate=None,
            entropy_raw=None,
            cluster_count=0,
            sample_count=len(cleaned),
            clusters=[],
            skipped_reason="nli_cluster_failed",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    aggregate, raw = _aggregate_from_clusters(clusters, sample_count=len(cleaned))
    return SemanticEntropyResult(
        aggregate=float(aggregate),
        entropy_raw=float(raw),
        cluster_count=len(clusters),
        sample_count=len(cleaned),
        clusters=list(clusters),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


__all__ = [
    "SemanticEntropyCluster",
    "SemanticEntropyResult",
    "cluster_samples_by_entailment",
    "compute_semantic_entropy",
    "default_contradiction_threshold",
    "default_entailment_threshold",
    "default_max_samples",
    "is_semantic_entropy_enabled",
]
