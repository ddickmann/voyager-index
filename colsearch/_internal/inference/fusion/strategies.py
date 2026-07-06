"""
Result Fusion Strategies

Implements various strategies for combining results from multiple search engines.

Supported strategies:
- Reciprocal Rank Fusion (RRF): Robust, no normalization needed
- Weighted Sum: Flexible, requires normalization
- Max/Min: Simple aggregation
- Borda Count: Rank-based voting

Author: ColBERT Team
License: Apache-2.0
"""

from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

from ..config import FusionConfig
from ..engines.base import SearchResult


def normalize_min_max(scores: List[float]) -> List[float]:
    """
    Min-max normalization: Scale scores to [0, 1].

    Args:
        scores: List of scores.

    Returns:
        Normalized scores.

    Example:
        >>> normalize_min_max([1.0, 2.0, 3.0])
        [0.0, 0.5, 1.0]
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def normalize_z_score(scores: List[float]) -> List[float]:
    """
    Z-score normalization: Standardize to mean=0, std=1.

    Args:
        scores: List of scores.

    Returns:
        Normalized scores.

    Example:
        >>> normalize_z_score([1.0, 2.0, 3.0])
        [-1.224..., 0.0, 1.224...]
    """
    if not scores:
        return []

    mean = np.mean(scores)
    std = np.std(scores)

    if std == 0:
        return [0.0] * len(scores)

    return [(s - mean) / std for s in scores]


def normalize_softmax(scores: List[float]) -> List[float]:
    """
    Softmax normalization: Convert to probability distribution.

    Args:
        scores: List of scores.

    Returns:
        Normalized scores (sum to 1.0).

    Example:
        >>> normalize_softmax([1.0, 2.0, 3.0])
        [0.090..., 0.244..., 0.665...]
    """
    if not scores:
        return []

    exp_scores = np.exp(np.array(scores) - np.max(scores))  # Subtract max for stability
    return (exp_scores / exp_scores.sum()).tolist()


def reciprocal_rank_fusion(
    results_dict: Dict[str, List[SearchResult]],
    config: FusionConfig
) -> List[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF).

    Combines rankings from multiple engines using reciprocal ranks.
    Robust and doesn't require score normalization.

    Formula: score(doc) = Σ 1/(k + rank_i)
    where k is a constant (typically 60) and rank_i is the rank from engine i.

    Args:
        results_dict: Dictionary mapping engine names to result lists.
        config: Fusion configuration.

    Returns:
        Fused and ranked results.

    Reference:
        Cormack, Gordon V., et al. "Reciprocal rank fusion outperforms
        condorcet and individual rank learning methods." SIGIR 2009.

    Example:
        >>> results = {
        ...     'colbert': [SearchResult(doc_id=1, score=0.9, rank=1, source='colbert')],
        ...     'bm25': [SearchResult(doc_id=1, score=15.0, rank=1, source='bm25')]
        ... }
        >>> fused = reciprocal_rank_fusion(results, config)
    """
    k = config.rrf_k
    doc_scores = defaultdict(float)
    doc_engines = defaultdict(list)

    # Aggregate scores from all engines
    for engine_name, results in results_dict.items():
        for result in results:
            doc_id = result.doc_id
            rank = result.rank

            # RRF formula
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] += rrf_score
            doc_engines[doc_id].append(engine_name)

    # Create fused results
    fused_results = []
    for doc_id, score in doc_scores.items():
        fused_results.append(SearchResult(
            doc_id=doc_id,
            score=score,
            rank=0,  # Will be set after sorting
            source='fused_rrf',
            metadata={'engines': doc_engines[doc_id]}
        ))

    # Sort by score (descending) and assign ranks
    fused_results.sort(key=lambda x: x.score, reverse=True)
    for i, result in enumerate(fused_results):
        result.rank = i + 1

    # Return top_k
    return fused_results[:config.top_k]


def weighted_sum_fusion(
    results_dict: Dict[str, List[SearchResult]],
    config: FusionConfig
) -> List[SearchResult]:
    """
    Weighted sum fusion with score normalization.

    Combines scores from multiple engines using weighted average.
    Requires score normalization to make scores comparable.

    Formula: score(doc) = Σ weight_i × normalized_score_i

    Args:
        results_dict: Dictionary mapping engine names to result lists.
        config: Fusion configuration (must include weights).

    Returns:
        Fused and ranked results.

    Raises:
        ValueError: If weights are not provided.

    Example:
        >>> config = FusionConfig(
        ...     strategy='weighted_sum',
        ...     weights={'colbert': 0.7, 'bm25': 0.3},
        ...     normalization='min_max'
        ... )
        >>> fused = weighted_sum_fusion(results, config)
    """
    if config.weights is None:
        raise ValueError("weighted_sum requires weights in config")

    # Normalize scores for each engine
    normalized_results = {}
    for engine_name, results in results_dict.items():
        scores = [r.score for r in results]

        if config.normalization == 'min_max':
            norm_scores = normalize_min_max(scores)
        elif config.normalization == 'z_score':
            norm_scores = normalize_z_score(scores)
        elif config.normalization == 'softmax':
            norm_scores = normalize_softmax(scores)
        else:
            norm_scores = scores

        normalized_results[engine_name] = [
            SearchResult(
                doc_id=r.doc_id,
                score=norm_scores[i],
                rank=r.rank,
                source=r.source,
                metadata=r.metadata
            )
            for i, r in enumerate(results)
        ]

    # Aggregate weighted scores
    doc_scores = defaultdict(float)
    doc_engines = defaultdict(list)

    for engine_name, results in normalized_results.items():
        weight = config.weights.get(engine_name, 0.0)

        for result in results:
            doc_id = result.doc_id
            doc_scores[doc_id] += weight * result.score
            doc_engines[doc_id].append(engine_name)

    # Create fused results
    fused_results = []
    for doc_id, score in doc_scores.items():
        fused_results.append(SearchResult(
            doc_id=doc_id,
            score=score,
            rank=0,
            source='fused_weighted',
            metadata={'engines': doc_engines[doc_id]}
        ))

    # Sort and assign ranks
    fused_results.sort(key=lambda x: x.score, reverse=True)
    for i, result in enumerate(fused_results):
        result.rank = i + 1

    return fused_results[:config.top_k]


def max_fusion(
    results_dict: Dict[str, List[SearchResult]],
    config: FusionConfig
) -> List[SearchResult]:
    """
    Max score fusion.

    For each document, take the maximum score across all engines.

    Args:
        results_dict: Dictionary mapping engine names to result lists.
        config: Fusion configuration.

    Returns:
        Fused and ranked results.
    """
    doc_scores = {}
    doc_engines = defaultdict(list)

    for engine_name, results in results_dict.items():
        for result in results:
            doc_id = result.doc_id

            if doc_id not in doc_scores or result.score > doc_scores[doc_id]:
                doc_scores[doc_id] = result.score

            doc_engines[doc_id].append(engine_name)

    # Create fused results
    fused_results = [
        SearchResult(
            doc_id=doc_id,
            score=score,
            rank=0,
            source='fused_max',
            metadata={'engines': doc_engines[doc_id]}
        )
        for doc_id, score in doc_scores.items()
    ]

    # Sort and assign ranks
    fused_results.sort(key=lambda x: x.score, reverse=True)
    for i, result in enumerate(fused_results):
        result.rank = i + 1

    return fused_results[:config.top_k]


def min_fusion(
    results_dict: Dict[str, List[SearchResult]],
    config: FusionConfig
) -> List[SearchResult]:
    """
    Min score fusion.

    For each document, take the minimum score across all engines.

    Args:
        results_dict: Dictionary mapping engine names to result lists.
        config: Fusion configuration.

    Returns:
        Fused and ranked results.
    """
    doc_scores = {}
    doc_engines = defaultdict(list)

    for engine_name, results in results_dict.items():
        for result in results:
            doc_id = result.doc_id

            if doc_id not in doc_scores or result.score < doc_scores[doc_id]:
                doc_scores[doc_id] = result.score

            doc_engines[doc_id].append(engine_name)

    # Create fused results
    fused_results = [
        SearchResult(
            doc_id=doc_id,
            score=score,
            rank=0,
            source='fused_min',
            metadata={'engines': doc_engines[doc_id]}
        )
        for doc_id, score in doc_scores.items()
    ]

    # Sort and assign ranks
    fused_results.sort(key=lambda x: x.score, reverse=True)
    for i, result in enumerate(fused_results):
        result.rank = i + 1

    return fused_results[:config.top_k]


# Strategy registry
FUSION_STRATEGIES: Dict[str, Callable] = {
    'rrf': reciprocal_rank_fusion,
    'reciprocal_rank': reciprocal_rank_fusion,
    'weighted_sum': weighted_sum_fusion,
    'max': max_fusion,
    'min': min_fusion
}


def fuse_results(
    results_dict: Dict[str, List[SearchResult]],
    config: FusionConfig
) -> List[SearchResult]:
    """
    Fuse results from multiple engines using configured strategy.

    This is the main entry point for result fusion.

    Args:
        results_dict: Dictionary mapping engine names to result lists.
        config: Fusion configuration.

    Returns:
        Fused and ranked results.

    Raises:
        ValueError: If strategy is not supported.

    Example:
        >>> results = {
        ...     'colbert': [...],
        ...     'bm25': [...]
        ... }
        >>> config = FusionConfig(strategy='rrf', top_k=10)
        >>> fused = fuse_results(results, config)
    """
    strategy = config.strategy

    if strategy not in FUSION_STRATEGIES:
        raise ValueError(f"Unknown fusion strategy: {strategy}. "
                        f"Supported: {list(FUSION_STRATEGIES.keys())}")

    fusion_func = FUSION_STRATEGIES[strategy]
    results = fusion_func(results_dict, config)

    # Apply minimum score filter if specified
    if config.min_score is not None:
        results = [r for r in results if r.score >= config.min_score]

    return results


__all__ = [
    'reciprocal_rank_fusion',
    'weighted_sum_fusion',
    'max_fusion',
    'min_fusion',
    'fuse_results',
    'normalize_min_max',
    'normalize_z_score',
    'normalize_softmax'
]



