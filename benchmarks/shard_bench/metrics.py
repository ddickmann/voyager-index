"""
Retrieval quality metrics: Recall@K, MRR@K, NDCG@K, MAP@K.

Supports both binary relevance (list of relevant IDs) and graded relevance
(dict mapping doc_id -> relevance grade).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

Relevance = Union[Sequence[int], Dict[int, int]]


def _to_grade_dict(relevant: Relevance) -> Dict[int, int]:
    if isinstance(relevant, dict):
        return relevant
    return {doc_id: 1 for doc_id in relevant}


def _to_set(relevant: Relevance) -> set:
    if isinstance(relevant, dict):
        return set(relevant.keys())
    return set(relevant)


def recall_at_k(
    retrieved_ids: Sequence[int],
    relevant: Relevance,
    k: int,
) -> float:
    relevant_set = _to_set(relevant)
    if not relevant_set:
        return 1.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & relevant_set) / min(k, len(relevant_set))


def mrr_at_k(
    retrieved_ids: Sequence[int],
    relevant: Relevance,
    k: int,
) -> float:
    relevant_set = _to_set(relevant)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[int],
    relevant: Relevance,
    k: int,
) -> float:
    """Graded NDCG@K. Falls back to binary if relevant is a list."""
    grades = _to_grade_dict(relevant)
    if not grades:
        return 1.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = grades.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(rank + 1)

    ideal_gains = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(r + 1) for r, g in enumerate(ideal_gains, start=1))
    if idcg == 0.0:
        return 1.0
    return dcg / idcg


def map_at_k(
    retrieved_ids: Sequence[int],
    relevant: Relevance,
    k: int,
) -> float:
    """Mean Average Precision at K (binary relevance)."""
    relevant_set = _to_set(relevant)
    if not relevant_set:
        return 1.0

    n_relevant_seen = 0
    sum_precisions = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            n_relevant_seen += 1
            sum_precisions += n_relevant_seen / rank

    n_relevant_total = min(k, len(relevant_set))
    if n_relevant_total == 0:
        return 0.0
    return sum_precisions / n_relevant_total


def compute_all_metrics(
    all_retrieved: List[List[int]],
    all_relevant: List[Relevance],
    ks: Tuple[int, ...] = (10, 100),
) -> dict:
    """Aggregate metrics over a query set. Supports graded relevance dicts."""
    results = {}
    for k in ks:
        recalls = [recall_at_k(r, g, k) for r, g in zip(all_retrieved, all_relevant)]
        results[f"recall@{k}"] = float(np.mean(recalls))

    for k in ks:
        ndcgs = [ndcg_at_k(r, g, k) for r, g in zip(all_retrieved, all_relevant)]
        results[f"NDCG@{k}"] = float(np.mean(ndcgs))

    for k in ks:
        maps = [map_at_k(r, g, k) for r, g in zip(all_retrieved, all_relevant)]
        results[f"MAP@{k}"] = float(np.mean(maps))

    mrrs = [mrr_at_k(r, g, 10) for r, g in zip(all_retrieved, all_relevant)]
    results["MRR@10"] = float(np.mean(mrrs))

    return results
