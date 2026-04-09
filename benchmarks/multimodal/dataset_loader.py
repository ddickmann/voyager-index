"""
Dataset loaders for multimodal evaluation benchmarks.

Provides abstract interface and concrete implementations for OKVQA, EVQA,
and synthetic datasets.  Real datasets expect pre-extracted embeddings in a
HuggingFace-style cache layout; the synthetic loader generates random data
for pipeline testing without external dependencies.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MultimodalDataset(ABC):
    """Abstract base for multimodal benchmark datasets."""

    @abstractmethod
    def load(self) -> None:
        """Load or generate the dataset."""
        ...

    @abstractmethod
    def queries(self) -> List[np.ndarray]:
        """Return list of query embedding matrices (n_tokens, dim)."""
        ...

    @abstractmethod
    def documents(self) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        """Return (all_vectors, doc_ids, offsets) suitable for GemSegment.build()."""
        ...

    @abstractmethod
    def ground_truth(self) -> Dict[int, List[int]]:
        """Return mapping from query index to list of relevant doc ids."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBEDDING_INSTRUCTIONS = """\
Pre-extracted embeddings not found at: {path}

To generate them, run the embedding extraction script:

    python -m benchmarks.multimodal.extract_embeddings \\
        --dataset {dataset} --split {split} --output_dir {data_dir}

Expected layout:
    {data_dir}/{dataset}/{split}/queries.npy    — (n_queries, max_tokens, dim) float32
    {data_dir}/{dataset}/{split}/documents.npy  — (total_tokens, dim) float32
    {data_dir}/{dataset}/{split}/doc_offsets.npy — (n_docs, 2) int64  [start, end)
    {data_dir}/{dataset}/{split}/qrels.json     — {{query_idx: [doc_id, ...]}}
"""


def _load_cached_dataset(
    data_dir: str,
    dataset_name: str,
    split: str,
    max_queries: Optional[int],
) -> Tuple[List[np.ndarray], np.ndarray, List[int], List[Tuple[int, int]], Dict[int, List[int]]]:
    """Load pre-extracted embeddings from the cache directory."""

    base = Path(data_dir) / dataset_name / split

    queries_path = base / "queries.npy"
    docs_path = base / "documents.npy"
    offsets_path = base / "doc_offsets.npy"
    qrels_path = base / "qrels.json"

    for p in (queries_path, docs_path, offsets_path, qrels_path):
        if not p.exists():
            raise FileNotFoundError(
                _EMBEDDING_INSTRUCTIONS.format(
                    path=p,
                    dataset=dataset_name,
                    split=split,
                    data_dir=data_dir,
                )
            )

    raw_queries = np.load(queries_path)  # (n_queries, max_tokens, dim)
    all_vectors = np.load(docs_path)     # (total_tokens, dim)
    raw_offsets = np.load(offsets_path)   # (n_docs, 2)

    with open(qrels_path) as f:
        raw_qrels: Dict[str, List[int]] = json.load(f)

    doc_ids = list(range(len(raw_offsets)))
    offsets = [(int(s), int(e)) for s, e in raw_offsets]
    qrels = {int(k): v for k, v in raw_qrels.items()}

    queries = [raw_queries[i] for i in range(raw_queries.shape[0])]
    if max_queries is not None and len(queries) > max_queries:
        queries = queries[:max_queries]
        qrels = {k: v for k, v in qrels.items() if k < max_queries}

    return queries, all_vectors, doc_ids, offsets, qrels


# ---------------------------------------------------------------------------
# OKVQA
# ---------------------------------------------------------------------------

class OKVQADataset(MultimodalDataset):
    """OK-VQA dataset with pre-extracted multimodal embeddings."""

    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        max_queries: int = 1000,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.max_queries = max_queries

        self._queries: List[np.ndarray] = []
        self._all_vectors: Optional[np.ndarray] = None
        self._doc_ids: List[int] = []
        self._offsets: List[Tuple[int, int]] = []
        self._qrels: Dict[int, List[int]] = {}

    def load(self) -> None:
        self._queries, self._all_vectors, self._doc_ids, self._offsets, self._qrels = (
            _load_cached_dataset(self.data_dir, "okvqa", self.split, self.max_queries)
        )

    def queries(self) -> List[np.ndarray]:
        return self._queries

    def documents(self) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        assert self._all_vectors is not None, "Call load() first"
        return self._all_vectors, self._doc_ids, self._offsets

    def ground_truth(self) -> Dict[int, List[int]]:
        return self._qrels


# ---------------------------------------------------------------------------
# EVQA
# ---------------------------------------------------------------------------

class EVQADataset(MultimodalDataset):
    """Encyclopedic-VQA dataset with pre-extracted multimodal embeddings."""

    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        max_queries: int = 1000,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.max_queries = max_queries

        self._queries: List[np.ndarray] = []
        self._all_vectors: Optional[np.ndarray] = None
        self._doc_ids: List[int] = []
        self._offsets: List[Tuple[int, int]] = []
        self._qrels: Dict[int, List[int]] = {}

    def load(self) -> None:
        self._queries, self._all_vectors, self._doc_ids, self._offsets, self._qrels = (
            _load_cached_dataset(self.data_dir, "evqa", self.split, self.max_queries)
        )

    def queries(self) -> List[np.ndarray]:
        return self._queries

    def documents(self) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        assert self._all_vectors is not None, "Call load() first"
        return self._all_vectors, self._doc_ids, self._offsets

    def ground_truth(self) -> Dict[int, List[int]]:
        return self._qrels


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

class SyntheticMultimodalDataset(MultimodalDataset):
    """Random synthetic data for pipeline testing without real datasets."""

    def __init__(
        self,
        n_docs: int = 500,
        n_queries: int = 50,
        dim: int = 128,
        vecs_per_doc: int = 32,
        vecs_per_query: int = 32,
        seed: int = 42,
    ) -> None:
        self.n_docs = n_docs
        self.n_queries = n_queries
        self.dim = dim
        self.vecs_per_doc = vecs_per_doc
        self.vecs_per_query = vecs_per_query
        self.seed = seed

        self._queries: List[np.ndarray] = []
        self._all_vectors: Optional[np.ndarray] = None
        self._doc_ids: List[int] = []
        self._offsets: List[Tuple[int, int]] = []

    def load(self) -> None:
        rng = np.random.RandomState(self.seed)

        self._all_vectors = rng.randn(
            self.n_docs * self.vecs_per_doc, self.dim,
        ).astype(np.float32)

        self._doc_ids = list(range(self.n_docs))
        self._offsets = [
            (i * self.vecs_per_doc, (i + 1) * self.vecs_per_doc)
            for i in range(self.n_docs)
        ]

        self._queries = [
            rng.randn(self.vecs_per_query, self.dim).astype(np.float32)
            for _ in range(self.n_queries)
        ]

    def queries(self) -> List[np.ndarray]:
        return self._queries

    def documents(self) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        assert self._all_vectors is not None, "Call load() first"
        return self._all_vectors, self._doc_ids, self._offsets

    def ground_truth(self) -> Dict[int, List[int]]:
        return {}
