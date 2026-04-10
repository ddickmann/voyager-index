"""
MS MARCO Document Ranking dataset loader for Graph QA tests.

Downloads pre-embedded ColBERT-Zero embeddings from HuggingFace Hub,
caches locally, and provides a structured dataset object.

Falls back to synthetic clustered data if download fails.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

HF_REPO = "latence-ai/voyager-msmarco-doc-2k5-colbert-zero"
NPZ_FILENAME = "msmarco_doc_2k5.npz"
CQADUPSTACK_NPZ = "cqadupstack_english.npz"
BEIR_100K_NPZ = "beir_100k.npz"
CACHE_DIR = Path(os.environ.get("VOYAGER_QA_CACHE", Path.home() / ".cache" / "voyager-qa"))


@dataclass
class MSMARCODataset:
    """Pre-embedded MS MARCO Document Ranking dataset."""

    doc_vecs: List[np.ndarray]
    doc_ids: List[int]
    query_vecs: List[np.ndarray]
    qrels: Dict[int, List[int]]
    all_vectors: np.ndarray
    offsets: List[Tuple[int, int]]
    query_offsets: List[Tuple[int, int]]
    dim: int
    is_synthetic: bool = False
    _subset_cache: Dict[int, "MSMARCODataset"] = field(default_factory=dict, repr=False)

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def n_queries(self) -> int:
        return len(self.query_vecs)

    def subset(self, n_docs: int, seed: int = 42) -> "MSMARCODataset":
        """Return a random subset of n_docs documents, preserving qrels."""
        if n_docs in self._subset_cache:
            return self._subset_cache[n_docs]
        if n_docs >= self.n_docs:
            return self

        rng = np.random.RandomState(seed)

        relevant_doc_idxs = set()
        for rel_list in self.qrels.values():
            relevant_doc_idxs.update(rel_list)

        relevant_list = sorted(relevant_doc_idxs)
        if len(relevant_list) > n_docs:
            relevant_list = rng.choice(relevant_list, n_docs, replace=False).tolist()

        remaining = n_docs - len(relevant_list)
        if remaining > 0:
            pool = [i for i in range(self.n_docs) if i not in set(relevant_list)]
            extra = rng.choice(pool, min(remaining, len(pool)), replace=False).tolist()
            selected = sorted(set(relevant_list) | set(extra))
        else:
            selected = sorted(relevant_list[:n_docs])

        old_to_new = {old: new for new, old in enumerate(selected)}

        sub_doc_vecs = [self.doc_vecs[i] for i in selected]
        sub_doc_ids = list(range(len(selected)))

        sub_all = np.vstack(sub_doc_vecs)
        sub_offsets = []
        pos = 0
        for v in sub_doc_vecs:
            sub_offsets.append((pos, pos + len(v)))
            pos += len(v)

        sub_qrels: Dict[int, List[int]] = {}
        for qi, rel_list in self.qrels.items():
            mapped = [old_to_new[r] for r in rel_list if r in old_to_new]
            if mapped:
                sub_qrels[qi] = mapped

        result = MSMARCODataset(
            doc_vecs=sub_doc_vecs,
            doc_ids=sub_doc_ids,
            query_vecs=self.query_vecs,
            qrels=sub_qrels,
            all_vectors=sub_all,
            offsets=sub_offsets,
            query_offsets=self.query_offsets,
            dim=self.dim,
            is_synthetic=self.is_synthetic,
        )
        self._subset_cache[n_docs] = result
        return result


def _download_from_hub() -> Optional[Path]:
    """Download the .npz from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=NPZ_FILENAME,
            repo_type="dataset",
            cache_dir=str(CACHE_DIR),
            local_dir=str(CACHE_DIR),
        )
        return Path(local_path)
    except Exception as exc:
        log.warning("Failed to download dataset from HF Hub: %s", exc)
        return None


def _load_from_npz(path: Path) -> MSMARCODataset:
    """Load dataset from compressed .npz file."""
    data = np.load(path, allow_pickle=True)

    all_doc_vectors = data["doc_vectors"].astype(np.float32)
    doc_offsets_arr = data["doc_offsets"]
    all_query_vectors = data["query_vectors"].astype(np.float32)
    query_offsets_arr = data["query_offsets"]
    qrels_matrix = data["qrels"]
    dim = int(data["dim"])

    doc_vecs = []
    offsets = []
    for start, end in doc_offsets_arr:
        doc_vecs.append(all_doc_vectors[start:end])
        offsets.append((int(start), int(end)))

    query_vecs = []
    query_offsets = []
    for start, end in query_offsets_arr:
        query_vecs.append(all_query_vectors[start:end])
        query_offsets.append((int(start), int(end)))

    n_queries = len(query_vecs)
    qrels: Dict[int, List[int]] = {}
    for qi in range(n_queries):
        rel = [int(x) for x in qrels_matrix[qi] if x >= 0]
        if rel:
            qrels[qi] = rel

    doc_ids = list(range(len(doc_vecs)))

    return MSMARCODataset(
        doc_vecs=doc_vecs,
        doc_ids=doc_ids,
        query_vecs=query_vecs,
        qrels=qrels,
        all_vectors=all_doc_vectors,
        offsets=offsets,
        query_offsets=query_offsets,
        dim=dim,
        is_synthetic=False,
    )


def generate_synthetic_dataset(
    n_docs: int = 2500,
    dim: int = 128,
    tokens_per_doc: int = 64,
    n_queries: int = 200,
    tokens_per_query: int = 32,
    n_clusters: int = 16,
    seed: int = 42,
) -> MSMARCODataset:
    """Generate synthetic clustered dataset as fallback."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32) * 3.0

    doc_vecs = []
    all_vecs_list = []
    offsets = []
    pos = 0
    for i in range(n_docs):
        cluster = i % n_clusters
        vecs = (centers[cluster] + rng.randn(tokens_per_doc, dim).astype(np.float32) * 0.3)
        doc_vecs.append(vecs)
        all_vecs_list.append(vecs)
        offsets.append((pos, pos + tokens_per_doc))
        pos += tokens_per_doc

    all_vectors = np.vstack(all_vecs_list).astype(np.float32)

    query_vecs = []
    query_offsets = []
    qpos = 0
    for i in range(n_queries):
        cluster = rng.randint(0, n_clusters)
        q = centers[cluster] + rng.randn(tokens_per_query, dim).astype(np.float32) * 0.3
        query_vecs.append(q)
        query_offsets.append((qpos, qpos + tokens_per_query))
        qpos += tokens_per_query

    qrels: Dict[int, List[int]] = {}
    for qi in range(n_queries):
        cluster = qi % n_clusters
        relevant = [d for d in range(n_docs) if d % n_clusters == cluster]
        qrels[qi] = rng.choice(relevant, min(3, len(relevant)), replace=False).tolist()

    return MSMARCODataset(
        doc_vecs=doc_vecs,
        doc_ids=list(range(n_docs)),
        query_vecs=query_vecs,
        qrels=qrels,
        all_vectors=all_vectors,
        offsets=offsets,
        query_offsets=query_offsets,
        dim=dim,
        is_synthetic=True,
    )


def _merge_datasets(a: MSMARCODataset, b: MSMARCODataset) -> MSMARCODataset:
    """Merge two datasets into one combined corpus with both query sets."""
    assert a.dim == b.dim, f"Dimension mismatch: {a.dim} vs {b.dim}"

    doc_id_offset = a.n_docs
    vec_offset = a.all_vectors.shape[0]
    query_offset = a.n_queries

    merged_doc_vecs = a.doc_vecs + b.doc_vecs
    merged_doc_ids = a.doc_ids + [did + doc_id_offset for did in b.doc_ids]
    merged_all_vectors = np.vstack([a.all_vectors, b.all_vectors])
    merged_offsets = list(a.offsets) + [
        (s + vec_offset, e + vec_offset) for s, e in b.offsets
    ]

    merged_query_vecs = a.query_vecs + b.query_vecs
    merged_query_offsets = list(a.query_offsets)
    q_vec_offset = a.all_vectors.shape[0] if not a.query_offsets else 0
    for s, e in b.query_offsets:
        merged_query_offsets.append((s + (a.query_offsets[-1][1] if a.query_offsets else 0),
                                     e + (a.query_offsets[-1][1] if a.query_offsets else 0)))

    merged_qrels: Dict[int, List[int]] = dict(a.qrels)
    for qi, rels in b.qrels.items():
        merged_qrels[qi + query_offset] = [r + doc_id_offset for r in rels]

    return MSMARCODataset(
        doc_vecs=merged_doc_vecs,
        doc_ids=merged_doc_ids,
        query_vecs=merged_query_vecs,
        qrels=merged_qrels,
        all_vectors=merged_all_vectors,
        offsets=merged_offsets,
        query_offsets=merged_query_offsets,
        dim=a.dim,
        is_synthetic=False,
    )


def load_msmarco_dataset() -> MSMARCODataset:
    """Load MS MARCO dataset, falling back to synthetic if unavailable."""
    cached = CACHE_DIR / NPZ_FILENAME
    if cached.exists():
        log.info("Loading cached dataset from %s", cached)
        return _load_from_npz(cached)

    path = _download_from_hub()
    if path is not None and path.exists():
        log.info("Loading downloaded dataset from %s", path)
        return _load_from_npz(path)

    log.warning(
        "MS MARCO dataset unavailable — using synthetic fallback. "
        "Run benchmarks/data/prepare_msmarco_fast.py to create the real dataset."
    )
    return generate_synthetic_dataset()


def load_cqadupstack_dataset() -> Optional[MSMARCODataset]:
    """Load CQADupStack English dataset if available."""
    cached = CACHE_DIR / CQADUPSTACK_NPZ
    if cached.exists():
        log.info("Loading CQADupStack from %s", cached)
        return _load_from_npz(cached)
    return None


def load_beir_100k_dataset() -> MSMARCODataset:
    """Load the 100K BeIR dataset (CQADupStack + FiQA + MS MARCO).

    Falls back to the 7.5K combined dataset if the 100K NPZ is not found.
    Run `python -m benchmarks.data.prepare_beir_100k` to create it.
    """
    cached = CACHE_DIR / BEIR_100K_NPZ
    if cached.exists():
        log.info("Loading BeIR 100K dataset from %s", cached)
        return _load_from_npz(cached)
    log.warning(
        "BeIR 100K dataset not found at %s — falling back to combined 7.5K dataset. "
        "Run: python -m benchmarks.data.prepare_beir_100k",
        cached,
    )
    return load_combined_dataset()


def load_combined_dataset() -> MSMARCODataset:
    """Load MS MARCO + CQADupStack as a single merged corpus."""
    msmarco = load_msmarco_dataset()
    cqa = load_cqadupstack_dataset()
    if cqa is not None:
        log.info(
            "Merging MS MARCO (%d docs, %d queries) + CQADupStack (%d docs, %d queries)",
            msmarco.n_docs, msmarco.n_queries, cqa.n_docs, cqa.n_queries,
        )
        return _merge_datasets(msmarco, cqa)
    log.info("CQADupStack not found — using MS MARCO only")
    return msmarco
