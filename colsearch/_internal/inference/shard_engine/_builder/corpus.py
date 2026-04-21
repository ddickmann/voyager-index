"""Corpus loading helpers for offline shard builds."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
DEFAULT_NPZ = Path.home() / ".cache" / "voyager-qa" / "beir_100k.npz"

def _mem_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1.0

def load_corpus(npz_path: Path, max_docs: int = 0):
    log.info("Loading corpus from %s ...", npz_path)
    npz = np.load(str(npz_path), allow_pickle=True)
    doc_offsets_arr = npz["doc_offsets"]
    n_docs = int(npz["n_docs"])
    dim = int(npz["dim"])
    if 0 < max_docs < n_docs:
        n_docs = max_docs
        doc_offsets_arr = doc_offsets_arr[:n_docs]
        log.info("Truncating to %d docs", n_docs)
    last_vec = int(doc_offsets_arr[-1][1])
    all_vectors = npz["doc_vectors"][:last_vec]
    doc_offsets = [(int(s), int(e)) for s, e in doc_offsets_arr]
    doc_ids = list(range(n_docs))
    query_offsets = npz["query_offsets"]
    all_q = npz["query_vectors"]
    query_vecs = [all_q[int(s):int(e)].astype(np.float32) for s, e in query_offsets]
    qrels_mat = npz["qrels"]
    qrels = {}
    for qi in range(qrels_mat.shape[0]):
        rels = [int(x) for x in qrels_mat[qi] if 0 <= x < n_docs]
        if rels:
            qrels[qi] = rels
    tok_counts = [e - s for s, e in doc_offsets]
    log.info(
        "Corpus loaded: %d docs, %d vectors, dim=%d, tokens/doc mean=%.0f p50=%.0f p95=%.0f, RSS=%.1f GB",
        n_docs, int(all_vectors.shape[0]), dim,
        np.mean(tok_counts), np.median(tok_counts), np.percentile(tok_counts, 95),
        _mem_gb(),
    )
    return all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim

