"""
Prepare individual BEIR dataset NPZ files for the voyager-index benchmark suite.

Encodes 6 standard BEIR datasets with GTE-ModernColBERT-v1 (128-dim multi-vector)
and saves per-dataset NPZ files with doc embeddings, query embeddings, and qrels.

Usage:
    python benchmarks/data/prepare_beir_datasets.py
    python benchmarks/data/prepare_beir_datasets.py --datasets fiqa scifact
    python benchmarks/data/prepare_beir_datasets.py --batch-size 64
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"
CACHE_DIR = Path.home() / ".cache" / "voyager-qa" / "beir"

BEIR_DATASETS: Dict[str, Dict[str, Any]] = {
    "arguana": {
        "corpus_repo": "BeIR/arguana",
        "corpus_config": None,
        "qrels_repo": "mteb/arguana",
        "expected_docs": 8_674,
    },
    "fiqa": {
        "corpus_repo": "BeIR/fiqa",
        "corpus_config": None,
        "qrels_repo": "mteb/fiqa",
        "expected_docs": 57_638,
    },
    "nfcorpus": {
        "corpus_repo": "BeIR/nfcorpus",
        "corpus_config": None,
        "qrels_repo": "mteb/nfcorpus",
        "expected_docs": 3_633,
    },
    "quora": {
        "corpus_repo": "BeIR/quora",
        "corpus_config": None,
        "qrels_repo": "mteb/quora",
        "expected_docs": 23_301,
    },
    "scidocs": {
        "corpus_repo": "BeIR/scidocs",
        "corpus_config": None,
        "qrels_repo": "mteb/scidocs",
        "expected_docs": 25_657,
    },
    "scifact": {
        "corpus_repo": "BeIR/scifact",
        "corpus_config": None,
        "qrels_repo": "mteb/scifact",
        "expected_docs": 5_183,
    },
    # Added 2026-04-20 to enable the BEIR-8 head-to-head against FastPlaid
    # (`benchmarks/fast_plaid_head_to_head.py`). Both datasets are large
    # (171k / 382k docs) and their CPU-only `prepare_beir_datasets.py`
    # encode pass takes a noticeable chunk of wall-time on a workstation;
    # use `python benchmarks/data/prepare_beir_datasets.py --datasets
    # trec-covid webis-touche2020 --batch-size 64` on the GPU box where
    # the head-to-head will run.
    "trec-covid": {
        "corpus_repo": "BeIR/trec-covid",
        "corpus_config": None,
        "qrels_repo": "mteb/trec-covid",
        "expected_docs": 171_332,
    },
    "webis-touche2020": {
        "corpus_repo": "BeIR/webis-touche2020",
        "corpus_config": None,
        # 2026-04: `mteb/webis-touche2020` is gated/private (HF 401), so use
        # the canonical BeIR-side qrels repo, which exposes the same
        # (query-id, corpus-id, score) schema under a `test` split.
        "qrels_repo": "BeIR/webis-touche2020-qrels",
        "expected_docs": 382_545,
    },
}


def load_beir_split(
    name: str, spec: dict
) -> Tuple[List[str], List[str], Dict[int, Dict[int, int]]]:
    """Load corpus texts, query texts, and graded qrels for a BEIR split.

    Returns qrels as {query_idx: {doc_idx: relevance_grade}} to support
    graded MAP/NDCG computation.
    """
    from datasets import load_dataset

    log.info("Loading %s corpus from %s ...", name, spec["corpus_repo"])
    corpus_ds = load_dataset(spec["corpus_repo"], "corpus", split="corpus")

    log.info("Loading %s queries ...", name)
    queries_ds = load_dataset(spec["corpus_repo"], "queries", split="queries")

    log.info("Loading %s qrels from %s ...", name, spec["qrels_repo"])
    qrels_ds = load_dataset(spec["qrels_repo"], split="test")

    log.info("  corpus=%d, queries=%d, qrels=%d", len(corpus_ds), len(queries_ds), len(qrels_ds))

    corpus_id_to_text: Dict[str, str] = {}
    for row in corpus_ds:
        cid = str(row["_id"])
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        corpus_id_to_text[cid] = f"{title}\n{text}".strip() if title else text

    query_id_to_text: Dict[str, str] = {}
    for row in queries_ds:
        qid = str(row["_id"])
        query_id_to_text[qid] = row["text"]

    qrel_pairs: Dict[str, Dict[str, int]] = {}
    referenced_cids: set = set()
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = int(row.get("score", 1))
        if score >= 1:
            qrel_pairs.setdefault(qid, {})[cid] = score
            referenced_cids.add(cid)

    # By default keep the FULL corpus so the benchmark mirrors the
    # public BEIR cardinalities (e.g. quora=522 931). Set
    # ``BEIR_QRELS_FILTER=1`` to opt back into the qrels-only subset for
    # quick smoke-tests on small VRAM. The full corpus is required to
    # report apples-to-apples numbers vs PLAID / FastPlaid public
    # benchmarks (quora full ~522 k docs).
    qrels_filter = os.environ.get("BEIR_QRELS_FILTER", "0") == "1"
    if qrels_filter:
        all_cids = sorted(cid for cid in corpus_id_to_text if cid in referenced_cids)
        if len(all_cids) < len(corpus_id_to_text):
            log.info("  Filtered corpus from %d to %d docs (qrels-referenced only; BEIR_QRELS_FILTER=1)",
                     len(corpus_id_to_text), len(all_cids))
    else:
        all_cids = sorted(corpus_id_to_text.keys())
        unreferenced = len(all_cids) - sum(1 for c in all_cids if c in referenced_cids)
        log.info("  Keeping full corpus: %d docs (%d not in qrels; set BEIR_QRELS_FILTER=1 to subset)",
                 len(all_cids), unreferenced)
    cid_to_idx = {cid: i for i, cid in enumerate(all_cids)}
    doc_texts = [corpus_id_to_text[cid] for cid in all_cids]

    valid_qids = sorted(
        qid for qid in qrel_pairs
        if qid in query_id_to_text
        and any(cid in cid_to_idx for cid in qrel_pairs[qid])
    )
    query_texts = [query_id_to_text[qid] for qid in valid_qids]

    idx_qrels: Dict[int, Dict[int, int]] = {}
    for qi, qid in enumerate(valid_qids):
        mapped = {}
        for cid, score in qrel_pairs[qid].items():
            if cid in cid_to_idx:
                mapped[cid_to_idx[cid]] = score
        if mapped:
            idx_qrels[qi] = mapped

    log.info("  %s: %d docs, %d queries, %d qrels",
             name, len(doc_texts), len(query_texts), len(idx_qrels))
    return doc_texts, query_texts, idx_qrels


def embed_texts(
    doc_texts: List[str],
    query_texts: List[str],
    batch_size: int = 32,
    model_name: str = MODEL_NAME,
):
    from pylate import models

    log.info("Loading %s ...", model_name)
    model = models.ColBERT(model_name)

    log.info("Embedding %d documents (batch_size=%d) ...", len(doc_texts), batch_size)
    t0 = time.time()
    doc_embs = model.encode(doc_texts, batch_size=batch_size, is_query=False, show_progress_bar=True)
    doc_vecs = [np.array(e, dtype=np.float16) for e in doc_embs]
    doc_elapsed = time.time() - t0
    log.info("  Document encoding: %.1fs (%.1f docs/s)", doc_elapsed, len(doc_texts) / doc_elapsed)

    log.info("Embedding %d queries ...", len(query_texts))
    t0 = time.time()
    query_embs = model.encode(query_texts, batch_size=batch_size, is_query=True, show_progress_bar=True)
    query_vecs = [np.array(e, dtype=np.float16) for e in query_embs]
    query_elapsed = time.time() - t0
    log.info("  Query encoding: %.1fs (%.1f q/s)", query_elapsed, len(query_texts) / max(query_elapsed, 0.001))

    return doc_vecs, query_vecs, doc_elapsed


def save_npz(path: Path, doc_vecs, query_vecs, idx_qrels: Dict[int, Dict[int, int]]):
    all_doc = np.vstack([v.astype(np.float16) for v in doc_vecs])
    offsets = []
    pos = 0
    for v in doc_vecs:
        offsets.append((pos, pos + len(v)))
        pos += len(v)

    all_q = np.vstack([v.astype(np.float16) for v in query_vecs])
    q_offsets = []
    pos = 0
    for v in query_vecs:
        q_offsets.append((pos, pos + len(v)))
        pos += len(v)

    max_rels = max((len(v) for v in idx_qrels.values()), default=1)
    qrels_mat = np.full((len(query_vecs), max_rels), -1, dtype=np.int32)
    grades_mat = np.zeros((len(query_vecs), max_rels), dtype=np.int32)
    for qi, rels in idx_qrels.items():
        for ri, (doc_idx, grade) in enumerate(sorted(rels.items())):
            qrels_mat[qi, ri] = doc_idx
            grades_mat[qi, ri] = grade

    np.savez_compressed(
        path,
        doc_vectors=all_doc,
        doc_offsets=np.array(offsets, dtype=np.int64),
        query_vectors=all_q,
        query_offsets=np.array(q_offsets, dtype=np.int64),
        qrels=qrels_mat,
        qrel_grades=grades_mat,
        doc_ids=np.array([str(i) for i in range(len(doc_vecs))], dtype=object),
        n_docs=np.array(len(doc_vecs)),
        n_queries=np.array(len(query_vecs)),
        dim=np.array(all_doc.shape[1]),
    )
    log.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1024 / 1024)


def prepare_dataset(name: str, spec: dict, batch_size: int = 32, cache_dir: Path = CACHE_DIR):
    output_path = cache_dir / f"{name}.npz"
    if output_path.exists():
        log.info("Dataset %s already prepared at %s, skipping", name, output_path)
        return output_path

    doc_texts, query_texts, idx_qrels = load_beir_split(name, spec)
    doc_vecs, query_vecs, _ = embed_texts(doc_texts, query_texts, batch_size=batch_size)

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, doc_vecs, query_vecs, idx_qrels)

    total_tokens = sum(v.shape[0] for v in doc_vecs)
    log.info("  %s: %d docs, %d queries, %d total tokens, dim=%d",
             name, len(doc_vecs), len(query_vecs), total_tokens, doc_vecs[0].shape[1])
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare BEIR datasets for voyager-index benchmarks")
    parser.add_argument("--datasets", nargs="*", default=list(BEIR_DATASETS.keys()),
                        help="Datasets to prepare (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=str(CACHE_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for name in args.datasets:
        if name not in BEIR_DATASETS:
            log.error("Unknown dataset: %s. Available: %s", name, list(BEIR_DATASETS.keys()))
            continue
        log.info("=" * 60)
        log.info("Preparing %s ...", name)
        log.info("=" * 60)
        prepare_dataset(name, BEIR_DATASETS[name], batch_size=args.batch_size, cache_dir=output_dir)

    log.info("All datasets prepared in %s", output_dir)


if __name__ == "__main__":
    main()
