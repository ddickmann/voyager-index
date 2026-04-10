"""
Prepare 100K+ document dataset from multiple BeIR splits.

Collects full CQADupStack English (~45K docs), FiQA (~57K docs),
and existing MS MARCO (2.5K docs) into a single NPZ file suitable
for large-scale GEM index benchmarking.

Embedding model: lightonai/ColBERT-Zero (128-dim multi-vector).

Checkpoints individual datasets to allow resumption if interrupted.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "lightonai/ColBERT-Zero"
CACHE_DIR = Path.home() / ".cache" / "voyager-qa"
MSMARCO_NPZ = CACHE_DIR / "msmarco_doc_2k5.npz"

BEIR_SOURCES = {
    "cqadupstack_english": {
        "corpus_repo": "BeIR/cqadupstack",
        "corpus_config": "english",
        "qrels_repo": "mteb/cqadupstack-english",
    },
    "fiqa": {
        "corpus_repo": "BeIR/fiqa",
        "corpus_config": None,
        "qrels_repo": "mteb/fiqa",
    },
}


def _load_beir_dataset(repo: str, kind: str, config: Optional[str] = None):
    """Load a BeIR HuggingFace dataset, handling two common config patterns."""
    from datasets import load_dataset

    if config is not None:
        return load_dataset(repo, config, split=kind)
    try:
        return load_dataset(repo, kind)
    except Exception:
        return load_dataset(repo, split=kind)


def load_beir_split(
    name: str, spec: dict
) -> Tuple[List[str], List[str], Dict[int, List[int]]]:
    """Load corpus texts, query texts, and index-mapped qrels for a BeIR split."""
    from datasets import load_dataset

    log.info("Loading %s corpus from %s ...", name, spec["corpus_repo"])
    corpus_ds = _load_beir_dataset(spec["corpus_repo"], "corpus", spec["corpus_config"])

    log.info("Loading %s queries ...", name)
    queries_ds = _load_beir_dataset(spec["corpus_repo"], "queries", spec["corpus_config"])

    log.info("Loading %s qrels from %s ...", name, spec["qrels_repo"])
    try:
        qrels_ds = load_dataset(spec["qrels_repo"], split="test")
    except Exception:
        qrels_ds = load_dataset(spec["qrels_repo"])

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

    qrel_pairs: Dict[str, List[str]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if row.get("score", 1) >= 1:
            qrel_pairs.setdefault(qid, []).append(cid)

    all_cids = sorted(corpus_id_to_text.keys())
    cid_to_idx = {cid: i for i, cid in enumerate(all_cids)}
    doc_texts = [corpus_id_to_text[cid] for cid in all_cids]

    valid_qids = sorted(
        qid for qid in qrel_pairs
        if qid in query_id_to_text
        and any(cid in cid_to_idx for cid in qrel_pairs[qid])
    )
    query_texts = [query_id_to_text[qid] for qid in valid_qids]

    idx_qrels: Dict[int, List[int]] = {}
    for qi, qid in enumerate(valid_qids):
        mapped = [cid_to_idx[cid] for cid in qrel_pairs[qid] if cid in cid_to_idx]
        if mapped:
            idx_qrels[qi] = mapped

    log.info("  %s: %d docs, %d queries, %d qrels",
             name, len(doc_texts), len(query_texts), len(idx_qrels))
    return doc_texts, query_texts, idx_qrels


def embed_texts(doc_texts: List[str], query_texts: List[str], batch_size: int = 32):
    """Embed with ColBERT-Zero via pylate."""
    from pylate import models

    log.info("Loading %s ...", MODEL_NAME)
    model = models.ColBERT(MODEL_NAME)

    log.info("Embedding %d documents (batch_size=%d) ...", len(doc_texts), batch_size)
    doc_embs = model.encode(doc_texts, batch_size=batch_size, is_query=False, show_progress_bar=True)
    doc_vecs = [np.array(e, dtype=np.float16) for e in doc_embs]

    log.info("Embedding %d queries ...", len(query_texts))
    query_embs = model.encode(query_texts, batch_size=batch_size, is_query=True, show_progress_bar=True)
    query_vecs = [np.array(e, dtype=np.float16) for e in query_embs]

    return doc_vecs, query_vecs


def save_npz(path: Path, doc_vecs, query_vecs, idx_qrels):
    """Save dataset in the standard voyager-qa NPZ format."""
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
    for qi, rels in idx_qrels.items():
        for ri, idx in enumerate(rels):
            qrels_mat[qi, ri] = idx

    np.savez_compressed(
        path,
        doc_vectors=all_doc,
        doc_offsets=np.array(offsets, dtype=np.int64),
        query_vectors=all_q,
        query_offsets=np.array(q_offsets, dtype=np.int64),
        qrels=qrels_mat,
        doc_ids=np.array([str(i) for i in range(len(doc_vecs))], dtype=object),
        n_docs=np.array(len(doc_vecs)),
        n_queries=np.array(len(query_vecs)),
        dim=np.array(all_doc.shape[1]),
    )
    log.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1024 / 1024)


def load_existing_msmarco() -> Optional[Tuple[List[np.ndarray], List[np.ndarray], Dict[int, List[int]]]]:
    """Load existing MS MARCO embeddings from cache."""
    if not MSMARCO_NPZ.exists():
        log.warning("MS MARCO NPZ not found at %s, skipping", MSMARCO_NPZ)
        return None

    data = np.load(str(MSMARCO_NPZ), allow_pickle=True)
    doc_vectors = data["doc_vectors"]
    doc_offsets = data["doc_offsets"]
    query_vectors = data["query_vectors"]
    query_offsets = data["query_offsets"]
    qrels_mat = data["qrels"]

    doc_vecs = [doc_vectors[s:e] for s, e in doc_offsets]
    query_vecs = [query_vectors[s:e] for s, e in query_offsets]

    idx_qrels: Dict[int, List[int]] = {}
    for qi in range(qrels_mat.shape[0]):
        rels = [int(r) for r in qrels_mat[qi] if r >= 0]
        if rels:
            idx_qrels[qi] = rels

    log.info("Loaded MS MARCO: %d docs, %d queries", len(doc_vecs), len(query_vecs))
    return doc_vecs, query_vecs, idx_qrels


def merge_parts(parts: List[Tuple[List[np.ndarray], List[np.ndarray], Dict[int, List[int]]]]):
    """Merge multiple (doc_vecs, query_vecs, idx_qrels) tuples, remapping IDs."""
    all_doc_vecs = []
    all_query_vecs = []
    all_qrels: Dict[int, List[int]] = {}
    doc_offset = 0
    query_offset = 0

    for doc_vecs, query_vecs, idx_qrels in parts:
        all_doc_vecs.extend(doc_vecs)
        all_query_vecs.extend(query_vecs)
        for qi, rels in idx_qrels.items():
            all_qrels[qi + query_offset] = [r + doc_offset for r in rels]
        doc_offset += len(doc_vecs)
        query_offset += len(query_vecs)

    log.info("Merged: %d docs, %d queries, %d qrels",
             len(all_doc_vecs), len(all_query_vecs), len(all_qrels))
    return all_doc_vecs, all_query_vecs, all_qrels


def main():
    parser = argparse.ArgumentParser(description="Prepare 100K BeIR dataset")
    parser.add_argument("--output", default=str(CACHE_DIR / "beir_100k.npz"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-msmarco", action="store_true")
    parser.add_argument("--add-scidocs", action="store_true",
                        help="Also include SciDocs (~25K docs)")
    args = parser.parse_args()

    if args.add_scidocs:
        BEIR_SOURCES["scidocs"] = {
            "corpus_repo": "BeIR/scidocs",
            "corpus_config": None,
            "qrels_repo": "mteb/scidocs",
        }

    parts = []

    if not args.skip_msmarco:
        msmarco = load_existing_msmarco()
        if msmarco is not None:
            parts.append(msmarco)

    for name, spec in BEIR_SOURCES.items():
        checkpoint = CACHE_DIR / f"_beir_part_{name}.npz"
        if checkpoint.exists():
            log.info("Loading checkpoint for %s from %s", name, checkpoint)
            cp = np.load(str(checkpoint), allow_pickle=True)
            doc_vectors = cp["doc_vectors"]
            doc_offsets_arr = cp["doc_offsets"]
            query_vectors = cp["query_vectors"]
            query_offsets_arr = cp["query_offsets"]
            qrels_mat = cp["qrels"]

            doc_vecs = [doc_vectors[s:e] for s, e in doc_offsets_arr]
            query_vecs = [query_vectors[s:e] for s, e in query_offsets_arr]
            idx_qrels: Dict[int, List[int]] = {}
            for qi in range(qrels_mat.shape[0]):
                rels = [int(r) for r in qrels_mat[qi] if r >= 0]
                if rels:
                    idx_qrels[qi] = rels

            log.info("  Checkpoint: %d docs, %d queries", len(doc_vecs), len(query_vecs))
            parts.append((doc_vecs, query_vecs, idx_qrels))
            continue

        doc_texts, query_texts, idx_qrels = load_beir_split(name, spec)
        doc_vecs, query_vecs = embed_texts(doc_texts, query_texts, batch_size=args.batch_size)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_npz(checkpoint, doc_vecs, query_vecs, idx_qrels)
        parts.append((doc_vecs, query_vecs, idx_qrels))

    merged_doc, merged_q, merged_qrels = merge_parts(parts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, merged_doc, merged_q, merged_qrels)

    total_tokens = sum(v.shape[0] for v in merged_doc)
    log.info("Final: %d docs, %d queries, %d total tokens, dim=%d",
             len(merged_doc), len(merged_q), total_tokens, merged_doc[0].shape[1])
    log.info("Done. Output: %s", out)


if __name__ == "__main__":
    main()
