#!/usr/bin/env python3
"""
Hybrid retrieval benchmark: RRF (BM25 + dense) vs Tabu/knapsack on the same candidate pool.

Dataset: BEIR SciFact (public; download on first run). Embedding model: sentence-transformers
`all-MiniLM-L6-v2` (384-dim) unless overridden.

Dependencies (install separately):
  pip install sentence-transformers

Optional (Wilcoxon p-value):
  pip install scipy

Usage:
  python tools/benchmarks/benchmark_hybrid_rrf_vs_tabu.py --output-json /tmp/results.json

System Python (recommended when the repo .venv has CPU-only PyTorch but system has CUDA):
  ./tools/benchmarks/run_benchmark_system_python.sh --large-chunk-pool --output-json /tmp/large.json
  # or explicitly: /usr/bin/python3 tools/benchmarks/benchmark_hybrid_rrf_vs_tabu.py ...

Large candidate pool (Tabu shines with many chunks: redundancy + budget vs rank-only RRF):
  python tools/benchmarks/benchmark_hybrid_rrf_vs_tabu.py --large-chunk-pool --output-json /tmp/large.json

That mode raises --top-m (min 250), allows up to ~64 selected chunks per query, larger token
budget, more solver iterations, and reports nDCG/MRR at K=10 and K=50.

Fair protocol (cross-encoder): --use-cross-encoder scores the pool once; ce_sort lists by CE;
tabu_ce uses the same CE scores as relevance and CE-ordered tail—compare Δ(tabu_ce − ce_sort)
for packing vs listwise CE order. --calibration minmax|zscore|none calibrates BM25/dense per
query before the heuristic mix. --lexical-redundancy-weight blends Jaccard overlap with embedding
similarity for QKP redundancy. --lambda-ablation runs tabu_ce with lambda_=0 vs configured λ.

GPU: uses CUDA or Apple MPS for sentence-transformers when available; Tabu ``use_gpu`` when
``latence_solver.gpu_available()``. Use ``--cpu`` to force CPU for models and Tabu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

# Repo root: tools/benchmarks -> voyager-index
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# latence_solver (Rust wheel or source tree)
_KNAP_PY = _REPO_ROOT / "src" / "kernels" / "knapsack_solver" / "python"
if _KNAP_PY.is_dir() and str(_KNAP_PY) not in sys.path:
    sys.path.insert(0, str(_KNAP_PY))

from voyager_index._internal.inference.config import BM25Config, FusionConfig
from voyager_index._internal.inference.engines.base import SearchResult
from voyager_index._internal.inference.engines.bm25 import BM25Engine
from voyager_index._internal.inference.fusion.strategies import fuse_results
from voyager_index._internal.inference.index_core.hnsw_manager import HnswSegmentManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("benchmark_hybrid_rrf_vs_tabu")

SCIFACT_ZIP_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Small cross-encoder; BEIR-style reranking (Nogueira & Cho). Optional --use-cross-encoder.
DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def normalize_torch_device(device: str) -> str:
    """Use an indexed CUDA id so HF/sentence-transformers place modules consistently."""
    s = device.strip()
    d = s.lower()
    if d == "cuda":
        return "cuda:0"
    return s


def coerce_torch_device_to_available(requested: str) -> str:
    """Avoid AssertionError from CPU-only PyTorch (``Torch not compiled with CUDA enabled``).

    If ``cuda``/``cuda:N`` or ``mps`` is requested but that backend is unusable, fall back to ``cpu``.
    """
    s = normalize_torch_device(requested.strip())
    sl = s.lower()
    if sl == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if sl.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "Requested device %s but torch.cuda.is_available() is False "
                "(CPU-only PyTorch build or no CUDA driver). Using CPU for bi-encoder / cross-encoder.",
                s,
            )
            return "cpu"
        return s
    if sl == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            logger.warning("Requested MPS but it is not available; using CPU.")
            return "cpu"
        return s
    return s


def resolve_benchmark_devices(*, force_cpu: bool) -> tuple[str, bool]:
    """Pick torch device for bi-encoder / cross-encoder and whether Tabu requests GPU backend.

    Tabu GPU follows ``latence_solver.gpu_available()`` (may still run CPU reference if no CUDA backend).
    """
    if force_cpu:
        return "cpu", False
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda:0"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass
    use_tabu_gpu = False
    try:
        from latence_solver import gpu_available

        use_tabu_gpu = bool(gpu_available())
    except Exception:
        use_tabu_gpu = False
    return device, use_tabu_gpu


def log_torch_device_status(torch_device: str) -> None:
    """Log whether PyTorch sees CUDA (helps debug CPU-only embedding runs)."""
    try:
        import torch

        ver_cuda = getattr(torch.version, "cuda", None)
        logger.info(
            "torch %s from %s",
            torch.__version__,
            torch.__file__,
        )
        logger.info(
            "torch.version.cuda=%s, cuda.is_available()=%s",
            ver_cuda,
            torch.cuda.is_available(),
        )
        if ver_cuda is None and torch_device.lower().startswith("cuda"):
            logger.warning(
                "This Python environment has a CPU-only PyTorch build (torch.version.cuda is None). "
                "Embeddings will stay on CPU even if an NVIDIA GPU is present. "
                "Install a CUDA wheel into this same venv, e.g.: "
                "pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124"
            )
        if torch.cuda.is_available():
            logger.info("CUDA device 0: %s", torch.cuda.get_device_name(0))
        elif ver_cuda is not None and not torch_device.lower().startswith("cpu"):
            logger.warning(
                "PyTorch is CUDA-capable (torch.version.cuda=%s) but cuda.is_available() is False "
                "(no visible GPU driver or CUDA_VISIBLE_DEVICES).",
                ver_cuda,
            )
        logger.info(
            "Embedding encode() target device=%s (passed explicitly to encode/predict)",
            torch_device,
        )
    except ImportError:
        logger.info("torch not installed; embeddings cannot use CUDA")


def first_module_parameter_device(model: Any) -> str:
    """Reliable device string; SentenceTransformer.device may follow HF and mis-report cpu."""
    try:
        import torch

        p = next(model.parameters())
        return str(p.device)
    except (StopIteration, ImportError):
        return "unknown"


def to_torch_device(device: str) -> Any:
    """torch.device for .to() / encode(device=...)."""
    import torch

    return torch.device(normalize_torch_device(device))


def move_sentence_transformer_to_device(model: Any, device: str) -> None:
    """Ensure weights live on the requested device (ST constructor alone is not always enough)."""
    if device == "cpu" or str(device).lower().startswith("cpu"):
        return
    dev_s = normalize_torch_device(device)
    if dev_s.lower().startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return
        except ImportError:
            return
    try:
        import torch

        dev = torch.device(dev_s)
        model.to(dev)
    except Exception as exc:
        logger.warning("Could not move SentenceTransformer to %s: %s", device, exc)


def _ranking_metrics(
    ranked_ids: list[int],
    relevance_map: dict[int, float],
    k: int = 10,
) -> dict[str, float]:
    if not ranked_ids:
        return {
            "mrr": 0.0,
            "ndcg": 0.0,
            "recall": 0.0,
            "evidence_recall": 0.0,
            "support_coverage": 0.0,
            "answer_utility": 0.0,
            "hit": 0.0,
        }
    ranked = [int(x) for x in ranked_ids[:k]]
    positives = {doc_id for doc_id, rel in relevance_map.items() if rel > 0.0}
    rr = 0.0
    dcg = 0.0
    hits = 0
    support_mass = 0.0
    answer_utility = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        relevance = float(relevance_map.get(doc_id, 0.0))
        if rr == 0.0 and relevance > 0.0:
            rr = 1.0 / float(rank)
        if relevance > 0.0:
            hits += 1
            support_mass += relevance
            dcg += ((2.0**relevance) - 1.0) / math.log2(rank + 1.0)
            answer_utility += relevance / math.sqrt(float(rank))
    ideal = sorted((v for v in relevance_map.values() if v > 0.0), reverse=True)[:k]
    idcg = sum(
        ((2.0**rel) - 1.0) / math.log2(i + 1.0) for i, rel in enumerate(ideal, start=1)
    )
    total_positive_mass = float(sum(v for v in relevance_map.values() if v > 0.0))
    ideal_answer_utility = sum(rel / math.sqrt(float(i)) for i, rel in enumerate(ideal, start=1))
    return {
        "mrr": rr,
        "ndcg": (dcg / idcg) if idcg > 0.0 else 0.0,
        "recall": hits / float(len(positives)) if positives else 0.0,
        "evidence_recall": hits / float(len(positives)) if positives else 0.0,
        "support_coverage": (support_mass / total_positive_mass) if total_positive_mass > 0.0 else 0.0,
        "answer_utility": (answer_utility / ideal_answer_utility) if ideal_answer_utility > 0.0 else 0.0,
        "hit": 1.0 if rr > 0.0 else 0.0,
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_scifact(cache_dir: Path) -> dict[str, Any]:
    """Download SciFact if missing; return paths and corpus file hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "scifact.zip"
    extract_root = cache_dir / "scifact_extracted"
    marker = extract_root / ".extract_ok"

    if not marker.is_file():
        logger.info("Downloading SciFact from %s", SCIFACT_ZIP_URL)
        urlretrieve(SCIFACT_ZIP_URL, zip_path)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        marker.write_text(_file_sha256(zip_path), encoding="utf-8")

    # BEIR zip layout: scifact/scifact.jsonl or scifact/corpus.jsonl
    candidates = list(extract_root.rglob("corpus.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No corpus.jsonl under {extract_root}")
    corpus_path = min(candidates, key=lambda p: len(str(p)))
    queries_paths = list(extract_root.rglob("queries.jsonl"))
    if not queries_paths:
        raise FileNotFoundError("No queries.jsonl in SciFact archive")
    queries_path = min(queries_paths, key=lambda p: len(str(p)))
    qrels_paths = list(extract_root.rglob("test.tsv")) + list(extract_root.rglob("qrels/test.tsv"))
    if not qrels_paths:
        # some layouts use dev.tsv
        qrels_paths = list(extract_root.rglob("dev.tsv"))
    if not qrels_paths:
        raise FileNotFoundError("No qrels test.tsv (or dev.tsv) in SciFact archive")
    qrels_path = qrels_paths[0]

    corpus_sha = _file_sha256(corpus_path)
    return {
        "corpus_path": corpus_path,
        "queries_path": queries_path,
        "qrels_path": qrels_path,
        "zip_sha256": marker.read_text(encoding="utf-8").strip(),
        "corpus_jsonl_sha256": corpus_sha,
        "zip_path": str(zip_path),
    }


def load_scifact(paths: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str], list[tuple[str, str, int]]]:
    corpus: list[dict[str, Any]] = []
    with open(paths["corpus_path"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row["_id"])
            title = str(row.get("title") or "")
            text = str(row.get("text") or "")
            corpus.append({"_id": cid, "text": (title + " " + text).strip()})

    queries: dict[str, str] = {}
    with open(paths["queries_path"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries[str(row["_id"])] = str(row["text"])

    qrels: list[tuple[str, str, int]] = []
    with open(paths["qrels_path"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                parts = line.split("\t")
            if len(parts) < 3:
                continue
            if parts[0].lower() in ("query-id", "query_id"):
                continue
            qid, did = str(parts[0]), str(parts[1])
            try:
                score = int(float(parts[2]))
            except ValueError:
                continue
            qrels.append((qid, did, score))

    return corpus, queries, qrels


def rrf_ranked_ids(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    *,
    rrf_k: float = 60.0,
) -> list[int]:
    fused: dict[int, float] = {}
    for rank, (doc_id, _) in enumerate(dense, start=1):
        fused[int(doc_id)] = fused.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
    for rank, (doc_id, _) in enumerate(sparse, start=1):
        fused[int(doc_id)] = fused.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
    return [doc_id for doc_id, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]


def fuse_results_rrf(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    *,
    rrf_k: int = 60,
    top_k: int = 1000,
) -> list[int]:
    """RRF via production fusion helpers (same semantics as _rrf_ranked_ids)."""
    dense_sr = [
        SearchResult(doc_id=d, score=s, rank=i + 1, source="dense") for i, (d, s) in enumerate(dense)
    ]
    sparse_sr = [
        SearchResult(doc_id=d, score=s, rank=i + 1, source="sparse") for i, (d, s) in enumerate(sparse)
    ]
    cfg = FusionConfig(strategy="rrf", top_k=top_k, rrf_k=rrf_k)
    fused = fuse_results({"dense": dense_sr, "sparse": sparse_sr}, cfg)
    return [int(r.doc_id) for r in fused]


def minmax_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def zscore_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mu = float(scores.mean())
    sd = float(scores.std())
    if sd < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - mu) / sd).astype(np.float32)


def calibrate_channel(scores: np.ndarray, mode: str) -> np.ndarray:
    """Per-query, within-pool calibration of one retrieval channel."""
    if mode == "none":
        return scores.astype(np.float32)
    if mode == "minmax":
        return minmax_norm(scores)
    if mode == "zscore":
        return zscore_norm(scores)
    raise ValueError(f"Unknown calibration mode: {mode}")


def _token_set(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", text.lower()) if len(t) > 2}


def lexical_jaccard_similarity_matrix(corpus_texts: list[str], candidate_ids: list[int]) -> np.ndarray:
    """Pairwise Jaccard on token sets; diagonal zero (redundancy off-diagonal in solver)."""
    n = len(candidate_ids)
    sets = [_token_set(corpus_texts[cid]) for cid in candidate_ids]
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            if not a and not b:
                jv = 1.0
            elif not a or not b:
                jv = 0.0
            else:
                inter = len(a & b)
                union = len(a | b)
                jv = float(inter / union) if union else 0.0
            mat[i, j] = mat[j, i] = jv
    return mat


def embedding_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalized rows; diagonal zero."""
    sim = (embs @ embs.T).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    return sim


def cross_encoder_scores_for_pool(
    ce_model: Any,
    query_text: str,
    corpus_texts: list[str],
    pool_ids: list[int],
    *,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[int, float]:
    """Cross-encoder relevance scores for each doc in the pool (BEIR-style rerank signal)."""
    pairs = [(query_text, corpus_texts[cid]) for cid in pool_ids]
    raw: list[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        scores = ce_model.predict(
            batch,
            show_progress_bar=False,
            device=to_torch_device(device),
        )
        raw.extend(float(x) for x in np.asarray(scores).flatten())
    return {cid: raw[j] for j, cid in enumerate(pool_ids)}


def _estimate_tokens(text: str) -> int:
    return max(32, min(768, len(text.split()) * 4 // 3 + 8))


_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _query_aspect_texts(query_text: str, *, max_aspects: int = 8) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", (query_text or "").lower())
        if len(token) > 2 and token not in _QUERY_STOPWORDS
    ]
    aspects: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        key = value.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        aspects.append(key)

    for width in (3, 2):
        for offset in range(len(tokens) - width + 1):
            _add(" ".join(tokens[offset : offset + width]))
            if len(aspects) >= max_aspects:
                return aspects[:max_aspects]
    for token in tokens:
        _add(token)
        if len(aspects) >= max_aspects:
            break
    return aspects[:max_aspects]


def _proxy_query_token_weights(coverage_matrix: np.ndarray) -> np.ndarray:
    if coverage_matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    best = coverage_matrix.max(axis=1).astype(np.float32, copy=False)
    if coverage_matrix.shape[1] > 1:
        second = np.partition(coverage_matrix, coverage_matrix.shape[1] - 2, axis=1)[
            :, coverage_matrix.shape[1] - 2
        ].astype(np.float32, copy=False)
    else:
        second = np.zeros_like(best, dtype=np.float32)
    rarity = (1.0 - coverage_matrix.mean(axis=1)).astype(np.float32, copy=False)
    hardness = (
        0.55 * (1.0 - best)
        + 0.25 * (1.0 - np.clip(best - second, 0.0, 1.0))
        + 0.20 * np.clip(rarity, 0.0, 1.0)
        + 1e-3
    ).astype(np.float32, copy=False)
    hardness /= float(max(hardness.sum(), 1e-8))
    return hardness


def _query_coverage_proxy(
    query_text: str,
    query_emb: np.ndarray,
    candidate_ids: list[int],
    candidate_embs: np.ndarray,
    corpus_texts: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    n = len(candidate_ids)
    if n == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
            np.zeros((0,), dtype=np.float32),
        )
    qn = (query_emb.astype(np.float32) / (np.linalg.norm(query_emb) + 1e-8)).astype(np.float32)
    row_labels: list[str] = ["semantic_query"]
    rows: list[np.ndarray] = [
        np.clip(candidate_embs @ qn, 0.0, 1.0).astype(np.float32, copy=False)
    ]
    aspect_texts = _query_aspect_texts(query_text)
    token_sets = [
        _token_set(corpus_texts[cid] if 0 <= cid < len(corpus_texts) else "") for cid in candidate_ids
    ]
    lowered_texts = [
        (corpus_texts[cid] if 0 <= cid < len(corpus_texts) else "").lower() for cid in candidate_ids
    ]
    for aspect in aspect_texts:
        aspect_tokens = _token_set(aspect)
        if not aspect_tokens:
            continue
        values = np.zeros((n,), dtype=np.float32)
        for idx, candidate_tokens in enumerate(token_sets):
            token_overlap = len(aspect_tokens & candidate_tokens) / float(max(len(aspect_tokens), 1))
            phrase_hit = 1.0 if aspect in lowered_texts[idx] else 0.0
            values[idx] = float(max(token_overlap, phrase_hit) if len(aspect_tokens) == 1 else 0.65 * phrase_hit + 0.35 * token_overlap)
        if float(values.max()) <= 1e-8:
            continue
        row_labels.append(f"lexical:{aspect}")
        rows.append(values.astype(np.float32, copy=False))
    coverage_matrix = np.clip(np.stack(rows).astype(np.float32, copy=False), 0.0, 1.0)
    query_token_weights = _proxy_query_token_weights(coverage_matrix)
    fulfilment_proxy = minmax_norm(
        (coverage_matrix * query_token_weights[:, None]).sum(axis=0).astype(np.float64, copy=False)
    ).astype(np.float32, copy=False)
    return coverage_matrix, query_token_weights, row_labels, fulfilment_proxy


def _selection_diagnostics(
    coverage_matrix: np.ndarray,
    query_token_weights: np.ndarray,
    similarity_matrix: np.ndarray,
    selected_indices: list[int],
    *,
    quorum_threshold: float = 0.55,
) -> dict[str, float]:
    uncovered_mass = 0.0
    quorum_mass = 0.0
    weighted_support_coverage = 0.0
    if coverage_matrix.size and query_token_weights.size and selected_indices:
        selected_cov = coverage_matrix[:, selected_indices]
        best = selected_cov.max(axis=1)
        uncovered_mass = float(np.sum(query_token_weights * (1.0 - best)))
        quorum_hits = (selected_cov >= quorum_threshold).sum(axis=1) >= 2
        quorum_mass = float(np.sum(query_token_weights * quorum_hits.astype(np.float32)))
        weighted_support_coverage = float(np.sum(query_token_weights * (best >= quorum_threshold).astype(np.float32)))
    redundancy_density = 0.0
    if similarity_matrix.size:
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        if np.any(mask):
            redundancy_density = float(np.mean(similarity_matrix[mask]))
    selected_redundancy = 0.0
    if len(selected_indices) > 1:
        selected_sim = similarity_matrix[np.ix_(selected_indices, selected_indices)]
        mask = ~np.eye(selected_sim.shape[0], dtype=bool)
        if np.any(mask):
            selected_redundancy = float(np.mean(selected_sim[mask]))
    return {
        "uncovered_mass": uncovered_mass,
        "quorum_mass": quorum_mass,
        "weighted_support_coverage": weighted_support_coverage,
        "redundancy_density": redundancy_density,
        "selected_redundancy": selected_redundancy,
    }


def _selection_change_summary(base_ids: list[int], alt_ids: list[int]) -> dict[str, float]:
    base_set = {int(doc_id) for doc_id in base_ids}
    alt_set = {int(doc_id) for doc_id in alt_ids}
    union = len(base_set | alt_set)
    overlap = len(base_set & alt_set)
    return {
        "selected_overlap": float(overlap),
        "selected_jaccard": float(overlap / float(union)) if union else 1.0,
        "selected_added": float(len(alt_set - base_set)),
        "selected_removed": float(len(base_set - alt_set)),
    }


def _rrf_scores_for_pool(
    dense_list: list[tuple[int, float]],
    sparse_list: list[tuple[int, float]],
    *,
    rrf_k: float,
) -> dict[int, float]:
    fused: dict[int, float] = {}
    for rank, (doc_id, _) in enumerate(dense_list, start=1):
        fused[int(doc_id)] = fused.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
    for rank, (doc_id, _) in enumerate(sparse_list, start=1):
        fused[int(doc_id)] = fused.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
    return fused


def tabu_ranking(
    candidate_ids: list[int],
    corpus_embs: np.ndarray,
    corpus_texts: list[str],
    query_emb: np.ndarray,
    dense_scores: dict[int, float],
    bm25_scores: dict[int, float],
    dense_list: list[tuple[int, float]],
    sparse_list: list[tuple[int, float]],
    *,
    rrf_k: float,
    solver_config: Any,
    constraints: Any,
    calibration: str = "minmax",
    relevance_source: str = "heuristic",
    ce_scores: dict[int, float] | None = None,
    tail_order: str = "rrf",
    lexical_redundancy_weight: float = 0.0,
    query_text: str = "",
    return_details: bool = False,
) -> Any:
    """Tabu selection order first; tail ordered by RRF or CE (fair protocol: same CE for sort + pack).

    relevance_source: "heuristic" = calibrated BM25+dense mix; "ce" = cross-encoder scores on pool.
    tail_order: "rrf" = RRF pool scores; "ce" = descending CE (isolates packing vs listwise CE sort).
    lexical_redundancy_weight: blend Jaccard text overlap with embedding similarity for QKP redundancy.
    """
    from latence_solver import TabuSearchSolver

    n = len(candidate_ids)
    if n == 0:
        return []
    idx = np.asarray(candidate_ids, dtype=np.intp)
    embs = corpus_embs[idx].astype(np.float32)

    dlist = np.array([dense_scores.get(cid, 0.0) for cid in candidate_ids], dtype=np.float32)
    blist = np.array([bm25_scores.get(cid, 0.0) for cid in candidate_ids], dtype=np.float32)
    d_cal = calibrate_channel(dlist, calibration)
    b_cal = calibrate_channel(blist, calibration)
    mix = 0.5 * d_cal + 0.5 * b_cal
    # Single pooled min-max on the combined signal keeps gains in [0, 1] for z-score channels too.
    rel_heuristic = minmax_norm(mix.astype(np.float64)).astype(np.float32)

    if relevance_source == "ce":
        if ce_scores is None:
            raise ValueError("relevance_source='ce' requires ce_scores")
        ce_arr = np.array([ce_scores.get(cid, 0.0) for cid in candidate_ids], dtype=np.float32)
        rel = minmax_norm(ce_arr)
    else:
        rel = rel_heuristic.astype(np.float32)

    rrf_score_map = _rrf_scores_for_pool(dense_list, sparse_list, rrf_k=rrf_k)
    if tail_order == "ce":
        if ce_scores is None:
            raise ValueError("tail_order='ce' requires ce_scores")
        tail_key = lambda c: float(ce_scores.get(c, 0.0))
    else:
        tail_key = lambda c: float(rrf_score_map.get(c, 0.0))

    solver = TabuSearchSolver(solver_config)
    emb_sim = embedding_similarity_matrix(embs)
    if lexical_redundancy_weight > 0.0:
        jac_sim = lexical_jaccard_similarity_matrix(corpus_texts, candidate_ids)
        w = float(max(0.0, min(1.0, lexical_redundancy_weight)))
        similarity = ((1.0 - w) * emb_sim + w * jac_sim).astype(np.float32)
    else:
        similarity = emb_sim

    token_costs = np.array(
        [_estimate_tokens(corpus_texts[cid] if 0 <= cid < len(corpus_texts) else "") for cid in candidate_ids],
        dtype=np.uint32,
    )
    linear_density_scores = rel.astype(np.float32, copy=False)
    linear_centrality_scores = rel.astype(np.float32, copy=False)
    recency_scores = np.full((n,), 0.5, dtype=np.float32)
    auxiliary_scores = minmax_norm(
        np.asarray(
            [0.5 * dense_scores.get(cid, 0.0) + 0.5 * bm25_scores.get(cid, 0.0) for cid in candidate_ids],
            dtype=np.float32,
        )
    )
    roles = np.full((n,), 255, dtype=np.uint8)
    clusters = np.asarray([i % 24 for i in range(n)], dtype=np.int32)
    coverage_matrix, query_token_weights, query_aspects, fulfil = _query_coverage_proxy(
        query_text=query_text,
        query_emb=query_emb,
        candidate_ids=candidate_ids,
        candidate_embs=embs,
        corpus_texts=corpus_texts,
    )
    qn = (query_emb.astype(np.float32) / (np.linalg.norm(query_emb) + 1e-8)).astype(np.float32)
    out = solver.solve_precomputed_numpy(
        embs,
        token_costs,
        linear_density_scores,
        linear_centrality_scores,
        recency_scores,
        auxiliary_scores,
        roles,
        clusters,
        rel.astype(np.float32),
        similarity_matrix=similarity,
        fulfilment_scores=fulfil,
        coverage_matrix=coverage_matrix,
        query_token_weights=query_token_weights,
        query_embedding=qn,
        constraints=constraints,
    )

    selected = [int(i) for i in out.selected_indices]
    selected_ids: list[int] = []
    seen: set[int] = set()
    for sel_i in selected:
        if 0 <= sel_i < n:
            did = candidate_ids[sel_i]
            if did not in seen:
                selected_ids.append(did)
                seen.add(did)
    order = sorted(selected_ids, key=tail_key, reverse=True)
    rest = sorted((cid for cid in candidate_ids if cid not in seen), key=tail_key, reverse=True)
    order.extend(rest)
    if not return_details:
        return order
    diagnostics = _selection_diagnostics(
        coverage_matrix=coverage_matrix,
        query_token_weights=query_token_weights,
        similarity_matrix=similarity,
        selected_indices=[int(i) for i in selected if 0 <= int(i) < n],
    )
    solver_output = {
        "objective_score": float(getattr(out, "objective_score", 0.0)),
        "fulfilment_total": float(getattr(out, "fulfilment_total", 0.0)),
        "redundancy_penalty": float(getattr(out, "redundancy_penalty", 0.0)),
        "total_tokens": int(getattr(out, "total_tokens", int(token_costs[[int(i) for i in selected if 0 <= int(i) < n]].sum()) if selected else 0)),
        "num_selected": int(getattr(out, "num_selected", len(selected_ids))),
        "solve_time_ms": float(getattr(out, "solve_time_ms", 0.0)),
        "constraints_satisfied": bool(getattr(out, "constraints_satisfied", True)),
    }
    diagnostics.update(
        {
            "mean_query_token_weight": float(query_token_weights.mean()) if query_token_weights.size else 0.0,
            "coverage_proxy_rows": float(coverage_matrix.shape[0]) if coverage_matrix.ndim == 2 else 0.0,
            "mean_selected_relevance": float(np.mean(rel[[int(i) for i in selected if 0 <= int(i) < n]])) if selected else 0.0,
            "mean_selected_fulfilment": float(np.mean(fulfil[[int(i) for i in selected if 0 <= int(i) < n]])) if selected else 0.0,
        }
    )
    return {
        "ranked_ids": order,
        "selected_ids": selected_ids,
        "selected_indices": [int(i) for i in selected if 0 <= int(i) < n],
        "query_aspects": query_aspects,
        "solver_output": solver_output,
        "diagnostics": diagnostics,
    }


def bootstrap_ci_deltas(
    deltas: list[float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean delta and percentile bootstrap CI on mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means[b] = float(sample.mean())
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def _solver_max_chunks_for_pool(
    pool_len: int,
    *,
    large_chunk_pool: bool,
    explicit_cap: int | None,
    eval_k: int,
) -> int:
    if explicit_cap is not None:
        return min(max(1, explicit_cap), pool_len)
    if large_chunk_pool:
        # Many chunks: Tabu optimizes redundancy under budget (regime vs shallow RRF fusion).
        return min(64, max(16, pool_len // 2))
    return min(6, max(1, eval_k))


def run_benchmark(
    *,
    cache_dir: Path,
    output_json: Path | None,
    top_m: int,
    eval_k: int,
    rrf_k: int,
    max_queries: int | None,
    bootstrap_samples: int,
    seed: int,
    embedding_model: str,
    large_chunk_pool: bool = False,
    solver_max_chunks: int | None = None,
    calibration: str = "minmax",
    use_cross_encoder: bool = False,
    ce_model_name: str = DEFAULT_CE_MODEL,
    ce_batch_size: int = 32,
    lexical_redundancy_weight: float = 0.0,
    tabu_lambda: float | None = None,
    lambda_ablation: bool = False,
    force_cpu: bool = False,
    torch_device_override: str | None = None,
) -> dict[str, Any]:
    try:
        import latence_solver  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "latence_solver is required. Build it with:\n"
            "  cd src/kernels/knapsack_solver && maturin develop --release\n"
            f"Original error: {e}"
        ) from e

    paths_meta = ensure_scifact(cache_dir)
    corpus, queries_map, qrels_rows = load_scifact(paths_meta)

    effective_top_m = max(top_m, 250) if large_chunk_pool else top_m
    if large_chunk_pool and effective_top_m > top_m:
        logger.info(
            "Large-chunk-pool mode: using top_m=%d (requested %d, minimum 250)",
            effective_top_m,
            top_m,
        )

    corpus_ids = [row["_id"] for row in corpus]
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    texts = [row["text"] for row in corpus]
    n_docs = len(texts)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "sentence-transformers is required: pip install sentence-transformers\n" + str(e)
        ) from e

    if force_cpu:
        torch_device, use_tabu_gpu = resolve_benchmark_devices(force_cpu=True)
    elif torch_device_override is not None and str(torch_device_override).strip() != "":
        torch_device = normalize_torch_device(str(torch_device_override).strip())
        use_tabu_gpu = False
        if torch_device != "cpu":
            try:
                from latence_solver import gpu_available

                use_tabu_gpu = bool(gpu_available())
            except Exception:
                use_tabu_gpu = False
    else:
        torch_device, use_tabu_gpu = resolve_benchmark_devices(force_cpu=False)

    torch_device = coerce_torch_device_to_available(torch_device)

    log_torch_device_status(torch_device)
    embed_batch = 128 if str(torch_device).startswith("cuda") else 64
    logger.info(
        "Devices: sentence-transformers=%s, Tabu solver use_gpu=%s",
        torch_device,
        use_tabu_gpu,
    )

    logger.info("Loading embedding model %s", embedding_model)
    st_model = SentenceTransformer(
        embedding_model,
        device=torch_device,
    )
    move_sentence_transformer_to_device(st_model, torch_device)
    logger.info(
        "SentenceTransformer weights device (first parameter): %s (model.device may lie on some HF versions)",
        first_module_parameter_device(st_model),
    )

    dim = int(
        st_model.get_embedding_dimension()
        if hasattr(st_model, "get_embedding_dimension")
        else st_model.get_sentence_embedding_dimension()
    )

    enc_device = to_torch_device(torch_device)
    logger.info("Encoding %d passages (batch)", n_docs)
    corpus_embs = st_model.encode(
        texts,
        batch_size=embed_batch,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=enc_device,
    ).astype(np.float32)

    logger.info("Building BM25 index")
    bm25 = BM25Engine(config=BM25Config())
    bm25.index_documents(texts, doc_ids=list(range(n_docs)))

    tmp = tempfile.mkdtemp(prefix="hnsw_bench_")
    hnsw_path = Path(tmp)
    hnsw = HnswSegmentManager(hnsw_path, dim=dim, on_disk=False)
    hnsw.add(corpus_embs, ids=list(range(n_docs)), payloads=[{} for _ in range(n_docs)])

    # group qrels by query
    qrel_by_q: dict[str, dict[int, float]] = {}
    for qid, did, sc in qrels_rows:
        if did not in id_to_idx:
            continue
        inner = qrel_by_q.setdefault(qid, {})
        inner[id_to_idx[did]] = float(max(sc, 0))

    query_ids = [q for q in qrel_by_q if q in queries_map]
    if max_queries is not None:
        query_ids = query_ids[: max_queries]

    from latence_solver import SolverConfig, SolverConstraints

    lam_default = 0.45 if large_chunk_pool else 0.35
    lam = float(tabu_lambda) if tabu_lambda is not None else lam_default
    solver_config = SolverConfig(
        iterations=120 if large_chunk_pool else 80,
        random_seed=seed,
        use_gpu=use_tabu_gpu,
        lambda_=lam,
        tabu_tenure=16 if large_chunk_pool else 12,
    )
    solver_config_lambda0 = SolverConfig(
        iterations=solver_config.iterations,
        random_seed=seed,
        use_gpu=use_tabu_gpu,
        lambda_=0.0,
        tabu_tenure=solver_config.tabu_tenure,
    )
    solver_config_mu0 = SolverConfig(
        iterations=solver_config.iterations,
        random_seed=seed,
        use_gpu=use_tabu_gpu,
        lambda_=lam,
        mu=0.0,
        tabu_tenure=solver_config.tabu_tenure,
    )
    if large_chunk_pool:
        token_cap = min(80000, max(20000, effective_top_m * 200))
    else:
        token_cap = min(12000, max(4000, effective_top_m * 120))

    metric_ks = sorted({10, eval_k, 50}) if large_chunk_pool else [eval_k]

    active_arms: list[str] = ["rrf", "tabu_heuristic"]
    if use_cross_encoder:
        active_arms.extend(["ce_sort", "tabu_ce"])
        if lambda_ablation:
            active_arms.append("tabu_ce_lambda0")

    metric_names = (
        "ndcg",
        "mrr",
        "recall",
        "evidence_recall",
        "support_coverage",
        "answer_utility",
        "hit",
    )
    metrics_by_arm: dict[str, dict[int, dict[str, list[float]]]] = {
        a: {kk: {metric: [] for metric in metric_names} for kk in metric_ks} for a in active_arms
    }
    primary_diagnostics: dict[str, list[float]] = {
        "uncovered_mass": [],
        "redundancy_density": [],
        "quorum_mass": [],
        "weighted_support_coverage": [],
        "lambda0_selected_jaccard": [],
        "mu0_selected_jaccard": [],
    }

    if lambda_ablation and not use_cross_encoder:
        logger.warning("--lambda-ablation requires --use-cross-encoder; ignoring lambda ablation arm")

    ce_model: Any = None
    if use_cross_encoder:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise SystemExit(
                "Cross-encoder requires sentence-transformers with CrossEncoder: pip install sentence-transformers\n"
                + str(e)
            ) from e
        logger.info("Loading cross-encoder %s (device=%s)", ce_model_name, torch_device)
        ce_model = CrossEncoder(ce_model_name, device=torch_device)
        try:
            if hasattr(ce_model, "model") and torch_device != "cpu":
                import torch

                ce_model.model.to(torch.device(normalize_torch_device(torch_device)))
        except Exception as exc:
            logger.warning("Could not move CrossEncoder model to %s: %s", torch_device, exc)

    per_query: list[dict[str, Any]] = []
    tabu_primary_key = "tabu_ce" if use_cross_encoder else "tabu_heuristic"

    for qid in query_ids:
        qtext = queries_map[qid]
        rel_map = qrel_by_q[qid]
        if not rel_map:
            continue

        q_emb = st_model.encode(
            [qtext],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=enc_device,
        )[0].astype(np.float32)

        bm25_hits = bm25.search(qtext, top_k=effective_top_m)
        dense_hits = hnsw.search(q_emb, k=effective_top_m)

        # dense = vector (HNSW); sparse = lexical (BM25)
        dense_list = [(int(d), float(s)) for d, s in dense_hits]
        sparse_list = [(int(r.doc_id), float(r.score)) for r in bm25_hits]

        pool = list(
            dict.fromkeys([d for d, _ in dense_list] + [d for d, _ in sparse_list])
        )
        if not pool:
            continue

        rrf_ids = fuse_results_rrf(dense_list, sparse_list, rrf_k=rrf_k, top_k=len(pool) + 10)

        qt = bm25.tokenize(qtext)
        bm25_scores: dict[int, float] = {}
        for doc_idx in pool:
            bm25_scores[doc_idx] = float(bm25._score_document(doc_idx, qt))

        dense_scores_map = {int(d): float(s) for d, s in dense_list}

        smc = _solver_max_chunks_for_pool(
            len(pool),
            large_chunk_pool=large_chunk_pool,
            explicit_cap=solver_max_chunks,
            eval_k=eval_k,
        )
        constraints = SolverConstraints(
            max_tokens=token_cap,
            max_chunks=min(smc, len(pool)),
            max_per_cluster=8 if large_chunk_pool else 4,
        )

        tabu_common = dict(
            candidate_ids=pool,
            corpus_embs=corpus_embs,
            corpus_texts=texts,
            query_emb=q_emb,
            query_text=qtext,
            dense_scores=dense_scores_map,
            bm25_scores=bm25_scores,
            dense_list=dense_list,
            sparse_list=sparse_list,
            rrf_k=float(rrf_k),
            constraints=constraints,
            calibration=calibration,
            lexical_redundancy_weight=lexical_redundancy_weight,
            return_details=True,
        )

        tabu_heuristic_result = tabu_ranking(
            **tabu_common,
            solver_config=solver_config,
            relevance_source="heuristic",
            tail_order="rrf",
            ce_scores=None,
        )
        tabu_heuristic_ids = list(tabu_heuristic_result["ranked_ids"])

        arms_ranked: dict[str, list[int]] = {
            "rrf": rrf_ids,
            "tabu_heuristic": tabu_heuristic_ids,
        }
        arm_details: dict[str, dict[str, Any]] = {"tabu_heuristic": tabu_heuristic_result}

        ce_scores: dict[int, float] | None = None
        if use_cross_encoder and ce_model is not None:
            ce_scores = cross_encoder_scores_for_pool(
                ce_model,
                qtext,
                texts,
                pool,
                batch_size=ce_batch_size,
                device=torch_device,
            )
            ce_sort_ids = sorted(pool, key=lambda c: float(ce_scores[c]), reverse=True)
            arms_ranked["ce_sort"] = ce_sort_ids
            tabu_ce_result = tabu_ranking(
                **tabu_common,
                solver_config=solver_config,
                relevance_source="ce",
                ce_scores=ce_scores,
                tail_order="ce",
            )
            arms_ranked["tabu_ce"] = list(tabu_ce_result["ranked_ids"])
            arm_details["tabu_ce"] = tabu_ce_result
            if lambda_ablation:
                tabu_ce_lambda0_result = tabu_ranking(
                    **tabu_common,
                    solver_config=solver_config_lambda0,
                    relevance_source="ce",
                    ce_scores=ce_scores,
                    tail_order="ce",
                )
                arms_ranked["tabu_ce_lambda0"] = list(tabu_ce_lambda0_result["ranked_ids"])
                arm_details["tabu_ce_lambda0"] = tabu_ce_lambda0_result

        primary_result = arm_details[tabu_primary_key]
        primary_lambda0 = tabu_ranking(
            **tabu_common,
            solver_config=solver_config_lambda0,
            relevance_source="ce" if use_cross_encoder and ce_scores is not None else "heuristic",
            ce_scores=ce_scores,
            tail_order="ce" if use_cross_encoder and ce_scores is not None else "rrf",
        )
        primary_mu0 = tabu_ranking(
            **tabu_common,
            solver_config=solver_config_mu0,
            relevance_source="ce" if use_cross_encoder and ce_scores is not None else "heuristic",
            ce_scores=ce_scores,
            tail_order="ce" if use_cross_encoder and ce_scores is not None else "rrf",
        )
        lambda_change = _selection_change_summary(
            primary_result["selected_ids"],
            primary_lambda0["selected_ids"],
        )
        mu_change = _selection_change_summary(
            primary_result["selected_ids"],
            primary_mu0["selected_ids"],
        )
        primary_diagnostics["uncovered_mass"].append(
            float(primary_result["diagnostics"]["uncovered_mass"])
        )
        primary_diagnostics["redundancy_density"].append(
            float(primary_result["diagnostics"]["redundancy_density"])
        )
        primary_diagnostics["quorum_mass"].append(
            float(primary_result["diagnostics"]["quorum_mass"])
        )
        primary_diagnostics["weighted_support_coverage"].append(
            float(primary_result["diagnostics"]["weighted_support_coverage"])
        )
        primary_diagnostics["lambda0_selected_jaccard"].append(
            float(lambda_change["selected_jaccard"])
        )
        primary_diagnostics["mu0_selected_jaccard"].append(float(mu_change["selected_jaccard"]))

        row: dict[str, Any] = {
            "query_id": qid,
            "pool_size": len(pool),
            "solver_max_chunks": smc,
            "tabu_diagnostics": {
                arm_name: {
                    "solver_output": arm_result["solver_output"],
                    "diagnostics": arm_result["diagnostics"],
                    "query_aspects": list(arm_result["query_aspects"]),
                }
                for arm_name, arm_result in arm_details.items()
            },
            "tabu_ablation": {
                "primary_arm": tabu_primary_key,
                "lambda0": {
                    **lambda_change,
                    "diagnostics": primary_lambda0["diagnostics"],
                },
                "mu0": {
                    **mu_change,
                    "diagnostics": primary_mu0["diagnostics"],
                },
            },
        }
        for arm_name, ranked in arms_ranked.items():
            arm_k: dict[str, Any] = {}
            for kk in metric_ks:
                m = _ranking_metrics(ranked, rel_map, k=kk)
                for metric_name in metric_names:
                    metrics_by_arm[arm_name][kk][metric_name].append(float(m[metric_name]))
                arm_k[f"k{kk}"] = m
            row[arm_name] = arm_k

        per_query.append(row)

    by_k: dict[str, Any] = {}
    primary_k = 50 if large_chunk_pool else eval_k

    def _paired_block(
        deltas_ndcg: list[float],
        deltas_mrr: list[float],
        *,
        boot_seed: int,
    ) -> dict[str, Any]:
        mean_d_ndcg, lo_ndcg, hi_ndcg = bootstrap_ci_deltas(
            deltas_ndcg, n_boot=bootstrap_samples, seed=boot_seed
        )
        mean_d_mrr, lo_mrr, hi_mrr = bootstrap_ci_deltas(
            deltas_mrr, n_boot=bootstrap_samples, seed=boot_seed + 777
        )
        wn: dict[str, Any] | None = None
        try:
            from scipy.stats import wilcoxon

            if len(deltas_ndcg) > 5:
                stat, p = wilcoxon(deltas_ndcg, alternative="two-sided", zero_method="wilcox")
                wn = {"statistic": float(stat), "p_value": float(p), "n": len(deltas_ndcg)}
        except Exception:
            pass
        return {
            "mean_delta_ndcg": mean_d_ndcg,
            "bootstrap_95ci_ndcg": [lo_ndcg, hi_ndcg],
            "mean_delta_mrr": mean_d_mrr,
            "bootstrap_95ci_mrr": [lo_mrr, hi_mrr],
            "wilcoxon_ndcg": wn,
        }

    for k in metric_ks:
        block: dict[str, Any] = {}
        for arm in active_arms:
            block[arm] = {
                f"mean_{metric_name}": (
                    float(np.mean(metrics_by_arm[arm][k][metric_name]))
                    if metrics_by_arm[arm][k][metric_name]
                    else 0.0
                )
                for metric_name in metric_names
            }
        deltas_tr = [
            t - r
            for t, r in zip(
                metrics_by_arm[tabu_primary_key][k]["ndcg"],
                metrics_by_arm["rrf"][k]["ndcg"],
            )
        ]
        deltas_tr_m = [
            t - r
            for t, r in zip(
                metrics_by_arm[tabu_primary_key][k]["mrr"],
                metrics_by_arm["rrf"][k]["mrr"],
            )
        ]
        block["paired_delta_tabu_minus_rrf"] = _paired_block(
            deltas_tr, deltas_tr_m, boot_seed=seed + 1 + k
        )
        if use_cross_encoder:
            deltas_pack = [
                t - c
                for t, c in zip(
                    metrics_by_arm["tabu_ce"][k]["ndcg"],
                    metrics_by_arm["ce_sort"][k]["ndcg"],
                )
            ]
            deltas_pack_m = [
                t - c
                for t, c in zip(
                    metrics_by_arm["tabu_ce"][k]["mrr"],
                    metrics_by_arm["ce_sort"][k]["mrr"],
                )
            ]
            block["paired_delta_tabu_ce_minus_ce_sort"] = _paired_block(
                deltas_pack, deltas_pack_m, boot_seed=seed + 50 + k
            )
        if lambda_ablation and "tabu_ce_lambda0" in active_arms:
            deltas_l0 = [
                t - z
                for t, z in zip(
                    metrics_by_arm["tabu_ce"][k]["ndcg"],
                    metrics_by_arm["tabu_ce_lambda0"][k]["ndcg"],
                )
            ]
            deltas_l0_m = [
                t - z
                for t, z in zip(
                    metrics_by_arm["tabu_ce"][k]["mrr"],
                    metrics_by_arm["tabu_ce_lambda0"][k]["mrr"],
                )
            ]
            block["paired_delta_tabu_ce_minus_tabu_ce_lambda0"] = _paired_block(
                deltas_l0, deltas_l0_m, boot_seed=seed + 90 + k
            )

        by_k[str(k)] = block

    result: dict[str, Any] = {
        "dataset": {
            "name": "beir_scifact",
            "cache_dir": str(cache_dir.resolve()),
            **{k: str(v) if hasattr(v, "resolve") else v for k, v in paths_meta.items()},
        },
        "protocol": {
            "large_chunk_pool": large_chunk_pool,
            "top_m_requested": top_m,
            "top_m_effective": effective_top_m,
            "eval_k": eval_k,
            "metric_ks": metric_ks,
            "primary_metric_k": primary_k,
            "rrf_k": rrf_k,
            "embedding_model": embedding_model,
            "use_cross_encoder": use_cross_encoder,
            "ce_model": ce_model_name if use_cross_encoder else None,
            "calibration": calibration,
            "lexical_redundancy_weight": lexical_redundancy_weight,
            "lambda_ablation": lambda_ablation,
            "coverage_proxy": "semantic_query_row + lexical_query_aspects",
            "downstream_metrics": [
                "ndcg",
                "mrr",
                "evidence_recall",
                "support_coverage",
                "answer_utility",
                "hit",
            ],
            "tabu_diagnostics": [
                "uncovered_mass",
                "quorum_mass",
                "redundancy_density",
                "weighted_support_coverage",
                "lambda0_selected_jaccard",
                "mu0_selected_jaccard",
            ],
            "eval_protocol_note": (
                "When use_cross_encoder is true, tabu_ce and ce_sort share the same CE relevance on the pool; "
                "paired_delta_tabu_ce_minus_ce_sort isolates packing vs listwise CE ordering. "
                "tabu_heuristic uses calibrated BM25+dense relevance plus a query-aspect fulfilment proxy; "
                "paired_delta_tabu_minus_rrf uses primary tabu "
                f"({tabu_primary_key}) vs RRF when CE is on. "
                "Per-query diagnostics also compare the selected set against lambda=0 and mu=0 ablations."
            ),
            "torch_device": torch_device,
            "torch_device_override": torch_device_override,
            "force_cpu": force_cpu,
            "tabu_solver_use_gpu_requested": use_tabu_gpu,
            "solver": {
                "iterations": solver_config.iterations,
                "random_seed": seed,
                "use_gpu": bool(solver_config.use_gpu),
                "lambda_": float(solver_config.lambda_),
                "lambda_override": tabu_lambda,
                "tabu_tenure": int(solver_config.tabu_tenure),
                "precomputed_relevance_solver_path": True,
                "relevance_mix_heuristic": f"0.5*cal_{calibration}(dense)+0.5*cal_{calibration}(bm25)",
                "fulfilment_proxy": "query-aspect coverage rows with learned hardness-style weights",
                "tabu_primary_arm": tabu_primary_key,
                "solver_max_chunks": "explicit flag or auto (large: up to 64, min 16, ~half pool)",
                "tabu_tail_order": "rrf for heuristic Tabu; ce for tabu_ce (fair protocol)",
                "selected_prefix_order": "sorted by the same tail-order signal after subset selection",
                "relevance_sources": {
                    "tabu_heuristic": "calibrated dense+bm25 minmax fusion",
                    "tabu_ce": "cross-encoder pool scores" if use_cross_encoder else None,
                    "tabu_ce_lambda0": "cross-encoder pool scores" if lambda_ablation else None,
                },
            },
        },
        "aggregate": {
            "query_count": len(per_query),
            "by_k": by_k,
            # Back-compat: primary tabu arm vs RRF (tabu_ce when CE enabled)
            "rrf": by_k[str(primary_k)]["rrf"],
            "tabu": by_k[str(primary_k)][tabu_primary_key],
            "paired_delta_tabu_minus_rrf": by_k[str(primary_k)]["paired_delta_tabu_minus_rrf"],
            "diagnostics": {
                key: float(np.mean(values)) if values else 0.0
                for key, values in primary_diagnostics.items()
            },
        },
        "per_query": per_query,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", output_json)

    # stdout tables (per K when large-chunk-pool compares shallow vs deep cut)
    agg = result["aggregate"]
    by_k_out = agg["by_k"]
    print("")
    if large_chunk_pool:
        print("Large chunk pool: top_m_effective={}, primary nDCG/MRR @K={}".format(
            result["protocol"]["top_m_effective"],
            primary_k,
        ))
    print(
        "Devices: torch={}, Tabu use_gpu={}".format(
            result["protocol"].get("torch_device", "?"),
            result["protocol"].get("solver", {}).get("use_gpu", False),
        )
    )
    for k_str, block in sorted(by_k_out.items(), key=lambda x: int(x[0])):
        k = int(k_str)
        print("")
        print("| K={} | mean nDCG | mean MRR |".format(k))
        print("|------|-----------|----------|")
        print("| RRF  | {:.4f}    | {:.4f}   |".format(block["rrf"]["mean_ndcg"], block["rrf"]["mean_mrr"]))
        tb = block[tabu_primary_key]
        print("| {:<13} | {:.4f}    | {:.4f}   |".format(tabu_primary_key, tb["mean_ndcg"], tb["mean_mrr"]))
        if "tabu_heuristic" in block and tabu_primary_key != "tabu_heuristic":
            th = block["tabu_heuristic"]
            print("| tabu_heurist | {:.4f}    | {:.4f}   |".format(th["mean_ndcg"], th["mean_mrr"]))
        if use_cross_encoder and "ce_sort" in block:
            cs = block["ce_sort"]
            tc = block["tabu_ce"]
            print("| ce_sort      | {:.4f}    | {:.4f}   |".format(cs["mean_ndcg"], cs["mean_mrr"]))
            print("| tabu_ce      | {:.4f}    | {:.4f}   |".format(tc["mean_ndcg"], tc["mean_mrr"]))
        d = block["paired_delta_tabu_minus_rrf"]
        print(
            "  Δ({}−RRF) nDCG: mean={:.4f}, 95% bootstrap CI [{:.4f}, {:.4f}]".format(
                tabu_primary_key,
                d["mean_delta_ndcg"],
                d["bootstrap_95ci_ndcg"][0],
                d["bootstrap_95ci_ndcg"][1],
            )
        )
        if d.get("wilcoxon_ndcg"):
            w = d["wilcoxon_ndcg"]
            print("  Wilcoxon ΔnDCG: p={:.4g} (n={})".format(w["p_value"], w["n"]))
        if use_cross_encoder and "paired_delta_tabu_ce_minus_ce_sort" in block:
            dp = block["paired_delta_tabu_ce_minus_ce_sort"]
            print(
                "  Δ(tabu_ce−ce_sort) nDCG: mean={:.4f}, 95% CI [{:.4f}, {:.4f}]".format(
                    dp["mean_delta_ndcg"],
                    dp["bootstrap_95ci_ndcg"][0],
                    dp["bootstrap_95ci_ndcg"][1],
                )
            )

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="RRF vs Tabu hybrid benchmark (SciFact)")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "voyager_index_benchmarks" / "scifact",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--top-m", type=int, default=100)
    p.add_argument("--eval-k", type=int, default=10)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument(
        "--large-chunk-pool",
        action="store_true",
        help="Large BM25/HNSW top-M (min 250), Tabu max_chunks up to ~64 / half pool, "
        "higher token budget and iterations; report nDCG/MRR at K ∈ {10, eval_k, 50}.",
    )
    p.add_argument(
        "--solver-max-chunks",
        type=int,
        default=None,
        help="Override automatic Tabu max_chunks cap (default: auto from pool size / profile).",
    )
    p.add_argument(
        "--calibration",
        choices=("none", "minmax", "zscore"),
        default="minmax",
        help="Per-query calibration of BM25 and dense scores on the candidate pool before heuristic mix.",
    )
    p.add_argument(
        "--use-cross-encoder",
        action="store_true",
        help="Score pool with a cross-encoder; adds ce_sort and tabu_ce arms (same CE relevance for sort vs pack).",
    )
    p.add_argument(
        "--ce-model",
        type=str,
        default=DEFAULT_CE_MODEL,
        help="sentence-transformers CrossEncoder model id (when --use-cross-encoder).",
    )
    p.add_argument(
        "--ce-batch-size",
        type=int,
        default=32,
        help="Batch size for CE scoring.",
    )
    p.add_argument(
        "--lexical-redundancy-weight",
        type=float,
        default=0.0,
        help="Blend Jaccard doc-doc similarity with embedding similarity for redundancy (0=embedding path only).",
    )
    p.add_argument(
        "--tabu-lambda",
        type=float,
        default=None,
        help="Override solver redundancy weight λ (default: 0.35 small pool / 0.45 large-chunk-pool).",
    )
    p.add_argument(
        "--lambda-ablation",
        action="store_true",
        help="Also run tabu_ce with λ=0 and report paired_delta vs full λ (requires --use-cross-encoder).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for bi-encoder, cross-encoder, and Tabu (default: CUDA/MPS + Tabu GPU when available).",
    )
    p.add_argument(
        "--torch-device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Override PyTorch device for bi/cross-encoder (e.g. cuda:0, cpu). "
        "If CUDA is requested but this PyTorch is CPU-only, the script falls back to CPU with a warning.",
    )
    args = p.parse_args()

    run_benchmark(
        cache_dir=args.cache_dir,
        output_json=args.output_json,
        top_m=args.top_m,
        eval_k=args.eval_k,
        rrf_k=args.rrf_k,
        max_queries=args.max_queries,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
        embedding_model=args.embedding_model,
        large_chunk_pool=args.large_chunk_pool,
        solver_max_chunks=args.solver_max_chunks,
        calibration=args.calibration,
        use_cross_encoder=args.use_cross_encoder,
        ce_model_name=args.ce_model,
        ce_batch_size=args.ce_batch_size,
        lexical_redundancy_weight=args.lexical_redundancy_weight,
        tabu_lambda=args.tabu_lambda,
        lambda_ablation=args.lambda_ablation,
        force_cpu=args.cpu,
        torch_device_override=args.torch_device,
    )


if __name__ == "__main__":
    main()
