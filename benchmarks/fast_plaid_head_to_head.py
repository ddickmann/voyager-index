"""
voyager-index vs FastPlaid head-to-head BEIR benchmark.

Runs the same BEIR matrix that FastPlaid publishes
(https://github.com/lightonai/fast-plaid) against voyager-index's
rroq158 (gs=128, the v0.1.6 SOTA default) and fp16 baselines using
*identical* per-token embeddings, so the only thing the comparison
varies is the indexing/scoring engine.

The script reuses the prepared NPZs from
`benchmarks/data/prepare_beir_datasets.py` (encoded once with
`lightonai/GTE-ModernColBERT-v1` at dim=128) and feeds the same doc /
query embedding tensors into both libraries:

  * voyager-index: builds a sharded index via the existing benchmark
    pipeline (`benchmarks/beir_benchmark.py::run_dataset`) and reports
    the GPU-corpus QPS / NDCG@10 / indexing-time row.
  * FastPlaid: wraps `fast_plaid.search.FastPlaid(...).create(...) /
    .search(...)` and reports the same three numbers from the same
    embeddings.

The output table is shaped exactly like FastPlaid's published table
(Dataset, library, NDCG@10, Indexing time (s), QPS) so a row-for-row
comparison is one diff away.

Usage:
    # First-time setup (one-time, run on the same box that has the GPU)
    pip install fast-plaid pylate
    python benchmarks/data/prepare_beir_datasets.py \\
        --datasets arguana fiqa nfcorpus quora scidocs scifact \\
                   trec-covid webis-touche2020

    # Head-to-head (default: BEIR-8, voyager-rroq158-gs128 + fast_plaid)
    python benchmarks/fast_plaid_head_to_head.py

    # Quick smoke (one small dataset, ~30 s on an H100)
    python benchmarks/fast_plaid_head_to_head.py --datasets nfcorpus

    # Skip FastPlaid lane (e.g. machine without it installed)
    python benchmarks/fast_plaid_head_to_head.py --libraries voyager_rroq158_gs128

    # Add the fp16 baseline for a 3-way table
    python benchmarks/fast_plaid_head_to_head.py \\
        --libraries voyager_fp16 voyager_rroq158_gs128 fast_plaid

See `docs/benchmarks/fast-plaid-head-to-head.md` for the full tutorial
including the recommended H100 cloud setup.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beir_benchmark import (
    BEIR_CACHE,
    DATASETS as VOYAGER_BEIR_6,
    TOP_K,
    evaluate as evaluate_retrieval,
    load_beir_npz,
    run_dataset as voyager_run_dataset,
)
from voyager_index._internal.inference.shard_engine.config import Compression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# FastPlaid's published BEIR matrix (https://github.com/lightonai/fast-plaid).
# We support the same 8 datasets; you can pass `--datasets` to override.
FAST_PLAID_BEIR_8 = [
    "arguana",
    "fiqa",
    "nfcorpus",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
]

DEFAULT_LIBRARIES = ["voyager_rroq158_gs128", "fast_plaid"]
SUPPORTED_LIBRARIES = {
    "voyager_fp16": "voyager-index, fp16 (unquantized MaxSim ceiling)",
    "voyager_rroq158_gs32": "voyager-index, rroq158 group_size=32 (pre-v0.1.6 baseline)",
    "voyager_rroq158_gs128": "voyager-index, rroq158 group_size=128 (v0.1.6 SOTA default)",
    "voyager_rroq4_riem": "voyager-index, rroq4_riem (no-quality-loss codec)",
    "fast_plaid": "lightonai/fast-plaid (PLAID with Rust GPU kernels)",
}

# Per-dataset recommended `n_samples_kmeans` for FastPlaid's centroid pass.
# Mirrors the heuristic in their indexing helper: cap kmeans samples at
# 32k tokens for medium corpora to keep indexing fast on smaller GPUs.
FAST_PLAID_KMEANS_CAP = 32_000


def _import_fast_plaid():
    """Import FastPlaid lazily so the script runs on machines without it."""
    try:
        from fast_plaid import search as fp_search  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "FastPlaid is not installed. Install it with `pip install fast-plaid` "
            "(see https://github.com/lightonai/fast-plaid). To run only the "
            "voyager-index lanes, pass `--libraries voyager_rroq158_gs128`."
        ) from exc
    return fp_search


def _load_npz_as_tensors(name: str) -> Dict[str, Any]:
    """Load a prepared BEIR NPZ and return per-doc / per-query torch tensors.

    The on-disk NPZ stores `doc_vectors` as a single concatenated [N_total, dim]
    fp16 array with `doc_offsets[i] = (start, end)` per doc. FastPlaid wants a
    list of [n_tok_i, dim] tensors (per-doc), so we slice once here and reuse
    the same embeddings for both libraries.
    """
    all_vectors, doc_offsets, doc_ids, query_vecs, graded_qrels, dim = load_beir_npz(name)

    doc_tensors: List[torch.Tensor] = [
        torch.from_numpy(np.ascontiguousarray(all_vectors[s:e])).to(torch.float32)
        for s, e in doc_offsets
    ]
    query_tensors: List[torch.Tensor] = [
        torch.from_numpy(np.ascontiguousarray(qv, dtype=np.float32)) for qv in query_vecs
    ]

    return {
        "name": name,
        "all_vectors": all_vectors,
        "doc_offsets": doc_offsets,
        "doc_ids": doc_ids,
        "doc_tensors": doc_tensors,
        "query_tensors": query_tensors,
        "query_vecs_np": query_vecs,
        "graded_qrels": graded_qrels,
        "dim": dim,
        "n_docs": len(doc_ids),
        "n_queries": len(query_vecs),
        "tokens_total": int(all_vectors.shape[0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# voyager-index lane
# ─────────────────────────────────────────────────────────────────────────────

_VOYAGER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "voyager_fp16": {
        "compression": Compression.FP16,
        "modes": ["gpu"],
        "kwargs": {},
    },
    "voyager_rroq158_gs32": {
        "compression": Compression.RROQ158,
        "modes": ["gpu"],
        "kwargs": {"rroq158_group_size": 32},
    },
    "voyager_rroq158_gs128": {
        "compression": Compression.RROQ158,
        "modes": ["gpu"],
        "kwargs": {"rroq158_group_size": 128},
    },
    "voyager_rroq4_riem": {
        "compression": Compression.RROQ4_RIEM,
        "modes": ["gpu"],
        "kwargs": {},
    },
}


# Root of the on-disk shard-bench cache. Each dataset can grow ~30 GB during
# build (fp16 + rroq158 variants + intermediate `index_*` directory). We GC
# per-dataset after all 4 lanes complete to keep peak disk = max(per-dataset)
# rather than Σ(per-dataset).
_SHARD_BENCH_ROOT = Path("/root/.cache/shard-bench")


def _cleanup_dataset_artifacts(dataset: str) -> None:
    """Remove built shard-bench indexes + intermediate build dirs for a
    finished dataset. Safe no-op if any path is missing.
    """
    targets: list[Path] = []
    ds_dir = _SHARD_BENCH_ROOT / "beir" / dataset
    if ds_dir.exists():
        targets.append(ds_dir)
    # Intermediate `index_<n_docs>_<algo>_proxy_grouped_lemur_uniform` build
    # dirs that the pipeline writes before sealing into the final per-dataset
    # location. Anything left behind (e.g. after an early failure) is dead
    # weight on the next dataset.
    for d in glob.glob(str(_SHARD_BENCH_ROOT / "index_*_proxy_grouped_lemur_uniform")):
        targets.append(Path(d))
    freed = 0
    for p in targets:
        try:
            if p.is_dir():
                # Cheap pre-stat — du takes ~1 s, but Path.stat over the
                # tree is too slow for our purposes; just count via shutil.
                freed_before = shutil.disk_usage(p.parent).free
                shutil.rmtree(p)
                freed_after = shutil.disk_usage(p.parent).free
                freed += max(0, freed_after - freed_before)
        except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
            logger = logging.getLogger("fast_plaid_h2h")
            logger.warning("cleanup: failed to rm %s (%s)", p, exc)
    if targets:
        logger = logging.getLogger("fast_plaid_h2h")
        logger.info(
            "cleanup[%s]: removed %d artefact dir(s), freed ~%.1f GB",
            dataset, len(targets), freed / 1e9,
        )


def run_voyager_lane(
    library: str,
    name: str,
    *,
    n_eval: int = 0,
    mode: str = "gpu",
) -> Dict[str, Any]:
    """Run a voyager-index lane via the existing benchmark pipeline.

    `mode` is "gpu" or "cpu" — the underlying `voyager_run_dataset` already
    knows how to dispatch each (rroq158/fp16 each have CPU/GPU lanes).
    Returns the FastPlaid-shaped row: {dataset, library, ndcg_at_10,
    indexing_time_s, qps, ...metadata}.
    """
    cfg = _VOYAGER_CONFIGS[library]
    if mode not in ("gpu", "cpu"):
        raise ValueError(f"Unknown voyager mode: {mode!r}")
    log.info("[voyager:%s/%s] %s: building + searching ...", library, mode, name)
    t_total = time.perf_counter()
    rows = voyager_run_dataset(
        name,
        modes=[mode],
        n_eval=n_eval,
        compression=cfg["compression"],
        **cfg["kwargs"],
    )
    total_wall_s = time.perf_counter() - t_total

    want_substr = "GPU" if mode == "gpu" else "CPU"
    matching = [r for r in rows if want_substr in r.get("mode", "")]
    if not matching:
        raise RuntimeError(
            f"voyager lane {library!r} returned no {mode.upper()} row for {name!r}"
        )
    r = matching[0]

    # `indexing_docs_per_sec` is reported as docs/sec; convert back to wall-s.
    n_docs = int(r.get("n_docs", 0))
    docs_per_s = r.get("indexing_docs_per_sec")
    indexing_time_s: Optional[float]
    if docs_per_s in (None, 0, float("inf")):
        # Reused a cached LEMUR build (build_index hit the warm-cache path).
        indexing_time_s = None
    else:
        indexing_time_s = n_docs / float(docs_per_s) if docs_per_s > 0 else None

    return {
        "dataset": name,
        "library": f"{library}_{mode}",
        "ndcg_at_10": float(r.get("NDCG@10", 0.0)),
        "recall_at_100": float(r.get("recall@100", 0.0)),
        "indexing_time_s": indexing_time_s,
        "qps": float(r.get("qps", 0.0)),
        "p50_ms": float(r.get("p50_ms", 0.0)),
        "p95_ms": float(r.get("p95_ms", 0.0)),
        "n_docs": n_docs,
        "n_queries": int(r.get("n_queries", 0)),
        "wall_s": total_wall_s,
        "params": r.get("params", {}),
        "device": mode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FastPlaid lane
# ─────────────────────────────────────────────────────────────────────────────


def run_fast_plaid_lane(
    name: str,
    *,
    n_eval: int = 0,
    device: str = "cuda:0",
    n_warmup: int = 10,
    index_root: Optional[Path] = None,
    time_budget_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the FastPlaid lane on the same prepared embeddings.

    FastPlaid's `search.FastPlaid` accepts pre-computed per-doc token
    embeddings via `documents_embeddings=`, so we reuse the exact same
    fp16 vectors that came out of the GTE-ModernColBERT encode pass —
    pinning the model lets us attribute any quality / throughput delta
    to the indexing engine, not the embedding model.
    """
    fp_search = _import_fast_plaid()
    bundle = _load_npz_as_tensors(name)
    n_docs = bundle["n_docs"]

    # FastPlaid persists the index to disk; use a tempdir to keep the
    # benchmark idempotent (no left-over state between datasets).
    if index_root is None:
        index_root = Path(tempfile.mkdtemp(prefix=f"fast_plaid_{name}_"))
    else:
        index_root.mkdir(parents=True, exist_ok=True)

    log.info(
        "[fast_plaid] %s: indexing %d docs (%d tokens, dim=%d) on %s ...",
        name, n_docs, bundle["tokens_total"], bundle["dim"], device,
    )
    fp = fp_search.FastPlaid(index=str(index_root), device=device)
    n_kmeans = min(bundle["tokens_total"], FAST_PLAID_KMEANS_CAP)

    t_index = time.perf_counter()
    create_kwargs: Dict[str, Any] = {"documents_embeddings": bundle["doc_tensors"]}
    # `n_samples_kmeans` is a memory-efficient knob added in FastPlaid 1.10;
    # silently drop it on older versions so the script stays portable.
    try:
        fp.create(**create_kwargs, n_samples_kmeans=n_kmeans)
    except TypeError:
        fp.create(**create_kwargs)
    indexing_time_s = time.perf_counter() - t_index
    log.info("[fast_plaid] %s: indexing finished in %.2f s", name, indexing_time_s)

    queries = bundle["query_tensors"]
    if n_eval and n_eval > 0:
        queries = queries[:n_eval]
    n_q = len(queries)

    # Single-client, sequential-query QPS (matches FastPlaid's published
    # benchmark methodology). Warm up the kernel + KV caches first; do
    # not include warmup in the timed window.
    log.info("[fast_plaid] %s: warming up %d queries ...", name, min(n_warmup, n_q))
    for qi in range(min(n_warmup, n_q)):
        # FastPlaid expects a 3-D tensor [batch=1, n_query_tok, dim].
        qt = queries[qi].unsqueeze(0)
        fp.search(queries_embeddings=qt, top_k=TOP_K)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    log.info(
        "[fast_plaid] %s: timing %d queries (time_budget=%s) ...",
        name, n_q, f"{time_budget_s:.0f}s" if time_budget_s else "none",
    )
    all_ids: List[List[int]] = []
    all_elapsed_ms: List[float] = []
    t_search_wall = time.perf_counter()
    n_completed = 0
    for qi in range(n_q):
        qt = queries[qi].unsqueeze(0)
        t0 = time.perf_counter()
        result = fp.search(queries_embeddings=qt, top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        all_elapsed_ms.append(elapsed_ms)
        ids = _extract_top_ids(result)
        all_ids.append(ids)
        n_completed = qi + 1
        # Adaptive early-stop: bail out if we've burned the time budget AND
        # have a statistically meaningful sample (>=128 queries). This keeps
        # the bench tractable on slow lanes (e.g. fast_plaid CPU on quora 522k)
        # while still producing a representative QPS / NDCG measurement.
        if time_budget_s is not None and (time.perf_counter() - t_search_wall) >= time_budget_s and n_completed >= 128:
            log.info(
                "[fast_plaid] %s: time-budget reached after %d/%d queries (%.1fs); stopping early",
                name, n_completed, n_q, time.perf_counter() - t_search_wall,
            )
            break
    wall_s = time.perf_counter() - t_search_wall
    n_q = n_completed

    qps = n_q / wall_s if wall_s > 0 else 0.0
    p50 = float(np.median(all_elapsed_ms)) if all_elapsed_ms else 0.0
    p95 = float(np.percentile(all_elapsed_ms, 95)) if all_elapsed_ms else 0.0

    metrics = evaluate_retrieval(all_ids, bundle["graded_qrels"], n_q)

    # Free the index; FastPlaid keeps a Rust-side handle alive.
    del fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fp_mode = "gpu" if str(device).startswith("cuda") else "cpu"
    return {
        "dataset": name,
        "library": f"fast_plaid_{fp_mode}",
        "ndcg_at_10": float(metrics.get("NDCG@10", 0.0)),
        "recall_at_100": float(metrics.get("recall@100", 0.0)),
        "indexing_time_s": float(indexing_time_s),
        "qps": float(qps),
        "p50_ms": p50,
        "p95_ms": p95,
        "n_docs": n_docs,
        "n_queries": n_q,
        "wall_s": wall_s,
        "params": {
            "device": device,
            "top_k": str(TOP_K),
            "n_samples_kmeans": str(n_kmeans),
            "index_root": str(index_root),
        },
        "device": fp_mode,
    }


def _extract_top_ids(fast_plaid_result: Any) -> List[int]:
    """Normalize FastPlaid's `search()` output into a flat list of doc ids.

    FastPlaid's API returns one of:
      - `[(doc_idx, score), ...]` per query
      - `[[doc_idx, ...], [score, ...]]` per query (list-of-pairs flavor)
      - a tensor-shaped `(ids, scores)` tuple per batch
    We tolerate all three so the benchmark stays robust across point
    releases.
    """
    if fast_plaid_result is None:
        return []
    # Most common shape: list with one element per query in the batch.
    first = fast_plaid_result[0] if len(fast_plaid_result) else None
    if first is None:
        return []

    # Tuple of (ids, scores) tensors / lists.
    if isinstance(first, tuple) and len(first) == 2:
        ids, _scores = first
        return [int(x) for x in (ids.tolist() if hasattr(ids, "tolist") else ids)]

    # List of (doc_idx, score) pairs.
    if isinstance(first, (list, tuple)):
        return [int(pair[0]) for pair in first]

    # Tensor of doc ids only.
    if hasattr(first, "tolist"):
        return [int(x) for x in first.tolist()]

    raise TypeError(
        f"Unrecognized FastPlaid search() result shape: {type(first)!r}; "
        f"first row: {first!r}. Update _extract_top_ids() in "
        "benchmarks/fast_plaid_head_to_head.py for the new FastPlaid API."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting (FastPlaid-style table)
# ─────────────────────────────────────────────────────────────────────────────

# Column widths chosen so the FastPlaid-published table prints right-aligned.
_COL_WIDTHS = {
    "dataset": 16,
    "size": 8,
    "library": 26,
    "ndcg": 8,
    "indexing": 18,
    "qps": 28,
}


def _fmt_qps_with_speedup(qps: float, baseline_qps: Optional[float]) -> str:
    """Format QPS with a `(+xx%)` speed-up against the baseline (PLAID line)."""
    base = f"{qps:>10.2f}"
    if baseline_qps is None or baseline_qps <= 0 or qps <= 0:
        return base
    pct = (qps - baseline_qps) / baseline_qps * 100.0
    return f"{base} ({pct:+.0f}%)"


def _fmt_indexing(t: Optional[float]) -> str:
    if t is None:
        return "       cached"
    return f"{t:>14.2f}"


def format_fast_plaid_style_table(rows: List[Dict[str, Any]]) -> str:
    """Print a table shaped like FastPlaid's published numbers.

    Groups rows by dataset, prints one line per library per dataset.
    The first library row per dataset prints the dataset / size columns;
    subsequent rows blank them out (matching FastPlaid's table).
    """
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    header = (
        f"{'Dataset':<{_COL_WIDTHS['dataset']}}"
        f"{'Size':>{_COL_WIDTHS['size']}}  "
        f"{'Library':<{_COL_WIDTHS['library']}}"
        f"{'NDCG@10':>{_COL_WIDTHS['ndcg']}}"
        f"{'Indexing Time (s)':>{_COL_WIDTHS['indexing']}}"
        f"{'Queries per second (QPS)':>{_COL_WIDTHS['qps']}}"
    )
    lines = [header, "-" * len(header)]

    for ds in sorted(by_dataset.keys()):
        ds_rows = by_dataset[ds]
        # "fast_plaid" (any device suffix) is the canonical baseline for the
        # speed-up column; pick the matching device per row when available.
        gpu_baseline = next(
            (r["qps"] for r in ds_rows if r["library"] == "fast_plaid_gpu"), None,
        )
        cpu_baseline = next(
            (r["qps"] for r in ds_rows if r["library"] == "fast_plaid_cpu"), None,
        )
        legacy_baseline = next(
            (r["qps"] for r in ds_rows if r["library"] == "fast_plaid"), None,
        )
        for i, r in enumerate(ds_rows):
            ds_label = r["dataset"] if i == 0 else ""
            size_label = f"{r['n_docs']:>{_COL_WIDTHS['size']}d}" if i == 0 else " " * _COL_WIDTHS["size"]
            row_baseline = (
                gpu_baseline if r.get("device") == "gpu"
                else cpu_baseline if r.get("device") == "cpu"
                else legacy_baseline
            )
            qps_str = _fmt_qps_with_speedup(
                r["qps"],
                row_baseline if not r["library"].startswith("fast_plaid") else None,
            )
            lines.append(
                f"{ds_label:<{_COL_WIDTHS['dataset']}}"
                f"{size_label}  "
                f"{r['library']:<{_COL_WIDTHS['library']}}"
                f"{r['ndcg_at_10']:>{_COL_WIDTHS['ndcg']}.4f}"
                f"{_fmt_indexing(r['indexing_time_s'])}"
                f"{qps_str:>{_COL_WIDTHS['qps']}}"
            )
    return "\n".join(lines)


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Markdown variant of the table for direct paste into a docs / blog post."""
    out = [
        "| Dataset | Size | Library | NDCG@10 | Indexing time (s) | QPS |",
        "|---------|-----:|---------|--------:|------------------:|----:|",
    ]
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    for ds in sorted(by_dataset.keys()):
        ds_rows = by_dataset[ds]
        for i, r in enumerate(ds_rows):
            ds_label = r["dataset"] if i == 0 else ""
            size_label = f"{r['n_docs']:,}" if i == 0 else ""
            indexing_label = "cached" if r["indexing_time_s"] is None else f"{r['indexing_time_s']:.2f}"
            out.append(
                f"| {ds_label} | {size_label} | `{r['library']}` "
                f"| {r['ndcg_at_10']:.4f} | {indexing_label} | {r['qps']:.2f} |"
            )
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate verdict
# ─────────────────────────────────────────────────────────────────────────────


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-library means across all datasets in the run."""
    by_lib: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_lib.setdefault(r["library"], []).append(r)
    summary: Dict[str, Any] = {}
    for lib, lib_rows in by_lib.items():
        ndcgs = [r["ndcg_at_10"] for r in lib_rows]
        qpss = [r["qps"] for r in lib_rows]
        idx_ts = [r["indexing_time_s"] for r in lib_rows if r["indexing_time_s"] is not None]
        summary[lib] = {
            "datasets": [r["dataset"] for r in lib_rows],
            "n_datasets": len(lib_rows),
            "ndcg_at_10_mean": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "qps_mean": float(np.mean(qpss)) if qpss else 0.0,
            "qps_geomean": float(np.exp(np.mean(np.log([q for q in qpss if q > 0])))) if any(q > 0 for q in qpss) else 0.0,
            "indexing_time_s_total": float(np.sum(idx_ts)) if idx_ts else 0.0,
        }
    return summary


def format_summary(summary: Dict[str, Any]) -> str:
    out = [
        "",
        "Per-library means across run:",
        f"  {'Library':<28} {'NDCG@10 mean':>14} {'QPS mean':>12} {'QPS geomean':>13} {'Total index (s)':>18}",
        "  " + "-" * 88,
    ]
    for lib in sorted(summary.keys()):
        s = summary[lib]
        out.append(
            f"  {lib:<28} {s['ndcg_at_10_mean']:>14.4f} {s['qps_mean']:>12.2f} "
            f"{s['qps_geomean']:>13.2f} {s['indexing_time_s_total']:>18.2f}"
        )
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _check_npz_present(datasets: Sequence[str]) -> List[str]:
    missing = [d for d in datasets if not (BEIR_CACHE / f"{d}.npz").exists()]
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="voyager-index vs FastPlaid head-to-head BEIR benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=FAST_PLAID_BEIR_8,
        help=f"Datasets to run (default: FastPlaid's BEIR-8 = {FAST_PLAID_BEIR_8})",
    )
    parser.add_argument(
        "--libraries",
        nargs="*",
        default=DEFAULT_LIBRARIES,
        choices=sorted(SUPPORTED_LIBRARIES.keys()),
        help=(
            "Libraries to benchmark (default: voyager_rroq158_gs128 + fast_plaid). "
            "Use `voyager_fp16 voyager_rroq158_gs128 fast_plaid` for the 3-way table."
        ),
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=0,
        help="Per-dataset query cap. Use 0 (default) for the full BEIR query set "
        "— matches FastPlaid's published table.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for the FastPlaid GPU lane (default: cuda:0). "
        "Used only when --modes includes 'gpu'.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["gpu"],
        choices=["gpu", "cpu"],
        help=(
            "Which device modes to run for each library (default: gpu only). "
            "Use `--modes gpu cpu` for the full 6-way matrix "
            "(voyager_fp16/voyager_rroq158_gs128/fast_plaid each x gpu/cpu)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/fast_plaid_head_to_head/results.jsonl",
        help="Where to write the per-row JSONL (default: reports/fast_plaid_head_to_head/results.jsonl)",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="reports/fast_plaid_head_to_head/summary.json",
        help="Where to write the per-library summary JSON",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip datasets whose NPZ is not yet prepared, instead of aborting.",
    )
    parser.add_argument(
        "--fast-plaid-cpu-time-budget-s",
        type=float,
        default=180.0,
        help=(
            "Wall-time budget (seconds) for the fast_plaid CPU search loop "
            "before early-stop with a partial QPS/NDCG sample. Required to "
            "keep the bench tractable on large corpora (e.g. quora 522k). "
            "Set 0 to disable. Default: 180s."
        ),
    )
    args = parser.parse_args()

    log.info("=" * 78)
    log.info("HEAD-TO-HEAD: voyager-index vs FastPlaid")
    log.info("Datasets : %s", args.datasets)
    log.info("Libraries: %s", args.libraries)
    log.info("Device   : %s (FastPlaid lane); voyager auto-selects cuda if available", args.device)
    log.info("=" * 78)

    missing = _check_npz_present(args.datasets)
    if missing:
        msg = (
            f"Missing prepared NPZ for datasets {missing}. "
            f"Run: python benchmarks/data/prepare_beir_datasets.py --datasets {' '.join(missing)}"
        )
        if not args.allow_missing:
            log.error(msg)
            return 2
        log.warning(msg + "  (--allow-missing: skipping)")
        args.datasets = [d for d in args.datasets if d not in missing]

    if not torch.cuda.is_available():
        log.warning(
            "No CUDA device detected. The published FastPlaid table is GPU-only; "
            "the comparison still runs on CPU but the QPS numbers will not be "
            "comparable to the published row."
        )

    # Resilient: write each row to JSONL as soon as it completes, and
    # skip cells already present (allows resume after a kill / crash).
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    completed: set = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows.append(r)
                completed.add((r.get("dataset"), r.get("library")))
            except json.JSONDecodeError:
                continue
        if completed:
            log.info("Resuming: found %d completed cells in %s", len(completed), out_path)

    def _persist_row(r: Dict[str, Any]) -> None:
        rows.append(r)
        with out_path.open("a") as fh:
            fh.write(json.dumps(r) + "\n")

    for ds in args.datasets:
        log.info("─" * 78)
        log.info("DATASET: %s", ds)
        log.info("─" * 78)

        for lib in args.libraries:
            for mode in args.modes:
                lane_label = f"{lib}/{mode}"
                row_key = (ds, f"{lib}_{mode}")
                if row_key in completed:
                    log.info("→ lane: %s — SKIP (already in %s)", lane_label, out_path)
                    continue
                log.info("→ lane: %s (%s)", lane_label, SUPPORTED_LIBRARIES[lib])
                try:
                    if lib == "fast_plaid":
                        fp_device = args.device if mode == "gpu" else "cpu"
                        # Apply a wall-time budget on fast_plaid CPU only;
                        # GPU is fast enough to time the full query set.
                        budget = (
                            args.fast_plaid_cpu_time_budget_s
                            if (mode == "cpu" and args.fast_plaid_cpu_time_budget_s and args.fast_plaid_cpu_time_budget_s > 0)
                            else None
                        )
                        row = run_fast_plaid_lane(
                            ds, n_eval=args.n_eval, device=fp_device,
                            time_budget_s=budget,
                        )
                    elif lib in _VOYAGER_CONFIGS:
                        row = run_voyager_lane(lib, ds, n_eval=args.n_eval, mode=mode)
                    else:
                        raise RuntimeError(f"Unknown library: {lib}")
                except Exception as exc:  # noqa: BLE001 — keep going; report at end
                    log.error("Cell failed (%s, %s): %s", ds, lane_label, exc, exc_info=True)
                    _persist_row({
                        "dataset": ds,
                        "library": f"{lib}_{mode}",
                        "ndcg_at_10": 0.0,
                        "recall_at_100": 0.0,
                        "indexing_time_s": None,
                        "qps": 0.0,
                        "p50_ms": 0.0,
                        "p95_ms": 0.0,
                        "n_docs": 0,
                        "n_queries": 0,
                        "wall_s": 0.0,
                        "device": mode,
                        "error": str(exc),
                    })
                    continue

                log.info(
                    "← %s/%s: NDCG@10=%.4f  index=%s  QPS=%.2f  p50=%.1fms  p95=%.1fms",
                    ds, lane_label, row["ndcg_at_10"],
                    "cached" if row["indexing_time_s"] is None else f"{row['indexing_time_s']:.2f}s",
                    row["qps"], row["p50_ms"], row["p95_ms"],
                )
                _persist_row(row)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        _cleanup_dataset_artifacts(ds)

    # ── Output ──────────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("HEAD-TO-HEAD RESULTS — same per-token embeddings (GTE-ModernColBERT-v1) into both libraries")
    print("=" * 100)
    print(format_fast_plaid_style_table(rows))

    summary = summarize(rows)
    print(format_summary(summary))

    print()
    print("Markdown table (paste-ready):")
    print(format_markdown_table(rows))

    # ── Persist ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    log.info("Per-row results written to %s", out_path)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "datasets": args.datasets,
        "libraries": args.libraries,
        "summary": summary,
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "voyager_index_version": _voyager_version(),
            "fast_plaid_version": _fast_plaid_version(),
        },
    }, indent=2, default=str), encoding="utf-8")
    log.info("Summary written to %s", summary_path)

    failed = [r for r in rows if "error" in r]
    if failed:
        log.error("%d / %d cells failed; see error fields in %s", len(failed), len(rows), out_path)
        return 1
    return 0


def _voyager_version() -> Optional[str]:
    try:
        import voyager_index
        return getattr(voyager_index, "__version__", None)
    except Exception:
        return None


def _fast_plaid_version() -> Optional[str]:
    try:
        import fast_plaid
        return getattr(fast_plaid, "__version__", None)
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
