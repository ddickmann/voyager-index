"""Brute-force top-K overlap of each codec vs fp16 across BEIR datasets.

This is the codec-fidelity *honesty supplement* to
`benchmarks/beir_2026q2_full_sweep.py`:

- The full sweep measures wrapper-included end-to-end p95 with LEMUR
  routing and the production GPU/CPU lanes. It answers "what does a real
  user experience?".
- This script measures *per-query top-K rank overlap* with fp16
  brute-force MaxSim, isolating codec quality from routing artifacts. It
  answers "how often does the codec put the right doc in the user's
  top-10 / top-20 / top-50 / top-100?".

The design rationale: the README needs to honestly characterise the
rroq158 top-K overlap with FP16 — the BEIR sweep shows ~−1.4 pt avg
NDCG@10 with flat R@100, but it does not say *whether* the right docs
are admitted vs simply displaced. This script answers that directly by
running brute-force MaxSim with both codecs and reporting per-query
top-K overlap. The Phase-7 BEIR-6 result (`reports/beir_2026q2/
topk_overlap.jsonl`) is rroq158 averaging ~79% top-10 / ~80% top-100
overlap with FP16 (range 73–83% top-10), with R@100 within −2.1 pt
of FP16 on every dataset — labeled relevant docs are still admitted;
the rank cost is in the displacement of FP16's top-K against the
non-relevant tail. Notably, top-K overlap is roughly flat with K,
so widening the serve window is **not** a reliable rescue mechanism.
rroq4_riem averages ~96% top-10 overlap, confirming the
no-quality-loss positioning.

Output: one JSONL line per (dataset, codec) cell with overlap@{10,20,50,100}
plus the codec's own NDCG@10 / R@100 on the same brute-force scoring
path (for the in-isolation quality measurement that the production
sweep cannot give us).

Usage:
    python benchmarks/topk_overlap_sweep.py \
        --output reports/beir_2026q2/topk_overlap.jsonl

    # quick smoke run on two datasets:
    python benchmarks/topk_overlap_sweep.py \
        --datasets nfcorpus scifact --n-queries 100 \
        --output /tmp/overlap_smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beir_benchmark import DATASETS as DEFAULT_DATASETS  # noqa: E402
from benchmarks.beir_benchmark import evaluate, load_beir_npz
from benchmarks.diag_rroq158_kernel_vs_fp16 import (  # noqa: E402
    fp16_maxsim_topk_gpu_batched,
    rroq4_riem_maxsim_topk_gpu,
    rroq158_maxsim_topk_gpu,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("topk_overlap")

# Codecs that have a brute-force MaxSim implementation in the diag suite.
# fp16 is the baseline (always present); the others compute overlap vs fp16.
DEFAULT_CODECS = ["rroq158", "rroq4_riem"]
DEFAULT_KS: Tuple[int, ...] = (10, 20, 50, 100)


# ---------------------------------------------------------------------------
# Provenance (mirrors beir_2026q2_full_sweep.py for cross-referenceable JSONL)
# ---------------------------------------------------------------------------

def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.stdout.strip()
    except Exception:
        return None


def _gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024 ** 3),
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def collect_provenance() -> Dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "gpu": _gpu_info(),
        "cpu": {
            "physical_cores": os.cpu_count(),
            "platform": platform.platform(),
            "host": socket.gethostname(),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
    }


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------

def topk_overlap(
    fp16_topk: List[List[int]],
    codec_topk: List[List[int]],
    k: int,
) -> Optional[float]:
    """Average per-query overlap of the top-k sets.

    Returns the mean fraction of fp16's top-k document IDs that also
    appear in the codec's top-k for the same query. 1.0 = perfect, 0.0
    = none. Symmetric in the sense that |A ∩ B| / k is identical for A
    and B both of length k.
    """
    if not fp16_topk or not codec_topk:
        return None
    pct: List[float] = []
    for a, b in zip(fp16_topk, codec_topk):
        sa = set(a[:k])
        sb = set(b[:k])
        if not sa:
            continue
        pct.append(len(sa & sb) / float(min(k, len(sa))))
    return float(np.mean(pct)) if pct else None


def _evaluate_against(
    topk: List[List[int]],
    qrels: Dict[int, Dict[int, int]],
    eval_qis: List[int],
) -> Dict[str, float]:
    """Re-key qrels to subset query indices, then evaluate."""
    sub_qrels = {i: qrels[qi] for i, qi in enumerate(eval_qis) if qi in qrels}
    return evaluate(topk, sub_qrels, len(eval_qis))


# ---------------------------------------------------------------------------
# Per-codec brute-force scoring dispatch
# ---------------------------------------------------------------------------

def _run_codec_bruteforce(
    *,
    codec: str,
    eval_queries: List[np.ndarray],
    all_vectors: np.ndarray,
    doc_offsets: list,
    top_k: int,
    device: str,
    rroq158_k: int,
    rroq4_riem_k: int,
    rroq4_riem_group_size: int,
    seed: int,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Dispatch to the right brute-force scorer; returns (topk_ids, info)."""
    t0 = time.time()
    if codec == "rroq158":
        topk, info = rroq158_maxsim_topk_gpu(
            eval_queries, all_vectors, doc_offsets, top_k, device,
            rroq158_k=rroq158_k, seed=seed,
        )
    elif codec == "rroq4_riem":
        topk, info = rroq4_riem_maxsim_topk_gpu(
            eval_queries, all_vectors, doc_offsets, top_k, device,
            k_centroids=rroq4_riem_k, group_size=rroq4_riem_group_size,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown brute-force codec {codec!r}. "
            f"Add a brute-force scorer in diag_rroq158_kernel_vs_fp16.py "
            f"and wire it here, or skip via --codecs."
        )
    info = dict(info)
    info["wallclock_s"] = time.time() - t0
    return topk, info


# ---------------------------------------------------------------------------
# Resumable JSONL output
# ---------------------------------------------------------------------------

def _cell_key(dataset: str, codec: str) -> str:
    return f"{dataset}/{codec}"


def _load_completed(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    completed = set()
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cell = obj.get("cell")
            if cell:
                completed.add(cell)
            elif obj.get("dataset") and obj.get("codec"):
                completed.add(_cell_key(obj["dataset"], obj["codec"]))
    return completed


def _append_cell(out_path: Path, cell: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(cell, default=str) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def run_one_dataset(
    *,
    dataset: str,
    codecs: List[str],
    n_queries: int,
    top_k: int,
    rroq158_k: int,
    rroq4_riem_k: int,
    rroq4_riem_group_size: int,
    seed: int,
    device: str,
    out_path: Path,
    completed: set,
    base_meta: Dict[str, Any],
    ks: Tuple[int, ...],
) -> None:
    """Compute fp16 baseline once per dataset, then loop codecs."""
    log.info("=" * 80)
    log.info("DATASET %s | n_queries=%d | top_k=%d | codecs=%s",
             dataset, n_queries, top_k, codecs)
    log.info("=" * 80)

    # Decide if we need to do the fp16 pass at all.
    needed = [c for c in codecs if _cell_key(dataset, c) not in completed]
    if not needed:
        log.info("All codecs already complete for %s, skipping.", dataset)
        return

    all_vectors, doc_offsets, doc_ids, query_vecs, qrels, dim = load_beir_npz(dataset)
    log.info("%s loaded: %d docs, %d queries, dim=%d",
             dataset, len(doc_offsets), len(query_vecs), dim)

    eval_qis = sorted([qi for qi in qrels.keys() if qi < len(query_vecs)])
    if n_queries > 0:
        eval_qis = eval_qis[:n_queries]
    eval_queries = [query_vecs[qi] for qi in eval_qis]
    log.info("Evaluating on %d queries (subset of %d total in qrels)",
             len(eval_qis), len(qrels))

    # ----- fp16 brute-force ground truth -----
    log.info("fp16 brute-force MaxSim ...")
    t0 = time.time()
    fp_ids = fp16_maxsim_topk_gpu_batched(
        eval_queries, all_vectors, doc_offsets, top_k, device,
    )
    fp_t = time.time() - t0
    fp_metrics = _evaluate_against(fp_ids, qrels, eval_qis)
    log.info("fp16: NDCG@10=%.4f R@100=%.4f R@10=%.4f (%.1fs)",
             fp_metrics["NDCG@10"], fp_metrics["recall@100"],
             fp_metrics["recall@10"], fp_t)

    fp_cell_meta = {
        **base_meta,
        "dataset": dataset,
        "n_docs": len(doc_offsets),
        "n_queries_evaluated": len(eval_qis),
        "top_k": top_k,
        "fp16_brute_force": {
            "ndcg_at_10": fp_metrics["NDCG@10"],
            "ndcg_at_100": fp_metrics["NDCG@100"],
            "recall_at_10": fp_metrics["recall@10"],
            "recall_at_100": fp_metrics["recall@100"],
            "wallclock_s": fp_t,
        },
    }

    # ----- per-codec brute-force + overlap -----
    for codec in codecs:
        key = _cell_key(dataset, codec)
        if key in completed:
            log.info("Cell %s already complete, skipping.", key)
            continue

        log.info("Running brute-force %s on %s ...", codec, dataset)
        try:
            codec_ids, info = _run_codec_bruteforce(
                codec=codec,
                eval_queries=eval_queries,
                all_vectors=all_vectors,
                doc_offsets=doc_offsets,
                top_k=top_k,
                device=device,
                rroq158_k=rroq158_k,
                rroq4_riem_k=rroq4_riem_k,
                rroq4_riem_group_size=rroq4_riem_group_size,
                seed=seed,
            )
        except Exception as exc:
            log.exception("Codec %s on %s FAILED: %s", codec, dataset, exc)
            err_row = {
                "cell": key,
                "dataset": dataset,
                "codec": codec,
                "skipped": True,
                "skip_reason": f"exception: {type(exc).__name__}: {exc}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                **base_meta,
            }
            _append_cell(out_path, err_row)
            continue

        codec_metrics = _evaluate_against(codec_ids, qrels, eval_qis)
        overlap_at: Dict[str, Optional[float]] = {
            str(k): topk_overlap(fp_ids, codec_ids, k) for k in ks
        }

        d_ndcg = codec_metrics["NDCG@10"] - fp_metrics["NDCG@10"]
        d_r100 = codec_metrics["recall@100"] - fp_metrics["recall@100"]

        log.info(
            "%s: NDCG@10=%.4f (Δ=%+.4f) R@100=%.4f (Δ=%+.4f) "
            "overlap@10=%.1f%% @20=%.1f%% @50=%.1f%% @100=%.1f%% (%.1fs)",
            codec, codec_metrics["NDCG@10"], d_ndcg,
            codec_metrics["recall@100"], d_r100,
            (overlap_at["10"] or 0.0) * 100,
            (overlap_at["20"] or 0.0) * 100,
            (overlap_at["50"] or 0.0) * 100,
            (overlap_at["100"] or 0.0) * 100,
            info["wallclock_s"],
        )

        cell_row = {
            **fp_cell_meta,
            "cell": key,
            "codec": codec,
            "skipped": False,
            "ndcg_at_10": codec_metrics["NDCG@10"],
            "ndcg_at_100": codec_metrics["NDCG@100"],
            "recall_at_10": codec_metrics["recall@10"],
            "recall_at_100": codec_metrics["recall@100"],
            "delta_ndcg_at_10_vs_fp16": d_ndcg,
            "delta_recall_at_100_vs_fp16": d_r100,
            "topk_overlap_vs_fp16": overlap_at,
            "codec_info": info,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_cell(out_path, cell_row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brute-force top-K overlap of each codec vs fp16 across BEIR datasets.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=DEFAULT_DATASETS,
        help=f"Subset of {DEFAULT_DATASETS}. Default: all six.",
    )
    parser.add_argument(
        "--codecs", nargs="*", default=DEFAULT_CODECS,
        choices=DEFAULT_CODECS,
        help=f"Codecs to overlap-test against fp16. Default: {DEFAULT_CODECS}.",
    )
    parser.add_argument(
        "--n-queries", type=int, default=0,
        help="Per-dataset query budget. 0 (default) = full BEIR query set.",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Maximum K (must be >= max(--ks)). Default 100.",
    )
    parser.add_argument(
        "--ks", type=int, nargs="*", default=list(DEFAULT_KS),
        help=f"Overlap K values. Default: {list(DEFAULT_KS)}.",
    )
    parser.add_argument(
        "--rroq158-k", type=int, default=8192,
        help="Centroid count for rroq158 (default 8192).",
    )
    parser.add_argument(
        "--rroq4-riem-k", type=int, default=8192,
        help="Centroid count for rroq4_riem (default 8192).",
    )
    parser.add_argument(
        "--rroq4-riem-group-size", type=int, default=32,
        help="Per-group block size for rroq4_riem (default 32).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Codec seed for FWHT + spherical-kmeans init.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="JSONL path. Cells are appended; existing cells are skipped.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cells already present in --output.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the matrix and exit without measuring.",
    )
    args = parser.parse_args()

    ks = tuple(sorted(set(int(k) for k in args.ks)))
    if max(ks) > args.top_k:
        raise SystemExit(
            f"--top-k={args.top_k} must be >= max(--ks)={max(ks)}"
        )

    completed = _load_completed(args.output) if args.resume else set()
    log.info("Overlap matrix: %d datasets × %d codecs (%d cells already complete)",
             len(args.datasets), len(args.codecs), len(completed))

    if args.dry_run:
        for ds in args.datasets:
            for cc in args.codecs:
                mark = " (done)" if _cell_key(ds, cc) in completed else ""
                print(f"  {ds:>10s}/{cc:<12s}{mark}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device != "cuda":
        log.warning("Brute-force MaxSim on CPU is *very* slow — recommend GPU.")

    provenance = collect_provenance()
    sweep_id = f"topk_overlap_{int(time.time())}"
    base_meta = {
        "sweep_id": sweep_id,
        "provenance": provenance,
        "rroq158_k": args.rroq158_k,
        "rroq4_riem_k": args.rroq4_riem_k,
        "rroq4_riem_group_size": args.rroq4_riem_group_size,
        "seed": args.seed,
        "ks": list(ks),
    }

    for ds in args.datasets:
        try:
            run_one_dataset(
                dataset=ds,
                codecs=args.codecs,
                n_queries=args.n_queries,
                top_k=args.top_k,
                rroq158_k=args.rroq158_k,
                rroq4_riem_k=args.rroq4_riem_k,
                rroq4_riem_group_size=args.rroq4_riem_group_size,
                seed=args.seed,
                device=device,
                out_path=args.output,
                completed=completed,
                base_meta=base_meta,
                ks=ks,
            )
        except Exception as exc:
            log.exception("Dataset %s FAILED: %s", ds, exc)
            err_row = {
                "cell": f"{ds}/__dataset_error__",
                "dataset": ds,
                "skipped": True,
                "skip_reason": f"dataset-level exception: {type(exc).__name__}: {exc}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                **base_meta,
            }
            _append_cell(args.output, err_row)

    log.info("Overlap sweep complete. Output: %s", args.output)


if __name__ == "__main__":
    main()
