"""BEIR 2026-Q2 full validation sweep — 4 codecs × 6 datasets × 2 modes.

This is the production validation harness referenced by
`docs/guides/rroq-mathematics.md` and the README BEIR table. It drives
`benchmarks/beir_benchmark.run_dataset` once per (codec, dataset, mode)
combination, materialising 42 measurement cells:

    codecs   = [fp16, int8, rroq158, rroq4_riem]    # 4
    datasets = [arguana, fiqa, nfcorpus, quora,     # 6
                scidocs, scifact]
    modes    = [gpu, cpu]                           # 2
    matrix   = 4 × 6 × 2 = 48
    skipped  = 6   (int8 × cpu × 6 datasets — int8 is GPU-only)
    cells    = 42

Each cell is appended as a single JSON object on its own line in the
output JSONL, with full provenance metadata (git sha, host, CUDA
runtime, wheel versions, encoder, kernel parameters, raw qps / p50 /
p95 / NDCG / Recall, plus the n_eval used for each cell). The sweep is
resumable: cells already present in the JSONL are skipped on a re-run.

Companion script: `benchmarks/topk_overlap_sweep.py` produces the
codec-fidelity overlap@K (K ∈ {10,20,50,100}) measurement vs fp16
brute-force MaxSim. The production sweep here measures wrapper-included
end-to-end p95 with LEMUR routing; the overlap sweep isolates pure
codec quality from routing artifacts. Both JSONL outputs are consumed
by `scripts/format_beir_2026q2_table.py` to render the README table
plus the "rroq158 displaces a few ranks lower, recovered by top-20"
honesty callout.

Usage:
    python benchmarks/beir_2026q2_full_sweep.py \
        --output reports/beir_2026q2/sweep.jsonl

    # quick smoke run on two datasets:
    python benchmarks/beir_2026q2_full_sweep.py \
        --datasets fiqa scifact \
        --output /tmp/sweep_smoke.jsonl

    # restrict the matrix:
    python benchmarks/beir_2026q2_full_sweep.py \
        --codecs rroq158 rroq4_riem --modes gpu \
        --output reports/beir_2026q2/rroq_only.jsonl
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beir_benchmark import (
    _COMPRESSION_BY_NAME,
    _GPU_ONLY_COMPRESSIONS,
    run_dataset,
)
from benchmarks.beir_benchmark import (  # noqa: E402
    DATASETS as DEFAULT_DATASETS,
)
from voyager_index._internal.inference.shard_engine.config import Compression  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("beir_sweep")

# Sweep scope: the four "strongest per bit-width" codecs. roq4 is excluded
# because rroq4_riem is its strict Riemannian successor; fp8 is excluded as
# experimental. Add via --codecs to extend.
DEFAULT_CODECS = ["fp16", "int8", "rroq158", "rroq4_riem"]
DEFAULT_MODES = ["gpu", "cpu"]


# ---------------------------------------------------------------------------
# Provenance — collected once per sweep, attached to every cell
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


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return bool(out.stdout.strip())
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


def _cpu_info() -> Dict[str, Any]:
    info = {
        "physical_cores": os.cpu_count(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "host": socket.gethostname(),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["model_name"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return info


def _wheel_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for pkg in (
        "voyager_index", "torch", "triton", "numpy",
        "latence_shard_engine",
    ):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", None)
        except Exception:
            versions[pkg] = None
    return versions


def collect_provenance() -> Dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "gpu": _gpu_info(),
        "cpu": _cpu_info(),
        "wheels": _wheel_versions(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
    }


# ---------------------------------------------------------------------------
# Cell key + resumable JSONL
# ---------------------------------------------------------------------------

def _cell_key(dataset: str, codec: str, mode: str) -> str:
    return f"{dataset}/{codec}/{mode}"


def _load_completed_cells(out_path: Path) -> set:
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
            else:
                ds = obj.get("dataset")
                cc = obj.get("codec")
                md = obj.get("mode")
                if ds and cc and md:
                    completed.add(_cell_key(ds, cc, md))
    return completed


def _append_cell(out_path: Path, cell: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(cell, default=str) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------

def _normalise_metric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _extract_cell_row(
    raw_row: Dict[str, Any],
    *,
    dataset: str, codec: str, mode: str,
    requested_n_eval: Optional[int],
    extra_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "cell": _cell_key(dataset, codec, mode),
        "dataset": dataset,
        "codec": codec,
        "mode": mode,
        "skipped": bool(raw_row.get("skipped", False)),
        "skip_reason": raw_row.get("skip_reason"),
        "n_docs": raw_row.get("n_docs"),
        "n_queries_evaluated": raw_row.get("n_queries"),
        "n_queries_requested": (
            None if requested_n_eval in (None, 0) else int(requested_n_eval)
        ),
        "top_k": raw_row.get("top_k"),
        "ndcg_at_10": _normalise_metric(raw_row.get("NDCG@10")),
        "ndcg_at_100": _normalise_metric(raw_row.get("NDCG@100")),
        "map_at_100": _normalise_metric(raw_row.get("MAP@100")),
        "recall_at_10": _normalise_metric(raw_row.get("recall@10")),
        "recall_at_100": _normalise_metric(raw_row.get("recall@100")),
        "qps": _normalise_metric(raw_row.get("qps")),
        "p50_ms": _normalise_metric(raw_row.get("p50_ms")),
        "p95_ms": _normalise_metric(raw_row.get("p95_ms")),
        "indexing_docs_per_sec": _normalise_metric(
            raw_row.get("indexing_docs_per_sec")
        ),
        "raw_mode_label": raw_row.get("mode"),
        "params": raw_row.get("params"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **extra_meta,
    }


def run_one_cell(
    *,
    dataset: str,
    codec: str,
    mode: str,
    n_workers: int,
    n_eval: Optional[int],
    rroq158_k: int,
    rroq158_group_size: int,
    rroq4_riem_k: int,
    rroq4_riem_group_size: int,
    seed: int,
    extra_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    compression: Optional[Compression] = _COMPRESSION_BY_NAME[codec]
    if mode == "cpu" and compression in _GPU_ONLY_COMPRESSIONS:
        return [{
            "cell": _cell_key(dataset, codec, mode),
            "dataset": dataset, "codec": codec, "mode": mode,
            "skipped": True,
            "skip_reason": f"{codec} is GPU-only (no CPU kernel)",
            "n_docs": None, "n_queries_evaluated": 0,
            "n_queries_requested": (
                None if n_eval in (None, 0) else int(n_eval)
            ),
            "top_k": None,
            "ndcg_at_10": None, "ndcg_at_100": None, "map_at_100": None,
            "recall_at_10": None, "recall_at_100": None,
            "qps": None, "p50_ms": None, "p95_ms": None,
            "indexing_docs_per_sec": None,
            "raw_mode_label": f"CPU-{n_workers}w",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **extra_meta,
        }]

    log.info("=" * 80)
    log.info("CELL %s/%s/%s  n_workers=%d  n_eval=%s",
             dataset, codec, mode, n_workers,
             "ALL" if n_eval in (None, 0) else n_eval)
    log.info("=" * 80)

    t_cell0 = time.time()
    rows = run_dataset(
        dataset, [mode],
        n_workers=n_workers,
        n_eval=n_eval or 0,
        compression=compression,
        rroq158_k=rroq158_k,
        rroq158_group_size=rroq158_group_size,
        rroq158_seed=seed,
        rroq4_riem_k=rroq4_riem_k,
        rroq4_riem_group_size=rroq4_riem_group_size,
        rroq4_riem_seed=seed,
    )
    cell_elapsed_s = time.time() - t_cell0
    log.info("CELL %s/%s/%s done in %.1fs", dataset, codec, mode, cell_elapsed_s)

    out: List[Dict[str, Any]] = []
    for r in rows:
        cell = _extract_cell_row(
            r,
            dataset=dataset, codec=codec, mode=mode,
            requested_n_eval=n_eval,
            extra_meta={**extra_meta, "cell_elapsed_s": cell_elapsed_s},
        )
        out.append(cell)
    if not out:
        out.append({
            "cell": _cell_key(dataset, codec, mode),
            "dataset": dataset, "codec": codec, "mode": mode,
            "skipped": True,
            "skip_reason": "run_dataset returned zero rows",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "cell_elapsed_s": cell_elapsed_s,
            **extra_meta,
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _iter_matrix(
    datasets: List[str], codecs: List[str], modes: List[str],
) -> Iterable[Tuple[str, str, str]]:
    for ds in datasets:
        for cc in codecs:
            for md in modes:
                yield ds, cc, md


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BEIR 2026-Q2 4-codec × 6-dataset × 2-mode validation sweep."
    )
    parser.add_argument(
        "--datasets", nargs="*", default=DEFAULT_DATASETS,
        help=f"Subset of {DEFAULT_DATASETS}. Default: all six.",
    )
    parser.add_argument(
        "--codecs", nargs="*", default=DEFAULT_CODECS,
        choices=list(_COMPRESSION_BY_NAME.keys()),
        help=f"Codecs to sweep. Default: {DEFAULT_CODECS}.",
    )
    parser.add_argument(
        "--modes", nargs="*", default=DEFAULT_MODES, choices=["gpu", "cpu"],
        help="GPU lane uses Triton, CPU lane uses 8-worker Rust SIMD.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=8,
        help="CPU worker count (default 8 to match prior README sweep).",
    )
    parser.add_argument(
        "--n-eval", type=int, default=0,
        help="Per-dataset query budget. 0 (default) = full BEIR query set.",
    )
    parser.add_argument(
        "--rroq158-k", type=int, default=8192,
        help="Centroid count for rroq158 (default 8192).",
    )
    parser.add_argument(
        "--rroq158-group-size", type=int, default=128,
        help="Per-group block size for rroq158 residual scales "
        "(default 128 — the SOTA flip from Phase 8; one scale per token at "
        "dim=128. encode_rroq158 transparently falls back via "
        "_resolve_group_size for incompatible dims. Pin to 32 to reproduce "
        "the pre-SOTA-flip baseline.).",
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

    matrix = list(_iter_matrix(args.datasets, args.codecs, args.modes))
    completed = _load_completed_cells(args.output) if args.resume else set()

    log.info("Sweep matrix: %d cells (%d skipped already complete)",
             len(matrix), len(completed))
    if args.dry_run:
        for ds, cc, md in matrix:
            mark = " (done)" if _cell_key(ds, cc, md) in completed else ""
            print(f"  {ds:>10s}/{cc:<10s}/{md}{mark}")
        return

    provenance = collect_provenance()
    log.info("Provenance: git_sha=%s gpu=%s host=%s",
             provenance["git_sha"], provenance["gpu"].get("name"),
             provenance["cpu"].get("host"))

    sweep_id = f"beir_2026q2_{int(time.time())}"
    base_meta = {
        "sweep_id": sweep_id,
        "provenance": provenance,
        "rroq158_k": args.rroq158_k,
        "rroq158_group_size": args.rroq158_group_size,
        "rroq4_riem_k": args.rroq4_riem_k,
        "rroq4_riem_group_size": args.rroq4_riem_group_size,
        "seed": args.seed,
        "n_workers_cpu": args.n_workers,
    }

    n_done = n_failed = 0
    for ds, cc, md in matrix:
        key = _cell_key(ds, cc, md)
        if args.resume and key in completed:
            log.info("Skipping cell %s (already in %s)", key, args.output)
            continue
        try:
            cell_rows = run_one_cell(
                dataset=ds, codec=cc, mode=md,
                n_workers=args.n_workers, n_eval=args.n_eval,
                rroq158_k=args.rroq158_k,
                rroq158_group_size=args.rroq158_group_size,
                rroq4_riem_k=args.rroq4_riem_k,
                rroq4_riem_group_size=args.rroq4_riem_group_size,
                seed=args.seed,
                extra_meta=base_meta,
            )
            for row in cell_rows:
                _append_cell(args.output, row)
            n_done += 1
        except Exception as exc:
            log.exception("Cell %s FAILED: %s", key, exc)
            err_row = {
                "cell": key,
                "dataset": ds, "codec": cc, "mode": md,
                "skipped": True,
                "skip_reason": f"exception: {type(exc).__name__}: {exc}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                **base_meta,
            }
            _append_cell(args.output, err_row)
            n_failed += 1

    log.info("Sweep complete: %d cells written, %d failed. Output: %s",
             n_done, n_failed, args.output)


if __name__ == "__main__":
    main()
