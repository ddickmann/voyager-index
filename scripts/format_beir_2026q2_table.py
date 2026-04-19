"""Format the BEIR 2026-Q2 sweep JSONL into the README codec table.

Reads the per-cell JSONL written by ``benchmarks/beir_2026q2_full_sweep.py``
and emits one of:

  - ``--format=markdown`` (default): the markdown table block intended for
    the README. One row per (codec, dataset) pair, columns:
      Dataset, Documents, NDCG@10, NDCG@100, Recall@10, Recall@100,
      MAP@100, GPU QPS, GPU P95 (ms), CPU QPS, CPU P95 (ms).
  - ``--format=summary``: per-codec averages across the 6 datasets, used
    for the README verdict block and the default-decision rule (Phase G).
  - ``--format=jsonl``: re-emit the cells as a flat JSONL with the
    `cell` key as a primary identifier (useful for downstream charting).

Usage:
    python scripts/format_beir_2026q2_table.py reports/beir_2026q2/sweep.jsonl
    python scripts/format_beir_2026q2_table.py reports/beir_2026q2/sweep.jsonl \
        --format summary
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

CODEC_ORDER = ["fp16", "int8", "rroq158", "rroq4_riem"]
DATASET_ORDER = ["arguana", "fiqa", "nfcorpus", "quora", "scidocs", "scifact"]

# Datasets where rroq158 averages must hit >= -1.5 NDCG@10 vs FP16
# AND <= 1.2x p95 vs FP16 to keep the default. See `default_decision`
# todo in the production validation plan.
NDCG_DROP_BUDGET_PT = -1.5
P95_RATIO_BUDGET = 1.2


def _load_cells(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _key(ds: str, codec: str) -> str:
    return f"{ds}/{codec}"


def _merge_modes(cells: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group cells by (dataset, codec) and merge the gpu/cpu sub-rows.

    Returns a dict keyed by ``"<dataset>/<codec>"`` mapping to a dict
    with quality + per-mode latency fields.
    """
    by_pair: Dict[str, Dict[str, Any]] = {}
    for c in cells:
        ds = c["dataset"]
        cc = c["codec"]
        md = c["mode"]
        key = _key(ds, cc)
        out = by_pair.setdefault(key, {
            "dataset": ds, "codec": cc,
            "n_docs": c.get("n_docs"),
            "ndcg_at_10": None, "ndcg_at_100": None,
            "recall_at_10": None, "recall_at_100": None,
            "map_at_100": None,
            "gpu_qps": None, "gpu_p95_ms": None,
            "cpu_qps": None, "cpu_p95_ms": None,
            "gpu_skipped": False, "cpu_skipped": False,
            "skip_reasons": [],
        })

        if c.get("n_docs") and not out["n_docs"]:
            out["n_docs"] = c["n_docs"]

        if c.get("skipped"):
            out[f"{md}_skipped"] = True
            sr = c.get("skip_reason")
            if sr:
                out["skip_reasons"].append(f"{md}: {sr}")
            continue

        if md == "gpu":
            out["gpu_qps"] = c.get("qps")
            out["gpu_p95_ms"] = c.get("p95_ms")
        elif md == "cpu":
            out["cpu_qps"] = c.get("qps")
            out["cpu_p95_ms"] = c.get("p95_ms")

        # Quality metrics: GPU and CPU should match modulo fp32 rounding.
        # Take the first present value, but verify the cross-mode match
        # within tolerance and log if a mismatch exceeds 0.01 NDCG@10.
        for k in ("ndcg_at_10", "ndcg_at_100", "map_at_100",
                  "recall_at_10", "recall_at_100"):
            v_new = c.get(k)
            if v_new is None:
                continue
            v_old = out[k]
            if v_old is None:
                out[k] = v_new
            elif abs(float(v_old) - float(v_new)) > 0.01:
                print(
                    f"WARNING: {key} {k} mismatch across modes: "
                    f"{v_old:.4f} vs {v_new:.4f}", file=sys.stderr,
                )
    return by_pair


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "n/a"
    return f"{int(v):,}"


def _fmt_lat(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}"


def render_markdown(by_pair: Dict[str, Dict[str, Any]]) -> str:
    out = []
    out.append("Encoder: lightonai/GTE-ModernColBERT-v1. CPU lane uses 8 native Rust workers.")
    out.append("")
    for codec in CODEC_ORDER:
        out.append(f"### `{codec}`")
        out.append("")
        out.append(
            "| Dataset | Docs | NDCG@10 | NDCG@100 | MAP@100 | Recall@10 | Recall@100 "
            "| GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |"
        )
        out.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for ds in DATASET_ORDER:
            r = by_pair.get(_key(ds, codec))
            if r is None:
                continue
            cpu_q = "N/A" if r["cpu_skipped"] else _fmt_lat(r["cpu_qps"])
            cpu_p = "N/A" if r["cpu_skipped"] else _fmt_lat(r["cpu_p95_ms"])
            gpu_q = "N/A" if r["gpu_skipped"] else _fmt_lat(r["gpu_qps"])
            gpu_p = "N/A" if r["gpu_skipped"] else _fmt_lat(r["gpu_p95_ms"])
            out.append(
                f"| {ds} | {_fmt_int(r['n_docs'])} "
                f"| {_fmt(r['ndcg_at_10'])} | {_fmt(r['ndcg_at_100'])} "
                f"| {_fmt(r['map_at_100'])} "
                f"| {_fmt(r['recall_at_10'])} | {_fmt(r['recall_at_100'])} "
                f"| {gpu_q} | {gpu_p} | {cpu_q} | {cpu_p} |"
            )
        out.append("")
    return "\n".join(out)


def render_summary(by_pair: Dict[str, Dict[str, Any]]) -> str:
    """Per-codec averages across the 6 datasets, plus default-decision verdict."""
    avg: Dict[str, Dict[str, Any]] = {}
    for codec in CODEC_ORDER:
        cells = [by_pair[_key(ds, codec)] for ds in DATASET_ORDER
                 if _key(ds, codec) in by_pair]
        if not cells:
            continue

        def mean(field: str) -> Optional[float]:
            vals = [r.get(field) for r in cells if r.get(field) is not None]
            return sum(vals) / len(vals) if vals else None

        avg[codec] = {
            "ndcg_at_10": mean("ndcg_at_10"),
            "ndcg_at_100": mean("ndcg_at_100"),
            "recall_at_10": mean("recall_at_10"),
            "recall_at_100": mean("recall_at_100"),
            "gpu_p95_ms": mean("gpu_p95_ms"),
            "cpu_p95_ms": mean("cpu_p95_ms"),
            "gpu_qps": mean("gpu_qps"),
            "cpu_qps": mean("cpu_qps"),
            "n_cells": len(cells),
        }

    out = ["## BEIR 2026-Q2 codec averages (6-dataset mean)\n"]
    out.append(
        "| Codec | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 "
        "| GPU P95 (ms) | CPU P95 (ms) | GPU QPS | CPU QPS |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for codec in CODEC_ORDER:
        a = avg.get(codec)
        if not a:
            continue
        out.append(
            f"| {codec} | {_fmt(a['ndcg_at_10'])} | {_fmt(a['ndcg_at_100'])} "
            f"| {_fmt(a['recall_at_10'])} | {_fmt(a['recall_at_100'])} "
            f"| {_fmt_lat(a['gpu_p95_ms'])} | {_fmt_lat(a['cpu_p95_ms'])} "
            f"| {_fmt_lat(a['gpu_qps'])} | {_fmt_lat(a['cpu_qps'])} |"
        )
    out.append("")

    # Default-decision rule applied to rroq158 vs fp16
    fp16 = avg.get("fp16")
    rroq = avg.get("rroq158")
    if fp16 and rroq:
        ndcg_drop_pt = (rroq["ndcg_at_10"] - fp16["ndcg_at_10"]) * 100
        if fp16["gpu_p95_ms"] and rroq["gpu_p95_ms"]:
            p95_ratio = rroq["gpu_p95_ms"] / fp16["gpu_p95_ms"]
        else:
            p95_ratio = None
        out.append("## Default-decision rule (Phase G)\n")
        out.append(
            f"- ΔNDCG@10 vs FP16 (avg over 6 datasets): "
            f"**{ndcg_drop_pt:+.2f} pt** "
            f"(budget ≥ {NDCG_DROP_BUDGET_PT:+.1f} pt)"
        )
        if p95_ratio is not None:
            out.append(
                f"- GPU p95 ratio vs FP16 (avg): "
                f"**{p95_ratio:.2f}×** "
                f"(budget ≤ {P95_RATIO_BUDGET:.1f}×)"
            )
        ok_quality = ndcg_drop_pt >= NDCG_DROP_BUDGET_PT
        ok_lat = p95_ratio is None or p95_ratio <= P95_RATIO_BUDGET
        verdict = "KEEP rroq158 as default" if (ok_quality and ok_lat) else "DEMOTE rroq158 to opt-in (revert default to FP16)"
        out.append("")
        out.append(f"### Verdict: **{verdict}**")
        out.append("")
        if not ok_quality:
            out.append(
                f"NDCG drop {ndcg_drop_pt:+.2f} pt exceeds the "
                f"{NDCG_DROP_BUDGET_PT:+.1f} pt budget."
            )
        if not ok_lat and p95_ratio is not None:
            out.append(
                f"GPU p95 ratio {p95_ratio:.2f}x exceeds the "
                f"{P95_RATIO_BUDGET:.1f}x budget."
            )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path,
                        help="Path to beir_2026q2 sweep JSONL")
    parser.add_argument(
        "--format", choices=["markdown", "summary", "jsonl"],
        default="markdown",
    )
    args = parser.parse_args()

    cells = _load_cells(args.input)
    by_pair = _merge_modes(cells)
    if args.format == "markdown":
        print(render_markdown(by_pair))
    elif args.format == "summary":
        print(render_summary(by_pair))
    elif args.format == "jsonl":
        for key in sorted(by_pair):
            print(json.dumps({"pair": key, **by_pair[key]}, default=str))


if __name__ == "__main__":
    main()
