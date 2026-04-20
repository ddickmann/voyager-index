"""Format the BEIR 2026-Q2 sweep JSONL into the README codec table.

Reads the per-cell JSONL written by ``benchmarks/beir_2026q2_full_sweep.py``
and (optionally) the brute-force codec-fidelity overlap JSONL from
``benchmarks/topk_overlap_sweep.py``, then emits one of:

  - ``--format=markdown`` (default): the markdown table block intended for
    the README. One row per (codec, dataset) pair, columns:
      Dataset, Documents, NDCG@10, NDCG@100, Recall@10, Recall@100,
      MAP@100, GPU QPS, GPU P95 (ms), CPU QPS, CPU P95 (ms).
  - ``--format=summary``: per-codec averages across the 6 datasets, plus
    the default-decision verdict block. Two decision rules are
    evaluated in parallel:
      1. **rroq4_riem promotion rule** (Phase F1, new default policy):
         is rroq4_riem within -0.5 pt avg NDCG@10 of fp16, within
         -0.3 pt avg R@100, AND <= fp16 GPU p95 AND <= fp16 CPU p95
         on **every** cell? If yes, keep `Compression.RROQ4_RIEM` as
         the build-time default. If no, revert default to RROQ158 and
         ship the Option-2 honesty-doc framing.
      2. **rroq158 retention rule** (legacy / Option-2 fallback): is
         rroq158 within -1.5 pt avg NDCG@10 and <= 1.2x avg GPU p95
         vs fp16? Used to confirm rroq158 stays a viable storage-saver
         alternative even if the rroq4_riem promotion fails.
  - ``--format=jsonl``: re-emit the cells as a flat JSONL with the
    `cell` key as a primary identifier (useful for downstream charting).

Usage:
    python scripts/format_beir_2026q2_table.py reports/beir_2026q2/sweep.jsonl
    python scripts/format_beir_2026q2_table.py reports/beir_2026q2/sweep.jsonl \
        --format summary
    python scripts/format_beir_2026q2_table.py reports/beir_2026q2/sweep.jsonl \
        --overlap reports/beir_2026q2/topk_overlap.jsonl --format summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CODEC_ORDER = ["fp16", "int8", "rroq158", "rroq4_riem"]
DATASET_ORDER = ["arguana", "fiqa", "nfcorpus", "quora", "scidocs", "scifact"]

# --- rroq4_riem default-promotion rule (Phase F1) -----------------------
# All four conditions must hold across the 6 BEIR datasets to keep the
# build-time default at Compression.RROQ4_RIEM.
RROQ4_RIEM_NDCG_BUDGET_PT = -0.5     # avg NDCG@10 within -0.5 pt of fp16
RROQ4_RIEM_R100_BUDGET_PT = -0.3     # avg R@100 within -0.3 pt of fp16
RROQ4_RIEM_GPU_P95_BUDGET = 1.0      # GPU p95 <= fp16 on every cell
RROQ4_RIEM_CPU_P95_BUDGET = 1.0      # CPU p95 <= fp16 on every cell

# --- rroq158 retention rule (Option-2 / legacy storage-saver) -----------
# Looser thresholds that confirm rroq158 is still a viable opt-in lane
# even if the rroq4_riem promotion rule fails.
RROQ158_NDCG_DROP_BUDGET_PT = -1.5
RROQ158_P95_RATIO_BUDGET = 1.2
# Backwards-compat aliases (older docs / scripts may still reference these).
NDCG_DROP_BUDGET_PT = RROQ158_NDCG_DROP_BUDGET_PT
P95_RATIO_BUDGET = RROQ158_P95_RATIO_BUDGET


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


def _per_codec_averages(
    by_pair: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
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
    return avg


def _per_cell_p95_violations(
    by_pair: Dict[str, Dict[str, Any]],
    codec: str,
    *,
    field: str,
    budget_ratio: float,
) -> List[Tuple[str, float, float, float]]:
    """Return (dataset, fp16_p95, codec_p95, ratio) for cells where the
    codec exceeds (budget_ratio × fp16_p95) on the given p95 field.
    """
    bad: List[Tuple[str, float, float, float]] = []
    for ds in DATASET_ORDER:
        fp = by_pair.get(_key(ds, "fp16"))
        cd = by_pair.get(_key(ds, codec))
        if not (fp and cd):
            continue
        fp_p = fp.get(field)
        cd_p = cd.get(field)
        if fp_p is None or cd_p is None or fp_p <= 0:
            continue
        ratio = cd_p / fp_p
        if ratio > budget_ratio + 1e-6:
            bad.append((ds, float(fp_p), float(cd_p), ratio))
    return bad


def _evaluate_rroq4_riem_promotion(
    by_pair: Dict[str, Dict[str, Any]],
    avg: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply the four-condition decision rule for keeping rroq4_riem default."""
    fp = avg.get("fp16")
    rr4 = avg.get("rroq4_riem")
    if not (fp and rr4):
        return {
            "applied": False,
            "reason": "Missing fp16 or rroq4_riem cells in sweep.",
        }

    if fp.get("ndcg_at_10") is None or rr4.get("ndcg_at_10") is None:
        return {"applied": False, "reason": "Missing NDCG@10 averages."}

    ndcg_drop_pt = (rr4["ndcg_at_10"] - fp["ndcg_at_10"]) * 100
    r100_drop_pt = (
        (rr4["recall_at_100"] - fp["recall_at_100"]) * 100
        if (fp.get("recall_at_100") is not None and rr4.get("recall_at_100") is not None)
        else None
    )

    gpu_violations = _per_cell_p95_violations(
        by_pair, "rroq4_riem", field="gpu_p95_ms",
        budget_ratio=RROQ4_RIEM_GPU_P95_BUDGET,
    )
    cpu_violations = _per_cell_p95_violations(
        by_pair, "rroq4_riem", field="cpu_p95_ms",
        budget_ratio=RROQ4_RIEM_CPU_P95_BUDGET,
    )

    failed: List[str] = []
    if ndcg_drop_pt < RROQ4_RIEM_NDCG_BUDGET_PT - 1e-6:
        failed.append(
            f"avg NDCG@10 drop {ndcg_drop_pt:+.2f} pt is worse than the "
            f"{RROQ4_RIEM_NDCG_BUDGET_PT:+.2f} pt budget"
        )
    if r100_drop_pt is not None and r100_drop_pt < RROQ4_RIEM_R100_BUDGET_PT - 1e-6:
        failed.append(
            f"avg R@100 drop {r100_drop_pt:+.2f} pt is worse than the "
            f"{RROQ4_RIEM_R100_BUDGET_PT:+.2f} pt budget"
        )
    if gpu_violations:
        per_cell = ", ".join(
            f"{ds}: {ratio:.2f}× ({fp_p:.1f}→{cd_p:.1f}ms)"
            for ds, fp_p, cd_p, ratio in gpu_violations
        )
        failed.append(
            f"{len(gpu_violations)} cell(s) exceed GPU p95 budget "
            f"≤{RROQ4_RIEM_GPU_P95_BUDGET:.2f}× fp16 [{per_cell}]"
        )
    if cpu_violations:
        per_cell = ", ".join(
            f"{ds}: {ratio:.2f}× ({fp_p:.1f}→{cd_p:.1f}ms)"
            for ds, fp_p, cd_p, ratio in cpu_violations
        )
        failed.append(
            f"{len(cpu_violations)} cell(s) exceed CPU p95 budget "
            f"≤{RROQ4_RIEM_CPU_P95_BUDGET:.2f}× fp16 [{per_cell}]"
        )

    promote = not failed
    return {
        "applied": True,
        "promote": promote,
        "ndcg_drop_pt": ndcg_drop_pt,
        "r100_drop_pt": r100_drop_pt,
        "gpu_p95_violations": gpu_violations,
        "cpu_p95_violations": cpu_violations,
        "failed_conditions": failed,
    }


def _evaluate_rroq158_retention(avg: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Legacy / Option-2 rule: rroq158 stays viable as opt-in storage-saver."""
    fp = avg.get("fp16")
    rr = avg.get("rroq158")
    if not (fp and rr):
        return {"applied": False, "reason": "Missing fp16 or rroq158 cells."}
    if fp.get("ndcg_at_10") is None or rr.get("ndcg_at_10") is None:
        return {"applied": False, "reason": "Missing NDCG@10 averages."}

    ndcg_drop_pt = (rr["ndcg_at_10"] - fp["ndcg_at_10"]) * 100
    p95_ratio = (
        rr["gpu_p95_ms"] / fp["gpu_p95_ms"]
        if (fp.get("gpu_p95_ms") and rr.get("gpu_p95_ms"))
        else None
    )

    ok_q = ndcg_drop_pt >= RROQ158_NDCG_DROP_BUDGET_PT - 1e-6
    ok_lat = p95_ratio is None or p95_ratio <= RROQ158_P95_RATIO_BUDGET + 1e-6
    return {
        "applied": True,
        "viable_alternative": ok_q and ok_lat,
        "ndcg_drop_pt": ndcg_drop_pt,
        "p95_ratio": p95_ratio,
        "ok_quality": ok_q,
        "ok_latency": ok_lat,
    }


def _format_overlap_block(
    overlap: Dict[str, Dict[str, Any]],
) -> List[str]:
    if not overlap:
        return []
    out = ["## Codec-fidelity overlap (brute-force, vs fp16)\n"]
    out.append(
        "Each cell shows the average per-query top-K overlap of the codec's "
        "brute-force MaxSim ranking with fp16 brute-force MaxSim. 100% means "
        "the codec returns exactly the same documents in the top-K as fp16."
    )
    out.append("")
    out.append("| Dataset | Codec | overlap@10 | overlap@20 | overlap@50 | overlap@100 |")
    out.append("|---|---|---:|---:|---:|---:|")
    for ds in DATASET_ORDER:
        for codec in ("rroq158", "rroq4_riem"):
            row = overlap.get(_key(ds, codec))
            if not row or row.get("skipped"):
                continue
            ov = row.get("topk_overlap_vs_fp16", {})
            out.append(
                f"| {ds} | {codec} "
                f"| {_fmt(ov.get('10'))} | {_fmt(ov.get('20'))} "
                f"| {_fmt(ov.get('50'))} | {_fmt(ov.get('100'))} |"
            )
    out.append("")
    return out


def render_summary(
    by_pair: Dict[str, Dict[str, Any]],
    *,
    overlap: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Per-codec averages, decision verdicts, and (optional) overlap block."""
    avg = _per_codec_averages(by_pair)

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

    # ---- Phase F1 — rroq4_riem default-promotion rule ------------------
    promo = _evaluate_rroq4_riem_promotion(by_pair, avg)
    out.append("## Decision rule 1 — rroq4_riem default promotion (Phase F1)\n")
    if not promo.get("applied"):
        out.append(f"_Skipped: {promo.get('reason', 'unknown')}_")
        out.append("")
    else:
        out.append(
            f"- avg ΔNDCG@10 vs fp16: **{promo['ndcg_drop_pt']:+.2f} pt** "
            f"(budget ≥ {RROQ4_RIEM_NDCG_BUDGET_PT:+.2f} pt)"
        )
        if promo["r100_drop_pt"] is not None:
            out.append(
                f"- avg ΔR@100 vs fp16: **{promo['r100_drop_pt']:+.2f} pt** "
                f"(budget ≥ {RROQ4_RIEM_R100_BUDGET_PT:+.2f} pt)"
            )
        out.append(
            f"- per-cell GPU p95 vs fp16: "
            f"**{len(promo['gpu_p95_violations'])} violation(s)** "
            f"(budget ≤ {RROQ4_RIEM_GPU_P95_BUDGET:.2f}× on every cell)"
        )
        out.append(
            f"- per-cell CPU p95 vs fp16: "
            f"**{len(promo['cpu_p95_violations'])} violation(s)** "
            f"(budget ≤ {RROQ4_RIEM_CPU_P95_BUDGET:.2f}× on every cell)"
        )
        out.append("")
        if promo["promote"]:
            out.append(
                "### Verdict: **KEEP `Compression.RROQ4_RIEM` as the build-time default**"
            )
            out.append("")
            out.append(
                "All four conditions hold. rroq4_riem matches fp16 quality "
                "at lower latency on every cell — the production-default "
                "no-degradation lane stands."
            )
        else:
            out.append(
                "### Verdict: **REVERT default to `Compression.RROQ158`** "
                "(ship Option-2 honesty-doc framing)"
            )
            out.append("")
            out.append("Failed condition(s):")
            for f in promo["failed_conditions"]:
                out.append(f"- {f}")
        out.append("")

    # ---- Option-2 / legacy — rroq158 retention rule -------------------
    ret = _evaluate_rroq158_retention(avg)
    out.append("## Decision rule 2 — rroq158 retention as opt-in storage-saver\n")
    if not ret.get("applied"):
        out.append(f"_Skipped: {ret.get('reason', 'unknown')}_")
    else:
        out.append(
            f"- avg ΔNDCG@10 vs fp16: **{ret['ndcg_drop_pt']:+.2f} pt** "
            f"(budget ≥ {RROQ158_NDCG_DROP_BUDGET_PT:+.2f} pt)"
        )
        if ret["p95_ratio"] is not None:
            out.append(
                f"- avg GPU p95 ratio vs fp16: **{ret['p95_ratio']:.2f}×** "
                f"(budget ≤ {RROQ158_P95_RATIO_BUDGET:.2f}×)"
            )
        viable = ret.get("viable_alternative")
        verdict = (
            "KEEP rroq158 as opt-in storage-saver alternative"
            if viable
            else "rroq158 fails its retention budget — investigate kernel regression"
        )
        out.append("")
        out.append(f"### Verdict: **{verdict}**")
    out.append("")

    # ---- Optional codec-fidelity overlap block ------------------------
    if overlap:
        out.extend(_format_overlap_block(overlap))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Overlap JSONL loader
# ---------------------------------------------------------------------------

def _load_overlap(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read benchmarks/topk_overlap_sweep.py JSONL into {dataset/codec: row}."""
    by_pair: Dict[str, Dict[str, Any]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ds = obj.get("dataset")
            cc = obj.get("codec")
            if not (ds and cc):
                continue
            by_pair[_key(ds, cc)] = obj
    return by_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path,
                        help="Path to beir_2026q2 sweep JSONL")
    parser.add_argument(
        "--format", choices=["markdown", "summary", "jsonl"],
        default="markdown",
    )
    parser.add_argument(
        "--overlap", type=Path, default=None,
        help=(
            "Optional path to topk_overlap_sweep.py JSONL. When provided, "
            "the summary renders the codec-fidelity overlap block; the "
            "decision-rule logic is unchanged."
        ),
    )
    args = parser.parse_args()

    cells = _load_cells(args.input)
    by_pair = _merge_modes(cells)
    overlap = _load_overlap(args.overlap) if args.overlap else None
    if args.format == "markdown":
        print(render_markdown(by_pair))
    elif args.format == "summary":
        print(render_summary(by_pair, overlap=overlap))
    elif args.format == "jsonl":
        for key in sorted(by_pair):
            row = {"pair": key, **by_pair[key]}
            if overlap and key in overlap:
                row["topk_overlap_vs_fp16"] = overlap[key].get(
                    "topk_overlap_vs_fp16"
                )
            print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
