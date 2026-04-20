#!/usr/bin/env python3
"""rroq158 (K x group_size) Pareto compression probe.

Sweeps a small grid of (K, group_size) configurations on a single BEIR
dataset (default: arguana) to find Pareto-dominant points along the
quality vs. compression frontier.

Each cell runs in its own subprocess so the encoded payload (which can
be hundreds of MB on larger corpora) is fully reclaimed between cells.
The shared FP16 LEMUR routing artifacts in
``~/.cache/shard-bench/beir/<ds>/s32_rroq158_proxy_grouped`` are reused
across cells (codec-agnostic per beir_benchmark.py:213-219), so the
sweep measures only the kernel + codec, not routing variance.

Outputs
-------
- ``reports/rroq158_pareto_cells/<ds>/K<K>_gs<gs>.jsonl``
    Raw beir_benchmark JSONL row for the cell (single line, full metrics).
- ``reports/rroq158_pareto_<ds>.md``
    Aggregated table sorted by predicted bytes/token, with deltas vs.
    the baseline (K=8192, gs=32) and a Pareto-frontier verdict.

Storage math (matches the encoder, dim is read from the result):
    sign+nz = 2 * (dim/8)              bytes (dense planes)
    scales  = (dim/group_size) * 2     bytes (fp16 per-group)
    cid     = 2                        bytes (int16 nominal)
    cos+sin = 4                        bytes (fp16 each)
Total bytes/tok and bits/coord are reported alongside QPS/p95/NDCG@10.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_K_VALUES = (1024, 4096, 8192)
DEFAULT_GS_VALUES = (32, 64, 128)


@dataclass
class CellResult:
    K: int
    group_size: int
    bytes_per_tok: float
    bits_per_coord: float
    qps: float
    p50_ms: float
    p95_ms: float
    ndcg10: float
    ndcg100: float
    recall10: float
    recall100: float
    n_queries: int
    n_docs: int
    dim: int
    raw_path: Path
    elapsed_s: float


def predicted_bytes_per_tok(dim: int, group_size: int) -> float:
    """Mirror the on-disk layout in ``_build_rroq158_gpu_payload``.

    See ``benchmarks/beir_benchmark.py:470-476`` for the canonical formula.
    """
    n_words = dim // 8  # bytes per ternary plane (sign or nz)
    n_groups = max(1, dim // group_size)
    sign_nz = 2 * n_words
    scales = n_groups * 2
    cid = 2
    norms = 4
    return float(sign_nz + scales + cid + norms)


def run_cell(
    dataset: str,
    K: int,
    group_size: int,
    n_workers: int,
    n_eval: int,
    seed: int,
    cells_dir: Path,
    extra_env: Optional[dict] = None,
) -> CellResult:
    """Run a single (K, group_size) cell as a subprocess and parse the JSONL."""
    cell_dir = cells_dir / dataset
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell_jsonl = cell_dir / f"K{K}_gs{group_size}.jsonl"
    cell_log = cell_dir / f"K{K}_gs{group_size}.log"

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "benchmarks" / "beir_benchmark.py"),
        "--datasets", dataset,
        "--modes", "cpu",
        "--n-workers", str(n_workers),
        "--n-eval", str(n_eval),
        "--compression", "rroq158",
        "--rroq158-k", str(K),
        "--rroq158-group-size", str(group_size),
        "--rroq158-seed", str(seed),
        "--output", str(cell_jsonl),
    ]
    print(f"[probe] cell K={K} gs={group_size} -> {cell_jsonl}", flush=True)
    print(f"[probe]   cmd: {' '.join(cmd)}", flush=True)

    t0 = time.time()
    with open(cell_log, "w") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT, env=extra_env)
    elapsed = time.time() - t0

    if rc != 0:
        tail = ""
        try:
            with open(cell_log) as logf:
                tail = "".join(logf.readlines()[-30:])
        except OSError:
            pass
        raise RuntimeError(
            f"Cell K={K} gs={group_size} failed with rc={rc} "
            f"after {elapsed:.1f}s. Log: {cell_log}\nTail:\n{tail}"
        )

    rows = []
    with open(cell_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected 1 result row in {cell_jsonl}, got {len(rows)}"
        )
    r = rows[0]
    dim = int(r["dim"])
    bytes_per_tok = predicted_bytes_per_tok(dim, group_size)
    bits_per_coord = 8.0 * bytes_per_tok / float(dim)
    return CellResult(
        K=K,
        group_size=group_size,
        bytes_per_tok=bytes_per_tok,
        bits_per_coord=bits_per_coord,
        qps=float(r["qps"]),
        p50_ms=float(r["p50_ms"]),
        p95_ms=float(r["p95_ms"]),
        ndcg10=float(r["NDCG@10"]),
        ndcg100=float(r["NDCG@100"]),
        recall10=float(r["recall@10"]),
        recall100=float(r["recall@100"]),
        n_queries=int(r["n_queries"]),
        n_docs=int(r["n_docs"]),
        dim=dim,
        raw_path=cell_jsonl,
        elapsed_s=elapsed,
    )


def find_baseline(results: List[CellResult]) -> CellResult:
    for r in results:
        if r.K == 8192 and r.group_size == 32:
            return r
    raise RuntimeError(
        "Baseline cell (K=8192, gs=32) missing from results - cannot compute deltas"
    )


def is_pareto_dominant(
    cand: CellResult,
    baseline: CellResult,
    ndcg_tolerance: float,
    p95_tolerance: float,
) -> bool:
    """A cell is Pareto-dominant if it strictly improves storage and is
    within tolerance on both quality (NDCG@10) and latency (p95)."""
    storage_better = cand.bytes_per_tok < baseline.bytes_per_tok - 1e-6
    ndcg_ok = cand.ndcg10 >= baseline.ndcg10 - ndcg_tolerance
    p95_ok = cand.p95_ms <= baseline.p95_ms * (1.0 + p95_tolerance)
    return storage_better and ndcg_ok and p95_ok


def write_report(
    dataset: str,
    results: List[CellResult],
    out_path: Path,
    ndcg_tolerance: float,
    p95_tolerance: float,
    n_eval_arg: int,
) -> None:
    baseline = find_baseline(results)

    header = [
        "# rroq158 Pareto compression probe",
        "",
        f"- **Dataset**: `{dataset}` (n_queries={baseline.n_queries:,}, "
        f"n_docs={baseline.n_docs:,}, dim={baseline.dim})",
        f"- **Mode**: CPU-8w (8 native Rust workers, n_eval={n_eval_arg or 'full'})",
        f"- **Baseline cell**: `K=8192, group_size=32` → "
        f"{baseline.bytes_per_tok:.1f} B/tok, "
        f"{baseline.bits_per_coord:.3f} bits/coord, "
        f"NDCG@10={baseline.ndcg10:.4f}, p95={baseline.p95_ms:.0f}ms",
        f"- **Pareto criteria**: storage strictly better, NDCG@10 within "
        f"-{ndcg_tolerance:.4f} of baseline, p95 within +{int(p95_tolerance*100)}% of baseline",
        "",
        "## Per-cell results",
        "",
        "| K | gs | B/tok | bits/coord | QPS | p50 ms | p95 ms | NDCG@10 | "
        "ΔNDCG@10 | Recall@100 | Pareto |",
        "|---|----|-------|-----------:|----:|-------:|-------:|--------:|"
        "---------:|-----------:|--------|",
    ]
    body = []
    for r in sorted(results, key=lambda x: (x.bytes_per_tok, -x.K)):
        d_ndcg = r.ndcg10 - baseline.ndcg10
        is_baseline = (r.K == baseline.K and r.group_size == baseline.group_size)
        if is_baseline:
            tag = "_baseline_"
        elif is_pareto_dominant(r, baseline, ndcg_tolerance, p95_tolerance):
            tag = "**YES**"
        else:
            tag = "no"
        body.append(
            f"| {r.K} | {r.group_size} | {r.bytes_per_tok:.1f} | "
            f"{r.bits_per_coord:.3f} | {r.qps:.1f} | "
            f"{r.p50_ms:.0f} | {r.p95_ms:.0f} | "
            f"{r.ndcg10:.4f} | {d_ndcg:+.4f} | "
            f"{r.recall100:.4f} | {tag} |"
        )

    pareto_winners = [
        r for r in results
        if (r.K, r.group_size) != (baseline.K, baseline.group_size)
        and is_pareto_dominant(r, baseline, ndcg_tolerance, p95_tolerance)
    ]

    verdict = ["", "## Verdict", ""]
    if pareto_winners:
        # Tie-break: smallest bytes/tok first, then highest NDCG@10. This
        # surfaces the strictly-dominant config (better quality at same
        # storage) when multiple cells tie on storage.
        ranked = sorted(
            pareto_winners,
            key=lambda r: (r.bytes_per_tok, -r.ndcg10),
        )
        winner = ranked[0]
        compression_gain = (1.0 - winner.bytes_per_tok / baseline.bytes_per_tok) * 100
        verdict.extend([
            f"**Pareto-dominant config found**: `K={winner.K}, group_size={winner.group_size}`.",
            "",
            f"- Storage: {winner.bytes_per_tok:.1f} B/tok "
            f"({winner.bits_per_coord:.3f} bits/coord) vs. baseline "
            f"{baseline.bytes_per_tok:.1f} B/tok → **{compression_gain:.1f}% smaller**.",
            f"- Quality: NDCG@10={winner.ndcg10:.4f} vs. baseline {baseline.ndcg10:.4f} "
            f"(Δ={winner.ndcg10 - baseline.ndcg10:+.4f}, within tolerance "
            f"-{ndcg_tolerance:.4f}).",
            f"- Latency: p95={winner.p95_ms:.0f}ms vs. baseline {baseline.p95_ms:.0f}ms "
            f"(ratio={winner.p95_ms / baseline.p95_ms:.2f}×).",
        ])
        if len(ranked) > 1:
            verdict.extend([
                "",
                f"_All {len(ranked)} Pareto-dominant cells (storage ASC, NDCG DESC):_",
                "",
                "| K | gs | B/tok | NDCG@10 | ΔNDCG | p95 ms | notes |",
                "|---|----|------:|--------:|------:|-------:|-------|",
            ])
            for i, r in enumerate(ranked):
                d = r.ndcg10 - baseline.ndcg10
                note = "**recommended**" if i == 0 else "alternate"
                verdict.append(
                    f"| {r.K} | {r.group_size} | {r.bytes_per_tok:.1f} | "
                    f"{r.ndcg10:.4f} | {d:+.4f} | {r.p95_ms:.0f} | {note} |"
                )
        verdict.extend([
            "",
            "**Recommended next step**: validate this config on the remaining 5 BEIR datasets "
            "(`fiqa, hotpotqa, nfcorpus, scifact, scidocs`); if quality holds within tolerance "
            "across the suite, open a follow-up PR to flip the production default in "
            "`voyager_index/_internal/inference/quantization/rroq158.py:Rroq158Config`.",
        ])
    else:
        verdict.extend([
            "**No Pareto-dominant uniform config found** — every smaller-storage cell "
            "either degrades NDCG@10 beyond tolerance or regresses p95 latency.",
            "",
            "Why uniform compression hits a floor on dim=128 ModernColBERT:",
            "",
            "- The bulk of the rroq158 payload is the dense ternary sign+nz pair "
            f"(2·dim/8 = {2 * baseline.dim // 8} bytes/tok) which is fixed across all "
            "(K, group_size) configurations. Only the per-group fp16 scales "
            f"(dim/group_size × 2 bytes) shrink with bigger `group_size`, capping "
            f"the win at ~{baseline.bytes_per_tok - predicted_bytes_per_tok(baseline.dim, 128):.0f} B/tok "
            "(~13% of the payload).",
            "- Reducing K only shrinks the codebook in memory; the per-token cid "
            "is still stored as int16 in the on-disk payload (2 bytes regardless of K).",
            "- So the *uniform* knobs we expose here can only buy us small storage gains, "
            "and they cost quality monotonically because the residual scales become "
            "coarser and the centroid table loses resolution.",
            "",
            "### Outlier-rescue follow-up (per the KV-cache hybrid pattern)",
            "",
            "The natural way to push past this floor is a **two-regime hybrid encoding**:",
            "",
            "1. Pick a small fraction `p` of \"rescue\" tokens (the highest-residual ones, "
            "selected by `sin_norm` after centroid lookup).",
            "2. Encode the rescue tokens at the current rich config "
            f"(K=8192, gs=32 → {baseline.bytes_per_tok:.0f} B/tok).",
            "3. Encode the remaining `1-p` tokens at the most aggressive uniform config "
            f"(K=1024, gs=128 → {predicted_bytes_per_tok(baseline.dim, 128):.0f} B/tok).",
            "",
            "Predicted blended storage:",
            "",
            "| p (rescue) | B/tok | bits/coord | quality story |",
            "|-----------:|------:|-----------:|---------------|",
        ])
        rich = baseline.bytes_per_tok
        cheap = predicted_bytes_per_tok(baseline.dim, 128)
        for p in (0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
            b = p * rich + (1 - p) * cheap
            bpc = 8.0 * b / float(baseline.dim)
            note = (
                "all aggressive (floor)" if p == 0.00 else
                "all rich (today)" if p == 1.00 else
                "blended"
            )
            verdict.append(
                f"| {p:.2f} | {b:.1f} | {bpc:.3f} | {note} |"
            )
        verdict.extend([
            "",
            "Implementation sketch: extend `Rroq158Encoded` with a per-token "
            "`regime` byte and store two scale streams. The Rust kernel "
            "(`fused_rroq158.rs::score_pair_body`) can branch on `regime[t]` to "
            "select the fp16 scale stride. This is the same trick the KV-cache "
            "smoke test validated (see message thread, p=0.30 → 1.85 b/d with "
            "measurable quality lift over pure binary).",
            "",
            "**Recommended next step**: prototype a Python-only outlier-rescue "
            "harness that scores each query against both the rich and the cheap "
            "encoding and merges the results per-token, before touching the kernel. "
            "This isolates the quality question from the kernel-engineering question.",
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(header + body + verdict) + "\n")
    print(f"[probe] wrote report -> {out_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="rroq158 (K x group_size) Pareto compression probe"
    )
    parser.add_argument("--dataset", default="arguana",
                        help="BEIR dataset to probe (default: arguana)")
    parser.add_argument("--K-values", nargs="*", type=int, default=list(DEFAULT_K_VALUES),
                        help=f"K (centroid count) values (default: {DEFAULT_K_VALUES})")
    parser.add_argument("--gs-values", nargs="*", type=int, default=list(DEFAULT_GS_VALUES),
                        help=f"group_size values (default: {DEFAULT_GS_VALUES})")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Native Rust workers per cell (default: 8)")
    parser.add_argument("--n-eval", type=int, default=0,
                        help="Queries per cell (0 = full dataset, default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="rroq158 FWHT + spherical k-means seed (default: 42)")
    parser.add_argument("--ndcg-tolerance", type=float, default=0.005,
                        help="Max NDCG@10 drop vs. baseline to count as Pareto (default: 0.005)")
    parser.add_argument("--p95-tolerance", type=float, default=0.10,
                        help="Max p95 latency growth vs. baseline (default: 0.10 = +10%%)")
    parser.add_argument("--cells-dir", default="reports/rroq158_pareto_cells",
                        help="Directory for per-cell JSONL + log artifacts")
    parser.add_argument("--report", default=None,
                        help="Path for the final markdown report "
                             "(default: reports/rroq158_pareto_<dataset>.md)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse cell JSONL files if they already exist "
                             "(useful for resume after a crash)")
    args = parser.parse_args()

    cells_dir = Path(args.cells_dir)
    if not cells_dir.is_absolute():
        cells_dir = REPO_ROOT / cells_dir
    report_path = (
        Path(args.report)
        if args.report
        else REPO_ROOT / f"reports/rroq158_pareto_{args.dataset}.md"
    )

    cells: List[Tuple[int, int]] = [(K, gs) for K in args.K_values for gs in args.gs_values]
    print(f"[probe] {len(cells)} cells to run on dataset={args.dataset}", flush=True)
    for K, gs in cells:
        print(f"[probe]   - K={K}, gs={gs}, "
              f"predicted bytes/tok={predicted_bytes_per_tok(128, gs):.1f} (dim=128)",
              flush=True)

    results: List[CellResult] = []
    t_start = time.time()
    for K, gs in cells:
        cell_jsonl = cells_dir / args.dataset / f"K{K}_gs{gs}.jsonl"
        if args.skip_existing and cell_jsonl.exists() and cell_jsonl.stat().st_size > 0:
            print(f"[probe] reusing existing cell K={K} gs={gs} from {cell_jsonl}", flush=True)
            with open(cell_jsonl) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            r = rows[0]
            dim = int(r["dim"])
            results.append(CellResult(
                K=K, group_size=gs,
                bytes_per_tok=predicted_bytes_per_tok(dim, gs),
                bits_per_coord=8.0 * predicted_bytes_per_tok(dim, gs) / dim,
                qps=float(r["qps"]),
                p50_ms=float(r["p50_ms"]),
                p95_ms=float(r["p95_ms"]),
                ndcg10=float(r["NDCG@10"]),
                ndcg100=float(r["NDCG@100"]),
                recall10=float(r["recall@10"]),
                recall100=float(r["recall@100"]),
                n_queries=int(r["n_queries"]),
                n_docs=int(r["n_docs"]),
                dim=dim,
                raw_path=cell_jsonl,
                elapsed_s=0.0,
            ))
            continue
        try:
            cell = run_cell(
                args.dataset, K, gs,
                n_workers=args.n_workers,
                n_eval=args.n_eval,
                seed=args.seed,
                cells_dir=cells_dir,
            )
            results.append(cell)
            print(f"[probe]   K={K} gs={gs} -> NDCG@10={cell.ndcg10:.4f}, "
                  f"p95={cell.p95_ms:.0f}ms, QPS={cell.qps:.1f}, "
                  f"{cell.bytes_per_tok:.1f} B/tok ({cell.elapsed_s:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"[probe] CELL FAILED K={K} gs={gs}: {e}", flush=True, file=sys.stderr)

    elapsed_total = time.time() - t_start
    print(f"[probe] {len(results)}/{len(cells)} cells succeeded in {elapsed_total/60:.1f} min",
          flush=True)

    if not any(r.K == 8192 and r.group_size == 32 for r in results):
        print("[probe] WARNING: baseline cell (K=8192, gs=32) missing - cannot write report",
              flush=True, file=sys.stderr)
        return 2

    write_report(
        dataset=args.dataset,
        results=results,
        out_path=report_path,
        ndcg_tolerance=args.ndcg_tolerance,
        p95_tolerance=args.p95_tolerance,
        n_eval_arg=args.n_eval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
