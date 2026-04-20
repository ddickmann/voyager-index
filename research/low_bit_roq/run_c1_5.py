"""
C1.5 — bit-width production bake-off matrix.

Sweep:

  - doc bits ∈ {1.0, 1.58, 2.0}             (the three A-best winners)
  - query bits ∈ {4, 6}                     (asymmetric kernel A2 / A2.5)
  - k_candidates ∈ {1000, 2000, 4000, 8000}
  - distillation reranker ∈ {off, on}

Decision rule (single objective):

  Pick the config with the **lowest end-to-end p95** that achieves
  Recall@10 ≥ fp16 - 0.5 points (reranker on) on ≥4 of 5 datasets.

Reported diagnostics for the merge memo:

  - (k_candidates, end-to-end p95) and (k_candidates, Recall@10) per bit-width
  - reranker uplift per bit-width (Recall@10 with rerank on minus off)
  - bytes-fetched/q at the chosen k per bit-width
  - VRAM and disk per bit-width
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import harness, progress_md

log = logging.getLogger(__name__)


BAKE_OFF_BITS = (1.0, 1.58, 2.0)
BAKE_OFF_QUERY_BITS = (4, 6)
BAKE_OFF_K_CANDIDATES = (1000, 2000, 4000, 8000)
BAKE_OFF_RERANKER = (False, True)


@dataclass
class BakeOffCell:
    cell_id: str
    bits: float
    query_bits: int
    k_candidates: int
    reranker: bool


def enumerate_cells() -> list[BakeOffCell]:
    cells: list[BakeOffCell] = []
    for i, (b, qb, kc, r) in enumerate(
        itertools.product(
            BAKE_OFF_BITS, BAKE_OFF_QUERY_BITS, BAKE_OFF_K_CANDIDATES, BAKE_OFF_RERANKER
        )
    ):
        cells.append(
            BakeOffCell(
                cell_id=f"c15-cell-{i:03d}",
                bits=float(b),
                query_bits=int(qb),
                k_candidates=int(kc),
                reranker=bool(r),
            )
        )
    return cells


def pick_winner(reports: list[dict], fp16: dict) -> dict | None:
    """Apply the C1.5 decision rule against a list of cell reports."""
    candidates: list[tuple[float, dict]] = []
    fp16_pd = fp16.get("per_dataset", {})
    for r in reports:
        per_ds = r.get("per_dataset", {})
        passes = 0
        for ds, info in per_ds.items():
            fp16_r10 = fp16_pd.get(ds, {}).get("recall_at_10_mean", 0.0)
            if info["recall_at_10_mean"] >= fp16_r10 - 0.005:
                passes += 1
        if passes < 4:
            continue
        warm_p95 = r["macro"]["warm_p95_ms"]
        candidates.append((warm_p95, r))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("research/low_bit_roq/reports"),
    )
    parser.add_argument("--fp16-report", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+",
        default=["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--bake-off-runner",
        type=str,
        required=True,
        help="dotted.path:factory_callable that returns a SearchRunner factory "
        "given a BakeOffCell.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fp16 = json.loads(args.fp16_report.read_text(encoding="utf-8"))
    cells = enumerate_cells()
    log.info("C1.5 bake-off: %d cells", len(cells))

    module_name, factory_name = args.bake_off_runner.split(":")
    module = __import__(module_name, fromlist=[factory_name])
    factory: Callable = getattr(module, factory_name)

    reports: list[dict] = []
    for cell in cells:
        cfg = harness.HarnessConfig(
            experiment_id=cell.cell_id,
            summary=(
                f"C1.5 bits={cell.bits} q_bits={cell.query_bits} "
                f"k={cell.k_candidates} rerank={cell.reranker}"
            ),
            datasets=args.datasets,
            seeds=args.seeds,
            config_snapshot=cell.__dict__,
            baseline_name="fp16",
            gate="decides C1.5",
        )
        runner_factory = factory(cell)
        agg = harness.run_sweep(cfg, runner_factory)
        harness.emit_progress_stub(agg, reports_dir=args.reports_dir)
        reports.append(agg)

    winner = pick_winner(reports, fp16)
    out_path = args.reports_dir / "c15-winner.json"
    out_path.write_text(
        json.dumps({"winner": winner, "n_candidates": len(reports)}, indent=2),
        encoding="utf-8",
    )
    if winner:
        progress_md.update_promoted(
            f"`{winner['id']}` — C1.5 production winner: bits={winner['config']['bits']} "
            f"k={winner['config']['k_candidates']} rerank={winner['config']['reranker']}"
        )
        progress_md.update_current_state(
            phase="C (combined + production)",
            most_recent_gate=f"C1.5 — winner {winner['id']}",
            production_candidate=winner["id"],
        )
        log.info("C1.5 winner: %s", winner["id"])
        return 0
    log.error("C1.5: no cell met the recall floor")
    progress_md.update_current_state(
        phase="C (combined + production)",
        most_recent_gate="C1.5 — no cell met recall floor",
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
