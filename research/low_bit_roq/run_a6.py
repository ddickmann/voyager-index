"""
A6 — production gate evaluator.

Reads the JSON reports of the A-best candidates per bit-width
(``a1-best-{1,1.58,2}.json``), runs the gate, and emits a single
``a6-gate.json`` report + a Promoted/Killed PROGRESS.md update.

Gate (bit-width-agnostic, per the plan; reranker MUST be on):

  1. Recall@10 vs fp16 (reranker on) ≥ -0.5 points
  2. NDCG@10 vs fp16   (reranker on) ≥ -0.7 points
  3. End-to-end p95 vs roq4          ≤ 0%   (parity or better)
  4. Disk vs roq4                    ≥ 40% smaller
  5. VRAM vs roq4                    ≥ 30% smaller

Plus operational: cold-p95 ≤ 1.3× warm-p95, throughput regression vs roq4
at SLA p95 ≤ 5%, no kernel-launch failures across a 1M-query stress.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import progress_md

log = logging.getLogger(__name__)


@dataclass
class GateResult:
    bit_width: str
    candidate_id: str
    recall_delta: float
    ndcg_delta: float
    p95_delta_pct: float
    disk_delta_pct: float
    vram_delta_pct: float
    cold_to_warm_p95_ratio: float
    qps_regression_pct: float
    n_kernel_failures: int
    passes_per_dataset: list[bool] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return (
            self.recall_delta >= -0.005  # 0.5 points = 0.005 in [0, 1]
            and self.ndcg_delta >= -0.007
            and self.p95_delta_pct <= 0.0
            and self.disk_delta_pct <= -0.40
            and self.vram_delta_pct <= -0.30
            and self.cold_to_warm_p95_ratio <= 1.3
            and self.qps_regression_pct <= 0.05
            and self.n_kernel_failures == 0
            and sum(self.passes_per_dataset) >= 4
        )


def evaluate_gate(
    candidate_report: dict, fp16_report: dict, roq4_report: dict
) -> GateResult:
    a_macro = candidate_report["macro"]
    fp16_macro = fp16_report["macro"]
    roq4_macro = roq4_report["macro"]
    cold = a_macro.get("cold_p95_ms", a_macro["warm_p95_ms"])
    warm = a_macro["warm_p95_ms"]
    p95_delta = (warm - roq4_macro["warm_p95_ms"]) / roq4_macro["warm_p95_ms"]

    disk_delta = (
        (a_macro.get("disk_bytes", 1) - roq4_macro.get("disk_bytes", 1))
        / max(1, roq4_macro.get("disk_bytes", 1))
    )
    vram_delta = (
        (a_macro.get("vram_mb", 1) - roq4_macro.get("vram_mb", 1))
        / max(1, roq4_macro.get("vram_mb", 1))
    )

    per_dataset = candidate_report.get("per_dataset", {})
    fp16_pd = fp16_report.get("per_dataset", {})
    passes_pd = []
    for ds in per_dataset:
        if ds not in fp16_pd:
            passes_pd.append(False)
            continue
        delta = per_dataset[ds]["recall_at_10_mean"] - fp16_pd[ds]["recall_at_10_mean"]
        passes_pd.append(delta >= -0.005)

    return GateResult(
        bit_width=candidate_report.get("bits", "?"),
        candidate_id=candidate_report["id"],
        recall_delta=a_macro["recall_at_10"] - fp16_macro["recall_at_10"],
        ndcg_delta=a_macro["ndcg_at_10"] - fp16_macro["ndcg_at_10"],
        p95_delta_pct=p95_delta,
        disk_delta_pct=disk_delta,
        vram_delta_pct=vram_delta,
        cold_to_warm_p95_ratio=cold / max(1e-6, warm),
        qps_regression_pct=(roq4_macro.get("qps", 1) - a_macro.get("qps", 1))
        / max(1, roq4_macro.get("qps", 1)),
        n_kernel_failures=int(a_macro.get("kernel_failures", 0)),
        passes_per_dataset=passes_pd,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("research/low_bit_roq/reports"),
    )
    parser.add_argument("--fp16-report", type=Path, required=True)
    parser.add_argument("--roq4-report", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="paths to A-best candidate JSON reports (one per bit-width)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fp16 = json.loads(args.fp16_report.read_text(encoding="utf-8"))
    roq4 = json.loads(args.roq4_report.read_text(encoding="utf-8"))

    results: list[GateResult] = []
    promoted: list[str] = []
    killed: list[str] = []
    for path in args.candidates:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        result = evaluate_gate(report, fp16, roq4)
        results.append(result)
        log.info(
            "%s -> %s  recall_delta=%.3f p95_delta=%+.1f%% disk_delta=%+.1f%% datasets_pass=%d/%d",
            result.candidate_id,
            "PASS" if result.passes else "FAIL",
            result.recall_delta,
            100 * result.p95_delta_pct,
            100 * result.disk_delta_pct,
            sum(result.passes_per_dataset),
            len(result.passes_per_dataset),
        )
        if result.passes:
            promoted.append(f"`{result.candidate_id}` (bits={result.bit_width}) — passes A6 gate")
        else:
            killed.append(
                f"`{result.candidate_id}` (bits={result.bit_width}) — fails A6 gate "
                f"(recall_delta={result.recall_delta:+.3f}, p95_delta={result.p95_delta_pct:+.0%})"
            )

    out_path = args.reports_dir / "a6-gate.json"
    out_path.write_text(
        json.dumps(
            {
                "results": [r.__dict__ for r in results],
                "promoted": promoted,
                "killed": killed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for bullet in promoted:
        progress_md.update_promoted(bullet)
    for bullet in killed:
        progress_md.update_killed(bullet)

    if any(r.passes for r in results):
        progress_md.update_current_state(
            phase="B (Riemannian additions)",
            most_recent_gate="A6 — PASS (at least one bit-width)",
            a_best=[r.candidate_id for r in results if r.passes],
        )
        log.info("A6 gate PASSED for at least one bit-width — proceed to Phase B")
        return 0
    progress_md.update_current_state(
        phase="A (mechanics — iterating)",
        most_recent_gate="A6 — FAIL (no bit-width passed)",
    )
    log.warning("A6 gate FAILED for all bit-widths — iterate or ship best-failing")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
