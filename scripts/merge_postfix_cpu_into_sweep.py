"""Merge post-fix CPU rows for rroq158 and rroq4_riem into the BEIR 2026-Q2 sweep.

Reads:
  - reports/beir_2026q2/sweep.jsonl (48 cells from the original sweep)
  - reports/beir_postfix_rroq158_cpu.jsonl   (6 fresh CPU rows)
  - reports/beir_postfix_rroq4_riem_cpu.jsonl (6 fresh CPU rows)

Writes:
  - reports/beir_2026q2/sweep.jsonl (in place; 48 cells with 12 CPU rows replaced)

The original file is preserved as ``sweep.jsonl.prefix.bak`` (call site responsibility).

Schema mapping from BEIR-harness JSONL -> sweep.jsonl row:
  mode='CPU-8w' -> mode='cpu', raw_mode_label='CPU-8w'
  NDCG@10/100   -> ndcg_at_10 / ndcg_at_100
  MAP@100       -> map_at_100
  recall@10/100 -> recall_at_10 / recall_at_100
  n_queries     -> n_queries_evaluated

Provenance and codec/k metadata are preserved from the matching pre-fix CPU row
so the rendered table keeps the original sweep_id, host, and wheel versions
visible alongside the new latency numbers (only qps/p50/p95 change for CPU
cells; quality metrics are unchanged because the kernel is deterministic).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPORTS = Path("reports")
SWEEP = REPORTS / "beir_2026q2" / "sweep.jsonl"
RROQ158_CPU = REPORTS / "beir_postfix_rroq158_cpu.jsonl"
RROQ4_RIEM_CPU = REPORTS / "beir_postfix_rroq4_riem_cpu.jsonl"

CODECS_TO_REPLACE = {"rroq158", "rroq4_riem"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _harness_row_to_sweep_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Translate harness JSONL keys -> sweep.jsonl key names for CPU cells."""
    return {
        "qps": row["qps"],
        "p50_ms": row["p50_ms"],
        "p95_ms": row["p95_ms"],
        "ndcg_at_10": row["NDCG@10"],
        "ndcg_at_100": row["NDCG@100"],
        "map_at_100": row["MAP@100"],
        "recall_at_10": row["recall@10"],
        "recall_at_100": row["recall@100"],
        "n_queries_evaluated": row["n_queries"],
        "n_docs": row["n_docs"],
        "top_k": row["top_k"],
        "indexing_docs_per_sec": row.get("indexing_docs_per_sec", float("inf")),
    }


def _index_by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["dataset"]: r for r in rows}


def main() -> None:
    sweep = _load_jsonl(SWEEP)
    rroq158_new = _index_by_dataset(_load_jsonl(RROQ158_CPU))
    rroq4_new = _index_by_dataset(_load_jsonl(RROQ4_RIEM_CPU))
    new_by_codec = {
        "rroq158": rroq158_new,
        "rroq4_riem": rroq4_new,
    }

    completed_at = datetime.now(timezone.utc).isoformat()
    n_replaced = 0
    out_rows: List[Dict[str, Any]] = []

    for cell in sweep:
        if cell.get("mode") != "cpu" or cell.get("codec") not in CODECS_TO_REPLACE:
            out_rows.append(cell)
            continue

        codec = cell["codec"]
        ds = cell["dataset"]
        new = new_by_codec[codec].get(ds)
        if new is None:
            print(f"WARNING: no replacement found for {ds}/{codec}/cpu, keeping original")
            out_rows.append(cell)
            continue

        merged = dict(cell)
        merged.update(_harness_row_to_sweep_fields(new))
        merged["completed_at"] = completed_at
        merged["postfix_refresh"] = True
        merged["postfix_source"] = (
            "reports/beir_postfix_rroq158_cpu.jsonl"
            if codec == "rroq158"
            else "reports/beir_postfix_rroq4_riem_cpu.jsonl"
        )
        out_rows.append(merged)
        n_replaced += 1

    with open(SWEEP, "w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, default=str) + "\n")

    print(f"Replaced {n_replaced} CPU rows; wrote {len(out_rows)} cells to {SWEEP}.")


if __name__ == "__main__":
    main()
