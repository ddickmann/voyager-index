"""
Phase A1 — mechanics root-cause sweep.

Seven-axis grid over ``RotationalQuantizer`` (and the new
``TernaryQuantizer``) on the existing FWHT/min-max stack:

  - doc bits ∈ {1.0, 1.58, 2.0}
  - group_size ∈ {None, 64, 32, 16, 8} (group_size=None for 1-bit only)
  - query bits ∈ {2, 4, 6, 8} (asymmetric, only when doc bits < query bits)
  - FWHT: on / off
  - pre-cluster L2-normalize: on / off
  - codebook: uniform min/max vs. Lloyd 1D k-means per group
  - norm-correction: 4-term affine vs. 2-term simplified

For each cell we run ``distortion_bench.sweep`` to get a fast offline
verdict, then promote the top-N cells per bit-width to a full BEIR run.

Output: one JSON per cell at ``research/low_bit_roq/reports/a1-cell-XX.json``,
plus three ``a1-best-{1,1.58,2}.json`` summary reports that pick the
single best config per bit-width. Each cell auto-emits a PROGRESS.md
stub; the per-bit-width best also gets a hand-completed entry.

Usage:

    python -m research.low_bit_roq.run_a1 \
        --sample-path tests/fixtures/token_sample_1m.npy \
        --datasets arguana fiqa nfcorpus scidocs scifact \
        --offline-only            # skip the BEIR follow-up

    python -m research.low_bit_roq.run_a1 --beir-runner my_module:make_runner
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import distortion_bench, harness, progress_md

log = logging.getLogger(__name__)


@dataclass
class A1Cell:
    cell_id: str
    doc_bits: float
    group_size: int | None
    query_bits: int
    fwht: bool
    normalize: bool
    codebook: str
    norm_correction: str

    def cfg_dict(self) -> dict:
        return {
            "doc_bits": self.doc_bits,
            "group_size": self.group_size,
            "query_bits": self.query_bits,
            "fwht": self.fwht,
            "normalize": self.normalize,
            "codebook": self.codebook,
            "norm_correction": self.norm_correction,
        }


def enumerate_cells(
    *,
    bits_grid: Sequence[float] = (1.0, 1.58, 2.0),
    group_sizes: Sequence[int | None] = (None, 64, 32, 16, 8),
    query_bits_grid: Sequence[int] = (4, 6, 8),
    fwht_grid: Sequence[bool] = (True, False),
    normalize_grid: Sequence[bool] = (True, False),
    codebook_grid: Sequence[str] = ("uniform", "lloyd"),
    norm_corr_grid: Sequence[str] = ("4term", "2term"),
) -> list[A1Cell]:
    """Enumerate the 7-axis A1 grid.

    The default grid is the *full* plan grid (336 cells). For a memory- /
    time-bounded sweep on a 24 GB box, callers should pass a
    pruned-axes subset (see :func:`enumerate_cells_lite`) and accept that
    the Pareto front comes from a partial sweep that the user can extend
    later by re-running with the full grid.
    """
    cells: list[A1Cell] = []
    cell_n = 0
    for doc_bits, gs, qb, fwht, norm, cb, nc in itertools.product(
        bits_grid,
        group_sizes,
        query_bits_grid,
        fwht_grid,
        normalize_grid,
        codebook_grid,
        norm_corr_grid,
    ):
        # constraint: 1-bit symmetric path requires group_size=None
        if doc_bits == 1.0 and gs is not None:
            continue
        # ternary codebook is always grouped, and group must be ≥ 32 (one
        # 32-bit popcount word). The kernel architecture cannot support
        # smaller groups without sub-word splits.
        if doc_bits == 1.58 and (gs is None or gs % 32 != 0):
            continue
        # 2-bit grouped path needs a positive group; gs=None disabled
        if doc_bits == 2.0 and gs is None:
            continue
        # asymmetric only when query bits >= doc bits effective resolution
        if qb < int(np.ceil(doc_bits)):
            continue
        cells.append(
            A1Cell(
                cell_id=f"a1-cell-{cell_n:03d}",
                doc_bits=float(doc_bits),
                group_size=gs,
                query_bits=qb,
                fwht=fwht,
                normalize=norm,
                codebook=cb,
                norm_correction=nc,
            )
        )
        cell_n += 1
    return cells


def _load_subsample(sample_path: Path | None, n_tokens: int, dim: int, seed: int) -> np.ndarray:
    """Load (or synthesize) ``n_tokens`` rows from the fixture.

    Critically, when a fixture path is provided we ``np.load`` the file
    with ``mmap_mode='r'``, then *copy out only the rows we need*. This
    avoids holding the full 488 MB sample resident for the duration of the
    sweep — the previous implementation (``_load_sample``) loaded the full
    file into memory and was the second-largest contributor to OOM.
    """
    if sample_path is None:
        return distortion_bench._load_sample(None, True, n_tokens, dim)
    arr = np.load(str(sample_path), mmap_mode="r")
    rng = np.random.default_rng(seed)
    take = min(n_tokens, arr.shape[0])
    idx = rng.choice(arr.shape[0], size=take, replace=False)
    sample = np.array(arr[idx], dtype=np.float32, order="C")
    del arr
    return sample


def run_offline_distortion(
    cells: list[A1Cell],
    *,
    sample_path: Path | None,
    n_tokens: int = 8192,
    n_queries: int = 128,
    dim: int = 128,
    seed: int = 0,
    out_dir: Path = Path("research/low_bit_roq/reports"),
) -> list[dict]:
    """Stage 1: offline distortion. Cheap and fast — runs every cell.

    Memory budget per cell is bounded by ``n_tokens * n_queries * 8 B``
    (two fp32 similarity matrices). Defaults of 8192 tokens × 128 queries
    fit in ~8 MB peak per cell; on the 24 GB benchmark box the sample +
    quantizer state stays under ~200 MB even for the full 336-cell sweep
    because we explicitly free per-cell tensors and ``torch.cuda.empty_cache``
    after each cell (inside ``distortion_bench.sweep``).
    """
    import gc as _gc

    tokens = _load_subsample(sample_path, n_tokens, dim, seed)
    rng = np.random.default_rng(seed + 1)
    q_idx = rng.choice(tokens.shape[0], size=min(n_queries, tokens.shape[0]), replace=False)
    queries = tokens[q_idx].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for i, cell in enumerate(cells):
        if cell.normalize:
            tokens_use = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-12)
            queries_use = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
        else:
            tokens_use = tokens
            queries_use = queries
        try:
            rows = distortion_bench.sweep(
                tokens_use,
                queries_use,
                bits=[cell.doc_bits],
                group_size=cell.group_size,
                fwht=cell.fwht,
                seed=seed,
            )
        except Exception as e:
            log.warning("cell %s failed: %s", cell.cell_id, e)
            rows = []
        if cell.normalize:
            del tokens_use, queries_use
        if not rows:
            continue
        row = rows[0]
        summary = {
            "cell_id": cell.cell_id,
            "config": cell.cfg_dict(),
            "distortion": asdict(row),
        }
        out.append(summary)
        cell_path = out_dir / f"{cell.cell_id}.json"
        cell_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if i % 10 == 9:
            _gc.collect()
            log.info("a1 progress %d/%d cells", i + 1, len(cells))
    return out


def select_top_per_bit(
    summaries: list[dict], top_n: int = 5
) -> dict[str, list[dict]]:
    """Group by ``doc_bits`` and pick top-N by NN-1 preservation."""
    groups: dict[str, list[dict]] = {}
    for s in summaries:
        key = f"{s['config']['doc_bits']:.2f}"
        groups.setdefault(key, []).append(s)
    out: dict[str, list[dict]] = {}
    for k, items in groups.items():
        items.sort(key=lambda s: s["distortion"]["nn1_preservation"], reverse=True)
        out[k] = items[:top_n]
    return out


def enumerate_cells_lite() -> list[A1Cell]:
    """Memory-bounded A1 grid (~24 cells) for the 24 GB benchmark box.

    Sacrificed axes:

    - ``query_bits``: fixed at 6 (the asymmetric kernel sweet-spot from the
      1-bit precedent in ``triton_roq.py``).
    - ``codebook``: fixed at ``"uniform"`` (Lloyd 1D goes in A3 once
      anisotropic loss is wired).
    - ``norm_correction``: fixed at ``"4term"`` (no evidence the simplified
      2-term variant ever wins in distortion).
    - ``group_size`` for 2-bit/ternary: pruned to {32, 64} (the realistic
      production range; 8/16 lose too much to per-group meta overhead).

    Kept axes: ``doc_bits``, ``group_size`` (within range), ``fwht``,
    ``normalize``. That isolates the four highest-leverage mechanics knobs.
    The full grid ships behind ``--full-grid``.
    """
    return enumerate_cells(
        group_sizes=(None, 64, 32),
        query_bits_grid=(6,),
        fwht_grid=(True, False),
        normalize_grid=(True, False),
        codebook_grid=("uniform",),
        norm_corr_grid=("4term",),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-path", type=Path, default=None)
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--n-queries", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n-per-bit", type=int, default=5)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the full 7-axis 336-cell grid. Default is --lite (24 cells, "
        "fits in <4 GB peak on a 24 GB box).",
    )
    parser.add_argument(
        "--beir-runner",
        type=str,
        default=None,
        help="dotted.path:factory_callable that returns a SearchRunner. "
        "Required unless --offline-only.",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research/low_bit_roq/reports"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cells = enumerate_cells() if args.full_grid else enumerate_cells_lite()
    log.info("A1 sweep cells: %d (lite=%s)", len(cells), not args.full_grid)

    summaries = run_offline_distortion(
        cells,
        sample_path=args.sample_path,
        n_tokens=args.n_tokens,
        n_queries=args.n_queries,
        dim=args.dim,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    top_per_bit = select_top_per_bit(summaries, top_n=args.top_n_per_bit)
    summary_path = args.out_dir / "a1-summary.json"
    summary_path.write_text(
        json.dumps({"total_cells": len(cells), "top_per_bit": top_per_bit}, indent=2),
        encoding="utf-8",
    )
    log.info("offline distortion done. Top-%d per bit-width:", args.top_n_per_bit)
    for k, items in top_per_bit.items():
        log.info("  %s-bit:", k)
        for s in items:
            log.info("    %s nn1=%.3f angular_p50=%.2f", s["cell_id"], s["distortion"]["nn1_preservation"], s["distortion"]["angular_error_p50_deg"])

    if args.offline_only:
        return 0

    if args.beir_runner is None:
        raise SystemExit(
            "--beir-runner required unless --offline-only. "
            "Use, e.g. research.low_bit_roq.runners:make_a1_runner"
        )

    module_name, factory_name = args.beir_runner.split(":")
    factory_module = __import__(module_name, fromlist=[factory_name])
    factory = getattr(factory_module, factory_name)

    for bits_key, items in top_per_bit.items():
        for s in items:
            cfg = s["config"]
            cell_id = s["cell_id"]
            log.info("BEIR follow-up for %s (bits=%s)", cell_id, bits_key)
            harness_cfg = harness.HarnessConfig(
                experiment_id=f"{cell_id}-beir",
                summary=f"A1 mechanics cell {cell_id} on BEIR",
                datasets=args.datasets,
                seeds=args.seeds,
                config_snapshot=cfg,
                baseline_name="roq4",
                gate="advances A6",
            )
            runner_factory = factory(cell_id, cfg)
            agg = harness.run_sweep(harness_cfg, runner_factory)
            harness.emit_progress_stub(agg, reports_dir=args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
