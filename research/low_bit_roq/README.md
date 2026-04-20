# Low-Bit ROQ Research

Workspace for the "Riemannian Low-Bit ROQ" plan (see project plan in
`/root/.cursor/plans/`). The full progress log lives in
[`PROGRESS.md`](PROGRESS.md). The structured machine-readable artifacts each
PROGRESS entry cites live under [`reports/`](reports/).

## Layout

```
research/low_bit_roq/
  PROGRESS.md               # human-readable, growing, append-only
  progress_md.py            # helper: harness writes auto-stub entries here
  harness.py                # Phase 0 - extends BEIR runner: multi-seed,
                            # paired bootstrap, candidate-recall, cold/warm
                            # p95, compute counters
  distortion_bench.py       # angular error / NN-preservation harness
                            # (no GPU required; works on a saved token
                            # sample)
  anisotropic.py            # A3 ScaNN-style anisotropic loss; also fits
                            # the ternary {-1, 0, +1} 3-level codebook used
                            # by A2.5
  ternary.py                # A2.5 ternary (1.58-bit) Python encoder /
                            # decoder; produces (sign_plane, nonzero_plane,
                            # per-group scales)
  salience.py               # A4 token salience pruning at index time
  spherical_kmeans.py       # B1 spherical Lloyd k-means
  tangent_query.py          # B0 log_c(q) router-side tangent feature
  lemur_tangent.py          # B2 LEMUR proxy weights with tangent residuals
  rroq.py                   # B3 Riemannian-aware ROQ (centroid + tangent
                            # residual + low-bit code)
  mixed_precision.py        # B4 promote high-salience tokens to 4-bit
  per_cluster_pca.py        # B5 OPQ-style per-centroid PCA basis
  distill_rerank.py         # cross-cut 1: tiny MLP rerank head trained on
                            # (q, lowbit-candidate, lowbit-score) ->
                            # fp16-rerank-rank
  shortcut_edges.py         # cross-cut 2: bridge edges from rank
                            # disagreement
  filter_aware.py           # cross-cut 3: per-cluster filter sketches
  integration.py            # A5: LEMUR / ANN / router production-lane glue
                            # for new ROQ variants
  kernels/
    triton_roq_2bit_asym.py # A2 asymmetric 2-bit kernel
    triton_roq_ternary.py   # A2.5 ternary (1.58-bit) kernel
  reports/                  # JSON artifacts cited by PROGRESS.md entries
  configs/                  # pinned config snapshots for each promoted run
  tests/                    # parity / sanity tests for kernels and
                            # encoders
  run_a1.py                 # A1 mechanics sweep runner
  run_a6.py                 # A6 production-gate evaluator
  run_c1_5.py               # C1.5 bit-width bake-off runner
  MEMO.md                   # final ship/no-ship memo (filled at end)
```

## Quickstart for an engineer joining the project

1. Read [`PROGRESS.md`](PROGRESS.md) top-to-bottom (5 minutes).
2. Run baseline sanity (no GPU needed):

   ```bash
   python -m research.low_bit_roq.distortion_bench \
     --sample-path tests/fixtures/token_sample_1m.npy \
     --bits 2 --group-size 16
   ```

3. Pick the next `[VERDICT-PENDING]` entry from PROGRESS.md, run the
   referenced experiment, fill in the Verdict + Why + Gate-impact, commit.

## Wiring into the production lane

Every experiment must go through:

```
LemurRouter (lemur_router.py + _lemur/ann.py)
  -> CentroidRouter (centroid_router.py)
    -> CentroidScreening (centroid_screening.py)
      -> ROQ rerank (triton_roq.py / new kernels)
        -> optional exact MaxSim (fp16/fp32)
```

`integration.py` exposes the helpers that thread new compression variants
through this lane without forking the existing pipeline.
