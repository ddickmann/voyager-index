# Quantization tuning guide

This page is the single source of truth for choosing and tuning a quantization
codec in `voyager-index`. It covers (a) the codec decision matrix, (b) the
SOTA default for `RROQ158`, (c) the per-dim `group_size` recipe table,
(d) when to override the default, and (e) the closing retrospective on the
outlier-rescue investigation we explored and dropped.

## TL;DR

- **Default**: `Compression.RROQ158` at `K=8192`, `group_size=128` — ~6.4× smaller
  per-token storage than FP16, ~−1 pt NDCG@10 average on BEIR with flat R@100,
  ~10–30% faster CPU p95 than the previous `gs=32` default. **You don't need
  to set anything**; the default in [`BuildConfig`](../api/python.md) is
  this configuration.
- **Zero-quality-loss alternative**: `Compression.RROQ4_RIEM` — ~3× smaller
  than FP16, NDCG@10 essentially at FP16 (avg +0.02 pt), ~5–7× slower in
  absolute latency at the BEIR batch shape. Use when you cannot tolerate any
  quality regression.
- **Max latency, no compression**: `Compression.FP16` — the legacy lane.

## Decision matrix

| if you care most about… | use | quality vs FP16 | storage vs FP16 | latency vs FP16 |
|---|---|---|---|---|
| Storage + good quality (recommended) | `RROQ158` (default) | ~−1 pt NDCG@10 avg, flat R@100 | ~6.4× smaller | ~p95 parity GPU, modestly faster CPU on most cells |
| Strict zero-quality regression | `RROQ4_RIEM` | +0.02 pt NDCG@10 avg | ~3× smaller | ~5–7× slower at BEIR batch shape |
| Max throughput | `FP16` | baseline | baseline | baseline |
| Backwards compatibility (legacy index) | whatever the manifest records | n/a | n/a | n/a |

Existing on-disk indexes always load with the codec they were built with —
the manifest carries the build-time codec; only newly built indexes pick up
the new SOTA default.

## RROQ158 SOTA default — why `group_size=128`

At `dim=128` (the most-tested production dim), `group_size=128` means **one
scale per token** instead of four. The doc-token storage breakdown:

| layout component | gs=32 (previous default) | gs=128 (new SOTA default) |
|---|---:|---:|
| `centroid_id` (uint16) | 2 B | 2 B |
| `cos_norm + sin_norm` (2 × fp16) | 4 B | 4 B |
| `sign_plane + nonzero_plane` (2 × dim/8) | 32 B | 32 B |
| `scales` (dim/group_size × fp16) | 8 B | 2 B |
| **total per token (dim=128)** | **46 B** | **40 B** |

That's ~13% smaller storage on top of the ~5.5× the codec already gives over
FP16, and one scale per token cuts the popcount kernel's per-group scale
load count by 4×, which is where the p95 improvement comes from.

### Validated quality — full BEIR cells we have at gs=128

CPU 8-worker, full eval (no n_eval cap), K=8192. Source raw cells:
[`reports/rroq158_pareto_cells/`](../../reports/rroq158_pareto_cells/);
per-dataset markdown:
[`reports/rroq158_pareto_arguana.md`](../../reports/rroq158_pareto_arguana.md),
[`reports/rroq158_pareto_fiqa.md`](../../reports/rroq158_pareto_fiqa.md),
[`reports/rroq158_pareto_nfcorpus.md`](../../reports/rroq158_pareto_nfcorpus.md).

| dataset | NDCG@10 (gs=32) | NDCG@10 (gs=128) | ΔNDCG | R@100 (gs=32) | R@100 (gs=128) | p95 ratio | B/tok ratio | gate (−0.005)? |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| arguana | 0.3713 | 0.3655 | **−0.0058** | 0.9633 | 0.9633 | 0.80× | 0.87× | marginal fail |
| fiqa | 0.4223 | 0.4260 | **+0.0037** | 0.7151 | 0.7118 | 0.91× | 0.87× | pass |
| nfcorpus | 0.3799 | 0.3790 | −0.0009 | 0.3354 | 0.3329 | 0.67× | 0.87× | pass |

The remaining BEIR-6 datasets (`scifact`, `scidocs`, `quora`, `hotpotqa`)
will be filled in by the post-merge full sweep — see the changelog and
[`reports/rroq158_pareto_cells/`](../../reports/rroq158_pareto_cells/) for
the live state.

## Per-dim `group_size` recipe (dim-aware fallback)

You don't need to think about this — the encoder picks the right value
automatically and logs a warning if it has to fall back. The rule is:
the largest value in `{requested, 64, 32}` that divides `dim`.

| dim | resolved gs (default) | resolved gs (request gs=64) | notes |
|---:|---:|---:|---|
| 64 | 64 (warning) | 64 | dim<128, can't pack a single 128-coord group |
| 96 | 32 (warning) | 32 (warning) | 96 not div by 64 either |
| 128 | **128** (default) | 64 | the production sweet spot |
| 160 | 32 (warning) | 32 (warning) | 160 not div by 64 either |
| 256 | **128** (default) | 64 | 2 scales per token at gs=128 |
| 384 | **128** (default) | 64 | 3 scales per token at gs=128 |
| 512 | **128** (default) | 64 | 4 scales per token at gs=128 |
| 768 | **128** (default) | 64 | 6 scales per token at gs=128 |
| 1024 | **128** (default) | 64 | 8 scales per token at gs=128 |

If your `dim` is not a multiple of 32 (e.g. `dim=384` is fine,
`dim=200` is not), the encoder falls back to `Compression.FP16` and logs a
warning — rroq158's popcount kernel requires a 32-bit-word ternary plane.

## When to override the default

Pin to `Rroq158Config(group_size=64)` if any of:

- Your corpus resembles **arguana** — short documents with high
  intra-token magnitude variance (one scale per token loses the per-region
  magnitudes that matter for ranking). The −0.0058 NDCG@10 fail above is the
  signature; gs=64 brings arguana back to **+0.0013** (clean Pareto pass).
- You want to **cap the worst-case quality regression** across heterogeneous
  corpora to within −0.005 ΔNDCG@10. gs=64 has been Pareto-clean on every
  BEIR dataset we've tested.
- You serve a **mixed-dim deployment** and want one config that behaves the
  same on dim=128 and dim=64 (gs=64 divides both with no fallback).

```python
from voyager_index._internal.inference.shard_engine.serving_config import (
    BuildConfig, Compression,
)

cfg = BuildConfig(
    dim=128,
    compression=Compression.RROQ158,
    rroq158_group_size=64,   # override the SOTA default
)
```

For workloads that **cannot accept any quality loss**, switch codec entirely
to `Compression.RROQ4_RIEM` — see the [shard engine guide](shard-engine.md).

## Closing retrospective — why we did not ship the outlier-rescue hybrid

We explored a two-regime "rich + cheap" hybrid (a ternary `sign-only`
cheap path at ~25 B/tok with a small fraction `p` of high-residual tokens
"rescued" at the rich `sign+nz` path) — same trick that pays off for KV-cache
quantization. The hope was deeper compression at quality parity by spending
bits only where they matter.

Full Python prototype on `arguana` (1401 queries, FP32 reconstruction
MaxSim, no kernel changes — see `reports/rroq158_hybrid_prototype_log.txt`):

| p (rescue) | NDCG@10 | Δ vs FP32 ceiling (0.3688) | R@100 | B/tok |
|---:|---:|---:|---:|---:|
| 0.00 (all cheap) | 0.3079 | −0.061 | 0.916 | 25.0 |
| 0.05 | 0.3127 | −0.056 | 0.919 | 26.1 |
| 0.10 | 0.3167 | −0.052 | 0.923 | 27.2 |
| 0.20 | 0.3217 | −0.047 | 0.927 | 29.4 |
| 0.30 | 0.3206 | −0.048 | 0.929 | 31.6 |
| 0.50 | 0.3259 | −0.043 | 0.929 | 36.0 |
| 1.00 (all rich) | 0.3402 | −0.029 | 0.945 | 47.0 |

The slope of the rescue curve flattens fast: each +5% rescue buys
+0.005 NDCG at p<0.10, dropping to +0.0025 at p=0.20 and going **negative**
between p=0.20 and p=0.30 (within noise). Even at p=1.00 (= pure rich) the
hybrid leaves −0.029 on the table vs the FP32 ceiling — the rescue mechanism
cannot recover what the ternary regime structurally loses.

Meanwhile, the **uniform `gs=128` win we already have** delivers the same
~13% storage reduction at the same or better quality without any kernel
changes, additional regime-tag overhead, or maintenance complexity. The
math from our [arguana Pareto report](../../reports/rroq158_pareto_arguana.md)
shows the uniform sweep can buy at most ~13% on dim=128 because the dense
ternary `sign + nz` pair (32 B/tok = 70% of the layout) is fixed regardless
of `(K, group_size)` — going below the 40 B/tok floor really would require
a different codec, but the ROI from the hybrid investigation didn't justify
the on-disk format + Rust kernel + Triton kernel work needed to ship it.

We close the investigation here. The full prototype log is preserved at
[`reports/rroq158_hybrid_prototype_log.txt`](../../reports/rroq158_hybrid_prototype_log.txt)
for anyone who wants to revisit the strategy on a different codec or a
different signal (e.g. cluster-level tagging instead of per-token rescue).
