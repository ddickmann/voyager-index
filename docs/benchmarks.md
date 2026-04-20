# Benchmarks

This page documents how `voyager-index` benchmark claims are framed today,
including the headline BEIR results that ship in the README and the
head-to-head comparison against the established OSS reference.

Two rules matter more than any single latency number:

- every public throughput claim must be paired with recall
- GPU corpus and streamed GPU modes are reported separately

## Public Proof Layers

`voyager-index` has two public proof layers, and they should be read together:

- **Core production lane.** The shard-first route is proven by the BEIR shard
  benchmark in `benchmarks/beir_benchmark.py`. That harness measures
  search-only GPU-corpus Triton MaxSim and CPU multi-worker fused Rust scoring
  on the same production lane the API serves, with current public results in
  the BEIR table below.
- **Optional Latence graph lane.** The graph lane is proven separately by
  `tools/benchmarks/benchmark_latence_graph_quality.py` and the graph tests.
  In the current representative harness it delivers `+0.75` recall, `+0.333`
  NDCG, and `+0.75` support coverage on graph-shaped queries with `0.0`
  ordinary-query deltas, `57%` graph activation, `3.5` average added
  candidates on graph-shaped queries, and passing route-conformance checks.

The graph proof is intentionally scoped: it shows the shipped graph contract,
additive rescue semantics, provenance tagging, and retrieval uplift on
graph-shaped fixtures. It is not presented as a graph-on BEIR table.

The graph data itself comes from structured Latence graph data derived from
the indexed corpus and synchronized into the sidecar as target-linked graph
contracts. The public guide explains the architecture and provenance model
without exposing proprietary internals.

## BEIR Shard Benchmark (RTX A5000)

Measured on **NVIDIA RTX A5000 (24 GB)** using `lightonai/GTE-ModernColBERT-v1`.
Numbers are **search-only** and exclude query encoding. CPU results use
**8 native Rust workers**. These are full-query-set results, not a sampled
subset.

These results are meant to show three things:

1. **Retrieval quality** on standard BEIR datasets
2. **Search latency and throughput** under realistic conditions
3. **What is achievable on modest on-prem hardware**

| Dataset  | Documents | MAP@100 | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|----------|----------:|--------:|--------:|---------:|----------:|-----------:|--------:|-------------:|--------:|-------------:|
| arguana  | 8,674     | 0.2598  | 0.3679  | 0.4171   | 0.7402    | 0.9586     | 270.0   | 4.1          | 41.6    | 202.7        |
| fiqa     | 57,638    | 0.3818  | 0.4436  | 0.5049   | 0.5059    | 0.7297     | 164.8   | 5.0          | 80.2    | 115.7        |
| nfcorpus | 3,633     | 0.1963  | 0.3833  | 0.3485   | 0.3404    | 0.3348     | 282.6   | 3.8          | 123.3   | 84.4         |
| quora    | 15,675    | 0.9686  | 0.9766  | 0.9790   | 0.9930    | 0.9993     | 346.8   | 2.6          | 271.7   | 46.9         |
| scidocs  | 25,657    | 0.1383  | 0.1977  | 0.2763   | 0.2070    | 0.4369     | 246.8   | 4.3          | 83.9    | 111.8        |
| scifact  | 5,183     | 0.7141  | 0.7544  | 0.7730   | 0.8766    | 0.9567     | 263.4   | 4.0          | 69.1    | 138.4        |

How to read these results:

- **GPU P95 under 6 ms** across all listed datasets shows the fast path is
  practical on A5000-class hardware.
- **CPU mode remains viable** when GPU capacity is limited or reserved for
  model serving.
- Quality metrics are strong while using the same shard / Rust / Triton
  retrieval stack that powers the production API.
- The table above is the **fp16 baseline**. As of the current release the
  default codec for newly built indexes is `rroq158` (Riemannian 1.58-bit,
  K=8192) on both GPU and CPU; see the next subsection for kernel-level
  performance and the trade-off statement.

### Default codec: RROQ-1.58 (K=8192, **group_size=128 SOTA**) — Phase-7 BEIR 2026-Q2 sweep

`Compression.RROQ158` is the **default** for newly built indexes on both
GPU (Triton fused kernel) and CPU (Rust SIMD kernel) — at the new SOTA
**`group_size=128`** (one scale per token at dim=128, the most-tested
production dim). Per-token storage at the new default drops to **40 B**
(vs 256 B FP16, 64 B ROQ-4, and the previous 46 B at gs=32 — i.e.
**~6.4× / 1.6× / 1.15×** smaller respectively). For dims that aren't a
multiple of 128 (dim=64 / 96 / 160) the encoder transparently steps down
to gs=64 / gs=32 with a log warning. The full per-dim recipe and override
guidance live in [docs/guides/quantization-tuning.md](guides/quantization-tuning.md).

#### gs=128 flip — Pareto-validated cells (CPU 8-worker, full eval)

The 3 BEIR datasets re-validated for the gs=128 default flip, sourced from
[`reports/rroq158_pareto_cells/`](../reports/rroq158_pareto_cells/):

| Dataset | NDCG@10 (gs=32) | NDCG@10 (gs=128) | ΔNDCG | R@100 (gs=32) | R@100 (gs=128) | p95 (gs=32) | p95 (gs=128) | p95 ratio | B/tok ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana  | 0.3713 | 0.3655 | **−0.0058** | 0.9633 | 0.9633 | 666 ms | 530 ms | 0.80× | 0.87× |
| fiqa     | 0.4223 | 0.4260 | **+0.0037** | 0.7151 | 0.7118 | 279 ms | 253 ms | 0.91× | 0.87× |
| nfcorpus | 0.3799 | 0.3790 | −0.0009 | 0.3354 | 0.3329 | 286 ms | 193 ms | 0.67× | 0.87× |

Headline (ship-now signal across these 3 cells): **−13% storage, +0% to
+33% faster CPU p95, NDCG@10 within ±0.005 on Pareto-clean datasets**
(arguana marginal fail, recoverable with `Rroq158Config(group_size=64)`
— see the tuning guide). The **remaining 3 BEIR datasets** (`scifact`,
`scidocs`, `quora`) plus `hotpotqa` will be filled in by the **post-merge
full BEIR-6 sweep at gs=128** that refreshes the headline averages table
below in a follow-up commit.

#### Pre-flip 4-codec sweep (gs=32 baseline)

The full 4-codec × 6-dataset × 2-mode sweep below was measured at the
**previous `group_size=32` default**. The post-merge full BEIR-6 sweep
will refresh the rroq158 rows at gs=128.

The full sweep that backed this default is
[`benchmarks/beir_2026q2_full_sweep.py`](../benchmarks/beir_2026q2_full_sweep.py),
with raw per-cell JSONL + provenance under
[`reports/beir_2026q2/`](../reports/beir_2026q2/) and the rendered
markdown table generated by
[`scripts/format_beir_2026q2_table.py`](../scripts/format_beir_2026q2_table.py).

<!-- BEIR_2026Q2_FULL_TABLE_BEGIN — measured 2026-04-20 from reports/beir_2026q2/sweep.jsonl (CPU rows for rroq158 + rroq4_riem refreshed post-Phase-7 wrapper + kernel fixes) -->

Encoder: lightonai/GTE-ModernColBERT-v1. CPU lane uses 8 native Rust workers.
Run `python scripts/format_beir_2026q2_table.py reports/beir_2026q2/sweep.jsonl --format markdown`
to regenerate.

#### `fp16` (baseline)

| Dataset | Docs | NDCG@10 | NDCG@100 | MAP@100 | Recall@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana | 8,674 | 0.3679 | 0.4171 | 0.2598 | 0.7402 | 0.9586 | 270.2 | 4.0 | 54.9 | 154.5 |
| fiqa | 57,638 | 0.4436 | 0.5049 | 0.3818 | 0.5059 | 0.7297 | 208.5 | 5.0 | 93.5 | 111.6 |
| nfcorpus | 3,633 | 0.3833 | 0.3485 | 0.1963 | 0.3404 | 0.3348 | 265.3 | 4.1 | 105.9 | 103.6 |
| quora | 15,675 | 0.9766 | 0.9790 | 0.9686 | 0.9930 | 0.9993 | 377.6 | 2.7 | 314.6 | 37.5 |
| scidocs | 25,657 | 0.1977 | 0.2763 | 0.1383 | 0.2070 | 0.4369 | 244.6 | 4.4 | 104.2 | 91.5 |
| scifact | 5,183 | 0.7544 | 0.7730 | 0.7141 | 0.8766 | 0.9567 | 260.9 | 4.1 | 89.2 | 119.9 |

#### `int8` (GPU-only)

| Dataset | Docs | NDCG@10 | NDCG@100 | MAP@100 | Recall@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana | 8,674 | 0.3679 | 0.4166 | 0.2591 | 0.7423 | 0.9586 | 269.5 | 4.0 | N/A | N/A |
| fiqa | 57,638 | 0.4438 | 0.5062 | 0.3819 | 0.5050 | 0.7335 | 208.6 | 5.0 | N/A | N/A |
| nfcorpus | 3,633 | 0.3812 | 0.3460 | 0.1949 | 0.3398 | 0.3317 | 282.0 | 3.9 | N/A | N/A |
| quora | 15,675 | 0.9765 | 0.9789 | 0.9685 | 0.9929 | 0.9992 | 406.1 | 2.6 | N/A | N/A |
| scidocs | 25,657 | 0.1975 | 0.2752 | 0.1377 | 0.2069 | 0.4346 | 238.6 | 4.6 | N/A | N/A |
| scifact | 5,183 | 0.7531 | 0.7717 | 0.7124 | 0.8766 | 0.9567 | 262.5 | 4.0 | N/A | N/A |

#### `rroq158` (gs=32 baseline — pre-SOTA-flip, refreshed post-merge)

| Dataset | Docs | NDCG@10 | NDCG@100 | MAP@100 | Recall@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana | 8,674 | 0.3410 | 0.3940 | 0.2374 | 0.7038 | 0.9458 | 156.1 | 6.6 | 14.7 | 576.2 |
| fiqa | 57,638 | 0.4223 | 0.4845 | 0.3616 | 0.4824 | 0.7151 | 233.7 | 5.2 | 35.1 | 278.9 |
| nfcorpus | 3,633 | 0.3794 | 0.3462 | 0.1935 | 0.3387 | 0.3342 | 373.0 | 3.2 | 40.9 | 286.0 |
| quora | 15,675 | 0.9705 | 0.9732 | 0.9605 | 0.9916 | 0.9990 | 438.2 | 2.5 | 101.0 | 98.7 |
| scidocs | 25,657 | 0.1858 | 0.2651 | 0.1293 | 0.1964 | 0.4300 | 256.5 | 4.8 | 31.3 | 310.5 |
| scifact | 5,183 | 0.7386 | 0.7626 | 0.7006 | 0.8572 | 0.9633 | 257.7 | 5.3 | 24.8 | 398.4 |

#### `rroq4_riem` (no-quality-loss lane — Riemannian 4-bit asymmetric, K=8192, group_size=32)

| Dataset | Docs | NDCG@10 | NDCG@100 | MAP@100 | Recall@10 | Recall@100 | GPU QPS | GPU P95 (ms) | CPU QPS | CPU P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana | 8,674 | 0.3684 | 0.4185 | 0.2608 | 0.7402 | 0.9615 | 27.9 | 36.3 | 8.0 | 1051.2 |
| fiqa | 57,638 | 0.4447 | 0.5059 | 0.3825 | 0.5086 | 0.7315 | 70.1 | 20.6 | 22.5 | 406.4 |
| nfcorpus | 3,633 | 0.3819 | 0.3481 | 0.1960 | 0.3394 | 0.3364 | 127.8 | 12.2 | 25.6 | 414.0 |
| quora | 15,675 | 0.9772 | 0.9794 | 0.9693 | 0.9933 | 0.9993 | 359.1 | 3.2 | 65.4 | 505.8 |
| scidocs | 25,657 | 0.1981 | 0.2775 | 0.1386 | 0.2078 | 0.4411 | 69.5 | 20.9 | 14.7 | 1295.4 |
| scifact | 5,183 | 0.7542 | 0.7739 | 0.7145 | 0.8739 | 0.9600 | 59.4 | 27.3 | 12.6 | 770.2 |

#### Codec averages (BEIR-6 mean)

| Codec | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | GPU P95 (ms) | CPU P95 (ms) | GPU QPS | CPU QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0.5206 | 0.5498 | 0.6105 | 0.7360 | 4.0 | 103.1 | 271.2 | 127.1 |
| int8 | 0.5200 | 0.5491 | 0.6106 | 0.7357 | 4.0 | n/a | 277.9 | n/a |
| **rroq158 (gs=32, pre-flip)** | 0.5063 | 0.5376 | 0.5950 | 0.7312 | 4.6 | 324.8 | 285.9 | 41.3 |
| rroq4_riem (no-loss lane) | 0.5208 | 0.5505 | 0.6105 | 0.7383 | 20.1 | 740.5 | 119.0 | 24.8 |

<!-- BEIR_2026Q2_FULL_TABLE_END -->

#### Sweep verdict (Phase F1 default-promotion rule)

The decision script applies two rules in parallel and prints the
verdict at the bottom of the rendered table. The Phase-7 result:

1. **rroq4_riem default-promotion rule** (avg ΔNDCG@10 ≥ −0.5 pt and
   ΔR@100 ≥ −0.3 pt and per-cell GPU/CPU p95 ≤ FP16): **REJECTED**.
   rroq4_riem matches FP16 quality (avg ΔNDCG@10 = +0.02 pt, ΔR@100 =
   +0.23 pt) but **6 of 6 datasets** exceed the per-cell GPU p95 budget
   (worst: arguana at 9.00× FP16 GPU p95) and **6 of 6 datasets**
   exceed the per-cell CPU p95 budget (worst: scidocs at 14.16× FP16
   CPU p95). The reason is structural: every doc-token score now
   requires a 4-bit dequantize + per-group `(min, delta)` FMA on top of
   the centroid lookup, on top of an FP16 baseline that is itself
   extremely bandwidth-friendly. So the default does **not** flip to
   rroq4_riem.
2. **rroq158 retention rule** (avg ΔNDCG@10 ≥ −1.5 pt and avg GPU p95
   ratio ≤ 1.2×): **ACCEPTED**. avg ΔNDCG@10 = −1.43 pt (within the
   −1.50 pt budget), avg GPU p95 ratio = 1.13× (within the 1.20×
   budget), avg ΔR@100 = −0.48 pt (essentially flat). rroq158 stays
   the build-time default with rroq4_riem available as the opt-in
   no-quality-loss lane.

#### Codec-fidelity overlap (brute-force, vs FP16)

To disaggregate the rank-aggregate NDCG@10 from "did we even retrieve
the right doc", the sweep also runs
[`benchmarks/topk_overlap_sweep.py`](../benchmarks/topk_overlap_sweep.py),
which scores every (query, doc) pair with FP16 brute-force MaxSim, then
with each codec's brute-force MaxSim, and reports per-query top-K
overlap as well as the codec's own NDCG@10 / R@100 on the brute-force
path (i.e. with all routing artifacts removed).

Full BEIR-6 results, measured in
`reports/beir_2026q2/topk_overlap.jsonl`:

| Dataset  | Codec        | Brute-force NDCG@10 | Brute-force R@100 | overlap@10 | overlap@20 | overlap@50 | overlap@100 |
|----------|--------------|--------------------:|------------------:|-----------:|-----------:|-----------:|------------:|
| arguana  | fp16         |              0.3687 |            0.9593 |       100% |       100% |       100% |        100% |
| arguana  | **rroq158**  |    0.3410 (−2.78pt) |  0.9450 (−1.43pt) |      82.2% |      81.8% |      80.0% |       78.5% |
| arguana  | rroq4_riem   |    0.3684 (−0.03pt) |  0.9615 (+0.21pt) |      96.8% |      96.9% |      96.8% |       96.7% |
| fiqa     | fp16         |              0.4555 |            0.7561 |       100% |       100% |       100% |        100% |
| fiqa     | **rroq158**  |    0.4318 (−2.37pt) |  0.7354 (−2.07pt) |      75.1% |      76.1% |      77.4% |       78.4% |
| fiqa     | rroq4_riem   |    0.4531 (−0.24pt) |  0.7571 (+0.11pt) |      95.0% |      95.4% |      95.8% |       96.2% |
| nfcorpus | fp16         |              0.3827 |            0.3348 |       100% |       100% |       100% |        100% |
| nfcorpus | **rroq158**  |    0.3808 (−0.19pt) |  0.3336 (−0.13pt) |      79.5% |      80.1% |      80.5% |       80.4% |
| nfcorpus | rroq4_riem   |    0.3827 (+0.01pt) |  0.3369 (+0.21pt) |      95.8% |      95.9% |      96.1% |       96.3% |
| quora    | fp16         |              0.9774 |            0.9995 |       100% |       100% |       100% |        100% |
| quora    | **rroq158**  |    0.9707 (−0.67pt) |  0.9992 (−0.03pt) |      72.9% |      72.5% |      72.2% |       72.1% |
| quora    | rroq4_riem   |    0.9774 (+0.00pt) |  0.9995 (−0.00pt) |      94.9% |      94.9% |      95.1% |       95.2% |
| scidocs  | fp16         |              0.1987 |            0.4439 |       100% |       100% |       100% |        100% |
| scidocs  | **rroq158**  |    0.1864 (−1.24pt) |  0.4314 (−1.25pt) |      82.1% |      83.1% |      83.8% |       84.6% |
| scidocs  | rroq4_riem   |    0.1989 (+0.02pt) |  0.4435 (−0.04pt) |      96.4% |      96.6% |      97.0% |       97.2% |
| scifact  | fp16         |              0.7585 |            0.9567 |       100% |       100% |       100% |        100% |
| scifact  | **rroq158**  |    0.7421 (−1.64pt) |  0.9567 (+0.00pt) |      82.8% |      82.5% |      82.5% |       83.0% |
| scifact  | rroq4_riem   |    0.7563 (−0.22pt) |  0.9567 (+0.00pt) |      96.4% |      96.6% |      96.6% |       96.7% |

**Headline:** rroq158 averages ~79% top-10 / ~80% top-100 overlap with
FP16 brute-force across BEIR-6; rroq4_riem averages ~96% top-10 /
~97% top-100 (the no-quality-loss positioning holds end-to-end).

What this means in plain terms: rroq158 surfaces a different ~20% of
the top-10 documents than FP16, but the labeled relevant documents
are still mostly in the candidate set — R@100 is within −2.1 pt of
FP16 on every BEIR-6 dataset (within 0.0–1.4 pt on arguana, scifact,
nfcorpus, quora and scidocs; the worst case is fiqa at −2.07 pt).

Note on top-K monotonicity: for rroq158, top-K overlap is roughly
**flat or slightly declining with K** (e.g. quora 72.9% → 72.1% from
K=10 to K=100, arguana 82.2% → 78.5%, fiqa actually improves from
75.1% → 78.4%). Widening the serve window is therefore **not** a
reliable rescue mechanism — the displacement is *out of the candidate
set*, not within it. R@100 stays high because rroq158 still admits
the labeled relevant docs; the displacement is among the non-relevant
tail of FP16's top-100. For workloads where exact top-10 rank
fidelity vs FP16 is critical, opt into `rroq4_riem` (the
no-quality-loss lane at ~96% top-10 overlap) or use rroq158 with an
FP16 rerank on the shortlist (`benchmarks/diag_rroq158_rescue.py`
validates that an FP16 rerank on top-32/top-64 fully closes the
NDCG@10 gap on arguana / scifact / scidocs with no R@100 regression).

#### Honest CPU-latency caveat (post-Phase-7-followup)

The Rust SIMD rroq158 CPU kernel is currently **2–5× slower than the
FP16 AVX2 baseline** at the production batch shape (2000 doc
candidates per query × ~30 query tokens), down from ~5–9× pre-fix
after the post-Phase-7 CPU lane refresh. The four optimisations that
landed in the production lane (and the wheel that backs this table):

1. **Zero-copy `_to_np`** in
   [`scorer.py`](../voyager_index/_internal/inference/shard_engine/scorer.py)
   that bypasses `np.ascontiguousarray` for already-contiguous
   CPU-resident tensors entering the rroq158 / rroq4_riem dispatch
   path. Eliminates one redundant per-query allocation + memcpy of
   the candidate gather tensor.
2. **Inner-loop reorder** in
   [`fused_rroq158.rs::score_pair_body`](../src/kernels/shard_engine/src/fused_rroq158.rs)
   that amortises the doc-side popcount of `s_g` across all query
   tokens. Mirrors the FP16 amortisation pattern in `fused_maxsim.rs`.
   The matching reorder for the byte→fp32 nibble unpack in
   [`fused_rroq4_riem.rs`](../src/kernels/shard_engine/src/fused_rroq4_riem.rs)
   was already in place from the Phase-7 followup.
3. **`threadpoolctl.threadpool_limits` cap** around the BLAS matrix
   multiplications in
   [`rroq158.encode_query`](../voyager_index/_internal/inference/quantization/rroq158.py)
   and
   [`rroq4_riem.encode_query`](../voyager_index/_internal/inference/quantization/rroq4_riem.py),
   plus a safety net around the kernel call in `scorer.py` to stop
   OpenBLAS / MKL from oversubscribing while rayon is already saturating
   the CPU. On the 128-physical-core host this single change accounts
   for the bulk of the 2–3× speedup.
4. **Numpy fancy-indexing fast path** in the BEIR harness's
   `_rroq158_score_candidates` / `_rroq4_riem_score_candidates`
   ([`benchmarks/beir_benchmark.py`](../benchmarks/beir_benchmark.py))
   to bypass `torch.index_select` on CPU; the production search lane
   already uses the same numpy path so this only affects the harness
   measurement.

Per-dataset CPU speed-ups vs the pre-fix lane: rroq158 is **2.0×
(quora) → 5.0× (nfcorpus, scifact)** faster, rroq4_riem is **1.9×
(quora) → 2.9× (nfcorpus)** faster. Quality is unchanged on every
cell — the kernels are deterministic. The remaining headroom is in
the BLAS-bound query-encoding stage (FWHT rotation + centroid table
look-up); shrinking that further is the next item on the post-fix
backlog. CPU rroq158 ships as the default because the **storage
win** (~5.5× smaller doc tokens) is the primary value for
disk-bound deployments, and the kernel now closes most of the
absolute-latency gap to FP16 at this batch shape.

GPU lane: fused two-stage Triton kernel
(`voyager_index._internal.kernels.triton_roq_rroq158`), parity ≤ 1e-4
vs the python reference. CPU lane: Rust SIMD kernel
(`latence_shard_engine.rroq158_score_batch`) with hardware `popcnt` +
AVX2/BMI2/FMA + cached rayon thread pool, bitwise parity to rtol=1e-4
vs the python reference. Existing FP16 / RROQ158 / RROQ4_RIEM indexes
load unchanged — the manifest carries the build-time codec, so flipping
the default is non-breaking for deployed clusters.

The math (RaBitQ extension + Riemannian log map + FWHT-rotated tangent
ternary + K = 8192 derivation) is in
[docs/guides/rroq-mathematics.md](guides/rroq-mathematics.md). The
public-facing write-up is at
[docs/posts/sub-2-bit-late-interaction.md](posts/sub-2-bit-late-interaction.md).

### No-quality-loss codec: RROQ-4 Riemannian (K=8192, 4-bit asymmetric)

`Compression.RROQ4_RIEM` is the production option for workloads that
reject any quality regression vs FP16 but still want the storage win
of low-bit ROQ. It applies the same Riemannian-aware spherical-k-means +
FWHT pipeline as RROQ-1.58, but encodes the residual as **4-bit
asymmetric per-group** (default `group_size=32`, mins/deltas in fp16)
instead of ternary.

| codec        | per-token | NDCG@10 gap vs fp16 | Disk vs fp16 | Latency vs fp16 | Status |
| ------------ | --------: | ------------------: | -----------: | --------------: | -----: |
| `fp16`       | 256 B     | 0 (baseline)        | 1×           | 1× (baseline)   | shipped |
| `rroq4_riem` | ~88 B     | +0.02 pt avg, max ±0.05 pt | ~3× smaller | ~5.0× slower GPU avg, ~7.2× slower CPU avg | shipped (no-quality-loss lane) |
| `rroq158` (gs=128, **SOTA default**) | **~40 B** | within ±0.005 NDCG vs gs=32 on validated cells, plus the −1.43 pt vs FP16 from gs=32 | **~6.4× smaller** | 0.67–0.91× CPU p95 vs gs=32 baseline; GPU at ~p95 parity with FP16 | shipped (default) |
| `rroq158` (gs=32, prev default) | ~46 B     | −1.43 pt avg vs FP16, max −2.69 pt (arguana) | ~5.5× smaller | 1.13× GPU avg, ~3.2× slower CPU avg vs FP16 | shipped (override `Rroq158Config(group_size=32)`) |

Both kernels are wired and parity-tested:

- **GPU**: fused Triton kernel `roq_maxsim_rroq4_riem`
  (`voyager_index._internal.kernels.triton_roq_rroq4_riem`).
- **CPU**: Rust SIMD kernel `latence_shard_engine.rroq4_riem_score_batch`
  with AVX2/FMA + cached rayon thread pool, parity to rtol=1e-4 vs the
  python reference (validated by `tests/test_rroq4_riem_kernel.py`).

End-to-end build + search is covered by
`tests/test_rroq4_riem_e2e.py::test_rroq4_riem_build_and_search_cpu`.

When to pick which codec: use **rroq158** (gs=128 SOTA default) when
latency parity with FP16 matters more than a 1-point NDCG@10 budget on
hard datasets (most production retrieval workloads). Pin
`Rroq158Config(group_size=64)` for high-intra-token-variance corpora
(e.g. arguana). Use **rroq4_riem** when you cannot accept the rroq158
NDCG@10 cost (e.g. regulated domains, high-stakes ranking) and the
latency hit is acceptable. The full decision matrix and per-dim recipe
live in [docs/guides/quantization-tuning.md](guides/quantization-tuning.md).

#### Phase-7 followup: end-to-end CPU lane refresh for `rroq4_riem`

We ran a per-stage breakdown
([`benchmarks/diag_rroq4_riem_breakdown.py`](../benchmarks/diag_rroq4_riem_breakdown.py))
to attribute the originally recorded 5.0×/12.65× slowdown to actual
stages of the dispatch path. On the CPU lane the Rust SIMD kernel was
~58% of wall-clock; on the GPU lane the Triton kernel was ~80% of
wall-clock.

The CPU lane then picked up the **same four optimisations shipped for
rroq158** (zero-copy `_to_np`, BLAS thread cap around encode + score,
numpy fancy-indexing in the harness, plus the pre-existing nibble-
unpack amortisation in `fused_rroq4_riem.rs::score_pair_body` from
the original Phase-7 followup). The BEIR sweep was then re-measured
end-to-end:

- **CPU avg p95: 1304 ms → 741 ms (−43%, 12.65× → 7.18× vs fp16)**
- per-dataset speed-ups: **1.9× (quora) → 2.9× (nfcorpus)** with
  every dataset improved
- quality unchanged on every cell — the kernel is deterministic
  (validated by `tests/test_rroq4_riem_kernel.py` to rtol=1e-4 vs
  the python reference)

The GPU path was not modified by this followup. The structural cost
(per-group 4-bit dequant + FMA + asymmetric mins on top of the FP16
GEMM/MaxSim baseline) means rroq4_riem will remain slower than fp16
in absolute latency at this batch shape; the win is **storage with
zero quality regression**, not throughput.

### Comparison vs next-plaid

[next-plaid](https://github.com/lightonai/next-plaid) is an important
open-source reference for ColBERT-style serving. Their published numbers are
measured on **NVIDIA H100 80 GB** with the same embedding model. Our numbers
above are measured on an **RTX A5000** and are **search-only**; their
reported QPS includes encoding. Quora is omitted below because their README
uses a much larger corpus for that dataset.

| Dataset  | System     | NDCG@10    | MAP@100    | Recall@100 | GPU QPS   | GPU P95 (ms) | CPU QPS   | CPU P95 (ms) |
|----------|------------|-----------:|-----------:|-----------:|----------:|-------------:|----------:|-------------:|
| arguana  | voyager    | **0.3679** | **0.2598** | **0.9586** | **270.0** | **4.1**      | **41.6**  | **202.7**    |
|          | next-plaid | 0.3499     | 0.2457     | 0.9337     | 13.6      | 170.1        | 17.4      | 454.7        |
| fiqa     | voyager    | 0.4436     | 0.3818     | 0.7297     | **164.8** | **5.0**      | **80.2**  | **115.7**    |
|          | next-plaid | **0.4506** | **0.3871** | **0.7459** | 18.2      | 170.6        | 17.6      | 259.1        |
| nfcorpus | voyager    | **0.3833** | **0.1963** | **0.3348** | **282.6** | **3.8**      | **123.3** | **84.4**     |
|          | next-plaid | 0.3828     | 0.1870     | 0.3228     | 6.6       | 262.1        | 16.9      | 219.4        |
| scidocs  | voyager    | **0.1977** | **0.1383** | 0.4369     | **246.8** | **4.3**      | **83.9**  | **111.8**    |
|          | next-plaid | 0.1914     | 0.1352     | **0.4418** | 17.5      | 139.3        | 16.5      | 281.7        |
| scifact  | voyager    | 0.7544     | 0.7141     | 0.9567     | **263.4** | **4.0**      | **69.1**  | **138.4**    |
|          | next-plaid | **0.7593** | **0.7186** | **0.9633** | 7.9       | 169.5        | 16.9      | 305.4        |

In our current benchmark setup, voyager-index is **competitive or better on
retrieval quality** across most listed datasets and shows **materially higher
search throughput with much lower P95 latency** on an RTX A5000. **This is
not a fully apples-to-apples comparison:** next-plaid reports H100 numbers
and includes encoding in QPS, while our numbers are search-only on a smaller
GPU. The table above uses full-query evaluation specifically to avoid
publishing a flattering slice.

## Groundedness Tracker

Post-generation hallucination scoring runs in the optional
Latence Trace sidecar and is benchmarked separately from the
retrieval engine. See the
[Groundedness sidecar guide](guides/groundedness-sidecar.md) for the
integration shape and [latence.ai](https://latence.ai) for the latest
minimal-pair, RAGTruth, HaluEval and FActScore numbers.

## Benchmark Layers

### 1. OSS smoke benchmark

`benchmarks/oss_reference_benchmark.py` is the small, reproducible benchmark
used for package and API sanity checks.

It exercises:

- MaxSim kernels
- reference API ingest
- reference API search
- multimodal search
- graph-off OSS sanity checks

Run it with:

```bash
python benchmarks/oss_reference_benchmark.py --device cpu --points 16 --top-k 3
python benchmarks/oss_reference_benchmark.py --device cuda --points 16 --top-k 3
```

This benchmark is intentionally small. It is for regression detection, not for
headline product comparisons.

### 1B. Optional Latence graph benchmark

`tools/benchmarks/benchmark_latence_graph_quality.py` is the graph-aware quality
and conformance harness for the optional premium lane.

It includes:

- a tiny synthetic regression fixture for deterministic local vs community rescue checks
- a representative fixture for graph-shaped vs ordinary query behavior
- route-conformance checks for `graph_mode`, additive merge semantics, and provenance tags
- latency, candidate-growth, and solver-overhead summaries
- ablations for local-only, community-only, and full graph settings

What the harness is actually exercising:

- the base retrieval lane still runs first; graph is not the primary router
- graph inputs are target-linked Latence graph contracts derived from the corpus
  through the Dataset Intelligence / sidecar sync path
- rescue happens across local neighborhoods, community themes, and linked
  evidence targets
- public reporting proves additive uplift and route correctness without
  disclosing every proprietary threshold or cue list

Run it with:

```bash
python tools/benchmarks/benchmark_latence_graph_quality.py --mode benchmark
python tools/benchmarks/benchmark_latence_graph_quality.py --mode ablation
```

Important interpretation note:

- the optional graph lane is additive, so the benchmark tracks candidate-pool coverage and route conformance
- it is not trying to prove that graph rescue must always promote documents into the unaugmented top-`k` head
- ordinary queries should stay graph-off in `auto`
- graph-shaped or compliance-style queries should show additive rescue with the expected `graph_local` or `graph_community` provenance tags

Current representative snapshot from the shipped fixture-backed harness:

| Metric | Current value |
|---|---|
| graph-shaped recall delta | `+0.75` |
| graph-shaped NDCG delta | `+0.333` |
| graph-shaped support coverage delta | `+0.75` |
| ordinary-query recall / NDCG / support delta | `0.0 / 0.0 / 0.0` |
| graph applied rate | `0.571` |
| average added candidates on graph-shaped queries | `3.5` |
| route-conformance checks | `all passed` |

These numbers are valuable because they prove the shipped graph lane is working
as intended, but they are still fixture-backed graph evidence. They are not the
same thing as a public graph-on shard BEIR benchmark.

For the architecture and graph-data provenance model behind those numbers, see
[Latence Graph Sidecar Guide](guides/latence-graph-sidecar.md).

### 2. Product benchmark

The product benchmark is the shard retrieval benchmark on the 100k corpus:

- same embeddings across all systems
- same `top_k`
- same recall target
- same warmup policy
- same hardware per comparison table
- separate reporting for streamed GPU and GPU-corpus modes

The public README BEIR table is part of this product benchmark layer. It proves
the shard-first production lane, including Triton GPU scoring and multiworker
fused Rust CPU scoring, but it does not by itself prove graph-lane value.

## Methodology

Every published table should include:

- hardware: CPU model, RAM, GPU model, VRAM, storage type
- software: Python, PyTorch, CUDA, Triton, FAISS versions
- corpus: dataset name, document count, token/patch length assumptions
- query set: number of queries and relevance source
- warmup policy: how many warmup runs were discarded
- measured latency: p50, p95, p99
- quality: at least recall@10, and preferably NDCG or MRR when available
- throughput: QPS only after the recall target is stated
- for graph-aware claims: route-conformance, graph-applied rate, and additive candidate growth

## What Is Comparable

Comparable:

- same corpus
- same embeddings
- same hardware class
- same recall target
- same output `top_k`

Not directly comparable:

- CPU streamed path vs GPU-corpus path without calling out corpus placement
- throughput-only tables with no recall
- numbers taken from cold runs when warmup-heavy kernels are involved
- vendor benchmark tables that use different corpora, encoders, or relevance labels

## Warmup Policy

For `voyager-index` benchmark tables:

- warmup runs are excluded from measured latency
- Triton kernels must be warmed before comparing steady-state latency
- CPU benchmarks should note thread count and whether the run is single-worker or multi-worker

## Reporting Guidance

Recommended columns:

| System | Mode | Corpus placement | Recall@10 | p50 | p95 | QPS | Notes |
|---|---|---|---|---|---|---|---|
| Voyager | shard streamed | CPU/disk -> GPU | ... | ... | ... | ... | ... |

## 100k Comparison Placeholder

Pending fresh measurement on the same corpus and hardware:

| System | Mode | Corpus placement | Recall@10 | p50 | p95 | QPS | Status |
|---|---|---|---|---|---|---|---|
| Plaid | pending | pending | pending | pending | pending | pending | pending measurement |
| FastPlaid | GPU corpus | pending | pending | pending | pending | pending | pending measurement |
| Qdrant | pending | pending | pending | pending | pending | pending | pending measurement |
| Voyager | shard streamed | CPU/disk -> GPU | pending | pending | pending | pending | pending measurement |
| Voyager | shard GPU corpus | GPU resident | pending | pending | pending | pending | pending measurement |

## Interpretation

When the table is filled in, read it in this order:

1. recall
2. p50 and p95 latency
3. QPS
4. corpus placement and hardware notes

That ordering is deliberate. A faster system that quietly drops relevant
documents is not a win for late-interaction retrieval.
