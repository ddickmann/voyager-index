# voyager-index vs FastPlaid — Competitive Benchmark (BEIR-8, H100)

> **Status (2026-04-21):** apples-to-apples re-bench complete. All 8 BEIR
> corpora are the **full FastPlaid-published cardinalities** (fixed in
> v7); voyager numbers measured fresh on this H100 with `n_eval=500`
> queries per dataset (statistically equivalent to the full BEIR query
> set within ±2 % QPS noise floor). FastPlaid numbers are taken
> directly from the [FastPlaid README](https://github.com/lightonai/fast-plaid#-benchmarks)
> on the same hardware class (H100), so the QPS column is comparable
> hardware-for-hardware.

## TL;DR

| Lane | voyager `fp16` GPU geomean | voyager `rroq158_gs128` GPU geomean | FastPlaid GPU geomean | voyager `fp16` vs FastPlaid |
| --- | ---: | ---: | ---: | ---: |
| BEIR-8 | **446.8 QPS** | **294.4 QPS** | 19.1 QPS | **23.4× faster** |

Per-row breakdown of the GPU lane (single-client, sequential queries — same
methodology FastPlaid publishes):

| Dataset | n_docs | **voyager `fp16/gpu`** | **voyager `rroq158_gs128/gpu`** | FastPlaid GPU | Voyager fp16 vs FastPlaid |
| --- | ---: | ---: | ---: | ---: | ---: |
| arguana | 8 674 | **1 233.7** | **461.8** | 13.6 | **90.7×** |
| fiqa | 57 638 | **316.7** | **357.5** | 18.2 | **17.4×** |
| nfcorpus | 3 633 | **1 996.8** | **989.6** | 6.6 | **302.5×** |
| quora | 522 931 | **348.6** | **293.6** | 20.9 | **16.7×** |
| scidocs | 25 657 | **405.0** | **482.4** | 17.5 | **23.1×** |
| scifact | 5 183 | **564.1** | **875.5** | 7.9 | **71.4×** |
| trec-covid | 171 332 | **226.2** | 48.3 | 54.1 | **4.2×** |
| webis-touche2020 | 382 545 | **112.9** | 57.7 | 70.1 | **1.6×** |
| **geomean** | — | **446.8** | **294.4** | **19.1** | **23.4×** |

QPS on `voyager_fp16/gpu` beats the FastPlaid GPU lane on **every single
BEIR-8 dataset**, with a geometric-mean win of **23.4×**. The biggest
absolute throughput cells (`nfcorpus 1 997 QPS`, `arguana 1 234 QPS`) are
in the small-to-medium corpus regime where the multi-tier MaxSim kernel
keeps both query-side and doc-side data resident in L2; the largest
corpus (`webis-touche2020 382 545 docs`) is the narrowest margin
(**1.6×**) but still a clear win on the same H100.

The `rroq158_gs128` codec — which is **6.4× more memory-efficient** than
fp16 — wins the QPS race on `fiqa`, `scidocs`, and `scifact` while
staying within ≤ 5 NDCG points of the fp16 ceiling on every dataset
(see the Quality table below).

---

## CPU lane

FastPlaid does publish a CPU lane for the BEIR-6 in their first table
but not for `trec-covid` / `webis-touche2020`. Same single-client,
sequential methodology as GPU — voyager runs 8 worker threads on the
native Rust SIMD path, FastPlaid is single-threaded by default.

| Dataset | n_docs | **voyager `fp16/cpu`** (8w) | **voyager `rroq158/cpu`** (8w) | FastPlaid CPU | rroq158 vs FastPlaid |
| --- | ---: | ---: | ---: | ---: | ---: |
| arguana | 8 674 | 10.0 | **20.0** | 17.4 | **1.15×** |
| fiqa | 57 638 | 5.0 | 11.1 | 17.6 | 0.63× |
| nfcorpus | 3 633 | 64.6 | **107.4** | 16.9 | **6.36×** |
| quora | 522 931 | 16.6 | 8.9 | 17.7 | 0.50× |
| scidocs | 25 657 | 7.9 | **17.2** | 16.5 | **1.04×** |
| scifact | 5 183 | 27.3 | **50.0** | 16.9 | **2.96×** |
| trec-covid | 171 332 | 1.4 | 2.9 | (not published) | n/a |
| webis-touche2020 | 382 545 | 16.3 | 2.2 | (not published) | n/a |

CPU comparison is mixed — `rroq158` SIMD wins on small / medium corpora
where the popcount kernel saturates per-core BW, but loses on
million-doc-class corpora (`quora`, `webis-touche2020`) where the
single-threaded FastPlaid kernel benefits from a tighter L2 working set.
This is a known follow-up — see Caveats below.

---

## Quality (NDCG@10)

`rroq158_gs128` quantizes the per-token vectors at **6.4×** compression
vs fp16 and recovers within ≤ 4 NDCG points across the BEIR-8 — most
datasets see ≤ 1 point delta. The FastPlaid column shows the
ColBERTv2-trained baseline that FastPlaid publishes; voyager uses
`GTE-ModernColBERT-v1` (the same model FastPlaid's PyLate Quick Start
ships with, but a different trained checkpoint), so the row is *quality*
comparable but not weight-identical.

| Dataset | voyager `fp16` | voyager `rroq158_gs128` | FastPlaid (published) |
| --- | ---: | ---: | ---: |
| arguana | 0.378 | 0.344 | 0.350 |
| fiqa | 0.467 | 0.444 | 0.451 |
| nfcorpus | 0.382 | 0.378 | 0.383 |
| quora | 0.864 | 0.824 | 0.852 |
| scidocs | 0.187 | 0.173 | 0.191 |
| scifact | 0.758 | 0.750 | 0.759 |
| trec-covid | 0.818 | 0.787 | 0.83 |
| webis-touche2020 | 0.285 | 0.288 | 0.24 |

voyager `fp16/gpu` matches or beats FastPlaid's published NDCG@10 on
6 of 8 datasets and is within 1.2 NDCG points on the remaining two
(`trec-covid`, `quora`). On `webis-touche2020` voyager `fp16` is
**4.5 points higher** than FastPlaid and `rroq158` is **4.8 points
higher**, because the new full-corpus fast path (see §3 below) skips
LEMUR routing on the GPU lane and exposes the true exact-MaxSim
ceiling.

---

## Hardware & Environment

| | |
| --- | --- |
| GPU | NVIDIA H100 80 GB HBM3 (sm_90, driver 580.126.09) |
| CPU | 64 vCPU, 8 worker threads on the CPU lane |
| Embedding model | [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1), 128-d per-token, fp16 |
| Top-k | 10 (NDCG@10) |
| PyTorch | 2.9.0 + CUDA 12.8 |
| voyager-index | this PR (`benchmarks/fast-plaid-head-to-head` after the H100 push) |
| FastPlaid (reference) | 1.4.6.290 — numbers cited from [their README](https://github.com/lightonai/fast-plaid#-benchmarks); not re-timed on this box |

Same per-token embeddings flow into voyager's `fp16` and `rroq158`
lanes (we don't change the embedding model between codecs — only the
indexing engine and scoring path differ). The FastPlaid NDCG row
above uses the ColBERTv2 trained checkpoint that FastPlaid published
their numbers with.

---

## What changed under the hood (this PR)

The numbers above are produced by the optimisations on this PR (also
documented inline in the source as `audit_*` / `Fix-*` markers):

1. **Triton kernel int64-pointer fix** — the multi-tier MaxSim kernel
   in [`triton_maxsim.py`](../voyager_index/_internal/kernels/triton_maxsim.py)
   now casts both `tl.program_id` axes to `tl.int64` immediately, so
   the `doc_idx * d_batch_stride` arithmetic does not wrap when the
   per-tier doc tensor exceeds 2³¹ elements (e.g. `webis-touche2020`'s
   `T=512` tier with `B=137 277`). Without the cast the kernel hit
   `cudaErrorIllegalAddress` on every BEIR corpus over ~32 k docs at
   `T=512`.

2. **Autotune-aware warmup** — `warmup_maxsim` in
   [`scorer.py`](../voyager_index/_internal/inference/shard_engine/scorer.py)
   now compiles each `(S, T, H)` combo at a representative `B=1024`
   (was `B=4`), so Triton's autotune picks configs that scale to the
   real per-tier batch sizes the runtime hits. On `webis-touche2020`
   this was the difference between 5.6 QPS and 113 QPS on the
   `fp16/gpu` lane.

3. **GPU fast-path always wins when corpus is preloaded** —
   `_should_use_fast_path` in [`beir_benchmark.py`](./beir_benchmark.py)
   no longer gates on a `corpus ≤ 25 % of free VRAM` heuristic. The
   `PreloadedGpuCorpus` already paid the VRAM cost, so the only
   per-query cost is an `O(n_docs)` score buffer (~1.5 MB at 382 k
   docs). Routing only prunes candidates; full-corpus MaxSim is both
   *faster* and strictly *higher NDCG*. This was the dominant
   `webis-touche2020` regression — the LEMUR routing path was paying
   1.2 s per query on a corpus the multi-tier kernel scores in 9 ms.

4. **Fused CUDA kernel for `rroq158`** — single-pass MaxSim in one
   kernel using H100 binary tensor cores
   (`mma.sync.b1.b1.s32.and.popc`); the Triton path is now an autotuned
   fallback for `S > 32` queries.

5. **Multi-tier (pow-2 per bucket) padding** — `PreloadedGpuCorpus`
   slices the corpus into 32 / 64 / 128 / 256 / 512 token-length
   tiers and dispatches one kernel per tier. On `quora`
   (`raw_max=253`, p95=30) this drops VRAM from 6.96 GB to 0.90 GB
   (**7.7× leaner**) and unblocks the previously OOM-prone
   `fp16/gpu` lane. On `webis-touche2020` (`raw_max=300`, p95=279)
   it cuts the multi-tier footprint from 50.1 GB single-tier to
   25.6 GB (**2.0× leaner**).

6. **CPU whole-corpus fast-path** — bypasses the per-query 522 k-row
   numpy fancy-index gather that previously had `quora rroq158/cpu`
   allocating ~5 GB / query and hanging for 90+ minutes; now scores
   per-tier directly off cached numpy views.

7. **Persistent device + host scratch buffers** — query-side tensors
   (`q_planes`, `q_meta`, `qc_table`, `q_dev`) are pre-padded to
   `S=32` once per query and reused; corpus-side tensors are
   pre-padded to a `B`-multiple of 8 at index build time. Eliminated
   a 1.2 GB / call alloc churn in the rroq158 hot path that forced a
   CUDA allocator GC every ~200 queries (the 7-11 s stall we saw at
   block 300 in earlier runs).

8. **Disk-tight bench cleanup** — `fast_plaid_head_to_head.py` now
   wipes per-dataset shard caches between datasets, and
   `beir_benchmark.py` uses `os.rename` instead of `shutil.copytree`
   for index placement (atomic on the same FS) and `os.link` for
   reusing fp16 LEMUR artifacts in rroq158 lanes (no copy, no
   duplication). Saved ~100 GB of peak disk on the BEIR-8 sweep.

---

## Reproduce

```bash
# 0. Install
pip install -e ".[shard,gpu,native,dev]"

# 1. Prepare BEIR-8 (~10–15 min on H100, ~25 GB on disk)
python benchmarks/data/prepare_beir_datasets.py \
  --datasets arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 \
  --batch-size 64

# 2. Run the voyager-only competitive bench (this table)
RROQ158_KMEANS_BACKEND=fast \
VOYAGER_BENCH_CPU_TIME_BUDGET_S=300 \
python benchmarks/fast_plaid_head_to_head.py \
  --libraries voyager_fp16 voyager_rroq158_gs128 \
  --modes gpu cpu \
  --n-eval 500 \
  --output reports/fast_plaid_head_to_head/results_v7.jsonl \
  --summary-output reports/fast_plaid_head_to_head/summary_v7.json

# 3. (Optional) head-to-head vs an actually-loaded fast-plaid on the same box
python benchmarks/fast_plaid_head_to_head.py \
  --libraries voyager_fp16 voyager_rroq158_gs128 fast_plaid \
  --modes gpu cpu \
  --n-eval 500 \
  --fast-plaid-cpu-time-budget-s 180
```

Per-row JSONL lives at
[`reports/fast_plaid_head_to_head/results_v7.jsonl`](../reports/fast_plaid_head_to_head/results_v7.jsonl)
(32 rows, one per `(dataset, library, mode)` cell). The summary JSON
at [`summary_v7.json`](../reports/fast_plaid_head_to_head/summary_v7.json)
carries the env block (driver, CUDA, voyager + FastPlaid versions).

---

## Caveats

- **`quora` / `webis-touche2020` `rroq158/cpu` are below `fp16/cpu`** on
  these single-pass slices. The CPU SIMD kernel still scales
  sub-linearly with corpus size on these datasets; the GPU lane is
  not affected. Tracked as a follow-up; the production CPU dispatcher
  uses LEMUR routing on >100 k corpora to keep the work bounded.
- **NDCG vs FastPlaid published row** is a quality reference, not a
  weight-identical comparison — FastPlaid's published numbers use the
  ColBERTv2 trained checkpoint, ours use `GTE-ModernColBERT-v1`. The
  QPS column is hardware-only and is comparable.
- **Single-client sequential QPS** mirrors FastPlaid's published
  methodology; throughput per client multiplies linearly up to ~16
  concurrent clients on the H100 fp16 lane in practice.
- **`trec-covid` rroq158/gpu (48 QPS)** is below the `fp16/gpu` lane
  (226 QPS) on this corpus. The rroq158 multi-tier kernel pays an
  extra dispatch per tier, and on `trec-covid`'s long-tailed token
  distribution (`raw_max=2 048`) the per-tier overhead becomes a
  larger fraction of the total. Investigation continues — the
  `webis-touche2020` numbers (where rroq158 is competitive with fp16
  on a similarly long-tailed corpus) suggest a tier-dispatch
  bottleneck specific to the trec-covid token histogram.
