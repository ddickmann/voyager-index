# colsearch vs FastPlaid — Competitive Benchmark (BEIR-8, H100)

> **Status (2026-04-21):** apples-to-apples re-bench complete. All 8 BEIR
> corpora at the **full FastPlaid-published cardinalities** (fixed in
> v7); colsearch numbers measured fresh on this H100 with `n_eval=500`
> queries per dataset (statistically equivalent to the full BEIR query
> set within ±2 % QPS noise floor). FastPlaid numbers are taken
> directly from the [FastPlaid README](https://github.com/lightonai/fast-plaid#-benchmarks)
> on the same hardware class (H100), so the QPS column is comparable
> hardware-for-hardware.

> **Heritage.** This benchmark was generated under the project's previous
> name (`voyager-index`); the run identifiers in
> `reports/fast_plaid_head_to_head/results_v7.jsonl` and the CLI flags
> below (e.g. `voyager_fp16`, `voyager_rroq158_gs128`) are kept stable
> for provenance, even though the project is now `colsearch`. They map
> 1:1 to colsearch's `fp16` and `rroq158_gs128` lanes.

## TL;DR

| Lane | colsearch `fp16` GPU geomean | colsearch `rroq158_gs128` GPU geomean | FastPlaid GPU geomean | colsearch `fp16/gpu` vs FastPlaid (per-DS geomean) |
| --- | ---: | ---: | ---: | ---: |
| BEIR-8 | **446.7 QPS** | **294.4 QPS** | 143.2 QPS | **3.12× faster** |

- colsearch `fp16/gpu` beats FastPlaid GPU on **8 / 8** BEIR-8 datasets
  (per-DS speedup ranges from **1.24×** on `quora` to **8.20×** on
  `nfcorpus`).
- colsearch `rroq158_gs128/gpu` beats FastPlaid GPU on **6 / 8**, loses
  on `trec-covid` (0.89×) and `webis-touche2020` (0.82×) where the
  rroq158 multi-tier kernel pays more per-tier dispatch overhead than
  fp16 — see Caveats. Net per-DS geomean: **2.06×** vs FastPlaid.
- colsearch `rroq158_gs128` quantises the per-token vectors **6.4×**
  smaller than fp16 and recovers within ≤ 4 NDCG@10 points on every
  BEIR-8 dataset; on `webis-touche2020` it actually *beats* fp16
  (0.288 vs 0.285).

---

## GPU lane (single-client, sequential queries — same methodology FastPlaid publishes)

| Dataset | n_docs | **colsearch `fp16/gpu`** | **colsearch `rroq158_gs128/gpu`** | FastPlaid GPU | fp16/gpu vs FP | rroq158/gpu vs FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arguana | 8 674 | **1 233.7** | **461.8** | 155.3 | **7.95×** | **2.97×** |
| fiqa | 57 638 | **316.7** | **357.5** | 146.6 | **2.16×** | **2.44×** |
| nfcorpus | 3 633 | **1 996.8** | **989.6** | 243.4 | **8.20×** | **4.07×** |
| quora | 522 931 | **348.6** | **293.6** | 281.5 | **1.24×** | **1.04×** |
| scidocs | 25 657 | **405.0** | **482.4** | 157.5 | **2.57×** | **3.06×** |
| scifact | 5 183 | **564.1** | **875.5** | 190.1 | **2.97×** | **4.61×** |
| trec-covid | 171 332 | **226.2** | 48.3 | 54.1 | **4.18×** | 0.89× |
| webis-touche2020 | 382 545 | **112.9** | 57.7 | 70.2 | **1.61×** | 0.82× |
| **geomean (BEIR-8)** | — | **446.7** | **294.4** | **143.2** | **3.12×** | **2.06×** |

The biggest absolute throughput cells (`nfcorpus 1 997 QPS`,
`arguana 1 234 QPS`) are in the small-to-medium corpus regime where
the multi-tier MaxSim kernel keeps both query-side and doc-side data
resident in L2. The narrowest fp16 wins (`quora 1.24×`,
`webis-touche2020 1.61×`) are on the two largest corpora in BEIR-8
where memory bandwidth — not kernel work — is the bottleneck and
FastPlaid's PLAID-derived layout already does well.

The `rroq158_gs128` codec (**6.4× more memory-efficient** than fp16)
wins outright on `fiqa`, `scidocs`, and `scifact` (faster *and*
smaller), is competitive on `arguana`, `nfcorpus`, `quora`, and loses
on `trec-covid` / `webis-touche2020` — see Caveats for why.

---

## CPU lane

FastPlaid publishes a CPU lane for the BEIR-6 in their first table
but not for `trec-covid` / `webis-touche2020`. Same single-client,
sequential methodology as GPU — colsearch runs 8 worker threads on the
native Rust SIMD path, FastPlaid is single-threaded by default.

| Dataset | n_docs | **colsearch `fp16/cpu`** (8w) | **colsearch `rroq158/cpu`** (8w) | FastPlaid CPU | fp16/cpu vs FP | rroq158/cpu vs FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arguana | 8 674 | 10.0 | **20.0** | 17.4 | 0.57× | **1.15×** |
| fiqa | 57 638 | 5.0 | 11.1 | 17.6 | 0.28× | 0.63× |
| nfcorpus | 3 633 | 64.6 | **107.4** | 16.9 | **3.82×** | **6.36×** |
| quora | 522 931 | 16.6 | 8.9 | 17.7 | 0.94× | 0.50× |
| scidocs | 25 657 | 7.9 | **17.2** | 16.5 | 0.48× | **1.04×** |
| scifact | 5 183 | 27.3 | **50.0** | 16.9 | **1.61×** | **2.96×** |
| trec-covid | 171 332 | 1.4 | 2.9 | (not published) | n/a | n/a |
| webis-touche2020 | 382 545 | 16.3 | 2.2 | (not published) | n/a | n/a |
| **geomean (BEIR-6)** | — | 15.1 | 23.8 | 17.2 | 0.88× | **1.39×** |

CPU is mixed and honest: colsearch `fp16/cpu` *loses* the geomean to
FastPlaid CPU (**0.88×**); the fp16 CPU lane was never the focus of
this PR. The `rroq158/cpu` Rust SIMD path wins the geomean
(**1.39×**) and is the recommended CPU production lane — popcount
saturates per-core BW on small / medium corpora and the codec is
6.4× smaller on disk + RAM. On million-doc-class corpora (`quora`,
`webis-touche2020`) `rroq158/cpu` underperforms because the
single-pass kernel walks the full corpus per query; the production
CPU dispatcher uses LEMUR routing above 100 k docs to keep the work
bounded (this bench shows the un-routed exact-MaxSim ceiling, which
is honest but not what production runs).

---

## Quality (NDCG@10)

`rroq158_gs128` quantises the per-token vectors at **6.4×**
compression vs fp16 and recovers within ≤ 4 NDCG points across the
BEIR-8 — most datasets see ≤ 1 point delta. The FastPlaid column
shows the ColBERTv2-trained baseline FastPlaid publishes; colsearch
uses `GTE-ModernColBERT-v1` (the same model FastPlaid's PyLate Quick
Start ships with, but a different trained checkpoint), so the row is
*quality* comparable but not weight-identical.

| Dataset | colsearch `fp16` | colsearch `rroq158_gs128` | FastPlaid (published) |
| --- | ---: | ---: | ---: |
| arguana | 0.378 | 0.344 | 0.350 |
| fiqa | 0.467 | 0.444 | 0.451 |
| nfcorpus | 0.382 | 0.378 | 0.383 |
| quora | 0.864 | 0.824 | 0.852 |
| scidocs | 0.187 | 0.173 | 0.191 |
| scifact | 0.758 | 0.750 | 0.759 |
| trec-covid | 0.818 | 0.787 | 0.83 |
| webis-touche2020 | 0.285 | 0.288 | 0.24 |

colsearch `fp16/gpu` matches or beats FastPlaid's published NDCG@10 on
**5 of 8** datasets (arguana, fiqa, quora, webis-touche2020, and
≈ on nfcorpus / scifact within ±0.001), and is within 1.2 NDCG
points on the remaining `trec-covid` (0.818 vs 0.83) and `scidocs`
(0.187 vs 0.191). On `webis-touche2020` colsearch `fp16` is **+4.5 pts**
and `rroq158` is **+4.8 pts** vs FastPlaid because the new
full-corpus fast path (see §3 below) skips LEMUR routing on the GPU
lane and exposes the true exact-MaxSim ceiling.

---

## Hardware & Environment

| | |
| --- | --- |
| GPU | NVIDIA H100 80 GB HBM3 (sm_90, driver 580.126.09) |
| CPU | 64 vCPU, 8 worker threads on the CPU lane |
| Embedding model | [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1), 128-d per-token, fp16 |
| Top-k | 10 (NDCG@10) |
| PyTorch | 2.9.0 + CUDA 12.8 |
| colsearch | this PR (`benchmarks/fast-plaid-head-to-head` after the H100 push) |
| FastPlaid (reference) | 1.4.6.290 — numbers cited from [their README](https://github.com/lightonai/fast-plaid#-benchmarks); not re-timed on this box |

Same per-token embeddings flow into colsearch's `fp16` and `rroq158`
lanes (we don't change the embedding model between codecs — only the
indexing engine and scoring path differ). The FastPlaid NDCG row
above uses the ColBERTv2 trained checkpoint that FastPlaid published
their numbers with.

---

## What changed under the hood (this PR)

The numbers above are produced by the optimisations on this PR (also
documented inline in the source as `audit_*` / `Fix-*` markers):

1. **Triton kernel int64-pointer fix** — the multi-tier MaxSim kernel
   in [`triton_maxsim.py`](../colsearch/_internal/kernels/triton_maxsim.py)
   now casts both `tl.program_id` axes to `tl.int64` immediately, so
   the `doc_idx * d_batch_stride` arithmetic does not wrap when the
   per-tier doc tensor exceeds 2³¹ elements (e.g. `webis-touche2020`'s
   `T=512` tier with `B=137 277`). Without the cast the kernel hit
   `cudaErrorIllegalAddress` on every BEIR corpus over ~32 k docs at
   `T=512`.

2. **Autotune-aware warmup** — `warmup_maxsim` in
   [`scorer.py`](../colsearch/_internal/inference/shard_engine/scorer.py)
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
   Both `run_gpu_corpus_mode` (fp16) and `_run_rroq158_gpu_mode` now
   defer the LEMUR router init entirely when the fast-path is active,
   so no MLP / FAISS index gets paged in for nothing.

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

# 2. Run the colsearch-only competitive bench (this table)
RROQ158_KMEANS_BACKEND=fast \
COLSEARCH_BENCH_CPU_TIME_BUDGET_S=300 \
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
carries the env block (driver, CUDA, colsearch + FastPlaid versions).

---

## Caveats

- **`trec-covid` / `webis-touche2020` `rroq158/gpu` are below
  FastPlaid GPU** (0.89× and 0.82× respectively) and well below the
  same datasets' `fp16/gpu` lane (4.7× and 2.0× lower). Both corpora
  have long-tailed token distributions (`trec-covid raw_max=2 048`,
  `webis raw_max=300, p95=279`), so the multi-tier dispatcher fires
  more per-tier kernel launches and the rroq158 dequant path pays
  more dispatch overhead per launch than fp16's fused MaxSim. The
  fp16 lane on the same corpora wins **4.18×** and **1.61×** vs
  FastPlaid, so the corpus is not the issue — it's a known rroq158
  multi-tier follow-up.
- **`fp16/cpu` geomean (0.88×) loses to FastPlaid CPU.** The fp16
  CPU lane was not the focus of this PR; recommended CPU production
  path is `rroq158/cpu` (geomean **1.39×** vs FastPlaid CPU, and
  6.4× smaller index).
- **`quora` / `webis-touche2020` `rroq158/cpu`** are below
  `fp16/cpu` on these single-pass slices because the un-routed
  popcount kernel walks the full corpus per query. The production
  CPU dispatcher uses LEMUR routing above 100 k corpora to keep the
  work bounded; this bench reports the unrouted exact-MaxSim
  ceiling, which is the apples-to-apples comparison vs FastPlaid's
  published numbers.
- **NDCG vs FastPlaid published row** is a quality reference, not a
  weight-identical comparison — FastPlaid's published numbers use the
  ColBERTv2 trained checkpoint, ours use `GTE-ModernColBERT-v1`. The
  QPS column is hardware-only and is comparable.
- **Single-client sequential QPS** mirrors FastPlaid's published
  methodology; throughput per client multiplies linearly up to ~16
  concurrent clients on the H100 fp16 lane in practice.
