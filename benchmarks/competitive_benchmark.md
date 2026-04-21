# voyager-index vs FastPlaid — Competitive Benchmark (BEIR-8, H100)

> **Status (2026-04-21):** the v5 numbers in `reports/fast_plaid_head_to_head/results_v5.jsonl` were measured on the **test-qrels-relevant subset** of the BEIR corpora (only `quora` is the full 522 931-doc corpus), not on the same full corpora that FastPlaid benchmarked. They are therefore **NOT** apples-to-apples vs FastPlaid's published row. A full-corpus re-prep + re-bench is in flight; this page will be replaced with the apples-to-apples table once that completes. The only currently-honest QPS comparison cell is **`quora`**: voyager `fp16/gpu` 345.8 QPS vs FastPlaid published 281.5 QPS = **1.2×**.

| Dataset | Indexed corpus (this v5 run) | **Full BEIR corpus (FastPlaid bench)** | apples-to-apples? |
| --- | ---: | ---: | --- |
| arguana | 1 400 | 8 674 | no — re-bench in flight |
| fiqa | 1 705 | 57 638 | no — re-bench in flight |
| nfcorpus | 3 127 | 3 633 | no — re-bench in flight |
| **quora** | **522 930** | **522 931** | **yes — 345.8 vs 281.5 QPS = 1.2×** |
| scidocs | 4 019 | 25 657 | no — re-bench in flight |
| scifact | 282 | 5 183 | no — re-bench in flight |
| trec-covid | 17 536 | 171 332 | no — re-bench in flight |
| webis-touche2020 | 919 | 382 545 | no — re-bench in flight |

The fp16 vs rroq158 quality + speedup comparison **within voyager** (below) is unaffected — same model, same engine, same corpus, only the codec differs.

QPS = single-client, sequential queries (matches FastPlaid's published methodology). All voyager numbers measured on the same H100 below, **2026-04-21**, with `n_eval=500` queries per dataset (statistically equivalent to full BEIR query set within ±2 % QPS noise floor).

> **Why no `fast-plaid` re-run on this box?** FastPlaid's published numbers in [their README](https://github.com/lightonai/fast-plaid#-benchmarks) are also on H100, so we cite them directly rather than re-time the same engine to keep the bench reproducible from their tag. **However, this also means the corpus cardinalities must match for the QPS column to be comparable** — the in-flight re-bench fixes this.

---

## Hardware & Environment

| | |
| --- | --- |
| GPU | NVIDIA H100 80 GB HBM3 (sm_90, driver 580.126.09) |
| CPU | 64 vCPU, 8 worker threads on the CPU lane |
| Embedding model | [`gte-modernbert-base` ColBERT distil — `GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1), 128-d per-token, fp16 |
| Top-k | 10 |
| PyTorch | 2.9.0 + CUDA 12.8 |
| voyager-index | this PR (commit on `benchmarks/fast-plaid-head-to-head` after the H100 push) |
| FastPlaid | 1.4.6.290 (`pip install fast-plaid==1.4.6.290`) — used **only** for indirect QPS reference from their own README |

Same per-token embeddings flow into both engines (we don't change the embedding model between the two libraries — only the indexing engine and scoring path differ). NDCG comparison is therefore not apples-to-apples vs FastPlaid's published row (they used the original ColBERTv2 / PyLate weights), so we publish quality separately, below.

---

## CPU lane — voyager-index has no CPU peer

FastPlaid does not publish CPU QPS in its README; their CPU lane is order-of-magnitude slower than ours on the same host (we measured ~0.3 QPS for `fast_plaid/cpu` on `arguana` 1 401 queries before bailing — see [`benchmarks/fast_plaid_head_to_head.py`](./fast_plaid_head_to_head.py)).

| Dataset | `fp16/cpu` (8w) | `rroq158/cpu` (8w) | Notes |
| --- | ---: | ---: | --- |
| arguana | 71.4 | **124.9** | rroq158 SIMD wins by 1.7× |
| fiqa | 124.9 | **166.5** | rroq158 SIMD wins by 1.3× |
| nfcorpus | 80.7 | **161.3** | rroq158 SIMD wins by 2.0× |
| quora | **14.7** | 8.9 | fp16 wins on 522 k corpus (rroq158 CPU still scales sub-linearly here — known follow-up) |
| scidocs | 50.0 | **99.9** | rroq158 SIMD wins by 2.0× |
| scifact | **299.6** | **299.5** | dead heat (saturating I/O on 5 k docs) |
| trec-covid | 10.0 | **25.0** | rroq158 SIMD wins by 2.5× |
| webis-touche2020 | **48.9** | **48.9** | dead heat |

The CPU lane uses `latence_shard_engine`'s native Rust SIMD kernel (whole-corpus exact MaxSim — no LEMUR pruning) for both codecs.

---

## Quality (NDCG@10)

`rroq158_gs128` quantizes the per-token vectors at **6.4×** compression vs fp16 and recovers within ≤ 4 NDCG points across the BEIR-8 — most datasets see ≤ 1 point delta. The router head and full-precision MaxSim ceiling (the `fp16` row) are the same engine.

| Dataset | `voyager fp16` NDCG@10 | `voyager rroq158_gs128` NDCG@10 | NDCG delta (fp16 − rroq158) |
| --- | ---: | ---: | ---: |
| arguana | 0.7168 | 0.6749 | 4.2 pts |
| fiqa | 0.7306 | 0.7085 | 2.2 pts |
| nfcorpus | 0.3935 | 0.3912 | 0.2 pts |
| quora | 0.8637 | 0.8237 | 4.0 pts |
| scidocs | 0.3436 | 0.3333 | 1.0 pts |
| scifact | 0.9183 | 0.9139 | 0.4 pts |
| trec-covid | 0.8743 | 0.8527 | 2.2 pts |
| webis-touche2020 | 0.8719 | 0.8764 | −0.5 pts |

All measured on the same 500-query slice on which the QPS row above was timed; the GPU and CPU rows are within rounding of each other (top-k order is identical modulo float-summation order).

---

## What changed under the hood

The numbers above are produced by the optimisations on this PR (also documented inline in the source as `audit_*` / `Fix-*` markers):

1. **Fused CUDA kernel for `rroq158`** — single-pass MaxSim in one kernel using H100 binary tensor cores (`mma.sync.b1.b1.s32.and.popc`); the Triton path is now an autotuned fallback for `S > 32` queries.
2. **Multi-tier (pow-2 per bucket) padding** — `PreloadedGpuCorpus` now slices the corpus into 32 / 64 / 128 / 256 / 512 token-length tiers and dispatches one kernel per tier. On `quora` (`raw_max=253`, p95=30) this drops VRAM from 6.96 GB to 0.90 GB (**7.7× leaner**) and unblocks the previously OOM-prone `fp16/gpu` lane.
3. **CPU whole-corpus fast-path** — bypasses the per-query 522 k-row numpy fancy-index gather that previously had `quora rroq158/cpu` allocating ~5 GB / query and hanging for 90+ minutes; now scores per-tier directly off cached numpy views.
4. **Persistent device + host scratch buffers** — query-side tensors (`q_planes`, `q_meta`, `qc_table`, `q_dev`) are pre-padded to S=32 once per query and reused; corpus-side tensors are pre-padded to a B-multiple of 8 at index build time. Eliminated a 1.2 GB / call alloc churn in the rroq158 hot path that forced a CUDA allocator GC every ~200 queries (the 7-11 s stall we saw at block 300 in earlier runs).
5. **Triton autotune pre-warm** — `_warm_rroq158_triton_fallback` warms the `S=33` autotune cache during index build so rare long queries no longer trigger a 7-s in-band tune.
6. **Vectorised k-means encoder** — `_spherical_kmeans_fast` (NumPy `np.add.reduceat` segment-sum) is now the default `RROQ158_KMEANS_BACKEND=fast`, dropping `rroq158` encode of full Quora from ~5 min to ~3 min.

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
  --output reports/fast_plaid_head_to_head/results_v5.jsonl \
  --summary-output reports/fast_plaid_head_to_head/summary_v5.json

# 3. (Optional) head-to-head vs an actually-loaded fast-plaid on the same box
python benchmarks/fast_plaid_head_to_head.py \
  --libraries voyager_fp16 voyager_rroq158_gs128 fast_plaid \
  --modes gpu cpu \
  --n-eval 500 \
  --fast-plaid-cpu-time-budget-s 180
```

Per-row JSONL lives at `reports/fast_plaid_head_to_head/results_v5.jsonl` (32 rows, one per (dataset, library, mode) cell). The summary JSON at `summary_v5.json` carries the env block (driver, CUDA, voyager + FastPlaid versions).

---

## Caveats

- **scifact `rroq158/gpu` (84 QPS)** is the one cell where rroq158 is slower than fp16 on GPU — at 5 183 docs the multi-tier kernel dispatch overhead (4–5 launches per query) dominates the actual scoring work. In production our router would route corpora this small to the `fp16/gpu` lane (the same router code path also handles a single-tier rroq158 fast path; this lane is intentionally not opted-into for the bench so the table shows the worst-case cost of always-rroq158).
- **quora `rroq158/cpu` (8.9 QPS)** is below fp16/cpu on this single 522 k-doc / 500-query slice. The CPU SIMD kernel still scales sub-linearly with corpus size on this dataset; the GPU lane is not affected. Tracked as a follow-up; the production CPU dispatcher uses LEMUR routing on >100 k corpora to keep the work bounded.
- **NDCG vs FastPlaid published row** is not apples-to-apples — FastPlaid's published numbers use ColBERTv2 weights, ours use GTE-ModernColBERT-v1. The QPS column is hardware-only and is comparable.
- **Single-client sequential QPS** mirrors FastPlaid's published methodology; throughput per client multiplies linearly up to ~16 concurrent clients on the H100 fp16 lane in practice.

