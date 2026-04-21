# voyager-index vs FastPlaid — head-to-head BEIR benchmark

A reproducible, single-command benchmark that runs voyager-index's
**rroq158 (group_size=128, the v0.1.6 SOTA default)** against
**[FastPlaid](https://github.com/lightonai/fast-plaid)** on the same
8-dataset BEIR matrix that FastPlaid publishes, using **identical
per-token embeddings** for both libraries so the comparison varies only
the indexing/scoring engine.

- Script: [`benchmarks/fast_plaid_head_to_head.py`](../../benchmarks/fast_plaid_head_to_head.py)
- Dataset prep: [`benchmarks/data/prepare_beir_datasets.py`](../../benchmarks/data/prepare_beir_datasets.py)
- Output: `reports/fast_plaid_head_to_head/{results.jsonl, summary.json}`

> [!IMPORTANT]
> FastPlaid's published benchmark was run on an **H100**. voyager-index's
> internal BEIR-6 sweep was run on an **A5000** (~4× less FP16 compute,
> ~4× less memory bandwidth). For an apples-to-apples QPS comparison you
> should run *both* libraries on the *same* GPU. The instructions below
> assume a single-H100 box; the script also runs end-to-end on an A5000
> / A100 / L4, you just won't be able to compare to FastPlaid's
> published table directly.

## What the script does

1. **Loads prepared BEIR NPZs** — the 8 datasets are encoded once with
   [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1)
   at dim=128 by `benchmarks/data/prepare_beir_datasets.py`. The same
   `[doc_tok, dim]` embeddings are then fed into **both** libraries —
   FastPlaid via `documents_embeddings=`, voyager-index via the existing
   `benchmarks/beir_benchmark.py::run_dataset` pipeline. This pins the
   embedding model so any quality / throughput delta is attributable to
   the indexing engine, not to the encoder.
2. **Runs each library on each dataset**, measuring:
   - **NDCG@10** against the BEIR test qrels
   - **Indexing time (s)** — wall-clock from the moment we hand
     embeddings to the library to the moment the index is searchable.
     voyager-index reports `cached` if the LEMUR routing artifacts are
     reused from a prior FP16 build (this is the documented codec-shared
     LEMUR behaviour from `benchmarks/beir_benchmark.py`); pass
     `--allow-missing` and remove `~/.cache/shard-bench/beir/<dataset>/`
     to force a clean rebuild.
   - **QPS** (single-client, sequential queries) and `p50` / `p95`
     latency. Single-client is the methodology used in the published
     FastPlaid table.
3. **Prints a FastPlaid-style table** (one row per `(dataset, library)`
   cell) and a per-library mean summary. Persists per-row JSONL +
   summary JSON for downstream charting.

## One-time setup

Anything that uses a workstation card you already own is fine for a
smoke run. For the apples-to-apples comparison against FastPlaid's
published numbers, a single H100 (80 GB or 40 GB SXM) is the target.

### Option A — local box with a CUDA GPU

```bash
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install -e ".[shard,gpu,native,dev]"
pip install fast-plaid pylate
```

### Option B — rent an H100 for the evening

The full BEIR-8 sweep at the v0.1.6 default takes ~30–45 min wall on an
H100 (~$2 of compute on RunPod / Lambda / Vast.ai at ~$3–4/hour). One
recipe that works:

```bash
# 1. Start a single-H100 pod (any provider; 80 GB HBM3 ideal, 40 GB OK)
#    Pick a base image with CUDA 12.x + Python 3.11 + PyTorch ≥ 2.4.
#    Examples:
#      - RunPod: "RunPod PyTorch 2.4.0 CUDA 12.1" community template
#      - Lambda: "PyTorch 2.4 / CUDA 12.1" stack
#      - Vast.ai: any "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" image

# 2. SSH in, then:
git clone https://github.com/ddickmann/voyager-index.git
cd voyager-index
pip install --upgrade pip
pip install -e ".[shard,gpu,native,dev]"
pip install fast-plaid pylate
```

Total install time on a fresh H100 pod: ~3–5 min.

## Step 1 — prepare the BEIR-8 corpus

This downloads the 8 BEIR datasets, encodes corpus + queries with
`GTE-ModernColBERT-v1`, and caches the per-token embeddings as NPZs
under `~/.cache/voyager-qa/beir/<dataset>.npz`. The encoding is GPU-bound
and takes ~10–15 min total on an H100 (~30 min on an A5000).

```bash
python benchmarks/data/prepare_beir_datasets.py \
    --datasets arguana fiqa nfcorpus quora scidocs scifact \
               trec-covid webis-touche2020 \
    --batch-size 64
```

The two large datasets (`trec-covid` 171k docs, `webis-touche2020`
382k docs) dominate the wall-time. If you only want a fast smoke first,
start with the small ones:

```bash
python benchmarks/data/prepare_beir_datasets.py --datasets nfcorpus scifact
```

## Step 2 — run the head-to-head

### Quick smoke (one small dataset, ~30 s on H100)

```bash
python benchmarks/fast_plaid_head_to_head.py --datasets nfcorpus
```

### Default — voyager-rroq158 vs FastPlaid on BEIR-8

```bash
python benchmarks/fast_plaid_head_to_head.py
```

### Full 3-way (fp16 baseline + rroq158 + FastPlaid)

This is the canonical table for a launch post — it shows the unquantized
ceiling, the v0.1.6 default codec, and the FastPlaid baseline side by
side on identical embeddings:

```bash
python benchmarks/fast_plaid_head_to_head.py \
    --libraries voyager_fp16 voyager_rroq158_gs128 fast_plaid
```

### Other useful invocations

```bash
# Just voyager (no FastPlaid installed): comparison vs your own fp16 ceiling
python benchmarks/fast_plaid_head_to_head.py \
    --libraries voyager_fp16 voyager_rroq158_gs128

# Compare the v0.1.6 SOTA flip vs the pre-flip baseline on the same matrix
python benchmarks/fast_plaid_head_to_head.py \
    --libraries voyager_rroq158_gs32 voyager_rroq158_gs128

# Multi-GPU box: pin FastPlaid to a specific device
python benchmarks/fast_plaid_head_to_head.py --device cuda:1
```

## Step 3 — read the table

The script prints a FastPlaid-shaped table (right-aligned columns,
QPS deltas in parentheses against the FastPlaid baseline) followed by
per-library mean rows. Sample output shape (numbers illustrative):

```
Dataset            Size    Library                   NDCG@10  Indexing Time (s)    Queries per second (QPS)
-----------------------------------------------------------------------------------------------------------
arguana            8674    voyager_rroq158_gs128      0.4612            24.18         842.31 (+443%)
                           fast_plaid                 0.4604             4.72         155.25
fiqa              57638    voyager_rroq158_gs128      0.4128           114.41         781.04 (+432%)
                           fast_plaid                 0.4108            12.62         146.62
...

Per-library means across run:
  Library                     NDCG@10 mean    QPS mean   QPS geomean   Total index (s)
  ----------------------------------------------------------------------------
  fast_plaid                       0.5067      131.40        128.70             65.41
  voyager_rroq158_gs128            0.5069      820.50        790.20            420.15
```

### How to read it

- **NDCG@10** — quality. Within ±0.005 between rroq158 gs=128 and
  fast_plaid means parity (we measured this in the v0.1.6 sweep on
  BEIR-6). The fp16 row, if present, is the unquantized ceiling — both
  quantized rows give up roughly the same amount of NDCG@10 to compress.
- **Indexing time (s)** — voyager-index's "cached" label means the
  LEMUR routing artifacts were reused from a prior FP16 build (codec-
  shared LEMUR is the documented production behaviour). For a clean
  measurement of fresh-index time, delete
  `~/.cache/shard-bench/beir/<dataset>/` first.
- **QPS** — single-client, sequential. Numbers in parentheses
  are `(qps - fast_plaid_qps) / fast_plaid_qps` — i.e. the speed-up vs
  the FastPlaid row in the same cell. **Negative values mean voyager
  is slower; positive values mean it is faster.**
- The summary block reports per-library means: `qps_mean` is arithmetic,
  `qps_geomean` is geometric (better aggregator across datasets with
  very different absolute throughput).

## Caveats and honest limits

1. **The embedding model is `GTE-ModernColBERT-v1`, not `colbertv2.0`.**
   FastPlaid's published table uses ColBERTv2 embeddings; rerunning
   FastPlaid here against `GTE-ModernColBERT-v1` lets us pin the model
   for the *comparison*, but the absolute NDCG@10 numbers are not
   directly comparable to the published FastPlaid row — only the
   *delta between libraries on the same embeddings* is. To run against
   ColBERTv2 instead, edit `MODEL_NAME` in
   `benchmarks/data/prepare_beir_datasets.py` and re-encode the NPZs.
2. **Single-client QPS.** Both libraries support batched query
   evaluation; the published FastPlaid numbers (and ours) are the
   single-client sequential path because that's what end-user RAG
   serving actually exercises. Concurrent / batched throughput is a
   separate measurement.
3. **`top_k=100` for both lanes.** FastPlaid's table doesn't pin
   `top_k`; we use 100 because that's voyager-index's documented
   benchmark default. Pass `--n-eval` to cap query count for a quick
   sanity run; do not pass it for the final table.
4. **FastPlaid keeps its index on disk.** The script writes to a
   `tempfile.mkdtemp()` per dataset; old runs can leave gigabytes of
   index files in `/tmp` if FastPlaid crashes mid-run. Run `rm -rf
   /tmp/fast_plaid_*` after a failed sweep to reclaim disk.
5. **`indexing_time_s` for voyager** is reconstructed from the
   pipeline's `indexing_docs_per_sec` field, which excludes encoding
   (we feed pre-computed embeddings on both lanes). FastPlaid's
   `indexing_time_s` excludes encoding too — both numbers are the
   "embeddings → searchable index" wall-clock.

## Sharing the result

The per-row JSONL written to
`reports/fast_plaid_head_to_head/results.jsonl` is the canonical
artifact. Cite it directly in any post / PR / blog:

```bash
cat reports/fast_plaid_head_to_head/results.jsonl | python -m json.tool
```

The summary JSON (`reports/fast_plaid_head_to_head/summary.json`)
includes the environment block (torch / CUDA / device name / library
versions) so the run is reproducible from the artifact alone.
