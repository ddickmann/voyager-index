# Benchmarks

This page documents how `voyager-index` benchmark claims are framed today.

Two rules matter more than any single latency number:

- every public throughput claim must be paired with recall
- GPU corpus and streamed GPU modes are reported separately

## Benchmark Layers

### 1. OSS smoke benchmark

`benchmarks/oss_reference_benchmark.py` is the small, reproducible benchmark
used for package and API sanity checks.

It exercises:

- MaxSim kernels
- reference API ingest
- reference API search
- multimodal search

Run it with:

```bash
python benchmarks/oss_reference_benchmark.py --device cpu --points 16 --top-k 3
python benchmarks/oss_reference_benchmark.py --device cuda --points 16 --top-k 3
```

This benchmark is intentionally small. It is for regression detection, not for
headline product comparisons.

### 2. Product benchmark

The product benchmark is the shard retrieval benchmark on the 100k corpus:

- same embeddings across all systems
- same `top_k`
- same recall target
- same warmup policy
- same hardware per comparison table
- separate reporting for streamed GPU and GPU-corpus modes

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
