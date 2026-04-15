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
