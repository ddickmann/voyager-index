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

## Groundedness Tracker (Beta) — real-world

Run on **RAGTruth** + **HaluEval**, per-stratum samples, A5000 batch 1.
Headline = `groundedness_v2` (calibrated reverse MaxSim + literal
guardrails + optional NLI peer with cross-encoder premise reranking,
atomic-claim decomposition, semantic-entropy consistency, and
structured-source verification).

| Lane                                     | Internal lex / sem / partial | RAGTruth macro F1 | HaluEval QA F1 | Latency p95 |
|------------------------------------------|-----------------------------:|------------------:|---------------:|------------:|
| Dense + literal only                     |           0.80 / 0.93 / 0.95 |              0.48 |           0.75 |      92 ms  |
| **+ NLI peer (reranker + atomic)**       |       **0.99 / 1.00 / 1.00** |              0.49 |       **0.90** |     102 ms  |
| **+ Semantic entropy (synthetic peers)** |       **0.98 / 1.00 / 1.00** |          **0.60** |           0.80 |     125 ms  |
| Pre-registered exit                      |    ≥ 0.80 / ≥ 0.70 / ≥ 0.65  |            ≥ 0.55 |         ≥ 0.75 |  ≤ 400 ms   |

Semantic-entropy + NLI lane hits **RAGTruth macro ≥ 0.55** and HaluEval QA
≥ 0.75 simultaneously; NLI-only lane sets the HaluEval QA headline at
**0.90** F1. Reports live under
`research/triangular_maxsim/reports/phase_j_no_nli.json`,
`phase_j_nli.json`, and `phase_j_nli_sem.json`. The full per-stratum
tables, harness, env vars, and reproduction guide live in
`research/triangular_maxsim/README.md`; the product framing, risk-band
policy, and Phase I structured-source verification are in the
[Groundedness Beta Guide](guides/groundedness-beta.md).

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
