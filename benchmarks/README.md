# Supported Benchmarks

This directory contains the supported shard-first benchmark surface.

Public entrypoints:

- `oss_reference_benchmark.py`: small deterministic regression benchmark
- `latency_opportunity_bench.py`: targeted latency experiment harness
- `centroid_approx_bench.py`: centroid-approximation experiment harness
- `beir_benchmark.py`: 6-dataset BEIR sweep across all built-in codecs
  (fp16 / int8 / rroq158 / rroq4_riem) on GPU + 8-worker CPU lanes
- `beir_2026q2_full_sweep.py`: the production-validation 2026-Q2 BEIR
  sweep used to gate the rroq158 SOTA flip (see
  [`docs/benchmarks.md`](../docs/benchmarks.md))
- `fast_plaid_head_to_head.py`: voyager-index vs
  [FastPlaid](https://github.com/lightonai/fast-plaid) on FastPlaid's
  published BEIR-8 matrix, using identical per-token embeddings so
  the comparison varies only the indexing/scoring engine — see the
  full tutorial at
  [`docs/benchmarks/fast-plaid-head-to-head.md`](../docs/benchmarks/fast-plaid-head-to-head.md)
- `shard_bench/`: shard benchmark helpers and reports

Historical GEM/HNSW benchmark suites and result bundles are no longer
published with this repository.
