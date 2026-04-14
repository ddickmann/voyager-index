# Benchmarks

This repository includes a small reproducible benchmark harness for the OSS
foundation surface.

Run the benchmark script:

```bash
python benchmarks/oss_reference_benchmark.py --device cpu
```

If a compatible GPU is available:

```bash
python benchmarks/oss_reference_benchmark.py --device cuda
```

The script currently measures:

- `fast_colbert_scores` on deterministic synthetic inputs
- dense ingest throughput for the reference API over a temporary local collection
- a reference API dense-search roundtrip over a temporary local collection
- multimodal search latency through `ColPaliEngine` using chunked local storage
- the small public OSS surface that should stay stable across local release builds

Useful flags:

```bash
python benchmarks/oss_reference_benchmark.py --device cpu --points 128 --top-k 10
```

The output is JSON so it can be saved, diffed, or fed into a larger reporting
pipeline. This harness is intentionally small and deterministic; it is a smoke
benchmark for regression detection rather than a substitute for full real-model
quality and recall evaluation.

Use this harness for quick regression checks. For larger validation bundles,
screening promotion evidence, and the broader release audit posture, see
`internal/validation/README.md`, `internal/memos/SCREENING_PROMOTION_DECISION_MEMO.md`, and
`scripts/full_feature_validation.py`.

## GEM-Lite Screening Checks

For multimodal GEM-lite experiments, the meaningful evidence comes from the
real-model `ColPaliEngine` screening path, not from the tiny synthetic smoke
benchmark above.

Use `scripts/full_feature_validation.py` plus a direct A/B comparison between:

- the legacy `prototype_hnsw` sidecar behavior
- the current GEM-lite `prototype_hnsw` sidecar behavior

Important rendering note:

- mixed source corpora such as `tmp_data` need the document-rendering
  dependencies installed (`PyMuPDF`, `python-docx`, `openpyxl`, `Pillow`)
- otherwise the validator may collapse to a tiny image-only subset and understate
  multimodal screening behavior

The current GEM-lite path is intentionally not a GEM-native index. It adopts a
subset of concepts from
[GEM: A Native Graph-based Index for Multi-Vector Retrieval](https://arxiv.org/abs/2603.20336)
inside the prototype screening sidecar while keeping exact MaxSim reranking
unchanged.

On the larger fully rendered `tmp_data` run used in this environment (`547`
page records, `4` benchmark queries), GEM-lite was active but neutral versus
the legacy prototype sidecar:

- legacy prototype sidecar: `recall_at_k=0.60`, `elapsed_ms≈2674.6`,
  `speedup_vs_full_precision≈1.16`
- GEM-lite prototype sidecar: `recall_at_k=0.60`, `elapsed_ms≈2674.9`,
  `speedup_vs_full_precision≈1.15`

Treat that as evidence that the GEM-inspired metadata path is safe to keep and
worth iterating on, not as proof that the OSS repo already exposes a native GEM
graph or a promoted new default.
