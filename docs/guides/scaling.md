# Scaling Shard Collections

This guide covers the supported shard-first scaling path in `voyager-index`.
Use it when you need to choose between CPU exact serving, streamed GPU serving,
or a GPU-corpus deployment on a single machine.

## What scales

The shard engine scales by splitting the corpus into independent shard files and
then combining:

- LEMUR routing to choose a small candidate set
- exact MaxSim scoring on CPU or Triton on GPU
- optional ColBANDIT reranking
- optional dense+BM25 hybrid fusion for dense collections
- WAL-backed mutations and restart-safe reloads

The key scaling knobs are about candidate control and memory movement, not about
switching to a different engine.

## Pick a deployment shape

### CPU-first exact serving

Use this when:

- the machine has no CUDA device
- latency is acceptable with CPU MaxSim
- you want the simplest operational path

Recommended defaults:

- `router_device="cpu"`
- `quantization_mode="none"` for quality-first exact serving
- `n_full_scores` sized to the exact-stage budget you can afford

### Streamed GPU serving

Use this when:

- the corpus does not fit comfortably in VRAM
- you still want Triton MaxSim for the exact stage
- CPU to GPU transfer cost is acceptable

Recommended defaults:

- `transfer_mode="pinned"` or `transfer_mode="double_buffered"`
- `router_device="cpu"` unless the router itself becomes the bottleneck
- `quantization_mode="int8"`, `fp8`, or `roq4` only after validating recall

### GPU-corpus serving

Use this when:

- the full rerank frontier or corpus fits in VRAM
- you want the highest query throughput on one box
- you can dedicate GPU memory to retrieval

Recommended defaults:

- increase `gpu_corpus_rerank_topn`
- keep request payloads base64-encoded
- use multi-worker serving only if the machine still has enough headroom

## The most important knobs

### Collection layout

- `n_shards`: more shards reduce per-shard work but increase merge overhead
- `k_candidates`: router shortlist size before exact scoring
- `max_docs_exact`: hard cap on the exact stage
- `n_full_scores`: proxy shortlist sent to exact full scoring

### Compute and transfer

- `router_device`: where LEMUR runs
- `transfer_mode`: `pageable`, `pinned`, or `double_buffered`
- `quantization_mode`: `none`, `int8`, `fp8`, or `roq4`
- `gpu_corpus_rerank_topn`: how much of the frontier stays on GPU

### Quality levers

- `use_colbandit`: enable ColBANDIT in the shard rerank path
- `n_centroid_approx`: use centroid approximation before exact scoring
- `variable_length_strategy`: tune batching for uneven token counts

## Practical starting points

### Small to mid-size shard corpus

Start with:

```python
from voyager_index import Index

idx = Index(
    "/data/my-index",
    dim=128,
    engine="shard",
    n_shards=128,
    k_candidates=2000,
    max_docs_exact=8000,
    n_full_scores=2048,
    router_device="cpu",
)
```

### GPU-scored collection

Start with:

```python
idx = Index(
    "/data/my-index",
    dim=128,
    engine="shard",
    n_shards=256,
    k_candidates=2000,
    max_docs_exact=10000,
    n_full_scores=4096,
    transfer_mode="pinned",
    quantization_mode="int8",
    use_colbandit=True,
    router_device="cpu",
)
```

## Multi-worker guidance

For the reference API on a single machine:

- start with `WORKERS=4`
- move to `WORKERS=8` only after checking CPU saturation, IO pressure, and GPU memory contention
- keep every worker on the same `VOYAGER_INDEX_PATH`

Mutations remain durable because the server coordinates collection revisions and
WAL-backed writes across workers on the same host.

## Mutation and recovery posture

Scaling is not just query throughput. The production path also keeps:

- WAL-backed CRUD mutations
- collection reload and readiness reporting after restart
- worker-visible mutation results
- deletion and upsert visibility across service instances

If you are tuning aggressively, validate both latency and restart behavior on
your real collection shape.

## Benchmark honestly

Use `docs/benchmarks.md` for the public benchmark story and
`docs/guides/max-performance-reference-api.md` for deployment guidance.

When testing a new profile, capture at least:

- recall or ranking parity versus your exact baseline
- p50, p95, and p99 latency
- QPS at the target worker count
- CPU RAM and GPU VRAM use

The shard engine is the supported scaling path. Historical GEM/HNSW material has
been moved to `research/legacy/`.
