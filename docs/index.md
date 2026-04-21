# colsearch Docs

`colsearch` is a shard-first late-interaction retrieval system for
multivector text and multimodal search.

The public story is:

- shard engine as the production retrieval path
- CPU fallback and GPU acceleration from the same collection format
- Triton MaxSim, ColBANDIT, and quantization as layered performance options
- CRUD, WAL, checkpoint, and recovery as standard features
- base64 vector transport as the preferred HTTP contract
- BM25 hybrid search with `rrf` or `tabu`
- optional Latence graph augmentation as an additive premium lane
- optional Latence Trace groundedness sidecar as a separate post-generation hallucination tracker

## Start Here

- [Quickstart](getting-started/quickstart.md): first local shard index
- [Installation](getting-started/installation.md): package extras and source install
- [Python API Reference](api/python.md): `Index`, `IndexBuilder`, transport helpers, and public classes
- [Reference API Tutorial](reference_api_tutorial.md): first HTTP collections and queries
- [Groundedness Sidecar Guide](guides/groundedness-sidecar.md): the Latence Trace post-generation hallucination tracker and its integration shape
- [Max-Performance Reference API Guide](guides/max-performance-reference-api.md): worker scaling, base64, GPU modes
- [Shard Engine Guide](guides/shard-engine.md): routing, scoring, durability, and admin endpoints
- [Latence Graph Sidecar](guides/latence-graph-sidecar.md): optional premium graph lane, policy, provenance, and health surfaces
- [Enterprise Control Plane Boundary](guides/control-plane.md): what the OSS data plane does and what the enterprise control plane must own
- [Benchmarks](benchmarks.md): methodology, caveats, and 100k placeholder table

## Who This Fits

This repo is a good fit if you are shipping:

- ColBERT-style token retrieval
- ColPali-style page or patch retrieval
- a single-host service that needs multi-worker QPS scaling
- a retrieval stack where exact MaxSim quality matters more than generic vector DB breadth

It is probably **not** the first choice if you need:

- a large distributed control plane across many nodes
- purely dense ANN retrieval at extreme scale
- a hosted multi-tenant SaaS search platform

## What Makes It Different

### No mandatory graph dependency

`colsearch` uses proxy routing plus exact MaxSim reranking without
requiring a graph build step in the OSS serving path. When installed, the
optional Latence graph sidecar is invoked after first-stage retrieval and
merged additively. That keeps the system simpler to operate while preserving
a premium graph lane for teams that want it.

### Rust + Triton hot paths

The CPU path is a native Rust extension (`latence_shard_engine`) with
memory-mapped shards, fused MaxSim, SIMD acceleration, and GIL-free
execution. The GPU path uses Triton kernels for exact and quantized scoring
with variable-length document scheduling. Both paths share the same
collection format and retrieval contract.

### Research-backed features in the serving path

LEMUR routing, ColBANDIT query-time pruning, ROQ rotational quantization,
and budget-aware context optimization are integrated into the shipped system
rather than isolated in research notebooks.

### Operational features, not just benchmarking

CRUD, WAL, checkpoint, crash recovery, payload metadata, scroll, and
retrieve are included because retrieval systems in production need
operational discipline — not just benchmark wins.

### Multimodal native

The same serving stack supports text token embeddings (ColBERT) and image
patch embeddings (ColPali / ColQwen), with preprocessing for PDF, DOCX,
XLSX, and image inputs.

## Product Shape

| Surface | What it does |
|---|---|
| `Index(engine="shard")` | Local shard collections with CRUD and search |
| `colsearch-server` | Reference HTTP API with dense, late-interaction, multimodal, and shard collections |
| `encode_vector_payload()` | Preferred base64 serializer for dense and multivector requests |
| `SearchPipeline` | In-process dense + BM25 fusion |
| `latence_solver` | Optional Tabu Search solver for `tabu` refinement and `/reference/optimize` |
| `LatenceGraphSidecar` | Optional premium graph plane for additive rescue, provenance, and freshness-aware graph metadata |
| Latence Trace sidecar | Optional post-generation groundedness tracker that reuses `chunk_ids` from this index |

## Design Principles

- Late interaction first
- Multimodal as a continuation of the same retrieval story
- Honest CPU and GPU modes
- Simple deployment over single-host sprawl
- Benchmark claims tied to recall and methodology
- Optional premium planes must be additive and failure-isolated
