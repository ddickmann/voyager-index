# voyager-index Docs

`voyager-index` is a shard-first late-interaction retrieval system for
multivector text and multimodal search.

The public story is:

- shard engine as the production retrieval path
- CPU fallback and GPU acceleration from the same collection format
- Triton MaxSim, ColBANDIT, and quantization as layered performance options
- CRUD, WAL, checkpoint, and recovery as standard features
- base64 vector transport as the preferred HTTP contract
- BM25 hybrid search with `rrf` or `tabu`
- groundedness / hallucination detection as a post-generation Beta feature
- optional Latence graph augmentation as an additive premium lane

## Start Here

- [Quickstart](getting-started/quickstart.md): first local shard index
- [Installation](getting-started/installation.md): package extras and source install
- [Python API Reference](api/python.md): `Index`, `IndexBuilder`, transport helpers, and public classes
- [Reference API Tutorial](reference_api_tutorial.md): first HTTP collections and queries
- [Groundedness Beta Guide](guides/groundedness-beta.md): post-generation groundedness, evidence heatmaps, and Beta limits
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

## Product Shape

| Surface | What it does |
|---|---|
| `Index(engine="shard")` | Local shard collections with CRUD and search |
| `voyager-index-server` | Reference HTTP API with dense, late-interaction, multimodal, and shard collections |
| `encode_vector_payload()` | Preferred base64 serializer for dense and multivector requests |
| `SearchPipeline` | In-process dense + BM25 fusion |
| `latence_solver` | Optional Tabu Search solver for `tabu` refinement and `/reference/optimize` |
| `POST /collections/{name}/groundedness` | Beta groundedness / hallucination detection over final `chunk_ids` or `raw_context` |
| `LatenceGraphSidecar` | Optional premium graph plane for additive rescue, provenance, and freshness-aware graph metadata |

## Design Principles

- Late interaction first
- Multimodal as a continuation of the same retrieval story
- Honest CPU and GPU modes
- Simple deployment over single-host sprawl
- Benchmark claims tied to recall and methodology
- Optional premium planes must be additive and failure-isolated
