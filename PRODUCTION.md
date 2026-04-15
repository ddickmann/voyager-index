# Production Notes

This page is intentionally short. `voyager-index` now has a shipped
shard-first production path, so this document is a deployment checklist rather
than a migration RFC.

## Production Path

Use:

- `Index(..., engine="shard")` for local SDK usage
- `voyager-index-server` for HTTP deployments
- `docs/guides/shard-engine.md` for engine internals and admin operations
- `docs/guides/max-performance-reference-api.md` for CPU/GPU tuning and worker guidance

## What Is Production-Wired

- shard collections
- CRUD and upsert semantics
- WAL-backed mutation logging
- checkpoint and restart recovery
- multi-worker single-host serving
- base64 vector transport
- ColBANDIT in the shard scoring path
- Triton MaxSim on CUDA
- `int8`, `fp8`, and `roq4` shard scoring modes
- dense BM25 hybrid search with `rrf` or `tabu`
- multimodal preprocessing and collection flows
- optional Latence graph augmentation as an additive premium lane when `voyager-index[latence-graph]` is installed

The production story is:

- `shard` is the high-performance HTTP lane for vector-first retrieval
- GPU and CPU performance claims in the README come from the shard benchmark harness
- the optional graph lane augments post-retrieval results and provenance; it does not replace the shard hot path

Operational sequence:

- first-stage retrieval stays shard-first, or dense-first on routes that expose `query_text`
- BM25 runs in parallel when the route supports it
- the graph policy decides whether the optional Latence graph lane runs after first-stage retrieval
- solver or result packing is the final arbitration layer

## Recommended Deploy Shapes

### CPU-first

- install `voyager-index[server,shard]`
- keep `router_device="cpu"`
- use exact scoring
- scale QPS with `WORKERS=4` or `WORKERS=8` on a single host

### GPU streamed

- install `voyager-index[server,shard,gpu]`
- use `transfer_mode="pinned"` or `transfer_mode="double_buffered"`
- enable `quantization_mode` when you want GPU-side acceleration
- for `compression="roq4"`, the shard assets persist `roq_codes` and `roq_meta` alongside offsets and IDs

### GPU corpus

- install `voyager-index[server,shard,gpu]`
- preload the corpus into VRAM when it fits
- report GPU-corpus numbers separately from streamed numbers

### Optional Latence graph plane

- install `voyager-index[server,shard,latence-graph]`
- keep `graph_mode="off"` as the safe baseline and enable `auto` or `force` per workflow
- graph data comes from Latence graph data derived from the customer corpus and linked back to collection targets through the sidecar sync path
- use `query_payload` to steer graph policy on vector-only HTTP routes such as shard search
- inspect `GET /collections/{name}/info` for `graph_health`, `graph_sync_status`, and freshness timestamps
- rely on `/ready` to surface degraded or failed graph sync states without taking down the base retrieval lane

## Minimal Server Start

```bash
export HOST=0.0.0.0
export PORT=8080
export WORKERS=4
export VOYAGER_INDEX_PATH=/data/voyager-index
voyager-index-server
```

## Operational Checklist

- keep every worker on the same `VOYAGER_INDEX_PATH`
- put TLS, auth, and request policy in front of the reference API
- use `/health` for liveness and `/ready` for degraded-state checks
- prefer base64 request payloads for dense and multivector traffic
- benchmark recall and latency together before changing `k_candidates` or `n_full_scores`
- treat graph benchmark evidence separately from shard performance evidence when publishing claims

## Where To Go Next

- [Shard Engine Guide](docs/guides/shard-engine.md)
- [Max-Performance Reference API Guide](docs/guides/max-performance-reference-api.md)
- [Reference API Tutorial](docs/reference_api_tutorial.md)
- [Benchmarks](docs/benchmarks.md)
