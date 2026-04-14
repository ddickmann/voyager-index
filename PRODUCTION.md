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

### GPU corpus

- install `voyager-index[server,shard,gpu]`
- preload the corpus into VRAM when it fits
- report GPU-corpus numbers separately from streamed numbers

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

## Where To Go Next

- [Shard Engine Guide](docs/guides/shard-engine.md)
- [Max-Performance Reference API Guide](docs/guides/max-performance-reference-api.md)
- [Reference API Tutorial](docs/reference_api_tutorial.md)
- [Benchmarks](docs/benchmarks.md)
