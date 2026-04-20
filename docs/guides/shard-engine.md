# Shard Engine Guide

The shard engine is a LEMUR-routed late-interaction retrieval backend built for
simplicity and high-throughput exact or quantized scoring. It does not require
graph construction in its hot path, relying instead on a learned routing MLP
(LEMUR) to reduce multi-vector candidate generation to single-vector MIPS, then
scoring candidates with Triton on CUDA or exact/full-precision fallback paths
on CPU. When the optional Latence graph lane is installed, graph augmentation
still happens after first-stage shard retrieval.

## Who this is for

Use the shard engine when you want:

- the mainline `voyager-index` production path
- a simpler mental model than graph-native retrieval
- CPU fallback and GPU acceleration from the same collection layout
- durable CRUD, WAL, checkpoint, and recovery
- a runtime built on PyTorch plus optional Triton/native deps rather than a pure-Python serving path

Not the best fit when you only need pooled dense vectors and do not care about
late interaction.

## Architecture

```
Query tokens
  → LEMUR MLP → latent features
  → FAISS ANN index → candidate doc IDs
  → GPU gather (from GPU-resident corpus or shard fetch)
  → optional ColBANDIT pruning
  → Triton (GPU) or Rust SIMD (CPU) scoring: RROQ-1.58 (default) / FP16 / INT8 / FP8 / ROQ4
  → top-K results
```

When the optional planes are enabled around shard, the broader production
sequence is:

```text
query
  -> shard first-stage retrieval
  -> optional BM25 in parallel where the route supports query_text
  -> exact or quantized MaxSim
  -> optional graph policy check
  -> optional Latence graph augmentation
  -> solver / result packing
```

Key properties:

- **No graph construction**: build time is dominated by LEMUR MLP training
  and FAISS index construction (seconds, not minutes); optional Latence graph
  augmentation is layered on later and does not change the shard hot path
- **GPU-resident fast path**: when corpus fits in VRAM, scoring is a single
  `D[candidate_ids]` gather followed by a fused MaxSim kernel launch
- **Disk-backed fallback**: for large corpora, safetensors-backed shards
  are fetched on-demand with pinned-memory pipelining
- **Quantized serving**: request or collection level selection of `rroq158`
  (default — Riemannian 1.58-bit, K=8192, fused on both GPU Triton and CPU
  Rust SIMD), exact, `int8`, `fp8`, or `roq4` scoring, with truthful
  fallback behavior when those kernels are unavailable. The CPU `rroq158`
  lane is wired through `latence_shard_engine.rroq158_score_batch`
  (hardware popcount + AVX2/BMI2/FMA + cached rayon thread pool).
- **Production ColBANDIT path**: query-time pruning is wired into the real
  shard serving flow instead of being a side experiment
- **Full CRUD**: insert, delete, upsert via WAL + memtable, identical
  durability guarantees to the rest of the shipped product surface

## Quick Start

### Python API

```python
from voyager_index import Index

# Build
idx = Index("my_shard_index", dim=128, engine="shard")
idx.add(embeddings, ids=list(range(len(embeddings))),
        payloads=[{"title": f"doc_{i}"} for i in range(len(embeddings))])

# Search
results = idx.search(query_vectors, k=10)
for r in results:
    print(f"  doc {r.doc_id}  score={r.score:.4f}")

# CRUD
idx.delete([0, 1])
idx.upsert(vectors, ids=[0, 1], payloads=[{"title": "updated"}, ...])

# Persistence
idx.close()
idx2 = Index("my_shard_index", dim=128, engine="shard")  # auto-loads
```

### IndexBuilder

```python
from voyager_index import IndexBuilder

idx = (IndexBuilder("my_index", dim=128)
       .with_shard(n_shards=256, k_candidates=2000)
       .build())
```

### HTTP API

```bash
# Create shard collection
curl -X POST http://localhost:8080/collections/my_col \
  -H "Content-Type: application/json" \
  -d '{
    "dimension": 128,
    "kind": "shard",
    "n_shards": 256,
    "compression": "rroq158",
    "rroq158_k": 8192,
    "rroq158_group_size": 128,
    "rroq158_seed": 42,
    "quantization_mode": "fp8",
    "transfer_mode": "pinned",
    "router_device": "cpu",
    "use_colbandit": true
  }'

# Add points
curl -X POST http://localhost:8080/collections/my_col/points \
  -H "Content-Type: application/json" \
  -d '{"points": [{"id": "doc_1", "vectors": {"encoding":"float16","shape":[32,128],"dtype":"float16","data_b64":"..."}, "payload": {"title": "Doc 1"}}]}'

# Search
curl -X POST http://localhost:8080/collections/my_col/search \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"encoding":"float16","shape":[32,128],"dtype":"float16","data_b64":"..."}, "top_k": 10, "quantization_mode": "fp8"}'
```

Optional graph-aware shard search uses the same endpoint:

```bash
curl -X POST http://localhost:8080/collections/my_col/search \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {"encoding":"float16","shape":[32,128],"dtype":"float16","data_b64":"..."},
    "top_k": 10,
    "quantization_mode": "fp8",
    "graph_mode": "auto",
    "graph_local_budget": 4,
    "graph_community_budget": 4,
    "graph_evidence_budget": 8,
    "graph_explain": true,
    "query_payload": {
      "ontology_terms": ["Service C", "Export Control"],
      "workflow_type": "compliance"
    }
  }'
```

## Configuration

### ShardEngineConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_shards` | 256 | Number of storage shards |
| `dim` | 128 | Embedding dimension |
| `compression` | `rroq158` | Storage compression. Default is `rroq158` (Riemannian 1.58-bit, K=8192, **group_size=128 SOTA**, GPU+CPU). Other options: `fp16`, `int8`, `roq4`, `rroq4_riem` (Riemannian 4-bit asymmetric — safe-fallback lane for zero-regression workloads). See [`docs/guides/quantization-tuning.md`](quantization-tuning.md). |
| `rroq158_k` | 8192 | rroq158 spherical k-means centroid count (must be a power of two ≥ effective `group_size`) |
| `rroq158_seed` | 42 | rroq158 FWHT rotator + k-means initialisation seed |
| `rroq158_group_size` | **128** | rroq158 ternary group size — SOTA default (one scale per token at dim=128). Must be a positive multiple of 32. The encoder transparently steps down to gs=64 / gs=32 with a log warning for dims that aren't a multiple of the requested value (dim=64 / 96 / 160 still build cleanly). Set to `64` for the safest cross-dataset choice (e.g. arguana-class corpora). See [`docs/guides/quantization-tuning.md`](quantization-tuning.md). |
| `rroq4_riem_k` | 8192 | rroq4_riem spherical k-means centroid count (must be a power of two ≥ `group_size`) |
| `rroq4_riem_seed` | 42 | rroq4_riem FWHT rotator + k-means initialisation seed |
| `rroq4_riem_group_size` | 32 | rroq4_riem 4-bit asymmetric residual group size (positive even integer that divides `dim`) |
| `k_candidates` | 2000 | LEMUR candidate count per query |
| `use_colbandit` | `false` | Enable ColBANDIT query-time pruning |
| `lemur_epochs` | 10 | LEMUR MLP training epochs |
| `transfer_mode` | `pinned` | CPU→GPU transfer strategy |
| `quantization_mode` | exact | Scoring mode (`none`, `int8`, `fp8`, `roq4`) |
| `router_device` | `cpu` | Device used by the LEMUR router |
| `lemur_search_k_cap` | 2048 | LEMUR search cap before routing/scoring |
| `max_docs_exact` | 10000 | Exact-stage document budget |
| `n_full_scores` | 4096 | Proxy shortlist size before exact full scoring |
| `pinned_pool_buffers` | 3 | Pinned-memory transfer buffer pool size |
| `pinned_buffer_max_tokens` | 50000 | Max tokens per pinned transfer buffer |
| `gpu_corpus_rerank_topn` | 16 | GPU rerank frontier size |
| `n_centroid_approx` | 0 | Optional centroid-approx candidate stage |
| `variable_length_strategy` | `bucketed` | Variable-length exact scheduling |
| `seed` | 42 | Random seed for reproducibility |

### Tuning Tips

- **`k_candidates`**: higher = better recall, slower latency. Start with 2000
  and reduce if latency is too high at your corpus size.
- **`n_shards`**: should be roughly `sqrt(n_docs / 100)` for balanced
  shard sizes. Default of 256 works well up to ~500K docs.
- **`compression` vs `quantization_mode`**: compression controls stored shard
  representation; quantization controls the active scoring kernel. The
  default `rroq158` codec is wired on both lanes (GPU Triton fused kernel,
  CPU Rust SIMD), so the same collection runs on either device with no
  per-device override.
- **`n_full_scores`**: reduce it to trim exact work after proxy pruning; raise it
  only when recall audits show the shortlist is too aggressive.
- **Pinned transfer knobs**: `pinned_pool_buffers` and
  `pinned_buffer_max_tokens` matter only for CPU->GPU fetch pipelines, not the
  pure CPU exact path.
- **RROQ-1.58 (default, `group_size=128` SOTA)**: **~6.4× smaller storage
  than fp16** (~40 B / token vs 256 B / token at dim=128 — down from
  ~46 B at the previous gs=32 default). On the full BEIR-6 production
  sweep at the gs=128 SOTA default: **avg −1.37 pt NDCG@10 vs fp16**
  (slightly improved from −1.43 pt at gs=32) with **avg −0.62 pt R@100**,
  avg GPU p95 **1.20× fp16** (at the retention budget). CPU p95 is
  currently **~3.0× slower than fp16** at the production batch shape
  (improved from 3.15× at gs=32 and from ~7.9× pre-fix after the
  post-Phase-7 wrapper + kernel refresh; see
  [docs/benchmarks.md](../benchmarks.md#honest-cpu-latency-caveat-post-phase-7-followup))
  — the storage win is the primary value here. The gs=128 flip itself is
  Pareto-equal to gs=32 in quality (per-dataset Δ vs gs=32 = +0.0006 NDCG@10
  averaged across BEIR-6) and **lower or equal CPU p95 on every dataset**
  (largest win nfcorpus −22%; only +/−2% bump fiqa +2% — one scale per
  token cuts kernel scale loads by 4×). For dims that aren't a
  multiple of 128 (dim=64 / 96 / 160) the encoder transparently steps
  down to gs=64 / gs=32 with a log warning. Override with
  `Rroq158Config(group_size=64)` for arguana-class corpora where the
  marginal −0.0024 NDCG@10 vs gs=32 matters more than the 13% storage
  win (see [`docs/guides/quantization-tuning.md`](quantization-tuning.md)). Brute-force codec-fidelity overlap with fp16
  across BEIR-6 (`reports/beir_2026q2/topk_overlap.jsonl`):
  **avg ~79% top-10 / ~80% top-100** (range 73–83% top-10), with
  R@100 within −2.1 pt of FP16 on every dataset. Importantly, top-K
  overlap is roughly *flat across K* (e.g. quora drops from 72.9% at
  K=10 to 72.1% at K=100), so widening the serve window is **not** a
  reliable rescue mechanism for displaced top-K docs — the
  displacement is *out of the candidate set*, not within it. R@100
  still recovers because rroq158 admits the labeled relevant docs;
  the displacement is among the non-relevant tail. Override with
  `compression="fp16"` only if you need parity with an older
  deployment or to disambiguate a quality-regression hypothesis. For
  workloads requiring exact top-10 rank fidelity vs FP16, opt into
  `rroq4_riem` (below — avg ~96% top-10 overlap) or use rroq158 with
  an FP16 rerank on the shortlist
  (`benchmarks/diag_rroq158_rescue.py`).
- **RROQ-4 Riemannian (no-quality-loss lane)**: ~3× smaller than fp16
  (~88 B / token), avg ΔNDCG@10 = +0.02 pt vs fp16 (max ±0.05 pt across
  BEIR-6) — fully wired on both GPU (Triton fused kernel
  `roq_maxsim_rroq4_riem`) and CPU (Rust SIMD kernel
  `latence_shard_engine.rroq4_riem_score_batch`, AVX2/FMA + cached rayon
  pool). Still slower than fp16 in absolute latency on the BEIR sweep
  (avg 5.0× on GPU, **avg ~7.2× on CPU** at the production batch
  shape, down from ~12.7× pre-fix after the post-Phase-7 CPU lane
  refresh). The CPU lane shipped the same four optimisations as
  rroq158 (zero-copy `_to_np`, BLAS thread cap around encode + score,
  numpy fancy-indexing in the harness, plus the pre-existing
  nibble-unpack amortisation in `fused_rroq4_riem.rs`). The win is
  still **storage with zero quality regression**, not throughput. Set
  `compression="rroq4_riem"` for workloads that reject any quality
  regression on hard datasets.
- **ROQ 4-bit**: still available for ~4× compression with the asymmetric
  Triton kernel. Set `compression="roq4"` and optionally
  `quantization_mode="roq4"` on CUDA.
- **ColBANDIT**: enable it when you want pruning in the production shard path;
  disable it when measuring exact latency/quality baselines.

## CRUD and Durability

The shard engine uses a WAL + memtable pattern:

1. **Write-Ahead Log**: every insert/delete/upsert/update_payload is logged
   to a binary WAL before being applied to the in-memory memtable.
   The `UPDATE_PAYLOAD` op records payload-only changes without vectors.
2. **MemTable**: in-memory buffer for mutable documents, searched alongside
   sealed shards via score merging
3. **Flush / Checkpoint**: `flush()` syncs the WAL and checkpoints memtable
   state. The memtable is retained for crash safety; real L0-to-sealed merge is
   still planned.
4. **Crash Recovery**: on `load()`, checkpoint state is restored first and only
   WAL entries after the saved logical offset are replayed into a fresh
   memtable.

## Admin Endpoints

All admin endpoints require a `shard` collection:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/compact` | POST | Sync WAL and checkpoint memtable state (no sealed compaction yet) |
| `/collections/{name}/shards` | GET | List all shards with doc counts and token stats |
| `/collections/{name}/shards/{id}` | GET | Detail for a specific shard |
| `/collections/{name}/wal/status` | GET | WAL entry count, memtable size, tombstone count |
| `/collections/{name}/checkpoint` | POST | Force a WAL sync + memtable checkpoint |
| `/collections/{name}/scroll` | POST | Paginated iteration over document IDs (shard collections only) |
| `/collections/{name}/retrieve` | POST | Retrieve specific documents by ID (shard collections only) |
| `/collections/{name}/search/batch` | POST | Batch search over multiple queries |

## Hybrid Search

The shard engine integrates with `HybridSearchManager` for BM25 + dense
fusion in programmatic flows:

```python
from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager

hybrid = HybridSearchManager(
    shard_path=Path("my_hybrid"),
    dim=128,
    dense_engine="shard",
)
```

Dense collections expose `dense_hybrid_mode="rrf"` and `"tabu"` directly on the
HTTP search request. Shard collection HTTP search remains vector-only and does
not accept `query_text`; use `HybridSearchManager` if you want BM25 fusion with
the shard backend in-process.

## Optional Latence graph sidecar on shard

Shard collections are still the mainline high-performance lane when the graph
sidecar is enabled:

- first stage remains routed shard retrieval with Triton or fused Rust scoring
- graph augmentation happens only after shard retrieval returns candidates
- merge behavior is additive, so the base shard order stays intact
- graph data comes from Latence graph data derived from the indexed corpus and
  synchronized as target-linked graph contracts
- shard HTTP search remains vector-only, so graph policy should be steered with
  `query_payload` instead of `query_text`

Use `GET /collections/{name}/info` and `/ready` to inspect graph health and sync
status when the optional premium lane is installed. The deeper graph data and
provenance story lives in the [Latence Graph Sidecar Guide](latence-graph-sidecar.md).

## GPU Memory and Auto-Tiering

The shard engine automatically detects available GPU memory:

- **GPU-resident** (corpus fits in VRAM): entire corpus is pre-loaded as a
  contiguous FP16 tensor. Scoring is zero-copy gather + MaxSim kernel.
- **Disk-backed** (corpus exceeds VRAM): safetensors shards are fetched
  on-demand with configurable pinned-memory buffering. Suitable for 1M+ docs.
- **CPU-only host**: when CUDA is unavailable, the same shard collection stays
  searchable with the CPU path and the same routed retrieval / ColBANDIT
  controls. The default `rroq158` codec runs natively on CPU via the Rust
  SIMD kernel (`latence_shard_engine.rroq158_score_batch`) — there is no
  silent fallback to fp16. Other Triton quantization modes (`int8`, `fp8`)
  fall back to full-precision scoring on CPU; `roq4` exact scoring remains
  CUDA-only.

Memory formula for GPU-resident mode:
```
VRAM = n_docs × max_tokens × dim × 2 bytes (FP16)
```

Example: 100K docs × 128 tokens × 128 dim = ~3.3 GB.

## Production Notes

- shard is the documented production retrieval path in this repo
- HTTP shard search is vector-only
- dense BM25 hybrid remains on `dense` collections over HTTP
- `voyager-index[shard,shard-native]` or `voyager-index[native]` enables the Rust shard CPU fast-path
- `latence_solver` remains the optional native add-on for `tabu`
- `voyager-index[latence-graph]` adds the optional premium graph lane on top of the shard production path
