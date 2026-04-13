# Shard Engine Guide

The shard engine is a LEMUR-routed late-interaction retrieval backend built for
simplicity and GPU-accelerated exact scoring. It avoids graph construction
entirely, relying instead on a learned routing MLP (LEMUR) to reduce
multi-vector candidate generation to single-vector MIPS, then scoring
candidates with the Triton MaxSim kernel.

## Architecture

```
Query tokens
  → LEMUR MLP → latent features
  → FAISS ANN index → candidate doc IDs
  → GPU gather (from GPU-resident corpus or shard fetch)
  → Triton MaxSim (FP16 / ROQ 4-bit)
  → optional Col-Bandit pruning
  → top-K results
```

Key properties:

- **No graph construction**: build time is dominated by LEMUR MLP training
  and FAISS index construction (seconds, not minutes)
- **GPU-resident fast path**: when corpus fits in VRAM, scoring is a single
  `D[candidate_ids]` gather followed by a fused MaxSim kernel launch
- **Disk-backed fallback**: for large corpora, safetensors-backed shards
  are fetched on-demand with pinned-memory pipelining
- **Full CRUD**: insert, delete, upsert via WAL + memtable, identical
  durability model to the GEM engine

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
  -d '{"dimension": 128, "kind": "shard", "n_shards": 256}'

# Add points
curl -X POST http://localhost:8080/collections/my_col/points \
  -H "Content-Type: application/json" \
  -d '{"points": [{"id": "doc_1", "vectors": [[0.1, ...], ...], "payload": {"title": "Doc 1"}}]}'

# Search
curl -X POST http://localhost:8080/collections/my_col/search \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[0.1, ...], ...], "top_k": 10}'
```

## Configuration

### ShardEngineConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_shards` | 256 | Number of storage shards |
| `dim` | 128 | Embedding dimension |
| `compression` | `fp16` | Storage compression (`fp16`, `roq4`) |
| `k_candidates` | 2000 | LEMUR candidate count per query |
| `use_colbandit` | `false` | Enable Col-Bandit query-time pruning |
| `lemur_epochs` | 10 | LEMUR MLP training epochs |
| `transfer_mode` | `pinned` | CPU→GPU transfer strategy |
| `seed` | 42 | Random seed for reproducibility |

### Tuning Tips

- **`k_candidates`**: higher = better recall, slower latency. Start with 2000
  and reduce if latency is too high at your corpus size.
- **`n_shards`**: should be roughly `sqrt(n_docs / 100)` for balanced
  shard sizes. Default of 256 works well up to ~500K docs.
- **ROQ 4-bit**: enables ~4x bandwidth reduction for disk-backed corpora.
  Set `compression="roq4"` via `ShardEngineConfig`.

## CRUD and Durability

The shard engine uses the same WAL + memtable pattern as the GEM engine:

1. **Write-Ahead Log**: every insert/delete/upsert/update_payload is logged
   to a binary WAL before being applied to the in-memory memtable.
   The `UPDATE_PAYLOAD` op records payload-only changes without vectors.
2. **MemTable**: in-memory buffer for mutable documents, searched alongside
   sealed shards via score merging
3. **Flush / Compaction**: `flush()` syncs the WAL to disk. The memtable is
   retained for crash safety (real L0-to-sealed merge is planned but not yet
   implemented). The `CompactionScheduler` runs WAL checkpoints periodically.
4. **Crash Recovery**: on `load()`, WAL entries are replayed into a fresh
   memtable, restoring all uncommitted mutations.

## Admin Endpoints

All admin endpoints require a `shard` collection:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/compact` | POST | Trigger memtable flush + WAL truncation |
| `/collections/{name}/shards` | GET | List all shards with doc counts and token stats |
| `/collections/{name}/shards/{id}` | GET | Detail for a specific shard |
| `/collections/{name}/wal/status` | GET | WAL entry count, memtable size, tombstone count |
| `/collections/{name}/checkpoint` | POST | Force WAL checkpoint (flush + truncate) |
| `/collections/{name}/scroll` | POST | Paginated iteration over document IDs |
| `/collections/{name}/retrieve` | POST | Retrieve specific documents by ID |
| `/collections/{name}/search/batch` | POST | Batch search over multiple queries |

## Hybrid Search

The shard engine integrates with `HybridSearchManager` for BM25 + dense
fusion:

```python
from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager

hybrid = HybridSearchManager(
    shard_path=Path("my_hybrid"),
    dim=128,
    dense_engine="shard",
)
```

Dense and sparse results are returned separately. RRF scores are computed internally for the optional Tabu Search solver refinement step (`refine()`).

## GPU Memory and Auto-Tiering

The shard engine automatically detects available GPU memory:

- **GPU-resident** (corpus fits in VRAM): entire corpus is pre-loaded as a
  contiguous FP16 tensor. Scoring is zero-copy gather + MaxSim kernel.
- **Disk-backed** (corpus exceeds VRAM): safetensors shards are fetched
  on-demand with configurable pinned-memory buffering. Suitable for 1M+ docs.

Memory formula for GPU-resident mode:
```
VRAM = n_docs × max_tokens × dim × 2 bytes (FP16)
```

Example: 100K docs × 128 tokens × 128 dim = ~3.3 GB.

## Comparison with GEM

| Feature | GEM | Shard |
|---------|-----|-------|
| Build method | Rust proximity graph | LEMUR MLP + FAISS |
| Search method | Graph traversal + qCH proxy | ANN routing + exact MaxSim |
| Build time (100K) | Minutes | Seconds |
| Search latency | Sub-linear in N | Linear in candidates (capped) |
| Dependencies | Rust crate required | Python + native deps (PyTorch, FAISS, safetensors, Triton) |
| Scale sweet spot | 100K–10M+ | 10K–500K |
| WAL + CRUD | Yes | Yes |
| ROQ 4-bit | Yes | Yes |
| Filters | Roaring bitmap | Payload scan |
| Hybrid search | BM25 + Tabu | BM25 + RRF/Tabu |
