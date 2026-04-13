# Shard Engine Hardening Plan V2

**Generated from 8 independent audits post-implementation of Hardening Plan V1 (FIX GROUPS 1-19).**

This plan aggregates every flaw, bug, error, inconsistency, and performance problem
found across all audits. Each FIX GROUP is self-contained and traceable. Issues are
cross-referenced to PRODUCTION.md section 17 features and the original audit chunk.

---

## Severity Legend

- **CRITICAL**: Data loss, crash, or silent wrong results in production
- **HIGH**: Significant correctness or performance gap; blocks "elite" quality
- **MEDIUM**: Functional gap, misleading behavior, or suboptimal implementation
- **LOW**: Polish, documentation, or minor edge case

---

## FIX GROUP 20 — WAL payload-only UPSERT encoding/decoding/replay [CRITICAL]

**Audits**: 1 (issue #2), 2 (issues #1-4)
**PRODUCTION.md**: #4 (`update_payload`), #19 (WAL ops: UPDATE_PAYLOAD), #23 (crash recovery)

### Problem

`upsert_payload()` logs `log_upsert(doc_id, None, payload)`. The writer omits the
vector block when `vectors is None`, but the reader (`_parse_entry`) always interprets
the first 8 bytes after `doc_id` as `(n_vecs, dim)` for any non-DELETE op. This means:

1. **Payload-only WAL entries are mis-parsed** — JSON payload bytes are read as vector
   metadata, producing garbage `np.ndarray` data or parse failures.
2. **`_replay_wal` skips payload-only ops** — it only applies UPSERT when
   `entry.vectors is not None`, so payload updates are **lost on crash recovery**.
3. **No `UPDATE_PAYLOAD` op type** — PRODUCTION.md #19 specifies this as a distinct
   WAL operation. The shard engine only defines INSERT, DELETE, UPSERT.
4. **CRC scope differs from `gem_wal.py`** — shard WAL CRCs header+payload; gem_wal
   CRCs payload only. Binary format is not compatible despite PRODUCTION.md #20
   claiming format reuse.

### Files

- `voyager_index/_internal/inference/shard_engine/wal.py` (WalOp enum, _write_entry, _parse_entry)
- `voyager_index/_internal/inference/shard_engine/manager.py` (_replay_wal, upsert_payload)

### Required Changes

1. Add `WalOp.UPDATE_PAYLOAD = 3` to the enum.
2. In `WalWriter`: add `log_update_payload(doc_id, payload)` that writes
   `op + doc_id + json_len + json_bytes` (no vector block).
3. In `WalWriter._write_entry`: when `vectors is None`, write a flag byte or use the
   new op code so the reader can distinguish payload-only entries.
4. In `WalReader._parse_entry`: branch on op type — for UPDATE_PAYLOAD, parse
   `doc_id + json_payload` only (no `n_vecs/dim`).
5. In `manager._replay_wal`: handle `UPDATE_PAYLOAD` entries by calling
   `self._memtable.upsert_payload(entry.doc_id, entry.payload)`.
6. In `manager.upsert_payload`: call `log_update_payload` instead of `log_upsert`.
7. Add tests: write UPDATE_PAYLOAD entry, replay, verify payload survives restart.
8. Document the shard WAL as a distinct format from gem_wal (or align CRC scope).

---

## FIX GROUP 21 — Search/delete consistency and TOCTOU races [CRITICAL]

**Audits**: 1 (issues #1, #9), 3 (issue #5)
**PRODUCTION.md**: #24 (reader-writer lock), #27 (snapshot isolation)

### Problem

`search_multivector` takes a tombstones snapshot under `self._lock`, then releases the
lock and performs routing, GPU scoring, and memtable search. A concurrent `delete()` can
tombstone a document after the snapshot but before results are merged, causing deleted
documents to appear in search results.

Additionally:
- The filter path calls `_get_payload()` twice per candidate without consistent locking.
- `PreloadedGpuCorpus.score_candidates()` has no synchronization for concurrent queries.
- `manager.save()` is not serialized with other lifecycle methods.

### Files

- `voyager_index/_internal/inference/shard_engine/manager.py` (search_multivector, save)
- `voyager_index/_internal/inference/shard_engine/scorer.py` (score_candidates)

### Required Changes

1. Take a **single consistent snapshot** at the start of `search_multivector` under
   `self._lock`: capture tombstones, sealed doc IDs, and memtable state together.
   Release the lock, then use only the snapshot for the entire search path.
2. In the filter path, call `_get_payload()` once per candidate and cache the result.
3. Add a `threading.Lock` around `PreloadedGpuCorpus.score_candidates()` or document
   that concurrent GPU scoring relies on PyTorch/CUDA serialization.
4. Wrap `save()` with `self._lock` to prevent races with close/concurrent writers.

---

## FIX GROUP 22 — retrieve() returns tombstoned documents [CRITICAL]

**Audits**: 1 (issue #3)
**PRODUCTION.md**: #7 (`get` payloads by doc_id)

### Problem

`retrieve()` calls `_get_payload(did)` and `_get_doc_vectors(did)` without passing a
tombstones set. Deleted documents are returned as if they still exist.

### Files

- `voyager_index/_internal/inference/shard_engine/manager.py` (retrieve)

### Required Changes

1. At the start of `retrieve()`, capture `tombstones = self._memtable.tombstones_snapshot()`.
2. For each requested `doc_id`, check `if doc_id in tombstones: continue` (skip it).
3. Pass `tombstones` to `_get_payload()` for consistency.
4. Add test: add doc, delete doc, verify `retrieve()` returns empty for that doc_id.

---

## FIX GROUP 23 — Index.search_batch TypeError with shard engine [CRITICAL]

**Audits**: 6 (issue #1)
**PRODUCTION.md**: #6 (search_batch)

### Problem

`Index.search_batch()` passes `ef=` and `n_probes=` keyword arguments to the engine's
`search_batch()`. `ShardSegmentManager.search_batch` does not accept these parameters,
causing a `TypeError` at runtime.

### Files

- `voyager_index/index.py` (search_batch)
- `voyager_index/_internal/inference/shard_engine/manager.py` (search_batch)

### Required Changes

1. Add `**kwargs` to `ShardSegmentManager.search_batch()` signature (and
   `search_multivector`) to accept and ignore engine-incompatible parameters, OR
2. Branch in `Index.search_batch()` based on engine type, omitting `ef`/`n_probes`
   for shard engine.
3. Add test: call `Index.search_batch()` with a shard engine, verify no TypeError.

---

## FIX GROUP 24 — Journal backup/restore incomplete for shard collections [CRITICAL]

**Audits**: 6 (issues #2, #3)
**PRODUCTION.md**: #92 (SearchService journal recovery)

### Problem

`_begin_collection_mutation()` only backs up 4 files (`engine_meta.json`,
`manifest.json`, `payloads.json`, `wal.bin`) from the `shard/` directory. A built index
also uses `shard_*.safetensors`, `doc_index.npz`, and the entire `lemur/` subdirectory.
After a failed mutation, restore can leave corrupt or partial state.

Additionally, the engine is not flushed or closed before copying, so backup files can be
torn/inconsistent (unlike the DENSE path which closes the engine first).

### Files

- `voyager_index/_internal/server/api/service.py` (_begin_collection_mutation, _restore_collection_from_backup)

### Required Changes

1. Before backup, call `runtime.engine.flush()` to sync WAL to disk.
2. Back up the **entire** `shard/` directory tree (including safetensors, npz, lemur/),
   mirroring how DENSE backs up the whole `hybrid/` directory.
3. Restore must restore the entire tree, not just the 4 known files.
4. Add test: simulate mutation failure, verify full restore produces a working engine.

---

## FIX GROUP 25 — IVF-PQ nprobe never set [CRITICAL]

**Audits**: 5 (issue #1)
**PRODUCTION.md**: #10 (LEMUR routing)

### Problem

When `ann_backend = "faiss_ivfpq_ip"`, the built `IndexIVFPQ` uses FAISS default
`nprobe=1`, which yields very low recall. No code in the repo sets `nprobe`.

### Files

- `voyager_index/_internal/inference/shard_engine/lemur_router.py` (_rebuild_ann, _load_if_present)

### Required Changes

1. After building or loading an IVF index, set `index.nprobe` from config (e.g.
   `min(nlist, 10)` as default, or a dedicated `LemurConfig.nprobe` field).
2. Add `nprobe` field to `LemurConfig` with a reasonable default.
3. Add test: build IVF-PQ router, verify `nprobe > 1` and recall is reasonable.

---

## FIX GROUP 26 — brute_force_maxsim crashes on empty input [CRITICAL]

**Audits**: 3 (issue #19)
**PRODUCTION.md**: #5 (search)

### Problem

`brute_force_maxsim()` in `scorer.py` with `len(doc_ids) == 0` produces
`all_scores = []`, then `torch.cat(all_scores)` raises an error. While
`MemTable.search` guards against this, any direct caller crashes.

### Files

- `voyager_index/_internal/inference/shard_engine/scorer.py` (brute_force_maxsim)

### Required Changes

1. Add early return at the top of `brute_force_maxsim`: if `not doc_ids`, return `[], []`.
2. Add test: call `brute_force_maxsim` with empty doc_ids, verify empty results.

---

## FIX GROUP 27 — CI tests CUDA-gated on CPU-only runners [CRITICAL]

**Audits**: 7 (issues #1-2), 8 (issue #21)
**PRODUCTION.md**: #32-33 (CPU fallback), CI pipeline

### Problem

Nearly all shard engine tests use `@pytest.mark.skipif(not torch.cuda.is_available())`.
The `shard-engine-test` CI job runs on `ubuntu-latest` (no GPU), so most tests are
skipped. Effective CI coverage is minimal.

### Files

- `tests/test_shard_engine.py`, `test_shard_crud_wal.py`, `test_shard_advanced.py`,
  `test_shard_hardening.py`, `test_shard_engine_api.py`, `test_shard_roq4.py`,
  `test_hybrid_shard.py`
- `.github/workflows/ci.yml`

### Required Changes

1. Split each test module into CPU-smoke tests (no CUDA skip) and GPU-specific tests.
2. Ensure `ShardSegmentManager` can run the full search path on CPU (it already has
   `device="cpu"` support in some paths — verify and expand).
3. Create CPU-specific test fixtures with `device="cpu"` that exercise build, add,
   search, delete, upsert, WAL replay, scroll, stats.
4. Keep GPU-specific tests (Triton kernels, PreloadedGpuCorpus, ROQ Triton scoring)
   under CUDA skip guards.
5. Verify CI runs at least 50%+ of tests on CPU-only runners.

---

## FIX GROUP 28 — Nested filter operators ($and/$or) not implemented [HIGH]

**Audits**: 1 (issue #12), 8 (issue #1)
**PRODUCTION.md**: #16 (Qdrant-style nested ops)

### Problem

`_evaluate_filter()` only supports flat field conditions (`$eq`, `$in`, `$contains`,
`$gt`, `$lt`). PRODUCTION.md #16 specifies "Qdrant-style nested ops" which requires
`$and`/`$or` and recursive evaluation. This is needed for parity with
`GemNativeSegmentManager._evaluate_filter`.

### Files

- `voyager_index/_internal/inference/shard_engine/manager.py` (_evaluate_filter)

### Required Changes

1. Add support for top-level `$and` (list of filter dicts, all must match) and `$or`
   (list of filter dicts, at least one must match).
2. Make evaluation recursive so nested `$and`/`$or` combinations work.
3. Add tests for `$and`, `$or`, nested combinations, and edge cases.

---

## FIX GROUP 29 — Statistics double-counting sealed+memtable documents [HIGH]

**Audits**: 1 (issue #4)
**PRODUCTION.md**: #10 (stats)

### Problem

`get_statistics()` computes `n_live = n_sealed + n_memtable - n_tombstones`.
When a document is upserted (exists in sealed AND memtable), it is counted twice.
`total_vectors()` has the same issue.

### Files

- `voyager_index/_internal/inference/shard_engine/manager.py` (get_statistics, total_vectors)

### Required Changes

1. Compute the set of unique live doc IDs: `sealed_ids - tombstones | memtable_ids`.
2. Report `n_live` as the cardinality of this set.
3. Ensure `total_vectors` is consistent with `n_live`.
4. Add test: upsert a sealed doc, verify stats report correct (not doubled) count.

---

## FIX GROUP 30 — LEMUR router not updated on incremental CRUD [HIGH]

**Audits**: 1 (issue #7), 5 (issues #2-3, #9)
**PRODUCTION.md**: #10 (LEMUR routing)

### Problem

1. `manager.delete()` never calls `router.delete_docs()` — deleted sealed docs remain
   as FAISS candidates, wasting routing and scoring work.
2. `manager.add_multidense()` never calls `router.add_or_update_docs()` — new docs are
   only found via memtable brute-force, not via optimized LEMUR routing.
3. `RouterRetrainManager` exists but is never invoked from the main engine path.

### Files

- `voyager_index/_internal/inference/shard_engine/manager.py` (delete, add_multidense)
- `voyager_index/_internal/inference/shard_engine/lemur_router.py` (delete_docs, add_or_update_docs)

### Required Changes

1. In `manager.delete()`: call `self._router.delete_docs(doc_ids)` under lock.
2. Consider periodic router refresh after compaction (hook into CompactionTask).
3. Document that new memtable docs are not LEMUR-routed until the next compaction/rebuild.
4. Optionally integrate `RouterRetrainManager` into CompactionScheduler for drift-based retrain.

---

## FIX GROUP 31 — _load_sealed_vectors O(n) round-trips [HIGH]

**Audits**: 1 (issues #5-6)
**PRODUCTION.md**: #15 (PreloadedGpuCorpus)

### Problem

`_load_sealed_vectors()` loops `for did in self._doc_ids: self._store.fetch_docs([did])`,
making one round-trip per document. `ShardStore.fetch_docs` already supports batching
by shard internally. For large indices, this is extremely slow.

Additionally, if a doc is missing from the store, zeros are silently appended — GPU
preload then scores zero vectors, which is hard to detect.

### Files

- `voyager_index/_internal/inference/shard_engine/manager.py` (_load_sealed_vectors)

### Required Changes

1. Replace the per-doc loop with a single `self._store.fetch_docs(self._doc_ids)`.
2. For missing docs, log a warning instead of silently using zeros.
3. Consider skipping GPU preload for missing docs entirely.

---

## FIX GROUP 32 — PreloadedGpuCorpus has no live update mechanism [HIGH]

**Audits**: 3 (issues #3-4)
**PRODUCTION.md**: #15 (zero-fetch search)

### Problem

`PreloadedGpuCorpus` is built once from sealed `_doc_vecs`/`_doc_ids`. There is no
refresh path when:
- New documents are added to the memtable
- Documents are deleted
- Compaction changes the sealed state

The "full corpus on GPU" claim is only true for the sealed snapshot at preload time.

### Files

- `voyager_index/_internal/inference/shard_engine/scorer.py` (PreloadedGpuCorpus)
- `voyager_index/_internal/inference/shard_engine/manager.py` (_try_gpu_preload)

### Required Changes

1. Add a `refresh(doc_ids, doc_vecs)` method to `PreloadedGpuCorpus` that updates the
   GPU tensors incrementally or rebuilds from a new snapshot.
2. Call `refresh()` after compaction materializes new sealed state.
3. Document that memtable docs are always scored via CPU brute-force (separate path).

---

## FIX GROUP 33 — INT8/FP8/ROQ4 quantized scoring not wired to shard search [HIGH]

**Audits**: 3 (issues #13, #15), 4 (issue #1)
**PRODUCTION.md**: #29-31 (Triton INT8/FP8/ROQ), #37 (scalar INT8)

### Problem

1. Triton MaxSim supports `quantization_mode="int8"` and `"fp8"`, but the shard scorer
   hot path (`score_all_docs_topk`, `PreloadedGpuCorpus.score_candidates`) always casts
   to float16 and never passes quantization flags.
2. `score_roq4_topk` falls back to `device="cpu"` when CUDA is unavailable, then calls
   `roq_maxsim_4bit` which launches a Triton kernel — **Triton cannot run on CPU**,
   causing a crash.
3. Shard store can claim `compression=ROQ4` in the manifest while storing only FP16
   shards (when `roq_doc_codes is None`), creating a metadata/data mismatch.

### Files

- `voyager_index/_internal/inference/shard_engine/scorer.py` (score_all_docs_topk, score_roq4_topk)
- `voyager_index/_internal/inference/shard_engine/shard_store.py` (build, _decode_embeddings)
- `voyager_index/_internal/inference/shard_engine/manager.py` (search_multivector)

### Required Changes

1. Thread `compression` / `quantization_mode` from `ShardEngineConfig` through to
   `fast_colbert_scores` calls so INT8/FP8 scoring is accessible.
2. In `score_roq4_topk`: when CUDA is unavailable, fall back to FP16 decode + CPU
   MaxSim instead of attempting Triton on CPU.
3. In `shard_store.build()`: when `compression=ROQ4` but `roq_doc_codes is None`,
   either raise an error or explicitly set `manifest.compression = "fp16"`.
4. Add tests for INT8 and FP8 scoring paths (can be CPU-fallback tests).

---

## FIX GROUP 34 — Missing HTTP API endpoints [HIGH]

**Audits**: 6 (issues #4-7), 8 (issue #9)
**PRODUCTION.md**: #6 (search_batch), #8 (scroll), #85-86 (points API), #91 (shard admin)

### Problem

The HTTP server lacks endpoints for:
- `POST .../search_batch` (batch search)
- `POST .../points/batch` (batch add)
- `POST .../scroll` (paginated iteration)
- `POST .../retrieve` (get by doc_id)
- `GET /health`, `/ready`, `/metrics` test coverage

Additionally, the service layer applies a **second** post-filter (`_matches_filter`)
after the engine already filtered, with incompatible semantics (flat equality only vs
Qdrant-style nested ops). This can silently drop valid results.

### Files

- `voyager_index/_internal/server/api/routes.py`
- `voyager_index/_internal/server/api/service.py`

### Required Changes

1. Add missing HTTP routes for search_batch, scroll, retrieve, batch_add.
2. Remove or unify the redundant `_matches_filter` post-filter in the service layer
   for shard collections (the engine already applies `_evaluate_filter`).
3. Fix `checkpoint_collection` to report actual WAL entry count.
4. Add HTTP test coverage for `/health`, `/ready`, `/metrics`.

---

## FIX GROUP 35 — Hybrid integration gaps [HIGH]

**Audits**: 7 (issues #3-5, #8-9)
**PRODUCTION.md**: #49-54 (hybrid search, Tabu, fusion)

### Problem

1. `HybridSearchManager.search()` returns separate dense/sparse lists; RRF scores are
   only stored in private `_last_search_context`, not returned to callers.
2. Query vector shape contract differs by engine (HNSW=1D, shard=2D) without validation.
3. `index()` with `ndim==3` numpy input falls through to `[vectors]` (single 3D array),
   which is wrong for `add_multidense`.
4. `_decode_roq_vector` in the solver path requires `self.hnsw.quantizer` which the
   shard engine does not have.
5. `shard-engine.md` overstates "RRF" as end-user fusion when it is internal-only.

### Files

- `voyager_index/_internal/inference/index_core/hybrid_manager.py`
- `docs/guides/shard-engine.md`

### Required Changes

1. In `search()`: validate/normalize query shape based on `_dense_engine_type`.
2. In `index()`: reject `ndim==3` with a clear error, or properly handle batch multi-vector.
3. In `_decode_roq_vector`: add shard-engine-aware branch (or skip ROQ decode for shard).
4. Update `shard-engine.md` to clarify that RRF is internal context for the solver.
5. Document query shape requirements per engine type.

---

## FIX GROUP 36 — Test coverage gaps for PRODUCTION.md features [HIGH]

**Audits**: 8 (issues #2-10)
**PRODUCTION.md**: Multiple features

### Missing Test Coverage

| Feature | PRODUCTION.md # | Status |
|---------|-----------------|--------|
| `Index.snapshot` tarball | #9 | No test |
| `Index.set_metrics_hook` | #11 | No test |
| `FileLock` cross-process | #25 | No test |
| `atomic_json_write` | #26 | No test |
| CPU/Rust MaxSim fallback | #32-33 | No test |
| Scalar INT8 quantization | #37 | No test |
| Tabu solver refinement | #49 | No test |
| HTTP `/health`, `/ready`, `/metrics` | #81-83 | No test |
| Realistic crash recovery (kill mid-write) | #23 | Weak test |
| Full `Index` public API for shard | #1-18 | Thin coverage |

### Required Changes

1. Add `Index`-level tests for: delete, upsert, scroll, get/retrieve, snapshot, filter search.
2. Add CPU-only fallback tests (mock Triton unavailable, verify PyTorch fallback).
3. Add realistic crash recovery test (subprocess write + kill + reload).
4. Add Tabu solver + fusion strategy tests (or mark as deferred with skip).
5. Add HTTP observability endpoint tests.
6. Strengthen weak assertions (corrupt WAL replay, compaction scheduler, ROQ fallback).

---

## FIX GROUP 37 — WAL performance and robustness [HIGH]

**Audits**: 2 (issues #6-7, #10-12)
**PRODUCTION.md**: #20-21 (WAL CRC, corruption tolerance)

### Problem

1. `_write_entry` calls `_fd.flush()` after every single entry — high overhead for
   bulk ingest (no batched writes).
2. `WalReader.replay()` reads the entire file into memory — large WAL files cause
   memory pressure and slow startup.
3. `_count_existing` swallows errors silently — `n_entries` stays 0 on corrupt WAL.
4. `truncate()` resets `_n_entries = 0` before reopening the file — if `open()` fails,
   writer state is inconsistent.
5. `_parse_entry` returns `None` for unknown ops without logging.

### Files

- `voyager_index/_internal/inference/shard_engine/wal.py`

### Required Changes

1. Add optional batch mode: buffer writes and flush on explicit `sync()` or threshold.
2. Implement streaming WAL replay with bounded buffer instead of full-file `read()`.
3. In `_count_existing`: distinguish empty WAL from corrupt WAL; log warning on error.
4. In `truncate()`: only reset `_n_entries` after successful reopen.
5. Log when `_parse_entry` returns None for a structurally valid but unknown op.

---

## FIX GROUP 38 — Shard store validation and manifest gaps [MEDIUM]

**Audits**: 4 (issues #1, #4-10)
**PRODUCTION.md**: #68-69 (safetensors, manifest)

### Problem

1. `build()` does not validate: `shard_assignments` in `[0, n_shards)`, `doc_offsets`
   covering full vector range, `n_shards >= 1`, ROQ code/meta list lengths.
2. `StoreManifest` lacks `version` field, tombstone tracking, and PRODUCTION.md #9.3
   prescribed fields.
3. `num_shards` counts physical (non-empty) shards, not logical `n_shards` from config.
4. `load_shard_roq4()` does not check `SAFETENSORS_AVAILABLE`.
5. `_load_raw_shard()` / `_decode_embeddings()` have no graceful error handling for
   missing/corrupt files or missing tensor keys.

### Files

- `voyager_index/_internal/inference/shard_engine/shard_store.py`
- `voyager_index/_internal/inference/shard_engine/config.py`

### Required Changes

1. Add input validation in `build()`: assert bounds on assignments, offsets, n_shards,
   ROQ list lengths.
2. Add `version` field to `StoreManifest`; persist both `logical_n_shards` and
   `physical_shard_count`.
3. Add `SAFETENSORS_AVAILABLE` check in `load_shard_roq4()`.
4. Wrap `_load_raw_shard` / `_decode_embeddings` with try/except for missing keys;
   provide clear error messages with shard path.
5. Reject duplicate `doc_ids` in `build()`.

---

## FIX GROUP 39 — Router state consistency and configuration [MEDIUM]

**Audits**: 5 (issues #4-5, #7-8, #10), 4 (issue #19)
**PRODUCTION.md**: #10 (LEMUR routing)

### Problem

1. `self.ann_backend` (from constructor) and `self._state.ann_backend` (from disk) are
   two sources of truth — `_load_if_present` loads state but does not sync
   `self.ann_backend` from it.
2. `_get_gpu_resources()` is not thread-safe — two threads can create duplicate
   `StandardGpuResources`.
3. `RouterState` default `ann_backend` is still `"faiss_hnsw_ip"` while implementation
   uses `"faiss_flat_ip"`.
4. `add_or_update_docs()` has a redundant branch (both sides call `_rebuild_ann()`).
5. IVF `nlist` heuristic (`// 39 + 1`) is unexplained and not configurable.
6. `BuildConfig` and `ShardEngineConfig` have different defaults for `router_type` and
   `layout` — can cause silent behavior differences.

### Files

- `voyager_index/_internal/inference/shard_engine/lemur_router.py`
- `voyager_index/_internal/inference/shard_engine/config.py`

### Required Changes

1. After `_load_if_present`, sync `self.ann_backend = self._state.ann_backend`.
2. Add a `threading.Lock` around `_get_gpu_resources()` singleton creation.
3. Update `RouterState` default to `"faiss_flat_ip"`.
4. Remove redundant branch in `add_or_update_docs`.
5. Document or make `nlist` configurable via `LemurConfig`.
6. Align `BuildConfig` and `ShardEngineConfig` defaults (or document the distinction).

---

## FIX GROUP 40 — Service layer and API model gaps [MEDIUM]

**Audits**: 6 (issues #5-12)
**PRODUCTION.md**: #84-92 (HTTP API)

### Problem

1. `SearchRequest` has no `ef`/`n_probes` fields — parameters are in manager signature
   but not exposed via HTTP.
2. `CollectionInfo` omits shard-specific config (n_shards, k_candidates, token totals).
3. `CreateCollectionRequest` stores HNSW fields (`m`, `ef_construction`, `distance`)
   for SHARD collections without effect.
4. `SearchStrategy` is ignored on the shard search path.
5. HTTP errors use plain `HTTPException`, not structured `ErrorResponse` with codes.
6. `checkpoint_collection` always returns `wal_entries_after: 0` regardless of state.

### Files

- `voyager_index/_internal/server/api/models.py`
- `voyager_index/_internal/server/api/service.py`
- `voyager_index/_internal/server/api/routes.py`

### Required Changes

1. Add `ef` and `n_probes` to `SearchRequest` (optional fields, passed through if present).
2. Include shard-specific fields in `CollectionInfo` response.
3. Validate that HNSW-specific fields are not set for SHARD collections, or ignore them.
4. Wire `SearchStrategy` for shard search (or remove from the model for shard).
5. Use `ErrorResponse` with stable error codes in route handlers.
6. Fix `checkpoint_collection` to report actual WAL state.

---

## FIX GROUP 41 — ef/n_probes parameters dead in search path [MEDIUM]

**Audits**: 1 (issue #8), 6 (issue #6)
**PRODUCTION.md**: #5 (search with k, ef, n_probes)

### Problem

`search_multivector` accepts `ef` and `n_probes` parameters but never passes them to
the router or scorer. Callers may assume HNSW/IVF behavior that is not implemented.

### Files

- `voyager_index/_internal/inference/shard_engine/manager.py` (search_multivector)
- `voyager_index/_internal/inference/shard_engine/lemur_router.py` (route)

### Required Changes

1. Either wire `n_probes` to `LemurRouter.route()` (for IVF-PQ nprobe) and `ef` to a
   relevant scoring parameter, OR
2. Remove `ef`/`n_probes` from the shard manager signature and document that they are
   not applicable to shard routing.
3. Ensure `Index.search()` does not pass unsupported kwargs to the shard engine.

---

## FIX GROUP 42 — Documentation and PRODUCTION.md alignment [MEDIUM]

**Audits**: 4 (issue #3), 7 (issues #9-10, #19-20), 8 (issue #19)
**PRODUCTION.md**: Multiple sections

### Problem

1. PRODUCTION.md #9.2 says ROQ tensor keys are `"codes"`, `"codebook"`, `"scales"` but
   code uses `"roq_codes"`, `"roq_meta"`, `"embeddings"`.
2. PRODUCTION.md marks many shard features as "future work" (emoji markers) that are
   already implemented.
3. `shard-engine.md` comparison table says "Pure Python" for shard dependencies, but
   shard uses faiss-cpu, safetensors, PyTorch, and Triton.
4. Feature numbering in PRODUCTION.md does not match audit cross-references consistently.

### Files

- `PRODUCTION.md` (sections 9, 17)
- `docs/guides/shard-engine.md`

### Required Changes

1. Update PRODUCTION.md #9.2 to match actual tensor key names in shard_store.py.
2. Refresh status emoji in PRODUCTION.md #17 for all implemented features.
3. Fix "Pure Python" in shard-engine.md to "Python-first; optional native deps."
4. Add a feature ID crosswalk table or standardize numbering.

---

## FIX GROUP 43 — Memtable and compaction improvements [MEDIUM]

**Audits**: 2 (issues #8, #13-14), 1 (issues #19)
**PRODUCTION.md**: #12 (flush), #27 (snapshot isolation)

### Problem

1. `MemTable.upsert_payload()` can create a payload entry for a doc_id that has no
   vectors in `_docs` — weak invariant.
2. `tombstones_snapshot()` and `snapshot()` return shallow copies — mutable payload
   dicts are shared, risking unexpected mutation.
3. `CompactionTask` reports `flushed_docs = memtable_size` but flush only syncs WAL —
   misleading metric name.
4. Memtable search releases the lock before running `brute_force_maxsim` — eventual
   consistency, not strict serializable.

### Files

- `voyager_index/_internal/inference/shard_engine/memtable.py`
- `voyager_index/_internal/inference/shard_engine/compaction.py`

### Required Changes

1. In `upsert_payload`: if `doc_id not in self._docs`, either raise or document that
   this is intentional for sealed-only docs.
2. Use `copy.deepcopy` for payload dicts in `snapshot()` (or document shallow copy
   semantics).
3. Rename CompactionTask fields to `wal_synced_entries` / `memtable_docs_at_sync`.
4. Document memtable search snapshot semantics.

---

## FIX GROUP 44 — Shard store fetch efficiency [MEDIUM]

**Audits**: 4 (issue #2), 1 (issue #5)
**PRODUCTION.md**: #68 (safetensors shards)

### Problem

`fetch_docs()` groups by shard but loads and decodes the **entire** shard to extract a
few documents. For selective access patterns, this is O(shard_size) not O(requested_docs).

### Files

- `voyager_index/_internal/inference/shard_engine/shard_store.py` (fetch_docs, load_docs_from_shard)

### Required Changes

1. For the current safetensors format: document that fetch is O(shard_size).
2. Consider mmap-based access or per-doc offset index for selective reads.
3. Short-term: cache loaded shard tensors to avoid redundant full-shard loads.

---

## FIX GROUP 45 — Minor polish and edge cases [LOW]

**Audits**: All

### Issues

1. `_evaluate_filter` has unused `doc_id` parameter (lint noise).
2. `scroll()` builds full sorted ID set per call — expensive for huge corpora.
3. `_explain_score` is private — no public wrapper for token attribution API.
4. `score_all_docs_topk` has redundant float32→float16 conversions.
5. `fits_on_gpu` byte estimate is wrong for bfloat16/FP8 dtypes.
6. `_search_lock` lazy init in `lemur_router._search` is redundant (already set in __init__).
7. `add_or_update_docs` redundant branch (both paths identical).
8. `CandidatePlan.raw_candidate_count` is misnamed (actually post-tombstone count).
9. `SearchConfig.batch_size` field is unused on the shard search path.
10. Duplicate `doc_ids` in `shard_store.build()` not rejected.
11. No explicit `del self._index` before GPU index replacement in router.
12. Test `test_roq_kernel_loader` has no assertions.
13. Test `test_fp16_fallback_when_roq_unavailable` does not simulate actual ROQ unavailability.
14. Test `test_compaction_scheduler` asserts nothing meaningful.
15. Loose recall threshold (0.5) in `test_recall_at_10`.
16. No single-document or near-max-capacity edge case tests.
17. `WalWriter` lacks `__enter__`/`__exit__` context manager.
18. Module docstrings on `manager.py` are stale (omit WAL, retrieve, scroll, flush).

### Required Changes

Apply each fix individually — these are independent cleanup items.

---

## Summary

| Severity | Count | FIX GROUPs |
|----------|-------|------------|
| CRITICAL | 8 | 20, 21, 22, 23, 24, 25, 26, 27 |
| HIGH | 9 | 28, 29, 30, 31, 32, 33, 34, 35, 36 |
| MEDIUM | 7 | 37, 38, 39, 40, 41, 42, 43 |
| LOW | 1 | 44 (18 sub-items) |
| **TOTAL** | **25** | FIX GROUPS 20-45 |

### Recommended Implementation Order

**Phase A (CRITICAL — must fix first):**
1. FIX GROUP 20: WAL payload-only encoding/replay
2. FIX GROUP 21: Search consistency races
3. FIX GROUP 22: retrieve() tombstone check
4. FIX GROUP 23: Index.search_batch TypeError
5. FIX GROUP 24: Journal backup completeness
6. FIX GROUP 25: IVF-PQ nprobe
7. FIX GROUP 26: brute_force_maxsim empty crash
8. FIX GROUP 27: CI CPU test coverage

**Phase B (HIGH — required for elite quality):**
9. FIX GROUP 28: $and/$or filters
10. FIX GROUP 29: Statistics double-counting
11. FIX GROUP 30: Router CRUD integration
12. FIX GROUP 31: Sealed vectors O(n) fix
13. FIX GROUP 32: GPU corpus refresh
14. FIX GROUP 33: Quantized scoring wiring
15. FIX GROUP 34: Missing HTTP endpoints
16. FIX GROUP 35: Hybrid integration
17. FIX GROUP 36: Test coverage gaps

**Phase C (MEDIUM — polish for production):**
18. FIX GROUP 37: WAL performance
19. FIX GROUP 38: Shard store validation
20. FIX GROUP 39: Router state consistency
21. FIX GROUP 40: Service/API models
22. FIX GROUP 41: ef/n_probes parameters
23. FIX GROUP 42: Documentation alignment
24. FIX GROUP 43: Memtable/compaction
25. FIX GROUP 44: Fetch efficiency

**Phase D (LOW — cleanup):**
26. FIX GROUP 45: Polish items (18 sub-items)
