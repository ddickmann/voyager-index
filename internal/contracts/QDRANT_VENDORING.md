# Qdrant Vendoring Boundary

This repository vendors a subtree of `qdrant/qdrant` under
`src/kernels/vendor/qdrant/`.

## Upstream Pin

- Upstream repository: [qdrant/qdrant](https://github.com/qdrant/qdrant)
- Upstream tag: **v1.16.3**
- Upstream commit: `bd49f45a8a2d4e4774cac50fa29507c4e8375af2`
- Upstream license: Apache-2.0
- Local license file: `src/kernels/vendor/qdrant/LICENSE`

## What Is Vendored

The vendored subtree is centered on the Qdrant segment and HNSW internals used
by `src/kernels/hnsw_indexer`, but it retains a broader slice of the upstream
workspace than just `lib/segment/`. The subtree currently includes:

- `lib/segment/` and related HNSW internals
- supporting workspace crates under `lib/`
- upstream root workspace files such as `Cargo.toml` and `Cargo.lock`
- `docs/`, `tools/`, `openapi/`, and `tests/`
- upstream packaging and workflow metadata that came along with the retained
  subtree

This is not a minimal "HNSW-only" extraction. It is a large vendored Qdrant
subtree with HNSW/segment code as the main reason it is present.

## What This Repository Uses

The primary consumer of the vendored subtree is:

- `src/kernels/hnsw_indexer/` — Python bindings on top of Qdrant's
  segment/HNSW internals

The `hnsw_indexer` crate depends on `lib/segment` and `lib/common/common` via
Cargo path dependencies.

## Local Modifications

The following files have been modified from the upstream v1.16.3 snapshot.
Each modified file carries a prominent header notice per Apache-2.0 Section
4(b).

| File | Change |
|---|---|
| `README.md` | Added security caution note for insecure Docker deployment |
| `docs/QUICK_START.md` | Added security caution note for insecure Docker deployment |
| `lib/segment/src/common/validate_snapshot_archive.rs` | Commented out deprecated `ar.set_sync(true)` for newer `tar` crate compatibility |
| `lib/segment/src/index/hnsw_index/graph_layers.rs` | Added `search_multi_entry()` method implementing GEM-style multi-entry beam search |
| `lib/segment/src/index/hnsw_index/hnsw.rs` | Added graceful empty-graph fallback in `load()` and `search_multi_entry_with_graph()` method |

All other files in `src/kernels/vendor/qdrant/` are unmodified copies of the
upstream v1.16.3 release.

## Apache-2.0 Compliance

Per Apache-2.0 Section 4, redistribution of this vendored subtree satisfies:

- **(a)** The Apache-2.0 license text is included at
  `src/kernels/vendor/qdrant/LICENSE` and is shipped in all source
  distributions, wheels, and container images.
- **(b)** All modified files carry prominent modification notices in their
  headers identifying the changes made.
- **(c)** All copyright, patent, trademark, and attribution notices from the
  upstream source are retained.
- **(d)** Qdrant upstream does not include a NOTICE file, so this obligation
  is not triggered.

## Maintenance Strategy

Short term:

- keep the vendored subtree pinned at v1.16.3
- document all modifications in this file and via in-file headers
- keep the public OSS package surface isolated from upstream docs/tests/tooling

Medium term:

- slim the vendor to only the crates actually consumed (`lib/segment/`,
  `lib/common/`, and their transitive workspace dependencies)
- track upstream releases and re-vendor as needed
- if the GEM graph index work replaces the HNSW backend, the Qdrant vendor
  dependency can be retired entirely
