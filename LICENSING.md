# Licensing Guide

## Repository-Level License

The OSS foundation surface of this repository is licensed under Apache-2.0.
See the root `LICENSE` file.

## Explicit Exceptions And Boundaries

### Vendored Qdrant Subtree

- Path: `src/kernels/vendor/qdrant/`
- License: Apache-2.0
- Details: see `internal/contracts/QDRANT_VENDORING.md`

### Native Crate Packages

All native crates in this repository are licensed under Apache-2.0, consistent
with the repository root license:

- `src/kernels/knapsack_solver/` — Apache-2.0
- `src/kernels/hnsw_indexer/` — Apache-2.0
- `src/kernels/gem_router/` — Apache-2.0

## Practical Rule

All code in this repository is Apache-2.0 unless a vendored third-party
component explicitly states otherwise. The only vendored third-party code is
the Qdrant subtree, which is also Apache-2.0.

## Distribution

Source distributions, wheels, and container images for the OSS foundation
should include:

- `LICENSE`
- `LICENSING.md`
- `THIRD_PARTY_NOTICES.md`
- `src/kernels/vendor/qdrant/LICENSE` when the vendored subtree or derivatives
  of it are included
