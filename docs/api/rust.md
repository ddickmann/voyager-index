# Rust Crates

The supported native crate story for the shard-first OSS surface is intentionally
small.

## `latence-solver`

Source: `src/kernels/knapsack_solver/`

`latence-solver` is the native refinement crate behind the optional Tabu-style
packing/refinement lane used by the public `/reference/optimize` API and the
solver-backed dense hybrid flow.

### What it provides

- knapsack-style refinement for retrieval result packing
- the native acceleration path behind `dense_hybrid_mode="tabu"`
- a shared contract between in-process refinement and the reference API

### Why it is still public

- it is part of the supported PyPI/source-build story
- CI builds and validates it as part of the release surface
- it remains relevant to the current shard-first product narrative

## Legacy native crates

Historical GEM, GEM-router, and HNSW-native material is still archived in-repo
for research and archaeology, but it is not part of the supported public Rust
crate surface.

See `research/legacy/README.md` if you need that older material.
