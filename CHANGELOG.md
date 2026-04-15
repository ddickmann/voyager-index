# Changelog

This changelog tracks the official shipped OSS release line. Older draft notes
that did not correspond to a published release were removed so version history
reads in release order again.

## Unreleased

## 0.1.3 — Production Release Hardening

This release closes the gap between the public product story and the shipped
package, native-wheel, and release pipeline surfaces.

### Packaging and install surface

- added a canonical `voyager-index[full]` install profile for the full public CPU-safe surface
- added `shard-native` and broadened `native` so the public native story now covers both `latence-shard-engine` and `latence-solver`
- bumped the root package and supported native packages onto the `0.1.3` line
- tightened package data so the shipped sdist includes the graph quality fixture required by release validation

### Graph-aware production path

- kept `latence-graph` as a public optional extra and pinned it to the verified public `latence>=0.1.1` line
- clarified throughout the docs that the graph lane can consume compatible prebuilt graph data directly and remains additive to the shard-first hot path
- preserved the graph route-conformance, provenance, and retrieval-uplift evidence as a distinct proof layer from shard performance benchmarks

### CI, release, and OSS hygiene

- expanded the native release bundle to include the shard-engine wheel alongside the solver wheel
- tightened release documentation and automation around clean-install rehearsal, native-wheel validation, and publish gating
- refreshed the README, install docs, issue templates, and contributor guidance around the supported production lane
- added repo-governance files for dependency updates, code ownership, and contributor conduct

## 0.1.2 — Shard Production Surface

This release makes the shard engine the clear public product surface.

### Retrieval and serving

- production-wired shard search with LEMUR routing, ColBANDIT, and Triton MaxSim
- shard scoring controls exposed for `int8`, `fp8`, and `roq4`
- durable CRUD, WAL, checkpoint, recovery, and shard admin endpoints
- multi-worker single-host reference server posture

### API and SDK

- base64 vector transport helpers exposed from `voyager_index.transport`
- public HTTP API accepts base64 payloads for dense and multivector requests
- shard configuration knobs surfaced on collection create, search, and info APIs
- dense hybrid mode selection documented and shipped as `rrf` or `tabu`

### Docs and DX

- README, quickstart, API docs, and top-level guides rewritten around the shard-first story
- benchmark methodology documented with a 100k comparison placeholder table
- reference API examples now lead with base64 and shard-friendly install profiles

### Release and packaging

- release notes and changelog chronology cleaned up
- CI trimmed to shard-only production lanes plus solver validation
- supported native add-on story reduced to `latence_solver`

## 0.1.0 — Initial OSS Foundation Release

Initial public package release for `voyager-index`.

### Foundation

- installable `voyager_index` package and published OSS packaging surface
- durable reference FastAPI service
- dense, late-interaction, and multimodal collection kinds
- CRUD, restart-safe persistence, and public examples

### Retrieval

- exact MaxSim exports through the public package
- CPU-safe MaxSim fallback when Triton is unavailable
- hybrid dense + BM25 retrieval
- optional solver-backed refinement via `latence_solver`

### Multimodal

- preprocessing helpers for renderable source documents
- multimodal model registry and provider seams
- ColPali-oriented multimodal retrieval surface
