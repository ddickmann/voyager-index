# Changelog

This changelog tracks the official shipped OSS release line. Older draft notes
that did not correspond to a published release were removed so version history
reads in release order again.

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
