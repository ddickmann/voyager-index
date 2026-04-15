# Documentation Guide

`voyager-index` ships one OSS product surface, but several different reading
paths.

Install:

- `pip install "voyager-index[full]"` is the canonical full public CPU install path
- `pip install "voyager-index[full,gpu]"` adds Triton GPU scoring on supported CUDA hosts
- optional extras: `[server]`, `[multimodal]`, `[preprocessing]`, `[gpu]`, `[shard-native]`, `[solver]`, `[native]`, `[latence-graph]`
- contributors and developers can install from source: `bash scripts/install_from_source.sh --cpu` or `make install-cpu`
- the install commands in `docs/reference_api_tutorial.md` and `docs/full_feature_cookbook.md` cover both paths
- `README.md` is the product homepage; the docs below expand the same public OSS story rather than replacing it

## Start Here

- New users: `README.md`
- End-to-end API tutorial: `docs/reference_api_tutorial.md`
- Full-feature cookbook: `docs/full_feature_cookbook.md`
- Runnable advanced feature tour: `examples/reference_api_feature_tour.py`
- Runnable examples: `examples/README.md`
- Notebooks: `notebooks/README.md`

## Engine Guides

- Shard engine (LEMUR-routed): `docs/guides/shard-engine.md`
- Latence graph sidecar: `docs/guides/latence-graph-sidecar.md`
- Enterprise control-plane boundary: `docs/guides/control-plane.md`
- ColBERT late-interaction: `docs/guides/colbert.md`
- ColPali multimodal: `docs/guides/colpali.md`
- Scaling and memory: `docs/guides/scaling.md`

## Public References

- `docs/benchmarks.md`: public benchmark framing and methodology
- `PRODUCTION.md`: deployment checklist for the shard-first path
- `CHANGELOG.md`: release history

## Contributor And Internal References

- `internal/README.md`: internal plans, validation, benchmark notes, and contracts
- `internal/contracts/OSS_FOUNDATION.md`: supported Python/API contract reference
- `internal/contracts/MULTIMODAL_FOUNDATION.md`: multimodal storage and serving notes
- `internal/contracts/ADAPTER_CONTRACTS.md`: cross-surface seams and integration notes
- `internal/validation/README.md`: archived validation bundles and supporting evidence

## Contributors And Release

- `CONTRIBUTING.md`: setup, validation, and contribution workflow
- `SECURITY.md`: supported surface and disclosure instructions
- `tools/README.md`: contributor-only scripts moved out of the repo landing zone
- `research/legacy/README.md`: archived GEM/HNSW research, tests, and benchmarks

## Legal And Vendor

- `LICENSING.md`: repo-level licensing guide
- `internal/contracts/QDRANT_VENDORING.md`: vendored Qdrant boundary
- `THIRD_PARTY_NOTICES.md`: redistributed third-party notices
