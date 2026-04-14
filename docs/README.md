# Documentation Guide

`voyager-index` ships one OSS product surface, but several different reading
paths.

Install:

- `pip install "voyager-index[shard]"` is the mainline install path
- optional extras: `[server]`, `[multimodal]`, `[preprocessing]`, `[gpu]`, `[native]`
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
- ColBERT late-interaction: `docs/guides/colbert.md`
- ColPali multimodal: `docs/guides/colpali.md`
- Scaling and memory: `docs/guides/scaling.md`
- GEM graph index: `docs/guides/gem-native.md` (historical / compatibility path)

## Public Contracts

- `OSS_FOUNDATION.md`: supported public Python and API contract
- `MULTIMODAL_FOUNDATION.md`: multimodal model matrix, storage, and scoring guidance
- `ADAPTER_CONTRACTS.md`: documentation-level seams between OSS, providers, and future sidecars

## Evaluation And Evidence

- `BENCHMARKS.md`: reproducible OSS smoke benchmark harness
- `SCREENING_PROMOTION_DECISION_MEMO.md`: current promotion policy for multimodal screening
- `docs/validation/README.md`: archived validation bundles and what each one proves

## Contributors And Release

- `CONTRIBUTING.md`: setup, validation, and contribution workflow
- `CHANGELOG.md`: release history
- `SECURITY.md`: supported surface and disclosure instructions
- `tools/README.md`: contributor-only scripts moved out of the repo landing zone

## Legal And Vendor

- `LICENSING.md`: repo-level licensing guide
- `QDRANT_VENDORING.md`: vendored Qdrant boundary
- `THIRD_PARTY_NOTICES.md`: redistributed third-party notices
