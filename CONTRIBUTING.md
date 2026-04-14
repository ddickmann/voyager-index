# Contributing

## Scope

The supported OSS surface for contributions is the `voyager-index` foundation:

- `voyager_index/`
- `src/` for the deprecated compatibility shims and native package sources
- `deploy/reference-api/`
- `docs/`, `examples/`, and `notebooks/`
- `tools/` for maintainer-oriented scripts and experiments
- `src/kernels/knapsack_solver/` for the canonical OSS knapsack solver package and bindings
- root packaging, docs, tests, and examples

Deferred or product-internal areas may exist in the repository, but they are
not the default contribution target for the OSS foundation cut.

## Canonical Docs And Boundaries

Before changing public behavior, treat these docs as the canonical OSS guidance:

- `ADAPTER_CONTRACTS.md`
- `docs/README.md`
- `README.md`
- `OSS_FOUNDATION.md`
- `MULTIMODAL_FOUNDATION.md`

Namespace rules:

- add or document public behavior under `voyager_index.*`
- treat `voyager_index._internal.*` as implementation detail rather than user-facing contract
- treat `src.*` as a deprecated compatibility namespace, not the preferred import path

Behavior rules:

- `SearchPipeline` is the vector-first dense+sparse pipeline
- direct late-interaction multivector queries belong to `ColbertIndex` or the reference API
- multimodal ingestion and query flows expect precomputed embeddings, with `VllmPoolingProvider` as the public provider seam
- optional `latence_solver` refinement and `/reference/optimize` share the same canonical OSS solver contract
- the package is distributed via PyPI (`pip install voyager-index`) with source builds for native crates

## Local Setup

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e ".[server,shard,multimodal,dev,native-solver-build]"
```

Optional native packages (require Rust toolchain):

```bash
python -m pip install ./src/kernels/knapsack_solver
```

Or use the one-command installer:

```bash
bash scripts/install_from_source.sh
```

Native truth:

- repo source presence does not make a native module active
- `latence_solver` only affects runtime after it is built and importable in the current environment
- `cargo` and `rustc` are required to build the solver package from source
- `latence_solver` is the canonical OSS solver package with CPU fallback and optional CUDA-backed execution when built with GPU features

## Before Opening A Change

- keep the public import surface under `voyager_index`
- keep public docs and examples aligned with the `voyager_index` surface in the same pass as behavior changes
- prefer CPU-safe tests when adding new coverage
- preserve the licensing boundary documented in `LICENSING.md` and `QDRANT_VENDORING.md`
- keep the public solver API aligned around `/reference/optimize` and the shared request contract instead of adding parallel one-off optimize paths
- keep `SearchPipeline` guidance limited to its vector-first dense/hybrid role
- distinguish the canonical OSS solver contract from any future premium productization, but do not describe the OSS solver as CPU-only
- when touching native code or native-lane docs, verify `latence_solver` with the same build/install rigor

## Validation

At minimum, run the tests relevant to your changes. For server or package
changes, prefer also checking:

```bash
python -m pip install -e .
pytest \
  tests/test_oss_foundation_surface.py \
  tests/test_reference_api_persistence.py \
  tests/test_reference_api_contract.py \
  tests/test_reference_api_security.py \
  tests/test_multimodal_storage.py \
  tests/test_ontology_fixture_contract.py
python benchmarks/oss_reference_benchmark.py --device cpu --points 16 --top-k 3
python scripts/full_feature_validation.py --tmp-data-dir ./tmp_data
```

Deeper maintainer-only helpers now live under:

- `tools/benchmarks/`
- `tools/verification/`
- `tools/dev/`
- `tools/sandbox/`

For native-specific changes, also verify:

```bash
python -c "import latence_solver; print(latence_solver.version())"
```

The ontology contract test resolves its fixture in this order:

- `VOYAGER_ONTOLOGY_FIXTURE`
- `tmp_data/dataset_di_825cbaae40335bc4265a3726.json` when present in the working tree or shared workspace
- `tests/fixtures/dataset_di_fixture.json`

## Releases

See [RELEASING.md](RELEASING.md) for the step-by-step release workflow.

## Pull Requests

Use concise titles that explain why the change is needed. Include:

- a short summary of the user-facing impact
- how you validated the change
- any known follow-up work or limitations
