# Releasing colsearch

This project now has one release narrative:

- shard-first public product
- public native wheels for the shard CPU fast-path and solver lane
- optional graph-aware augmentation through the `latence-graph` extra
- release publish parity with the curated CI and validation lanes

## Before You Cut A Release

Make sure these are true:

- `README.md`, `docs/`, and the package surface agree on the shard-first story
- the install surface clearly distinguishes `full`, `gpu`, `native`, `solver`, and `latence-graph`
- `CHANGELOG.md` has a new top entry for the release version
- `pyproject.toml` version matches that entry
- root and native package versions agree on the release line
- package build, wheel inspection, native wheel validation, and release CI lanes are green

## Supported Package Story

The release surface should match this install matrix:

- `colsearch[full]`
- `colsearch[full,gpu]`
- `colsearch[shard]`
- `colsearch[shard,shard-native]`
- `colsearch[server,shard]`
- `colsearch[server,shard,solver]`
- `colsearch[server,shard,native]`
- `colsearch[server,shard,latence-graph]`

Supported public native packages:

- `latence-shard-engine`
- `latence-solver`

Research HNSW and GEM crates are not part of the public release bundle.

## Standard Release Checklist

1. Decide the version using semver.
2. Update `pyproject.toml` and the public native package metadata under `src/kernels/`.
3. Add a top changelog entry in `CHANGELOG.md`.
4. Run the release validation commands:

```bash
python -m pip install --upgrade pip
python -m pip install build twine
python -m build
python -m twine check dist/*
pytest \
  tests/test_oss_release_hygiene.py \
  tests/test_release_polish_surface.py \
  tests/test_latence_graph_quality.py \
  tests/test_latence_graph_lightrag.py \
  tests/test_latence_graph_crud.py \
  -v --tb=short
```

5. Build the public native wheels and verify them locally when relevant:

```bash
python -m pip install ./src/kernels/shard_engine
python -m pip install ./src/kernels/knapsack_solver
python -c "import latence_shard_engine; print(latence_shard_engine.__file__)"
python -c "import latence_solver; print(latence_solver.version())"
python scripts/full_feature_validation.py --tmp-data-dir ./tmp_data
python benchmarks/oss_reference_benchmark.py --device cpu --points 16 --top-k 3
```

6. Rehearse clean installs from built artifacts, including:

```bash
pip install "colsearch[full]==0.X.Y"
pip install "colsearch[full,gpu]==0.X.Y"
```

7. Commit the version and changelog changes.
8. Push to `main`.
9. Wait for CI and release-surface validation.
10. Create the GitHub release:

```bash
gh release create v0.X.Y --title "0.X.Y" --notes-file CHANGELOG.md
```

11. Verify the published packages:

```bash
pip install colsearch==0.X.Y
python -c "import colsearch; print(colsearch.__version__)"
```

Also verify the supported native projects when they are part of the release:

```bash
pip install latence-shard-engine==0.X.Y
pip install latence-solver==0.X.Y
```

## Release Notes Framing

Prefer notes in this order:

1. user-facing product changes
2. API or SDK changes
3. benchmark or performance notes with caveats
4. docs, CI, and packaging cleanup

## Pre-releases

Use tags like `0.X.Yrc1` when you want to validate packaging or docs before the
final release.

```bash
gh release create v0.X.Yrc1 --title "0.X.Yrc1" --prerelease --notes "Release candidate."
```

## Hotfixes And Yanks

If a release is broken:

```bash
gh release edit v0.X.Y --draft
```

Then yank the broken PyPI release from the PyPI UI or API and publish a patch
release immediately.

## Prerequisites

- GitHub CLI authenticated for release creation
- PyPI trusted publisher configured for `colsearch`, `latence-shard-engine`, and `latence-solver`
