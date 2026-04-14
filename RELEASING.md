# Releasing voyager-index

This project now has one release narrative:

- shard-first public product
- optional `latence_solver` native add-on
- shard-only CI and validation lanes for the production story

## Before You Cut A Release

Make sure these are true:

- `README.md`, `docs/`, and the package surface agree on the shard-first story
- `CHANGELOG.md` has a new top entry for the release version
- `pyproject.toml` version matches that entry
- package build, wheel inspection, and shard CI lanes are green

## Supported Package Story

The release surface should match this install matrix:

- `voyager-index[shard]`
- `voyager-index[server,shard]`
- `voyager-index[server,shard,gpu]`
- `voyager-index[server,shard,native]`

`native` currently means the solver wheel only.

## Standard Release Checklist

1. Decide the version using semver.
2. Update `pyproject.toml`.
3. Add a top changelog entry in `CHANGELOG.md`.
4. Run the release validation commands:

```bash
python -m pip install --upgrade pip
python -m pip install build twine
python -m build
python -m twine check dist/*
pytest tests/test_oss_release_hygiene.py tests/test_release_polish_surface.py -v --tb=short
```

5. Verify the shard-only CI story locally when relevant:

```bash
python scripts/full_feature_validation.py --tmp-data-dir ./tmp_data
python benchmarks/oss_reference_benchmark.py --device cpu --points 16 --top-k 3
```

6. Commit the version and changelog changes.
7. Push to `main`.
8. Wait for CI.
9. Create the GitHub release:

```bash
gh release create v0.X.Y --title "0.X.Y" --notes-file CHANGELOG.md
```

10. Verify the published package:

```bash
pip install voyager-index==0.X.Y
python -c "import voyager_index; print(voyager_index.__version__)"
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
pip install twine
twine yank voyager-index 0.X.Y
```

Then publish a patch release immediately.

## Prerequisites

- GitHub CLI authenticated for release creation
- PyPI trusted publisher configured for `release.yml`
