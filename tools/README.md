# Contributor Tools

This directory holds maintainer and evaluator tooling that used to live at the
repo root.

Keeping these scripts in-tree is intentional. They are useful engineering
material, but they are not the first thing a new OSS user should have to sort
through.

## Layout

- `tools/benchmarks/`: one-off and platform-specific benchmark scripts
- `tools/verification/`: deeper verification harnesses and exploratory regression scripts
- `tools/dev/`: maintainer helpers, packaging utilities, and inspection scripts
- `tools/sandbox/`: experimental screening prototypes and scratch-space code

## Public First-Run Path

If you are trying `colsearch` for the first time, start with:

- `README.md`
- `docs/reference_api_tutorial.md`
- `examples/README.md`
- `notebooks/README.md`

Use this `tools/` tree when you are evaluating internals, reproducing older
experiments, or contributing changes.
