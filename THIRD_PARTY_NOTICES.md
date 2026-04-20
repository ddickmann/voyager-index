# Third-Party Notices

The umbrella `voyager-index` repository is licensed under
**CC-BY-NC-4.0** (see `LICENSE` and `LICENSING.md`). It additionally bundles
third-party code that carries its own, separate license. Each such component
and its license is documented below; those obligations apply independently
of the repository-level license whenever the third-party files (or
derivatives of them) are redistributed.

## Qdrant

- Component: `src/kernels/vendor/qdrant/`
- Upstream: [qdrant/qdrant](https://github.com/qdrant/qdrant)
- Upstream tag: v1.16.3
- Upstream commit: `bd49f45a8a2d4e4774cac50fa29507c4e8375af2`
- License: Apache License 2.0
- Local license copy: `src/kernels/vendor/qdrant/LICENSE`

The vendored Qdrant subtree provides segment/HNSW internals used by
`src/kernels/hnsw_indexer/`. Five files have been modified from the upstream
snapshot; each carries a prominent header notice per Apache-2.0 Section 4(b).
See `internal/contracts/QDRANT_VENDORING.md` for the complete list of modifications.

## Distribution Guidance

When distributing source archives, wheels, or container images, include:

- the repository root `LICENSE`
- this `THIRD_PARTY_NOTICES.md`
- `LICENSING.md`
- the vendored Qdrant license file at `src/kernels/vendor/qdrant/LICENSE`

All of these are included automatically by the `license-files` directive in
`pyproject.toml`.

If future vendored or bundled third-party components are added, list them here
with the same structure.
