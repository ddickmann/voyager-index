# Licensing Guide

## Repository-Level License

This repository is licensed under the **Apache License 2.0** (SPDX:
`Apache-2.0`). See the root `LICENSE` file for the full legal text and the
root `NOTICE` file for the attribution notice.

In plain English:

- You may **use, copy, modify, redistribute, and sublicense** this code for
  **any purpose, commercial or non-commercial**.
- You must **retain the license and notices** when you redistribute
  (Apache-2.0 Section 4), and **mark any files you modify** as changed
  (Section 4(b)).
- The license includes an **express patent grant** from contributors
  (Section 3).

This summary is a convenience and does not replace the legal text in
`LICENSE`.

## License History

- Versions `0.1.0` through `0.1.6`, published under the `voyager-index`
  name, were licensed under Apache-2.0.
- Version `0.1.7` through the current published release were distributed
  under `CC-BY-NC-4.0`.
- This source tree, and all releases made from it starting with the next
  version, are licensed under Apache-2.0.
- Recipients of earlier releases retain them under the license they were
  originally distributed with.

## Vendored Qdrant Subtree

- Path: `src/kernels/vendor/qdrant/`
- License: Apache-2.0 (upstream's own license)
- Reason: this subtree is a vendored copy of upstream
  [qdrant/qdrant](https://github.com/qdrant/qdrant). Each file inside this
  directory remains under its original upstream Apache-2.0 terms and
  copyright, and any local modifications are clearly marked per Apache-2.0
  Section 4(b).
- Details: see `internal/contracts/QDRANT_VENDORING.md` and
  `src/kernels/vendor/qdrant/LICENSE`.

The vendored subtree's license is consistent with the repository-level
license, but its upstream copyright and NOTICE obligations are separate and
travel with those files — including into the native Rust crates under
`src/kernels/` that depend on the vendored subtree via Cargo path
dependencies, and into any binaries built from them.

## Practical Rule

All code in this repository is `Apache-2.0`. Files inside
`src/kernels/vendor/qdrant/` carry upstream Qdrant's Apache-2.0 license and
copyright rather than this repository's; honor their notices when
redistributing.

## Distribution

Source distributions, wheels, and container images for this repository
should include:

- `LICENSE` (Apache License 2.0 text)
- `NOTICE`
- `LICENSING.md` (this file)
- `THIRD_PARTY_NOTICES.md`
- `src/kernels/vendor/qdrant/LICENSE` whenever the vendored Qdrant subtree
  or any derivative of it is included.
