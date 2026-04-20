# Licensing Guide

## Repository-Level License

This repository is licensed under the **Creative Commons Attribution-NonCommercial
4.0 International** license (SPDX: `CC-BY-NC-4.0`). See the root `LICENSE` file
for the full legal code.

In plain English:

- You may **use, copy, modify, and redistribute** this code for **research,
  evaluation, academic, personal, and other non-commercial** purposes, as long
  as you provide attribution.
- You **may not** use this code, in whole or in part, for any **commercial or
  revenue-generating purpose** (including, without limitation, selling it,
  offering it as a hosted or managed service, or embedding it in a product or
  service that you sell, license, or monetize) without first obtaining a
  separate commercial license from the copyright holder.

This summary is a convenience and does not replace the legal text in `LICENSE`.

## Explicit Exceptions And Boundaries

### Vendored Qdrant Subtree

- Path: `src/kernels/vendor/qdrant/`
- License: Apache-2.0
- Reason: this subtree is a vendored copy of upstream
  [qdrant/qdrant](https://github.com/qdrant/qdrant) and cannot be relicensed.
  Each file inside this directory remains under its original Apache-2.0 terms,
  including any local modifications, which are clearly marked per Apache-2.0
  Section 4(b).
- Details: see `internal/contracts/QDRANT_VENDORING.md` and
  `src/kernels/vendor/qdrant/LICENSE`.

This is the **only** carveout. All other code in the repository — including
the native Rust crates under `src/kernels/` that depend on the vendored
subtree via Cargo path dependencies — is licensed under CC-BY-NC-4.0.
When those crates are built into binaries that embed Qdrant code, the
embedded Qdrant code continues to carry its original Apache-2.0 terms at the
file level.

## Practical Rule

All code in this repository is `CC-BY-NC-4.0` unless it lives inside
`src/kernels/vendor/qdrant/`, in which case it is `Apache-2.0`.

## Distribution

Source distributions, wheels, and container images for this repository
should include:

- `LICENSE` (CC BY-NC 4.0 legal code)
- `LICENSING.md` (this file)
- `THIRD_PARTY_NOTICES.md`
- `src/kernels/vendor/qdrant/LICENSE` whenever the vendored Qdrant subtree
  or any derivative of it is included.

## Commercial Use

CC-BY-NC-4.0 forbids commercial use without a separate license. If you want
to use this code or any derivative of it commercially — for example, in a
revenue-generating product, in a hosted or managed service, in a paid
consulting engagement, or in any business activity that is primarily
intended for commercial advantage or monetary compensation — please contact
the copyright holder to discuss commercial licensing terms:

- Email: `commercial@latence.ai`

Pre-existing releases that were previously distributed under Apache-2.0
remain available under Apache-2.0 to those who already obtained them; the
license change above applies to this source tree and to releases made from
it going forward.
