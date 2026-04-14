# Validation Evidence

This directory keeps evaluator-facing and contributor-facing validation bundles
out of the repo root while preserving the underlying evidence.

These artifacts are not the first-run user path. They are here to back up
claims in the public docs, benchmark notes, and screening promotion memo.

## Bundles

- `validation-sidecar/`: main multimodal validation bundle for the prototype screening lane
- `validation-sidecar-slice/`: smaller real-model screening slice used for comparative checks
- `validation-centroid/`: centroid screening validation bundle
- `validation-centroid-targeted/`: targeted centroid regression checks
- `validation-screening-audit/`: promotion-gate evidence referenced by `internal/memos/SCREENING_PROMOTION_DECISION_MEMO.md`

## Intended Audience

- Evaluators comparing retrieval quality, latency, and storage behavior
- Contributors validating changes to screening, storage, or native-lane behavior

## Important Notes

- Raw JSON and log files are archived evidence and may be verbose
- The canonical product story remains in `README.md`, `docs/benchmarks.md`, and `internal/contracts/MULTIMODAL_FOUNDATION.md`
- Validation bundles support those docs; they are not themselves the OSS contract
- GEM-lite screening evidence should be read as sidecar-level validation only:
  the repo now adopts a subset of ideas from
  [GEM: A Native Graph-based Index for Multi-Vector Retrieval](https://arxiv.org/abs/2603.20336),
  but it does not expose a GEM-native graph index yet
