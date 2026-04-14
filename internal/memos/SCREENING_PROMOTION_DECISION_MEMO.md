# Screening Promotion Decision Memo

## Purpose

Set a clear repo policy for:

- what stays default
- what remains available but not promoted
- what remains experimental
- the exact thresholds a screening backend must beat before it graduates

This memo refers to the refreshed evidence in
`internal/validation/validation-screening-audit/quantization.json`.

## Current Evidence Snapshot

Primary real-model corpus:

- `corpus_points=539`
- `query_count=8`
- `embedding_source=real_model`

Refreshed multimodal screening results:

| Lane | `speedup_vs_full_precision` | `recall_at_k` | `fallback_rate` | `direct_gather_rate` | `top1/topk retention @128` |
| --- | ---: | ---: | ---: | ---: | --- |
| `prototype_sidecar_screening` | `1.1000x` | `0.6000` | `0.375` | `0.625` | `0.75 / 0.825` |
| `pooled_dense_screening` | `1.1029x` | `0.6000` | `0.375` | `0.625` | `0.75 / 0.825` |
| `centroid_screening` | `1.1033x` | `0.6250` | `0.375` | `0.625` | `0.75 / 0.80` |

Relevant low-bit reference at the same corpus size:

- `roq_1bit_screening` kernel lane: `1.6977x` kernel speedup, `top1/topk retention @128 = 1.0 / 0.925`
- this is strong as a kernel-level candidate generator, but it is not yet the promoted multimodal screening path because the repo-grade runtime contract is broader than kernel speed alone: exact fallback behavior, trust state, persistence, delta maintenance, restart safety, and API integration all matter

Scale smoke for centroid:

- `centroid_screening_scale_harness`: `n_docs=10000`, `speedup_vs_full_precision=0.288x`, `recall_at_k=1.0`
- interpretation: quality can be strong in the synthetic harness, but the current implementation still does not deliver an end-to-end latency win there

## Current Implemented Gate

The repo already contains a backend-specific promotion gate in `scripts/full_feature_validation.py` for centroid promotion.

Today that gate requires all of the following:

1. beat the current prototype backend end-to-end
2. beat the best low-bit baseline on candidate preservation by:
   - `topk_retention >= best_lowbit_topk_retention + 0.10`
   - `top1_retention >= best_lowbit_top1_retention`
3. stay competitive with the pooled backend by:
   - `recall_at_k >= pooled_recall_at_k - 0.05`
   - `elapsed_ms < pooled_elapsed_ms`
4. remain contract-ready

On the refreshed run, centroid fails all performance-related checks and stays `keep_experimental`.

## Decision

### 1. What stays default

- Default OSS multimodal model: `collfm2`
- Default multimodal scoring path: exact FP16 Triton MaxSim
- Default public recommendation: exact search remains the truthful baseline and recommended default

### 2. What stays available but not promoted

- `strategy="optimized"` with `prototype_hnsw`

Rationale:

- it is the most complete lightweight screening index path in repo terms
- it has trust controls, exact fallback, restart-safe persistence, delta maintenance, and CI coverage
- it does not yet earn promotion as the primary latency story because the refreshed win is only about `1.10x` on the real-model corpus and quality remains too soft for an unqualified recommendation

### 3. What remains experimental

- `centroid` screening backend
- `roq_1bit_screening` and `roq_2bit_screening` as integrated multimodal screening defaults
- FP8 MaxSim

Rationale:

- centroid is contract-wired but not clearly faster enough or recall-safe enough in end-to-end repo evidence
- low-bit screening kernels are promising, but the kernel lane is ahead of the full production runtime lane
- FP8 is still explicitly experimental in the current OSS guidance

## Graduation Policy

Use three levels, not one.

### Level A: Contract-ready optional lane

This means the backend is allowed to ship behind an explicit opt-in.

Required:

- bootstrap calibration passes
- `top1_retention >= 0.75` on calibration sample
- `topk_retention >= 0.60` on calibration sample
- exact fallback is proven correct
- risky-query bypass is working
- persistence, restart safety, and delete/append invariants are covered by tests
- readiness/reporting surfaces degraded or corrupted state correctly

This level is already satisfied by the current prototype path and by the centroid path as a guarded implementation.

### Level B: Promoted optimized backend

This means the backend may be described as the recommended path for users who explicitly choose `strategy="optimized"`.

Required on a representative real-model corpus:

- `speedup_vs_full_precision >= 1.15`
- `recall_at_k >= 0.85`
- `fallback_rate <= 0.20`
- `direct_gather_rate >= 0.75`
- `top1_retention @128 >= 0.90`
- `topk_retention @128 >= 0.92`
- `top1_retention @256 >= 0.95`
- `topk_retention @256 >= 0.95`
- end-to-end latency must beat the current optimized incumbent by at least `5%`
- no readiness or restart regressions

Current result:

- no screening backend meets this bar

### Level C: Hero latency path

This means the backend may be positioned as a headline OSS speed feature rather than a guarded advanced option.

Required:

- `speedup_vs_full_precision >= 1.25` on the primary real-model corpus
- `speedup_vs_full_precision >= 1.10` on at least one larger-scale corpus or scale harness that is closer to deployment conditions
- `recall_at_k >= 0.90`
- `fallback_rate <= 0.10`
- `top1_retention @128 >= 0.95`
- `topk_retention @128 >= 0.95`
- `top1_retention @256 >= 0.97`
- `topk_retention @256 >= 0.97`
- quality must not be materially worse than the best competing lane at the same budget

Current result:

- no backend meets this bar

## Lane-by-Lane Outcome

| Lane | Status | Why |
| --- | --- | --- |
| Exact FP16 Triton MaxSim | `default` | truthful baseline, stable, already the canonical OSS path |
| `prototype_hnsw` lightweight screening index | `available but not promoted` | best integrated runtime contract, but current win and retention are not strong enough for promotion |
| `centroid` lightweight screening index | `experimental` | slightly faster than prototype on the refreshed corpus, but still below promotion thresholds and loses the current implemented gate |
| `roq_1bit_screening` kernel lane | `experimental` | strongest kernel-level retention/speed reference, but not yet the promoted production runtime lane |
| `roq_2bit_screening` kernel lane | `experimental` | not strong enough vs the 1-bit lane to justify promotion today |
| FP8 MaxSim | `experimental` | still intentionally non-default in OSS guidance |

## Final Call

The correct repo policy now is:

- keep exact FP16 Triton MaxSim as the default
- keep `collfm2` as the default OSS vLLM ColPali model
- keep `prototype_hnsw` as the current implementation default only when users explicitly opt into optimized screening
- keep centroid and low-bit screening lanes experimental until one of them beats the Level B bar
- do not market any screening backend yet as the primary OSS latency hero

## Promotion Trigger

The first backend that should graduate is simply the first one that clears Level B on real-model evidence without losing its contract guarantees.

Until then, the repo should frame screening as:

- useful
- guarded
- recall-aware
- promising
- not yet a universally promoted default
