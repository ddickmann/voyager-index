# Changelog

This changelog tracks the official shipped OSS release line. Older draft notes
that did not correspond to a published release were removed so version history
reads in release order again.

## Unreleased

### Phase 8 — RROQ158 SOTA default at `group_size=128` + dim-aware fallback (2026-04-20)

- flipped the default `Rroq158Config.group_size` from `32` to `128` (one
  scale per token at dim=128, the most-tested production dim) — closes
  the Pareto sweep run after Phase 7. The SOTA flip is **~13% smaller
  storage** (~40 vs ~46 bytes/token at dim=128 → ~6.4× smaller than
  fp16, up from ~5.5×) with **CPU p95 ~10–30% faster** (one fewer scale
  load per group in the popcount kernel) and **NDCG@10 within ±0.005**
  of the previous gs=32 default on the 3 BEIR datasets re-validated for
  the flip (arguana, fiqa, nfcorpus, full-eval CPU 8-worker; per-cell
  data in `reports/rroq158_pareto_cells/`). The remaining 3 BEIR
  datasets (`scifact`, `scidocs`, `quora`) plus `hotpotqa` will be
  filled in by a post-merge full BEIR-6 sweep that refreshes the
  headline tables in a follow-up commit.
- added `_resolve_group_size(requested, dim)` to
  `voyager_index/_internal/inference/quantization/rroq158.py` — a
  dim-aware fallback that picks the largest of `{requested, 64, 32}`
  that divides `dim`. `encode_rroq158` calls it at entry and records
  the **resolved** group_size in the manifest, so the new
  `group_size=128` default works transparently on dim=64 / 96 / 160
  production corpora (steps down to gs=64 / gs=32 with a log warning)
  without callers having to special-case dim. Existing callers passing
  an explicit `Rroq158Config(group_size=...)` are unaffected when their
  value already divides dim.
- updated `BuildConfig.rroq158_group_size` default to `128` and
  refreshed the dim-aware pre-validation in
  `voyager_index/_internal/inference/shard_engine/_builder/pipeline.py`
  and `.../_manager/lifecycle.py` so the corpus-too-small check uses
  the resolved gs (not the raw request) and dim=64 corpora no longer
  silently fall back to FP16 under the new default.
- updated docstrings on `Rroq158Config`, `Compression.RROQ158`,
  `BuildConfig`, and `BuildConfig.rroq158_group_size` with the new
  headline numbers and the dim-aware fallback rule.
- added the canonical [`docs/guides/quantization-tuning.md`](docs/guides/quantization-tuning.md):
  decision matrix (rroq158 vs rroq4_riem vs fp16), per-dim recipe
  table, override guidance for high-intra-token-variance corpora
  (e.g. arguana → pin `Rroq158Config(group_size=64)` to recover the
  −0.0058 marginal fail), and the closing retrospective on the
  outlier-rescue investigation with the prototype p-curve.
- refreshed `README.md`, `docs/benchmarks.md`,
  `docs/guides/shard-engine.md`, `docs/guides/rroq-mathematics.md`,
  `docs/posts/sub-2-bit-late-interaction.md` (added v1.1 update
  section), `docs/api/python.md`,
  `docs/getting-started/quickstart.md`,
  `docs/full_feature_cookbook.md`, `docs/reference_api_tutorial.md`,
  and `docs/guides/max-performance-reference-api.md` for the new
  default + dim-aware fallback + new headline numbers. The detailed
  per-dataset tables in `README.md`/`docs/benchmarks.md` remain at
  the gs=32 baseline until the post-merge full BEIR-6 sweep refreshes
  them in a follow-up commit.
- closed the **outlier-rescue hybrid investigation** as
  "explored, didn't ship". The full Python prototype on arguana
  (preserved at `reports/rroq158_hybrid_prototype_log.txt`) showed
  the rescue curve flattening fast (slope drops from +0.005 NDCG per
  +5% rescue at p<0.10 to +0.0025 at p=0.20 and turns noisy beyond),
  with even p=1.00 leaving −0.029 NDCG vs the FP32 ceiling — the
  uniform `gs=128` win we already have delivers the same ~13% storage
  reduction at the same or better quality without any kernel work or
  on-disk format change. Removed `docs/posts/rroq158-outlier-rescue-design.md`
  (replaced by the one-paragraph retrospective in the tuning guide)
  and updated `reports/rroq158_pareto_arguana.md` to point at the
  tuning guide instead of the dropped design doc.
- migration: existing `RROQ158` indexes load unchanged — the manifest
  carries the build-time `group_size`. Only newly built indexes pick
  up the new default. Pin `Rroq158Config(group_size=32)` to restore
  the previous default exactly; pin `Rroq158Config(group_size=64)`
  for the safest cross-dataset choice.

### Phase 7 — RROQ production-validation sweep (2026-04-19 → 2026-04-20)

- ran the BEIR 2026-Q2 production-validation sweep
  (`benchmarks/beir_2026q2_full_sweep.py`): 6 datasets × 4 codecs
  (fp16, int8, rroq158, rroq4_riem) × 2 modes (GPU + 8-worker CPU) ×
  full BEIR query sets, on RTX A5000 + AMD EPYC 7B13 (128-thread) host;
  raw per-cell JSONL with full provenance under `reports/beir_2026q2/`
- applied the F1 default-promotion decision rule via
  `scripts/format_beir_2026q2_table.py`: rroq4_riem matches fp16
  quality (~−0.1 pt avg NDCG@10) but fails the per-cell GPU/CPU p95
  conditions (~2–3× slower on GPU, ~5–10× slower on CPU at the BEIR
  batch shape), so the build-time default reverts to
  `Compression.RROQ158`
- promoted `Compression.RROQ158` (Riemannian-aware 1.58-bit ternary,
  K=8192) as the default codec for newly built indexes on both GPU
  (Triton fused kernel) and CPU (Rust SIMD kernel with hardware
  popcount + cached rayon thread pool); avg −1 pt NDCG@10 vs fp16 with
  flat R@100 and ~5.5× smaller per-token storage. CPU p95 is currently
  slower than the fp16 AVX2 baseline at the production batch shape —
  the win is storage, not throughput, and closing the CPU-latency gap
  is on the post-Phase-7 backlog
- shipped `Compression.RROQ4_RIEM` as the no-quality-loss lane —
  Riemannian-aware 4-bit asymmetric per-group residual quantization
  with a fused Triton kernel (`roq_maxsim_rroq4_riem`) and a Rust SIMD
  kernel (`latence_shard_engine.rroq4_riem_score_batch`, AVX2/FMA +
  cached rayon pool); ~3× smaller than fp16 on disk, ~0.5 pt NDCG@10
  gap, parity-tested to rtol=1e-4 against the python reference on both
  lanes; opt-in via `compression=Compression.RROQ4_RIEM`
- added a brute-force codec-fidelity diagnostic harness
  (`benchmarks/topk_overlap_sweep.py`) that scores every (query, doc)
  pair with both fp16 and the codec and reports per-query top-K overlap
  (K ∈ {10, 20, 50, 100}); ran the full BEIR-6 sweep with results in
  `reports/beir_2026q2/topk_overlap.jsonl`. Headline: rroq158 averages
  ~79% top-10 / ~80% top-100 overlap with FP16 brute-force across
  BEIR-6 (range 73–83% top-10), with R@100 within −2.1 pt of FP16 on
  every dataset (within −1.4 pt on arguana, within 0 pt on scifact /
  quora). rroq4_riem averages ~96% top-10 / ~97% top-100, confirming
  the no-quality-loss positioning. Top-K overlap is roughly **flat
  across K** for rroq158 (e.g. quora 72.9% → 72.1% from K=10 to
  K=100) — this disconfirms the earlier "wider serve window de-risks
  displacement" framing; the displacement is *out of the candidate
  set*, not within it. R@100 still recovers because rroq158 admits
  the labeled relevant docs; the displacement happens among the
  non-relevant tail of FP16's top-100. Documentation across README,
  `docs/benchmarks.md`, `docs/posts/sub-2-bit-late-interaction.md`,
  `docs/guides/shard-engine.md`, `docs/guides/rroq-mathematics.md`,
  and `research/low_bit_roq/PROGRESS.md` was updated with the
  measured per-dataset numbers and the corrected window framing
- shipped a kernel-graceful fallback for small-corpus tests: if the
  build-time codec is rroq158/rroq4_riem and no kernel is reachable
  (no CUDA + no Rust SIMD), and the auto-K-shrink fired
  (`k_effective < k_requested`), the search path logs a warning and
  falls back to the FP16 scorer; production-scale indices (where K is
  not auto-shrunk) still raise a hard `RuntimeError` rather than
  silently degrade. Avoids spurious CI failures on minimal hosts
- shipped a `dim < group_size` / `n_tok < group_size` guard at index
  build time: lifecycle.py and pipeline.py log a warning and switch the
  effective compression to FP16 when the corpus shape is too small for
  the chosen rroq158/rroq4_riem `group_size`. Pre-encoded indexes are
  unaffected
- enriched the rroq158/rroq4_riem index-side metadata with
  `k_requested` and `k_effective` so the runtime fallback above can
  distinguish between "the user asked for K=8192 and got K=8192" vs
  "the corpus is too small and K shrunk to 16"
- exposed `rroq158_*` and `rroq4_riem_*` knobs (`K`, `seed`,
  `group_size`) through the Python API, the HTTP collection-create
  payload, and the shard-engine CLI; existing fp16 / rroq158 /
  rroq4_riem indexes load unchanged through the build-time codec
  recorded in the manifest
- added the auto-derive path so search-time `quantization_mode` is no
  longer required when the manifest already records the build-time
  codec for rroq158 or rroq4_riem
- added the math doc (`docs/guides/rroq-mathematics.md`) covering the
  RaBitQ extension, the Riemannian log map, the FWHT-rotated tangent
  ternary derivation, the K = 8192 bits-per-coord accounting, and the
  4-bit asymmetric variant
- added the public-facing write-up
  (`docs/posts/sub-2-bit-late-interaction.md`) framing the rroq158
  result for the ColBERT/PLAID community

## 0.1.5 — Release Gate Hotfix

This release republishes the shard-engine decomposition work on a clean CI line
after fixing the small lint regressions that slipped through the initial `0.1.4`
cut.

### Release integrity

- fixed the shard refactor parity script bootstrap so the release lint lane
  accepts the repo-local import setup
- normalized import ordering and explicit public exports in the refactor-touching
  files that failed the hosted Ruff gate
- bumped the root package and supported native packages onto the `0.1.5` line
  so the hotfix release cleanly supersedes the drafted `0.1.4` cut

## 0.1.4 — Shard Engine Decomposition And Release Evidence

This release keeps the shard product surface stable while decomposing the large
shard-engine modules behind compatibility facades and hardening the parity
evidence required to ship that refactor safely.

### Shard engine maintainability

- split the shard manager, store, fetch pipeline, LEMUR router, builder, WAL,
  and ColBANDIT reranker into focused internal modules while preserving public
  import paths
- reduced config coupling by separating serving configuration from sweep-only
  configuration behind compatibility exports
- introduced internal protocols for router, store, fetch, reranker, and native
  exact backends to narrow cross-module ownership

### Runtime capability visibility

- surfaced fallback and capability state for LEMUR routing, pinned staging, and
  native exact execution through shard statistics and reference API metadata
- added startup logging for shard capability selection so development and
  production runs expose fallback decisions explicitly

### Validation and release confidence

- added shard refactor contract coverage for import compatibility, artifact
  parity, query trace stability, and runtime capability reporting
- added a machine-readable shard refactor parity report and wired it into CI so
  release evidence is reproducible instead of ad hoc
- bumped the root package and supported native packages onto the `0.1.4` line
- refreshed release hygiene checks to validate the aligned package versions
## 0.1.3 — Production Release Hardening

This release closes the gap between the public product story and the shipped
package, native-wheel, and release pipeline surfaces.

### Packaging and install surface

- added a canonical `voyager-index[full]` install profile for the full public CPU-safe surface
- added `shard-native` and broadened `native` so the public native story now covers both `latence-shard-engine` and `latence-solver`
- bumped the root package and supported native packages onto the `0.1.3` line
- tightened package data so the shipped sdist includes the graph quality fixture required by release validation

### Graph-aware production path

- kept `latence-graph` as a public optional extra and pinned it to the verified public `latence>=0.1.1` line
- clarified throughout the docs that the graph lane can consume compatible prebuilt graph data directly and remains additive to the shard-first hot path
- preserved the graph route-conformance, provenance, and retrieval-uplift evidence as a distinct proof layer from shard performance benchmarks

### CI, release, and OSS hygiene

- expanded the native release bundle to include the shard-engine wheel alongside the solver wheel
- tightened release documentation and automation around clean-install rehearsal, native-wheel validation, and publish gating
- refreshed the README, install docs, issue templates, and contributor guidance around the supported production lane
- added repo-governance files for dependency updates, code ownership, and contributor conduct

## 0.1.2 — Shard Production Surface

This release makes the shard engine the clear public product surface.

### Retrieval and serving

- production-wired shard search with LEMUR routing, ColBANDIT, and Triton MaxSim
- shard scoring controls exposed for `int8`, `fp8`, and `roq4`
- durable CRUD, WAL, checkpoint, recovery, and shard admin endpoints
- multi-worker single-host reference server posture

### API and SDK

- base64 vector transport helpers exposed from `voyager_index.transport`
- public HTTP API accepts base64 payloads for dense and multivector requests
- shard configuration knobs surfaced on collection create, search, and info APIs
- dense hybrid mode selection documented and shipped as `rrf` or `tabu`

### Docs and DX

- README, quickstart, API docs, and top-level guides rewritten around the shard-first story
- benchmark methodology documented with a 100k comparison placeholder table
- reference API examples now lead with base64 and shard-friendly install profiles

### Release and packaging

- release notes and changelog chronology cleaned up
- CI trimmed to shard-only production lanes plus solver validation
- supported native add-on story reduced to `latence_solver`

## 0.1.0 — Initial OSS Foundation Release

Initial public package release for `voyager-index`.

### Foundation

- installable `voyager_index` package and published OSS packaging surface
- durable reference FastAPI service
- dense, late-interaction, and multimodal collection kinds
- CRUD, restart-safe persistence, and public examples

### Retrieval

- exact MaxSim exports through the public package
- CPU-safe MaxSim fallback when Triton is unavailable
- hybrid dense + BM25 retrieval
- optional solver-backed refinement via `latence_solver`

### Multimodal

- preprocessing helpers for renderable source documents
- multimodal model registry and provider seams
- ColPali-oriented multimodal retrieval surface
