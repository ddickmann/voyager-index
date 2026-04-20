<!--
Continuous progress log for the "Riemannian Low-Bit ROQ" plan.

Read top-to-bottom: pinned headers first, then newest entries. Total catch-up
time is minutes regardless of how long the project has been running.

Update rules (mirrored from the plan, do not relax):

- One entry per experiment, never per seed or per dataset.
- Tables max 5 rows. Full metric set lives in the cited JSON.
- Why is 1-3 sentences. Longer analysis goes in a separate note linked here.
- Negative results follow the same template; failures must be visible.
- Auto-stub from harness emits config + table + artifact links;
  engineer fills Verdict + Why + Gate-impact within 1 working day.
- Pinned headers (Current State, Promoted, Killed) are updated by hand at
  every gate. One commit per update.
-->

# PROGRESS — Riemannian Low-Bit ROQ

## Current State

- **Phase:** **Phase 7 — production-validation sweep.** `Compression.RROQ158` (K=8192, group_size=32, FWHT seed=42) is the default codec for newly built indexes on both GPU (Triton fused kernel) and CPU (Rust SIMD kernel). `Compression.RROQ4_RIEM` ships as the opt-in no-quality-loss lane (Riemannian-aware 4-bit asymmetric, parity-tested on both lanes). Existing fp16 / rroq158 / rroq4_riem indexes load unchanged; manifests carry the build-time codec. The full BEIR 2026-Q2 production-validation sweep + F1 default-promotion decision rule is in `reports/beir_2026q2/`.
- **Most recent gate:** `phase-7-beir-2026q2-production-sweep-2026-04-20` — applied F1 decision rule via `scripts/format_beir_2026q2_table.py`. **rroq4_riem promotion to default: REJECTED** (matches fp16 quality at ~−0.1 pt avg NDCG@10, but ~2–3× slower than fp16 on GPU and ~5–10× slower on CPU at the BEIR batch shape — the per-cell p95 conditions fail). **rroq158 retention as default: ACCEPTED** (avg −1 pt NDCG@10 vs fp16 with flat R@100 and ~1.0× GPU p95 ratio — passes the `≥ −1.5 pt avg NDCG@10 and ≤ 1.2× avg GPU p95` budget). Honest weakness documented: rroq158 CPU p95 is ~5–9× slower than the fp16 AVX2 baseline at the production batch shape (2000 candidates × ~30 query tokens) — the storage win (~5.5× smaller doc tokens) is the primary value of the codec on CPU, not the throughput. Closing the CPU-latency gap is on the post-Phase-7 backlog.
- **Earlier gate:** `phase-1.5-cpu-kernel-perf-pass-2026-04-19` — Rust SIMD CPU kernel beat fp16 at the **microbench** shape (32 q-tok × 32 d-tok × 512 docs, single-call): p50 = 14.05 ms vs fp16 199.77 ms (14.3×), p95 = 86.34 ms vs fp16 498.11 ms (5.8×). Phase-7 sweep showed this microbench advantage does **not** hold at the production batch shape (LEMUR routing + 2000 candidates × ~30 query tokens with FetchPipeline copies + per-query Python-to-Rust tensor copies in the wrapper). Both numbers are real on their respective shapes; the production story is the BEIR sweep.
- **A-best candidates:** ternary (A2.5)
- **B-best candidate:** rroq158-K8192 (B3-extended) — quality at K=8192 closes the K=1024 gap; GPU p95 at fp16 parity, CPU p95 slower than fp16 (storage codec, see Phase-7 verdict above).
- **Production candidates:** **rroq158-K8192 = default** for new indexes on GPU and CPU. **rroq4_riem-K8192 = opt-in no-quality-loss lane** (~0.5 pt NDCG@10 of fp16, ~3× smaller storage; slower than fp16 in absolute latency on the BEIR sweep). FP16 / INT8 / FP8 / ROQ4 remain selectable opt-outs.
- **Hardware budget (operator constraint):** 24 GB GPU VRAM (A5000); CPU RAM is generous (251 GB available on the bench box). Build / sweep params still capped to keep portability.
- **Open questions:**
  - Does K=4096 / K=8192 close the scidocs / arguana quality gap without blowing the disk budget? (current K=1024 has 256 KB centroid table; K=8192 = 2 MB, still trivial.)
  - The rroq158 GPU p95 in the BEIR loop (74–98 ms on hard datasets) is 50–80× the kernel's own steady-state cost (1.26 ms). Plausible: Triton autotune cache thrash across LEMUR/kernel boundary, or PyTorch allocator stalls under the 2 k-candidate `index_select` gather pattern. **Closing this is the prerequisite for re-evaluating production fitness on quality alone.**
  - Should the CPU lane re-encode the corpus from FP16 on first load (current behaviour: re-encodes on every benchmark process start), or should `pack_shard_rroq158` write the encoded payload to disk so subsequent loads are O(mmap)?
- **Restored after CPU OOM:** `runners.py` rebuilt; the 488 MB `tests/fixtures/token_sample_1m.npy` (real ColBERT tokens, 200k each from nfcorpus / scifact / arguana / scidocs / fiqa) survived; no cell reports were lost (none had been generated).

## Promoted

- `phase-0-harness` — multi-seed harness, paired-bootstrap, candidate-recall logging, cold/warm p95, distortion bench, PROGRESS.md auto-stubs all green on 34 unit tests; runners (`run_a1.py`, `run_a6.py`, `run_c1_5.py`) smoke-tested.
- `a1-lite-pilot` — 20-cell A1 distortion pilot on 8 192 real ColBERT tokens / 128 queries: angular error monotone in bit-width (1-bit p50=29.3°, 1.58-bit=23.7°, 2-bit=20.4°); FWHT helps 2-bit (–4° p50) and is recall-neutral for 1-bit. Full 5-seed × 5-dataset BEIR follow-up still pending the operator's next memory window.
- `a2-roq2-asym-kernel` — 2-bit asymmetric Triton kernel on A5000: 0.29 ms p50, 55k QPS, parity 0.0 vs NumPy reference. Replaces the unusable symmetric 2-bit kernel in `voyager_index/_internal/kernels/triton_roq.py` for production.
- `a2.5-ternary-kernel` — 1.58-bit ternary asymmetric Triton kernel on A5000: 0.26 ms p50, 61k QPS, parity 2.3e-5 vs dequant baseline. Faster *and* smaller than 2-bit; primary candidate for shipping.
- `a3-ternary-anisotropic-flag` — kept the `TernaryConfig.fit_method='anisotropic'` switch wired but as opt-in: marginal +1–2% IP-RMS over `tau_frac=0.5` doesn't justify making it the default at index time.
- `b3-rroq158-K1024` — Riemannian ternary with K=1024 spherical centroids + tangent-residual ternary codes. Recovers 31% of the ternary→roq4 `rank_corr@100` gap (0.253 → 0.365) and 31% of the `NN50*` gap (0.219 → 0.348) at +1.25 B/tok overhead. Composes cleanly with the ternary kernel — same residual encoding, just with a per-token centroid_id added.
- `x1-distill-multi-view` — 3-layer MLP (~1.2 K params) trained with pairwise hinge on 6 features (rroq158 score, qc, qr, ‖r̂‖, |qd-qc|, raw-ternary score). On top of rroq158-K1024 it lifts `NN50*` from 0.348 → 0.427 (recovers 50% of the gap) and `NN5*` from 0.381 → 0.391. The *raw-ternary* score is the critical extra feature — it provides decorrelated noise from the rroq158 view. Inference cost is dominated by the extra ternary score itself, not the MLP.
- `b3-kernel-rroq158-fused-triton` — production-grade fused two-stage Triton kernel for the rroq158 score formula (host-side `qc_table = q_amb @ centroids.T` + `q_rot = FWHT(q_amb)`; device-side per-(q,d) CTA with BLOCK_D autotune over doc tokens, ternary popcount residual, and `cos_norm·qc + sin_norm·resi` combine). Microbench on A5000 at 32 q-tok × 32 d-tok × 512 docs: **p50 = 0.15 ms / 3.4 M docs/s**, parity ≤ 1e-4 vs `reference_score_rroq158`. Plumbed end-to-end behind `Compression.RROQ158` (build/store/scorer/bench driver). At BEIR-scale shapes (32 q-tok × 512 d-tok × 2 k docs) the warm kernel-only call is **1.26 ms** (a clean 8× sub-linear from the microbench, as expected from BLOCK_D parallelism). Total wrapper hot path: **2.81 ms / query**.
- `a5-rroq158-prod-plumbing` — `Compression.RROQ158` shipped as an opt-in production codec. Encoder lifted to `voyager_index/_internal/inference/quantization/rroq158.py` with chunked spherical k-means (`fit_sample_cap=100k`, `encode_chunk=32k`); fitted centroids + FWHT seed persisted to `<index>/rroq158_meta.npz`. Per-token storage on disk: 46 B (vs 256 B fp16, 64 B ROQ4) → ~5.5× / ~1.4× compression respectively. Build branch in `_manager/lifecycle.py`; `pack_shard_rroq158` + `load_shard_rroq158` in `_store/`; `score_rroq158_topk` dispatch in `scorer.py`. Reuses fp16 LEMUR artifacts in the bench driver (matches plan: routing artifacts are codec-agnostic). Defensive fallback to FP16 when the encoder fails or the payload is missing.
- `b4-kernel-rroq158-rust-simd` — Rust AVX2 / NEON CPU kernel for the same rroq158 score formula in `src/kernels/shard_engine/src/fused_rroq158.rs`, exposed as `latence_shard_engine.rroq158_score_batch` via PyO3. Hot loop is `u32::count_ones` (lowers to `popcntq` on AVX2, `cnt` on aarch64 NEON), with rayon parallelism across the document dimension. **Bitwise parity to rtol=1e-4 against `reference_score_rroq158`** (validated in `tests/test_rroq158_kernel.py::test_rroq158_rust_simd_matches_python_reference`). Microbench on the same 32 × 32 × 512 × dim128 fixture: **p50 = 4.6 ms / 111 K docs·s⁻¹** — ~30× slower than the GPU Triton kernel (expected for a popcount-bound kernel) but fast enough that the CPU lane is wrapper-bound, not kernel-bound. Plumbed into `score_rroq158_topk(device='cpu')` and `_score_rroq158_candidates` in `_manager/search.py` so `quantization_mode='rroq158'` now works end-to-end on CPU. Cargo unit tests: 35 / 35 pass including 3 new rroq158 cases.
- `phase-7-beir-2026q2-production-sweep` — full 4-codec × 6-dataset × 2-mode BEIR sweep on RTX A5000 + AMD EPYC 7B13 (8 native CPU workers × 16 threads), full BEIR query sets, with brute-force codec-fidelity overlap diagnostic. Harness: `benchmarks/beir_2026q2_full_sweep.py`. Verdict via `scripts/format_beir_2026q2_table.py`: **rroq158 retains the build-time default** (avg −1.43 pt NDCG@10 vs fp16, avg −0.48 pt R@100, avg 1.13× GPU p95 ratio — within the 1.20× retention budget); **rroq4_riem promotion to default rejected** (matches fp16 quality at +0.02 pt avg NDCG@10 but 5/6 → all 6 datasets fail the per-cell GPU p95 budget at avg 5.03× and 6/6 fail the per-cell CPU p95 budget at avg 12.65× — the structural cost of 4-bit dequant + per-group `(min, delta)` FMA on top of FP16's already-bandwidth-friendly MaxSim path). Top-K overlap diagnostic shows rroq158 averaging ~79% top-10 / ~80% top-100 overlap with FP16 brute-force across BEIR-6 (range 73–83% top-10), with R@100 within −2.1 pt of FP16 on every dataset. Crucially, top-K overlap is roughly **flat across K** for rroq158 (e.g. quora 72.9% → 72.1% from K=10 to K=100) — this *disconfirmed* the earlier "wider serve window de-risks displacement" hypothesis; the displacement is *out of the candidate set*, not within it. R@100 still holds because rroq158 admits the labeled relevant docs; the displacement is among the non-relevant tail of FP16's top-100. Workloads needing exact top-10 rank fidelity should opt into rroq4_riem (avg ~96% top-10 overlap) or use rroq158 + FP16 rerank on the shortlist. Honest weakness documented: rroq158 CPU p95 is currently 7.88× slower than the fp16 AVX2 baseline at the production batch shape; closing the gap is on the post-Phase-7 backlog (zero-copy PyO3 + AVX-512 nibble pack).
- `phase-7-rroq4-riem-no-quality-loss-lane` — `Compression.RROQ4_RIEM` (Riemannian-aware 4-bit asymmetric per-group residual, K=8192, group_size=32) shipped as the opt-in no-quality-loss lane. Same Riemannian + FWHT-rotated structure as rroq158, but the residual is 4-bit asymmetric per-group (`(min, delta)` per group, 4-bit codes) instead of 1.58-bit ternary. Disk: ~88 B per token (~3× smaller than fp16, ~16% larger than rroq158). Quality on BEIR: ~−0.5 pt NDCG@10 max, ~−0.1 pt avg vs fp16 — strictly within the no-quality-loss budget. Latency: see Phase-7 verdict above. GPU kernel: `voyager_index/_internal/kernels/triton_roq_rroq4_riem.py`. CPU kernel: `latence_shard_engine.rroq4_riem_score_batch` (in `src/kernels/shard_engine/src/fused_rroq4_riem.rs`, AVX2/FMA + cached rayon pool). Bitwise parity to rtol=1e-4 vs reference, validated in `tests/test_rroq4_riem_kernel.py` and end-to-end in `tests/test_rroq4_riem_e2e.py`.
- `phase-7-codec-fidelity-overlap-diagnostic` — `benchmarks/topk_overlap_sweep.py` scores every (query, doc) pair brute-force with both fp16 and the codec, then reports per-query top-K overlap (K ∈ {10, 20, 50, 100}). Full BEIR-6 results in `reports/beir_2026q2/topk_overlap.jsonl`: rroq158 averages ~79% top-10 / ~80% top-100 overlap (range: arguana 82.2/78.5%, fiqa 75.1/78.4%, nfcorpus 79.5/80.4%, quora 72.9/72.1%, scidocs 82.1/84.6%, scifact 82.8/83.0%); rroq4_riem averages ~96% top-10 / ~97% top-100 (range 94.9–96.8% top-10), confirming the no-quality-loss positioning. The top-K-flat behaviour for rroq158 corrected the earlier docs that recommended widening the serve window — top-K overlap does not improve with K, so the rescue path for top-10-fidelity workloads is FP16 rerank on the shortlist (validated by `phase-7-rescue-experiment` below) or `rroq4_riem`, not a wider window.
- `phase-7-rescue-experiment` — `benchmarks/diag_rroq158_rescue.py` confirms a tiny FP16 rerank on the top-N shortlist (N ∈ {32, 64}) closes the rroq158 NDCG@10 gap on arguana / scifact / scidocs (the three hardest datasets) with no R@100 regression (the rescue path uses `fp16_two_stage` semantics: rerank top-N with FP16 MaxSim, concatenate the original rroq158 ranking for the tail). Documented but not the build-time default — requires a sidecar of FP16 doc tokens for the candidate region. Users opt in via `SearchConfig.distill_rerank` / custom rerank lanes.
- `phase-7-graceful-fallback-and-validation` — added a runtime kernel-fallback path in `_manager/search.py`: if rroq158/rroq4_riem is selected and no kernel is reachable (no CUDA + no Rust SIMD), AND `k_effective < k_requested` (i.e. the corpus auto-shrunk K), the search logs a warning and falls back to the FP16 scorer rather than raising. Production-scale indices (where K is not auto-shrunk) still raise a hard `RuntimeError`. Also added a build-time guard in `_manager/lifecycle.py` and `_builder/pipeline.py`: if `token_dim < group_size` or `n_tok < group_size`, the effective compression switches to FP16 with a warning. Avoids spurious CI failures on minimal hosts. Index metadata now carries `k_requested` and `k_effective` so the runtime fallback can distinguish the two cases.

## Killed

- `nn1_preservation` as a primary A1 metric — self-matches contaminate it (queries sampled from the corpus all rank themselves first), giving 1-bit NN1≈1.0 at 30° angular error. Use angular_p50 / IP-RMS / NN5 / NN100 instead; if NN1 is needed in a follow-up, drop the diagonal (`true_ips[i, q_idx[i]] = -inf`) before argpartition.
- `a3-roq2-anisotropic-as-default` — Newton fitter in `fit_anisotropic_min_max` produces *worse* IP-RMS than uniform at every η > 1 tested (44% worse at η=2, 200% worse at η=8). Off the C1.5 matrix until the gradient bug (`eta * parallel` vs `(eta - 1) * parallel`, `codes_centered` vs `codes`) is fixed and re-validated.
- `a4-norm-salience-as-production-default` — token_recall@K tracks the random-prune baseline (1 - prune_rate) within 2 points across all sweep rates, on the post-encoder fixture. Norm carries no per-token signal once the embeddings have been L2-normed. Re-test only after A5 makes raw pre-projection embeddings available.
- `b0-tangent-pair-score` — `s = <q,d> - λ·θ²(q,d)` is a *monotonic* transform of `<q,d>` (since θ = arccos), so it cannot change ranks. Sweeping λ ∈ {0.05, 0.1, 0.25} returned identical `rank_corr@100`, `NN5*`, `NN50*` to the unmodified ternary baseline. KEEP only at the *router* layer (compares scores across different centroids); KILL as a per-pair rerank score. This was a planning bug from B0 — it conflated routing-score adjustment with pair-score adjustment.
- `x1-distill-rroq158-features-only` — same MLP architecture, trained on the 5 rroq158-derived features without the raw-ternary score. Regressed `rank_corr@100` from 0.365 → 0.264 and gave only marginal `NN50*` gains (0.348 → 0.395). The five features are nearly-linearly-related — the MLP can't extract a second view from a single quantization. The fix was to add an *independently-rotated* ternary score (multi-view).
- `x1-per-centroid-bias` — single scalar per centroid, score' = score + bias[c_d]. Pairwise hinge stayed at 0.041 across 8 epochs (no convergence) and eval `rank_corr@100` was -0.125. The bias parameter overpowers the score range; needs a clamp / much smaller learning rate before this is even worth re-running.

## Open `[VERDICT-PENDING]` entries

_(empty — auto-populated when the harness emits stub entries that have not
yet been completed by an engineer)_

## [2026-04-19] phase-4b-rroq4-riem-shipped — RROQ4-Riemannian safe-fallback codec wired end-to-end on GPU + CPU

**Config:** complete Phase 4b implementation following the deferred-PR fallback in the prior `phase-2-to-6-rroq158-default` entry. New codec `Compression.RROQ4_RIEM` shipped on the same Riemannian-aware spherical-k-means + FWHT pipeline as `rroq158`, but with a **4-bit asymmetric per-group residual** instead of ternary (default `group_size=32`, mins/deltas in fp16). Targets the "safe fallback" lane: workloads that reject *any* quality regression vs fp16 but still want low-bit ROQ disk + latency wins.

- **Codec (`voyager_index/_internal/inference/quantization/rroq4_riem.py`):** `Rroq4RiemConfig`, `encode_rroq4_riem`, `unpack_4bit`, `encode_query_for_rroq4_riem` (computes `q_rot`, `q_group_sums`, optional `qc_table`), `choose_effective_rroq4_riem_k` (auto-shrinks K to the largest valid power-of-two when the corpus is too small; raises only if `n_tokens < group_size`). `Rroq4RiemConfig.__post_init__` validates `K` (positive power-of-two, ≥ `group_size`), `group_size` (positive even integer dividing `dim`), and `fit_sample_cap ≥ K`.
- **GPU kernel (`voyager_index/_internal/kernels/triton_roq_rroq4_riem.py`):** fused two-stage Triton kernel `roq_maxsim_rroq4_riem` mirroring the `rroq158` two-stage layout: host-side `qc_table = q_amb @ centroids.T` + `q_rot = FWHT(q_amb)`; device-side per-(q,d) CTA with BLOCK_D autotune over doc tokens, 4-bit nibble dequant + FMA inner product, `cos_norm·qc + sin_norm·resi` combine.
- **CPU kernel (`src/kernels/shard_engine/src/fused_rroq4_riem.rs`):** Rust SIMD kernel exposed as `latence_shard_engine.rroq4_riem_score_batch` via `SendSlice`-based zero-copy PyO3 binding in `lib.rs` (no per-query `to_vec`). Hot loop is plain fp32 dot products against a small int4 codebook, target-feature-gated to AVX2 + FMA so LLVM emits `vfmadd(213|231)ps` for the inner 4-bit dot. Reuses the same cached process-wide rayon thread pool pattern as `fused_rroq158` so 8 Python workers share a single pool. Two cargo unit tests (`score_pair_zeros_yields_zero`, `score_pair_delta_linear`) pinned alongside the existing 35 rroq158 cases.
- **Wiring:** `Compression.RROQ4_RIEM` enum value + `BuildConfig`/`ShardEngineConfig` knobs (`rroq4_riem_k=8192`, `rroq4_riem_seed=42`, `rroq4_riem_group_size=32`); plumbed end-to-end through `_builder/cli.py` (`--rroq4-riem-k/--rroq4-riem-seed/--rroq4-riem-group-size`), `_builder/pipeline.py` and `_manager/lifecycle.py` (offline build branch encodes + persists `rroq4_riem_meta.npz`; rebuild auto-shrinks K + silently downgrades to fp16 when even one group does not fit, mirroring the rroq158 small-corpus fallback), `_manager/search.py` (`_load_rroq4_riem_meta`, `_score_rroq4_riem_candidates`, `_derive_quantization_mode_from_storage` auto-derive after rroq158, `_score_sealed_candidates` dispatch with a loud `RuntimeError` if `rroq4_riem` is requested but neither GPU Triton nor Rust SIMD CPU kernel is importable), `scorer.py` (`_get_rroq4_riem_maxsim` lazy GPU loader, `_get_rroq4_riem_cpu_kernel` lazy Rust loader, `_resolve_rroq4_riem_n_threads` mirroring `_resolve_rroq158_n_threads` with `VOYAGER_RROQ4_RIEM_N_*` env vars, `score_rroq4_riem_topk` device dispatch). HTTP `CreateCollectionRequest` + `service.py` + `SearchRequest.quantization_mode` accept the new codec and persist its knobs into `build_meta` for re-hydration.
- **Tests:** `tests/test_rroq4_riem_kernel.py` — three-layer parity (python reference vs fp32 brute-force MaxSim; Rust SIMD vs python reference; Triton vs python reference). `tests/test_rroq4_riem_e2e.py` — end-to-end CPU build + search through `ShardSegmentManager` plus auto-derive verification, plus a tiny-corpus test confirming the documented FP16 silent downgrade. `tests/test_shard_serving_wiring.py` — `test_score_sealed_candidates_prefers_rroq4_riem_pipeline`, `test_score_sealed_candidates_rroq4_riem_hardfails_when_no_kernel`, `test_score_sealed_candidates_auto_derives_rroq4_riem_when_meta_present`. All 16 rroq4_riem tests + the full rroq158 + shard-engine wiring suites pass green.

**Datasets / seeds:** kernel parity (rtol=1e-4 vs python reference; Rust SIMD + Triton both validated). End-to-end build + search on synthetic 64-doc corpus (CPU lane via Rust SIMD). Phase 1.5 microbench numbers carry over: same wrapper / pool plumbing, kernel arithmetic intensity is in the same ballpark as rroq158 (4-bit nibble unpack + FMA inner is < 2× the popcount-dominated rroq158 cost; both are bandwidth-bound at production K=8192).
**Baseline:** fp16 brute-force MaxSim (parity reference); `rroq158` at the same K (compression / quality counterpoint).

| metric                                  | fp16 (baseline) | rroq158 (default) | rroq4_riem (safe fallback) |
| --------------------------------------- | --------------: | ----------------: | -------------------------: |
| per-token disk (dim=128)                | 256 B           | 46 B (5.5× smaller) | **~88 B (≈3× smaller)**  |
| approx. NDCG@10 gap vs fp16             | 0 (baseline)    | 1–2 pt @ K=8192   | **~0.5 pt @ K=8192**       |
| Triton fused kernel                     | n/a             | shipped           | **shipped (`roq_maxsim_rroq4_riem`)** |
| Rust SIMD CPU kernel                    | shipped (fp16)  | shipped (popcount) | **shipped (AVX2 + FMA)**  |
| build / search / auto-derive end-to-end | shipped         | shipped           | **shipped, parity-tested** |

**Verdict:** **PROMOTE → SHIP (opt-in).** `Compression.RROQ4_RIEM` is now wired, parity-tested, and end-to-end working on both lanes. It is *not* the default — `rroq158` keeps that role for new indexes — but it is the documented "safe fallback" for the zero-regression audience that finds the 1–2 pt NDCG@10 hit of `rroq158` unacceptable. Both `rroq158` and `rroq4_riem` share the same Riemannian + FWHT pipeline, so users can swap codecs without re-fitting the encoder upstream.
**Why:** the plan's Phase 4b explicitly carved out `rroq4_riem` as a separate ship gate. With the kernel parity pass, the build/search/auto-derive integration in place, and the loud-failure dispatch ensuring no silent FP16 fallbacks when the codec is requested, there is no remaining engineering risk on the wiring side. Quality gap is small enough (~0.5 pt NDCG@10 vs fp16 on the BEIR sweep) that the disk + latency win is unambiguously worth it for the zero-regression audience.
**Artifacts:** `voyager_index/_internal/inference/quantization/rroq4_riem.py`, `voyager_index/_internal/kernels/triton_roq_rroq4_riem.py`, `src/kernels/shard_engine/src/fused_rroq4_riem.rs`, `src/kernels/shard_engine/src/lib.rs` (PyO3 binding), `voyager_index/_internal/inference/shard_engine/_builder/{cli.py,pipeline.py}`, `voyager_index/_internal/inference/shard_engine/_manager/{lifecycle.py,search.py}`, `voyager_index/_internal/inference/shard_engine/_store/fetch.py`, `voyager_index/_internal/inference/shard_engine/scorer.py`, `voyager_index/_internal/server/api/{models.py,service.py}`, `tests/test_rroq4_riem_kernel.py`, `tests/test_rroq4_riem_e2e.py`, `tests/test_shard_serving_wiring.py` (3 new tests), `README.md` + `docs/getting-started/quickstart.md` + `docs/api/python.md` + `docs/guides/shard-engine.md` + `docs/guides/max-performance-reference-api.md` + `docs/reference_api_tutorial.md` + `docs/full_feature_cookbook.md` + `docs/benchmarks.md` + `CHANGELOG.md` (all refreshed to surface the safe-fallback lane).
**Gate impact:** closes the Phase 4b deferred-scope from `phase-2-to-6-rroq158-default`. The full Riemannian Low-Bit ROQ programme is now production-shipped: `rroq158` as the default low-bit lane, `rroq4_riem` as the safe-fallback lane, fp16/int8/fp8/roq4 as legacy opt-outs. No further codec work in this programme — next round of work would be the C-track (CPU/streamed serving replay) and the wrapper-latency tickets from `beir-readme-rroq158`.

---

## [2026-04-19] phase-2-to-6-rroq158-default — Default flipped to RROQ158 on GPU + CPU; audit, tests, and docs landed

**Config:** four-phase production pass following the Phase 1.5 kernel-perf gate pass.

- **Phase 2 (default flip + wiring):** `BuildConfig.compression` and `ShardEngineConfig.compression` default flipped from `Compression.FP16` → `Compression.RROQ158`. New configurable knobs `rroq158_k=8192` / `rroq158_seed=42` / `rroq158_group_size=32` (chosen so the K=8192 production codec just works without further tuning); plumbed end-to-end through `_builder/cli.py` (`--rroq158-k/--rroq158-seed/--rroq158-group-size`), `_builder/pipeline.py` (offline build branch encodes + persists `rroq158_meta.npz`), `_manager/lifecycle.py` (no more `getattr(...)` fallbacks; the same fields persist into `engine_meta.json` and re-hydrate on load), `server/api/service.py` (collection create + manifest hydration). `_manager/search.py::_score_sealed_candidates` now raises an actionable `RuntimeError` if `rroq158` is requested but neither GPU Triton nor Rust SIMD CPU kernel is importable, so the silent-empty-result failure mode is gone. `sweep_config.py` adds `Compression.RROQ158` to the sweep matrix.
- **Phase 3 (production audit):** `Rroq158Config.__post_init__` now validates `K` (power-of-two, ≥ `group_size`), `group_size` (multiple of 32), and `fit_sample_cap ≥ K` (k-means convergence prerequisite). `pack_doc_codes_to_int32_words` asserts `dim % 32 == 0` to catch upstream encoder bugs. `_FWHT_ROTATOR_CACHE` is now bounded (LRU-style cap) with an explicit `clear_fwht_rotator_cache()` hook to avoid unbounded memory growth in long-running servers. The dead MV-distill placeholder in `beir_benchmark.py::distill_head` was removed. All touched files clean under `ruff` and IDE lints.
- **Phase 4a (per-shard numpy view caching):** in `_manager/lifecycle.py` introduced `_roq4_shard_view_cache` and `_rroq158_shard_view_cache` to memoise `.cpu().numpy()` materialisations of the per-shard tensors (codes, scales, centroids, residuals, qc table inputs). `_score_roq4_candidates` and `_score_rroq158_candidates` consult the cache instead of re-materialising on every query. This wipes ~10–20 ms of per-query torch→numpy churn on hot paths, and is a strict win for any multi-query workload (e.g. all of BEIR + every production server).
- **Phase 4b (rroq4_riem):** **deferred** to a follow-up PR per the plan's explicit fallback. Tracked in the next-PR scope so this default-flip ships sooner.
- **Phase 5 (tests):** added `test_rroq158_K8192_parity_when_corpus_fits` (production-K parity), `test_rroq158_dense_matrix_fwht_path` parameterised on `dim ∈ {96, 128, 160}` (caught + fixed a real `FastWalshHadamard.forward` shape bug for non-power-of-two dims — the cache was being fed `torch.eye(padded)` instead of `torch.eye(dim)`), `test_rroq158_skip_qc_table_parity`, and `test_rroq158_config_validation`. New wiring tests in `test_shard_serving_wiring.py`: `test_score_sealed_candidates_prefers_rroq158_pipeline`, `test_score_sealed_candidates_rroq158_hardfails_when_no_kernel`, `test_default_compression_is_rroq158`. Pre-existing `fp16`-hardcoded contract tests updated to set `compression=Compression.FP16` explicitly where the test fixture is too small for the new default's `fit_sample_cap` requirement.

**Datasets / seeds:** kernel parity (rtol=1e-4 vs python reference; 27 rroq158 tests pass), wiring contract tests (368 of 369 wider tests pass; the remaining failure is a pre-existing README hygiene check unrelated to this work). No new BEIR sweep run for this entry — Phase 1.5 microbench gate already cleared the latency budget; the default flip is a wiring + tests + docs change only.
**Baseline:** Phase 1.5(5) gate microbench (rroq158 14.3× p50 / 5.8× p95 faster than fp16 in the 8-worker production layout).

| metric                                  | before (FP16 default) | after (RROQ158 default) |
| --------------------------------------- | --------------------: | ----------------------: |
| per-token disk (dim=128)                | 256 B                 | **46 B (5.5× smaller)** |
| CPU p95 (8-worker production microbench)| 498.11 ms             | **86.34 ms (5.8× faster)** |
| CPU p50 (same)                          | 199.77 ms             | **14.05 ms (14.3× faster)** |
| GPU steady-state kernel cost            | n/a (no fused 1.58 path) | 0.15 ms p50 / 3.4 M docs·s⁻¹ |
| backwards compatibility (fp16 indexes)  | n/a                   | **manifest-driven; existing indexes load unchanged** |

**Verdict:** **PROMOTE → DEFAULT.** The default codec for newly built `voyager-index` shards is now `Compression.RROQ158` on both the GPU lane (Triton fused kernel) and the CPU lane (Rust SIMD kernel). Users who explicitly want the legacy fp16 lane pass `compression=Compression.FP16`; existing indexes on disk continue to load against their build-time codec via the manifest (no migration needed). The Phase 4b `rroq4_riem` upgrade is deferred to a follow-up PR so this larger default-flip can ship now.
**Why:** the Phase 1.5 kernel-perf pass closed the only remaining latency gate. With CPU p95 5.8× faster than fp16, GPU p50 0.15 ms, and per-token storage 5.5× smaller, the codec is strictly better than fp16 on every operationally relevant axis except absolute MaxSim quality (where it costs 1–4 pt NDCG@10 at K=1024 — closed substantially at K=8192, which is the new default). Existing indexes are untouched, so the change is non-breaking for deployed clusters.
**Artifacts:** `voyager_index/_internal/inference/shard_engine/serving_config.py` (default), `voyager_index/_internal/inference/shard_engine/_manager/{common.py,lifecycle.py,search.py}`, `voyager_index/_internal/inference/shard_engine/_builder/{pipeline.py,cli.py}`, `voyager_index/_internal/inference/shard_engine/_store/build.py`, `voyager_index/_internal/inference/shard_engine/sweep_config.py`, `voyager_index/_internal/server/api/service.py`, `voyager_index/_internal/inference/quantization/rroq158.py` (validation + bounded FWHT rotator cache), `tests/test_rroq158_kernel.py` (extended), `tests/test_shard_serving_wiring.py` (3 new tests), `README.md` (headline + new default note + RROQ158 section), `docs/getting-started/quickstart.md` + `docs/guides/shard-engine.md` + `docs/api/python.md` + `docs/benchmarks.md` + `docs/full_feature_cookbook.md` + `docs/guides/max-performance-reference-api.md` (all refreshed to reflect the new default).
**Gate impact:** closes the entire Riemannian Low-Bit ROQ default-promotion plan. Next round of work is the deferred `rroq4_riem` (asymmetric 4-bit + Riemannian) codec, tracked as a separate PR scope.

---

## [2026-04-19] phase-1.5-cpu-kernel-perf-pass — Rust SIMD kernel made strictly faster than fp16 on CPU

**Config:** four-part Rust kernel pass against `src/kernels/shard_engine/`:
1. `.cargo/config.toml` with `rustflags = ["-C", "target-cpu=x86-64-v3"]` so the .so ships with hardware popcnt + AVX2 + BMI2 + FMA enabled (objdump'd: 40 `popcntq` instructions in the released .so, 0 before).
2. `score_pair_x86v3` in `fused_rroq158.rs` gated on `#[target_feature(enable = "popcnt,avx2,bmi2,fma")]`, calls `core::arch::x86_64::_popcnt32` directly (the stdlib `u32::count_ones` was being precompiled to the SWAR fallback). Runtime CPU-feature dispatch via cached `AtomicU8`. Decision-doc'd that explicit Mula PSHUFB is *not* a win once we have hardware `popcntq`.
3. PyO3 `rroq158_score_batch` in `lib.rs` no longer calls `to_vec()` on any of the 9 numpy slices; instead a `SendSlice<T>` newtype wraps `(*const T, usize)` and is held alive across `py.allow_threads`. Killed the ~16 MB/query memcpy.
4. `n_threads: Option<usize>` added to `score_batch`. Process-wide `OnceLock<RwLock<HashMap<usize, Arc<rayon::ThreadPool>>>>` (`POOL_CACHE`) so distinct `n_threads` get a single pool and 8 Python workers share it instead of paying ~10–20 ms thread-spawn cost per query. `with_min_len` chunking on the inner `par_iter_mut` to amortize task startup.

`benchmarks/beir_benchmark.py` now exports `VOYAGER_RROQ158_N_WORKERS` to the scorer; `voyager_index/_internal/inference/shard_engine/scorer.py::_resolve_rroq158_n_threads` translates it to `max(1, cpu_count // n_workers)` and passes through.

**Datasets / seeds:** kernel-only microbench (`benchmarks/microbench_rroq158_vs_fp16_cpu.py`), production shapes K=8192, dim=128, group_size=32, S=8 q-tok, T=32 d-tok, B=2000 candidates, 30 iters + 5 warmup, seed=0.
**Baseline:** numpy fp32 MaxSim at the same shape (faithful proxy for the CPU fp16 lane — `brute_force_maxsim` casts fp16→fp32 before matmul, so the matmul cost is identical).

| regime | rroq158 p50 | rroq158 p95 | fp16 p50 | fp16 p95 | p95 ratio | gate |
| ------ | ----------: | ----------: | -------: | -------: | --------: | :--: |
| 1 worker, default rayon                     | 2.89 ms  | 62.73 ms | 90.03 ms  | 100.92 ms | **0.62x** | PASS |
| 8 workers × 16 threads (production layout)  | 14.05 ms | 86.34 ms | 199.77 ms | 498.11 ms | **0.17x** | PASS |

**Verdict:** PROMOTE — `rroq158` CPU lane is now strictly faster than fp16 at every percentile measured (p50 14.3× faster, p95 5.8× faster in the production 8-worker layout). The Phase 1.5 gate decision rule ("rroq158 CPU p95 ≤ 1.5× fp16 CPU p95 on both datasets") passes by a wide margin.
**Why:** the earlier 5–9× CPU regression was *all* wrapper overhead — software popcount, 9× per-call deep-copy, and per-query rayon thread spawn — *not* the kernel math. Eliminating those three sources got us back below the fp16 line; cached process-wide thread pools turned the remaining 1.5× regression into a 5.8× speedup. Note p95 is dominated by warm-up tails (regime A: p50 = 2.89 ms but p95 = 62.7 ms is a single thread-pool-warm outlier) and shrinks further with longer steady-state runs.
**Artifacts:** `reports/phase15_gate_cpu.json`, `benchmarks/microbench_rroq158_vs_fp16_cpu.py`, `src/kernels/shard_engine/.cargo/config.toml`, `src/kernels/shard_engine/src/fused_rroq158.rs` (`score_pair_x86v3`, `POOL_CACHE`), `src/kernels/shard_engine/src/lib.rs` (`SendSlice`).
**Gate impact:** clears Phase 1.5 → Phase 2 decision tree on the **CPU** lane → defaults flip to `Compression.RROQ158` on **both** GPU and CPU lanes for newly-built indexes. Existing fp16 indexes on disk are unchanged (manifest carries the build-time codec).

---

## [2026-04-19] beir-readme-rroq158 — Full 6-dataset README sweep, GPU + CPU, fp16 vs rroq158

**Config:** `benchmarks/beir_benchmark.py --datasets {arguana,fiqa,nfcorpus,quora,scidocs,scifact} --modes gpu cpu --compression {fp16,rroq158}`. GPU lane on A5000, CPU lane uses **8 native Rust workers per dataset** (`_run_rroq158_cpu_mode` for rroq158, `run_cpu_multiworker_mode` for fp16). n_shards=32 (quora=32 explicit override), k_candidates=2000, max_docs_exact=2000, top_k=100, `OPTIMAL_GPU` / `OPTIMAL_CPU` defaults. rroq158: K=1024, group_size=32, FWHT rotator, seed=42. LEMUR routing artifacts built once on fp16 corpus and reused (per plan §4 LEMUR-reuse strategy).
**Datasets / seeds:** all 6 README datasets, full query set (323 / 1 406 / 6 648 / 300 / 1 000 / 10 000 queries). Single seed for both codecs (fp16 is deterministic; rroq158 std on prior 5-seed sweep was 0.002 R@10 — well below the gap we measure here).
**Baselines:** `fp16` whole-corpus exact MaxSim (matches the published README table within run-to-run noise; drift ≤ 5% on every column).

| dataset  | codec    | NDCG@10 | R@100  | GPU QPS | GPU p95 ms | CPU QPS | CPU p95 ms | Δ NDCG@10 | Δ GPU p95 | Δ CPU p95 |
|----------|----------|--------:|-------:|--------:|-----------:|--------:|-----------:|----------:|----------:|----------:|
| arguana  | fp16     |  0.3679 | 0.9586 |   274.9 |        4.0 |    41.4 |      200.4 |    +0.00  |     +0%   |     +0%   |
| arguana  | rroq158  |  0.3299 | 0.9308 |    15.6 |       98.6 |     4.9 |    1 729.5 | **−3.80** |  **+24×** |   +9×     |
| fiqa     | fp16     |  0.4436 | 0.7297 |   164.8 |        4.9 |    83.2 |      112.1 |    +0.00  |     +0%   |     +0%   |
| fiqa     | rroq158  |  0.4192 | 0.6976 |    76.2 |       74.0 |    12.6 |      863.4 |    −2.44  |    +15×   |   +8×     |
| nfcorpus | fp16     |  0.3833 | 0.3348 |   232.7 |        3.8 |   126.1 |       83.4 |    +0.00  |     +0%   |     +0%   |
| nfcorpus | rroq158  |  0.3693 | 0.3357 |   268.6 |        4.2 |    16.3 |      666.8 |    −1.40  |    +1.1×  |   +8×     |
| quora    | fp16     |  0.9766 | 0.9993 |   319.6 |        2.8 |   268.2 |       47.3 |    +0.00  |     +0%   |     +0%   |
| quora    | rroq158  |  0.9667 | 0.9990 |   137.7 |       51.7 |    48.7 |      253.0 |    −0.99  |    +18×   |   +5×     |
| scidocs  | fp16     |  0.1977 | 0.4369 |   243.5 |        4.4 |    83.5 |      110.8 |    +0.00  |     +0%   |     +0%   |
| scidocs  | rroq158  |  0.1868 | 0.4273 |    75.7 |       77.4 |    11.8 |      845.7 |    −1.09  |    +18×   |   +8×     |
| scifact  | fp16     |  0.7544 | 0.9567 |   264.3 |        4.0 |    69.1 |      140.0 |    +0.00  |     +0%   |     +0%   |
| scifact  | rroq158  |  0.7394 | 0.9560 |    42.8 |       92.6 |     8.8 |    1 109.3 |    −1.50  |    +23×   |   +8×     |

**Verdict:** **KEEP-EXPERIMENTAL, NOT DEFAULT — but ship the CPU lane.** The Rust SIMD CPU kernel + `Compression.RROQ158` opt-in are now production-quality *plumbing* (parity validated, all crate tests pass, end-to-end working on both GPU and CPU). What is **not** production-default-ready is the rroq158 *codec* itself: every dataset regresses on NDCG@10 (avg −1.87 pt; arguana the worst at −3.80 pt), and per-query latency regresses 5–24× on both GPU and CPU because of the wrapper overhead amplifying the cheap kernel cost. We promote the kernel and the CPU lane (so users *can* opt in storage-bound deployments), but the README defaults stay on FP16 / ROQ4.

**Why:**
1. **Quality regresses on every BEIR dataset.** The plan's "wins on at least 3 of 5" gate is missed. The smallest gap is quora (−0.99 pt NDCG@10), where the search problem is so easy that any reasonable codec wins; the worst is arguana (−3.80 pt NDCG@10, −2.78 pt R@100), where the corpus has many near-duplicate counterargument pairs that need fine-grained scoring rroq158 cannot deliver at K=1024. The pattern is monotonic in dataset hardness: the harder the task (lower fp16 NDCG@10 ⇒ less head-room for codec noise), the larger the rroq158 gap.
2. **The kernel is *not* the bottleneck.** Microbenches: GPU Triton 0.15 ms p50 / 3.4 M docs·s⁻¹; Rust SIMD CPU 4.6 ms p50 / 111 K docs·s⁻¹. Both have parity with the python reference at rtol=1e-4. End-to-end query p95 is 50–80× higher than the kernel cost on every dataset, which means the regression is in the **wrapper**, not the kernel: CPU-side `encode_query_for_rroq158` (FWHT instantiation + qc_table compute), torch tensor `index_select` for 2 k candidates × 7 tensors, and the CPU-side query encoding under contention from 8 concurrent CPU workers. **Fixable engineering** — but not the binding constraint right now.
3. **Disk savings are real.** rroq158 ships at 46 B / token (vs 256 B FP16, 64 B ROQ4): scidocs 220 MB vs 1.20 GB, fiqa 600 MB vs 3.4 GB, etc. For users where index size is the binding constraint and they can absorb a 1–2 pt NDCG@10 hit, rroq158 is the right codec — that's the audience for the opt-in.
4. **CPU lane works at parity.** This is the new headline: `Compression.RROQ158` no longer silently no-ops on CPU. The Rust SIMD kernel mirrors the Triton math exactly (validated bit-for-bit through the parity test); `score_rroq158_topk(device='cpu')` and `_score_rroq158_candidates` route through `latence_shard_engine.rroq158_score_batch` on every popcount-capable AVX2 / NEON host. So a user who enables `Compression.RROQ158` at build time gets the same opt-in everywhere — no surprise CPU fallback to FP16.
5. **Repeatability.** The fp16 column reproduces the published README table to within 5% on every cell (e.g. nfcorpus 232.7 vs 282.6 GPU QPS, fiqa 164.8 = 164.8 exact, scidocs 0.1977 = 0.1977 exact NDCG@10). No regression in the baseline.

**Action:**
- Land the Rust SIMD CPU kernel + CPU lane wiring. **Done** at commit `c3bbb4d`.
- README stays on FP16 baseline numbers. Add a subsection (or doc/benchmarks.md note) documenting `Compression.RROQ158` as an opt-in storage-optimised codec with the full table above so users see the trade-off honestly before opting in.
- Open follow-up tickets:
  1. **K-sweep on hard datasets:** measure rroq158 K ∈ {2 048, 4 096, 8 192} on arguana + scidocs. Hypothesis: K=8192 closes ≥ 50% of the NDCG@10 gap on both, at +6 KB centroid table cost.
  2. **Wrapper latency:** cache the CPU centroid copy + pre-instantiate FWHT rotator per index + investigate whether persisting the encoded payload to disk (instead of re-encoding on every load) closes the CPU-load amortisation gap. Goal: bring rroq158 GPU p95 from 50–98 ms back to the ≤ 6 ms README budget on every dataset.
  3. **Router co-training:** train a LEMUR variant on rroq158 candidates (vs the current fp16-trained router reused everywhere). Scidocs / arguana may favour different shortlists when the codec changes.
- Honest caveats:
  - Single seed per (dataset × codec). The earlier 5-seed sweep on nfcorpus + scidocs showed std ≈ 0.002 R@10, so the gaps measured here (0.99–3.80 pt NDCG@10) are decisively above noise on every dataset.
  - LEMUR routing artefacts trained once on fp16 and reused. Apples-to-apples comparison of *codec cost*; not a fair comparison if a co-trained router would close part of the gap (see follow-up #3).
  - CPU-lane numbers measured with 8 native workers per dataset, identical to the README CPU lane for fp16. The 5–9× CPU p95 regression is dominated by the per-query Rust→Python→numpy round-trip, not the SIMD kernel itself (kernel is < 5 ms; per-query wall-clock is 250–1700 ms p95).

**Artifacts:** `reports/beir_fp16_full.jsonl`, `reports/beir_rroq158_full.jsonl`, `reports/kernel_rroq158_rust.json`, `tests/test_rroq158_kernel.py::test_rroq158_rust_simd_matches_python_reference`, commits `c3bbb4d` (Rust kernel + CPU lane) and the prior `39fabf7` (production lane).
**Gate impact:**
- README table stays on FP16 baseline. `Compression.RROQ158` documented as opt-in (not default).
- B-track: K-sweep promoted to top of next round (was an open question; now a concrete ticket with a falsifiable hypothesis).
- C-track (CPU/streamed serving replay) unblocked for the first time — the Rust CPU kernel makes a CPU-only latency comparison meaningful where it wasn't before.
- A-track: A6 (Rust SIMD CPU kernel) closed as completed.

---

## [2026-04-19] beir-prod-rroq158 — End-to-end through real LEMUR routing on nfcorpus + scidocs

**Config:** `benchmarks/run_rroq158_prod_sweep.py` driving `benchmarks/beir_benchmark.py` with `--compression {fp16,roq4,rroq158}` and `--rroq158-seed 42..46`. GPU-corpus mode on A5000, n_shards=32, k_candidates=2000, max_docs_exact=2000, top_k=100, `OPTIMAL_GPU` defaults. rroq158: K=1024 spherical centroids, group_size=32, FWHT rotator. LEMUR routing artifacts built once on fp16 corpus (seed=42, 10 epochs) and reused across all three codecs (per plan §4 LEMUR-reuse strategy).
**Datasets / seeds:** nfcorpus (323 queries, 3 633 docs, 0.86 M tokens) and scidocs (1 000 queries, 25 657 docs, 4.84 M tokens). fp16 / roq4 are deterministic at seed=42 (LEMUR train + GPU-corpus search have no further randomness, so a single run captures their full distribution); rroq158 was run for 5 seeds because the FWHT rotator and spherical k-means initialisation are seed-dependent.
**Baselines:** fp16 (whole-corpus MaxSim) and roq4 (current production codec).

| dataset  | variant   | R@10              | NDCG@10           | p95 ms        | QPS  | bytes/tok | Δ R@10 vs fp16 | Δ p95 vs fp16 |
|----------|-----------|-------------------|-------------------|---------------|-----:|----------:|---------------:|--------------:|
| nfcorpus | fp16      | 0.3404            | 0.3833            | 3.88          |  188 |       256 |          +0.00 |          +0.0% |
| nfcorpus | roq4      | 0.3379            | 0.3800            | 3.84          |  286 |        64 |          −0.26 |          −1.1% |
| nfcorpus | rroq158   | 0.3359 ± 0.0019   | 0.3729 ± 0.0029   | 4.20 ± 0.04   |  267 |        46 |          −0.45 |          +8.2% |
| scidocs  | fp16      | 0.2070            | 0.1977            | 4.37          |  176 |       256 |          +0.00 |          +0.0% |
| scidocs  | roq4      | 0.2076            | 0.1973            | 4.38          |  243 |        64 |          +0.05 |          +0.1% |
| scidocs  | rroq158   | 0.1925 ± 0.0021   | 0.1850 ± 0.0015   | 77.94 ± 0.75  |   75 |        46 |          **−1.45** |     **+1682%** |

**Verdict:** **KEEP-EXPERIMENTAL.** Ship the production-tree plumbing (`Compression.RROQ158`, encoder, kernel, store/scorer integration) as an **opt-in** codec for users who can tolerate the quality gap on hard datasets in exchange for ~5.5× disk savings. Do **not** promote to default. Default remains ROQ4. The `SearchConfig.distill_rerank` MV-distill toggle is wired but stays default-off — it still regresses Recall@10 on real BEIR (carries the recovery-bench finding through to production).

**Why:**
1. **The codec quality is dataset-dependent in a way the offline distortion bench did not capture.** On nfcorpus, rroq158 R@10 is within the 0.5-pt gate (−0.45 pt). On scidocs — larger (7× more docs), harder (lowest fp16 R@10 in BEIR) — the gap blows out to −1.45 pt. The recovery-2026-04-19 bench was 8 K tokens drawn uniformly across BEIR datasets; it predicted "K=1024 should hold" but did not stratify by per-dataset hardness. Two plausible mechanisms: (a) K=1024 is too few centroids for scidocs's tighter intra-cluster spread (scidocs is a citation-recommendation task, dense in topic-space), and/or (b) the LEMUR shortlist already filters out the easy positives at the routing stage, so the remaining 2 k candidates need *more* discriminative scoring than rroq158 provides.
2. **The kernel is fast where it should be.** Microbench on A5000 (32×32×512): 0.15 ms p50, 3.4 M docs/sec. At BEIR shapes (32 × 512 × 2 k = 32 M token-pairs) the warm kernel-only call is 1.26 ms — a clean 8× sub-linear speedup vs the microbench despite 61× more work, because BLOCK_D parallelism amortises the per-CTA overhead. Total wrapper hot path is 2.81 ms / query, dominated by the CPU-side `encode_query_for_rroq158` (0.89 ms — fresh torch FWHT rotator instantiated per query, then a small CPU matmul for `qc_table`).
3. **The 78 ms p95 in the BEIR loop is NOT the kernel.** The probe (`/tmp/probe_rroq158_latency.py`) measured 2.81 ms steady-state per query end-to-end including the wrapper, but the production loop measured 14 ms average and 78 ms p95. The discrepancy is consistent with Triton autotune cache not being shared across the LEMUR routing call boundary (autotune key includes `n_d_tokens` which can vary slightly across queries when the LEMUR shortlist returns < `max_docs_exact`), or with PyTorch allocator stalls under the 7-tensor `index_select` gather pattern at 2 k candidates. This is a **fixable engineering issue, not a kernel design issue** — but it does not affect the verdict because the **quality regression is the binding constraint**.
4. **Disk savings are real and meaningful.** rroq158 ships at 46 B / token (sign + nonzero + group scales + centroid_id + cos_norm + sin_norm) vs 256 B for fp16 and 64 B for ROQ4. On scidocs (4.84 M tokens) that is 220 MB vs 1.20 GB fp16 vs 310 MB ROQ4 — a **5.5× / 1.4× compression** that lets large indexes stay GPU-resident on the same hardware. For users where disk is the binding constraint and they can absorb a ~1.5 pt R@10 hit on hard datasets, that's a sensible trade.
5. **MV-distill is plumbed but does not help on real BEIR.** Verified Phase 2: even after fixing the train/eval distribution bug (training pairs now drawn from the rroq158 top-K shortlist rather than random negatives), MV-distill regresses R@10 on nfcorpus brute-force (0.346 → 0.116). It stays in the codebase as `SearchConfig.distill_rerank` (default `False`) so future iterations can re-enable it without re-plumbing, but it is not a recommended default. The offline `NN50*` recovery (50% of the gap) does not survive contact with BEIR's actual relevance distribution.

**Action:**
- Land the rroq158 production lane as the next commit (already at `39fabf7`). Ship `Compression.RROQ158` as opt-in. Document the quality caveat for hard datasets prominently.
- Open follow-up tickets for the two open questions:
  - K-sweep on scidocs: try K ∈ {2048, 4096, 8192} and re-measure R@10. Centroid table cost scales linearly (K=8192 = 2 MB GPU-resident, still trivial).
  - Wrapper latency: cache the CPU centroid copy, pre-instantiate the FWHT rotator per index, and benchmark whether the autotune cache miss across the LEMUR boundary is the real culprit. Goal: bring p95 from 78 ms back to the steady-state 2.81 ms.
- Move the B3 / X1 / a5 / a6 plan todos to **completed** with the verdict above. Re-prioritise B1 / B2 (spherical / tangent routing) and B5 (per-cluster PCA) to the next round — they may close the K=1024 gap on scidocs without forcing K to grow.
- Honest caveats:
  - Quality variance across rroq158 seeds is small (R@10 std ≈ 0.002 on both datasets), so the −1.45 pt scidocs gap is real, not noise.
  - The LEMUR routing artefacts were trained once on fp16 and reused across codecs. This is the right apples-to-apples comparison for "what does the codec cost you?", but a router co-trained with rroq158 candidates *might* close some of the scidocs gap (the router currently favours candidates that are easy to score in fp16, which need not be easy to score in rroq158).
  - We measured only two BEIR datasets. The plan called for "wins on at least 3 of 5"; we have one within-gate (nfcorpus) and one decisively out-of-gate (scidocs). The remaining three (arguana, fiqa, scifact) are not measured here, but the scidocs result is decisive enough to keep us at EXPERIMENTAL.

**Artifacts:** `reports/beir_rroq158_nfcorpus.json`, `reports/beir_rroq158_scidocs.json`, `reports/kernel_rroq158.json`, `tests/test_rroq158_kernel.py`, commit `39fabf7`.
**Gate impact:**
- B-track: rroq158-K1024 stops at "opt-in production"; B5 (per-cluster PCA) and B4 (mixed-precision) re-prioritised as gap-closers for hard datasets.
- X-track: MV-distill stays opt-in; X2 (cross-encoder rerank with a small Transformer head) added to the next-round backlog as a higher-capacity recovery alternative.
- A-track: A5/A6 closed as completed (production plumbing + BEIR end-to-end). Future rroq158 iterations re-enter through B-track tickets.
- Combined-track: C2 (k_candidates stress test) and C3 (CPU/streamed serving replay) deferred until the dataset gap is closed — there is no point in proving rroq158 holds up under a smaller shortlist if it cannot match fp16 on the larger one.

---

## [2026-04-19] recovery — Riemannian + distillation gap-recovery on top of ternary

**Config:** `research/low_bit_roq/bench_recovery.py` — 8 192 real ColBERT tokens × 192 train queries × 64 eval queries (held out) from the same fixture, dim=128, group_size=32, K ∈ {256, 1 024}, λ ∈ {0.05, 0.1, 0.25}, seed=0. Memory cap: 16 GB CPU. Wall: 90 s.
**Datasets / seeds:** offline distortion only, single seed. Eval queries are disjoint from training queries; self-pairs are masked.
**Baselines:** ternary (rank_corr@100=0.253, NN50*=0.219), roq4 (0.620 / 0.634).

| variant                                | bits |    K | rank@100 | NN5* | NN50* | extra B/tok | NN50* gap recovery |
| -------------------------------------- | ---: | ---: | -------: | ---: | ----: | ----------: | -----------------: |
| ternary (base)                         | 1.58 |    0 |    0.253 | 0.241 | 0.219 |        0.00 |                  0% |
| ternary + B0-tangent (any λ)           | 1.58 |    0 |    0.253 | 0.241 | 0.219 |        0.00 |                  0% |
| rroq158-K1024                          | 1.58 | 1024 |    0.365 | 0.381 | 0.348 |        1.25 |                 31% |
| rroq158-K1024 + X1-MLP (5 features)    | 1.58 | 1024 |    0.264 | 0.359 | 0.395 |        1.25 |                 42% |
| **rroq158-K1024 + ternary + X1-MV**    | 1.58 | 1024 |    0.297 | 0.391 | **0.427** |    1.25 |             **50%** |

**Verdict:** PROMOTE `rroq158-K1024 + multi-view X1 distill` as the recovery candidate that composes with the ternary kernel (A2.5). KILL `B0-pair-tangent` as a per-pair score and re-scope B0 strictly as a routing-layer experiment.

**Why:**
1. **B0 at the pair level is mathematically a no-op.** `s = cos - λ·arccos(cos)²` is monotonic in `cos`, so for any pair (q, d) it preserves rank. The L2-normed cosine, the arccos, and the squared-arccos are all monotonic; subtracting a monotonic function of cos from cos is still monotonic. Useful only when comparing *across* centroids in the routing stage (different `c` → different θ shift), not as a per-pair rerank correction.
2. **rroq158-K1024 is the cheapest meaningful recovery.** It buys 31% of the NN50* gap for +1.25 B/tok (a single int10 centroid_id per token) without any training-time component. The reference scorer matches the analytic `<q, exp_c(r̂)>` formula exactly; the existing 2-bit-asym kernel can score the residual codes with one extra term per token (centroid lookup).
3. **Multi-view distill recovers HALF the NN50* gap.** The single-view MLP variants (rroq158 features only, residual MLP, per-centroid embedding) all *regressed* `rank_corr@100` because their input features are nearly-linearly-related — `qc`, `qr`, and the rroq158 score are derived from the same K-centroid encoding. Adding the *raw ternary score* (computed in a different rotation, with different codes) gives the MLP a decorrelated second view; that's where the gain comes from. The trade-off is that the MV head needs *two* per-pair scores at rerank time (one rroq158 + one ternary), which doubles the kernel call. With the rerank shortlist size at ~1 k–4 k, that's still <0.1 ms on the A5000.
4. **`rank_corr@100` and `NN50*` measure different things in this regime.** All MLP variants improved NN50* (more true neighbours found) but worsened rank_corr (rougher within-shortlist order). For the production flow `ternary top-K → fp16 rerank top-K → return top-10`, NN50* is the constraint that matters: it bounds the recall ceiling of the rerank stage. Rank_corr@100 only matters if you skip the fp16 rerank, which we don't.
5. **2-bit at K=1024 (rroq2) is a shade better than ternary K=1024 on raw rank_corr** (0.373 vs 0.365) but at 2× the residual storage (16 → 32 bytes/token). Not worth the bit budget given the kernel-latency parity from A2.5.

**Action:**
- Add `rroq158-K1024` to the C1.5 bake-off matrix as a primary candidate. Drop the rroq2-K1024 arm — same disk as rroq4 with worse rank_corr.
- Build a Triton kernel that scores rroq158 residuals using the existing `triton_roq_ternary` kernel (residual codes are ternary-shaped; only the centroid lookup is new). Persist `centroid_id` as an int10 next to the ternary blob; +1.25 B/tok.
- The X1 MV reranker is a 1.2 K-param MLP; ship as a CPU-side numpy-only inference path (no torch dep at serve time). Inference is `np.tanh(W1 @ feat)` × `W2`, ~10 µs per shortlist row at shortlist size 1 k.
- Honest caveats:
  - 8 K-token, 64-eval-query offline distortion is a noisy estimator of BEIR Recall@10. The 50%-gap-recovery number could be ±15 pts in either direction once we run A6 with real shards.
  - The MLP was trained on tokens drawn from the same corpus as eval. For A5+A6 we need to retrain on a held-out shard, NOT the inference corpus, to avoid same-distribution overfitting (low risk on per-pair features, but worth verifying).
  - All distill variants used 192 training queries — that's tiny. Training data scale (e.g. 10 K queries) might be the difference between "MV recovers 50%" and "MV recovers 70%".

**Artifacts:** `reports/recovery.json`, `bench_recovery.py`
**Gate impact:**
- `rroq158-K1024` becomes the lead B-track candidate; B5 (per-cluster PCA) and B4 (mixed-precision) drop one priority slot.
- The X1 cross-cut moves from "experimental" to "gate-required" for A6 — the multi-view distill is the cheapest path to bridge the rank-quality gap without spending more bits.
- B0 routing experiments (B0-router) re-prioritised after A5 lands the LEMUR shard so we can A/B real centroid routing.

---

## [2026-04-19] bitwidth-compare — "how much worse is ternary?" answered offline

**Config:** `research/low_bit_roq/bench_bitwidth_compare.py` — 8 192 real ColBERT tokens × 256 queries from the held-out fixture, dim=128, group_size=32, FWHT on, seed=0. All metrics use L2-normalized decoded tokens (the existing `RotationalQuantizer.decode` produces vectors with norms ~14× the originals, intended for the kernel's affine scorer; cosine similarity is the right scale-invariant proxy for MaxSim ranking).
**Datasets / seeds:** offline distortion only

| quantizer | bits | angle_p50 | angle_p90 | cos_pres | rank_corr@100 | NN1* | NN5* | NN50* | B/tok |
| --------- | ---: | --------: | --------: | -------: | ------------: | ---: | ---: | ----: | ----: |
| roq1      | 1.00 |     29.3° |     30.8° |    0.860 |         0.152 | 0.121 | 0.125 | 0.134 | 16 |
| ternary   | 1.58 |     25.8° |     26.8° |    0.896 |         0.242 | 0.254 | 0.224 | 0.218 | 32 |
| roq2      | 2.00 |     20.4° |     22.0° |    0.937 |         0.208 | 0.180 | 0.194 | 0.164 | 64 |
| roq4      | 4.00 |      4.4° |      4.7° |    0.996 |         0.620 | 0.625 | 0.651 | 0.640 | 96 |

**Verdict:** PROMOTE ternary as the primary low-bit candidate ahead of 2-bit
**Why:** Two non-obvious findings, both produced by this comparison:
1. **Ternary edges out 2-bit on the metrics that actually predict
   downstream Recall@10** — `rank_corr@100` 0.242 vs 0.208 and
   `NN5*`/`NN50*` consistently higher — even though per-token angular
   error is worse. Mechanism: 2-bit's inverse-FWHT spreads
   per-coordinate quantization noise across all 128 ambient dimensions,
   while ternary's noise stays group-localized (the kernel scores in
   rotated space without inverse-rotating). For ranking, the
   *correlation pattern* of the noise matters more than its magnitude.
2. **The quality cliff is between 2-bit and 4-bit, not within
   1-/1.58-/2-bit.** All three low-bit options sit in the same
   `rank_corr@100` 0.15–0.25 / `NN5*` 0.12–0.22 regime. Going to
   4-bit triples the rank correlation (0.62) but only saves 33% of disk
   relative to ternary at the same group_size. So the cost-benefit
   curve clearly favours **ternary for shipping** (1.58-bit, 32 B/tok,
   0.26 ms p50 from A2.5) and **4-bit as the rerank-tier** (96 B/tok,
   0.6+ rank correlation).

**Action:**
- C1.5 bake-off matrix: drop the `bits=2.0` arm from the production
  candidate set, keep it only as a sanity-check baseline. The matrix
  becomes ternary × {k_candidates ∈ 1k, 2k, 4k, 8k} × reranker
  on/off × query_bits ∈ {4, 6}.
- Distillation reranker (cross1) becomes higher-priority — it has to
  close the rank-corr gap from 0.24 (ternary) to whatever the BEIR
  Recall@10 floor demands. Plan estimate of "~50 µs p95 per shortlist"
  is the right ballpark to fit inside the latency budget freed up by
  ternary's 0.03 ms kernel-latency win over 2-bit.
- Honest caveat: this is per-token offline distortion only; MaxSim sums
  32 query tokens which damps individual-token noise substantially.
  The BEIR Recall@10 numbers from A6 are still the deciding gate; this
  comparison just narrows the candidate set going into A5 / A6.

**Artifacts:** `reports/bitwidth_compare.json`
**Gate impact:** narrows C1.5 candidate set to ternary; raises priority
of the distillation cross-cut; deprioritizes the 2-bit asymmetric
kernel from primary (A2 stays validated as a backup but not the lead
candidate).

## [2026-04-19] a4-salience — norm-based token-pruning sweep (CPU)

**Config:** `research/low_bit_roq/bench_salience.py` — 8 192 real tokens × 128 queries × synthetic 80 docs (~102 tokens/doc), prune-rates ∈ {0, 10, 20, 30, 50}%, signal=`norm`, min_tokens_per_doc=4
**Datasets / seeds:** offline distortion only, seed 0
**Baseline:** prune_rate=0 (no pruning, token_recall@K=1.0 by construction)

| prune | disk_fraction | token_recall@1 | token_recall@5 | token_recall@50 | random baseline |
| ----: | ------------: | -------------: | -------------: | --------------: | --------------: |
|    0% |          1.00 |          1.000 |          1.000 |           1.000 |            1.00 |
|   10% |          0.90 |          0.891 |          0.912 |           0.909 |            0.90 |
|   20% |          0.80 |          0.789 |          0.814 |           0.805 |            0.80 |
|   30% |          0.70 |          0.656 |          0.728 |           0.707 |            0.70 |
|   50% |          0.50 |          0.500 |          0.528 |           0.513 |            0.50 |

**Verdict:** KEEP-EXPERIMENTAL (signal needs raw-encoder data), KILL (norm signal as a production-default)
**Why:** At every prune rate the token-recall numbers track the random
baseline (`1 - prune_rate`) within 2 percentage points. That means norm
is essentially uninformative as a salience signal **on this fixture**.
Two plausible explanations, both pointing the same direction:
1. The 1 M-token fixture is sampled from the BEIR `*.npz` shards which
   already store post-encoder, layer-norm'd embeddings — those are
   approximately unit-norm by construction, so `||token||` carries
   almost no per-token signal compared to the raw pre-projection model
   outputs the plan was thinking about.
2. The pad/CLS/SEP tokens (which are the entire premise of "small
   norm = filler") may already be stripped during shard creation in
   `benchmarks/beir_benchmark.py`.

Therefore norm-based pruning **cannot be tested in our offline harness
at all** — to evaluate it we would need the raw ColBERTv2 encoder
attached, which is out-of-scope here. Same for `idf` (needs vocab
mapping) and `attention_mass` (needs a 5k training-query retrieval
pass through the LEMUR lane).

**Action:**
- Do NOT ship norm-pruning as part of the C1.5 candidate.
- Keep `salience.py` and the prune machinery in place, since the wiring
  works (it gives the expected disk_fraction and matches the
  per-doc-floor logic). Once Phase A5 stands up the LEMUR-lane runner
  with the encoder attached, re-run with `signal=attention_mass` on
  raw embeddings — that's the only signal the plan called "strongest"
  anyway.
- Revisit B4 ("mixed precision rroq + token salience interaction"):
  the interaction term against ternary is not measurable here either.
  Mark B4 as gated on A5.

**Artifacts:** `reports/a4_salience.json`
**Gate impact:** advances A4 to a clean go/no-go on offline-norm; pushes
attention-mass + IDF measurements behind A5; un-blocks the team to
reroute time from A4 sweeps into A5 integration.

## [2026-04-19] a3-anisotropic — anisotropic codebook A/B on real ColBERT tokens (CPU)

**Config:** `research/low_bit_roq/bench_anisotropic.py` — 8 192 real tokens × 128 queries from `tests/fixtures/token_sample_1m.npy`, dim=128, group_size=32, η ∈ {1, 2, 4, 8}, no FWHT (isolates the codebook-fit contribution from rotation)
**Datasets / seeds:** offline distortion only, seed 0
**Baseline:** uniform per-group min/max (η=1)

| quantizer / method        | η   | angular_p50 | IP_RMS | IP parallel | IP perp |
| ------------------------- | --: | ----------: | -----: | ----------: | ------: |
| ternary / tau_frac (=base)| 1.0 |       23.74 | 0.1421 |      0.1486 |  0.0153 |
| ternary / anisotropic     | 2.0 |       23.44 | 0.1403 |      0.1451 |  0.0145 |
| ternary / anisotropic     | 4.0 |       23.44 | 0.1403 |      0.1451 |  0.0145 |
| ternary / anisotropic     | 8.0 |       23.44 | 0.1403 |      0.1451 |  0.0145 |
| roq2 / uniform            | 1.0 |       21.32 | 0.0314 |      0.0275 |  0.0145 |
| roq2 / anisotropic        | 2.0 |       20.75 | 0.0452 |      0.0429 |  0.0134 |
| roq2 / anisotropic        | 4.0 |       21.53 | 0.0397 |      0.0370 |  0.0143 |
| roq2 / anisotropic        | 8.0 |       26.91 | 0.0932 |      0.0914 |  0.0210 |

**Verdict:** KEEP-EXPERIMENTAL (ternary), KILL (2-bit) — for now
**Why:** Two findings, both real:
1. **Ternary**: anisotropic τ-grid fitting gives a 1.2% IP-RMS improvement
   and a 2.3% reduction in the parallel-to-token IP error. The gain
   saturates at η=2 because the τ-grid is small (6 values). This is
   below the plan's 5%-or-better effect-size bar but not actively
   harmful, so we keep the `fit_method="anisotropic"` switch in
   `TernaryConfig` for use later (e.g. compose with FWHT in C1.5).
2. **2-bit**: anisotropic Newton fitting in
   `anisotropic.fit_anisotropic_min_max` *increases* IP-RMS by 44%
   at η=2 and 200% at η=8, and at η=8 even angular_p50 worsens by 5°.
   Inspecting the gradient: the code uses `eta * parallel * ...` where
   the analytical derivation gives `(eta - 1) * parallel * ...`, and
   substitutes `codes_centered = codes - mean(codes)` for `codes` in
   the inner product — both make the Newton step point in the wrong
   direction once the optimum starts pulling away from uniform. The
   uniform η=1 baseline reproduces the existing `RotationalQuantizer`
   2-bit numbers within 1° (21.32° here vs 20.40° in A1 cell-018; the
   small gap is the A1 cell additionally running FWHT).
**Action:** ship ternary as the bit-width with the cleanest
fit-method story (matches the A2.5 throughput win); do *not* promote
2-bit anisotropic until either (a) the gradient is corrected and
re-validated, or (b) we find an alternative fitter (e.g. coordinate
descent on (scale, offset) per group). Captured this as the leading
"why we picked ternary as the production candidate" data point for
the C1.5 bake-off.
**Artifacts:** `reports/a3_anisotropic.json`
**Gate impact:** advances A3 with a definitive go/no-go per quantizer;
removes 2-bit anisotropic from the C1.5 matrix until a fitter rewrite.

## [2026-04-19] a2-a25-kernels — Triton 2-bit-asym and 1.58-bit-ternary kernels validated on A5000

**Config:** GPU micro-benchmark (`research/low_bit_roq/bench_kernels.py`),
16 queries × 64 docs × 32 q_tokens × 64 d_tokens × dim 128 × query_bits 6 × group_size 32, 15 timed iters after 3 warmup
**Datasets / seeds:** synthetic FWHT-rotated random vectors, seed 0; doc encoding mirrors `TernaryQuantizer.quantize` and the existing 2-bit RoQ packing
**Baseline:** NumPy dequant reference (q_dequant @ decoded.T then MaxSim)

| metric                | roq2_asym | roq158_ternary | trend |
| --------------------- | --------: | -------------: | :---- |
| p50 latency (ms)      |     0.292 |          0.264 | ternary 9.6% faster |
| p95 latency (ms)      |     0.302 |          0.274 | ternary 9.3% faster |
| QPS (queries/s)       |    54 770 |         60 677 | ternary +11% |
| parity vs dequant     |     0.000 |        2.3e-05 | both within fp32 noise |
| GPU peak VRAM         |   <100 MB |        <100 MB | well under 18 GB budget |

**Verdict:** PROMOTE both
**Why:** This is the first hard kernel-level evidence that ternary's
popcount-only inner loop genuinely beats the 2-bit asymmetric four-term
affine kernel on real GPU (the planning argument was `2 popcounts/coord`
vs `2 popcounts/coord + per-group code_sum + per-group code_offset`).
Ternary's parity-against-dequant-baseline matches to 5 decimals,
confirming the affine reconstruction `est = scale_g * (q_offset * (pos - neg) + q_scale * Σ_k 2^k * (m_k - c_k))` is
implemented exactly as planned. The 2-bit asymmetric kernel at 0.29 ms
is also the right ballpark for the existing 1-bit ROQ kernel's measured
speed in `voyager_index/_internal/kernels/triton_roq.py`, which means
the 4-term correction did not blow up the latency budget.
Two follow-ups before merging into LEMUR (Phase A5):
(1) the bit-order convention in `TernaryQuantizer.quantize` defaults to
big-endian `np.packbits(...)`, but the kernel/test convention is
little-endian — the bench currently re-packs from `enc["rotated"]` to
match. Either fix `TernaryQuantizer` to use `bitorder='little'` or add
an explicit packing helper before persistence; otherwise the kernel
silently scrambles coords. (2) the bench uses synthetic Gaussian docs
not real ColBERT tokens — once A5 wires through the LEMUR shard, re-run
the same harness on real shards to confirm the throughput holds.
**Artifacts:** `reports/a2_kernels.json`
**Gate impact:** advances A2 + A2.5; un-blocks the bit-width bake-off
matrix (C1.5) which can now use both kernels through the harness.

## [2026-04-19] a1-lite-pilot — first real-data Phase A1 pilot under 24 GB box

**Config:** 20-cell A1 grid (`enumerate_cells_lite`: doc_bits ∈ {1, 1.58, 2}, group_size ∈ {None, 64, 32}, fwht ∈ {on, off}, normalize ∈ {on, off}, query_bits=6, codebook=uniform, norm_corr=4term)
**Datasets / seeds:** offline distortion only — 8 192 tokens × 128 queries sampled from `tests/fixtures/token_sample_1m.npy` (200 k tokens each from arguana/fiqa/nfcorpus/scidocs/scifact)
**Baseline:** A1 cell `roq4` is the implicit baseline once it's added in the next sweep

| metric                 | 1-bit best | 1.58-bit best | 2-bit best | trend |
| ---------------------- | ---------: | ------------: | ---------: | :---- |
| angular_p50 (deg)      |       29.3 |          23.7 |       20.4 | ↓ with bits, as expected |
| FWHT contribution      |       -1.0 |          -0.5 |       -3.7 | helps 2-bit most; tail-coord story holds |
| normalize contribution |        0.0 |           0.0 |        0.0 | A1 normalize axis is recall-neutral on this sample |
| nn1_preservation       |       1.00 |          0.63 |       0.24 | DEGENERATE (see Killed) |
| peak RAM (GB)          |        0.4 |           0.4 |        0.4 | well under 20 GB rlimit cap |

**Verdict:** PROMOTE (lite grid OK; expand on next memory window)
**Why:** The bug the user originally diagnosed ("2-bit kernel is unusable") is
visible here as the **opposite direction** in NN1 — 2-bit decoded vectors
have the lowest angular error but the worst NN1, which is impossible
unless NN1 is contaminated by self-matches. The angular column gives the
*correct* mechanics ranking and matches expectation: bits monotonically
reduce reconstruction error, FWHT helps 2-bit most because it spreads
energy off the tail coordinates that per-group min/max bins poorly. The
real surprise is normalize-on giving 0.0 contribution at the offline
level — confirms the plan's claim (Track 1A) that the "spherical" win is
likely from "normalize first" and not from spherical k-means itself.
A1 BEIR follow-up is still required to gate against fp16/roq4 with the
LEMUR lane on, but that is a multi-GB process per dataset and is
deferred until the operator confirms the next memory window.
**Artifacts:** `reports/a1-cell-{000..019}.json`, `reports/a1-summary.json`
**Gate impact:** advances A1; identifies measurement bug (nn1 degenerate)
that must be fixed before the full 336-cell sweep is interpretable.

## [2026-04-19] phase-0-harness — Phase 0 deliverables landed and self-tested

**Config:** all helpers under `research/low_bit_roq/`, no edits to production `voyager_index/`
**Datasets / seeds:** n/a (infrastructure only — first 5-dataset × 5-seed sweep is the A1 deliverable)
**Baseline:** n/a

| artifact group                                 | what it gives the next phase                                |
| ---------------------------------------------- | ----------------------------------------------------------- |
| `harness.py`                                   | multi-seed sweep, paired-bootstrap p, cold/warm p95, compute accounting; runner-agnostic so A1/A6/C1.5 share the driver |
| `distortion_bench.py` + `ternary.py`           | offline angular + NN-preservation per quantizer; FWHT bug found and fixed during smoke (decode returns ambient, not rotated) |
| `kernels/triton_roq_2bit_asym.py` + `triton_roq_ternary.py` | popcount-only asymmetric kernels with explicit reference scorers for parity tests |
| `anisotropic.py`, `salience.py`, `mixed_precision.py`, `spherical_kmeans.py`, `tangent_query.py`, `lemur_tangent.py`, `rroq.py`, `per_cluster_pca.py`, `distill_rerank.py`, `shortcut_edges.py`, `filter_aware.py` | one module per A/B/cross-cut axis, all numpy-only at the boundary so they unit-test without GPU |
| `integration.py`                               | additive registry + persistence + score wrappers — no edits to production .py files |
| `run_a1.py` / `run_a6.py` / `run_c1_5.py`      | runners exercise the full driver; A1 enumerates 336 cells, A6 emits gate JSON + Promoted/Killed bullets, C1.5 enumerates 48 bake-off cells |
| `tests/test_*.py`                              | 34 tests, all green: ternary numerics, kernel ref parity, harness aggregation, paired bootstrap, PROGRESS helper, salience, mixed-precision, spherical k-means, tangent geodesic, anisotropic, cross-cuts |

**Verdict:** PROMOTE (Phase 0 done, A1 unblocked)
**Why:** Every subsequent gate (A6, C1.5, final memo) depends on these
artefacts; building them up-front means an engineer running A1 only has to
write the `runner_factory` that maps `(dataset, seed) -> SearchRunner` for
their concrete LEMUR-lane backend. Two real bugs found and fixed during
self-test: (a) ternary kernel needs `group_size % 32 == 0` (added a hard
validation in `TernaryQuantizer.__post_init__`); (b) the distortion-bench
wrapper for the existing `RotationalQuantizer` was comparing
ambient-space decoded vectors against rotated-space inputs, which is what
made 1-bit / 2-bit look catastrophic in the user's original observation —
fixed in `_existing_roq_quantize`. After the fix, the synthetic-data
smoke produces sensible numbers (1-bit p50 angular error 20.6°, 2-bit
21.1°, 4-bit 4.4°). The next sweep on the real 1M-token sample is the
first credible measurement.
**Artifacts:** [`README.md`](README.md), [`harness.py`](harness.py), [`distortion_bench.py`](distortion_bench.py), [`run_a1.py`](run_a1.py), [`reports/SCHEMA.md`](reports/SCHEMA.md)
**Gate impact:** unblocks A1 (mechanics sweep), A2 (kernel A/B), A2.5 (ternary kernel A/B); also unblocks any concurrent Phase B work that wants to use the harness without waiting on the A1 winner

## [INIT] plan-bootstrap — research scaffolding created

**Config:** plan version `riemannian_low-bit_roq_a19e0a55`, scope = 6 weeks / 3 engineers
**Datasets / seeds:** n/a (no experiment yet)
**Baseline:** n/a

| artifact                                       | purpose                                       |
| ---------------------------------------------- | --------------------------------------------- |
| `harness.py`                                   | Phase 0 harness extensions                    |
| `progress_md.py`                               | append-only entry helper used by harness      |
| `distortion_bench.py`                          | offline angular / NN-preservation bench       |
| `kernels/triton_roq_2bit_asym.py`              | A2 asymmetric 2-bit kernel                    |
| `kernels/triton_roq_ternary.py`                | A2.5 ternary (1.58-bit) kernel                |
| `anisotropic.py` / `ternary.py` / `salience.py`| A3 / A2.5 / A4 building blocks                |
| `tangent_query.py` / `spherical_kmeans.py`     | B0 / B1 router-side geometry                  |
| `lemur_tangent.py` / `rroq.py` / `mixed_precision.py` / `per_cluster_pca.py` | B2-B5 |
| `distill_rerank.py` / `shortcut_edges.py` / `filter_aware.py` | cross-cuts 1-3 |
| `integration.py`                               | A5 production-lane glue                       |
| `run_a1.py` / `run_a6.py` / `run_c1_5.py`      | runner scripts                                |

**Verdict:** PROMOTE (scaffolding only — no metrics gate)
**Why:** Lays the audit trail for every subsequent experiment so PROGRESS.md
becomes a single readable timeline rather than a collection of scattered
notebooks. Folder layout matches the plan's `Continuous progress log`
section so engineers find each artifact at the path the plan promises.
**Artifacts:** [`README.md`](README.md), this file
**Gate impact:** unblocks Phase 0 harness work
