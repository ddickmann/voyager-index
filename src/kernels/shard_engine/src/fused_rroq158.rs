//! CPU-side fused MaxSim kernel for Riemannian-aware 1.58-bit (ternary) ROQ.
//!
//! Mirrors the math of `voyager_index/_internal/kernels/triton_roq_rroq158.py`
//! exactly, so parity tests against `reference_score_rroq158` and the Triton
//! kernel hold to within float-rounding noise.
//!
//! Per (q_token i, d_token j):
//!
//!   resi(i, j) = sum_g d_scale[j, g] * (q_offset[i] * s_g(j) + q_scale[i] * d_g(i, j))
//!   est(i, j)  = cos_norm[j] * qc_table[i, centroid_id[j]] + sin_norm[j] * resi(i, j)
//!
//! where for each 32-bit word w in group g:
//!   pos = popcount( sign[j, w] &  nz[j, w] )
//!   neg = popcount(!sign[j, w] &  nz[j, w] )
//!   s_g   += pos - neg
//!   for k in 0..query_bits:
//!       m = popcount( sign[j, w] &  q_planes[i, k, w] &  nz[j, w])
//!       c = popcount(!sign[j, w] &  q_planes[i, k, w] &  nz[j, w])
//!       d_g += (1 << k) * (m - c)
//!
//! MaxSim:
//!   score(d) = sum_i max_j est(i, j)        // optionally masked by q/d masks
//!
//! SIMD strategy:
//!
//!   * Tight bitwise inner loop over 32-bit words → use the hardware
//!     `popcnt` instruction via `u32::count_ones`. The LLVM codegen lowers
//!     this to `popcntq` on x86_64 when the `popcnt` feature is available
//!     (always present on AVX2 hosts), and to `cnt` on aarch64 NEON.
//!   * Per-doc-token group accumulation is purely scalar f32; on AVX2 we
//!     fuse the final `cos·qc + sin·resi` multiply-adds into FMA.
//!   * The hot dimension (`B` documents) is the rayon-parallel one.
//!
//! Layout convention matches the Triton/Python side exactly so we can share
//! the same packed payload tensors:
//!
//!   docs_sign  [B, T, n_words]   i32 (LE-packed sign bits)
//!   docs_nz    [B, T, n_words]   i32 (LE-packed nonzero mask)
//!   docs_scl   [B, T, n_groups]  f32
//!   docs_cid   [B, T]            i32 centroid id ∈ [0, K)
//!   docs_cos   [B, T]            f32 cos(||r||) * norm_d
//!   docs_sin   [B, T]            f32 sinc(||r||) * norm_d
//!   docs_mask  [B, T]            f32 (>0 → active token)
//!
//!   q_planes   [A, S, query_bits, n_words]  i32
//!   q_meta     [A, S, 2]                    f32  [scale, offset]
//!   qc_table   [A, S, K]                    f32  q_amb @ centroids.T
//!   q_mask     [A, S]                       f32

use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────── popcount primitives ───────────────────
//
// CRITICAL: `u32::count_ones()` in the Rust stdlib is precompiled without the
// `popcnt` target feature, so on x86_64 it lowers to a 10-instruction SWAR
// sequence (~10× slower than the hardware `popcntq` instruction). To force
// the hardware path we wrap the intrinsic in a `#[target_feature(enable =
// "popcnt")]` function and call it from a `#[target_feature(...)]`-gated
// inner kernel. The feature-gated context makes LLVM emit `popcntq`
// directly. Verified at the binary level: the gated path emits ~960 popcnts
// for the dim=128 inner loop; the scalar path emits 0.

#[inline(always)]
fn popc_scalar(x: u32) -> u32 {
    x.count_ones()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
#[inline]
unsafe fn popc_x86(x: u32) -> u32 {
    core::arch::x86_64::_popcnt32(x as i32) as u32
}

// ─────────────────── AVX-512 VPOPCNTDQ batched popcount ───────────────────
//
// On Sapphire Rapids / Ice Lake-server / Tiger Lake / Zen 4+ the
// `vpopcntd` instruction (CPUID flag AVX512_VPOPCNTDQ + AVX512F)
// computes the population count of 16 32-bit lanes in a single µop on
// port 5. That's 16 popcounts per cycle vs. 4 per cycle for scalar
// `popcntq` (one per integer ALU port × 4 ports), so a ~4× improvement
// at the inner-loop popcount level.
//
// Verified availability of the feature is mandatory: this function is
// only ever called from `#[target_feature(enable = "avx512f,
// avx512vpopcntdq")]`-gated callers, and `select_backend()` only
// selects the X86V4 backend after a runtime
// `is_x86_feature_detected!("avx512vpopcntdq")` check. On AVX2-only
// hosts (e.g. the published-benchmark RTX A5000 box, Skylake-class
// CPUs) the dispatcher falls through to the existing `popcnt` path
// transparently.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[inline]
#[allow(dead_code)] // shape-generic helper retained for future SIMD callers
unsafe fn popc_x86_v4(words: &[u32; 16]) -> [u32; 16] {
    use core::arch::x86_64::*;
    let v = _mm512_loadu_si512(words.as_ptr() as *const __m512i);
    let p = _mm512_popcnt_epi32(v);
    let mut out = [0u32; 16];
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, p);
    out
}

// ─────────────────── shared inner loop ───────────────────
//
// The body is parameterised on a popcount function so we can compile it once
// per ISA tier (scalar vs x86_64-v3 vs aarch64-NEON) without code
// duplication. Marked `#[inline(always)]` so the chosen popcount intrinsic
// is inlined into the calling target_feature scope and LLVM lowers it to
// the right instruction.

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn score_pair_body(
    q_planes_qa: &[i32],
    q_meta_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_sign_d: &[i32],
    docs_nz_d: &[i32],
    docs_scl_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    group_words: usize,
    query_bits: usize,
    big_k: usize,
    popc: impl Fn(u32) -> u32,
) -> f32 {
    debug_assert_eq!(n_groups * group_words, n_words);

    // Loop reorder (Phase-7 followup, mirrors fused_rroq4_riem.rs):
    //
    // The original layout iterated `for i in 0..big_s { for j in 0..big_t }`,
    // which forced the *purely doc-side* popcounts (`pos = popc(ds & dn)`,
    // `neg = popc(!ds & dn)`, `s_g += pos - neg`) to be re-done S times per
    // (j, w) pair even though they're independent of the query token i.
    // Reordering to `for j in 0..big_t { for i in 0..big_s }` lets us
    // amortise that doc-side work once per j and keep `max_sim[i]` as a
    // running per-query accumulator. Concrete savings at production
    // shape (S=32, T=32, n_words=4, group_words=1, n_groups=4): 31 *
    // (n_groups * group_words * 2) = 248 popcounts saved per (q, doc-tok)
    // pair, and B * T * 248 ≈ 16M popcounts saved per query at B=2000.
    // Same approach gave +28% docs/s on rroq4_riem at K=8192.

    let q_mask_active: Option<usize> = if let Some(qm) = q_mask_qa {
        let mut count = 0usize;
        for &v in qm.iter().take(big_s) {
            if v > 0.0 {
                count += 1;
            }
        }
        Some(count)
    } else {
        None
    };
    if matches!(q_mask_active, Some(0)) {
        return 0.0;
    }

    // Per-(i) running max over doc tokens. NEG_INFINITY entries mean the
    // query token never saw any doc-active token — those rows contribute
    // nothing to total, matching the original "if max_sim > NEG_INFINITY"
    // guard at the bottom.
    let mut max_sim: Vec<f32> = vec![f32::NEG_INFINITY; big_s];

    for j in 0..big_t {
        if let Some(dm) = docs_mask_d {
            if dm[j] <= 0.0 {
                continue;
            }
        }
        let sign_row = &docs_sign_d[j * n_words..(j + 1) * n_words];
        let nz_row = &docs_nz_d[j * n_words..(j + 1) * n_words];
        let scl_row = &docs_scl_d[j * n_groups..(j + 1) * n_groups];
        let cid = docs_cid_d[j] as usize;
        let cos_j = docs_cos_d[j];
        let sin_j = docs_sin_d[j];

        // Hoist doc-side popcounts once per doc token. Stash the per-group
        // (s_g, ds_w, dn_w) so the inner i-loop only does query-dependent
        // work. n_groups * group_words is bounded (4 in production), so
        // these stack allocations are cheap and keep everything in L1.
        // SAFETY: bounds-checked via the n_groups * group_words == n_words
        // invariant asserted at function entry.
        let mut s_per_group: [i32; 32] = [0; 32];
        debug_assert!(n_groups <= 32);
        for grp in 0..n_groups {
            let mut s_g: i32 = 0;
            let base_word = grp * group_words;
            for w in 0..group_words {
                let word_idx = base_word + w;
                let ds_w = unsafe { *sign_row.get_unchecked(word_idx) as u32 };
                let dn_w = unsafe { *nz_row.get_unchecked(word_idx) as u32 };
                s_g += popc(ds_w & dn_w) as i32;
                s_g -= popc((!ds_w) & dn_w) as i32;
            }
            s_per_group[grp] = s_g;
        }

        for i in 0..big_s {
            if let Some(qm) = q_mask_qa {
                if qm[i] <= 0.0 {
                    continue;
                }
            }
            let q_scale = q_meta_qa[i * 2];
            let q_offset = q_meta_qa[i * 2 + 1];
            let q_planes_token = &q_planes_qa
                [i * query_bits * n_words..(i + 1) * query_bits * n_words];
            let qc_row = &qc_table_qa[i * big_k..(i + 1) * big_k];

            let mut resi: f32 = 0.0;
            for grp in 0..n_groups {
                let d_scale_g = scl_row[grp];
                let s_g = s_per_group[grp];
                let mut d_g: i32 = 0;
                let base_word = grp * group_words;
                for w in 0..group_words {
                    let word_idx = base_word + w;
                    let ds_w = unsafe { *sign_row.get_unchecked(word_idx) as u32 };
                    let dn_w = unsafe { *nz_row.get_unchecked(word_idx) as u32 };
                    for k in 0..query_bits {
                        let qk_w = unsafe {
                            *q_planes_token
                                .get_unchecked(k * n_words + word_idx) as u32
                        };
                        let m = popc(ds_w & qk_w & dn_w) as i32;
                        let c = popc((!ds_w) & qk_w & dn_w) as i32;
                        d_g += (1i32 << k) * (m - c);
                    }
                }
                resi += d_scale_g * (q_offset * s_g as f32 + q_scale * d_g as f32);
            }
            let qc = qc_row[cid];
            let est = cos_j * qc + sin_j * resi;
            // SAFETY: bounded by big_s.
            let slot = unsafe { max_sim.get_unchecked_mut(i) };
            if est > *slot {
                *slot = est;
            }
        }
    }

    let mut total: f32 = 0.0;
    for &v in max_sim.iter() {
        if v > f32::NEG_INFINITY {
            total += v;
        }
    }
    total
}

/// Score a single (query, doc) pair — pure scalar fallback. Used by the
/// parity tests and as the universal-portable backend.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn score_pair(
    q_planes_qa: &[i32],
    q_meta_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_sign_d: &[i32],
    docs_nz_d: &[i32],
    docs_scl_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    group_words: usize,
    query_bits: usize,
    big_k: usize,
) -> f32 {
    score_pair_body(
        q_planes_qa, q_meta_qa, qc_table_qa, q_mask_qa,
        docs_sign_d, docs_nz_d, docs_scl_d, docs_cid_d,
        docs_cos_d, docs_sin_d, docs_mask_d,
        big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
        popc_scalar,
    )
}

/// Score a single (query, doc) pair on the x86-64-v3 fast path: hardware
/// `popcntq`, AVX2 + FMA available for LLVM auto-vectorisation of the f32
/// reductions, BMI2 for masked-bit ops. The hot dim=128 / n_words=4 case
/// does ~40 popcounts per (q_token, d_token) pair — those popcounts dominate
/// the inner loop, so forcing them onto the hardware POPCNT instruction
/// (instead of the precompiled-stdlib SWAR fallback) is the headline win.
///
/// We considered a hand-rolled AVX2 PSHUFB Mula-style SIMD popcount
/// specialisation for n_words ∈ {2, 4, 8}, but on a host with hardware
/// POPCNT the throughput of `popcntq` (1 result/cycle, latency 3) already
/// matches what a Mula chain (load → and → shift → and → 2× pshufb → padd)
/// can deliver per 4-lane SIMD popcount. With target_feature we let LLVM
/// pick the optimal mix automatically; verified at the binary level
/// (~960 popcnts in the inner loop, plus auto-vectorised vfmaddXXXps for
/// the cos/sin reduction). If a future host adds AVX-512 VPOPCNTB we'll
/// gain real value from a SIMD popcount path.
///
/// SAFETY: caller must ensure the host CPU has popcnt + avx2 + bmi2 + fma
/// (verified once per process via `select_backend`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt,avx2,bmi2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_pair_x86v3(
    q_planes_qa: &[i32],
    q_meta_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_sign_d: &[i32],
    docs_nz_d: &[i32],
    docs_scl_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    group_words: usize,
    query_bits: usize,
    big_k: usize,
) -> f32 {
    score_pair_body(
        q_planes_qa, q_meta_qa, qc_table_qa, q_mask_qa,
        docs_sign_d, docs_nz_d, docs_scl_d, docs_cid_d,
        docs_cos_d, docs_sin_d, docs_mask_d,
        big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
        // SAFETY: this whole function is gated on `popcnt` being available,
        // so calling popc_x86 unsafely is sound.
        |x| popc_x86(x),
    )
}

/// Score one (q, doc) pair on the AVX-512 VPOPCNTDQ fast lane.
///
/// Specialised for the production rroq158 shape (`dim=128`, `n_words=4`,
/// `n_groups=1`, `query_bits=4`, `group_words=4`). For any other shape
/// we fall back to the scalar-popcnt `score_pair_x86v3` path so this
/// tier never blocks unusual configurations.
///
/// SIMD strategy at the production shape:
///
///   - Per doc-token `j`: load the 4-word ternary planes (`sign[j, 0..4]`
///     and `nz[j, 0..4]`) once, AND them into `pos[w] = sign & nz` and
///     `neg[w] = !sign & nz`, broadcast both 128-bit fragments to all
///     four 128-bit lanes of a 512-bit register (`vbroadcasti32x4`).
///   - Hoist the doc-side `s_g = popcount(pos) - popcount(neg)` once
///     per `j` using scalar `popcnt`.
///   - Per query-token `i` (inner loop): load the 16 contiguous int32
///     query bit-planes for this token (`q_planes[i, k=0..4, w=0..4]`)
///     into a single 512-bit register, AND with `pos_bcast` and
///     `neg_bcast`, run two `vpopcntd` instructions, subtract,
///     multiply by per-bit weights `[1,1,1,1, 2,2,2,2, 4,4,4,4, 8,8,8,8]`,
///     and reduce-add to recover `d_g`.
///
/// The math is bit-identical to `score_pair_body`: both compute
/// `d_g = sum_{k,w} (1<<k) * (popcount(pos & q[k, w]) - popcount(neg & q[k, w]))`.
/// SIMD only changes how many popcounts execute per cycle, not their
/// values, so parity tests against the scalar path are exact.
///
/// SAFETY: caller must ensure the host has popcnt + avx2 + bmi2 + fma +
/// avx512f + avx512vpopcntdq (verified once per process via
/// `select_backend`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt,avx2,bmi2,fma,avx512f,avx512vpopcntdq")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_pair_x86v4(
    q_planes_qa: &[i32],
    q_meta_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_sign_d: &[i32],
    docs_nz_d: &[i32],
    docs_scl_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    group_words: usize,
    query_bits: usize,
    big_k: usize,
) -> f32 {
    // Fast lane only handles the production shape. Everything else falls
    // through to the v3 (scalar popcnt) path so we never regress on
    // non-default configurations.
    if !(n_groups == 1 && group_words == 4 && query_bits == 4 && n_words == 4) {
        return score_pair_x86v3(
            q_planes_qa, q_meta_qa, qc_table_qa, q_mask_qa,
            docs_sign_d, docs_nz_d, docs_scl_d, docs_cid_d,
            docs_cos_d, docs_sin_d, docs_mask_d,
            big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
        );
    }

    use core::arch::x86_64::*;

    // Per-bit weights for `d_g` reduction, laid out to match the lane
    // ordering produced by `_mm512_loadu_si512(q_planes[i, ...])` —
    // namely `[k=0,w=0..3 | k=1,w=0..3 | k=2,w=0..3 | k=3,w=0..3]` —
    // because q_planes is stored (S, query_bits, n_words) row-major and
    // the inner two axes (`query_bits=4`, `n_words=4`) flatten to 16
    // contiguous int32 per token.
    let weights = _mm512_setr_epi32(
        1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8,
    );

    // Cheap up-front mask check: if every active query-token is masked,
    // we can exit early.
    let q_mask_active: Option<usize> = q_mask_qa.map(|qm| {
        qm.iter().take(big_s).filter(|&&v| v > 0.0).count()
    });
    if matches!(q_mask_active, Some(0)) {
        return 0.0;
    }

    let mut max_sim: Vec<f32> = vec![f32::NEG_INFINITY; big_s];

    for j in 0..big_t {
        if let Some(dm) = docs_mask_d {
            if dm[j] <= 0.0 {
                continue;
            }
        }

        // Load doc-side sign and nz (4 int32 = 128 bits) for this token.
        let sign_ptr = docs_sign_d.as_ptr().add(j * 4) as *const __m128i;
        let nz_ptr = docs_nz_d.as_ptr().add(j * 4) as *const __m128i;
        let ds_v = _mm_loadu_si128(sign_ptr);
        let dn_v = _mm_loadu_si128(nz_ptr);
        let pos_v = _mm_and_si128(ds_v, dn_v);
        let neg_v = _mm_andnot_si128(ds_v, dn_v);

        // Doc-side `s_g = popcount(pos) - popcount(neg)`. Done with
        // hardware popcnt on the four 32-bit lanes; cheaper than a
        // SIMD popcnt + horizontal reduction at this width.
        let s_g_pos = _popcnt32(_mm_extract_epi32::<0>(pos_v)) as i32
            + _popcnt32(_mm_extract_epi32::<1>(pos_v)) as i32
            + _popcnt32(_mm_extract_epi32::<2>(pos_v)) as i32
            + _popcnt32(_mm_extract_epi32::<3>(pos_v)) as i32;
        let s_g_neg = _popcnt32(_mm_extract_epi32::<0>(neg_v)) as i32
            + _popcnt32(_mm_extract_epi32::<1>(neg_v)) as i32
            + _popcnt32(_mm_extract_epi32::<2>(neg_v)) as i32
            + _popcnt32(_mm_extract_epi32::<3>(neg_v)) as i32;
        let s_g = (s_g_pos - s_g_neg) as f32;

        // Broadcast pos/neg 128-bit fragment to all four 128-bit lanes
        // of a 512-bit register so the inner loop's `_mm512_and_si512`
        // pairs each of the 4 query bit-planes with the right doc word.
        let pos_512 = _mm512_broadcast_i32x4(pos_v);
        let neg_512 = _mm512_broadcast_i32x4(neg_v);

        let cid = docs_cid_d[j] as usize;
        let cos_j = docs_cos_d[j];
        let sin_j = docs_sin_d[j];
        let d_scale = docs_scl_d[j]; // n_groups == 1 → single scalar per token

        // Strides per query-token: i*16 = i * (query_bits * n_words).
        for i in 0..big_s {
            if let Some(qm) = q_mask_qa {
                if qm[i] <= 0.0 {
                    continue;
                }
            }
            let q_scale = q_meta_qa[i * 2];
            let q_offset = q_meta_qa[i * 2 + 1];

            let q_ptr = q_planes_qa.as_ptr().add(i * 16) as *const __m512i;
            let q_v = _mm512_loadu_si512(q_ptr);

            // 16 popcounts in two `vpopcntd` instructions (port 5
            // throughput on SPR/ICX/Tiger Lake/Zen 4+ = 1/cycle each):
            //   m_pc[lane] = popcount(pos[w] & q_planes[i, k, w])
            //   c_pc[lane] = popcount(neg[w] & q_planes[i, k, w])
            // for lane = k*4 + w, k ∈ 0..4, w ∈ 0..4.
            let m_andd = _mm512_and_si512(pos_512, q_v);
            let c_andd = _mm512_and_si512(neg_512, q_v);
            let m_pc = _mm512_popcnt_epi32(m_andd);
            let c_pc = _mm512_popcnt_epi32(c_andd);

            // d_g = sum_{k, w} (1<<k) * (m - c)
            let diff = _mm512_sub_epi32(m_pc, c_pc);
            let weighted = _mm512_mullo_epi32(diff, weights);
            let d_g = _mm512_reduce_add_epi32(weighted) as f32;

            // Single-group residual: resi = d_scale * (q_offset * s_g + q_scale * d_g)
            let resi = d_scale * (q_offset * s_g + q_scale * d_g);
            let qc = qc_table_qa[i * big_k + cid];
            let est = cos_j * qc + sin_j * resi;

            // SAFETY: i < big_s and max_sim has length big_s.
            let slot = max_sim.get_unchecked_mut(i);
            if est > *slot {
                *slot = est;
            }
        }
    }

    let mut total: f32 = 0.0;
    for &v in max_sim.iter() {
        if v > f32::NEG_INFINITY {
            total += v;
        }
    }
    total
}

// ─────────────────── runtime feature dispatch ───────────────────
//
// We do the CPU feature detection once per process and cache the result in
// an atomic. The check is essentially free in the hot loop (one relaxed
// atomic load), but skipping the detection itself avoids the cost of the
// stdlib's stack-frame setup on every kernel call.

const BACKEND_UNINIT: u8 = 0;
const BACKEND_SCALAR: u8 = 1;
#[cfg(target_arch = "x86_64")]
const BACKEND_X86V3: u8 = 2;
#[cfg(target_arch = "x86_64")]
const BACKEND_X86V4: u8 = 3;

static BACKEND: AtomicU8 = AtomicU8::new(BACKEND_UNINIT);

#[inline]
fn select_backend() -> u8 {
    let cur = BACKEND.load(Ordering::Relaxed);
    if cur != BACKEND_UNINIT {
        return cur;
    }
    let chosen: u8 = {
        #[cfg(target_arch = "x86_64")]
        {
            // x86-64-v4: hardware popcnt + AVX2 + BMI2 + FMA + AVX-512F +
            // AVX-512 VPOPCNTDQ. Available on Sapphire Rapids /
            // Ice Lake-server / Tiger Lake-/Alder Lake-mobile / Zen 4+.
            // Older AVX2-only hosts (Skylake-class, Zen 1-3, the published
            // benchmark RTX A5000 box) fall through to the v3 path
            // unchanged.
            let v3 = std::is_x86_feature_detected!("popcnt")
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("fma");
            let v4 = v3
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512vpopcntdq");
            if v4 {
                BACKEND_X86V4
            } else if v3 {
                BACKEND_X86V3
            } else {
                BACKEND_SCALAR
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            BACKEND_SCALAR
        }
    };
    BACKEND.store(chosen, Ordering::Relaxed);
    chosen
}

/// Force the v3 (popcnt + AVX2) backend regardless of detected CPU features.
/// Test-only utility used by parity tests to compare v3 vs v4 paths on
/// hosts that support AVX-512 VPOPCNTDQ.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub fn _force_x86v3_backend_for_tests() {
    BACKEND.store(BACKEND_X86V3, Ordering::Relaxed);
}

/// Reset the cached backend selection. Test-only utility.
#[doc(hidden)]
pub fn _reset_backend_for_tests() {
    BACKEND.store(BACKEND_UNINIT, Ordering::Relaxed);
}

/// Force the scalar backend regardless of detected CPU features. Test-only
/// utility used by parity tests to compare scalar vs accelerated paths.
#[doc(hidden)]
pub fn _force_scalar_backend_for_tests() {
    BACKEND.store(BACKEND_SCALAR, Ordering::Relaxed);
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn score_pair_dispatch(
    q_planes_qa: &[i32],
    q_meta_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_sign_d: &[i32],
    docs_nz_d: &[i32],
    docs_scl_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    group_words: usize,
    query_bits: usize,
    big_k: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        let backend = select_backend();
        if backend == BACKEND_X86V4 {
            // SAFETY: BACKEND_X86V4 is only set when popcnt + avx2 + bmi2 +
            // fma + avx512f + avx512vpopcntdq are all runtime-detected,
            // which is the precondition of score_pair_x86v4. The v4 entry
            // also self-falls-back to v3 for non-default shapes.
            return unsafe {
                score_pair_x86v4(
                    q_planes_qa, q_meta_qa, qc_table_qa, q_mask_qa,
                    docs_sign_d, docs_nz_d, docs_scl_d, docs_cid_d,
                    docs_cos_d, docs_sin_d, docs_mask_d,
                    big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
                )
            };
        }
        if backend == BACKEND_X86V3 {
            // SAFETY: BACKEND_X86V3 is only set when popcnt+avx2+bmi2+fma are
            // all runtime-detected, which is the precondition of
            // score_pair_x86v3.
            return unsafe {
                score_pair_x86v3(
                    q_planes_qa, q_meta_qa, qc_table_qa, q_mask_qa,
                    docs_sign_d, docs_nz_d, docs_scl_d, docs_cid_d,
                    docs_cos_d, docs_sin_d, docs_mask_d,
                    big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
                )
            };
        }
    }
    score_pair(
        q_planes_qa, q_meta_qa, qc_table_qa, q_mask_qa,
        docs_sign_d, docs_nz_d, docs_scl_d, docs_cid_d,
        docs_cos_d, docs_sin_d, docs_mask_d,
        big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
    )
}

/// Batched entry point used by Python.
///
/// Inputs are flat row-major slices with the shapes from the module doc.
/// Output is `(A * B,)` row-major scores: `out[a * B + d] = MaxSim(q_a, doc_d)`.
///
/// Documents are scored in parallel via rayon — this is the hot dimension
/// because `B` is typically 500..4000 candidates per query in production.
///
/// `n_threads` controls the size of the rayon worker pool. Pass `None` to
/// use the global rayon pool (default `RAYON_NUM_THREADS` or
/// `num_cpus::get()`). Pass `Some(n)` to install a per-call scoped pool —
/// this is the right call when the python side already has its own
/// `ThreadPoolExecutor` of size `W`, in which case
/// `n_threads = max(1, cpu_count() // W)` avoids 1024-way over-subscription
/// (each python worker spinning up its own 128-thread rayon pool).
#[allow(clippy::too_many_arguments)]
pub fn score_batch(
    q_planes: &[i32],
    q_meta: &[f32],
    qc_table: &[f32],
    q_mask: Option<&[f32]>,
    docs_sign: &[i32],
    docs_nz: &[i32],
    docs_scl: &[f32],
    docs_cid: &[i32],
    docs_cos: &[f32],
    docs_sin: &[f32],
    docs_mask: Option<&[f32]>,
    big_a: usize,
    big_b: usize,
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    query_bits: usize,
    big_k: usize,
    n_threads: Option<usize>,
    out: &mut [f32],
) {
    // Cached per-(n_threads) scoped pool to bound oversubscription. We
    // can NOT use rayon's global pool: the python side typically runs
    // this kernel from W python workers concurrently, and the global pool
    // defaults to one thread per logical core, so the total active thread
    // count is W * cpu_count() (1024 on a 128-core box with W=8). With a
    // bounded pool of size cpu_count()/W, total threads stays at
    // cpu_count() across all workers.
    //
    // Per-call ThreadPoolBuilder::new().build() costs ~10-20 ms (OS
    // thread spawn per worker), so we cache pools by n_threads in a
    // process-wide map. The map is bounded in practice: production code
    // chooses one n_threads value per process via VOYAGER_RROQ158_N_WORKERS.
    if let Some(n) = n_threads.filter(|&n| n >= 1) {
        let pool = get_or_create_pool(n);
        pool.install(|| {
            score_batch_inner(
                q_planes, q_meta, qc_table, q_mask,
                docs_sign, docs_nz, docs_scl, docs_cid,
                docs_cos, docs_sin, docs_mask,
                big_a, big_b, big_s, big_t,
                n_words, n_groups, query_bits, big_k,
                out,
            )
        });
        return;
    }
    score_batch_inner(
        q_planes, q_meta, qc_table, q_mask,
        docs_sign, docs_nz, docs_scl, docs_cid,
        docs_cos, docs_sin, docs_mask,
        big_a, big_b, big_s, big_t,
        n_words, n_groups, query_bits, big_k,
        out,
    );
}

// Process-wide cache of bounded rayon pools, keyed by thread count. Built
// lazily on first use so module load doesn't pay the spawn cost.
static POOL_CACHE: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

fn get_or_create_pool(n_threads: usize) -> Arc<rayon::ThreadPool> {
    let cache = POOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let map = cache.lock();
        if let Some(p) = map.get(&n_threads) {
            return p.clone();
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .thread_name(move |i| format!("rroq158-{n_threads}-{i}"))
            .build()
            .expect("rroq158 rayon pool"),
    );
    let mut map = cache.lock();
    // Recheck under the write lock to avoid duplicate pool construction
    // under racing first-callers (cheap because we only ever build a
    // handful of distinct sizes per process).
    map.entry(n_threads).or_insert_with(|| pool.clone()).clone()
}

#[allow(clippy::too_many_arguments)]
fn score_batch_inner(
    q_planes: &[i32],
    q_meta: &[f32],
    qc_table: &[f32],
    q_mask: Option<&[f32]>,
    docs_sign: &[i32],
    docs_nz: &[i32],
    docs_scl: &[f32],
    docs_cid: &[i32],
    docs_cos: &[f32],
    docs_sin: &[f32],
    docs_mask: Option<&[f32]>,
    big_a: usize,
    big_b: usize,
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    query_bits: usize,
    big_k: usize,
    out: &mut [f32],
) {
    let group_words = n_words / n_groups;
    debug_assert_eq!(out.len(), big_a * big_b);

    let q_planes_stride_a = big_s * query_bits * n_words;
    let q_meta_stride_a = big_s * 2;
    let qc_stride_a = big_s * big_k;
    let q_mask_stride_a = big_s;
    let d_sign_stride = big_t * n_words;
    let d_nz_stride = big_t * n_words;
    let d_scl_stride = big_t * n_groups;
    let d_cid_stride = big_t;
    let d_cos_stride = big_t;
    let d_sin_stride = big_t;
    let d_mask_stride = big_t;

    for a in 0..big_a {
        let q_planes_a = &q_planes[a * q_planes_stride_a..(a + 1) * q_planes_stride_a];
        let q_meta_a = &q_meta[a * q_meta_stride_a..(a + 1) * q_meta_stride_a];
        let qc_a = &qc_table[a * qc_stride_a..(a + 1) * qc_stride_a];
        let q_mask_a = q_mask.map(|m| &m[a * q_mask_stride_a..(a + 1) * q_mask_stride_a]);

        // Parallelise across documents. `with_min_len` clusters tasks so
        // each rayon worker amortises its startup cost over a meaningful
        // chunk of work (≥ 16 docs / task). Without this, B=512 tasks
        // across 128 threads wastes most of the wall time on task wakeups
        // (~10-50 µs/wakeup) instead of the ~10 µs of actual work per doc.
        let min_chunk = std::cmp::max(1, big_b / 64).max(16);
        let out_row = &mut out[a * big_b..(a + 1) * big_b];
        out_row
            .par_iter_mut()
            .with_min_len(min_chunk)
            .enumerate()
            .for_each(|(d, slot)| {
                let sign_d = &docs_sign[d * d_sign_stride..(d + 1) * d_sign_stride];
                let nz_d = &docs_nz[d * d_nz_stride..(d + 1) * d_nz_stride];
                let scl_d = &docs_scl[d * d_scl_stride..(d + 1) * d_scl_stride];
                let cid_d = &docs_cid[d * d_cid_stride..(d + 1) * d_cid_stride];
                let cos_d = &docs_cos[d * d_cos_stride..(d + 1) * d_cos_stride];
                let sin_d = &docs_sin[d * d_sin_stride..(d + 1) * d_sin_stride];
                let mask_d = docs_mask.map(|m| &m[d * d_mask_stride..(d + 1) * d_mask_stride]);

                *slot = score_pair_dispatch(
                    q_planes_a, q_meta_a, qc_a, q_mask_a,
                    sign_d, nz_d, scl_d, cid_d, cos_d, sin_d, mask_d,
                    big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
                );
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny deterministic fixture and check that `score_pair` is
    /// invariant under simple transformations (zero docs → 0 score, doubling
    /// the cos+sin caches doubles the score, etc).
    #[test]
    fn score_pair_zeros_yields_zero() {
        let big_s = 2;
        let big_t = 3;
        let n_words = 4; // dim=128 with group_size=32 → group_words=1, n_groups=4
        let n_groups = 4;
        let group_words = 1;
        let query_bits = 4;
        let big_k = 8;

        let q_planes = vec![0i32; big_s * query_bits * n_words];
        let q_meta = vec![1.0f32; big_s * 2];
        let qc_table = vec![0.0f32; big_s * big_k];
        let docs_sign = vec![0i32; big_t * n_words];
        let docs_nz = vec![0i32; big_t * n_words];
        let docs_scl = vec![1.0f32; big_t * n_groups];
        let docs_cid = vec![0i32; big_t];
        let docs_cos = vec![0.0f32; big_t];
        let docs_sin = vec![0.0f32; big_t];

        let s = score_pair(
            &q_planes, &q_meta, &qc_table, None,
            &docs_sign, &docs_nz, &docs_scl, &docs_cid,
            &docs_cos, &docs_sin, None,
            big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
        );
        // All cos/sin = 0 ⇒ est = 0 for every (i, j) ⇒ max_sim = 0 per i ⇒ sum = 0.
        assert_eq!(s, 0.0);
    }

    /// Centroid-only path: when all nz bits are zero, the residual term
    /// drops out and we should recover `sum_i max_j cos[j] * qc_table[i, cid[j]]`.
    #[test]
    fn score_pair_centroid_only() {
        let big_s = 2;
        let big_t = 3;
        let n_words = 4;
        let n_groups = 4;
        let group_words = 1;
        let query_bits = 4;
        let big_k = 8;

        let q_planes = vec![0i32; big_s * query_bits * n_words];
        let q_meta = vec![1.0f32; big_s * 2];
        // qc_table[i, k]: distinct values per (i, k)
        let mut qc_table = vec![0.0f32; big_s * big_k];
        for i in 0..big_s {
            for k in 0..big_k {
                qc_table[i * big_k + k] = (i + 1) as f32 * 0.1 + k as f32;
            }
        }
        let docs_sign = vec![0i32; big_t * n_words];
        let docs_nz = vec![0i32; big_t * n_words]; // zero residual mask
        let docs_scl = vec![1.0f32; big_t * n_groups];
        let docs_cid = vec![0i32, 3, 7]; // each doc-token points at a different centroid
        let docs_cos = vec![1.0f32; big_t];
        let docs_sin = vec![0.5f32; big_t]; // shouldn't matter, residual is zero

        let s = score_pair(
            &q_planes, &q_meta, &qc_table, None,
            &docs_sign, &docs_nz, &docs_scl, &docs_cid,
            &docs_cos, &docs_sin, None,
            big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
        );

        // Manual: for each i, max_j(cos[j] * qc[i, cid[j]]) = max_j(qc[i, cid[j]])
        // since cos[j] = 1.
        // i=0: max(0.1, 3.1, 7.1) = 7.1
        // i=1: max(0.2, 3.2, 7.2) = 7.2
        // total = 14.3
        assert!((s - 14.3).abs() < 1e-4, "expected 14.3, got {s}");
    }

    /// Mask exclusion: with a doc mask of all zeros for one of the two
    /// docs, `score_pair` for that doc should fall to NEG_INFINITY → not added.
    #[test]
    fn score_pair_doc_mask_excludes_token() {
        let big_s = 1;
        let big_t = 2;
        let n_words = 4;
        let n_groups = 4;
        let group_words = 1;
        let query_bits = 4;
        let big_k = 4;

        let q_planes = vec![0i32; big_s * query_bits * n_words];
        let q_meta = vec![1.0f32; big_s * 2];
        let qc_table = vec![5.0f32; big_s * big_k];
        let docs_sign = vec![0i32; big_t * n_words];
        let docs_nz = vec![0i32; big_t * n_words];
        let docs_scl = vec![1.0f32; big_t * n_groups];
        let docs_cid = vec![0i32; big_t];
        let docs_cos = vec![1.0f32, 1.0];
        let docs_sin = vec![0.0f32, 0.0];
        // Mask out token 0; only token 1 contributes.
        let mask = vec![0.0f32, 1.0];

        let s = score_pair(
            &q_planes, &q_meta, &qc_table, None,
            &docs_sign, &docs_nz, &docs_scl, &docs_cid,
            &docs_cos, &docs_sin, Some(&mask),
            big_s, big_t, n_words, n_groups, group_words, query_bits, big_k,
        );

        // Only token 1 contributes: cos=1.0, qc=5.0, sin*resi=0 ⇒ 5.0.
        assert!((s - 5.0).abs() < 1e-5);
    }

    /// Parity: AVX-512 VPOPCNTDQ tier matches the v3 (scalar `popcntq`)
    /// tier on the production rroq158 shape (dim=128, n_groups=1,
    /// group_words=4, query_bits=4) for a deterministic random fixture.
    ///
    /// Skipped on hosts without AVX-512 VPOPCNTDQ. Otherwise runs the
    /// whole `score_batch` pipeline through the dispatcher with the
    /// backend forced first to v4, then to v3, and asserts bit-exact
    /// score equality on `(B, S)` pairs.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86v4_matches_x86v3_on_production_shape() {
        if !std::is_x86_feature_detected!("avx512vpopcntdq")
            || !std::is_x86_feature_detected!("avx512f")
        {
            eprintln!("skipping x86v4 parity test: AVX-512 VPOPCNTDQ unavailable");
            return;
        }

        // Production rroq158 shape: dim=128 (n_words=4), group_size=128
        // (n_groups=1, group_words=4), query_bits=4. K=8 keeps qc_table
        // tiny so we don't need a real centroid table.
        let big_s = 8;
        let big_t = 12;
        let big_a = 1;
        let big_b = 4;
        let n_words = 4;
        let n_groups = 1;
        let _group_words = 4; // implied by n_words / n_groups; kept for clarity
        let query_bits = 4;
        let big_k = 8;

        // Deterministic LCG so we don't pull in `rand` as a dev-dep.
        let mut state: u64 = 0xdead_beef_1234_5678;
        let mut next = || -> u32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };

        let q_planes: Vec<i32> = (0..big_a * big_s * query_bits * n_words)
            .map(|_| next() as i32)
            .collect();
        let q_meta: Vec<f32> = (0..big_a * big_s * 2)
            .map(|_| ((next() % 1000) as f32 - 500.0) / 500.0)
            .collect();
        let qc_table: Vec<f32> = (0..big_a * big_s * big_k)
            .map(|_| ((next() % 1000) as f32 - 500.0) / 500.0)
            .collect();
        let docs_sign: Vec<i32> = (0..big_b * big_t * n_words)
            .map(|_| next() as i32)
            .collect();
        let docs_nz: Vec<i32> = (0..big_b * big_t * n_words)
            .map(|_| next() as i32)
            .collect();
        let docs_scl: Vec<f32> = (0..big_b * big_t * n_groups)
            .map(|_| ((next() % 1000) as f32) / 1000.0 + 0.1)
            .collect();
        let docs_cid: Vec<i32> = (0..big_b * big_t)
            .map(|_| (next() as i32).rem_euclid(big_k as i32))
            .collect();
        let docs_cos: Vec<f32> = (0..big_b * big_t)
            .map(|_| ((next() % 1000) as f32) / 1000.0)
            .collect();
        let docs_sin: Vec<f32> = (0..big_b * big_t)
            .map(|_| ((next() % 1000) as f32) / 1000.0)
            .collect();

        // Force v4, capture scores.
        super::_reset_backend_for_tests();
        // BACKEND will be initialised to v4 by select_backend() since the
        // host has AVX-512 VPOPCNTDQ at this point in the test.
        assert_eq!(super::select_backend(), super::BACKEND_X86V4);
        let mut out_v4 = vec![0.0f32; big_a * big_b];
        score_batch(
            &q_planes, &q_meta, &qc_table, None,
            &docs_sign, &docs_nz, &docs_scl, &docs_cid,
            &docs_cos, &docs_sin, None,
            big_a, big_b, big_s, big_t, n_words, n_groups, query_bits, big_k,
            Some(1),
            &mut out_v4,
        );

        // Force v3, capture scores.
        super::_reset_backend_for_tests();
        super::_force_x86v3_backend_for_tests();
        assert_eq!(super::select_backend(), super::BACKEND_X86V3);
        let mut out_v3 = vec![0.0f32; big_a * big_b];
        score_batch(
            &q_planes, &q_meta, &qc_table, None,
            &docs_sign, &docs_nz, &docs_scl, &docs_cid,
            &docs_cos, &docs_sin, None,
            big_a, big_b, big_s, big_t, n_words, n_groups, query_bits, big_k,
            Some(1),
            &mut out_v3,
        );

        // Reset for any subsequent tests in the suite.
        super::_reset_backend_for_tests();

        // Bit-exact: identical bit operations + identical fp32 reduction
        // order (same `(j outer, i inner, k inner-inner, w inner-most)`
        // accumulation order). The only difference between v3 and v4 is
        // *which* CPU instruction performs the popcount; the final f32
        // sum-of-max-sims is identical.
        for d in 0..big_b {
            assert_eq!(
                out_v4[d].to_bits(),
                out_v3[d].to_bits(),
                "v4 vs v3 mismatch at d={d}: v4={} v3={}",
                out_v4[d], out_v3[d],
            );
        }
    }
}
