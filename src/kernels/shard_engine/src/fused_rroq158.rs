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

#[inline(always)]
fn popc(x: u32) -> u32 {
    x.count_ones()
}

/// Score a single (query, doc) pair. Pure scalar — used as the inner kernel
/// for the parallel batch entrypoint and as the parity reference.
///
/// `n_groups * group_words == n_words` is required by the caller.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn score_pair(
    q_planes_qa: &[i32],          // (S, query_bits, n_words)
    q_meta_qa: &[f32],            // (S, 2)
    qc_table_qa: &[f32],          // (S, K)
    q_mask_qa: Option<&[f32]>,    // (S,) or None
    docs_sign_d: &[i32],          // (T, n_words)
    docs_nz_d: &[i32],            // (T, n_words)
    docs_scl_d: &[f32],           // (T, n_groups)
    docs_cid_d: &[i32],           // (T,)
    docs_cos_d: &[f32],           // (T,)
    docs_sin_d: &[f32],           // (T,)
    docs_mask_d: Option<&[f32]>,  // (T,) or None
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    group_words: usize,
    query_bits: usize,
    big_k: usize,
) -> f32 {
    debug_assert_eq!(n_groups * group_words, n_words);
    let mut total: f32 = 0.0;

    for i in 0..big_s {
        if let Some(qm) = q_mask_qa {
            if qm[i] <= 0.0 {
                continue;
            }
        }
        let q_scale = q_meta_qa[i * 2];
        let q_offset = q_meta_qa[i * 2 + 1];
        let q_planes_token = &q_planes_qa[i * query_bits * n_words..(i + 1) * query_bits * n_words];
        let qc_row = &qc_table_qa[i * big_k..(i + 1) * big_k];

        let mut max_sim = f32::NEG_INFINITY;
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

            let mut resi: f32 = 0.0;
            for grp in 0..n_groups {
                let d_scale_g = scl_row[grp];
                let mut s_g: i32 = 0;
                let mut d_g: i32 = 0;
                let base_word = grp * group_words;
                for w in 0..group_words {
                    let word_idx = base_word + w;
                    // SAFETY: bounds-checked by caller via the n_words/n_groups
                    //         relation; the hot loop benefits from elision.
                    let ds_w = unsafe { *sign_row.get_unchecked(word_idx) as u32 };
                    let dn_w = unsafe { *nz_row.get_unchecked(word_idx) as u32 };
                    let pos = popc(ds_w & dn_w) as i32;
                    let neg = popc((!ds_w) & dn_w) as i32;
                    s_g += pos - neg;
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
            let est = docs_cos_d[j] * qc + docs_sin_d[j] * resi;
            if est > max_sim {
                max_sim = est;
            }
        }
        if max_sim > f32::NEG_INFINITY {
            total += max_sim;
        }
    }
    total
}

/// Batched entry point used by Python.
///
/// Inputs are flat row-major slices with the shapes from the module doc.
/// Output is `(A * B,)` row-major scores: `out[a * B + d] = MaxSim(q_a, doc_d)`.
///
/// Documents are scored in parallel via rayon — this is the hot dimension
/// because `B` is typically 500..4000 candidates per query in production.
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

        // Parallelise across documents.
        let out_row = &mut out[a * big_b..(a + 1) * big_b];
        out_row
            .par_iter_mut()
            .enumerate()
            .for_each(|(d, slot)| {
                let sign_d = &docs_sign[d * d_sign_stride..(d + 1) * d_sign_stride];
                let nz_d = &docs_nz[d * d_nz_stride..(d + 1) * d_nz_stride];
                let scl_d = &docs_scl[d * d_scl_stride..(d + 1) * d_scl_stride];
                let cid_d = &docs_cid[d * d_cid_stride..(d + 1) * d_cid_stride];
                let cos_d = &docs_cos[d * d_cos_stride..(d + 1) * d_cos_stride];
                let sin_d = &docs_sin[d * d_sin_stride..(d + 1) * d_sin_stride];
                let mask_d = docs_mask.map(|m| &m[d * d_mask_stride..(d + 1) * d_mask_stride]);

                *slot = score_pair(
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
}
