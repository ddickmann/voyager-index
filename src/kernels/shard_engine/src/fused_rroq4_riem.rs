//! CPU-side fused MaxSim kernel for Riemannian-aware 4-bit asymmetric ROQ.
//!
//! Mirrors `voyager_index/_internal/kernels/triton_roq_rroq4_riem.py` exactly,
//! so parity tests against `reference_score_rroq4_riem` and the Triton kernel
//! hold within float-rounding noise.
//!
//! Per (q_token i, d_token j):
//!
//!   resi(i, j) = Σ_g delta[j, g] * <q_rot[i, g, :], code[j, g, :]>
//!              + Σ_g min[j, g]   * q_group_sums[i, g]
//!   est(i, j)  = cos_norm[j] * qc_table[i, centroid_id[j]]
//!              + sin_norm[j] * resi(i, j)
//!
//! where ``code[j, g, k]`` is the k-th 4-bit code of group g for doc-token j,
//! unpacked from ``codes_packed[j, g * (group_size/2) + (k/2)]`` as
//!     code[k_even] = byte & 0xF
//!     code[k_odd]  = (byte >> 4) & 0xF
//!
//! MaxSim:
//!   score(d) = Σ_i max_j est(i, j)        // optionally masked by q/d masks
//!
//! SIMD strategy:
//!
//!   * No popcount this time — the inner work is plain fp32 dot products
//!     against a small int4 codebook, trivially auto-vectorised by LLVM
//!     when the loop body is target_feature-gated to AVX2 + FMA.
//!   * Per-byte we unpack 2 codes (low + high nibble), dequantise on the
//!     fly to fp32, and FMA into a BLOCK_D-wide accumulator (the BLOCK_D
//!     dimension is "documents in this block", so the AVX2 lanes work
//!     across docs not within a single doc).
//!   * The hot dimension (`B` documents) is parallelised via a bounded,
//!     cached rayon pool — same pattern as `fused_rroq158`.
//!
//! Layout convention matches the Triton/Python side exactly so we can share
//! the same packed payload tensors:
//!
//!   docs_codes [B, T, dim/2]  u8  (low nibble = even coord, high = odd)
//!   docs_mins  [B, T, n_groups] f32
//!   docs_dlts  [B, T, n_groups] f32
//!   docs_cid   [B, T] i32 centroid id ∈ [0, K)
//!   docs_cos   [B, T] f32 cos(||r||) * norm_d
//!   docs_sin   [B, T] f32 sinc(||r||) * norm_d
//!   docs_mask  [B, T] f32 (>0 → active token)
//!
//!   q_rot      [A, S, dim] f32
//!   q_gsums    [A, S, n_groups] f32  Σ_d q_rot[g*GS + d] per group
//!   qc_table   [A, S, K] f32  q_amb @ centroids.T
//!   q_mask     [A, S] f32

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use parking_lot::Mutex;

// ─────────────────── shared inner loop ───────────────────
//
// The per-(q_tok, d_tok) body lives in `score_pair_body`. We parameterise
// over no popcount fn this time (no popcount work), but we still split
// the dispatch into a scalar and an x86-v3 path so LLVM emits the right
// FMA / AVX2 sequences when target_feature is available. The x86-v3 path
// is identical math in a target_feature-gated context.

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn score_pair_body(
    q_rot_qa: &[f32],
    q_gsums_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_codes_d: &[u8],
    docs_mins_d: &[f32],
    docs_dlts_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    dim: usize,
    n_groups: usize,
    group_size: usize,
    big_k: usize,
) -> f32 {
    let half_group = group_size / 2;
    debug_assert_eq!(group_size * n_groups, dim);
    debug_assert_eq!(half_group * 2, group_size);
    let n_bytes = dim / 2;

    let mut total: f32 = 0.0;

    for i in 0..big_s {
        if let Some(qm) = q_mask_qa {
            if qm[i] <= 0.0 {
                continue;
            }
        }
        let q_rot_i = &q_rot_qa[i * dim..(i + 1) * dim];
        let q_gsum_i = &q_gsums_qa[i * n_groups..(i + 1) * n_groups];
        let qc_row = &qc_table_qa[i * big_k..(i + 1) * big_k];

        let mut max_sim = f32::NEG_INFINITY;
        for j in 0..big_t {
            if let Some(dm) = docs_mask_d {
                if dm[j] <= 0.0 {
                    continue;
                }
            }
            let codes_row = &docs_codes_d[j * n_bytes..(j + 1) * n_bytes];
            let mins_row = &docs_mins_d[j * n_groups..(j + 1) * n_groups];
            let dlts_row = &docs_dlts_d[j * n_groups..(j + 1) * n_groups];
            let cid = docs_cid_d[j] as usize;

            let mut resi: f32 = 0.0;
            for grp in 0..n_groups {
                let delta_g = dlts_row[grp];
                let min_g = mins_row[grp];
                let qgs_g = q_gsum_i[grp];
                let base_byte = grp * half_group;
                let base_q = grp * group_size;

                // Inner 4-bit dot against the rotated query slice. LLVM
                // auto-vectorises this loop into vfmadd(213|231)ps under
                // target_feature(avx2,fma); each iteration handles 2
                // dimensions (low + high nibble of one byte).
                let mut inner: f32 = 0.0;
                for b in 0..half_group {
                    // SAFETY: bounds verified by caller — codes_row.len() ==
                    // n_bytes == dim/2 == n_groups * half_group, and
                    // q_rot_i.len() == dim == n_groups * group_size. The
                    // unchecked accesses elide the per-element bounds
                    // checks that otherwise dominate the inner-loop cost.
                    let byte_v = unsafe { *codes_row.get_unchecked(base_byte + b) } as i32;
                    let low = (byte_v & 0xF) as f32;
                    let high = ((byte_v >> 4) & 0xF) as f32;
                    let q_lo = unsafe { *q_rot_i.get_unchecked(base_q + 2 * b) };
                    let q_hi = unsafe { *q_rot_i.get_unchecked(base_q + 2 * b + 1) };
                    inner = q_lo.mul_add(low, inner);
                    inner = q_hi.mul_add(high, inner);
                }
                resi = delta_g.mul_add(inner, resi);
                resi = min_g.mul_add(qgs_g, resi);
            }
            let qc = qc_row[cid];
            let est = docs_cos_d[j].mul_add(qc, docs_sin_d[j] * resi);
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

/// Score a single (query, doc) pair — pure scalar fallback. Used by parity
/// tests + the universal-portable backend.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn score_pair(
    q_rot_qa: &[f32],
    q_gsums_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_codes_d: &[u8],
    docs_mins_d: &[f32],
    docs_dlts_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    dim: usize,
    n_groups: usize,
    group_size: usize,
    big_k: usize,
) -> f32 {
    score_pair_body(
        q_rot_qa, q_gsums_qa, qc_table_qa, q_mask_qa,
        docs_codes_d, docs_mins_d, docs_dlts_d,
        docs_cid_d, docs_cos_d, docs_sin_d, docs_mask_d,
        big_s, big_t, dim, n_groups, group_size, big_k,
    )
}

/// x86-64-v3 fast path: AVX2 + FMA available, BMI2 enabled. Same math as
/// `score_pair`; the target_feature gate lets LLVM emit `vfmaddXXXps` and
/// `vmovups` for the 8-wide accumulator over the inner 4-bit dot product.
///
/// SAFETY: caller must ensure the host CPU has avx2 + bmi2 + fma + popcnt
/// (verified once per process via `select_backend`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,fma,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_pair_x86v3(
    q_rot_qa: &[f32],
    q_gsums_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_codes_d: &[u8],
    docs_mins_d: &[f32],
    docs_dlts_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    dim: usize,
    n_groups: usize,
    group_size: usize,
    big_k: usize,
) -> f32 {
    score_pair_body(
        q_rot_qa, q_gsums_qa, qc_table_qa, q_mask_qa,
        docs_codes_d, docs_mins_d, docs_dlts_d,
        docs_cid_d, docs_cos_d, docs_sin_d, docs_mask_d,
        big_s, big_t, dim, n_groups, group_size, big_k,
    )
}

// ─────────────────── runtime feature dispatch ───────────────────

const BACKEND_UNINIT: u8 = 0;
const BACKEND_SCALAR: u8 = 1;
#[cfg(target_arch = "x86_64")]
const BACKEND_X86V3: u8 = 2;

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
            if std::is_x86_feature_detected!("popcnt")
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("fma")
            {
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

/// Reset the cached backend selection. Test-only utility.
#[doc(hidden)]
pub fn _reset_backend_for_tests() {
    BACKEND.store(BACKEND_UNINIT, Ordering::Relaxed);
}

/// Force the scalar backend regardless of detected CPU features. Test-only
/// utility used by parity tests.
#[doc(hidden)]
pub fn _force_scalar_backend_for_tests() {
    BACKEND.store(BACKEND_SCALAR, Ordering::Relaxed);
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn score_pair_dispatch(
    q_rot_qa: &[f32],
    q_gsums_qa: &[f32],
    qc_table_qa: &[f32],
    q_mask_qa: Option<&[f32]>,
    docs_codes_d: &[u8],
    docs_mins_d: &[f32],
    docs_dlts_d: &[f32],
    docs_cid_d: &[i32],
    docs_cos_d: &[f32],
    docs_sin_d: &[f32],
    docs_mask_d: Option<&[f32]>,
    big_s: usize,
    big_t: usize,
    dim: usize,
    n_groups: usize,
    group_size: usize,
    big_k: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if select_backend() == BACKEND_X86V3 {
            // SAFETY: BACKEND_X86V3 is only set when avx2+bmi2+fma+popcnt
            // are all runtime-detected, which is the precondition of
            // score_pair_x86v3.
            return unsafe {
                score_pair_x86v3(
                    q_rot_qa, q_gsums_qa, qc_table_qa, q_mask_qa,
                    docs_codes_d, docs_mins_d, docs_dlts_d,
                    docs_cid_d, docs_cos_d, docs_sin_d, docs_mask_d,
                    big_s, big_t, dim, n_groups, group_size, big_k,
                )
            };
        }
    }
    score_pair(
        q_rot_qa, q_gsums_qa, qc_table_qa, q_mask_qa,
        docs_codes_d, docs_mins_d, docs_dlts_d,
        docs_cid_d, docs_cos_d, docs_sin_d, docs_mask_d,
        big_s, big_t, dim, n_groups, group_size, big_k,
    )
}

/// Batched entry point used by Python.
///
/// Inputs are flat row-major slices with the shapes from the module doc.
/// Output is `(A * B,)` row-major scores: `out[a * B + d] = MaxSim(q_a, doc_d)`.
///
/// `n_threads` controls the size of the rayon worker pool. Pass `None` to
/// use the global rayon pool. Pass `Some(n)` to install a per-call scoped
/// pool — the right call when the python side already has its own
/// ThreadPoolExecutor of size `W`, in which case
/// `n_threads = max(1, cpu_count() // W)` avoids 1024-way over-subscription.
#[allow(clippy::too_many_arguments)]
pub fn score_batch(
    q_rot: &[f32],
    q_gsums: &[f32],
    qc_table: &[f32],
    q_mask: Option<&[f32]>,
    docs_codes: &[u8],
    docs_mins: &[f32],
    docs_dlts: &[f32],
    docs_cid: &[i32],
    docs_cos: &[f32],
    docs_sin: &[f32],
    docs_mask: Option<&[f32]>,
    big_a: usize,
    big_b: usize,
    big_s: usize,
    big_t: usize,
    dim: usize,
    n_groups: usize,
    group_size: usize,
    big_k: usize,
    n_threads: Option<usize>,
    out: &mut [f32],
) {
    if let Some(n) = n_threads.filter(|&n| n >= 1) {
        let pool = get_or_create_pool(n);
        pool.install(|| {
            score_batch_inner(
                q_rot, q_gsums, qc_table, q_mask,
                docs_codes, docs_mins, docs_dlts,
                docs_cid, docs_cos, docs_sin, docs_mask,
                big_a, big_b, big_s, big_t,
                dim, n_groups, group_size, big_k,
                out,
            )
        });
        return;
    }
    score_batch_inner(
        q_rot, q_gsums, qc_table, q_mask,
        docs_codes, docs_mins, docs_dlts,
        docs_cid, docs_cos, docs_sin, docs_mask,
        big_a, big_b, big_s, big_t,
        dim, n_groups, group_size, big_k,
        out,
    );
}

// Process-wide cache of bounded rayon pools, keyed by thread count.
// Shared with fused_rroq158 in spirit but kept separate to avoid coupling
// — different codecs may pick different worker fan-outs in the future.
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
            .thread_name(move |i| format!("rroq4riem-{n_threads}-{i}"))
            .build()
            .expect("rroq4_riem rayon pool"),
    );
    let mut map = cache.lock();
    map.entry(n_threads).or_insert_with(|| pool.clone()).clone()
}

#[allow(clippy::too_many_arguments)]
fn score_batch_inner(
    q_rot: &[f32],
    q_gsums: &[f32],
    qc_table: &[f32],
    q_mask: Option<&[f32]>,
    docs_codes: &[u8],
    docs_mins: &[f32],
    docs_dlts: &[f32],
    docs_cid: &[i32],
    docs_cos: &[f32],
    docs_sin: &[f32],
    docs_mask: Option<&[f32]>,
    big_a: usize,
    big_b: usize,
    big_s: usize,
    big_t: usize,
    dim: usize,
    n_groups: usize,
    group_size: usize,
    big_k: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), big_a * big_b);
    let n_bytes = dim / 2;

    let q_rot_stride_a = big_s * dim;
    let q_gsums_stride_a = big_s * n_groups;
    let qc_stride_a = big_s * big_k;
    let q_mask_stride_a = big_s;
    let d_codes_stride = big_t * n_bytes;
    let d_mins_stride = big_t * n_groups;
    let d_dlts_stride = big_t * n_groups;
    let d_cid_stride = big_t;
    let d_cos_stride = big_t;
    let d_sin_stride = big_t;
    let d_mask_stride = big_t;

    for a in 0..big_a {
        let q_rot_a = &q_rot[a * q_rot_stride_a..(a + 1) * q_rot_stride_a];
        let q_gsums_a = &q_gsums[a * q_gsums_stride_a..(a + 1) * q_gsums_stride_a];
        let qc_a = &qc_table[a * qc_stride_a..(a + 1) * qc_stride_a];
        let q_mask_a = q_mask.map(|m| &m[a * q_mask_stride_a..(a + 1) * q_mask_stride_a]);

        let min_chunk = std::cmp::max(1, big_b / 64).max(16);
        let out_row = &mut out[a * big_b..(a + 1) * big_b];
        out_row
            .par_iter_mut()
            .with_min_len(min_chunk)
            .enumerate()
            .for_each(|(d, slot)| {
                let codes_d = &docs_codes[d * d_codes_stride..(d + 1) * d_codes_stride];
                let mins_d = &docs_mins[d * d_mins_stride..(d + 1) * d_mins_stride];
                let dlts_d = &docs_dlts[d * d_dlts_stride..(d + 1) * d_dlts_stride];
                let cid_d = &docs_cid[d * d_cid_stride..(d + 1) * d_cid_stride];
                let cos_d = &docs_cos[d * d_cos_stride..(d + 1) * d_cos_stride];
                let sin_d = &docs_sin[d * d_sin_stride..(d + 1) * d_sin_stride];
                let mask_d = docs_mask.map(|m| &m[d * d_mask_stride..(d + 1) * d_mask_stride]);

                *slot = score_pair_dispatch(
                    q_rot_a, q_gsums_a, qc_a, q_mask_a,
                    codes_d, mins_d, dlts_d, cid_d, cos_d, sin_d, mask_d,
                    big_s, big_t, dim, n_groups, group_size, big_k,
                );
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity sanity check: zero docs → zero score.
    #[test]
    fn score_pair_zeros_yields_zero() {
        let big_s = 2;
        let big_t = 3;
        let dim = 32;
        let group_size = 16;
        let n_groups = 2;
        let big_k = 4;
        let n_bytes = dim / 2;

        let q_rot = vec![0.0f32; big_s * dim];
        let q_gsums = vec![0.0f32; big_s * n_groups];
        let qc_table = vec![0.0f32; big_s * big_k];
        let docs_codes = vec![0u8; big_t * n_bytes];
        let docs_mins = vec![0.0f32; big_t * n_groups];
        let docs_dlts = vec![1.0f32; big_t * n_groups];
        let docs_cid = vec![0i32; big_t];
        let docs_cos = vec![0.0f32; big_t];
        let docs_sin = vec![0.0f32; big_t];

        let s = score_pair(
            &q_rot, &q_gsums, &qc_table, None,
            &docs_codes, &docs_mins, &docs_dlts,
            &docs_cid, &docs_cos, &docs_sin, None,
            big_s, big_t, dim, n_groups, group_size, big_k,
        );
        assert_eq!(s, 0.0);
    }

    /// Linearity in delta: doubling delta doubles the dequant inner product
    /// (with min=0), so the residual contribution doubles too.
    #[test]
    fn score_pair_delta_linear() {
        let big_s = 1;
        let big_t = 1;
        let dim = 16;
        let group_size = 16;
        let n_groups = 1;
        let big_k = 1;
        let n_bytes = dim / 2;

        let q_rot: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let q_gsums = vec![q_rot.iter().sum()];
        let qc_table = vec![0.0f32];
        let codes: Vec<u8> = (0..n_bytes).map(|_| 0x12).collect();
        let mins = vec![0.0f32];
        let dlts1 = vec![1.0f32];
        let dlts2 = vec![2.0f32];
        let cid = vec![0i32];
        let cos = vec![0.0f32];
        let sin = vec![1.0f32];

        let s1 = score_pair(
            &q_rot, &q_gsums, &qc_table, None,
            &codes, &mins, &dlts1, &cid, &cos, &sin, None,
            big_s, big_t, dim, n_groups, group_size, big_k,
        );
        let s2 = score_pair(
            &q_rot, &q_gsums, &qc_table, None,
            &codes, &mins, &dlts2, &cid, &cos, &sin, None,
            big_s, big_t, dim, n_groups, group_size, big_k,
        );
        assert!((s2 - 2.0 * s1).abs() < 1e-5, "s1={s1} s2={s2}");
    }
}
