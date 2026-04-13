/// Fused MaxSim scorer that operates directly on merged mmap FP16 data.
///
/// Reads FP16 embeddings from the mmap, converts to f32 on the fly, computes
/// MaxSim(query, doc) = Σ_i max_j dot(q_i, d_j), and returns top-k via
/// heap extraction — all without leaving Rust or allocating large buffers.
///
/// This eliminates the Rust→Python→NumPy→Torch staging that previously
/// dominated CPU latency on the exact scoring path.
use rayon::prelude::*;

use crate::merged_mmap::MergedMmap;
use crate::simd_proxy::dot_product;
use crate::topk::heap_topk;
use crate::types::ScoredDoc;

/// Score each candidate document against a query using exact MaxSim,
/// reading FP16 embeddings directly from the merged mmap.
///
/// Returns top-k `(doc_ids, scores)` sorted by score descending.
///
/// # Arguments
/// * `query_f32` - query token embeddings, row-major `[n_q * dim]`
/// * `candidate_ids` - document IDs to score
/// * `merged` - merged mmap backing store
/// * `dim` - embedding dimension
/// * `k` - number of results to return
pub fn fused_maxsim_topk(
    query_f32: &[f32],
    candidate_ids: &[u64],
    merged: &MergedMmap,
    dim: usize,
    k: usize,
) -> Vec<ScoredDoc> {
    if dim == 0 || query_f32.is_empty() || candidate_ids.is_empty() || k == 0 {
        return Vec::new();
    }
    let n_q = query_f32.len() / dim;
    if n_q == 0 {
        return Vec::new();
    }

    let scored: Vec<ScoredDoc> = candidate_ids
        .par_iter()
        .map_with(Vec::<f32>::new(), |doc_buf, &did| {
            let raw = match merged.get_embeddings_f16_bytes(did) {
                Some(r) if !r.is_empty() => r,
                _ => return None,
            };
            let bytes_per_token = dim * 2;
            let n_d = raw.len() / bytes_per_token;
            if n_d == 0 {
                return None;
            }
            let score = maxsim_f16_slice(query_f32, raw, dim, n_q, n_d, doc_buf);
            Some(ScoredDoc { doc_id: did, score })
        })
        .flatten()
        .collect();

    heap_topk(scored.into_iter(), k)
}

/// Compute MaxSim between an f32 query and FP16 document bytes.
///
/// Converts the entire document to f32 once, then runs MaxSim over it.
/// For a 256-token × 128-dim document this is ~128 KB — fits comfortably
/// in L2 cache and avoids repeated f16→f32 conversion per query token.
#[inline]
fn maxsim_f16_slice(
    query_f32: &[f32],
    doc_f16_bytes: &[u8],
    dim: usize,
    n_q: usize,
    n_d: usize,
    doc_f32_buf: &mut Vec<f32>,
) -> f32 {
    let total_floats = n_d * dim;
    doc_f32_buf.resize(total_floats, 0.0);
    f16_bytes_to_f32(&doc_f16_bytes[..n_d * dim * 2], doc_f32_buf);

    let mut total = 0.0f32;
    for qi in 0..n_q {
        let q_start = qi * dim;
        let q_row = &query_f32[q_start..q_start + dim];
        let mut max_dot = f32::NEG_INFINITY;

        for dj in 0..n_d {
            let d_start = dj * dim;
            let dot = dot_product(q_row, &doc_f32_buf[d_start..d_start + dim]);
            if dot > max_dot {
                max_dot = dot;
            }
        }
        total += max_dot;
    }
    total
}

/// Batch-convert raw FP16 bytes (little-endian) to f32.
///
/// Uses the `half` crate's optimized batch conversion which leverages
/// platform-specific SIMD when available (F16C on x86_64).
#[inline]
fn f16_bytes_to_f32(src: &[u8], dst: &mut [f32]) {
    use half::prelude::*;
    let n = dst.len();
    // Safety: src points to packed little-endian f16 values.
    // f16 and u16 have identical size/alignment, and the mmap is
    // page-aligned so byte_start (header + even offset) is 2-byte aligned.
    let src_f16: &[half::f16] = unsafe {
        std::slice::from_raw_parts(src.as_ptr() as *const half::f16, n)
    };
    src_f16.convert_to_f32_slice(dst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_f16_bytes_to_f32() {
        let one = f16::from_f32(1.0);
        let half = f16::from_f32(0.5);
        let src: Vec<u8> = [one.to_le_bytes(), half.to_le_bytes()].concat();
        let mut dst = [0.0f32; 2];
        f16_bytes_to_f32(&src, &mut dst);
        assert!((dst[0] - 1.0).abs() < 1e-3);
        assert!((dst[1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_maxsim_f16_slice_identity() {
        let dim = 4;
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let doc_f16: Vec<u8> = [
            f16::from_f32(1.0), f16::from_f32(0.0),
            f16::from_f32(0.0), f16::from_f32(0.0),
            f16::from_f32(0.0), f16::from_f32(1.0),
            f16::from_f32(0.0), f16::from_f32(0.0),
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

        let mut buf = Vec::new();
        let score = maxsim_f16_slice(&q, &doc_f16, dim, 2, 2, &mut buf);
        assert!((score - 2.0).abs() < 1e-2, "expected ~2.0, got {score}");
    }
}
