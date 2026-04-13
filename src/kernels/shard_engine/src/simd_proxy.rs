/// SIMD-accelerated centroid proxy scoring.
///
/// Computes `max_i(dot(query_token_i, doc_mean_d))` for each candidate
/// document and returns the top-n doc IDs by score. This is the fast
/// pre-filter that narrows candidates before full MaxSim scoring.
///
/// Dispatch order: AVX2 → NEON → scalar fallback.
use crate::topk::heap_topk;
use crate::types::{DocId, ScoredDoc};

// ───────────────────── scalar fallback ─────────────────────

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ───────────────────── x86 AVX2 ─────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // Horizontal sum of 8 floats
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let total128 = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(total128);
    // Tail
    for i in (chunks * 8)..n {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    total
}

// ───────────────────── ARM NEON ─────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 4;
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        acc = vfmaq_f32(acc, va, vb);
    }
    let mut total = vaddvq_f32(acc);
    for i in (chunks * 4)..n {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    total
}

// ───────────────────── dispatch ─────────────────────

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_neon(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// For each candidate document, compute `max_i dot(q_token_i, doc_mean_d)`,
/// then return the top `n_full` doc IDs by score, fused with heap_topk.
///
/// # Arguments
/// - `query_flat`:     query token embeddings, shape `(n_q_tokens * dim,)` row-major
/// - `doc_means_flat`: all document mean embeddings, shape `(n_all_docs * dim,)` row-major
/// - `doc_mean_ids`:   doc_id for each row in `doc_means_flat`
/// - `candidate_ids`:  subset of doc_ids to score (if empty, score all)
/// - `dim`:            embedding dimension
/// - `n_full`:         how many top candidates to return
pub fn proxy_score_topn(
    query_flat: &[f32],
    doc_means_flat: &[f32],
    doc_mean_ids: &[DocId],
    candidate_ids: &[DocId],
    dim: usize,
    n_full: usize,
) -> Vec<DocId> {
    let (ids, _scores) = proxy_score_topn_with_scores(
        query_flat, doc_means_flat, doc_mean_ids, candidate_ids, dim, n_full,
    );
    ids
}

/// Same as `proxy_score_topn` but returns (doc_ids, scores).
pub fn proxy_score_topn_with_scores(
    query_flat: &[f32],
    doc_means_flat: &[f32],
    doc_mean_ids: &[DocId],
    candidate_ids: &[DocId],
    dim: usize,
    n_full: usize,
) -> (Vec<DocId>, Vec<f32>) {
    if dim == 0 || n_full == 0 || query_flat.is_empty() || doc_means_flat.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n_q = query_flat.len() / dim;
    let n_docs = doc_mean_ids.len();

    if n_docs == 0 || n_q == 0 {
        return (Vec::new(), Vec::new());
    }

    if candidate_ids.is_empty() {
        // Fast path: score ALL docs, no HashSet overhead
        let scored_iter = (0..n_docs).filter_map(|doc_idx| {
            let d_start = doc_idx * dim;
            let d_end = d_start + dim;
            if d_end > doc_means_flat.len() {
                return None;
            }
            let doc_mean = &doc_means_flat[d_start..d_end];
            let score = max_query_dot(query_flat, doc_mean, dim, n_q);
            Some(ScoredDoc { doc_id: doc_mean_ids[doc_idx], score })
        });
        let topk = heap_topk(scored_iter, n_full);
        let ids: Vec<DocId> = topk.iter().map(|sd| sd.doc_id).collect();
        let scores: Vec<f32> = topk.iter().map(|sd| sd.score).collect();
        return (ids, scores);
    }

    // Filtered path: only score candidates
    let candidate_set: std::collections::HashSet<DocId> = candidate_ids.iter().copied().collect();

    let scored_iter = (0..n_docs).filter_map(|doc_idx| {
        let did = doc_mean_ids[doc_idx];
        if !candidate_set.contains(&did) {
            return None;
        }
        let d_start = doc_idx * dim;
        let d_end = d_start + dim;
        if d_end > doc_means_flat.len() {
            return None;
        }
        let doc_mean = &doc_means_flat[d_start..d_end];
        let score = max_query_dot(query_flat, doc_mean, dim, n_q);
        Some(ScoredDoc { doc_id: did, score })
    });

    let topk = heap_topk(scored_iter, n_full);
    let ids: Vec<DocId> = topk.iter().map(|sd| sd.doc_id).collect();
    let scores: Vec<f32> = topk.iter().map(|sd| sd.score).collect();
    (ids, scores)
}

/// Compute max over query tokens of dot(q_token, doc_mean).
#[inline]
fn max_query_dot(query_flat: &[f32], doc_mean: &[f32], dim: usize, n_q: usize) -> f32 {
    let mut max_score = f32::NEG_INFINITY;
    for qi in 0..n_q {
        let q_start = qi * dim;
        let q_end = q_start + dim;
        if q_end > query_flat.len() {
            break;
        }
        let score = dot_product(&query_flat[q_start..q_end], doc_mean);
        if score > max_score {
            max_score = score;
        }
    }
    max_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_scalar_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        assert!((dot_scalar(&a, &b) - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_dispatch() {
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let scalar_result = dot_scalar(&a, &b);
        let dispatch_result = dot_product(&a, &b);
        assert!(
            (scalar_result - dispatch_result).abs() < 1e-3,
            "scalar={scalar_result} dispatch={dispatch_result}"
        );
    }

    #[test]
    fn test_proxy_score_topn_basic() {
        let dim = 4;
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 tokens
        let doc_means = vec![
            0.9, 0.0, 0.0, 0.0, // doc 1: very aligned with token 0
            0.0, 0.0, 0.0, 1.0, // doc 2: not aligned
            0.0, 0.8, 0.0, 0.0, // doc 3: aligned with token 1
        ];
        let doc_ids = vec![1, 2, 3];

        let top = proxy_score_topn(&query, &doc_means, &doc_ids, &[], dim, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], 1); // 0.9
        assert_eq!(top[1], 3); // 0.8
    }

    #[test]
    fn test_proxy_score_with_candidate_filter() {
        let dim = 4;
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let doc_means = vec![
            0.9, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0, 0.0,
            0.1, 0.0, 0.0, 0.0,
        ];
        let doc_ids = vec![10, 20, 30];
        let candidates = vec![20, 30]; // exclude doc 10

        let top = proxy_score_topn(&query, &doc_means, &doc_ids, &candidates, dim, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], 20); // 0.5
        assert_eq!(top[1], 30); // 0.1
    }

    #[test]
    fn test_proxy_score_empty() {
        assert!(proxy_score_topn(&[], &[], &[], &[], 4, 10).is_empty());
        assert!(proxy_score_topn(&[1.0], &[], &[], &[], 1, 10).is_empty());
    }

    #[test]
    fn test_proxy_score_with_scores() {
        let dim = 2;
        let query = vec![1.0, 0.0];
        let doc_means = vec![0.5, 0.0, 0.3, 0.0];
        let doc_ids = vec![1, 2];

        let (ids, scores) = proxy_score_topn_with_scores(&query, &doc_means, &doc_ids, &[], dim, 2);
        assert_eq!(ids, vec![1, 2]);
        assert!((scores[0] - 0.5).abs() < 1e-5);
        assert!((scores[1] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_large_dim_simd() {
        let dim = 128;
        let n_docs = 1000;
        let n_q_tokens = 32;

        let query: Vec<f32> = (0..n_q_tokens * dim).map(|i| ((i % 100) as f32) * 0.01).collect();
        let doc_means: Vec<f32> = (0..n_docs * dim).map(|i| ((i % 50) as f32) * 0.02).collect();
        let doc_ids: Vec<u64> = (0..n_docs as u64).collect();

        let top = proxy_score_topn(&query, &doc_means, &doc_ids, &[], dim, 100);
        assert_eq!(top.len(), 100);

        // Verify ordering: scores should be descending
        let (_, scores) = proxy_score_topn_with_scores(&query, &doc_means, &doc_ids, &[], dim, 100);
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "scores not descending: {} < {}", w[0], w[1]);
        }
    }
}
