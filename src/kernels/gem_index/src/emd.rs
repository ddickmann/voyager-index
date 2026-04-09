use latence_gem_router::codebook::TwoStageCodebook;

/// Compute IP-based asymmetric Chamfer distance between two documents
/// using their centroid codes. For each code in doc_a, finds the maximum
/// inner product with any code in doc_b via the centroid distance matrix.
///
/// For L2-normalized centroids: IP(a,b) = 1 - ||a-b||^2 / 2
/// Returns: 1 - avg(max_IP) in [roughly -1..2] range.
///
/// Hot path during graph construction — optimized with direct row access
/// and auto-vectorizable reduction.
#[inline]
pub fn qch_proxy_between_docs(
    codebook: &TwoStageCodebook,
    codes_a: &[u16],
    codes_b: &[u16],
) -> f32 {
    if codes_a.is_empty() || codes_b.is_empty() {
        return f32::MAX;
    }
    let n_fine = codebook.n_fine;
    let dists = &codebook.centroid_dists;

    debug_assert!(
        codes_a.iter().all(|&c| (c as usize) < n_fine),
        "codes_a contains out-of-range centroid"
    );
    debug_assert!(
        codes_b.iter().all(|&c| (c as usize) < n_fine),
        "codes_b contains out-of-range centroid"
    );

    let mut total = 0.0f32;
    for &ca in codes_a {
        let row_base = (ca as usize) * n_fine;
        let max_ip = max_ip_from_row(dists, row_base, codes_b);
        total += max_ip;
    }
    1.0 - total / codes_a.len() as f32
}

/// Core inner loop: for a single query centroid (row in distance matrix),
/// find the maximum inner-product score against all document centroids.
#[inline]
fn max_ip_from_row(dists: &[f32], row_base: usize, codes_b: &[u16]) -> f32 {
    let mut max_ip = f32::NEG_INFINITY;

    // Process in chunks of 8 for auto-vectorization
    let chunks = codes_b.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Gather 8 L2 distances and convert to IP
        let d0 = unsafe { *dists.get_unchecked(row_base + *chunk.get_unchecked(0) as usize) };
        let d1 = unsafe { *dists.get_unchecked(row_base + *chunk.get_unchecked(1) as usize) };
        let d2 = unsafe { *dists.get_unchecked(row_base + *chunk.get_unchecked(2) as usize) };
        let d3 = unsafe { *dists.get_unchecked(row_base + *chunk.get_unchecked(3) as usize) };
        let d4 = unsafe { *dists.get_unchecked(row_base + *chunk.get_unchecked(4) as usize) };
        let d5 = unsafe { *dists.get_unchecked(row_base + *chunk.get_unchecked(5) as usize) };
        let d6 = unsafe { *dists.get_unchecked(row_base + *chunk.get_unchecked(6) as usize) };
        let d7 = unsafe { *dists.get_unchecked(row_base + *chunk.get_unchecked(7) as usize) };

        let ip0 = 1.0 - d0 * d0 * 0.5;
        let ip1 = 1.0 - d1 * d1 * 0.5;
        let ip2 = 1.0 - d2 * d2 * 0.5;
        let ip3 = 1.0 - d3 * d3 * 0.5;
        let ip4 = 1.0 - d4 * d4 * 0.5;
        let ip5 = 1.0 - d5 * d5 * 0.5;
        let ip6 = 1.0 - d6 * d6 * 0.5;
        let ip7 = 1.0 - d7 * d7 * 0.5;

        let m01 = if ip0 > ip1 { ip0 } else { ip1 };
        let m23 = if ip2 > ip3 { ip2 } else { ip3 };
        let m45 = if ip4 > ip5 { ip4 } else { ip5 };
        let m67 = if ip6 > ip7 { ip6 } else { ip7 };
        let m0123 = if m01 > m23 { m01 } else { m23 };
        let m4567 = if m45 > m67 { m45 } else { m67 };
        let chunk_max = if m0123 > m4567 { m0123 } else { m4567 };

        if chunk_max > max_ip {
            max_ip = chunk_max;
        }
    }

    for &cb in remainder {
        let l2 = unsafe { *dists.get_unchecked(row_base + cb as usize) };
        let ip = 1.0 - l2 * l2 * 0.5;
        if ip > max_ip {
            max_ip = ip;
        }
    }

    max_ip
}

/// IDF-weighted qCH proxy score: weights each query token's best match
/// by the IDF of the matched centroid. Emphasizes rare/discriminative tokens.
#[inline]
pub fn qch_proxy_score_idf_weighted(
    query_centroid_scores: &[f32],
    n_query_vecs: usize,
    n_centroids: usize,
    doc_codes: &[u16],
    idf: &[f32],
) -> f32 {
    if n_query_vecs == 0 || doc_codes.is_empty() {
        return f32::MAX;
    }

    let mut weighted_total = 0.0f32;
    let mut weight_sum = 0.0f32;
    for qi in 0..n_query_vecs {
        let row_start = qi * n_centroids;
        let mut max_score = f32::NEG_INFINITY;
        let mut best_centroid = 0usize;
        for &dc in doc_codes {
            let idx = dc as usize;
            if idx < n_centroids {
                let s = unsafe { *query_centroid_scores.get_unchecked(row_start + idx) };
                if s > max_score {
                    max_score = s;
                    best_centroid = idx;
                }
            }
        }
        if max_score > f32::NEG_INFINITY {
            let w = if best_centroid < idf.len() {
                unsafe { *idf.get_unchecked(best_centroid) }
            } else {
                1.0
            };
            weighted_total += max_score * w;
            weight_sum += w;
        }
    }
    if weight_sum > 0.0 {
        1.0 - weighted_total / weight_sum
    } else {
        f32::MAX
    }
}

/// Symmetric Chamfer: average of both asymmetric directions.
#[inline]
pub fn qch_proxy_symmetric(
    codebook: &TwoStageCodebook,
    codes_a: &[u16],
    codes_b: &[u16],
) -> f32 {
    if codes_a.is_empty() || codes_b.is_empty() {
        return f32::MAX;
    }
    let ab = qch_proxy_between_docs(codebook, codes_a, codes_b);
    let ba = qch_proxy_between_docs(codebook, codes_b, codes_a);
    (ab + ba) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    #[test]
    fn test_self_distance_is_minimal() {
        let dim = 8;
        let n = 20;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let cb = TwoStageCodebook::build(&data, n, dim, 8, 4, 10, 42);
        let assignments = cb.assign_vectors(&data, n);
        let codes: Vec<u16> = assignments[0..4].iter().map(|&c| c as u16).collect();
        let self_dist = qch_proxy_between_docs(&cb, &codes, &codes);
        assert!(self_dist <= 0.1, "self distance should be near 0: {}", self_dist);
    }

    #[test]
    fn test_symmetric_matches() {
        let dim = 8;
        let n = 20;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let cb = TwoStageCodebook::build(&data, n, dim, 8, 4, 10, 42);
        let assignments = cb.assign_vectors(&data, n);
        let codes_a: Vec<u16> = assignments[0..4].iter().map(|&c| c as u16).collect();
        let codes_b: Vec<u16> = assignments[4..8].iter().map(|&c| c as u16).collect();
        let ab = qch_proxy_between_docs(&cb, &codes_a, &codes_b);
        let ba = qch_proxy_between_docs(&cb, &codes_b, &codes_a);
        let sym = qch_proxy_symmetric(&cb, &codes_a, &codes_b);
        let expected = (ab + ba) * 0.5;
        assert!((sym - expected).abs() < 1e-6);
    }
}
