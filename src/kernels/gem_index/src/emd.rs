use std::collections::HashMap;

use latence_gem_router::codebook::TwoStageCodebook;

use crate::network_simplex::emd_sinkhorn;

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
    let dists_len = dists.len();

    let mut total = 0.0f32;
    for &ca in codes_a {
        let ca_idx = ca as usize;
        if ca_idx >= n_fine {
            continue;
        }
        let row_base = ca_idx * n_fine;
        if row_base + n_fine > dists_len {
            continue;
        }
        let max_ip = max_ip_from_row(dists, row_base, codes_b, n_fine);
        total += max_ip;
    }
    1.0 - total / codes_a.len() as f32
}

/// Core inner loop: for a single query centroid (row in distance matrix),
/// find the maximum inner-product score against all document centroids.
/// `n_fine` is the row width — all codes_b entries must be < n_fine.
#[inline]
fn max_ip_from_row(dists: &[f32], row_base: usize, codes_b: &[u16], n_fine: usize) -> f32 {
    let mut max_ip = f32::NEG_INFINITY;

    let chunks = codes_b.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let c0 = chunk[0] as usize;
        let c1 = chunk[1] as usize;
        let c2 = chunk[2] as usize;
        let c3 = chunk[3] as usize;
        let c4 = chunk[4] as usize;
        let c5 = chunk[5] as usize;
        let c6 = chunk[6] as usize;
        let c7 = chunk[7] as usize;

        if c0 >= n_fine || c1 >= n_fine || c2 >= n_fine || c3 >= n_fine
            || c4 >= n_fine || c5 >= n_fine || c6 >= n_fine || c7 >= n_fine
        {
            // Fallback: process valid codes individually instead of skipping
            for &cb in chunk {
                let idx = cb as usize;
                if idx < n_fine {
                    let l2 = dists[row_base + idx];
                    let ip = 1.0 - l2 * l2 * 0.5;
                    if ip > max_ip {
                        max_ip = ip;
                    }
                }
            }
            continue;
        }

        let d0 = dists[row_base + c0];
        let d1 = dists[row_base + c1];
        let d2 = dists[row_base + c2];
        let d3 = dists[row_base + c3];
        let d4 = dists[row_base + c4];
        let d5 = dists[row_base + c5];
        let d6 = dists[row_base + c6];
        let d7 = dists[row_base + c7];

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
        let idx = cb as usize;
        if idx >= n_fine {
            continue;
        }
        let l2 = dists[row_base + idx];
        let ip = 1.0 - l2 * l2 * 0.5;
        if ip > max_ip {
            max_ip = ip;
        }
    }

    max_ip
}

/// Build a normalized histogram from centroid codes.
/// Returns (unique_ids, weights) where weights sum to 1.0.
fn build_histogram(codes: &[u16], n_fine: usize) -> (Vec<u16>, Vec<f64>) {
    let mut counts: HashMap<u16, u32> = HashMap::with_capacity(codes.len().min(n_fine));
    for &c in codes {
        if (c as usize) < n_fine {
            *counts.entry(c).or_insert(0) += 1;
        }
    }
    let total = counts.values().sum::<u32>() as f64;
    if total == 0.0 {
        return (Vec::new(), Vec::new());
    }
    let mut ids: Vec<u16> = counts.keys().copied().collect();
    ids.sort_unstable();
    let weights: Vec<f64> = ids.iter().map(|id| counts[id] as f64 / total).collect();
    (ids, weights)
}

/// Quantized Earth Mover's Distance between two documents using their centroid codes.
///
/// 1. Reduces each document's code sequence to a centroid histogram (unique centroids + weights).
/// 2. Extracts the relevant sub-matrix from the precomputed centroid distance table.
/// 3. Solves the transport problem via network simplex.
///
/// The cost matrix uses L2 distances between centroids (stored in `codebook.centroid_dists`).
/// For L2-normalized centroids: d_X(a,b) = centroid_dists[a][b] (already L2 distance).
///
/// Returns a distance value (lower = more similar). Satisfies triangle inequality.
pub fn qemd_between_docs(
    codebook: &TwoStageCodebook,
    codes_a: &[u16],
    codes_b: &[u16],
) -> f32 {
    if codes_a.is_empty() || codes_b.is_empty() {
        return f32::MAX;
    }
    let n_fine = codebook.n_fine;
    let (ids_a, weights_a) = build_histogram(codes_a, n_fine);
    let (ids_b, weights_b) = build_histogram(codes_b, n_fine);

    if ids_a.is_empty() || ids_b.is_empty() {
        return f32::MAX;
    }

    let na = ids_a.len();
    let nb = ids_b.len();

    // Extract sub-matrix of centroid distances for the relevant centroids
    let dists = &codebook.centroid_dists;
    let mut cost_matrix: Vec<f64> = Vec::with_capacity(na * nb);
    for &ca in &ids_a {
        let row_base = (ca as usize) * n_fine;
        for &cb in &ids_b {
            let d = if row_base + (cb as usize) < dists.len() {
                dists[row_base + cb as usize] as f64
            } else {
                1.0 // fallback
            };
            cost_matrix.push(d);
        }
    }

    // Sinkhorn with high regularization (lambda=100) and 200 iterations gives
    // near-exact EMD while being inherently symmetric and robust.
    let result = emd_sinkhorn(&weights_a, &weights_b, &cost_matrix, 100.0, 200);
    result as f32
}

/// Symmetric qEMD: average of both directions.
/// Since EMD with matching marginals is symmetric by definition, this just calls
/// `qemd_between_docs` once (EMD is already symmetric when both histograms sum to 1).
#[inline]
pub fn qemd_symmetric(
    codebook: &TwoStageCodebook,
    codes_a: &[u16],
    codes_b: &[u16],
) -> f32 {
    qemd_between_docs(codebook, codes_a, codes_b)
}

/// Choose between qEMD and qCH distance based on `use_emd` flag.
/// This is the primary distance function for graph construction.
#[inline]
pub fn construction_distance(
    codebook: &TwoStageCodebook,
    codes_a: &[u16],
    codes_b: &[u16],
    use_emd: bool,
) -> f32 {
    if use_emd {
        qemd_between_docs(codebook, codes_a, codes_b)
    } else {
        qch_proxy_between_docs(codebook, codes_a, codes_b)
    }
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

    let scores_len = query_centroid_scores.len();
    let mut weighted_total = 0.0f32;
    let mut weight_sum = 0.0f32;
    for qi in 0..n_query_vecs {
        let row_start = qi * n_centroids;
        if row_start + n_centroids > scores_len {
            continue;
        }
        let mut max_score = f32::NEG_INFINITY;
        let mut best_centroid = 0usize;
        for &dc in doc_codes {
            let idx = dc as usize;
            if idx < n_centroids {
                let s = query_centroid_scores[row_start + idx];
                if s > max_score {
                    max_score = s;
                    best_centroid = idx;
                }
            }
        }
        if max_score > f32::NEG_INFINITY {
            let w = if best_centroid < idf.len() {
                idf[best_centroid]
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

    #[test]
    fn test_qemd_self_distance_is_zero() {
        let dim = 8;
        let n = 20;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let cb = TwoStageCodebook::build(&data, n, dim, 8, 4, 10, 42);
        let assignments = cb.assign_vectors(&data, n);
        let codes: Vec<u16> = assignments[0..5].iter().map(|&c| c as u16).collect();
        let self_dist = qemd_between_docs(&cb, &codes, &codes);
        assert!(self_dist < 1e-4, "qEMD self-distance should be ~0: {self_dist}");
    }

    #[test]
    fn test_qemd_triangle_inequality() {
        let dim = 8;
        let n = 30;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let cb = TwoStageCodebook::build(&data, n, dim, 8, 4, 10, 42);
        let assignments = cb.assign_vectors(&data, n);
        let a: Vec<u16> = assignments[0..5].iter().map(|&c| c as u16).collect();
        let b: Vec<u16> = assignments[5..10].iter().map(|&c| c as u16).collect();
        let c: Vec<u16> = assignments[10..15].iter().map(|&c| c as u16).collect();
        let ab = qemd_between_docs(&cb, &a, &b);
        let bc = qemd_between_docs(&cb, &b, &c);
        let ac = qemd_between_docs(&cb, &a, &c);
        assert!(
            ac <= ab + bc + 1e-4,
            "triangle inequality: AC={ac} > AB={ab} + BC={bc}"
        );
    }

    #[test]
    fn test_qemd_is_symmetric() {
        let dim = 8;
        let n = 20;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let cb = TwoStageCodebook::build(&data, n, dim, 8, 4, 10, 42);
        let assignments = cb.assign_vectors(&data, n);
        let a: Vec<u16> = assignments[0..5].iter().map(|&c| c as u16).collect();
        let b: Vec<u16> = assignments[5..10].iter().map(|&c| c as u16).collect();
        let ab = qemd_between_docs(&cb, &a, &b);
        let ba = qemd_between_docs(&cb, &b, &a);
        assert!((ab - ba).abs() < 1e-4, "qEMD should be symmetric: ab={ab}, ba={ba}");
    }

    #[test]
    fn test_construction_distance_switch() {
        let dim = 8;
        let n = 20;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let cb = TwoStageCodebook::build(&data, n, dim, 8, 4, 10, 42);
        let assignments = cb.assign_vectors(&data, n);
        let a: Vec<u16> = assignments[0..5].iter().map(|&c| c as u16).collect();
        let b: Vec<u16> = assignments[5..10].iter().map(|&c| c as u16).collect();
        let emd = construction_distance(&cb, &a, &b, true);
        let ch = construction_distance(&cb, &a, &b, false);
        assert!(emd > 0.0, "qEMD should be positive: {emd}");
        assert!(ch > 0.0, "qCH should be positive: {ch}");
    }
}
