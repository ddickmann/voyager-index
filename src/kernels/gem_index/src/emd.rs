use latence_gem_router::codebook::TwoStageCodebook;

/// Compute IP-based asymmetric Chamfer distance between two documents
/// using their centroid codes. For each code in doc_a, finds the maximum
/// inner product with any code in doc_b via the centroid distance matrix.
///
/// For L2-normalized centroids: IP(a,b) = 1 - ||a-b||^2 / 2
/// Returns: 1 - avg(max_IP) in [roughly -1..2] range.
pub fn qch_proxy_between_docs(
    codebook: &TwoStageCodebook,
    codes_a: &[u16],
    codes_b: &[u16],
) -> f32 {
    if codes_a.is_empty() || codes_b.is_empty() {
        return f32::MAX;
    }
    let mut total = 0.0f32;
    for &ca in codes_a {
        let mut max_ip = f32::NEG_INFINITY;
        for &cb in codes_b {
            let l2 = codebook.centroid_dist(ca as u32, cb as u32);
            let ip = 1.0 - l2 * l2 / 2.0;
            if ip > max_ip {
                max_ip = ip;
            }
        }
        if max_ip > f32::NEG_INFINITY {
            total += max_ip;
        }
    }
    1.0 - total / codes_a.len() as f32
}

/// Symmetric Chamfer: average of both asymmetric directions.
pub fn qch_proxy_symmetric(
    codebook: &TwoStageCodebook,
    codes_a: &[u16],
    codes_b: &[u16],
) -> f32 {
    let ab = qch_proxy_between_docs(codebook, codes_a, codes_b);
    let ba = qch_proxy_between_docs(codebook, codes_b, codes_a);
    (ab + ba) / 2.0
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
}
