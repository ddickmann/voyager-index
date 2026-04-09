use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoStageCodebook {
    /// Fine centroids: (n_fine, dim)
    pub cquant: Vec<f32>,
    pub n_fine: usize,
    pub dim: usize,
    /// Coarse cluster labels for each fine centroid
    pub cindex_labels: Vec<u32>,
    pub n_coarse: usize,
    /// Pairwise distances between fine centroids: (n_fine, n_fine) row-major
    pub centroid_dists: Vec<f32>,
    /// IDF weights per fine centroid
    pub idf: Vec<f32>,
}

impl TwoStageCodebook {
    pub fn build(
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
        n_fine: usize,
        n_coarse: usize,
        max_iter: usize,
        seed: u64,
    ) -> Self {
        if n_vectors == 0 || dim == 0 {
            return Self {
                cquant: Vec::new(),
                n_fine: 0,
                dim,
                cindex_labels: Vec::new(),
                n_coarse: 0,
                centroid_dists: Vec::new(),
                idf: Vec::new(),
            };
        }

        let n_fine = n_fine.min(n_vectors).max(2);
        let n_coarse = n_coarse.min(n_fine / 2).max(2);

        let data = ArrayView2::from_shape((n_vectors, dim), vectors)
            .expect("vector shape mismatch");

        let normalized = l2_normalize_rows(&data.to_owned());
        let (centroids, _labels) = kmeans(&normalized.view(), n_fine, max_iter, seed);

        let (_coarse_centroids, cindex_labels) =
            kmeans(&centroids.view(), n_coarse, max_iter, seed + 1);
        let cindex_labels: Vec<u32> = cindex_labels.iter().map(|&l| l as u32).collect();

        let centroid_dists = pairwise_l2_flat(&centroids.view());

        let cquant_flat: Vec<f32> = centroids.as_slice().unwrap().to_vec();

        let idf = vec![1.0f32; n_fine];

        Self {
            cquant: cquant_flat,
            n_fine,
            dim,
            cindex_labels,
            n_coarse,
            centroid_dists,
            idf,
        }
    }

    pub fn cquant_matrix(&self) -> ArrayView2<'_, f32> {
        ArrayView2::from_shape((self.n_fine, self.dim), &self.cquant)
            .expect("codebook shape mismatch")
    }

    pub fn assign_vectors(&self, vectors: &[f32], n_vectors: usize) -> Vec<u32> {
        let cquant = self.cquant_matrix();
        let data = ArrayView2::from_shape((n_vectors, self.dim), vectors)
            .expect("vector shape mismatch");
        let normalized = l2_normalize_rows(&data.to_owned());

        (0..n_vectors)
            .into_par_iter()
            .map(|i| {
                let row = normalized.row(i);
                nearest_centroid(&row, &cquant)
            })
            .collect()
    }

    pub fn centroid_dist(&self, a: u32, b: u32) -> f32 {
        let ai = a as usize;
        let bi = b as usize;
        if ai >= self.n_fine || bi >= self.n_fine {
            return f32::MAX;
        }
        self.centroid_dists[ai * self.n_fine + bi]
    }

    pub fn update_idf(&mut self, doc_centroid_sets: &[Vec<u32>]) {
        let n_docs = doc_centroid_sets.len().max(1) as f32;
        let mut df = vec![0.0f32; self.n_fine];
        for doc_cids in doc_centroid_sets {
            let mut seen = vec![false; self.n_fine];
            for &cid in doc_cids {
                let idx = cid as usize;
                if idx < self.n_fine && !seen[idx] {
                    df[idx] += 1.0;
                    seen[idx] = true;
                }
            }
        }
        self.idf = df
            .iter()
            .map(|&d| ((n_docs + 1.0) / (d + 1.0)).ln())
            .collect();
    }

    /// Refine centroids by running weighted k-means iterations.
    /// Rare centroids (high IDF) get higher weight in the objective,
    /// improving discrimination for tail tokens.
    pub fn refine_centroids_idf(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        max_iter: usize,
    ) {
        if self.idf.len() != self.n_fine || n_vectors == 0 || self.dim == 0 {
            return;
        }

        let data = ArrayView2::from_shape((n_vectors, self.dim), vectors)
            .expect("vector shape mismatch");
        let normalized = l2_normalize_rows(&data.to_owned());

        let mut centroids = Array2::from_shape_vec(
            (self.n_fine, self.dim),
            self.cquant.clone(),
        ).expect("centroid reshape");

        for _iter in 0..max_iter {
            let assignments: Vec<u32> = (0..n_vectors)
                .into_par_iter()
                .map(|i| {
                    let row = normalized.row(i);
                    nearest_centroid(&row, &centroids.view())
                })
                .collect();

            let mut new_centroids = Array2::<f32>::zeros((self.n_fine, self.dim));
            let mut weights = vec![0.0f32; self.n_fine];

            for (i, &cid) in assignments.iter().enumerate() {
                let c = cid as usize;
                if c >= self.n_fine {
                    continue;
                }
                let w = self.idf[c];
                for d in 0..self.dim {
                    new_centroids[(c, d)] += normalized[(i, d)] * w;
                }
                weights[c] += w;
            }

            for c in 0..self.n_fine {
                if weights[c] > 0.0 {
                    for d in 0..self.dim {
                        new_centroids[(c, d)] /= weights[c];
                    }
                    let norm: f32 = (0..self.dim)
                        .map(|d| new_centroids[(c, d)] * new_centroids[(c, d)])
                        .sum::<f32>()
                        .sqrt()
                        .max(1e-10);
                    for d in 0..self.dim {
                        new_centroids[(c, d)] /= norm;
                    }
                } else {
                    for d in 0..self.dim {
                        new_centroids[(c, d)] = centroids[(c, d)];
                    }
                }
            }

            centroids = new_centroids;
        }

        self.cquant = centroids.as_slice().unwrap().to_vec();
        self.centroid_dists = pairwise_l2_flat(&centroids.view());
    }

    /// Compute query-centroid dot product matrix using matrixmultiply sgemm.
    ///
    /// Computes scores = query_vecs @ cquant^T, shape (n_query, n_fine).
    /// matrixmultiply auto-detects AVX2+FMA and uses optimized micro-kernels.
    pub fn compute_query_centroid_scores(&self, query_vecs: &[f32], n_query: usize) -> Vec<f32> {
        let mut scores = vec![0.0f32; n_query * self.n_fine];
        self.compute_query_centroid_scores_into(query_vecs, n_query, &mut scores);
        scores
    }

    /// Write query-centroid dot products into a pre-allocated buffer.
    /// Buffer must have length >= n_query * n_fine.
    pub fn compute_query_centroid_scores_into(
        &self,
        query_vecs: &[f32],
        n_query: usize,
        out: &mut [f32],
    ) {
        let m = n_query;
        let k = self.dim;
        let n = self.n_fine;
        assert!(
            query_vecs.len() >= m * k,
            "sgemm: query_vecs too short ({} < {})", query_vecs.len(), m * k
        );
        assert!(
            out.len() >= m * n,
            "sgemm: output buffer too short ({} < {})", out.len(), m * n
        );
        assert!(
            self.cquant.len() >= n * k,
            "sgemm: codebook too short ({} < {})", self.cquant.len(), n * k
        );

        if m == 0 || k == 0 || n == 0 {
            return;
        }

        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                1.0,
                query_vecs.as_ptr(), k as isize, 1,
                self.cquant.as_ptr(), 1, k as isize,
                0.0,
                out.as_mut_ptr(), n as isize, 1,
            );
        }
    }

    /// Scalar fallback (kept for testing correctness)
    #[cfg(test)]
    pub fn compute_query_centroid_scores_scalar(&self, query_vecs: &[f32], n_query: usize) -> Vec<f32> {
        let n_fine = self.n_fine;
        let dim = self.dim;
        let mut scores = vec![0.0f32; n_query * n_fine];
        let cquant = &self.cquant;
        for qi in 0..n_query {
            let q_off = qi * dim;
            for ci in 0..n_fine {
                let c_off = ci * dim;
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += query_vecs[q_off + d] * cquant[c_off + d];
                }
                scores[qi * n_fine + ci] = dot;
            }
        }
        scores
    }
}

/// Compute qCH: for each query centroid, find min distance to any doc centroid
pub fn qch_score(
    codebook: &TwoStageCodebook,
    query_cids: &[u32],
    doc_cids: &[u32],
) -> f32 {
    if query_cids.is_empty() || doc_cids.is_empty() {
        return f32::MAX;
    }
    let mut total = 0.0f32;
    for &qc in query_cids {
        let mut min_d = f32::MAX;
        for &dc in doc_cids {
            let d = codebook.centroid_dist(qc, dc);
            if d < min_d {
                min_d = d;
            }
        }
        total += min_d;
    }
    total / query_cids.len() as f32
}

/// GEM-style proxy score using pre-computed query-centroid dot products (u32 codes).
/// Kept for backwards compatibility with DocProfile.centroid_ids.
pub fn qch_proxy_score(
    query_centroid_scores: &[f32],
    n_query_vecs: usize,
    n_centroids: usize,
    doc_codes: &[u32],
) -> f32 {
    if n_query_vecs == 0 || doc_codes.is_empty() {
        return f32::MAX;
    }
    let mut total = 0.0f32;
    for qi in 0..n_query_vecs {
        let row_offset = qi * n_centroids;
        let mut max_score = f32::NEG_INFINITY;
        for &dc in doc_codes {
            let idx = dc as usize;
            if idx < n_centroids {
                let s = query_centroid_scores[row_offset + idx];
                if s > max_score {
                    max_score = s;
                }
            }
        }
        if max_score > f32::NEG_INFINITY {
            total += max_score;
        }
    }
    1.0 - total / n_query_vecs as f32
}

/// Fast proxy score for flat u16 codes — the hot scoring path.
/// Dispatches to AVX2 gather when available, scalar fallback otherwise.
pub fn qch_proxy_score_u16(
    query_centroid_scores: &[f32],
    n_query_vecs: usize,
    n_centroids: usize,
    doc_codes: &[u16],
) -> f32 {
    if n_query_vecs == 0 || doc_codes.is_empty() {
        return f32::MAX;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                qch_proxy_score_u16_avx2(query_centroid_scores, n_query_vecs, n_centroids, doc_codes)
            };
        }
    }

    qch_proxy_score_u16_scalar(query_centroid_scores, n_query_vecs, n_centroids, doc_codes)
}

fn qch_proxy_score_u16_scalar(
    query_centroid_scores: &[f32],
    n_query_vecs: usize,
    n_centroids: usize,
    doc_codes: &[u16],
) -> f32 {
    let scores_len = query_centroid_scores.len();
    let mut total = 0.0f32;
    for qi in 0..n_query_vecs {
        let start = qi * n_centroids;
        let end = start + n_centroids;
        if end > scores_len {
            continue;
        }
        let row = &query_centroid_scores[start..end];
        let mut max_score = f32::NEG_INFINITY;
        for &dc in doc_codes {
            let idx = dc as usize;
            if idx < n_centroids {
                let s = row[idx];
                if s > max_score {
                    max_score = s;
                }
            }
        }
        if max_score > f32::NEG_INFINITY {
            total += max_score;
        }
    }
    1.0 - total / n_query_vecs as f32
}

/// AVX2 gather-based proxy scoring.
/// For each query row, gathers 8 scores at a time via centroid code indices,
/// tracks running max with _mm256_max_ps, then horizontal reduces.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn qch_proxy_score_u16_avx2(
    query_centroid_scores: &[f32],
    n_query_vecs: usize,
    n_centroids: usize,
    doc_codes: &[u16],
) -> f32 {
    use std::arch::x86_64::*;

    let n_codes = doc_codes.len();
    let chunks = n_codes / 8;
    let _remainder = n_codes % 8;
    let mut total = 0.0f32;

    for qi in 0..n_query_vecs {
        let row_ptr = query_centroid_scores.as_ptr().add(qi * n_centroids);

        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);

        for chunk_i in 0..chunks {
            let base = chunk_i * 8;
            // Widen u16 codes to i32 indices for gather
            let idx = _mm256_set_epi32(
                doc_codes[base + 7] as i32,
                doc_codes[base + 6] as i32,
                doc_codes[base + 5] as i32,
                doc_codes[base + 4] as i32,
                doc_codes[base + 3] as i32,
                doc_codes[base + 2] as i32,
                doc_codes[base + 1] as i32,
                doc_codes[base] as i32,
            );
            let gathered = _mm256_i32gather_ps::<4>(row_ptr, idx);
            max_vec = _mm256_max_ps(max_vec, gathered);
        }

        // Horizontal max of the 8-wide vector
        let hi128 = _mm256_extractf128_ps(max_vec, 1);
        let lo128 = _mm256_castps256_ps128(max_vec);
        let max128 = _mm_max_ps(hi128, lo128);
        let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
        let mut row_max = _mm_cvtss_f32(max32);

        // Handle remaining codes with scalar
        for &code in &doc_codes[(chunks * 8)..n_codes] {
            let idx = code as usize;
            if idx < n_centroids {
                let s = *row_ptr.add(idx);
                if s > row_max {
                    row_max = s;
                }
            }
        }

        if row_max > f32::NEG_INFINITY {
            total += row_max;
        }
    }

    1.0 - total / n_query_vecs as f32
}

/// Compute coarse cluster profile (C_top) for a set of centroid IDs
pub fn compute_ctop(
    codebook: &TwoStageCodebook,
    centroid_ids: &[u32],
    top_r: usize,
) -> Vec<u32> {
    let mut coarse_scores = vec![0.0f32; codebook.n_coarse];
    for &cid in centroid_ids {
        let idx = cid as usize;
        if idx >= codebook.n_fine {
            continue;
        }
        let coarse_label = codebook.cindex_labels[idx] as usize;
        if coarse_label < codebook.n_coarse {
            coarse_scores[coarse_label] += codebook.idf[idx];
        }
    }
    let mut indexed: Vec<(usize, f32)> = coarse_scores
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0.0)
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(top_r);
    indexed.iter().map(|&(i, _)| i as u32).collect()
}

/// Compute coarse cluster profile with adaptive cutoff via decision tree.
///
/// Instead of a fixed top_r, uses a CutoffTree to predict the optimal number
/// of clusters based on per-document features (TF-IDF scores and doc length).
pub fn compute_ctop_adaptive(
    codebook: &TwoStageCodebook,
    centroid_ids: &[u32],
    tree: &crate::adaptive_cutoff::CutoffTree,
    r_max: usize,
) -> Vec<u32> {
    let mut coarse_scores = vec![0.0f32; codebook.n_coarse];
    for &cid in centroid_ids {
        let idx = cid as usize;
        if idx >= codebook.n_fine {
            continue;
        }
        let coarse_label = codebook.cindex_labels[idx] as usize;
        if coarse_label < codebook.n_coarse {
            coarse_scores[coarse_label] += codebook.idf[idx];
        }
    }

    let mut indexed: Vec<(usize, f32)> = coarse_scores
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0.0)
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Build feature vector: top r_max scores + normalized doc length
    let mut features = vec![0.0f32; r_max + 1];
    for (i, &(_, score)) in indexed.iter().take(r_max).enumerate() {
        features[i] = score;
    }
    features[r_max] = centroid_ids.len() as f32 / 128.0; // normalize by typical doc length

    let predicted_r = tree.predict(&features).max(1).min(indexed.len());

    indexed.truncate(predicted_r);
    indexed.iter().map(|&(i, _)| i as u32).collect()
}

pub fn cluster_overlap(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    for &x in a {
        for &y in b {
            if x == y {
                count += 1;
                break;
            }
        }
    }
    count
}

// -- internal helpers --

fn l2_normalize_rows(a: &Array2<f32>) -> Array2<f32> {
    let norms = a.map_axis(Axis(1), |row| {
        let n = row.dot(&row).sqrt();
        if n < 1e-8 { 1e-8 } else { n }
    });
    let mut out = a.clone();
    for (mut row, norm) in out.rows_mut().into_iter().zip(norms.iter()) {
        row /= *norm;
    }
    out
}

fn nearest_centroid(vec: &ArrayView1<f32>, centroids: &ArrayView2<f32>) -> u32 {
    let mut best_idx = 0u32;
    let mut best_dist = f32::MAX;
    for (i, centroid) in centroids.rows().into_iter().enumerate() {
        let d = squared_l2(vec, &centroid);
        if d < best_dist {
            best_dist = d;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn squared_l2(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn kmeans(
    data: &ArrayView2<f32>,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Array2<f32>, Vec<usize>) {
    let n = data.nrows();
    let dim = data.ncols();
    let k = k.min(n);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    indices.truncate(k);

    let mut centroids = Array2::zeros((k, dim));
    for (ci, &idx) in indices.iter().enumerate() {
        centroids.row_mut(ci).assign(&data.row(idx));
    }

    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        let new_labels: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = data.row(i);
                nearest_centroid(&row, &centroids.view()) as usize
            })
            .collect();

        if new_labels == labels {
            break;
        }
        labels = new_labels;

        let mut sums: Array2<f32> = Array2::zeros((k, dim));
        let mut counts = vec![0usize; k];
        for (i, &label) in labels.iter().enumerate() {
            let row = data.row(i);
            for d in 0..dim {
                sums[[label, d]] += row[d];
            }
            counts[label] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                for d in 0..dim {
                    centroids[[c, d]] = sums[[c, d]] / cnt;
                }
            }
        }
    }

    let centroids = l2_normalize_rows(&centroids);
    (centroids, labels)
}

fn pairwise_l2_flat(centroids: &ArrayView2<f32>) -> Vec<f32> {
    let n = centroids.nrows();
    let mut dists = vec![0.0f32; n * n];
    for i in 0..n {
        let ri = centroids.row(i);
        for j in i..n {
            let rj = centroids.row(j);
            let d = squared_l2(&ri, &rj).sqrt();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_build_and_assign() {
        let dim = 8;
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = vec![0.0f32; n * dim];
        for v in data.iter_mut() {
            *v = rng.gen::<f32>() - 0.5;
        }

        let cb = TwoStageCodebook::build(&data, n, dim, 8, 4, 10, 42);
        assert_eq!(cb.n_fine, 8);
        assert_eq!(cb.n_coarse, 4);
        assert_eq!(cb.dim, dim);
        assert_eq!(cb.cquant.len(), 8 * dim);
        assert_eq!(cb.cindex_labels.len(), 8);
        assert_eq!(cb.centroid_dists.len(), 64);

        let labels = cb.assign_vectors(&data, n);
        assert_eq!(labels.len(), n);
        for &l in &labels {
            assert!((l as usize) < cb.n_fine);
        }
    }

    #[test]
    fn test_qch_proxy_bounds_check() {
        let dim = 4;
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let cb = TwoStageCodebook::build(&data, 2, dim, 2, 2, 5, 0);
        let scores = vec![0.5f32; 2 * 2];
        let result = qch_proxy_score(&scores, 2, 2, &[0, 1, 99]);
        assert!(result.is_finite());

        assert_eq!(cb.centroid_dist(0, 99), f32::MAX);
    }

    #[test]
    fn test_sgemm_matches_scalar() {
        let dim = 128;
        let n = 200;
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = vec![0.0f32; n * dim];
        for v in data.iter_mut() {
            *v = rng.gen::<f32>() - 0.5;
        }
        let cb = TwoStageCodebook::build(&data, n, dim, 32, 8, 10, 42);

        let n_q = 16;
        let mut query = vec![0.0f32; n_q * dim];
        for v in query.iter_mut() {
            *v = rng.gen::<f32>() - 0.5;
        }

        let fast = cb.compute_query_centroid_scores(&query, n_q);
        let scalar = cb.compute_query_centroid_scores_scalar(&query, n_q);

        assert_eq!(fast.len(), scalar.len());
        for i in 0..fast.len() {
            assert!(
                (fast[i] - scalar[i]).abs() < 1e-4,
                "mismatch at {}: sgemm={} vs scalar={}", i, fast[i], scalar[i]
            );
        }
    }

    #[test]
    fn test_u16_proxy_matches_u32() {
        let dim = 8;
        let n = 50;
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = vec![0.0f32; n * dim];
        for v in data.iter_mut() {
            *v = rng.gen::<f32>() - 0.5;
        }
        let cb = TwoStageCodebook::build(&data, n, dim, 16, 4, 10, 42);

        let n_q = 4;
        let mut query = vec![0.0f32; n_q * dim];
        for v in query.iter_mut() {
            *v = rng.gen::<f32>() - 0.5;
        }
        let scores = cb.compute_query_centroid_scores(&query, n_q);

        let codes_u32: Vec<u32> = vec![0, 3, 7, 15, 1, 5, 10, 12];
        let codes_u16: Vec<u16> = codes_u32.iter().map(|&c| c as u16).collect();

        let result_u32 = qch_proxy_score(&scores, n_q, cb.n_fine, &codes_u32);
        let result_u16 = qch_proxy_score_u16(&scores, n_q, cb.n_fine, &codes_u16);

        assert!(
            (result_u32 - result_u16).abs() < 1e-6,
            "u32={} vs u16={}", result_u32, result_u16
        );
    }

    #[test]
    fn test_qch_score() {
        let dim = 4;
        let data = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let cb = TwoStageCodebook::build(&data, 4, dim, 4, 2, 5, 0);
        let score = qch_score(&cb, &[0, 1], &[0, 1]);
        assert!(score >= 0.0);
        assert!(score < 2.0);
    }

    #[test]
    fn test_cluster_overlap() {
        assert_eq!(cluster_overlap(&[0, 1, 2], &[1, 3]), 1);
        assert_eq!(cluster_overlap(&[0, 1, 2], &[0, 1, 2]), 3);
        assert_eq!(cluster_overlap(&[0], &[1]), 0);
    }
}
