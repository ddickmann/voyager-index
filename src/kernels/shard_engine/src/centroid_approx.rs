/// Centroid-code approximate MaxSim scoring (PLAID-style).
///
/// Each document token is represented by a u16 centroid code instead of a
/// full 128-dim embedding. At query time we precompute a lookup table
/// `Q × C^T  →  [n_q_tokens, n_centroids]` and score each document via
/// table lookups:
///
///   approx_score(q, doc) = Σ_i max_j lookup[i][code_j]
///
/// This reduces per-document scoring from O(n_q × n_d × dim) FP multiply-adds
/// to O(n_q × n_d) table lookups + comparisons.
///
/// Tier 3: Rayon parallelism across shard groups for the mmap read + score loop.
use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::merged_mmap::MergedMmap;
use crate::mmap_reader::MmapShard;
use crate::simd_proxy::dot_product;
use crate::state::ShardState;
use crate::topk::heap_topk;
use crate::types::{DocId, ScoredDoc};

/// Precompute `Q × C^T` → row-major `[n_q, n_centroids]`.
#[inline]
pub fn compute_lookup_table(
    query_flat: &[f32],
    centroids_flat: &[f32],
    n_q: usize,
    n_centroids: usize,
    dim: usize,
) -> Vec<f32> {
    let mut table = vec![0.0f32; n_q * n_centroids];
    for qi in 0..n_q {
        let q_row = &query_flat[qi * dim..(qi + 1) * dim];
        let row_start = qi * n_centroids;
        for ci in 0..n_centroids {
            let c_row = &centroids_flat[ci * dim..(ci + 1) * dim];
            table[row_start + ci] = dot_product(q_row, c_row);
        }
    }
    table
}

/// Approximate MaxSim for a single document using centroid codes.
#[inline]
pub fn approximate_maxsim_single(
    lookup_table: &[f32],
    codes: &[u16],
    n_q: usize,
    n_centroids: usize,
) -> f32 {
    if codes.is_empty() || n_q == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for qi in 0..n_q {
        let row_start = qi * n_centroids;
        let mut max_score = f32::NEG_INFINITY;
        for &code in codes {
            let idx = code as usize;
            if idx >= n_centroids {
                continue;
            }
            let score = unsafe { *lookup_table.get_unchecked(row_start + idx) };
            if score > max_score {
                max_score = score;
            }
        }
        total += max_score;
    }
    total
}

/// Score a batch of candidate documents using approximate MaxSim.
///
/// Groups candidates by shard for mmap locality, then uses Rayon to
/// parallelize across shard groups. Returns top `n_top` by approximate score.
pub fn score_candidates_approx(
    query_flat: &[f32],
    centroids_flat: &[f32],
    n_centroids: usize,
    dim: usize,
    candidate_ids: &[DocId],
    state: &ShardState,
    shard_snap: &HashMap<u32, Arc<MmapShard>>,
    merged: Option<&MergedMmap>,
    n_top: usize,
) -> Vec<ScoredDoc> {
    let n_q = query_flat.len() / dim;
    if n_q == 0 || candidate_ids.is_empty() || n_centroids == 0 {
        return Vec::new();
    }

    let lookup_table = compute_lookup_table(query_flat, centroids_flat, n_q, n_centroids, dim);

    // Fast path: merged mmap codes available, no shard-level reads needed.
    if let Some(mm) = merged {
        if mm.has_codes() {
            let all_scored: Vec<ScoredDoc> = candidate_ids
                .par_iter()
                .filter_map(|&did| {
                    let codes = mm.get_codes(did)?;
                    let score =
                        approximate_maxsim_single(&lookup_table, codes, n_q, n_centroids);
                    Some(ScoredDoc { doc_id: did, score })
                })
                .collect();
            return heap_topk(all_scored.into_iter(), n_top);
        }
    }

    // Group candidates by shard for mmap locality
    let mut by_shard: HashMap<u32, Vec<(DocId, usize, usize)>> = HashMap::new();
    for &did in candidate_ids {
        if let Some(meta) = state.docs.get(&did) {
            if meta.shard_id != u32::MAX {
                by_shard
                    .entry(meta.shard_id)
                    .or_default()
                    .push((did, meta.row_start, meta.row_end));
            }
        }
    }

    // Rayon parallel scoring across shard groups
    let shard_groups: Vec<_> = by_shard.into_iter().collect();
    let all_scored: Vec<ScoredDoc> = shard_groups
        .par_iter()
        .flat_map(|(sid, docs)| {
            let shard = match shard_snap.get(sid) {
                Some(s) => s,
                None => return Vec::new(),
            };
            if !shard.has_tensor("centroid_codes") {
                return Vec::new();
            }
            docs.iter()
                .filter_map(|&(did, rs, re)| {
                    let codes = shard
                        .read_selected_u16("centroid_codes", &[(rs, re)])
                        .ok()?;
                    let score =
                        approximate_maxsim_single(&lookup_table, &codes, n_q, n_centroids);
                    Some(ScoredDoc {
                        doc_id: did,
                        score,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();

    heap_topk(all_scored.into_iter(), n_top)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_lookup_table() {
        let dim = 2;
        let n_q = 2;
        let n_c = 3;
        let query = vec![1.0, 0.0, 0.0, 1.0];
        let centroids = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];

        let table = compute_lookup_table(&query, &centroids, n_q, n_c, dim);
        assert_eq!(table.len(), 6);
        assert!((table[0] - 1.0).abs() < 1e-5);
        assert!((table[1] - 0.0).abs() < 1e-5);
        assert!((table[2] - 0.5).abs() < 1e-5);
        assert!((table[3] - 0.0).abs() < 1e-5);
        assert!((table[4] - 1.0).abs() < 1e-5);
        assert!((table[5] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_approximate_maxsim_single() {
        let n_q = 2;
        let n_c = 3;
        let table = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5];

        let codes: Vec<u16> = vec![0, 2];
        let score = approximate_maxsim_single(&table, &codes, n_q, n_c);
        assert!((score - 1.5).abs() < 1e-5);

        let codes2: Vec<u16> = vec![1];
        let score2 = approximate_maxsim_single(&table, &codes2, n_q, n_c);
        assert!((score2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_approximate_maxsim_empty() {
        let table = vec![1.0, 0.0];
        assert_eq!(approximate_maxsim_single(&table, &[], 1, 2), 0.0);
        assert_eq!(approximate_maxsim_single(&table, &[0], 0, 2), 0.0);
    }
}
