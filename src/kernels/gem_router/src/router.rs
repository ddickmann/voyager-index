use std::collections::HashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use roaring::RoaringBitmap;

use crate::codebook::{TwoStageCodebook, cluster_overlap, compute_ctop, qch_proxy_score_u16};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocProfile {
    pub centroid_ids: Vec<u32>,
    pub ctop: Vec<u32>,
}

/// Flat contiguous u16 codes array for cache-sequential proxy scoring.
/// All document codes are packed into a single Vec<u16> with offset/length metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatDocCodes {
    pub codes: Vec<u16>,
    pub offsets: Vec<u32>,
    pub lengths: Vec<u16>,
}

impl Default for FlatDocCodes {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatDocCodes {
    pub fn new() -> Self {
        Self {
            codes: Vec::new(),
            offsets: Vec::new(),
            lengths: Vec::new(),
        }
    }

    pub fn add_doc(&mut self, centroid_ids: &[u32]) {
        let offset = self.codes.len() as u32;
        let len = centroid_ids.len().min(u16::MAX as usize) as u16;
        self.offsets.push(offset);
        self.lengths.push(len);
        let take_n = len as usize;
        self.codes.extend(centroid_ids[..take_n].iter().map(|&c| c as u16));
    }

    pub fn doc_codes(&self, doc_idx: usize) -> &[u16] {
        if doc_idx >= self.offsets.len() {
            return &[];
        }
        let off = self.offsets[doc_idx] as usize;
        let len = self.lengths[doc_idx] as usize;
        let end = (off + len).min(self.codes.len());
        &self.codes[off..end]
    }
}

/// Per-cluster per-field-value Roaring bitmaps for filter-aware routing.
#[derive(Debug, Clone, Default)]
pub struct FilterIndex {
    /// cluster_bitmaps[cluster_id][(field, value)] = bitmap of doc indices in that cluster matching
    pub cluster_bitmaps: Vec<HashMap<(String, String), RoaringBitmap>>,
    /// Total doc count per cluster (for ratio computation)
    pub cluster_counts: Vec<u32>,
}

impl FilterIndex {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            cluster_bitmaps: vec![HashMap::new(); n_clusters],
            cluster_counts: vec![0; n_clusters],
        }
    }

    /// Build filter summaries from doc payloads.
    /// `doc_payloads`: (doc_internal_id, vec of (field, value) pairs)
    /// `postings`: used to determine which clusters each doc belongs to
    pub fn build_from_payloads(
        &mut self,
        doc_payloads: &[(u32, Vec<(String, String)>)],
        postings: &ClusterPostings,
    ) {
        let n_clusters = postings.lists.len();
        self.cluster_bitmaps = vec![HashMap::new(); n_clusters];
        self.cluster_counts = vec![0; n_clusters];

        let mut doc_to_clusters: HashMap<u32, Vec<usize>> = HashMap::new();
        for (c, members) in postings.lists.iter().enumerate() {
            self.cluster_counts[c] = members.len() as u32;
            for &doc_id in members {
                doc_to_clusters.entry(doc_id).or_default().push(c);
            }
        }

        for &(doc_id, ref field_values) in doc_payloads {
            if let Some(clusters) = doc_to_clusters.get(&doc_id) {
                for &c in clusters {
                    for (field, value) in field_values {
                        self.cluster_bitmaps[c]
                            .entry((field.clone(), value.clone()))
                            .or_default()
                            .insert(doc_id);
                    }
                }
            }
        }
    }

    /// Returns cluster IDs where the filter covers >= min_ratio of the cluster's docs.
    /// filter: AND-semantics — all (field, value) pairs must match.
    pub fn clusters_passing_filter(
        &self,
        filter: &[(String, String)],
        min_ratio: f32,
    ) -> Vec<u32> {
        let n = self.cluster_bitmaps.len();
        let mut passing = Vec::new();
        for c in 0..n {
            let total = self.cluster_counts[c];
            if total == 0 {
                continue;
            }
            let cluster_map = &self.cluster_bitmaps[c];

            let mut intersection: Option<RoaringBitmap> = None;
            let mut all_present = true;
            for (field, value) in filter {
                match cluster_map.get(&(field.clone(), value.clone())) {
                    Some(bm) => {
                        intersection = Some(match intersection {
                            None => bm.clone(),
                            Some(acc) => &acc & bm,
                        });
                    }
                    None => {
                        all_present = false;
                        break;
                    }
                }
            }

            if !all_present {
                continue;
            }

            let matched = intersection.map_or(total as u64, |bm| bm.len());
            let ratio = matched as f32 / total as f32;
            if ratio >= min_ratio {
                passing.push(c as u32);
            }
        }
        passing
    }

    /// Build a doc-level boolean mask for docs matching the filter.
    /// Returns Vec<bool> of length n_docs where true = matches filter.
    pub fn build_filter_mask(
        &self,
        filter: &[(String, String)],
        n_docs: usize,
    ) -> Vec<bool> {
        let mut mask = vec![false; n_docs];
        if filter.is_empty() {
            mask.iter_mut().for_each(|m| *m = true);
            return mask;
        }

        let mut global_intersection: Option<RoaringBitmap> = None;
        for (field, value) in filter {
            let mut union = RoaringBitmap::new();
            for cluster_map in &self.cluster_bitmaps {
                if let Some(bm) = cluster_map.get(&(field.clone(), value.clone())) {
                    union |= bm;
                }
            }
            global_intersection = Some(match global_intersection {
                None => union,
                Some(acc) => &acc & &union,
            });
        }

        if let Some(bm) = global_intersection {
            for doc_id in bm {
                let idx = doc_id as usize;
                if idx < n_docs {
                    mask[idx] = true;
                }
            }
        }
        mask
    }
}

/// Per-cluster posting list: maps coarse cluster ID -> list of doc indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPostings {
    pub lists: Vec<Vec<u32>>,
    pub cluster_reps: Vec<Option<u32>>,
}

impl ClusterPostings {
    pub fn new(n_coarse: usize) -> Self {
        Self {
            lists: vec![Vec::new(); n_coarse],
            cluster_reps: vec![None; n_coarse],
        }
    }

    pub fn add_doc(&mut self, doc_idx: u32, ctop: &[u32]) {
        for &cluster in ctop {
            let c = cluster as usize;
            if c < self.lists.len() {
                self.lists[c].push(doc_idx);
                if self.cluster_reps[c].is_none() {
                    self.cluster_reps[c] = Some(doc_idx);
                }
            }
        }
    }

    pub fn docs_in_clusters(&self, clusters: &[u32]) -> Vec<u32> {
        let mut seen = HashMap::new();
        for &c in clusters {
            let c = c as usize;
            if c < self.lists.len() {
                for &doc_idx in &self.lists[c] {
                    seen.entry(doc_idx).or_insert(0u32);
                    *seen.get_mut(&doc_idx).unwrap() += 1;
                }
            }
        }
        let mut docs: Vec<(u32, u32)> = seen.into_iter().collect();
        docs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        docs.into_iter().map(|(d, _)| d).collect()
    }

    /// Recompute cluster representatives as geometric medoids.
    ///
    /// For each cluster, the medoid is the member document whose total qCH
    /// proxy distance to all other cluster members is minimized. This places
    /// the representative at a central position, improving beam search entry.
    pub fn compute_medoids(
        &mut self,
        codebook: &TwoStageCodebook,
        flat_codes: &FlatDocCodes,
    ) {
        for (c, members) in self.lists.iter().enumerate() {
            if members.len() <= 1 {
                continue;
            }
            let effective = if members.len() > 64 {
                &members[..64]
            } else {
                members.as_slice()
            };

            let mut best_idx = effective[0];
            let mut best_total = f32::MAX;

            for &candidate in effective {
                let c_codes = flat_codes.doc_codes(candidate as usize);
                if c_codes.is_empty() {
                    continue;
                }
                let mut total = 0.0f32;
                let mut n_compared = 0u32;
                for &other in effective {
                    if other == candidate {
                        continue;
                    }
                    let o_codes = flat_codes.doc_codes(other as usize);
                    if o_codes.is_empty() {
                        continue;
                    }
                    let mut d = 0.0f32;
                    for &cc in c_codes {
                        let mut min_d = f32::MAX;
                        for &oc in o_codes {
                            let cd = codebook.centroid_dist(cc as u32, oc as u32);
                            if cd < min_d {
                                min_d = cd;
                            }
                        }
                        if min_d < f32::MAX {
                            d += min_d;
                        }
                    }
                    d /= c_codes.len() as f32;
                    total += d;
                    n_compared += 1;
                }
                if n_compared > 0 {
                    total /= n_compared as f32;
                }
                if total < best_total {
                    best_total = total;
                    best_idx = candidate;
                }
            }
            self.cluster_reps[c] = Some(best_idx);
        }
    }

    pub fn representatives_for_clusters(&self, clusters: &[u32]) -> Vec<u32> {
        clusters
            .iter()
            .filter_map(|&c| self.cluster_reps.get(c as usize).and_then(|r| *r))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemRouterState {
    pub codebook: TwoStageCodebook,
    pub doc_profiles: Vec<DocProfile>,
    pub doc_ids: Vec<u64>,
    pub postings: ClusterPostings,
    pub ctop_r: usize,
    pub flat_codes: FlatDocCodes,
}

pub struct GemRouter {
    state: Option<GemRouterState>,
}

impl Default for GemRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl GemRouter {
    pub fn new() -> Self {
        Self {
            state: None,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.state.is_some()
    }

    pub fn state(&self) -> Option<&GemRouterState> {
        self.state.as_ref()
    }

    pub fn build(
        &mut self,
        all_vectors: &[f32],
        n_vectors: usize,
        dim: usize,
        doc_ids: &[u64],
        doc_vector_ranges: &[(usize, usize)],
        n_fine: usize,
        n_coarse: usize,
        max_kmeans_iter: usize,
        ctop_r: usize,
    ) {
        let mut codebook =
            TwoStageCodebook::build(all_vectors, n_vectors, dim, n_fine, n_coarse, max_kmeans_iter, 42);

        let all_assignments = codebook.assign_vectors(all_vectors, n_vectors);

        let mut doc_centroid_sets = Vec::with_capacity(doc_ids.len());
        for &(start, end) in doc_vector_ranges.iter() {
            doc_centroid_sets.push(all_assignments[start..end].to_vec());
        }

        codebook.update_idf(&doc_centroid_sets);

        let mut doc_profiles = Vec::with_capacity(doc_ids.len());
        let mut flat_codes = FlatDocCodes::new();
        let mut postings = ClusterPostings::new(n_coarse);
        for (doc_idx, cids) in doc_centroid_sets.into_iter().enumerate() {
            let ctop = compute_ctop(&codebook, &cids, ctop_r);
            postings.add_doc(doc_idx as u32, &ctop);
            flat_codes.add_doc(&cids);
            doc_profiles.push(DocProfile {
                centroid_ids: cids,
                ctop,
            });
        }

        postings.compute_medoids(&codebook, &flat_codes);

        self.state = Some(GemRouterState {
            codebook,
            doc_profiles,
            doc_ids: doc_ids.to_vec(),
            postings,
            ctop_r,
            flat_codes,
        });
    }

    pub fn add_documents(
        &mut self,
        new_vectors: &[f32],
        n_new_vectors: usize,
        _dim: usize,
        new_doc_ids: &[u64],
        doc_vector_ranges: &[(usize, usize)],
    ) {
        let state = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        let assignments = state.codebook.assign_vectors(new_vectors, n_new_vectors);
        let base_doc_idx = state.doc_profiles.len() as u32;

        for (local_idx, &(start, end)) in doc_vector_ranges.iter().enumerate() {
            let cids: Vec<u32> = assignments[start..end].to_vec();
            let ctop = compute_ctop(&state.codebook, &cids, state.ctop_r);
            let doc_idx = base_doc_idx + local_idx as u32;
            state.postings.add_doc(doc_idx, &ctop);
            state.flat_codes.add_doc(&cids);
            state.doc_profiles.push(DocProfile {
                centroid_ids: cids,
                ctop,
            });
        }

        state.doc_ids.extend_from_slice(new_doc_ids);
    }

    /// Route a query to candidate documents using cluster-based selection
    /// and qCH proxy scoring via flat u16 codes and reusable score buffer.
    pub fn route_query(
        &self,
        query_vectors: &[f32],
        n_query_vecs: usize,
        _dim: usize,
        n_probes: usize,
        max_candidates: usize,
    ) -> Vec<(u64, f32)> {
        let state = match &self.state {
            Some(s) => s,
            None => return Vec::new(),
        };

        let query_cids = state.codebook.assign_vectors(query_vectors, n_query_vecs);
        let query_ctop = compute_ctop(&state.codebook, &query_cids, n_probes.max(state.ctop_r));
        let candidate_doc_idxs = state.postings.docs_in_clusters(&query_ctop);

        let n_fine = state.codebook.n_fine;
        let needed = n_query_vecs * n_fine;

        let mut buf = vec![0.0f32; needed];
        state.codebook.compute_query_centroid_scores_into(query_vectors, n_query_vecs, &mut buf);
        state.codebook.apply_idf_weights(&mut buf, n_query_vecs);
        let query_centroid_scores = &buf[..needed];

        let flat_codes = &state.flat_codes;
        let query_ctop_ref = &query_ctop;

        let mut scored: Vec<(u32, f32)> = candidate_doc_idxs
            .par_iter()
            .map(|&doc_idx| {
                let doc_codes = flat_codes.doc_codes(doc_idx as usize);

                let proxy = qch_proxy_score_u16(
                    query_centroid_scores,
                    n_query_vecs,
                    n_fine,
                    doc_codes,
                );

                let profile = &state.doc_profiles[doc_idx as usize];
                let overlap = cluster_overlap(query_ctop_ref, &profile.ctop) as f32;
                let overlap_bonus = overlap / query_ctop_ref.len().max(1) as f32;

                let combined = (1.0 / (1.0 + proxy)) + 0.2 * overlap_bonus;

                (doc_idx, combined)
            })
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(max_candidates);

        scored
            .into_iter()
            .map(|(doc_idx, score)| {
                let doc_id = state.doc_ids[doc_idx as usize];
                (doc_id, score)
            })
            .collect()
    }

    pub fn get_cluster_entries(
        &self,
        query_vectors: &[f32],
        n_query_vecs: usize,
        _dim: usize,
        n_probes: usize,
    ) -> Vec<u64> {
        let state = match &self.state {
            Some(s) => s,
            None => return Vec::new(),
        };

        let query_cids = state.codebook.assign_vectors(query_vectors, n_query_vecs);
        let query_ctop = compute_ctop(&state.codebook, &query_cids, n_probes.max(state.ctop_r));
        let rep_idxs = state.postings.representatives_for_clusters(&query_ctop);

        rep_idxs
            .into_iter()
            .filter_map(|idx| state.doc_ids.get(idx as usize).copied())
            .collect()
    }

    pub fn compute_query_profile(
        &self,
        query_vectors: &[f32],
        n_query_vecs: usize,
        n_probes: usize,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        let state = self.state.as_ref()?;
        let query_cids = state.codebook.assign_vectors(query_vectors, n_query_vecs);
        let query_ctop = compute_ctop(&state.codebook, &query_cids, n_probes.max(state.ctop_r));
        Some((query_cids, query_ctop))
    }

    pub fn score_doc(
        &self,
        query_centroid_scores: &[f32],
        n_query_vecs: usize,
        doc_idx: usize,
    ) -> f32 {
        let state = match &self.state {
            Some(s) => s,
            None => return 0.0,
        };
        if doc_idx >= state.doc_profiles.len() {
            return 0.0;
        }
        let doc_codes = state.flat_codes.doc_codes(doc_idx);
        qch_proxy_score_u16(
            query_centroid_scores,
            n_query_vecs,
            state.codebook.n_fine,
            doc_codes,
        )
    }

    pub fn n_docs(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.doc_profiles.len())
    }

    pub fn n_fine(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.codebook.n_fine)
    }

    pub fn n_coarse(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.codebook.n_coarse)
    }

    pub fn restore_state(&mut self, state: GemRouterState) {
        self.state = Some(state);
    }

    /// Train an adaptive cutoff decision tree from query-positive training pairs.
    ///
    /// For each (query, positive_doc) pair, computes the optimal cluster cutoff
    /// and trains a decision tree to predict it from document features.
    ///
    /// - `training_queries`: flat query vectors
    /// - `n_query_vecs`: number of vectors per query
    /// - `training_positives`: positive document indices (internal)
    /// - `t`: top centroids per query token for label computation
    /// - `r_max`: maximum clusters to consider
    /// - `max_depth`: tree depth limit
    pub fn train_adaptive_cutoff(
        &self,
        training_queries: &[f32],
        n_query_vecs: &[usize],
        training_positives: &[usize],
        t: usize,
        r_max: usize,
        max_depth: usize,
    ) -> Option<crate::adaptive_cutoff::CutoffTree> {
        let state = self.state.as_ref()?;
        let n_pairs = training_positives.len();
        if n_pairs == 0 || n_query_vecs.len() != n_pairs {
            return None;
        }
        let expected_q_len: usize = n_query_vecs.iter().sum::<usize>() * state.codebook.dim;
        if training_queries.len() < expected_q_len {
            return None;
        }

        let mut query_top_clusters: Vec<Vec<u32>> = Vec::with_capacity(n_pairs);
        let mut offset = 0usize;
        for &nq in n_query_vecs {
            let qvecs = &training_queries[offset..offset + nq * state.codebook.dim];
            let q_cids = state.codebook.assign_vectors(qvecs, nq);
            let q_ctop = compute_ctop(&state.codebook, &q_cids, t);
            query_top_clusters.push(q_ctop);
            offset += nq * state.codebook.dim;
        }

        // Build doc sorted clusters for each positive
        let doc_sorted_clusters: Vec<Vec<u32>> = training_positives
            .iter()
            .map(|&doc_idx| {
                if doc_idx < state.doc_profiles.len() {
                    let cids = &state.doc_profiles[doc_idx].centroid_ids;
                    let mut coarse_scores = vec![0.0f32; state.codebook.n_coarse];
                    for &cid in cids {
                        let idx = cid as usize;
                        if idx < state.codebook.n_fine {
                            let coarse = state.codebook.cindex_labels[idx] as usize;
                            if coarse < state.codebook.n_coarse {
                                coarse_scores[coarse] += state.codebook.idf[idx];
                            }
                        }
                    }
                    let mut indexed: Vec<(usize, f32)> = coarse_scores
                        .iter()
                        .enumerate()
                        .filter(|(_, &s)| s > 0.0)
                        .map(|(i, &s)| (i, s))
                        .collect();
                    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
                    indexed.iter().map(|&(i, _)| i as u32).collect()
                } else {
                    Vec::new()
                }
            })
            .collect();

        // Compute labels
        let labels = crate::adaptive_cutoff::compute_training_labels(
            &query_top_clusters,
            &doc_sorted_clusters,
            r_max,
        );

        // Build features for each positive doc
        let cluster_scores: Vec<Vec<f32>> = training_positives
            .iter()
            .map(|&doc_idx| {
                if doc_idx < state.doc_profiles.len() {
                    let cids = &state.doc_profiles[doc_idx].centroid_ids;
                    let mut coarse_scores = vec![0.0f32; state.codebook.n_coarse];
                    for &cid in cids {
                        let idx = cid as usize;
                        if idx < state.codebook.n_fine {
                            let coarse = state.codebook.cindex_labels[idx] as usize;
                            if coarse < state.codebook.n_coarse {
                                coarse_scores[coarse] += state.codebook.idf[idx];
                            }
                        }
                    }
                    coarse_scores
                } else {
                    vec![0.0; state.codebook.n_coarse]
                }
            })
            .collect();

        let doc_lengths: Vec<usize> = training_positives
            .iter()
            .map(|&doc_idx| {
                if doc_idx < state.doc_profiles.len() {
                    state.doc_profiles[doc_idx].centroid_ids.len()
                } else {
                    0
                }
            })
            .collect();

        let (doc_features, len_normalizer) = crate::adaptive_cutoff::build_doc_features(
            &cluster_scores,
            &doc_lengths,
            r_max,
        );

        Some(crate::adaptive_cutoff::CutoffTree::train(
            &doc_features,
            &labels,
            max_depth,
            r_max,
            len_normalizer,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn make_test_data(n_docs: usize, vecs_per_doc: usize, dim: usize) -> (Vec<f32>, Vec<u64>, Vec<(usize, usize)>) {
        let mut rng = StdRng::seed_from_u64(42);
        let total = n_docs * vecs_per_doc;
        let data: Vec<f32> = (0..total * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let doc_ids: Vec<u64> = (0..n_docs as u64).collect();
        let ranges: Vec<(usize, usize)> = (0..n_docs)
            .map(|i| (i * vecs_per_doc, (i + 1) * vecs_per_doc))
            .collect();
        (data, doc_ids, ranges)
    }

    #[test]
    fn test_router_build_and_route() {
        let dim = 16;
        let (data, doc_ids, ranges) = make_test_data(50, 10, dim);
        let n_vectors = 50 * 10;

        let mut router = GemRouter::new();
        router.build(&data, n_vectors, dim, &doc_ids, &ranges, 16, 4, 10, 3);
        assert!(router.is_ready());
        assert_eq!(router.n_docs(), 50);

        let state = router.state().unwrap();
        assert_eq!(state.flat_codes.offsets.len(), 50);
        assert_eq!(state.flat_codes.lengths.len(), 50);
        assert!(!state.flat_codes.codes.is_empty());

        let mut rng = StdRng::seed_from_u64(99);
        let query: Vec<f32> = (0..5 * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let results = router.route_query(&query, 5, dim, 4, 20);
        assert!(!results.is_empty());
        assert!(results.len() <= 20);
        for &(_, score) in &results {
            assert!(score > 0.0);
        }
    }

    #[test]
    fn test_flat_codes_consistency() {
        let dim = 8;
        let (data, doc_ids, ranges) = make_test_data(30, 5, dim);
        let n_vectors = 30 * 5;

        let mut router = GemRouter::new();
        router.build(&data, n_vectors, dim, &doc_ids, &ranges, 8, 4, 10, 3);

        let state = router.state().unwrap();
        for (i, profile) in state.doc_profiles.iter().enumerate() {
            let flat = state.flat_codes.doc_codes(i);
            assert_eq!(flat.len(), profile.centroid_ids.len());
            for (j, &cid) in profile.centroid_ids.iter().enumerate() {
                assert_eq!(flat[j], cid as u16);
            }
        }
    }

    #[test]
    fn test_router_cluster_entries() {
        let dim = 8;
        let (data, doc_ids, ranges) = make_test_data(30, 5, dim);
        let n_vectors = 30 * 5;

        let mut router = GemRouter::new();
        router.build(&data, n_vectors, dim, &doc_ids, &ranges, 8, 4, 10, 3);

        let mut rng = StdRng::seed_from_u64(77);
        let query: Vec<f32> = (0..3 * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let entries = router.get_cluster_entries(&query, 3, dim, 4);
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_router_add_documents() {
        let dim = 8;
        let (data, doc_ids, ranges) = make_test_data(20, 5, dim);
        let n_vectors = 20 * 5;

        let mut router = GemRouter::new();
        router.build(&data, n_vectors, dim, &doc_ids, &ranges, 8, 4, 10, 3);
        assert_eq!(router.n_docs(), 20);
        assert_eq!(router.state().unwrap().flat_codes.offsets.len(), 20);

        let (new_data, _new_ids, new_ranges) = make_test_data(10, 5, dim);
        let new_ids: Vec<u64> = (20..30).collect();
        router.add_documents(&new_data, 10 * 5, dim, &new_ids, &new_ranges);
        assert_eq!(router.n_docs(), 30);
        assert_eq!(router.state().unwrap().flat_codes.offsets.len(), 30);
    }
}
