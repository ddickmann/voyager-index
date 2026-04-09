use std::collections::HashMap;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
    /// Pre-allocated buffer for query-centroid scores, reused across queries
    score_buffer: Mutex<Vec<f32>>,
}

impl GemRouter {
    pub fn new() -> Self {
        Self {
            state: None,
            score_buffer: Mutex::new(Vec::new()),
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

        // Pre-allocate score buffer for expected query size
        let buf_size = 64 * codebook.n_fine;
        *self.score_buffer.lock() = vec![0.0f32; buf_size];

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

        // Reuse pre-allocated score buffer
        let mut buf = self.score_buffer.lock();
        if buf.len() < needed {
            buf.resize(needed, 0.0);
        }
        state.codebook.compute_query_centroid_scores_into(query_vectors, n_query_vecs, &mut buf);
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

        drop(buf);

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
        let buf_size = 64 * state.codebook.n_fine;
        *self.score_buffer.lock() = vec![0.0f32; buf_size];
        self.state = Some(state);
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
        assert!(state.flat_codes.codes.len() > 0);

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
