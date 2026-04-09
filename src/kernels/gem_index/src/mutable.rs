use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop};
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes};

use crate::graph::{select_neighbors_heuristic, shrink_neighbors};
use crate::id_tracker::IdTracker;
use crate::search::{beam_search, beam_search_construction};

/// Mutable GEM segment supporting insert, delete, upsert, and compaction.
pub struct MutableGemSegment {
    pub adjacency: Vec<Vec<u32>>,
    pub max_degree: usize,
    pub ef_construction: usize,
    pub n_probes: usize,
    pub codebook: TwoStageCodebook,
    pub flat_codes: FlatDocCodes,
    pub doc_profiles: Vec<DocProfile>,
    pub postings: ClusterPostings,
    pub ctop_r: usize,
    pub id_tracker: IdTracker,
    pub dim: usize,
    pub all_vectors: Vec<f32>,
    pub doc_offsets: Vec<(usize, usize)>,
    initial_edges: usize,
}

impl MutableGemSegment {
    /// Build from a seed batch of documents.
    pub fn build(
        all_vectors: &[f32],
        dim: usize,
        doc_ids: &[u64],
        doc_offsets: &[(usize, usize)],
        n_fine: usize,
        n_coarse: usize,
        max_degree: usize,
        ef_construction: usize,
        max_kmeans_iter: usize,
        ctop_r: usize,
        n_probes: usize,
    ) -> Self {
        let n_docs = doc_ids.len();
        let n_vectors = doc_offsets.last().map_or(0, |&(_, e)| e);

        let mut codebook = TwoStageCodebook::build(
            all_vectors, n_vectors, dim, n_fine, n_coarse, max_kmeans_iter, 42,
        );
        let all_assignments = codebook.assign_vectors(all_vectors, n_vectors);

        let mut doc_centroid_sets = Vec::with_capacity(n_docs);
        for &(start, end) in doc_offsets.iter() {
            doc_centroid_sets.push(all_assignments[start..end].to_vec());
        }
        codebook.update_idf(&doc_centroid_sets);

        let mut doc_profiles = Vec::with_capacity(n_docs);
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

        let mut id_tracker = IdTracker::new();
        for &id in doc_ids {
            id_tracker.add(id);
        }

        let graph = crate::graph::build_graph(
            all_vectors,
            dim,
            doc_offsets,
            &codebook,
            &flat_codes,
            &doc_profiles,
            &postings,
            max_degree,
            ef_construction,
        );

        let initial_edges = graph.n_edges();

        Self {
            adjacency: graph.adjacency,
            max_degree,
            ef_construction,
            n_probes,
            codebook,
            flat_codes,
            doc_profiles,
            postings,
            ctop_r,
            id_tracker,
            dim,
            all_vectors: all_vectors.to_vec(),
            doc_offsets: doc_offsets.to_vec(),
            initial_edges,
        }
    }

    /// Insert a single document into the mutable graph.
    pub fn insert(&mut self, vectors: &[f32], n_tokens: usize, doc_id: u64) {
        let int_id = self.id_tracker.add(doc_id);
        let idx = int_id as usize;

        let assignments = self.codebook.assign_vectors(vectors, n_tokens);
        let cids: Vec<u32> = assignments;
        let ctop = compute_ctop(&self.codebook, &cids, self.ctop_r);

        self.postings.add_doc(int_id, &ctop);
        self.flat_codes.add_doc(&cids);
        self.doc_profiles.push(DocProfile {
            centroid_ids: cids,
            ctop,
        });

        let vec_start = self.all_vectors.len() / self.dim;
        self.all_vectors.extend_from_slice(vectors);
        self.doc_offsets.push((vec_start, vec_start + n_tokens));

        self.adjacency.push(Vec::new());

        let n_fine = self.codebook.n_fine;
        let query_scores = self.codebook.compute_query_centroid_scores(vectors, n_tokens);

        let mut entries: Vec<u32> = Vec::new();
        for &cluster in &self.doc_profiles[idx].ctop {
            if let Some(reps) = self.postings.cluster_reps.get(cluster as usize) {
                if let Some(&rep) = reps.as_ref() {
                    if (rep as usize) < idx && !entries.contains(&rep) {
                        entries.push(rep);
                    }
                }
            }
        }
        if entries.is_empty() && idx > 0 {
            entries.push(0);
        }

        if !entries.is_empty() {
            let candidates = beam_search_construction(
                &self.adjacency,
                &entries,
                &query_scores,
                n_tokens,
                &self.flat_codes,
                n_fine,
                self.ef_construction,
                idx,
            );

            let neighbors = select_neighbors_heuristic(
                &candidates,
                self.max_degree,
                &self.codebook,
                &self.flat_codes,
            );

            for &(nbr_idx, _score) in &neighbors {
                self.adjacency[idx].push(nbr_idx);
                self.adjacency[nbr_idx as usize].push(int_id);

                if self.adjacency[nbr_idx as usize].len() > self.max_degree {
                    shrink_neighbors(
                        nbr_idx as usize,
                        self.max_degree,
                        &mut self.adjacency,
                        &self.codebook,
                        &self.flat_codes,
                    );
                }
            }
        }
    }

    /// Soft-delete a document by external ID. Returns true if found and deleted.
    pub fn delete(&mut self, doc_id: u64) -> bool {
        self.id_tracker.delete(doc_id)
    }

    /// Upsert: delete old version (if exists) then insert new.
    pub fn upsert(&mut self, vectors: &[f32], n_tokens: usize, doc_id: u64) {
        self.id_tracker.delete(doc_id);
        self.insert(vectors, n_tokens, doc_id);
    }

    /// Compact: rebuild the graph without soft-deleted nodes.
    pub fn compact(&mut self) {
        let (new_tracker, mapping) = self.id_tracker.compact();
        let n_new = new_tracker.n_total();

        let mut new_adjacency: Vec<Vec<u32>> = vec![Vec::new(); n_new];
        let mut new_flat_codes = FlatDocCodes::new();
        let mut new_doc_profiles = Vec::with_capacity(n_new);
        let mut new_postings = ClusterPostings::new(self.codebook.n_coarse);
        let mut new_vectors = Vec::new();
        let mut new_offsets = Vec::new();

        for old_idx in 0..self.id_tracker.n_total() {
            if let Some(new_idx) = mapping[old_idx] {
                let new_i = new_idx as usize;

                // Remap adjacency
                let old_neighbors = &self.adjacency[old_idx];
                for &old_nbr in old_neighbors {
                    if let Some(new_nbr) = mapping[old_nbr as usize] {
                        new_adjacency[new_i].push(new_nbr);
                    }
                }

                // Copy profile and flat codes
                let profile = &self.doc_profiles[old_idx];
                new_flat_codes.add_doc(&profile.centroid_ids);
                let ctop = &profile.ctop;
                new_postings.add_doc(new_idx, ctop);
                new_doc_profiles.push(profile.clone());

                // Copy vectors
                let (vs, ve) = self.doc_offsets[old_idx];
                let vec_start = new_vectors.len() / self.dim;
                let n_tokens = ve - vs;
                new_vectors.extend_from_slice(
                    &self.all_vectors[vs * self.dim..ve * self.dim],
                );
                new_offsets.push((vec_start, vec_start + n_tokens));
            }
        }

        self.adjacency = new_adjacency;
        self.flat_codes = new_flat_codes;
        self.doc_profiles = new_doc_profiles;
        self.postings = new_postings;
        self.id_tracker = new_tracker;
        self.all_vectors = new_vectors;
        self.doc_offsets = new_offsets;
        self.initial_edges = self.n_edges();
    }

    /// Search the mutable segment.
    pub fn search(
        &self,
        query_scores: &[f32],
        n_query: usize,
        k: usize,
        ef: usize,
    ) -> Vec<(u64, f32)> {
        if self.adjacency.is_empty() {
            return Vec::new();
        }

        let mut entries: Vec<u32> = Vec::new();
        // Use first live node as entry
        for i in 0..self.adjacency.len() {
            if !self.id_tracker.is_deleted(i as u32) {
                entries.push(i as u32);
                break;
            }
        }
        if entries.is_empty() {
            return Vec::new();
        }

        let empty_shortcuts = vec![Vec::new(); self.adjacency.len()];
        let results = beam_search(
            &self.adjacency,
            &empty_shortcuts,
            &entries,
            query_scores,
            n_query,
            &self.flat_codes,
            self.codebook.n_fine,
            ef,
            Some(self.id_tracker.deleted_flags()),
            false,
        );

        results
            .into_iter()
            .take(k)
            .map(|(int_id, score)| {
                let ext_id = self.id_tracker.int_to_ext(int_id);
                (ext_id, score)
            })
            .collect()
    }

    pub fn n_nodes(&self) -> usize {
        self.adjacency.len()
    }

    pub fn n_edges(&self) -> usize {
        self.adjacency.iter().map(|a| a.len()).sum()
    }

    pub fn n_live(&self) -> usize {
        self.id_tracker.n_live()
    }

    pub fn quality_score(&self) -> f64 {
        1.0 - self.delete_ratio()
    }

    pub fn delete_ratio(&self) -> f64 {
        let total = self.id_tracker.n_total();
        if total == 0 {
            return 0.0;
        }
        self.id_tracker.n_deleted() as f64 / total as f64
    }

    pub fn avg_degree(&self) -> f64 {
        let n = self.adjacency.len();
        if n == 0 {
            return 0.0;
        }
        self.n_edges() as f64 / n as f64
    }

    pub fn memory_bytes(&self) -> usize {
        let adj_bytes: usize = self.adjacency.iter()
            .map(|a| a.len() * std::mem::size_of::<u32>() + std::mem::size_of::<Vec<u32>>())
            .sum();
        let codes_bytes = self.flat_codes.codes.len() * 2
            + self.flat_codes.offsets.len() * 4
            + self.flat_codes.lengths.len() * 2;
        adj_bytes + codes_bytes
    }
}
