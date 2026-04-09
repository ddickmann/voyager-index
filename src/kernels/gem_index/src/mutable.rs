use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop, qch_proxy_score_u16};
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes};

use crate::graph::{select_neighbors_heuristic, shrink_neighbors, VecAdjacency};
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
        debug_assert_eq!(n_docs, doc_offsets.len(), "doc_ids and doc_offsets must have the same length");
        let n_vectors = doc_offsets.last().map_or(0, |&(_, e)| e);
        debug_assert!(
            all_vectors.len() >= n_vectors * dim,
            "all_vectors too short: {} < {} * {}",
            all_vectors.len(), n_vectors, dim,
        );

        let mut codebook = TwoStageCodebook::build(
            all_vectors, n_vectors, dim, n_fine, n_coarse, max_kmeans_iter, 42,
        );
        let all_assignments = codebook.assign_vectors(all_vectors, n_vectors);

        let mut doc_centroid_sets = Vec::with_capacity(n_docs);
        for &(start, end) in doc_offsets.iter() {
            doc_centroid_sets.push(all_assignments[start..end].to_vec());
        }
        codebook.update_idf(&doc_centroid_sets);

        // IDF-weighted codebook refinement for better tail-token discrimination
        codebook.refine_centroids_idf(all_vectors, n_vectors, 3);
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
        let adjacency = graph.levels[0].to_adj_lists();

        Self {
            adjacency,
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
        debug_assert_eq!(
            vectors.len(), n_tokens * self.dim,
            "insert: vectors.len() ({}) != n_tokens ({}) * dim ({})",
            vectors.len(), n_tokens, self.dim,
        );
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
                    if (rep as usize) < idx
                        && !self.id_tracker.is_deleted(rep)
                        && !entries.contains(&rep)
                    {
                        entries.push(rep);
                    }
                }
            }
        }
        if entries.is_empty() && idx > 0 {
            for i in (0..idx).rev() {
                if !self.id_tracker.is_deleted(i as u32) {
                    entries.push(i as u32);
                    break;
                }
            }
        }

        if !entries.is_empty() {
            let adj = VecAdjacency(&self.adjacency);
            let candidates = beam_search_construction(
                &adj,
                &entries,
                &query_scores,
                n_tokens,
                &self.flat_codes,
                n_fine,
                self.ef_construction,
                idx,
            );
            drop(adj);

            let doc_codes = self.flat_codes.doc_codes(idx);
            let neighbors = select_neighbors_heuristic(
                &candidates,
                doc_codes,
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

        self.incremental_bridge_repair(idx);
    }

    /// Lightweight bridge repair for a single newly inserted node.
    /// Ensures the node connects to cluster representatives, maintaining
    /// cross-cluster connectivity. Evicts weakest edge when at capacity.
    fn incremental_bridge_repair(&mut self, node_idx: usize) {
        let ctop: Vec<u32> = self.doc_profiles[node_idx].ctop.clone();

        for &cluster in &ctop {
            let cluster_id = cluster as usize;
            match self.postings.lists.get(cluster_id) {
                Some(m) if m.len() > 1 => {},
                _ => continue,
            };

            let rep = match self.postings.cluster_reps.get(cluster_id) {
                Some(Some(r)) => *r,
                _ => continue,
            };

            if self.id_tracker.is_deleted(rep) || rep as usize == node_idx {
                continue;
            }

            let rep_usize = rep as usize;
            let node_u32 = node_idx as u32;

            let connected_to_rep = self.adjacency[node_idx].contains(&rep)
                || self.adjacency[rep_usize].contains(&node_u32);

            if connected_to_rep {
                continue;
            }

            if self.adjacency[node_idx].len() >= self.max_degree {
                shrink_neighbors(
                    node_idx,
                    self.max_degree - 1,
                    &mut self.adjacency,
                    &self.codebook,
                    &self.flat_codes,
                );
            }
            self.adjacency[node_idx].push(rep);
            self.adjacency[rep_usize].push(node_u32);
            if self.adjacency[rep_usize].len() > self.max_degree {
                shrink_neighbors(
                    rep_usize,
                    self.max_degree,
                    &mut self.adjacency,
                    &self.codebook,
                    &self.flat_codes,
                );
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

    /// Search the mutable segment with cluster-guided entry points
    /// and adaptive ef expansion.
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

        // Cluster-guided entry points (same as sealed search)
        let n_fine = self.codebook.n_fine;
        let mut entries: Vec<u32> = Vec::new();

        // Find query's top clusters and use their representatives
        for cid in 0..self.postings.lists.len() {
            if let Some(Some(&rep)) = self.postings.cluster_reps.get(cid).map(|r| r.as_ref()) {
                if !self.id_tracker.is_deleted(rep) {
                    entries.push(rep);
                }
            }
        }

        // Rank entry points by score and keep top n_probes
        if entries.len() > self.n_probes {
            let mut scored: Vec<(u32, f32)> = entries
                .iter()
                .map(|&ep| {
                    let codes = self.flat_codes.doc_codes(ep as usize);
                    let s = qch_proxy_score_u16(query_scores, n_query, n_fine, codes);
                    (ep, s)
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            entries = scored.into_iter().take(self.n_probes).map(|(e, _)| e).collect();
        }

        // Fallback to first live node
        if entries.is_empty() {
            for i in 0..self.adjacency.len() {
                if !self.id_tracker.is_deleted(i as u32) {
                    entries.push(i as u32);
                    break;
                }
            }
        }
        if entries.is_empty() {
            return Vec::new();
        }

        let adj = VecAdjacency(&self.adjacency);

        // Dynamic ef: start with requested ef, check if results are tight
        let mut actual_ef = ef;
        let results = beam_search(
            &adj,
            None,
            &entries,
            query_scores,
            n_query,
            &self.flat_codes,
            n_fine,
            actual_ef,
            Some(self.id_tracker.deleted_flags()),
            false,
        );

        // If margin between k-th and (k+1)-th result is tight, re-search with 2x ef
        let needs_expansion = results.len() > k && k > 0 && {
            let k_score = results[k - 1].1;
            let next_score = results[k].1;
            let margin = (next_score - k_score).abs();
            margin < k_score.abs() * 0.01
        };

        let final_results = if needs_expansion {
            actual_ef = (ef * 2).min(self.adjacency.len());
            beam_search(
                &adj,
                None,
                &entries,
                query_scores,
                n_query,
                &self.flat_codes,
                n_fine,
                actual_ef,
                Some(self.id_tracker.deleted_flags()),
                false,
            )
        } else {
            results
        };

        final_results
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
        let vectors_bytes = self.all_vectors.len() * std::mem::size_of::<f32>();
        let offsets_bytes = self.doc_offsets.len() * std::mem::size_of::<(usize, usize)>();
        let profiles_bytes = self.doc_profiles.len() * std::mem::size_of::<DocProfile>();
        let tracker_bytes = self.id_tracker.n_total()
            * (std::mem::size_of::<u64>() + std::mem::size_of::<bool>() + 16);
        adj_bytes + codes_bytes + vectors_bytes + offsets_bytes + profiles_bytes + tracker_bytes
    }
}
