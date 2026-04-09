use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop, qch_proxy_score_u16};
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes};

use crate::graph::{select_neighbors_heuristic_cached, shrink_neighbors_emd, VecAdjacency};
use crate::id_tracker::IdTracker;
use crate::search::{beam_search, beam_search_construction};

/// Mutable GEM segment supporting insert, delete, upsert, and compaction.
///
/// Uses a single flat adjacency layer (not multi-level HNSW). This is a deliberate
/// trade-off: mutable segments sacrifice navigability for O(1) insert. Sealed segments
/// use multi-level HNSW and provide better recall/latency. Keep mutable segments small
/// and seal promptly.
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
    pub use_emd: bool,
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

        postings.compute_medoids(&codebook, &flat_codes);

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
            use_emd: false,
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
            let candidates = {
                let adj = VecAdjacency(&self.adjacency);
                beam_search_construction(
                    &adj,
                    &entries,
                    &query_scores,
                    n_tokens,
                    &self.flat_codes,
                    n_fine,
                    self.ef_construction,
                    idx,
                )
            };

            let doc_codes = self.flat_codes.doc_codes(idx);
            let neighbors = select_neighbors_heuristic_cached(
                &candidates,
                doc_codes,
                self.max_degree,
                &self.codebook,
                &self.flat_codes,
                None,
                self.use_emd,
            );

            for &(nbr_idx, _score) in &neighbors {
                self.adjacency[idx].push(nbr_idx);
                self.adjacency[nbr_idx as usize].push(int_id);

                if self.adjacency[nbr_idx as usize].len() > self.max_degree {
                    shrink_neighbors_emd(
                        nbr_idx as usize,
                        self.max_degree,
                        &mut self.adjacency,
                        &self.codebook,
                        &self.flat_codes,
                        self.use_emd,
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
                shrink_neighbors_emd(
                    node_idx,
                    self.max_degree - 1,
                    &mut self.adjacency,
                    &self.codebook,
                    &self.flat_codes,
                    self.use_emd,
                );
            }
            self.adjacency[node_idx].push(rep);
            self.adjacency[rep_usize].push(node_u32);
            if self.adjacency[rep_usize].len() > self.max_degree {
                shrink_neighbors_emd(
                    rep_usize,
                    self.max_degree,
                    &mut self.adjacency,
                    &self.codebook,
                    &self.flat_codes,
                    self.use_emd,
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

        self.postings.compute_medoids(&self.codebook, &self.flat_codes);
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
            .filter_map(|(int_id, score)| {
                let ext_id = self.id_tracker.int_to_ext(int_id);
                if ext_id == u64::MAX { None } else { Some((ext_id, score)) }
            })
            .take(k)
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

    /// Graph quality metrics for drift detection.
    ///
    /// Returns (delete_ratio, avg_degree, isolated_node_ratio, stale_rep_ratio).
    pub fn graph_quality_metrics(&self) -> (f64, f64, f64, f64) {
        let total = self.id_tracker.n_total();
        if total == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let del_ratio = self.delete_ratio();
        let avg_deg = self.avg_degree();

        let mut isolated = 0usize;
        let mut live_count = 0usize;
        for i in 0..total {
            if !self.id_tracker.is_deleted(i as u32) {
                live_count += 1;
                if self.adjacency[i].is_empty() {
                    isolated += 1;
                }
            }
        }
        let isolated_ratio = if live_count > 0 {
            isolated as f64 / live_count as f64
        } else {
            0.0
        };

        let n_clusters = self.postings.lists.len();
        let mut stale_reps = 0usize;
        let mut active_clusters = 0usize;
        for c in 0..n_clusters {
            if self.postings.lists[c].is_empty() {
                continue;
            }
            active_clusters += 1;
            match self.postings.cluster_reps[c] {
                None => { stale_reps += 1; }
                Some(rep) => {
                    let rep_idx = rep as usize;
                    let is_stale = self.id_tracker.is_deleted(rep)
                        || rep_idx >= self.adjacency.len()
                        || self.adjacency[rep_idx].len() < 2;
                    if is_stale {
                        stale_reps += 1;
                    }
                }
            }
        }
        let stale_rep_ratio = if active_clusters > 0 {
            stale_reps as f64 / active_clusters as f64
        } else {
            0.0
        };

        (del_ratio, avg_deg, isolated_ratio, stale_rep_ratio)
    }

    /// Deep graph connectivity report via BFS.
    ///
    /// Returns (n_components, giant_component_frac, cross_cluster_edge_ratio):
    ///   - n_components: number of connected components among live nodes
    ///   - giant_component_frac: fraction of live nodes in the largest component
    ///   - cross_cluster_edge_ratio: fraction of edges that connect nodes in
    ///     different coarse clusters (higher = better navigability)
    pub fn graph_connectivity_report(&self) -> (usize, f64, f64) {
        let total = self.id_tracker.n_total();
        if total == 0 {
            return (0, 0.0, 0.0);
        }

        let mut visited = vec![false; total];
        let mut component_sizes: Vec<usize> = Vec::new();
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        for start in 0..total {
            if visited[start] || self.id_tracker.is_deleted(start as u32) {
                continue;
            }
            let mut comp_size = 0usize;
            queue.push_back(start);
            visited[start] = true;
            while let Some(node) = queue.pop_front() {
                comp_size += 1;
                for &nbr in &self.adjacency[node] {
                    let n = nbr as usize;
                    if n < total && !visited[n] && !self.id_tracker.is_deleted(nbr) {
                        visited[n] = true;
                        queue.push_back(n);
                    }
                }
            }
            component_sizes.push(comp_size);
        }

        let n_components = component_sizes.len();
        let live_count: usize = component_sizes.iter().sum();
        let giant = component_sizes.iter().copied().max().unwrap_or(0);
        let giant_frac = if live_count > 0 {
            giant as f64 / live_count as f64
        } else {
            0.0
        };

        // Cross-cluster edge ratio: fraction of edges connecting different clusters.
        // Build node -> primary cluster mapping from postings.
        let mut node_cluster = vec![u32::MAX; total];
        for (c, members) in self.postings.lists.iter().enumerate() {
            for &doc in members {
                let d = doc as usize;
                if d < total {
                    node_cluster[d] = c as u32;
                }
            }
        }
        let mut total_edges = 0u64;
        let mut cross_edges = 0u64;
        for node in 0..total {
            if self.id_tracker.is_deleted(node as u32) {
                continue;
            }
            for &nbr in &self.adjacency[node] {
                let n = nbr as usize;
                if n < total && !self.id_tracker.is_deleted(nbr) {
                    total_edges += 1;
                    if node_cluster[node] != node_cluster[n] {
                        cross_edges += 1;
                    }
                }
            }
        }
        let cross_ratio = if total_edges > 0 {
            cross_edges as f64 / total_edges as f64
        } else {
            0.0
        };

        (n_components, giant_frac, cross_ratio)
    }

    /// Returns true if the graph needs healing based on drift thresholds.
    pub fn needs_healing(&self) -> bool {
        let (del_ratio, _avg_deg, isolated_ratio, stale_rep_ratio) = self.graph_quality_metrics();
        isolated_ratio > 0.05 || stale_rep_ratio > 0.2 || del_ratio > 0.15
    }

    /// Periodic local repair: fix stale reps, reconnect isolated nodes,
    /// and refresh entry points.
    pub fn heal(&mut self) {
        let total = self.id_tracker.n_total();
        if total == 0 {
            return;
        }

        // Phase 1: Recompute medoids for clusters with deleted/stale reps
        for c in 0..self.postings.lists.len() {
            let needs_recompute = match self.postings.cluster_reps[c] {
                None => !self.postings.lists[c].is_empty(),
                Some(rep) => {
                    let rep_idx = rep as usize;
                    self.id_tracker.is_deleted(rep)
                        || rep_idx >= self.adjacency.len()
                        || self.adjacency[rep_idx].len() < 2
                }
            };
            if needs_recompute {
                let members: Vec<u32> = self.postings.lists[c]
                    .iter()
                    .copied()
                    .filter(|&d| !self.id_tracker.is_deleted(d))
                    .collect();
                if members.is_empty() {
                    self.postings.cluster_reps[c] = None;
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
                    let c_codes = self.flat_codes.doc_codes(candidate as usize);
                    if c_codes.is_empty() { continue; }
                    let mut total_d = 0.0f32;
                    let mut n_compared = 0u32;
                    for &other in effective {
                        if other == candidate { continue; }
                        let o_codes = self.flat_codes.doc_codes(other as usize);
                        if o_codes.is_empty() { continue; }
                        let mut d = 0.0f32;
                        for &cc in c_codes {
                            let mut min_d = f32::MAX;
                            for &oc in o_codes {
                                let cd = self.codebook.centroid_dist(cc as u32, oc as u32);
                                if cd < min_d { min_d = cd; }
                            }
                            if min_d < f32::MAX { d += min_d; }
                        }
                        d /= c_codes.len() as f32;
                        total_d += d;
                        n_compared += 1;
                    }
                    if n_compared > 0 { total_d /= n_compared as f32; }
                    if total_d < best_total {
                        best_total = total_d;
                        best_idx = candidate;
                    }
                }
                self.postings.cluster_reps[c] = Some(best_idx);
            }
        }

        // Phase 2: Reconnect isolated live nodes to nearest cluster rep
        for i in 0..total {
            if self.id_tracker.is_deleted(i as u32) { continue; }
            if !self.adjacency[i].is_empty() { continue; }

            let ctop = &self.doc_profiles[i].ctop;
            let mut connected = false;
            for &cluster in ctop {
                if let Some(Some(rep)) = self.postings.cluster_reps.get(cluster as usize) {
                    let rep = *rep;
                    if rep as usize != i && !self.id_tracker.is_deleted(rep) {
                        self.adjacency[i].push(rep);
                        self.adjacency[rep as usize].push(i as u32);
                        if self.adjacency[rep as usize].len() > self.max_degree {
                            shrink_neighbors_emd(
                                rep as usize, self.max_degree,
                                &mut self.adjacency, &self.codebook,
                                &self.flat_codes, self.use_emd,
                            );
                        }
                        connected = true;
                        break;
                    }
                }
            }

            if !connected {
                // Fallback: connect to nearest live node by centroid distance.
                // Cap scan to avoid O(n²) for large segments.
                let i_codes = self.flat_codes.doc_codes(i);
                if i_codes.is_empty() { continue; }
                let mut best: Option<(u32, f32)> = None;
                let scan_limit = total.min(256);
                let mut scanned = 0usize;
                for j in 0..total {
                    if j == i || self.id_tracker.is_deleted(j as u32) { continue; }
                    let j_codes = self.flat_codes.doc_codes(j);
                    if j_codes.is_empty() { continue; }
                    let mut d = 0.0f32;
                    for &ic in i_codes {
                        let mut min_d = f32::MAX;
                        for &jc in j_codes {
                            let cd = self.codebook.centroid_dist(ic as u32, jc as u32);
                            if cd < min_d { min_d = cd; }
                        }
                        if min_d < f32::MAX { d += min_d; }
                    }
                    d /= i_codes.len().max(1) as f32;
                    match best {
                        Some((_, bs)) if d < bs => { best = Some((j as u32, d)); }
                        None => { best = Some((j as u32, d)); }
                        _ => {}
                    }
                    scanned += 1;
                    if scanned >= scan_limit { break; }
                }
                if let Some((nbr, _)) = best {
                    self.adjacency[i].push(nbr);
                    self.adjacency[nbr as usize].push(i as u32);
                    if self.adjacency[nbr as usize].len() > self.max_degree {
                        shrink_neighbors_emd(
                            nbr as usize, self.max_degree,
                            &mut self.adjacency, &self.codebook,
                            &self.flat_codes, self.use_emd,
                        );
                    }
                }
            }
        }

        // Phase 3: Remove edges pointing to deleted nodes and deduplicate
        for i in 0..total {
            if self.id_tracker.is_deleted(i as u32) { continue; }
            self.adjacency[i].retain(|&nbr| !self.id_tracker.is_deleted(nbr));
            self.adjacency[i].sort_unstable();
            self.adjacency[i].dedup();
        }
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
