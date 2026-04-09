use latence_gem_router::codebook::TwoStageCodebook;
use latence_gem_router::router::{FlatDocCodes, DocProfile, ClusterPostings};

use crate::emd::qch_proxy_between_docs;
use crate::search::beam_search_construction;

/// Sealed GEM proximity graph with optional semantic shortcuts.
pub struct GemGraph {
    pub adjacency: Vec<Vec<u32>>,
    pub shortcuts: Vec<Vec<u32>>,
    pub max_degree: usize,
}

impl GemGraph {
    pub fn n_nodes(&self) -> usize {
        self.adjacency.len()
    }

    pub fn n_edges(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum()
    }

    pub fn total_shortcuts(&self) -> usize {
        self.shortcuts.iter().map(|s| s.len()).sum()
    }

    pub fn neighbors(&self, idx: usize) -> &[u32] {
        &self.adjacency[idx]
    }

    pub fn inject_shortcuts(
        &mut self,
        pairs: &[(Vec<f32>, u32)],
        max_per_node: usize,
        codebook: &TwoStageCodebook,
        flat_codes: &FlatDocCodes,
        dim: usize,
    ) {
        for (query_flat, target_int) in pairs {
            let target = *target_int as usize;
            if target >= self.adjacency.len() {
                continue;
            }
            let n_query = query_flat.len() / dim;
            if n_query == 0 {
                continue;
            }
            let query_scores = codebook.compute_query_centroid_scores(query_flat, n_query);
            let n_fine = codebook.n_fine;

            let candidates = beam_search_construction(
                &self.adjacency,
                &[0],
                &query_scores,
                n_query,
                flat_codes,
                n_fine,
                64,
                self.adjacency.len(),
            );

            for (cand_idx, _) in candidates.iter().take(max_per_node) {
                let cand = *cand_idx as usize;
                if cand != target && !self.shortcuts[target].contains(&(*cand_idx)) {
                    if self.shortcuts[target].len() < max_per_node {
                        self.shortcuts[target].push(*cand_idx);
                    }
                }
            }
        }
    }
}

/// Select neighbors using HNSW diversity heuristic.
/// Candidates must be sorted ascending by score (best-first).
pub fn select_neighbors_heuristic(
    candidates: &[(u32, f32)],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) -> Vec<(u32, f32)> {
    let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_degree);

    for &(cand_idx, cand_dist) in candidates {
        if selected.len() >= max_degree {
            break;
        }
        let cand_codes = flat_codes.doc_codes(cand_idx as usize);

        let too_close = selected.iter().any(|&(sel_idx, _)| {
            let sel_codes = flat_codes.doc_codes(sel_idx as usize);
            let pair_dist = qch_proxy_between_docs(codebook, cand_codes, sel_codes);
            pair_dist < cand_dist
        });

        if !too_close {
            selected.push((cand_idx, cand_dist));
        }
    }

    selected
}

/// Shrink a node's neighbor list to max_degree using the diversity heuristic.
pub fn shrink_neighbors(
    node_idx: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) {
    let node_codes = flat_codes.doc_codes(node_idx);
    let neighbors: Vec<u32> = adjacency[node_idx].clone();

    let mut scored: Vec<(u32, f32)> = neighbors
        .iter()
        .map(|&nbr| {
            let nbr_codes = flat_codes.doc_codes(nbr as usize);
            let dist = qch_proxy_between_docs(codebook, node_codes, nbr_codes);
            (nbr, dist)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let kept = select_neighbors_heuristic(&scored, max_degree, codebook, flat_codes);
    adjacency[node_idx] = kept.iter().map(|&(idx, _)| idx).collect();
}

/// Build a GEM proximity graph using sequential HNSW-style insertion
/// with cluster-guided entry points and diversity-based neighbor selection.
pub fn build_graph(
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    doc_profiles: &[DocProfile],
    postings: &ClusterPostings,
    max_degree: usize,
    ef_construction: usize,
) -> GemGraph {
    let n_docs = doc_offsets.len();
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n_docs];
    let n_fine = codebook.n_fine;

    for i in 1..n_docs {
        let (start, end) = doc_offsets[i];
        let n_tokens = end - start;
        let doc_vecs = &all_vectors[start * dim..end * dim];
        let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);

        let mut entries: Vec<u32> = Vec::new();
        for &cluster in &doc_profiles[i].ctop {
            if let Some(reps) = postings.cluster_reps.get(cluster as usize) {
                if let Some(rep) = reps {
                    if (*rep as usize) < i && !entries.contains(rep) {
                        entries.push(*rep);
                    }
                }
            }
        }
        if entries.is_empty() {
            entries.push(0);
        }

        let candidates = beam_search_construction(
            &adjacency,
            &entries,
            &query_scores,
            n_tokens,
            flat_codes,
            n_fine,
            ef_construction,
            i,
        );

        let neighbors = select_neighbors_heuristic(
            &candidates,
            max_degree,
            codebook,
            flat_codes,
        );

        for &(nbr_idx, _score) in &neighbors {
            adjacency[i].push(nbr_idx);
            adjacency[nbr_idx as usize].push(i as u32);

            if adjacency[nbr_idx as usize].len() > max_degree {
                shrink_neighbors(
                    nbr_idx as usize,
                    max_degree,
                    &mut adjacency,
                    codebook,
                    flat_codes,
                );
            }
        }
    }

    GemGraph {
        adjacency,
        shortcuts: vec![Vec::new(); n_docs],
        max_degree,
    }
}
