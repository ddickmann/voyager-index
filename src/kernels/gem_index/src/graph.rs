use serde::{Deserialize, Serialize};

use latence_gem_router::codebook::TwoStageCodebook;
use latence_gem_router::router::{FlatDocCodes, DocProfile, ClusterPostings};

use rand::SeedableRng;
use rayon::prelude::*;

use crate::emd::qch_proxy_between_docs;
use crate::score_cache::ScoreCache;
use crate::search::beam_search_construction;

/// Trait for graph adjacency access, generic over CSR and Vec<Vec<u32>>.
pub trait Adjacency {
    fn neighbors(&self, node: usize) -> &[u32];
    fn n_nodes(&self) -> usize;
    fn n_edges(&self) -> usize;
}

/// Compressed Sparse Row adjacency — single contiguous allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrAdjacency {
    pub edges: Vec<u32>,
    pub offsets: Vec<usize>,
}

impl CsrAdjacency {
    pub fn empty(n_nodes: usize) -> Self {
        Self {
            edges: Vec::new(),
            offsets: vec![0; n_nodes + 1],
        }
    }

    pub fn from_adj_lists(adj: &[Vec<u32>]) -> Self {
        let n = adj.len();
        let mut offsets = Vec::with_capacity(n + 1);
        let total_edges: usize = adj.iter().map(|a| a.len()).sum();
        let mut edges = Vec::with_capacity(total_edges);

        let mut off = 0usize;
        for neighbors in adj {
            offsets.push(off);
            edges.extend_from_slice(neighbors);
            off += neighbors.len();
        }
        offsets.push(off);

        Self { edges, offsets }
    }

    pub fn to_adj_lists(&self) -> Vec<Vec<u32>> {
        let n = self.n_nodes();
        (0..n).map(|i| self.neighbors(i).to_vec()).collect()
    }
}

impl Adjacency for CsrAdjacency {
    #[inline]
    fn neighbors(&self, node: usize) -> &[u32] {
        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        &self.edges[start..end]
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        if self.offsets.is_empty() { 0 } else { self.offsets.len() - 1 }
    }

    fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Vec<Vec<u32>> adjacency for mutable segments.
pub struct VecAdjacency<'a>(pub &'a [Vec<u32>]);

impl<'a> Adjacency for VecAdjacency<'a> {
    #[inline]
    fn neighbors(&self, node: usize) -> &[u32] {
        &self.0[node]
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.0.len()
    }

    fn n_edges(&self) -> usize {
        self.0.iter().map(|a| a.len()).sum()
    }
}

/// Assign HNSW level to a node using geometric distribution.
/// ml = 1 / ln(max_degree) gives optimal level distribution.
fn random_level(ml: f64, rng: &mut impl rand::Rng) -> usize {
    let r: f64 = rng.gen::<f64>().max(1e-15);
    (-r.ln() * ml).floor() as usize
}

/// Sealed multi-level GEM graph with CSR adjacency and semantic shortcuts.
pub struct GemGraph {
    /// Layer 0 is the bottom (all nodes), higher layers are sparser.
    pub levels: Vec<CsrAdjacency>,
    pub shortcuts: Vec<Vec<u32>>,
    pub node_levels: Vec<usize>,
    pub entry_point: u32,
    pub max_degree: usize,
}

impl GemGraph {
    pub fn n_nodes(&self) -> usize {
        if self.levels.is_empty() { 0 } else { self.levels[0].n_nodes() }
    }

    pub fn n_edges(&self) -> usize {
        self.levels.iter().map(|l| l.n_edges()).sum()
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn total_shortcuts(&self) -> usize {
        self.shortcuts.iter().map(|s| s.len()).sum()
    }

    /// Convenience: bottom-layer adjacency for backward compatibility.
    pub fn bottom_adjacency(&self) -> &CsrAdjacency {
        &self.levels[0]
    }

    pub fn inject_shortcuts(
        &mut self,
        pairs: &[(Vec<f32>, u32)],
        max_per_node: usize,
        codebook: &TwoStageCodebook,
        flat_codes: &FlatDocCodes,
        dim: usize,
    ) {
        if dim == 0 || self.levels.is_empty() {
            return;
        }
        let bottom = &self.levels[0];
        for (query_flat, target_int) in pairs {
            let target = *target_int as usize;
            if target >= bottom.n_nodes() {
                continue;
            }
            let n_query = query_flat.len() / dim;
            if n_query == 0 {
                continue;
            }
            let query_scores = codebook.compute_query_centroid_scores(query_flat, n_query);
            let n_fine = codebook.n_fine;

            let candidates = beam_search_construction(
                bottom,
                &[self.entry_point],
                &query_scores,
                n_query,
                flat_codes,
                n_fine,
                64,
                bottom.n_nodes(),
            );

            for (cand_idx, _) in candidates.iter().take(max_per_node) {
                let cand = *cand_idx as usize;
                if cand != target && !self.shortcuts[target].contains(cand_idx) {
                    if self.shortcuts[target].len() < max_per_node {
                        self.shortcuts[target].push(*cand_idx);
                    }
                }
            }
        }
    }
}

/// Select neighbors using HNSW diversity heuristic (GEM paper Algorithm 2).
/// Uses optional score cache to avoid redundant distance computations.
pub fn select_neighbors_heuristic(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_cached(candidates, query_codes, max_degree, codebook, flat_codes, None)
}

pub fn select_neighbors_heuristic_cached(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    mut cache: Option<&mut ScoreCache>,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_inner(candidates, query_codes, max_degree, codebook, flat_codes, &mut cache, None, 0)
}

/// Payload-aware variant: candidates sharing a payload cluster with query_node
/// get a distance bonus, biasing edges toward same-payload documents.
pub fn select_neighbors_payload_aware(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    mut cache: Option<&mut ScoreCache>,
    payload_clusters: &[u32],
    query_node: u32,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_inner(candidates, query_codes, max_degree, codebook, flat_codes, &mut cache, Some(payload_clusters), query_node)
}

fn select_neighbors_heuristic_inner(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    cache: &mut Option<&mut ScoreCache>,
    payload_clusters: Option<&[u32]>,
    query_node: u32,
) -> Vec<(u32, f32)> {
    let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_degree);

    // Pre-compute candidate-to-query scores in parallel for large candidate sets
    let cand_to_query_scores: Vec<f32> = if candidates.len() > 64 {
        candidates.par_iter()
            .map(|&(cand_idx, _)| {
                let cand_codes = flat_codes.doc_codes(cand_idx as usize);
                qch_proxy_between_docs(codebook, cand_codes, query_codes)
            })
            .collect()
    } else {
        candidates.iter()
            .map(|&(cand_idx, _)| {
                let cand_codes = flat_codes.doc_codes(cand_idx as usize);
                qch_proxy_between_docs(codebook, cand_codes, query_codes)
            })
            .collect()
    };

    // Payload affinity bonus: 5% distance reduction for same-cluster pairs
    const PAYLOAD_BONUS: f32 = 0.95;

    for (ci, &(cand_idx, _)) in candidates.iter().enumerate() {
        if selected.len() >= max_degree {
            break;
        }
        let mut cand_to_query = cand_to_query_scores[ci];
        if let Some(pc) = payload_clusters {
            let qn = query_node as usize;
            let cn = cand_idx as usize;
            if qn < pc.len() && cn < pc.len() && pc[qn] == pc[cn] {
                cand_to_query *= PAYLOAD_BONUS;
            }
        }
        let cand_codes = flat_codes.doc_codes(cand_idx as usize);

        let too_close = selected.iter().any(|&(sel_idx, _)| {
            let dist = if let Some(ref mut cache) = cache {
                cache.get(cand_idx, sel_idx)
            } else {
                None
            };
            let cand_to_sel = dist.unwrap_or_else(|| {
                let sel_codes = flat_codes.doc_codes(sel_idx as usize);
                qch_proxy_between_docs(codebook, cand_codes, sel_codes)
            });
            cand_to_sel < cand_to_query
        });

        if !too_close {
            if let Some(ref mut c) = cache {
                for &(sel_idx, _) in &selected {
                    if c.get(cand_idx, sel_idx).is_none() {
                        let sel_codes = flat_codes.doc_codes(sel_idx as usize);
                        let d = qch_proxy_between_docs(codebook, cand_codes, sel_codes);
                        c.insert(cand_idx, sel_idx, d);
                    }
                }
            }
            selected.push((cand_idx, cand_to_query));
        }
    }

    selected
}

/// Shrink a node's neighbor list to max_degree using the diversity heuristic.
/// Maintains bidirectional consistency: removed neighbors drop their reverse edge.
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
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let kept = select_neighbors_heuristic(&scored, node_codes, max_degree, codebook, flat_codes);
    let kept_set: std::collections::HashSet<u32> = kept.iter().map(|&(idx, _)| idx).collect();

    for &old_nbr in &neighbors {
        if !kept_set.contains(&old_nbr) {
            adjacency[old_nbr as usize].retain(|&x| x != node_idx as u32);
        }
    }

    adjacency[node_idx] = kept.iter().map(|&(idx, _)| idx).collect();
}

/// Build a multi-level GEM graph using HNSW-style hierarchical insertion.
/// `payload_clusters`: optional per-doc cluster IDs from payload similarity.
/// When provided, neighbor selection biases toward same-cluster edges,
/// making filtered search faster (Qdrant uses post-hoc filtering; we do it natively).
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
    build_graph_with_payload(all_vectors, dim, doc_offsets, codebook, flat_codes, doc_profiles, postings, max_degree, ef_construction, None)
}

pub fn build_graph_with_payload(
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    doc_profiles: &[DocProfile],
    postings: &ClusterPostings,
    max_degree: usize,
    ef_construction: usize,
    payload_clusters: Option<&[u32]>,
) -> GemGraph {
    let n_docs = doc_offsets.len();
    if n_docs == 0 {
        return GemGraph {
            levels: vec![CsrAdjacency::empty(0)],
            shortcuts: Vec::new(),
            node_levels: Vec::new(),
            entry_point: 0,
            max_degree,
        };
    }

    let n_fine = codebook.n_fine;
    let ml = 1.0 / (max_degree as f64).ln();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut node_levels: Vec<usize> = Vec::with_capacity(n_docs);
    let mut max_level = 0usize;
    for _ in 0..n_docs {
        let level = random_level(ml, &mut rng);
        if level > max_level {
            max_level = level;
        }
        node_levels.push(level);
    }

    let n_levels = max_level + 1;
    let mut score_cache = ScoreCache::new(n_docs * 16);
    let mut levels: Vec<Vec<Vec<u32>>> = (0..n_levels)
        .map(|_| vec![Vec::new(); n_docs])
        .collect();

    let max_degree_upper = max_degree;
    let max_degree_bottom = max_degree;

    let mut entry_point: u32 = 0;

    for i in 1..n_docs {
        let (start, end) = doc_offsets[i];
        let n_tokens = end - start;
        let doc_vecs = &all_vectors[start * dim..end * dim];
        let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);
        let node_level = node_levels[i];
        let ep_level = node_levels[entry_point as usize];

        let mut cur_entry = entry_point;

        // Greedy search from top to node_level+1 (upper layers, ef=1)
        for level in (node_level + 1..=ep_level.min(max_level)).rev() {
            let adj = VecAdjacency(&levels[level]);
            let candidates = beam_search_construction(
                &adj, &[cur_entry], &query_scores, n_tokens,
                flat_codes, n_fine, 1, i,
            );
            if let Some(&(best, _)) = candidates.first() {
                cur_entry = best;
            }
        }

        // Search and connect at each level from min(node_level, ep_level) down to 0
        let start_level = node_level.min(ep_level.min(max_level));
        for level in (0..=start_level).rev() {
            let md = if level == 0 { max_degree_bottom } else { max_degree_upper };

            // Use cluster-guided entry points for level 0
            let mut entries = vec![cur_entry];
            if level == 0 {
                for &cluster in &doc_profiles[i].ctop {
                    if let Some(reps) = postings.cluster_reps.get(cluster as usize) {
                        if let Some(rep) = reps {
                            if (*rep as usize) < i && !entries.contains(rep) {
                                entries.push(*rep);
                            }
                        }
                    }
                }
            }

            let adj = VecAdjacency(&levels[level]);
            let candidates = beam_search_construction(
                &adj, &entries, &query_scores, n_tokens,
                flat_codes, n_fine, ef_construction, i,
            );
            drop(adj);

            if let Some(&(best, _)) = candidates.first() {
                cur_entry = best;
            }

            let doc_codes = flat_codes.doc_codes(i);
            let neighbors = if let Some(pc) = payload_clusters {
                select_neighbors_payload_aware(
                    &candidates, doc_codes, md, codebook, flat_codes,
                    Some(&mut score_cache), pc, i as u32,
                )
            } else {
                select_neighbors_heuristic_cached(
                    &candidates, doc_codes, md, codebook, flat_codes,
                    Some(&mut score_cache),
                )
            };

            for &(nbr_idx, _score) in &neighbors {
                levels[level][i].push(nbr_idx);
                levels[level][nbr_idx as usize].push(i as u32);

                if levels[level][nbr_idx as usize].len() > md {
                    shrink_neighbors(
                        nbr_idx as usize, md,
                        &mut levels[level], codebook, flat_codes,
                    );
                }
            }
        }

        if node_level > node_levels[entry_point as usize] {
            entry_point = i as u32;
        }
    }

    // Bridge repair on bottom layer only
    bridge_repair(&mut levels[0], max_degree, postings, codebook, flat_codes);

    let csr_levels: Vec<CsrAdjacency> = levels
        .iter()
        .map(|l| CsrAdjacency::from_adj_lists(l))
        .collect();

    GemGraph {
        levels: csr_levels,
        shortcuts: vec![Vec::new(); n_docs],
        node_levels,
        entry_point,
        max_degree,
    }
}

/// Bridge repair on a mutable adjacency layer.
pub fn bridge_repair(
    adjacency: &mut Vec<Vec<u32>>,
    max_degree: usize,
    postings: &ClusterPostings,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) {
    let n_clusters = postings.lists.len();

    for cluster_id in 0..n_clusters {
        let entry_rep = match postings.cluster_reps.get(cluster_id) {
            Some(Some(rep)) => *rep as usize,
            _ => continue,
        };

        let members = &postings.lists[cluster_id];
        if members.len() <= 1 {
            continue;
        }

        let mut reachable = std::collections::HashSet::new();
        let mut bfs_queue = std::collections::VecDeque::new();
        let member_set: std::collections::HashSet<u32> = members.iter().copied().collect();

        bfs_queue.push_back(entry_rep as u32);
        reachable.insert(entry_rep as u32);

        let mut spare_capacity: Vec<u32> = Vec::new();
        if adjacency[entry_rep].len() < max_degree {
            spare_capacity.push(entry_rep as u32);
        }

        while let Some(node) = bfs_queue.pop_front() {
            for &nbr in &adjacency[node as usize] {
                if member_set.contains(&nbr) && !reachable.contains(&nbr) {
                    reachable.insert(nbr);
                    bfs_queue.push_back(nbr);
                    if adjacency[nbr as usize].len() < max_degree {
                        spare_capacity.push(nbr);
                    }
                }
            }
        }

        let mut spare_idx = 0;
        for &member in members {
            if reachable.contains(&member) {
                continue;
            }

            let bridge_source = if spare_idx < spare_capacity.len() {
                let src = spare_capacity[spare_idx] as usize;
                spare_idx += 1;
                src
            } else {
                evict_worst_neighbor(entry_rep, max_degree, adjacency, codebook, flat_codes);
                entry_rep
            };

            let m = member as usize;
            if !adjacency[bridge_source].contains(&member) {
                adjacency[bridge_source].push(member);
            }
            if !adjacency[m].contains(&(bridge_source as u32)) {
                adjacency[m].push(bridge_source as u32);
                if adjacency[m].len() > max_degree {
                    evict_worst_neighbor(m, max_degree, adjacency, codebook, flat_codes);
                }
            }
        }
    }
}

fn evict_worst_neighbor(
    node: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) -> Option<u32> {
    if adjacency[node].len() < max_degree {
        return None;
    }
    if adjacency[node].is_empty() {
        return None;
    }
    let node_codes = flat_codes.doc_codes(node);
    let worst_idx = adjacency[node]
        .iter()
        .enumerate()
        .map(|(i, &nbr)| {
            let nbr_codes = flat_codes.doc_codes(nbr as usize);
            let dist = qch_proxy_between_docs(codebook, node_codes, nbr_codes);
            (i, dist)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i);
    worst_idx.map(|i| adjacency[node].swap_remove(i))
}
