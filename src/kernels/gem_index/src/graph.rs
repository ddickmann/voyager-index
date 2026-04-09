use serde::{Deserialize, Serialize};

use latence_gem_router::codebook::TwoStageCodebook;
use latence_gem_router::router::{FlatDocCodes, DocProfile, ClusterPostings};

use rand::SeedableRng;
use rayon::prelude::*;

use crate::emd::construction_distance;
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
    /// Per-shortcut generation counter for aging-based pruning.
    pub shortcut_generations: Vec<Vec<usize>>,
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

    /// BFS-based connectivity report on the bottom layer.
    ///
    /// Returns (n_components, giant_component_frac) where giant_component_frac
    /// is the fraction of nodes in the largest connected component.
    pub fn connectivity_report(&self) -> (usize, f64) {
        if self.levels.is_empty() {
            return (0, 0.0);
        }
        let n = self.levels[0].n_nodes();
        if n == 0 {
            return (0, 0.0);
        }

        let adj = &self.levels[0];
        let mut visited = vec![false; n];
        let mut component_sizes: Vec<usize> = Vec::new();
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut size = 0usize;
            queue.push_back(start);
            visited[start] = true;
            while let Some(node) = queue.pop_front() {
                size += 1;
                for &nbr in adj.neighbors(node) {
                    let nb = nbr as usize;
                    if nb < n && !visited[nb] {
                        visited[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }
            component_sizes.push(size);
        }

        let n_components = component_sizes.len();
        let giant = component_sizes.iter().copied().max().unwrap_or(0);
        let frac = if n > 0 { giant as f64 / n as f64 } else { 0.0 };
        (n_components, frac)
    }

    /// Remove shortcuts pointing to deleted nodes. Optionally prune shortcuts
    /// older than `max_age` generations (tracked via `shortcut_generations`).
    pub fn prune_stale_shortcuts(&mut self, deleted: &[bool], max_age: Option<usize>, current_generation: usize) {
        let n = self.shortcuts.len();
        let has_gens = self.shortcut_generations.len() == n;

        // Phase 1: Remove shortcuts pointing to deleted nodes, keeping
        // shortcut_generations in lockstep so indices stay aligned.
        for i in 0..n {
            let mut keep_sc = Vec::new();
            let mut keep_gen = Vec::new();
            for (idx, &target) in self.shortcuts[i].iter().enumerate() {
                let t = target as usize;
                if t < deleted.len() && deleted[t] {
                    continue;
                }
                keep_sc.push(target);
                if has_gens {
                    if let Some(&g) = self.shortcut_generations[i].get(idx) {
                        keep_gen.push(g);
                    }
                }
            }
            self.shortcuts[i] = keep_sc;
            if has_gens {
                self.shortcut_generations[i] = keep_gen;
            }
        }

        // Phase 2: Age-based pruning (only if generations are tracked).
        if let Some(age_limit) = max_age {
            if has_gens {
                for i in 0..n {
                    let gens = &mut self.shortcut_generations[i];
                    let sc = &mut self.shortcuts[i];
                    let mut keep = Vec::with_capacity(sc.len());
                    let mut keep_gen = Vec::with_capacity(gens.len());
                    for (idx, &target) in sc.iter().enumerate() {
                        let gen = gens.get(idx).copied().unwrap_or(0);
                        if current_generation.saturating_sub(gen) <= age_limit {
                            keep.push(target);
                            keep_gen.push(gen);
                        }
                    }
                    *sc = keep;
                    *gens = keep_gen;
                }
            }
        }
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
                if cand != target && !self.shortcuts[target].contains(cand_idx)
                    && self.shortcuts[target].len() < max_per_node {
                        self.shortcuts[target].push(*cand_idx);
                    }
            }
        }
    }
}

/// Select neighbors using HNSW diversity heuristic (GEM paper Algorithm 2).
/// When `use_emd` is true, uses qEMD (Earth Mover's Distance) for edge scoring
/// during construction (metric decoupling from the GEM paper, Section 4.2).
pub fn select_neighbors_heuristic(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_cached(candidates, query_codes, max_degree, codebook, flat_codes, None, false)
}

pub fn select_neighbors_heuristic_cached(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    mut cache: Option<&mut ScoreCache>,
    use_emd: bool,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_inner(candidates, query_codes, max_degree, codebook, flat_codes, &mut cache, None, 0, use_emd)
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
    use_emd: bool,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_inner(candidates, query_codes, max_degree, codebook, flat_codes, &mut cache, Some(payload_clusters), query_node, use_emd)
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
    use_emd: bool,
) -> Vec<(u32, f32)> {
    let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_degree);

    // Pre-compute candidate-to-query distances using the construction metric.
    // When use_emd=true, this uses qEMD (metric, triangle inequality) instead of qCH.
    let cand_to_query_scores: Vec<f32> = if candidates.len() > 16 {
        candidates.par_iter()
            .map(|&(cand_idx, _)| {
                let cand_codes = flat_codes.doc_codes(cand_idx as usize);
                construction_distance(codebook, cand_codes, query_codes, use_emd)
            })
            .collect()
    } else {
        candidates.iter()
            .map(|&(cand_idx, _)| {
                let cand_codes = flat_codes.doc_codes(cand_idx as usize);
                construction_distance(codebook, cand_codes, query_codes, use_emd)
            })
            .collect()
    };

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
                construction_distance(codebook, cand_codes, sel_codes, use_emd)
            });
            cand_to_sel < cand_to_query
        });

        if !too_close {
            if let Some(ref mut c) = cache {
                for &(sel_idx, _) in &selected {
                    if c.get(cand_idx, sel_idx).is_none() {
                        let sel_codes = flat_codes.doc_codes(sel_idx as usize);
                        let d = construction_distance(codebook, cand_codes, sel_codes, use_emd);
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
    shrink_neighbors_emd(node_idx, max_degree, adjacency, codebook, flat_codes, false)
}

pub fn shrink_neighbors_emd(
    node_idx: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) {
    shrink_neighbors_emd_cached(node_idx, max_degree, adjacency, codebook, flat_codes, use_emd, None)
}

pub fn shrink_neighbors_emd_cached(
    node_idx: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
    mut cache: Option<&mut ScoreCache>,
) {
    let node_codes = flat_codes.doc_codes(node_idx);
    let neighbors: Vec<u32> = adjacency[node_idx].clone();
    let node_u32 = node_idx as u32;

    let mut scored: Vec<(u32, f32)> = neighbors
        .iter()
        .map(|&nbr| {
            let dist = if let Some(ref mut c) = cache {
                if let Some(d) = c.get(node_u32, nbr) {
                    d
                } else {
                    let nbr_codes = flat_codes.doc_codes(nbr as usize);
                    let d = construction_distance(codebook, node_codes, nbr_codes, use_emd);
                    c.insert(node_u32, nbr, d);
                    d
                }
            } else {
                let nbr_codes = flat_codes.doc_codes(nbr as usize);
                construction_distance(codebook, node_codes, nbr_codes, use_emd)
            };
            (nbr, dist)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let kept = select_neighbors_heuristic_cached(
        &scored, node_codes, max_degree, codebook, flat_codes, cache, use_emd,
    );
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
    build_graph_with_payload(all_vectors, dim, doc_offsets, codebook, flat_codes, doc_profiles, postings, max_degree, ef_construction, None, false)
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
    use_emd: bool,
) -> GemGraph {
    let n_docs = doc_offsets.len();
    if n_docs == 0 {
        return GemGraph {
            levels: vec![CsrAdjacency::empty(0)],
            shortcuts: Vec::new(),
            shortcut_generations: Vec::new(),
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

    for i in 0..n_docs {
        let (start, end) = doc_offsets[i];
        let n_tokens = end - start;
        let doc_vecs = &all_vectors[start * dim..end * dim];
        let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);
        let node_level = node_levels[i];

        if i == 0 {
            // First node: no neighbors to search yet, just update entry point
            if node_level > node_levels[entry_point as usize] {
                entry_point = i as u32;
            }
            continue;
        }

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
                    if let Some(Some(rep)) = postings.cluster_reps.get(cluster as usize) {
                        if (*rep as usize) < i && !entries.contains(rep) {
                            entries.push(*rep);
                        }
                    }
                }
            }

            let candidates = {
                let adj = VecAdjacency(&levels[level]);
                beam_search_construction(
                    &adj, &entries, &query_scores, n_tokens,
                    flat_codes, n_fine, ef_construction, i,
                )
            };

            if let Some(&(best, _)) = candidates.first() {
                cur_entry = best;
            }

            let doc_codes = flat_codes.doc_codes(i);
            let neighbors = if let Some(pc) = payload_clusters {
                select_neighbors_payload_aware(
                    &candidates, doc_codes, md, codebook, flat_codes,
                    Some(&mut score_cache), pc, i as u32, use_emd,
                )
            } else {
                select_neighbors_heuristic_cached(
                    &candidates, doc_codes, md, codebook, flat_codes,
                    Some(&mut score_cache), use_emd,
                )
            };

            for &(nbr_idx, _score) in &neighbors {
                levels[level][i].push(nbr_idx);
                levels[level][nbr_idx as usize].push(i as u32);

                if levels[level][nbr_idx as usize].len() > md {
                    shrink_neighbors_emd_cached(
                        nbr_idx as usize, md,
                        &mut levels[level], codebook, flat_codes, use_emd,
                        Some(&mut score_cache),
                    );
                }
            }
        }

        if node_level > node_levels[entry_point as usize] {
            entry_point = i as u32;
        }
    }

    // Bridge repair on bottom layer only
    bridge_repair_emd(&mut levels[0], max_degree, postings, codebook, flat_codes, use_emd);
    global_bridge_repair(&mut levels[0], max_degree, codebook, flat_codes, use_emd);

    let csr_levels: Vec<CsrAdjacency> = levels
        .iter()
        .map(|l| CsrAdjacency::from_adj_lists(l))
        .collect();

    GemGraph {
        levels: csr_levels,
        shortcuts: vec![Vec::new(); n_docs],
        shortcut_generations: vec![Vec::new(); n_docs],
        node_levels,
        entry_point,
        max_degree,
    }
}

/// Bridge repair on a mutable adjacency layer.
pub fn bridge_repair(
    adjacency: &mut [Vec<u32>],
    max_degree: usize,
    postings: &ClusterPostings,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) {
    bridge_repair_emd(adjacency, max_degree, postings, codebook, flat_codes, false)
}

pub fn bridge_repair_emd(
    adjacency: &mut [Vec<u32>],
    max_degree: usize,
    postings: &ClusterPostings,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) {
    let n_clusters = postings.lists.len();

    // Phase 1 (parallel): BFS per-cluster to discover unreachable members.
    // Returns (cluster_entry_rep, spare_capacity_nodes, unreachable_members).
    let adj_snapshot: &[Vec<u32>] = adjacency;
    let cluster_plans: Vec<(usize, Vec<u32>, Vec<u32>)> = (0..n_clusters)
        .into_par_iter()
        .filter_map(|cluster_id| {
            let entry_rep = match postings.cluster_reps.get(cluster_id) {
                Some(Some(rep)) => *rep as usize,
                _ => return None,
            };
            let members = &postings.lists[cluster_id];
            if members.len() <= 1 {
                return None;
            }

            let member_set: std::collections::HashSet<u32> = members.iter().copied().collect();
            let mut reachable = std::collections::HashSet::new();
            let mut bfs_queue = std::collections::VecDeque::new();

            bfs_queue.push_back(entry_rep as u32);
            reachable.insert(entry_rep as u32);

            let mut spare = Vec::new();
            if adj_snapshot[entry_rep].len() < max_degree {
                spare.push(entry_rep as u32);
            }

            while let Some(node) = bfs_queue.pop_front() {
                for &nbr in &adj_snapshot[node as usize] {
                    if member_set.contains(&nbr) && !reachable.contains(&nbr) {
                        reachable.insert(nbr);
                        bfs_queue.push_back(nbr);
                        if adj_snapshot[nbr as usize].len() < max_degree {
                            spare.push(nbr);
                        }
                    }
                }
            }

            let unreachable: Vec<u32> = members.iter()
                .copied()
                .filter(|m| !reachable.contains(m))
                .collect();

            if unreachable.is_empty() {
                return None;
            }
            Some((entry_rep, spare, unreachable))
        })
        .collect();

    // Phase 2 (sequential): Apply bridge edges.
    for (entry_rep, spare_capacity, unreachable) in cluster_plans {
        let mut spare_idx = 0;
        for &member in &unreachable {
            let bridge_source = if spare_idx < spare_capacity.len() {
                let src = spare_capacity[spare_idx] as usize;
                spare_idx += 1;
                src
            } else {
                evict_worst_neighbor_emd(entry_rep, max_degree, adjacency, codebook, flat_codes, use_emd);
                entry_rep
            };

            let m = member as usize;
            if !adjacency[bridge_source].contains(&member) {
                adjacency[bridge_source].push(member);
            }
            if !adjacency[m].contains(&(bridge_source as u32)) {
                adjacency[m].push(bridge_source as u32);
                if adjacency[m].len() > max_degree {
                    evict_worst_neighbor_emd(m, max_degree, adjacency, codebook, flat_codes, use_emd);
                }
            }
        }
    }
}

fn bfs_components(adjacency: &[Vec<u32>]) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut visited = vec![false; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(node) = queue.pop_front() {
            comp.push(node);
            for &nbr in &adjacency[node] {
                let nb = nbr as usize;
                if nb < n && !visited[nb] {
                    visited[nb] = true;
                    queue.push_back(nb);
                }
            }
        }
        components.push(comp);
    }
    components
}

/// Add a bridge edge that cannot be evicted. We allow max_degree+1 for bridge
/// nodes rather than risk the bridge being evicted as the "worst" neighbor,
/// which would leave components disconnected.
fn add_bridge_edge(
    adjacency: &mut [Vec<u32>],
    a: usize,
    b: usize,
    _max_degree: usize,
    _codebook: &TwoStageCodebook,
    _flat_codes: &FlatDocCodes,
    _use_emd: bool,
) {
    if !adjacency[a].contains(&(b as u32)) {
        adjacency[a].push(b as u32);
    }
    if !adjacency[b].contains(&(a as u32)) {
        adjacency[b].push(a as u32);
    }
}

/// Global bridge repair: connect ALL disconnected components into a single
/// connected graph. Runs iteratively until only one component remains.
///
/// For each small component, finds the nearest bridge edge to the giant
/// component by scanning all giant nodes. Falls back to connecting the first
/// node of each small component to the giant entry if no code-based pair exists.
/// Bridge edges are never evicted (they may exceed max_degree by 1).
pub fn global_bridge_repair(
    adjacency: &mut [Vec<u32>],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) {
    let n = adjacency.len();
    if n <= 1 {
        return;
    }

    const MAX_ROUNDS: usize = 10;
    for round in 0..MAX_ROUNDS {
        let components = bfs_components(adjacency);
        if components.len() <= 1 {
            return;
        }

        let giant_idx = components
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.len())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let giant_entry = components[giant_idx][0];
        let giant_comp = &components[giant_idx];

        // Parallel: find best bridge pair per small component.
        let small_comps: Vec<(usize, &Vec<usize>)> = components.iter().enumerate()
            .filter(|&(ci, c)| ci != giant_idx && !c.is_empty())
            .collect();

        let bridge_pairs: Vec<(usize, usize)> = small_comps.par_iter()
            .map(|&(_, comp)| {
                let mut best_pair: Option<(usize, usize, f32)> = None;
                for &small_node in comp {
                    let small_codes = flat_codes.doc_codes(small_node);
                    if small_codes.is_empty() {
                        continue;
                    }
                    for &giant_node in giant_comp {
                        let giant_codes = flat_codes.doc_codes(giant_node);
                        if giant_codes.is_empty() {
                            continue;
                        }
                        let d = construction_distance(codebook, small_codes, giant_codes, use_emd);
                        if best_pair.is_none() || d < best_pair.unwrap().2 {
                            best_pair = Some((small_node, giant_node, d));
                        }
                    }
                }
                match best_pair {
                    Some((s, g, _)) => (s, g),
                    None => (comp[0], giant_entry),
                }
            })
            .collect();

        if bridge_pairs.is_empty() {
            break;
        }

        for (bridge_small, bridge_giant) in bridge_pairs {
            add_bridge_edge(adjacency, bridge_small, bridge_giant, max_degree, codebook, flat_codes, use_emd);
        }

        let remaining = bfs_components(adjacency).len();
        if remaining <= 1 {
            return;
        }
        if round == MAX_ROUNDS - 1 {
            log::warn!(
                "global_bridge_repair: {} components remain after {} rounds (expected 1)",
                remaining, MAX_ROUNDS,
            );
        }
    }
}

fn evict_worst_neighbor_emd(
    node: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
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
            let dist = construction_distance(codebook, node_codes, nbr_codes, use_emd);
            (i, dist)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i);
    worst_idx.map(|i| adjacency[node].swap_remove(i))
}

// ---------------------------------------------------------------------------
// Dual-Graph Construction (GEM Paper Section 4.3, Algorithms 1-3)
// ---------------------------------------------------------------------------

/// Build a GEM graph using the paper's dual-graph per-cluster construction.
///
/// Instead of HNSW-style sequential insertion over all documents, this builds
/// per-cluster local graphs, then merges them via bridge sets. This gives
/// better intra-cluster locality and explicit cross-cluster connectivity.
///
/// Algorithm 1: For each cluster C_i, process its member documents.
/// - First encounter (not yet inserted): `build_cluster_local` connects the doc
///   to existing cluster members using beam search + diversity heuristic.
/// - Subsequent encounter (already inserted via another cluster): `update_bridges`
///   merges old neighbors with new cluster neighbors, enforcing at least one
///   neighbor from each of the document's assigned clusters.
pub fn build_graph_dual(
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    doc_profiles: &[DocProfile],
    postings: &ClusterPostings,
    max_degree: usize,
    ef_construction: usize,
    use_emd: bool,
) -> GemGraph {
    let n_docs = doc_offsets.len();
    if n_docs == 0 {
        return GemGraph {
            levels: vec![CsrAdjacency::empty(0)],
            shortcuts: Vec::new(),
            shortcut_generations: Vec::new(),
            node_levels: Vec::new(),
            entry_point: 0,
            max_degree,
        };
    }

    let n_fine = codebook.n_fine;
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n_docs];
    let mut inserted: Vec<bool> = vec![false; n_docs];
    let mut score_cache = ScoreCache::new(n_docs * 16);

    let n_clusters = postings.lists.len();

    // Process each cluster's member documents (Algorithm 1)
    for cluster_id in 0..n_clusters {
        let members = &postings.lists[cluster_id];
        if members.is_empty() {
            continue;
        }

        // Collect already-inserted members as potential entry points for this cluster
        let cluster_entries: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&m| inserted[m as usize])
            .collect();

        for &doc_u32 in members {
            let doc_idx = doc_u32 as usize;
            if doc_idx >= n_docs {
                continue;
            }

            if !inserted[doc_idx] {
                // First insertion: connect to existing cluster members
                build_cluster_local(
                    doc_idx,
                    &cluster_entries,
                    &mut adjacency,
                    all_vectors,
                    dim,
                    doc_offsets,
                    codebook,
                    flat_codes,
                    n_fine,
                    max_degree,
                    ef_construction,
                    use_emd,
                    &mut score_cache,
                );
                inserted[doc_idx] = true;
            } else {
                // Already inserted from another cluster: update bridges
                let new_neighbors = find_new_cluster_neighbors(
                    doc_idx,
                    &cluster_entries,
                    &adjacency,
                    all_vectors,
                    dim,
                    doc_offsets,
                    codebook,
                    flat_codes,
                    n_fine,
                    ef_construction,
                );
                update_bridges_cached(
                    doc_idx,
                    &new_neighbors,
                    &mut adjacency,
                    doc_profiles,
                    max_degree,
                    codebook,
                    flat_codes,
                    use_emd,
                    Some(&mut score_cache),
                );
            }
        }
    }

    // Repair isolated nodes (degree 0): connect to nearest neighbor by brute-force.
    for doc_idx in 0..n_docs {
        if !adjacency[doc_idx].is_empty() {
            continue;
        }
        let doc_codes = flat_codes.doc_codes(doc_idx);
        if doc_codes.is_empty() {
            continue;
        }
        let mut best_nbr = u32::MAX;
        let mut best_dist = f32::MAX;
        #[allow(clippy::needless_range_loop)]
        for other in 0..n_docs {
            if other == doc_idx || adjacency[other].is_empty() {
                continue;
            }
            let other_codes = flat_codes.doc_codes(other);
            let d = construction_distance(codebook, doc_codes, other_codes, use_emd);
            if d < best_dist {
                best_dist = d;
                best_nbr = other as u32;
            }
        }
        if best_nbr != u32::MAX {
            adjacency[doc_idx].push(best_nbr);
            adjacency[best_nbr as usize].push(doc_idx as u32);
            if adjacency[best_nbr as usize].len() > max_degree {
                shrink_neighbors_emd_cached(best_nbr as usize, max_degree, &mut adjacency, codebook, flat_codes, use_emd, Some(&mut score_cache));
            }
        }
    }

    // Safety net: bridge repair for any remaining disconnected components
    bridge_repair_emd(&mut adjacency, max_degree, postings, codebook, flat_codes, use_emd);
    global_bridge_repair(&mut adjacency, max_degree, codebook, flat_codes, use_emd);

    // Build single-level graph (dual-graph uses flat structure, not multi-level HNSW)
    let csr = CsrAdjacency::from_adj_lists(&adjacency);

    // Entry point: the node with the most connections (hub-like)
    let entry_point = (0..n_docs)
        .max_by_key(|&i| adjacency[i].len())
        .unwrap_or(0) as u32;

    GemGraph {
        levels: vec![csr],
        shortcuts: vec![Vec::new(); n_docs],
        shortcut_generations: vec![Vec::new(); n_docs],
        node_levels: vec![0; n_docs],
        entry_point,
        max_degree,
    }
}

/// First insertion of a document into the graph via a specific cluster.
/// Beam-searches among already-connected cluster members to find candidates,
/// then selects neighbors via the diversity heuristic.
fn build_cluster_local(
    doc_idx: usize,
    cluster_entries: &[u32],
    adjacency: &mut [Vec<u32>],
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    n_fine: usize,
    max_degree: usize,
    ef_construction: usize,
    use_emd: bool,
    score_cache: &mut ScoreCache,
) {
    if cluster_entries.is_empty() {
        return;
    }

    let (start, end) = doc_offsets[doc_idx];
    let n_tokens = end - start;
    if n_tokens == 0 || dim == 0 {
        return;
    }
    let doc_vecs = &all_vectors[start * dim..end * dim];
    let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);

    let adj = VecAdjacency(adjacency);
    let candidates = beam_search_construction(
        &adj,
        cluster_entries,
        &query_scores,
        n_tokens,
        flat_codes,
        n_fine,
        ef_construction,
        doc_idx,
    );

    let doc_codes = flat_codes.doc_codes(doc_idx);
    let neighbors = select_neighbors_heuristic_cached(
        &candidates,
        doc_codes,
        max_degree,
        codebook,
        flat_codes,
        Some(score_cache),
        use_emd,
    );

    for &(nbr_idx, _) in &neighbors {
        adjacency[doc_idx].push(nbr_idx);
        adjacency[nbr_idx as usize].push(doc_idx as u32);

        if adjacency[nbr_idx as usize].len() > max_degree {
            shrink_neighbors_emd_cached(
                nbr_idx as usize, max_degree, adjacency, codebook, flat_codes, use_emd,
                Some(score_cache),
            );
        }
    }
}

/// Find new candidate neighbors for a doc when encountering it in a second cluster.
fn find_new_cluster_neighbors(
    doc_idx: usize,
    cluster_entries: &[u32],
    adjacency: &[Vec<u32>],
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    n_fine: usize,
    ef_construction: usize,
) -> Vec<(u32, f32)> {
    if cluster_entries.is_empty() {
        return Vec::new();
    }

    let (start, end) = doc_offsets[doc_idx];
    let n_tokens = end - start;
    if n_tokens == 0 || dim == 0 {
        return Vec::new();
    }
    let doc_vecs = &all_vectors[start * dim..end * dim];
    let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);

    let adj = VecAdjacency(adjacency);
    beam_search_construction(
        &adj,
        cluster_entries,
        &query_scores,
        n_tokens,
        flat_codes,
        n_fine,
        ef_construction,
        doc_idx,
    )
}

/// Update bridge edges for a document encountered in a second (or subsequent) cluster.
///
/// GEM Paper Algorithm 3:
/// 1. C_old = existing neighbors
/// 2. C_all = C_new ∪ C_old
/// 3. If |C_all| <= M: keep all
/// 4. Else: keep M closest by construction distance, then for each cluster in ctop(P):
///    if no neighbor from that cluster is in the final set, force-include the closest
///    from C_all that belongs to that cluster (replacing the farthest).
pub fn update_bridges(
    doc_idx: usize,
    new_neighbors: &[(u32, f32)],
    adjacency: &mut [Vec<u32>],
    doc_profiles: &[DocProfile],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) {
    update_bridges_cached(doc_idx, new_neighbors, adjacency, doc_profiles, max_degree, codebook, flat_codes, use_emd, None)
}

pub fn update_bridges_cached(
    doc_idx: usize,
    new_neighbors: &[(u32, f32)],
    adjacency: &mut [Vec<u32>],
    doc_profiles: &[DocProfile],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
    mut cache: Option<&mut ScoreCache>,
) {
    let doc_codes = flat_codes.doc_codes(doc_idx);
    let doc_u32 = doc_idx as u32;

    let old_neighbors: Vec<u32> = adjacency[doc_idx].clone();
    let mut all_candidates: Vec<(u32, f32)> = Vec::new();

    for &nbr in &old_neighbors {
        let dist = if let Some(ref mut c) = cache {
            if let Some(d) = c.get(doc_u32, nbr) {
                d
            } else {
                let nbr_codes = flat_codes.doc_codes(nbr as usize);
                let d = construction_distance(codebook, doc_codes, nbr_codes, use_emd);
                c.insert(doc_u32, nbr, d);
                d
            }
        } else {
            let nbr_codes = flat_codes.doc_codes(nbr as usize);
            construction_distance(codebook, doc_codes, nbr_codes, use_emd)
        };
        all_candidates.push((nbr, dist));
    }

    for &(nbr, _) in new_neighbors {
        if nbr as usize != doc_idx && !old_neighbors.contains(&nbr) {
            let dist = if let Some(ref mut c) = cache {
                if let Some(d) = c.get(doc_u32, nbr) {
                    d
                } else {
                    let nbr_codes = flat_codes.doc_codes(nbr as usize);
                    let d = construction_distance(codebook, doc_codes, nbr_codes, use_emd);
                    c.insert(doc_u32, nbr, d);
                    d
                }
            } else {
                let nbr_codes = flat_codes.doc_codes(nbr as usize);
                construction_distance(codebook, doc_codes, nbr_codes, use_emd)
            };
            all_candidates.push((nbr, dist));
        }
    }

    if all_candidates.len() <= max_degree {
        for &old_nbr in &old_neighbors {
            adjacency[old_nbr as usize].retain(|&x| x != doc_idx as u32);
        }
        adjacency[doc_idx] = all_candidates.iter().map(|&(id, _)| id).collect();
        for &(nbr, _) in &all_candidates {
            if !adjacency[nbr as usize].contains(&(doc_idx as u32)) {
                adjacency[nbr as usize].push(doc_idx as u32);
                if adjacency[nbr as usize].len() > max_degree {
                    shrink_neighbors_emd_cached(nbr as usize, max_degree, adjacency, codebook, flat_codes, use_emd, cache.as_deref_mut());
                }
            }
        }
        return;
    }

    // Step 3: Select top-M by distance
    all_candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let mut selected: Vec<(u32, f32)> = all_candidates.iter().take(max_degree).cloned().collect();

    // Step 4: Enforce cross-cluster constraint (iterated to convergence)
    // Iterate up to |ctop| times since replacing for cluster A can evict the only
    // neighbor covering cluster B.
    if doc_idx < doc_profiles.len() {
        let ctop = &doc_profiles[doc_idx].ctop;
        let max_passes = ctop.len();

        for _ in 0..max_passes {
            let mut any_replaced = false;
            for &cluster_id in ctop {
                let has_cluster_neighbor = selected.iter().any(|&(nbr, _)| {
                    let nbr_idx = nbr as usize;
                    nbr_idx < doc_profiles.len()
                        && doc_profiles[nbr_idx].ctop.contains(&cluster_id)
                });

                if !has_cluster_neighbor {
                    if let Some(&(bridge_nbr, bridge_dist)) = all_candidates.iter().find(|&&(nbr, _)| {
                        let ni = nbr as usize;
                        ni != doc_idx
                            && ni < doc_profiles.len()
                            && doc_profiles[ni].ctop.contains(&cluster_id)
                    }) {
                        if let Some(farthest_idx) = selected
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
                            .map(|(i, _)| i)
                        {
                            selected[farthest_idx] = (bridge_nbr, bridge_dist);
                            any_replaced = true;
                        }
                    }
                }
            }
            if !any_replaced {
                break;
            }
        }
    }

    // Apply: remove old reverse edges, set new ones
    for &old_nbr in &old_neighbors {
        adjacency[old_nbr as usize].retain(|&x| x != doc_idx as u32);
    }
    adjacency[doc_idx] = selected.iter().map(|&(id, _)| id).collect();
    for &(nbr, _) in &selected {
        if !adjacency[nbr as usize].contains(&(doc_idx as u32)) {
            adjacency[nbr as usize].push(doc_idx as u32);
            if adjacency[nbr as usize].len() > max_degree {
                evict_worst_neighbor_emd(nbr as usize, max_degree + 1, adjacency, codebook, flat_codes, use_emd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop};
    use latence_gem_router::router::{FlatDocCodes, DocProfile, ClusterPostings};

    fn setup_test_data(n_docs: usize, tokens_per_doc: usize) -> (
        Vec<f32>, usize, Vec<(usize, usize)>, TwoStageCodebook,
        FlatDocCodes, Vec<DocProfile>, ClusterPostings, Vec<u64>,
    ) {
        let dim = 8;
        let n_fine = 8;
        let n_coarse = 4;
        let ctop_r = 2;
        let mut rng = StdRng::seed_from_u64(42);
        let n_vectors = n_docs * tokens_per_doc;
        let all_vectors: Vec<f32> = (0..n_vectors * dim)
            .map(|_| rng.gen::<f32>() - 0.5)
            .collect();
        let doc_offsets: Vec<(usize, usize)> = (0..n_docs)
            .map(|i| (i * tokens_per_doc, (i + 1) * tokens_per_doc))
            .collect();
        let doc_ids: Vec<u64> = (0..n_docs as u64).collect();

        let codebook = TwoStageCodebook::build(
            &all_vectors, n_vectors, dim, n_fine, n_coarse, 10, 42,
        );
        let all_assignments = codebook.assign_vectors(&all_vectors, n_vectors);

        let mut doc_centroid_sets = Vec::with_capacity(n_docs);
        for &(start, end) in &doc_offsets {
            doc_centroid_sets.push(all_assignments[start..end].to_vec());
        }

        let mut flat_codes = FlatDocCodes::new();
        let mut doc_profiles = Vec::with_capacity(n_docs);
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

        (all_vectors, dim, doc_offsets, codebook, flat_codes, doc_profiles, postings, doc_ids)
    }

    #[test]
    fn test_build_graph_basic() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(20, 5);
        let graph = build_graph(&vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32);
        assert_eq!(graph.n_nodes(), 20);
        assert!(graph.n_edges() > 0, "graph should have edges");
        assert!(!graph.levels.is_empty());
        for i in 0..20 {
            let nbrs = graph.levels[0].neighbors(i);
            assert!(nbrs.len() <= 8, "node {} has {} neighbors > max_degree 8", i, nbrs.len());
        }
    }

    #[test]
    fn test_build_graph_single_doc() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(1, 5);
        let graph = build_graph(&vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32);
        assert_eq!(graph.n_nodes(), 1);
    }

    #[test]
    fn test_build_graph_with_payload() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(20, 5);
        let payload: Vec<u32> = (0..20).map(|i| (i % 3) as u32).collect();
        let graph = build_graph_with_payload(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32,
            Some(&payload), false,
        );
        assert_eq!(graph.n_nodes(), 20);
        assert!(graph.n_edges() > 0);
    }

    #[test]
    fn test_build_graph_dual_basic() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(20, 5);
        let graph = build_graph_dual(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32, false,
        );
        assert_eq!(graph.n_nodes(), 20);
        assert!(graph.n_edges() > 0, "dual graph should have edges");
        for i in 0..20 {
            let nbrs = graph.levels[0].neighbors(i);
            assert!(nbrs.len() <= 8, "node {} has {} neighbors > max_degree 8", i, nbrs.len());
        }
    }

    #[test]
    fn test_build_graph_dual_with_emd() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(10, 4);
        let graph = build_graph_dual(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 6, 16, true,
        );
        assert_eq!(graph.n_nodes(), 10);
        assert!(graph.n_edges() > 0);
    }

    #[test]
    fn test_heuristic_emd_vs_ch() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(15, 5);
        let graph_ch = build_graph_with_payload(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32, None, false,
        );
        let graph_emd = build_graph_with_payload(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32, None, true,
        );
        assert_eq!(graph_ch.n_nodes(), graph_emd.n_nodes());
        assert!(graph_ch.n_edges() > 0);
        assert!(graph_emd.n_edges() > 0);
    }

    #[test]
    fn test_update_bridges_degree_limit() {
        let (_vecs, _dim, _offsets, cb, fc, dp, _post, _) = setup_test_data(20, 5);
        let max_degree = 4;
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); 20];
        adjacency[0] = vec![1, 2, 3, 4];
        adjacency[1] = vec![0];
        adjacency[2] = vec![0];
        adjacency[3] = vec![0];
        adjacency[4] = vec![0];

        let new_neighbors: Vec<(u32, f32)> = vec![(5, 0.5), (6, 0.6), (7, 0.7)];
        update_bridges(0, &new_neighbors, &mut adjacency, &dp, max_degree, &cb, &fc, false);

        assert!(
            adjacency[0].len() <= max_degree,
            "node 0 has {} neighbors > max_degree {}", adjacency[0].len(), max_degree
        );
    }

    #[test]
    fn test_csr_roundtrip() {
        let adj = vec![
            vec![1u32, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let csr = CsrAdjacency::from_adj_lists(&adj);
        assert_eq!(csr.n_nodes(), 3);
        assert_eq!(csr.neighbors(0), &[1, 2]);
        assert_eq!(csr.neighbors(1), &[0, 2]);
        assert_eq!(csr.neighbors(2), &[0, 1]);
        let restored = csr.to_adj_lists();
        assert_eq!(restored, adj);
    }
}
