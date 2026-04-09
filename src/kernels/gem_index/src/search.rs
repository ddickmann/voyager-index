use std::collections::BinaryHeap;
use std::cmp::Ordering;

use latence_gem_router::codebook::qch_proxy_score_u16;
use latence_gem_router::router::FlatDocCodes;

use crate::graph::Adjacency;
use crate::visited::with_visited;

/// Telemetry for a single search operation.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub nodes_visited: u32,
    pub distance_computations: u32,
}

#[derive(Clone, Copy)]
struct MinCand {
    score: f32,
    idx: u32,
}

impl Eq for MinCand {}
impl PartialEq for MinCand {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}
impl PartialOrd for MinCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinCand {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.total_cmp(&self.score)
            .then(other.idx.cmp(&self.idx))
    }
}

#[derive(Clone, Copy)]
struct MaxCand {
    score: f32,
    idx: u32,
}

impl Eq for MaxCand {}
impl PartialEq for MaxCand {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}
impl PartialOrd for MaxCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxCand {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
            .then(self.idx.cmp(&other.idx))
    }
}

#[inline(always)]
fn prefetch_doc_codes(flat_codes: &FlatDocCodes, node: usize) {
    if node < flat_codes.offsets.len() {
        let off = flat_codes.offsets[node] as usize;
        let ptr = unsafe { flat_codes.codes.as_ptr().add(off) };
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_READ, std::arch::aarch64::_PREFETCH_LOCALITY3);
        }
    }
}

/// HNSW-style beam search on the GEM graph. Generic over adjacency format
/// (CSR for sealed segments, Vec<Vec<u32>> for mutable segments).
///
/// Returns candidates sorted ascending by score (best-first).
pub fn beam_search<A: Adjacency>(
    adjacency: &A,
    shortcuts: Option<&[Vec<u32>]>,
    entry_points: &[u32],
    query_scores: &[f32],
    n_query: usize,
    flat_codes: &FlatDocCodes,
    n_fine: usize,
    ef: usize,
    deleted: Option<&[bool]>,
    enable_shortcuts: bool,
) -> Vec<(u32, f32)> {
    let n_nodes = adjacency.n_nodes();
    if n_nodes == 0 || entry_points.is_empty() {
        return Vec::new();
    }

    with_visited(n_nodes, |visited| {
        let mut candidates: BinaryHeap<MinCand> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxCand> = BinaryHeap::new();

        for &ep in entry_points {
            let ep_usize = ep as usize;
            if ep_usize >= n_nodes || visited.contains(ep_usize) {
                continue;
            }
            visited.set(ep_usize);

            let is_del = deleted.is_some_and(|d| ep_usize < d.len() && d[ep_usize]);
            let doc_codes = flat_codes.doc_codes(ep_usize);
            let score = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);

            candidates.push(MinCand { score, idx: ep });
            if !is_del {
                results.push(MaxCand { score, idx: ep });
            }
        }

        while let Some(MinCand { score: c_dist, idx: c_node }) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if c_dist > worst.score {
                        break;
                    }
                }
            }

            let c_usize = c_node as usize;
            let neighbors = adjacency.neighbors(c_usize);

            for (i, &nbr) in neighbors.iter().enumerate() {
                if i + 1 < neighbors.len() {
                    let next_nbr = neighbors[i + 1] as usize;
                    if next_nbr < n_nodes && !visited.contains(next_nbr) {
                        prefetch_doc_codes(flat_codes, next_nbr);
                    }
                }

                let nbr_usize = nbr as usize;
                if nbr_usize >= n_nodes || visited.contains(nbr_usize) {
                    continue;
                }
                visited.set(nbr_usize);

                let doc_codes = flat_codes.doc_codes(nbr_usize);
                let dist = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);

                let should_add = results.len() < ef || {
                    results.peek().is_none_or(|w| dist < w.score)
                };

                if should_add {
                    candidates.push(MinCand { score: dist, idx: nbr });
                    let is_del = deleted.is_some_and(|d| nbr_usize < d.len() && d[nbr_usize]);
                    if !is_del {
                        results.push(MaxCand { score: dist, idx: nbr });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }

            if enable_shortcuts {
                if let Some(sc) = shortcuts {
                    if c_usize < sc.len() {
                        let sc_neighbors = &sc[c_usize];
                        for (i, &nbr) in sc_neighbors.iter().enumerate() {
                            if i + 1 < sc_neighbors.len() {
                                let next_nbr = sc_neighbors[i + 1] as usize;
                                if next_nbr < n_nodes && !visited.contains(next_nbr) {
                                    prefetch_doc_codes(flat_codes, next_nbr);
                                }
                            }

                            let nbr_usize = nbr as usize;
                            if nbr_usize >= n_nodes || visited.contains(nbr_usize) {
                                continue;
                            }
                            visited.set(nbr_usize);

                            let doc_codes = flat_codes.doc_codes(nbr_usize);
                            let dist = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);

                            let should_add = results.len() < ef || {
                                results.peek().is_none_or(|w| dist < w.score)
                            };

                            if should_add {
                                candidates.push(MinCand { score: dist, idx: nbr });
                                let is_del = deleted.is_some_and(|d| nbr_usize < d.len() && d[nbr_usize]);
                                if !is_del {
                                    results.push(MaxCand { score: dist, idx: nbr });
                                    if results.len() > ef {
                                        results.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = results
            .into_iter()
            .map(|mc| (mc.idx, mc.score))
            .collect();
        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        out
    })
}

/// Identical to [`beam_search`] but also returns [`SearchStats`] tracking
/// nodes visited and distance computations.
pub fn beam_search_with_stats<A: Adjacency>(
    adjacency: &A,
    shortcuts: Option<&[Vec<u32>]>,
    entry_points: &[u32],
    query_scores: &[f32],
    n_query: usize,
    flat_codes: &FlatDocCodes,
    n_fine: usize,
    ef: usize,
    deleted: Option<&[bool]>,
    enable_shortcuts: bool,
) -> (Vec<(u32, f32)>, SearchStats) {
    let n_nodes = adjacency.n_nodes();
    if n_nodes == 0 || entry_points.is_empty() {
        return (Vec::new(), SearchStats::default());
    }

    with_visited(n_nodes, |visited| {
        let mut stats = SearchStats::default();
        let mut candidates: BinaryHeap<MinCand> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxCand> = BinaryHeap::new();

        for &ep in entry_points {
            let ep_usize = ep as usize;
            if ep_usize >= n_nodes || visited.contains(ep_usize) {
                continue;
            }
            visited.set(ep_usize);
            stats.nodes_visited += 1;

            let is_del = deleted.is_some_and(|d| ep_usize < d.len() && d[ep_usize]);
            let doc_codes = flat_codes.doc_codes(ep_usize);
            let score = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);
            stats.distance_computations += 1;

            candidates.push(MinCand { score, idx: ep });
            if !is_del {
                results.push(MaxCand { score, idx: ep });
            }
        }

        while let Some(MinCand { score: c_dist, idx: c_node }) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if c_dist > worst.score {
                        break;
                    }
                }
            }

            let c_usize = c_node as usize;
            let neighbors = adjacency.neighbors(c_usize);

            for (i, &nbr) in neighbors.iter().enumerate() {
                if i + 1 < neighbors.len() {
                    let next_nbr = neighbors[i + 1] as usize;
                    if next_nbr < n_nodes && !visited.contains(next_nbr) {
                        prefetch_doc_codes(flat_codes, next_nbr);
                    }
                }

                let nbr_usize = nbr as usize;
                if nbr_usize >= n_nodes || visited.contains(nbr_usize) {
                    continue;
                }
                visited.set(nbr_usize);
                stats.nodes_visited += 1;

                let doc_codes = flat_codes.doc_codes(nbr_usize);
                let dist = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);
                stats.distance_computations += 1;

                let should_add = results.len() < ef || {
                    results.peek().is_none_or(|w| dist < w.score)
                };

                if should_add {
                    candidates.push(MinCand { score: dist, idx: nbr });
                    let is_del = deleted.is_some_and(|d| nbr_usize < d.len() && d[nbr_usize]);
                    if !is_del {
                        results.push(MaxCand { score: dist, idx: nbr });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }

            if enable_shortcuts {
                if let Some(sc) = shortcuts {
                    if c_usize < sc.len() {
                        let sc_neighbors = &sc[c_usize];
                        for (i, &nbr) in sc_neighbors.iter().enumerate() {
                            if i + 1 < sc_neighbors.len() {
                                let next_nbr = sc_neighbors[i + 1] as usize;
                                if next_nbr < n_nodes && !visited.contains(next_nbr) {
                                    prefetch_doc_codes(flat_codes, next_nbr);
                                }
                            }

                            let nbr_usize = nbr as usize;
                            if nbr_usize >= n_nodes || visited.contains(nbr_usize) {
                                continue;
                            }
                            visited.set(nbr_usize);
                            stats.nodes_visited += 1;

                            let doc_codes = flat_codes.doc_codes(nbr_usize);
                            let dist = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);
                            stats.distance_computations += 1;

                            let should_add = results.len() < ef || {
                                results.peek().is_none_or(|w| dist < w.score)
                            };

                            if should_add {
                                candidates.push(MinCand { score: dist, idx: nbr });
                                let is_del = deleted.is_some_and(|d| nbr_usize < d.len() && d[nbr_usize]);
                                if !is_del {
                                    results.push(MaxCand { score: dist, idx: nbr });
                                    if results.len() > ef {
                                        results.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = results
            .into_iter()
            .map(|mc| (mc.idx, mc.score))
            .collect();
        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        (out, stats)
    })
}

/// Beam search for graph construction (no delete mask, no shortcuts).
pub fn beam_search_construction<A: Adjacency>(
    adjacency: &A,
    entry_points: &[u32],
    query_scores: &[f32],
    n_query: usize,
    flat_codes: &FlatDocCodes,
    n_fine: usize,
    ef: usize,
    n_built: usize,
) -> Vec<(u32, f32)> {
    if n_built == 0 || entry_points.is_empty() {
        return Vec::new();
    }

    with_visited(n_built, |visited| {
        let mut candidates: BinaryHeap<MinCand> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxCand> = BinaryHeap::new();

        for &ep in entry_points {
            let ep_usize = ep as usize;
            if ep_usize >= n_built || visited.contains(ep_usize) {
                continue;
            }
            visited.set(ep_usize);

            let doc_codes = flat_codes.doc_codes(ep_usize);
            let score = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);
            candidates.push(MinCand { score, idx: ep });
            results.push(MaxCand { score, idx: ep });
        }

        while let Some(MinCand { score: c_dist, idx: c_node }) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if c_dist > worst.score {
                        break;
                    }
                }
            }

            let c_usize = c_node as usize;
            let neighbors = adjacency.neighbors(c_usize);

            for (i, &nbr) in neighbors.iter().enumerate() {
                if i + 1 < neighbors.len() {
                    let next_nbr = neighbors[i + 1] as usize;
                    if next_nbr < n_built && !visited.contains(next_nbr) {
                        prefetch_doc_codes(flat_codes, next_nbr);
                    }
                }

                let nbr_usize = nbr as usize;
                if nbr_usize >= n_built || visited.contains(nbr_usize) {
                    continue;
                }
                visited.set(nbr_usize);

                let doc_codes = flat_codes.doc_codes(nbr_usize);
                let dist = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);

                let should_add = results.len() < ef || {
                    results.peek().is_none_or(|w| dist < w.score)
                };

                if should_add {
                    candidates.push(MinCand { score: dist, idx: nbr });
                    results.push(MaxCand { score: dist, idx: nbr });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = results
            .into_iter()
            .map(|mc| (mc.idx, mc.score))
            .collect();
        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        out
    })
}
