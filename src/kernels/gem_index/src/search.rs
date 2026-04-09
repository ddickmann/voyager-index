use std::collections::BinaryHeap;
use std::cmp::Ordering;

use latence_gem_router::codebook::qch_proxy_score_u16;
use latence_gem_router::router::FlatDocCodes;

use crate::visited::VisitedSet;

/// Candidate for min-heap: smallest score (best match) has highest priority.
#[derive(Clone)]
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
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then(other.idx.cmp(&self.idx))
    }
}

/// Candidate for max-heap: largest score (worst match) has highest priority.
#[derive(Clone)]
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
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then(self.idx.cmp(&other.idx))
    }
}

/// HNSW-style beam search on the GEM graph.
///
/// Returns candidates sorted ascending by score (best-first).
pub fn beam_search(
    adjacency: &[Vec<u32>],
    shortcuts: &[Vec<u32>],
    entry_points: &[u32],
    query_scores: &[f32],
    n_query: usize,
    flat_codes: &FlatDocCodes,
    n_fine: usize,
    ef: usize,
    deleted: Option<&[bool]>,
    enable_shortcuts: bool,
) -> Vec<(u32, f32)> {
    let n_nodes = adjacency.len();
    if n_nodes == 0 || entry_points.is_empty() {
        return Vec::new();
    }

    let mut visited = VisitedSet::new(n_nodes);
    let mut candidates: BinaryHeap<MinCand> = BinaryHeap::new();
    let mut results: BinaryHeap<MaxCand> = BinaryHeap::new();

    for &ep in entry_points {
        let ep_usize = ep as usize;
        if ep_usize >= n_nodes || visited.contains(ep_usize) {
            continue;
        }
        visited.set(ep_usize);

        let is_del = deleted.map_or(false, |d| ep_usize < d.len() && d[ep_usize]);
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

        let neighbor_iter: Box<dyn Iterator<Item = u32>> = if enable_shortcuts && c_usize < shortcuts.len() && !shortcuts[c_usize].is_empty() {
            Box::new(
                adjacency[c_usize].iter().copied()
                    .chain(shortcuts[c_usize].iter().copied())
            )
        } else {
            Box::new(adjacency[c_usize].iter().copied())
        };

        for nbr in neighbor_iter {
            let nbr_usize = nbr as usize;
            if nbr_usize >= n_nodes || visited.contains(nbr_usize) {
                continue;
            }
            visited.set(nbr_usize);

            let doc_codes = flat_codes.doc_codes(nbr_usize);
            let dist = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);

            let should_add = results.len() < ef || {
                results.peek().map_or(true, |w| dist < w.score)
            };

            if should_add {
                candidates.push(MinCand { score: dist, idx: nbr });
                let is_del = deleted.map_or(false, |d| nbr_usize < d.len() && d[nbr_usize]);
                if !is_del {
                    results.push(MaxCand { score: dist, idx: nbr });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
    }

    let mut out: Vec<(u32, f32)> = results
        .into_iter()
        .map(|mc| (mc.idx, mc.score))
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    out
}

/// Beam search for graph construction (no delete mask, no shortcuts).
pub fn beam_search_construction(
    adjacency: &[Vec<u32>],
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

    let mut visited = VisitedSet::new(n_built);
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
        for &nbr in &adjacency[c_usize] {
            let nbr_usize = nbr as usize;
            if nbr_usize >= n_built || visited.contains(nbr_usize) {
                continue;
            }
            visited.set(nbr_usize);

            let doc_codes = flat_codes.doc_codes(nbr_usize);
            let dist = qch_proxy_score_u16(query_scores, n_query, n_fine, doc_codes);

            let should_add = results.len() < ef || {
                results.peek().map_or(true, |w| dist < w.score)
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
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    out
}
