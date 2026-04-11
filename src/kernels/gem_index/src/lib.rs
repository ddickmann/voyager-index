#![allow(clippy::useless_conversion, clippy::too_many_arguments)]

extern crate libc;

pub mod visited;
pub mod id_tracker;
pub mod emd;
pub mod network_simplex;
pub mod graph;
pub mod search;
pub mod persistence;
pub mod mutable;
pub mod score_cache;
pub mod ensemble;

use std::io::Write;
use std::time::Instant;

fn _progress_write(msg: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open("/tmp/gem_build_progress.log")
    {
        let _ = writeln!(f, "{}", msg);
    }
    let bytes = format!("{}\n", msg);
    unsafe {
        libc::write(2, bytes.as_ptr() as *const libc::c_void, bytes.len());
    }
}

macro_rules! progress {
    ($($arg:tt)*) => {{
        _progress_write(&format!($($arg)*));
    }};
}

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyArray1, PyArray2, PyUntypedArrayMethods};

use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop, compute_ctop_adaptive, l2_normalize_rows_inplace};
use latence_gem_router::adaptive_cutoff::CutoffTree;
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes, FilterIndex};

use graph::{Adjacency, GemGraph};
use persistence::{SegmentData, save_segment, load_segment};

use search::{beam_search, beam_search_with_stats};
use mutable::MutableGemSegment;

struct SealedInner {
    graph: GemGraph,
    codebook: TwoStageCodebook,
    flat_codes: FlatDocCodes,
    doc_ids: Vec<u64>,
    doc_profiles: Vec<DocProfile>,
    postings: ClusterPostings,
    ctop_r: usize,
    dim: usize,
    cutoff_tree: Option<CutoffTree>,
    filter_index: Option<FilterIndex>,
    raw_vectors: Option<Vec<f32>>,
    doc_offsets: Vec<(usize, usize)>,
}

/// Compute MaxSim(Q, D) = Σ_i max_j dot(q_i, d_j) using BLAS sgemm.
/// Returns NEGATIVE MaxSim so that lower = better (consistent with qCH proxy).
///
/// Computes the full Q × D^T matrix via sgemm, then takes max per query row.
fn maxsim_score(query: &[f32], doc: &[f32], dim: usize) -> f32 {
    let n_q = query.len() / dim;
    let n_d = doc.len() / dim;
    if n_q == 0 || n_d == 0 || dim == 0 {
        return 0.0;
    }

    // scores = Q (n_q × dim) × D^T (dim × n_d) → (n_q × n_d)
    let mut scores = vec![0.0f32; n_q * n_d];
    unsafe {
        matrixmultiply::sgemm(
            n_q, dim, n_d,
            1.0,
            query.as_ptr(), dim as isize, 1,
            doc.as_ptr(), 1, dim as isize,
            0.0,
            scores.as_mut_ptr(), n_d as isize, 1,
        );
    }

    let mut total = 0.0f32;
    for qi in 0..n_q {
        let row = &scores[qi * n_d..(qi + 1) * n_d];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val > f32::NEG_INFINITY {
            total += max_val;
        }
    }
    -total
}

/// Rerank beam search candidates using exact MaxSim on raw vectors.
fn rerank_by_maxsim(
    candidates: Vec<(u32, f32)>,
    query_flat: &[f32],
    raw_vectors: &[f32],
    doc_offsets: &[(usize, usize)],
    dim: usize,
    k: usize,
) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = candidates
        .into_iter()
        .map(|(int_idx, _proxy_score)| {
            let idx = int_idx as usize;
            if idx < doc_offsets.len() {
                let (start, end) = doc_offsets[idx];
                let doc_vecs = &raw_vectors[start * dim..end * dim];
                let ms = maxsim_score(query_flat, doc_vecs, dim);
                (int_idx, ms)
            } else {
                (int_idx, f32::MAX)
            }
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(k);
    scored
}

/// Sealed GEM segment for multi-vector retrieval.
#[pyclass]
pub struct GemSegment {
    inner: Option<SealedInner>,
}

#[pymethods]
impl GemSegment {
    #[new]
    fn new() -> Self {
        Self { inner: None }
    }

    /// Build a sealed GEM segment from document vectors.
    ///
    /// Args:
    ///   all_vectors: (N, D) float32 matrix of all token vectors
    ///   doc_ids: external document IDs
    ///   doc_offsets: (start, end) index pairs into all_vectors per document
    ///   n_fine: number of fine centroids (codebook size)
    ///   n_coarse: number of coarse clusters
    ///   max_degree: max neighbors per graph node (M parameter)
    ///   ef_construction: beam width during graph construction
    ///   max_kmeans_iter: k-means iteration limit
    ///   ctop_r: number of top coarse clusters per document
    ///   payload_clusters: optional per-doc cluster IDs for payload-aware construction
    ///   use_emd: use qEMD (Earth Mover's Distance) for graph construction instead of qCH
    ///   dual_graph: use per-cluster dual-graph construction (GEM paper Algorithm 1)
    #[pyo3(signature = (all_vectors, doc_ids, doc_offsets, n_fine = 0, n_coarse = 32, max_degree = 32, ef_construction = 200, max_kmeans_iter = 15, ctop_r = 3, payload_clusters = None, use_emd = true, dual_graph = true, store_raw_vectors = true, refine_graph = false))]
    fn build(
        &mut self,
        py: Python<'_>,
        mut all_vectors: PyReadwriteArray2<f32>,
        doc_ids: Vec<u64>,
        doc_offsets: Vec<(usize, usize)>,
        n_fine: usize,
        n_coarse: usize,
        max_degree: usize,
        ef_construction: usize,
        max_kmeans_iter: usize,
        ctop_r: usize,
        payload_clusters: Option<Vec<u32>>,
        use_emd: bool,
        dual_graph: bool,
        store_raw_vectors: bool,
        refine_graph: bool,
    ) -> PyResult<()> {
        let (n_vectors, dim) = {
            let shape = all_vectors.shape();
            (shape[0], shape[1])
        };

        let n_docs = doc_ids.len();
        if n_docs != doc_offsets.len() {
            return Err(PyValueError::new_err(format!(
                "doc_ids length ({}) != doc_offsets length ({})",
                n_docs, doc_offsets.len()
            )));
        }
        for (i, &(start, end)) in doc_offsets.iter().enumerate() {
            if start > end || end > n_vectors {
                return Err(PyValueError::new_err(format!(
                    "doc_offsets[{}] = ({}, {}) out of range for {} vectors",
                    i, start, end, n_vectors
                )));
            }
        }
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        if n_docs > u32::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "too many documents ({}) — max supported is {}", n_docs, u32::MAX
            )));
        }
        if dual_graph && payload_clusters.is_some() {
            return Err(PyValueError::new_err(
                "dual_graph=True and payload_clusters are mutually exclusive: \
                 dual-graph uses the router's cluster postings, not external payload clusters"
            ));
        }

        // Phase 1: L2-normalize in-place directly on the numpy array (GIL held).
        // This avoids a 6 GB .to_vec() copy — the numpy buffer IS the working buffer.
        {
            let flat_mut = all_vectors.as_slice_mut().map_err(|_| {
                PyValueError::new_err("array must be C-contiguous")
            })?;
            l2_normalize_rows_inplace(flat_mut, n_vectors, dim);
        }

        // Extract raw pointer before releasing GIL. The numpy array stays alive
        // for the duration of allow_threads because Python can't GC it while we
        // hold a reference (the &mut borrow ends, but Python keeps the object).
        let flat_addr = all_vectors.as_slice().map_err(|_| {
            PyValueError::new_err("array must be C-contiguous")
        })?.as_ptr() as usize;
        let flat_len = n_vectors * dim;

        let sealed = py.allow_threads(move || {
            // SAFETY: The numpy array backing this pointer is kept alive by the
            // Python interpreter for the duration of this call. We only read from
            // it after normalization is complete. The pointer is valid and aligned
            // because it came from a C-contiguous numpy float32 array.
            let flat: &[f32] = unsafe { std::slice::from_raw_parts(flat_addr as *const f32, flat_len) };

            let build_start = Instant::now();
            let n_fine = if n_fine == 0 {
                ((n_vectors as f64).sqrt() as usize).clamp(64, 2048)
            } else {
                n_fine
            };
            if n_fine < 128 && n_docs > 500 {
                log::warn!(
                    "n_fine={} may cause poor ranking for {} docs; consider n_fine >= 128",
                    n_fine, n_docs
                );
            }

            progress!("[GEM build] ══════════════════════════════════════════════════");
            progress!("[GEM build] {} docs, {} vectors, dim={}", n_docs, n_vectors, dim);
            progress!("[GEM build] n_fine={}, n_coarse={}, max_degree={}, ef_construction={}",
                n_fine, n_coarse, max_degree, ef_construction);
            progress!("[GEM build] use_emd={}, dual_graph={}, store_raw={}", use_emd, dual_graph, store_raw_vectors);
            progress!("[GEM build] zero-copy vectors (no .to_vec() copy)");
            progress!("[GEM build] ══════════════════════════════════════════════════");

            progress!("[GEM build] Phase 1/5: L2-normalization already done (zero-copy)");

            progress!("[GEM build] Phase 2/5: Building two-stage codebook...");
            let t = Instant::now();
            let mut codebook = TwoStageCodebook::build_prenorm(
                flat, n_vectors, dim, n_fine, n_coarse, max_kmeans_iter, 42,
            );
            progress!("[GEM build] Phase 2/5 done in {:.1}s", t.elapsed().as_secs_f64());

            progress!("[GEM build] Phase 3/5: Assigning vectors + IDF refinement...");
            let t = Instant::now();
            let all_assignments = codebook.assign_vectors_prenorm(flat, n_vectors);

            let mut doc_centroid_sets = Vec::with_capacity(n_docs);
            for &(start, end) in &doc_offsets {
                doc_centroid_sets.push(all_assignments[start..end].to_vec());
            }
            codebook.update_idf(&doc_centroid_sets);

            codebook.refine_centroids_idf_prenorm(flat, n_vectors, 1);
            let all_assignments = codebook.assign_vectors_prenorm(flat, n_vectors);
            let mut doc_centroid_sets = Vec::with_capacity(n_docs);
            for &(start, end) in &doc_offsets {
                doc_centroid_sets.push(all_assignments[start..end].to_vec());
            }
            codebook.update_idf(&doc_centroid_sets);
            progress!("[GEM build] Phase 3/5 done in {:.1}s", t.elapsed().as_secs_f64());

            progress!("[GEM build] Phase 4/5: Building doc profiles + postings...");
            let t = Instant::now();
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
            progress!("[GEM build] Phase 4/5 done in {:.1}s", t.elapsed().as_secs_f64());

            progress!("[GEM build] Phase 5/5: Graph construction ({})...",
                if dual_graph { "nndescent" } else { "payload-graph" });
            let t = Instant::now();
            let gem_graph = if dual_graph {
                let refine = if refine_graph {
                    Some((flat, dim, &doc_offsets[..], ef_construction))
                } else {
                    None
                };
                graph::build_graph_nndescent(
                    n_docs, &codebook, &flat_codes,
                    &doc_profiles, &postings, max_degree, use_emd, refine,
                )
            } else {
                graph::build_graph_with_payload(
                    flat, dim, &doc_offsets, &codebook, &flat_codes,
                    &doc_profiles, &postings, max_degree, ef_construction,
                    payload_clusters.as_deref(), use_emd,
                )
            };
            progress!("[GEM build] Phase 5/5 done in {:.1}s", t.elapsed().as_secs_f64());
            progress!("[GEM build] ══════════════════════════════════════════════════");
            progress!("[GEM build] TOTAL BUILD TIME: {:.1}s ({:.1} min)",
                build_start.elapsed().as_secs_f64(), build_start.elapsed().as_secs_f64() / 60.0);
            progress!("[GEM build] ══════════════════════════════════════════════════");

            // Only copy vectors into the segment if raw storage is requested.
            // This is the segment's permanent storage — unavoidable copy, but
            // it happens AFTER all the heavy codebook/graph work is done.
            let raw_vectors = if store_raw_vectors {
                Some(flat.to_vec())
            } else {
                None
            };

            SealedInner {
                graph: gem_graph,
                codebook,
                flat_codes,
                doc_ids,
                doc_profiles,
                postings,
                ctop_r,
                dim,
                cutoff_tree: None,
                filter_index: None,
                raw_vectors,
                doc_offsets,
            }
        });

        self.inner = Some(sealed);
        Ok(())
    }

    /// Search the sealed segment for nearest neighbors to a multi-vector query.
    ///
    /// Returns: list of (doc_id, qch_proxy_score) tuples, sorted best-first.
    /// filter: optional list of (field, value) pairs for filter-aware routing.
    /// min_cluster_ratio: when filtering, skip clusters where fewer than this
    ///   fraction of docs match (selectivity-aware pruning).
    #[pyo3(signature = (query_vectors, k = 10, ef = 100, n_probes = 4, enable_shortcuts = false, filter = None, min_cluster_ratio = 0.01))]
    fn search(
        &self,
        py: Python<'_>,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        n_probes: usize,
        enable_shortcuts: bool,
        filter: Option<Vec<(String, String)>>,
        min_cluster_ratio: f32,
    ) -> PyResult<Vec<(u64, f32)>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built; call build() or load() first")
        })?;
        if inner.graph.levels.is_empty() {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let ef = ef.max(k);

        let arr = query_vectors.as_array();
        let (n_query, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != inner.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", inner.dim, dim
            )));
        }
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        let codebook = &inner.codebook;
        let flat_codes = &inner.flat_codes;
        let postings = &inner.postings;
        let graph = &inner.graph;
        let doc_ids = &inner.doc_ids;
        let inner_ctop_r = inner.ctop_r;
        let cutoff_tree = inner.cutoff_tree.as_ref();
        let filter_index = inner.filter_index.as_ref();
        let n_docs = doc_ids.len();

        let graph_entry = inner.graph.entry_point;

        // Build filter mask if filter is provided
        let filter_mask: Option<Vec<bool>> = filter.as_ref().and_then(|f| {
            if f.is_empty() { return None; }
            match filter_index {
                Some(fi) => {
                    let mask = fi.build_filter_mask(f, n_docs);
                    Some(mask.iter().map(|&m| !m).collect())
                }
                None => {
                    log::warn!("filter provided but no filter_index built; call set_doc_payloads() first");
                    None
                }
            }
        });

        let out = py.allow_threads(|| {
            let mut query_scores = codebook.compute_query_centroid_scores(&flat, n_query);
            codebook.apply_idf_weights(&mut query_scores, n_query);
            let n_fine = codebook.n_fine;

            // Multi-level search: greedy descend through upper layers
            let mut cur_entry = graph_entry;
            let n_levels = graph.levels.len();
            for level in (1..n_levels).rev() {
                let cands = beam_search(
                    &graph.levels[level], None, &[cur_entry],
                    &query_scores, n_query, flat_codes, n_fine,
                    1, None, false,
                );
                if let Some(&(best, _)) = cands.first() {
                    cur_entry = best;
                }
            }

            // Bottom layer: cluster-guided entry points with adaptive or fixed ctop
            let query_cids = codebook.assign_vectors(&flat, n_query);
            let mut query_ctop = match cutoff_tree {
                Some(tree) => compute_ctop_adaptive(
                    codebook, &query_cids, tree, n_probes.max(inner_ctop_r),
                ),
                None => compute_ctop(codebook, &query_cids, n_probes.max(inner_ctop_r)),
            };

            // Selectivity-aware cluster pruning: when filtering, skip clusters
            // where fewer than min_cluster_ratio of docs match the filter.
            if let (Some(f), Some(fi)) = (filter.as_ref(), filter_index) {
                if !f.is_empty() {
                    let viable: std::collections::HashSet<u32> =
                        fi.clusters_passing_filter(f, min_cluster_ratio).into_iter().collect();
                    if !viable.is_empty() {
                        query_ctop.retain(|c| viable.contains(c));
                        if query_ctop.is_empty() {
                            query_ctop = viable.into_iter().collect();
                        }
                    }
                }
            }

            let mut entries: Vec<u32> = postings.representatives_for_clusters(&query_ctop);
            entries.push(cur_entry);
            entries.sort_unstable();
            entries.dedup();

            let deleted_ref = filter_mask.as_deref();

            let results = beam_search(
                &graph.levels[0],
                Some(&graph.shortcuts),
                &entries,
                &query_scores,
                n_query,
                flat_codes,
                n_fine,
                ef,
                deleted_ref,
                enable_shortcuts,
            );

            let final_results = if let Some(ref raw_vecs) = inner.raw_vectors {
                let reranked = rerank_by_maxsim(
                    results, &flat, raw_vecs, &inner.doc_offsets, dim, k,
                );
                reranked
                    .into_iter()
                    .filter_map(|(int_idx, score)| {
                        doc_ids.get(int_idx as usize).map(|&doc_id| (doc_id, score))
                    })
                    .collect::<Vec<(u64, f32)>>()
            } else {
                results
                    .into_iter()
                    .take(k)
                    .filter_map(|(int_idx, score)| {
                        doc_ids.get(int_idx as usize).map(|&doc_id| (doc_id, score))
                    })
                    .collect::<Vec<(u64, f32)>>()
            };
            final_results
        });

        Ok(out)
    }

    /// Like `search` but also returns compute stats: (nodes_visited, distance_computations).
    /// Supports the same filter and min_cluster_ratio parameters as `search`.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (query_vectors, k = 10, ef = 100, n_probes = 4, enable_shortcuts = false, filter = None, min_cluster_ratio = 0.01))]
    fn search_with_stats(
        &self,
        py: Python<'_>,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        n_probes: usize,
        enable_shortcuts: bool,
        filter: Option<Vec<(String, String)>>,
        min_cluster_ratio: f32,
    ) -> PyResult<(Vec<(u64, f32)>, (u32, u32))> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built; call build() or load() first")
        })?;
        if inner.graph.levels.is_empty() {
            return Ok((Vec::new(), (0, 0)));
        }
        if k == 0 {
            return Ok((Vec::new(), (0, 0)));
        }
        let ef = ef.max(k);

        let arr = query_vectors.as_array();
        let (n_query, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != inner.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", inner.dim, dim
            )));
        }
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        let codebook = &inner.codebook;
        let flat_codes = &inner.flat_codes;
        let postings = &inner.postings;
        let graph = &inner.graph;
        let doc_ids = &inner.doc_ids;
        let inner_ctop_r = inner.ctop_r;
        let cutoff_tree = inner.cutoff_tree.as_ref();
        let filter_index = inner.filter_index.as_ref();
        let n_docs = doc_ids.len();

        let graph_entry = inner.graph.entry_point;

        let filter_mask: Option<Vec<bool>> = filter.as_ref().and_then(|f| {
            if f.is_empty() { return None; }
            match filter_index {
                Some(fi) => {
                    let mask = fi.build_filter_mask(f, n_docs);
                    Some(mask.iter().map(|&m| !m).collect())
                }
                None => {
                    log::warn!("filter provided but no filter_index built; call set_doc_payloads() first");
                    None
                }
            }
        });

        let out = py.allow_threads(|| {
            let mut query_scores = codebook.compute_query_centroid_scores(&flat, n_query);
            codebook.apply_idf_weights(&mut query_scores, n_query);
            let n_fine = codebook.n_fine;

            let mut total_stats = search::SearchStats::default();

            let mut cur_entry = graph_entry;
            let n_levels = graph.levels.len();
            for level in (1..n_levels).rev() {
                let (cands, lvl_stats) = beam_search_with_stats(
                    &graph.levels[level], None, &[cur_entry],
                    &query_scores, n_query, flat_codes, n_fine,
                    1, None, false,
                );
                total_stats.nodes_visited += lvl_stats.nodes_visited;
                total_stats.distance_computations += lvl_stats.distance_computations;
                if let Some(&(best, _)) = cands.first() {
                    cur_entry = best;
                }
            }

            let query_cids = codebook.assign_vectors(&flat, n_query);
            let mut query_ctop = match cutoff_tree {
                Some(tree) => compute_ctop_adaptive(
                    codebook, &query_cids, tree, n_probes.max(inner_ctop_r),
                ),
                None => compute_ctop(codebook, &query_cids, n_probes.max(inner_ctop_r)),
            };

            if let (Some(f), Some(fi)) = (filter.as_ref(), filter_index) {
                if !f.is_empty() {
                    let viable: std::collections::HashSet<u32> =
                        fi.clusters_passing_filter(f, min_cluster_ratio).into_iter().collect();
                    if !viable.is_empty() {
                        query_ctop.retain(|c| viable.contains(c));
                        if query_ctop.is_empty() {
                            query_ctop = viable.into_iter().collect();
                        }
                    }
                }
            }

            let mut entries: Vec<u32> = postings.representatives_for_clusters(&query_ctop);
            entries.push(cur_entry);
            entries.sort_unstable();
            entries.dedup();

            let deleted_ref = filter_mask.as_deref();

            let (results, bottom_stats) = beam_search_with_stats(
                &graph.levels[0],
                Some(&graph.shortcuts),
                &entries,
                &query_scores,
                n_query,
                flat_codes,
                n_fine,
                ef,
                deleted_ref,
                enable_shortcuts,
            );
            total_stats.nodes_visited += bottom_stats.nodes_visited;
            total_stats.distance_computations += bottom_stats.distance_computations;

            let hits: Vec<(u64, f32)> = if let Some(ref raw_vecs) = inner.raw_vectors {
                let reranked = rerank_by_maxsim(
                    results, &flat, raw_vecs, &inner.doc_offsets, dim, k,
                );
                reranked
                    .into_iter()
                    .filter_map(|(int_idx, score)| {
                        doc_ids.get(int_idx as usize).map(|&doc_id| (doc_id, score))
                    })
                    .collect()
            } else {
                results
                    .into_iter()
                    .take(k)
                    .filter_map(|(int_idx, score)| {
                        doc_ids.get(int_idx as usize).map(|&doc_id| (doc_id, score))
                    })
                    .collect()
            };
            (hits, (total_stats.nodes_visited, total_stats.distance_computations))
        });

        Ok(out)
    }

    /// Save sealed segment to disk.
    #[pyo3(signature = (path))]
    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("no segment to save")
        })?;

        let graph = &inner.graph;
        let codebook = &inner.codebook;
        let doc_profiles = &inner.doc_profiles;
        let doc_ids = &inner.doc_ids;
        let flat_codes = &inner.flat_codes;
        let postings = &inner.postings;
        let dim = inner.dim;
        let ctop_r = inner.ctop_r;

        let raw_vectors = inner.raw_vectors.clone();
        let doc_offsets = inner.doc_offsets.clone();

        py.allow_threads(move || {
            let data = SegmentData {
                dim,
                max_degree: graph.max_degree,
                levels: graph.levels.clone(),
                shortcuts: graph.shortcuts.clone(),
                node_levels: graph.node_levels.clone(),
                entry_point: graph.entry_point,
                codebook: codebook.clone(),
                doc_profiles: doc_profiles.clone(),
                doc_ids: doc_ids.clone(),
                flat_codes: flat_codes.clone(),
                postings: postings.clone(),
                ctop_r,
                raw_vectors,
                doc_offsets,
            };
            save_segment(&data, &std::path::PathBuf::from(path))
        }).map_err(|e| PyValueError::new_err(format!("save failed: {e}")))
    }

    /// Load sealed segment from disk.
    #[pyo3(signature = (path))]
    fn load(&mut self, py: Python<'_>, path: String) -> PyResult<()> {
        let data = py.allow_threads(move || {
            load_segment(&std::path::PathBuf::from(path))
        }).map_err(|e| PyValueError::new_err(format!("load failed: {e}")))?;

        self.inner = Some(SealedInner {
            graph: GemGraph {
                levels: data.levels,
                shortcuts: data.shortcuts.clone(),
                shortcut_generations: vec![Vec::new(); data.shortcuts.len()],
                node_levels: data.node_levels,
                entry_point: data.entry_point,
                max_degree: data.max_degree,
            },
            codebook: data.codebook,
            flat_codes: data.flat_codes,
            doc_ids: data.doc_ids,
            doc_profiles: data.doc_profiles,
            postings: data.postings,
            ctop_r: data.ctop_r,
            dim: data.dim,
            cutoff_tree: None,
            filter_index: None,
            raw_vectors: data.raw_vectors,
            doc_offsets: data.doc_offsets,
        });

        Ok(())
    }

    /// Inject semantic shortcuts from training pairs.
    /// training_pairs: list of (query_vectors_flat, positive_doc_internal_id)
    #[pyo3(signature = (training_pairs, max_shortcuts_per_node = 4))]
    fn inject_shortcuts(
        &mut self,
        py: Python<'_>,
        training_pairs: Vec<(Vec<f32>, u32)>,
        max_shortcuts_per_node: usize,
    ) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;

        if max_shortcuts_per_node == 0 {
            return Ok(());
        }
        for (i, (flat, target)) in training_pairs.iter().enumerate() {
            if flat.len() % inner.dim != 0 {
                return Err(PyValueError::new_err(format!(
                    "training_pairs[{}]: flat vector length {} is not a multiple of dim {}",
                    i, flat.len(), inner.dim
                )));
            }
            if (*target as usize) >= inner.graph.n_nodes() {
                return Err(PyValueError::new_err(format!(
                    "training_pairs[{}]: target {} >= n_nodes {}", i, target, inner.graph.n_nodes()
                )));
            }
        }

        let graph = &mut inner.graph;
        let codebook = &inner.codebook;
        let flat_codes = &inner.flat_codes;
        let dim = inner.dim;

        py.allow_threads(|| {
            graph.inject_shortcuts(
                &training_pairs, max_shortcuts_per_node,
                codebook, flat_codes, dim,
            );
        });

        Ok(())
    }

    /// Load a pre-trained adaptive cutoff tree from serialized bytes.
    ///
    /// Once loaded, search/search_batch will use per-query adaptive cluster
    /// probing instead of a fixed n_probes, as described in GEM paper §4.4.2.
    #[pyo3(signature = (tree_bytes,))]
    fn load_cutoff_tree(&mut self, tree_bytes: Vec<u8>) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        let tree = CutoffTree::from_bytes(&tree_bytes).ok_or_else(|| {
            PyValueError::new_err("failed to deserialize cutoff tree")
        })?;
        inner.cutoff_tree = Some(tree);
        Ok(())
    }

    /// Batch search: process multiple queries in parallel via rayon.
    #[pyo3(signature = (queries, k = 10, ef = 100, n_probes = 4, enable_shortcuts = false))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: Vec<PyReadonlyArray2<f32>>,
        k: usize,
        ef: usize,
        n_probes: usize,
        enable_shortcuts: bool,
    ) -> PyResult<Vec<Vec<(u64, f32)>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        if queries.is_empty() || k == 0 || inner.graph.levels.is_empty() {
            return Ok(vec![Vec::new(); queries.len()]);
        }
        let ef = ef.max(k);

        let mut query_data: Vec<(Vec<f32>, usize)> = Vec::with_capacity(queries.len());
        for q in &queries {
            let arr = q.as_array();
            let (n_q, dim) = (arr.shape()[0], arr.shape()[1]);
            if dim != inner.dim {
                return Err(PyValueError::new_err(format!(
                    "dimension mismatch: expected {}, got {}", inner.dim, dim
                )));
            }
            let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
                PyValueError::new_err("array must be C-contiguous")
            })?.to_vec();
            query_data.push((flat, n_q));
        }

        let codebook = &inner.codebook;
        let flat_codes = &inner.flat_codes;
        let postings = &inner.postings;
        let graph = &inner.graph;
        let doc_ids = &inner.doc_ids;
        let ctop_r = inner.ctop_r;
        let cutoff_tree = inner.cutoff_tree.as_ref();

        let results = py.allow_threads(|| {
            use rayon::prelude::*;
            query_data.par_iter().map(|(flat, n_query)| {
                let n_query = *n_query;
                let mut query_scores = codebook.compute_query_centroid_scores(flat, n_query);
                codebook.apply_idf_weights(&mut query_scores, n_query);
                let n_fine = codebook.n_fine;

                // Multi-level descent
                let mut cur_entry = graph.entry_point;
                for level in (1..graph.levels.len()).rev() {
                    let cands = beam_search(
                        &graph.levels[level], None, &[cur_entry],
                        &query_scores, n_query, flat_codes, n_fine,
                        1, None, false,
                    );
                    if let Some(&(best, _)) = cands.first() {
                        cur_entry = best;
                    }
                }

                let query_cids = codebook.assign_vectors(flat, n_query);
                let query_ctop = match cutoff_tree {
                    Some(tree) => compute_ctop_adaptive(
                        codebook, &query_cids, tree, n_probes.max(ctop_r),
                    ),
                    None => compute_ctop(codebook, &query_cids, n_probes.max(ctop_r)),
                };
                let mut entries: Vec<u32> = postings.representatives_for_clusters(&query_ctop);
                entries.push(cur_entry);
                entries.sort_unstable();
                entries.dedup();

                let hits = beam_search(
                    &graph.levels[0],
                    Some(&graph.shortcuts),
                    &entries,
                    &query_scores,
                    n_query,
                    flat_codes,
                    n_fine,
                    ef,
                    None,
                    enable_shortcuts,
                );

                if let Some(ref raw_vecs) = inner.raw_vectors {
                    let dim = inner.dim;
                    let reranked = rerank_by_maxsim(
                        hits, flat, raw_vecs, &inner.doc_offsets, dim, k,
                    );
                    reranked
                        .into_iter()
                        .filter_map(|(int_idx, score)| {
                            doc_ids.get(int_idx as usize).map(|&did| (did, score))
                        })
                        .collect::<Vec<(u64, f32)>>()
                } else {
                    hits.into_iter()
                        .take(k)
                        .filter_map(|(int_idx, score)| {
                            doc_ids.get(int_idx as usize).map(|&did| (did, score))
                        })
                        .collect::<Vec<(u64, f32)>>()
                }
            }).collect()
        });

        Ok(results)
    }

    fn n_docs(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.doc_ids.len())
    }

    fn n_nodes(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.graph.n_nodes())
    }

    fn n_edges(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.graph.n_edges())
    }

    fn dim(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.dim)
    }

    fn is_ready(&self) -> bool {
        self.inner.is_some()
    }

    /// BFS connectivity report on the bottom graph layer.
    ///
    /// Returns (n_components, giant_component_frac, cross_cluster_edge_ratio):
    ///   - n_components: number of connected components
    ///   - giant_component_frac: fraction of nodes in the largest component (1.0 = fully connected)
    ///   - cross_cluster_edge_ratio: fraction of edges connecting different clusters
    fn graph_connectivity_report(&self) -> PyResult<(usize, f64, f64)> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built; call build() or load() first")
        })?;
        let (n_components, giant_frac) = inner.graph.connectivity_report();

        let n_docs = inner.doc_ids.len();
        let mut node_cluster = vec![u32::MAX; n_docs];
        for (c, members) in inner.postings.lists.iter().enumerate() {
            for &doc_int_id in members {
                let d = doc_int_id as usize;
                if d < n_docs {
                    node_cluster[d] = c as u32;
                }
            }
        }
        let adj = inner.graph.bottom_adjacency();
        let mut total_edges: u64 = 0;
        let mut cross_edges: u64 = 0;
        for node in 0..adj.n_nodes().min(n_docs) {
            for &nbr in adj.neighbors(node) {
                let n = nbr as usize;
                if n < n_docs {
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

        Ok((n_components, giant_frac, cross_ratio))
    }

    /// Set per-document payload field-value pairs for filter-aware routing.
    ///
    /// payloads: list of (doc_external_id, [(field, value), ...])
    #[pyo3(signature = (payloads,))]
    fn set_doc_payloads(
        &mut self,
        payloads: Vec<(u64, Vec<(String, String)>)>,
    ) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;

        let ext_to_int: std::collections::HashMap<u64, u32> = inner
            .doc_ids
            .iter()
            .enumerate()
            .map(|(i, &ext)| (ext, i as u32))
            .collect();

        let int_payloads: Vec<(u32, Vec<(String, String)>)> = payloads
            .into_iter()
            .filter_map(|(ext_id, fv)| ext_to_int.get(&ext_id).map(|&int_id| (int_id, fv)))
            .collect();

        let n_clusters = inner.postings.lists.len();
        let mut fi = FilterIndex::new(n_clusters);
        fi.build_from_payloads(&int_payloads, &inner.postings);
        inner.filter_index = Some(fi);
        Ok(())
    }

    /// Prune shortcuts pointing to deleted nodes and optionally age-based pruning.
    #[pyo3(signature = (deleted_flags, max_age = None, current_generation = 0))]
    fn prune_stale_shortcuts(
        &mut self,
        deleted_flags: Vec<bool>,
        max_age: Option<usize>,
        current_generation: usize,
    ) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        inner.graph.prune_stale_shortcuts(&deleted_flags, max_age, current_generation);
        Ok(())
    }

    fn total_shortcuts(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.graph.total_shortcuts())
    }

    /// Brute-force rank all docs by the segment's own qCH proxy score.
    ///
    /// Returns the top-k docs scored identically to what beam_search uses
    /// (IDF-weighted centroid dot products via flat u16 codes). This serves
    /// as the "proxy oracle" — separating quantization error from graph error.
    #[pyo3(signature = (query_vectors, k = 10))]
    fn brute_force_proxy(
        &self,
        py: Python<'_>,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        let arr = query_vectors.as_array();
        let (n_query, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != inner.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", inner.dim, dim
            )));
        }
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        let codebook = &inner.codebook;
        let flat_codes = &inner.flat_codes;
        let doc_ids = &inner.doc_ids;
        let n_docs = doc_ids.len();

        let out = py.allow_threads(|| {
            use latence_gem_router::codebook::qch_proxy_score_u16;
            let mut query_scores = codebook.compute_query_centroid_scores(&flat, n_query);
            codebook.apply_idf_weights(&mut query_scores, n_query);
            let n_fine = codebook.n_fine;

            let mut scored: Vec<(usize, f32)> = (0..n_docs)
                .map(|i| {
                    let codes = flat_codes.doc_codes(i);
                    let s = qch_proxy_score_u16(&query_scores, n_query, n_fine, codes);
                    (i, s)
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            scored.truncate(k);
            scored
                .into_iter()
                .filter_map(|(idx, score)| {
                    doc_ids.get(idx).map(|&did| (did, score))
                })
                .collect::<Vec<(u64, f32)>>()
        });
        Ok(out)
    }

    /// Brute-force rank all docs by exact MaxSim on raw vectors.
    ///
    /// Returns the true top-k docs by MaxSim similarity (negated, so lower = better).
    /// Only available when raw vectors are stored (i.e., after build(), not after load()).
    #[pyo3(signature = (query_vectors, k = 10))]
    fn brute_force_maxsim(
        &self,
        py: Python<'_>,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        let raw_vecs = inner.raw_vectors.as_ref().ok_or_else(|| {
            PyValueError::new_err("raw vectors not available (loaded from disk without vectors)")
        })?;
        let arr = query_vectors.as_array();
        let dim = arr.shape()[1];
        if dim != inner.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", inner.dim, dim
            )));
        }
        let query_flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        let doc_ids = &inner.doc_ids;
        let doc_offsets = &inner.doc_offsets;
        let n_docs = doc_ids.len();

        let out = py.allow_threads(|| {
            let mut scored: Vec<(usize, f32)> = (0..n_docs)
                .map(|i| {
                    let (start, end) = doc_offsets[i];
                    let doc_vecs = &raw_vecs[start * dim..end * dim];
                    let s = maxsim_score(&query_flat, doc_vecs, dim);
                    (i, s)
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            scored.truncate(k);
            scored
                .into_iter()
                .filter_map(|(idx, score)| {
                    doc_ids.get(idx).map(|&did| (did, score))
                })
                .collect::<Vec<(u64, f32)>>()
        });
        Ok(out)
    }

    /// Return raw vectors for each document, keyed by external doc_id.
    ///
    /// Returns None if raw vectors were not stored (store_raw_vectors=False
    /// at build time, or loaded from a v1 segment file).
    /// Otherwise returns a list of (doc_id, ndarray) tuples.
    #[allow(clippy::type_complexity)]
    fn get_doc_vectors<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Vec<(u64, Bound<'py, PyArray2<f32>>)>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built or loaded")
        })?;
        let raw_vecs = match &inner.raw_vectors {
            Some(v) => v,
            None => return Ok(None),
        };
        let dim = inner.dim;
        let mut result = Vec::with_capacity(inner.doc_ids.len());
        for (i, &doc_id) in inner.doc_ids.iter().enumerate() {
            if i >= inner.doc_offsets.len() {
                break;
            }
            let (start, end) = inner.doc_offsets[i];
            let n_tokens = end.saturating_sub(start);
            if n_tokens == 0 || start * dim >= raw_vecs.len() {
                continue;
            }
            let slice = &raw_vecs[start * dim..end * dim];
            let rows: Vec<Vec<f32>> = (0..n_tokens)
                .map(|t| slice[t * dim..(t + 1) * dim].to_vec())
                .collect();
            let arr = PyArray2::from_vec2_bound(py, &rows)
                .map_err(|e| PyValueError::new_err(format!("array creation failed: {e}")))?;
            result.push((doc_id, arr));
        }
        Ok(Some(result))
    }

    /// Return true if this segment has raw vectors available for MaxSim reranking.
    fn has_raw_vectors(&self) -> bool {
        self.inner
            .as_ref()
            .map(|i| i.raw_vectors.is_some())
            .unwrap_or(false)
    }

    /// Export codebook centroids as a flat float32 array: shape (n_fine, dim).
    fn get_codebook_centroids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        let n_fine = inner.codebook.n_fine;
        let dim = inner.codebook.dim;
        let data = &inner.codebook.cquant;
        let rows: Vec<Vec<f32>> = (0..n_fine).map(|i| {
            data[i * dim..(i + 1) * dim].to_vec()
        }).collect();
        let arr = PyArray2::from_vec2_bound(py, &rows)
            .map_err(|e| PyValueError::new_err(format!("array error: {e}")))?;
        Ok(arr)
    }

    /// Export IDF weights as float32 array: shape (n_fine,).
    fn get_idf<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        Ok(PyArray1::from_slice_bound(py, &inner.codebook.idf))
    }

    /// Export flat doc codes as (codes: u16[], offsets: u32[], lengths: u16[]).
    #[allow(clippy::type_complexity)]
    fn get_flat_codes<'py>(&self, py: Python<'py>) -> PyResult<(
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u16>>,
    )> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        let codes = PyArray1::from_slice_bound(py, &inner.flat_codes.codes);
        let offsets = PyArray1::from_slice_bound(py, &inner.flat_codes.offsets);
        let lengths = PyArray1::from_slice_bound(py, &inner.flat_codes.lengths);
        Ok((codes, offsets, lengths))
    }
}

/// Mutable GEM segment supporting insert, delete, upsert, and compaction.
#[pyclass]
pub struct PyMutableGemSegment {
    inner: Option<MutableGemSegment>,
}

#[pymethods]
impl PyMutableGemSegment {
    #[new]
    fn new() -> Self {
        Self { inner: None }
    }

    /// Build a mutable segment from a seed batch: trains codebook + builds initial graph.
    ///   use_emd: use qEMD for neighbor selection during insert (default false for latency)
    #[pyo3(signature = (all_vectors, doc_ids, doc_offsets, n_fine = 256, n_coarse = 32, max_degree = 32, ef_construction = 200, max_kmeans_iter = 30, ctop_r = 3, n_probes = 4, use_emd = false))]
    fn build(
        &mut self,
        py: Python<'_>,
        all_vectors: PyReadonlyArray2<f32>,
        doc_ids: Vec<u64>,
        doc_offsets: Vec<(usize, usize)>,
        n_fine: usize,
        n_coarse: usize,
        max_degree: usize,
        ef_construction: usize,
        max_kmeans_iter: usize,
        ctop_r: usize,
        n_probes: usize,
        use_emd: bool,
    ) -> PyResult<()> {
        let arr = all_vectors.as_array();
        let (n_vectors, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        let n_docs = doc_ids.len();
        if n_docs != doc_offsets.len() {
            return Err(PyValueError::new_err(format!(
                "doc_ids length ({}) != doc_offsets length ({})",
                n_docs, doc_offsets.len()
            )));
        }
        if n_docs > u32::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "too many documents ({}) — max supported is {}", n_docs, u32::MAX
            )));
        }
        for (i, &(start, end)) in doc_offsets.iter().enumerate() {
            if start > end || end > n_vectors {
                return Err(PyValueError::new_err(format!(
                    "doc_offsets[{}] = ({}, {}) out of range for {} vectors",
                    i, start, end, n_vectors
                )));
            }
        }
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        let mut seg = py.allow_threads(move || {
            MutableGemSegment::build(
                &flat, dim, &doc_ids, &doc_offsets,
                n_fine, n_coarse, max_degree, ef_construction,
                max_kmeans_iter, ctop_r, n_probes,
            )
        });
        seg.use_emd = use_emd;

        self.inner = Some(seg);
        Ok(())
    }

    /// Search the mutable segment. `n_probes` is accepted for API parity but
    /// unused: mutable segments use a flat single-layer graph without cluster
    /// routing. Use sealed segments for cluster-guided multi-probe search.
    #[pyo3(signature = (query_vectors, k = 10, ef = 100, n_probes = 4))]
    fn search(
        &self,
        py: Python<'_>,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        #[allow(unused_variables)]
        n_probes: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        let seg = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;

        let arr = query_vectors.as_array();
        let (n_query, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != seg.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", seg.dim, dim
            )));
        }
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        let out = py.allow_threads(|| {
            let mut query_scores = seg.codebook.compute_query_centroid_scores(&flat, n_query);
            seg.codebook.apply_idf_weights(&mut query_scores, n_query);
            seg.search(&query_scores, n_query, k, ef)
        });

        Ok(out)
    }

    /// Insert a single document (multi-vector) into the mutable graph.
    #[pyo3(signature = (vectors, doc_id))]
    fn insert(
        &mut self,
        py: Python<'_>,
        vectors: PyReadonlyArray2<f32>,
        doc_id: u64,
    ) -> PyResult<()> {
        let seg = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;

        let arr = vectors.as_array();
        let (n_tokens, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != seg.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", seg.dim, dim
            )));
        }
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        py.allow_threads(|| {
            seg.insert(&flat, n_tokens, doc_id);
        });
        Ok(())
    }

    /// Batch insert multiple documents in a single lock acquisition.
    #[pyo3(signature = (vectors_list, doc_ids))]
    fn insert_batch(
        &mut self,
        py: Python<'_>,
        vectors_list: Vec<PyReadonlyArray2<f32>>,
        doc_ids: Vec<u64>,
    ) -> PyResult<()> {
        let seg = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;

        if vectors_list.len() != doc_ids.len() {
            return Err(PyValueError::new_err(format!(
                "vectors_list length ({}) != doc_ids length ({})",
                vectors_list.len(), doc_ids.len()
            )));
        }

        let mut batch: Vec<(Vec<f32>, usize, u64)> = Vec::with_capacity(vectors_list.len());
        for (vecs, &doc_id) in vectors_list.iter().zip(doc_ids.iter()) {
            let arr = vecs.as_array();
            let (n_tokens, dim) = (arr.shape()[0], arr.shape()[1]);
            if dim != seg.dim {
                return Err(PyValueError::new_err(format!(
                    "dimension mismatch: expected {}, got {}", seg.dim, dim
                )));
            }
            let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
                PyValueError::new_err("array must be C-contiguous")
            })?.to_vec();
            batch.push((flat, n_tokens, doc_id));
        }

        py.allow_threads(|| {
            for (flat, n_tokens, doc_id) in &batch {
                seg.insert(flat, *n_tokens, *doc_id);
            }
        });
        Ok(())
    }

    /// Soft-delete a document by external ID. Returns True if found.
    #[pyo3(signature = (doc_id))]
    fn delete(&mut self, doc_id: u64) -> PyResult<bool> {
        let seg = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        Ok(seg.delete(doc_id))
    }

    /// Upsert: delete old version (if exists) then insert new.
    #[pyo3(signature = (vectors, doc_id))]
    fn upsert(
        &mut self,
        py: Python<'_>,
        vectors: PyReadonlyArray2<f32>,
        doc_id: u64,
    ) -> PyResult<()> {
        let seg = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;

        let arr = vectors.as_array();
        let (n_tokens, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != seg.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", seg.dim, dim
            )));
        }
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        py.allow_threads(|| {
            seg.upsert(&flat, n_tokens, doc_id);
        });
        Ok(())
    }

    fn compact(&mut self, py: Python<'_>) -> PyResult<()> {
        let seg = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        py.allow_threads(|| {
            seg.compact();
        });
        Ok(())
    }

    /// Run local graph repair: fix stale reps, reconnect isolated nodes.
    fn heal(&mut self, py: Python<'_>) -> PyResult<()> {
        let seg = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        py.allow_threads(|| {
            seg.heal();
        });
        Ok(())
    }

    /// Graph quality metrics: (delete_ratio, avg_degree, isolated_node_ratio, stale_rep_ratio).
    fn graph_quality_metrics(&self) -> PyResult<(f64, f64, f64, f64)> {
        let seg = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        Ok(seg.graph_quality_metrics())
    }

    /// Deep connectivity report: (n_components, giant_component_frac, cross_cluster_edge_ratio).
    ///
    /// - n_components: number of connected components among live nodes
    /// - giant_component_frac: fraction of live nodes in the largest component (1.0 = fully connected)
    /// - cross_cluster_edge_ratio: fraction of edges connecting different clusters (higher = better navigability)
    fn graph_connectivity_report(&self) -> PyResult<(usize, f64, f64)> {
        let seg = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        Ok(seg.graph_connectivity_report())
    }

    /// Returns True if the graph needs healing based on drift thresholds.
    fn needs_healing(&self) -> PyResult<bool> {
        let seg = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        Ok(seg.needs_healing())
    }

    /// Batch search: process multiple queries in parallel.
    #[pyo3(signature = (queries, k = 10, ef = 100))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: Vec<PyReadonlyArray2<f32>>,
        k: usize,
        ef: usize,
    ) -> PyResult<Vec<Vec<(u64, f32)>>> {
        let seg = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;

        let mut query_data: Vec<(Vec<f32>, usize)> = Vec::with_capacity(queries.len());
        for q in &queries {
            let arr = q.as_array();
            let (n_q, dim) = (arr.shape()[0], arr.shape()[1]);
            if dim != seg.dim {
                return Err(PyValueError::new_err(format!(
                    "dimension mismatch: expected {}, got {}", seg.dim, dim
                )));
            }
            let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
                PyValueError::new_err("array must be C-contiguous")
            })?.to_vec();
            query_data.push((flat, n_q));
        }

        let results = py.allow_threads(|| {
            use rayon::prelude::*;
            query_data.par_iter().map(|(flat, n_query)| {
                let n_query = *n_query;
                let mut query_scores = seg.codebook.compute_query_centroid_scores(flat, n_query);
                seg.codebook.apply_idf_weights(&mut query_scores, n_query);
                seg.search(&query_scores, n_query, k, ef)
            }).collect()
        });

        Ok(results)
    }

    fn n_nodes(&self) -> usize {
        self.inner.as_ref().map_or(0, |s| s.n_nodes())
    }

    fn n_edges(&self) -> usize {
        self.inner.as_ref().map_or(0, |s| s.n_edges())
    }

    fn n_live(&self) -> usize {
        self.inner.as_ref().map_or(0, |s| s.n_live())
    }

    /// Quality score: 1.0 = same as initial build, < 1.0 = degraded.
    fn quality_score(&self) -> f64 {
        self.inner.as_ref().map_or(1.0, |s| s.quality_score())
    }

    fn delete_ratio(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, |s| s.delete_ratio())
    }

    fn avg_degree(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, |s| s.avg_degree())
    }

    fn memory_bytes(&self) -> usize {
        self.inner.as_ref().map_or(0, |s| s.memory_bytes())
    }

    fn dim(&self) -> usize {
        self.inner.as_ref().map_or(0, |s| s.dim)
    }

    fn is_ready(&self) -> bool {
        self.inner.is_some()
    }
}

/// Multi-index ensemble for modality-specific codebooks with RRF fusion.
#[pyclass]
pub struct PyEnsembleGemSegment {
    inner: Option<ensemble::EnsembleSegment>,
}

#[pymethods]
impl PyEnsembleGemSegment {
    #[new]
    fn new() -> Self {
        Self { inner: None }
    }

    /// Build ensemble from multi-modal document vectors.
    ///
    /// Args:
    ///   all_vectors: (N, D) float32 matrix of all token vectors
    ///   doc_ids: external document IDs
    ///   doc_offsets: (start, end) index pairs per document
    ///   modality_tags: per-token modality ID (0..n_modalities-1), length N
    ///   n_modalities: number of modality types
    #[pyo3(signature = (all_vectors, doc_ids, doc_offsets, modality_tags, n_modalities, n_fine = 256, n_coarse = 32, max_degree = 32, ef_construction = 200, max_kmeans_iter = 30, ctop_r = 3))]
    fn build(
        &mut self,
        py: Python<'_>,
        all_vectors: PyReadonlyArray2<f32>,
        doc_ids: Vec<u64>,
        doc_offsets: Vec<(usize, usize)>,
        modality_tags: Vec<u8>,
        n_modalities: u8,
        n_fine: usize,
        n_coarse: usize,
        max_degree: usize,
        ef_construction: usize,
        max_kmeans_iter: usize,
        ctop_r: usize,
    ) -> PyResult<()> {
        let arr = all_vectors.as_array();
        let (n_vectors, dim) = (arr.shape()[0], arr.shape()[1]);
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        if modality_tags.len() != n_vectors {
            return Err(PyValueError::new_err(format!(
                "modality_tags length ({}) != n_vectors ({})",
                modality_tags.len(), n_vectors
            )));
        }
        if doc_ids.len() != doc_offsets.len() {
            return Err(PyValueError::new_err(format!(
                "doc_ids length ({}) != doc_offsets length ({})",
                doc_ids.len(), doc_offsets.len()
            )));
        }

        let ens = py.allow_threads(move || {
            ensemble::EnsembleSegment::build(
                &flat, dim, &doc_ids, &doc_offsets,
                &modality_tags, n_modalities,
                n_fine, n_coarse, max_degree, ef_construction,
                max_kmeans_iter, ctop_r,
            )
        });
        self.inner = Some(ens);
        Ok(())
    }

    /// Search ensemble with multi-modal query.
    ///
    /// Args:
    ///   query_vectors: (Q, D) float32 query token vectors
    ///   query_modality_tags: per-query-token modality ID, length Q
    ///   k: top-k results
    ///   ef: beam width
    ///   n_probes: number of coarse clusters to probe
    #[pyo3(signature = (query_vectors, query_modality_tags, k = 10, ef = 100, n_probes = 4))]
    fn search(
        &self,
        py: Python<'_>,
        query_vectors: PyReadonlyArray2<f32>,
        query_modality_tags: Vec<u8>,
        k: usize,
        ef: usize,
        n_probes: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        let ens = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("ensemble not built")
        })?;
        let arr = query_vectors.as_array();
        let (n_query, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != ens.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", ens.dim, dim
            )));
        }
        if query_modality_tags.len() != n_query {
            return Err(PyValueError::new_err(format!(
                "query_modality_tags length ({}) != n_query ({})",
                query_modality_tags.len(), n_query
            )));
        }
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

        let out = py.allow_threads(|| {
            ens.search(&flat, n_query, &query_modality_tags, k, ef, n_probes)
        });
        Ok(out)
    }

    fn n_docs(&self) -> usize {
        self.inner.as_ref().map_or(0, |e| e.doc_ids.len())
    }

    fn n_modalities(&self) -> u8 {
        self.inner.as_ref().map_or(0, |e| e.n_modalities)
    }

    fn is_ready(&self) -> bool {
        self.inner.is_some()
    }
}

#[pymodule]
fn latence_gem_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GemSegment>()?;
    m.add_class::<PyMutableGemSegment>()?;
    m.add_class::<PyEnsembleGemSegment>()?;
    Ok(())
}
