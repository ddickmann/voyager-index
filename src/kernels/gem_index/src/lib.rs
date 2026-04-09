#![allow(clippy::useless_conversion, clippy::too_many_arguments)]

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

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray2, PyArray1, PyArray2};

use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop, compute_ctop_adaptive};
use latence_gem_router::adaptive_cutoff::CutoffTree;
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes, FilterIndex};

use graph::GemGraph;
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
    #[pyo3(signature = (all_vectors, doc_ids, doc_offsets, n_fine = 256, n_coarse = 32, max_degree = 32, ef_construction = 200, max_kmeans_iter = 30, ctop_r = 3, payload_clusters = None, use_emd = false, dual_graph = true))]
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
        payload_clusters: Option<Vec<u32>>,
        use_emd: bool,
        dual_graph: bool,
    ) -> PyResult<()> {
        let arr = all_vectors.as_array();
        let (n_vectors, dim) = (arr.shape()[0], arr.shape()[1]);
        let flat: Vec<f32> = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?.to_vec();

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

        let sealed = py.allow_threads(move || {
            let mut codebook = TwoStageCodebook::build(
                &flat, n_vectors, dim, n_fine, n_coarse, max_kmeans_iter, 42,
            );
            let all_assignments = codebook.assign_vectors(&flat, n_vectors);

            let mut doc_centroid_sets = Vec::with_capacity(n_docs);
            for &(start, end) in &doc_offsets {
                doc_centroid_sets.push(all_assignments[start..end].to_vec());
            }
            codebook.update_idf(&doc_centroid_sets);

            // Phase 5.4: IDF-weighted codebook refinement — refine centroids
            // by weighting rare centroids higher, improving tail-token discrimination.
            let idf_refine_iters = 3;
            codebook.refine_centroids_idf(&flat, n_vectors, idf_refine_iters);
            // Re-assign after refinement
            let all_assignments = codebook.assign_vectors(&flat, n_vectors);
            let mut doc_centroid_sets = Vec::with_capacity(n_docs);
            for &(start, end) in &doc_offsets {
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

            let gem_graph = if dual_graph {
                graph::build_graph_dual(
                    &flat, dim, &doc_offsets, &codebook, &flat_codes,
                    &doc_profiles, &postings, max_degree, ef_construction,
                    use_emd,
                )
            } else {
                graph::build_graph_with_payload(
                    &flat, dim, &doc_offsets, &codebook, &flat_codes,
                    &doc_profiles, &postings, max_degree, ef_construction,
                    payload_clusters.as_deref(), use_emd,
                )
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
            }
        });

        self.inner = Some(sealed);
        Ok(())
    }

    /// Search the sealed segment for nearest neighbors to a multi-vector query.
    ///
    /// Returns: list of (doc_id, qch_proxy_score) tuples, sorted best-first
    /// filter: optional list of (field, value) pairs for filter-aware routing
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
        #[allow(unused_variables)]
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
            let query_ctop = match cutoff_tree {
                Some(tree) => compute_ctop_adaptive(
                    codebook, &query_cids, tree, n_probes.max(inner_ctop_r),
                ),
                None => compute_ctop(codebook, &query_cids, n_probes.max(inner_ctop_r)),
            };
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

            results
                .into_iter()
                .take(k)
                .filter_map(|(int_idx, score)| {
                    doc_ids.get(int_idx as usize).map(|&doc_id| (doc_id, score))
                })
                .collect::<Vec<(u64, f32)>>()
        });

        Ok(out)
    }

    /// Like `search` but also returns compute stats: (nodes_visited, distance_computations).
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (query_vectors, k = 10, ef = 100, n_probes = 4, enable_shortcuts = false))]
    fn search_with_stats(
        &self,
        py: Python<'_>,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        n_probes: usize,
        enable_shortcuts: bool,
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

        let graph_entry = inner.graph.entry_point;

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
            let query_ctop = match cutoff_tree {
                Some(tree) => compute_ctop_adaptive(
                    codebook, &query_cids, tree, n_probes.max(inner_ctop_r),
                ),
                None => compute_ctop(codebook, &query_cids, n_probes.max(inner_ctop_r)),
            };
            let mut entries: Vec<u32> = postings.representatives_for_clusters(&query_ctop);
            entries.push(cur_entry);
            entries.sort_unstable();
            entries.dedup();

            let (results, bottom_stats) = beam_search_with_stats(
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
            total_stats.nodes_visited += bottom_stats.nodes_visited;
            total_stats.distance_computations += bottom_stats.distance_computations;

            let hits: Vec<(u64, f32)> = results
                .into_iter()
                .take(k)
                .filter_map(|(int_idx, score)| {
                    doc_ids.get(int_idx as usize).map(|&doc_id| (doc_id, score))
                })
                .collect();
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

                hits.into_iter()
                    .take(k)
                    .filter_map(|(int_idx, score)| {
                        doc_ids.get(int_idx as usize).map(|&did| (did, score))
                    })
                    .collect::<Vec<(u64, f32)>>()
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
    /// unused: mutable segments use a flat single-layer graph.
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
