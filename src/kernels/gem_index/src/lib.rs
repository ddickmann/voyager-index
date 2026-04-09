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

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray2;

use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop};
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes};

use graph::GemGraph;
use persistence::{SegmentData, save_segment, load_segment};
use search::beam_search;
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
    #[pyo3(signature = (all_vectors, doc_ids, doc_offsets, n_fine = 256, n_coarse = 32, max_degree = 32, ef_construction = 200, max_kmeans_iter = 30, ctop_r = 3, payload_clusters = None, use_emd = true, dual_graph = true))]
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
            }
        });

        self.inner = Some(sealed);
        Ok(())
    }

    /// Search the sealed segment for nearest neighbors to a multi-vector query.
    ///
    /// Returns: list of (doc_id, qch_proxy_score) tuples, sorted best-first
    #[pyo3(signature = (query_vectors, k = 10, ef = 100, n_probes = 4, enable_shortcuts = false))]
    fn search(
        &self,
        py: Python<'_>,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        n_probes: usize,
        enable_shortcuts: bool,
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

        let graph_entry = inner.graph.entry_point;

        let out = py.allow_threads(|| {
            let query_scores = codebook.compute_query_centroid_scores(&flat, n_query);
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

            // Bottom layer: full beam search with cluster-guided entry points
            let query_cids = codebook.assign_vectors(&flat, n_query);
            let query_ctop = compute_ctop(codebook, &query_cids, n_probes.max(inner_ctop_r));
            let mut entries: Vec<u32> = postings.representatives_for_clusters(&query_ctop);
            entries.push(cur_entry);
            entries.sort_unstable();
            entries.dedup();

            let results = beam_search(
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
                shortcuts: data.shortcuts,
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

        let results = py.allow_threads(|| {
            use rayon::prelude::*;
            query_data.par_iter().map(|(flat, n_query)| {
                let n_query = *n_query;
                let query_scores = codebook.compute_query_centroid_scores(flat, n_query);
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
                let query_ctop = compute_ctop(codebook, &query_cids, n_probes.max(ctop_r));
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

    fn total_shortcuts(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.graph.total_shortcuts())
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
            let query_scores = seg.codebook.compute_query_centroid_scores(&flat, n_query);
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
                let query_scores = seg.codebook.compute_query_centroid_scores(flat, n_query);
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

#[pymodule]
fn latence_gem_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GemSegment>()?;
    m.add_class::<PyMutableGemSegment>()?;
    Ok(())
}
