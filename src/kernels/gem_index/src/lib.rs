pub mod visited;
pub mod id_tracker;
pub mod emd;
pub mod graph;
pub mod search;
pub mod persistence;
pub mod mutable;

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
    #[pyo3(signature = (all_vectors, doc_ids, doc_offsets, n_fine = 256, n_coarse = 32, max_degree = 32, ef_construction = 200, max_kmeans_iter = 30, ctop_r = 3))]
    fn build(
        &mut self,
        all_vectors: PyReadonlyArray2<f32>,
        doc_ids: Vec<u64>,
        doc_offsets: Vec<(usize, usize)>,
        n_fine: usize,
        n_coarse: usize,
        max_degree: usize,
        ef_construction: usize,
        max_kmeans_iter: usize,
        ctop_r: usize,
    ) -> PyResult<()> {
        let arr = all_vectors.as_array();
        let (n_vectors, dim) = (arr.shape()[0], arr.shape()[1]);
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;

        let n_docs = doc_ids.len();

        // Build codebook
        let mut codebook = TwoStageCodebook::build(
            flat, n_vectors, dim, n_fine, n_coarse, max_kmeans_iter, 42,
        );
        let all_assignments = codebook.assign_vectors(flat, n_vectors);

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

        // Build graph
        let gem_graph = graph::build_graph(
            flat,
            dim,
            &doc_offsets,
            &codebook,
            &flat_codes,
            &doc_profiles,
            &postings,
            max_degree,
            ef_construction,
        );

        self.inner = Some(SealedInner {
            graph: gem_graph,
            codebook,
            flat_codes,
            doc_ids,
            doc_profiles,
            postings,
            ctop_r,
            dim,
        });

        Ok(())
    }

    /// Search the sealed segment for nearest neighbors to a multi-vector query.
    ///
    /// Returns: list of (doc_id, qch_proxy_score) tuples, sorted best-first
    #[pyo3(signature = (query_vectors, k = 10, ef = 100, n_probes = 4, enable_shortcuts = false))]
    fn search(
        &self,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        n_probes: usize,
        enable_shortcuts: bool,
    ) -> PyResult<Vec<(u64, f32)>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("segment not built; call build() or load() first")
        })?;

        let arr = query_vectors.as_array();
        let (n_query, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != inner.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", inner.dim, dim
            )));
        }
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;

        let query_scores = inner.codebook.compute_query_centroid_scores(flat, n_query);
        let n_fine = inner.codebook.n_fine;

        // Get entry points from cluster representatives
        let query_cids = inner.codebook.assign_vectors(flat, n_query);
        let query_ctop = compute_ctop(&inner.codebook, &query_cids, n_probes.max(inner.ctop_r));
        let mut entries: Vec<u32> = inner.postings
            .representatives_for_clusters(&query_ctop);
        if entries.is_empty() {
            entries.push(0);
        }

        let results = beam_search(
            &inner.graph.adjacency,
            &inner.graph.shortcuts,
            &entries,
            &query_scores,
            n_query,
            &inner.flat_codes,
            n_fine,
            ef,
            None,
            enable_shortcuts,
        );

        let out: Vec<(u64, f32)> = results
            .into_iter()
            .take(k)
            .map(|(int_idx, score)| {
                let doc_id = inner.doc_ids[int_idx as usize];
                (doc_id, score)
            })
            .collect();

        Ok(out)
    }

    /// Save sealed segment to disk.
    #[pyo3(signature = (path))]
    fn save(&self, path: String) -> PyResult<()> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("no segment to save")
        })?;

        let data = SegmentData {
            dim: inner.dim,
            max_degree: inner.graph.max_degree,
            adjacency: inner.graph.adjacency.clone(),
            shortcuts: inner.graph.shortcuts.clone(),
            codebook: inner.codebook.clone(),
            doc_profiles: inner.doc_profiles.clone(),
            doc_ids: inner.doc_ids.clone(),
            flat_codes: inner.flat_codes.clone(),
            postings: inner.postings.clone(),
            ctop_r: inner.ctop_r,
        };

        save_segment(&data, &std::path::PathBuf::from(path))
            .map_err(|e| PyValueError::new_err(format!("save failed: {e}")))
    }

    /// Load sealed segment from disk.
    #[pyo3(signature = (path))]
    fn load(&mut self, path: String) -> PyResult<()> {
        let data = load_segment(&std::path::PathBuf::from(path))
            .map_err(|e| PyValueError::new_err(format!("load failed: {e}")))?;

        self.inner = Some(SealedInner {
            graph: GemGraph {
                adjacency: data.adjacency,
                shortcuts: data.shortcuts,
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
        training_pairs: Vec<(Vec<f32>, u32)>,
        max_shortcuts_per_node: usize,
    ) -> PyResult<()> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;

        inner.graph.inject_shortcuts(
            &training_pairs,
            max_shortcuts_per_node,
            &inner.codebook,
            &inner.flat_codes,
            inner.dim,
        );

        Ok(())
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
    #[pyo3(signature = (all_vectors, doc_ids, doc_offsets, n_fine = 256, n_coarse = 32, max_degree = 32, ef_construction = 200, max_kmeans_iter = 30, ctop_r = 3, n_probes = 4))]
    fn build(
        &mut self,
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
    ) -> PyResult<()> {
        let arr = all_vectors.as_array();
        let (_n_vectors, dim) = (arr.shape()[0], arr.shape()[1]);
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;

        self.inner = Some(MutableGemSegment::build(
            flat,
            dim,
            &doc_ids,
            &doc_offsets,
            n_fine,
            n_coarse,
            max_degree,
            ef_construction,
            max_kmeans_iter,
            ctop_r,
            n_probes,
        ));

        Ok(())
    }

    /// Search the mutable segment.
    #[pyo3(signature = (query_vectors, k = 10, ef = 100, _n_probes = 4))]
    fn search(
        &self,
        query_vectors: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        _n_probes: usize,
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
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;

        let query_scores = seg.codebook.compute_query_centroid_scores(flat, n_query);
        Ok(seg.search(&query_scores, n_query, k, ef))
    }

    /// Insert a single document (multi-vector) into the mutable graph.
    #[pyo3(signature = (vectors, doc_id))]
    fn insert(
        &mut self,
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
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;

        seg.insert(flat, n_tokens, doc_id);
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
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;

        seg.upsert(flat, n_tokens, doc_id);
        Ok(())
    }

    fn compact(&mut self) -> PyResult<()> {
        let seg = self.inner.as_mut().ok_or_else(|| {
            PyValueError::new_err("segment not built")
        })?;
        seg.compact();
        Ok(())
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
