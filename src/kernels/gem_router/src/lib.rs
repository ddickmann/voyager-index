#![allow(clippy::useless_conversion, clippy::too_many_arguments)]

extern crate libc;

pub mod codebook;
pub mod router;
pub mod persistence;
pub mod adaptive_cutoff;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray2;
use std::path::PathBuf;

use router::GemRouter;

/// Python-facing GEM router for multi-vector candidate generation.
#[pyclass]
pub struct PyGemRouter {
    inner: GemRouter,
    dim: usize,
}

#[pymethods]
impl PyGemRouter {
    #[new]
    #[pyo3(signature = (dim))]
    fn new(dim: usize) -> Self {
        Self {
            inner: GemRouter::new(),
            dim,
        }
    }

    /// Build the router from a matrix of all document vectors.
    ///
    /// Args:
    ///   all_vectors: (N, D) float32 matrix of all token/prototype vectors
    ///   doc_ids: list of document IDs (length = n_docs)
    ///   doc_offsets: list of (start, end) tuples indexing into all_vectors for each doc
    ///   n_fine: number of fine centroids (codebook size)
    ///   n_coarse: number of coarse clusters
    ///   max_kmeans_iter: k-means iteration limit
    ///   ctop_r: number of top coarse clusters per document
    #[pyo3(signature = (all_vectors, doc_ids, doc_offsets, n_fine = 256, n_coarse = 32, max_kmeans_iter = 30, ctop_r = 3))]
    fn build(
        &mut self,
        all_vectors: PyReadonlyArray2<f32>,
        doc_ids: Vec<u64>,
        doc_offsets: Vec<(usize, usize)>,
        n_fine: usize,
        n_coarse: usize,
        max_kmeans_iter: usize,
        ctop_r: usize,
    ) -> PyResult<()> {
        let arr = all_vectors.as_array();
        let (n_vectors, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", self.dim, dim
            )));
        }
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;

        self.inner.build(
            flat,
            n_vectors,
            dim,
            &doc_ids,
            &doc_offsets,
            n_fine,
            n_coarse,
            max_kmeans_iter,
            ctop_r,
        );
        Ok(())
    }

    /// Add new documents to an existing router.
    #[pyo3(signature = (new_vectors, new_doc_ids, doc_offsets))]
    fn add_documents(
        &mut self,
        new_vectors: PyReadonlyArray2<f32>,
        new_doc_ids: Vec<u64>,
        doc_offsets: Vec<(usize, usize)>,
    ) -> PyResult<()> {
        if !self.inner.is_ready() {
            return Err(PyValueError::new_err("router not built yet; call build() first"));
        }
        let arr = new_vectors.as_array();
        let (n_vectors, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", self.dim, dim
            )));
        }
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;
        self.inner.add_documents(flat, n_vectors, dim, &new_doc_ids, &doc_offsets);
        Ok(())
    }

    /// Route a query to candidate documents.
    ///
    /// Args:
    ///   query_vectors: (Q, D) float32 matrix of query token vectors
    ///   n_probes: number of coarse clusters to probe
    ///   max_candidates: max number of candidates to return
    ///
    /// Returns:
    ///   list of (doc_id, proxy_score) tuples, sorted best-first
    #[pyo3(signature = (query_vectors, n_probes = 4, max_candidates = 500))]
    fn route_query(
        &self,
        query_vectors: PyReadonlyArray2<f32>,
        n_probes: usize,
        max_candidates: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        if !self.inner.is_ready() {
            return Err(PyValueError::new_err("router not built"));
        }
        let arr = query_vectors.as_array();
        let (n_query_vecs, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", self.dim, dim
            )));
        }
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;
        Ok(self.inner.route_query(flat, n_query_vecs, dim, n_probes, max_candidates))
    }

    /// Get HNSW entry point hints for multi-entry traversal.
    ///
    /// Returns doc IDs of cluster representatives for the query's most relevant clusters.
    #[pyo3(signature = (query_vectors, n_probes = 4))]
    fn get_cluster_entries(
        &self,
        query_vectors: PyReadonlyArray2<f32>,
        n_probes: usize,
    ) -> PyResult<Vec<u64>> {
        if !self.inner.is_ready() {
            return Err(PyValueError::new_err("router not built"));
        }
        let arr = query_vectors.as_array();
        let (n_query_vecs, dim) = (arr.shape()[0], arr.shape()[1]);
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {}, got {}", self.dim, dim
            )));
        }
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;
        Ok(self.inner.get_cluster_entries(flat, n_query_vecs, dim, n_probes))
    }

    /// Compute query centroid IDs and C_top.
    ///
    /// Returns: (centroid_ids: list[int], ctop: list[int])
    #[pyo3(signature = (query_vectors, n_probes = 4))]
    fn compute_query_profile(
        &self,
        query_vectors: PyReadonlyArray2<f32>,
        n_probes: usize,
    ) -> PyResult<(Vec<u32>, Vec<u32>)> {
        if !self.inner.is_ready() {
            return Err(PyValueError::new_err("router not built"));
        }
        let arr = query_vectors.as_array();
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;
        let n_query_vecs = arr.shape()[0];
        self.inner
            .compute_query_profile(flat, n_query_vecs, n_probes)
            .ok_or_else(|| PyValueError::new_err("router state missing"))
    }

    /// Save router state to disk.
    #[pyo3(signature = (path))]
    fn save(&self, path: String) -> PyResult<()> {
        let state = self.inner.state().ok_or_else(|| {
            PyValueError::new_err("no state to save; call build() first")
        })?;
        persistence::save_state(state, &PathBuf::from(path))
            .map_err(|e| PyValueError::new_err(format!("save failed: {e}")))
    }

    /// Load router state from disk.
    #[pyo3(signature = (path))]
    fn load(&mut self, path: String) -> PyResult<()> {
        let state = persistence::load_state(&PathBuf::from(path))
            .map_err(|e| PyValueError::new_err(format!("load failed: {e}")))?;
        self.dim = state.codebook.dim;
        self.inner = GemRouter::new();
        self.inner.restore_state(state);
        Ok(())
    }

    /// Check if the router is built and ready for queries.
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    /// Number of indexed documents.
    fn n_docs(&self) -> usize {
        self.inner.n_docs()
    }

    /// Number of fine centroids in the codebook.
    fn n_fine(&self) -> usize {
        self.inner.n_fine()
    }

    /// Number of coarse clusters.
    fn n_coarse(&self) -> usize {
        self.inner.n_coarse()
    }

    /// Get per-document cluster overlap with a query's C_top.
    ///
    /// Returns: list of (doc_id, overlap_count)
    #[pyo3(signature = (query_vectors, n_probes = 4))]
    fn query_cluster_overlaps(
        &self,
        query_vectors: PyReadonlyArray2<f32>,
        n_probes: usize,
    ) -> PyResult<Vec<(u64, usize)>> {
        if !self.inner.is_ready() {
            return Err(PyValueError::new_err("router not built"));
        }
        let arr = query_vectors.as_array();
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;
        let n_query_vecs = arr.shape()[0];
        let (_, query_ctop) = self.inner
            .compute_query_profile(flat, n_query_vecs, n_probes)
            .ok_or_else(|| PyValueError::new_err("router state missing"))?;

        let state = self.inner.state().ok_or_else(|| {
            PyValueError::new_err("router state missing")
        })?;
        let results: Vec<(u64, usize)> = state.doc_profiles
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let overlap = codebook::cluster_overlap(&query_ctop, &p.ctop);
                (state.doc_ids[i], overlap)
            })
            .collect();
        Ok(results)
    }

    /// Train an adaptive cluster cutoff decision tree from query-positive pairs.
    ///
    /// Args:
    ///   training_queries: (Q_total, D) float32 matrix of all query vectors
    ///   n_query_vecs: list of ints — number of vectors per query
    ///   training_positives: list of internal doc indices (positive examples)
    ///   t: top centroids per query token (default 3)
    ///   r_max: maximum cluster cutoff to consider (default 8)
    ///   max_depth: decision tree depth limit (default 6)
    ///
    /// Returns: serialized tree as bytes (use with CutoffTree.from_bytes)
    #[pyo3(signature = (training_queries, n_query_vecs, training_positives, t = 3, r_max = 8, max_depth = 6))]
    fn train_adaptive_cutoff(
        &self,
        training_queries: PyReadonlyArray2<f32>,
        n_query_vecs: Vec<usize>,
        training_positives: Vec<usize>,
        t: usize,
        r_max: usize,
        max_depth: usize,
    ) -> PyResult<Vec<u8>> {
        if !self.inner.is_ready() {
            return Err(PyValueError::new_err("router not built"));
        }
        let arr = training_queries.as_array();
        let flat = arr.as_slice().ok_or_else(|| {
            PyValueError::new_err("array must be C-contiguous")
        })?;

        let tree = self.inner.train_adaptive_cutoff(
            flat,
            &n_query_vecs,
            &training_positives,
            t,
            r_max,
            max_depth,
        ).ok_or_else(|| PyValueError::new_err("training failed: no state or no pairs"))?;

        let bytes = tree.to_bytes().map_err(|e| {
            PyValueError::new_err(format!("serialization failed: {e}"))
        })?;
        Ok(bytes)
    }
}

#[pymodule]
fn latence_gem_router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGemRouter>()?;
    Ok(())
}
