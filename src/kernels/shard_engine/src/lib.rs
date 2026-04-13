#![allow(clippy::too_many_arguments)]

pub mod types;
pub mod topk;
pub mod wal;
pub mod mmap_reader;
pub mod state;
pub mod metadata;
pub mod simd_proxy;
pub mod centroid_approx;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};

use crate::mmap_reader::MmapShard;
use crate::state::{DocMeta, ShardState, StateHandle};
use crate::topk::heap_topk;
use crate::types::{DocId, ScoredDoc};
use crate::wal::{WalWriter, replay, tombstones_from_entries, WalOp};

/// GPU scoring callback type.
///
/// Python passes an `Option<Py<PyAny>>` callable that accepts:
///   (query: np.ndarray[float32, (n_q, dim)],
///    docs:  np.ndarray[float32, (n_d, dim)],
///    doc_ids: List[int],
///    offsets: List[Tuple[int, int]])
/// and returns List[float] of MaxSim scores.
pub type GpuScoreFn = dyn Fn(&[f32], &[f32], &[u64], &[(usize, usize)], usize) -> Vec<f32>
    + Send
    + Sync;

fn wrap_gpu_score_fn(py_fn: Py<PyAny>) -> Box<GpuScoreFn> {
    Box::new(
        move |query: &[f32],
              doc_vecs: &[f32],
              doc_ids: &[u64],
              offsets: &[(usize, usize)],
              dim: usize| {
            Python::with_gil(|py| {
                if dim == 0 || query.len() % dim != 0 || doc_vecs.len() % dim != 0 {
                    log::error!("GPU score fn: misaligned input dimensions");
                    return vec![f32::NEG_INFINITY; doc_ids.len()];
                }
                let n_q = query.len() / dim;
                let n_d = doc_vecs.len() / dim;

                let q_rows: Vec<Vec<f32>> = (0..n_q)
                    .map(|i| query[i * dim..(i + 1) * dim].to_vec())
                    .collect();
                let d_rows: Vec<Vec<f32>> = (0..n_d)
                    .map(|i| doc_vecs[i * dim..(i + 1) * dim].to_vec())
                    .collect();

                let q_arr = match PyArray2::from_vec2_bound(py, &q_rows) {
                    Ok(a) => a,
                    Err(e) => {
                        log::error!("GPU score fn: failed to create query array: {e}");
                        return vec![f32::NEG_INFINITY; doc_ids.len()];
                    }
                };
                let d_arr = match PyArray2::from_vec2_bound(py, &d_rows) {
                    Ok(a) => a,
                    Err(e) => {
                        log::error!("GPU score fn: failed to create doc array: {e}");
                        return vec![f32::NEG_INFINITY; doc_ids.len()];
                    }
                };

                let id_list: Vec<u64> = doc_ids.to_vec();
                let off_list: Vec<(usize, usize)> = offsets.to_vec();

                match py_fn.call1(py, (q_arr, d_arr, id_list, off_list)) {
                    Ok(result) => {
                        let list = result.bind(py);
                        match list.extract::<Vec<f32>>() {
                            Ok(scores) if scores.len() == doc_ids.len() => scores,
                            Ok(scores) => {
                                log::error!(
                                    "GPU score fn returned {} scores for {} docs",
                                    scores.len(),
                                    doc_ids.len()
                                );
                                vec![f32::NEG_INFINITY; doc_ids.len()]
                            }
                            Err(_) => {
                                log::error!("GPU score fn returned non-float list");
                                vec![f32::NEG_INFINITY; doc_ids.len()]
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("GPU score fn failed: {e}");
                        vec![f32::NEG_INFINITY; doc_ids.len()]
                    }
                }
            })
        },
    )
}

// ---------------------------------------------------------------
// Shared shard map — readable without the writer lock
// ---------------------------------------------------------------

/// Writer state protected by a mutex (only one writer at a time).
struct WriterInner {
    wal: Option<WalWriter>,
    #[allow(dead_code)]
    next_shard_id: u32,
}

/// Native shard engine with lock-free reads and WAL-backed writes.
///
/// Readers obtain an `Arc<ShardState>` + `Arc<HashMap<…, Arc<MmapShard>>>`
/// snapshot via `ArcSwap`, zero contention with writers.  Writers hold the
/// `Mutex<WriterInner>` only for WAL append + state swap.
#[pyclass]
pub struct ShardIndex {
    #[allow(dead_code)]
    base_dir: PathBuf,
    state: Arc<StateHandle>,
    shards: ArcSwap<HashMap<u32, Arc<MmapShard>>>,
    writer: Mutex<WriterInner>,
    dim: usize,
    gpu_score_fn: Option<Box<GpuScoreFn>>,
    centroids: Option<Arc<Vec<f32>>>,
    n_centroids: usize,
    closed: std::sync::atomic::AtomicBool,
}

impl ShardIndex {
    fn ensure_open(&self) -> PyResult<()> {
        if self.closed.load(std::sync::atomic::Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("ShardIndex is closed"));
        }
        Ok(())
    }

    fn build_state_from_docs(
        docs: &HashMap<DocId, DocMeta>,
        shard_count: u32,
    ) -> ShardState {
        let mut doc_ids: Vec<DocId> = docs.keys().copied().collect();
        doc_ids.sort_unstable();
        let total_vectors: u64 = docs.values().map(|d| (d.row_end - d.row_start) as u64).sum();
        ShardState {
            docs: docs.clone(),
            doc_ids,
            doc_means: None,
            doc_mean_dim: 0,
            doc_mean_ids: Vec::new(),
            shard_count,
            total_vectors,
        }
    }

    /// Load the shard map snapshot (lock-free).
    fn load_shards(&self) -> Arc<HashMap<u32, Arc<MmapShard>>> {
        self.shards.load_full()
    }
}

#[pymethods]
impl ShardIndex {
    /// Create or open a ShardIndex at `base_dir`.
    #[new]
    #[pyo3(signature = (base_dir, dim, gpu_score_fn=None))]
    fn new(base_dir: &str, dim: usize, gpu_score_fn: Option<Py<PyAny>>) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }

        let base = PathBuf::from(base_dir);
        std::fs::create_dir_all(&base)
            .map_err(|e| PyIOError::new_err(format!("cannot create base_dir: {e}")))?;

        let wal_path = base.join("shard.wal");
        let entries = replay(&wal_path)
            .map_err(|e| PyIOError::new_err(format!("WAL replay failed: {e}")))?;

        let tombstones = tombstones_from_entries(&entries);

        // Discover existing shard files FIRST so we know valid shard_ids
        let mut shard_map: HashMap<u32, Arc<MmapShard>> = HashMap::new();
        let mut next_shard_id: u32 = 0;
        for i in 0u32.. {
            let spath = base.join(format!("shard_{i}.safetensors"));
            if spath.exists() {
                match MmapShard::open(&spath) {
                    Ok(s) => { shard_map.insert(i, Arc::new(s)); }
                    Err(e) => log::warn!("Could not open shard {i}: {e}"),
                }
                next_shard_id = i + 1;
            } else {
                break;
            }
        }

        // Rebuild in-memory state from WAL
        let mut docs: HashMap<DocId, DocMeta> = HashMap::new();

        for entry in &entries {
            match entry.op {
                WalOp::Insert | WalOp::Upsert => {
                    if tombstones.contains(&entry.doc_id) {
                        continue;
                    }
                    let n_rows = entry.vec_rows as usize;
                    // WAL-buffered docs use shard_id u32::MAX sentinel
                    docs.insert(entry.doc_id, DocMeta {
                        shard_id: u32::MAX,
                        row_start: 0,
                        row_end: n_rows,
                        dim,
                        payload_json: entry.payload_json.clone(),
                    });
                }
                WalOp::Delete => {
                    docs.remove(&entry.doc_id);
                }
                WalOp::UpdatePayload => {
                    if let Some(meta) = docs.get_mut(&entry.doc_id) {
                        meta.payload_json = entry.payload_json.clone();
                    }
                }
            }
        }

        let shard_count = next_shard_id;
        let state = Self::build_state_from_docs(&docs, shard_count);
        let n_wal_entries = entries.len() as u64;

        let mut wal_writer = WalWriter::open(&wal_path)
            .map_err(|e| PyIOError::new_err(format!("WAL open failed: {e}")))?;
        wal_writer.set_n_entries(n_wal_entries);

        let gpu_closure = gpu_score_fn.map(wrap_gpu_score_fn);

        Ok(Self {
            base_dir: base,
            state: Arc::new(StateHandle::new(state)),
            shards: ArcSwap::from_pointee(shard_map),
            writer: Mutex::new(WriterInner {
                wal: Some(wal_writer),
                next_shard_id,
            }),
            dim,
            gpu_score_fn: gpu_closure,
            centroids: None,
            n_centroids: 0,
            closed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Number of documents in the index.
    #[getter]
    fn doc_count(&self) -> PyResult<usize> {
        self.ensure_open()?;
        Ok(self.state.load().doc_count())
    }

    /// Total vectors across all documents.
    #[getter]
    fn total_vectors(&self) -> PyResult<u64> {
        self.ensure_open()?;
        Ok(self.state.load().total_vectors)
    }

    /// Embedding dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    /// Insert a document with its multi-vector embeddings.
    #[pyo3(signature = (doc_id, embeddings, payload=None))]
    fn insert(
        &self,
        doc_id: u64,
        embeddings: PyReadonlyArray2<f32>,
        payload: Option<&str>,
    ) -> PyResult<()> {
        self.ensure_open()?;

        let shape = embeddings.shape();
        let n_rows = shape[0];
        let e_dim = shape[1];
        if e_dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "embedding dim {e_dim} != index dim {}",
                self.dim
            )));
        }

        let vec_data: Vec<f32> = embeddings.as_slice()
            .map_err(|_| PyValueError::new_err("embeddings must be contiguous"))?
            .to_vec();

        let pld_bytes = payload.map(|s| s.as_bytes().to_vec());

        let mut w = self.writer.lock();
        if let Some(ref mut wal) = w.wal {
            wal.log_insert(
                doc_id,
                &vec_data,
                n_rows as u32,
                e_dim as u32,
                pld_bytes.as_deref(),
            )
            .map_err(|e| PyIOError::new_err(format!("WAL write failed: {e}")))?;
        }

        // Snapshot update under writer lock to avoid concurrent state races
        let snap = self.state.load();
        let mut new_docs = snap.docs.clone();
        new_docs.insert(doc_id, DocMeta {
            shard_id: u32::MAX, // WAL-buffered sentinel
            row_start: 0,
            row_end: n_rows,
            dim: self.dim,
            payload_json: pld_bytes,
        });
        let new_state = Self::build_state_from_docs(&new_docs, snap.shard_count);
        self.state.store(new_state);

        Ok(())
    }

    /// Delete a document by ID.
    fn delete(&self, doc_id: u64) -> PyResult<bool> {
        self.ensure_open()?;

        let mut w = self.writer.lock();

        // Check existence under the lock to avoid TOCTOU
        let snap = self.state.load();
        if !snap.docs.contains_key(&doc_id) {
            return Ok(false);
        }

        if let Some(ref mut wal) = w.wal {
            wal.log_delete(doc_id)
                .map_err(|e| PyIOError::new_err(format!("WAL write failed: {e}")))?;
        }

        let mut new_docs = snap.docs.clone();
        new_docs.remove(&doc_id);
        let new_state = Self::build_state_from_docs(&new_docs, snap.shard_count);
        self.state.store(new_state);

        Ok(true)
    }

    /// Update the JSON payload for a document.
    #[pyo3(signature = (doc_id, payload))]
    fn update_payload(&self, doc_id: u64, payload: &str) -> PyResult<bool> {
        self.ensure_open()?;

        let pld_bytes = payload.as_bytes().to_vec();

        let mut w = self.writer.lock();

        let snap = self.state.load();
        if !snap.docs.contains_key(&doc_id) {
            return Ok(false);
        }

        if let Some(ref mut wal) = w.wal {
            wal.log_update_payload(doc_id, &pld_bytes)
                .map_err(|e| PyIOError::new_err(format!("WAL write failed: {e}")))?;
        }

        let mut new_docs = snap.docs.clone();
        if let Some(meta) = new_docs.get_mut(&doc_id) {
            meta.payload_json = Some(pld_bytes);
        }
        let new_state = Self::build_state_from_docs(&new_docs, snap.shard_count);
        self.state.store(new_state);

        Ok(true)
    }

    /// Search for the top-k documents most similar to a query.
    ///
    /// Returns (doc_ids, scores) — each as a 1-D numpy array.
    /// GIL is released during all pure-Rust work (mmap reads, CPU MaxSim, top-k).
    #[pyo3(signature = (query, k=10))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f32>>)> {
        self.ensure_open()?;

        let shape = query.shape();
        let n_q = shape[0];
        let q_dim = shape[1];
        if q_dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {q_dim} != index dim {}",
                self.dim
            )));
        }

        // Phase 1 (GIL held): extract owned data from numpy
        let q_data: Vec<f32> = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be contiguous"))?
            .to_vec();

        // Load snapshots (lock-free)
        let snap = self.state.load();
        let shard_snap = self.load_shards();

        if snap.doc_count() == 0 {
            let empty_u64: Vec<u64> = vec![];
            let empty_f32: Vec<f32> = vec![];
            return Ok((
                PyArray1::from_slice_bound(py, &empty_u64),
                PyArray1::from_slice_bound(py, &empty_f32),
            ));
        }

        let dim = self.dim;
        let has_gpu = self.gpu_score_fn.is_some();

        // Phase 2 (GIL released): all Rust work — mmap reads, data collection
        let (all_doc_vecs, doc_ids, offsets) = py.allow_threads(|| {
            let mut all_doc_vecs: Vec<f32> = Vec::new();
            let mut doc_ids: Vec<u64> = Vec::new();
            let mut offsets: Vec<(usize, usize)> = Vec::new();
            let mut cursor: usize = 0;

            for (&did, meta) in &snap.docs {
                if meta.shard_id == u32::MAX {
                    continue;
                }
                if let Some(shard) = shard_snap.get(&meta.shard_id) {
                    match shard.read_selected_f32(
                        "embeddings",
                        &[(meta.row_start, meta.row_end)],
                    ) {
                        Ok(vecs) => {
                            let n_vecs = meta.row_end - meta.row_start;
                            doc_ids.push(did);
                            offsets.push((cursor, cursor + n_vecs));
                            cursor += n_vecs;
                            all_doc_vecs.extend_from_slice(&vecs);
                        }
                        Err(e) => {
                            log::warn!("Failed to read doc {did} from shard {}: {e}", meta.shard_id);
                        }
                    }
                }
            }
            (all_doc_vecs, doc_ids, offsets)
        });

        if doc_ids.is_empty() {
            let empty_u64: Vec<u64> = vec![];
            let empty_f32: Vec<f32> = vec![];
            return Ok((
                PyArray1::from_slice_bound(py, &empty_u64),
                PyArray1::from_slice_bound(py, &empty_f32),
            ));
        }

        // GPU path: re-acquire GIL for Python callback, then release for top-k
        if has_gpu {
            if let Some(ref gpu_fn) = self.gpu_score_fn {
                let scores = gpu_fn(&q_data, &all_doc_vecs, &doc_ids, &offsets, dim);

                // Phase 3 (GIL released): top-k extraction
                let (ids, scs) = py.allow_threads(|| {
                    let scored: Vec<ScoredDoc> = doc_ids
                        .iter()
                        .zip(scores.iter())
                        .map(|(&id, &sc)| ScoredDoc { doc_id: id, score: sc })
                        .collect();
                    let topk = heap_topk(scored.into_iter(), k);
                    let ids: Vec<u64> = topk.iter().map(|d| d.doc_id).collect();
                    let scs: Vec<f32> = topk.iter().map(|d| d.score).collect();
                    (ids, scs)
                });

                return Ok((
                    PyArray1::from_slice_bound(py, &ids),
                    PyArray1::from_slice_bound(py, &scs),
                ));
            }
        }

        // CPU fallback (GIL released): dot-product MaxSim + top-k
        let (ids, scs) = py.allow_threads(|| {
            let mut scored_docs: Vec<ScoredDoc> = Vec::with_capacity(doc_ids.len());
            for (i, &did) in doc_ids.iter().enumerate() {
                let (off_start, off_end) = offsets[i];
                let doc_vecs = &all_doc_vecs[off_start * dim..off_end * dim];
                let score = cpu_maxsim(&q_data, doc_vecs, dim, n_q);
                scored_docs.push(ScoredDoc { doc_id: did, score });
            }
            let topk = heap_topk(scored_docs.into_iter(), k);
            let ids: Vec<u64> = topk.iter().map(|d| d.doc_id).collect();
            let scs: Vec<f32> = topk.iter().map(|d| d.score).collect();
            (ids, scs)
        });

        Ok((
            PyArray1::from_slice_bound(py, &ids),
            PyArray1::from_slice_bound(py, &scs),
        ))
    }

    /// SIMD-accelerated centroid proxy scoring.
    ///
    /// Computes `max_i(dot(query_token_i, doc_mean_d))` for each candidate
    /// and returns the top-n doc IDs by proxy score. GIL released during
    /// the Rust computation.
    #[pyo3(signature = (query, candidate_ids=None, n_full=4096))]
    fn proxy_score<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<f32>,
        candidate_ids: Option<Vec<u64>>,
        n_full: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f32>>)> {
        self.ensure_open()?;

        let shape = query.shape();
        let q_dim = shape[1];
        if q_dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {q_dim} != index dim {}",
                self.dim
            )));
        }

        let q_data: Vec<f32> = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be contiguous"))?
            .to_vec();

        // Grab an Arc to the snapshot — zero-copy, keeps data alive in allow_threads
        let snap = self.state.load();
        if snap.doc_means.is_none() {
            return Err(PyRuntimeError::new_err(
                "doc_means not loaded — call set_doc_means first",
            ));
        }
        let dim = snap.doc_mean_dim;
        if dim == 0 {
            return Err(PyRuntimeError::new_err("doc_mean_dim is 0"));
        }
        if dim != q_dim {
            return Err(PyValueError::new_err(format!(
                "doc_mean_dim {dim} != query dim {q_dim}",
            )));
        }
        let candidates = candidate_ids.unwrap_or_default();

        let (ids, scores) = py.allow_threads(move || {
            let doc_means = snap.doc_means.as_deref().unwrap();
            let doc_mean_ids = &snap.doc_mean_ids;
            simd_proxy::proxy_score_topn_with_scores(
                &q_data,
                doc_means,
                doc_mean_ids,
                &candidates,
                dim,
                n_full,
            )
        });

        Ok((
            PyArray1::from_slice_bound(py, &ids),
            PyArray1::from_slice_bound(py, &scores),
        ))
    }

    /// Set the mean-pooled document embeddings for proxy scoring.
    ///
    /// `means` shape: `(n_docs, dim)`, `doc_ids` length: `n_docs`.
    #[pyo3(signature = (means, doc_ids))]
    fn set_doc_means(
        &self,
        means: PyReadonlyArray2<f32>,
        doc_ids: Vec<u64>,
    ) -> PyResult<()> {
        self.ensure_open()?;

        let shape = means.shape();
        let n_docs = shape[0];
        let dim = shape[1];

        if dim == 0 {
            return Err(PyValueError::new_err("means dim must be > 0"));
        }
        if doc_ids.len() != n_docs {
            return Err(PyValueError::new_err(format!(
                "doc_ids len {} != means rows {n_docs}",
                doc_ids.len()
            )));
        }

        let flat: Vec<f32> = means
            .as_slice()
            .map_err(|_| PyValueError::new_err("means must be contiguous"))?
            .to_vec();

        // Hold writer lock to prevent race with insert/delete
        let _w = self.writer.lock();
        let snap = self.state.load();
        let mut new_state = Self::build_state_from_docs(&snap.docs, snap.shard_count);
        new_state.doc_means = Some(flat);
        new_state.doc_mean_dim = dim;
        new_state.doc_mean_ids = doc_ids;
        self.state.store(new_state);

        Ok(())
    }

    /// Register documents that already exist in shard files.
    ///
    /// Used when loading a pre-built index (e.g. from ShardSegmentManager).
    #[pyo3(signature = (doc_ids, shard_ids, row_starts, row_ends))]
    fn register_shard_docs(
        &self,
        doc_ids: Vec<u64>,
        shard_ids: Vec<u32>,
        row_starts: Vec<usize>,
        row_ends: Vec<usize>,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let n = doc_ids.len();
        if shard_ids.len() != n || row_starts.len() != n || row_ends.len() != n {
            return Err(PyValueError::new_err("all arrays must have the same length"));
        }

        let _w = self.writer.lock();
        let snap = self.state.load();
        let mut new_docs = snap.docs.clone();
        for i in 0..n {
            new_docs.insert(doc_ids[i], DocMeta {
                shard_id: shard_ids[i],
                row_start: row_starts[i],
                row_end: row_ends[i],
                dim: self.dim,
                payload_json: None,
            });
        }
        let max_registered = shard_ids.iter().copied().max().unwrap_or(0);
        let shard_count = snap.shard_count.max(max_registered + 1);
        let new_state = Self::build_state_from_docs(&new_docs, shard_count);
        self.state.store(new_state);
        Ok(())
    }

    /// Load centroid embeddings for approximate scoring.
    ///
    /// `centroids` shape: `(n_centroids, dim)`.
    #[pyo3(signature = (centroids,))]
    fn set_centroids(
        &mut self,
        centroids: PyReadonlyArray2<f32>,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let shape = centroids.shape();
        let n_c = shape[0];
        let c_dim = shape[1];
        if c_dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "centroid dim {c_dim} != index dim {}", self.dim
            )));
        }
        if n_c == 0 {
            return Err(PyValueError::new_err("centroids must have at least 1 row"));
        }
        let flat: Vec<f32> = centroids
            .as_slice()
            .map_err(|_| PyValueError::new_err("centroids must be contiguous"))?
            .to_vec();
        self.centroids = Some(Arc::new(flat));
        self.n_centroids = n_c;
        log::info!("Loaded {} centroids (dim={})", n_c, c_dim);
        Ok(())
    }

    /// Approximate MaxSim scoring using centroid codes (Rayon-parallel).
    ///
    /// Pre-filter stage: scores `candidate_ids` using lightweight centroid
    /// code lookups and returns the top `n_top` doc IDs + scores.
    /// GIL is released during all Rust work.
    #[pyo3(signature = (query, candidate_ids, n_top=512))]
    fn score_candidates_approx<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<f32>,
        candidate_ids: Vec<u64>,
        n_top: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f32>>)> {
        self.ensure_open()?;
        let centroids = self.centroids.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("centroids not loaded — call set_centroids first")
        })?;
        let shape = query.shape();
        let q_dim = shape[1];
        if q_dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {q_dim} != index dim {}", self.dim
            )));
        }
        let q_data: Vec<f32> = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be contiguous"))?
            .to_vec();

        let snap = self.state.load();
        let shard_snap = self.load_shards();
        let n_centroids = self.n_centroids;
        let dim = self.dim;
        let centroids_arc = Arc::clone(centroids);

        let (ids, scs) = py.allow_threads(move || {
            let results = centroid_approx::score_candidates_approx(
                &q_data,
                &centroids_arc,
                n_centroids,
                dim,
                &candidate_ids,
                &snap,
                &shard_snap,
                n_top,
            );
            let ids: Vec<u64> = results.iter().map(|d| d.doc_id).collect();
            let scs: Vec<f32> = results.iter().map(|d| d.score).collect();
            (ids, scs)
        });

        Ok((
            PyArray1::from_slice_bound(py, &ids),
            PyArray1::from_slice_bound(py, &scs),
        ))
    }

    /// Flush the WAL and sync to disk.
    fn sync(&self) -> PyResult<()> {
        self.ensure_open()?;
        let mut w = self.writer.lock();
        if let Some(ref mut wal) = w.wal {
            wal.sync()
                .map_err(|e| PyIOError::new_err(format!("WAL sync failed: {e}")))?;
        }
        Ok(())
    }

    /// Get all document IDs currently in the index (sorted).
    fn doc_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u64>>> {
        self.ensure_open()?;
        let snap = self.state.load();
        Ok(PyArray1::from_slice_bound(py, &snap.doc_ids))
    }

    /// Check if a document exists.
    fn contains(&self, doc_id: u64) -> PyResult<bool> {
        self.ensure_open()?;
        Ok(self.state.load().docs.contains_key(&doc_id))
    }

    /// Get the payload JSON for a document, or None.
    fn get_payload(&self, doc_id: u64) -> PyResult<Option<String>> {
        self.ensure_open()?;
        let snap = self.state.load();
        match snap.docs.get(&doc_id) {
            Some(meta) => {
                match &meta.payload_json {
                    Some(bytes) => {
                        let s = String::from_utf8(bytes.clone())
                            .map_err(|e| PyValueError::new_err(format!("payload not valid UTF-8: {e}")))?;
                        Ok(Some(s))
                    }
                    None => Ok(None),
                }
            }
            None => Ok(None),
        }
    }

    /// Close the index, flushing WAL.
    fn close(&mut self) -> PyResult<()> {
        if self
            .closed
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            return Ok(());
        }
        let mut w = self.writer.lock();
        if let Some(ref mut wal) = w.wal {
            wal.close()
                .map_err(|e| PyIOError::new_err(format!("WAL close failed: {e}")))?;
        }
        w.wal = None;
        // Replace with empty — in-flight searches still hold their Arc snapshots
        self.shards.store(Arc::new(HashMap::new()));
        self.state.store(ShardState::empty());
        Ok(())
    }
}

/// CPU MaxSim: Σ_i max_j dot(q_i, d_j) — uses SIMD-accelerated dot product.
fn cpu_maxsim(query: &[f32], doc: &[f32], dim: usize, n_q: usize) -> f32 {
    let n_d = doc.len() / dim;
    if n_d == 0 || n_q == 0 || dim == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for qi in 0..n_q {
        let q_start = qi * dim;
        let q_row = &query[q_start..q_start + dim];
        let mut max_dot = f32::NEG_INFINITY;
        for dj in 0..n_d {
            let d_start = dj * dim;
            let d_row = &doc[d_start..d_start + dim];
            let dot = simd_proxy::dot_product(q_row, d_row);
            if dot > max_dot {
                max_dot = dot;
            }
        }
        total += max_dot;
    }
    total
}

// ---------------------------------------------------------------
// Python module
// ---------------------------------------------------------------

#[pymodule]
fn latence_shard_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ShardIndex>()?;
    m.add_class::<metadata::MetadataStore>()?;
    Ok(())
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_maxsim() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let d = vec![1.0, 0.0];
        assert!((cpu_maxsim(&q, &d, 2, 2) - 1.0).abs() < 1e-6);

        let q2 = vec![1.0, 0.0];
        let d2 = vec![1.0, 0.0, 0.0, 1.0];
        assert!((cpu_maxsim(&q2, &d2, 2, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_maxsim_identity() {
        let v = vec![1.0, 0.0, 0.0, 1.0];
        assert!((cpu_maxsim(&v, &v, 2, 2) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_maxsim_empty() {
        assert_eq!(cpu_maxsim(&[], &[], 2, 0), 0.0);
        assert_eq!(cpu_maxsim(&[1.0, 0.0], &[], 2, 1), 0.0);
    }
}
