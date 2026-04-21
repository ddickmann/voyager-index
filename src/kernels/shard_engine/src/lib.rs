#![allow(clippy::too_many_arguments)]

pub mod types;
pub mod topk;
pub mod wal;
pub mod mmap_reader;
pub mod state;
pub mod metadata;
pub mod simd_proxy;
pub mod centroid_approx;
pub mod codec;
pub mod merged_mmap;
pub mod fused_maxsim;
pub mod fused_rroq158;
pub mod fused_rroq4_riem;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, PyArrayMethods};

use crate::codec::ResidualCodec;
use crate::merged_mmap::MergedMmap;
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

/// Send + Sync raw-pointer slice. Used to thread numpy buffers across
/// `py.allow_threads` without copying. The numpy crate's `PyReadonlyArray`
/// wrapper borrows from the GIL token via a non-`Send` `&[T]` slice; we
/// strip that lifetime by capturing only the raw pointer + length, and
/// reconstitute the slice inside the GIL-released closure.
///
/// SAFETY contract for callers:
///   1. The pointer must outlive every `as_slice()` call. In practice this
///      means the originating PyReadonlyArray handle must remain alive on
///      the parent stack frame for the entire duration of the closure
///      (true for the rroq158 binding because the handles are stack
///      locals scoped to the pyfunction body).
///   2. No other thread (Python or otherwise) may mutate the underlying
///      buffer between the slice construction and the closure return.
///      `PyReadonlyArrayN` enforces this for numpy: it requires the array
///      to be non-writable for the lifetime of the borrow, which numpy
///      verifies internally.
struct SendSlice<T> {
    ptr: *const T,
    len: usize,
}

impl<T> SendSlice<T> {
    #[inline]
    fn from_slice(s: &[T]) -> Self {
        Self { ptr: s.as_ptr(), len: s.len() }
    }

    /// SAFETY: see struct docstring; the caller must uphold the lifetime
    /// + non-mutation invariants.
    #[inline]
    unsafe fn as_slice<'a>(&self) -> &'a [T] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }
}

// SAFETY: SendSlice is just a raw pointer + length; `Send + Sync` for
// `T: Send + Sync` matches the safety invariant on callers (no concurrent
// mutation of the pointee). PyO3's `PyReadonlyArray` upholds this by
// statically forbidding writes to the borrowed numpy buffer.
unsafe impl<T: Send> Send for SendSlice<T> {}
unsafe impl<T: Sync> Sync for SendSlice<T> {}

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
    codec: Option<Arc<ResidualCodec>>,
    merged: Option<Arc<MergedMmap>>,
    use_compression: bool,
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
            codec: None,
            merged: None,
            use_compression: false,
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
        let q_data: &[f32] = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be contiguous"))?;

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

        let q_data: &[f32] = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be contiguous"))?;

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

    /// Set up the residual codec for compressed fetch/decompression.
    ///
    /// Must be called AFTER `set_centroids`. `bucket_weights` is the 1-D array
    /// of representative values for each bucket (length = 2^nbits).
    #[pyo3(signature = (bucket_weights, nbits))]
    fn set_codec(
        &mut self,
        bucket_weights: Vec<f32>,
        nbits: usize,
    ) -> PyResult<()> {
        self.ensure_open()?;
        let centroids = self.centroids.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("set_centroids must be called before set_codec")
        })?;
        if nbits == 0 || 8 % nbits != 0 {
            return Err(PyValueError::new_err(format!("nbits must divide 8, got {nbits}")));
        }
        let n_buckets = 1 << nbits;
        if bucket_weights.len() != n_buckets {
            return Err(PyValueError::new_err(format!(
                "bucket_weights len {} != 2^nbits = {n_buckets}", bucket_weights.len()
            )));
        }
        let codec = ResidualCodec::new(
            Arc::clone(centroids),
            self.n_centroids,
            self.dim,
            bucket_weights,
            nbits,
        );
        self.codec = Some(Arc::new(codec));
        log::info!("Residual codec ready: nbits={nbits}, packed_dim={}", self.dim * nbits / 8);
        Ok(())
    }

    /// Load merged flat mmap files for zero-copy document access.
    ///
    /// `dir` should contain merged_embeddings.bin, merged_offsets.bin,
    /// merged_doc_map.bin, and optionally merged_codes.bin.
    #[pyo3(signature = (dir,))]
    fn load_merged(&mut self, dir: &str) -> PyResult<()> {
        self.ensure_open()?;
        let path = std::path::Path::new(dir);
        let mm = MergedMmap::load(path)
            .map_err(|e| PyIOError::new_err(format!("failed to load merged mmap: {e}")))?;
        log::info!(
            "Merged mmap loaded: {} docs, dim={}, has_codes={}",
            mm.n_docs(),
            mm.dim(),
            mm.has_codes(),
        );
        self.merged = Some(Arc::new(mm));
        Ok(())
    }

    /// Enable or disable residual compression for fetch_candidate_embeddings.
    ///
    /// When false (default), raw FP16 embeddings are read — lossless.
    /// When true, packed residuals are decompressed via the codec — lossy but
    /// smaller I/O.
    #[pyo3(signature = (enabled,))]
    fn set_use_compression(&mut self, enabled: bool) -> PyResult<()> {
        self.ensure_open()?;
        self.use_compression = enabled;
        log::info!("use_compression = {}", enabled);
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
        let q_data: &[f32] = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be contiguous"))?;

        let snap = self.state.load();
        let shard_snap = self.load_shards();
        let n_centroids = self.n_centroids;
        let dim = self.dim;
        let centroids_arc = Arc::clone(centroids);
        let merged_arc = self.merged.as_ref().map(Arc::clone);

        let (ids, scs) = py.allow_threads(move || {
            let results = centroid_approx::score_candidates_approx(
                q_data,
                &centroids_arc,
                n_centroids,
                dim,
                &candidate_ids,
                &snap,
                &shard_snap,
                merged_arc.as_deref(),
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

    /// Fused exact MaxSim scoring over merged mmap — no Python staging.
    ///
    /// Reads FP16 embeddings directly from the merged mmap, converts to f32
    /// on the fly, computes MaxSim per candidate, and returns top-k results.
    /// GIL is released during all Rust work.
    #[pyo3(signature = (query, candidate_ids, k=10))]
    fn score_candidates_exact<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<f32>,
        candidate_ids: Vec<u64>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f32>>)> {
        self.ensure_open()?;
        let merged = self.merged.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("merged mmap not loaded — call load_merged first")
        })?;
        let shape = query.shape();
        let q_dim = shape[1];
        if q_dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {q_dim} != index dim {}", self.dim
            )));
        }
        let q_data: &[f32] = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be contiguous"))?;

        let dim = self.dim;
        let merged_arc = Arc::clone(merged);

        let (ids, scs) = py.allow_threads(move || {
            let topk = fused_maxsim::fused_maxsim_topk(
                q_data,
                &candidate_ids,
                &merged_arc,
                dim,
                k,
            );
            let ids: Vec<u64> = topk.iter().map(|d| d.doc_id).collect();
            let scs: Vec<f32> = topk.iter().map(|d| d.score).collect();
            (ids, scs)
        });

        Ok((
            PyArray1::from_slice_bound(py, &ids),
            PyArray1::from_slice_bound(py, &scs),
        ))
    }

    /// Fetch FP16 embeddings for candidate documents from mmap'd shards.
    ///
    /// Returns `(flat_embeddings_u8, offsets, doc_ids)` where flat_embeddings_u8
    /// is the raw bytes of FP16 data (caller reinterprets as float16),
    /// offsets is `(n_docs, 2)` with `[start, end)` row indices,
    /// and doc_ids is the ordered list.
    ///
    /// When a residual codec is configured AND shards contain `packed_residuals`,
    /// reads compressed data and decompresses to f16 (7.5x less I/O for nbits=2).
    /// Falls back to raw FP16 embedding reads otherwise.
    /// GIL is released during the Rust I/O work.
    #[pyo3(signature = (candidate_ids,))]
    fn fetch_candidate_embeddings<'py>(
        &self,
        py: Python<'py>,
        candidate_ids: Vec<u64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray2<i64>>,
        Bound<'py, PyArray1<u64>>,
    )> {
        self.ensure_open()?;
        let snap = self.state.load();
        let shard_snap = self.load_shards();
        let dim = self.dim;
        let codec_opt = if self.use_compression {
            self.codec.as_ref().map(Arc::clone)
        } else {
            None
        };
        let merged_arc = self.merged.as_ref().map(Arc::clone);

        let (flat_bytes, offsets_vec, out_ids) = py.allow_threads(move || {
            let bytes_per_row_fp16 = dim * 2;
            let mut flat: Vec<u8> = Vec::new();
            let mut offsets: Vec<[i64; 2]> = Vec::new();
            let mut ids: Vec<u64> = Vec::new();
            let mut pos: i64 = 0;

            // Fast path: merged mmap available
            if let Some(ref mm) = merged_arc {
                for &did in &candidate_ids {
                    if let Some(raw) = mm.get_embeddings_f16_bytes(did) {
                        let n_tokens = raw.len() / bytes_per_row_fp16;
                        if n_tokens == 0 {
                            continue;
                        }
                        flat.extend_from_slice(raw);
                        offsets.push([pos, pos + n_tokens as i64]);
                        ids.push(did);
                        pos += n_tokens as i64;
                    }
                }
                return (flat, offsets, ids);
            }

            let mut by_shard: HashMap<u32, Vec<(u64, usize, usize)>> = HashMap::new();
            for &did in &candidate_ids {
                if let Some(meta) = snap.docs.get(&did) {
                    if meta.shard_id != u32::MAX {
                        by_shard
                            .entry(meta.shard_id)
                            .or_default()
                            .push((did, meta.row_start, meta.row_end));
                    }
                }
            }

            for (sid, docs) in &by_shard {
                let shard = match shard_snap.get(sid) {
                    Some(s) => s,
                    None => continue,
                };

                let use_codec = codec_opt.is_some()
                    && shard.has_tensor("packed_residuals")
                    && shard.has_tensor("centroid_codes");

                for &(did, rs, re) in docs {
                    let n_tokens = re - rs;
                    if n_tokens == 0 {
                        continue;
                    }

                    if use_codec {
                        let codec = codec_opt.as_ref().unwrap();
                        let codes_res = shard.read_selected_u16("centroid_codes", &[(rs, re)]);
                        let packed_res = shard.read_rows_raw("packed_residuals", rs, re);
                        if let (Ok(codes), Ok(packed_bytes)) = (codes_res, packed_res) {
                            let decompressed = codec.decompress_to_f16(&codes, packed_bytes, n_tokens);
                            flat.extend_from_slice(&decompressed);
                            offsets.push([pos, pos + n_tokens as i64]);
                            ids.push(did);
                            pos += n_tokens as i64;
                        }
                    } else if let Ok(raw) = shard.read_rows_raw("embeddings", rs, re) {
                        let actual_tokens = raw.len() / bytes_per_row_fp16;
                        flat.extend_from_slice(raw);
                        offsets.push([pos, pos + actual_tokens as i64]);
                        ids.push(did);
                        pos += actual_tokens as i64;
                    }
                }
            }
            (flat, offsets, ids)
        });

        let raw_arr = PyArray1::from_vec_bound(py, flat_bytes);

        let n_docs = offsets_vec.len();
        let offsets_flat: Vec<i64> = offsets_vec.into_iter().flat_map(|o| o).collect();
        let off_1d = PyArray1::from_vec_bound(py, offsets_flat);
        let offsets_arr = off_1d
            .reshape([n_docs, 2])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape offsets: {e}")))?;

        let ids_arr = PyArray1::from_slice_bound(py, &out_ids);

        Ok((raw_arr, offsets_arr, ids_arr))
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

// ---------------------------------------------------------------
// rroq158 fused CPU scorer (free function)
// ---------------------------------------------------------------

/// Score a batch of (query × document) pairs with the rroq158 fused MaxSim
/// kernel on CPU. Mirrors `voyager_index._internal.kernels.triton_roq_rroq158
/// .roq_maxsim_rroq158` exactly so per-token scores are bitwise-identical
/// (ignoring float-rounding) to the Triton GPU kernel and the python reference.
///
/// All bit-plane tensors must be `int32`; all float tensors must be `float32`.
///
/// Returns a flat `(A * B,)` `float32` numpy array of scores; the python
/// caller reshapes and runs top-k.
#[pyfunction]
#[pyo3(signature = (
    q_planes, q_meta, qc_table,
    docs_sign, docs_nz, docs_scl, docs_cid, docs_cos, docs_sin,
    big_a, big_b, big_s, big_t,
    n_words, n_groups, query_bits, big_k,
    q_mask = None, docs_mask = None, n_threads = None,
))]
#[allow(clippy::too_many_arguments)]
fn rroq158_score_batch<'py>(
    py: Python<'py>,
    q_planes: PyReadonlyArray1<i32>,
    q_meta: PyReadonlyArray1<f32>,
    qc_table: PyReadonlyArray1<f32>,
    docs_sign: PyReadonlyArray1<i32>,
    docs_nz: PyReadonlyArray1<i32>,
    docs_scl: PyReadonlyArray1<f32>,
    docs_cid: PyReadonlyArray1<i32>,
    docs_cos: PyReadonlyArray1<f32>,
    docs_sin: PyReadonlyArray1<f32>,
    big_a: usize,
    big_b: usize,
    big_s: usize,
    big_t: usize,
    n_words: usize,
    n_groups: usize,
    query_bits: usize,
    big_k: usize,
    q_mask: Option<PyReadonlyArray1<f32>>,
    docs_mask: Option<PyReadonlyArray1<f32>>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    // Shape sanity checks — we treat the inputs as flat row-major buffers
    // and rely on the python caller to pass contiguous arrays in the right
    // logical shape.
    if n_groups == 0 {
        return Err(PyValueError::new_err("n_groups must be > 0"));
    }
    if n_words == 0 {
        return Err(PyValueError::new_err("n_words must be > 0"));
    }
    if big_a == 0 || big_b == 0 || big_s == 0 || big_t == 0 || big_k == 0 || query_bits == 0 {
        return Err(PyValueError::new_err(format!(
            "All shape dims must be > 0 (got A={big_a}, B={big_b}, S={big_s}, \
             T={big_t}, K={big_k}, query_bits={query_bits})"
        )));
    }
    if query_bits > 8 {
        return Err(PyValueError::new_err(format!(
            "query_bits must be <= 8 (got {query_bits}); the inner accumulator \
             uses (1 << k) which would overflow for k >= 31"
        )));
    }
    let group_words = n_words / n_groups;
    if group_words * n_groups != n_words {
        return Err(PyValueError::new_err(format!(
            "n_words ({n_words}) must be divisible by n_groups ({n_groups})"
        )));
    }
    let exp_q_planes = big_a * big_s * query_bits * n_words;
    let exp_q_meta = big_a * big_s * 2;
    let exp_qc = big_a * big_s * big_k;
    let exp_d_bp = big_b * big_t * n_words;
    let exp_d_scl = big_b * big_t * n_groups;
    let exp_d_cid = big_b * big_t;
    let exp_d_norm = big_b * big_t;

    let q_planes_slice = q_planes
        .as_slice()
        .map_err(|_| PyValueError::new_err("q_planes must be contiguous"))?;
    if q_planes_slice.len() != exp_q_planes {
        return Err(PyValueError::new_err(format!(
            "q_planes len {} != A*S*query_bits*n_words {}",
            q_planes_slice.len(),
            exp_q_planes
        )));
    }
    let q_meta_slice = q_meta
        .as_slice()
        .map_err(|_| PyValueError::new_err("q_meta must be contiguous"))?;
    if q_meta_slice.len() != exp_q_meta {
        return Err(PyValueError::new_err(format!(
            "q_meta len {} != A*S*2 {}",
            q_meta_slice.len(),
            exp_q_meta
        )));
    }
    let qc_table_slice = qc_table
        .as_slice()
        .map_err(|_| PyValueError::new_err("qc_table must be contiguous"))?;
    if qc_table_slice.len() != exp_qc {
        return Err(PyValueError::new_err(format!(
            "qc_table len {} != A*S*K {}",
            qc_table_slice.len(),
            exp_qc
        )));
    }
    let docs_sign_slice = docs_sign
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_sign must be contiguous"))?;
    if docs_sign_slice.len() != exp_d_bp {
        return Err(PyValueError::new_err(format!(
            "docs_sign len {} != B*T*n_words {}",
            docs_sign_slice.len(),
            exp_d_bp
        )));
    }
    let docs_nz_slice = docs_nz
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_nz must be contiguous"))?;
    if docs_nz_slice.len() != exp_d_bp {
        return Err(PyValueError::new_err(format!(
            "docs_nz len {} != B*T*n_words {}",
            docs_nz_slice.len(),
            exp_d_bp
        )));
    }
    let docs_scl_slice = docs_scl
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_scl must be contiguous"))?;
    if docs_scl_slice.len() != exp_d_scl {
        return Err(PyValueError::new_err(format!(
            "docs_scl len {} != B*T*n_groups {}",
            docs_scl_slice.len(),
            exp_d_scl
        )));
    }
    let docs_cid_slice = docs_cid
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_cid must be contiguous"))?;
    if docs_cid_slice.len() != exp_d_cid {
        return Err(PyValueError::new_err(format!(
            "docs_cid len {} != B*T {}",
            docs_cid_slice.len(),
            exp_d_cid
        )));
    }
    let docs_cos_slice = docs_cos
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_cos must be contiguous"))?;
    if docs_cos_slice.len() != exp_d_norm {
        return Err(PyValueError::new_err(format!(
            "docs_cos len {} != B*T {}",
            docs_cos_slice.len(),
            exp_d_norm
        )));
    }
    let docs_sin_slice = docs_sin
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_sin must be contiguous"))?;
    if docs_sin_slice.len() != exp_d_norm {
        return Err(PyValueError::new_err(format!(
            "docs_sin len {} != B*T {}",
            docs_sin_slice.len(),
            exp_d_norm
        )));
    }
    let q_mask_slice_opt = q_mask
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()
        .map_err(|_| PyValueError::new_err("q_mask must be contiguous"))?;
    if let Some(qm) = q_mask_slice_opt {
        if qm.len() != big_a * big_s {
            return Err(PyValueError::new_err(format!(
                "q_mask len {} != A*S {}",
                qm.len(),
                big_a * big_s
            )));
        }
    }
    let d_mask_slice_opt = docs_mask
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()
        .map_err(|_| PyValueError::new_err("docs_mask must be contiguous"))?;
    if let Some(dm) = d_mask_slice_opt {
        if dm.len() != big_b * big_t {
            return Err(PyValueError::new_err(format!(
                "docs_mask len {} != B*T {}",
                dm.len(),
                big_b * big_t
            )));
        }
    }

    // Build Send + Sync raw-pointer aliases for every input slice so we can
    // drop the GIL inside `py.allow_threads` without copying ~16 MB/query of
    // numpy buffers. The numpy crate's PyReadonlyArray handles keep the
    // Python objects alive for the entire scope of this function (they are
    // owned by stack locals on this PyO3 callstack), so the underlying
    // pointers remain valid across the GIL release.
    //
    // SAFETY rationale (see SendSlice docstring): callers must not mutate
    // the input numpy arrays from any other Python thread for the duration
    // of this call. PyReadonlyArrayN enforces this at construction by
    // requiring the array to be non-writeable (it borrows immutably; numpy
    // panics if the buffer is later modified through a different alias
    // while this borrow is live). This is standard PyO3 + numpy practice
    // for native extensions that release the GIL.
    let qp_send = SendSlice::from_slice(q_planes_slice);
    let qm_send = SendSlice::from_slice(q_meta_slice);
    let qc_send = SendSlice::from_slice(qc_table_slice);
    let ds_send = SendSlice::from_slice(docs_sign_slice);
    let dn_send = SendSlice::from_slice(docs_nz_slice);
    let dscl_send = SendSlice::from_slice(docs_scl_slice);
    let dcid_send = SendSlice::from_slice(docs_cid_slice);
    let dcos_send = SendSlice::from_slice(docs_cos_slice);
    let dsin_send = SendSlice::from_slice(docs_sin_slice);
    let qmask_send = q_mask_slice_opt.map(SendSlice::from_slice);
    let dmask_send = d_mask_slice_opt.map(SendSlice::from_slice);

    let scores = py.allow_threads(move || {
        // SAFETY: each `as_slice()` reconstitutes the borrow inside this
        // GIL-released closure. The lifetime is bounded by `move` capture
        // semantics; the underlying numpy buffers are kept alive by the
        // PyReadonlyArrayN handles still owned on the parent stack frame
        // (they are dropped only after this closure returns).
        let q_planes_v = unsafe { qp_send.as_slice() };
        let q_meta_v = unsafe { qm_send.as_slice() };
        let qc_table_v = unsafe { qc_send.as_slice() };
        let docs_sign_v = unsafe { ds_send.as_slice() };
        let docs_nz_v = unsafe { dn_send.as_slice() };
        let docs_scl_v = unsafe { dscl_send.as_slice() };
        let docs_cid_v = unsafe { dcid_send.as_slice() };
        let docs_cos_v = unsafe { dcos_send.as_slice() };
        let docs_sin_v = unsafe { dsin_send.as_slice() };
        let q_mask_v = qmask_send.as_ref().map(|s| unsafe { s.as_slice() });
        let d_mask_v = dmask_send.as_ref().map(|s| unsafe { s.as_slice() });

        let mut out = vec![0f32; big_a * big_b];
        fused_rroq158::score_batch(
            q_planes_v,
            q_meta_v,
            qc_table_v,
            q_mask_v,
            docs_sign_v,
            docs_nz_v,
            docs_scl_v,
            docs_cid_v,
            docs_cos_v,
            docs_sin_v,
            d_mask_v,
            big_a,
            big_b,
            big_s,
            big_t,
            n_words,
            n_groups,
            query_bits,
            big_k,
            n_threads,
            &mut out,
        );
        out
    });

    Ok(PyArray1::from_vec_bound(py, scores))
}

// ---------------------------------------------------------------
// rroq4_riem fused CPU scorer (free function)
// ---------------------------------------------------------------

/// Score a batch of (query × document) pairs with the rroq4_riem fused
/// MaxSim kernel on CPU. Mirrors
/// `voyager_index._internal.kernels.triton_roq_rroq4_riem.roq_maxsim_rroq4_riem`
/// exactly so per-token scores match Triton + the python reference within
/// float-rounding noise.
///
/// The doc-side codes are 4-bit asymmetric per-group; the query stays
/// fp32 (FWHT-rotated, plus per-group sums precomputed once per query).
///
/// Returns a flat `(A * B,)` `float32` numpy array of scores; the python
/// caller reshapes and runs top-k.
#[pyfunction]
#[pyo3(signature = (
    q_rot, q_gsums, qc_table,
    docs_codes, docs_mins, docs_dlts, docs_cid, docs_cos, docs_sin,
    big_a, big_b, big_s, big_t,
    dim, n_groups, group_size, big_k,
    q_mask = None, docs_mask = None, n_threads = None,
))]
#[allow(clippy::too_many_arguments)]
fn rroq4_riem_score_batch<'py>(
    py: Python<'py>,
    q_rot: PyReadonlyArray1<f32>,
    q_gsums: PyReadonlyArray1<f32>,
    qc_table: PyReadonlyArray1<f32>,
    docs_codes: PyReadonlyArray1<u8>,
    docs_mins: PyReadonlyArray1<f32>,
    docs_dlts: PyReadonlyArray1<f32>,
    docs_cid: PyReadonlyArray1<i32>,
    docs_cos: PyReadonlyArray1<f32>,
    docs_sin: PyReadonlyArray1<f32>,
    big_a: usize,
    big_b: usize,
    big_s: usize,
    big_t: usize,
    dim: usize,
    n_groups: usize,
    group_size: usize,
    big_k: usize,
    q_mask: Option<PyReadonlyArray1<f32>>,
    docs_mask: Option<PyReadonlyArray1<f32>>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if n_groups == 0 {
        return Err(PyValueError::new_err("n_groups must be > 0"));
    }
    if group_size == 0 || group_size % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "group_size must be a positive even integer (got {group_size}); \
             4-bit codes pack two coords per byte"
        )));
    }
    if dim == 0 {
        return Err(PyValueError::new_err("dim must be > 0"));
    }
    if big_a == 0 || big_b == 0 || big_s == 0 || big_t == 0 || big_k == 0 {
        return Err(PyValueError::new_err(format!(
            "All shape dims must be > 0 (got A={big_a}, B={big_b}, \
             S={big_s}, T={big_t}, K={big_k})"
        )));
    }
    if n_groups * group_size != dim {
        return Err(PyValueError::new_err(format!(
            "n_groups ({n_groups}) * group_size ({group_size}) != dim ({dim})"
        )));
    }
    let n_bytes = dim / 2;

    let exp_q_rot = big_a * big_s * dim;
    let exp_q_gsums = big_a * big_s * n_groups;
    let exp_qc = big_a * big_s * big_k;
    let exp_d_codes = big_b * big_t * n_bytes;
    let exp_d_grp = big_b * big_t * n_groups;
    let exp_d_cid = big_b * big_t;
    let exp_d_norm = big_b * big_t;

    let q_rot_slice = q_rot
        .as_slice()
        .map_err(|_| PyValueError::new_err("q_rot must be contiguous"))?;
    if q_rot_slice.len() != exp_q_rot {
        return Err(PyValueError::new_err(format!(
            "q_rot len {} != A*S*dim {}",
            q_rot_slice.len(),
            exp_q_rot
        )));
    }
    let q_gsums_slice = q_gsums
        .as_slice()
        .map_err(|_| PyValueError::new_err("q_gsums must be contiguous"))?;
    if q_gsums_slice.len() != exp_q_gsums {
        return Err(PyValueError::new_err(format!(
            "q_gsums len {} != A*S*n_groups {}",
            q_gsums_slice.len(),
            exp_q_gsums
        )));
    }
    let qc_table_slice = qc_table
        .as_slice()
        .map_err(|_| PyValueError::new_err("qc_table must be contiguous"))?;
    if qc_table_slice.len() != exp_qc {
        return Err(PyValueError::new_err(format!(
            "qc_table len {} != A*S*K {}",
            qc_table_slice.len(),
            exp_qc
        )));
    }
    let docs_codes_slice = docs_codes
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_codes must be contiguous"))?;
    if docs_codes_slice.len() != exp_d_codes {
        return Err(PyValueError::new_err(format!(
            "docs_codes len {} != B*T*(dim/2) {}",
            docs_codes_slice.len(),
            exp_d_codes
        )));
    }
    let docs_mins_slice = docs_mins
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_mins must be contiguous"))?;
    if docs_mins_slice.len() != exp_d_grp {
        return Err(PyValueError::new_err(format!(
            "docs_mins len {} != B*T*n_groups {}",
            docs_mins_slice.len(),
            exp_d_grp
        )));
    }
    let docs_dlts_slice = docs_dlts
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_dlts must be contiguous"))?;
    if docs_dlts_slice.len() != exp_d_grp {
        return Err(PyValueError::new_err(format!(
            "docs_dlts len {} != B*T*n_groups {}",
            docs_dlts_slice.len(),
            exp_d_grp
        )));
    }
    let docs_cid_slice = docs_cid
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_cid must be contiguous"))?;
    if docs_cid_slice.len() != exp_d_cid {
        return Err(PyValueError::new_err(format!(
            "docs_cid len {} != B*T {}",
            docs_cid_slice.len(),
            exp_d_cid
        )));
    }
    let docs_cos_slice = docs_cos
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_cos must be contiguous"))?;
    if docs_cos_slice.len() != exp_d_norm {
        return Err(PyValueError::new_err(format!(
            "docs_cos len {} != B*T {}",
            docs_cos_slice.len(),
            exp_d_norm
        )));
    }
    let docs_sin_slice = docs_sin
        .as_slice()
        .map_err(|_| PyValueError::new_err("docs_sin must be contiguous"))?;
    if docs_sin_slice.len() != exp_d_norm {
        return Err(PyValueError::new_err(format!(
            "docs_sin len {} != B*T {}",
            docs_sin_slice.len(),
            exp_d_norm
        )));
    }
    let q_mask_slice_opt = q_mask
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()
        .map_err(|_| PyValueError::new_err("q_mask must be contiguous"))?;
    if let Some(qm) = q_mask_slice_opt {
        if qm.len() != big_a * big_s {
            return Err(PyValueError::new_err(format!(
                "q_mask len {} != A*S {}",
                qm.len(),
                big_a * big_s
            )));
        }
    }
    let d_mask_slice_opt = docs_mask
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()
        .map_err(|_| PyValueError::new_err("docs_mask must be contiguous"))?;
    if let Some(dm) = d_mask_slice_opt {
        if dm.len() != big_b * big_t {
            return Err(PyValueError::new_err(format!(
                "docs_mask len {} != B*T {}",
                dm.len(),
                big_b * big_t
            )));
        }
    }

    // Same Send + Sync raw-pointer alias trick as `rroq158_score_batch` —
    // see that function's SAFETY comment for the full justification. Net
    // effect: we drop the GIL inside `py.allow_threads` without copying
    // any of the input buffers (codes alone can be ~16 MB/query at
    // production candidate counts).
    let qrot_send = SendSlice::from_slice(q_rot_slice);
    let qgs_send = SendSlice::from_slice(q_gsums_slice);
    let qc_send = SendSlice::from_slice(qc_table_slice);
    let dcodes_send = SendSlice::from_slice(docs_codes_slice);
    let dmins_send = SendSlice::from_slice(docs_mins_slice);
    let ddlts_send = SendSlice::from_slice(docs_dlts_slice);
    let dcid_send = SendSlice::from_slice(docs_cid_slice);
    let dcos_send = SendSlice::from_slice(docs_cos_slice);
    let dsin_send = SendSlice::from_slice(docs_sin_slice);
    let qmask_send = q_mask_slice_opt.map(SendSlice::from_slice);
    let dmask_send = d_mask_slice_opt.map(SendSlice::from_slice);

    let scores = py.allow_threads(move || {
        let q_rot_v = unsafe { qrot_send.as_slice() };
        let q_gsums_v = unsafe { qgs_send.as_slice() };
        let qc_v = unsafe { qc_send.as_slice() };
        let docs_codes_v = unsafe { dcodes_send.as_slice() };
        let docs_mins_v = unsafe { dmins_send.as_slice() };
        let docs_dlts_v = unsafe { ddlts_send.as_slice() };
        let docs_cid_v = unsafe { dcid_send.as_slice() };
        let docs_cos_v = unsafe { dcos_send.as_slice() };
        let docs_sin_v = unsafe { dsin_send.as_slice() };
        let q_mask_v = qmask_send.as_ref().map(|s| unsafe { s.as_slice() });
        let d_mask_v = dmask_send.as_ref().map(|s| unsafe { s.as_slice() });

        let mut out = vec![0f32; big_a * big_b];
        fused_rroq4_riem::score_batch(
            q_rot_v,
            q_gsums_v,
            qc_v,
            q_mask_v,
            docs_codes_v,
            docs_mins_v,
            docs_dlts_v,
            docs_cid_v,
            docs_cos_v,
            docs_sin_v,
            d_mask_v,
            big_a,
            big_b,
            big_s,
            big_t,
            dim,
            n_groups,
            group_size,
            big_k,
            n_threads,
            &mut out,
        );
        out
    });

    Ok(PyArray1::from_vec_bound(py, scores))
}

/// Test/benchmarking helper: force the rroq158 backend tier to "scalar",
/// "x86v3", or "auto". Returns the actually-selected tier as a string.
///
/// "auto" resets the cached backend so the next call to `select_backend`
/// detects CPU features fresh and picks the best available tier.
///
/// Intended for parity tests and microbenchmarks that need to A/B
/// compare AVX-512 VPOPCNTDQ vs scalar `popcntq`. Not for production.
#[pyfunction]
#[allow(non_snake_case)]
fn _rroq158_force_backend_for_tests(name: &str) -> PyResult<String> {
    use fused_rroq158::{
        _force_scalar_backend_for_tests, _reset_backend_for_tests,
    };
    #[cfg(target_arch = "x86_64")]
    use fused_rroq158::_force_x86v3_backend_for_tests;
    match name {
        "scalar" => {
            _force_scalar_backend_for_tests();
            Ok("scalar".to_string())
        }
        "x86v3" => {
            #[cfg(target_arch = "x86_64")]
            {
                _force_x86v3_backend_for_tests();
                Ok("x86v3".to_string())
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Err(PyValueError::new_err("x86v3 backend not available on non-x86_64"))
            }
        }
        "auto" => {
            _reset_backend_for_tests();
            Ok("auto".to_string())
        }
        other => Err(PyValueError::new_err(format!(
            "unknown backend '{other}': expected 'scalar', 'x86v3', or 'auto'"
        ))),
    }
}

#[pymodule]
fn latence_shard_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ShardIndex>()?;
    m.add_class::<metadata::MetadataStore>()?;
    m.add_function(wrap_pyfunction!(rroq158_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rroq4_riem_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(_rroq158_force_backend_for_tests, m)?)?;
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
