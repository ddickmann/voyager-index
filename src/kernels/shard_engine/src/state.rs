/// Lock-free immutable state snapshots via ArcSwap.
///
/// `ShardState` is the immutable snapshot.  Readers load a guard from the
/// `ArcSwap<ShardState>` and can hold it for the duration of a search without
/// blocking writers.  Writers build a *new* `ShardState` and atomically swap
/// it in; in-flight readers continue with the old snapshot until they drop
/// their guard.
use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::mmap_reader::MmapShard;
use crate::types::{DocId, ShardId};

/// Per-document metadata stored in the snapshot.
#[derive(Debug, Clone)]
pub struct DocMeta {
    pub shard_id: ShardId,
    pub row_start: usize,
    pub row_end: usize,
    pub dim: usize,
    pub payload_json: Option<Vec<u8>>,
}

/// Immutable snapshot of the entire shard index state.
#[derive(Debug)]
pub struct ShardState {
    pub docs: HashMap<DocId, DocMeta>,
    pub doc_ids: Vec<DocId>,
    /// Mean-pooled centroid per document for proxy scoring.
    pub doc_means: Option<Vec<f32>>,
    pub doc_mean_dim: usize,
    /// Ordered mapping: doc_means[idx * dim .. (idx+1) * dim] corresponds to
    /// doc_mean_ids[idx].
    pub doc_mean_ids: Vec<DocId>,
    pub shard_count: u32,
    pub total_vectors: u64,
}

impl ShardState {
    pub fn empty() -> Self {
        Self {
            docs: HashMap::new(),
            doc_ids: Vec::new(),
            doc_means: None,
            doc_mean_dim: 0,
            doc_mean_ids: Vec::new(),
            shard_count: 0,
            total_vectors: 0,
        }
    }

    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }
}

/// Thread-safe handle to the current immutable state.
///
/// - `load()` → `Arc<ShardState>` — O(1), lock-free
/// - `store(new)` — atomic pointer swap, O(1)
pub struct StateHandle {
    inner: ArcSwap<ShardState>,
}

impl StateHandle {
    pub fn new(initial: ShardState) -> Self {
        Self {
            inner: ArcSwap::from_pointee(initial),
        }
    }

    /// Load the current snapshot.  The returned `Arc` keeps the snapshot
    /// alive even if a new one is swapped in concurrently.
    pub fn load(&self) -> Arc<ShardState> {
        self.inner.load_full()
    }

    /// Atomically replace the current snapshot.
    pub fn store(&self, new_state: ShardState) {
        self.inner.store(Arc::new(new_state));
    }

    /// Swap in a new state and return the previous one.
    pub fn swap(&self, new_state: ShardState) -> Arc<ShardState> {
        self.inner.swap(Arc::new(new_state))
    }
}

/// Open all shard files in a directory and return them keyed by shard id.
pub fn open_shards(
    shard_dir: &std::path::Path,
    shard_count: u32,
) -> std::io::Result<HashMap<ShardId, MmapShard>> {
    let mut shards = HashMap::new();
    for sid in 0..shard_count {
        let path = shard_dir.join(format!("shard_{sid}.safetensors"));
        if path.exists() {
            shards.insert(sid, MmapShard::open(&path)?);
        }
    }
    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_handle_swap() {
        let h = StateHandle::new(ShardState::empty());
        assert_eq!(h.load().doc_count(), 0);

        let mut new_state = ShardState::empty();
        new_state.docs.insert(1, DocMeta {
            shard_id: 0,
            row_start: 0,
            row_end: 5,
            dim: 128,
            payload_json: None,
        });
        new_state.doc_ids.push(1);
        h.store(new_state);

        assert_eq!(h.load().doc_count(), 1);
    }

    #[test]
    fn test_concurrent_read_during_swap() {
        let h = Arc::new(StateHandle::new(ShardState::empty()));
        let snap_before = h.load();
        assert_eq!(snap_before.doc_count(), 0);

        let mut new_state = ShardState::empty();
        new_state.docs.insert(42, DocMeta {
            shard_id: 0,
            row_start: 0,
            row_end: 1,
            dim: 64,
            payload_json: None,
        });
        new_state.doc_ids.push(42);
        h.store(new_state);

        // Old snapshot is still valid
        assert_eq!(snap_before.doc_count(), 0);
        // New snapshot sees the update
        assert_eq!(h.load().doc_count(), 1);
    }
}
