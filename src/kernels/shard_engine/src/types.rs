/// Common type aliases used across the shard engine.

pub type DocId = u64;
pub type Score = f32;
pub type ShardId = u32;

/// A scored document candidate.
#[derive(Debug, Clone, Copy)]
pub struct ScoredDoc {
    pub doc_id: DocId,
    pub score: Score,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap: BinaryHeap pops the element with the
        // lowest actual score so we keep the highest.
        // total_cmp handles NaN deterministically (NaN sorts low).
        match other.score.total_cmp(&self.score) {
            std::cmp::Ordering::Equal => self.doc_id.cmp(&other.doc_id),
            ord => ord,
        }
    }
}
