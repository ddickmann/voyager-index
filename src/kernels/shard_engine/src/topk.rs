/// Heap-based partial sort for top-k extraction.
///
/// Uses a min-heap of size k: for each candidate, if its score exceeds the
/// current minimum we swap it in.  Final cost is O(n log k) instead of
/// O(n log n) for a full sort.
use std::collections::BinaryHeap;

use crate::types::ScoredDoc;

/// Extract the top-k highest-scoring documents from an iterator.
/// Returns results sorted by score descending.
pub fn heap_topk(iter: impl Iterator<Item = ScoredDoc>, k: usize) -> Vec<ScoredDoc> {
    if k == 0 {
        return Vec::new();
    }
    // ScoredDoc's Ord is reversed (min-heap), so BinaryHeap pops lowest score.
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for doc in iter {
        heap.push(doc);
        if heap.len() > k {
            heap.pop(); // drop the lowest score
        }
    }
    // into_sorted_vec gives ascending by Ord; our Ord is reversed,
    // so ascending-in-Ord = descending-by-actual-score.
    heap.into_sorted_vec()
}

/// Merge multiple sorted (descending) result lists and return top-k.
pub fn merge_topk(lists: &[Vec<ScoredDoc>], k: usize) -> Vec<ScoredDoc> {
    let total: usize = lists.iter().map(|l| l.len()).sum();
    let iter = lists.iter().flat_map(|l| l.iter().copied());
    if total <= k {
        let mut merged: Vec<ScoredDoc> = iter.collect();
        merged.sort_by(|a, b| b.score.total_cmp(&a.score));
        return merged;
    }
    heap_topk(iter, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_topk_basic() {
        let docs = vec![
            ScoredDoc { doc_id: 1, score: 0.5 },
            ScoredDoc { doc_id: 2, score: 0.9 },
            ScoredDoc { doc_id: 3, score: 0.1 },
            ScoredDoc { doc_id: 4, score: 0.7 },
            ScoredDoc { doc_id: 5, score: 0.3 },
        ];
        let top3 = heap_topk(docs.into_iter(), 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].doc_id, 2); // 0.9
        assert_eq!(top3[1].doc_id, 4); // 0.7
        assert_eq!(top3[2].doc_id, 1); // 0.5
    }

    #[test]
    fn test_heap_topk_fewer_than_k() {
        let docs = vec![ScoredDoc { doc_id: 1, score: 1.0 }];
        let top5 = heap_topk(docs.into_iter(), 5);
        assert_eq!(top5.len(), 1);
    }

    #[test]
    fn test_merge_topk() {
        let a = vec![
            ScoredDoc { doc_id: 1, score: 0.9 },
            ScoredDoc { doc_id: 2, score: 0.5 },
        ];
        let b = vec![
            ScoredDoc { doc_id: 3, score: 0.8 },
            ScoredDoc { doc_id: 4, score: 0.4 },
        ];
        let merged = merge_topk(&[a, b], 3);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].doc_id, 1);
        assert_eq!(merged[1].doc_id, 3);
        assert_eq!(merged[2].doc_id, 2);
    }
}
