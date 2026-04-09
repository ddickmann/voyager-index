use std::collections::HashMap;

/// Score cache for memoizing pairwise doc-doc distance computations
/// during graph construction. Uses partial random eviction to avoid
/// the performance cliff of clearing the entire cache at once.
pub struct ScoreCache {
    cache: HashMap<(u32, u32), f32>,
    max_size: usize,
    hits: u64,
    misses: u64,
}

impl ScoreCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size.min(65536)),
            max_size: max_size.max(16),
            hits: 0,
            misses: 0,
        }
    }

    #[inline]
    fn key(a: u32, b: u32) -> (u32, u32) {
        if a <= b { (a, b) } else { (b, a) }
    }

    #[inline]
    pub fn get(&mut self, a: u32, b: u32) -> Option<f32> {
        let result = self.cache.get(&Self::key(a, b)).copied();
        if result.is_some() {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        result
    }

    #[inline]
    pub fn insert(&mut self, a: u32, b: u32, score: f32) {
        if self.cache.len() >= self.max_size {
            let evict_count = self.max_size / 4;
            let keys_to_remove: Vec<(u32, u32)> = self.cache.keys()
                .take(evict_count)
                .copied()
                .collect();
            for k in keys_to_remove {
                self.cache.remove(&k);
            }
        }
        self.cache.insert(Self::key(a, b), score);
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_cache_symmetric() {
        let mut cache = ScoreCache::new(100);
        cache.insert(5, 10, 0.42);
        assert_eq!(cache.get(5, 10), Some(0.42));
        assert_eq!(cache.get(10, 5), Some(0.42));
        assert_eq!(cache.get(5, 11), None);
    }

    #[test]
    fn test_score_cache_eviction() {
        let mut cache = ScoreCache::new(16);
        for i in 0..16 {
            cache.insert(0, i, i as f32);
        }
        assert_eq!(cache.len(), 16);
        cache.insert(0, 100, 100.0);
        assert!(cache.len() <= 13); // evicted ~25%
        assert_eq!(cache.get(0, 100), Some(100.0));
    }
}
