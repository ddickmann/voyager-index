/// Compact bit-set for tracking visited nodes during beam search.
pub struct VisitedSet {
    bits: Vec<u64>,
}

impl VisitedSet {
    pub fn new(capacity: usize) -> Self {
        let n_words = (capacity + 63) / 64;
        Self {
            bits: vec![0u64; n_words],
        }
    }

    #[inline]
    pub fn set(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        if word < self.bits.len() {
            self.bits[word] |= 1u64 << bit;
        }
    }

    #[inline]
    pub fn contains(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        if word < self.bits.len() {
            (self.bits[word] >> bit) & 1 == 1
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        for w in self.bits.iter_mut() {
            *w = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visited_set() {
        let mut vs = VisitedSet::new(100);
        assert!(!vs.contains(0));
        assert!(!vs.contains(99));
        vs.set(42);
        assert!(vs.contains(42));
        assert!(!vs.contains(43));
        vs.set(0);
        vs.set(99);
        assert!(vs.contains(0));
        assert!(vs.contains(99));
        vs.clear();
        assert!(!vs.contains(42));
    }
}
