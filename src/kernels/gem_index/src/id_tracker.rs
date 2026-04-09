use std::collections::HashMap;

/// Maps external document IDs (u64) to internal graph indices (u32)
/// and tracks soft-deleted documents.
pub struct IdTracker {
    ext_to_int: HashMap<u64, u32>,
    int_to_ext: Vec<u64>,
    deleted: Vec<bool>,
    n_deleted: usize,
}

impl IdTracker {
    pub fn new() -> Self {
        Self {
            ext_to_int: HashMap::new(),
            int_to_ext: Vec::new(),
            deleted: Vec::new(),
            n_deleted: 0,
        }
    }

    pub fn from_ids(doc_ids: &[u64]) -> Self {
        let mut tracker = Self::new();
        for &id in doc_ids {
            tracker.add(id);
        }
        tracker
    }

    pub fn add(&mut self, ext_id: u64) -> u32 {
        let int_id = self.int_to_ext.len() as u32;
        self.ext_to_int.insert(ext_id, int_id);
        self.int_to_ext.push(ext_id);
        self.deleted.push(false);
        int_id
    }

    pub fn delete(&mut self, ext_id: u64) -> bool {
        if let Some(&int_id) = self.ext_to_int.get(&ext_id) {
            let idx = int_id as usize;
            if idx < self.deleted.len() && !self.deleted[idx] {
                self.deleted[idx] = true;
                self.n_deleted += 1;
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn is_deleted(&self, int_id: u32) -> bool {
        let idx = int_id as usize;
        idx < self.deleted.len() && self.deleted[idx]
    }

    pub fn n_live(&self) -> usize {
        self.int_to_ext.len() - self.n_deleted
    }

    pub fn n_total(&self) -> usize {
        self.int_to_ext.len()
    }

    pub fn n_deleted(&self) -> usize {
        self.n_deleted
    }

    pub fn ext_to_int(&self, ext_id: u64) -> Option<u32> {
        self.ext_to_int.get(&ext_id).copied()
    }

    pub fn int_to_ext(&self, int_id: u32) -> u64 {
        self.int_to_ext[int_id as usize]
    }

    pub fn deleted_flags(&self) -> &[bool] {
        &self.deleted
    }

    /// Build a compacted tracker, returning (new_tracker, old_to_new mapping).
    /// old_to_new[old_int] = Some(new_int) if the node was live.
    pub fn compact(&self) -> (IdTracker, Vec<Option<u32>>) {
        let mut new_tracker = IdTracker::new();
        let mut mapping: Vec<Option<u32>> = vec![None; self.n_total()];

        for (old_int, &ext_id) in self.int_to_ext.iter().enumerate() {
            if !self.deleted[old_int] {
                let new_int = new_tracker.add(ext_id);
                mapping[old_int] = Some(new_int);
            }
        }

        (new_tracker, mapping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_tracker() {
        let mut t = IdTracker::new();
        assert_eq!(t.add(100), 0);
        assert_eq!(t.add(200), 1);
        assert_eq!(t.add(300), 2);
        assert_eq!(t.n_total(), 3);
        assert_eq!(t.n_live(), 3);

        assert_eq!(t.ext_to_int(200), Some(1));
        assert_eq!(t.int_to_ext(1), 200);

        assert!(t.delete(200));
        assert!(!t.delete(200)); // already deleted
        assert!(!t.delete(999)); // doesn't exist
        assert_eq!(t.n_live(), 2);
        assert!(t.is_deleted(1));

        let (new_t, mapping) = t.compact();
        assert_eq!(new_t.n_total(), 2);
        assert_eq!(mapping[0], Some(0));
        assert_eq!(mapping[1], None);
        assert_eq!(mapping[2], Some(1));
    }
}
