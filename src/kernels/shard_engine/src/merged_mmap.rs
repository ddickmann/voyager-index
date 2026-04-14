/// Merged flat mmap files for zero-copy document access.
///
/// Replaces per-shard scattered reads with contiguous mmap regions:
///   - merged_codes.bin:      [8B total_tokens i64] [total_tokens × 2B u16]
///   - merged_embeddings.bin: [16B header: total_tokens i64, dim i64] [total_tokens × dim × 2B f16]
///   - merged_offsets.bin:    [8B n_entries i64] [n_entries × 8B i64]
///   - merged_doc_map.bin:    [8B n_docs i64] [n_docs × 8B u64]
///
/// All document accesses are O(1) HashMap lookup + pointer arithmetic — no
/// per-doc allocation, no shard-level indirection.
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;

pub struct MergedMmap {
    codes_mmap: Option<Mmap>,
    #[allow(dead_code)]
    codes_file: Option<File>,

    embeddings_mmap: Mmap,
    #[allow(dead_code)]
    embeddings_file: File,

    offsets: Vec<i64>,
    doc_to_pos: HashMap<u64, usize>,
    dim: usize,
    #[allow(dead_code)]
    total_tokens: usize,
}

impl MergedMmap {
    /// Load all four merged files from `dir`.
    pub fn load(dir: &Path) -> io::Result<Self> {
        let emb_path = dir.join("merged_embeddings.bin");
        let offsets_path = dir.join("merged_offsets.bin");
        let doc_map_path = dir.join("merged_doc_map.bin");
        let codes_path = dir.join("merged_codes.bin");

        // --- embeddings (required) ---
        let emb_file = File::open(&emb_path)?;
        let emb_mmap = unsafe { Mmap::map(&emb_file)? };
        if emb_mmap.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "merged_embeddings.bin too small for header",
            ));
        }
        let total_tokens =
            i64::from_le_bytes(emb_mmap[0..8].try_into().unwrap()) as usize;
        let dim =
            i64::from_le_bytes(emb_mmap[8..16].try_into().unwrap()) as usize;

        let expected_emb_size = 16 + total_tokens * dim * 2;
        if emb_mmap.len() < expected_emb_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "merged_embeddings.bin too small: {} < {} expected",
                    emb_mmap.len(),
                    expected_emb_size
                ),
            ));
        }

        // --- codes (optional) ---
        let (codes_mmap, codes_file) = if codes_path.exists() {
            let file = File::open(&codes_path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            if mmap.len() < 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "merged_codes.bin too small for header",
                ));
            }
            let codes_total =
                i64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let expected_codes_size = 8 + codes_total * 2;
            if mmap.len() < expected_codes_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "merged_codes.bin too small: {} < {} expected",
                        mmap.len(),
                        expected_codes_size
                    ),
                ));
            }
            (Some(mmap), Some(file))
        } else {
            (None, None)
        };

        // --- offsets ---
        let off_file = File::open(&offsets_path)?;
        let off_mmap = unsafe { Mmap::map(&off_file)? };
        if off_mmap.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "merged_offsets.bin too small",
            ));
        }
        let n_entries =
            i64::from_le_bytes(off_mmap[0..8].try_into().unwrap()) as usize;
        let expected_off_size = 8 + n_entries * 8;
        if off_mmap.len() < expected_off_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "merged_offsets.bin data truncated",
            ));
        }
        let mut offsets = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let s = 8 + i * 8;
            offsets.push(i64::from_le_bytes(
                off_mmap[s..s + 8].try_into().unwrap(),
            ));
        }

        // --- doc_map ---
        let map_file = File::open(&doc_map_path)?;
        let map_mmap = unsafe { Mmap::map(&map_file)? };
        if map_mmap.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "merged_doc_map.bin too small",
            ));
        }
        let n_docs =
            i64::from_le_bytes(map_mmap[0..8].try_into().unwrap()) as usize;
        let expected_map_size = 8 + n_docs * 8;
        if map_mmap.len() < expected_map_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "merged_doc_map.bin data truncated",
            ));
        }
        let mut doc_to_pos = HashMap::with_capacity(n_docs);
        for i in 0..n_docs {
            let s = 8 + i * 8;
            let doc_id = u64::from_le_bytes(
                map_mmap[s..s + 8].try_into().unwrap(),
            );
            doc_to_pos.insert(doc_id, i);
        }

        log::info!(
            "MergedMmap loaded: {} docs, {} tokens, dim={}, has_codes={}",
            n_docs,
            total_tokens,
            dim,
            codes_mmap.is_some(),
        );

        Ok(Self {
            codes_mmap,
            codes_file,
            embeddings_mmap: emb_mmap,
            embeddings_file: emb_file,
            offsets,
            doc_to_pos,
            dim,
            total_tokens,
        })
    }

    /// Zero-copy slice of centroid codes (u16) for one document.
    #[inline]
    pub fn get_codes(&self, doc_id: u64) -> Option<&[u16]> {
        let pos = *self.doc_to_pos.get(&doc_id)?;
        let start = self.offsets[pos] as usize;
        let end = self.offsets[pos + 1] as usize;
        if start >= end {
            return Some(&[]);
        }
        let mmap = self.codes_mmap.as_ref()?;
        let byte_start = 8 + start * 2;
        let byte_end = 8 + end * 2;
        if byte_end > mmap.len() {
            return None;
        }
        let data = &mmap[byte_start..byte_end];
        // Safety: header is 8 bytes (aligned), start*2 is even, so byte_start
        // is always 2-byte aligned within the page-aligned mmap.
        let codes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u16, end - start)
        };
        Some(codes)
    }

    /// Zero-copy slice of raw FP16 embedding bytes for one document.
    #[inline]
    pub fn get_embeddings_f16_bytes(&self, doc_id: u64) -> Option<&[u8]> {
        let pos = *self.doc_to_pos.get(&doc_id)?;
        let start = self.offsets[pos] as usize;
        let end = self.offsets[pos + 1] as usize;
        if start >= end {
            return Some(&[]);
        }
        let byte_start = 16 + start * self.dim * 2;
        let byte_end = 16 + end * self.dim * 2;
        if byte_end > self.embeddings_mmap.len() {
            return None;
        }
        Some(&self.embeddings_mmap[byte_start..byte_end])
    }

    /// Token range (start, end) for a document, if present.
    #[inline]
    pub fn get_token_range(&self, doc_id: u64) -> Option<(usize, usize)> {
        let pos = *self.doc_to_pos.get(&doc_id)?;
        let start = self.offsets[pos] as usize;
        let end = self.offsets[pos + 1] as usize;
        Some((start, end))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn has_codes(&self) -> bool {
        self.codes_mmap.is_some()
    }

    pub fn n_docs(&self) -> usize {
        self.doc_to_pos.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_files(dir: &Path, n_docs: usize, dim: usize) {
        let tokens_per_doc = 4;
        let total_tokens = n_docs * tokens_per_doc;

        // merged_embeddings.bin
        let mut emb = File::create(dir.join("merged_embeddings.bin")).unwrap();
        emb.write_all(&(total_tokens as i64).to_le_bytes()).unwrap();
        emb.write_all(&(dim as i64).to_le_bytes()).unwrap();
        for _ in 0..total_tokens * dim {
            emb.write_all(&0x3C00u16.to_le_bytes()).unwrap(); // f16 = 1.0
        }

        // merged_codes.bin
        let mut codes = File::create(dir.join("merged_codes.bin")).unwrap();
        codes
            .write_all(&(total_tokens as i64).to_le_bytes())
            .unwrap();
        for t in 0..total_tokens {
            codes
                .write_all(&((t % 8) as u16).to_le_bytes())
                .unwrap();
        }

        // merged_offsets.bin
        let n_entries = n_docs + 1;
        let mut off = File::create(dir.join("merged_offsets.bin")).unwrap();
        off.write_all(&(n_entries as i64).to_le_bytes()).unwrap();
        for i in 0..=n_docs {
            off.write_all(&((i * tokens_per_doc) as i64).to_le_bytes())
                .unwrap();
        }

        // merged_doc_map.bin
        let mut dmap = File::create(dir.join("merged_doc_map.bin")).unwrap();
        dmap.write_all(&(n_docs as i64).to_le_bytes()).unwrap();
        for i in 0..n_docs {
            dmap.write_all(&((100 + i) as u64).to_le_bytes())
                .unwrap();
        }
    }

    #[test]
    fn test_load_and_access() {
        let dir = tempfile::tempdir().unwrap();
        write_test_files(dir.path(), 3, 4);

        let mm = MergedMmap::load(dir.path()).unwrap();
        assert_eq!(mm.n_docs(), 3);
        assert_eq!(mm.dim(), 4);
        assert!(mm.has_codes());

        let codes = mm.get_codes(100).unwrap();
        assert_eq!(codes.len(), 4);

        let emb = mm.get_embeddings_f16_bytes(101).unwrap();
        assert_eq!(emb.len(), 4 * 4 * 2); // 4 tokens * 4 dim * 2 bytes

        let range = mm.get_token_range(102).unwrap();
        assert_eq!(range, (8, 12));

        assert!(mm.get_codes(999).is_none());
    }
}
