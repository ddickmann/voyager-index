use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use latence_gem_router::codebook::TwoStageCodebook;
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes};

use crate::graph::CsrAdjacency;

const MAGIC: &[u8; 4] = b"GEMS";
const VERSION: u32 = 2;
const COMPAT_VERSIONS: &[u32] = &[1, 2];

#[derive(Debug, Serialize, Deserialize)]
pub struct SegmentData {
    pub dim: usize,
    pub max_degree: usize,
    pub levels: Vec<CsrAdjacency>,
    pub shortcuts: Vec<Vec<u32>>,
    pub node_levels: Vec<usize>,
    pub entry_point: u32,
    pub codebook: TwoStageCodebook,
    pub doc_profiles: Vec<DocProfile>,
    pub doc_ids: Vec<u64>,
    pub flat_codes: FlatDocCodes,
    pub postings: ClusterPostings,
    pub ctop_r: usize,
    // v2: raw vectors + offsets for MaxSim reranking after load
    #[serde(default)]
    pub raw_vectors: Option<Vec<f32>>,
    #[serde(default)]
    pub doc_offsets: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub enum PersistError {
    Io(std::io::Error),
    Bincode(bincode::Error),
    BadMagic,
    BadVersion(u32),
    BadChecksum { expected: u32, actual: u32 },
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO: {e}"),
            Self::Bincode(e) => write!(f, "bincode: {e}"),
            Self::BadMagic => write!(f, "invalid file magic"),
            Self::BadVersion(v) => write!(f, "unsupported version {v}"),
            Self::BadChecksum { expected, actual } => {
                write!(f, "CRC32 mismatch: expected {expected:#x}, got {actual:#x}")
            }
        }
    }
}

impl From<std::io::Error> for PersistError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<bincode::Error> for PersistError {
    fn from(e: bincode::Error) -> Self {
        Self::Bincode(e)
    }
}

/// Save segment to disk with atomic rename and CRC32 integrity check.
///
/// Format: [MAGIC 4B][VERSION 4B][DATA_LEN 8B][bincode data][CRC32 4B]
pub fn save_segment(data: &SegmentData, path: &Path) -> Result<(), PersistError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let encoded = bincode::serialize(data)?;
    let crc = crc32fast::hash(&encoded);

    let tmp = path.with_extension("tmp");
    {
        let file = fs::File::create(&tmp)?;
        let mut w = BufWriter::new(file);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(encoded.len() as u64).to_le_bytes())?;
        w.write_all(&encoded)?;
        w.write_all(&crc.to_le_bytes())?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Load segment from disk with integrity verification.
/// Uses mmap for efficient I/O on large segment files,
/// falling back to buffered read for small files.
pub fn load_segment(path: &Path) -> Result<SegmentData, PersistError> {
    let file = fs::File::open(path)?;
    let file_len = file.metadata()?.len() as usize;

    const MMAP_THRESHOLD: usize = 1024 * 1024; // 1MB

    if file_len >= MMAP_THRESHOLD {
        return load_segment_mmap(&file, file_len);
    }

    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(PersistError::BadMagic);
    }

    let mut ver_buf = [0u8; 4];
    r.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if !COMPAT_VERSIONS.contains(&version) {
        return Err(PersistError::BadVersion(version));
    }

    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)?;
    let data_len = u64::from_le_bytes(len_buf) as usize;

    let max_data = file_len.saturating_sub(16 + 4);
    if data_len > max_data {
        return Err(PersistError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("declared data_len {} exceeds file capacity {}", data_len, max_data),
        )));
    }

    let mut data_buf = vec![0u8; data_len];
    r.read_exact(&mut data_buf)?;

    let mut crc_buf = [0u8; 4];
    r.read_exact(&mut crc_buf)?;
    let expected_crc = u32::from_le_bytes(crc_buf);
    let actual_crc = crc32fast::hash(&data_buf);
    if expected_crc != actual_crc {
        return Err(PersistError::BadChecksum {
            expected: expected_crc,
            actual: actual_crc,
        });
    }

    let segment: SegmentData = bincode::deserialize(&data_buf)?;
    Ok(segment)
}

/// Memory-mapped load for large segment files.
/// Avoids copying file contents into heap -- the OS page cache serves reads.
fn load_segment_mmap(file: &fs::File, file_len: usize) -> Result<SegmentData, PersistError> {
    let mmap = unsafe { Mmap::map(file)? };
    let bytes = &mmap[..];

    if bytes.len() < 16 {
        return Err(PersistError::BadMagic);
    }

    if &bytes[0..4] != MAGIC {
        return Err(PersistError::BadMagic);
    }

    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if !COMPAT_VERSIONS.contains(&version) {
        return Err(PersistError::BadVersion(version));
    }

    let data_len = u64::from_le_bytes([
        bytes[8], bytes[9], bytes[10], bytes[11],
        bytes[12], bytes[13], bytes[14], bytes[15],
    ]) as usize;

    let data_start: usize = 16;
    let data_end = data_start.checked_add(data_len).ok_or_else(|| {
        PersistError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "data_len overflow",
        ))
    })?;
    let crc_end = data_end.checked_add(4).ok_or_else(|| {
        PersistError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "crc offset overflow",
        ))
    })?;

    if crc_end > file_len {
        return Err(PersistError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "file truncated",
        )));
    }

    let data_buf = &bytes[data_start..data_end];
    let expected_crc = u32::from_le_bytes([
        bytes[data_end], bytes[data_end + 1], bytes[data_end + 2], bytes[data_end + 3],
    ]);
    let actual_crc = crc32fast::hash(data_buf);

    if expected_crc != actual_crc {
        return Err(PersistError::BadChecksum {
            expected: expected_crc,
            actual: actual_crc,
        });
    }

    let segment: SegmentData = bincode::deserialize(data_buf)?;
    Ok(segment)
}

/// Migrate a segment file to the current format version.
/// Reads from `src`, upgrades in-memory, writes to `dst` with current VERSION.
pub fn migrate_segment(src: &Path, dst: &Path) -> Result<(), PersistError> {
    let data = load_segment(src)?;
    save_segment(&data, dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Adjacency;
    use tempfile::tempdir;

    fn make_test_segment_data() -> SegmentData {
        SegmentData {
            dim: 32,
            max_degree: 16,
            levels: vec![CsrAdjacency::from_adj_lists(&[vec![1, 2], vec![0, 2], vec![0, 1]])],
            shortcuts: vec![Vec::new(); 3],
            node_levels: vec![0, 0, 0],
            entry_point: 0,
            codebook: TwoStageCodebook {
                cquant: vec![0.0; 16],
                n_fine: 2,
                dim: 8,
                cindex_labels: vec![0, 1],
                n_coarse: 2,
                centroid_dists: vec![0.0, 1.0, 1.0, 0.0],
                idf: vec![1.0, 1.0],
            },
            doc_profiles: vec![
                DocProfile { centroid_ids: vec![0], ctop: vec![0] },
                DocProfile { centroid_ids: vec![1], ctop: vec![1] },
                DocProfile { centroid_ids: vec![0, 1], ctop: vec![0, 1] },
            ],
            doc_ids: vec![100, 200, 300],
            flat_codes: FlatDocCodes {
                codes: vec![0, 1, 0, 1],
                offsets: vec![0, 1, 2],
                lengths: vec![1, 1, 2],
            },
            postings: ClusterPostings {
                lists: vec![vec![0, 2], vec![1, 2]],
                cluster_reps: vec![Some(0), Some(1)],
            },
            ctop_r: 2,
            raw_vectors: Some(vec![1.0; 32 * 3]),
            doc_offsets: vec![(0, 1), (1, 2), (2, 3)],
        }
    }

    #[test]
    fn test_save_load_roundtrip() {
        let data = make_test_segment_data();

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gem");

        save_segment(&data, &path).unwrap();
        let loaded = load_segment(&path).unwrap();

        assert_eq!(loaded.dim, 32);
        assert_eq!(loaded.doc_ids, vec![100, 200, 300]);
        assert_eq!(loaded.levels[0].n_nodes(), 3);
        assert!(loaded.raw_vectors.is_some());
        assert_eq!(loaded.raw_vectors.as_ref().unwrap().len(), 32 * 3);
        assert_eq!(loaded.doc_offsets.len(), 3);
    }

    #[test]
    fn test_migration_roundtrip() {
        let data = make_test_segment_data();

        let dir = tempdir().unwrap();
        let src_path = dir.path().join("original.gem");
        let dst_path = dir.path().join("migrated.gem");

        save_segment(&data, &src_path).unwrap();
        migrate_segment(&src_path, &dst_path).unwrap();

        let migrated = load_segment(&dst_path).unwrap();

        assert_eq!(migrated.dim, data.dim);
        assert_eq!(migrated.max_degree, data.max_degree);
        assert_eq!(migrated.doc_ids, data.doc_ids);
        assert_eq!(migrated.entry_point, data.entry_point);
        assert_eq!(migrated.ctop_r, data.ctop_r);
        assert_eq!(migrated.levels[0].n_nodes(), data.levels[0].n_nodes());
    }
}
