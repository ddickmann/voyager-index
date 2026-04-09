use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use latence_gem_router::codebook::TwoStageCodebook;
use latence_gem_router::router::{ClusterPostings, DocProfile, FlatDocCodes};

const MAGIC: &[u8; 4] = b"GEMS";
const VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct SegmentData {
    pub dim: usize,
    pub max_degree: usize,
    pub adjacency: Vec<Vec<u32>>,
    pub shortcuts: Vec<Vec<u32>>,
    pub codebook: TwoStageCodebook,
    pub doc_profiles: Vec<DocProfile>,
    pub doc_ids: Vec<u64>,
    pub flat_codes: FlatDocCodes,
    pub postings: ClusterPostings,
    pub ctop_r: usize,
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
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Load segment from disk with integrity verification.
pub fn load_segment(path: &Path) -> Result<SegmentData, PersistError> {
    let file = fs::File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(PersistError::BadMagic);
    }

    let mut ver_buf = [0u8; 4];
    r.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if version != VERSION {
        return Err(PersistError::BadVersion(version));
    }

    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)?;
    let data_len = u64::from_le_bytes(len_buf) as usize;

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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_load_roundtrip() {
        let data = SegmentData {
            dim: 32,
            max_degree: 16,
            adjacency: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            shortcuts: vec![Vec::new(); 3],
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
        };

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gem");

        save_segment(&data, &path).unwrap();
        let loaded = load_segment(&path).unwrap();

        assert_eq!(loaded.dim, 32);
        assert_eq!(loaded.doc_ids, vec![100, 200, 300]);
        assert_eq!(loaded.adjacency.len(), 3);
    }
}
