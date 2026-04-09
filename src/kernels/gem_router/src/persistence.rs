use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::router::GemRouterState;

const MAGIC: &[u8; 4] = b"GEMR";
const VERSION: u32 = 2;
const COMPAT_VERSIONS: &[u32] = &[1, 2];

#[derive(Debug)]
pub enum PersistenceError {
    Io(std::io::Error),
    Serde(serde_json::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Serde(e) => write!(f, "serialization error: {e}"),
            Self::InvalidMagic => write!(f, "invalid file magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
        }
    }
}

impl From<std::io::Error> for PersistenceError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for PersistenceError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e)
    }
}

/// Save router state to disk.
///
/// Format:
///   [4 bytes magic "GEMR"]
///   [4 bytes version u32 LE]
///   [8 bytes json_len u64 LE]
///   [json_len bytes of JSON-serialized GemRouterState]
pub fn save_state(state: &GemRouterState, path: &Path) -> Result<(), PersistenceError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_vec(state)?;
    let checksum = crc32fast::hash(&json);

    let tmp_path = path.with_extension("tmp");
    {
        let file = fs::File::create(&tmp_path)?;
        let mut w = BufWriter::new(file);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(json.len() as u64).to_le_bytes())?;
        w.write_all(&json)?;
        w.write_all(&checksum.to_le_bytes())?;
        w.flush()?;
    }
    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Load router state from disk.
pub fn load_state(path: &Path) -> Result<GemRouterState, PersistenceError> {
    let file = fs::File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(PersistenceError::InvalidMagic);
    }

    let mut ver_buf = [0u8; 4];
    r.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if !COMPAT_VERSIONS.contains(&version) {
        return Err(PersistenceError::UnsupportedVersion(version));
    }

    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)?;
    let json_len = u64::from_le_bytes(len_buf) as usize;

    let mut json_buf = vec![0u8; json_len];
    r.read_exact(&mut json_buf)?;

    if version >= 2 {
        let mut crc_buf = [0u8; 4];
        r.read_exact(&mut crc_buf)?;
        let stored_crc = u32::from_le_bytes(crc_buf);
        let actual_crc = crc32fast::hash(&json_buf);
        if stored_crc != actual_crc {
            return Err(PersistenceError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("CRC32 mismatch: stored {stored_crc:#x}, computed {actual_crc:#x}"),
            )));
        }
    }

    let state: GemRouterState = serde_json::from_slice(&json_buf)?;
    Ok(state)
}

/// Migrate a router state file to the current format version.
/// Reads from `src`, upgrades in-memory, writes to `dst` with current VERSION.
pub fn migrate_router_state(src: &Path, dst: &Path) -> Result<(), PersistenceError> {
    let state = load_state(src)?;
    save_state(&state, dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::GemRouter;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    #[test]
    fn test_save_load_roundtrip() {
        let dim = 8;
        let n_docs = 10;
        let vecs_per_doc = 4;
        let n_vectors = n_docs * vecs_per_doc;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n_vectors * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let doc_ids: Vec<u64> = (0..n_docs as u64).collect();
        let ranges: Vec<(usize, usize)> = (0..n_docs)
            .map(|i| (i * vecs_per_doc, (i + 1) * vecs_per_doc))
            .collect();

        let mut router = GemRouter::new();
        router.build(&data, n_vectors, dim, &doc_ids, &ranges, 8, 4, 10, 3);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("router.gemr");

        save_state(router.state().unwrap(), &path).unwrap();
        let loaded = load_state(&path).unwrap();

        assert_eq!(loaded.doc_ids.len(), n_docs);
        assert_eq!(loaded.doc_profiles.len(), n_docs);
        assert_eq!(loaded.codebook.n_fine, router.state().unwrap().codebook.n_fine);
    }
}
