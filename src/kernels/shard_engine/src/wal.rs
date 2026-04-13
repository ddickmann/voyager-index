/// Write-Ahead Log with CRC32 integrity checks.
///
/// Binary format per entry:
///   [op: u8][doc_id: u64][payload_len: u32][payload: bytes][vec_rows: u32][vec_dim: u32][vec_data: f32×rows×dim][crc32: u32]
///
/// `crc32` covers all preceding bytes of the entry.
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::types::DocId;

/// Limits to prevent OOM from corrupt WAL files.
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024; // 64 MiB
const MAX_VEC_FLOATS: usize = 64 * 1024 * 1024; // 256 MiB of f32

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalOp {
    Insert = 1,
    Delete = 2,
    Upsert = 3,
    UpdatePayload = 4,
}

impl WalOp {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Insert),
            2 => Some(Self::Delete),
            3 => Some(Self::Upsert),
            4 => Some(Self::UpdatePayload),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub op: WalOp,
    pub doc_id: DocId,
    pub payload_json: Option<Vec<u8>>,
    pub vectors: Option<Vec<f32>>,
    pub vec_rows: u32,
    pub vec_dim: u32,
}

// ---------------------------------------------------------------
// Writer
// ---------------------------------------------------------------

pub struct WalWriter {
    #[allow(dead_code)]
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    n_entries: u64,
}

impl WalWriter {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: Some(BufWriter::new(file)),
            n_entries: 0, // caller sets from replay if needed
        })
    }

    pub fn set_n_entries(&mut self, n: u64) {
        self.n_entries = n;
    }

    pub fn n_entries(&self) -> u64 {
        self.n_entries
    }

    pub fn log_insert(&mut self, doc_id: DocId, vectors: &[f32], rows: u32, dim: u32, payload: Option<&[u8]>) -> io::Result<()> {
        self.write_entry(WalOp::Insert, doc_id, vectors, rows, dim, payload)
    }

    pub fn log_delete(&mut self, doc_id: DocId) -> io::Result<()> {
        self.write_entry(WalOp::Delete, doc_id, &[], 0, 0, None)
    }

    pub fn log_upsert(&mut self, doc_id: DocId, vectors: &[f32], rows: u32, dim: u32, payload: Option<&[u8]>) -> io::Result<()> {
        self.write_entry(WalOp::Upsert, doc_id, vectors, rows, dim, payload)
    }

    pub fn log_update_payload(&mut self, doc_id: DocId, payload: &[u8]) -> io::Result<()> {
        self.write_entry(WalOp::UpdatePayload, doc_id, &[], 0, 0, Some(payload))
    }

    pub fn sync(&mut self) -> io::Result<()> {
        if let Some(ref mut w) = self.writer {
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        Ok(())
    }

    pub fn close(&mut self) -> io::Result<()> {
        self.sync()?;
        self.writer = None;
        Ok(())
    }

    fn write_entry(
        &mut self,
        op: WalOp,
        doc_id: DocId,
        vectors: &[f32],
        rows: u32,
        dim: u32,
        payload: Option<&[u8]>,
    ) -> io::Result<()> {
        let w = self.writer.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "WAL writer is closed")
        })?;

        let mut buf: Vec<u8> = Vec::with_capacity(64 + vectors.len() * 4);
        buf.write_u8(op as u8)?;
        buf.write_u64::<LittleEndian>(doc_id)?;

        let pld = payload.unwrap_or(&[]);
        buf.write_u32::<LittleEndian>(pld.len() as u32)?;
        buf.write_all(pld)?;

        buf.write_u32::<LittleEndian>(rows)?;
        buf.write_u32::<LittleEndian>(dim)?;
        for &v in vectors {
            buf.write_f32::<LittleEndian>(v)?;
        }

        let crc = crc32fast::hash(&buf);
        buf.write_u32::<LittleEndian>(crc)?;

        w.write_all(&buf)?;
        self.n_entries += 1;
        Ok(())
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            log::error!("WAL close on drop failed: {e}");
        }
    }
}

// ---------------------------------------------------------------
// Reader
// ---------------------------------------------------------------

pub fn replay(path: &Path) -> io::Result<Vec<WalEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut entries = Vec::new();

    loop {
        match read_one_entry(&mut reader) {
            Ok(entry) => entries.push(entry),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                log::warn!("WAL replay stopped at entry {}: {}", entries.len(), e);
                break;
            }
        }
    }
    Ok(entries)
}

fn read_one_entry(r: &mut impl Read) -> io::Result<WalEntry> {
    let op_byte = r.read_u8()?;
    let op = WalOp::from_u8(op_byte).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("bad WAL op: {op_byte}"))
    })?;
    let doc_id = r.read_u64::<LittleEndian>()?;

    let pld_len = r.read_u32::<LittleEndian>()? as usize;
    if pld_len > MAX_PAYLOAD_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAL payload_len {pld_len} exceeds limit {MAX_PAYLOAD_BYTES}"),
        ));
    }
    let mut pld_buf = vec![0u8; pld_len];
    r.read_exact(&mut pld_buf)?;

    let rows = r.read_u32::<LittleEndian>()?;
    let dim = r.read_u32::<LittleEndian>()?;
    let n_floats = (rows as usize).checked_mul(dim as usize).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "rows*dim overflow")
    })?;
    if n_floats > MAX_VEC_FLOATS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAL vec size {n_floats} exceeds limit {MAX_VEC_FLOATS}"),
        ));
    }
    let mut vecs = vec![0.0f32; n_floats];
    for slot in &mut vecs {
        *slot = r.read_f32::<LittleEndian>()?;
    }

    let stored_crc = r.read_u32::<LittleEndian>()?;

    // Reconstruct CRC from the data we just read
    let mut check_buf: Vec<u8> = Vec::with_capacity(64 + n_floats * 4);
    check_buf.write_u8(op as u8).unwrap();
    check_buf.write_u64::<LittleEndian>(doc_id).unwrap();
    check_buf.write_u32::<LittleEndian>(pld_len as u32).unwrap();
    check_buf.write_all(&pld_buf).unwrap();
    check_buf.write_u32::<LittleEndian>(rows).unwrap();
    check_buf.write_u32::<LittleEndian>(dim).unwrap();
    for &v in &vecs {
        check_buf.write_f32::<LittleEndian>(v).unwrap();
    }
    let computed_crc = crc32fast::hash(&check_buf);
    if computed_crc != stored_crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("CRC mismatch: stored={stored_crc:#x} computed={computed_crc:#x}"),
        ));
    }

    Ok(WalEntry {
        op,
        doc_id,
        payload_json: if pld_buf.is_empty() { None } else { Some(pld_buf) },
        vectors: if vecs.is_empty() { None } else { Some(vecs) },
        vec_rows: rows,
        vec_dim: dim,
    })
}

// ---------------------------------------------------------------
// Tombstone set from WAL replay
// ---------------------------------------------------------------

pub fn tombstones_from_entries(entries: &[WalEntry]) -> HashSet<DocId> {
    let mut tombstones = HashSet::new();
    for e in entries {
        match e.op {
            WalOp::Delete => { tombstones.insert(e.doc_id); }
            WalOp::Insert | WalOp::Upsert => { tombstones.remove(&e.doc_id); }
            WalOp::UpdatePayload => {}
        }
    }
    tombstones
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut w = WalWriter::open(&wal_path).unwrap();
            w.log_insert(42, &[1.0, 2.0, 3.0, 4.0], 2, 2, Some(b"{\"k\":1}")).unwrap();
            w.log_delete(99).unwrap();
            w.log_upsert(42, &[5.0, 6.0], 1, 2, None).unwrap();
            w.log_update_payload(42, b"{\"k\":2}").unwrap();
            w.sync().unwrap();
            assert_eq!(w.n_entries(), 4);
        }

        let entries = replay(&wal_path).unwrap();
        assert_eq!(entries.len(), 4);

        assert_eq!(entries[0].op, WalOp::Insert);
        assert_eq!(entries[0].doc_id, 42);
        assert_eq!(entries[0].vec_rows, 2);
        assert_eq!(entries[0].vec_dim, 2);
        assert_eq!(entries[0].vectors.as_ref().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(entries[0].payload_json.as_ref().unwrap(), b"{\"k\":1}");

        assert_eq!(entries[1].op, WalOp::Delete);
        assert_eq!(entries[1].doc_id, 99);

        assert_eq!(entries[2].op, WalOp::Upsert);
        assert_eq!(entries[3].op, WalOp::UpdatePayload);

        let ts = tombstones_from_entries(&entries);
        assert!(ts.contains(&99));
        assert!(!ts.contains(&42));
    }

    #[test]
    fn test_wal_crc_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("corrupt.wal");

        {
            let mut w = WalWriter::open(&wal_path).unwrap();
            w.log_insert(1, &[0.0], 1, 1, None).unwrap();
            w.sync().unwrap();
        }

        // Corrupt a byte in the middle of the file
        let mut data = std::fs::read(&wal_path).unwrap();
        if data.len() > 5 {
            data[5] ^= 0xFF;
        }
        std::fs::write(&wal_path, &data).unwrap();

        let entries = replay(&wal_path).unwrap();
        assert!(entries.is_empty(), "corrupted entry should be rejected");
    }
}
