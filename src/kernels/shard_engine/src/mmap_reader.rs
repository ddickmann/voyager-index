/// Memory-mapped shard reader for zero-copy safetensors access.
///
/// Opens a safetensors file via mmap and exposes doc-selective slicing:
/// given a set of (offset_start, offset_end) ranges within the embeddings
/// tensor, returns only those rows without loading the entire file.
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::types::DocId;

/// Parsed safetensors tensor metadata.
#[derive(Debug, Clone)]
struct TensorMeta {
    dtype: String,
    #[allow(dead_code)]
    dtype_size: usize,
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

/// An mmap-backed shard file.
pub struct MmapShard {
    path: PathBuf,
    _file: File,
    mmap: Mmap,
    tensors: HashMap<String, TensorMeta>,
    header_size: usize,
}

impl MmapShard {
    /// Open a safetensors shard file with mmap.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small for safetensors header"));
        }

        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if 8 + header_len > mmap.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "safetensors header overflows file"));
        }

        let header_bytes = &mmap[8..8 + header_len];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let parsed: serde_json::Value = serde_json::from_str(header_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let data_base = 8 + header_len;
        let mut tensors = HashMap::new();

        if let Some(obj) = parsed.as_object() {
            for (name, meta_val) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let meta_obj = match meta_val.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                let dtype = meta_obj
                    .get("dtype")
                    .and_then(|d: &serde_json::Value| d.as_str())
                    .unwrap_or("F32");
                let dtype_size = match dtype {
                    "F16" | "BF16" => 2,
                    "F32" => 4,
                    "F64" => 8,
                    "I8" | "U8" => 1,
                    "I16" => 2,
                    "I32" => 4,
                    "I64" => 8,
                    _ => 4,
                };
                let shape: Vec<usize> = meta_obj
                    .get("shape")
                    .and_then(|s: &serde_json::Value| s.as_array())
                    .map(|arr: &Vec<serde_json::Value>| {
                        arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
                    })
                    .unwrap_or_default();
                let offsets: Option<&Vec<serde_json::Value>> = meta_obj
                    .get("data_offsets")
                    .and_then(|o: &serde_json::Value| o.as_array());
                let (start, end) = match offsets {
                    Some(arr) if arr.len() == 2 => {
                        let s = arr[0].as_u64().unwrap_or(0) as usize;
                        let e = arr[1].as_u64().unwrap_or(0) as usize;
                        (data_base + s, data_base + e)
                    }
                    _ => continue,
                };
                tensors.insert(name.clone(), TensorMeta { dtype: dtype.to_string(), dtype_size, shape, data_start: start, data_end: end });
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            _file: file,
            mmap,
            tensors,
            header_size: data_base,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the shape of a named tensor.
    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// Read a contiguous slice of rows [row_start..row_end) from a named tensor.
    /// Returns raw bytes (caller is responsible for reinterpreting dtype).
    pub fn read_rows_raw(&self, tensor_name: &str, row_start: usize, row_end: usize) -> io::Result<&[u8]> {
        let meta = self.tensors.get(tensor_name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor '{}' not found in {}", tensor_name, self.path.display()))
        })?;

        if meta.shape.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "tensor has empty shape"));
        }

        let n_rows = meta.shape[0];
        if n_rows == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "tensor has 0 rows"));
        }
        let total_bytes = meta.data_end - meta.data_start;
        if total_bytes % n_rows != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("data size {} not evenly divisible by {} rows", total_bytes, n_rows),
            ));
        }
        let row_bytes = total_bytes / n_rows;

        if row_end > n_rows || row_start > row_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("row range [{row_start}, {row_end}) out of bounds for {n_rows} rows"),
            ));
        }

        let byte_start = meta.data_start + row_start * row_bytes;
        let byte_end = meta.data_start + row_end * row_bytes;

        if byte_end > self.mmap.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "slice extends beyond mmap"));
        }

        Ok(&self.mmap[byte_start..byte_end])
    }

    /// Read selected row ranges from a tensor and return concatenated raw bytes.
    /// `ranges` is a list of (row_start, row_end) pairs.
    pub fn read_selected_rows(&self, tensor_name: &str, ranges: &[(usize, usize)]) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        for &(s, e) in ranges {
            let bytes = self.read_rows_raw(tensor_name, s, e)?;
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }

    /// Read selected rows as f32 slices.
    /// Returns an error if the tensor dtype is not F32.
    pub fn read_selected_f32(&self, tensor_name: &str, ranges: &[(usize, usize)]) -> io::Result<Vec<f32>> {
        if let Some(meta) = self.tensors.get(tensor_name) {
            if meta.dtype != "F32" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("read_selected_f32 requires F32 dtype, got '{}'", meta.dtype),
                ));
            }
        }
        let raw = self.read_selected_rows(tensor_name, ranges)?;
        if raw.len() % 4 != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "raw bytes not aligned to f32"));
        }
        let floats: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(floats)
    }

    /// Read all rows of a tensor as raw bytes.
    pub fn read_all_rows_raw(&self, tensor_name: &str) -> io::Result<&[u8]> {
        let meta = self.tensors.get(tensor_name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor '{}' not found", tensor_name))
        })?;
        let start = meta.data_start;
        let end = meta.data_end;
        if end > self.mmap.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "tensor extends beyond mmap"));
        }
        Ok(&self.mmap[start..end])
    }

    #[allow(dead_code)]
    pub fn header_size(&self) -> usize {
        self.header_size
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

/// A document's location within a shard.
#[derive(Debug, Clone)]
pub struct DocLocation {
    pub doc_id: DocId,
    pub shard_id: u32,
    pub row_start: usize,
    pub row_end: usize,
}

/// Batch-load embeddings for a set of documents from open MmapShard instances.
pub fn load_docs_selective(
    shards: &HashMap<u32, MmapShard>,
    locations: &[DocLocation],
) -> io::Result<Vec<f32>> {
    let mut all_data = Vec::new();
    for loc in locations {
        let shard = shards.get(&loc.shard_id).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("shard {} not open", loc.shard_id))
        })?;
        let floats = shard.read_selected_f32("embeddings", &[(loc.row_start, loc.row_end)])?;
        all_data.extend_from_slice(&floats);
    }
    Ok(all_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_safetensors_f32(path: &Path, tensor_name: &str, data: &[f32], shape: &[usize]) {
        let n_bytes = data.len() * 4;
        let header = serde_json::json!({
            tensor_name: {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [0, n_bytes]
            }
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let header_bytes = header_str.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = File::create(path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(header_bytes).unwrap();
        for &v in data {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.sync_all().unwrap();
    }

    #[test]
    fn test_mmap_shard_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.safetensors");

        // 4 rows, dim 3
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        write_safetensors_f32(&path, "embeddings", &data, &[4, 3]);

        let shard = MmapShard::open(&path).unwrap();
        assert_eq!(shard.tensor_shape("embeddings"), Some(&[4, 3][..]));

        let row1 = shard.read_selected_f32("embeddings", &[(1, 2)]).unwrap();
        assert_eq!(row1, vec![3.0, 4.0, 5.0]);

        let rows_02 = shard.read_selected_f32("embeddings", &[(0, 1), (2, 3)]).unwrap();
        assert_eq!(rows_02, vec![0.0, 1.0, 2.0, 6.0, 7.0, 8.0]);
    }
}
