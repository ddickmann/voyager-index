/// SQLite-backed metadata store with FTS5 full-text search.
///
/// Replaces JSON-file payloads with a durable, indexed store that supports:
/// - O(1) payload CRUD by doc_id
/// - Full-text search via FTS5
/// - Indexed filter queries for structured fields
use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use rusqlite::{params, Connection, OpenFlags};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS docs (
    doc_id    INTEGER PRIMARY KEY,
    payload   TEXT NOT NULL DEFAULT '{}',
    text_content TEXT NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_docs_updated ON docs(updated_at);
";

const FTS_SCHEMA: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    text_content,
    content=docs,
    content_rowid=doc_id,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
    INSERT INTO docs_fts(rowid, text_content) VALUES (new.doc_id, new.text_content);
END;
CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, text_content) VALUES('delete', old.doc_id, old.text_content);
END;
CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, text_content) VALUES('delete', old.doc_id, old.text_content);
    INSERT INTO docs_fts(rowid, text_content) VALUES (new.doc_id, new.text_content);
END;
";

#[pyclass]
pub struct MetadataStore {
    conn: Connection,
}

impl MetadataStore {
    fn conn(&self) -> &Connection {
        &self.conn
    }
}

#[pymethods]
impl MetadataStore {
    /// Open or create a metadata store at the given path.
    /// Pass ":memory:" for an in-memory database.
    #[new]
    #[pyo3(signature = (path=":memory:"))]
    fn new(path: &str) -> PyResult<Self> {
        let conn = if path == ":memory:" {
            Connection::open_in_memory()
        } else {
            Connection::open_with_flags(
                path,
                OpenFlags::SQLITE_OPEN_READ_WRITE
                    | OpenFlags::SQLITE_OPEN_CREATE
                    | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )
        }
        .map_err(|e| PyIOError::new_err(format!("SQLite open failed: {e}")))?;

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-64000;")
            .map_err(|e| PyIOError::new_err(format!("PRAGMA failed: {e}")))?;

        conn.execute_batch(SCHEMA)
            .map_err(|e| PyIOError::new_err(format!("schema init failed: {e}")))?;

        conn.execute_batch(FTS_SCHEMA)
            .map_err(|e| PyIOError::new_err(format!("FTS5 init failed: {e}")))?;

        Ok(Self { conn })
    }

    /// Set (upsert) the JSON payload for a document.
    /// `text_content` is indexed for FTS if provided.
    #[pyo3(signature = (doc_id, payload_json, text_content=None))]
    fn set_payload(&self, doc_id: u64, payload_json: &str, text_content: Option<&str>) -> PyResult<()> {
        if doc_id > i64::MAX as u64 {
            return Err(PyValueError::new_err("doc_id exceeds SQLite INTEGER range"));
        }
        let text = text_content.unwrap_or("");
        self.conn()
            .execute(
                "INSERT INTO docs(doc_id, payload, text_content, updated_at) VALUES(?1, ?2, ?3, strftime('%s','now'))
                 ON CONFLICT(doc_id) DO UPDATE SET payload=excluded.payload, text_content=excluded.text_content, updated_at=excluded.updated_at",
                params![doc_id as i64, payload_json, text],
            )
            .map_err(|e| PyIOError::new_err(format!("set_payload failed: {e}")))?;
        Ok(())
    }

    /// Get the JSON payload for a document, or None.
    fn get_payload(&self, doc_id: u64) -> PyResult<Option<String>> {
        let mut stmt = self.conn()
            .prepare_cached("SELECT payload FROM docs WHERE doc_id = ?1")
            .map_err(|e| PyIOError::new_err(format!("prepare failed: {e}")))?;

        let result = stmt
            .query_row(params![doc_id as i64], |row| row.get::<_, String>(0))
            .ok();
        Ok(result)
    }

    /// Delete a document's metadata. Returns True if it existed.
    fn delete_payload(&self, doc_id: u64) -> PyResult<bool> {
        let n = self.conn()
            .execute("DELETE FROM docs WHERE doc_id = ?1", params![doc_id as i64])
            .map_err(|e| PyIOError::new_err(format!("delete failed: {e}")))?;
        Ok(n > 0)
    }

    /// Bulk set payloads in a single transaction.
    /// `items` is a list of (doc_id, payload_json, text_content) tuples.
    fn set_payloads_bulk(&self, items: Vec<(u64, String, String)>) -> PyResult<usize> {
        let tx = self.conn.unchecked_transaction()
            .map_err(|e| PyIOError::new_err(format!("transaction failed: {e}")))?;

        let mut count = 0usize;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT INTO docs(doc_id, payload, text_content, updated_at) VALUES(?1, ?2, ?3, strftime('%s','now'))
                     ON CONFLICT(doc_id) DO UPDATE SET payload=excluded.payload, text_content=excluded.text_content, updated_at=excluded.updated_at",
                )
                .map_err(|e| PyIOError::new_err(format!("prepare failed: {e}")))?;

            for (doc_id, payload, text) in &items {
                stmt.execute(params![*doc_id as i64, payload, text])
                    .map_err(|e| PyIOError::new_err(format!("bulk insert failed: {e}")))?;
                count += 1;
            }
        }

        tx.commit()
            .map_err(|e| PyIOError::new_err(format!("commit failed: {e}")))?;
        Ok(count)
    }

    /// Full-text search. Returns doc_ids matching the FTS5 query.
    #[pyo3(signature = (query, limit=100))]
    fn search_text(&self, query: &str, limit: usize) -> PyResult<Vec<u64>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn()
            .prepare_cached(
                "SELECT rowid FROM docs_fts WHERE docs_fts MATCH ?1 ORDER BY rank LIMIT ?2",
            )
            .map_err(|e| PyIOError::new_err(format!("FTS prepare failed: {e}")))?;

        let rows = stmt
            .query_map(params![query, limit as i64], |row| row.get::<_, i64>(0))
            .map_err(|e| PyIOError::new_err(format!("FTS query failed: {e}")))?;

        let mut ids = Vec::new();
        for row in rows {
            let id = row.map_err(|e| PyIOError::new_err(format!("FTS row failed: {e}")))?;
            ids.push(id as u64);
        }
        Ok(ids)
    }

    /// Create an expression index on a JSON field for fast filtering.
    /// `field_path` must be alphanumeric/underscore/dot only (e.g. "color", "nested.field").
    fn create_field_index(&self, field_path: &str) -> PyResult<()> {
        if field_path.is_empty() {
            return Err(PyValueError::new_err("field_path cannot be empty"));
        }
        // Reject any non-alphanumeric/underscore/dot characters to prevent injection
        if !field_path.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.') {
            return Err(PyValueError::new_err(
                "field_path must contain only alphanumeric, underscore, or dot characters",
            ));
        }
        let json_path = if field_path.starts_with("$.") {
            field_path.to_string()
        } else {
            format!("$.{field_path}")
        };
        let safe_name: String = field_path
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let sql = format!(
            "CREATE INDEX IF NOT EXISTS idx_json_{safe_name} ON docs(json_extract(payload, '{json_path}'))"
        );
        self.conn()
            .execute_batch(&sql)
            .map_err(|e| PyIOError::new_err(format!("create index failed: {e}")))?;
        Ok(())
    }

    /// Filter doc_ids by a JSON field equality check.
    /// Uses json_extract — call `create_field_index` first for fast lookups.
    #[pyo3(signature = (field_path, value, limit=10000))]
    fn filter_by_field(&self, field_path: &str, value: &str, limit: usize) -> PyResult<Vec<u64>> {
        if field_path.is_empty() {
            return Err(PyValueError::new_err("field_path cannot be empty"));
        }
        let json_path = if field_path.starts_with("$.") {
            field_path.to_string()
        } else {
            format!("$.{field_path}")
        };

        let mut stmt = self.conn()
            .prepare_cached(
                "SELECT doc_id FROM docs WHERE json_extract(payload, ?1) = ?2 LIMIT ?3",
            )
            .map_err(|e| PyIOError::new_err(format!("filter prepare failed: {e}")))?;

        let rows = stmt
            .query_map(params![json_path, value, limit as i64], |row| {
                row.get::<_, i64>(0)
            })
            .map_err(|e| PyIOError::new_err(format!("filter query failed: {e}")))?;

        let mut ids = Vec::new();
        for row in rows {
            let id = row.map_err(|e| PyIOError::new_err(format!("filter row failed: {e}")))?;
            ids.push(id as u64);
        }
        Ok(ids)
    }

    /// Number of documents in the store.
    fn count(&self) -> PyResult<u64> {
        let n: i64 = self.conn()
            .query_row("SELECT COUNT(*) FROM docs", [], |row| row.get(0))
            .map_err(|e| PyIOError::new_err(format!("count failed: {e}")))?;
        Ok(n as u64)
    }

    /// Get all doc_ids in the store.
    fn all_doc_ids(&self) -> PyResult<Vec<u64>> {
        let mut stmt = self.conn()
            .prepare_cached("SELECT doc_id FROM docs ORDER BY doc_id")
            .map_err(|e| PyIOError::new_err(format!("prepare failed: {e}")))?;

        let rows = stmt
            .query_map([], |row| row.get::<_, i64>(0))
            .map_err(|e| PyIOError::new_err(format!("query failed: {e}")))?;

        let mut ids = Vec::new();
        for row in rows {
            let id = row.map_err(|e| PyIOError::new_err(format!("row failed: {e}")))?;
            ids.push(id as u64);
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_store() -> MetadataStore {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA journal_mode=WAL;").unwrap();
        conn.execute_batch(SCHEMA).unwrap();
        conn.execute_batch(FTS_SCHEMA).unwrap();
        MetadataStore { conn }
    }

    #[test]
    fn test_crud() {
        let s = mem_store();
        s.conn().execute(
            "INSERT INTO docs(doc_id, payload, text_content) VALUES(?1, ?2, ?3)",
            params![1i64, r#"{"color":"red"}"#, "hello world"],
        ).unwrap();

        let p = s.conn().query_row("SELECT payload FROM docs WHERE doc_id=1", [], |r| r.get::<_, String>(0)).unwrap();
        assert_eq!(p, r#"{"color":"red"}"#);

        s.conn().execute("DELETE FROM docs WHERE doc_id=1", []).unwrap();
        let n: i64 = s.conn().query_row("SELECT COUNT(*) FROM docs", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_fts5_search() {
        let s = mem_store();
        s.conn().execute(
            "INSERT INTO docs(doc_id, payload, text_content) VALUES(?1, ?2, ?3)",
            params![10i64, "{}", "the quick brown fox jumps over the lazy dog"],
        ).unwrap();
        s.conn().execute(
            "INSERT INTO docs(doc_id, payload, text_content) VALUES(?1, ?2, ?3)",
            params![20i64, "{}", "a cat sat on the mat"],
        ).unwrap();

        let mut stmt = s.conn().prepare("SELECT rowid FROM docs_fts WHERE docs_fts MATCH ?1").unwrap();
        let ids: Vec<i64> = stmt.query_map(params!["fox"], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(ids, vec![10]);

        let ids2: Vec<i64> = stmt.query_map(params!["cat"], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(ids2, vec![20]);
    }

    #[test]
    fn test_json_filter() {
        let s = mem_store();
        s.conn().execute(
            "INSERT INTO docs(doc_id, payload, text_content) VALUES(?1, ?2, ?3)",
            params![1i64, r#"{"color":"red","size":10}"#, ""],
        ).unwrap();
        s.conn().execute(
            "INSERT INTO docs(doc_id, payload, text_content) VALUES(?1, ?2, ?3)",
            params![2i64, r#"{"color":"blue","size":20}"#, ""],
        ).unwrap();
        s.conn().execute(
            "INSERT INTO docs(doc_id, payload, text_content) VALUES(?1, ?2, ?3)",
            params![3i64, r#"{"color":"red","size":30}"#, ""],
        ).unwrap();

        let mut stmt = s.conn().prepare(
            "SELECT doc_id FROM docs WHERE json_extract(payload, '$.color') = ?1"
        ).unwrap();
        let ids: Vec<i64> = stmt.query_map(params!["red"], |r| r.get(0)).unwrap().filter_map(|r| r.ok()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_bulk_insert() {
        let s = mem_store();
        let tx = s.conn.unchecked_transaction().unwrap();
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO docs(doc_id, payload, text_content) VALUES(?1, ?2, ?3)"
            ).unwrap();
            for i in 0..1000i64 {
                stmt.execute(params![i, "{}", format!("doc {i}")]).unwrap();
            }
        }
        tx.commit().unwrap();

        let n: i64 = s.conn.query_row("SELECT COUNT(*) FROM docs", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 1000);
    }
}
