"""CRUD, filter, and retrieval helpers for the shard manager."""
from __future__ import annotations

from .common import *  # noqa: F401,F403

class ShardSegmentManagerMutationMixin:
    @staticmethod
    def _evaluate_filter(payload: dict, filters: Dict) -> bool:
        """Evaluate Qdrant-style payload filters with $and/$or support."""
        if not filters:
            return True
        if "$and" in filters:
            return all(ShardSegmentManagerMutationMixin._evaluate_filter(payload, sub) for sub in filters["$and"])
        if "$or" in filters:
            return any(ShardSegmentManagerMutationMixin._evaluate_filter(payload, sub) for sub in filters["$or"])
        for key, condition in filters.items():
            val = payload.get(key)
            if isinstance(condition, dict):
                if "$eq" in condition and val != condition["$eq"]:
                    return False
                if "$in" in condition and val not in condition["$in"]:
                    return False
                if "$contains" in condition:
                    if not isinstance(val, (list, str)) or condition["$contains"] not in val:
                        return False
                if "$gt" in condition:
                    try:
                        if val is None or val <= condition["$gt"]:
                            return False
                    except TypeError:
                        return False
                if "$lt" in condition:
                    try:
                        if val is None or val >= condition["$lt"]:
                            return False
                    except TypeError:
                        return False
            elif val != condition:
                return False
        return True

    def _match_filter(self, doc_id: int, filters: Dict) -> bool:
        """Check if a document matches the given filters (sealed + memtable)."""
        payload = self._get_payload(doc_id)
        return self._evaluate_filter(payload, filters)

    def _get_payload(self, doc_id: int, tombstones: Optional[Set[int]] = None) -> Optional[dict]:
        """Retrieve merged payload for a doc from sealed store and memtable."""
        if tombstones and doc_id in tombstones:
            return None
        payload = {}
        if hasattr(self, "_sealed_payloads") and self._sealed_payloads:
            payload = dict(self._sealed_payloads.get(doc_id, {}))
        if self._memtable:
            _, mt_payloads, _ = self._memtable.snapshot()
            if doc_id in mt_payloads:
                payload.update(mt_payloads[doc_id])
        return payload

    def _get_doc_vectors(self, doc_id: int) -> Optional[np.ndarray]:
        """Retrieve vectors for a doc from memtable or sealed store."""
        if self._memtable:
            docs, _, _ = self._memtable.snapshot()
            if doc_id in docs:
                return docs[doc_id]
        if self._doc_vecs and self._doc_ids:
            try:
                idx = self._doc_ids.index(doc_id)
                v = self._doc_vecs[idx]
                return np.asarray(v, dtype=np.float32)
            except (ValueError, IndexError):
                pass
        if self._store:
            try:
                fetched = self._store.fetch_docs([doc_id])
                if fetched:
                    return list(fetched.values())[0]
            except Exception:
                pass
        return None

    def add_multidense(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        if not self._is_built:
            self.build(vectors, ids, payloads)
            return
        with self._lock:
            plds = payloads or [None] * len(vectors)
            for v, did, p in zip(vectors, ids, plds):
                arr = np.asarray(v, dtype=np.float32)
                if self._wal_writer:
                    self._wal_writer.log_insert(did, arr, p)
                self._memtable.insert(did, arr, p)
                self._next_doc_id = max(self._next_doc_id, did + 1)

    def delete(self, ids: List[int]) -> None:
        with self._lock:
            for did in ids:
                if self._wal_writer:
                    self._wal_writer.log_delete(did)
                if self._memtable:
                    self._memtable.delete(did)
            if self._router:
                self._router.delete_docs(ids)

    def upsert_multidense(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        if not self._is_built:
            self.build(vectors, ids, payloads)
            return
        with self._lock:
            plds = payloads or [None] * len(vectors)
            for v, did, p in zip(vectors, ids, plds):
                arr = np.asarray(v, dtype=np.float32)
                if self._wal_writer:
                    self._wal_writer.log_upsert(did, arr, p)
                if self._memtable:
                    self._memtable.upsert(did, arr, p)
                self._next_doc_id = max(self._next_doc_id, did + 1)

    def _compute_live_count(self) -> int:
        """Compute distinct live doc count (sealed + memtable - tombstones), no double-counting."""
        sealed_ids = set(self._doc_ids) if self._doc_ids else set()
        if self._memtable:
            docs, _, tombstones = self._memtable.snapshot()
            unique_ids = (sealed_ids | set(docs.keys())) - tombstones
        else:
            unique_ids = sealed_ids
        return len(unique_ids)

    def total_vectors(self) -> int:
        return self._compute_live_count()

    def allocate_ids(self, n: int) -> List[int]:
        """Allocate n new sequential document IDs using monotonic counter."""
        with self._lock:
            start = self._next_doc_id
            self._next_doc_id += n
            return list(range(start, start + n))

    def upsert_payload(self, doc_id: int, payload: Dict[str, Any]) -> None:
        """Update or insert payload for a document."""
        with self._lock:
            if hasattr(self, "_sealed_payloads") and doc_id in (self._sealed_payloads or {}):
                self._sealed_payloads[doc_id].update(payload)
            if self._memtable:
                _, mt_payloads, _ = self._memtable.snapshot()
                merged = mt_payloads.get(doc_id, {})
                merged.update(payload)
                self._memtable.upsert_payload(doc_id, merged)
            if self._wal_writer:
                self._wal_writer.log_update_payload(doc_id, payload)

    def retrieve(self, ids: List[int], with_vector: bool = False, with_payload: bool = True) -> list:
        """Retrieve documents by ID from sealed store and memtable.

        Tombstoned documents are skipped (not included in results).
        """
        tombstones = self._memtable.tombstones_snapshot() if self._memtable else set()
        results = []
        for did in ids:
            if did in tombstones:
                continue
            entry: Dict[str, Any] = {"id": did}
            if with_payload:
                entry["payload"] = self._get_payload(did, tombstones) or {}
            else:
                entry["payload"] = {}
            if with_vector:
                entry["vector"] = self._get_doc_vectors(did)
            results.append(entry)
        return results

