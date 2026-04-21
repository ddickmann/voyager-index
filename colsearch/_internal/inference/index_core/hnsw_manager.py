"""
HNSW Segment Manager
====================

Manages active + sealed HNSW segments using Qdrant's segment library.
Integrates with existing ShardedStorage architecture.
"""

import gc
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from latence_hnsw import HnswSegment
except ImportError:
    class HnswSegment:
        """
        Local fallback segment used when the Rust extension is unavailable.

        The fallback keeps the same high-level interface as the PyO3 binding so
        the public Python package remains usable on CPU-only and unbuilt setups.
        """

        def __init__(
            self,
            path: str,
            dim: int,
            distance_metric: str = "cosine",
            m: int = 16,
            ef_construct: int = 100,
            on_disk: bool = False,
            is_appendable: bool = True,
            multivector_comparator: Optional[str] = None,
        ):
            self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
            self.dim = dim
            self.distance_metric = distance_metric
            self.multivector_comparator = multivector_comparator
            self.is_appendable = is_appendable
            self.manifest_file = self.path / "segment_state.json"
            self.vectors_file = self.path / "segment_vectors.npz"
            self.legacy_state_file = self.path / "segment_state.pkl"
            self.items: Dict[int, Dict[str, Any]] = {}
            self.next_id = 0
            self._load()

        def _load(self) -> None:
            if self.legacy_state_file.exists():
                raise RuntimeError(
                    "Legacy pickle-based HNSW fallback state is no longer supported. "
                    "Rebuild the collection from a trusted source to migrate it."
                )
            if not self.manifest_file.exists() or not self.vectors_file.exists():
                return
            with open(self.manifest_file, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            data = np.load(self.vectors_file, allow_pickle=False)
            vectors = data["vectors"]

            self.items = {}
            offset = 0
            for item in manifest.get("items", []):
                item_id = int(item["id"])
                rows = int(item["rows"])
                chunk = vectors[offset:offset + rows]
                offset += rows
                vector = chunk if item.get("is_multivector", False) else chunk.reshape(self.dim)
                self.items[item_id] = {
                    "vector": vector.astype(np.float32, copy=False),
                    "payload": dict(item.get("payload") or {}),
                }
            self.next_id = int(manifest.get("next_id", 0))

        def _atomic_replace(self, destination: Path, writer) -> None:
            with tempfile.NamedTemporaryFile(
                dir=self.path,
                prefix=f".{destination.name}.",
                suffix=destination.suffix,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
            try:
                writer(temp_path)
                if temp_path.exists():
                    with open(temp_path, "rb") as handle:
                        os.fsync(handle.fileno())
                temp_path.replace(destination)
                parent_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

        def _save(self) -> None:
            ids = sorted(self.items.keys())
            manifest_items = []
            flat_vectors = []
            for item_id in ids:
                item = self.items[item_id]
                vector = np.asarray(item["vector"], dtype=np.float32)
                is_multivector = vector.ndim == 2
                matrix = vector if is_multivector else vector.reshape(1, -1)
                flat_vectors.append(matrix)
                manifest_items.append(
                    {
                        "id": int(item_id),
                        "rows": int(matrix.shape[0]),
                        "is_multivector": is_multivector,
                        "payload": dict(item.get("payload") or {}),
                    }
                )

            vectors = (
                np.concatenate(flat_vectors, axis=0).astype(np.float32, copy=False)
                if flat_vectors
                else np.empty((0, self.dim), dtype=np.float32)
            )
            manifest = {
                "version": 1,
                "dim": self.dim,
                "next_id": self.next_id,
                "items": manifest_items,
            }

            self._atomic_replace(
                self.manifest_file,
                lambda path: path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"),
            )
            self._atomic_replace(
                self.vectors_file,
                lambda path: np.savez_compressed(path, vectors=vectors),
            )

        def _normalize(self, vector: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vector) + 1e-8
            return vector / norm

        def _score(self, query: np.ndarray, vector: np.ndarray) -> float:
            if self.distance_metric == "cosine":
                return float(np.dot(self._normalize(query), self._normalize(vector)))
            if self.distance_metric == "dot":
                return float(np.dot(query, vector))
            if self.distance_metric in {"euclid", "euclidean"}:
                return float(1.0 / (1.0 + np.linalg.norm(query - vector)))
            if self.distance_metric == "manhattan":
                return float(1.0 / (1.0 + np.abs(query - vector).sum()))
            return float(np.dot(self._normalize(query), self._normalize(vector)))

        def _matches_filter(self, payload: Dict[str, Any], filter: Optional[Dict[str, Any]]) -> bool:
            if not filter:
                return True
            for key, value in filter.items():
                if payload.get(key) != value:
                    return False
            return True

        def add(self, vectors, ids=None, payloads=None):
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {vectors.shape}")
            payloads = payloads or [{} for _ in range(len(vectors))]
            if ids is None:
                ids = list(range(self.next_id, self.next_id + len(vectors)))
            for idx, vector in enumerate(vectors):
                item_id = int(ids[idx])
                self.items[item_id] = {
                    "vector": vector,
                    "payload": dict(payloads[idx] or {}),
                }
                self.next_id = max(self.next_id, item_id + 1)
            self._save()
            return ids

        def add_multidense(self, vectors, ids=None, payloads=None):
            payloads = payloads or [{} for _ in range(len(vectors))]
            if ids is None:
                ids = list(range(self.next_id, self.next_id + len(vectors)))
            for idx, matrix in enumerate(vectors):
                array = np.asarray(matrix, dtype=np.float32)
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                if array.ndim != 2:
                    raise ValueError(f"Expected 2D array, got shape {array.shape}")
                item_id = int(ids[idx])
                self.items[item_id] = {
                    "vector": array,
                    "payload": dict(payloads[idx] or {}),
                }
                self.next_id = max(self.next_id, item_id + 1)
            self._save()
            return ids

        def search(self, query, k=10, ef=None, filter=None):
            query = np.asarray(query, dtype=np.float32)
            results: List[Tuple[int, float]] = []
            for item_id, item in self.items.items():
                payload = item.get("payload", {})
                if not self._matches_filter(payload, filter):
                    continue
                vector = item["vector"]
                if vector.ndim == 2:
                    if self.multivector_comparator == "max_sim":
                        score = max(self._score(query, token) for token in vector)
                    else:
                        score = self._score(query, vector.mean(axis=0))
                else:
                    score = self._score(query, vector)
                results.append((item_id, score))
            results.sort(key=lambda item: item[1], reverse=True)
            return results[:k]

        def retrieve(self, ids):
            results = []
            for item_id in ids:
                item = self.items.get(int(item_id))
                if item is None:
                    continue
                results.append((int(item_id), item["vector"], item.get("payload", {})))
            return results

        def flush(self):
            self._save()

        def len(self):
            return len(self.items)

        def delete(self, ids):
            deleted = 0
            for item_id in ids:
                if self.items.pop(int(item_id), None) is not None:
                    deleted += 1
            self._save()
            return deleted

        def upsert_payload(self, id, payload):
            item_id = int(id)
            if item_id not in self.items:
                return
            self.items[item_id]["payload"] = {
                **self.items[item_id].get("payload", {}),
                **dict(payload or {}),
            }
            self._save()

    logging.getLogger(__name__).warning(
        "latence_hnsw is unavailable; falling back to a local Python segment. "
        "This path is suitable for development, testing, and small local indexes."
    )

from colsearch._internal.inference.quantization.rotational import RoQConfig, RotationalQuantizer


class HnswSegmentManager:
    """
    Manages HNSW segments for a single shard using Qdrant's library.

    Architecture:
    - active_segment: Mutable in-memory HNSW
    - sealed_segments: Read-only mmap'd HNSW segments
    """

    def __init__(
        self,
        shard_path: Path,
        dim: int,
        distance_metric: str = "cosine",
        m: int = 16,
        ef_construct: int = 100,
        on_disk: bool = False,
        multivector_comparator: Optional[str] = None,
        roq_bits: Optional[int] = None
    ):
        """
        Initialize segment manager for a shard.

        Args:
            shard_path: Base directory for this shard
            dim: Vector dimension
            distance_metric: 'cosine', 'dot', 'euclid', or 'manhattan'
            m: HNSW M parameter (edges per node)
            ef_construct: HNSW construction  parameter
            on_disk: If True, use RocksDB/Mmap persistence. If False, use RAM.
            multivector_comparator: 'max_sim' for ColBERT
            roq_bits: If set (4 or 8), enable Rotational Quantization
        """
        self.shard_path = Path(shard_path)
        self.dim = dim
        self.distance_metric = distance_metric
        self.m = m
        self.ef_construct = ef_construct
        self.on_disk = on_disk
        self.roq_bits = roq_bits

        # Initialize Quantizer if requested
        self.quantizer = None
        if roq_bits:
            logger.info(f"Initializing Rotational Quantizer ({roq_bits}-bit)")
            self.quantizer = RotationalQuantizer(
                RoQConfig(
                    dim=dim,
                    num_bits=roq_bits,
                )
            )

        if not self.shard_path.exists():
            self.shard_path.mkdir(parents=True)

        active_path = self.shard_path / "active"
        if not active_path.exists():
            active_path.mkdir(parents=True)

        # Qdrant might create a UUID subdirectory inside the path we give it.
        # Check if we have an existing segment in a subdirectory.
        real_active_path = active_path

        # Look for subdirectories containing segment.json
        existing_subdirs = [d for d in active_path.iterdir() if d.is_dir()]
        found_existing = False
        for d in existing_subdirs:
            if (d / "segment.json").exists():
                real_active_path = d
                found_existing = True
                break

        if found_existing:
             logger.info(f"Loading existing active segment from: {real_active_path}")
        else:
             logger.info(f"Creating new active segment at: {active_path}")

        # Active segment (appendable, Plain index)
        # Note: If we point to active_path and it creates a subdir, next time we must find it.
        # If we point to active_path/UUID (real_active_path), it should load it.
        self.active_segment = HnswSegment(
            str(real_active_path),
            dim=dim,
            distance_metric=distance_metric,
            m=m,
            ef_construct=ef_construct,
            on_disk=on_disk,
            is_appendable=True,  # Active segment is mutable
            multivector_comparator=multivector_comparator  # Pass comparator
        )

        # Sealed segments (immutable, mmap HNSW)
        self.sealed_segments: List[HnswSegment] = []
        self._load_sealed_segments()

        logger.info(
            f"Initialized HnswSegmentManager: "
            f"{len(self.sealed_segments)} sealed + 1 active segment "
            f"(on_disk={on_disk})"
        )

    def _load_sealed_segments(self):
        """Load existing sealed segments from disk."""
        sealed_dir = self.shard_path / "sealed"
        if not sealed_dir.exists():
            return

        for seg_dir in sorted(sealed_dir.iterdir()):
            if seg_dir.is_dir():
                try:
                    # Check if this dir has a subdir with segment.json (nested UUID)
                    # or if it is the segment itself.
                    # Qdrant persistence seems to prefer UUID subdirs.
                    target_path = seg_dir
                    subdirs = [d for d in seg_dir.iterdir() if d.is_dir()]
                    for sub in subdirs:
                        if (sub / "segment.json").exists():
                            target_path = sub
                            break

                    # Sealed segments are immutable and use HNSW
                    segment = HnswSegment(
                        str(target_path),
                        dim=self.dim,
                        distance_metric=self.distance_metric,
                        m=self.m,
                        ef_construct=self.ef_construct,
                        on_disk=self.on_disk,
                        is_appendable=False,
                    )
                    self.sealed_segments.append(segment)
                    logger.info(f"Loaded sealed segment: {seg_dir.name} (path: {target_path})")
                except Exception as e:
                    logger.warning(f"Failed to load segment {seg_dir}: {e}")

    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add vectors to the active segment.
        """
        # Quantize if enabled
        if hasattr(self, 'quantizer') and self.quantizer:
            q_res = self.quantizer.quantize(vectors, store=False)

            # Pack RoQ data into payloads
            # We need to distribute q_res dict items to list of payloads
            # RoQ dict: {'codes': (N, ...), 'scales': (N,), 'offsets': (N,), 'norms_sq': (N,)}

            if payloads is None:
                payloads = [{} for _ in range(len(vectors))]

            codes = q_res['codes']
            scales = q_res['scales']
            offsets = q_res['offsets']
            norms = q_res['norms_sq']

            # Assuming 4-bit packed codes
            # We store as list/bytes in payload to be JSON serializable?
            # Qdrant payload supports basic types. Binary might need encoding (base64 or list of ints).
            # List of ints (uint8) is safest for generic JSON.

            for i in range(len(vectors)):
                p = payloads[i] if payloads[i] is not None else {}

                # Store RoQ components
                # Codes: numpy array -> list
                p['roq_codes'] = codes[i].tolist()
                scale_value = scales[i]
                offset_value = offsets[i]
                p['roq_scale'] = (
                    np.asarray(scale_value, dtype=np.float32).tolist()
                    if np.ndim(scale_value) > 0
                    else float(scale_value)
                )
                p['roq_offset'] = (
                    np.asarray(offset_value, dtype=np.float32).tolist()
                    if np.ndim(offset_value) > 0
                    else float(offset_value)
                )
                p['roq_norm'] = float(norms[i])
                if 'code_sums' in q_res:
                     sum_value = q_res['code_sums'][i]
                     p['roq_sum'] = (
                         np.asarray(sum_value, dtype=np.float32).tolist()
                         if np.ndim(sum_value) > 0
                         else float(sum_value)
                     )
                if 'group_size' in q_res:
                    p['roq_group_size'] = int(q_res['group_size'])

                payloads[i] = p

        return self._add_internal(vectors, ids, payloads)

    def _add_internal(self, vectors, ids, payloads):
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dim}, "
f"got {vectors.shape[1]}"
            )

        # Convert payloads to Python dicts if needed
        if payloads is not None:
            payloads = [dict(p) if p is not None else {} for p in payloads]

        return self.active_segment.add(vectors, ids=ids, payloads=payloads)

    def add_multidense(
        self,
        vectors: List[np.ndarray],
        ids: Optional[List[int]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add Multi-Vector (ColBERT/MaxSim) points to the active segment.

        Args:
            vectors: List of (M, D) matrices. Each item is a single point (document) with M vectors.
            ids: Optional custom IDs
            payloads: Optional metadata
        """
        # Validate dimensions
        for i, mat in enumerate(vectors):
            if mat.ndim != 2:
                # Allow (D,) as (1,D) for single vectors mixed in?
                if mat.ndim == 1 and mat.shape[0] == self.dim:
                     vectors[i] = mat.reshape(1, -1)
                else:
                    raise ValueError(f"Expected 2D matrix for point {i}, got {mat.shape}")
            elif mat.shape[1] != self.dim:
                raise ValueError(f"Dim mismatch at {i}: expected {self.dim}, got {mat.shape[1]}")

        # Convert payloads
        if payloads is not None:
             payloads = [dict(p) if p is not None else {} for p in payloads]

        # Quantize if enabled
        if hasattr(self, 'quantizer') and self.quantizer:
             if payloads is None:
                  payloads = [{} for _ in range(len(vectors))]

             for i, mat in enumerate(vectors):
                  # mat is (M, D)
                  q_res = self.quantizer.quantize(mat, store=False)

                  p = payloads[i] if payloads[i] is not None else {}

                  # Store RoQ components for Multi-Vector
                  # Codes: (M, D) -> list of lists? or flatten?
                  # We should store as list of lists to preserve M structure in JSON?
                  # Or just flatten and trust shape inference?
                  # Let's store as nested list for clarity in JSON for now.
                  p['roq_codes'] = q_res['codes'].tolist()
                  p['roq_scale'] = q_res['scales'].tolist() # (M,)
                  p['roq_offset'] = q_res['offsets'].tolist() # (M,)
                  p['roq_norm'] = q_res['norms_sq'].tolist() # (M,)
                  if 'code_sums' in q_res:
                       p['roq_sum'] = q_res['code_sums'].tolist()
                  if 'group_size' in q_res:
                       p['roq_group_size'] = int(q_res['group_size'])

                  payloads[i] = p

        return self.active_segment.add_multidense(vectors, ids=ids, payloads=payloads)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float]]:
        """
        Search across all segments.

        Args:
            query: (D,) query vector
            k: Number of neighbors
            ef: HNSW ef search parameter (default: k * 2)
            filters: Optional payload filters

        Returns:
            List of (id, score) tuples, sorted by score desc
        """
        if query.ndim != 1:
            raise ValueError(f"Expected 1D query, got shape {query.shape}")

        if query.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dim}, "
                f"got {query.shape[0]}"
            )

        if ef is None:
            ef = k * 2

        all_results = []

        # Search active segment
        results = self.active_segment.search(
            query, k=k, ef=ef, filter=filters
        )
        all_results.extend(results)

        # Search sealed segments in parallel
        if self.sealed_segments:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(seg.search, query, k, ef, filters)
                    for seg in self.sealed_segments
                ]
                for future in futures:
                    try:
                        all_results.extend(future.result())
                    except Exception as e:
                        logger.warning(f"Sealed segment search failed: {e}")

        # Merge and re-rank
        return sorted(all_results, key=lambda x: x[1], reverse=True)[:k]

    def retrieve(self, ids: List[int], with_vector: bool = True, with_payload: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve vectors and payloads for specific IDs.
        Useful for feeding the Knapsack Solver.
        """
        results = {}

        # Check active
        try:
            # Rust retrieve signature is retrieve(ids) -> List[(id, vector, payload)]
            # It always returns contents if found.
            active_res = self.active_segment.retrieve(ids)
            for item in active_res:
                if isinstance(item, tuple) and len(item) == 3:
                     pid, vec, payload = item
                     # Only add if we found something (vector or payload)
                     if vec is not None or payload is not None:
                         results[pid] = {"id": pid, "vector": vec, "payload": payload}
        except (AttributeError, TypeError, ValueError):
             # logging.debug(f"Active segment retrieve failed: {e}")
             pass

        # Check sealed (if supported)
        for seg in self.sealed_segments:
            try:
                sealed_res = seg.retrieve(ids)
                for item in sealed_res:
                    if isinstance(item, tuple) and len(item) == 3:
                        pid, vec, payload = item
                        if pid not in results and (vec is not None or payload is not None):
                            results[pid] = {"id": pid, "vector": vec, "payload": payload}
            except Exception:
                pass

        return list(results.values())


    def seal_active_segment(self):
        """
        Move active segment to sealed (read-only).
        Creates a new active segment.
        """
        if self.active_segment.len() == 0:
            logger.info("Active segment is empty, skipping seal")
            return

        # Flush active segment
        self.active_segment.flush()

        # Move to sealed directory
        sealed_dir = self.shard_path / "sealed"
        sealed_dir.mkdir(exist_ok=True)

        segment_id = len(self.sealed_segments)
        sealed_path = sealed_dir / f"segment_{segment_id:04d}"

        # Move active to sealed
        active_path = self.shard_path / "active"
        active_path.rename(sealed_path)

        # Load as sealed segment
        sealed_segment = HnswSegment(str(sealed_path), dim=self.dim)
        self.sealed_segments.append(sealed_segment)

        # Create new active segment
        self.active_segment = HnswSegment(
            str(active_path),
            dim=self.dim,
            distance_metric=self.distance_metric,
            m=self.m,
            ef_construct=self.ef_construct
        )

        logger.info(f"Sealed segment {segment_id}, created new active segment")

    def delete(self, ids: List[int]) -> int:
        """
        Delete points by ID from all segments when supported.
        """
        deleted = self.active_segment.delete(ids)
        for segment in self.sealed_segments:
            try:
                deleted += segment.delete(ids)
            except Exception:
                continue
        return deleted

    def upsert_payload(self, id: int, payload: Dict[str, Any]):
        """
        Update payload for a point in the active segment.
        TODO: Support updates in sealed segments.
        """
        self.active_segment.upsert_payload(id, payload)

    def total_vectors(self) -> int:
        """Get total vector count across all segments."""
        total = self.active_segment.len()
        for seg in self.sealed_segments:
            total += seg.len()
        return total

    def flush(self):
        """Flush active segment to disk."""
        self.active_segment.flush()

    def close(self):
        """Release native segment handles so the shard can be reopened in-process."""
        active_segment = getattr(self, "active_segment", None)
        if active_segment is not None:
            try:
                active_segment.flush()
            except Exception:
                pass
        self.sealed_segments = []
        self.active_segment = None
        gc.collect()
