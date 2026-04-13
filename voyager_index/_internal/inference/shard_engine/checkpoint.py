"""Atomic checkpoint manager for shard engine memtable state.

Writes snapshots to a temporary directory and atomically renames to
``checkpoints/current/`` under the shard path, mirroring the GEM
``CheckpointManager`` pattern from ``gem_wal.py``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class ShardCheckpointManager:
    """Atomic save / load of memtable checkpoints.

    Layout on disk::

        <shard_path>/
          checkpoints/
            current/          <- atomically renamed from .tmp/
              meta.json       <- CheckpointData (minus tensor data)
              vectors.npz     <- per-doc numpy arrays
    """

    def __init__(self, shard_path: Path):
        self._root = Path(shard_path) / "checkpoints"
        self._current = self._root / "current"

    @property
    def exists(self) -> bool:
        return (self._current / "meta.json").exists()

    def save(
        self,
        memtable_docs: Dict[int, np.ndarray],
        payloads: Dict[int, dict],
        tombstones: Set[int],
        next_doc_id: int,
        wal_offset: int,
        sealed_payloads: Optional[Dict[int, dict]] = None,
    ) -> None:
        """Serialise snapshot to disk with atomic rename."""
        self._root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=self._root, prefix=".tmp_ckpt_"))
        try:
            doc_meta: Dict[str, Any] = {}
            np_arrs: Dict[str, np.ndarray] = {}
            for doc_id, vecs in memtable_docs.items():
                key = str(doc_id)
                arr = np.asarray(vecs, dtype=np.float32)
                doc_meta[key] = {"shape": list(arr.shape), "dtype": "float32"}
                np_arrs[f"doc_{key}"] = arr

            if np_arrs:
                np.savez(tmp_dir / "vectors.npz", **np_arrs)

            payload_ser = {str(k): v for k, v in payloads.items()}
            sealed_ser = {str(k): v for k, v in (sealed_payloads or {}).items()}
            meta = {
                "wal_offset": wal_offset,
                "next_doc_id": next_doc_id,
                "tombstones": sorted(tombstones),
                "payloads": payload_ser,
                "sealed_payloads": sealed_ser,
                "docs": doc_meta,
            }
            meta_path = tmp_dir / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f)
                f.flush()
                os.fsync(f.fileno())

            target = self._current
            old = self._root / ".old_ckpt"
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            if target.exists():
                target.rename(old)
            tmp_dir.rename(target)
            try:
                dir_fd = os.open(str(self._root), os.O_RDONLY)
                os.fsync(dir_fd)
                os.close(dir_fd)
            except OSError:
                pass
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            logger.debug("Checkpoint saved: %d docs, wal_offset=%d", len(memtable_docs), wal_offset)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if present.

        Returns a dict with keys: ``wal_offset``, ``next_doc_id``,
        ``tombstones`` (set[int]), ``payloads`` (dict[int,dict]),
        ``sealed_payloads`` (dict[int,dict]),
        ``docs`` (dict[int, np.ndarray]) -- or ``None``.
        """
        meta_path = self._current / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)

            docs: Dict[int, np.ndarray] = {}
            vectors_path = self._current / "vectors.npz"
            if vectors_path.exists():
                npz = np.load(vectors_path)
                for key in npz.files:
                    doc_id = int(key.split("_", 1)[1])
                    docs[doc_id] = npz[key]
                npz.close()

            payloads = {int(k): v for k, v in meta.get("payloads", {}).items()}
            sealed_payloads = {int(k): v for k, v in meta.get("sealed_payloads", {}).items()}
            tombstones = set(meta.get("tombstones", []))

            return {
                "wal_offset": meta["wal_offset"],
                "next_doc_id": meta["next_doc_id"],
                "tombstones": tombstones,
                "payloads": payloads,
                "sealed_payloads": sealed_payloads,
                "docs": docs,
            }
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)
            return None

    def clear(self) -> None:
        """Remove checkpoint directory."""
        if self._current.exists():
            shutil.rmtree(self._current, ignore_errors=True)
