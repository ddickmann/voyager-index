"""Shard fetch orchestration and pipelined exact-stage execution."""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..config import TransferMode
from ..interfaces import StoreProtocol
from ..profiler import Timer
from ..shard_store import ShardStore
from .common import ShardChunk
from .pinned import PinnedBufferPool

class FetchPipeline:
    """
    Orchestrates shard fetching from disk/CPU to GPU.

    Primary API for the fast path is fetch_per_shard(), which returns a list
    of per-shard chunks that can be scored independently (no cross-shard padding).

    pipelined_search() overlaps fetch with scoring for maximum throughput.
    """

    def __init__(
        self,
        store: StoreProtocol,
        mode: TransferMode = TransferMode.PINNED,
        pinned_pool: Optional[PinnedBufferPool] = None,
        device: str = "cuda",
    ):
        self.store = store
        self.mode = mode
        self.pool = pinned_pool
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def capability_snapshot(self) -> Dict[str, object]:
        return {
            "transfer_mode": self.mode.value if isinstance(self.mode, TransferMode) else str(self.mode),
            "device": str(self.device),
            "pinned_staging_enabled": bool(self.pool and self.pool.uses_pinned_memory),
            "pinned_staging_mode": self.pool.staging_mode if self.pool is not None else None,
        }

    # ------------------------------------------------------------------
    # New fast path: per-shard chunks
    # ------------------------------------------------------------------

    def fetch_per_shard(
        self,
        shard_ids: List[int],
        max_docs: int = 0,
    ) -> Tuple[List[ShardChunk], dict]:
        """
        Fetch shards and return per-shard (flat_emb, offsets, doc_ids) chunks.

        Always uses CPU load (no pinned staging) because per-shard tensors are
        small enough that pinned allocation overhead exceeds H2D benefit.
        The caller's scoring loop handles H2D inside _pad_shard_on_device().

        Returns:
            chunks: list of (flat_emb, offsets, doc_ids)
            stats:  dict with fetch_ms, h2d_bytes, num_shards, num_docs
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunks: List[ShardChunk] = []
        total_docs = 0
        total_bytes = 0

        def _load_one(shard_id: int) -> Optional[ShardChunk]:
            emb, offsets, dids = self.store.load_shard(shard_id, device="cpu")
            return (emb, offsets, dids)

        n_workers = min(8, len(shard_ids)) if shard_ids else 1

        with Timer() as t_fetch:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_idx = {
                    pool.submit(_load_one, sid): i
                    for i, sid in enumerate(shard_ids)
                }
                results_by_idx: Dict[int, ShardChunk] = {}
                for future in as_completed(future_to_idx):
                    results_by_idx[future_to_idx[future]] = future.result()

            for i in range(len(shard_ids)):
                if i not in results_by_idx:
                    continue
                emb, offsets, dids = results_by_idx[i]
                if max_docs and total_docs >= max_docs:
                    break

                if max_docs and total_docs + len(dids) > max_docs:
                    n_take = max_docs - total_docs
                    offsets = offsets[:n_take]
                    dids = dids[:n_take]
                    last_end = offsets[-1][1] if offsets else 0
                    emb = emb[:last_end]

                total_docs += len(dids)
                total_bytes += emb.nelement() * emb.element_size()
                chunks.append((emb, offsets, dids))

        stats = {
            "fetch_ms": t_fetch.elapsed_ms,
            "h2d_bytes": total_bytes,
            "num_shards": len(chunks),
            "num_docs": total_docs,
        }
        return chunks, stats

    def fetch_candidate_docs(
        self,
        docs_by_shard: Dict[int, List[int]],
        max_docs: int = 0,
    ) -> Tuple[List[ShardChunk], dict]:
        """Fetch only specific candidate docs, grouped by shard (LEMUR path).

        Uses a thread pool for parallel I/O across shards since safetensors
        mmap reads release the GIL during the actual I/O.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunks: List[ShardChunk] = []
        total_docs = 0
        total_h2d_bytes = 0

        items = list(docs_by_shard.items())
        if max_docs:
            budget = max_docs
            trimmed_items = []
            for shard_id, doc_ids in items:
                if budget <= 0:
                    break
                take = doc_ids[:budget]
                trimmed_items.append((shard_id, take))
                budget -= len(take)
            items = trimmed_items

        def _load_one(shard_id: int, request_ids: List[int]) -> Optional[ShardChunk]:
            emb, offsets, loaded_ids = self.store.load_docs_from_shard(
                shard_id, request_ids, device="cpu",
            )
            if not loaded_ids:
                return None
            return (emb, offsets, loaded_ids)

        n_workers = min(8, len(items)) if items else 1

        with Timer() as t_fetch:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_idx = {
                    pool.submit(_load_one, sid, dids): i
                    for i, (sid, dids) in enumerate(items)
                }
                results_by_idx: Dict[int, ShardChunk] = {}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    if result is not None:
                        results_by_idx[idx] = result

            for i in range(len(items)):
                if i in results_by_idx:
                    emb, offsets, loaded_ids = results_by_idx[i]
                    chunks.append((emb, offsets, loaded_ids))
                    total_docs += len(loaded_ids)
                    total_h2d_bytes += emb.nelement() * emb.element_size()

        return chunks, {
            "fetch_ms": t_fetch.elapsed_ms,
            "h2d_bytes": total_h2d_bytes,
            "num_shards": len(chunks),
            "num_docs": total_docs,
        }

    def pipelined_search(
        self,
        shard_ids: List[int],
        score_fn: Callable[[ShardChunk], None],
        max_docs: int = 0,
    ) -> dict:
        """
        Overlap disk fetch of shard N+1 with GPU scoring of shard N.

        score_fn receives one (flat_emb, offsets, doc_ids) chunk at a time.
        The caller accumulates scores in its own top-k heap.

        Returns stats dict.
        """
        total_docs = 0
        total_bytes = 0
        total_fetch_ms = 0.0

        ready_q: queue.Queue[Optional[ShardChunk]] = queue.Queue(maxsize=2)

        def _fetcher():
            nonlocal total_docs, total_bytes, total_fetch_ms
            for sid in shard_ids:
                if max_docs and total_docs >= max_docs:
                    break
                with Timer() as t:
                    if self.mode in (TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED):
                        emb, offsets, dids = self.store.load_shard_to_pinned(sid)
                    else:
                        emb, offsets, dids = self.store.load_shard(sid, device="cpu")
                total_fetch_ms += t.elapsed_ms

                if max_docs and total_docs + len(dids) > max_docs:
                    n_take = max_docs - total_docs
                    offsets = offsets[:n_take]
                    dids = dids[:n_take]
                    last_end = offsets[-1][1] if offsets else 0
                    emb = emb[:last_end]

                total_docs += len(dids)
                total_bytes += emb.nelement() * emb.element_size()
                ready_q.put((emb, offsets, dids))

            ready_q.put(None)  # sentinel

        fetch_thread = threading.Thread(target=_fetcher, daemon=True)
        fetch_thread.start()

        while True:
            chunk = ready_q.get()
            if chunk is None:
                break
            score_fn(chunk)

        fetch_thread.join()

        return {
            "fetch_ms": total_fetch_ms,
            "h2d_bytes": total_bytes,
            "num_shards": len(shard_ids),
            "num_docs": total_docs,
        }

    # ------------------------------------------------------------------
    # Legacy merged-tensor API (kept for baselines / backward compat)
    # ------------------------------------------------------------------

    def fetch(
        self,
        shard_ids: List[int],
        max_docs: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], dict]:
        """
        Legacy: fetch shards and deliver padded documents to GPU.
        Prefer fetch_per_shard() + score_shards_and_topk() for new code.
        """
        if self.mode == TransferMode.PAGEABLE:
            return self._fetch_pageable(shard_ids, max_docs)
        elif self.mode == TransferMode.PINNED:
            return self._fetch_pinned(shard_ids, max_docs)
        elif self.mode == TransferMode.DOUBLE_BUFFERED:
            return self._fetch_double_buffered(shard_ids, max_docs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _fetch_pageable(self, shard_ids, max_docs):
        with Timer() as t_fetch:
            emb_flat, offsets, doc_ids = self.store.load_shards(shard_ids, device="cpu")

        if max_docs and len(doc_ids) > max_docs:
            offsets = offsets[:max_docs]
            doc_ids = doc_ids[:max_docs]

        with Timer() as t_pad:
            doc_emb, doc_mask = _pad_docs(emb_flat, offsets)

        h2d_bytes = doc_emb.nelement() * doc_emb.element_size()
        with Timer(sync_cuda=True) as t_h2d:
            doc_emb = doc_emb.to(self.device)
            doc_mask = doc_mask.to(self.device)

        stats = {
            "fetch_ms": t_fetch.elapsed_ms,
            "pad_ms": t_pad.elapsed_ms,
            "h2d_ms": t_h2d.elapsed_ms,
            "h2d_bytes": h2d_bytes,
            "num_shards": len(shard_ids),
            "num_docs": len(doc_ids),
        }
        return doc_emb, doc_mask, doc_ids, stats

    def _fetch_pinned(self, shard_ids, max_docs):
        with Timer() as t_fetch:
            all_emb = []
            all_offsets = []
            all_ids = []
            global_offset = 0

            for sid in shard_ids:
                if self.pool:
                    emb, offsets, dids = self.store.load_shard_to_pinned(sid)
                else:
                    emb, offsets, dids = self.store.load_shard(sid, device="cpu")

                for s, e in offsets:
                    all_offsets.append((global_offset + s, global_offset + e))
                global_offset += emb.shape[0]
                all_emb.append(emb)
                all_ids.extend(dids)

            if not all_emb:
                dim = self.store.manifest.dim if self.store.manifest else 128
                emb_flat = torch.empty(0, dim, dtype=torch.float16)
            else:
                emb_flat = torch.cat(all_emb, dim=0)

        if max_docs and len(all_ids) > max_docs:
            all_offsets = all_offsets[:max_docs]
            all_ids = all_ids[:max_docs]

        with Timer() as t_pad:
            doc_emb, doc_mask = _pad_docs(emb_flat, all_offsets)

        h2d_bytes = doc_emb.nelement() * doc_emb.element_size()
        with Timer(sync_cuda=True) as t_h2d:
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    doc_emb_gpu = doc_emb.to(self.device, non_blocking=True)
                    doc_mask_gpu = doc_mask.to(self.device, non_blocking=True)
                self._stream.synchronize()
            else:
                doc_emb_gpu = doc_emb.to(self.device)
                doc_mask_gpu = doc_mask.to(self.device)

        stats = {
            "fetch_ms": t_fetch.elapsed_ms,
            "pad_ms": t_pad.elapsed_ms,
            "h2d_ms": t_h2d.elapsed_ms,
            "h2d_bytes": h2d_bytes,
            "num_shards": len(shard_ids),
            "num_docs": len(all_ids),
        }
        return doc_emb_gpu, doc_mask_gpu, all_ids, stats

    def _fetch_double_buffered(self, shard_ids, max_docs):
        """Overlap fetch of shard N+1 with H2D transfer of shard N."""
        if not self._stream:
            return self._fetch_pinned(shard_ids, max_docs)

        all_gpu_emb = []
        all_gpu_mask = []
        all_ids: List[int] = []
        total_fetch_ms = 0.0
        total_h2d_ms = 0.0
        total_h2d_bytes = 0

        pending_h2d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        for i, sid in enumerate(shard_ids):
            with Timer() as t_fetch:
                if self.pool:
                    emb, offsets, dids = self.store.load_shard_to_pinned(sid)
                else:
                    emb, offsets, dids = self.store.load_shard(sid, device="cpu")

            total_fetch_ms += t_fetch.elapsed_ms

            if max_docs and len(all_ids) + len(dids) > max_docs:
                n_take = max_docs - len(all_ids)
                offsets = offsets[:n_take]
                dids = dids[:n_take]

            doc_emb, doc_mask = _pad_docs(emb, offsets)
            h2d_bytes = doc_emb.nelement() * doc_emb.element_size()

            if pending_h2d is not None:
                self._stream.synchronize()

            with Timer(sync_cuda=False) as t_h2d:
                with torch.cuda.stream(self._stream):
                    gpu_emb = doc_emb.to(self.device, non_blocking=True)
                    gpu_mask = doc_mask.to(self.device, non_blocking=True)

            pending_h2d = (gpu_emb, gpu_mask)
            total_h2d_ms += t_h2d.elapsed_ms
            total_h2d_bytes += h2d_bytes

            all_gpu_emb.append(gpu_emb)
            all_gpu_mask.append(gpu_mask)
            all_ids.extend(dids)

            if max_docs and len(all_ids) >= max_docs:
                break

        if pending_h2d is not None:
            self._stream.synchronize()

        if not all_gpu_emb:
            dim = self.store.manifest.dim if self.store.manifest else 128
            empty = torch.empty(0, 1, dim, dtype=torch.float16, device=self.device)
            mask = torch.empty(0, 1, dtype=torch.float32, device=self.device)
            return empty, mask, [], {"fetch_ms": 0, "h2d_ms": 0, "h2d_bytes": 0, "num_shards": 0, "num_docs": 0}

        max_t = max(e.shape[1] for e in all_gpu_emb)
        dim = all_gpu_emb[0].shape[2]
        combined_emb = []
        combined_mask = []
        for e, m in zip(all_gpu_emb, all_gpu_mask):
            if e.shape[1] < max_t:
                pad = torch.zeros(e.shape[0], max_t - e.shape[1], dim, dtype=e.dtype, device=e.device)
                e = torch.cat([e, pad], dim=1)
                mpad = torch.zeros(m.shape[0], max_t - m.shape[1], dtype=m.dtype, device=m.device)
                m = torch.cat([m, mpad], dim=1)
            combined_emb.append(e)
            combined_mask.append(m)

        doc_emb_gpu = torch.cat(combined_emb, dim=0)
        doc_mask_gpu = torch.cat(combined_mask, dim=0)

        stats = {
            "fetch_ms": total_fetch_ms,
            "h2d_ms": total_h2d_ms,
            "h2d_bytes": total_h2d_bytes,
            "num_shards": len(shard_ids),
            "num_docs": len(all_ids),
        }
        return doc_emb_gpu, doc_mask_gpu, all_ids, stats

