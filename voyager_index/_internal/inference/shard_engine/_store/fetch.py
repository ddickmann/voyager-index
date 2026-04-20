"""Shard read and selective fetch helpers."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .common import _MMAP_AVAILABLE, _st_safe_open
from .manifest import DocMeta

class ShardStoreFetchMixin:
    def _load_raw_shard(self, shard_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and cache decoded shard data. Returns (emb, offsets_tensor, ids_tensor)."""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required: pip install safetensors")
        cached = self._shard_cache.get(shard_id)
        if cached is not None:
            return cached

        meta = self._meta_by_id(shard_id)
        shard_path = self.shard_dir / meta.file_name
        try:
            data = st_load(str(shard_path), device="cpu")
        except Exception as exc:
            raise IOError(f"Failed to load shard file {shard_path}: {exc}") from exc
        try:
            emb = self._decode_embeddings(meta, data)
            offsets_t = data["doc_offsets"]
            ids_t = data["doc_ids"]
        except KeyError as exc:
            raise KeyError(
                f"Missing tensor key {exc} in shard {shard_path} "
                f"(compression={meta.compression})"
            ) from exc
        self._shard_cache.put(shard_id, emb, offsets_t, ids_t)
        return emb, offsets_t, ids_t

    def load_shard(self, shard_id: int, device: str = "cpu") -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        emb, offsets_t, ids_t = self._load_raw_shard(shard_id)
        if device != "cpu":
            emb = emb.to(device)
        offsets = [(int(offsets_t[i, 0]), int(offsets_t[i, 1])) for i in range(offsets_t.shape[0])]
        doc_ids = ids_t.tolist()
        return emb, offsets, doc_ids

    def load_docs_from_shard(
        self,
        shard_id: int,
        doc_ids: List[int],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        """Load only specific documents from a shard (doc-selective access).

        When safetensors mmap is available, reads only the requested byte
        ranges from disk instead of loading the entire shard file.
        """
        if not doc_ids:
            dim = self.manifest.dim if self.manifest else 128
            return torch.empty((0, dim), dtype=torch.float16), [], []

        selected = [
            (int(d), self._doc_index[int(d)])
            for d in doc_ids
            if int(d) in self._doc_index and self._doc_index[int(d)].shard_id == shard_id
        ]
        if not selected:
            dim = self.manifest.dim if self.manifest else 128
            return torch.empty((0, dim), dtype=torch.float16), [], []

        if _MMAP_AVAILABLE:
            return self._load_docs_selective(shard_id, selected, device)

        emb, _offsets_t, _ids_t = self._load_raw_shard(shard_id)
        pieces: List[torch.Tensor] = []
        offsets: List[Tuple[int, int]] = []
        out_ids: List[int] = []
        pos = 0
        for doc_id, meta_doc in selected:
            piece = emb[meta_doc.local_offset_start:meta_doc.local_offset_end]
            pieces.append(piece)
            offsets.append((pos, pos + piece.shape[0]))
            out_ids.append(doc_id)
            pos += int(piece.shape[0])

        merged = torch.cat(pieces, dim=0)
        if device != "cpu":
            merged = merged.to(device)
        return merged, offsets, out_ids

    def _load_docs_selective(
        self,
        shard_id: int,
        selected: List[Tuple[int, "DocMeta"]],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        """Mmap-backed selective doc loading — reads only the requested rows.

        Uses ``safetensors.safe_open`` + ``get_slice`` so the OS only pages
        in the byte ranges for the requested documents, not the entire shard.
        Falls back to the eager ``_load_raw_shard`` path if ``get_slice`` is
        unavailable (older safetensors) or if the shard lacks an embeddings key.
        """
        meta = self._meta_by_id(shard_id)
        shard_path = self.shard_dir / meta.file_name
        is_int8 = meta.compression == "int8"

        try:
            return self._do_selective_read(shard_path, is_int8, selected, device)
        except (AttributeError, KeyError, OSError) as exc:
            logger.debug("Selective mmap read fell back to eager load: %s", exc)
            return self._load_docs_eager_fallback(shard_id, selected, device)

    def _do_selective_read(
        self,
        shard_path: Path,
        is_int8: bool,
        selected: List[Tuple[int, "DocMeta"]],
        device: str,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        ordered = [
            (idx, int(doc_id), dm)
            for idx, (doc_id, dm) in enumerate(selected)
        ]
        sorted_selected = sorted(ordered, key=lambda item: (item[2].local_offset_start, item[2].local_offset_end))
        runs: List[List[Tuple[int, int, "DocMeta"]]] = []
        current_run: List[Tuple[int, int, "DocMeta"]] = []
        current_end = -1

        for item in sorted_selected:
            _idx, _doc_id, dm = item
            if not current_run or dm.local_offset_start > current_end:
                if current_run:
                    runs.append(current_run)
                current_run = [item]
            else:
                current_run.append(item)
            current_end = max(current_end, int(dm.local_offset_end))
        if current_run:
            runs.append(current_run)

        ordered_pieces: Dict[int, torch.Tensor] = {}

        with _st_safe_open(str(shard_path), framework="pt", device="cpu") as f:
            emb_slice = f.get_slice("embeddings")
            scales_slice = f.get_slice("scales") if is_int8 else None

            for run in runs:
                run_start = int(run[0][2].local_offset_start)
                run_end = int(max(item[2].local_offset_end for item in run))
                if is_int8 and scales_slice is not None:
                    raw_run = emb_slice[run_start:run_end].float()
                    scale_run = scales_slice[run_start:run_end].unsqueeze(-1)
                    run_piece = (raw_run * scale_run).to(torch.float16)
                else:
                    run_piece = emb_slice[run_start:run_end].to(torch.float16)

                for idx, _doc_id, dm in run:
                    rel_start = int(dm.local_offset_start - run_start)
                    rel_end = int(dm.local_offset_end - run_start)
                    ordered_pieces[idx] = run_piece[rel_start:rel_end]

        if not ordered_pieces:
            dim = self.manifest.dim if self.manifest else 128
            return torch.empty((0, dim), dtype=torch.float16), [], []

        pieces: List[torch.Tensor] = []
        offsets: List[Tuple[int, int]] = []
        out_ids: List[int] = []
        pos = 0
        for idx, (doc_id, _dm) in enumerate(selected):
            piece = ordered_pieces.get(idx)
            if piece is None:
                continue
            pieces.append(piece)
            offsets.append((pos, pos + piece.shape[0]))
            out_ids.append(int(doc_id))
            pos += int(piece.shape[0])

        merged = torch.cat(pieces, dim=0)
        if device != "cpu":
            merged = merged.to(device)
        return merged, offsets, out_ids

    def _load_docs_eager_fallback(
        self,
        shard_id: int,
        selected: List[Tuple[int, "DocMeta"]],
        device: str,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        """Fallback: decode the full shard and slice (pre-C1 behaviour)."""
        emb, _offsets_t, _ids_t = self._load_raw_shard(shard_id)
        pieces: List[torch.Tensor] = []
        offsets: List[Tuple[int, int]] = []
        out_ids: List[int] = []
        pos = 0
        for doc_id, meta_doc in selected:
            piece = emb[meta_doc.local_offset_start:meta_doc.local_offset_end]
            pieces.append(piece)
            offsets.append((pos, pos + piece.shape[0]))
            out_ids.append(doc_id)
            pos += int(piece.shape[0])
        if not pieces:
            dim = self.manifest.dim if self.manifest else 128
            return torch.empty((0, dim), dtype=torch.float16), [], []
        merged = torch.cat(pieces, dim=0)
        if device != "cpu":
            merged = merged.to(device)
        return merged, offsets, out_ids

    def load_shards(self, shard_ids: List[int], device: str = "cpu") -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        all_emb = []
        all_offsets = []
        all_ids = []
        global_offset = 0
        for sid in shard_ids:
            emb, offsets, doc_ids = self.load_shard(sid, device=device)
            for s, e in offsets:
                all_offsets.append((global_offset + s, global_offset + e))
            global_offset += emb.shape[0]
            all_emb.append(emb)
            all_ids.extend(doc_ids)
        if not all_emb:
            dim = self.manifest.dim if self.manifest else 128
            return torch.empty((0, dim), dtype=torch.float16), [], []
        return torch.cat(all_emb, dim=0), all_offsets, all_ids

    def load_shard_to_pinned(
        self,
        shard_id: int,
        pinned_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
        emb, offsets, doc_ids = self.load_shard(shard_id, device="cpu")
        n_tokens = emb.shape[0]
        if pinned_buffer is not None and pinned_buffer.shape[0] >= n_tokens:
            pinned_buffer[:n_tokens].copy_(emb)
            return pinned_buffer[:n_tokens], offsets, doc_ids
        pinned = torch.empty_like(emb, pin_memory=True)
        pinned.copy_(emb)
        return pinned, offsets, doc_ids

    def load_merged_layout(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Load merged-mmap document order and token offsets."""
        merged_path = self.root / "merged_embeddings.bin"
        offsets_path = self.root / "merged_offsets.bin"
        doc_map_path = self.root / "merged_doc_map.bin"
        if not merged_path.exists():
            raise FileNotFoundError(f"Missing merged embeddings at {merged_path}")
        if not offsets_path.exists():
            raise FileNotFoundError(f"Missing merged offsets at {offsets_path}")
        if not doc_map_path.exists():
            raise FileNotFoundError(f"Missing merged doc map at {doc_map_path}")

        with open(merged_path, "rb") as f:
            emb_header = f.read(16)
        if len(emb_header) != 16:
            raise ValueError("merged_embeddings.bin header is truncated")
        total_tokens = int(np.frombuffer(emb_header[:8], dtype=np.int64)[0])
        dim = int(np.frombuffer(emb_header[8:], dtype=np.int64)[0])

        with open(offsets_path, "rb") as f:
            offsets_header = f.read(8)
            offsets_bytes = f.read()
        if len(offsets_header) != 8:
            raise ValueError("merged_offsets.bin header is truncated")
        n_entries = int(np.frombuffer(offsets_header, dtype=np.int64)[0])
        offsets = np.frombuffer(offsets_bytes, dtype=np.int64, count=n_entries).copy()

        with open(doc_map_path, "rb") as f:
            doc_map_header = f.read(8)
            doc_map_bytes = f.read()
        if len(doc_map_header) != 8:
            raise ValueError("merged_doc_map.bin header is truncated")
        n_docs = int(np.frombuffer(doc_map_header, dtype=np.int64)[0])
        doc_ids = np.frombuffer(doc_map_bytes, dtype=np.uint64, count=n_docs).astype(np.int64, copy=True)

        if len(offsets) != len(doc_ids) + 1:
            raise ValueError(
                f"merged layout mismatch: {len(doc_ids)} doc ids but {len(offsets)} offsets",
            )
        return doc_ids, offsets, dim, total_tokens

    def iter_merged_doc_chunks(
        self,
        chunk_docs: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
        """Yield merged-mmap documents as chunked dense CPU batches.

        Each yielded item is ``(doc_ids, emb_chunk, mask_chunk, chunk_max_tok, global_max_tok)``.
        The dense chunks are bounded by *chunk_docs* so GPU preload can stream
        without materializing the full corpus on the host.
        """
        merged_path = self.root / "merged_embeddings.bin"
        doc_ids, offsets, dim, total_tokens = self.load_merged_layout()
        lengths = offsets[1:] - offsets[:-1]
        global_max_tok = int(lengths.max()) if lengths.size else 1

        emb_mmap = np.memmap(
            str(merged_path),
            mode="r",
            dtype=np.float16,
            offset=16,
            shape=(total_tokens, dim),
        )

        n_docs = len(doc_ids)
        for start_idx in range(0, n_docs, max(1, int(chunk_docs))):
            end_idx = min(start_idx + int(chunk_docs), n_docs)
            ids_chunk = doc_ids[start_idx:end_idx]
            starts_chunk = offsets[start_idx:end_idx]
            ends_chunk = offsets[start_idx + 1:end_idx + 1]
            lengths_chunk = ends_chunk - starts_chunk
            chunk_max_tok = int(lengths_chunk.max()) if lengths_chunk.size else 1

            emb_chunk = np.zeros((len(ids_chunk), chunk_max_tok, dim), dtype=np.float16)
            mask_chunk = np.zeros((len(ids_chunk), chunk_max_tok), dtype=np.float32)
            for row_idx, (tok_start, tok_end) in enumerate(zip(starts_chunk, ends_chunk)):
                tok_start_i = int(tok_start)
                tok_end_i = int(tok_end)
                tok_len = tok_end_i - tok_start_i
                if tok_len <= 0:
                    continue
                emb_chunk[row_idx, :tok_len] = emb_mmap[tok_start_i:tok_end_i]
                mask_chunk[row_idx, :tok_len] = 1.0

            yield ids_chunk, emb_chunk, mask_chunk, chunk_max_tok, global_max_tok

    def fetch_docs(self, doc_ids: List[int]) -> Dict[int, np.ndarray]:
        """Fetch individual documents by ID. Returns {doc_id: np.ndarray}.

        Performance note: access is O(shard_size) per shard because the
        full shard tensor must be decoded.  The internal ``_ShardLRU``
        cache amortises cost for repeated accesses to the same shard.
        """
        result: Dict[int, np.ndarray] = {}
        by_shard: Dict[int, List[int]] = {}
        for did in doc_ids:
            if did in self._doc_index:
                sid = self._doc_index[did].shard_id
                by_shard.setdefault(sid, []).append(did)
        for sid, dids in by_shard.items():
            emb, offsets, loaded_ids = self.load_docs_from_shard(sid, dids, device="cpu")
            for i, loaded_did in enumerate(loaded_ids):
                s, e = offsets[i]
                result[loaded_did] = emb[s:e].numpy().astype(np.float32)
        return result

    def load_shard_roq4(self, shard_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load ROQ4-specific codes and meta for a shard.

        Returns (codes, meta, offsets_tensor, ids_tensor) or None if not ROQ4.
        """
        import importlib

        compat_mod = importlib.import_module("voyager_index._internal.inference.shard_engine.shard_store")
        safetensors_available = getattr(compat_mod, "SAFETENSORS_AVAILABLE", SAFETENSORS_AVAILABLE)
        if not safetensors_available:
            raise ImportError("safetensors is required: pip install safetensors")
        meta = self._meta_by_id(shard_id)
        if meta.compression != "roq4":
            return None
        data = st_load(str(self.shard_dir / meta.file_name), device="cpu")
        if "roq_codes" not in data:
            return None
        return data["roq_codes"], data["roq_meta"], data["doc_offsets"], data["doc_ids"]

    def load_shard_rroq158(self, shard_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load RROQ158-specific tensors for a shard.

        Returns dict with keys
        ``{sign_plane, nonzero_plane, scales, centroid_id, cos_norm,
        sin_norm, doc_offsets, doc_ids}`` or ``None`` if the shard isn't
        RROQ158 / can't be parsed.
        """
        import importlib

        compat_mod = importlib.import_module("voyager_index._internal.inference.shard_engine.shard_store")
        safetensors_available = getattr(compat_mod, "SAFETENSORS_AVAILABLE", SAFETENSORS_AVAILABLE)
        if not safetensors_available:
            raise ImportError("safetensors is required: pip install safetensors")
        meta = self._meta_by_id(shard_id)
        if meta.compression != "rroq158":
            return None
        data = st_load(str(self.shard_dir / meta.file_name), device="cpu")
        if "rroq158_sign_plane" not in data:
            return None
        return {
            "sign_plane": data["rroq158_sign_plane"],
            "nonzero_plane": data["rroq158_nonzero_plane"],
            "scales": data["rroq158_scales"],
            "centroid_id": data["rroq158_centroid_id"],
            "cos_norm": data["rroq158_cos_norm"],
            "sin_norm": data["rroq158_sin_norm"],
            "doc_offsets": data["doc_offsets"],
            "doc_ids": data["doc_ids"],
        }

    def load_shard_rroq4_riem(self, shard_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load RROQ4_RIEM-specific tensors for a shard.

        Returns dict with keys
        ``{codes, mins, deltas, centroid_id, cos_norm, sin_norm,
        doc_offsets, doc_ids}`` or ``None`` if the shard isn't
        RROQ4_RIEM / can't be parsed. ``codes`` is a packed uint8 nibble
        stream (two 4-bit codes per byte), ``mins`` and ``deltas`` are
        per-(token, group) float16 codebook parameters.
        """
        import importlib

        compat_mod = importlib.import_module("voyager_index._internal.inference.shard_engine.shard_store")
        safetensors_available = getattr(compat_mod, "SAFETENSORS_AVAILABLE", SAFETENSORS_AVAILABLE)
        if not safetensors_available:
            raise ImportError("safetensors is required: pip install safetensors")
        meta = self._meta_by_id(shard_id)
        if meta.compression != "rroq4_riem":
            return None
        data = st_load(str(self.shard_dir / meta.file_name), device="cpu")
        if "rroq4_riem_codes" not in data:
            return None
        return {
            "codes": data["rroq4_riem_codes"],
            "mins": data["rroq4_riem_mins"],
            "deltas": data["rroq4_riem_deltas"],
            "centroid_id": data["rroq4_riem_centroid_id"],
            "cos_norm": data["rroq4_riem_cos_norm"],
            "sin_norm": data["rroq4_riem_sin_norm"],
            "doc_offsets": data["doc_offsets"],
            "doc_ids": data["doc_ids"],
        }

