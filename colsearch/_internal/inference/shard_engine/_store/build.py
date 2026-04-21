"""Shard build and write-side storage helpers."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .manifest import DocMeta, ShardMeta, StoreManifest

class ShardStoreBuildMixin:
    def build(
        self,
        all_vectors: np.ndarray,
        doc_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        shard_assignments: np.ndarray,
        n_shards: int,
        dim: int,
        compression: Compression = Compression.FP16,
        centroid_to_shard: Optional[Dict[int, int]] = None,
        uniform_shard_tokens: bool = False,
        roq_doc_codes: Optional[list] = None,
        roq_doc_meta: Optional[list] = None,
        rroq158_payload: Any = None,
        rroq4_riem_payload: Any = None,
    ) -> StoreManifest:
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required: pip install safetensors")

        if len(doc_offsets) != len(doc_ids):
            raise ValueError(f"doc_offsets ({len(doc_offsets)}) != doc_ids ({len(doc_ids)})")
        if len(shard_assignments) != len(doc_ids):
            raise ValueError(f"shard_assignments ({len(shard_assignments)}) != doc_ids ({len(doc_ids)})")
        if all_vectors.ndim != 2 or all_vectors.shape[1] != dim:
            raise ValueError(f"all_vectors shape {all_vectors.shape} incompatible with dim={dim}")
        if n_shards < 1:
            raise ValueError(f"n_shards must be >= 1, got {n_shards}")
        if len(shard_assignments) > 0:
            sa_min, sa_max = int(np.min(shard_assignments)), int(np.max(shard_assignments))
            if sa_min < 0 or sa_max >= n_shards:
                raise ValueError(
                    f"shard_assignments out of bounds [0, {n_shards}): min={sa_min}, max={sa_max}"
                )
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Duplicate doc_ids detected")
        if roq_doc_codes is not None and len(roq_doc_codes) != len(doc_ids):
            raise ValueError(
                f"roq_doc_codes length ({len(roq_doc_codes)}) != doc_ids ({len(doc_ids)})"
            )
        if roq_doc_meta is not None and len(roq_doc_meta) != len(doc_ids):
            raise ValueError(
                f"roq_doc_meta length ({len(roq_doc_meta)}) != doc_ids ({len(doc_ids)})"
            )

        if compression == Compression.ROQ4 and roq_doc_codes is None:
            logger.warning("ROQ4 requested but no roq_doc_codes provided; falling back to FP16")
            compression = Compression.FP16
        if compression == Compression.RROQ158 and rroq158_payload is None:
            logger.warning(
                "RROQ158 requested but no rroq158_payload provided; falling back to FP16"
            )
            compression = Compression.FP16
        if compression == Compression.RROQ4_RIEM and rroq4_riem_payload is None:
            logger.warning(
                "RROQ4_RIEM requested but no rroq4_riem_payload provided; falling back to FP16"
            )
            compression = Compression.FP16

        self.shard_dir.mkdir(parents=True, exist_ok=True)

        shard_to_centroid: Dict[int, List[int]] = {}
        if centroid_to_shard:
            for cid, sid in centroid_to_shard.items():
                shard_to_centroid.setdefault(int(sid), []).append(int(cid))

        all_token_counts = [e - s for s, e in doc_offsets]
        shard_metas: List[ShardMeta] = []
        doc_index_rows: List[Tuple[int, int, int, int, int]] = []

        # Compute one global target_len for the entire corpus
        global_target_len = 0
        if uniform_shard_tokens:
            tc_global = np.array(all_token_counts, dtype=np.float64)
            raw_p95 = int(np.ceil(np.percentile(tc_global, 95))) if len(tc_global) else 1
            raw_p95 = max(raw_p95, 1)
            global_target_len = 1
            while global_target_len < raw_p95:
                global_target_len <<= 1
            logger.info(
                "Global uniform target_len = %d (p95=%d, rounded to next pow2, %d docs)",
                global_target_len, raw_p95, len(tc_global),
            )

        for shard_id in range(n_shards):
            shard_doc_indices = np.where(shard_assignments == shard_id)[0]
            if len(shard_doc_indices) == 0:
                continue

            shard_doc_ids = [int(doc_ids[i]) for i in shard_doc_indices]
            shard_offsets = [doc_offsets[i] for i in shard_doc_indices]
            token_counts = [int(e - s) for s, e in shard_offsets]
            tc_arr = np.array(token_counts, dtype=np.float64)

            shard_max_tokens = 0

            if uniform_shard_tokens:
                target_len = global_target_len
                shard_max_tokens = target_len

                uniform_chunks = []
                local_offsets = []
                pos = 0
                for idx in shard_doc_indices:
                    s, e = doc_offsets[idx]
                    doc_vec = all_vectors[s:e]
                    n_tok = doc_vec.shape[0]
                    if n_tok >= target_len:
                        piece = doc_vec[:target_len]
                    else:
                        pad = np.zeros((target_len - n_tok, dim), dtype=doc_vec.dtype)
                        piece = np.concatenate([doc_vec, pad], axis=0)
                    uniform_chunks.append(piece)
                    local_offsets.append((pos, pos + target_len))
                    pos += target_len
                shard_vectors = np.concatenate(uniform_chunks, axis=0)
                token_counts = [target_len] * len(shard_doc_ids)
            else:
                chunks = [all_vectors[s:e] for s, e in shard_offsets]
                shard_vectors = np.concatenate(chunks, axis=0) if chunks else np.empty((0, dim), dtype=all_vectors.dtype)
                local_offsets = []
                pos = 0
                for tc in token_counts:
                    local_offsets.append((pos, pos + tc))
                    pos += tc

            file_name = f"shard_{shard_id:05d}.safetensors"
            shard_path = self.shard_dir / file_name

            if compression == Compression.ROQ4 and roq_doc_codes is not None:
                shard_codes_list = []
                shard_meta_list = []
                roq_offsets = []
                pos = 0
                for idx_in_shard, doc_global_idx in enumerate(shard_doc_indices):
                    codes_i = roq_doc_codes[doc_global_idx]
                    meta_i = roq_doc_meta[doc_global_idx]
                    n_tok = codes_i.shape[0]
                    if uniform_shard_tokens:
                        target = global_target_len
                        if n_tok >= target:
                            codes_i = codes_i[:target]
                            meta_i = meta_i[:target]
                        else:
                            pad_c = np.zeros((target - n_tok,) + codes_i.shape[1:], dtype=codes_i.dtype)
                            pad_m = np.zeros((target - n_tok,) + meta_i.shape[1:], dtype=meta_i.dtype)
                            codes_i = np.concatenate([codes_i, pad_c], axis=0)
                            meta_i = np.concatenate([meta_i, pad_m], axis=0)
                        n_tok = target
                    shard_codes_list.append(codes_i)
                    shard_meta_list.append(meta_i)
                    roq_offsets.append((pos, pos + n_tok))
                    pos += n_tok
                tensors = self.pack_shard_roq4(
                    shard_codes_list, shard_meta_list, roq_offsets, shard_doc_ids,
                )
                tensors["embeddings"] = shard_vectors.astype(np.float16)
            elif compression == Compression.RROQ158 and rroq158_payload is not None:
                shard_doc_token_ranges = []
                rroq_offsets = []
                pos = 0
                for idx_in_shard, doc_global_idx in enumerate(shard_doc_indices):
                    s, e = doc_offsets[doc_global_idx]
                    n_tok = e - s
                    if uniform_shard_tokens:
                        target = global_target_len
                        if n_tok >= target:
                            shard_doc_token_ranges.append((s, s + target))
                            n_tok = target
                        else:
                            shard_doc_token_ranges.append((s, e))
                    else:
                        shard_doc_token_ranges.append((s, e))
                    rroq_offsets.append((pos, pos + (target if uniform_shard_tokens else n_tok)))
                    pos += (target if uniform_shard_tokens else n_tok)
                tensors = self.pack_shard_rroq158(
                    rroq158_payload,
                    shard_doc_token_ranges,
                    rroq_offsets,
                    shard_doc_ids,
                    pad_to=global_target_len if uniform_shard_tokens else None,
                )
                tensors["embeddings"] = shard_vectors.astype(np.float16)
            elif compression == Compression.RROQ4_RIEM and rroq4_riem_payload is not None:
                shard_doc_token_ranges = []
                r4r_offsets = []
                pos = 0
                for idx_in_shard, doc_global_idx in enumerate(shard_doc_indices):
                    s, e = doc_offsets[doc_global_idx]
                    n_tok = e - s
                    if uniform_shard_tokens:
                        target = global_target_len
                        if n_tok >= target:
                            shard_doc_token_ranges.append((s, s + target))
                            n_tok = target
                        else:
                            shard_doc_token_ranges.append((s, e))
                    else:
                        shard_doc_token_ranges.append((s, e))
                    r4r_offsets.append((pos, pos + (target if uniform_shard_tokens else n_tok)))
                    pos += (target if uniform_shard_tokens else n_tok)
                tensors = self.pack_shard_rroq4_riem(
                    rroq4_riem_payload,
                    shard_doc_token_ranges,
                    r4r_offsets,
                    shard_doc_ids,
                    pad_to=global_target_len if uniform_shard_tokens else None,
                )
                tensors["embeddings"] = shard_vectors.astype(np.float16)
            else:
                tensors = self._pack_shard(shard_vectors, local_offsets, shard_doc_ids, compression)
            st_save_np(tensors, str(shard_path))
            byte_size = shard_path.stat().st_size

            for row_index, (doc_id, (start, end)) in enumerate(zip(shard_doc_ids, local_offsets)):
                doc_index_rows.append((doc_id, shard_id, int(start), int(end), row_index))

            meta = ShardMeta(
                shard_id=shard_id,
                num_docs=len(shard_doc_ids),
                total_tokens=int(shard_vectors.shape[0]),
                centroid_ids=shard_to_centroid.get(shard_id, []),
                byte_size=byte_size,
                file_name=file_name,
                compression=compression.value,
                p50_tokens=float(np.median(tc_arr)) if len(tc_arr) else 0.0,
                p95_tokens=float(np.percentile(tc_arr, 95)) if len(tc_arr) else 0.0,
                shard_max_tokens=shard_max_tokens,
            )
            shard_metas.append(meta)

        tc_all = np.array(all_token_counts, dtype=np.float64)
        self.manifest = StoreManifest(
            num_shards=len(shard_metas),
            num_docs=len(doc_ids),
            dim=dim,
            total_tokens=int(all_vectors.shape[0]),
            avg_tokens_per_chunk=float(np.mean(tc_all)) if len(tc_all) else 0.0,
            p50_tokens=float(np.median(tc_all)) if len(tc_all) else 0.0,
            p95_tokens=float(np.percentile(tc_all, 95)) if len(tc_all) else 0.0,
            compression=compression.value,
            shards=shard_metas,
            global_target_len=global_target_len,
        )
        self.manifest.save(self.manifest_path)
        self._meta_cache = {m.shard_id: m for m in shard_metas}
        self._save_doc_index(doc_index_rows)

        self.persist_merged_layout(all_vectors, doc_offsets, doc_ids, dim)

        logger.info(
            "ShardStore built: %d shards, %d docs, %d tokens, compression=%s, uniform=%s",
            len(shard_metas), len(doc_ids), int(all_vectors.shape[0]),
            compression.value, uniform_shard_tokens,
        )
        return self.manifest

    def _pack_shard(
        self,
        vectors: np.ndarray,
        local_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        compression: Compression,
    ) -> Dict[str, np.ndarray]:
        offsets_arr = np.array(local_offsets, dtype=np.int64)
        ids_arr = np.array(doc_ids, dtype=np.int64)

        if compression == Compression.FP16:
            return {
                "embeddings": vectors.astype(np.float16),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        if compression == Compression.INT8:
            vf = vectors.astype(np.float32)
            abs_max = np.abs(vf).max(axis=-1, keepdims=True)
            abs_max = np.where(abs_max == 0, 1.0, abs_max)
            scales = (abs_max / 127.0).astype(np.float32)
            quantized = np.clip(np.round(vf / scales), -127, 127).astype(np.int8)
            return {
                "embeddings": quantized,
                "scales": scales.squeeze(-1).astype(np.float32),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        if compression == Compression.ROQ4:
            logger.warning(
                "ROQ4 shard packing requires pre-encoded codes via pack_shard_roq4(); "
                "falling back to FP16 storage for this shard"
            )
            return {
                "embeddings": vectors.astype(np.float16),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        if compression == Compression.RROQ158:
            logger.warning(
                "RROQ158 shard packing requires pre-encoded payload via "
                "pack_shard_rroq158(); falling back to FP16 storage for this shard"
            )
            return {
                "embeddings": vectors.astype(np.float16),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        if compression == Compression.RROQ4_RIEM:
            logger.warning(
                "RROQ4_RIEM shard packing requires pre-encoded payload via "
                "pack_shard_rroq4_riem(); falling back to FP16 storage for this shard"
            )
            return {
                "embeddings": vectors.astype(np.float16),
                "doc_offsets": offsets_arr,
                "doc_ids": ids_arr,
            }
        raise ValueError(f"Unknown compression: {compression}")

    @staticmethod
    def pack_shard_rroq158(
        rroq158_payload: Any,
        token_ranges: List[Tuple[int, int]],
        local_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        *,
        pad_to: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Pack RROQ158-encoded tokens for a shard.

        ``rroq158_payload`` is the corpus-wide ``Rroq158Encoded``. Each
        ``(s, e)`` token range slices its per-token arrays. With uniform
        sharding (``pad_to`` set), each per-doc slice is padded with
        zeros (and a zero centroid_id) so all docs match the shard
        target_len; padding tokens are masked out at score time via the
        ``cos_norm`` / ``sin_norm`` zeros (they contribute zero to the
        approximate score).
        """
        sign_chunks: List[np.ndarray] = []
        nz_chunks: List[np.ndarray] = []
        scales_chunks: List[np.ndarray] = []
        cid_chunks: List[np.ndarray] = []
        cos_chunks: List[np.ndarray] = []
        sin_chunks: List[np.ndarray] = []
        for (s, e) in token_ranges:
            n_tok = e - s
            sp = rroq158_payload.sign_plane[s:e]
            np_ = rroq158_payload.nonzero_plane[s:e]
            sc = rroq158_payload.scales[s:e]
            ci = rroq158_payload.centroid_id[s:e]
            cn = rroq158_payload.cos_norm[s:e]
            sn = rroq158_payload.sin_norm[s:e]
            if pad_to is not None and n_tok < pad_to:
                pad = pad_to - n_tok
                sp = np.concatenate([sp, np.zeros((pad,) + sp.shape[1:], dtype=sp.dtype)], axis=0)
                np_ = np.concatenate([np_, np.zeros((pad,) + np_.shape[1:], dtype=np_.dtype)], axis=0)
                sc = np.concatenate([sc, np.zeros((pad,) + sc.shape[1:], dtype=sc.dtype)], axis=0)
                ci = np.concatenate([ci, np.zeros((pad,), dtype=ci.dtype)], axis=0)
                cn = np.concatenate([cn, np.zeros((pad,), dtype=cn.dtype)], axis=0)
                sn = np.concatenate([sn, np.zeros((pad,), dtype=sn.dtype)], axis=0)
            sign_chunks.append(sp)
            nz_chunks.append(np_)
            scales_chunks.append(sc)
            cid_chunks.append(ci)
            cos_chunks.append(cn)
            sin_chunks.append(sn)
        return {
            "rroq158_sign_plane": np.concatenate(sign_chunks, axis=0),
            "rroq158_nonzero_plane": np.concatenate(nz_chunks, axis=0),
            "rroq158_scales": np.concatenate(scales_chunks, axis=0).astype(np.float16),
            "rroq158_centroid_id": np.concatenate(cid_chunks, axis=0).astype(np.int32),
            "rroq158_cos_norm": np.concatenate(cos_chunks, axis=0).astype(np.float16),
            "rroq158_sin_norm": np.concatenate(sin_chunks, axis=0).astype(np.float16),
            "doc_offsets": np.array(local_offsets, dtype=np.int64),
            "doc_ids": np.array(doc_ids, dtype=np.int64),
        }

    @staticmethod
    def pack_shard_rroq4_riem(
        rroq4_riem_payload: Any,
        token_ranges: List[Tuple[int, int]],
        local_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        *,
        pad_to: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Pack RROQ4_RIEM-encoded tokens for a shard.

        Mirrors :func:`pack_shard_rroq158` but with the per-token tensors
        from a ``Rroq4RiemEncoded`` payload (4-bit packed codes + per-group
        mins/deltas instead of ternary planes + scales). Padding tokens get
        zeroed payloads so the per-token cos/sin norms zero them out at
        scoring time without needing an explicit mask in the kernel.
        """
        codes_chunks: List[np.ndarray] = []
        mins_chunks: List[np.ndarray] = []
        dlts_chunks: List[np.ndarray] = []
        cid_chunks: List[np.ndarray] = []
        cos_chunks: List[np.ndarray] = []
        sin_chunks: List[np.ndarray] = []
        for (s, e) in token_ranges:
            n_tok = e - s
            cp = rroq4_riem_payload.codes_packed[s:e]
            mn = rroq4_riem_payload.mins[s:e]
            dl = rroq4_riem_payload.deltas[s:e]
            ci = rroq4_riem_payload.centroid_id[s:e]
            cn = rroq4_riem_payload.cos_norm[s:e]
            sn = rroq4_riem_payload.sin_norm[s:e]
            if pad_to is not None and n_tok < pad_to:
                pad = pad_to - n_tok
                cp = np.concatenate([cp, np.zeros((pad,) + cp.shape[1:], dtype=cp.dtype)], axis=0)
                mn = np.concatenate([mn, np.zeros((pad,) + mn.shape[1:], dtype=mn.dtype)], axis=0)
                # Pad deltas with 1.0 not 0.0 so a stray dot-product against
                # a padding token still computes 0 (because cos/sin norms are
                # zeroed) but never tries to multiply by an exact-zero scale
                # that would generate NaNs in any numerical-debug path.
                dl = np.concatenate(
                    [dl, np.ones((pad,) + dl.shape[1:], dtype=dl.dtype)], axis=0
                )
                ci = np.concatenate([ci, np.zeros((pad,), dtype=ci.dtype)], axis=0)
                cn = np.concatenate([cn, np.zeros((pad,), dtype=cn.dtype)], axis=0)
                sn = np.concatenate([sn, np.zeros((pad,), dtype=sn.dtype)], axis=0)
            codes_chunks.append(cp)
            mins_chunks.append(mn)
            dlts_chunks.append(dl)
            cid_chunks.append(ci)
            cos_chunks.append(cn)
            sin_chunks.append(sn)
        return {
            "rroq4_riem_codes": np.concatenate(codes_chunks, axis=0).astype(np.uint8),
            "rroq4_riem_mins": np.concatenate(mins_chunks, axis=0).astype(np.float16),
            "rroq4_riem_deltas": np.concatenate(dlts_chunks, axis=0).astype(np.float16),
            "rroq4_riem_centroid_id": np.concatenate(cid_chunks, axis=0).astype(np.int32),
            "rroq4_riem_cos_norm": np.concatenate(cos_chunks, axis=0).astype(np.float16),
            "rroq4_riem_sin_norm": np.concatenate(sin_chunks, axis=0).astype(np.float16),
            "doc_offsets": np.array(local_offsets, dtype=np.int64),
            "doc_ids": np.array(doc_ids, dtype=np.int64),
        }

    @staticmethod
    def pack_shard_roq4(
        codes_list: list,
        meta_list: list,
        local_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
    ) -> Dict[str, np.ndarray]:
        """Pack ROQ4-encoded tokens into a shard file.

        codes_list/meta_list are per-doc arrays produced by RotationalQuantizer.
        Each codes entry is (n_tok, NB) uint8, each meta entry is (n_tok, 4) float32.
        """
        all_codes = np.concatenate(codes_list, axis=0) if codes_list else np.empty((0, 0), dtype=np.uint8)
        all_meta = np.concatenate(meta_list, axis=0) if meta_list else np.empty((0, 4), dtype=np.float32)
        return {
            "roq_codes": all_codes,
            "roq_meta": all_meta.astype(np.float32),
            "doc_offsets": np.array(local_offsets, dtype=np.int64),
            "doc_ids": np.array(doc_ids, dtype=np.int64),
        }

    def persist_merged_layout(
        self,
        all_vectors: np.ndarray,
        doc_offsets: List[Tuple[int, int]],
        doc_ids: List[int],
        dim: int,
    ) -> None:
        """Write merged mmap files for native Rust fused exact scoring.

        Produces ``merged_embeddings.bin``, ``merged_offsets.bin``, and
        ``merged_doc_map.bin`` at the store root.  These are consumed by
        ``latence_shard_engine.ShardIndex.load_merged()`` and enable the
        ``score_candidates_exact()`` CPU fast-path.
        """
        emb_path = self.root / "merged_embeddings.bin"
        off_path = self.root / "merged_offsets.bin"
        map_path = self.root / "merged_doc_map.bin"

        if emb_path.exists() and off_path.exists() and map_path.exists():
            return

        self.root.mkdir(parents=True, exist_ok=True)
        total_tokens = int(all_vectors.shape[0])
        f16 = np.ascontiguousarray(all_vectors.astype(np.float16, copy=False))

        offsets = np.array(
            [int(s) for s, _e in doc_offsets]
            + [int(doc_offsets[-1][1]) if doc_offsets else 0],
            dtype=np.int64,
        )
        doc_id_arr = np.array(doc_ids, dtype=np.uint64)

        with open(emb_path, "wb") as f:
            f.write(np.array([total_tokens], dtype=np.int64).tobytes())
            f.write(np.array([dim], dtype=np.int64).tobytes())
            f.write(f16.tobytes())

        with open(off_path, "wb") as f:
            f.write(np.array([len(offsets)], dtype=np.int64).tobytes())
            f.write(offsets.tobytes())

        with open(map_path, "wb") as f:
            f.write(np.array([len(doc_id_arr)], dtype=np.int64).tobytes())
            f.write(doc_id_arr.tobytes())

        logger.info(
            "Merged mmap layout persisted: %d docs, %d tokens at %s",
            len(doc_ids),
            total_tokens,
            self.root,
        )

    def _save_doc_index(self, rows: List[Tuple[int, int, int, int, int]]) -> None:
        rows = sorted(rows, key=lambda x: x[0])
        np.savez_compressed(
            self.doc_index_path,
            doc_ids=np.array([r[0] for r in rows], dtype=np.int64),
            shard_ids=np.array([r[1] for r in rows], dtype=np.int32),
            local_starts=np.array([r[2] for r in rows], dtype=np.int64),
            local_ends=np.array([r[3] for r in rows], dtype=np.int64),
            row_indices=np.array([r[4] for r in rows], dtype=np.int32),
        )
        self._load_doc_index()

