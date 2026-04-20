"""Lifecycle, persistence, and warmup helpers for the shard manager."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .common import _FileLock
from ..interfaces import (
    CandidateScorerProtocol,
    FetchPipelineProtocol,
    NativeExactBackendProtocol,
    RerankerProtocol,
    RouterProtocol,
    StoreProtocol,
)

class ShardSegmentManagerLifecycleMixin:
    def __init__(
        self,
        path: Path | str,
        config: ShardEngineConfig | None = None,
        device: str = "cuda",
    ):
        self._path = Path(path)
        self._config = config or ShardEngineConfig()
        self._device = device
        self._router_device = self._config.router_device or "cpu"
        self._lock = threading.RLock()

        self._store: Optional[StoreProtocol] = None
        self._router: Optional[RouterProtocol] = None
        self._pipeline: Optional[FetchPipelineProtocol] = None
        self._gpu_corpus: Optional[CandidateScorerProtocol] = None
        self._doc_vecs: Optional[list] = None
        self._doc_ids: Optional[List[int]] = None
        self._dim: int = self._config.dim
        self._is_built = False
        self._warmup_done = False
        self._memtable: Optional[MemTable] = None
        self._wal_writer: Optional[WalWriter] = None
        self._next_doc_id: int = 0
        self._metrics_hook: Optional[Callable[[str, float], None]] = None
        self._page_cache_last: Optional[Tuple[float, float]] = None  # (timestamp, hit_rate)
        self._doc_means: Optional[torch.Tensor] = None
        self._doc_mean_id_to_idx: Optional[Dict[int, int]] = None
        self._rust_index: Optional[NativeExactBackendProtocol] = None
        self._rust_tmpdir: Optional[str] = None
        self._rust_merged_ready: bool = False
        self._native_backend_available: bool = False
        self._native_backend_reason: str = "not_probed"
        self._colbandit_reranker: Optional[RerankerProtocol] = None
        self._roq_quantizer = None
        self._rroq158_meta = None  # lazy: dict{centroids, fwht_seed, dim, group_size} or False
        self._rroq4_riem_meta = None  # lazy: dict{centroids, fwht_seed, dim, group_size} or False
        # Per-shard numpy view cache for ROQ4 / RROQ158 / RROQ4_RIEM
        # scoring. Caches the `.cpu().numpy()` materialization of the
        # per-shard tensors so the inner per-query loop does not
        # re-materialize them every call. Keyed by shard_id; values are
        # dicts of numpy arrays. See `_score_roq4_candidates` /
        # `_score_rroq158_candidates` / `_score_rroq4_riem_candidates`.
        self._roq4_shard_view_cache: Dict[int, Dict[str, Any]] = {}
        self._rroq158_shard_view_cache: Dict[int, Dict[str, Any]] = {}
        self._rroq4_riem_shard_view_cache: Dict[int, Dict[str, Any]] = {}
        self._last_exact_path: Optional[str] = None
        self._last_prune_path: Optional[str] = None
        self._checkpoint_mgr = ShardCheckpointManager(self._path)
        self._file_lock: Optional[Any] = None
        if _FileLock is not None:
            try:
                self._file_lock = _FileLock(self._path / ".lock")
                self._file_lock.acquire()
            except Exception as exc:
                logger.warning("FileLock not acquired (non-fatal): %s", exc)
                self._file_lock = None

        if (self._path / "manifest.json").exists():
            self.load()

    def build(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        """Build the shard index from a corpus of multi-vector documents.

        Args:
            vectors: Per-document token embeddings (list of [n_tokens, dim] arrays).
            ids: Document IDs (same length as vectors).
            payloads: Optional per-document metadata (stored but not indexed).
        """
        with self._lock:
            self._path.mkdir(parents=True, exist_ok=True)
            n_docs = len(vectors)
            if n_docs == 0:
                raise ValueError("Cannot build index from empty corpus")
            if len(ids) != n_docs:
                raise ValueError(f"vectors ({n_docs}) and ids ({len(ids)}) length mismatch")

            self._dim = vectors[0].shape[1]
            cfg = self._config

            all_vecs = np.concatenate(vectors, axis=0).astype(np.float16)
            doc_offsets: List[Tuple[int, int]] = []
            offset = 0
            for v in vectors:
                doc_offsets.append((offset, offset + v.shape[0]))
                offset += v.shape[0]
            doc_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int32)

            logger.info("Building shard index: %d docs, dim=%d, %d total tokens", n_docs, self._dim, all_vecs.shape[0])

            from ..builder import assign_storage_shards

            lemur_dir = self._path / "lemur"
            router = LemurRouter(
                index_dir=lemur_dir,
                ann_backend=cfg.ann_backend.value,
                device=self._router_device,
            )
            doc_vecs_f16 = torch.from_numpy(all_vecs).to(torch.float16)
            doc_counts_t = torch.from_numpy(doc_counts)

            shard_assignments = assign_storage_shards(
                doc_offsets,
                cfg.n_shards,
                cfg.seed,
                StorageLayout.TOKEN_BALANCED,
            )
            doc_id_to_shard = {did: int(shard_assignments[i]) for i, did in enumerate(ids)}
            router.fit_initial(
                pooled_doc_vectors=doc_vecs_f16,
                pooled_doc_counts=doc_counts_t,
                doc_ids=list(ids),
                doc_id_to_shard=doc_id_to_shard,
                epochs=cfg.lemur_epochs,
            )

            proxy_weights = router._weights.clone()
            shard_assignments = assign_storage_shards(
                doc_offsets,
                cfg.n_shards,
                cfg.seed,
                cfg.layout,
                proxy_weights=proxy_weights,
            )
            doc_id_to_shard = {did: int(shard_assignments[i]) for i, did in enumerate(ids)}
            router.fit_initial(
                pooled_doc_vectors=doc_vecs_f16,
                pooled_doc_counts=doc_counts_t,
                doc_ids=list(ids),
                doc_id_to_shard=doc_id_to_shard,
                epochs=cfg.lemur_epochs,
            )
            del doc_vecs_f16, doc_counts_t
            gc.collect()

            roq_doc_codes = None
            roq_doc_meta = None
            rroq158_payload = None
            rroq4_riem_payload = None
            effective_compression = cfg.compression
            if cfg.compression == Compression.ROQ4:
                try:
                    import pickle

                    from voyager_index._internal.inference.quantization.rotational import (
                        RoQConfig,
                        RotationalQuantizer,
                    )

                    logger.info("Training ROQ 4-bit quantizer ...")
                    roq_q = RotationalQuantizer(RoQConfig(dim=self._dim, num_bits=4, seed=cfg.seed))
                    roq_doc_codes = []
                    roq_doc_meta = []
                    for i, (s, e) in enumerate(doc_offsets):
                        vecs_slice = np.asarray(all_vecs[s:e], dtype=np.float32)
                        q = roq_q.quantize(vecs_slice, store=False)
                        roq_doc_codes.append(np.asarray(q["codes"], dtype=np.uint8))
                        roq_doc_meta.append(roq_q.build_triton_meta(q, include_norm_sq=True))
                    with open(self._path / "roq_quantizer.pkl", "wb") as f:
                        pickle.dump(roq_q, f)
                    logger.info("ROQ 4-bit encoding done for %d docs", n_docs)
                except (ImportError, Exception) as exc:
                    logger.warning("ROQ quantizer unavailable (%s), falling back to FP16", exc)
                    effective_compression = Compression.FP16
            elif cfg.compression == Compression.RROQ158:
                from voyager_index._internal.inference.quantization.rroq158 import (
                    Rroq158Config,
                    _resolve_group_size,
                    choose_effective_rroq158_k,
                    encode_rroq158,
                )

                requested_gs = int(cfg.rroq158_group_size)
                n_tok = int(all_vecs.shape[0])
                token_dim = int(all_vecs.shape[1])
                # Resolve the dim-aware group_size so the corpus-size check
                # below uses the value the encoder will actually apply
                # (e.g. requested gs=128 + dim=64 -> effective gs=64).
                if token_dim < 32 or token_dim % 32 != 0:
                    logger.warning(
                        "RROQ158 requested but token dim=%d is not a positive "
                        "multiple of 32. rroq158 cannot encode this dim — "
                        "falling back to FP16. Production dims (64, 96, 128, "
                        "160, 256, 384, 768, 1024) all satisfy this; pass "
                        "compression=FP16 explicitly to silence.",
                        token_dim,
                    )
                    effective_compression = Compression.FP16
                    gs = requested_gs
                else:
                    try:
                        gs = _resolve_group_size(requested_gs, token_dim)
                    except ValueError as exc:
                        logger.warning(
                            "RROQ158 requested but dim=%d cannot fit any of "
                            "{128, 64, 32} group_size (%s). Falling back to "
                            "FP16.",
                            token_dim, exc,
                        )
                        effective_compression = Compression.FP16
                        gs = requested_gs
                if effective_compression == Compression.RROQ158 and n_tok < gs:
                    # Data-shape fallback (NOT a codec error): a corpus with
                    # fewer tokens than a single ternary group cannot be
                    # quantized at all (one popcount word == one group). Drop
                    # to FP16 and log loudly so the operator sees the auto-
                    # downgrade. The "no silent fallback" rule covers codec
                    # failures (which would silently 12× index size); this
                    # path applies to corpora that physically cannot host
                    # any ternary codebook.
                    logger.warning(
                        "RROQ158 requested but corpus has only %d tokens "
                        "(< effective group_size=%d). Falling back to FP16 — "
                        "rroq158 needs at least one ternary group of tokens "
                        "to encode. Pass compression=FP16 explicitly to "
                        "silence.",
                        n_tok, gs,
                    )
                    effective_compression = Compression.FP16
                if effective_compression == Compression.RROQ158:
                    # Auto-shrink K when the corpus has fewer tokens than
                    # the requested codebook size (typical only in tests /
                    # demos / very small shards). This keeps the user's
                    # explicit choice of the rroq158 codec — only the
                    # centroid count adapts — so we don't silently flip the
                    # codec back to fp16.
                    effective_k = choose_effective_rroq158_k(
                        n_tokens=n_tok,
                        requested_k=int(cfg.rroq158_k),
                        group_size=gs,
                    )
                    logger.info(
                        "Training RROQ158 (Riemannian 1.58-bit) quantizer "
                        "(K=%d, group_size=%d, seed=%d) ...",
                        effective_k, gs, int(cfg.rroq158_seed),
                    )
                    rroq_cfg = Rroq158Config(
                        K=effective_k,
                        group_size=gs,
                        seed=int(cfg.rroq158_seed),
                        fit_sample_cap=max(100_000, effective_k),
                    )
                    rroq158_payload = encode_rroq158(
                        np.asarray(all_vecs, dtype=np.float32), rroq_cfg
                    )
                    np.savez(
                        self._path / "rroq158_meta.npz",
                        centroids=rroq158_payload.centroids,
                        fwht_seed=np.array(
                            rroq158_payload.fwht_seed, dtype=np.int64
                        ),
                        dim=np.array(rroq158_payload.dim, dtype=np.int32),
                        group_size=np.array(
                            rroq158_payload.group_size, dtype=np.int32
                        ),
                        k_requested=np.array(
                            int(cfg.rroq158_k), dtype=np.int32
                        ),
                        k_effective=np.array(effective_k, dtype=np.int32),
                    )
                    logger.info(
                        "RROQ158 encoding done for %d docs (%d tokens, K=%d)",
                        n_docs, n_tok, effective_k,
                    )
            elif cfg.compression == Compression.RROQ4_RIEM:
                from voyager_index._internal.inference.quantization.rroq4_riem import (
                    Rroq4RiemConfig,
                    choose_effective_rroq4_riem_k,
                    encode_rroq4_riem,
                )

                gs = int(cfg.rroq4_riem_group_size)
                n_tok = int(all_vecs.shape[0])
                token_dim = int(all_vecs.shape[1])
                if token_dim < gs:
                    # Embedding dim smaller than one 4-bit group → no
                    # quantization possible. Same policy as RROQ158:
                    # drop to FP16 with a loud warning. Production
                    # embeddings (dim >= 128) never trip this branch.
                    logger.warning(
                        "RROQ4_RIEM requested but token dim=%d is smaller "
                        "than group_size=%d. Falling back to FP16 — "
                        "rroq4_riem needs at least group_size coordinates "
                        "per token. Pass compression=FP16 explicitly to "
                        "silence, or use a group_size that divides %d.",
                        token_dim, gs, token_dim,
                    )
                    effective_compression = Compression.FP16
                elif n_tok < gs:
                    # Same data-shape fallback story as RROQ158: a corpus
                    # with fewer tokens than a single 4-bit group cannot be
                    # quantized at all. Drop to FP16 with a loud warning so
                    # the operator notices the auto-downgrade. The "no
                    # silent FP16 fallback" rule covers codec failures
                    # (which would silently 8x index size); this path
                    # applies to corpora that physically cannot host any
                    # asymmetric codebook.
                    logger.warning(
                        "RROQ4_RIEM requested but corpus has only %d tokens "
                        "(< group_size=%d). Falling back to FP16 — rroq4_riem "
                        "needs at least one 4-bit group of tokens to encode. "
                        "Pass compression=FP16 explicitly to silence.",
                        n_tok, gs,
                    )
                    effective_compression = Compression.FP16
                else:
                    # Auto-shrink K when the corpus has fewer tokens than the
                    # requested codebook size (typical only in tests / demos /
                    # very small shards). Same policy as RROQ158: keep the
                    # user's explicit choice of the rroq4_riem codec — only
                    # the centroid count adapts.
                    effective_k = choose_effective_rroq4_riem_k(
                        n_tokens=n_tok,
                        requested_k=int(cfg.rroq4_riem_k),
                        group_size=gs,
                    )
                    logger.info(
                        "Training RROQ4_RIEM (Riemannian 4-bit asymmetric) "
                        "quantizer (K=%d, group_size=%d, seed=%d) ...",
                        effective_k, gs, int(cfg.rroq4_riem_seed),
                    )
                    r4r_cfg = Rroq4RiemConfig(
                        K=effective_k,
                        group_size=gs,
                        seed=int(cfg.rroq4_riem_seed),
                        fit_sample_cap=max(100_000, effective_k),
                    )
                    rroq4_riem_payload = encode_rroq4_riem(
                        np.asarray(all_vecs, dtype=np.float32), r4r_cfg
                    )
                    np.savez(
                        self._path / "rroq4_riem_meta.npz",
                        centroids=rroq4_riem_payload.centroids,
                        fwht_seed=np.array(
                            rroq4_riem_payload.fwht_seed, dtype=np.int64
                        ),
                        dim=np.array(rroq4_riem_payload.dim, dtype=np.int32),
                        group_size=np.array(
                            rroq4_riem_payload.group_size, dtype=np.int32
                        ),
                        k_requested=np.array(
                            int(cfg.rroq4_riem_k), dtype=np.int32
                        ),
                        k_effective=np.array(effective_k, dtype=np.int32),
                    )
                    logger.info(
                        "RROQ4_RIEM encoding done for %d docs (%d tokens, K=%d)",
                        n_docs, n_tok, effective_k,
                    )

            store = ShardStore(self._path)
            store.build(
                all_vectors=all_vecs,
                doc_offsets=doc_offsets,
                doc_ids=list(ids),
                shard_assignments=shard_assignments,
                n_shards=cfg.n_shards,
                dim=self._dim,
                compression=effective_compression,
                uniform_shard_tokens=cfg.uniform_shard_tokens,
                roq_doc_codes=roq_doc_codes,
                roq_doc_meta=roq_doc_meta,
                rroq158_payload=rroq158_payload,
                rroq4_riem_payload=rroq4_riem_payload,
            )

            meta = {
                "dim": self._dim,
                "n_docs": n_docs,
                "n_shards": cfg.n_shards,
                "compression": effective_compression.value,
                "layout": cfg.layout.value,
                "router_type": cfg.router_type.value,
            }
            if effective_compression == Compression.RROQ158:
                meta["rroq158_k"] = int(cfg.rroq158_k)
                meta["rroq158_seed"] = int(cfg.rroq158_seed)
                meta["rroq158_group_size"] = int(cfg.rroq158_group_size)
            elif effective_compression == Compression.RROQ4_RIEM:
                meta["rroq4_riem_k"] = int(cfg.rroq4_riem_k)
                meta["rroq4_riem_seed"] = int(cfg.rroq4_riem_seed)
                meta["rroq4_riem_group_size"] = int(cfg.rroq4_riem_group_size)
            atomic_json_write(self._path / "engine_meta.json", meta)

            payload_dict = {}
            if payloads:
                payload_dict = {str(did): p for did, p in zip(ids, payloads)}
            atomic_json_write(self._path / "payloads.json", payload_dict)

            self._store = store
            self._router = router
            self._doc_vecs = vectors
            self._doc_ids = list(ids)
            self._sealed_payloads: Dict[int, dict] = {did: payload_dict.get(str(did), {}) for did in ids}
            self._is_built = True
            self._next_doc_id = max(ids) + 1 if ids else 0
            self._init_pipeline()

            self._try_gpu_preload()
            self._init_wal_and_memtable()
            self._build_and_save_doc_means(all_vecs, doc_offsets, list(ids))
            store.persist_merged_layout(all_vecs, doc_offsets, list(ids), self._dim)
            self._init_rust_index()

            logger.info("Shard index built: %d docs at %s", n_docs, self._path)

    def load(self) -> None:
        """Load a previously built index from disk."""
        with self._lock:
            meta_path = self._path / "engine_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self._dim = meta.get("dim", self._config.dim)

            self._store = ShardStore(self._path)
            if not self._store.manifest:
                raise FileNotFoundError(f"No manifest at {self._path}")

            lemur_dir = self._path / "lemur"
            self._router = LemurRouter(
                index_dir=lemur_dir,
                ann_backend=self._config.ann_backend.value,
                device=self._router_device,
            )
            self._router.load()

            self._doc_ids = self._store.all_doc_ids()
            self._is_built = True
            self._next_doc_id = max(self._doc_ids) + 1 if self._doc_ids else 0

            self._sealed_payloads = self._load_sealed_payloads()

            self._init_wal_and_memtable()
            self._replay_wal()
            self._load_doc_means()
            self._init_rust_index()

            if self._rust_index is not None:
                # Only preload when both GPU VRAM and CPU staging are safe.
                doc_index_path = self._path / "doc_index.npz"
                merged_emb_path = self._path / "merged_embeddings.bin"
                gpu_preloaded = False
                if doc_index_path.exists() and torch.cuda.is_available() and str(self._device).startswith("cuda"):
                    try:
                        with np.load(str(doc_index_path), allow_pickle=True) as di:
                            actual_max_tok = int(np.max(di["local_ends"] - di["local_starts"]))
                        from ..scorer import PreloadedGpuCorpus

                        fits_gpu = PreloadedGpuCorpus.fits_on_gpu(
                            len(self._doc_ids),
                            actual_max_tok,
                            self._dim,
                        )
                        stream_chunk_docs = PreloadedGpuCorpus.suggest_streaming_chunk_docs(
                            actual_max_tok,
                            self._dim,
                        )
                        fits_cpu_stream = PreloadedGpuCorpus.fits_cpu_streaming(
                            actual_max_tok,
                            self._dim,
                            chunk_docs=stream_chunk_docs,
                        )
                        fits_cpu_stage = PreloadedGpuCorpus.fits_cpu_staging(
                            len(self._doc_ids),
                            actual_max_tok,
                            self._dim,
                        )

                        if fits_gpu and merged_emb_path.exists() and fits_cpu_stream and self._store is not None:
                            try:
                                self._gpu_corpus = PreloadedGpuCorpus.from_merged_streaming(
                                    self._store,
                                    self._dim,
                                    device=self._device,
                                    chunk_docs=stream_chunk_docs,
                                )
                                gpu_preloaded = True
                                logger.info(
                                    "Streaming GPU preload succeeded (%d docs, max_tok=%d, chunk_docs=%d)",
                                    len(self._gpu_corpus.doc_ids),
                                    actual_max_tok,
                                    stream_chunk_docs,
                                )
                            except Exception as exc:
                                logger.warning("Streaming GPU preload failed: %s", exc)
                                self._gpu_corpus = None

                        if not gpu_preloaded:
                            if fits_gpu and fits_cpu_stage:
                                self._doc_vecs = self._load_sealed_vectors()
                                self._try_gpu_preload()
                                if self._gpu_corpus is not None:
                                    gpu_preloaded = True
                                    logger.info(
                                        "GPU preload succeeded (%d docs, max_tok=%d)",
                                        len(self._doc_ids),
                                        actual_max_tok,
                                    )
                                else:
                                    self._doc_vecs = None
                            elif fits_gpu:
                                if merged_emb_path.exists() and not fits_cpu_stream:
                                    stream_mb = (
                                        PreloadedGpuCorpus.estimate_streaming_cpu_bytes(
                                            stream_chunk_docs,
                                            actual_max_tok,
                                            self._dim,
                                        )
                                        / 1e6
                                    )
                                    logger.info(
                                        "Skipping streaming GPU preload (%d docs, max_tok=%d) — "
                                        "chunk staging would require ~%.1f MB; using non-preloaded exact path",
                                        len(self._doc_ids),
                                        actual_max_tok,
                                        stream_mb,
                                    )
                                else:
                                    stage_gb = (
                                        PreloadedGpuCorpus.estimate_cpu_staging_bytes(
                                            len(self._doc_ids),
                                            actual_max_tok,
                                            self._dim,
                                        )
                                        / 1e9
                                    )
                                    logger.info(
                                        "Skipping GPU preload (%d docs, max_tok=%d) — "
                                        "CPU staging would require ~%.1f GB; using non-preloaded exact path",
                                        len(self._doc_ids),
                                        actual_max_tok,
                                        stage_gb,
                                    )
                            else:
                                logger.info(
                                    "Corpus too large for GPU preload (%d docs, max_tok=%d) — "
                                    "using non-preloaded exact path",
                                    len(self._doc_ids),
                                    actual_max_tok,
                                )
                    except Exception as exc:
                        logger.warning("GPU preload check failed: %s", exc)

                if not gpu_preloaded:
                    self._doc_vecs = None
            else:
                self._doc_vecs = self._load_sealed_vectors()
                self._try_gpu_preload()
                if self._gpu_corpus is None:
                    self._doc_vecs = None
            if self._rust_index is None or self._gpu_corpus is not None or str(self._device).startswith("cuda"):
                self._init_pipeline()
            logger.info("Shard index loaded: %d docs from %s", len(self._doc_ids), self._path)

    def flush(self) -> None:
        """Checkpoint: persist WAL but do NOT drain memtable.

        Real L0-to-sealed merge is not yet implemented; draining memtable
        without persisting to ShardStore would cause data loss. The WAL
        ensures crash recovery of memtable state.
        """
        with self._lock:
            if self._wal_writer:
                self._wal_writer.sync()
            if self._memtable:
                mt_docs, mt_payloads, mt_tombstones = self._memtable.snapshot()
                wal_offset = self._wal_writer.n_entries if self._wal_writer else 0
                sealed_snap = (
                    dict(self._sealed_payloads) if hasattr(self, "_sealed_payloads") and self._sealed_payloads else {}
                )
                try:
                    self._checkpoint_mgr.save(
                        memtable_docs=mt_docs,
                        payloads=mt_payloads,
                        tombstones=mt_tombstones,
                        next_doc_id=self._next_doc_id,
                        wal_offset=wal_offset,
                        sealed_payloads=sealed_snap,
                    )
                except Exception as exc:
                    logger.warning("Checkpoint save failed (WAL still safe): %s", exc)
            logger.debug("flush(): WAL synced + checkpoint saved")

    def refresh_gpu_corpus(self) -> None:
        """Rebuild GPU corpus tensors from current sealed snapshot (call after compaction)."""
        if self._gpu_corpus and self._doc_vecs and self._doc_ids:
            self._gpu_corpus.refresh(self._doc_vecs, self._doc_ids)

    def save(self) -> None:
        """Persist current state (router already auto-saves during fit)."""
        with self._lock:
            if self._router:
                self._router.save()

    def close(self) -> None:
        """Release resources."""
        with self._lock:
            if self._wal_writer:
                self._wal_writer.close()
                self._wal_writer = None
            self._memtable = None
            self._gpu_corpus = None
            self._pipeline = None
            self._store = None
            self._router = None
            self._doc_vecs = None
            self._doc_means = None
            self._doc_mean_id_to_idx = None
            if self._rust_index is not None:
                try:
                    self._rust_index.close()
                except Exception:
                    pass
                self._rust_index = None
            if self._rust_tmpdir:
                import shutil

                shutil.rmtree(self._rust_tmpdir, ignore_errors=True)
                self._rust_tmpdir = None
            self._sealed_payloads = {}
            self._is_built = False
            if self._file_lock is not None:
                try:
                    self._file_lock.release()
                except Exception:
                    pass
                self._file_lock = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _init_pipeline(self) -> None:
        if self._store is None:
            return
        cfg = self._config
        pool = None
        if cfg.transfer_mode in (TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED):
            pool = PinnedBufferPool(
                max_tokens=cfg.pinned_buffer_max_tokens,
                dim=self._dim,
                n_buffers=cfg.pinned_pool_buffers,
            )
        self._pipeline = FetchPipeline(
            store=self._store,
            mode=cfg.transfer_mode,
            pinned_pool=pool,
            device=self._device,
        )
        if self._router is not None:
            router_caps = self._router.capability_snapshot()
            logger.info(
                "Shard engine capabilities: router_backend=%s ann_backend=%s pinned_staging=%s",
                router_caps.get("backend_name"),
                router_caps.get("ann_backend"),
                self._pipeline.capability_snapshot().get("pinned_staging_mode"),
            )

    def _try_gpu_preload(self) -> None:
        """Attempt to pre-load corpus to GPU (GEM pattern) if it fits."""
        if self._doc_vecs is None or not self._doc_ids or not str(self._device).startswith("cuda"):
            return
        max_tok = max((v.shape[0] for v in self._doc_vecs), default=1)
        if PreloadedGpuCorpus.fits_on_gpu(len(self._doc_vecs), max_tok, self._dim):
            self._gpu_corpus = PreloadedGpuCorpus(
                self._doc_vecs,
                self._doc_ids,
                self._dim,
                device=self._device,
            )
        else:
            logger.info("Corpus too large for GPU preload (%d docs) — using shard fetch", len(self._doc_ids))

    def _ensure_warmup(self) -> None:
        if self._warmup_done:
            return
        token_counts = [128, 256]
        if self._store and self._store.manifest:
            token_counts = [int(s.p95_tokens) for s in self._store.manifest.shards if s.p95_tokens > 0] or token_counts
        warmup_maxsim(dim=self._dim, doc_token_counts=token_counts, device=self._device)
        self._warmup_done = True

    def _init_wal_and_memtable(self) -> None:
        self._memtable = MemTable(dim=self._dim, device=self._device)
        wal_path = self._path / "wal.bin"
        self._wal_writer = WalWriter(wal_path)
        self._wal_writer.open()

    def _replay_wal(self) -> None:
        """Replay WAL entries from disk into the memtable for crash recovery.

        If a checkpoint exists, its memtable state is restored first and only
        WAL entries written *after* the checkpoint are replayed.
        """
        ckpt = self._checkpoint_mgr.load()
        if ckpt is not None:
            logger.info(
                "Restoring checkpoint: %d docs, wal_offset=%d",
                len(ckpt["docs"]),
                ckpt["wal_offset"],
            )
            for doc_id, vecs in ckpt["docs"].items():
                self._memtable.insert(doc_id, vecs, ckpt["payloads"].get(doc_id))
            for doc_id in ckpt["tombstones"]:
                self._memtable.delete(doc_id)
            self._next_doc_id = max(self._next_doc_id, ckpt["next_doc_id"])
            sealed_ckpt = ckpt.get("sealed_payloads")
            if sealed_ckpt and hasattr(self, "_sealed_payloads"):
                self._sealed_payloads.update(sealed_ckpt)

        wal_path = self._path / "wal.bin"
        if not wal_path.exists():
            return
        entries = WalReader(wal_path).replay()
        if not entries:
            return
        wal_skip = ckpt["wal_offset"] if ckpt else 0
        entries = entries[wal_skip:]
        if not entries:
            return
        logger.info("Replaying %d WAL entries for crash recovery", len(entries))
        for entry in entries:
            if entry.op == WalOp.INSERT:
                if entry.vectors is not None:
                    self._memtable.insert(entry.doc_id, entry.vectors, entry.payload)
                    self._next_doc_id = max(self._next_doc_id, entry.doc_id + 1)
            elif entry.op == WalOp.DELETE:
                self._memtable.delete(entry.doc_id)
            elif entry.op == WalOp.UPSERT:
                if entry.vectors is not None:
                    self._memtable.upsert(entry.doc_id, entry.vectors, entry.payload)
                    self._next_doc_id = max(self._next_doc_id, entry.doc_id + 1)
            elif entry.op == WalOp.UPDATE_PAYLOAD:
                if entry.payload is not None:
                    self._memtable.upsert_payload(entry.doc_id, entry.payload)
                    if (
                        hasattr(self, "_sealed_payloads")
                        and self._sealed_payloads
                        and entry.doc_id in self._sealed_payloads
                    ):
                        self._sealed_payloads[entry.doc_id].update(entry.payload)
        logger.info(
            "WAL replay done: memtable has %d docs, %d tombstones", self._memtable.size, self._memtable.tombstone_count
        )

    def _build_and_save_doc_means(
        self,
        all_vecs: np.ndarray,
        doc_offsets: List[Tuple[int, int]],
        ids: List[int],
    ) -> None:
        """Compute mean-pooled doc embeddings and persist for proxy scoring."""
        dim = all_vecs.shape[1]
        means = np.zeros((len(ids), dim), dtype=np.float16)
        for i, (s, e) in enumerate(doc_offsets):
            if e > s:
                means[i] = all_vecs[s:e].astype(np.float32).mean(axis=0).astype(np.float16)
        np.savez_compressed(
            self._path / "doc_means.npz",
            means=means,
            ids=np.array(ids, dtype=np.int64),
        )
        dev = torch.device(self._device if torch.cuda.is_available() else "cpu")
        self._doc_means = torch.from_numpy(means).to(dev)
        self._doc_mean_id_to_idx = {int(did): i for i, did in enumerate(ids)}
        logger.info("Doc-mean proxy embeddings saved: %d docs, %.1f MB", len(ids), means.nbytes / 1e6)

    def _load_doc_means(self) -> None:
        """Load mean-pooled doc embeddings for proxy scoring (if available)."""
        path = self._path / "doc_means.npz"
        if not path.exists():
            return
        try:
            with np.load(path) as data:
                means = data["means"]
                ids = data["ids"]
            dev = torch.device(self._device if torch.cuda.is_available() else "cpu")
            self._doc_means = torch.from_numpy(means.copy()).to(dev)
            self._doc_mean_id_to_idx = {int(did): i for i, did in enumerate(ids)}
            logger.info("Doc-mean proxy embeddings loaded: %d docs on %s", len(ids), dev)
        except Exception as exc:
            logger.warning("Could not load doc means for proxy scoring: %s", exc)

    def _init_rust_index(self) -> None:
        """Initialize the Rust ShardIndex.

        The index is created whenever ``doc_index.npz`` and shard files
        exist.  Centroids and codec are loaded optionally for approximate
        scoring.  Merged mmap files, when present, enable the fused exact
        scoring CPU fast-path (``score_candidates_exact``).
        """
        doc_index_path = self._path / "doc_index.npz"
        if not doc_index_path.exists():
            self._native_backend_reason = "doc_index_missing"
            logger.info("doc_index.npz not found — skipping Rust index init")
            return
        try:
            import latence_shard_engine
        except ImportError:
            self._native_backend_reason = "latence_shard_engine_missing"
            logger.warning("latence_shard_engine not installed — skipping Rust index init")
            return
        try:
            with np.load(str(doc_index_path), allow_pickle=True) as doc_index:
                doc_ids = doc_index["doc_ids"]
                shard_ids = doc_index["shard_ids"]
                local_starts = doc_index["local_starts"]
                local_ends = doc_index["local_ends"]

            shard_dir = self._path / "shards"
            shard_files = sorted(shard_dir.glob("shard_*.safetensors"))
            if not shard_files:
                self._native_backend_reason = "shard_files_missing"
                logger.warning("No shard files found — skipping Rust index init")
                return

            import os
            import shutil
            import tempfile

            if self._rust_tmpdir and os.path.isdir(self._rust_tmpdir):
                shutil.rmtree(self._rust_tmpdir, ignore_errors=True)
            tmpdir = tempfile.mkdtemp(prefix="shard_idx_")
            self._rust_tmpdir = tmpdir
            for i, sf in enumerate(shard_files):
                os.symlink(str(sf), os.path.join(tmpdir, f"shard_{i}.safetensors"))
            open(os.path.join(tmpdir, "shard.wal"), "wb").close()

            rust_idx = latence_shard_engine.ShardIndex(tmpdir, self._dim)
            rust_idx.register_shard_docs(
                doc_ids.tolist(),
                shard_ids.astype(np.uint32).tolist(),
                local_starts.tolist(),
                local_ends.tolist(),
            )

            centroids_path = self._path / "centroids.npy"
            if centroids_path.exists():
                centroids = np.load(str(centroids_path)).astype(np.float32)
                rust_idx.set_centroids(centroids)
                logger.info("Centroids loaded: %d centroids for approx scoring", len(centroids))

                codec_meta_path = self._path / "codec_meta.npz"
                if codec_meta_path.exists():
                    with np.load(str(codec_meta_path)) as meta:
                        bw = meta["bucket_weights"].astype(np.float32)
                        nbits_val = int(meta["nbits"][0])
                    rust_idx.set_codec(bw.tolist(), nbits_val)
                    logger.info(
                        "Residual codec loaded: nbits=%d, %d bucket weights",
                        nbits_val,
                        len(bw),
                    )

            merged_emb_path = self._path / "merged_embeddings.bin"
            if merged_emb_path.exists():
                try:
                    rust_idx.load_merged(str(self._path))
                    self._rust_merged_ready = True
                    logger.info("Merged mmap files loaded — native fused exact scoring enabled")
                except Exception as e:
                    self._native_backend_reason = f"merged_layout_error:{type(e).__name__}"
                    logger.warning("Failed to load merged mmap files: %s", e)

            self._rust_index = rust_idx
            self._native_backend_available = True
            self._native_backend_reason = "ready"
            logger.info(
                "Rust ShardIndex ready: %d docs, merged_exact=%s",
                rust_idx.doc_count,
                getattr(self, "_rust_merged_ready", False),
            )
        except Exception as exc:
            logger.warning("Failed to initialize Rust ShardIndex: %s", exc)
            self._rust_index = None
            self._native_backend_available = False
            self._native_backend_reason = f"init_failed:{type(exc).__name__}"

    def _load_sealed_payloads(self) -> Dict[int, dict]:
        """Load sealed payloads from payloads.json."""
        payload_path = self._path / "payloads.json"
        if not payload_path.exists():
            return {}
        try:
            with open(payload_path) as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Failed to load sealed payloads: %s", exc)
            return {}

    def _load_sealed_vectors(self) -> Optional[List[np.ndarray]]:
        """Load sealed doc vectors from the shard store for GPU preload (single batch)."""
        if not self._store or not self._doc_ids:
            return None
        try:
            fetched = self._store.fetch_docs(self._doc_ids)
            all_vecs = []
            missing = 0
            for did in self._doc_ids:
                if did in fetched:
                    all_vecs.append(fetched[did])
                else:
                    missing += 1
                    all_vecs.append(np.zeros((1, self._dim), dtype=np.float16))
            if missing > 0:
                logger.warning("%d docs missing from shard store during GPU preload", missing)
            return all_vecs
        except Exception as exc:
            logger.warning("Could not load sealed vectors for GPU preload: %s", exc)
            return None

