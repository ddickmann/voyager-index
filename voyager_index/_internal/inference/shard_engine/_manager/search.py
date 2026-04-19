"""Search orchestration and exact-stage execution helpers."""
from __future__ import annotations

from .common import *  # noqa: F401,F403

class ShardSegmentManagerSearchMixin:
    @staticmethod
    def _apply_search_overrides(scfg: SearchConfig, **kwargs) -> SearchConfig:
        override_keys = (
            "max_docs_exact",
            "lemur_search_k_cap",
            "n_full_scores",
            "n_centroid_approx",
            "transfer_mode",
            "pinned_pool_buffers",
            "pinned_buffer_max_tokens",
            "use_colbandit",
            "quantization_mode",
            "variable_length_strategy",
            "gpu_corpus_rerank_topn",
        )
        for key in override_keys:
            if key not in kwargs or kwargs[key] is None:
                continue
            value = kwargs[key]
            if key == "transfer_mode" and isinstance(value, str):
                value = TransferMode(value)
            setattr(scfg, key, value)
        return scfg

    def _get_colbandit_reranker(self) -> ColBanditReranker:
        if self._colbandit_reranker is None:
            self._colbandit_reranker = ColBanditReranker()
        return self._colbandit_reranker

    def _load_roq_quantizer(self):
        if self._roq_quantizer is False:
            return None
        if self._roq_quantizer is not None:
            return self._roq_quantizer
        quantizer_path = self._path / "roq_quantizer.pkl"
        if not quantizer_path.exists():
            self._roq_quantizer = False
            return None
        try:
            with open(quantizer_path, "rb") as handle:
                self._roq_quantizer = pickle.load(handle)
        except Exception as exc:
            logger.warning("Failed to load ROQ quantizer from %s: %s", quantizer_path, exc)
            self._roq_quantizer = False
            return None
        return self._roq_quantizer

    def _load_rroq158_meta(self):
        """Lazy-load `rroq158_meta.npz` (centroids + fwht_seed) from the index dir.

        The artifact is written by `lifecycle.build()` when
        `Compression.RROQ158` is selected. If absent we cache `False` so we
        don't keep stat-ing the path on every query.
        """
        if self._rroq158_meta is False:
            return None
        if self._rroq158_meta is not None:
            return self._rroq158_meta
        meta_path = self._path / "rroq158_meta.npz"
        if not meta_path.exists():
            self._rroq158_meta = False
            return None
        try:
            arr = np.load(str(meta_path))
            self._rroq158_meta = {
                "centroids": np.asarray(arr["centroids"], dtype=np.float32),
                "fwht_seed": int(np.asarray(arr["fwht_seed"]).item()),
                "dim": int(np.asarray(arr["dim"]).item()),
                "group_size": int(np.asarray(arr["group_size"]).item()),
            }
            logger.info(
                "RROQ158 meta loaded: K=%d, dim=%d, group_size=%d",
                self._rroq158_meta["centroids"].shape[0],
                self._rroq158_meta["dim"],
                self._rroq158_meta["group_size"],
            )
        except Exception as exc:
            logger.warning("Failed to load RROQ158 meta from %s: %s", meta_path, exc)
            self._rroq158_meta = False
            return None
        return self._rroq158_meta

    def _score_pipeline_fetch(
        self,
        q: torch.Tensor,
        shard_groups: Dict[int, List[int]],
        internal_k: int,
        scfg: SearchConfig,
        dev: torch.device,
        *,
        exact_path: str,
        use_colbandit: bool,
    ) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        if self._pipeline is None:
            return [], self._empty_exact_stage_stats(exact_path)
        shard_chunks, fetch_stats = self._pipeline.fetch_candidate_docs(
            shard_groups,
            max_docs=scfg.max_docs_exact,
        )
        stage_stats = self._empty_exact_stage_stats(exact_path)
        stage_stats["fetch_ms"] = fetch_stats.get("fetch_ms", 0.0)
        stage_stats["h2d_bytes"] = fetch_stats.get("h2d_bytes", 0)
        stage_stats["num_shards_fetched"] = fetch_stats.get("num_shards", 0)
        stage_stats["num_docs_scored"] = fetch_stats.get("num_docs", 0)
        if not shard_chunks:
            return [], stage_stats
        if use_colbandit:
            ids, scores, score_stats = self._get_colbandit_reranker().rerank_shard_chunks(
                q,
                shard_chunks,
                k=internal_k,
                device=dev,
                quantization_mode=scfg.quantization_mode,
                variable_length_strategy=scfg.variable_length_strategy,
            )
        else:
            ids, scores, score_stats = score_all_docs_topk(
                q,
                shard_chunks,
                k=internal_k,
                device=dev,
                quantization_mode=scfg.quantization_mode,
                variable_length_strategy=scfg.variable_length_strategy,
                return_stats=True,
            )
        stage_stats.update(score_stats)
        stage_stats["num_docs_scored"] = fetch_stats.get("num_docs", stage_stats.get("num_docs_scored", 0))
        return list(zip(ids, scores)), stage_stats

    def _score_roq4_candidates(
        self,
        q: torch.Tensor,
        shard_groups: Dict[int, List[int]],
        internal_k: int,
        dev: torch.device,
    ) -> Optional[Tuple[List[Tuple[int, float]], Dict[str, Any]]]:
        if dev.type != "cuda" or self._store is None:
            return None
        quantizer = self._load_roq_quantizer()
        if quantizer is None:
            return None
        try:
            with Timer(sync_cuda=False) as t_prepare:
                query_np = q.detach().cpu().numpy().astype(np.float32)
                quantized_query = quantizer.quantize(query_np, store=False)
                query_codes = torch.from_numpy(np.asarray(quantized_query["codes"], dtype=np.uint8)[np.newaxis])
                query_meta = torch.from_numpy(
                    quantizer.build_triton_meta(quantized_query, include_norm_sq=True)[np.newaxis],
                )

            with Timer(sync_cuda=False) as t_fetch:
                doc_ids: List[int] = []
                doc_codes_rows: List[np.ndarray] = []
                doc_meta_rows: List[np.ndarray] = []
                num_shards = 0
                for shard_id, dids in shard_groups.items():
                    loaded = self._store.load_shard_roq4(shard_id)
                    if loaded is None:
                        continue
                    num_shards += 1
                    shard_codes, shard_meta, shard_offsets, shard_ids = loaded
                    row_by_doc = {int(doc_id): idx for idx, doc_id in enumerate(shard_ids.tolist())}
                    for did in dids:
                        row = row_by_doc.get(int(did))
                        if row is None:
                            continue
                        start = int(shard_offsets[row, 0])
                        end = int(shard_offsets[row, 1])
                        doc_ids.append(int(did))
                        doc_codes_rows.append(shard_codes[start:end].cpu().numpy())
                        doc_meta_rows.append(shard_meta[start:end].cpu().numpy())
                if not doc_ids:
                    return None
                max_tok = max(item.shape[0] for item in doc_codes_rows)
                n_bytes = doc_codes_rows[0].shape[1]
                meta_cols = doc_meta_rows[0].shape[1]
                doc_codes = np.zeros((len(doc_ids), max_tok, n_bytes), dtype=np.uint8)
                doc_meta = np.zeros((len(doc_ids), max_tok, meta_cols), dtype=np.float32)
                documents_mask = np.zeros((len(doc_ids), max_tok), dtype=np.float32)
                for idx, (codes, meta) in enumerate(zip(doc_codes_rows, doc_meta_rows)):
                    tok_count = codes.shape[0]
                    doc_codes[idx, :tok_count] = codes
                    doc_meta[idx, :tok_count] = meta
                    documents_mask[idx, :tok_count] = 1.0

            with Timer(sync_cuda=True) as t_exact:
                ids, scores = score_roq4_topk(
                    query_codes=query_codes,
                    query_meta=query_meta,
                    doc_codes=torch.from_numpy(doc_codes),
                    doc_meta=torch.from_numpy(doc_meta),
                    doc_ids=doc_ids,
                    k=internal_k,
                    documents_mask=torch.from_numpy(documents_mask),
                    device=dev,
                )
            if not ids and doc_ids:
                return None
            stage_stats = self._empty_exact_stage_stats("roq4_pipeline")
            stage_stats["prepare_ms"] = t_prepare.elapsed_ms
            stage_stats["fetch_ms"] = t_fetch.elapsed_ms
            stage_stats["exact_ms"] = t_exact.elapsed_ms
            stage_stats["maxsim_ms"] = t_exact.elapsed_ms
            stage_stats["num_shards_fetched"] = num_shards
            stage_stats["num_docs_scored"] = len(doc_ids)
            stage_stats["h2d_bytes"] = int(doc_codes.nbytes + doc_meta.nbytes + documents_mask.nbytes)
            return list(zip(ids, scores)), stage_stats
        except Exception as exc:
            logger.warning("ROQ4 serving path failed; falling back to standard scoring: %s", exc)
            return None

    def _score_rroq158_candidates(
        self,
        q: torch.Tensor,
        shard_groups: Dict[int, List[int]],
        internal_k: int,
        dev: torch.device,
    ) -> Optional[Tuple[List[Tuple[int, float]], Dict[str, Any]]]:
        """Score `shard_groups` using the rroq158 packed shards.

        Works on both CUDA (Triton kernel) and CPU (Rust SIMD kernel via
        `latence_shard_engine`). The shard payloads must have been built with
        `Compression.RROQ158`, otherwise `_load_rroq158_meta` returns None
        and we fall through.
        """
        if self._store is None:
            return None
        meta = self._load_rroq158_meta()
        if meta is None:
            return None
        try:
            from voyager_index._internal.inference.quantization.rroq158 import (
                encode_query_for_rroq158,
                pack_doc_codes_to_int32_words,
            )
            from voyager_index._internal.inference.shard_engine.scorer import (
                score_rroq158_topk,
            )
        except ImportError as exc:
            logger.warning("rroq158 scoring imports unavailable: %s", exc)
            return None

        try:
            with Timer(sync_cuda=False) as t_prepare:
                q_np = q.detach().cpu().numpy().astype(np.float32)
                centroids = meta["centroids"]
                q_inputs = encode_query_for_rroq158(
                    q_np,
                    centroids,
                    fwht_seed=meta["fwht_seed"],
                    query_bits=4,
                )
                qp = torch.from_numpy(q_inputs["q_planes"][None, :, :, :])
                qm = torch.from_numpy(q_inputs["q_meta"][None, :, :])
                qct = torch.from_numpy(q_inputs["qc_table"][None, :, :])

            with Timer(sync_cuda=False) as t_fetch:
                doc_ids: List[int] = []
                sign_rows: List[np.ndarray] = []
                nz_rows: List[np.ndarray] = []
                scl_rows: List[np.ndarray] = []
                cid_rows: List[np.ndarray] = []
                cos_rows: List[np.ndarray] = []
                sin_rows: List[np.ndarray] = []
                num_shards = 0
                for shard_id, dids in shard_groups.items():
                    loaded = self._store.load_shard_rroq158(shard_id)
                    if loaded is None:
                        continue
                    num_shards += 1
                    sign_t = loaded["sign_plane"].cpu().numpy()
                    nz_t = loaded["nonzero_plane"].cpu().numpy()
                    scl_t = loaded["scales"].cpu().numpy().astype(np.float32, copy=False)
                    cid_t = loaded["centroid_id"].cpu().numpy().astype(np.int32, copy=False)
                    cos_t = loaded["cos_norm"].cpu().numpy().astype(np.float32, copy=False)
                    sin_t = loaded["sin_norm"].cpu().numpy().astype(np.float32, copy=False)
                    offsets = loaded["doc_offsets"].cpu().numpy()
                    shard_ids_arr = loaded["doc_ids"].cpu().numpy()
                    row_by_doc = {int(d): idx for idx, d in enumerate(shard_ids_arr.tolist())}
                    # sign/nz are stored as packed bytes per token; convert to
                    # int32 words exactly as the kernel/Triton consume them.
                    sign_words = pack_doc_codes_to_int32_words(sign_t)
                    nz_words = pack_doc_codes_to_int32_words(nz_t)
                    for did in dids:
                        row = row_by_doc.get(int(did))
                        if row is None:
                            continue
                        start = int(offsets[row, 0])
                        end = int(offsets[row, 1])
                        doc_ids.append(int(did))
                        sign_rows.append(sign_words[start:end])
                        nz_rows.append(nz_words[start:end])
                        scl_rows.append(scl_t[start:end])
                        cid_rows.append(cid_t[start:end])
                        cos_rows.append(cos_t[start:end])
                        sin_rows.append(sin_t[start:end])
                if not doc_ids:
                    return None
                T_max = max(r.shape[0] for r in sign_rows)
                B = len(doc_ids)
                n_words = sign_rows[0].shape[1]
                n_groups = scl_rows[0].shape[1]
                doc_sign = np.zeros((B, T_max, n_words), dtype=np.int32)
                doc_nz = np.zeros((B, T_max, n_words), dtype=np.int32)
                doc_scl = np.zeros((B, T_max, n_groups), dtype=np.float32)
                doc_cid = np.zeros((B, T_max), dtype=np.int32)
                doc_cos = np.zeros((B, T_max), dtype=np.float32)
                doc_sin = np.zeros((B, T_max), dtype=np.float32)
                docs_mask = np.zeros((B, T_max), dtype=np.float32)
                for i, (sg, nz, sc, ci, co, si) in enumerate(
                    zip(sign_rows, nz_rows, scl_rows, cid_rows, cos_rows, sin_rows)
                ):
                    t = sg.shape[0]
                    doc_sign[i, :t] = sg
                    doc_nz[i, :t] = nz
                    doc_scl[i, :t] = sc
                    doc_cid[i, :t] = ci
                    doc_cos[i, :t] = co
                    doc_sin[i, :t] = si
                    docs_mask[i, :t] = 1.0

            with Timer(sync_cuda=dev.type == "cuda") as t_exact:
                ids, scores = score_rroq158_topk(
                    query_planes=qp,
                    query_meta=qm,
                    qc_table=qct,
                    doc_centroid_id=torch.from_numpy(doc_cid),
                    doc_cos_norm=torch.from_numpy(doc_cos),
                    doc_sin_norm=torch.from_numpy(doc_sin),
                    doc_sign=torch.from_numpy(doc_sign),
                    doc_nz=torch.from_numpy(doc_nz),
                    doc_scales=torch.from_numpy(doc_scl),
                    doc_ids=doc_ids,
                    k=internal_k,
                    documents_mask=torch.from_numpy(docs_mask),
                    device=dev,
                )
            if not ids and doc_ids:
                return None
            stage_stats = self._empty_exact_stage_stats("rroq158_pipeline")
            stage_stats["prepare_ms"] = t_prepare.elapsed_ms
            stage_stats["fetch_ms"] = t_fetch.elapsed_ms
            stage_stats["exact_ms"] = t_exact.elapsed_ms
            stage_stats["maxsim_ms"] = t_exact.elapsed_ms
            stage_stats["num_shards_fetched"] = num_shards
            stage_stats["num_docs_scored"] = len(doc_ids)
            stage_stats["h2d_bytes"] = int(
                doc_sign.nbytes + doc_nz.nbytes + doc_scl.nbytes
                + doc_cid.nbytes + doc_cos.nbytes + doc_sin.nbytes
            )
            return list(zip(ids, scores)), stage_stats
        except Exception as exc:
            logger.warning(
                "RROQ158 serving path failed; falling back to standard scoring: %s", exc,
            )
            return None

    def _resolve_scoring_device(self) -> torch.device:
        if str(self._device).startswith("cuda") and torch.cuda.is_available():
            return torch.device(self._device)
        return torch.device("cpu")

    @staticmethod
    def _route_prefetch_cap(scfg: SearchConfig) -> int:
        return max(
            1,
            int(scfg.max_docs_exact),
            int(scfg.k_candidates),
            int(scfg.n_full_scores),
            int(scfg.n_centroid_approx),
        )

    def _group_candidate_ids_by_shard(self, candidate_ids: List[int]) -> Dict[int, List[int]]:
        if self._store is None:
            return {}
        by_shard: Dict[int, List[int]] = {}
        for did in candidate_ids:
            try:
                shard_id = int(self._store.doc_shard_id(int(did)))
            except KeyError:
                continue
            by_shard.setdefault(shard_id, []).append(int(did))
        return by_shard

    @staticmethod
    def _empty_exact_stage_stats(exact_path: str = "none") -> Dict[str, Any]:
        return {
            "exact_path": exact_path,
            "fetch_ms": 0.0,
            "exact_ms": 0.0,
            "prepare_ms": 0.0,
            "h2d_ms": 0.0,
            "maxsim_ms": 0.0,
            "topk_ms": 0.0,
            "h2d_bytes": 0,
            "num_shards_fetched": 0,
            "num_docs_scored": 0,
        }

    def _prune_routed_candidates(
        self,
        q: torch.Tensor,
        routed: CandidatePlan,
        scfg: SearchConfig,
        dev: torch.device,
    ) -> Tuple[List[int], Dict[int, List[int]], str]:
        if self._rust_index is not None and scfg.n_centroid_approx > 0 and len(routed.doc_ids) > scfg.n_centroid_approx:
            q_np = q.cpu().numpy().astype(np.float32)
            approx_ids, _approx_scores = self._rust_index.score_candidates_approx(
                q_np,
                routed.doc_ids,
                scfg.n_centroid_approx,
            )
            pruned_ids = approx_ids.tolist()
            pruned_set = set(pruned_ids)
            routed_by_shard = {sid: [d for d in dids if d in pruned_set] for sid, dids in routed.by_shard.items()}
            routed_by_shard = {sid: dids for sid, dids in routed_by_shard.items() if dids}
            return pruned_ids, routed_by_shard, "centroid_approx"

        if self._doc_means is not None and self._doc_mean_id_to_idx is not None:
            pruned_ids = proxy_score_candidates(
                query=q,
                doc_means=self._doc_means,
                candidate_doc_ids=routed.doc_ids,
                doc_id_to_idx=self._doc_mean_id_to_idx,
                n_full_scores=scfg.n_full_scores,
                device=dev,
            )
            pruned_set = set(pruned_ids)
            routed_by_shard = {sid: [d for d in dids if d in pruned_set] for sid, dids in routed.by_shard.items()}
            routed_by_shard = {sid: dids for sid, dids in routed_by_shard.items() if dids}
            return pruned_ids, routed_by_shard, "doc_mean_proxy"

        return routed.doc_ids, routed.by_shard, "none"

    def _score_sealed_candidates(
        self,
        q: torch.Tensor,
        candidate_ids: List[int],
        docs_by_shard: Dict[int, List[int]],
        internal_k: int,
        scfg: SearchConfig,
        dev: torch.device,
    ) -> Tuple[List[Tuple[int, float]], List[int], str, Dict[str, Any]]:
        exact_candidate_ids = candidate_ids[: scfg.max_docs_exact]
        if not exact_candidate_ids:
            return [], [], "none", self._empty_exact_stage_stats("none")

        shard_groups = docs_by_shard or self._group_candidate_ids_by_shard(exact_candidate_ids)
        quant_mode = str(getattr(scfg, "quantization_mode", "") or "").strip().lower()
        want_roq4 = dev.type == "cuda" and quant_mode == "roq4"
        want_rroq158 = quant_mode == "rroq158"
        want_quantized_kernel = dev.type == "cuda" and quant_mode in {"int8", "fp8"}
        want_colbandit = bool(getattr(scfg, "use_colbandit", False))

        if want_roq4:
            roq_result = self._score_roq4_candidates(
                q,
                shard_groups,
                internal_k,
                dev,
            )
            if roq_result is not None:
                results, stage_stats = roq_result
                return results, exact_candidate_ids, "roq4_pipeline", stage_stats

        if want_rroq158:
            # rroq158 works on both CUDA (Triton) and CPU (Rust SIMD).
            rroq_result = self._score_rroq158_candidates(
                q,
                shard_groups,
                internal_k,
                dev,
            )
            if rroq_result is not None:
                results, stage_stats = rroq_result
                return results, exact_candidate_ids, "rroq158_pipeline", stage_stats

        if want_colbandit and self._pipeline is not None:
            results, stage_stats = self._score_pipeline_fetch(
                q,
                shard_groups,
                internal_k,
                scfg,
                dev,
                exact_path="colbandit_pipeline_fetch",
                use_colbandit=True,
            )
            if results or stage_stats.get("num_docs_scored", 0) > 0:
                return results, exact_candidate_ids, "colbandit_pipeline_fetch", stage_stats

        if want_quantized_kernel and self._pipeline is not None:
            results, stage_stats = self._score_pipeline_fetch(
                q,
                shard_groups,
                internal_k,
                scfg,
                dev,
                exact_path="pipeline_quantized",
                use_colbandit=False,
            )
            if results or stage_stats.get("num_docs_scored", 0) > 0:
                return results, exact_candidate_ids, "pipeline_quantized", stage_stats

        if self._gpu_corpus is not None and dev.type == "cuda":
            rerank_topn = max(int(getattr(scfg, "gpu_corpus_rerank_topn", 0) or 0), internal_k)
            ids, scores, score_stats = self._gpu_corpus.score_candidates(
                q,
                exact_candidate_ids,
                k=rerank_topn,
                return_stats=True,
            )
            stage_stats = self._empty_exact_stage_stats("gpu_corpus")
            stage_stats.update(score_stats)
            stage_stats["num_docs_scored"] = len(exact_candidate_ids)

            if self._rust_index is not None and rerank_topn > internal_k and len(ids) > internal_k:
                q_np = q.cpu().numpy().astype(np.float32) if not isinstance(q, np.ndarray) else q.astype(np.float32)
                with Timer(sync_cuda=True) as t_rerank:
                    rerank_ids, rerank_scores = self._rust_index.score_candidates_exact(
                        q_np,
                        ids,
                        internal_k,
                    )
                stage_stats["exact_ms"] += t_rerank.elapsed_ms
                stage_stats["maxsim_ms"] += t_rerank.elapsed_ms
                return (
                    list(zip(rerank_ids.tolist(), rerank_scores.tolist())),
                    exact_candidate_ids,
                    "gpu_corpus_rerank",
                    stage_stats,
                )

            return list(zip(ids, scores)), exact_candidate_ids, "gpu_corpus", stage_stats

        if self._rust_index is not None:
            q_np = q.cpu().numpy().astype(np.float32) if not isinstance(q, np.ndarray) else q.astype(np.float32)
            try:
                with Timer(sync_cuda=dev.type == "cuda") as t_exact:
                    top_ids, top_scores = self._rust_index.score_candidates_exact(
                        q_np,
                        exact_candidate_ids,
                        internal_k,
                    )
                exact_path = "rust_fused_exact" if dev.type == "cpu" else "rust_fused_exact_no_preload"
                stage_stats = self._empty_exact_stage_stats(exact_path)
                stage_stats["exact_ms"] = t_exact.elapsed_ms
                stage_stats["maxsim_ms"] = t_exact.elapsed_ms
                stage_stats["num_docs_scored"] = len(exact_candidate_ids)
                return (
                    list(zip(top_ids.tolist(), top_scores.tolist())),
                    exact_candidate_ids,
                    exact_path,
                    stage_stats,
                )
            except (AttributeError, RuntimeError):
                if self._pipeline is not None and dev.type == "cuda":
                    results, stage_stats = self._score_pipeline_fetch(
                        q,
                        shard_groups,
                        internal_k,
                        scfg,
                        dev,
                        exact_path="cuda_pipeline_fetch_fallback",
                        use_colbandit=False,
                    )
                    return results, exact_candidate_ids, "cuda_pipeline_fetch_fallback", stage_stats

                with Timer(sync_cuda=False) as t_fetch:
                    raw_bytes, offsets_arr, doc_ids_arr = self._rust_index.fetch_candidate_embeddings(
                        exact_candidate_ids,
                    )
                if len(doc_ids_arr) > 0:
                    emb_f16 = np.frombuffer(np.array(raw_bytes, copy=False), dtype=np.float16).reshape(-1, self._dim)
                    emb_t = torch.from_numpy(emb_f16)
                    offsets_list = [(int(offsets_arr[i, 0]), int(offsets_arr[i, 1])) for i in range(len(doc_ids_arr))]
                    doc_ids_list = doc_ids_arr.tolist()
                    shard_chunks = [(emb_t, offsets_list, doc_ids_list)]
                    ids, scores, score_stats = score_all_docs_topk(
                        q,
                        shard_chunks,
                        k=internal_k,
                        device=dev,
                        quantization_mode=scfg.quantization_mode,
                        variable_length_strategy=scfg.variable_length_strategy,
                        return_stats=True,
                    )
                    stage_stats = self._empty_exact_stage_stats("rust_fetch_torch_fallback")
                    stage_stats.update(score_stats)
                    stage_stats["fetch_ms"] = t_fetch.elapsed_ms
                    stage_stats["h2d_bytes"] = len(raw_bytes)
                    stage_stats["num_shards_fetched"] = 1
                    stage_stats["num_docs_scored"] = len(doc_ids_list)
                    return list(zip(ids, scores)), exact_candidate_ids, "rust_fetch_torch_fallback", stage_stats
                stage_stats = self._empty_exact_stage_stats("rust_fetch_torch_fallback")
                stage_stats["fetch_ms"] = t_fetch.elapsed_ms
                stage_stats["num_shards_fetched"] = 1
                return [], exact_candidate_ids, "rust_fetch_torch_fallback", stage_stats

        if self._pipeline is not None:
            results, stage_stats = self._score_pipeline_fetch(
                q,
                shard_groups,
                internal_k,
                scfg,
                dev,
                exact_path="pipeline_fetch",
                use_colbandit=False,
            )
            if results:
                return results, exact_candidate_ids, "pipeline_fetch", stage_stats
            return [], exact_candidate_ids, "pipeline_fetch", stage_stats
        return [], exact_candidate_ids, "none", self._empty_exact_stage_stats("none")

    def inspect_query_pipeline(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        n_probes: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self._is_built:
            raise RuntimeError("Index not built or loaded")

        self._ensure_warmup()
        q = torch.from_numpy(query_vectors).float() if isinstance(query_vectors, np.ndarray) else query_vectors.float()
        dev = self._resolve_scoring_device()
        scfg = self._apply_search_overrides(self._config.to_search_config(), **kwargs)
        internal_k = k
        n_sealed = len(self._doc_ids) if self._doc_ids else 0
        effective_k = min(scfg.k_candidates, max(n_sealed, 1))

        with Timer(sync_cuda=True) as t_total:
            with Timer(sync_cuda=True) as t_route:
                routed = self._router.route(
                    q,
                    k_candidates=effective_k,
                    prefetch_doc_cap=self._route_prefetch_cap(scfg),
                    nprobe_override=n_probes,
                    search_k_cap=scfg.lemur_search_k_cap,
                )

            with Timer(sync_cuda=dev.type == "cuda") as t_prune:
                pruned_ids, routed_by_shard, prune_path = self._prune_routed_candidates(
                    q,
                    routed,
                    scfg,
                    dev,
                )
            sealed_results, exact_candidate_ids, exact_path, exact_stats = self._score_sealed_candidates(
                q,
                pruned_ids,
                routed_by_shard,
                internal_k,
                scfg,
                dev,
            )
        self._last_prune_path = prune_path
        self._last_exact_path = exact_path

        result_ids = [int(doc_id) for doc_id, _score in sealed_results[:k]]
        return {
            "route_ms": t_route.elapsed_ms,
            "prune_ms": t_prune.elapsed_ms,
            "fetch_ms": exact_stats["fetch_ms"],
            "exact_ms": exact_stats["exact_ms"],
            "exact_prepare_ms": exact_stats["prepare_ms"],
            "h2d_ms": exact_stats["h2d_ms"],
            "maxsim_ms": exact_stats["maxsim_ms"],
            "topk_ms": exact_stats["topk_ms"],
            "total_ms": t_total.elapsed_ms,
            "h2d_bytes": exact_stats["h2d_bytes"],
            "num_shards_fetched": exact_stats["num_shards_fetched"],
            "num_docs_scored": exact_stats["num_docs_scored"],
            "router_device": self._router_device,
            "exact_device": str(dev),
            "prune_path": prune_path,
            "exact_path": exact_path,
            "routed_ids": [int(did) for did in routed.doc_ids],
            "pruned_ids": [int(did) for did in pruned_ids],
            "exact_candidate_ids": [int(did) for did in exact_candidate_ids],
            "result_ids": result_ids,
        }

    def search_multivector(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        n_probes: Optional[int] = None,
        filters: Optional[Dict] = None,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        """Search with a multi-vector query (ColBERT-style token embeddings).

        Args:
            query_vectors: Query token embeddings, shape [n_tokens, dim].
            k: Number of results to return.
            ef: Accepted for API compatibility with HNSW; not used by shard routing.
            n_probes: Optional nprobe override for IVF-PQ routing indices.
            filters: Optional payload filters (post-filter with widened k).

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if not self._is_built:
            raise RuntimeError("Index not built or loaded")

        t_start = time.perf_counter()
        self._ensure_warmup()

        q = torch.from_numpy(query_vectors).float() if isinstance(query_vectors, np.ndarray) else query_vectors.float()
        dev = self._resolve_scoring_device()

        scfg = self._apply_search_overrides(self._config.to_search_config(), **kwargs)
        n_sealed = len(self._doc_ids) if self._doc_ids else 0

        internal_k = k * 4 if filters else k
        effective_k = min(scfg.k_candidates, max(n_sealed, 1))

        with self._lock:
            if self._memtable:
                _mt_docs_snap, mt_payloads_snap, tombstones = self._memtable.snapshot()
            else:
                _mt_docs_snap, mt_payloads_snap, tombstones = {}, {}, set()
            sealed_payloads_snap = (
                dict(self._sealed_payloads) if hasattr(self, "_sealed_payloads") and self._sealed_payloads else {}
            )

        with Timer(sync_cuda=True) as t_route:
            routed = self._router.route(
                q,
                k_candidates=effective_k,
                prefetch_doc_cap=self._route_prefetch_cap(scfg),
                nprobe_override=n_probes,
                search_k_cap=scfg.lemur_search_k_cap,
            )

        with Timer(sync_cuda=dev.type == "cuda") as t_prune:
            pruned_ids, routed_by_shard, _prune_path = self._prune_routed_candidates(
                q,
                routed,
                scfg,
                dev,
            )
        sealed_results, exact_candidate_ids, _exact_path, exact_stats = self._score_sealed_candidates(
            q,
            pruned_ids,
            routed_by_shard,
            internal_k,
            scfg,
            dev,
        )
        self._last_prune_path = _prune_path
        self._last_exact_path = _exact_path

        memtable_results: List[Tuple[int, float]] = []
        if self._memtable and self._memtable.size > 0:
            memtable_results = self._memtable.search(
                query_vectors if isinstance(query_vectors, np.ndarray) else query_vectors.numpy(),
                k=internal_k,
            )

        with Timer(sync_cuda=False) as t_postprocess:
            merged: Dict[int, float] = {}
            for did, sc in sealed_results:
                if did not in tombstones:
                    merged[did] = max(merged.get(did, float("-inf")), sc)
            for did, sc in memtable_results:
                merged[did] = max(merged.get(did, float("-inf")), sc)

            ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

            if filters:
                filtered = []
                for did, sc in ranked:
                    payload = dict(sealed_payloads_snap.get(did, {}))
                    if did in mt_payloads_snap:
                        payload.update(mt_payloads_snap[did])
                    if self._evaluate_filter(payload, filters):
                        filtered.append((did, sc))
                ranked = filtered

            result = ranked[:k]

        elapsed_us = (time.perf_counter() - t_start) * 1_000_000
        self._emit_metric("search_latency_us", elapsed_us)
        self._emit_metric("candidates_routed", len(routed.doc_ids))
        self._emit_metric("candidates_scored", len(exact_candidate_ids))
        self._emit_metric("route_ms", t_route.elapsed_ms)
        self._emit_metric("prune_ms", t_prune.elapsed_ms)
        self._emit_metric("fetch_ms", exact_stats["fetch_ms"])
        self._emit_metric("exact_ms", exact_stats["exact_ms"])
        self._emit_metric("exact_prepare_ms", exact_stats["prepare_ms"])
        self._emit_metric("h2d_ms", exact_stats["h2d_ms"])
        self._emit_metric("h2d_bytes", exact_stats["h2d_bytes"])
        self._emit_metric("num_shards_fetched", exact_stats["num_shards_fetched"])
        self._emit_metric("maxsim_ms", exact_stats["maxsim_ms"])
        self._emit_metric("topk_ms", exact_stats["topk_ms"])
        self._emit_metric("postprocess_ms", t_postprocess.elapsed_ms)

        now = time.perf_counter()
        if (
            self._metrics_hook
            and self._store
            and (self._page_cache_last is None or now - self._page_cache_last[0] > 5.0)
        ):
            try:
                pc = self._store.page_cache_residency()
                if pc is not None:
                    self._page_cache_last = (now, pc["hit_rate"])
                    self._emit_metric("page_cache_hit_rate", pc["hit_rate"])
            except Exception:
                pass

        return result

    def search_batch(
        self,
        queries: List[np.ndarray],
        k: int = 10,
        filters: Optional[Dict] = None,
        max_workers: int = 1,
        n_probes: Optional[int] = None,
        **kwargs,
    ) -> List[List[Tuple[int, float]]]:
        """Batch search over multiple queries."""
        if not queries:
            return []
        if filters:
            return [self.search_multivector(q, k=k, filters=filters, n_probes=n_probes, **kwargs) for q in queries]
        if not self._is_built:
            raise RuntimeError("Index not built or loaded")

        self._ensure_warmup()
        dev = self._resolve_scoring_device()
        scfg = self._apply_search_overrides(self._config.to_search_config(), **kwargs)
        n_sealed = len(self._doc_ids) if self._doc_ids else 0
        effective_k = min(scfg.k_candidates, max(n_sealed, 1))
        query_tensors = [torch.from_numpy(q).float() if isinstance(q, np.ndarray) else q.float() for q in queries]

        with self._lock:
            if self._memtable:
                _mt_docs_snap, _mt_payloads_snap, tombstones = self._memtable.snapshot()
            else:
                tombstones = set()

        routed_plans = self._router.route_batch(
            query_tensors,
            k_candidates=effective_k,
            prefetch_doc_cap=self._route_prefetch_cap(scfg),
            nprobe_override=n_probes,
            search_k_cap=scfg.lemur_search_k_cap,
        )

        def _finish_one(item: Tuple[int, torch.Tensor, CandidatePlan]) -> Tuple[int, List[Tuple[int, float]]]:
            idx, q, routed = item
            pruned_ids, routed_by_shard, _prune_path = self._prune_routed_candidates(
                q,
                routed,
                scfg,
                dev,
            )
            sealed_results, _exact_candidate_ids, _exact_path, _exact_stats = self._score_sealed_candidates(
                q,
                pruned_ids,
                routed_by_shard,
                k,
                scfg,
                dev,
            )
            merged: Dict[int, float] = {}
            for did, sc in sealed_results:
                if did not in tombstones:
                    merged[did] = max(merged.get(did, float("-inf")), sc)
            if self._memtable and self._memtable.size > 0:
                query_np = queries[idx] if isinstance(queries[idx], np.ndarray) else queries[idx].detach().cpu().numpy()
                for did, sc in self._memtable.search(query_np, k=k):
                    merged[did] = max(merged.get(did, float("-inf")), sc)
            ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
            return idx, ranked[:k]

        work_items = list(zip(range(len(query_tensors)), query_tensors, routed_plans))
        if max_workers <= 1:
            ordered = [_finish_one(item) for item in work_items]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(max_workers, len(work_items))) as pool:
                ordered = list(pool.map(_finish_one, work_items))
        ordered.sort(key=lambda pair: pair[0])
        return [result for _idx, result in ordered]

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[int, float]]:
        """Alias for search_multivector."""
        return self.search_multivector(query, k=k, filters=filters)

    def scroll(
        self, limit: int = 100, offset: int = 0, filters: Optional[Dict] = None
    ) -> Tuple[List[int], Optional[int]]:
        """Paginate through all document IDs.

        Returns:
            (page_ids, next_offset) — next_offset is None when done.
        """
        with self._lock:
            all_ids = sorted(set(self._doc_ids or []))
            if self._memtable:
                docs, _, tombstones = self._memtable.snapshot()
                all_ids = sorted((set(all_ids) | set(docs.keys())) - tombstones)

            if filters:
                all_ids = [did for did in all_ids if self._match_filter(did, filters)]

        page = all_ids[offset : offset + limit]
        next_off = offset + limit if offset + limit < len(all_ids) else None
        return page, next_off

    def explain_score(
        self,
        query: np.ndarray,
        doc_id: int,
    ) -> Tuple[Optional[List[float]], Optional[List[int]]]:
        """Per-query-token score attribution for a single document.

        Returns (token_scores, matched_doc_tokens) or (None, None) if unavailable.
        """
        doc_vec = self._get_doc_vectors(doc_id)
        if doc_vec is None:
            return None, None
        q = torch.from_numpy(query).float()
        d = torch.from_numpy(doc_vec).float()
        sim = q @ d.T
        max_sim, matched = sim.max(dim=1)
        return max_sim.tolist(), matched.tolist()

    def _explain_score(
        self,
        query: np.ndarray,
        doc_id: int,
    ) -> Tuple[Optional[List[float]], Optional[List[int]]]:
        """Backward-compatible private alias for token attribution."""
        return self.explain_score(query, doc_id)

