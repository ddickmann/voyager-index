"""ANN backend management for the LEMUR router."""
from __future__ import annotations

from .common import *  # noqa: F401,F403
from .backends import _ProjectionFallbackModel, _TorchMipsIndex
from .common import FAISS_AVAILABLE, OFFICIAL_LEMUR_AVAILABLE, _OfficialLemur, faiss, logger

class LemurRouterAnnMixin:
    def _new_lemur_backend(self, load_saved: bool = False):
        backend_dir = self.index_dir / "lemur_model"
        backend_dir.mkdir(parents=True, exist_ok=True)
        if OFFICIAL_LEMUR_AVAILABLE:
            inst = _OfficialLemur(index=str(backend_dir), device=self.device)
            if load_saved:
                mlp_path = backend_dir / "mlp.pt"
                if mlp_path.exists():
                    inst.load_mlp(mlp_path)
                    logger.info("Loaded LEMUR MLP from %s", mlp_path)
            return inst
        logger.warning("Official LEMUR package unavailable; using projection fallback backend")
        return _ProjectionFallbackModel(backend_dir)

    def _get_gpu_resources(self):
        """Singleton GPU resources to avoid leaks on rebuilds (thread-safe)."""
        with self._gpu_res_lock:
            if self._gpu_res is None and hasattr(faiss, "StandardGpuResources"):
                self._gpu_res = faiss.StandardGpuResources()
            return self._gpu_res

    @staticmethod
    def _set_nprobe_if_ivf(index) -> None:
        """Set nprobe on IVF indices (including those wrapped in IndexIDMap2)."""
        target = index
        if hasattr(target, 'index') and target.index is not None:
            target = target.index
        if hasattr(target, 'nprobe'):
            nlist = getattr(target, 'nlist', 10)
            target.nprobe = max(1, min(nlist, 10))

    def _apply_nprobe_override(self, nprobe_override: Optional[int]):
        """Temporarily set nprobe on IVF indices; returns (target, prev) or None."""
        if nprobe_override is None or self._index is None:
            return None
        target = self._index
        if hasattr(target, "index") and target.index is not None:
            target = target.index
        if hasattr(target, "nprobe"):
            prev = target.nprobe
            target.nprobe = nprobe_override
            return (target, prev)
        return None

    @staticmethod
    def _restore_nprobe(saved) -> None:
        """Restore nprobe to the previous value. No-op if *saved* is None."""
        if saved is not None:
            target, prev = saved
            target.nprobe = prev

    def _rebuild_ann(self) -> None:
        old_index = self._index
        self._gpu_index_active = False
        if self._weights.ndim != 2 or self._weights.shape[0] == 0:
            self._index = _TorchMipsIndex(device=self.device)
            del old_index
            return
        if self._use_faiss:
            dim = int(self._weights.shape[1])
            w_np = self._weights.cpu().numpy().astype(np.float32)
            ids_np = np.arange(self._weights.shape[0], dtype=np.int64)
            use_gpu = (
                self.device != "cpu"
                and torch.cuda.is_available()
                and hasattr(faiss, "StandardGpuResources")
            )
            if self.ann_backend == "faiss_ivfpq_ip":
                n_vectors = self._weights.shape[0]
                nlist = max(1, min(int(math.sqrt(n_vectors)), n_vectors // 39 + 1))
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, min(16, dim), 8, faiss.METRIC_INNER_PRODUCT)
                index.train(w_np)
                index.nprobe = max(1, min(nlist, 10))
                id_index = faiss.IndexIDMap2(index)
                id_index.add_with_ids(w_np, ids_np)
            else:
                base = faiss.IndexFlatIP(dim)
                id_index = faiss.IndexIDMap2(base)
                id_index.add_with_ids(w_np, ids_np)
            if use_gpu:
                try:
                    res = self._get_gpu_resources()
                    id_index = faiss.index_cpu_to_gpu(res, 0, id_index)
                    self._gpu_index_active = True
                    logger.info("ANN index moved to GPU (faiss-gpu)")
                except Exception as e:
                    logger.warning("faiss-gpu transfer failed, using CPU: %s", e)
            self._index = id_index
        else:
            idx = _TorchMipsIndex(device=self.device)
            idx.build(self._weights, list(range(self._weights.shape[0])))
            self._index = idx
        del old_index

    def _search(
        self,
        feats: torch.Tensor,
        k: int,
        use_lock: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("ANN index is not built")
        if self._use_faiss:
            feats_np = self._as_numpy_float32_contiguous(feats)
            if use_lock is None:
                use_lock = self._gpu_index_active
            n_queries = int(feats_np.shape[0])
            # Cap FAISS OMP threads to the number of queries (clamped to 8).
            # Online single-query latency on a CPU IVFPQ over a few-thousand-
            # vector router index is dominated by 64-thread fork/join sync;
            # forcing a small thread count drops route() from ~85 ms p50
            # to ~0.5 ms p50 on H100 (64-core box). Indexing paths that run
            # FAISS train/add are not affected because they use the FAISS
            # APIs directly, not _search.
            target_threads = max(1, min(n_queries, 8))
            prev_threads = faiss.omp_get_max_threads()
            try:
                if target_threads != prev_threads:
                    faiss.omp_set_num_threads(target_threads)
                if use_lock:
                    with self._search_lock:
                        return self._index.search(feats_np, k)
                return self._index.search(feats_np, k)
            finally:
                if target_threads != prev_threads:
                    faiss.omp_set_num_threads(prev_threads)
        return self._index.search(feats, k)

