from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None
    FAISS_AVAILABLE = False

try:
    from ._lemur_vendor import Lemur as _OfficialLemur
    OFFICIAL_LEMUR_AVAILABLE = True
except Exception:
    try:
        from lemur import Lemur as _OfficialLemur  # type: ignore
        OFFICIAL_LEMUR_AVAILABLE = True
    except Exception:  # pragma: no cover - optional dependency
        _OfficialLemur = None
        OFFICIAL_LEMUR_AVAILABLE = False


@dataclass(slots=True)
class CandidatePlan:
    doc_ids: List[int]
    shard_ids: List[int]
    by_shard: Dict[int, List[int]]
    generation: int
    post_tombstone_count: int


@dataclass(slots=True)
class RouterState:
    generation: int = 0
    ann_backend: str = "faiss_flat_ip"
    feature_dim: int = 0
    backend_name: str = "official_lemur"
    dirty_ops_count: int = 0
    dirty_doc_ratio: float = 0.0
    dirty_shard_ratio: float = 0.0
    live_docs: int = 0
    total_docs: int = 0
    max_age_hours: float = 0.0


class _TorchMipsIndex:
    """GPU-accelerated MIPS index (brute-force matmul + topk)."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self.weights = torch.empty((0, 0), dtype=torch.float32)
        self.ids: List[int] = []

    def build(self, weights: torch.Tensor, ids: Sequence[int]) -> None:
        self.weights = weights.detach().to(self._device, dtype=torch.float32).contiguous()
        self.ids = [int(x) for x in ids]

    def add(self, weights: torch.Tensor, ids: Sequence[int]) -> None:
        weights = weights.detach().to(self._device, dtype=torch.float32).contiguous()
        if self.weights.numel() == 0:
            self.weights = weights
        else:
            self.weights = torch.cat([self.weights, weights], dim=0)
        self.ids.extend(int(x) for x in ids)

    def search(self, queries: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.weights.numel() == 0:
            return np.empty((queries.shape[0], 0), dtype=np.float32), np.empty((queries.shape[0], 0), dtype=np.int64)
        q = queries.detach().to(self._device, dtype=torch.float32)
        scores = q @ self.weights.T
        topk = min(k, scores.shape[1])
        vals, idx = torch.topk(scores, topk, dim=1)
        return vals.cpu().numpy(), idx.cpu().numpy().astype(np.int64)


class _ProjectionFallbackModel:
    """Very small self-contained proxy if official LEMUR is unavailable.

    This is not a research-faithful reimplementation of LEMUR. It is a durable
    fallback that keeps the pipeline operational with a predictable interface.
    """

    def __init__(self, index_dir: Path, projection_dim: int = 128, seed: int = 42) -> None:
        self.index_dir = Path(index_dir)
        self.projection_dim = int(projection_dim)
        self.seed = int(seed)
        self.proj: Optional[torch.Tensor] = None

    def fit(self, train: torch.Tensor, train_counts: torch.Tensor, epochs: int = 0, verbose: bool = False) -> None:
        dim = int(train.shape[1])
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        proj = torch.randn(dim, self.projection_dim, generator=g, dtype=torch.float32)
        proj = torch.nn.functional.normalize(proj, dim=0)
        self.proj = proj.contiguous()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.proj, self.index_dir / "fallback_proj.pt")

    def compute_features(self, query_pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.proj is None:
            self.proj = torch.load(self.index_dir / "fallback_proj.pt", weights_only=True)
        q_tokens, q_counts = query_pair
        return _aggregate_doc_matrix(q_tokens, q_counts) @ self.proj

    def compute_weights(self, docs: torch.Tensor, doc_counts: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            self.proj = torch.load(self.index_dir / "fallback_proj.pt", weights_only=True)
        return _aggregate_doc_matrix(docs, doc_counts) @ self.proj


class LemurRouter:
    """Production wrapper around LEMUR-style proxy routing.

    The class keeps the retrieval-facing contract stable even if the underlying
    ANN backend or the official LEMUR package changes.
    """

    def __init__(
        self,
        index_dir: Path,
        ann_backend: str = "faiss_flat_ip",
        device: str = "cpu",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        ann_backend = ann_backend.replace("hnsw", "flat")
        self.ann_backend = ann_backend
        self.device = device
        self._lemur = None
        self._index = None
        self._state = RouterState(ann_backend=ann_backend)
        self._doc_ids: List[int] = []
        self._row_to_doc_id: List[int] = []
        self._doc_id_to_row: Dict[int, int] = {}
        self._doc_id_to_shard: Dict[int, int] = {}
        self._tombstones: set[int] = set()
        self._weights = torch.empty((0, 0), dtype=torch.float32)
        self._use_faiss = FAISS_AVAILABLE and ann_backend.startswith("faiss")
        self._gpu_res = None
        self._gpu_res_lock = __import__("threading").Lock()
        self._search_lock = __import__("threading").Lock()
        self._load_if_present()

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------
    def fit_initial(
        self,
        pooled_doc_vectors: torch.Tensor,
        pooled_doc_counts: torch.Tensor,
        doc_ids: Sequence[int],
        doc_id_to_shard: Dict[int, int],
        epochs: int = 10,
    ) -> None:
        vectors = pooled_doc_vectors.detach().cpu().contiguous()
        counts = pooled_doc_counts.detach().cpu().to(torch.int32).contiguous()
        if int(counts.sum(dtype=torch.int64).item()) != int(vectors.shape[0]):
            raise ValueError("pooled_doc_counts do not sum to pooled_doc_vectors rows")

        self._lemur = self._new_lemur_backend()
        self._lemur.fit(train=vectors, train_counts=counts, epochs=int(epochs), verbose=False)
        weights = self._lemur.compute_weights(vectors, counts).detach().cpu().to(torch.float32).contiguous()
        self._weights = weights
        self._doc_ids = [int(x) for x in doc_ids]
        self._row_to_doc_id = list(self._doc_ids)
        self._doc_id_to_row = {doc_id: idx for idx, doc_id in enumerate(self._row_to_doc_id)}
        self._doc_id_to_shard = {int(k): int(v) for k, v in doc_id_to_shard.items()}
        self._tombstones.clear()
        self._state.generation += 1
        self._state.feature_dim = int(weights.shape[1]) if weights.ndim == 2 else 0
        self._state.backend_name = "official_lemur" if OFFICIAL_LEMUR_AVAILABLE else "fallback_proxy"
        self._state.live_docs = len(self._doc_ids)
        self._state.total_docs = len(self._doc_ids)
        self._state.dirty_ops_count = 0
        self._state.dirty_doc_ratio = 0.0
        self._state.dirty_shard_ratio = 0.0
        self._rebuild_ann()
        self.save()

    def add_or_update_docs(
        self,
        pooled_doc_vectors: torch.Tensor,
        pooled_doc_counts: torch.Tensor,
        doc_ids: Sequence[int],
        doc_id_to_shard: Dict[int, int],
    ) -> None:
        if self._lemur is None:
            raise RuntimeError("router is not initialized; call fit_initial first")
        vectors = pooled_doc_vectors.detach().cpu().to(torch.float32).contiguous()
        counts = pooled_doc_counts.detach().cpu().to(torch.int32).contiguous()
        new_weights = self._lemur.compute_weights(vectors, counts).detach().cpu().to(torch.float32).contiguous()

        added_rows = []
        added_ids = []
        for i, doc_id in enumerate(doc_ids):
            doc_id = int(doc_id)
            shard_id = int(doc_id_to_shard[doc_id])
            if doc_id in self._doc_id_to_row:
                row = self._doc_id_to_row[doc_id]
                self._weights[row] = new_weights[i]
                self._doc_id_to_shard[doc_id] = shard_id
                self._tombstones.discard(doc_id)
            else:
                row = len(self._row_to_doc_id)
                self._row_to_doc_id.append(doc_id)
                self._doc_id_to_row[doc_id] = row
                self._doc_id_to_shard[doc_id] = shard_id
                self._doc_ids.append(doc_id)
                added_rows.append(new_weights[i].unsqueeze(0))
                added_ids.append(doc_id)

        if added_rows:
            self._weights = torch.cat([self._weights, torch.cat(added_rows, dim=0)], dim=0)
            self._state.total_docs += len(added_rows)
            self._state.live_docs += len(added_rows)

        self._rebuild_ann()
        self._mark_dirty(len(doc_ids), set(doc_id_to_shard.values()))
        self.save()

    def delete_docs(self, doc_ids: Sequence[int]) -> None:
        deleted_shards = set()
        for doc_id in doc_ids:
            doc_id = int(doc_id)
            if doc_id in self._doc_id_to_shard:
                deleted_shards.add(self._doc_id_to_shard[doc_id])
                self._tombstones.add(doc_id)
        self._state.live_docs = max(0, self._state.total_docs - len(self._tombstones))
        self._mark_dirty(len(doc_ids), deleted_shards)
        self.save()

    # ------------------------------------------------------------------
    # Query-time routing
    # ------------------------------------------------------------------
    def route(
        self,
        query_vectors: torch.Tensor,
        k_candidates: int = 2000,
        prefetch_doc_cap: int = 10000,
        nprobe_override: Optional[int] = None,
    ) -> CandidatePlan:
        if self._lemur is None or self._index is None:
            raise RuntimeError("router is not loaded")
        q = query_vectors.detach().cpu().to(torch.float32)
        q_counts = torch.tensor([q.shape[0]], dtype=torch.int32)
        feats = self._lemur.compute_features((q, q_counts)).detach().cpu().to(torch.float32).contiguous()
        tombstone_headroom = max(64, int(len(self._tombstones) * 1.5))
        index_size = self._weights.shape[0] if self._weights is not None else k_candidates
        search_k = min(k_candidates + tombstone_headroom, index_size)
        saved_nprobe = self._apply_nprobe_override(nprobe_override)
        try:
            _, row_ids = self._search(feats, max(search_k, 1))
        finally:
            self._restore_nprobe(saved_nprobe)
        doc_ids: List[int] = []
        for row in row_ids[0].tolist() if row_ids.size else []:
            if row < 0 or row >= len(self._row_to_doc_id):
                continue
            doc_id = self._row_to_doc_id[int(row)]
            if doc_id in self._tombstones:
                continue
            doc_ids.append(doc_id)
            if len(doc_ids) >= int(prefetch_doc_cap):
                break
        by_shard: Dict[int, List[int]] = {}
        for doc_id in doc_ids:
            shard_id = int(self._doc_id_to_shard[doc_id])
            by_shard.setdefault(shard_id, []).append(doc_id)
        return CandidatePlan(
            doc_ids=doc_ids,
            shard_ids=sorted(by_shard.keys()),
            by_shard=by_shard,
            generation=self._state.generation,
            post_tombstone_count=len(doc_ids),
        )

    # ------------------------------------------------------------------
    # Persistence / retraining state
    # ------------------------------------------------------------------
    def should_full_retrain(
        self,
        retrain_every_ops: int = 50_000,
        retrain_dirty_doc_ratio: float = 0.05,
        retrain_dirty_shard_ratio: float = 0.10,
    ) -> bool:
        return (
            self._state.dirty_ops_count >= retrain_every_ops
            or self._state.dirty_doc_ratio >= retrain_dirty_doc_ratio
            or self._state.dirty_shard_ratio >= retrain_dirty_shard_ratio
        )

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._weights, self.index_dir / "weights.pt")
        with open(self.index_dir / "doc_maps.json", "w") as f:
            json.dump(
                {
                    "row_to_doc_id": self._row_to_doc_id,
                    "doc_id_to_shard": self._doc_id_to_shard,
                    "tombstones": sorted(self._tombstones),
                },
                f,
            )
        with open(self.index_dir / "router_state.json", "w") as f:
            json.dump(asdict(self._state), f, indent=2)
        if self._use_faiss and self._index is not None:
            idx_to_save = self._index
            if hasattr(faiss, "index_gpu_to_cpu"):
                try:
                    idx_to_save = faiss.index_gpu_to_cpu(self._index)
                except Exception:
                    pass
            faiss.write_index(idx_to_save, str(self.index_dir / "ann.index"))

    def load(self) -> None:
        self._load_if_present(required=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_if_present(self, required: bool = False) -> None:
        state_path = self.index_dir / "router_state.json"
        if not state_path.exists():
            if required:
                raise FileNotFoundError(f"router state missing at {state_path}")
            return
        with open(state_path) as f:
            self._state = RouterState(**json.load(f))
        self.ann_backend = self._state.ann_backend.replace("hnsw", "flat")
        self._weights = torch.load(self.index_dir / "weights.pt", weights_only=True).detach().cpu().to(torch.float32)
        with open(self.index_dir / "doc_maps.json") as f:
            maps = json.load(f)
        self._row_to_doc_id = [int(x) for x in maps["row_to_doc_id"]]
        self._doc_ids = list(self._row_to_doc_id)
        self._doc_id_to_row = {doc_id: idx for idx, doc_id in enumerate(self._row_to_doc_id)}
        self._doc_id_to_shard = {int(k): int(v) for k, v in maps["doc_id_to_shard"].items()}
        self._tombstones = {int(x) for x in maps.get("tombstones", [])}
        self._lemur = self._new_lemur_backend(load_saved=True)
        if self._use_faiss and (self.index_dir / "ann.index").exists():
            cpu_index = faiss.read_index(str(self.index_dir / "ann.index"))
            self._set_nprobe_if_ivf(cpu_index)
            use_gpu = (
                self.device != "cpu"
                and torch.cuda.is_available()
                and hasattr(faiss, "StandardGpuResources")
            )
            if use_gpu:
                try:
                    res = self._get_gpu_resources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    logger.info("ANN index loaded to GPU (faiss-gpu)")
                except Exception:
                    self._index = cpu_index
            else:
                self._index = cpu_index
        else:
            self._rebuild_ann()

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
                    logger.info("ANN index moved to GPU (faiss-gpu)")
                except Exception as e:
                    logger.warning("faiss-gpu transfer failed, using CPU: %s", e)
            self._index = id_index
        else:
            idx = _TorchMipsIndex(device=self.device)
            idx.build(self._weights, list(range(self._weights.shape[0])))
            self._index = idx
        del old_index

    def _search(self, feats: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("ANN index is not built")
        if self._use_faiss:
            feats_np = feats.cpu().numpy().astype(np.float32)
            with self._search_lock:
                return self._index.search(feats_np, k)
        return self._index.search(feats, k)

    def _mark_dirty(self, changed_ops: int, changed_shards: Iterable[int]) -> None:
        self._state.dirty_ops_count += int(changed_ops)
        if self._state.total_docs > 0:
            self._state.dirty_doc_ratio = min(1.0, self._state.dirty_ops_count / float(self._state.total_docs))
        total_shards = max(1, len(set(self._doc_id_to_shard.values())))
        changed = len(set(int(x) for x in changed_shards))
        if changed > 0:
            self._state.dirty_shard_ratio = min(1.0, max(self._state.dirty_shard_ratio, changed / float(total_shards)))


def _aggregate_doc_matrix(flat_vectors: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Fallback aggregation: mean + max pooling mixed together."""
    flat_vectors = flat_vectors.detach().cpu().to(torch.float32)
    counts = counts.detach().cpu().to(torch.long)
    out = []
    pos = 0
    for count in counts.tolist():
        count = int(count)
        block = flat_vectors[pos:pos + count]
        pos += count
        if count == 0:
            out.append(torch.zeros(flat_vectors.shape[1], dtype=torch.float32))
            continue
        mean = block.mean(dim=0)
        mx = block.max(dim=0).values
        out.append(torch.nn.functional.normalize(0.5 * (mean + mx), dim=0))
    return torch.stack(out, dim=0) if out else torch.empty((0, flat_vectors.shape[1]), dtype=torch.float32)
