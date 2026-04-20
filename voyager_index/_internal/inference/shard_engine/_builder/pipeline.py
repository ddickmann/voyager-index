"""Offline shard-engine build orchestration."""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..centroid_router import CentroidRouter
from ..config import AnnBackend, BuildConfig, Compression, RouterType, StorageLayout
from ..lemur_router import LemurRouter
from ..pooling import TokenPooler
from ..shard_store import ShardStore
from .corpus import DEFAULT_NPZ, load_corpus
from .layout import _index_dir, assign_storage_shards

log = logging.getLogger(__name__)


def _mem_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1.0

def build(cfg: BuildConfig, npz_path: Path = DEFAULT_NPZ, device: str = "cuda") -> Path:
    index_dir = _index_dir(cfg)

    if (index_dir / "manifest.json").exists():
        router_path = index_dir / ("lemur" if cfg.router_type == RouterType.LEMUR else "router")
        if (router_path / "router_state.json").exists():
            log.info("Index already exists at %s, skipping build", index_dir)
            return index_dir

    index_dir.mkdir(parents=True, exist_ok=True)
    all_vectors, doc_offsets, doc_ids, *_rest, dim = load_corpus(npz_path, max_docs=cfg.corpus_size)

    active_vectors = all_vectors
    active_offsets = doc_offsets
    active_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int32)
    centroid_to_shard = None

    # Optional token pooling
    if cfg.pooling.enabled:
        pooler = TokenPooler(
            method=cfg.pooling.method,
            pool_factor=cfg.pooling.pool_factor,
            protected_tokens=cfg.pooling.protected_tokens,
        )
        pooled_vectors, pooled_offsets, pooled_counts = pooler.pool_docs(all_vectors, doc_offsets)
        active_vectors = pooled_vectors.numpy()
        active_offsets = pooled_offsets
        active_counts = pooled_counts.numpy()
        log.info(
            "Token pooling reduced corpus from %d to %d token vectors (%.2fx)",
            int(all_vectors.shape[0]), int(active_vectors.shape[0]),
            float(all_vectors.shape[0]) / max(1, int(active_vectors.shape[0])),
        )

    # Train router
    if cfg.router_type == RouterType.CENTROID:
        t0 = time.time()
        router, shard_assignments, centroid_to_shard = CentroidRouter.train(
            all_vectors=np.asarray(active_vectors, dtype=np.float32),
            doc_offsets=active_offsets,
            n_centroids=cfg.n_centroids,
            n_shards=cfg.n_shards,
            sample_fraction=cfg.kmeans_sample_fraction,
            max_iter=cfg.max_kmeans_iter,
            seed=cfg.seed,
            device=device,
        )
        train_s = time.time() - t0
        log.info("Centroid router training done in %.1fs", train_s)
    else:
        t0 = time.time()
        lemur_device = device if device != "cpu" else cfg.lemur.device
        router = LemurRouter(
            index_dir=index_dir / "lemur",
            ann_backend=cfg.lemur.ann_backend.value,
            device=lemur_device,
        )
        # Build FP16 tensor once, reuse for both passes
        doc_vecs_f16 = torch.from_numpy(np.asarray(active_vectors)).to(torch.float16)
        doc_counts_t = torch.from_numpy(active_counts)

        # First pass: temporary token-balanced shards for LEMUR training
        shard_assignments = assign_storage_shards(
            active_offsets, cfg.n_shards, cfg.seed, StorageLayout.TOKEN_BALANCED,
        )
        doc_id_to_shard = {doc_id: int(shard_assignments[i]) for i, doc_id in enumerate(doc_ids)}
        router.fit_initial(
            pooled_doc_vectors=doc_vecs_f16,
            pooled_doc_counts=doc_counts_t,
            doc_ids=doc_ids,
            doc_id_to_shard=doc_id_to_shard,
            epochs=cfg.lemur.epochs,
        )
        # Second pass: use learned proxy weights for better shard layout
        proxy_weights = router._weights.clone()
        shard_assignments = assign_storage_shards(
            active_offsets, cfg.n_shards, cfg.seed, cfg.layout, proxy_weights=proxy_weights,
        )
        doc_id_to_shard = {doc_id: int(shard_assignments[i]) for i, doc_id in enumerate(doc_ids)}
        router.fit_initial(
            pooled_doc_vectors=doc_vecs_f16,
            pooled_doc_counts=doc_counts_t,
            doc_ids=doc_ids,
            doc_id_to_shard=doc_id_to_shard,
            epochs=cfg.lemur.epochs,
        )
        del doc_vecs_f16, doc_counts_t
        gc.collect()
        train_s = time.time() - t0
        log.info("LEMUR router training done in %.1fs", train_s)

    if cfg.layout == StorageLayout.RANDOM:
        rng = np.random.RandomState(cfg.seed + 1)
        shard_assignments = rng.randint(0, cfg.n_shards, size=len(doc_ids)).astype(np.int32)
        if cfg.router_type == RouterType.LEMUR:
            doc_id_to_shard = {doc_id: int(shard_assignments[i]) for i, doc_id in enumerate(doc_ids)}
            router._doc_id_to_shard = doc_id_to_shard
            router.save()

    # ROQ4: train quantizer and pre-encode all documents
    roq_quantizer = None
    roq_doc_codes = None
    roq_doc_meta = None
    rroq158_payload = None
    rroq4_riem_payload = None
    if cfg.compression == Compression.RROQ158:
        # Hard-fail on actual codec errors (silently writing a 12× larger
        # FP16 index that scores in a different range would be a debugging
        # nightmare) — but auto-shrink K and auto-fall-back to FP16 when
        # the corpus is physically too small to host the codec, with a
        # loud log. Mirrors
        # ``voyager_index/_internal/inference/shard_engine/_manager/lifecycle.py``.
        from voyager_index._internal.inference.quantization.rroq158 import (
            Rroq158Config,
            choose_effective_rroq158_k,
            encode_rroq158,
        )
        active_arr = np.asarray(active_vectors)
        n_tokens = int(active_arr.shape[0])
        token_dim = int(active_arr.shape[1]) if active_arr.ndim >= 2 else 0
        gs = int(cfg.rroq158_group_size)
        if token_dim and token_dim < gs:
            log.warning(
                "RROQ158 requested but token dim=%d is smaller than "
                "group_size=%d. Falling back to FP16 — rroq158 needs at "
                "least group_size coordinates per token.",
                token_dim, gs,
            )
            cfg.compression = Compression.FP16
        elif n_tokens < gs:
            log.warning(
                "RROQ158 requested but corpus has only %d tokens "
                "(< group_size=%d). Falling back to FP16 — rroq158 needs "
                "at least one ternary group of tokens to encode.",
                n_tokens, gs,
            )
            cfg.compression = Compression.FP16
        else:
            effective_k = choose_effective_rroq158_k(
                n_tokens=n_tokens,
                requested_k=int(cfg.rroq158_k),
                group_size=gs,
            )
            log.info(
                "Training RROQ158 (Riemannian 1.58-bit) quantizer "
                "(K=%d, group_size=%d, seed=%d) ...",
                effective_k, gs, int(cfg.rroq158_seed),
            )
            t_rroq = time.time()
            rroq158_payload = encode_rroq158(
                np.asarray(active_vectors, dtype=np.float32),
                Rroq158Config(
                    K=effective_k,
                    group_size=gs,
                    seed=int(cfg.rroq158_seed),
                    fit_sample_cap=max(100_000, effective_k),
                ),
            )
            np.savez(
                index_dir / "rroq158_meta.npz",
                centroids=rroq158_payload.centroids,
                fwht_seed=np.array(rroq158_payload.fwht_seed, dtype=np.int64),
                dim=np.array(rroq158_payload.dim, dtype=np.int32),
                group_size=np.array(rroq158_payload.group_size, dtype=np.int32),
                k_requested=np.array(int(cfg.rroq158_k), dtype=np.int32),
                k_effective=np.array(effective_k, dtype=np.int32),
            )
            log.info(
                "RROQ158 encoding done in %.1fs (%d tokens, K=%d)",
                time.time() - t_rroq, n_tokens, effective_k,
            )
    if cfg.compression == Compression.RROQ4_RIEM:
        from voyager_index._internal.inference.quantization.rroq4_riem import (
            Rroq4RiemConfig,
            choose_effective_rroq4_riem_k,
            encode_rroq4_riem,
        )
        active_arr = np.asarray(active_vectors)
        n_tokens = int(active_arr.shape[0])
        token_dim = int(active_arr.shape[1]) if active_arr.ndim >= 2 else 0
        gs = int(cfg.rroq4_riem_group_size)
        if token_dim and token_dim < gs:
            log.warning(
                "RROQ4_RIEM requested but token dim=%d is smaller than "
                "group_size=%d. Falling back to FP16 — rroq4_riem needs at "
                "least group_size coordinates per token.",
                token_dim, gs,
            )
            cfg.compression = Compression.FP16
        elif n_tokens < gs:
            log.warning(
                "RROQ4_RIEM requested but corpus has only %d tokens "
                "(< group_size=%d). Falling back to FP16 — rroq4_riem needs "
                "at least one 4-bit group of tokens to encode.",
                n_tokens, gs,
            )
            cfg.compression = Compression.FP16
        else:
            effective_k = choose_effective_rroq4_riem_k(
                n_tokens=n_tokens,
                requested_k=int(cfg.rroq4_riem_k),
                group_size=gs,
            )
            log.info(
                "Training RROQ4_RIEM (Riemannian 4-bit asymmetric) quantizer "
                "(K=%d, group_size=%d, seed=%d) ...",
                effective_k, gs, int(cfg.rroq4_riem_seed),
            )
            t_r4r = time.time()
            rroq4_riem_payload = encode_rroq4_riem(
                np.asarray(active_vectors, dtype=np.float32),
                Rroq4RiemConfig(
                    K=effective_k,
                    group_size=gs,
                    seed=int(cfg.rroq4_riem_seed),
                    fit_sample_cap=max(100_000, effective_k),
                ),
            )
            np.savez(
                index_dir / "rroq4_riem_meta.npz",
                centroids=rroq4_riem_payload.centroids,
                fwht_seed=np.array(rroq4_riem_payload.fwht_seed, dtype=np.int64),
                dim=np.array(rroq4_riem_payload.dim, dtype=np.int32),
                group_size=np.array(rroq4_riem_payload.group_size, dtype=np.int32),
                k_requested=np.array(int(cfg.rroq4_riem_k), dtype=np.int32),
                k_effective=np.array(effective_k, dtype=np.int32),
            )
            log.info(
                "RROQ4_RIEM encoding done in %.1fs (%d tokens, K=%d)",
                time.time() - t_r4r, n_tokens, effective_k,
            )
    if cfg.compression == Compression.ROQ4:
        try:
            from voyager_index._internal.inference.quantization.rotational import (
                RotationalQuantizer, RoQConfig,
            )
            log.info("Training ROQ 4-bit quantizer ...")
            roq_quantizer = RotationalQuantizer(RoQConfig(dim=dim, num_bits=4, seed=cfg.seed))
            roq_doc_codes = []
            roq_doc_meta = []
            t_roq = time.time()
            for i, (s, e) in enumerate(active_offsets):
                vecs = np.asarray(active_vectors[s:e], dtype=np.float32)
                q = roq_quantizer.quantize(vecs, store=False)
                roq_doc_codes.append(np.asarray(q["codes"], dtype=np.uint8))
                roq_doc_meta.append(roq_quantizer.build_triton_meta(q, include_norm_sq=True))
                if (i + 1) % 2000 == 0:
                    log.info("  ROQ encoded %d/%d docs (%.1fs)", i + 1, len(active_offsets), time.time() - t_roq)
            log.info("ROQ encoding done in %.1fs", time.time() - t_roq)

            # Save quantizer for query-time use
            import pickle
            with open(index_dir / "roq_quantizer.pkl", "wb") as f:
                pickle.dump(roq_quantizer, f)
        except ImportError:
            log.warning("ROQ quantizer not available, falling back to FP16 storage for ROQ4")
            cfg.compression = Compression.FP16

    # Build shard store
    t0 = time.time()
    store = ShardStore(index_dir)
    store.build(
        all_vectors=np.asarray(active_vectors),
        doc_offsets=active_offsets,
        doc_ids=doc_ids,
        shard_assignments=shard_assignments,
        n_shards=cfg.n_shards,
        dim=dim,
        compression=cfg.compression,
        centroid_to_shard=centroid_to_shard,
        uniform_shard_tokens=cfg.uniform_shard_tokens,
        roq_doc_codes=roq_doc_codes,
        roq_doc_meta=roq_doc_meta,
        rroq158_payload=rroq158_payload,
        rroq4_riem_payload=rroq4_riem_payload,
    )
    build_s = time.time() - t0
    log.info("Shard store built in %.1fs", build_s)

    # Save router (LEMUR already saved itself via fit_initial; centroid needs explicit save)
    if cfg.router_type == RouterType.CENTROID:
        router.save(index_dir / "router")
        log.info("Router saved to %s", index_dir / "router")

    build_meta = {
        "corpus_size": cfg.corpus_size,
        "dim": dim,
        "n_centroids": cfg.n_centroids,
        "n_shards": cfg.n_shards,
        "compression": cfg.compression.value,
        "layout": cfg.layout.value,
        "router_type": cfg.router_type.value,
        "pooling_enabled": cfg.pooling.enabled,
        "pool_factor": cfg.pooling.pool_factor,
        "train_time_s": train_s,
        "build_time_s": build_s,
        "total_tokens": int(np.asarray(active_vectors).shape[0]),
        "avg_tokens_per_doc": float(np.mean([e - s for s, e in active_offsets])),
    }
    if cfg.compression == Compression.RROQ158:
        build_meta["rroq158_k"] = int(cfg.rroq158_k)
        build_meta["rroq158_seed"] = int(cfg.rroq158_seed)
        build_meta["rroq158_group_size"] = int(cfg.rroq158_group_size)
    elif cfg.compression == Compression.RROQ4_RIEM:
        build_meta["rroq4_riem_k"] = int(cfg.rroq4_riem_k)
        build_meta["rroq4_riem_seed"] = int(cfg.rroq4_riem_seed)
        build_meta["rroq4_riem_group_size"] = int(cfg.rroq4_riem_group_size)
    with open(index_dir / "build_meta.json", "w") as f:
        json.dump(build_meta, f, indent=2)

    gc.collect()
    log.info("Build complete. Index at %s, RSS=%.1f GB", index_dir, _mem_gb())
    return index_dir

