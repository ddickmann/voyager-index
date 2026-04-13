"""Add centroid codes to an existing shard index (post-hoc augmentation).

Reads all embeddings, clusters token embeddings via K-means, assigns
per-token centroid codes, and rewrites each shard file with the new
'centroid_codes' tensor alongside 'embeddings'.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors import safe_open
from safetensors.numpy import save_file as st_save_np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX = "/root/.cache/shard-bench/index_100000_fp16_proxy_grouped_lemur_uniform"
INDEX_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_INDEX)
SHARD_DIR = INDEX_DIR / "shards"
N_CENTROIDS = 1024
MAX_TRAIN_TOKENS = 500_000


def main():
    t0 = time.perf_counter()

    shard_files = sorted(SHARD_DIR.glob("shard_*.safetensors"))
    logger.info("Found %d shard files in %s", len(shard_files), SHARD_DIR)

    logger.info("Pass 1: sampling tokens for K-means...")
    all_sample_vecs = []
    sampled_so_far = 0
    total_tokens = 0
    for sf in shard_files:
        with safe_open(str(sf), framework="numpy") as f:
            emb = f.get_tensor("embeddings")
            n = emb.shape[0]
            total_tokens += n
            if sampled_so_far >= MAX_TRAIN_TOKENS:
                continue
            need = MAX_TRAIN_TOKENS - sampled_so_far
            if n <= need:
                all_sample_vecs.append(emb.astype(np.float32))
                sampled_so_far += n
            else:
                idx = np.random.RandomState(42).choice(n, need, replace=False)
                all_sample_vecs.append(emb[idx].astype(np.float32))
                sampled_so_far += need

    train_data = np.concatenate(all_sample_vecs, axis=0)
    dim = train_data.shape[1]
    logger.info("Training K-means: %d tokens (sampled from %d total), dim=%d, K=%d",
                len(train_data), total_tokens, dim, N_CENTROIDS)

    kmeans = faiss.Kmeans(dim, N_CENTROIDS, niter=20, verbose=True,
                          gpu=torch.cuda.is_available())
    kmeans.train(train_data)
    centroids = kmeans.centroids.copy()
    np.save(str(INDEX_DIR / "centroids.npy"), centroids)
    logger.info("Centroids saved to %s", INDEX_DIR / "centroids.npy")
    del train_data, all_sample_vecs

    assign_index = faiss.IndexFlatIP(dim)
    assign_index.add(centroids)

    logger.info("Pass 2: assigning centroid codes and rewriting shards...")
    for i, sf in enumerate(shard_files):
        with safe_open(str(sf), framework="numpy") as f:
            tensor_names = list(f.keys())
            tensors = {}
            for name in tensor_names:
                tensors[name] = f.get_tensor(name)

        emb = tensors["embeddings"]
        emb_f32 = emb.astype(np.float32)
        n_tokens = emb_f32.shape[0]

        _, I = assign_index.search(emb_f32, 1)
        codes = I[:, 0].astype(np.uint16)
        tensors["centroid_codes"] = codes

        st_save_np(tensors, str(sf))

        if (i + 1) % 50 == 0 or i == len(shard_files) - 1:
            logger.info("  Processed %d/%d shards (%d tokens in this shard)",
                        i + 1, len(shard_files), n_tokens)

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.1fs. Centroid codes added to all %d shards.", elapsed, len(shard_files))

    sf0 = shard_files[0]
    with safe_open(str(sf0), framework="numpy") as f:
        codes_check = f.get_tensor("centroid_codes")
        emb_check = f.get_tensor("embeddings")
    logger.info("Verification: shard_0 has embeddings %s and centroid_codes %s (dtype=%s)",
                emb_check.shape, codes_check.shape, codes_check.dtype)
    assert codes_check.dtype == np.uint16
    assert codes_check.shape[0] == emb_check.shape[0]
    logger.info("Verification passed!")


if __name__ == "__main__":
    main()
