"""CLI entrypoint for offline shard builds."""
from __future__ import annotations

import argparse
from pathlib import Path

from .corpus import DEFAULT_NPZ
from .pipeline import build
from ..config import AnnBackend, BuildConfig, Compression, RouterType, StorageLayout

def main() -> None:
    parser = argparse.ArgumentParser(description="Build shard index")
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--n-centroids", type=int, default=1024)
    parser.add_argument("--n-shards", type=int, default=256)
    parser.add_argument(
        "--compression",
        choices=["fp16", "int8", "roq4", "rroq158", "rroq4_riem"],
        default="rroq158",
        help="Compression codec for shard storage. Default 'rroq158' "
             "(Riemannian-aware 1.58-bit ROQ): ~5.5x smaller than fp16, "
             "strictly faster on GPU and CPU. Pass 'rroq4_riem' for the "
             "no-degradation safe fallback (~3x smaller than fp16, ≤0.5%% "
             "NDCG@10 drop). Pass 'fp16' to opt out of quantization entirely.",
    )
    parser.add_argument(
        "--rroq158-k", type=int, default=8192,
        help="Centroid codebook size for the rroq158 codec. Power of two, "
             ">= group_size. Default 8192.",
    )
    parser.add_argument(
        "--rroq158-seed", type=int, default=42,
        help="Seed for FWHT rotation + spherical k-means init in rroq158.",
    )
    parser.add_argument(
        "--rroq158-group-size", type=int, default=32,
        help="Ternary group size in coords for rroq158 (multiple of 32).",
    )
    parser.add_argument(
        "--rroq4-riem-k", type=int, default=8192,
        help="Centroid codebook size for the rroq4_riem codec. Power of two, "
             ">= group_size. Default 8192.",
    )
    parser.add_argument(
        "--rroq4-riem-seed", type=int, default=42,
        help="Seed for FWHT rotation + spherical k-means init in rroq4_riem.",
    )
    parser.add_argument(
        "--rroq4-riem-group-size", type=int, default=32,
        help="4-bit asymmetric group size in coords for rroq4_riem "
             "(positive even integer; default 32).",
    )
    parser.add_argument("--layout", choices=[x.value for x in StorageLayout], default=StorageLayout.PROXY_GROUPED.value)
    parser.add_argument("--router", choices=[x.value for x in RouterType], default=RouterType.LEMUR.value)
    parser.add_argument("--enable-pooling", action="store_true")
    parser.add_argument("--pool-factor", type=int, default=2)
    parser.add_argument("--lemur-epochs", type=int, default=10)
    parser.add_argument("--ann-backend", choices=[x.value for x in AnnBackend], default=AnnBackend.FAISS_HNSW_IP.value)
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = BuildConfig(
        corpus_size=args.corpus_size,
        n_centroids=args.n_centroids,
        n_shards=args.n_shards,
        compression=Compression(args.compression),
        layout=StorageLayout(args.layout),
        router_type=RouterType(args.router),
        rroq158_k=int(args.rroq158_k),
        rroq158_seed=int(args.rroq158_seed),
        rroq158_group_size=int(args.rroq158_group_size),
        rroq4_riem_k=int(args.rroq4_riem_k),
        rroq4_riem_seed=int(args.rroq4_riem_seed),
        rroq4_riem_group_size=int(args.rroq4_riem_group_size),
    )
    cfg.pooling.enabled = bool(args.enable_pooling)
    cfg.pooling.pool_factor = int(args.pool_factor)
    cfg.lemur.enabled = cfg.router_type == RouterType.LEMUR
    cfg.lemur.epochs = int(args.lemur_epochs)
    cfg.lemur.ann_backend = AnnBackend(args.ann_backend)
    build(cfg, npz_path=Path(args.npz), device=args.device)

