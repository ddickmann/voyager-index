from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from voyager_index._internal.inference.shard_engine import (
    BuildConfig,
    SearchConfig,
)
from voyager_index._internal.inference.shard_engine import (
    ShardSegmentManager as PackageShardSegmentManager,
)
from voyager_index._internal.inference.shard_engine.builder import _index_dir
from voyager_index._internal.inference.shard_engine.config import AnnBackend
from voyager_index._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)
from voyager_index._internal.inference.shard_engine.wal import WalOp, WalReader, WalWriter
from voyager_index._internal.server.main import create_app


def _make_corpus(
    n_docs: int = 12,
    dim: int = 16,
    min_tok: int = 4,
    max_tok: int = 8,
    seed: int = 42,
) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [rng.randn(rng.randint(min_tok, max_tok + 1), dim).astype(np.float32) for _ in range(n_docs)]


def _build_manager(path: Path, *, dim: int = 16) -> ShardSegmentManager:
    # 12-doc fixture is far too small for the new RROQ158 default (K=8192);
    # explicitly pin to FP16 so this contract test exercises the legacy lane.
    from voyager_index._internal.inference.shard_engine.config import Compression

    config = ShardEngineConfig(
        n_shards=3,
        dim=dim,
        lemur_epochs=1,
        k_candidates=8,
        max_docs_exact=8,
        n_full_scores=8,
        ann_backend=AnnBackend.TORCH_EXACT_IP,
        compression=Compression.FP16,
    )
    mgr = ShardSegmentManager(path, config=config, device="cpu")
    mgr.build(_make_corpus(dim=dim), list(range(12)))
    return mgr


def test_shard_export_surface_preserves_legacy_imports() -> None:
    assert PackageShardSegmentManager is ShardSegmentManager
    assert BuildConfig.__name__ == "BuildConfig"
    assert SearchConfig.__name__ == "SearchConfig"
    assert WalWriter.__name__ == "WalWriter"
    assert WalReader.__name__ == "WalReader"
    assert WalOp.__name__ == "WalOp"
    assert callable(_index_dir)


def test_shard_build_artifacts_capture_router_and_manifest_contracts(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path / "shard")
    try:
        manifest = json.loads((tmp_path / "shard" / "manifest.json").read_text(encoding="utf-8"))
        engine_meta = json.loads((tmp_path / "shard" / "engine_meta.json").read_text(encoding="utf-8"))
        router_state = json.loads((tmp_path / "shard" / "lemur" / "router_state.json").read_text(encoding="utf-8"))
        with np.load(tmp_path / "shard" / "doc_index.npz") as doc_index:
            doc_ids = doc_index["doc_ids"].tolist()
            shard_ids = doc_index["shard_ids"].tolist()
            local_starts = doc_index["local_starts"].tolist()
            local_ends = doc_index["local_ends"].tolist()
            row_indices = doc_index["row_indices"].tolist()
    finally:
        mgr.close()

    assert manifest["num_docs"] == 12
    assert manifest["num_shards"] == 3
    assert manifest["compression"] == "fp16"
    assert "global_target_len" in manifest

    assert engine_meta["dim"] == 16
    assert engine_meta["router_type"] == "lemur"
    assert engine_meta["n_shards"] == 3
    assert engine_meta["layout"] == "proxy_grouped"

    assert router_state["ann_backend"] == "torch_exact_ip"
    assert router_state["backend_name"] in {"official_lemur", "fallback_proxy"}
    assert router_state["live_docs"] == 12
    assert router_state["total_docs"] == 12

    assert len(doc_ids) == 12
    assert len(shard_ids) == 12
    assert len(local_starts) == 12
    assert len(local_ends) == 12
    assert len(row_indices) == 12


def test_query_trace_contract_survives_reload(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard"
    query = np.random.RandomState(7).randn(5, 16).astype(np.float32)

    mgr = _build_manager(shard_path)
    try:
        trace_before = mgr.inspect_query_pipeline(query, k=3)
    finally:
        mgr.close()

    reopened = ShardSegmentManager(
        shard_path, config=ShardEngineConfig(dim=16, ann_backend=AnnBackend.TORCH_EXACT_IP), device="cpu"
    )
    try:
        trace_after = reopened.inspect_query_pipeline(query, k=3)
    finally:
        reopened.close()

    expected_keys = {
        "route_ms",
        "prune_ms",
        "fetch_ms",
        "exact_ms",
        "exact_prepare_ms",
        "h2d_ms",
        "maxsim_ms",
        "topk_ms",
        "total_ms",
        "h2d_bytes",
        "num_shards_fetched",
        "num_docs_scored",
        "router_device",
        "exact_device",
        "prune_path",
        "exact_path",
        "routed_ids",
        "pruned_ids",
        "exact_candidate_ids",
        "result_ids",
    }

    assert expected_keys <= set(trace_before)
    assert set(trace_before) == set(trace_after)
    assert trace_before["exact_path"] == trace_after["exact_path"]
    assert trace_before["prune_path"] == trace_after["prune_path"]
    assert len(trace_after["result_ids"]) <= 3


def test_shard_stats_expose_runtime_capabilities(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path / "shard")
    try:
        stats = mgr.get_statistics()
    finally:
        mgr.close()

    caps = stats["runtime_capabilities"]
    assert "official_lemur_available" in caps
    assert "faiss_available" in caps
    assert "native_shard_engine_importable" in caps
    assert "router_backend" in caps
    assert "pinned_staging_mode" in caps
    assert "last_exact_path" in caps


def test_reference_api_health_and_collection_info_expose_runtime_capabilities(tmp_path: Path) -> None:
    with TestClient(create_app(index_path=str(tmp_path), version="1.2.3")) as client:
        health = client.get("/health")
        create_response = client.post(
            "/collections/shard-cap",
            json={"dimension": 4, "kind": "shard", "n_shards": 4},
        )
        info = client.get("/collections/shard-cap/info")

    assert health.status_code == 200
    assert "runtime_capabilities" in health.json()
    assert create_response.status_code == 200
    assert info.status_code == 200
    caps = info.json()["runtime_capabilities"]
    assert "router_backend" in caps
    assert "native_exact_available" in caps
    assert "pinned_staging_mode" in caps
