# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colsearch._internal.inference.shard_engine.config import AnnBackend
from colsearch._internal.inference.shard_engine.manager import ShardEngineConfig, ShardSegmentManager
from colsearch._internal.server.main import create_app


def _make_corpus(
    n_docs: int = 12,
    dim: int = 16,
    min_tok: int = 4,
    max_tok: int = 8,
    seed: int = 42,
) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [rng.randn(rng.randint(min_tok, max_tok + 1), dim).astype(np.float32) for _ in range(n_docs)]


def build_report(workdir: Path) -> dict[str, object]:
    shard_path = workdir / "shard"
    config = ShardEngineConfig(
        n_shards=3,
        dim=16,
        lemur_epochs=1,
        k_candidates=8,
        max_docs_exact=8,
        n_full_scores=8,
        ann_backend=AnnBackend.TORCH_EXACT_IP,
    )
    query = np.random.RandomState(7).randn(5, 16).astype(np.float32)

    manager = ShardSegmentManager(shard_path, config=config, device="cpu")
    try:
        manager.build(_make_corpus(dim=16), list(range(12)))
        trace_before = manager.inspect_query_pipeline(query, k=3)
        stats_before = manager.get_statistics()
    finally:
        manager.close()

    reopened = ShardSegmentManager(shard_path, config=config, device="cpu")
    try:
        trace_after = reopened.inspect_query_pipeline(query, k=3)
        stats_after = reopened.get_statistics()
    finally:
        reopened.close()

    manifest = json.loads((shard_path / "manifest.json").read_text(encoding="utf-8"))
    engine_meta = json.loads((shard_path / "engine_meta.json").read_text(encoding="utf-8"))
    router_state = json.loads((shard_path / "lemur" / "router_state.json").read_text(encoding="utf-8"))
    with np.load(shard_path / "doc_index.npz") as doc_index:
        doc_index_arrays = sorted(doc_index.files)

    with TestClient(create_app(index_path=str(workdir), version="1.0.0")) as client:
        health = client.get("/health")
        create_response = client.post(
            "/collections/shard-cap",
            json={"dimension": 4, "kind": "shard", "n_shards": 4},
        )
        collection_info = client.get("/collections/shard-cap/info")

    return {
        "artifacts": {
            "manifest_keys": sorted(manifest.keys()),
            "engine_meta_keys": sorted(engine_meta.keys()),
            "router_state_keys": sorted(router_state.keys()),
            "doc_index_arrays": doc_index_arrays,
        },
        "parity": {
            "trace_keys_equal_after_reopen": set(trace_before) == set(trace_after),
            "exact_path_equal_after_reopen": trace_before["exact_path"] == trace_after["exact_path"],
            "prune_path_equal_after_reopen": trace_before["prune_path"] == trace_after["prune_path"],
            "result_ids_equal_after_reopen": trace_before["result_ids"] == trace_after["result_ids"],
            "runtime_capability_keys": sorted(stats_before["runtime_capabilities"].keys()),
            "runtime_capabilities_stable": sorted(stats_before["runtime_capabilities"].keys())
            == sorted(stats_after["runtime_capabilities"].keys()),
        },
        "api": {
            "health_status_code": health.status_code,
            "health_has_runtime_capabilities": "runtime_capabilities" in health.json(),
            "create_collection_status_code": create_response.status_code,
            "collection_info_status_code": collection_info.status_code,
            "collection_runtime_capability_keys": sorted(
                (collection_info.json().get("runtime_capabilities") or {}).keys()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True, help="Path to write the parity report JSON.")
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="shard-refactor-parity-") as tmpdir:
        report = build_report(Path(tmpdir))

    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
