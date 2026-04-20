from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from voyager_index._internal.inference.shard_engine.manager import (
    ShardEngineConfig,
    ShardSegmentManager,
)


def _make_manager(tmp_path: Path) -> ShardSegmentManager:
    return ShardSegmentManager(tmp_path / "shard", config=ShardEngineConfig(dim=4), device="cpu")


def test_score_sealed_candidates_prefers_colbandit_pipeline(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    rust_index = Mock()
    try:
        mgr._pipeline = object()
        mgr._rust_index = rust_index
        mgr._gpu_corpus = object()

        scfg = mgr._config.to_search_config()
        scfg.use_colbandit = True

        def fake_pipeline_fetch(*args, **kwargs):
            assert kwargs["exact_path"] == "colbandit_pipeline_fetch"
            assert kwargs["use_colbandit"] is True
            return [(7, 0.9)], {"num_docs_scored": 1}

        mgr._score_pipeline_fetch = fake_pipeline_fetch  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [7],
            {0: [7]},
            1,
            scfg,
            torch.device("cuda"),
        )
    finally:
        mgr.close()

    assert results == [(7, 0.9)]
    assert exact_ids == [7]
    assert exact_path == "colbandit_pipeline_fetch"
    assert stats["num_docs_scored"] == 1
    rust_index.score_candidates_exact.assert_not_called()


def test_score_sealed_candidates_prefers_quantized_pipeline(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    rust_index = Mock()
    try:
        mgr._pipeline = object()
        mgr._rust_index = rust_index

        scfg = mgr._config.to_search_config()
        scfg.quantization_mode = "fp8"

        def fake_pipeline_fetch(*args, **kwargs):
            assert kwargs["exact_path"] == "pipeline_quantized"
            assert kwargs["use_colbandit"] is False
            return [(11, 0.7)], {"num_docs_scored": 1}

        mgr._score_pipeline_fetch = fake_pipeline_fetch  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [11],
            {0: [11]},
            1,
            scfg,
            torch.device("cuda"),
        )
    finally:
        mgr.close()

    assert results == [(11, 0.7)]
    assert exact_ids == [11]
    assert exact_path == "pipeline_quantized"
    assert stats["num_docs_scored"] == 1
    rust_index.score_candidates_exact.assert_not_called()


def test_score_sealed_candidates_prefers_roq4_pipeline(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    rust_index = Mock()
    try:
        mgr._rust_index = rust_index

        scfg = mgr._config.to_search_config()
        scfg.quantization_mode = "roq4"

        def fake_roq_score(*args, **kwargs):
            return ([(21, 0.95)], {"num_docs_scored": 1})

        mgr._score_roq4_candidates = fake_roq_score  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [21],
            {0: [21]},
            1,
            scfg,
            torch.device("cuda"),
        )
    finally:
        mgr.close()

    assert results == [(21, 0.95)]
    assert exact_ids == [21]
    assert exact_path == "roq4_pipeline"
    assert stats["num_docs_scored"] == 1
    rust_index.score_candidates_exact.assert_not_called()


def test_score_sealed_candidates_prefers_rroq158_pipeline(tmp_path: Path) -> None:
    """When ``quantization_mode='rroq158'`` is set, the rroq158 lane wins
    over the fp16 / colbandit / quantized fall-throughs even on CPU
    (Phase 1.5 gate flipped CPU default to rroq158)."""
    mgr = _make_manager(tmp_path)
    try:
        scfg = mgr._config.to_search_config()
        scfg.quantization_mode = "rroq158"

        def fake_rroq_score(*args, **kwargs):
            return ([(31, 0.88)], {"num_docs_scored": 1})

        mgr._score_rroq158_candidates = fake_rroq_score  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [31],
            {0: [31]},
            1,
            scfg,
            torch.device("cpu"),
        )
    finally:
        mgr.close()

    assert results == [(31, 0.88)]
    assert exact_ids == [31]
    assert exact_path == "rroq158_pipeline"
    assert stats["num_docs_scored"] == 1


def test_score_sealed_candidates_rroq158_hardfails_when_no_kernel(
    tmp_path: Path, monkeypatch
) -> None:
    """When the index was built with rroq158 (rroq158_meta exists) but neither
    the GPU Triton kernel nor the Rust SIMD CPU kernel is reachable, the
    fallback chain must raise an actionable error rather than silently
    returning no results."""
    import sys

    mgr = _make_manager(tmp_path)
    try:
        scfg = mgr._config.to_search_config()
        scfg.quantization_mode = "rroq158"

        mgr._score_rroq158_candidates = lambda *a, **kw: None  # type: ignore[method-assign]
        # Fake the meta presence (cache in mgr) so the hard-fail branch fires.
        mgr._rroq158_meta = {"centroids": np.zeros((4, 4)), "fwht_seed": 0}

        # Hide the Rust kernel from the search.py importer.
        monkeypatch.setitem(sys.modules, "latence_shard_engine", None)

        with __import__("pytest").raises(RuntimeError, match="rroq158 shards selected"):
            mgr._score_sealed_candidates(
                torch.zeros((2, 4), dtype=torch.float32),
                [42],
                {0: [42]},
                1,
                scfg,
                torch.device("cpu"),
            )
    finally:
        mgr.close()


def test_score_sealed_candidates_auto_derives_rroq158_when_meta_present(
    tmp_path: Path,
) -> None:
    """When the storage codec is rroq158 (meta on disk / cached) and the
    caller did NOT set ``quantization_mode``, the search path must
    auto-route to the rroq158 lane. This is what makes the new
    ``Compression.RROQ158`` default work end-to-end without forcing every
    caller to also touch ``SearchConfig.quantization_mode``."""
    mgr = _make_manager(tmp_path)
    try:
        scfg = mgr._config.to_search_config()
        assert scfg.quantization_mode == ""

        mgr._rroq158_meta = {"centroids": np.zeros((4, 4)), "fwht_seed": 0}

        captured = {}

        def fake_rroq_score(*args, **kwargs):
            captured["called"] = True
            return ([(31, 0.88)], {"num_docs_scored": 1})

        mgr._score_rroq158_candidates = fake_rroq_score  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [31],
            {0: [31]},
            1,
            scfg,
            torch.device("cpu"),
        )
    finally:
        mgr.close()

    assert captured.get("called") is True
    assert exact_path == "rroq158_pipeline"
    assert results == [(31, 0.88)]
    assert exact_ids == [31]


def test_score_sealed_candidates_prefers_rroq4_riem_pipeline(tmp_path: Path) -> None:
    """When ``quantization_mode='rroq4_riem'`` is set, the rroq4_riem lane
    wins over the fp16 / colbandit / quantized fall-throughs even on CPU.
    Mirrors the rroq158 dispatch test — both codecs share the same
    "explicit override wins" policy.
    """
    mgr = _make_manager(tmp_path)
    try:
        scfg = mgr._config.to_search_config()
        scfg.quantization_mode = "rroq4_riem"

        def fake_rroq_score(*args, **kwargs):
            return ([(42, 0.91)], {"num_docs_scored": 1})

        mgr._score_rroq4_riem_candidates = fake_rroq_score  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [42],
            {0: [42]},
            1,
            scfg,
            torch.device("cpu"),
        )
    finally:
        mgr.close()

    assert results == [(42, 0.91)]
    assert exact_ids == [42]
    assert exact_path == "rroq4_riem_pipeline"
    assert stats["num_docs_scored"] == 1


def test_score_sealed_candidates_rroq4_riem_hardfails_when_no_kernel(
    tmp_path: Path, monkeypatch
) -> None:
    """Same loud-fail policy as rroq158: when shards are encoded with
    rroq4_riem but neither the GPU nor the Rust SIMD kernel is reachable,
    the dispatch must raise an actionable error instead of silently
    returning empty results.
    """
    import sys

    mgr = _make_manager(tmp_path)
    try:
        scfg = mgr._config.to_search_config()
        scfg.quantization_mode = "rroq4_riem"

        mgr._score_rroq4_riem_candidates = lambda *a, **kw: None  # type: ignore[method-assign]
        mgr._rroq4_riem_meta = {"centroids": np.zeros((4, 4)), "fwht_seed": 0}

        monkeypatch.setitem(sys.modules, "latence_shard_engine", None)

        with __import__("pytest").raises(RuntimeError, match="rroq4_riem shards selected"):
            mgr._score_sealed_candidates(
                torch.zeros((2, 4), dtype=torch.float32),
                [42],
                {0: [42]},
                1,
                scfg,
                torch.device("cpu"),
            )
    finally:
        mgr.close()


def test_score_sealed_candidates_auto_derives_rroq4_riem_when_meta_present(
    tmp_path: Path,
) -> None:
    """When the storage codec is rroq4_riem (meta on disk / cached) and
    the caller did NOT set ``quantization_mode``, the search path must
    auto-route to the rroq4_riem lane.
    """
    mgr = _make_manager(tmp_path)
    try:
        scfg = mgr._config.to_search_config()
        assert scfg.quantization_mode == ""

        mgr._rroq4_riem_meta = {"centroids": np.zeros((4, 4)), "fwht_seed": 0}

        captured = {}

        def fake_rroq_score(*args, **kwargs):
            captured["called"] = True
            return ([(42, 0.91)], {"num_docs_scored": 1})

        mgr._score_rroq4_riem_candidates = fake_rroq_score  # type: ignore[method-assign]

        results, exact_ids, exact_path, stats = mgr._score_sealed_candidates(
            torch.zeros((2, 4), dtype=torch.float32),
            [42],
            {0: [42]},
            1,
            scfg,
            torch.device("cpu"),
        )
    finally:
        mgr.close()

    assert captured.get("called") is True
    assert exact_path == "rroq4_riem_pipeline"
    assert results == [(42, 0.91)]
    assert exact_ids == [42]


def test_default_compression_is_rroq158() -> None:
    """The library default for newly constructed configs must be RROQ158
    after the Phase 7 production-validation sweep verdict.

    The full BEIR 2026-Q2 sweep applied the F1 default-promotion rule
    (avg ΔNDCG@10 ≥ -0.5 pt and per-cell GPU/CPU p95 ≤ fp16): rroq4_riem
    failed both the latency conditions decisively (~2-3x slower on GPU,
    ~10x on CPU on the BEIR 6-dataset cells), so the default reverts to
    rroq158 (avg -1 pt NDCG@10, flat R@100, ~5.5x smaller than fp16).
    rroq4_riem stays available as the opt-in no-quality-loss lane.
    Existing fp16/rroq158/rroq4_riem indexes on disk are unaffected
    because the manifest carries the build-time codec."""
    from voyager_index._internal.inference.shard_engine._manager.common import (
        ShardEngineConfig as _SEC,
    )
    from voyager_index._internal.inference.shard_engine.serving_config import (
        BuildConfig,
        Compression,
    )

    assert BuildConfig().compression == Compression.RROQ158
    assert _SEC().compression == Compression.RROQ158
    assert BuildConfig().rroq158_k == 8192
    assert BuildConfig().rroq4_riem_k == 8192


def test_inspect_query_pipeline_accepts_runtime_override_kwargs(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    try:
        captured: dict[str, object] = {}
        mgr._is_built = True
        mgr._doc_ids = [1]
        mgr._ensure_warmup = lambda: None  # type: ignore[assignment]
        mgr._resolve_scoring_device = lambda: torch.device("cpu")  # type: ignore[assignment]
        mgr._route_prefetch_cap = lambda scfg: 0  # type: ignore[assignment]
        mgr._router = Mock()
        mgr._router.route.return_value = SimpleNamespace(doc_ids=[1], by_shard={0: [1]})

        def fake_apply_search_overrides(scfg, **kwargs):
            captured.update(kwargs)
            return scfg

        mgr._apply_search_overrides = fake_apply_search_overrides  # type: ignore[method-assign]
        mgr._prune_routed_candidates = lambda q, routed, scfg, dev: ([1], {0: [1]}, "none")  # type: ignore[assignment]
        mgr._score_sealed_candidates = lambda q, candidate_ids, docs_by_shard, internal_k, scfg, dev: (  # type: ignore[assignment]
            [(1, 0.9)],
            [1],
            "pipeline_fetch",
            mgr._empty_exact_stage_stats("pipeline_fetch"),
        )

        trace = mgr.inspect_query_pipeline(
            np.zeros((2, 4), dtype=np.float32),
            k=1,
            quantization_mode="fp8",
            max_docs_exact=8,
            n_full_scores=16,
        )
    finally:
        mgr.close()

    assert captured["quantization_mode"] == "fp8"
    assert captured["max_docs_exact"] == 8
    assert captured["n_full_scores"] == 16
    assert trace["result_ids"] == [1]
    assert trace["exact_path"] == "pipeline_fetch"
