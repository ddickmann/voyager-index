from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from colsearch._internal.inference.engines.colpali import ColPaliConfig, ColPaliEngine


def _engine(path: Path, *, quantized: bool = False) -> ColPaliEngine:
    return ColPaliEngine(
        path,
        config=ColPaliConfig(embed_dim=4, device="cpu", use_quantization=quantized),
        device="cpu",
        load_if_exists=True,
    )


def test_colpali_chunked_storage_roundtrips_and_persists(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "colpali")
    docs = np.asarray(
        [
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 0, 0]],
        ],
        dtype=np.float32,
    )
    engine.add_documents(docs, doc_ids=["page-1", "page-2"])

    results = engine.search(query_embedding=np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32), top_k=1)
    vectors = engine.get_document_embeddings(["page-1"])
    stats = engine.get_statistics()
    engine.cleanup()

    reloaded = _engine(tmp_path / "colpali")
    persisted = reloaded.search(
        query_embedding=np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        top_k=1,
    )

    assert results[0].doc_id == "page-1"
    assert persisted[0].doc_id == "page-1"
    assert vectors["page-1"].tolist() == [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    assert stats["num_documents"] == 2
    assert (tmp_path / "colpali" / "manifest.json").exists()


def test_colpali_delete_and_compact_keep_active_documents(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "colpali")
    engine.add_documents(
        np.asarray(
            [
                [[1, 0, 0, 0], [1, 0, 0, 0]],
                [[0, 1, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [0, 0, 1, 0]],
            ],
            dtype=np.float32,
        ),
        doc_ids=["a", "b", "c"],
    )
    engine.delete_documents(["b"])
    engine.compact()

    stats = engine.get_statistics()
    embeddings = engine.get_document_embeddings(["a", "c"])
    results = engine.search(
        query_embedding=np.asarray([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.float32),
        top_k=3,
    )

    assert stats["num_documents"] == 2
    assert set(embeddings) == {"a", "c"}
    assert [item.doc_id for item in results] == ["c", "a"]


def test_colpali_search_honors_ragged_patch_lengths(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "colpali")
    docs = np.asarray(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 0, 0]],
        ],
        dtype=np.float32,
    )
    engine.index(
        embeddings=docs,
        doc_ids=["short", "full"],
        lengths=[1, 2],
    )

    results = engine.search(
        query_embedding=np.asarray([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        top_k=2,
    )

    assert [item.doc_id for item in results] == ["full", "short"]
    assert results[0].score > results[1].score


def test_colpali_search_across_multiple_chunks_returns_global_top_k(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "colpali")
    engine.add_documents(
        np.asarray(
            [
                [[1, 0, 0, 0], [1, 0, 0, 0]],
                [[0.8, 0.2, 0, 0], [0.8, 0.2, 0, 0]],
            ],
            dtype=np.float32,
        ),
        doc_ids=["a", "b"],
    )
    engine.add_documents(
        np.asarray(
            [
                [[0.6, 0.4, 0, 0], [0.6, 0.4, 0, 0]],
                [[0, 1, 0, 0], [0, 1, 0, 0]],
            ],
            dtype=np.float32,
        ),
        doc_ids=["c", "d"],
    )

    results = engine.search(
        query_embedding=np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        top_k=3,
    )

    assert [item.doc_id for item in results] == ["a", "b", "c"]


def test_colpali_rejects_batched_queries(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "colpali")
    engine.add_documents(
        np.asarray([[[1, 0, 0, 0], [1, 0, 0, 0]]], dtype=np.float32),
        doc_ids=["page-1"],
    )

    with pytest.raises(ValueError, match="exactly one query"):
        engine.search(
            query_embedding=np.asarray(
                [
                    [[1, 0, 0, 0], [1, 0, 0, 0]],
                    [[0, 1, 0, 0], [0, 1, 0, 0]],
                ],
                dtype=np.float32,
            ),
            top_k=1,
        )


def test_colpali_legacy_snapshot_is_rejected(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy"
    legacy_path.mkdir(parents=True, exist_ok=True)
    (legacy_path / "colpali_index.pt").write_bytes(b"legacy")

    try:
        _engine(legacy_path)
    except RuntimeError as exc:
        assert "Legacy ColPali `.pt` indexes" in str(exc)
    else:
        raise AssertionError("Legacy ColPali snapshots should be rejected")


def test_colpali_profiles_capture_write_and_search_timings(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "colpali", quantized=True)
    engine.add_documents(
        np.asarray([[[1, 0, 0, 0], [1, 0, 0, 0]]], dtype=np.float32),
        doc_ids=["page-1"],
    )

    results = engine.search(
        query_embedding=np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        top_k=1,
    )

    assert results[0].doc_id == "page-1"
    assert engine.last_write_profile["mode"] == "append_chunk"
    assert engine.last_write_profile["doc_count"] == 1
    assert engine.last_search_profile["mode"] == "colpali_search"
    assert engine.last_search_profile["doc_count"] == 1
    assert engine.last_search_profile["chunk_count"] == 1
    assert engine.last_search_profile["quantized_storage"] is True
    assert engine.last_search_profile["score_ms"] >= 0.0
