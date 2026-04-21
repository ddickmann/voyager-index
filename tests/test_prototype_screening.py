from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
import numpy as np

from colsearch._internal.inference.engines.colpali import ColPaliConfig, ColPaliEngine
from colsearch._internal.inference.index_core.centroid_screening import CentroidScreeningIndex
from colsearch._internal.inference.index_core.prototype_screening import PrototypeScreeningIndex
from colsearch._internal.server.main import create_app


def _three_doc_embeddings() -> np.ndarray:
    return np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.05, 0.95, 0.0], [0.0, 0.1, 0.9, 0.0]],
        ],
        dtype=np.float32,
    )


def test_prototype_screening_index_prefers_matching_document():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = PrototypeScreeningIndex(Path(tmpdir) / "screen", dim=4, on_disk=False)
        doc_ids = ["doc-a", "doc-b"]
        embeddings = np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
        index.rebuild(doc_ids=doc_ids, embeddings=embeddings, max_prototypes=3)

        candidates = index.search(
            np.asarray([[1.0, 0.0, 0.0, 0.0], [0.92, 0.08, 0.0, 0.0]], dtype=np.float32),
            top_k=1,
            candidate_budget=1,
        )

        assert candidates == ["doc-a"]
        stats = index.get_statistics()
        assert stats["prototype_count"] >= 2


def test_colpali_engine_screen_candidates_matches_exact_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ColPaliEngine(
            Path(tmpdir) / "colpali",
            config=ColPaliConfig(
                embed_dim=4,
                device="cpu",
                use_quantization=False,
                use_prototype_screening=True,
                prototype_doc_prototypes=3,
                prototype_query_prototypes=3,
            ),
            device="cpu",
            load_if_exists=False,
        )
        docs = np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
        engine.index(embeddings=docs, doc_ids=[101, 202], lengths=[3, 3])

        query = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.92, 0.08, 0.0, 0.0]], dtype=np.float32)
        screened = engine.screen_candidates(query, top_k=1, candidate_budget=1)
        exact = engine.search(query_embedding=query, top_k=1, candidate_ids=screened)

        assert screened == [101]
        assert exact[0].doc_id == 101
        assert engine.last_screening_profile["candidate_count"] == 1


def test_centroid_screening_index_prefers_matching_document_and_reloads():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "centroid-screen"
        doc_ids = ["doc-a", "doc-b", "doc-c"]
        embeddings = _three_doc_embeddings()
        query = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.92, 0.08, 0.0, 0.0]], dtype=np.float32)
        doc_c_query = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.0, 0.08, 0.92, 0.0]], dtype=np.float32)

        index = CentroidScreeningIndex(root, dim=4, default_doc_prototypes=3, device="cpu", load_if_exists=False)
        index.build(doc_ids=doc_ids[:2], embeddings=embeddings[:2])
        index.append(doc_ids=[doc_ids[2]], embeddings=embeddings[2:])
        assert index.delete(["doc-b"]) == 1

        assert index.search(query, top_k=1, candidate_budget=1) == ["doc-a"]
        assert index.search(doc_c_query, top_k=1, candidate_budget=2)[0] == "doc-c"
        assert index.get_statistics()["inactive_doc_count"] == 1

        reloaded = CentroidScreeningIndex(root, dim=4, default_doc_prototypes=3, device="cpu", load_if_exists=True)
        assert reloaded.search(query, top_k=1, candidate_budget=1) == ["doc-a"]
        assert reloaded.search(doc_c_query, top_k=1, candidate_budget=2)[0] == "doc-c"
        assert reloaded.get_statistics()["doc_count"] == 2


def test_prototype_screening_index_append_delete_and_reloads():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "prototype-screen"
        embeddings = _three_doc_embeddings()
        index = PrototypeScreeningIndex(root, dim=4, load_if_exists=False)
        index.rebuild(doc_ids=["doc-a", "doc-b"], embeddings=embeddings[:2], max_prototypes=3)
        index.append(doc_ids=["doc-c"], embeddings=embeddings[2:], max_prototypes=3)
        assert index.delete(["doc-b"]) == 1
        index.close()
        reloaded = PrototypeScreeningIndex(root, dim=4, load_if_exists=True)

        query = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.0, 0.08, 0.92, 0.0]], dtype=np.float32)
        assert reloaded.search(query, top_k=1, candidate_budget=2)[0] == "doc-c"
        assert reloaded.get_statistics()["doc_count"] == 2


def test_colpali_engine_centroid_backend_screen_candidates_matches_exact_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ColPaliEngine(
            Path(tmpdir) / "colpali",
            config=ColPaliConfig(
                embed_dim=4,
                device="cpu",
                use_quantization=False,
                use_prototype_screening=True,
                prototype_doc_prototypes=3,
                prototype_query_prototypes=3,
                screening_backend="centroid",
            ),
            device="cpu",
            load_if_exists=False,
        )
        docs = np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
        engine.index(embeddings=docs, doc_ids=[101, 202], lengths=[3, 3])

        query = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.92, 0.08, 0.0, 0.0]], dtype=np.float32)
        screened = engine.screen_candidates(query, top_k=1, candidate_budget=1)
        exact = engine.search(query_embedding=query, top_k=1, candidate_ids=screened)

        assert screened == [101]
        assert exact[0].doc_id == 101
        assert engine.last_screening_profile["screening_backend"] == "centroid"


def test_colpali_engine_reopen_preserves_persisted_centroid_backend():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "colpali"
        docs = np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
        query = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.92, 0.08, 0.0, 0.0]], dtype=np.float32)

        engine = ColPaliEngine(
            root,
            config=ColPaliConfig(
                embed_dim=4,
                device="cpu",
                use_quantization=False,
                use_prototype_screening=True,
                prototype_doc_prototypes=3,
                prototype_query_prototypes=3,
                screening_backend="centroid",
            ),
            device="cpu",
            load_if_exists=False,
        )
        engine.index(embeddings=docs, doc_ids=[101, 202], lengths=[3, 3])
        engine.close()

        reopened = ColPaliEngine(
            root,
            config=ColPaliConfig(
                embed_dim=999,
                device="cpu",
                use_quantization=True,
                use_prototype_screening=True,
            ),
            device="cpu",
            load_if_exists=True,
        )
        screened = reopened.screen_candidates(query, top_k=1, candidate_budget=1)

        assert reopened.config.embed_dim == 4
        assert reopened.config.use_quantization is False
        assert reopened.config.screening_backend == "centroid"
        assert isinstance(reopened.screening_index, CentroidScreeningIndex)
        assert reopened.get_statistics()["screening_state"]["health"] == "healthy"
        assert screened == [101]


def test_colpali_engine_delta_sidecar_updates_and_direct_gather():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "colpali"
        docs = _three_doc_embeddings()
        engine = ColPaliEngine(
            root,
            config=ColPaliConfig(
                embed_dim=4,
                device="cpu",
                use_quantization=False,
                use_prototype_screening=True,
                prototype_doc_prototypes=3,
                prototype_query_prototypes=3,
            ),
            device="cpu",
            load_if_exists=False,
        )
        engine.index(embeddings=docs[:2], doc_ids=[101, 202], lengths=[3, 3])
        assert engine.get_statistics()["screening_state"]["health"] == "healthy"

        engine.add_documents(embeddings=docs[2:], doc_ids=[303], lengths=[3])
        query = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.0, 0.08, 0.92, 0.0]], dtype=np.float32)
        screened = engine.screen_candidates(query, top_k=1, candidate_budget=2)
        assert screened is not None
        assert 303 in screened

        exact = engine.search(query_embedding=query, top_k=1, candidate_ids=screened)
        assert exact[0].doc_id == 303
        assert engine.last_search_profile["direct_gather"] is True

        engine.delete_documents([202])
        assert 202 not in engine.get_document_embeddings([202])
        assert engine.get_statistics()["num_documents"] == 2


def test_colpali_engine_risky_query_bypasses_sidecar():
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = _three_doc_embeddings()
        engine = ColPaliEngine(
            Path(tmpdir) / "colpali",
            config=ColPaliConfig(
                embed_dim=4,
                device="cpu",
                use_quantization=False,
                use_prototype_screening=True,
                prototype_doc_prototypes=3,
                prototype_query_prototypes=3,
            ),
            device="cpu",
            load_if_exists=False,
        )
        engine.index(embeddings=docs[:2], doc_ids=[101, 202], lengths=[3, 3])

        risky_query = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (100, 1))
        screened = engine.screen_candidates(risky_query, top_k=1, candidate_budget=2)

        assert screened is None
        assert engine.last_screening_profile["reason"] == "risky_query_token_count"


def test_multimodal_optimized_search_uses_prototype_screening():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(index_path=tmpdir)
        with TestClient(app) as client:
            create = client.post(
                "/collections/mm",
                json={"dimension": 4, "kind": "multimodal"},
            )
            assert create.status_code == 200

            points = [
                {
                    "id": "doc-a",
                    "vectors": [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                    "payload": {"text": "alpha"},
                },
                {
                    "id": "doc-b",
                    "vectors": [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
                    "payload": {"text": "beta"},
                },
            ]
            ingest = client.post("/collections/mm/points", json={"points": points})
            assert ingest.status_code == 200

            search = client.post(
                "/collections/mm/search",
                json={
                    "vectors": [[1.0, 0.0, 0.0, 0.0], [0.92, 0.08, 0.0, 0.0]],
                    "top_k": 1,
                    "strategy": "optimized",
                },
            )
            assert search.status_code == 200
            assert search.json()["results"][0]["id"] == "doc-a"

            service = app.state.search_service
            profile = service.collections["mm"].engine.last_search_profile
            assert profile["screening"]["candidate_count"] >= 1
            assert profile["direct_gather"] is True


def test_multimodal_optimized_search_falls_back_to_exact_when_sidecar_degraded():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(index_path=tmpdir)
        with TestClient(app) as client:
            create = client.post(
                "/collections/mm",
                json={"dimension": 4, "kind": "multimodal"},
            )
            assert create.status_code == 200

            points = [
                {
                    "id": "doc-a",
                    "vectors": [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                    "payload": {"text": "alpha"},
                },
                {
                    "id": "doc-b",
                    "vectors": [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
                    "payload": {"text": "beta"},
                },
            ]
            ingest = client.post("/collections/mm/points", json={"points": points})
            assert ingest.status_code == 200

            service = app.state.search_service
            engine = service.collections["mm"].engine
            engine._set_screening_state(health="degraded", reason="forced_test")

            search = client.post(
                "/collections/mm/search",
                json={
                    "vectors": [[1.0, 0.0, 0.0, 0.0], [0.92, 0.08, 0.0, 0.0]],
                    "top_k": 1,
                    "strategy": "optimized",
                },
            )
            assert search.status_code == 200
            assert search.json()["results"][0]["id"] == "doc-a"

            profile = service.collections["mm"].engine.last_search_profile
            assert profile["screening"]["reason"] == "health_degraded"
            assert profile["direct_gather"] is False


def test_multimodal_filtered_optimized_search_falls_back_to_exact_subset():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(index_path=tmpdir)
        with TestClient(app) as client:
            create = client.post(
                "/collections/mm",
                json={"dimension": 4, "kind": "multimodal"},
            )
            assert create.status_code == 200

            points = [
                {
                    "id": "doc-b",
                    "vectors": [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0], [0.0, 0.9, 0.1, 0.0]],
                    "payload": {"text": "beta", "label": "keep"},
                }
            ]
            for idx in range(69):
                basis = [1.0, 0.0, 0.0, 0.0] if idx % 2 == 0 else [0.0, 0.0, 1.0, 0.0]
                points.append(
                    {
                        "id": f"doc-{idx}",
                        "vectors": [
                            basis,
                            [basis[0] * 0.95, basis[1], basis[2] * 0.95, 0.05 if basis[0] else 0.0],
                            [basis[0] * 0.9, basis[1], basis[2] * 0.9, 0.1 if basis[0] else 0.0],
                        ],
                        "payload": {"text": f"filler-{idx}", "label": "keep"},
                    }
                )
            points.append(
                {
                    "id": "doc-drop",
                    "vectors": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.05, 0.95, 0.0], [0.0, 0.1, 0.9, 0.0]],
                    "payload": {"text": "gamma", "label": "drop"},
                }
            )
            ingest = client.post("/collections/mm/points", json={"points": points})
            assert ingest.status_code == 200

            service = app.state.search_service
            engine = service.collections["mm"].engine
            engine._set_screening_state(health="degraded", reason="forced_test")

            search = client.post(
                "/collections/mm/search",
                json={
                    "vectors": [[0.0, 1.0, 0.0, 0.0], [0.0, 0.92, 0.08, 0.0]],
                    "top_k": 1,
                    "strategy": "optimized",
                    "filter": {"label": "keep"},
                },
            )
            assert search.status_code == 200
            assert search.json()["results"][0]["id"] == "doc-b"

            profile = service.collections["mm"].engine.last_search_profile
            assert profile["screening"]["reason"] == "health_degraded"
            assert profile["requested_candidate_count"] == 70
            assert profile["direct_gather"] is True


def _many_doc_embeddings(n_docs: int, tokens: int = 8, dim: int = 32, seed: int = 42) -> np.ndarray:
    """Generate n_docs with distinct embeddings for GEM-lite testing."""
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n_docs, tokens, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-8
    return embs / norms


def test_gem_lite_codebook_is_built_for_sufficient_corpus():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = PrototypeScreeningIndex(Path(tmpdir) / "screen", dim=32, on_disk=False)
        n_docs = 10
        embs = _many_doc_embeddings(n_docs, tokens=8, dim=32)
        doc_ids = [f"doc-{i}" for i in range(n_docs)]
        index.rebuild(doc_ids=doc_ids, embeddings=embs, max_prototypes=4)

        stats = index.get_statistics()
        assert stats["gem_lite_enabled"] is True
        assert stats["gem_fine_centroids"] > 0
        assert stats["gem_coarse_clusters"] >= 2
        assert stats["gem_docs_with_profiles"] == n_docs


def test_gem_lite_codebook_not_built_for_tiny_corpus():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = PrototypeScreeningIndex(Path(tmpdir) / "screen", dim=4, on_disk=False)
        embs = np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.95, 0.05, 0.0]],
            ],
            dtype=np.float32,
        )
        index.rebuild(doc_ids=["a", "b"], embeddings=embs, max_prototypes=2)
        assert index.get_statistics()["gem_lite_enabled"] is False


def test_gem_lite_search_uses_cluster_overlap_and_qch():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = PrototypeScreeningIndex(Path(tmpdir) / "screen", dim=32, on_disk=False)
        n_docs = 12
        embs = _many_doc_embeddings(n_docs, tokens=8, dim=32)
        doc_ids = [f"doc-{i}" for i in range(n_docs)]
        index.rebuild(doc_ids=doc_ids, embeddings=embs, max_prototypes=4)

        assert index.gem_enabled

        query = embs[0, :4]
        candidates = index.search(query, top_k=3, candidate_budget=5)
        assert len(candidates) > 0
        assert "doc-0" in candidates

        profile = index.last_search_profile
        assert profile["gem_active"] is True
        assert "n_fine_centroids" in profile
        assert "n_coarse_clusters" in profile
        assert "query_ctop" in profile


def test_gem_lite_persists_and_reloads():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "screen"
        n_docs = 10
        embs = _many_doc_embeddings(n_docs, tokens=8, dim=32)
        doc_ids = [f"doc-{i}" for i in range(n_docs)]

        index = PrototypeScreeningIndex(root, dim=32, on_disk=False)
        index.rebuild(doc_ids=doc_ids, embeddings=embs, max_prototypes=4)
        assert index.gem_enabled
        original_stats = index.get_statistics()
        index.close()

        reloaded = PrototypeScreeningIndex(root, dim=32, on_disk=False, load_if_exists=True)
        assert reloaded.gem_enabled
        reloaded_stats = reloaded.get_statistics()
        assert reloaded_stats["gem_fine_centroids"] == original_stats["gem_fine_centroids"]
        assert reloaded_stats["gem_coarse_clusters"] == original_stats["gem_coarse_clusters"]
        assert reloaded_stats["gem_docs_with_profiles"] == n_docs

        query = embs[0, :4]
        candidates = reloaded.search(query, top_k=3, candidate_budget=5)
        assert len(candidates) > 0
        assert reloaded.last_search_profile["gem_active"] is True


def test_gem_lite_colpali_engine_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        n_docs = 12
        dim = 32
        embs = _many_doc_embeddings(n_docs, tokens=8, dim=dim)

        engine = ColPaliEngine(
            Path(tmpdir) / "colpali",
            config=ColPaliConfig(
                embed_dim=dim,
                device="cpu",
                use_quantization=False,
                use_prototype_screening=True,
                prototype_doc_prototypes=4,
                prototype_query_prototypes=4,
            ),
            device="cpu",
            load_if_exists=False,
        )
        engine.index(
            embeddings=embs,
            doc_ids=list(range(n_docs)),
            lengths=[8] * n_docs,
        )

        assert engine.screening_index is not None
        stats = engine.get_statistics()
        screening = stats.get("screening", {})
        assert screening.get("gem_lite_enabled") is True

        query = embs[0, :4]
        screened = engine.screen_candidates(query, top_k=3, candidate_budget=5)
        if screened is not None:
            assert len(screened) > 0
            sidecar_profile = engine.last_screening_profile.get("sidecar_profile", {})
            assert sidecar_profile.get("gem_active") is True


def test_gem_lite_fallback_when_codebook_unavailable():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = PrototypeScreeningIndex(Path(tmpdir) / "screen", dim=32, on_disk=False)
        n_docs = 12
        embs = _many_doc_embeddings(n_docs, tokens=8, dim=32)
        doc_ids = [f"doc-{i}" for i in range(n_docs)]
        index.rebuild(doc_ids=doc_ids, embeddings=embs, max_prototypes=4)

        index._gem_cquant = None
        index._gem_centroid_dists = None

        query = embs[0, :4]
        candidates = index.search(query, top_k=3, candidate_budget=5)
        assert len(candidates) > 0
        assert index.last_search_profile["gem_active"] is False
