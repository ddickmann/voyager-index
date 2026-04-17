from __future__ import annotations

import hashlib
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voyager_index._internal.server.api.groundedness import (
    SupportUnitInput,
    count_text_tokens,
    encode_texts,
    score_groundedness,
    score_groundedness_chunked,
    segment_text,
    tokenize_text,
)
from voyager_index._internal.server.api.models import CreateCollectionRequest, GroundednessRequest, PointVector
from voyager_index._internal.server.api.service import SearchService, ValidationError
from voyager_index._internal.server.main import create_app

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class DummyGroundednessProvider:
    """Deterministic multi-vector text encoder used for groundedness tests.

    It intentionally collapses a few token pairs into the same semantic bucket
    so the validation matrix can show where an embedding-style scorer starts to
    break on role/entity swaps.
    """

    model_name = "dummy-groundedness"

    def __init__(self, dim: int = 24):
        self.dim = dim
        self._semantic_aliases = {
            "supports": "relation_support",
            "support": "relation_support",
            "backs": "relation_support",
            "refutes": "relation_support",
            "paris": "capital_city",
            "london": "capital_city",
        }

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text)

    def _canonical(self, token: str) -> str:
        return self._semantic_aliases.get(token.lower(), token.lower())

    def _token_vector(self, token: str) -> np.ndarray:
        canonical = self._canonical(token)
        digest = hashlib.sha1(canonical.encode("utf-8")).digest()
        vec = np.zeros((self.dim,), dtype=np.float32)
        first = digest[0] % self.dim
        second = digest[1] % self.dim
        vec[first] = 1.0
        vec[second] = max(vec[second], 0.5)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def encode(self, inputs, **_kwargs):
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        encoded = []
        for text in texts:
            tokens = self.tokenize(text) or ["<empty>"]
            encoded.append(np.stack([self._token_vector(token) for token in tokens]).astype(np.float32))
        return encoded


def _create_client(index_path: Path) -> TestClient:
    return TestClient(create_app(index_path=str(index_path)))


def _encode_text(provider: DummyGroundednessProvider, text: str) -> list[list[float]]:
    return provider.encode([text])[0].tolist()


def _patch_provider(provider: DummyGroundednessProvider):
    return patch.object(
        SearchService,
        "_get_groundedness_provider",
        new=lambda self, runtime, model_name=None: provider,
    )


def _assert_heatmap_shape(payload: dict) -> None:
    assert {"collection", "mode", "scores", "response_tokens", "support_units", "top_evidence", "eligibility", "time_ms"} <= set(
        payload
    )
    assert payload["scores"]["primary_name"] in {"reverse_context", "triangular"}
    assert payload["response_tokens"]
    assert payload["support_units"]
    for token in payload["response_tokens"]:
        assert {"index", "token", "weight", "reverse_context", "heatmap_score"} <= set(token)
    for unit in payload["support_units"]:
        assert unit["token_count"] == len(unit["tokens"]) == len(unit["token_scores"])


def test_groundedness_chunk_id_mode_late_interaction_is_heatmap_ready(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider()

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            assert client.post("/collections/li", json={"dimension": provider.dim, "kind": "late_interaction"}).status_code == 200
            points = [
                {
                    "id": "support-a",
                    "vectors": _encode_text(provider, "alpha supports claim"),
                    "payload": {"text": "alpha supports claim", "source": "doc-a"},
                },
                {
                    "id": "support-b",
                    "vectors": _encode_text(provider, "beta unrelated note"),
                    "payload": {"text": "beta unrelated note", "source": "doc-b"},
                },
            ]
            assert client.post("/collections/li/points", json={"points": points}).status_code == 200

            grounded = client.post(
                "/collections/li/groundedness",
                json={
                    "chunk_ids": ["support-a", "support-b"],
                    "response_text": "alpha supports claim",
                    "query_text": "who supports claim",
                    "debug_dense_matrices": True,
                    "evidence_limit": 3,
                },
            )
            ungrounded = client.post(
                "/collections/li/groundedness",
                json={
                    "chunk_ids": ["support-a", "support-b"],
                    "response_text": "gamma drifts elsewhere",
                    "query_text": "who supports claim",
                },
            )

    assert grounded.status_code == 200
    assert ungrounded.status_code == 200
    grounded_payload = grounded.json()
    ungrounded_payload = ungrounded.json()
    _assert_heatmap_shape(grounded_payload)
    assert grounded_payload["mode"] == "chunk_ids"
    assert grounded_payload["eligibility"]["collection_kind"] == "late_interaction"
    assert grounded_payload["eligibility"]["user_facing_supported"] is True
    assert grounded_payload["debug"]["response_to_support"]
    assert grounded_payload["query_tokens"]
    assert len(grounded_payload["top_evidence"]) <= 3
    assert grounded_payload["scores"]["reverse_context"] > ungrounded_payload["scores"]["reverse_context"]


def test_groundedness_raw_context_mode_defaults_to_packed_windows(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider()

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            assert client.post("/collections/li", json={"dimension": provider.dim, "kind": "late_interaction"}).status_code == 200

            response = client.post(
                "/collections/li/groundedness",
                json={
                    "raw_context": "alpha supports claim. beta unrelated note.",
                    "response_text": "alpha supports claim",
                    "query_text": "who supports claim",
                },
            )

    assert response.status_code == 200
    payload = response.json()
    _assert_heatmap_shape(payload)
    assert payload["mode"] == "raw_context"
    assert all(unit["source_mode"] == "raw_context" for unit in payload["support_units"])
    assert payload["eligibility"]["vector_source"] == "encoded_raw_context"
    assert len(payload["support_units"]) == 1
    assert payload["support_units"][0]["text"] == "alpha supports claim. beta unrelated note."


def test_groundedness_raw_context_chunk_budget_is_user_configurable(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider()

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            assert client.post("/collections/li", json={"dimension": provider.dim, "kind": "late_interaction"}).status_code == 200

            response = client.post(
                "/collections/li/groundedness",
                json={
                    "raw_context": "alpha supports claim. beta supports note. gamma supports proof.",
                    "response_text": "gamma supports proof",
                    "query_text": "who supports proof",
                    "raw_context_chunk_tokens": 4,
                },
            )

    assert response.status_code == 200
    payload = response.json()
    _assert_heatmap_shape(payload)
    assert len(payload["support_units"]) == 3
    assert [unit["text"] for unit in payload["support_units"]] == [
        "alpha supports claim.",
        "beta supports note.",
        "gamma supports proof.",
    ]
    assert [unit["offset_start"] for unit in payload["support_units"]] == [0, 22, 42]


def test_groundedness_packed_window_merge_matches_direct_reference() -> None:
    provider = DummyGroundednessProvider()
    raw_context = "alpha supports claim. beta unrelated note. gamma supports proof."
    response_text = "gamma supports proof"

    segments = segment_text(
        raw_context,
        "sentence_packed",
        provider=provider,
        chunk_token_budget=4,
    )
    assert len(segments) == 3

    support_embeddings = encode_texts(provider, [segment["text"] for segment in segments], is_query=False)
    support_units = [
        SupportUnitInput(
            support_id=f"raw-{idx}",
            text=segment["text"],
            embeddings=embedding,
            tokens=tokenize_text(provider, segment["text"], expected_len=int(embedding.shape[0])),
            source_mode="raw_context",
            offset_start=int(segment["offset_start"]),
            offset_end=int(segment["offset_end"]),
        )
        for idx, (segment, embedding) in enumerate(zip(segments, support_embeddings))
    ]

    response_embeddings = encode_texts(provider, [response_text], is_query=False)[0]
    response_tokens = tokenize_text(provider, response_text, expected_len=int(response_embeddings.shape[0]))

    merged = score_groundedness_chunked(
        support_batches=[[unit] for unit in support_units],
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        evidence_limit=4,
        primary_metric="reverse_context",
    )
    reference = score_groundedness(
        support_units=support_units,
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        evidence_limit=4,
        primary_metric="reverse_context",
    )

    assert merged["scores"]["reverse_context"] == pytest.approx(reference["scores"]["reverse_context"], abs=1e-6)
    assert merged["scores"]["primary_score"] == pytest.approx(reference["scores"]["primary_score"], abs=1e-6)
    assert len(merged["response_tokens"]) == len(reference["response_tokens"])
    for merged_row, reference_row in zip(merged["response_tokens"], reference["response_tokens"]):
        assert merged_row["reverse_context"] == pytest.approx(reference_row["reverse_context"], abs=1e-6)
        assert merged_row["support_unit_index"] == reference_row["support_unit_index"]
        assert merged_row["support_token_index"] == reference_row["support_token_index"]
        assert merged_row["support_token"] == reference_row["support_token"]

    assert [
        (
            evidence["response_token_index"],
            evidence["support_unit_index"],
            evidence["support_token_index"],
        )
        for evidence in merged["top_evidence"]
    ] == [
        (
            evidence["response_token_index"],
            evidence["support_unit_index"],
            evidence["support_token_index"],
        )
        for evidence in reference["top_evidence"]
    ]


def test_groundedness_token_helpers_ignore_mapping_tokenize_outputs() -> None:
    class MappingTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False):
            token_count = len(_TOKEN_RE.findall(text))
            return {"input_ids": list(range(token_count))}

        def convert_ids_to_tokens(self, input_ids):
            return [f"tok_{idx}" for idx, _value in enumerate(input_ids)]

    class MappingTokenProvider(DummyGroundednessProvider):
        def __init__(self):
            super().__init__()
            self.tokenizer = MappingTokenizer()

        def tokenize(self, text: str):
            token_count = len(_TOKEN_RE.findall(text))
            return {
                "input_ids": [[idx for idx in range(token_count)]],
                "attention_mask": [[1] * token_count],
            }

    provider = MappingTokenProvider()
    text = "alpha supports claim. beta unrelated note."

    assert count_text_tokens(provider, text) == len(_TOKEN_RE.findall(text))
    assert tokenize_text(provider, text) == [f"tok_{idx}" for idx in range(len(_TOKEN_RE.findall(text)))]


def test_groundedness_warns_when_packed_budget_exceeds_encoder_limit(tmp_path: Path) -> None:
    class LimitedTokenizer:
        model_max_length = 4

        def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False):
            return {"input_ids": list(range(len(_TOKEN_RE.findall(text))))}

        def convert_ids_to_tokens(self, input_ids):
            return [f"tok_{idx}" for idx, _value in enumerate(input_ids)]

    class LimitedBudgetProvider(DummyGroundednessProvider):
        def __init__(self):
            super().__init__()
            self.tokenizer = LimitedTokenizer()

    provider = LimitedBudgetProvider()

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            assert client.post("/collections/li", json={"dimension": provider.dim, "kind": "late_interaction"}).status_code == 200

            response = client.post(
                "/collections/li/groundedness",
                json={
                    "raw_context": "alpha supports claim. beta supports note.",
                    "response_text": "alpha supports claim",
                    "query_text": "who supports claim",
                    "raw_context_chunk_tokens": 8,
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert any("exceeds the groundedness encoder token limit" in warning for warning in payload["warnings"])


def test_groundedness_validation_matrix_exposes_breakpoints(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider()

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            assert client.post("/collections/li", json={"dimension": provider.dim, "kind": "late_interaction"}).status_code == 200

            cases = {
                "grounded": {
                    "raw_context": "paris supports summit plan. beta unrelated note.",
                    "response_text": "paris supports summit plan",
                    "query_text": "who supports the summit plan",
                },
                "unsupported": {
                    "raw_context": "paris supports summit plan. beta unrelated note.",
                    "response_text": "omega invents a new theorem",
                    "query_text": "who supports the summit plan",
                },
                "role_swap_breakpoint": {
                    "raw_context": "paris supports summit plan.",
                    "response_text": "paris refutes summit plan",
                    "query_text": "who supports the summit plan",
                },
                "entity_swap_breakpoint": {
                    "raw_context": "paris supports summit plan.",
                    "response_text": "london supports summit plan",
                    "query_text": "who supports the summit plan",
                },
            }

            scores = {}
            for name, payload in cases.items():
                response = client.post("/collections/li/groundedness", json=payload)
                assert response.status_code == 200
                scores[name] = response.json()["scores"]["reverse_context"]

    assert scores["grounded"] > scores["unsupported"]
    assert scores["role_swap_breakpoint"] > scores["unsupported"]
    assert scores["entity_swap_breakpoint"] > scores["unsupported"]
    assert scores["entity_swap_breakpoint"] >= scores["grounded"] - 0.15


def test_groundedness_multimodal_quantized_storage_materializes_vectors(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider()

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            assert client.post("/collections/mm", json={"dimension": provider.dim, "kind": "multimodal"}).status_code == 200
            runtime = client.app.state.search_service.get_collection("mm")
            runtime.engine.config.use_quantization = True
            points = [
                {
                    "id": "page-1",
                    "vectors": _encode_text(provider, "alpha supports claim"),
                    "payload": {"text": "alpha supports claim", "page_number": 1},
                }
            ]
            assert client.post("/collections/mm/points", json={"points": points}).status_code == 200

            response = client.post(
                "/collections/mm/groundedness",
                json={
                    "chunk_ids": ["page-1"],
                    "response_text": "alpha supports claim",
                    "query_text": "who supports claim",
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["eligibility"]["collection_kind"] == "multimodal"
    assert payload["eligibility"]["storage_compression"] == "int8"
    assert payload["eligibility"]["dequantized"] is True
    assert any("quantized storage" in warning for warning in payload["warnings"])


def test_groundedness_shard_int8_fetch_path_is_supported(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider(dim=8)

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            assert (
                client.post(
                    "/collections/shard-int8",
                    json={"dimension": provider.dim, "kind": "shard", "n_shards": 1, "compression": "int8"},
                ).status_code
                == 200
            )
            points = [
                {
                    "id": "doc-1",
                    "vectors": _encode_text(provider, "alpha supports claim"),
                    "payload": {"text": "alpha supports claim"},
                }
            ]
            assert client.post("/collections/shard-int8/points", json={"points": points}).status_code == 200

            response = client.post(
                "/collections/shard-int8/groundedness",
                json={
                    "chunk_ids": ["doc-1"],
                    "response_text": "alpha supports claim",
                    "query_text": "who supports claim",
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["eligibility"]["collection_kind"] == "shard"
    assert payload["eligibility"]["storage_compression"] == "int8"
    assert payload["eligibility"]["dequantized"] is True


def test_groundedness_rejects_roq4_without_fp16_sidecar(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider(dim=8)
    service = SearchService(str(tmp_path))
    service.create_collection("roq", CreateCollectionRequest(dimension=provider.dim, kind="shard", n_shards=1))
    service.add_points(
        "roq",
        [
            PointVector(
                id="doc-1",
                vectors=_encode_text(provider, "alpha supports claim"),
                payload={"text": "alpha supports claim"},
            )
        ],
    )
    runtime = service.get_collection("roq")
    runtime.meta["compression"] = "roq4"

    with patch.object(service, "_roq4_sidecar_available", return_value=False), patch.object(
        service,
        "_get_groundedness_provider",
        return_value=provider,
    ):
        with pytest.raises(ValidationError, match="unsupported"):
            service.groundedness(
                "roq",
                GroundednessRequest(
                    chunk_ids=["doc-1"],
                    response_text="alpha supports claim",
                    query_text="who supports claim",
                ),
            )
    service.close()
