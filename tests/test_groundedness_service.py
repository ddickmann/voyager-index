from __future__ import annotations

import hashlib
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from voyager_index._internal.server.api.groundedness import (
    SupportUnitInput,
    calibrate_per_token_scores,
    collect_support_literal_set,
    compute_null_distribution,
    count_text_tokens,
    default_null_bank_texts,
    diff_literals,
    encode_texts,
    extract_literals,
    literal_guarded_score,
    partition_support_units,
    score_groundedness,
    score_groundedness_chunked,
    segment_text,
    support_content_mask,
    support_unit_signature,
    tokenize_text,
)
from voyager_index._internal.server.api.models import CreateCollectionRequest, GroundednessRequest, PointVector
from voyager_index._internal.server.api.service import SearchService, ValidationError
from voyager_index._internal.server.main import create_app
from voyager_index.multimodal import VllmFactoryModernColBERTProvider

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
    assert "consensus_hardened" in payload["scores"]
    assert payload["response_tokens"]
    assert payload["support_units"]
    for token in payload["response_tokens"]:
        assert {"index", "token", "weight", "reverse_context", "heatmap_score"} <= set(token)
        assert "consensus_hardened" in token
    for unit in payload["support_units"]:
        assert unit["token_count"] == len(unit["tokens"]) == len(unit["token_scores"])


def test_groundedness_request_defaults_to_256_chunk_budget() -> None:
    request = GroundednessRequest(raw_context="alpha supports claim", response_text="alpha supports claim")
    assert request.segmentation_mode.value == "sentence_packed"
    assert request.raw_context_chunk_tokens == 256


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


def test_groundedness_sentence_packing_carries_whole_sentence_forward() -> None:
    provider = DummyGroundednessProvider()
    text = "alpha supports claim. beta supports note. gamma supports proof."

    segments = segment_text(
        text,
        "sentence_packed",
        provider=provider,
        chunk_token_budget=7,
    )

    assert [segment["text"] for segment in segments] == [
        "alpha supports claim.",
        "beta supports note.",
        "gamma supports proof.",
    ]


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
        support_batches=partition_support_units(support_units, batch_size=2),
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
    assert merged["scores"]["consensus_hardened"] == pytest.approx(reference["scores"]["consensus_hardened"], abs=1e-6)
    assert len(merged["response_tokens"]) == len(reference["response_tokens"])
    for merged_row, reference_row in zip(merged["response_tokens"], reference["response_tokens"]):
        assert merged_row["reverse_context"] == pytest.approx(reference_row["reverse_context"], abs=1e-6)
        assert merged_row["consensus_hardened"] == pytest.approx(reference_row["consensus_hardened"], abs=1e-6)
        assert merged_row["support_unit_hits_above_threshold"] == reference_row["support_unit_hits_above_threshold"]
        assert merged_row["effective_support_units"] == pytest.approx(reference_row["effective_support_units"], abs=1e-6)
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


def test_vllm_factory_moderncolbert_provider_uses_plugin_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    requests_seen: list[dict] = []

    class DummyTokenizer:
        model_max_length = 8192

        def __call__(
            self,
            text: str,
            add_special_tokens: bool = True,
            truncation: bool = True,
            max_length: int | None = None,
            padding: bool = False,
            return_tensors=None,
        ):
            token_count = len(_TOKEN_RE.findall(text))
            return {"input_ids": [101, *range(200, 200 + token_count), 102]}

        def convert_ids_to_tokens(self, input_ids):
            return [f"tok_{token_id}" for token_id in input_ids]

    class DummyConfig:
        colbert_dim = 4
        query_length = 256
        document_length = 8192

    class DummyResponse:
        text = "ok"

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DummyClient:
        def get(self, path: str, timeout=None):
            assert path == "/health"
            return DummyResponse({"status": "ok"})

        def post(self, path: str, json=None):
            assert path == "/pooling"
            requests_seen.append(json)
            return DummyResponse({"data": [float(idx) for idx in range(8)]})

        def close(self):
            return None

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: DummyConfig())

    provider = VllmFactoryModernColBERTProvider(
        endpoint="http://localhost:8000",
        model="dummy-moderncolbert",
        batch_size=2,
        max_concurrency=2,
    )
    monkeypatch.setattr(provider, "_get_http_client", lambda: DummyClient())

    assert provider.healthcheck() == {"status": "ok"}
    embeddings = provider.encode(["alpha supports claim", "beta supports note"], is_query=False)

    assert len(embeddings) == 2
    assert all(embedding.shape == (2, 4) for embedding in embeddings)
    assert all(request["task"] == "plugin" for request in requests_seen)
    assert all(request["data"]["is_query"] is False for request in requests_seen)
    assert {request["data"]["text"] for request in requests_seen} == {
        "alpha supports claim",
        "beta supports note",
    }


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


# ----------------------------------------------------------------------
# Phase A: foundation - parity, support-side content masking, dedup
# ----------------------------------------------------------------------


def test_support_content_mask_blocks_filler_punctuation_and_stopwords() -> None:
    tokens = ["[CLS]", "Ġalpha", ".", "Ġthe", "Ġ26", "Ġsupports", "##s", "Ġa"]
    mask = support_content_mask(tokens).tolist()

    assert mask == [False, True, False, False, True, True, True, False]


def test_support_content_mask_handles_empty_input() -> None:
    mask = support_content_mask([])
    assert mask.shape == (0,)
    assert mask.dtype.is_floating_point is False


def test_score_groundedness_masks_support_filler_from_evidence() -> None:
    response_tokens = ["alpha"]
    response_embeddings = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    support_tokens = ["the", "alpha", "."]
    support_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.8, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    support_units = [
        SupportUnitInput(
            support_id="unit-0",
            chunk_id="unit-0",
            source_mode="chunk_ids",
            text="the alpha .",
            embeddings=support_embeddings,
            tokens=support_tokens,
        )
    ]

    result = score_groundedness(
        support_units=support_units,
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        evidence_limit=2,
    )

    response_row = result["response_tokens"][0]
    assert response_row["support_token"] == "alpha", "filler tokens should not be selected as evidence"
    assert response_row["reverse_context"] == pytest.approx(0.6, abs=1e-5)


def test_score_groundedness_falls_back_when_all_support_tokens_are_filler() -> None:
    response_tokens = ["alpha"]
    response_embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    support_units = [
        SupportUnitInput(
            support_id="unit-0",
            chunk_id="unit-0",
            source_mode="chunk_ids",
            text="the .",
            embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            tokens=["the", "."],
        )
    ]

    result = score_groundedness(
        support_units=support_units,
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        evidence_limit=1,
    )

    response_row = result["response_tokens"][0]
    assert response_row["reverse_context"] == pytest.approx(1.0, abs=1e-5), (
        "when no content support tokens exist the scorer must still report a finite score"
    )


def test_support_unit_signature_uses_chunk_id_when_present() -> None:
    unit_with_id = SupportUnitInput(
        support_id="s-1",
        chunk_id="doc-7",
        source_mode="chunk_ids",
        text="some text",
        embeddings=torch.zeros((1, 3)),
        tokens=["x"],
    )
    unit_text_only_a = SupportUnitInput(
        support_id="s-2",
        chunk_id=None,
        source_mode="raw_context",
        text="Hello WORLD",
        embeddings=torch.zeros((1, 3)),
        tokens=["x"],
    )
    unit_text_only_b = SupportUnitInput(
        support_id="s-3",
        chunk_id=None,
        source_mode="raw_context",
        text="hello   world",
        embeddings=torch.zeros((1, 3)),
        tokens=["x"],
    )

    assert support_unit_signature(unit_with_id) == "chunk:doc-7"
    assert support_unit_signature(unit_text_only_a) == support_unit_signature(unit_text_only_b)


def test_score_groundedness_dedups_duplicate_support_units_for_consensus() -> None:
    response_tokens = ["alpha"]
    response_embeddings = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    embedding = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    duplicates = [
        SupportUnitInput(
            support_id=f"dup-{idx}",
            chunk_id="doc-7",
            source_mode="chunk_ids",
            text="alpha",
            embeddings=embedding.clone(),
            tokens=["alpha"],
        )
        for idx in range(3)
    ]
    unique = [
        SupportUnitInput(
            support_id="dup-0",
            chunk_id="doc-7",
            source_mode="chunk_ids",
            text="alpha",
            embeddings=embedding.clone(),
            tokens=["alpha"],
        )
    ]

    duplicated = score_groundedness(
        support_units=duplicates,
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        evidence_limit=1,
    )
    deduped = score_groundedness(
        support_units=unique,
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        evidence_limit=1,
    )

    assert duplicated["response_tokens"][0]["support_unit_hits_above_threshold"] == 1
    assert duplicated["response_tokens"][0]["effective_support_units"] == pytest.approx(
        deduped["response_tokens"][0]["effective_support_units"], abs=1e-6
    )
    assert any("support_unit_dedup" in warning for warning in duplicated["warnings"])
    assert not any("support_unit_dedup" in warning for warning in deduped["warnings"])


def test_score_groundedness_chunked_dedups_across_batches() -> None:
    response_tokens = ["alpha"]
    response_embeddings = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    embedding = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    units = [
        SupportUnitInput(
            support_id=f"dup-{idx}",
            chunk_id="doc-7",
            source_mode="chunk_ids",
            text="alpha",
            embeddings=embedding.clone(),
            tokens=["alpha"],
        )
        for idx in range(4)
    ]

    merged = score_groundedness_chunked(
        support_batches=partition_support_units(units, batch_size=2),
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        evidence_limit=1,
    )

    assert merged["response_tokens"][0]["support_unit_hits_above_threshold"] == 1
    dedup_warnings = [warning for warning in merged["warnings"] if "support_unit_dedup" in warning]
    assert len(dedup_warnings) == 1, "chunked path must emit a single merged dedup warning"


def test_count_text_tokens_prefers_provider_encoded_token_count_hook() -> None:
    class HookProvider:
        model_name = "hook-provider"

        def encoded_token_count(self, text: str, *, is_query: bool = False) -> int:
            return 99 if not is_query else 11

        def tokenize(self, text: str, *, is_query: bool = False):
            return ["wrong"] * 3

    assert count_text_tokens(HookProvider(), "anything") == 99
    assert count_text_tokens(HookProvider(), "anything", is_query=True) == 11


def test_vllm_factory_encoded_token_count_matches_token_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTokenizer:
        model_max_length = 8192

        def __call__(
            self,
            text: str,
            add_special_tokens: bool = True,
            truncation: bool = True,
            max_length: int | None = None,
            padding: bool = False,
            return_tensors=None,
        ):
            token_count = len(_TOKEN_RE.findall(text))
            return {"input_ids": [101, *range(200, 200 + token_count), 102]}

        def convert_ids_to_tokens(self, input_ids):
            return [f"tok_{token_id}" for token_id in input_ids]

    class DummyConfig:
        colbert_dim = 4
        query_length = 256
        document_length = 8192

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: DummyConfig())

    provider = VllmFactoryModernColBERTProvider(
        endpoint="http://localhost:8000",
        model="dummy-moderncolbert",
    )

    text = "alpha supports claim"
    token_ids_count = len(provider._token_ids(text, is_query=False))
    assert provider.encoded_token_count(text) == token_ids_count
    assert provider.encoded_token_count(text, is_query=True) == len(provider._token_ids(text, is_query=True))


def test_vllm_factory_decode_embedding_rejects_malformed_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTokenizer:
        model_max_length = 8192

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

        def convert_ids_to_tokens(self, input_ids):
            return ["a"] * len(input_ids)

    class DummyConfig:
        colbert_dim = 4
        query_length = 256
        document_length = 8192

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: DummyConfig())

    provider = VllmFactoryModernColBERTProvider(
        endpoint="http://localhost:8000",
        model="dummy-moderncolbert",
    )

    with pytest.raises(ValueError, match="not divisible by colbert_dim"):
        provider._decode_embedding({"data": [0.1, 0.2, 0.3, 0.4, 0.5]})

    with pytest.raises(ValueError, match="inner dim"):
        provider._decode_embedding({"data": [[0.1, 0.2, 0.3]]})


def test_vllm_factory_decode_embedding_accepts_valid_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTokenizer:
        model_max_length = 8192

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2]}

        def convert_ids_to_tokens(self, input_ids):
            return ["a"] * len(input_ids)

    class DummyConfig:
        colbert_dim = 4
        query_length = 256
        document_length = 8192

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: DummyConfig())

    provider = VllmFactoryModernColBERTProvider(
        endpoint="http://localhost:8000",
        model="dummy-moderncolbert",
    )

    flat = provider._decode_embedding({"data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]})
    assert flat.shape == (2, 4)

    nested = provider._decode_embedding({"data": [[0.1, 0.2, 0.3, 0.4]]})
    assert nested.shape == (1, 4)


# ----------------------------------------------------------------------
# Phase B: per-token null-distribution calibration
# ----------------------------------------------------------------------


def test_default_null_bank_texts_returns_diverse_non_empty_corpus() -> None:
    bank = default_null_bank_texts()

    assert isinstance(bank, list)
    assert len(bank) >= 8
    assert all(isinstance(text, str) and text.strip() for text in bank)
    assert len(set(bank)) == len(bank)


def test_compute_null_distribution_returns_per_token_mean_std_and_size() -> None:
    response = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    bank = [
        torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    ]

    mean, std, size = compute_null_distribution(response, bank)

    assert size == 3
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert torch.all(std >= 0)


def test_compute_null_distribution_handles_empty_bank() -> None:
    response = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    mean, std, size = compute_null_distribution(response, [])

    assert size == 0
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert torch.all(std > 0)


def test_calibrate_per_token_scores_widens_dynamic_range() -> None:
    rc = torch.tensor([0.95, 0.30, 0.75], dtype=torch.float32)
    null_mean = torch.tensor([0.50, 0.50, 0.50], dtype=torch.float32)
    null_std = torch.tensor([0.10, 0.10, 0.10], dtype=torch.float32)

    z, prob = calibrate_per_token_scores(rc, null_mean, null_std, temperature=1.0)

    assert torch.allclose(z, (rc - null_mean) / null_std)
    assert torch.all((prob > 0) & (prob < 1))
    assert prob[0] > prob[2] > prob[1]


def test_score_groundedness_emits_calibrated_score_when_null_bank_provided() -> None:
    provider = DummyGroundednessProvider(dim=12)

    def _support_unit(idx: int, text: str) -> SupportUnitInput:
        embedding = provider.encode([text])[0]
        return SupportUnitInput(
            support_id=f"sup-{idx}",
            chunk_id=f"chunk-{idx}",
            source_mode="raw_context",
            text=text,
            embeddings=torch.as_tensor(embedding, dtype=torch.float32),
            tokens=provider.tokenize(text),
        )

    units = [_support_unit(0, "alpha supports claim")]
    response_text = "alpha supports claim"
    response_embedding = torch.as_tensor(provider.encode([response_text])[0], dtype=torch.float32)
    response_tokens = provider.tokenize(response_text)
    null_bank = [
        torch.as_tensor(provider.encode([text])[0], dtype=torch.float32)
        for text in ("orange marmalade tastes bitter", "the river bends north", "violins were tuned earlier")
    ]

    scored = score_groundedness(
        support_units=units,
        response_embeddings=response_embedding,
        response_tokens=response_tokens,
        null_bank_embeddings=null_bank,
    )

    assert scored["scores"]["null_bank_size"] == 3
    assert scored["scores"]["reverse_context_calibrated"] is not None
    for row in scored["response_tokens"]:
        assert row["reverse_context_calibrated"] is not None
        assert row["reverse_context_z"] is not None
        assert row["null_mean"] is not None
        assert row["null_std"] is not None
    calibrated = scored["scores"]["reverse_context_calibrated"]
    raw = scored["scores"]["reverse_context"]
    assert 0.0 < calibrated < 1.0
    assert raw > calibrated


def test_score_groundedness_warns_and_falls_back_when_no_null_bank() -> None:
    provider = DummyGroundednessProvider(dim=8)
    text = "alpha supports claim"
    embedding = torch.as_tensor(provider.encode([text])[0], dtype=torch.float32)
    units = [
        SupportUnitInput(
            support_id="sup-0",
            chunk_id="chunk-0",
            source_mode="raw_context",
            text=text,
            embeddings=embedding,
            tokens=provider.tokenize(text),
        )
    ]

    scored = score_groundedness(
        support_units=units,
        response_embeddings=embedding,
        response_tokens=provider.tokenize(text),
    )

    assert scored["scores"]["null_bank_size"] == 0
    assert scored["scores"]["reverse_context_calibrated"] is None
    assert any(w.startswith("calibration_disabled") for w in scored["warnings"])


def test_score_groundedness_chunked_propagates_calibration_to_first_batch_only() -> None:
    provider = DummyGroundednessProvider(dim=10)

    def _support_unit(idx: int, text: str) -> SupportUnitInput:
        embedding = torch.as_tensor(provider.encode([text])[0], dtype=torch.float32)
        return SupportUnitInput(
            support_id=f"sup-{idx}",
            chunk_id=f"chunk-{idx}",
            source_mode="raw_context",
            text=text,
            embeddings=embedding,
            tokens=provider.tokenize(text),
        )

    batches = [
        [_support_unit(0, "alpha supports claim")],
        [_support_unit(1, "beta refutes claim")],
    ]
    response_text = "alpha supports claim"
    response_embedding = torch.as_tensor(provider.encode([response_text])[0], dtype=torch.float32)
    response_tokens = provider.tokenize(response_text)
    null_bank = [
        torch.as_tensor(provider.encode([text])[0], dtype=torch.float32)
        for text in ("orange marmalade tastes bitter", "the river bends north")
    ]

    scored = score_groundedness_chunked(
        support_batches=batches,
        response_embeddings=response_embedding,
        response_tokens=response_tokens,
        null_bank_embeddings=null_bank,
    )

    assert scored["scores"]["null_bank_size"] == 2
    assert scored["scores"]["reverse_context_calibrated"] is not None
    calibration_warnings = [w for w in scored["warnings"] if w.startswith("calibration_disabled")]
    assert calibration_warnings == []
    for row in scored["response_tokens"]:
        assert row["reverse_context_calibrated"] is not None
        assert row["null_mean"] is not None


def test_service_groundedness_endpoint_returns_calibrated_score(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider(dim=8)

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            response = client.post(
                "/collections/groundedness-calib",
                json={"dimension": provider.dim, "kind": "shard", "n_shards": 1},
            )
            assert response.status_code == 200
            assert (
                client.post(
                    "/collections/groundedness-calib/points",
                    json={
                        "points": [
                            {
                                "id": "doc-1",
                                "vectors": _encode_text(provider, "alpha supports claim"),
                                "payload": {"text": "alpha supports claim"},
                            }
                        ]
                    },
                ).status_code
                == 200
            )

            scored = client.post(
                "/collections/groundedness-calib/groundedness",
                json={
                    "chunk_ids": ["doc-1"],
                    "response_text": "alpha supports claim",
                },
            )

    assert scored.status_code == 200
    payload = scored.json()
    assert payload["scores"]["null_bank_size"] >= 1
    assert payload["scores"]["reverse_context_calibrated"] is not None
    for row in payload["response_tokens"]:
        assert row["reverse_context_calibrated"] is not None


# ----------------------------------------------------------------------
# Phase C: narrow-scope literal extraction and guardrails
# ----------------------------------------------------------------------


def test_extract_literals_finds_dates_numbers_units_and_identifiers() -> None:
    text = (
        "The release date was 20 July 1981 and the price was $19.99. "
        "It weighs 5 kg and is sold under code GTE-3.5. "
        "Visit https://example.com/x for details, or email a@b.co."
    )
    literals = extract_literals(text)
    kinds = {literal["kind"] for literal in literals}
    values = {literal["value"] for literal in literals}

    assert "date" in kinds
    assert "currency" in kinds
    assert "measurement" in kinds
    assert "identifier" in kinds
    assert "url" in kinds
    assert "email" in kinds
    assert "20 July 1981" in values
    assert "$19.99" in values
    assert "5 kg" in values


def test_extract_literals_drops_year_when_full_date_overlaps() -> None:
    literals = extract_literals("Released on 20 July 1981 in the UK.")
    kinds = [literal["kind"] for literal in literals]

    assert "date" in kinds
    assert "year" not in kinds


def test_collect_support_literal_set_buckets_by_kind() -> None:
    units = [
        SupportUnitInput(
            support_id="s-0",
            chunk_id="c-0",
            source_mode="raw_context",
            text="Released on 20 July 1981 in the United States.",
            embeddings=torch.zeros((1, 4)),
            tokens=["x"],
        ),
        SupportUnitInput(
            support_id="s-1",
            chunk_id="c-1",
            source_mode="raw_context",
            text="The total cost was $19.99 plus tax.",
            embeddings=torch.zeros((1, 4)),
            tokens=["x"],
        ),
    ]
    bucket = collect_support_literal_set(units)
    assert "date" in bucket
    assert any("1981" in date for date in bucket["date"])
    assert "currency" in bucket


def test_diff_literals_flags_unsupported_response_literals() -> None:
    response = "Released on 22 July 1981 for $24.99."
    units = [
        SupportUnitInput(
            support_id="s-0",
            chunk_id="c-0",
            source_mode="raw_context",
            text="Released on 20 July 1981 for $19.99 in the UK.",
            embeddings=torch.zeros((1, 4)),
            tokens=["x"],
        )
    ]
    response_literals, mismatches, matches = diff_literals(response, units)

    mismatch_values = {literal["value"] for literal in mismatches}
    assert "22 July 1981" in mismatch_values
    assert "$24.99" in mismatch_values
    assert response_literals
    assert all(literal in response_literals for literal in matches + mismatches)


def test_diff_literals_treats_year_as_match_when_support_has_full_date() -> None:
    response = "Teardrops shipped in 1981."
    units = [
        SupportUnitInput(
            support_id="s-0",
            chunk_id="c-0",
            source_mode="raw_context",
            text="Teardrops was released on 20 July 1981.",
            embeddings=torch.zeros((1, 4)),
            tokens=["x"],
        )
    ]
    _, mismatches, matches = diff_literals(response, units)
    assert any(literal["value"] == "1981" for literal in matches)
    assert not any(literal["value"] == "1981" for literal in mismatches)


def test_literal_guarded_score_applies_multiplicative_penalty_with_floor() -> None:
    base = 0.9
    mismatches = [{"kind": "date", "value": "x", "normalized": "x", "start": 0, "end": 1}]
    guarded_one = literal_guarded_score(base, mismatches, rate=0.5, floor=0.0)
    guarded_two = literal_guarded_score(base, mismatches * 2, rate=0.5, floor=0.0)
    assert guarded_one == pytest.approx(0.45, rel=1e-6)
    assert guarded_two == pytest.approx(0.225, rel=1e-6)
    assert literal_guarded_score(base, [], rate=0.5) == pytest.approx(base)
    assert literal_guarded_score(base, mismatches * 10, rate=0.5, floor=0.05) >= 0.05


def test_score_groundedness_emits_literal_diagnostics_and_guarded_score() -> None:
    provider = DummyGroundednessProvider(dim=12)
    support_text = "Teardrops was released on 20 July 1981 for $19.99 in the United States."
    response_text = "Teardrops was released on 22 July 1981 for $24.99."

    support_embedding = torch.as_tensor(provider.encode([support_text])[0], dtype=torch.float32)
    units = [
        SupportUnitInput(
            support_id="sup-0",
            chunk_id="chunk-0",
            source_mode="raw_context",
            text=support_text,
            embeddings=support_embedding,
            tokens=provider.tokenize(support_text),
        )
    ]
    response_embedding = torch.as_tensor(provider.encode([response_text])[0], dtype=torch.float32)

    scored = score_groundedness(
        support_units=units,
        response_embeddings=response_embedding,
        response_tokens=provider.tokenize(response_text),
        response_text=response_text,
    )

    diagnostics = scored["literal_diagnostics"]
    mismatch_kinds = {item["kind"] for item in diagnostics["mismatches"]}
    assert {"date", "currency"} <= mismatch_kinds
    assert scored["scores"]["literal_mismatch_count"] >= 2
    assert scored["scores"]["literal_total_count"] == len(diagnostics["response_literals"])
    assert scored["scores"]["literal_guarded"] < scored["scores"]["reverse_context"]
    assert any(w.startswith("literal_mismatch") for w in scored["warnings"])


def test_score_groundedness_emits_no_mismatch_warning_when_response_is_clean() -> None:
    provider = DummyGroundednessProvider(dim=12)
    support_text = "Teardrops was released on 20 July 1981 for $19.99."
    response_text = "Released on 20 July 1981 for $19.99."

    support_embedding = torch.as_tensor(provider.encode([support_text])[0], dtype=torch.float32)
    units = [
        SupportUnitInput(
            support_id="sup-0",
            chunk_id="chunk-0",
            source_mode="raw_context",
            text=support_text,
            embeddings=support_embedding,
            tokens=provider.tokenize(support_text),
        )
    ]
    response_embedding = torch.as_tensor(provider.encode([response_text])[0], dtype=torch.float32)

    scored = score_groundedness(
        support_units=units,
        response_embeddings=response_embedding,
        response_tokens=provider.tokenize(response_text),
        response_text=response_text,
    )

    assert scored["scores"]["literal_mismatch_count"] == 0
    assert scored["scores"]["literal_guarded"] == pytest.approx(
        scored["scores"]["reverse_context_calibrated"]
        if scored["scores"].get("reverse_context_calibrated") is not None
        else scored["scores"]["reverse_context"],
    )
    assert not any(w.startswith("literal_mismatch") for w in scored["warnings"])


def test_score_groundedness_chunked_runs_literal_diff_against_full_support_union() -> None:
    provider = DummyGroundednessProvider(dim=10)
    response_text = "Released on 20 July 1981 for $19.99."

    response_embedding = torch.as_tensor(provider.encode([response_text])[0], dtype=torch.float32)

    def _unit(idx: int, text: str) -> SupportUnitInput:
        embedding = torch.as_tensor(provider.encode([text])[0], dtype=torch.float32)
        return SupportUnitInput(
            support_id=f"sup-{idx}",
            chunk_id=f"chunk-{idx}",
            source_mode="raw_context",
            text=text,
            embeddings=embedding,
            tokens=provider.tokenize(text),
        )

    batches = [
        [_unit(0, "Released on 20 July 1981 in the United States.")],
        [_unit(1, "Sold for $19.99 plus tax in select markets.")],
    ]

    scored = score_groundedness_chunked(
        support_batches=batches,
        response_embeddings=response_embedding,
        response_tokens=provider.tokenize(response_text),
        response_text=response_text,
    )

    assert scored["scores"]["literal_mismatch_count"] == 0
    mismatch_warnings = [w for w in scored["warnings"] if w.startswith("literal_mismatch")]
    assert mismatch_warnings == []
    assert {literal["value"] for literal in scored["literal_diagnostics"]["matches"]} >= {
        "20 July 1981",
        "$19.99",
    }


def test_service_groundedness_endpoint_surfaces_literal_diagnostics(tmp_path: Path) -> None:
    provider = DummyGroundednessProvider(dim=8)
    support_text = "Released on 20 July 1981 for $19.99."

    with _patch_provider(provider):
        with _create_client(tmp_path) as client:
            response = client.post(
                "/collections/groundedness-literals",
                json={"dimension": provider.dim, "kind": "shard", "n_shards": 1},
            )
            assert response.status_code == 200
            assert (
                client.post(
                    "/collections/groundedness-literals/points",
                    json={
                        "points": [
                            {
                                "id": "doc-1",
                                "vectors": _encode_text(provider, support_text),
                                "payload": {"text": support_text},
                            }
                        ]
                    },
                ).status_code
                == 200
            )

            scored = client.post(
                "/collections/groundedness-literals/groundedness",
                json={
                    "chunk_ids": ["doc-1"],
                    "response_text": "Released on 22 July 1981 for $24.99.",
                },
            )

    assert scored.status_code == 200
    payload = scored.json()
    diagnostics = payload["literal_diagnostics"]
    assert diagnostics is not None
    assert payload["scores"]["literal_mismatch_count"] >= 2
    mismatch_kinds = {item["kind"] for item in diagnostics["mismatches"]}
    assert {"date", "currency"} <= mismatch_kinds


# ----------------------------------------------------------------------
# Phase D: NLI / claim verifier as primary peer
# ----------------------------------------------------------------------


from voyager_index._internal.server.api.groundedness_nli import (  # noqa: E402  (after section delimiter)
    Claim,
    ClaimVerification,
    HuggingFaceNLIProvider,
    aggregate_nli_score,
    fuse_groundedness_v2,
    project_claim_scores_to_tokens,
    split_claims,
    verify_claims,
)


class FakeNLIProvider:
    """Deterministic NLI backend used to drive Phase D unit tests.

    Returns ``entail=1.0`` when every content word in the hypothesis appears
    in the premise, ``contradiction=1.0`` when the hypothesis contains
    explicit ``not``/``never``, and a neutral split otherwise. The
    ``call_log`` exposes the exact pairs handed to the provider so tests can
    audit batching behavior.
    """

    def __init__(self) -> None:
        self.call_log: list[list[tuple[str, str]]] = []

    @staticmethod
    def _content(text: str) -> set:
        return {token.lower() for token in re.findall(r"\w+", text or "") if len(token) > 2}

    def entail(self, premises, hypotheses):
        self.call_log.append(list(zip(premises, hypotheses)))
        triples: list[tuple[float, float, float]] = []
        for premise, hypothesis in zip(premises, hypotheses):
            hyp_terms = self._content(hypothesis)
            prem_terms = self._content(premise)
            if any(token in hyp_terms for token in {"not", "never", "no"}):
                triples.append((0.05, 0.10, 0.85))
                continue
            if not hyp_terms:
                triples.append((0.33, 0.34, 0.33))
                continue
            overlap = hyp_terms & prem_terms
            ratio = len(overlap) / max(1, len(hyp_terms))
            if ratio >= 0.8:
                triples.append((0.90, 0.07, 0.03))
            elif ratio >= 0.4:
                triples.append((0.55, 0.30, 0.15))
            else:
                triples.append((0.10, 0.20, 0.70))
        return triples


def test_split_claims_produces_sentence_level_claims_with_offsets() -> None:
    text = (
        "Teardrops was released in July 1981. "
        "It charted at number 102 in the United States; however, the UK release "
        "was delayed."
    )
    claims = split_claims(text)
    assert len(claims) >= 2
    for claim in claims:
        assert text[claim.char_start : claim.char_end] == claim.text
    joined = " | ".join(claim.text for claim in claims)
    assert "1981" in joined
    assert "however" in joined or "UK" in joined


def test_split_claims_respects_max_claims_cap() -> None:
    text = ". ".join("Sentence number {}".format(idx) for idx in range(20)) + "."
    capped = split_claims(text, max_claims=5)
    assert len(capped) == 5


def test_aggregate_nli_score_returns_none_when_all_skipped() -> None:
    verifications = [
        ClaimVerification(
            claim=Claim(index=0, text="x", char_start=0, char_end=1),
            entailment=0.0,
            neutral=0.0,
            contradiction=0.0,
            score=0.0,
            premises=[],
            skipped=True,
            skip_reason="no_premises",
        )
    ]
    assert aggregate_nli_score(verifications) is None


def test_aggregate_nli_score_maps_signed_margin_into_unit_interval() -> None:
    verifications = [
        ClaimVerification(
            claim=Claim(index=0, text="a", char_start=0, char_end=1),
            entailment=0.9,
            neutral=0.05,
            contradiction=0.05,
            score=0.85,
            premises=["p"],
        ),
        ClaimVerification(
            claim=Claim(index=1, text="b", char_start=2, char_end=3),
            entailment=0.10,
            neutral=0.20,
            contradiction=0.70,
            score=-0.60,
            premises=["p"],
        ),
    ]
    aggregate = aggregate_nli_score(verifications)
    assert aggregate is not None
    assert 0.0 <= aggregate <= 1.0
    assert aggregate == pytest.approx(0.5 + 0.5 * (0.85 - 0.60) / 2.0, rel=1e-6)


def test_fuse_groundedness_v2_renormalizes_when_some_channels_missing() -> None:
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.8,
        literal_guarded=None,
        nli_aggregate=0.6,
        weights={"calibrated": 0.5, "literal": 0.2, "nli": 0.3},
    )
    assert fused is not None
    assert fused == pytest.approx((0.8 * 0.5 + 0.6 * 0.3) / (0.5 + 0.3))


def test_fuse_groundedness_v2_returns_none_when_no_channel_available() -> None:
    assert fuse_groundedness_v2(
        reverse_context_calibrated=None,
        literal_guarded=None,
        nli_aggregate=None,
    ) is None


def test_project_claim_scores_to_tokens_assigns_scores_inside_spans() -> None:
    response_text = "Alpha beats beta. Gamma is above zero."
    response_tokens = ["Alpha", "beats", "beta", ".", "Gamma", "is", "above", "zero", "."]
    verifications = [
        ClaimVerification(
            claim=Claim(index=0, text="Alpha beats beta.", char_start=0, char_end=17),
            entailment=0.9, neutral=0.05, contradiction=0.05, score=0.85, premises=["p"],
        ),
        ClaimVerification(
            claim=Claim(index=1, text="Gamma is above zero.", char_start=18, char_end=38),
            entailment=0.1, neutral=0.2, contradiction=0.7, score=-0.6, premises=["p"],
        ),
    ]
    projected = project_claim_scores_to_tokens(response_tokens, response_text, verifications)
    assert projected[0] == pytest.approx(0.85)
    assert projected[2] == pytest.approx(0.85)
    assert projected[4] == pytest.approx(-0.6)
    assert projected[7] == pytest.approx(-0.6)


def test_verify_claims_runs_per_claim_and_skips_when_no_premises() -> None:
    response = "Apples are sweet. Asteroids orbit the sun."
    units = [
        SupportUnitInput(
            support_id="s-0",
            chunk_id="c-0",
            source_mode="raw_context",
            text="Apples are sweet and red fruits found in many orchards.",
            embeddings=torch.zeros((1, 4)),
            tokens=["x"],
        )
    ]
    nli = FakeNLIProvider()
    verifications, warnings = verify_claims(response, units, nli)
    assert len(verifications) == 2
    apples_claim = next(v for v in verifications if "apple" in v.claim.text.lower())
    asteroid_claim = next(v for v in verifications if "asteroid" in v.claim.text.lower())
    assert not apples_claim.skipped
    assert apples_claim.entailment > 0.5
    assert asteroid_claim.skipped or asteroid_claim.score <= 0


def test_verify_claims_marks_latency_budget_breach() -> None:
    response = "First. Second. Third."
    units = [
        SupportUnitInput(
            support_id=f"s-{i}",
            chunk_id=f"c-{i}",
            source_mode="raw_context",
            text="first second third statement",
            embeddings=torch.zeros((1, 4)),
            tokens=["x"],
        )
        for i in range(3)
    ]

    class SlowNLI(FakeNLIProvider):
        def entail(self, premises, hypotheses):
            time.sleep(0.05)
            return super().entail(premises, hypotheses)

    import time

    verifications, warnings = verify_claims(
        response,
        units,
        SlowNLI(),
        max_batch=1,
        max_latency_ms=1.0,
    )
    assert any(w.startswith("nli_budget_exceeded") for w in warnings)
    assert any(v.skipped for v in verifications)


def test_score_groundedness_emits_groundedness_v2_when_nli_provider_present() -> None:
    provider = DummyGroundednessProvider(dim=12)
    support_text = "Apples are sweet red fruits from temperate orchards."
    response_text = "Apples are sweet. Apples are red."
    support_embedding = torch.as_tensor(provider.encode([support_text])[0], dtype=torch.float32)
    units = [
        SupportUnitInput(
            support_id="sup-0",
            chunk_id="chunk-0",
            source_mode="raw_context",
            text=support_text,
            embeddings=support_embedding,
            tokens=provider.tokenize(support_text),
        )
    ]
    response_embedding = torch.as_tensor(provider.encode([response_text])[0], dtype=torch.float32)

    nli = FakeNLIProvider()
    scored = score_groundedness(
        support_units=units,
        response_embeddings=response_embedding,
        response_tokens=provider.tokenize(response_text),
        response_text=response_text,
        nli_provider=nli,
        nli_max_claims=4,
        nli_top_k_premises=2,
        nli_max_batch=4,
        nli_max_latency_ms=2_000.0,
    )

    assert scored["scores"]["nli_aggregate"] is not None
    assert scored["scores"]["groundedness_v2"] is not None
    assert 0.0 <= scored["scores"]["groundedness_v2"] <= 1.0
    assert scored["scores"]["nli_claim_count"] >= 2
    assert scored["nli_diagnostics"] is not None
    assert all(claim["score"] > 0 for claim in scored["nli_diagnostics"]["claims"])
    assert any(row["nli_score"] is not None for row in scored["response_tokens"])
    assert nli.call_log


def test_score_groundedness_omits_nli_fields_when_provider_is_none() -> None:
    provider = DummyGroundednessProvider(dim=12)
    support_text = "Apples are sweet red fruits."
    response_text = "Apples are sweet."
    support_embedding = torch.as_tensor(provider.encode([support_text])[0], dtype=torch.float32)
    units = [
        SupportUnitInput(
            support_id="sup-0",
            chunk_id="chunk-0",
            source_mode="raw_context",
            text=support_text,
            embeddings=support_embedding,
            tokens=provider.tokenize(support_text),
        )
    ]
    response_embedding = torch.as_tensor(provider.encode([response_text])[0], dtype=torch.float32)

    scored = score_groundedness(
        support_units=units,
        response_embeddings=response_embedding,
        response_tokens=provider.tokenize(response_text),
        response_text=response_text,
    )
    assert scored["scores"]["nli_aggregate"] is None
    assert scored["scores"]["nli_claim_count"] == 0
    # groundedness_v2 still fuses the remaining peer channels (calibrated +
    # literal-guarded) when NLI is disabled; the only requirement is that the
    # NLI channel itself does not contribute and the fused value remains in [0,1].
    assert scored["scores"]["groundedness_v2"] is None or 0.0 <= scored["scores"]["groundedness_v2"] <= 1.0
    assert scored["nli_diagnostics"] is None
    assert all(row["nli_score"] is None for row in scored["response_tokens"])


def test_score_groundedness_chunked_runs_nli_against_full_support_union() -> None:
    provider = DummyGroundednessProvider(dim=10)
    response_text = "Apples are sweet. Apples are red."
    response_embedding = torch.as_tensor(provider.encode([response_text])[0], dtype=torch.float32)

    def _unit(idx: int, text: str) -> SupportUnitInput:
        embedding = torch.as_tensor(provider.encode([text])[0], dtype=torch.float32)
        return SupportUnitInput(
            support_id=f"sup-{idx}",
            chunk_id=f"chunk-{idx}",
            source_mode="raw_context",
            text=text,
            embeddings=embedding,
            tokens=provider.tokenize(text),
        )

    batches = [
        [_unit(0, "Apples are sweet temperate fruits.")],
        [_unit(1, "Apples are red and grow in orchards.")],
    ]

    nli = FakeNLIProvider()
    scored = score_groundedness_chunked(
        support_batches=batches,
        response_embeddings=response_embedding,
        response_tokens=provider.tokenize(response_text),
        response_text=response_text,
        nli_provider=nli,
    )
    assert scored["scores"]["groundedness_v2"] is not None
    assert scored["scores"]["nli_claim_count"] >= 2
    assert scored["nli_diagnostics"] is not None
    assert any(row["nli_score"] is not None for row in scored["response_tokens"])


def test_huggingface_nli_provider_resolves_label_indices_from_id2label(monkeypatch) -> None:
    class _DummyConfig:
        id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}

    class _DummyTokenizer:
        def __call__(self, premises, hypotheses, **kwargs):
            import torch

            return {"input_ids": torch.zeros((len(premises), 3), dtype=torch.long)}

    class _DummyModel:
        config = _DummyConfig()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            import torch

            class _Out:
                logits = torch.tensor([[2.0, 0.0, -2.0]])

            return _Out()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        lambda *args, **kwargs: _DummyModel(),
    )

    nli = HuggingFaceNLIProvider("dummy-mnli", device="cpu")
    triples = nli.entail(["premise"], ["hypothesis"])
    assert len(triples) == 1
    entail, neutral, contradiction = triples[0]
    # softmax([2, 0, -2]) = approx [0.867, 0.117, 0.016]
    assert entail == pytest.approx(0.867, abs=1e-2)
    assert contradiction < neutral < entail


# ----------------------------------------------------------------------
# Phase E: minimal-pair fixture and external benchmark adapters
# ----------------------------------------------------------------------


def test_minimal_pairs_cover_all_strata_with_at_least_200_total() -> None:
    from research.triangular_maxsim.groundedness_minimal_pairs import (
        build_minimal_pairs,
        stratum_summary,
    )

    pairs = build_minimal_pairs(pairs_per_stratum=30)
    assert len(pairs) >= 200
    summary = stratum_summary(pairs)
    expected_strata = {
        "entity_swap",
        "date_swap",
        "number_swap",
        "unit_swap",
        "negation",
        "role_swap",
        "partial",
    }
    assert set(summary.keys()) == expected_strata
    for stratum, count in summary.items():
        assert count >= 28, "stratum {0} only has {1} pairs".format(stratum, count)


def test_minimal_pairs_are_deterministic_for_seed() -> None:
    from research.triangular_maxsim.groundedness_minimal_pairs import build_minimal_pairs

    first = build_minimal_pairs(pairs_per_stratum=5, seed=11)
    second = build_minimal_pairs(pairs_per_stratum=5, seed=11)
    assert [pair.signature() for pair in first] == [pair.signature() for pair in second]
    third = build_minimal_pairs(pairs_per_stratum=5, seed=12)
    assert [pair.signature() for pair in first] != [pair.signature() for pair in third]


def test_minimal_pair_positive_and_negative_differ_within_stratum() -> None:
    from research.triangular_maxsim.groundedness_minimal_pairs import build_minimal_pairs

    pairs = build_minimal_pairs(pairs_per_stratum=4, seed=23)
    assert pairs, "expected at least one minimal pair"
    for pair in pairs:
        assert pair.positive != pair.negative
        assert pair.context, "context must not be empty"
        assert pair.stratum in {
            "entity_swap",
            "date_swap",
            "number_swap",
            "unit_swap",
            "negation",
            "role_swap",
            "partial",
        }


def test_external_benchmark_adapters_skip_when_data_unavailable(monkeypatch) -> None:
    from research.triangular_maxsim.groundedness_external_benchmarks import (
        available_benchmarks,
        load_factscore,
        load_halueval,
        load_ragtruth,
    )

    monkeypatch.delenv("VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR", raising=False)
    monkeypatch.delenv("VOYAGER_GROUNDEDNESS_HALUEVAL_DIR", raising=False)
    monkeypatch.delenv("VOYAGER_GROUNDEDNESS_FACTSCORE_DIR", raising=False)

    assert load_ragtruth() is None
    assert load_halueval() is None
    assert load_factscore() is None
    assert available_benchmarks() == []


def test_ragtruth_loader_parses_jsonl_with_span_labels(tmp_path, monkeypatch) -> None:
    from research.triangular_maxsim.groundedness_external_benchmarks import load_ragtruth

    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    (qa_dir / "test.jsonl").write_text(
        '{"id": "rt-1", "source_info": "sky is blue", "response": "The sky is blue.", "labels": []}\n'
        '{"id": "rt-2", "source_info": "sky is blue", "response": "The sky is green.", "labels": [{"start": 0, "end": 12}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR", str(tmp_path))
    samples = load_ragtruth()
    assert samples is not None
    assert len(samples) == 2
    assert samples[0].label == "faithful"
    assert samples[1].label == "hallucinated"
    assert all(sample.benchmark == "ragtruth" for sample in samples)
    assert all(sample.stratum == "qa" for sample in samples)


def test_halueval_loader_emits_paired_positive_and_negative(tmp_path, monkeypatch) -> None:
    from research.triangular_maxsim.groundedness_external_benchmarks import load_halueval

    payload = (
        '{"knowledge": "Einstein won the Nobel Prize in 1921.",'
        ' "question": "When did Einstein win the Nobel Prize?",'
        ' "right_answer": "1921",'
        ' "hallucinated_answer": "1925"}\n'
    )
    (tmp_path / "qa_data.jsonl").write_text(payload, encoding="utf-8")
    monkeypatch.setenv("VOYAGER_GROUNDEDNESS_HALUEVAL_DIR", str(tmp_path))
    samples = load_halueval()
    assert samples is not None
    assert len(samples) == 2
    labels = sorted(sample.label for sample in samples)
    assert labels == ["faithful", "hallucinated"]
    assert all(sample.query for sample in samples)


def test_factscore_loader_aggregates_per_claim_precision(tmp_path, monkeypatch) -> None:
    from research.triangular_maxsim.groundedness_external_benchmarks import load_factscore

    payload = (
        '{"topic": "Marie Curie", "output": "She won two Nobel Prizes.",'
        ' "annotations": [{"is_supported": true}, {"is_supported": true}]}\n'
        '{"topic": "Other", "output": "Mostly invented.",'
        ' "annotations": [{"is_supported": false}, {"is_supported": false}]}\n'
    )
    (tmp_path / "biographies.jsonl").write_text(payload, encoding="utf-8")
    monkeypatch.setenv("VOYAGER_GROUNDEDNESS_FACTSCORE_DIR", str(tmp_path))
    samples = load_factscore()
    assert samples is not None
    assert len(samples) == 2
    assert samples[0].label == "faithful"
    assert samples[1].label == "hallucinated"
    assert samples[0].raw["precision"] == 1.0
    assert samples[1].raw["precision"] == 0.0


def test_preregistered_targets_cover_required_lanes() -> None:
    from research.triangular_maxsim.groundedness_external_benchmarks import (
        PREREGISTERED_TARGETS,
    )

    expected_keys = {
        "ragtruth",
        "halueval_qa",
        "factscore",
        "minimal_pairs_lexical",
        "minimal_pairs_semantic",
        "minimal_pairs_partial",
        "latency_score_only",
        "latency_with_nli",
    }
    assert expected_keys.issubset(PREREGISTERED_TARGETS.keys())
    for key, target in PREREGISTERED_TARGETS.items():
        assert "metric" in target
        assert "notes" in target


def test_evaluate_minimal_pairs_returns_per_stratum_paired_accuracy() -> None:
    from research.triangular_maxsim.groundedness_external_eval import evaluate_minimal_pairs
    from research.triangular_maxsim.groundedness_minimal_pairs import build_minimal_pairs

    provider = DummyGroundednessProvider(dim=24)
    pairs = build_minimal_pairs(pairs_per_stratum=2, seed=29)
    results = evaluate_minimal_pairs(pairs, provider)
    assert "per_stratum" in results
    assert "overall_accuracy" in results
    assert results["pair_count"] == len(pairs)
    for stats in results["per_stratum"].values():
        assert 0.0 <= stats["paired_accuracy"] <= 1.0
        assert stats["ci_lower"] <= stats["paired_accuracy"]


def test_assemble_report_marks_skipped_when_external_data_missing() -> None:
    from research.triangular_maxsim.groundedness_external_eval import assemble_report

    minimal_pair_results = {
        "per_stratum": {
            "entity_swap": {"n": 30, "paired_accuracy": 0.9, "ci_lower": 0.85, "ci_upper": 0.95},
            "date_swap": {"n": 30, "paired_accuracy": 0.85, "ci_lower": 0.80, "ci_upper": 0.90},
            "number_swap": {"n": 30, "paired_accuracy": 0.82, "ci_lower": 0.77, "ci_upper": 0.88},
            "unit_swap": {"n": 30, "paired_accuracy": 0.84, "ci_lower": 0.78, "ci_upper": 0.89},
            "negation": {"n": 30, "paired_accuracy": 0.72, "ci_lower": 0.66, "ci_upper": 0.78},
            "role_swap": {"n": 30, "paired_accuracy": 0.74, "ci_lower": 0.68, "ci_upper": 0.80},
            "partial": {"n": 30, "paired_accuracy": 0.66, "ci_lower": 0.61, "ci_upper": 0.72},
        },
        "latency_p95_ms": 80.0,
    }
    report = assemble_report(
        minimal_pair_results=minimal_pair_results,
        external_results={"ragtruth": None, "halueval": None, "factscore": None},
    )
    assert report["criteria"]["ragtruth"]["status"] == "skipped"
    assert report["criteria"]["halueval_qa"]["status"] == "skipped"
    assert report["criteria"]["factscore"]["status"] == "skipped"
    assert report["criteria"]["minimal_pairs_lexical"]["met"] is True
    assert report["criteria"]["minimal_pairs_semantic"]["met"] is True
    assert report["criteria"]["minimal_pairs_partial"]["met"] is True
    assert report["criteria"]["latency_score_only"]["met"] is True
    assert report["criteria"]["latency_with_nli"]["met"] is True
