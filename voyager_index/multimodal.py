"""
Multimodal model metadata and provider helpers for voyager-index.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class MultimodalModelSpec:
    """Metadata for a supported multimodal embedding model.

    Attributes:
        plugin_name: Short alias used in config files and CLI.
        model_id: HuggingFace model ID or path.
        architecture: Human-readable description of the backbone.
        embedding_style: ``"colpali"`` (patch-level) or ``"colbert"`` (token-level).
        modalities: Tuple of supported modality names (e.g. ``("text", "image")``).
        pooling_task: vLLM pooling task name (e.g. ``"token_embed"``).
        serve_command: Example CLI command to launch the model with vLLM.
    """

    plugin_name: str
    model_id: str
    architecture: str
    embedding_style: str
    modalities: tuple[str, ...]
    pooling_task: str
    serve_command: str


DEFAULT_MULTIMODAL_MODEL = "collfm2"


SUPPORTED_MULTIMODAL_MODELS: dict[str, MultimodalModelSpec] = {
    "collfm2": MultimodalModelSpec(
        plugin_name="collfm2",
        model_id="VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1",
        architecture="LFM2-VL + ColPali-style pooling",
        embedding_style="colpali",
        modalities=("text", "image"),
        pooling_task="token_embed",
        serve_command="vllm serve VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1 --trust-remote-code --dtype bfloat16 --port 8200",
    ),
    "colqwen3": MultimodalModelSpec(
        plugin_name="colqwen3",
        model_id="VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1",
        architecture="Qwen3-VL + ColPali-style pooling",
        embedding_style="colpali",
        modalities=("text", "image"),
        pooling_task="token_embed",
        serve_command="vllm serve VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1 --trust-remote-code --dtype bfloat16 --port 8200",
    ),
    "nemotron_colembed": MultimodalModelSpec(
        plugin_name="nemotron_colembed",
        model_id="nvidia/nemotron-colembed-vl-4b-v2",
        architecture="Qwen3-VL bidirectional + token-level ColBERT-style output",
        embedding_style="colbert",
        modalities=("text", "image"),
        pooling_task="token_embed",
        serve_command="vllm serve nvidia/nemotron-colembed-vl-4b-v2 --trust-remote-code --dtype bfloat16 --no-enable-prefix-caching --no-enable-chunked-prefill --port 8200",
    ),
}

DEFAULT_MULTIMODAL_MODEL_SPEC = SUPPORTED_MULTIMODAL_MODELS[DEFAULT_MULTIMODAL_MODEL]


class VllmPoolingProvider:
    """
    Thin HTTP client for vLLM-hosted multimodal pooling endpoints.

    Sends embedding requests to a running vLLM server that exposes a
    ``/v1/pooling`` endpoint (OpenAI-compatible).  Callers control the
    input payload; this class standardizes the request shape and handles
    serialization.

    Args:
        endpoint: Base URL of the vLLM server (e.g. ``"http://localhost:8200"``).
        model: Model identifier matching the served model.
        timeout: HTTP timeout in seconds (default 60).
    """

    def __init__(self, endpoint: str, model: str, timeout: float = 60.0):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def build_payload(
        self,
        input_data: Any,
        extra_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a JSON-serializable request body without sending it.

        Returns:
            Dict with ``model``, ``input``, and optional ``extra_kwargs``.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
        }
        if extra_kwargs:
            payload["extra_kwargs"] = dict(extra_kwargs)
        payload.update(kwargs)
        return payload

    def pool(
        self,
        input_data: Any,
        extra_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a single pooling request and return the parsed JSON response.

        Raises:
            ImportError: If ``httpx`` is not installed.
            httpx.HTTPStatusError: On non-2xx response.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for VllmPoolingProvider. Install voyager-index[multimodal]."
            ) from exc

        payload = self.build_payload(input_data, extra_kwargs=extra_kwargs, **kwargs)
        response = httpx.post(f"{self.endpoint}/v1/pooling", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def batch_pool(
        self,
        inputs: Iterable[Any],
        extra_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Sequentially pool each input and collect results.

        Returns:
            List of parsed JSON responses, one per input.
        """
        return [self.pool(item, extra_kwargs=extra_kwargs, **kwargs) for item in inputs]


class VllmFactoryModernColBERTProvider:
    """ModernColBERT-specific HTTP client for vllm-factory `/pooling`.

    This provider implements the minimal encode/tokenize surface required by
    downstream services while keeping request construction compatible with the
    `moderncolbert_io` plugin contract.
    """

    query_prefix_id = 50368
    document_prefix_id = 50369

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 60.0,
        health_timeout: float = 10.0,
        batch_size: int = 16,
        max_concurrency: int = 8,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.model_name = model
        self.model_name_or_path = model
        self.timeout = float(timeout)
        self.health_timeout = float(health_timeout)
        self.batch_size = max(1, int(batch_size))
        self.max_concurrency = max(1, int(max_concurrency))
        self._http_client: Any = None

        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional install shape
            raise ImportError(
                "transformers is required for VllmFactoryModernColBERTProvider."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=True,
            trust_remote_code=True,
        )
        config = None
        try:
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        except Exception:
            config = None
        self.colbert_dim = int(getattr(config, "colbert_dim", getattr(config, "dim", 128)) if config else 128)
        self.query_maxlen = int(getattr(config, "query_length", getattr(config, "query_maxlen", 256)) if config else 256)
        self.doc_maxlen = int(
            getattr(
                config,
                "document_length",
                getattr(config, "document_maxlen", getattr(config, "max_position_embeddings", 8192)),
            )
            if config
            else 8192
        )

    def _get_http_client(self):
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for VllmFactoryModernColBERTProvider. Install voyager-index[multimodal]."
            ) from exc

        limits = httpx.Limits(
            max_connections=max(self.max_concurrency * 2, 8),
            max_keepalive_connections=max(self.max_concurrency, 4),
        )
        self._http_client = httpx.Client(base_url=self.endpoint, timeout=self.timeout, limits=limits)
        return self._http_client

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def healthcheck(self) -> dict[str, Any]:
        client = self._get_http_client()
        response = client.get("/health", timeout=self.health_timeout)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"status": response.text}

    def _max_length(self, *, is_query: bool) -> int:
        return self.query_maxlen if is_query else self.doc_maxlen

    def _token_ids(self, text: str, *, is_query: bool) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max(1, self._max_length(is_query=is_query) - 1),
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_ids = list(input_ids)
        if not input_ids:
            return []
        prefix_id = self.query_prefix_id if is_query else self.document_prefix_id
        return [int(input_ids[0]), int(prefix_id), *[int(item) for item in input_ids[1:]]]

    def tokenize(self, text: str, *, is_query: bool = False) -> list[str]:
        input_ids = self._token_ids(text, is_query=is_query)
        if not input_ids:
            return []
        try:
            tokens = list(self.tokenizer.convert_ids_to_tokens(input_ids))
        except Exception:
            tokens = [str(item) for item in input_ids]
        normalized: list[str] = []
        for token_id, token in zip(input_ids, tokens):
            if token_id == self.query_prefix_id and (token is None or str(token) == str(token_id)):
                normalized.append("[Q]")
            elif token_id == self.document_prefix_id and (token is None or str(token) == str(token_id)):
                normalized.append("[D]")
            else:
                normalized.append(str(token))
        return normalized

    def encoded_token_count(self, text: str, *, is_query: bool = False) -> int:
        """Strict, server-parity token count for packing budgets.

        Returns the exact length of the token id sequence the provider will send
        to the backend, including special tokens and the ``[D]/[Q]`` prefix.
        Downstream callers prefer this over a naive tokenizer call so packed
        windows match what the encoder will actually consume.
        """

        return len(self._token_ids(text, is_query=is_query))

    def build_payload(self, text: str, *, is_query: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "data": {
                "text": text,
                "is_query": bool(is_query),
            },
            "task": "plugin",
        }

    def _decode_embedding(self, payload: Any) -> np.ndarray:
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        if isinstance(data, dict):
            for key in ("data", "embedding", "embeddings", "output"):
                if key in data:
                    data = data[key]
                    break

        if isinstance(data, str):
            raw = np.frombuffer(base64.b64decode(data.encode("ascii")), dtype=np.float32)
            if raw.size == 0:
                return np.zeros((0, self.colbert_dim), dtype=np.float32)
            if raw.size % self.colbert_dim != 0:
                raise ValueError(
                    f"ModernColBERT response length {raw.size} is not divisible by colbert_dim={self.colbert_dim}"
                )
            return raw.reshape(-1, self.colbert_dim)

        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            if array.size == 0:
                return np.zeros((0, self.colbert_dim), dtype=np.float32)
            if array.size % self.colbert_dim != 0:
                raise ValueError(
                    "ModernColBERT /pooling response length {size} is not divisible by colbert_dim={dim}; "
                    "the server did not return a multi-vector matrix. Check the model id and IO processor wiring.".format(
                        size=array.size,
                        dim=self.colbert_dim,
                    )
                )
            return array.reshape(-1, self.colbert_dim)
        if array.ndim == 2:
            if array.shape[1] != self.colbert_dim:
                raise ValueError(
                    "ModernColBERT /pooling response inner dim {got} does not match colbert_dim={expected}".format(
                        got=int(array.shape[1]),
                        expected=self.colbert_dim,
                    )
                )
            return array
        raise TypeError("Unsupported ModernColBERT /pooling payload shape")

    def _pool_text(self, text: str, *, is_query: bool) -> np.ndarray:
        client = self._get_http_client()
        response = client.post("/pooling", json=self.build_payload(text, is_query=is_query))
        response.raise_for_status()
        return self._decode_embedding(response.json())

    def encode(self, inputs: Any, **kwargs: Any) -> list[np.ndarray]:
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if not texts:
            return []
        is_query = bool(kwargs.get("is_query", False))

        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            max_workers = min(self.max_concurrency, len(batch))
            if max_workers <= 1:
                outputs.extend(self._pool_text(text, is_query=is_query) for text in batch)
                continue
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._pool_text, text, is_query=is_query) for text in batch]
                outputs.extend(future.result() for future in futures)
        return outputs


__all__ = [
    "DEFAULT_MULTIMODAL_MODEL",
    "DEFAULT_MULTIMODAL_MODEL_SPEC",
    "MultimodalModelSpec",
    "SUPPORTED_MULTIMODAL_MODELS",
    "VllmPoolingProvider",
    "VllmFactoryModernColBERTProvider",
]
