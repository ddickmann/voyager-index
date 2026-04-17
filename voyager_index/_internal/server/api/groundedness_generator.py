"""Generator providers for the semantic-entropy peer (Phase G2).

The groundedness service needs multiple response samples drawn from the
same LLM at non-zero sampling temperature to compute semantic entropy
(see ``groundedness_semantic_entropy``). Two ingress paths are supported:

1. **Caller-supplied** samples via ``GroundednessRequest.verification_samples``
   (Phase G1 API) — the service forwards them directly without hitting an
   external LLM. This is the default and zero-latency-overhead path.

2. **Cooperative generation**, where the service calls the caller's LLM
   itself ``N`` times. This module provides the ``GeneratorProvider``
   protocol plus two implementations:

   - ``VllmFactoryGeneratorProvider``: POSTs to vllm-factory's OpenAI-
     compatible ``/v1/chat/completions`` endpoint. Kept lightweight so it
     can be swapped for the upcoming BYOP NLI path without disruption.
   - ``CustomCallbackGeneratorProvider``: drives a user-supplied async
     callable. Primarily for tests and bespoke integrations.

The cooperative path stays deferred behind an explicit caller flag until
the vllm-factory BYOP work lands — we ship the protocol and a reference
HTTP implementation now so the plumbing is in place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass
class GeneratorEndpointConfig:
    """Caller-supplied config describing where and how to draw samples."""

    endpoint: str
    model: str
    num_samples: int = 5
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256
    api_key: Optional[str] = None
    timeout_s: float = 15.0


class GeneratorProvider(Protocol):
    """Return ``n`` alternate responses for ``(context, query)`` at T > 0."""

    def generate_samples(
        self,
        *,
        context: str,
        query: Optional[str],
        seed_response: Optional[str],
        num_samples: int,
    ) -> List[str]:
        ...


class VllmFactoryGeneratorProvider:
    """Draw samples from a vllm-factory ``/v1/chat/completions`` endpoint.

    Kept deliberately minimal: we run sequential HTTP POSTs because even
    for ``num_samples=5`` with a 256-token response and a hot server, the
    round-trip already dominates the groundedness latency budget. Adding
    concurrency here only helps when the upstream server serializes
    requests; upstream vllm-factory handles batching internally.
    """

    def __init__(
        self,
        config: GeneratorEndpointConfig,
        *,
        http_client: Any = None,
    ) -> None:
        self.config = config
        self._http_client = http_client

    def _get_http_client(self):
        if self._http_client is not None:
            return self._http_client
        import httpx  # noqa: WPS433 — lazy to keep import graph cheap

        self._http_client = httpx.Client(timeout=self.config.timeout_s)
        return self._http_client

    def _format_messages(self, *, context: str, query: Optional[str]) -> List[dict]:
        system = (
            "You are answering from the supplied context only. If the context "
            "does not contain enough information, say so."
        )
        user_parts: List[str] = []
        if query:
            user_parts.append("Question: " + query.strip())
        user_parts.append("Context:\n" + (context or "").strip())
        user_parts.append("Answer:")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def generate_samples(
        self,
        *,
        context: str,
        query: Optional[str],
        seed_response: Optional[str],
        num_samples: int,
    ) -> List[str]:
        del seed_response  # unused for this provider
        n = max(1, int(num_samples))
        messages = self._format_messages(context=context, query=query)
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        payload_base = {
            "model": self.config.model,
            "messages": messages,
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "max_tokens": int(self.config.max_tokens),
        }
        samples: List[str] = []
        client = self._get_http_client()
        url = self.config.endpoint.rstrip("/") + "/v1/chat/completions"
        for _ in range(n):
            try:
                resp = client.post(url, json=payload_base, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if not choices:
                    continue
                content = (
                    choices[0]
                    .get("message", {})
                    .get("content", "")
                )
                if isinstance(content, str) and content.strip():
                    samples.append(content.strip())
            except Exception as exc:  # noqa: BLE001 — degrade gracefully
                logger.warning("generator_sample_failed", extra={"error": str(exc)})
                continue
        return samples


class CustomCallbackGeneratorProvider:
    """Delegate sample generation to a user-supplied callable.

    Useful for tests or custom LLM wrappers that do not speak the
    OpenAI-compat protocol used by vllm-factory.
    """

    def __init__(
        self,
        callback: Callable[..., Sequence[str]],
    ) -> None:
        self.callback = callback

    def generate_samples(
        self,
        *,
        context: str,
        query: Optional[str],
        seed_response: Optional[str],
        num_samples: int,
    ) -> List[str]:
        try:
            result = self.callback(
                context=context,
                query=query,
                seed_response=seed_response,
                num_samples=num_samples,
            )
        except TypeError:
            result = self.callback(context, query, seed_response, num_samples)
        except Exception as exc:  # noqa: BLE001
            logger.warning("custom_generator_failed", extra={"error": str(exc)})
            return []
        return [str(item) for item in (result or []) if item]


__all__ = [
    "CustomCallbackGeneratorProvider",
    "GeneratorEndpointConfig",
    "GeneratorProvider",
    "VllmFactoryGeneratorProvider",
]
