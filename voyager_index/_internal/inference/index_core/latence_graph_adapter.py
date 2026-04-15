from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .graph_contract import GraphContractClass

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional premium dependency
    from latence import Latence

    _LATENCE_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - optional premium dependency
    Latence = None  # type: ignore[assignment]
    _LATENCE_SDK_AVAILABLE = False


class LatenceGraphAdapter:
    """Premium LatenceAI graph adapter with a safe OSS fallback."""

    def __init__(self, client: Optional[Any] = None) -> None:
        self.client = client

    @staticmethod
    def sdk_available() -> bool:
        return bool(_LATENCE_SDK_AVAILABLE)

    def _ensure_client(self) -> Optional[Any]:
        if self.client is not None:
            return self.client
        if not self.sdk_available() or Latence is None:
            return None
        try:  # pragma: no cover - requires premium dependency and auth
            self.client = Latence()
        except Exception as exc:
            logger.warning("Latence SDK is installed but could not initialize a client: %s", exc)
            return None
        return self.client

    def is_available(self) -> bool:
        return self._ensure_client() is not None or self.sdk_available()

    def normalize_contract(
        self,
        payload: Any,
        *,
        target_id: Optional[str] = None,
        target_kind: str = "document",
        dataset_id: Optional[str] = None,
    ) -> GraphContractClass:
        if isinstance(payload, GraphContractClass):
            return payload
        if isinstance(payload, dict):
            if {"bundle_version", "targets"} <= set(payload.keys()):
                return GraphContractClass.from_dict(dict(payload))
            if target_id is None:
                target_id = str(payload.get("external_id") or payload.get("target_id") or "target")
            return GraphContractClass.from_payload(
                dict(payload),
                target_id=str(target_id),
                target_kind=target_kind,
                dataset_id=dataset_id,
            )
        if isinstance(payload, str):
            return GraphContractClass.from_turtle(
                payload,
                target_id=str(target_id or "turtle-target"),
                target_kind=target_kind,
                dataset_id=dataset_id,
            )
        return GraphContractClass.empty(target_kind=target_kind)

    def create_dataset(
        self,
        payload: Any,
        *,
        name: Optional[str] = None,
        total_pages: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        client = self._ensure_client()
        if client is None:
            return {
                "status": "skipped",
                "reason": "latence_sdk_unavailable",
                "dataset_id": None,
                "name": name,
            }
        try:  # pragma: no cover - requires premium dependency
            service = client.experimental.dataset_intelligence_service
            response = service.run(
                input_data=payload,
                name=name,
                total_pages=total_pages,
                config_overrides=config_overrides,
                return_job=True,
            )
            return {
                "status": getattr(response, "status", "QUEUED"),
                "job_id": getattr(response, "job_id", None),
                "dataset_id": getattr(response, "dataset_id", None),
                "poll_url": getattr(response, "poll_url", None),
            }
        except Exception as exc:
            logger.warning("Latence dataset create failed: %s", exc)
            return {
                "status": "error",
                "reason": str(exc),
                "dataset_id": None,
                "name": name,
            }

    def append_dataset(
        self,
        payload: Any,
        *,
        dataset_id: str,
        total_pages: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        client = self._ensure_client()
        if client is None:
            return {
                "status": "skipped",
                "reason": "latence_sdk_unavailable",
                "dataset_id": dataset_id,
            }
        try:  # pragma: no cover - requires premium dependency
            service = client.experimental.dataset_intelligence_service
            response = service.run(
                input_data=payload,
                dataset_id=dataset_id,
                mode="append",
                total_pages=total_pages,
                config_overrides=config_overrides,
                return_job=True,
            )
            return {
                "status": getattr(response, "status", "QUEUED"),
                "job_id": getattr(response, "job_id", None),
                "dataset_id": getattr(response, "dataset_id", dataset_id),
                "poll_url": getattr(response, "poll_url", None),
                "delta_summary": getattr(response, "delta_summary", None),
            }
        except Exception as exc:
            logger.warning("Latence dataset append failed: %s", exc)
            return {
                "status": "error",
                "reason": str(exc),
                "dataset_id": dataset_id,
            }
