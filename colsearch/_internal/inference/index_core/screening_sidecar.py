from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, runtime_checkable

SCREENING_HEALTH_WARMING = "warming"
SCREENING_HEALTH_HEALTHY = "healthy"
SCREENING_HEALTH_DEGRADED = "degraded"
SCREENING_HEALTH_DISABLED = "disabled"
SCREENING_HEALTH_VALUES = {
    SCREENING_HEALTH_WARMING,
    SCREENING_HEALTH_HEALTHY,
    SCREENING_HEALTH_DEGRADED,
    SCREENING_HEALTH_DISABLED,
}


@dataclass
class ScreeningCalibrationSummary:
    sample_size: int
    top_k: int
    candidate_budget: int
    top1_retention: float
    topk_retention: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScreeningCalibrationSummary":
        return cls(
            sample_size=int(payload.get("sample_size", 0)),
            top_k=int(payload.get("top_k", 0)),
            candidate_budget=int(payload.get("candidate_budget", 0)),
            top1_retention=float(payload.get("top1_retention", 0.0)),
            topk_retention=float(payload.get("topk_retention", 0.0)),
        )


def default_screening_state(
    *,
    health: str,
    reason: str,
    updated_at: Optional[str] = None,
    calibration: Optional[ScreeningCalibrationSummary] = None,
) -> Dict[str, Any]:
    if health not in SCREENING_HEALTH_VALUES:
        raise ValueError(f"Unsupported screening health state: {health!r}")
    payload: Dict[str, Any] = {
        "version": 1,
        "health": health,
        "reason": reason,
        "updated_at": updated_at,
        "calibration": calibration.to_dict() if calibration is not None else None,
    }
    return payload


def normalize_screening_state(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not payload:
        return default_screening_state(
            health=SCREENING_HEALTH_WARMING,
            reason="awaiting_calibration",
        )
    health = str(payload.get("health", SCREENING_HEALTH_WARMING))
    if health not in SCREENING_HEALTH_VALUES:
        health = SCREENING_HEALTH_WARMING
    calibration_payload = payload.get("calibration")
    calibration = (
        ScreeningCalibrationSummary.from_dict(dict(calibration_payload))
        if isinstance(calibration_payload, dict)
        else None
    )
    return default_screening_state(
        health=health,
        reason=str(payload.get("reason", "awaiting_calibration")),
        updated_at=payload.get("updated_at"),
        calibration=calibration,
    )


@runtime_checkable
class ScreeningSidecar(Protocol):
    last_search_profile: Dict[str, Any]

    def rebuild(
        self,
        *,
        doc_ids: Sequence[Any],
        embeddings,
        lengths: Optional[Sequence[int]] = None,
        max_prototypes: Optional[int] = None,
    ) -> None: ...

    def append(
        self,
        *,
        doc_ids: Sequence[Any],
        embeddings,
        lengths: Optional[Sequence[int]] = None,
        max_prototypes: Optional[int] = None,
    ) -> None: ...

    def delete(self, doc_ids: Iterable[Any]) -> int: ...

    def search(
        self,
        query_embedding,
        *,
        top_k: int,
        candidate_budget: Optional[int] = None,
        max_query_prototypes: Optional[int] = None,
        allowed_doc_ids: Optional[Iterable[Any]] = None,
    ) -> list[Any]: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...

    def get_statistics(self) -> Dict[str, Any]: ...
