from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


RELATION_CUES = (
    "related",
    "depends",
    "dependency",
    "impact",
    "why",
    "cause",
    "caused",
    "connected",
    "relationship",
    "lineage",
    "owner",
    "trace",
    "policy",
    "compliance",
    "incident",
    "root cause",
)

KNOWN_ITEM_CUES = (
    "faq",
    "manual",
    "guide",
    "documentation",
    "docs",
    "reference",
    "runbook",
    "playbook",
)

MANDATORY_WORKFLOWS = {
    "compliance",
    "lineage",
    "policy_traceability",
    "traceability",
    "knowledge_discovery",
    "graph_copilot",
    "graph_native_copilot",
}

MULTIMODAL_STITCH_KEYS = (
    "evidence_types",
    "source_modalities",
    "stitched_modalities",
    "multimodal_sources",
)


@dataclass
class LatenceGraphDecision:
    applied: bool
    reason: str
    mode: str
    entity_count: int
    relation_cue_count: int
    low_agreement: bool
    low_confidence: bool
    graph_available: bool
    query_class: str = "ordinary"
    trigger_reasons: List[str] | None = None
    mandatory_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": bool(self.applied),
            "reason": self.reason,
            "mode": self.mode,
            "entity_count": int(self.entity_count),
            "relation_cue_count": int(self.relation_cue_count),
            "low_agreement": bool(self.low_agreement),
            "low_confidence": bool(self.low_confidence),
            "graph_available": bool(self.graph_available),
            "query_class": self.query_class,
            "trigger_reasons": list(self.trigger_reasons or []),
            "mandatory_reason": self.mandatory_reason,
        }


class LatenceGraphPolicy:
    """Selective graph invocation policy inspired by LightRAG routing."""

    def __init__(self, relation_cues: Optional[Sequence[str]] = None) -> None:
        self.relation_cues = tuple(str(value).strip().lower() for value in (relation_cues or RELATION_CUES) if str(value).strip())

    @staticmethod
    def _normalized_text(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))

    @classmethod
    def _resolved_query_text(cls, query_text: str, query_payload: Optional[Dict[str, Any]]) -> str:
        if str(query_text or "").strip():
            return str(query_text)
        if not isinstance(query_payload, dict):
            return ""
        for key in ("query_text", "query", "question", "prompt"):
            value = query_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def _query_entities(query_payload: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(query_payload, dict):
            return []
        values: List[str] = []
        for key in ("ontology_terms", "entities", "graph_terms", "graph_entities"):
            for value in list(query_payload.get(key) or []):
                text = str(value).strip()
                if text:
                    values.append(text)
        direct_ids = query_payload.get("graph_entity_ids")
        if isinstance(direct_ids, list):
            values.extend(str(value).strip() for value in direct_ids if str(value).strip())
        return list(dict.fromkeys(values))

    @staticmethod
    def _rrf_agreement(
        dense_results: Sequence[Tuple[int, float]],
        sparse_results: Sequence[Tuple[int, float]],
        *,
        cap: int = 5,
    ) -> float:
        dense_ids = {int(doc_id) for doc_id, _ in list(dense_results)[:cap]}
        sparse_ids = {int(doc_id) for doc_id, _ in list(sparse_results)[:cap]}
        if not dense_ids or not sparse_ids:
            return 1.0
        union = dense_ids | sparse_ids
        if not union:
            return 1.0
        return len(dense_ids & sparse_ids) / float(len(union))

    @staticmethod
    def _low_confidence(
        dense_results: Sequence[Tuple[int, float]],
        sparse_results: Sequence[Tuple[int, float]],
    ) -> bool:
        def _gap(values: Sequence[Tuple[int, float]]) -> float:
            if len(values) < 2:
                return 1.0
            return float(values[0][1]) - float(values[1][1])

        dense_gap = _gap(list(dense_results)[:2])
        sparse_gap = _gap(list(sparse_results)[:2])
        return bool((dense_results and dense_gap < 0.08) or (sparse_results and sparse_gap < 0.08))

    @staticmethod
    def _token_count(query_text: str) -> int:
        return len(re.findall(r"[a-z0-9]+", (query_text or "").lower()))

    @staticmethod
    def _mandatory_reason(query_payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(query_payload, dict):
            return None
        if bool(query_payload.get("graph_required")) or bool(query_payload.get("graph_mandatory")):
            return "request_required"
        normalized_policy = str(
            query_payload.get("tenant_graph_policy")
            or query_payload.get("graph_policy")
            or query_payload.get("graph_route")
            or ""
        ).strip().lower()
        if normalized_policy in {"force", "required", "mandatory"}:
            return "tenant_policy"
        workflow = str(query_payload.get("workflow_type") or query_payload.get("graph_workflow") or "").strip().lower()
        if workflow in MANDATORY_WORKFLOWS:
            return f"workflow:{workflow}"
        return None

    @classmethod
    def _multimodal_stitch_requested(cls, query_payload: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(query_payload, dict):
            return False
        if bool(query_payload.get("multimodal_stitch")) or bool(query_payload.get("graph_multimodal")):
            return True
        for key in MULTIMODAL_STITCH_KEYS:
            values = query_payload.get(key)
            if isinstance(values, list) and len(values) >= 2:
                return True
        return False

    @classmethod
    def _query_class(
        cls,
        *,
        normalized_query: str,
        entity_count: int,
        relation_cue_count: int,
        query_payload: Optional[Dict[str, Any]],
    ) -> str:
        if cls._mandatory_reason(query_payload) is not None:
            return "graph_native"
        if cls._multimodal_stitch_requested(query_payload):
            return "multimodal_stitch"
        if relation_cue_count > 0:
            return "relation"
        token_count = cls._token_count(normalized_query)
        if token_count <= 4 and entity_count <= 1:
            return "known_item"
        if any(cue in normalized_query for cue in KNOWN_ITEM_CUES):
            return "known_item"
        return "ordinary"

    def decide(
        self,
        *,
        graph_mode: str,
        query_text: str,
        query_payload: Optional[Dict[str, Any]],
        dense_results: Sequence[Tuple[int, float]],
        sparse_results: Sequence[Tuple[int, float]],
        graph_available: bool,
    ) -> LatenceGraphDecision:
        normalized_mode = str(graph_mode or "off").strip().lower()
        resolved_query_text = self._resolved_query_text(query_text, query_payload)
        normalized_query = self._normalized_text(resolved_query_text)
        entity_terms = self._query_entities(query_payload)
        relation_cue_count = sum(1 for cue in self.relation_cues if cue in normalized_query)
        low_agreement = self._rrf_agreement(dense_results, sparse_results) < 0.25
        low_confidence = self._low_confidence(dense_results, sparse_results)
        mandatory_reason = self._mandatory_reason(query_payload)
        query_class = self._query_class(
            normalized_query=normalized_query,
            entity_count=len(entity_terms),
            relation_cue_count=relation_cue_count,
            query_payload=query_payload,
        )

        if normalized_mode in {"off", "disabled"}:
            return LatenceGraphDecision(
                applied=False,
                reason="graph_mode_off",
                mode="off",
                entity_count=len(entity_terms),
                relation_cue_count=relation_cue_count,
                low_agreement=low_agreement,
                low_confidence=low_confidence,
                graph_available=graph_available,
                query_class=query_class,
            )
        if not graph_available:
            return LatenceGraphDecision(
                applied=False,
                reason="graph_unavailable",
                mode=normalized_mode or "off",
                entity_count=len(entity_terms),
                relation_cue_count=relation_cue_count,
                low_agreement=low_agreement,
                low_confidence=low_confidence,
                graph_available=False,
                query_class=query_class,
            )
        if normalized_mode == "force" or mandatory_reason is not None:
            return LatenceGraphDecision(
                applied=True,
                reason="forced",
                mode="force",
                entity_count=len(entity_terms),
                relation_cue_count=relation_cue_count,
                low_agreement=low_agreement,
                low_confidence=low_confidence,
                graph_available=True,
                query_class="graph_native" if mandatory_reason is not None else query_class,
                trigger_reasons=["forced"],
                mandatory_reason=mandatory_reason or "graph_mode_force",
            )

        if query_class == "known_item":
            return LatenceGraphDecision(
                applied=False,
                reason="known_item_query",
                mode="auto",
                entity_count=len(entity_terms),
                relation_cue_count=relation_cue_count,
                low_agreement=low_agreement,
                low_confidence=low_confidence,
                graph_available=True,
                query_class=query_class,
            )

        trigger_reasons: List[str] = []
        if len(entity_terms) >= 2:
            trigger_reasons.append("entity_heavy")
        if relation_cue_count > 0:
            trigger_reasons.append("relation_query")
        if low_agreement:
            trigger_reasons.append("low_dense_sparse_agreement")
        if low_confidence:
            trigger_reasons.append("low_first_stage_confidence")
        if self._multimodal_stitch_requested(query_payload):
            trigger_reasons.append("multimodal_stitch")

        should_apply = bool(trigger_reasons)
        return LatenceGraphDecision(
            applied=should_apply,
            reason=trigger_reasons[0] if should_apply else "auto_skip",
            mode="auto",
            entity_count=len(entity_terms),
            relation_cue_count=relation_cue_count,
            low_agreement=low_agreement,
            low_confidence=low_confidence,
            graph_available=True,
            query_class=query_class,
            trigger_reasons=trigger_reasons,
        )
