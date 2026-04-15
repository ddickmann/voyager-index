from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .graph_contract import (
    GraphCommunity,
    GraphContractClass,
    GraphEdge,
    GraphEvidenceLink,
    GraphNode,
    GraphTarget,
)
from .latence_graph_adapter import LatenceGraphAdapter

logger = logging.getLogger(__name__)

GRAPH_HEALTH_HEALTHY = "healthy"
GRAPH_HEALTH_DEGRADED = "degraded"
GRAPH_HEALTH_DISABLED = "disabled"


def _clamp01(value: Any, default: float = 0.0) -> float:
    if not isinstance(value, (int, float)):
        return default
    return max(0.0, min(1.0, float(value)))


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


@dataclass
class GraphAugmentationResult:
    applied: bool
    reason: str
    graph_results: List[tuple[int, float]] = field(default_factory=list)
    added_candidate_ids: List[int] = field(default_factory=list)
    retrieval_features: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    provenance_by_target: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": bool(self.applied),
            "reason": self.reason,
            "graph_results": [(int(doc_id), float(score)) for doc_id, score in self.graph_results],
            "added_candidate_ids": [int(value) for value in self.added_candidate_ids],
            "retrieval_features": {int(key): dict(value) for key, value in self.retrieval_features.items()},
            "provenance_by_target": {str(key): dict(value) for key, value in self.provenance_by_target.items()},
            "summary": dict(self.summary),
        }


class LatenceGraphSidecar:
    """Optional LatenceAI graph plane backed by a persisted sidecar state."""

    def __init__(self, state_path: Path, adapter: Optional[LatenceGraphAdapter] = None) -> None:
        self.state_path = Path(state_path)
        self.adapter = adapter or LatenceGraphAdapter()
        self.reset()
        self.load()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _default_dataset_name(self) -> str:
        parent = self.state_path.parent
        if parent.name == "hybrid" and parent.parent.name:
            return parent.parent.name
        return parent.name or "voyager-graph"

    def reset(self) -> None:
        self.health = GRAPH_HEALTH_DISABLED
        self.reason = "no_graph_data"
        self.dataset_id = None
        self.target_kind = "document"
        self.contract_format = "turtle"
        self.contract_version = "1"
        self.sync_history = []
        self.last_sync_at: Optional[str] = None
        self.last_successful_sync_at: Optional[str] = None
        self.sync_status = "never"
        self.sync_reason: Optional[str] = None
        self.dataset_job: Dict[str, Any] = {}
        self.target_contracts = {}
        self.targets = {}
        self.nodes = {}
        self.edges = []
        self.communities = {}
        self.evidence_links = {}
        self.target_to_nodes = {}
        self.node_to_targets = {}
        self.target_to_internal = {}
        self.internal_to_target = {}
        self.community_to_targets = {}
        self.target_to_communities = {}
        self.adjacency = {}
        self.node_alias_lookup = {}
        self.last_search_profile = {}
        self._target_contract_cache: Dict[str, GraphContractClass] = {}
        self._node_contributions: Dict[str, Dict[str, GraphNode]] = {}
        self._edge_contributions: Dict[Tuple[str, str, str], Dict[str, GraphEdge]] = {}
        self._edge_materialized: Dict[Tuple[str, str, str], GraphEdge] = {}
        self._community_contributions: Dict[str, Dict[str, GraphCommunity]] = {}

    def load(self) -> None:
        if not self.state_path.exists():
            self.reason = "no_graph_data"
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load Latence graph sidecar from %s: %s", self.state_path, exc)
            self.reset()
            self.health = GRAPH_HEALTH_DEGRADED
            self.reason = f"load_failed:{exc}"
            return
        self.health = str(payload.get("health") or GRAPH_HEALTH_DISABLED)
        self.reason = str(payload.get("reason") or "loaded")
        self.dataset_id = str(payload.get("dataset_id")) if payload.get("dataset_id") else None
        self.target_kind = str(payload.get("target_kind") or "document")
        self.contract_format = str(payload.get("contract_format") or "turtle")
        self.contract_version = str(payload.get("contract_version") or "1")
        self.sync_history = [dict(item or {}) for item in list(payload.get("sync_history") or [])][-20:]
        self.last_sync_at = str(payload.get("last_sync_at")) if payload.get("last_sync_at") else None
        self.last_successful_sync_at = (
            str(payload.get("last_successful_sync_at")) if payload.get("last_successful_sync_at") else None
        )
        self.sync_status = str(payload.get("sync_status") or "never")
        self.sync_reason = str(payload.get("sync_reason")) if payload.get("sync_reason") else None
        self.dataset_job = dict(payload.get("dataset_job") or {})
        self.target_contracts = {
            str(target_id): dict(contract or {})
            for target_id, contract in dict(payload.get("target_contracts") or {}).items()
        }
        self._rebuild_aggregate_graph()
        if self.targets and self.health != GRAPH_HEALTH_DEGRADED:
            self.health = GRAPH_HEALTH_HEALTHY
            self.reason = self.reason if self.reason not in {"not_initialized", "no_graph_data"} else "loaded"
        elif not self.targets and self.health != GRAPH_HEALTH_DEGRADED:
            self.health = GRAPH_HEALTH_DISABLED
            self.reason = "no_graph_data"

    def save(self) -> None:
        if not self.state_path.parent.exists() and not self.state_path.exists():
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "health": self.health,
            "reason": self.reason,
            "dataset_id": self.dataset_id,
            "target_kind": self.target_kind,
            "contract_format": self.contract_format,
            "contract_version": self.contract_version,
            "updated_at": self._now(),
            "last_sync_at": self.last_sync_at,
            "last_successful_sync_at": self.last_successful_sync_at,
            "sync_status": self.sync_status,
            "sync_reason": self.sync_reason,
            "dataset_job": dict(self.dataset_job),
            "sync_history": list(self.sync_history[-20:]),
            "target_contracts": dict(self.target_contracts),
        }
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def close(self) -> None:
        self.save()

    def _record_sync_event(self, action: str, *, changed_targets: Sequence[str]) -> None:
        self.sync_history.append(
            {
                "action": action,
                "dataset_id": self.dataset_id,
                "changed_targets": [str(value) for value in changed_targets],
                "at": self._now(),
            }
        )
        self.sync_history = self.sync_history[-20:]

    def _record_dataset_sync(
        self,
        action: str,
        *,
        changed_targets: Sequence[str],
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved = dict(result or {})
        self.last_sync_at = self._now()
        status = str(resolved.get("status") or "ok").strip().lower()
        if status in {"queued", "running"}:
            normalized_status = status
        elif status in {"completed", "complete", "success", "ok", "delta_applied"}:
            normalized_status = "ok"
        else:
            normalized_status = status
        self.sync_status = normalized_status or "ok"
        self.sync_reason = str(resolved.get("reason")) if resolved.get("reason") else None
        if resolved.get("dataset_id"):
            self.dataset_id = str(resolved["dataset_id"])
        if self.sync_status not in {"error", "failed", "skipped", "never", "local_only_delete"}:
            self.last_successful_sync_at = self.last_sync_at
        self.dataset_job = {
            key: value
            for key, value in {
                "action": action,
                "at": self.last_sync_at,
                "dataset_id": self.dataset_id,
                "status": self.sync_status,
                "reason": self.sync_reason,
                "job_id": resolved.get("job_id"),
                "poll_url": resolved.get("poll_url"),
                "delta_summary": resolved.get("delta_summary"),
                "changed_targets": [str(value) for value in changed_targets],
            }.items()
            if value not in (None, [], {})
        }

    def _dataset_payload_for_targets(self, target_ids: Sequence[str]) -> Dict[str, Any]:
        contracts: List[GraphContractClass] = []
        for target_id in target_ids:
            fragment = self._target_contract_cache.get(str(target_id))
            if fragment is None and str(target_id) in self.target_contracts:
                fragment = GraphContractClass.from_dict(dict(self.target_contracts[str(target_id)]))
            if fragment is not None and not fragment.is_empty():
                contracts.append(fragment)
        merged = GraphContractClass.merge_many(
            contracts,
            target_kind=self.target_kind,
            contract_format=self.contract_format,
            dataset_id=self.dataset_id,
        )
        return merged.to_dict()

    def _sync_dataset_lifecycle(self, action: str, *, changed_targets: Sequence[str]) -> None:
        normalized_targets = [str(value) for value in changed_targets if str(value).strip()]
        if not normalized_targets:
            return
        if action == "delete":
            self._record_dataset_sync(
                action,
                changed_targets=normalized_targets,
                result={
                    "status": "local_only_delete",
                    "reason": "dataset_delete_not_supported",
                    "dataset_id": self.dataset_id,
                },
            )
            return
        if action == "append_dataset":
            self._record_dataset_sync(
                action,
                changed_targets=normalized_targets,
                result={"status": "delta_applied", "dataset_id": self.dataset_id},
            )
            return
        payload = self._dataset_payload_for_targets(normalized_targets)
        if not list(payload.get("targets") or []):
            self._record_dataset_sync(
                action,
                changed_targets=normalized_targets,
                result={"status": "skipped", "reason": "empty_dataset_payload", "dataset_id": self.dataset_id},
            )
            return
        if self.dataset_id:
            result = self.adapter.append_dataset(payload, dataset_id=self.dataset_id)
        else:
            result = self.adapter.create_dataset(payload, name=self._default_dataset_name())
        self._record_dataset_sync(action, changed_targets=normalized_targets, result=result)

    def _target_contract_fragment(self, contract: GraphContractClass, target_id: str) -> GraphContractClass:
        target = next((item for item in contract.targets if item.target_id == target_id), None)
        if target is None:
            return GraphContractClass.empty(target_kind=contract.target_kind, contract_format=contract.contract_format)
        evidence = next((item for item in contract.evidence_links if item.target_id == target_id), None)
        node_ids = set(evidence.node_ids if evidence is not None else [])
        if not node_ids:
            for entity in target.entities:
                node_ids.add(GraphContractClass._node_id(entity, prefix="entity"))
            for concept in target.concepts:
                node_ids.add(GraphContractClass._node_id(concept, prefix="concept"))
        nodes = [node for node in contract.nodes if node.node_id in node_ids]
        edges = [edge for edge in contract.edges if edge.source_id in node_ids or edge.target_id in node_ids]
        for edge in edges:
            node_ids.add(edge.source_id)
            node_ids.add(edge.target_id)
        nodes = [node for node in contract.nodes if node.node_id in node_ids]
        communities = [
            community
            for community in contract.communities
            if node_ids.intersection(set(community.node_ids))
            or target_id in list(community.metadata.get("target_ids") or [])
        ]
        evidence_links = [evidence] if evidence is not None else []
        return GraphContractClass(
            bundle_version=contract.bundle_version,
            target_kind=contract.target_kind,
            targets=[target],
            nodes=nodes,
            edges=edges,
            communities=communities,
            evidence_links=evidence_links,
            dataset_id=contract.dataset_id,
            contract_format=contract.contract_format,
            metadata=dict(contract.metadata),
        )

    def _merge_node_contributions(self, values: Dict[str, GraphNode]) -> GraphNode:
        items = list(values.values())
        base = items[0]
        aliases = list(dict.fromkeys(alias for item in items for alias in [*item.aliases, item.label] if alias))
        metadata: Dict[str, Any] = {}
        for item in items:
            metadata.update(dict(item.metadata))
        best_label = max((item.label for item in items if item.label), key=len, default=base.label)
        return GraphNode(
            node_id=base.node_id,
            label=best_label,
            node_type=base.node_type,
            aliases=[alias for alias in aliases if alias != best_label],
            confidence=max(item.confidence for item in items),
            metadata=metadata,
        )

    def _merge_edge_contributions(self, values: Dict[str, GraphEdge]) -> GraphEdge:
        items = list(values.values())
        base = items[0]
        metadata: Dict[str, Any] = {}
        for item in items:
            metadata.update(dict(item.metadata))
        return GraphEdge(
            source_id=base.source_id,
            target_id=base.target_id,
            relation=base.relation,
            confidence=max(item.confidence for item in items),
            weight=max(item.weight for item in items),
            metadata=metadata,
        )

    def _merge_community_contributions(self, values: Dict[str, GraphCommunity]) -> GraphCommunity:
        items = list(values.values())
        base = items[0]
        metadata: Dict[str, Any] = {}
        target_ids: List[str] = []
        node_ids: List[str] = []
        for item in items:
            metadata.update(dict(item.metadata))
            node_ids.extend(list(item.node_ids))
            target_ids.extend(str(value) for value in list(item.metadata.get("target_ids") or []) if str(value).strip())
        if target_ids:
            metadata["target_ids"] = list(dict.fromkeys(target_ids))
        return GraphCommunity(
            community_id=base.community_id,
            label=max((item.label for item in items if item.label), key=len, default=base.label),
            node_ids=list(dict.fromkeys(node_ids)),
            summary=max((item.summary or "" for item in items), key=len) or None,
            confidence=max(item.confidence for item in items),
            metadata=metadata,
        )

    def _update_materialized_node(self, node_id: str) -> None:
        contributions = self._node_contributions.get(node_id) or {}
        if not contributions:
            self._node_contributions.pop(node_id, None)
            self.nodes.pop(node_id, None)
            return
        self.nodes[node_id] = self._merge_node_contributions(contributions)

    def _update_materialized_edge(self, edge_key: Tuple[str, str, str]) -> None:
        contributions = self._edge_contributions.get(edge_key) or {}
        if not contributions:
            self._edge_contributions.pop(edge_key, None)
            self._edge_materialized.pop(edge_key, None)
            return
        self._edge_materialized[edge_key] = self._merge_edge_contributions(contributions)

    def _update_materialized_community(self, community_id: str) -> None:
        contributions = self._community_contributions.get(community_id) or {}
        if not contributions:
            self._community_contributions.pop(community_id, None)
            self.communities.pop(community_id, None)
            return
        self.communities[community_id] = self._merge_community_contributions(contributions)

    def _refresh_relationship_maps(self) -> None:
        self.edges = sorted(self._edge_materialized.values(), key=lambda edge: edge.key())
        self.target_to_nodes = {}
        self.node_to_targets = {}
        self.target_to_internal = {}
        self.internal_to_target = {}
        self.community_to_targets = {}
        self.target_to_communities = {}
        self.adjacency = {}
        self.node_alias_lookup = {}

        for node in self.nodes.values():
            for alias in [node.label, *node.aliases]:
                normalized = _normalized_text(alias)
                if normalized:
                    self.node_alias_lookup.setdefault(normalized, set()).add(node.node_id)

        for edge in self.edges:
            self.adjacency.setdefault(edge.source_id, []).append(edge)
            self.adjacency.setdefault(edge.target_id, []).append(edge)

        for target_id, target in self.targets.items():
            evidence = self.evidence_links.get(target_id)
            node_ids = set(evidence.node_ids if evidence is not None else [])
            if not node_ids:
                for entity in target.entities:
                    node_id = GraphContractClass._node_id(entity, prefix="entity")
                    if node_id in self.nodes:
                        node_ids.add(node_id)
                for concept in target.concepts:
                    node_id = GraphContractClass._node_id(concept, prefix="concept")
                    if node_id in self.nodes:
                        node_ids.add(node_id)
            self.target_to_nodes[target_id] = node_ids
            for node_id in node_ids:
                self.node_to_targets.setdefault(node_id, set()).add(target_id)
            internal_id = target.metadata.get("internal_id")
            if internal_id is None and evidence is not None:
                internal_id = evidence.metadata.get("internal_id")
            if isinstance(internal_id, int):
                self.target_to_internal[target_id] = int(internal_id)
                self.internal_to_target[int(internal_id)] = target_id
            elif isinstance(internal_id, str) and internal_id.isdigit():
                self.target_to_internal[target_id] = int(internal_id)
                self.internal_to_target[int(internal_id)] = target_id
            elif target_id.isdigit():
                self.target_to_internal[target_id] = int(target_id)
                self.internal_to_target[int(target_id)] = target_id

        for community_id, community in self.communities.items():
            target_ids = {str(value) for value in list(community.metadata.get("target_ids") or []) if str(value).strip()}
            if not target_ids:
                community_node_ids = set(community.node_ids)
                for target_id, node_ids in self.target_to_nodes.items():
                    if community_node_ids.intersection(node_ids):
                        target_ids.add(target_id)
            self.community_to_targets[community_id] = target_ids
            for target_id in target_ids:
                self.target_to_communities.setdefault(target_id, set()).add(community_id)

    def _apply_fragment(
        self,
        fragment: GraphContractClass,
        *,
        persist: bool,
        refresh_maps: bool,
    ) -> List[str]:
        if fragment.is_empty() or not fragment.targets:
            return []
        source_target = fragment.targets[0]
        target_id = str(source_target.target_id)
        if not target_id:
            return []
        self.contract_version = str(fragment.bundle_version or self.contract_version or "1")
        self.target_kind = str(fragment.target_kind or self.target_kind or "document")
        self.contract_format = str(fragment.contract_format or self.contract_format or "json")
        if fragment.dataset_id:
            self.dataset_id = fragment.dataset_id
        if persist:
            self.target_contracts[target_id] = fragment.to_dict()
        self._target_contract_cache[target_id] = fragment
        self.targets[target_id] = source_target

        evidence = next((item for item in fragment.evidence_links if item.target_id == target_id), None)
        if evidence is not None:
            self.evidence_links[target_id] = evidence
        else:
            self.evidence_links.pop(target_id, None)

        for node in fragment.nodes:
            self._node_contributions.setdefault(node.node_id, {})[target_id] = node
            self._update_materialized_node(node.node_id)
        for edge in fragment.edges:
            edge_key = edge.key()
            self._edge_contributions.setdefault(edge_key, {})[target_id] = edge
            self._update_materialized_edge(edge_key)
        for community in fragment.communities:
            self._community_contributions.setdefault(community.community_id, {})[target_id] = community
            self._update_materialized_community(community.community_id)

        if refresh_maps:
            self._refresh_relationship_maps()
        return [target_id]

    def _remove_target_fragment(self, target_id: str, *, persist: bool, refresh_maps: bool) -> bool:
        normalized_target = str(target_id)
        fragment = self._target_contract_cache.get(normalized_target)
        if fragment is None and normalized_target in self.target_contracts:
            fragment = GraphContractClass.from_dict(dict(self.target_contracts[normalized_target]))
        if fragment is None and normalized_target not in self.targets:
            return False
        if persist:
            self.target_contracts.pop(normalized_target, None)
        self._target_contract_cache.pop(normalized_target, None)
        self.targets.pop(normalized_target, None)
        self.evidence_links.pop(normalized_target, None)
        if fragment is not None:
            for node in fragment.nodes:
                contributors = self._node_contributions.get(node.node_id)
                if contributors is not None:
                    contributors.pop(normalized_target, None)
                self._update_materialized_node(node.node_id)
            for edge in fragment.edges:
                edge_key = edge.key()
                contributors = self._edge_contributions.get(edge_key)
                if contributors is not None:
                    contributors.pop(normalized_target, None)
                self._update_materialized_edge(edge_key)
            for community in fragment.communities:
                contributors = self._community_contributions.get(community.community_id)
                if contributors is not None:
                    contributors.pop(normalized_target, None)
                self._update_materialized_community(community.community_id)
        if refresh_maps:
            self._refresh_relationship_maps()
        return True

    def _rebuild_aggregate_graph(self) -> None:
        self.targets = {}
        self.nodes = {}
        self.edges = []
        self.communities = {}
        self.evidence_links = {}
        self.target_to_nodes = {}
        self.node_to_targets = {}
        self.target_to_internal = {}
        self.internal_to_target = {}
        self.community_to_targets = {}
        self.target_to_communities = {}
        self.adjacency = {}
        self.node_alias_lookup = {}
        self._target_contract_cache = {}
        self._node_contributions = {}
        self._edge_contributions = {}
        self._edge_materialized = {}
        self._community_contributions = {}

        for target_id, payload in sorted(self.target_contracts.items()):
            fragment = GraphContractClass.from_dict(dict(payload))
            if not fragment.targets and str(target_id).strip():
                continue
            self._apply_fragment(fragment, persist=False, refresh_maps=False)
        self._refresh_relationship_maps()

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "health": self.health,
            "reason": self.reason,
            "dataset_id": self.dataset_id,
            "target_kind": self.target_kind,
            "contract_format": self.contract_format,
            "contract_version": self.contract_version,
            "num_targets": len(self.targets),
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_communities": len(self.communities),
            "last_sync_at": self.last_sync_at,
            "last_successful_sync_at": self.last_successful_sync_at,
            "sync_status": self.sync_status,
            "sync_reason": self.sync_reason,
            "dataset_job": dict(self.dataset_job),
            "sync_history": list(self.sync_history[-5:]),
        }

    def is_available(self) -> bool:
        return self.health == GRAPH_HEALTH_HEALTHY and bool(self.targets)

    def rebuild_from_records(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        dataset_id: Optional[str] = None,
        target_kind: str = "document",
    ) -> None:
        self.target_contracts = {}
        self._target_contract_cache = {}
        self.dataset_id = dataset_id or self.dataset_id
        self.target_kind = target_kind or self.target_kind
        self.append_records(records, dataset_id=self.dataset_id, target_kind=self.target_kind, action="rebuild")

    def append_records(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        dataset_id: Optional[str] = None,
        target_kind: str = "document",
        action: str = "append",
    ) -> None:
        changed_targets: List[str] = []
        if dataset_id:
            self.dataset_id = dataset_id
        self.target_kind = target_kind or self.target_kind
        for record in records:
            payload = dict(record.get("payload") or {})
            payload.setdefault("internal_id", record.get("internal_id"))
            payload.setdefault("external_id", record.get("external_id") or record.get("target_id"))
            target_id = str(record.get("external_id") or record.get("target_id") or record.get("id") or "")
            if not target_id:
                continue
            contract = self.adapter.normalize_contract(
                payload,
                target_id=target_id,
                target_kind=self.target_kind,
                dataset_id=self.dataset_id,
            )
            if contract.is_empty():
                continue
            for target in contract.targets:
                fragment = self._target_contract_fragment(contract, target.target_id)
                if fragment.is_empty():
                    continue
                if target.target_id in self.target_contracts:
                    self._remove_target_fragment(target.target_id, persist=True, refresh_maps=False)
                self._apply_fragment(fragment, persist=True, refresh_maps=False)
                changed_targets.append(target.target_id)
        self._refresh_relationship_maps()
        if self.targets:
            self.health = GRAPH_HEALTH_HEALTHY
            self.reason = f"{action}_ok"
        else:
            self.health = GRAPH_HEALTH_DISABLED
            self.reason = "no_graph_data"
        self._sync_dataset_lifecycle(action, changed_targets=changed_targets)
        self._record_sync_event(action, changed_targets=changed_targets)
        self.save()

    def append_dataset_delta(
        self,
        payload: Any,
        *,
        dataset_id: str,
        target_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        contract = self.adapter.normalize_contract(payload, target_kind=target_kind or self.target_kind, dataset_id=dataset_id)
        self.dataset_id = dataset_id
        changed_targets: List[str] = []
        for target in contract.targets:
            fragment = self._target_contract_fragment(contract, target.target_id)
            if fragment.is_empty():
                continue
            if target.target_id in self.target_contracts:
                self._remove_target_fragment(target.target_id, persist=True, refresh_maps=False)
            self._apply_fragment(fragment, persist=True, refresh_maps=False)
            changed_targets.append(target.target_id)
        self._refresh_relationship_maps()
        self.health = GRAPH_HEALTH_HEALTHY if self.targets else GRAPH_HEALTH_DISABLED
        self.reason = "append_dataset_ok" if self.targets else "no_graph_data"
        self._sync_dataset_lifecycle("append_dataset", changed_targets=changed_targets)
        self._record_sync_event("append_dataset", changed_targets=changed_targets)
        self.save()
        return {
            "dataset_id": dataset_id,
            "changed_targets": changed_targets,
            "status": "ok",
        }

    def delete(self, target_ids: Iterable[Any]) -> int:
        deleted = 0
        changed_targets: List[str] = []
        for target_id in [str(value) for value in target_ids]:
            if self._remove_target_fragment(target_id, persist=True, refresh_maps=False):
                deleted += 1
                changed_targets.append(target_id)
        self._refresh_relationship_maps()
        if self.targets:
            self.health = GRAPH_HEALTH_HEALTHY
            self.reason = "delete_ok"
        else:
            self.health = GRAPH_HEALTH_DISABLED
            self.reason = "no_graph_data"
        self._sync_dataset_lifecycle("delete", changed_targets=changed_targets)
        self._record_sync_event("delete", changed_targets=changed_targets)
        self.save()
        return deleted

    def _target_id_from_candidate(self, candidate_id: Any) -> Optional[str]:
        if isinstance(candidate_id, int):
            return self.internal_to_target.get(candidate_id, str(candidate_id))
        text = str(candidate_id)
        if text.isdigit():
            return self.internal_to_target.get(int(text), text)
        return text if text in self.targets else None

    def resolve_entities(
        self,
        query_text: str,
        docs: Optional[Sequence[Any]] = None,
        query_payload: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        resolved: List[str] = []
        direct_ids = list((query_payload or {}).get("graph_entity_ids") or [])
        for value in direct_ids:
            text = str(value).strip()
            if text in self.nodes:
                resolved.append(text)

        query_terms: List[str] = []
        for key in ("ontology_terms", "entities", "graph_terms", "graph_entities"):
            query_terms.extend(str(value).strip() for value in list((query_payload or {}).get(key) or []) if str(value).strip())
        normalized_query = _normalized_text(query_text)
        if normalized_query:
            query_terms.append(normalized_query)
        normalized_terms = {_normalized_text(value) for value in query_terms if _normalized_text(value)}
        for term in normalized_terms:
            resolved.extend(sorted(self.node_alias_lookup.get(term, set())))
            if term and normalized_query and term in normalized_query:
                for alias_key, node_ids in self.node_alias_lookup.items():
                    if alias_key and alias_key in normalized_query:
                        resolved.extend(sorted(node_ids))

        if not resolved and docs:
            for candidate in docs:
                target_id = self._target_id_from_candidate(candidate)
                if target_id:
                    resolved.extend(sorted(self.target_to_nodes.get(target_id, set())))
        return list(dict.fromkeys(resolved))

    def expand_local(
        self,
        node_ids: Sequence[str],
        budget: int,
        *,
        max_hops: int = 1,
    ) -> Dict[str, Any]:
        seeds = [node_id for node_id in node_ids if node_id in self.nodes]
        if not seeds or budget <= 0:
            return {"nodes": [], "edges": []}
        visited = self._bfs(seeds, max_hops=max_hops)
        sorted_nodes = sorted(
            (
                (
                    node_id,
                    1.0 / (1.0 + float(data["distance"])),
                    data,
                )
                for node_id, data in visited.items()
                if node_id not in seeds
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:budget]
        node_id_set = {node_id for node_id, _score, _data in sorted_nodes}
        target_ids = sorted(
            {
                target_id
                for node_id in node_id_set
                for target_id in self.node_to_targets.get(node_id, set())
            }
        )
        return {
            "nodes": [self.nodes[node_id].to_dict() for node_id in node_id_set if node_id in self.nodes],
            "edges": [
                edge.to_dict()
                for edge in self.edges
                if edge.source_id in node_id_set or edge.target_id in node_id_set
            ][: max(budget * 2, 1)],
            "node_ids": list(node_id_set),
            "target_ids": target_ids,
        }

    def retrieve_community(self, node_ids: Sequence[str], budget: int) -> List[Dict[str, Any]]:
        if budget <= 0:
            return []
        seeds = {node_id for node_id in node_ids if node_id in self.nodes}
        communities: List[tuple[float, GraphCommunity]] = []
        for community in self.communities.values():
            overlap = len(seeds.intersection(set(community.node_ids)))
            if overlap <= 0:
                continue
            score = (overlap / float(max(len(seeds), 1))) * 0.65 + (community.confidence * 0.35)
            communities.append((score, community))
        communities.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "community_id": community.community_id,
                "label": community.label,
                "summary": community.summary,
                "score": float(score),
                "target_ids": sorted(self.community_to_targets.get(community.community_id, set())),
            }
            for score, community in communities[:budget]
        ]

    def linked_evidence(self, node_ids: Sequence[str], budget: int) -> List[str]:
        if budget <= 0:
            return []
        targets: Dict[str, float] = {}
        for node_id in node_ids:
            for target_id in self.node_to_targets.get(node_id, set()):
                evidence = self.evidence_links.get(target_id)
                targets[target_id] = max(
                    targets.get(target_id, 0.0),
                    _clamp01(getattr(evidence, "score", 0.0), default=0.0),
                )
        ranked = sorted(targets.items(), key=lambda item: item[1], reverse=True)
        return [target_id for target_id, _score in ranked[:budget]]

    def _bfs(self, seed_nodes: Sequence[str], *, max_hops: int) -> Dict[str, Dict[str, Any]]:
        visited: Dict[str, Dict[str, Any]] = {}
        queue: deque[tuple[str, int]] = deque()
        for seed in seed_nodes:
            if seed not in self.nodes:
                continue
            visited[seed] = {"distance": 0, "confidence": 1.0, "parent": None, "relation": None}
            queue.append((seed, 0))
        while queue:
            node_id, distance = queue.popleft()
            if distance >= max_hops:
                continue
            for edge in self.adjacency.get(node_id, []):
                neighbor = edge.target_id if edge.source_id == node_id else edge.source_id
                next_distance = distance + 1
                next_confidence = min(float(visited[node_id]["confidence"]), edge.confidence)
                current = visited.get(neighbor)
                if current is None or next_distance < int(current["distance"]):
                    visited[neighbor] = {
                        "distance": next_distance,
                        "confidence": next_confidence,
                        "parent": node_id,
                        "relation": edge.relation,
                    }
                    queue.append((neighbor, next_distance))
        return visited

    def _path_to_node(self, visited: Dict[str, Dict[str, Any]], node_id: str) -> List[str]:
        path: List[str] = []
        current = node_id
        while current is not None and current in visited:
            path.append(current)
            parent = visited[current].get("parent")
            current = str(parent) if isinstance(parent, str) else None
        return list(reversed(path))

    def _candidate_graph_features(
        self,
        *,
        target_id: str,
        visited: Dict[str, Dict[str, Any]],
        query_text: str,
        query_payload: Optional[Dict[str, Any]],
        explain: bool,
    ) -> Dict[str, Any]:
        node_ids = list(self.target_to_nodes.get(target_id, set()))
        reachable = [node_id for node_id in node_ids if node_id in visited]
        min_distance = min((int(visited[node_id]["distance"]) for node_id in reachable), default=99)
        graph_distance = 0.0 if min_distance == 99 else 1.0 / (1.0 + float(min_distance))
        relation_confidence = 0.0
        if reachable:
            relation_confidence = float(
                sum(float(visited[node_id]["confidence"]) for node_id in reachable) / float(len(reachable))
            )
        support_count = len(reachable)
        evidence = self.evidence_links.get(target_id)
        provenance_strength = _clamp01(getattr(evidence, "score", 0.0), default=0.0)
        target = self.targets.get(target_id)
        label_match = 0.0
        if isinstance(query_payload, dict):
            query_labels = {str(value).strip().lower() for value in list(query_payload.get("ontology_terms") or []) if str(value).strip()}
            query_labels.update(
                str(value).strip().lower() for value in list(query_payload.get("entities") or []) if str(value).strip()
            )
            if target is not None:
                target_terms = {value.strip().lower() for value in target.entities if value.strip()}
                label_match = 1.0 if query_labels.intersection(target_terms) else 0.0
        query_overlap = 0.0
        normalized_query = _normalized_text(query_text)
        if normalized_query and target is not None:
            normalized_terms = {_normalized_text(term) for term in target.entities if _normalized_text(term)}
            if normalized_query in normalized_terms or any(term and term in normalized_query for term in normalized_terms):
                query_overlap = 1.0
        path_coherence = _clamp01(0.55 * graph_distance + 0.30 * relation_confidence + 0.15 * provenance_strength)
        local_score = _clamp01(
            0.35 * graph_distance
            + 0.25 * relation_confidence
            + 0.20 * min(1.0, support_count / 3.0)
            + 0.20 * provenance_strength
        )
        if support_count <= 0 and graph_distance <= 0.0:
            local_score = _clamp01(0.25 * provenance_strength)
        feature_payload: Dict[str, Any] = {
            "graph_distance": graph_distance,
            "graph_relation_confidence": relation_confidence,
            "graph_path_coherence": path_coherence,
            "graph_support_count": float(support_count),
            "graph_provenance_strength": provenance_strength,
            "graph_query_overlap": query_overlap,
            "graph_type_match": label_match,
            "graph_local_score": local_score,
            "graph_community_score": 0.0,
            "graph_score": local_score,
            "graph_reason": "graph_local" if local_score > 0.0 else "graph_unscored",
            "graph_lanes": ["graph_local"] if local_score > 0.0 else [],
        }
        if explain and reachable:
            best_node = min(reachable, key=lambda node_id: int(visited[node_id]["distance"]))
            feature_payload["graph_path_node_ids"] = self._path_to_node(visited, best_node)
            feature_payload["graph_path_labels"] = [
                self.nodes[node_id].label for node_id in feature_payload["graph_path_node_ids"] if node_id in self.nodes
            ]
        return feature_payload

    def graph_features(
        self,
        query_text: str,
        candidate_ids: Sequence[Any],
        *,
        query_payload: Optional[Dict[str, Any]] = None,
        max_hops: int = 2,
        explain: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        target_ids = [self._target_id_from_candidate(candidate_id) for candidate_id in candidate_ids]
        resolved_entities = self.resolve_entities(query_text, candidate_ids, query_payload=query_payload)
        if not resolved_entities:
            return {}
        visited = self._bfs(resolved_entities, max_hops=max_hops)
        features: Dict[int, Dict[str, Any]] = {}
        for candidate_id, target_id in zip(candidate_ids, target_ids):
            if target_id is None:
                continue
            internal_id = self.target_to_internal.get(target_id)
            if internal_id is None:
                if isinstance(candidate_id, int):
                    internal_id = candidate_id
                elif str(candidate_id).isdigit():
                    internal_id = int(str(candidate_id))
            if internal_id is None:
                continue
            features[int(internal_id)] = self._candidate_graph_features(
                target_id=target_id,
                visited=visited,
                query_text=query_text,
                query_payload=query_payload,
                explain=explain,
            )
        return features

    @staticmethod
    def _merge_feature_payload(
        existing: Optional[Dict[str, Any]],
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(existing or {})
        merged.update(dict(updates))
        lanes = sorted({str(value) for value in list(merged.get("graph_lanes") or []) if str(value).strip()})
        if lanes:
            merged["graph_lanes"] = lanes
        if float(merged.get("graph_local_score", 0.0) or 0.0) <= 0.0 and float(
            merged.get("graph_community_score", 0.0) or 0.0
        ) > 0.0:
            merged["graph_reason"] = "graph_community"
        elif lanes:
            merged["graph_reason"] = lanes[0]
        merged["graph_score"] = _clamp01(
            0.60 * float(merged.get("graph_local_score", 0.0) or 0.0)
            + 0.30 * float(merged.get("graph_community_score", 0.0) or 0.0)
            + 0.10 * float(merged.get("graph_provenance_strength", 0.0) or 0.0)
        )
        return merged

    def augment_candidates(
        self,
        candidate_ids: Sequence[Any],
        *,
        query_text: str = "",
        query_payload: Optional[Dict[str, Any]] = None,
        local_budget: int = 4,
        community_budget: int = 4,
        evidence_budget: int = 8,
        max_hops: int = 2,
        explain: bool = False,
    ) -> GraphAugmentationResult:
        if not self.is_available():
            return GraphAugmentationResult(applied=False, reason="graph_unavailable")

        target_ids = [target_id for target_id in (self._target_id_from_candidate(value) for value in candidate_ids) if target_id]
        resolved_entities = self.resolve_entities(query_text, candidate_ids, query_payload=query_payload)
        if not resolved_entities:
            return GraphAugmentationResult(applied=False, reason="no_query_entities")

        visited = self._bfs(resolved_entities, max_hops=max_hops)
        retrieval_features: Dict[int, Dict[str, Any]] = {}
        provenance_by_target: Dict[str, Dict[str, Any]] = {}
        target_scores: Dict[str, Dict[str, Any]] = {}

        def _ensure_target_features(target_id: str) -> Dict[str, Any]:
            normalized_target = str(target_id)
            feature_payload = target_scores.get(normalized_target)
            if feature_payload is None:
                feature_payload = self._candidate_graph_features(
                    target_id=normalized_target,
                    visited=visited,
                    query_text=query_text,
                    query_payload=query_payload,
                    explain=explain,
                )
                target_scores[normalized_target] = dict(feature_payload)
            return target_scores[normalized_target]

        def _update_provenance(target_id: str, features: Dict[str, Any]) -> None:
            normalized_target = str(target_id)
            record = provenance_by_target.setdefault(
                normalized_target,
                {
                    "reason": features.get("graph_reason", "graph_local"),
                    "lanes": list(features.get("graph_lanes") or []),
                    "graph_score": 0.0,
                    "community_ids": [],
                    "support_count": features.get("graph_support_count", 0.0),
                },
            )
            record["reason"] = features.get("graph_reason", record.get("reason", "graph_local"))
            record["lanes"] = sorted(
                {str(value) for value in [*list(record.get("lanes") or []), *list(features.get("graph_lanes") or [])]}
            )
            record["graph_score"] = float(features.get("graph_score", 0.0) or 0.0)
            record["support_count"] = float(features.get("graph_support_count", record.get("support_count", 0.0)) or 0.0)
            if explain and "graph_path_labels" in features:
                record["path"] = list(features["graph_path_labels"])

        base_target_ids = [str(target_id) for target_id in target_ids]
        base_target_set = set(base_target_ids)
        for target_id in base_target_ids:
            features = _ensure_target_features(target_id)
            _update_provenance(target_id, features)

        local_expansion = self.expand_local(resolved_entities, local_budget, max_hops=max_hops)
        local_node_ids = [str(value) for value in list(local_expansion.get("node_ids") or [])]
        local_target_scores: Dict[str, float] = {}
        for target_id in list(local_expansion.get("target_ids") or []):
            features = _ensure_target_features(str(target_id))
            features["graph_lanes"] = sorted({*list(features.get("graph_lanes") or []), "graph_local"})
            features["graph_reason"] = "graph_local"
            merged = self._merge_feature_payload(target_scores.get(str(target_id)), features)
            target_scores[str(target_id)] = merged
            _update_provenance(str(target_id), merged)
            local_target_scores[str(target_id)] = float(merged.get("graph_local_score", 0.0) or 0.0)

        evidence_seed_nodes = list(dict.fromkeys([*resolved_entities, *local_node_ids]))
        evidence_targets = self.linked_evidence(evidence_seed_nodes, evidence_budget)
        evidence_target_scores: Dict[str, float] = {}
        for target_id in evidence_targets:
            features = _ensure_target_features(str(target_id))
            features["graph_lanes"] = sorted({*list(features.get("graph_lanes") or []), "graph_local"})
            features["graph_reason"] = "graph_local"
            merged = self._merge_feature_payload(target_scores.get(str(target_id)), features)
            target_scores[str(target_id)] = merged
            _update_provenance(str(target_id), merged)
            evidence_target_scores[str(target_id)] = float(merged.get("graph_provenance_strength", 0.0) or 0.0)

        community_payloads = self.retrieve_community(resolved_entities, community_budget)
        community_target_scores: Dict[str, float] = {}
        for community_payload in community_payloads:
            community_id = str(community_payload["community_id"])
            community_score = _clamp01(community_payload["score"], default=0.0)
            for target_id in community_payload.get("target_ids", []):
                normalized_target = str(target_id)
                base_features = _ensure_target_features(normalized_target)
                community_features = dict(base_features)
                community_features["graph_community_score"] = max(
                    float(community_features.get("graph_community_score", 0.0) or 0.0),
                    community_score,
                )
                community_features["graph_lanes"] = sorted(
                    {str(value) for value in [*list(community_features.get("graph_lanes") or []), "graph_community"]}
                )
                if float(community_features.get("graph_local_score", 0.0) or 0.0) <= 0.0:
                    community_features["graph_reason"] = "graph_community"
                merged = self._merge_feature_payload(target_scores.get(normalized_target), community_features)
                target_scores[normalized_target] = merged
                community_target_scores[normalized_target] = max(
                    community_target_scores.get(normalized_target, 0.0),
                    float(merged.get("graph_community_score", 0.0) or 0.0),
                )
                record = provenance_by_target.setdefault(
                    normalized_target,
                    {"reason": merged.get("graph_reason", "graph_community"), "lanes": [], "graph_score": 0.0},
                )
                record.setdefault("community_ids", [])
                record["community_ids"] = list(dict.fromkeys([*list(record.get("community_ids") or []), community_id]))
                _update_provenance(normalized_target, merged)

        local_rescued_targets: List[str] = []
        for target_id, _score in sorted(local_target_scores.items(), key=lambda item: item[1], reverse=True):
            if target_id not in base_target_set and target_id not in local_rescued_targets and len(local_rescued_targets) < local_budget:
                local_rescued_targets.append(target_id)

        evidence_rescued_targets: List[str] = []
        for target_id, _score in sorted(evidence_target_scores.items(), key=lambda item: item[1], reverse=True):
            if (
                target_id not in base_target_set
                and target_id not in local_rescued_targets
                and target_id not in evidence_rescued_targets
                and len(evidence_rescued_targets) < evidence_budget
            ):
                evidence_rescued_targets.append(target_id)

        community_rescued_targets: List[str] = []
        for target_id, _score in sorted(community_target_scores.items(), key=lambda item: item[1], reverse=True):
            if (
                target_id not in base_target_set
                and target_id not in local_rescued_targets
                and target_id not in evidence_rescued_targets
                and target_id not in community_rescued_targets
                and len(community_rescued_targets) < community_budget
            ):
                community_rescued_targets.append(target_id)

        rescued_target_ids = set(local_rescued_targets) | set(evidence_rescued_targets) | set(community_rescued_targets)

        ranked_targets = sorted(
            (
                (
                    target_id,
                    _clamp01(features.get("graph_score"), default=0.0),
                    features,
                )
                for target_id, features in target_scores.items()
                if _clamp01(features.get("graph_score"), default=0.0) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        added_candidate_ids: List[int] = []
        graph_results: List[tuple[int, float]] = []
        for target_id, score, features in ranked_targets:
            internal_id = self.target_to_internal.get(target_id)
            if internal_id is None:
                continue
            graph_results.append((internal_id, score))
            retrieval_features[int(internal_id)] = dict(features)
            if target_id in rescued_target_ids:
                added_candidate_ids.append(int(internal_id))

        graph_results = sorted(graph_results, key=lambda item: item[1], reverse=True)
        self.last_search_profile = {
            "graph_available": True,
            "graph_applied": bool(graph_results),
            "seed_nodes": list(resolved_entities),
            "base_candidate_count": len(candidate_ids),
            "base_target_count": len(base_target_set),
            "graph_result_count": len(graph_results),
            "added_candidate_count": len(added_candidate_ids),
            "local_budget": int(local_budget),
            "community_budget": int(community_budget),
            "evidence_budget": int(evidence_budget),
            "max_hops": int(max_hops),
            "local_node_count": len(local_node_ids),
            "community_count": len(community_payloads),
            "local_rescue_count": len(local_rescued_targets),
            "community_rescue_count": len(community_rescued_targets),
            "evidence_rescue_count": len(evidence_rescued_targets),
            "local_target_ids": list(local_rescued_targets),
            "community_target_ids": list(community_rescued_targets),
            "evidence_target_ids": list(evidence_rescued_targets),
            "base_target_ids": list(base_target_ids),
            "invoked_after_first_stage": True,
            "merge_mode": "additive",
            "query_text": query_text,
        }
        return GraphAugmentationResult(
            applied=bool(graph_results or retrieval_features),
            reason="ok" if graph_results else "no_graph_matches",
            graph_results=graph_results,
            added_candidate_ids=added_candidate_ids,
            retrieval_features=retrieval_features,
            provenance_by_target=provenance_by_target,
            summary=dict(self.last_search_profile),
        )
