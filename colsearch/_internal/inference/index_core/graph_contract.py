from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _slug(value: str, *, fallback: str) -> str:
    parts = re.findall(r"[a-z0-9]+", (value or "").lower())
    if not parts:
        return fallback
    return "-".join(parts[:12])


def _clamp01(value: Any, default: float = 1.0) -> float:
    if not isinstance(value, (int, float)):
        return default
    return max(0.0, min(1.0, float(value)))


@dataclass
class GraphNode:
    node_id: str
    label: str
    node_type: str = "entity"
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "aliases": list(self.aliases),
            "confidence": float(self.confidence),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphNode":
        return cls(
            node_id=str(payload.get("node_id") or payload.get("id") or ""),
            label=str(payload.get("label") or payload.get("name") or ""),
            node_type=str(payload.get("node_type") or payload.get("type") or "entity"),
            aliases=[str(value) for value in list(payload.get("aliases") or []) if str(value).strip()],
            confidence=_clamp01(payload.get("confidence"), default=1.0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str = "related_to"
    confidence: float = 1.0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        if self.source_id <= self.target_id:
            return (self.source_id, self.target_id, self.relation)
        return (self.target_id, self.source_id, self.relation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "confidence": float(self.confidence),
            "weight": float(self.weight),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphEdge":
        return cls(
            source_id=str(payload.get("source_id") or payload.get("source") or ""),
            target_id=str(payload.get("target_id") or payload.get("target") or ""),
            relation=str(payload.get("relation") or payload.get("type") or "related_to"),
            confidence=_clamp01(payload.get("confidence"), default=1.0),
            weight=float(payload.get("weight", 1.0) or 1.0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class GraphCommunity:
    community_id: str
    label: str
    node_ids: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "community_id": self.community_id,
            "label": self.label,
            "node_ids": list(self.node_ids),
            "summary": self.summary,
            "confidence": float(self.confidence),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphCommunity":
        return cls(
            community_id=str(payload.get("community_id") or payload.get("id") or ""),
            label=str(payload.get("label") or payload.get("name") or ""),
            node_ids=[str(value) for value in list(payload.get("node_ids") or payload.get("members") or [])],
            summary=str(payload["summary"]) if payload.get("summary") is not None else None,
            confidence=_clamp01(payload.get("confidence"), default=1.0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class GraphEvidenceLink:
    target_id: str
    node_ids: List[str] = field(default_factory=list)
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "node_ids": list(self.node_ids),
            "score": float(self.score),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphEvidenceLink":
        return cls(
            target_id=str(payload.get("target_id") or ""),
            node_ids=[str(value) for value in list(payload.get("node_ids") or [])],
            score=_clamp01(payload.get("score"), default=1.0),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class GraphTarget:
    target_id: str
    entities: List[str] = field(default_factory=list)
    relations: List[Any] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    scores: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "entities": list(self.entities),
            "relations": list(self.relations),
            "concepts": list(self.concepts),
            "scores": dict(self.scores),
            "constraints": dict(self.constraints),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphTarget":
        return cls(
            target_id=str(payload.get("target_id") or payload.get("id") or ""),
            entities=[str(value) for value in list(payload.get("entities") or []) if str(value).strip()],
            relations=list(payload.get("relations") or []),
            concepts=[str(value) for value in list(payload.get("concepts") or []) if str(value).strip()],
            scores=dict(payload.get("scores") or {}),
            constraints=dict(payload.get("constraints") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


class GraphContractClass:
    """Canonical Latence graph contract with Turtle-friendly ingestion."""

    def __init__(
        self,
        *,
        bundle_version: str = "1",
        target_kind: str = "document",
        targets: Optional[List[GraphTarget]] = None,
        nodes: Optional[List[GraphNode]] = None,
        edges: Optional[List[GraphEdge]] = None,
        communities: Optional[List[GraphCommunity]] = None,
        evidence_links: Optional[List[GraphEvidenceLink]] = None,
        dataset_id: Optional[str] = None,
        contract_format: str = "turtle",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bundle_version = str(bundle_version or "1")
        self.target_kind = str(target_kind or "document")
        self.targets = list(targets or [])
        self.nodes = list(nodes or [])
        self.edges = list(edges or [])
        self.communities = list(communities or [])
        self.evidence_links = list(evidence_links or [])
        self.dataset_id = str(dataset_id) if dataset_id else None
        self.contract_format = str(contract_format or "turtle")
        self.metadata = dict(metadata or {})

    def is_empty(self) -> bool:
        return not (self.targets or self.nodes or self.edges or self.communities or self.evidence_links)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "target_kind": self.target_kind,
            "dataset_id": self.dataset_id,
            "contract_format": self.contract_format,
            "metadata": dict(self.metadata),
            "targets": [target.to_dict() for target in self.targets],
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "communities": [community.to_dict() for community in self.communities],
            "evidence_links": [link.to_dict() for link in self.evidence_links],
        }

    @classmethod
    def empty(cls, *, target_kind: str = "document", contract_format: str = "turtle") -> "GraphContractClass":
        return cls(target_kind=target_kind, contract_format=contract_format)

    @staticmethod
    def _node_id(label: str, *, prefix: str = "entity") -> str:
        return f"{prefix}:{_slug(label, fallback='node')}"

    @classmethod
    def _parse_relation_entry(cls, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            source = str(value.get("source") or value.get("source_id") or "").strip()
            target = str(value.get("target") or value.get("target_id") or "").strip()
            relation = str(value.get("relation") or value.get("type") or "related_to").strip() or "related_to"
            if not source or not target:
                return None
            return {
                "source": source,
                "target": target,
                "relation": relation,
                "confidence": _clamp01(value.get("confidence"), default=1.0),
                "metadata": dict(value.get("metadata") or {}),
            }
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None
        if "->" in raw:
            left, right = raw.split("->", 1)
            return {
                "source": left.strip(),
                "target": right.strip(),
                "relation": "related_to",
                "confidence": 1.0,
                "metadata": {},
            }
        match = re.match(
            r"^\s*(?P<source>.+?)\s+(?P<relation>related_to|depends_on|causes|owns|part_of|connected_to)\s+(?P<target>.+?)\s*$",
            raw,
            flags=re.IGNORECASE,
        )
        if match:
            return {
                "source": match.group("source").strip(),
                "target": match.group("target").strip(),
                "relation": match.group("relation").strip().lower(),
                "confidence": 1.0,
                "metadata": {},
            }
        return None

    @classmethod
    def _from_target_fields(
        cls,
        *,
        target: GraphTarget,
        bundle_version: str,
        target_kind: str,
        dataset_id: Optional[str],
        contract_format: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GraphContractClass":
        nodes: Dict[str, GraphNode] = {}
        edges: Dict[tuple[str, str, str], GraphEdge] = {}
        communities: Dict[str, GraphCommunity] = {}
        evidence_node_ids: List[str] = []

        for entity in target.entities:
            node_id = cls._node_id(entity, prefix="entity")
            nodes[node_id] = GraphNode(node_id=node_id, label=entity, node_type="entity")
            evidence_node_ids.append(node_id)

        for concept in target.concepts:
            community_id = f"community:{_slug(concept, fallback='concept')}"
            concept_node_id = cls._node_id(concept, prefix="concept")
            nodes.setdefault(concept_node_id, GraphNode(node_id=concept_node_id, label=concept, node_type="concept"))
            evidence_node_ids.append(concept_node_id)
            communities[community_id] = GraphCommunity(
                community_id=community_id,
                label=concept,
                node_ids=[*evidence_node_ids, concept_node_id],
                metadata={"target_ids": [target.target_id]},
            )

        for relation in target.relations:
            parsed = cls._parse_relation_entry(relation)
            if parsed is None:
                continue
            source_id = cls._node_id(parsed["source"], prefix="entity")
            target_id = cls._node_id(parsed["target"], prefix="entity")
            nodes.setdefault(source_id, GraphNode(node_id=source_id, label=parsed["source"], node_type="entity"))
            nodes.setdefault(target_id, GraphNode(node_id=target_id, label=parsed["target"], node_type="entity"))
            edge = GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relation=str(parsed["relation"] or "related_to"),
                confidence=_clamp01(parsed.get("confidence"), default=1.0),
                metadata=dict(parsed.get("metadata") or {}),
            )
            edges[edge.key()] = edge
            evidence_node_ids.extend([source_id, target_id])

        evidence_node_ids = list(dict.fromkeys(evidence_node_ids))
        confidence = _clamp01(
            target.scores.get("confidence", target.metadata.get("ontology_confidence")),
            default=target.metadata.get("ontology_confidence", 1.0),
        )
        evidence_score = _clamp01(target.scores.get("relevance"), default=confidence)
        evidence_links = [
            GraphEvidenceLink(
                target_id=target.target_id,
                node_ids=evidence_node_ids,
                score=evidence_score,
                metadata={
                    "ontology_confidence": confidence,
                    "internal_id": target.metadata.get("internal_id"),
                    "external_id": target.metadata.get("external_id", target.target_id),
                },
            )
        ]
        return cls(
            bundle_version=bundle_version,
            target_kind=target_kind,
            targets=[target],
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            communities=list(communities.values()),
            evidence_links=evidence_links,
            dataset_id=dataset_id,
            contract_format=contract_format,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_ontology_sidecar(cls, payload: Dict[str, Any]) -> "GraphContractClass":
        bundle_version = str(payload.get("bundle_version") or "1")
        target_kind = str(payload.get("target_kind") or payload.get("kind") or "document")
        dataset_id = str(payload.get("dataset_id")) if payload.get("dataset_id") else None
        contract_format = str(payload.get("contract_format") or "json")
        metadata = dict(payload.get("metadata") or {})

        contracts: List[GraphContractClass] = []
        for raw_target in list(payload.get("targets") or []):
            target = GraphTarget.from_dict(dict(raw_target or {}))
            if not target.target_id:
                continue
            contracts.append(
                cls._from_target_fields(
                    target=target,
                    bundle_version=bundle_version,
                    target_kind=target_kind,
                    dataset_id=dataset_id,
                    contract_format=contract_format,
                    metadata=metadata,
                )
            )
        return cls.merge_many(contracts, target_kind=target_kind, contract_format=contract_format, dataset_id=dataset_id)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphContractClass":
        if payload.get("targets") and not payload.get("nodes") and not payload.get("edges"):
            return cls.from_ontology_sidecar(payload)
        return cls(
            bundle_version=str(payload.get("bundle_version") or "1"),
            target_kind=str(payload.get("target_kind") or "document"),
            targets=[GraphTarget.from_dict(dict(item or {})) for item in list(payload.get("targets") or [])],
            nodes=[GraphNode.from_dict(dict(item or {})) for item in list(payload.get("nodes") or [])],
            edges=[GraphEdge.from_dict(dict(item or {})) for item in list(payload.get("edges") or [])],
            communities=[
                GraphCommunity.from_dict(dict(item or {})) for item in list(payload.get("communities") or [])
            ],
            evidence_links=[
                GraphEvidenceLink.from_dict(dict(item or {})) for item in list(payload.get("evidence_links") or [])
            ],
            dataset_id=str(payload.get("dataset_id")) if payload.get("dataset_id") else None,
            contract_format=str(payload.get("contract_format") or payload.get("format") or "json"),
            metadata=dict(payload.get("metadata") or {}),
        )

    @staticmethod
    def _turtle_token_to_text(value: str) -> str:
        raw = str(value or "").strip().rstrip(".").strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        if ":" in raw and not raw.startswith("http"):
            raw = raw.split(":", 1)[1]
        return raw.replace("_", " ").replace("-", " ").strip()

    @classmethod
    def from_turtle(
        cls,
        text: str,
        *,
        target_id: Optional[str] = None,
        target_kind: str = "document",
        dataset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GraphContractClass":
        stripped = str(text or "").strip()
        if not stripped:
            return cls.empty(target_kind=target_kind)
        try:
            import rdflib  # type: ignore

            graph = rdflib.Graph()
            graph.parse(data=stripped, format="turtle")
            nodes: Dict[str, GraphNode] = {}
            edges: Dict[tuple[str, str, str], GraphEdge] = {}
            for subject, predicate, obj in graph:
                subject_text = cls._turtle_token_to_text(subject.n3())
                predicate_text = cls._turtle_token_to_text(predicate.n3()) or "related_to"
                subject_id = cls._node_id(subject_text, prefix="entity")
                nodes.setdefault(subject_id, GraphNode(node_id=subject_id, label=subject_text, node_type="entity"))
                if getattr(obj, "language", None) is not None or getattr(obj, "datatype", None) is not None or obj.n3().startswith('"'):
                    if predicate_text in {"label", "name"}:
                        nodes[subject_id].label = cls._turtle_token_to_text(obj.n3()) or nodes[subject_id].label
                    continue
                object_text = cls._turtle_token_to_text(obj.n3())
                object_id = cls._node_id(object_text, prefix="entity")
                nodes.setdefault(object_id, GraphNode(node_id=object_id, label=object_text, node_type="entity"))
                edge = GraphEdge(source_id=subject_id, target_id=object_id, relation=predicate_text)
                edges[edge.key()] = edge
            tgt = GraphTarget(target_id=str(target_id or "turtle-target"), metadata=dict(metadata or {}))
            return cls(
                target_kind=target_kind,
                targets=[tgt],
                nodes=list(nodes.values()),
                edges=list(edges.values()),
                evidence_links=[
                    GraphEvidenceLink(
                        target_id=tgt.target_id,
                        node_ids=[node.node_id for node in nodes.values()],
                        metadata=dict(metadata or {}),
                    )
                ],
                dataset_id=dataset_id,
                contract_format="turtle",
                metadata=dict(metadata or {}),
            )
        except Exception:
            nodes: Dict[str, GraphNode] = {}
            edges: Dict[tuple[str, str, str], GraphEdge] = {}
            for line in stripped.splitlines():
                candidate = line.strip()
                if not candidate or candidate.startswith("#") or candidate.startswith("@prefix"):
                    continue
                match = re.match(r"^(?P<subject>\S+)\s+(?P<predicate>\S+)\s+(?P<object>.+?)\s*\.\s*$", candidate)
                if not match:
                    continue
                subject_text = cls._turtle_token_to_text(match.group("subject"))
                predicate_text = cls._turtle_token_to_text(match.group("predicate")) or "related_to"
                object_text = cls._turtle_token_to_text(match.group("object"))
                subject_id = cls._node_id(subject_text, prefix="entity")
                nodes.setdefault(subject_id, GraphNode(node_id=subject_id, label=subject_text, node_type="entity"))
                if predicate_text in {"label", "name"}:
                    nodes[subject_id].label = object_text or nodes[subject_id].label
                    continue
                object_id = cls._node_id(object_text, prefix="entity")
                nodes.setdefault(object_id, GraphNode(node_id=object_id, label=object_text, node_type="entity"))
                edge = GraphEdge(source_id=subject_id, target_id=object_id, relation=predicate_text)
                edges[edge.key()] = edge
            tgt = GraphTarget(target_id=str(target_id or "turtle-target"), metadata=dict(metadata or {}))
            return cls(
                target_kind=target_kind,
                targets=[tgt],
                nodes=list(nodes.values()),
                edges=list(edges.values()),
                evidence_links=[
                    GraphEvidenceLink(
                        target_id=tgt.target_id,
                        node_ids=[node.node_id for node in nodes.values()],
                        metadata=dict(metadata or {}),
                    )
                ],
                dataset_id=dataset_id,
                contract_format="turtle",
                metadata=dict(metadata or {}),
            )

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        target_id: str,
        target_kind: str = "document",
        dataset_id: Optional[str] = None,
    ) -> "GraphContractClass":
        graph_payload = payload.get("latence_graph_contract") or payload.get("graph_contract")
        if isinstance(graph_payload, dict):
            contract = cls.from_dict(dict(graph_payload))
            for target in contract.targets:
                target.metadata.setdefault("external_id", target.metadata.get("external_id", target_id))
            return contract
        turtle_payload = payload.get("latence_graph_turtle") or payload.get("graph_turtle")
        if isinstance(turtle_payload, str) and turtle_payload.strip():
            return cls.from_turtle(
                turtle_payload,
                target_id=target_id,
                target_kind=target_kind,
                dataset_id=dataset_id,
                metadata={
                    "internal_id": payload.get("internal_id"),
                    "external_id": payload.get("external_id", target_id),
                },
            )

        entities = [str(value).strip() for value in list(payload.get("ontology_terms") or payload.get("entities") or [])]
        concepts = [
            str(value).strip()
            for value in list(payload.get("concepts") or payload.get("ontology_concepts") or payload.get("ontology_labels") or [])
        ]
        relations = list(
            payload.get("graph_relations")
            or payload.get("ontology_relations")
            or payload.get("relations")
            or payload.get("graph_edges")
            or []
        )
        if not any([entities, concepts, relations]):
            return cls.empty(target_kind=target_kind)
        target = GraphTarget(
            target_id=str(target_id),
            entities=[value for value in entities if value],
            relations=relations,
            concepts=[value for value in concepts if value],
            scores={
                "relevance": float(payload.get("graph_relevance", payload.get("ontology_confidence", 1.0)) or 1.0),
                "confidence": float(payload.get("ontology_confidence", 1.0) or 1.0),
            },
            metadata={
                "internal_id": payload.get("internal_id"),
                "external_id": payload.get("external_id", target_id),
                "ontology_relation_density": payload.get("ontology_relation_density"),
                "ontology_concept_density": payload.get("ontology_concept_density"),
                "ontology_match_count": payload.get("ontology_match_count"),
                "ontology_evidence_counts": list(payload.get("ontology_evidence_counts") or []),
            },
        )
        return cls._from_target_fields(
            target=target,
            bundle_version="1",
            target_kind=target_kind,
            dataset_id=dataset_id,
            contract_format="json",
        )

    @classmethod
    def merge_many(
        cls,
        contracts: Iterable["GraphContractClass"],
        *,
        target_kind: str = "document",
        contract_format: str = "json",
        dataset_id: Optional[str] = None,
    ) -> "GraphContractClass":
        merged_targets: Dict[str, GraphTarget] = {}
        merged_nodes: Dict[str, GraphNode] = {}
        merged_edges: Dict[tuple[str, str, str], GraphEdge] = {}
        merged_communities: Dict[str, GraphCommunity] = {}
        merged_evidence: Dict[str, GraphEvidenceLink] = {}
        merged_metadata: Dict[str, Any] = {}
        resolved_dataset_id = dataset_id

        for contract in contracts:
            if contract is None or contract.is_empty():
                continue
            resolved_dataset_id = resolved_dataset_id or contract.dataset_id
            merged_metadata.update(dict(contract.metadata))
            for target in contract.targets:
                merged_targets[target.target_id] = target
            for node in contract.nodes:
                existing = merged_nodes.get(node.node_id)
                if existing is None:
                    merged_nodes[node.node_id] = node
                else:
                    alias_values = list(dict.fromkeys([*existing.aliases, *node.aliases]))
                    existing.aliases = alias_values
                    if len(node.label) > len(existing.label):
                        existing.label = node.label
                    existing.confidence = max(existing.confidence, node.confidence)
                    existing.metadata.update(dict(node.metadata))
            for edge in contract.edges:
                key = edge.key()
                existing_edge = merged_edges.get(key)
                if existing_edge is None:
                    merged_edges[key] = edge
                else:
                    existing_edge.confidence = max(existing_edge.confidence, edge.confidence)
                    existing_edge.weight = max(existing_edge.weight, edge.weight)
                    existing_edge.metadata.update(dict(edge.metadata))
            for community in contract.communities:
                existing_community = merged_communities.get(community.community_id)
                if existing_community is None:
                    merged_communities[community.community_id] = community
                else:
                    existing_community.node_ids = list(
                        dict.fromkeys([*existing_community.node_ids, *community.node_ids])
                    )
                    existing_community.confidence = max(existing_community.confidence, community.confidence)
                    existing_community.metadata.update(dict(community.metadata))
                    merged_target_ids = list(
                        dict.fromkeys(
                            [
                                *list(existing_community.metadata.get("target_ids") or []),
                                *list(community.metadata.get("target_ids") or []),
                            ]
                        )
                    )
                    if merged_target_ids:
                        existing_community.metadata["target_ids"] = merged_target_ids
            for evidence in contract.evidence_links:
                merged_evidence[evidence.target_id] = evidence

        return cls(
            target_kind=target_kind,
            contract_format=contract_format,
            targets=list(merged_targets.values()),
            nodes=list(merged_nodes.values()),
            edges=list(merged_edges.values()),
            communities=list(merged_communities.values()),
            evidence_links=list(merged_evidence.values()),
            dataset_id=resolved_dataset_id,
            metadata=merged_metadata,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
