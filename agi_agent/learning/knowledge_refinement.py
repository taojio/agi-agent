import numpy as np
import time
import hashlib
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque


class KnowledgeType(Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    SEMI_STRUCTURED = "semi_structured"
    PROCEDURAL = "procedural"
    DECLARATIVE = "declarative"


class KnowledgeFormat(Enum):
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    GRAPH = "graph"
    RULE = "rule"
    CODE = "code"
    EMBEDDING = "embedding"


class VersionStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class KnowledgeNode:
    node_id: str
    content: Any
    knowledge_type: KnowledgeType
    format: KnowledgeFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.9
    source: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class KnowledgeVersion:
    version_id: str
    knowledge_id: str
    content: Any
    version_number: int
    status: VersionStatus
    parent_version: Optional[str] = None
    change_description: str = ""
    created_at: float = 0.0
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeRelation:
    relation_id: str
    source_node: str
    target_node: str
    relation_type: str
    weight: float = 1.0
    confidence: float = 0.9


@dataclass
class KnowledgeExtractionResult:
    success: bool
    nodes: List[KnowledgeNode] = field(default_factory=list)
    relations: List[KnowledgeRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_time: float = 0.0
    confidence: float = 0.0


class KnowledgeExtractor:
    def __init__(self):
        self.extractors: Dict[KnowledgeFormat, Callable] = {}
        self._init_extractors()

    def _init_extractors(self):
        self.extractors[KnowledgeFormat.TEXT] = self._extract_from_text
        self.extractors[KnowledgeFormat.JSON] = self._extract_from_json
        self.extractors[KnowledgeFormat.CSV] = self._extract_from_csv
        self.extractors[KnowledgeFormat.RULE] = self._extract_from_rule

    def _extract_from_text(self, text: str) -> KnowledgeExtractionResult:
        start_time = time.time()
        nodes = []

        sentences = text.replace('\n', ' ').split('.')
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:
                node = KnowledgeNode(
                    node_id=f"text_{int(time.time() * 1000)}_{i}",
                    content=sentence,
                    knowledge_type=KnowledgeType.UNSTRUCTURED,
                    format=KnowledgeFormat.TEXT,
                    metadata={"sentence_index": i},
                    confidence=0.85,
                    source="text_extraction",
                    created_at=time.time()
                )
                nodes.append(node)

        relations = []
        for i in range(len(nodes) - 1):
            relations.append(KnowledgeRelation(
                relation_id=f"rel_{int(time.time() * 1000)}_{i}",
                source_node=nodes[i].node_id,
                target_node=nodes[i + 1].node_id,
                relation_type="follows",
                weight=0.8
            ))

        return KnowledgeExtractionResult(
            success=True,
            nodes=nodes,
            relations=relations,
            extraction_time=time.time() - start_time,
            confidence=0.85
        )

    def _extract_from_json(self, json_str: str) -> KnowledgeExtractionResult:
        start_time = time.time()
        nodes = []
        relations = []

        try:
            data = json.loads(json_str)
            self._extract_json_recursive(data, "root", nodes, relations, 0)
        except Exception:
            return KnowledgeExtractionResult(success=False, extraction_time=time.time() - start_time)

        return KnowledgeExtractionResult(
            success=True,
            nodes=nodes,
            relations=relations,
            extraction_time=time.time() - start_time,
            confidence=0.95
        )

    def _extract_json_recursive(self, data: Any, path: str, nodes: List[KnowledgeNode],
                                relations: List[KnowledgeRelation], depth: int):
        if depth > 10:
            return

        node_id = f"json_{hashlib.md5(path.encode()).hexdigest()[:8]}_{int(time.time() * 1000)}"

        if isinstance(data, dict):
            node = KnowledgeNode(
                node_id=node_id,
                content=list(data.keys()),
                knowledge_type=KnowledgeType.STRUCTURED,
                format=KnowledgeFormat.JSON,
                metadata={"path": path, "type": "object"},
                confidence=0.95,
                source="json_extraction",
                created_at=time.time()
            )
            nodes.append(node)

            for key, value in data.items():
                child_path = f"{path}.{key}"
                self._extract_json_recursive(value, child_path, nodes, relations, depth + 1)

                child_id = f"json_{hashlib.md5(child_path.encode()).hexdigest()[:8]}_{int(time.time() * 1000)}"
                relations.append(KnowledgeRelation(
                    relation_id=f"rel_{int(time.time() * 1000)}_{len(relations)}",
                    source_node=node_id,
                    target_node=child_id,
                    relation_type="contains",
                    weight=0.9
                ))

        elif isinstance(data, list):
            node = KnowledgeNode(
                node_id=node_id,
                content={"length": len(data), "type": "array"},
                knowledge_type=KnowledgeType.STRUCTURED,
                format=KnowledgeFormat.JSON,
                metadata={"path": path, "type": "array"},
                confidence=0.95,
                source="json_extraction",
                created_at=time.time()
            )
            nodes.append(node)

        elif isinstance(data, (str, int, float, bool)):
            node = KnowledgeNode(
                node_id=node_id,
                content=data,
                knowledge_type=KnowledgeType.DECLARATIVE,
                format=KnowledgeFormat.JSON,
                metadata={"path": path, "type": type(data).__name__},
                confidence=0.98,
                source="json_extraction",
                created_at=time.time()
            )
            nodes.append(node)

    def _extract_from_csv(self, csv_str: str) -> KnowledgeExtractionResult:
        start_time = time.time()
        nodes = []

        lines = csv_str.strip().split('\n')
        if len(lines) < 2:
            return KnowledgeExtractionResult(success=False, extraction_time=time.time() - start_time)

        headers = lines[0].split(',')
        headers = [h.strip() for h in headers]

        header_node = KnowledgeNode(
            node_id=f"csv_header_{int(time.time() * 1000)}",
            content=headers,
            knowledge_type=KnowledgeType.STRUCTURED,
            format=KnowledgeFormat.CSV,
            metadata={"column_count": len(headers)},
            confidence=0.95,
            source="csv_extraction",
            created_at=time.time()
        )
        nodes.append(header_node)

        for i, line in enumerate(lines[1:]):
            values = line.split(',')
            values = [v.strip() for v in values]

            row_node = KnowledgeNode(
                node_id=f"csv_row_{int(time.time() * 1000)}_{i}",
                content=dict(zip(headers, values)),
                knowledge_type=KnowledgeType.SEMI_STRUCTURED,
                format=KnowledgeFormat.CSV,
                metadata={"row_index": i, "header_node": header_node.node_id},
                confidence=0.9,
                source="csv_extraction",
                created_at=time.time()
            )
            nodes.append(row_node)

        return KnowledgeExtractionResult(
            success=True,
            nodes=nodes,
            extraction_time=time.time() - start_time,
            confidence=0.92
        )

    def _extract_from_rule(self, rule_str: str) -> KnowledgeExtractionResult:
        start_time = time.time()
        nodes = []

        rule_parts = rule_str.split('->')
        if len(rule_parts) == 2:
            condition = rule_parts[0].strip()
            action = rule_parts[1].strip()

            condition_node = KnowledgeNode(
                node_id=f"rule_cond_{int(time.time() * 1000)}",
                content=condition,
                knowledge_type=KnowledgeType.PROCEDURAL,
                format=KnowledgeFormat.RULE,
                metadata={"type": "condition"},
                confidence=0.9,
                source="rule_extraction",
                created_at=time.time()
            )
            nodes.append(condition_node)

            action_node = KnowledgeNode(
                node_id=f"rule_action_{int(time.time() * 1000)}",
                content=action,
                knowledge_type=KnowledgeType.PROCEDURAL,
                format=KnowledgeFormat.RULE,
                metadata={"type": "action"},
                confidence=0.9,
                source="rule_extraction",
                created_at=time.time()
            )
            nodes.append(action_node)

            relations = [KnowledgeRelation(
                relation_id=f"rel_rule_{int(time.time() * 1000)}",
                source_node=condition_node.node_id,
                target_node=action_node.node_id,
                relation_type="implies",
                weight=0.95
            )]

            return KnowledgeExtractionResult(
                success=True,
                nodes=nodes,
                relations=relations,
                extraction_time=time.time() - start_time,
                confidence=0.9
            )

        return KnowledgeExtractionResult(
            success=False,
            extraction_time=time.time() - start_time
        )

    def extract(self, content: str, format_type: KnowledgeFormat) -> KnowledgeExtractionResult:
        extractor = self.extractors.get(format_type)
        if extractor:
            return extractor(content)
        return KnowledgeExtractionResult(success=False)


class KnowledgeVersionManager:
    def __init__(self):
        self.versions: Dict[str, List[KnowledgeVersion]] = {}
        self.latest_versions: Dict[str, KnowledgeVersion] = {}
        self._version_counter: Dict[str, int] = {}

    def create_version(self, knowledge_id: str, content: Any,
                       status: VersionStatus = VersionStatus.DRAFT,
                       change_description: str = "",
                       created_by: str = "system") -> KnowledgeVersion:
        if knowledge_id not in self._version_counter:
            self._version_counter[knowledge_id] = 0

        self._version_counter[knowledge_id] += 1
        version_number = self._version_counter[knowledge_id]

        parent_version = self.latest_versions.get(knowledge_id)
        parent_id = parent_version.version_id if parent_version else None

        version = KnowledgeVersion(
            version_id=f"ver_{knowledge_id}_{version_number}_{int(time.time() * 1000)}",
            knowledge_id=knowledge_id,
            content=content,
            version_number=version_number,
            status=status,
            parent_version=parent_id,
            change_description=change_description,
            created_at=time.time(),
            created_by=created_by
        )

        if knowledge_id not in self.versions:
            self.versions[knowledge_id] = []
        self.versions[knowledge_id].append(version)
        self.latest_versions[knowledge_id] = version

        return version

    def get_version(self, version_id: str) -> Optional[KnowledgeVersion]:
        for versions in self.versions.values():
            for version in versions:
                if version.version_id == version_id:
                    return version
        return None

    def get_latest_version(self, knowledge_id: str) -> Optional[KnowledgeVersion]:
        return self.latest_versions.get(knowledge_id)

    def get_all_versions(self, knowledge_id: str) -> List[KnowledgeVersion]:
        return self.versions.get(knowledge_id, [])

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        v1 = self.get_version(version_id1)
        v2 = self.get_version(version_id2)

        if not v1 or not v2:
            return {"error": "One or both versions not found"}

        if v1.knowledge_id != v2.knowledge_id:
            return {"error": "Versions belong to different knowledge items"}

        content1 = json.dumps(v1.content, default=str, sort_keys=True) if isinstance(v1.content, (dict, list)) else str(v1.content)
        content2 = json.dumps(v2.content, default=str, sort_keys=True) if isinstance(v2.content, (dict, list)) else str(v2.content)

        diff_size = abs(len(content1) - len(content2))
        similarity = 1.0 - diff_size / max(len(content1), len(content2), 1)

        return {
            "knowledge_id": v1.knowledge_id,
            "version_1": {"version_id": v1.version_id, "version_number": v1.version_number, "status": v1.status.value},
            "version_2": {"version_id": v2.version_id, "version_number": v2.version_number, "status": v2.status.value},
            "similarity": similarity,
            "diff_size": diff_size,
            "change_description_v1": v1.change_description,
            "change_description_v2": v2.change_description
        }

    def rollback_version(self, knowledge_id: str, version_number: int) -> Optional[KnowledgeVersion]:
        versions = self.versions.get(knowledge_id, [])
        for version in versions:
            if version.version_number == version_number:
                version.status = VersionStatus.ACTIVE
                self.latest_versions[knowledge_id] = version

                for v in versions:
                    if v.version_number > version_number:
                        v.status = VersionStatus.DEPRECATED

                return version
        return None

    def deprecate_version(self, knowledge_id: str):
        if knowledge_id in self.latest_versions:
            self.latest_versions[knowledge_id].status = VersionStatus.DEPRECATED

    def archive_version(self, knowledge_id: str):
        if knowledge_id in self.latest_versions:
            self.latest_versions[knowledge_id].status = VersionStatus.ARCHIVED

    def get_version_history(self, knowledge_id: str) -> List[Dict[str, Any]]:
        versions = self.versions.get(knowledge_id, [])
        return [{
            "version_id": v.version_id,
            "version_number": v.version_number,
            "status": v.status.value,
            "parent_version": v.parent_version,
            "change_description": v.change_description,
            "created_at": v.created_at,
            "created_by": v.created_by
        } for v in versions]


class KnowledgeRefiner:
    def __init__(self):
        self.extractor = KnowledgeExtractor()
        self.version_manager = KnowledgeVersionManager()
        self.knowledge_base: Dict[str, KnowledgeNode] = {}
        self.relations: List[KnowledgeRelation] = []
        self._refinement_history: deque = deque(maxlen=500)

    def ingest_knowledge(self, content: str, format_type: KnowledgeFormat,
                         source: str = "") -> KnowledgeExtractionResult:
        result = self.extractor.extract(content, format_type)

        for node in result.nodes:
            node.source = source
            self.knowledge_base[node.node_id] = node

            self.version_manager.create_version(
                knowledge_id=node.node_id,
                content=node.content,
                status=VersionStatus.ACTIVE,
                change_description="Initial ingestion",
                created_by="system"
            )

        self.relations.extend(result.relations)

        self._refinement_history.append({
            "action": "ingest",
            "timestamp": time.time(),
            "nodes_added": len(result.nodes),
            "relations_added": len(result.relations),
            "source": source,
            "format": format_type.value
        })

        return result

    def refine_knowledge(self, node_id: str, new_content: Any,
                         change_description: str = "") -> bool:
        if node_id not in self.knowledge_base:
            return False

        old_node = self.knowledge_base[node_id]
        old_node.content = new_content
        old_node.updated_at = time.time()
        old_node.confidence = min(1.0, old_node.confidence + 0.05)

        self.version_manager.create_version(
            knowledge_id=node_id,
            content=new_content,
            status=VersionStatus.ACTIVE,
            change_description=change_description,
            created_by="system"
        )

        self._refinement_history.append({
            "action": "refine",
            "timestamp": time.time(),
            "node_id": node_id,
            "change_description": change_description
        })

        return True

    def consolidate_knowledge(self, threshold: float = 0.8) -> List[Tuple[str, str]]:
        merged_pairs = []
        nodes = list(self.knowledge_base.values())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                similarity = self._compute_similarity(nodes[i], nodes[j])
                if similarity >= threshold:
                    merged_pairs.append((nodes[i].node_id, nodes[j].node_id))
                    self._merge_nodes(nodes[i], nodes[j])

        return merged_pairs

    def _compute_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        if node1.embedding is not None and node2.embedding is not None:
            norm1 = np.linalg.norm(node1.embedding)
            norm2 = np.linalg.norm(node2.embedding)
            if norm1 > 0 and norm2 > 0:
                return float(np.dot(node1.embedding, node2.embedding) / (norm1 * norm2))

        content1 = str(node1.content)
        content2 = str(node2.content)

        common_chars = set(content1.lower()) & set(content2.lower())
        all_chars = set(content1.lower()) | set(content2.lower())

        if not all_chars:
            return 0.0

        return len(common_chars) / len(all_chars)

    def _merge_nodes(self, primary: KnowledgeNode, secondary: KnowledgeNode):
        if isinstance(primary.content, dict) and isinstance(secondary.content, dict):
            primary.content.update(secondary.content)
        elif isinstance(primary.content, list) and isinstance(secondary.content, list):
            primary.content = list(set(primary.content + secondary.content))
        else:
            primary.content = f"{primary.content} | {secondary.content}"

        primary.confidence = min(1.0, (primary.confidence + secondary.confidence) / 2 + 0.05)
        primary.updated_at = time.time()

        self.version_manager.create_version(
            knowledge_id=primary.node_id,
            content=primary.content,
            status=VersionStatus.ACTIVE,
            change_description=f"Merged with {secondary.node_id}",
            created_by="system"
        )

        self.knowledge_base.pop(secondary.node_id, None)

        self._refinement_history.append({
            "action": "merge",
            "timestamp": time.time(),
            "primary_node": primary.node_id,
            "secondary_node": secondary.node_id
        })

    def validate_knowledge(self, node_id: str) -> Dict[str, Any]:
        if node_id not in self.knowledge_base:
            return {"valid": False, "error": "Node not found"}

        node = self.knowledge_base[node_id]

        checks = []

        checks.append({
            "check": "content_presence",
            "result": node.content is not None and len(str(node.content)) > 0,
            "message": "Content is present"
        })

        checks.append({
            "check": "confidence_threshold",
            "result": node.confidence >= 0.5,
            "message": f"Confidence {node.confidence} meets threshold"
        })

        checks.append({
            "check": "version_exists",
            "result": node_id in self.version_manager.latest_versions,
            "message": "Version tracking exists"
        })

        all_valid = all(c["result"] for c in checks)

        return {
            "valid": all_valid,
            "node_id": node_id,
            "knowledge_type": node.knowledge_type.value,
            "confidence": node.confidence,
            "checks": checks,
            "version_info": {
                "version_id": self.version_manager.latest_versions[node_id].version_id,
                "version_number": self.version_manager.latest_versions[node_id].version_number
            } if node_id in self.version_manager.latest_versions else {}
        }

    def get_knowledge_graph(self) -> Dict[str, Any]:
        nodes = []
        edges = []

        for node_id, node in self.knowledge_base.items():
            nodes.append({
                "node_id": node_id,
                "content": str(node.content)[:200],
                "type": node.knowledge_type.value,
                "format": node.format.value,
                "confidence": node.confidence,
                "source": node.source
            })

        for relation in self.relations:
            edges.append({
                "relation_id": relation.relation_id,
                "source": relation.source_node,
                "target": relation.target_node,
                "type": relation.relation_type,
                "weight": relation.weight
            })

        return {"nodes": nodes, "edges": edges}

    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        results = []

        for node_id, node in self.knowledge_base.items():
            content = str(node.content).lower()
            query_lower = query.lower()

            if query_lower in content:
                similarity = len(query_lower) / max(len(content), 1)
                results.append({
                    "node_id": node_id,
                    "content": str(node.content)[:300],
                    "type": node.knowledge_type.value,
                    "confidence": node.confidence,
                    "similarity": similarity
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def get_refinement_stats(self) -> Dict[str, Any]:
        total_nodes = len(self.knowledge_base)
        total_relations = len(self.relations)
        total_versions = sum(len(v) for v in self.version_manager.versions.values())

        type_distribution = {}
        for node in self.knowledge_base.values():
            t = node.knowledge_type.value
            type_distribution[t] = type_distribution.get(t, 0) + 1

        return {
            "total_nodes": total_nodes,
            "total_relations": total_relations,
            "total_versions": total_versions,
            "type_distribution": type_distribution,
            "refinement_count": len(self._refinement_history),
            "average_confidence": sum(n.confidence for n in self.knowledge_base.values()) / max(total_nodes, 1)
        }