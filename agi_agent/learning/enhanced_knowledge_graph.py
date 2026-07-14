import torch
import numpy as np
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from ..config.settings import DEVICE


class RelationType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    CAUSES = "causes"
    LOCATED_IN = "located_in"
    USED_FOR = "used_for"
    HAS_PROPERTY = "has_property"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    RELATED_TO = "related_to"
    DEFINES = "defines"
    INSTANCE_OF = "instance_of"
    SUBCLASS_OF = "subclass_of"
    COMPOSED_OF = "composed_of"
    INTERACTS_WITH = "interacts_with"
    DEPENDS_ON = "depends_on"
    IMPLIES = "implies"
    CONTRADICTS = "contradicts"
    EQUIVALENT_TO = "equivalent_to"


class EntityType(Enum):
    CONCEPT = "concept"
    INSTANCE = "instance"
    EVENT = "event"
    ACTION = "action"
    ATTRIBUTE = "attribute"
    RELATION = "relation"
    RULE = "rule"
    SCHEMA = "schema"


class KGNode:
    def __init__(self, node_id: str, label: str, entity_type: EntityType = EntityType.CONCEPT,
                 features: Optional[torch.Tensor] = None, description: str = ""):
        self.node_id = node_id
        self.label = label
        self.entity_type = entity_type
        self.features = features if features is not None else torch.tensor([])
        self.description = description
        self.properties: Dict[str, Any] = {}
        self.activation_count = 0
        self.confidence = 0.5
        self.created_at = np.random.randint(1000000)
        self.last_accessed = self.created_at
        self.tags: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def activate(self):
        self.activation_count += 1
        self.last_accessed = np.random.randint(1000000)
        self.confidence = min(1.0, self.confidence + 0.02)

    def set_property(self, key: str, value: Any):
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "label": self.label,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "activation_count": self.activation_count,
            "confidence": self.confidence,
            "property_count": len(self.properties),
            "tags": self.tags,
            "metadata": self.metadata
        }


class KGEdge:
    def __init__(self, source: str, target: str, relation_type: RelationType,
                 weight: float = 1.0, strength: float = 0.5,
                 directed: bool = True, description: str = ""):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.weight = weight
        self.strength = strength
        self.directed = directed
        self.description = description
        self.count = 1
        self.evidence_count = 0
        self.confidence = strength

    def update_weight(self, delta: float):
        self.weight = min(1.0, max(0.0, self.weight + delta))
        self.count += 1

    def add_evidence(self):
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence + 0.1)

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "strength": self.strength,
            "directed": self.directed,
            "description": self.description,
            "count": self.count,
            "evidence_count": self.evidence_count,
            "confidence": self.confidence
        }


class EnhancedKnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: Dict[str, List[KGEdge]] = {}
        self.relation_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.node_count = 0
        self.max_nodes = 2000
        self.max_edges_per_node = 50
        self.merge_threshold = 0.85
        self.cluster_interval = 100
        self.last_cluster_step = 0

    def add_node(self, label: str, entity_type: EntityType = EntityType.CONCEPT,
                 features: Optional[torch.Tensor] = None, description: str = "",
                 properties: Optional[Dict] = None) -> str:
        similar_node, similarity = self.find_similar_node(features, threshold=self.merge_threshold)

        if similar_node is not None and similarity > self.merge_threshold:
            self.nodes[similar_node].activate()
            if features is not None:
                self._update_node_features(similar_node, features)
            if properties:
                for k, v in properties.items():
                    self.nodes[similar_node].set_property(k, v)
            return similar_node

        node_id = f"node_{self.node_count}"
        self.node_count += 1

        node = KGNode(node_id, label, entity_type, features, description)
        if properties:
            for k, v in properties.items():
                node.set_property(k, v)

        self.nodes[node_id] = node

        if self.node_count > self.max_nodes:
            self._prune_low_activation_node()

        self.last_cluster_step += 1
        if self.last_cluster_step >= self.cluster_interval:
            self._cluster_nodes()
            self.last_cluster_step = 0

        return node_id

    def _update_node_features(self, node_id: str, new_features: torch.Tensor):
        node = self.nodes[node_id]
        if node.features is None or len(node.features) == 0:
            node.features = new_features.detach().cpu()
        else:
            min_len = min(len(node.features), len(new_features))
            node.features[:min_len] = (node.features[:min_len] * (node.activation_count - 1) +
                                       new_features[:min_len].detach().cpu()) / node.activation_count

    def add_edge(self, from_node: str, to_node: str, relation_type: RelationType,
                 weight: float = 1.0, strength: float = 0.5,
                 directed: bool = True, description: str = "") -> bool:
        if from_node not in self.nodes or to_node not in self.nodes:
            return False

        if len(self.edges.get(from_node, [])) >= self.max_edges_per_node:
            return False

        existing_edge = self._find_edge(from_node, to_node, relation_type)
        if existing_edge:
            existing_edge.update_weight(0.1)
            existing_edge.add_evidence()
            return True

        edge = KGEdge(from_node, to_node, relation_type, weight, strength, directed, description)
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(edge)

        relation_key = relation_type.value
        self.relation_index[relation_key].append((from_node, to_node))

        if not directed:
            reverse_edge = KGEdge(to_node, from_node, relation_type, weight, strength, True, description)
            if to_node not in self.edges:
                self.edges[to_node] = []
            self.edges[to_node].append(reverse_edge)
            self.relation_index[relation_key].append((to_node, from_node))

        return True

    def _find_edge(self, from_node: str, to_node: str, relation_type: RelationType) -> Optional[KGEdge]:
        for edge in self.edges.get(from_node, []):
            if edge.target == to_node and edge.relation_type == relation_type:
                return edge
        return None

    def update_edge(self, from_node: str, to_node: str, relation_type: RelationType,
                    weight_delta: float = 0.1, strength_delta: float = 0.05):
        edge = self._find_edge(from_node, to_node, relation_type)
        if edge:
            edge.update_weight(weight_delta)
            edge.strength = min(1.0, max(0.0, edge.strength + strength_delta))
            edge.confidence = min(1.0, edge.confidence + 0.05)
            return True

        return False

    def query_neighbors(self, node_id: str, k: int = 5,
                        relation_type: Optional[RelationType] = None) -> List[Dict]:
        if node_id not in self.edges:
            return []

        edges = self.edges[node_id]
        if relation_type:
            edges = [e for e in edges if e.relation_type == relation_type]

        sorted_edges = sorted(edges, key=lambda x: x.weight * x.confidence, reverse=True)
        results = []
        for edge in sorted_edges[:k]:
            target_node = self.nodes.get(edge.target)
            results.append({
                "node_id": edge.target,
                "label": target_node.label if target_node else "",
                "relation_type": edge.relation_type.value,
                "weight": edge.weight,
                "confidence": edge.confidence,
                "description": edge.description
            })
        return results

    def find_similar_node(self, features: Optional[torch.Tensor],
                          threshold: float = 0.8) -> Tuple[Optional[str], float]:
        if features is None or len(features) == 0:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for node_id, node in self.nodes.items():
            if node.features is None or len(node.features) == 0:
                continue

            node_feat = node.features.to(DEVICE)

            if len(features) != len(node_feat):
                min_len = min(len(features), len(node_feat))
                feature_trimmed = features[:min_len]
                node_feat_trimmed = node_feat[:min_len]
            else:
                feature_trimmed = features
                node_feat_trimmed = node_feat

            similarity = torch.nn.functional.cosine_similarity(
                feature_trimmed, node_feat_trimmed, dim=-1
            ).item()

            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = node_id

        return best_match, best_similarity

    def query_by_relation(self, relation_type: RelationType, source: str = None,
                          target: str = None) -> List[Dict]:
        results = []
        relation_key = relation_type.value

        for from_node, to_node in self.relation_index.get(relation_key, []):
            if (source is None or from_node == source) and (target is None or to_node == target):
                edge = self._find_edge(from_node, to_node, relation_type)
                if edge:
                    results.append(edge.to_dict())

        return results

    def query_by_entity_type(self, entity_type: EntityType) -> List[Dict]:
        results = []
        for node_id, node in self.nodes.items():
            if node.entity_type == entity_type:
                results.append(node.to_dict())
        return results

    def get_node_data(self, node_id: str) -> Optional[Dict]:
        if node_id not in self.nodes:
            return None
        node = self.nodes[node_id]
        data = node.to_dict()
        data["neighbors"] = self.query_neighbors(node_id, k=10)
        return data

    def find_path(self, from_node: str, to_node: str, max_depth: int = 3,
                  allowed_relations: Optional[List[RelationType]] = None) -> List[List[str]]:
        if from_node == to_node:
            return [[from_node]]

        paths = []
        visited = {from_node}
        queue = [(from_node, [from_node])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for edge in self.edges.get(current, []):
                if allowed_relations and edge.relation_type not in allowed_relations:
                    continue

                if edge.target == to_node:
                    paths.append(path + [edge.target])
                elif edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))

        return paths

    def infer_new_relations(self, threshold: float = 0.7) -> List[Dict]:
        new_relations = []

        for node_id in list(self.nodes.keys()):
            neighbors = self.query_neighbors(node_id, k=20)

            for i, n1 in enumerate(neighbors):
                for j, n2 in enumerate(neighbors[i + 1:], i + 1):
                    n1_edges = set(e["relation_type"] for e in self.query_neighbors(n1["node_id"], k=50))
                    n2_edges = set(e["relation_type"] for e in self.query_neighbors(n2["node_id"], k=50))

                    common_relations = n1_edges & n2_edges
                    if len(common_relations) >= 2:
                        if not self._has_direct_edge(n1["node_id"], n2["node_id"]):
                            inferred_strength = len(common_relations) / max(len(n1_edges), len(n2_edges))
                            if inferred_strength > threshold:
                                new_relations.append({
                                    "source": n1["node_id"],
                                    "target": n2["node_id"],
                                    "relation_type": RelationType.SIMILAR_TO.value,
                                    "inferred_strength": inferred_strength,
                                    "evidence": list(common_relations)
                                })

        return new_relations

    def _has_direct_edge(self, from_node: str, to_node: str) -> bool:
        for edge in self.edges.get(from_node, []):
            if edge.target == to_node:
                return True
        return False

    def _prune_low_activation_node(self):
        if not self.nodes:
            return

        worst_node = None
        worst_score = float('inf')

        for node_id, node in self.nodes.items():
            score = node.created_at / (node.activation_count + 1)
            if score < worst_score:
                worst_score = score
                worst_node = node_id

        if worst_node is not None:
            self._remove_node(worst_node)

    def _remove_node(self, node_id: str):
        if node_id not in self.nodes:
            return

        del self.nodes[node_id]

        if node_id in self.edges:
            del self.edges[node_id]

        for from_node in list(self.edges.keys()):
            self.edges[from_node] = [e for e in self.edges[from_node] if e.target != node_id]

        for relation_key in list(self.relation_index.keys()):
            self.relation_index[relation_key] = [
                (s, t) for s, t in self.relation_index[relation_key]
                if s != node_id and t != node_id
            ]

    def _cluster_nodes(self):
        if len(self.nodes) < 2:
            return

        node_ids = list(self.nodes.keys())
        clusters = []

        for i in range(len(node_ids)):
            node_id = node_ids[i]
            already_clustered = False

            for cluster in clusters:
                representative = cluster[0]
                rep_feat = self.nodes[representative].features.to(DEVICE) if self.nodes[representative].features is not None else None
                curr_feat = self.nodes[node_id].features.to(DEVICE) if self.nodes[node_id].features is not None else None

                if rep_feat is None or curr_feat is None:
                    continue

                min_len = min(len(rep_feat), len(curr_feat))
                similarity = torch.nn.functional.cosine_similarity(
                    rep_feat[:min_len], curr_feat[:min_len], dim=-1
                ).item()

                if similarity > self.merge_threshold * 0.9:
                    cluster.append(node_id)
                    already_clustered = True
                    break

            if not already_clustered:
                clusters.append([node_id])

        for cluster in clusters:
            if len(cluster) > 1:
                self._merge_cluster(cluster)

    def _merge_cluster(self, cluster):
        if len(cluster) < 2:
            return

        main_node = cluster[0]
        total_activation = self.nodes[main_node].activation_count
        main_feat = self.nodes[main_node].features

        merged_features = main_feat.clone() if main_feat is not None and len(main_feat) > 0 else None

        for node_id in cluster[1:]:
            node = self.nodes[node_id]
            total_activation += node.activation_count

            if merged_features is not None and node.features is not None and len(node.features) > 0:
                min_len = min(len(merged_features), len(node.features))
                merged_features[:min_len] += node.features[:min_len] * node.activation_count

            for edge in self.edges.get(node_id, []):
                self.add_edge(main_node, edge.target, edge.relation_type, edge.weight, edge.strength)

            for from_node, edges in list(self.edges.items()):
                for edge in edges:
                    if edge.target == node_id:
                        edge.target = main_node

            if node_id in self.edges:
                del self.edges[node_id]

            del self.nodes[node_id]

        if merged_features is not None:
            merged_features /= total_activation
            self.nodes[main_node].features = merged_features
        self.nodes[main_node].activation_count = total_activation

    def get_summary(self) -> Dict[str, Any]:
        total_edges = sum(len(edges) for edges in self.edges.values())
        avg_activation = np.mean([n.activation_count for n in self.nodes.values()]) if self.nodes else 0.0
        avg_confidence = np.mean([n.confidence for n in self.nodes.values()]) if self.nodes else 0.0

        relation_distribution = {}
        for rel_type in RelationType:
            count = len(self.relation_index.get(rel_type.value, []))
            if count > 0:
                relation_distribution[rel_type.value] = count

        entity_distribution = {}
        for entity_type in EntityType:
            count = sum(1 for n in self.nodes.values() if n.entity_type == entity_type)
            if count > 0:
                entity_distribution[entity_type.value] = count

        return {
            "nodes": len(self.nodes),
            "edges": total_edges,
            "max_nodes": self.max_nodes,
            "avg_activation": float(avg_activation),
            "avg_confidence": float(avg_confidence),
            "relation_distribution": relation_distribution,
            "entity_distribution": entity_distribution
        }

    def get_statistics(self) -> Dict[str, Any]:
        summary = self.get_summary()
        return {
            **summary,
            "total_relations": len(self.relation_index),
            "relation_types": list(self.relation_index.keys()),
            "entity_types": list(EntityType.__members__.keys())
        }