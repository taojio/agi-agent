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
    VIOLATES = "violates"
    PREVENTS = "prevents"
    ENABLES = "enables"
    REQUIRES = "requires"


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


class CommonsenseRule(KGNode):
    """常识规则节点，扩展KGNode支持条件-结论形式的规则表达"""
    
    def __init__(self, rule_id: str, antecedent: List[str], consequent: str,
                 confidence: float = 0.9, category: str = "physics",
                 severity: str = "medium", description: str = "",
                 tags: List[str] = None):
        rule_label = f"Rule: {', '.join(antecedent)} -> {consequent}"
        super().__init__(rule_id, rule_label, EntityType.RULE, description=description)
        
        self.antecedent = antecedent
        self.consequent = consequent
        self.confidence = confidence
        self.category = category
        self.severity = severity
        self.violation_count = 0
        self.trigger_count = 0
        self.tags = tags if tags is not None else []
        self.properties["antecedent"] = antecedent
        self.properties["consequent"] = consequent
        self.properties["category"] = category
        self.properties["severity"] = severity
        self.properties["tags"] = self.tags
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估规则是否被违反"""
        conditions_met = []
        conditions_not_met = []
        
        all_context_text = ""
        for key, value in context.items():
            if isinstance(value, str):
                all_context_text += " " + value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        all_context_text += " " + item
        
        for condition in self.antecedent:
            condition_lower = condition.lower()
            found = False
            
            if condition_lower in all_context_text.lower():
                found = True
            else:
                for key, value in context.items():
                    if isinstance(value, str):
                        value_lower = value.lower()
                        if condition_lower in value_lower or value_lower in condition_lower:
                            found = True
                            break
                        for word in condition_lower.split():
                            if word in value_lower:
                                found = True
                                break
                        if found:
                            break
                    elif isinstance(value, (list, tuple)):
                        for item in value:
                            if isinstance(item, str):
                                item_lower = item.lower()
                                if condition_lower in item_lower or item_lower in condition_lower:
                                    found = True
                                    break
                                for word in condition_lower.split():
                                    if word in item_lower:
                                        found = True
                                        break
                                if found:
                                    break
                        if found:
                            break
            
            if found:
                conditions_met.append(condition)
            else:
                conditions_not_met.append(condition)
        
        all_conditions_met = len(conditions_not_met) == 0
        
        return {
            "violated": all_conditions_met,
            "conditions_met": conditions_met,
            "conditions_not_met": conditions_not_met,
            "rule_id": self.node_id,
            "consequent": self.consequent,
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence
        }
    
    def record_violation(self):
        """记录规则违反"""
        self.violation_count += 1
        self.confidence = min(1.0, self.confidence + 0.01)
    
    def record_trigger(self):
        """记录规则触发"""
        self.trigger_count += 1
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "category": self.category,
            "severity": self.severity,
            "violation_count": self.violation_count,
            "trigger_count": self.trigger_count
        })
        return base_dict


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

    def get_visualization_data(self, max_nodes: int = 100, max_edges: int = 200,
                               focus_node: str = None, depth: int = 2) -> Dict[str, Any]:
        """获取知识图谱可视化数据，包含布局信息。"""
        nodes_data = []
        edges_data = []
        included_nodes = set()

        if focus_node and focus_node in self.nodes:
            included_nodes.add(focus_node)
            queue = [(focus_node, 0)]
            
            while queue and len(included_nodes) < max_nodes:
                current, current_depth = queue.pop(0)
                if current_depth >= depth:
                    continue
                
                for edge in self.edges.get(current, []):
                    if edge.target not in included_nodes:
                        included_nodes.add(edge.target)
                        queue.append((edge.target, current_depth + 1))
        else:
            sorted_nodes = sorted(self.nodes.items(), key=lambda x: -x[1].activation_count)
            for node_id, node in sorted_nodes[:max_nodes]:
                included_nodes.add(node_id)

        layout_positions = self._compute_layout(list(included_nodes))

        for node_id in included_nodes:
            node = self.nodes[node_id]
            pos = layout_positions.get(node_id, {"x": 0, "y": 0})
            nodes_data.append({
                "id": node_id,
                "label": node.label,
                "entity_type": node.entity_type.value,
                "activation_count": node.activation_count,
                "confidence": node.confidence,
                "x": pos["x"],
                "y": pos["y"],
                "size": max(10, min(50, node.activation_count * 2 + node.confidence * 10)),
                "color": self._get_node_color(node.entity_type, node.confidence),
                "properties": node.properties,
                "tags": node.tags,
                "description": node.description
            })

        edge_count = 0
        for node_id in included_nodes:
            for edge in self.edges.get(node_id, []):
                if edge_count >= max_edges:
                    break
                if edge.target in included_nodes:
                    edges_data.append({
                        "source": node_id,
                        "target": edge.target,
                        "relation_type": edge.relation_type.value,
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                        "evidence_count": edge.evidence_count,
                        "directed": edge.directed,
                        "color": self._get_edge_color(edge.relation_type),
                        "width": max(1, min(8, edge.weight * edge.confidence * 6))
                    })
                    edge_count += 1

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "layout_method": "force_directed",
            "total_nodes": len(self.nodes),
            "total_edges": sum(len(edges) for edges in self.edges.values()),
            "focus_node": focus_node
        }

    def _compute_layout(self, node_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """使用力导向布局算法计算节点位置。"""
        positions = {}
        n = len(node_ids)
        
        for i, node_id in enumerate(node_ids):
            base_angle = (i / n) * 2 * np.pi
            radius = 100 + (i % 3) * 50
            
            if len(self.nodes) <= 10:
                positions[node_id] = {
                    "x": np.cos(base_angle) * 80,
                    "y": np.sin(base_angle) * 80
                }
            else:
                ring = i // 20
                angle_in_ring = (i % 20) / 20 * 2 * np.pi
                ring_radius = 60 + ring * 50
                
                positions[node_id] = {
                    "x": np.cos(angle_in_ring) * ring_radius + (np.random.rand() - 0.5) * 20,
                    "y": np.sin(angle_in_ring) * ring_radius + (np.random.rand() - 0.5) * 20
                }

        return positions

    def _get_node_color(self, entity_type: EntityType, confidence: float) -> str:
        """根据实体类型和置信度获取节点颜色。"""
        base_colors = {
            EntityType.CONCEPT: "#4F81BD",
            EntityType.INSTANCE: "#9BBB59",
            EntityType.EVENT: "#F79646",
            EntityType.ACTION: "#C0504D",
            EntityType.ATTRIBUTE: "#8064A2",
            EntityType.RELATION: "#4BACC6",
            EntityType.RULE: "#F2F2F2",
            EntityType.SCHEMA: "#000000"
        }
        
        base_color = base_colors.get(entity_type, "#4F81BD")
        confidence_factor = 0.5 + confidence * 0.5
        
        try:
            hex_color = base_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            r = int(r * confidence_factor)
            g = int(g * confidence_factor)
            b = int(b * confidence_factor)
            
            return f"#{r:02X}{g:02X}{b:02X}"
        except:
            return base_color

    def _get_edge_color(self, relation_type: RelationType) -> str:
        """根据关系类型获取边颜色。"""
        edge_colors = {
            RelationType.IS_A: "#0070C0",
            RelationType.PART_OF: "#00B050",
            RelationType.CAUSES: "#FF6B6B",
            RelationType.LOCATED_IN: "#70AD47",
            RelationType.USED_FOR: "#5B9BD5",
            RelationType.HAS_PROPERTY: "#ED7D31",
            RelationType.SIMILAR_TO: "#A5A5A5",
            RelationType.OPPOSITE_OF: "#FF0000",
            RelationType.PRECEDES: "#FFC000",
            RelationType.FOLLOWS: "#00B0F0",
            RelationType.RELATED_TO: "#7030A0",
            RelationType.DEFINES: "#255E91",
            RelationType.INSTANCE_OF: "#385485",
            RelationType.SUBCLASS_OF: "#1F4E79",
            RelationType.COMPOSED_OF: "#548235",
            RelationType.INTERACTS_WITH: "#C00000",
            RelationType.DEPENDS_ON: "#7030A0",
            RelationType.IMPLIES: "#FF6600",
            RelationType.CONTRADICTS: "#FF0000",
            RelationType.EQUIVALENT_TO: "#00B050",
            RelationType.VIOLATES: "#FF4444",
            RelationType.PREVENTS: "#44FF44",
            RelationType.ENABLES: "#4444FF",
            RelationType.REQUIRES: "#FF44FF"
        }
        
        return edge_colors.get(relation_type, "#808080")

    def validate_and_clean_relations(self):
        """验证并清理低质量关系，提高关系准确性。"""
        cleaned_count = 0
        
        for from_node in list(self.edges.keys()):
            original_count = len(self.edges[from_node])
            self.edges[from_node] = [
                edge for edge in self.edges[from_node]
                if edge.confidence > 0.2 and edge.evidence_count > 0
            ]
            cleaned_count += original_count - len(self.edges[from_node])

        for relation_key in list(self.relation_index.keys()):
            self.relation_index[relation_key] = [
                (s, t) for s, t in self.relation_index[relation_key]
                if s in self.edges and any(e.target == t for e in self.edges[s])
            ]

        return cleaned_count

    def batch_import(self, entities: List[Dict], relations: List[Dict]) -> Dict[str, int]:
        """批量导入实体和关系。"""
        imported_nodes = 0
        imported_edges = 0
        
        for entity in entities:
            try:
                entity_type = EntityType(entity.get("entity_type", "concept"))
            except ValueError:
                entity_type = EntityType.CONCEPT
                
            node_id = self.add_node(
                label=entity.get("label", ""),
                entity_type=entity_type,
                description=entity.get("description", ""),
                properties=entity.get("properties")
            )
            imported_nodes += 1

        for relation in relations:
            from_node = relation.get("from")
            to_node = relation.get("to")
            if from_node in self.nodes and to_node in self.nodes:
                try:
                    rel_type = RelationType(relation.get("relation_type", "related_to"))
                except ValueError:
                    rel_type = RelationType.RELATED_TO
                
                success = self.add_edge(
                    from_node=from_node,
                    to_node=to_node,
                    relation_type=rel_type,
                    weight=relation.get("weight", 1.0),
                    strength=relation.get("strength", 0.5),
                    description=relation.get("description", "")
                )
                if success:
                    imported_edges += 1

        return {
            "imported_nodes": imported_nodes,
            "imported_edges": imported_edges,
            "total_nodes": len(self.nodes),
            "total_edges": sum(len(edges) for edges in self.edges.values())
        }

    def get_top_nodes_by_activation(self, k: int = 20) -> List[Dict]:
        """获取激活次数最高的节点。"""
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: -x[1].activation_count
        )[:k]
        
        return [
            {
                "id": node_id,
                "label": node.label,
                "entity_type": node.entity_type.value,
                "activation_count": node.activation_count,
                "confidence": node.confidence,
                "neighbor_count": len(self.edges.get(node_id, []))
            }
            for node_id, node in sorted_nodes
        ]

    def get_relation_statistics(self) -> Dict[str, Any]:
        """获取关系统计信息。"""
        stats = {}
        
        for rel_type in RelationType:
            edges = self.query_by_relation(rel_type)
            if edges:
                avg_weight = np.mean([e["weight"] for e in edges])
                avg_confidence = np.mean([e["confidence"] for e in edges])
                stats[rel_type.value] = {
                    "count": len(edges),
                    "avg_weight": float(avg_weight),
                    "avg_confidence": float(avg_confidence),
                    "evidence_count": sum(e["evidence_count"] for e in edges)
                }
        
        return stats

    def search_nodes(self, query: str, max_results: int = 10) -> List[Dict]:
        """搜索节点。"""
        query_lower = query.lower()
        results = []
        
        for node_id, node in self.nodes.items():
            if query_lower in node.label.lower() or \
               query_lower in node.description.lower():
                results.append({
                    "id": node_id,
                    "label": node.label,
                    "entity_type": node.entity_type.value,
                    "confidence": node.confidence,
                    "activation_count": node.activation_count,
                    "match_score": (
                        0.6 if query_lower in node.label.lower() else 0
                    ) + (
                        0.4 if query_lower in node.description.lower() else 0
                    )
                })
        
        results.sort(key=lambda x: -x["match_score"])
        return results[:max_results]