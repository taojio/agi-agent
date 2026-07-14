import numpy as np
import torch
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from ..config.settings import DEVICE


class AbstractionLevel(Enum):
    INSTANCE = "instance"
    CONCEPT = "concept"
    CATEGORY = "category"
    METACONCEPT = "metaconcept"


class ConceptNode:
    def __init__(self, concept_id: str, label: str, level: AbstractionLevel = AbstractionLevel.CONCEPT,
                 features: Optional[np.ndarray] = None, description: str = ""):
        self.concept_id = concept_id
        self.label = label
        self.level = level
        self.features = features if features is not None else np.array([])
        self.description = description
        self.instances = []
        self.parent_concepts = []
        self.child_concepts = []
        self.related_concepts = []
        self.activation_count = 0
        self.confidence = 0.5
        self.timestamp = np.random.randint(1000000)

    def activate(self):
        self.activation_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)

    def add_instance(self, instance_id: str):
        if instance_id not in self.instances:
            self.instances.append(instance_id)

    def add_parent(self, parent_id: str):
        if parent_id not in self.parent_concepts:
            self.parent_concepts.append(parent_id)

    def add_child(self, child_id: str):
        if child_id not in self.child_concepts:
            self.child_concepts.append(child_id)

    def add_related(self, related_id: str, relation_type: str = "related"):
        if related_id not in [r[0] for r in self.related_concepts]:
            self.related_concepts.append((related_id, relation_type))

    def to_dict(self):
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "level": self.level.value,
            "description": self.description,
            "instance_count": len(self.instances),
            "parent_count": len(self.parent_concepts),
            "child_count": len(self.child_concepts),
            "related_count": len(self.related_concepts),
            "activation_count": self.activation_count,
            "confidence": self.confidence
        }


class AbstractionEngine:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.concepts: Dict[str, ConceptNode] = {}
        self.concept_history = deque(maxlen=200)
        self.abstraction_threshold = 0.7
        self.generalization_threshold = 0.6
        self.max_concepts = 500

    def add_concept(self, label: str, features: np.ndarray = None,
                    level: AbstractionLevel = AbstractionLevel.CONCEPT,
                    description: str = "") -> str:
        similar_concept = self._find_similar_concept(features, threshold=self.abstraction_threshold)

        if similar_concept is not None:
            self.concepts[similar_concept].activate()
            if features is not None:
                self._update_concept_features(similar_concept, features)
            return similar_concept

        concept_id = f"concept_{len(self.concepts) + 1}"
        concept = ConceptNode(concept_id, label, level, features, description)
        self.concepts[concept_id] = concept

        if len(self.concepts) > self.max_concepts:
            self._prune_low_activation_concepts()

        self._check_hierarchy(concept_id)

        return concept_id

    def _find_similar_concept(self, features: np.ndarray, threshold: float = 0.7) -> Optional[str]:
        if features is None or len(features) == 0:
            return None

        best_match = None
        best_similarity = 0.0

        for concept_id, concept in self.concepts.items():
            if concept.features is None or len(concept.features) == 0:
                continue

            min_len = min(len(features), len(concept.features))
            if min_len == 0:
                continue

            similarity = np.dot(features[:min_len], concept.features[:min_len]) / (
                np.linalg.norm(features[:min_len]) * np.linalg.norm(concept.features[:min_len]) + 1e-8
            )

            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = concept_id

        return best_match

    def _update_concept_features(self, concept_id: str, new_features: np.ndarray):
        concept = self.concepts[concept_id]
        if concept.features is None or len(concept.features) == 0:
            concept.features = new_features.copy()
        else:
            min_len = min(len(concept.features), len(new_features))
            concept.features[:min_len] = (concept.features[:min_len] * (concept.activation_count - 1) +
                                          new_features[:min_len]) / concept.activation_count

    def _check_hierarchy(self, concept_id: str):
        concept = self.concepts[concept_id]
        if concept.features is None:
            return

        for other_id, other_concept in self.concepts.items():
            if other_id == concept_id or other_concept.features is None:
                continue

            similarity = self._compute_similarity(concept.features, other_concept.features)

            if similarity > self.abstraction_threshold:
                if concept.level.value > other_concept.level.value:
                    concept.add_parent(other_id)
                    other_concept.add_child(concept_id)
                elif concept.level.value < other_concept.level.value:
                    concept.add_child(other_id)
                    other_concept.add_parent(concept_id)

    def _compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        min_len = min(len(features1), len(features2))
        if min_len == 0:
            return 0.0
        return np.dot(features1[:min_len], features2[:min_len]) / (
            np.linalg.norm(features1[:min_len]) * np.linalg.norm(features2[:min_len]) + 1e-8
        )

    def abstract_from_instances(self, instance_ids: List[str], new_label: str, description: str = "") -> Optional[str]:
        if len(instance_ids) < 2:
            return None

        features_list = []
        for inst_id in instance_ids:
            if inst_id in self.concepts:
                inst = self.concepts[inst_id]
                if inst.features is not None and len(inst.features) > 0:
                    features_list.append(inst.features)

        if len(features_list) < 2:
            return None

        avg_features = np.mean(features_list, axis=0)
        new_concept_id = self.add_concept(new_label, avg_features, AbstractionLevel.CONCEPT, description)

        for inst_id in instance_ids:
            if inst_id in self.concepts:
                self.concepts[inst_id].add_parent(new_concept_id)
                self.concepts[new_concept_id].add_child(inst_id)

        return new_concept_id

    def generalize(self, concept_id: str, new_label: str, description: str = "") -> Optional[str]:
        if concept_id not in self.concepts:
            return None

        concept = self.concepts[concept_id]

        all_features = [concept.features] if concept.features is not None else []
        for child_id in concept.child_concepts:
            if child_id in self.concepts and self.concepts[child_id].features is not None:
                all_features.append(self.concepts[child_id].features)

        if len(all_features) < 2:
            return None

        generalized_features = np.mean(all_features, axis=0)

        new_level = self._get_next_level(concept.level)
        new_concept_id = self.add_concept(new_label, generalized_features, new_level, description)

        concept.add_parent(new_concept_id)
        self.concepts[new_concept_id].add_child(concept_id)

        return new_concept_id

    def _get_next_level(self, current_level: AbstractionLevel) -> AbstractionLevel:
        levels = list(AbstractionLevel)
        current_idx = levels.index(current_level)
        return levels[min(current_idx + 1, len(levels) - 1)]

    def find_analogies(self, concept_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if concept_id not in self.concepts:
            return []

        concept = self.concepts[concept_id]
        if concept.features is None:
            return []

        similarities = []
        for other_id, other_concept in self.concepts.items():
            if other_id == concept_id or other_concept.features is None:
                continue

            similarity = self._compute_similarity(concept.features, other_concept.features)
            if similarity > self.generalization_threshold:
                similarities.append({
                    "concept_id": other_id,
                    "label": other_concept.label,
                    "similarity": similarity,
                    "level": other_concept.level.value,
                    "relation_path": self._find_relation_path(concept_id, other_id)
                })

        similarities.sort(key=lambda x: -x["similarity"])
        return similarities[:top_k]

    def _find_relation_path(self, from_id: str, to_id: str) -> List[str]:
        if from_id == to_id:
            return []

        visited = {from_id}
        queue = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)
            current_concept = self.concepts.get(current)
            if not current_concept:
                continue

            for child_id in current_concept.child_concepts:
                if child_id == to_id:
                    return path + [child_id]
                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, path + [child_id]))

            for parent_id in current_concept.parent_concepts:
                if parent_id == to_id:
                    return path + [parent_id]
                if parent_id not in visited:
                    visited.add(parent_id)
                    queue.append((parent_id, path + [parent_id]))

        for rel_id, rel_type in current_concept.related_concepts:
            if rel_id == to_id:
                return path + [rel_id]

        return []

    def transfer_knowledge(self, source_concept_id: str, target_concept_id: str) -> Dict[str, Any]:
        if source_concept_id not in self.concepts or target_concept_id not in self.concepts:
            return {"success": False, "message": "Concept not found"}

        source = self.concepts[source_concept_id]
        target = self.concepts[target_concept_id]

        transferred = []

        for rel_id, rel_type in source.related_concepts:
            if rel_id not in [r[0] for r in target.related_concepts]:
                target.add_related(rel_id, rel_type)
                transferred.append({"relation_type": rel_type, "related_concept": rel_id})

        if source.description and not target.description:
            target.description = f"类似{source.label}: {source.description}"
            transferred.append({"description": target.description})

        return {
            "success": True,
            "transferred_count": len(transferred),
            "transferred": transferred,
            "source_concept": source.label,
            "target_concept": target.label
        }

    def get_concept_hierarchy(self, concept_id: str) -> Dict[str, Any]:
        if concept_id not in self.concepts:
            return {"error": "Concept not found"}

        concept = self.concepts[concept_id]

        hierarchy = {
            "concept": concept.to_dict(),
            "parents": [],
            "children": [],
            "related": []
        }

        for parent_id in concept.parent_concepts:
            if parent_id in self.concepts:
                hierarchy["parents"].append(self.concepts[parent_id].to_dict())

        for child_id in concept.child_concepts:
            if child_id in self.concepts:
                hierarchy["children"].append(self.concepts[child_id].to_dict())

        for rel_id, rel_type in concept.related_concepts:
            if rel_id in self.concepts:
                hierarchy["related"].append({
                    "concept": self.concepts[rel_id].to_dict(),
                    "relation_type": rel_type
                })

        return hierarchy

    def _prune_low_activation_concepts(self):
        concepts_list = list(self.concepts.items())
        concepts_list.sort(key=lambda x: x[1].activation_count)

        to_remove = concepts_list[:int(len(concepts_list) * 0.1)]
        for concept_id, _ in to_remove:
            self._remove_concept(concept_id)

    def _remove_concept(self, concept_id: str):
        if concept_id not in self.concepts:
            return

        concept = self.concepts[concept_id]

        for parent_id in concept.parent_concepts:
            if parent_id in self.concepts:
                self.concepts[parent_id].child_concepts.remove(concept_id)

        for child_id in concept.child_concepts:
            if child_id in self.concepts:
                self.concepts[child_id].parent_concepts.remove(concept_id)

        del self.concepts[concept_id]

    def get_abstraction_stats(self) -> Dict[str, Any]:
        stats = {
            "total_concepts": len(self.concepts),
            "level_distribution": {},
            "avg_activation": 0.0,
            "avg_confidence": 0.0,
            "hierarchy_depth": 0
        }

        for level in AbstractionLevel:
            stats["level_distribution"][level.value] = sum(
                1 for c in self.concepts.values() if c.level == level
            )

        if self.concepts:
            stats["avg_activation"] = float(np.mean([c.activation_count for c in self.concepts.values()]))
            stats["avg_confidence"] = float(np.mean([c.confidence for c in self.concepts.values()]))

            max_depth = 0
            for concept in self.concepts.values():
                depth = self._compute_depth(concept.concept_id)
                max_depth = max(max_depth, depth)
            stats["hierarchy_depth"] = max_depth

        return stats

    def _compute_depth(self, concept_id: str, visited: Optional[set] = None) -> int:
        if visited is None:
            visited = set()

        if concept_id in visited or concept_id not in self.concepts:
            return 0

        visited.add(concept_id)
        concept = self.concepts[concept_id]

        if not concept.parent_concepts:
            return 1

        return 1 + max(self._compute_depth(p, visited.copy()) for p in concept.parent_concepts)

    def resize(self, new_dim):
        self.feature_dim = new_dim