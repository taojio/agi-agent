import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class RuleType(Enum):
    HEURISTIC = "heuristic"
    STATISTICAL = "statistical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"


class MetaRule:
    def __init__(self, rule_id: str, rule_type: RuleType,
                 condition: str, action: str,
                 confidence: float = 0.5, priority: float = 0.5):
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.condition = condition
        self.action = action
        self.confidence = confidence
        self.priority = priority
        self.usage_count: int = 0
        self.success_count: int = 0
        self.created_at = np.random.randint(1000000)
        self.last_used_at: Optional[int] = None

    def apply(self, context: Dict[str, Any]) -> bool:
        self.usage_count += 1
        self.last_used_at = np.random.randint(1000000)
        
        if np.random.random() < self.confidence:
            self.success_count += 1
            return True
        return False

    def update_confidence(self, success: bool):
        self.usage_count += 1
        if success:
            self.success_count += 1
        
        if self.usage_count > 0:
            self.confidence = self.success_count / self.usage_count

    def get_effectiveness(self) -> float:
        if self.usage_count == 0:
            return self.confidence
        return self.success_count / self.usage_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "condition": self.condition,
            "action": self.action,
            "confidence": self.confidence,
            "priority": self.priority,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "effectiveness": self.get_effectiveness(),
            "created_at": self.created_at,
            "last_used_at": self.last_used_at
        }


class MetaPattern:
    def __init__(self, pattern_id: str, description: str,
                 features: np.ndarray, similarity_threshold: float = 0.8):
        self.pattern_id = pattern_id
        self.description = description
        self.features = features
        self.similarity_threshold = similarity_threshold
        self.matches: List[Dict[str, Any]] = []

    def match(self, input_features: np.ndarray) -> float:
        if self.features.shape != input_features.shape:
            return 0.0
        norm = np.linalg.norm(self.features) * np.linalg.norm(input_features)
        if norm == 0:
            return 0.0
        return float(np.dot(self.features, input_features) / norm)

    def is_match(self, input_features: np.ndarray) -> bool:
        return self.match(input_features) >= self.similarity_threshold

    def record_match(self, context: Dict[str, Any]):
        self.matches.append({
            "timestamp": np.random.randint(1000000),
            "context": context,
            "similarity": self.match(context.get("features", np.zeros_like(self.features)))
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "feature_dim": self.features.shape[0],
            "similarity_threshold": self.similarity_threshold,
            "match_count": len(self.matches),
            "avg_match_similarity": float(np.mean([m["similarity"] for m in self.matches])) if self.matches else 0.0
        }


class KnowledgeTransfer:
    def __init__(self, source_domain: str, target_domain: str,
                 knowledge_id: str, transfer_type: str):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.knowledge_id = knowledge_id
        self.transfer_type = transfer_type
        self.effectiveness: float = 0.0
        self.transfer_count: int = 0
        self.success_count: int = 0
        self.created_at = np.random.randint(1000000)

    def record_transfer(self, success: bool, effectiveness: float):
        self.transfer_count += 1
        if success:
            self.success_count += 1
        self.effectiveness = effectiveness

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "knowledge_id": self.knowledge_id,
            "transfer_type": self.transfer_type,
            "effectiveness": self.effectiveness,
            "transfer_count": self.transfer_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.transfer_count if self.transfer_count > 0 else 0.0,
            "created_at": self.created_at
        }


class KnowledgeConsolidation:
    def __init__(self, knowledge_id: str):
        self.knowledge_id = knowledge_id
        self.consolidation_level: float = 0.0
        self.versions: List[Dict[str, Any]] = []
        self.last_consolidated_at: Optional[int] = None

    def consolidate(self, new_version: Dict[str, Any],
                    consolidation_strength: float):
        self.versions.append({
            "version": len(self.versions) + 1,
            "timestamp": np.random.randint(1000000),
            "data": new_version,
            "consolidation_strength": consolidation_strength
        })
        self.consolidation_level = min(1.0, self.consolidation_level + consolidation_strength * 0.2)
        self.last_consolidated_at = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_id": self.knowledge_id,
            "consolidation_level": self.consolidation_level,
            "version_count": len(self.versions),
            "last_consolidated_at": self.last_consolidated_at
        }


class MetaKnowledgeBase:
    def __init__(self):
        self.rules: Dict[str, MetaRule] = {}
        self.patterns: Dict[str, MetaPattern] = {}
        self.transfers: List[KnowledgeTransfer] = []
        self.consolidations: Dict[str, KnowledgeConsolidation] = {}
        self.knowledge_graph: Dict[str, List[str]] = {}
        self._rule_history: deque = deque(maxlen=200)

    def add_rule(self, rule: MetaRule):
        self.rules[rule.rule_id] = rule

    def create_rule(self, rule_id: str, rule_type: RuleType,
                    condition: str, action: str,
                    confidence: float = 0.5, priority: float = 0.5) -> MetaRule:
        rule = MetaRule(rule_id, rule_type, condition, action, confidence, priority)
        self.add_rule(rule)
        return rule

    def add_pattern(self, pattern: MetaPattern):
        self.patterns[pattern.pattern_id] = pattern

    def create_pattern(self, pattern_id: str, description: str,
                       features: np.ndarray,
                       similarity_threshold: float = 0.8) -> MetaPattern:
        pattern = MetaPattern(pattern_id, description, features, similarity_threshold)
        self.add_pattern(pattern)
        return pattern

    def match_patterns(self, input_features: np.ndarray,
                      top_k: int = 5) -> List[Dict[str, Any]]:
        matches = []
        for pattern_id, pattern in self.patterns.items():
            similarity = pattern.match(input_features)
            if similarity >= pattern.similarity_threshold:
                matches.append({
                    "pattern_id": pattern_id,
                    "description": pattern.description,
                    "similarity": similarity,
                    "threshold": pattern.similarity_threshold
                })
        
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]

    def apply_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        applicable_rules = []
        
        for rule_id, rule in self.rules.items():
            if self._evaluate_condition(rule.condition, context):
                result = rule.apply(context)
                applicable_rules.append({
                    "rule_id": rule_id,
                    "rule_type": rule.rule_type.value,
                    "action": rule.action,
                    "success": result,
                    "confidence": rule.confidence,
                    "priority": rule.priority
                })
        
        applicable_rules.sort(key=lambda x: x["priority"], reverse=True)
        
        self._rule_history.append({
            "timestamp": np.random.randint(1000000),
            "applicable_rules": len(applicable_rules),
            "successful_rules": len([r for r in applicable_rules if r["success"]])
        })
        
        return applicable_rules

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        try:
            for key, value in context.items():
                condition = condition.replace(key, str(value))
            return eval(condition)
        except Exception:
            return False

    def transfer_knowledge(self, source_domain: str, target_domain: str,
                           knowledge_id: str, transfer_type: str = "direct") -> Dict[str, Any]:
        transfer = KnowledgeTransfer(source_domain, target_domain, knowledge_id, transfer_type)
        
        effectiveness = np.random.uniform(0.3, 0.9)
        success = effectiveness > 0.5
        
        transfer.record_transfer(success, effectiveness)
        self.transfers.append(transfer)
        
        if source_domain not in self.knowledge_graph:
            self.knowledge_graph[source_domain] = []
        if target_domain not in self.knowledge_graph[source_domain]:
            self.knowledge_graph[source_domain].append(target_domain)
        
        return {
            "success": success,
            "effectiveness": effectiveness,
            "transfer": transfer.to_dict()
        }

    def consolidate_knowledge(self, knowledge_id: str,
                             new_version: Dict[str, Any],
                             consolidation_strength: float = 0.5):
        if knowledge_id not in self.consolidations:
            self.consolidations[knowledge_id] = KnowledgeConsolidation(knowledge_id)
        
        self.consolidations[knowledge_id].consolidate(new_version, consolidation_strength)

    def get_rules_by_type(self, rule_type: RuleType) -> List[Dict[str, Any]]:
        return [rule.to_dict() for rule in self.rules.values() if rule.rule_type == rule_type]

    def get_pattern_summary(self) -> Dict[str, Any]:
        return {
            "total_patterns": len(self.patterns),
            "total_matches": sum(len(p.matches) for p in self.patterns.values()),
            "avg_pattern_dim": float(np.mean([p.features.shape[0] for p in self.patterns.values()])) if self.patterns else 0.0
        }

    def get_transfer_summary(self) -> Dict[str, Any]:
        if not self.transfers:
            return {"total_transfers": 0, "avg_effectiveness": 0.0}
        
        return {
            "total_transfers": len(self.transfers),
            "successful_transfers": len([t for t in self.transfers if t.success_count > 0]),
            "avg_effectiveness": float(np.mean([t.effectiveness for t in self.transfers])),
            "domain_pairs": len(set((t.source_domain, t.target_domain) for t in self.transfers))
        }

    def get_knowledge_graph(self) -> Dict[str, List[str]]:
        return self.knowledge_graph

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_rules": len(self.rules),
            "total_patterns": len(self.patterns),
            "total_transfers": len(self.transfers),
            "total_consolidations": len(self.consolidations),
            "rules_by_type": {rt.value: len(self.get_rules_by_type(rt)) for rt in RuleType},
            "pattern_summary": self.get_pattern_summary(),
            "transfer_summary": self.get_transfer_summary(),
            "knowledge_graph_nodes": len(self.knowledge_graph)
        }