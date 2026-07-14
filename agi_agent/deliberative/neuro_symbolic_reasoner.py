import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque, defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from ..config.settings import DEVICE


class SymbolType(Enum):
    PREDICATE = "predicate"
    CONCEPT = "concept"
    RELATION = "relation"
    RULE = "rule"
    EVENT = "event"
    ACTION = "action"


class NeuralSymbol:
    def __init__(self, symbol_id: str, symbol_type: SymbolType,
                 features: Optional[torch.Tensor] = None,
                 embedding: Optional[torch.Tensor] = None):
        self.symbol_id = symbol_id
        self.symbol_type = symbol_type
        self.features = features if features is not None else torch.tensor([])
        self.embedding = embedding if embedding is not None else torch.tensor([])
        self.symbolic_representation: Dict[str, Any] = {}
        self.confidence = 0.5
        self.activation_level = 0.0
        self.temporal_context: List[float] = []

    def activate(self, strength: float = 1.0):
        self.activation_level = min(1.0, self.activation_level + strength * 0.1)
        self.confidence = min(1.0, self.confidence + 0.02)

    def deactivate(self):
        self.activation_level = max(0.0, self.activation_level - 0.05)

    def to_dict(self):
        return {
            "symbol_id": self.symbol_id,
            "symbol_type": self.symbol_type.value,
            "confidence": self.confidence,
            "activation_level": self.activation_level,
            "has_features": len(self.features) > 0,
            "has_embedding": len(self.embedding) > 0
        }


class SymbolicExpression:
    def __init__(self, operator: str, operands: List[Union['SymbolicExpression', NeuralSymbol]]):
        self.operator = operator
        self.operands = operands
        self.confidence = 0.5
        self.truth_value = None

    def evaluate(self, neural_module) -> float:
        if not self.operands:
            return 0.5

        values = []
        for operand in self.operands:
            if isinstance(operand, SymbolicExpression):
                values.append(operand.evaluate(neural_module))
            elif isinstance(operand, NeuralSymbol):
                values.append(operand.confidence)
            else:
                values.append(float(operand))

        if self.operator == "AND":
            result = np.prod(values)
        elif self.operator == "OR":
            result = 1.0 - np.prod(1.0 - np.array(values))
        elif self.operator == "NOT":
            result = 1.0 - values[0] if values else 0.5
        elif self.operator == "IMPLIES":
            if len(values) >= 2:
                result = 1.0 - values[0] + values[0] * values[1]
            else:
                result = np.mean(values)
        elif self.operator == "EQUALS":
            if len(values) >= 2:
                result = 1.0 - abs(values[0] - values[1])
            else:
                result = np.mean(values)
        elif self.operator == "GREATER":
            if len(values) >= 2:
                result = 1.0 if values[0] > values[1] else 0.0
            else:
                result = np.mean(values)
        elif self.operator == "LESS":
            if len(values) >= 2:
                result = 1.0 if values[0] < values[1] else 0.0
            else:
                result = np.mean(values)
        elif self.operator == "ADD":
            result = min(1.0, sum(values))
        elif self.operator == "MULTIPLY":
            result = np.prod(values)
        else:
            result = np.mean(values)

        self.truth_value = result
        self.confidence = result
        return result

    def to_dict(self):
        return {
            "operator": self.operator,
            "operand_count": len(self.operands),
            "confidence": self.confidence,
            "truth_value": self.truth_value
        }


class InferenceRule:
    def __init__(self, rule_id: str, condition: SymbolicExpression,
                 conclusion: SymbolicExpression, confidence: float = 0.9,
                 priority: int = 1):
        self.rule_id = rule_id
        self.condition = condition
        self.conclusion = conclusion
        self.confidence = confidence
        self.priority = priority
        self.usage_count = 0
        self.last_used = 0

    def apply(self, neural_module) -> bool:
        condition_value = self.condition.evaluate(neural_module)
        if condition_value > 0.5:
            conclusion_value = self.conclusion.evaluate(neural_module)
            self.conclusion.confidence = condition_value * self.confidence
            self.usage_count += 1
            self.last_used = time.time() if 'time' in dir() else np.random.randint(1000000)
            return True
        return False

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "priority": self.priority,
            "usage_count": self.usage_count
        }


class NeuralSymbolicModule(nn.Module):
    def __init__(self, symbol_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.symbol_dim = symbol_dim
        self.hidden_dim = hidden_dim

        self.symbol_encoder = nn.Sequential(
            nn.Linear(symbol_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, symbol_dim)
        )

        self.relation_encoder = nn.Sequential(
            nn.Linear(symbol_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, symbol_dim)
        )

        self.logic_gate = nn.Sequential(
            nn.Linear(symbol_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.attention = nn.MultiheadAttention(embed_dim=symbol_dim, num_heads=4)

        self.symbol_memory: Dict[str, NeuralSymbol] = {}
        self.relation_memory: Dict[Tuple[str, str], torch.Tensor] = {}
        self.inference_rules: List[InferenceRule] = []
        self.reasoning_graph: Dict[str, List[str]] = defaultdict(list)

    def encode_symbol(self, symbol: NeuralSymbol) -> torch.Tensor:
        if len(symbol.features) > 0:
            features = symbol.features
            if len(features) < self.symbol_dim:
                features = torch.cat([features, torch.zeros(self.symbol_dim - len(features))])
            elif len(features) > self.symbol_dim:
                features = features[:self.symbol_dim]
        else:
            features = torch.randn(self.symbol_dim)

        embedding = self.symbol_encoder(features.unsqueeze(0)).squeeze(0)
        symbol.embedding = embedding
        return embedding

    def encode_relation(self, source: NeuralSymbol, target: NeuralSymbol) -> torch.Tensor:
        source_emb = self.encode_symbol(source) if len(source.embedding) == 0 else source.embedding
        target_emb = self.encode_symbol(target) if len(target.embedding) == 0 else target.embedding

        combined = torch.cat([source_emb, target_emb])
        relation_emb = self.relation_encoder(combined.unsqueeze(0)).squeeze(0)

        self.relation_memory[(source.symbol_id, target.symbol_id)] = relation_emb
        self.reasoning_graph[source.symbol_id].append(target.symbol_id)

        return relation_emb

    def apply_logic(self, operator: str, operands: List[NeuralSymbol]) -> float:
        if len(operands) == 0:
            return 0.5

        operand_embs = []
        for op in operands:
            emb = self.encode_symbol(op) if len(op.embedding) == 0 else op.embedding
            operand_embs.append(emb)

        if len(operand_embs) >= 3:
            combined = torch.cat(operand_embs[:3])
        elif len(operand_embs) == 2:
            combined = torch.cat([operand_embs[0], operand_embs[1], torch.zeros(self.symbol_dim)])
        else:
            combined = torch.cat([operand_embs[0], torch.zeros(self.symbol_dim * 2)])

        result = self.logic_gate(combined.unsqueeze(0)).squeeze(0).item()
        return result

    def forward(self, symbols: List[NeuralSymbol], relations: List[Tuple[str, str]]) -> Dict[str, Any]:
        embeddings = []
        for symbol in symbols:
            emb = self.encode_symbol(symbol)
            embeddings.append(emb)

        if embeddings:
            emb_tensor = torch.stack(embeddings).unsqueeze(1)
            attn_output, _ = self.attention(emb_tensor, emb_tensor, emb_tensor)
            attn_scores = attn_output.squeeze(1)
        else:
            attn_scores = torch.tensor([])

        relation_results = {}
        for source_id, target_id in relations:
            if source_id in self.symbol_memory and target_id in self.symbol_memory:
                source = self.symbol_memory[source_id]
                target = self.symbol_memory[target_id]
                rel_emb = self.encode_relation(source, target)
                relation_results[f"{source_id}->{target_id}"] = rel_emb.norm().item()

        return {
            "embeddings": embeddings,
            "attention_scores": attn_scores,
            "relation_results": relation_results
        }


class NeuroSymbolicReasoner:
    def __init__(self, symbol_dim: int = 64, hidden_dim: int = 128):
        self.symbol_dim = symbol_dim
        self.hidden_dim = hidden_dim

        self.neural_module = NeuralSymbolicModule(symbol_dim, hidden_dim).to(DEVICE)
        self.symbol_memory: Dict[str, NeuralSymbol] = {}
        self.relation_memory: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.inference_rules: List[InferenceRule] = []
        self.reasoning_history = deque(maxlen=200)
        self.explanation_history = deque(maxlen=100)
        self.confidence_threshold = 0.7

        self._init_default_rules()

    def _init_default_rules(self):
        pass

    def add_symbol(self, symbol_id: str, symbol_type: SymbolType,
                   features: Optional[Union[np.ndarray, torch.Tensor]] = None) -> NeuralSymbol:
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        elif features is None:
            features = torch.randn(self.symbol_dim)

        symbol = NeuralSymbol(symbol_id, symbol_type, features=features)
        self.neural_module.encode_symbol(symbol)
        self.symbol_memory[symbol_id] = symbol
        return symbol

    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                     weight: float = 1.0, confidence: float = 0.5) -> bool:
        if source_id not in self.symbol_memory or target_id not in self.symbol_memory:
            return False

        source = self.symbol_memory[source_id]
        target = self.symbol_memory[target_id]

        rel_emb = self.neural_module.encode_relation(source, target)

        self.relation_memory[(source_id, target_id)] = {
            "relation_type": relation_type,
            "weight": weight,
            "confidence": confidence,
            "embedding": rel_emb
        }

        return True

    def add_inference_rule(self, rule_id: str, condition_expr: SymbolicExpression,
                           conclusion_expr: SymbolicExpression, confidence: float = 0.9,
                           priority: int = 1):
        rule = InferenceRule(rule_id, condition_expr, conclusion_expr, confidence, priority)
        self.inference_rules.append(rule)
        self.inference_rules.sort(key=lambda r: r.priority, reverse=True)

    def reason(self, query: SymbolicExpression) -> Dict[str, Any]:
        explanation_steps = []

        direct_eval = query.evaluate(self.neural_module)
        explanation_steps.append({
            "step": "direct_evaluation",
            "value": direct_eval,
            "description": "直接评估查询表达式"
        })

        result = {
            "query": query.to_dict(),
            "truth_value": direct_eval,
            "confidence": direct_eval,
            "steps": [],
            "explanation": [],
            "deduced_symbols": []
        }

        if direct_eval < self.confidence_threshold:
            for rule in sorted(self.inference_rules, key=lambda r: r.priority, reverse=True):
                if rule.apply(self.neural_module):
                    explanation_steps.append({
                        "step": "rule_application",
                        "rule_id": rule.rule_id,
                        "condition_value": rule.condition.truth_value,
                        "conclusion_value": rule.conclusion.truth_value,
                        "description": f"应用规则 {rule.rule_id}"
                    })

                    for operand in rule.conclusion.operands:
                        if isinstance(operand, NeuralSymbol):
                            operand.activate()
                            result["deduced_symbols"].append(operand.symbol_id)

        final_value = query.evaluate(self.neural_module)
        explanation_steps.append({
            "step": "final_evaluation",
            "value": final_value,
            "description": "最终评估"
        })

        result["steps"] = explanation_steps
        result["truth_value"] = final_value
        result["confidence"] = final_value
        result["explanation"] = self._generate_explanation(explanation_steps)

        self.reasoning_history.append(result)
        self.explanation_history.append(result["explanation"])

        return result

    def _generate_explanation(self, steps: List[Dict]) -> str:
        explanation = "推理过程:\n"
        for i, step in enumerate(steps):
            desc = step.get("description", "")
            value = step.get("value", "")
            explanation += f"{i+1}. {desc}"
            if value:
                explanation += f" (值: {value:.3f})"
            explanation += "\n"
        return explanation

    def get_symbol_activation(self) -> Dict[str, float]:
        return {sym_id: sym.activation_level for sym_id, sym in self.symbol_memory.items()}

    def get_reasoning_graph(self) -> Dict[str, List[str]]:
        return dict(self.neural_module.reasoning_graph)

    def explain_symbol(self, symbol_id: str) -> Dict[str, Any]:
        if symbol_id not in self.symbol_memory:
            return {"error": "Symbol not found"}

        symbol = self.symbol_memory[symbol_id]
        related = []
        for (src, tgt), info in self.relation_memory.items():
            if src == symbol_id or tgt == symbol_id:
                related.append({
                    "source": src,
                    "target": tgt,
                    "relation_type": info["relation_type"],
                    "confidence": info["confidence"]
                })

        return {
            "symbol_id": symbol.symbol_id,
            "symbol_type": symbol.symbol_type.value,
            "confidence": symbol.confidence,
            "activation_level": symbol.activation_level,
            "related_symbols": related,
            "has_embedding": len(symbol.embedding) > 0
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "symbol_count": len(self.symbol_memory),
            "relation_count": len(self.relation_memory),
            "rule_count": len(self.inference_rules),
            "reasoning_history_length": len(self.reasoning_history),
            "explanation_history_length": len(self.explanation_history)
        }