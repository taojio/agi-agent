import numpy as np
import torch
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from ..config.settings import DEVICE


class InteractionProtocol(Enum):
    SYMBOL_TO_ENTITY = "symbol_to_entity"
    ENTITY_TO_SYMBOL = "entity_to_symbol"
    CAUSAL_TO_RULE = "causal_to_rule"
    RULE_TO_CAUSAL = "rule_to_causal"
    SIMULATION_TO_REASONING = "simulation_to_reasoning"
    REASONING_TO_SIMULATION = "reasoning_to_simulation"


class KnowledgeRepresentation:
    def __init__(self, knowledge_id: str, content_type: str,
                 data: Dict[str, Any], confidence: float = 0.5):
        self.knowledge_id = knowledge_id
        self.content_type = content_type
        self.data = data
        self.confidence = confidence
        self.timestamp = np.random.randint(1000000)
        self.source = "unknown"

    def to_dict(self):
        return {
            "knowledge_id": self.knowledge_id,
            "content_type": self.content_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source": self.source,
            "data_keys": list(self.data.keys())
        }


class CoordinationMessage:
    def __init__(self, message_id: str, protocol: InteractionProtocol,
                 sender: str, receiver: str,
                 payload: Dict[str, Any], priority: int = 1):
        self.message_id = message_id
        self.protocol = protocol
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.priority = priority
        self.timestamp = np.random.randint(1000000)
        self.processed = False

    def mark_processed(self):
        self.processed = True

    def to_dict(self):
        return {
            "message_id": self.message_id,
            "protocol": self.protocol.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "priority": self.priority,
            "processed": self.processed,
            "timestamp": self.timestamp
        }


class NeuroSymbolicWorldCoordinator:
    def __init__(self, neuro_symbolic_reasoner=None, world_model_engine=None):
        self.neuro_symbolic_reasoner = neuro_symbolic_reasoner
        self.world_model_engine = world_model_engine

        self.message_queue = deque()
        self.knowledge_cache: Dict[str, KnowledgeRepresentation] = {}
        self.symbol_entity_mapping: Dict[str, str] = {}
        self.entity_symbol_mapping: Dict[str, str] = {}

        self.coordination_history = deque(maxlen=200)
        self.synchronization_interval = 10

    def set_neuro_symbolic_reasoner(self, reasoner):
        self.neuro_symbolic_reasoner = reasoner

    def set_world_model_engine(self, engine):
        self.world_model_engine = engine

    def add_message(self, protocol: InteractionProtocol, sender: str,
                    receiver: str, payload: Dict[str, Any], priority: int = 1):
        message_id = f"msg_{len(self.message_queue) + 1}_{np.random.randint(10000)}"
        message = CoordinationMessage(message_id, protocol, sender, receiver, payload, priority)
        self.message_queue.append(message)

        self.message_queue = deque(sorted(self.message_queue, key=lambda m: -m.priority))

    def process_messages(self):
        processed_count = 0
        while self.message_queue:
            message = self.message_queue.popleft()
            self._process_message(message)
            processed_count += 1

        return processed_count

    def _process_message(self, message: CoordinationMessage):
        protocol = message.protocol

        if protocol == InteractionProtocol.SYMBOL_TO_ENTITY:
            self._handle_symbol_to_entity(message)
        elif protocol == InteractionProtocol.ENTITY_TO_SYMBOL:
            self._handle_entity_to_symbol(message)
        elif protocol == InteractionProtocol.CAUSAL_TO_RULE:
            self._handle_causal_to_rule(message)
        elif protocol == InteractionProtocol.RULE_TO_CAUSAL:
            self._handle_rule_to_causal(message)
        elif protocol == InteractionProtocol.SIMULATION_TO_REASONING:
            self._handle_simulation_to_reasoning(message)
        elif protocol == InteractionProtocol.REASONING_TO_SIMULATION:
            self._handle_reasoning_to_simulation(message)

        message.mark_processed()
        self.coordination_history.append(message.to_dict())

    def _get_symbol_to_entity_map(self):
        from ..cognitive.world_model import EntityCategory
        return {
            "PREDICATE": EntityCategory.CONCEPT,
            "CONCEPT": EntityCategory.CONCEPT,
            "RELATION": EntityCategory.RELATION,
            "RULE": EntityCategory.RULE,
            "EVENT": EntityCategory.EVENT,
            "ACTION": EntityCategory.ACTION,
        }
    
    def _get_entity_to_symbol_map(self):
        from .neuro_symbolic_reasoner import SymbolType
        return {
            "OBJECT": SymbolType.CONCEPT,
            "AGENT": SymbolType.CONCEPT,
            "ENVIRONMENT": SymbolType.CONCEPT,
            "EVENT": SymbolType.EVENT,
            "ACTION": SymbolType.ACTION,
            "RULE": SymbolType.RULE,
            "RELATION": SymbolType.RELATION,
            "CONCEPT": SymbolType.CONCEPT,
        }

    def _handle_symbol_to_entity(self, message: CoordinationMessage):
        if self.world_model_engine is None:
            return

        symbol_id = message.payload.get("symbol_id")
        symbol_data = message.payload.get("symbol_data")

        if symbol_id and symbol_data:
            category = message.payload.get("category", "object")
            features = message.payload.get("features")

            from ..cognitive.world_model import EntityCategory
            mapping = self._get_symbol_to_entity_map()
            entity_category = mapping.get(category.upper(), EntityCategory.OBJECT)

            entity = self.world_model_engine.add_entity(
                entity_id=f"entity_{symbol_id}",
                category=entity_category,
                features=features
            )

            self.symbol_entity_mapping[symbol_id] = entity.entity_id
            self.entity_symbol_mapping[entity.entity_id] = symbol_id

    def _handle_entity_to_symbol(self, message: CoordinationMessage):
        if self.neuro_symbolic_reasoner is None:
            return

        entity_id = message.payload.get("entity_id")
        entity_data = message.payload.get("entity_data")

        if entity_id and entity_data:
            symbol_type = message.payload.get("symbol_type", "concept")

            from .neuro_symbolic_reasoner import SymbolType
            mapping = self._get_entity_to_symbol_map()
            sym_type = mapping.get(symbol_type.upper(), SymbolType.CONCEPT)

            features = entity_data.get("features")
            symbol = self.neuro_symbolic_reasoner.add_symbol(
                symbol_id=f"symbol_{entity_id}",
                symbol_type=sym_type,
                features=features
            )

            self.entity_symbol_mapping[entity_id] = symbol.symbol_id
            self.symbol_entity_mapping[symbol.symbol_id] = entity_id

    def _handle_causal_to_rule(self, message: CoordinationMessage):
        if self.neuro_symbolic_reasoner is None:
            return

        causal_data = message.payload.get("causal_data")
        if not causal_data:
            return

        cause_id = causal_data.get("cause_id")
        effect_id = causal_data.get("effect_id")
        strength = causal_data.get("strength", 0.5)

        if cause_id in self.entity_symbol_mapping and effect_id in self.entity_symbol_mapping:
            symbol_cause = self.neuro_symbolic_reasoner.symbol_memory.get(
                self.entity_symbol_mapping[cause_id]
            )
            symbol_effect = self.neuro_symbolic_reasoner.symbol_memory.get(
                self.entity_symbol_mapping[effect_id]
            )

            if symbol_cause and symbol_effect:
                from .neuro_symbolic_reasoner import SymbolicExpression

                condition_expr = SymbolicExpression("AND", [symbol_cause])
                conclusion_expr = SymbolicExpression("IMPLIES", [symbol_cause, symbol_effect])

                self.neuro_symbolic_reasoner.add_inference_rule(
                    rule_id=f"rule_{cause_id}_{effect_id}",
                    condition_expr=condition_expr,
                    conclusion_expr=conclusion_expr,
                    confidence=strength,
                    priority=2
                )

    def _handle_rule_to_causal(self, message: CoordinationMessage):
        if self.world_model_engine is None:
            return

        rule_data = message.payload.get("rule_data")
        if not rule_data:
            return

        condition_symbols = rule_data.get("condition_symbols", [])
        conclusion_symbols = rule_data.get("conclusion_symbols", [])
        confidence = rule_data.get("confidence", 0.5)

        for cond_sym in condition_symbols:
            if cond_sym in self.symbol_entity_mapping:
                cause_id = self.symbol_entity_mapping[cond_sym]
                for conc_sym in conclusion_symbols:
                    if conc_sym in self.symbol_entity_mapping:
                        effect_id = self.symbol_entity_mapping[conc_sym]
                        self.world_model_engine.add_causal_relation(
                            cause_id=cause_id,
                            effect_id=effect_id,
                            strength=confidence
                        )

    def _handle_simulation_to_reasoning(self, message: CoordinationMessage):
        if self.neuro_symbolic_reasoner is None:
            return

        simulation_result = message.payload.get("simulation_result")
        if not simulation_result:
            return

        final_state = simulation_result.get("final_state", {})
        for entity_id, state_info in final_state.items():
            if entity_id in self.entity_symbol_mapping:
                symbol_id = self.entity_symbol_mapping[entity_id]
                symbol = self.neuro_symbolic_reasoner.symbol_memory.get(symbol_id)
                if symbol:
                    confidence = state_info.get("confidence", 0.5)
                    symbol.confidence = confidence
                    symbol.activate()

    def _handle_reasoning_to_simulation(self, message: CoordinationMessage):
        if self.world_model_engine is None:
            return

        reasoning_result = message.payload.get("reasoning_result")
        if not reasoning_result:
            return

        deduced_symbols = reasoning_result.get("deduced_symbols", [])
        for symbol_id in deduced_symbols:
            if symbol_id in self.symbol_entity_mapping:
                entity_id = self.symbol_entity_mapping[symbol_id]
                entity = self.world_model_engine.entities.get(entity_id)
                if entity:
                    entity.activate()

    def synchronize_knowledge(self):
        if self.neuro_symbolic_reasoner and self.world_model_engine:
            for symbol_id, symbol in self.neuro_symbolic_reasoner.symbol_memory.items():
                if symbol_id not in self.symbol_entity_mapping:
                    self.add_message(
                        protocol=InteractionProtocol.SYMBOL_TO_ENTITY,
                        sender="neuro_symbolic",
                        receiver="world_model",
                        payload={
                            "symbol_id": symbol_id,
                            "symbol_data": symbol.to_dict(),
                            "category": symbol.symbol_type.value,
                            "features": symbol.features.numpy() if len(symbol.features) > 0 else None
                        },
                        priority=2
                    )

            for entity_id, entity in self.world_model_engine.entities.items():
                if entity_id not in self.entity_symbol_mapping:
                    self.add_message(
                        protocol=InteractionProtocol.ENTITY_TO_SYMBOL,
                        sender="world_model",
                        receiver="neuro_symbolic",
                        payload={
                            "entity_id": entity_id,
                            "entity_data": entity.to_dict(),
                            "symbol_type": entity.category.value,
                            "features": entity.features.numpy() if len(entity.features) > 0 else None
                        },
                        priority=2
                    )

            self.process_messages()

    def coordinated_reasoning(self, query: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        result = {
            "reasoning_result": None,
            "simulation_result": None,
            "coordination_steps": [],
            "confidence": 0.5
        }

        if context and self.world_model_engine:
            initial_state = {}
            for entity_id, state_info in context.items():
                if entity_id in self.world_model_engine.entities:
                    initial_state[entity_id] = state_info

            if initial_state:
                simulation = self.world_model_engine.simulate_scenario(
                    scenario_id=f"reasoning_scenario_{np.random.randint(10000)}",
                    initial_state=initial_state,
                    actions=context.get("actions", []),
                    max_steps=10
                )

                self.add_message(
                    protocol=InteractionProtocol.SIMULATION_TO_REASONING,
                    sender="world_model",
                    receiver="neuro_symbolic",
                    payload={"simulation_result": simulation.to_dict()},
                    priority=3
                )

                self.process_messages()
                result["simulation_result"] = simulation.to_dict()
                result["coordination_steps"].append("simulation_completed")

        if self.neuro_symbolic_reasoner and query:
            reasoning_result = self.neuro_symbolic_reasoner.reason(query)

            self.add_message(
                protocol=InteractionProtocol.REASONING_TO_SIMULATION,
                sender="neuro_symbolic",
                receiver="world_model",
                payload={"reasoning_result": reasoning_result},
                priority=3
            )

            self.process_messages()
            result["reasoning_result"] = reasoning_result
            result["coordination_steps"].append("reasoning_completed")
            result["confidence"] = reasoning_result.get("confidence", 0.5)

        return result

    def plan_with_world_model(self, goal: Dict[str, Any],
                              current_state: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "plan": [],
            "simulation_validated": False,
            "confidence": 0.5
        }

        if self.world_model_engine:
            plan = self.world_model_engine.plan_long_term(
                goal_state=goal,
                current_state=current_state,
                max_planning_steps=20
            )

            result["plan"] = plan

            if plan:
                actions = [{"entity_id": step.get("entity_id")} for step in plan]
                simulation = self.world_model_engine.simulate_scenario(
                    scenario_id=f"planning_validation_{np.random.randint(10000)}",
                    initial_state=current_state,
                    actions=actions,
                    max_steps=len(plan)
                )

                result["simulation_validated"] = True
                result["confidence"] = simulation.confidence

        return result

    def get_coordination_summary(self) -> Dict[str, Any]:
        return {
            "message_queue_size": len(self.message_queue),
            "knowledge_cache_size": len(self.knowledge_cache),
            "symbol_entity_mappings": len(self.symbol_entity_mapping),
            "entity_symbol_mappings": len(self.entity_symbol_mapping),
            "coordination_history_length": len(self.coordination_history)
        }