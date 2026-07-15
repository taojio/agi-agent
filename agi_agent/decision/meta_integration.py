import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..meta_learning.maml_algorithm import MAMLAlgorithm, MAMLMode, MAMLTask
from ..meta_learning.hyperparameter_controller import HyperparameterController, ParameterType
from ..meta_learning.task_strategy_knowledge import (
    TaskStrategyKnowledgeBase, TaskType, TaskComplexity, DataDistribution
)


class MetaEnhancedDecisionStrategy:
    def __init__(self, state_dim: int = 128, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.maml = MAMLAlgorithm(state_dim, action_dim, hidden_dim=256)
        self.hp_controller = HyperparameterController()
        self.knowledge_base = TaskStrategyKnowledgeBase()
        self.strategy_history: List[Dict[str, Any]] = []

    def select_action(self, state: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if state.ndim == 1:
            state = state.reshape(1, -1)

        logits = self.maml.model.forward(state)

        temperature = self.hp_controller.get_parameter(ParameterType.TEMPERATURE)
        exp_logits = np.exp(logits / temperature)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        action = int(np.argmax(probs))
        confidence = float(probs[0, action])

        exploration_rate = self.hp_controller.get_parameter(ParameterType.EXPLORATION_RATE)
        if np.random.random() < exploration_rate:
            action = np.random.randint(self.action_dim)
            confidence = float(exploration_rate)

        result = {
            "action": action,
            "confidence": confidence,
            "action_probabilities": probs.tolist(),
            "temperature": temperature,
            "exploration_rate": exploration_rate,
            "timestamp": np.random.randint(1000000)
        }

        if context:
            self.hp_controller.set_task_context(
                context.get("complexity", 0.5),
                context.get("uncertainty", 0.5)
            )

        self.strategy_history.append(result)

        return result

    def adapt_strategy(self, support_states: np.ndarray, support_actions: np.ndarray,
                       query_states: np.ndarray, query_actions: np.ndarray,
                       task_id: str = "decision_task") -> Dict[str, Any]:
        task = MAMLTask(task_id, "reinforcement",
                        (support_states, support_actions), (query_states, query_actions))

        learning_rate = self.hp_controller.get_parameter(ParameterType.LEARNING_RATE)
        result = self.maml.adapt_to_task(task, learning_rate=learning_rate)

        self.hp_controller.update_performance(
            result.get_best_metrics()["train_loss"],
            result.get_best_metrics()["query_accuracy"]
        )
        self.hp_controller.adjust_all(task_type="reinforcement")

        return {
            "task_id": task_id,
            "adaptation_result": result.to_dict(),
            "learning_rate": learning_rate,
            "strategy_history_count": len(self.strategy_history)
        }

    def register_task(self, task_id: str, task_type: TaskType,
                      complexity: TaskComplexity, data_distribution: DataDistribution,
                      input_dim: int, output_dim: int, num_samples: int):
        self.knowledge_base.register_task(
            task_id, task_type, complexity, data_distribution,
            input_dim, output_dim, num_samples
        )

    def recommend_strategy(self, task_id: str) -> List[Dict[str, Any]]:
        return self.knowledge_base.recommend_strategies(task_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "maml_mode": self.maml.mode.value,
            "hp_summary": self.hp_controller.get_parameter_summary(),
            "knowledge_summary": self.knowledge_base.get_summary(),
            "strategy_history_count": len(self.strategy_history)
        }


class MetaEnhancedRiskAssessor:
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim
        self.maml = MAMLAlgorithm(feature_dim, 2, hidden_dim=128)
        self.hp_controller = HyperparameterController()
        self.risk_threshold = 0.5

    def assess_risk(self, decision_features: np.ndarray) -> Dict[str, Any]:
        if decision_features.ndim == 1:
            decision_features = decision_features.reshape(1, -1)

        logits = self.maml.model.forward(decision_features)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        risk_probability = float(probs[0, 1])
        is_high_risk = risk_probability > self.risk_threshold

        result = {
            "risk_probability": risk_probability,
            "is_high_risk": is_high_risk,
            "risk_threshold": self.risk_threshold,
            "confidence": float(np.max(probs))
        }

        return result

    def adapt_assessor(self, support_data: Tuple[np.ndarray, np.ndarray],
                       query_data: Tuple[np.ndarray, np.ndarray]):
        task = MAMLTask("risk_assessment", "classification", support_data, query_data)

        result = self.maml.adapt_to_task(task)

        return {
            "adaptation_result": result.to_dict(),
            "risk_threshold": self.risk_threshold
        }

    def set_risk_threshold(self, threshold: float):
        self.risk_threshold = max(0.0, min(1.0, threshold))

    def get_status(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "risk_threshold": self.risk_threshold,
            "maml_mode": self.maml.mode.value,
            "hp_summary": self.hp_controller.get_parameter_summary()
        }


class MetaEnhancedActionPlanner:
    def __init__(self, state_dim: int = 128, action_dim: int = 10, max_steps: int = 100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.maml = MAMLAlgorithm(state_dim, action_dim, hidden_dim=256)
        self.hp_controller = HyperparameterController()
        self.plans: Dict[str, List[int]] = {}

    def generate_plan(self, initial_state: np.ndarray, goal_state: np.ndarray,
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        plan = []
        current_state = initial_state.copy()

        for step in range(self.max_steps):
            action_result = self.select_action(current_state, context)
            plan.append(action_result["action"])

            simulated_next_state = self._simulate_step(current_state, action_result["action"])
            current_state = simulated_next_state

            distance = np.linalg.norm(current_state - goal_state)
            if distance < 0.1:
                break

        plan_id = f"plan_{np.random.randint(1000000)}"
        self.plans[plan_id] = plan

        return {
            "plan_id": plan_id,
            "plan": plan,
            "plan_length": len(plan),
            "goal_distance": float(np.linalg.norm(current_state - goal_state))
        }

    def select_action(self, state: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if state.ndim == 1:
            state = state.reshape(1, -1)

        logits = self.maml.model.forward(state)
        action = int(np.argmax(logits))

        return {"action": action}

    def _simulate_step(self, state: np.ndarray, action: int) -> np.ndarray:
        delta = np.random.randn(*state.shape) * 0.1
        delta[0, min(action, state.shape[1] - 1)] += 0.1
        return np.clip(state + delta, 0, 1)

    def adapt_planner(self, support_data: Tuple[np.ndarray, np.ndarray],
                      query_data: Tuple[np.ndarray, np.ndarray],
                      task_id: str = "planning_task") -> Dict[str, Any]:
        task = MAMLTask(task_id, "reinforcement", support_data, query_data)

        learning_rate = self.hp_controller.get_parameter(ParameterType.LEARNING_RATE)
        result = self.maml.adapt_to_task(task, learning_rate=learning_rate)

        return {
            "task_id": task_id,
            "adaptation_result": result.to_dict(),
            "learning_rate": learning_rate,
            "num_plans": len(self.plans)
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_steps": self.max_steps,
            "num_plans": len(self.plans),
            "maml_mode": self.maml.mode.value,
            "hp_summary": self.hp_controller.get_parameter_summary()
        }


class DecisionMetaIntegration:
    def __init__(self, state_dim: int = 128, action_dim: int = 10):
        self.decision_strategy = MetaEnhancedDecisionStrategy(state_dim, action_dim)
        self.risk_assessor = MetaEnhancedRiskAssessor(feature_dim=state_dim)
        self.action_planner = MetaEnhancedActionPlanner(state_dim, action_dim)
        self.integration_history: List[Dict[str, Any]] = []

    def make_decision(self, state: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        risk_result = self.risk_assessor.assess_risk(state)

        if risk_result["is_high_risk"]:
            strategy_result = self.decision_strategy.select_action(state, context)
            strategy_result["risk_assessment"] = risk_result
            strategy_result["decision_type"] = "risk-aware"
        else:
            strategy_result = self.decision_strategy.select_action(state, context)
            strategy_result["risk_assessment"] = risk_result
            strategy_result["decision_type"] = "normal"

        result = {
            "decision": strategy_result,
            "risk_assessment": risk_result,
            "timestamp": np.random.randint(1000000)
        }

        self.integration_history.append(result)

        return result

    def generate_plan(self, initial_state: np.ndarray, goal_state: np.ndarray,
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.action_planner.generate_plan(initial_state, goal_state, context)

    def adapt(self, support_states: np.ndarray, support_actions: np.ndarray,
              query_states: np.ndarray, query_actions: np.ndarray,
              task_id: str = "decision_adapt") -> Dict[str, Any]:
        strategy_result = self.decision_strategy.adapt_strategy(
            support_states, support_actions, query_states, query_actions, task_id
        )

        return {
            "decision_strategy": strategy_result,
            "risk_assessor": self.risk_assessor.get_status(),
            "action_planner": self.action_planner.get_status(),
            "overall_status": self.get_status()
        }

    def register_task(self, task_id: str, task_type: str, complexity: str,
                      data_distribution: str, input_dim: int, output_dim: int, num_samples: int):
        from ..meta_learning.task_strategy_knowledge import TaskType, TaskComplexity, DataDistribution
        self.decision_strategy.register_task(
            task_id,
            TaskType(task_type),
            TaskComplexity(complexity),
            DataDistribution(data_distribution),
            input_dim, output_dim, num_samples
        )

    def get_strategy_recommendation(self, task_id: str) -> List[Dict[str, Any]]:
        return self.decision_strategy.recommend_strategy(task_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "decision_strategy": self.decision_strategy.get_status(),
            "risk_assessor": self.risk_assessor.get_status(),
            "action_planner": self.action_planner.get_status(),
            "integration_history_count": len(self.integration_history)
        }