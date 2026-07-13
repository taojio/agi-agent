import numpy as np
from enum import Enum
from collections import deque
from .problem_formulator import ProblemFormulator, TriggerType
from .hypothesis_generator import HypothesisGenerator
from .logical_deductor import LogicalDeductor
from .causal_reasoner import CausalReasoner
from .simulation_engine import SimulationEngine
from .decision_optimizer import DecisionOptimizer, OptimizationCriteria


class ThinkingPhase(Enum):
    FORMULATION = "formulation"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    DEDUCTION = "deduction"
    CAUSAL_REASONING = "causal_reasoning"
    SIMULATION = "simulation"
    OPTIMIZATION = "optimization"
    CONVERGENCE = "convergence"


class ThinkingResult:
    def __init__(self, thinking_id, problem, solution, alternatives=None, phase_results=None, confidence=0.0, status="completed"):
        self.thinking_id = thinking_id
        self.problem = problem
        self.solution = solution
        self.alternatives = alternatives or []
        self.phase_results = phase_results or {}
        self.confidence = confidence
        self.status = status
        self.duration = 0.0

    def to_dict(self):
        return {
            "thinking_id": self.thinking_id,
            "problem": self.problem.to_dict() if hasattr(self.problem, 'to_dict') else str(self.problem),
            "solution": self.solution.to_dict() if hasattr(self.solution, 'to_dict') else str(self.solution),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "confidence": self.confidence,
            "status": self.status,
            "duration": self.duration
        }


class AutonomousThinkingEngine:
    def __init__(self, feature_dim=16, rule_registry=None):
        self.feature_dim = feature_dim
        self.problem_formulator = ProblemFormulator(feature_dim=feature_dim)
        self.hypothesis_generator = HypothesisGenerator(feature_dim=feature_dim)
        self.logical_deductor = LogicalDeductor(feature_dim=feature_dim, rule_registry=rule_registry)
        self.causal_reasoner = CausalReasoner(feature_dim=feature_dim, rule_registry=rule_registry)
        self.simulation_engine = SimulationEngine(feature_dim=feature_dim, rule_registry=rule_registry)
        self.decision_optimizer = DecisionOptimizer(feature_dim=feature_dim, rule_registry=rule_registry)   
        
        self.thinking_history = deque(maxlen=200)
        self.current_phase = ThinkingPhase.FORMULATION
        self.min_confidence_threshold = 0.7
        self.max_retries = 3

    def think(self, input_vector, context=None):
        context = context or {}
        thinking_id = f"thinking_{len(self.thinking_history) + 1}"
        phase_results = {}
        
        problem = self.problem_formulator.formulate_problem(input_vector, context)
        
        if problem is None:
            return ThinkingResult(
                thinking_id=thinking_id,
                problem=None,
                solution=None,
                status="no_problem_detected",
                confidence=0.0
            )
        
        phase_results[ThinkingPhase.FORMULATION.value] = {
            "problem_id": problem.problem_id,
            "trigger_type": problem.trigger_type.value,
            "priority": problem.priority
        }
        
        hypotheses = self.hypothesis_generator.generate_hypotheses(problem)
        phase_results[ThinkingPhase.HYPOTHESIS_GENERATION.value] = {
            "generated_count": len(hypotheses),
            "avg_confidence": float(np.mean([h.confidence for h in hypotheses])) if hypotheses else 0.0
        }
        
        if not hypotheses:
            return self._handle_no_hypotheses(thinking_id, problem)
        
        validated_hypotheses = []
        for hypothesis in hypotheses:
            deduction_result = self.logical_deductor.deduce(hypothesis, problem)
            consistency_result = self.logical_deductor.check_consistency(hypothesis, problem)
            
            if consistency_result["consistent"] and deduction_result["valid"]:
                validated_hypotheses.append(hypothesis)
        
        phase_results[ThinkingPhase.DEDUCTION.value] = {
            "validated_count": len(validated_hypotheses),
            "rejected_count": len(hypotheses) - len(validated_hypotheses)
        }
        
        if not validated_hypotheses:
            return self._handle_no_valid_hypotheses(thinking_id, problem)
        
        for hypothesis in validated_hypotheses:
            self.causal_reasoner.infer_causal_chain(hypothesis, problem)
        
        phase_results[ThinkingPhase.CAUSAL_REASONING.value] = {
            "chains_inferred": len(validated_hypotheses)
        }
        
        simulation_results = self.simulation_engine.batch_simulate(validated_hypotheses, problem)
        phase_results[ThinkingPhase.SIMULATION.value] = {
            "simulated_count": len(simulation_results),
            "avg_success_rate": float(np.mean([s.success_rate for s in simulation_results])) if simulation_results else 0.0
        }
        
        valid_simulations = [(h, s) for h, s in zip(validated_hypotheses, simulation_results) if s.valid]
        
        if not valid_simulations:
            return self._handle_no_valid_simulations(thinking_id, problem)
        
        solution, alternatives = self.decision_optimizer.optimize(
            [h for h, _ in valid_simulations],
            [s for _, s in valid_simulations],
            criteria=OptimizationCriteria.BALANCED
        )
        
        phase_results[ThinkingPhase.OPTIMIZATION.value] = {
            "solution_id": solution.solution_id,
            "priority": solution.priority,
            "alternatives_count": len(alternatives)
        }
        
        final_confidence = solution.confidence * (1 - solution.priority * 0.1)
        
        result = ThinkingResult(
            thinking_id=thinking_id,
            problem=problem,
            solution=solution,
            alternatives=alternatives,
            phase_results=phase_results,
            confidence=final_confidence,
            status="completed"
        )
        
        self.thinking_history.append({
            "thinking_id": thinking_id,
            "problem_id": problem.problem_id,
            "solution_id": solution.solution_id,
            "confidence": final_confidence,
            "status": "completed",
            "timestamp": np.random.randint(1000000)
        })
        
        return result

    def _handle_no_hypotheses(self, thinking_id, problem):
        return ThinkingResult(
            thinking_id=thinking_id,
            problem=problem,
            solution=None,
            status="no_hypotheses",
            confidence=0.0,
            phase_results={"error": "No hypotheses generated"}
        )

    def _handle_no_valid_hypotheses(self, thinking_id, problem):
        return ThinkingResult(
            thinking_id=thinking_id,
            problem=problem,
            solution=None,
            status="no_valid_hypotheses",
            confidence=0.0,
            phase_results={"error": "All hypotheses failed logical validation"}
        )

    def _handle_no_valid_simulations(self, thinking_id, problem):
        return ThinkingResult(
            thinking_id=thinking_id,
            problem=problem,
            solution=None,
            status="no_valid_simulations",
            confidence=0.0,
            phase_results={"error": "All simulations failed"}
        )

    def reflect_on_thinking(self, thinking_result, actual_outcome):
        if thinking_result.status != "completed" or thinking_result.solution is None:
            return None
        
        hypothesis_id = thinking_result.solution.hypothesis_id
        expected_confidence = thinking_result.confidence
        
        actual_success = actual_outcome.get("success", False)
        
        reflection = {
            "thinking_id": thinking_result.thinking_id,
            "hypothesis_id": hypothesis_id,
            "expected_confidence": expected_confidence,
            "actual_success": actual_success,
            "deviation": expected_confidence - (1.0 if actual_success else 0.0),
            "factors": [],
            "recommendations": []
        }
        
        if actual_success:
            reflection["result"] = "success"
            reflection["factors"].append("Solution executed successfully")
            reflection["recommendations"].append("Consider consolidating this approach")
        else:
            reflection["result"] = "failure"
            if expected_confidence > 0.7:
                reflection["factors"].append("Overconfidence in hypothesis")
                reflection["recommendations"].append("Adjust confidence estimation")
            else:
                reflection["factors"].append("Insufficient analysis depth")
                reflection["recommendations"].append("Increase deliberation steps")
        
        return reflection

    def get_thinking_stats(self):
        stats = {
            "total_thinking_cycles": len(self.thinking_history),
            "completed_cycles": len([t for t in self.thinking_history if t["status"] == "completed"]),
            "avg_confidence": 0.0,
            "formulator_stats": self.problem_formulator.get_problem_stats(),
            "hypothesis_stats": self.hypothesis_generator.get_hypothesis_stats(),
            "deduction_stats": self.logical_deductor.get_deduction_stats(),
            "causal_stats": self.causal_reasoner.get_causal_stats(),
            "simulation_stats": self.simulation_engine.get_simulation_stats(),
            "optimizer_stats": self.decision_optimizer.get_optimization_stats()
        }
        
        if self.thinking_history:
            completed = [t for t in self.thinking_history if t["status"] == "completed"]
            if completed:
                stats["avg_confidence"] = float(np.mean([t["confidence"] for t in completed]))
        
        return stats

    def resize(self, new_dim):
        self.feature_dim = new_dim
        self.problem_formulator.resize(new_dim)
        self.hypothesis_generator.resize(new_dim)
        self.logical_deductor.resize(new_dim)
        self.causal_reasoner.resize(new_dim)
        self.simulation_engine.resize(new_dim)
        self.decision_optimizer.resize(new_dim)