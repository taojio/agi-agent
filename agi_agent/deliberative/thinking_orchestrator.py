import time
import numpy as np
from enum import Enum
from collections import deque
from typing import Dict, Any, List
from .autonomous_thinking_engine import AutonomousThinkingEngine, ThinkingResult
from ..reflex import ReflexController


class ThinkingMode(Enum):
    FAST = "fast"
    SLOW = "slow"
    AUTO = "auto"


class ThinkingOrchestrator:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.reflex_controller = ReflexController(feature_dim=feature_dim)
        self.thinking_engine = AutonomousThinkingEngine(feature_dim=feature_dim)
        
        self.mode = ThinkingMode.AUTO
        self.confidence_threshold = 0.85
        self.thinking_history = deque(maxlen=200)
        self.pending_problems = []
        
        self.system1_count = 0
        self.system2_count = 0

    def process(self, input_vector, context=None):
        context = context or {}
        
        reflex_response = self.reflex_controller.process(input_vector, context)
        
        if self.mode == ThinkingMode.FAST:
            self.system1_count += 1
            return {
                "mode": "system1",
                "response": reflex_response,
                "confidence": reflex_response.get("confidence", 0.0),
                "delegate": False
            }
        
        if self.mode == ThinkingMode.SLOW:
            self.system2_count += 1
            thinking_result = self.thinking_engine.think(input_vector, context)
            return {
                "mode": "system2",
                "response": thinking_result.to_dict(),
                "confidence": thinking_result.confidence,
                "delegate": False
            }
        
        confidence = reflex_response.get("confidence", 0.0)
        should_delegate = reflex_response.get("delegate", False)
        
        if confidence >= self.confidence_threshold and not should_delegate:
            self.system1_count += 1
            return {
                "mode": "system1",
                "response": reflex_response,
                "confidence": confidence,
                "delegate": False
            }
        
        self.system2_count += 1
        thinking_result = self.thinking_engine.think(input_vector, context)
        
        if thinking_result.status == "completed" and thinking_result.solution:
            return {
                "mode": "system2",
                "response": thinking_result.to_dict(),
                "confidence": thinking_result.confidence,
                "delegate": False
            }
        else:
            return {
                "mode": "system1_fallback",
                "response": reflex_response,
                "confidence": confidence,
                "delegate": True
            }

    def run_full_thinking_cycle(self, input_vector, context=None):
        context = context or {}
        
        thinking_result = self.thinking_engine.think(input_vector, context)
        
        self.thinking_history.append({
            "thinking_id": thinking_result.thinking_id,
            "confidence": thinking_result.confidence,
            "status": thinking_result.status,
            "mode": "system2",
            "timestamp": time.time()
        })
        
        return thinking_result

    def decompose_problem(self, problem_description: str, context=None) -> Dict[str, Any]:
        context = context or {}

        sub_problems = []
        problem_keywords = ["learn", "explore", "solve", "optimize", "create", "analyze"]
        for keyword in problem_keywords:
            if keyword in problem_description.lower():
                sub_problems.append({
                    "id": f"sub_{keyword}_{len(sub_problems)}",
                    "description": f"{keyword} related task",
                    "priority": "high" if keyword in ["solve", "optimize"] else "medium",
                    "required_capabilities": self._infer_required_capabilities(keyword)
                })

        if not sub_problems:
            sub_problems = [{
                "id": "sub_default_0",
                "description": "General problem analysis",
                "priority": "medium",
                "required_capabilities": {"cognition": 0.6, "action": 0.5}
            }]

        return {
            "problem_description": problem_description,
            "sub_problems": sub_problems,
            "decomposition_strategy": "keyword_based",
            "estimated_steps": len(sub_problems) * 2
        }

    def _infer_required_capabilities(self, keyword: str) -> Dict[str, float]:
        capability_map = {
            "learn": {"learning": 0.8, "cognition": 0.6},
            "explore": {"perception": 0.7, "action": 0.6},
            "solve": {"cognition": 0.8, "action": 0.7},
            "optimize": {"cognition": 0.7, "action": 0.6},
            "create": {"cognition": 0.6, "action": 0.7},
            "analyze": {"cognition": 0.8, "perception": 0.6},
        }
        return capability_map.get(keyword, {"cognition": 0.5, "action": 0.5})

    def chain_of_thought(self, input_vector, goal, max_steps=5, context=None) -> Dict[str, Any]:
        context = context or {}
        thoughts = []
        current_vector = input_vector
        confidence = 0.5

        for step in range(max_steps):
            thought = self._generate_thought(current_vector, goal, step, context)
            thoughts.append(thought)

            if thought.get("confidence", 0.5) >= 0.8 or thought.get("is_terminal"):
                confidence = thought["confidence"]
                break

            current_vector = thought.get("next_state", current_vector)

        return {
            "goal": goal,
            "thought_chain": thoughts,
            "final_confidence": confidence,
            "steps_taken": len(thoughts),
            "converged": confidence >= 0.8
        }

    def _generate_thought(self, state, goal, step, context) -> Dict[str, Any]:
        goal_type = goal.get("goal_type", "general")
        progress = goal.get("progress", 0.0)

        thought_types = [
            "analyze", "hypothesize", "evaluate", "synthesize", "decide"
        ]
        thought_type = thought_types[step % len(thought_types)]

        thought_content = f"Step {step}: {thought_type} - Evaluating progress toward {goal_type} goal (progress: {progress:.2f})"

        if thought_type == "decide":
            is_terminal = True
            confidence = min(1.0, 0.5 + progress * 0.5 + step * 0.1)
        else:
            is_terminal = False
            confidence = min(1.0, 0.3 + step * 0.15)

        return {
            "step": step,
            "type": thought_type,
            "content": thought_content,
            "confidence": confidence,
            "is_terminal": is_terminal,
            "next_state": state,
            "timestamp": time.time()
        }

    def synthesize_knowledge(self, knowledge_fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not knowledge_fragments:
            return {"synthesis": "No knowledge to synthesize", "confidence": 0.0}

        themes = {}
        for fragment in knowledge_fragments:
            for theme in fragment.get("themes", []):
                themes[theme] = themes.get(theme, 0) + 1

        main_theme = max(themes, key=themes.get) if themes else "general"

        synthesis = f"Integrated knowledge around theme: {main_theme}. "
        synthesis += f"Combined {len(knowledge_fragments)} fragments with {len(themes)} distinct themes."

        return {
            "synthesis": synthesis,
            "main_theme": main_theme,
            "themes_detected": themes,
            "confidence": min(1.0, 0.5 + len(knowledge_fragments) * 0.1),
            "fragment_count": len(knowledge_fragments)
        }

    def critical_analysis(self, idea: str, context=None) -> Dict[str, Any]:
        context = context or {}

        strengths = []
        weaknesses = []
        assumptions = []

        if "learn" in idea.lower() or "explore" in idea.lower():
            strengths.append("High learning potential")
            strengths.append("Opportunity for growth")
            weaknesses.append("Potential resource cost")
            assumptions.append("Environment is learnable")

        if "optimize" in idea.lower() or "improve" in idea.lower():
            strengths.append("Efficiency gain potential")
            strengths.append("Continuous improvement")
            weaknesses.append("May require significant effort")
            assumptions.append("Current state can be improved")

        if "risk" in idea.lower() or "danger" in idea.lower():
            strengths.append("Risk awareness")
            weaknesses.append("May be overly cautious")
            assumptions.append("Risk can be mitigated")

        critique = f"Analysis of '{idea}':\n"
        critique += f"Strengths: {', '.join(strengths) if strengths else 'None identified'}\n"
        critique += f"Weaknesses: {', '.join(weaknesses) if weaknesses else 'None identified'}\n"
        critique += f"Assumptions: {', '.join(assumptions) if assumptions else 'None identified'}"

        overall_score = 0.5
        if strengths:
            overall_score += len(strengths) * 0.1
        if weaknesses:
            overall_score -= len(weaknesses) * 0.05

        return {
            "idea": idea,
            "critique": critique,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "assumptions": assumptions,
            "overall_score": min(1.0, max(0.0, overall_score)),
            "recommended_action": "proceed" if overall_score > 0.5 else "reconsider"
        }

    def set_mode(self, mode):
        if isinstance(mode, ThinkingMode):
            self.mode = mode
        elif isinstance(mode, str):
            for m in ThinkingMode:
                if m.value == mode.lower():
                    self.mode = m
                    break

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.reflex_controller.confidence_threshold = self.confidence_threshold

    def learn_from_outcome(self, input_vector, outcome, success):
        self.reflex_controller.learn(input_vector, outcome, success)
        
        if hasattr(outcome, 'thinking_id'):
            self.thinking_engine.reflect_on_thinking(outcome, {"success": success})

    def get_stats(self):
        thinking_stats = self.thinking_engine.get_thinking_stats()
        recent_thinking = list(self.thinking_history)[-10:]
        
        return {
            "mode": self.mode.value,
            "confidence_threshold": self.confidence_threshold,
            "system1_usage": self.system1_count,
            "system2_usage": self.system2_count,
            "total_thinking_cycles": len(self.thinking_history),
            "system1_ratio": self.system1_count / (self.system1_count + self.system2_count) if (self.system1_count + self.system2_count) > 0 else 0.5,
            "reflex_stats": self.reflex_controller.get_activity_summary(),
            "thinking_engine_stats": thinking_stats,
            "current_phase": self.thinking_engine.current_phase.value,
            "recent_avg_confidence": float(np.mean([t["confidence"] for t in recent_thinking])) if recent_thinking else 0.0,
            "pending_problems": len(self.pending_problems),
            "min_confidence_threshold": self.thinking_engine.min_confidence_threshold,
            "max_retries": self.thinking_engine.max_retries,
            "formulation_stats": thinking_stats.get("formulator_stats", {}),
            "hypothesis_stats": thinking_stats.get("hypothesis_stats", {}),
            "deduction_stats": thinking_stats.get("deduction_stats", {}),
            "causal_stats": thinking_stats.get("causal_stats", {}),
            "simulation_stats": thinking_stats.get("simulation_stats", {}),
            "optimizer_stats": thinking_stats.get("optimizer_stats", {})
        }

    def get_recent_thinking(self, limit=10):
        return list(self.thinking_history)[-limit:]

    def resize(self, new_dim):
        self.feature_dim = new_dim
        self.reflex_controller.resize(new_dim)
        self.thinking_engine.resize(new_dim)

    def enter_sleep_mode(self):
        self.reflex_controller.enter_sleep_mode()

    def wake_up(self):
        self.reflex_controller.wake_up()