import time
import numpy as np
from collections import deque
from typing import Dict, Any, List


class IdentityRepresentation:
    def __init__(self, name, role, goals=None, boundaries=None, permissions=None):
        self.name = name
        self.role = role
        self.goals = goals or []
        self.boundaries = boundaries or {}
        self.permissions = permissions or {}
        self.version = "1.0.0"

    def to_dict(self):
        return {
            "name": self.name,
            "role": self.role,
            "goals": self.goals,
            "boundaries": self.boundaries,
            "permissions": self.permissions,
            "version": self.version
        }


class CapabilityProfile:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.task_success_rates = {}
        self.task_latency = {}
        self.resource_consumption = {}
        self.strengths = []
        self.weaknesses = []
        self.learning_progress = {}

    def update_task_performance(self, task_type, success, latency, resources):
        if task_type not in self.task_success_rates:
            self.task_success_rates[task_type] = {"success": 0, "total": 0}
            self.task_latency[task_type] = []
            self.resource_consumption[task_type] = []
        
        self.task_success_rates[task_type]["total"] += 1
        if success:
            self.task_success_rates[task_type]["success"] += 1
        
        self.task_latency[task_type].append(latency)
        self.resource_consumption[task_type].append(resources)
        
        success_rate = self.task_success_rates[task_type]["success"] / self.task_success_rates[task_type]["total"]
        if success_rate > 0.8:
            if task_type not in self.strengths:
                self.strengths.append(task_type)
            if task_type in self.weaknesses:
                self.weaknesses.remove(task_type)
        elif success_rate < 0.4:
            if task_type not in self.weaknesses:
                self.weaknesses.append(task_type)
            if task_type in self.strengths:
                self.strengths.remove(task_type)

    def get_success_rate(self, task_type):
        if task_type not in self.task_success_rates:
            return 0.5
        stats = self.task_success_rates[task_type]
        return stats["success"] / stats["total"] if stats["total"] > 0 else 0.5

    def to_dict(self):
        return {
            "task_success_rates": {k: float(v["success"] / v["total"]) if v["total"] > 0 else 0.5 
                                   for k, v in self.task_success_rates.items()},
            "task_latency": {k: float(np.mean(v)) if v else 0.0 for k, v in self.task_latency.items()},
            "resource_consumption": {k: float(np.mean(v)) if v else 0.0 for k, v in self.resource_consumption.items()},
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "learning_progress": self.learning_progress
        }


class StateRepresentation:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.computational_load = 0.0
        self.memory_usage = 0.0
        self.current_tasks = []
        self.active_skills = []
        self.health_score = 1.0
        self.last_updated = 0

    def update(self, load, memory, tasks, skills, health):
        self.computational_load = max(0.0, min(1.0, load))
        self.memory_usage = max(0.0, min(1.0, memory))
        self.current_tasks = tasks[:10]
        self.active_skills = skills[:10]
        self.health_score = max(0.0, min(1.0, health))
        self.last_updated = time.time()

    def to_dict(self):
        return {
            "computational_load": self.computational_load,
            "memory_usage": self.memory_usage,
            "current_tasks": self.current_tasks,
            "active_skills": self.active_skills,
            "health_score": self.health_score
        }


class HistoryRepresentation:
    def __init__(self):
        self.action_history = deque(maxlen=500)
        self.learning_events = deque(maxlen=200)
        self.evolution_records = deque(maxlen=100)

    def record_action(self, action_type, success, confidence, outcome):
        self.action_history.append({
            "action_type": action_type,
            "success": success,
            "confidence": confidence,
            "outcome": outcome,
            "timestamp": time.time()
        })

    def record_learning(self, topic, type_, details):
        self.learning_events.append({
            "topic": topic,
            "type": type_,
            "details": details,
            "timestamp": time.time()
        })

    def record_evolution(self, level, change_type, details):
        self.evolution_records.append({
            "level": level,
            "change_type": change_type,
            "details": details,
            "timestamp": time.time()
        })

    def get_recent_actions(self, limit=20):
        return list(self.action_history)[-limit:]

    def get_learning_summary(self):
        stats = {"total_events": len(self.learning_events), "types": {}}
        for event in self.learning_events:
            event_type = event["type"]
            stats["types"][event_type] = stats["types"].get(event_type, 0) + 1
        return stats

    def to_dict(self):
        return {
            "action_count": len(self.action_history),
            "learning_events_count": len(self.learning_events),
            "evolution_records_count": len(self.evolution_records),
            "recent_actions": self.get_recent_actions(10),
            "learning_summary": self.get_learning_summary()
        }


class SelfModel:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.identity = IdentityRepresentation("AGI_Agent", "autonomous_agent")
        self.capabilities = CapabilityProfile(feature_dim=feature_dim)
        self.state = StateRepresentation(feature_dim=feature_dim)
        self.history = HistoryRepresentation()
        
        self._update_callbacks = []

        self.self_referential_knowledge = {
            "self_recognition": 0.7,
            "capability_awareness": 0.5,
            "limitation_awareness": 0.4,
            "existence_awareness": 0.6,
            "temporal_continuity": 0.5
        }
        self.introspection_history = deque(maxlen=100)
        self.identity_evolution_history = deque(maxlen=50)
        
        self.boundary_detector = BoundaryDetector()

    def update_identity(self, name=None, role=None, goals=None, boundaries=None):
        if name:
            self.identity.name = name
        if role:
            self.identity.role = role
        if goals:
            self.identity.goals = goals
        if boundaries:
            self.identity.boundaries = boundaries

    def assess_capability(self, task_type):
        success_rate = self.capabilities.get_success_rate(task_type)
        is_strength = task_type in self.capabilities.strengths
        is_weakness = task_type in self.capabilities.weaknesses
        
        return {
            "task_type": task_type,
            "success_rate": success_rate,
            "is_strength": is_strength,
            "is_weakness": is_weakness,
            "can_handle": success_rate >= 0.5
        }

    def check_feasibility(self, action_plan):
        action_type = action_plan.get("action", "unknown")
        capability = self.assess_capability(action_type)
        
        resource_required = action_plan.get("resources", 0.5)
        resources_available = 1.0 - self.state.computational_load
        
        return {
            "capable": capability["can_handle"],
            "success_rate": capability["success_rate"],
            "resources_available": resources_available >= resource_required,
            "health_ok": self.state.health_score > 0.3,
            "overall_feasible": capability["can_handle"] and resources_available >= resource_required and self.state.health_score > 0.3
        }

    def get_self_summary(self):
        return {
            "identity": self.identity.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "state": self.state.to_dict(),
            "history": self.history.to_dict(),
            "self_referential_knowledge": self.self_referential_knowledge
        }

    def snapshot(self):
        return {
            "identity": self.identity.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "state": self.state.to_dict()
        }

    def introspect(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        introspection_result = {
            "timestamp": time.time(),
            "self_recognition": self._assess_self_recognition(),
            "capability_awareness": self._assess_capability_awareness(),
            "limitation_awareness": self._assess_limitation_awareness(),
            "existence_awareness": self._assess_existence_awareness(),
            "temporal_continuity": self._assess_temporal_continuity(),
            "context": context or {}
        }

        self.introspection_history.append(introspection_result)

        for key in self.self_referential_knowledge:
            if key in introspection_result:
                self.self_referential_knowledge[key] = (
                    0.8 * self.self_referential_knowledge[key] +
                    0.2 * introspection_result[key]
                )

        return introspection_result

    def _assess_self_recognition(self) -> float:
        action_count = len(self.history.action_history)
        if action_count < 10:
            return 0.3

        successes = sum(1 for a in self.history.action_history if a["success"])
        success_rate = successes / action_count

        if hasattr(self.capabilities, 'strengths') and self.capabilities.strengths:
            return min(1.0, 0.5 + success_rate * 0.3 + len(self.capabilities.strengths) * 0.1)

        return min(1.0, 0.5 + success_rate * 0.3)

    def _assess_capability_awareness(self) -> float:
        if not self.capabilities.task_success_rates:
            return 0.3

        rates = list(self.capabilities.task_success_rates.values())
        avg_rate = np.mean([r["success"] / r["total"] if r["total"] > 0 else 0.5 for r in rates])

        strength_count = len(self.capabilities.strengths) if hasattr(self.capabilities, 'strengths') else 0
        weakness_count = len(self.capabilities.weaknesses) if hasattr(self.capabilities, 'weaknesses') else 0

        return min(1.0, avg_rate * 0.6 + (strength_count + weakness_count) * 0.1)

    def _assess_limitation_awareness(self) -> float:
        if not hasattr(self.capabilities, 'weaknesses') or not self.capabilities.weaknesses:
            return 0.2

        weakness_count = len(self.capabilities.weaknesses)
        action_count = len(self.history.action_history)

        if action_count == 0:
            return 0.2

        failures = sum(1 for a in self.history.action_history if not a["success"])
        failure_rate = failures / action_count

        return min(1.0, weakness_count * 0.15 + failure_rate * 0.4)

    def _assess_existence_awareness(self) -> float:
        action_count = len(self.history.action_history)
        learning_count = len(self.history.learning_events)

        if action_count == 0:
            return 0.3

        return min(1.0, 0.4 + action_count * 0.001 + learning_count * 0.002)

    def _assess_temporal_continuity(self) -> float:
        actions = list(self.history.action_history)
        if len(actions) < 5:
            return 0.3

        timestamps = [a["timestamp"] for a in actions]
        gaps = []
        for i in range(1, len(timestamps)):
            gaps.append(timestamps[i] - timestamps[i-1])

        avg_gap = np.mean(gaps) if gaps else 1.0
        normalized_gap = min(1.0, avg_gap / 3600)

        return min(1.0, 0.5 + (1.0 - normalized_gap) * 0.3)

    def query_self_knowledge(self, query_type: str) -> Dict[str, Any]:
        queries = {
            "who_am_i": self._query_who_am_i,
            "what_can_i_do": self._query_what_can_i_do,
            "what_cant_i_do": self._query_what_cant_i_do,
            "what_have_i_learned": self._query_what_have_i_learned,
            "what_are_my_goals": self._query_what_are_my_goals,
            "how_am_i_feeling": self._query_how_am_i_feeling,
        }

        if query_type in queries:
            return queries[query_type]()

        return {"error": f"Unknown query type: {query_type}"}

    def _query_who_am_i(self) -> Dict[str, Any]:
        return {
            "query": "who_am_i",
            "identity": self.identity.to_dict(),
            "self_recognition_score": self.self_referential_knowledge["self_recognition"],
            "existence_awareness": self.self_referential_knowledge["existence_awareness"]
        }

    def _query_what_can_i_do(self) -> Dict[str, Any]:
        strengths = self.capabilities.strengths if hasattr(self.capabilities, 'strengths') else []
        success_rates = {}
        for task_type, stats in self.capabilities.task_success_rates.items():
            if stats["total"] > 0:
                success_rates[task_type] = float(stats["success"] / stats["total"])

        return {
            "query": "what_can_i_do",
            "strengths": strengths,
            "success_rates": success_rates,
            "capability_awareness": self.self_referential_knowledge["capability_awareness"]
        }

    def _query_what_cant_i_do(self) -> Dict[str, Any]:
        weaknesses = self.capabilities.weaknesses if hasattr(self.capabilities, 'weaknesses') else []
        failure_rates = {}
        for task_type, stats in self.capabilities.task_success_rates.items():
            if stats["total"] > 0:
                rate = stats["success"] / stats["total"]
                if rate < 0.5:
                    failure_rates[task_type] = float(1.0 - rate)

        return {
            "query": "what_cant_i_do",
            "weaknesses": weaknesses,
            "failure_rates": failure_rates,
            "limitation_awareness": self.self_referential_knowledge["limitation_awareness"]
        }

    def _query_what_have_i_learned(self) -> Dict[str, Any]:
        learning_summary = self.history.get_learning_summary()
        recent_actions = self.history.get_recent_actions(10)

        return {
            "query": "what_have_i_learned",
            "learning_events": learning_summary,
            "recent_actions": recent_actions,
            "temporal_continuity": self.self_referential_knowledge["temporal_continuity"]
        }

    def _query_what_are_my_goals(self) -> Dict[str, Any]:
        return {
            "query": "what_are_my_goals",
            "goals": self.identity.goals,
            "boundaries": self.identity.boundaries,
            "permissions": self.identity.permissions
        }

    def _query_how_am_i_feeling(self) -> Dict[str, Any]:
        return {
            "query": "how_am_i_feeling",
            "health_score": self.state.health_score,
            "computational_load": self.state.computational_load,
            "memory_usage": self.state.memory_usage,
            "overall_wellbeing": (1.0 - self.state.computational_load) * 0.5 + self.state.health_score * 0.5
        }

    def evolve_identity(self, experiences: List[Dict[str, Any]]):
        if not experiences:
            return

        original_identity = self.snapshot()

        for exp in experiences:
            if exp.get("type") == "success" and exp.get("impact") == "major":
                if "growth" not in self.identity.goals:
                    self.identity.goals.append("growth")
            elif exp.get("type") == "failure" and exp.get("impact") == "major":
                if "improvement" not in self.identity.goals:
                    self.identity.goals.append("improvement")

        self.identity_evolution_history.append({
            "timestamp": time.time(),
            "original": original_identity,
            "updated": self.snapshot(),
            "triggering_experiences": len(experiences)
        })

    def register_update_callback(self, callback):
        self._update_callbacks.append(callback)

    def resize(self, new_dim):
        self.feature_dim = new_dim
        self.capabilities.feature_dim = new_dim
        self.state.feature_dim = new_dim

    def predict_self_trajectory(self, internal_state_vec, steps=5):
        trajectory = []
        current_state = np.array(internal_state_vec, dtype=np.float32)
        
        for _ in range(steps):
            next_state = current_state * 0.9 + np.random.normal(0, 0.05, current_state.shape)
            next_state = np.clip(next_state, 0.0, 1.0)
            trajectory.append(next_state.tolist())
            current_state = next_state
        
        return trajectory

    def generate_self_reflection(self, internal_state_vec, trajectory, action_result=None):
        avg_trajectory = np.mean(trajectory, axis=0)
        detected_problems = []
        
        if avg_trajectory[0] < 0.3:
            detected_problems.append("low_energy")
        if avg_trajectory[2] < 0.4:
            detected_problems.append("security_concern")
        if avg_trajectory[3] < 0.3:
            detected_problems.append("low_curiosity")
        
        recommendations = []
        if "low_energy" in detected_problems:
            recommendations.append("increase_energy")
        if action_result is not None and hasattr(action_result, 'flatten'):
            action_avg = float(np.mean(action_result.flatten()))
            if action_avg < 0.1:
                detected_problems.append("low_action_output")
                recommendations.append("increase_action")
        
        return {
            "self_reflection_score": float(np.mean(avg_trajectory)),
            "trajectory_prediction": trajectory,
            "detected_problems": detected_problems,
            "recommendations": recommendations
        }


    def detect_future_problems(self, trajectory):
        problems = []
        
        if isinstance(trajectory, list) and len(trajectory) > 0:
            last_state = trajectory[-1]
            if isinstance(last_state, list):
                avg_value = np.mean(last_state)
                if avg_value < 0.3:
                    problems.append("potential_system_degradation")
                if avg_value > 0.8:
                    problems.append("potential_overload")
        
        return problems

    def get_learning_summary(self):
        learning_events = []
        for exp in self.history.experiences:
            if exp.get("type") == "learning":
                learning_events.append(exp)
        
        return {
            "total_learning_events": len(learning_events),
            "recent_events": learning_events[-5:]
        }

    def get_self_model_state(self):
        return {
            "identity": self.identity.to_dict(),
            "capabilities": {
                "task_success_rates": self.capabilities.task_success_rates,
                "strengths": self.capabilities.strengths if hasattr(self.capabilities, 'strengths') else [],
                "weaknesses": self.capabilities.weaknesses if hasattr(self.capabilities, 'weaknesses') else []
            },
            "state": {
                "health_score": self.state.health_score,
                "computational_load": self.state.computational_load,
                "memory_usage": self.state.memory_usage
            },
            "self_referential_knowledge": self.self_referential_knowledge
        }


class BoundaryDetector:
    def __init__(self):
        self.self_states = deque(maxlen=100)
    
    def record_self_state(self, state):
        self.self_states.append({
            "timestamp": time.time(),
            "state": state.flatten().tolist()[:10] if hasattr(state, 'flatten') else list(state)[:10]
        })