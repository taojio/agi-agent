import numpy as np
from collections import deque
from .self_model import SelfModel
from .cognitive_monitor import CognitiveMonitor
from .strategy_regulator import StrategyRegulator
from .boundary_guardian import BoundaryGuardian
from .meta_learning_engine import MetaLearningEngine


class MetaCognitiveOrchestrator:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.self_model = SelfModel(feature_dim=feature_dim)
        self.cognitive_monitor = CognitiveMonitor()
        self.strategy_regulator = StrategyRegulator()
        self.boundary_guardian = BoundaryGuardian()
        self.meta_learning_engine = MetaLearningEngine()
        
        self.meta_cognitive_history = deque(maxlen=200)
        self.is_active = True

    def monitor_and_regulate(self, context):
        if not self.is_active:
            return {"status": "inactive"}

        self.cognitive_monitor.monitor_resources(
            cpu=context.get("cpu_load", 0.0),
            memory=context.get("memory_load", 0.0),
            latency=context.get("latency", 0.0)
        )

        self.strategy_regulator.adjust_thinking_strategy(context)
        self.strategy_regulator.adjust_action_strategy(context)
        self.strategy_regulator.adjust_resource_allocation(context)

        self.self_model.state.update(
            load=context.get("cpu_load", 0.0),
            memory=context.get("memory_load", 0.0),
            tasks=context.get("current_tasks", []),
            skills=context.get("active_skills", []),
            health=context.get("health_score", 1.0)
        )

        return {
            "strategy": self.strategy_regulator.get_current_strategy(),
            "resource_health": self.cognitive_monitor.resource_monitor.check_resource_health(),
            "alerts": self.cognitive_monitor.get_alerts(5),
            "self_model_state": self.self_model.state.to_dict()
        }

    def evaluate_decision(self, action_plan):
        feasibility = self.self_model.check_feasibility(action_plan)
        boundaries = self.boundary_guardian.check_all_boundaries(action_plan)
        
        evaluation = {
            "feasibility": feasibility,
            "boundaries": boundaries,
            "overall_approved": feasibility["overall_feasible"] and boundaries["overall_allowed"]
        }

        if evaluation["overall_approved"]:
            self.cognitive_monitor.monitor_execution(
                execution_id=f"exec_{np.random.randint(1000000)}",
                action_plan=action_plan
            )

        return evaluation

    def reflect_on_outcome(self, action_plan, outcome, success):
        action_type = action_plan.get("action", "unknown")
        
        self.self_model.capabilities.update_task_performance(
            task_type=action_type,
            success=success,
            latency=outcome.get("latency", 0.0),
            resources=outcome.get("resources", 0.0)
        )

        self.self_model.history.record_action(
            action_type=action_type,
            success=success,
            confidence=outcome.get("confidence", 0.5),
            outcome=outcome
        )

        self.meta_learning_engine.perform_meta_learning([{
            "strategy": action_type,
            "success": success,
            "context": action_plan
        }])

        reflection = {
            "action_type": action_type,
            "success": success,
            "capability_update": self.self_model.assess_capability(action_type),
            "learning_recorded": True
        }

        self.meta_cognitive_history.append({
            "reflection": reflection,
            "timestamp": np.random.randint(1000000)
        })

        return reflection

    def get_self_awareness(self):
        return {
            "identity": self.self_model.identity.to_dict(),
            "capabilities": self.self_model.capabilities.to_dict(),
            "current_state": self.self_model.state.to_dict(),
            "history_summary": self.self_model.history.to_dict()
        }

    def get_all_stats(self):
        return {
            "self_model": self.self_model.get_self_summary(),
            "cognitive_monitor": self.cognitive_monitor.get_all_stats(),
            "strategy_regulator": self.strategy_regulator.get_strategy_stats(),
            "boundary_guardian": self.boundary_guardian.get_boundary_stats(),
            "meta_learning": self.meta_learning_engine.get_meta_learning_stats(),
            "meta_cognitive_cycles": len(self.meta_cognitive_history)
        }

    def get_stats(self):
        all_stats = self.get_all_stats()
        self_summary = self.self_model.get_self_summary()
        
        return {
            **all_stats,
            "self_model_summary": {
                "identity": {
                    "name": self_summary["identity"]["name"],
                    "role": self_summary["identity"]["role"],
                    "version": self_summary["identity"]["version"],
                    "goal_count": len(self_summary["identity"]["goals"]),
                    "boundary_count": len(self_summary["identity"]["boundaries"]),
                    "permission_count": len(self_summary["identity"]["permissions"])
                },
                "capabilities": {
                    "total_tasks": len(self_summary["capabilities"]["task_success_rates"]),
                    "strengths": self_summary["capabilities"]["strengths"],
                    "weaknesses": self_summary["capabilities"]["weaknesses"],
                    "avg_success_rate": float(np.mean(list(self_summary["capabilities"]["task_success_rates"].values()))) if self_summary["capabilities"]["task_success_rates"] else 0.5
                },
                "state": {
                    "computational_load": self_summary["state"]["computational_load"],
                    "memory_usage": self_summary["state"]["memory_usage"],
                    "health_score": self_summary["state"]["health_score"],
                    "current_task_count": len(self_summary["state"]["current_tasks"]),
                    "active_skill_count": len(self_summary["state"]["active_skills"])
                },
                "history": {
                    "action_count": self_summary["history"]["action_count"],
                    "learning_events": self_summary["history"]["learning_events_count"],
                    "evolution_records": self_summary["history"]["evolution_records_count"]
                }
            },
            "cognitive_monitor_summary": {
                "alerts_count": len(self.cognitive_monitor.get_alerts(100)),
                "resource_health": self.cognitive_monitor.resource_monitor.check_resource_health(),
                "execution_tracking_count": len(getattr(self.cognitive_monitor.execution_tracker, 'execution_history', []))
            },
            "strategy_regulator_summary": {
                "current_thinking_strategy": self.strategy_regulator.get_current_strategy().get("thinking", "auto"),
                "current_action_strategy": self.strategy_regulator.get_current_strategy().get("action", "balanced"),
                "strategy_changes": len(getattr(self.strategy_regulator, 'strategy_history', []))
            },
            "boundary_guardian_summary": {
                "violations_count": len(self.boundary_guardian.safety_boundary.violation_history),
                "safety_boundaries": len(self.boundary_guardian.safety_boundary.red_lines),
                "permission_boundaries": len(self.boundary_guardian.permission_boundary.permissions),
                "ethical_boundaries": len(self.boundary_guardian.ethical_boundary.principles)
            },
            "meta_learning_summary": {
                "learning_cycles": getattr(self.meta_learning_engine, 'meta_learning_cycles', 0),
                "strategies_optimized": len(getattr(self.meta_learning_engine, 'strategy_performance', {})),
                "current_strategy_effectiveness": getattr(self.meta_learning_engine, 'get_current_effectiveness', lambda: 0.0)()
            }
        }

    def pause(self):
        self.is_active = False

    def resume(self):
        self.is_active = True

    def get_alerts(self, limit=10):
        return self.cognitive_monitor.get_alerts(limit)

    def resize(self, new_dim):
        self.feature_dim = new_dim
        self.self_model.resize(new_dim)