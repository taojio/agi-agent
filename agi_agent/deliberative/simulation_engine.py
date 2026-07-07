import numpy as np
from collections import deque


class SimulationResult:
    def __init__(self, simulation_id, hypothesis_id, success_rate=0.0, estimated_time=0.0, resource_consumption=0.0, risk_score=0.0, detailed_trace=None):
        self.simulation_id = simulation_id
        self.hypothesis_id = hypothesis_id
        self.success_rate = success_rate
        self.estimated_time = estimated_time
        self.resource_consumption = resource_consumption
        self.risk_score = risk_score
        self.detailed_trace = detailed_trace or []
        self.valid = success_rate > 0.5

    def to_dict(self):
        return {
            "simulation_id": self.simulation_id,
            "hypothesis_id": self.hypothesis_id,
            "success_rate": self.success_rate,
            "estimated_time": self.estimated_time,
            "resource_consumption": self.resource_consumption,
            "risk_score": self.risk_score,
            "valid": self.valid
        }


class SimulationEngine:
    def __init__(self, feature_dim=16, max_simulation_steps=10):
        self.feature_dim = feature_dim
        self.max_simulation_steps = max_simulation_steps
        self.simulation_history = deque(maxlen=200)
        self.historical_data = {}

    def _simulate_step(self, current_state, action, transition_model=None):
        state = np.array(current_state).flatten()
        
        if transition_model is not None:
            next_state = transition_model(state, action)
        else:
            if isinstance(action, dict):
                numeric_values = []
                for v in action.values():
                    if isinstance(v, (int, float)):
                        numeric_values.append(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, (int, float)):
                                numeric_values.append(item)
                
                if len(numeric_values) == 0:
                    action_values = np.zeros(len(state))
                else:
                    action_values = np.array(numeric_values).flatten()
            else:
                action_values = np.array(action).flatten()
            
            action_values = action_values[:len(state)]
            if len(action_values) < len(state):
                action_values = np.pad(action_values, (0, len(state) - len(action_values)))
            
            next_state = state * 0.9 + action_values.astype(float) * 0.1
        
        success_prob = min(1.0, max(0.0, 0.7 + np.random.normal(0, 0.1)))
        
        return next_state, success_prob

    def simulate(self, hypothesis, problem):
        simulation_id = f"sim_{len(self.simulation_history) + 1}"
        current_state = np.array(problem.current_state).flatten()
        target_state = np.array(problem.goal).flatten() if isinstance(problem.goal, (list, np.ndarray)) else current_state
        action = hypothesis.solution.get("action", "")
        
        trace = []
        total_success_prob = 1.0
        resource_used = 0.0
        elapsed_time = 0.0
        
        for step in range(self.max_simulation_steps):
            next_state, success_prob = self._simulate_step(current_state, hypothesis.solution)
            
            distance = np.mean(np.abs(next_state[:min(len(next_state), len(target_state))] - target_state[:min(len(next_state), len(target_state))]))
            
            trace.append({
                "step": step,
                "state": next_state[:5].tolist(),
                "distance": float(distance),
                "success_prob": float(success_prob)
            })
            
            total_success_prob *= success_prob
            resource_used += 0.1
            elapsed_time += 1.0
            
            if distance < 0.1:
                break
            
            current_state = next_state
        
        risk_score = hypothesis.potential_risk * (1 - total_success_prob)
        success_rate = total_success_prob * (1 - risk_score)
        
        result = SimulationResult(
            simulation_id=simulation_id,
            hypothesis_id=hypothesis.hypothesis_id,
            success_rate=success_rate,
            estimated_time=elapsed_time,
            resource_consumption=resource_used,
            risk_score=risk_score,
            detailed_trace=trace
        )
        
        self.simulation_history.append({
            "simulation_id": simulation_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "success_rate": success_rate,
            "risk_score": risk_score,
            "steps": step + 1,
            "timestamp": np.random.randint(1000000)
        })
        
        return result

    def batch_simulate(self, hypotheses, problem):
        results = []
        for hypothesis in hypotheses:
            result = self.simulate(hypothesis, problem)
            results.append(result)
        return results

    def get_simulation_stats(self):
        stats = {
            "total_simulations": len(self.simulation_history),
            "avg_success_rate": 0.0,
            "avg_risk_score": 0.0,
            "avg_steps": 0.0
        }
        
        if self.simulation_history:
            recent = list(self.simulation_history)[-20:]
            stats["avg_success_rate"] = float(np.mean([s["success_rate"] for s in recent]))
            stats["avg_risk_score"] = float(np.mean([s["risk_score"] for s in recent]))
            stats["avg_steps"] = float(np.mean([s["steps"] for s in recent]))
        
        return stats

    def resize(self, new_dim):
        self.feature_dim = new_dim