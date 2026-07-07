import numpy as np
from collections import deque


class CausalChain:
    def __init__(self, chain_id, cause, effect, intermediates=None, strength=0.5, evidence=None):
        self.chain_id = chain_id
        self.cause = cause
        self.effect = effect
        self.intermediates = intermediates or []
        self.strength = strength
        self.evidence = evidence or []
        self.confirmed = False

    def to_dict(self):
        return {
            "chain_id": self.chain_id,
            "cause": self.cause,
            "effect": self.effect,
            "intermediates": self.intermediates,
            "strength": self.strength,
            "evidence_count": len(self.evidence),
            "confirmed": self.confirmed
        }


class CausalReasoner:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.causal_chains = {}
        self.causal_graph = {}
        self.reasoning_history = deque(maxlen=200)

    def _build_causal_graph(self, observations):
        for obs in observations:
            if "cause" in obs and "effect" in obs:
                cause = obs["cause"]
                effect = obs["effect"]
                
                if cause not in self.causal_graph:
                    self.causal_graph[cause] = []
                self.causal_graph[cause].append({
                    "effect": effect,
                    "strength": obs.get("strength", 0.5),
                    "evidence": obs.get("evidence", [])
                })

    def infer_causal_chain(self, hypothesis, problem):
        chain_id = f"chain_{len(self.causal_chains) + 1}"
        cause = hypothesis.solution.get("action", "unknown_action")
        effect = problem.goal
        
        intermediates = []
        strength = hypothesis.expected_benefit * (1 - hypothesis.potential_risk)
        
        if cause in self.causal_graph:
            for link in self.causal_graph[cause]:
                if link["effect"] == effect:
                    strength = max(strength, link["strength"])
                    intermediates.append(link)
        
        chain = CausalChain(
            chain_id=chain_id,
            cause=cause,
            effect=effect,
            intermediates=[i["effect"] for i in intermediates],
            strength=strength,
            evidence=[str(i) for i in intermediates]
        )
        
        self.causal_chains[chain_id] = chain
        
        self.reasoning_history.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "chain_id": chain_id,
            "cause": cause,
            "effect": effect,
            "strength": strength,
            "timestamp": np.random.randint(1000000)
        })
        
        return chain

    def evaluate_counterfactual(self, hypothesis, problem, alternative_action):
        original_effect = problem.goal
        original_strength = hypothesis.expected_benefit
        
        counterfactual_strength = original_strength * 0.7
        
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "original_action": hypothesis.solution.get("action"),
            "alternative_action": alternative_action,
            "original_effect": original_effect,
            "counterfactual_strength": counterfactual_strength,
            "recommendation": "stick_with_original" if counterfactual_strength < original_strength else "consider_alternative"
        }

    def detect_spurious_correlation(self, hypothesis):
        features = hypothesis.solution
        spurious = []
        
        for key1, val1 in features.items():
            for key2, val2 in features.items():
                if key1 != key2:
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        if abs(val1 - val2) < 0.01:
                            spurious.append((key1, key2))
        
        return {"hypothesis_id": hypothesis.hypothesis_id, "spurious_pairs": spurious}

    def validate_causal_chain(self, chain_id, evidence):
        if chain_id in self.causal_chains:
            chain = self.causal_chains[chain_id]
            chain.evidence.extend(evidence)
            
            if len(chain.evidence) >= 2:
                chain.confirmed = True
                chain.strength = min(1.0, chain.strength + 0.1)
            
            return chain.to_dict()
        return None

    def get_causal_stats(self):
        stats = {
            "total_chains": len(self.causal_chains),
            "confirmed_chains": len([c for c in self.causal_chains.values() if c.confirmed]),
            "avg_strength": 0.0,
            "reasoning_count": len(self.reasoning_history)
        }
        
        if self.causal_chains:
            stats["avg_strength"] = float(np.mean([c.strength for c in self.causal_chains.values()]))
        
        return stats

    def resize(self, new_dim):
        self.feature_dim = new_dim