import numpy as np
from collections import deque


class Hypothesis:
    def __init__(self, hypothesis_id, description, solution, premises=None, expected_benefit=0.0, potential_risk=0.0, confidence=0.5):
        self.hypothesis_id = hypothesis_id
        self.description = description
        self.solution = solution
        self.premises = premises or []
        self.expected_benefit = expected_benefit
        self.potential_risk = potential_risk
        self.confidence = confidence
        self.validation_status = "pending"
        self.validation_score = 0.0
        self.deduction_path = []

    def to_dict(self):
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "solution": self.solution,
            "premises": self.premises,
            "expected_benefit": self.expected_benefit,
            "potential_risk": self.potential_risk,
            "confidence": self.confidence,
            "validation_status": self.validation_status,
            "validation_score": self.validation_score
        }


class HypothesisGenerator:
    def __init__(self, feature_dim=16, max_hypotheses=10):
        self.feature_dim = feature_dim
        self.max_hypotheses = max_hypotheses
        self.hypotheses = {}
        self.hypothesis_history = deque(maxlen=200)
        self.solution_templates = []
        
        self._init_default_templates()

    def _init_default_templates(self):
        self.solution_templates = [
            {
                "name": "direct_correction",
                "description": "直接修正偏差",
                "applicable_when": lambda p: p.trigger_type.value == "deviation" and p.constraints.get("deviation_severity", 0) < 0.5,
                "solution_generator": lambda p: {"action": "correct", "target": p.goal, "method": "direct"}
            },
            {
                "name": "stepwise_approach",
                "description": "分步逼近目标",
                "applicable_when": lambda p: p.trigger_type.value == "deviation" and p.constraints.get("deviation_severity", 0) >= 0.5,
                "solution_generator": lambda p: {"action": "approach", "target": p.goal, "steps": 3}
            },
            {
                "name": "explore_unknown",
                "description": "探索未知领域",
                "applicable_when": lambda p: p.trigger_type.value == "unknown",
                "solution_generator": lambda p: {"action": "explore", "direction": p.current_state, "depth": 2}
            },
            {
                "name": "optimize_opportunity",
                "description": "利用优化机会",
                "applicable_when": lambda p: p.trigger_type.value == "opportunity",
                "solution_generator": lambda p: {"action": "optimize", "target": p.goal, "gain": p.constraints.get("opportunity_gain", 0.3)}
            },
            {
                "name": "delegate_to_skill",
                "description": "委托专业技能处理",
                "applicable_when": lambda p: len(p.boundaries.get("allowed_actions", [])) > 0,
                "solution_generator": lambda p: {"action": "delegate", "skill": p.boundaries["allowed_actions"][0]}
            }
        ]

    def _generate_from_templates(self, problem):
        hypotheses = []
        
        for template in self.solution_templates:
            if template["applicable_when"](problem):
                solution = template["solution_generator"](problem)
                hypothesis = Hypothesis(
                    hypothesis_id=f"hyp_{len(self.hypotheses) + 1}_{template['name']}",
                    description=template["description"],
                    solution=solution,
                    premises=[f"Template: {template['name']}", f"Trigger: {problem.trigger_type.value}"],
                    expected_benefit=self._estimate_benefit(problem, template),
                    potential_risk=self._estimate_risk(problem, template),
                    confidence=self._estimate_confidence(problem, template)
                )
                hypotheses.append(hypothesis)
        
        return hypotheses

    def _estimate_benefit(self, problem, template):
        base_benefit = {
            "direct_correction": 0.7,
            "stepwise_approach": 0.6,
            "explore_unknown": 0.5,
            "optimize_opportunity": 0.8,
            "delegate_to_skill": 0.75
        }.get(template["name"], 0.5)
        
        if problem.trigger_type.value == "opportunity":
            base_benefit *= 1.2
        
        return min(1.0, base_benefit)

    def _estimate_risk(self, problem, template):
        base_risk = {
            "direct_correction": 0.2,
            "stepwise_approach": 0.15,
            "explore_unknown": 0.4,
            "optimize_opportunity": 0.25,
            "delegate_to_skill": 0.3
        }.get(template["name"], 0.3)
        
        return min(1.0, base_risk)

    def _estimate_confidence(self, problem, template):
        base_confidence = {
            "direct_correction": 0.8,
            "stepwise_approach": 0.7,
            "explore_unknown": 0.4,
            "optimize_opportunity": 0.75,
            "delegate_to_skill": 0.65
        }.get(template["name"], 0.5)
        
        return min(1.0, base_confidence)

    def _generate_from_knowledge(self, problem, knowledge_graph=None):
        hypotheses = []
        
        if knowledge_graph:
            similar_problems = knowledge_graph.find_similar_nodes(
                np.array(problem.current_state).flatten(),
                threshold=0.5
            )
            
            for similar_node in similar_problems[:3]:
                node_data = knowledge_graph.get_node_data(similar_node)
                if node_data and "solution" in node_data:
                    hypothesis = Hypothesis(
                        hypothesis_id=f"hyp_{len(self.hypotheses) + 1}_knowledge",
                        description=f"基于历史案例 {similar_node}",
                        solution=node_data["solution"],
                        premises=[f"Knowledge node: {similar_node}"],
                        expected_benefit=node_data.get("success_rate", 0.6),
                        potential_risk=0.2,
                        confidence=node_data.get("confidence", 0.6)
                    )
                    hypotheses.append(hypothesis)
        
        return hypotheses

    def generate_hypotheses(self, problem, knowledge_graph=None):
        hypotheses = []
        
        template_hypotheses = self._generate_from_templates(problem)
        hypotheses.extend(template_hypotheses)
        
        knowledge_hypotheses = self._generate_from_knowledge(problem, knowledge_graph)
        hypotheses.extend(knowledge_hypotheses)
        
        hypotheses = sorted(hypotheses, key=lambda x: -x.expected_benefit * (1 - x.potential_risk))
        hypotheses = hypotheses[:self.max_hypotheses]
        
        for hyp in hypotheses:
            self.hypotheses[hyp.hypothesis_id] = hyp
        
        self.hypothesis_history.append({
            "problem_id": problem.problem_id,
            "generated_count": len(hypotheses),
            "avg_confidence": float(np.mean([h.confidence for h in hypotheses])) if hypotheses else 0.0,
            "timestamp": np.random.randint(1000000)
        })
        
        return hypotheses

    def validate_hypothesis(self, hypothesis_id, validation_result):
        if hypothesis_id in self.hypotheses:
            hyp = self.hypotheses[hypothesis_id]
            hyp.validation_status = "validated" if validation_result else "invalidated"
            hyp.validation_score = validation_result

    def get_hypothesis_stats(self):
        stats = {
            "total_hypotheses": len(self.hypotheses),
            "validated_count": len([h for h in self.hypotheses.values() if h.validation_status == "validated"]),
            "invalidated_count": len([h for h in self.hypotheses.values() if h.validation_status == "invalidated"]),
            "avg_confidence": 0.0,
            "avg_benefit": 0.0,
            "avg_risk": 0.0
        }
        
        if self.hypotheses:
            stats["avg_confidence"] = float(np.mean([h.confidence for h in self.hypotheses.values()]))
            stats["avg_benefit"] = float(np.mean([h.expected_benefit for h in self.hypotheses.values()]))
            stats["avg_risk"] = float(np.mean([h.potential_risk for h in self.hypotheses.values()]))
        
        return stats

    def resize(self, new_dim):
        self.feature_dim = new_dim