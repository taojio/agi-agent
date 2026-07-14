import numpy as np
from collections import deque
from ..config.settings import FREE_ENERGY_THRESHOLD, NOVELTY_THRESHOLD


class SelfMonitor:
    def __init__(self):
        self.confidence_history = deque(maxlen=50)
        self.success_history = deque(maxlen=100)
        self.task_progress = {}
        self.knowledge_boundary = {}
        self.cognitive_impasses = deque(maxlen=20)

    def assess_confidence(self, prediction, uncertainty=0.0):
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))
        self.confidence_history.append(confidence)
        return confidence

    def assess_knowledge_boundary(self, feature_key, confidence):
        if confidence < 0.3:
            self.knowledge_boundary[feature_key] = "unknown"
            return False
        elif confidence < 0.6:
            self.knowledge_boundary[feature_key] = "partial"
            return False
        else:
            self.knowledge_boundary[feature_key] = "known"
            return True

    def update_task_progress(self, task_id, progress, max_progress=1.0):
        self.task_progress[task_id] = {
            "current": progress,
            "max": max_progress,
            "ratio": progress / max_progress if max_progress > 0 else 0.0
        }
        return self.task_progress[task_id]["ratio"]

    def detect_cognitive_impasse(self, confidence, fe, task_progress_ratio):
        is_stuck = confidence < 0.3 and fe > FREE_ENERGY_THRESHOLD and task_progress_ratio < 0.1
        if is_stuck:
            impasse_record = {
                "timestamp": np.random.randint(1000000),
                "confidence": confidence,
                "free_energy": fe,
                "task_progress": task_progress_ratio
            }
            self.cognitive_impasses.append(impasse_record)
            return True, impasse_record
        return False, None

    def get_confidence_trend(self):
        if len(self.confidence_history) < 5:
            return 0.0
        recent = list(self.confidence_history)[-5:]
        return (recent[-1] - recent[0]) / (recent[0] + 1e-8)


class SelfReflection:
    def __init__(self):
        self.decision_history = deque(maxlen=200)
        self.strategy_rules = []
        self.rule_usage_count = {}
        self.rule_success_rate = {}

    def record_decision(self, decision, context, outcome, reward, confidence):
        record = {
            "decision": decision,
            "context": context,
            "outcome": outcome,
            "reward": reward,
            "confidence": confidence,
            "success": reward > 0
        }
        self.decision_history.append(record)

    def attribute_decision(self, decision_idx=-1):
        if not self.decision_history:
            return None

        record = self.decision_history[decision_idx]
        context = record["context"]
        outcome = record["outcome"]
        reward = record["reward"]

        attribution = {
            "decision": record["decision"],
            "confidence_at_decision": record["confidence"],
            "outcome": outcome,
            "reward": reward,
            "factors": [],
            "recommendations": []
        }

        if reward > 0.5:
            attribution["result"] = "success"
            attribution["factors"].append("High confidence decision matched expected outcome")
            attribution["recommendations"].append("Consider consolidating this strategy as a rule")
        elif reward > 0:
            attribution["result"] = "partial_success"
            attribution["factors"].append("Strategy worked but confidence was moderate")
            attribution["recommendations"].append("Refine strategy parameters")
        else:
            attribution["result"] = "failure"
            if record["confidence"] > 0.7:
                attribution["factors"].append("Overconfidence led to incorrect decision")
                attribution["recommendations"].append("Reduce confidence threshold for similar decisions")
            else:
                attribution["factors"].append("Insufficient knowledge for this context")
                attribution["recommendations"].append("Trigger learning for this context")

        return attribution

    def consolidate_rule(self, condition, action, confidence_threshold=0.8):
        rule_id = f"rule_{len(self.strategy_rules) + 1}"
        rule = {
            "id": rule_id,
            "condition": condition,
            "action": action,
            "confidence_threshold": confidence_threshold,
            "usage_count": 0,
            "success_count": 0
        }
        self.strategy_rules.append(rule)
        self.rule_usage_count[rule_id] = 0
        self.rule_success_rate[rule_id] = 1.0
        return rule_id

    def update_rule_performance(self, rule_id, success):
        if rule_id in self.rule_usage_count:
            self.rule_usage_count[rule_id] += 1
            if success:
                self.strategy_rules = [
                    r for r in self.strategy_rules
                    if r["id"] != rule_id
                ] + [{
                    **next(r for r in self.strategy_rules if r["id"] == rule_id),
                    "success_count": next(r for r in self.strategy_rules if r["id"] == rule_id) + 1
                }]
            total = self.rule_usage_count[rule_id]
            rule = next(r for r in self.strategy_rules if r["id"] == rule_id)
            self.rule_success_rate[rule_id] = rule["success_count"] / total

    def prune_low_performance_rules(self, min_success_rate=0.3, min_usage=10):
        pruned_ids = []
        for rule in list(self.strategy_rules):
            if (self.rule_usage_count.get(rule["id"], 0) >= min_usage and
                    self.rule_success_rate.get(rule["id"], 1.0) < min_success_rate):
                pruned_ids.append(rule["id"])
                self.strategy_rules.remove(rule)
        return pruned_ids


class SubgoalGenerator:
    def __init__(self):
        self.active_subgoals = []
        self.subgoal_history = deque(maxlen=100)
        self.subgoal_depth = 0
        self.max_depth = 5

    def generate_subgoal(self, main_goal, impasse_reason, context):
        if self.subgoal_depth >= self.max_depth:
            return None

        self.subgoal_depth += 1
        subgoal_id = f"subgoal_{len(self.subgoal_history) + 1}"
        subgoal = {
            "id": subgoal_id,
            "parent_goal": main_goal,
            "reason": impasse_reason,
            "context": context,
            "status": "active",
            "depth": self.subgoal_depth,
            "created_at": np.random.randint(1000000)
        }

        self.active_subgoals.append(subgoal)
        return subgoal

    def complete_subgoal(self, subgoal_id, result):
        for sg in self.active_subgoals:
            if sg["id"] == subgoal_id:
                sg["status"] = "completed"
                sg["result"] = result
                self.subgoal_history.append(sg)
                self.active_subgoals.remove(sg)
                self.subgoal_depth = max(0, self.subgoal_depth - 1)
                return sg
        return None

    def fail_subgoal(self, subgoal_id, reason):
        for sg in self.active_subgoals:
            if sg["id"] == subgoal_id:
                sg["status"] = "failed"
                sg["failure_reason"] = reason
                self.subgoal_history.append(sg)
                self.active_subgoals.remove(sg)
                self.subgoal_depth = max(0, self.subgoal_depth - 1)
                return sg
        return None

    def get_active_subgoals(self):
        return self.active_subgoals


class MetaLearner:
    def __init__(self):
        self.strategy_pool = [
            {"name": "trial_and_error", "type": "exploration", "suitable_for": ["novel", "high_uncertainty"]},
            {"name": "analogical_transfer", "type": "exploitation", "suitable_for": ["similar", "medium_uncertainty"]},
            {"name": "logical_deduction", "type": "reasoning", "suitable_for": ["structured", "low_uncertainty"]},
            {"name": "random_exploration", "type": "exploration", "suitable_for": ["unknown", "very_high_uncertainty"]}
        ]
        self.strategy_performance = {s["name"]: {"success": 0, "total": 0} for s in self.strategy_pool}
        self.current_strategy = "trial_and_error"

    def select_strategy(self, context_type, uncertainty):
        best_strategy = self.current_strategy
        best_score = -1.0

        for strategy in self.strategy_pool:
            suitability = 0.0
            if context_type in strategy["suitable_for"]:
                suitability += 0.5

            if strategy["type"] == "exploration" and uncertainty > 0.5:
                suitability += 0.3
            elif strategy["type"] == "exploitation" and uncertainty < 0.5:
                suitability += 0.3
            elif strategy["type"] == "reasoning" and uncertainty < 0.3:
                suitability += 0.3

            perf = self.strategy_performance[strategy["name"]]
            if perf["total"] > 0:
                suitability += 0.2 * (perf["success"] / perf["total"])

            if suitability > best_score:
                best_score = suitability
                best_strategy = strategy["name"]

        self.current_strategy = best_strategy
        return best_strategy

    def update_strategy_performance(self, strategy_name, success):
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name]["total"] += 1
            if success:
                self.strategy_performance[strategy_name]["success"] += 1

    def get_strategy_stats(self):
        stats = {}
        for name, perf in self.strategy_performance.items():
            if perf["total"] > 0:
                stats[name] = perf["success"] / perf["total"]
            else:
                stats[name] = 0.5
        return stats


class MetaCognitionLayer:
    def __init__(self):
        self.self_monitor = SelfMonitor()
        self.self_reflection = SelfReflection()
        self.subgoal_generator = SubgoalGenerator()
        self.meta_learner = MetaLearner()

        self.cognitive_metrics = {
            "free_energy": 0.0,
            "entropy": 0.0,
            "confidence": 1.0,
            "learning_progress": 0.0
        }
        self.resource_metrics = {
            "cuda_usage": 0.0,
            "latency": 0.0,
            "memory_usage": 0.0
        }
        self.environment_metrics = {
            "novelty": 0.0,
            "dist_shift": 0.0,
            "stability": 1.0
        }

        self.fe_history = deque(maxlen=100)
        self.novelty_history = deque(maxlen=100)
        self.resource_history = deque(maxlen=50)
        self.system_status = "healthy"

        self.impasse_count = 0
        self.stagnation_score = 0.0

    def monitor(self, fe, entropy_val, kl_shift, step_time):
        self.cognitive_metrics["free_energy"] = fe
        self.cognitive_metrics["entropy"] = entropy_val
        self.cognitive_metrics["confidence"] = max(0.0, min(1.0, 1.0 - fe / (FREE_ENERGY_THRESHOLD * 2)))

        self.environment_metrics["dist_shift"] = kl_shift
        self.environment_metrics["novelty"] = min(1.0, kl_shift / NOVELTY_THRESHOLD)

        self.resource_metrics["latency"] = step_time

        self.fe_history.append(fe)
        self.novelty_history.append(self.environment_metrics["novelty"])

        self._update_system_status()

    def _update_system_status(self):
        if self.cognitive_metrics["free_energy"] > FREE_ENERGY_THRESHOLD * 5:
            self.system_status = "critical"
        elif self.cognitive_metrics["free_energy"] > FREE_ENERGY_THRESHOLD * 2:
            self.system_status = "warning"
        else:
            self.system_status = "healthy"

    def check_cognitive_impasse(self, task_progress_ratio=0.0):
        confidence = self.cognitive_metrics["confidence"]
        fe = self.cognitive_metrics["free_energy"]
        return self.self_monitor.detect_cognitive_impasse(confidence, fe, task_progress_ratio)

    def handle_impasse(self, main_goal, context):
        is_impasse, impasse_record = self.check_cognitive_impasse()
        if not is_impasse:
            return None

        impasse_reason = f"Low confidence ({impasse_record['confidence']:.2f}) and high FE ({impasse_record['free_energy']:.2f})"
        subgoal = self.subgoal_generator.generate_subgoal(main_goal, impasse_reason, context)

        return subgoal

    def reflect_on_decision(self, decision, context, outcome, reward, confidence):
        self.self_reflection.record_decision(decision, context, outcome, reward, confidence)
        attribution = self.self_reflection.attribute_decision()

        if attribution and attribution["result"] == "success":
            self.self_reflection.consolidate_rule(context, decision)

        return attribution

    def select_learning_strategy(self, context_type, uncertainty):
        return self.meta_learner.select_strategy(context_type, uncertainty)

    def update_strategy_performance(self, strategy_name, success):
        self.meta_learner.update_strategy_performance(strategy_name, success)

    def resource_schedule(self):
        novelty = self.environment_metrics["novelty"]
        if novelty > NOVELTY_THRESHOLD:
            return {"perception": 0.6, "cognitive": 0.3, "evolve": 0.1}
        else:
            return {"perception": 0.2, "cognitive": 0.3, "evolve": 0.5}

    def need_evolve(self):
        avg_fe = sum(self.fe_history) / len(self.fe_history) if self.fe_history else 0.0
        return avg_fe > FREE_ENERGY_THRESHOLD and \
               self.environment_metrics["novelty"] < NOVELTY_THRESHOLD * 0.5

    def get_all_metrics(self):
        return {
            "cognitive": self.cognitive_metrics,
            "resource": self.resource_metrics,
            "environment": self.environment_metrics,
            "system_status": self.system_status,
            "active_subgoals": len(self.subgoal_generator.get_active_subgoals()),
            "strategy_stats": self.meta_learner.get_strategy_stats(),
            "confidence_trend": self.self_monitor.get_confidence_trend()
        }

    def get_trend_analysis(self):
        if len(self.fe_history) < 10:
            return {"fe_trend": 0.0, "novelty_trend": 0.0}

        recent_fe = list(self.fe_history)[-10:]
        recent_novelty = list(self.novelty_history)[-10:]

        fe_trend = (recent_fe[-1] - recent_fe[0]) / (recent_fe[0] + 1e-8)
        novelty_trend = (recent_novelty[-1] - recent_novelty[0]) / (recent_novelty[0] + 1e-8)

        return {"fe_trend": fe_trend, "novelty_trend": novelty_trend}

    def get_impasse_count(self):
        """获取认知僵局计数。

        优先使用 SelfMonitor 中已记录的 cognitive_impasses 数量，
        并同步更新至 impasse_count 属性。
        """
        try:
            count = len(self.self_monitor.cognitive_impasses)
            self.impasse_count = count
            return count
        except Exception:
            return self.impasse_count

    def get_stagnation_score(self):
        """获取停滞分数（0.0~1.0）。

        基于自由能趋势与置信度趋势综合估算；若无足够历史数据
        则返回 stagnation_score 属性值（默认 0.0）。
        """
        try:
            trend = self.get_trend_analysis()
            fe_trend = trend.get("fe_trend", 0.0)
            conf_trend = self.self_monitor.get_confidence_trend()
            # 自由能上升 + 置信度下降 => 停滞加重
            score = 0.5 * min(1.0, max(0.0, fe_trend)) + 0.5 * min(1.0, max(0.0, -conf_trend))
            score = max(0.0, min(1.0, score))
            self.stagnation_score = score
            return score
        except Exception:
            return self.stagnation_score
