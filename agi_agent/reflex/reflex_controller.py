import numpy as np
from collections import deque
from .spiking_core import SpikingCore
from .pattern_matcher import PatternMatcher
from .rule_engine import RuleEngine, ProductionRule
from .instinct_actions import InstinctActions, InstinctType


class ReflexController:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        
        self.spiking_core = SpikingCore(input_dim=feature_dim, hidden_dim=32, output_dim=feature_dim)
        self.pattern_matcher = PatternMatcher(feature_dim=feature_dim)
        self.rule_engine = RuleEngine(feature_dim=feature_dim)
        self.instinct_actions = InstinctActions()
        
        self.response_history = deque(maxlen=200)
        self.confidence_threshold = 0.85
        self.fallback_to_deliberate = False
        
        self._is_active = True
        self._response_latency = 0.0
        
        self._init_default_rules()

    def _init_default_rules(self):
        default_rules = [
            ProductionRule(
                condition=lambda d: d.get("risk_level", 0) > 0.7,
                action=lambda d: {"type": "emergency", "action": "stop", "reason": "high risk"},
                priority=10.0,
                confidence=0.95,
                rule_id="emergency_stop"
            ),
            ProductionRule(
                condition=lambda d: d.get("novelty", 0) > 0.6 and d.get("confidence", 0) < 0.5,
                action=lambda d: {"type": "delegate", "action": "deliberate", "reason": "unknown pattern"},
                priority=8.0,
                confidence=0.9,
                rule_id="delegate_unknown"
            ),
            ProductionRule(
                condition=lambda d: d.get("confidence", 0) > 0.9 and d.get("pattern_match", False),
                action=lambda d: {"type": "execute", "action": "direct", "pattern_id": d.get("pattern_id")},
                priority=5.0,
                confidence=0.95,
                rule_id="direct_execute"
            ),
            ProductionRule(
                condition=lambda d: d.get("error_detected", False),
                action=lambda d: {"type": "correct", "action": "retry", "attempts": 0},
                priority=7.0,
                confidence=0.85,
                rule_id="auto_correct"
            ),
            ProductionRule(
                condition=lambda d: d.get("resource_low", False),
                action=lambda d: {"type": "conserve", "action": "reduce_load", "reason": "resource constraint"},
                priority=6.0,
                confidence=0.9,
                rule_id="resource_conservation"
            ),
            
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "查看任务状态" in d["input_text"],
                action=lambda d: {"type": "task_status", "action": "query_task_status"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_check_task_status"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "列出文件" in d["input_text"],
                action=lambda d: {"type": "file_list", "action": "list_files"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_list_files"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "执行数据统计" in d["input_text"],
                action=lambda d: {"type": "data_stats", "action": "execute_statistics"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_data_statistics"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "保存状态" in d["input_text"],
                action=lambda d: {"type": "save", "action": "save_agent_state"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_save_state"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "加载状态" in d["input_text"],
                action=lambda d: {"type": "load", "action": "load_agent_state"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_load_state"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "查看版本" in d["input_text"],
                action=lambda d: {"type": "version", "action": "query_version_info"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_check_version"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "帮助" in d["input_text"],
                action=lambda d: {"type": "help", "action": "show_help"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_show_help"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "状态报告" in d["input_text"],
                action=lambda d: {"type": "report", "action": "generate_status_report"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_generate_report"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "重置" in d["input_text"],
                action=lambda d: {"type": "reset", "action": "reset_agent"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_reset"
            ),
            ProductionRule(
                condition=lambda d: isinstance(d.get("input_text"), str) and "退出" in d["input_text"],
                action=lambda d: {"type": "exit", "action": "shutdown"},
                priority=5.0,
                confidence=0.95,
                rule_id="trigger_exit"
            ),
            
            ProductionRule(
                condition=lambda d: d.get("fallback_type") == "no_match",
                action=lambda d: {"type": "fallback", "action": "rule_not_matched", 
                    "message": "未识别指令，请使用明确指令，如：查看任务状态、列出文件、执行数据统计、保存状态、加载状态"},
                priority=2.0,
                confidence=1.0,
                rule_id="fallback_no_match"
            ),
            ProductionRule(
                condition=lambda d: d.get("fallback_type") == "timeout",
                action=lambda d: {"type": "fallback", "action": "execution_timeout",
                    "message": "任务执行超时，请检查参数后重试"},
                priority=2.0,
                confidence=1.0,
                rule_id="fallback_timeout"
            ),
            ProductionRule(
                condition=lambda d: d.get("fallback_type") == "exception",
                action=lambda d: {"type": "fallback", "action": "system_exception",
                    "message": "当前无法处理该请求，已记录异常"},
                priority=2.0,
                confidence=1.0,
                rule_id="fallback_exception"
            )
        ]
        self.rule_engine.add_rules(default_rules)

    def process(self, input_vector, context=None):
        if not self._is_active:
            return {"status": "inactive", "latency": 0.0}
        
        input_vec = np.array(input_vector).flatten()
        
        snn_output = self.spiking_core.forward(input_vec)
        anomaly_score, _ = self.spiking_core.detect_anomaly(input_vec)
        
        pattern_result = self.pattern_matcher.match(input_vec)
        
        instinct_result, instinct_priority = self.instinct_actions.trigger_highest_priority({
            "risk_level": context.get("risk_level", 0) if context else 0,
            "threat_detected": context.get("threat_detected", False) if context else False,
            "goal_detected": context.get("goal_detected", False) if context else False,
            "novelty": pattern_result["anomaly_score"],
            "unknown_detected": not pattern_result["is_known"],
            "fatigue": context.get("fatigue", 0) if context else 0
        })
        
        rule_input = {
            "risk_level": context.get("risk_level", 0) if context else 0,
            "novelty": pattern_result["anomaly_score"],
            "confidence": pattern_result["confidence"],
            "pattern_match": pattern_result["is_known"],
            "pattern_id": pattern_result["pattern_id"],
            "error_detected": context.get("error_detected", False) if context else False,
            "resource_low": context.get("resource_low", False) if context else False
        }
        
        rule_id, rule_result, rule_score = self.rule_engine.execute_best_rule(rule_input)
        
        confidence = pattern_result["confidence"]
        should_delegate = False
        
        if confidence < self.confidence_threshold or pattern_result["is_anomalous"]:
            should_delegate = True
        
        if instinct_result and instinct_priority > 5.0:
            final_response = {
                "type": "instinct",
                "instinct_type": instinct_result.get("action", "unknown"),
                "priority": instinct_priority,
                "confidence": confidence,
                "pattern_id": pattern_result["pattern_id"],
                "delegate": False
            }
        elif rule_result and rule_score > 0.5:
            final_response = {
                "type": "rule",
                "rule_id": rule_id,
                "action": rule_result,
                "priority": rule_score,
                "confidence": confidence,
                "pattern_id": pattern_result["pattern_id"],
                "delegate": should_delegate
            }
        else:
            final_response = {
                "type": "direct",
                "action": snn_output.tolist(),
                "priority": confidence,
                "confidence": confidence,
                "pattern_id": pattern_result["pattern_id"],
                "delegate": should_delegate
            }
        
        self.response_history.append({
            "type": final_response["type"],
            "confidence": confidence,
            "delegate": should_delegate,
            "pattern_id": pattern_result["pattern_id"],
            "anomaly_score": anomaly_score,
            "latency": self._response_latency
        })
        
        return final_response

    def learn(self, input_vector, outcome, success):
        input_vec = np.array(input_vector).flatten()
        
        self.spiking_core.learn()
        
        if success:
            self.pattern_matcher.add_pattern(
                pattern_id=f"auto_{len(self.pattern_matcher.patterns)}",
                feature_vector=input_vec,
                prediction=outcome,
                confidence=0.8
            )
        else:
            if input_vec.shape[0] == self.feature_dim:
                anomaly_score, _ = self.spiking_core.detect_anomaly(input_vec)
                if anomaly_score > 1.5:
                    self.pattern_matcher.add_pattern(
                        pattern_id=f"anomaly_{len(self.pattern_matcher.patterns)}",
                        feature_vector=input_vec,
                        prediction=None,
                        confidence=0.1
                    )

    def update_rule_performance(self, rule_id, success):
        self.rule_engine.update_rule_performance(rule_id, success)

    def update_pattern_performance(self, pattern_id, success):
        self.pattern_matcher.update_pattern_confidence(pattern_id, success)

    def enter_sleep_mode(self):
        self.spiking_core.enter_sleep_mode()
        self._is_active = False

    def wake_up(self):
        self.spiking_core.wake_up()
        self._is_active = True

    def is_active(self):
        return self._is_active

    def get_activity_summary(self):
        recent_responses = list(self.response_history)[-10:]
        recent_confidence = [r["confidence"] for r in recent_responses]
        
        return {
            "spiking_activity": self.spiking_core.get_activity_level(),
            "pattern_stats": self.pattern_matcher.get_pattern_stats(),
            "rule_stats": self.rule_engine.get_rule_stats(),
            "instinct_stats": self.instinct_actions.get_instinct_stats(),
            "response_history_count": len(self.response_history),
            "is_active": self.is_active(),
            "confidence_threshold": self.confidence_threshold,
            "recent_avg_confidence": float(np.mean(recent_confidence)) if recent_confidence else 0.0,
            "delegate_rate": sum(1 for r in recent_responses if r["delegate"]) / len(recent_responses) if recent_responses else 0.0,
            "response_latency": self._response_latency,
            "total_rules": len(self.rule_engine.rules),
            "total_patterns": len(self.pattern_matcher.patterns),
            "sleep_mode": not self._is_active
        }

    def get_recent_responses(self, limit=10):
        return list(self.response_history)[-limit:]

    def resize(self, new_dim):
        self.feature_dim = new_dim
        self.spiking_core = SpikingCore(input_dim=new_dim, hidden_dim=32, output_dim=new_dim)
        self.pattern_matcher = PatternMatcher(feature_dim=new_dim)
        self.rule_engine.resize(new_dim)
        self._init_default_rules()