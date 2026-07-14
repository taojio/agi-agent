import numpy as np
from collections import deque
from ..utils.metrics import calc_free_energy, calc_entropy, calc_confidence
from ..utils.logger import setup_logger


class PerformanceEvaluator:
    def __init__(self):
        self.logger = setup_logger("performance_evaluator")
        self.evaluation_history = deque(maxlen=1000)
        self.metric_history = deque(maxlen=500)
        
    def evaluate_step(self, step: int, metrics: dict):
        evaluation = {
            "step": step,
            "free_energy": metrics.get("free_energy", 0.0),
            "confidence": metrics.get("confidence", 0.0),
            "novelty": metrics.get("novelty", 0.0),
            "latency": metrics.get("latency", 0.0),
            "timestamp": step
        }
        
        self.evaluation_history.append(evaluation)
        self.metric_history.append(metrics)
        
        return evaluation
    
    def calculate_performance_score(self, window_size: int = 50):
        if len(self.evaluation_history) < window_size:
            return {
                "total_score": 0.0,
                "components": {
                    "free_energy_score": 0.0,
                    "confidence_score": 0.0,
                    "latency_score": 0.0
                },
                "averages": {
                    "free_energy": 0.0,
                    "confidence": 0.0,
                    "latency": 0.0
                }
            }
        
        recent = list(self.evaluation_history)[-window_size:]
        avg_fe = np.mean([e["free_energy"] for e in recent])
        avg_conf = np.mean([e["confidence"] for e in recent])
        avg_latency = np.mean([e["latency"] for e in recent])
        
        fe_score = max(0.0, 1.0 - avg_fe / 0.1)
        conf_score = avg_conf
        latency_score = max(0.0, 1.0 - avg_latency / 1000.0)
        
        total_score = 0.4 * fe_score + 0.4 * conf_score + 0.2 * latency_score
        
        return {
            "total_score": total_score,
            "components": {
                "free_energy_score": fe_score,
                "confidence_score": conf_score,
                "latency_score": latency_score
            },
            "averages": {
                "free_energy": avg_fe,
                "confidence": avg_conf,
                "latency": avg_latency
            }
        }
    
    def analyze_convergence(self):
        history_list = list(self.evaluation_history)
        if len(history_list) < 20:
            return {"converged": False, "rate": 0.0}
        
        recent_fe = [e["free_energy"] for e in history_list[-20:]]
        earlier_fe = [e["free_energy"] for e in history_list[-40:-20]] if len(history_list) >= 40 else recent_fe
        
        recent_avg = np.mean(recent_fe)
        earlier_avg = np.mean(earlier_fe)
        
        if earlier_avg < 1e-8:
            return {"converged": recent_avg < 0.01, "rate": 0.0}
        
        improvement_rate = (earlier_avg - recent_avg) / earlier_avg
        
        return {
            "converged": recent_avg < 0.01 and improvement_rate < 0.01,
            "rate": improvement_rate,
            "recent_avg_fe": recent_avg,
            "earlier_avg_fe": earlier_avg
        }
    
    def get_evaluation_report(self):
        performance = self.calculate_performance_score()
        convergence = self.analyze_convergence()
        metric_list = list(self.metric_history)
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "performance_score": performance,
            "convergence": convergence,
            "recent_metrics": metric_list[-10:] if metric_list else []
        }
    
    def log_evaluation(self, step: int, metrics: dict):
        report = self.evaluate_step(step, metrics)
        if step % 100 == 0:
            self.logger.info(f"[EVAL] Step {step} - FE: {report['free_energy']:.4f}, Confidence: {report['confidence']:.4f}")