import numpy as np
from collections import deque


class ThinkingTracer:
    def __init__(self):
        self.trace_stack = []
        self.trace_history = deque(maxlen=200)
        self.current_trace = None

    def start_trace(self, thinking_id, problem_id):
        self.current_trace = {
            "thinking_id": thinking_id,
            "problem_id": problem_id,
            "steps": [],
            "start_time": np.random.randint(1000000),
            "end_time": None,
            "status": "in_progress"
        }
        self.trace_stack.append(self.current_trace)

    def record_step(self, step_type, details, confidence=0.0):
        if self.current_trace:
            self.current_trace["steps"].append({
                "step_type": step_type,
                "details": details,
                "confidence": confidence,
                "timestamp": np.random.randint(1000000)
            })

    def end_trace(self, status, confidence=0.0):
        if self.current_trace:
            self.current_trace["end_time"] = np.random.randint(1000000)
            self.current_trace["status"] = status
            self.current_trace["final_confidence"] = confidence
            self.trace_history.append(self.current_trace)
            self.current_trace = None

    def detect_loop(self, max_steps=50):
        if self.current_trace and len(self.current_trace["steps"]) > max_steps:
            recent_steps = [s["step_type"] for s in self.current_trace["steps"][-10:]]
            if len(set(recent_steps)) < 3:
                return True, {"warning": "Potential thinking loop detected", "step_count": len(self.current_trace["steps"])}
        return False, None

    def get_recent_traces(self, limit=10):
        return list(self.trace_history)[-limit:]

    def get_trace_stats(self):
        stats = {
            "total_traces": len(self.trace_history),
            "completed_traces": len([t for t in self.trace_history if t["status"] == "completed"]),
            "avg_steps": 0.0,
            "avg_confidence": 0.0
        }
        
        if self.trace_history:
            completed = [t for t in self.trace_history if t["status"] == "completed"]
            if completed:
                stats["avg_steps"] = float(np.mean([len(t["steps"]) for t in completed]))
                stats["avg_confidence"] = float(np.mean([t.get("final_confidence", 0.0) for t in completed]))
        
        return stats


class ExecutionTracker:
    def __init__(self):
        self.active_executions = {}
        self.execution_history = deque(maxlen=200)

    def start_execution(self, execution_id, action_plan):
        self.active_executions[execution_id] = {
            "execution_id": execution_id,
            "action_plan": action_plan,
            "start_time": np.random.randint(1000000),
            "steps_completed": 0,
            "total_steps": len(action_plan.get("execution_steps", [])),
            "status": "running",
            "errors": []
        }

    def update_execution(self, execution_id, step_completed, error=None):
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["steps_completed"] = step_completed
            if error:
                self.active_executions[execution_id]["errors"].append(error)

    def complete_execution(self, execution_id, success, outcome):
        if execution_id in self.active_executions:
            exec_record = self.active_executions.pop(execution_id)
            exec_record["end_time"] = np.random.randint(1000000)
            exec_record["status"] = "completed" if success else "failed"
            exec_record["outcome"] = outcome
            self.execution_history.append(exec_record)
            return exec_record
        return None

    def get_active_executions(self):
        return list(self.active_executions.values())

    def get_execution_stats(self):
        stats = {
            "active_count": len(self.active_executions),
            "total_executions": len(self.execution_history),
            "success_rate": 0.0,
            "avg_steps": 0.0
        }
        
        if self.execution_history:
            success_count = len([e for e in self.execution_history if e["status"] == "completed"])
            stats["success_rate"] = success_count / len(self.execution_history)
            stats["avg_steps"] = float(np.mean([e["steps_completed"] for e in self.execution_history]))
        
        return stats


class ResourceMonitor:
    def __init__(self):
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.peak_load = 0.0

    def record_resource_usage(self, cpu, memory, latency):
        self.cpu_history.append(cpu)
        self.memory_history.append(memory)
        self.latency_history.append(latency)
        self.peak_load = max(self.peak_load, cpu)

    def check_resource_health(self):
        avg_cpu = float(np.mean(self.cpu_history)) if self.cpu_history else 0.0
        avg_memory = float(np.mean(self.memory_history)) if self.memory_history else 0.0
        avg_latency = float(np.mean(self.latency_history)) if self.latency_history else 0.0
        
        return {
            "cpu_healthy": avg_cpu < 0.8,
            "memory_healthy": avg_memory < 0.85,
            "latency_healthy": avg_latency < 1000,
            "overall_healthy": avg_cpu < 0.8 and avg_memory < 0.85 and avg_latency < 1000,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
            "avg_latency": avg_latency,
            "peak_load": self.peak_load
        }

    def get_resource_stats(self):
        return {
            "avg_cpu": float(np.mean(self.cpu_history)) if self.cpu_history else 0.0,
            "avg_memory": float(np.mean(self.memory_history)) if self.memory_history else 0.0,
            "avg_latency": float(np.mean(self.latency_history)) if self.latency_history else 0.0,
            "peak_load": self.peak_load,
            "sample_count": len(self.cpu_history)
        }


class CognitiveMonitor:
    def __init__(self):
        self.thinking_tracer = ThinkingTracer()
        self.execution_tracker = ExecutionTracker()
        self.resource_monitor = ResourceMonitor()
        self.alerts = deque(maxlen=50)

    def monitor_thinking(self, thinking_id, problem_id, steps=None, status=None, confidence=0.0):
        if steps is None:
            self.thinking_tracer.start_trace(thinking_id, problem_id)
        elif status is None:
            for step in steps:
                self.thinking_tracer.record_step(step["step_type"], step["details"], step.get("confidence", 0.0))
        else:
            self.thinking_tracer.end_trace(status, confidence)
            is_loop, loop_info = self.thinking_tracer.detect_loop()
            if is_loop:
                self._create_alert("thinking_loop", loop_info)

    def monitor_execution(self, execution_id, action_plan=None, step_completed=None, success=None, outcome=None):
        if action_plan is not None:
            self.execution_tracker.start_execution(execution_id, action_plan)
        elif step_completed is not None:
            self.execution_tracker.update_execution(execution_id, step_completed)
        elif success is not None:
            self.execution_tracker.complete_execution(execution_id, success, outcome)

    def monitor_resources(self, cpu, memory, latency):
        self.resource_monitor.record_resource_usage(cpu, memory, latency)
        health = self.resource_monitor.check_resource_health()
        if not health["overall_healthy"]:
            self._create_alert("resource_warning", health)

    def _create_alert(self, alert_type, details):
        alert = {
            "alert_type": alert_type,
            "details": details,
            "timestamp": np.random.randint(1000000),
            "severity": "warning" if alert_type == "resource_warning" else "critical"
        }
        self.alerts.append(alert)

    def get_alerts(self, limit=10):
        return list(self.alerts)[-limit:]

    def get_all_stats(self):
        return {
            "thinking": self.thinking_tracer.get_trace_stats(),
            "execution": self.execution_tracker.get_execution_stats(),
            "resources": self.resource_monitor.get_resource_stats(),
            "active_alerts": len(self.alerts)
        }