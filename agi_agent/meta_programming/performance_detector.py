import time
import os
import threading
import gc
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


class BottleneckType(Enum):
    CPU_BOUND = "cpu_bound"
    MEMORY_LEAK = "memory_leak"
    MEMORY_HIGH = "memory_high"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    THREAD_CONCURRENCY = "thread_concurrency"
    GARBAGE_COLLECTION = "garbage_collection"
    LOCK_CONTENTION = "lock_contention"
    RECURSION_DEPTH = "recursion_depth"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MetricType(Enum):
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    MEMORY_ALLOCATED = "memory_allocated"
    MEMORY_LEAK_RATE = "memory_leak_rate"
    THREAD_COUNT = "thread_count"
    GC_COLLECTIONS = "gc_collections"
    GC_TIME = "gc_time"
    LOCK_WAIT_TIME = "lock_wait_time"
    RECURSION_DEPTH = "recursion_depth"
    EXECUTION_TIME = "execution_time"
    CALL_FREQUENCY = "call_frequency"
    UNKNOWN = "unknown"


@dataclass
class PerformanceSample:
    timestamp: float
    metrics: Dict[str, float]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckDetection:
    detection_id: str
    bottleneck_type: BottleneckType
    severity: SeverityLevel
    metric_name: str
    current_value: float
    threshold_value: float
    trend: str
    description: str
    code_location: str = ""
    timestamp: float = 0.0
    duration: float = 0.0


@dataclass
class OptimizationSuggestion:
    suggestion_id: str
    bottleneck_type: BottleneckType
    priority: int
    category: str
    description: str
    code_change: str = ""
    expected_improvement: float = 0.0
    estimated_effort: str = "low"
    related_metrics: List[str] = field(default_factory=list)


@dataclass
class PerformanceThreshold:
    metric_type: MetricType
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    description: str = ""


class PerformanceCollector:
    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.samples: deque = deque(maxlen=1000)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._custom_metrics: Dict[str, float] = {}
        self._gc_stats: Dict[str, int] = {"collections": 0, "time_ms": 0}
        self._last_gc_count = 0
        self._last_gc_time = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _collection_loop(self):
        while self._running:
            sample = self._collect_sample()
            with self._lock:
                self.samples.append(sample)
            time.sleep(self.sample_interval_ms / 1000.0)

    def _collect_sample(self) -> PerformanceSample:
        metrics = {}

        try:
            import psutil
            process = psutil.Process(os.getpid())
            metrics["cpu_usage"] = process.cpu_percent(interval=None)
            mem_info = process.memory_info()
            metrics["memory_usage"] = mem_info.rss / (1024 * 1024)
            metrics["memory_vms"] = mem_info.vms / (1024 * 1024)
        except ImportError:
            metrics["cpu_usage"] = 0.0
            metrics["memory_usage"] = 0.0

        try:
            gc.collect()
            gc_stats = gc.get_stats()
            if gc_stats:
                collections = sum(s["collections"] for s in gc_stats)
                time_ms = sum(s["time"] for s in gc_stats) * 1000
                metrics["gc_collections"] = collections - self._last_gc_count
                metrics["gc_time_ms"] = time_ms - self._last_gc_time
                self._last_gc_count = collections
                self._last_gc_time = time_ms
        except Exception:
            metrics["gc_collections"] = 0
            metrics["gc_time_ms"] = 0.0

        metrics["thread_count"] = threading.active_count()

        with self._lock:
            for name, value in self._custom_metrics.items():
                metrics[name] = value

        return PerformanceSample(
            timestamp=time.time(),
            metrics=metrics,
            context={"pid": os.getpid()}
        )

    def add_custom_metric(self, name: str, value: float):
        with self._lock:
            self._custom_metrics[name] = value

    def get_recent_samples(self, count: int = 100) -> List[PerformanceSample]:
        with self._lock:
            return list(self.samples)[-count:]

    def get_latest_sample(self) -> Optional[PerformanceSample]:
        with self._lock:
            return self.samples[-1] if self.samples else None

    def get_metric_history(self, metric_name: str, count: int = 100) -> List[Tuple[float, float]]:
        with self._lock:
            samples = list(self.samples)[-count:]
            return [(s.timestamp, s.metrics.get(metric_name, 0.0)) for s in samples]


class BottleneckAnalyzer:
    def __init__(self):
        self.thresholds: Dict[MetricType, PerformanceThreshold] = self._init_default_thresholds()
        self.detections: List[BottleneckDetection] = []

    def _init_default_thresholds(self) -> Dict[MetricType, PerformanceThreshold]:
        return {
            MetricType.CPU_USAGE: PerformanceThreshold(
                metric_type=MetricType.CPU_USAGE,
                warning_high=70.0,
                critical_high=90.0,
                description="CPU usage percentage"
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_high=500.0,
                critical_high=1000.0,
                description="Memory usage in MB"
            ),
            MetricType.MEMORY_LEAK_RATE: PerformanceThreshold(
                metric_type=MetricType.MEMORY_LEAK_RATE,
                warning_high=10.0,
                critical_high=50.0,
                description="Memory leak rate in MB per minute"
            ),
            MetricType.THREAD_COUNT: PerformanceThreshold(
                metric_type=MetricType.THREAD_COUNT,
                warning_high=50,
                critical_high=100,
                description="Active thread count"
            ),
            MetricType.GC_COLLECTIONS: PerformanceThreshold(
                metric_type=MetricType.GC_COLLECTIONS,
                warning_high=10,
                critical_high=50,
                description="GC collections per sample interval"
            ),
            MetricType.GC_TIME: PerformanceThreshold(
                metric_type=MetricType.GC_TIME,
                warning_high=10.0,
                critical_high=50.0,
                description="GC time in milliseconds"
            ),
            MetricType.LOCK_WAIT_TIME: PerformanceThreshold(
                metric_type=MetricType.LOCK_WAIT_TIME,
                warning_high=5.0,
                critical_high=20.0,
                description="Lock wait time in milliseconds"
            ),
            MetricType.RECURSION_DEPTH: PerformanceThreshold(
                metric_type=MetricType.RECURSION_DEPTH,
                warning_high=100,
                critical_high=1000,
                description="Recursion depth"
            ),
            MetricType.EXECUTION_TIME: PerformanceThreshold(
                metric_type=MetricType.EXECUTION_TIME,
                warning_high=1.0,
                critical_high=5.0,
                description="Execution time in seconds"
            ),
        }

    def analyze_samples(self, samples: List[PerformanceSample]) -> List[BottleneckDetection]:
        if not samples:
            return []

        detections = []

        for metric_name, values in self._extract_metrics(samples).items():
            metric_type = self._metric_name_to_type(metric_name)
            threshold = self.thresholds.get(metric_type)
            if not threshold:
                continue

            current_value = values[-1] if values else 0.0
            avg_value = sum(values) / len(values) if values else 0.0
            trend = self._calculate_trend(values)

            detection = self._check_threshold(
                metric_type,
                threshold,
                current_value,
                avg_value,
                trend
            )
            if detection:
                detections.append(detection)

        memory_leak_detection = self._detect_memory_leak(samples)
        if memory_leak_detection:
            detections.append(memory_leak_detection)

        gc_bottleneck = self._detect_gc_bottleneck(samples)
        if gc_bottleneck:
            detections.append(gc_bottleneck)

        self.detections.extend(detections)
        return detections

    def _extract_metrics(self, samples: List[PerformanceSample]) -> Dict[str, List[float]]:
        metrics: Dict[str, List[float]] = {}
        for sample in samples:
            for name, value in sample.metrics.items():
                if name not in metrics:
                    metrics[name] = []
                metrics[name].append(value)
        return metrics

    def _metric_name_to_type(self, name: str) -> MetricType:
        mapping = {
            "cpu_usage": MetricType.CPU_USAGE,
            "memory_usage": MetricType.MEMORY_USAGE,
            "memory_allocated": MetricType.MEMORY_ALLOCATED,
            "thread_count": MetricType.THREAD_COUNT,
            "gc_collections": MetricType.GC_COLLECTIONS,
            "gc_time_ms": MetricType.GC_TIME,
            "lock_wait_time": MetricType.LOCK_WAIT_TIME,
            "recursion_depth": MetricType.RECURSION_DEPTH,
            "execution_time": MetricType.EXECUTION_TIME,
        }
        return mapping.get(name, MetricType.UNKNOWN)

    def _calculate_trend(self, values: List[float]) -> str:
        if len(values) < 10:
            return "stable"

        recent = values[-5:]
        earlier = values[:5]
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        if earlier_avg == 0:
            return "increasing" if recent_avg > 0 else "stable"

        ratio = recent_avg / earlier_avg
        if ratio > 1.3:
            return "increasing"
        elif ratio < 0.7:
            return "decreasing"
        else:
            return "stable"

    def _check_threshold(self, metric_type: MetricType, threshold: PerformanceThreshold,
                         current: float, avg: float, trend: str) -> Optional[BottleneckDetection]:
        bottleneck_map = {
            MetricType.CPU_USAGE: BottleneckType.CPU_BOUND,
            MetricType.MEMORY_USAGE: BottleneckType.MEMORY_HIGH,
            MetricType.THREAD_COUNT: BottleneckType.THREAD_CONCURRENCY,
            MetricType.GC_COLLECTIONS: BottleneckType.GARBAGE_COLLECTION,
            MetricType.GC_TIME: BottleneckType.GARBAGE_COLLECTION,
            MetricType.LOCK_WAIT_TIME: BottleneckType.LOCK_CONTENTION,
            MetricType.RECURSION_DEPTH: BottleneckType.RECURSION_DEPTH,
            MetricType.EXECUTION_TIME: BottleneckType.CPU_BOUND,
        }

        bottleneck_type = bottleneck_map.get(metric_type, BottleneckType.UNKNOWN)

        if threshold.critical_high and current >= threshold.critical_high:
            return BottleneckDetection(
                detection_id=f"det_{int(time.time() * 1000)}_{metric_type.value}",
                bottleneck_type=bottleneck_type,
                severity=SeverityLevel.CRITICAL,
                metric_name=metric_type.value,
                current_value=current,
                threshold_value=threshold.critical_high,
                trend=trend,
                description=f"{threshold.description} exceeds critical threshold"
            )

        if threshold.warning_high and current >= threshold.warning_high:
            return BottleneckDetection(
                detection_id=f"det_{int(time.time() * 1000)}_{metric_type.value}",
                bottleneck_type=bottleneck_type,
                severity=SeverityLevel.HIGH if trend == "increasing" else SeverityLevel.MEDIUM,
                metric_name=metric_type.value,
                current_value=current,
                threshold_value=threshold.warning_high,
                trend=trend,
                description=f"{threshold.description} exceeds warning threshold"
            )

        if threshold.critical_low and current <= threshold.critical_low:
            return BottleneckDetection(
                detection_id=f"det_{int(time.time() * 1000)}_{metric_type.value}",
                bottleneck_type=bottleneck_type,
                severity=SeverityLevel.CRITICAL,
                metric_name=metric_type.value,
                current_value=current,
                threshold_value=threshold.critical_low,
                trend=trend,
                description=f"{threshold.description} below critical threshold"
            )

        if threshold.warning_low and current <= threshold.warning_low:
            return BottleneckDetection(
                detection_id=f"det_{int(time.time() * 1000)}_{metric_type.value}",
                bottleneck_type=bottleneck_type,
                severity=SeverityLevel.MEDIUM,
                metric_name=metric_type.value,
                current_value=current,
                threshold_value=threshold.warning_low,
                trend=trend,
                description=f"{threshold.description} below warning threshold"
            )

        return None

    def _detect_memory_leak(self, samples: List[PerformanceSample]) -> Optional[BottleneckDetection]:
        memory_values = []
        timestamps = []
        for sample in samples:
            if "memory_usage" in sample.metrics:
                memory_values.append(sample.metrics["memory_usage"])
                timestamps.append(sample.timestamp)

        if len(memory_values) < 20:
            return None

        initial = memory_values[:5]
        recent = memory_values[-5:]
        initial_avg = sum(initial) / len(initial)
        recent_avg = sum(recent) / len(recent)

        time_diff = timestamps[-1] - timestamps[0]
        if time_diff < 60:
            return None

        leak_rate_mb_per_min = ((recent_avg - initial_avg) / time_diff) * 60

        if leak_rate_mb_per_min > 50.0:
            return BottleneckDetection(
                detection_id=f"det_{int(time.time() * 1000)}_memory_leak",
                bottleneck_type=BottleneckType.MEMORY_LEAK,
                severity=SeverityLevel.CRITICAL,
                metric_name="memory_leak_rate",
                current_value=leak_rate_mb_per_min,
                threshold_value=50.0,
                trend="increasing",
                description=f"Memory leak detected at {leak_rate_mb_per_min:.1f} MB/min"
            )
        elif leak_rate_mb_per_min > 10.0:
            return BottleneckDetection(
                detection_id=f"det_{int(time.time() * 1000)}_memory_leak",
                bottleneck_type=BottleneckType.MEMORY_LEAK,
                severity=SeverityLevel.HIGH,
                metric_name="memory_leak_rate",
                current_value=leak_rate_mb_per_min,
                threshold_value=10.0,
                trend="increasing",
                description=f"Memory leak suspected at {leak_rate_mb_per_min:.1f} MB/min"
            )

        return None

    def _detect_gc_bottleneck(self, samples: List[PerformanceSample]) -> Optional[BottleneckDetection]:
        gc_times = []
        gc_counts = []
        for sample in samples:
            if "gc_time_ms" in sample.metrics:
                gc_times.append(sample.metrics["gc_time_ms"])
            if "gc_collections" in sample.metrics:
                gc_counts.append(sample.metrics["gc_collections"])

        if gc_times:
            avg_gc_time = sum(gc_times) / len(gc_times)
            if avg_gc_time > 50.0:
                return BottleneckDetection(
                    detection_id=f"det_{int(time.time() * 1000)}_gc_bottleneck",
                    bottleneck_type=BottleneckType.GARBAGE_COLLECTION,
                    severity=SeverityLevel.CRITICAL,
                    metric_name="gc_time_ms",
                    current_value=avg_gc_time,
                    threshold_value=50.0,
                    trend="stable",
                    description=f"GC time bottleneck: {avg_gc_time:.1f}ms average"
                )

        if gc_counts:
            avg_gc_count = sum(gc_counts) / len(gc_counts)
            if avg_gc_count > 50:
                return BottleneckDetection(
                    detection_id=f"det_{int(time.time() * 1000)}_gc_frequent",
                    bottleneck_type=BottleneckType.GARBAGE_COLLECTION,
                    severity=SeverityLevel.HIGH,
                    metric_name="gc_collections",
                    current_value=avg_gc_count,
                    threshold_value=50.0,
                    trend="stable",
                    description=f"Excessive GC collections: {avg_gc_count:.1f} per interval"
                )

        return None

    def get_detections(self, severity: SeverityLevel = None) -> List[BottleneckDetection]:
        if severity:
            return [d for d in self.detections if d.severity.value >= severity.value]
        return self.detections

    def get_recent_detections(self, count: int = 20) -> List[BottleneckDetection]:
        return self.detections[-count:]


class OptimizationSuggestionGenerator:
    def __init__(self):
        self.suggestion_templates = {
            BottleneckType.CPU_BOUND: [
                {
                    "priority": 1,
                    "category": "algorithm_optimization",
                    "description": "Consider optimizing algorithms with high computational complexity. Look for O(n^2) operations that could be reduced to O(n log n).",
                    "expected_improvement": 0.3,
                    "estimated_effort": "high"
                },
                {
                    "priority": 2,
                    "category": "caching",
                    "description": "Implement caching for frequently computed values or memoization for recursive functions.",
                    "expected_improvement": 0.25,
                    "estimated_effort": "medium"
                },
                {
                    "priority": 3,
                    "category": "vectorization",
                    "description": "Use vectorized operations (NumPy) instead of Python loops for numerical computations.",
                    "expected_improvement": 0.4,
                    "estimated_effort": "medium"
                },
            ],
            BottleneckType.MEMORY_LEAK: [
                {
                    "priority": 1,
                    "category": "resource_cleanup",
                    "description": "Ensure proper cleanup of resources. Check for unused references, event handlers, and cache entries that are not being released.",
                    "expected_improvement": 0.4,
                    "estimated_effort": "high"
                },
                {
                    "priority": 2,
                    "category": "weak_references",
                    "description": "Replace strong references with weak references where appropriate to allow garbage collection.",
                    "expected_improvement": 0.2,
                    "estimated_effort": "medium"
                },
                {
                    "priority": 3,
                    "category": "batch_processing",
                    "description": "Use batch processing instead of incremental processing to limit memory accumulation.",
                    "expected_improvement": 0.3,
                    "estimated_effort": "medium"
                },
            ],
            BottleneckType.MEMORY_HIGH: [
                {
                    "priority": 1,
                    "category": "data_structure",
                    "description": "Optimize data structures to reduce memory footprint. Consider using generators instead of lists for large data sets.",
                    "expected_improvement": 0.35,
                    "estimated_effort": "medium"
                },
                {
                    "priority": 2,
                    "category": "lazy_loading",
                    "description": "Implement lazy loading or on-demand processing to reduce peak memory usage.",
                    "expected_improvement": 0.25,
                    "estimated_effort": "medium"
                },
                {
                    "priority": 3,
                    "category": "memory_profiling",
                    "description": "Run memory profiling to identify specific objects consuming excessive memory.",
                    "expected_improvement": 0.3,
                    "estimated_effort": "low"
                },
            ],
            BottleneckType.IO_BOUND: [
                {
                    "priority": 1,
                    "category": "async_io",
                    "description": "Replace synchronous I/O operations with asynchronous operations using asyncio.",
                    "expected_improvement": 0.4,
                    "estimated_effort": "high"
                },
                {
                    "priority": 2,
                    "category": "buffering",
                    "description": "Implement proper buffering for file operations to reduce I/O overhead.",
                    "expected_improvement": 0.2,
                    "estimated_effort": "low"
                },
                {
                    "priority": 3,
                    "category": "caching",
                    "description": "Cache frequently accessed data to reduce repeated I/O operations.",
                    "expected_improvement": 0.35,
                    "estimated_effort": "medium"
                },
            ],
            BottleneckType.GARBAGE_COLLECTION: [
                {
                    "priority": 1,
                    "category": "object_pool",
                    "description": "Implement object pooling to reduce the number of allocations and garbage collections.",
                    "expected_improvement": 0.3,
                    "estimated_effort": "high"
                },
                {
                    "priority": 2,
                    "category": "gc_tuning",
                    "description": "Tune garbage collector thresholds and disable GC during critical performance sections.",
                    "expected_improvement": 0.2,
                    "estimated_effort": "low"
                },
                {
                    "priority": 3,
                    "category": "memory_efficiency",
                    "description": "Reduce object churn by reusing objects and avoiding temporary object creation.",
                    "expected_improvement": 0.25,
                    "estimated_effort": "medium"
                },
            ],
            BottleneckType.THREAD_CONCURRENCY: [
                {
                    "priority": 1,
                    "category": "thread_pool",
                    "description": "Use a thread pool with limited size instead of creating unlimited threads.",
                    "expected_improvement": 0.35,
                    "estimated_effort": "medium"
                },
                {
                    "priority": 2,
                    "category": "async_concurrency",
                    "description": "Consider using asyncio for I/O-bound operations instead of threading.",
                    "expected_improvement": 0.3,
                    "estimated_effort": "high"
                },
                {
                    "priority": 3,
                    "category": "worker_pattern",
                    "description": "Implement a worker pattern with fixed pool size to limit concurrent execution.",
                    "expected_improvement": 0.25,
                    "estimated_effort": "medium"
                },
            ],
            BottleneckType.LOCK_CONTENTION: [
                {
                    "priority": 1,
                    "category": "lock_granularity",
                    "description": "Reduce lock granularity by using finer-grained locks or lock-free data structures.",
                    "expected_improvement": 0.35,
                    "estimated_effort": "high"
                },
                {
                    "priority": 2,
                    "category": "concurrent_data_structures",
                    "description": "Use thread-safe data structures from the concurrent.futures or queue modules.",
                    "expected_improvement": 0.25,
                    "estimated_effort": "medium"
                },
                {
                    "priority": 3,
                    "category": "lock_free_design",
                    "description": "Redesign to avoid locks where possible using atomic operations.",
                    "expected_improvement": 0.4,
                    "estimated_effort": "high"
                },
            ],
            BottleneckType.RECURSION_DEPTH: [
                {
                    "priority": 1,
                    "category": "iterative_refactor",
                    "description": "Replace recursive implementations with iterative approaches using loops.",
                    "expected_improvement": 0.3,
                    "estimated_effort": "medium"
                },
                {
                    "priority": 2,
                    "category": "tail_recursion",
                    "description": "If applicable, convert to tail recursion and use trampoline pattern.",
                    "expected_improvement": 0.25,
                    "estimated_effort": "medium"
                },
                {
                    "priority": 3,
                    "category": "stack_optimization",
                    "description": "Increase recursion limit or use explicit stack data structure.",
                    "expected_improvement": 0.15,
                    "estimated_effort": "low"
                },
            ],
        }

    def generate_suggestions(self, detections: List[BottleneckDetection]) -> List[OptimizationSuggestion]:
        suggestions = []
        for detection in detections:
            templates = self.suggestion_templates.get(detection.bottleneck_type, [])
            for template in templates:
                suggestion = OptimizationSuggestion(
                    suggestion_id=f"suggest_{int(time.time() * 1000)}_{detection.bottleneck_type.value}_{template['priority']}",
                    bottleneck_type=detection.bottleneck_type,
                    priority=template["priority"],
                    category=template["category"],
                    description=f"{detection.description} - {template['description']}",
                    expected_improvement=template["expected_improvement"],
                    estimated_effort=template["estimated_effort"],
                    related_metrics=[detection.metric_name]
                )
                suggestions.append(suggestion)

        suggestions.sort(key=lambda s: (s.priority, s.expected_improvement), reverse=False)
        return suggestions

    def prioritize_suggestions(self, suggestions: List[OptimizationSuggestion],
                               max_count: int = 10) -> List[OptimizationSuggestion]:
        return suggestions[:max_count]


class PerformanceDashboard:
    def __init__(self):
        self.collector = PerformanceCollector()
        self.analyzer = BottleneckAnalyzer()
        self.suggestion_generator = OptimizationSuggestionGenerator()
        self._last_analysis_time = 0.0
        self._analysis_interval = 5.0

    def start_monitoring(self):
        self.collector.start()

    def stop_monitoring(self):
        self.collector.stop()

    def run_analysis(self) -> Dict[str, Any]:
        now = time.time()
        if now - self._last_analysis_time < self._analysis_interval:
            return {"status": "skipped", "reason": "analysis_interval_not_elapsed"}

        self._last_analysis_time = now

        samples = self.collector.get_recent_samples(1000)
        detections = self.analyzer.analyze_samples(samples)
        suggestions = self.suggestion_generator.generate_suggestions(detections)
        prioritized = self.suggestion_generator.prioritize_suggestions(suggestions, 10)

        latest_sample = self.collector.get_latest_sample()

        return {
            "status": "completed",
            "timestamp": now,
            "sample_count": len(samples),
            "detections": [self._detection_to_dict(d) for d in detections],
            "suggestions": [self._suggestion_to_dict(s) for s in prioritized],
            "current_metrics": latest_sample.metrics if latest_sample else {},
            "total_detections": len(self.analyzer.detections),
        }

    def _detection_to_dict(self, detection: BottleneckDetection) -> Dict[str, Any]:
        return {
            "detection_id": detection.detection_id,
            "bottleneck_type": detection.bottleneck_type.value,
            "severity": detection.severity.name,
            "severity_value": detection.severity.value,
            "metric_name": detection.metric_name,
            "current_value": detection.current_value,
            "threshold_value": detection.threshold_value,
            "trend": detection.trend,
            "description": detection.description,
            "code_location": detection.code_location,
            "timestamp": detection.timestamp,
            "duration": detection.duration,
        }

    def _suggestion_to_dict(self, suggestion: OptimizationSuggestion) -> Dict[str, Any]:
        return {
            "suggestion_id": suggestion.suggestion_id,
            "bottleneck_type": suggestion.bottleneck_type.value,
            "priority": suggestion.priority,
            "category": suggestion.category,
            "description": suggestion.description,
            "code_change": suggestion.code_change,
            "expected_improvement": suggestion.expected_improvement,
            "estimated_effort": suggestion.estimated_effort,
            "related_metrics": suggestion.related_metrics,
        }

    def get_status(self) -> Dict[str, Any]:
        latest = self.collector.get_latest_sample()
        return {
            "monitoring": True,
            "sample_interval_ms": self.collector.sample_interval_ms,
            "total_samples": len(self.collector.samples),
            "latest_metrics": latest.metrics if latest else {},
            "total_detections": len(self.analyzer.detections),
            "analysis_interval": self._analysis_interval,
        }

    def get_recent_detections(self, count: int = 20) -> List[Dict[str, Any]]:
        detections = self.analyzer.get_recent_detections(count)
        return [self._detection_to_dict(d) for d in detections]

    def get_metric_data(self, metric_name: str, count: int = 100) -> List[Tuple[float, float]]:
        return self.collector.get_metric_history(metric_name, count)

    def add_custom_metric(self, name: str, value: float):
        self.collector.add_custom_metric(name, value)