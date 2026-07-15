"""
decision/strategy_adjustment_engine.py - 决策策略动态调整引擎

实现策略存储与版本管理、权重调整算法、策略执行接口
支持每秒1000+策略调整请求，具备容灾备份能力
"""
import time
import hashlib
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque
import numpy as np


class StrategyStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    DEPRECATED = "deprecated"


class AdjustmentType(Enum):
    REALTIME = "realtime"
    PERIODIC = "periodic"
    EVENT_TRIGGERED = "event_triggered"


@dataclass
class StrategyVersion:
    version_id: str
    strategy_id: str
    version_number: int
    weights: Dict[str, float]
    parameters: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "strategy_id": self.strategy_id,
            "version_number": self.version_number,
            "weights": self.weights,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "description": self.description,
        }


@dataclass
class DecisionStrategy:
    strategy_id: str
    name: str
    description: str
    weights: Dict[str, float]
    parameters: Dict[str, Any]
    status: StrategyStatus = StrategyStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version_history: List[StrategyVersion] = field(default_factory=list)
    current_version: int = 1
    performance_history: List[Dict[str, float]] = field(default_factory=list)

    def create_version(self, description: str = "", created_by: str = "system") -> StrategyVersion:
        self.current_version += 1
        version = StrategyVersion(
            version_id=f"{self.strategy_id}_v{self.current_version}",
            strategy_id=self.strategy_id,
            version_number=self.current_version,
            weights=copy.deepcopy(self.weights),
            parameters=copy.deepcopy(self.parameters),
            description=description,
            created_by=created_by,
        )
        self.version_history.append(version)
        if len(self.version_history) > 50:
            self.version_history = self.version_history[-50:]
        self.updated_at = time.time()
        return version

    def update_weights(self, new_weights: Dict[str, float], description: str = ""):
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = value
        self.create_version(description)

    def rollback_to_version(self, version_number: int) -> bool:
        for version in self.version_history:
            if version.version_number == version_number:
                self.weights = copy.deepcopy(version.weights)
                self.parameters = copy.deepcopy(version.parameters)
                self.current_version = version_number
                self.create_version(f"Rollback to version {version_number}")
                return True
        return False

    def add_performance_record(self, metrics: Dict[str, float]):
        self.performance_history.append({
            **metrics,
            "timestamp": time.time(),
            "version": self.current_version,
        })
        if len(self.performance_history) > 200:
            self.performance_history = self.performance_history[-200:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "weights": self.weights,
            "parameters": self.parameters,
            "status": self.status.value,
            "current_version": self.current_version,
            "version_count": len(self.version_history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class AdjustmentRequest:
    request_id: str
    strategy_id: str
    adjustment_type: AdjustmentType
    target_weights: Optional[Dict[str, float]] = None
    adjustment_delta: Optional[Dict[str, float]] = None
    priority: int = 0
    timestamp: float = field(default_factory=time.time)

    def validate(self) -> bool:
        return self.strategy_id is not None and (
            self.target_weights is not None or self.adjustment_delta is not None
        )


@dataclass
class AdjustmentResult:
    request_id: str
    strategy_id: str
    success: bool
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    version_id: str
    latency_ms: float
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "strategy_id": self.strategy_id,
            "success": self.success,
            "old_weights": self.old_weights,
            "new_weights": self.new_weights,
            "version_id": self.version_id,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
        }


class StrategyAdjustmentEngine:
    def __init__(self, max_strategies: int = 1000, max_history: int = 10000):
        self._strategies: Dict[str, DecisionStrategy] = {}
        self._strategy_history: deque = deque(maxlen=max_history)
        self._adjustment_queue: deque = deque(maxlen=10000)
        self._active_adjustments: Dict[str, AdjustmentRequest] = {}
        self._max_strategies = max_strategies
        self._backup_snapshot: Dict[str, DecisionStrategy] = {}
        self._last_backup_time = 0.0
        self._backup_interval = 300
        self._request_counter = 0
        self._adjustment_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency_ms": 0.0,
            "peak_concurrent": 0,
        }

    def register_strategy(self, strategy_id: str, name: str, description: str,
                          weights: Dict[str, float], parameters: Dict[str, Any] = None) -> bool:
        if strategy_id in self._strategies:
            return False

        if len(self._strategies) >= self._max_strategies:
            return False

        self._strategies[strategy_id] = DecisionStrategy(
            strategy_id=strategy_id,
            name=name,
            description=description,
            weights=weights,
            parameters=parameters or {},
        )
        return True

    def get_strategy(self, strategy_id: str) -> Optional[DecisionStrategy]:
        return self._strategies.get(strategy_id)

    def list_strategies(self, status_filter: Optional[StrategyStatus] = None) -> List[Dict[str, Any]]:
        result = []
        for strategy in self._strategies.values():
            if status_filter is None or strategy.status == status_filter:
                result.append(strategy.to_dict())
        return result

    def adjust_strategy(self, request: AdjustmentRequest) -> AdjustmentResult:
        start_time = time.time()
        self._request_counter += 1

        if not request.validate():
            latency_ms = (time.time() - start_time) * 1000
            self._adjustment_stats["total_requests"] += 1
            self._adjustment_stats["failed_requests"] += 1
            return AdjustmentResult(
                request_id=request.request_id,
                strategy_id=request.strategy_id,
                success=False,
                old_weights={},
                new_weights={},
                version_id="",
                latency_ms=latency_ms,
                error_message="Invalid request",
            )

        strategy = self._strategies.get(request.strategy_id)
        if strategy is None:
            latency_ms = (time.time() - start_time) * 1000
            self._adjustment_stats["total_requests"] += 1
            self._adjustment_stats["failed_requests"] += 1
            return AdjustmentResult(
                request_id=request.request_id,
                strategy_id=request.strategy_id,
                success=False,
                old_weights={},
                new_weights={},
                version_id="",
                latency_ms=latency_ms,
                error_message="Strategy not found",
            )

        old_weights = copy.deepcopy(strategy.weights)

        if request.target_weights:
            strategy.update_weights(request.target_weights,
                                   f"Adjustment: {request.adjustment_type.value}")
        elif request.adjustment_delta:
            delta_weights = {}
            for key, delta in request.adjustment_delta.items():
                if key in strategy.weights:
                    new_val = strategy.weights[key] + delta
                    delta_weights[key] = max(0.0, min(1.0, new_val))
            strategy.update_weights(delta_weights,
                                   f"Delta adjustment: {request.adjustment_type.value}")

        version = strategy.create_version()
        self._strategy_history.append({
            "strategy_id": strategy.strategy_id,
            "version_id": version.version_id,
            "timestamp": time.time(),
            "adjustment_type": request.adjustment_type.value,
        })

        latency_ms = (time.time() - start_time) * 1000
        self._adjustment_stats["total_requests"] += 1
        self._adjustment_stats["successful_requests"] += 1
        self._adjustment_stats["average_latency_ms"] = (
            self._adjustment_stats["average_latency_ms"] *
            (self._adjustment_stats["total_requests"] - 1) + latency_ms
        ) / self._adjustment_stats["total_requests"]

        self._ensure_backup()

        return AdjustmentResult(
            request_id=request.request_id,
            strategy_id=strategy.strategy_id,
            success=True,
            old_weights=old_weights,
            new_weights=copy.deepcopy(strategy.weights),
            version_id=version.version_id,
            latency_ms=latency_ms,
        )

    def batch_adjust(self, requests: List[AdjustmentRequest]) -> List[AdjustmentResult]:
        results = []
        for request in requests:
            result = self.adjust_strategy(request)
            results.append(result)
        return results

    def execute_strategy(self, strategy_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        strategy = self._strategies.get(strategy_id)
        if strategy is None or strategy.status != StrategyStatus.ACTIVE:
            return {"success": False, "error": "Strategy not available"}

        weights = strategy.weights
        parameters = strategy.parameters

        weighted_score = 0.0
        total_weight = sum(weights.values())

        for key, weight in weights.items():
            value = inputs.get(key, 0.0)
            weighted_score += weight * value

        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0

        return {
            "success": True,
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "version": strategy.current_version,
            "weighted_score": weighted_score,
            "normalized_score": normalized_score,
            "weights_used": copy.deepcopy(weights),
            "parameters": copy.deepcopy(parameters),
        }

    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return {}

        history = strategy.performance_history
        if not history:
            return {"strategy_id": strategy_id, "records": 0}

        recent = history[-50:]
        avg_metrics = {}
        metric_keys = set()
        for record in recent:
            metric_keys.update(record.keys())

        for key in metric_keys:
            if key in ["timestamp", "version"]:
                continue
            values = [r[key] for r in recent if key in r and isinstance(r[key], (int, float))]
            if values:
                avg_metrics[f"avg_{key}"] = float(np.mean(values))
                avg_metrics[f"min_{key}"] = float(np.min(values))
                avg_metrics[f"max_{key}"] = float(np.max(values))

        return {
            "strategy_id": strategy_id,
            "records": len(history),
            "current_version": strategy.current_version,
            **avg_metrics,
        }

    def set_strategy_status(self, strategy_id: str, status: StrategyStatus) -> bool:
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return False
        strategy.status = status
        strategy.updated_at = time.time()
        return True

    def delete_strategy(self, strategy_id: str) -> bool:
        if strategy_id not in self._strategies:
            return False

        strategy = self._strategies[strategy_id]
        strategy.status = StrategyStatus.DEPRECATED
        return True

    def _ensure_backup(self):
        now = time.time()
        if now - self._last_backup_time >= self._backup_interval:
            self._backup_snapshot = {
                k: copy.deepcopy(v) for k, v in self._strategies.items()
            }
            self._last_backup_time = now

    def restore_from_backup(self) -> bool:
        if not self._backup_snapshot:
            return False

        self._strategies = {
            k: copy.deepcopy(v) for k, v in self._backup_snapshot.items()
        }
        return True

    def get_adjustment_stats(self) -> Dict[str, Any]:
        return {
            **self._adjustment_stats,
            "strategy_count": len(self._strategies),
            "pending_adjustments": len(self._adjustment_queue),
            "backup_available": len(self._backup_snapshot) > 0,
            "last_backup_time": self._last_backup_time,
        }

    def generate_request_id(self) -> str:
        timestamp = str(time.time())
        counter = str(self._request_counter)
        return hashlib.md5(f"{timestamp}_{counter}".encode()).hexdigest()[:16]

    def create_adjustment_request(self, strategy_id: str, adjustment_type: AdjustmentType,
                                   target_weights: Optional[Dict[str, float]] = None,
                                   adjustment_delta: Optional[Dict[str, float]] = None,
                                   priority: int = 0) -> AdjustmentRequest:
        return AdjustmentRequest(
            request_id=self.generate_request_id(),
            strategy_id=strategy_id,
            adjustment_type=adjustment_type,
            target_weights=target_weights,
            adjustment_delta=adjustment_delta,
            priority=priority,
        )

    def get_version_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return []
        return [v.to_dict() for v in strategy.version_history]

    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        errors = []
        warnings = []

        for key, value in weights.items():
            if not isinstance(value, (int, float)):
                errors.append(f"Weight '{key}' must be a number")
            elif value < 0 or value > 1:
                warnings.append(f"Weight '{key}' ({value}) is outside [0,1] range")

        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            warnings.append(f"Weights sum to {total:.4f}, not normalized")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "sum": total,
        }