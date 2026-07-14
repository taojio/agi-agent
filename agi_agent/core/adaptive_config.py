import os
import sys
import json
import time
import psutil
import torch
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum


class PerformanceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AdaptiveStrategy(Enum):
    ENVIRONMENT = "environment"
    PERFORMANCE = "performance"
    LEARNING = "learning"
    USER_PREFERENCE = "user_preference"


@dataclass
class AdaptiveParameter:
    name: str
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    strategy: AdaptiveStrategy = AdaptiveStrategy.PERFORMANCE
    dynamic_fn: Optional[Callable[[Dict[str, float]], Any]] = None
    dependencies: List[str] = field(default_factory=list)
    
    def compute(self, context: Dict[str, float]) -> Any:
        if self.dynamic_fn:
            try:
                value = self.dynamic_fn(context)
            except Exception:
                value = self.default_value
        else:
            value = self.default_value
        
        if self.min_value is not None and isinstance(value, (int, float)):
            value = max(self.min_value, value)
        if self.max_value is not None and isinstance(value, (int, float)):
            value = min(self.max_value, value)
        
        return value


class EnvironmentContext:
    def __init__(self):
        self._context = self._detect_environment()
    
    def _detect_environment(self) -> Dict[str, float]:
        context = {}
        
        try:
            context["cpu_cores"] = float(psutil.cpu_count(logical=True) or 1)
            context["cpu_percent"] = float(psutil.cpu_percent())
            context["memory_total_gb"] = float(psutil.virtual_memory().total / (1024 ** 3))
            context["memory_used_gb"] = float(psutil.virtual_memory().used / (1024 ** 3))
            context["memory_percent"] = float(psutil.virtual_memory().percent)
        except Exception:
            context["cpu_cores"] = 4.0
            context["cpu_percent"] = 0.0
            context["memory_total_gb"] = 8.0
            context["memory_used_gb"] = 2.0
            context["memory_percent"] = 25.0
        
        try:
            context["gpu_available"] = float(torch.cuda.is_available())
            if torch.cuda.is_available():
                context["gpu_count"] = float(torch.cuda.device_count())
                context["gpu_memory_gb"] = float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
            else:
                context["gpu_count"] = 0.0
                context["gpu_memory_gb"] = 0.0
        except Exception:
            context["gpu_available"] = 0.0
            context["gpu_count"] = 0.0
            context["gpu_memory_gb"] = 0.0
        
        context["python_version"] = float(f"{sys.version_info.major}.{sys.version_info.minor}")
        
        try:
            disk = psutil.disk_usage("/")
            context["disk_total_gb"] = float(disk.total / (1024 ** 3))
            context["disk_used_gb"] = float(disk.used / (1024 ** 3))
            context["disk_percent"] = float(disk.percent)
        except Exception:
            context["disk_total_gb"] = 100.0
            context["disk_used_gb"] = 20.0
            context["disk_percent"] = 20.0
        
        return context
    
    def get_performance_level(self) -> PerformanceLevel:
        if self._context["gpu_available"] > 0 and self._context["gpu_memory_gb"] > 4:
            return PerformanceLevel.HIGH
        if self._context["cpu_cores"] >= 8 and self._context["memory_total_gb"] >= 16:
            return PerformanceLevel.HIGH
        if self._context["cpu_cores"] >= 4 and self._context["memory_total_gb"] >= 8:
            return PerformanceLevel.MEDIUM
        return PerformanceLevel.LOW
    
    def update(self):
        self._context.update(self._detect_environment())
    
    def get(self, key: str, default: float = 0.0) -> float:
        return self._context.get(key, default)
    
    def to_dict(self) -> Dict[str, float]:
        return self._context.copy()


class AdaptiveConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        if self._initialized:
            return
        
        self.env_context = EnvironmentContext()
        self.performance_level = self.env_context.get_performance_level()
        
        self._params: Dict[str, AdaptiveParameter] = {}
        self._runtime_context: Dict[str, float] = {}
        self._param_cache: Dict[str, Any] = {}
        
        self._register_core_parameters()
        self._register_agent_parameters()
        self._register_learning_parameters()
        self._register_perception_parameters()
        self._register_memory_parameters()
        self._register_evolution_parameters()
        self._register_self_improvement_parameters()
        self._register_homeostatic_parameters()
        self._register_active_inference_parameters()
        self._register_security_parameters()
        
        self._initialized = True
    
    def _register_core_parameters(self):
        self.register(AdaptiveParameter(
            name="input_dim",
            default_value=16,
            min_value=8,
            max_value=128,
            description="输入特征维度",
            strategy=AdaptiveStrategy.ENVIRONMENT,
            dynamic_fn=self._compute_input_dim
        ))
        
        self.register(AdaptiveParameter(
            name="action_dim",
            default_value=8,
            min_value=4,
            max_value=64,
            description="动作空间维度",
            strategy=AdaptiveStrategy.ENVIRONMENT,
            dynamic_fn=self._compute_action_dim
        ))
        
        self.register(AdaptiveParameter(
            name="hidden_dim",
            default_value=64,
            min_value=16,
            max_value=512,
            description="隐藏层维度",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_hidden_dim
        ))
        
        self.register(AdaptiveParameter(
            name="log_interval",
            default_value=20,
            min_value=5,
            max_value=100,
            description="日志输出间隔",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_log_interval
        ))
        
        self.register(AdaptiveParameter(
            name="save_interval",
            default_value=1000,
            min_value=100,
            max_value=5000,
            description="模型保存间隔",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_save_interval
        ))
        
        self.register(AdaptiveParameter(
            name="eval_interval",
            default_value=500,
            min_value=50,
            max_value=2000,
            description="性能评估间隔",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_eval_interval
        ))
    
    def _register_agent_parameters(self):
        self.register(AdaptiveParameter(
            name="free_energy_threshold",
            default_value=0.3,
            min_value=0.1,
            max_value=0.8,
            description="自由能阈值",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_free_energy_threshold
        ))
        
        self.register(AdaptiveParameter(
            name="novelty_threshold",
            default_value=0.5,
            min_value=0.2,
            max_value=0.9,
            description="新颖度阈值",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_novelty_threshold
        ))
        
        self.register(AdaptiveParameter(
            name="confidence_threshold",
            default_value=0.5,
            min_value=0.2,
            max_value=0.9,
            description="置信度阈值",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_confidence_threshold
        ))
        
        self.register(AdaptiveParameter(
            name="max_inference_step",
            default_value=5,
            min_value=1,
            max_value=20,
            description="最大推理步数",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_max_inference_step
        ))
        
        self.register(AdaptiveParameter(
            name="max_concurrent_tasks",
            default_value=3,
            min_value=1,
            max_value=10,
            description="最大并发任务数",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_max_concurrent_tasks
        ))
    
    def _register_learning_parameters(self):
        self.register(AdaptiveParameter(
            name="initial_learning_rate",
            default_value=1e-3,
            min_value=1e-6,
            max_value=1e-1,
            description="初始学习率",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_initial_learning_rate
        ))
        
        self.register(AdaptiveParameter(
            name="learning_rate_pool",
            default_value=[1e-4, 5e-4, 1e-3, 2e-3],
            description="学习率池",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_learning_rate_pool
        ))
        
        self.register(AdaptiveParameter(
            name="growth_step",
            default_value=8,
            min_value=2,
            max_value=32,
            description="网络增长步长",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_growth_step
        ))
        
        self.register(AdaptiveParameter(
            name="prune_step",
            default_value=4,
            min_value=1,
            max_value=16,
            description="网络剪枝步长",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_prune_step
        ))
        
        self.register(AdaptiveParameter(
            name="min_hidden_dim",
            default_value=16,
            min_value=8,
            max_value=64,
            description="最小隐藏层维度",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_min_hidden_dim
        ))
        
        self.register(AdaptiveParameter(
            name="max_hidden_dim",
            default_value=256,
            min_value=64,
            max_value=1024,
            description="最大隐藏层维度",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_max_hidden_dim
        ))
    
    def _register_perception_parameters(self):
        self.register(AdaptiveParameter(
            name="feature_dim",
            default_value=64,
            min_value=16,
            max_value=256,
            description="特征维度",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_feature_dim
        ))
        
        self.register(AdaptiveParameter(
            name="autoencoder_layers",
            default_value=3,
            min_value=1,
            max_value=5,
            description="自编码器层数",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_autoencoder_layers
        ))
        
        self.register(AdaptiveParameter(
            name="fusion_output_dim",
            default_value=64,
            min_value=16,
            max_value=256,
            description="多模态融合输出维度",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_fusion_output_dim
        ))
    
    def _register_memory_parameters(self):
        self.register(AdaptiveParameter(
            name="memory_buffer_size",
            default_value=200,
            min_value=50,
            max_value=1000,
            description="记忆缓冲区大小",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_memory_buffer_size
        ))
        
        self.register(AdaptiveParameter(
            name="knowledge_max_size",
            default_value=1000,
            min_value=200,
            max_value=5000,
            description="知识图谱最大大小",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_knowledge_max_size
        ))
        
        self.register(AdaptiveParameter(
            name="history_max_len",
            default_value=50,
            min_value=10,
            max_value=200,
            description="历史记录最大长度",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_history_max_len
        ))
    
    def _register_evolution_parameters(self):
        self.register(AdaptiveParameter(
            name="evolve_trigger_step",
            default_value=200,
            min_value=50,
            max_value=1000,
            description="进化触发间隔",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_evolve_trigger_step
        ))
        
        self.register(AdaptiveParameter(
            name="population_size",
            default_value=10,
            min_value=3,
            max_value=50,
            description="进化种群大小",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_population_size
        ))
        
        self.register(AdaptiveParameter(
            name="mutation_rate",
            default_value=0.1,
            min_value=0.01,
            max_value=0.5,
            description="变异率",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_mutation_rate
        ))
    
    def _register_self_improvement_parameters(self):
        self.register(AdaptiveParameter(
            name="improvement_interval",
            default_value=100,
            min_value=20,
            max_value=500,
            description="自我改进检查间隔",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_improvement_interval
        ))
        
        self.register(AdaptiveParameter(
            name="evaluation_window",
            default_value=20,
            min_value=5,
            max_value=100,
            description="评估窗口大小",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_evaluation_window
        ))
        
        self.register(AdaptiveParameter(
            name="min_improvement_required",
            default_value=0.05,
            min_value=0.01,
            max_value=0.2,
            description="最小改进要求",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_min_improvement_required
        ))
        
        self.register(AdaptiveParameter(
            name="max_concurrent_improvements",
            default_value=3,
            min_value=1,
            max_value=10,
            description="最大并发改进数",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_max_concurrent_improvements
        ))
        
        self.register(AdaptiveParameter(
            name="free_energy_critical_high",
            default_value=0.8,
            min_value=0.5,
            max_value=0.95,
            description="自由能临界高值",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_free_energy_critical_high
        ))
        
        self.register(AdaptiveParameter(
            name="confidence_critical_low",
            default_value=0.2,
            min_value=0.05,
            max_value=0.4,
            description="置信度临界低值",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_confidence_critical_low
        ))
        
        self.register(AdaptiveParameter(
            name="free_energy_warning_high",
            default_value=0.6,
            min_value=0.4,
            max_value=0.8,
            description="自由能警告高值",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_free_energy_warning_high
        ))
        
        self.register(AdaptiveParameter(
            name="confidence_warning_low",
            default_value=0.4,
            min_value=0.2,
            max_value=0.6,
            description="置信度警告低值",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_confidence_warning_low
        ))
    
    def _register_homeostatic_parameters(self):
        self.register(AdaptiveParameter(
            name="energy_baseline",
            default_value=0.7,
            min_value=0.3,
            max_value=0.9,
            description="能量基线值",
            strategy=AdaptiveStrategy.LEARNING,
            dynamic_fn=self._compute_energy_baseline
        ))
        self.register(AdaptiveParameter(
            name="energy_threshold_low",
            default_value=0.3,
            min_value=0.1,
            max_value=0.5,
            description="能量低阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="energy_threshold_high",
            default_value=0.9,
            min_value=0.7,
            max_value=0.99,
            description="能量高阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="energy_decay_rate",
            default_value=0.005,
            min_value=0.001,
            max_value=0.02,
            description="能量衰减率",
            strategy=AdaptiveStrategy.PERFORMANCE
        ))
        self.register(AdaptiveParameter(
            name="energy_gain_rate",
            default_value=0.08,
            min_value=0.02,
            max_value=0.2,
            description="能量增益率",
            strategy=AdaptiveStrategy.LEARNING
        ))

        self.register(AdaptiveParameter(
            name="attention_baseline",
            default_value=0.6,
            min_value=0.3,
            max_value=0.8,
            description="注意力基线值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="attention_threshold_low",
            default_value=0.2,
            min_value=0.1,
            max_value=0.4,
            description="注意力低阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="attention_threshold_high",
            default_value=0.85,
            min_value=0.6,
            max_value=0.95,
            description="注意力高阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="attention_decay_rate",
            default_value=0.01,
            min_value=0.002,
            max_value=0.03,
            description="注意力衰减率",
            strategy=AdaptiveStrategy.PERFORMANCE
        ))
        self.register(AdaptiveParameter(
            name="attention_gain_rate",
            default_value=0.06,
            min_value=0.02,
            max_value=0.15,
            description="注意力增益率",
            strategy=AdaptiveStrategy.LEARNING
        ))

        self.register(AdaptiveParameter(
            name="security_baseline",
            default_value=0.8,
            min_value=0.5,
            max_value=0.95,
            description="安全感基线值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="security_threshold_low",
            default_value=0.4,
            min_value=0.2,
            max_value=0.6,
            description="安全感低阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="security_threshold_high",
            default_value=0.95,
            min_value=0.8,
            max_value=0.99,
            description="安全感高阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="security_decay_rate",
            default_value=0.003,
            min_value=0.001,
            max_value=0.01,
            description="安全感衰减率",
            strategy=AdaptiveStrategy.PERFORMANCE
        ))
        self.register(AdaptiveParameter(
            name="security_gain_rate",
            default_value=0.07,
            min_value=0.02,
            max_value=0.15,
            description="安全感增益率",
            strategy=AdaptiveStrategy.LEARNING
        ))

        self.register(AdaptiveParameter(
            name="curiosity_baseline",
            default_value=0.5,
            min_value=0.2,
            max_value=0.8,
            description="好奇心基线值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="curiosity_threshold_low",
            default_value=0.2,
            min_value=0.1,
            max_value=0.4,
            description="好奇心低阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="curiosity_threshold_high",
            default_value=0.8,
            min_value=0.6,
            max_value=0.95,
            description="好奇心高阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="curiosity_decay_rate",
            default_value=0.008,
            min_value=0.002,
            max_value=0.02,
            description="好奇心衰减率",
            strategy=AdaptiveStrategy.PERFORMANCE
        ))
        self.register(AdaptiveParameter(
            name="curiosity_gain_rate",
            default_value=0.04,
            min_value=0.01,
            max_value=0.1,
            description="好奇心增益率",
            strategy=AdaptiveStrategy.LEARNING
        ))

        self.register(AdaptiveParameter(
            name="competence_baseline",
            default_value=0.5,
            min_value=0.2,
            max_value=0.8,
            description="能力感基线值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="competence_threshold_low",
            default_value=0.25,
            min_value=0.1,
            max_value=0.5,
            description="能力感低阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="competence_threshold_high",
            default_value=0.85,
            min_value=0.6,
            max_value=0.95,
            description="能力感高阈值",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="competence_decay_rate",
            default_value=0.002,
            min_value=0.001,
            max_value=0.01,
            description="能力感衰减率",
            strategy=AdaptiveStrategy.PERFORMANCE
        ))
        self.register(AdaptiveParameter(
            name="competence_gain_rate",
            default_value=0.03,
            min_value=0.01,
            max_value=0.08,
            description="能力感增益率",
            strategy=AdaptiveStrategy.LEARNING
        ))

    def _register_active_inference_parameters(self):
        self.register(AdaptiveParameter(
            name="prediction_history_len",
            default_value=200,
            min_value=50,
            max_value=500,
            description="预测历史长度",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_prediction_history_len
        ))
        self.register(AdaptiveParameter(
            name="variational_params_len",
            default_value=100,
            min_value=20,
            max_value=200,
            description="变分参数历史长度",
            strategy=AdaptiveStrategy.PERFORMANCE
        ))
        self.register(AdaptiveParameter(
            name="action_noise_scale",
            default_value=0.1,
            min_value=0.01,
            max_value=0.3,
            description="动作噪声尺度",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="active_inference_lr",
            default_value=1e-3,
            min_value=1e-5,
            max_value=1e-1,
            description="主动推理学习率",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="epistemic_weight",
            default_value=0.5,
            min_value=0.1,
            max_value=0.9,
            description="认知权重",
            strategy=AdaptiveStrategy.LEARNING
        ))
        self.register(AdaptiveParameter(
            name="active_inference_beta",
            default_value=1.0,
            min_value=0.1,
            max_value=2.0,
            description="主动推理温度系数",
            strategy=AdaptiveStrategy.LEARNING
        ))

    def _register_security_parameters(self):
        self.register(AdaptiveParameter(
            name="safety_max_memory_gb",
            default_value=4.0,
            min_value=1.0,
            max_value=32.0,
            description="安全最大内存使用",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_safety_max_memory_gb
        ))
        
        self.register(AdaptiveParameter(
            name="safety_max_gpu_util",
            default_value=0.95,
            min_value=0.7,
            max_value=0.99,
            description="安全最大GPU使用率",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_safety_max_gpu_util
        ))
        
        self.register(AdaptiveParameter(
            name="safety_max_latency_ms",
            default_value=1000,
            min_value=100,
            max_value=5000,
            description="安全最大延迟",
            strategy=AdaptiveStrategy.PERFORMANCE,
            dynamic_fn=self._compute_safety_max_latency_ms
        ))
    
    def register(self, param: AdaptiveParameter):
        self._params[param.name] = param
    
    def get(self, name: str, default: Any = None) -> Any:
        if name not in self._params:
            return default
        
        param = self._params[name]
        cache_key = f"{name}_{hash(tuple(self._runtime_context.items()))}"
        
        if cache_key not in self._param_cache:
            context = {**self.env_context.to_dict(), **self._runtime_context}
            value = param.compute(context)
            self._param_cache[cache_key] = value
        
        return self._param_cache[cache_key]
    
    def set_runtime_context(self, key: str, value: float):
        self._runtime_context[key] = value
        self._param_cache.clear()
    
    def update_environment(self):
        self.env_context.update()
        self.performance_level = self.env_context.get_performance_level()
        self._param_cache.clear()
    
    def clear_cache(self):
        self._param_cache.clear()
    
    def get_all_params(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, param in self._params.items():
            result[name] = {
                "value": self.get(name),
                "default": param.default_value,
                "min": param.min_value,
                "max": param.max_value,
                "description": param.description,
                "strategy": param.strategy.value
            }
        return result
    
    def get_environment_info(self) -> Dict[str, float]:
        return self.env_context.to_dict()
    
    def get_performance_level(self) -> str:
        return self.performance_level.value
    
    def _compute_input_dim(self, context: Dict[str, float]) -> int:
        if context.get("gpu_available", 0) > 0:
            return 32
        if context.get("memory_total_gb", 0) >= 16:
            return 24
        return 16
    
    def _compute_action_dim(self, context: Dict[str, float]) -> int:
        return max(4, min(16, int(context.get("cpu_cores", 4) * 2)))
    
    def _compute_hidden_dim(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 256
        if self.performance_level == PerformanceLevel.MEDIUM:
            return 128
        return 64
    
    def _compute_log_interval(self, context: Dict[str, float]) -> int:
        if context.get("cpu_percent", 0) > 80:
            return 50
        return 20
    
    def _compute_save_interval(self, context: Dict[str, float]) -> int:
        if context.get("disk_percent", 0) > 80:
            return 2000
        return 1000
    
    def _compute_eval_interval(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 200
        return 500
    
    def _compute_free_energy_threshold(self, context: Dict[str, float]) -> float:
        return 0.3
    
    def _compute_novelty_threshold(self, context: Dict[str, float]) -> float:
        exploration_bonus = context.get("exploration_factor", 0)
        return min(0.9, 0.5 + exploration_bonus * 0.2)
    
    def _compute_confidence_threshold(self, context: Dict[str, float]) -> float:
        return 0.5
    
    def _compute_max_inference_step(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 10
        if self.performance_level == PerformanceLevel.MEDIUM:
            return 7
        return 5
    
    def _compute_max_concurrent_tasks(self, context: Dict[str, float]) -> int:
        return min(10, max(2, int(context.get("cpu_cores", 4) // 2)))
    
    def _compute_initial_learning_rate(self, context: Dict[str, float]) -> float:
        return 1e-3
    
    def _compute_learning_rate_pool(self, context: Dict[str, float]) -> List[float]:
        return [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    
    def _compute_growth_step(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 16
        return 8
    
    def _compute_prune_step(self, context: Dict[str, float]) -> int:
        return self.get("growth_step") // 2
    
    def _compute_min_hidden_dim(self, context: Dict[str, float]) -> int:
        return 16
    
    def _compute_max_hidden_dim(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 512
        if self.performance_level == PerformanceLevel.MEDIUM:
            return 256
        return 128
    
    def _compute_feature_dim(self, context: Dict[str, float]) -> int:
        return self.get("hidden_dim")
    
    def _compute_autoencoder_layers(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 4
        return 3
    
    def _compute_fusion_output_dim(self, context: Dict[str, float]) -> int:
        return self.get("feature_dim")
    
    def _compute_memory_buffer_size(self, context: Dict[str, float]) -> int:
        mem_gb = context.get("memory_total_gb", 8)
        return min(1000, max(100, int(mem_gb * 100)))
    
    def _compute_knowledge_max_size(self, context: Dict[str, float]) -> int:
        mem_gb = context.get("memory_total_gb", 8)
        return min(5000, max(500, int(mem_gb * 500)))
    
    def _compute_history_max_len(self, context: Dict[str, float]) -> int:
        return 50
    
    def _compute_evolve_trigger_step(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 100
        if self.performance_level == PerformanceLevel.MEDIUM:
            return 150
        return 200
    
    def _compute_population_size(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 20
        return 10
    
    def _compute_mutation_rate(self, context: Dict[str, float]) -> float:
        return 0.1
    
    def _compute_improvement_interval(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 50
        if self.performance_level == PerformanceLevel.MEDIUM:
            return 75
        return 100
    
    def _compute_evaluation_window(self, context: Dict[str, float]) -> int:
        return 20
    
    def _compute_min_improvement_required(self, context: Dict[str, float]) -> float:
        return 0.05
    
    def _compute_max_concurrent_improvements(self, context: Dict[str, float]) -> int:
        return min(5, max(2, int(context.get("cpu_cores", 4) // 3)))
    
    def _compute_free_energy_critical_high(self, context: Dict[str, float]) -> float:
        return 0.8
    
    def _compute_confidence_critical_low(self, context: Dict[str, float]) -> float:
        return 0.2
    
    def _compute_free_energy_warning_high(self, context: Dict[str, float]) -> float:
        return 0.6
    
    def _compute_confidence_warning_low(self, context: Dict[str, float]) -> float:
        return 0.4
    
    def _compute_safety_max_memory_gb(self, context: Dict[str, float]) -> float:
        return context.get("memory_total_gb", 8) * 0.8
    
    def _compute_safety_max_gpu_util(self, context: Dict[str, float]) -> float:
        return 0.95
    
    def _compute_safety_max_latency_ms(self, context: Dict[str, float]) -> int:
        if self.performance_level == PerformanceLevel.HIGH:
            return 500
        return 1000

    def _compute_energy_baseline(self, context: Dict[str, float]) -> float:
        cpu_util = context.get("cpu_percent", 0)
        if cpu_util > 70:
            return 0.6
        return 0.7

    def _compute_prediction_history_len(self, context: Dict[str, float]) -> int:
        mem_gb = context.get("memory_total_gb", 8)
        return min(500, max(100, int(mem_gb * 50)))


_adaptive_config = None


def get_adaptive_config() -> AdaptiveConfigManager:
    global _adaptive_config
    if _adaptive_config is None:
        _adaptive_config = AdaptiveConfigManager()
        _adaptive_config.initialize()
    return _adaptive_config


def adapt_param(name: str, default: Any = None) -> Any:
    return get_adaptive_config().get(name, default)