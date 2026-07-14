"""
meta_evolution/__init__.py - 元进化+遗传算法模块

实现基于遗传算法的系统自我优化与进化机制，支持参数自适应调整和结构优化

核心组件：
- GeneticAlgorithm: 遗传算法核心引擎
- GenePool: 基因池管理
- EvolutionController: 进化控制器
- ParameterOptimizer: 参数自适应优化器
- StructuralOptimizer: 结构优化器
- EvolutionOrchestrator: 进化编排器
"""
from .genetic_algorithm import (
    GeneticAlgorithm, Genome, Gene, FitnessFunction,
    SelectionMethod, CrossoverMethod, MutationMethod,
    EvolutionResult, Population,
)
from .gene_pool import GenePool, GeneTemplate, GeneLibrary
from .evolution_controller import (
    EvolutionController, EvolutionPhase, EvolutionStrategy,
    EvolutionMonitor, EvolutionConfig,
)
from .parameter_optimizer import (
    ParameterOptimizer, AdaptiveParameter, ParameterSpace,
    ParameterTuningResult,
)
from .structural_optimizer import (
    StructuralOptimizer, StructureEncoding, StructuralMutation,
    StructureEvolutionResult,
)
from .orchestrator import EvolutionOrchestrator

__all__ = [
    # 遗传算法核心
    "GeneticAlgorithm", "Genome", "Gene", "FitnessFunction",
    "SelectionMethod", "CrossoverMethod", "MutationMethod",
    "EvolutionResult", "Population",
    # 基因池
    "GenePool", "GeneTemplate", "GeneLibrary",
    # 进化控制
    "EvolutionController", "EvolutionPhase", "EvolutionStrategy",
    "EvolutionMonitor", "EvolutionConfig",
    # 参数优化
    "ParameterOptimizer", "AdaptiveParameter", "ParameterSpace",
    "ParameterTuningResult",
    # 结构优化
    "StructuralOptimizer", "StructureEncoding", "StructuralMutation",
    "StructureEvolutionResult",
    # 编排器
    "EvolutionOrchestrator",
]