"""
ai_algorithms/scheduling.py - 智能调度组件

支持：
- 遗传算法调度
- 模拟退火调度
- 贪心调度
"""
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import AIAlgorithmComponent, AlgorithmMetrics, AlgorithmStatus


@dataclass
class Task:
    """任务"""
    task_id: str
    duration: float                # 持续时间
    priority: int = 1              # 优先级
    resource_requirement: Dict[str, float] = field(default_factory=dict)
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    value: float = 1.0             # 完成价值

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "duration": float(self.duration),
            "priority": self.priority,
            "resource_requirement": dict(self.resource_requirement),
            "deadline": self.deadline,
            "dependencies": list(self.dependencies),
            "value": float(self.value),
        }


@dataclass
class ScheduleResult:
    """调度结果"""
    schedule: List[Tuple[str, float]]    # (task_id, start_time)
    makespan: float                       # 总完成时间
    total_value: float                    # 总价值
    fitness: float                        # 适应度
    method: str = ""
    iterations: int = 0
    feasible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule": [(t, float(s)) for t, s in self.schedule],
            "makespan": float(self.makespan),
            "total_value": float(self.total_value),
            "fitness": float(self.fitness),
            "method": self.method,
            "iterations": self.iterations,
            "feasible": self.feasible,
        }


class GeneticScheduler(AIAlgorithmComponent):
    """遗传算法调度器"""

    def __init__(self, name: str = "genetic_scheduler",
                 population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, elite_ratio: float = 0.1):
        super().__init__(name, population_size=population_size,
                         generations=generations, mutation_rate=mutation_rate)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self._tasks: List[Task] = []
        self._best_schedule: Optional[List[int]] = None
        self._best_fitness: float = 0.0

    @property
    def component_type(self) -> str:
        return "scheduling"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GeneticScheduler":
        """X 为任务列表"""
        start = self._start_training()
        try:
            if isinstance(X, list) and X and isinstance(X[0], Task):
                self._tasks = X
            else:
                raise ValueError("X should be list of Task objects")

            # 初始化种群
            n_tasks = len(self._tasks)
            population = [list(np.random.permutation(n_tasks))
                         for _ in range(self.population_size)]

            for gen in range(self.generations):
                # 评估适应度
                fitness_scores = [self._evaluate_schedule(ind) for ind in population]

                # 记录最佳
                best_idx = int(np.argmax(fitness_scores))
                if fitness_scores[best_idx] > self._best_fitness:
                    self._best_fitness = fitness_scores[best_idx]
                    self._best_schedule = list(population[best_idx])

                # 选择（轮盘赌）
                total_fitness = sum(max(f, 0) for f in fitness_scores)
                if total_fitness == 0:
                    # 全部不可行，随机
                    new_population = [list(np.random.permutation(n_tasks))
                                    for _ in range(self.population_size)]
                else:
                    probs = [max(f, 0) / total_fitness for f in fitness_scores]
                    # 精英保留
                    n_elite = max(1, int(self.elite_ratio * self.population_size))
                    elite_indices = np.argsort(fitness_scores)[-n_elite:]
                    new_population = [list(population[i]) for i in elite_indices]
                    # 交叉
                    while len(new_population) < self.population_size:
                        parent1, parent2 = random.choices(population, weights=probs, k=2)
                        child = self._crossover(parent1, parent2, n_tasks)
                        new_population.append(child)

                # 变异
                for i in range(len(new_population)):
                    if random.random() < self.mutation_rate:
                        new_population[i] = self._mutate(new_population[i])

                population = new_population

            self.metrics.sample_count = n_tasks
            self.metrics.custom["best_fitness"] = self._best_fitness
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def _evaluate_schedule(self, schedule: List[int]) -> float:
        """评估调度方案（适应度）"""
        if not self._is_feasible(schedule):
            return 0.0
        makespan, total_value = self._compute_metrics(schedule)
        # 适应度：价值高 + 完成时间短
        return total_value / (makespan + 1e-6)

    def _is_feasible(self, schedule: List[int]) -> bool:
        """检查依赖是否满足"""
        scheduled_set = set()
        for idx in schedule:
            task = self._tasks[idx]
            for dep in task.dependencies:
                if dep not in [self._tasks[i].task_id for i in scheduled_set]:
                    return False
            scheduled_set.add(idx)
        return True

    def _compute_metrics(self, schedule: List[int]) -> Tuple[float, float]:
        """计算 makespan 和总价值"""
        current_time = 0.0
        total_value = 0.0
        for idx in schedule:
            task = self._tasks[idx]
            current_time += task.duration
            total_value += task.value
        return current_time, total_value

    @staticmethod
    def _crossover(parent1: List[int], parent2: List[int],
                    n_tasks: int) -> List[int]:
        """顺序交叉（OX）"""
        if n_tasks <= 1:
            return list(parent1)
        start, end = sorted(random.sample(range(n_tasks), 2))
        child = [-1] * n_tasks
        child[start:end] = parent1[start:end]
        # 从 parent2 填充剩余
        fill_idx = 0
        for gene in parent2:
            if gene not in child:
                while fill_idx < n_tasks and child[fill_idx] != -1:
                    fill_idx += 1
                if fill_idx < n_tasks:
                    child[fill_idx] = gene
        return child

    @staticmethod
    def _mutate(schedule: List[int]) -> List[int]:
        """交换变异"""
        if len(schedule) < 2:
            return list(schedule)
        i, j = random.sample(range(len(schedule)), 2)
        result = list(schedule)
        result[i], result[j] = result[j], result[i]
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回最佳调度"""
        if not self.is_trained or self._best_schedule is None:
            raise RuntimeError("Model not trained")
        return np.array(self._best_schedule)

    def get_result(self) -> ScheduleResult:
        if self._best_schedule is None:
            raise RuntimeError("Model not trained")
        makespan, total_value = self._compute_metrics(self._best_schedule)
        # 计算开始时间
        schedule = []
        current_time = 0.0
        for idx in self._best_schedule:
            task = self._tasks[idx]
            schedule.append((task.task_id, current_time))
            current_time += task.duration
        return ScheduleResult(
            schedule=schedule,
            makespan=makespan,
            total_value=total_value,
            fitness=self._best_fitness,
            method="genetic",
            iterations=self.generations,
            feasible=self._is_feasible(self._best_schedule),
        )


class SimulatedAnnealingScheduler(AIAlgorithmComponent):
    """模拟退火调度器"""

    def __init__(self, name: str = "sa_scheduler",
                 initial_temp: float = 100.0, cooling_rate: float = 0.95,
                 min_temp: float = 0.1, iterations_per_temp: int = 10):
        super().__init__(name, initial_temp=initial_temp,
                         cooling_rate=cooling_rate, min_temp=min_temp)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
        self._tasks: List[Task] = []
        self._best_schedule: Optional[List[int]] = None
        self._best_fitness: float = 0.0
        self._iterations = 0

    @property
    def component_type(self) -> str:
        return "scheduling"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SimulatedAnnealingScheduler":
        start = self._start_training()
        try:
            if isinstance(X, list) and X and isinstance(X[0], Task):
                self._tasks = X
            else:
                raise ValueError("X should be list of Task objects")

            n_tasks = len(self._tasks)
            current = list(range(n_tasks))
            random.shuffle(current)
            current_fitness = self._evaluate(current)

            self._best_schedule = list(current)
            self._best_fitness = current_fitness

            temp = self.initial_temp
            self._iterations = 0
            while temp > self.min_temp:
                for _ in range(self.iterations_per_temp):
                    # 生成邻居
                    if n_tasks >= 2:
                        i, j = random.sample(range(n_tasks), 2)
                        neighbor = list(current)
                        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    else:
                        neighbor = list(current)

                    neighbor_fitness = self._evaluate(neighbor)
                    delta = neighbor_fitness - current_fitness

                    if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-6)):
                        current = neighbor
                        current_fitness = neighbor_fitness
                        if current_fitness > self._best_fitness:
                            self._best_fitness = current_fitness
                            self._best_schedule = list(current)

                    self._iterations += 1
                temp *= self.cooling_rate

            self.metrics.sample_count = n_tasks
            self.metrics.custom["best_fitness"] = self._best_fitness
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def _evaluate(self, schedule: List[int]) -> float:
        if not self._is_feasible(schedule):
            return -1000.0
        makespan, total_value = self._compute_metrics(schedule)
        return total_value / (makespan + 1e-6)

    def _is_feasible(self, schedule: List[int]) -> bool:
        scheduled_set = set()
        for idx in schedule:
            task = self._tasks[idx]
            for dep in task.dependencies:
                if dep not in [self._tasks[i].task_id for i in scheduled_set]:
                    return False
            scheduled_set.add(idx)
        return True

    def _compute_metrics(self, schedule: List[int]) -> Tuple[float, float]:
        current_time = 0.0
        total_value = 0.0
        for idx in schedule:
            task = self._tasks[idx]
            current_time += task.duration
            total_value += task.value
        return current_time, total_value

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self._best_schedule is None:
            raise RuntimeError("Model not trained")
        return np.array(self._best_schedule)

    def get_result(self) -> ScheduleResult:
        if self._best_schedule is None:
            raise RuntimeError("Model not trained")
        makespan, total_value = self._compute_metrics(self._best_schedule)
        schedule = []
        current_time = 0.0
        for idx in self._best_schedule:
            task = self._tasks[idx]
            schedule.append((task.task_id, current_time))
            current_time += task.duration
        return ScheduleResult(
            schedule=schedule,
            makespan=makespan,
            total_value=total_value,
            fitness=self._best_fitness,
            method="simulated_annealing",
            iterations=self._iterations,
            feasible=self._is_feasible(self._best_schedule),
        )


class GreedyScheduler(AIAlgorithmComponent):
    """贪心调度器（按优先级+价值/持续时间）"""

    def __init__(self, name: str = "greedy_scheduler"):
        super().__init__(name)
        self._tasks: List[Task] = []
        self._best_schedule: Optional[List[int]] = None

    @property
    def component_type(self) -> str:
        return "scheduling"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GreedyScheduler":
        start = self._start_training()
        try:
            if isinstance(X, list) and X and isinstance(X[0], Task):
                self._tasks = X
            else:
                raise ValueError("X should be list of Task objects")

            # 贪心：按优先级 + 价值/时间 排序
            self._best_schedule = self._greedy_schedule()
            self.metrics.sample_count = len(self._tasks)
            self._end_training(start)
            return self
        except Exception:
            self.status = AlgorithmStatus.ERROR
            raise

    def _greedy_schedule(self) -> List[int]:
        """贪心调度，考虑依赖"""
        scheduled = []
        scheduled_ids = set()
        remaining = list(range(len(self._tasks)))

        while remaining:
            # 找出依赖已满足的任务
            available = []
            for idx in remaining:
                task = self._tasks[idx]
                if all(dep in scheduled_ids for dep in task.dependencies):
                    available.append(idx)

            if not available:
                # 有循环依赖，强制选剩余的
                available = remaining

            # 按 价值/时间 * 优先级 排序
            available.sort(key=lambda i: -(self._tasks[i].value /
                                           max(self._tasks[i].duration, 1e-6) *
                                           self._tasks[i].priority))
            chosen = available[0]
            scheduled.append(chosen)
            scheduled_ids.add(self._tasks[chosen].task_id)
            remaining.remove(chosen)

        return scheduled

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self._best_schedule is None:
            raise RuntimeError("Model not trained")
        return np.array(self._best_schedule)

    def get_result(self) -> ScheduleResult:
        if self._best_schedule is None:
            raise RuntimeError("Model not trained")
        current_time = 0.0
        total_value = 0.0
        schedule = []
        for idx in self._best_schedule:
            task = self._tasks[idx]
            schedule.append((task.task_id, current_time))
            current_time += task.duration
            total_value += task.value
        return ScheduleResult(
            schedule=schedule,
            makespan=current_time,
            total_value=total_value,
            fitness=total_value / (current_time + 1e-6),
            method="greedy",
            iterations=1,
            feasible=True,
        )


class SmartScheduler:
    """智能调度统一接口"""

    def __init__(self):
        self._schedulers: Dict[str, AIAlgorithmComponent] = {}

    def schedule(self, tasks: List[Task], method: str = "auto",
                  **kwargs) -> ScheduleResult:
        """调度

        Args:
            tasks: 任务列表
            method: 调度方法 ("genetic", "sa", "greedy", "auto")

        Returns:
            ScheduleResult: 调度结果
        """
        if method == "auto":
            if len(tasks) > 20:
                method = "sa"  # 大规模用模拟退火
            else:
                method = "greedy"  # 小规模用贪心

        if method == "genetic":
            scheduler = GeneticScheduler(**kwargs)
        elif method == "sa":
            scheduler = SimulatedAnnealingScheduler(**kwargs)
        elif method == "greedy":
            scheduler = GreedyScheduler()
        else:
            raise ValueError(f"Unknown method: {method}")

        scheduler.fit(tasks)
        self._schedulers[method] = scheduler
        return scheduler.get_result()

    def compare_methods(self, tasks: List[Task]) -> Dict[str, Any]:
        """对比所有调度方法"""
        results = {}
        for method in ["greedy", "sa", "genetic"]:
            try:
                scheduler_cls = {"greedy": GreedyScheduler,
                                "sa": SimulatedAnnealingScheduler,
                                "genetic": GeneticScheduler}[method]
                scheduler = scheduler_cls()
                scheduler.fit(tasks)
                result = scheduler.get_result()
                results[method] = {
                    "makespan": result.makespan,
                    "total_value": result.total_value,
                    "fitness": result.fitness,
                    "training_time": scheduler.metrics.training_time,
                }
            except Exception as e:
                results[method] = {"error": str(e)}
        return results
