import numpy as np
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple
from .genetic_algorithm import GeneticAlgorithm, FitnessFunction, EvolutionResult
from .gene_pool import GenePool
from .evolution_controller import EvolutionController, EvolutionConfig
from .parameter_optimizer import ParameterOptimizer, ParameterSpace
from .structural_optimizer import StructuralOptimizer, StructureEncoding


class EvolutionOrchestrator:
    def __init__(self):
        self.genetic_algorithm: Optional[GeneticAlgorithm] = None
        self.gene_pool = GenePool()
        self.controller = EvolutionController()
        self.parameter_optimizer = ParameterOptimizer()
        self.structural_optimizer = StructuralOptimizer()
        self._evolution_tasks: Dict[str, Dict[str, Any]] = {}
        self._orchestration_history: deque = deque(maxlen=200)

    def setup_genetic_algorithm(self, fitness_function: FitnessFunction,
                               gene_templates: List[Dict[str, Any]],
                               config: EvolutionConfig = None):
        config = config or EvolutionConfig()
        
        self.genetic_algorithm = GeneticAlgorithm(
            fitness_function=fitness_function,
            gene_templates=gene_templates,
            population_size=config.population_size,
            max_generations=config.max_generations,
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate
        )
        
        self.controller = EvolutionController(config)

    def run_evolution(self, task_id: str = "default") -> Dict[str, Any]:
        if not self.genetic_algorithm:
            return {"success": False, "error": "Genetic algorithm not configured"}
        
        self._evolution_tasks[task_id] = {
            "status": "running",
            "start_time": np.random.randint(1000000)
        }
        
        def callback(result: EvolutionResult) -> bool:
            self.controller.advance_generation(
                fitness_values=[g.fitness for g in self.genetic_algorithm.population.genomes],
                diversity=result.diversity
            )
            
            control_signal = self.controller.get_control_signal()
            if control_signal["should_terminate"]:
                return False
            
            return True
        
        final_result = self.genetic_algorithm.run(callback=callback)
        
        self._evolution_tasks[task_id]["status"] = "completed"
        self._evolution_tasks[task_id]["end_time"] = np.random.randint(1000000)
        
        self._orchestration_history.append({
            "task_id": task_id,
            "result": final_result.to_dict(),
            "controller_summary": self.controller.get_monitor_summary(),
            "timestamp": np.random.randint(1000000)
        })
        
        return {
            "success": True,
            "task_id": task_id,
            "evolution_result": final_result.to_dict(),
            "controller_summary": self.controller.get_monitor_summary(),
            "genetic_summary": self.genetic_algorithm.get_evolution_summary()
        }

    def run_parameter_optimization(self, space_id: str,
                                  objective_function: Callable[[Dict[str, float]], float]) -> Dict[str, Any]:
        result = self.parameter_optimizer.optimize(space_id, objective_function)
        
        self._orchestration_history.append({
            "task_id": f"param_opt_{space_id}",
            "type": "parameter_optimization",
            "result": result.to_dict(),
            "timestamp": np.random.randint(1000000)
        })
        
        return {
            "success": result.success,
            "space_id": space_id,
            "result": result.to_dict(),
            "optimizer_summary": self.parameter_optimizer.get_optimization_summary()
        }

    def run_structural_optimization(self, initial_structure: StructureEncoding,
                                   fitness_function: Callable[[StructureEncoding], float]) -> Dict[str, Any]:
        result = self.structural_optimizer.evolve(initial_structure, fitness_function)
        
        self._orchestration_history.append({
            "task_id": "structural_opt",
            "type": "structural_optimization",
            "result": result.to_dict(),
            "timestamp": np.random.randint(1000000)
        })
        
        return {
            "success": True,
            "result": result.to_dict(),
            "optimizer_summary": self.structural_optimizer.get_evolution_summary()
        }

    def run_full_optimization(self, fitness_function: FitnessFunction,
                             gene_templates: List[Dict[str, Any]],
                             param_space_id: str = None) -> Dict[str, Any]:
        self.setup_genetic_algorithm(fitness_function, gene_templates)
        
        ga_result = self.run_evolution("full_optimization")
        
        if param_space_id and param_space_id in self.parameter_optimizer.spaces:
            param_result = self.run_parameter_optimization(
                param_space_id,
                lambda params: fitness_function(params)
            )
        else:
            param_result = None
        
        return {
            "genetic_algorithm": ga_result,
            "parameter_optimization": param_result,
            "overall_summary": self.get_overview()
        }

    def create_parameter_space(self, space_id: str, params: Dict[str, Dict[str, float]]) -> ParameterSpace:
        space = self.parameter_optimizer.create_space(space_id)
        
        for name, config in params.items():
            space.create_parameter(
                name=name,
                value=config.get("value", 0.5),
                min_value=config.get("min", 0.0),
                max_value=config.get("max", 1.0),
                step=config.get("step", 0.01),
                adaptive=config.get("adaptive", True)
            )
        
        return space

    def register_gene_template(self, name: str, gene_type: str = "float",
                              min_value: Any = None, max_value: Any = None,
                              description: str = "", category: str = "default"):
        self.gene_pool.library.create_template(
            name=name,
            gene_type=gene_type,
            min_value=min_value,
            max_value=max_value,
            description=description,
            category=category
        )

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._evolution_tasks.get(task_id)

    def get_overview(self) -> Dict[str, Any]:
        return {
            "genetic_algorithm": self.genetic_algorithm.get_evolution_summary() if self.genetic_algorithm else {"status": "not_configured"},
            "gene_pool": self.gene_pool.get_pool_summary(),
            "controller": self.controller.get_monitor_summary(),
            "parameter_optimizer": self.parameter_optimizer.get_optimization_summary(),
            "structural_optimizer": self.structural_optimizer.get_evolution_summary(),
            "active_tasks": len([t for t in self._evolution_tasks.values() if t.get("status") == "running"]),
            "total_tasks": len(self._evolution_tasks),
            "orchestration_history_count": len(self._orchestration_history)
        }

    def get_recent_evolution(self, limit: int = 10) -> List[Dict[str, Any]]:
        recent = list(self._orchestration_history)[-limit:]
        return recent[::-1]

    def reset(self):
        self.genetic_algorithm = None
        self.gene_pool = GenePool()
        self.controller = EvolutionController()
        self._evolution_tasks = {}
        self._orchestration_history = deque(maxlen=200)