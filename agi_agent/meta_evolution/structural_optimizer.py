import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class StructuralMutationType(Enum):
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    MODIFY_NODE = "modify_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    MODIFY_EDGE = "modify_edge"
    REORDER = "reorder"
    DUPLICATE = "duplicate"


class StructureEncoding:
    def __init__(self, nodes: List[Dict[str, Any]] = None, edges: List[Dict[str, Any]] = None):
        self.nodes = nodes or []
        self.edges = edges or []
        self.fitness: float = 0.0

    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "properties": properties or {}
        })

    def remove_node(self, node_id: str):
        self.nodes = [n for n in self.nodes if n["id"] != node_id]
        self.edges = [e for e in self.edges if e.get("source") != node_id and e.get("target") != node_id]

    def add_edge(self, source: str, target: str, edge_type: str = "default"):
        self.edges.append({
            "source": source,
            "target": target,
            "type": edge_type
        })

    def remove_edge(self, source: str, target: str):
        self.edges = [e for e in self.edges if not (e.get("source") == source and e.get("target") == target)]

    def mutate(self, mutation_type: StructuralMutationType, params: Dict[str, Any] = None) -> "StructureEncoding":
        new_encoding = StructureEncoding(
            nodes=[n.copy() for n in self.nodes],
            edges=[e.copy() for e in self.edges]
        )
        
        params = params or {}
        
        if mutation_type == StructuralMutationType.ADD_NODE:
            new_node_id = f"node_{len(self.nodes)}_{np.random.randint(1000)}"
            new_encoding.add_node(new_node_id, params.get("type", "default"))
        
        elif mutation_type == StructuralMutationType.REMOVE_NODE:
            if self.nodes:
                node_to_remove = np.random.choice(self.nodes)
                new_encoding.remove_node(node_to_remove["id"])
        
        elif mutation_type == StructuralMutationType.MODIFY_NODE:
            if self.nodes:
                node_to_modify = np.random.choice(new_encoding.nodes)
                node_to_modify["properties"] = {**node_to_modify.get("properties", {}), **params.get("properties", {})}
        
        elif mutation_type == StructuralMutationType.ADD_EDGE:
            if len(self.nodes) >= 2:
                source, target = np.random.choice(self.nodes, 2, replace=False)
                new_encoding.add_edge(source["id"], target["id"], params.get("type", "default"))
        
        elif mutation_type == StructuralMutationType.REMOVE_EDGE:
            if self.edges:
                edge_to_remove = np.random.choice(self.edges)
                new_encoding.remove_edge(edge_to_remove["source"], edge_to_remove["target"])
        
        return new_encoding

    def crossover(self, other: "StructureEncoding") -> Tuple["StructureEncoding", "StructureEncoding"]:
        new1 = StructureEncoding()
        new2 = StructureEncoding()
        
        for node in self.nodes:
            if np.random.random() < 0.5:
                new1.add_node(node["id"], node["type"], node.get("properties"))
            else:
                new2.add_node(node["id"], node["type"], node.get("properties"))
        
        for node in other.nodes:
            if node["id"] not in [n["id"] for n in new1.nodes]:
                if np.random.random() < 0.5:
                    new1.add_node(node["id"], node["type"], node.get("properties"))
                else:
                    new2.add_node(node["id"], node["type"], node.get("properties"))
        
        for edge in self.edges:
            if edge["source"] in [n["id"] for n in new1.nodes] and edge["target"] in [n["id"] for n in new1.nodes]:
                new1.add_edge(edge["source"], edge["target"], edge.get("type"))
        
        for edge in other.edges:
            if edge["source"] in [n["id"] for n in new2.nodes] and edge["target"] in [n["id"] for n in new2.nodes]:
                new2.add_edge(edge["source"], edge["target"], edge.get("type"))
        
        return new1, new2

    def get_size(self) -> int:
        return len(self.nodes) + len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "fitness": self.fitness,
            "size": self.get_size(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges)
        }


class StructuralMutation:
    def __init__(self, mutation_type: StructuralMutationType, probability: float = 0.1):
        self.mutation_type = mutation_type
        self.probability = probability
        self.applied_count: int = 0
        self.success_count: int = 0

    def apply(self, encoding: StructureEncoding) -> StructureEncoding:
        if np.random.random() > self.probability:
            return encoding
        
        self.applied_count += 1
        return encoding.mutate(self.mutation_type)

    def record_success(self):
        self.success_count += 1

    def get_effectiveness(self) -> float:
        if self.applied_count == 0:
            return 0.5
        return self.success_count / self.applied_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_type": self.mutation_type.value,
            "probability": self.probability,
            "applied_count": self.applied_count,
            "success_count": self.success_count,
            "effectiveness": self.get_effectiveness()
        }


class StructureEvolutionResult:
    def __init__(self, generation: int):
        self.generation = generation
        self.best_structure: Optional[StructureEncoding] = None
        self.best_fitness: float = 0.0
        self.avg_fitness: float = 0.0
        self.structures_evaluated: int = 0
        self.mutations_applied: int = 0
        self.converged: bool = False
        self.timestamp = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "structures_evaluated": self.structures_evaluated,
            "mutations_applied": self.mutations_applied,
            "converged": self.converged,
            "best_structure": self.best_structure.to_dict() if self.best_structure else None,
            "timestamp": self.timestamp
        }


class StructuralOptimizer:
    def __init__(self):
        self.mutations: List[StructuralMutation] = [
            StructuralMutation(StructuralMutationType.ADD_NODE, 0.2),
            StructuralMutation(StructuralMutationType.REMOVE_NODE, 0.1),
            StructuralMutation(StructuralMutationType.MODIFY_NODE, 0.2),
            StructuralMutation(StructuralMutationType.ADD_EDGE, 0.2),
            StructuralMutation(StructuralMutationType.REMOVE_EDGE, 0.1),
            StructuralMutation(StructuralMutationType.MODIFY_EDGE, 0.1),
            StructuralMutation(StructuralMutationType.REORDER, 0.05),
            StructuralMutation(StructuralMutationType.DUPLICATE, 0.05),
        ]
        self.evolution_history: deque = deque(maxlen=200)

    def evolve(self, initial_structure: StructureEncoding,
              fitness_function: Callable[[StructureEncoding], float],
              population_size: int = 20, max_generations: int = 50) -> StructureEvolutionResult:
        
        population = [initial_structure]
        
        for _ in range(population_size - 1):
            new_structure = initial_structure.mutate(
                np.random.choice([m.mutation_type for m in self.mutations])
            )
            population.append(new_structure)
        
        for gen in range(max_generations):
            for structure in population:
                structure.fitness = fitness_function(structure)
            
            population.sort(key=lambda s: s.fitness, reverse=True)
            
            result = StructureEvolutionResult(gen)
            result.best_structure = population[0]
            result.best_fitness = population[0].fitness
            result.avg_fitness = float(np.mean([s.fitness for s in population]))
            result.structures_evaluated = len(population)
            
            self.evolution_history.append(result)
            
            if gen > 0 and abs(result.best_fitness - self.evolution_history[-2].best_fitness) < 1e-6:
                result.converged = True
                return result
            
            new_population = [population[0]]
            
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(population[:5], 2, replace=False)
                child1, child2 = parent1.crossover(parent2)
                
                for mutation in self.mutations:
                    child1 = mutation.apply(child1)
                    child2 = mutation.apply(child2)
                
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)
            
            population = new_population
        
        return self.evolution_history[-1]

    def optimize_structure(self, initial_structure: StructureEncoding,
                          fitness_function: Callable[[StructureEncoding], float],
                          max_iterations: int = 100) -> StructureEvolutionResult:
        
        current_structure = initial_structure
        current_fitness = fitness_function(current_structure)
        
        best_structure = current_structure
        best_fitness = current_fitness
        
        for i in range(max_iterations):
            mutation = np.random.choice(self.mutations)
            new_structure = mutation.apply(current_structure)
            new_fitness = fitness_function(new_structure)
            
            if new_fitness > best_fitness:
                best_structure = new_structure
                best_fitness = new_fitness
                mutation.record_success()
            
            if new_fitness > current_fitness:
                current_structure = new_structure
                current_fitness = new_fitness
        
        result = StructureEvolutionResult(max_iterations)
        result.best_structure = best_structure
        result.best_fitness = best_fitness
        result.converged = True
        
        return result

    def get_evolution_summary(self) -> Dict[str, Any]:
        if not self.evolution_history:
            return {"total_generations": 0}
        
        results = list(self.evolution_history)
        
        return {
            "total_generations": len(results),
            "final_best_fitness": results[-1].best_fitness,
            "final_avg_fitness": results[-1].avg_fitness,
            "converged": results[-1].converged,
            "mutation_stats": [m.to_dict() for m in self.mutations],
            "fitness_trend": [r.best_fitness for r in results]
        }