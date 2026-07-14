import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class Gene:
    def __init__(self, name: str, value: Any, min_value: Any = None,
                 max_value: Any = None, gene_type: str = "float"):
        self.name = name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.gene_type = gene_type

    def mutate(self, mutation_rate: float = 0.1) -> "Gene":
        if np.random.random() > mutation_rate:
            return Gene(self.name, self.value, self.min_value, self.max_value, self.gene_type)
        
        if self.gene_type == "float":
            delta = np.random.normal(0, (self.max_value - self.min_value) * 0.1) if self.max_value else np.random.normal(0, 0.1)
            new_value = self.value + delta
            if self.min_value is not None:
                new_value = max(self.min_value, new_value)
            if self.max_value is not None:
                new_value = min(self.max_value, new_value)
            return Gene(self.name, new_value, self.min_value, self.max_value, self.gene_type)
        
        elif self.gene_type == "integer":
            delta = np.random.randint(-2, 3)
            new_value = self.value + delta
            if self.min_value is not None:
                new_value = max(self.min_value, new_value)
            if self.max_value is not None:
                new_value = min(self.max_value, new_value)
            return Gene(self.name, new_value, self.min_value, self.max_value, self.gene_type)
        
        elif self.gene_type == "categorical":
            categories = self.max_value if isinstance(self.max_value, list) else []
            if categories:
                new_value = np.random.choice(categories)
                return Gene(self.name, new_value, self.min_value, categories, self.gene_type)
        
        return Gene(self.name, self.value, self.min_value, self.max_value, self.gene_type)

    def crossover(self, other: "Gene") -> Tuple["Gene", "Gene"]:
        if self.gene_type == "float":
            alpha = np.random.random()
            new_value1 = alpha * self.value + (1 - alpha) * other.value
            new_value2 = (1 - alpha) * self.value + alpha * other.value
            return (
                Gene(self.name, new_value1, self.min_value, self.max_value, self.gene_type),
                Gene(self.name, new_value2, self.min_value, self.max_value, self.gene_type)
            )
        
        elif self.gene_type == "integer":
            if np.random.random() < 0.5:
                return Gene(self.name, self.value, self.min_value, self.max_value, self.gene_type), \
                       Gene(self.name, other.value, other.min_value, other.max_value, other.gene_type)
            else:
                return Gene(self.name, other.value, other.min_value, other.max_value, other.gene_type), \
                       Gene(self.name, self.value, self.min_value, self.max_value, self.gene_type)
        
        return Gene(self.name, self.value, self.min_value, self.max_value, self.gene_type), \
               Gene(self.name, other.value, other.min_value, other.max_value, other.gene_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "gene_type": self.gene_type
        }


class Genome:
    def __init__(self, genes: List[Gene]):
        self.genes = genes
        self.fitness: float = 0.0
        self.age: int = 0
        self.id = np.random.randint(1000000)

    def get_gene(self, name: str) -> Optional[Gene]:
        for gene in self.genes:
            if gene.name == name:
                return gene
        return None

    def set_gene(self, name: str, value: Any):
        for gene in self.genes:
            if gene.name == name:
                gene.value = value
                break

    def mutate(self, mutation_rate: float = 0.1):
        new_genes = [gene.mutate(mutation_rate) for gene in self.genes]
        return Genome(new_genes)

    def crossover(self, other: "Genome") -> Tuple["Genome", "Genome"]:
        new_genes1 = []
        new_genes2 = []
        
        for gene1, gene2 in zip(self.genes, other.genes):
            if gene1.name == gene2.name:
                g1, g2 = gene1.crossover(gene2)
                new_genes1.append(g1)
                new_genes2.append(g2)
            else:
                new_genes1.append(gene1)
                new_genes2.append(gene2)
        
        return Genome(new_genes1), Genome(new_genes2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fitness": self.fitness,
            "age": self.age,
            "genes": [gene.to_dict() for gene in self.genes]
        }

    def to_parameter_dict(self) -> Dict[str, Any]:
        return {gene.name: gene.value for gene in self.genes}


class Population:
    def __init__(self, genomes: List[Genome] = None):
        self.genomes = genomes or []

    def add_genome(self, genome: Genome):
        self.genomes.append(genome)

    def sort_by_fitness(self, reverse: bool = True):
        self.genomes.sort(key=lambda g: g.fitness, reverse=reverse)

    def get_best(self, n: int = 1) -> List[Genome]:
        self.sort_by_fitness()
        return self.genomes[:n]

    def get_avg_fitness(self) -> float:
        if not self.genomes:
            return 0.0
        return float(np.mean([g.fitness for g in self.genomes]))

    def get_max_fitness(self) -> float:
        if not self.genomes:
            return 0.0
        return max(g.fitness for g in self.genomes)

    def get_min_fitness(self) -> float:
        if not self.genomes:
            return 0.0
        return min(g.fitness for g in self.genomes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": len(self.genomes),
            "avg_fitness": self.get_avg_fitness(),
            "max_fitness": self.get_max_fitness(),
            "min_fitness": self.get_min_fitness(),
            "genomes": [g.to_dict() for g in self.genomes]
        }


class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    ELITISM = "elitism"
    RANK = "rank"


class CrossoverMethod(Enum):
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"


class MutationMethod(Enum):
    RANDOM = "random"
    GAUSSIAN = "gaussian"
    BIT_FLIP = "bit_flip"
    POLYNOMIAL = "polynomial"


class FitnessFunction(Callable[[Dict[str, Any]], float]):
    pass


class EvolutionResult:
    def __init__(self, generation: int):
        self.generation = generation
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = 0.0
        self.avg_fitness: float = 0.0
        self.std_fitness: float = 0.0
        self.population_size: int = 0
        self.converged: bool = False
        self.diversity: float = 0.0
        self.timestamp = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "std_fitness": self.std_fitness,
            "population_size": self.population_size,
            "converged": self.converged,
            "diversity": self.diversity,
            "best_genome": self.best_genome.to_dict() if self.best_genome else None,
            "timestamp": self.timestamp
        }


class GeneticAlgorithm:
    def __init__(self, fitness_function: FitnessFunction,
                 gene_templates: List[Dict[str, Any]],
                 population_size: int = 50,
                 max_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
                 crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM,
                 mutation_method: MutationMethod = MutationMethod.GAUSSIAN):
        
        self.fitness_function = fitness_function
        self.gene_templates = gene_templates
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        
        self.population: Population = Population()
        self.current_generation = 0
        self.evolution_history: deque = deque(maxlen=200)

    def initialize_population(self):
        self.population = Population()
        for _ in range(self.population_size):
            genes = []
            for template in self.gene_templates:
                gene_type = template.get("type", "float")
                min_val = template.get("min")
                max_val = template.get("max")
                
                if gene_type == "float":
                    value = np.random.uniform(min_val, max_val) if min_val is not None else np.random.random()
                elif gene_type == "integer":
                    value = np.random.randint(min_val, max_val + 1) if min_val is not None else np.random.randint(0, 10)
                elif gene_type == "categorical":
                    categories = template.get("categories", [])
                    value = np.random.choice(categories) if categories else ""
                else:
                    value = np.random.random()
                
                genes.append(Gene(template["name"], value, min_val, max_val, gene_type))
            
            self.population.add_genome(Genome(genes))

    def evaluate_population(self):
        for genome in self.population.genomes:
            params = genome.to_parameter_dict()
            genome.fitness = self.fitness_function(params)

    def select(self, population: Population, num_selected: int) -> List[Genome]:
        selected = []
        
        if self.selection_method == SelectionMethod.TOURNAMENT:
            tournament_size = 3
            for _ in range(num_selected):
                participants = np.random.choice(population.genomes, tournament_size, replace=False)
                winner = max(participants, key=lambda g: g.fitness)
                selected.append(winner)
        
        elif self.selection_method == SelectionMethod.ROULETTE:
            total_fitness = sum(g.fitness for g in population.genomes)
            if total_fitness == 0:
                return np.random.choice(population.genomes, num_selected, replace=False)
            
            probabilities = [g.fitness / total_fitness for g in population.genomes]
            selected = np.random.choice(population.genomes, num_selected, p=probabilities, replace=True)
        
        elif self.selection_method == SelectionMethod.ELITISM:
            population.sort_by_fitness()
            selected = population.genomes[:num_selected]
        
        elif self.selection_method == SelectionMethod.RANK:
            population.sort_by_fitness()
            ranks = np.arange(1, len(population.genomes) + 1)
            total_rank = sum(ranks)
            probabilities = ranks / total_rank
            selected = np.random.choice(population.genomes, num_selected, p=probabilities, replace=True)
        
        return selected

    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        if np.random.random() > self.crossover_rate:
            return parent1, parent2
        
        return parent1.crossover(parent2)

    def mutate(self, genome: Genome) -> Genome:
        return genome.mutate(self.mutation_rate)

    def evolve(self) -> EvolutionResult:
        if not self.population.genomes:
            self.initialize_population()
        
        self.evaluate_population()
        
        result = EvolutionResult(self.current_generation)
        result.best_genome = self.population.get_best(1)[0]
        result.best_fitness = result.best_genome.fitness
        result.avg_fitness = self.population.get_avg_fitness()
        result.population_size = len(self.population.genomes)
        
        fitness_values = [g.fitness for g in self.population.genomes]
        result.std_fitness = float(np.std(fitness_values))
        result.diversity = float(np.std(fitness_values) / (result.avg_fitness + 1e-10))
        
        if result.std_fitness < 1e-5 or self.current_generation >= self.max_generations:
            result.converged = True
        
        self.evolution_history.append(result)
        
        if result.converged:
            return result
        
        selected = self.select(self.population, self.population_size)
        
        new_population = Population()
        
        for i in range(0, len(selected) - 1, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1]
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.add_genome(child1)
            new_population.add_genome(child2)
        
        if len(selected) % 2 == 1:
            last = selected[-1]
            mutated = self.mutate(last)
            new_population.add_genome(mutated)
        
        self.population = new_population
        self.current_generation += 1
        
        return result

    def run(self, callback: Callable[[EvolutionResult], bool] = None) -> EvolutionResult:
        self.initialize_population()
        
        for _ in range(self.max_generations):
            result = self.evolve()
            
            if callback and not callback(result):
                break
            
            if result.converged:
                break
        
        return self.evolution_history[-1] if self.evolution_history else result

    def get_evolution_summary(self) -> Dict[str, Any]:
        if not self.evolution_history:
            return {"total_generations": 0}
        
        results = list(self.evolution_history)
        
        return {
            "total_generations": len(results),
            "current_generation": self.current_generation,
            "population_size": self.population_size,
            "final_best_fitness": results[-1].best_fitness,
            "final_avg_fitness": results[-1].avg_fitness,
            "converged": results[-1].converged,
            "best_genome": results[-1].best_genome.to_dict() if results[-1].best_genome else None,
            "fitness_trend": [r.best_fitness for r in results],
            "avg_fitness_trend": [r.avg_fitness for r in results],
            "diversity_trend": [r.diversity for r in results]
        }