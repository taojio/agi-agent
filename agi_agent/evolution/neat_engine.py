import numpy as np
import neat
import os


class EvolutionEngine:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "neat_config.txt")
        self.config_path = config_path
        self.population = None
        self.best_genome = None
        self.evolve_step = 0
        self.fitness_history = []
        self.species_count_history = []
        self._config = None

    def _load_config(self):
        if self._config is None:
            self._config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                self.config_path
            )
        return self._config

    def init_population(self):
        config = self._load_config()
        self.population = neat.Population(config)
        self.population.add_reporter(neat.StdOutReporter(False))
        
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)

    def fitness_func(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            test_input = np.random.rand(16)
            output = net.activate(test_input)
            fe = np.mean(np.square(output - test_input))
            
            genome_size = genome.size()
            if isinstance(genome_size, tuple):
                genome_size = sum(genome_size)
            
            genome.fitness = 1 / (fe + genome_size / 1000)

    def evolve(self, generations=5):
        if self.population is None:
            self.init_population()
        
        if self.population is not None:
            winner = self.population.run(self.fitness_func, n=generations)
            self.best_genome = winner
            self.evolve_step += 1
            
            fitnesses = [c.fitness for c in self.population.population.values() if c.fitness is not None]
            if fitnesses:
                self.fitness_history.append({
                    "step": self.evolve_step,
                    "mean": np.mean(fitnesses),
                    "max": np.max(fitnesses),
                    "min": np.min(fitnesses)
                })
            
            species_count = len(self.population.species.species)
            self.species_count_history.append(species_count)
            
            return winner
        
        return None

    def get_evolution_stats(self):
        if not self.fitness_history:
            return {"step": 0, "fitness": {"mean": 0, "max": 0, "min": 0}, "species": 0}
        
        latest = self.fitness_history[-1]
        return {
            "step": self.evolve_step,
            "fitness": latest,
            "species": self.species_count_history[-1] if self.species_count_history else 0
        }