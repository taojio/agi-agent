import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class EvolutionPhase(Enum):
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    CONSOLIDATION = "consolidation"
    ADAPTATION = "adaptation"
    TERMINATION = "termination"


class EvolutionStrategy(Enum):
    STEADY_STATE = "steady_state"
    GENERATIONAL = "generational"
    COEVOLUTION = "coevolution"
    ADAPTIVE = "adaptive"
    BAYESIAN = "bayesian"


class EvolutionConfig:
    def __init__(self):
        self.population_size: int = 50
        self.max_generations: int = 100
        self.mutation_rate: float = 0.1
        self.crossover_rate: float = 0.7
        self.elitism_rate: float = 0.1
        self.diversity_threshold: float = 0.1
        self.convergence_threshold: float = 1e-5
        self.strategy: EvolutionStrategy = EvolutionStrategy.GENERATIONAL
        self.phase_duration: Dict[EvolutionPhase, int] = {
            EvolutionPhase.INITIALIZATION: 1,
            EvolutionPhase.EXPLORATION: 20,
            EvolutionPhase.EXPLOITATION: 50,
            EvolutionPhase.CONSOLIDATION: 20,
            EvolutionPhase.ADAPTATION: 10,
            EvolutionPhase.TERMINATION: 1,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism_rate": self.elitism_rate,
            "diversity_threshold": self.diversity_threshold,
            "convergence_threshold": self.convergence_threshold,
            "strategy": self.strategy.value,
            "phase_duration": {phase.value: duration for phase, duration in self.phase_duration.items()}
        }


class EvolutionMonitor:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.events: List[Dict[str, Any]] = []
        self.current_phase: EvolutionPhase = EvolutionPhase.INITIALIZATION

    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def record_event(self, event_type: str, details: Dict[str, Any] = None):
        self.events.append({
            "event_type": event_type,
            "details": details or {},
            "phase": self.current_phase.value,
            "timestamp": np.random.randint(1000000)
        })

    def set_phase(self, phase: EvolutionPhase):
        self.current_phase = phase
        self.record_event("phase_change", {"phase": phase.value})

    def get_metric_summary(self) -> Dict[str, Any]:
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "latest": values[-1],
                    "count": len(values)
                }
        return summary

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.events[-limit:]

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        anomalies = []
        
        if "fitness" in self.metrics:
            fitness_values = self.metrics["fitness"]
            if len(fitness_values) > 5:
                recent = fitness_values[-5:]
                if np.std(recent) < 1e-6:
                    anomalies.append({
                        "type": "stagnation",
                        "metric": "fitness",
                        "details": "Fitness has stagnated"
                    })
        
        if "diversity" in self.metrics:
            diversity_values = self.metrics["diversity"]
            if diversity_values and diversity_values[-1] < 0.01:
                anomalies.append({
                    "type": "low_diversity",
                    "metric": "diversity",
                    "details": f"Current diversity: {diversity_values[-1]}"
                })
        
        return anomalies


class EvolutionController:
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.monitor = EvolutionMonitor()
        self.current_generation = 0
        self.current_phase: EvolutionPhase = EvolutionPhase.INITIALIZATION
        self.adaptation_count: int = 0
        self._phase_generations = 0

    def get_current_phase(self) -> EvolutionPhase:
        return self.current_phase

    def advance_generation(self, fitness_values: List[float] = None,
                          diversity: float = 0.0):
        self.current_generation += 1
        self._phase_generations += 1
        
        if fitness_values:
            self.monitor.record_metric("fitness", float(np.mean(fitness_values)))
            self.monitor.record_metric("best_fitness", float(max(fitness_values)))
            self.monitor.record_metric("worst_fitness", float(min(fitness_values)))
        
        self.monitor.record_metric("diversity", diversity)
        self.monitor.record_metric("generation", self.current_generation)

        self._check_phase_transition()

    def _check_phase_transition(self):
        phase_duration = self.config.phase_duration.get(self.current_phase, 10)
        
        if self._phase_generations >= phase_duration:
            self._transition_to_next_phase()

    def _transition_to_next_phase(self):
        phases = list(EvolutionPhase)
        current_idx = phases.index(self.current_phase)
        
        if current_idx < len(phases) - 1:
            self.current_phase = phases[current_idx + 1]
            self._phase_generations = 0
            self.monitor.set_phase(self.current_phase)
            
            self.adapt_strategy()

    def adapt_strategy(self):
        if self.current_phase == EvolutionPhase.EXPLORATION:
            self.config.mutation_rate = 0.2
            self.config.crossover_rate = 0.8
        elif self.current_phase == EvolutionPhase.EXPLOITATION:
            self.config.mutation_rate = 0.05
            self.config.crossover_rate = 0.6
        elif self.current_phase == EvolutionPhase.CONSOLIDATION:
            self.config.mutation_rate = 0.02
            self.config.crossover_rate = 0.5
        elif self.current_phase == EvolutionPhase.ADAPTATION:
            self.config.mutation_rate = 0.15
            self.config.crossover_rate = 0.7

    def should_terminate(self) -> bool:
        if self.current_phase == EvolutionPhase.TERMINATION:
            return True
        
        if self.current_generation >= self.config.max_generations:
            return True
        
        anomalies = self.monitor.detect_anomalies()
        if any(a["type"] == "stagnation" for a in anomalies):
            self.adaptation_count += 1
            if self.adaptation_count >= 3:
                return True
        
        return False

    def get_control_signal(self) -> Dict[str, Any]:
        return {
            "current_generation": self.current_generation,
            "current_phase": self.current_phase.value,
            "phase_generations": self._phase_generations,
            "config": self.config.to_dict(),
            "adaptation_count": self.adaptation_count,
            "should_terminate": self.should_terminate()
        }

    def get_monitor_summary(self) -> Dict[str, Any]:
        return {
            "metrics": self.monitor.get_metric_summary(),
            "current_phase": self.current_phase.value,
            "events": self.monitor.get_recent_events(),
            "anomalies": self.monitor.detect_anomalies()
        }

    def reset(self):
        self.current_generation = 0
        self.current_phase = EvolutionPhase.INITIALIZATION
        self._phase_generations = 0
        self.adaptation_count = 0
        self.monitor = EvolutionMonitor()
        self.monitor.set_phase(EvolutionPhase.INITIALIZATION)