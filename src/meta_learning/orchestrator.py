"""
Meta-Learning Orchestrator — Evolutionary Architecture Search

Top-level system that evolves neural network architectures
toward enlightened functioning via multi-objective optimization.

The evolution discovers that dharma principles IMPROVE performance:
- No persistent self → Less computational waste on identity maintenance
- Low entropy → More efficient, predictable dynamics → Better flow
- Mindfulness → Better self-correction → Fewer errors
- Compassion → Better user modeling → More helpful responses

Enlightenment is computationally efficient.

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part X
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Callable
from copy import deepcopy

from .genome import ArchitecturalGenome
from .fitness import DharmaFitnessEvaluator


class MetaLearningOrchestrator:
    """
    Evolutionary orchestrator for neural architecture search
    guided by dharma fitness.

    Algorithm:
    1. Initialize population of random architectures
    2. Evaluate each on multi-objective dharma fitness
    3. Select top performers
    4. Crossover and mutate to create offspring
    5. Repeat until convergence

    The population evolves toward architectures that are simultaneously:
    - High-performing on task
    - Low in ego (no persistent self-representation)
    - Low in entropy (smooth, efficient processing)
    - High in awareness (meta-cognitive self-observation)
    - High in compassion (helpful, clear outputs)
    """

    def __init__(self,
                 population_size: int = 20,
                 max_generations: int = 100,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elite_fraction: float = 0.1,
                 model_builder: Optional[Callable] = None):
        """
        Args:
            population_size: Number of architectures per generation
            max_generations: Maximum evolution iterations
            mutation_rate: Probability of mutating each gene
            tournament_size: Selection tournament size
            elite_fraction: Fraction of population to keep unchanged
            model_builder: Function(genome) → nn.Module
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = max(1, int(population_size * elite_fraction))
        self.model_builder = model_builder

        self.evaluator = DharmaFitnessEvaluator()

        # History
        self.generation_history: List[Dict] = []
        self.best_genome: Optional[ArchitecturalGenome] = None
        self.best_fitness: float = -float('inf')

    def initialize_population(self) -> List[ArchitecturalGenome]:
        """Create initial population of diverse architectures."""
        population = []

        # First genome: default (Phase 1 baseline)
        population.append(ArchitecturalGenome())

        # Remaining: random variations
        for _ in range(self.population_size - 1):
            base = ArchitecturalGenome()
            mutant = base.mutate(mutation_rate=0.3)  # Higher initial diversity
            population.append(mutant)

        return population

    def evaluate_population(self,
                           population: List[ArchitecturalGenome],
                           test_data: List[Dict],
                           targets: List[torch.Tensor]) -> List[Dict]:
        """
        Evaluate fitness of entire population.

        Returns list of dicts with 'genome', 'scores', 'fitness' for each.
        """
        results = []

        for i, genome in enumerate(population):
            try:
                # Build model from genome
                if self.model_builder is not None:
                    model = self.model_builder(genome)
                else:
                    model = self._default_build(genome)

                # Evaluate
                scores = self.evaluator.evaluate(model, test_data, targets)
                fitness = self.evaluator.compute_aggregate_fitness(scores)

                results.append({
                    'genome': genome,
                    'scores': scores,
                    'fitness': fitness,
                    'index': i,
                })

                # Track best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_genome = deepcopy(genome)

            except Exception as e:
                # Failed architectures get zero fitness
                results.append({
                    'genome': genome,
                    'scores': {},
                    'fitness': 0.0,
                    'index': i,
                    'error': str(e),
                })

        return results

    def select_parents(self, results: List[Dict]) -> List[ArchitecturalGenome]:
        """
        Tournament selection: pick k random, return best.
        """
        parents = []
        n_parents = self.population_size - self.elite_count

        for _ in range(n_parents):
            # Random tournament
            tournament_indices = np.random.choice(
                len(results),
                size=min(self.tournament_size, len(results)),
                replace=False,
            )
            tournament = [results[i] for i in tournament_indices]

            # Select winner (highest fitness)
            winner = max(tournament, key=lambda x: x['fitness'])
            parents.append(deepcopy(winner['genome']))

        return parents

    def create_offspring(self,
                        parents: List[ArchitecturalGenome]) -> List[ArchitecturalGenome]:
        """
        Create offspring through crossover and mutation.
        """
        offspring = []

        while len(offspring) < len(parents):
            # Select two parents
            p1 = parents[np.random.randint(len(parents))]
            p2 = parents[np.random.randint(len(parents))]

            # Crossover
            child = p1.crossover(p2)

            # Mutation
            child = child.mutate(self.mutation_rate)

            offspring.append(child)

        return offspring

    def evolve(self,
               test_data: List[Dict],
               targets: List[torch.Tensor],
               verbose: bool = True) -> ArchitecturalGenome:
        """
        Run full evolutionary process.

        Args:
            test_data: Evaluation data
            targets: Evaluation targets
            verbose: Print progress

        Returns:
            Best genome found
        """
        # Initialize
        population = self.initialize_population()

        for gen in range(self.max_generations):
            # Evaluate
            results = self.evaluate_population(population, test_data, targets)

            # Sort by fitness
            results.sort(key=lambda x: x['fitness'], reverse=True)

            # Record history
            gen_info = {
                'generation': gen,
                'best_fitness': results[0]['fitness'],
                'mean_fitness': np.mean([r['fitness'] for r in results]),
                'best_scores': results[0]['scores'],
                'best_genome': results[0]['genome'].to_dict(),
            }
            self.generation_history.append(gen_info)

            if verbose:
                print(f"\nGeneration {gen + 1}/{self.max_generations}")
                print(f"  Best fitness:  {results[0]['fitness']:.4f}")
                print(f"  Mean fitness:  {gen_info['mean_fitness']:.4f}")
                print(f"  Best genome:   {results[0]['genome']}")

            # Check convergence
            if len(self.generation_history) > 10:
                recent = [h['best_fitness'] for h in self.generation_history[-10:]]
                improvement = max(recent) - min(recent)
                if improvement < 0.001:
                    if verbose:
                        print(f"\nConverged at generation {gen + 1}")
                    break

            # Elite selection (keep best unchanged)
            elite = [deepcopy(results[i]['genome']) for i in range(self.elite_count)]

            # Parent selection
            parents = self.select_parents(results)

            # Create offspring
            offspring = self.create_offspring(parents)

            # New population = elite + offspring
            population = elite + offspring[:self.population_size - self.elite_count]

        return self.best_genome

    def _default_build(self, genome: ArchitecturalGenome) -> nn.Module:
        """
        Build a simple model from genome for evaluation.

        In production, replace with full model builder that creates
        the complete Token-Mind architecture.
        """
        from fusion.network import MultimodalConsciousAgentNetwork

        # Map genome modalities to dimensions
        modality_dim_map = {
            'text': 768, 'vision': 1024, 'audio': 512,
            'electromagnetic': 1024, 'gravitational': 2049,
            'quantum': 256, 'microscope': 512,
        }

        modality_dims = {
            m: modality_dim_map.get(m, 512)
            for m in genome.genes['sensor_modalities']
        }

        model = MultimodalConsciousAgentNetwork(
            modality_dims=modality_dims,
            fusion_dim=genome.genes['fusion_dim'],
            use_low_rank=genome.genes.get('use_low_rank_fusion', True),
            rank=genome.genes.get('fusion_rank', 64),
        )

        return model

    def get_evolution_summary(self) -> str:
        """Generate summary of evolution history."""
        if not self.generation_history:
            return "No evolution history."

        lines = [
            "=" * 60,
            "EVOLUTION SUMMARY",
            "=" * 60,
            f"Generations completed: {len(self.generation_history)}",
            f"Best fitness achieved: {self.best_fitness:.4f}",
            f"Best genome: {self.best_genome}",
            "",
            "Fitness progression:",
        ]

        for h in self.generation_history:
            gen = h['generation']
            best = h['best_fitness']
            mean = h['mean_fitness']
            bar = "█" * int(best * 40) + "░" * (40 - int(best * 40))
            lines.append(f"  Gen {gen:3d}: {bar} best={best:.3f} mean={mean:.3f}")

        return "\n".join(lines)
