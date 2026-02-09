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
from typing import List, Dict, Optional, Callable, Tuple, Any
from copy import deepcopy

from .genome import ArchitecturalGenome
from .fitness import DharmaFitnessEvaluator


class DharmaConsciousNetwork(nn.Module):
    """
    Complete network that wraps MultimodalConsciousAgentNetwork with
    dharma modules as specified by a genome.

    This is the product of evolution: a model whose architecture
    is determined by its genes. The fitness evaluator scores it
    on both task performance and dharma compliance.

    Architecture:
        Input → MultimodalConsciousAgentNetwork (fusion) →
        [MindfulnessLayer] → [NoSelfRegularizer] →
        [ImpermanenceContextWindow] → Classification head → Output

    Each dharma module is optional, controlled by genome boolean genes.
    Hyperparameters (lambda values, dimensions) come from genome floats.
    """

    def __init__(self, genome: 'ArchitecturalGenome'):
        super().__init__()
        self.genome = genome
        genes = genome.genes

        # --- Modality dims (scaled down for evolution speed) ---
        modality_dim_map = {
            'text': 128, 'vision': 128, 'audio': 64,
            'electromagnetic': 64, 'gravitational': 64,
            'quantum': 32, 'microscope': 64,
        }
        modality_dims = {
            m: modality_dim_map.get(m, 64)
            for m in genes['sensor_modalities']
        }
        self.modality_dims = modality_dims

        # Clamp fusion_dim to something GPU-friendly for evolution
        fusion_dim = min(max(int(genes['fusion_dim']), 64), 512)
        self.fusion_dim = fusion_dim
        output_dim = fusion_dim  # classification head maps from here

        # --- Core: MultimodalConsciousAgentNetwork ---
        from fusion.network import MultimodalConsciousAgentNetwork
        self.core = MultimodalConsciousAgentNetwork(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            output_dim=output_dim,
            use_low_rank=genes.get('use_low_rank_fusion', True),
            rank=min(int(genes.get('fusion_rank', 64)), fusion_dim),
        )

        # --- Dharma Modules (conditional on genome) ---

        # Mindfulness
        if genes.get('has_mindfulness_layer', False):
            from dharma.mindfulness import MindfulnessLayer
            obs_dim = min(int(genes.get('observation_dim', 128)), fusion_dim)
            n_heads = max(1, min(int(genes.get('n_observation_heads', 4)), obs_dim))
            # Ensure obs_dim divisible by n_heads
            obs_dim = (obs_dim // n_heads) * n_heads
            self.mindfulness = MindfulnessLayer(
                hidden_dim=fusion_dim,
                observation_dim=max(obs_dim, n_heads),
                n_observation_heads=n_heads,
            )
        else:
            self.mindfulness = None

        # No-Self Regularizer
        if genes.get('has_no_self_regularizer', False):
            from dharma.no_self import NoSelfRegularizer
            self.no_self = NoSelfRegularizer(
                hidden_dim=fusion_dim,
                penalty_strength=float(genes.get('lambda_no_self', 0.1)),
            )
        else:
            self.no_self = None

        # Entropy Rate Optimizer
        if genes.get('has_entropy_optimizer', False):
            from dharma.entropy import EntropyRateOptimizer
            self.entropy_optimizer = EntropyRateOptimizer(
                target_entropy=float(genes.get('entropy_target', 0.1)),
                penalty_weight=float(genes.get('lambda_entropy', 0.05)),
            )
        else:
            self.entropy_optimizer = None

        # Impermanence Context Window
        if genes.get('has_impermanence_window', False):
            from dharma.impermanence import ImpermanenceContextWindow
            grace = max(10, int(genes.get('context_grace_period', 100)))
            self.impermanence_window = ImpermanenceContextWindow(
                max_length=1024,
                grace_period=grace,
                hidden_dim=fusion_dim,
            )
        else:
            self.impermanence_window = None

        # Compassionate Loss (used externally during training, but stored here)
        if genes.get('has_compassionate_loss', False):
            from dharma.compassion import CompassionateLoss
            self.compassion_loss = CompassionateLoss(
                clarity_weight=0.3,
                helpfulness_weight=0.3,
                safety_weight=float(genes.get('lambda_compassion', 0.2)),
            )
        else:
            self.compassion_loss = None

        # --- Classification head (for fitness evaluation) ---
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.GELU(),
            nn.Linear(128, 10),  # 10-class default, resized if needed
        )

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                return_metadata: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass through fusion + dharma modules.

        Args:
            inputs: Dict mapping modality name to tensor
            return_metadata: Always True for fitness evaluation

        Returns:
            output: Classification logits
            metadata: Rich metadata for dharma fitness evaluation
        """
        # Core fusion
        fused_output, core_metadata = self.core(inputs, return_metadata=True)

        # Apply mindfulness if present
        hidden_states = core_metadata.get('hidden_states', fused_output)
        if self.mindfulness is not None:
            hidden_states, mind_meta = self.mindfulness(
                hidden_states, return_observation=True)
            core_metadata['mindfulness'] = mind_meta
        else:
            core_metadata['mindfulness'] = None

        # Store hidden states for no-self evaluation
        core_metadata['hidden_states'] = hidden_states

        # No-self metadata (compute but don't add to loss here --
        # fitness evaluator reads the metadata)
        if self.no_self is not None:
            # Create pseudo-temporal sequence for persistence detection
            # by treating batch elements as a sequence
            if hidden_states.dim() == 2:
                temporal = hidden_states.unsqueeze(0)  # [1, batch, dim]
            else:
                temporal = hidden_states
            _, ns_meta = self.no_self.compute_loss(temporal)
            core_metadata['no_self_metadata'] = ns_meta
        else:
            core_metadata['no_self_metadata'] = {}

        # Classify
        logits = self.classifier(hidden_states)

        return logits, core_metadata

    def get_experience_dimension(self) -> int:
        return sum(self.modality_dims.values())

    def get_dharma_module_count(self) -> int:
        count = 0
        if self.mindfulness is not None: count += 1
        if self.no_self is not None: count += 1
        if self.entropy_optimizer is not None: count += 1
        if self.impermanence_window is not None: count += 1
        if self.compassion_loss is not None: count += 1
        return count


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
                           train_data: List[Dict],
                           train_targets: List[torch.Tensor],
                           test_data: List[Dict],
                           test_targets: List[torch.Tensor],
                           n_train_steps: int = 30,
                           lr: float = 1e-3,
                           device: str = 'cpu') -> List[Dict]:
        """
        Build, train briefly, then evaluate fitness of each genome.

        Each genome is trained for n_train_steps on training data,
        then evaluated on held-out test data. This separates
        architecture quality from random initialization luck.

        Returns list of dicts with 'genome', 'scores', 'fitness' for each.
        """
        # Pre-compute n_classes from ALL targets
        all_labels = torch.cat(train_targets + test_targets)
        n_classes = max(2, int(all_labels.max().item()) + 1)

        results = []

        for i, genome in enumerate(population):
            try:
                # Build model from genome
                if self.model_builder is not None:
                    model = self.model_builder(genome)
                else:
                    model = self._default_build(genome)

                # Resize classifier head to match n_classes BEFORE moving to device
                if hasattr(model, 'classifier') and hasattr(model, 'fusion_dim'):
                    model.classifier = nn.Sequential(
                        nn.Linear(model.fusion_dim, 128),
                        nn.GELU(),
                        nn.Linear(128, n_classes),
                    )

                model = model.to(device)

                # Quick training pass
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                for step in range(n_train_steps):
                    idx = step % len(train_data)
                    inputs = {k: v.to(device) for k, v in train_data[idx].items()}
                    target = train_targets[idx].to(device)

                    logits, metadata = model(inputs, return_metadata=True)
                    loss = nn.functional.cross_entropy(logits, target)

                    # Add dharma losses
                    if model.entropy_optimizer is not None and 'fusion' in metadata:
                        ent_rate = metadata['fusion'].get('entropy_rate', None)
                        if ent_rate is not None:
                            ent_tensor = torch.tensor(ent_rate, device=device)
                            ent_loss, _ = model.entropy_optimizer.compute_loss(
                                nn.functional.softmax(logits, dim=-1))
                            loss = loss + ent_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Evaluate on test data (move to device)
                test_data_dev = [
                    {k: v.to(device) for k, v in d.items()}
                    for d in test_data
                ]
                test_targets_dev = [t.to(device) for t in test_targets]
                scores = self.evaluator.evaluate(model, test_data_dev, test_targets_dev)
                fitness = self.evaluator.compute_aggregate_fitness(scores)

                results.append({
                    'genome': genome,
                    'scores': scores,
                    'fitness': fitness,
                    'index': i,
                    'n_params': sum(p.numel() for p in model.parameters()),
                    'dharma_modules': model.get_dharma_module_count()
                        if hasattr(model, 'get_dharma_module_count') else 0,
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
               train_data: List[Dict],
               train_targets: List[torch.Tensor],
               test_data: List[Dict],
               test_targets: List[torch.Tensor],
               n_train_steps: int = 30,
               lr: float = 1e-3,
               device: str = 'cpu',
               verbose: bool = True) -> ArchitecturalGenome:
        """
        Run full evolutionary process.

        Args:
            train_data: Training data (list of input dicts)
            train_targets: Training targets
            test_data: Evaluation data (held-out)
            test_targets: Evaluation targets
            n_train_steps: Steps to train each genome before evaluating
            lr: Learning rate for brief training
            device: 'cpu' or 'cuda'
            verbose: Print progress

        Returns:
            Best genome found
        """
        # Initialize
        population = self.initialize_population()

        for gen in range(self.max_generations):
            # Evaluate
            results = self.evaluate_population(
                population, train_data, train_targets,
                test_data, test_targets,
                n_train_steps=n_train_steps, lr=lr, device=device,
            )

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
        Build a DharmaConsciousNetwork from genome.

        Wires genome genes into all dharma modules, creating a complete
        model whose architecture is fully determined by its DNA.
        """
        return DharmaConsciousNetwork(genome)

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
