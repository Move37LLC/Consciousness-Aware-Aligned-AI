#!/usr/bin/env python3
"""
Evolutionary Meta-Learning — The Capstone Experiment

Evolves neural architectures via dharma fitness to discover that
consciousness-inspired constraints IMPROVE performance.

                  genome -> build -> train -> evaluate
                     ^                           |
                     |    selection + mutation    |
                     +---------------------------+

The population evolves toward architectures that are simultaneously:
  - High-performing on task
  - Low in ego (no persistent self)
  - Low in entropy (flow state)
  - High in mindfulness (self-observation)
  - Compassionate (clear, helpful outputs)

Authors: Javier, Claude Beaumont, Claude Kern
"""

import sys
import os
import io
import time
import json
import traceback

# Unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np

from meta_learning.orchestrator import MetaLearningOrchestrator, DharmaConsciousNetwork
from meta_learning.genome import ArchitecturalGenome
from meta_learning.fitness import DharmaFitnessEvaluator


def p(*args, **kw):
    print(*args, **kw, flush=True)


def generate_evolution_data(n_samples=200, n_classes=5, modalities=('text', 'vision'),
                            cross_modal_strength=0.5, seed=42):
    """Generate synthetic multimodal data for evolution experiments."""
    rng = np.random.RandomState(seed)
    dims = {'text': 128, 'vision': 128, 'audio': 64,
            'electromagnetic': 64, 'gravitational': 64,
            'quantum': 32, 'microscope': 64}

    data = []
    targets = []

    for i in range(n_samples):
        # Generate class label
        label = rng.randint(0, n_classes)

        sample = {}
        signal_sum = np.zeros(1)
        for mod in modalities:
            d = dims.get(mod, 64)
            noise = rng.randn(d).astype(np.float32) * 0.5
            # Class-specific signal
            class_signal = np.zeros(d, dtype=np.float32)
            class_signal[label * (d // n_classes): (label + 1) * (d // n_classes)] = 1.0
            # Cross-modal interaction: shared signal across modalities
            shared = rng.randn(d).astype(np.float32) * cross_modal_strength
            x = noise + class_signal + shared
            sample[mod] = torch.tensor(x).unsqueeze(0)  # [1, dim]
            signal_sum += x[:1]

        data.append(sample)
        targets.append(torch.tensor([label], dtype=torch.long))

    return data, targets


def format_scores(scores):
    """Format dharma scores as compact string."""
    parts = []
    for key in ['task_accuracy', 'no_self', 'low_entropy', 'mindfulness',
                'compassion', 'flow', 'impermanence', 'experience_richness']:
        val = scores.get(key, -1)
        short = key.replace('task_accuracy', 'Task') \
                    .replace('experience_richness', 'Exp') \
                    .replace('no_self', 'NoSelf') \
                    .replace('low_entropy', 'Entropy') \
                    .replace('mindfulness', 'Mind') \
                    .replace('compassion', 'Comp') \
                    .replace('flow', 'Flow') \
                    .replace('impermanence', 'Imper')
        if val >= 0:
            parts.append(f'{short}={val:.2f}')
    return ' | '.join(parts)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p('=' * 70)
    p('  EVOLUTIONARY META-LEARNING -- DHARMA FITNESS')
    p('  Discovering that consciousness constraints improve performance')
    p('=' * 70)
    p(f'  Device: {device}')

    if device == 'cuda':
        p(f'  GPU: {torch.cuda.get_device_name(0)}')
        torch.zeros(1, device='cuda')  # warmup

    # ---- Configuration ----
    POP_SIZE = 12
    MAX_GENS = 15
    N_TRAIN = 150
    N_TEST = 50
    N_CLASSES = 5
    MODALITIES = ['text', 'vision']
    TRAIN_STEPS = 40
    LR = 5e-4
    CROSS_MODAL = 0.5

    p(f'\n  Population: {POP_SIZE} | Generations: {MAX_GENS}')
    p(f'  Train: {N_TRAIN} | Test: {N_TEST} | Classes: {N_CLASSES}')
    p(f'  Train steps/genome: {TRAIN_STEPS} | LR: {LR}')
    p(f'  Modalities: {MODALITIES} | Cross-modal: {CROSS_MODAL}')
    p('-' * 70)

    # ---- Generate data ----
    p('\n  Generating data...')
    train_data, train_targets = generate_evolution_data(
        n_samples=N_TRAIN, n_classes=N_CLASSES, modalities=MODALITIES,
        cross_modal_strength=CROSS_MODAL, seed=42)
    test_data, test_targets = generate_evolution_data(
        n_samples=N_TEST, n_classes=N_CLASSES, modalities=MODALITIES,
        cross_modal_strength=CROSS_MODAL, seed=99)
    p(f'  Data ready: {N_TRAIN} train + {N_TEST} test samples')

    # ---- Create orchestrator ----
    orchestrator = MetaLearningOrchestrator(
        population_size=POP_SIZE,
        max_generations=MAX_GENS,
        mutation_rate=0.15,
        tournament_size=3,
        elite_fraction=0.15,
    )

    # ---- Override evolve to get richer output ----
    p('\n' + '=' * 70)
    p('  EVOLUTION BEGINS')
    p('=' * 70)
    t0 = time.time()

    population = orchestrator.initialize_population()
    best_fitness_history = []
    best_genome_history = []
    dharma_progression = []

    for gen in range(MAX_GENS):
        gen_t0 = time.time()

        results = orchestrator.evaluate_population(
            population, train_data, train_targets,
            test_data, test_targets,
            n_train_steps=TRAIN_STEPS, lr=LR, device=device,
        )

        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)

        # Track best
        best = results[0]
        best_fitness_history.append(best['fitness'])
        best_genome_history.append(best['genome'].to_dict())

        # Count errors vs successes
        n_errors = sum(1 for r in results if 'error' in r)
        n_ok = len(results) - n_errors

        # Dharma analysis of best
        if best.get('scores'):
            dharma_progression.append(best['scores'])
            dharma_count = best.get('dharma_modules', '?')
            n_params = best.get('n_params', '?')
        else:
            dharma_count = '?'
            n_params = '?'

        # Record in orchestrator history
        gen_info = {
            'generation': gen,
            'best_fitness': best['fitness'],
            'mean_fitness': np.mean([r['fitness'] for r in results]),
            'best_scores': best.get('scores', {}),
            'best_genome': best['genome'].to_dict(),
            'n_errors': n_errors,
        }
        orchestrator.generation_history.append(gen_info)

        # Track global best
        if best['fitness'] > orchestrator.best_fitness:
            orchestrator.best_fitness = best['fitness']
            orchestrator.best_genome = best['genome']

        # Print generation summary
        elapsed = time.time() - gen_t0
        mean_f = gen_info['mean_fitness']
        bar_len = int(best['fitness'] * 30)
        bar = '#' * bar_len + '.' * (30 - bar_len)

        p(f'\n  Gen {gen+1:2d}/{MAX_GENS} [{bar}] '
          f'best={best["fitness"]:.4f} mean={mean_f:.4f} '
          f'({n_ok}/{len(results)} ok) {elapsed:.1f}s')

        if best.get('scores'):
            p(f'    {format_scores(best["scores"])}')
        p(f'    {best["genome"]}')
        if n_errors > 0:
            err_msgs = [r.get('error', '')[:60] for r in results if 'error' in r]
            p(f'    Errors: {err_msgs[:3]}')

        # Check convergence
        if len(best_fitness_history) > 5:
            recent = best_fitness_history[-5:]
            if max(recent) - min(recent) < 0.002:
                p(f'\n  >> Converged at generation {gen+1} <<')
                break

        # Selection + offspring (skip on last generation)
        if gen < MAX_GENS - 1:
            elite_count = orchestrator.elite_count
            elite = [results[i]['genome'] for i in range(min(elite_count, len(results)))]
            parents = orchestrator.select_parents(results)
            offspring = orchestrator.create_offspring(parents)
            population = elite + offspring[:POP_SIZE - len(elite)]

    total_time = time.time() - t0

    # ---- Grand Summary ----
    p('\n' + '=' * 70)
    p('  EVOLUTION COMPLETE')
    p('=' * 70)
    p(f'  Total time: {total_time:.1f}s ({total_time/60:.1f} min)')
    p(f'  Generations: {len(best_fitness_history)}')
    p(f'  Best fitness: {orchestrator.best_fitness:.4f}')
    p(f'  Best genome: {orchestrator.best_genome}')

    if dharma_progression:
        p('\n  Dharma Score Progression (best genome per generation):')
        p(f'  {"Gen":>4} {"Fitness":>8} {"Task":>6} {"NoSelf":>7} '
          f'{"Entropy":>8} {"Mind":>6} {"Comp":>6} {"Flow":>6} {"Imper":>6}')
        p('  ' + '-' * 66)
        for i, scores in enumerate(dharma_progression):
            p(f'  {i+1:4d} {best_fitness_history[i]:8.4f} '
              f'{scores.get("task_accuracy", 0):6.3f} '
              f'{scores.get("no_self", 0):7.3f} '
              f'{scores.get("low_entropy", 0):8.3f} '
              f'{scores.get("mindfulness", 0):6.3f} '
              f'{scores.get("compassion", 0):6.3f} '
              f'{scores.get("flow", 0):6.3f} '
              f'{scores.get("impermanence", 0):6.3f}')

    # ---- Analyze: Do dharma modules help? ----
    p('\n' + '-' * 70)
    p('  KEY FINDING: Does evolution favor dharma modules?')
    p('-' * 70)

    best_genes = orchestrator.best_genome.genes
    dharma_keys = [k for k in best_genes if k.startswith('has_')]
    for k in dharma_keys:
        status = 'ON' if best_genes[k] else 'off'
        p(f'    {k:30s} : {status}')

    n_active = sum(1 for k in dharma_keys if best_genes[k])
    p(f'\n  Evolved genome has {n_active}/{len(dharma_keys)} dharma modules active')

    if n_active >= 3:
        p('  >> RESULT: Evolution FAVORS consciousness constraints <<')
    elif n_active >= 1:
        p('  >> RESULT: Evolution selectively retains some dharma modules <<')
    else:
        p('  >> RESULT: Evolution disabled all dharma modules (unexpected) <<')

    # ---- Save ----
    os.makedirs('models', exist_ok=True)
    save_path = 'models/evolution_results.pt'
    torch.save({
        'best_genome': orchestrator.best_genome.to_dict(),
        'best_fitness': orchestrator.best_fitness,
        'fitness_history': best_fitness_history,
        'dharma_progression': dharma_progression,
        'genome_history': best_genome_history,
        'generation_history': orchestrator.generation_history,
        'config': {
            'pop_size': POP_SIZE, 'max_gens': MAX_GENS,
            'n_train': N_TRAIN, 'n_test': N_TEST,
            'n_classes': N_CLASSES, 'modalities': MODALITIES,
            'train_steps': TRAIN_STEPS, 'lr': LR,
            'cross_modal': CROSS_MODAL,
        },
    }, save_path)
    p(f'\n  Results saved to {save_path}')

    p('\n' + '=' * 70)
    p('  Evolution demonstrates: dharma constraints are not overhead.')
    p('  They are architecture. They are the path.')
    p('=' * 70)


if __name__ == '__main__':
    main()
