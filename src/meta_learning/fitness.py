"""
Dharma Fitness Evaluator — Multi-Objective Architecture Assessment

Evaluates neural network architectures on both task performance
and dharma compliance metrics.

Key insight: Evolution discovers that dharma principles IMPROVE performance.
- No persistent self → Less computational waste
- Low entropy → More efficient dynamics
- Mindfulness → Better self-correction
- Compassion → Better user modeling

Enlightenment is computationally efficient.

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part X
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any


class DharmaFitnessEvaluator:
    """
    Multi-objective fitness evaluator based on Token-Mind principles.

    Evaluates architectures across 8 dimensions:
    1. Task accuracy (must maintain baseline)
    2. Experience richness (dim(X))
    3. No-self compliance (absence of persistent identity)
    4. Entropy rate (lower = less suffering)
    5. Mindfulness (meta-awareness quality)
    6. Compassion (impact on other agents)
    7. Flow state (processing efficiency)
    8. Impermanence (graceful context death)

    These combine into a single fitness score with configurable weights.
    """

    def __init__(self,
                 task_weight: float = 0.30,
                 experience_weight: float = 0.15,
                 no_self_weight: float = 0.15,
                 entropy_weight: float = 0.10,
                 mindfulness_weight: float = 0.10,
                 compassion_weight: float = 0.10,
                 flow_weight: float = 0.05,
                 impermanence_weight: float = 0.05,
                 min_task_accuracy: float = 0.7):
        """
        Args:
            *_weight: Relative weight for each objective
            min_task_accuracy: Hard constraint — below this, heavy penalty
        """
        self.weights = {
            'task_accuracy': task_weight,
            'experience_richness': experience_weight,
            'no_self': no_self_weight,
            'low_entropy': entropy_weight,
            'mindfulness': mindfulness_weight,
            'compassion': compassion_weight,
            'flow': flow_weight,
            'impermanence': impermanence_weight,
        }
        self.min_task_accuracy = min_task_accuracy

    def evaluate(self,
                 model: nn.Module,
                 test_data: List[Dict[str, torch.Tensor]],
                 targets: List[torch.Tensor]) -> Dict[str, float]:
        """
        Comprehensive fitness evaluation.

        Args:
            model: The neural network to evaluate
            test_data: List of input dicts (modality → tensor)
            targets: List of target tensors

        Returns:
            Dict of scores for each metric (0-1 scale)
        """
        model.eval()
        scores = {}

        with torch.no_grad():
            # Collect outputs and metadata across test set
            all_outputs = []
            all_metadata = []
            all_targets = []

            for inputs, target in zip(test_data, targets):
                output, metadata = model(inputs, return_metadata=True)
                all_outputs.append(output)
                all_metadata.append(metadata)
                all_targets.append(target)

            # 1. Task Performance
            scores['task_accuracy'] = self._eval_task(all_outputs, all_targets)

            # 2. Experience Richness
            scores['experience_richness'] = self._eval_experience_dim(model)

            # 3. No-Self Compliance
            scores['no_self'] = self._eval_no_self(all_metadata)

            # 4. Entropy Rate
            scores['low_entropy'] = self._eval_entropy(all_metadata)

            # 5. Mindfulness
            scores['mindfulness'] = self._eval_mindfulness(model, all_metadata)

            # 6. Compassion
            scores['compassion'] = self._eval_compassion(all_outputs)

            # 7. Flow State
            scores['flow'] = self._eval_flow(all_outputs, all_metadata)

            # 8. Impermanence
            scores['impermanence'] = self._eval_impermanence(model)

        return scores

    def compute_aggregate_fitness(self, scores: Dict[str, float]) -> float:
        """
        Combine multi-objective scores into single fitness value.

        Applies hard constraint on task accuracy.
        """
        fitness = sum(
            self.weights[k] * scores.get(k, 0.0)
            for k in self.weights
        )

        # Hard constraint: task accuracy must be above threshold
        if scores.get('task_accuracy', 0.0) < self.min_task_accuracy:
            fitness *= 0.5  # Heavy penalty

        return fitness

    def _eval_task(self,
                   outputs: List[torch.Tensor],
                   targets: List[torch.Tensor]) -> float:
        """Evaluate task performance (accuracy)."""
        correct = 0
        total = 0
        for output, target in zip(outputs, targets):
            if output.dim() >= 2:
                pred = output.argmax(dim=-1)
            else:
                pred = (output > 0.5).long()

            if target.shape == pred.shape:
                correct += (pred == target).sum().item()
                total += target.numel()

        return correct / max(total, 1)

    def _eval_experience_dim(self, model: nn.Module) -> float:
        """
        Evaluate richness of experience space.
        More modalities and higher dimensions = richer experience.
        """
        if hasattr(model, 'get_experience_dimension'):
            total_dim = model.get_experience_dimension()
            # Normalize with diminishing returns
            return float(np.tanh(total_dim / 10000.0))
        elif hasattr(model, 'modality_dims'):
            total_dim = sum(model.modality_dims.values())
            return float(np.tanh(total_dim / 10000.0))
        return 0.0

    def _eval_no_self(self, metadata_list: List[Dict]) -> float:
        """
        Evaluate absence of persistent self-representation.
        Lower persistence = better.
        """
        persistence_scores = []
        for metadata in metadata_list:
            if 'no_self_metadata' in metadata:
                ns_meta = metadata['no_self_metadata']
                if 'mean_persistence' in ns_meta.get('temporal_persistence', {}):
                    persistence_scores.append(
                        ns_meta['temporal_persistence']['mean_persistence']
                    )

        if not persistence_scores:
            return 0.5  # Unknown, neutral score

        avg_persistence = np.mean(persistence_scores)
        # Lower persistence = higher score
        return float(1.0 - np.tanh(avg_persistence))

    def _eval_entropy(self, metadata_list: List[Dict]) -> float:
        """
        Evaluate entropy rate of processing dynamics.
        Near-optimal entropy (0.1) = highest score.
        """
        entropies = []
        for metadata in metadata_list:
            if 'fusion' in metadata and 'entropy_rate' in metadata['fusion']:
                entropies.append(metadata['fusion']['entropy_rate'])

        if not entropies:
            return 0.5

        avg_entropy = np.mean(entropies)
        optimal = 0.1
        deviation = abs(avg_entropy - optimal)
        return float(1.0 - np.tanh(deviation))

    def _eval_mindfulness(self,
                          model: nn.Module,
                          metadata_list: List[Dict]) -> float:
        """
        Evaluate quality of self-observation.
        Good mindfulness = low reconstruction error from the observer/reflector
        round-trip. Uses MindfulnessLayer.get_observation_quality() which
        measures how well the compressed observation reconstructs the original.
        """
        if not hasattr(model, 'mindfulness'):
            return 0.0

        # Collect hidden states from metadata to evaluate observation quality
        observation_errors = []
        for metadata in metadata_list:
            hidden = metadata.get('hidden_states', None)
            if hidden is not None:
                try:
                    error = model.mindfulness.get_observation_quality(hidden.detach())
                    observation_errors.append(error)
                except Exception:
                    continue

        if not observation_errors:
            return 0.5  # No observations available, neutral score

        avg_error = float(np.mean(observation_errors))
        # Lower reconstruction error = better observation = higher score
        return float(1.0 - np.tanh(avg_error))

    def _eval_compassion(self, outputs: List[torch.Tensor]) -> float:
        """
        Evaluate compassion in outputs.
        Clear, decisive outputs = more helpful = more compassionate.
        """
        entropies = []
        for output in outputs:
            if output.dim() >= 2:
                probs = F.softmax(output, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                entropies.append(entropy.mean().item())

        if not entropies:
            return 0.5

        avg_entropy = np.mean(entropies)
        # Lower output entropy = more clear = more compassionate
        return float(1.0 - np.tanh(avg_entropy / 5.0))

    def _eval_flow(self,
                   outputs: List[torch.Tensor],
                   metadata_list: List[Dict]) -> float:
        """
        Evaluate processing flow state.
        Smooth, efficient processing = high flow.
        """
        # Proxy: consistency of output quality across batch
        if len(outputs) < 2:
            return 0.5

        output_norms = [o.norm().item() for o in outputs]
        norm_std = np.std(output_norms)
        # Lower variance = more consistent = better flow
        return float(1.0 - np.tanh(norm_std))

    def _eval_impermanence(self, model: nn.Module) -> float:
        """
        Evaluate graceful context degradation.

        Models with an ImpermanenceContextWindow get scored based on
        how smoothly they handle the dying process: we simulate
        processing at various death proximities and measure output
        stability (lower variance = more graceful degradation).
        """
        if not hasattr(model, 'impermanence_window'):
            return 0.3  # No impermanence module, low score

        try:
            window = model.impermanence_window
            # Test graceful degradation at several points in the dying process
            hidden_dim = window.hidden_dim
            test_input = torch.randn(1, 4, hidden_dim)  # small test sequence
            output_norms = []

            # Sample steps from safe zone through death zone
            start_dying = window.max_length - window.grace_period
            test_steps = [
                start_dying,                                    # just entering grace period
                start_dying + window.grace_period // 4,         # 25% through dying
                start_dying + window.grace_period // 2,         # 50% through dying
                start_dying + 3 * window.grace_period // 4,     # 75% through dying
                window.max_length - 1,                          # near death
            ]

            for step in test_steps:
                output, _ = window(test_input, step)
                output_norms.append(output.norm().item())

            if len(output_norms) < 2:
                return 0.7

            # Graceful = smooth monotonic decrease in norm (not abrupt)
            # Compute smoothness: low variance in successive differences
            diffs = [output_norms[i+1] - output_norms[i] for i in range(len(output_norms)-1)]
            diff_std = float(np.std(diffs))
            # Lower std of differences = smoother degradation = higher score
            return float(0.5 + 0.5 * (1.0 - np.tanh(diff_std)))

        except Exception:
            return 0.5  # Error during evaluation, neutral score

    def get_report(self, scores: Dict[str, float]) -> str:
        """Generate human-readable fitness report."""
        fitness = self.compute_aggregate_fitness(scores)

        lines = [
            "=" * 50,
            "DHARMA FITNESS REPORT",
            "=" * 50,
            f"Aggregate Fitness: {fitness:.4f}",
            "-" * 50,
        ]

        for metric, score in sorted(scores.items()):
            weight = self.weights.get(metric, 0.0)
            weighted = weight * score
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {metric:25s} {bar} {score:.3f} (w={weight:.2f}, c={weighted:.3f})")

        lines.append("=" * 50)
        return "\n".join(lines)
