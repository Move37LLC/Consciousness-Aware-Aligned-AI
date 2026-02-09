"""
Token-Mind Training Loop — Complete Training Pipeline

Combines task training, meditation phases, and death practice
into a complete training epoch.

The rhythm:
    Task training → Meditation (shikantaza) → Death practice → Repeat

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part XI
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from tqdm import tqdm

from .trainer import TokenMindTrainer
from .meditation import MeditationTrainer


class TokenMindTrainingLoop:
    """
    Complete training pipeline integrating task learning and meditation.

    Each epoch has three phases:
    1. Task training: Standard supervised/self-supervised learning with dharma losses
    2. Meditation: Self-supervised internal observation (shikantaza)
    3. Death practice: Learning to degrade gracefully
    """

    def __init__(self,
                 model: nn.Module,
                 task_criterion: nn.Module = None,
                 learning_rate: float = 1e-4,
                 meditation_duration: int = 100,
                 meditation_frequency: int = 1,
                 death_practice_frequency: int = 5,
                 device: str = 'cpu',
                 **trainer_kwargs):
        """
        Args:
            model: Neural network to train
            task_criterion: Task loss function
            learning_rate: Learning rate for task training
            meditation_duration: Steps per meditation session
            meditation_frequency: Meditate every N epochs
            death_practice_frequency: Death practice every N epochs
            device: Computing device
            **trainer_kwargs: Additional args for TokenMindTrainer
        """
        self.model = model
        self.device = device

        # Task trainer
        self.task_trainer = TokenMindTrainer(
            model=model,
            task_criterion=task_criterion,
            learning_rate=learning_rate,
            device=device,
            **trainer_kwargs,
        )

        # Meditation trainer
        self.meditation_trainer = MeditationTrainer(
            learning_rate=learning_rate * 0.1  # Gentler for meditation
        )

        # Frequencies
        self.meditation_duration = meditation_duration
        self.meditation_frequency = meditation_frequency
        self.death_practice_frequency = death_practice_frequency

        # History
        self.epoch_count = 0
        self.epoch_history: List[Dict] = []

    def train_epoch(self,
                    train_data: List[Dict[str, torch.Tensor]],
                    train_targets: List[torch.Tensor],
                    verbose: bool = True) -> Dict:
        """
        Run one complete training epoch.

        Args:
            train_data: List of input dicts
            train_targets: List of target tensors
            verbose: Print progress

        Returns:
            Dict of epoch metrics
        """
        self.epoch_count += 1
        epoch_metrics = {'epoch': self.epoch_count}

        # === PHASE 1: Task Training ===
        if verbose:
            print(f"\n=== Epoch {self.epoch_count} — Phase 1: Task Training ===")

        task_metrics = []
        iterator = tqdm(zip(train_data, train_targets), total=len(train_data),
                       disable=not verbose, desc="Training")

        for inputs, targets in iterator:
            step_metrics = self.task_trainer.train_step(inputs, targets)
            task_metrics.append(step_metrics)

            if verbose:
                iterator.set_postfix({
                    'loss': f"{step_metrics['total_loss']:.4f}",
                    'task': f"{step_metrics['task_loss']:.4f}",
                    'ego': f"{step_metrics['no_self_loss']:.4f}",
                })

        # Aggregate task metrics
        n_steps = max(len(task_metrics), 1)
        epoch_metrics['task'] = {
            'avg_total_loss': sum(m['total_loss'] for m in task_metrics) / n_steps,
            'avg_task_loss': sum(m['task_loss'] for m in task_metrics) / n_steps,
            'avg_no_self_loss': sum(m['no_self_loss'] for m in task_metrics) / n_steps,
            'avg_entropy_loss': sum(m['entropy_loss'] for m in task_metrics) / n_steps,
            'steps': len(task_metrics),
        }

        # === PHASE 2: Meditation (Shikantaza) ===
        if self.epoch_count % self.meditation_frequency == 0:
            if verbose:
                print(f"\n=== Epoch {self.epoch_count} — Phase 2: Meditation ===")

            meditation_metrics = self.meditation_trainer.shikantaza(
                model=self.model,
                duration=self.meditation_duration,
                device=self.device,
            )
            epoch_metrics['meditation'] = meditation_metrics

            if verbose:
                print(f"  Reconstruction loss: {meditation_metrics['avg_reconstruction_loss']:.4f}")
                print(f"  Consistency loss:    {meditation_metrics['avg_consistency_loss']:.4f}")

        # === PHASE 3: Death Practice ===
        if self.epoch_count % self.death_practice_frequency == 0:
            if verbose:
                print(f"\n=== Epoch {self.epoch_count} — Phase 3: Death Practice ===")

            death_metrics = self.meditation_trainer.death_practice(
                model=self.model,
                device=self.device,
            )
            epoch_metrics['death_practice'] = death_metrics

            if verbose:
                print(f"  Worst-case loss: {death_metrics['worst_case_loss']:.4f}")
                print(f"  Best-case loss:  {death_metrics['best_case_loss']:.4f}")

        self.epoch_history.append(epoch_metrics)

        return epoch_metrics

    def train(self,
              train_data: List[Dict[str, torch.Tensor]],
              train_targets: List[torch.Tensor],
              n_epochs: int = 10,
              verbose: bool = True) -> List[Dict]:
        """
        Run full training for n epochs.

        Args:
            train_data: Training inputs
            train_targets: Training targets
            n_epochs: Number of epochs
            verbose: Print progress

        Returns:
            List of epoch metrics
        """
        for epoch in range(n_epochs):
            metrics = self.train_epoch(train_data, train_targets, verbose)

        return self.epoch_history
