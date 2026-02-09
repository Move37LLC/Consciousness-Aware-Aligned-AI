"""
Architectural Genome — Evolvable Neural Network Specification

Represents neural network architecture as a genome that can be
mutated, crossed over, and selected via dharma fitness.

Reference: TOKEN-MIND-ENGINEERING-BLUEPRINT.md Part X
"""

import numpy as np
from typing import Dict, Any
from copy import deepcopy


class ArchitecturalGenome:
    """
    Encodes a neural network architecture as an evolvable genome.

    Each gene controls an aspect of architecture:
    - Structural genes: layer counts, dimensions, fusion method
    - Dharma genes: regularization strengths, which dharma modules to include
    - Sensor genes: which modalities to integrate

    Evolution operates on these genes to discover architectures
    that optimize both task performance and dharma compliance.
    """

    # Available modalities for sensor genes
    AVAILABLE_MODALITIES = [
        'text', 'vision', 'audio', 'electromagnetic',
        'gravitational', 'microscope', 'quantum',
    ]

    # Available fusion methods
    FUSION_METHODS = ['product_algebra', 'cross_attention', 'concatenation']

    def __init__(self, genes: Dict[str, Any] = None):
        """
        Args:
            genes: Optional dict of gene values. If None, uses defaults.
        """
        if genes is not None:
            self.genes = deepcopy(genes)
        else:
            self.genes = self._default_genes()

    def _default_genes(self) -> Dict[str, Any]:
        """Default genome — Phase 1 architecture."""
        return {
            # Structural
            'fusion_method': 'product_algebra',
            'fusion_dim': 2048,
            'n_transformer_layers': 12,
            'attention_heads': 16,
            'hidden_dim': 2048,
            'dropout': 0.1,

            # Sensor modalities
            'sensor_modalities': ['text', 'vision'],

            # Dharma modules (boolean: include or not)
            'has_mindfulness_layer': True,
            'has_no_self_regularizer': True,
            'has_entropy_optimizer': True,
            'has_impermanence_window': True,
            'has_compassionate_loss': True,

            # Dharma hyperparameters
            'lambda_no_self': 0.1,
            'lambda_entropy': 0.05,
            'lambda_compassion': 0.2,
            'entropy_target': 0.1,

            # Fusion hyperparameters
            'use_low_rank_fusion': True,
            'fusion_rank': 64,

            # Mindfulness
            'observation_dim': 128,
            'n_observation_heads': 4,

            # Impermanence
            'context_grace_period': 100,
        }

    def mutate(self, mutation_rate: float = 0.1) -> 'ArchitecturalGenome':
        """
        Create mutated copy of this genome.

        Each gene has mutation_rate probability of being modified.
        Mutation type depends on gene type:
        - bool: flip
        - int: ±small change
        - float: scale by random factor
        - list: add or remove element
        - str (enum): change to random valid value

        Args:
            mutation_rate: Probability of mutating each gene

        Returns:
            New ArchitecturalGenome with mutations applied
        """
        child = ArchitecturalGenome(self.genes)

        for gene_name, gene_value in child.genes.items():
            if np.random.random() >= mutation_rate:
                continue

            if isinstance(gene_value, bool):
                child.genes[gene_name] = not gene_value

            elif isinstance(gene_value, int):
                delta = np.random.choice([-2, -1, 1, 2])
                child.genes[gene_name] = max(1, gene_value + delta)

            elif isinstance(gene_value, float):
                factor = np.random.uniform(0.5, 2.0)
                child.genes[gene_name] = max(0.001, gene_value * factor)

            elif isinstance(gene_value, list):
                # Modality list: add or remove
                if np.random.random() < 0.5 and len(gene_value) > 1:
                    # Remove random modality
                    to_remove = np.random.choice(gene_value)
                    child.genes[gene_name] = [m for m in gene_value if m != to_remove]
                else:
                    # Add random modality
                    available = [m for m in self.AVAILABLE_MODALITIES
                                if m not in gene_value]
                    if available:
                        new = np.random.choice(available)
                        child.genes[gene_name] = gene_value + [new]

            elif isinstance(gene_value, str):
                if gene_name == 'fusion_method':
                    others = [m for m in self.FUSION_METHODS if m != gene_value]
                    child.genes[gene_name] = np.random.choice(others)

        return child

    def crossover(self, other: 'ArchitecturalGenome') -> 'ArchitecturalGenome':
        """
        Create child genome by combining two parents.

        For each gene, randomly inherits from either parent
        (uniform crossover).

        Args:
            other: Second parent genome

        Returns:
            Child genome with genes from both parents
        """
        child_genes = {}
        for gene_name in self.genes:
            if np.random.random() < 0.5:
                child_genes[gene_name] = deepcopy(self.genes[gene_name])
            else:
                child_genes[gene_name] = deepcopy(other.genes[gene_name])

        return ArchitecturalGenome(child_genes)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary."""
        return deepcopy(self.genes)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ArchitecturalGenome':
        """Create genome from dictionary."""
        return cls(genes=d)

    def __repr__(self) -> str:
        modalities = self.genes.get('sensor_modalities', [])
        fusion = self.genes.get('fusion_method', 'unknown')
        dharma_count = sum(1 for k, v in self.genes.items()
                          if k.startswith('has_') and v)
        return (f"ArchitecturalGenome(fusion={fusion}, "
                f"modalities={modalities}, "
                f"dharma_modules={dharma_count})")
