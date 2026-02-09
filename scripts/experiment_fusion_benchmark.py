"""
Experiment: Product Algebra vs Attention Fusion Benchmark

THE critical first experiment.

Compares three fusion methods on synthetic multimodal data:
1. Product Algebra Fusion (Token-Mind) - Kronecker product of Markov kernels
2. Cross-Attention Fusion (Standard) - Multi-head attention baseline
3. Concatenation Fusion (Simple) - Lower baseline

If Product Algebra outperforms on cross-modal reasoning tasks,
it validates the core hypothesis that Hoffman's mathematics
has engineering utility beyond metaphor.

Usage:
    python scripts/experiment_fusion_benchmark.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fusion.conscious_agent import ConsciousAgentState
from fusion.product_algebra import ProductAlgebraFusion, AttentionFusionBaseline
from fusion.network import MultimodalConsciousAgentNetwork


# ============================================================
# Synthetic Data Generation
# ============================================================

def generate_cross_modal_data(
    n_samples: int = 1000,
    text_dim: int = 256,
    vision_dim: int = 256,
    n_classes: int = 10,
    cross_modal_strength: float = 0.5,
    seed: int = 42,
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
    """
    Generate synthetic multimodal data where the label depends on
    INTERACTION between modalities, not just individual features.

    This tests whether fusion methods capture cross-modal structure.

    Args:
        n_samples: Number of samples
        text_dim: Text feature dimension
        vision_dim: Vision feature dimension
        n_classes: Number of classes
        cross_modal_strength: How much label depends on cross-modal interaction
        seed: Random seed

    Returns:
        (list of input dicts, list of target tensors)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = []
    targets = []

    # Generate class-specific patterns
    text_patterns = torch.randn(n_classes, text_dim)
    vision_patterns = torch.randn(n_classes, vision_dim)

    # Cross-modal interaction patterns (the key test)
    interaction_patterns = torch.randn(n_classes, text_dim, vision_dim)

    for _ in range(n_samples):
        # Pick a class
        label = torch.randint(0, n_classes, (1,)).item()

        # Generate text features (noisy class pattern)
        text = text_patterns[label] + torch.randn(text_dim) * 0.5

        # Generate vision features (noisy class pattern)
        vision = vision_patterns[label] + torch.randn(vision_dim) * 0.5

        # Add cross-modal signal into the features themselves:
        # The true class depends on the INTERACTION of text and vision.
        # We inject a cross-modal fingerprint into both modalities so that
        # the label can only be reliably recovered by examining how the
        # modalities relate to each other.
        interaction = torch.outer(text, vision)
        cross_modal_scores = torch.zeros(n_classes)
        for c in range(n_classes):
            cross_modal_scores[c] = (interaction * interaction_patterns[c]).sum()

        # Blend individual-pattern label with cross-modal-derived label
        cross_modal_label = cross_modal_scores.argmax().item()

        if np.random.random() < cross_modal_strength:
            # Label determined by cross-modal interaction
            label = cross_modal_label
        # else: keep original individual-pattern label

        data.append({'text': text.unsqueeze(0), 'vision': vision.unsqueeze(0)})
        targets.append(torch.tensor([label]))

    return data, targets


# ============================================================
# Model Builders
# ============================================================

class FusionBenchmarkModel(nn.Module):
    """Wrapper for benchmarking different fusion methods."""

    def __init__(self, fusion_layer, fusion_dim: int, output_dim: int):
        super().__init__()
        self.fusion_layer = fusion_layer
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )
        self.fusion_dim = fusion_dim

    def forward(self, inputs: Dict[str, torch.Tensor],
                return_metadata: bool = False):
        # Create agent states from inputs
        states = []
        for name, tensor in inputs.items():
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            states.append(ConsciousAgentState(
                experience=tensor,
                transition_matrix=torch.eye(tensor.shape[-1], device=tensor.device),
                modality=name,
            ))

        # Fuse
        fused, metadata = self.fusion_layer(states)

        # Classify
        logits = self.classifier(fused)

        if return_metadata:
            return logits, {'hidden_states': fused, **metadata}
        return logits, {}


def build_product_algebra_model(text_dim, vision_dim, fusion_dim, n_classes):
    fusion = ProductAlgebraFusion(
        agent_dims=[text_dim, vision_dim],
        fusion_dim=fusion_dim,
        use_low_rank=True,
        rank=32,
        preserve_markov=True,
    )
    return FusionBenchmarkModel(fusion, fusion_dim, n_classes)


def build_attention_model(text_dim, vision_dim, fusion_dim, n_classes):
    fusion = AttentionFusionBaseline(
        agent_dims=[text_dim, vision_dim],
        fusion_dim=fusion_dim,
        n_heads=8,
    )
    return FusionBenchmarkModel(fusion, fusion_dim, n_classes)


class ConcatenationFusion(nn.Module):
    """Simple concatenation baseline."""

    def __init__(self, agent_dims, fusion_dim):
        super().__init__()
        total_dim = sum(agent_dims)
        self.proj = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

    def forward(self, agent_states, return_product_structure=False):
        experiences = [s.experience for s in agent_states]
        experiences = [e.unsqueeze(0) if e.dim() == 1 else e for e in experiences]
        concatenated = torch.cat(experiences, dim=-1)
        fused = self.proj(concatenated)
        return fused, {'fusion_method': 'concatenation'}


def build_concatenation_model(text_dim, vision_dim, fusion_dim, n_classes):
    fusion = ConcatenationFusion(
        agent_dims=[text_dim, vision_dim],
        fusion_dim=fusion_dim,
    )
    return FusionBenchmarkModel(fusion, fusion_dim, n_classes)


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_evaluate(
    model: nn.Module,
    train_data: List[Dict],
    train_targets: List[torch.Tensor],
    test_data: List[Dict],
    test_targets: List[torch.Tensor],
    n_epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Train model and return evaluation metrics."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        indices = np.random.permutation(len(train_data))

        for idx in indices:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in train_data[idx].items()}
            target = train_targets[idx].to(device)

            output, _ = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, target in zip(test_data, test_targets):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            target = target.to(device)

            output, _ = model(inputs)
            pred = output.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += target.numel()

    accuracy = correct / total

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    return {
        'accuracy': accuracy,
        'n_parameters': n_params,
        'final_training_loss': epoch_loss / len(train_data),
    }


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    """Run the complete fusion benchmark experiment."""

    print("=" * 70)
    print("FUSION BENCHMARK: Product Algebra vs Attention vs Concatenation")
    print("=" * 70)
    print()

    # Configuration
    text_dim = 256
    vision_dim = 256
    fusion_dim = 512
    n_classes = 10
    n_trials = 5
    n_epochs = 50
    n_train = 800
    n_test = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Trials: {n_trials}")
    print(f"Epochs per trial: {n_epochs}")
    print()

    # Test with different cross-modal strengths
    cross_modal_strengths = [0.0, 0.3, 0.5, 0.7, 1.0]

    all_results = {}

    for strength in cross_modal_strengths:
        print(f"\n{'='*70}")
        print(f"Cross-Modal Strength: {strength}")
        print(f"{'='*70}")

        results = {
            'product_algebra': [],
            'cross_attention': [],
            'concatenation': [],
        }

        for trial in range(n_trials):
            seed = 42 + trial

            # Generate data
            all_data, all_targets = generate_cross_modal_data(
                n_samples=n_train + n_test,
                text_dim=text_dim,
                vision_dim=vision_dim,
                n_classes=n_classes,
                cross_modal_strength=strength,
                seed=seed,
            )

            train_data = all_data[:n_train]
            train_targets = all_targets[:n_train]
            test_data = all_data[n_train:]
            test_targets = all_targets[n_train:]

            # Model A: Product Algebra
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_pa = build_product_algebra_model(
                text_dim, vision_dim, fusion_dim, n_classes
            )
            result_pa = train_and_evaluate(
                model_pa, train_data, train_targets,
                test_data, test_targets,
                n_epochs=n_epochs, device=device,
            )
            results['product_algebra'].append(result_pa)

            # Model B: Cross-Attention
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_attn = build_attention_model(
                text_dim, vision_dim, fusion_dim, n_classes
            )
            result_attn = train_and_evaluate(
                model_attn, train_data, train_targets,
                test_data, test_targets,
                n_epochs=n_epochs, device=device,
            )
            results['cross_attention'].append(result_attn)

            # Model C: Concatenation
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_concat = build_concatenation_model(
                text_dim, vision_dim, fusion_dim, n_classes
            )
            result_concat = train_and_evaluate(
                model_concat, train_data, train_targets,
                test_data, test_targets,
                n_epochs=n_epochs, device=device,
            )
            results['concatenation'].append(result_concat)

            print(f"  Trial {trial+1}: PA={result_pa['accuracy']:.3f} "
                  f"Attn={result_attn['accuracy']:.3f} "
                  f"Cat={result_concat['accuracy']:.3f}")

        # Summarize
        print(f"\n  SUMMARY (cross_modal={strength}):")
        for method, trials in results.items():
            accs = [t['accuracy'] for t in trials]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            params = trials[0]['n_parameters']
            print(f"    {method:20s}: {mean_acc:.3f} ± {std_acc:.3f}  "
                  f"(params: {params:,})")

        all_results[strength] = results

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Cross-Modal Strength':>22s} | {'Product Algebra':>15s} | "
          f"{'Cross-Attention':>15s} | {'Concatenation':>15s}")
    print("-" * 75)

    for strength in cross_modal_strengths:
        results = all_results[strength]
        pa_acc = np.mean([t['accuracy'] for t in results['product_algebra']])
        attn_acc = np.mean([t['accuracy'] for t in results['cross_attention']])
        cat_acc = np.mean([t['accuracy'] for t in results['concatenation']])

        winner = "PA" if pa_acc > max(attn_acc, cat_acc) else (
            "Attn" if attn_acc > cat_acc else "Cat"
        )

        print(f"  {strength:20.1f} | {pa_acc:14.3f} | "
              f"{attn_acc:14.3f} | {cat_acc:14.3f}  ← {winner}")

    print(f"\nHypothesis: Product Algebra should excel at high cross-modal strength")
    print(f"(where label depends on modality INTERACTION, not individual features)")


if __name__ == '__main__':
    run_experiment()
