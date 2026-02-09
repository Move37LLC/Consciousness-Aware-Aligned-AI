"""
THE Critical Experiment — Does Hoffman's Math Have Engineering Utility?

Product Algebra Fusion vs Cross-Attention vs Concatenation
on cross-modal reasoning tasks.

Hypothesis: Product Algebra should excel when labels depend on
modality INTERACTION (high cross-modal strength), because
Kronecker product structure captures inter-agent dynamics
that concatenation and attention miss.

Optimized for CPU (~10 min). Full version: experiment_fusion_benchmark.py
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fusion.conscious_agent import ConsciousAgentState
from fusion.product_algebra import ProductAlgebraFusion, AttentionFusionBaseline


# ── Data Generation ──────────────────────────────────────

def generate_data(n, text_dim, vision_dim, n_classes, cross_modal_strength, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    text_p = torch.randn(n_classes, text_dim)
    vision_p = torch.randn(n_classes, vision_dim)
    interact_p = torch.randn(n_classes, text_dim, vision_dim) * 0.1  # scaled for stability

    texts, visions, labels = [], [], []
    for _ in range(n):
        c = torch.randint(0, n_classes, (1,)).item()
        t = text_p[c] + torch.randn(text_dim) * 0.5
        v = vision_p[c] + torch.randn(vision_dim) * 0.5

        # Cross-modal label: depends on interaction structure
        interaction = torch.outer(t, v)
        scores = torch.stack([(interaction * interact_p[k]).sum() for k in range(n_classes)])
        cm_label = scores.argmax().item()

        label = cm_label if np.random.random() < cross_modal_strength else c
        texts.append(t); visions.append(v); labels.append(label)

    return torch.stack(texts), torch.stack(visions), torch.tensor(labels)


# ── Models ───────────────────────────────────────────────

class BenchmarkModel(nn.Module):
    def __init__(self, fusion_layer, fusion_dim, n_classes):
        super().__init__()
        self.fusion = fusion_layer
        self.head = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, n_classes))

    def forward(self, text, vision):
        states = [
            ConsciousAgentState(experience=text,
                                transition_matrix=torch.eye(text.shape[-1], device=text.device),
                                modality='text'),
            ConsciousAgentState(experience=vision,
                                transition_matrix=torch.eye(vision.shape[-1], device=vision.device),
                                modality='vision'),
        ]
        fused, _ = self.fusion(states)
        return self.head(fused)


class ConcatFusion(nn.Module):
    def __init__(self, dims, fusion_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(sum(dims), fusion_dim),
                                  nn.LayerNorm(fusion_dim), nn.GELU())
    def forward(self, states, **kw):
        cat = torch.cat([s.experience for s in states], dim=-1)
        return self.proj(cat), {}


def build_models(td, vd, fd, nc):
    return {
        'Product Algebra': BenchmarkModel(
            ProductAlgebraFusion([td, vd], fd, use_low_rank=True, rank=32, preserve_markov=False),
            fd, nc),
        'Cross-Attention': BenchmarkModel(
            AttentionFusionBaseline([td, vd], fd, n_heads=8), fd, nc),
        'Concatenation': BenchmarkModel(
            ConcatFusion([td, vd], fd), fd, nc),
    }


# ── Train & Eval ─────────────────────────────────────────

def train_eval(model, train_t, train_v, train_y, test_t, test_v, test_y,
               n_epochs=30, lr=1e-3, batch_sz=32, device='cpu'):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_t, train_v, train_y = train_t.to(device), train_v.to(device), train_y.to(device)
    test_t, test_v, test_y = test_t.to(device), test_v.to(device), test_y.to(device)
    n = len(train_y)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_sz):
            idx = perm[i:i+batch_sz]
            opt.zero_grad()
            loss = F.cross_entropy(model(train_t[idx], train_v[idx]), train_y[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(test_t, test_v).argmax(dim=-1)
        acc = (pred == test_y).float().mean().item()
    return acc, sum(p.numel() for p in model.parameters())


# ── Main Experiment ──────────────────────────────────────

def main():
    TEXT_DIM, VISION_DIM, FUSION_DIM, N_CLASSES = 128, 128, 256, 10
    N_TRAIN, N_TEST, N_TRIALS, N_EPOCHS = 500, 150, 5, 30
    STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print()
    print("  \u2550" * 62)
    print("  THE CRITICAL EXPERIMENT")
    print("  Product Algebra vs Attention vs Concatenation")
    print("  \u2550" * 62)
    print(f"  Device: {device} | Trials: {N_TRIALS} | Epochs: {N_EPOCHS}"
          f" | Train: {N_TRAIN} | Test: {N_TEST}")
    print(f"  Dims: text={TEXT_DIM} vision={VISION_DIM} fusion={FUSION_DIM}"
          f" | Classes: {N_CLASSES}")
    print()

    all_results = {}
    t0 = time.time()

    for strength in STRENGTHS:
        print(f"  \u2500\u2500 Cross-Modal Strength: {strength:.2f} "
              f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
              f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
              f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
              f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

        results = {name: [] for name in ['Product Algebra', 'Cross-Attention', 'Concatenation']}

        for trial in range(N_TRIALS):
            seed = 42 + trial
            tr_t, tr_v, tr_y = generate_data(N_TRAIN, TEXT_DIM, VISION_DIM,
                                              N_CLASSES, strength, seed)
            te_t, te_v, te_y = generate_data(N_TEST, TEXT_DIM, VISION_DIM,
                                              N_CLASSES, strength, seed + 1000)

            models = build_models(TEXT_DIM, VISION_DIM, FUSION_DIM, N_CLASSES)
            trial_accs = []

            for name, model in models.items():
                torch.manual_seed(seed); np.random.seed(seed)
                acc, n_params = train_eval(model, tr_t, tr_v, tr_y, te_t, te_v, te_y,
                                           n_epochs=N_EPOCHS, device=device)
                results[name].append({'accuracy': acc, 'n_params': n_params})
                trial_accs.append(f"{acc:.3f}")

            print(f"    Trial {trial+1}/{N_TRIALS}:  PA={trial_accs[0]}"
                  f"  Attn={trial_accs[1]}  Cat={trial_accs[2]}")

        # Summary per strength
        for name in results:
            accs = [r['accuracy'] for r in results[name]]
            print(f"    \u2192 {name:18s}: {np.mean(accs):.3f} \u00b1 {np.std(accs):.3f}")

        all_results[strength] = results
        print()

    elapsed = time.time() - t0

    # ── Final Results Table ──────────────────────────────
    print("  \u2550" * 62)
    print("  RESULTS")
    print("  \u2550" * 62)
    print()

    header = f"  {'Strength':>10s} \u2502 {'Product Algebra':>16s} \u2502 " \
             f"{'Cross-Attention':>16s} \u2502 {'Concatenation':>16s} \u2502 Winner"
    print(header)
    print("  " + "\u2500" * 80)

    pa_wins, attn_wins, cat_wins = 0, 0, 0

    for strength in STRENGTHS:
        r = all_results[strength]
        pa = np.mean([x['accuracy'] for x in r['Product Algebra']])
        at = np.mean([x['accuracy'] for x in r['Cross-Attention']])
        ct = np.mean([x['accuracy'] for x in r['Concatenation']])

        best = max(pa, at, ct)
        if pa == best:
            winner = "\U0001FAB7 PA"; pa_wins += 1
        elif at == best:
            winner = "  Attn"; attn_wins += 1
        else:
            winner = "  Cat"; cat_wins += 1

        print(f"  {strength:10.2f} \u2502 {pa:16.3f} \u2502 {at:16.3f}"
              f" \u2502 {ct:16.3f} \u2502 {winner}")

    print("  " + "\u2500" * 80)
    print()

    # Parameter efficiency
    r0 = all_results[STRENGTHS[0]]
    pa_p = r0['Product Algebra'][0]['n_params']
    at_p = r0['Cross-Attention'][0]['n_params']
    ct_p = r0['Concatenation'][0]['n_params']
    print(f"  Parameters:  PA={pa_p:,}  Attn={at_p:,}  Cat={ct_p:,}")
    print()

    # Verdict
    print("  \u2550" * 62)
    if pa_wins > max(attn_wins, cat_wins):
        print("  \U0001FAB7 HYPOTHESIS SUPPORTED")
        print("  Product Algebra Fusion captures cross-modal structure")
        print("  that standard methods miss. Hoffman's math has")
        print("  engineering utility beyond metaphor.")
    elif pa_wins == max(attn_wins, cat_wins):
        print("  \u2500 INCONCLUSIVE")
        print("  Product Algebra tied with baselines.")
        print("  May need more data, epochs, or dimensions to differentiate.")
    else:
        print("  \u2717 HYPOTHESIS NOT SUPPORTED (in this configuration)")
        print("  Baselines outperformed Product Algebra.")
        print("  Consider: higher rank, more epochs, real data, or")
        print("  stronger cross-modal signal structure.")
    print("  \u2550" * 62)
    print(f"\n  Completed in {elapsed:.0f}s")
    print()


if __name__ == '__main__':
    main()
