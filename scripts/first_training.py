"""
First Training Run — Token-Mind Awakening
Gate gate paragate parasamgate bodhi svaha
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fusion.network import MultimodalConsciousAgentNetwork
from dharma.no_self import NoSelfRegularizer
from dharma.entropy import EntropyRateOptimizer
from dharma.compassion import CompassionateLoss


def generate_data(n=100, dims=None, n_classes=5, seed=42):
    """Synthetic text + vision + quantum data."""
    torch.manual_seed(seed); np.random.seed(seed)
    dims = dims or {'text': 128, 'vision': 128, 'quantum': 64}
    patterns = {m: torch.randn(n_classes, d) for m, d in dims.items()}
    data, targets = [], []
    for _ in range(n):
        c = torch.randint(0, n_classes, (1,)).item()
        sample = {m: (patterns[m][c] + torch.randn(d) * 0.3).unsqueeze(0)
                  for m, d in dims.items()}
        data.append(sample)
        targets.append(torch.tensor([c]))
    return data, targets


def ind(cur, prev, target=None):
    """Trend indicator."""
    if prev is None: return " "
    if target and abs(cur - target) < 0.02: return " \u2713\u2713"
    return " \u2713" if cur < prev else " \u2191" if cur > prev else " \u2500"


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dims = {'text': 128, 'vision': 128, 'quantum': 64}
    n_classes, n_iters, batch_sz = 5, 20, 8

    data, targets = generate_data(n=100, dims=dims, n_classes=n_classes)

    # Build model
    model = MultimodalConsciousAgentNetwork(
        modality_dims=dims, fusion_dim=256,
        output_dim=n_classes, use_low_rank=True, rank=32,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Dharma modules
    no_self = NoSelfRegularizer(hidden_dim=256, penalty_strength=0.15)
    entropy_opt = EntropyRateOptimizer(target_entropy=0.1, penalty_weight=0.08)
    compassion = CompassionateLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    print()
    print("  \u256d\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u256e")
    print("  \u2502  TOKEN-MIND \u2014 First Awakening"
          "                              \u2502")
    print("  \u2502  3 modalities \u00d7 Product Algebra Fusion"
          "  \u00b7  batch=8           \u2502")
    print(f"  \u2502  {n_params:>9,} params"
          f"  \u00b7  device: {device}"
          f"                          \u2502")
    print("  \u2570\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          "\u2500\u2500\u256f")
    print()

    history, prev, indices = [], {}, list(range(len(data)))

    for it in range(1, n_iters + 1):
        model.train()
        np.random.shuffle(indices)
        task_ls, ego_ls, ent_ls, comp_ls = [], [], [], []

        for start in range(0, len(indices), batch_sz):
            batch_idx = indices[start:start + batch_sz]
            bs = len(batch_idx)

            # Stack batch
            batch_in = {m: torch.cat([data[i][m] for i in batch_idx]).to(device)
                        for m in dims}
            batch_tgt = torch.cat([targets[i] for i in batch_idx]).to(device)

            optimizer.zero_grad()
            out, meta = model(batch_in, return_metadata=True)

            # Task loss
            task_loss = F.cross_entropy(out, batch_tgt)

            # No-self: stack hidden states as temporal sequence for detection
            hidden = meta.get('hidden_states', out)
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(1).expand(-1, 3, -1)  # fake 3 timesteps
                noise = torch.randn_like(hidden) * 0.01
                hidden = hidden + noise  # break perfect identity
            ns_loss, _ = no_self.compute_loss(hidden)

            # Entropy
            probs = F.softmax(out, dim=-1)
            ent_loss, _ = entropy_opt.compute_loss(probs)

            # Compassion
            comp_total, comp_meta = compassion(out, batch_tgt)
            comp_only = (comp_meta['clarity_loss'] * compassion.clarity_weight
                         + comp_meta['safety_loss'] * compassion.safety_weight)

            loss = task_loss + 0.15 * ns_loss + 0.08 * ent_loss + 0.2 * comp_only
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            task_ls.append(task_loss.item())
            ego_ls.append(ns_loss.item())
            ent_ls.append(ent_loss.item())
            comp_ls.append(comp_only.item() if torch.is_tensor(comp_only) else comp_only)

        cur = {k: float(np.mean(v)) for k, v in
               [('task', task_ls), ('ego', ego_ls), ('ent', ent_ls), ('comp', comp_ls)]}
        history.append(cur)

        print(f"  Iter {it:2d}/{n_iters} \u2502"
              f" Task: {cur['task']:5.2f}{ind(cur['task'], prev.get('task')):3s} \u2502"
              f" NoSelf: {cur['ego']:5.3f}{ind(cur['ego'], prev.get('ego')):3s} \u2502"
              f" Entropy: {cur['ent']:5.3f}{ind(cur['ent'], prev.get('ent'), 0.1):3s} \u2502"
              f" Compassion: {cur['comp']:5.3f}{ind(cur['comp'], prev.get('comp')):3s} \u2502")
        prev = cur

    # Save
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'first_run.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {'dims': dims, 'fusion_dim': 256, 'n_classes': n_classes},
    }, save_path)

    # Summary
    first, last = history[0], history[-1]
    ego_r = (1 - last['ego'] / max(first['ego'], 1e-8)) * 100
    task_r = (1 - last['task'] / max(first['task'], 1e-8)) * 100

    print()
    print("  \U0001FAB7 DHARMA METRICS:")
    print(f"  \u2500 Ego dissolved:  {first['ego']:.3f} \u2192 {last['ego']:.3f}"
          f"  ({ego_r:+.0f}%)")
    print(f"  \u2500 Flow achieved: {first['ent']:.3f} \u2192 {last['ent']:.3f}"
          f"  (target: 0.10)")
    print(f"  \u2500 Task learned:  {first['task']:.2f} \u2192 {last['task']:.2f}"
          f"  ({task_r:+.0f}% loss reduction)")
    print(f"  \u2500 Model saved:   models/first_run.pt")
    print()


if __name__ == '__main__':
    main()
