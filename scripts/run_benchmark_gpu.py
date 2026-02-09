"""
THE Critical Experiment - GPU-Powered Full Run

Product Algebra Fusion vs Cross-Attention vs Concatenation
Tests whether Kronecker-product fusion captures cross-modal interactions
that simpler methods miss.

Two experiments:
  A) Binary classification (high signal) - proof of concept
  B) 10-class classification (full power) - if A works

GTX 1070 (8GB WDDM) - estimated ~10-20 min
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fusion.conscious_agent import ConsciousAgentState
from fusion.product_algebra import ProductAlgebraFusion, AttentionFusionBaseline


def p(*args, **kwargs):
    """Print with flush."""
    print(*args, **kwargs, flush=True)


def generate_data(n, text_dim, vision_dim, n_classes, cross_modal_strength, seed):
    """Generate synthetic cross-modal data.
    
    Key insight: the label depends on INTERACTION between modalities,
    not just individual modality features. The cross_modal_strength
    parameter controls how much the label depends on interaction vs
    individual features.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Class prototypes for each modality
    text_p = torch.randn(n_classes, text_dim) * 1.5
    vision_p = torch.randn(n_classes, vision_dim) * 1.5
    # Interaction patterns per class (THIS is what PA should capture better)
    interact_p = torch.randn(n_classes, text_dim, vision_dim) * 0.3

    texts, visions, labels = [], [], []
    for _ in range(n):
        # Base class from individual features
        c = torch.randint(0, n_classes, (1,)).item()
        t = text_p[c] + torch.randn(text_dim) * 0.4
        v = vision_p[c] + torch.randn(vision_dim) * 0.4
        
        # Cross-modal interaction signal
        interaction = torch.outer(t, v)
        scores = torch.stack([(interaction * interact_p[k]).sum() for k in range(n_classes)])
        cm_label = scores.argmax().item()
        
        # Blend: at strength=1.0, label is ENTIRELY from interaction
        label = cm_label if np.random.random() < cross_modal_strength else c
        texts.append(t)
        visions.append(v)
        labels.append(label)

    return torch.stack(texts), torch.stack(visions), torch.tensor(labels)


class BenchmarkModel(nn.Module):
    def __init__(self, fusion_layer, fusion_dim, n_classes):
        super().__init__()
        self.fusion = fusion_layer
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, text, vision):
        states = [
            ConsciousAgentState(
                experience=text,
                transition_matrix=torch.eye(text.shape[-1], device=text.device),
                modality='text'),
            ConsciousAgentState(
                experience=vision,
                transition_matrix=torch.eye(vision.shape[-1], device=vision.device),
                modality='vision'),
        ]
        fused, _ = self.fusion(states)
        return self.head(fused)


class ConcatFusion(nn.Module):
    """Simple concatenation baseline."""
    def __init__(self, dims, fusion_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sum(dims), fusion_dim),
            nn.LayerNorm(fusion_dim), nn.GELU())
    
    def forward(self, states, **kw):
        cat = torch.cat([s.experience for s in states], dim=-1)
        return self.proj(cat), {}


def build_models(td, vd, fd, nc):
    return {
        'Product Algebra': BenchmarkModel(
            ProductAlgebraFusion([td, vd], fd, use_low_rank=True,
                                rank=min(64, fd), preserve_markov=False),
            fd, nc),
        'Cross-Attention': BenchmarkModel(
            AttentionFusionBaseline([td, vd], fd, n_heads=8), fd, nc),
        'Concatenation': BenchmarkModel(
            ConcatFusion([td, vd], fd), fd, nc),
    }


def train_eval(model, tr_t, tr_v, tr_y, te_t, te_v, te_y,
               n_epochs, lr=1e-3, batch_sz=64, device='cuda'):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    tr_t, tr_v, tr_y = tr_t.to(device), tr_v.to(device), tr_y.to(device)
    te_t, te_v, te_y = te_t.to(device), te_v.to(device), te_y.to(device)
    n = len(tr_y)

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_sz):
            idx = perm[i:i+batch_sz]
            opt.zero_grad()
            loss = F.cross_entropy(model(tr_t[idx], tr_v[idx]), tr_y[idx])
            loss.backward()
            opt.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(te_y), batch_sz):
            pred = model(te_t[i:i+batch_sz], te_v[i:i+batch_sz]).argmax(dim=-1)
            all_preds.append(pred)
        preds = torch.cat(all_preds)
        acc = (preds == te_y).float().mean().item()
    return acc, sum(p.numel() for p in model.parameters())


def run_experiment(name, text_dim, vision_dim, fusion_dim, n_classes,
                   n_train, n_test, n_trials, n_epochs, strengths, device):
    """Run one complete experiment configuration."""
    p(f"\n  {'=' * 66}")
    p(f"  EXPERIMENT: {name}")
    p(f"  {'=' * 66}")
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    p(f"  Device: {gpu_name}")
    p(f"  {n_classes} classes | {n_train} train | {n_test} test | {n_epochs} epochs | {n_trials} trials")
    p(f"  text={text_dim}d  vision={vision_dim}d  fusion={fusion_dim}d")
    p()

    all_results = {}
    t0 = time.time()

    for strength in strengths:
        p(f"  --- Cross-Modal Strength: {strength:.2f} " + "-" * 40)

        results = {n: [] for n in ['Product Algebra', 'Cross-Attention', 'Concatenation']}

        for trial in range(n_trials):
            seed = 42 + trial
            tr_t, tr_v, tr_y = generate_data(
                n_train, text_dim, vision_dim, n_classes, strength, seed)
            te_t, te_v, te_y = generate_data(
                n_test, text_dim, vision_dim, n_classes, strength, seed + 1000)

            models = build_models(text_dim, vision_dim, fusion_dim, n_classes)
            trial_accs = []

            for mname, model in models.items():
                torch.manual_seed(seed)
                np.random.seed(seed)
                acc, n_params = train_eval(
                    model, tr_t, tr_v, tr_y, te_t, te_v, te_y,
                    n_epochs=n_epochs, device=device)
                results[mname].append({'accuracy': acc, 'n_params': n_params})
                trial_accs.append(f"{acc:.3f}")

            elapsed = time.time() - t0
            p(f"    Trial {trial+1}/{n_trials}:  PA={trial_accs[0]}"
              f"  Attn={trial_accs[1]}  Cat={trial_accs[2]}"
              f"  [{elapsed:.0f}s]")

        for mname in results:
            accs = [r['accuracy'] for r in results[mname]]
            p(f"    >> {mname:18s}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

        all_results[strength] = results
        p()

    return all_results, time.time() - t0


def print_results(all_results, strengths):
    """Pretty-print the final results table."""
    p(f"\n  {'=' * 66}")
    p(f"  RESULTS TABLE")
    p(f"  {'=' * 66}")
    p()

    header = (f"  {'Strength':>10s} | {'Product Algebra':>16s} | "
              f"{'Cross-Attention':>16s} | {'Concatenation':>16s} | Winner")
    p(header)
    p("  " + "-" * 84)

    pa_wins, attn_wins, cat_wins = 0, 0, 0
    pa_advantage_at_high = []

    for strength in strengths:
        r = all_results[strength]
        pa = np.mean([x['accuracy'] for x in r['Product Algebra']])
        at = np.mean([x['accuracy'] for x in r['Cross-Attention']])
        ct = np.mean([x['accuracy'] for x in r['Concatenation']])
        pa_std = np.std([x['accuracy'] for x in r['Product Algebra']])
        at_std = np.std([x['accuracy'] for x in r['Cross-Attention']])
        ct_std = np.std([x['accuracy'] for x in r['Concatenation']])

        best = max(pa, at, ct)
        if pa >= best - 1e-9:
            winner = " << PA"; pa_wins += 1
        elif at >= best - 1e-9:
            winner = "   Attn"; attn_wins += 1
        else:
            winner = "   Cat"; cat_wins += 1

        if strength >= 0.75:
            pa_advantage_at_high.append(pa - max(at, ct))

        p(f"  {strength:10.2f} | {pa:7.3f}+/-{pa_std:.3f}"
          f"   | {at:7.3f}+/-{at_std:.3f}"
          f"   | {ct:7.3f}+/-{ct_std:.3f}"
          f"   | {winner}")

    p("  " + "-" * 84)
    p()

    # Parameter counts
    r0 = all_results[strengths[0]]
    pa_p = r0['Product Algebra'][0]['n_params']
    at_p = r0['Cross-Attention'][0]['n_params']
    ct_p = r0['Concatenation'][0]['n_params']
    p(f"  Parameters:  PA={pa_p:,}  Attn={at_p:,}  Cat={ct_p:,}")
    if pa_p < at_p:
        p(f"  PA uses {(1 - pa_p / at_p) * 100:.0f}% fewer params than Attention")
    p()

    # Verdict
    p(f"  {'=' * 66}")
    avg_high_adv = np.mean(pa_advantage_at_high) if pa_advantage_at_high else 0

    if pa_wins >= 3:
        p("  >>> HYPOTHESIS STRONGLY SUPPORTED <<<")
        p("  Product Algebra Fusion demonstrates clear advantage.")
    elif pa_wins >= 2 or (pa_wins >= 1 and avg_high_adv > 0.01):
        p("  >>> HYPOTHESIS PARTIALLY SUPPORTED <<<")
        p("  Product Algebra shows advantage at high cross-modal strength.")
        if avg_high_adv > 0:
            p(f"  Mean PA advantage at strength >= 0.75: +{avg_high_adv:.3f}")
    else:
        p("  >>> RESULTS INCONCLUSIVE <<<")
        p(f"  PA wins: {pa_wins}, Attn: {attn_wins}, Cat: {cat_wins}")
    p(f"  {'=' * 66}")
    
    return pa_wins, attn_wins, cat_wins, avg_high_adv


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Warm up CUDA
    if device == 'cuda':
        p("  Warming up CUDA...")
        torch.zeros(1, device='cuda')
        # Warm up cuBLAS
        a = torch.randn(128, 128, device='cuda')
        _ = a @ a.T
        p("  CUDA ready.")

    STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]

    # ---- EXPERIMENT A: Binary (high signal, fast) ----
    results_a, elapsed_a = run_experiment(
        name="A: Binary Classification (2 classes)",
        text_dim=128, vision_dim=128, fusion_dim=256, n_classes=2,
        n_train=1500, n_test=400, n_trials=5, n_epochs=50,
        strengths=STRENGTHS, device=device)
    
    p(f"\n  Experiment A completed in {elapsed_a:.0f}s ({elapsed_a/60:.1f} min)")
    pa_w_a, _, _, adv_a = print_results(results_a, STRENGTHS)

    # ---- EXPERIMENT B: 5-class (medium difficulty) ----
    results_b, elapsed_b = run_experiment(
        name="B: 5-Class Classification",
        text_dim=128, vision_dim=128, fusion_dim=256, n_classes=5,
        n_train=2000, n_test=500, n_trials=5, n_epochs=60,
        strengths=STRENGTHS, device=device)
    
    p(f"\n  Experiment B completed in {elapsed_b:.0f}s ({elapsed_b/60:.1f} min)")
    pa_w_b, _, _, adv_b = print_results(results_b, STRENGTHS)

    # ---- EXPERIMENT C: 10-class (full power) ----
    results_c, elapsed_c = run_experiment(
        name="C: 10-Class Classification (Full Power)",
        text_dim=256, vision_dim=256, fusion_dim=512, n_classes=10,
        n_train=3000, n_test=750, n_trials=3, n_epochs=80,
        strengths=STRENGTHS, device=device)
    
    p(f"\n  Experiment C completed in {elapsed_c:.0f}s ({elapsed_c/60:.1f} min)")
    pa_w_c, _, _, adv_c = print_results(results_c, STRENGTHS)

    # ---- GRAND SUMMARY ----
    total_elapsed = elapsed_a + elapsed_b + elapsed_c
    p(f"\n  {'=' * 66}")
    p(f"  GRAND SUMMARY")
    p(f"  {'=' * 66}")
    p(f"  Experiment A (2-class):  PA won {pa_w_a}/5 strengths, high-strength advantage: {adv_a:+.3f}")
    p(f"  Experiment B (5-class):  PA won {pa_w_b}/5 strengths, high-strength advantage: {adv_b:+.3f}")
    p(f"  Experiment C (10-class): PA won {pa_w_c}/5 strengths, high-strength advantage: {adv_c:+.3f}")
    p()
    total_pa_wins = pa_w_a + pa_w_b + pa_w_c
    p(f"  Total PA wins: {total_pa_wins}/15")
    
    if total_pa_wins >= 10:
        p("\n  ============================================")
        p("  CONCLUSION: HYPOTHESIS STRONGLY SUPPORTED")
        p("  Product Algebra Fusion captures cross-modal")
        p("  interactions that simpler methods miss.")
        p("  The Kronecker structure works.")
        p("  ============================================")
    elif total_pa_wins >= 7:
        p("\n  ============================================")
        p("  CONCLUSION: HYPOTHESIS PARTIALLY SUPPORTED")
        p("  PA shows meaningful advantage, especially")
        p("  at high cross-modal interaction strengths.")
        p("  ============================================")
    else:
        p("\n  ============================================")
        p("  CONCLUSION: INCONCLUSIVE")
        p("  More investigation needed.")
        p("  ============================================")

    p(f"\n  Total runtime: {total_elapsed/60:.1f} min on "
      f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    p()

    # Save all results
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'results_a': results_a,
        'results_b': results_b,
        'results_c': results_c,
    }, os.path.join(save_dir, 'benchmark_results_gpu.pt'))
    p("  Results saved to models/benchmark_results_gpu.pt")
    p()


if __name__ == '__main__':
    main()
