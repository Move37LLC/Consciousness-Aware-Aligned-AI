"""
Experiment H: Properly Controlled Experiment
=============================================

KEY INSIGHT from previous experiments:
- The biggest source of noise is DATA VARIATION between seeds
- Different seeds create dramatically different task difficulties
- This drowns out the fusion method differences

FIX: Separate data randomness from model randomness.
- Generate ONE fixed dataset per cross-modal strength
- Only vary the model initialization seed
- This is a PAIRED design: all models see the EXACT same data

Also uses optimal config from previous experiments:
- dim=32 (where PA showed +0.115 advantage in Exp G)
- rank=4 (best from rank ablation in Exp E)
- Compared against dim=128 to see if the effect scales

20 model-init trials per condition for robust statistics.
"""
import sys, os, io, time, math
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
    print(*args, **kwargs, flush=True)


def generate_data(n, text_dim, vision_dim, n_classes, cross_modal_strength, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    text_p = torch.randn(n_classes, text_dim) * 1.5
    vision_p = torch.randn(n_classes, vision_dim) * 1.5
    interact_p = torch.randn(n_classes, text_dim, vision_dim) * 0.3

    texts, visions, labels = [], [], []
    for _ in range(n):
        c = torch.randint(0, n_classes, (1,)).item()
        t = text_p[c] + torch.randn(text_dim) * 0.4
        v = vision_p[c] + torch.randn(vision_dim) * 0.4
        interaction = torch.outer(t, v)
        scores = torch.stack([(interaction * interact_p[k]).sum() for k in range(n_classes)])
        cm_label = scores.argmax().item()
        label = cm_label if np.random.random() < cross_modal_strength else c
        texts.append(t); visions.append(v); labels.append(label)

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
    def __init__(self, dims, fusion_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sum(dims), fusion_dim),
            nn.LayerNorm(fusion_dim), nn.GELU())

    def forward(self, states, **kw):
        cat = torch.cat([s.experience for s in states], dim=-1)
        return self.proj(cat), {}


class BilinearFusion(nn.Module):
    """Explicit bilinear baseline -- learns W such that output = text^T W vision."""
    def __init__(self, dims, fusion_dim):
        super().__init__()
        # Low-rank bilinear: project each to fusion_dim, then element-wise product
        self.proj_a = nn.Linear(dims[0], fusion_dim)
        self.proj_b = nn.Linear(dims[1], fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, states, **kw):
        a = self.proj_a(states[0].experience)
        b = self.proj_b(states[1].experience)
        out = self.norm(a * b)  # Element-wise product = bilinear interaction
        return out, {}


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
            idx = perm[i:i + batch_sz]
            opt.zero_grad()
            loss = F.cross_entropy(model(tr_t[idx], tr_v[idx]), tr_y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(te_y), batch_sz):
            pred = model(te_t[i:i + batch_sz], te_v[i:i + batch_sz]).argmax(dim=-1)
            all_preds.append(pred)
        preds = torch.cat(all_preds)
        acc = (preds == te_y).float().mean().item()
    return acc, sum(pp.numel() for pp in model.parameters())


def paired_ttest(a, b):
    a, b = np.array(a), np.array(b)
    d = a - b
    n = len(d)
    if n < 2: return 0.0, 1.0
    mean_d = d.mean()
    std_d = d.std(ddof=1)
    if std_d < 1e-12:
        return (float('inf') if mean_d > 0 else float('-inf')), 0.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    df = n - 1
    # Normal approximation for p-value
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    p_val = min(1.0, p_val * (1 + 0.5 / max(df, 1)))
    return t_stat, p_val


def effect_size(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = math.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    if pooled_std < 1e-12: return 0.0
    return (a.mean() - b.mean()) / pooled_std


def sig_stars(p_val):
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    if p_val < 0.10: return "."
    return ""


def run_controlled(dim, rank, n_classes, n_train, n_test, n_epochs, n_model_trials,
                   strengths, data_seed, device, label):
    """Run a controlled experiment with fixed data, varying only model init."""
    fusion_dim = dim * 2

    p(f"\n  {'=' * 70}")
    p(f"  {label}")
    p(f"  dim={dim}, fusion={fusion_dim}, rank={rank}, classes={n_classes}")
    p(f"  {n_train} train, {n_test} test, {n_epochs} epochs, {n_model_trials} model trials")
    p(f"  DATA SEED FIXED at {data_seed} -- only model init varies")
    p(f"  {'=' * 70}")
    p()

    all_results = {}
    t0 = time.time()

    for strength in strengths:
        p(f"  --- Cross-Modal Strength: {strength:.2f} " + "-" * 40)

        # FIXED data for this strength level
        tr_t, tr_v, tr_y = generate_data(
            n_train, dim, dim, n_classes, strength, data_seed)
        te_t, te_v, te_y = generate_data(
            n_test, dim, dim, n_classes, strength, data_seed + 9999)

        # Check label distribution
        unique, counts = torch.unique(tr_y, return_counts=True)
        balance = counts.float() / counts.sum()
        majority = balance.max().item()
        p(f"    Label balance: {n_classes} classes, majority={majority:.2f}")

        pa_accs, attn_accs, cat_accs, bilinear_accs = [], [], [], []

        for trial in range(n_model_trials):
            model_seed = 1000 + trial  # Only model init varies

            # Product Algebra (rank from ablation)
            torch.manual_seed(model_seed)
            pa_model = BenchmarkModel(
                ProductAlgebraFusion([dim, dim], fusion_dim,
                                    use_low_rank=True, rank=rank, preserve_markov=False),
                fusion_dim, n_classes)
            pa_acc, pa_params = train_eval(pa_model, tr_t, tr_v, tr_y,
                                           te_t, te_v, te_y, n_epochs, device=device)

            # Cross-Attention
            torch.manual_seed(model_seed)
            attn_model = BenchmarkModel(
                AttentionFusionBaseline([dim, dim], fusion_dim, n_heads=min(8, dim)),
                fusion_dim, n_classes)
            attn_acc, attn_params = train_eval(attn_model, tr_t, tr_v, tr_y,
                                                te_t, te_v, te_y, n_epochs, device=device)

            # Concatenation
            torch.manual_seed(model_seed)
            cat_model = BenchmarkModel(
                ConcatFusion([dim, dim], fusion_dim),
                fusion_dim, n_classes)
            cat_acc, cat_params = train_eval(cat_model, tr_t, tr_v, tr_y,
                                              te_t, te_v, te_y, n_epochs, device=device)

            # Bilinear (explicit bilinear baseline)
            torch.manual_seed(model_seed)
            bi_model = BenchmarkModel(
                BilinearFusion([dim, dim], fusion_dim),
                fusion_dim, n_classes)
            bi_acc, bi_params = train_eval(bi_model, tr_t, tr_v, tr_y,
                                            te_t, te_v, te_y, n_epochs, device=device)

            pa_accs.append(pa_acc)
            attn_accs.append(attn_acc)
            cat_accs.append(cat_acc)
            bilinear_accs.append(bi_acc)

            elapsed = time.time() - t0
            if (trial + 1) % 5 == 0 or trial == 0:
                p(f"    Trial {trial+1:2d}/{n_model_trials}:"
                  f" PA={pa_acc:.3f} Attn={attn_acc:.3f}"
                  f" Cat={cat_acc:.3f} Bilin={bi_acc:.3f}  [{elapsed:.0f}s]")

        # Statistics
        methods = {
            'Product Algebra': pa_accs,
            'Cross-Attention': attn_accs,
            'Concatenation': cat_accs,
            'Bilinear': bilinear_accs,
        }

        p(f"    Summary (n={n_model_trials}, same data):")
        for name, accs in methods.items():
            p(f"      {name:18s}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}"
              f"  [min={np.min(accs):.3f}, max={np.max(accs):.3f}]")

        # PA vs each other method
        for other_name, other_accs in methods.items():
            if other_name == 'Product Algebra':
                continue
            t_stat, p_val = paired_ttest(pa_accs, other_accs)
            d = effect_size(pa_accs, other_accs)
            adv = np.mean(pa_accs) - np.mean(other_accs)
            p(f"    PA vs {other_name:18s}: diff={adv:+.3f}, t={t_stat:+.2f},"
              f" p={p_val:.4f} {sig_stars(p_val):>3}, d={d:+.2f}")

        p()

        all_results[strength] = {
            'pa': pa_accs, 'attn': attn_accs, 'cat': cat_accs, 'bilinear': bilinear_accs,
            'params': {'pa': pa_params, 'attn': attn_params,
                       'cat': cat_params, 'bilinear': bi_params},
        }

    return all_results, time.time() - t0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p()
    p("  " + "=" * 70)
    p("  EXPERIMENT H: CONTROLLED EXPERIMENT SUITE")
    p("  Fixed data, varying only model initialization")
    p("  Using optimal config from Experiments E & G")
    p("  + Bilinear baseline (explicit interaction modeling)")
    p("  " + "=" * 70)
    if device == 'cuda':
        p(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.zeros(1, device='cuda')
        a = torch.randn(128, 128, device='cuda'); _ = a @ a.T; del a
        p("  CUDA ready.")
    p()

    total_t0 = time.time()
    STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
    N_MODEL_TRIALS = 20

    # ---- H1: Optimal config (dim=32, rank=4) ----
    results_h1, time_h1 = run_controlled(
        dim=32, rank=4, n_classes=2,
        n_train=3000, n_test=750, n_epochs=80,
        n_model_trials=N_MODEL_TRIALS, strengths=STRENGTHS,
        data_seed=42, device=device,
        label="H1: OPTIMAL CONFIG (dim=32, rank=4, binary)")

    # ---- H2: Same but with 5 classes ----
    results_h2, time_h2 = run_controlled(
        dim=32, rank=4, n_classes=5,
        n_train=4000, n_test=1000, n_epochs=80,
        n_model_trials=N_MODEL_TRIALS, strengths=STRENGTHS,
        data_seed=42, device=device,
        label="H2: OPTIMAL CONFIG, 5 CLASSES (dim=32, rank=4)")

    # ---- H3: Higher dim to see if effect holds ----
    results_h3, time_h3 = run_controlled(
        dim=128, rank=4, n_classes=2,
        n_train=3000, n_test=750, n_epochs=80,
        n_model_trials=N_MODEL_TRIALS, strengths=STRENGTHS,
        data_seed=42, device=device,
        label="H3: HIGH DIM WITH LOW RANK (dim=128, rank=4, binary)")

    total_elapsed = time.time() - total_t0

    # ============================================================
    # GRAND RESULTS
    # ============================================================
    p()
    p("  " + "=" * 70)
    p("  EXPERIMENT H: GRAND RESULTS")
    p("  " + "=" * 70)

    for label, results, config_desc in [
        ("H1 (dim=32, rank=4, 2-class)", results_h1, "Optimal from E+G"),
        ("H2 (dim=32, rank=4, 5-class)", results_h2, "Multi-class test"),
        ("H3 (dim=128, rank=4, 2-class)", results_h3, "Scalability test"),
    ]:
        p(f"\n  {label} [{config_desc}]")
        p(f"  {'Str':>5} | {'PA':>12} | {'Attn':>12} | {'Cat':>12} |"
          f" {'Bilinear':>12} | Winner")
        p("  " + "-" * 80)

        for s in STRENGTHS:
            r = results[s]
            pa_m = np.mean(r['pa'])
            at_m = np.mean(r['attn'])
            ct_m = np.mean(r['cat'])
            bi_m = np.mean(r['bilinear'])
            means = {'PA': pa_m, 'Attn': at_m, 'Cat': ct_m, 'Bilin': bi_m}
            winner = max(means, key=means.get)
            if winner == 'PA': winner = '<< PA'

            p(f"  {s:5.2f} | {pa_m:.3f}+/-{np.std(r['pa']):.3f}"
              f" | {at_m:.3f}+/-{np.std(r['attn']):.3f}"
              f" | {ct_m:.3f}+/-{np.std(r['cat']):.3f}"
              f" | {bi_m:.3f}+/-{np.std(r['bilinear']):.3f}"
              f" | {winner}")
        p("  " + "-" * 80)

        # Params
        r0 = results[STRENGTHS[0]]
        pp = r0['params']
        p(f"  Params: PA={pp['pa']:,} Attn={pp['attn']:,}"
          f" Cat={pp['cat']:,} Bilin={pp['bilinear']:,}")

    # Count significant results
    p(f"\n  {'=' * 70}")
    p(f"  STATISTICAL SIGNIFICANCE SUMMARY")
    p(f"  {'=' * 70}")

    total_sig = 0
    total_tests = 0
    for label, results in [("H1", results_h1), ("H2", results_h2), ("H3", results_h3)]:
        p(f"\n  {label}:")
        for s in STRENGTHS:
            r = results[s]
            for other_name, other_key in [("Attn", "attn"), ("Cat", "cat"), ("Bilinear", "bilinear")]:
                t_stat, p_val = paired_ttest(r['pa'], r[other_key])
                d = effect_size(r['pa'], r[other_key])
                adv = np.mean(r['pa']) - np.mean(r[other_key])
                total_tests += 1
                is_sig = p_val < 0.05 and adv > 0
                if is_sig: total_sig += 1
                marker = "SIG!" if is_sig else ""
                if abs(adv) > 0.03 or is_sig:
                    p(f"    s={s:.2f} PA vs {other_name:>7s}: {adv:+.3f}"
                      f"  p={p_val:.4f}{sig_stars(p_val):>3} d={d:+.2f} {marker}")

    p(f"\n  Significant PA wins: {total_sig}/{total_tests} tests")

    # Final
    p(f"\n  {'=' * 70}")
    if total_sig >= 5:
        p("  PRODUCT ALGEBRA: STATISTICALLY VALIDATED")
    elif total_sig >= 2:
        p("  PRODUCT ALGEBRA: PARTIALLY VALIDATED")
    else:
        p("  PRODUCT ALGEBRA: REQUIRES FURTHER INVESTIGATION")
        p("  (High variance in training suggests need for")
        p("   better optimization or cleaner signal generation)")
    p(f"  {'=' * 70}")

    p(f"\n  Total runtime: {total_elapsed/60:.1f} min")
    p(f"    H1: {time_h1/60:.1f} min | H2: {time_h2/60:.1f} min | H3: {time_h3/60:.1f} min")

    # Save
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'h1': results_h1, 'h2': results_h2, 'h3': results_h3,
        'total_runtime': total_elapsed,
    }, os.path.join(save_dir, 'controlled_experiment.pt'))
    p(f"\n  Results saved to models/controlled_experiment.pt")
    p()


if __name__ == '__main__':
    main()
