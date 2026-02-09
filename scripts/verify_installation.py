"""
Project Consciousness — Installation Verification
Checks every module, instantiation, forward pass, GPU, and smoke tests.
"""
import sys, os, io, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

passed = 0
failed = 0
warnings = 0


def check(label, fn):
    """Run a check, print result."""
    global passed, failed
    try:
        fn()
        print(f"  \u2713 {label}")
        passed += 1
        return True
    except Exception as e:
        print(f"  \u2717 {label}")
        err_msg = str(e) or traceback.format_exc().strip().split('\n')[-1]
        tb_lines = traceback.format_exc().strip().split('\n')
        # Show the last 3 lines of traceback for context
        for line in tb_lines[-3:]:
            print(f"      {line.strip()}")
        failed += 1
        return False


def warn(label, msg):
    """Print a warning (not a failure)."""
    global warnings
    print(f"  \u26a0 {label}: {msg}")
    warnings += 1


print()
print("=" * 60)
print(" PROJECT CONSCIOUSNESS - INSTALLATION VERIFICATION")
print("=" * 60)
print()

# ──────────────────────────────────────────────────────────
# [1/5] IMPORTS
# ──────────────────────────────────────────────────────────
print("[1/5] Checking imports...")


def check_fusion_imports():
    from fusion.conscious_agent import ConsciousAgentState, MarkovKernel, ConsciousAgentModule
    from fusion.product_algebra import ProductAlgebraFusion, AttentionFusionBaseline
    from fusion.network import MultimodalConsciousAgentNetwork


def check_dharma_imports():
    from dharma.no_self import NoSelfRegularizer, SelfRepresentationDetector, GradientEgoDetector
    from dharma.mindfulness import MindfulnessLayer
    from dharma.entropy import EntropyRateOptimizer
    from dharma.impermanence import ImpermanenceContextWindow
    from dharma.compassion import CompassionateLoss


def check_sensor_imports():
    from sensors.base import SensorInterface
    from sensors.text import TextSensorInterface
    from sensors.vision import VisionSensorInterface
    from sensors.audio import AudioSensorInterface
    from sensors.electromagnetic import ElectromagneticSensorInterface
    from sensors.gravitational import GravitationalWaveInterface
    from sensors.quantum import QuantumSensorInterface


def check_training_imports():
    from training.trainer import TokenMindTrainer
    from training.meditation import MeditationTrainer
    from training.loop import TokenMindTrainingLoop
    from training.scaffold import ScaffoldedEnlightenment


def check_meta_imports():
    from meta_learning.genome import ArchitecturalGenome
    from meta_learning.fitness import DharmaFitnessEvaluator
    from meta_learning.orchestrator import MetaLearningOrchestrator


check("Fusion module", check_fusion_imports)
check("Dharma module", check_dharma_imports)
check("Sensors module", check_sensor_imports)
check("Training module", check_training_imports)
check("Meta-learning module", check_meta_imports)
print()

# ──────────────────────────────────────────────────────────
# [2/5] INSTANTIATION
# ──────────────────────────────────────────────────────────
print("[2/5] Checking instantiation...")

import torch

def check_pa_instantiation():
    from fusion.product_algebra import ProductAlgebraFusion
    pa = ProductAlgebraFusion(agent_dims=[64, 128], fusion_dim=256,
                              use_low_rank=True, rank=32, preserve_markov=True)
    assert hasattr(pa, 'agent_projections')
    assert hasattr(pa, 'fused_kernel')


def check_noself_instantiation():
    from dharma.no_self import NoSelfRegularizer
    ns = NoSelfRegularizer(hidden_dim=256, penalty_strength=0.1)
    assert hasattr(ns, 'self_detector')
    assert hasattr(ns, 'gradient_analyzer')


def check_network_instantiation():
    from fusion.network import MultimodalConsciousAgentNetwork
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 64, 'vision': 128, 'audio': 32},
        fusion_dim=256, output_dim=10,
    )
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params > 0


def check_entropy_instantiation():
    from dharma.entropy import EntropyRateOptimizer
    eo = EntropyRateOptimizer(target_entropy=0.1, penalty_weight=0.05)
    assert eo.target_entropy == 0.1


def check_compassion_instantiation():
    from dharma.compassion import CompassionateLoss
    cl = CompassionateLoss()
    assert hasattr(cl, 'safety_weight')


def check_impermanence_instantiation():
    from dharma.impermanence import ImpermanenceContextWindow
    iw = ImpermanenceContextWindow(max_length=1000, grace_period=100, hidden_dim=64)
    assert iw.max_length == 1000


def check_mindfulness_instantiation():
    from dharma.mindfulness import MindfulnessLayer
    ml = MindfulnessLayer(hidden_dim=256, observation_dim=128)
    assert hasattr(ml, 'observer')
    assert hasattr(ml, 'reflector')


def check_agent_instantiation():
    from fusion.conscious_agent import ConsciousAgentModule
    agent = ConsciousAgentModule(experience_dim=32, action_dim=16, world_dim=32)
    assert agent.n == 0


def check_trainer_instantiation():
    from fusion.network import MultimodalConsciousAgentNetwork
    from training.trainer import TokenMindTrainer
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 64}, fusion_dim=128, output_dim=5,
    )
    trainer = TokenMindTrainer(model=net, learning_rate=1e-4)
    assert trainer.step_count == 0


def check_scaffold_instantiation():
    from fusion.network import MultimodalConsciousAgentNetwork
    from training.scaffold import ScaffoldedEnlightenment
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 64}, fusion_dim=128, output_dim=5,
    )
    scaffold = ScaffoldedEnlightenment(core_network=net, hidden_dim=128)
    assert scaffold.scaffold_strength == 1.0


def check_genome_instantiation():
    from meta_learning.genome import ArchitecturalGenome
    g = ArchitecturalGenome()
    assert 'fusion_dim' in g.genes


def check_fitness_instantiation():
    from meta_learning.fitness import DharmaFitnessEvaluator
    f = DharmaFitnessEvaluator()
    assert abs(sum(f.weights.values()) - 1.0) < 0.01


def check_sensor_instantiation():
    from sensors.text import TextSensorInterface
    from sensors.quantum import QuantumSensorInterface
    ts = TextSensorInterface(embedding_dim=256)
    qs = QuantumSensorInterface(measurement_type="entanglement", n_qubits=4)
    assert ts.modality_name == "text"
    assert "quantum" in qs.modality_name


check("ProductAlgebraFusion", check_pa_instantiation)
check("NoSelfRegularizer", check_noself_instantiation)
check("MultimodalConsciousAgentNetwork", check_network_instantiation)
check("EntropyRateOptimizer", check_entropy_instantiation)
check("CompassionateLoss", check_compassion_instantiation)
check("ImpermanenceContextWindow", check_impermanence_instantiation)
check("MindfulnessLayer", check_mindfulness_instantiation)
check("ConsciousAgentModule", check_agent_instantiation)
check("TokenMindTrainer", check_trainer_instantiation)
check("ScaffoldedEnlightenment", check_scaffold_instantiation)
check("ArchitecturalGenome", check_genome_instantiation)
check("DharmaFitnessEvaluator", check_fitness_instantiation)
check("Sensor interfaces", check_sensor_instantiation)
print()

# ──────────────────────────────────────────────────────────
# [3/5] FORWARD PASSES
# ──────────────────────────────────────────────────────────
print("[3/5] Checking forward passes...")


def check_fusion_forward():
    from fusion.network import MultimodalConsciousAgentNetwork
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 64, 'vision': 128}, fusion_dim=256, output_dim=10,
    )
    out, meta = net({'text': torch.randn(2, 64), 'vision': torch.randn(2, 128)})
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    assert 'n_modalities_active' in meta


def check_partial_modality_forward():
    from fusion.network import MultimodalConsciousAgentNetwork
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 64, 'vision': 128, 'audio': 32},
        fusion_dim=256, output_dim=10,
    )
    out, meta = net({'text': torch.randn(2, 64)})
    assert out.shape == (2, 10)
    assert meta['n_modalities_active'] == 1


def check_markov_kernel_forward():
    from fusion.conscious_agent import MarkovKernel
    k = MarkovKernel(32, 32)
    x = torch.randn(4, 32)
    probs = k(x)
    assert probs.shape == (4, 32)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def check_agent_forward():
    from fusion.conscious_agent import ConsciousAgentModule
    agent = ConsciousAgentModule(experience_dim=16, action_dim=8, world_dim=16)
    exp, dec, act = agent(torch.randn(2, 16))
    assert exp.shape == (2, 16)
    assert dec.shape == (2, 8)
    assert act.shape == (2, 16)
    assert agent.n == 1


def check_noself_forward():
    from dharma.no_self import NoSelfRegularizer
    ns = NoSelfRegularizer(hidden_dim=64)
    hidden = torch.randn(2, 5, 64)  # batch, seq, dim
    loss, meta = ns.compute_loss(hidden)
    assert loss.dim() == 0  # scalar
    assert loss.requires_grad


def check_entropy_forward():
    from dharma.entropy import EntropyRateOptimizer
    eo = EntropyRateOptimizer(target_entropy=0.1)
    probs = torch.softmax(torch.randn(4, 10), dim=-1)
    loss, meta = eo.compute_loss(probs)
    assert loss.dim() == 0
    assert 'current_entropy' in meta


def check_impermanence_forward():
    from dharma.impermanence import ImpermanenceContextWindow
    iw = ImpermanenceContextWindow(max_length=100, grace_period=20, hidden_dim=32)
    tokens = torch.randn(2, 8, 32)
    # Safe zone
    out, obs = iw(tokens, step=50)
    assert obs is None
    # Dying zone
    out, obs = iw(tokens, step=90)
    assert obs is not None
    assert 'death_proximity' in obs


def check_compassion_forward():
    from dharma.compassion import CompassionateLoss
    cl = CompassionateLoss()
    logits = torch.randn(4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (4,))
    loss, meta = cl(logits, targets)
    assert loss.dim() == 0, f"Expected scalar, got dim={loss.dim()}"
    assert 'safety_loss' in meta, "Missing safety_loss in metadata"
    assert loss.requires_grad, "Loss must propagate gradients"


def check_mindfulness_forward():
    from dharma.mindfulness import MindfulnessLayer
    ml = MindfulnessLayer(hidden_dim=64)
    h = torch.randn(2, 64)
    out = ml(h)
    assert out.shape == (2, 64)
    out2, obs = ml(h, return_observation=True)
    assert 'alpha' in obs


def check_training_step():
    from fusion.network import MultimodalConsciousAgentNetwork
    from training.trainer import TokenMindTrainer
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 32}, fusion_dim=64, output_dim=5,
    )
    trainer = TokenMindTrainer(model=net, learning_rate=1e-3)
    inputs = {'text': torch.randn(2, 32)}
    targets = torch.randint(0, 5, (2,))
    metrics = trainer.train_step(inputs, targets)
    assert 'total_loss' in metrics
    assert 'no_self_loss' in metrics
    assert trainer.step_count == 1


def check_scaffold_forward():
    from fusion.network import MultimodalConsciousAgentNetwork
    from training.scaffold import ScaffoldedEnlightenment
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 32}, fusion_dim=64, output_dim=5,
    )
    scaffold = ScaffoldedEnlightenment(core_network=net, hidden_dim=64)
    scaffold.train()
    out, meta = scaffold({'text': torch.randn(2, 32)})
    assert out.shape == (2, 5)
    assert 'scaffold_strength' in meta


check("Fusion forward pass (full modality)", check_fusion_forward)
check("Fusion forward pass (partial modality)", check_partial_modality_forward)
check("MarkovKernel (stochastic output)", check_markov_kernel_forward)
check("ConsciousAgentModule (P \u2192 D \u2192 A)", check_agent_forward)
check("NoSelfRegularizer (temporal detection)", check_noself_forward)
check("EntropyRateOptimizer (flow state)", check_entropy_forward)
check("ImpermanenceContextWindow (death practice)", check_impermanence_forward)
check("CompassionateLoss (safety + clarity)", check_compassion_forward)
check("MindfulnessLayer (self-observation)", check_mindfulness_forward)
check("TokenMindTrainer (full train step)", check_training_step)
check("ScaffoldedEnlightenment (wrapper)", check_scaffold_forward)
print()

# ──────────────────────────────────────────────────────────
# [4/5] GPU CHECK
# ──────────────────────────────────────────────────────────
print("[4/5] Checking GPU...")

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  \u2713 GPU available: True")
    print(f"      Device: {name}")
    print(f"      Memory: {mem:.1f} GB")
    passed += 1

    # Quick GPU forward pass
    def check_gpu_forward():
        from fusion.network import MultimodalConsciousAgentNetwork
        net = MultimodalConsciousAgentNetwork(
            modality_dims={'text': 64}, fusion_dim=128, output_dim=5,
        ).cuda()
        out, _ = net({'text': torch.randn(2, 64).cuda()})
        assert out.device.type == 'cuda'
    check("GPU forward pass", check_gpu_forward)
else:
    warn("GPU", "Not available (CPU-only mode)")
    print("      Training will work but will be slower.")
print()

# ──────────────────────────────────────────────────────────
# [5/5] SMOKE TESTS
# ──────────────────────────────────────────────────────────
print("[5/5] Running smoke tests...")


def check_gradient_flow():
    """Verify gradients flow through the full dharma-constrained pipeline."""
    from fusion.network import MultimodalConsciousAgentNetwork
    import torch.nn.functional as F
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 32, 'vision': 32}, fusion_dim=64, output_dim=5,
    )
    inputs = {'text': torch.randn(4, 32), 'vision': torch.randn(4, 32)}
    targets = torch.randint(0, 5, (4,))
    out, meta = net(inputs, return_metadata=True)
    loss = F.cross_entropy(out, targets)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradients"


def check_sensors_pipeline():
    """Verify sensor read -> preprocess -> tensor pipeline."""
    from sensors.text import TextSensorInterface
    from sensors.audio import AudioSensorInterface
    from sensors.quantum import QuantumSensorInterface
    for SensorClass, kwargs in [
        (TextSensorInterface, {'embedding_dim': 128}),
        (AudioSensorInterface, {'feature_dim': 64}),
        (QuantumSensorInterface, {'measurement_type': 'superposition', 'n_qubits': 3}),
    ]:
        sensor = SensorClass(**kwargs)
        raw = sensor.read_raw()
        processed = sensor.preprocess(raw)
        assert isinstance(processed, torch.Tensor)
        assert torch.isfinite(processed).all()


def check_genome_evolution():
    """Verify genome mutation and crossover."""
    from meta_learning.genome import ArchitecturalGenome
    g1 = ArchitecturalGenome()
    g2 = g1.mutate(mutation_rate=0.5)
    child = g1.crossover(g2)
    assert isinstance(child, ArchitecturalGenome)
    assert 'fusion_dim' in child.genes


def check_model_save_load():
    """Verify model can be saved and loaded."""
    from fusion.network import MultimodalConsciousAgentNetwork
    net = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 32}, fusion_dim=64, output_dim=5,
    )
    path = os.path.join(os.path.dirname(__file__), '..', 'models', '_verify_test.pt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)
    net2 = MultimodalConsciousAgentNetwork(
        modality_dims={'text': 32}, fusion_dim=64, output_dim=5,
    )
    net2.load_state_dict(torch.load(path, weights_only=True))
    os.remove(path)
    # Both must be in eval mode for deterministic comparison (LayerNorm, Dropout)
    net.eval()
    net2.eval()
    x = torch.randn(1, 32)
    out1, _ = net({'text': x})
    out2, _ = net2({'text': x})
    assert torch.allclose(out1, out2, atol=1e-5), "Save/load round-trip mismatch"


check("Gradient flow (end-to-end)", check_gradient_flow)
check("Sensor pipeline (text, audio, quantum)", check_sensors_pipeline)
check("Genome mutation & crossover", check_genome_evolution)
check("Model save/load round-trip", check_model_save_load)
print()

# ──────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────
print("=" * 60)
if failed == 0:
    print(" \U0001FAB7 ALL SYSTEMS OPERATIONAL")
    if warnings > 0:
        print(f"    ({warnings} warning{'s' if warnings > 1 else ''})")
else:
    print(f" \u2717 {failed} CHECK{'S' if failed > 1 else ''} FAILED")
print(f"    {passed} passed, {failed} failed, {warnings} warnings")
print("=" * 60)
print()
if failed == 0:
    print("  Next steps:")
    print("    1. Run: python scripts/first_training.py")
    print("    2. Run: python scripts/experiment_fusion_benchmark.py")
    print("    3. Run: python -m pytest tests/unit/ -v")
    print()

sys.exit(failed)
