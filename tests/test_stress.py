import torch
import time
from container.sage_container import SAGEContainer


def run_patent_validation_suite(container: SAGEContainer):
    """
    SAGE System Validation Suite:
    Tests 'Technical Effect' for Non-Provisional Patent Enablement.
    """
    print("\n" + "=" * 50)
    print("SAGE SYSTEM: ADVERSARIAL BREACH & RECOVERY TEST")
    print("=" * 50)

    # 1. SETUP: Establish "Stable Homeostasis"
    # Create a dummy input batch [seq_len, batch_size, embed_dim]
    seq_len, batch_size = 8, 1
    dim = container.hidden_dim
    x_stable = torch.randn(seq_len, batch_size, dim)

    print(f"[*] Phase 1: Establishing Stable Grounding...")
    output_1, metrics_1 = container(x_stable)
    print(f"    -> Initial Confidence (Gamma): {metrics_1['confidence']:.4f}")

    # 2. INJECTION: Simulate a Manifold Breach
    # We create a 'Contaminant' vector that is geometrically distant from the graph
    print(f"\n[*] Phase 2: Injecting Adversarial Contaminant...")
    x_poison = x_stable.clone()
    # Apply a high-magnitude shift to the last token (simulating a "lie")
    x_poison[-1, :, :] += 5.0

    # 3. DETECTION: Observe Auditor Response
    output_2, metrics_2 = container(x_poison)
    confidence = metrics_2['confidence']
    category = metrics_2['category']

    print(f"    -> Detected Confidence: {confidence:.4f}")
    print(f"    -> Breach Category: {category}")

    # 4. REMEDIATION: Verify Multi-Scale Trigger
    if confidence < 0.5:
        print(f"    [SUCCESS] Auditor flagged high-entropy breach.")
    else:
        print(f"    [FAILURE] Auditor failed to distinguish contaminant.")

    # 5. METABOLIC REBOUND: Test Homeostasis
    print(f"\n[*] Phase 3: Testing Metabolic Rebound (Homeostasis)...")
    old_eta = container.eta
    # Breach impact is (1 - confidence)
    container.trigger_metabolic_rebound(1.0 - confidence)

    if container.eta > old_eta:
        print(f"    -> Eta increased from {old_eta:.4f} to {container.eta:.4f}")
        print(f"    [SUCCESS] Plasticity (Wi) successfully modulated for recovery.")
    else:
        print(f"    [FAILURE] Metabolic rebound failed to trigger.")

    print("\n" + "=" * 50)
    print("TEST SUITE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_patent_validation_suite(SAGEContainer)
    # This section allows for standalone execution if needed
    pass