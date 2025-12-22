# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np

# Core SAGE Framework Imports
from graph.shared_concept_graph import SharedConceptGraph
from container.sage_container import SAGEContainer
from monitoring.sage_auditor import SageAuditor

# Stage Imports
from stages.infant_transformer import InfantTransformer
from stages.toddler_transformer import ToddlerTransformer
from stages.preschool_transformer import PreschoolTransformer
from stages.gradeschool_transformer import GradeschoolTransformer
from stages.teen_transformer import TeenTransformer
from stages.adult_transformer import AdultTransformer
from stages.sage_transformer import SageTransformer

# 12 pt Unicode compliant stage ordering
STAGE_ORDER = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Sage"]

class SensoryFrontend(nn.Module):
    """
    SAGE Frontend: Encodes raw sensory data into the Conceptual Manifold.
    """
    def __init__(self, input_dim=784, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 10)

    def forward(self, x, sage_container):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        # SAGE requires [seq_len, batch, dim]
        sage_input = latent.unsqueeze(0)

        # SAGE returns the Governed Latent and the Confidence Metric (Gamma)
        fused_latent, telemetry = sage_container(sage_input)

        # Map the governed manifold back to discrete classification
        return self.classifier(fused_latent.squeeze(0)), telemetry


def real_training(frontend, sage, auditor, total_tasks=5, epochs_per_task=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend.to(device)
    sage.to(device)

    # Note: stage_weight parameters are optimized alongside the frontend
    optimizer = optim.AdamW(list(frontend.parameters()) + list(sage.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for task_id in range(total_tasks):
        current_stage_name = STAGE_ORDER[sage.current_stage]
        print(f"\n--- TASK {task_id + 1}: {current_stage_name} ABSTRACTION LEVEL ---")
        loader = get_permuted_mnist_loader(task_id, train=True)

        frontend.train()
        sage.train()

        for epoch in range(epochs_per_task):
            loss_total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                logits, telemetry = frontend(x, sage)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                # TOPOLOGICAL REINFORCEMENT
                # Gamma (Confidence) threshold for Hebbian growth: > 0.7
                if telemetry['confidence'] > 0.7:
                    # Only reinforce if the Sage Stage provides a stable trace
                    sage.graph.update_stage_aware_hebbian(
                        attention_map=telemetry.get('trace'),
                        stage_plasticity=sage.eta
                    )

                loss_total += loss.item()

            # Maturation Parameter Decay (Decaying Synaptic Plasticity)
            sage.update_parameters(epoch)
            print(f"Epoch {epoch + 1} | Loss: {loss_total/len(loader):.4f} | Γ: {telemetry['confidence']:.3f} | η: {sage.eta:.3f}")

        # Post-Task Maturation Check
        avg_acc = evaluate_all_tasks(frontend, sage, task_id + 1, device)
        if sage.evaluate_gate({"pattern_acc": avg_acc}):
            print(f"[*] MATURATION EVENT: Promoting to {STAGE_ORDER[sage.current_stage]}")

    auditor.shutdown()

# Helper for Task Switching (MNIST Permutation)
def get_permuted_mnist_loader(task_id, train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root="data", train=train, download=True, transform=transform)
    gen = torch.Generator().manual_seed(42 + task_id)
    perm = torch.randperm(784, generator=gen)
    data = dataset.data.view(-1, 784).float()
    permuted_data = data[:, perm].view(-1, 1, 28, 28) / 255.0
    return DataLoader(TensorDataset(permuted_data, dataset.targets), batch_size=64 if train else 256, shuffle=train)

def evaluate_all_tasks(frontend, sage, num_seen, device):
    frontend.eval()
    sage.eval()
    all_accs = []
    with torch.no_grad():
        for tid in range(num_seen):
            loader = get_permuted_mnist_loader(tid, train=False)
            task_accs = [ (frontend(x.to(device), sage)[0].argmax(-1) == y.to(device)).float().mean().item() for x, y in loader ]
            all_accs.append(np.mean(task_accs))
    return np.mean(all_accs)

def main():
    # 1. TOPOLOGICAL INITIALIZATION
    # lambda_ewma=0.1 ensures stable temporal anchors
    graph = SharedConceptGraph(embedding_dim=128, lambda_ewma=0.1)
    for c in ["Linear-Edge", "Curvature", "Numerical-Manifold"]:
        graph.add_node(c)

    # 2. GOVERNANCE (Asynchronous Auditor)
    auditor = SageAuditor(graph, mode="SAGE_DELEGATED")

    # 3. HIERARCHICAL LADDER SETUP
    thresholds = {s: {"pattern_acc": 0.1 + (i*0.13)} for i, s in enumerate(STAGE_ORDER)}

    stage_models = {
        "Infant": InfantTransformer(),
        "Toddler": ToddlerTransformer(),
        "Preschool": PreschoolTransformer(),
        "Gradeschool": GradeschoolTransformer(),
        "Teen": TeenTransformer(),
        "Adult": AdultTransformer(),
        "Sage": SageTransformer()
    }

    # 4. CONTAINER ORCHESTRATION
    sage_container = SAGEContainer(graph, stage_models, thresholds, auditor)
    frontend = SensoryFrontend()

    print("[SYSTEM] SAGE Container initialized. Beginning Logarithmic Maturity Cycle.")
    try:
        real_training(frontend, sage_container, auditor)
    except KeyboardInterrupt:
        auditor.shutdown()

if __name__ == "__main__":
    main()