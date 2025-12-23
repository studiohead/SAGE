import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Config & SAGE Core
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS
from src.graph.shared_concept_graph import SharedConceptGraph
from src.container.sage_container import SAGEContainer
from src.monitoring.sage_auditor import SageAuditor
from src.graph.graph_seeder import SAGEGraphSeeder
from src.monitoring.analytics_engine import SAGEAnalyticsEngine

# Data & Stage Imports
from data.sage_dataloader import SAGEDataset, collate_sage_batch
from src.stages.infant_transformer import InfantTransformer
from src.stages.toddler_transformer import ToddlerTransformer
from src.stages.preschool_transformer import PreschoolTransformer
from src.stages.gradeschool_transformer import GradeschoolTransformer
from src.stages.teen_transformer import TeenTransformer
from src.stages.adult_transformer import AdultTransformer
from src.stages.elder_transformer import ElderTransformer

# Aligned with config.py keys
STAGE_ORDER = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Elder"]


class SensoryFrontend(nn.Module):
    """SAGE Frontend: Standardizes input shapes for the Conceptual Manifold."""

    def __init__(self, input_dim=784, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 10)

    def forward(self, x, sage_container, graph_matrix=None):
        # 1. Standardize Input to 2D [Batch, 784]
        # This resolves any DataLoader/Seeder wrapper dims [1, 1, 64, 784] -> [64, 784]
        x_flat = x.view(-1, 784)
        batch_size = x_flat.size(0)

        # 2. Encode to Latent [Batch, 128]
        latent = self.encoder(x_flat)

        # 3. PROMOTE TO SAGE STANDARD: [Sequence=1, Batch, Dim]
        # Multi-Head Attention strictly requires 3D (or unbatched 2D).
        sage_input = latent.unsqueeze(0)

        # 4. Align Graph Matrix: [Nodes, Batch, Dim]
        if graph_matrix is not None:
            # Remove wrapper dim if present [1, Batch, Nodes, Dim]
            if graph_matrix.dim() == 4:
                graph_matrix = graph_matrix.squeeze(0)

            # Transpose if Batch is at dim 0: [Batch, Nodes, Dim] -> [Nodes, Batch, Dim]
            if graph_matrix.dim() == 3 and graph_matrix.size(0) == batch_size:
                graph_matrix = graph_matrix.transpose(0, 1)

        # 5. CALL CONTAINER
        # Now that InfantTransformer is fixed, fused_latent will be 3D [1, Batch, Dim]
        fused_latent, telemetry = sage_container(sage_input, graph_matrix=graph_matrix)

        # 6. Final Output Recovery: Squeeze out Sequence for classifier [Batch, 10]
        # We use a standard squeeze(0) to ensure we have [Batch, Dim]
        return self.classifier(fused_latent.squeeze(0)), telemetry


# --- OPERATIONAL MODES ---

def run_train_cycle(frontend, sage, auditor, analytics, stage_name, args):
    """Surgical Stage Training with Integrated Analytics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    frontend.to(device)
    sage.to(device)

    if stage_name not in STAGE_ORDER:
        raise ValueError(f"Stage {stage_name} not found.")

    stage_idx = STAGE_ORDER.index(stage_name)
    sage.current_stage_idx = stage_idx
    sage.update_plasticity_window()

    hparams = STAGE_HYPERPARAMS[stage_name]
    print(f"\n=== SAGE SURGICAL TRAINING: {stage_name.upper()} ===")
    print(f"[*] Layer Range: {hparams['layer_start']} - {hparams['layer_end']}")

    trainable_params = [p for p in frontend.parameters() if p.requires_grad] + \
                       [p for p in sage.parameters() if p.requires_grad]

    optimizer = optim.AdamW(
        trainable_params,
        lr=hparams["learning_rate"],
        weight_decay=hparams.get("weight_decay", 0.01)
    )
    criterion = nn.CrossEntropyLoss()

    loader = get_sage_mnist_loader(stage_name, sage.graph, train=True)

    for epoch in range(args.epochs):
        frontend.train()
        sage.train()
        loss_total = 0

        for batch in loader:
            x = batch['input_ids'].to(device)
            g_matrix = batch['graph_matrix'].to(device)
            y = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, telemetry = frontend(x, sage, graph_matrix=g_matrix)

            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, hparams.get("gradient_clip", 1.0))
            optimizer.step()

            if telemetry.get('confidence', 0) > 0.7:
                sage.graph.update_stage_aware_hebbian(
                    attention_map=telemetry.get('trace'),
                    stage_plasticity=hparams["plasticity_scale"]
                )
            loss_total += loss.item()

        avg_loss = loss_total / len(loader)
        snapshot = analytics.capture_snapshot(
            stage_name=stage_name, stage_idx=stage_idx, epoch=epoch,
            loss=avg_loss, telemetry=telemetry, graph=sage.graph
        )

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Conf: {snapshot['metrics']['gamma_confidence']:.3f} | "
              f"Var: {snapshot['metrics']['manifold_variance']:.6f}")

    checkpoint_path = f"checkpoints/SAGE_STATE_{stage_name}.pth"
    torch.save({
        'stage_name': stage_name,
        'frontend_state': frontend.state_dict(),
        'sage_state': sage.state_dict(),
        'graph_state': sage.graph.state_dict()
    }, checkpoint_path)

    analytics.save_stage_report(stage_name)
    print(f"[+] Surgical training complete. State saved: {checkpoint_path}")


# --- HELPERS ---

def get_sage_mnist_loader(stage_name, graph, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_data = datasets.MNIST(root="data", train=train, download=True, transform=transform)

    allowed_targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if stage_name == "Infant" else list(range(10))
    samples = []
    limit = 1000 if train else 500

    for i in range(len(mnist_data)):
        img, label = mnist_data[i]
        if label in allowed_targets:
            samples.append({'input_data': img, 'concepts': [label], 'target': label})
        if len(samples) >= limit: break

    def mock_tokenizer(img_tensor):
        return img_tensor.view(-1)

    return DataLoader(
        SAGEDataset(samples, graph, transform=mock_tokenizer),
        batch_size=64, shuffle=train, collate_fn=collate_sage_batch
    )


def main():
    parser = argparse.ArgumentParser(description="SAGE Surgical Interface")
    parser.add_argument("mode", type=str, choices=["train", "test_manifold"])
    parser.add_argument("--stage", type=str, default="Infant")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--load", type=str)
    args = parser.parse_args()

    target_stage = args.stage.capitalize()
    analytics = SAGEAnalyticsEngine()
    graph = SharedConceptGraph(embedding_dim=128)
    auditor = SageAuditor(graph, mode="SAGE_DELEGATED")
    frontend = SensoryFrontend()

    stage_models = {
        "Infant": InfantTransformer(), "Toddler": ToddlerTransformer(),
        "Preschool": PreschoolTransformer(), "Gradeschool": GradeschoolTransformer(),
        "Teen": TeenTransformer(), "Adult": AdultTransformer(), "Elder": ElderTransformer()
    }

    thresholds = {s: {"pattern_acc": 0.8} for s in STAGE_ORDER}
    sage_container = SAGEContainer(graph, stage_models, thresholds, auditor)

    if args.load:
        load_source = args.load.capitalize()
        path = f"checkpoints/SAGE_STATE_{load_source}.pth"
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            frontend.load_state_dict(ckpt['frontend_state'])
            sage_container.load_state_dict(ckpt['sage_state'], strict=False)
            graph.load_state_dict(ckpt['graph_state'])
            print(f"[*] Inherited stable foundation from: {load_source}")
    else:
        print("[*] Starting fresh. Seeding manifold from sensory patterns...")
        seeder = SAGEGraphSeeder(graph, frontend)

        # Expanded to 10 seeds for a full 0-9 Infant foundation
        seeder.seed_from_sensory_patterns({
            0: torch.ones(1, 784),
            1: torch.zeros(1, 784),
            2: torch.randn(1, 784),
            3: torch.full((1, 784), 0.5),
            4: torch.full((1, 784), -0.5),
            5: torch.linspace(-1, 1, 784).view(1, -1),
            6: torch.linspace(1, -1, 784).view(1, -1),
            7: torch.sin(torch.linspace(0, 3.14, 784)).view(1, -1),
            8: torch.cos(torch.linspace(0, 3.14, 784)).view(1, -1),
            9: torch.rand(1, 784)
        })

    if args.mode == "train":
        run_train_cycle(frontend, sage_container, auditor, analytics, target_stage, args)
    elif args.mode == "test_manifold":
        embeddings = [node.embedding.data for node in graph.nodes.values()]
        if embeddings:
            all_embs = torch.stack(embeddings)
            variance = torch.var(all_embs, dim=0).mean().item()
            print(f"[-] Node Variance: {variance:.6f}")

    auditor.shutdown()


if __name__ == "__main__":
    main()