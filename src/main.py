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
        x_flat = x.view(-1, 784)
        latent = self.encoder(x_flat)
        sage_input = latent.unsqueeze(0)

        if graph_matrix is not None:
            if graph_matrix.dim() == 4:
                graph_matrix = graph_matrix.squeeze(0)
            if graph_matrix.dim() == 3 and graph_matrix.size(0) == x_flat.size(0):
                graph_matrix = graph_matrix.transpose(0, 1)

        fused_latent, telemetry = sage_container(sage_input, graph_matrix=graph_matrix)
        return self.classifier(fused_latent.squeeze(0)), telemetry


# -------------------------
# MANIFOLD VARIANCE
# -------------------------

def compute_manifold_variance(graph: SharedConceptGraph):
    variances = []

    for node_id, centroid in graph.anchor_tensor.items():
        if node_id not in graph.nodes:
            continue

        node = graph.nodes[node_id]
        if node.is_tombstoned or node.alignment_score <= 0.0:
            continue

        diffs = []

        for other in graph.nodes.values():
            if other.is_tombstoned or other.alignment_score <= 0.0:
                continue
            diffs.append(torch.sum((other.embedding - centroid) ** 2))

        if diffs:
            variances.append(torch.mean(torch.stack(diffs)))

    if not variances:
        return 0.0

    return torch.mean(torch.stack(variances)).item()


# -------------------------
# TRAINING
# -------------------------

def run_train_cycle(frontend, sage, auditor, analytics, stage_name, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    frontend.to(device)
    sage.to(device)

    if stage_name not in STAGE_ORDER:
        raise ValueError(f"Stage {stage_name} not found.")

    stage_idx = STAGE_ORDER.index(stage_name)
    sage.current_stage_idx = stage_idx
    sage.update_plasticity_window(cumulative=True)

    hparams = STAGE_HYPERPARAMS[stage_name]
    print(f"\n=== SAGE SURGICAL TRAINING: {stage_name.upper()} ===")
    print(f"[*] Layer Range: {hparams['layer_start']} - {hparams['layer_end']}")

    trainable_params = [p for p in sage.parameters() if p.requires_grad] + list(frontend.parameters())

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
        loss_total = 0.0

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

            if telemetry.get('confidence', 0.0) > 0.7:
                sage.graph.update_stage_aware_hebbian(
                    attention_map=telemetry.get('trace'),
                    stage_plasticity=hparams["plasticity_scale"]
                )

            loss_total += loss.item()

        avg_loss = loss_total / len(loader)

        snapshot = analytics.capture_snapshot(
            stage_name=stage_name,
            stage_idx=stage_idx,
            epoch=epoch,
            loss=avg_loss,
            telemetry=telemetry,
            graph=sage.graph
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Γ: {snapshot['metrics']['gamma_confidence']:.3f} | "
            f"Var: {snapshot['metrics']['manifold_variance']:.6f}"
        )

    checkpoint_path = f"checkpoints/SAGE_STATE_{stage_name}.pth"
    torch.save({
        'stage_name': stage_name,
        'frontend_state': frontend.state_dict(),
        'sage_state': sage.state_dict(),
        'graph_state': sage.graph.state_dict()
    }, checkpoint_path)

    analytics.save_stage_report(stage_name)
    print(f"[+] Surgical training complete. State saved: {checkpoint_path}")


# -------------------------
# DATA
# -------------------------

def get_sage_mnist_loader(stage_name, graph, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_data = datasets.MNIST(
        root="data", train=train, download=True, transform=transform
    )

    samples = []
    limit = 1000 if train else 500

    for i in range(len(mnist_data)):
        img, label = mnist_data[i]
        samples.append({'input_data': img, 'concepts': [label], 'target': label})
        if len(samples) >= limit:
            break

    def mock_tokenizer(img_tensor):
        return img_tensor.view(-1)

    return DataLoader(
        SAGEDataset(samples, graph, transform=mock_tokenizer),
        batch_size=64,
        shuffle=train,
        collate_fn=collate_sage_batch
    )


# -------------------------
# ENTRY
# -------------------------

def main():
    parser = argparse.ArgumentParser("SAGE Surgical Interface")
    parser.add_argument("mode", choices=["train", "test_manifold"])
    parser.add_argument("--stage", default="Infant")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--load", type=str)
    args = parser.parse_args()

    analytics = SAGEAnalyticsEngine()
    graph = SharedConceptGraph(embedding_dim=128)
    auditor = SageAuditor(graph, mode="SAGE_DELEGATED")
    frontend = SensoryFrontend()

    stage_models = {
        "Infant": InfantTransformer(),
        "Toddler": ToddlerTransformer(),
        "Preschool": PreschoolTransformer(),
        "Gradeschool": GradeschoolTransformer(),
        "Teen": TeenTransformer(),
        "Adult": AdultTransformer(),
        "Elder": ElderTransformer(),
    }

    thresholds = {s: {"pattern_acc": 0.8} for s in STAGE_ORDER}
    sage_container = SAGEContainer(graph, stage_models, thresholds, auditor)

    if args.load:
        path = f"checkpoints/SAGE_STATE_{args.load.capitalize()}.pth"
        ckpt = torch.load(path, map_location="cpu")
        frontend.load_state_dict(ckpt['frontend_state'])
        sage_container.load_state_dict(ckpt['sage_state'], strict=False)
        graph.load_state_dict(ckpt['graph_state'])
        print(f"[*] Loaded checkpoint: {path}")
    else:
        seeder = SAGEGraphSeeder(graph, frontend)
        seeder.seed_from_sensory_patterns({i: torch.rand(1, 784) for i in range(10)})

    if args.mode == "train":
        if args.stage.lower() == "all":
            start_idx = STAGE_ORDER.index(args.load.capitalize()) if args.load else 0
            for stage_idx in range(start_idx, len(STAGE_ORDER)):
                sage_container.current_stage_idx = stage_idx
                sage_container.update_plasticity_window(cumulative=True)
                stage_name = STAGE_ORDER[stage_idx]
                run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args)
        else:
            run_train_cycle(frontend, sage_container, auditor, analytics, args.stage.capitalize(), args)

    elif args.mode == "test_manifold":
        var = compute_manifold_variance(graph)
        print(f"[MANIFOLD] Conceptual Variance: {var:.6f}")

    auditor.shutdown()


if __name__ == "__main__":
    main()
