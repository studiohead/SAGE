import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import data.seeder_concepts
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
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 10)

    def forward(self, x, sage_container, graph_matrix=None):
        if x.dim() == 4:  # MNIST: [B, C, H, W]
            x_flat = x.view(x.size(0), -1)
        else:  # Text: [B, max_tokens]
            x_flat = x  # Already flattened to fixed length (784)

        latent = self.encoder(x_flat)
        sage_input = latent.unsqueeze(0)

        if graph_matrix is not None:
            if graph_matrix.dim() == 4:
                graph_matrix = graph_matrix.squeeze(0)
            if graph_matrix.dim() == 3 and graph_matrix.size(0) == x_flat.size(0):
                graph_matrix = graph_matrix.transpose(0, 1)

        fused_latent, telemetry = sage_container(sage_input, graph_matrix=graph_matrix)
        return self.classifier(latent), telemetry


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

def run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args, governor, loader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    frontend.to(device)
    sage_container.to(device)

    stage_idx = STAGE_ORDER.index(stage_name.capitalize())
    sage_container.current_stage_idx = stage_idx
    sage_container.update_plasticity_window(cumulative=True)

    hparams = STAGE_HYPERPARAMS[stage_name.capitalize()]
    print(f"\n=== SAGE SURGICAL TRAINING: {stage_name.upper()} ===")
    print(f"[*] Layer Range: {hparams['layer_start']} - {hparams['layer_end']}")

    trainable_params = [p for p in sage_container.parameters() if p.requires_grad] + list(frontend.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=hparams["learning_rate"],
        weight_decay=hparams.get("weight_decay", 0.01)
    )
    criterion = nn.CrossEntropyLoss()

    if loader is None:
        if args.data == "mnist":
            loader = get_sage_mnist_loader(stage_name.lower(), sage_container.graph, train=True)
        else:
            raise ValueError("DataLoader must be provided for non-MNIST data")

    for epoch in range(args.epochs):
        frontend.train()
        sage_container.train()
        loss_total = 0.0
        avg_confidence = 0.0
        batch_count = 0

        for batch in loader:
            x = batch['input_ids'].to(device)

            if args.data == "text" and x.shape[1] != 784:
                padded = torch.zeros((x.size(0), 784), device=device)
                padded[:, :x.size(1)] = x[:, :784]
                x = padded

            g_matrix = batch['graph_matrix'].to(device)
            y = batch['labels'].to(device) if args.data == "mnist" else None

            optimizer.zero_grad()
            logits, telemetry = frontend(x, sage_container, graph_matrix=g_matrix)

            batch_count += 1
            avg_confidence += telemetry.get('confidence', 0.0)

            if args.data == "mnist":
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, hparams.get("gradient_clip", 1.0))
                optimizer.step()
                loss_total += loss.item()
            else:
                if telemetry.get('confidence', 0.0) > 0.7:
                    sage_container.graph.update_stage_aware_hebbian(
                        attention_map=telemetry.get('trace'),
                        stage_plasticity=hparams["plasticity_scale"]
                    )

        avg_loss = loss_total / len(loader) if args.data == "mnist" else 0.0
        avg_confidence /= max(batch_count, 1)
        print(f"AVG Loss: {avg_loss}")
        print(f"AVG Confidence: {avg_confidence}")

        live_graph = getattr(sage_container, "graph", None)
        snapshot = analytics.capture_snapshot(
            stage_name=stage_name,
            stage_idx=stage_idx,
            epoch=epoch,
            loss=avg_loss,
            telemetry=telemetry,
            graph=live_graph
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Γ: {snapshot['metrics']['gamma_confidence']:.3f} | "
            f"Var: {snapshot['metrics']['manifold_variance']:.6f} | "
            f"Batch Conf: {avg_confidence:.3f}"
        )

    # Agnostic Save sequence
    sage_container.save_agnostic_stage(stage_name.capitalize())
    governor.secure_save(live_graph)
    torch.save(frontend.state_dict(), f"checkpoints/FRONTEND_{stage_name.capitalize()}.pth")

    analytics.save_stage_report(stage_name)
    print(f"[+] Stage {stage_name} Complete. Weights & Graph Secured.")


# -------------------------
# DATA
# -------------------------

def get_sage_mnist_loader(stage_name, graph, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_data = datasets.MNIST(root="data", train=train, download=True, transform=transform)
    samples = []
    limit = 1000 if train else 500
    for i in range(len(mnist_data)):
        img, label = mnist_data[i]
        samples.append({'input_data': img, 'concepts': [label], 'target': label})
        if len(samples) >= limit: break

    return DataLoader(
        SAGEDataset(samples, graph, transform=lambda x: x.view(-1)),
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
    parser.add_argument("--up_to_stage", default=None)
    parser.add_argument("--data", choices=["mnist", "text"], default="mnist")
    parser.add_argument("--text_dir", type=str, default="data/text")
    parser.add_argument("--audit_level",
                        choices=["SAGE_DELEGATED", "FORCED_INCINERATE", "FORCED_TOMBSTONE", "DISABLED"],
                        default="SAGE_DELEGATED")
    args = parser.parse_args()

    analytics = SAGEAnalyticsEngine()
    from src.graph.graph_governance import GraphGovernance
    governor = GraphGovernance("checkpoints/global_manifold.pth")
    graph = SharedConceptGraph(embedding_dim=128)
    manifold_loaded = governor.secure_load(graph)
    auditor = SageAuditor(graph, mode=args.audit_level)

    if args.data == "mnist":
        frontend = SensoryFrontend(input_dim=784, embed_dim=128)
    else:
        frontend = SensoryFrontend(input_dim=784, embed_dim=128)

    stage_models = {
        "Infant": InfantTransformer(),
        "Toddler": ToddlerTransformer(),
        "Preschool": PreschoolTransformer(),
        "Gradeschool": GradeschoolTransformer(),
        "Teen": TeenTransformer(),
        "Adult": AdultTransformer(),
        "Elder": ElderTransformer(),
    }
    sage_container = SAGEContainer(graph, stage_models, {s: {"pattern_acc": 0.8} for s in STAGE_ORDER}, auditor)

    # Initial Load Logic
    if args.load:
        path = f"checkpoints/SAGE_STATE_{args.load.capitalize()}.pth"
        ckpt = torch.load(path, map_location="cpu")
        if 'frontend_state' in ckpt:
            frontend.load_state_dict(ckpt['frontend_state'])
        sage_container.load_state_dict(ckpt.get('sage_state', ckpt), strict=False)
        print(f"[*] Initial Load: {path}")
    elif not manifold_loaded:
        print("[!] No Manifold found. Seeding...")
        seeder = SAGEGraphSeeder(graph, frontend)
        seeder.seed_all(data.seeder_concepts.BROAD_CONCEPTS, {i: torch.rand(1, 784) for i in range(10)})
        governor.secure_save(graph)

    # DataLoader Setup
    loader = None
    if args.data == "text":
        from data.text_dataloader import get_sage_text_loader
        loader = get_sage_text_loader(args.text_dir, graph, batch_size=64)

    # -------------------------
    # Training (Unified Dynamic Relay)
    # -------------------------
    if args.mode == "train":
        # 1. Determine the target range of stages
        if args.stage.lower() == "all" or args.up_to_stage:
            start_idx = STAGE_ORDER.index(args.load.capitalize()) if args.load else 0
            end_idx = STAGE_ORDER.index(args.up_to_stage.capitalize()) + 1 if args.up_to_stage else len(STAGE_ORDER)
            target_stages = STAGE_ORDER[start_idx:end_idx]
        else:
            # Single stage requested: still treat as a relay target
            target_stages = [args.stage.capitalize()]

        # 2. Iterate through targets and handle weight inheritance
        for stage_name in target_stages:
            idx = STAGE_ORDER.index(stage_name)

            # Look for current stage checkpoint (to resume) or previous stage (to inherit)
            curr_p = f"checkpoints/SAGE_STATE_{stage_name}.pth"
            prev_p = f"checkpoints/SAGE_STATE_{STAGE_ORDER[idx - 1].capitalize()}.pth" if idx > 0 else None

            # Priority: Resume existing work > Inherit from parent stage
            load_p = curr_p if os.path.exists(curr_p) else (prev_p if prev_p and os.path.exists(prev_p) else None)

            if load_p:
                print(f"[*] Relay: Injecting {load_p} into {stage_name}")
                # Load to CPU first to avoid VRAM fragmentation before run_train_cycle moves it
                ckpt = torch.load(load_p, map_location="cpu")

                # Extract state dict based on your SAGEContainer.save_agnostic_stage format
                state = ckpt.get('model_state', ckpt.get('sage_state', ckpt))

                # strict=False allows for the "Muscles" (Transformer layers) to shift
                # between stage-specific architectures (Infant -> Toddler)
                sage_container.load_state_dict(state, strict=False)

            # Execute the surgical training for this stage
            run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args, governor, loader)

    elif args.mode == "test_manifold":
        print(f"[MANIFOLD] Conceptual Variance: {compute_manifold_variance(graph):.6f}")

    # Clean shutdown of the Auditor thread/processes
    auditor.shutdown()


if __name__ == "__main__":
    main()