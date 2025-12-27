import os
import sys
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
from src.monitoring.health_tracker import ManifoldHealthTracker

# Data & Stage Imports
from data.sage_dataloader import SAGEDataset, collate_sage_batch
from src.stages.infant_transformer import InfantTransformer
from src.stages.toddler_transformer import ToddlerTransformer
from src.stages.preschool_transformer import PreschoolTransformer
from src.stages.gradeschool_transformer import GradeschoolTransformer
from src.stages.teen_transformer import TeenTransformer

# Note: Add Adult/Elder imports here if they exist in your stages dir

STAGE_ORDER = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Elder"]
EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')


# -------------------------
# MODELS & UTILS
# -------------------------

class SensoryFrontend(nn.Module):
    def __init__(self, input_dim, embed_dim=EMBED_DIM):
        super().__init__()
        # Preserving your exact architecture: Linear -> ReLU -> Linear -> LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 10)

    def forward(self, x, sage_container, graph_matrix=None, centroid_addresses=None):
        # Flatten handles both [B, 1, 28, 28] and [B, Seq, Dim]
        x_flat = x.view(x.size(0), -1)
        latent = self.encoder(x_flat)
        sage_input = latent.unsqueeze(0)

        # YOUR VECTORIZED SPEED HANDLING (Preserved exactly)
        if graph_matrix is not None:
            if graph_matrix.dim() == 4:
                graph_matrix = graph_matrix.squeeze(0)
            if graph_matrix.dim() == 3 and graph_matrix.size(0) == x_flat.size(0):
                graph_matrix = graph_matrix.transpose(0, 1)

        fused_latent, telemetry = sage_container(
            sage_input,
            graph_matrix=graph_matrix,
            centroid_addresses=centroid_addresses
        )
        return self.classifier(latent), telemetry


def compute_manifold_variance(graph: SharedConceptGraph):
    """GLOBAL RELATIONAL VARIANCE: Measures clustering around Z-anchors."""
    variances = []
    for node_id, centroid in graph.anchor_tensor.items():
        node_key = str(node_id)
        if node_key not in graph.nodes:
            continue
        node = graph.nodes[node_key]
        if node.is_tombstoned or node.alignment_score <= 0.0:
            continue
        diffs = []
        for other in graph.nodes.values():
            if other.is_tombstoned or other.alignment_score <= 0.0:
                continue
            diffs.append(torch.sum((other.embedding - centroid.to(other.embedding.device)) ** 2))
        if diffs:
            variances.append(torch.mean(torch.stack(diffs)))
    return torch.mean(torch.stack(variances)).item() if variances else 0.0


## -------------------------
# TRAINING CYCLE
# -------------------------

def run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args, governor, loader=None):
    # Detect Hardware: Priority MPS (Mac) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs("checkpoints", exist_ok=True)

    # 1. THE SURGICAL SUTURE (Conditional Freezing)
    # First, lock everything in the container globally
    for param in sage_container.parameters():
        param.requires_grad = False

    stage_key = stage_name.capitalize()
    target_arg_stage = args.stage.capitalize()

    if stage_key in sage_container.stages:
        target_stage = sage_container.stages[stage_key]

        # HANDSHAKE LOGIC:
        # Only unlock gradients if this loop matches the actual target stage.
        # Otherwise, keep it in eval() for a pure forward-pass handshake.
        if stage_key == target_arg_stage:
            print(f"[*] ACTIVE TRAINING: Unlocking {stage_key} gradients.")
            target_stage.train()
            for param in target_stage.parameters():
                param.requires_grad = True

            if hasattr(target_stage, 'layers'):
                for layer in target_stage.layers:
                    if hasattr(layer, 'gradient_checkpointing'):
                        layer.gradient_checkpointing = True
        else:
            print(f"[*] SUTURE MODE: {stage_key} is frozen. Handshake only.")
            target_stage.eval()
    else:
        print(f"WARNING: Stage {stage_key} not found in container.")
        return

    # Frontend must stay trainable to project into the Manifold
    # This is the "bridge" that learns to map MNIST pixels to Text nodes
    for param in frontend.parameters():
        param.requires_grad = True

    frontend.to(device)
    sage_container.to(device)

    # 2. Optimizer & Speed Hardening
    hparams = STAGE_HYPERPARAMS[stage_key]

    # Only optimize parameters that were actually unlocked
    trainable_params = [p for p in sage_container.parameters() if p.requires_grad] + list(frontend.parameters())

    optimizer = optim.AdamW(trainable_params, lr=hparams["learning_rate"],
                            weight_decay=hparams.get("weight_decay", 0.01))
    criterion = nn.CrossEntropyLoss()

    # 3. Tracker State
    health_tracker = ManifoldHealthTracker(sage_container.graph)
    stage_idx = STAGE_ORDER.index(stage_key)
    sage_container.current_stage_idx = stage_idx
    sage_container.update_plasticity_window(cumulative=True)

    if loader is None:
        loader = get_sage_mnist_loader(stage_name.lower(), sage_container.graph, train=True, device=device)

    print(f"\n=== SAGE SURGICAL TRAINING: {stage_name.upper()} ({device}) ===")

    for epoch in range(args.epochs):
        health_tracker.check_health(stage_name, epoch)

        # Only set training mode for the container if we are in the target stage
        frontend.train()
        if stage_key == target_arg_stage:
            sage_container.train()
        else:
            sage_container.eval()

        loss_total, batch_count, total_gamma = 0.0, 0, 0.0
        telemetry = {}  # Initialize to prevent reference errors

        # SPEED: Pre-fetch cached manifold matrix once per epoch
        static_graph = sage_container.graph.get_graph_embedding_matrix()

        for batch in loader:
            # Shift data to match Graph device (Non-blocking for MPS/CUDA)
            x = batch['input_ids'].to(device, non_blocking=True)
            batch_indices = batch.get('node_indices')
            y = batch['labels'].to(device, non_blocking=True) if args.data == "mnist" else None

            # Toddler Centroid Injection (Preserved Umbilical)
            c_addresses = batch.get('centroid_addresses')
            if c_addresses is not None:
                c_addresses = [c.to(device, non_blocking=True) for c in c_addresses]

            # Shape Correction for Text
            if args.data == "text" and x.shape[1] != EMBED_DIM:
                padded = torch.zeros((x.size(0), EMBED_DIM), device=device)
                padded[:, :min(x.size(1), EMBED_DIM)] = x[:, :min(x.size(1), EMBED_DIM)]
                x = padded

            optimizer.zero_grad(set_to_none=True)

            logits, telemetry = frontend(
                x,
                sage_container,
                graph_matrix=static_graph,
                centroid_addresses=c_addresses
            )

            # Teen Active Inquiry (Preserved)
            if stage_key == "Teen":
                teen = sage_container.stages["Teen"]
                if hasattr(teen, 'last_inquiry') and teen.last_inquiry is not None:
                    node_id = teen.last_inquiry
                    node_key = str(node_id)
                    if node_key in sage_container.graph.nodes:
                        label = sage_container.graph.nodes[node_key].label
                        print(f"\n>>> [!] SAGE ACTIVE INQUIRY: '{label}' (Node {node_id})")

            # Training Step
            # If gradients are frozen for this stage, optimizer.step() effectively does nothing
            if args.data == "mnist":
                loss = criterion(logits, y)
                if stage_key == target_arg_stage:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, hparams.get("gradient_clip", 1.0))
                    optimizer.step()
                loss_total += loss.item()
            else:
                # Hebbian Relational Update (Unsupervised / Text mode)
                # Only update the graph if this is our active training stage
                if stage_key == target_arg_stage and telemetry.get('confidence', 0.0) > 0.7:
                    sage_container.graph.update_stage_aware_hebbian(
                        attention_map=telemetry.get('trace'),
                        batch_indices=batch_indices,
                        stage_plasticity=hparams["plasticity_scale"]
                    )

            # Accumulate Gamma (Predictive Error)
            current_gamma = telemetry.get('gamma_divergence', torch.tensor(0.0, device=device))
            total_gamma += current_gamma.item() if not torch.isnan(current_gamma) else 0.0
            batch_count += 1

            # Memory Purge for Mac/GPU
            if batch_count % 100 == 0:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()

        # --- EPOCH DASHBOARD ---
        avg_loss = loss_total / max(len(loader), 1) if args.data == "mnist" else 0.0
        avg_gamma = total_gamma / max(batch_count, 1)

        # Capture the manifold state (Variance is calculated here)
        snapshot = analytics.capture_snapshot(stage_name, stage_idx, epoch, avg_loss, telemetry, sage_container.graph)

        # Edge counting logic
        edge_count = sum(len(node.connections) for node in sage_container.graph.nodes.values())

        print(f"\n[EPOCH {epoch + 1} COMPLETE]")
        print(f" > Status:      {'TRAINING' if stage_key == target_arg_stage else 'SUTURED'}")
        print(f" > Loss:        {avg_loss:.6f}")
        print(f" > Gamma (Γ):   {avg_gamma:.6f}  (Predictive Error)")
        print(f" > Variance:    {snapshot['metrics']['manifold_variance']:.6f} (Relational Spread)")
        print(f" > Active Edges: {edge_count}")
        print("-" * 45)

    # 4. Global State Persistence
    # Only save weights if we actually trained this stage
    if stage_key == target_arg_stage:
        sage_container.save_agnostic_stage(stage_key)
        governor.secure_save(sage_container.graph)
        torch.save(frontend.state_dict(), f"checkpoints/FRONTEND_{stage_key}.pth")

    analytics.save_stage_report(stage_name)
    health_tracker.check_health(stage_name, "FINAL")

    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    print(f"[+] Stage {stage_name} Complete. Hardware Purged.")

# -------------------------
# DATA & ENTRY
# -------------------------

def get_sage_mnist_loader(stage_name, graph, train=True, device=None):
    batch_size = STAGE_HYPERPARAMS.get(stage_name.capitalize(), {}).get('batch_size', 4)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_data = datasets.MNIST(root="data", train=train, download=True, transform=transform)
    samples = [{'input_data': mnist_data[i][0], 'concepts': [mnist_data[i][1]], 'target': mnist_data[i][1]} for i in
               range(min(len(mnist_data), 1000 if train else 500))]

    # Speed: Enable pin_memory only for non-CPU devices
    use_pin = device is not None and device.type != "cpu"
    return DataLoader(SAGEDataset(samples, graph, transform=lambda x: x.view(-1)), batch_size=batch_size, shuffle=train,
                      collate_fn=collate_sage_batch, pin_memory=use_pin)


def main():
    parser = argparse.ArgumentParser("SAGE Surgical Interface")
    parser.add_argument("mode", choices=["train", "test_manifold"])
    parser.add_argument("--stage", default="Infant")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--load", type=str)
    parser.add_argument("--up_to_stage", default=None)
    parser.add_argument("--data", choices=["mnist", "text"], default="mnist")
    parser.add_argument("--text_dir", type=str, default="training_data/lit1.txt")
    parser.add_argument("--audit_level",
                        choices=["SAGE_DELEGATED", "FORCED_INCINERATE", "FORCED_TOMBSTONE", "DISABLED"],
                        default="SAGE_DELEGATED")
    args = parser.parse_args()

    from src.graph.graph_governance import GraphGovernance
    analytics = SAGEAnalyticsEngine()
    governor = GraphGovernance("checkpoints/global_manifold.pth")

    # Auto-detect device for main initialization
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    graph = SharedConceptGraph(embedding_dim=EMBED_DIM).to(device)
    manifold_loaded = governor.secure_load(graph)

    # --- SURGICAL FIX: DYNAMIC INPUT DIMENSION ---
    # Detect if we are feeding 784 pixels (MNIST) or 128 latents (Text)
    input_dim = 784 if args.data == "mnist" else EMBED_DIM
    print(f"[*] Initializing SensoryFrontend for {args.data.upper()} (Input Dim: {input_dim})")

    frontend = SensoryFrontend(input_dim=input_dim, embed_dim=EMBED_DIM).to(device)

    stage_models = {
        "Infant": InfantTransformer(),
        "Toddler": ToddlerTransformer(),
        "Preschool": PreschoolTransformer(),
        "Gradeschool": GradeschoolTransformer(),
        "Teen": TeenTransformer()
    }

    auditor = SageAuditor(graph, mode=args.audit_level)
    sage_container = SAGEContainer(graph, stage_models, {s: {"pattern_acc": 0.8} for s in STAGE_ORDER}, auditor).to(
        device)

    if args.load:
        path = f"checkpoints/SAGE_STATE_{args.load.capitalize()}.pth"
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)

            # --- SURGICAL FIX: SHAPE-AWARE FRONTEND LOAD ---
            # If we are pivoting from Text to MNIST, the frontend weights will mismatch.
            # We catch the error to allow the manifold to load even if the frontend resets.
            try:
                if 'frontend_state' in ckpt:
                    frontend.load_state_dict(ckpt['frontend_state'])
            except RuntimeError:
                print("[!] Frontend shape mismatch detected. Re-initializing weights for data pivot.")

            sage_container.load_state_dict(ckpt.get('sage_state', ckpt), strict=False)
        else:
            print(f"FAILED: Checkpoint {path} not found.")
            sys.exit(1)

    elif not manifold_loaded:
        seeder = SAGEGraphSeeder(graph, frontend)
        # --- SURGICAL FIX: Ensure the seeding tensors match the model's device ---
        seeder.seed_all(
            data.seeder_concepts.BROAD_CONCEPTS,
            {i: torch.rand(1, EMBED_DIM).to(device) for i in range(10)}
        )
        governor.secure_save(graph)

    if args.mode == "train":
        start_idx = STAGE_ORDER.index(args.load.capitalize()) if args.load else 0
        end_idx = STAGE_ORDER.index(args.up_to_stage.capitalize()) + 1 if args.up_to_stage else (
                STAGE_ORDER.index(args.stage.capitalize()) + 1)

        for stage_name in STAGE_ORDER[start_idx:end_idx]:
            loader = None
            if args.data == "text":
                from data.text_dataloader import get_sage_text_loader
                loader = get_sage_text_loader(args.text_dir, graph, stage_name)
            # If loader is None here, run_train_cycle will trigger the MNIST loader default
            run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args, governor, loader)

    elif args.mode == "test_manifold":
        print(f"[MANIFOLD] Conceptual Variance: {compute_manifold_variance(graph):.6f}")

    auditor.shutdown()


if __name__ == "__main__":
    main()