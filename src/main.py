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
from src.inference.graph_tokenizer import GraphTokenizer
from src.inference.sage_inference import SAGEInference
from src.graph.shared_concept_graph import SharedConceptGraph
from src.container.sage_container import SAGEContainer
from src.monitoring.sage_auditor import SageAuditor
from src.graph.graph_seeder import SAGEGraphSeeder
from src.monitoring.analytics_engine import SAGEAnalyticsEngine
from src.monitoring.health_tracker import ManifoldHealthTracker
from src.utils.growth_hormone import GrowthHormone

# Data & Stage Imports
from data.sage_dataloader import SAGEDataset, collate_sage_batch
# --- SURGICAL IMPORT: Using your new dataset class ---
from data.mnist_dataloader import SageMNISTDataset

from src.stages.infant_transformer import InfantTransformer
from src.stages.toddler_transformer import ToddlerTransformer
from src.stages.preschool_transformer import PreschoolTransformer
from src.stages.gradeschool_transformer import GradeschoolTransformer
from src.stages.teen_transformer import TeenTransformer
from src.stages.adult_transformer import AdultTransformer
from src.stages.elder_transformer import ElderTransformer

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


##################
# TRAINING
##################
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
    for param in sage_container.parameters():
        param.requires_grad = False

    stage_key = stage_name.capitalize()
    target_arg_stage = args.stage.capitalize()

    if stage_key in sage_container.stages:
        target_stage = sage_container.stages[stage_key]

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

    for param in frontend.parameters():
        param.requires_grad = True

    frontend.to(device)
    sage_container.to(device)

    # 2. Optimizer & Speed Hardening
    hparams = STAGE_HYPERPARAMS[stage_key]
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

        frontend.train()
        if stage_key == target_arg_stage:
            sage_container.train()
        else:
            sage_container.eval()

        loss_total, batch_count, total_gamma = 0.0, 0, 0.0
        telemetry = {}

        static_graph = None
        if stage_key != "Infant":
            static_graph = sage_container.graph.get_graph_embedding_matrix()

        for batch in loader:
            # Shift data to device
            x = batch['input_ids'].to(device, non_blocking=True)
            batch_indices = batch.get('node_indices')
            y = batch['labels'].to(device, non_blocking=True) if 'labels' in batch else None

            # --- PROBE: INPUT & MEMORY HEARTBEAT (Fixed Subscript Error) ---
            if batch_count % 5 == 0:
                mem_str = ""
                if device.type == "mps":
                    mem_used = torch.mps.current_allocated_memory() / 1024 ** 2
                    mem_str = f" | MPS Mem: {mem_used:.1f}MB"

                # Use empty list if batch_indices is None
                display_indices = batch_indices[:5] if batch_indices is not None else "N/A"
                print(f"[PROBE] Batch {batch_count} | Winners: {display_indices}{mem_str}")

            c_addresses = batch.get('centroid_addresses')
            if c_addresses is not None:
                c_addresses = [c.to(device, non_blocking=True) for c in c_addresses]

            if args.data == "text" and x.shape[1] != EMBED_DIM:
                padded = torch.zeros((x.size(0), EMBED_DIM), device=device)
                padded[:, :min(x.size(1), EMBED_DIM)] = x[:, :min(x.size(1), EMBED_DIM)]
                x = padded

            current_g_matrix = static_graph if static_graph is not None else \
                sage_container.graph.get_graph_embedding_matrix()

            optimizer.zero_grad(set_to_none=True)

            logits, telemetry = frontend(
                x,
                sage_container,
                graph_matrix=current_g_matrix,
                centroid_addresses=c_addresses
            )

            # In the Teen conditional during training
            if stage_key == "Teen":
                teen_hparams = STAGE_HYPERPARAMS["Teen"]

                growth_hormone = GrowthHormone(
                    floor=teen_hparams.get("growth_confidence_floor", 0.3),
                    ceiling=teen_hparams.get("confidence_threshold", 0.7)
                )

                # Growth is now driven purely by telemetry, not model-internal state
                new_node_id = growth_hormone.maybe_create_node(
                    telemetry=telemetry,
                    graph=sage_container.graph
                )

                if new_node_id is not None:
                    print(
                        f"\n>>> [!] GROWTH-HORMONES TRIGGERED "
                        f"(New Node Created: {new_node_id})"
                    )

            if args.data == "mnist":
                loss = criterion(logits, y)
                if stage_key == target_arg_stage:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, hparams.get("gradient_clip", 1.0))
                    optimizer.step()
                loss_total += loss.item()

                # --- MNIST HEBBIAN BRANCH ---
                target_threshold = hparams.get("confidence_threshold", 0.7)
                conf = telemetry.get('confidence', 0.0)

                if (stage_key == target_arg_stage) and (conf >= target_threshold) and batch_indices is not None:
                    trace = telemetry.get('trace')
                    print(f" [!] Wiring Edges (Conf: {conf:.2f})...", end="", flush=True)
                    sage_container.graph.update_stage_aware_hebbian(
                        attention_map=trace,
                        batch_indices=batch_indices,
                        stage_plasticity=hparams["plasticity_scale"],
                        threshold=target_threshold
                    )
                    print(" Done.")
            else:
                # --- TEXT BRANCH ---
                target_threshold = hparams.get("confidence_threshold", 0.7)
                is_infant = (stage_key == "Infant")

                if is_infant and stage_key == target_arg_stage:
                    should_update = True
                    current_threshold = 0.0
                else:
                    conf = telemetry.get('confidence', 0.0)
                    should_update = (stage_key == target_arg_stage) and (conf >= target_threshold)
                    current_threshold = target_threshold

                if should_update and batch_indices:
                    trace = telemetry.get('trace')
                    if is_infant or trace is None or trace.sum() == 0:
                        trace = torch.ones((1, len(batch_indices)), device=device)

                    print(f" [!] Text Wiring (Conf: {telemetry.get('confidence', 0.0):.2f})...", end="", flush=True)
                    sage_container.graph.update_stage_aware_hebbian(
                        attention_map=trace,
                        batch_indices=batch_indices,
                        stage_plasticity=hparams["plasticity_scale"],
                        threshold=current_threshold
                    )
                    print(" Done.")

            current_gamma = telemetry.get('gamma_divergence', torch.tensor(0.0, device=device))
            total_gamma += current_gamma.item() if not torch.isnan(current_gamma) else 0.0
            batch_count += 1

            if batch_count % 50 == 0:
                if device.type == "mps":
                    torch.mps.empty_cache()

        avg_loss = loss_total / max(len(loader), 1) if args.data == "mnist" else 0.0
        avg_gamma = total_gamma / max(batch_count, 1)

        snapshot = analytics.capture_snapshot(stage_name, stage_idx, epoch, avg_loss, telemetry, sage_container.graph)
        node_count = len(sage_container.graph.nodes)
        edge_count = sum(len(node.connections) for node in sage_container.graph.nodes.values())
        avg_degree = edge_count / max(node_count, 1)
        degrees = [len(node.connections) for node in sage_container.graph.nodes.values()]
        print(f" > Max Node Degree: {max(degrees)}")
        print(f" > Min Node Degree: {min(degrees)}")

        print(f"\n[EPOCH {epoch + 1} COMPLETE]")
        print(f" > Status:      {'TRAINING' if stage_key == target_arg_stage else 'SUTURED'}")
        print(f" > Loss:        {avg_loss:.6f}")
        print(f" > Gamma (Γ):   {avg_gamma:.6f}")
        print(f" > Variance:    {snapshot['metrics']['manifold_variance']:.6f}")
        print(f" > Nodes:       {node_count}")
        print(f" > Avg Node Degree: {avg_degree:.2f}")
        print(f" > Active Edges: {edge_count}")
        print("-" * 45)

    if stage_key == target_arg_stage:
        sage_container.save_agnostic_stage(stage_key)
        governor.secure_save(sage_container.graph)
        torch.save(frontend.state_dict(), f"checkpoints/FRONTEND_{stage_key}.pth")

    analytics.save_stage_report(stage_name)
    health_tracker.check_health(stage_name, "FINAL")

    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    print(f"[+] Stage {stage_name} Complete. Hardware Purged.")


# -------------------------
# DATA & ENTRY
# -------------------------

def get_sage_mnist_loader(stage_name, graph, train=True, device=None):
    """
    Surgically repaired loader using the SageMNISTDataset class to bridge
    MNIST patterns to seeded BROAD_CONCEPTS.
    """
    batch_size = STAGE_HYPERPARAMS.get(stage_name.capitalize(), {}).get('batch_size', 4)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_data = datasets.MNIST(root="data", train=train, download=True, transform=transform)

    # --- FIX: Bridging to the seeded strings '0'-'9' ---
    sage_dataset = SageMNISTDataset(mnist_data, graph)

    # Speed: Enable pin_memory only for non-CPU devices
    use_pin = device is not None and device.type != "cpu"

    return DataLoader(
        sage_dataset,
        batch_size=batch_size,
        shuffle=train,
        pin_memory=use_pin
    )


def main():
    parser = argparse.ArgumentParser("SAGE Surgical Interface")
    parser.add_argument("mode", choices=["train", "test_manifold", "inference"])
    parser.add_argument("--stage", default="Infant")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--load", type=str)
    parser.add_argument("--up_to_stage", default=None)
    parser.add_argument("--data", choices=["mnist", "text"], default="mnist")
    parser.add_argument("--text_path", type=str, default="training_data/lit1.txt")
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

    # --- AUTO-DISCOVERY LOGIC ---
    if not args.load:
        for stage in reversed(STAGE_ORDER):
            if os.path.exists(f"checkpoints/SAGE_STATE_{stage.capitalize()}.pth"):
                args.load = stage
                print(f"[*] AUTO-RESUME: Detected existing state '{stage}'. Loading...")
                break

    # --- SURGICAL FIX: DYNAMIC INPUT DIMENSION ---
    # Detect if we are feeding 784 pixels (MNIST) or 128 latents (Text)
    input_dim = 784 if args.data == "mnist" else EMBED_DIM
    print(f"[*] Initializing SensoryFrontend for {args.data.upper()} (Input Dim: {input_dim})")

    frontend = SensoryFrontend(input_dim=input_dim, embed_dim=EMBED_DIM).to(device)

    # Include the full lifecycle to match STAGE_ORDER
    stage_models = {
        "Infant": InfantTransformer(),
        "Toddler": ToddlerTransformer(),
        "Preschool": PreschoolTransformer(),
        "Gradeschool": GradeschoolTransformer(),
        "Teen": TeenTransformer(),
        "Adult": AdultTransformer(),
        "Elder": ElderTransformer()
    }

    auditor = SageAuditor(graph, mode=args.audit_level)
    sage_container = SAGEContainer(graph, stage_models, {s: {"pattern_acc": 0.8} for s in STAGE_ORDER}, auditor).to(
        device)

    if args.load:
        path = f"checkpoints/SAGE_STATE_{args.load.capitalize()}.pth"
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)

            # --- SURGICAL FIX: SHAPE-AWARE FRONTEND LOAD ---
            try:
                if 'frontend_state' in ckpt:
                    frontend.load_state_dict(ckpt['frontend_state'])
            except RuntimeError:
                print("[!] Frontend shape mismatch detected. Re-initializing weights for data pivot.")

            sage_container.load_state_dict(ckpt.get('sage_state', ckpt), strict=False)
        else:
            print(f"FAILED: Checkpoint {path} not found.")
            sys.exit(1)

    # --- CONDITIONAL SEEDING ---
    if not manifold_loaded or len(graph.nodes) == 0:
        print("[!] Empty Manifold detected. Initializing First Birth (Seeding)...")
        seeder = SAGEGraphSeeder(graph, frontend)
        # Match BROAD_CONCEPTS strings "0"-"9"
        mnist_seeds = {str(i): torch.rand(1, EMBED_DIM).to(device) for i in range(10)}
        seeder.seed_all(
            data.seeder_concepts.BROAD_CONCEPTS,
            mnist_seeds
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
                loader = get_sage_text_loader(args.text_path, graph, stage_name)

            run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args, governor, loader)

    elif args.mode == "test_manifold":
        print(f"[MANIFOLD] Conceptual Variance: {compute_manifold_variance(graph):.6f}")

    # Inside main(), after your existing mode handling
    elif args.mode == "inference":
        print("[*] Entering SAGE Inference Mode. Type 'exit' to quit.")

        # Initialize tokenizer and inference
        tokenizer = GraphTokenizer(graph=graph)
        sage_infer = SAGEInference(graph=graph, tokenizer=tokenizer)

        print("[*] Entering SAGE Inference Mode. Type 'exit' to quit.")
        while True:
            prompt = input(">>> ")
            if prompt.lower() in {"exit", "quit"}:
                break
            response = sage_infer.respond(prompt, top_k=5)
            print(response)
            print("-" * 40)

        while True:
            prompt = input(">>> ")
            if prompt.lower() in {"exit", "quit"}:
                break
            response = sage_infer.respond(prompt)
            print(response)

    auditor.shutdown()


if __name__ == "__main__":
    main()