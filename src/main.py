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
from data.mnist_dataloader import SageMNISTDataset

from src.stages.infant_transformer import InfantTransformer
from src.stages.toddler_transformer import ToddlerTransformer
from src.stages.preschool_transformer import PreschoolTransformer
from src.stages.gradeschool_transformer import GradeschoolTransformer
from src.stages.teen_transformer import TeenTransformer
from src.stages.adult_transformer import AdultTransformer
from src.stages.elder_transformer import ElderTransformer
from src.training.training import run_train_cycle

# Note: Add Adult/Elder imports here if they exist in your stages dir

STAGE_ORDER = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Elder"]
EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')


# -------------------------
# MODELS & UTILS
# -------------------------

class Frontend(nn.Module):
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
    print(f"[*] Initializing Frontend for {args.data.upper()} (Input Dim: {input_dim})")

    frontend = Frontend(input_dim=input_dim, embed_dim=EMBED_DIM).to(device)

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