import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader
# Removed the ambiguous 'from torch.utils.tensorboard.summary import hparams'
# to prevent naming collisions with STAGE_HYPERPARAMS
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
from data.mnist_dataloader import SageMNISTDataset, get_sage_mnist_loader

from src.stages.infant_transformer import InfantTransformer
from src.stages.toddler_transformer import ToddlerTransformer
from src.stages.preschool_transformer import PreschoolTransformer
from src.stages.gradeschool_transformer import GradeschoolTransformer
from src.stages.teen_transformer import TeenTransformer
from src.stages.adult_transformer import AdultTransformer
from src.stages.elder_transformer import ElderTransformer
from src.training.training import run_train_cycle
from src.utils.growth_hormone import GrowthHormone

# -------------------------
# STAGE & CONFIG CONSTANTS
# -------------------------
STAGE_ORDER = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Elder"]
EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')


# -------------------------
# MODELS & UTILS
# -------------------------

class Frontend(nn.Module):
    def __init__(self, input_dim, embed_dim=EMBED_DIM, data_mode="mnist"):
        super().__init__()
        self.data_mode = data_mode

        if self.data_mode == "imagenet":
            from torchvision import models
            self.backbone = models.resnet18(weights=None)
            self.backbone.fc = nn.Sequential(
                nn.Linear(self.backbone.fc.in_features, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, EMBED_DIM),
                nn.ReLU(),
                nn.Linear(EMBED_DIM, embed_dim),
                nn.LayerNorm(embed_dim)
            )

        # The classifier MUST act on the output of the SAGE Container
        self.classifier = nn.Linear(embed_dim, 1000)

    def forward(self, x, sage_container, graph_matrix=None):
        if self.data_mode == "imagenet" and x.dim() == 4:
            latent = self.backbone(x)
        else:
            if x.dim() == 4:
                x = x.mean(dim=(2, 3))
            elif x.dim() == 3:
                x = x.mean(dim=1)
            else:
                x = x.view(x.size(0), -1)
            latent = self.encoder(x)

        sage_input = latent.unsqueeze(0)

        fused_output, telemetry = sage_container(
            sage_input,
            graph_matrix=graph_matrix,
            centroid_addresses=None  # anchors no longer needed
        )

        if fused_output.dim() == 3:
            fused_output = fused_output.squeeze(0)

        return self.classifier(fused_output), telemetry


def main():
    parser = argparse.ArgumentParser("SAGE Surgical Interface")
    parser.add_argument("mode", choices=["train", "test_manifold", "inference"])
    parser.add_argument("--stage", default="Infant")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--load", type=str)
    parser.add_argument("--up_to_stage", default=None)
    parser.add_argument("--data", choices=["mnist", "text", "imagenet"], default="mnist")
    parser.add_argument("--text_path", type=str, default="training_data/pr_25000.txt")
    parser.add_argument("--audit_level",
                        choices=["SAGE_DELEGATED", "FORCED_INCINERATE", "FORCED_TOMBSTONE", "DISABLED"],
                        default="SAGE_DELEGATED")
    args = parser.parse_args()

    from src.graph.graph_governance import GraphGovernance
    analytics = SAGEAnalyticsEngine()
    governor = GraphGovernance("checkpoints/global_manifold.pth")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. GRAPH INITIALIZATION
    graph = SharedConceptGraph(embedding_dim=EMBED_DIM).to(device)
    manifold_loaded = governor.secure_load(graph)

    # 2. SEEDING (Only if fresh start or empty manifold)
    if not manifold_loaded or len(graph.nodes) == 0:
        print("[!] Empty Manifold detected. Initializing First Birth (Seeding)...")

        # Temporary frontend for seeder context
        temp_frontend = Frontend(input_dim=EMBED_DIM, embed_dim=EMBED_DIM, data_mode="text").to(device)
        seeder = SAGEGraphSeeder(graph, temp_frontend)

        with torch.no_grad():
            # Identity-based basis for the "Respected Ten" (0-9)
            basis = torch.eye(10, EMBED_DIM).to(device)
            basis += torch.randn_like(basis) * 0.01
            mnist_seeds = {str(i): basis[i].unsqueeze(0) for i in range(10)}

        # Seed concepts (BROAD_CONCEPTS)
        seeder.seed_all(data.seeder_concepts.BROAD_CONCEPTS, mnist_seeds)

        # Final migration sync
        graph.to(device)
        governor.secure_save(graph)
        print(f"[*] Manifold Seeded: {len(graph.nodes)} nodes initialized on {device}.")

    # 3. AUTO-RESUME CHECKPOINT DETECTION
    if not args.load:
        for stage in reversed(STAGE_ORDER):
            if os.path.exists(f"checkpoints/SAGE_STATE_{stage.capitalize()}.pth"):
                args.load = stage
                print(f"[*] AUTO-RESUME: Detected existing state '{stage}'. Loading...")
                break

    # 4. CONTAINER & FRONTEND SETUP
    input_dim = 784 if args.data == "mnist" else (3 if args.data == "imagenet" else EMBED_DIM)
    print(f"[*] Initializing Frontend for {args.data.upper()} (Input Dim: {input_dim})")

    frontend = Frontend(input_dim=input_dim, embed_dim=EMBED_DIM, data_mode=args.data).to(device)

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

    # SUTURE: SAGEContainer now accepts 'gh' to allow internal birth logic
    sage_container = SAGEContainer(
        graph,
        stage_models,
        {s: {"pattern_acc": 0.8} for s in STAGE_ORDER},
        auditor
    ).to(device)

    tokenizer = GraphTokenizer(graph=graph)

    # 5. LOAD WEIGHTS
    if args.load:
        path = f"checkpoints/SAGE_STATE_{args.load.capitalize()}.pth"
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            if isinstance(ckpt, dict) and 'frontend_state' in ckpt:
                frontend.load_state_dict(ckpt['frontend_state'])
            sage_container.load_agnostic_stage(args.load.capitalize())
        else:
            print(f"FAILED: Checkpoint {path} not found.")
            sys.exit(1)

    # 6. EXECUTION MODES
    if args.mode == "train":
        start_idx = STAGE_ORDER.index(args.load.capitalize()) if args.load else 0
        target_stage = args.up_to_stage.capitalize() if args.up_to_stage else args.stage.capitalize()
        end_idx = STAGE_ORDER.index(target_stage) + 1

        for stage_name in STAGE_ORDER[start_idx:end_idx]:
            loader = None
            if args.data == "text":
                from data.text_dataloader import get_sage_text_loader
                loader = get_sage_text_loader(args.text_path, graph, stage_name)
            elif args.data == "mnist":
                loader = get_sage_mnist_loader(stage_name, graph, tokenizer, train=True, device=device)
            elif args.data == "imagenet":
                from data.imagenet_dataloader import get_sage_imagenet_loader
                loader = get_sage_imagenet_loader(stage_name, graph, train=True, device=device)

            # --- SAGE GRADIENT SUTURE VERIFICATION ---
            graph.train()
            for param in graph.parameters():
                param.requires_grad = True

            # RESOLVED: Corrected variable naming to 'current_hparams' and 'stage_name'
            current_hparams = STAGE_HYPERPARAMS[stage_name]

            # Initialize GH with stage-specific thresholds
            gh = GrowthHormone(
                floor=current_hparams.get("growth_confidence_floor", 0.99),
                ceiling=current_hparams.get("confidence_ceiling", 1.0)
            )

            # REMOVE 'gh' from this call
            run_train_cycle(frontend, sage_container, auditor, analytics, stage_name, args, governor, loader)

    elif args.mode == "test_manifold":
        graph.train()
        variance = graph.compute_manifold_variance()
        print(f"[MANIFOLD] Nodes: {len(graph.nodes)} | Conceptual Variance: {variance.item():.6f}")

    elif args.mode == "inference":
        tokenizer = GraphTokenizer(graph=graph)
        sage_infer = SAGEInference(graph=graph, tokenizer=tokenizer)
        print("[*] Entering SAGE Inference Mode. Type 'exit' to quit.")
        while True:
            prompt = input(">>> ")
            if prompt.lower() in {"exit", "quit"}: break
            response = sage_infer.respond(prompt, top_k=5, interactive=True)
            print(f"SAGE: {response}")
            print("-" * 40)

    auditor.shutdown()


if __name__ == "__main__":
    main()