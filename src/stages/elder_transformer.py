# src/stages/elder_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class ElderTransformer(DevelopmentalTransformer):
    """
    Stage 7: Highest-Tier Recursive Meta-Governance.
    Implements Paragraph [0014] of the SAGE Patent:
    Compares Active Reasoning Paths against Multi-Scale Centroid Anchors.
    """

    def __init__(self):
        # 1. Pull Shared Architecture from Config
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # 2. Pull Stage-Specific Hyperparams from Config
        stage_cfg = STAGE_HYPERPARAMS.get("Elder", STAGE_HYPERPARAMS.get("Adult"))

        # Dynamic Layer Allocation from config boundaries (e.g., 24-28)
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # 3. Dynamic Transformer Stack (Aperture-controlled)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.25)
            ) for _ in range(num_stage_layers)
        ])

        # 4. Governance Head: Projects reasoning into the "Anchor Space"
        self.governance_gate = nn.Linear(embed_dim, embed_dim)

        # 5. Meta-Governance Projection: Reconciles Active Path vs Centroids
        self.path_reconciler = nn.MultiheadAttention(embed_dim, nhead)

        # 6. Output Refinement
        self.refiner_norm = nn.LayerNorm(embed_dim)

        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, adult_attn_map=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        graph_matrix: [num_nodes, dim]
        centroid_addresses: List of [dim] tensors (Fast, Mid, Slow)
        """
        gamma = torch.tensor(1.0, device=x.device)

        # 1. RECONSTRUCT ACTIVE REASONING PATH (Patent [0014])
        if adult_attn_map is not None:
            # Reconstruct trajectory: [seq_len, batch, dim]
            # adult_attn_map is [batch, seq_len, num_nodes] -> transpose for matmul
            # graph_matrix is [num_nodes, dim]
            path_active = torch.matmul(adult_attn_map, graph_matrix).transpose(0, 1)

            # 2. COMPUTE MULTI-SCALE CONTEXT ANCHOR (Weighted α-scales)
            if centroid_addresses:
                # Weighted average per Patent [0014]: Slow scale (0.5) is the primary anchor
                weights = torch.tensor([0.2, 0.3, 0.5], device=x.device).view(-1, 1)
                anchors_stacked = torch.stack(centroid_addresses)  # [3, dim]
                context_anchor_base = (anchors_stacked * weights).sum(dim=0).unsqueeze(0).unsqueeze(0)
                context_anchor = context_anchor_base.expand(path_active.size(0), path_active.size(1), -1)
            else:
                context_anchor = path_active

            # 3. RECURSIVE META-GOVERNANCE (Path Reconciliation)
            path_anchor, _ = self.path_reconciler(path_active, context_anchor, context_anchor)

            # 4. COMPUTE COMPARATIVE GEOMETRIC DIVERGENCE (Γ)
            divergence = F.cosine_similarity(path_active, path_anchor, dim=-1).mean()
            gamma = torch.clamp(divergence, 0, 1)

            # 5. GOVERNANCE TRANSFORMATION
            governed_context = self.governance_gate(path_anchor)
            x = x + (Wi * governed_context)

        # 6. TRANSFORMER PROCESSING (Sliding Window Managed by Container)
        for layer in self.layers:
            x = layer(x)

        # 7. FINAL SCHEMA REFINEMENT
        x = self.refiner_norm(x)

        # Return governed trace and remediation confidence (1 - Γ)
        return self.stage_weight * x, 1.0 - gamma.item()