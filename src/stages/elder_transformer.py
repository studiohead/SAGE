# src/stages/elder_transformer.py

##############################################################################
# Elder | Functional Recursive Meta-Governance Layer
# Purpose:
# Accepts active execution paths, context centroids, and global anchors.
# Applies recursive governance over lower-stage outputs.
# Emits governed traces and comparative divergence metrics for self-evaluation.
##############################################################################

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
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # 3. Dynamic Transformer Stack
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.25)
            ) for _ in range(num_stage_layers)
        ])

        # 4. Governance Components
        self.governance_gate = nn.Linear(embed_dim, embed_dim)
        self.path_reconciler = nn.MultiheadAttention(embed_dim, nhead)
        self.refiner_norm = nn.LayerNorm(embed_dim)
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, adult_attn_map=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        graph_matrix: [num_nodes, dim]
        """
        gamma = torch.tensor(0.0, device=x.device)

        # 1. RECONSTRUCT ACTIVE REASONING PATH
        if adult_attn_map is not None:
            path_active = torch.matmul(adult_attn_map, graph_matrix).transpose(0, 1)

            if centroid_addresses:
                weights = torch.tensor([0.2, 0.3, 0.5], device=x.device).view(-1, 1)
                anchors_stacked = torch.stack(centroid_addresses)
                context_anchor_base = (anchors_stacked * weights).sum(dim=0).unsqueeze(0).unsqueeze(0)
                context_anchor = context_anchor_base.expand(path_active.size(0), path_active.size(1), -1)
            else:
                context_anchor = path_active

            path_anchor, _ = self.path_reconciler(path_active, context_anchor, context_anchor)
            divergence = F.cosine_similarity(path_active, path_anchor, dim=-1).mean()
            gamma = torch.clamp(divergence, 0, 1)

            governed_context = self.governance_gate(path_anchor)
            x = x + (Wi * governed_context)

        else:
            # FALLBACK: Explicit shape-reconciliation
            if graph_matrix.size(0) > 0:
                # 1. Get the target embedding dimension (last dim of x)
                target_dim = x.size(-1)

                # 2. Get the manifold centroid and FORCE it to be a 1D vector of target_dim
                # This fixes the [1, 64, 64] vs [128, 64] error
                manifold_centroid = graph_matrix.mean(dim=0).flatten()[:target_dim]

                # 3. Create an anchor that matches x's shape [seq, batch, target_dim]
                # We use unsqueeze(0).unsqueeze(0) to create [1, 1, 64] then expand
                context_anchor = manifold_centroid.view(1, 1, -1).expand(x.size(0), x.size(1), -1)

                # 4. Compare across the embedding dimension (-1)
                fallback_div = F.cosine_similarity(x, context_anchor, dim=-1).mean()
                gamma = torch.clamp(fallback_div, 0, 1)
            else:
                gamma = torch.tensor(0.0, device=x.device)

        # 6. TRANSFORMER PROCESSING
        for layer in self.layers:
            x = layer(x)

        # 7. FINAL SCHEMA REFINEMENT
        x = self.refiner_norm(x)

        return self.stage_weight * x, 1.0 - gamma.item()