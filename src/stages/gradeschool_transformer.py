# src/stages/gradeschool_transformer.py

##############################################################################
# Gradeschool | y = x + W * (G * W_proj)
# Purpose:
# Linear projection of graph manifolds into task-aligned subspaces.
# Establishes stable representational geometry.
# Reduces stochasticity in favor of deterministic structure learning.
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class GradeschoolTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Gradeschool"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.15),
                batch_first=False
            ) for _ in range(num_stage_layers)
        ])

        self.subspace_proj = nn.Linear(embed_dim, embed_dim)
        self.refiner = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None and graph_matrix.size(0) > 0:
            S, B, D = x.shape

            # 1. Standardize nodes to [B, N, D]
            # No matter what comes in, we flatten to [N, D] then expand to [B, N, D]
            g_flat = graph_matrix.view(-1, D)
            N = g_flat.size(0)
            g_aligned = g_flat.unsqueeze(0).expand(B, N, D).contiguous()

            nodes_proj = self.subspace_proj(g_aligned)  # [B, N, D]

            # 2. FIX: Structural Anchor Reduction
            # We don't care about the shape of anchor_raw; we force it to [B, 1, D]
            if centroid_addresses and len(centroid_addresses) > 1:
                anchor_raw = centroid_addresses[1]
                # Mean all elements down to a single vector [D], then to [B, 1, D]
                # This kills the 3150-node leak immediately.
                anchor_vec = anchor_raw.reshape(-1, D).mean(dim=0)  # [D]
                anchor_final = anchor_vec.view(1, 1, D).expand(B, 1, D)
            else:
                anchor_final = nodes_proj.mean(dim=1, keepdim=True)  # [B, 1, D]

            # 3. Attention calculation
            # [B, N, D] * [B, 1, D] -> [B, N]
            attn_logits = torch.sum(nodes_proj * anchor_final, dim=-1)
            attn_weights = F.softmax(attn_logits / (D ** 0.5), dim=-1).unsqueeze(-1)  # [B, N, 1]

            # 4. Weighted sum over nodes (dim 1)
            # [B, N, D] * [B, N, 1] -> [B, D]
            subspace_context = torch.sum(nodes_proj * attn_weights, dim=1)

            # 5. Final Integration: [S, B, D] + [1, B, D]
            context_final = subspace_context.view(1, B, D)

            x_context = x + (Wi * context_final)
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        for layer in self.layers:
            x = layer(x)

        x = self.refiner(x)
        return self.stage_weight * x, gamma_divergence