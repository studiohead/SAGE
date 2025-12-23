# src/stages/gradeschool_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class GradeschoolTransformer(DevelopmentalTransformer):
    """
    Stage 4: Structural Subspacing & Categorical Abstraction.
    Logic: y = x + W * (G * W_proj)
    Learns to project the Graph Manifold into an internal categorical subspace.
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Gradeschool"]

        # Subspace Projection (W_proj) - Rotates the manifold
        self.subspace_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Integration Weight (W)
        self.integration_weight = nn.Parameter(torch.ones(1) * 0.1)

        # Structural Refinement: 4-layer depth for categorical stability
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.plasticity_scale = stage_cfg["plasticity_scale"]
        self.training_layers = stage_cfg["training_layers"]
        self.trainable_layer_range = (0, self.training_layers)

    def _set_trainable_layers(self):
        start, num_layers = self.trainable_layer_range
        for i, layer in enumerate(self.children()):
            requires_grad = start <= i < (start + num_layers)
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        centroid_addresses: [Fast, Mid, Slow] - Gradeschool uses Mid (Index 1)
        """
        self._set_trainable_layers()
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. Access Mid-Scale Contextual Centroid (Paragraph-level context)
            if centroid_addresses and len(centroid_addresses) > 1:
                anchor_z = centroid_addresses[1]  # Mid-Scale Alpha
            else:
                anchor_z = graph_matrix.mean(dim=0)

            # 2. Linear Projection of the Manifold (W_proj)
            # This aligns the global graph knowledge with the stage's categorical space
            projected_memory = self.subspace_proj(anchor_z)  # [dim]

            # 3. Knowledge Integration
            # Modulated by global plasticity (Wi) and learned integration weight
            update_signal = (Wi * self.integration_weight * projected_memory).unsqueeze(0).unsqueeze(0)
            x_integrated = x + update_signal

            # 4. Comparative Geometric Divergence
            # Detects if the manifold rotation is causing "Shearing" (unstable updates)
            gamma_divergence = F.mse_loss(x_integrated, x).detach()
            x = self.norm(x_integrated)

        # 5. Categorical Refinement (4-layer Structural logic)
        for _ in range(4):
            x = self.refiner(x) + x

        # Return output and the update trace (projected memory vector)
        return self.stage_weight * x, gamma_divergence