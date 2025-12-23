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
        # 1. Pull Shared Architecture from Config
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # 2. Pull Stage-Specific Hyperparams from Config
        stage_cfg = STAGE_HYPERPARAMS["Gradeschool"]

        # Calculate dynamic layer count from config boundaries
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # 3. Dynamic Layer Allocation (Replaces manual refiner)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.15)
            ) for _ in range(num_stage_layers)
        ])

        # 4. Subspace Projection (W_proj) - Rotates the manifold
        self.subspace_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 5. Integration Weight (W)
        self.integration_weight = nn.Parameter(torch.ones(1) * 0.1)

        # 6. Output Refinement
        self.refiner_norm = nn.LayerNorm(embed_dim)

        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        centroid_addresses: [Fast, Mid, Slow] - Gradeschool uses Mid (Index 1)
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)

        # 1. STRUCTURAL SUBSPACING (Patent Logic)
        if graph_matrix is not None:
            # Access Mid-Scale Contextual Centroid (Paragraph-level context)
            if centroid_addresses and len(centroid_addresses) > 1:
                anchor_z = centroid_addresses[1]
            else:
                anchor_z = graph_matrix.mean(dim=0)

            # Linear Projection of the Manifold (W_proj)
            # Aligns global graph knowledge with the stage's categorical space
            projected_memory = self.subspace_proj(anchor_z)

            # Knowledge Integration modulated by plasticity and learned weight
            update_signal = (Wi * self.integration_weight * projected_memory).unsqueeze(0).unsqueeze(0)
            x_integrated = x + update_signal

            # Comparative Geometric Divergence (Gamma)
            gamma_divergence = F.mse_loss(x_integrated, x).detach()
            x = self.norm(x_integrated)

        # 2. TRANSFORMER PROCESSING (Aperture managed by SAGEContainer)
        for layer in self.layers:
            x = layer(x)

        # 3. OUTPUT REFINEMENT
        x = self.refiner_norm(x)

        return self.stage_weight * x, gamma_divergence