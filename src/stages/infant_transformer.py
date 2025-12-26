# src/stages/infant_transformer.py

##############################################################################
# Infant | y = x + (W * mean(G)) + ε
# Purpose:
# Stochastic grounding stage with high plasticity.
# Introduces noise (ε) to encourage exploratory representations.
# Graph influence is averaged globally, prioritizing stability over structure.
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class InfantTransformer(DevelopmentalTransformer):
    """
    Stage 1: Stochastic Grounding & Global Manifold Anchoring.
    Logic: y = x + (W * mean(G)) + ε
    Foundational grounding anchored to the Slow-Scale (Global) Centroid.
    """

    def __init__(self):
        # 1. Pull Shared Architecture from Config
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # 2. Pull Stage-Specific Hyperparams from Config
        stage_cfg = STAGE_HYPERPARAMS["Infant"]

        # Calculate how many layers this stage owns based on config boundaries
        # This removes the hardcoded '6' and respects the 48-layer stack.
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # 3. Dynamic Layer Allocation
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.1)
            ) for _ in range(num_stage_layers)
        ])

        # 4. Grounding Projection (W in the patent logic)
        self.grounding_proj = nn.Linear(embed_dim, embed_dim)

        # 5. Output Refinement
        self.refiner = nn.LayerNorm(embed_dim)

        self.epsilon_scale = stage_cfg["epsilon_scale"]
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)

        # 1. TOPOLOGICAL GROUNDING
        if graph_matrix is not None:
            # Initialize the centroid first
            if centroid_addresses and len(centroid_addresses) > 2:
                graph_centroid = centroid_addresses[2]
            else:
                # Fallback: Compute global mean from the graph matrix
                graph_centroid = graph_matrix

            # Now safely collapse dimensions to a single [Dim] vector
            while graph_centroid.dim() > 1:
                graph_centroid = graph_centroid.mean(dim=0)

            # Stochastic Exploration (ε)
            if self.training:
                epsilon = torch.randn_like(graph_centroid) * (self.epsilon_scale * Wi)
                graph_centroid = graph_centroid + epsilon

            # Project and Integrate
            anchored_ground = self.grounding_proj(graph_centroid)

            # Use view for clean 3D broadcasting: [1, 1, Dim]
            context_contribution = (Wi * anchored_ground).view(1, 1, -1)
            x_context = x + context_contribution

            # Comparative Geometric Divergence (Gamma)
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # 2. TRANSFORMER PROCESSING
        for layer in self.layers:
            x = layer(x)

        # 3. OUTPUT REFINEMENT
        x = self.refiner(x)

        return self.stage_weight * x, gamma_divergence