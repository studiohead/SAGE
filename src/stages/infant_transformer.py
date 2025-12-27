##############################################################################
# Infant | y = x + (W * mean(G)) + ε
# Purpose:
# Stochastic grounding stage with high plasticity.
# Optimized: Uses Direct Centroid Anchoring for efficiency.
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class InfantTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Infant"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.1)
            ) for _ in range(num_stage_layers)
        ])

        # Grounding Projection (W)
        self.grounding_proj = nn.Linear(embed_dim, embed_dim)
        self.refiner = nn.LayerNorm(embed_dim)

        self.epsilon_scale = stage_cfg["epsilon_scale"]
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        centroid_addresses: List from Auditor [addr_fast, addr_mid, addr_slow]
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. DIRECT CENTROID ANCHORING
            # We prioritize the 'Slow-Scale' address (index 2) from the Auditor
            # as it represents the most stable global mean of the manifold.
            if centroid_addresses is not None and len(centroid_addresses) > 2:
                graph_centroid = centroid_addresses[2]
            else:
                # Optimized Fallback: Single-pass mean is sufficient
                graph_centroid = graph_matrix.view(-1, self.embed_dim).mean(dim=0)

            # 2. STOCHASTIC INJECTION (ε)
            # We apply epsilon to the projected space to ensure the noise
            # actually challenges the Transformer's stability.
            anchored_ground = self.grounding_proj(graph_centroid)

            if self.training:
                # Epsilon must be scaled by Wi to decay as the stage stabilizes
                epsilon = torch.randn_like(anchored_ground) * (self.epsilon_scale * Wi)
                anchored_ground = anchored_ground + epsilon

            # 3. GLOBAL INTEGRATION
            # y = x + (W * mean(G))
            context_contribution = (Wi * anchored_ground).view(1, 1, -1)
            x_context = x + context_contribution

            # 4. TELEMETRY
            # Captured before the norm to measure raw 'Plastic Pressure'
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # 5. TRANSFORMER BACKBONE
        for layer in self.layers:
            x = layer(x)

        x = self.refiner(x)
        return self.stage_weight * x, gamma_divergence