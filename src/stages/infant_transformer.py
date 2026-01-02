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

        self.grounding_proj = nn.Linear(embed_dim, embed_dim)
        self.refiner = nn.LayerNorm(embed_dim)
        self.epsilon_scale = stage_cfg["epsilon_scale"]
        self.embed_dim = embed_dim

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        device = x.device

        # 1. ROBUST CENTROID EXTRACTION
        # Check for empty graph matrix [0, dim]
        if graph_matrix is not None and graph_matrix.size(0) > 0:
            if centroid_addresses is not None and len(centroid_addresses) > 2:
                graph_centroid = centroid_addresses[2].to(device)
            else:
                # Force calculation to stay on the hardware device
                graph_centroid = self.get_mean_field(graph_matrix).to(device)
        else:
            # SUTURE: If graph is empty, create a zero-anchor ON THE DEVICE
            graph_centroid = torch.zeros(self.embed_dim, device=device)

        # 2. PROJECTION (Now guaranteed to be on the same device)
        anchored_ground = self.grounding_proj(graph_centroid)

        if self.training:
            epsilon = torch.randn_like(anchored_ground) * (self.epsilon_scale * Wi)
            anchored_ground = anchored_ground + epsilon

        # 3. GLOBAL INTEGRATION
        context_contribution = (Wi * anchored_ground).view(1, 1, -1)
        x_context = x + context_contribution
        gamma_divergence = F.mse_loss(x_context, x).detach()

        # 4. BACKBONE
        x = self.norm(x_context)
        for layer in self.layers:
            x = layer(x)

        x = self.refiner(x)
        return self.stage_weight * x, gamma_divergence