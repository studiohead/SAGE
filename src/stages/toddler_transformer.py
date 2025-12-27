##############################################################################
# Toddler | y = x + (Wi * (x ⊙ Sigmoid(Centroid ⊙ W)))
# Purpose:
# Relational gating via element-wise modulation.
# Maintains global graph averaging while introducing selective weighting.
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class ToddlerTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Toddler"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.1)
            ) for _ in range(num_stage_layers)
        ])

        # Relational Weight (W): Learns which latent features the graph should 'unlock'
        # Initialized to zeros so sigmoid starts at 0.5 (neutral)
        self.relational_weight = nn.Parameter(torch.zeros(embed_dim))

        self.refiner = nn.LayerNorm(embed_dim)
        self.epsilon_scale = stage_cfg["epsilon_scale"]
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. ROBUST CENTROID EXTRACTION
            # Toddler focuses on 'Mid-Scale' addresses (index 1) - local neighborhood stability
            if centroid_addresses is not None and len(centroid_addresses) > 1:
                graph_centroid = centroid_addresses[1]
            else:
                graph_centroid = graph_matrix.view(-1, self.embed_dim).mean(dim=0)

            # 2. STOCHASTIC GROUNDING
            if self.training:
                epsilon = torch.randn_like(graph_centroid) * (self.epsilon_scale * Wi)
                graph_centroid = graph_centroid + epsilon

            # 3. HADAMARD GATING (The Patent Logic)
            # We filter the current latent state 'x' by the graph's 'opinion'
            gate = torch.sigmoid(graph_centroid * self.relational_weight)

            # Align gate for broadcasting across [Seq, Batch, Dim]
            if gate.dim() == 1:
                gate = gate.view(1, 1, -1)
            elif gate.dim() == 2:
                gate = gate.unsqueeze(0)

            # 4. SELECTIVE INTEGRATION
            # The Toddler doesn't add the graph; it allows the graph to 'shape' the identity.
            grounding_signal = Wi * (x * gate)
            x_context = x + grounding_signal

            # 5. TELEMETRY
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # 6. TRANSFORMER PROCESSING
        for layer in self.layers:
            x = layer(x)

        x = self.refiner(x)
        return self.stage_weight * x, gamma_divergence