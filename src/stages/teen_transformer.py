##############################################################################
# Teen | y = Gumbel-Softmax(x_internal, x_memory)
# Purpose: Discrete stochastic arbitration.
# Optimized: Actually uses the TCR (Retrieved Memory) provided by the Container.
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class TeenTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Teen"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.15)
            ) for _ in range(num_stage_layers)
        ])

        # Gate Predictor: 2 channels (0: Stay Internal, 1: Trust Graph/Memory)
        self.gate_predictor = nn.Linear(embed_dim, 2)

        # Bias toward internal processing for early stability
        with torch.no_grad():
            self.gate_predictor.bias.data = torch.tensor([1.0, -1.0])

        self.refiner_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [S_aug, B, D]
        graph_matrix: [N, D]
        centroid_addresses: Optional list of anchor tensors
        Wi: Plasticity weight
        """
        impact = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. GATE PREDICTION (Gumbel-Softmax Arbitration)
            gate_logits = self.gate_predictor(x)
            tau = max(0.5, 1.5 * Wi)

            if self.training:
                gate = F.gumbel_softmax(gate_logits, tau=tau, hard=True)
            else:
                gate_probs = F.softmax(gate_logits / tau, dim=-1)
                gate = torch.zeros_like(gate_probs).scatter_(
                    -1, gate_probs.argmax(-1, True), 1.0
                )

            # 2. WORLD ANCHOR (Contextual Baseline)
            if centroid_addresses and len(centroid_addresses) > 0:
                anchor_vec = centroid_addresses[0].reshape(-1, self.embed_dim).mean(0)
            else:
                anchor_vec = graph_matrix.mean(0)

            anchor_context = anchor_vec.view(1, 1, -1)

            # 3. CURIOSITY SIGNAL (Telemetry-only, no state stored)
            with torch.no_grad():
                token_dist = 1.0 - F.cosine_similarity(x, anchor_context, dim=-1)

                # Moderate-novelty emphasis (used downstream via telemetry)
                token_curiosity = torch.exp(-((token_dist - 0.5) ** 2) / 0.08)
                curiosity_score = token_curiosity.mean()

            # 4. COMPETITIVE INTEGRATION
            x_integrated = (gate[..., 0:1] * x) + (gate[..., 1:2] * anchor_context)

            # 5. IMPACT (Gamma Divergence Proxy)
            cos_sim = F.cosine_similarity(x_integrated, x, dim=-1).mean()
            impact = (1.0 - torch.clamp(cos_sim, 0, 1)).detach()

            x = self.norm(x_integrated)

        # 6. TRANSFORMER BACKBONE
        for layer in self.layers:
            x = layer(x)

        x = self.refiner_norm(x)

        # Return representation + impact only
        # Curiosity is intentionally NOT stored on self
        return self.stage_weight * x, impact
