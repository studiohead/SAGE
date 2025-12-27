###########################################################################################
# STAGE 6: ADULT | y = Softmax((Q_x * K_G^T)/sqrt(d_k)) * V_G
# Purpose: High-Fidelity Multi-Head Retrieval from the Full Manifold.
# Patent Ref [0014]: Generates Active Reasoning Path (Path_active) and
# Multi-Scale Context Centroids (EWMA) for Comparative Geometric Divergence (Γ).
###########################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class AdultTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Adult"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.2)
            ) for _ in range(num_stage_layers)
        ])

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.refiner_norm = nn.LayerNorm(embed_dim)

        # [PATENT REF 0014]: Multi-Scale Context Centroid Buffers
        # We store moving averages of the reasoning path at different decay rates
        self.register_buffer("short_term_centroid", torch.zeros(embed_dim))
        self.register_buffer("long_term_centroid", torch.zeros(embed_dim))
        self.alpha_fast = 0.9  # Rapid micro-drift
        self.alpha_slow = 0.99  # Stable context

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [batch, seq_len, dim]
        graph_matrix: [nodes, dim]
        """
        # Ensure x is [batch, seq, dim] for batch_first=True
        if x.dim() == 3 and x.size(0) != graph_matrix.size(0) and x.size(1) == graph_matrix.size(0):
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.shape
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. PREPARE THE MEMORY POOL
            nodes = graph_matrix.view(-1, self.embed_dim)

            # K_g/V_g shape: [batch, nodes, dim]
            K_g = self.k_proj(nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            V_g = self.v_proj(nodes).unsqueeze(0).repeat(batch_size, 1, 1)

            # 2. SURGICAL CROSS-ATTENTION (Retrieval)
            attn_out, attn_weights = self.cross_attn(
                query=x,
                key=K_g,
                value=V_g,
                need_weights=True
            )

            # 3. [PATENT REF 0014]: COMPUTE ACTIVE REASONING PATH
            # Path_active = adult_attn_map × graph_matrix
            # attn_weights: [batch, seq, nodes]
            with torch.no_grad():
                # Average attention across the sequence to get the 'trajectory'
                avg_attn = attn_weights.mean(dim=1)  # [batch, nodes]
                path_active = torch.matmul(avg_attn, nodes).mean(dim=0)  # [dim]

                # Update Multi-Scale Centroids (EWMA)
                self.short_term_centroid = (self.alpha_fast * self.short_term_centroid) + (
                            (1 - self.alpha_fast) * path_active)
                self.long_term_centroid = (self.alpha_slow * self.long_term_centroid) + (
                            (1 - self.alpha_slow) * path_active)

                # 4. COMPUTE INTERIM GAMMA (Divergence between fast and slow scales)
                # This signals the Elder that the model is 'drifting' from context
                gamma_divergence = 1.0 - F.cosine_similarity(self.short_term_centroid.unsqueeze(0),
                                                             self.long_term_centroid.unsqueeze(0))

            x = self.refiner_norm(x + (Wi * attn_out))

        # 5. TRANSFORMER BACKBONE
        for layer in self.layers:
            x = layer(x)

        # Telemetry now includes path_active and centroids for the Elder Stage
        telemetry = {
            "gamma_divergence": gamma_divergence,
            "path_active": self.short_term_centroid,  # The 'trace'
            "context_anchor": self.long_term_centroid  # The 'stable reference'
        }

        return self.stage_weight * x, telemetry