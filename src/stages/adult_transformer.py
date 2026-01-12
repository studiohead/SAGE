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
        [PATENT REF 0014]: Active Reasoning Path & Centroid Cohesion

        Surgical Update: Divergence is now differentiable to act as
        structural pressure, penalizing drift from the long-term context.

        x: [batch, seq_len, dim]
        graph_matrix: [nodes, dim]
        """
        # Ensure x is [batch, seq, dim] for batch_first=True
        if x.dim() == 3 and x.size(0) != graph_matrix.size(0) and x.size(1) == graph_matrix.size(0):
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.shape
        gamma_divergence = torch.tensor(0.0, device=x.device, requires_grad=True)

        if graph_matrix is not None:
            # 1. PREPARE THE MEMORY POOL
            # Nodes represent the 'fixed' conceptual anchors for this stage
            nodes = graph_matrix.view(-1, self.embed_dim)

            # K_g/V_g shape: [batch, nodes, dim]
            K_g = self.k_proj(nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            V_g = self.v_proj(nodes).unsqueeze(0).repeat(batch_size, 1, 1)

            # 2. SURGICAL CROSS-ATTENTION (Knowledge Retrieval)
            # Query: Current Sequence | Key: Graph Manifold
            attn_out, attn_weights = self.cross_attn(
                query=x,
                key=K_g,
                value=V_g,
                need_weights=True
            )

            # 3. [PATENT REF 0014]: COMPUTE ACTIVE REASONING PATH
            # We allow gradient flow here so the projections (k_proj, v_proj)
            # learn to steer the reasoning path toward stable centroids.
            #avg_attn = attn_weights.mean(dim=1)  # [batch, nodes]
            epsilon = 5e-3
            avg_attn = attn_weights.mean(dim=1) + (torch.rand_like(attn_weights.mean(dim=1)) * epsilon)

            path_active = torch.matmul(avg_attn, nodes).mean(dim=0)  # [dim]

            # 4. DIFFERENTIABLE COHESION PRESSURE
            # Compare current path (Fast) to the long-term context (Slow Wisdom)
            # Long-term centroid is detached to act as a fixed 'North Star' for the gradient
            cohesion_sim = F.cosine_similarity(
                path_active.unsqueeze(0),
                self.long_term_centroid.detach().unsqueeze(0)
            )

            # This is the 'Friction' signal. High values indicate the Adult
            # is hallucinating logic outside the established manifold.
            gamma_divergence = 1.0 - cohesion_sim

            # 5. SELECTIVE INTEGRATION & RESIDUAL REFINEMENT
            # The 'Reasoning Path' is integrated into the sequence
            x = self.refiner_norm(x + (Wi * attn_out))

            # 6. PASSIVE CENTROID UPDATES (Exponential Weighted Moving Averages)
            # We update the buffers without tracking gradients to maintain temporal stability
            with torch.no_grad():
                self.short_term_centroid.copy_(
                    (self.alpha_fast * self.short_term_centroid) + ((1 - self.alpha_fast) * path_active)
                )
                self.long_term_centroid.copy_(
                    (self.alpha_slow * self.long_term_centroid) + ((1 - self.alpha_slow) * path_active)
                )

        # 7. TRANSFORMER BACKBONE
        # Processes the grounded sequence through deterministic layers
        for layer in self.layers:
            x = layer(x)

        # Telemetry provides the Elder stage with the 'Path History'
        # and the Centroid Loss needed for the total training objective.
        telemetry = {
            "gamma_divergence": gamma_divergence.detach(),  # Clean metric for logs
            "path_active": self.short_term_centroid,  # Rapid drift trace
            "context_anchor": self.long_term_centroid,  # Long-term stability
            "centroid_loss": gamma_divergence  # Gradient-active loss
        }

        return self.stage_weight * x, telemetry