##############################################################################
# Preschool | Adaptive Relational Grounding
# Purpose:
# Saliency-weighted node prioritization via Latent Cross-Attention.
# Bridging Toddler (Filtering) and Gradeschool (Projection).
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class PreschoolTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Preschool"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.12)
            ) for _ in range(num_stage_layers)
        ])

        # THE RELATIONAL BRIDGE
        # These project the manifold into a 'Key-Value' memory space
        self.graph_key_proj = nn.Linear(embed_dim, embed_dim)
        self.graph_val_proj = nn.Linear(embed_dim, embed_dim)

        # Saliency Gate: Decides the intensity of the grounding per token
        self.relational_gate = nn.Linear(embed_dim, 1)

        self.refiner = nn.LayerNorm(embed_dim)
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, embed_dim]
        graph_matrix: [nodes, embed_dim]
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. RECONSTRUCT MANIFOLD
            # Treat the global manifold as a pool of available concept nodes
            nodes = graph_matrix.reshape(-1, self.embed_dim)

            # 2. GENERATE GRAPH KEYS & VALUES
            # We map the graph nodes into a space where they can be queried
            K_g = self.graph_key_proj(nodes)  # [Nodes, Dim]
            V_g = self.graph_val_proj(nodes)  # [Nodes, Dim]

            # 3. CROSS-ATTENTION SALIENCY
            # Each token (Query) looks at the Graph (Keys) to find relevant concepts
            # x: [S, B, E] -> q: [B, S, E]
            q = x.transpose(0, 1)

            # Compute Saliency Scores: [B, S, Nodes]
            # How much does 'Token X' care about 'Node Y'?
            attn_scores = torch.matmul(q, K_g.transpose(0, 1)) / (self.embed_dim ** 0.5)
            saliency_weights = F.softmax(attn_scores, dim=-1)

            # Extract Essence: [B, S, E]
            # This is the 'Knowledge Retrieval' from the manifold
            essence = torch.matmul(saliency_weights, V_g).transpose(0, 1)

            # 4. DYNAMIC RELATIONAL GATING
            # Learns if the current context actually benefits from the graph
            gate = torch.sigmoid(self.relational_gate(x))

            # 5. SELECTIVE INTEGRATION
            # We use a 0.5 multiplier to allow significant influence
            # while maintaining the residual identity of the transformer.
            x_context = x + (0.5 * Wi * gate * essence)

            # 6. TELEMETRY
            # Captured before Norm to see the true Relational Friction
            gamma_divergence = F.mse_loss(x_context, x).detach()

            x = self.norm(x_context)

        # 7. TRANSFORMER BACKBONE
        for layer in self.layers:
            x = layer(x)

        x = self.refiner(x)
        return self.stage_weight * x, gamma_divergence