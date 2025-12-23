# src/stages/gradeschool_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class GradeschoolTransformer(DevelopmentalTransformer):
    """
    Stage 4: Structural Subspacing & Categorical Abstraction.
    Logic: y = x + Projected(G_nodes, centroid_mid)
    Learns to project the Graph Manifold into an internal categorical subspace.
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Gradeschool"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # Transformer stack
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.15)
            ) for _ in range(num_stage_layers)
        ])

        # Subspace projection (learn categorical axes)
        self.subspace_proj = nn.Linear(embed_dim, embed_dim)
        self.refiner = nn.LayerNorm(embed_dim)
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        graph_matrix: [nodes, batch, dim]
        centroid_addresses: [fast, mid, slow]
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. Use Mid-Scale Centroid for categorical alignment
            if centroid_addresses and len(centroid_addresses) > 1:
                anchor_mid = centroid_addresses[1]
            else:
                anchor_mid = graph_matrix.mean(dim=0)  # [batch, dim]

            # 2. Project node embeddings into categorical subspace
            nodes_proj = self.subspace_proj(graph_matrix)  # [nodes, batch, dim]

            # 3. Compute attention weights from projected nodes to centroid
            # anchor_mid: [batch, dim] -> [1, batch, dim] for broadcasting
            anchor_exp = anchor_mid.unsqueeze(0)
            attn_logits = torch.sum(nodes_proj * anchor_exp, dim=-1)  # [nodes, batch]
            attn_weights = F.softmax(attn_logits / (x.size(-1) ** 0.5), dim=0)  # [nodes, batch]

            # 4. Weighted aggregation of projected nodes
            # nodes_proj: [nodes, batch, dim], attn_weights: [nodes, batch, 1]
            weighted_nodes = nodes_proj * attn_weights.unsqueeze(-1)
            subspace_context = weighted_nodes.sum(dim=0)  # [batch, dim]

            # 5. Broadcast to sequence dimension
            subspace_context = subspace_context.unsqueeze(0)  # [1, batch, dim]

            # 6. Integrate with input
            x_context = x + Wi * subspace_context

            # 7. Gamma divergence measures actual structural movement
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # Transformer processing
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = self.refiner(x)

        return self.stage_weight * x, gamma_divergence
