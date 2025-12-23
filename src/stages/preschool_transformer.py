# src/stages/preschool_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class PreschoolTransformer(DevelopmentalTransformer):
    """
    Stage 3: Global Saliency & Static Relevance.
    Logic: y = x + W * sum(Softmax(G) * G)
    Identifies globally relevant nodes guided by the Slow-Scale Anchor.
    """

    def __init__(self):
        # 1. Pull Shared Architecture from Config
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # 2. Pull Stage-Specific Hyperparams from Config
        stage_cfg = STAGE_HYPERPARAMS["Preschool"]

        # Calculate dynamic layer count (e.g., layers 8-12 = 4 layers)
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # 3. Dynamic Layer Allocation
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.12)
            ) for _ in range(num_stage_layers)
        ])

        # 4. Saliency Head: Projects Graph Nodes into Relevance Space
        self.saliency_proj = nn.Linear(embed_dim, embed_dim)
        self.relevance_query = nn.Parameter(torch.randn(1, embed_dim))

        # 5. Output Refinement
        self.refiner = nn.LayerNorm(embed_dim)

        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        centroid_addresses: [Fast, Mid, Slow] - Preschool uses Slow (Index 2)
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)
        saliency_weights = None

        # 1. GLOBAL SALIENCY INTEGRATION (Patent Logic)
        if graph_matrix is not None:
            # Use the Slow-Scale Centroid (Stable Truth) to weight node importance
            if centroid_addresses and len(centroid_addresses) > 2:
                anchor_slow = centroid_addresses[2]
            else:
                anchor_slow = graph_matrix.mean(dim=0)

            # Compute Saliency (Relevance Logits)
            # Dot product of Projected Graph and the Slow Anchor
            G_projected = self.saliency_proj(graph_matrix)  # [num_nodes, dim]
            relevance_logits = torch.matmul(G_projected, anchor_slow.unsqueeze(-1))  # [num_nodes, 1]

            # Saliency-weighted prioritization (Softmax)
            saliency_weights = F.softmax(relevance_logits / (self.embed_dim ** 0.5), dim=0)

            # Knowledge Integration (The "Essence" of the manifold)
            focused_essence = torch.sum(graph_matrix * saliency_weights, dim=0)  # [dim]

            # Add essence to tokens, modulated by global plasticity (Wi)
            x_context = x + (Wi * focused_essence).unsqueeze(0).unsqueeze(0)

            # Comparative Geometric Divergence (Gamma)
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # 2. TRANSFORMER PROCESSING (Sliding Window Managed by Container)
        for layer in self.layers:
            x = layer(x)

        # 3. OUTPUT REFINEMENT
        x = self.refiner(x)

        # Return output and the divergence for monitoring
        return self.stage_weight * x, gamma_divergence