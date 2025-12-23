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
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Preschool"]

        # Saliency Head: Projects Graph Nodes into Relevance Space
        self.saliency_proj = nn.Linear(embed_dim, embed_dim)
        self.relevance_query = nn.Parameter(torch.randn(1, embed_dim))

        # Preschool Refinement: 3-layer depth for conceptual grounding
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.plasticity_scale = stage_cfg["plasticity_scale"]
        self.training_layers = stage_cfg["training_layers"]
        self.trainable_layer_range = (0, self.training_layers)

    def _set_trainable_layers(self):
        start, num_layers = self.trainable_layer_range
        for i, layer in enumerate(self.children()):
            requires_grad = start <= i < (start + num_layers)
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        centroid_addresses: [Fast, Mid, Slow] - Preschool uses Slow (Index 2)
        """
        self._set_trainable_layers()
        gamma_divergence = torch.tensor(0.0, device=x.device)
        saliency_weights = None

        if graph_matrix is not None:
            # 1. Anchor-Guided Relevance
            # Use the Slow-Scale Centroid (Stable Truth) to weight node importance
            if centroid_addresses and len(centroid_addresses) > 2:
                anchor_slow = centroid_addresses[2]
            else:
                anchor_slow = graph_matrix.mean(dim=0)

            # 2. Compute Saliency (Relevance Logits)
            # Dot product of Projected Graph and the Slow Anchor
            G_projected = self.saliency_proj(graph_matrix)  # [num_nodes, dim]
            relevance_logits = torch.matmul(G_projected, anchor_slow.unsqueeze(-1))  # [num_nodes, 1]

            # Saliency-weighted prioritization (Softmax)
            saliency_weights = F.softmax(relevance_logits / (self.embed_dim ** 0.5), dim=0)

            # 3. Knowledge Integration (The "Essence" of the manifold)
            focused_essence = torch.sum(graph_matrix * saliency_weights, dim=0)  # [dim]

            # Add essence to tokens, modulated by plasticity (Wi)
            x_context = x + (Wi * focused_essence).unsqueeze(0).unsqueeze(0)

            # 4. Comparative Geometric Divergence
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # 5. Conceptual Refinement (3-layer)
        for _ in range(3):
            x = self.refiner(x) + x

        # Return output and the saliency map for Auditor forensic tracking
        return self.stage_weight * x, saliency_weights