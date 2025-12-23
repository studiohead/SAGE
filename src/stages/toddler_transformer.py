# src/stages/toddler_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class ToddlerTransformer(DevelopmentalTransformer):
    """
    Stage 2: Relational Orientation & Contextual Filtering.
    Logic: y = x + (W ⊙ mean(G))
    Uses Element-wise Hadamard gating to filter inputs based on Graph Centroids.
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Toddler"]

        # Relational Bias (W in the patent logic)
        self.relational_weight = nn.Parameter(torch.randn(embed_dim))

        # Toddler Refinement: 2-layer shallow depth
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.epsilon_scale = stage_cfg["epsilon_scale"]
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
        centroid_addresses: [Fast, Mid, Slow] - Toddler anchors to Mid (Index 1)
        """
        self._set_trainable_layers()
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. Access Mid-Scale Contextual Centroid
            if centroid_addresses and len(centroid_addresses) > 1:
                graph_centroid = centroid_addresses[1]
            else:
                graph_centroid = graph_matrix.mean(dim=0)

            # 2. Stochastic Grounding (Epsilon Noise)
            # High in Toddler stage (0.08) to encourage exploration of the graph manifold
            if self.training:
                epsilon = torch.randn_like(graph_centroid) * (self.epsilon_scale * Wi)
                graph_centroid = graph_centroid + epsilon

            # 3. Element-wise Hadamard Gating (Patent Logic)
            # Learns to mask specific dimensions of the embedding space
            context_filter = torch.sigmoid(graph_centroid * self.relational_weight)

            # Apply gated context to input tokens
            # Wi (Plasticity) dictates how much the gated signal influences the state
            x_context = x + (Wi * (x * context_filter.unsqueeze(0).unsqueeze(0)))

            # 4. Comparative Geometric Divergence
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # 5. Toddler 2-layer shallow refinement
        for _ in range(2):
            x = self.refiner(x) + x

        # Return output and the filter mask for Auditor monitoring
        return self.stage_weight * x, gamma_divergence