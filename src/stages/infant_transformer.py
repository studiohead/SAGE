# src/stages/infant_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class InfantTransformer(DevelopmentalTransformer):
    """
    Stage 1: Stochastic Grounding & Global Manifold Anchoring.
    Logic: y = x + (W * mean(G)) + ε
    Foundational grounding anchored to the Slow-Scale (Global) Centroid.
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Infant"]

        # Learned Grounding Projection (W in the patent logic)
        self.grounding_proj = nn.Linear(embed_dim, embed_dim)

        # Infant Refinement: 1-layer shallow refinement
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
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
        centroid_addresses: [Fast, Mid, Slow] - Infant anchors to Slow (Index 2)
        """
        self._set_trainable_layers()
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. Access Slow-Scale (Global) Centroid
            # This is the "Immutable Truth" the infant is grounding against
            if centroid_addresses and len(centroid_addresses) > 2:
                graph_centroid = centroid_addresses[2]
            else:
                graph_centroid = graph_matrix.mean(dim=0)

            # 2. Stochastic Exploration (ε)
            # High plasticity grounding allows the model to 'jitter' around the truth
            if self.training:
                epsilon = torch.randn_like(graph_centroid) * (self.epsilon_scale * Wi)
                graph_centroid = graph_centroid + epsilon

            # 3. Grounding Projection
            # Projects global graph state into the current sequence context
            anchored_ground = self.grounding_proj(graph_centroid)

            # Integrate global context modulated by Wi (Plasticity)
            x_context = x + (Wi * anchored_ground).unsqueeze(0).unsqueeze(0)

            # 4. Comparative Geometric Divergence
            # Monitoring how much the foundational grounding shifts the input tokens
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # 5. Shallow Refinement
        x = self.refiner(x) + x

        return self.stage_weight * x, gamma_divergence