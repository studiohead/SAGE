import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from ..config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS

class ToddlerTransformer(DevelopmentalTransformer):
    """
    Stage 2: Relational Orientation & Contextual Filtering.
    Uses configurable hyperparameters from STAGE_HYPERPARAMS['Toddler'].
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Toddler"]

        # Relational Bias
        self.relational_bias = nn.Parameter(torch.zeros(embed_dim))

        # Toddler Refinement: 2-layer depth
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Load hyperparameters from config
        self.epsilon_scale = stage_cfg["epsilon_scale"]
        self.training_layers = stage_cfg["training_layers"]
        self.learning_rate = stage_cfg["learning_rate"]
        self.weight_decay = stage_cfg["weight_decay"]
        self.dropout = stage_cfg["dropout"]
        self.gradient_clip = stage_cfg["gradient_clip"]
        self.plasticity_scale = stage_cfg["plasticity_scale"]
        self.scheduler_cfg = stage_cfg["scheduler"]

        # By default, train only the first N layers
        self.trainable_layer_range = (0, self.training_layers)

    def _set_trainable_layers(self):
        """
        Enables gradients for layers in trainable_layer_range and freezes the rest.
        """
        start, num_layers = self.trainable_layer_range
        layers = list(self.children())
        total_layers = len(layers)
        end = min(start + num_layers, total_layers)

        for i, layer in enumerate(layers):
            requires_grad = start <= i < end
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def forward(self, x, graph_matrix, Wi=1.0):
        # Apply trainable layers selection
        self._set_trainable_layers()

        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # Compute the global centroid of the graph
            graph_centroid = graph_matrix.mean(dim=0)

            if self.training:
                epsilon = torch.randn_like(graph_centroid) * (self.epsilon_scale * Wi)
                graph_centroid = graph_centroid + epsilon

            # Stage 2 Hadamard Gating
            context_filter = torch.sigmoid(graph_centroid + self.relational_bias)
            x_gated = x * context_filter
            x_context = x + (Wi * x_gated)

            gamma_divergence = (x_context - x).pow(2).mean().detach()
            x = self.norm(x_context)

        # Toddler 2-layer refinement
        for _ in range(2):
            x = self.refiner(x) + x

        return self.stage_weight * x, gamma_divergence
