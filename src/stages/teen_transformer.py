import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from ..config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS

class TeenTransformer(DevelopmentalTransformer):
    """
    Stage 5: Competitive Abstraction & Stochastic Routing.
    Uses configurable hyperparameters from STAGE_HYPERPARAMS['Teen'].
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Teen"]

        # Competitive Gate: Learns to switch based on EWMA Centroid proximity
        self.gate_predictor = nn.Linear(embed_dim, 2)

        # Deep Refinement: 5-layer reasoning depth
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
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

        gamma = torch.tensor(1.0, device=x.device)

        if graph_matrix is not None:
            y_graph = graph_matrix.mean(dim=0)

            gate_logits = self.gate_predictor(x)
            if self.training:
                gate = F.gumbel_softmax(gate_logits, tau=max(0.1, Wi), hard=True)
            else:
                gate = F.softmax(gate_logits, dim=-1)

            x_internal = x
            x_anchored = y_graph
            x_integrated = (gate[..., 0:1] * x_internal) + (gate[..., 1:2] * x_anchored)

            divergence = (x_integrated - x_internal).pow(2).mean()
            gamma = 1.0 - torch.clamp(divergence, 0, 1)

            x = self.norm(x_integrated)

        for _ in range(5):
            x = self.refiner(x) + x

        return self.stage_weight * x, 1.0 - gamma
