import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from ..config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS

class PreschoolTransformer(DevelopmentalTransformer):
    """
    Stage 3: Global Saliency & Static Relevance.
    Uses configurable hyperparameters from STAGE_HYPERPARAMS['Preschool'].
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Preschool"]

        # Saliency Head: identifies globally relevant nodes
        self.saliency_head = nn.Linear(embed_dim, 1)

        # Preschool Refinement: 3-layer depth
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
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
            # Compute global saliency
            relevance_logits = self.saliency_head(graph_matrix)
            saliency_weights = F.softmax(relevance_logits, dim=0)
            focused_essence = torch.sum(graph_matrix * saliency_weights, dim=0)

            # Knowledge integration
            x_context = x + (Wi * focused_essence)

            # Passive witness monitoring
            gamma_divergence = (x_context - x).pow(2).mean().detach()
            x = self.norm(x_context)

        # Conceptual refinement (3-layer)
        for _ in range(3):
            x = self.refiner(x) + x

        return self.stage_weight * x, gamma_divergence
