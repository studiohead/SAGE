import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from ..config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS

class AdultTransformer(DevelopmentalTransformer):
    """
    Stage 6: Selective Relational Retrieval.
    Uses configurable hyperparameters from STAGE_HYPERPARAMS['Adult'].
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Adult"]

        # Surgical Cross-Attention: Retrieves specific conceptual instances
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead)

        # Deep Reasoning Pass: 6-layer refinement for relational depth
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

    def graph_projection(self, graph_matrix):
        """
        Placeholder for projecting the graph into the Adult's relational subspace.
        Can be replaced with Stage 4 subspace logic or learned projection.
        """
        return graph_matrix.mean(dim=0)  # [embed_dim]

    def forward(self, x, graph_matrix, Wi=1.0, return_attn=True):
        # Apply trainable layers selection
        self._set_trainable_layers()

        attn_map = None

        if graph_matrix is not None:
            G_projected = self.graph_projection(graph_matrix)
            k_v = G_projected.unsqueeze(1).expand(-1, x.size(1), -1)

            attn_out, attn_map = self.cross_attn(
                query=x,
                key=k_v,
                value=k_v,
                need_weights=True,
                average_attn_weights=False
            )

            x = self.norm(x + (Wi * attn_out))

        for _ in range(6):
            x = self.refiner(x) + x

        if return_attn:
            return self.stage_weight * x, attn_map

        return self.stage_weight * x
