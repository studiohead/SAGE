import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from ..config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS

class SageTransformer(DevelopmentalTransformer):
    """
    Stage 7: Global Topology Governance & Hierarchical Abstraction.
    Uses configurable hyperparameters from STAGE_HYPERPARAMS['Elder'].
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Elder"]

        # Governance Head: Projects reasoning paths into the trust-space
        self.governance_gate = nn.Linear(embed_dim, embed_dim)

        # Schema Induction: Recursive Meta-Policy (Hierarchical Abstraction)
        self.schema_induction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
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

    def forward(self, x, graph_matrix, adult_attn_map=None, Wi=1.0):
        # Apply trainable layers selection
        self._set_trainable_layers()

        gamma = torch.tensor(1.0, device=x.device)

        if adult_attn_map is not None:
            # 1. Reconstruct Active Path
            path_active = torch.matmul(adult_attn_map, graph_matrix).transpose(0, 1)

            # 2. Generate Governance Mask
            path_anchor = torch.sigmoid(self.governance_gate(path_active))

            # 3. Confidence Metric
            divergence = F.mse_loss(path_active, path_active * path_anchor)
            gamma = 1.0 - torch.clamp(divergence, 0, 1)

            # 4. Recursive Meta-Attention
            governed_context = path_active * path_anchor
            x = x + (Wi * governed_context)

        # 5. Schema Induction (Hierarchical Abstraction)
        out = self.schema_induction(x)

        return self.stage_weight * out, 1.0 - gamma

    def induce_super_node(self, node_cluster_embs):
        """
        Wisdom Operator:
        Collapses a cluster of nodes into a single abstract centroid
        (Meta-Policy Creation).
        """
        with torch.no_grad():
            cluster_mean = node_cluster_embs.mean(dim=0)
            return self.schema_induction(cluster_mean)
