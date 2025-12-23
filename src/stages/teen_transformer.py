# src/stages/teen_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class TeenTransformer(DevelopmentalTransformer):
    """
    Stage 5: Competitive Abstraction & Stochastic Routing.
    Logic: y = Gumbel-Softmax(x, W * G)
    Arbitrates between internal representation and Global Graph Centroids.
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Teen"]

        # Competitive Gate: Projects internal state vs Graph Centroid
        # Index 0: Internal Priority, Index 1: Graph/Centroid Priority
        self.gate_predictor = nn.Linear(embed_dim, 2)

        # Deep Refinement: 5-layer reasoning depth
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
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
        centroid_addresses: List of scale addresses [Fast, Mid, Slow]
        """
        self._set_trainable_layers()
        gamma = torch.tensor(1.0, device=x.device)

        if graph_matrix is not None:
            # 1. Access Multi-Scale Anchor (Fast Scale)
            # Teen logic focuses on immediate context alignment
            if centroid_addresses:
                y_graph = centroid_addresses[0].unsqueeze(0).unsqueeze(0)
            else:
                y_graph = graph_matrix.mean(dim=0).unsqueeze(0).unsqueeze(0)

            # 2. Discrete Stochastic Arbitration (Gumbel-Softmax)
            # Wi acts as the temperature: Higher plasticity = more stochastic exploration
            gate_logits = self.gate_predictor(x)

            if self.training:
                # Tau (temperature) scales with plasticity (Wi)
                gate = F.gumbel_softmax(gate_logits, tau=max(0.1, Wi), hard=True)
            else:
                gate = F.softmax(gate_logits, dim=-1)

            # 3. Path Selection
            x_internal = x
            # Align graph centroid to batch/sequence size
            x_anchored = y_graph.expand(x.size(0), x.size(1), -1)

            # Combine based on Gumbel Gate
            x_integrated = (gate[..., 0:1] * x_internal) + (gate[..., 1:2] * x_anchored)

            # 4. Comparative Geometric Divergence
            # Divergence is high when the gate forces a jump between internal and anchored states
            divergence = F.mse_loss(x_integrated, x_internal)
            gamma = 1.0 - torch.clamp(divergence, 0, 1)

            x = self.norm(x_integrated)

        # 5. Reasoning Pass
        for _ in range(5):
            x = self.refiner(x) + x

        # Return governed output and Impact trace (Gumbel decision map)
        # Note: We return the gate map so the Auditor can see "Routing Breaches"
        return self.stage_weight * x, gate