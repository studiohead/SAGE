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
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.15)
            ) for _ in range(num_stage_layers)
        ])

        # Gumbel gate predictor: 0 = internal, 1 = anchor/graph
        self.gate_predictor = nn.Linear(embed_dim, 2)
        self.refiner_norm = nn.LayerNorm(embed_dim)
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, embed_dim]
        centroid_addresses: List of scale addresses [Fast, Mid, Slow]
        """
        gamma = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            seq_len, batch, embed_dim = x.shape

            # 1. Use Fast-scale centroid or mean graph
            if centroid_addresses and len(centroid_addresses) > 0:
                anchor = centroid_addresses[0]
            else:
                anchor = graph_matrix.mean(dim=0)  # [batch, embed_dim]

            # Ensure shape: [1, batch, embed_dim] for broadcast
            anchor_context = anchor.unsqueeze(0)
            if anchor_context.shape[1] != batch:
                anchor_context = anchor_context.expand(1, batch, embed_dim)
            anchor_context = anchor_context.expand(seq_len, batch, embed_dim)

            # 2. Gumbel-softmax gate
            gate_logits = self.gate_predictor(x)  # [seq, batch, 2]
            if self.training:
                gate = F.gumbel_softmax(gate_logits, tau=max(0.1, Wi), hard=True)
            else:
                gate = F.softmax(gate_logits, dim=-1)

            # 3. Integrate paths with proper broadcasting
            x_integrated = gate[..., 0:1] * x + gate[..., 1:2] * anchor_context

            # 4. Comparative Geometric Divergence
            gamma = F.mse_loss(x_integrated, x).detach()
            x = self.norm(x_integrated)

        # 5. Transformer stack
        for layer in self.layers:
            x = layer(x)

        # 6. Output refinement
        x = self.refiner_norm(x)

        return self.stage_weight * x, gamma
