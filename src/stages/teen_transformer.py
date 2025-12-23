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
        # 1. Pull Shared Architecture from Config
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # 2. Pull Stage-Specific Hyperparams from Config
        stage_cfg = STAGE_HYPERPARAMS["Teen"]

        # Calculate dynamic layer count from config boundaries
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # 3. Dynamic Layer Allocation (The 48-layer stack slice)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.15)
            ) for _ in range(num_stage_layers)
        ])

        # 4. Competitive Gate: Projects internal state vs Graph Centroid
        # Index 0: Internal Priority, Index 1: Graph/Centroid Priority
        self.gate_predictor = nn.Linear(embed_dim, 2)

        # 5. Output Refinement
        self.refiner_norm = nn.LayerNorm(embed_dim)

        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        centroid_addresses: List of scale addresses [Fast, Mid, Slow]
        """
        gamma = torch.tensor(1.0, device=x.device)
        gate = None

        if graph_matrix is not None:
            # 1. Access Multi-Scale Anchor (Fast Scale)
            if centroid_addresses:
                y_graph = centroid_addresses[0].unsqueeze(0).unsqueeze(0)
            else:
                y_graph = graph_matrix.mean(dim=0).unsqueeze(0).unsqueeze(0)

            # 2. Discrete Stochastic Arbitration (Gumbel-Softmax)
            # Wi acts as the temperature (Tau): Higher plasticity = more stochastic exploration
            gate_logits = self.gate_predictor(x)

            if self.training:
                gate = F.gumbel_softmax(gate_logits, tau=max(0.1, Wi), hard=True)
            else:
                gate = F.softmax(gate_logits, dim=-1)

            # 3. Path Selection
            x_internal = x
            x_anchored = y_graph.expand(x.size(0), x.size(1), -1)

            # Combine based on Gumbel Gate
            x_integrated = (gate[..., 0:1] * x_internal) + (gate[..., 1:2] * x_anchored)

            # 4. Comparative Geometric Divergence (Gamma)
            divergence = F.mse_loss(x_integrated, x_internal)
            gamma = divergence.detach()  # Used by SAGEContainer for confidence calculation

            x = self.norm(x_integrated)

        # 5. TRANSFORMER PROCESSING (Sliding Window Managed by Container)
        for layer in self.layers:
            x = layer(x)

        # 6. OUTPUT REFINEMENT
        x = self.refiner_norm(x)

        # Return governed output and the gate/impact trace
        return self.stage_weight * x, gamma