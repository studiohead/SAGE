import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class PreschoolTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Preschool"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.12)
            ) for _ in range(num_stage_layers)
        ])

        self.saliency_proj = nn.Linear(embed_dim, embed_dim)
        self.refiner = nn.LayerNorm(embed_dim)
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq, batch, embed_dim]
        graph_matrix: Likely arriving as [8192] or [batch, 8192]
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. FORCE REALIGNMENT
            # If we are getting a flattened 8192, we must reconstruct the node-space
            # 8192 / 512 = 16 nodes.
            working_graph = graph_matrix.view(-1, self.embed_dim)  # [16, 512]

            # 2. ANCHOR ALIGNMENT
            if centroid_addresses and len(centroid_addresses) > 2:
                anchor_slow = centroid_addresses[2]
                if anchor_slow.numel() != self.embed_dim:
                    anchor_slow = anchor_slow.view(-1, self.embed_dim).mean(dim=0)
            else:
                anchor_slow = working_graph.mean(dim=0)

            # 3. SALIENCY CALCULATION
            # Project nodes: [64, 128] -> [16, 512]
            G_projected = self.saliency_proj(working_graph)

            # anchor_slow: [512] -> [512, 1) for matmul
            # logits: [16, 1]
            relevance_logits = torch.matmul(G_projected, anchor_slow.view(self.embed_dim, 1))
            saliency_weights = F.softmax(relevance_logits / (self.embed_dim ** 0.5), dim=0)

            # 4. KNOWLEDGE INTEGRATION
            # G_projected: [16, 512], weights: [16, 1]
            # essence: [512]
            focused_essence = torch.sum(G_projected * saliency_weights, dim=0)

            # 5. BROADCAST PREP
            # Explicitly force essence to [1, 1, 512]
            essence_context = focused_essence.reshape(1, 1, self.embed_dim)

            # 6. ADDITION
            # a (512) + b (512)
            x_context = x + (Wi * essence_context)

            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        for layer in self.layers:
            x = layer(x)

        x = self.refiner(x)
        return self.stage_weight * x, gamma_divergence