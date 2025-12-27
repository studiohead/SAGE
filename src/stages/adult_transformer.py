##############################################################################
# Adult | y = Softmax((Q_x * K_G^T)/sqrt(d_k)) * V_G
# Purpose: High-Fidelity Multi-Head Retrieval from the Full Manifold.
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class AdultTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Adult"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.2)
            ) for _ in range(num_stage_layers)
        ])

        # Separate projections for Key and Value to allow the model to
        # index the graph differently than it retrieves from it.
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead)
        self.refiner_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        graph_matrix: [nodes, dim]
        """
        impact = torch.tensor(0.5, device=x.device)

        if graph_matrix is not None:
            # 1. PREPARE THE MEMORY POOL
            # We treat every node as a separate 'Key-Value' pair in memory
            nodes = graph_matrix.view(-1, self.embed_dim)

            # Project nodes into K and V spaces
            # K_g shape: [Nodes, Dim]
            K_g = self.k_proj(nodes).unsqueeze(1).expand(-1, x.size(1), -1)
            V_g = self.v_proj(nodes).unsqueeze(1).expand(-1, x.size(1), -1)

            # 2. SURGICAL CROSS-ATTENTION
            # Query: Input Tokens (x)
            # Keys/Values: The Graph Manifold
            attn_out, attn_weights = self.cross_attn(
                query=x,
                key=K_g,
                value=V_g,
                need_weights=True
            )

            # 3. SELECTIVE INTEGRATION
            # Wi (Plasticity) now acts as the 'Recall Strength'
            x_context = x + (Wi * attn_out)

            # 4. TELEMETRY: Entropy-Based Impact
            # We measure how 'focused' the retrieval was.
            # High entropy = searching everywhere; Low entropy = pinpoint recall.
            with torch.no_grad():
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1).mean()
                # Normalize entropy by the log of node count to get impact [0, 1]
                node_count = nodes.size(0)
                impact = torch.clamp(entropy / torch.log(torch.tensor(node_count, dtype=torch.float)), 0, 1)

            x = self.norm(x_context)

        # 5. TRANSFORMER BACKBONE
        for layer in self.layers:
            x = layer(x)

        x = self.refiner_norm(x)
        return self.stage_weight * x, impact