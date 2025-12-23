# src/stages/adult_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class AdultTransformer(DevelopmentalTransformer):
    """
    Stage 6: Selective Relational Retrieval.
    Implements y = Softmax((Q_x * K_G^T)/sqrt(d_k)) * V_G.
    Provides the 'Active Reasoning Path' for Stage 7 Governance.
    """

    def __init__(self):
        # 1. Pull Shared Architecture from Config
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # 2. Pull Stage-Specific Hyperparams from Config
        stage_cfg = STAGE_HYPERPARAMS["Adult"]

        # Dynamic Layer Allocation: derived from config boundaries (e.g., 20-24)
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # 3. Transformer Stack (Managed by SAGEContainer's 4-layer sliding window)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.2)
            ) for _ in range(num_stage_layers)
        ])

        # 4. Relational Subspace Projection
        self.graph_proj = nn.Linear(embed_dim, embed_dim)

        # 5. Surgical Cross-Attention: Retrieves specific conceptual instances
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead)

        # 6. Output Refinement
        self.refiner_norm = nn.LayerNorm(embed_dim)

        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, Wi=1.0):
        """
        Returns:
            output: [seq_len, batch, dim]
            impact/trace: The attention map (Active Reasoning Path) for Stage 7.
        """
        attn_map = None

        # 1. CROSS-ATTENTION RETRIEVAL (Patent Logic)
        if graph_matrix is not None:
            # Project graph into relational subspace
            G_projected = self.graph_proj(graph_matrix)

            # Retrieval (Q=input tokens, K/V=Graph Manifold)
            # x is [seq_len, batch, dim], k_v is [num_nodes, batch, dim]
            k_v = G_projected.unsqueeze(1).expand(-1, x.size(1), -1)

            attn_out, attn_map = self.cross_attn(
                query=x,
                key=k_v,
                value=k_v,
                need_weights=True,
                average_attn_weights=True
            )

            # Modulate update by Global Plasticity (Wi)
            x = self.norm(x + (Wi * attn_out))

        # 2. TRANSFORMER PROCESSING (Aperture managed by SAGEContainer)
        for layer in self.layers:
            x = layer(x)

        # 3. OUTPUT REFINEMENT
        x = self.refiner_norm(x)

        # 4. IMPACT CALCULATION (Confidence Signal)
        # Low entropy = Targeted/Confident retrieval
        if attn_map is not None:
            entropy = -torch.sum(attn_map * torch.log(attn_map + 1e-9), dim=-1).mean()
            # Normalize impact score: 0 = high confidence, 1 = chaotic/uncertain
            impact = torch.clamp(entropy / 5.0, 0, 1)
        else:
            impact = torch.tensor(0.5, device=x.device)

        # Return governed output and the attention map for Stage 7 Path Reconstruction
        return self.stage_weight * x, impact