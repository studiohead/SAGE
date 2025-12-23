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
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Adult"]

        # Relational Subspace Projection
        self.graph_proj = nn.Linear(embed_dim, embed_dim)

        # Surgical Cross-Attention: Retrieves specific conceptual instances
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead)

        # Deep Reasoning Pass: Refinement for relational depth
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

    def forward(self, x, graph_matrix, Wi=1.0):
        """
        Returns:
            output: [seq_len, batch, dim]
            impact/trace: The attention map (Active Reasoning Path) for Stage 7.
        """
        self._set_trainable_layers()
        attn_map = None

        if graph_matrix is not None:
            # 1. Project graph into relational subspace
            # [num_nodes, dim]
            G_projected = self.graph_proj(graph_matrix)

            # 2. Cross-Attention Retrieval (Q=input tokens, K/V=Graph Manifold)
            # Need weights for Stage 7 Meta-Governance
            # x is [seq_len, batch, dim], k_v is [num_nodes, batch, dim]
            k_v = G_projected.unsqueeze(1).expand(-1, x.size(1), -1)

            attn_out, attn_map = self.cross_attn(
                query=x,
                key=k_v,
                value=k_v,
                need_weights=True,
                average_attn_weights=True  # Averaged across heads for the path
            )

            # Modulate update by Global Plasticity (Wi)
            x = self.norm(x + (Wi * attn_out))

        # 3. Deep Reasoning Refinement
        # Standard 6-pass residual block
        for _ in range(6):
            x = self.refiner(x) + x

        # 4. Impact Calculation (Confidence Signal)
        # Calculate entropy of retrieval: Low entropy = Targeted/Confident retrieval
        # If attn_map is [batch, seq_len, num_nodes]
        if attn_map is not None:
            entropy = -torch.sum(attn_map * torch.log(attn_map + 1e-9), dim=-1).mean()
            # Normalize impact score for the Container
            impact = torch.clamp(entropy / 5.0, 0, 1)
        else:
            impact = torch.tensor(0.5, device=x.device)

        # IMPORTANT: Returning the attn_map as the 'impact/trace' 
        # so SAGEContainer can pass it to Stage 7 for Path Reconstruction.
        return self.stage_weight * x, attn_map