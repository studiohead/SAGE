import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer


class AdultTransformer(DevelopmentalTransformer):
    """
    Stage 6: Selective Relational Retrieval.
    - Milestone: Full Cross-Attention over the Shared Concept Graph.
    - Operator: y = MultiheadAttention(Query=x, Key/Value=Graph)
    - Responsibility: Generates the Reasoning Trace for the Sage Auditor.
    - Authority: Secondary to Sage; focuses on relational accuracy.
    """

    def __init__(self, embed_dim=128, nhead=8):
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # Surgical Cross-Attention: Retrieves specific conceptual instances
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead)

        # Deep Reasoning Pass (6 layers for high relational depth)
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, graph_matrix, Wi=1.0, return_attn=True):
        """
        x: [seq_len, batch, embed_dim]
        graph_matrix: [num_nodes, embed_dim]
        """
        attn_map = None

        if graph_matrix is not None:
            # 1. MANIFOLD ALIGNMENT
            # Rotate the graph into the Adult's relational subspace
            # This is Stage 4 logic utilized within the Stage 6 operator
            G_projected = self.graph_projection(graph_matrix)

            # 2. CROSS-ATTENTION (The Adult Operator)
            # Query (Q) = Input Thought (x)
            # Key/Value (K,V) = Projected Graph [num_nodes, batch, embed_dim]
            k_v = G_projected.unsqueeze(1).expand(-1, x.size(1), -1)

            # attn_map shape: [batch, seq_len, num_nodes]
            # This map is the critical 'Evidence' for Find-and-Incinerate
            attn_out, attn_map = self.cross_attn(
                query=x,
                key=k_v,
                value=k_v,
                need_weights=True,
                average_attn_weights=False
            )

            # Integrate Graph Context scaled by Temperature (Wi)
            x = self.norm(x + (Wi * attn_out))

        # 3. DEEP REFINEMENT PASS
        # Iterative logic pass to stabilize the retrieved relational data
        for _ in range(6):
            x = self.refiner(x) + x

        # Return the output and the Reasoning Trace for the Sage/Auditor
        if return_attn:
            return self.stage_weight * x, attn_map

        return self.stage_weight * x