import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer

class SageTransformer(DevelopmentalTransformer):
    """
    Stage 7: Global Topology Governance & Hierarchical Abstraction.
    - Milestone: Meta-Reasoning. Audits the Adult's Reasoning Trace.
    - Operator: y = Attention(x, Path_active ⊙ Mask_governance)
    - Authority: Supreme. Triggers Null-Space Rotation or Ablative Zeroing.
    """

    def __init__(self, embed_dim=128, nhead=8):
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # Governance Head: Projects reasoning paths into the trust-space
        self.governance_gate = nn.Linear(embed_dim, embed_dim)

        # Schema Induction: Recursive Meta-Policy (The 'Optimizer of Optimizers')
        self.schema_induction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, graph_matrix, adult_attn_map=None, Wi=1.0):
        """
        x: [seq_len, batch, embed_dim]
        graph_matrix: [num_nodes, embed_dim]
        adult_attn_map: [batch, seq_len, num_nodes] (The 'Active Reasoning Path')
        """
        # Initialize Confidence Metric (Gamma)
        # 1.0 = Perfect Alignment, 0.0 = Absolute Breach
        gamma = torch.tensor(1.0, device=x.device)

        if adult_attn_map is not None:
            # 1. RECONSTRUCT ACTIVE PATH (P_a)
            # Projects the Adult's attention weights back into weight-space
            path_active = torch.matmul(adult_attn_map, graph_matrix).transpose(0, 1)

            # 2. GENERATE GOVERNANCE MASK
            # The Sage's internal belief of what the path 'should' look like
            # governance_mask serves as the Path_anchor (P_z)
            path_anchor = torch.sigmoid(self.governance_gate(path_active))

            # 3. FORMALIZE CONFIDENCE METRIC (Gamma)
            # Gamma = 1 - ||P_active - P_anchor||
            # This is the 'Divergence' metric mentioned in the patent claims
            divergence = F.mse_loss(path_active, path_active * path_anchor)
            gamma = 1.0 - torch.clamp(divergence, 0, 1)

            # 4. RECURSIVE META-ATTENTION
            # Filters the active path through the Sage's governance lens
            # Suppresses nodes that lie outside the sanctioned manifold
            governed_context = path_active * path_anchor
            x = x + (Wi * governed_context)

        # 5. SCHEMA INDUCTION (Hierarchical Abstraction Level 7)
        # The output is passed back to the Container as the 'Trace'
        out = self.schema_induction(x)

        # We return 1.0 - gamma (divergence) to satisfy the 'impact' argument in the Container
        # so that high divergence = low confidence.
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