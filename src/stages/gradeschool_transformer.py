import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer

class GradeschoolTransformer(DevelopmentalTransformer):
    """
    Stage 4: Structural Subspacing & Categorical Abstraction.
    - Operator: y = x + Wi * (G * W_proj)
    - Milestone: Manifold rotation for categorical isolation.
    - Feature: 4-layer structural refinement for categorical depth.
    """

    def __init__(self, embed_dim=128, nhead=8):
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # Subspace Projection (W_proj):
        # Rotates the EWMA-anchored centroid into a task-specific basis.
        self.subspace_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Structural Refinement: 4-layer depth for concept categorization
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, graph_matrix, Wi=1.0):
        """
        x: [seq_len, batch, embed_dim]
        graph_matrix: [num_nodes, embed_dim]
        """
        # Confidence Divergence (Impact)
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. RETRIEVE EWMA-ANCHORED CENTROID
            # Grade-school uses the global mean field of the neighborhood
            # anchor_z acts as the Intrinsic Centroid
            anchor_z = graph_matrix.mean(dim=0)

            # 2. SUBSPACE ROTATION (The Stage 4 Operator)
            # This implements the patent claim: y = x + Wi * (G * W_proj)
            # It enables categorical isolation (e.g. separating 'Physics' from 'Ethics')
            rotated_memory = self.subspace_proj(anchor_z)

            # 3. KNOWLEDGE INTEGRATION
            # Integration is additive, reflecting 'Structural Grounding'
            x_integrated = x + (Wi * rotated_memory)

            # 4. PASSIVE WITNESS MONITORING
            # We measure the L2 divergence between the input and the subspace projection.
            # If the projection is zero (Ablated) or orthogonal (Tombstoned),
            # this stage registers a trust anomaly.
            gamma_divergence = (x_integrated - x).pow(2).mean().detach()

            x = self.norm(x_integrated)

        # 5. CATEGORICAL REFINEMENT (4-Layer depth)
        for _ in range(4):
            x = self.refiner(x) + x

        # Return output and 1 - gamma divergence
        return self.stage_weight * x, gamma_divergence