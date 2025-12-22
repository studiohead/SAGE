import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer

class ToddlerTransformer(DevelopmentalTransformer):
    """
    Stage 2: Relational Orientation & Contextual Filtering.
    - Operator: y = x + (Wi * (x ⊙ Sigmoid(mean(G) + b)))
    - Milestone: Elementary Gating based on Topological Resonance.
    - Feature: 2-layer refinement for basic relational stability.
    """

    def __init__(self, embed_dim=128, nhead=8):
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # Relational Bias: Adjusts the sensitivity of the Hadamard filter.
        # This allows the model to develop 'preferences' for specific manifolds.
        self.relational_bias = nn.Parameter(torch.zeros(embed_dim))

        # Toddler Refinement: 2-layer depth for early feature stabilization.
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, graph_matrix, Wi=1.0):
        """
        x: [seq_len, batch, embed_dim]
        graph_matrix: [num_nodes, embed_dim]
        """
        # Confidence Divergence Tracking
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. RETRIEVE GLOBAL MEAN FIELD
            # Toddler stage sees the graph as a single, uniform 'feeling' or 'field'.
            graph_centroid = graph_matrix.mean(dim=0) # [embed_dim]

            # 2. HADAMARD GATING (The Stage 2 Operator)
            # This implements the patent claim: y = x + (Wi ⊙ mean(G))
            # The sigmoid turns the graph centroid into a 'Binary-Like' filter.
            # Dimensions that are high in the graph 'open' the gate for input x.
            context_filter = torch.sigmoid(graph_centroid + self.relational_bias)

            # Element-wise modulation (Topological Resonance)
            x_gated = x * context_filter
            x_context = x + (Wi * x_gated)

            # 3. PASSIVE WITNESS MONITORING
            # Measures how much the input 'vibrates' with the graph manifold.
            # If the filter is zero (Ablated), current_impact will be 0.0.
            gamma_divergence = (x_context - x).pow(2).mean().detach()

            x = self.norm(x_context)

        # 4. ELEMENTARY REFINEMENT (2-Layer depth)
        for _ in range(2):
            x = self.refiner(x) + x

        # Return output and preliminary divergence
        return self.stage_weight * x, gamma_divergence