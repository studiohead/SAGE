import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer


class TeenTransformer(DevelopmentalTransformer):
    """
    Stage 5: Competitive Abstraction & Stochastic Routing.
    - Operator: y = Gumbel_Softmax(x_internal, x_graph)
    - Milestone: First instance of Active Arbitration.
    - Feature: 5-layer recursive refinement for independent reasoning depth.
    """

    def __init__(self, embed_dim=128, nhead=8):
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # Competitive Gate: Learns to switch based on EWMA Centroid proximity
        self.gate_predictor = nn.Linear(embed_dim, 2)

        # Deep Refinement: Preserving the 5-layer reasoning depth
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # Increased width for Stage 5
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x, graph_matrix, Wi=1.0):
        """
        x: [seq_len, batch, embed_dim]
        graph_matrix: [num_nodes, embed_dim]
        """
        # Initial confidence state (gamma = 1.0)
        gamma = torch.tensor(1.0, device=x.device)

        if graph_matrix is not None:
            # 1. RETRIEVE EWMA-ANCHORED CONTEXT
            # Teen stage uses the Mean Field (Geometric Center) of the graph manifold
            # y_graph acts as the 'Historical Anchor' for this stage
            y_graph = graph_matrix.mean(dim=0)

            # 2. COMPETITIVE ROUTING (Gumbel-Softmax)
            # Predict logits for [0: Internal, 1: Graph-Anchored]
            # gate_logits shape: [seq_len, batch, 2]
            gate_logits = self.gate_predictor(x)

            # Wi (Plasticity) acts as temperature for the routing decision
            if self.training:
                gate = F.gumbel_softmax(gate_logits, tau=max(0.1, Wi), hard=True)
            else:
                gate = F.softmax(gate_logits, dim=-1)

            # 3. COMPUTE ROUTED MANIFOLD
            x_internal = x
            x_anchored = y_graph  # The Graph-Anchor path

            x_integrated = (gate[..., 0:1] * x_internal) + \
                           (gate[..., 1:2] * x_anchored)

            # 4. PATH DIVERGENCE (Preliminary Gamma)
            # We measure the L2 distance between the internal path and the
            # integrated path to detect "Reasoning Drift."
            divergence = (x_integrated - x_internal).pow(2).mean()
            gamma = 1.0 - torch.clamp(divergence, 0, 1)

            x = self.norm(x_integrated)

        # 5. RECURSIVE INDEPENDENT REASONING (5-Layer depth)
        # This is where the Teen stage 'mulls over' the choice
        for _ in range(5):
            x = self.refiner(x) + x

        # Return the output and the divergence (1 - gamma)
        # High divergence here alerts Stage 7 to perform a Null-Space Audit
        return self.stage_weight * x, 1.0 - gamma