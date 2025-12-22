import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer

class InfantTransformer(DevelopmentalTransformer):
    """
    Stage 1: Stochastic Grounding & Global Manifold Anchoring.
    - Operator: y = x + (Wi * mean(G)) + ε
    - Milestone: High-plasticity immersion in the global topological field.
    - Feature: 1-layer refinement logic for maximum gradient flow.
    """

    def __init__(self, embed_dim=128, nhead=8):
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # Infant Refinement: 1-layer depth ensures minimal structural bias.
        self.refiner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # Stochastic Exploration Scale (ε)
        # Represents the 'Critical Period' noise necessary for synaptic exploration.
        self.epsilon_scale = 0.1

    def forward(self, x, graph_matrix, Wi=1.0):
        """
        x: [seq_len, batch, embed_dim]
        graph_matrix: [num_nodes, embed_dim]
        """
        # Confidence Divergence Tracking (Initial Baseline)
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. RETRIEVE GLOBAL MANIFOLD CENTROID
            # The infant stage observes the entire forest, not the individual trees.
            # This is the Intrinsic Centroid for the global graph.
            graph_centroid = graph_matrix.mean(dim=0) # [embed_dim]

            # 2. STOCHASTIC PERTURBATION (The ε Operator)
            # This implements the patent claim: y = x + (Wi * Mean_Field) + ε
            # epsilon prevents the model from settling into local minima too early.
            if self.training:
                epsilon = torch.randn_like(graph_centroid) * (self.epsilon_scale * Wi)
                graph_centroid = graph_centroid + epsilon

            # 3. GLOBAL ADDITIVE GROUNDING
            # Simple additive integration of the world-view.
            x_context = x + (Wi * graph_centroid)

            # 4. PASSIVE WITNESS MONITORING
            # Measures the 'Global Magnitude' of the graph's influence.
            # If the graph has been Ablated to Zero, x_context will equal x,
            # indicating a total loss of conceptual grounding.
            gamma_divergence = (x_context - x).pow(2).mean().detach()

            x = self.norm(x_context)

        # 5. FOUNDATIONAL REFINEMENT (1-Layer depth)
        x = self.refiner(x) + x

        # Return output and preliminary divergence (1 - Gamma)
        return self.stage_weight * x, gamma_divergence