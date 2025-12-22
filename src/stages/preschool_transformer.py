import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer


class PreschoolTransformer(DevelopmentalTransformer):
    """
    Stage 3: Global Saliency & Static Relevance.
    - Operator: y = x + Wi * sum(Softmax(G) * G)
    - Milestone: Identification of high-centrality, 'load-bearing' nodes.
    - Feature: 3-layer structural refinement for preliminary concept weighting.
    """

    def __init__(self, embed_dim=128, nhead=8):
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # Saliency Head: Learns to identify globally relevant nodes.
        # This acts as a topological importance filter.
        self.saliency_head = nn.Linear(embed_dim, 1)

        # Preschool Refinement: 3-layer depth for concept stability
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
        # Confidence Divergence Tracking
        gamma_divergence = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None:
            # 1. COMPUTE GLOBAL SALIENCY (The Stage 3 Operator)
            # This identifies 'Anchor Points' in the manifold.
            # Unlike Stage 6, it is context-agnostic (Self-Saliency).
            # relevance_logits shape: [num_nodes, 1]
            relevance_logits = self.saliency_head(graph_matrix)

            # 2. SOFTMAX-WEIGHTED AGGREGATION
            # Creates a 'weighted essence' of the graph.
            # Nodes with 0.0 alignment (Scar Tissue) will yield negligible logits.
            saliency_weights = F.softmax(relevance_logits, dim=0)

            # Weighted average across all known conceptual material
            # focused_essence shape: [embed_dim]
            focused_essence = torch.sum(graph_matrix * saliency_weights, dim=0)

            # 3. KNOWLEDGE INTEGRATION
            # Anchors the reasoning trace to globally significant concepts.
            x_context = x + (Wi * focused_essence)

            # 4. PASSIVE WITNESS MONITORING
            # Measures the 'Gravitational Pull' of high-saliency nodes.
            # If the focused essence is zero (due to widespread Ablation),
            # divergence drops, signaling a lack of structural support.
            gamma_divergence = (x_context - x).pow(2).mean().detach()

            x = self.norm(x_context)

        # 5. CONCEPTUAL REFINEMENT (3-Layer depth)
        for _ in range(3):
            x = self.refiner(x) + x

        # Return output and preliminary divergence (1 - Gamma)
        return self.stage_weight * x, gamma_divergence