# src/stages/toddler_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class ToddlerTransformer(DevelopmentalTransformer):
    """
    Stage 2: Relational Orientation & Contextual Filtering.
    Logic: y = x + (W ⊙ mean(G))
    Uses Element-wise Hadamard gating to filter inputs based on Graph Centroids.
    """

    def __init__(self):
        # 1. Pull Shared Architecture from Config
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # 2. Pull Stage-Specific Hyperparams from Config
        stage_cfg = STAGE_HYPERPARAMS["Toddler"]

        # Calculate how many layers this stage owns.
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        # 3. Dynamic Layer Allocation
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.1)
            ) for _ in range(num_stage_layers)
        ])

        # 4. Relational Bias (W in the patent logic - Hadamard Gate)
        self.relational_weight = nn.Parameter(torch.zeros(embed_dim))

        # 5. Output Refinement
        self.refiner = nn.LayerNorm(embed_dim)

        self.epsilon_scale = stage_cfg["epsilon_scale"]
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [seq_len, batch, dim] (Standard SAGE 3D)
        graph_matrix: [nodes, batch, dim]
        """
        gamma_divergence = torch.tensor(0.0, device=x.device)

        # 1. CONTEXTUAL FILTERING (Patent Logic)
        if graph_matrix is not None:
            # Robust Centroid Extraction: Fallback if addresses are missing
            if centroid_addresses and len(centroid_addresses) > 1:
                graph_centroid = centroid_addresses[1]
            else:
                # Collapse graph_matrix to a [Batch, Dim] or [Dim] representation
                graph_centroid = graph_matrix
                # Ensure we squeeze out extra dimensions (e.g., from Node-Level stacking)
                while graph_centroid.dim() > 2:
                    graph_centroid = graph_centroid.mean(dim=0)

                # If it's [Nodes, Dim], mean it down to [Dim] to allow broadcasting
                if graph_centroid.dim() == 2 and graph_centroid.size(0) != x.size(1):
                    graph_centroid = graph_centroid.mean(dim=0)

            # Stochastic Grounding (Epsilon Noise)
            if self.training:
                epsilon = torch.randn_like(graph_centroid) * (self.epsilon_scale * Wi)
                graph_centroid = graph_centroid + epsilon

            # Element-wise Hadamard Gating
            context_filter = torch.sigmoid(graph_centroid * self.relational_weight)

            # DIMENSION GUARD: Prepare for [Seq, Batch, Dim]
            filter_view = context_filter
            if filter_view.dim() == 1:  # [Dim] -> [1, 1, Dim]
                filter_view = filter_view.view(1, 1, -1)
            elif filter_view.dim() == 2:  # [Batch, Dim] -> [1, Batch, Dim]
                filter_view = filter_view.unsqueeze(0)

            # Hadamard Integration: This is the core patent math
            x_context = x + (Wi * (x * filter_view))

            # Comparative Geometric Divergence (Gamma)
            # This should now be > 0.0, which makes Gamma < 1.0
            gamma_divergence = F.mse_loss(x_context, x).detach()
            x = self.norm(x_context)

        # 2. TRANSFORMER PROCESSING
        for layer in self.layers:
            x = layer(x)

        # 3. OUTPUT REFINEMENT
        x = self.refiner(x)

        return self.stage_weight * x, gamma_divergence