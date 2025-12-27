##############################################################################
# Elder | Functional Recursive Meta-Governance
# Purpose: Reconciles Active Reasoning (Adult) with Global Stability (Centroids).
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class ElderTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS.get("Elder", STAGE_HYPERPARAMS.get("Adult"))
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.25)
            ) for _ in range(num_stage_layers)
        ])

        # The 'Judge': Maps reconciled paths back to the residual stream
        self.governance_gate = nn.Linear(embed_dim, embed_dim)

        # Cross-Attention between 'What I'm thinking' (Adult Path)
        # and 'What is True' (Centroid Anchors)
        self.path_reconciler = nn.MultiheadAttention(embed_dim, nhead)

        self.refiner_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_matrix, centroid_addresses=None, adult_attn_map=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        adult_attn_map: [batch, seq_len, nodes] - Passed from Adult Stage
        """
        gamma = torch.tensor(1.0, device=x.device)  # Default to 'High Stability'

        if graph_matrix is not None and adult_attn_map is not None:
            # 1. RECONSTRUCT THE 'ADULT' REASONING PATH
            # This is what the Adult stage 'saw' in the graph.
            # path_active: [seq_len, batch, dim]
            path_active = torch.matmul(adult_attn_map, graph_matrix).transpose(0, 1)

            # 2. ALIGN GLOBAL ANCHORS
            if centroid_addresses and len(centroid_addresses) > 2:
                # We prioritize the Slow-Scale (Index 2) as the 'Immutable Truth'
                anchor_vec = centroid_addresses[2].reshape(-1, self.embed_dim).mean(0)
                context_anchor = anchor_vec.view(1, 1, -1)  # Broadcasts to [S, B, D]
            else:
                context_anchor = path_active  # Self-reconciliation if centroids missing

            # 3. RECONCILIATION (The Governance Check)
            # We see how well the 'Active Path' aligns with the 'Global Anchor'
            reconciled_path, _ = self.path_reconciler(
                query=path_active,
                key=context_anchor,
                value=context_anchor
            )

            # 4. COMPUTE TOPOLOGICAL DIVERGENCE (Gamma)
            # If the Elder has to change the Adult's mind a lot, Gamma drops.
            cos_sim = F.cosine_similarity(path_active, reconciled_path, dim=-1).mean()
            gamma = torch.clamp(cos_sim, 0, 1)

            # 5. INTEGRATE GOVERNED CONTEXT
            governed_signal = self.governance_gate(reconciled_path)
            x = x + (Wi * governed_signal)

        # 6. TRANSFORMER BACKBONE
        for layer in self.layers:
            x = layer(x)

        x = self.refiner_norm(x)

        # We return 1.0 - gamma so that 'High Divergence' (Disagreement)
        # is visible in the logs.
        return self.stage_weight * x, 1.0 - gamma.detach()