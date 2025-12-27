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
        [PATENT REF 0014]: Recursive Meta-Governance
        adult_attn_map: [batch, seq_len, nodes] - The 'Reasoning Path'
        """
        # 1. Path Reconstruction (Preserved)
        path_active = torch.matmul(adult_attn_map, graph_matrix).transpose(0, 1)

        # 2. Multi-Scale Context Centroid (Patent Step 2)
        # We use the 'Long-term' (Slow) centroid as the Anchor
        if centroid_addresses and len(centroid_addresses) >= 2:
            # Anchor represents the 'Sanctioned Conceptual Manifold' [0014]
            anchor_vec = centroid_addresses[1].mean(0)  # Long-term EWMA
            context_anchor = anchor_vec.view(1, 1, -1).expand(path_active.size(0), -1, -1)
        else:
            context_anchor = path_active

        # 3. Reconciliation (The Recursive Meta-Governance Layer)
        # reconciled_path represents the 'Path_anchor' from the patent
        reconciled_path, reconciliation_weights = self.path_reconciler(
            query=path_active,
            key=context_anchor,
            value=context_anchor
        )

        # 4. Comparative Geometric Divergence Metric (Γ) [0014]
        with torch.no_grad():
            # We calculate Gamma as the distance between
            # where the Adult went vs where the Anchor stays.
            cos_sim = F.cosine_similarity(path_active, reconciled_path, dim=-1).mean()
            gamma = torch.clamp(cos_sim, 0, 1)

            # [CRITICAL UPDATE]: Identify the 'Trust-Breaching' Nodes
            # We look for nodes in the adult_attn_map that deviate most from the anchor
            breach_signal = (1.0 - cos_sim)

        # 5. Integration (Preserved)
        governed_signal = self.governance_gate(reconciled_path)
        x = x + (Wi * governed_signal)

        # 6. Backbone (Preserved)
        for layer in self.layers:
            x = layer(x)

        # Return the trace and (1-Gamma) as the Remediation Confidence Signal [0014]
        # We also pass the 'breach_signal' for the SageAuditor
        telemetry = {
            "gamma": 1.0 - gamma.detach(),
            "breach_nodes": adult_attn_map if gamma < 0.7 else None  # Threshold trigger
        }

        return self.stage_weight * x, telemetry