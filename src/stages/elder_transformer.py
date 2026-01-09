##############################################################################
# Elder | Functional Recursive Meta-Governance (Final Maturity)
# Purpose: Reconciles Active Reasoning (Adult) with Global Stability (Centroids).
# Suture: Implements Token-Wise Cross-Querying for grounded communication.
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

        # Ensure Elder uses its specific configuration, fallback to Adult
        stage_cfg = STAGE_HYPERPARAMS.get("Elder", STAGE_HYPERPARAMS.get("Adult"))
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.25),
                batch_first=True
            ) for _ in range(num_stage_layers)
        ])

        # Governance Infrastructure
        self.governance_gate = nn.Linear(embed_dim, embed_dim)
        # Cross-Attention: Allows the text sequence to query the Graph manifold
        self.path_reconciler = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.refiner_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_matrix, centroid_addresses=None, adult_attn_map=None, Wi=1.0):
        """
        [PATENT REF 0014]: Recursive Meta-Governance

        Updated to transform the manifold into a queryable memory space
        rather than a collapsed global vector.
        """
        batch, seq_len, embed_dim = x.shape

        # ------------------------------------------------------------------
        # 0. Elder Bootstrap
        # ------------------------------------------------------------------
        if adult_attn_map is None:
            num_nodes = graph_matrix.size(0)
            adult_attn_map = torch.eye(num_nodes, device=graph_matrix.device)

        # ------------------------------------------------------------------
        # 1. Path Reconstruction (Node → Embedding Alignment)
        # ------------------------------------------------------------------
        # Project the graph nodes through the adult relational map
        path_active = adult_attn_map @ graph_matrix  # [nodes, dim]
        path_active = path_active.unsqueeze(0)  # [1, nodes, dim]

        # ------------------------------------------------------------------
        # 2. Contextual Grounding (Centroid Pivot)
        # ------------------------------------------------------------------
        if centroid_addresses and len(centroid_addresses) >= 2:
            # Pivot against the Mid-Scale Centroid for stability
            anchor = centroid_addresses[1].mean(0)
            context_anchor = anchor.view(1, 1, -1).expand(1, path_active.size(1), embed_dim)
        else:
            context_anchor = path_active

        # ------------------------------------------------------------------
        # 3. Recursive Meta-Governance (Geometric Reconciliation)
        # ------------------------------------------------------------------
        # Harmonize the active path against the stable anchors
        reconciled_path, _ = self.path_reconciler(
            query=path_active,
            key=context_anchor,
            value=context_anchor
        )
        reconciled_path = self.refiner_norm(reconciled_path)  # [1, nodes, dim]

        # ------------------------------------------------------------------
        # 4. Divergence Metric (Γ) - The 'Sanity Check'
        # ------------------------------------------------------------------
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                path_active.squeeze(0),
                reconciled_path.squeeze(0),
                dim=-1
            ).mean()
            # gamma represents the 'Friction' between current logic and graph stability
            gamma = torch.clamp(cos_sim, 0.0, 1.0)
            friction = 1.0 - gamma

        # ------------------------------------------------------------------
        # 5. Token-Wise Cross-Querying (Grounded Communication)
        # ------------------------------------------------------------------
        # Instead of mean-pooling, we let 'x' act as a Query into the manifold
        # reconciled_path: [1, nodes, dim] -> [batch, nodes, dim]
        memory_bank = reconciled_path.expand(batch, -1, -1)

        # High-resolution knowledge extraction
        governor_dynamic, weights = self.path_reconciler(
            query=x,  # What the model is saying/thinking
            key=memory_bank,  # The stable graph nodes
            value=memory_bank
        )

        # Dynamic Gating: Learns which tokens need graph-grounding vs. simple syntax
        gate = torch.sigmoid(self.governance_gate(x))

        # Maturity Scaling: If friction is high, boost the graph influence to prevent drift
        maturity_scale = 1.0 + friction
        x = x + (Wi * gate * governor_dynamic * maturity_scale)

        # ------------------------------------------------------------------
        # 6. Final Backbone Processing
        # ------------------------------------------------------------------
        for layer in self.layers:
            x = layer(x)

        # ------------------------------------------------------------------
        # 7. Telemetry (The 'Thought Trace')
        # ------------------------------------------------------------------
        telemetry = {
            "gamma": friction.detach(),
            "concept_saliency": weights.detach(),  # Heatmap of which nodes 'spoke'
            "breach_nodes": adult_attn_map if friction > 0.3 else None,
            "confidence": gamma.item()
        }

        return self.stage_weight * x, telemetry