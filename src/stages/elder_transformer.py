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
                dropout=stage_cfg.get("dropout", 0.25),
                batch_first=True
            ) for _ in range(num_stage_layers)
        ])

        self.governance_gate = nn.Linear(embed_dim, embed_dim)
        self.path_reconciler = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.refiner_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_matrix, centroid_addresses=None, adult_attn_map=None, Wi=1.0):
        """
        [PATENT REF 0014]: Recursive Meta-Governance

        x:               [batch, seq_len, embed_dim]
        graph_matrix:    [nodes, embed_dim]
        adult_attn_map:  [nodes, nodes]
        """

        batch, seq_len, embed_dim = x.shape

        # ------------------------------------------------------------------
        # Elder Bootstrap (Option 1)
        # ------------------------------------------------------------------
        if adult_attn_map is None:
            num_nodes = graph_matrix.size(0)
            adult_attn_map = torch.eye(
                num_nodes,
                device=graph_matrix.device
            )

        # ------------------------------------------------------------------
        # 1. Path Reconstruction (Node → Embedding)
        # ------------------------------------------------------------------
        # [nodes, nodes] @ [nodes, embed_dim] → [nodes, embed_dim]
        path_active = adult_attn_map @ graph_matrix

        # Treat nodes as sequence, batch=1
        path_active = path_active.unsqueeze(0)  # [1, nodes, embed_dim]

        # ------------------------------------------------------------------
        # 2. Context Anchor
        # ------------------------------------------------------------------
        if centroid_addresses and len(centroid_addresses) >= 2:
            anchor = centroid_addresses[1].mean(0)  # [embed_dim]
            context_anchor = anchor.view(1, 1, -1).expand(
                1, path_active.size(1), embed_dim
            )
        else:
            context_anchor = path_active

        # ------------------------------------------------------------------
        # 3. Recursive Meta-Governance
        # ------------------------------------------------------------------
        reconciled_path, _ = self.path_reconciler(
            query=path_active,
            key=context_anchor,
            value=context_anchor
        )

        reconciled_path = self.refiner_norm(reconciled_path)

        # ------------------------------------------------------------------
        # 4. Divergence Metric Γ
        # ------------------------------------------------------------------
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                path_active.squeeze(0),
                reconciled_path.squeeze(0),
                dim=-1
            ).mean()
            gamma = torch.clamp(cos_sim, 0.0, 1.0)

        # ------------------------------------------------------------------
        # 5. Governance Signal (GLOBAL → SEQUENCE)
        # ------------------------------------------------------------------
        # Collapse node dimension → single governing vector
        governor = reconciled_path.mean(dim=1)          # [1, embed_dim]
        governor = self.governance_gate(governor)        # [1, embed_dim]

        # Broadcast across sequence
        governor = governor.unsqueeze(1).expand(batch, seq_len, embed_dim)

        x = x + (Wi * governor)

        # ------------------------------------------------------------------
        # 6. Backbone
        # ------------------------------------------------------------------
        for layer in self.layers:
            x = layer(x)

        telemetry = {
            "gamma": 1.0 - gamma.detach(),
            "breach_nodes": adult_attn_map if gamma < 0.7 else None
        }

        return self.stage_weight * x, telemetry
