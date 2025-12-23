# src/stages/sage_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class SageTransformer(DevelopmentalTransformer):
    """
    Stage 7: Highest-Tier Recursive Meta-Governance.
    Implements Paragraph [0014] of the SAGE Patent: 
    Compares Active Reasoning Paths against Multi-Scale Centroid Anchors.
    """

    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        # Sage/Elder stage config
        stage_cfg = STAGE_HYPERPARAMS.get("Sage", STAGE_HYPERPARAMS.get("Elder"))

        # Governance Head: Projects reasoning into the "Anchor Space"
        self.governance_gate = nn.Linear(embed_dim, embed_dim)

        # Meta-Governance Projection: Reconciles Active Path vs Centroids
        self.path_reconciler = nn.MultiheadAttention(embed_dim, nhead)

        # Schema Induction: Recursive Meta-Policy (Hierarchical Abstraction)
        self.schema_induction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.plasticity_scale = stage_cfg["plasticity_scale"]
        self.training_layers = stage_cfg["training_layers"]
        self.trainable_layer_range = (0, self.training_layers)

    def forward(self, x, graph_matrix, centroid_addresses=None, adult_attn_map=None, Wi=1.0):
        """
        x: [seq_len, batch, dim]
        graph_matrix: [num_nodes, dim]
        centroid_addresses: List of [dim] tensors (Fast, Mid, Slow)
        """
        self._set_trainable_layers()
        gamma = torch.tensor(1.0, device=x.device)

        # 1. RECONSTRUCT ACTIVE REASONING PATH (Per Patent [0014])
        # Projects current attention back into the conceptual manifold
        if adult_attn_map is not None:
            # Reconstruct trajectory: [seq_len, dim]
            path_active = torch.matmul(adult_attn_map, graph_matrix).transpose(0, 1)

            # 2. COMPUTE MULTI-SCALE CONTEXT ANCHOR
            # Aggregate multiple temporal resolutions (Fast, Mid, Slow)
            if centroid_addresses:
                # Weighted average of scales (Slow scale usually carries more weight for Stage 7)
                # alpha_i values per Patent [0014]
                weights = torch.tensor([0.2, 0.3, 0.5], device=x.device).view(-1, 1)
                anchors_stacked = torch.stack(centroid_addresses)  # [3, dim]
                context_anchor = (anchors_stacked * weights).sum(dim=0).unsqueeze(0).unsqueeze(0)
                # Expand context_anchor to match sequence length for comparison
                context_anchor = context_anchor.expand(path_active.size(0), -1, -1)
            else:
                context_anchor = path_active

            # 3. RECURSIVE META-GOVERNANCE
            # Use Multi-head attention to reconcile current path against the multi-scale anchor
            path_anchor, _ = self.path_reconciler(path_active, context_anchor, context_anchor)

            # 4. COMPUTE COMPARATIVE GEOMETRIC DIVERGENCE (Γ)
            # Gamma is the inverse of the distance between active path and sanctioned anchor
            divergence = F.cosine_similarity(path_active, path_anchor, dim=-1).mean()
            gamma = torch.clamp(divergence, 0, 1)

            # 5. GOVERNANCE TRANSFORMATION
            # Modulate input x based on the reconciled path
            governed_context = self.governance_gate(path_anchor)
            x = x + (Wi * governed_context)

        # 6. SCHEMA INDUCTION
        out = self.schema_induction(x)

        # Return governed trace and remediation confidence (1 - Γ)
        # 1 - Gamma tells the Auditor how much 'hallucination' or 'drift' was detected
        return self.stage_weight * out, 1.0 - gamma.item()

    def _set_trainable_layers(self):
        start, num_layers = self.trainable_layer_range
        for i, layer in enumerate(self.children()):
            requires_grad = start <= i < (start + num_layers)
            for param in layer.parameters():
                param.requires_grad = requires_grad