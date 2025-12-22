import torch
import torch.nn as nn
import torch.nn.functional as F


class StageFusion(nn.Module):
    """
    Implements Hierarchical Abstraction Fusion.
    Uses Intrinsic Centroid Anchoring to weight stage consensus.
    """

    def __init__(self, num_stages, hidden_dim):
        super().__init__()
        self.centroid_comparator = nn.Linear(hidden_dim, num_stages)
        self.refiner = nn.LayerNorm(hidden_dim)

    def forward(self, x_input, stage_outputs):
        # Calculate current EWMA-aligned centroid
        # x_input shape: [seq_len, batch, hidden_dim]
        current_centroid = x_input.mean(dim=0).mean(dim=0)

        # Determine weightings based on geometric alignment
        weights = torch.softmax(self.centroid_comparator(current_centroid), dim=-1)

        # Slice to original sequence length (handles TCR memory expansion)
        target_len = x_input.size(0)
        processed_outputs = [out[:target_len] for out in stage_outputs]

        stacked = torch.stack(processed_outputs, dim=-1)
        weights = weights.view(1, 1, 1, -1)

        # Weighted integration across the 7 levels of abstraction
        weighted_sum = (stacked * weights).sum(dim=-1)
        return self.refiner(weighted_sum)


class SAGEContainer(nn.Module):
    def __init__(self, graph, stage_models, thresholds, auditor=None, governance_mode="SAGE_DELEGATED"):
        super().__init__()
        self.graph = graph
        self.stage_models = nn.ModuleDict(stage_models)
        self.thresholds = thresholds
        self.auditor = auditor
        if self.auditor:
            self.auditor.mode = governance_mode

        self.current_stage = 0
        self.eta = 1.0  # Global Plasticity Factor

        sample_transformer = list(stage_models.values())[0]
        self.hidden_dim = getattr(sample_transformer, 'embed_dim', 128)
        self.fusion = StageFusion(len(stage_models), self.hidden_dim)

    def forward(self, x, graph_matrix=None):
        stage_outputs = []
        min_confidence = 1.0
        breach_category = "NULL"

        if graph_matrix is None:
            graph_matrix = self.graph.get_graph_embedding_matrix()

        # Iterate through 7 Hierarchical Abstraction Levels
        for i, (name, model) in enumerate(self.stage_models.items()):
            if i > self.current_stage:
                break

            # 1. O(1) MANIFOLD RETRIEVAL (TCR)
            if i >= 5:  # Adult/Sage stages inject non-linear context
                memory_block = self.graph.retrieve_manifold_context(x.mean(0))
                if memory_block.size(0) > 1:  # Avoid empty pad
                    mem_expanded = memory_block.unsqueeze(1).expand(-1, x.size(1), -1)
                    x = torch.cat([x, mem_expanded], dim=0)

            # 2. PROCESS ABSTRACTION LEVEL
            out, impact = model(x, graph_matrix, Wi=self.eta)
            stage_outputs.append(out)

            # 3. FORMALIZE CONFIDENCE METRIC (Gamma)
            # Gamma = 1 - ||Path_active - Path_anchor||
            # We treat 'impact' as the divergence score
            current_gamma = 1.0 - torch.clamp(impact, 0, 1).item()

            if current_gamma < min_confidence:
                min_confidence = current_gamma
                # Structural (Logic) vs Factual (Manifold) categorization
                breach_category = "STRUCTURAL_ERROR" if i < 4 else "FACTUAL_DISPUTE"

            # 4. EARLY EXIT (99% Certainty Gate)
            if current_gamma > 0.99 and i > 1:
                break

        # 5. HIERARCHICAL FUSION
        fused_output = self.fusion(x, stage_outputs) if len(stage_outputs) > 1 else stage_outputs[0]

        # 6. NULL-SPACE REMEDIATION DISPATCH
        # Threshold: 0.5 Confidence Divergence
        if min_confidence < 0.5 and self.auditor:
            self.dispatch_remediation_request(
                x_input=x,
                trace=fused_output,
                confidence=min_confidence,
                category=breach_category
            )

        return fused_output, {"confidence": min_confidence, "category": breach_category}

    def dispatch_remediation_request(self, x_input, trace, confidence, category):
        """
        Executes Null-Space Rotation or Ablative Zeroing via the Auditor.
        """
        # Calculate EWMA-aligned address for the breach
        breach_address = x_input.mean(dim=0).mean(dim=0).detach()

        # Dispatch with formalized Confidence Metric
        self.auditor.report_candidate_breach({
            "confidence": confidence,
            "divergence": 1.0 - confidence,
            "category": category,
            "reasoning_trace": trace.detach().clone(),
            "centroid_address": breach_address
        })

    def evaluate_gate(self, metrics):
        """Standard accuracy-based stage progression"""
        current_stage_name = list(self.stage_models.keys())[self.current_stage]
        targets = self.thresholds.get(current_stage_name, {})
        if all(metrics.get(k, 0) >= v for k, v in targets.items()):
            if self.current_stage < len(self.stage_models) - 1:
                self.current_stage += 1
                return True
        return False

    def update_parameters(self, epoch):
        self.eta = max(0.01, 1.0 * (0.9 ** (self.current_stage + epoch)))

    def shutdown(self):
        if self.auditor: self.auditor.shutdown()