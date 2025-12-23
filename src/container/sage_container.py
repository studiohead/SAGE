import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import STAGE_HYPERPARAMS

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
        # 1. Calculate current EWMA-aligned centroid for fusion weighting
        # x_input shape: [seq, batch, dim]
        current_centroid = x_input.mean(dim=0).mean(dim=0)

        # 2. Get Raw Weights (Initial shape: [num_total_stages])
        raw_weights = self.centroid_comparator(current_centroid)

        # 3. DYNAMIC SLICE: Only take weights for active stages
        num_active = len(stage_outputs)
        active_weights = torch.softmax(raw_weights[:num_active], dim=-1)

        # 4. Slice to original sequence length (handles TCR memory expansion)
        target_len = x_input.size(0)
        processed_outputs = [out[:target_len] for out in stage_outputs]

        # 5. Stack outputs and apply weights
        # weights: [num_active] -> [1, 1, 1, num_active]
        stacked = torch.stack(processed_outputs, dim=-1)
        weights_view = active_weights.view(1, 1, 1, -1)

        # 6. Weighted summation and normalization
        weighted_sum = (stacked * weights_view).sum(dim=-1)
        return self.refiner(weighted_sum)


class SAGEContainer(nn.Module):
    def __init__(self, graph, stage_models, thresholds, auditor=None, governance_mode="SAGE_DELEGATED"):
        super().__init__()
        self.graph = graph
        self.stage_names = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Elder"]
        self.stages = nn.ModuleDict(stage_models)
        self.thresholds = thresholds

        self.auditor = auditor
        if self.auditor:
            self.auditor.mode = governance_mode
            self.auditor.container = self

        self.current_stage_idx = 0
        self.eta = 1.0  # Global Plasticity Factor (Wi)

        # Infer hidden dim from the first available stage model
        sample_transformer = next(iter(stage_models.values()))
        self.hidden_dim = getattr(sample_transformer, 'embed_dim', 128)
        self.fusion = StageFusion(len(stage_models), self.hidden_dim)

    def update_plasticity_window(self):
        """
        SLIDING WINDOW PLASTICITY:
        Freezes the foundation and thaws the active 4-layer training window.
        """
        current_name = self.stage_names[self.current_stage_idx]
        params = STAGE_HYPERPARAMS[current_name]
        start, end = params["layer_start"], params["layer_end"]

        # 1. Global Freeze (Everything locked by default)
        for param in self.parameters():
            param.requires_grad = False

        # 2. Sequential Layer Harvesting
        all_layers = []
        for name in self.stage_names:
            if name in self.stages:
                # Harvesting Transformer blocks from the stage-specific ModuleList
                all_layers.extend(self.stages[name].layers)

        # 3. Surgical Thaw (The Aperture)
        # Only the layers in the config-defined range for the current stage are thawed
        print(f"[*] SAGE Aperture: Activating Layers {start}-{end} for {current_name}")
        for i in range(start, min(end, len(all_layers))):
            layer = all_layers[i]
            for param in layer.parameters():
                param.requires_grad = True
            layer.train()

        self.graph.ensure_stage_initialized(self.current_stage_idx)

    def forward(self, x, graph_matrix=None):
        stage_outputs = []
        min_confidence = 1.0
        breach_category = "NULL"

        if graph_matrix is None:
            graph_matrix = self.graph.get_graph_embedding_matrix()

        seq_len_orig = x.size(0)  # store original sequence length

        # Iterate through stages up to the current developmental maturity
        for i, name in enumerate(self.stage_names):
            if i > self.current_stage_idx:
                break

            model = self.stages[name]

            # 1. TCR (Tombstone/Incineration/Retrieval) Logic for Higher Stages
            x_aug = x
            if i >= 5:  # Adult/Elder levels
                memory_block = self.graph.retrieve_manifold_context(x.mean(0))
                if memory_block.size(0) > 0:
                    mem_expanded = memory_block.unsqueeze(1).expand(-1, x.size(1), -1)
                    x_aug = torch.cat([x, mem_expanded], dim=0)

            # 2. Process Abstraction Level
            out, impact = model(x_aug, graph_matrix, Wi=self.eta)

            # 3. Restore original sequence length before fusion
            if out.size(0) != seq_len_orig:
                out = out[:seq_len_orig]

            stage_outputs.append(out)

            # 4. Confidence Tracking
            current_gamma = 1.0 - torch.clamp(torch.as_tensor(impact), 0, 1).item()
            if current_gamma < min_confidence:
                min_confidence = current_gamma
                breach_category = "STRUCTURAL_ERROR" if i < 4 else "FACTUAL_DISPUTE"

            # 5. Early Exit (Inference Optimization)
            if not self.training and current_gamma > 0.98 and i > 1:
                break

        # 6. Hierarchical Fusion (Consensus of all active stages)
        fused_output = self.fusion(x, stage_outputs) if len(stage_outputs) > 1 else stage_outputs[0]

        # 7. Remediation Triggering (Auditor Integration)
        if min_confidence < 0.5 and self.auditor and not self.training:
            self.dispatch_remediation_request(x, fused_output, min_confidence, breach_category)

        return fused_output, {"confidence": min_confidence, "category": breach_category}

    def dispatch_remediation_request(self, x_input, trace, confidence, category):
        # Extract addresses for the Auditor's multi-scale verification
        addr_fast = x_input.mean(dim=0).mean(dim=0).detach()
        addr_mid = (x_input.mean(0) * 0.7 + trace.mean(0) * 0.3).mean(0).detach()
        addr_slow = trace.mean(dim=0).mean(dim=0).detach()

        self.auditor.report_candidate_breach({
            "confidence": confidence,
            "divergence": 1.0 - confidence,
            "category": category,
            "reasoning_trace": trace.detach().clone(),
            "centroid_addresses": [addr_fast, addr_mid, addr_slow]
        })

    def evaluate_gate(self, metrics):
        """Determines if the model is ready to promote to the next stage."""
        current_name = self.stage_names[self.current_stage_idx]
        targets = self.thresholds.get(current_name, {})

        if all(metrics.get(k, 0) >= v for k, v in targets.items()):
            if self.current_stage_idx < len(self.stage_names) - 1:
                self.current_stage_idx += 1
                self.update_plasticity_window()
                return True
        return False

    def update_parameters(self, epoch):
        """Decay global plasticity (Wi) as training progresses."""
        base_decay = 0.9 ** (self.current_stage_idx + epoch)
        self.eta = max(0.01, self.eta * base_decay)

    def shutdown(self):
        if self.auditor:
            self.auditor.shutdown()