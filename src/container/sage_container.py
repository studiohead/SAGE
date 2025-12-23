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
        # Calculate current EWMA-aligned centroid for fusion weighting
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

        # Auditor integration with back-reference for metabolic feedback
        self.auditor = auditor
        if self.auditor:
            self.auditor.mode = governance_mode
            self.auditor.container = self  # Allow auditor to trigger rebound

        self.current_stage = 0
        self.eta = 1.0  # Global Plasticity Factor (Wi)

        sample_transformer = list(stage_models.values())[0]
        self.hidden_dim = getattr(sample_transformer, 'embed_dim', 128)
        self.fusion = StageFusion(len(stage_models), self.hidden_dim)

    def trigger_metabolic_rebound(self, breach_impact):
        """
        HOMEOTRANSIS OPERATOR:
        Temporarily increases plasticity (eta) to allow the model to
        re-anchor its logic around a newly incinerated or tombstoned zone.
        """
        # Proportional boost to avoid total system shock
        rebound_boost = torch.clamp(torch.tensor(breach_impact * 2.0), 0, 0.5).item()

        # Reset eta toward higher plasticity
        self.eta = min(1.0, self.eta + rebound_boost)

        print(f"[SAGE_SYSTEM] Metabolic Rebound Triggered. Impact: {breach_impact:.4f}. New Eta: {self.eta:.4f}")

    def forward(self, x, graph_matrix=None):
        stage_outputs = []
        min_confidence = 1.0
        breach_category = "NULL"

        if graph_matrix is None:
            graph_matrix = self.graph.get_graph_embedding_matrix()

        # Iterate through Hierarchical Abstraction Levels
        for i, (name, model) in enumerate(self.stage_models.items()):
            if i > self.current_stage:
                break

            # 1. O(1) MANIFOLD RETRIEVAL (TCR)
            # Higher stages pull global context to verify local tokens
            if i >= 5:
                memory_block = self.graph.retrieve_manifold_context(x.mean(0))
                if memory_block.size(0) > 1:
                    mem_expanded = memory_block.unsqueeze(1).expand(-1, x.size(1), -1)
                    x = torch.cat([x, mem_expanded], dim=0)

            stage_name = name
            stage_cfg = STAGE_HYPERPARAMS.get(stage_name, {})

            # Dynamic Hyperparameter Injection from config/config.py
            if hasattr(model, "trainable_layer_range"):
                model.trainable_layer_range = (0, stage_cfg.get("training_layers", 1))

            # 2. PROCESS ABSTRACTION LEVEL
            # Pass global eta (Plasticity) and config params to the stage
            out, impact = model(x, graph_matrix, Wi=self.eta)
            stage_outputs.append(out)

            # 3. GAMMA (CONFIDENCE) CALCULATION
            current_gamma = 1.0 - torch.clamp(impact, 0, 1).item()

            if current_gamma < min_confidence:
                min_confidence = current_gamma
                # Mapping: Stages 0-3 = Structural (Filtering/Grounding)
                #          Stages 4-6 = Factual (Estimation/Retrieval)
                breach_category = "STRUCTURAL_ERROR" if i < 4 else "FACTUAL_DISPUTE"

            # 4. EARLY EXIT GATE
            if current_gamma > 0.99 and i > 1:
                break

        # 5. HIERARCHICAL FUSION
        fused_output = self.fusion(x, stage_outputs) if len(stage_outputs) > 1 else stage_outputs[0]

        # 6. ASYNCHRONOUS REMEDIATION DISPATCH (MULTI-SCALE)
        # Threshold: < 0.5 confidence indicates a high-entropy manifold breach
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
        Calculates Multi-Scale Context Centroids (Patent Paragraph [0014])
        and sends them to the Auditor for surgical isolation.
        """
        # Calculate a spectrum of addresses for the breach area
        # This represents the "Conceptual Zip Code" at different temporal resolutions
        # Address 1: Instantaneous Mean (High Alpha / Fast)
        addr_fast = x_input.mean(dim=0).mean(dim=0).detach()

        # Address 2: Weighted Variance (Mid Alpha)
        addr_mid = (x_input.mean(0) * 0.7 + trace.mean(0) * 0.3).mean(0).detach()

        # Address 3: Trace-dominant (Low Alpha / Slow/Structural)
        addr_slow = trace.mean(dim=0).mean(dim=0).detach()

        self.auditor.report_candidate_breach({
            "confidence": confidence,
            "divergence": 1.0 - confidence,
            "category": category,
            "reasoning_trace": trace.detach().clone(),
            "centroid_addresses": [addr_fast, addr_mid, addr_slow]  # Multi-scale list
        })

    def evaluate_gate(self, metrics):
        """Monitors performance to unlock the next developmental stage."""
        current_stage_name = list(self.stage_models.keys())[self.current_stage]
        targets = self.thresholds.get(current_stage_name, {})

        if all(metrics.get(k, 0) >= v for k, v in targets.items()):
            if self.current_stage < len(self.stage_models) - 1:
                self.current_stage += 1
                return True
        return False

    def update_parameters(self, epoch):
        """Standard eta decay - overridden by Auditor's Metabolic Rebound."""
        base_decay = 0.9 ** (self.current_stage + epoch)
        self.eta = max(0.01, self.eta * base_decay)

    def shutdown(self):
        if self.auditor:
            self.auditor.shutdown()