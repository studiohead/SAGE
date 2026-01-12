import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS

EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')


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
        # Reduction to single vector
        # x_input shape: [seq, batch, dim]
        current_centroid = x_input.mean(dim=0).mean(dim=0)
        raw_weights = self.centroid_comparator(current_centroid)

        num_active = len(stage_outputs)
        active_weights = torch.softmax(raw_weights[:num_active], dim=-1)

        # Slice to original sequence length
        target_len = x_input.size(0)
        processed_outputs = [out[:target_len] for out in stage_outputs]

        # Stack outputs and apply weights
        stacked = torch.stack(processed_outputs, dim=-1)
        weights_view = active_weights.view(1, 1, 1, -1)

        # Weighted summation and normalization
        weighted_sum = (stacked * weights_view).sum(dim=-1)
        return self.refiner(weighted_sum)


class SAGEContainer(nn.Module):
    def __init__(self, graph, stage_models, thresholds, auditor=None, gh=None, governance_mode="SAGE_DELEGATED",
                 layer_start=0, layer_end=12, training_layers=12):
        super().__init__()
        self.graph = graph
        self.gh = gh  # <--- SUTURE: Direct link to the Growth Engine
        self.stage_names = ["Infant", "Toddler", "Preschool", "Gradeschool", "Teen", "Adult", "Elder"]
        self.stages = nn.ModuleDict(stage_models)
        self.thresholds = thresholds
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.training_layers = training_layers

        self.auditor = auditor
        if self.auditor:
            self.auditor.mode = governance_mode
            self.auditor.container = self

        self.current_stage_idx = 0
        self.eta = 1.0

        # Hardware Source of Truth
        self.register_buffer("_hw_fix", torch.zeros(1))

        sample_transformer = next(iter(stage_models.values()))
        self.hidden_dim = getattr(sample_transformer, 'embed_dim', EMBED_DIM)
        self.fusion = StageFusion(len(stage_models), self.hidden_dim)

    @property
    def device(self):
        return self._hw_fix.device

    def forward(self, x, graph_matrix=None, centroid_addresses=None):
        """Hardware Firewall enforced via property detection."""
        dev = self.device
        x = x.to(dev)
        stage_outputs = []

        # SUTURE: Always fetch the graph matrix fresh from the graph.
        if graph_matrix is None:
            graph_matrix = self.graph.get_graph_embedding_matrix().to(dev)

        if centroid_addresses is not None:
            centroid_addresses = [addr.to(dev) for addr in centroid_addresses]

        seq_len_orig = x.size(0)
        batch_size = x.size(1)

        final_telemetry = {
            "confidence": 1.0,
            "gamma_divergence": torch.tensor(0.0, device=dev),
            "category": "NULL",
            "trace": None,
            "winner_node_ids": None  # CRITICAL: For Hebbian distribution
        }

        for i, name in enumerate(self.stage_names):
            if i > self.current_stage_idx:
                break

            model = self.stages[name].to(dev)

            x_aug = x
            if i >= 4:  # Teen+ Manifold Retrieval
                memory_block = self.graph.retrieve_manifold_context(x.mean(0))
                if memory_block:
                    context_tensors = torch.stack([item[0].to(dev) for item in memory_block])
                    mem_expanded = context_tensors.unsqueeze(1).expand(-1, batch_size, -1)
                    x_aug = torch.cat([x, mem_expanded], dim=0)

            out, impact = model(
                x_aug,
                graph_matrix,
                centroid_addresses=centroid_addresses,
                Wi=self.eta
            )

            # Standardizing impact tensor
            if isinstance(impact, dict):
                impact_val = impact.get("gamma_divergence", torch.tensor(0.0, device=dev))
            else:
                impact_val = torch.as_tensor(impact, device=dev)

            # Stability check for NaNs before fusion
            if torch.isnan(impact_val) or torch.isnan(out).any():
                impact_val = torch.tensor(1.0, device=dev)
                out = torch.nan_to_num(out, nan=0.0)

            # Output alignment
            if out.size(0) != seq_len_orig:
                out = out[:seq_len_orig]
            stage_outputs.append(out)

            if i == self.current_stage_idx:
                # SUTURE: BREAKING THE NODE 0 COLLAPSE
                # We project the output trace against the manifold to find topological winners.
                with torch.no_grad():
                    # Projection: [Batch, Dim]
                    projection = out.mean(0)
                    # Similarity to every node in the graph: [Batch, NumNodes]
                    # graph_matrix is normalized by the graph, so this is Cosine Similarity
                    logits = torch.matmul(projection, graph_matrix.t())

                    # Apply Temperature scaling based on stage (Infant needs high entropy)
                    temp = 2.0 if self.current_stage_idx == 0 else 1.0
                    probs = torch.softmax(logits / temp, dim=-1)

                    # Sample winners to ensure we don't just hit the top-1 (identity trap)
                    winner_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    final_telemetry["winner_node_ids"] = winner_indices.tolist()

                final_telemetry.update({
                    "confidence": torch.exp(-impact_val).item(),
                    "gamma_divergence": impact_val,
                    "category": "STRUCTURAL_ERROR" if i < 4 else "FACTUAL_DISPUTE",
                    "trace": out.detach()
                })

        # Fusion logic
        self.fusion.to(dev)
        fused_output = self.fusion(x, stage_outputs) if len(stage_outputs) > 1 else stage_outputs[0]
        final_telemetry["trace"] = fused_output

        return fused_output, final_telemetry

    def dispatch_remediation_request(self, x_input, trace, confidence, category):
        """Forces all remediation data to device before crossing thread boundaries."""
        dev = self.device
        addr_fast = x_input.mean(dim=0).mean(dim=0).detach().to(dev)
        addr_mid = (x_input.mean(0) * 0.7 + trace.mean(0) * 0.3).mean(0).detach().to(dev)
        addr_slow = trace.mean(dim=0).mean(dim=0).detach().to(dev)

        if self.auditor:
            self.auditor.report_candidate_breach({
                "confidence": confidence,
                "divergence": 1.0 - confidence,
                "category": category,
                "reasoning_trace": trace.detach().clone().to(dev),
                "centroid_addresses": [addr_fast, addr_mid, addr_slow]
            })

    def trigger_metabolic_rebound(self, impact_score):
        """Adjusts plasticity based on auditor feedback."""
        rebound_factor = 1.0 + (impact_score * 0.5)
        self.eta = min(2.0, self.eta * rebound_factor)

    def update_plasticity_window(self, cumulative=False):
        """Locks/Unlocks layers based on the current developmental stage."""
        current_name = self.stage_names[self.current_stage_idx]
        params = STAGE_HYPERPARAMS[current_name]
        start, end = params["layer_start"], params["layer_end"]

        if not cumulative:
            for param in self.parameters():
                param.requires_grad = False

        all_layers = []
        for name in self.stage_names:
            if name in self.stages:
                all_layers.extend(self.stages[name].layers)

        for i in range(start, min(end, len(all_layers))):
            layer = all_layers[i]
            for param in layer.parameters():
                param.requires_grad = True
            layer.train()

        self.graph.ensure_stage_initialized(self.current_stage_idx)

    def evaluate_gate(self, metrics):
        """Evaluates if the model can move to the next developmental stage."""
        current_name = self.stage_names[self.current_stage_idx]
        targets = self.thresholds.get(current_name, {})
        if all(metrics.get(k, 0) >= v for k, v in targets.items()):
            if self.current_stage_idx < len(self.stage_names) - 1:
                self.current_stage_idx += 1
                self.update_plasticity_window()
                return True
        return False

    def update_parameters(self, epoch):
        """Decays the authority of eta over time."""
        stage_boost = 1.0 / (1.0 + self.current_stage_idx * 0.1)
        self.eta = max(0.5, stage_boost * (0.95 ** epoch))

    def save_agnostic_stage(self, stage_name, checkpoint_dir="checkpoints"):
        """Saves weights while stripping hardware-specific metadata."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        file_path = os.path.join(checkpoint_dir, f"SAGE_STATE_{stage_name}.pth")
        state_to_save = {
            "stage": stage_name,
            "model_state": self.state_dict(),
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "current_stage_idx": self.current_stage_idx
        }
        torch.save(state_to_save, file_path)
        print(f"[*] Agnostic Stage Weights secured for {stage_name} at {file_path}")

    def load_agnostic_stage(self, stage_name, checkpoint_dir="checkpoints"):
        """Loads weights into the model, ensuring hardware compatibility."""
        load_path = os.path.join(checkpoint_dir, f"SAGE_STATE_{stage_name}.pth")
        if not os.path.exists(load_path):
            return False

        checkpoint = torch.load(load_path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint)

        new_state = self.state_dict()
        updated_keys = 0
        for key, value in state.items():
            if key in new_state and value.shape == new_state[key].shape:
                new_state[key] = value
                updated_keys += 1

        self.load_state_dict(new_state, strict=False)
        self.current_stage_idx = checkpoint.get("current_stage_idx", self.current_stage_idx)
        print(f"[*] Resumed Agnostic Stage: {stage_name} ({updated_keys} keys updated)")
        return True

    def shutdown(self):
        """Properly terminates the Auditor thread."""
        if self.auditor:
            self.auditor.shutdown()