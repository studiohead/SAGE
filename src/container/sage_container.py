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
        # 1. Calculate current EWMA-aligned centroid for fusion weighting
        current_centroid = x_input.mean(dim=0).mean(dim=0)

        # 2. Get Raw Weights
        raw_weights = self.centroid_comparator(current_centroid)

        # 3. DYNAMIC SLICE: Only take weights for active stages
        num_active = len(stage_outputs)
        active_weights = torch.softmax(raw_weights[:num_active], dim=-1)

        # 4. Slice to original sequence length
        target_len = x_input.size(0)
        processed_outputs = [out[:target_len] for out in stage_outputs]

        # 5. Stack outputs and apply weights
        stacked = torch.stack(processed_outputs, dim=-1)
        weights_view = active_weights.view(1, 1, 1, -1)

        # 6. Weighted summation and normalization
        weighted_sum = (stacked * weights_view).sum(dim=-1)
        return self.refiner(weighted_sum)


class SAGEContainer(nn.Module):
    def __init__(self, graph, stage_models, thresholds, auditor=None, governance_mode="SAGE_DELEGATED",
                 layer_start=0, layer_end=12, training_layers=12):
        super().__init__()
        self.graph = graph
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sample_transformer = next(iter(stage_models.values()))
        self.hidden_dim = getattr(sample_transformer, 'embed_dim', EMBED_DIM)
        self.fusion = StageFusion(len(stage_models), self.hidden_dim)

        self._cached_matrix = None
        self._last_graph_version = -1

    def _get_cached_graph_matrix(self):
        current_version = getattr(self.graph, 'version', 0)
        if self._cached_matrix is None or current_version != self._last_graph_version:
            self._cached_matrix = self.graph.get_graph_embedding_matrix().to(self.device)
            self._last_graph_version = current_version
        return self._cached_matrix

    def update_plasticity_window(self, cumulative=False):
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

    def forward(self, x, graph_matrix=None, centroid_addresses=None):
        stage_outputs = []
        final_telemetry = {
            "confidence": 1.0,
            "gamma_divergence": torch.tensor(0.0, device=x.device),
            "category": "NULL",
            "trace": None
        }

        if graph_matrix is None:
            graph_matrix = self._get_cached_graph_matrix()

        if torch.isnan(graph_matrix).any():
            graph_matrix = torch.nan_to_num(graph_matrix, nan=0.0)

        seq_len_orig = x.size(0)

        for i, name in enumerate(self.stage_names):
            if i > self.current_stage_idx:
                break

            model = self.stages[name]

            # Contextual Augmentation
            x_aug = x
            if i >= 4:  # Teen stage and above
                # memory_block is a list of tuples: [(embedding, node_id), ...]
                memory_block = self.graph.retrieve_manifold_context(x.mean(0))

                if memory_block is not None and len(memory_block) > 0:
                    # Unpack the list of tuples into a stacked tensor of embeddings
                    context_tensors = torch.stack([item[0] for item in memory_block])

                    # Align dimensions for transformer concatenation
                    mem_expanded = context_tensors.unsqueeze(1).expand(-1, x.size(1), -1)
                    x_aug = torch.cat([x, mem_expanded], dim=0)

            # Model Execution
            out, impact = model(
                x_aug,
                graph_matrix,
                centroid_addresses=centroid_addresses,
                Wi=self.eta
            )

            # Numerical Stability
            if isinstance(impact, dict):
                impact_tensor = impact.get("gamma_divergence", torch.tensor(0.0, device=x.device))
            else:
                impact_tensor = torch.as_tensor(impact, device=x.device)

            if torch.isnan(impact_tensor) or torch.isnan(out).any():
                impact_tensor = torch.tensor(1.0, device=x.device)
                out = torch.nan_to_num(out, nan=0.0)

            if out.size(0) != seq_len_orig:
                out = out[:seq_len_orig]

            stage_outputs.append(out)

            # Telemetry Extraction
            current_confidence = torch.exp(-impact_tensor)
            if i == self.current_stage_idx:
                final_telemetry.update({
                    "confidence": current_confidence.item(),
                    "gamma_divergence": impact_tensor,
                    "category": "STRUCTURAL_ERROR" if i < 4 else "FACTUAL_DISPUTE",
                    "trace": out.detach()
                })

        # Hierarchical Fusion
        fused_output = self.fusion(x, stage_outputs) if len(stage_outputs) > 1 else stage_outputs[0]
        final_telemetry["trace"] = fused_output

        return fused_output, final_telemetry

    def dispatch_remediation_request(self, x_input, trace, confidence, category):
        addr_fast = x_input.mean(dim=0).mean(dim=0).detach()
        addr_mid = (x_input.mean(0) * 0.7 + trace.mean(0) * 0.3).mean(0).detach()
        addr_slow = trace.mean(dim=0).mean(dim=0).detach()

        if self.auditor:
            self.auditor.report_candidate_breach({
                "confidence": confidence,
                "divergence": 1.0 - confidence,
                "category": category,
                "reasoning_trace": trace.detach().clone(),
                "centroid_addresses": [addr_fast, addr_mid, addr_slow]
            })

    def evaluate_gate(self, metrics):
        current_name = self.stage_names[self.current_stage_idx]
        targets = self.thresholds.get(current_name, {})

        if all(metrics.get(k, 0) >= v for k, v in targets.items()):
            if self.current_stage_idx < len(self.stage_names) - 1:
                self.current_stage_idx += 1
                self.update_plasticity_window()
                return True
        return False

    def update_parameters(self, epoch):
        stage_boost = 1.0 / (1.0 + self.current_stage_idx * 0.1)
        self.eta = max(0.5, stage_boost * (0.95 ** epoch))

    def save_agnostic_stage(self, stage_name, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        file_path = os.path.join(checkpoint_dir, f"SAGE_STATE_{stage_name}.pth")
        state_to_save = {
            "stage": stage_name,
            "model_state": self.state_dict(),
            "layer_start": self.layer_start,
            "layer_end": self.layer_end
        }
        torch.save(state_to_save, file_path)
        print(f"[*] Agnostic Stage Weights secured for {stage_name} at {file_path}")

    def load_agnostic_stage(self, stage_name, checkpoint_dir="checkpoints"):
        load_path = os.path.join(checkpoint_dir, f"SAGE_STATE_{stage_name}.pth")
        if not os.path.exists(load_path):
            return False

        checkpoint = torch.load(load_path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        new_state = self.state_dict()

        updated_keys = 0
        for key, value in state.items():
            if key in new_state and value.shape == new_state[key].shape:
                new_state[key] = value
                updated_keys += 1

        self.load_state_dict(new_state)
        print(f"[*] SAGEContainer: Surgically injected {updated_keys} keys from {stage_name}.")
        return True

    def shutdown(self):
        if self.auditor:
            self.auditor.shutdown()