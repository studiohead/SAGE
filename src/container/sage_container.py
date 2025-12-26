# src/container/sage_container.py
import os
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

        # RESTORED SURGICAL STATE
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.training_layers = training_layers

        self.auditor = auditor
        if self.auditor:
            self.auditor.mode = governance_mode
            self.auditor.container = self

        self.current_stage_idx = 0
        self.eta = 1.0  # Global Plasticity Factor (Wi)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Infer hidden dim from the first available stage model
        sample_transformer = next(iter(stage_models.values()))
        self.hidden_dim = getattr(sample_transformer, 'embed_dim', 512)
        self.fusion = StageFusion(len(stage_models), self.hidden_dim)

    def update_plasticity_window(self, cumulative=False):
        current_name = self.stage_names[self.current_stage_idx]
        params = STAGE_HYPERPARAMS[current_name]
        start, end = params["layer_start"], params["layer_end"]

        if not cumulative:
            # Global Freeze
            for param in self.parameters():
                param.requires_grad = False

        # Sequential Layer Harvesting
        all_layers = []
        for name in self.stage_names:
            if name in self.stages:
                all_layers.extend(self.stages[name].layers)

        # Surgical Thaw (aperture)
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

        # MANIFOLD INTEGRITY CHECK:
        # If the manifold itself is corrupted, we must stop before the models touch it.
        if torch.isnan(graph_matrix).any():
            print("[!] CRITICAL: Manifold corruption (NaN) detected. Recovering zero-field.")
            graph_matrix = torch.nan_to_num(graph_matrix, nan=0.0)

        seq_len_orig = x.size(0)

        for i, name in enumerate(self.stage_names):
            if i > self.current_stage_idx:
                break

            model = self.stages[name]

            # 1. TCR Logic
            x_aug = x
            if i >= 5:  # Teen+ Retrieval
                memory_block = self.graph.retrieve_manifold_context(x.mean(0))
                if memory_block.size(0) > 0:
                    mem_expanded = memory_block.unsqueeze(1).expand(-1, x.size(1), -1)
                    x_aug = torch.cat([x, mem_expanded], dim=0)

            # 2. Process Abstraction Level
            out, impact = model(x_aug, graph_matrix, Wi=self.eta)

            # --- NUMERICAL HARDENING ---
            if torch.isnan(impact) or torch.isnan(out).any():
                if self.training:
                    print(f"[!] Warning: Stage {i} ({name}) exploded. Clamping for recovery.")
                impact = torch.tensor(1.0, device=x.device)  # Force max divergence
                out = torch.nan_to_num(out, nan=0.0)

            # 3. Restore original sequence length
            if out.size(0) != seq_len_orig:
                out = out[:seq_len_orig]

            stage_outputs.append(out)

            # --- DIAGNOSTIC PROBE ---
            current_gamma = 1.0 - torch.clamp(torch.as_tensor(impact), 0, 1).item()
            if self.training:
                print(f"[PROBE] Stage {i} ({name}) -> Divergence: {impact.item():.6f} | Γ: {current_gamma:.4f}")

            # 4. Confidence Tracking
            if i == self.current_stage_idx:
                min_confidence = current_gamma
                breach_category = "STRUCTURAL_ERROR" if i < 4 else "FACTUAL_DISPUTE"
            else:
                min_confidence = min(min_confidence, current_gamma)

            # 5. Early Exit (Inference only)
            if not self.training and current_gamma > 0.98 and i > 1:
                break

        # 6. Hierarchical Fusion
        fused_output = self.fusion(x, stage_outputs) if len(stage_outputs) > 1 else stage_outputs[0]

        # Final Safety Check for Fusion
        if torch.isnan(fused_output).any():
            fused_output = torch.nan_to_num(fused_output, nan=0.0)

        # 7. Remediation Triggering
        if min_confidence < 0.5 and self.auditor and not self.training:
            self.dispatch_remediation_request(x, fused_output, min_confidence, breach_category)

        return fused_output, {"confidence": min_confidence, "category": breach_category, "trace": fused_output}

    def dispatch_remediation_request(self, x_input, trace, confidence, category):
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

    def save_agnostic_stage(self, stage_name: str, checkpoint_dir: str = "checkpoints"):
        """Saves stage-specific weights (Muscles)."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        file_path = os.path.join(checkpoint_dir, f"SAGE_STATE_{stage_name}.pth")

        state_to_save = {
            "stage": stage_name,
            "model_state": self.state_dict(), # standardized key for loader
            "layer_start": self.layer_start,
            "layer_end": self.layer_end
        }
        torch.save(state_to_save, file_path)
        print(f"[*] Agnostic Stage Weights secured for {stage_name} at {file_path}")

    def load_agnostic_stage(self, stage_name: str, checkpoint_dir: str = "checkpoints"):
        load_path = os.path.join(checkpoint_dir, f"SAGE_STATE_{stage_name}.pth")
        if not os.path.exists(load_path):
            print(f"[!] Warning: No checkpoint found at {load_path}")
            return False

        checkpoint = torch.load(load_path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))

        # 1. Get the current live state dict
        new_state = self.state_dict()

        # 2. Filter and Map: We want to inherit layers (Muscles) from the PREVIOUS stage
        # but keep our CURRENT stage's unique parameters (Relational Weights, etc.)
        updated_keys = 0
        for key, value in state.items():
            # If the key exists in our current container, try to map it
            if key in new_state:
                # OPTIONAL: Add logic here to skip stage-specific parameters
                # e.g., if "relational_weight" in key: continue
                if value.shape == new_state[key].shape:
                    new_state[key] = value
                    updated_keys += 1

        self.load_state_dict(new_state)
        print(f"[*] SAGEContainer: Surgically injected {updated_keys} keys from {stage_name}.")
        return True

    def shutdown(self):
        if self.auditor:
            self.auditor.shutdown()