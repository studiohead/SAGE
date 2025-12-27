import torch
import json
import os
import time
from datetime import datetime


class SAGEAnalyticsEngine:
    def __init__(self, output_dir="analytics"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = []

    def capture_snapshot(self, stage_name, stage_idx, epoch, loss, telemetry, graph):
        # 1. OPTIMIZED EMBEDDING EXTRACTION
        # Uses the shared graph's internal matrix method we've been using elsewhere
        all_embs = graph.get_graph_embedding_matrix().detach()

        if all_embs.shape[0] > 0:
            variance = torch.var(all_embs, dim=0).mean().item()
            radius = torch.norm(all_embs, p=2, dim=1).mean().item()
        else:
            variance = 0.0
            radius = 0.0

        # 2. ALIGN WITH NEW STAGE OUTPUTS
        # We now check for 'impact' (Adult) or 'confidence' (Infant-Teen)
        gamma = telemetry.get('impact', telemetry.get('confidence', 0.0))

        # Elder specific divergence check
        divergence = telemetry.get('divergence', 0.0)

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "stage_idx": stage_idx,
            "epoch": epoch,
            "metrics": {
                "loss": round(float(loss), 6),
                "gamma_confidence": round(float(gamma), 4),
                "manifold_variance": round(float(variance), 6),
                "manifold_radius": round(float(radius), 4),
                "topological_divergence": round(float(divergence), 4)  # New for Elder
            },
            "governance": {
                "active_nodes": all_embs.shape[0],
                "edge_density": self._calc_density(graph)
            }
        }

        self.history.append(snapshot)
        return snapshot

    def _calc_density(self, graph):
        n = len(graph.nodes)
        if n < 2: return 0.0
        e = sum(len(node.connections) for node in graph.nodes.values())
        return e / (n * (n - 1))

    def save_stage_report(self, stage_name):
        filename = f"SAGE_LOG_{stage_name}_{int(time.time())}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"[+] Analytics Report saved: {path}")
        self.history = []

    def get_manifold_drift(self):
        if len(self.history) < 2:
            return 0.0
        prev_var = self.history[-2]["metrics"]["manifold_variance"]
        curr_var = self.history[-1]["metrics"]["manifold_variance"]
        return curr_var - prev_var