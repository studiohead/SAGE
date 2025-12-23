import torch
import json
import os
import time
from datetime import datetime


class SAGEAnalyticsEngine:
    """
    Developmental Analytics Suite for SAGE.
    Tracks Manifold Stability, Aperture Efficiency, and Abstraction Drift.
    """

    def __init__(self, output_dir="analytics"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = []

    def capture_snapshot(self, stage_name, stage_idx, epoch, loss, telemetry, graph):
        nodes = list(graph.nodes.values())

        # Calculate variance/radius only if nodes exist, otherwise use defaults
        if nodes:
            all_embs = torch.stack([n.embedding.detach() for n in nodes])
            variance = torch.var(all_embs, dim=0).mean().item()
            radius = torch.norm(all_embs, p=2, dim=1).mean().item()
        else:
            variance = 0.0
            radius = 0.0

        gamma = telemetry.get('confidence', 0.0)
        category = telemetry.get('category', 'NULL')

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
            },
            "governance": {
                "breach_category": category,
                "plasticity_eta": None
            }
        }

        self.history.append(snapshot)
        return snapshot

    def save_stage_report(self, stage_name):
        """Dumps the session history to a JSON file for the Auditor."""
        filename = f"SAGE_LOG_{stage_name}_{int(time.time())}.json"
        path = os.path.join(self.output_dir, filename)

        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"[+] Analytics Report saved: {path}")
        # Clear history for next stage to prevent memory bloat
        self.history = []

    def get_manifold_drift(self):
        """
        Calculates how much the concept locations shifted compared
        to the previous snapshot. (Useful for detecting 'Concept Bleed').
        """
        if len(self.history) < 2:
            return 0.0

        prev_var = self.history[-2]["metrics"]["manifold_variance"]
        curr_var = self.history[-1]["metrics"]["manifold_variance"]
        return curr_var - prev_var