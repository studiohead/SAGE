import torch
from config.config import STAGE_HYPERPARAMS


class ManifoldHealthTracker:
    def __init__(self, graph):
        self.graph = graph

    def check_health(self, stage_name, epoch):
        """
        SAGE HEALTH MONITOR: Evaluates the latent geometry and topological density
        to detect 'Hairballs' or 'Manifold Collapse'.

        Surgically updated to align with the Hinge-Loss Pressure engine.
        """
        # Extract embeddings using the live master tensors
        node_matrix = self.graph.get_graph_embedding_matrix().detach()

        # SUTURE: Handle empty graph state to prevent NaN/Crash
        if node_matrix is None or node_matrix.size(0) == 0:
            print(f"\n>>> [MANIFOLD HEALTH: {stage_name} | Epoch {epoch}] - EMPTY GRAPH")
            return

        # 1. GEOMETRIC DIVERSITY (The Variance Term)
        if node_matrix.size(0) > 1:
            # We calculate global variance across the feature dimension
            variance = torch.var(node_matrix).item()
        else:
            variance = 0.0

        # 2. LATENT PRESSURE (Synced with Hinge Loss logic)
        # We now pull the specific target for the current stage.
        stage_key = stage_name.capitalize()
        hparams = STAGE_HYPERPARAMS.get(stage_key, {})
        target_var = hparams.get("max_structural_density", 0.20)

        # Pressure is now 'Potential Repulsion'.
        # 0.0 means the manifold has reached its target breathing room.
        latent_pressure = max(0.0, target_var - variance)

        # 3. CENTROID DRIFT
        # Measures the average distance of nodes from the center of the manifold.
        global_centroid = node_matrix.mean(dim=0, keepdim=True)
        dist = torch.norm(node_matrix - global_centroid, p=2, dim=1)
        centroid_radius = dist.mean().item()

        # 4. STRUCTURAL DENSITY
        node_count = len(self.graph.nodes)
        total_edges = sum(len(node.connections) for node in self.graph.nodes.values())
        structural_density = total_edges / max(1, node_count * 100)

        # 5. Output Report
        print(f"\n>>> [MANIFOLD HEALTH: {stage_name} | Epoch {epoch}]")
        print(f"    - Variance:    {variance:.6f}  (Geometric Spread)")
        print(f"    - Pressure:    {latent_pressure:.4f}  (Residual Repulsion)")
        print(f"    - Centroid Radius:       {centroid_radius:.4f}  (Centroid Stability)")
        print(f"    - Density:     {structural_density:.4f}  (Structural Hairball)")
        print(f"    - Total Nodes: {node_count}")
        print(f"    - Total Edges: {total_edges}")

        # HEURISTIC ALERTS
        if structural_density > hparams.get("max_structural_density", 0.20) * 4:  # Adaptive threshold
            print("    [!] WARNING: HIGH STRUCTURAL DENSITY. PRUNING RECOMMENDED.")

        if latent_pressure > 0.1:
            print("    [!] WARNING: HIGH LATENT PRESSURE. MANIFOLD IS COMPRESSED.")
        elif latent_pressure == 0.0 and variance > target_var * 1.5:
            print("    [!] WARNING: MANIFOLD DISSIPATION. REDUCE PRESSURE WEIGHT.")

        print("-" * 45)