import torch

class ManifoldHealthTracker:
    def __init__(self, graph):
        self.graph = graph

    def check_health(self, stage_name, epoch):
        # Extract embeddings
        node_matrix = self.graph.get_graph_embedding_matrix().detach()

        # SUTURE: Handle empty graph state to prevent NaN/Crash
        if node_matrix is None or node_matrix.size(0) == 0:
            print(f"\n>>> [MANIFOLD HEALTH: {stage_name} | Epoch {epoch}] - EMPTY GRAPH")
            return

        # 1. Geometric Spread (Diversity)
        # unbiased=False prevents 1/0 errors if there is only 1 node
        if node_matrix.size(0) > 1:
            variance = torch.var(node_matrix, dim=0, unbiased=False).mean().item()
        else:
            variance = 0.0

        # 2. Centroid Stability (Tightness)
        global_centroid = node_matrix.mean(dim=0, keepdim=True)
        dist = torch.norm(node_matrix - global_centroid, p=2, dim=1)
        tightness = dist.mean().item()

        # 3. Map Complexity (Hebbian Growth)
        total_edges = sum(len(node.connections) for node in self.graph.nodes.values())

        print(f"\n>>> [MANIFOLD HEALTH: {stage_name} | Epoch {epoch}]")
        print(f"    - Variance:  {variance:.6f}")
        print(f"    - Tightness: {tightness:.4f}")
        print(f"    - Edges:     {total_edges}")
        print("-" * 45)