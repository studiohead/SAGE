import torch
import torch.nn.functional as F


class SAGEInference:
    """
    Inference interface for SharedConceptGraph (SAGE / Citizen/Elder).
    Returns human-readable labels, optional novelty/impact scores,
    and can optionally traverse the graph to show full paths.
    """

    def __init__(self, graph, tokenizer):
        self.graph = graph
        self.tokenizer = tokenizer

    def respond(self, prompt, top_k=5, return_impact=False, return_full_path=False):
        """
        Retrieves top_k semantically relevant nodes for a prompt.
        Optionally returns novelty score and full connected concept paths.

        Args:
            prompt (str): Input text prompt.
            top_k (int): Number of nearest nodes to retrieve.
            return_impact (bool): Whether to compute novelty/impact.
            return_full_path (bool): Whether to traverse graph starting from top nodes.

        Returns:
            List[str]: Node labels or full graph path.
            Optional[float]: Novelty/impact score if requested.
        """
        if not hasattr(self.tokenizer, "encode"):
            raise RuntimeError("Tokenizer must have encode() method")

        # --- 1. Encode prompt ---
        prompt_emb = self.tokenizer.encode(prompt)
        if prompt_emb.dim() == 1:
            prompt_emb = prompt_emb.unsqueeze(0)
        device = getattr(self.graph, "device", "cpu")
        prompt_emb = prompt_emb.to(device)

        # --- 2. Retrieve top-k context ---
        context_data = self.graph.retrieve_manifold_context(prompt_emb, top_k=top_k)

        response_labels = []
        retrieved_indices = []

        for _, node_id in context_data:
            node_key = str(node_id)
            node = self.graph.nodes[node_key] if node_key in self.graph.nodes else None
            if node is None:
                response_labels.append(f"<missing:{node_key}>")
            else:
                response_labels.append(str(node.label) if node.label is not None else node_key)
                if node_key in self.graph.node_order:
                    retrieved_indices.append(self.graph.node_order.index(node_key))

        # --- 3. Impact / novelty score ---
        impact_score = None
        if return_impact:
            if retrieved_indices:
                manifold = self.graph.get_graph_embedding_matrix()
                if len(retrieved_indices) > 0 and manifold.size(0) > 0:
                    retrieved_embs = F.normalize(manifold[retrieved_indices], p=2, dim=1)
                    prompt_vec = F.normalize(prompt_emb.mean(dim=0), p=2, dim=0)
                    cos_sim = torch.matmul(retrieved_embs, prompt_vec)
                    novelty = 1.0 - cos_sim.clamp(0.0, 1.0)
                    impact_score = novelty.mean().item()
                else:
                    impact_score = 1.0
            else:
                impact_score = 1.0

        # --- 4. Optional full graph path ---
        if return_full_path and retrieved_indices:
            start_nodes = [str(self.graph.node_order[idx]) for idx in retrieved_indices]
            full_path_labels = self.traverse_graph(start_nodes=start_nodes, max_depth=10)
            if return_impact:
                return full_path_labels, impact_score
            return full_path_labels

        if return_impact:
            return response_labels, impact_score
        return response_labels

    def traverse_graph(self, start_nodes=None, max_depth=3):
        """
        Returns a list of nodes following connections up to max_depth.
        """
        if start_nodes is None:
            start_nodes = list(self.graph.node_order)

        visited = set()
        path_nodes = []

        def dfs(node_key, depth):
            if node_key in visited or depth > max_depth:
                return
            visited.add(node_key)
            node = self.graph.nodes[node_key]
            path_nodes.append(node.label if node.label else node_key)
            for nbr in node.connections:
                dfs(str(nbr), depth + 1)

        for nk in start_nodes:
            dfs(str(nk), 0)

        return path_nodes
