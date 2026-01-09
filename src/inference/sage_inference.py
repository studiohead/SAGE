import torch
import torch.nn.functional as F


class SAGEInference:
    """
    Inference interface for SharedConceptGraph (SAGE / Citizen/Elder).
    Provides conceptual transparency by exposing thinking trajectories,
    anchors, and identifying the state of graph nodes (Active, Tombstoned, or Incinerated).
    """

    def __init__(self, graph, tokenizer, broad_concepts=None):
        """
        Initialize the inference engine.

        Args:
            graph: The SharedConceptGraph instance.
            tokenizer: Tokenizer with an encode() method.
            broad_concepts: List of strings (e.g., ["0"..."9", "A"..."Z"]) used for anchor detection.
        """
        self.graph = graph
        self.tokenizer = tokenizer
        # BROAD_CONCEPTS: list of strings at the front of the seeding
        self.broad_concepts = broad_concepts if broad_concepts else []

    def respond(self, prompt, top_k=5, return_impact=False, return_full_path=False):
        """
        Retrieves top_k semantically relevant nodes and optionally traverses the graph.

        Returns:
            Labels, novelty scores, or a full thinking trace dictionary.
        """
        if not hasattr(self.tokenizer, "encode"):
            raise RuntimeError("Tokenizer must have encode() method")

        # --- 1. Encode prompt ---
        prompt_emb = self.tokenizer.encode(prompt)
        if prompt_emb.dim() == 1:
            prompt_emb = prompt_emb.unsqueeze(0)

        # Ensure device consistency
        device = getattr(self.graph, "device", "cpu")
        prompt_emb = prompt_emb.to(device)

        # --- 2. Retrieve top-k context ---
        # The graph's retrieve_manifold_context handles the 256-dim matrix math
        context_data = self.graph.retrieve_manifold_context(prompt_emb, top_k=top_k)

        response_labels = []
        retrieved_indices = []

        for _, node_id in context_data:
            node_key = str(node_id)

            # ModuleDict does not support .get(); using 'in' check for Incineration detection
            if node_key not in self.graph.nodes:
                response_labels.append(f"[INCINERATED:{node_key}]")
            else:
                node = self.graph.nodes[node_key]
                if getattr(node, "is_tombstoned", False):
                    response_labels.append(f"[TOMBSTONED:{node_key}]")
                else:
                    label = str(node.label) if node.label is not None else node_key
                    response_labels.append(label)

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
                    # Vectorized similarity check for novelty
                    cos_sim = torch.matmul(retrieved_embs, prompt_vec)
                    novelty = 1.0 - cos_sim.clamp(0.0, 1.0)
                    impact_score = novelty.mean().item()
                else:
                    impact_score = 1.0
            else:
                impact_score = 1.0

        # --- 4. Optional full graph path with Analysis ---
        if return_full_path and retrieved_indices:
            start_nodes = [str(self.graph.node_order[idx]) for idx in retrieved_indices]
            thinking_trace = self.traverse_graph(start_nodes=start_nodes, max_depth=10)

            if return_impact:
                return thinking_trace, impact_score
            return thinking_trace

        if return_impact:
            return response_labels, impact_score
        return response_labels

    def traverse_graph(self, start_nodes=None, max_depth=3):
        """
        Recursive traversal to show how the model moves through concepts.
        Identifies if paths hit Tombstones or break due to Incineration.
        """
        if start_nodes is None:
            start_nodes = list(self.graph.node_order)

        visited = set()
        trace = {
            "path_labels": [],
            "node_indices": [],
            "anchors_detected": set(),
            "tombstones_encountered": [],
            "incinerations_encountered": []
        }

        def dfs(node_key, depth):
            if node_key in visited or depth > max_depth:
                return
            visited.add(node_key)

            # --- 1. Incineration Detection (Key missing from ModuleDict) ---
            if node_key not in self.graph.nodes:
                trace["path_labels"].append(f"[INCINERATED:{node_key}]")
                trace["node_indices"].append(node_key)
                trace["incinerations_encountered"].append(node_key)
                return

            node = self.graph.nodes[node_key]

            # --- 2. Tombstone Detection (Marker exists but node is dead) ---
            if getattr(node, "is_tombstoned", False):
                trace["path_labels"].append(f"[TOMBSTONE:{node_key}]")
                trace["node_indices"].append(node_key)
                trace["tombstones_encountered"].append(node_key)
                return  # Stop traversal; do not follow connections of a tombstone

            # --- 3. Active Node Processing ---
            trace["path_labels"].append(node.label if node.label else node_key)
            trace["node_indices"].append(node_key)

            # --- Path Analysis: Anchor Detection (Elder Logic) ---
            try:
                # Check numeric index against the BROAD_CONCEPTS range
                node_idx_val = int(node_key)
                if node_idx_val < len(self.broad_concepts):
                    trace["anchors_detected"].add(self.broad_concepts[node_idx_val])
            except ValueError:
                # Check for direct string matches if keys are alphanumeric
                if node_key in self.broad_concepts:
                    trace["anchors_detected"].add(node_key)

            # Recurse through logical connections established during training
            if hasattr(node, "connections") and node.connections:
                for nbr in node.connections:
                    dfs(str(nbr), depth + 1)

        for nk in start_nodes:
            dfs(str(nk), 0)

        # Finalize trace
        trace["anchors_detected"] = list(trace["anchors_detected"])
        return trace

    def get_path_logic_summary(self, thinking_trace):
        """
        Narrative summary of the Thinking Trace for human review.
        """
        labels = " -> ".join(thinking_trace["path_labels"])
        anchors = ", ".join(thinking_trace["anchors_detected"])

        summary = f"Logic Chain: {labels}\nRoot Anchors: {anchors}"

        if thinking_trace["tombstones_encountered"]:
            summary += f"\nWarning: Path stopped by Tombstones at {thinking_trace['tombstones_encountered']}"
        if thinking_trace["incinerations_encountered"]:
            summary += f"\nCRITICAL: Path broken by missing (Incinerated) nodes at {thinking_trace['incinerations_encountered']}"

        return summary