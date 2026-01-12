import torch
import torch.nn.functional as F


class SAGEInference:
    """
    Inference interface for SharedConceptGraph (SAGE / Citizen/Elder).
    Provides conceptual transparency by exposing thinking trajectories,
    anchors, and identifying the state of graph nodes (Active, Tombstoned, or Incinerated).
    Supports interactive demotion feedback during inference.
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
        self.broad_concepts = broad_concepts if broad_concepts else []

    def respond(self, prompt, top_k=5, return_impact=False, return_full_path=False, interactive=False):
        """
        Retrieves top_k semantically relevant nodes and optionally traverses the graph.
        Supports interactive feedback for demotion.

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
        context_data = self.graph.retrieve_manifold_context(prompt_emb, top_k=top_k)

        response_labels = []
        retrieved_indices = []

        for _, node_id in context_data:
            node_key = str(node_id)

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

        # --- 5. Interactive feedback for demotion ---
        if interactive and response_labels:
            print(f"\nTop-{len(response_labels)} nodes for prompt '{prompt}':")
            for idx, label in enumerate(response_labels):
                print(f"{idx}: {label}")
            print("\nType '/demote <index> [<decay>]' to weaken a node, or press Enter to continue.")

            user_input = input(">> ").strip()
            if user_input.startswith("/demote"):
                try:
                    parts = user_input.split()
                    if len(parts) >= 2:
                        demote_idx = int(parts[1])
                        decay_factor = float(parts[2]) if len(parts) == 3 else 0.5
                        if 0 <= demote_idx < len(context_data):
                            _, node_id = context_data[demote_idx]
                            node_key = str(node_id)
                            node = self.graph.nodes.get(node_key, None)
                            if node is not None:
                                for target_key in list(node.connections.keys()):
                                    node.connections[target_key] *= decay_factor
                                    if target_key in self.graph.nodes:
                                        target_node = self.graph.nodes[target_key]
                                        if node_key in target_node.connections:
                                            target_node.connections[node_key] *= decay_factor
                                self.graph.version += 1
                                print(f"[+] Demoted node {node_key} with decay factor {decay_factor}")
                        else:
                            print("[!] Invalid index for demotion.")
                except Exception as e:
                    print(f"[!] Error processing demotion: {e}")

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

            if node_key not in self.graph.nodes:
                trace["path_labels"].append(f"[INCINERATED:{node_key}]")
                trace["node_indices"].append(node_key)
                trace["incinerations_encountered"].append(node_key)
                return

            node = self.graph.nodes[node_key]

            if getattr(node, "is_tombstoned", False):
                trace["path_labels"].append(f"[TOMBSTONE:{node_key}]")
                trace["node_indices"].append(node_key)
                trace["tombstones_encountered"].append(node_key)
                return

            trace["path_labels"].append(node.label if node.label else node_key)
            trace["node_indices"].append(node_key)

            try:
                node_idx_val = int(node_key)
                if node_idx_val < len(self.broad_concepts):
                    trace["anchors_detected"].add(self.broad_concepts[node_idx_val])
            except ValueError:
                if node_key in self.broad_concepts:
                    trace["anchors_detected"].add(node_key)

            if hasattr(node, "connections") and node.connections:
                for nbr in node.connections:
                    dfs(str(nbr), depth + 1)

        for nk in start_nodes:
            dfs(str(nk), 0)

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
