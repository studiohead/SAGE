import torch
import torch.nn.functional as F


class SAGEInference:
    def __init__(self, graph, tokenizer):
        self.graph = graph
        self.tokenizer = tokenizer

    def respond(self, prompt, top_k=5):
        """
        Retrieves semantic labels from the graph based on the prompt.
        Returns human-readable labels (not internal node IDs).
        """

        # 1. Tokenize / Encode prompt
        if not hasattr(self.tokenizer, "encode"):
            raise RuntimeError("Tokenizer must have an encode() method.")

        # GraphTokenizer.encode() returns a tensor
        prompt_emb = self.tokenizer.encode(prompt)

        # Ensure 2D tensor
        if prompt_emb.dim() == 1:
            prompt_emb = prompt_emb.unsqueeze(0)

        # 2. Device alignment
        device = self.graph.device
        prompt_emb = prompt_emb.to(device)

        # 3. Retrieve semantic context
        # Returns: [(embedding, node_id), ...]
        context_data = self.graph.retrieve_manifold_context(
            prompt_emb, top_k=top_k
        )

        # 4. Resolve node IDs -> labels at inference boundary
        response_labels = []
        for _, node_id in context_data:
            node_key = str(node_id)

            # nn.ModuleDict does NOT have .get(); check existence first
            if node_key not in self.graph.nodes:
                continue

            node = self.graph.nodes[node_key]

            # Always return the label
            label = getattr(node, "label", None)
            if label is not None:
                response_labels.append(str(label))
            else:
                # fallback if label somehow missing
                response_labels.append(node_key)

        return response_labels
