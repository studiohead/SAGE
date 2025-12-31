import torch
import torch.nn.functional as F

class SAGEInference:
    def __init__(self, graph, tokenizer):
        self.graph = graph
        self.tokenizer = tokenizer

    def respond(self, prompt, top_k=5):
        """
        Retrieves semantic labels from the graph based on the prompt.
        Optimized to use direct node lookup instead of tensor comparison.
        """
        # 1. Tokenize/Encode the prompt
        if hasattr(self.tokenizer, "encode"):
            # Ensure we get a tensor back
            prompt_emb = self.tokenizer.encode(prompt, convert_to_tensor=True)
            if prompt_emb.dim() == 1:
                prompt_emb = prompt_emb.unsqueeze(0)
        else:
            raise RuntimeError("Tokenizer must have an encode() method.")

        # 2. Alignment
        device = self.graph.device
        prompt_emb = prompt_emb.to(device)

        # 3. Retrieve Context (Returns list of tuples: [(tensor, node_id), ...])
        context_data = self.graph.retrieve_manifold_context(prompt_emb, top_k=top_k)

        # 4. Extract Labels directly from the Graph nodes
        response_texts = []
        for _, node_id in context_data:
            # We already have the ID, so lookup is O(1)
            node_key = str(node_id)
            if node_key in self.graph.nodes:
                node = self.graph.nodes[node_key]
                # Priority: 'label' property -> 'node_id' string fallback
                label = getattr(node, "label", None)
                response_texts.append(str(label) if label is not None else node_key)
            else:
                response_texts.append("") # Fallback for ghost indices

        return response_texts