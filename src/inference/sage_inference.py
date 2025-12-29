import torch
import torch.nn.functional as F

class SAGEInference:
    def __init__(self, graph, tokenizer):
        self.graph = graph
        self.tokenizer = tokenizer

    def respond(self, prompt, top_k=5):
        # Tokenize the prompt into latent representation
        if hasattr(self.tokenizer, "encode"):
            prompt_emb = self.tokenizer.encode(prompt).unsqueeze(0)  # [1, embed_dim]
        else:
            raise RuntimeError("Tokenizer must have an encode() method.")

        # Move to same device as graph
        device = self.graph.device
        prompt_emb = prompt_emb.to(device)

        # Retrieve top-k nearest nodes
        context_tensors = self.graph.retrieve_manifold_context(prompt_emb, top_k=top_k)

        response_texts = []
        for i, tensor in enumerate(context_tensors):
            # Try to find the corresponding node
            node_id = None
            for nid, node in self.graph.nodes.items():
                if torch.allclose(F.normalize(node.embedding, p=2, dim=0), F.normalize(tensor, p=2, dim=0)):
                    node_id = nid
                    break

            if node_id is not None:
                node = self.graph.nodes[node_id]
                label = getattr(node, "label", None)
                response_texts.append(label if label is not None else str(node_id))  # always string
            else:
                response_texts.append("")  # fallback empty string
