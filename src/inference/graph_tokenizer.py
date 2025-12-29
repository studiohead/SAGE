import torch
import torch.nn.functional as F
import re

class GraphTokenizer:
    """
    Simple tokenizer that maps text tokens to graph node embeddings.
    Works with SAGE SharedConceptGraph.
    """
    def __init__(self, graph, device=None):
        self.graph = graph
        self.device = device or self.graph.device
        self.node_matrix = None
        self.node_keys = list(graph.node_order)

    def encode(self, text):
        """
        Convert input text into a tensor of embeddings from nodes in the graph.
        For simplicity, matches words/phrases to nodes and returns their embeddings.
        """
        tokens = text.split()  # simple whitespace tokenizer
        embs = []
        for t in tokens:
            for node_id, node in self.graph.nodes.items():
                # crude matching: node_id or text in node_key (or your own metadata)
                if t.lower() in str(node_id).lower():
                    embs.append(node.embedding.detach())
        if not embs:
            # fallback: random vector
            embs.append(torch.randn(self.graph.embedding_dim, device=self.graph.device))
        return torch.stack(embs)

    def _update_cache(self):
        """Refresh embedding matrix for token lookup."""
        self.node_matrix = self.graph.get_graph_embedding_matrix().detach().to(self.device)
        self.node_keys = list(self.graph.node_order)

    def tokenize(self, text):
        """
        Convert text into a list of node embeddings.
        Splits text by words, punctuation, and maps to nearest node in embedding space.
        """
        if self.node_matrix is None or self.graph.version != getattr(self, "_last_version", -1):
            self._update_cache()
            self._last_version = self.graph.version

        # Simple word tokenization
        tokens = re.findall(r"\w+|\S", text.lower())

        embeddings = []
        for token in tokens:
            # Convert token to a small random vector (placeholder)
            token_vec = torch.randn(self.graph.embedding_dim, device=self.device)

            # Normalize for cosine similarity
            token_vec = F.normalize(token_vec, p=2, dim=0)

            # Compute similarity to all graph nodes
            sims = torch.matmul(self.node_matrix, token_vec)
            best_idx = torch.argmax(sims).item()
            node_key = self.node_keys[best_idx]

            embeddings.append(self.graph.nodes[node_key].embedding)

        return embeddings

    def decode(self, embeddings):
        """
        Convert embeddings back to closest node tokens (node IDs as strings).
        """
        if self.node_matrix is None or self.graph.version != getattr(self, "_last_version", -1):
            self._update_cache()
            self._last_version = self.graph.version

        decoded_tokens = []
        for emb in embeddings:
            emb = emb.to(self.device)
            sims = torch.matmul(self.node_matrix, F.normalize(emb, p=2, dim=0))
            best_idx = torch.argmax(sims).item()
            node_key = self.node_keys[best_idx]
            decoded_tokens.append(str(node_key))

        return decoded_tokens
