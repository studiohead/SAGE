import torch
import torch.nn.functional as F
import re


class GraphTokenizer:
    """
    Tokenizer that maps text <-> graph node embeddings and returns node labels.
    Designed to work with SharedConceptGraph.
    Optimized for high-node counts (8,000+) using O(1) hash-map lookups.
    """

    def __init__(self, graph, device=None):
        self.graph = graph
        self.device = device or self.graph.device
        self.node_matrix = None
        self.node_keys = []
        self.label_to_id = {}  # The O(1) Speed Bridge
        self._last_version = -1

    # ---------- INTERNAL CACHE ----------

    def _update_cache(self):
        """
        Rebuilds the vectorized matrix and the label hash-map.
        Called whenever graph.version incremented.
        """
        # 1. Math Cache: Vectorized matrix for cosine similarity (decode)
        self.node_matrix = (
            self.graph.get_graph_embedding_matrix()
            .detach()
            .to(self.device)
        )
        self.node_keys = list(self.graph.node_order)

        # 2. Symbolic Cache: Rebuild the O(1) Label Map for 8,000+ nodes
        self.label_to_id = {
            str(node.label).lower(): nid
            for nid, node in self.graph.nodes.items()
            if hasattr(node, 'label') and node.label is not None
        }

        self._last_version = self.graph.version

    def _ensure_cache(self):
        """Lazy update: Only rebuilds if the graph has evolved."""
        if self.node_matrix is None or self.graph.version != self._last_version:
            self._update_cache()

    # ---------- ENCODE ----------

    def encode(self, text):
        """
        Convert input text into a tensor of embeddings via O(1) lookup.
        Matches words/phrases to node labels directly.
        """
        self._ensure_cache()

        # Clean and split text
        tokens = text.lower().split()
        embs = []

        for t in tokens:
            # DIRECT LOOKUP: O(1) instead of looping 8,000 times
            node_id = self.label_to_id.get(t)
            if node_id:
                node = self.graph.nodes[node_id]
                embs.append(node.embedding.detach())

        if not embs:
            # Null-Space Fallback: use the mean embedding of the current manifold
            embs.append(torch.mean(self.node_matrix, dim=0))

        return torch.stack(embs)

    # ---------- DECODE ----------

    def decode(self, embeddings, return_labels=True):
        """
        Convert embeddings back to node labels via vectorized dot-products.
        Highly efficient on MPS/CUDA for 8,000+ nodes.
        """
        self._ensure_cache()

        outputs = []

        for emb in embeddings:
            emb = emb.to(self.device)
            emb = F.normalize(emb, p=2, dim=0)

            # Parallel similarity check across the entire manifold
            sims = torch.matmul(self.node_matrix, emb)
            best_idx = torch.argmax(sims).item()
            node_key = self.node_keys[best_idx]

            if return_labels:
                node = self.graph.nodes[node_key]
                outputs.append(getattr(node, 'label', node_key))
            else:
                outputs.append(str(node_key))

        return outputs