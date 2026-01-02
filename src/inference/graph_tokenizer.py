import torch
import torch.nn.functional as F
import re


class GraphTokenizer:
    """
    Tokenizer that maps text <-> graph node embeddings and returns node labels.
    Designed to work with SharedConceptGraph.
    """

    def __init__(self, graph, device=None):
        self.graph = graph
        self.device = device or self.graph.device
        self.node_matrix = None
        self.node_keys = []
        self._last_version = -1

    # ---------- INTERNAL CACHE ----------

    def _update_cache(self):
        self.node_matrix = (
            self.graph.get_graph_embedding_matrix()
            .detach()
            .to(self.device)
        )
        self.node_keys = list(self.graph.node_order)
        self._last_version = self.graph.version

    def _ensure_cache(self):
        if self.node_matrix is None or self.graph.version != self._last_version:
            self._update_cache()

    # ---------- ENCODE ----------

    def encode(self, text):
        """
        Convert input text into a tensor of embeddings from nodes in the graph.
        Matches words/phrases to node labels instead of node IDs.
        """
        tokens = text.lower().split()
        embs = []
        for t in tokens:
            for node_id, node in self.graph.nodes.items():
                if hasattr(node, "label") and node.label and t in str(node.label).lower():
                    embs.append(node.embedding.detach())
        if not embs:
            # fallback: use the mean embedding of all nodes
            embs.append(torch.mean(self.graph.get_graph_embedding_matrix(), dim=0))
        return torch.stack(embs)

    # ---------- DECODE ----------

    def decode(self, embeddings, return_labels=True):
        """
        Convert embeddings back to node labels (default) or node IDs.
        """
        self._ensure_cache()

        outputs = []

        for emb in embeddings:
            emb = emb.to(self.device)
            emb = F.normalize(emb, p=2, dim=0)

            sims = torch.matmul(self.node_matrix, emb)
            best_idx = torch.argmax(sims).item()
            node_key = self.node_keys[best_idx]
            node = self.graph.nodes[node_key]

            if return_labels and hasattr(node, "label"):
                outputs.append(node.label)
            else:
                outputs.append(str(node_key))

        return outputs
