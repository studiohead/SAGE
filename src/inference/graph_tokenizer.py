import torch
import torch.nn.functional as F


class GraphTokenizer:
    """
    Tokenizer mapping text <-> graph node embeddings.
    Works with SharedConceptGraph and large node counts (8k+ nodes).
    """

    def __init__(self, graph, device=None):
        self.graph = graph
        self.device = device or self.graph.device
        self.node_matrix = None
        self.node_keys = []
        self.label_to_id = {}  # O(1) lookup
        self._last_version = -1

    # ---------- CACHE MANAGEMENT ----------

    def _update_cache(self):
        """
        Rebuilds vectorized embedding matrix and label hash-map.
        """
        num_nodes = len(self.graph.node_order)
        if num_nodes == 0:
            self.node_matrix = torch.zeros((0, self.graph.embedding_dim), device=self.device)
            self.node_keys = []
            self.label_to_id = {}
            self._last_version = self.graph.version
            return

        # Pull embeddings from master_embeddings
        indices = [self.graph.node_order.index(k) for k in self.graph.node_order]
        self.node_matrix = self.graph.master_embeddings[indices].detach().to(self.device)
        self.node_keys = list(self.graph.node_order)

        # Build label -> node_id map
        self.label_to_id = {
            str(node.label).lower(): nid
            for nid, node in self.graph.nodes.items()
            if node.label is not None
        }

        self._last_version = self.graph.version

    def _ensure_cache(self):
        if self.node_matrix is None or self.graph.version != self._last_version:
            self._update_cache()

    # ---------- ENCODING ----------

    def encode(self, text):
        """
        Converts input text into tensor of embeddings via O(1) lookup.
        Uses mean manifold embedding if no match.
        Returns a [tokens, embed_dim] tensor.
        """
        self._ensure_cache()
        tokens = text.lower().split()
        embs = []

        for t in tokens:
            node_id = self.label_to_id.get(t)
            if node_id is not None:
                node_key = str(node_id)
                if node_key in self.graph.nodes:
                    idx = self.graph.node_order.index(node_key)
                    embs.append(self.graph.master_embeddings[idx].detach())
                    continue

            # Fallback: use mean of active embeddings
            if self.node_matrix.size(0) > 0:
                embs.append(self.node_matrix.mean(dim=0))
            else:
                embs.append(torch.zeros(self.graph.embedding_dim, device=self.device))

        return torch.stack(embs)

    # ---------- DECODING ----------

    def decode(self, embeddings, return_labels=True):
        """
        Converts embeddings back to node labels via cosine similarity.
        Returns a list of labels or node IDs.
        """
        self._ensure_cache()
        outputs = []

        for emb in embeddings:
            emb = F.normalize(emb.to(self.device), p=2, dim=0)

            if self.node_matrix.size(0) == 0:
                outputs.append("<empty>")
                continue

            sims = torch.matmul(self.node_matrix, emb)
            best_idx = torch.argmax(sims).item()
            node_key = self.node_keys[best_idx]

            if return_labels:
                node = self.graph.nodes.get(node_key)
                if node and getattr(node, "label", None) is not None:
                    outputs.append(node.label)
                else:
                    outputs.append(node_key)
            else:
                outputs.append(str(node_key))

        return outputs
