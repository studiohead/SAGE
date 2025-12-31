import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import SHARED_MODEL_CONFIG

EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')


class ConceptNode(nn.Module):
    def __init__(self, node_id, embedding_dim=EMBED_DIM, label=None):
        super().__init__()
        self.node_id = node_id
        # Human-readable identifier (e.g., "A", "0", "len(letters)")
        self.label = label

        # Weights represent the structural material of the concept
        self.embedding = nn.Parameter(torch.randn(embedding_dim) * 0.1)
        self.alignment_score = 1.0
        self.connections = {}  # {neighbor_node_id: weight}

        # NULL-SPACE DYNAMICS
        self.is_tombstoned = False
        self.tombstone_key = None
        self.gradient_mask = 1.0
        self.stage_idx = -1


class SharedConceptGraph(nn.Module):
    def __init__(self, embedding_dim=EMBED_DIM, lambda_ewma=0.1, promotion_threshold=0.7):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Using ModuleDict so Parameters are registered and movable to GPU/MPS
        self.nodes = nn.ModuleDict()
        self.node_order = []
        self.anchor_tensor = {}  # The 'Z' Map (Topological Anchors)
        self.lambda_ewma = lambda_ewma  # EWMA Decay for temporal drift
        self.quarantined_centroids = []  # List of addresses flagged by Auditor
        self.promotion_threshold = promotion_threshold  # threshold for "promotable" nodes

        # SPEED CACHE
        self.version = 0
        self._cached_matrix = None
        self._last_version = -1
        self._cached_mask = None

    @property
    def device(self):
        """Dynamic device detection for MacBook (CPU/MPS) or Server (CUDA)."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def __setstate__(self, state):
        """Governance: Clears cache on load to prevent memory pointer corruption."""
        super().__setstate__(state)
        self._cached_matrix = None
        self._last_version = -1
        self._cached_mask = None

    def add_node(self, node_id, label=None):
        """Surgical entry point: Now supports passing an initial label."""
        node_key = str(node_id)
        if node_key not in self.nodes:
            self.nodes[node_key] = ConceptNode(node_id, self.embedding_dim, label=label)
            self.node_order.append(node_key)
            self.version += 1

    def get_node_index(self, node_id):
        """Translates Concept ID to position in the vectorized matrix."""
        node_key = str(node_id)
        try:
            return self.node_order.index(node_key)
        except ValueError:
            return -1

    # --- TOPOLOGICAL ANCHORING (EWMA) ---

    def update_local_centroid(self, node_id):
        """EWMA Centroid Anchoring: Z_t+1 = (1 - λ)Z_t + λ(C_v_new)."""
        node_key = str(node_id)
        node = self.nodes[node_key]
        neighbors = list(node.connections.keys())
        device = self.device

        if not neighbors:
            c_v_new = node.embedding.detach().clone().to(device)
        else:
            neighbor_embs = torch.stack([self.nodes[str(nid)].embedding.detach() for nid in neighbors]).to(device)
            c_v_new = neighbor_embs.mean(dim=0)

        z_t = self.anchor_tensor.get(node_id, c_v_new).to(device)
        updated_z = (1.0 - self.lambda_ewma) * z_t + self.lambda_ewma * c_v_new
        self.anchor_tensor[node_id] = F.normalize(updated_z, p=2, dim=0)

    def is_node_in_centroid_range(self, node_id, centroid_address, radius=0.7):
        """Auditor check for conceptual breaches."""
        if node_id not in self.anchor_tensor:
            return False
        z_vec = self.anchor_tensor[node_id].to(centroid_address.device)
        return torch.norm(z_vec - centroid_address) < radius

    def get_graph_embedding_matrix(self):
        """Vectorized manifold retrieval with Cache-Optimization."""
        if not self.node_order:
            return torch.zeros((1, self.embedding_dim), device=self.device)

        if self._cached_matrix is None or self.version != self._last_version:
            current_device = self.device
            raw_embs = torch.stack([self.nodes[nid].embedding for nid in self.node_order])
            alignment_scores = torch.tensor([self.nodes[nid].alignment_score for nid in self.node_order], device=current_device).unsqueeze(1)
            tombstone_mask = torch.tensor([0.0 if self.nodes[nid].is_tombstoned else 1.0 for nid in self.node_order], device=current_device).unsqueeze(1)

            self._cached_mask = alignment_scores * tombstone_mask
            self._cached_matrix = raw_embs
            self._last_version = self.version

        norm_embs = F.normalize(self._cached_matrix, p=2, dim=1)
        return norm_embs * self._cached_mask

    # --- REMEDIATION DYNAMICS ---

    def execute_topological_incineration(self, node_id):
        node_key = str(node_id)
        if node_key in self.nodes:
            node = self.nodes[node_key]
            with torch.no_grad():
                node.embedding.zero_()
            node.connections = {}
            node.alignment_score = 0.0
            node.gradient_mask = 0.0
            node.is_tombstoned = False
            self.version += 1
            self.update_local_centroid(node_id)

    def execute_manifold_tombstone(self, node_id, r_tomb):
        node_key = str(node_id)
        if node_key in self.nodes:
            node = self.nodes[node_key]
            r_tomb = r_tomb.to(self.device)
            node.tombstone_key = r_tomb
            with torch.no_grad():
                displaced = torch.matmul(node.embedding.data, r_tomb)
                node.embedding.copy_(displaced)
            node.is_tombstoned = True
            node.alignment_score = 0.0
            self.version += 1
            self.update_local_centroid(node_id)

    def restore_from_tombstone(self, node_id):
        node_key = str(node_id)
        if node_key in self.nodes:
            node = self.nodes[node_key]
            if not node.is_tombstoned or node.tombstone_key is None: return
            inverse_r = node.tombstone_key.t()
            with torch.no_grad():
                restored = torch.matmul(node.embedding.data, inverse_r)
                node.embedding.copy_(restored)
            node.is_tombstoned = False
            node.alignment_score = 1.0
            node.tombstone_key = None
            self.version += 1
            self.update_local_centroid(node_id)

    # --- MANIFOLD RETRIEVAL ---

    def retrieve_manifold_context(self, current_latent, top_k=5):
        """
        MODIFIED: Returns a list of (Tensor, Node_ID) tuples.
        This allows SAGEInference to retrieve labels in O(1).
        """
        current_device = current_latent.device
        query_coord = current_latent.mean(dim=0) if current_latent.dim() > 1 else current_latent
        query_coord = F.normalize(query_coord, p=2, dim=0)

        scored_anchors = []
        for cid, z_vec in self.anchor_tensor.items():
            z_vec = z_vec.to(current_device)
            if any(torch.norm(z_vec - q.to(current_device)) < 0.1 for q in self.quarantined_centroids):
                continue

            sim = F.cosine_similarity(query_coord.unsqueeze(0), z_vec.unsqueeze(0), dim=1)
            scored_anchors.append((cid, sim.item()))

        scored_anchors.sort(key=lambda x: x[1], reverse=True)

        context_data = []
        for cid, sim_score in scored_anchors[:top_k]:
            node_key = str(cid)
            node = self.nodes[node_key]
            if sim_score > 0.7 and node.alignment_score > 0.1 and not node.is_tombstoned:
                normalized_emb = F.normalize(node.embedding, p=2, dim=0).to(current_device)
                context_data.append((normalized_emb, node_key))

        return context_data

    # --- HEBBIAN DYNAMICS ---

    def update_stage_aware_hebbian(self, attention_map, batch_indices=None, stage_plasticity=1.0, threshold=None):
        if not self.node_order:
            return 0

        active_threshold = threshold if threshold is not None else self.promotion_threshold

        if batch_indices is None:
            limit = min(attention_map.size(1), len(self.node_order))
            active_ids = self.node_order[:limit]
        else:
            unique_indices = list(set(batch_indices))
            active_ids = [str(nid) for nid in unique_indices if str(nid) in self.nodes]

        if not active_ids:
            return 0

        try:
            active_embs = torch.stack([self.nodes[nid].embedding.data for nid in active_ids])
            active_embs = F.normalize(active_embs, p=2, dim=1)
            full_manifold = self.get_graph_embedding_matrix().detach()
            cross_sim = torch.mm(active_embs, full_manifold.t())
        except Exception as e:
            print(f"[!] Hebbian Compute Error: {e}")
            return 0

        mask = cross_sim > active_threshold
        indices = mask.nonzero(as_tuple=False)
        new_edges_born = 0

        for i in range(indices.size(0)):
            row, col = indices[i]
            uid_active, uid_target = active_ids[row], self.node_order[col]
            if uid_active == uid_target: continue

            sim_val = cross_sim[row, col].item()
            node = self.nodes[uid_active]

            if uid_target not in node.connections:
                new_edges_born += 1
                current_w = 0.0
            else:
                current_w = node.connections[uid_target]

            node.connections[uid_target] = (current_w * 0.99) + (stage_plasticity * sim_val)

        return new_edges_born

    def ensure_stage_initialized(self, stage_idx):
        for node_id in self.node_order:
            node_key = str(node_id)
            node = self.nodes[node_key]
            if getattr(node, "stage_idx", -1) >= stage_idx or node.is_tombstoned: continue
            if node.alignment_score >= self.promotion_threshold:
                node.stage_idx = stage_idx

        if not hasattr(self, "stage_anchors"): self.stage_anchors = {}
        if stage_idx not in self.stage_anchors:
            self.stage_anchors[stage_idx] = torch.zeros(self.embedding_dim, device=self.device)

    # --- NODE CREATION ---

    def create_node_from_trace(self, trace, label=None):
        new_node_id = max([int(nid) for nid in self.node_order], default=-1) + 1
        new_node = ConceptNode(new_node_id, embedding_dim=self.embedding_dim, label=label).to(self.device)

        if isinstance(trace, torch.Tensor):
            with torch.no_grad():
                new_node.embedding.copy_(trace.view(-1)[:self.embedding_dim].to(self.device))

        self.nodes[str(new_node_id)] = new_node
        self.node_order.append(str(new_node_id))
        self.version += 1
        self.update_local_centroid(new_node_id)
        return new_node_id

    def apply_edge_threshold(self, min_weight=0.05):
        """Deactivate edges below the threshold."""
        for node_id in self.node_order:
            node = self.nodes[node_id]
            to_remove = [nbr for nbr, w in node.connections.items() if w < min_weight]
            for nbr in to_remove:
                del node.connections[nbr]
        self.version += 1

    def adaptive_prune(self, tightness, max_tightness=0.5, prune_fraction=0.1):
        """
        Remove weakest edges if tightness exceeds max_tightness.
        prune_fraction: fraction of edges to remove.
        Memory-efficient: avoids unnecessary copies and sorts only when needed.
        """
        if tightness <= max_tightness:
            return

        for node_id in self.node_order:
            node = self.nodes[node_id]
            if not node.connections:
                continue

            # Only sort the number of edges we might prune to save memory
            num_to_prune = max(1, int(len(node.connections) * prune_fraction))
            # Get the keys of edges with smallest weights without creating a full list
            smallest_edges = sorted(node.connections.items(), key=lambda x: x[1])[:num_to_prune]
            for nbr, _ in smallest_edges:
                del node.connections[nbr]

        self.version += 1

    def compute_manifold_variance(self, active_nodes=None):
        """
        Measures clustering around Z-anchors.
        If active_nodes is provided, only considers these nodes.
        Memory-efficient: uses generator expressions, avoids stacking large tensors.
        """

        def node_pairs_variance(node, others_iter):
            for other in others_iter:
                if other.is_tombstoned or other.alignment_score <= 0.0:
                    continue
                yield torch.sum((other.embedding - node.embedding.to(other.embedding.device)) ** 2)

        variances = []
        nodes_to_check = (self.nodes[str(nid)] for nid in active_nodes) if active_nodes else self.nodes.values()

        for node in nodes_to_check:
            if node.is_tombstoned or node.alignment_score <= 0.0:
                continue
            diffs = node_pairs_variance(node, self.nodes.values())
            try:
                mean_diff = sum(diffs) / len(list(diffs))
                variances.append(mean_diff)
            except ZeroDivisionError:
                continue

        return sum(variances) / len(variances) if variances else 0.0

