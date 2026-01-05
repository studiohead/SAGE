import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS

EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')


class ConceptNode(nn.Module):
    """
    Surgical Node Metadata: Tracks the topological 'soul' of a concept.
    Maintains connectivity, maturation state, and tombstone status.
    """

    def __init__(self, node_id, label=None):
        super().__init__()
        self.node_id = node_id
        self.label = label

        # --- STRUCTURAL INFRASTRUCTURE ---
        self.connections = {}  # {neighbor_node_id: weight}
        self.is_tombstoned = False
        self.tombstone_key = None  # Stores the rotation matrix for displacement
        self.stage_idx = -1
        self.gradient_mask = 1.0


class SharedConceptGraph(nn.Module):
    def __init__(self, embedding_dim=EMBED_DIM, lambda_ewma=0.1, promotion_threshold=0.7, max_nodes=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_nodes = max_nodes
        self.nodes = nn.ModuleDict()
        self.node_order = []
        self.anchor_tensor = {}
        self.lambda_ewma = lambda_ewma
        self.promotion_threshold = promotion_threshold

        # --- CONTIGUOUS GRADIENT BLOCK ---
        # We use a very tight uniform initialization to maximize 'Repulsive Tension'
        # against your 0.20 target_var.
        self.master_embeddings = nn.Parameter(torch.empty(max_nodes, embedding_dim).uniform_(-0.01, 0.01))
        self.master_alignments = nn.Parameter(torch.empty(max_nodes, 1).uniform_(0.4, 0.9))

        # --- STATE BUFFERS ---
        self.register_buffer("active_mask", torch.zeros(max_nodes, 1))
        self.register_buffer("tombstone_mask", torch.ones(max_nodes, 1))

        # --- CACHE & VERSIONING ---
        self.version = 0
        self._cached_matrix = None
        self._last_version = -1
        self._cached_anchor_matrix = None

    @property
    def device(self):
        return self.master_embeddings.device

    def __setstate__(self, state):
        super().__setstate__(state)
        self._cached_matrix = None
        self._last_version = -1
        self._cached_anchor_matrix = None

    def add_node(self, node_id, label=None):
        """Registers a new node and activates its slot in the master parameters."""
        node_key = str(node_id)
        if node_key not in self.nodes:
            idx = len(self.node_order)
            if idx >= self.max_nodes:
                return

            new_node = ConceptNode(node_id, label=label)
            self.nodes[node_key] = new_node
            self.node_order.append(node_key)

            # Activate and initialize the slot
            with torch.no_grad():
                self.active_mask[idx] = 1.0
                # Jitter helps the pressure engine find different directions for concepts
                self.master_embeddings[idx].uniform_(-0.01, 0.01)

            self.version += 1
            self.update_local_centroid(node_id)

    def update_local_centroid(self, node_id):
        """Updates the EWMA anchor for context retrieval and structural grounding."""
        node_key = str(node_id)
        if node_key not in self.nodes: return

        idx = self.node_order.index(node_key)
        neighbors = [str(nid) for nid in self.nodes[node_key].connections.keys() if str(nid) in self.nodes]

        if not neighbors:
            c_v_new = F.normalize(self.master_embeddings[idx].detach().clone(), p=2, dim=0)
        else:
            n_indices = [self.node_order.index(nid) for nid in neighbors]
            neighbor_embs = self.master_embeddings[n_indices].detach()
            c_v_new = neighbor_embs.mean(dim=0)

        z_t = self.anchor_tensor.get(node_id, c_v_new).to(self.device)
        updated_z = (1.0 - self.lambda_ewma) * z_t + self.lambda_ewma * c_v_new
        self.anchor_tensor[node_id] = F.normalize(updated_z, p=2, dim=0)
        self._cached_anchor_matrix = None

    def get_graph_embedding_matrix(self):
        """
        The Central Matrix: Returns the live, differentiable manifold.

        DYNAMIC LOGIC (REPAIRED):
        1. Employs a Dimension-Aware radial ceiling to prevent spherical saturation.
        2. Scaled for 256-D geometry: Replaces the restrictive 2.5 cap with
           a ceiling tied to sqrt(EMBED_DIM).
        3. Maintains Autograd 'Wet Suture' by bypassing cache during training.
        """
        num_active = len(self.node_order)
        if num_active == 0:
            return torch.zeros((0, self.embedding_dim), device=self.device)

        # In training mode, we MUST ignore the cache to ensure the gradient
        # path back to master_embeddings and master_alignments is never broken.
        if self.training or self._cached_matrix is None or self.version != self._last_version:
            embs = self.master_embeddings[:num_active]

            # --- DYNAMIC POPULATION CEILING (Living Graph Final Fix) ---
            # We use the square root of the dimension (16.0 for 256-D) as the
            # anchor for conceptual volume expansion.
            # This ensures the Manifold can scale to 100,000+ nodes without
            # ever hitting a 'Hard Wall' or stalling Pressure.

            dim_base = math.sqrt(self.embedding_dim)  # 16.0
            pop_factor = math.log10(max(10, num_active))  # ~3.89 for 8k nodes

            # This expands the ceiling to ~15.5, giving the Pressure Engine
            # massive runway to push past the .1756 stall point.
            dynamic_ceiling = dim_base * (pop_factor / 4.0)

            # Clamp alignments using the new expanded dynamic ceiling.
            aligns = torch.clamp(self.master_alignments[:num_active], 0.05, dynamic_ceiling)

            # Pull the tombstone mask to zero-out suppressed/incinerated nodes
            masks = self.tombstone_mask[:num_active]

            # Normalization (L2) ensures we maintain a directional hypersphere basis.
            # Added eps=1e-8 to prevent NaNs during high-pressure repulsion.
            norm_embs = F.normalize(embs, p=2, dim=1, eps=1e-8)

            # RES is the final 'Sutured' tensor that the Frontend and Loss will see.
            res = (norm_embs * aligns) * masks

            # Cache is only used for Inference/Testing to save compute.
            if not self.training:
                self._cached_matrix = res
                self._last_version = self.version
            return res

        return self._cached_matrix

    def retrieve_manifold_context(self, current_latent, top_k=5):
        """Retrieves semantic neighbors using vectorized similarity against anchors."""
        if not self.anchor_tensor: return []

        current_device = current_latent.device
        query = F.normalize(current_latent.mean(dim=0) if current_latent.dim() > 1 else current_latent, p=2, dim=0)

        if self._cached_anchor_matrix is None:
            anchor_list = [self.anchor_tensor[int(nid)] for nid in self.node_order if int(nid) in self.anchor_tensor]
            if not anchor_list: return []
            self._cached_anchor_matrix = torch.stack(anchor_list).to(current_device)

        similarities = torch.mv(self._cached_anchor_matrix, query)
        vals, idxs = torch.topk(similarities, k=min(top_k, similarities.size(0)))

        context_data = []
        for val, idx in zip(vals, idxs):
            if val > 0.7:
                node_key = self.node_order[idx.item()]
                if not self.nodes[node_key].is_tombstoned:
                    # Pull from live embeddings to allow context to influence training
                    live_vec = F.normalize(self.master_embeddings[idx.item()], p=2, dim=0)
                    context_data.append((live_vec.to(current_device), node_key))
        return context_data

    def update_stage_aware_hebbian(self, stage_key, tightness, attention_map, batch_indices=None, stage_plasticity=1.0,
                                   threshold=None):
        """Updates synaptic weights between nodes based on co-occurrence in the manifold."""
        if not self.node_order: return 0

        active_threshold = threshold if threshold is not None else self.promotion_threshold

        # Initial wiring boost for empty graphs
        if sum(len(n.connections) for n in self.nodes.values()) == 0:
            active_threshold = 0.2

        if batch_indices is None:
            active_idxs = list(range(len(self.node_order)))
        else:
            active_idxs = [self.node_order.index(str(nid)) for nid in batch_indices if str(nid) in self.nodes]

        if not active_idxs: return 0

        # Detach for structural updates to prevent graph-internal cycles
        active_embs = F.normalize(self.master_embeddings[active_idxs].detach(), p=2, dim=1)
        full_manifold = self.get_graph_embedding_matrix().detach()
        cross_sim = torch.mm(active_embs, full_manifold.t())

        top_vals, top_indices = torch.topk(cross_sim, k=min(50, full_manifold.size(0)), dim=1)

        new_edges = 0
        # Decay logic: As tightness (pressure) decreases, we solidify connections
        decay = 0.95 if tightness > 0.1 else 0.99

        for i, master_idx in enumerate(active_idxs):
            node_key = self.node_order[master_idx]
            node = self.nodes[node_key]
            for val, idx in zip(top_vals[i], top_indices[i]):
                if val <= active_threshold: continue
                target_key = self.node_order[idx.item()]
                if node_key == target_key: continue

                if target_key not in node.connections: new_edges += 1
                curr_w = node.connections.get(target_key, 0.0)
                node.connections[target_key] = (curr_w * decay) + (stage_plasticity * val.item())

        self.version += 1
        return new_edges

    def execute_topological_incineration(self, node_id):
        """Hard reset: Erase node and its influence completely."""
        node_key = str(node_id)
        if node_key in self.nodes:
            idx = self.node_order.index(node_key)
            with torch.no_grad():
                self.master_embeddings[idx].zero_()
                self.master_alignments[idx].fill_(0.0)
                self.tombstone_mask[idx] = 1.0
            self.nodes[node_key].connections = {}
            self.nodes[node_key].is_tombstoned = False
            self.version += 1
            self.update_local_centroid(node_id)

    def execute_manifold_tombstone(self, node_id, r_tomb):
        """Soft reset: Displace node and mask its variance contribution."""
        node_key = str(node_id)
        if node_key in self.nodes:
            idx = self.node_order.index(node_key)
            node = self.nodes[node_key]
            node.tombstone_key = r_tomb.to(self.device)
            with torch.no_grad():
                displaced = torch.matmul(self.master_embeddings[idx], node.tombstone_key)
                self.master_embeddings[idx].copy_(displaced)
                self.tombstone_mask[idx] = 0.0  # Suppression from Pressure calculation
            node.is_tombstoned = True
            self.version += 1

    def restore_from_tombstone(self, node_id):
        """Reverse rotation and restore presence to the manifold."""
        node_key = str(node_id)
        if node_key in self.nodes:
            idx = self.node_order.index(node_key)
            node = self.nodes[node_key]
            if not node.is_tombstoned or node.tombstone_key is None: return
            with torch.no_grad():
                restored = torch.matmul(self.master_embeddings[idx], node.tombstone_key.t())
                self.master_embeddings[idx].copy_(restored)
                self.tombstone_mask[idx] = 1.0
            node.is_tombstoned = False
            node.tombstone_key = None
            self.version += 1

    def apply_edge_threshold(self, min_weight=0.08):
        """Topological cleanup of weak synaptic associations."""
        for node in self.nodes.values():
            to_remove = [nbr for nbr, w in node.connections.items() if w < min_weight]
            for nbr in to_remove: del node.connections[nbr]
        self.version += 1

    def adaptive_prune(self, tightness, max_tightness=0.15, prune_fraction=0.4, nodes_to_consider=None):
        """Reduces edge density when structural tightness exceeds limits."""
        if tightness <= max_tightness: return
        keys = [str(n) for n in nodes_to_consider] if nodes_to_consider else self.node_order
        for k in keys:
            node = self.nodes.get(k)
            if node and node.connections:
                num = max(1, int(len(node.connections) * prune_fraction))
                weak = sorted(node.connections.items(), key=lambda x: x[1])[:num]
                for nbr, _ in weak: node.connections.pop(nbr, None)
        self.version += 1

    def compute_manifold_variance(self):
        """Calculates global variance to drive the Pressure penalty."""
        # CRITICAL: We call get_graph_embedding_matrix() to ensure the gradient
        # chain is wet and leads back to master_embeddings.
        matrix = self.get_graph_embedding_matrix()
        if matrix.size(0) <= 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return torch.var(matrix)

    def ensure_stage_initialized(self, stage_idx):
        """Promotes nodes that have achieved alignment maturity."""
        for i, node_key in enumerate(self.node_order):
            node = self.nodes[node_key]
            if node.stage_idx >= stage_idx or node.is_tombstoned: continue
            if self.master_alignments[i] >= self.promotion_threshold:
                node.stage_idx = stage_idx

    def create_node_from_trace(self, trace, label=None):
        """Directly seeds a new node from a latent vector (Teen/Adult stage growth)."""
        new_id = len(self.node_order)
        self.add_node(new_id, label=label)
        if isinstance(trace, torch.Tensor):
            with torch.no_grad():
                source_vec = trace.view(-1)[:self.embedding_dim].to(self.device)
                self.master_embeddings[new_id].copy_(source_vec)
        return new_id