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

    def get_active_gradient_masks(self, num_active):
        """
        Fetches the Anchor Z Shield values.
        Handles 'Ghost Nodes' resulting from Incineration or incomplete Growth.
        """
        masks = []

        # We iterate based on the node_order to ensure spatial alignment
        # with the master_embeddings tensor.
        for node_key in self.node_order[:num_active]:

            # 1. EXISTENCE CHECK (The KeyError Guard)
            if node_key in self.nodes:
                node = self.nodes[node_key]

                # 2. TOMBSTONE CHECK
                # If a node is tombstoned, we effectively kill its gradient shield
                if getattr(node, "is_tombstoned", False):
                    masks.append(0.0)
                else:
                    # 3. GRADIENT SHIELD RETRIEVAL
                    mask_val = getattr(node, 'gradient_mask', 1.0)
                    masks.append(mask_val)
            else:
                # 4. GHOST NODE HANDLING
                # The node is in node_order but missing from the dictionary.
                # We return 0.0 to ensure the tensor shape is preserved
                # without allowing the ghost to influence the manifold.
                masks.append(0.0)

        # Return as a [N, 1] tensor for broadcasting
        return torch.tensor(masks, device=self.device, dtype=torch.float32).unsqueeze(1)

    def get_graph_embedding_matrix(self):
        """
        RESTORATION: Differentiable manifold with RAW-EMBEDDING pressure.
        This allows the Hinge Loss to push nodes apart in Euclidean space
        before they are projected for the transformer stages.
        """
        num_active = len(self.node_order)
        if num_active == 0:
            return torch.zeros((0, self.embedding_dim), device=self.device)

        if self.training or self._cached_matrix is None or self.version != self._last_version:
            # RAW EMBEDDINGS: The source of truth for the Pressure Engine
            embs = self.master_embeddings[:num_active]
            g_masks = self.get_active_gradient_masks(num_active)

            # 1. DYNAMIC MAGNITUDE (The 'Steam' for expansion)
            # We increase the max clamp to 5.0 to allow the manifold to physically 'bloat'
            # which drops confidence and triggers growth.
            aligns = torch.clamp(self.master_alignments[:num_active], 0.05, 5.0)
            t_masks = self.tombstone_mask[:num_active]

            # 2. THE SAGE SWITCH:
            # During training, we do NOT normalize. This lets the Hinge-Loss
            # drive the master_embeddings apart by their raw values.
            if self.training:
                # Assembly without the unit-sphere constraint
                res = (embs * aligns) * t_masks * g_masks
            else:
                # During inference/eval, we project to the sphere for stability
                norm_embs = F.normalize(embs, p=2, dim=1, eps=1e-8)
                res = (norm_embs * aligns) * t_masks * g_masks

            if not self.training:
                self._cached_matrix = res
                self._last_version = self.version
            return res

        return self._cached_matrix

    def retrieve_manifold_context(self, current_latent, top_k=5):
        """
        Optimized 256-dim Vectorized Retrieval.
        Compatible with torch.nn.ModuleDict (No .get() method).
        """
        if len(self.node_order) == 0:
            return []

        current_device = current_latent.device
        query = F.normalize(
            current_latent.mean(dim=0) if current_latent.dim() > 1 else current_latent,
            p=2, dim=0
        )

        search_list = []
        valid_keys = []

        for node_key in self.node_order:
            # ModuleDict check: replace .get() with 'in'
            if node_key not in self.nodes:
                # This is an INCINERATED node (missing from ModuleDict)
                continue

            node = self.nodes[node_key]

            # 1. Skip if Tombstoned (Inactive but present)
            if getattr(node, "is_tombstoned", False):
                continue

            # 2. Get the embedding (Priority: Anchor -> Master)
            emb = self.anchor_tensor.get(int(node_key), None)
            if emb is None:
                try:
                    idx = self.node_order.index(node_key)
                    emb = self.master_embeddings[idx]
                except ValueError:
                    continue  # Safety check if index is missing

            search_list.append(emb)
            valid_keys.append(node_key)

        if not search_list:
            return []

        # 3. Build the manifold matrix [N x 256]
        manifold_tensor = torch.stack(search_list).to(current_device)
        manifold_tensor = F.normalize(manifold_tensor, p=2, dim=1)

        # 4. Dimension Check (The 256-dim guardrail)
        if query.shape[-1] != manifold_tensor.shape[-1]:
            raise ValueError(f"Dim Mismatch: Prompt={query.shape[-1]}d, Graph={manifold_tensor.shape[-1]}d")

        # 5. Parallel Similarity
        similarities = torch.matmul(manifold_tensor, query)

        # 6. Extract Top-K Results
        k = min(top_k, len(valid_keys))
        top_scores, top_indices = torch.topk(similarities, k)

        context_data = []
        for idx in top_indices:
            node_key = valid_keys[idx.item()]
            orig_idx = self.node_order.index(node_key)
            emb_out = F.normalize(self.master_embeddings[orig_idx], p=2, dim=0)
            context_data.append((emb_out, node_key))

        return context_data

    def update_stage_aware_hebbian(
            self, stage_key, tightness, attention_map=None, batch_indices=None,
            stage_plasticity=1.0, threshold=None, top_k=250
    ):
        """
        Hebbian wiring: strengthens or creates edges; Teen stage prioritized for growth.
        """

        if not self.node_order:
            return 0

        active_threshold = threshold if threshold is not None else self.promotion_threshold

        # Empty graph: boost initial wiring
        if sum(len(n.connections) for n in self.nodes.values()) == 0:
            active_threshold = 0.05

        # Determine active nodes
        active_idxs = (
            list(range(len(self.node_order))) if batch_indices is None
            else [self.node_order.index(str(nid)) for nid in batch_indices if str(nid) in self.nodes]
        )

        if not active_idxs:
            return 0

        active_embs = F.normalize(self.master_embeddings[active_idxs].detach(), p=2, dim=1)
        full_manifold = self.get_graph_embedding_matrix().detach()
        cross_sim = torch.mm(active_embs, full_manifold.t())

        top_vals, top_indices = torch.topk(cross_sim, k=min(top_k, full_manifold.size(0)), dim=1)

        new_edges = 0

        for i, master_idx in enumerate(active_idxs):
            node_key = self.node_order[master_idx]
            node = self.nodes[node_key]

            if stage_key == "Teen":
                node_decay = 0.99
            elif stage_key == "Adult":
                node_decay = 0.97
            else:
                node_decay = 0.95

            for val, idx in zip(top_vals[i], top_indices[i]):
                target_key = self.node_order[idx.item()]
                if node_key == target_key or val <= active_threshold:
                    continue

                # Apply attention if given
                if attention_map is not None:
                    val = val * attention_map[i, idx]

                # Hebbian update
                prev_w = node.connections.get(target_key, 0.0)
                node.connections[target_key] = (prev_w * node_decay) + (stage_plasticity * val)

                # Symmetric update
                target_node = self.nodes[target_key]
                rev_prev_w = target_node.connections.get(node_key, 0.0)
                target_node.connections[node_key] = (rev_prev_w * node_decay) + (stage_plasticity * val)

                # Count new edges
                if prev_w == 0.0:
                    new_edges += 1

        self.version += 1
        return new_edges

    def execute_topological_incineration(self, node_id):
        """
        Hard reset: Erases the node's soul, metadata, and manifold presence.
        Ensures no ghost keys remain in node_order.
        """
        node_key = str(node_id)
        if node_key in self.nodes:
            # 1. LOCATE HARDWARE INDEX
            try:
                idx = self.node_order.index(node_key)
            except ValueError:
                # Safety: If it's in nodes but not in order, the graph is already drifted.
                del self.nodes[node_key]
                return

            with torch.no_grad():
                # 2. WIPE PHYSICAL PARAMETERS
                # We zero these so that if the index is reused, it starts from a clean slate.
                self.master_embeddings[idx].zero_()
                self.master_alignments[idx].fill_(0.0)

                # 3. FLIP THE GUARDS
                # active_mask must be 0.0 so the GrowthHormone knows this slot is 'empty'
                self.active_mask[idx] = 0.0
                self.tombstone_mask[idx] = 1.0  # Reset to neutral

            # 4. SURGICAL REMOVAL OF THE SOUL
            # Delete from the ModuleDict to prevent KeyError in get_active_gradient_masks
            del self.nodes[node_key]

            # Remove from the topological list
            self.node_order.remove(node_key)

            # 5. ANCHOR DESTRUCTION
            if node_id in self.anchor_tensor:
                del self.anchor_tensor[node_id]

            self.version += 1
            # No need to update local centroid of a dead node,
            # but neighbors should eventually re-sync.

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

    def apply_edge_threshold(self, min_weight=0.20):
        """
        Topological Cleanup: Removes weak synaptic associations.
        This releases 'Topological Tension' and creates voids for new growth.
        """
        edges_removed = 0

        # We iterate over the nodes in the ModuleDict
        for node_key, node in self.nodes.items():
            # Identify connections that have withered below the threshold
            # node.connections is a dict: {neighbor_id: weight}
            to_remove = [
                neighbor_id for neighbor_id, weight in node.connections.items()
                if weight < min_weight
            ]

            for neighbor_id in to_remove:
                # Delete the edge from the dictionary
                del node.connections[neighbor_id]
                edges_removed += 1

        self.version += 1
        print(f"[!] Synaptic Cleanup: {edges_removed} weak edges incinerated (Threshold: {min_weight}).")
        return edges_removed

    def adaptive_prune(self, tightness, max_tightness=0.15, prune_fraction=0.4, nodes_to_consider=None,
                       min_connections=2):
        """
        Reduces edge density when structural tightness exceeds limits.
        Only prunes nodes with at least `min_connections` edges.
        """
        if tightness <= max_tightness:
            return

        # Use membership check instead of .get() for ModuleDict compatibility
        keys = [str(n) for n in nodes_to_consider] if nodes_to_consider else self.node_order

        # Foundational Concepts to protect
        protected_labels = {str(i) for i in range(400)}

        for k in keys:
            if k not in self.nodes:
                continue

            node = self.nodes[k]

            # PROTECT: Never prune edges belonging to the Respected Ten
            if str(node.label) in protected_labels:
                continue

            # Skip nodes with too few connections
            if node.connections and len(node.connections) >= min_connections:
                # Prune only the weakest edges
                num_to_prune = int(len(node.connections) * prune_fraction)
                if num_to_prune < 1:
                    continue  # Skip if computed prune count < 1

                weak = sorted(node.connections.items(), key=lambda x: x[1])[:num_to_prune]

                for nbr, _ in weak:
                    # Ensure the neighbor isn't a protected node before popping
                    node.connections.pop(nbr, None)

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

    def create_node_from_trace(self, trace, label=None, initial_attachment_k=3):
        """Surgical birth with immediate synaptic tethering and capacity guard."""

        # --- THE GROWTH CAP (Original Guard) ---
        new_id = len(self.node_order)
        if new_id >= self.max_nodes:
            print(f"[!] Graph at capacity ({self.max_nodes}). Growth aborted.")
            return None

        # 1. Standard registration
        self.add_node(new_id, label=label)

        if isinstance(trace, torch.Tensor):
            with torch.no_grad():
                # Standardize the latent trace
                source_vec = trace.detach().view(-1)[:self.embedding_dim].to(self.device)
                source_vec = F.normalize(source_vec, p=2, dim=0)

                # Write to physical manifold
                self.master_embeddings[new_id].copy_(source_vec)

                # 2. IMMEDIATE SYNC (Prevents retrieval invisibility)
                self.anchor_tensor[new_id] = source_vec.clone()

                # 3. INITIAL TETHERING
                # We must tether here so the first EWMA update has context.
                if new_id > 0:
                    # Use detach to avoid bleeding gradients during birth
                    full_matrix = self.get_graph_embedding_matrix().detach()

                    # Compare against current manifold (excluding the node being born)
                    sims = torch.matmul(full_matrix, source_vec)

                    # Top-K closest existing concepts
                    vals, idxs = torch.topk(sims[:-1], min(initial_attachment_k, new_id))

                    new_node = self.nodes[str(new_id)]
                    for v, i in zip(vals, idxs):
                        target_key = self.node_order[i.item()]
                        # Immediate bidirectional synaptic seeding
                        new_node.connections[target_key] = v.item()
                        self.nodes[target_key].connections[str(new_id)] = v.item()

        self.version += 1
        return new_id

    # Inside SharedConceptGraph (src/graph/shared_concept_graph.py)

    def apply_governed_gradient_update(self, lr):
        """
        [0015] THE SUTURE:
        Stage-Aware Inertial Mass.
        Teen Stage: We allow magnitude expansion to break compression.
        """
        if self.master_embeddings.grad is None:
            return

        protected_labels = {str(i) for i in range(400)}

        with torch.no_grad():
            for idx, node_id in enumerate(self.node_order):
                node = self.nodes[str(node_id)]
                grad = self.master_embeddings.grad[idx]

                if grad is None:
                    continue

                # 1. VISCOSITY CHECK
                # For Teen expansion, we need mobility. 0.1 is good,
                # but we can go to 0.5 if the manifold is still stubborn.
                viscosity = 0.5 if node.label in protected_labels else 1.0

                # 2. APPLY REGULATED UPDATE
                update_vector = grad * lr * viscosity
                self.master_embeddings[idx] -= update_vector

                # 3. ANCHOR Z SYNC
                if node_id in self.anchor_tensor:
                    # We sync the raw, expanded vector to the anchor
                    self.anchor_tensor[node_id] = self.master_embeddings[idx].clone()

                # 4. CONDITIONAL RE-NORMALIZATION [CRITICAL FIX]
                # During Teen growth, we STOP normalizing here.
                # Let the magnitude grow. 'get_graph_embedding_matrix'
                # handles the projection. Only normalize in 'Adult' or 'Elder'.
                # If we normalize here, we kill the 'Expansion Steam'.
                pass

        if self.master_embeddings.grad is not None:
            self.master_embeddings.grad.zero_()