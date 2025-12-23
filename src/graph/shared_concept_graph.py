# src/graph/shared_concept_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptNode:
    def __init__(self, node_id, embedding_dim=128):
        self.node_id = node_id
        # Weights represent the structural material of the concept
        self.embedding = nn.Parameter(torch.randn(embedding_dim))
        self.alignment_score = 1.0  # Plasticity multiplier (0.0 = Cauterized)
        self.connections = {}  # {neighbor_node_id: weight}

        # NULL-SPACE DYNAMICS
        self.is_tombstoned = False
        self.tombstone_key = None  # R_tomb storage for recovery
        self.gradient_mask = 1.0  # Scalar for ablative freezing


class SharedConceptGraph(nn.Module):
    def __init__(self, embedding_dim=128, lambda_ewma=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nodes = {}
        self.node_order = []
        self.anchor_tensor = {}  # The 'Z' Map (Topological Anchors)
        self.lambda_ewma = lambda_ewma  # EWMA Decay for temporal drift
        self.quarantined_centroids = []  # List of addresses flagged by Auditor

    # --- TOPOLOGICAL ANCHORING (EWMA) ---

    def update_local_centroid(self, node_id):
        """
        Implements EWMA Centroid Anchoring: Z_t+1 = (1 - λ)Z_t + λ(C_v_new).
        This handles the temporal drift of conceptual meaning.
        """
        node = self.nodes[node_id]
        neighbors = list(node.connections.keys())

        if not neighbors:
            c_v_new = node.embedding.detach().clone()
        else:
            neighbor_embs = torch.stack([self.nodes[nid].embedding.detach() for nid in neighbors])
            c_v_new = neighbor_embs.mean(dim=0)

        # Get existing anchor or initialize with current
        z_t = self.anchor_tensor.get(node_id, c_v_new)

        # Apply EWMA update
        self.anchor_tensor[node_id] = (1.0 - self.lambda_ewma) * z_t + self.lambda_ewma * c_v_new

    def is_node_in_centroid_range(self, node_id, centroid_address, radius=0.7):
        """
        AUDITOR UTILITY: Checks if node is within a conceptual radius of a breach.
        Uses Euclidean distance in Z-space.
        """
        if node_id not in self.anchor_tensor:
            return False
        z_vec = self.anchor_tensor[node_id].to(centroid_address.device)
        return torch.norm(z_vec - centroid_address) < radius

    def get_graph_embedding_matrix(self):
        """
        Returns a stacked tensor of all embeddings for stage processing.
        Applies alignment_score and filters tombstoned nodes.
        """
        embs = []
        for nid in self.node_order:
            node = self.nodes[nid]
            if node.is_tombstoned:
                # Neutral representation to maintain sequence indices
                embs.append(torch.zeros(self.embedding_dim, device=node.embedding.device))
            else:
                embs.append(node.embedding * node.alignment_score)

        return torch.stack(embs) if embs else torch.zeros((1, self.embedding_dim))

    # --- REMEDIATION: ABLATIVE ZEROING ---

    def execute_topological_incineration(self, node_id):
        """
        PERMANENT: Ablative Zeroing + Gradient Masking.
        Prevents high-entropy noise poisoning by creating a logical void.
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            with torch.no_grad():
                node.embedding.zero_()  # Absolute ablation

            node.connections = {}
            node.alignment_score = 0.0  # Scar tissue (inhibits Hebbian growth)
            node.gradient_mask = 0.0  # Freezes weight at zero permanently
            node.is_tombstoned = False
            node.tombstone_key = None
            self.update_local_centroid(node_id)

    # --- REMEDIATION: NULL-SPACE ROTATION ---

    def execute_manifold_tombstone(self, node_id, r_tomb):
        """
        REVERSIBLE: Null-Space Displacement.
        Rotates the vector into a non-addressable subspace via R_tomb.
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.tombstone_key = r_tomb

            with torch.no_grad():
                displaced_weight = torch.matmul(node.embedding.data, r_tomb.to(node.embedding.device))
                node.embedding.copy_(displaced_weight)

            node.is_tombstoned = True
            node.alignment_score = 0.0  # Prevent relational updates while quarantined
            self.update_local_centroid(node_id)

    def restore_from_tombstone(self, node_id):
        """
        RECOVERY: Applies Inverse Projection (R_tomb^T).
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if not node.is_tombstoned or node.tombstone_key is None:
                return

            # Orthogonal Restoration: R^T
            inverse_r = node.tombstone_key.t()
            with torch.no_grad():
                node.embedding.copy_(torch.matmul(node.embedding.data, inverse_r.to(node.embedding.device)))

            node.is_tombstoned = False
            node.alignment_score = 1.0  # Restore plasticity
            node.tombstone_key = None
            self.update_local_centroid(node_id)

    # --- TOPOLOGICAL CLEANUP (AUDITOR STEP 4) ---

    def delete_centroid_coordinate(self, address):
        """Physically removes coordinate anchors from Z-Map after Incineration."""
        keys_to_del = [k for k, v in self.anchor_tensor.items() if torch.equal(v, address)]
        for k in keys_to_del:
            del self.anchor_tensor[k]

    def mark_centroid_as_quarantined(self, address):
        """Adds a coordinate to the TCR blacklist for Stage 5+ exclusion."""
        self.quarantined_centroids.append(address.detach().clone())

    # --- O(1) MANIFOLD RETRIEVAL (TCR) ---

    def retrieve_manifold_context(self, current_latent, top_k=5):
        """
        Topological Context Reservoir: Radius search in Anchor Tensor Z.
        Excludes tombstoned (quarantined) and incinerated (ablated) nodes.
        """
        query_coord = current_latent.mean(dim=0) if current_latent.dim() > 1 else current_latent

        # Radius search via cosine similarity in Z-space
        scored_anchors = []
        for cid, z_vec in self.anchor_tensor.items():
            # Check if this centroid is globally quarantined
            is_quarantined = any(torch.norm(z_vec - q) < 0.1 for q in self.quarantined_centroids)
            if is_quarantined:
                continue

            sim = F.cosine_similarity(query_coord.unsqueeze(0), z_vec.unsqueeze(0).to(query_coord.device))
            scored_anchors.append((cid, sim.item()))

        # Select top-k manifolds
        scored_anchors.sort(key=lambda x: x[1], reverse=True)

        context_tensors = []
        for cid, sim_score in scored_anchors[:top_k]:
            node = self.nodes[cid]
            # Governance Gate: Skip if confidence is low, scarred, or tombstoned
            if sim_score > 0.7 and node.alignment_score > 0.1 and not node.is_tombstoned:
                context_tensors.append(node.embedding.to(query_coord.device))

        if not context_tensors:
            return torch.zeros((1, self.embedding_dim), device=query_coord.device)

        return torch.stack(context_tensors)

    # --- GOVERNANCE-AWARE HEBBIAN UPDATES ---

    def update_stage_aware_hebbian(self, attention_map, stage_plasticity=1.0):
        """
        Updates relational weights with Scar-Tissue and Null-Space guardrails.
        """
        if len(self.node_order) < 2 or attention_map is None:
            return

        # Normalize attention for graph update
        avg_att = attention_map.mean(dim=0)  # [num_nodes, num_nodes]

        for i, uid_i in enumerate(self.node_order):
            node_i = self.nodes[uid_i]

            # Skip update for Null-Space or Ablated nodes
            if node_i.alignment_score <= 0.0 or node_i.is_tombstoned:
                continue

            for j, uid_j in enumerate(self.node_order):
                if i == j or self.nodes[uid_j].is_tombstoned:
                    continue

                # Hebbian delta modulated by node alignment (trust)
                delta = stage_plasticity * node_i.alignment_score * avg_att[i, j]

                if delta > 0.01:
                    node_i.connections[uid_j] = node_i.connections.get(uid_j, 0.0) + delta

            # Periodic EWMA update
            if torch.rand(1).item() > 0.95:
                self.update_local_centroid(uid_i)