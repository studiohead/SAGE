import torch
from tqdm import tqdm
from config.config import SHARED_MODEL_CONFIG
from src.graph.shared_concept_graph import ConceptNode

EMBED_DIM = SHARED_MODEL_CONFIG.get("embed_dim")


class SAGEGraphSeeder:
    """
    Utility to initialize the SharedConceptGraph with semantic priors.
    This provides the initial 'Topological Grounding' for Stage 1.
    """

    def __init__(self, graph, embedding_model):
        self.graph = graph
        self.model = embedding_model

    @property
    def device(self):
        """Surgically detect where the model is living (CPU/MPS/CUDA)."""
        # Note: If model has no parameters, we check the graph's device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.graph.device

    def seed(self, concept_dictionary):
        print(f"[*] Seeding {len(concept_dictionary)} nodes into the Manifold...")
        for node_id, concept_name in concept_dictionary.items():
            with torch.no_grad():
                # Handling different model interfaces (SentenceTransformers vs Custom)
                if hasattr(self.model, "encode"):
                    vector = self.model.encode(
                        concept_name, convert_to_tensor=True
                    ).to(self.device)
                else:
                    # Fallback for models without direct .encode
                    dummy_pattern = torch.rand(1, EMBED_DIM, device=self.device)
                    vector = self.model.encoder(dummy_pattern).squeeze(0)

            self._inject_to_node(node_id, vector, label=concept_name)

        print("[+] Manifold Seeding Complete.")

    def seed_from_sensory_patterns(self, pattern_dict):
        """
        Grounds the manifold using raw sensory tensors projected through
        the Frontend's encoder.
        """
        print(f"[*] Seeding {len(pattern_dict)} nodes via Sensory Projection...")
        target_device = self.device

        with torch.no_grad():
            for node_id, raw_pattern in pattern_dict.items():
                raw_pattern = raw_pattern.to(target_device)
                vector = self.model.encoder(raw_pattern).squeeze(0)
                self._inject_to_node(node_id, vector, label=str(node_id))

        print("[+] Manifold Grounding Complete.")

    def _inject_to_node(self, node_id, vector, label=None):
        """
        FIXED: Writes to the Contiguous Master Block instead of individual node attributes.
        This preserves the graph's ability to optimize for variance.
        """
        node_key = str(node_id)

        # 1. Ensure the node exists in the graph's registry
        if node_key not in self.graph.nodes:
            self.graph.add_node(node_id, label=label)

        if label is not None:
            self.graph.nodes[node_key].label = str(label)

        # 2. Normalize the vector for spherical grounding
        if vector.norm() > 0:
            vector = vector / vector.norm()

        # 3. Locate the node's position in the contiguous master block
        try:
            # We look up the index where this node's parameters live
            node_idx = self.graph.node_order.index(node_key)

            with torch.no_grad():
                # Ensure the vector is moved to the target device and matched to EMBED_DIM
                target_vec = vector.to(self.device).view(-1)
                if target_vec.size(0) != EMBED_DIM:
                    # Resize/Pad if necessary to maintain master block integrity
                    new_vec = torch.zeros(EMBED_DIM, device=self.device)
                    slice_size = min(target_vec.size(0), EMBED_DIM)
                    new_vec[:slice_size] = target_vec[:slice_size]
                    target_vec = new_vec

                # WRITE TO MASTER EMBEDDING BLOCK
                # Replacing old self.graph.nodes[node_key].embedding access
                self.graph.master_embeddings[node_idx].copy_(target_vec)

                # SEEDING BOOST: Ensure seeded nodes start with strong alignment (Radial Depth)
                # This ensures they are visible to the manifold immediately.
                if hasattr(self.graph, 'master_alignments'):
                    self.graph.master_alignments[node_idx].fill_(0.85)

                # Ensure the node is visible (not tombstoned)
                if hasattr(self.graph, 'tombstone_mask'):
                    self.graph.tombstone_mask[node_idx] = 1.0

        except ValueError:
            print(f"[!] Seeder Sync Error: {node_key} exists in dict but not in node_order.")

        # 4. Update the grounding anchor for topological stability
        self.graph.update_local_centroid(node_id)

    def seed_all(self, semantic_concepts=None, sensory_patterns=None):
        """
        Complete Seeding Orchestrator.
        Maintains your specific logic for handling iterable normalization.
        """
        # --------------------
        # SEMANTIC SEEDING
        # --------------------
        if semantic_concepts:
            # Normalize ANY iterable (set, list, tuple, generator) to dict[str, str]
            if not isinstance(semantic_concepts, dict):
                semantic_concepts = {
                    str(i): str(concept)
                    for i, concept in enumerate(semantic_concepts)
                }

            # Your specific branching logic for model capability
            if hasattr(self.model, "encode"):
                self.seed(semantic_concepts)
            else:
                print("[*] Converting semantic concepts to tensors for Frontend...")
                target_device = self.device

                for node_id, concept_name in semantic_concepts.items():
                    # Your dummy pattern logic preserved exactly
                    dummy_pattern = torch.rand(
                        1, EMBED_DIM, device=target_device
                    )
                    with torch.no_grad():
                        vector = self.model.encoder(dummy_pattern).squeeze(0)
                    self._inject_to_node(node_id, vector, label=concept_name)

        # --------------------
        # SENSORY SEEDING
        # --------------------
        if sensory_patterns:
            self.seed_from_sensory_patterns(sensory_patterns)