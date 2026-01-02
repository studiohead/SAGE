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
        return next(self.model.parameters()).device

    def seed(self, concept_dictionary):
        print(f"[*] Seeding {len(concept_dictionary)} nodes into the Manifold...")
        for node_id, concept_name in concept_dictionary.items():
            with torch.no_grad():
                vector = self.model.encode(
                    concept_name, convert_to_tensor=True
                ).to(self.device)

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
        node_key = str(node_id)

        if node_key not in self.graph.nodes:
            self.graph.add_node(node_id, label=label)

        if label is not None:
            self.graph.nodes[node_key].label = str(label)

        if vector.norm() > 0:
            vector = vector / vector.norm()

        with torch.no_grad():
            self.graph.nodes[node_key].embedding.copy_(vector.to(self.device))

        self.graph.update_local_centroid(node_id)

    def seed_all(self, semantic_concepts=None, sensory_patterns=None):
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

            if hasattr(self.model, "encode"):
                self.seed(semantic_concepts)
            else:
                print("[*] Converting semantic concepts to tensors for Frontend...")
                target_device = self.device

                for node_id, concept_name in semantic_concepts.items():
                    dummy_pattern = torch.rand(
                        1, EMBED_DIM, device=target_device
                    )
                    vector = self.model.encoder(dummy_pattern).squeeze(0)
                    self._inject_to_node(node_id, vector, label=concept_name)

        # --------------------
        # SENSORY SEEDING
        # --------------------
        if sensory_patterns:
            self.seed_from_sensory_patterns(sensory_patterns)
