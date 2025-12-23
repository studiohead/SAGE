# src/graph/graph_seeder.py
import torch
from tqdm import tqdm
from src.graph.shared_concept_graph import ConceptNode


class SAGEGraphSeeder:
    """
    Utility to initialize the SharedConceptGraph with semantic priors.
    This provides the initial 'Topological Grounding' for Stage 1.
    """

    def __init__(self, graph, embedding_model):
        """
        graph: Your SharedConceptGraph instance
        embedding_model: Can be a SentenceTransformer (for .seed)
                         or a SensoryFrontend (for .seed_from_sensory_patterns).
        """
        self.graph = graph
        self.model = embedding_model

    def seed(self, concept_dictionary):
        """
        Original method using an embedding model with an .encode() method.
        """
        print(f"[*] Seeding {len(concept_dictionary)} nodes into the Manifold...")

        for node_id, concept_name in concept_dictionary.items():
            # 1. Generate semantic vector
            with torch.no_grad():
                vector = self.model.encode(concept_name, convert_to_tensor=True)

            # 2. Inject using the internal helper to handle ConceptNode objects
            self._inject_to_node(node_id, vector)

        print("[+] Manifold Seeding Complete.")

    def seed_from_sensory_patterns(self, pattern_dict):
        """
        New method to ground the manifold using raw sensory tensors
        projected through the Frontend's encoder.
        """
        print(f"[*] Seeding {len(pattern_dict)} nodes via Sensory Projection...")

        # Ensure we are not tracking gradients during seeding
        with torch.no_grad():
            for node_id, raw_pattern in pattern_dict.items():
                # Project raw pattern (e.g., 784 pixels) -> Manifold (128 dims)
                # This assumes self.model is the SensoryFrontend
                vector = self.model.encoder(raw_pattern).squeeze(0)

                # Inject using the internal helper
                self._inject_to_node(node_id, vector)

        print("[+] Manifold Grounding Complete.")

    def _inject_to_node(self, node_id, vector):
        """
        Helper to maintain SAGE-specific ConceptNode logic,
        node ordering, and topological anchoring.
        """
        # Create the Node object if it doesn't exist
        if node_id not in self.graph.nodes:
            new_node = ConceptNode(node_id, embedding_dim=self.graph.embedding_dim)
            self.graph.nodes[node_id] = new_node
            self.graph.node_order.append(node_id)

        # Copy data into the Parameter to ensure it's part of the grad graph later
        with torch.no_grad():
            self.graph.nodes[node_id].embedding.copy_(vector)

        # Initialize the topological anchor (Z-Map) for the Auditor
        self.graph.update_local_centroid(node_id)