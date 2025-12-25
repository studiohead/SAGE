import torch
from tqdm import tqdm
from src.graph.shared_concept_graph import ConceptNode

# Example list of 100 broad semantic concepts
BROAD_CONCEPTS = [
    "person", "animal", "plant", "vehicle", "building", "food", "tool", "furniture",
    "clothing", "sport", "music", "emotion", "color", "weather", "nature", "technology",
    "transport", "education", "health", "medicine", "science", "art", "literature", "language",
    "history", "geography", "mathematics", "physics", "chemistry", "biology", "economics",
    "politics", "law", "philosophy", "religion", "mythology", "storytelling", "game", "toy",
    "instrument", "dance", "festival", "holiday", "celebration", "social", "family", "friendship",
    "communication", "media", "entertainment", "internet", "computer", "robot", "energy", "environment",
    "space", "time", "emotion", "memory", "dream", "fear", "love", "anger", "happiness", "sadness",
    "curiosity", "knowledge", "skill", "job", "career", "finance", "market", "trade", "industry",
    "agriculture", "transportation", "infrastructure", "city", "village", "ocean", "river", "mountain",
    "forest", "desert", "lake", "animal behavior", "human behavior", "cognition", "perception",
    "decision making", "ethics", "morality", "strategy", "problem solving", "technology use",
    "innovation", "creativity", "design", "architecture", "engineering", "mathematics concept"
]

# src/graph/graph_seeder.py

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

        with torch.no_grad():
            for node_id, raw_pattern in pattern_dict.items():
                # Project raw pattern (e.g., 784 pixels) -> Manifold (128 dims)
                vector = self.model.encoder(raw_pattern).squeeze(0)
                self._inject_to_node(node_id, vector)

        print("[+] Manifold Grounding Complete.")

    def _inject_to_node(self, node_id, vector):
        """
        Helper to maintain SAGE-specific ConceptNode logic,
        node ordering, and topological anchoring.
        """
        if node_id not in self.graph.nodes:
            new_node = ConceptNode(node_id, embedding_dim=self.graph.embedding_dim)
            self.graph.nodes[node_id] = new_node
            self.graph.node_order.append(node_id)

        with torch.no_grad():
            self.graph.nodes[node_id].embedding.copy_(vector)

        self.graph.update_local_centroid(node_id)

    def seed_all(self, semantic_concepts=None, sensory_patterns=None):
        """
        Seed both semantic concepts and sensory patterns if provided.
        Determines automatically which method to use based on model type.
        """
        if semantic_concepts:
            if hasattr(self.model, "encode"):
                self.seed(semantic_concepts)
            else:
                # Convert concept names to dummy tensors if frontend only
                print("[*] Converting semantic concepts to tensors for SensoryFrontend...")
                tensor_dict = {i: torch.rand(1, 784) for i in range(len(semantic_concepts))}
                self.seed_from_sensory_patterns(tensor_dict)

        if sensory_patterns:
            self.seed_from_sensory_patterns(sensory_patterns)
