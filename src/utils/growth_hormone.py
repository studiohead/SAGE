import torch


class GrowthHormone:
    """
    Controller for Manifold Expansion.
    Birthed nodes are anchored via semantic labels to maintain a shared concept map.
    """

    def __init__(self, floor=0.1, ceiling=0.7):
        self.confidence_floor = floor
        self.confidence_ceiling = ceiling

    def preliminary_gate(self, telemetry):
        """
        Cheap check: Growth occurs in the 'Learning Zone'.
        High confidence implies mastery; low confidence implies noise.
        """
        conf = telemetry.get("confidence", 1.0)
        return self.confidence_floor < conf < self.confidence_ceiling

    def has_potential_growth(self, telemetry):
        """
        Deep evaluation: High gamma divergence suggests the existing manifold
        cannot represent the incoming sensory trace.
        """
        # Threshold: Divergence > 0.01 indicates a conceptual gap
        return telemetry.get("gamma_divergence", 0.0) > 0.01

    def maybe_create_node(self, telemetry, graph):
        """
        Surgical entry point for node birth.
        Connects the mathematical trace to the human-readable label.
        """
        if self.preliminary_gate(telemetry):
            if self.has_potential_growth(telemetry):
                # 1. Extract the latent trace (tensor)
                trace_vec = telemetry.get("trace")

                # 2. Extract the semantic anchor (string)
                # This ensures the new node is compatible with BROAD_CONCEPTS
                label_text = telemetry.get("text")

                # Validation: Ensure we have a valid trace before modifying the graph
                if trace_vec is None:
                    return None

                # 3. Create node in the Graph Manifold
                new_node_id = graph.create_node_from_trace(
                    trace=trace_vec,
                    label=label_text
                )

                if new_node_id is not None:
                    print(f"[+] Growth Hormone triggered: New node {new_node_id} (Label: {label_text})")

                return new_node_id

        return None