import torch


class GrowthHormone:
    """
    Controller for Manifold Expansion (SAGE / Citizen / Elder).
    Regulates the birth of new nodes based on structural 'Pressure'.
    """

    def __init__(self, floor: float = 0.99, ceiling: float = 1.00, winner_indices: object = None) -> None:
        self.confidence_floor = floor
        print(self.confidence_floor)
        self.confidence_ceiling = ceiling
        self.winner_indices = winner_indices
        self.batch_created_labels = set()  # Reset per batch to prevent explosion

    def preliminary_gate(self, telemetry):
        """Checks if the model is in the 'Learning Zone' (not mastered, not clueless)."""
        conf = telemetry.get("confidence", 1.0)
        return self.confidence_floor < conf < self.confidence_ceiling

    def has_potential_growth(self, telemetry):
        """
        Determines if the structural divergence justifies a new node.
        Target Gamma: > 0.05 (Structural Tension).
        """
        gamma = telemetry.get("gamma_divergence", 0.0)
        return 0.70 < gamma < 0.5

    def maybe_create_node(self, telemetry, graph):
        base_label = telemetry.get("text", "node")
        conf = float(telemetry.get("confidence", 0.0))
        pressure = telemetry.get("latent_pressure", 0.0)
        gamma = telemetry.get("gamma_divergence", 0.0)
        is_compressed = pressure > 0.10

        # 1. CEILING CHECK
        if conf >= self.confidence_ceiling:
            # If this prints, your ceiling is too low
            # print(f"DEBUG: Ceiling Blocked (Conf {conf} >= Ceil {self.confidence_ceiling})")
            return None

        # 2. THE GROWTH GATE
        # This should be TRUE if Conf is 0.9881 and Floor is 0.99
        if conf < self.confidence_floor:

            # 3. LABEL CHECK
            if base_label in self.batch_created_labels:
                # This is likely why growth is 'silent'
                # print(f"DEBUG: Label '{base_label}' already created in this batch.")
                return None

            # 4. DIVERGENCE CHECK
            if not (self.has_potential_growth(telemetry) or is_compressed):
                # If your gamma is low and pressure isn't registering here, growth dies.
                # print(f"DEBUG: No Divergence/Pressure (Gamma: {gamma:.4f})")
                return None

            # 5. EXECUTE
            new_node_id = graph.create_node_from_trace(telemetry.get("trace"), label=base_label)
            if new_node_id is not None:
                self.batch_created_labels.add(base_label)
                print(f"[+] BIRTH SUCCESS: Node {new_node_id}: {base_label}")
                return new_node_id
            else:
                print("DEBUG: graph.create_node_from_trace returned None (Capacity?)")

        return None

    def reset_batch(self):
        """Called at the end of every training step to clear the label lockout."""
        self.batch_created_labels.clear()