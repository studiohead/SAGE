# src/utils/growth_hormone.py
class GrowthHormone:
    def __init__(self, floor=0.1, ceiling=0.7):
        self.confidence_floor = floor
        self.confidence_ceiling = ceiling

    def preliminary_gate(self, telemetry):
        """Cheap check: confidence in range."""
        conf = telemetry.get("confidence", 1.0)
        return self.confidence_floor < conf < self.confidence_ceiling

    def has_potential_growth(self, telemetry):
        """Deep evaluation for node creation."""
        # Edge stats, trace analysis, etc.
        # Returns True if node should be created
        return telemetry.get("gamma_divergence", 0.0) > 0.01

    def maybe_create_node(self, telemetry, graph):
        """Trigger node creation if criteria met."""
        if self.preliminary_gate(telemetry):
            if self.has_potential_growth(telemetry):
                new_node_id = graph.create_node_from_trace(telemetry.get("trace"))
                print(f"[+] Growth Hormone triggered: New node {new_node_id}")
                return new_node_id
        return None
