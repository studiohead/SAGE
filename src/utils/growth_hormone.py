import torch


class GrowthHormone:
    """
    Controller for Manifold Expansion (SAGE / Citizen/Elder).
    Birthed nodes are anchored via semantic labels to maintain a shared concept map.

    Updated: Implements Novelty Bypass to prevent 'Mastery Stagnation'
    where high confidence scores block the creation of new conceptual anchors.
    """

    def __init__(self, floor=0.55, ceiling=0.85):
        """
        Args:
            floor: The minimum confidence required to trust a sensory trace (Learning Zone Start).
            ceiling: The maturity gate where a node is considered 'Mastered' (Learning Zone End).
        """
        self.confidence_floor = floor
        self.confidence_ceiling = ceiling

    def preliminary_gate(self, telemetry):
        """
        Standard Check: Evaluates if the batch falls within the 'Learning Zone'.
        We look for the 'Sweet Spot' between noise and mastery.
        """
        conf = telemetry.get("confidence", 1.0)
        return self.confidence_floor < conf < self.confidence_ceiling

    def has_potential_growth(self, telemetry):
        """
        Deep Evaluation: Analyzes Gamma Divergence.
        A divergence > 0.01 indicates that even if the category is known,
        the current manifold shape is failing to represent this specific trace.
        """
        return telemetry.get("gamma_divergence", 0.0) > 0.01

    def maybe_create_node(self, telemetry, graph):
        """
        Surgical entry point for node birth.
        Logic Flow:
        1. If the Label is brand new -> GROW (Novelty Bypass).
        2. If the Label exists but the model is 'Uncertain' and 'Divergent' -> GROW (Mathematical Expansion).
        """
        # 1. Data Extraction
        trace_vec = telemetry.get("trace")
        label_text = telemetry.get("text")

        # --- COLLISION GUARD 1: SEMANTIC LABEL ---
        # Check if the label already exists in the graph's active nodes
        existing_labels = {node.label for node in graph.nodes.values() if node.label is not None}

        if label_text in existing_labels:
            # LOGGING: Optional, but helpful to see why growth was aborted
            # print(f"[-] Growth Aborted: Label '{label_text}' already exists.")
            return None

        conf = telemetry.get("confidence", 0.0)

        if trace_vec is None:
            return None

        # 2. NOVELTY BYPASS CHECK
        # We scan the graph to see if we've ever birthed a node for this specific label.
        # This is the primary fix for your 0.97 confidence stagnation.
        existing_labels = {node.label for node in graph.nodes.values() if node.label is not None}
        is_novel = label_text not in existing_labels if label_text else False

        # 3. CONVERGENCE CHECK (Standard SAGE Growth)
        # Only check the Gates if we aren't already bypassing via Novelty.
        in_learning_zone = self.preliminary_gate(telemetry)
        is_divergent = self.has_potential_growth(telemetry)

        # 4. DECISION ENGINE
        should_grow = is_novel or (in_learning_zone and is_divergent)

        if should_grow:
            # Create node in the Graph Manifold via the 256-dim trace
            # graph.create_node_from_trace handles index allocation and ModuleDict registration
            new_node_id = graph.create_node_from_trace(
                trace=trace_vec,
                label=label_text
            )

            if new_node_id is not None:
                # Logging to verify which trigger was used
                trigger_type = "NOVELTY" if is_novel else "DIVERGENCE"
                print(
                    f"[+] Growth Hormone triggered [{trigger_type}]: New node {new_node_id} (Label: {label_text}) | Conf: {conf:.4f}")

            return new_node_id

        return None