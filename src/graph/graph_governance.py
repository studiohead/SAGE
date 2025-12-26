# src/graph/graph_governance.py
import os
import torch
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphGovernance")


class GraphGovernance:
    """
    Manages the persistence and integrity of the SharedConceptGraph.
    Uses atomic 'Save-and-Swap' to ensure state integrity without
    relying on fragile filesystem locks (fcntl).
    """

    def __init__(self, graph_path: str = "checkpoints/global_manifold.pth"):
        self.graph_path = graph_path
        # Removed self.lock_path to prevent 'lock file' creation

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)

    def secure_save(self, graph: torch.nn.Module, metadata: Optional[dict] = None):
        try:
            # 1. Grab the PyTorch state (likely empty, but good for buffers)
            graph_state = graph.state_dict()

            # 2. MANUALLY grab the nodes (The missing 256-dim data)
            nodes_data = {
                node_id: {
                    'embedding': node.embedding.data.cpu(),
                    'alignment_score': node.alignment_score,
                    'connections': node.connections
                } for node_id, node in graph.nodes.items()
            }

            # Your existing integrity check
            for node_id, data in nodes_data.items():
                if torch.isnan(data['embedding']).any():
                    logger.error(f"[Governance] FATAL: Node {node_id} contains NaN.")
                    return False

            temp_path = f"{self.graph_path}.tmp"

            # 3. Add 'nodes_data' to the save dictionary
            state_to_save = {
                "graph_state": graph_state,
                "nodes_data": nodes_data,  # <--- This makes Toddler work
                "node_order": getattr(graph, "node_order", []),
                "anchor_tensor": getattr(graph, "anchor_tensor", {}),
                "metadata": metadata or {}
            }

            torch.save(state_to_save, temp_path)
            os.replace(temp_path, self.graph_path)

            logger.info(f"[Governance] Manifold state secured at {self.graph_path}")
            return True
        except Exception as e:
            logger.error(f"[Governance] Critical Save Failure: {e}")
            return False

    def secure_load(self, graph: torch.nn.Module):
        if not os.path.exists(self.graph_path):
            return False
        try:
            device = next(graph.parameters()).device if list(graph.parameters()) else "cpu"
            checkpoint = torch.load(self.graph_path, map_location=device)

            # 1. RECONSTRUCT the ConceptNode objects first
            nodes_data = checkpoint.get("nodes_data", {})
            from src.graph.shared_concept_graph import ConceptNode

            for node_id, data in nodes_data.items():
                node = ConceptNode(node_id, embedding_dim=graph.embedding_dim)
                node.embedding.data.copy_(data['embedding'])
                node.alignment_score = data['alignment_score']
                node.connections = data['connections']
                graph.nodes[node_id] = node

            # 2. Restore Order and Anchors
            graph.node_order = checkpoint.get("node_order", [])
            graph.anchor_tensor = checkpoint.get("anchor_tensor", {})

            # 3. Final State Dict check (for any other Module properties)
            graph.load_state_dict(checkpoint.get("graph_state", {}), strict=False)

            logger.info(f"[Governance] Manifold RESTORED: {len(graph.nodes)} nodes identified.")
            return True
        except Exception as e:
            logger.error(f"Load Failure: {e}")
            return False