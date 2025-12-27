import os
import torch
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphGovernance")


class GraphGovernance:
    """
    Manages the persistence and integrity of the SharedConceptGraph.
    Uses atomic 'Save-and-Swap' and ensures MPS/CPU device compatibility.
    """

    def __init__(self, graph_path: str = "checkpoints/global_manifold.pth"):
        self.graph_path = graph_path
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)

    def secure_save(self, graph: torch.nn.Module, metadata: Optional[dict] = None):
        try:
            # 1. Capture the structural data
            # .cpu() is vital here so the file isn't hardware-locked to MPS
            nodes_data = {
                node_id: {
                    'embedding': node.embedding.data.cpu(),
                    'alignment_score': node.alignment_score,
                    'connections': node.connections,
                    'stage_idx': getattr(node, 'stage_idx', -1),
                    'is_tombstoned': node.is_tombstoned
                } for node_id, node in graph.nodes.items()
            }

            # Integrity Guard: Check for NaN before overwriting good data
            for node_id, data in nodes_data.items():
                if torch.isnan(data['embedding']).any():
                    logger.error(f"[Governance] FATAL: Node {node_id} contains NaN. Save aborted.")
                    return False

            temp_path = f"{self.graph_path}.tmp"

            state_to_save = {
                "graph_state": graph.state_dict(),
                "nodes_data": nodes_data,
                "node_order": getattr(graph, "node_order", []),
                # Anchors are moved to CPU for storage
                "anchor_tensor": {k: v.cpu() for k, v in getattr(graph, "anchor_tensor", {}).items()},
                "metadata": metadata or {}
            }

            # Atomic Swap: Write to temp, then rename
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
            # Detect target device (MPS for your Mac)
            device = next(graph.parameters()).device if list(graph.parameters()) else torch.device("cpu")
            checkpoint = torch.load(self.graph_path, map_location="cpu")  # Load to RAM first

            # 1. RECONSTRUCT using add_node to ensure ModuleDict registration
            nodes_data = checkpoint.get("nodes_data", {})

            # Clear existing to prevent duplicates
            graph.nodes.clear()
            graph.node_order = []

            for node_id, data in nodes_data.items():
                # add_node registers the ConceptNode in the ModuleDict correctly
                graph.add_node(node_id)
                node_key = str(node_id)
                node = graph.nodes[node_key]

                # Copy data and move to device
                node.embedding.data.copy_(data['embedding'].to(device))
                node.alignment_score = data['alignment_score']
                node.connections = data['connections']
                node.is_tombstoned = data.get('is_tombstoned', False)
                if 'stage_idx' in data:
                    node.stage_idx = data['stage_idx']

            # 2. Restore Order and Anchors (Move anchors to the right hardware)
            graph.node_order = checkpoint.get("node_order", [])
            graph.anchor_tensor = {
                k: v.to(device) for k, v in checkpoint.get("anchor_tensor", {}).items()
            }

            # 3. Final State Dict check
            graph.load_state_dict(checkpoint.get("graph_state", {}), strict=False)

            # Move entire graph structure to the target device
            graph.to(device)

            logger.info(f"[Governance] Manifold RESTORED: {len(graph.nodes)} nodes active on {device}.")
            return True
        except Exception as e:
            logger.error(f"Load Failure: {e}")
            import traceback
            traceback.print_exc()
            return False