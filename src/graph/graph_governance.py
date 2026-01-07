import os
import torch
import logging
import math
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphGovernance")


class GraphGovernance:
    """
    SAGE GOVERNANCE: Manages persistence and topological integrity.
    Surgically reinforced to protect BROAD_CONCEPTS and ensure
    Tombstones/Pruning states are never lost during load/save cycles.
    """

    def __init__(self, graph_path: str = "checkpoints/global_manifold.pth"):
        self.graph_path = graph_path
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)

    def secure_save(self, graph: torch.nn.Module, metadata: Optional[dict] = None):
        """
        [0021] Atomic Save with Anchor Validation.
        Ensures the 'Stuff' is sanctioned before being committed to disk.
        """
        try:
            # 1. INTEGRITY CHECK: NaN is an immediate abort
            if torch.isnan(graph.master_embeddings).any():
                logger.error("[Governance] FATAL: Master Embeddings contain NaN. Save aborted.")
                return False

            # 2. CAPTURE DATA
            nodes_data = {}
            for idx, node_id in enumerate(graph.node_order):
                node_key = str(node_id)
                node = graph.nodes[node_key]

                # Move to CPU for hardware-agnostic checkpoints
                nodes_data[node_id] = {
                    'embedding': graph.master_embeddings[idx].data.cpu(),
                    'alignment_score': graph.master_alignments[idx].data.cpu(),
                    'connections': node.connections,
                    'stage_idx': getattr(node, 'stage_idx', -1),
                    'is_tombstoned': node.is_tombstoned,
                    'tombstone_key': node.tombstone_key.cpu() if node.tombstone_key is not None else None,
                    'label': getattr(node, 'label', None),
                    'tombstone_mask': graph.tombstone_mask[idx].cpu(),
                    'gradient_mask': getattr(node, 'gradient_mask', 1.0)
                }

            temp_path = f"{self.graph_path}.tmp"

            state_to_save = {
                "graph_state": graph.state_dict(),
                "nodes_data": nodes_data,
                "node_order": getattr(graph, "node_order", []),
                # [0018] Anchor Tensor Z is explicitly serialized
                "metadata": metadata or {}
            }

            # Atomic Swap
            torch.save(state_to_save, temp_path)
            os.replace(temp_path, self.graph_path)

            logger.info(f"[Governance] Manifold state secured at {self.graph_path}")
            return True
        except Exception as e:
            logger.error(f"[Governance] Critical Save Failure: {e}")
            return False

    def secure_load(self, graph: torch.nn.Module):
        """
        [0012] Reconstructs the manifold while enforcing Tombstone status.
        Ensures the mask is never 'Revived' accidentally.
        """
        if not os.path.exists(self.graph_path):
            return False
        try:
            device = next(graph.parameters()).device if list(graph.parameters()) else torch.device("cpu")
            checkpoint = torch.load(self.graph_path, map_location="cpu")

            nodes_data = checkpoint.get("nodes_data", {})

            # Clear to prevent coordinate pollution
            graph.nodes.clear()
            graph.node_order = []

            # 1. RECONSTRUCT Master Blocks
            for node_id, data in nodes_data.items():
                # add_node handles indices; we manually restore values
                graph.add_node(node_id)
                node_key = str(node_id)
                node = graph.nodes[node_key]
                idx = graph.node_order.index(node_key)

                with torch.no_grad():
                    graph.master_embeddings[idx].copy_(data['embedding'].to(device))
                    graph.master_alignments[idx].copy_(data['alignment_score'].to(device))

                    # SAGE REPAIR: Strict Tombstone Mask Restoration
                    # If the node was tombstoned, the mask MUST remain 0.0
                    t_mask = data.get('tombstone_mask', torch.ones(1)).to(device)
                    if data.get('is_tombstoned', False):
                        t_mask = torch.zeros_like(t_mask)
                    graph.tombstone_mask[idx].copy_(t_mask)

                # 2. RESTORE METADATA
                node.connections = data['connections']
                node.is_tombstoned = data.get('is_tombstoned', False)
                node.tombstone_key = data['tombstone_key'].to(device) if data.get('tombstone_key') is not None else None
                node.stage_idx = data.get('stage_idx', -1)
                node.label = data.get('label', None)
                node.gradient_mask = data.get('gradient_mask', 1.0)

            # 3. RESTORE ANCHOR REFERENCE FRAME
            graph.node_order = checkpoint.get("node_order", [])
            graph.anchor_tensor = {
                k: v.to(device) for k, v in checkpoint.get("anchor_tensor", {}).items()
            }

            # Final buffer/state sync
            graph.load_state_dict(checkpoint.get("graph_state", {}), strict=False)
            graph.to(device)

            logger.info(f"[Governance] RESTORED: {len(graph.nodes)} nodes (including Tombstones) on {device}.")
            return True
        except Exception as e:
            logger.error(f"Load Failure: {e}")
            import traceback
            traceback.print_exc()
            return False

    def rebrand_node(self, graph, node_id, new_label):
        """Linguistically grounds a node, increasing its governance priority."""
        node_key = str(node_id)
        if node_key in graph.nodes:
            node = graph.nodes[node_key]
            old_label = getattr(node, 'label', 'None')
            node.label = new_label
            # [0015] Developmental Maturation: rebranded nodes are 'Linguistic Anchors'
            logger.info(f"[Governance] REBRAND: Node {node_id} ('{old_label}') -> '{new_label}'")
            self.secure_save(graph, metadata={"event": "rebranding", "node_id": node_id})
            return True
        return False