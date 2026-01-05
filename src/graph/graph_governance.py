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
    Surgically updated for Contiguous Parameter Blocks.
    """

    def __init__(self, graph_path: str = "checkpoints/global_manifold.pth"):
        self.graph_path = graph_path
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)

    def secure_save(self, graph: torch.nn.Module, metadata: Optional[dict] = None):
        try:
            # 1. Capture the structural data from Contiguous Master Tensors
            # We map node_id to its index in the master tensors via node_order
            nodes_data = {}
            for idx, node_id in enumerate(graph.node_order):
                node_key = str(node_id)
                node = graph.nodes[node_key]

                # Fetching from Master Blocks (The Breakout Fix)
                # We move to .cpu() here to prevent hardware-locking the checkpoint
                nodes_data[node_id] = {
                    'embedding': graph.master_embeddings[idx].data.cpu(),
                    'alignment_score': graph.master_alignments[idx].data.cpu(),
                    'connections': node.connections,
                    'stage_idx': getattr(node, 'stage_idx', -1),
                    'is_tombstoned': node.is_tombstoned,
                    'label': getattr(node, 'label', None),
                    # Preserve mask state
                    'tombstone_mask': graph.tombstone_mask[idx].cpu()
                }

            # Integrity Guard: Check for NaN before overwriting good data
            # Check the Master Tensors directly for global health
            if torch.isnan(graph.master_embeddings).any():
                logger.error("[Governance] FATAL: Master Embeddings contain NaN. Save aborted.")
                return False

            temp_path = f"{self.graph_path}.tmp"

            state_to_save = {
                "graph_state": graph.state_dict(),
                "nodes_data": nodes_data,
                "node_order": getattr(graph, "node_order", []),
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
            import traceback
            traceback.print_exc()
            return False

    def secure_load(self, graph: torch.nn.Module):
        if not os.path.exists(self.graph_path):
            return False
        try:
            device = next(graph.parameters()).device if list(graph.parameters()) else torch.device("cpu")
            checkpoint = torch.load(self.graph_path, map_location="cpu")

            nodes_data = checkpoint.get("nodes_data", {})

            # Clear existing to prevent duplicates
            graph.nodes.clear()
            graph.node_order = []

            # 1. RECONSTRUCT: add_node registers the ID and activates the Master Block slot
            for node_id, data in nodes_data.items():
                graph.add_node(node_id)
                node_key = str(node_id)
                node = graph.nodes[node_key]
                idx = graph.node_order.index(node_key)

                # Restore into Master Tensors
                with torch.no_grad():
                    graph.master_embeddings[idx].copy_(data['embedding'].to(device))
                    graph.master_alignments[idx].copy_(data['alignment_score'].to(device))
                    graph.tombstone_mask[idx].copy_(data.get('tombstone_mask', torch.ones(1)).to(device))

                # Restore Metadata to the Node Object
                node.connections = data['connections']
                node.is_tombstoned = data.get('is_tombstoned', False)
                node.stage_idx = data.get('stage_idx', -1)
                node.label = data.get('label', None)

            graph.node_order = checkpoint.get("node_order", [])
            graph.anchor_tensor = {
                k: v.to(device) for k, v in checkpoint.get("anchor_tensor", {}).items()
            }

            # 3. Final State Dict check (captures buffers and versioning)
            graph.load_state_dict(checkpoint.get("graph_state", {}), strict=False)
            graph.to(device)

            logger.info(f"[Governance] Manifold RESTORED: {len(graph.nodes)} nodes active on {device}.")
            return True
        except Exception as e:
            logger.error(f"Load Failure: {e}")
            import traceback
            traceback.print_exc()
            return False

    def rebrand_node(self, graph, node_id, new_label):
        node_key = str(node_id)
        if node_key in graph.nodes:
            node = graph.nodes[node_key]
            old_label = getattr(node, 'label', 'None')
            node.label = new_label
            node.is_linguistically_grounded = True
            logger.info(f"[Governance] REBRAND: Node {node_id} ('{old_label}') -> '{new_label}'")
            self.secure_save(graph, metadata={"event": "rebranding", "node_id": node_id})
            return True
        return False