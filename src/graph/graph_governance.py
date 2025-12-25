# src/graph/graph_governance.py
import os
import torch
import fcntl
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphGovernance")


class GraphGovernance:
    """
    Manages the persistence and integrity of the SharedConceptGraph.
    Enforces atomic saves and file-locking to prevent state drift
    during multi-stage/multi-model checkpointing.
    """

    def __init__(self, graph_path: str = "checkpoints/global_manifold.pth"):
        self.graph_path = graph_path
        self.lock_path = f"{graph_path}.lock"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)

    def secure_save(self, graph: torch.nn.Module, metadata: Optional[dict] = None):
        """
        Performs an atomic 'Save-and-Swap'.
        Writes to a temporary file first, then renames to the target.
        """
        # Create or open the lock file
        with open(self.lock_path, "w") as lock_file:
            try:
                # Exclusive lock (LOCK_EX) - prevents other reads/writes
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                # --- INTEGRITY GATE: PRE-SAVE VALIDATION ---
                # Check for geometric blowouts before allowing a write to disk.
                graph_state = graph.state_dict()
                for key, tensor in graph_state.items():
                    if torch.is_tensor(tensor):
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            logger.error(f"[Governance] FATAL: {key} contains NaN/Inf. Aborting save to protect manifold.")
                            return False

                temp_path = f"{self.graph_path}.tmp"

                # Capture the graph state
                # Note: We save the state_dict of the SharedConceptGraph specifically
                state_to_save = {
                    "graph_state": graph_state,
                    "node_order": getattr(graph, "node_order", []),
                    "metadata": metadata or {}
                }

                torch.save(state_to_save, temp_path)

                # Atomic rename (POSIX compliant)
                os.replace(temp_path, self.graph_path)

                logger.info(f"[Governance] Manifold state secured at {self.graph_path}")
                return True

            except Exception as e:
                logger.error(f"[Governance] Critical Save Failure: {e}")
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
            finally:
                # Unlock
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def secure_load(self, graph: torch.nn.Module):
        """
        Loads the manifold state into the graph instance with a shared lock.
        """
        if not os.path.exists(self.graph_path):
            logger.warning(f"[Governance] No manifold file found at {self.graph_path}. Starting fresh.")
            return False

        # Create lock file if it doesn't exist to allow reading
        if not os.path.exists(self.lock_path):
            open(self.lock_path, 'a').close()

        with open(self.lock_path, "r") as lock_file:
            try:
                # Shared lock (LOCK_SH) - others can read, but no one can write
                fcntl.flock(lock_file, fcntl.LOCK_SH)

                # Map location ensures weights are loaded to the correct device
                device = next(graph.parameters()).device
                checkpoint = torch.load(self.graph_path, weights_only=False, map_location=device)

                # Load weights into the SharedConceptGraph
                graph.load_state_dict(checkpoint["graph_state"])

                # Restore topological metadata
                if "node_order" in checkpoint:
                    graph.node_order = checkpoint["node_order"]

                logger.info(f"[Governance] Manifold successfully restored from {self.graph_path}")
                return True

            except Exception as e:
                logger.error(f"[Governance] Load Failure: {e}")
                return False
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)