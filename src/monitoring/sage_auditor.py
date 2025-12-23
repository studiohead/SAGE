# src/monitoring/sage_auditor.py
import queue
import threading
import torch
import torch.nn.functional as F


class SageAuditor:
    def __init__(self, graph, mode="SAGE_DELEGATED"):
        self.graph = graph
        self.mode = mode  # Options: FORCED_INCINERATE, FORCED_TOMBSTONE, SAGE_DELEGATED
        self.breach_queue = queue.Queue()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._audit_loop, daemon=True)
        self.worker_thread.start()

    def report_candidate_breach(self, event):
        """
        Receives requests.
        Event keys: {confidence, category, reasoning_trace, centroid_address}
        """
        self.breach_queue.put(event)

    def _audit_loop(self):
        while self.is_running:
            try:
                # Wait for dispatch from SAGEContainer
                event = self.breach_queue.get(timeout=1.0)

                # 1. GOVERNANCE DECISION
                action = self._decide_action(event)
                if action == "NULL":
                    self.breach_queue.task_done()
                    continue

                # 2. MANIFOLD ISOLATION
                # Isolate specific nodes participating in the low-confidence trace
                target_ids = self.find_target_nodes(event['reasoning_trace'], event['centroid_address'])

                # 3. EXECUTE REMEDIATION PHYSICS
                for node_id in target_ids:
                    if action == "INCINERATE":
                        # ABLATIVE ZEROING: Physical weight destruction + gradient masking
                        # This creates a "Logical Void" instead of noisy interference
                        self.graph.execute_topological_incineration(node_id)

                    elif action == "TOMBSTONE":
                        # NULL-SPACE ROTATION: Displaces vector into non-addressable subspace
                        # r_tomb ensures zero-convergence for future dot-product attention
                        r_tomb = self._generate_null_space_projection(self.graph.embedding_dim)
                        self.graph.execute_manifold_tombstone(node_id, r_tomb)

                # 4. TOPOLOGICAL UPDATING
                if action == "INCINERATE":
                    self.graph.delete_centroid_coordinate(event['centroid_address'])
                elif action == "TOMBSTONE":
                    # Mark as quarantined to prevent TCR from retrieving this manifold
                    self.graph.mark_centroid_as_quarantined(event['centroid_address'])

                self.breach_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"SageAuditor internal error: {e}")
                continue
            except Exception as e:
                # Log internal auditor errors here to maintain system stability
                continue

    def _decide_action(self, event):
        """Decouples Governance Policy from Remediation Capability."""
        if self.mode == "FORCED_INCINERATE":
            return "INCINERATE"
        if self.mode == "FORCED_TOMBSTONE":
            return "TOMBSTONE"

        # SAGE-DELEGATED: Abstraction-Level specific governance
        if self.mode == "SAGE_DELEGATED":
            # Structural Errors (Logic/Stability) require permanent ablation
            if event.get('category') == "STRUCTURAL_ERROR":
                return "INCINERATE"

            # Factual/Contextual disputes allow for reversible tombstoning
            if event.get('category') == "FACTUAL_DISPUTE":
                return "TOMBSTONE"

        return "NULL"

    def _generate_null_space_projection(self, dim):
        """
        Generates an orthogonal rotation matrix.
        In a patent sense, this is the 'Key' to the quarantined manifold.
        """
        # QR decomposition yields an orthogonal matrix Q
        q, _ = torch.linalg.qr(torch.randn(dim, dim))
        return q

    def find_target_nodes(self, reasoning_trace, centroid_address, threshold=0.15):
        """
        Locates nodes within a specific EWMA-centroid range
        that exhibit high participation in a trust-breaching reasoning path.
        """
        # Flatten attention trace: [num_nodes]
        avg_attn = reasoning_trace.mean(dim=0).mean(dim=0)

        # Identify high-impact indices
        active_indices = torch.where(avg_attn > threshold)[0]

        targets = []
        for idx in active_indices:
            node_id = self.graph.node_order[idx.item()]

            # Verify node is part of the offending conceptual zip-code
            if self.graph.is_node_in_centroid_range(node_id, centroid_address):
                targets.append(node_id)

        return targets

    def shutdown(self):
        self.is_running = False
        self.worker_thread.join()