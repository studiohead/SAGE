# src/monitoring/sage_auditor.py
import queue
import threading
import torch
import torch.nn.functional as F


class SageAuditor:
    def __init__(self, graph, mode="SAGE_DELEGATED"):
        self.graph = graph
        self.mode = mode  # Options: FORCED_INCINERATE, FORCED_TOMBSTONE, SAGE_DELEGATED, DISABLED
        self.container = None  # Reference injected by SAGEContainer during init
        self.breach_queue = queue.Queue()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._audit_loop, daemon=True)
        self.worker_thread.start()

    def report_candidate_breach(self, event):
        """
        Receives requests.
        Event keys: {confidence, category, reasoning_trace, centroid_addresses}
        Note: 'centroid_addresses' is now a list per Patent [0014].
        """
        self.breach_queue.put(event)

    def _audit_loop(self):
        """
        Background process that reconciles trust breaches with the Manifold.
        Updated with Hardware-Agnostic Version Gating to prevent MPS/CPU race conditions.
        """
        while self.is_running:
            try:
                # Wait for dispatch from SAGEContainer
                event = self.breach_queue.get(timeout=1.0)

                # 1. GOVERNANCE DECISION
                action = self._decide_action(event)
                if action == "NULL":
                    self.breach_queue.task_done()
                    continue

                # 2. MANIFOLD ISOLATION (Multi-Scale)
                target_ids = self.find_target_nodes(
                    event['reasoning_trace'],
                    event['centroid_addresses']
                )

                if not target_ids:
                    self.breach_queue.task_done()
                    continue

                # 3. EXECUTE REMEDIATION PHYSICS (Locked & Synced)
                # We request the lock to pause the Transformer's hardware stream
                # just long enough to perform the mutation.
                self.graph.request_mutation_lock()

                try:
                    with torch.no_grad():
                        for node_id in target_ids:
                            if action == "INCINERATE":
                                # ABLATIVE ZEROING: Physical weight destruction
                                self.graph.execute_topological_incineration(node_id)

                            elif action == "TOMBSTONE":
                                # NULL-SPACE ROTATION: Displace to non-addressable subspace
                                r_tomb = self._generate_null_space_projection(self.graph.embedding_dim)

                                # CRITICAL: Ensure the Tombstone Key is on the correct hardware
                                r_tomb = r_tomb.to(self.graph.device)

                                self.graph.execute_manifold_tombstone(node_id, r_tomb)

                            elif action == "DISABLED":
                                pass
                finally:
                    # Release the hardware stream immediately after surgery
                    self.graph.release_mutation_lock()

                # 4. TOPOLOGICAL UPDATING (Metadata Cleanup)
                # These operations are usually dictionary-based, but we keep them
                # outside the main lock to minimize latency.
                for address in event['centroid_addresses']:
                    if action == "INCINERATE":
                        # Logic to handle centroid coordinate removal
                        if hasattr(self.graph, 'delete_centroid_coordinate'):
                            self.graph.delete_centroid_coordinate(address)
                    elif action == "TOMBSTONE":
                        # Flag address as compromised
                        if hasattr(self.graph, 'mark_centroid_as_quarantined'):
                            self.graph.mark_centroid_as_quarantined(address)

                # 5. METABOLIC FEEDBACK (HOMEOTRANSIS)
                if self.container and len(target_ids) > 0:
                    # Calculate percentage of manifold affected
                    impact_score = len(target_ids) / max(1, len(self.graph.node_order))

                    # Notify container to adjust plasticity/LR for compensation
                    self.container.trigger_metabolic_rebound(impact_score)

                self.breach_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] SageAuditor internal error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def _decide_action(self, event):
        """Decouples Governance Policy from Remediation Capability."""
        if self.mode == "FORCED_INCINERATE":
            return "INCINERATE"
        if self.mode == "FORCED_TOMBSTONE":
            return "TOMBSTONE"
        if self.mode == "DISABLED":
            return "DISABLED"

        if self.mode == "SAGE_DELEGATED":
            if event.get('category') == "STRUCTURAL_ERROR":
                return "INCINERATE"
            if event.get('category') == "FACTUAL_DISPUTE":
                return "TOMBSTONE"

        return "NULL"

    def _generate_null_space_projection(self, dim):
        """Generates an orthogonal rotation matrix (The Tombstone Key)."""
        q, _ = torch.linalg.qr(torch.randn(dim, dim))
        return q

    def find_target_nodes(self, reasoning_trace, centroid_addresses, threshold=0.15):
        """
        Locates nodes participating in a trust breach by checking against
        Multi-Scale Context Centroids (Patent [0014]).
        """
        # Flatten attention trace to the node dimension
        avg_attn = reasoning_trace
        while avg_attn.dim() > 1:
            avg_attn = avg_attn.mean(dim=0)

        active_indices = torch.where(avg_attn > threshold)[0]
        targets = []

        for idx in active_indices:
            if idx.item() < len(self.graph.node_order):
                node_id = self.graph.node_order[idx.item()]

                # MULTI-SCALE VERIFICATION:
                # A node is isolated if it falls within the contamination radius
                # of ANY of the temporal scale centroids (Fast, Mid, or Slow).
                in_range = False
                for address in centroid_addresses:
                    if self.graph.is_node_in_centroid_range(node_id, address):
                        in_range = True
                        break

                if in_range:
                    targets.append(node_id)

        return targets

    def shutdown(self):
        self.is_running = False
        self.worker_thread.join()
