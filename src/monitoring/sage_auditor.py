import queue
import threading
import torch
import torch.nn.functional as F


class SageAuditor:
    def __init__(self, graph, mode="SAGE_DELEGATED"):
        self.graph = graph
        self.mode = mode
        self.container = None
        self.breach_queue = queue.Queue()
        self.is_running = True

        # SUTURE 1: Do not store a static self.device string.
        # Use the graph's dynamic property to ensure the thread always sees the latest hardware.
        self.worker_thread = threading.Thread(target=self._audit_loop, daemon=True)
        self.worker_thread.start()

    @property
    def device(self):
        """Dynamic hardware detection synced with the Manifold."""
        return getattr(self.graph, 'device', torch.device("cpu"))

    def report_candidate_breach(self, event):
        self.breach_queue.put(event)

    def _audit_loop(self):
        while self.is_running:
            try:
                event = self.breach_queue.get(timeout=1.0)
                action = self._decide_action(event)

                if action == "NULL" or action == "DISABLED":
                    self.breach_queue.task_done()
                    continue

                # SUTURE 2: Pull the current hardware device at the top of the loop
                dev = self.device

                # Align incoming event data to the active hardware
                trace = event['reasoning_trace'].to(dev)
                addresses = [addr.to(dev) for addr in event['centroid_addresses']]

                target_ids = self.find_target_nodes(trace, addresses)

                if not target_ids:
                    self.breach_queue.task_done()
                    continue

                # SUTURE 3: Lock-Free Mutex check.
                # If your Graph doesn't have request_mutation_lock, we use no_grad as the primary guard.
                try:
                    with torch.no_grad():
                        for node_id in target_ids:
                            # SHIELD: Check if this node is a protected BROAD_CONCEPT
                            node_label = getattr(self.graph.nodes.get(str(node_id)), 'label', None)
                            if node_label in [str(i) for i in range(400)]:
                                print(
                                    f"[!] AUDIT BLOCK: Attempted {action} on Foundational Anchor {node_id}. Action Aborted.")
                                continue  # Immune to incineration/tombstoning
                            if action == "INCINERATE":
                                self.graph.execute_topological_incineration(node_id)

                            elif action == "TOMBSTONE":
                                # Generate rotation on the dynamic device
                                r_tomb = self._generate_null_space_projection(self.graph.embedding_dim)

                                # SUTURE 4: Align method signature.
                                # Your Graph's execute_manifold_tombstone only takes node_id and r_tomb.
                                self.graph.execute_manifold_tombstone(node_id, r_tomb)
                except Exception as e:
                    print(f"[!] Remediation failed: {e}")
                finally:
                    pass

                    # 5. Metadata Cleanup (Aligned with your Graph's actual attributes)
                for address in addresses:
                    if action == "TOMBSTONE":
                        # If graph doesn't have this, it's handled by anchor_tensor updates internally
                        if hasattr(self.graph, 'quarantined_centroids'):
                            self.graph.quarantined_centroids.append(address)

                if self.container and len(target_ids) > 0:
                    impact_score = len(target_ids) / max(1, len(self.graph.node_order))
                    self.container.trigger_metabolic_rebound(impact_score)

                self.breach_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] SageAuditor internal error: {e}")
                continue

    def _decide_action(self, event):
        if self.mode == "FORCED_INCINERATE": return "INCINERATE"
        if self.mode == "FORCED_TOMBSTONE": return "TOMBSTONE"
        if self.mode == "DISABLED": return "DISABLED"
        if self.mode == "SAGE_DELEGATED":
            if event.get('category') == "STRUCTURAL_ERROR": return "INCINERATE"
            if event.get('category') == "FACTUAL_DISPUTE": return "TOMBSTONE"
        return "NULL"

    def _generate_null_space_projection(self, dim):
        # Always generate on current hardware
        q, _ = torch.linalg.qr(torch.randn(dim, dim, device=self.device))
        return q

    def find_target_nodes(self, reasoning_trace, centroid_addresses, threshold=0.15):
        dev = self.device
        avg_attn = reasoning_trace
        while avg_attn.dim() > 1:
            avg_attn = avg_attn.mean(dim=0)

        active_indices = torch.where(avg_attn > threshold)[0]
        targets = []

        for idx in active_indices:
            idx_val = idx.item()
            if idx_val < len(self.graph.node_order):
                node_id = self.graph.node_order[idx_val]

                # Check ranges (Graph method now handles device alignment internally)
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