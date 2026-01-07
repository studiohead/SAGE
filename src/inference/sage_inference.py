import torch
import torch.nn.functional as F

class SAGEInference:
    """
    Inference interface for a dynamically growing graph (SAGE / Citizen/Elder model).
    Retrieves human-readable semantic labels from the graph.
    Provides novelty/impact telemetry for each prompt.
    """

    def __init__(self, graph, tokenizer):
        self.graph = graph
        self.tokenizer = tokenizer

    def respond(self, prompt, top_k=5, return_impact=False):
        """
        Retrieves the top_k most semantically relevant nodes for a given prompt.

        Args:
            prompt (str): Input prompt to query the graph.
            top_k (int): Number of top nodes to retrieve.
            return_impact (bool): If True, returns a novelty score for the prompt.

        Returns:
            List[str]: Human-readable labels corresponding to the retrieved nodes.
            Optional[float]: Novelty/impact score if return_impact=True.
        """

        # --- 1. Encode prompt ---
        if not hasattr(self.tokenizer, "encode"):
            raise RuntimeError("Tokenizer must have an encode() method.")

        prompt_emb = self.tokenizer.encode(prompt)
        if prompt_emb.dim() == 1:
            prompt_emb = prompt_emb.unsqueeze(0)  # [1, embed_dim]

        # --- 2. Ensure device alignment ---
        device = getattr(self.graph, "device", "cpu")
        prompt_emb = prompt_emb.to(device)

        # --- 3. Retrieve semantic context from dynamic graph ---
        try:
            context_data = self.graph.retrieve_manifold_context(prompt_emb, top_k=top_k)
        except AttributeError:
            context_data = []

        response_labels = []

        # --- 4. Map node IDs to labels, handling newly added nodes ---
        for _, node_id in context_data:
            node_key = str(node_id)
            node = self.graph.nodes.get(node_key, None)

            if node is None:
                response_labels.append(f"<missing:{node_key}>")
                continue

            label = getattr(node, "label", None)
            response_labels.append(str(label) if label is not None else node_key)

        # --- 5. Telemetry: Novelty / Impact Score ---
        impact_score = None
        if return_impact and context_data:
            # Compute cosine similarity between prompt and retrieved nodes
            retrieved_embs = torch.stack([self.graph.nodes[str(nid)].embedding
                                          for _, nid in context_data
                                          if str(nid) in self.graph.nodes], dim=0)  # [top_k, embed_dim]

            if retrieved_embs.size(0) > 0:
                # [top_k, embed_dim] @ [1, embed_dim] → [top_k]
                cos_sim = F.cosine_similarity(retrieved_embs, prompt_emb.expand_as(retrieved_embs), dim=-1)
                novelty = 1.0 - cos_sim.clamp(0.0, 1.0)
                impact_score = novelty.mean().item()  # mean novelty across top_k nodes
            else:
                impact_score = 1.0  # fully novel if no nodes exist

        if return_impact:
            return response_labels, impact_score
        return response_labels
