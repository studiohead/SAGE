##############################################################################
# Teen | y = Gumbel-Softmax(x_internal, x_memory)
# Purpose: Discrete stochastic arbitration.
# Optimized: Actually uses the TCR (Retrieved Memory) provided by the Container.
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS


class TeenTransformer(DevelopmentalTransformer):
    def __init__(self):
        embed_dim = SHARED_MODEL_CONFIG["embed_dim"]
        nhead = SHARED_MODEL_CONFIG["nhead"]
        super().__init__(embed_dim=embed_dim, num_heads=nhead)

        stage_cfg = STAGE_HYPERPARAMS["Teen"]
        num_stage_layers = stage_cfg["layer_end"] - stage_cfg["layer_start"]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                dropout=stage_cfg.get("dropout", 0.15)
            ) for _ in range(num_stage_layers)
        ])

        # Gate Predictor: 2 channels (0: Stay Internal, 1: Trust Graph/Memory)
        self.gate_predictor = nn.Linear(embed_dim, 2)

        # INITIALIZATION: Bias toward 'Internal' initially to ensure stability
        with torch.no_grad():
            self.gate_predictor.bias.data = torch.tensor([1.0, -1.0])

        self.refiner_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        """
        x: [S_aug, B, D] -> Contains input tokens and TCR memory.
        graph_matrix: [N, D] -> The shared concept manifold (2,761 nodes).
        centroid_addresses: List of tensors -> Global/Contextual anchors.
        Wi: float -> Plasticity weight (decays over the stage lifecycle).
        """
        # Initialize Telemetry
        impact = torch.tensor(0.0, device=x.device)
        self.curiosity_score = torch.tensor(0.0, device=x.device)
        self.last_inquiry = None  # Reset inquiry for this pass

        if graph_matrix is not None:
            # 1. GATE PREDICTION (Gumbel-Softmax)
            # 0: Internal Thought, 1: World Anchor
            gate_logits = self.gate_predictor(x)
            tau = max(0.5, 1.5 * Wi)

            if self.training:
                gate = F.gumbel_softmax(gate_logits, tau=tau, hard=True)
            else:
                gate_probs = F.softmax(gate_logits / tau, dim=-1)
                gate = torch.zeros_like(gate_probs).scatter_(-1, gate_probs.argmax(-1, True), 1.0)

            # 2. DEFINE THE WORLD ANCHOR (Contextual Baseline)
            if centroid_addresses and len(centroid_addresses) > 0:
                # Teen looks for immediate relational relevance (Index 0)
                anchor_vec = centroid_addresses[0].reshape(-1, self.embed_dim).mean(0)
            else:
                anchor_vec = graph_matrix.mean(0)

            anchor_context = anchor_vec.view(1, 1, -1)

            # 3. CURIOSITY & INQUIRY (The Sweet Spot Logic)
            # We measure the distance between the current token and the anchor.
            with torch.no_grad():
                # Distance: 0.0 (identical) to 1.0 (opposite)
                token_dist = 1.0 - F.cosine_similarity(x, anchor_context, dim=-1)

                # Gaussian curiosity curve: Peaks at distance 0.5 (Moderate Novelty)
                # This ignores things it knows (dist < 0.2) and things that are noise (dist > 0.8)
                token_curiosity = torch.exp(-((token_dist - 0.5) ** 2) / 0.08)
                self.curiosity_score = token_curiosity.mean()

                # 4. TOPIC EXTRACTION: Which node caused the peak curiosity?
                if self.training and token_curiosity.max() > 0.6:
                    # Find the specific token in the sequence that triggered curiosity
                    most_curious_token_idx = token_curiosity.argmax()
                    curious_vector = x[most_curious_token_idx].unsqueeze(0)

                    # Search the 2,761 nodes for the closest match to this vector
                    # This tells us WHICH concept the model wants more data on
                    concept_sims = F.cosine_similarity(curious_vector, graph_matrix, dim=-1)
                    self.last_inquiry = concept_sims.argmax().item()

            # 5. COMPETITIVE INTEGRATION
            # Merge Internal and External based on the Gate
            x_integrated = (gate[..., 0:1] * x) + (gate[..., 1:2] * anchor_context)

            # 6. TELEMETRY: Impact Calculation
            cos_sim = F.cosine_similarity(x_integrated, x, dim=-1).mean()
            impact = (1.0 - torch.clamp(cos_sim, 0, 1)).detach()

            # Apply LayerNorm to the integrated representation
            x = self.norm(x_integrated)

        # 7. TRANSFORMER BACKBONE
        # Process the arbitrated choices through the Teen's specific layers
        for layer in self.layers:
            x = layer(x)

        x = self.refiner_norm(x)

        # We pass back the 'Impact' (actual change) as the primary telemetry,
        # but the caller can now access self.last_inquiry for curiosity logging.
        return self.stage_weight * x, impact