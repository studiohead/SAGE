# src/stages/teen_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_transformer import DevelopmentalTransformer
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS

class TeenTransformer(DevelopmentalTransformer):
    """
    Stage 5: Competitive Abstraction & Stochastic Routing.
    Logic: y = Gumbel-Softmax(x, W * G)
    """

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
                dropout=stage_cfg.get("dropout", 0.15),
                batch_first=False
            ) for _ in range(num_stage_layers)
        ])

        self.gate_predictor = nn.Linear(embed_dim, 2)

        with torch.no_grad():
            self.gate_predictor.weight.fill_(0.0)
            self.gate_predictor.bias.data = torch.tensor([1.5, -1.5])

        self.refiner_norm = nn.LayerNorm(embed_dim)
        self.plasticity_scale = stage_cfg["plasticity_scale"]

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        impact = torch.tensor(0.0, device=x.device)

        if graph_matrix is not None and graph_matrix.size(0) > 0:
            S, B, D = x.shape

            # 1. THE GRADESCHOOL FIX: Clean Anchor Reduction
            # This handles the 3150-node leak by forcing a centroid mean
            if centroid_addresses and len(centroid_addresses) > 0:
                anchor_raw = centroid_addresses[0]
                anchor_vec = anchor_raw.reshape(-1, D).mean(dim=0) # Guaranteed [D]
            else:
                anchor_vec = graph_matrix.reshape(-1, D).mean(dim=0)

            # 2. Align Anchor to x: [1, 1, D] -> [S, B, D]
            anchor_unit = F.normalize(anchor_vec, p=2, dim=-1)
            anchor_context = anchor_unit.view(1, 1, D).expand(S, B, D)

            # 3. Gumbel Gate (Discrete Choice)
            # gate_logits shape: [S, B, 2]
            gate_logits = self.gate_predictor(x)
            tau = max(0.8, Wi)

            if self.training:
                gate = F.gumbel_softmax(gate_logits, tau=tau, hard=True)
            else:
                gate_probs = F.softmax(gate_logits / tau, dim=-1)
                indices = gate_probs.argmax(dim=-1, keepdim=True)
                gate = torch.zeros_like(gate_probs).scatter_(-1, indices, 1.0)

            # 4. Integrate paths
            # Path 0: Internal (x), Path 1: Graph (anchor_context)
            # gate[..., 0:1] is [S, B, 1]
            x_integrated = gate[..., 0:1] * x + gate[..., 1:2] * anchor_context

            # 5. Telemetry
            cos_sim = F.cosine_similarity(x_integrated, x, dim=-1).mean()
            impact = 1.0 - torch.clamp(cos_sim, 0, 1)

            x = self.norm(x_integrated)

        for layer in self.layers:
            x = layer(x)

        x = self.refiner_norm(x)
        return self.stage_weight * x, impact