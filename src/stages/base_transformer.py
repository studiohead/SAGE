import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.config import SHARED_MODEL_CONFIG

EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim')

class DevelopmentalTransformer(nn.Module):
    """
    Base Class for the SAGE Logarithmic Complexity Ladder.
    Implements the core geometric operators defined in Patent Paragraph [0013].
    """

    def __init__(self, embed_dim=EMBED_DIM, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Stage-specific authority weight
        self.stage_weight = nn.Parameter(torch.tensor(1.0))

        # Foundational subspace projection layer (The 'Lens')
        self.graph_projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # Stability operator
        self.norm = nn.LayerNorm(embed_dim)

    def get_multi_scale_anchor(self, centroid_addresses, scale_idx):
        if centroid_addresses and len(centroid_addresses) > scale_idx:
            # SUTURE: Ensure the anchor is on the same device as the model
            return centroid_addresses[scale_idx]
        return None

    def get_mean_field(self, graph_matrix):
        """
        SUTURE: Hardware-native mean field extraction.
        Removes redundant .to(device) calls to prevent MPS synchronization stalls.
        """
        if graph_matrix.dim() == 2:  # [num_nodes, dim]
            return graph_matrix.mean(dim=0)
        return graph_matrix.mean(dim=1).mean(dim=0)

    def scaled_dot_product_attention(self, Q, K, V, Wi=1.0, return_attn=False):
        d_k = Q.size(-1)
        device = Q.device
        # SUTURE: Explicitly cast Wi to device to keep the entire operation on MPS
        tau = torch.as_tensor(Wi, device=device).clamp(min=0.1, max=1.0)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d_k) * tau)
        attn_probs = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_probs, V)
        return (context, attn_probs) if return_attn else context

    # --- SUTURE: REMOVED MANUAL TO() METHOD ---
    # PyTorch's native recursion handles the migration of self.norm,
    # self.graph_projection, and self.stage_weight automatically.

    def forward(self, x, graph_matrix, centroid_addresses=None, Wi=1.0):
        raise NotImplementedError("Each stage must implement its specific Geometric Operator.")