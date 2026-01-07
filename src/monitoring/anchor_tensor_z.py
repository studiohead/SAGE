# src/monitoring/anchor_tensor_z.py

import torch
import torch.nn as nn


class AnchorTensorZ(nn.Module):
    def __init__(self, embed_dim, broad_concepts):
        super().__init__()
        self.embed_dim = embed_dim

        # [0021] Stable Reference Frame: The Foundation (0-9, A-Z)
        # We use torch.eye to create an Orthonormal Basis.
        # These are the "North Stars" that Stage 6 cannot move.
        self.num_foundational = len(broad_concepts)
        self.register_buffer("foundational_basis", torch.eye(self.num_foundational, embed_dim))

        # Mapping for O(1) retrieval
        self.label_to_idx = {label: i for i, label in enumerate(broad_concepts)}

        # Inertial Mass: High = Frozen/Immutable, Low = Plastic/Teen
        # BROAD_CONCEPTS are initialized with float('inf') mass.
        self.register_buffer("inertial_mass", torch.full((self.num_foundational,), float('inf')))

        # EWMA Centroid Cache (Z)
        self.register_buffer("Z_centroids", self.foundational_basis.clone())

    def get_anchor(self, label):
        """O(1) retrieval of the Stable Reference Coordinate."""
        idx = self.label_to_idx.get(label)
        if idx is not None:
            return self.Z_centroids[idx]
        return None

    def calculate_gamma_gate(self, current_embedding, label):
        """
        [0017] The Gating Mechanism:
        Calculates the trust metric (Gamma) based on drift from the Z-Anchor.
        """
        anchor = self.get_anchor(label)
        if anchor is None: return 1.0  # New nodes start with full trust

        # Relational Invariance: Euclidean distance on the hypersphere
        dist = torch.norm(current_embedding - anchor, p=2)
        gamma = torch.exp(-dist)  # 1.0 = Perfect Alignment

        return gamma
