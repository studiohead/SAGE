# --- Core model structure (shared across all stages) ---
SHARED_MODEL_CONFIG = {
    "embed_dim": 256,  # Embedding dimension
    "nhead": 8,  # Attention heads
    "head_dim": 32,  # Each head's dimension
    "num_layers": 48  # Maximum layers in the full model stack
}

# --- Stage-specific hyperparameters & Hardware-Aware Layer Mapping ---
# Each stage owns a block. To ensure continuity, we use a 2-layer handshake.

STAGE_HYPERPARAMS = {
    "Infant": {
        "layer_start": 0,
        "layer_end": 4,
        "training_layers": 4,
        "epsilon_scale": 0.15,
        "learning_rate": 1e-3,  # INCREASED: Higher baseline for Infant
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.5,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 800},
        "batch_size": 16,
        "epochs": 5,
        "confidence_threshold": 0.45,

        # --- THE PRESSURE ENGINE ---
        "pressure_weight": 0.5,  # NAPALM: Forced repulsion to break 0.9983
        "max_manifold_tightness": 0.20,
        "prune_fraction": 0.6
    },
    "Toddler": {
        "layer_start": 2,
        "layer_end": 8,
        "training_layers": 6,
        "epsilon_scale": 0.08,
        "learning_rate": 8e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 600},
        "batch_size": 4,
        "epochs": 5,
        "confidence_threshold": 0.6,

        # --- THE PRESSURE ENGINE ---
        "pressure_weight": 0.5,  # Moderate repulsion
        "max_manifold_tightness": 0.25,
        "prune_fraction": 0.45
    },
    "Preschool": {
        "layer_start": 6,
        "layer_end": 14,
        "training_layers": 8,
        "epsilon_scale": 0.05,
        "learning_rate": 5e-4,
        "weight_decay": 5e-6,
        "dropout": 0.12,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.3,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 400},
        "batch_size": 4,
        "epochs": 5,
        "confidence_threshold": 0.7,

        # --- THE PRESSURE ENGINE ---
        "pressure_weight": 0.4,
        "max_manifold_tightness": 0.3,
        "prune_fraction": 0.35
    },
    "Gradeschool": {
        "layer_start": 12,
        "layer_end": 22,
        "training_layers": 10,
        "epsilon_scale": 0.03,
        "learning_rate": 4e-4,
        "weight_decay": 5e-6,
        "dropout": 0.15,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.25,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 200},
        "batch_size": 2,
        "epochs": 5,
        "confidence_threshold": 0.82,

        # --- THE PRESSURE ENGINE ---
        "pressure_weight": 0.3,  # Stability takes over
        "max_manifold_tightness": 0.35,
        "prune_fraction": 0.3
    },
    "Teen": {
        "layer_start": 20,
        "layer_end": 32,
        "training_layers": 10,
        "epsilon_scale": 0.02,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "dropout": 0.15,
        "gradient_clip": 0.8,
        "plasticity_scale": 0.2,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 150},
        "batch_size": 2,
        "epochs": 5,
        "confidence_threshold": 0.88,
        "growth_confidence_floor": 0.51,

        # --- THE PRESSURE ENGINE ---
        "pressure_weight": 0.2,
        "max_manifold_tightness": 0.35,
        "prune_fraction": 0.4
    },
    "Adult": {
        "layer_start": 20,
        "layer_end": 30,
        "training_layers": 10,
        "epsilon_scale": 0.01,
        "learning_rate": 2e-4,
        "weight_decay": 1e-6,
        "dropout": 0.2,
        "gradient_clip": 0.7,
        "plasticity_scale": 0.1,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 100},
        "batch_size": 1,
        "epochs": 5,
        "confidence_threshold": 0.92,
        "growth_confidence_floor": 0.6,

        # --- THE PRESSURE ENGINE ---
        "pressure_weight": 0.2,
        "max_manifold_tightness": 0.3,
        "prune_fraction": 0.50
    },
    "Elder": {
        "layer_start": 28,
        "layer_end": 40,
        "training_layers": 12,
        "epsilon_scale": 0.0,
        "learning_rate": 5e-5,
        "weight_decay": 0,
        "dropout": 0.25,
        "gradient_clip": 0.5,
        "plasticity_scale": 0.05,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 50},
        "batch_size": 1,
        "epochs": 5,
        "confidence_threshold": 0.95,
        "growth_confidence_floor": 0.7,

        # --- THE PRESSURE ENGINE ---
        "pressure_weight": 0.1,
        "max_manifold_tightness": 0.25,
        "prune_fraction": 0.40
    }
}