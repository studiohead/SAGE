# --- Core model structure (shared across all stages) ---
SHARED_MODEL_CONFIG = {
    "embed_dim": 256,  # Embedding dimension
    "nhead": 8,  # Attention heads
    "head_dim": 32,  # Each head's dimension
    "num_layers": 48  # Maximum layers in the full model stack
}

# --- Stage-specific hyperparameters & Hardware-Aware Layer Mapping ---


STAGE_HYPERPARAMS = {
    "Infant": {
        "layer_start": 0,
        "layer_end": 2,
        "training_layers": 2,
        "epsilon_scale": 0.05,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,  # Passive Centripetal pull
        "dropout": 0.1,
        "gradient_clip": 0.8,
        "plasticity_scale": 0.5,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 800},
        "batch_size": 128,
        "epochs": 5,
        "confidence_threshold": 0.45,
        "pressure_weight": 0.8,
        "max_structural_density": 0.20,
        "prune_fraction": 0.675,
        "drift_penalty": 0.0
    },
    "Toddler": {
        "layer_start": 0,
        "layer_end": 4,
        "training_layers": 4,
        "epsilon_scale": 0.08,    # Peak curiosity/exploration
        "learning_rate": 8e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 600},
        "batch_size": 32,
        "epochs": 5,
        "confidence_threshold": 0.6,
        "pressure_weight": 0.5,
        "max_structural_density": 0.25, # Space expands for relational depth
        "prune_fraction": 0.5,
        "drift_penalty": 0.05
    },
    "Preschool": {
        "layer_start": 0,
        "layer_end": 4,
        "training_layers": 4,
        "epsilon_scale": 0.05,
        "learning_rate": 5e-4,
        "weight_decay": 5e-6,
        "dropout": 0.12,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.3,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 400},
        "batch_size": 32,
        "epochs": 5,
        "confidence_threshold": 0.7,
        "pressure_weight": 0.4,   # Focus starts shifting to Saliency
        "max_structural_density": 0.25,
        "prune_fraction": 0.4
    },
    "Gradeschool": {
        "layer_start": 0,
        "layer_end": 6,
        "training_layers": 6,
        "epsilon_scale": 0.03,
        "learning_rate": 4e-4,
        "weight_decay": 5e-6,
        "dropout": 0.15,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.25,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 200},
        "batch_size": 16,
        "epochs": 5,
        "confidence_threshold": 0.77,
        "pressure_weight": 0.3,   # Categorical logic requires stability
        "max_structural_density": 0.30,
        "prune_fraction": 0.4
    },
    "Teen": {
        "layer_start": 0,
        "layer_end": 8,
        "training_layers": 8,
        "epsilon_scale": 0.04, # Peak curiosity/exploration
        "learning_rate": 1.2e-4,
        "weight_decay": 1e-6,
        "dropout": 0.15,
        "gradient_clip": 0.8,
        "plasticity_scale": 0.2,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 150},
        "batch_size": 2,
        "epochs": 5,
        "confidence_threshold": 0.80,
        "growth_confidence_floor": 0.65,
        "pressure_weight": 0.2,
        "max_structural_density": 0.35,
        "prune_fraction": 0.4
    },
    "Adult": {
        "layer_start": 0,
        "layer_end": 8,
        "training_layers": 8,
        "epsilon_scale": 0.01,
        "learning_rate": 2e-4,
        "weight_decay": 1e-6,
        "dropout": 0.2,
        "gradient_clip": 0.7,
        "plasticity_scale": 0.1,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 100},
        "batch_size": 2,
        "epochs": 5,
        "confidence_threshold": 0.92,
        "growth_confidence_floor": 0.6,
        "pressure_weight": 0.1,
        "max_structural_density": 0.30,
        "prune_fraction": 0.5
    },
    "Elder": {
        "layer_start": 0,
        "layer_end": 10,
        "training_layers": 10,
        "epsilon_scale": 0.0,     # Crystallized Intelligence
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
        "pressure_weight": 0.05,  # Minimal repulsion
        "max_structural_density": 0.25,
        "prune_fraction": 0.40
    }
    # and so on...
}