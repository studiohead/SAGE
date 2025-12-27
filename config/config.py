# --- Core model structure (shared across all stages) ---
SHARED_MODEL_CONFIG = {
    "embed_dim": 128,      # Embedding dimension
    "nhead": 8,            # Attention heads
    "head_dim": 16,        # Each head's dimension
    "num_layers": 48       # Maximum layers in the full model stack
}

# --- Stage-specific hyperparameters & Hardware-Aware Layer Mapping ---
# Each stage owns a block. To ensure continuity, we use a 2-layer handshake.
# The 'training_layers' parameter determines the active plasticity window.

STAGE_HYPERPARAMS = {
    "Infant": {
        "layer_start": 0,
        "layer_end": 4,
        "training_layers": 4,  # Foundational grounding
        "epsilon_scale": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 1.0,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 800},
        "batch_size": 6,
        "epochs": 5,
        "confidence_threshold": 0.1,  # Forced wiring
    },
    "Toddler": {
        "layer_start": 2,      # Handshake: Overlaps Layers 10 & 11 from Infant
        "layer_end": 8,        # Ownership block
        "training_layers": 6,   # Actual training focus
        "epsilon_scale": 0.08,
        "learning_rate": 8e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.95,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 600},
        "batch_size": 4,
        "epochs": 5,
        "confidence_threshold": 0.2,  # Low bar for visual discovery
    },
    "Preschool": {
        "layer_start": 6,      # Handshake: Overlaps Layers 18 & 19 from Toddler
        "layer_end": 14,
        "training_layers": 8,
        "epsilon_scale": 0.05,
        "learning_rate": 5e-4,
        "weight_decay": 5e-6,
        "dropout": 0.12,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.9,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 400},
        "batch_size": 4,
        "epochs": 5,
        "confidence_threshold": 0.5,  # Starting to require logic
    },
    "Gradeschool": {
        "layer_start": 12,      # Handshake: Overlaps Layers 26 & 27 from Preschool
        "layer_end": 22,
        "training_layers": 10,
        "epsilon_scale": 0.03,
        "learning_rate": 4e-4,
        "weight_decay": 5e-6,
        "dropout": 0.15,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.85,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 200},
        "batch_size": 1,
        "epochs": 5,
        "confidence_threshold": 0.7,  # Standard rigor
    },
    "Teen": {
        "layer_start": 20,      # Handshake: Overlaps Layers 34 & 35 from Gradeschool
        "layer_end": 32,
        "training_layers": 10,
        "epsilon_scale": 0.02,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "dropout": 0.15,
        "gradient_clip": 0.8,
        "plasticity_scale": 0.8,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 150},
        "batch_size": 1,
        "epochs": 5,
        "confidence_threshold": 0.7,  # Standard rigor
    },
    "Adult": {
        "layer_start": 20,      # Handshake: Overlaps Layers 40 & 41 from Teen
        "layer_end": 30,
        "training_layers": 10,
        "epsilon_scale": 0.01,
        "learning_rate": 2e-4,
        "weight_decay": 1e-6,
        "dropout": 0.2,
        "gradient_clip": 0.7,
        "plasticity_scale": 0.7,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 100},
        "batch_size": 1,
        "epochs": 5,
        "confidence_threshold": 0.7,  # Standard rigor
    },
    "Elder": {
        "layer_start": 28,      # Handshake: Overlaps Layers 44 & 45 from Adult
        "layer_end": 40,        # Final synthesis
        "training_layers": 12,   # Capstone
        "epsilon_scale": 0.0,
        "learning_rate": 5e-5,
        "weight_decay": 0,
        "dropout": 0.25,
        "gradient_clip": 0.5,
        "plasticity_scale": 0.5,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 50},
        "batch_size": 1,
        "epochs": 5,
        "confidence_threshold": 0.7,  # Standard rigor
    }
}
