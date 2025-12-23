# /config/config.py

# --- Core model structure (shared across all stages) ---
SHARED_MODEL_CONFIG = {
    "embed_dim": 128,      # Embedding dimension
    "nhead": 8,            # Attention heads
    "head_dim": 16,        # Each head's dimension
    "num_layers": 48       # Maximum layers in the full model stack
}

# --- Stage-specific hyperparameters & Hardware-Aware Layer Mapping ---
# Note: Each stage owns a block, but only trains the top 4 layers (The Rule of 4).
# We use a 2-layer handshake (overlap) to prevent 0.00 Variance collapse.

STAGE_HYPERPARAMS = {
    "Infant": {
        "layer_start": 0,
        "layer_end": 12,
        "training_layers": 12, # Foundational heavy-lift
        "epsilon_scale": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 1.0,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 1000}
    },
    "Toddler": {
        "layer_start": 10,      # Handshake: Overlaps final 2 layers of Infant
        "layer_end": 20,        # Ownership block: 10 layers
        "training_layers": 8,   # Actual training on top 4 (16-19)
        "epsilon_scale": 0.08,
        "learning_rate": 8e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.95,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 800}
    },
    "Preschool": {
        "layer_start": 18,      # Handshake: Overlaps final 2 layers of Toddler
        "layer_end": 28,
        "training_layers": 8,   # Actual training on top 4 (24-27)
        "epsilon_scale": 0.05,
        "learning_rate": 5e-4,
        "weight_decay": 5e-6,
        "dropout": 0.12,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.9,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 600}
    },
    "Gradeschool": {
        "layer_start": 26,      # Handshake
        "layer_end": 36,
        "training_layers": 4,   # Actual training (32-35)
        "epsilon_scale": 0.03,
        "learning_rate": 4e-4,
        "weight_decay": 5e-6,
        "dropout": 0.15,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.85,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 400}
    },
    "Teen": {
        "layer_start": 34,
        "layer_end": 42,
        "training_layers": 4,   # Actual training (38-41)
        "epsilon_scale": 0.02,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "dropout": 0.15,
        "gradient_clip": 0.8,
        "plasticity_scale": 0.8,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 300}
    },
    "Adult": {
        "layer_start": 40,
        "layer_end": 46,
        "training_layers": 4,   # Actual training (42-45)
        "epsilon_scale": 0.01,
        "learning_rate": 2e-4,
        "weight_decay": 1e-6,
        "dropout": 0.2,
        "gradient_clip": 0.7,
        "plasticity_scale": 0.7,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 200}
    },
    "Elder": {
        "layer_start": 44,     # Final synthesis
        "layer_end": 48,
        "training_layers": 4,   # Capstone (44-47)
        "epsilon_scale": 0.0,
        "learning_rate": 5e-5,
        "weight_decay": 0,
        "dropout": 0.25,
        "gradient_clip": 0.5,
        "plasticity_scale": 0.5,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 100}
    }
}