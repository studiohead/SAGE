# /config/config.py

# --- Core model structure (shared across all stages) ---
SHARED_MODEL_CONFIG = {
    "embed_dim": 128,      # Embedding dimension
    "nhead": 8,            # Attention heads
    "head_dim": 16,        # Each head's dimension
    "num_layers": 48       # Maximum layers in the full model stack
}

# --- Stage-specific hyperparameters & Hardware-Aware Layer Mapping ---
# Note: Each stage defines a 4-layer window to respect hardware constraints.
# Total stack is 48 layers; stages are trained sequentially to build the hierarchy.

STAGE_HYPERPARAMS = {
    # python -m src.main train --stage infant --epochs 5
    "Infant": {
        "layer_start": 0,      # Physical layer index in the 48-layer stack
        "layer_end": 6,        # Exclusive
        "epsilon_scale": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 1.0,
        "training_layers": 6,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 1000}
    },
    #python -m src.main train --stage toddler --epochs 5 --load infant
    "Toddler": {
        "layer_start": 10,      # Next window (Hardware limit: 4 layers)
        "layer_end": 14,
        "epsilon_scale": 0.08,
        "learning_rate": 8e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.95,
        "training_layers": 4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 800}
    },
    "Preschool": {
        "layer_start": 8,
        "layer_end": 12,
        "epsilon_scale": 0.05,
        "learning_rate": 5e-4,
        "weight_decay": 5e-6,
        "dropout": 0.12,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.9,
        "training_layers": 4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 600}
    },
    "Gradeschool": {
        "layer_start": 12,
        "layer_end": 16,
        "epsilon_scale": 0.03,
        "learning_rate": 4e-4,
        "weight_decay": 5e-6,
        "dropout": 0.15,
        "gradient_clip": 1.0,
        "plasticity_scale": 0.85,
        "training_layers": 4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 400}
    },
    "Teen": {
        "layer_start": 16,
        "layer_end": 20,
        "epsilon_scale": 0.02,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "dropout": 0.15,
        "gradient_clip": 0.8,
        "plasticity_scale": 0.8,
        "training_layers": 4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 300}
    },
    "Adult": {
        "layer_start": 20,
        "layer_end": 24,
        "epsilon_scale": 0.01,
        "learning_rate": 2e-4,
        "weight_decay": 1e-6,
        "dropout": 0.2,
        "gradient_clip": 0.7,
        "plasticity_scale": 0.7,
        "training_layers": 4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 200}
    },
    "Elder": {
        "layer_start": 24,     # Elder continues the climb toward the 48th layer
        "layer_end": 28,
        "epsilon_scale": 0.0,
        "learning_rate": 5e-5,
        "weight_decay": 0,
        "dropout": 0.25,
        "gradient_clip": 0.5,
        "plasticity_scale": 0.5,
        "training_layers": 4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 100}
    }
}