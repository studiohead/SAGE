# /config/config.py

# --- Core model structure (shared across all stages) ---
SHARED_MODEL_CONFIG = {
    "embed_dim": 128,      # Embedding dimension
    "nhead": 8,            # Attention heads
    "head_dim": 16,        # Each head's dimension
    "num_layers": 48       # Maximum layers in the full model
}

# --- Stage-specific hyperparameters ---
STAGE_HYPERPARAMS = {
    "Infant": {
        "epsilon_scale": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "plasticity_scale": 1.0,
        "training_layers": 4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 1000}
    },
    "Toddler": {
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
        "epsilon_scale": 0.03,
        "learning_rate": 4e-4,
        "weight_decay": 5e-6,
        "dropout": 0.15,
        "gradient_clip": 0.9,
        "plasticity_scale": 0.85,
        "training_layers": 4,
        "scheduler": {"type": "linear_warmup", "warmup_steps": 400}
    },
    "Teen": {
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
