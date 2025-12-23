# SAGE: Self-Actuating Governance Engine

A Multistage Developmental Cognitive Architecture with Asynchronous Topological Incineration, Null-Space Rotation & Null-Space Convergence

[![License: Proprietary](https://img.shields.io/badge/License-All_Rights_Reserved-red.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

SAGE is not a standard LLM. It is a Self-Correcting Cognitive Architecture designed to solve the “Immutability of Lies” in neural networks. Unlike traditional models that require total retraining to unlearn misinformation, SAGE utilizes a Hierarchical Governance Ladder and a Structured Concept Graph to surgically identify, isolate, and remediate deceptive weight manifolds in real-time.

CORE INNOVATIONS
1. The Logarithmic Complexity Ladder

SAGE processes information through 7 distinct developmental stages, each representing an increase in mathematical complexity and trust authority. As the system matures, it shifts from stochastic grounding to recursive meta-governance.

| Stage | Name        | Mathematical Operator             | Cognitive Milestone            |
| ----- | ----------- | --------------------------------- | ------------------------------ |
| 1     | Infant      | y = x + (W_i · mean(G)) + ε       | Stochastic Grounding           |
| 2     | Toddler     | y = x + (W_i · (x ⊙ σ(mean(G))))  | Relational Orientation         |
| 3     | Preschool   | y = x + W_i · Σ(Softmax(G) · G)   | Global Saliency Focus          |
| 4     | Gradeschool | y = x + W_i · (G · W_proj)        | Categorical Logic & Manifolds  |
| 5     | Teen        | y = Gumbel-Softmax(x, W_i · G)    | Competitive Arbitration        |
| 6     | Adult       | y = CrossAttention(Q=x, K,V=G)    | Selective Relational Reasoning |
| 7     | Sage        | y = SchemaInduction(Trace ⊗ Mask) | Recursive Meta-Governance      |


Each stage supports dynamic hyperparameters via config.config.STAGE_HYPERPARAMS, allowing adjustment of plasticity, training layers, and epsilon scaling during inference.

## 2. Shared Concept Graph (Topological Memory)

Unlike black-box models, SAGE stores persistent relational knowledge in a structured, addressable graph:

Intrinsic Centroid Anchoring: Concepts are addressed via centroids in a Versioned Anchor Tensor (Z), enabling O(1) targeting of conceptual clusters with Exponentially Weighted Moving Averages (EWMA).

Digital Scar Tissue: Nodes involved in trust breaches have alignment_score reduced to zero, creating permanent “cauterized” zones that inhibit relearning deceptive patterns.

Governance-Aware Hebbian Updates: Attention-driven relational updates respect scar tissue and null-space quarantine, ensuring safe incremental learning.

## 3. Asynchronous Remediation Physics

The “Immune System” of the architecture. When the Sage Stage detects a trust breach (low Γ confidence):

Evidence Capture: Isolates the Adult Stage’s Attention Map (the Evidence Trace).

Dispatch: Sends a report to the Sage Auditor running on a high-priority background thread.

Remediation:

Ablative Zeroing (Incineration): Surgically zeroes weights and masks gradients permanently.

Null-Space Rotation (Tombstoning): Reversible displacement of disputed facts into a non-addressable orthogonal subspace (R_tomb) for quarantine.

Recovery: Tombstoned nodes can be restored via inverse projection (R_tombᵀ), resuming plasticity without global retraining.

## PROJECT STRUCTURE
```
sage_project/
├── src/
│   ├── container/
│   │   └── sage_container.py      # System Kernel, orchestrates stages & hyperparameters
│   ├── graph/
│   │   └── shared_concept_graph.py # EWMA-Anchored Persistent Memory
│   ├── stages/                    # 7-Stage Developmental Ladder
│   │   ├── base_transformer.py
│   │   ├── infant_stage.py
│   │   ├── toddler_stage.py
│   │   ├── preschool_stage.py
│   │   ├── gradeschool_stage.py
│   │   ├── teen_transformer.py
│   │   ├── adult_transformer.py
│   │   └── sage_transformer.py    # Meta-Governance Head
│   └── monitoring/
│       └── sage_auditor.py        # Asynchronous Remediation Engine
├── tests/                         # Null-Space Recovery & Breach Stress Tests
└── main.py                        # Entry point & Simulation
```

## INSTALLATION & USAGE
Prerequisites

Python 3.9+

PyTorch 2.0+

## Setup
```
git clone https://github.com/studiohead/sage.git
cd sage-cognitive-architecture
pip install -r requirements.txt
```

## Training

You can train a single stage:

`python -m src.main train --stage Teen --epochs 5 --load Gradeschool`


Or train all stages sequentially, automatically saving each stage’s checkpoint:

`python -m src.main train --stage all --epochs 5 --load Infant`


This ensures stages like Teen, which rely on prior layers, train with proper unfreezing and produce stable confidence scores.

Checkpoints are saved in checkpoints/SAGE_STATE_<StageName>.pth.

Testing Manifold Variance

`python -m src.main train --stage Teen --mode test_manifold`


Computes topological dispersion of centroids (manifold_variance) as a sanity check.

Simulated Breach Test

`python main.py --simulate-breach`


Injects a simulated “deceptive fact” into a centroid.

Triggers Auditor and performs Null-Space Rotation / Remediation.

## WHY SAGE MATTERS

**Surgical Deletion**: Comply with “Right to be Forgotten” by permanently incinerating conceptual manifolds.

**Traceability**: Audit the exact attention map (Reasoning Trace) used to justify any output.

**Resilience**: Immune system cleans memory post-facto, maintaining model stability even under noisy or adversarial inputs.

SAGE allows real-time remediation of misinformation without global retraining, using Hierarchical Governance and Centroid-Anchored Memory.
