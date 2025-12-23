import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import argparse


def visualize_manifold(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Checkpoint {checkpoint_path} not found.")
        return

    print(f"[*] Loading manifold from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Target the sage_state where your keys are
    data_source = checkpoint.get('sage_state', {})

    node_labels = []
    embeddings = []

    print("[*] Extracting graph projection states...")
    for key, value in data_source.items():
        # Your keys are 'stages.Infant.graph_projection.weight', etc.
        if 'graph_projection.weight' in key:
            stage_name = key.split('.')[1]  # Extracts 'Infant', 'Toddler', etc.

            # Since this is a weight matrix (Out_Features x In_Features),
            # we treat each output row as a "concept" direction in that stage.
            weights = value.detach().cpu().numpy()

            # We add each row of the projection matrix as a node to plot
            for i in range(weights.shape[0]):
                node_labels.append(f"{stage_name} Dim {i}")
                embeddings.append(weights[i])

    if not embeddings:
        print("[!] Still no nodes found. Let's look at what's actually there:")
        print(list(data_source.keys())[:5])
        return

    # 3. PCA
    embs_np = torch.stack([torch.tensor(e) for e in embeddings]).numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embs_np)

    # 4. Plotting
    plt.figure(figsize=(12, 9))

    # Color code by stage
    colors = {'Infant': 'blue', 'Toddler': 'green', 'Preschool': 'orange'}

    for i, label in enumerate(node_labels):
        stage = label.split()[0]
        color = colors.get(stage, 'gray')
        plt.scatter(coords[i, 0], coords[i, 1], s=100, c=color, alpha=0.6, edgecolors='black')
        plt.annotate(label, (coords[i, 0], coords[i, 1]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=0)

    plt.title(f"SAGE Conceptual Manifold: {checkpoint.get('stage_name', 'Stage Logic')}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    visualize_manifold(args.checkpoint)