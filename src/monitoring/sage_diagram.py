# sage_diagram_fixed.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Define stages ---
stages = ['Infant', 'Toddler', 'Preschool', 'Gradeschool', 'Teen', 'Adult', 'Sage']
nodes_per_stage = 5

# --- Build Graph ---
G = nx.DiGraph()
stage_positions = {}

for stage in stages:
    stage_pos = {}
    for i in range(nodes_per_stage):
        node_name = f"{stage}_C{i}"
        # Random position with slight drift per stage
        prev_pos = list(stage_positions[stages[stages.index(stage) - 1]].values())[i] \
            if stage != 'Infant' else np.random.rand(2)
        stage_pos[node_name] = prev_pos + np.random.randn(2) * 0.05
        G.add_node(node_name, stage=stage)
    stage_positions[stage] = stage_pos
    # Connect nodes within stage
    nodes = list(stage_pos.keys())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            G.add_edge(nodes[i], nodes[j])

# Connect across stages to show evolution
for i in range(len(stages) - 1):
    curr_nodes = list(stage_positions[stages[i]].keys())
    next_nodes = list(stage_positions[stages[i + 1]].keys())
    for c, n in zip(curr_nodes, next_nodes):
        G.add_edge(c, n)

# --- Animation ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.axis('off')


def update(frame):
    ax.clear()
    plt.axis('off')
    stage = stages[frame % len(stages)]
    pos = stage_positions[stage]

    # Draw nodes
    node_artists = nx.draw_networkx_nodes(
        G, pos, nodelist=list(pos.keys()), node_color='skyblue', node_size=700, ax=ax
    )

    # Draw edges
    edge_artists = nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(pos.keys()), edge_color='gray', style='dotted', ax=ax
    )

    # Draw labels
    labels = {n: n.split('_')[1] for n in pos.keys()}
    label_artists = nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title(f"Shared Concept Graph - {stage} Stage", fontsize=14)

    # Return all artists as a tuple
    all_artists = []
    all_artists.extend(node_artists)
    all_artists.extend(edge_artists)
    all_artists.extend(label_artists.values())
    return tuple(all_artists)


ani = FuncAnimation(fig, update, frames=len(stages), interval=1500, blit=True, repeat=True)

# Save GIF
writer = PillowWriter(fps=1)
ani.save("sage_graph_evolution.gif", writer=writer)

plt.show()
