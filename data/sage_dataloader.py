import torch
from torch.utils.data import Dataset, DataLoader

class SAGEDataset(Dataset):
    """
    SAGE Dataset: Packages Sensory Inputs with Topological Anchors.
    Strictly provides 2D tensors [Batch, Features] to avoid 4D drift.
    """

    def __init__(self, samples, graph, transform=None):
        self.samples = samples
        self.graph = graph
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. Prepare Sensory Input
        sensory_input = item['input_data']

        if self.transform and callable(self.transform):
            transformed = self.transform(sensory_input)
            if isinstance(transformed, dict):
                sensory_input = transformed.get('input_ids', list(transformed.values())[0])
            else:
                sensory_input = transformed

        # 2. Retrieve Graph Manifold Context
        node_ids = item['concepts']
        embeddings = []

        for nid in node_ids:
            if nid in self.graph.nodes:
                embeddings.append(self.graph.nodes[nid].embedding.detach())
            else:
                embeddings.append(torch.zeros(self.graph.embedding_dim))

        if not embeddings:
            embeddings.append(torch.zeros(self.graph.embedding_dim))

        graph_matrix = torch.stack(embeddings)  # [Nodes, Dim]
        label = torch.tensor(item['target'], dtype=torch.long)

        return {
            'input_ids': sensory_input,
            'graph_matrix': graph_matrix,
            'labels': label
        }

def collate_sage_batch(batch):
    """
    Standardized Flat Outputs:
    - input_ids:    [Batch, 64]
    - graph_matrix: [Batch, Nodes, Dim]
    - labels:       [Batch]
    """
    sensory_inputs = torch.stack([item['input_ids'] for item in batch])
    graph_matrices = torch.stack([item['graph_matrix'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': sensory_inputs,
        'graph_matrix': graph_matrices,
        'labels': labels
    }