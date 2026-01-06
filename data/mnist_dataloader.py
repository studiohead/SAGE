# data/mnist_dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class SageMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, graph, tokenizer=None):
        self.dataset = mnist_dataset
        self.graph = graph
        # SUTURE: Pass the tokenizer to use its O(1) label_to_index map
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        node_key = str(label)

        # OPTIMIZED INDEX RESOLUTION
        if self.tokenizer:
            # The tokenizer already handles the logic of finding the
            # correct index in the embedding matrix.
            node_idx = self.tokenizer.label_to_id.get(node_key, -1)
        else:
            # Fallback to the slower method if no tokenizer is provided
            node_idx = self.graph.get_node_index(node_key)

        return {
            'input_ids': img.view(-1),
            'labels': label,
            'node_indices': node_idx
        }

def get_sage_mnist_loader(stage_name, graph, tokenizer, train=True, device=None):
    from config.config import STAGE_HYPERPARAMS
    # Normalizing stage name for config lookup
    stage_key = stage_name.capitalize()
    hparams = STAGE_HYPERPARAMS[stage_key]
    batch_size = hparams.get("batch_size", 16)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    raw_mnist = datasets.MNIST('./data', train=train, download=True, transform=transform)
    # Passing the tokenizer here allows the O(1) SUTURE to work
    sage_dataset = SageMNISTDataset(raw_mnist, graph, tokenizer=tokenizer)

    return DataLoader(
        sage_dataset,
        batch_size=batch_size,
        shuffle=train,
        pin_memory=False # Keep False for MPS stability
    )