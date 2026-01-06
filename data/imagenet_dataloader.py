import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os

from torchvision.models import ResNet18_Weights


class SageImageNetDataset(Dataset):
    def __init__(self, imagenet_root, graph, split='train'):
        self.graph = graph
        # 1. Load the official Category mapping (Text Labels)
        weights = ResNet18_Weights.DEFAULT
        self.categories = weights.meta["categories"]

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        split_path = os.path.join(imagenet_root, split)
        self.dataset = datasets.ImageFolder(split_path, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]

        # Node ID is the string form of the number (the 'address')
        node_id_str = str(label)

        # Label/Semantic Meaning is the actual text string
        # We wrap it in a list to match your 'raw_texts' requirement
        semantic_label = [self.categories[label]]

        # Resolve position in the manifold matrix
        try:
            node_idx = self.graph.get_node_index(node_id_str)
        except (KeyError, ValueError):
            node_idx = 0  # Safety fallback during early training

        # EXACT RETURN STRUCTURE MATCHING YOUR FORMAT
        return {
            'input_ids': img,
            'labels': label,
            'node_indices': node_idx,
            'raw_texts': semantic_label
        }


def get_sage_imagenet_loader(stage_name, graph, train=True, device="cpu"):
    from config.config import STAGE_HYPERPARAMS, SHARED_MODEL_CONFIG

    hparams = STAGE_HYPERPARAMS.get(stage_name.capitalize(), {"batch_size": 4})
    imagenet_path = SHARED_MODEL_CONFIG.get('imagenet_path', './data/imagenet')

    dataset = SageImageNetDataset(
        imagenet_path,
        graph,
        split='train' if train else 'val'
    )

    return DataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        shuffle=train,
        pin_memory=False  # Maintained for MPS stability
    )