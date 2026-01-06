import os
import torch
import hashlib
from torch.utils.data import Dataset, DataLoader
from config.config import SHARED_MODEL_CONFIG, STAGE_HYPERPARAMS
from src.inference.graph_tokenizer import GraphTokenizer

EMBED_DIM = SHARED_MODEL_CONFIG.get('embed_dim', 256)


class TextDataset(Dataset):
    def __init__(self, folder_path, graph):
        """
        Dataset that maps text lines to deterministic noise vectors
        and high-speed manifold node indices.
        """
        self.graph = graph
        self.max_dim = EMBED_DIM

        # Initialize the tokenizer to leverage the pre-normalized matrix and O(1) maps
        self.tokenizer = GraphTokenizer(graph)

        # Handle file vs directory inputs
        if os.path.isfile(folder_path):
            self.files = [folder_path]
        elif os.path.isdir(folder_path):
            self.files = [os.path.join(folder_path, f)
                          for f in os.listdir(folder_path)
                          if os.path.isfile(os.path.join(folder_path, f))]
        else:
            self.files = [folder_path]

        self.texts = []
        for path in self.files:
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.texts.append(line)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Generates data on the fly with O(1) Manifold Grounding.
        """
        text = self.texts[idx]
        word = text.split()[0].lower() if text.split() else "null"

        # 1. Deterministic Input Vector Generation (Noise 'Fingerprint' for the word)
        seed = int(hashlib.md5(word.encode()).hexdigest(), 16) % (10 ** 8)
        torch.manual_seed(seed)
        input_tensor = torch.randn(self.max_dim)

        # 2. O(1) Speed Suture: Direct Index Lookup
        # Ensures the tokenizer's label_to_index map is populated
        self.tokenizer._ensure_cache()

        # Resolve the index via the tokenizer's pre-built hash map.
        # This replaces the slow .index() list scan.
        # Fallback to 0 if the word isn't seeded (prevents crash).
        node_idx = getattr(self.tokenizer, 'label_to_index', {}).get(word, 0)

        return {
            'input_ids': input_tensor,
            'node_idx': node_idx,
            'word': word
        }


def collate_text_batch(batch):
    """
    Standardizes the batch structure for the training loop.
    No GPU math is performed here to keep the data pipeline moving.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    node_indices = [item['node_idx'] for item in batch]
    raw_texts = [item['word'] for item in batch]

    # Standard labels (dummy tensor) to satisfy loss function signature
    labels = torch.zeros(len(batch), dtype=torch.long)

    return {
        'input_ids': input_ids,
        'node_indices': node_indices,
        'labels': labels,
        'raw_texts': raw_texts
    }


def get_sage_text_loader(folder_path, graph, stage_name, train=True):
    """
    Returns the configured DataLoader for the current developmental stage.
    """
    stage_key = stage_name.capitalize()
    batch_size = STAGE_HYPERPARAMS.get(stage_key, {}).get('batch_size', 8)

    dataset = TextDataset(folder_path, graph)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate_text_batch,
        num_workers=0,  # Kept at 0 for MPS stability during the Infant stage
        pin_memory=False  # MPS does not benefit from pinned memory in this context
    )